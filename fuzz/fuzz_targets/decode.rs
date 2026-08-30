#![no_main]

//! Decode arbitrary fuzz-supplied bytes through both public decode
//! paths and pin their agreement.
//!
//! The whole-stream driver [`oxideav_shorten::decode_stream`] and the
//! constant-memory iterator [`oxideav_shorten::StreamDecoder`] walk the
//! same per-block command loop (`spec/03` §2 + `spec/05` §1 + §2) with
//! independent state machines. Contract under test:
//!
//! 1. Neither path panics, overflows, indexes out of bounds, hangs, or
//!    allocates beyond the header-derived caps on arbitrary bytes.
//! 2. **Differential**: when the streaming path finishes cleanly the
//!    whole-stream path must too, with identical header, verbatim
//!    prefix, per-channel samples, `stream_proper_len` and
//!    `quit_padding`; when the streaming path fails the whole-stream
//!    path must fail with the *same* `Error` value.
//!
//! ## Output bound (target-side, not a decoder cap)
//!
//! A `BLOCK_FN_ZERO` command costs 5 bits and emits a full sub-block of
//! samples, so a 1 KiB input with `H_blocksize` near `BLOCKSIZE_MAX`
//! legitimately expands to gigabytes of silence. That is a valid
//! stream, not a defect, so the decoder is never truncated; instead
//! this target drains the streaming path first, counting samples, and
//! only invokes the accumulating `decode_stream` when the total stays
//! under [`SAMPLE_CAP`] (1 Mi samples = 4 MiB per accumulating copy, keeping sanitizer allocator churn low). Over-cap inputs are
//! still fully exercised through the bounded-memory iterator up to the
//! cap and then abandoned.

use libfuzzer_sys::fuzz_target;
use oxideav_shorten::{decode_stream, DecodedBlock, StreamDecoder};

/// Total decoded-sample cap before the accumulating path is skipped.
const SAMPLE_CAP: usize = 1 << 20;

/// Per-block sample cap before the input is abandoned. The library
/// accepts blocks up to `BLOCKSIZE_MAX` (1 Mi samples — a 4 MiB `Vec`
/// per block), which is semantically valid (huge cheap silence via
/// `BLOCK_FN_ZERO`) but churns the sanitizer allocator hard enough to
/// trip libFuzzer's rss limit over a long run. Inputs that raise the
/// running sub-block size beyond this cap have already had the
/// oversized block decoded once; the target then stops.
const BLOCK_CAP: u32 = 1 << 14;

fuzz_target!(|data: &[u8]| {
    let mut dec = match StreamDecoder::new(data) {
        Ok(d) => d,
        Err(e) => {
            let whole = decode_stream(data);
            assert_eq!(whole.err(), Some(e), "header-stage failure class must agree");
            return;
        }
    };
    let header = *dec.header();
    if header.blocksize > BLOCK_CAP {
        // Still exercised: construction validated the header and the
        // first next_block call below would only replay the
        // BLOCKSIZE_MAX path the robustness suite already pins.
        return;
    }
    let n = header.channels as usize;
    let mut planes: Vec<Vec<i32>> = vec![Vec::new(); n];
    let mut verbatim = Vec::new();
    let mut total = 0usize;
    let outcome = loop {
        if dec.current_block_size() > BLOCK_CAP {
            // A BLOCKSIZE override raised the sub-block size past the
            // churn cap; the oversized block itself was already decoded
            // by the call that absorbed the override.
            return;
        }
        match dec.next_block() {
            Ok(Some(DecodedBlock::Samples { channel, samples })) => {
                assert!(channel < n, "channel cursor out of range");
                assert_eq!(channel, dec.current_channel().wrapping_add(n - 1) % n);
                total += samples.len();
                if total > SAMPLE_CAP {
                    return;
                }
                planes[channel].extend_from_slice(&samples);
            }
            Ok(Some(DecodedBlock::Verbatim { bytes })) => verbatim.extend_from_slice(&bytes),
            Ok(None) => break Ok(()),
            Err(e) => break Err(e),
        }
    };

    let whole = decode_stream(data);
    match outcome {
        Err(e) => {
            assert_eq!(whole.err(), Some(e), "failure class must agree across both paths");
        }
        Ok(()) => {
            assert!(dec.is_finished());
            let whole = whole.expect("streaming path finished cleanly; whole-stream path must too");
            assert_eq!(whole.header, header);
            assert_eq!(whole.verbatim, verbatim);
            assert_eq!(whole.channels, planes);
            assert_eq!(Some(whole.stream_proper_len), dec.stream_proper_len());
            assert_eq!(Some(whole.quit_padding), dec.quit_padding());
            assert_eq!(
                dec.trailer_len(),
                Some(data.len() - whole.stream_proper_len),
                "trailer length is the bytes past the QUIT boundary"
            );
        }
    }
});
