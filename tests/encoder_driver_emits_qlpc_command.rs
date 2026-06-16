//! End-to-end proof that the public whole-stream encode driver
//! ([`oxideav_shorten::encode_stream`]) actually selects and emits a
//! `BLOCK_FN_QLPC` command into the produced `.shn` byte stream — and
//! that [`oxideav_shorten::decode_stream`] reconstructs the input
//! sample-exact.
//!
//! The existing QLPC coverage exercises the *selector* surface
//! (`select_predictor_auto` returning `Choice::Qlpc`) and the
//! *manual* block-writer path (`write_selected_block`). This test
//! closes the remaining gap: it drives only the high-level
//! [`encode_stream`] entry point that a downstream caller actually
//! invokes, then **walks the resulting bit stream command-by-command**
//! and counts the `BLOCK_FN_QLPC` (`FunctionCode::Qlpc`, wire value 7)
//! commands the driver emitted. This converts the spec's
//! cross-codec-verified `BLOCK_FN_QLPC = 7` status into a crate-level
//! proof that the public driver genuinely produces conformant QLPC
//! commands.
//!
//! Clean-room provenance (all truth from `docs/audio/shorten/`):
//!
//! * `spec/04-function-code-resolution.md` §5 — `BLOCK_FN_QLPC = 7`,
//!   validation-corrected (audit/01 §5) from a by-elimination caveat
//!   to a direct cross-codec confirmation: synthetic PCM encoded
//!   through the reference codec at LPC orders 1–4 reconstructs
//!   byte-exact under an independent decoder, pinning both the
//!   function-code numeric value 7 and the parameter-stream layout
//!   (`uvar(LPCQSIZE) order + order × svar(LPCQUANT) coefs +
//!   uvar(ENERGYSIZE) energy + residuals`).
//! * `spec/03-block-and-predictor.md` §3.5 — the QLPC wire layout, the
//!   unscaled signed-integer coefficient domain, the `H_maxlpcorder`
//!   per-block order bound; §2 — the round-robin channel cursor; §3.6
//!   — the per-block `BLOCK_FN_BLOCKSIZE` override that precedes a
//!   trailing partial block.
//! * `spec/02-variable-length-coding.md` §4.1 — `FNSIZE = 2`; §4.2 —
//!   the energy-field-plus-one residual-width convention.
//! * `spec/05-state-and-quirks.md` §1 — sample-history carry seeding;
//!   §2 — the running-mean estimator (these streams pin
//!   `H_meanblocks = 0`, so `mu_chan = 0` throughout, which keeps the
//!   command-walker's DIFF0 reconstruction trivial).
//!
//! The walker below mirrors the decoder's per-command dispatch only so
//! that the bit cursor advances in lock-step with the encoder; it does
//! not re-implement any wire-format decision. It uses the crate's own
//! public payload readers (`decode_diff_block`, `decode_qlpc_block`,
//! `fill_zero_block`, `read_blocksize_payload`, `read_verbatim_payload`)
//! so the consumed bit widths are exactly the decoder's.

use oxideav_shorten::{
    decode_diff_block, decode_qlpc_block, decode_stream, encode_stream, fill_zero_block,
    parse_stream_header, read_blocksize_payload, read_function_code, read_verbatim_payload,
    BitReader, ChannelCarry, FunctionCode, MeanEstimator, PolyOrder, ShortenStreamHeader,
};

fn v2_header(
    channels: u32,
    blocksize: u32,
    maxlpcorder: u32,
    meanblocks: u32,
) -> ShortenStreamHeader {
    ShortenStreamHeader {
        version: 2,
        filetype: 5,
        channels,
        blocksize,
        maxlpcorder,
        meanblocks,
        skipbytes: 0,
    }
}

/// A bounded two-term integer recurrence `s(t) = s(t-1) - s(t-2)`
/// (a period-6 oscillation). It is modelled *exactly* by the QLPC
/// coefficient vector `[1, -1]`, so once the per-channel carry warms
/// up the QLPC residual stream collapses to all-zero — making
/// `BLOCK_FN_QLPC` the genuinely cheapest predictor for the warm
/// blocks and forcing the driver's auto-selector to emit it.
fn oscillation(a: i32, b: i32, len: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(len);
    let (mut s2, mut s1) = (a, b);
    for _ in 0..len {
        let next = s1 - s2;
        out.push(next);
        s2 = s1;
        s1 = next;
    }
    out
}

/// Walk an encoded `.shn` byte stream produced by [`encode_stream`]
/// command-by-command, returning the count of `BLOCK_FN_QLPC`
/// commands emitted. Panics on any structural inconsistency (which
/// would itself be a regression). The walker maintains the same
/// per-channel carry / mean / cursor / block-size state the decoder
/// does so that each command's payload is consumed at exactly the
/// right bit width.
fn count_qlpc_commands(bytes: &[u8]) -> usize {
    let parsed = parse_stream_header(bytes).expect("header parses");
    let header = parsed.header;
    let n = header.channels as usize;
    assert!(n >= 1);

    // Position a reader at the first per-block command — identical to
    // StreamDecoder::new's bit-position calculation (spec/01): skip the
    // 5-byte ajkg+version prefix, then the variable-length parameter
    // block.
    let post_version = &bytes[5..];
    let mut reader = BitReader::new(post_version);
    reader
        .skip_bits(parsed.bits_consumed_after_v)
        .expect("skip parameter block");

    let carry_len = header.sample_history_carry_len() as usize;
    let mut carries: Vec<ChannelCarry> = (0..n).map(|_| ChannelCarry::new(carry_len)).collect();
    let mut means: Vec<MeanEstimator> = (0..n)
        .map(|_| MeanEstimator::new(header.meanblocks))
        .collect();

    let mut cursor = 0usize;
    let mut block_size = header.blocksize;
    let mut qlpc = 0usize;

    loop {
        let code = read_function_code(&mut reader).expect("function code");
        match code {
            FunctionCode::Quit => break,
            FunctionCode::Verbatim => {
                // Consume the opaque payload; does not advance the
                // channel cursor (spec/03 §3.10).
                let _ = read_verbatim_payload(&mut reader).expect("verbatim payload");
            }
            FunctionCode::Blocksize => {
                block_size = read_blocksize_payload(&mut reader).expect("blocksize payload");
            }
            FunctionCode::Bitshift => {
                panic!("driver is lossless (bshift = 0) and must not emit BITSHIFT");
            }
            FunctionCode::Diff0
            | FunctionCode::Diff1
            | FunctionCode::Diff2
            | FunctionCode::Diff3 => {
                let order = PolyOrder::from_function_code(code).unwrap();
                let mu = means[cursor].mu_chan();
                let block = decode_diff_block(&mut reader, order, block_size, &carries[cursor], mu)
                    .expect("diff payload");
                carries[cursor].update_after_block(&block);
                means[cursor].record_block(&block);
                cursor = (cursor + 1) % n;
            }
            FunctionCode::Qlpc => {
                qlpc += 1;
                let block = decode_qlpc_block(&mut reader, block_size, &carries[cursor])
                    .expect("qlpc payload");
                carries[cursor].update_after_block(&block);
                means[cursor].record_block(&block);
                cursor = (cursor + 1) % n;
            }
            FunctionCode::Zero => {
                let mu = means[cursor].mu_chan();
                let block = fill_zero_block(block_size, mu).expect("zero block");
                carries[cursor].update_after_block(&block);
                means[cursor].record_block(&block);
                cursor = (cursor + 1) % n;
            }
            other => panic!("driver emitted unexpected function code {other:?}"),
        }
    }
    qlpc
}

#[test]
fn public_encode_stream_emits_qlpc_for_recurrence_signal() {
    // Four 64-sample blocks of the exact [1,-1] recurrence on a single
    // channel, mean estimator disabled (mu_chan = 0). maxlpcorder = 2
    // makes the auto-derived order-2 QLPC candidate available.
    let blocksize = 64u32;
    let header = v2_header(1, blocksize, 2, 0);
    let signal = oscillation(100, 50, 4 * blocksize as usize);

    let bytes = encode_stream(&header, &signal, &[]).expect("encode succeeds");

    // The high-level driver must have emitted at least one QLPC command.
    let qlpc = count_qlpc_commands(&bytes);
    assert!(
        qlpc >= 1,
        "public encode_stream driver should emit >= 1 BLOCK_FN_QLPC command \
         for an exact two-term recurrence; got {qlpc}"
    );

    // And the whole stream must round-trip sample-exact.
    let dec = decode_stream(&bytes).expect("decode succeeds");
    assert_eq!(dec.channels.len(), 1);
    assert_eq!(dec.channels[0], signal);
    assert!(dec.verbatim.is_empty());
}

#[test]
fn public_encode_stream_emits_qlpc_stereo_with_tail_block() {
    // Two channels, blocksize not dividing the per-channel length so a
    // trailing partial block (preceded by a BLOCK_FN_BLOCKSIZE override,
    // spec/03 §3.6) exercises the walker's blocksize-state path. Each
    // channel carries its own phase of the recurrence.
    let blocksize = 32u32;
    let per_channel = 4 * blocksize as usize + 19; // 147 -> one tail block
    let ch0 = oscillation(7, 13, per_channel);
    let ch1 = oscillation(-21, 4, per_channel);

    let mut interleaved = Vec::with_capacity(per_channel * 2);
    for t in 0..per_channel {
        interleaved.push(ch0[t]);
        interleaved.push(ch1[t]);
    }

    let header = v2_header(2, blocksize, 2, 0);
    let bytes = encode_stream(&header, &interleaved, &[]).expect("encode succeeds");

    let qlpc = count_qlpc_commands(&bytes);
    assert!(
        qlpc >= 1,
        "stereo recurrence stream should emit >= 1 BLOCK_FN_QLPC command; got {qlpc}"
    );

    let dec = decode_stream(&bytes).expect("decode succeeds");
    assert_eq!(dec.channels.len(), 2);
    assert_eq!(dec.channels[0], ch0);
    assert_eq!(dec.channels[1], ch1);
}

#[test]
fn maxlpcorder_zero_driver_emits_no_qlpc_command() {
    // Control: with H_maxlpcorder = 0 the spec (spec/03 §3.5) forbids
    // any BLOCK_FN_QLPC command; the same recurrence signal must encode
    // entirely via the polynomial-difference predictors and still
    // round-trip sample-exact.
    let blocksize = 64u32;
    let header = v2_header(1, blocksize, 0, 0);
    let signal = oscillation(100, 50, 4 * blocksize as usize);

    let bytes = encode_stream(&header, &signal, &[]).expect("encode succeeds");

    let qlpc = count_qlpc_commands(&bytes);
    assert_eq!(
        qlpc, 0,
        "H_maxlpcorder = 0 stream must contain zero BLOCK_FN_QLPC commands"
    );

    let dec = decode_stream(&bytes).expect("decode succeeds");
    assert_eq!(dec.channels[0], signal);
}
