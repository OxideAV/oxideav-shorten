//! Fixture-anchored byte-exact decode test — fixture `F1`'s first
//! `BLOCK_FN_DIFF1` block.
//!
//! The behavioural anchor is the **ffmpeg-byte-exact** vector pinned in
//! `docs/audio/shorten/spec/05-state-and-quirks.md` §3.1 and its
//! footnote `T15`. That section resolves the residual-mantissa-width
//! `+1` rule against a real `.shn` fixture (`luckynight.shn`, the
//! corpus's `F1`) by reading the fixture's first per-block command and
//! comparing two candidate interpretations against FFmpeg 7.1.2's
//! decoded PCM:
//!
//!   * `energy_field = 3` decoded as `uvar(3) = 3`. Reading the
//!     subsequent residual stream at the *un-incremented* width
//!     `svar(3)` yields residuals `[2, 4, -17, 0, 9, 12, …]` — which
//!     reconstruct to ch0 samples matching **no** expected pattern.
//!   * Reading the *same bits* at the `+1`-incremented width `svar(4)`
//!     yields residuals `[4, 0, -26, 42, -17, -14, …]`. Under the
//!     `BLOCK_FN_DIFF1` recurrence `s(t) = s(t-1) + e1(t)` with a
//!     zero-initialised carry (`s(-1) = 0`), this reproduces ch0
//!     samples `[4, 4, -22, 20, 3, -11, …]` — **byte-for-byte** with
//!     FFmpeg's `s16le`-decoded ch0 output for `F1` across the first
//!     256 samples.
//!
//! This test does not require the (large, out-of-tree) `F1` bytes: it
//! reconstructs the *exact pinned residual stream* of the `svar(4)`
//! reading on the wire, drives it through the public `decode_stream`
//! decode path, and asserts the spec/ffmpeg-pinned ch0 PCM. Because the
//! reconstructed samples are an external ground truth (FFmpeg's PCM, not
//! our own encoder's output), this is a genuine fixture-anchored
//! byte-exact decode test, not a self-roundtrip.
//!
//! It also pins the **negative** half of T15: the `svar(3)` reading of
//! the same bit positions yields the documented wrong residual stream
//! `[2, 4, -17, 0, 9, 12]`, confirming the energy field really is read
//! as `n - 1` and that decoding at the un-incremented width desyncs the
//! sample reconstruction.

use oxideav_shorten::{decode_stream, BitReader, FunctionCode, MAGIC};

// ---- minimal MSB-first bit-stream builders (mirror the synthetic
//      builders used by the other integration tests) ----

fn pack_bits_msb_first(bits: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut byte = 0u8;
    let mut n = 0u32;
    for &b in bits {
        byte = (byte << 1) | (b as u8 & 1);
        n += 1;
        if n == 8 {
            out.push(byte);
            byte = 0;
            n = 0;
        }
    }
    if n > 0 {
        out.push(byte << (8 - n));
    }
    out
}

fn encode_uvar(value: u32, n: u32) -> Vec<u32> {
    if n == 0 {
        let mut bits = vec![0u32; value as usize];
        bits.push(1);
        bits
    } else {
        let span = 1u32 << n;
        let prefix_zeros = value / span;
        let mantissa = value % span;
        let mut bits = vec![0u32; prefix_zeros as usize];
        bits.push(1);
        for i in (0..n).rev() {
            bits.push((mantissa >> i) & 1);
        }
        bits
    }
}

fn encode_svar(value: i64, n: u32) -> Vec<u32> {
    // spec/02 §2.2 folding: non-negative `v` → `2v`; negative → `2|v|-1`
    // (== `(!v << 1) | 1`).
    let u: u64 = if value >= 0 {
        (value as u64) << 1
    } else {
        (((!value) as u64) << 1) | 1
    };
    let u32_val = u32::try_from(u).expect("residual fits in u32 for this fixture vector");
    encode_uvar(u32_val, n)
}

fn encode_ulong(value: u32, w: u32) -> Vec<u32> {
    // ulong() per spec/02 §3: width `w` as uvar(ULONGSIZE=2), then the
    // value as uvar(w).
    let mut bits = encode_uvar(w, 2);
    bits.extend(encode_uvar(value, w));
    bits
}

/// Six-field v2 parameter block matching fixture `F1`'s header choices
/// (`spec/01` §3 + `spec/05` §6: filetype `s16lh = 5`, default
/// blocksize 256, no LPC, four mean-estimator blocks, no skipbytes),
/// using the smallest `ulong()` width that admits each value.
///
/// `H_channels` is pinned to **1** rather than `F1`'s real value of 2:
/// the spec/05 §3.1 vector is `F1`'s *ch0* reconstruction, and the
/// DIFF1 + energy-field-plus-one + zero-carry semantics under test are
/// per-channel — decoding ch0 in isolation reproduces the same pinned
/// PCM while keeping the round-robin cursor on ch0 for a single block.
fn f1_header_param_bits(blocksize: u32) -> Vec<u32> {
    let bits_for = |v: u32| -> u32 {
        if v == 0 {
            0
        } else {
            32 - v.leading_zeros()
        }
    };
    let mut bits = Vec::new();
    bits.extend(encode_ulong(5, bits_for(5))); // H_filetype = s16lh
    bits.extend(encode_ulong(1, bits_for(1))); // H_channels = 1 (F1 ch0 in isolation)
    bits.extend(encode_ulong(blocksize, bits_for(blocksize))); // H_blocksize
    bits.extend(encode_ulong(0, bits_for(0))); // H_maxlpcorder = 0
    bits.extend(encode_ulong(4, bits_for(4))); // H_meanblocks = 4
    bits.extend(encode_ulong(0, bits_for(0))); // H_skipbytes = 0
    bits
}

fn assemble(all_bits: &[u32]) -> Vec<u8> {
    let body = pack_bits_msb_first(all_bits);
    let mut buf = Vec::with_capacity(5 + body.len());
    buf.extend_from_slice(&MAGIC);
    buf.push(2); // version 2
    buf.extend_from_slice(&body);
    buf
}

// FNSIZE = 2, ENERGYSIZE = 3 (spec/02 §4.1 / spec/05 §3).
const FN_DIFF1: u32 = 1;
const FN_BLOCKSIZE: u32 = 5;
const FN_QUIT: u32 = 4;

/// The spec/05 §3.1 / `T15` ffmpeg-byte-exact residual stream of `F1`'s
/// first `BLOCK_FN_DIFF1` block, read at the `+1`-incremented width
/// `svar(4)`.
const F1_DIFF1_RESIDUALS_SVAR4: [i64; 6] = [4, 0, -26, 42, -17, -14];

/// The corresponding ffmpeg ch0 PCM samples, reconstructed under
/// `s(t) = s(t-1) + e1(t)` with `s(-1) = 0` (spec/05 §3.1).
const F1_DIFF1_CH0_PCM: [i32; 6] = [4, 4, -22, 20, 3, -11];

/// The wrong residual stream produced by reading the *same bits* at the
/// un-incremented width `svar(3)` (spec/05 §3.1 first bullet / `T15`).
const F1_DIFF1_RESIDUALS_SVAR3_WRONG: [i64; 6] = [2, 4, -17, 0, 9, 12];

/// Build the single-channel `F1`-first-block stream: header, one
/// `BLOCK_FN_BLOCKSIZE` override down to the six pinned samples, one
/// `BLOCK_FN_DIFF1` block carrying the six `svar(4)` residuals, then
/// `BLOCK_FN_QUIT`. A blocksize override (a spec-conformant command,
/// `spec/04` §4) keeps the test compact while exercising the real
/// blocksize-change path; the energy field carries the encoded value
/// `3` so the decoder's `+1` rule reads the residuals at width `4`.
fn build_f1_diff1_stream() -> Vec<u8> {
    let block_len = F1_DIFF1_RESIDUALS_SVAR4.len() as u32;

    // H_channels = 1 so the round-robin cursor stays on ch0; the pinned
    // vector is F1's ch0 reconstruction (see f1_header_param_bits).
    let mut bits = f1_header_param_bits(block_len);

    // BLOCKSIZE override to `block_len` so the DIFF1 block carries
    // exactly the six pinned residuals.
    bits.extend(encode_uvar(FN_BLOCKSIZE, 2));
    let w = if block_len == 0 {
        0
    } else {
        32 - block_len.leading_zeros()
    };
    bits.extend(encode_ulong(block_len, w));

    // DIFF1 block: fn, energy-field encoded value 3 (residual width 4),
    // then the six svar(4) residuals.
    bits.extend(encode_uvar(FN_DIFF1, 2));
    bits.extend(encode_uvar(3, 3)); // ENERGYSIZE = 3, encoded value 3 → width 4
    for &r in &F1_DIFF1_RESIDUALS_SVAR4 {
        bits.extend(encode_svar(r, 4));
    }

    bits.extend(encode_uvar(FN_QUIT, 2));
    assemble(&bits)
}

#[test]
fn f1_first_diff1_block_decodes_to_ffmpeg_pinned_ch0_pcm() {
    let stream = build_f1_diff1_stream();
    let decoded = decode_stream(&stream).expect("F1 first-block synthetic stream decodes");

    assert_eq!(
        decoded.channels.len(),
        1,
        "single-channel F1-first-block stream yields one channel plane"
    );
    let ch0 = &decoded.channels[0];
    assert_eq!(
        ch0.as_slice(),
        F1_DIFF1_CH0_PCM,
        "spec/05 §3.1 / T15: F1's first DIFF1 block must reconstruct to \
         ffmpeg's byte-exact ch0 PCM under the energy-field-plus-one rule"
    );
}

#[test]
fn f1_diff1_residual_stream_reads_as_svar4_not_svar3() {
    // Re-derive the two residual readings directly from the wire to pin
    // the negative half of T15: the *same* residual bits read at the
    // un-incremented width `svar(3)` give the documented wrong stream,
    // and only `svar(4)` gives the ffmpeg-byte-exact one. This exercises
    // the public BitReader::read_svar primitive against the spec vector.
    let mut bits = Vec::new();
    for &r in &F1_DIFF1_RESIDUALS_SVAR4 {
        bits.extend(encode_svar(r, 4));
    }
    let bytes = pack_bits_msb_first(&bits);

    // svar(4) round-trips the pinned residual stream exactly.
    let mut r4 = BitReader::new(&bytes);
    let got4: Vec<i64> = (0..F1_DIFF1_RESIDUALS_SVAR4.len())
        .map(|_| r4.read_svar(4).expect("svar(4) residual read"))
        .collect();
    assert_eq!(
        got4.as_slice(),
        F1_DIFF1_RESIDUALS_SVAR4,
        "the wire residuals decode at width 4 to the pinned svar(4) stream"
    );

    // svar(3) over the same bytes yields the documented wrong stream.
    let mut r3 = BitReader::new(&bytes);
    let got3: Vec<i64> = (0..F1_DIFF1_RESIDUALS_SVAR3_WRONG.len())
        .map(|_| r3.read_svar(3).expect("svar(3) residual read"))
        .collect();
    assert_eq!(
        got3.as_slice(),
        F1_DIFF1_RESIDUALS_SVAR3_WRONG,
        "spec/05 §3.1 first bullet: reading the same bits at the \
         un-incremented width svar(3) reproduces the documented wrong \
         residual stream [2, 4, -17, 0, 9, 12]"
    );
}

#[test]
fn f1_diff1_quit_terminates_with_byte_aligned_stream_proper() {
    // The synthetic F1-first-block stream terminates at BLOCK_FN_QUIT;
    // the reported SHN-stream-proper length must cover the whole buffer
    // (no trailing sidecar) and be byte-aligned (spec/04 §2.1).
    let stream = build_f1_diff1_stream();
    let decoded = decode_stream(&stream).expect("decodes");
    assert_eq!(
        decoded.stream_proper_len,
        stream.len(),
        "QUIT padding consumes to end-of-buffer; the whole stream is SHN-proper"
    );
}

// Keep the FunctionCode import meaningful: assert the numeric wire
// values the builder hard-codes match the crate's pinned mapping
// (spec/04 §3..§6), so a future renumbering can't silently desync this
// fixture vector.
#[test]
fn pinned_function_code_wire_values_match_crate_mapping() {
    assert_eq!(FunctionCode::Diff1.wire_value(), FN_DIFF1);
    assert_eq!(FunctionCode::Blocksize.wire_value(), FN_BLOCKSIZE);
    assert_eq!(FunctionCode::Quit.wire_value(), FN_QUIT);
}
