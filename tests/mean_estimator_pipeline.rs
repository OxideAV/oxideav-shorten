//! Integration test exercising the per-channel running mean estimator
//! through a sequence of DIFF0 + ZERO blocks dispatched against the
//! same channel.
//!
//! The behavioural anchor is `docs/audio/shorten/spec/05-state-and-
//! quirks.md` §2 + §2.3 + §2.4 + §2.5:
//!
//! * §2.1 — each channel's mean buffer is zero-initialised at stream
//!   start; the very first sample-producing block sees `mu_chan = 0`.
//! * §2.3 — `BLOCK_FN_DIFF0` reconstructs `s(t) = e0(t) + mu_chan`.
//! * §2.4 — `BLOCK_FN_ZERO` emits `bs` samples all equal to `mu_chan`.
//! * §2.5 — the sliding-window update rule:
//!   `mu_blk = trunc_div(sum + bs/2, bs)`,
//!   `mu_chan = trunc_div(sum_of_slots + H_meanblocks/2, H_meanblocks)`,
//!   both with truncation toward zero and the always-positive bias.
//!
//! The test builds a synthetic v2 stream with the same header field
//! choices as fixture `F1` but with `H_channels = 1` for clarity (the
//! cross-channel cursor was already exercised in the round-3 DIFF1
//! pipeline test), then dispatches:
//!
//!   1. A `BLOCK_FN_DIFF0` block of `bs = 4` residuals = [10, 20, 30,
//!      40]. With initial `mu_chan = 0`, samples = [10, 20, 30, 40];
//!      per-block mean = trunc_div(100 + 2, 4) = trunc_div(102, 4) =
//!      25.
//!   2. A `BLOCK_FN_ZERO` block of `bs = 4`. The decoder reads the
//!      block-size override (BLOCKSIZE command) — not wired in round
//!      4 — so we instead inline-override the block size to 4 in the
//!      test driver, exactly as the round-3 DIFF1 integration test
//!      does. With `mu_chan` after block 1, the four ZERO samples
//!      should equal the channel's running mean.
//!   3. A `BLOCK_FN_DIFF0` block of `bs = 4` residuals = [0, 0, 0, 0].
//!      Under `s(t) = e0(t) + mu_chan`, all four samples should equal
//!      `mu_chan` at that block's start.
//!
//! `H_meanblocks` is set to `4` (matching fixture F1's choice). After
//! block 1 the estimator holds `[0, 0, 0, 25]`; mu_chan =
//! trunc_div(25 + 2, 4) = trunc_div(27, 4) = 6. The ZERO block fills
//! four samples = 6. The estimator records mu_blk = 6 (sum = 24,
//! trunc_div(24 + 2, 4) = trunc_div(26, 4) = 6) and now holds
//! `[0, 0, 25, 6]`; mu_chan = trunc_div(31 + 2, 4) = trunc_div(33, 4)
//! = 8. The third block (DIFF0 with zero residuals) thus emits four
//! samples = 8.

use oxideav_shorten::{
    decode_diff_block, fill_zero_block, parse_stream_header, read_function_code, BitReader,
    ChannelCarry, FunctionCode, MeanEstimator, PolyOrder, ENERGYSIZE, FNSIZE, MAGIC,
};

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
    let u: u64 = if value >= 0 {
        (value as u64) << 1
    } else {
        (((!value) as u64) << 1) | 1
    };
    let u32_val = u32::try_from(u).expect("svar fits in u32 in this test");
    encode_uvar(u32_val, n)
}

fn encode_ulong(value: u32, w: u32) -> Vec<u32> {
    let mut bits = Vec::new();
    bits.extend(encode_uvar(w, 2));
    bits.extend(encode_uvar(value, w));
    bits
}

/// Append a synthetic `BLOCK_FN_DIFF0` command (function code + energy
/// + residuals) to `out_bits`.
fn append_diff0_block(out_bits: &mut Vec<u32>, energy_encoded: u32, residuals: &[i64]) {
    out_bits.extend(encode_uvar(0, FNSIZE)); // DIFF0 = 0
    out_bits.extend(encode_uvar(energy_encoded, ENERGYSIZE));
    let width = energy_encoded + 1;
    for &r in residuals {
        out_bits.extend(encode_svar(r, width));
    }
}

/// Append a synthetic `BLOCK_FN_ZERO` command (bare function code; no
/// payload bits per `spec/03` §3.9 / `spec/04` §6).
fn append_zero_block(out_bits: &mut Vec<u32>) {
    out_bits.extend(encode_uvar(8, FNSIZE)); // ZERO = 8
}

#[test]
fn header_then_diff0_zero_diff0_pipeline_tracks_running_mean() {
    // --- Stage 1: build the header bits (single-channel variant of
    // F1's parameter choices). ---
    let mut header_bits = Vec::new();
    header_bits.extend(encode_ulong(5, 3)); // H_filetype = 5
    header_bits.extend(encode_ulong(1, 2)); // H_channels = 1
    header_bits.extend(encode_ulong(256, 9)); // H_blocksize = 256
    header_bits.extend(encode_ulong(0, 0)); // H_maxlpcorder = 0
    header_bits.extend(encode_ulong(4, 3)); // H_meanblocks = 4
    header_bits.extend(encode_ulong(0, 0)); // H_skipbytes = 0

    // --- Stage 2: build the block-stream bits. ---
    let mut block_bits = Vec::new();
    // Block 1: DIFF0 residuals [10, 20, 30, 40] under mu_chan = 0
    //   -> samples [10, 20, 30, 40]; mu_blk = trunc_div(100+2,4) = 25.
    // Window after = [0, 0, 0, 25]; mu_chan = trunc_div(25+2,4) = 6.
    append_diff0_block(&mut block_bits, 5, &[10, 20, 30, 40]);
    // Block 2: ZERO (no payload bits) under mu_chan = 6
    //   -> samples [6, 6, 6, 6]; mu_blk = trunc_div(24+2,4) = 6.
    // Window after = [0, 0, 25, 6]; mu_chan = trunc_div(31+2,4) = 8.
    append_zero_block(&mut block_bits);
    // Block 3: DIFF0 residuals [0, 0, 0, 0] under mu_chan = 8
    //   -> samples [8, 8, 8, 8].
    append_diff0_block(&mut block_bits, 0, &[0, 0, 0, 0]);
    block_bits.extend(encode_uvar(4, FNSIZE)); // QUIT = 4

    // --- Stage 3: assemble + parse the header. ---
    let mut all_bits = Vec::with_capacity(header_bits.len() + block_bits.len());
    all_bits.extend(&header_bits);
    all_bits.extend(&block_bits);
    let body = pack_bits_msb_first(&all_bits);
    let mut buf = Vec::with_capacity(5 + body.len());
    buf.extend_from_slice(&MAGIC);
    buf.push(2);
    buf.extend_from_slice(&body);

    let parsed = parse_stream_header(&buf).expect("header must parse");
    assert_eq!(parsed.header.channels, 1);
    assert_eq!(parsed.header.blocksize, 256);
    assert_eq!(parsed.header.maxlpcorder, 0);
    assert_eq!(parsed.header.meanblocks, 4);

    // --- Stage 4: re-open a BitReader and skip the header bits. ---
    let post_version = &buf[5..];
    let mut reader = BitReader::new(post_version);
    let _ = reader.read_bits(32).expect("burn first 32 header bits");
    let _ = reader
        .read_bits(parsed.bits_consumed_after_v - 32)
        .expect("burn remaining header bits");

    // --- Stage 5: dispatch blocks against the single channel with the
    // per-channel carry + mean estimator. ---
    let mut carry = ChannelCarry::new(parsed.header.sample_history_carry_len() as usize);
    let mut mean = MeanEstimator::new(parsed.header.meanblocks);
    let bs = 4u32; // local override; H_blocksize=256 not wired up yet.

    // Block 1: DIFF0.
    let fc1 = read_function_code(&mut reader).unwrap();
    assert_eq!(fc1, FunctionCode::Diff0);
    let order1 = PolyOrder::from_function_code(fc1).unwrap();
    let mu1 = mean.mu_chan();
    assert_eq!(
        mu1, 0,
        "first block sees zero-init mu_chan per spec/05 §2.1"
    );
    let blk1 = decode_diff_block(&mut reader, order1, bs, &carry, mu1).unwrap();
    assert_eq!(blk1, vec![10, 20, 30, 40]);
    carry.update_after_block(&blk1);
    mean.record_block(&blk1);
    let mu_after_b1 = mean.mu_chan();
    assert_eq!(
        mu_after_b1, 6,
        "after DIFF0 block of mu_blk=25 with window of 4 the running mean is 6"
    );

    // Block 2: ZERO. No payload bits; the helper synthesises the
    // samples.
    let fc2 = read_function_code(&mut reader).unwrap();
    assert_eq!(fc2, FunctionCode::Zero);
    let blk2 = fill_zero_block(bs, mu_after_b1).unwrap();
    assert_eq!(blk2, vec![6, 6, 6, 6]);
    carry.update_after_block(&blk2);
    mean.record_block(&blk2);
    let mu_after_b2 = mean.mu_chan();
    assert_eq!(
        mu_after_b2, 8,
        "after ZERO block of mu_blk=6 the window is [0,0,25,6] -> mu_chan = 8"
    );

    // Block 3: DIFF0 zero residuals.
    let fc3 = read_function_code(&mut reader).unwrap();
    assert_eq!(fc3, FunctionCode::Diff0);
    let order3 = PolyOrder::from_function_code(fc3).unwrap();
    let blk3 = decode_diff_block(&mut reader, order3, bs, &carry, mu_after_b2).unwrap();
    assert_eq!(blk3, vec![8, 8, 8, 8]);

    // --- Stage 6: QUIT terminator. ---
    let fc_quit = read_function_code(&mut reader).unwrap();
    assert_eq!(fc_quit, FunctionCode::Quit);
}
