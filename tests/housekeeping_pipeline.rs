//! Integration test exercising the header parse + per-block command
//! dispatch with the two housekeeping commands `BLOCK_FN_BLOCKSIZE`
//! and `BLOCK_FN_BITSHIFT` interleaved among predictor blocks.
//!
//! The behavioural anchors here are
//! `docs/audio/shorten/spec/03-block-and-predictor.md` §3.6 (the
//! `BLOCK_FN_BLOCKSIZE` `<fn=5> <new_bs:ulong()>` layout + "subsequent
//! predictor commands produce blocks of `new_bs` samples per channel
//! until the stream ends or until another `BLOCK_FN_BLOCKSIZE` command
//! is encountered" semantics), §3.7 (the `BLOCK_FN_BITSHIFT`
//! `<fn=6> <bshift:uvar(BITSHIFTSIZE)>` layout + "subsequent samples
//! produced by predictor commands are left-shifted by `bshift`
//! positions before being emitted" semantics), `spec/02` §4.6
//! (`BITSHIFTSIZE = 2`), `spec/05` §1.4 ("bit-shift application is
//! external to the carry. The carry stores the samples in their
//! pre-bit-shift form"), and `spec/04` §3 + §4 (the `BLOCK_FN_BITSHIFT
//! = 6` parameter values 1/4/8/12 observed in `F5..F8` per `T10`; the
//! `BLOCK_FN_BLOCKSIZE = 5` tail-block override at command 11,377 of
//! `F2` with `new_bs = 155` per `T12`).
//!
//! The test builds a synthetic v2 stream containing:
//!   1. A v2 header with `H_blocksize = 256` (the spec/02 §6 anchor),
//!      single channel, mean estimator disabled.
//!   2. A `BLOCK_FN_BITSHIFT` command setting `bshift = 4`.
//!   3. A `BLOCK_FN_DIFF1` block of the default size (256 samples).
//!      The block's emitted samples are *pre-shift*; left-shifting
//!      them by `bshift = 4` is the driver's responsibility (this
//!      test verifies the carry's *pre-shift* form, matching
//!      `spec/05` §1.4).
//!   4. A `BLOCK_FN_BLOCKSIZE` command overriding the size to 4
//!      samples (a small tail-block, analogous to F2's tail at
//!      command 11,377).
//!   5. A `BLOCK_FN_DIFF1` block of the *new* size (4 samples), with
//!      its predictor reading the prior block's tail samples from the
//!      carry.
//!   6. A `BLOCK_FN_QUIT` sentinel.
//!
//! The test parses the header, dispatches each command, and verifies
//! the running sub-block-size + running bit-shift state updates plus
//! the carry hand-off across the BLOCKSIZE-override boundary.

use oxideav_shorten::{
    decode_diff_block, parse_stream_header, read_bitshift_payload, read_blocksize_payload,
    read_function_code, BitReader, ChannelCarry, FunctionCode, PolyOrder, BITSHIFTSIZE, ENERGYSIZE,
    FNSIZE, MAGIC,
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

/// Append a synthetic `BLOCK_FN_DIFF1` command (function code + energy +
/// residuals) to `out_bits`. Energy is the encoded value `e = n - 1` per
/// `spec/05` §3; the residual mantissa width is `e + 1`.
fn append_diff1_block(out_bits: &mut Vec<u32>, energy_encoded: u32, residuals: &[i64]) {
    out_bits.extend(encode_uvar(1, FNSIZE)); // DIFF1 = 1
    out_bits.extend(encode_uvar(energy_encoded, ENERGYSIZE));
    let width = energy_encoded + 1;
    for &r in residuals {
        out_bits.extend(encode_svar(r, width));
    }
}

#[test]
fn header_then_bitshift_diff_blocksize_diff_quit_pipeline() {
    // --- Stage 1: header bits, H_blocksize = 8 (small so the test
    // stays compact). H_meanblocks = 0 disables the mean estimator so
    // DIFFn reconstruction is independent of the mean.
    let mut header_bits = Vec::new();
    header_bits.extend(encode_ulong(5, 3)); // H_filetype = 5 (s16lh)
    header_bits.extend(encode_ulong(1, 1)); // H_channels = 1
    header_bits.extend(encode_ulong(8, 4)); // H_blocksize = 8
    header_bits.extend(encode_ulong(0, 0)); // H_maxlpcorder = 0
    header_bits.extend(encode_ulong(0, 0)); // H_meanblocks = 0 (disabled)
    header_bits.extend(encode_ulong(0, 0)); // H_skipbytes = 0

    // --- Stage 2: block-stream bits. ---
    // (a) BITSHIFT bshift = 4 — sets the per-stream shift state.
    // (b) DIFF1 block of 8 samples (the default H_blocksize), residuals
    //     [3, 1, -2, 4, 0, -1, 2, 5], zero carry:
    //       s0 = 0 + 3 = 3
    //       s1 = 3 + 1 = 4
    //       s2 = 4 - 2 = 2
    //       s3 = 2 + 4 = 6
    //       s4 = 6 + 0 = 6
    //       s5 = 6 - 1 = 5
    //       s6 = 5 + 2 = 7
    //       s7 = 7 + 5 = 12
    //   block A samples (pre-shift, per spec/05 §1.4) = [3,4,2,6,6,5,7,12].
    // (c) BLOCKSIZE new_bs = 4 — overrides the sub-block size.
    // (d) DIFF1 block of 4 samples, residuals [1, 1, 1, 1] from carry
    //     [12, 7, 5] (s(t-1)=12, s(t-2)=7, s(t-3)=5):
    //       s0 = 12 + 1 = 13
    //       s1 = 13 + 1 = 14
    //       s2 = 14 + 1 = 15
    //       s3 = 15 + 1 = 16
    //   block B samples = [13, 14, 15, 16].
    // (e) QUIT terminator.
    let mut block_bits = Vec::new();
    // BITSHIFT command: function code 6 then uvar(BITSHIFTSIZE) bshift.
    block_bits.extend(encode_uvar(6, FNSIZE));
    block_bits.extend(encode_uvar(4, BITSHIFTSIZE));
    // DIFF1 block A — default block size 8, residuals fit in width 4
    // (encoded energy = 3 since `e + 1 = 4`).
    append_diff1_block(&mut block_bits, 3, &[3, 1, -2, 4, 0, -1, 2, 5]);
    // BLOCKSIZE command: function code 5 then ulong() new_bs = 4.
    block_bits.extend(encode_uvar(5, FNSIZE));
    block_bits.extend(encode_ulong(4, 3)); // 4 fits in 3 bits
                                           // DIFF1 block B — 4 samples, residuals all 1 (width 2 ⇒ encoded
                                           // energy = 1).
    append_diff1_block(&mut block_bits, 1, &[1, 1, 1, 1]);
    // QUIT terminator.
    block_bits.extend(encode_uvar(4, FNSIZE));

    // --- Stage 3: assemble + parse. ---
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
    assert_eq!(parsed.header.blocksize, 8);
    assert_eq!(parsed.header.meanblocks, 0);
    assert_eq!(parsed.header.sample_history_carry_len(), 3);

    // --- Stage 4: skip the header bits via a fresh reader. ---
    let post_version = &buf[5..];
    let mut reader = BitReader::new(post_version);
    let header_bit_count = parsed.bits_consumed_after_v;
    let mut remaining = header_bit_count;
    while remaining > 0 {
        let chunk = remaining.min(32);
        let _ = reader.read_bits(chunk).expect("burn header bits");
        remaining -= chunk;
    }

    // --- Stage 5: dispatch each command and verify state updates. ---
    // The driver-style state here is a single `current_block_size` cell
    // (updated by the BLOCKSIZE command) and a single `current_bshift`
    // cell (updated by the BITSHIFT command). We track both so the
    // test mirrors how a real full-fixture decode driver will carry
    // them across the per-block dispatch loop.
    let mut carry = ChannelCarry::new(parsed.header.sample_history_carry_len() as usize);
    let mut current_block_size = parsed.header.blocksize;

    // (a) BITSHIFT bshift = 4.
    let fc_bs = read_function_code(&mut reader).expect("BITSHIFT code classifies");
    assert_eq!(fc_bs, FunctionCode::Bitshift);
    assert!(!fc_bs.advances_channel_cursor(), "BITSHIFT is housekeeping");
    let current_bshift = read_bitshift_payload(&mut reader).expect("BITSHIFT payload decodes");
    assert_eq!(current_bshift, 4);

    // (b) DIFF1 block A — 8 samples at the default H_blocksize.
    let fc_a = read_function_code(&mut reader).expect("DIFF1 A code classifies");
    assert_eq!(fc_a, FunctionCode::Diff1);
    assert!(fc_a.advances_channel_cursor(), "DIFF1 advances cursor");
    let block_a = decode_diff_block(
        &mut reader,
        PolyOrder::Order1,
        current_block_size,
        &carry,
        0,
    )
    .expect("DIFF1 block A decodes");
    assert_eq!(block_a, vec![3, 4, 2, 6, 6, 5, 7, 12]);
    carry.update_after_block(&block_a);
    // Carry stores pre-shift form per spec/05 §1.4.
    assert_eq!(carry.at(0), 12);
    assert_eq!(carry.at(1), 7);
    assert_eq!(carry.at(2), 5);

    // (c) BLOCKSIZE new_bs = 4.
    let fc_bsz = read_function_code(&mut reader).expect("BLOCKSIZE code classifies");
    assert_eq!(fc_bsz, FunctionCode::Blocksize);
    assert!(
        !fc_bsz.advances_channel_cursor(),
        "BLOCKSIZE is housekeeping"
    );
    let new_bs = read_blocksize_payload(&mut reader).expect("BLOCKSIZE payload decodes");
    assert_eq!(new_bs, 4);
    current_block_size = new_bs;

    // (d) DIFF1 block B — 4 samples at the new sub-block size.
    let fc_b = read_function_code(&mut reader).expect("DIFF1 B code classifies");
    assert_eq!(fc_b, FunctionCode::Diff1);
    let block_b = decode_diff_block(
        &mut reader,
        PolyOrder::Order1,
        current_block_size,
        &carry,
        0,
    )
    .expect("DIFF1 block B decodes");
    assert_eq!(block_b, vec![13, 14, 15, 16]);

    // (e) QUIT terminator.
    let fc_quit = read_function_code(&mut reader).expect("QUIT code classifies");
    assert_eq!(fc_quit, FunctionCode::Quit);

    // The driver-side bit-shift state stayed at 4 throughout — the
    // BLOCKSIZE override doesn't touch the bit-shift, and there was no
    // second BITSHIFT command. Re-assert here to make the persistence
    // explicit.
    assert_eq!(current_bshift, 4);
}
