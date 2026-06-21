//! Integration test for the round-7 full-stream decode driver
//! [`oxideav_shorten::decode_stream`].
//!
//! This is the end-to-end counterpart to the per-element pipeline
//! tests (`diff_block_pipeline`, `mean_estimator_pipeline`,
//! `qlpc_block_pipeline`, `housekeeping_pipeline`): instead of
//! re-implementing the orchestration loop inline, it builds a single
//! synthetic v2 stream that exercises every command class in one pass
//! and decodes it through the public driver API.
//!
//! Behavioural anchors:
//! * `docs/audio/shorten/spec/03-block-and-predictor.md` §2 — the
//!   round-robin channel cursor advances on sample-producing commands
//!   and is unchanged by housekeeping commands.
//! * `spec/03` §3.6 / §3.7 — the running sub-block-size and bit-shift
//!   state cells, set by `BLOCK_FN_BLOCKSIZE` / `BLOCK_FN_BITSHIFT`.
//! * `spec/05-state-and-quirks.md` §1.4 — the per-channel carry stores
//!   the pre-shift sample form; the driver applies the bit-shift on
//!   emission only.
//! * `spec/03` §3.10 — verbatim prefix collection.
//! * `spec/03` §3.8 — `BLOCK_FN_QUIT` termination.

use oxideav_shorten::{decode_stream, FNSIZE, LPCQSIZE, LPCQUANT, MAGIC};

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

fn bits_for(v: u32) -> u32 {
    if v == 0 {
        0
    } else {
        32 - v.leading_zeros()
    }
}

fn header_param_bits(
    filetype: u32,
    channels: u32,
    blocksize: u32,
    maxlpcorder: u32,
    meanblocks: u32,
    skipbytes: u32,
) -> Vec<u32> {
    let mut bits = Vec::new();
    bits.extend(encode_ulong(filetype, bits_for(filetype)));
    bits.extend(encode_ulong(channels, bits_for(channels)));
    bits.extend(encode_ulong(blocksize, bits_for(blocksize)));
    bits.extend(encode_ulong(maxlpcorder, bits_for(maxlpcorder)));
    bits.extend(encode_ulong(meanblocks, bits_for(meanblocks)));
    bits.extend(encode_ulong(skipbytes, bits_for(skipbytes)));
    bits
}

fn append_diff_block(out: &mut Vec<u32>, code: u32, energy_encoded: u32, residuals: &[i64]) {
    out.extend(encode_uvar(code, FNSIZE));
    out.extend(encode_uvar(energy_encoded, ENERGYSIZE));
    let width = energy_encoded + 1;
    for &r in residuals {
        out.extend(encode_svar(r, width));
    }
}

const ENERGYSIZE: u32 = 3;

fn assemble(all_bits: &[u32]) -> Vec<u8> {
    let body = pack_bits_msb_first(all_bits);
    let mut buf = Vec::with_capacity(5 + body.len());
    buf.extend_from_slice(&MAGIC);
    buf.push(2);
    buf.extend_from_slice(&body);
    buf
}

/// A single stereo v2 stream exercising VERBATIM + a BITSHIFT +
/// a BLOCKSIZE override + DIFF predictors interleaved across two
/// channels + QUIT, decoded end-to-end via the public driver.
#[test]
fn full_stream_two_channel_with_verbatim_bitshift_blocksize_quit() {
    // Header: filetype = 5 (s16lh), 2 channels, default blocksize = 4,
    // no LPC, no mean, no skip.
    let mut bits = header_param_bits(5, 2, 4, 0, 0, 0);

    // 1. VERBATIM (4-byte "RIFF"-ish prefix). Cursor unchanged.
    bits.extend(encode_uvar(9, FNSIZE)); // VERBATIM = 9
    bits.extend(encode_uvar(4, 5)); // length = 4
    for b in [0x52u8, 0x49, 0x46, 0x46] {
        bits.extend(encode_uvar(b as u32, 8));
    }

    // 2. BITSHIFT bshift = 2 — emitted samples left-shifted by 2,
    //    carry stays in pre-shift form. Cursor unchanged.
    bits.extend(encode_uvar(6, FNSIZE)); // BITSHIFT = 6
    bits.extend(encode_uvar(2, 2)); // bshift = 2 (uvar(BITSHIFTSIZE=2))

    // 3. DIFF1 ch0: residuals [1, 1, 1, 1] over zero carry
    //    -> pre-shift [1, 2, 3, 4] -> emitted (<<2) [4, 8, 12, 16].
    //    Cursor 0 -> 1.
    append_diff_block(&mut bits, 1, 3, &[1, 1, 1, 1]);

    // 4. DIFF0 ch1: residuals [5, 6, 7, 8] (mean = 0)
    //    -> pre-shift [5, 6, 7, 8] -> emitted (<<2) [20, 24, 28, 32].
    //    Cursor 1 -> 0.
    append_diff_block(&mut bits, 0, 3, &[5, 6, 7, 8]);

    // 5. BLOCKSIZE override new_bs = 2. Cursor unchanged (still ch0).
    bits.extend(encode_uvar(5, FNSIZE)); // BLOCKSIZE = 5
    bits.extend(encode_ulong(2, 2));

    // 6. DIFF1 ch0 second block, bs = 2: residuals [0, 0] over the
    //    pre-shift carry (carry[0] = 4 from block 3) -> pre-shift
    //    [4, 4] -> emitted (<<2) [16, 16]. Cursor 0 -> 1.
    append_diff_block(&mut bits, 1, 3, &[0, 0]);

    // 7. QUIT.
    bits.extend(encode_uvar(4, FNSIZE));

    let buf = assemble(&bits);
    let dec = decode_stream(&buf).expect("full stream decodes");

    // Header round-trip.
    assert_eq!(dec.header.channels, 2);
    assert_eq!(dec.header.blocksize, 4);
    assert_eq!(dec.header.filetype, 5);

    // Verbatim prefix collected.
    assert_eq!(dec.verbatim, vec![0x52, 0x49, 0x46, 0x46]);

    // Two channels.
    assert_eq!(dec.channels.len(), 2);
    // ch0: first DIFF1 block (<<2) then the bs=2 second DIFF1 block.
    assert_eq!(dec.channels[0], vec![4, 8, 12, 16, 16, 16]);
    // ch1: the single DIFF0 block (<<2).
    assert_eq!(dec.channels[1], vec![20, 24, 28, 32]);
    assert_eq!(dec.channel_len(0), 6);
    assert_eq!(dec.channel_len(1), 4);
}

/// A stereo v2 stream exercising the order-2 and order-3
/// polynomial-difference predictors (`BLOCK_FN_DIFF2` / `BLOCK_FN_DIFF3`)
/// end-to-end through the public [`decode_stream`] driver, including the
/// per-channel sample-history carry hand-off across two consecutive
/// blocks of the same channel.
///
/// Behavioural anchors: `spec/03` §3.3 (DIFF2 line-fit predictor
/// `ŝ₂(t) = 2·s(t-1) − s(t-2)`), §3.4 (DIFF3 quadratic-fit predictor
/// `ŝ₃(t) = 3·s(t-1) − 3·s(t-2) + s(t-3)`), §2 (round-robin channel
/// cursor), and `spec/05` §1 (most-recent-first carry indexing, zero
/// initialisation, full refresh from blocks of `bs ≥ CARRY_LEN`).
///
/// The prior `full_stream_*` test covers DIFF0 + DIFF1 via the driver;
/// this test closes the DIFF2 + DIFF3 driver path so all four
/// polynomial-difference orders are decoder-direct byte-exact through
/// `decode_stream` (not only through the encoder round-trip suite).
#[test]
fn full_stream_diff2_diff3_carry_handoff_through_driver() {
    // Header: filetype = 5 (s16lh), 2 channels, blocksize = 4, no LPC,
    // no mean, no skip.
    let mut bits = header_param_bits(5, 2, 4, 0, 0, 0);

    // Block 1 — DIFF2 ch0, residuals [0, 0, 0, 0] over zero carry.
    //   carry seeds s(t-1)=s(t-2)=0.
    //   s0 = 2*0 - 0 + 0 = 0
    //   s1 = 2*0 - 0 + 0 = 0
    //   s2 = 2*0 - 0 + 0 = 0
    //   s3 = 2*0 - 0 + 0 = 0
    //   ch0 block 1 = [0, 0, 0, 0]; carry = [0, 0, 0]. Cursor 0 -> 1.
    append_diff_block(&mut bits, 2, 3, &[0, 0, 0, 0]);

    // Block 2 — DIFF3 ch1, residuals [1, 0, 0, 0] over zero carry.
    //   s0 = 3*0 - 3*0 + 0 + 1 = 1
    //   s1 = 3*1 - 3*0 + 0 + 0 = 3   (s(t-1)=1, s(t-2)=0, s(t-3)=0)
    //   s2 = 3*3 - 3*1 + 0 + 0 = 6   (s(t-1)=3, s(t-2)=1, s(t-3)=0)
    //   s3 = 3*6 - 3*3 + 1 + 0 = 10  (s(t-1)=6, s(t-2)=3, s(t-3)=1)
    //   ch1 block 1 = [1, 3, 6, 10]; carry = [10, 6, 3]. Cursor 1 -> 0.
    append_diff_block(&mut bits, 3, 3, &[1, 0, 0, 0]);

    // Block 3 — DIFF2 ch0 second block, residuals [2, 0, 0, 0] over the
    //   ch0 carry [0, 0, 0] from block 1 (still all zero):
    //   s0 = 2*0 - 0 + 2 = 2
    //   s1 = 2*2 - 0 + 0 = 4   (s(t-1)=2, s(t-2)=0)
    //   s2 = 2*4 - 2 + 0 = 6   (s(t-1)=4, s(t-2)=2)
    //   s3 = 2*6 - 4 + 0 = 8   (s(t-1)=6, s(t-2)=4)
    //   ch0 block 2 = [2, 4, 6, 8]. Cursor 0 -> 1.
    append_diff_block(&mut bits, 2, 3, &[2, 0, 0, 0]);

    // Block 4 — DIFF3 ch1 second block, residuals [0, 0, 0, 0] over the
    //   ch1 carry [10, 6, 3] from block 2 (s(t-1)=10, s(t-2)=6, s(t-3)=3):
    //   s0 = 3*10 - 3*6 + 3 + 0 = 15
    //   s1 = 3*15 - 3*10 + 6 + 0 = 21   (s(t-1)=15, s(t-2)=10, s(t-3)=6)
    //   s2 = 3*21 - 3*15 + 10 + 0 = 28  (s(t-1)=21, s(t-2)=15, s(t-3)=10)
    //   s3 = 3*28 - 3*21 + 15 + 0 = 36  (s(t-1)=28, s(t-2)=21, s(t-3)=15)
    //   ch1 block 2 = [15, 21, 28, 36]. Cursor 1 -> 0.
    append_diff_block(&mut bits, 3, 3, &[0, 0, 0, 0]);

    // QUIT.
    bits.extend(encode_uvar(4, FNSIZE));

    let buf = assemble(&bits);
    let dec = decode_stream(&buf).expect("DIFF2/DIFF3 stream decodes");

    assert_eq!(dec.header.channels, 2);
    assert_eq!(dec.channels.len(), 2);
    // ch0: DIFF2 block 1 then DIFF2 block 2 (carry-continued).
    assert_eq!(dec.channels[0], vec![0, 0, 0, 0, 2, 4, 6, 8]);
    // ch1: DIFF3 block 1 then DIFF3 block 2 (carry-continued).
    assert_eq!(dec.channels[1], vec![1, 3, 6, 10, 15, 21, 28, 36]);
    assert_eq!(dec.channel_len(0), 8);
    assert_eq!(dec.channel_len(1), 8);
    // QUIT zero-padding to the next byte boundary is spec-conformant
    // (`spec/05` §4).
    assert!(dec.quit_padding.is_spec_conformant());
}

/// A mono v2 stream exercising the running-mean estimator's effect on
/// `BLOCK_FN_DIFF0` reconstruction and `BLOCK_FN_ZERO` emission end-to-end
/// through the public [`decode_stream`] driver, with the mean updating
/// across consecutive blocks.
///
/// Behavioural anchors: `spec/05` §2.3 (DIFF0 reconstruct
/// `s(t) = e0(t) + μ_chan`), §2.4 (ZERO emits `bs` samples all equal to
/// `μ_chan`), §2.5 (per-block mean `μ_blk = trunc_div(Σ + bs/2, bs)` and
/// running mean `μ_chan = trunc_div(Σ_slots + N/2, N)`, both with the
/// always-positive `+divisor/2` bias and truncation toward zero), and
/// §2.2 (the sliding window evicts the oldest slot and appends the new
/// per-block mean at the most-recent slot).
///
/// The mean estimator's non-zero path was previously verified only at
/// the [`MeanEstimator`]/predictor level and through the encoder
/// round-trip suite; this closes the decoder-direct driver path on a
/// hand-built bitstream where DIFF0 carries a non-zero running mean and
/// ZERO emits it.
#[test]
fn full_stream_mean_estimator_diff0_and_zero_through_driver() {
    // Header: filetype = 5 (s16lh), 1 channel, blocksize = 4,
    // no LPC, H_meanblocks = 1 (one-slot sliding window), no skip.
    let mut bits = header_param_bits(5, 1, 4, 0, 1, 0);

    // The estimator holds a single slot initialised to 0, so:
    //   μ_chan = trunc_div(slot + 1/2, 1) = slot   (1/2 = 0).
    //
    // Block 1 — DIFF0, μ_chan = 0, residuals [8, 8, 8, 8]:
    //   s(t) = e0(t) + 0 = [8, 8, 8, 8].
    //   per-block mean μ_blk = trunc_div(32 + 4/2, 4) = trunc_div(34, 4)
    //                        = 8. slot -> 8 (μ_chan becomes 8).
    append_diff_block(&mut bits, 0, 3, &[8, 8, 8, 8]);

    // Block 2 — ZERO, μ_chan = 8 at block start. Emits 4 samples all = 8.
    //   no residuals. per-block mean μ_blk = trunc_div(32 + 2, 4) = 8.
    //   slot stays 8 (μ_chan stays 8).
    bits.extend(encode_uvar(8, FNSIZE)); // ZERO = 8

    // Block 3 — DIFF0, μ_chan = 8, residuals [0, 2, -2, 0]:
    //   s(t) = e0(t) + 8 = [8, 10, 6, 8].
    //   per-block mean μ_blk = trunc_div(32 + 2, 4) = 8. slot stays 8.
    append_diff_block(&mut bits, 0, 3, &[0, 2, -2, 0]);

    // Block 4 — DIFF0, μ_chan = 8, residuals [-8, -8, -8, -8]:
    //   s(t) = e0(t) + 8 = [0, 0, 0, 0].
    //   per-block mean μ_blk = trunc_div(0 + 2, 4) = 0. slot -> 0.
    append_diff_block(&mut bits, 0, 3, &[-8, -8, -8, -8]);

    // Block 5 — ZERO, μ_chan = 0 at block start. Emits 4 samples all = 0.
    bits.extend(encode_uvar(8, FNSIZE)); // ZERO = 8

    // QUIT.
    bits.extend(encode_uvar(4, FNSIZE));

    let buf = assemble(&bits);
    let dec = decode_stream(&buf).expect("mean-estimator stream decodes");

    assert_eq!(dec.header.channels, 1);
    assert_eq!(dec.header.meanblocks, 1);
    assert_eq!(dec.channels.len(), 1);
    assert_eq!(
        dec.channels[0],
        vec![
            8, 8, 8, 8, // block 1 DIFF0 (μ = 0)
            8, 8, 8, 8, // block 2 ZERO (μ = 8)
            8, 10, 6, 8, // block 3 DIFF0 (μ = 8)
            0, 0, 0, 0, // block 4 DIFF0 (μ = 8, residuals -8)
            0, 0, 0, 0, // block 5 ZERO (μ = 0)
        ]
    );
    assert_eq!(dec.channel_len(0), 20);
    assert!(dec.quit_padding.is_spec_conformant());
}

/// Append a synthetic `BLOCK_FN_QLPC` command (function code + order +
/// coefficients + energy + residuals) to `out`.
fn append_qlpc_block(out: &mut Vec<u32>, coefs: &[i64], energy_encoded: u32, residuals: &[i64]) {
    out.extend(encode_uvar(7, FNSIZE)); // QLPC = 7
    out.extend(encode_uvar(coefs.len() as u32, LPCQSIZE));
    for &c in coefs {
        out.extend(encode_svar(c, LPCQUANT));
    }
    out.extend(encode_uvar(energy_encoded, ENERGYSIZE));
    let width = energy_encoded + 1;
    for &r in residuals {
        out.extend(encode_svar(r, width));
    }
}

/// A stereo v2 stream with `H_maxlpcorder = 3` exercising the
/// quantised-LPC predictor (`BLOCK_FN_QLPC`) end-to-end through the
/// public [`decode_stream`] driver, interleaved with a DIFF block on the
/// other channel, including the per-channel carry hand-off across two
/// consecutive QLPC blocks of the same channel.
///
/// Behavioural anchors: `spec/03` §3.5 (QLPC wire layout
/// `<order> <coef>×order <energy> <residual>×bs` and reconstruction
/// `s(t) = Σᵢ aᵢ·s(t-i) + e(t)` applied without scaling), §3.11 (carry
/// length `max(3, H_maxlpcorder)`), §2 (round-robin cursor), `spec/02`
/// §4.3/§4.4 (`LPCQSIZE = 2`, `LPCQUANT = 2`), and `spec/05` §3.2 (the
/// QLPC energy field follows the same `+1` residual-width convention).
///
/// The prior `qlpc_block_pipeline` test drives `decode_qlpc_block`
/// directly with a locally-overridden block size; this is the first
/// QLPC decode that flows through the public `decode_stream` driver with
/// the header-declared block size and the driver-owned channel cursor +
/// carry update.
#[test]
fn full_stream_qlpc_carry_handoff_through_driver() {
    // Header: filetype = 5 (s16lh), 2 channels, blocksize = 4,
    // H_maxlpcorder = 3 (carry len = max(3,3) = 3), no mean, no skip.
    let mut bits = header_param_bits(5, 2, 4, 3, 0, 0);

    // Block 1 — QLPC ch0, order 2 (a1 = 2, a2 = -1, the DIFF2 line-fit),
    //   residuals [1, 0, 0, 0] over zero carry (s(t-1)=s(t-2)=0):
    //     s0 = 2*0 - 1*0 + 1 = 1
    //     s1 = 2*1 - 1*0 + 0 = 2
    //     s2 = 2*2 - 1*1 + 0 = 3
    //     s3 = 2*3 - 1*2 + 0 = 4
    //   ch0 block 1 = [1, 2, 3, 4]; carry = [4, 3, 2]. Cursor 0 -> 1.
    append_qlpc_block(&mut bits, &[2, -1], 3, &[1, 0, 0, 0]);

    // Block 2 — DIFF1 ch1, residuals [5, -1, -1, -1] over zero carry:
    //   s0 = 0 + 5 = 5; s1 = 5 - 1 = 4; s2 = 4 - 1 = 3; s3 = 3 - 1 = 2.
    //   ch1 block 1 = [5, 4, 3, 2]. Cursor 1 -> 0.
    append_diff_block(&mut bits, 1, 3, &[5, -1, -1, -1]);

    // Block 3 — QLPC ch0 second block, order 2 (a1 = 2, a2 = -1),
    //   default block size 4, residuals [0, 0, 0, 0] over the ch0 carry
    //   [4, 3, 2] (s(t-1)=4, s(t-2)=3):
    //     s0 = 2*4 - 1*3 + 0 = 5
    //     s1 = 2*5 - 1*4 + 0 = 6   (s(t-1)=5, s(t-2)=4)
    //     s2 = 2*6 - 1*5 + 0 = 7   (s(t-1)=6, s(t-2)=5)
    //     s3 = 2*7 - 1*6 + 0 = 8   (s(t-1)=7, s(t-2)=6)
    //   ch0 block 2 = [5, 6, 7, 8]. Cursor 0 -> 1.
    append_qlpc_block(&mut bits, &[2, -1], 3, &[0, 0, 0, 0]);

    // Block 4 — DIFF1 ch1 second block, residuals [0, 0, 0, 0] over the
    //   ch1 carry [2, 3, 4] (s(t-1)=2): s = [2, 2, 2, 2]. Cursor 1 -> 0.
    append_diff_block(&mut bits, 1, 3, &[0, 0, 0, 0]);

    // QUIT.
    bits.extend(encode_uvar(4, FNSIZE));

    let buf = assemble(&bits);
    let dec = decode_stream(&buf).expect("QLPC stream decodes");

    assert_eq!(dec.header.channels, 2);
    assert_eq!(dec.header.maxlpcorder, 3);
    assert_eq!(dec.channels.len(), 2);
    // ch0: QLPC block 1 then QLPC block 2 (carry-continued).
    assert_eq!(dec.channels[0], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    // ch1: DIFF1 block 1 then DIFF1 block 2 (carry-continued).
    assert_eq!(dec.channels[1], vec![5, 4, 3, 2, 2, 2, 2, 2]);
    assert_eq!(dec.channel_len(0), 8);
    assert_eq!(dec.channel_len(1), 8);
    assert!(dec.quit_padding.is_spec_conformant());
}
