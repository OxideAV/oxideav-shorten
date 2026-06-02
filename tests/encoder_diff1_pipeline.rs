//! Integration test for the round-14 `BLOCK_FN_DIFF1` predictor
//! encoder.
//!
//! Builds full Shorten files (header + DIFF1 blocks across channels +
//! optional VERBATIM splice + QUIT + zero-pad to byte boundary) using
//! only the public encoder surface, then decodes them through the
//! round-7 whole-stream driver [`oxideav_shorten::decode_stream`] and
//! confirms the recovered per-channel samples are byte-exact with the
//! input.
//!
//! Clean-room provenance: `docs/audio/shorten/spec/03-block-and-predictor.md`
//! §3.2 (DIFF1 wire layout + `s(t) = s(t − 1) + e₁(t)` reconstruction) +
//! `spec/05-state-and-quirks.md` §1 (per-channel sample-history carry
//! convention) + §3.1 (encoded-value-plus-one rule for the residual
//! width) + `spec/02` §1 / §2.1 / §2.2 (MSB-first uvar / svar
//! primitives).

use oxideav_shorten::{
    decode_stream, min_energy_for_diff1, write_byte_aligned_prefix, write_diff1_block,
    write_parameter_block, write_quit_command, write_verbatim_block, BitWriter, ChannelCarry,
    ShortenStreamHeader,
};

fn synth_header(
    version: u8,
    filetype: u32,
    channels: u32,
    blocksize: u32,
    maxlpcorder: u32,
    meanblocks: u32,
    skipbytes: u32,
) -> ShortenStreamHeader {
    ShortenStreamHeader {
        version,
        filetype,
        channels,
        blocksize,
        maxlpcorder,
        meanblocks,
        skipbytes,
    }
}

/// Compute the per-sample DIFF1 first-difference residual stream for
/// `samples` given the per-channel carry-supplied `s(t − 1)` seed.
///
/// Public-surface mirror of the crate-internal helper — callers that
/// only see the encoder's public API replicate this scan to drive
/// [`min_energy_for_diff1`]'s width selection. Per `spec/03` §3.2 the
/// residual is `e₁(t) = s(t) − s(t − 1)` with the initial `s(t − 1)`
/// supplied by `carry.at(0)` per `spec/05` §1.1.
fn diff1_residuals_external(samples: &[i32], carry: &ChannelCarry) -> Vec<i64> {
    let mut s_m1: i64 = carry.at(0) as i64;
    let mut out: Vec<i64> = Vec::with_capacity(samples.len());
    for &s in samples {
        let s_i64 = s as i64;
        out.push(s_i64 - s_m1);
        s_m1 = s_i64;
    }
    out
}

/// Assemble a Shorten file with a sequence of per-channel sample
/// blocks emitted in round-robin order per `spec/03` §2.
///
/// `channel_blocks[c]` is a vector of per-block sample vectors for
/// channel `c`. Each channel must contribute the same number of
/// blocks; the per-block sample-length must equal `header.blocksize`
/// so the round-robin decoder uses the default sub-block size
/// throughout. The encoder maintains a per-channel
/// [`ChannelCarry`] that is updated after each emitted block via
/// `update_after_block`, matching the decoder's `spec/05` §1.3 rule.
fn assemble_diff1_stream(
    header: &ShortenStreamHeader,
    channel_blocks: &[Vec<Vec<i32>>],
) -> Vec<u8> {
    assert_eq!(channel_blocks.len(), header.channels as usize);
    let n_blocks_per_chan = channel_blocks[0].len();
    for ch in channel_blocks {
        assert_eq!(ch.len(), n_blocks_per_chan);
        for blk in ch {
            assert_eq!(blk.len(), header.blocksize as usize);
        }
    }

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, header);

    let n_channels = header.channels as usize;
    let mut carries: Vec<ChannelCarry> = (0..n_channels).map(|_| ChannelCarry::new(3)).collect();

    // Round-robin: for each block index, walk every channel before
    // moving to the next block index.
    for b in 0..n_blocks_per_chan {
        for (c, ch_blocks) in channel_blocks.iter().enumerate() {
            let samples = &ch_blocks[b];
            let residuals = diff1_residuals_external(samples, &carries[c]);
            let energy =
                min_energy_for_diff1(&residuals).expect("residuals fit in natural energy range");
            write_diff1_block(&mut writer, energy, samples, &carries[c])
                .expect("write_diff1_block");
            // Advance the per-channel carry as the decoder will after
            // this block — spec/05 §1.3.
            carries[c].update_after_block(samples);
        }
    }

    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());
    out
}

#[test]
fn mono_single_block_diff1_roundtrips_byte_exact() {
    // Mono, default block size 8, no mean, v2. Carry is zero-initialised
    // at stream start (spec/05 §1.2), so s(t-1) for the first sample is
    // 0 and the residual is the sample value itself.
    let header = synth_header(2, 5, 1, 8, 0, 0, 0);
    let samples: Vec<i32> = vec![10, -7, 3, 0, -1, 5, -4, 2];
    let blocks = vec![vec![samples.clone()]];
    let bytes = assemble_diff1_stream(&header, &blocks);

    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.header, header);
    assert_eq!(dec.channels.len(), 1);
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn stereo_two_blocks_diff1_roundtrips_per_channel() {
    // Two channels, two blocks per channel — verifies the per-channel
    // carry update propagates across consecutive blocks for the same
    // channel under round-robin dispatch.
    let header = synth_header(2, 5, 2, 4, 0, 0, 0);
    let ch0_blocks: Vec<Vec<i32>> = vec![vec![1, 3, 2, 4], vec![5, 4, 6, 5]];
    let ch1_blocks: Vec<Vec<i32>> = vec![vec![0, -1, -2, -1], vec![-3, -2, -4, -3]];
    let blocks = vec![ch0_blocks.clone(), ch1_blocks.clone()];
    let bytes = assemble_diff1_stream(&header, &blocks);

    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.channels.len(), 2);
    let expected_ch0: Vec<i32> = ch0_blocks.into_iter().flatten().collect();
    let expected_ch1: Vec<i32> = ch1_blocks.into_iter().flatten().collect();
    assert_eq!(dec.channels[0], expected_ch0);
    assert_eq!(dec.channels[1], expected_ch1);
}

#[test]
fn diff1_carry_continuity_across_consecutive_blocks_for_same_channel() {
    // The load-bearing assertion: spec/05 §1.1 + §1.3 — after a
    // channel's first block ends on sample `last`, the next block's
    // residual for its first sample is computed against `last`, not
    // against zero. If the carry update were broken, the second
    // block would reconstruct with s(-1) = 0 and disagree with the
    // input by exactly `last` at the block boundary.
    let header = synth_header(2, 5, 1, 4, 0, 0, 0);
    // First block ends at 50; second block starts at 53. The cross-
    // block first-difference is 53 − 50 = 3.
    let block_a: Vec<i32> = vec![10, 20, 35, 50];
    let block_b: Vec<i32> = vec![53, 60, 55, 52];
    let blocks = vec![vec![block_a.clone(), block_b.clone()]];
    let bytes = assemble_diff1_stream(&header, &blocks);

    let dec = decode_stream(&bytes).expect("decode");
    let mut expected: Vec<i32> = block_a;
    expected.extend(block_b);
    assert_eq!(dec.channels[0], expected);
}

#[test]
fn diff1_with_verbatim_prefix_envelope_preserves_payload_and_samples() {
    // Splice a VERBATIM payload before the DIFF1 block — spec/03
    // §3.10 lets the encoder emit the verbatim prefix before any
    // sample-producing command, and §2 says VERBATIM does NOT
    // advance the channel cursor.
    let header = synth_header(2, 5, 1, 4, 0, 0, 32);
    let preamble: Vec<u8> = (0..32).collect();
    let samples: Vec<i32> = vec![3, 6, 4, 8];

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);
    write_verbatim_block(&mut writer, &preamble).expect("verbatim");
    let carry = ChannelCarry::new(3);
    let residuals = diff1_residuals_external(&samples, &carry);
    let energy = min_energy_for_diff1(&residuals).expect("fits");
    write_diff1_block(&mut writer, energy, &samples, &carry).expect("diff1");
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode full stream");
    assert_eq!(dec.verbatim, preamble);
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn diff1_silent_block_uses_minimum_energy() {
    // An all-zero sample block produces all-zero first-differences,
    // which fold to u_max = 0 and select energy 0 (residual width 1)
    // — the narrowest natural width per spec/05 §3.1.
    let carry = ChannelCarry::new(3);
    let samples: Vec<i32> = vec![0; 16];
    let residuals = diff1_residuals_external(&samples, &carry);
    assert_eq!(min_energy_for_diff1(&residuals), Some(0));

    let header = synth_header(2, 5, 1, 16, 0, 0, 0);
    let blocks = vec![vec![samples.clone()]];
    let bytes = assemble_diff1_stream(&header, &blocks);
    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn diff1_block_with_max_natural_first_differences_roundtrips() {
    // First-differences of ±127 fold to u ≤ 254 < 256 = 2^8, so
    // energy = 7 (width 8). Verifies the upper natural-energy edge.
    let header = synth_header(2, 5, 1, 4, 0, 0, 0);
    let samples: Vec<i32> = vec![100, -27, 100, -27];
    // Differences: [100, -127, 127, -127] (zero-seeded carry).
    let blocks = vec![vec![samples.clone()]];
    let bytes = assemble_diff1_stream(&header, &blocks);
    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn diff1_constant_signal_collapses_to_single_first_difference() {
    // A constant signal s(t) = 42 has first-differences [42, 0, 0, ...]
    // (the very first residual carries the seed jump, then every
    // subsequent residual is zero). The natural-energy scan picks the
    // width that fits 42 — folded u = 84 < 128 = 2^7, so energy = 6.
    let header = synth_header(2, 5, 1, 8, 0, 0, 0);
    let samples: Vec<i32> = vec![42; 8];
    let carry = ChannelCarry::new(3);
    let residuals = diff1_residuals_external(&samples, &carry);
    assert_eq!(residuals[0], 42);
    assert!(residuals[1..].iter().all(|&r| r == 0));
    assert_eq!(min_energy_for_diff1(&residuals), Some(6));

    let blocks = vec![vec![samples.clone()]];
    let bytes = assemble_diff1_stream(&header, &blocks);
    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn diff1_three_channels_three_blocks_each_roundtrips() {
    // Stress the round-robin dispatch + per-channel carry update for
    // three channels with non-trivial per-channel trajectories.
    let header = synth_header(2, 5, 3, 4, 0, 0, 0);
    let ch0: Vec<Vec<i32>> = vec![vec![1, 2, 3, 4], vec![5, 4, 6, 5], vec![6, 7, 8, 9]];
    let ch1: Vec<Vec<i32>> = vec![
        vec![-1, -2, -3, -4],
        vec![-5, -4, -6, -5],
        vec![-6, -7, -8, -9],
    ];
    let ch2: Vec<Vec<i32>> = vec![
        vec![10, 12, 14, 13],
        vec![15, 14, 13, 12],
        vec![11, 10, 9, 10],
    ];
    let blocks = vec![ch0.clone(), ch1.clone(), ch2.clone()];
    let bytes = assemble_diff1_stream(&header, &blocks);
    let dec = decode_stream(&bytes).expect("decode");

    let expected_ch0: Vec<i32> = ch0.into_iter().flatten().collect();
    let expected_ch1: Vec<i32> = ch1.into_iter().flatten().collect();
    let expected_ch2: Vec<i32> = ch2.into_iter().flatten().collect();
    assert_eq!(dec.channels[0], expected_ch0);
    assert_eq!(dec.channels[1], expected_ch1);
    assert_eq!(dec.channels[2], expected_ch2);
}

#[test]
fn diff1_min_energy_returns_none_for_over_natural_range() {
    // A first-difference of 1000 folds to u = 2000 ≥ 256 = 2^8,
    // exceeds e = 7. The auto-selection helper surfaces None and the
    // caller would either accept the prefix-zero blow-up by passing
    // MAX_NATURAL_ENERGY explicitly or fall back to a wider svar.
    assert_eq!(min_energy_for_diff1(&[1000, 0]), None);
}
