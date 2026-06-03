//! Integration test for the round-15 `BLOCK_FN_DIFF2` predictor
//! encoder.
//!
//! Builds full Shorten files (header + DIFF2 blocks across channels +
//! optional VERBATIM splice + QUIT + zero-pad to byte boundary) using
//! only the public encoder surface, then decodes them through the
//! round-7 whole-stream driver [`oxideav_shorten::decode_stream`] and
//! confirms the recovered per-channel samples are byte-exact with the
//! input.
//!
//! Clean-room provenance: `docs/audio/shorten/spec/03-block-and-predictor.md`
//! §3.3 (DIFF2 wire layout + `s(t) = 2·s(t − 1) − s(t − 2) + e₂(t)`
//! reconstruction) + `spec/05-state-and-quirks.md` §1 (per-channel
//! sample-history carry convention: `carry.at(0)` is `s(t − 1)`,
//! `carry.at(1)` is `s(t − 2)`) + §3.1 (encoded-value-plus-one rule
//! for the residual width) + `spec/02` §1 / §2.1 / §2.2 (MSB-first
//! uvar / svar primitives).

use oxideav_shorten::{
    decode_stream, min_energy_for_diff2, write_byte_aligned_prefix, write_diff2_block,
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

/// Compute the per-sample DIFF2 second-difference residual stream for
/// `samples` given the per-channel carry-supplied `s(t − 1)` /
/// `s(t − 2)` seeds.
///
/// Public-surface mirror of the crate-internal helper — callers that
/// only see the encoder's public API replicate this scan to drive
/// [`min_energy_for_diff2`]'s width selection. Per `spec/03` §3.3 the
/// residual is `e₂(t) = s(t) − (2·s(t − 1) − s(t − 2))` with
/// the initial `s(t − 1)` supplied by `carry.at(0)` and `s(t − 2)` by
/// `carry.at(1)` per `spec/05` §1.1.
fn diff2_residuals_external(samples: &[i32], carry: &ChannelCarry) -> Vec<i64> {
    let mut s_m1: i64 = carry.at(0) as i64;
    let mut s_m2: i64 = if carry.len() > 1 {
        carry.at(1) as i64
    } else {
        0
    };
    let mut out: Vec<i64> = Vec::with_capacity(samples.len());
    for &s in samples {
        let s_i64 = s as i64;
        let predicted = 2 * s_m1 - s_m2;
        out.push(s_i64 - predicted);
        s_m2 = s_m1;
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
fn assemble_diff2_stream(
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
            let residuals = diff2_residuals_external(samples, &carries[c]);
            let energy =
                min_energy_for_diff2(&residuals).expect("residuals fit in natural energy range");
            write_diff2_block(&mut writer, energy, samples, &carries[c])
                .expect("write_diff2_block");
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
fn mono_single_block_diff2_roundtrips_byte_exact() {
    // Mono, default block size 8, no mean, v2. Carry is zero-initialised
    // at stream start (spec/05 §1.2), so s(t-1) = s(t-2) = 0 for the
    // first sample.
    let header = synth_header(2, 5, 1, 8, 0, 0, 0);
    let samples: Vec<i32> = vec![10, -7, 3, 0, -1, 5, -4, 2];
    let blocks = vec![vec![samples.clone()]];
    let bytes = assemble_diff2_stream(&header, &blocks);

    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.header, header);
    assert_eq!(dec.channels.len(), 1);
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn stereo_two_blocks_diff2_roundtrips_per_channel() {
    // Stereo, default block size 4, no mean, v2. Two blocks per channel
    // (4 total commands) exercising the round-robin cursor + carry
    // hand-off across consecutive same-channel blocks.
    let header = synth_header(2, 5, 2, 4, 0, 0, 0);
    let ch0_b0: Vec<i32> = vec![5, 10, 15, 20];
    let ch0_b1: Vec<i32> = vec![25, 30, 35, 40];
    let ch1_b0: Vec<i32> = vec![-3, -6, -9, -12];
    let ch1_b1: Vec<i32> = vec![-15, -18, -21, -24];
    let blocks = vec![
        vec![ch0_b0.clone(), ch0_b1.clone()],
        vec![ch1_b0.clone(), ch1_b1.clone()],
    ];
    let bytes = assemble_diff2_stream(&header, &blocks);

    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.channels.len(), 2);
    // Channel 0 carries the concatenated block sequence in time order.
    let mut ch0_expected = ch0_b0.clone();
    ch0_expected.extend(ch0_b1.iter());
    let mut ch1_expected = ch1_b0.clone();
    ch1_expected.extend(ch1_b1.iter());
    assert_eq!(dec.channels[0], ch0_expected);
    assert_eq!(dec.channels[1], ch1_expected);
}

#[test]
fn diff2_carry_continuity_across_consecutive_blocks_for_same_channel() {
    // Mono, two consecutive DIFF2 blocks. The second block's first
    // residual `e₂(0)` depends on the prior block's last two samples
    // via the carry: predicted₀ = 2·s_prev_last − s_prev_2nd_last.
    // This test fails if the round-robin emitter does not propagate
    // the carry from block 0 into block 1's first sample.
    let header = synth_header(2, 5, 1, 4, 0, 0, 0);
    let block0: Vec<i32> = vec![1, 2, 3, 4];
    // Block 1 continues the linear ramp; second-differences on a pure
    // ramp are 0 from sample 2 onward, and the carry-aware seeding
    // ensures samples 0 and 1 also have small residuals.
    let block1: Vec<i32> = vec![5, 6, 7, 8];
    let blocks = vec![vec![block0.clone(), block1.clone()]];
    let bytes = assemble_diff2_stream(&header, &blocks);

    let dec = decode_stream(&bytes).expect("decode");
    let mut expected = block0.clone();
    expected.extend(block1.iter());
    assert_eq!(dec.channels[0], expected);
}

#[test]
fn diff2_with_verbatim_prefix_envelope_preserves_payload_and_samples() {
    // Header + VERBATIM prefix + DIFF2 sample blocks + QUIT. The
    // verbatim prefix should be recovered byte-for-byte, and the
    // per-channel sample stream should round-trip.
    let header = synth_header(2, 5, 1, 4, 0, 0, 16);
    let preamble = b"RIFF\x10\x00\x00\x00WAVEfmt ";
    assert_eq!(preamble.len(), 16);
    let samples: Vec<i32> = vec![2, 5, 9, 14];

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);
    write_verbatim_block(&mut writer, preamble).expect("verbatim");
    let carry = ChannelCarry::new(3);
    let residuals = diff2_residuals_external(&samples, &carry);
    let energy = min_energy_for_diff2(&residuals).expect("residuals fit");
    write_diff2_block(&mut writer, energy, &samples, &carry).expect("diff2");
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    assert_eq!(dec.header, header);
    assert_eq!(dec.verbatim, preamble.to_vec());
    assert_eq!(dec.channels.len(), 1);
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn diff2_silent_block_uses_minimum_energy() {
    // All-zero samples → all-zero residuals (predictor = 0 from
    // zero carry). min_energy_for_diff2 returns 0 (width 1).
    let samples: Vec<i32> = vec![0; 8];
    let carry = ChannelCarry::new(3);
    let residuals = diff2_residuals_external(&samples, &carry);
    assert_eq!(residuals, vec![0i64; 8]);
    assert_eq!(min_energy_for_diff2(&residuals), Some(0));
}

#[test]
fn diff2_block_with_pure_ramp_collapses_to_seed_residuals() {
    // A pure linear ramp s(t) = a·t + b. Second-differences are
    // identically zero from sample 2 onward; the first two samples
    // carry the seed residuals because the carry is zero.
    //
    // For samples [1, 2, 3, 4, 5, 6]:
    //   t=0: predicted = 0;        r = 1 − 0 = 1
    //   t=1: predicted = 2·1 − 0 = 2; r = 2 − 2 = 0
    //   t=2 onward: pure-ramp recurrence holds → r = 0.
    let samples: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
    let carry = ChannelCarry::new(3);
    let residuals = diff2_residuals_external(&samples, &carry);
    assert_eq!(residuals, vec![1i64, 0, 0, 0, 0, 0]);
    // u_max for r = 1 folds to u = 2; 2 < 2^2 = 4 → e = 1 (width 2).
    assert_eq!(min_energy_for_diff2(&residuals), Some(1));
}

#[test]
fn diff2_block_with_max_natural_second_differences_roundtrips() {
    // Construct a sample stream whose second-differences span the
    // edge of the natural energy range. Use a sample stream whose
    // r ∈ {±127} folds to u ∈ {254, 253} → e = 7 (width 8).
    //
    // We choose samples manually: starting from carry = 0,
    //   s(0) = 0 → r(0) = 0;
    //   s(1) = 127 → r(1) = 127 − 2·0 = 127;
    //   s(2) = 0 → r(2) = 0 − (2·127 − 0) = -254;  // OUT OF RANGE
    // Instead pick a simpler construction:
    //   s = [0, 127, 254, 381]:
    //     r(0) = 0;
    //     r(1) = 127 − 0 = 127;
    //     r(2) = 254 − (2·127 − 0) = 0;
    //     r(3) = 381 − (2·254 − 127) = 0.
    let header = synth_header(2, 5, 1, 4, 0, 0, 0);
    let samples: Vec<i32> = vec![0, 127, 254, 381];
    let carry = ChannelCarry::new(3);
    let residuals = diff2_residuals_external(&samples, &carry);
    assert_eq!(residuals, vec![0i64, 127, 0, 0]);
    let energy = min_energy_for_diff2(&residuals).expect("natural fits");
    assert_eq!(energy, 7);

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);
    write_diff2_block(&mut writer, energy, &samples, &carry).expect("diff2");
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn diff2_three_channels_three_blocks_each_roundtrips() {
    // Stress test: 3 channels × 3 blocks each = 9 round-robin commands.
    // Each channel carries an independent ramp + dip pattern; the
    // round-robin emitter must thread per-channel carries through 9
    // emissions without crossing channels.
    let header = synth_header(2, 5, 3, 4, 0, 0, 0);
    let ch0 = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]];
    let ch1 = vec![
        vec![100, 99, 98, 97],
        vec![96, 95, 94, 93],
        vec![92, 91, 90, 89],
    ];
    let ch2 = vec![
        vec![-1, -2, -3, -4],
        vec![-5, -6, -7, -8],
        vec![-9, -10, -11, -12],
    ];
    let blocks = vec![ch0.clone(), ch1.clone(), ch2.clone()];
    let bytes = assemble_diff2_stream(&header, &blocks);
    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.channels.len(), 3);
    for (idx, ch_blocks) in [&ch0, &ch1, &ch2].iter().enumerate() {
        let mut expected: Vec<i32> = Vec::new();
        for b in ch_blocks.iter() {
            expected.extend(b.iter());
        }
        assert_eq!(dec.channels[idx], expected, "channel {idx}");
    }
}

#[test]
fn diff2_min_energy_returns_none_for_over_natural_range() {
    // A residual of magnitude 200 folds to u = 400 ≥ 2^8 = 256 →
    // outside the natural range; the helper returns None.
    assert_eq!(min_energy_for_diff2(&[200, 0]), None);
}
