//! Integration test for the round-16 `BLOCK_FN_DIFF3` predictor
//! encoder.
//!
//! Builds full Shorten files (header + DIFF3 blocks across channels +
//! optional VERBATIM splice + QUIT + zero-pad to byte boundary) using
//! only the public encoder surface, then decodes them through the
//! round-7 whole-stream driver [`oxideav_shorten::decode_stream`] and
//! confirms the recovered per-channel samples are byte-exact with the
//! input.
//!
//! Clean-room provenance: `docs/audio/shorten/spec/03-block-and-predictor.md`
//! §3.4 (DIFF3 wire layout plus the reconstruction recurrence
//! `s(t) = 3·s(t − 1) − 3·s(t − 2) + s(t − 3) + e₃(t)`),
//! `spec/05-state-and-quirks.md` §1 (per-channel sample-history carry
//! convention: `carry.at(0)` is `s(t − 1)`, `carry.at(1)` is
//! `s(t − 2)`, `carry.at(2)` is `s(t − 3)`), `spec/05` §3.1
//! (encoded-value-plus-one rule for the residual width), and `spec/02`
//! §1 / §2.1 / §2.2 (MSB-first uvar / svar primitives).

use oxideav_shorten::{
    decode_stream, min_energy_for_diff3, write_byte_aligned_prefix, write_diff3_block,
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

/// Compute the per-sample DIFF3 third-difference residual stream for
/// `samples` given the per-channel carry-supplied `s(t − 1)` /
/// `s(t − 2)` / `s(t − 3)` seeds.
///
/// Public-surface mirror of the crate-internal helper — callers that
/// only see the encoder's public API replicate this scan to drive
/// [`min_energy_for_diff3`]'s width selection. Per `spec/03` §3.4 the
/// residual is `e₃(t) = s(t) − (3·s(t − 1) − 3·s(t − 2) + s(t − 3))`
/// with the initial `s(t − 1)` supplied by `carry.at(0)`, `s(t − 2)`
/// by `carry.at(1)`, and `s(t − 3)` by `carry.at(2)` per `spec/05`
/// §1.1.
fn diff3_residuals_external(samples: &[i32], carry: &ChannelCarry) -> Vec<i64> {
    let mut s_m1: i64 = carry.at(0) as i64;
    let mut s_m2: i64 = if carry.len() > 1 {
        carry.at(1) as i64
    } else {
        0
    };
    let mut s_m3: i64 = if carry.len() > 2 {
        carry.at(2) as i64
    } else {
        0
    };
    let mut out: Vec<i64> = Vec::with_capacity(samples.len());
    for &s in samples {
        let s_i64 = s as i64;
        let predicted = 3 * s_m1 - 3 * s_m2 + s_m3;
        out.push(s_i64 - predicted);
        s_m3 = s_m2;
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
/// throughout. The encoder maintains a per-channel [`ChannelCarry`]
/// that is updated after each emitted block via `update_after_block`,
/// matching the decoder's `spec/05` §1.3 rule.
fn assemble_diff3_stream(
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
            let residuals = diff3_residuals_external(samples, &carries[c]);
            let energy =
                min_energy_for_diff3(&residuals).expect("residuals fit in natural energy range");
            write_diff3_block(&mut writer, energy, samples, &carries[c])
                .expect("write_diff3_block");
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
fn mono_single_block_diff3_roundtrips_byte_exact() {
    // Mono, default block size 8, no mean, v2. Carry is zero-initialised
    // at stream start (spec/05 §1.2), so s(t-1) = s(t-2) = s(t-3) = 0
    // for the first sample.
    let header = synth_header(2, 5, 1, 8, 0, 0, 0);
    let samples: Vec<i32> = vec![3, 5, 2, 0, -1, 4, -2, 1];
    let blocks = vec![vec![samples.clone()]];
    let bytes = assemble_diff3_stream(&header, &blocks);

    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.header, header);
    assert_eq!(dec.channels.len(), 1);
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn stereo_two_blocks_diff3_roundtrips_per_channel() {
    // Stereo, default block size 4, no mean, v2. Two blocks per channel
    // (4 total commands) exercising the round-robin cursor + carry
    // hand-off across consecutive same-channel blocks.
    let header = synth_header(2, 5, 2, 4, 0, 0, 0);
    let ch0_b0: Vec<i32> = vec![2, 5, 9, 14];
    let ch0_b1: Vec<i32> = vec![20, 27, 35, 44];
    let ch1_b0: Vec<i32> = vec![-3, -8, -15, -24];
    let ch1_b1: Vec<i32> = vec![-35, -48, -63, -80];
    let blocks = vec![
        vec![ch0_b0.clone(), ch0_b1.clone()],
        vec![ch1_b0.clone(), ch1_b1.clone()],
    ];
    let bytes = assemble_diff3_stream(&header, &blocks);

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
fn diff3_carry_continuity_across_consecutive_blocks_for_same_channel() {
    // Mono, two consecutive DIFF3 blocks. The second block's first
    // residual `e₃(0)` depends on the prior block's last three samples
    // via the carry: predicted₀ = 3·s_prev_last − 3·s_prev_2nd_last +
    // s_prev_3rd_last. This test fails if the round-robin emitter does
    // not propagate the carry from block 0 into block 1's first sample.
    let header = synth_header(2, 5, 1, 4, 0, 0, 0);
    let block0: Vec<i32> = vec![1, 2, 4, 7];
    // Block 1 continues a smooth third-order-bounded sequence; the
    // carry-aware seeding ensures the residuals stay small.
    let block1: Vec<i32> = vec![11, 16, 22, 29];
    let blocks = vec![vec![block0.clone(), block1.clone()]];
    let bytes = assemble_diff3_stream(&header, &blocks);

    let dec = decode_stream(&bytes).expect("decode");
    let mut expected = block0.clone();
    expected.extend(block1.iter());
    assert_eq!(dec.channels[0], expected);
}

#[test]
fn diff3_with_verbatim_prefix_envelope_preserves_payload_and_samples() {
    // Header + VERBATIM prefix + DIFF3 sample blocks + QUIT. The
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
    let residuals = diff3_residuals_external(&samples, &carry);
    let energy = min_energy_for_diff3(&residuals).expect("residuals fit");
    write_diff3_block(&mut writer, energy, &samples, &carry).expect("diff3");
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
fn diff3_silent_block_uses_minimum_energy() {
    // All-zero samples → all-zero residuals (predictor = 0 from
    // zero carry). min_energy_for_diff3 returns 0 (width 1).
    let samples: Vec<i32> = vec![0; 8];
    let carry = ChannelCarry::new(3);
    let residuals = diff3_residuals_external(&samples, &carry);
    assert_eq!(residuals, vec![0i64; 8]);
    assert_eq!(min_energy_for_diff3(&residuals), Some(0));
}

#[test]
fn diff3_block_with_pure_quadratic_collapses_to_seed_residuals() {
    // A pure quadratic ramp s(t) = a·t² + b·t + c. Third-differences are
    // identically zero from sample 3 onward; the first three samples
    // carry the seed residuals because the carry is zero.
    //
    // For s(t) = t² + 1 → samples [1, 2, 5, 10, 17, 26]:
    //   t=0: pred = 0;                            r = 1  − 0   = 1
    //   t=1: pred = 3·1 − 3·0 + 0 = 3;            r = 2  − 3   = -1
    //   t=2: pred = 3·2 − 3·1 + 0 = 3;            r = 5  − 3   = 2
    //   t=3 onward: pure-quadratic recurrence holds → r = 0.
    let samples: Vec<i32> = vec![1, 2, 5, 10, 17, 26];
    let carry = ChannelCarry::new(3);
    let residuals = diff3_residuals_external(&samples, &carry);
    assert_eq!(residuals, vec![1i64, -1, 2, 0, 0, 0]);
    // u_max for r ∈ {-1, 1, 2} folds to u ∈ {1, 2, 4}; 4 < 2^3 = 8
    // → e = 2 (width 3).
    assert_eq!(min_energy_for_diff3(&residuals), Some(2));
}

#[test]
fn diff3_block_with_max_natural_third_differences_roundtrips() {
    // Construct a sample stream whose third-differences span the
    // edge of the natural energy range. r ∈ {±127} folds to
    // u ∈ {254, 253} → e = 7 (width 8).
    //
    // Seed sample 1 to produce a r = 127 residual at t=1, then make
    // the recurrence pure-cubic-bounded for the rest:
    //   carry = 0; s(0) = 0 → r(0) = 0.
    //   s(1) = 127 → pred = 3·0 = 0; r(1) = 127.
    //   s(2) = 381 → pred = 3·127 − 3·0 + 0 = 381; r(2) = 0.
    //   s(3) = 762 → pred = 3·381 − 3·127 + 0 = 762; r(3) = 0.
    let header = synth_header(2, 5, 1, 4, 0, 0, 0);
    let samples: Vec<i32> = vec![0, 127, 381, 762];
    let carry = ChannelCarry::new(3);
    let residuals = diff3_residuals_external(&samples, &carry);
    assert_eq!(residuals, vec![0i64, 127, 0, 0]);
    let energy = min_energy_for_diff3(&residuals).expect("natural fits");
    assert_eq!(energy, 7);

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);
    write_diff3_block(&mut writer, energy, &samples, &carry).expect("diff3");
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn diff3_three_channels_three_blocks_each_roundtrips() {
    // Stress test: 3 channels × 3 blocks each = 9 round-robin commands.
    // Each channel carries an independent smooth ramp; the round-robin
    // emitter must thread per-channel carries through 9 emissions
    // without crossing channels. The patterns are kept smooth (linear
    // ramps) so the third-differences stay inside the natural
    // svar-energy range from the zero-carry seed onward.
    let header = synth_header(2, 5, 3, 4, 0, 0, 0);
    let ch0 = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]];
    let ch1 = vec![
        vec![0, -1, -2, -3],
        vec![-4, -5, -6, -7],
        vec![-8, -9, -10, -11],
    ];
    let ch2 = vec![vec![1, 3, 5, 7], vec![9, 11, 13, 15], vec![17, 19, 21, 23]];
    let blocks = vec![ch0.clone(), ch1.clone(), ch2.clone()];
    let bytes = assemble_diff3_stream(&header, &blocks);
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
fn diff3_min_energy_returns_none_for_over_natural_range() {
    // A residual of magnitude 200 folds to u = 400 ≥ 2^8 = 256 →
    // outside the natural range; the helper returns None.
    assert_eq!(min_energy_for_diff3(&[200, 0]), None);
}
