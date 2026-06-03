//! Integration test for the round-17 `BLOCK_FN_QLPC` predictor encoder.
//!
//! Builds full Shorten files (header + QLPC blocks across channels +
//! optional VERBATIM splice + QUIT + zero-pad to byte boundary) using
//! only the public encoder surface, then decodes them through the
//! round-7 whole-stream driver [`oxideav_shorten::decode_stream`] and
//! confirms the recovered per-channel samples are byte-exact with the
//! input.
//!
//! Clean-room provenance: `docs/audio/shorten/spec/03-block-and-predictor.md`
//! §3.5 (QLPC wire layout `<fn=7> <order> <coef>×order <energy>
//! <residual>×bs` plus the reconstruction recurrence
//! `s(t) = Σᵢ aᵢ·s(t − i) + e_QLPC(t)`),
//! `spec/05-state-and-quirks.md` §1 (per-channel sample-history carry
//! convention: `carry.at(i) = s(t − i − 1)` per-channel), `spec/05`
//! §3.1 (encoded-value-plus-one rule for the residual width),
//! `spec/02` §1 / §2.1 / §2.2 (MSB-first uvar / svar primitives), and
//! `spec/02` §4.3 / §4.4 (`LPCQSIZE = 2`, `LPCQUANT = 2`).

use oxideav_shorten::{
    decode_stream, min_energy_for_qlpc, qlpc_residuals, write_byte_aligned_prefix,
    write_parameter_block, write_qlpc_block, write_quit_command, write_verbatim_block, BitWriter,
    ChannelCarry, ShortenStreamHeader,
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

/// Assemble a Shorten file with a sequence of per-channel QLPC sample
/// blocks emitted in round-robin order per `spec/03` §2.
///
/// `channel_blocks[c]` is a vector of per-block sample vectors for
/// channel `c`. Each channel must contribute the same number of
/// blocks; each block must use the same coefficient vector
/// (`coefs[c]`) so the test's residual scan corresponds to a fixed
/// predictor across blocks. The per-block sample-length must equal
/// `header.blocksize`. The encoder maintains a per-channel
/// [`ChannelCarry`] that is updated after each emitted block via
/// `update_after_block`, matching the decoder's `spec/05` §1.3 rule.
fn assemble_qlpc_stream(
    header: &ShortenStreamHeader,
    coefs: &[Vec<i64>],
    channel_blocks: &[Vec<Vec<i32>>],
) -> Vec<u8> {
    assert_eq!(channel_blocks.len(), header.channels as usize);
    assert_eq!(coefs.len(), header.channels as usize);
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
    // Carry length sized to the decoder's max(3, H_maxlpcorder).
    let carry_len = core::cmp::max(3, header.maxlpcorder as usize);
    let mut carries: Vec<ChannelCarry> = (0..n_channels)
        .map(|_| ChannelCarry::new(carry_len))
        .collect();

    for b in 0..n_blocks_per_chan {
        for (c, ch_blocks) in channel_blocks.iter().enumerate() {
            let samples = &ch_blocks[b];
            let residuals = qlpc_residuals(samples, &coefs[c], &carries[c]).expect("residual scan");
            let energy =
                min_energy_for_qlpc(&residuals).expect("residuals fit in natural energy range");
            write_qlpc_block(&mut writer, energy, &coefs[c], samples, &carries[c])
                .expect("write_qlpc_block");
            carries[c].update_after_block(samples);
        }
    }

    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());
    out
}

#[test]
fn mono_single_block_qlpc_order_zero_roundtrips() {
    // Order 0 → no coefficients → predictor outputs 0 → residual is
    // the sample. With H_maxlpcorder = 0, the decoder still accepts
    // a QLPC block whose per-block order field is 0.
    let header = synth_header(2, 5, 1, 6, 0, 0, 0);
    let samples: Vec<i32> = vec![3, -2, 7, 0, -5, 1];
    let coefs: Vec<Vec<i64>> = vec![vec![]];
    let blocks = vec![vec![samples.clone()]];
    let bytes = assemble_qlpc_stream(&header, &coefs, &blocks);

    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.header, header);
    assert_eq!(dec.channels.len(), 1);
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn mono_single_block_qlpc_order_one_a1_one_roundtrips() {
    // a1 = 1 with zero carry → predictor = previous sample → residual
    // = first-difference. The decoder reconstructs by running-sum.
    let header = synth_header(2, 5, 1, 6, 1, 0, 0);
    let samples: Vec<i32> = vec![5, 8, 12, 11, 14, 18];
    let coefs: Vec<Vec<i64>> = vec![vec![1]];
    let blocks = vec![vec![samples.clone()]];
    let bytes = assemble_qlpc_stream(&header, &coefs, &blocks);

    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.header, header);
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn mono_single_block_qlpc_order_two_with_two_coefficients_roundtrips() {
    // a1 = 2, a2 = -1: this is the order-2 polynomial-difference
    // predictor (DIFF2) wire-encoded as QLPC. The reconstruction is
    // identical: s(t) = 2·s(t-1) − s(t-2) + e(t).
    let header = synth_header(2, 5, 1, 6, 2, 0, 0);
    let samples: Vec<i32> = vec![1, 3, 5, 7, 9, 11];
    let coefs: Vec<Vec<i64>> = vec![vec![2, -1]];
    let blocks = vec![vec![samples.clone()]];
    let bytes = assemble_qlpc_stream(&header, &coefs, &blocks);

    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn mono_single_block_qlpc_order_three_roundtrips() {
    // a1 = 3, a2 = -3, a3 = 1: this is the order-3 polynomial-
    // difference predictor (DIFF3) wire-encoded as QLPC. Residuals
    // for a pure-cubic ramp are zero from sample 3 onward.
    let header = synth_header(2, 5, 1, 6, 3, 0, 0);
    let samples: Vec<i32> = vec![0, 1, 8, 27, 64, 125];
    let coefs: Vec<Vec<i64>> = vec![vec![3, -3, 1]];
    let blocks = vec![vec![samples.clone()]];
    let bytes = assemble_qlpc_stream(&header, &coefs, &blocks);

    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn stereo_two_blocks_qlpc_roundtrips_per_channel() {
    // Stereo, default block size 4, H_maxlpcorder = 1. Two blocks per
    // channel exercising round-robin cursor + per-channel carry
    // hand-off across consecutive same-channel blocks.
    let header = synth_header(2, 5, 2, 4, 1, 0, 0);
    let ch0_b0: Vec<i32> = vec![2, 4, 6, 8];
    let ch0_b1: Vec<i32> = vec![10, 12, 14, 16];
    let ch1_b0: Vec<i32> = vec![-1, -3, -5, -7];
    let ch1_b1: Vec<i32> = vec![-9, -11, -13, -15];
    let coefs: Vec<Vec<i64>> = vec![vec![1], vec![1]];
    let blocks = vec![
        vec![ch0_b0.clone(), ch0_b1.clone()],
        vec![ch1_b0.clone(), ch1_b1.clone()],
    ];
    let bytes = assemble_qlpc_stream(&header, &coefs, &blocks);

    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.channels.len(), 2);
    let mut ch0_expected = ch0_b0.clone();
    ch0_expected.extend(ch0_b1.iter());
    let mut ch1_expected = ch1_b0.clone();
    ch1_expected.extend(ch1_b1.iter());
    assert_eq!(dec.channels[0], ch0_expected);
    assert_eq!(dec.channels[1], ch1_expected);
}

#[test]
fn qlpc_carry_continuity_across_consecutive_blocks() {
    // Mono, two consecutive QLPC blocks (order 1, a1 = 1). The second
    // block's first residual depends on the first block's last sample
    // via the per-channel carry. Failure mode: a missing
    // update_after_block call would corrupt block 1's reconstruction.
    let header = synth_header(2, 5, 1, 4, 1, 0, 0);
    let block0: Vec<i32> = vec![1, 4, 9, 16];
    let block1: Vec<i32> = vec![25, 36, 49, 64];
    let coefs: Vec<Vec<i64>> = vec![vec![1]];
    let blocks = vec![vec![block0.clone(), block1.clone()]];
    let bytes = assemble_qlpc_stream(&header, &coefs, &blocks);

    let dec = decode_stream(&bytes).expect("decode");
    let mut expected = block0.clone();
    expected.extend(block1.iter());
    assert_eq!(dec.channels[0], expected);
}

#[test]
fn qlpc_with_verbatim_prefix_envelope_preserves_payload_and_samples() {
    // Header + VERBATIM prefix + QLPC sample block + QUIT. The verbatim
    // prefix should be recovered byte-for-byte, and the per-channel
    // sample stream should round-trip.
    let header = synth_header(2, 5, 1, 4, 2, 0, 16);
    let preamble = b"RIFF\x10\x00\x00\x00WAVEfmt ";
    assert_eq!(preamble.len(), 16);
    let samples: Vec<i32> = vec![2, 5, 9, 14];
    let coefs: Vec<i64> = vec![1, -1];

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);
    write_verbatim_block(&mut writer, preamble).expect("verbatim");
    let carry = ChannelCarry::new(3);
    let residuals = qlpc_residuals(&samples, &coefs, &carry).expect("scan");
    let energy = min_energy_for_qlpc(&residuals).expect("residuals fit");
    write_qlpc_block(&mut writer, energy, &coefs, &samples, &carry).expect("qlpc");
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
fn qlpc_silent_block_uses_minimum_energy_at_order_zero() {
    // All-zero samples → all-zero residuals at order 0 (predictor = 0).
    // min_energy_for_qlpc returns Some(0) — width 1.
    let samples: Vec<i32> = vec![0; 8];
    let carry = ChannelCarry::new(3);
    let residuals = qlpc_residuals(&samples, &[], &carry).expect("scan");
    assert_eq!(residuals, vec![0i64; 8]);
    assert_eq!(min_energy_for_qlpc(&residuals), Some(0));
}

#[test]
fn qlpc_three_channels_two_blocks_each_roundtrips() {
    // Stress test: 3 channels × 2 blocks each = 6 round-robin commands.
    // Each channel uses a different order-1 coefficient (1, 1, 0). The
    // round-robin emitter must thread per-channel carries across 6
    // emissions without crossing channels.
    let header = synth_header(2, 5, 3, 4, 1, 0, 0);
    let ch0 = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
    let ch1 = vec![vec![0, -1, -2, -3], vec![-4, -5, -6, -7]];
    let ch2 = vec![vec![1, 1, 1, 1], vec![1, 1, 1, 1]];
    let coefs: Vec<Vec<i64>> = vec![vec![1], vec![1], vec![0]];
    let blocks = vec![ch0.clone(), ch1.clone(), ch2.clone()];
    let bytes = assemble_qlpc_stream(&header, &coefs, &blocks);
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
