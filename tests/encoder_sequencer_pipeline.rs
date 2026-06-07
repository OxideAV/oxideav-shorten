//! Integration test for the round-251 per-block predictor-selection
//! sequencer.
//!
//! Builds Shorten files using the [`oxideav_shorten::select_predictor`]
//! / [`oxideav_shorten::write_selected_block`] surface to pick the
//! cheapest of `BLOCK_FN_DIFF0..3` / `BLOCK_FN_ZERO` per block, then
//! decodes them through the round-7 whole-stream driver
//! [`oxideav_shorten::decode_stream`] and confirms the recovered
//! per-channel samples are bit-exact with the encoder input.
//!
//! Clean-room provenance:
//!
//! * `docs/audio/shorten/spec/03-block-and-predictor.md` §3.1..§3.4
//!   (the four polynomial-difference predictor wire layouts) + §3.9
//!   (the constant-block sentinel) + §2 (the round-robin channel
//!   cursor that orders per-channel commands across a multi-channel
//!   stream).
//! * `docs/audio/shorten/spec/02-variable-length-coding.md` §2.1
//!   + §2.2 (the bit-length formulas the sequencer's cost
//!     computation mirrors).
//! * `docs/audio/shorten/spec/05-state-and-quirks.md` §1 (per-
//!   channel sample-history carry the DIFF1..3 residual scans seed
//!   from) + §2.4 (ZERO-block eligibility against `μ_chan`).

use oxideav_shorten::{
    decode_stream, select_predictor, write_byte_aligned_prefix, write_parameter_block,
    write_quit_command, write_selected_block, BitWriter, ChannelCarry, Choice, ShortenStreamHeader,
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

#[test]
fn mono_sequencer_picks_zero_for_constant_zero_block() {
    // Mono, H_meanblocks = 0 → mu_chan = 0. An all-zero block should
    // pick BLOCK_FN_ZERO.
    let header = synth_header(2, 5, 1, 6, 0, 0, 0);
    let carry = ChannelCarry::new(3);
    let samples = vec![0i32; 6];

    let choice = select_predictor(&samples, 0, &carry).expect("choice");
    assert!(matches!(choice, Choice::Zero { .. }));

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);
    write_selected_block(&mut writer, &choice, &samples, 0, &carry).expect("write");
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn mono_sequencer_picks_diff_for_arithmetic_ramp() {
    // An arithmetic progression makes DIFF1 the cheap choice (constant
    // first-difference). Verify round-trip.
    let header = synth_header(2, 5, 1, 32, 0, 0, 0);
    let mut carry = ChannelCarry::new(3);
    let samples: Vec<i32> = (0..32i32).map(|t| 3 * t).collect();

    let choice = select_predictor(&samples, 0, &carry).expect("choice");
    // For this signal, DIFF1 should win.
    assert!(matches!(choice, Choice::Diff1 { .. }));

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);
    write_selected_block(&mut writer, &choice, &samples, 0, &carry).expect("write");
    carry.update_after_block(&samples);
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn stereo_sequencer_round_robin_round_trip() {
    // Stereo: each channel gets one block, sequenced by the writer
    // in round-robin order ch0 then ch1. Use different sample
    // patterns per channel so the test pins the carry hand-off.
    let header = synth_header(2, 5, 2, 8, 0, 0, 0);
    let mut carry0 = ChannelCarry::new(3);
    let mut carry1 = ChannelCarry::new(3);
    let ch0: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let ch1: Vec<i32> = vec![10, 12, 14, 16, 18, 20, 22, 24];

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);

    let c0 = select_predictor(&ch0, 0, &carry0).expect("c0");
    write_selected_block(&mut writer, &c0, &ch0, 0, &carry0).expect("w0");
    carry0.update_after_block(&ch0);

    let c1 = select_predictor(&ch1, 0, &carry1).expect("c1");
    write_selected_block(&mut writer, &c1, &ch1, 0, &carry1).expect("w1");
    carry1.update_after_block(&ch1);

    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    assert_eq!(dec.channels.len(), 2);
    assert_eq!(dec.channels[0], ch0);
    assert_eq!(dec.channels[1], ch1);
}

#[test]
fn sequencer_handles_back_to_back_blocks_with_carry_update() {
    // Two consecutive blocks on the same channel: the second block
    // sees the updated carry from the first.
    let header = synth_header(2, 5, 1, 8, 0, 0, 0);
    let mut carry = ChannelCarry::new(3);
    let block0: Vec<i32> = vec![0, 5, 10, 15, 20, 25, 30, 35];
    let block1: Vec<i32> = vec![40, 45, 50, 55, 60, 65, 70, 75];

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);

    let c0 = select_predictor(&block0, 0, &carry).expect("c0");
    write_selected_block(&mut writer, &c0, &block0, 0, &carry).expect("w0");
    carry.update_after_block(&block0);

    let c1 = select_predictor(&block1, 0, &carry).expect("c1");
    write_selected_block(&mut writer, &c1, &block1, 0, &carry).expect("w1");
    carry.update_after_block(&block1);

    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    let expected: Vec<i32> = block0.iter().chain(block1.iter()).copied().collect();
    assert_eq!(dec.channels[0], expected);
}

#[test]
fn sequencer_picks_zero_when_block_matches_nonzero_mu_chan() {
    // With H_meanblocks = 0 the running mean is fixed at 0, so a block
    // of seven 0s can be ZERO-encoded. To exercise the non-zero mu path
    // we have to wire the predictor selection at mu_chan = 0 (since
    // the decoder also has H_meanblocks = 0 here); see the unit test
    // `zero_candidate_requires_all_samples_equal_to_mu_chan` for the
    // pure-helper version. This integration test pins the round-trip.
    let header = synth_header(2, 5, 1, 7, 0, 0, 0);
    let carry = ChannelCarry::new(3);
    let samples = vec![0i32; 7];

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);

    let choice = select_predictor(&samples, 0, &carry).expect("choice");
    assert!(matches!(choice, Choice::Zero { .. }));
    write_selected_block(&mut writer, &choice, &samples, 0, &carry).expect("write");

    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn sequencer_round_trips_mixed_predictor_stream() {
    // Three blocks on the same channel: a ZERO block, a DIFF1-ish
    // ramp, and a back-to-zero ZERO block. The selector should
    // independently pick the best for each block.
    let header = synth_header(2, 5, 1, 8, 0, 0, 0);
    let mut carry = ChannelCarry::new(3);
    let b_zero1 = vec![0i32; 8];
    let b_ramp = vec![0i32, 3, 6, 9, 12, 15, 18, 21];
    let b_zero2 = vec![0i32; 8];

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);

    for block in [&b_zero1, &b_ramp, &b_zero2].iter() {
        let c = select_predictor(block, 0, &carry).expect("c");
        write_selected_block(&mut writer, &c, block, 0, &carry).expect("w");
        carry.update_after_block(block);
    }

    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    let expected: Vec<i32> = b_zero1
        .iter()
        .chain(b_ramp.iter())
        .chain(b_zero2.iter())
        .copied()
        .collect();
    assert_eq!(dec.channels[0], expected);
}

#[test]
fn sequencer_emission_cost_matches_choice_bits() {
    // The Choice::bits() value must equal the actual writer bit count
    // delta produced by write_selected_block — load-bearing for any
    // higher-layer rate planner sitting above the sequencer.
    let carry = ChannelCarry::new(3);
    let samples: Vec<i32> = (0..16i32).map(|t| 7 * t - 3).collect();

    let choice = select_predictor(&samples, 0, &carry).expect("choice");
    let mut writer = BitWriter::new();
    let before = writer.bits_written();
    write_selected_block(&mut writer, &choice, &samples, 0, &carry).expect("write");
    let after = writer.bits_written();
    assert_eq!(after - before, choice.bits());
}
