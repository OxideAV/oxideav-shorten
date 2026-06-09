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
    decode_stream, select_predictor, select_predictor_with_qlpc, write_blocksize_command,
    write_byte_aligned_prefix, write_parameter_block, write_quit_command, write_selected_block,
    BitWriter, ChannelCarry, Choice, ShortenStreamHeader,
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
    // An arithmetic progression has sparse higher-order differences:
    //   first-differences  = [0, 3, 3, 3, ..., 3]    (constant after seed)
    //   second-differences = [0, 3, 0, 0, ..., 0]    (one-shot non-zero)
    // Round 251's natural-energy rule picked DIFF1 (constant non-zero
    // first-diff stream). Round 254's TR.156 §3.3 statistical optimum
    // beats it with DIFF2: a single non-zero second-difference outlier
    // sits inside an e=0 (width=1) svar mantissa and pays a small
    // prefix-zero count, while the other 31 samples cost only 2 bits
    // each. Verify the selector picks DIFF2 and the stream round-trips.
    let header = synth_header(2, 5, 1, 32, 0, 0, 0);
    let mut carry = ChannelCarry::new(3);
    let samples: Vec<i32> = (0..32i32).map(|t| 3 * t).collect();

    let choice = select_predictor(&samples, 0, &carry).expect("choice");
    // TR.156 §3.3 optimum picks DIFF2 on a sparse second-diff stream.
    assert!(
        matches!(choice, Choice::Diff2 { .. }),
        "TR.156 §3.3 optimum should pick DIFF2 on an arithmetic ramp, got {choice:?}"
    );

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

// ---- round 266: QLPC auto-selection integration ----

#[test]
fn mono_sequencer_picks_qlpc_when_candidate_beats_diff_family() {
    // Round 266 integration: a two-block stream that seeds the
    // per-channel sample-history carry on block 0 and then exploits
    // it on block 1 with a QLPC candidate whose coefficients model
    // the underlying second-order recurrence exactly. Block 1's QLPC
    // residuals collapse to all zero — the cheapest possible
    // residual stream — and the selector must pick Choice::Qlpc on
    // that block. The stream must round-trip byte-exact through
    // `decode_stream`.
    //
    // Block 0 (setup): samples [5, 10] under DIFF1 (small first-
    // diffs). After update_after_block the carry holds
    // carry.at(0) = 10, carry.at(1) = 5.
    //
    // Block 1 (QLPC): a Fibonacci-style stream s(t) = s(t-1) + s(t-2)
    // under coefs [1, 1]. The recurrence is genuinely order-2 and
    // does NOT collapse to a lower polynomial-difference order, so
    // the DIFFn family cannot match it; QLPC residuals are all zero,
    // a strict win for QLPC under the Rice-n metric.
    //
    // The stream uses H_blocksize = 2 to size block 0 then a
    // BLOCK_FN_BLOCKSIZE override to size the QLPC block. The
    // override does not advance the channel cursor per spec/03 §3.6.
    let qlpc_bs: u32 = 12;
    let header = synth_header(2, 5, 1, 2, 2, 0, 0);

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);

    let mut carry = ChannelCarry::new(3);

    // Block 0: setup — DIFF1 on [5, 10] seeds carry to [10, 5].
    let setup = vec![5i32, 10];
    let setup_choice = select_predictor(&setup, 0, &carry).expect("setup choice");
    write_selected_block(&mut writer, &setup_choice, &setup, 0, &carry).expect("setup write");
    carry.update_after_block(&setup);
    assert_eq!(carry.at(0), 10);
    assert_eq!(carry.at(1), 5);

    // BLOCKSIZE override → qlpc_bs samples per block from now on
    // (does NOT advance the channel cursor per spec/03 §3.6).
    write_blocksize_command(&mut writer, qlpc_bs).expect("blocksize override");

    // Block 1: QLPC — Fibonacci-style recurrence under coefs [1, 1]
    // with the seeded carry produces all-zero residuals.
    let mut ramp: Vec<i32> = Vec::with_capacity(qlpc_bs as usize);
    let mut a = 5i32;
    let mut b = 10i32;
    for _ in 0..qlpc_bs {
        let next = a + b;
        ramp.push(next);
        a = b;
        b = next;
    }
    let qlpc_coefs: Vec<i64> = vec![1, 1];
    let q_choice =
        select_predictor_with_qlpc(&ramp, 0, &carry, Some(&qlpc_coefs)).expect("qlpc choice");
    assert!(
        matches!(q_choice, Choice::Qlpc { .. }),
        "QLPC must beat DIFFn on a stream whose QLPC residuals are all zero, got {q_choice:?}"
    );
    write_selected_block(&mut writer, &q_choice, &ramp, 0, &carry).expect("qlpc write");
    carry.update_after_block(&ramp);

    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    let expected: Vec<i32> = setup.iter().chain(ramp.iter()).copied().collect();
    assert_eq!(dec.channels[0], expected);
}

#[test]
fn stereo_sequencer_with_qlpc_candidate_round_robin_round_trip() {
    // Stereo: each channel gets a QLPC-friendly block. The selector
    // picks per channel; the round-robin order of spec/03 §2 keeps
    // the per-channel state isolated. Both channels use a Fibonacci-
    // style generating recurrence under coefs [1, 1] so the QLPC
    // candidate produces all-zero residuals and the DIFFn family
    // (which doesn't match the order-2 non-polynomial recurrence)
    // cannot win.
    let header = synth_header(2, 5, 2, 8, 2, 0, 0);

    let mut carry0 = ChannelCarry::new(3);
    let mut carry1 = ChannelCarry::new(3);

    let setup0 = vec![5i32, 10, 15, 20, 25, 30, 35, 40];
    let setup1 = vec![3i32, 6, 9, 12, 15, 18, 21, 24];

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);

    // Channel 0 block 0: setup ramp (DIFF1-friendly).
    let c0a = select_predictor(&setup0, 0, &carry0).expect("c0a");
    write_selected_block(&mut writer, &c0a, &setup0, 0, &carry0).expect("w0a");
    carry0.update_after_block(&setup0);
    // Channel 1 block 0: setup ramp.
    let c1a = select_predictor(&setup1, 0, &carry1).expect("c1a");
    write_selected_block(&mut writer, &c1a, &setup1, 0, &carry1).expect("w1a");
    carry1.update_after_block(&setup1);

    // Channel 0 block 1: Fibonacci-style continuation. With the
    // seeded carry (carry.at(0)=40, at(1)=35) the QLPC coefs [1, 1]
    // produce all-zero residuals.
    let mut cont0: Vec<i32> = Vec::with_capacity(8);
    let mut a = carry0.at(1);
    let mut b = carry0.at(0);
    for _ in 0..8 {
        let next = a + b;
        cont0.push(next);
        a = b;
        b = next;
    }
    let qcoefs: Vec<i64> = vec![1, 1];
    let c0b = select_predictor_with_qlpc(&cont0, 0, &carry0, Some(&qcoefs)).expect("c0b");
    assert!(
        matches!(c0b, Choice::Qlpc { .. }),
        "ch0: QLPC must be selected, got {c0b:?}"
    );
    write_selected_block(&mut writer, &c0b, &cont0, 0, &carry0).expect("w0b");
    carry0.update_after_block(&cont0);

    // Channel 1 block 1: Fibonacci-style continuation.
    let mut cont1: Vec<i32> = Vec::with_capacity(8);
    let mut a = carry1.at(1);
    let mut b = carry1.at(0);
    for _ in 0..8 {
        let next = a + b;
        cont1.push(next);
        a = b;
        b = next;
    }
    let c1b = select_predictor_with_qlpc(&cont1, 0, &carry1, Some(&qcoefs)).expect("c1b");
    assert!(
        matches!(c1b, Choice::Qlpc { .. }),
        "ch1: QLPC must be selected, got {c1b:?}"
    );
    write_selected_block(&mut writer, &c1b, &cont1, 0, &carry1).expect("w1b");
    carry1.update_after_block(&cont1);

    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    let expected0: Vec<i32> = setup0.iter().chain(cont0.iter()).copied().collect();
    let expected1: Vec<i32> = setup1.iter().chain(cont1.iter()).copied().collect();
    assert_eq!(dec.channels[0], expected0);
    assert_eq!(dec.channels[1], expected1);
}

#[test]
fn sequencer_with_qlpc_emission_cost_matches_choice_bits() {
    // Round-251's bits_written == Choice::bits() invariant must
    // extend to Choice::Qlpc. Uses the Fibonacci-style construction
    // (coefs [1, 1] modelling s(t) = s(t-1) + s(t-2)) so the QLPC
    // candidate's all-zero residual stream beats the DIFFn family
    // (which doesn't match the order-2 non-polynomial recurrence).
    let mut carry = ChannelCarry::new(3);
    let setup = vec![5i32, 10];
    let setup_choice = select_predictor(&setup, 0, &carry).expect("setup");
    let mut writer = BitWriter::new();
    write_selected_block(&mut writer, &setup_choice, &setup, 0, &carry).expect("setup write");
    carry.update_after_block(&setup);
    // Generate Fibonacci-style stream against the seeded carry.
    let mut fib: Vec<i32> = Vec::with_capacity(16);
    let mut a = carry.at(1);
    let mut b = carry.at(0);
    for _ in 0..16 {
        let next = a + b;
        fib.push(next);
        a = b;
        b = next;
    }
    let qcoefs: Vec<i64> = vec![1, 1];
    let choice = select_predictor_with_qlpc(&fib, 0, &carry, Some(&qcoefs)).expect("qlpc choice");
    assert!(matches!(choice, Choice::Qlpc { .. }));

    let before = writer.bits_written();
    write_selected_block(&mut writer, &choice, &fib, 0, &carry).expect("qlpc write");
    let after = writer.bits_written();
    assert_eq!(after - before, choice.bits());
}
