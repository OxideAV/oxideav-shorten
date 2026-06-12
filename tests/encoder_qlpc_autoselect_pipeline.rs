//! Integration test for the round-282 QLPC auto-selection in the
//! per-block predictor-selection sequencer.
//!
//! Builds Shorten files using the
//! [`oxideav_shorten::select_predictor_auto`] /
//! [`oxideav_shorten::write_selected_block`] surface — the sequencer
//! now derives and quantises the per-block LPC coefficient vector
//! itself (order search + least-squares derivation + integer
//! quantisation per `spec/03` §3.5, cost model including the
//! `uvar(LPCQSIZE)` order field and the `order × svar(LPCQUANT)`
//! coefficient-transmission overhead per `spec/02` §4.3 + §4.4) —
//! then decodes the streams through the round-7 whole-stream driver
//! [`oxideav_shorten::decode_stream`] and confirms the recovered
//! per-channel samples are bit-exact with the encoder input.
//!
//! Clean-room provenance:
//!
//! * `docs/audio/shorten/spec/03-block-and-predictor.md` §3.5 (the
//!   `BLOCK_FN_QLPC` wire layout + the integer coefficient domain +
//!   the `H_maxlpcorder` order bound) + §2 (round-robin channel
//!   cursor) + §3.11 (per-channel sample-history carry).
//! * `docs/audio/shorten/spec/02-variable-length-coding.md` §4.3
//!   (`LPCQSIZE = 2`, zero-anchored order search) + §4.4
//!   (`LPCQUANT = 2` coefficient-transmission cost) + §2.1 + §2.2
//!   (the bit-length formulas behind the cost model).
//! * `docs/audio/shorten/spec/04-function-code-resolution.md` §5
//!   (`BLOCK_FN_QLPC = 7`, validation-corrected to byte-exact
//!   cross-codec confirmation of the parameter-stream layout).
//! * `docs/audio/shorten/spec/05-state-and-quirks.md` §1 (carry
//!   seeding) + §2 (mean-estimator state; these streams pin
//!   `H_meanblocks = 0` so `μ_chan = 0` throughout).

use oxideav_shorten::{
    decode_stream, select_predictor, select_predictor_auto, write_byte_aligned_prefix,
    write_parameter_block, write_quit_command, write_selected_block, BitWriter, ChannelCarry,
    Choice, ShortenStreamHeader,
};

fn synth_header(channels: u32, blocksize: u32, maxlpcorder: u32) -> ShortenStreamHeader {
    ShortenStreamHeader {
        version: 2,
        filetype: 5,
        channels,
        blocksize,
        maxlpcorder,
        meanblocks: 0,
        skipbytes: 0,
    }
}

/// A bounded period-6 oscillation `s(t) = s(t − 1) − s(t − 2)` from
/// seeds `(a, b)` — an exact two-term integer recurrence outside the
/// polynomial-difference family, so the auto-derived `[1, −1]` QLPC
/// vector is the genuinely cheapest predictor once the carry warms up.
fn oscillation(a: i32, b: i32, len: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(len);
    let (mut s2, mut s1) = (a, b);
    for _ in 0..len {
        let next = s1 - s2;
        out.push(next);
        s2 = s1;
        s1 = next;
    }
    out
}

#[test]
fn mono_auto_qlpc_multi_block_roundtrip() {
    // Four 64-sample blocks of the period-6 oscillation. The first
    // block sees a cold (zero) carry — its two seed-jump residuals
    // are the only non-zero QLPC residuals — and the warm-carry
    // blocks should drive the residual stream to all-zero.
    let blocksize = 64u32;
    let header = synth_header(1, blocksize, 2);
    let signal = oscillation(100, 50, 4 * blocksize as usize);

    let mut carry = ChannelCarry::new(3); // max(3, H_maxlpcorder = 2)
    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);

    let mut qlpc_blocks = 0usize;
    for block in signal.chunks(blocksize as usize) {
        let choice = select_predictor_auto(block, 0, &carry, header.maxlpcorder).expect("choice");
        if matches!(choice, Choice::Qlpc { .. }) {
            qlpc_blocks += 1;
        }
        let before = writer.bits_written();
        write_selected_block(&mut writer, &choice, block, 0, &carry).expect("write");
        // Round-251 invariant extended through the auto path: the
        // writer's bit delta equals the Choice's reported cost.
        assert_eq!(writer.bits_written() - before, choice.bits());
        carry.update_after_block(block);
    }
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    assert!(
        qlpc_blocks >= 3,
        "warm-carry oscillation blocks must auto-select QLPC, got {qlpc_blocks}/4"
    );

    // Decode parity: the existing decode path must consume the
    // auto-selected QLPC blocks bit-exactly.
    let dec = decode_stream(&out).expect("decode");
    assert_eq!(dec.channels.len(), 1);
    assert_eq!(dec.channels[0], signal);
}

#[test]
fn stereo_auto_mixed_predictors_round_robin_roundtrip() {
    // ch0 carries the oscillation (QLPC territory); ch1 carries an
    // arithmetic ramp (DIFF2 territory — the auto-derived [2, −1]
    // vector is residual-identical to DIFF2, which wins on its lower
    // fixed overhead). Two blocks per channel, round-robin emission.
    let blocksize = 48u32;
    let header = synth_header(2, blocksize, 2);
    let ch0 = oscillation(80, 40, 2 * blocksize as usize);
    let ch1: Vec<i32> = (0..2 * blocksize as i32).map(|t| 7 * t).collect();

    let mut carry0 = ChannelCarry::new(3);
    let mut carry1 = ChannelCarry::new(3);
    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);

    let mut ch0_choices = Vec::new();
    let mut ch1_choices = Vec::new();
    for blk in 0..2 {
        let lo = blk * blocksize as usize;
        let hi = lo + blocksize as usize;
        let b0 = &ch0[lo..hi];
        let b1 = &ch1[lo..hi];
        let c0 = select_predictor_auto(b0, 0, &carry0, header.maxlpcorder).expect("ch0");
        write_selected_block(&mut writer, &c0, b0, 0, &carry0).expect("write ch0");
        carry0.update_after_block(b0);
        ch0_choices.push(c0);
        let c1 = select_predictor_auto(b1, 0, &carry1, header.maxlpcorder).expect("ch1");
        write_selected_block(&mut writer, &c1, b1, 0, &carry1).expect("write ch1");
        carry1.update_after_block(b1);
        ch1_choices.push(c1);
    }
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    // The warm-carry ch0 block must be QLPC with the derived [1, −1].
    match &ch0_choices[1] {
        Choice::Qlpc { coefs, .. } => assert_eq!(coefs, &vec![1, -1]),
        other => panic!("ch0 warm block should auto-select QLPC, got {other:?}"),
    }
    // The ramp channel must stay in the DIFFn family (overhead rule).
    for (i, c) in ch1_choices.iter().enumerate() {
        assert!(
            !matches!(c, Choice::Qlpc { .. }),
            "ch1 block {i} must not pay QLPC overhead, got {c:?}"
        );
    }

    let dec = decode_stream(&out).expect("decode");
    assert_eq!(dec.channels.len(), 2);
    assert_eq!(dec.channels[0], ch0);
    assert_eq!(dec.channels[1], ch1);
}

#[test]
fn auto_selection_never_costs_more_than_legacy_and_wins_on_lpc_material() {
    // Rate sanity on synthetic material: for every block the auto
    // selector's pick costs no more bits than the legacy
    // DIFF0..3/ZERO pick (the candidate set is a superset), and on
    // the oscillation signal the whole-stream saving is strict.
    let blocksize = 64usize;
    let signal = oscillation(100, 50, 8 * blocksize);

    let mut legacy_bits = 0u64;
    let mut auto_bits = 0u64;
    let mut carry_l = ChannelCarry::new(3);
    let mut carry_a = ChannelCarry::new(3);
    for block in signal.chunks(blocksize) {
        let legacy = select_predictor(block, 0, &carry_l).expect("legacy");
        legacy_bits += legacy.bits();
        carry_l.update_after_block(block);
        let auto = select_predictor_auto(block, 0, &carry_a, 2).expect("auto");
        assert!(
            auto.bits() <= legacy.bits(),
            "auto pick ({} bits) regressed past legacy ({} bits)",
            auto.bits(),
            legacy.bits()
        );
        auto_bits += auto.bits();
        carry_a.update_after_block(block);
    }
    assert!(
        auto_bits < legacy_bits,
        "QLPC auto-selection must strictly beat DIFFn on the oscillation \
         ({auto_bits} vs {legacy_bits} bits)"
    );
}

#[test]
fn rice_optimum_respects_decoder_prefix_cap_regression() {
    // Regression for a latent round-254 encoder/decoder mismatch the
    // round-282 QLPC work surfaced: a sparse residual stream with one
    // large seed-jump outlier (here a constant-50 block over a cold
    // zero carry → DIFF1 residuals [50, 0, 0, …]) used to drive the
    // Rice-n sweep to e = 0, whose single outlier code needs 50
    // prefix-zero bits — beyond the decoder's 32-zero `uvar` prefix
    // cap, so the emitted stream failed to decode. The cost model now
    // treats over-cap codes as unrepresentable and the sweep settles
    // on a decodable energy. Pin the full encode → decode roundtrip.
    let blocksize = 256u32;
    let header = synth_header(1, blocksize, 0);
    let signal = vec![50i32; blocksize as usize];

    let carry = ChannelCarry::new(3);
    let choice = select_predictor(&signal, 0, &carry).expect("choice");

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);
    write_selected_block(&mut writer, &choice, &signal, 0, &carry).expect("write");
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode must accept the selected energy");
    assert_eq!(dec.channels[0], signal);
}

#[test]
fn max_lpc_order_zero_stream_reproduces_legacy_bytes() {
    // A H_maxlpcorder = 0 stream is polynomial-difference-only per
    // spec/03 §3.5; the auto path with max 0 must produce a
    // byte-identical stream to the legacy selector.
    let blocksize = 32u32;
    let header = synth_header(1, blocksize, 0);
    let signal: Vec<i32> = (0..(2 * blocksize as i32))
        .map(|t| (t * t) % 97 - 48)
        .collect();

    let build = |auto: bool| -> Vec<u8> {
        let mut carry = ChannelCarry::new(3);
        let mut out = Vec::new();
        write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
        let mut writer = BitWriter::new();
        write_parameter_block(&mut writer, &header);
        for block in signal.chunks(blocksize as usize) {
            let choice = if auto {
                select_predictor_auto(block, 0, &carry, 0).expect("auto")
            } else {
                select_predictor(block, 0, &carry).expect("legacy")
            };
            write_selected_block(&mut writer, &choice, block, 0, &carry).expect("write");
            carry.update_after_block(block);
        }
        write_quit_command(&mut writer);
        writer.pad_to_byte();
        out.extend(writer.into_bytes());
        out
    };

    let legacy_stream = build(false);
    let auto_stream = build(true);
    assert_eq!(legacy_stream, auto_stream, "max 0 must be byte-identical");
    let dec = decode_stream(&auto_stream).expect("decode");
    assert_eq!(dec.channels[0], signal);
}
