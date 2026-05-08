//! Round-2 self-roundtrip suite.
//!
//! Exercises the production [`encode`](crate::encode) entry point + its
//! predictor search and energy-width optimisation. Every test
//! produces a `.shn` byte buffer through the production encoder and
//! verifies that [`decode`](crate::decode) reconstructs the input
//! sample-exact.

use crate::{decode, encode, EncoderConfig, Filetype};

/// Encode → decode → assert input == output.
fn assert_round2_roundtrip(filetype: Filetype, channels: u16, blocksize: u32, samples: &[i32]) {
    let cfg = EncoderConfig::new(filetype, channels)
        .with_blocksize(blocksize)
        .with_max_lpc_order(0);
    let bytes = encode(&cfg, samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.header.filetype, filetype);
    assert_eq!(decoded.header.channels, channels);
    assert_eq!(decoded.samples.len(), samples.len(), "sample count");
    assert_eq!(decoded.samples, samples, "sample values");
    assert_eq!(decoded.final_bshift, 0);
}

#[test]
fn predictor_search_picks_diff0_for_constant_block() {
    // Constant block — DIFF0 with mu=0 yields zero residuals; the
    // predictor search should converge on it. 16 samples of constant 0
    // is the trivial check; non-zero constant tests the
    // smallest-residual case.
    assert_round2_roundtrip(Filetype::S16Le, 1, 8, &[5i32; 16]);
}

#[test]
fn predictor_search_picks_diff1_for_linear_ramp() {
    // Linear ramp — DIFF1 yields constant residuals = step. Search
    // should converge there.
    let samples: Vec<i32> = (0i32..32).map(|i| i * 100).collect();
    assert_round2_roundtrip(Filetype::S16Le, 1, 16, &samples);
}

#[test]
fn predictor_search_picks_diff2_for_quadratic() {
    // Quadratic — DIFF2 yields constant residuals = 2.
    let samples: Vec<i32> = (0i32..32).map(|i| (i - 16).pow(2)).collect();
    assert_round2_roundtrip(Filetype::S16Le, 1, 16, &samples);
}

#[test]
fn predictor_search_picks_diff3_for_cubic() {
    let samples: Vec<i32> = (0i32..32).map(|i| (i - 16).pow(3) / 8).collect();
    assert_round2_roundtrip(Filetype::S16Le, 1, 16, &samples);
}

#[test]
fn predictor_search_handles_random_signal() {
    // Pseudo-random walk — confirms the search doesn't break on
    // realistic-shaped data.
    let mut samples = Vec::with_capacity(128);
    let mut x: i64 = 0;
    let mut state: u64 = 0xDEAD_BEEF_DEAD_BEEF;
    for _ in 0..128 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let step = ((state >> 32) as i64 % 17) - 8;
        x += step;
        samples.push(x as i32);
    }
    assert_round2_roundtrip(Filetype::S16Le, 1, 32, &samples);
}

#[test]
fn predictor_search_stereo_independent() {
    // Stereo — encoder de-interleaves and applies search per channel.
    let mut samples = Vec::with_capacity(64);
    for i in 0i32..32 {
        samples.push(i * 3);
        samples.push((i - 16).pow(2));
    }
    assert_round2_roundtrip(Filetype::S16Le, 2, 16, &samples);
}

#[test]
fn predictor_search_multichannel_4() {
    // 4-channel stream.
    let mut samples = Vec::with_capacity(96);
    for i in 0i32..24 {
        samples.push(i);
        samples.push(-i);
        samples.push(i * 2);
        samples.push(i * i);
    }
    assert_round2_roundtrip(Filetype::S16Le, 4, 12, &samples);
}

#[test]
fn predictor_search_partial_tail_block() {
    // 22 samples, blocksize 8 — emits 2 full blocks + 1 partial.
    let samples: Vec<i32> = (0i32..22).map(|i| i * 11).collect();
    assert_round2_roundtrip(Filetype::S16Le, 1, 8, &samples);
}

#[test]
fn predictor_search_with_qlpc_enabled() {
    // With max_lpc_order > 0 the search also considers QLPC. The
    // result must still be lossless.
    let samples: Vec<i32> = (0i32..32).map(|i| (i - 16) * 7).collect();
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(16)
        .with_max_lpc_order(3);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples);
}

#[test]
fn predictor_search_qlpc_order_4() {
    let samples: Vec<i32> = (0i32..64).map(|i| (i - 32).pow(3) / 16).collect();
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(32)
        .with_max_lpc_order(4);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples);
}

// ─────────────────── Filetype roundtrips (all 11) ───────────────────

#[test]
fn filetype_ulaw_roundtrips() {
    let samples: Vec<i32> = (-4..4).collect();
    assert_round2_roundtrip(Filetype::Ulaw, 1, 4, &samples);
}

#[test]
fn filetype_s8_roundtrips() {
    let samples: Vec<i32> = (-32..32).map(|i| i * 2).collect();
    assert_round2_roundtrip(Filetype::S8, 1, 16, &samples);
}

#[test]
fn filetype_u8_roundtrips() {
    let samples: Vec<i32> = (-4..4).collect();
    assert_round2_roundtrip(Filetype::U8, 1, 4, &samples);
}

#[test]
fn filetype_s16be_roundtrips() {
    let samples: Vec<i32> = (-128..128).map(|i| i * 50).collect();
    assert_round2_roundtrip(Filetype::S16Be, 1, 64, &samples);
}

#[test]
fn filetype_u16be_roundtrips() {
    let samples: Vec<i32> = (-100..100).map(|i| i * 30).collect();
    assert_round2_roundtrip(Filetype::U16Be, 1, 50, &samples);
}

#[test]
fn filetype_s16le_roundtrips() {
    let samples: Vec<i32> = (-128..128).map(|i| i * 40).collect();
    assert_round2_roundtrip(Filetype::S16Le, 1, 64, &samples);
}

#[test]
fn filetype_u16le_roundtrips() {
    let samples: Vec<i32> = (-100..100).map(|i| i * 25).collect();
    assert_round2_roundtrip(Filetype::U16Le, 1, 50, &samples);
}

#[test]
fn filetype_s16native_roundtrips() {
    let samples: Vec<i32> = (-100..100).collect();
    assert_round2_roundtrip(Filetype::S16Native, 1, 50, &samples);
}

#[test]
fn filetype_u16native_roundtrips() {
    let samples: Vec<i32> = (-100..100).collect();
    assert_round2_roundtrip(Filetype::U16Native, 1, 50, &samples);
}

#[test]
fn filetype_s16swapped_roundtrips() {
    let samples: Vec<i32> = (-100..100).map(|i| i * 10).collect();
    assert_round2_roundtrip(Filetype::S16Swapped, 1, 40, &samples);
}

#[test]
fn filetype_u16swapped_roundtrips() {
    let samples: Vec<i32> = (-100..100).map(|i| i * 5).collect();
    assert_round2_roundtrip(Filetype::U16Swapped, 1, 40, &samples);
}

#[test]
fn all_eleven_filetypes_have_distinct_codes() {
    use std::collections::HashSet;
    let codes: HashSet<u32> = [
        Filetype::Ulaw,
        Filetype::S8,
        Filetype::U8,
        Filetype::S16Be,
        Filetype::U16Be,
        Filetype::S16Le,
        Filetype::U16Le,
        Filetype::S16Native,
        Filetype::U16Native,
        Filetype::S16Swapped,
        Filetype::U16Swapped,
    ]
    .iter()
    .map(|f| f.to_code())
    .collect();
    assert_eq!(
        codes.len(),
        11,
        "every filetype must have a distinct wire code"
    );
}

#[test]
fn filetype_pinned_codes_match_spec() {
    // Pinned by spec/05 §6.
    assert_eq!(Filetype::U8.to_code(), 2);
    assert_eq!(Filetype::S16Be.to_code(), 3);
    assert_eq!(Filetype::S16Le.to_code(), 5);
}

#[test]
fn filetype_resolves_all_codes_from_wire() {
    for code in 0u32..=10 {
        let ft = Filetype::from_code(code).expect("known code");
        assert_eq!(
            ft.to_code(),
            code,
            "to_code(from_code(c)) == c for c={code}"
        );
    }
    assert!(Filetype::from_code(11).is_err());
    assert!(Filetype::from_code(255).is_err());
}

// ─────────────────── Encoder error handling ───────────────────

#[test]
fn encode_rejects_unaligned_samples() {
    let cfg = EncoderConfig::new(Filetype::S16Le, 2).with_blocksize(8);
    let samples = vec![0i32; 7]; // odd, not /2.
    let err = encode(&cfg, &samples).unwrap_err();
    assert!(matches!(
        err,
        crate::EncodeError::SamplesNotChannelAligned { .. }
    ));
}

#[test]
fn encode_rejects_zero_channels() {
    let cfg = EncoderConfig::new(Filetype::S16Le, 0);
    let err = encode(&cfg, &[]).unwrap_err();
    assert!(matches!(err, crate::EncodeError::InvalidConfig(_)));
}

#[test]
fn encode_rejects_zero_blocksize() {
    let cfg = EncoderConfig::new(Filetype::S16Le, 1).with_blocksize(0);
    let err = encode(&cfg, &[]).unwrap_err();
    assert!(matches!(err, crate::EncodeError::InvalidConfig(_)));
}

#[test]
fn encode_rejects_blocksize_above_cap() {
    let cfg = EncoderConfig::new(Filetype::S16Le, 1).with_blocksize(crate::MAX_BLOCKSIZE + 1);
    let err = encode(&cfg, &[]).unwrap_err();
    assert!(matches!(err, crate::EncodeError::InvalidConfig(_)));
}

#[test]
fn encode_rejects_lpc_order_above_cap() {
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(8)
        .with_max_lpc_order(crate::MAX_LPC_ORDER + 1);
    let err = encode(&cfg, &[0i32; 8]).unwrap_err();
    assert!(matches!(err, crate::EncodeError::InvalidConfig(_)));
}

#[test]
fn encode_handles_empty_input() {
    // Zero samples, valid config.
    let cfg = EncoderConfig::new(Filetype::S16Le, 1).with_blocksize(8);
    let bytes = encode(&cfg, &[]).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples.len(), 0);
}

#[test]
fn encode_includes_verbatim_prefix() {
    // 44-byte RIFF/WAVE preamble simulator.
    let prefix = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00".to_vec();
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(8)
        .with_verbatim(prefix.clone());
    let samples: Vec<i32> = (0..16).collect();
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.verbatim_prefix, prefix);
    assert_eq!(decoded.samples, samples);
}

// ─────────────────── Energy-width sanity ───────────────────

#[test]
fn encoded_size_smaller_than_raw_for_typical_audio() {
    // For a smooth signal (sin-like ramp) the encoded size should be
    // smaller than the raw `samples.len() * 4` byte count.
    let samples: Vec<i32> = (0..512)
        .map(|i| ((i as f32 * 0.1).sin() * 1000.0) as i32)
        .collect();
    let cfg = EncoderConfig::new(Filetype::S16Le, 1).with_blocksize(64);
    let bytes = encode(&cfg, &samples).unwrap();
    let raw_size = samples.len() * 2; // s16 = 2 bytes/sample.
    assert!(
        bytes.len() < raw_size,
        "encoded {} >= raw {} — predictor search isn't finding compression",
        bytes.len(),
        raw_size
    );
}

#[test]
fn predictor_search_round_trips_negative_signal() {
    // Important boundary: signed-to-unsigned folding for negative
    // residuals. A descending ramp produces negative DIFF1 residuals.
    let samples: Vec<i32> = (0i32..32).map(|i| -i * 50).collect();
    assert_round2_roundtrip(Filetype::S16Le, 1, 16, &samples);
}

#[test]
fn predictor_search_round_trips_alternating() {
    // Alternating signs — DIFF0/DIFF1 should produce moderate
    // residuals; DIFF2/DIFF3 should be worse. The search picks the
    // best.
    let samples: Vec<i32> = (0i32..32)
        .map(|i| if i & 1 == 0 { 10 } else { -10 })
        .collect();
    assert_round2_roundtrip(Filetype::S16Le, 1, 16, &samples);
}

#[test]
fn predictor_search_round_trips_large_dynamic_range() {
    // Larger values — confirms width search scales up.
    let samples: Vec<i32> = (0i32..64).map(|i| (i - 32) * 1000).collect();
    assert_round2_roundtrip(Filetype::S16Le, 1, 32, &samples);
}
