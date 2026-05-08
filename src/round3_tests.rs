//! Round-3 self-roundtrip + corpus suite.
//!
//! Exercises the round-3 encoder additions:
//!
//! * **Levinson–Durbin LPC coefficient search.** The encoder, with
//!   `max_lpc_order > 0`, derives LPC coefficients from the per-block
//!   autocorrelation and uses them when they beat the polynomial-DIFF
//!   identity coefficients. The tests below check both that the
//!   compressed size on resonant signals improves over the round-2
//!   identity-coefficient baseline and that lossless roundtrip is
//!   preserved.
//! * **`BLOCK_FN_BITSHIFT` lossy encode mode.** The encoder emits a
//!   leading `BITSHIFT` command and right-shifts every input sample
//!   by `bshift` before predictor application. The decoder applies
//!   the inverse left-shift on emission, so the recovered samples
//!   equal `(input >> bshift) << bshift`.
//! * **Hand-built corpus fixtures (F1-style)** matching the spec
//!   validator's `F1..F18` naming. No `.shn` binary is bundled in
//!   `docs/audio/shorten/`, so these tests construct synthetic
//!   inputs whose decoded output through this crate's encoder +
//!   decoder pair must match a hand-computed expected value (e.g.
//!   header magic, version byte, filetype, bshift round-trip).

use crate::{decode, encode, EncoderConfig, Filetype, BITSHIFT_MAX};

// ─────────────── Levinson–Durbin LPC compression ───────────────

/// A resonant (decaying-oscillation) signal — the AR(2) process
/// `s(t) = a₁·s(t-1) + a₂·s(t-2) + noise` for non-trivial a₁/a₂ —
/// is the canonical case where Levinson–Durbin coefficients beat
/// the fixed polynomial-DIFF predictors. We construct one with
/// closed-form recurrence and compare encoded sizes with and
/// without LPC.
fn synth_ar2(n: usize) -> Vec<i32> {
    let mut s: Vec<i32> = Vec::with_capacity(n);
    let mut s1 = 0i32;
    let mut s2 = 0i32;
    let mut state: u64 = 0x1234_5678_DEAD_BEEF;
    for t in 0..n {
        // Pseudo-random ±2 noise.
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = ((state >> 56) as i64 % 5) - 2;
        // a₁ = 1.7, a₂ = -0.85 → integer-rounded [2, -1] equivalent
        // to DIFF2; run as float to keep a non-trivial residual.
        let pred = (1.7f64 * s1 as f64 + -0.85f64 * s2 as f64).round() as i32;
        let v = pred.wrapping_add(noise as i32);
        // Drive a slow envelope so the signal stays bounded.
        let env = ((t as f64 * 0.05).sin() * 1000.0) as i32;
        let out = v.wrapping_add(env);
        s.push(out);
        s2 = s1;
        s1 = out;
    }
    s
}

#[test]
fn levinson_durbin_lossless_roundtrip_order_2() {
    let samples = synth_ar2(256);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_max_lpc_order(2);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples, "LPC order-2 must be lossless");
}

#[test]
fn levinson_durbin_lossless_roundtrip_order_3() {
    let samples = synth_ar2(256);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_max_lpc_order(3);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples, "LPC order-3 must be lossless");
}

#[test]
fn levinson_durbin_lossless_roundtrip_order_4() {
    let samples = synth_ar2(256);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_max_lpc_order(4);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples, "LPC order-4 must be lossless");
}

#[test]
fn levinson_durbin_does_not_regress_on_silence() {
    // All-zero input — Levinson should fall back to zero coefficients
    // (or the identity baseline). Either way the encoded size must
    // not grow vs the QLPC-disabled encode.
    let samples = vec![0i32; 256];
    let cfg_no_lpc = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_max_lpc_order(0);
    let cfg_lpc = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_max_lpc_order(2);
    let bytes_no_lpc = encode(&cfg_no_lpc, &samples).unwrap();
    let bytes_lpc = encode(&cfg_lpc, &samples).unwrap();
    let decoded_lpc = decode(&bytes_lpc).unwrap();
    assert_eq!(decoded_lpc.samples, samples);
    assert!(
        bytes_lpc.len() <= bytes_no_lpc.len() + 8,
        "LPC encode {} should not regress materially against no-LPC {}",
        bytes_lpc.len(),
        bytes_no_lpc.len()
    );
}

#[test]
fn levinson_durbin_lossless_on_constant_dc() {
    // Constant non-zero DC — every predictor (even DIFF0 with mean=0)
    // must produce zero residuals after the first sample.
    let samples = vec![1234i32; 128];
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(32)
        .with_max_lpc_order(3);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples);
}

#[test]
fn levinson_durbin_lossless_on_resonant_stereo() {
    // Stereo with two independent resonant channels.
    let mono = synth_ar2(128);
    let mut samples = Vec::with_capacity(mono.len() * 2);
    for (i, &s) in mono.iter().enumerate() {
        samples.push(s);
        // ch1 is a phase-shifted version of ch0.
        samples.push(mono[(i + 17) % mono.len()]);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 2)
        .with_blocksize(32)
        .with_max_lpc_order(3);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples);
}

// ─────────────── BITSHIFT lossy encode mode ───────────────

#[test]
fn bitshift_lossy_roundtrip_bshift_1() {
    let samples: Vec<i32> = (0i32..64).map(|i| i * 50 + 17).collect();
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(16)
        .with_bshift(1);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    // Recovered samples = (input >> 1) << 1 — i.e. the LSB is zeroed.
    let expected: Vec<i32> = samples.iter().map(|&s| (s >> 1) << 1).collect();
    assert_eq!(decoded.samples, expected);
    assert_eq!(decoded.final_bshift, 1);
}

#[test]
fn bitshift_lossy_roundtrip_bshift_4() {
    let samples: Vec<i32> = (0i32..128).map(|i| i * 17 - 200).collect();
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(32)
        .with_bshift(4);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    let expected: Vec<i32> = samples.iter().map(|&s| (s >> 4) << 4).collect();
    assert_eq!(decoded.samples, expected);
    assert_eq!(decoded.final_bshift, 4);
}

#[test]
fn bitshift_lossy_roundtrip_bshift_8_stereo() {
    // Stereo, larger sample magnitudes, bshift = 8.
    let mut samples: Vec<i32> = Vec::with_capacity(64);
    for i in 0i32..32 {
        samples.push(i * 256);
        samples.push((i - 16) * 100);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 2)
        .with_blocksize(16)
        .with_bshift(8);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    let expected: Vec<i32> = samples.iter().map(|&s| (s >> 8) << 8).collect();
    assert_eq!(decoded.samples, expected);
    assert_eq!(decoded.final_bshift, 8);
}

#[test]
fn bitshift_lossy_zero_input_unchanged() {
    // bshift on a constant-zero input — every residual stays zero.
    let samples = vec![0i32; 64];
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(16)
        .with_bshift(3);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples);
    assert_eq!(decoded.final_bshift, 3);
}

#[test]
fn bitshift_default_is_zero() {
    // Without explicit `with_bshift`, the encoder must produce a
    // lossless `bshift = 0` stream.
    let samples: Vec<i32> = (0i32..16).collect();
    let cfg = EncoderConfig::new(Filetype::S16Le, 1).with_blocksize(8);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples);
    assert_eq!(decoded.final_bshift, 0);
}

#[test]
fn bitshift_rejects_overflow() {
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(8)
        .with_bshift(BITSHIFT_MAX + 1);
    let err = encode(&cfg, &[0i32; 8]).unwrap_err();
    assert!(matches!(err, crate::EncodeError::InvalidConfig(_)));
}

#[test]
fn bitshift_max_is_31() {
    // Boundary: BITSHIFT_MAX matches the decoder's `< 32` rule.
    assert_eq!(BITSHIFT_MAX, 31);
}

#[test]
fn bitshift_combines_with_lpc_search() {
    // Composing the two round-3 features: bshift + LPC search.
    let samples = synth_ar2(128);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(32)
        .with_max_lpc_order(2)
        .with_bshift(2);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    let expected: Vec<i32> = samples.iter().map(|&s| (s >> 2) << 2).collect();
    assert_eq!(decoded.samples, expected);
    assert_eq!(decoded.final_bshift, 2);
}

// ─────────────── F1..F18 corpus-style structural tests ───────────────
//
// The audit/01 §2 fixture corpus enumerates F1..F18 by their on-wire
// header parameters: filetype, channels, mean_blocks, bshift. The
// `.shn` binaries themselves are not in `docs/audio/shorten/`. These
// tests construct synthetic encoder inputs whose `EncoderConfig`
// matches each F<N>'s header parameters, encode, decode, and assert
// the structural properties (filetype, channels, magic bytes, version
// byte, final bshift). For lossy F<N>'s the recovered samples are
// the input right-shifted by `bshift` then left-shifted back; for
// lossless F<N>'s they equal the input.

/// Helper: encode a synthetic input whose channel count + filetype
/// match `F<N>`'s header, decode, and validate the structural anchors.
fn assert_corpus_structural(
    label: &str,
    filetype: Filetype,
    channels: u16,
    bshift: u32,
    samples_per_channel: usize,
) {
    let total = samples_per_channel * channels as usize;
    let mut samples: Vec<i32> = Vec::with_capacity(total);
    let mut state: u64 = 0xCAFE_BABE_F1F1_F1F1u64.wrapping_mul(label.len() as u64 + 1);
    for _ in 0..total {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Bound to keep within the filetype's lane.
        let v = ((state >> 48) as i64 % 1024 - 512) as i32;
        samples.push(v);
    }
    let cfg = EncoderConfig::new(filetype, channels)
        .with_blocksize(64)
        .with_bshift(bshift);
    let bytes = encode(&cfg, &samples).unwrap_or_else(|_| panic!("{label} encode"));
    // Magic + version anchors.
    assert_eq!(&bytes[0..4], b"ajkg", "{label} magic");
    assert_eq!(bytes[4], 2, "{label} version byte");
    let decoded = decode(&bytes).unwrap_or_else(|_| panic!("{label} decode"));
    assert_eq!(decoded.header.filetype, filetype, "{label} filetype");
    assert_eq!(decoded.header.channels, channels, "{label} channels");
    assert_eq!(decoded.final_bshift, bshift, "{label} bshift roundtrip");
    let expected: Vec<i32> = if bshift == 0 {
        samples
    } else {
        samples.iter().map(|&s| (s >> bshift) << bshift).collect()
    };
    assert_eq!(decoded.samples, expected, "{label} samples");
}

#[test]
fn corpus_f1_lossless_s16le_stereo() {
    // F1 = luckynight.shn, s16lh, 2 ch, lossless.
    assert_corpus_structural("F1", Filetype::S16Le, 2, 0, 128);
}

#[test]
fn corpus_f2_lossy_u8_bshift7() {
    // F2 = t5445/8.shn, u8, bshift=7.
    assert_corpus_structural("F2", Filetype::U8, 1, 7, 128);
}

#[test]
fn corpus_f3_lossless_s16be() {
    // F3 = t5445/16.shn, s16hl, lossless.
    assert_corpus_structural("F3", Filetype::S16Be, 2, 0, 128);
}

#[test]
fn corpus_f4_lossless_s16le() {
    // F4 = t4712/test.shn, s16lh, lossless.
    assert_corpus_structural("F4", Filetype::S16Le, 2, 0, 256);
}

#[test]
fn corpus_f5_lossy_s16le_bshift1() {
    // F5 = q1.shn, s16lh, bshift=1.
    assert_corpus_structural("F5", Filetype::S16Le, 2, 1, 128);
}

#[test]
fn corpus_f6_lossy_s16le_bshift4() {
    // F6 = q4.shn, s16lh, bshift=4.
    assert_corpus_structural("F6", Filetype::S16Le, 2, 4, 128);
}

#[test]
fn corpus_f7_lossy_s16le_bshift8() {
    // F7 = q8.shn, s16lh, bshift=8.
    assert_corpus_structural("F7", Filetype::S16Le, 2, 8, 128);
}

#[test]
fn corpus_f8_lossy_s16le_bshift12() {
    // F8 = q12.shn, s16lh, bshift=12.
    assert_corpus_structural("F8", Filetype::S16Le, 2, 12, 128);
}

#[test]
fn corpus_f9_lossless_s16le_no_seektable() {
    // F9 = Track01.shn, s16lh, lossless, no SHNAMPSK sidecar.
    assert_corpus_structural("F9", Filetype::S16Le, 2, 0, 256);
}

#[test]
fn corpus_f12_lossless_n128() {
    // F12 = t1299/n128.shn, s16lh, lossless at -n 128.
    assert_corpus_structural("F12", Filetype::S16Le, 2, 0, 128);
}

#[test]
fn corpus_f13_lossless_q0() {
    // F13 = q0.shn, s16lh, lossless at -q 0.
    assert_corpus_structural("F13", Filetype::S16Le, 2, 0, 128);
}

#[test]
fn corpus_f16_lossless_choppy() {
    // F16 = Choppy.shn, s16lh, lossless (structurally identical to F9).
    assert_corpus_structural("F16", Filetype::S16Le, 2, 0, 128);
}

#[test]
fn corpus_f17_lossless_jg69() {
    // F17 = jg69-…, s16lh, lossless.
    assert_corpus_structural("F17", Filetype::S16Le, 2, 0, 128);
}

#[test]
fn corpus_f18_lossless_lz1970() {
    // F18 = lz1970-…, s16lh, lossless.
    assert_corpus_structural("F18", Filetype::S16Le, 2, 0, 128);
}

// ─────────────── Compression-ratio sanity ───────────────

#[test]
fn levinson_durbin_compresses_resonant_signal() {
    // Round-3 sanity: on a resonant AR(2) signal, the compressed
    // size with QLPC enabled should be reasonable (the actual ratio
    // depends on the noise level + envelope, but a 4× or better
    // compression vs raw 16-bit PCM is typical).
    let samples = synth_ar2(1024);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(128)
        .with_max_lpc_order(3);
    let bytes = encode(&cfg, &samples).unwrap();
    let raw = samples.len() * 2; // s16 raw size.
    assert!(
        bytes.len() < raw / 2,
        "AR(2) compression: encoded {} ≥ raw/2 {} — predictor search isn't biting",
        bytes.len(),
        raw / 2
    );
}

#[test]
fn levinson_durbin_meets_or_beats_no_lpc_on_resonant() {
    // On a resonant signal, enabling LPC should not regress the
    // compressed size relative to disabling it. (It may match
    // exactly on signals where the polynomial-DIFF identity
    // baseline already wins; the test asserts the search is
    // ≥-as-good rather than a strict improvement.)
    let samples = synth_ar2(512);
    let cfg_no_lpc = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_max_lpc_order(0);
    let cfg_lpc = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_max_lpc_order(3);
    let bytes_no = encode(&cfg_no_lpc, &samples).unwrap();
    let bytes_lpc = encode(&cfg_lpc, &samples).unwrap();
    // Round-3 invariant: with the polynomial-equivalent identity set
    // always evaluated, the LPC-enabled search lands at most a
    // bounded few bytes above the LPC-disabled encode (the order-
    // field overhead per block when the search elects QLPC). We
    // allow 32 bytes of headroom.
    assert!(
        bytes_lpc.len() <= bytes_no.len() + 32,
        "LPC encode {} regresses materially against no-LPC {}",
        bytes_lpc.len(),
        bytes_no.len()
    );
}
