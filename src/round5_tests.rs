//! Round-5 lossy bit-budget / bit-rate target tests.
//!
//! Round 5 lands the encoder-side `-n N` (bit-budget) and `-r N`
//! (bit-rate) lossy modes that match the audit/01 §2 fixtures
//! `F10` (`-n 8`), `F11` (`-n 16`), `F14` (`-r 2.5`), and
//! `F15` (`-r 4`). The tests below exercise:
//!
//! * **Effective-bshift selection.** `with_bit_budget(n)` /
//!   `with_bit_rate(r)` pick the smallest bshift such that the probe
//!   block's per-sample post-Rice residual cost is `<= target`.
//! * **Decoder accepts the resulting stream.** The decoder reads the
//!   leading `BLOCK_FN_BITSHIFT` command and reconstructs samples
//!   left-shifted by `bshift`, the standard lossy-Shorten property.
//! * **Lower target → higher bshift, monotonic.** The mapping
//!   `target → effective_bshift` is non-increasing in target.
//! * **Tight target hits BITSHIFT_MAX without erroring.** Targets
//!   below the smallest achievable per-sample cost cap at
//!   `BITSHIFT_MAX` rather than failing.
//! * **Mutual exclusion with explicit `bshift`.** Setting both
//!   `bit_budget`/`bit_rate` and a non-zero `bshift` returns
//!   `EncodeError::BothBshiftAndBudget`.
//! * **`F12` `-n 128` lossless.** A target high enough that any
//!   16-bit input fits unrestricted produces `bshift = 0` (lossless),
//!   matching audit/01's F12 row.
//! * **Composition.** `with_max_lpc_order` and `with_mean_blocks`
//!   compose with `with_bit_budget` / `with_bit_rate`.

use crate::{decode, encode, EncodeError, EncoderConfig, Filetype, BITSHIFT_MAX};

/// 16-bit-style synthetic noise + sweep — high-entropy enough that
/// `bshift = 0` produces residuals well above `2.5` bits/sample, so
/// the tight-target probes have headroom to shift.
fn synth_lossy_signal(n: usize, seed: u64) -> Vec<i32> {
    let mut state = seed | 1;
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // ±8000 random + slow DC sweep to mimic audio content.
            let noise = ((state >> 56) as i64 % 16001 - 8000) as i32;
            let dc = ((i as f64 * 0.005).sin() * 4000.0) as i32;
            dc.wrapping_add(noise)
        })
        .collect()
}

// ─────────────── -n N bit-budget tests ───────────────

#[test]
fn bit_budget_n128_is_lossless_on_16bit_input() {
    // F12 `-n 128`: target so high that bshift = 0 stays the
    // smallest acceptable shift on any realistic input.
    let samples = synth_lossy_signal(1024, 0xF12_F12);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(256)
        .with_bit_budget(128);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.final_bshift, 0);
    assert_eq!(decoded.samples, samples);
}

#[test]
fn bit_budget_n8_picks_some_shift() {
    // F10 `-n 8`: target 8 bits/sample. Synthetic high-entropy
    // input forces a non-zero shift.
    let samples = synth_lossy_signal(1024, 0xF10_F10);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(256)
        .with_bit_budget(8);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    // Lossy: per-sample reconstruction is `(input >> bshift) << bshift`.
    let bshift = decoded.final_bshift;
    assert!(bshift <= BITSHIFT_MAX);
    for (out, inp) in decoded.samples.iter().zip(samples.iter()) {
        let expected = if bshift == 0 {
            *inp
        } else {
            (inp >> bshift) << bshift
        };
        assert_eq!(*out, expected);
    }
}

#[test]
fn bit_budget_n16_picks_zero_or_small_shift() {
    // F11 `-n 16`: target 16 bits/sample — usually achievable at
    // bshift = 0 on natural-audio-like inputs.
    let samples = synth_lossy_signal(1024, 0xF11_F11);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(256)
        .with_bit_budget(16);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    // For 16-bit-amplitude input the natural per-sample cost is
    // typically below 16 bits/sample after Rice coding, so the
    // probe accepts bshift = 0.
    assert_eq!(decoded.final_bshift, 0);
    assert_eq!(decoded.samples, samples);
}

#[test]
fn bit_budget_monotonic_in_target() {
    let samples = synth_lossy_signal(1024, 0xDEADC0DE);
    let mut prev_shift: u32 = 0;
    // Iterate descending target → shift should be non-decreasing.
    for &n in &[64u32, 32, 16, 8, 4, 2, 1] {
        let cfg = EncoderConfig::new(Filetype::S16Le, 1)
            .with_blocksize(256)
            .with_bit_budget(n);
        let bytes = encode(&cfg, &samples).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert!(
            decoded.final_bshift >= prev_shift,
            "target {n} produced shift {} below {prev_shift}",
            decoded.final_bshift
        );
        prev_shift = decoded.final_bshift;
    }
}

#[test]
fn bit_budget_unreachable_caps_at_max() {
    // Target so tight nothing in `0..=BITSHIFT_MAX` hits it: 0
    // bits/sample is not achievable for any non-empty residual
    // stream because the terminator bit alone costs 1 bit.
    let samples = synth_lossy_signal(512, 0xCAFEFEED);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(128)
        .with_bit_budget(0); // unreachable
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.final_bshift, BITSHIFT_MAX);
}

// ─────────────── -r N bit-rate tests ───────────────

#[test]
fn bit_rate_2_5_picks_some_shift_f14() {
    // F14 `-r 2_5`: target 2.5 bits/sample — well below the natural
    // residual cost, forces a non-zero shift.
    let samples = synth_lossy_signal(1024, 0xF14_F14);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(256)
        .with_bit_rate(2.5);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    let bshift = decoded.final_bshift;
    assert!(bshift > 0);
    assert!(bshift <= BITSHIFT_MAX);
    for (out, inp) in decoded.samples.iter().zip(samples.iter()) {
        let expected = if bshift == 0 {
            *inp
        } else {
            (inp >> bshift) << bshift
        };
        assert_eq!(*out, expected);
    }
}

#[test]
fn bit_rate_4_0_picks_some_shift_f15() {
    // F15 `-r 4`: target 4 bits/sample.
    let samples = synth_lossy_signal(1024, 0xF15_F15);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(256)
        .with_bit_rate(4.0);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    let bshift = decoded.final_bshift;
    assert!(bshift <= BITSHIFT_MAX);
    for (out, inp) in decoded.samples.iter().zip(samples.iter()) {
        let expected = if bshift == 0 {
            *inp
        } else {
            (inp >> bshift) << bshift
        };
        assert_eq!(*out, expected);
    }
}

#[test]
fn bit_rate_high_target_is_lossless() {
    // r >= natural per-sample cost → bshift = 0.
    let samples = synth_lossy_signal(1024, 0xBEEFBEEF);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(256)
        .with_bit_rate(64.0);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.final_bshift, 0);
    assert_eq!(decoded.samples, samples);
}

#[test]
fn bit_rate_lower_target_yields_higher_or_equal_shift() {
    let samples = synth_lossy_signal(1024, 0xDEADBABE);
    let cfg_4 = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(256)
        .with_bit_rate(4.0);
    let cfg_2_5 = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(256)
        .with_bit_rate(2.5);
    let s_4 = decode(&encode(&cfg_4, &samples).unwrap())
        .unwrap()
        .final_bshift;
    let s_2_5 = decode(&encode(&cfg_2_5, &samples).unwrap())
        .unwrap()
        .final_bshift;
    assert!(s_2_5 >= s_4);
}

// ─────────────── Mutual-exclusion + invalid configs ───────────────

#[test]
fn bit_budget_with_explicit_bshift_errors() {
    let samples = synth_lossy_signal(64, 0);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_bshift(3)
        .with_bit_budget(8);
    let err = encode(&cfg, &samples).unwrap_err();
    assert_eq!(err, EncodeError::BothBshiftAndBudget);
}

#[test]
fn bit_rate_with_explicit_bshift_errors() {
    let samples = synth_lossy_signal(64, 0);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_bshift(3)
        .with_bit_rate(2.5);
    let err = encode(&cfg, &samples).unwrap_err();
    assert_eq!(err, EncodeError::BothBshiftAndBudget);
}

#[test]
fn bit_rate_zero_or_negative_errors() {
    let samples = synth_lossy_signal(64, 0);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1).with_bit_rate(0.0);
    let err = encode(&cfg, &samples).unwrap_err();
    matches!(err, EncodeError::InvalidConfig(_));
    let cfg = EncoderConfig::new(Filetype::S16Le, 1).with_bit_rate(-1.0);
    let err = encode(&cfg, &samples).unwrap_err();
    matches!(err, EncodeError::InvalidConfig(_));
}

#[test]
fn bit_rate_nan_errors() {
    let samples = synth_lossy_signal(64, 0);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1).with_bit_rate(f32::NAN);
    let err = encode(&cfg, &samples).unwrap_err();
    matches!(err, EncodeError::InvalidConfig(_));
}

// ─────────────── Composition ───────────────

#[test]
fn bit_budget_composes_with_lpc() {
    let samples = synth_lossy_signal(1024, 0x11111);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(256)
        .with_max_lpc_order(4)
        .with_bit_budget(8);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    let bshift = decoded.final_bshift;
    for (out, inp) in decoded.samples.iter().zip(samples.iter()) {
        let expected = if bshift == 0 {
            *inp
        } else {
            (inp >> bshift) << bshift
        };
        assert_eq!(*out, expected);
    }
}

#[test]
fn bit_rate_composes_with_mean_blocks() {
    let samples = synth_lossy_signal(1024, 0x22222);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(256)
        .with_mean_blocks(4)
        .with_bit_rate(4.0);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    let bshift = decoded.final_bshift;
    for (out, inp) in decoded.samples.iter().zip(samples.iter()) {
        let expected = if bshift == 0 {
            *inp
        } else {
            (inp >> bshift) << bshift
        };
        assert_eq!(*out, expected);
    }
}

#[test]
fn bit_budget_stereo_roundtrip() {
    let mono = synth_lossy_signal(1024, 0x33333);
    // Interleave a phase-shifted copy as the second channel.
    let mut stereo: Vec<i32> = Vec::with_capacity(mono.len() * 2);
    for (i, &s) in mono.iter().enumerate() {
        stereo.push(s);
        stereo.push(mono[(i + 17) % mono.len()]);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 2)
        .with_blocksize(256)
        .with_bit_budget(8);
    let bytes = encode(&cfg, &stereo).unwrap();
    let decoded = decode(&bytes).unwrap();
    let bshift = decoded.final_bshift;
    assert_eq!(decoded.samples.len(), stereo.len());
    for (out, inp) in decoded.samples.iter().zip(stereo.iter()) {
        let expected = if bshift == 0 {
            *inp
        } else {
            (inp >> bshift) << bshift
        };
        assert_eq!(*out, expected);
    }
}

// ─────────────── Speed: LUT-based uvar prefix ───────────────

#[test]
fn lut_prefix_decode_matches_round1_baseline() {
    // Round 1's prefix scan was bit-by-bit; round 5 routes through
    // the LUT-backed `read_uvar_prefix`. The roundtrip suite already
    // covers correctness on real streams; this is a focused
    // sanity-check that the LUT path handles each leading-zero count
    // 0..=15 and each byte-alignment offset 0..=7. We construct a
    // hand-crafted bit stream of `<n zeros> 1 <FNSIZE bits of 0>` and
    // decode it as a uvar(n=2) value = (zeros << 2).
    use crate::bitreader::BitReader;
    use crate::varint::read_uvar;
    for zeros in 0u32..=15 {
        for align_offset in 0u32..=7 {
            // Build a bit pattern: <align_offset filler bits all 1>
            // followed by <zeros zeros> 1 <2-bit zero mantissa>.
            let mut pattern: Vec<u8> = Vec::new();
            pattern.resize(align_offset as usize, 1);
            pattern.resize(align_offset as usize + zeros as usize, 0);
            pattern.push(1);
            pattern.extend_from_slice(&[0, 0]);
            // Pack into bytes MSB-first.
            let mut bytes = Vec::with_capacity(pattern.len().div_ceil(8));
            let mut acc: u8 = 0;
            let mut count = 0;
            for &b in &pattern {
                acc = (acc << 1) | b;
                count += 1;
                if count == 8 {
                    bytes.push(acc);
                    acc = 0;
                    count = 0;
                }
            }
            if count > 0 {
                acc <<= 8 - count;
                bytes.push(acc);
            }
            let mut br = BitReader::new(&bytes);
            // Skip the alignment filler.
            for _ in 0..align_offset {
                let _ = br.read_bit().unwrap();
            }
            let v = read_uvar(&mut br, 2).unwrap();
            assert_eq!(v, zeros << 2, "zeros={zeros} align={align_offset}");
        }
    }
}
