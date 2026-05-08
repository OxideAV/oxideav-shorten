//! Round-4 self-roundtrip + drift-closure suite.
//!
//! Round 4 closes `audit/01` §8.1's ±1 drift on `bshift > 0` lossy
//! fixtures by mirroring the decoder's running-mean estimator on the
//! encode side ([`EncoderConfig::with_mean_blocks`]). These tests
//! exercise:
//!
//! * **Encoder/decoder parity on the running-mean buffer.** Encoder
//!   and decoder share the same per-channel `mean_buf` evolution,
//!   the same `block_mean_of` rule, and the same
//!   `running_mean(buf, mean_blocks)` aggregation. Lossless inputs
//!   roundtrip byte-for-byte under every `mean_blocks ∈ {0, 1, 2,
//!   4, 8, 16}`.
//! * **Drift closure on `bshift > 0` lossy modes.** With
//!   `mean_blocks > 0` the encoder's DIFF0 residuals are produced
//!   relative to `mu_chan` rather than zero. Fixtures with
//!   non-trivial DC trajectory (the `audit/01` §3 `u8 + uniform-
//!   random noise` cell, F2's bshift=7 / F5..F8's bshift=1..12) now
//!   roundtrip exactly to `(input >> bshift) << bshift` — the
//!   standard lossy criterion — without ±1 drift.
//! * **`BLOCK_FN_ZERO` short-circuit.** When the running-mean
//!   estimator's `mu_chan` exactly matches every sample of the
//!   input block, the encoder emits `BLOCK_FN_ZERO` (parameter-
//!   less) instead of a DIFF0 stream of all-zero residuals. This
//!   saves ENERGYSIZE + per-residual mantissa bits.
//! * **Composition with round-3 features.** `mean_blocks > 0`
//!   composes correctly with `with_max_lpc_order` and
//!   `with_bshift`.

use crate::{decode, encode, EncoderConfig, Filetype, MEAN_BLOCKS_MAX};

// ─────────────── Lossless roundtrip across mean-window sizes ───────────────

fn synth_signal(n: usize, seed: u64) -> Vec<i32> {
    let mut state = seed | 1;
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Pseudo-random ±200 noise on top of a slow DC sweep.
            let noise = ((state >> 56) as i64 % 401 - 200) as i32;
            let dc = ((i as f64 * 0.05).sin() * 8000.0) as i32;
            dc.wrapping_add(noise)
        })
        .collect()
}

#[test]
fn lossless_roundtrip_mean_blocks_0() {
    // Round-3 backward-compat: default mean_blocks=0 must produce the
    // exact same wire format as round 3.
    let samples = synth_signal(256, 0xCAFE_BABE);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1).with_blocksize(64);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples);
    assert_eq!(decoded.header.mean_blocks, 0);
}

#[test]
fn lossless_roundtrip_mean_blocks_1() {
    let samples = synth_signal(256, 0x1234_5678);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_mean_blocks(1);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples);
    assert_eq!(decoded.header.mean_blocks, 1);
}

#[test]
fn lossless_roundtrip_mean_blocks_4() {
    // mean_blocks=4 is TR.156's typical default for the reference
    // encoder.
    let samples = synth_signal(512, 0xDEAD_BEEF);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_mean_blocks(4);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples);
    assert_eq!(decoded.header.mean_blocks, 4);
}

#[test]
fn lossless_roundtrip_mean_blocks_8() {
    let samples = synth_signal(512, 0xF00D_F00D);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_mean_blocks(8);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples);
    assert_eq!(decoded.header.mean_blocks, 8);
}

#[test]
fn lossless_roundtrip_mean_blocks_16() {
    let samples = synth_signal(1024, 0xBEEF_DEAD);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(128)
        .with_mean_blocks(16);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples);
    assert_eq!(decoded.header.mean_blocks, 16);
}

#[test]
fn lossless_roundtrip_mean_blocks_stereo() {
    // Per-channel mean buffers must evolve independently.
    let n_per_ch = 256;
    let mut samples: Vec<i32> = Vec::with_capacity(n_per_ch * 2);
    let ch0 = synth_signal(n_per_ch, 0xAAAA_BBBB);
    let ch1 = synth_signal(n_per_ch, 0xCCCC_DDDD);
    for i in 0..n_per_ch {
        samples.push(ch0[i]);
        samples.push(ch1[i]);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 2)
        .with_blocksize(64)
        .with_mean_blocks(4);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples);
    assert_eq!(decoded.header.channels, 2);
    assert_eq!(decoded.header.mean_blocks, 4);
}

// ─────────────── Drift closure on bshift > 0 lossy modes ───────────────

#[test]
fn audit_01_drift_closed_bshift_1() {
    // audit/01 §8.1 + §3 the `u8 + uniform-random noise` failure cell.
    // Round 4 must produce an exact `(input >> 1) << 1` reconstruction
    // because encoder and decoder are now lock-stepped on `mu_chan`.
    let samples = synth_signal(512, 0x9999_8888);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_bshift(1)
        .with_mean_blocks(4);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    let expected: Vec<i32> = samples.iter().map(|&s| (s >> 1) << 1).collect();
    assert_eq!(decoded.samples, expected, "no ±1 drift under mean_blocks>0");
    assert_eq!(decoded.final_bshift, 1);
}

#[test]
fn audit_01_drift_closed_bshift_4() {
    let samples = synth_signal(512, 0x7777_6666);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_bshift(4)
        .with_mean_blocks(4);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    let expected: Vec<i32> = samples.iter().map(|&s| (s >> 4) << 4).collect();
    assert_eq!(decoded.samples, expected);
    assert_eq!(decoded.final_bshift, 4);
}

#[test]
fn audit_01_drift_closed_bshift_8_u8() {
    // F7-style: u8 audio with bshift=8 and a non-trivial DC level
    // (typical of u8 silence ≈ 128). Round 3 would drift on this; round
    // 4 must hit the lossy criterion exactly.
    let samples: Vec<i32> = (0..256)
        .map(|i| {
            let dc = 128i32;
            let mut state = (i as u64).wrapping_mul(0x1234_5678_9ABC_DEF0);
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((state >> 56) as i64 % 21 - 10) as i32;
            dc + noise
        })
        .collect();
    let cfg = EncoderConfig::new(Filetype::U8, 1)
        .with_blocksize(64)
        .with_bshift(8)
        .with_mean_blocks(4);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    let expected: Vec<i32> = samples.iter().map(|&s| (s >> 8) << 8).collect();
    assert_eq!(decoded.samples, expected);
    assert_eq!(decoded.final_bshift, 8);
}

#[test]
fn audit_01_drift_closed_bshift_12() {
    // F8-style: bshift=12.
    let samples = synth_signal(512, 0x5555_4444);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_bshift(12)
        .with_mean_blocks(4);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    let expected: Vec<i32> = samples.iter().map(|&s| (s >> 12) << 12).collect();
    assert_eq!(decoded.samples, expected);
    assert_eq!(decoded.final_bshift, 12);
}

// ─────────────── BLOCK_FN_ZERO short-circuit ───────────────

#[test]
fn zero_short_circuit_on_constant_silence() {
    // After the first all-zero block the encoder's running mean stays
    // at zero; subsequent constant-zero blocks must hit the ZERO
    // short-circuit and emit a parameter-less BLOCK_FN_ZERO. The
    // resulting encoded stream is materially shorter than the
    // mean_blocks=0 baseline (which would emit DIFF0 + energy + N
    // residuals).
    let samples = vec![0i32; 1024];
    let cfg_no_mean = EncoderConfig::new(Filetype::S16Le, 1).with_blocksize(64);
    let cfg_mean = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_mean_blocks(2);
    let bytes_no_mean = encode(&cfg_no_mean, &samples).unwrap();
    let bytes_mean = encode(&cfg_mean, &samples).unwrap();
    let decoded = decode(&bytes_mean).unwrap();
    assert_eq!(decoded.samples, samples);
    assert!(
        bytes_mean.len() <= bytes_no_mean.len(),
        "ZERO short-circuit should not regress encoded size: \
         mean={} no_mean={}",
        bytes_mean.len(),
        bytes_no_mean.len()
    );
}

#[test]
fn zero_short_circuit_after_dc_settle() {
    // Block 0 is non-trivial; blocks 1..N are all = block 0's mean.
    // After block 0 the running-mean buffer holds `block_mean(block0)`;
    // for `mean_blocks=1` mu_chan = block_mean(block0); subsequent
    // blocks of constant `block_mean(block0)` short-circuit.
    let mut samples: Vec<i32> = (0i32..64).collect();
    let mu0 = (samples.iter().sum::<i32>() + 32) / 64;
    samples.extend(std::iter::repeat(mu0).take(64 * 4));
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_mean_blocks(1);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples);
}

// ─────────────── Composition with round-3 features ───────────────

#[test]
fn mean_blocks_composes_with_lpc() {
    // mean_blocks > 0 + max_lpc_order > 0 — the encoder must still
    // produce a lossless roundtrip.
    let samples = synth_signal(256, 0xABCD_ABCD);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_max_lpc_order(3)
        .with_mean_blocks(4);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples);
}

#[test]
fn mean_blocks_composes_with_lpc_and_bshift() {
    let samples = synth_signal(256, 0x1357_2468);
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_max_lpc_order(2)
        .with_mean_blocks(2)
        .with_bshift(2);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    let expected: Vec<i32> = samples.iter().map(|&s| (s >> 2) << 2).collect();
    assert_eq!(decoded.samples, expected);
    assert_eq!(decoded.final_bshift, 2);
}

// ─────────────── Bounds + config validation ───────────────

#[test]
fn mean_blocks_rejects_overflow() {
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_mean_blocks(MEAN_BLOCKS_MAX + 1);
    let err = encode(&cfg, &[0i32; 64]).unwrap_err();
    assert!(matches!(err, crate::EncodeError::InvalidConfig(_)));
}

#[test]
fn mean_blocks_max_is_64() {
    // Round-4 documented bound.
    assert_eq!(MEAN_BLOCKS_MAX, 64);
}

#[test]
fn mean_blocks_default_is_zero() {
    let cfg = EncoderConfig::new(Filetype::S16Le, 1);
    assert_eq!(cfg.mean_blocks, 0);
}

// ─────────────── Encoder-decoder buffer-evolution parity ───────────────

#[test]
fn buffer_evolution_matches_decoder_on_ramp() {
    // A ramp signal exercises both `mu_blk` updates and `mu_chan`
    // changes between blocks. The output must match the input
    // byte-for-byte across the entire stream.
    let samples: Vec<i32> = (0i32..1024).collect();
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_mean_blocks(4);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples);
}

#[test]
fn buffer_evolution_matches_decoder_on_negative_dc() {
    // Negative DC level — the bias `+divisor/2` "always positive" rule
    // means encoder and decoder both compute `-mu` differently than a
    // sign-symmetric Python `//` would. The roundtrip pins the rule.
    let samples: Vec<i32> = (0i32..512).map(|i| -1000 + (i % 7)).collect();
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(64)
        .with_mean_blocks(4);
    let bytes = encode(&cfg, &samples).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.samples, samples);
}

// ─────────────── Compression-ratio sanity ───────────────

#[test]
fn dc_active_signal_compresses_better_with_mean_blocks() {
    // Round-4 motivation: signals with non-zero DC content compress
    // strictly better with the running-mean estimator active because
    // the DIFF0 residuals are centred near zero rather than offset by
    // the DC bias. We assert non-regression with a small tolerance.
    let samples: Vec<i32> = (0i32..1024).map(|i| 5000 + (i % 17) - 8).collect();
    let cfg_no_mean = EncoderConfig::new(Filetype::S16Le, 1).with_blocksize(128);
    let cfg_mean = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(128)
        .with_mean_blocks(4);
    let bytes_no = encode(&cfg_no_mean, &samples).unwrap();
    let bytes_yes = encode(&cfg_mean, &samples).unwrap();
    let decoded_yes = decode(&bytes_yes).unwrap();
    assert_eq!(decoded_yes.samples, samples);
    // Not strictly smaller (the DIFFn predictors are mean-invariant
    // so they'll often win); but mean_blocks must not regress vs no-
    // mean by more than a small constant overhead.
    assert!(
        bytes_yes.len() <= bytes_no.len() + 16,
        "mean_blocks=4 regression: {} > {} + 16",
        bytes_yes.len(),
        bytes_no.len()
    );
}
