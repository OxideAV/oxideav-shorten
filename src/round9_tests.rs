//! Round-9 tests — 64-bit-reservoir BitWriter on the encode side
//! plus the DIFF0 ↔ running-mean correctness fix.
//!
//! ## Round-9 deltas covered by this corpus
//!
//! ### Bit-writer reservoir
//!
//! Symmetric to round 6's decoder reservoir reader: the new
//! [`crate::bitwriter64::BitWriter64`] consumes multi-bit `write_bits`
//! / `write_uvar` calls into a `u64` accumulator and flushes whole
//! 8-byte `to_be_bytes` chunks, replacing the bit-at-a-time round-2
//! BitWriter on the production encode path. Cross-check tests in
//! `bitwriter64.rs` verify the two writers' byte streams are
//! byte-identical over a randomised matrix.
//!
//! ### DIFF0 ↔ running-mean correctness fix
//!
//! Master rounds 4–8 had a latent bug in
//! [`crate::encoder::best_predictor_with_mean`]: the round-4
//! mean-aware DIFF0 candidate (`block - mu_chan`) was *added* on top
//! of the round-3 baseline candidate set, but the round-3 search's
//! DIFF0 candidate produced residuals as `block - 0` regardless of
//! `mean_blocks`. If that round-3 DIFF0 won the bit-cost race the
//! encoder emitted DIFF0 with `block` residuals, while the decoder's
//! `s = r + mu_chan` on DIFF0 then offset every decoded sample by
//! `mu_chan`. The round-9 fix passes `mu_chan` down into
//! `best_predictor` so its DIFF0 candidate always subtracts the
//! at-block-start running mean — bit-exactly mirroring the decoder.
//!
//! The bug was not surfaced by the round-4 corpus because every
//! existing test either:
//! * had `mean_blocks == 0` (so `mu_chan = 0` and the two paths agree);
//! * tested `mean_blocks > 0` *alone* (where the mean-aware DIFF0 is
//!   the only DIFF0 the search ever evaluates, since the round-3
//!   identity DIFF0 ties); or
//! * tested `bshift > 0` lossy mode where the per-residual cost is
//!   tightly bounded by the shift, masking the cost-comparison
//!   pathology.
//!
//! `round9_diff0_mean_correctness_*` below pins the fix.

use crate::{decode, encode, EncoderConfig, Filetype};

// ───────────── BitWriter swap: byte-exact decode equality ─────────────

#[test]
fn round9_mono_s16_lossless_roundtrip_after_writer_swap() {
    let mut samples: Vec<i32> = Vec::with_capacity(4096);
    let mut state: u32 = 0xABCD_1234;
    for _ in 0..4096 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        samples.push((state >> 16) as i16 as i32 / 8);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 1).with_blocksize(256);
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples, samples);
}

#[test]
fn round9_stereo_s16_lossless_roundtrip_after_writer_swap() {
    let mut samples: Vec<i32> = Vec::with_capacity(8192);
    let mut state: u32 = 0x1234_ABCD;
    for _ in 0..8192 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        samples.push((state >> 16) as i16 as i32 / 8);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 2).with_blocksize(256);
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples, samples);
    assert_eq!(decoded.samples_per_channel, 4096);
}

#[test]
fn round9_mono_u8_roundtrip_after_writer_swap() {
    let samples: Vec<i32> = (0..1024).map(|i| (i & 0xFF) - 128).collect();
    let cfg = EncoderConfig::new(Filetype::U8, 1);
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples, samples);
}

#[test]
fn round9_lpc_order_4_roundtrip_after_writer_swap() {
    let mut samples: Vec<i32> = Vec::with_capacity(1024);
    let mut state: u32 = 0xFADE_C0DE;
    for _ in 0..1024 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        samples.push(((state >> 16) as i16 as i32) / 16);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(256)
        .with_max_lpc_order(4);
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples, samples);
}

#[test]
fn round9_bshift_lossy_roundtrip_after_writer_swap() {
    let mut samples: Vec<i32> = Vec::with_capacity(2048);
    let mut state: u32 = 0xCAFE_BABE;
    for _ in 0..2048 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        samples.push((state >> 16) as i16 as i32);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(256)
        .with_bshift(4);
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples.len(), samples.len());
    for (orig, recovered) in samples.iter().zip(decoded.samples.iter()) {
        let expected = (orig >> 4) << 4;
        assert_eq!(*recovered, expected);
    }
}

#[test]
fn round9_4kb_block_long_encode_roundtrip_after_writer_swap() {
    let bs = 4096;
    let n_blocks_per_channel = 8;
    let total = bs * 2 * n_blocks_per_channel;
    let mut samples: Vec<i32> = Vec::with_capacity(total);
    let mut state: u32 = 0xACE_FACE;
    for _ in 0..total {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        samples.push((state >> 16) as i16 as i32 / 4);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 2).with_blocksize(bs as u32);
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples, samples);
    assert_eq!(decoded.samples_per_channel, bs * n_blocks_per_channel);
}

#[test]
fn round9_verbatim_prefix_roundtrip_after_writer_swap() {
    let samples: Vec<i32> = (0..512).map(|i| (i & 0xFF) - 128).collect();
    let verbatim: Vec<u8> = b"RIFF\x24\x08\x00\x00WAVEfmt ".to_vec();
    let cfg = EncoderConfig::new(Filetype::U8, 1).with_verbatim(verbatim.clone());
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.verbatim_prefix, verbatim);
    assert_eq!(decoded.samples, samples);
}

// ───────────── DIFF0 ↔ running-mean correctness regression ─────────────

#[test]
fn round9_diff0_mean_correctness_mono_lpc3_blocksize128() {
    // The exact `(rng, blocksize, mean_blocks, lpc_order)` combination
    // that surfaced the round-4–round-8 DIFF0+mean bug: the encoder's
    // round-3 baseline DIFF0 (`block - 0`) wins the cost race against
    // the round-3 DIFF1..3 candidates at some block-1+ block, but the
    // decoder's `s = r + mu_chan` then offsets that block by the
    // running-mean estimator. The round-9 fix passes `mu_chan` down
    // into `best_predictor` so its DIFF0 always subtracts `mu_chan`.
    let mut samples: Vec<i32> = Vec::with_capacity(512);
    let mut state: u32 = 0x1357_9BDF;
    for _ in 0..512 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        samples.push(((state >> 16) as i16 as i32) / 4);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(128)
        .with_mean_blocks(4)
        .with_max_lpc_order(3);
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples, samples);
}

#[test]
fn round9_diff0_mean_correctness_stereo_lpc3_blocksize128() {
    let mut samples: Vec<i32> = Vec::with_capacity(2048);
    let mut state: u32 = 0x1357_9BDF;
    for _ in 0..2048 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        samples.push(((state >> 16) as i16 as i32) / 4);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 2)
        .with_blocksize(128)
        .with_mean_blocks(4)
        .with_max_lpc_order(3);
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples, samples);
}

#[test]
fn round9_diff0_mean_correctness_matrix() {
    // Sweep the (mean_blocks, max_lpc_order, blocksize, channels)
    // matrix that gates the DIFF0+mean path. Every entry must
    // round-trip bit-exactly.
    for &mean_blocks in &[1u32, 2, 4, 8, 16] {
        for &lpc_order in &[0u32, 2, 3, 4] {
            for &blocksize in &[64u32, 128, 256] {
                for &channels in &[1u16, 2] {
                    let total = (blocksize as usize) * 4 * (channels as usize);
                    let mut samples: Vec<i32> = Vec::with_capacity(total);
                    let mut state: u32 = 0xC0FE_FACE_u32
                        .wrapping_add(mean_blocks * 7)
                        .wrapping_add(lpc_order * 13)
                        .wrapping_add(blocksize * 17)
                        .wrapping_add(channels as u32 * 23);
                    for _ in 0..total {
                        state = state.wrapping_mul(1103515245).wrapping_add(12345);
                        samples.push(((state >> 16) as i16 as i32) / 4);
                    }
                    let cfg = EncoderConfig::new(Filetype::S16Le, channels)
                        .with_blocksize(blocksize)
                        .with_mean_blocks(mean_blocks)
                        .with_max_lpc_order(lpc_order);
                    let bytes = encode(&cfg, &samples).unwrap_or_else(|e| {
                        panic!("encode failed for mean={mean_blocks} lpc={lpc_order} bs={blocksize} ch={channels}: {e}")
                    });
                    let decoded = decode(&bytes).unwrap_or_else(|e| {
                        panic!("decode failed for mean={mean_blocks} lpc={lpc_order} bs={blocksize} ch={channels}: {e}")
                    });
                    assert_eq!(
                        decoded.samples, samples,
                        "DIFF0+mean roundtrip failed: mean_blocks={mean_blocks} lpc={lpc_order} blocksize={blocksize} channels={channels}"
                    );
                }
            }
        }
    }
}

#[test]
fn round9_diff0_mean_correctness_with_bshift() {
    // The fix also has to compose with `with_bshift` — the encoder
    // right-shifts samples before predictor application, and the
    // mean-aware DIFF0 then subtracts the running mean of the
    // shifted block.
    let mut samples: Vec<i32> = Vec::with_capacity(1024);
    let mut state: u32 = 0xBEEF_DEAD;
    for _ in 0..1024 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        samples.push((state >> 16) as i16 as i32);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(128)
        .with_mean_blocks(4)
        .with_max_lpc_order(3)
        .with_bshift(4);
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    let expected: Vec<i32> = samples.iter().map(|&s| (s >> 4) << 4).collect();
    assert_eq!(decoded.samples, expected);
}

// ───────────── Throughput / bench ─────────────

/// Print the best-of-5 wallclock for the round-9 stereo 4 KB-block
/// encode under the reservoir writer. Surfaces under `--nocapture` so
/// the round-8-vs-round-9 encode delta is reproducible.
#[test]
fn round9_stereo_long_block_encode_best_of_5_print() {
    use std::time::Instant;
    let bs = 4096;
    let n_blocks_per_channel = 64;
    let total = bs * 2 * n_blocks_per_channel;
    let mut samples: Vec<i32> = Vec::with_capacity(total);
    let mut state: u32 = 0xC0DE_C0DE;
    for _ in 0..total {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        samples.push(((state >> 16) as i16 as i32) / 4);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 2).with_blocksize(bs as u32);
    // Warmup.
    let _ = encode(&cfg, &samples).expect("encode");
    let mut best = std::time::Duration::from_secs(60);
    for _ in 0..5 {
        let t0 = Instant::now();
        let _ = encode(&cfg, &samples).expect("encode");
        let e = t0.elapsed();
        if e < best {
            best = e;
        }
    }
    println!(
        "round-9 stereo encode {} samples best-of-5: {:?}",
        total, best
    );
}

/// Throughput floor for the round-9 encode. The reservoir writer
/// should let a 64-block × 4096-sample × 2-channel encode finish well
/// under 3 s on contemporary hardware. The assertion is generous to
/// tolerate noisy CI shared runners; its purpose is regression-only.
#[test]
fn round9_stereo_long_block_encode_throughput_floor() {
    use std::time::Instant;
    let bs = 4096;
    let n_blocks_per_channel = 64;
    let total = bs * 2 * n_blocks_per_channel;
    let mut samples: Vec<i32> = Vec::with_capacity(total);
    let mut state: u32 = 0xFEED_FACE;
    for _ in 0..total {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        samples.push(((state >> 16) as i16 as i32) / 4);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 2).with_blocksize(bs as u32);
    let start = Instant::now();
    let bytes = encode(&cfg, &samples).expect("encode");
    let elapsed = start.elapsed();
    assert!(!bytes.is_empty());
    assert!(
        elapsed.as_millis() < 3000,
        "stereo encode of {} samples took {:?} (regression?)",
        total,
        elapsed
    );
}

/// A side-by-side benchmark of round-2 single-bit `BitWriter` against
/// the round-9 reservoir `BitWriter64` on the same residual-stream
/// workload. Prints both wallclocks for the round-9 commit body.
/// The reservoir writer must be at least 1.2× the round-2 writer's
/// speed on long residual blocks — if it isn't, the round-9 win has
/// regressed and the test fails.
#[test]
fn round9_writer_reservoir_vs_round2_bench() {
    use crate::bitwriter64::BitWriter64;
    use crate::encoder::BitWriter;
    use std::time::Instant;
    let n_residuals: u32 = 200_000;
    let width: u32 = 7;
    // Pre-compute a synthetic residual stream — narrow values to
    // exercise the mantissa-write path under a short prefix
    // (representative of the common encoder-emitted distribution).
    let mut state: u32 = 0xDEAD_BEEF;
    let mut values: Vec<u32> = Vec::with_capacity(n_residuals as usize);
    for _ in 0..n_residuals {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let v = state & ((1u32 << (width + 2)) - 1); // values up to 2^9-1
        values.push(v);
    }
    // Round-2 writer (single-bit accumulator).
    let mut best_r2 = std::time::Duration::from_secs(60);
    for _ in 0..5 {
        let t0 = Instant::now();
        let mut w = BitWriter::new();
        for &v in &values {
            w.write_uvar(v, width);
        }
        let bytes = w.finish();
        let e = t0.elapsed();
        if e < best_r2 {
            best_r2 = e;
        }
        std::hint::black_box(bytes.len());
    }
    // Round-9 reservoir writer.
    let mut best_r9 = std::time::Duration::from_secs(60);
    for _ in 0..5 {
        let t0 = Instant::now();
        let mut w = BitWriter64::with_capacity(n_residuals as usize);
        for &v in &values {
            w.write_uvar(v, width);
        }
        let bytes = w.finish();
        let e = t0.elapsed();
        if e < best_r9 {
            best_r9 = e;
        }
        std::hint::black_box(bytes.len());
    }
    println!(
        "round-9 writer bench: round-2 {:?} vs reservoir {:?} on {} uvar({}) codes",
        best_r2, best_r9, n_residuals, width
    );
    // Cross-check: both writers must produce identical bytes.
    let mut w2 = BitWriter::new();
    let mut w9 = BitWriter64::with_capacity(n_residuals as usize);
    for &v in &values {
        w2.write_uvar(v, width);
        w9.write_uvar(v, width);
    }
    assert_eq!(w2.finish(), w9.finish());
    // Regression assertion: reservoir must beat single-bit by ≥ 1.2×.
    let r2_ns = best_r2.as_nanos();
    let r9_ns = best_r9.as_nanos();
    assert!(
        r9_ns * 12 < r2_ns * 10,
        "round-9 reservoir writer ({:?}) is not at least 1.2× the round-2 BitWriter ({:?})",
        best_r9,
        best_r2
    );
}
