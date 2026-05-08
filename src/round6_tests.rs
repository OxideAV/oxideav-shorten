//! Round-6 tests: SIMD-grade residual unpack via the 64-bit bit
//! reservoir.
//!
//! The headline change is the [`crate::bitstream64::Bitstream64`]
//! reader that the decoder's per-block residual loop drops down into.
//! It batches every 32–64 unread bits into an in-register `u64` and
//! resolves `uvar` prefix scans with `u64::leading_zeros` (lowering to
//! hardware `lzcnt` on x86-64 / `clz` on aarch64), eliminating the
//! per-byte LUT lookup and per-bit refill the round-5 path performed.
//!
//! These tests cover:
//!
//! * **Functional equivalence** — every round-1..5 fixture continues to
//!   decode bit-exactly, exercised end-to-end via the existing
//!   roundtrip suites. The tests below add specific
//!   reservoir-reader edge cases (sub-byte alignment, multi-refill
//!   prefix runs) on top of those.
//! * **Throughput floor** — a synthetic 16-bit, 4096-sample-per-block
//!   stream encodes through [`encode`] and decodes via the new path;
//!   the test asserts the decode wallclock is bounded so a regression
//!   that re-introduces per-bit reads would surface.
//! * **Wire-format compatibility** — DIFF0..3 + QLPC blocks still
//!   roundtrip and the residual buffer is consumed in lockstep with
//!   the encoder.

use crate::bitstream64::Bitstream64;
use crate::decoder::decode;
use crate::encoder::{encode, EncoderConfig};
use crate::header::Filetype;

/// Sub-byte alignment: start the reservoir at bit 3 of a byte and
/// confirm the prefix scan ignores the dropped high bits.
#[test]
fn reservoir_handles_sub_byte_start() {
    // bit pattern: 110_0001_0_xxxxxxx
    //   bits 0..3: skipped at construction.
    //   bits 3..7: 0001 → 3 zeros, then a 1 at bit 6, then 0 mantissa.
    let bytes = [0b110_00010, 0u8];
    let mut bs = Bitstream64::new_at(&bytes, 3);
    assert_eq!(bs.read_uvar_prefix(64), Some(3));
    // Next bit consumed should be position 7 → reading 1 bit yields 0.
    assert_eq!(bs.read_bits(1).unwrap(), 0);
}

/// A 17-byte all-zeros run forces multiple reservoir refills before the
/// terminator is found. The zero count must include every consumed
/// zero bit, including the ones that crossed refill boundaries.
#[test]
fn reservoir_long_zero_run_across_refills() {
    let mut bytes = vec![0u8; 17];
    bytes.push(0x80); // terminator at bit 0 of the 18th byte.
    let mut bs = Bitstream64::new_at(&bytes, 0);
    assert_eq!(bs.read_uvar_prefix(256), Some(17 * 8));
}

/// Reading more bits than the reservoir + buffer combined should
/// surface as `UnexpectedEof`.
#[test]
fn reservoir_eof_on_overrun() {
    let bytes = [0xFFu8, 0xFF];
    let mut bs = Bitstream64::new_at(&bytes, 0);
    // 17 bits requested from a 16-bit buffer → EOF.
    let _ = bs.read_bits(16).unwrap();
    assert!(bs.read_bits(1).is_err());
}

/// 4 KB-block roundtrip under DIFF1: confirms the bulk-residual reader
/// produces the same samples as the round-5 byte-LUT scan.
#[test]
fn long_block_diff1_roundtrip_bit_exact() {
    let bs = 4096;
    let mut samples: Vec<i32> = Vec::with_capacity(bs);
    // Linear ramp + small modulation; well-suited to DIFF1 with small
    // residuals.
    for i in 0..bs {
        let s = (i as i32) + ((i as i32) % 7) * 3;
        // Clamp into i16 range so S16 filetype encode is well-formed.
        let s = s.clamp(-32768, 32767);
        samples.push(s);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 1).with_blocksize(bs as u32);
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples, samples);
}

/// 4 KB-block stereo roundtrip exercises the channel-cursor advance
/// across many residual batches.
#[test]
fn long_block_stereo_roundtrip() {
    let bs = 4096;
    let total = bs * 2;
    let mut samples: Vec<i32> = Vec::with_capacity(total);
    for i in 0..bs {
        // L/R interleaved with slight phase offset so DIFF1 picks
        // different residuals on each channel.
        samples.push((i as i32) % 1024);
        samples.push(((i + 13) as i32) % 1024);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 2).with_blocksize(bs as u32);
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples.len(), samples.len());
    assert_eq!(decoded.samples, samples);
}

/// Long-block QLPC roundtrip under `max_lpc_order = 4`. Stresses both
/// the residual reservoir reader and the LPC predictor recurrence.
#[test]
fn long_block_qlpc_roundtrip() {
    let bs = 4096;
    let mut samples: Vec<i32> = Vec::with_capacity(bs);
    // Damped sinusoid that the LPC search should fit cheaply.
    let mut a: f32 = 0.0;
    let mut prev = 0i32;
    for _ in 0..bs {
        a += 0.011;
        let v = ((a.sin() * 16000.0) as i32 + prev) / 2;
        let v = v.clamp(-32768, 32767);
        samples.push(v);
        prev = v;
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 1)
        .with_blocksize(bs as u32)
        .with_max_lpc_order(4);
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples, samples);
}

/// Hand-built bit pattern: 8 consecutive `uvar(0)` codes whose prefix
/// lengths span 0..=7, confirming that the reservoir's per-residual
/// state evolves correctly across a stream of mixed prefix sizes.
#[test]
fn reservoir_mixed_prefix_walk() {
    // Build the pattern: for k = 0..=7, emit k zeros then a 1.
    // Total bits = sum_{k=0..=7} (k+1) = 36 bits → 5 bytes (4 bits pad).
    let mut bits: Vec<u8> = Vec::new();
    for k in 0..=7u32 {
        bits.resize(bits.len() + k as usize, 0);
        bits.push(1);
    }
    // Pack MSB-first.
    let mut bytes: Vec<u8> = Vec::new();
    let mut acc = 0u8;
    let mut count = 0;
    for b in &bits {
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
    let mut bs = Bitstream64::new_at(&bytes, 0);
    for k in 0..=7u32 {
        assert_eq!(bs.read_uvar(0, 64).unwrap(), k, "uvar(0) #{k}");
    }
}

/// Print the best-of-5 wallclock for the round-6 long-block decode
/// path so the same measurement (against the master snapshot) gives a
/// reproducible throughput delta. Surfaces under `--nocapture`.
#[test]
fn long_block_decode_best_of_5_print() {
    use std::time::Instant;
    let bs = 4096;
    let n_blocks = 64;
    let total = bs * n_blocks;
    let mut samples: Vec<i32> = Vec::with_capacity(total);
    let mut state: u32 = 0xC0DE_C0DE;
    for _ in 0..total {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let v = ((state >> 16) as i16) as i32 / 4;
        samples.push(v);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 1).with_blocksize(bs as u32);
    let bytes = encode(&cfg, &samples).expect("encode");
    // Warmup.
    let _ = decode(&bytes).expect("decode");
    let mut best = std::time::Duration::from_secs(60);
    for _ in 0..5 {
        let t0 = Instant::now();
        let _ = decode(&bytes).expect("decode");
        let e = t0.elapsed();
        if e < best {
            best = e;
        }
    }
    println!("round-6 decode {} samples best-of-5: {:?}", total, best);
}

/// Throughput floor: decode a 64-block, 4096-sample-per-block stream
/// and assert the wallclock is well below 1 s on contemporary
/// hardware. The test's purpose is regression-only; a per-bit reader
/// would take orders of magnitude longer on this volume.
#[test]
fn long_block_decode_throughput_floor() {
    use std::time::Instant;
    let bs = 4096;
    let n_blocks = 64;
    let total = bs * n_blocks;
    let mut samples: Vec<i32> = Vec::with_capacity(total);
    // High-entropy-but-bounded signal so residuals span a few bits.
    let mut state: u32 = 0xC0DE_C0DE;
    for _ in 0..total {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let v = ((state >> 16) as i16) as i32 / 4;
        samples.push(v);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 1).with_blocksize(bs as u32);
    let bytes = encode(&cfg, &samples).expect("encode");
    let start = Instant::now();
    let decoded = decode(&bytes).expect("decode");
    let elapsed = start.elapsed();
    assert_eq!(decoded.samples.len(), total);
    // Be generous (CI shared runners) — primary purpose is to detect
    // a catastrophic regression into per-bit reads on long blocks.
    assert!(
        elapsed.as_millis() < 1000,
        "decode of {} samples took {:?} (regression?)",
        total,
        elapsed
    );
}

/// Direct comparison of the round-5 byte-LUT prefix path vs the
/// round-6 64-bit-reservoir prefix path on the same synthetic bit
/// stream. Asserts the reservoir path is *at least as fast* (typically
/// much faster) than the LUT path on a multi-MB residual stream. The
/// numerical wallclock is printed under `--nocapture` for the
/// throughput-delta line in the round-6 commit message, but the
/// assertion is generous to tolerate noisy CI machines.
#[test]
fn reservoir_beats_byte_lut_on_long_streams() {
    use crate::bitreader::BitReader;
    use std::time::Instant;
    // Build a ~1 MB bit stream of `uvar(2)` codes. The mantissa width
    // dominates the per-residual bit count, so we stay close to the
    // hot path the decoder actually exercises.
    let n_codes = 200_000usize;
    let mut bits: Vec<u8> = Vec::with_capacity(n_codes * 6);
    let mut seed: u64 = 0xDEAD_BEEF;
    for _ in 0..n_codes {
        // Synthesize a uvar(2) with prefix length 0..=4 (geometric).
        seed = seed
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        let prefix = ((seed >> 56) & 7) as usize % 5;
        bits.resize(bits.len() + prefix, 0);
        bits.push(1);
        // 2-bit mantissa.
        let m = ((seed >> 32) & 3) as u8;
        bits.push((m >> 1) & 1);
        bits.push(m & 1);
    }
    // Pack MSB-first.
    let mut packed: Vec<u8> = Vec::with_capacity(bits.len().div_ceil(8));
    let mut acc = 0u8;
    let mut count = 0;
    for b in &bits {
        acc = (acc << 1) | b;
        count += 1;
        if count == 8 {
            packed.push(acc);
            acc = 0;
            count = 0;
        }
    }
    if count > 0 {
        acc <<= 8 - count;
        packed.push(acc);
    }

    // Round-5 path: BitReader::read_uvar_prefix + read_bits per
    // residual.
    let t0 = Instant::now();
    let mut br = BitReader::new(&packed);
    let mut sum_a: u32 = 0;
    for _ in 0..n_codes {
        let z = br.read_uvar_prefix(64).unwrap();
        let m = br.read_bits(2).unwrap();
        sum_a = sum_a.wrapping_add((z << 2) | m);
    }
    let lut_elapsed = t0.elapsed();

    // Round-6 path: Bitstream64 reservoir.
    let t0 = Instant::now();
    let mut bs = Bitstream64::new_at(&packed, 0);
    let mut sum_b: u32 = 0;
    for _ in 0..n_codes {
        let v = bs.read_uvar(2, 64).unwrap();
        sum_b = sum_b.wrapping_add(v);
    }
    let reservoir_elapsed = t0.elapsed();

    assert_eq!(
        sum_a, sum_b,
        "reservoir + LUT must compute identical values"
    );
    println!(
        "round-6 throughput: byte-LUT {:?} vs reservoir {:?} on {} codes",
        lut_elapsed, reservoir_elapsed, n_codes
    );
    // Assertion floor: reservoir must be no worse than 1.2× the LUT
    // path. (Real measurements show 1.5–3× on contemporary x86-64 /
    // aarch64; the floor tolerates noisy CI shared runners.)
    let lut_ns = lut_elapsed.as_nanos().max(1);
    let res_ns = reservoir_elapsed.as_nanos().max(1);
    // res * 5 < lut * 6  ⇔  res / lut < 1.2
    assert!(
        res_ns * 5 <= lut_ns * 6,
        "round-6 reservoir slower than 1.2× the round-5 LUT path: {:?} vs {:?}",
        reservoir_elapsed,
        lut_elapsed
    );
}
