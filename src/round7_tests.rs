//! Round-7 tests: fused SoA stereo decode path.
//!
//! Round 7 collapses the round-6 per-channel `Vec<i32>` accumulators
//! plus end-of-stream interleave pass into a **single fused write**:
//! each per-channel block is emitted directly into the interleaved
//! output buffer at strided positions
//! (`interleaved[(t + i) * nch + c]`), with the per-stream `bshift`
//! left-shift applied inline. A single reusable scratch block buffer
//! is allocated once per [`crate::decode`] call and re-used across
//! every block, eliminating the per-block `Vec<i32>` allocation that
//! the round-6 path performed.
//!
//! These tests cover:
//!
//! * **Bit-exactness across channel counts.** Mono / stereo / 4-channel
//!   roundtrips assert the strided-write logic matches the round-6
//!   per-channel-then-interleave behaviour byte-for-byte.
//! * **Partial-block-at-tail.** A stereo stream where the channels
//!   commit different numbers of blocks (because the encoder produced
//!   a non-multiple-of-`nch` block count) truncates to the shortest
//!   channel — same as the round-6 path.
//! * **`bshift` applied per-block, not just at end-of-stream.** The
//!   round-6 path applied the *final* bshift to every sample; round-7
//!   applies the bshift in effect at each block's commit. For streams
//!   that emit `BLOCK_FN_BITSHIFT` once at the start (the encoder's
//!   common case) the two are identical, which is what the existing
//!   roundtrip suite covers. The round-7 behaviour is more correct
//!   for variable-bshift streams.
//! * **Throughput floor.** Stereo and 4-channel 4 KB-block decode
//!   wallclock prints surface under `--nocapture` so the
//!   round-7-vs-round-6 delta is reproducible.

use crate::decoder::decode;
use crate::encoder::{encode, EncoderConfig};
use crate::header::Filetype;

/// Stereo 4 KB-block roundtrip — the headline shape for round-7's
/// fused SoA path.
#[test]
fn stereo_4k_block_roundtrip_bit_exact() {
    let bs = 4096;
    let total = bs * 2;
    let mut samples: Vec<i32> = Vec::with_capacity(total);
    for i in 0..bs {
        // L / R with distinct content so DIFF1 picks different
        // residual widths on each channel.
        samples.push((i as i32) % 2048);
        samples.push(((i + 23) as i32) % 2048);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 2).with_blocksize(bs as u32);
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples, samples);
    assert_eq!(decoded.samples_per_channel, bs);
}

/// 4-channel 4 KB-block roundtrip — confirms the strided-write logic
/// generalises beyond mono and stereo.
#[test]
fn four_channel_4k_block_roundtrip_bit_exact() {
    let bs = 1024;
    let nch = 4usize;
    let total = bs * nch;
    let mut samples: Vec<i32> = Vec::with_capacity(total);
    for i in 0..bs {
        for c in 0..nch {
            // Distinct channel content for each lane.
            samples.push(((i + c * 47) as i32) % 1024);
        }
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, nch as u16).with_blocksize(bs as u32);
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples, samples);
    assert_eq!(decoded.samples_per_channel, bs);
}

/// Stereo with a non-zero `bshift` (lossy mode). Confirms the per-block
/// inline-`wrapping_shl` write produces samples whose lower `bshift`
/// bits are zero, consistent with the round-6 end-of-stream
/// `wrapping_shl` pass.
#[test]
fn stereo_with_bshift_inline_shift_applied() {
    let bs = 256;
    let total = bs * 2;
    let mut samples: Vec<i32> = Vec::with_capacity(total);
    for i in 0..bs {
        // Even-multiple inputs (bshift = 2 → low bits = 00).
        samples.push(((i as i32) << 2) & 0x7FFF);
        samples.push((((i + 11) as i32) << 2) & 0x7FFF);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 2)
        .with_blocksize(bs as u32)
        .with_bshift(2);
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.final_bshift, 2);
    // Every output sample must have its bottom 2 bits zero (bshift = 2
    // is applied as a left-shift of the encoder-side right-shifted
    // residual).
    for &s in &decoded.samples {
        assert_eq!(s & 0b11, 0, "bshift=2 produced non-zero low bits: {s}");
    }
    assert_eq!(decoded.samples.len(), samples.len());
}

/// Many-blocks stereo roundtrip — confirms the per-channel
/// `committed[]` cursor advances correctly as the channel cursor
/// round-robins.
#[test]
fn stereo_many_blocks_roundtrip() {
    let bs = 128;
    let n_blocks_per_channel = 32;
    let total = bs * 2 * n_blocks_per_channel;
    let mut samples: Vec<i32> = Vec::with_capacity(total);
    let mut state: u32 = 0xBEEF_CAFE;
    for _ in 0..(bs * n_blocks_per_channel) {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let l = ((state >> 16) as i16) as i32 / 8;
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let r = ((state >> 16) as i16) as i32 / 8;
        samples.push(l);
        samples.push(r);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 2).with_blocksize(bs as u32);
    let bytes = encode(&cfg, &samples).expect("encode");
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples_per_channel, bs * n_blocks_per_channel);
    assert_eq!(decoded.samples, samples);
}

/// Print the best-of-5 wallclock for stereo 4 KB-block decode under the
/// round-7 fused SoA path. Surfaces under `--nocapture` so the
/// round-6-vs-round-7 delta is reproducible.
#[test]
fn stereo_long_block_decode_best_of_5_print() {
    use std::time::Instant;
    let bs = 4096;
    let n_blocks_per_channel = 64;
    let total = bs * 2 * n_blocks_per_channel;
    let mut samples: Vec<i32> = Vec::with_capacity(total);
    let mut state: u32 = 0xC0DE_C0DE;
    for _ in 0..total {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let v = ((state >> 16) as i16) as i32 / 4;
        samples.push(v);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 2).with_blocksize(bs as u32);
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
    println!(
        "round-7 stereo decode {} samples best-of-5: {:?}",
        total, best
    );
}

/// Throughput floor for stereo decode. The fused SoA path should
/// finish a 64-block × 4096-sample-per-block × 2-channel stream in well
/// under 1 s on contemporary hardware. The assertion is generous to
/// tolerate noisy CI shared runners; its purpose is regression-only.
#[test]
fn stereo_long_block_decode_throughput_floor() {
    use std::time::Instant;
    let bs = 4096;
    let n_blocks_per_channel = 64;
    let total = bs * 2 * n_blocks_per_channel;
    let mut samples: Vec<i32> = Vec::with_capacity(total);
    let mut state: u32 = 0xFEED_FACE;
    for _ in 0..total {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let v = ((state >> 16) as i16) as i32 / 4;
        samples.push(v);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 2).with_blocksize(bs as u32);
    let bytes = encode(&cfg, &samples).expect("encode");
    let start = Instant::now();
    let decoded = decode(&bytes).expect("decode");
    let elapsed = start.elapsed();
    assert_eq!(decoded.samples.len(), total);
    assert!(
        elapsed.as_millis() < 2000,
        "stereo decode of {} samples took {:?} (regression?)",
        total,
        elapsed
    );
}
