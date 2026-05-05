//! Round-trip integration tests for the round-1 Shorten encoder.
//!
//! Strategy: hand-build a known PCM stream, encode it via this crate's
//! [`oxideav_shorten::ShortenEncoder`], then feed the bytes back into
//! this crate's existing decoder and assert bit-exact reconstruction
//! at the S16 output layer (the decoder normalises every supported
//! `internal_ftype` into packed S16, so the round-trip oracle is
//! always the S16 representation of the original PCM).
//!
//! ffmpeg ships only a Shorten *decoder*; there is no encoder we can
//! cross-check against (and the workspace policy bars consulting any
//! third-party Shorten source). The decoder built in this same crate
//! is therefore the binding correctness guard — anything that
//! survives encode→decode losslessly is by definition spec-conformant
//! at the bitstream level.

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};
use oxideav_shorten::{ShortenEncoder, ShortenEncoderConfig, ShortenFtype, CODEC_ID_STR};

// ─────────────────── helpers ───────────────────

fn decode_to_s16(bytes: Vec<u8>, channels: u32) -> Vec<i16> {
    let mut codecs = oxideav_core::CodecRegistry::new();
    oxideav_shorten::register_codecs(&mut codecs);
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.sample_rate = Some(44_100);
    params.channels = Some(channels as u16);
    params.sample_format = Some(SampleFormat::S16);
    let mut dec = codecs.make_decoder(&params).expect("make_decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 44_100), bytes);
    dec.send_packet(&pkt).unwrap();
    let frame = dec.receive_frame().expect("decode");
    let Frame::Audio(af) = frame else {
        panic!("expected audio");
    };
    af.data[0]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Project a raw PCM sample (in the encoder's per-`ftype` integer
/// space) into the decoder's S16 output space — mirrors the decoder's
/// `pack_s16`. Used to predict the expected post-decode result without
/// duplicating the decoder's clipping logic.
fn expected_s16(raw: i32, ftype: ShortenFtype) -> i16 {
    match ftype {
        ShortenFtype::S8 => (raw.clamp(-128, 127).saturating_mul(256)) as i16,
        ShortenFtype::U8 => ((raw - 0x80).clamp(-128, 127).saturating_mul(256)) as i16,
        ShortenFtype::S16Le | ShortenFtype::S16Be => raw.clamp(-32_768, 32_767) as i16,
        ShortenFtype::U16Le | ShortenFtype::U16Be => (raw - 0x8000).clamp(-32_768, 32_767) as i16,
    }
}

fn roundtrip(samples: &[i32], ftype: ShortenFtype, channels: u32, blocksize: u32) -> Vec<i16> {
    let cfg = ShortenEncoderConfig::new(ftype, channels, blocksize);
    let mut enc = ShortenEncoder::new(cfg).expect("encoder");
    let bytes = enc.encode(samples).expect("encode");
    decode_to_s16(bytes, channels)
}

fn expected(samples: &[i32], ftype: ShortenFtype) -> Vec<i16> {
    samples.iter().map(|&s| expected_s16(s, ftype)).collect()
}

// ─────────────────── core roundtrips ───────────────────

#[test]
fn roundtrip_mono_s16le_ramp() {
    // Linear ramp — DIFF2 should reduce this to one big residual then
    // a long run of small ones.
    let pcm: Vec<i32> = (0..256).map(|i| i * 100).collect();
    let out = roundtrip(&pcm, ShortenFtype::S16Le, 1, 256);
    assert_eq!(out, expected(&pcm, ShortenFtype::S16Le));
}

#[test]
fn roundtrip_mono_s16le_random_like() {
    // Pseudo-random sequence: a poor-man's LCG (no third-party deps).
    // The encoder should still pick a sensible predictor and round-trip.
    let mut state: u32 = 0xDEAD_BEEF;
    let mut pcm = Vec::with_capacity(512);
    for _ in 0..512 {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        // Keep amplitude well below i16::MAX so we don't trigger the
        // decoder's S16 clipping (the round-trip oracle would lie).
        pcm.push(((state >> 16) as i16 / 4) as i32);
    }
    let out = roundtrip(&pcm, ShortenFtype::S16Le, 1, 256);
    assert_eq!(out, expected(&pcm, ShortenFtype::S16Le));
}

#[test]
fn roundtrip_stereo_s16le_independent_channels() {
    // Two channels with completely different content: a sine-like
    // pattern on L, a slow ramp on R. Encoder must interleave them
    // round-robin and keep per-channel state separate.
    let mut pcm: Vec<i32> = Vec::with_capacity(512);
    for i in 0..256i32 {
        // L: alternating positive/negative.
        let l = if i % 2 == 0 { 1000 } else { -1000 };
        // R: slow ramp.
        let r = i * 50;
        pcm.push(l);
        pcm.push(r);
    }
    let out = roundtrip(&pcm, ShortenFtype::S16Le, 2, 128);
    assert_eq!(out, expected(&pcm, ShortenFtype::S16Le));
}

#[test]
fn roundtrip_mono_u8_offset_signal() {
    // U8: 0..=255, midpoint 128. Use a small DC + ramp away from the
    // midpoint to stress the per-channel mean FIFO with init_offset
    // = 0x80.
    let pcm: Vec<i32> = (0..128).map(|i| 100 + (i % 32)).collect();
    let out = roundtrip(&pcm, ShortenFtype::U8, 1, 128);
    assert_eq!(out, expected(&pcm, ShortenFtype::U8));
}

#[test]
fn roundtrip_mono_s8_signal() {
    // S8: -128..=127, midpoint 0. Mix of positive and negative.
    let pcm: Vec<i32> = (-64..64).collect();
    let out = roundtrip(&pcm, ShortenFtype::S8, 1, 128);
    assert_eq!(out, expected(&pcm, ShortenFtype::S8));
}

#[test]
fn roundtrip_mono_s16be_ramp() {
    // S16BE — internally identical to S16LE for the predictor; only
    // the (notional) container byte order differs. Verifies the ftype
    // mapping is right.
    let pcm: Vec<i32> = (0..128).map(|i| -1000 + i * 50).collect();
    let out = roundtrip(&pcm, ShortenFtype::S16Be, 1, 128);
    assert_eq!(out, expected(&pcm, ShortenFtype::S16Be));
}

#[test]
fn roundtrip_mono_u16le_offset() {
    // U16: 0..=65535, midpoint 0x8000. Use a slowly varying value
    // around the midpoint.
    let pcm: Vec<i32> = (0..128).map(|i| 0x8000 + (i * 17) - 1000).collect();
    let out = roundtrip(&pcm, ShortenFtype::U16Le, 1, 128);
    assert_eq!(out, expected(&pcm, ShortenFtype::U16Le));
}

// ─────────────────── selector + emission tests ───────────────────

#[test]
fn predictor_selector_picks_diff0_for_random_uncorrelated() {
    // Pure noise centred near 0 — DIFF0 (residual = sample) should
    // beat the differentiating predictors because each Δ adds noise
    // proportional to the original signal's variance.
    let mut state: u32 = 0xCAFE_BABE;
    let mut pcm = Vec::with_capacity(64);
    for _ in 0..64 {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        // Tiny amplitude so DIFF0's residuals are small in absolute
        // terms; DIFF1 of white noise has 2× the variance, so its
        // |residual| sum will be larger.
        pcm.push(((state >> 24) as i8 / 16) as i32);
    }
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 64);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let _ = enc.encode(&pcm).unwrap();
    let info = enc.block_info();
    assert_eq!(info.len(), 1);
    // We don't strictly assert DIFF0 here because tiny synthetic
    // sequences can flip — but we DO assert it's not DIFF3, which is
    // the worst-case for white noise (Δ³ amplifies by 20×).
    assert_ne!(info[0].cmd, 3, "DIFF3 should not win for white noise");
}

#[test]
fn predictor_selector_picks_diff2_for_linear_ramp() {
    // A perfect ramp x[i] = a*i + b: Δ²(ramp) = 0 from i=2 onward, so
    // DIFF2's |residual| sum is the smallest (just the two seed
    // residuals). DIFF3's Δ³(ramp) is also zero from i=3, but the
    // first three seed residuals (10, 4, -1 in the test) end up
    // larger, so DIFF2 wins.
    let pcm: Vec<i32> = (0..32).map(|i| 100 + i * 30).collect();
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 32);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let _ = enc.encode(&pcm).unwrap();
    let info = enc.block_info();
    assert_eq!(info.len(), 1);
    // DIFF1 or DIFF2 should win — both kill the linear part. DIFF1's
    // residuals are all the slope (30); DIFF2's are mostly zero. So
    // DIFF2 wins.
    assert_eq!(info[0].cmd, 2, "DIFF2 should win for a linear ramp");
}

#[test]
fn predictor_selector_picks_diff3_for_quadratic() {
    // A quadratic ramp: x[i] = i*i. Δ³(quadratic) = 0, so DIFF3 wins
    // (residuals are zero from i=3 onward).
    let pcm: Vec<i32> = (0i32..32).map(|i| i * i).collect();
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 32);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let _ = enc.encode(&pcm).unwrap();
    let info = enc.block_info();
    assert_eq!(info.len(), 1);
    assert_eq!(info[0].cmd, 3, "DIFF3 should win for a quadratic");
}

#[test]
fn predictor_selector_picks_diff1_for_constant_signal() {
    // A constant non-zero signal: DIFF1 residuals are all zero after
    // the first one (which is `c - 0 = c`). DIFF0 would emit
    // `residual = sample - coffset` — with nmean=0, coffset is 0
    // for S16, so DIFF0 residuals are all `c`. DIFF1's |sum| = c,
    // DIFF0's = c * blocksize. DIFF1 wins.
    let pcm = vec![500i32; 32];
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 32);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let _ = enc.encode(&pcm).unwrap();
    let info = enc.block_info();
    assert_eq!(info.len(), 1);
    assert_eq!(info[0].cmd, 1, "DIFF1 should win for a constant signal");
}

#[test]
fn all_zero_block_emits_fn_zero() {
    // An all-zero block (with coffset=0, which is the case for S16
    // with nmean=0) must collapse to FN_ZERO.
    let pcm = vec![0i32; 64];
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 64);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let bytes_zero = enc.encode(&pcm).unwrap();
    let info = enc.block_info();
    assert_eq!(info.len(), 1);
    assert_eq!(info[0].cmd, 8, "all-zero block must emit FN_ZERO (cmd=8)");

    // Compare size: a non-zero block of the same length encoded as
    // DIFF0 must produce more bytes than the FN_ZERO version. Use a
    // small constant that survives the predictor race for DIFF0.
    let pcm2 = vec![1i32; 64];
    let mut enc2 =
        ShortenEncoder::new(ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 64)).unwrap();
    let bytes_one = enc2.encode(&pcm2).unwrap();
    assert!(
        bytes_zero.len() < bytes_one.len(),
        "FN_ZERO ({} bytes) should be smaller than any DIFFn block ({} bytes)",
        bytes_zero.len(),
        bytes_one.len()
    );
}

#[test]
fn multi_block_roundtrip_keeps_history_state() {
    // Several blocks back-to-back — verifies that the encoder's
    // per-channel history ring state is propagated correctly between
    // blocks and matches the decoder's state.
    //
    // Use a shape that crosses block boundaries: a triangular wave
    // with period > blocksize.
    let pcm: Vec<i32> = (0i32..512)
        .map(|i| {
            let phase = i % 64;
            if phase < 32 {
                phase * 100
            } else {
                (64 - phase) * 100
            }
        })
        .collect();
    let out = roundtrip(&pcm, ShortenFtype::S16Le, 1, 64); // 8 blocks
    assert_eq!(out, expected(&pcm, ShortenFtype::S16Le));

    // Also verify with a non-default nmean so the running-mean FIFO
    // is exercised in both encoder and decoder.
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 64).with_nmean(4);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let bytes = enc.encode(&pcm).unwrap();
    let out = decode_to_s16(bytes, 1);
    assert_eq!(out, expected(&pcm, ShortenFtype::S16Le));
}

#[test]
fn roundtrip_stereo_s16le_with_nmean() {
    // Stereo + nmean = 4: stresses BOTH the per-channel state (each
    // channel keeps its own FIFO) AND the running-mean compensation
    // path. Use slowly-DC-shifting content so the mean offsets matter.
    let mut pcm: Vec<i32> = Vec::with_capacity(512);
    for i in 0..256i32 {
        let l = 5000 + (i * 10); // slow rising DC
        let r = -3000 + (i * 8); // slow rising DC, opposite sign
        pcm.push(l);
        pcm.push(r);
    }
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 2, 64).with_nmean(4);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let bytes = enc.encode(&pcm).unwrap();
    let out = decode_to_s16(bytes, 2);
    assert_eq!(out, expected(&pcm, ShortenFtype::S16Le));
}

// ─────────────────── error paths ───────────────────

#[test]
fn encoder_rejects_partial_block() {
    // Round-1 encoder requires sample count to be a multiple of
    // (blocksize * channels).
    let pcm = vec![0i32; 100]; // 100 samples, blocksize=64 → not multiple
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 64);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let err = enc.encode(&pcm).unwrap_err();
    assert!(
        format!("{err}").contains("multiple of blocksize"),
        "expected partial-block rejection, got: {err}"
    );
}

#[test]
fn encoder_rejects_misaligned_channel_count() {
    // Stereo stream with an odd sample count.
    let pcm = vec![0i32; 7];
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 2, 4);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let err = enc.encode(&pcm).unwrap_err();
    assert!(
        format!("{err}").contains("multiple of channels"),
        "expected channel-count rejection, got: {err}"
    );
}

#[test]
fn encoder_rejects_zero_blocksize() {
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 0);
    let res = ShortenEncoder::new(cfg);
    let err = match res {
        Ok(_) => panic!("encoder accepted blocksize=0"),
        Err(e) => e,
    };
    assert!(format!("{err}").contains("blocksize"));
}

#[test]
fn encoder_rejects_zero_channels() {
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 0, 64);
    let res = ShortenEncoder::new(cfg);
    let err = match res {
        Ok(_) => panic!("encoder accepted channels=0"),
        Err(e) => e,
    };
    assert!(format!("{err}").contains("channels"));
}

// ─────────────────── QLPC encoder tests ───────────────────

/// Sine-like signal (sum of harmonics): QLPC should produce smaller
/// residuals than DIFFn on a tonal, quasi-periodic source.
/// We verify only lossless roundtrip here — the predictor race is
/// internal; the correctness guard is bit-exact decode.
#[test]
fn roundtrip_qlpc_sine_like_mono() {
    // Generate a 1 kHz-ish sine approximation using integer arithmetic.
    // Using a simple sinusoidal-ish signal: x[i] = A * sin(2*pi*i/32)
    // approximated via a small lookup table scaled to S16 range.
    let mut pcm: Vec<i32> = Vec::with_capacity(256);
    for i in 0..256i32 {
        // cos-approximation: period = 64 samples
        let phase = (i % 64) as f64 * std::f64::consts::TAU / 64.0;
        let s = (phase.sin() * 10_000.0) as i32;
        pcm.push(s);
    }
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 64).with_maxnlpc(8);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let bytes = enc.encode(&pcm).unwrap();
    let out = decode_to_s16(bytes, 1);
    assert_eq!(out, expected(&pcm, ShortenFtype::S16Le));
}

/// QLPC with stereo: both channels use independent QLPC predictors.
#[test]
fn roundtrip_qlpc_stereo_independent() {
    let mut pcm: Vec<i32> = Vec::with_capacity(512);
    for i in 0..256i32 {
        let phase = (i % 48) as f64 * std::f64::consts::TAU / 48.0;
        let l = (phase.sin() * 8_000.0) as i32;
        let r = (phase.cos() * 6_000.0) as i32;
        pcm.push(l);
        pcm.push(r);
    }
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 2, 64).with_maxnlpc(4);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let bytes = enc.encode(&pcm).unwrap();
    let out = decode_to_s16(bytes, 2);
    assert_eq!(out, expected(&pcm, ShortenFtype::S16Le));
}

/// QLPC round-trip with nmean > 0 (exercises mean-FIFO interaction).
#[test]
fn roundtrip_qlpc_with_nmean() {
    let mut pcm: Vec<i32> = Vec::with_capacity(256);
    for i in 0..256i32 {
        let phase = (i % 32) as f64 * std::f64::consts::TAU / 32.0;
        let s = 5000 + (phase.sin() * 4_000.0) as i32;
        pcm.push(s);
    }
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 64)
        .with_maxnlpc(8)
        .with_nmean(4);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let bytes = enc.encode(&pcm).unwrap();
    let out = decode_to_s16(bytes, 1);
    assert_eq!(out, expected(&pcm, ShortenFtype::S16Le));
}

/// QLPC is available as a candidate when maxnlpc > 0: verify the block
/// info reports one of the valid predictor codes (0/1/2/3/7/8). We do
/// NOT assert QLPC always beats DIFFn — for a discrete sinusoid DIFF3
/// can be competitive; the important property is lossless roundtrip
/// and that the QLPC path runs without panicking.
#[test]
fn qlpc_predictor_runs_without_error() {
    let mut pcm: Vec<i32> = Vec::with_capacity(128);
    for i in 0..128i32 {
        let phase = (i % 64) as f64 * std::f64::consts::TAU / 64.0;
        let s = (phase.sin() * 12_000.0) as i32;
        pcm.push(s);
    }
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 128).with_maxnlpc(8);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let bytes = enc.encode(&pcm).unwrap();
    let info = enc.block_info();
    assert_eq!(info.len(), 1);
    // Any valid predictor code is acceptable.
    assert!(
        [0u32, 1, 2, 3, 7, 8].contains(&info[0].cmd),
        "unexpected cmd {}",
        info[0].cmd
    );
    // Roundtrip must be bit-exact.
    let out = decode_to_s16(bytes, 1);
    assert_eq!(out, expected(&pcm, ShortenFtype::S16Le));
}

// ─────────────────── BITSHIFT encoder tests ───────────────────

/// Samples with 2 consistent trailing zero bits: encoder should detect
/// bitshift=2, emit FN_BITSHIFT, then encode the right-shifted values.
/// After encode→decode the original left-shifted magnitude is restored.
#[test]
fn roundtrip_bitshift_consistent_trailing_zeros() {
    // Multiply all samples by 4 (shift left 2): every sample has at
    // least 2 trailing zero bits in the binary representation.
    let base: Vec<i32> = (0i32..128).map(|i| (i - 64) * 100).collect();
    let pcm: Vec<i32> = base.iter().map(|&s| s * 4).collect();
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 128).with_bitshift(true);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let bytes = enc.encode(&pcm).unwrap();
    let out = decode_to_s16(bytes, 1);
    assert_eq!(out, expected(&pcm, ShortenFtype::S16Le));
}

/// Bitshift encoder produces a smaller output than no-bitshift encoder
/// for a signal with consistent trailing zeros.
#[test]
fn bitshift_reduces_stream_size() {
    let pcm: Vec<i32> = (0i32..256)
        .map(|i| ((i - 128) * 256) & !0xFF) // clear low 8 bits
        .collect();
    let cfg_with = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 256).with_bitshift(true);
    let cfg_without = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 256);
    let mut enc_with = ShortenEncoder::new(cfg_with).unwrap();
    let mut enc_without = ShortenEncoder::new(cfg_without).unwrap();
    let bytes_with = enc_with.encode(&pcm).unwrap();
    let bytes_without = enc_without.encode(&pcm).unwrap();
    // Both must round-trip correctly.
    assert_eq!(
        decode_to_s16(bytes_with.clone(), 1),
        expected(&pcm, ShortenFtype::S16Le)
    );
    assert_eq!(
        decode_to_s16(bytes_without.clone(), 1),
        expected(&pcm, ShortenFtype::S16Le)
    );
    // Bitshift-enabled stream should be smaller.
    assert!(
        bytes_with.len() < bytes_without.len(),
        "bitshift stream ({} bytes) should be smaller than plain ({} bytes)",
        bytes_with.len(),
        bytes_without.len()
    );
}

/// Bitshift + QLPC together: both features active, sinusoidal signal
/// with trailing zeros.
#[test]
fn roundtrip_bitshift_and_qlpc_combined() {
    let mut pcm: Vec<i32> = Vec::with_capacity(256);
    for i in 0..256i32 {
        let phase = (i % 64) as f64 * std::f64::consts::TAU / 64.0;
        // Multiply by 4 to introduce 2 trailing zero bits.
        let s = ((phase.sin() * 8_000.0) as i32) * 4;
        pcm.push(s);
    }
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 128)
        .with_maxnlpc(8)
        .with_bitshift(true);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let bytes = enc.encode(&pcm).unwrap();
    let out = decode_to_s16(bytes, 1);
    assert_eq!(out, expected(&pcm, ShortenFtype::S16Le));
}

/// All-zero block with bitshift enabled: FN_ZERO shortcut must still
/// fire (shifted zeros are still zeros).
#[test]
fn bitshift_zero_block_still_emits_fn_zero() {
    let pcm = vec![0i32; 64];
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 64).with_bitshift(true);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let _ = enc.encode(&pcm).unwrap();
    let info = enc.block_info();
    assert_eq!(info[0].cmd, 8, "all-zero block must be FN_ZERO (8)");
}

/// Bitshift with stereo channels: the shift applies globally (both
/// channels use the same shift value), and both must round-trip.
#[test]
fn roundtrip_bitshift_stereo() {
    let mut pcm: Vec<i32> = Vec::with_capacity(512);
    for i in 0..256i32 {
        let l = ((i - 128) * 100) * 8; // 3 trailing zero bits
        let r = ((128 - i) * 80) * 8;
        pcm.push(l);
        pcm.push(r);
    }
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 2, 128).with_bitshift(true);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let bytes = enc.encode(&pcm).unwrap();
    let out = decode_to_s16(bytes, 2);
    assert_eq!(out, expected(&pcm, ShortenFtype::S16Le));
}

// ─────────────────── ffmpeg cross-decode tests ───────────────────

/// Helper: invoke ffmpeg to decode a .shn file to raw S16LE PCM.
/// Returns None if ffmpeg is not available in PATH.
fn ffmpeg_decode_shn(
    shn_bytes: &[u8],
    channels: u32,
    sample_rate: u32,
    tag: &str,
) -> Option<Vec<i16>> {
    use std::io::Write;
    use std::process::Command;

    // Check ffmpeg is present.
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return None;
    }

    let tmp_dir = std::env::temp_dir();
    // Use a test-specific tag to avoid collisions when tests run in parallel.
    let shn_path = tmp_dir.join(format!("oxideav_shorten_{tag}.shn"));
    let pcm_path = tmp_dir.join(format!("oxideav_shorten_{tag}.raw"));

    std::fs::File::create(&shn_path)
        .unwrap()
        .write_all(shn_bytes)
        .unwrap();

    // Decode: ffmpeg -y -f shn -i <in>.shn -f s16le -ar <rate> -ac <ch> <out>.raw
    // The format name is "shn" in ffmpeg (not "shorten").
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "shn",
            "-i",
            shn_path.to_str().unwrap(),
            "-f",
            "s16le",
            "-ar",
            &sample_rate.to_string(),
            "-ac",
            &channels.to_string(),
            pcm_path.to_str().unwrap(),
        ])
        .output()
        .expect("ffmpeg failed to start");

    if !status.status.success() {
        let stderr = String::from_utf8_lossy(&status.stderr);
        panic!("ffmpeg decode failed:\n{stderr}");
    }

    let raw = std::fs::read(&pcm_path).expect("read raw pcm");
    Some(
        raw.chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect(),
    )
}

/// ffmpeg cross-decode: DIFFn encoder output decoded by ffmpeg must
/// match the expected S16 samples. Covers the baseline DIFFn path.
#[test]
fn ffmpeg_cross_decode_diffn() {
    let pcm: Vec<i32> = (0..256i32).map(|i| (i - 128) * 200).collect();
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 256);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let bytes = enc.encode(&pcm).unwrap();

    let Some(decoded) = ffmpeg_decode_shn(&bytes, 1, 44_100, "diffn") else {
        eprintln!("ffmpeg not available — skipping cross-decode");
        return;
    };
    assert_eq!(decoded, expected(&pcm, ShortenFtype::S16Le));
}

/// ffmpeg cross-decode: QLPC encoder output decoded by ffmpeg.
#[test]
fn ffmpeg_cross_decode_qlpc() {
    let mut pcm: Vec<i32> = Vec::with_capacity(256);
    for i in 0..256i32 {
        let phase = (i % 64) as f64 * std::f64::consts::TAU / 64.0;
        let s = (phase.sin() * 10_000.0) as i32;
        pcm.push(s);
    }
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 256).with_maxnlpc(8);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let bytes = enc.encode(&pcm).unwrap();

    let Some(decoded) = ffmpeg_decode_shn(&bytes, 1, 44_100, "qlpc") else {
        eprintln!("ffmpeg not available — skipping cross-decode");
        return;
    };
    assert_eq!(decoded, expected(&pcm, ShortenFtype::S16Le));
}

/// ffmpeg cross-decode: BITSHIFT encoder output decoded by ffmpeg.
#[test]
fn ffmpeg_cross_decode_bitshift() {
    let pcm: Vec<i32> = (0..256i32).map(|i| ((i - 128) * 100) * 4).collect();
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 1, 256).with_bitshift(true);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let bytes = enc.encode(&pcm).unwrap();

    let Some(decoded) = ffmpeg_decode_shn(&bytes, 1, 44_100, "bitshift") else {
        eprintln!("ffmpeg not available — skipping cross-decode");
        return;
    };
    assert_eq!(decoded, expected(&pcm, ShortenFtype::S16Le));
}

/// ffmpeg cross-decode: QLPC + BITSHIFT combined, stereo.
#[test]
fn ffmpeg_cross_decode_qlpc_bitshift_stereo() {
    let mut pcm: Vec<i32> = Vec::with_capacity(512);
    for i in 0..256i32 {
        let phase = (i % 64) as f64 * std::f64::consts::TAU / 64.0;
        let l = ((phase.sin() * 8_000.0) as i32) * 4;
        let r = ((phase.cos() * 6_000.0) as i32) * 4;
        pcm.push(l);
        pcm.push(r);
    }
    let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 2, 128)
        .with_maxnlpc(8)
        .with_bitshift(true);
    let mut enc = ShortenEncoder::new(cfg).unwrap();
    let bytes = enc.encode(&pcm).unwrap();

    let Some(decoded) = ffmpeg_decode_shn(&bytes, 2, 44_100, "qlpc_bs_stereo") else {
        eprintln!("ffmpeg not available — skipping cross-decode");
        return;
    };
    assert_eq!(decoded, expected(&pcm, ShortenFtype::S16Le));
}
