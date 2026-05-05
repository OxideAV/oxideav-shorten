//! End-to-end decode of hand-built `.shn` bitstreams.
//!
//! ffmpeg 8.x ships only the Shorten *decoder*, not an encoder, and we
//! refuse to consult libavcodec source code as an oracle. So the
//! integration coverage is built bottom-up: we hand-encode small
//! Shorten streams in this file using primitives that exactly mirror
//! the reverse-engineered §5 bitstream layout, then decode them
//! through the public crate API and verify bit-exact reconstruction.
//!
//! Round-2 work: drop a public-domain `.shn` fixture in `tests/data/`
//! and add a black-box decode test against it.

use oxideav_core::bits::BitWriter;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};
use oxideav_shorten::CODEC_ID_STR;

// ────────────────────── tiny .shn writer (clean-room) ──────────────────────

const FN_DIFF0: u32 = 0;
const FN_DIFF1: u32 = 1;
const FN_DIFF2: u32 = 2;
const FN_QUIT: u32 = 4;
const FN_BITSHIFT: u32 = 6;
const FN_ZERO: u32 = 8;
const FN_VERBATIM: u32 = 9;

fn w_unsigned(bw: &mut BitWriter, value: u32, k: u32) {
    let q = value >> k;
    for _ in 0..q {
        bw.write_u32(0, 1);
    }
    bw.write_u32(1, 1);
    if k > 0 {
        let low = value & ((1u32 << k) - 1);
        bw.write_u32(low, k);
    }
}

fn w_signed(bw: &mut BitWriter, value: i32, k: u32) {
    let u = ((value << 1) ^ (value >> 31)) as u32;
    w_unsigned(bw, u, k + 1);
}

fn w_ulong(bw: &mut BitWriter, value: u32) {
    let mut k = 0u32;
    while k < 31 && (value >> k) > 8 {
        k += 1;
    }
    w_unsigned(bw, k, 2);
    w_unsigned(bw, value, k);
}

/// Build a v2 stream prelude: magic + version + 6 ulong header fields
/// + leading FN_VERBATIM with `verbatim_size` zero bytes.
fn write_prelude(
    bw: &mut BitWriter,
    internal_ftype: u32,
    channels: u32,
    blocksize: u32,
    maxnlpc: u32,
    nmean: u32,
    verbatim_size: u32,
) {
    for &b in b"ajkg" {
        bw.write_u32(b as u32, 8);
    }
    bw.write_u32(2, 8); // version
    w_ulong(bw, internal_ftype);
    w_ulong(bw, channels);
    w_ulong(bw, blocksize);
    w_ulong(bw, maxnlpc);
    w_ulong(bw, nmean);
    w_ulong(bw, 0); // skip_bytes
                    // Leading FN_VERBATIM (cmd=9) with verbatim_size zero bytes.
    w_unsigned(bw, FN_VERBATIM, 2);
    w_unsigned(bw, verbatim_size, 5);
    for _ in 0..verbatim_size {
        w_unsigned(bw, 0, 8);
    }
}

fn make_decoder(channels: u32) -> Box<dyn oxideav_core::Decoder> {
    let mut codecs = oxideav_core::CodecRegistry::new();
    oxideav_shorten::register_codecs(&mut codecs);
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.sample_rate = Some(44_100);
    params.channels = Some(channels as u16);
    params.sample_format = Some(SampleFormat::S16);
    codecs.first_decoder(&params).expect("make_decoder")
}

fn decode_to_s16(bytes: Vec<u8>, channels: u32) -> Vec<i16> {
    let mut dec = make_decoder(channels);
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

// ─────────────────────────── tests ───────────────────────────

/// Mono S16 stream with two blocks: a DIFF1 block then a ZERO block.
/// DIFF1 predictor: predicted = decoded[i-1]. With initial history [0,0,0]
/// and residuals [10, 1, -2, 3]: decoded = [10, 11, 9, 12].
/// Then ZERO block: 4 literal zeros.
#[test]
fn decode_diff1_then_zero_mono() {
    let mut bw = BitWriter::new();
    write_prelude(&mut bw, /*ftype S16LH*/ 5, 1, 4, 0, 0, 44);
    // Block 1: DIFF1, energy/k=0, residuals [10, 1, -2, 3].
    w_unsigned(&mut bw, FN_DIFF1, 2);
    w_unsigned(&mut bw, 0, 3); // k
    for v in [10i32, 1, -2, 3] {
        w_signed(&mut bw, v, 0);
    }
    // Block 2: ZERO.
    w_unsigned(&mut bw, FN_ZERO, 2);
    // Quit.
    w_unsigned(&mut bw, FN_QUIT, 2);
    let s = decode_to_s16(bw.into_bytes(), 1);
    assert_eq!(s, vec![10, 11, 9, 12, 0, 0, 0, 0]);
}

/// Mono S16 stream with one DIFF2 block. DIFF2 predictor:
/// predicted = 2*x[i-1] - x[i-2]. With history [0,0,0] and residuals
/// [5, -3, 0, 1]:
///   i=0: pred = 2*0 - 0 = 0  -> 5
///   i=1: pred = 2*5 - 0 = 10 -> 10 + (-3) = 7
///   i=2: pred = 2*7 - 5 = 9  -> 9 + 0 = 9
///   i=3: pred = 2*9 - 7 = 11 -> 11 + 1 = 12
#[test]
fn decode_diff2_mono() {
    let mut bw = BitWriter::new();
    write_prelude(&mut bw, 5, 1, 4, 0, 0, 44);
    w_unsigned(&mut bw, FN_DIFF2, 2);
    w_unsigned(&mut bw, 0, 3); // k = 0
    for v in [5i32, -3, 0, 1] {
        w_signed(&mut bw, v, 0);
    }
    w_unsigned(&mut bw, FN_QUIT, 2);
    let s = decode_to_s16(bw.into_bytes(), 1);
    assert_eq!(s, vec![5, 7, 9, 12]);
}

/// Stereo S16 stream: each channel decoded independently. Left:
/// DIFF0 with coffset=0, residuals [1,2,3,4] → [1,2,3,4]. Right:
/// DIFF1 with residuals [-1, 0, 0, 0] → [-1, -1, -1, -1].
#[test]
fn decode_stereo_independent_channels() {
    let mut bw = BitWriter::new();
    write_prelude(&mut bw, 5, 2, 4, 0, 0, 44);
    // Left = DIFF0, residuals [1,2,3,4].
    w_unsigned(&mut bw, FN_DIFF0, 2);
    w_unsigned(&mut bw, 0, 3);
    for v in [1i32, 2, 3, 4] {
        w_signed(&mut bw, v, 0);
    }
    // Right = DIFF1, residuals [-1, 0, 0, 0].
    w_unsigned(&mut bw, FN_DIFF1, 2);
    w_unsigned(&mut bw, 0, 3);
    for v in [-1i32, 0, 0, 0] {
        w_signed(&mut bw, v, 0);
    }
    w_unsigned(&mut bw, FN_QUIT, 2);
    let s = decode_to_s16(bw.into_bytes(), 2);
    // Interleaved: L0, R0, L1, R1, ...
    assert_eq!(s, vec![1, -1, 2, -1, 3, -1, 4, -1]);
}

/// A FN_BITSHIFT changes the per-stream shift; subsequent samples are
/// left-shifted on output. Use BITSHIFT=2 then DIFF0 with residuals
/// [1, 2] → decoded [1, 2] → output shifted = [4, 8].
#[test]
fn decode_bitshift_applied_to_output() {
    let mut bw = BitWriter::new();
    write_prelude(&mut bw, 5, 1, 2, 0, 0, 44);
    // BITSHIFT = 2.
    w_unsigned(&mut bw, FN_BITSHIFT, 2);
    w_unsigned(&mut bw, 2, 2);
    // DIFF0, k=0, residuals [1, 2].
    w_unsigned(&mut bw, FN_DIFF0, 2);
    w_unsigned(&mut bw, 0, 3);
    for v in [1i32, 2] {
        w_signed(&mut bw, v, 0);
    }
    w_unsigned(&mut bw, FN_QUIT, 2);
    let s = decode_to_s16(bw.into_bytes(), 1);
    assert_eq!(s, vec![4, 8]);
}

/// Full lossless roundtrip (in spirit): hand-encode a known PCM
/// sequence as DIFF0 residuals (the trivial case where every residual
/// equals the sample itself, since coffset=0 and DIFF0's predictor is
/// just coffset), decode, and compare.
#[test]
fn lossless_diff0_roundtrip() {
    let pcm: Vec<i16> = vec![0, 1, -1, 100, -100, 1234, -1234, 32000, -32000, 0];
    let mut bw = BitWriter::new();
    write_prelude(&mut bw, 5, 1, pcm.len() as u32, 0, 0, 44);
    // DIFF0: with coffset=0, decoded = residual + 0 = residual. So
    // the residuals ARE the PCM samples. Pick k=14 to fit ±32k in
    // a single Rice code without massive unary runs.
    w_unsigned(&mut bw, FN_DIFF0, 2);
    w_unsigned(&mut bw, 14, 3);
    for &v in &pcm {
        w_signed(&mut bw, v as i32, 14);
    }
    w_unsigned(&mut bw, FN_QUIT, 2);
    let decoded = decode_to_s16(bw.into_bytes(), 1);
    assert_eq!(decoded, pcm);
}
