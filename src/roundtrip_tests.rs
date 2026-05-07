//! Self-roundtrip tests against the test-only minimal encoder.
//!
//! Round 1 deliverable: encode a small PCM input through the
//! [`encoder::encode_minimal`] helper and confirm [`decode`]
//! reconstructs the input byte-exactly. The encoder is *not* a
//! production encoder; it constructs valid wire-format streams
//! sufficient to drive the decoder along all of the
//! function-code / predictor / verbatim / blocksize / quit paths.

use crate::decoder::decode;
use crate::encoder::{encode_minimal, encode_qlpc_block, encode_with_bitshift_and_zero};
use crate::header::Filetype;

fn assert_roundtrip(
    filetype: Filetype,
    channels: u16,
    blocksize: u32,
    samples: &[i32],
    predictor: u32,
    verbatim: &[u8],
) {
    let bytes = encode_minimal(filetype, channels, blocksize, samples, predictor, verbatim);
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(
        decoded.header.channels, channels,
        "channels mismatch on roundtrip"
    );
    assert_eq!(
        decoded.header.filetype, filetype,
        "filetype mismatch on roundtrip"
    );
    assert_eq!(
        decoded.verbatim_prefix, verbatim,
        "verbatim prefix mismatch"
    );
    assert_eq!(decoded.samples.len(), samples.len(), "sample count");
    assert_eq!(decoded.samples, samples, "sample values");
    assert_eq!(decoded.final_bshift, 0);
}

#[test]
fn mono_diff1_short() {
    // 16 samples mono, blocksize 8 → 2 blocks.
    let samples: Vec<i32> = (0i32..16).map(|i| i * 100).collect();
    assert_roundtrip(Filetype::S16Le, 1, 8, &samples, 1, &[]);
}

#[test]
fn stereo_diff1_short() {
    // 32 interleaved samples (16 per channel), blocksize 8 → 4 blocks
    // (2 per channel, alternating).
    let mut samples = Vec::with_capacity(32);
    for i in 0i32..16 {
        samples.push(i * 10);
        samples.push(-i * 7);
    }
    assert_roundtrip(Filetype::S16Le, 2, 8, &samples, 1, &[]);
}

#[test]
fn mono_diff0_short() {
    let samples: Vec<i32> = (-8..8).collect();
    assert_roundtrip(Filetype::S16Le, 1, 8, &samples, 0, &[]);
}

#[test]
fn mono_diff2_short() {
    let samples: Vec<i32> = (0i32..16).map(|i| i * i - 32).collect();
    assert_roundtrip(Filetype::S16Le, 1, 8, &samples, 2, &[]);
}

#[test]
fn mono_diff3_short() {
    let samples: Vec<i32> = (0i32..24).map(|i| (i - 12).pow(3) / 8).collect();
    assert_roundtrip(Filetype::S16Le, 1, 8, &samples, 3, &[]);
}

#[test]
fn mono_with_verbatim_prefix() {
    // Simulate a tiny RIFF-style header preserved through the codec.
    let prefix = b"RIFF\x10\x00\x00\x00WAVE";
    let samples: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
    assert_roundtrip(Filetype::S16Le, 1, 8, &samples, 1, prefix);
}

#[test]
fn stereo_with_partial_block_at_tail() {
    // 22 samples per channel, blocksize 8 → channels emit 2 full
    // blocks (16 samples) then a final partial block of 6 samples
    // each via BLOCK_FN_BLOCKSIZE override.
    let mut samples = Vec::with_capacity(44);
    for i in 0i32..22 {
        samples.push(i * 3);
        samples.push(-i * 5);
    }
    assert_roundtrip(Filetype::S16Le, 2, 8, &samples, 1, &[]);
}

#[test]
fn s16be_filetype_roundtrips() {
    let samples: Vec<i32> = (-4..4).collect();
    assert_roundtrip(Filetype::S16Be, 1, 4, &samples, 1, &[]);
}

#[test]
fn u8_filetype_roundtrips_signed_lane() {
    // U8 wire form: residual lane is signed, so we feed signed
    // values centered on zero. The output i32 lanes should
    // reconstruct exactly.
    let samples: Vec<i32> = (-4..4).collect();
    assert_roundtrip(Filetype::U8, 1, 4, &samples, 1, &[]);
}

#[test]
fn rejects_bad_magic() {
    let bytes = vec![0u8; 64];
    let err = decode(&bytes).unwrap_err();
    assert!(matches!(err, crate::Error::BadMagic(_)));
}

#[test]
fn rejects_truncated_header() {
    let bytes = b"ajk".to_vec();
    let err = decode(&bytes).unwrap_err();
    assert!(matches!(err, crate::Error::Truncated));
}

#[test]
fn rejects_unsupported_version() {
    let mut bytes = vec![0u8; 16];
    bytes[..4].copy_from_slice(b"ajkg");
    bytes[4] = 7;
    let err = decode(&bytes).unwrap_err();
    assert!(matches!(err, crate::Error::UnsupportedVersion(7)));
}

#[test]
fn rejects_extra_bytes_only_via_unknown_fn_code() {
    // A truncated stream that omits BLOCK_FN_QUIT and runs out of
    // bits in the middle of a block should surface as either
    // UnexpectedEof or UnknownFunctionCode (the trailing bits are
    // garbage). Verify it does *not* silently succeed.
    let mut bytes = encode_minimal(Filetype::S16Le, 1, 4, &[1i32, 2, 3, 4], 1, &[]);
    // Truncate aggressively.
    bytes.truncate(7);
    let err = decode(&bytes);
    assert!(err.is_err(), "decode should reject truncated stream");
}

#[test]
fn f1_header_bytes_decode_to_pinned_values() {
    // Bytes 0x00..=0x0A of F1 (luckynight.shn) per spec/02 §6.
    let f1_prefix: [u8; 11] = [
        0x61, 0x6A, 0x6B, 0x67, 0x02, 0xFB, 0xB1, 0x70, 0x09, 0xF9, 0x25,
    ];
    let header = crate::parse_header(&f1_prefix).expect("parse F1 header");
    assert_eq!(header.version, 2);
    assert_eq!(header.filetype, Filetype::S16Le);
    assert_eq!(header.channels, 2);
    assert_eq!(header.blocksize, 256);
    assert_eq!(header.max_lpc_order, 0);
    assert_eq!(header.mean_blocks, 4);
    assert_eq!(header.skip_bytes, 0);
    assert_eq!(header.header_end_bit, 43);
}

#[test]
fn bitshift_and_zero_blocks_decode() {
    let diff_block: Vec<i32> = (0i32..8).map(|i| i * 3).collect();
    let bytes = encode_with_bitshift_and_zero(Filetype::S16Le, 1, 8, 3, 2, &diff_block);
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.final_bshift, 3);
    // Output: 2 ZERO blocks (8 zero samples each) + 1 DIFF1 block of 8.
    // The shift of 3 multiplies all samples by 8.
    assert_eq!(decoded.samples_per_channel, 24);
    // First 16 samples are zero (from ZERO blocks); ZERO block samples
    // are unaffected by bshift since they are 0 << 3 = 0.
    for &s in &decoded.samples[..16] {
        assert_eq!(s, 0);
    }
    // Trailing 8 samples are diff_block << 3.
    let expected_tail: Vec<i32> = diff_block.iter().map(|&v| v << 3).collect();
    assert_eq!(&decoded.samples[16..], expected_tail.as_slice());
}

#[test]
fn qlpc_order_1_decodes_byte_exact() {
    // QLPC with a 1-tap predictor: a single coefficient of 1 is the
    // identity predictor (s(t) = s(t-1)), equivalent to DIFF1.
    let block: Vec<i32> = (0i32..8).map(|i| i * 5).collect();
    let bytes = encode_qlpc_block(Filetype::S16Le, 1, 8, 4, &[1], std::slice::from_ref(&block));
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples, block);
}

#[test]
fn qlpc_order_2_decodes_byte_exact() {
    // QLPC with a 2-tap predictor [2, -1] computes
    // ŝ(t) = 2 * s(t-1) - s(t-2), i.e. the DIFF2 polynomial form.
    let block: Vec<i32> = (0i32..16).map(|i| (i - 8).pow(2)).collect();
    let bytes = encode_qlpc_block(
        Filetype::S16Le,
        1,
        16,
        4,
        &[2, -1],
        std::slice::from_ref(&block),
    );
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples, block);
}

#[test]
fn qlpc_stereo_independent_channels() {
    let ch0: Vec<i32> = (0i32..8).map(|i| i * 11).collect();
    let ch1: Vec<i32> = (0i32..8).map(|i| -i * 7).collect();
    let bytes = encode_qlpc_block(Filetype::S16Le, 2, 8, 4, &[1], &[ch0.clone(), ch1.clone()]);
    let decoded = decode(&bytes).expect("decode");
    // Output is interleaved per channel.
    let mut expected = Vec::new();
    for i in 0..8 {
        expected.push(ch0[i]);
        expected.push(ch1[i]);
    }
    assert_eq!(decoded.samples, expected);
}

#[test]
fn empty_stream_after_header_quit_only() {
    // Build a stream whose only command is QUIT — no samples at all.
    let bytes = encode_minimal(Filetype::S16Le, 1, 4, &[], 1, &[]);
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(decoded.samples.len(), 0);
    assert_eq!(decoded.samples_per_channel, 0);
}
