//! Round 12 integration tests for the envelope-only encoder.
//!
//! The encoder side of `docs/audio/shorten/spec/01..04` lands in
//! round 12 as a partial surface: file-header + verbatim + QUIT only.
//! Per-block predictor commands (`BLOCK_FN_DIFFn` / `BLOCK_FN_QLPC` /
//! `BLOCK_FN_BLOCKSIZE` / `BLOCK_FN_BITSHIFT` / `BLOCK_FN_ZERO`) are
//! out of scope for this round; they require an encoder-side parameter
//! search the spec narrative pins for the decode direction only.
//!
//! These tests exercise the round-trip across the encode/decode
//! boundary: encode a synthetic envelope-only stream from a chosen
//! header + verbatim prefix, decode it back via the round-7
//! [`decode_stream`] driver, and assert the recovered header / verbatim
//! / channel population match the encoder's input exactly.
//!
//! The encoder is the round-12 reference oracle for the wire-format
//! parsing of the round-1 header parser and the round-2 verbatim /
//! quit dispatch.

use oxideav_shorten::{
    decode_stream, encode_envelope_stream, parse_stream_header, write_byte_aligned_prefix,
    write_parameter_block, write_quit_command, write_stream_header, write_verbatim_block,
    BitWriter, EncodeError, ShortenStreamHeader,
};

fn synth_header(
    version: u8,
    filetype: u32,
    channels: u32,
    blocksize: u32,
    maxlpcorder: u32,
    meanblocks: u32,
    skipbytes: u32,
) -> ShortenStreamHeader {
    ShortenStreamHeader {
        version,
        filetype,
        channels,
        blocksize,
        maxlpcorder,
        meanblocks,
        skipbytes,
    }
}

#[test]
fn envelope_roundtrip_f1_like_header() {
    // F1's header values per spec/05 §6 + spec/02 §6:
    //   H_filetype = 5 (s16lh), H_channels = 2, H_blocksize = 256,
    //   H_maxlpcorder = 0,  H_meanblocks = 4, H_skipbytes = 0.
    let h = synth_header(2, 5, 2, 256, 0, 4, 0);
    let bytes = encode_envelope_stream(&h, &[]).expect("encode");
    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.header, h);
    assert_eq!(dec.channels.len(), 2);
    for ch in &dec.channels {
        assert!(ch.is_empty(), "no sample commands -> empty channels");
    }
    assert!(dec.verbatim.is_empty());
}

#[test]
fn envelope_roundtrip_three_pinned_filetypes() {
    for &ft in &[2u32, 3, 5] {
        let h = synth_header(2, ft, 1, 256, 0, 0, 0);
        let bytes = encode_envelope_stream(&h, &[]).expect("encode");
        let dec = decode_stream(&bytes).expect("decode");
        assert_eq!(dec.header.filetype, ft);
        assert_eq!(dec.header, h);
    }
}

#[test]
fn envelope_roundtrip_recovers_44_byte_wav_preamble() {
    // The 44-byte WAV preamble F1 carries per spec/02 §4.5 / §6.7.
    let preamble: &[u8] = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\
                            \x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\
                            \x02\x00\x10\x00data\x00\x00\x00\x00";
    assert_eq!(preamble.len(), 44);
    let h = synth_header(2, 5, 1, 256, 0, 0, 44);
    let bytes = encode_envelope_stream(&h, preamble).expect("encode");
    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.verbatim, preamble.to_vec());
    assert_eq!(dec.header, h);
}

#[test]
fn envelope_roundtrip_handles_64_byte_aifc_preamble() {
    // F3-style 64-byte AIFC preamble.
    let preamble: Vec<u8> = (0..64u8).collect();
    let h = synth_header(2, 3, 2, 256, 0, 4, 64);
    let bytes = encode_envelope_stream(&h, &preamble).expect("encode");
    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.verbatim, preamble);
    assert_eq!(dec.header, h);
}

#[test]
fn envelope_roundtrip_extreme_parameter_widths() {
    // Force the encoder to pick wider `ulong()` mantissa widths for
    // each field; the decoder should recover identical values.
    let h = synth_header(2, 9, 6, 4096, 12, 16, 1024);
    let bytes = encode_envelope_stream(&h, &[]).expect("encode");
    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.header, h);
}

#[test]
fn header_only_writer_roundtrips_via_parse_stream_header() {
    // `write_stream_header` alone (without a QUIT terminator) is
    // sufficient for the header-parser API.
    let h = synth_header(2, 5, 2, 256, 0, 4, 0);
    let mut out = Vec::new();
    let bits_after = write_stream_header(&mut out, &h).expect("write");
    // The parser needs at least one trailing byte to read past the
    // parameter block boundary.
    out.push(0);
    let parsed = parse_stream_header(&out).expect("parse");
    assert_eq!(parsed.header, h);
    assert_eq!(parsed.bits_consumed_after_v, bits_after);
}

#[test]
fn manual_assembly_via_primitives_decodes_identically() {
    // Build the same stream by composing the lower-level primitives
    // directly, rather than calling `encode_envelope_stream`. The
    // decoded form must match.
    let h = synth_header(2, 5, 1, 256, 0, 0, 0);
    let payload = b"abc";

    // Via the high-level driver.
    let via_driver = encode_envelope_stream(&h, payload).expect("encode");

    // Via the primitives.
    let mut via_primitives = Vec::new();
    write_byte_aligned_prefix(&mut via_primitives, h.version).expect("prefix");
    let mut bw = BitWriter::new();
    write_parameter_block(&mut bw, &h);
    write_verbatim_block(&mut bw, payload).expect("verbatim");
    write_quit_command(&mut bw);
    bw.pad_to_byte();
    via_primitives.extend(bw.into_bytes());

    assert_eq!(
        via_driver, via_primitives,
        "high-level driver and primitive composition produce identical bytes"
    );

    let dec = decode_stream(&via_primitives).expect("decode");
    assert_eq!(dec.header, h);
    assert_eq!(dec.verbatim, payload.to_vec());
}

#[test]
fn unsupported_version_rejected_at_prefix_stage() {
    let h = synth_header(0, 5, 1, 256, 0, 0, 0);
    let res = encode_envelope_stream(&h, &[]);
    assert_eq!(res, Err(EncodeError::UnsupportedVersion(0)));
}

#[test]
fn envelope_output_byte_size_is_small_for_minimal_header() {
    // Mono / minimal envelope: the encoded byte count should be a
    // small constant (a handful of bytes) — the parameter block fits
    // in a few bytes when all values are at their minimum widths.
    let h = synth_header(2, 5, 1, 256, 0, 0, 0);
    let bytes = encode_envelope_stream(&h, &[]).expect("encode");
    // 5 bytes magic + version + a handful of body bytes.
    assert!(bytes.len() >= 6);
    assert!(bytes.len() <= 16);
}
