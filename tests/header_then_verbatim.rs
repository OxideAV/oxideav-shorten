//! Integration test exercising the header parse + per-block command
//! dispatch as a single pipeline.
//!
//! The behavioural anchor is `docs/audio/shorten/spec/02-variable-
//! length-coding.md` §6 + §4.1 + §4.5: the header parameter block
//! ends at bit 43 (relative to byte `0x05`) of fixture `F1`, and the
//! per-block command stream's very first command — starting at that
//! bit — is `BLOCK_FN_VERBATIM = 9` carrying a 44-byte payload that
//! reproduces the input WAV file's leading 44 bytes (test `T6`).
//!
//! This test builds a synthetic v2 stream from scratch:
//!   1. Header bytes for `H_channels = 2, H_blocksize = 256, H_filetype = 5,
//!      H_maxlpcorder = 0, H_meanblocks = 4, H_skipbytes = 0` (the
//!      same values fixture F1's header decodes to, per `spec/02`
//!      §6).
//!   2. A `BLOCK_FN_VERBATIM` command carrying a 3-byte payload
//!      `[0xAA, 0xBB, 0xCC]`.
//!   3. A `BLOCK_FN_QUIT` sentinel.
//!
//! The pipeline parses the header, hands the post-header byte slice
//! to a fresh `BitReader`, advances by the residual bit position
//! reported by `ParsedHeader::bits_consumed_after_v` modulo 8 (the
//! header's bit position is reported relative to byte `0x05`; the
//! per-block command stream's reader must skip the same number of
//! bits inside its first byte), reads one function code, decodes
//! the verbatim payload, then reads the QUIT sentinel.
//!
//! Both reads must succeed bit-exactly. This is the smallest
//! end-to-end test that covers everything rounds 1 + 2 of the
//! clean-room rebuild have wired up.

use oxideav_shorten::{
    parse_stream_header, read_function_code, read_verbatim_payload, BitReader, FunctionCode,
    FNSIZE, MAGIC, VERBATIM_BYTE_SIZE, VERBATIM_CHUNK_SIZE,
};

fn pack_bits_msb_first(bits: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut byte = 0u8;
    let mut n = 0u32;
    for &b in bits {
        byte = (byte << 1) | (b as u8 & 1);
        n += 1;
        if n == 8 {
            out.push(byte);
            byte = 0;
            n = 0;
        }
    }
    if n > 0 {
        out.push(byte << (8 - n));
    }
    out
}

fn encode_uvar(value: u32, n: u32) -> Vec<u32> {
    if n == 0 {
        let mut bits = vec![0u32; value as usize];
        bits.push(1);
        bits
    } else {
        let span = 1u32 << n;
        let prefix_zeros = value / span;
        let mantissa = value % span;
        let mut bits = vec![0u32; prefix_zeros as usize];
        bits.push(1);
        for i in (0..n).rev() {
            bits.push((mantissa >> i) & 1);
        }
        bits
    }
}

fn encode_ulong(value: u32, w: u32) -> Vec<u32> {
    // Width-of-mantissa first (ULONGSIZE = 2), then mantissa.
    let mut bits = Vec::new();
    bits.extend(encode_uvar(w, 2));
    bits.extend(encode_uvar(value, w));
    bits
}

#[test]
fn header_then_verbatim_then_quit_round_trips_end_to_end() {
    // Mirror fixture F1's header field choices (spec/02 §6):
    //   H_filetype  = 5
    //   H_channels  = 2
    //   H_blocksize = 256
    //   H_maxlpcorder = 0
    //   H_meanblocks  = 4
    //   H_skipbytes   = 0
    // with widths w = 3, 2, 9, 0, 3, 0 (which is exactly the
    // sequence of widths fixture F1's encoder picked per spec/02
    // §6.1..6.6). This yields a 43-bit header parameter block,
    // ending at bit 43 relative to byte 0x05 (per spec/02 §6.7).
    let mut header_bits = Vec::new();
    header_bits.extend(encode_ulong(5, 3));
    header_bits.extend(encode_ulong(2, 2));
    header_bits.extend(encode_ulong(256, 9));
    header_bits.extend(encode_ulong(0, 0));
    header_bits.extend(encode_ulong(4, 3));
    header_bits.extend(encode_ulong(0, 0));
    assert_eq!(header_bits.len(), 43);

    // First post-header command: BLOCK_FN_VERBATIM with a 3-byte
    // payload.
    let verbatim_payload: [u8; 3] = [0xAA, 0xBB, 0xCC];
    let mut block_bits = Vec::new();
    block_bits.extend(encode_uvar(9, FNSIZE)); // VERBATIM = 9
    block_bits.extend(encode_uvar(
        verbatim_payload.len() as u32,
        VERBATIM_CHUNK_SIZE,
    ));
    for &b in verbatim_payload.iter() {
        block_bits.extend(encode_uvar(b as u32, VERBATIM_BYTE_SIZE));
    }

    // Second post-header command: BLOCK_FN_QUIT.
    block_bits.extend(encode_uvar(4, FNSIZE));

    // Assemble the full bit stream (header bits then block bits)
    // and pack into bytes.
    let mut all_bits = Vec::with_capacity(header_bits.len() + block_bits.len());
    all_bits.extend(&header_bits);
    all_bits.extend(&block_bits);
    let body = pack_bits_msb_first(&all_bits);

    // Prepend the byte-aligned magic + version byte.
    let mut buf = Vec::with_capacity(5 + body.len());
    buf.extend_from_slice(&MAGIC);
    buf.push(2); // version
    buf.extend_from_slice(&body);

    // Stage 1: header parse.
    let parsed = parse_stream_header(&buf).expect("synthetic header must parse");
    assert_eq!(parsed.header.version, 2);
    assert_eq!(parsed.header.filetype, 5);
    assert_eq!(parsed.header.channels, 2);
    assert_eq!(parsed.header.blocksize, 256);
    assert_eq!(parsed.header.maxlpcorder, 0);
    assert_eq!(parsed.header.meanblocks, 4);
    assert_eq!(parsed.header.skipbytes, 0);
    assert_eq!(parsed.bits_consumed_after_v, 43);

    // Stage 2: open a fresh BitReader over the post-version bytes
    // and skip the 43 header bits to align with the block stream.
    //
    // The header parser's internal reader has been dropped; we
    // re-seek to the post-header position by constructing a new
    // reader and explicitly burning the header bits.
    let post_version = &buf[5..];
    let mut reader = BitReader::new(post_version);
    let _ = reader
        .read_bits(32)
        .expect("burn header bits 0..31 (covers all 32-bit-wide chunks)");
    let _ = reader
        .read_bits(parsed.bits_consumed_after_v - 32)
        .expect("burn remaining header bits");

    // Stage 3: read the first block command — VERBATIM.
    let fc = read_function_code(&mut reader).expect("first command code must classify");
    assert_eq!(fc, FunctionCode::Verbatim);
    let chunk = read_verbatim_payload(&mut reader).expect("verbatim payload must decode");
    assert_eq!(chunk.bytes.as_slice(), verbatim_payload.as_slice());

    // Stage 4: read the second block command — QUIT.
    let fc2 = read_function_code(&mut reader).expect("second command code must classify");
    assert_eq!(fc2, FunctionCode::Quit);
}
