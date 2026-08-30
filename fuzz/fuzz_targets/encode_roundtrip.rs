#![no_main]

//! Structure-aware encode → decode round trip.
//!
//! Every parameter the fuzzer derives is mapped into the encoder's
//! **contract-valid** domain, so `encode_stream` / `encode_stream_lossy`
//! MUST return `Ok` and any `expect` below fires only on a real
//! encoder / decoder defect — never on an input the contract rejects.
//!
//! ## Fuzz input layout
//!
//! ```text
//!   byte 0     : version        → 2..=3 (the encoder's accepted set;
//!                                 bit 7 set additionally asserts that
//!                                 version 1 is rejected up front)
//!   byte 1     : channels       → 1..=8
//!   bytes 2-3  : blocksize      → 1..=2048 (LE u16 masked)
//!   byte 4     : maxlpcorder    → 0..=40 (spans the MAX_QLPC_AUTO_ORDER
//!                                 clamp at 32)
//!   byte 5     : meanblocks     → 0..=8
//!   byte 6     : filetype       → raw byte (the encoder writes any u32;
//!                                 the decoder does not interpret it)
//!   byte 7     : bit depth      → 1..=24; samples are sign-extended from
//!                                 this many bits so residuals fit the
//!                                 decoder's 30-bit residual cap
//!   byte 8     : bshift         → 0..=31 (BITSHIFT_MAX); 0 = lossless
//!   byte 9     : verbatim len   → 0..=64 bytes taken from the payload
//!   byte 10    : skipbytes      → raw byte
//!   bytes 11.. : verbatim prefix bytes, then interleaved sample bytes
//!                (3 per sample, LE, sign-extended to `bit depth`);
//!                capped at MAX_SAMPLES total
//! ```
//!
//! ## Contract under test
//!
//! 1. `encode_stream` (bshift = 0) / `encode_stream_lossy` return `Ok`.
//! 2. `decode_stream` on the bytes returns `Ok` with the same header,
//!    the same verbatim prefix, and per-channel samples equal to
//!    `(s >> bshift) << bshift` (exact when `bshift = 0`).
//! 3. The streaming iterator reconstructs the identical planes.
//! 4. `stream_proper_len == bytes.len()` (no trailer) and the QUIT
//!    padding is `spec/05` §4-conformant.

use libfuzzer_sys::fuzz_target;
use oxideav_shorten::{
    decode_stream, encode_stream, encode_stream_lossy, DecodedBlock, EncodeError,
    ShortenStreamHeader, StreamDecoder,
};

const MAX_SAMPLES: usize = 16 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() < 11 {
        return;
    }
    let version = 2 + (data[0] % 2);
    let channels = 1 + (data[1] % 8) as u32;
    let blocksize = 1 + (u16::from_le_bytes([data[2], data[3]]) % 2048) as u32;
    let maxlpcorder = (data[4] % 41) as u32;
    let meanblocks = (data[5] % 9) as u32;
    let filetype = data[6] as u32;
    let depth = 1 + (data[7] % 24) as u32;
    let bshift = (data[8] % 32) as u32;
    let verbatim_len = (data[9] % 65) as usize;
    let skipbytes = data[10] as u32;

    let mut rest = &data[11..];
    let verbatim_len = verbatim_len.min(rest.len());
    let verbatim = &rest[..verbatim_len];
    rest = &rest[verbatim_len..];

    let n = channels as usize;
    let mut samples: Vec<i32> = rest
        .chunks_exact(3)
        .map(|c| {
            let raw = u32::from_le_bytes([c[0], c[1], c[2], 0]);
            let masked = raw & ((1u32 << depth) - 1);
            // Sign-extend from `depth` bits.
            ((masked << (32 - depth)) as i32) >> (32 - depth)
        })
        .take(MAX_SAMPLES)
        .collect();
    samples.truncate(samples.len() - samples.len() % n);

    let header = ShortenStreamHeader {
        version,
        filetype,
        channels,
        blocksize,
        maxlpcorder,
        meanblocks,
        skipbytes,
    };

    if data[0] & 0x80 != 0 {
        // Version 1's layout is unpinned and the decoder rejects it, so
        // the encoder must refuse rather than emit undecodable bytes.
        let v1 = ShortenStreamHeader {
            version: 1,
            ..header
        };
        assert_eq!(
            encode_stream(&v1, &samples, verbatim).err(),
            Some(EncodeError::UnsupportedVersion(1))
        );
    }

    let bytes = if bshift == 0 {
        encode_stream(&header, &samples, verbatim).expect("contract-valid input encodes")
    } else {
        encode_stream_lossy(&header, &samples, verbatim, bshift)
            .expect("contract-valid lossy input encodes")
    };

    let expected: Vec<Vec<i32>> = (0..n)
        .map(|c| {
            samples
                .iter()
                .skip(c)
                .step_by(n)
                .map(|&s| if bshift == 0 { s } else { (s >> bshift) << bshift })
                .collect()
        })
        .collect();

    let dec = decode_stream(&bytes).expect("encoder output decodes");
    assert_eq!(dec.header, header, "header round-trips");
    assert_eq!(dec.verbatim, verbatim, "verbatim prefix round-trips");
    assert_eq!(dec.channels, expected, "samples round-trip");
    assert_eq!(dec.stream_proper_len, bytes.len(), "no trailer on encoder output");
    assert!(dec.quit_padding.is_spec_conformant(), "QUIT padding is spec/05 §4 conformant");

    let mut it = StreamDecoder::new(&bytes).expect("streaming path opens");
    let mut planes: Vec<Vec<i32>> = vec![Vec::new(); n];
    let mut vb = Vec::new();
    while let Some(block) = it.next_block().expect("streaming path decodes") {
        match block {
            DecodedBlock::Samples { channel, samples } => planes[channel].extend(samples),
            DecodedBlock::Verbatim { bytes } => vb.extend(bytes),
        }
    }
    assert_eq!(planes, expected, "streaming samples round-trip");
    assert_eq!(vb, verbatim);
});
