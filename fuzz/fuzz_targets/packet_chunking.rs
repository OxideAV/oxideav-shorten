#![no_main]

//! The two `oxideav_core::Decoder` wrappers fed the same stream across
//! fuzz-chosen packet boundaries must agree with `decode_stream`.
//!
//! `ShortenDecoder` (whole-stream: buffers until `decode_stream`
//! succeeds) and `ShortenStreamingDecoder` (chop-anywhere: emits one
//! frame per channel round as soon as it completes) both take a
//! `Packet` sequence. Contract under test:
//!
//! 1. Neither wrapper panics on arbitrary bytes at arbitrary packet
//!    boundaries; `receive_frame` after `flush` terminates with `Eof`
//!    or an error, never spins.
//! 2. For a **valid** stream (composite mode, `data[0] & 1 == 1`, built
//!    from the fuzz bytes with a pinned `s16lh` file type) both wrappers
//!    reconstruct the same per-channel planes as `decode_stream`, packed
//!    little-endian `i16`, regardless of how the bytes were chunked, and
//!    report the same `stream_proper_len` / `quit_padding`.
//!
//! ## Fuzz input layout (composite mode)
//!
//! ```text
//!   byte 0     : bit 0 = composite flag; bits 1..3 = channels - 1 (1..=4)
//!   byte 1     : blocksize 1..=64
//!   byte 2     : chunk-size seed (0 = one byte per packet)
//!   bytes 3..  : 16-bit LE samples
//! ```
//!
//! In raw mode (flag clear) the bytes themselves are the stream and the
//! same chunking rule applies; only the panic-free / termination
//! contract is asserted.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{
    CodecId, CodecParameters, Decoder, Error as CoreError, Frame, Packet, TimeBase,
};
use oxideav_shorten::{
    decode_stream, encode_stream, make_decoder, make_streaming_decoder, ShortenStreamHeader,
    CODEC_ID_STR, FILETYPE_S16LH,
};

const MAX_FRAMES: usize = 1 << 20;

fn feed(dec: &mut Box<dyn Decoder>, bytes: &[u8], chunk: usize) -> Result<Vec<Vec<u8>>, CoreError> {
    let tb = TimeBase::new(1, 44_100);
    let mut planes: Vec<Vec<u8>> = Vec::new();
    let mut frames = 0usize;
    let mut drain = |dec: &mut Box<dyn Decoder>, planes: &mut Vec<Vec<u8>>| -> Result<(), CoreError> {
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    frames += 1;
                    assert!(frames <= MAX_FRAMES, "frame flood");
                    if planes.is_empty() {
                        planes.resize(a.data.len(), Vec::new());
                    }
                    assert_eq!(planes.len(), a.data.len(), "channel count changed mid-stream");
                    for (p, d) in planes.iter_mut().zip(a.data.iter()) {
                        p.extend_from_slice(d);
                    }
                }
                Ok(other) => panic!("non-audio frame {other:?}"),
                Err(CoreError::NeedMore) | Err(CoreError::Eof) => return Ok(()),
                Err(e) => return Err(e),
            }
        }
    };
    let chunk = chunk.max(1);
    for (i, piece) in bytes.chunks(chunk).enumerate() {
        dec.send_packet(&Packet::new(0, tb, piece.to_vec()).with_pts(i as i64))?;
        drain(dec, &mut planes)?;
    }
    dec.flush()?;
    drain(dec, &mut planes)?;
    // After flush the decoder must terminate.
    match dec.receive_frame() {
        Err(CoreError::Eof) | Err(_) => {}
        Ok(_) => panic!("frame produced after flush drained"),
    }
    // Normalise the zero-sample representations: the whole-stream
    // wrapper emits one frame of empty planes for a valid
    // zero-sample stream while the streaming wrapper emits no frame
    // at all. Both mean "no PCM"; compare them as such.
    if planes.iter().all(|p| p.is_empty()) {
        planes.clear();
    }
    Ok(planes)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let composite = data[0] & 1 == 1;
    let chunk = data[2] as usize;
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));

    let (stream, expected): (Vec<u8>, Option<Vec<Vec<u8>>>) = if composite {
        let channels = 1 + ((data[0] >> 1) & 3) as u32;
        let blocksize = 1 + (data[1] % 64) as u32;
        let n = channels as usize;
        let mut samples: Vec<i32> = data[3..]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as i32)
            .collect();
        samples.truncate(samples.len() - samples.len() % n);
        let header = ShortenStreamHeader {
            version: 2,
            filetype: FILETYPE_S16LH,
            channels,
            blocksize,
            maxlpcorder: (data[1] % 4) as u32,
            meanblocks: (data[1] % 3) as u32,
            skipbytes: 0,
        };
        params.channels = Some(channels as u16);
        let bytes = encode_stream(&header, &samples, &[]).expect("valid stream encodes");
        let dec = decode_stream(&bytes).expect("valid stream decodes");
        let mut planes: Vec<Vec<u8>> = dec
            .channels
            .iter()
            .map(|ch| ch.iter().flat_map(|&s| (s as i16).to_le_bytes()).collect())
            .collect();
        // Zero-sample normalisation, mirroring `feed` (a valid
        // zero-sample stream yields no PCM under either wrapper).
        if planes.iter().all(|p| p.is_empty()) {
            planes.clear();
        }
        (bytes, Some(planes))
    } else {
        (data[3..].to_vec(), None)
    };

    let mut whole = make_decoder(&params).expect("make_decoder");
    let mut streaming = make_streaming_decoder(&params).expect("make_streaming_decoder");
    let a = feed(&mut whole, &stream, chunk);
    let b = feed(&mut streaming, &stream, chunk);
    if let Some(expected) = expected {
        let a = a.expect("whole-stream wrapper decodes a valid stream");
        let b = b.expect("streaming wrapper decodes a valid stream");
        assert_eq!(a, expected, "whole-stream wrapper planes");
        assert_eq!(b, expected, "streaming wrapper planes");
    } else {
        // Raw mode: both wrappers must agree on success vs failure.
        assert_eq!(a.is_ok(), b.is_ok(), "wrappers disagree on validity: {a:?} vs {b:?}");
        if let (Ok(a), Ok(b)) = (a, b) {
            assert_eq!(a, b, "wrappers disagree on planes");
        }
    }
});
