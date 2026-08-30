#![no_main]

//! SHNAMPSK seek-table trailer detection on arbitrary tails, plus the
//! top-down / bottom-up boundary agreement on composites.
//!
//! `spec/05` §5 pins only the trailer's *envelope*: the file's last 12
//! bytes are a little-endian `len_u32` followed by the 8-byte
//! `SHNAMPSK` signature, and the sidecar of `len_u32` bytes begins
//! with the `SEEK` magic at `len(file) - len_u32`. Contract under test:
//!
//! 1. [`oxideav_shorten::detect_shnampsk_trailer`] and
//!    [`oxideav_shorten::split_off_shnampsk_trailer`] never panic.
//! 2. Signature present ⇒ the result is `Ok(Some)` or
//!    `Err(MalformedShnampskTrailer)`, never `Ok(None)`; signature
//!    absent ⇒ `Ok(None)`.
//! 3. `split_off` agrees with `detect`: the stream slice ends at
//!    `sidecar_start` and the sidecar slice is exactly `sidecar_len`
//!    bytes starting with `SEEK`.
//! 4. **Composite mode** (`data[0] & 1 == 1`): a valid stream is
//!    encoded from the fuzz bytes and a well-formed sidecar of
//!    fuzz-chosen length is appended; the detector's top-down
//!    `sidecar_start` must equal the decoder's bottom-up
//!    `stream_proper_len` (`spec/05` §5.2 three-way agreement), and the
//!    split stream must decode identically to the un-suffixed stream.

use libfuzzer_sys::fuzz_target;
use oxideav_shorten::{
    decode_stream, detect_shnampsk_trailer, encode_stream, split_off_shnampsk_trailer, Error,
    ShortenStreamHeader, StreamDecoder, SEEK_MAGIC, SHNAMPSK_SIGNATURE, TRAILER_TAIL_LEN,
};

fn check_raw(bytes: &[u8]) {
    let detected = detect_shnampsk_trailer(bytes);
    let split = split_off_shnampsk_trailer(bytes);
    // The detector is documented conservative: a file of `<=
    // TRAILER_TAIL_LEN` bytes cannot admit a trailer (there is no room
    // for the 4-byte `len_u32` before the signature, let alone a
    // stream), so a signature-suffixed short file is still `Ok(None)`.
    let has_sig = bytes.len() > TRAILER_TAIL_LEN && bytes[bytes.len() - 8..] == SHNAMPSK_SIGNATURE;
    match (&detected, &split) {
        (Ok(None), Ok((stream, None))) => {
            assert!(!has_sig, "signature present but reported absent");
            assert_eq!(stream.len(), bytes.len());
        }
        (Ok(Some(t)), Ok((stream, Some(sidecar)))) => {
            assert!(has_sig);
            assert_eq!(stream.len(), t.sidecar_start);
            assert_eq!(sidecar.len(), t.sidecar_len as usize);
            assert_eq!(stream.len() + sidecar.len(), bytes.len());
            assert!(sidecar.starts_with(&SEEK_MAGIC));
            assert_eq!(sidecar[4], t.seek_format_version);
            assert!(sidecar.len() >= SEEK_MAGIC.len() + TRAILER_TAIL_LEN);
        }
        (Err(Error::MalformedShnampskTrailer), Err(Error::MalformedShnampskTrailer)) => {
            assert!(has_sig, "malformed-trailer error without the signature");
        }
        other => panic!("detect / split disagree: {other:?}"),
    }
}

fuzz_target!(|data: &[u8]| {
    check_raw(data);
    if data.len() < 6 || data[0] & 1 == 0 {
        return;
    }
    // Composite: encoded stream + fuzz-shaped sidecar.
    let channels = 1 + (data[1] % 3) as u32;
    let blocksize = 1 + (data[2] % 64) as u32;
    let sidecar_body_len = u16::from_le_bytes([data[3], data[4]]) as usize % 4096;
    let seek_version = data[5];
    let n = channels as usize;
    let mut samples: Vec<i32> = data[6..].iter().map(|&b| b as i8 as i32 * 37).collect();
    samples.truncate(samples.len() - samples.len() % n);
    let header = ShortenStreamHeader {
        version: 2,
        filetype: 5,
        channels,
        blocksize,
        maxlpcorder: 0,
        meanblocks: 0,
        skipbytes: 0,
    };
    let stream = encode_stream(&header, &samples, &[]).expect("valid stream encodes");
    let clean = decode_stream(&stream).expect("valid stream decodes");
    assert_eq!(clean.stream_proper_len, stream.len());

    let mut file = stream.clone();
    let start = file.len();
    file.extend_from_slice(&SEEK_MAGIC);
    file.push(seek_version);
    for i in 0..sidecar_body_len {
        file.push(data[i % data.len()].wrapping_mul(i as u8 | 1));
    }
    let sidecar_len = (file.len() - start + TRAILER_TAIL_LEN) as u32;
    file.extend_from_slice(&sidecar_len.to_le_bytes());
    file.extend_from_slice(&SHNAMPSK_SIGNATURE);

    check_raw(&file);
    let t = detect_shnampsk_trailer(&file)
        .expect("well-formed composite detects")
        .expect("well-formed composite has a trailer");
    assert_eq!(t.sidecar_start, start, "top-down boundary equals the QUIT boundary");
    assert_eq!(t.sidecar_len, sidecar_len);
    assert_eq!(t.seek_format_version, seek_version);

    // The decoder ignores the trailer and reports the same boundary.
    let with_trailer = decode_stream(&file).expect("trailer is ignored by decode_stream");
    assert_eq!(with_trailer.stream_proper_len, start);
    assert_eq!(with_trailer.channels, clean.channels);
    let mut it = StreamDecoder::new(&file).expect("streaming opens");
    while it.next_block().expect("streaming decodes past-trailer stream").is_some() {}
    assert_eq!(it.trailer_len(), Some(sidecar_len as usize));

    let (s, side) = split_off_shnampsk_trailer(&file).expect("split");
    assert_eq!(s, &stream[..]);
    assert_eq!(side.map(|x| x.len()), Some(sidecar_len as usize));
});
