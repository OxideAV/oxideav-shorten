//! `SHNAMPSK`-tagged seek-table trailer detector.
//!
//! Several Shorten files distributed publicly carry an extra
//! non-standard trailer appended after the encoder's own end-of-stream
//! byte alignment. The trailer is **not** part of the Shorten wire
//! format produced by the encoder; it is an external sidecar added by
//! Wayne Stielau's seek-table utility. For decoder implementations
//! that consume publicly distributed `.shn` fixtures, identifying this
//! trailer is necessary so a caller can either ignore the trailing
//! bytes, hand them to a seek-table parser, or surface them as an
//! out-of-band annotation.
//!
//! ## Trailer structure (`spec/05` §5.1)
//!
//! The trailer's last 12 bytes have the layout:
//!
//! | Offset (from EOF) | Length | Content                                  |
//! | ----------------- | ------ | ---------------------------------------- |
//! | `-12`             | 4      | Sidecar total length, little-endian `u32`. |
//! | `-8`              | 8      | ASCII signature `SHNAMPSK`.              |
//!
//! The sidecar's start offset within the file is therefore
//! `len(file) − len_u32`. At that offset the sidecar begins with the
//! 4-byte ASCII magic `SEEK` (hex `53 45 45 4B`) followed by
//! sidecar-internal fields (version byte, a stored copy of the
//! SHN-stream-proper length, and a sequence of seek-record entries).
//! Per `spec/05` §5 the seek-record entries' field schema is external
//! to the Shorten wire format and out of scope here.
//!
//! ## Decoder behaviour (`spec/05` §5.2)
//!
//! A standards-compliant Shorten decoder consumes the bit stream up to
//! and including the `BLOCK_FN_QUIT` command and its zero-bit padding
//! to the next byte boundary. Bytes after that boundary are out of
//! scope for the wire format itself; the decoder may ignore them, may
//! parse them as a `SHNAMPSK`-tagged seek table for seek-related
//! functionality, or may diagnose them as a non-standard appendage.
//! This module supplies the detector; the existing `decode_stream`
//! driver of `spec/03` §3.8 already stops at `BLOCK_FN_QUIT` so the
//! trailer (if present) does not affect decode of the integer-PCM
//! samples.
//!
//! ## Verification anchor (`spec/05` §5.3)
//!
//! Fixtures `F1..F8` carry the trailer; fixture `F9` (and the
//! structurally-identical `Choppy.shn`) do not. The unit tests exercise
//! both the present-trailer and absent-trailer paths against synthetic
//! byte buffers shaped per the §5.1 layout table; the `len_u32` sanity
//! window matches what the §5.1 + `SEEK` magic together imply.

use crate::error::{Error, Result};

/// ASCII signature occupying the trailer's last 8 bytes
/// (`spec/05` §5.1 row 2).
pub const SHNAMPSK_SIGNATURE: [u8; 8] = *b"SHNAMPSK";

/// ASCII magic at the sidecar's start offset
/// (`spec/05` §5.1 narrative — `len(file) − len_u32` points at this).
pub const SEEK_MAGIC: [u8; 4] = *b"SEEK";

/// Trailer length in bytes (4-byte LE `len_u32` + 8-byte signature)
/// (`spec/05` §5.1).
pub const TRAILER_TAIL_LEN: usize = 12;

/// Minimum number of bytes a sidecar can occupy: the 4-byte `SEEK`
/// magic at the sidecar's start plus the 12-byte trailer tail
/// (`spec/05` §5.1).
pub const MIN_SIDECAR_LEN: usize = SEEK_MAGIC.len() + TRAILER_TAIL_LEN;

/// Maximum sidecar length the detector accepts. The §5.1 narrative
/// does not pin a numeric cap; we set the cap to 16 MiB so that
/// detecting a `SHNAMPSK` signature whose preceding `len_u32` reports
/// an obviously-absurd value (e.g. the bytes happen to spell
/// `SHNAMPSK` by coincidence inside a malformed file) does not
/// confuse a caller into reading past the SHN stream proper. The
/// publicly-distributed seek-table corpus the spec was reconstructed
/// against carries sidecars well below this cap.
pub const SIDECAR_LEN_CAP: u32 = 16 * 1024 * 1024;

/// Parsed `SHNAMPSK`-tagged trailer record.
///
/// `sidecar_start` is the byte offset within the original file at
/// which the sidecar begins (i.e. where the `SEEK` magic sits). The
/// SHN stream proper occupies `bytes[..sidecar_start]`; the sidecar
/// (including its trailing `SHNAMPSK` signature) occupies
/// `bytes[sidecar_start..]`.
///
/// `sidecar_len` is the value read out of the trailer's 4-byte
/// little-endian length field; it equals `bytes.len() -
/// sidecar_start` for a well-formed trailer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShnampskTrailer {
    /// Offset of the first byte of the sidecar (the `S` of `SEEK`)
    /// within the original file bytes.
    pub sidecar_start: usize,
    /// Sidecar length in bytes, as read from the trailer's 4-byte LE
    /// length field. Equals `bytes.len() - sidecar_start` for a
    /// well-formed trailer.
    pub sidecar_len: u32,
}

/// Try to detect a `SHNAMPSK`-tagged seek-table sidecar at the tail of
/// `bytes`.
///
/// Returns:
///
/// * `Ok(Some(trailer))` if the last 8 bytes spell `SHNAMPSK` AND the
///   preceding 4-byte little-endian `len_u32` decodes to a value
///   `LEN ∈ [MIN_SIDECAR_LEN, min(SIDECAR_LEN_CAP, bytes.len())]` AND
///   the bytes at offset `bytes.len() - LEN` begin with the `SEEK`
///   magic. This is the well-formed-trailer case pinned by `spec/05`
///   §5.1.
/// * `Ok(None)` if the last 8 bytes do not spell `SHNAMPSK`, or the
///   total file length is too short to even hold a trailer
///   (`bytes.len() < TRAILER_TAIL_LEN + 1` ⇒ no SHN-stream-proper
///   byte left over). This matches fixture `F9` (and the structurally
///   identical `Choppy.shn`), per `spec/05` §5.3.
/// * `Err(Error::MalformedShnampskTrailer)` if the `SHNAMPSK`
///   signature is present but the `len_u32` window is outside the
///   `[MIN_SIDECAR_LEN, min(SIDECAR_LEN_CAP, bytes.len())]` range or
///   the bytes at the computed sidecar start do not begin with the
///   `SEEK` magic — the trailer is structurally invalid and the
///   caller should be told rather than silently treated as "no
///   trailer present".
///
/// The detector does not touch the SHN stream proper and does not
/// invoke the wire-format decoder; callers that want the integer-PCM
/// samples should drive [`crate::decode_stream`] separately. Per
/// `spec/05` §5.2 the SHN wire format itself terminates at
/// `BLOCK_FN_QUIT`'s zero-bit padding, so the bytes the detector
/// strips off the tail never overlap with the bytes the wire-format
/// decoder reads.
pub fn detect_shnampsk_trailer(bytes: &[u8]) -> Result<Option<ShnampskTrailer>> {
    if bytes.len() <= TRAILER_TAIL_LEN {
        // Even a zero-byte SHN stream proper plus a 12-byte trailer
        // totals 12 bytes, but the spec requires at least the
        // 5-byte `ajkg` + version header (spec/01) plus a non-trivial
        // body to precede any sidecar. We're conservative and return
        // None for any file too short to admit a trailer at all.
        return Ok(None);
    }

    let sig_off = bytes.len() - SHNAMPSK_SIGNATURE.len();
    if bytes[sig_off..] != SHNAMPSK_SIGNATURE[..] {
        return Ok(None);
    }

    // SHNAMPSK signature present — read the 4-byte LE length field
    // immediately preceding it.
    let len_off = bytes.len() - TRAILER_TAIL_LEN;
    let mut len_bytes = [0u8; 4];
    len_bytes.copy_from_slice(&bytes[len_off..len_off + 4]);
    let sidecar_len = u32::from_le_bytes(len_bytes);

    // Validate the reported sidecar length against the file size and
    // against the implementation safety cap.
    if (sidecar_len as usize) < MIN_SIDECAR_LEN {
        return Err(Error::MalformedShnampskTrailer);
    }
    if sidecar_len > SIDECAR_LEN_CAP {
        return Err(Error::MalformedShnampskTrailer);
    }
    if (sidecar_len as usize) > bytes.len() {
        return Err(Error::MalformedShnampskTrailer);
    }

    let sidecar_start = bytes.len() - sidecar_len as usize;
    if bytes[sidecar_start..sidecar_start + SEEK_MAGIC.len()] != SEEK_MAGIC[..] {
        return Err(Error::MalformedShnampskTrailer);
    }

    Ok(Some(ShnampskTrailer {
        sidecar_start,
        sidecar_len,
    }))
}

/// Split `bytes` into `(shn_stream_proper, sidecar_opt)` per the
/// `spec/05` §5.1 trailer layout.
///
/// On a well-formed trailer this returns
/// `(bytes[..trailer.sidecar_start], Some(bytes[trailer.sidecar_start..]))`.
/// On no trailer it returns `(bytes, None)`. On a malformed trailer
/// (signature present but `len_u32` / `SEEK` magic inconsistent) the
/// detector's error is propagated; the caller can choose to fall back
/// to `(bytes, None)` if it would rather treat malformed trailers as
/// "no trailer present" but the detector itself does not make that
/// choice silently.
///
/// Convenient wrapper around [`detect_shnampsk_trailer`] for callers
/// that want to hand the SHN stream proper to
/// [`crate::decode_stream`] without having to compute the slice
/// themselves.
pub fn split_off_shnampsk_trailer(bytes: &[u8]) -> Result<(&[u8], Option<&[u8]>)> {
    match detect_shnampsk_trailer(bytes)? {
        Some(t) => Ok((&bytes[..t.sidecar_start], Some(&bytes[t.sidecar_start..]))),
        None => Ok((bytes, None)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesise a well-formed `SHNAMPSK` trailer plus a body of
    /// arbitrary opaque sidecar bytes per `spec/05` §5.1.
    ///
    /// Layout:
    ///   [shn_proper] [SEEK] [body_bytes] [len_u32 LE] [SHNAMPSK]
    /// The `len_u32` value spans from `SEEK` through `SHNAMPSK` (the
    /// sidecar's total length per the §5.1 narrative).
    fn build_with_trailer(shn_proper: &[u8], body_bytes: &[u8]) -> Vec<u8> {
        let sidecar_len = SEEK_MAGIC.len() + body_bytes.len() + TRAILER_TAIL_LEN;
        assert!(sidecar_len <= u32::MAX as usize);
        let mut out = Vec::with_capacity(shn_proper.len() + sidecar_len);
        out.extend_from_slice(shn_proper);
        out.extend_from_slice(&SEEK_MAGIC);
        out.extend_from_slice(body_bytes);
        out.extend_from_slice(&(sidecar_len as u32).to_le_bytes());
        out.extend_from_slice(&SHNAMPSK_SIGNATURE);
        out
    }

    #[test]
    fn no_trailer_when_file_shorter_than_trailer_tail() {
        // A 5-byte file (ajkg + version) cannot hold a 12-byte trailer.
        let bytes = [b'a', b'j', b'k', b'g', 2];
        assert_eq!(detect_shnampsk_trailer(&bytes).unwrap(), None);
    }

    #[test]
    fn no_trailer_when_signature_absent() {
        // 64 bytes of opaque content, no SHNAMPSK at the tail.
        let bytes = vec![0xABu8; 64];
        assert_eq!(detect_shnampsk_trailer(&bytes).unwrap(), None);
    }

    #[test]
    fn detects_minimal_well_formed_trailer() {
        // shn_proper = 5 bytes; sidecar = 16 bytes (4 SEEK + 0 body + 12 tail).
        let shn = vec![b'a', b'j', b'k', b'g', 2];
        let file = build_with_trailer(&shn, &[]);
        let t = detect_shnampsk_trailer(&file).unwrap().unwrap();
        assert_eq!(t.sidecar_start, 5);
        assert_eq!(t.sidecar_len, 16);
        assert_eq!(t.sidecar_len as usize, MIN_SIDECAR_LEN);
    }

    #[test]
    fn detects_trailer_with_body_bytes() {
        // 100 bytes of body between SEEK and len_u32.
        let shn = vec![0u8; 200];
        let body = vec![0x42u8; 100];
        let file = build_with_trailer(&shn, &body);
        let t = detect_shnampsk_trailer(&file).unwrap().unwrap();
        assert_eq!(t.sidecar_start, 200);
        // 4 (SEEK) + 100 (body) + 12 (tail) = 116
        assert_eq!(t.sidecar_len, 116);
        assert_eq!(file.len(), 200 + 116);
    }

    #[test]
    fn split_off_returns_shn_proper_and_sidecar_slices() {
        let shn = vec![0xAAu8; 50];
        let body = vec![0xBBu8; 30];
        let file = build_with_trailer(&shn, &body);
        let (proper, sidecar) = split_off_shnampsk_trailer(&file).unwrap();
        assert_eq!(proper.len(), 50);
        assert!(proper.iter().all(|&b| b == 0xAA));
        let sc = sidecar.expect("sidecar present");
        // sidecar starts with SEEK
        assert_eq!(&sc[..4], &SEEK_MAGIC);
        // sidecar ends with SHNAMPSK
        assert_eq!(&sc[sc.len() - 8..], &SHNAMPSK_SIGNATURE);
        // sidecar length matches the len_u32 field
        let mut len_le = [0u8; 4];
        len_le.copy_from_slice(&sc[sc.len() - 12..sc.len() - 8]);
        assert_eq!(u32::from_le_bytes(len_le) as usize, sc.len());
    }

    #[test]
    fn split_off_returns_no_sidecar_when_signature_absent() {
        let file = vec![0u8; 64];
        let (proper, sidecar) = split_off_shnampsk_trailer(&file).unwrap();
        assert_eq!(proper.len(), 64);
        assert!(sidecar.is_none());
    }

    #[test]
    fn malformed_when_len_field_too_small() {
        // Signature present but len_u32 < MIN_SIDECAR_LEN.
        let mut file = vec![0u8; 32];
        // Set the last 8 bytes to SHNAMPSK and the 4 bytes before to
        // a tiny len.
        let n = file.len();
        file[n - 12..n - 8].copy_from_slice(&3u32.to_le_bytes());
        file[n - 8..].copy_from_slice(&SHNAMPSK_SIGNATURE);
        let err = detect_shnampsk_trailer(&file).unwrap_err();
        assert_eq!(err, Error::MalformedShnampskTrailer);
    }

    #[test]
    fn malformed_when_len_field_exceeds_file_size() {
        let mut file = vec![0u8; 32];
        let n = file.len();
        // len_u32 = 1_000_000 ≫ file.len()
        file[n - 12..n - 8].copy_from_slice(&1_000_000u32.to_le_bytes());
        file[n - 8..].copy_from_slice(&SHNAMPSK_SIGNATURE);
        let err = detect_shnampsk_trailer(&file).unwrap_err();
        assert_eq!(err, Error::MalformedShnampskTrailer);
    }

    #[test]
    fn malformed_when_len_field_exceeds_safety_cap() {
        // File is large enough to contain the reported len but the
        // len exceeds the implementation safety cap.
        let n = SIDECAR_LEN_CAP as usize + 100;
        let mut file = vec![0u8; n];
        let bogus_len = SIDECAR_LEN_CAP + 1;
        file[n - 12..n - 8].copy_from_slice(&bogus_len.to_le_bytes());
        file[n - 8..].copy_from_slice(&SHNAMPSK_SIGNATURE);
        // Don't bother making SEEK valid — we should fail on the cap.
        let err = detect_shnampsk_trailer(&file).unwrap_err();
        assert_eq!(err, Error::MalformedShnampskTrailer);
    }

    #[test]
    fn malformed_when_seek_magic_missing_at_computed_offset() {
        // Build a "well-formed" trailer, then overwrite the SEEK
        // magic with garbage. The signature + len pass; the SEEK
        // anchor check should reject.
        let shn = vec![0u8; 16];
        let body = vec![0u8; 8];
        let mut file = build_with_trailer(&shn, &body);
        // Sidecar starts at offset 16; overwrite SEEK with zeros.
        file[16..20].copy_from_slice(&[0u8; 4]);
        let err = detect_shnampsk_trailer(&file).unwrap_err();
        assert_eq!(err, Error::MalformedShnampskTrailer);
    }

    #[test]
    fn shnampsk_signature_constant_is_correct_ascii() {
        assert_eq!(&SHNAMPSK_SIGNATURE, b"SHNAMPSK");
        assert_eq!(SHNAMPSK_SIGNATURE.len(), 8);
    }

    #[test]
    fn seek_magic_constant_is_correct_ascii() {
        assert_eq!(&SEEK_MAGIC, b"SEEK");
        assert_eq!(SEEK_MAGIC, [0x53, 0x45, 0x45, 0x4B]);
    }

    #[test]
    fn trailer_tail_len_constant_matches_layout() {
        // 4 bytes len_u32 + 8 bytes signature = 12.
        assert_eq!(TRAILER_TAIL_LEN, 4 + SHNAMPSK_SIGNATURE.len());
    }

    #[test]
    fn min_sidecar_len_constant_matches_layout() {
        // SEEK magic (4) + trailer tail (12) = 16.
        assert_eq!(MIN_SIDECAR_LEN, SEEK_MAGIC.len() + TRAILER_TAIL_LEN);
        assert_eq!(MIN_SIDECAR_LEN, 16);
    }

    #[test]
    fn split_off_returns_full_file_when_no_trailer_present() {
        let file = vec![0xCCu8; 100];
        let (proper, sidecar) = split_off_shnampsk_trailer(&file).unwrap();
        assert_eq!(proper, &file[..]);
        assert!(sidecar.is_none());
    }

    #[test]
    fn split_off_propagates_malformed_error() {
        // Signature present but len bogus — split_off should bubble
        // up the error, not silently degrade to (full file, None).
        let mut file = vec![0u8; 32];
        let n = file.len();
        file[n - 12..n - 8].copy_from_slice(&3u32.to_le_bytes());
        file[n - 8..].copy_from_slice(&SHNAMPSK_SIGNATURE);
        let res = split_off_shnampsk_trailer(&file);
        assert!(res.is_err());
    }

    #[test]
    fn detect_then_split_indices_are_consistent() {
        let shn = vec![0x11u8; 77];
        let body = vec![0x22u8; 13];
        let file = build_with_trailer(&shn, &body);
        let t = detect_shnampsk_trailer(&file).unwrap().unwrap();
        let (proper, sidecar) = split_off_shnampsk_trailer(&file).unwrap();
        assert_eq!(proper.len(), t.sidecar_start);
        assert_eq!(sidecar.unwrap().len(), t.sidecar_len as usize);
    }

    #[test]
    fn coincidental_shnampsk_at_tail_with_no_seek_anchor_rejects() {
        // A byte sequence whose last 8 bytes happen to spell SHNAMPSK
        // but whose len_u32 / SEEK pair is meaningless should be
        // surfaced as malformed (not silently treated as a valid
        // trailer the caller might then chop off).
        let mut file = vec![0xDDu8; 64];
        // len = 16 (well-formed-looking), but no SEEK at offset 48.
        let n = file.len();
        file[n - 12..n - 8].copy_from_slice(&16u32.to_le_bytes());
        file[n - 8..].copy_from_slice(&SHNAMPSK_SIGNATURE);
        let err = detect_shnampsk_trailer(&file).unwrap_err();
        assert_eq!(err, Error::MalformedShnampskTrailer);
    }
}
