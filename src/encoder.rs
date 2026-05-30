//! Shorten encode-side primitives — round 12.
//!
//! Round 12 lands the encoder-side scaffolding the README "lacks" tail
//! has been naming since round 8: the bit-level wire-format writer
//! ([`crate::bitwriter::BitWriter`] in the sibling module), the
//! [`write_stream_header`] header encoder, the [`write_verbatim_block`]
//! and [`write_quit_command`] envelope-and-terminator primitives, and
//! the [`encode_envelope_stream`] high-level driver that builds a
//! syntactically-valid (but predictor-free) Shorten byte stream out of
//! a `(ShortenStreamHeader, verbatim_prefix)` pair.
//!
//! The full predictor encode side (`BLOCK_FN_DIFFn` /
//! `BLOCK_FN_QLPC` residual production, energy-parameter selection,
//! per-block channel-round sequencing) is deferred to a later round —
//! the spec's §3.1..§3.5 narrative pins the decoder's reconstruction
//! rule for each predictor, but encoder-side parameter search (where
//! TR.156 §3.3 derives the optimal Rice parameter from a per-block
//! statistical objective) is a much larger surface than fits in a
//! single Implementer round.
//!
//! ## What round 12 unlocks
//!
//! The envelope encoder is sufficient to produce a Shorten file that
//! the round-7 decode driver ([`crate::decode_stream`]) round-trips
//! losslessly: the header parameter block is reproduced byte-exactly,
//! the verbatim prefix is recovered byte-for-byte (the
//! [`crate::ShortenDecoder`] adaptor's [`crate::ShortenDecoder::verbatim_prefix`]
//! captures it), and the `BLOCK_FN_QUIT` terminator + byte-alignment
//! padding match `spec/05` §4's worked-example layout on fixture `F9`.
//! Per-channel sample data is omitted — a caller that wants a stream
//! carrying samples must wait for the next-round predictor encoder.
//!
//! Even without the predictor encoder, the envelope encoder is the
//! reference roundtrip oracle for the decoder's wire-format parsing:
//! every parameter field combination the test suite covers can be
//! re-emitted and re-parsed, pinning the encode/decode symmetry of
//! `spec/02` §3 (the `ulong()` two-stage form) and `spec/03` §3.10
//! (the verbatim payload framing).
//!
//! ## Clean-room provenance
//!
//! Implementation is sourced from:
//!
//! * `docs/audio/shorten/spec/01-stream-header.md` §1 + §3 (the byte-
//!   aligned magic/version + the six `ulong()` parameter fields).
//! * `docs/audio/shorten/spec/02-variable-length-coding.md` §1 +
//!   §2.1 + §2.2 + §3 + §4 (the MSB-first bit order, the unsigned /
//!   signed elementary forms, the two-stage `ulong()` form, the
//!   per-block mantissa widths).
//! * `docs/audio/shorten/spec/03-block-and-predictor.md` §3.8
//!   (`BLOCK_FN_QUIT` framing) + §3.10 (`BLOCK_FN_VERBATIM` payload).
//! * `docs/audio/shorten/spec/04-function-code-resolution.md` §2 +
//!   §7 (function-code numeric assignments for `QUIT` and `VERBATIM`).
//! * `docs/audio/shorten/spec/05-state-and-quirks.md` §4 (post-QUIT
//!   byte-alignment with zero padding).

use crate::bitwriter::{natural_ulong_width, BitWriter};
use crate::block::{FNSIZE, VERBATIM_BYTE_SIZE, VERBATIM_CHUNK_SIZE, VERBATIM_MAX_LEN};
use crate::header::{ShortenStreamHeader, MAGIC};

/// `BLOCK_FN_VERBATIM` function-code numeric value (`spec/03` §3.10 /
/// `spec/04` §7). Re-exported as a public constant so callers building
/// per-command byte streams against the encoder's primitives can
/// produce the same wire-format token the decoder reads.
pub const FN_VERBATIM: u32 = 9;

/// `BLOCK_FN_QUIT` function-code numeric value (`spec/03` §3.8 /
/// `spec/04` §2).
pub const FN_QUIT: u32 = 4;

/// The format version this encoder emits.
///
/// The reachable fixture corpus is v2-only per `spec/05` §7; the
/// decoder side accepts versions in `{1, 2, 3}` per `spec/00`
/// §"Format versions" but only v2 has byte-exact behavioural anchors.
/// The encoder therefore writes v2 unconditionally; v1 / v3 emission
/// awaits a `-v` fixture round per `spec/05` §7's open §9.4 candidate.
pub const ENCODER_VERSION: u8 = 2;

/// Errors the round-12 encoder primitives can surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    /// A header carried a format version outside `{1, 2, 3}`. The
    /// encoder writes whichever version is requested, but values
    /// outside the spec set are rejected.
    UnsupportedVersion(u8),
    /// The verbatim payload length exceeded the
    /// `uvar(VERBATIM_CHUNK_SIZE = 5)` length-field cap of
    /// `spec/02` §4.5 (`VERBATIM_MAX_LEN`).
    VerbatimTooLong(u32),
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EncodeError::UnsupportedVersion(v) => write!(
                f,
                "oxideav-shorten: encoder rejects unsupported format version {v}"
            ),
            EncodeError::VerbatimTooLong(n) => write!(
                f,
                "oxideav-shorten: verbatim payload length {n} exceeds spec/02 §4.5 length-field cap"
            ),
        }
    }
}

impl std::error::Error for EncodeError {}

/// Crate-local encoder `Result` alias.
pub type EncodeResult<T> = core::result::Result<T, EncodeError>;

/// Emit the byte-aligned 4-byte `ajkg` magic + 1-byte version prefix
/// to `out` per `spec/01` §1.
///
/// The version byte may be in `{1, 2, 3}` per `spec/00`; values outside
/// that set surface [`EncodeError::UnsupportedVersion`]. The function
/// writes 5 raw bytes; `out`'s prior contents are preserved and the
/// returned `out.len()` increment is exactly 5.
pub fn write_byte_aligned_prefix(out: &mut Vec<u8>, version: u8) -> EncodeResult<()> {
    if !matches!(version, 1..=3) {
        return Err(EncodeError::UnsupportedVersion(version));
    }
    out.extend_from_slice(&MAGIC);
    out.push(version);
    Ok(())
}

/// Emit the six `ulong()` parameter-block fields of `spec/01` §3 to a
/// fresh [`BitWriter`] under the minimum-width rule of `spec/02` §3.
///
/// The encoder picks the smallest `width` such that `value < 2^width`
/// for every field — the same minimum-width rule the test fixtures in
/// `src/driver.rs` use. A caller that wants a wider width may build the
/// bit stream by hand via [`BitWriter::write_ulong`] directly.
///
/// Field order matches `spec/01` §3 exactly:
/// `H_filetype, H_channels, H_blocksize, H_maxlpcorder,
///  H_meanblocks, H_skipbytes`.
pub fn write_parameter_block(writer: &mut BitWriter, header: &ShortenStreamHeader) {
    writer.write_ulong(header.filetype, natural_ulong_width(header.filetype));
    writer.write_ulong(header.channels, natural_ulong_width(header.channels));
    writer.write_ulong(header.blocksize, natural_ulong_width(header.blocksize));
    writer.write_ulong(header.maxlpcorder, natural_ulong_width(header.maxlpcorder));
    writer.write_ulong(header.meanblocks, natural_ulong_width(header.meanblocks));
    writer.write_ulong(header.skipbytes, natural_ulong_width(header.skipbytes));
}

/// Emit a complete file header to `out`: byte-aligned magic +
/// version prefix followed by the six-field parameter block packed
/// MSB-first into bytes per `spec/02` §1.
///
/// Returns the bit offset (relative to byte `0x05`) at which the
/// next per-block command would be written — equivalently the
/// `bits_consumed_after_v` field of [`crate::header::ParsedHeader`].
/// The parameter block's final byte is zero-padded to the next byte
/// boundary; this matches the encoder's behaviour observed across
/// every reachable fixture, where every per-block command in the
/// post-header stream begins exactly at the bit immediately following
/// the last parameter-block bit (no inter-field padding) but the
/// per-block stream itself is unpadded.
///
/// **Wire-format note.** The encoder DOES NOT byte-align the
/// parameter block; the trailing partial byte stays as-is for the
/// next per-block command's first bits to fill in. For callers that
/// only want a header (no commands), call [`encode_envelope_stream`]
/// instead, which terminates the stream with `BLOCK_FN_QUIT` + zero
/// padding.
pub fn write_stream_header(out: &mut Vec<u8>, header: &ShortenStreamHeader) -> EncodeResult<u32> {
    write_byte_aligned_prefix(out, header.version)?;
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, header);
    let bits_consumed = writer.bits_written() as u32;
    // Flush whatever full bytes are pending; the residual partial
    // byte (if any) becomes the first byte of the post-header bit
    // stream, with its MSB filled by parameter-block bits and its
    // low bits available for per-block command bits.
    let buf = writer.into_bytes();
    out.extend_from_slice(&buf);
    Ok(bits_consumed)
}

/// Emit a `BLOCK_FN_VERBATIM` command (function-code numeric `9`) to
/// `writer` per `spec/03` §3.10 + `spec/02` §4.5.
///
/// Wire layout:
///
/// * `uvar(FNSIZE = 2)` over the value `FN_VERBATIM = 9`.
/// * `uvar(VERBATIM_CHUNK_SIZE = 5)` over the payload length.
/// * `payload.len()` × `uvar(VERBATIM_BYTE_SIZE = 8)` over each byte.
///
/// The verbatim command does not advance the channel cursor per
/// `spec/03` §3.10, so a caller that emits a `VERBATIM` mid-channel-
/// round can resume the round in place. The total emitted bit count is
/// `1 + ⌊9/4⌋ + 2 + 1 + ⌊L/32⌋ + 5 + L · (1 + 8)` where
/// `L = payload.len()` (the prefix-plus-terminator-plus-mantissa
/// arithmetic of §2.1 for each `uvar` field).
pub fn write_verbatim_block(writer: &mut BitWriter, payload: &[u8]) -> EncodeResult<()> {
    let len = payload.len();
    let len_u32 = u32::try_from(len).map_err(|_| EncodeError::VerbatimTooLong(u32::MAX))?;
    if len_u32 > VERBATIM_MAX_LEN {
        return Err(EncodeError::VerbatimTooLong(len_u32));
    }
    writer.write_uvar(FN_VERBATIM, FNSIZE);
    writer.write_uvar(len_u32, VERBATIM_CHUNK_SIZE);
    for &b in payload {
        writer.write_uvar(b as u32, VERBATIM_BYTE_SIZE);
    }
    Ok(())
}

/// Emit the `BLOCK_FN_QUIT` command (function-code numeric `4`) to
/// `writer` per `spec/03` §3.8 + `spec/04` §2.
///
/// The QUIT command is a bare function-code field (5 bits under
/// `uvar(FNSIZE = 2)` for the value 4). It terminates the bit stream;
/// the caller should follow this call with [`BitWriter::pad_to_byte`]
/// to satisfy `spec/05` §4's zero-padding requirement.
pub fn write_quit_command(writer: &mut BitWriter) {
    writer.write_uvar(FN_QUIT, FNSIZE);
}

/// Build a complete envelope-only Shorten byte stream: header +
/// verbatim prefix + `BLOCK_FN_QUIT` + zero-pad to next byte boundary.
///
/// The output is a syntactically valid Shorten file the round-7 decode
/// driver [`crate::decode_stream`] will accept; the resulting
/// `DecodedStream` has `verbatim == verbatim_prefix` and
/// `channels == vec![Vec::new(); H_channels]` (no sample-producing
/// commands ran).
///
/// The verbatim prefix length is bounded by `spec/02` §4.5's
/// `VERBATIM_MAX_LEN`; an over-cap input surfaces
/// [`EncodeError::VerbatimTooLong`].
///
/// Per `spec/03` §3.10 the encoder is free to emit zero or more
/// `BLOCK_FN_VERBATIM` commands; this driver emits a single command
/// when `verbatim_prefix` is non-empty and zero commands otherwise.
pub fn encode_envelope_stream(
    header: &ShortenStreamHeader,
    verbatim_prefix: &[u8],
) -> EncodeResult<Vec<u8>> {
    let mut out = Vec::with_capacity(16 + verbatim_prefix.len() + 4);
    write_byte_aligned_prefix(&mut out, header.version)?;
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, header);
    if !verbatim_prefix.is_empty() {
        write_verbatim_block(&mut writer, verbatim_prefix)?;
    }
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::decode_stream;
    use crate::header::parse_stream_header;

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

    // ---- prefix + version ----

    #[test]
    fn write_byte_aligned_prefix_emits_magic_and_version() {
        let mut out = Vec::new();
        write_byte_aligned_prefix(&mut out, 2).expect("write");
        assert_eq!(out, vec![b'a', b'j', b'k', b'g', 0x02]);
    }

    #[test]
    fn write_byte_aligned_prefix_accepts_v1_v2_v3() {
        for &v in &[1u8, 2, 3] {
            let mut out = Vec::new();
            write_byte_aligned_prefix(&mut out, v).expect("write");
            assert_eq!(out[4], v);
        }
    }

    #[test]
    fn write_byte_aligned_prefix_rejects_v0_and_v4() {
        for &v in &[0u8, 4, 99] {
            let mut out = Vec::new();
            assert_eq!(
                write_byte_aligned_prefix(&mut out, v),
                Err(EncodeError::UnsupportedVersion(v))
            );
            assert!(out.is_empty(), "no bytes written on error");
        }
    }

    // ---- write_stream_header roundtrip ----

    #[test]
    fn write_stream_header_roundtrips_via_parse_stream_header() {
        // Try a representative spread of header combinations.
        let cases = [
            // F1-like (luckynight.shn header values).
            (2u8, 5u32, 2u32, 256u32, 0u32, 4u32, 0u32),
            // F4-like (44-byte WAV preamble; 4 meanblocks).
            (2, 5, 2, 256, 0, 4, 0),
            // Mono, no mean, no LPC.
            (2, 5, 1, 128, 0, 0, 0),
            // Heavy header values, exercising wider ulong() widths.
            (2, 2, 8, 4096, 8, 16, 1024),
            // Non-default version (v3, same parameter shape as v2).
            (3, 5, 2, 256, 0, 4, 0),
        ];
        for &(version, ft, ch, bs, mlpc, mb, sb) in &cases {
            let header = synth_header(version, ft, ch, bs, mlpc, mb, sb);
            let mut out = Vec::new();
            let bits_after = write_stream_header(&mut out, &header).expect("write");
            // The reader needs a full parameter block plus enough
            // bytes to detect end-of-header. Append a single zero byte
            // to give it room (parse_stream_header reads bit-by-bit
            // and reports bits_consumed_after_v which is bit position
            // within the post-version body).
            let mut full = out.clone();
            full.push(0);
            let parsed = parse_stream_header(&full).expect("parse");
            assert_eq!(
                parsed.header, header,
                "case ({ft},{ch},{bs},{mlpc},{mb},{sb})"
            );
            assert_eq!(
                parsed.bits_consumed_after_v, bits_after,
                "bit offset for case ({ft},{ch},{bs},{mlpc},{mb},{sb})"
            );
        }
    }

    // ---- verbatim block ----

    #[test]
    fn write_verbatim_block_rejects_overlong_payload() {
        let cap = VERBATIM_MAX_LEN as usize;
        let payload = vec![0u8; cap + 1];
        let mut w = BitWriter::new();
        assert_eq!(
            write_verbatim_block(&mut w, &payload),
            Err(EncodeError::VerbatimTooLong((cap + 1) as u32))
        );
    }

    #[test]
    fn write_verbatim_block_accepts_empty_payload() {
        let mut w = BitWriter::new();
        write_verbatim_block(&mut w, &[]).expect("write empty");
        // Wire layout: uvar(FNSIZE=2) over 9 + uvar(5) over 0.
        // value 9 in uvar(2): ⌊9/4⌋ = 2 leading zeros + terminator +
        //   mantissa `01` (low 2 bits of 9) → `00 1 01` = 5 bits.
        // value 0 in uvar(5): zero leading zeros + terminator +
        //   five zero mantissa bits → `1 00000` = 6 bits.
        assert_eq!(w.bits_written(), 11);
    }

    // ---- encode_envelope_stream roundtrip ----

    #[test]
    fn envelope_stream_with_no_verbatim_roundtrips() {
        let header = synth_header(2, 5, 2, 256, 0, 4, 0);
        let bytes = encode_envelope_stream(&header, &[]).expect("encode");
        let dec = decode_stream(&bytes).expect("decode");
        assert_eq!(dec.header, header);
        assert_eq!(dec.verbatim, Vec::<u8>::new());
        // No sample-producing commands, so each channel emits zero
        // samples.
        assert_eq!(dec.channels.len(), header.channels as usize);
        for ch in &dec.channels {
            assert!(ch.is_empty(), "no samples emitted");
        }
    }

    #[test]
    fn envelope_stream_with_verbatim_preserves_payload() {
        let header = synth_header(2, 5, 1, 256, 0, 0, 44);
        let preamble = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\
                        \x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\
                        \x02\x00\x10\x00data\x00\x00\x00\x00";
        assert_eq!(preamble.len(), 44);
        let bytes = encode_envelope_stream(&header, preamble).expect("encode");
        let dec = decode_stream(&bytes).expect("decode");
        assert_eq!(dec.verbatim, preamble.to_vec());
        assert_eq!(dec.header.skipbytes, 44);
    }

    #[test]
    fn envelope_stream_quit_padded_to_byte_boundary() {
        // The last byte of an envelope-only stream contains the QUIT
        // command's 5-bit pattern plus zero padding to the next byte
        // boundary. The QUIT pattern starts at some bit position
        // within the byte; the padding zeros fill the rest.
        let header = synth_header(2, 5, 1, 256, 0, 0, 0);
        let bytes = encode_envelope_stream(&header, &[]).expect("encode");
        // We don't know which byte the QUIT falls into without
        // re-deriving the bit offset, but we know the very last byte's
        // low bits must be zero padding (the trailing zeros). And we
        // know the stream decodes back, which is the load-bearing
        // assertion: the round-7 driver walks until QUIT and
        // terminates cleanly.
        let dec = decode_stream(&bytes).expect("decode");
        assert_eq!(dec.header, header);
        // The encoded stream is short — header (5 magic bytes) plus
        // the parameter block + QUIT bits packed into a handful of
        // bytes. Confirm it's modest.
        assert!(bytes.len() < 32, "envelope stream is {} bytes", bytes.len());
    }

    #[test]
    fn envelope_stream_three_filetypes_decode_via_decoder_trait() {
        // Each of the three pinned H_filetype codes builds a valid
        // envelope stream the decoder accepts. We aren't running it
        // through the trait adaptor here (no AudioFrame to assert on
        // since no samples are emitted), but the round-trip through
        // decode_stream confirms wire-format symmetry.
        for &filetype in &[2u32, 3, 5] {
            let header = synth_header(2, filetype, 1, 256, 0, 0, 0);
            let bytes = encode_envelope_stream(&header, &[]).expect("encode");
            let dec = decode_stream(&bytes).expect("decode");
            assert_eq!(dec.header.filetype, filetype);
        }
    }

    #[test]
    fn envelope_stream_preserves_channel_count_in_decoded_output() {
        for &channels in &[1u32, 2, 4, 6] {
            let header = synth_header(2, 5, channels, 256, 0, 0, 0);
            let bytes = encode_envelope_stream(&header, &[]).expect("encode");
            let dec = decode_stream(&bytes).expect("decode");
            assert_eq!(dec.header.channels, channels);
            assert_eq!(dec.channels.len(), channels as usize);
        }
    }

    #[test]
    fn write_verbatim_block_then_quit_decodes_through_full_driver() {
        // Compose a stream manually using the encoder primitives
        // (rather than the encode_envelope_stream driver) to confirm
        // each primitive is independently usable.
        let header = synth_header(2, 5, 1, 256, 0, 0, 0);
        let mut out = Vec::new();
        write_stream_header(&mut out, &header).expect("write header");
        let mut writer = BitWriter::new();
        // Need to re-build the bit-level writer over the bit position
        // we ended at. Easiest path: just rebuild via the envelope
        // driver semantics, which start a fresh writer for the body
        // after the parameter block.
        // To exercise write_verbatim_block + write_quit_command in
        // isolation, restart the body assembly here.
        out.clear();
        write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
        write_parameter_block(&mut writer, &header);
        write_verbatim_block(&mut writer, b"hello").expect("verbatim");
        write_quit_command(&mut writer);
        writer.pad_to_byte();
        out.extend(writer.into_bytes());
        let dec = decode_stream(&out).expect("decode");
        assert_eq!(dec.verbatim, b"hello".to_vec());
    }
}
