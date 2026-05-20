//! Shorten stream-header parser.
//!
//! Implements the file-header layout pinned in
//! `docs/audio/shorten/spec/01-stream-header.md`:
//!
//! 1. The byte-addressable prefix at offsets `0x00..=0x04` — four
//!    bytes of ASCII `ajkg` magic (`spec/01` §1) and a one-byte
//!    format version field. Versions `1`, `2`, `3` are in scope per
//!    `spec/00` §"Format versions"; v0 is explicitly out of scope.
//! 2. The variable-length-integer parameter block beginning at file
//!    offset `0x05` (`spec/01` §3): six `ulong()` fields in the
//!    fixed order
//!
//!    | Order | Notation         | Role                                                |
//!    | ----- | ---------------- | --------------------------------------------------- |
//!    | 1     | `H_filetype`     | File-type code naming the sample encoding.          |
//!    | 2     | `H_channels`     | Channel count.                                      |
//!    | 3     | `H_blocksize`    | Default block size in samples per channel.          |
//!    | 4     | `H_maxlpcorder`  | Maximum LPC order; 0 disables LPC.                  |
//!    | 5     | `H_meanblocks`   | Mean-estimator block count (v2+; see note below).   |
//!    | 6     | `H_skipbytes`    | Verbatim-prefix length carried by the encoder.      |
//!
//! Round 1 hard-codes the six-field v2/v3 layout. v1's layout differs
//! (mean-estimator field is absent or implicit per `spec/01` §3.5,
//! flagged as open §9.4 candidate #2 in `spec/01` §7) and the v1
//! branch is rejected with `Error::UnsupportedVersion(1)` here until
//! a v1 fixture is reachable. v2 and v3 share the six-field layout
//! per the spec.
//!
//! The bit stream is read MSB-first from byte `0x05` per `spec/02`
//! §1. Each parameter is the `ulong()` two-stage form of `spec/02`
//! §3: `w = uvar(ULONGSIZE)`, `v = uvar(w)`.
//!
//! No per-block command stream is decoded in this round — the parser
//! returns the parsed header plus the bit position at which the
//! per-block stream begins (`spec/02` §6.7 — fixture `F1` ends at
//! bit 43 relative to byte `0x05` = file bit offset 83).

use crate::bitreader::BitReader;
use crate::error::{Error, Result};

/// ASCII `ajkg` — pinned by `spec/01` §1 (LOC FDD + behavioural test
/// `T2` on fixture `F1`).
pub const MAGIC: [u8; 4] = *b"ajkg";

/// Minimum number of header bytes required by the byte-aligned
/// prefix alone (4-byte magic + 1-byte version). The bit-stream
/// parameter block consumes additional bytes from offset `0x05`
/// onward.
pub const MIN_HEADER_BYTES: usize = 5;

/// Parsed Shorten stream header — the six variable-length-integer
/// fields of `spec/01` §3 plus the byte-aligned magic + version
/// prefix of §1.
///
/// Round 1 names the v2/v3 layout. A v1 stream is rejected at parse
/// time with [`Error::UnsupportedVersion`] pending the v1-specific
/// branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShortenStreamHeader {
    /// Format version at file offset `0x04`. In-scope values: `1`,
    /// `2`, `3` per `spec/00` §"Format versions".
    pub version: u8,
    /// `H_filetype` — file-type code naming the sample encoding
    /// (`spec/01` §3.1). Three of the eleven labels are pinned by
    /// behavioural observation (`u8 = 2`, `s16hl = 3`, `s16lh = 5`,
    /// per `spec/05` §6 / `spec/02` §7). The remaining eight codes
    /// are open §9.4 candidate #1.
    pub filetype: u32,
    /// `H_channels` — number of interleaved channels (`spec/01`
    /// §3.2).
    pub channels: u32,
    /// `H_blocksize` — default samples-per-channel block size
    /// (`spec/01` §3.3). TR.156's default is 256.
    pub blocksize: u32,
    /// `H_maxlpcorder` — maximum LPC order considered by the
    /// encoder (`spec/01` §3.4). Zero disables LPC; only the fixed
    /// polynomial predictors of orders 0..3 (TR.156 §3.2 equations
    /// 3..6) appear in the per-block stream when this field is `0`.
    pub maxlpcorder: u32,
    /// `H_meanblocks` — number of past blocks the mean estimator
    /// averages over (`spec/01` §3.5). For v2/v3 streams the encoder
    /// defaults this to a non-zero value (TR.156 Appendix §"-m
    /// blocks").
    pub meanblocks: u32,
    /// `H_skipbytes` — count of bytes the encoder consumed verbatim
    /// from the input file before predictor encoding began
    /// (`spec/01` §3.6). The verbatim bytes themselves appear later
    /// in the bit stream as a `BLOCK_FN_VERBATIM` command rather
    /// than inline in the header.
    pub skipbytes: u32,
}

impl ShortenStreamHeader {
    /// Length in samples of the per-channel sample-history carry
    /// buffer the decoder will allocate after the header lands
    /// (`spec/01` §4). The buffer length is the maximum of the
    /// fixed-polynomial reach (3) and `H_maxlpcorder`. This quantity
    /// is **derived** at decode time; it is not transmitted on the
    /// wire.
    pub fn sample_history_carry_len(&self) -> u32 {
        core::cmp::max(3, self.maxlpcorder)
    }
}

/// Outcome of [`parse_stream_header`]: the parsed header plus the
/// bit-position immediately after the parameter block. The bit
/// position is reported relative to the start of byte `0x05` — i.e.,
/// the offset within the bit stream that begins after the byte-aligned
/// magic + version prefix. To convert to a file-bit offset add `40`
/// (5 bytes × 8 bits).
///
/// On fixture `F1` (`spec/02` §6.7) this is `43`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedHeader {
    /// The header fields themselves.
    pub header: ShortenStreamHeader,
    /// Bit-stream position at which the per-block command stream
    /// begins, measured from the start of byte `0x05`.
    pub bits_consumed_after_v: u32,
}

/// Parse the stream header from a byte buffer beginning at file
/// offset `0`. Returns the parsed fields plus the bit position at
/// which the per-block command stream of `spec/03` (not yet
/// implemented) begins.
///
/// Rejects:
///
/// * Buffers shorter than five bytes ([`Error::InvalidMagic`]).
/// * Buffers whose first four bytes are not the four-byte `ajkg`
///   magic ([`Error::InvalidMagic`]).
/// * Buffers whose version byte is outside `{1, 2, 3}`
///   ([`Error::UnsupportedVersion`]).
/// * Version-1 streams ([`Error::UnsupportedVersion`]). v1's
///   parameter-block layout differs from v2/v3 (the mean-estimator
///   field is absent or implicit per `spec/01` §3.5) and no v1
///   fixture is reachable through the spec set's allow-list.
/// * Buffers that truncate mid-bit-stream ([`Error::Truncated`]).
pub fn parse_stream_header(bytes: &[u8]) -> Result<ParsedHeader> {
    if bytes.len() < MIN_HEADER_BYTES {
        return Err(Error::InvalidMagic);
    }
    if bytes[0..4] != MAGIC {
        return Err(Error::InvalidMagic);
    }
    let version = bytes[4];
    if !matches!(version, 1..=3) {
        return Err(Error::UnsupportedVersion(version));
    }
    if version == 1 {
        // v1 layout is not pinned by the references in the spec set's
        // allow-list (open §9.4 candidate #2). Rejecting here keeps
        // the parser deterministic until a v1 fixture is reachable.
        return Err(Error::UnsupportedVersion(1));
    }

    // Six v2/v3 ulong() fields, MSB-first from byte 0x05.
    let bit_input = &bytes[5..];
    let total_bits_in_input = (bit_input.len() as u32).saturating_mul(8);
    let mut reader = BitReader::new(bit_input);

    let filetype = reader.read_ulong()?;
    let channels = reader.read_ulong()?;
    let blocksize = reader.read_ulong()?;
    let maxlpcorder = reader.read_ulong()?;
    let meanblocks = reader.read_ulong()?;
    let skipbytes = reader.read_ulong()?;

    let bits_consumed_after_v = reader.bits_consumed_so_far(total_bits_in_input);

    Ok(ParsedHeader {
        header: ShortenStreamHeader {
            version,
            filetype,
            channels,
            blocksize,
            maxlpcorder,
            meanblocks,
            skipbytes,
        },
        bits_consumed_after_v,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-packer helper: takes an MSB-first bit sequence and packs
    /// it into a `Vec<u8>`, left-justifying the trailing byte.
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

    /// Build a bit sequence encoding `ulong(value)` for a chosen
    /// mantissa width `w`. The first stage encodes `w` as
    /// `uvar(ULONGSIZE = 2)`; the second stage encodes `value` as
    /// `uvar(w)` (using a one-bit prefix terminator and `w` mantissa
    /// bits — sufficient when `value < 2^w`, which all the values
    /// the unit tests construct satisfy).
    fn encode_ulong(value: u32, w: u32) -> Vec<u32> {
        // Stage 1: uvar(2) for w. We pick the smallest legal
        // encoding — zero prefix-zeros + terminator + 2-bit mantissa
        // — by representing `w` directly in the 2-bit mantissa for
        // `w in 0..=3`, otherwise expanding the prefix.
        let mut bits = Vec::new();
        let prefix_zeros_w = w / 4;
        let mantissa_w = w % 4;
        bits.resize(bits.len() + prefix_zeros_w as usize, 0);
        bits.push(1); // terminator
        bits.push((mantissa_w >> 1) & 1);
        bits.push(mantissa_w & 1);

        // Stage 2: uvar(w) for value. Same logic with chosen w.
        if w == 0 {
            assert_eq!(value, 0, "uvar(0) can only encode 0");
            bits.push(1);
        } else {
            let span = 1u32 << w;
            let prefix_zeros_v = value / span;
            let mantissa_v = value % span;
            bits.resize(bits.len() + prefix_zeros_v as usize, 0);
            bits.push(1);
            for i in (0..w).rev() {
                bits.push((mantissa_v >> i) & 1);
            }
        }
        bits
    }

    /// Build a synthetic v2 stream header byte buffer encoding the
    /// six chosen header fields with the per-field widths used by
    /// fixture `F1` (`spec/02` §6).
    fn synth_v2_header(
        filetype: u32,
        channels: u32,
        blocksize: u32,
        maxlpcorder: u32,
        meanblocks: u32,
        skipbytes: u32,
    ) -> Vec<u8> {
        let mut bits = Vec::new();
        // Chosen widths mirror `F1`'s: w = 3, 2, 9, 0, 3, 0.
        bits.extend(encode_ulong(filetype, 3));
        bits.extend(encode_ulong(channels, 2));
        bits.extend(encode_ulong(blocksize, 9));
        bits.extend(encode_ulong(maxlpcorder, 0));
        bits.extend(encode_ulong(meanblocks, 3));
        bits.extend(encode_ulong(skipbytes, 0));
        let mut out = Vec::with_capacity(5 + bits.len() / 8 + 1);
        out.extend_from_slice(&MAGIC);
        out.push(2); // version
        out.extend(pack_bits_msb_first(&bits));
        out
    }

    #[test]
    fn rejects_short_buffer() {
        assert_eq!(parse_stream_header(&[]), Err(Error::InvalidMagic));
        assert_eq!(parse_stream_header(b"ajkg"), Err(Error::InvalidMagic));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut buf = vec![0u8; 16];
        buf[0..4].copy_from_slice(b"AJKG");
        buf[4] = 2;
        assert_eq!(parse_stream_header(&buf), Err(Error::InvalidMagic));
    }

    #[test]
    fn rejects_version_zero_and_four() {
        let mut buf = synth_v2_header(5, 2, 256, 0, 4, 0);
        buf[4] = 0;
        assert_eq!(parse_stream_header(&buf), Err(Error::UnsupportedVersion(0)));
        buf[4] = 4;
        assert_eq!(parse_stream_header(&buf), Err(Error::UnsupportedVersion(4)));
    }

    #[test]
    fn rejects_version_one_pending_layout_resolution() {
        // v1's parameter-block layout differs from v2/v3 per `spec/01`
        // §3.5 (the mean-estimator field is version-conditional) and
        // is open §9.4 candidate #2. The parser surfaces a clean
        // `UnsupportedVersion(1)` until a v1 fixture is reachable.
        let mut buf = synth_v2_header(5, 2, 256, 0, 4, 0);
        buf[4] = 1;
        assert_eq!(parse_stream_header(&buf), Err(Error::UnsupportedVersion(1)));
    }

    #[test]
    fn parses_fixture_f1_byte_sequence_exactly() {
        // Fixture `F1`'s first nine bytes per `spec/02` §1: `61 6A 6B
        // 67 02 FB B1 70 09 F9 25` (the §6 table extends to byte
        // 0x0A). The header-only decode of `spec/02` §6 yields
        //   H_filetype  = 5  (spec/02 §6.1)
        //   H_channels  = 2  (spec/02 §6.2)
        //   H_blocksize = 256 (spec/02 §6.3)
        //   H_maxlpcorder = 0 (spec/02 §6.4)
        //   H_meanblocks = 4 (spec/02 §6.5)
        //   H_skipbytes = 0  (spec/02 §6.6)
        // The bits-consumed-after-v counter should land at 43
        // (spec/02 §6.7).
        let buf = [
            0x61, 0x6A, 0x6B, 0x67, 0x02, 0xFB, 0xB1, 0x70, 0x09, 0xF9, 0x25,
        ];
        let parsed = parse_stream_header(&buf).expect("F1 header must parse");
        assert_eq!(parsed.header.version, 2);
        assert_eq!(parsed.header.filetype, 5);
        assert_eq!(parsed.header.channels, 2);
        assert_eq!(parsed.header.blocksize, 256);
        assert_eq!(parsed.header.maxlpcorder, 0);
        assert_eq!(parsed.header.meanblocks, 4);
        assert_eq!(parsed.header.skipbytes, 0);
        assert_eq!(parsed.bits_consumed_after_v, 43);
    }

    #[test]
    fn parses_synthetic_v3_with_distinct_field_values() {
        // Round-trip through the synth helper to make sure the
        // parser reads back each field's chosen value. Values are
        // picked to stay within the chosen mantissa widths and to be
        // mutually distinct so a swapped-field bug would surface
        // immediately.
        let mut buf = synth_v2_header(3, 1, 512, 0, 7, 0);
        buf[4] = 3; // version 3 reuses the v2 six-field layout per
                    // `spec/00` §"Format versions".
        let parsed = parse_stream_header(&buf).expect("synthetic v3 must parse");
        assert_eq!(parsed.header.version, 3);
        assert_eq!(parsed.header.filetype, 3);
        assert_eq!(parsed.header.channels, 1);
        assert_eq!(parsed.header.blocksize, 512);
        assert_eq!(parsed.header.maxlpcorder, 0);
        assert_eq!(parsed.header.meanblocks, 7);
        assert_eq!(parsed.header.skipbytes, 0);
    }

    #[test]
    fn truncated_header_returns_truncated() {
        // A full F1 prefix is 11 bytes long. Truncate to 6 — the
        // parser should exhaust mid-`H_blocksize`.
        let buf = [0x61, 0x6A, 0x6B, 0x67, 0x02, 0xFB];
        assert_eq!(parse_stream_header(&buf), Err(Error::Truncated));
    }

    #[test]
    fn sample_history_carry_floor_is_three() {
        let h = ShortenStreamHeader {
            version: 2,
            filetype: 5,
            channels: 2,
            blocksize: 256,
            maxlpcorder: 0,
            meanblocks: 4,
            skipbytes: 0,
        };
        assert_eq!(h.sample_history_carry_len(), 3);
        let h2 = ShortenStreamHeader {
            maxlpcorder: 7,
            ..h
        };
        assert_eq!(h2.sample_history_carry_len(), 7);
    }
}
