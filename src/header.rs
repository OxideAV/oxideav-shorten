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

    /// Resolve the raw `H_filetype` numeric value to the named
    /// [`Filetype`] variant when `spec/05` §6 pins the code.
    ///
    /// Returns `Some(Filetype::U8)` for `2`, `Some(Filetype::S16HL)`
    /// for `3`, `Some(Filetype::S16LH)` for `5` (the three codes
    /// behaviourally anchored by fixtures `F2`, `F3`, `F1`
    /// respectively per `spec/05` §6). All other numeric values
    /// return `None`: the remaining eight TR.156 labels (`ulaw`,
    /// `s8`, `s16`, `u16`, `s16x`, `u16x`, `u16hl`, `u16lh`) are
    /// unpinned by the current fixture corpus (`spec/05` §8 open
    /// §9.4 candidate #3) and the caller cannot infer the byte
    /// packing for them safely.
    pub fn filetype_pinned(&self) -> Option<Filetype> {
        Filetype::from_wire(self.filetype)
    }
}

/// Named sample-format codes carried by `H_filetype` whose numeric
/// values are pinned in `docs/audio/shorten/spec/05-state-and-quirks.md`
/// §6 by behavioural verification against the public fixture corpus.
///
/// The three variants correspond to the three numeric codes that
/// fixtures `F1`, `F2`, `F3` force into the stream:
///
/// | Variant   | Wire value | TR.156 label | Pinning fixture |
/// | --------- | ---------- | ------------ | --------------- |
/// | [`Self::U8`]    | `2` | `u8`    | `F2` (AIFC u8 source). |
/// | [`Self::S16HL`] | `3` | `s16hl` | `F3` (AIFC s16 big-endian source). |
/// | [`Self::S16LH`] | `5` | `s16lh` | `F1` (WAV s16 little-endian source). |
///
/// `spec/05` §6 explicitly leaves the remaining eight TR.156 labels
/// (`ulaw`, `s8`, `s16`, `u16`, `s16x`, `u16x`, `u16hl`, `u16lh`) with
/// unpinned numeric codes; this enum is therefore `#[non_exhaustive]`
/// so a later round can add their variants without breaking callers
/// once the additional fixtures or reference-encoder observations
/// pin them (`spec/05` §8 open §9.4 candidate #3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Filetype {
    /// `H_filetype = 2`: 8-bit unsigned PCM (`u8`), one byte per
    /// sample, byte order is not meaningful. Pinned by fixture
    /// `F2` per `spec/05` §6.
    U8,
    /// `H_filetype = 3`: 16-bit signed PCM, high-byte-first (`s16hl`).
    /// Two bytes per sample emitted big-endian on the host plane.
    /// Pinned by fixture `F3` per `spec/05` §6.
    S16HL,
    /// `H_filetype = 5`: 16-bit signed PCM, low-byte-first (`s16lh`).
    /// Two bytes per sample emitted little-endian on the host plane.
    /// Pinned by fixture `F1` per `spec/05` §6.
    S16LH,
}

impl Filetype {
    /// Map a wire-level `H_filetype` numeric value to its named
    /// variant when `spec/05` §6 pins the code, otherwise `None`.
    pub fn from_wire(code: u32) -> Option<Self> {
        match code {
            2 => Some(Filetype::U8),
            3 => Some(Filetype::S16HL),
            5 => Some(Filetype::S16LH),
            _ => None,
        }
    }

    /// Wire-level `H_filetype` numeric value of this variant.
    pub fn wire_value(self) -> u32 {
        match self {
            Filetype::U8 => 2,
            Filetype::S16HL => 3,
            Filetype::S16LH => 5,
        }
    }

    /// TR.156 textual label (the encoder's `-t <label>` argument name).
    pub fn label(self) -> &'static str {
        match self {
            Filetype::U8 => "u8",
            Filetype::S16HL => "s16hl",
            Filetype::S16LH => "s16lh",
        }
    }

    /// Number of host bytes one decoded sample occupies on a packed
    /// PCM plane: 1 for `u8`, 2 for `s16hl` / `s16lh`.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Filetype::U8 => 1,
            Filetype::S16HL | Filetype::S16LH => 2,
        }
    }

    /// Whether the sample format is signed. `u8` is unsigned;
    /// `s16hl` / `s16lh` are signed.
    pub fn is_signed(self) -> bool {
        match self {
            Filetype::U8 => false,
            Filetype::S16HL | Filetype::S16LH => true,
        }
    }

    /// Host byte order the samples are packed in when emitted.
    ///
    /// * `Some(false)` for [`Self::S16HL`] (high-byte-first =
    ///   big-endian).
    /// * `Some(true)` for [`Self::S16LH`] (low-byte-first =
    ///   little-endian).
    /// * `None` for [`Self::U8`]: a single-byte sample has no
    ///   byte-order distinction.
    pub fn is_little_endian(self) -> Option<bool> {
        match self {
            Filetype::U8 => None,
            Filetype::S16HL => Some(false),
            Filetype::S16LH => Some(true),
        }
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
    fn filetype_pinned_resolves_three_anchored_codes() {
        // spec/05 §6 pins H_filetype = 2/u8 (F2), 3/s16hl (F3), 5/s16lh (F1).
        let mut h = ShortenStreamHeader {
            version: 2,
            filetype: 2,
            channels: 1,
            blocksize: 256,
            maxlpcorder: 0,
            meanblocks: 4,
            skipbytes: 0,
        };
        assert_eq!(h.filetype_pinned(), Some(Filetype::U8));
        h.filetype = 3;
        assert_eq!(h.filetype_pinned(), Some(Filetype::S16HL));
        h.filetype = 5;
        assert_eq!(h.filetype_pinned(), Some(Filetype::S16LH));
    }

    #[test]
    fn filetype_pinned_returns_none_for_unpinned_labels() {
        // spec/05 §6 + §8 candidate #3: eight TR.156 labels (`ulaw`,
        // `s8`, `s16`, `u16`, `s16x`, `u16x`, `u16hl`, `u16lh`) have
        // no pinned numeric code in the spec set. The accessor returns
        // None for any value outside { 2, 3, 5 }.
        let mut h = ShortenStreamHeader {
            version: 2,
            filetype: 0,
            channels: 1,
            blocksize: 256,
            maxlpcorder: 0,
            meanblocks: 4,
            skipbytes: 0,
        };
        for unpinned in [0u32, 1, 4, 6, 7, 8, 9, 10, 11, 12, 255] {
            h.filetype = unpinned;
            assert_eq!(
                h.filetype_pinned(),
                None,
                "wire value {unpinned} is unpinned by spec/05 §6 + §8"
            );
        }
    }

    #[test]
    fn filetype_wire_value_round_trips() {
        for ft in [Filetype::U8, Filetype::S16HL, Filetype::S16LH] {
            assert_eq!(Filetype::from_wire(ft.wire_value()), Some(ft));
        }
    }

    #[test]
    fn filetype_label_matches_tr156_naming() {
        assert_eq!(Filetype::U8.label(), "u8");
        assert_eq!(Filetype::S16HL.label(), "s16hl");
        assert_eq!(Filetype::S16LH.label(), "s16lh");
    }

    #[test]
    fn filetype_bytes_per_sample_matches_pinned_widths() {
        // spec/05 §6 row: u8 → 1 byte, s16hl / s16lh → 2 bytes.
        assert_eq!(Filetype::U8.bytes_per_sample(), 1);
        assert_eq!(Filetype::S16HL.bytes_per_sample(), 2);
        assert_eq!(Filetype::S16LH.bytes_per_sample(), 2);
    }

    #[test]
    fn filetype_is_signed_matches_tr156_label_prefix() {
        // `u` prefix → unsigned, `s` prefix → signed per TR.156's
        // file-type naming (spec/05 §6 narrative).
        assert!(!Filetype::U8.is_signed());
        assert!(Filetype::S16HL.is_signed());
        assert!(Filetype::S16LH.is_signed());
    }

    #[test]
    fn filetype_byte_order_matches_label_suffix() {
        // `hl` = high-byte-first = big-endian; `lh` = low-byte-first =
        // little-endian; single-byte u8 carries no byte-order
        // distinction so the accessor returns None per the doc rule.
        assert_eq!(Filetype::U8.is_little_endian(), None);
        assert_eq!(Filetype::S16HL.is_little_endian(), Some(false));
        assert_eq!(Filetype::S16LH.is_little_endian(), Some(true));
    }

    #[test]
    fn filetype_pinned_on_real_f1_byte_sequence() {
        // F1's header decodes to H_filetype = 5 (`s16lh`) per
        // spec/02 §6.1; the typed accessor must agree.
        let buf = [
            0x61, 0x6A, 0x6B, 0x67, 0x02, 0xFB, 0xB1, 0x70, 0x09, 0xF9, 0x25,
        ];
        let parsed = parse_stream_header(&buf).expect("F1 header must parse");
        assert_eq!(parsed.header.filetype_pinned(), Some(Filetype::S16LH));
        let ft = parsed.header.filetype_pinned().unwrap();
        assert_eq!(ft.bytes_per_sample(), 2);
        assert!(ft.is_signed());
        assert_eq!(ft.is_little_endian(), Some(true));
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
