//! Stream header parser (`spec/01-stream-header.md`).
//!
//! The Shorten stream starts with a five-byte byte-aligned prefix:
//!
//! ```text
//! 0x00..=0x03  ASCII magic 'ajkg'
//! 0x04         u8 version (1, 2, or 3 in scope)
//! ```
//!
//! From byte 0x05 onward, every field is encoded under the
//! variable-length integer scheme of [`crate::varint`]. The header
//! parameter block is six [`read_ulong`](crate::varint::read_ulong)
//! reads, in this fixed order:
//!
//! 1. `H_filetype`     — file-type code (sample encoding identifier)
//! 2. `H_channels`     — channel count
//! 3. `H_blocksize`    — default block size in samples per channel
//! 4. `H_maxlpcorder`  — maximum LPC order (0 disables LPC)
//! 5. `H_meanblocks`   — running-mean estimator window length
//! 6. `H_skipbytes`    — encoder-side verbatim-prefix byte count
//!
//! `H_meanblocks` is conditionally present in v2/v3 streams; v0/v1
//! streams omit the field (the running-mean estimator defaults to
//! "off" — TR.156 Appendix §"-m blocks"). This crate accepts v1
//! streams syntactically (the FFmpeg decoder accepts them per the
//! T3 tampering test) but reads the field unconditionally; v1
//! fixtures with no mean-estimator field would mis-parse, and no v1
//! fixture is currently reachable to confirm the v1 layout.

use crate::bitreader::BitReader;
use crate::varint::read_ulong;
use crate::{Error, Result, MAX_BLOCKSIZE, MAX_CHANNELS, MAX_LPC_ORDER};

/// Magic bytes at file offset 0..=3.
pub const MAGIC: [u8; 4] = *b"ajkg";

/// File-type code mapping for the eleven TR.156 labels.
///
/// `spec/05-state-and-quirks.md` §6 pins three numeric values from the
/// public fixture corpus: **`u8 = 2`, `s16hl = 3`, `s16lh = 5`**. The
/// remaining eight labels' on-wire numeric codes are *not* pinned by
/// any reference in the spec set's allow-list — neither TR.156 nor the
/// behavioural anchors enumerate them. Round 2 admits each remaining
/// label as a distinct enum variant so the decoder can name what it
/// observed and the encoder can produce a representative wire byte;
/// the numeric codes selected for the eight unpinned labels are
/// **chosen by this implementation** to fill the gaps in the TR.156
/// list-position scheme without colliding with the three pinned codes:
///
/// | TR.156 list pos | Label   | Wire code |
/// | --------------- | ------- | --------- |
/// | 0               | `ulaw`  | 0         |
/// | 1               | `s8`    | 1         |
/// | 2               | `u8`    | 2  (pinned by F2) |
/// | 3               | `s16hl` | 3  (pinned by F3) |
/// | 4               | `u16hl` | 4         |
/// | 5               | `s16lh` | 5  (pinned by F1) |
/// | 6               | `u16lh` | 6         |
/// | 7               | `s16`   | 7         |
/// | 8               | `u16`   | 8         |
/// | 9               | `s16x`  | 9         |
/// | 10              | `u16x`  | 10        |
///
/// Streams encoded by this crate using one of the eight unpinned codes
/// roundtrip through this crate but **may not roundtrip through other
/// Shorten implementations** until `spec/05-state-and-quirks.md` §6
/// pins the remaining mapping (a §9.4 escalation candidate). See
/// `spec/05-state-and-quirks.md` §6 for the spec gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Filetype {
    /// µ-law 8-bit per ITU-T G.711, file-type code 0 (unpinned).
    Ulaw,
    /// 8-bit signed, file-type code 1 (unpinned).
    S8,
    /// 8-bit unsigned, file-type code 2 (pinned by F2).
    U8,
    /// 16-bit signed, big-endian byte order on the *output* (`s16hl`),
    /// file-type code 3 (pinned by F3).
    S16Be,
    /// 16-bit unsigned, big-endian byte order (`u16hl`),
    /// file-type code 4 (unpinned).
    U16Be,
    /// 16-bit signed, little-endian byte order on the *output*
    /// (`s16lh`), file-type code 5 (pinned by F1).
    S16Le,
    /// 16-bit unsigned, little-endian byte order (`u16lh`),
    /// file-type code 6 (unpinned).
    U16Le,
    /// 16-bit signed, host-natural byte order (`s16`),
    /// file-type code 7 (unpinned).
    S16Native,
    /// 16-bit unsigned, host-natural byte order (`u16`),
    /// file-type code 8 (unpinned).
    U16Native,
    /// 16-bit signed, byte-swapped relative to host (`s16x`),
    /// file-type code 9 (unpinned).
    S16Swapped,
    /// 16-bit unsigned, byte-swapped relative to host (`u16x`),
    /// file-type code 10 (unpinned).
    U16Swapped,
}

impl Filetype {
    /// Resolve a numeric file-type code to a [`Filetype`] variant, or
    /// return an [`Error::UnsupportedFiletype`] for codes outside the
    /// known set.
    pub fn from_code(code: u32) -> Result<Self> {
        match code {
            0 => Ok(Self::Ulaw),
            1 => Ok(Self::S8),
            2 => Ok(Self::U8),
            3 => Ok(Self::S16Be),
            4 => Ok(Self::U16Be),
            5 => Ok(Self::S16Le),
            6 => Ok(Self::U16Le),
            7 => Ok(Self::S16Native),
            8 => Ok(Self::U16Native),
            9 => Ok(Self::S16Swapped),
            10 => Ok(Self::U16Swapped),
            other => Err(Error::UnsupportedFiletype(other)),
        }
    }

    /// Numeric file-type code on the wire.
    pub fn to_code(self) -> u32 {
        match self {
            Self::Ulaw => 0,
            Self::S8 => 1,
            Self::U8 => 2,
            Self::S16Be => 3,
            Self::U16Be => 4,
            Self::S16Le => 5,
            Self::U16Le => 6,
            Self::S16Native => 7,
            Self::U16Native => 8,
            Self::S16Swapped => 9,
            Self::U16Swapped => 10,
        }
    }

    /// Bytes per output sample.
    pub fn bytes_per_output_sample(self) -> usize {
        match self {
            Self::Ulaw | Self::S8 | Self::U8 => 1,
            Self::S16Be
            | Self::U16Be
            | Self::S16Le
            | Self::U16Le
            | Self::S16Native
            | Self::U16Native
            | Self::S16Swapped
            | Self::U16Swapped => 2,
        }
    }

    /// Whether the predictor lane is signed (true for every TR.156
    /// label — Shorten's predictor always operates in a signed
    /// domain; the unsigned output filetypes are biased by the
    /// running-mean estimator at decode time).
    pub fn is_signed_internal(self) -> bool {
        true
    }

    /// Whether the output bytes are little-endian. For 8-bit and µ-law
    /// labels this is irrelevant; we return `true` (single-byte values
    /// are byte-order-agnostic). For `*_Native` and `*_Swapped` we
    /// resolve against host byte order — `Native` is LE on a
    /// little-endian host, BE on a big-endian host; `Swapped` is the
    /// converse.
    pub fn output_endian_le(self) -> bool {
        match self {
            Self::Ulaw | Self::S8 | Self::U8 => true,
            Self::S16Le | Self::U16Le => true,
            Self::S16Be | Self::U16Be => false,
            Self::S16Native | Self::U16Native => cfg!(target_endian = "little"),
            Self::S16Swapped | Self::U16Swapped => cfg!(target_endian = "big"),
        }
    }

    /// Whether the *output* sample format is unsigned (the encoder's
    /// input had a half-range bias, so the decoder restores it on
    /// emission).
    pub fn is_unsigned_output(self) -> bool {
        matches!(
            self,
            Self::U8 | Self::U16Be | Self::U16Le | Self::U16Native | Self::U16Swapped
        )
    }
}

/// Parsed stream header.
#[derive(Debug, Clone)]
pub struct StreamHeader {
    /// Format version byte (1, 2, or 3 in scope).
    pub version: u8,
    /// File-type code resolved to the pinned [`Filetype`] enum.
    pub filetype: Filetype,
    /// Channel count (`H_channels`).
    pub channels: u16,
    /// Default block size in samples per channel (`H_blocksize`).
    pub blocksize: u32,
    /// Maximum LPC order (`H_maxlpcorder`); 0 disables LPC.
    pub max_lpc_order: u32,
    /// Running-mean estimator window length (`H_meanblocks`).
    pub mean_blocks: u32,
    /// Verbatim-prefix byte count (`H_skipbytes`).
    pub skip_bytes: u32,
    /// Bit-position immediately after the parameter block, measured
    /// in bits from the start of byte 0x05 of the file. Callers can
    /// continue parsing the per-block command stream from this
    /// position via a [`BitReader`] over `&file[5..]`.
    pub header_end_bit: usize,
}

impl StreamHeader {
    /// Carry-buffer length (`max(3, max_lpc_order)`).
    pub fn carry_len(&self) -> usize {
        core::cmp::max(3, self.max_lpc_order as usize)
    }
}

/// Parse the byte-aligned 5-byte prefix and the variable-length
/// integer parameter block.
///
/// Returns the parsed [`StreamHeader`] on success. The bit cursor's
/// final position is exposed via [`StreamHeader::header_end_bit`] so
/// the caller can resume bit-stream parsing.
pub fn parse_header(file_bytes: &[u8]) -> Result<StreamHeader> {
    if file_bytes.len() < 5 {
        return Err(Error::Truncated);
    }
    let magic = [file_bytes[0], file_bytes[1], file_bytes[2], file_bytes[3]];
    if magic != MAGIC {
        return Err(Error::BadMagic(magic));
    }
    let version = file_bytes[4];
    if !(1..=3).contains(&version) {
        return Err(Error::UnsupportedVersion(version));
    }

    let mut br = BitReader::new(&file_bytes[5..]);

    let filetype_code = read_ulong(&mut br)?;
    let filetype = Filetype::from_code(filetype_code)?;

    let channels = read_ulong(&mut br)?;
    if channels == 0 || channels > MAX_CHANNELS as u32 {
        return Err(Error::UnsupportedChannelCount(channels));
    }

    let blocksize = read_ulong(&mut br)?;
    if blocksize == 0 || blocksize > MAX_BLOCKSIZE {
        return Err(Error::UnsupportedBlockSize(blocksize));
    }

    let max_lpc_order = read_ulong(&mut br)?;
    if max_lpc_order > MAX_LPC_ORDER {
        return Err(Error::UnsupportedLpcOrder(max_lpc_order));
    }

    // v2/v3: H_meanblocks present. v1: per spec/01 §3.5, the field is
    // version-conditional (default off). No v1 fixture is reachable;
    // we follow the FFmpeg-observed v2/v3 layout for v >= 2 and treat
    // v1 the same way (the syntactic accept on tampered v1 is what the
    // decoder we audit against does).
    let mean_blocks = read_ulong(&mut br)?;

    let skip_bytes = read_ulong(&mut br)?;

    Ok(StreamHeader {
        version,
        filetype,
        channels: channels as u16,
        blocksize,
        max_lpc_order,
        mean_blocks,
        skip_bytes,
        header_end_bit: br.bit_pos(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F1's first 11 bytes (magic + version + the 6 var-int params + a
    /// little post-header tail) reproduced from spec/02 §6.
    /// Bytes `0x00..=0x0A` of `luckynight.shn`:
    /// `61 6A 6B 67 02 FB B1 70 09 F9 25`.
    const F1_HEADER_BYTES: [u8; 11] = [
        0x61, 0x6A, 0x6B, 0x67, 0x02, 0xFB, 0xB1, 0x70, 0x09, 0xF9, 0x25,
    ];

    #[test]
    fn parses_f1_header() {
        let h = parse_header(&F1_HEADER_BYTES).expect("parse");
        assert_eq!(h.version, 2);
        assert_eq!(h.filetype, Filetype::S16Le);
        assert_eq!(h.channels, 2);
        assert_eq!(h.blocksize, 256);
        assert_eq!(h.max_lpc_order, 0);
        assert_eq!(h.mean_blocks, 4);
        assert_eq!(h.skip_bytes, 0);
        // The header occupies 43 bits past byte 5 per spec/02 §6.7.
        assert_eq!(h.header_end_bit, 43);
        assert_eq!(h.carry_len(), 3);
    }

    #[test]
    fn rejects_bad_magic() {
        let bytes = [0u8; 16];
        let err = parse_header(&bytes).unwrap_err();
        assert!(matches!(err, Error::BadMagic(_)));
    }

    #[test]
    fn rejects_truncated() {
        let bytes = [0x61, 0x6A, 0x6B];
        assert!(matches!(
            parse_header(&bytes).unwrap_err(),
            Error::Truncated
        ));
    }

    #[test]
    fn rejects_version_zero() {
        let mut bytes = [0u8; 16];
        bytes[..4].copy_from_slice(b"ajkg");
        bytes[4] = 0;
        assert!(matches!(
            parse_header(&bytes).unwrap_err(),
            Error::UnsupportedVersion(0)
        ));
    }
}
