//! Crate-local error and result types.
//!
//! All decoder failures surface as a [`Error`] variant. The variants
//! intentionally mirror the spec set's wire-format layers so that a
//! caller can distinguish a structural problem (bad magic, truncated
//! header) from a per-block parse problem (unknown function code,
//! truncated residual).

use core::fmt;

/// Decoder error type.
///
/// `Error` surfaces both stream-format violations (bad magic,
/// unsupported version, truncated bitstream) and decoder-runtime
/// problems (residual mantissa width too large, channel cursor
/// desync). Each variant carries enough context to identify the
/// failing field.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// File is shorter than the 5-byte byte-aligned prefix.
    Truncated,
    /// First four bytes were not ASCII `ajkg`.
    BadMagic([u8; 4]),
    /// Version byte is outside the 1..=3 in-scope range.
    UnsupportedVersion(u8),
    /// `H_filetype` decoded to a value the spec set has not pinned.
    UnsupportedFiletype(u32),
    /// `H_channels` is zero or exceeds [`MAX_CHANNELS`](crate::MAX_CHANNELS).
    UnsupportedChannelCount(u32),
    /// `H_blocksize` is zero or exceeds [`MAX_BLOCKSIZE`](crate::MAX_BLOCKSIZE).
    UnsupportedBlockSize(u32),
    /// `H_maxlpcorder` exceeds the in-scope cap of 32.
    UnsupportedLpcOrder(u32),
    /// Function code outside the spec set's enumerated 0..=9 range.
    UnknownFunctionCode(u32),
    /// Bit-stream ended mid-field (residual, header field, or command).
    UnexpectedEof,
    /// Residual mantissa width would overflow the decoder's i32 lanes
    /// (`>= 32` after the +1 offset).
    ResidualWidthOverflow(u32),
    /// `BLOCK_FN_QLPC` block specified an `order` exceeding the
    /// header's `H_maxlpcorder`.
    LpcOrderExceedsHeader { block_order: u32, header_max: u32 },
    /// LPC `coef` quantised value is outside the spec set's plausible
    /// range (used as a structural sanity check; the wire encoding
    /// itself does not bound `svar(LPCQUANT)`).
    LpcCoefOutOfRange(i32),
    /// Bit-shift command emitted a shift that would overflow a 32-bit
    /// PCM sample lane.
    BitShiftOverflow(u32),
    /// Internal invariant violated. Should never happen in practice;
    /// surfaces as a structured failure rather than a panic so callers
    /// can surface it cleanly.
    Internal(&'static str),
    /// I/O failure when reading from a [`std::io::Read`] source.
    Io(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("oxideav-shorten: file shorter than 5-byte prefix"),
            Self::BadMagic(m) => write!(
                f,
                "oxideav-shorten: bad magic bytes {:02x?}, expected 'ajkg'",
                m
            ),
            Self::UnsupportedVersion(v) => {
                write!(f, "oxideav-shorten: unsupported version byte {v}")
            }
            Self::UnsupportedFiletype(t) => {
                write!(f, "oxideav-shorten: unsupported H_filetype {t}")
            }
            Self::UnsupportedChannelCount(c) => {
                write!(f, "oxideav-shorten: unsupported channel count {c}")
            }
            Self::UnsupportedBlockSize(b) => {
                write!(f, "oxideav-shorten: unsupported block size {b}")
            }
            Self::UnsupportedLpcOrder(o) => {
                write!(f, "oxideav-shorten: H_maxlpcorder {o} exceeds the in-scope cap")
            }
            Self::UnknownFunctionCode(c) => {
                write!(f, "oxideav-shorten: unknown function code {c}")
            }
            Self::UnexpectedEof => f.write_str("oxideav-shorten: unexpected end of bitstream"),
            Self::ResidualWidthOverflow(w) => {
                write!(f, "oxideav-shorten: residual mantissa width {w} too large")
            }
            Self::LpcOrderExceedsHeader { block_order, header_max } => write!(
                f,
                "oxideav-shorten: per-block LPC order {block_order} exceeds H_maxlpcorder {header_max}"
            ),
            Self::LpcCoefOutOfRange(c) => {
                write!(f, "oxideav-shorten: LPC coefficient {c} outside plausible range")
            }
            Self::BitShiftOverflow(s) => write!(f, "oxideav-shorten: bit-shift {s} overflows i32"),
            Self::Internal(msg) => write!(f, "oxideav-shorten: internal: {msg}"),
            Self::Io(msg) => write!(f, "oxideav-shorten: I/O: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

/// Crate-local `Result` alias.
pub type Result<T> = core::result::Result<T, Error>;
