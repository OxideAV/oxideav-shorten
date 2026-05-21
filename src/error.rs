//! Crate-local error type.
//!
//! Round 1 surfaces only the failure modes the file-header parser can
//! produce against synthetic byte buffers built per
//! `docs/audio/shorten/spec/01-stream-header.md` + `spec/02`.

/// Errors produced by the Shorten file-header parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The stream is missing the four-byte `ajkg` magic at offset 0
    /// or is shorter than five bytes total. Per `spec/01` §1 the
    /// magic + version byte together occupy file offsets `0x00..=0x04`.
    InvalidMagic,
    /// The version byte at offset `0x04` carried a value outside the
    /// `{1, 2, 3}` set this spec set covers. `spec/00` §"Format
    /// versions" pins v1, v2, v3 as in-scope; v0 is explicitly out of
    /// scope for this round.
    UnsupportedVersion(u8),
    /// The bit stream ran out of bytes mid-`uvar`/`ulong` decode.
    /// `spec/02` §1 reads each header field MSB-first from offset
    /// `0x05` onward; if the parameter block cannot complete within
    /// the supplied buffer the parse aborts here.
    Truncated,
    /// A `uvar(n)` prefix-zeros run exceeded the implementation's
    /// safety cap (32 leading zeros for a 32-bit value). A
    /// well-formed Shorten file should never produce a header field
    /// of this magnitude — `spec/01` §3's six header fields are all
    /// small integers in practice.
    OverflowingUvar,
    /// The per-block command stream named a function code outside the
    /// 0..=9 set pinned in `spec/03` §3 / `spec/04`. v2/v3 encoders
    /// never emit codes outside that set; receiving one is either
    /// stream corruption or an out-of-scope format-version feature.
    UnknownFunctionCode(u32),
    /// Round 2 lands the verbatim and quit commands of `spec/03`; the
    /// predictor and LPC commands (codes 0..3, 7) plus the
    /// housekeeping commands (5, 6, 8) are not yet decoded. The block
    /// reader surfaces this error when one of the not-yet-implemented
    /// codes is encountered so the caller knows the stream needs a
    /// later round's command set.
    BlockCommandNotImplemented(u32),
    /// Round 1 does not decode the per-block command stream that
    /// follows the parameter block. Returned from any non-header API
    /// surface that the orphan-rebuild scaffold has not wired up yet.
    NotImplemented,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::InvalidMagic => f.write_str("oxideav-shorten: invalid 'ajkg' magic"),
            Error::UnsupportedVersion(v) => {
                write!(f, "oxideav-shorten: unsupported format version {v}")
            }
            Error::Truncated => f.write_str("oxideav-shorten: stream truncated mid-header"),
            Error::OverflowingUvar => {
                f.write_str("oxideav-shorten: header uvar prefix exceeded safety cap")
            }
            Error::UnknownFunctionCode(c) => {
                write!(f, "oxideav-shorten: unknown per-block function code {c}")
            }
            Error::BlockCommandNotImplemented(c) => write!(
                f,
                "oxideav-shorten: per-block function code {c} not implemented in this round"
            ),
            Error::NotImplemented => f.write_str(
                "oxideav-shorten: feature not implemented in this round (file-header parser only)",
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Crate-local `Result` alias.
pub type Result<T> = core::result::Result<T, Error>;
