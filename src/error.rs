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
    /// Round 2 lands the verbatim and quit commands of `spec/03`.
    /// Round 3 closes the polynomial-difference predictors (codes
    /// 0..3); round 4 closes `BLOCK_FN_ZERO = 8` (via `fill_zero_block`
    /// plus the round 4 running-mean estimator); round 5 closes the
    /// quantised-LPC predictor `BLOCK_FN_QLPC = 7` (via
    /// `decode_qlpc_block`); round 6 closes the housekeeping commands
    /// `BLOCK_FN_BLOCKSIZE = 5` (via `read_blocksize_payload`) and
    /// `BLOCK_FN_BITSHIFT = 6` (via `read_bitshift_payload`). With
    /// rounds 1..=6 every code 0..=9 has a payload decoder, so this
    /// variant is retained only as a forward-compatibility sentinel
    /// for any future per-block command a later format-version delta
    /// (`spec/05` §7) might introduce.
    BlockCommandNotImplemented(u32),
    /// A `BLOCK_FN_DIFFn` / `BLOCK_FN_QLPC` energy parameter decoded
    /// to a value whose `+1`-adjusted residual mantissa width would
    /// exceed the implementation's safety cap. `spec/02` §4.2 pins
    /// the field as `uvar(ENERGYSIZE = 3)` and `spec/05` §3 pins the
    /// `+1` adjustment; legitimate encoder choices produce widths well
    /// below the cap.
    EnergyTooLarge(u32),
    /// A per-block sample count exceeded the implementation's safety
    /// cap. `H_blocksize` in TR.156's reference encoder defaults to 256
    /// and is bounded by the encoder side; an over-cap value indicates
    /// either stream corruption or a sub-block-size override outside
    /// the cap.
    BlockTooLarge(u32),
    /// A reconstructed predictor sample fell outside the `i32` range
    /// the carry buffer stores. Reachable only for pathological inputs
    /// (the `i64`-headroom reconstruction intermediate guards against
    /// the recurrence's worst-case intermediates).
    SampleOverflow,
    /// A `BLOCK_FN_QLPC` block named a per-block LPC order that exceeds
    /// the per-channel sample-history carry's length. `spec/03` §3.11
    /// pins the carry length at `max(3, H_maxlpcorder)` and §3.5 bounds
    /// the per-block order at `H_maxlpcorder`; an order beyond the
    /// carry's length cannot supply the `order` past samples the LPC
    /// recurrence reads for its first sample, so the stream is either
    /// corrupt or the decoder's carry was allocated against a header
    /// with a smaller `H_maxlpcorder` than the block requests.
    LpcOrderTooLarge { order: u32, carry_len: u32 },
    /// A `BLOCK_FN_BLOCKSIZE` command carried a zero `new_bs` parameter.
    /// `spec/03` §3.6 describes the override as "the size of subsequent
    /// blocks" — a zero-sample block leaves the predictor's residual
    /// loop empty and defeats the override's purpose, so the encoder
    /// never emits this value.
    ZeroBlockSize,
    /// A `BLOCK_FN_BITSHIFT` command carried a `bshift` value exceeding
    /// the implementation safety cap (`BITSHIFT_MAX`, 31 positions).
    /// `spec/02` §4.6 notes encoder-side shifts above ~16 are not
    /// encountered for 16-bit audio; typical `-q N` invocations are in
    /// `1..=12`. A larger value is either stream corruption or an
    /// out-of-scope format-version feature.
    BitshiftTooLarge(u32),
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
            Error::EnergyTooLarge(e) => write!(
                f,
                "oxideav-shorten: residual mantissa width derived from energy {e} exceeds safety cap"
            ),
            Error::BlockTooLarge(bs) => write!(
                f,
                "oxideav-shorten: per-block sample count {bs} exceeds safety cap"
            ),
            Error::SampleOverflow => {
                f.write_str("oxideav-shorten: reconstructed sample fell outside i32 range")
            }
            Error::LpcOrderTooLarge { order, carry_len } => write!(
                f,
                "oxideav-shorten: QLPC block order {order} exceeds sample-history carry length {carry_len}"
            ),
            Error::ZeroBlockSize => f.write_str(
                "oxideav-shorten: BLOCKSIZE command carried zero new_bs (encoder never emits)",
            ),
            Error::BitshiftTooLarge(b) => write!(
                f,
                "oxideav-shorten: BITSHIFT command bshift {b} exceeds safety cap"
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
