//! Shorten encode-side primitives — round 12 envelope + round 13 DIFF0
//! + round 14 DIFF1 predictor encoder.
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
//! Round 13 lands the smallest predictor encoder: [`write_diff0_block`]
//! emits a `BLOCK_FN_DIFF0` command (function code 0 per `spec/03` §3.1)
//! carrying `bs` per-sample residuals `e₀(t) = s(t) − μ_chan` under
//! `svar(e + 1)` mantissa width (`spec/03` §3.1 + `spec/05` §3.1's
//! encoded-value-plus-one rule). [`min_energy_for_diff0`] picks the
//! smallest encoded energy in `0..=7` such that the maximum folded
//! residual fits inside the svar mantissa with a zero prefix-zero
//! count, matching `spec/05` §3.1's "smallest sensible `n` is 1"
//! observation. Output blocks round-trip through
//! [`crate::decode_diff_block`] byte-exactly.
//!
//! Round 14 lands the order-1 polynomial-difference predictor encoder:
//! [`write_diff1_block`] emits a `BLOCK_FN_DIFF1` command (function
//! code 1 per `spec/03` §3.2) carrying `bs` per-sample residuals
//! `e₁(t) = s(t) − s(t − 1)` under `svar(e + 1)` mantissa width.  The
//! per-channel sample-history carry of `spec/05` §1 supplies the
//! initial `s(t − 1)` seed (`carry.at(0)`); the rolling `s_m1` slides
//! to the just-emitted sample for each subsequent residual, exactly
//! mirroring the decoder's order-1 reconstruction recurrence
//! `s(t) = s(t − 1) + e₁(t)`.  The order-1 predictor is mean-invariant
//! per `spec/05` §2 introductory paragraph (the running mean cancels
//! in the difference), so [`write_diff1_block`] takes no `mu_chan`
//! parameter.  [`min_energy_for_diff1`] picks the smallest natural
//! energy under the same svar-prefix-zero rule as DIFF0.
//!
//! The remaining predictor encoders (`BLOCK_FN_DIFF2..3` and
//! `BLOCK_FN_QLPC` residual production, plus the per-block channel-
//! round sequencer that picks `BLOCK_FN_DIFFn` order from a per-block
//! statistical objective per TR.156 §3.3) are deferred to later rounds.
//! `DIFF0` was the natural starting point because (a) its
//! reconstruction formula `s(t) = e₀(t) + μ_chan` involves no carry of
//! past samples, and (b) it's the only predictor whose residual
//! genuinely depends on the per-channel running mean (`spec/05` §2.3).
//! DIFF1 is the natural follow-on because the per-sample recurrence
//! window is one sample deep and the residual stream is the first-
//! differences of the source — a closed form that needs no statistical
//! fitting (in contrast to DIFF2/DIFF3, which carry the same
//! recurrence depth at orders 2 and 3).
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
use crate::predictor::{ChannelCarry, ENERGYSIZE};

/// `BLOCK_FN_VERBATIM` function-code numeric value (`spec/03` §3.10 /
/// `spec/04` §7). Re-exported as a public constant so callers building
/// per-command byte streams against the encoder's primitives can
/// produce the same wire-format token the decoder reads.
pub const FN_VERBATIM: u32 = 9;

/// `BLOCK_FN_QUIT` function-code numeric value (`spec/03` §3.8 /
/// `spec/04` §2).
pub const FN_QUIT: u32 = 4;

/// `BLOCK_FN_DIFF0` function-code numeric value (`spec/03` §3.1 /
/// `spec/04` §7). The order-0 polynomial-difference predictor — the
/// simplest of the four `DIFFn` predictors.
pub const FN_DIFF0: u32 = 0;

/// `BLOCK_FN_DIFF1` function-code numeric value (`spec/03` §3.2 /
/// `spec/04` §7). The order-1 polynomial-difference predictor —
/// `ŝ₁(t) = s(t − 1)`, residual `e₁(t) = s(t) − s(t − 1)`.
pub const FN_DIFF1: u32 = 1;

/// Largest energy encoded value the round-13 `DIFF0` encoder will pick
/// automatically.
///
/// The energy field is `uvar(ENERGYSIZE = 3)` carrying the value `e`
/// such that the residual mantissa width is `e + 1` per `spec/05` §3.
/// With `e = 7` the residual width is 8 bits, which covers any
/// `|s − μ_chan| < 128`. Larger residuals push the encoder out of the
/// "natural" uvar(3)-width energy field and into the prefix-zero
/// fallback (which still encodes correctly via `write_uvar`, but at
/// the cost of extra bits). [`min_energy_for_diff0`] caps its
/// auto-selection at `MAX_NATURAL_ENERGY` and surfaces
/// [`EncodeError::ResidualOutOfRange`] when no width up to `8` fits.
pub const MAX_NATURAL_ENERGY: u32 = 7;

/// The format version this encoder emits.
///
/// The reachable fixture corpus is v2-only per `spec/05` §7; the
/// decoder side accepts versions in `{1, 2, 3}` per `spec/00`
/// §"Format versions" but only v2 has byte-exact behavioural anchors.
/// The encoder therefore writes v2 unconditionally; v1 / v3 emission
/// awaits a `-v` fixture round per `spec/05` §7's open §9.4 candidate.
pub const ENCODER_VERSION: u8 = 2;

/// Errors the round-12 + round-13 + round-14 encoder primitives can
/// surface.
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
    /// The caller's `energy_encoded` parameter to a `BLOCK_FN_DIFFn`
    /// encoder would imply a residual mantissa width outside `1..=8`.
    /// The width is `energy_encoded + 1` per `spec/05` §3, so the
    /// accepted range is `0..=7`.
    EnergyOutOfRange(u32),
    /// The samples passed to a `BLOCK_FN_DIFFn` encoder (after
    /// subtracting the predictor's prediction) include a residual
    /// whose magnitude exceeds what [`MAX_NATURAL_ENERGY`]-width svar
    /// can hold without a prefix-zero blow-up. The caller can retry
    /// with an explicit `energy_encoded` and accept the wider
    /// encoding.
    ResidualOutOfRange(i64),
    /// The block-sample count exceeds `u32::MAX` or hits the
    /// implementation's safety cap on per-block samples.
    BlockTooLong(usize),
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
            EncodeError::EnergyOutOfRange(e) => write!(
                f,
                "oxideav-shorten: energy encoded value {e} implies residual width {} outside 1..=8",
                e.saturating_add(1)
            ),
            EncodeError::ResidualOutOfRange(r) => write!(
                f,
                "oxideav-shorten: DIFFn residual {r} exceeds the natural-energy auto-selection range"
            ),
            EncodeError::BlockTooLong(n) => write!(
                f,
                "oxideav-shorten: DIFFn block sample count {n} exceeds u32::MAX"
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

/// Compute the smallest energy encoded value `e ∈ 0..=MAX_NATURAL_ENERGY`
/// such that every `DIFF0` residual fits within the `svar(e + 1)`
/// mantissa with **zero** prefix-zero bits.
///
/// Per `spec/02` §2.2, `svar(n)` folds a signed `s` to an unsigned
/// `u = 2s` for `s ≥ 0` or `u = 2|s| − 1` for `s < 0`, then encodes `u`
/// as `uvar(n)`. The prefix-zero count of `uvar(n)` over value `u` is
/// `⌊u / 2^n⌋`; the encoding is most compact when `u < 2^n`, i.e. zero
/// prefix bits.
///
/// This helper picks the smallest `e ∈ 0..=7` such that the maximum
/// folded residual satisfies `u_max < 2^(e + 1)` — the natural energy
/// in `spec/05` §3.1's "smallest sensible `n` is 1" sense. It is **not**
/// the optimal Rice parameter of TR.156 §3.3 (which minimises total
/// encoded bit count, not just prefix bits); callers that want the
/// statistical optimum should compute it themselves and pass it as
/// `energy_encoded` to [`write_diff0_block`].
///
/// Returns `None` if no `e` in the natural range fits the largest
/// folded residual; the caller may either accept the prefix-zero cost
/// by passing `MAX_NATURAL_ENERGY` explicitly or fall back to a wider
/// (non-natural) width up to the decoder's `MAX_RESIDUAL_WIDTH = 30`
/// cap.
///
/// Per `spec/03` §3.1 the `DIFF0` residual is `s(t) − μ_chan`; this
/// helper takes the residuals already computed (the caller is
/// responsible for the subtraction).
pub fn min_energy_for_diff0(residuals: &[i64]) -> Option<u32> {
    min_natural_energy_for_residuals(residuals)
}

/// Compute the smallest energy encoded value `e ∈ 0..=MAX_NATURAL_ENERGY`
/// such that every `DIFF1` residual fits within the `svar(e + 1)`
/// mantissa with **zero** prefix-zero bits.
///
/// Per `spec/03` §3.2 the `DIFF1` residual is `e₁(t) = s(t) − s(t − 1)`
/// with `s(t − 1)` supplied by the per-channel sample-history carry of
/// `spec/05` §1 (zero on the very first block for each channel). The
/// caller is responsible for computing the first-differences and
/// passing them as `residuals`; this helper makes no assumption about
/// the carry seed.
///
/// The svar-mantissa-fit rule is identical to [`min_energy_for_diff0`]
/// — folding signed `r` to unsigned `u = 2r` (`r ≥ 0`) or `u = 2|r| − 1`
/// (`r < 0`) per `spec/02` §2.2 and finding the smallest `e` such that
/// `u_max < 2^(e + 1)`. The two helpers are kept separate by name so
/// the caller's intent is explicit at the call site and the helper
/// signature documents which predictor's residual definition the
/// caller has applied.
///
/// Returns `None` if no `e` in the natural range fits the largest
/// folded residual; the caller may either accept the prefix-zero cost
/// by passing `MAX_NATURAL_ENERGY` explicitly or fall back to a wider
/// (non-natural) width up to the decoder's `MAX_RESIDUAL_WIDTH = 30`
/// cap.
pub fn min_energy_for_diff1(residuals: &[i64]) -> Option<u32> {
    min_natural_energy_for_residuals(residuals)
}

/// Shared svar-mantissa-fit scan: smallest `e ∈ 0..=MAX_NATURAL_ENERGY`
/// such that every folded residual is strictly less than `2^(e + 1)`.
///
/// Reused by [`min_energy_for_diff0`] and [`min_energy_for_diff1`];
/// the per-predictor entrypoints exist to make the call site's intent
/// explicit. The helper carries no DIFFn-specific assumption — the
/// caller has already computed the residuals in the predictor's
/// definition.
fn min_natural_energy_for_residuals(residuals: &[i64]) -> Option<u32> {
    // Build a per-sample upper bound on the folded magnitude.
    let mut u_max: u64 = 0;
    for &r in residuals {
        let u: u64 = if r >= 0 {
            (r as u64).checked_shl(1)?
        } else {
            // |r| folded as 2|r| − 1; |r| is bounded by i64::MIN's
            // magnitude. Use unsigned arithmetic to avoid overflow.
            let mag = (r as i128).unsigned_abs() as u64;
            mag.checked_shl(1)?.checked_sub(1)?
        };
        if u > u_max {
            u_max = u;
        }
    }
    // Find the smallest e such that u_max < 2^(e + 1).
    for e in 0..=MAX_NATURAL_ENERGY {
        let cap = 1u64 << (e + 1);
        if u_max < cap {
            return Some(e);
        }
    }
    None
}

/// Emit a `BLOCK_FN_DIFF0` command (function-code numeric `0`) to
/// `writer` per `spec/03` §3.1 + `spec/05` §3.
///
/// Wire layout:
///
/// * `uvar(FNSIZE = 2)` over the value `FN_DIFF0 = 0`.
/// * `uvar(ENERGYSIZE = 3)` over `energy_encoded` (the residual width
///   is `energy_encoded + 1` per `spec/05` §3.1).
/// * `samples.len()` × `svar(energy_encoded + 1)` over each per-sample
///   residual `s(t) − μ_chan`.
///
/// The order-0 polynomial-difference predictor predicts zero for every
/// sample (`ŝ₀(t) = 0` per TR.156 eq. 3), so the residual stored on
/// the wire is the sample itself minus the per-channel running mean
/// (`spec/03` §3.1 + `spec/05` §2.3). When the mean estimator is
/// disabled (`H_meanblocks = 0`) the caller passes `mu_chan = 0` and
/// the residual reduces to the sample value directly.
///
/// The block advances the channel cursor on decode per `spec/03` §3.1.
///
/// Caller responsibilities:
///
/// * Choose `energy_encoded ∈ 0..=7`. The width applied is
///   `energy_encoded + 1` bits. [`min_energy_for_diff0`] picks the
///   smallest natural choice; a smaller value than necessary still
///   encodes correctly (the svar prefix-zero count grows) but is
///   wasteful.
/// * Provide the per-channel `μ_chan` if the mean estimator is
///   enabled; the encoder subtracts it from each sample before
///   folding-and-emitting the residual.
/// * Update the per-channel sample-history carry
///   ([`crate::ChannelCarry::update_after_block`]) and the mean
///   estimator ([`crate::MeanEstimator::record_block`]) after a
///   successful return — this function does not maintain any state.
///
/// Errors:
///
/// * [`EncodeError::EnergyOutOfRange`] if `energy_encoded > 7`.
/// * [`EncodeError::BlockTooLong`] if `samples.len() > u32::MAX as usize`.
///
/// The decoder side ([`crate::decode_diff_block`] with
/// [`crate::PolyOrder::Order0`]) consumes the emitted bits and
/// reconstructs `s(t) = e₀(t) + μ_chan` per `spec/05` §2.3.
pub fn write_diff0_block(
    writer: &mut BitWriter,
    energy_encoded: u32,
    samples: &[i32],
    mu_chan: i64,
) -> EncodeResult<()> {
    if energy_encoded > MAX_NATURAL_ENERGY {
        return Err(EncodeError::EnergyOutOfRange(energy_encoded));
    }
    if samples.len() > u32::MAX as usize {
        return Err(EncodeError::BlockTooLong(samples.len()));
    }
    let width = energy_encoded + 1;
    writer.write_uvar(FN_DIFF0, FNSIZE);
    writer.write_uvar(energy_encoded, ENERGYSIZE);
    for &s in samples {
        let residual = (s as i64) - mu_chan;
        writer.write_svar(residual, width);
    }
    Ok(())
}

/// Compute the per-sample `DIFF1` residual stream for `samples` given
/// the per-channel carry-supplied `s(t − 1)` seed.
///
/// Per `spec/03` §3.2 the residual is `e₁(t) = s(t) − s(t − 1)`. The
/// initial `s(t − 1)` comes from `carry.at(0)` (`spec/05` §1.1); the
/// rolling `s_m1` slides to each just-emitted sample as the recurrence
/// advances. The result is a `Vec<i64>` of length `samples.len()` —
/// `i64` headroom keeps the subtraction safe against `i32::MIN −
/// i32::MAX` underflow that would overflow plain `i32` arithmetic.
///
/// Used by the inline test harness to verify the recurrence direction
/// matches [`crate::decode_diff_block`]'s order-1 reconstruction
/// without re-implementing it. [`write_diff1_block`] inlines the
/// equivalent recurrence into its sample-emission loop to avoid an
/// intermediate allocation.
#[cfg(test)]
pub(crate) fn diff1_residuals(samples: &[i32], carry: &ChannelCarry) -> Vec<i64> {
    let mut s_m1: i64 = carry.at(0) as i64;
    let mut out: Vec<i64> = Vec::with_capacity(samples.len());
    for &s in samples {
        let s_i64 = s as i64;
        out.push(s_i64 - s_m1);
        s_m1 = s_i64;
    }
    out
}

/// Emit a `BLOCK_FN_DIFF1` command (function-code numeric `1`) to
/// `writer` per `spec/03` §3.2 + `spec/05` §3.
///
/// Wire layout:
///
/// * `uvar(FNSIZE = 2)` over the value `FN_DIFF1 = 1`.
/// * `uvar(ENERGYSIZE = 3)` over `energy_encoded` (the residual width
///   is `energy_encoded + 1` per `spec/05` §3.1).
/// * `samples.len()` × `svar(energy_encoded + 1)` over each per-sample
///   residual `e₁(t) = s(t) − s(t − 1)`.
///
/// The order-1 polynomial-difference predictor predicts the previous
/// sample (`ŝ₁(t) = s(t − 1)` per TR.156 eq. 4), so the residual stored
/// on the wire is the first-difference of the channel-local sample
/// stream. The very first sample of a channel-block reads `s(t − 1)`
/// from the per-channel sample-history carry (`spec/05` §1.1: index 0
/// is the most-recent past sample, zero-initialised at stream start);
/// subsequent samples within the block read their own predecessor.
///
/// DIFF1 is **mean-invariant** per `spec/05` §2 introductory paragraph
/// (the running mean cancels in the difference), so this function
/// takes no `mu_chan` parameter.
///
/// The block advances the channel cursor on decode per `spec/03` §3.2.
///
/// Caller responsibilities:
///
/// * Choose `energy_encoded ∈ 0..=7`. The width applied is
///   `energy_encoded + 1` bits. [`min_energy_for_diff1`] picks the
///   smallest natural choice; a smaller value than necessary still
///   encodes correctly (the svar prefix-zero count grows) but is
///   wasteful.
/// * Provide the per-channel sample-history `carry`; [`carry.at(0)`]
///   supplies the initial `s(t − 1)` seed. The mean estimator's state
///   is NOT consulted by the DIFF1 encoder (mean-invariant predictor).
/// * Update the per-channel sample-history carry
///   ([`crate::ChannelCarry::update_after_block`]) and the mean
///   estimator ([`crate::MeanEstimator::record_block`]) after a
///   successful return — this function does not maintain any state.
///
/// Errors:
///
/// * [`EncodeError::EnergyOutOfRange`] if `energy_encoded > 7`.
/// * [`EncodeError::BlockTooLong`] if `samples.len() > u32::MAX as usize`.
///
/// The decoder side ([`crate::decode_diff_block`] with
/// [`crate::PolyOrder::Order1`]) consumes the emitted bits and
/// reconstructs `s(t) = s(t − 1) + e₁(t)` per `spec/03` §3.2.
pub fn write_diff1_block(
    writer: &mut BitWriter,
    energy_encoded: u32,
    samples: &[i32],
    carry: &ChannelCarry,
) -> EncodeResult<()> {
    if energy_encoded > MAX_NATURAL_ENERGY {
        return Err(EncodeError::EnergyOutOfRange(energy_encoded));
    }
    if samples.len() > u32::MAX as usize {
        return Err(EncodeError::BlockTooLong(samples.len()));
    }
    let width = energy_encoded + 1;
    writer.write_uvar(FN_DIFF1, FNSIZE);
    writer.write_uvar(energy_encoded, ENERGYSIZE);
    let mut s_m1: i64 = carry.at(0) as i64;
    for &s in samples {
        let s_i64 = s as i64;
        let residual = s_i64 - s_m1;
        writer.write_svar(residual, width);
        s_m1 = s_i64;
    }
    Ok(())
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

    // ---- DIFF0 encoder ----

    #[test]
    fn min_energy_for_diff0_picks_zero_for_zero_residuals() {
        // Folded u_max = 0; 0 < 2^1 = 2, so e = 0 (width 1). Width 1
        // is the natural floor per `spec/05` §3.1's "smallest sensible
        // n is 1" — an all-zero block fits at encoded energy 0.
        assert_eq!(min_energy_for_diff0(&[0, 0, 0, 0]), Some(0));
    }

    #[test]
    fn min_energy_for_diff0_picks_smallest_width_per_max_residual() {
        // r = ±3 folds to u ∈ {6, 5}; 6 < 2^3 = 8 → e = 2 (width 3).
        assert_eq!(min_energy_for_diff0(&[3, -3, 0, 1]), Some(2));
        // r = ±10 folds to u ∈ {20, 19}; 20 < 2^5 = 32 → e = 4 (width 5).
        assert_eq!(min_energy_for_diff0(&[10, -7, -10, 1]), Some(4));
        // r = ±127 folds to u ∈ {254, 253}; 254 < 2^8 = 256 → e = 7 (width 8).
        assert_eq!(min_energy_for_diff0(&[127, -128, 0]), Some(7));
    }

    #[test]
    fn min_energy_for_diff0_returns_none_for_over_natural_range() {
        // r = 1000 folds to u = 2000; 2000 >= 2^8 = 256, exceeds e = 7.
        assert_eq!(min_energy_for_diff0(&[1000, 0]), None);
    }

    #[test]
    fn write_diff0_block_rejects_oversize_energy() {
        let mut w = BitWriter::new();
        assert_eq!(
            write_diff0_block(&mut w, 8, &[0, 0], 0),
            Err(EncodeError::EnergyOutOfRange(8))
        );
    }

    #[test]
    fn write_diff0_block_emits_function_code_then_energy_then_residuals() {
        // Hand-verify bit count: fn = uvar(2) over 0 = "100" (3 bits);
        // energy = uvar(3) over 0 = "1000" (4 bits); each residual is
        // svar(1) = uvar(1) over a folded u ∈ {0, 1}, which is 2 bits
        // (terminator + 1-bit mantissa). 2 residuals → 4 bits. Total 11.
        let mut w = BitWriter::new();
        write_diff0_block(&mut w, 0, &[0, 0], 0).expect("encode");
        assert_eq!(w.bits_written(), 11);
    }

    #[test]
    fn diff0_block_roundtrips_through_decode_diff_block() {
        use crate::bitreader::BitReader;
        use crate::block::{read_function_code, FunctionCode};
        use crate::predictor::{decode_diff_block, ChannelCarry, PolyOrder};

        // A representative spread of DIFF0 residual blocks at width
        // values covering e ∈ {0..7}.
        let cases: &[(u32, &[i32], i64)] = &[
            // (energy_encoded, samples, mu_chan)
            (0, &[0, 0, 0, -1], 0),
            (1, &[1, -1, 0, 1, -1], 0),
            (2, &[3, -3, 1, 0, -2], 0),
            (3, &[7, -7, 5, -5, 0, 1], 0),
            (4, &[15, -16, 0, 8, -8], 0),
            (7, &[100, -100, 64, -64], 0),
            // mu_chan != 0 — residuals are computed as s − μ.
            (3, &[105, 95, 110, 90], 100),
            (4, &[200, 100, 50, 250], 150),
        ];
        for (i, &(energy, samples, mu_chan)) in cases.iter().enumerate() {
            // write_diff0_block emits the function code + energy +
            // per-sample residuals — full DIFF0 command.
            let mut w = BitWriter::new();
            write_diff0_block(&mut w, energy, samples, mu_chan).expect("encode");
            let bytes = w.into_bytes();

            // Decode side: read the function code, then dispatch
            // into decode_diff_block (which starts at the energy
            // field).
            let mut r = BitReader::new(&bytes);
            let fc = read_function_code(&mut r).expect("read fn code");
            assert_eq!(fc, FunctionCode::Diff0, "case {i}");
            let carry = ChannelCarry::new(3);
            let block = decode_diff_block(
                &mut r,
                PolyOrder::Order0,
                samples.len() as u32,
                &carry,
                mu_chan,
            )
            .expect("decode_diff_block");
            assert_eq!(block, samples.to_vec(), "case {i}");
        }
    }

    #[test]
    fn write_diff0_block_then_quit_decodes_via_decode_stream() {
        // End-to-end: build a full Shorten file with header + DIFF0
        // block(s) + QUIT, then run it through the round-7
        // whole-stream decoder.
        //
        // Mono, default block size 4, no mean estimator, v2.
        let header = synth_header(2, 5, 1, 4, 0, 0, 0);
        let samples: Vec<i32> = vec![3, -2, 5, -1];
        let mu_chan: i64 = 0; // H_meanblocks = 0 → μ_chan = 0.
        let energy = min_energy_for_diff0(&samples.iter().map(|&s| s as i64).collect::<Vec<_>>())
            .expect("natural energy fits");

        let mut out = Vec::new();
        write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
        let mut writer = BitWriter::new();
        write_parameter_block(&mut writer, &header);
        write_diff0_block(&mut writer, energy, &samples, mu_chan).expect("diff0");
        write_quit_command(&mut writer);
        writer.pad_to_byte();
        out.extend(writer.into_bytes());

        let dec = decode_stream(&out).expect("decode stream");
        assert_eq!(dec.header, header);
        assert_eq!(dec.channels.len(), 1);
        assert_eq!(dec.channels[0], samples);
    }

    #[test]
    fn write_diff0_block_mean_consistency_full_roundtrip() {
        use crate::bitreader::BitReader;
        use crate::block::read_function_code;
        use crate::predictor::{decode_diff_block, ChannelCarry, PolyOrder};

        // Spec/05 §2.3 reconstruction: s(t) = e0(t) + μ_chan. Encoder
        // emits e0(t) = s(t) − μ_chan; decoder adds μ_chan back. The
        // sample value recovered must equal the original input
        // sample, confirming the encode/decode signs are consistent.
        let mu_chan: i64 = 1000;
        let samples: Vec<i32> = vec![1003, 998, 1005, 1001, 997];
        // Residuals: [3, -2, 5, 1, -3]; max folded u = 10 < 16 → e = 3.
        let energy: u32 = 3;

        let mut w = BitWriter::new();
        write_diff0_block(&mut w, energy, &samples, mu_chan).expect("encode");
        let bytes = w.into_bytes();

        let mut r = BitReader::new(&bytes);
        let _fc = read_function_code(&mut r).expect("fn code");
        let carry = ChannelCarry::new(3);
        let decoded = decode_diff_block(
            &mut r,
            PolyOrder::Order0,
            samples.len() as u32,
            &carry,
            mu_chan,
        )
        .expect("decode");
        assert_eq!(decoded, samples);
    }

    #[test]
    fn write_diff0_block_two_blocks_into_envelope_stream() {
        // Stereo, two channels, default block size 2, no mean. Two
        // DIFF0 blocks round-robin across the channels before QUIT.
        let header = synth_header(2, 5, 2, 2, 0, 0, 0);
        let ch0: Vec<i32> = vec![4, -1];
        let ch1: Vec<i32> = vec![-3, 2];
        let mu_chan: i64 = 0;
        let energy_ch0 = min_energy_for_diff0(&ch0.iter().map(|&s| s as i64).collect::<Vec<_>>())
            .expect("ch0 fits");
        let energy_ch1 = min_energy_for_diff0(&ch1.iter().map(|&s| s as i64).collect::<Vec<_>>())
            .expect("ch1 fits");

        let mut out = Vec::new();
        write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
        let mut writer = BitWriter::new();
        write_parameter_block(&mut writer, &header);
        // Round-robin: ch0 then ch1 (spec/03 §2).
        write_diff0_block(&mut writer, energy_ch0, &ch0, mu_chan).expect("ch0");
        write_diff0_block(&mut writer, energy_ch1, &ch1, mu_chan).expect("ch1");
        write_quit_command(&mut writer);
        writer.pad_to_byte();
        out.extend(writer.into_bytes());

        let dec = decode_stream(&out).expect("decode stream");
        assert_eq!(dec.channels.len(), 2);
        assert_eq!(dec.channels[0], ch0);
        assert_eq!(dec.channels[1], ch1);
    }

    // ---- DIFF1 encoder ----

    #[test]
    fn min_energy_for_diff1_picks_zero_for_zero_residuals() {
        // First-differences of a constant input fold to 0; u_max = 0 <
        // 2^1, so e = 0 (width 1). Matches DIFF0's behaviour on an
        // all-zero residual stream.
        assert_eq!(min_energy_for_diff1(&[0, 0, 0, 0]), Some(0));
    }

    #[test]
    fn min_energy_for_diff1_picks_smallest_width_per_max_residual() {
        // Same scan rule as min_energy_for_diff0 — these expectations
        // are duplicated rather than shared so a regression in the
        // shared helper is caught at both call sites.
        assert_eq!(min_energy_for_diff1(&[3, -3, 0, 1]), Some(2));
        assert_eq!(min_energy_for_diff1(&[10, -7, -10, 1]), Some(4));
        assert_eq!(min_energy_for_diff1(&[127, -128, 0]), Some(7));
    }

    #[test]
    fn min_energy_for_diff1_returns_none_for_over_natural_range() {
        assert_eq!(min_energy_for_diff1(&[1000, 0]), None);
    }

    #[test]
    fn write_diff1_block_rejects_oversize_energy() {
        let mut w = BitWriter::new();
        let carry = crate::predictor::ChannelCarry::new(3);
        assert_eq!(
            write_diff1_block(&mut w, 8, &[0, 0], &carry),
            Err(EncodeError::EnergyOutOfRange(8))
        );
    }

    #[test]
    fn write_diff1_block_emits_function_code_then_energy_then_residuals() {
        // fn = uvar(2) over 1: ⌊1/4⌋ = 0 leading zeros + terminator +
        //   mantissa `01` → `1 01` = 3 bits.
        // energy = uvar(3) over 0 = "1000" (4 bits).
        // 2 residuals at width 1 over folded u ∈ {0, 1}: 2 bits each →
        //   4 bits. Total 11.
        let mut w = BitWriter::new();
        let carry = crate::predictor::ChannelCarry::new(3);
        write_diff1_block(&mut w, 0, &[0, 0], &carry).expect("encode");
        assert_eq!(w.bits_written(), 11);
    }

    #[test]
    fn diff1_residuals_match_first_differences_against_zero_carry() {
        use crate::predictor::ChannelCarry;
        // Zero-initialised carry → s(t − 1) for the first sample is 0.
        let carry = ChannelCarry::new(3);
        let samples: &[i32] = &[5, 7, 4, 4, -3];
        let residuals = diff1_residuals(samples, &carry);
        // Hand-computed first-differences: [5-0, 7-5, 4-7, 4-4, -3-4]
        //                               = [5, 2, -3, 0, -7].
        assert_eq!(residuals, vec![5i64, 2, -3, 0, -7]);
    }

    #[test]
    fn diff1_residuals_seed_from_carry_at_zero() {
        use crate::predictor::ChannelCarry;
        let mut carry = ChannelCarry::new(3);
        // Pretend a previous block ended on sample 100 — update the
        // carry as the decoder would after `update_after_block`.
        carry.update_after_block(&[10, 20, 30, 100]);
        // carry.at(0) is now the last sample of the prior block = 100.
        assert_eq!(carry.at(0), 100);
        let samples: &[i32] = &[103, 110, 108];
        let residuals = diff1_residuals(samples, &carry);
        // First-differences seeded from 100: [103-100, 110-103, 108-110]
        //                                  = [3, 7, -2].
        assert_eq!(residuals, vec![3i64, 7, -2]);
    }

    #[test]
    fn diff1_block_roundtrips_through_decode_diff_block() {
        use crate::bitreader::BitReader;
        use crate::block::{read_function_code, FunctionCode};
        use crate::predictor::{decode_diff_block, ChannelCarry, PolyOrder};

        // Representative spread covering small and large per-sample
        // first-differences. Carry is zero-initialised at the start of
        // each case (matching a fresh-channel block).
        let cases: &[(u32, &[i32])] = &[
            (0, &[0, 0, 0, -1]),
            (1, &[1, -1, 0, 1, -1]),
            (2, &[3, 0, -3, 1, 2]),
            (3, &[7, 0, -7, 5, -5, 0, 1]),
            (4, &[15, 0, -16, 8, -8]),
            (7, &[100, 0, -100, 64, -64]),
        ];
        for (i, &(energy, samples)) in cases.iter().enumerate() {
            // Encode side: write the full DIFF1 command.
            let mut w = BitWriter::new();
            let enc_carry = ChannelCarry::new(3);
            write_diff1_block(&mut w, energy, samples, &enc_carry).expect("encode");
            let bytes = w.into_bytes();

            // Decode side: read fn code, then dispatch into
            // decode_diff_block (which starts at the energy field).
            let mut r = BitReader::new(&bytes);
            let fc = read_function_code(&mut r).expect("read fn code");
            assert_eq!(fc, FunctionCode::Diff1, "case {i}");
            let dec_carry = ChannelCarry::new(3);
            // mu_chan is ignored by Order1 — pass 0.
            let block = decode_diff_block(
                &mut r,
                PolyOrder::Order1,
                samples.len() as u32,
                &dec_carry,
                0,
            )
            .expect("decode_diff_block");
            assert_eq!(block, samples.to_vec(), "case {i}");
        }
    }

    #[test]
    fn diff1_block_roundtrips_with_non_zero_carry_seed() {
        use crate::bitreader::BitReader;
        use crate::block::read_function_code;
        use crate::predictor::{decode_diff_block, ChannelCarry, PolyOrder};

        // The encoder seeds s(t-1) from carry.at(0); the decoder reads
        // the same carry. Both sides start from a carry that has just
        // absorbed the prior block ending on sample 100.
        let mut enc_carry = ChannelCarry::new(3);
        enc_carry.update_after_block(&[10, 20, 30, 100]);
        let mut dec_carry = ChannelCarry::new(3);
        dec_carry.update_after_block(&[10, 20, 30, 100]);

        let samples: &[i32] = &[103, 110, 108, 107];
        let energy: u32 = 3; // residuals span ±7

        let mut w = BitWriter::new();
        write_diff1_block(&mut w, energy, samples, &enc_carry).expect("encode");
        let bytes = w.into_bytes();

        let mut r = BitReader::new(&bytes);
        let _fc = read_function_code(&mut r).expect("fn code");
        let block = decode_diff_block(
            &mut r,
            PolyOrder::Order1,
            samples.len() as u32,
            &dec_carry,
            0,
        )
        .expect("decode_diff_block");
        assert_eq!(block, samples.to_vec());
    }

    #[test]
    fn write_diff1_block_then_quit_decodes_via_decode_stream() {
        // Mono, default block size 5, no mean estimator, v2. DIFF1's
        // residual stream is the first-difference of the samples; the
        // decoder seeds s(t-1) = 0 from the zero-initialised carry.
        let header = synth_header(2, 5, 1, 5, 0, 0, 0);
        let samples: Vec<i32> = vec![3, 7, 4, 4, -3];
        let carry = crate::predictor::ChannelCarry::new(3);
        let residuals = diff1_residuals(&samples, &carry);
        let energy = min_energy_for_diff1(&residuals).expect("natural energy fits");

        let mut out = Vec::new();
        write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
        let mut writer = BitWriter::new();
        write_parameter_block(&mut writer, &header);
        write_diff1_block(&mut writer, energy, &samples, &carry).expect("diff1");
        write_quit_command(&mut writer);
        writer.pad_to_byte();
        out.extend(writer.into_bytes());

        let dec = decode_stream(&out).expect("decode stream");
        assert_eq!(dec.header, header);
        assert_eq!(dec.channels.len(), 1);
        assert_eq!(dec.channels[0], samples);
    }

    #[test]
    fn write_diff1_block_two_channels_round_robin() {
        // Stereo: channels round-robin on each emitted sample-producing
        // command. Each channel has its own zero-initialised carry.
        let header = synth_header(2, 5, 2, 3, 0, 0, 0);
        let ch0: Vec<i32> = vec![4, 6, 3];
        let ch1: Vec<i32> = vec![-2, 0, -5];
        let enc_carry_ch0 = crate::predictor::ChannelCarry::new(3);
        let enc_carry_ch1 = crate::predictor::ChannelCarry::new(3);
        let r0 = diff1_residuals(&ch0, &enc_carry_ch0);
        let r1 = diff1_residuals(&ch1, &enc_carry_ch1);
        let e0 = min_energy_for_diff1(&r0).expect("ch0 fits");
        let e1 = min_energy_for_diff1(&r1).expect("ch1 fits");

        let mut out = Vec::new();
        write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
        let mut writer = BitWriter::new();
        write_parameter_block(&mut writer, &header);
        // Per spec/03 §2: ch0 first, then ch1, then back to ch0…
        write_diff1_block(&mut writer, e0, &ch0, &enc_carry_ch0).expect("ch0");
        write_diff1_block(&mut writer, e1, &ch1, &enc_carry_ch1).expect("ch1");
        write_quit_command(&mut writer);
        writer.pad_to_byte();
        out.extend(writer.into_bytes());

        let dec = decode_stream(&out).expect("decode");
        assert_eq!(dec.channels.len(), 2);
        assert_eq!(dec.channels[0], ch0);
        assert_eq!(dec.channels[1], ch1);
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
