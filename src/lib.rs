//! # oxideav-shorten
//!
//! **Status:** clean-room rebuild — round 7.
//!
//! The crate was orphan-rebuilt after the 2026-05-18 audit. Rounds
//! 1+2+3+4+5+6 land the **file-header parser**, the per-block command
//! dispatch (every code 0..=9 has a payload decoder), the
//! polynomial-difference predictor kernels, the per-channel running
//! mean estimator, the quantised-LPC predictor, and the
//! `BLOCK_FN_BLOCKSIZE` / `BLOCK_FN_BITSHIFT` housekeeping commands.
//! Round 7 ties them together with the **full-stream decode driver**
//! ([`decode_stream`]) of the integer-PCM decode path documented in
//! `docs/audio/shorten/spec/01-stream-header.md` through `spec/05`:
//!
//! * Round 1 — byte-aligned `ajkg` magic + one-byte format version
//!   (`spec/01` §1), the six variable-length-integer parameter-block
//!   fields the v2/v3 header carries (`spec/01` §3), and the
//!   MSB-first bit reader plus `uvar(n)` / `ulong()` primitives of
//!   `docs/audio/shorten/spec/02-variable-length-coding.md`.
//! * Round 2 — the signed `svar(n)` primitive (`spec/02` §2.2,
//!   one's-complement folding) plus the per-block command dispatch:
//!   the [`FunctionCode`] enumeration of all ten v2/v3 codes
//!   (`spec/03` §3 + `spec/04`), the [`read_function_code`] helper,
//!   and full payload decode for `BLOCK_FN_VERBATIM` (`spec/03`
//!   §3.10 + `spec/02` §4.5) and `BLOCK_FN_QUIT` (`spec/03` §3.8 +
//!   `spec/04` §2).
//! * Round 3 — the polynomial-difference predictor kernels of orders
//!   0..3 (`spec/03` §3.1..§3.4), the per-channel sample-history
//!   carry (`spec/03` §3.11 / `spec/05` §1), the Rice-coded residual
//!   decode with the `+1`-adjusted mantissa width (`spec/05` §3 / T15),
//!   and the [`decode_diff_block`] entry point that ties them together.
//! * Round 4 — the per-channel running mean estimator
//!   ([`MeanEstimator`]) of `spec/05` §2 + §2.5 with the
//!   validation-corrected truncate-toward-zero `+ divisor/2` arithmetic
//!   for both the per-block mean and the channel-wide running mean.
//!   `decode_diff_block` now takes a `mu_chan: i64` parameter, consumed
//!   by `PolyOrder::Order0` per `spec/05` §2.3 and ignored (mean-
//!   invariant) by orders 1..3 per `spec/05` §2 introductory paragraph.
//!   The new [`fill_zero_block`] helper emits `BLOCK_FN_ZERO`'s
//!   payload — `bs` samples all equal to `mu_chan` per `spec/05` §2.4.
//! * Round 5 — the quantised-LPC predictor [`decode_qlpc_block`] of
//!   `spec/03` §3.5 (`BLOCK_FN_QLPC`): per-block order (`uvar(LPCQSIZE
//!   = 2)`), `order` quantised coefficients (`svar(LPCQUANT = 2)`),
//!   the shared energy field + `+1` residual-width adjustment
//!   (`spec/02` §4.3/§4.4 + `spec/05` §3.2), and the general LPC
//!   reconstruction `s(t) = Σᵢ aᵢ·s(t-i) + e(t)` applied without
//!   scaling per `spec/03` §3.5. The predictor is mean-invariant
//!   (`spec/03` §3.12), reads its history window from the per-channel
//!   carry (`spec/05` §1), and feeds each just-emitted sample back as
//!   the next prediction's most-recent past sample.
//! * Round 6 — the two housekeeping commands that mutate per-stream
//!   decoder state without producing samples:
//!   [`read_blocksize_payload`] decodes `BLOCK_FN_BLOCKSIZE`'s
//!   `new_bs` (`ulong()`) sub-block-size override (`spec/03` §3.6 /
//!   `spec/04` §4), and [`read_bitshift_payload`] decodes
//!   `BLOCK_FN_BITSHIFT`'s `bshift` (`uvar(BITSHIFTSIZE = 2)`)
//!   per-stream bit-shift amount (`spec/02` §4.6 + `spec/03` §3.7 /
//!   `spec/04` §3). Neither command advances the channel cursor; the
//!   higher-level driver swaps the returned value into its running
//!   sub-block-size or bit-shift state and continues per-channel
//!   dispatch on the same channel.
//!
//! * Round 7 — the full-stream decode driver [`decode_stream`]
//!   (`spec/03` §2 + §3.6/§3.7 + §3.8 + `spec/05` §1 + §1.4 + §2):
//!   parses the header, seeks past it, and runs the per-block command
//!   loop, carrying the round-robin channel cursor, the running
//!   sub-block size, the running bit-shift, the per-channel carries,
//!   and the per-channel mean estimators across the entire stream
//!   until `BLOCK_FN_QUIT`. Returns a [`DecodedStream`] holding the
//!   verbatim prefix plus per-channel `Vec<i32>` sample vectors. The
//!   per-channel carry stores pre-shift samples (`spec/05` §1.4); the
//!   driver applies the bit-shift on emission only.
//!
//! With round 7 the integer-PCM decode path is end-to-end: every
//! per-block command 0..=9 is dispatched by the driver. The next step
//! is the `oxideav-core` `Decoder` impl wired into `register(ctx)`
//! (sample-format byte packing per the `spec/05` §6 file-type table,
//! and the host-format envelope emission from the verbatim prefix).
//!
//! The public entry points are [`decode_stream`], [`parse_stream_header`],
//! [`read_function_code`], [`read_verbatim_payload`],
//! [`read_blocksize_payload`], [`read_bitshift_payload`],
//! [`decode_diff_block`], [`decode_qlpc_block`], [`fill_zero_block`],
//! and [`MeanEstimator`]. The [`Error::NotImplemented`] sentinel
//! remains available for any API the orphan-rebuild scaffold has not
//! yet wired up.
//!
//! ## Clean-room provenance
//!
//! Rounds 1+2+3+4+5+6 were implemented strictly from
//! `docs/audio/shorten/spec/00-scope.md` through `spec/05-…md`. No
//! external library source, no FFmpeg shorten source, no Tony
//! Robinson reference encoder source, no archived `old` branch of
//! this crate, and no `reference-impl/python/` material (forbidden
//! to the Implementer per `docs/IMPLEMENTOR_ROUND.md` §"Special
//! case: clean-room rebuild rounds") was read at any phase. The
//! behavioural anchors driving the test suite (fixture `F1`'s
//! header field values + end-of-header bit position; the post-header
//! `BLOCK_FN_VERBATIM`'s 44-byte WAV-preamble recovery; the per-
//! channel carry indexing convention; the mean estimator's
//! validation-corrected `+ divisor/2` always-positive bias; the
//! `F2` `BLOCK_FN_BLOCKSIZE` tail-block override at command 11,377
//! with `new_bs = 155`; the `F5..F8` `BLOCK_FN_BITSHIFT` parameter
//! values 1/4/8/12 matching the encoder's `-q N` invocation) come
//! from `spec/02` §6 + §4.1 + §4.5 + §4.6, `spec/03` §3, `spec/04`
//! §3 + §4, and `spec/05` §1 + §2.5 + §3.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

mod bitreader;
mod block;
mod driver;
mod error;
mod header;
mod predictor;

pub use crate::bitreader::{BitReader, ULONGSIZE};
pub use crate::block::{
    read_bitshift_payload, read_blocksize_payload, read_function_code, read_verbatim_payload,
    FunctionCode, VerbatimChunk, BITSHIFTSIZE, BITSHIFT_MAX, BLOCKSIZE_MAX, FNSIZE,
    VERBATIM_BYTE_SIZE, VERBATIM_CHUNK_SIZE, VERBATIM_MAX_LEN,
};
pub use crate::driver::{decode_stream, DecodedStream, MAX_COMMANDS};
pub use crate::error::{Error, Result};
pub use crate::header::{
    parse_stream_header, ParsedHeader, ShortenStreamHeader, MAGIC, MIN_HEADER_BYTES,
};
pub use crate::predictor::{
    decode_diff_block, decode_qlpc_block, fill_zero_block, ChannelCarry, MeanEstimator, PolyOrder,
    CARRY_LEN_FLOOR, ENERGYSIZE, LPCQSIZE, LPCQUANT,
};

/// No-op codec registration — rounds 1..=7 land the file-header
/// parser, the per-block command dispatch (every code 0..=9 has a
/// payload decoder), the polynomial-difference predictor kernels, the
/// running mean estimator, the quantised-LPC predictor, the
/// `BLOCK_FN_BLOCKSIZE` / `BLOCK_FN_BITSHIFT` housekeeping commands,
/// and the full-stream decode driver [`decode_stream`] that
/// orchestrates them into per-channel `Vec<i32>` sample streams.
/// What still needs to land before this becomes a real
/// `oxideav-core::Decoder` is the sample-format byte-packing layer (the
/// `spec/05` §6 file-type table mapping the reconstructed `i32`
/// channel samples + the verbatim host-format envelope into the
/// container the registry expects).
#[cfg(feature = "registry")]
pub fn register(_ctx: &mut oxideav_core::RuntimeContext) {}

#[cfg(feature = "registry")]
oxideav_core::register!("shorten", register);
