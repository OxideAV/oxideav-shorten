//! # oxideav-shorten
//!
//! **Status:** clean-room rebuild — round 2.
//!
//! The crate was orphan-rebuilt after the 2026-05-18 audit. Rounds
//! 1+2 land the **file-header parser** and the first slice of the
//! **per-block command stream** documented in
//! `docs/audio/shorten/spec/01-stream-header.md` through `spec/04`:
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
//!
//! No predictor / Rice residual decoder yet — those land in
//! subsequent rounds against `spec/03` §3.1..3.5 + `spec/05`.
//!
//! The public entry points are [`parse_stream_header`],
//! [`read_function_code`], and [`read_verbatim_payload`]. The
//! [`Error::NotImplemented`] sentinel remains available for any
//! API the orphan-rebuild scaffold has not yet wired up.
//!
//! ## Clean-room provenance
//!
//! Rounds 1+2 were implemented strictly from
//! `docs/audio/shorten/spec/00-scope.md` through `spec/05-…md`. No
//! external library source, no FFmpeg shorten source, no Tony
//! Robinson reference encoder source, and no archived `old` branch
//! of this crate was read at any phase. The behavioural anchors
//! driving the test suite (fixture `F1`'s header field values + end-
//! of-header bit position; the post-header `BLOCK_FN_VERBATIM`'s
//! 44-byte WAV-preamble recovery) come from `spec/02` §6 + §4.1 +
//! §4.5.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

mod bitreader;
mod block;
mod error;
mod header;

pub use crate::bitreader::{BitReader, ULONGSIZE};
pub use crate::block::{
    read_function_code, read_verbatim_payload, FunctionCode, VerbatimChunk, FNSIZE,
    VERBATIM_BYTE_SIZE, VERBATIM_CHUNK_SIZE, VERBATIM_MAX_LEN,
};
pub use crate::error::{Error, Result};
pub use crate::header::{
    parse_stream_header, ParsedHeader, ShortenStreamHeader, MAGIC, MIN_HEADER_BYTES,
};

/// No-op codec registration — rounds 1+2 land the file-header parser
/// plus the verbatim / quit slice of the per-block stream, but no
/// `oxideav-core` `Decoder` / `Encoder` is wired up yet. The
/// framework callback intentionally registers nothing into the
/// runtime context; a later round will add the Decoder/Encoder impls
/// once the predictor and Rice residual decode of `spec/03` §3.1..3.5
/// land.
#[cfg(feature = "registry")]
pub fn register(_ctx: &mut oxideav_core::RuntimeContext) {}

#[cfg(feature = "registry")]
oxideav_core::register!("shorten", register);
