//! # oxideav-shorten
//!
//! **Status:** clean-room rebuild — round 1.
//!
//! The crate was orphan-rebuilt after the 2026-05-18 audit. Round 1
//! lands the **file-header parser** documented in
//! `docs/audio/shorten/spec/01-stream-header.md`:
//!
//! * The byte-aligned `ajkg` magic + one-byte format version field
//!   (`spec/01` §1).
//! * The six variable-length-integer parameter-block fields the v2/v3
//!   header carries (`spec/01` §3 — `H_filetype`, `H_channels`,
//!   `H_blocksize`, `H_maxlpcorder`, `H_meanblocks`, `H_skipbytes`).
//! * The MSB-first bit reader + `uvar(n)` / `ulong()` primitives of
//!   `docs/audio/shorten/spec/02-variable-length-coding.md` that the
//!   parameter block is encoded under.
//!
//! No per-block command stream, no predictor, no Rice residual
//! decoder yet — those land in subsequent rounds against `spec/03`
//! through `spec/05`.
//!
//! The public entry point is [`parse_stream_header`]. The
//! [`Error::NotImplemented`] sentinel remains available for any API
//! the orphan-rebuild scaffold has not yet wired up.
//!
//! ## Clean-room provenance
//!
//! Round 1 was implemented strictly from
//! `docs/audio/shorten/spec/00-scope.md` through `spec/02-…md` and
//! `spec/05-state-and-quirks.md` for the file-type-code anchors. No
//! external library source, no FFmpeg shorten source, no Tony
//! Robinson reference encoder source, and no archived `old` branch
//! of this crate was read at any phase. The four behavioural
//! anchors driving the test suite (fixture `F1`'s field values and
//! end-of-header bit position) come from `spec/02` §6.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

mod bitreader;
mod error;
mod header;

pub use crate::bitreader::{BitReader, ULONGSIZE};
pub use crate::error::{Error, Result};
pub use crate::header::{
    parse_stream_header, ParsedHeader, ShortenStreamHeader, MAGIC, MIN_HEADER_BYTES,
};

/// No-op codec registration — round 1 lands the file-header parser
/// only; no `oxideav-core` `Decoder` / `Encoder` is wired up yet, so
/// the framework callback intentionally registers nothing into the
/// runtime context. A later round will add the Decoder/Encoder impls
/// once `spec/03`–`spec/05` are implemented.
#[cfg(feature = "registry")]
pub fn register(_ctx: &mut oxideav_core::RuntimeContext) {}

#[cfg(feature = "registry")]
oxideav_core::register!("shorten", register);
