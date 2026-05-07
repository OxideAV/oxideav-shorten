//! Pure-Rust Shorten lossless audio codec.
//!
//! **Round 1 — clean-room rebuild from `docs/audio/shorten/`.** This
//! decoder consumes the v2/v3 wire format pinned in
//! `spec/00..05` (with v1 syntactically accepted; no v1 fixture is
//! reachable to confirm the v1 layout). The implementation reads
//! TR.156's narrative + the spec set's variable-length integer rules
//! + behavioural anchors on fixtures `F1..F18`; **no** code is
//!   adapted from FFmpeg's `libavcodec/shorten.c`, Tony Robinson's
//!   reference C source, or any third-party Shorten implementation.
//!
//! ## What's implemented (round 1)
//!
//! * Header parse — magic + version + the six variable-length integer
//!   parameter-block fields (`H_filetype`, `H_channels`,
//!   `H_blocksize`, `H_maxlpcorder`, `H_meanblocks`, `H_skipbytes`).
//! * Per-block command stream — function codes 0..=9 per
//!   `spec/04`:
//!   `BLOCK_FN_DIFF0..3` (polynomial-difference predictors,
//!   TR.156 §3.2 eq. 3..6),
//!   `BLOCK_FN_QLPC` (general LPC predictor, TR.156 §3.2 eq. 1),
//!   `BLOCK_FN_BLOCKSIZE` (sub-block-size override),
//!   `BLOCK_FN_BITSHIFT` (lossy shift),
//!   `BLOCK_FN_VERBATIM` (inline byte payload),
//!   `BLOCK_FN_QUIT` (end-of-stream),
//!   `BLOCK_FN_ZERO` (constant-mean block).
//! * Rice / Golomb residual coding — `uvar(n)` / `svar(n)` /
//!   `ulong()`, energy-field-plus-one residual width
//!   (`spec/05` §3).
//! * Per-channel sample-history carry of `max(3, H_maxlpcorder)`
//!   samples (`spec/05` §1).
//! * Running-mean estimator with the
//!   `mu = (sum + n/2) / n` truncation-toward-zero rule
//!   (`spec/05` §2.5).
//! * Pinned filetypes `u8 = 2`, `s16hl = 3`, `s16lh = 5`
//!   (`spec/05` §6).
//! * Mono and stereo PCM output (i32 lanes; the
//!   `oxideav-core`-integrated decoder packs to `u8` / `s16` per the
//!   stream's filetype and the framework's [`SampleFormat`] target).
//!
//! ## What's intentionally out (deferred to round 2+)
//!
//! * High-throughput optimisations (table-driven uvar prefix decode,
//!   SIMD residual unpacking).
//! * `BLOCK_FN_QLPC` byte-exactness verified through the TR.156 test
//!   oracle (Variant B asymmetric oracle): the coefficient
//!   quantisation domain is the encoder's choice; the round-1
//!   decoder applies coefficients verbatim per `spec/03` §3.5.
//! * Production encoder. The crate's test scaffold ships a minimal
//!   encoder for self-roundtrip; a real encoder belongs in round 2.
//!
//! ## Cargo features
//!
//! * **`registry`** (default): wire the crate's `register(ctx)` entry
//!   point into `oxideav-core`'s codec registry. Disable for
//!   standalone builds that want the decoder without the framework
//!   dependency.

#![forbid(unsafe_code)]

mod bitreader;
mod decoder;
mod encoder;
mod error;
mod header;
#[cfg(feature = "registry")]
mod registry;
mod varint;

pub use crate::decoder::{decode, DecodedStream};
pub use crate::error::{Error, Result};
pub use crate::header::{parse_header, Filetype, StreamHeader, MAGIC};

#[cfg(feature = "registry")]
pub use crate::registry::{register, register_codecs, CODEC_ID_STR};

/// Maximum channels accepted by the decoder. The Hydrogenaudio entry
/// notes Shorten "lacks ... support for multichannel ... and high
/// sampling rates"; we cap above the realistic 2-channel use case
/// to admit any encoder-emitted small count without a hard limit.
pub const MAX_CHANNELS: u16 = 8;

/// Maximum block size accepted by the decoder. TR.156's default is
/// 256 samples per channel; production encoders cap blocksize well
/// below 65k. `0x10_0000` keeps headroom for a permissive but
/// sanity-checked `H_blocksize` field.
pub const MAX_BLOCKSIZE: u32 = 0x10_0000;

/// Maximum LPC order accepted by the decoder. TR.156's man-page
/// notes `-p prediction order` defaults to 0 (LPC disabled); the
/// reference encoder tops out around 32 in practice.
pub const MAX_LPC_ORDER: u32 = 32;

#[cfg(feature = "registry")]
oxideav_core::register!("oxideav-shorten", register);

#[cfg(test)]
mod roundtrip_tests;
