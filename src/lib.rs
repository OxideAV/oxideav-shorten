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
//! ## Round 2 additions
//!
//! * **Production encoder** ([`encode`], [`EncoderConfig`]) — predictor
//!   search across `DIFF0..3` and `QLPC` (when `max_lpc_order > 0`),
//!   energy-width optimisation per TR.156 §3.3 eq. 21.
//! * **All eleven filetypes** of TR.156's `-t` enumeration. The
//!   numeric codes for the eight unpinned labels are chosen by this
//!   implementation per `header::Filetype` rustdoc; cross-implementation
//!   roundtrip on those codes is not guaranteed until `spec/05` §6
//!   pins them.
//! * **Container demuxer + 'ajkg' probe** registered on the
//!   `oxideav-core` `ContainerRegistry` under the name `"shorten"`. A
//!   `.shn` file emits a single packet containing the entire stream;
//!   the codec-side decoder handles framing internally.
//!
//! ## What's intentionally out (deferred to round 3+)
//!
//! * High-throughput optimisations (table-driven uvar prefix decode,
//!   SIMD residual unpacking).
//! * Lossy `BLOCK_FN_BITSHIFT` mode on the encoder side. The round-2
//!   encoder always writes `bshift = 0`.
//! * Mean-estimator on the encode side. The round-2 encoder writes
//!   `H_meanblocks = 0`, sidestepping the ±1 sub-bit-precision drift
//!   documented in `audit/01` §8.1. The decoder still handles
//!   `mean_blocks > 0` for streams produced externally.
//! * Format-version 1 / 3 wire-format deltas. No v1 or v3 fixture is
//!   reachable in the docs corpus; v1 is syntactically accepted (per
//!   the FFmpeg-observed T3 tampering test) but the v1 header layout
//!   is not behaviourally pinned.
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
pub use crate::encoder::{encode, EncodeError, EncoderConfig};
pub use crate::error::{Error, Result};
pub use crate::header::{parse_header, Filetype, StreamHeader, MAGIC};

#[cfg(feature = "registry")]
pub use crate::registry::{
    container_register, register, register_codecs, shorten_probe, CODEC_ID_STR, CONTAINER_NAME,
};

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

#[cfg(test)]
mod round2_tests;
