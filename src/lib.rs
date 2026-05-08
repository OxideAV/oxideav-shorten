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
//! ## Round 3 additions
//!
//! * **Levinson–Durbin LPC coefficient search** ([`encode`] when
//!   `max_lpc_order > 0`). The encoder solves the Yule-Walker system
//!   over the per-block-plus-carry autocorrelation to derive integer
//!   LPC coefficients (TR.156 §3.5 narrative). The polynomial-DIFF
//!   identity coefficient set is retained as a baseline candidate so
//!   regressions on flat-spectrum blocks fall back to the polynomial
//!   predictor rather than to a numerically-degenerate Levinson
//!   estimate.
//! * **`BLOCK_FN_BITSHIFT` lossy mode** on the encoder side. Setting
//!   [`EncoderConfig::with_bshift`] to a non-zero value emits a
//!   leading `BLOCK_FN_BITSHIFT` command with that count and
//!   right-shifts every input sample by `bshift` before encoding.
//!   The decoder inverts the shift on emission. Per `spec/04` §3 the
//!   command may appear anywhere in the stream; round 3 emits it once
//!   at stream start.
//!
//! ## Round 4 additions
//!
//! * **Running-mean estimator on encode side**
//!   ([`EncoderConfig::with_mean_blocks`]). The encoder mirrors the
//!   decoder's per-channel mean-buffer state (`spec/05` §2.5; the
//!   Validator-pinned C-truncation rule of `audit/01` §6.1) so that
//!   `BLOCK_FN_DIFF0` residuals are produced relative to `mu_chan`
//!   rather than zero, and constant-`mu_chan` blocks short-circuit to
//!   `BLOCK_FN_ZERO`. This closes `audit/01` §8.1's ±1 drift on
//!   `bshift > 0` lossy fixtures because encoder and decoder share the
//!   same arithmetic. Default remains `mean_blocks = 0` (the round-3
//!   wire format).
//!
//! ## Round 5 additions
//!
//! * **Bit-budget `-n N` / bit-rate `-r N` lossy modes**
//!   ([`EncoderConfig::with_bit_budget`] /
//!   [`EncoderConfig::with_bit_rate`]). Both compute an effective
//!   `bshift` such that the resulting per-sample post-Rice residual
//!   bit cost satisfies the target. Closes the encoder-side coverage
//!   gap for audit/01 fixtures `F10`/`F11`/`F14`/`F15`. Mutually
//!   exclusive with an explicit non-zero `bshift`.
//! * **Speed: table-driven `uvar` prefix decode.** The MSB-first
//!   bit reader exposes `read_uvar_prefix(max_zeros)` which consumes
//!   up to a full byte of zeros per LUT lookup
//!   ([`crate::bitreader`] `UVAR_PREFIX_LUT`). Wired into the
//!   [`varint::read_uvar`](crate) hot path used by every per-block
//!   command and per-residual decode.
//!
//! ## What's intentionally out (deferred to round 6+)
//!
//! * SIMD residual unpacking (the LUT-driven uvar prefix decode lands
//!   in round 5; SIMD batching of the residual mantissa reads is the
//!   next throughput tier).
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
pub use crate::encoder::{encode, EncodeError, EncoderConfig, BITSHIFT_MAX, MEAN_BLOCKS_MAX};
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

#[cfg(test)]
mod round3_tests;

#[cfg(test)]
mod round4_tests;

#[cfg(test)]
mod round5_tests;
