//! Shorten (.shn) lossless audio decoder.
//!
//! Shorten is Tony Robinson's 1994 lossless waveform compressor (CUED
//! technical report TR.156, *"SHORTEN: Simple lossless and near-lossless
//! waveform compression"*). The bitstream is a 4-byte `'ajkg'` magic, a
//! version byte (`2` in everything observed in the wild), six adaptive-
//! Rice "ulong" header fields, a leading FN_VERBATIM capsule carrying
//! the original RIFF/AIFF header, and then a flat sequence of per-channel
//! audio commands (DIFF0..3, QLPC, ZERO) plus non-audio commands
//! (BLOCKSIZE, BITSHIFT, VERBATIM, QUIT). All numeric fields use one
//! of three Rice/Golomb flavours — there is no Huffman, no LZ77, no
//! arithmetic coding.
//!
//! # Round-1 coverage
//!
//! - Versions 0/1/2 streams are accepted; v >= 2 rounding biases applied.
//! - All fixed DIFFn predictors (orders 0..3).
//! - QLPC predictor with quantised coefficients (`maxnlpc <= 8`).
//! - ZERO blocks (silence shortcut).
//! - BLOCKSIZE / BITSHIFT / VERBATIM / QUIT non-audio commands.
//! - `internal_ftype` values 1..6 (S8 / U8 / S16HL / U16HL / S16LH / U16LH).
//! - Mono and multi-channel decode (channels independent — Shorten has
//!   no inter-channel decorrelation).
//! - The per-channel `nmean`-deep mean FIFO with v >= 2 rounding bias and
//!   bitshift compensation.
//! - Output is always packed S16 interleaved (suitable for downstream
//!   consumption); 8-bit and U16 inputs are normalised to S16 before
//!   emission. Sub-sample bit-depth fidelity is preserved (the decoder
//!   treats the post-shift integer as the "true" sample and only widens
//!   to S16 for the output buffer).
//!
//! # Round-1 encoder coverage
//!
//! - Bitwriter primitive (inverse of `BitReader`) and Rice/Golomb
//!   writers (unsigned, signed-zig-zag, adaptive ulong).
//! - Per-block predictor selection across DIFF0/1/2/3 (greedy: pick
//!   the predictor with the smallest sum of `|residual|`).
//! - FN_ZERO silence shortcut for all-zero blocks.
//! - Per-block Rice-`k` selection from `floor(log2(mean(|residual|)))`.
//! - Per-channel state mirroring the decoder: `nwrap = 3` history
//!   ring, `nmean`-deep mean FIFO with v >= 2 rounding bias.
//! - Round-robin channel interleave matching the decoder's emission
//!   order.
//! - All `internal_ftype` values 1..6 mapped from [`encoder::ShortenFtype`].
//! - Header magic + version `2` + 6 ulong fields + minimal leading
//!   FN_VERBATIM placeholder + trailing FN_QUIT.
//!
//! # Round-2 work (documented in CHANGELOG)
//!
//! - Streaming over multiple packets (today: each packet is treated as
//!   a self-contained `.shn` file).
//! - Producing native-bit-depth output (S8 / S32) instead of always S16.
//! - Demuxer / probe registration in `oxideav-container`.
//! - QLPC encoder (needs Levinson-Durbin + LPC quantisation at qshift=5).
//! - BITSHIFT encoder for streams with consistent low-zero-bits (e.g.
//!   24-bit-in-32-bit containers).
//! - Mid-stream FN_BLOCKSIZE so the encoder can handle non-multiple-
//!   of-blocksize inputs.
//! - Reading the leading VERBATIM capsule for sample-rate /
//!   bits-per-coded-sample (today the decoder skips the capsule bytes
//!   and relies on `CodecParameters::sample_rate` from the caller).

#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop)]

pub mod codec;
pub mod decoder;
pub mod encoder;
pub mod rice;

pub use encoder::{BlockInfo, ShortenEncoder, ShortenEncoderConfig, ShortenFtype};

use oxideav_core::CodecRegistry;

/// Canonical codec id matching FFmpeg's `shorten`.
pub const CODEC_ID_STR: &str = "shorten";

/// Register the Shorten decoder under [`CODEC_ID_STR`].
pub fn register_codecs(reg: &mut CodecRegistry) {
    codec::register(reg);
}
