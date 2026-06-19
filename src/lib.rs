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
//! * Round 9 — the [`SHNAMPSK`-tagged seek-table trailer detector]
//!   ([`detect_shnampsk_trailer`] / [`split_off_shnampsk_trailer`])
//!   per `spec/05` §5.1 + §5.2 + §5.3. Several publicly-distributed
//!   `.shn` files carry a non-standard sidecar appended after the
//!   encoder's `BLOCK_FN_QUIT` zero-bit padding (added by Wayne
//!   Stielau's seek-table utility per `spec/05` §5); the detector
//!   identifies the trailer's last 12 bytes (4-byte LE `len_u32` +
//!   8-byte `SHNAMPSK` signature) and the sidecar's `SEEK` magic
//!   anchor at `len(file) − len_u32` so the caller can either ignore
//!   the trailer (the default — [`decode_stream`] terminates at
//!   `BLOCK_FN_QUIT` and never reads into the sidecar) or hand it
//!   to an external seek-table parser. The detector returns `None`
//!   when the signature is absent (matches fixture `F9` /
//!   `Choppy.shn` per `spec/05` §5.3) and surfaces
//!   [`Error::MalformedShnampskTrailer`] when the signature is
//!   present but the `len_u32` / `SEEK` anchor pair is structurally
//!   inconsistent.
//!
//! * Round 8 — the [`oxideav_core::Decoder`] trait wrapper
//!   [`ShortenDecoder`] (`spec/05` §6 file-type table). Adapts the
//!   round-7 whole-stream driver into the framework's packet-in /
//!   frame-out shape: `send_packet` buffers bytes until
//!   [`decode_stream`] returns successfully, then emits one
//!   [`oxideav_core::AudioFrame`] holding planar PCM packed per the
//!   `H_filetype` byte-order table (filetype `2`/`u8` → 1 byte,
//!   `3`/`s16hl` → big-endian `i16`, `5`/`s16lh` → little-endian
//!   `i16`). [`make_decoder`] / [`register_codecs`] expose the
//!   factory under the codec id `"shorten"`; [`register`] now wires
//!   it into a `RuntimeContext`'s codec registry. The encoder side
//!   stays unimplemented — that's the remaining README "lacks" tail.
//!
//! * Round 10 — the block-by-block streaming decode iterator
//!   [`StreamDecoder`] / [`DecodedBlock`] / [`decode_stream_iter`]
//!   (`spec/03` §2 + §3.6/§3.7/§3.8/§3.10 + `spec/05` §1.4 + §2).
//!   Walks the same per-block command loop as [`decode_stream`] but
//!   yields each sample-producing block (or `VERBATIM` envelope
//!   payload) to the caller one at a time and discards it after the
//!   carry/mean state has been updated — memory bounded by the header
//!   parameters and independent of stream length, in contrast to
//!   [`decode_stream`] which accumulates every decoded sample into
//!   [`DecodedStream::channels`] ahead of the caller.
//!
//! With round 7 the integer-PCM decode path is end-to-end; round 8
//! ties it into the framework's resolver; round 10 adds an
//! on-demand iterator surface. Every per-block command 0..=9 is
//! dispatched by both driver shapes, and a `RuntimeContext` can
//! resolve a Shorten decoder by codec id.
//!
//! * Round 12 — the **encoder side's envelope and bit-level
//!   primitives** (`spec/01` §1 + §3 + `spec/02` §1..§3 + `spec/03`
//!   §3.8 + §3.10 + `spec/04` §2 + §7 + `spec/05` §4). The MSB-first
//!   [`BitWriter`] is the bit-level dual of [`BitReader`];
//!   [`encode_envelope_stream`] composes a syntactically-valid
//!   Shorten file from a `(ShortenStreamHeader, &[u8]
//!   verbatim_prefix)` pair using the header / verbatim / QUIT
//!   primitives ([`write_stream_header`], [`write_verbatim_block`],
//!   [`write_quit_command`]). Output round-trips losslessly through
//!   [`decode_stream`].
//!
//! * Round 13 — the **`BLOCK_FN_DIFF0` predictor encoder**
//!   ([`write_diff0_block`]) per `spec/03` §3.1 + `spec/05` §3.
//!   Emits the order-0 polynomial-difference predictor's per-block
//!   command (function code 0 + `uvar(ENERGYSIZE = 3)` energy +
//!   `bs × svar(energy + 1)` residuals) for samples whose residual
//!   `e₀(t) = s(t) − μ_chan` decoder-reconstructs via
//!   [`decode_diff_block`] with [`PolyOrder::Order0`].
//!   [`min_energy_for_diff0`] picks the natural energy parameter
//!   (smallest `e ∈ 0..=7` such that every folded residual fits
//!   inside the svar mantissa with zero prefix-zero bits, matching
//!   `spec/05` §3.1's "smallest sensible n is 1" floor). DIFF0 is
//!   the only predictor whose residual genuinely depends on the
//!   per-channel running mean (`spec/05` §2.3) — orders 1..3 are
//!   mean-invariant by §2 introductory paragraph.
//!
//! * Round 14 — the **`BLOCK_FN_DIFF1` predictor encoder**
//!   ([`write_diff1_block`]) per `spec/03` §3.2 + `spec/05` §1 + §3.
//!   Emits the order-1 polynomial-difference predictor's per-block
//!   command (function code 1 + `uvar(ENERGYSIZE = 3)` energy +
//!   `bs × svar(energy + 1)` residuals) carrying the per-sample
//!   first-differences `e₁(t) = s(t) − s(t − 1)`. The very first
//!   `s(t − 1)` of a channel-block comes from the per-channel sample-
//!   history carry (`spec/05` §1.1: index 0 is the most-recent past
//!   sample, zero-initialised at stream start); subsequent samples
//!   read their own predecessor. The decoder side
//!   [`decode_diff_block`] with [`PolyOrder::Order1`] reconstructs by
//!   `s(t) = s(t − 1) + e₁(t)`. DIFF1 is mean-invariant per `spec/05`
//!   §2 introductory paragraph. [`min_energy_for_diff1`] picks the
//!   smallest natural energy under the same svar-prefix-zero rule as
//!   DIFF0.
//!
//! * Round 15 — the **`BLOCK_FN_DIFF2` predictor encoder**
//!   ([`write_diff2_block`]) per `spec/03` §3.3 + `spec/05` §1 + §3.
//!   Emits the order-2 polynomial-difference predictor's per-block
//!   command (function code 2 + `uvar(ENERGYSIZE = 3)` energy +
//!   `bs × svar(energy + 1)` residuals) carrying the per-sample
//!   second-differences `e₂(t) = s(t) − (2·s(t − 1) − s(t − 2))`. The
//!   first two `s(t − k)` values of a channel-block come from
//!   `carry.at(0)` (`s(t − 1)`) and `carry.at(1)` (`s(t − 2)`) per
//!   `spec/05` §1.1, both zero-initialised at stream start; subsequent
//!   samples slide the window forward against just-emitted samples.
//!   The decoder side [`decode_diff_block`] with [`PolyOrder::Order2`]
//!   reconstructs by `s(t) = 2·s(t − 1) − s(t − 2) + e₂(t)`. DIFF2 is
//!   mean-invariant per `spec/05` §2 introductory paragraph.
//!   [`min_energy_for_diff2`] picks the smallest natural energy under
//!   the same svar-prefix-zero rule as DIFF0 / DIFF1.
//!
//! * Round 18 — the **`BLOCK_FN_ZERO` sentinel encoder**
//!   ([`write_zero_block`]) per `spec/03` §3.9 + `spec/04` §6 +
//!   `spec/05` §2.4. Emits the constant-block command (function code
//!   8) as a bare `uvar(FNSIZE = 2)` field with no further payload.
//!   The decoder side ([`fill_zero_block`] + the round-7 driver's
//!   `FunctionCode::Zero` arm) reconstructs `bs` samples all equal
//!   to the channel's running-mean estimate `μ_chan` (zero when
//!   `H_meanblocks = 0`). The block advances the channel cursor on
//!   decode. The total encoded cost is 5 bits — the cheapest sample-
//!   producing command in the format, hence its name. The caller is
//!   responsible for only emitting ZERO when the source block
//!   genuinely consists of `bs` copies of the current `μ_chan` at
//!   the producer end; the encoder does not verify this because the
//!   per-block `μ_chan` value is the higher-level sequencer's
//!   knowledge, not the writer primitive's.
//!
//! * Round 16 — the **`BLOCK_FN_DIFF3` predictor encoder**
//!   ([`write_diff3_block`]) per `spec/03` §3.4 + `spec/05` §1 + §3.
//!   Emits the order-3 polynomial-difference predictor's per-block
//!   command (function code 3 + `uvar(ENERGYSIZE = 3)` energy +
//!   `bs × svar(energy + 1)` residuals) carrying the per-sample
//!   third-differences
//!   `e₃(t) = s(t) − (3·s(t − 1) − 3·s(t − 2) + s(t − 3))` (TR.156
//!   §3.2 eq. 6). The first three `s(t − k)` values of a channel-block
//!   come from `carry.at(0)` (`s(t − 1)`), `carry.at(1)` (`s(t − 2)`),
//!   and `carry.at(2)` (`s(t − 3)`) per `spec/05` §1.1, all
//!   zero-initialised at stream start; subsequent samples slide the
//!   window forward against just-emitted samples. The decoder side
//!   [`decode_diff_block`] with [`PolyOrder::Order3`] reconstructs by
//!   `s(t) = 3·s(t − 1) − 3·s(t − 2) + s(t − 3) + e₃(t)`. DIFF3 is
//!   mean-invariant per `spec/05` §2 introductory paragraph.
//!   [`min_energy_for_diff3`] picks the smallest natural energy under
//!   the same svar-prefix-zero rule as DIFF0 / DIFF1 / DIFF2. Round 16
//!   closes the polynomial-difference predictor family on the encoder
//!   side; the `BLOCK_FN_QLPC` predictor encoder + the per-block
//!   channel-round sequencer remain unwritten.
//!
//! * Round 251 — the **per-block predictor-selection sequencer**
//!   per `spec/03` §3.1..§3.4 + §3.9 + `spec/02` §2.1 + §2.2 +
//!   `spec/05` §2.3 + §2.4. [`select_predictor`] compares the
//!   cheapest of `BLOCK_FN_DIFF0..3` and `BLOCK_FN_ZERO` for a
//!   given channel block by computing each candidate's natural-
//!   energy total encoded bit count (function code + energy field +
//!   per-sample `svar(energy + 1)` residuals) and picks the
//!   candidate with the smallest cost. Ties break in priority order
//!   `ZERO > DIFF0 > DIFF1 > DIFF2 > DIFF3`. [`write_selected_block`]
//!   dispatches the returned [`Choice`] to the matching per-
//!   predictor writer; [`evaluate_candidates`] returns every
//!   eligible candidate in priority order for inspection.
//!   `BLOCK_FN_QLPC` is not part of round-251 auto-selection; round
//!   266 layered it in behind a caller-supplied candidate coefficient
//!   vector ([`select_predictor_with_qlpc`]).
//!
//! * Round 282 — **QLPC auto-derivation** in the selection sequencer
//!   per `spec/03` §3.5 + `spec/02` §4.3 + §4.4.
//!   [`select_predictor_auto`] / [`evaluate_candidates_auto`] run the
//!   zero-anchored per-block order search over `0..=max_lpc_order`
//!   themselves; [`derive_qlpc_coefs`] derives each order's quantised
//!   coefficient vector (least-squares normal equations over the
//!   carry-seeded prediction contexts, rounded into `spec/03` §3.5's
//!   unscaled signed-integer coefficient domain) and
//!   [`derive_qlpc_candidate`] scores every order under the full cost
//!   model — order field + coefficient-transmission overhead + Rice-n
//!   optimal residual stream — so `BLOCK_FN_QLPC` is picked exactly
//!   when it is genuinely cheapest. `max_lpc_order = 0` reproduces
//!   [`select_predictor`] bit-for-bit.
//!
//! The public entry points are [`decode_stream`], [`parse_stream_header`],
//! [`read_function_code`], [`read_verbatim_payload`],
//! [`read_blocksize_payload`], [`read_bitshift_payload`],
//! [`decode_diff_block`], [`decode_qlpc_block`], [`fill_zero_block`],
//! [`MeanEstimator`], plus the round-8 trait wiring [`ShortenDecoder`]
//! / [`make_decoder`] / [`register_codecs`] and the round-12 encoder
//! surface [`BitWriter`] / [`encode_envelope_stream`] /
//! [`write_stream_header`] / [`write_verbatim_block`] /
//! [`write_quit_command`] plus the round-13 [`write_diff0_block`] /
//! [`min_energy_for_diff0`], round-14 [`write_diff1_block`] /
//! [`min_energy_for_diff1`], round-15 [`write_diff2_block`] /
//! [`min_energy_for_diff2`], and round-16 [`write_diff3_block`] /
//! [`min_energy_for_diff3`] predictor encoders plus the round-18
//! [`write_zero_block`] sentinel encoder. The
//! [`Error::NotImplemented`] sentinel remains available for any API
//! the orphan-rebuild scaffold has not yet wired up.
//!
//! ## Clean-room provenance
//!
//! Rounds 1+2+3+4+5+6 were implemented strictly from
//! `docs/audio/shorten/spec/00-scope.md` through `spec/05-…md`. The
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
mod bitwriter;
mod block;
#[cfg(feature = "registry")]
mod codec;
mod driver;
mod encode_driver;
mod encoder;
#[cfg(feature = "registry")]
mod encoder_trait;
mod error;
mod header;
mod predictor;
mod sequencer;
mod sidecar;
mod stream_iter;

pub use crate::bitreader::{BitReader, BytePadding, ULONGSIZE};
pub use crate::bitwriter::{natural_ulong_width, BitWriter};
pub use crate::block::{
    read_bitshift_payload, read_blocksize_payload, read_function_code, read_verbatim_payload,
    FunctionCode, VerbatimChunk, BITSHIFTSIZE, BITSHIFT_MAX, BLOCKSIZE_MAX, FNSIZE,
    VERBATIM_BYTE_SIZE, VERBATIM_CHUNK_SIZE, VERBATIM_MAX_LEN,
};
#[cfg(feature = "registry")]
pub use crate::codec::{
    make_decoder, make_streaming_decoder, register_codecs, register_streaming_codecs,
    ShortenDecoder, ShortenStreamingDecoder, CODEC_ID_STR, FILETYPE_S16HL, FILETYPE_S16LH,
    FILETYPE_U8, STREAMING_CODEC_ID_STR,
};
pub use crate::driver::{decode_stream, DecodedStream, MAX_COMMANDS};
pub use crate::encode_driver::{encode_stream, encode_stream_lossy};
pub use crate::encoder::{
    encode_envelope_stream, min_energy_for_diff0, min_energy_for_diff1, min_energy_for_diff2,
    min_energy_for_diff3, min_energy_for_qlpc, optimal_energy_for_diff0, optimal_energy_for_diff1,
    optimal_energy_for_diff2, optimal_energy_for_diff3, optimal_energy_for_qlpc,
    optimal_energy_for_residuals, optimal_energy_for_residuals_wide, qlpc_residuals,
    residual_bits_at_energy, write_bitshift_command, write_blocksize_command,
    write_byte_aligned_prefix, write_diff0_block, write_diff1_block, write_diff2_block,
    write_diff3_block, write_parameter_block, write_qlpc_block, write_quit_command,
    write_stream_header, write_verbatim_block, write_zero_block, EncodeError, EncodeResult,
    ENCODER_VERSION, FN_BITSHIFT, FN_BLOCKSIZE, FN_DIFF0, FN_DIFF1, FN_DIFF2, FN_DIFF3, FN_QLPC,
    FN_QUIT, FN_VERBATIM, FN_ZERO, MAX_ENERGY, MAX_NATURAL_ENERGY, MAX_QLPC_ORDER,
};
#[cfg(feature = "registry")]
pub use crate::encoder_trait::{
    make_encoder, register_encoder, ShortenEncoder, DEFAULT_BLOCKSIZE, ENCODER_HEADER_VERSION,
};
pub use crate::error::{Error, Result};
pub use crate::header::{
    parse_stream_header, Filetype, ParsedHeader, ShortenStreamHeader, MAGIC, MIN_HEADER_BYTES,
};
pub use crate::predictor::{
    decode_diff_block, decode_qlpc_block, fill_zero_block, ChannelCarry, MeanEstimator, PolyOrder,
    CARRY_LEN_FLOOR, ENERGYSIZE, LPCQSIZE, LPCQUANT,
};
pub use crate::sequencer::{
    derive_qlpc_candidate, derive_qlpc_coefs, evaluate_candidates, evaluate_candidates_auto,
    evaluate_candidates_with_qlpc, select_predictor, select_predictor_auto,
    select_predictor_with_qlpc, write_selected_block, Choice, MAX_QLPC_AUTO_ORDER,
};
pub use crate::sidecar::{
    detect_shnampsk_trailer, split_off_shnampsk_trailer, ShnampskTrailer, MIN_SIDECAR_LEN,
    SEEK_MAGIC, SEEK_VERSION_OFFSET, SHNAMPSK_SIGNATURE, SIDECAR_LEN_CAP, TRAILER_TAIL_LEN,
};
pub use crate::stream_iter::{decode_stream_iter, DecodedBlock, StreamDecoder};

/// Install the Shorten decoder factories into the runtime context's
/// codec registry. Round 8 (paired with the round-7 whole-stream
/// driver [`decode_stream`]) wires [`ShortenDecoder`] in under codec
/// id `"shorten"`; round 11 additionally wires the block-by-block
/// streaming adaptor [`ShortenStreamingDecoder`] in under codec id
/// `"shorten-streaming"` so a caller can opt in to the bounded-memory
/// emission shape explicitly. The encoder side stays unimplemented
/// per the README "lacks" tail.
#[cfg(feature = "registry")]
pub fn register(ctx: &mut oxideav_core::RuntimeContext) {
    codec::register_codecs(&mut ctx.codecs);
    codec::register_streaming_codecs(&mut ctx.codecs);
    encoder_trait::register_encoder(&mut ctx.codecs);
}

#[cfg(feature = "registry")]
oxideav_core::register!("shorten", register);
