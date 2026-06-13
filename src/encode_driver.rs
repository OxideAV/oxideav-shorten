//! Shorten whole-stream encode driver — the encoder mirror of
//! [`crate::decode_stream`].
//!
//! Rounds 12..=282 landed every per-block writer primitive
//! ([`write_diff0_block`]..[`write_diff3_block`], [`write_qlpc_block`],
//! [`write_zero_block`], the [`write_blocksize_command`] /
//! [`write_bitshift_command`] / [`write_verbatim_block`] housekeeping
//! writers) plus the per-block predictor-selection sequencer
//! ([`select_predictor_auto`] / [`write_selected_block`]). What was
//! missing — the top item on the crate README's "What's not yet here"
//! list — is the orchestration layer that composes those pieces into a
//! single end-to-end encode call: deinterleave the channels, walk the
//! round-robin channel cursor, carry the per-channel sample-history and
//! running-mean state across blocks, handle the trailing partial block
//! via a [`BLOCK_FN_BLOCKSIZE`](crate::FN_BLOCKSIZE) override, and emit
//! the envelope + `BLOCK_FN_QUIT` + byte-alignment padding.
//!
//! [`encode_stream`] is the inverse of [`crate::decode_stream`]: for any
//! interleaved `&[i32]` sample buffer and a [`ShortenStreamHeader`]
//! describing the channelisation, it produces a syntactically-valid
//! `.shn` byte stream that [`crate::decode_stream`] reconstructs
//! **sample-exact** (lossless, `bshift = 0`). The two share an identical
//! per-channel state model — this is the load-bearing correctness
//! property, verified by the round-trip tests.
//!
//! ## Channelisation (`spec/03` §2 / TR.156 §3.1)
//!
//! The input buffer is interleaved `a(0), b(0), a(1), b(1), …` per
//! TR.156 §3.1's `-c channels` convention. The driver deinterleaves it
//! into `H_channels` per-channel sample vectors of equal length (a
//! ragged buffer — one whose length is not a multiple of `H_channels` —
//! is rejected). Each channel is then partitioned into blocks of
//! `H_blocksize` samples, and the blocks are emitted in the round-robin
//! cursor order [`crate::decode_stream`] consumes: block 0 of channel 0,
//! block 0 of channel 1, …, block 0 of channel `n−1`, block 1 of
//! channel 0, … — exactly the order a `(cursor + 1) % n_channels`
//! advance produces.
//!
//! ## Tail block (`spec/03` §3.6 / `spec/04` §4.1)
//!
//! When the per-channel sample count is not an exact multiple of
//! `H_blocksize`, the trailing samples form a partial block of
//! `new_bs < H_blocksize` samples. The driver emits a single
//! [`BLOCK_FN_BLOCKSIZE`](crate::FN_BLOCKSIZE) override carrying
//! `new_bs` immediately before the first partial block of the tail
//! round, mirroring fixture `F2`'s lone tail-block override at command
//! 11,377 (`new_bs = 155`, `spec/04` §4.1 test `T12`). The override is
//! a per-stream state mutator that does not advance the channel cursor,
//! so the partial blocks for every channel follow it within the same
//! round.
//!
//! ## Per-block state update order (`spec/05` §1.4 + §2)
//!
//! Each emitted block refreshes the channel's [`ChannelCarry`]
//! (`spec/05` §1) and [`MeanEstimator`] (`spec/05` §2) from the
//! **pre-shift** sample block — exactly the order
//! [`crate::decode_stream`]'s `commit_block` applies after decoding —
//! before the cursor advances. Because this driver encodes losslessly
//! (`bshift = 0`), the pre-shift and emitted forms coincide; the state
//! update consumes the samples directly.
//!
//! ## Predictor selection
//!
//! Each block's predictor is chosen by [`select_predictor_auto`], which
//! scores `BLOCK_FN_ZERO`, `BLOCK_FN_DIFF0..3`, and an auto-derived
//! `BLOCK_FN_QLPC` candidate (up to `H_maxlpcorder`) at the Rice-`n`
//! statistical optimum and returns the cheapest. A header with
//! `H_maxlpcorder = 0` reduces the candidate set to the
//! polynomial-difference family plus ZERO (`spec/03` §3.5).
//!
//! ## Natural-energy range (current limitation)
//!
//! [`select_predictor_auto`] scores candidates only at the
//! *natural* Rice energies `e ∈ 0..=MAX_NATURAL_ENERGY` (`spec/05`
//! §3.1), i.e. residual mantissa widths `1..=8`. A block whose
//! best-predictor residual stream overflows that band — for instance a
//! cold-carry DIFF0 first block of full-scale samples — has no scored
//! candidate and surfaces [`EncodeError::NoPredictorFits`]. Real PCM
//! after a predictor's de-correlation almost always lands inside the
//! band; widening the selector's energy sweep up to the decoder's
//! `MAX_RESIDUAL_WIDTH = 30` cap is a sequencer-layer refinement that
//! does not change the wire format (the per-predictor writers already
//! accept any energy in range).
//!
//! ## Clean-room provenance
//!
//! The orchestration sequence is assembled strictly from
//! `docs/audio/shorten/spec/03-block-and-predictor.md` §2 (channel
//! cursor / interleaving) + §3.6 (sub-block-size override) + §3.8 (QUIT
//! termination) + §3.10 (verbatim prefix),
//! `spec/04-function-code-resolution.md` §4.1 (`F2` tail-block override
//! `T12`), `spec/05-state-and-quirks.md` §1 (carry update) + §2 (mean
//! estimator) + §4 (post-QUIT byte alignment), and
//! `spec/01-stream-header.md` (header + carry-length derivation). It
//! reuses the same per-block writers and selector the earlier rounds
//! pinned against the spec; this module adds no new wire-format
//! decisions of its own.

use crate::bitwriter::BitWriter;
use crate::encoder::{
    write_blocksize_command, write_byte_aligned_prefix, write_parameter_block, write_quit_command,
    write_verbatim_block, EncodeError, EncodeResult,
};
use crate::header::ShortenStreamHeader;
use crate::predictor::{ChannelCarry, MeanEstimator};
use crate::sequencer::{select_predictor_auto, write_selected_block};

/// Encode an interleaved `i32` PCM buffer into a complete v2/v3 Shorten
/// byte stream that [`crate::decode_stream`] reconstructs sample-exact.
///
/// `header` supplies the channelisation parameters the encoder writes
/// into the file header and reads back as its own state model:
/// `H_channels`, `H_blocksize`, `H_maxlpcorder` (the upper bound on the
/// per-block QLPC order search), `H_meanblocks` (the running-mean
/// window), `H_filetype`, `H_skipbytes`, and the format `version`.
/// `samples` is the interleaved sample buffer
/// (`a(0), b(0), a(1), b(1), …` per `spec/03` §2). `verbatim_prefix`, if
/// non-empty, is emitted as a single [`BLOCK_FN_VERBATIM`](crate::FN_VERBATIM)
/// command ahead of the first predictor block (`spec/03` §3.10) — the
/// host-format envelope (RIFF/WAVE, AIFF, AU) the decoder surfaces in
/// [`DecodedStream::verbatim`](crate::DecodedStream).
///
/// The output is lossless: [`crate::decode_stream`] applied to the
/// returned bytes yields a [`DecodedStream`](crate::DecodedStream)
/// whose `header` equals `header`, whose `verbatim` equals
/// `verbatim_prefix`, and whose per-channel sample vectors equal the
/// deinterleaved `samples`.
///
/// Errors:
///
/// * [`EncodeError::ZeroChannels`] if `header.channels == 0`;
/// * [`EncodeError::RaggedInterleave`] if `samples.len()` is not a
///   multiple of `header.channels`;
/// * [`EncodeError::UnsupportedVersion`] if `header.version` is outside
///   `{1, 2, 3}` (surfaced by [`write_byte_aligned_prefix`]);
/// * [`EncodeError::VerbatimTooLong`] if `verbatim_prefix` exceeds the
///   length-field cap (surfaced by [`write_verbatim_block`]);
/// * [`EncodeError::NoPredictorFits`] if a block's residuals overflow
///   every natural-energy width while `BLOCK_FN_ZERO` is ineligible
///   (unreachable for in-range PCM);
/// * any error the per-block writers or the BLOCKSIZE-override writer
///   surface.
pub fn encode_stream(
    header: &ShortenStreamHeader,
    samples: &[i32],
    verbatim_prefix: &[u8],
) -> EncodeResult<Vec<u8>> {
    if header.channels == 0 {
        return Err(EncodeError::ZeroChannels);
    }
    let n_channels = header.channels as usize;
    if samples.len() % n_channels != 0 {
        return Err(EncodeError::RaggedInterleave {
            samples: samples.len(),
            channels: header.channels,
        });
    }
    let per_channel = samples.len() / n_channels;

    // Deinterleave a(0), b(0), a(1), b(1), … into per-channel vectors
    // (spec/03 §2 / TR.156 §3.1).
    let mut planes: Vec<Vec<i32>> = (0..n_channels)
        .map(|_| Vec::with_capacity(per_channel))
        .collect();
    for (i, &s) in samples.iter().enumerate() {
        planes[i % n_channels].push(s);
    }

    // Per-channel state, allocated exactly as decode_stream does.
    let carry_len = header.sample_history_carry_len() as usize;
    let mut carries: Vec<ChannelCarry> = (0..n_channels)
        .map(|_| ChannelCarry::new(carry_len))
        .collect();
    let mut means: Vec<MeanEstimator> = (0..n_channels)
        .map(|_| MeanEstimator::new(header.meanblocks))
        .collect();

    // Byte-aligned magic + version prefix, then the bit-level body.
    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version)?;

    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, header);
    if !verbatim_prefix.is_empty() {
        write_verbatim_block(&mut writer, verbatim_prefix)?;
    }

    // The default block size is H_blocksize. The reference encoder
    // partitions each channel into full blocks of this size, then a
    // single trailing partial block of `tail_len` samples preceded by
    // one BLOCK_FN_BLOCKSIZE override (spec/03 §3.6 / spec/04 §4.1).
    let block_size = header.blocksize.max(1) as usize;
    let full_blocks = per_channel / block_size;
    let tail_len = per_channel % block_size;

    let max_lpc_order = header.maxlpcorder;

    // Emit the full-size block rounds: for each block index, walk every
    // channel in cursor order. The cursor advances `(c + 1) % n` per
    // sample-producing command — which, since we iterate channels in
    // order within each round, is exactly the channel index.
    for b in 0..full_blocks {
        let start = b * block_size;
        for c in 0..n_channels {
            let block = &planes[c][start..start + block_size];
            emit_block(
                &mut writer,
                block,
                &mut carries[c],
                &mut means[c],
                max_lpc_order,
            )?;
        }
    }

    // Tail round: one partial block per channel of `tail_len` samples,
    // preceded by a single BLOCKSIZE override (the override is a
    // per-stream mutator that does not advance the channel cursor, so it
    // sits once at the head of the tail round and governs every
    // channel's partial block — spec/03 §3.6).
    if tail_len > 0 {
        write_blocksize_command(&mut writer, tail_len as u32)?;
        let start = full_blocks * block_size;
        for c in 0..n_channels {
            let block = &planes[c][start..start + tail_len];
            emit_block(
                &mut writer,
                block,
                &mut carries[c],
                &mut means[c],
                max_lpc_order,
            )?;
        }
    }

    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());
    Ok(out)
}

/// Select the cheapest predictor for one block, write the chosen
/// command, then refresh the channel's carry + mean estimator from the
/// (pre-shift) block — the encode-side mirror of `decode_stream`'s
/// `commit_block` ordering (`spec/05` §1.4 + §2).
fn emit_block(
    writer: &mut BitWriter,
    block: &[i32],
    carry: &mut ChannelCarry,
    mean: &mut MeanEstimator,
    max_lpc_order: u32,
) -> EncodeResult<()> {
    let mu_chan = mean.mu_chan();
    let choice = select_predictor_auto(block, mu_chan, carry, max_lpc_order)
        .ok_or(EncodeError::NoPredictorFits)?;
    write_selected_block(writer, &choice, block, mu_chan, carry)?;

    // State update consumes the pre-shift block (spec/05 §1.4); this
    // driver is lossless (bshift = 0) so the emitted form is identical.
    carry.update_after_block(block);
    mean.record_block(block);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::decode_stream;

    fn v2_header(
        channels: u32,
        blocksize: u32,
        maxlpcorder: u32,
        meanblocks: u32,
    ) -> ShortenStreamHeader {
        ShortenStreamHeader {
            version: 2,
            filetype: 5, // s16lh
            channels,
            blocksize,
            maxlpcorder,
            meanblocks,
            skipbytes: 0,
        }
    }

    /// Deinterleave an interleaved buffer the same way the driver does,
    /// for building per-channel expectations in the tests.
    fn deinterleave(samples: &[i32], n: usize) -> Vec<Vec<i32>> {
        let mut planes = vec![Vec::new(); n];
        for (i, &s) in samples.iter().enumerate() {
            planes[i % n].push(s);
        }
        planes
    }

    fn roundtrip(header: &ShortenStreamHeader, samples: &[i32], verbatim: &[u8]) {
        let bytes = encode_stream(header, samples, verbatim).expect("encode succeeds");
        let dec = decode_stream(&bytes).expect("decode succeeds");
        assert_eq!(dec.header, *header, "header round-trips");
        assert_eq!(dec.verbatim, verbatim, "verbatim prefix round-trips");
        let expected = deinterleave(samples, header.channels as usize);
        assert_eq!(dec.channels, expected, "per-channel samples round-trip");
    }

    #[test]
    fn mono_exact_multiple_of_blocksize_roundtrips() {
        let header = v2_header(1, 4, 0, 0);
        let samples: Vec<i32> = (0..16).collect();
        roundtrip(&header, &samples, &[]);
    }

    #[test]
    fn mono_with_tail_block_roundtrips() {
        // 14 samples, bs = 4 -> three full blocks + a 2-sample tail.
        let header = v2_header(1, 4, 0, 0);
        let samples: Vec<i32> = (0..14).map(|x| x * 3 - 5).collect();
        roundtrip(&header, &samples, &[]);
    }

    #[test]
    fn stereo_round_robin_roundtrips() {
        // Interleaved a,b,a,b,…; 2 channels × 10 samples each, bs = 4
        // -> 2 full block rounds + a 2-sample tail round.
        let header = v2_header(2, 4, 0, 0);
        let mut samples = Vec::new();
        for t in 0..10i32 {
            samples.push(100 + t); // channel a
            samples.push(-200 - t * 2); // channel b
        }
        roundtrip(&header, &samples, &[]);
    }

    #[test]
    fn three_channels_with_tail_roundtrips() {
        let header = v2_header(3, 5, 0, 0);
        let mut samples = Vec::new();
        for t in 0..13i32 {
            samples.push(t);
            samples.push(t * t - 7);
            samples.push(-t);
        }
        roundtrip(&header, &samples, &[]);
    }

    #[test]
    fn verbatim_prefix_roundtrips() {
        let header = v2_header(1, 8, 0, 0);
        let samples: Vec<i32> = (0..24).map(|x| (x % 7) * 11).collect();
        let verbatim = b"RIFF\x24\x00\x00\x00WAVEfmt ";
        roundtrip(&header, &samples, verbatim);
    }

    #[test]
    fn constant_signal_selects_zero_or_diff_and_roundtrips() {
        // A constant signal with a running-mean window: blocks after the
        // window warms up are eligible for BLOCK_FN_ZERO. The round-trip
        // must hold regardless of which predictor the selector picks.
        let header = v2_header(1, 4, 0, 4);
        let samples: Vec<i32> = vec![5; 40];
        roundtrip(&header, &samples, &[]);
    }

    #[test]
    fn lpc_material_with_maxlpcorder_roundtrips() {
        // A bounded integer recurrence s(t) = s(t-1) - s(t-2) is exactly
        // modelled by a derived QLPC vector [1, -1]; with maxlpcorder = 2
        // the auto-selector may pick QLPC. Round-trip must hold.
        let header = v2_header(1, 8, 2, 0);
        let mut s = vec![3i32, 7];
        for t in 2..64 {
            let v = s[t - 1] - s[t - 2];
            s.push(v);
        }
        roundtrip(&header, &s, &[]);
    }

    #[test]
    fn meanblocks_affects_diff0_and_zero_but_roundtrips() {
        let header = v2_header(2, 4, 0, 3);
        let mut samples = Vec::new();
        for t in 0..24i32 {
            samples.push(50 + (t % 5));
            samples.push(-30 + (t % 3));
        }
        roundtrip(&header, &samples, &[]);
    }

    #[test]
    fn empty_sample_buffer_roundtrips_to_empty_channels() {
        let header = v2_header(2, 256, 0, 0);
        let bytes = encode_stream(&header, &[], &[]).expect("encode succeeds");
        let dec = decode_stream(&bytes).expect("decode succeeds");
        assert_eq!(dec.channels.len(), 2);
        assert!(dec.channels[0].is_empty());
        assert!(dec.channels[1].is_empty());
        assert!(dec.verbatim.is_empty());
    }

    #[test]
    fn ragged_interleave_is_rejected() {
        let header = v2_header(2, 4, 0, 0);
        // 5 samples across 2 channels is ragged.
        let err = encode_stream(&header, &[1, 2, 3, 4, 5], &[]).unwrap_err();
        assert!(matches!(
            err,
            EncodeError::RaggedInterleave {
                samples: 5,
                channels: 2
            }
        ));
    }

    #[test]
    fn zero_channels_is_rejected() {
        let header = v2_header(0, 4, 0, 0);
        assert!(matches!(
            encode_stream(&header, &[], &[]).unwrap_err(),
            EncodeError::ZeroChannels
        ));
    }

    #[test]
    fn tail_only_no_full_blocks_roundtrips() {
        // Fewer samples than one full block: the whole channel is a
        // single tail block under a BLOCKSIZE override.
        let header = v2_header(1, 256, 0, 0);
        let samples: Vec<i32> = (0..10).map(|x| x * x).collect();
        roundtrip(&header, &samples, &[]);
    }

    #[test]
    fn smoothly_varying_signal_roundtrips() {
        // A smoothly varying two-channel signal whose first- and
        // second-differences stay inside the natural-energy band
        // (folded svar magnitude < 2^(MAX_NATURAL_ENERGY + 1)) that
        // `select_predictor_auto` scores. Exercises DIFFn selection plus
        // the running-mean window across many blocks.
        let header = v2_header(2, 16, 2, 4);
        let mut samples = Vec::new();
        let mut a = 0i32;
        let mut b = 100i32;
        for t in 0..100i32 {
            a += (t % 7) - 3; // slope in [-3, 3]
            b += 2 - (t % 5); // slope in [-2, 2]
            samples.push(a);
            samples.push(b);
        }
        roundtrip(&header, &samples, &[]);
    }

    #[test]
    fn out_of_natural_energy_range_surfaces_no_predictor_fits() {
        // A first block whose samples (= DIFF0 residuals over a cold
        // carry) exceed the natural-energy band the current selector
        // scores surfaces NoPredictorFits rather than emitting an
        // undecodable block. This pins the documented limitation of the
        // natural-energy selector (a wider-energy sweep is a future
        // sequencer-layer refinement, not a wire-format change). The
        // amplitude here (50_000) overflows every e ∈ 0..=7 width.
        let header = v2_header(1, 4, 0, 0);
        let samples: Vec<i32> = vec![50_000, -50_000, 40_000, -30_000];
        assert!(matches!(
            encode_stream(&header, &samples, &[]),
            Err(EncodeError::NoPredictorFits)
        ));
    }
}
