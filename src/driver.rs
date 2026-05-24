//! Shorten full-stream decode driver — the orchestration loop.
//!
//! Rounds 1..=6 landed the file-header parser plus a payload decoder
//! for every per-block command 0..=9. This module (round 7) ties them
//! together into [`decode_stream`], the end-to-end driver the
//! README's "What's not yet here" list and the CHANGELOG's "Next"
//! section have been pointing at:
//!
//! 1. Parse the file header ([`crate::parse_stream_header`]) and seek
//!    a fresh [`BitReader`] past the byte-aligned magic + version
//!    prefix and the variable-length parameter block to the first
//!    per-block command.
//! 2. Allocate the per-stream and per-channel decode state:
//!    * the round-robin **channel cursor** of `spec/03` §2, advanced
//!      modulo `H_channels` by every sample-producing command;
//!    * the running **sub-block size** (default `H_blocksize`,
//!      overridden by `BLOCK_FN_BLOCKSIZE` per `spec/03` §3.6);
//!    * the running **bit-shift** (zero by default, set by
//!      `BLOCK_FN_BITSHIFT` per `spec/03` §3.7);
//!    * the per-channel **sample-history carries**
//!      ([`crate::ChannelCarry`], `spec/05` §1);
//!    * the per-channel **running mean estimators**
//!      ([`crate::MeanEstimator`], `spec/05` §2).
//! 3. Loop: read one function code, dispatch on it, and update state.
//!    The predictor commands (`DIFF0..3`, `QLPC`) and `BLOCK_FN_ZERO`
//!    produce a block for the current channel, refresh that channel's
//!    carry + mean estimator, emit the (bit-shifted) samples, and
//!    advance the cursor; the housekeeping commands (`BLOCKSIZE`,
//!    `BITSHIFT`, `VERBATIM`) mutate per-stream state without
//!    advancing the cursor.
//! 4. Terminate on `BLOCK_FN_QUIT` (`spec/03` §3.8).
//!
//! ## Bit-shift application (`spec/05` §1.4)
//!
//! The per-channel carry stores samples in their **pre-shift** form —
//! the form the predictor reads and writes — so the predictor
//! recurrences see the same integer relationships across a
//! `BLOCK_FN_BITSHIFT` boundary. The driver applies the left-shift on
//! **emission** to the PCM sink only: `carry.update_after_block` and
//! `mean.record_block` consume the pre-shift block, while the samples
//! pushed into the per-channel output vectors are
//! `sample << bshift`.
//!
//! ## Verbatim prefix (`spec/03` §3.10)
//!
//! `BLOCK_FN_VERBATIM` byte payloads are collected into
//! [`DecodedStream::verbatim`] in encounter order. The bytes are the
//! host-format envelope (a RIFF/WAVE / AIFF / AU preamble) the encoder
//! preserved verbatim; the codec stores them opaquely.
//!
//! ## Clean-room provenance
//!
//! The orchestration sequence is assembled strictly from
//! `docs/audio/shorten/spec/03-block-and-predictor.md` §2 (channel
//! cursor) + §3.6/§3.7 (housekeeping state) + §3.8 (QUIT termination),
//! `spec/05-state-and-quirks.md` §1 (carry update + §1.4 bit-shift
//! externality) + §2 (mean estimator), and `spec/01-stream-header.md`
//! (header parse + carry-length derivation). No external decoder
//! source, no FFmpeg `shorten.c`, and no archived `old` branch were
//! consulted.

use crate::bitreader::BitReader;
use crate::block::{
    read_bitshift_payload, read_blocksize_payload, read_function_code, read_verbatim_payload,
    FunctionCode,
};
use crate::error::{Error, Result};
use crate::header::{parse_stream_header, ShortenStreamHeader};
use crate::predictor::{
    decode_diff_block, decode_qlpc_block, fill_zero_block, ChannelCarry, MeanEstimator, PolyOrder,
};

/// Implementation-side safety cap on the total number of per-block
/// commands a single stream may carry before the driver gives up. A
/// well-formed stream terminates with `BLOCK_FN_QUIT`; this cap stops a
/// pathological loop (e.g. a corrupt stream of nothing but housekeeping
/// commands that never advances toward `QUIT`) from running unbounded.
/// The bound is generous: fixture `F2`'s entire command stream is
/// 11,380 commands, so a cap many orders of magnitude above that
/// (~1 billion) cannot reject any realistic stream while still bounding
/// the worst case.
pub const MAX_COMMANDS: u64 = 1_000_000_000;

/// The fully-decoded output of [`decode_stream`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedStream {
    /// The parsed stream header.
    pub header: ShortenStreamHeader,
    /// The verbatim byte prefix(es) collected from `BLOCK_FN_VERBATIM`
    /// commands, in encounter order (`spec/03` §3.10). Typically a
    /// single host-format envelope header (RIFF/WAVE, AIFF, AU). Empty
    /// if the stream carried no verbatim command.
    pub verbatim: Vec<u8>,
    /// The decoded PCM samples, one `Vec<i32>` per channel, in channel
    /// order (`spec/03` §2). Each channel's vector holds that channel's
    /// samples in time order, already left-shifted by the running
    /// bit-shift in effect at emission time (`spec/05` §1.4). For a
    /// lossless stream (`bshift = 0` throughout) the samples are the
    /// reconstructed PCM directly.
    pub channels: Vec<Vec<i32>>,
}

impl DecodedStream {
    /// Total number of samples decoded for `channel` (panics in debug
    /// builds on an out-of-range channel index). Convenience accessor
    /// for callers that want a per-channel length without indexing
    /// [`Self::channels`] directly.
    pub fn channel_len(&self, channel: usize) -> usize {
        self.channels[channel].len()
    }
}

/// Decode a complete v2/v3 Shorten stream from `bytes`.
///
/// Runs the orchestration loop of this module: parse the header, seek
/// past it, then dispatch per-block commands until `BLOCK_FN_QUIT`,
/// carrying the round-robin channel cursor, running sub-block size,
/// running bit-shift, per-channel carries, and per-channel mean
/// estimators. Returns the [`DecodedStream`] holding the verbatim
/// prefix plus the per-channel sample vectors.
///
/// Errors:
///
/// * any error [`parse_stream_header`] surfaces (bad magic,
///   unsupported version, truncation);
/// * [`Error::Truncated`] if the block stream ends before a
///   `BLOCK_FN_QUIT` command (the per-block payload readers surface
///   this when the bit stream exhausts mid-command);
/// * any per-command payload-decode error (see the individual command
///   decoders for the variants each can produce);
/// * [`Error::BlockTooLarge`] / [`Error::SampleOverflow`] if a
///   bit-shift application would overflow the `i32` sample slot;
/// * [`Error::BlockCommandNotImplemented`] if the command-count safety
///   cap [`MAX_COMMANDS`] is reached without encountering
///   `BLOCK_FN_QUIT` (a malformed never-terminating stream).
pub fn decode_stream(bytes: &[u8]) -> Result<DecodedStream> {
    let parsed = parse_stream_header(bytes)?;
    let header = parsed.header;

    // The number of channels must be at least one for the round-robin
    // cursor to be well-defined; a zero-channel header is malformed.
    if header.channels == 0 {
        return Err(Error::Truncated);
    }
    let n_channels = header.channels as usize;

    // Re-open a fresh reader over the bytes after the byte-aligned
    // magic + version prefix, then seek past the variable-length
    // parameter block to the first per-block command. The header parser
    // read those same bits from `bytes[5..]` and reported their total
    // width as `bits_consumed_after_v`.
    let post_version = &bytes[5..];
    let mut reader = BitReader::new(post_version);
    reader.skip_bits(parsed.bits_consumed_after_v)?;

    // Per-stream state.
    let carry_len = header.sample_history_carry_len() as usize;
    let mut carries: Vec<ChannelCarry> = (0..n_channels)
        .map(|_| ChannelCarry::new(carry_len))
        .collect();
    let mut means: Vec<MeanEstimator> = (0..n_channels)
        .map(|_| MeanEstimator::new(header.meanblocks))
        .collect();
    let mut channels: Vec<Vec<i32>> = (0..n_channels).map(|_| Vec::new()).collect();
    let mut verbatim: Vec<u8> = Vec::new();

    let mut cursor: usize = 0;
    // Running sub-block size: default H_blocksize, overridden by
    // BLOCK_FN_BLOCKSIZE (spec/03 §3.6).
    let mut block_size = header.blocksize;
    // Running per-stream bit-shift: zero default, set by
    // BLOCK_FN_BITSHIFT (spec/03 §3.7 / spec/05 §1.4).
    let mut bshift: u32 = 0;

    let mut commands: u64 = 0;
    loop {
        if commands >= MAX_COMMANDS {
            // A well-formed stream always reaches BLOCK_FN_QUIT well
            // before this cap; hitting it means the stream never
            // terminates (corruption).
            return Err(Error::BlockCommandNotImplemented(
                FunctionCode::Quit.wire_value(),
            ));
        }
        commands += 1;

        let fc = read_function_code(&mut reader)?;
        match fc {
            FunctionCode::Quit => break,

            // --- Housekeeping commands: mutate state, no samples,
            //     cursor unchanged. ---
            FunctionCode::Blocksize => {
                block_size = read_blocksize_payload(&mut reader)?;
            }
            FunctionCode::Bitshift => {
                bshift = read_bitshift_payload(&mut reader)?;
            }
            FunctionCode::Verbatim => {
                let chunk = read_verbatim_payload(&mut reader)?;
                verbatim.extend_from_slice(&chunk.bytes);
            }

            // --- Sample-producing commands: decode a block for the
            //     current channel, refresh state, emit, advance. ---
            FunctionCode::Diff0
            | FunctionCode::Diff1
            | FunctionCode::Diff2
            | FunctionCode::Diff3 => {
                let order =
                    PolyOrder::from_function_code(fc).expect("DIFF0..3 always map to a PolyOrder");
                let mu_chan = means[cursor].mu_chan();
                let block =
                    decode_diff_block(&mut reader, order, block_size, &carries[cursor], mu_chan)?;
                commit_block(
                    &block,
                    bshift,
                    &mut carries[cursor],
                    &mut means[cursor],
                    &mut channels[cursor],
                )?;
                cursor = (cursor + 1) % n_channels;
            }
            FunctionCode::Qlpc => {
                let block = decode_qlpc_block(&mut reader, block_size, &carries[cursor])?;
                commit_block(
                    &block,
                    bshift,
                    &mut carries[cursor],
                    &mut means[cursor],
                    &mut channels[cursor],
                )?;
                cursor = (cursor + 1) % n_channels;
            }
            FunctionCode::Zero => {
                let mu_chan = means[cursor].mu_chan();
                let block = fill_zero_block(block_size, mu_chan)?;
                commit_block(
                    &block,
                    bshift,
                    &mut carries[cursor],
                    &mut means[cursor],
                    &mut channels[cursor],
                )?;
                cursor = (cursor + 1) % n_channels;
            }
        }
    }

    Ok(DecodedStream {
        header,
        verbatim,
        channels,
    })
}

/// Refresh a channel's carry + mean estimator from the just-decoded
/// (pre-shift) `block`, then push the bit-shifted samples into the
/// channel's output vector.
///
/// Per `spec/05` §1.4 the carry and mean estimator consume the
/// pre-shift block (so the predictor recurrences are bit-shift
/// invariant), while the emitted samples are `sample << bshift`. The
/// shift is applied in `i64` headroom and narrowed back to `i32` so a
/// pathological `sample << bshift` that overflows the slot surfaces
/// [`Error::SampleOverflow`] rather than silently wrapping.
fn commit_block(
    block: &[i32],
    bshift: u32,
    carry: &mut ChannelCarry,
    mean: &mut MeanEstimator,
    out: &mut Vec<i32>,
) -> Result<()> {
    // State update consumes the pre-shift block (spec/05 §1.4).
    carry.update_after_block(block);
    mean.record_block(block);

    out.reserve(block.len());
    if bshift == 0 {
        out.extend_from_slice(block);
    } else {
        for &s in block {
            let shifted = (s as i64)
                .checked_shl(bshift)
                .ok_or(Error::SampleOverflow)?;
            let s_i32: i32 = shifted.try_into().map_err(|_| Error::SampleOverflow)?;
            out.push(s_i32);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::FNSIZE;
    use crate::predictor::ENERGYSIZE;

    // ---- synthetic-stream builders (mirror the integration tests) ----

    fn pack_bits_msb_first(bits: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut byte = 0u8;
        let mut n = 0u32;
        for &b in bits {
            byte = (byte << 1) | (b as u8 & 1);
            n += 1;
            if n == 8 {
                out.push(byte);
                byte = 0;
                n = 0;
            }
        }
        if n > 0 {
            out.push(byte << (8 - n));
        }
        out
    }

    fn encode_uvar(value: u32, n: u32) -> Vec<u32> {
        if n == 0 {
            let mut bits = vec![0u32; value as usize];
            bits.push(1);
            bits
        } else {
            let span = 1u32 << n;
            let prefix_zeros = value / span;
            let mantissa = value % span;
            let mut bits = vec![0u32; prefix_zeros as usize];
            bits.push(1);
            for i in (0..n).rev() {
                bits.push((mantissa >> i) & 1);
            }
            bits
        }
    }

    fn encode_svar(value: i64, n: u32) -> Vec<u32> {
        let u: u64 = if value >= 0 {
            (value as u64) << 1
        } else {
            (((!value) as u64) << 1) | 1
        };
        let u32_val = u32::try_from(u).expect("svar fits in u32 in this test");
        encode_uvar(u32_val, n)
    }

    fn encode_ulong(value: u32, w: u32) -> Vec<u32> {
        let mut bits = Vec::new();
        bits.extend(encode_uvar(w, 2));
        bits.extend(encode_uvar(value, w));
        bits
    }

    /// Six-field v2 header parameter block (mirrors F1's choices but
    /// caller-chosen for the test).
    fn header_param_bits(
        filetype: u32,
        channels: u32,
        blocksize: u32,
        maxlpcorder: u32,
        meanblocks: u32,
        skipbytes: u32,
    ) -> Vec<u32> {
        let bits_for = |v: u32| -> u32 {
            // smallest power-of-two width good enough for `ulong()`.
            if v == 0 {
                0
            } else {
                32 - v.leading_zeros()
            }
        };
        let mut bits = Vec::new();
        bits.extend(encode_ulong(filetype, bits_for(filetype)));
        bits.extend(encode_ulong(channels, bits_for(channels)));
        bits.extend(encode_ulong(blocksize, bits_for(blocksize)));
        bits.extend(encode_ulong(maxlpcorder, bits_for(maxlpcorder)));
        bits.extend(encode_ulong(meanblocks, bits_for(meanblocks)));
        bits.extend(encode_ulong(skipbytes, bits_for(skipbytes)));
        bits
    }

    fn assemble(all_bits: &[u32]) -> Vec<u8> {
        let body = pack_bits_msb_first(all_bits);
        let mut buf = Vec::with_capacity(5 + body.len());
        buf.extend_from_slice(&crate::header::MAGIC);
        buf.push(2);
        buf.extend_from_slice(&body);
        buf
    }

    fn append_diff_block(out: &mut Vec<u32>, code: u32, energy_encoded: u32, residuals: &[i64]) {
        out.extend(encode_uvar(code, FNSIZE));
        out.extend(encode_uvar(energy_encoded, ENERGYSIZE));
        let width = energy_encoded + 1;
        for &r in residuals {
            out.extend(encode_svar(r, width));
        }
    }

    #[test]
    fn single_channel_diff1_two_blocks_then_quit() {
        // H_channels = 1, H_blocksize = 4, no mean (meanblocks = 0),
        // no LPC. Two DIFF1 blocks for the single channel:
        //   block A: residuals [10, 5, -3, 2] over zero carry
        //     -> [10, 15, 12, 14]; carry[0] = 14.
        //   block B: residuals [7, -2, -5, 1] over carry[0] = 14
        //     -> [21, 19, 14, 15].
        let mut bits = header_param_bits(5, 1, 4, 0, 0, 0);
        append_diff_block(&mut bits, 1, 3, &[10, 5, -3, 2]);
        append_diff_block(&mut bits, 1, 3, &[7, -2, -5, 1]);
        bits.extend(encode_uvar(4, FNSIZE)); // QUIT
        let buf = assemble(&bits);

        let dec = decode_stream(&buf).expect("stream decodes");
        assert_eq!(dec.header.channels, 1);
        assert_eq!(dec.channels.len(), 1);
        assert_eq!(dec.channels[0], vec![10, 15, 12, 14, 21, 19, 14, 15]);
        assert!(dec.verbatim.is_empty());
    }

    #[test]
    fn verbatim_prefix_is_collected_in_order() {
        // VERBATIM (3 bytes) then DIFF0 (1 channel, bs = 2) then QUIT.
        let mut bits = header_param_bits(5, 1, 2, 0, 0, 0);
        // VERBATIM = 9: length uvar(5) then bytes uvar(8).
        bits.extend(encode_uvar(9, FNSIZE));
        bits.extend(encode_uvar(3, 5));
        for b in [0x52u8, 0x49, 0x46] {
            bits.extend(encode_uvar(b as u32, 8));
        }
        // DIFF0 (code 0): residuals are the samples (mean = 0).
        append_diff_block(&mut bits, 0, 3, &[100, -50]);
        bits.extend(encode_uvar(4, FNSIZE)); // QUIT
        let buf = assemble(&bits);

        let dec = decode_stream(&buf).expect("stream decodes");
        assert_eq!(dec.verbatim, vec![0x52, 0x49, 0x46]);
        assert_eq!(dec.channels[0], vec![100, -50]);
    }

    #[test]
    fn round_robin_cursor_interleaves_two_channels() {
        // H_channels = 2, bs = 2. Sequence:
        //   DIFF0 ch0 [1, 2], DIFF0 ch1 [10, 20],
        //   DIFF0 ch0 [3, 4], DIFF0 ch1 [30, 40], QUIT.
        let mut bits = header_param_bits(5, 2, 2, 0, 0, 0);
        append_diff_block(&mut bits, 0, 3, &[1, 2]);
        append_diff_block(&mut bits, 0, 3, &[10, 20]);
        append_diff_block(&mut bits, 0, 3, &[3, 4]);
        append_diff_block(&mut bits, 0, 3, &[30, 40]);
        bits.extend(encode_uvar(4, FNSIZE)); // QUIT
        let buf = assemble(&bits);

        let dec = decode_stream(&buf).expect("stream decodes");
        assert_eq!(dec.channels.len(), 2);
        assert_eq!(dec.channels[0], vec![1, 2, 3, 4]);
        assert_eq!(dec.channels[1], vec![10, 20, 30, 40]);
    }

    #[test]
    fn blocksize_override_changes_subsequent_block_lengths() {
        // Default bs = 4, then BLOCKSIZE -> 2 before the second block.
        //   DIFF0 ch0 [1, 2, 3, 4]   (default bs = 4)
        //   BLOCKSIZE new_bs = 2     (housekeeping; cursor unchanged)
        //   DIFF0 ch0 [5, 6]         (new bs = 2)
        //   QUIT
        let mut bits = header_param_bits(5, 1, 4, 0, 0, 0);
        append_diff_block(&mut bits, 0, 3, &[1, 2, 3, 4]);
        bits.extend(encode_uvar(5, FNSIZE)); // BLOCKSIZE = 5
        bits.extend(encode_ulong(2, 2));
        append_diff_block(&mut bits, 0, 3, &[5, 6]);
        bits.extend(encode_uvar(4, FNSIZE)); // QUIT
        let buf = assemble(&bits);

        let dec = decode_stream(&buf).expect("stream decodes");
        assert_eq!(dec.channels[0], vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn bitshift_left_shifts_emitted_samples_but_not_the_carry() {
        // BITSHIFT bshift = 3 then DIFF1 (bs = 4).
        //   residuals [1, 1, 1, 1] over zero carry, DIFF1
        //     -> pre-shift samples [1, 2, 3, 4]
        //     -> emitted (<<3) = [8, 16, 24, 32].
        //   A second DIFF1 block must read the *pre-shift* carry
        //     (carry[0] = 4, not 32): residuals [0, 0]
        //     -> pre-shift [4, 4] -> emitted [32, 32].
        let mut bits = header_param_bits(5, 1, 4, 0, 0, 0);
        bits.extend(encode_uvar(6, FNSIZE)); // BITSHIFT = 6
        bits.extend(encode_uvar(3, 2)); // bshift = 3 (uvar(BITSHIFTSIZE=2))
        append_diff_block(&mut bits, 1, 3, &[1, 1, 1, 1]);
        // second block: override bs to 2 first.
        bits.extend(encode_uvar(5, FNSIZE)); // BLOCKSIZE = 5
        bits.extend(encode_ulong(2, 2));
        append_diff_block(&mut bits, 1, 3, &[0, 0]);
        bits.extend(encode_uvar(4, FNSIZE)); // QUIT
        let buf = assemble(&bits);

        let dec = decode_stream(&buf).expect("stream decodes");
        // First block emitted at <<3; second block reads pre-shift carry.
        assert_eq!(dec.channels[0], vec![8, 16, 24, 32, 32, 32]);
    }

    #[test]
    fn zero_command_emits_running_mean_block() {
        // meanblocks = 4 (a 4-slot sliding window zero-initialised per
        // spec/05 §2.1). The ZERO block emits the running mean computed
        // at the *start* of the block.
        //   block A DIFF0 residuals [6, 6, 6, 6] (mean = 0 at start)
        //     -> [6, 6, 6, 6]; per-block mean mu_blk = (24 + 2)/4 = 6
        //        slides into one slot; the window now holds {6, 0, 0, 0}.
        //   block B ZERO: running mu_chan = trunc_div(6 + 4/2, 4)
        //     = trunc_div(8, 4) = 2, so the ZERO block emits four 2s.
        //   QUIT.
        // (The window only reaches a steady 6 after four DIFF0 blocks of
        //  6s; with a single prior block the three zero slots pull the
        //  average down — this is exactly the sliding-window arithmetic
        //  of spec/05 §2.5 pinned by MeanEstimator.)
        let mut bits = header_param_bits(5, 1, 4, 0, 4, 0);
        append_diff_block(&mut bits, 0, 3, &[6, 6, 6, 6]);
        bits.extend(encode_uvar(8, FNSIZE)); // ZERO = 8 (no payload)
        bits.extend(encode_uvar(4, FNSIZE)); // QUIT
        let buf = assemble(&bits);

        let dec = decode_stream(&buf).expect("stream decodes");
        assert_eq!(dec.channels[0], vec![6, 6, 6, 6, 2, 2, 2, 2]);
    }

    #[test]
    fn truncated_block_stream_without_quit_is_truncated() {
        // A DIFF0 block whose residuals run out mid-decode (no QUIT,
        // stream just ends).
        let mut bits = header_param_bits(5, 1, 4, 0, 0, 0);
        // DIFF0 claims bs = 4 residuals but supply only one.
        bits.extend(encode_uvar(0, FNSIZE));
        bits.extend(encode_uvar(3, ENERGYSIZE));
        bits.extend(encode_svar(1, 4));
        let buf = assemble(&bits);

        assert_eq!(decode_stream(&buf), Err(Error::Truncated));
    }

    #[test]
    fn zero_channel_header_is_rejected() {
        // H_channels = 0 is malformed; the round-robin cursor would be
        // undefined.
        let bits = header_param_bits(5, 0, 4, 0, 0, 0);
        let buf = assemble(&bits);
        assert_eq!(decode_stream(&buf), Err(Error::Truncated));
    }

    #[test]
    fn empty_stream_just_quit_yields_empty_channels() {
        // A header followed immediately by QUIT: zero samples decoded.
        let mut bits = header_param_bits(5, 2, 256, 0, 0, 0);
        bits.extend(encode_uvar(4, FNSIZE)); // QUIT
        let buf = assemble(&bits);

        let dec = decode_stream(&buf).expect("stream decodes");
        assert_eq!(dec.channels.len(), 2);
        assert!(dec.channels[0].is_empty());
        assert!(dec.channels[1].is_empty());
        assert!(dec.verbatim.is_empty());
    }
}
