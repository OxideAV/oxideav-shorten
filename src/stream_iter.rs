//! Block-by-block streaming decode iterator — round 181.
//!
//! [`decode_stream`](crate::decode_stream) accumulates every decoded
//! sample into `DecodedStream::channels: Vec<Vec<i32>>`. For a 10.6 MB
//! lossless input (the size order of fixture `F1` per
//! `docs/audio/shorten/spec/05-state-and-quirks.md` §2.5) that buffers
//! the entire decoded sample population in memory ahead of the caller.
//!
//! This module exposes an alternative shape: a `Iterator<Item =
//! Result<DecodedBlock>>` that walks the same per-block command loop
//! but yields each sample-producing block to the caller one at a time
//! and discards it after the carry / mean state has been updated. The
//! per-channel sample-history carries (`spec/05` §1) and running mean
//! estimators (`spec/05` §2) still need to live for the full stream
//! (the next block on the same channel reads from them), so memory
//! reaches `O(n_channels × max(3, H_maxlpcorder + H_meanblocks))` plus
//! the BitReader's tiny internal cache — bounded by the header
//! parameters and independent of stream length.
//!
//! ## Spec anchors
//!
//! - `spec/03-block-and-predictor.md` §2 — round-robin per-channel
//!   command dispatch (sample-producing commands advance a single
//!   `cursor` modulo `H_channels`; housekeeping commands do not).
//! - `spec/03` §3.6 / §3.7 — `BLOCK_FN_BLOCKSIZE` / `BLOCK_FN_BITSHIFT`
//!   update the running sub-block size + running bit-shift in place.
//! - `spec/03` §3.8 — `BLOCK_FN_QUIT` terminates the command stream.
//! - `spec/03` §3.10 — `BLOCK_FN_VERBATIM` carries opaque host-format
//!   envelope bytes (RIFF/WAVE / AIFF / AU preamble). The iterator
//!   surfaces these via a dedicated [`DecodedBlock::Verbatim`] variant
//!   so callers that want to forward the envelope (e.g. to a WAV
//!   writer) do not have to re-walk the stream.
//! - `spec/05` §1.4 — the per-channel carry stores **pre-shift** samples
//!   so the predictor recurrences stay bit-shift invariant; the
//!   left-shift is applied on emission only. The iterator preserves
//!   this contract by applying the shift inside `next_block` after the
//!   carry has been refreshed from the pre-shift block.
//!
//! ## Iterator contract
//!
//! - First call to [`StreamDecoder::next_block`] (or
//!   [`Iterator::next`]) parses the header — every subsequent call
//!   reuses the cached header. The header is also surfaced via
//!   [`StreamDecoder::header`] before any block has been pulled.
//! - The iterator yields `Ok(DecodedBlock::Samples { .. })` for every
//!   `DIFFn` / `QLPC` / `ZERO` block, `Ok(DecodedBlock::Verbatim
//!   { .. })` for every `VERBATIM` block, and `None` after
//!   `BLOCK_FN_QUIT`. `BLOCK_FN_BLOCKSIZE` / `BLOCK_FN_BITSHIFT` are
//!   absorbed into the iterator's internal state and never surface
//!   as `DecodedBlock` items (they are housekeeping commands per
//!   `spec/03` §3.6/§3.7).
//! - Any per-block decode error short-circuits the iterator: the
//!   `Err` is yielded once, then subsequent `next` calls return
//!   `None` (the underlying bit reader is presumed unrecoverable).
//!   This matches the standard `Iterator<Item = Result<T, E>>`
//!   convention.
//!
//! ## Clean-room provenance
//!
//! This module is assembled from the per-block dispatch already in
//! [`crate::driver`] (round 7) — same channel-cursor / blocksize /
//! bitshift state, same carry + mean-estimator update sequencing,
//! same termination criterion. The re-shaping into an `Iterator`
//! surface is a pure API change: no new wire-format behaviour, no new
//! reconstruction recurrences. The contracts cited above are all
//! traced to `docs/audio/shorten/spec/03` and `spec/05`.

use crate::bitreader::BitReader;
use crate::block::{
    read_bitshift_payload, read_blocksize_payload, read_function_code, read_verbatim_payload,
    FunctionCode,
};
use crate::driver::MAX_COMMANDS;
use crate::error::{Error, Result};
use crate::header::{parse_stream_header, ShortenStreamHeader};
use crate::predictor::{
    decode_diff_block, decode_qlpc_block, fill_zero_block, ChannelCarry, MeanEstimator, PolyOrder,
};

/// One item yielded by [`StreamDecoder::next_block`].
///
/// The iterator surfaces sample-producing blocks (`DIFFn` / `QLPC` /
/// `ZERO` per `spec/03` §3.1..§3.5 + §3.9) and verbatim envelope
/// payloads (`spec/03` §3.10) directly. The housekeeping commands
/// `BLOCKSIZE` / `BITSHIFT` (`spec/03` §3.6 / §3.7) are absorbed into
/// the iterator's internal state and do NOT produce an item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodedBlock {
    /// A sample-producing block for one channel.
    ///
    /// Per `spec/03` §2 the per-channel cursor advances modulo
    /// `H_channels` after a sample-producing command, so successive
    /// `Samples` items typically rotate through the channels (e.g.
    /// for `H_channels = 2` the cursor is `0, 1, 0, 1, …`).
    Samples {
        /// Index of the channel this block belongs to
        /// (`0..H_channels`). Per `spec/03` §2.
        channel: usize,
        /// The decoded samples for the block, already left-shifted by
        /// the running bit-shift in effect at emission time
        /// (`spec/05` §1.4). Length equals the current running
        /// sub-block size (default `H_blocksize`, updated by
        /// `BLOCK_FN_BLOCKSIZE`).
        samples: Vec<i32>,
    },
    /// A verbatim envelope payload (`BLOCK_FN_VERBATIM`, `spec/03`
    /// §3.10).
    ///
    /// The bytes are the host-format envelope (a RIFF/WAVE, AIFF, or
    /// AU preamble) the encoder preserved verbatim. The iterator does
    /// not interpret these bytes; the caller may concatenate
    /// consecutive `Verbatim` items in encounter order to reconstruct
    /// the original envelope (the same shape that
    /// [`crate::DecodedStream::verbatim`] presents).
    Verbatim {
        /// Opaque envelope bytes.
        bytes: Vec<u8>,
    },
}

impl DecodedBlock {
    /// Number of PCM samples this block contributes to the channel's
    /// output. Returns `0` for [`DecodedBlock::Verbatim`] items.
    pub fn sample_count(&self) -> usize {
        match self {
            DecodedBlock::Samples { samples, .. } => samples.len(),
            DecodedBlock::Verbatim { .. } => 0,
        }
    }

    /// Returns `true` if this is a [`DecodedBlock::Samples`] item.
    pub fn is_samples(&self) -> bool {
        matches!(self, DecodedBlock::Samples { .. })
    }

    /// Returns `true` if this is a [`DecodedBlock::Verbatim`] item.
    pub fn is_verbatim(&self) -> bool {
        matches!(self, DecodedBlock::Verbatim { .. })
    }
}

/// Streaming Shorten decoder: an `Iterator` over [`DecodedBlock`]s.
///
/// Created with [`StreamDecoder::new`] which parses the header
/// eagerly. Subsequent calls to [`StreamDecoder::next_block`] (or
/// `Iterator::next`) walk the per-block command loop, dispatching
/// one block at a time. The iterator finishes when `BLOCK_FN_QUIT` is
/// reached (or when an unrecoverable decode error is yielded once).
///
/// ## Memory characteristics
///
/// Unlike [`crate::decode_stream`], this iterator does not retain any
/// emitted samples once they have been yielded. The only state held
/// across calls is the per-channel carry buffers (size
/// `n_channels × max(3, H_maxlpcorder)`), the per-channel mean
/// estimators (size `n_channels × H_meanblocks`), and the BitReader's
/// own small cache — none of which grow with stream length.
pub struct StreamDecoder<'a> {
    /// Parsed stream header (the same shape [`crate::decode_stream`]
    /// surfaces inside [`crate::DecodedStream::header`]).
    header: ShortenStreamHeader,
    /// MSB-first bit reader positioned at the first per-block command.
    reader: BitReader<'a>,
    /// Per-channel sample-history carries (`spec/05` §1).
    carries: Vec<ChannelCarry>,
    /// Per-channel running mean estimators (`spec/05` §2).
    means: Vec<MeanEstimator>,
    /// Round-robin channel cursor (`spec/03` §2).
    cursor: usize,
    /// Running sub-block size — defaults to `H_blocksize`; overridden
    /// by `BLOCK_FN_BLOCKSIZE` (`spec/03` §3.6).
    block_size: u32,
    /// Running per-stream bit-shift — zero default, set by
    /// `BLOCK_FN_BITSHIFT` (`spec/03` §3.7 / `spec/05` §1.4).
    bshift: u32,
    /// Count of commands processed so far. Bounded by [`MAX_COMMANDS`].
    commands: u64,
    /// True once `BLOCK_FN_QUIT` has been observed or an error has
    /// been yielded. Subsequent calls return `None`.
    finished: bool,
}

impl<'a> std::fmt::Debug for StreamDecoder<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamDecoder")
            .field("header", &self.header)
            .field("cursor", &self.cursor)
            .field("block_size", &self.block_size)
            .field("bshift", &self.bshift)
            .field("commands", &self.commands)
            .field("finished", &self.finished)
            .finish()
    }
}

impl<'a> StreamDecoder<'a> {
    /// Construct a new streaming decoder over `bytes` — parses the
    /// file header eagerly and positions the bit reader at the first
    /// per-block command.
    ///
    /// Surfaces any error [`parse_stream_header`] can produce
    /// (`InvalidMagic`, `UnsupportedVersion`, `Truncated`,
    /// `OverflowingUvar`) plus `Error::Truncated` when the header
    /// names a zero-channel stream (the round-robin cursor would be
    /// undefined per `spec/03` §2).
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let parsed = parse_stream_header(bytes)?;
        let header = parsed.header;

        if header.channels == 0 {
            return Err(Error::Truncated);
        }
        let n_channels = header.channels as usize;

        // Re-open a fresh reader past the magic + version prefix and
        // skip the variable-length parameter block to land on the
        // first per-block command — identical to the round-7 driver's
        // bit-position calculation.
        let post_version = &bytes[5..];
        let mut reader = BitReader::new(post_version);
        reader.skip_bits(parsed.bits_consumed_after_v)?;

        let carry_len = header.sample_history_carry_len() as usize;
        let carries: Vec<ChannelCarry> = (0..n_channels)
            .map(|_| ChannelCarry::new(carry_len))
            .collect();
        let means: Vec<MeanEstimator> = (0..n_channels)
            .map(|_| MeanEstimator::new(header.meanblocks))
            .collect();

        let block_size = header.blocksize;

        Ok(Self {
            header,
            reader,
            carries,
            means,
            cursor: 0,
            block_size,
            bshift: 0,
            commands: 0,
            finished: false,
        })
    }

    /// The parsed stream header. Available before any block has been
    /// pulled; the field values match
    /// [`crate::DecodedStream::header`] for the same input bytes.
    pub fn header(&self) -> &ShortenStreamHeader {
        &self.header
    }

    /// The running sub-block size that would apply to the next
    /// sample-producing command (`spec/03` §3.6). Defaults to
    /// `H_blocksize`; mutates as `BLOCK_FN_BLOCKSIZE` commands are
    /// absorbed.
    pub fn current_block_size(&self) -> u32 {
        self.block_size
    }

    /// The running per-stream bit-shift that would apply to the next
    /// sample-producing command (`spec/03` §3.7 / `spec/05` §1.4).
    /// Defaults to `0`; mutates as `BLOCK_FN_BITSHIFT` commands are
    /// absorbed. Emitted samples in
    /// [`DecodedBlock::Samples::samples`] are left-shifted by this
    /// amount.
    pub fn current_bitshift(&self) -> u32 {
        self.bshift
    }

    /// The channel index the next sample-producing command will
    /// target. Modulo `H_channels` per the round-robin cursor of
    /// `spec/03` §2.
    pub fn current_channel(&self) -> usize {
        self.cursor
    }

    /// True once the iterator has terminated — either because
    /// `BLOCK_FN_QUIT` was consumed or because an error was yielded.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Pull the next [`DecodedBlock`].
    ///
    /// Walks per-block commands, absorbing housekeeping commands
    /// (`BLOCKSIZE` / `BITSHIFT`) silently into internal state, until
    /// it either produces a [`DecodedBlock::Samples`] / `Verbatim`
    /// item, observes `BLOCK_FN_QUIT` (returns `Ok(None)`), or hits a
    /// decode error (returns `Err(e)` once, then subsequent calls
    /// return `Ok(None)`).
    ///
    /// The error variants this can surface are the same as
    /// [`crate::decode_stream`]'s per-block payload decoders surface.
    pub fn next_block(&mut self) -> Result<Option<DecodedBlock>> {
        if self.finished {
            return Ok(None);
        }
        loop {
            if self.commands >= MAX_COMMANDS {
                // Same cap as the round-7 whole-stream driver — a
                // well-formed stream reaches BLOCK_FN_QUIT well
                // before this limit; tripping it means the stream is
                // corrupt and never terminates.
                self.finished = true;
                return Err(Error::BlockCommandNotImplemented(
                    FunctionCode::Quit.wire_value(),
                ));
            }
            self.commands += 1;

            let fc = match read_function_code(&mut self.reader) {
                Ok(fc) => fc,
                Err(e) => {
                    self.finished = true;
                    return Err(e);
                }
            };
            match fc {
                FunctionCode::Quit => {
                    self.finished = true;
                    return Ok(None);
                }
                // --- Housekeeping: mutate state in place, loop. ---
                FunctionCode::Blocksize => {
                    let new_bs = match read_blocksize_payload(&mut self.reader) {
                        Ok(v) => v,
                        Err(e) => {
                            self.finished = true;
                            return Err(e);
                        }
                    };
                    self.block_size = new_bs;
                }
                FunctionCode::Bitshift => {
                    let bshift = match read_bitshift_payload(&mut self.reader) {
                        Ok(v) => v,
                        Err(e) => {
                            self.finished = true;
                            return Err(e);
                        }
                    };
                    self.bshift = bshift;
                }
                FunctionCode::Verbatim => {
                    let chunk = match read_verbatim_payload(&mut self.reader) {
                        Ok(v) => v,
                        Err(e) => {
                            self.finished = true;
                            return Err(e);
                        }
                    };
                    return Ok(Some(DecodedBlock::Verbatim { bytes: chunk.bytes }));
                }

                // --- Sample-producing commands. ---
                FunctionCode::Diff0
                | FunctionCode::Diff1
                | FunctionCode::Diff2
                | FunctionCode::Diff3 => {
                    let order = PolyOrder::from_function_code(fc)
                        .expect("DIFF0..3 always map to a PolyOrder");
                    let mu_chan = self.means[self.cursor].mu_chan();
                    let block = match decode_diff_block(
                        &mut self.reader,
                        order,
                        self.block_size,
                        &self.carries[self.cursor],
                        mu_chan,
                    ) {
                        Ok(b) => b,
                        Err(e) => {
                            self.finished = true;
                            return Err(e);
                        }
                    };
                    let item = match self.emit_sample_block(&block) {
                        Ok(item) => item,
                        Err(e) => {
                            self.finished = true;
                            return Err(e);
                        }
                    };
                    return Ok(Some(item));
                }
                FunctionCode::Qlpc => {
                    let block = match decode_qlpc_block(
                        &mut self.reader,
                        self.block_size,
                        &self.carries[self.cursor],
                    ) {
                        Ok(b) => b,
                        Err(e) => {
                            self.finished = true;
                            return Err(e);
                        }
                    };
                    let item = match self.emit_sample_block(&block) {
                        Ok(item) => item,
                        Err(e) => {
                            self.finished = true;
                            return Err(e);
                        }
                    };
                    return Ok(Some(item));
                }
                FunctionCode::Zero => {
                    let mu_chan = self.means[self.cursor].mu_chan();
                    let block = match fill_zero_block(self.block_size, mu_chan) {
                        Ok(b) => b,
                        Err(e) => {
                            self.finished = true;
                            return Err(e);
                        }
                    };
                    let item = match self.emit_sample_block(&block) {
                        Ok(item) => item,
                        Err(e) => {
                            self.finished = true;
                            return Err(e);
                        }
                    };
                    return Ok(Some(item));
                }
            }
        }
    }

    /// Internal helper: refresh the current channel's carry + mean
    /// estimator from `block` (pre-shift, per `spec/05` §1.4), then
    /// build the [`DecodedBlock::Samples`] item by left-shifting the
    /// samples by the running bit-shift on emission, and finally
    /// advance the round-robin cursor (`spec/03` §2).
    fn emit_sample_block(&mut self, block: &[i32]) -> Result<DecodedBlock> {
        // State update consumes pre-shift samples.
        self.carries[self.cursor].update_after_block(block);
        self.means[self.cursor].record_block(block);

        let samples = if self.bshift == 0 {
            block.to_vec()
        } else {
            let mut out = Vec::with_capacity(block.len());
            for &s in block {
                let shifted = (s as i64)
                    .checked_shl(self.bshift)
                    .ok_or(Error::SampleOverflow)?;
                let s_i32: i32 = shifted.try_into().map_err(|_| Error::SampleOverflow)?;
                out.push(s_i32);
            }
            out
        };
        let channel = self.cursor;
        // Round-robin advance — only sample-producing commands move
        // the cursor per `spec/03` §2.
        self.cursor = (self.cursor + 1) % self.header.channels as usize;
        Ok(DecodedBlock::Samples { channel, samples })
    }
}

impl<'a> Iterator for StreamDecoder<'a> {
    type Item = Result<DecodedBlock>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_block() {
            Ok(Some(b)) => Some(Ok(b)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Convenience wrapper: start a fresh streaming decode of `bytes`.
///
/// Equivalent to [`StreamDecoder::new`]; the function form exists so
/// callers can write `let iter = oxideav_shorten::decode_stream_iter(&buf)?;`
/// without naming the [`StreamDecoder`] type explicitly.
pub fn decode_stream_iter(bytes: &[u8]) -> Result<StreamDecoder<'_>> {
    StreamDecoder::new(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::FNSIZE;
    use crate::decode_stream;
    use crate::predictor::ENERGYSIZE;

    // ---- synthetic-stream builders (mirror driver.rs) ----

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

    fn header_param_bits(
        filetype: u32,
        channels: u32,
        blocksize: u32,
        maxlpcorder: u32,
        meanblocks: u32,
        skipbytes: u32,
    ) -> Vec<u32> {
        let bits_for = |v: u32| -> u32 {
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

    // ---- Behaviour tests ----

    #[test]
    fn iterator_yields_one_sample_block_then_quit() {
        // Single-channel single-block stream: DIFF0 [10, 20] then QUIT.
        let mut bits = header_param_bits(5, 1, 2, 0, 0, 0);
        append_diff_block(&mut bits, 0, 3, &[10, 20]);
        bits.extend(encode_uvar(4, FNSIZE));
        let buf = assemble(&bits);

        let mut iter = StreamDecoder::new(&buf).expect("new");
        assert_eq!(iter.header().channels, 1);
        assert!(!iter.is_finished());

        let item = iter.next_block().expect("next_block ok").expect("Some");
        match item {
            DecodedBlock::Samples { channel, samples } => {
                assert_eq!(channel, 0);
                assert_eq!(samples, vec![10, 20]);
            }
            other => panic!("expected Samples, got {other:?}"),
        }

        let none = iter.next_block().expect("next_block ok");
        assert!(none.is_none());
        assert!(iter.is_finished());

        // Iterator protocol — third call still returns None.
        assert!(iter.next().is_none());
    }

    #[test]
    fn iterator_yields_verbatim_then_samples() {
        // VERBATIM (3 bytes) then DIFF0 (bs = 2) then QUIT.
        let mut bits = header_param_bits(5, 1, 2, 0, 0, 0);
        bits.extend(encode_uvar(9, FNSIZE));
        bits.extend(encode_uvar(3, 5));
        for b in [b'R', b'I', b'F'] {
            bits.extend(encode_uvar(b as u32, 8));
        }
        append_diff_block(&mut bits, 0, 3, &[100, -50]);
        bits.extend(encode_uvar(4, FNSIZE));
        let buf = assemble(&bits);

        let mut iter = StreamDecoder::new(&buf).expect("new");

        let v = iter.next_block().expect("ok").expect("Some");
        assert!(v.is_verbatim());
        match v {
            DecodedBlock::Verbatim { bytes } => assert_eq!(bytes, vec![b'R', b'I', b'F']),
            other => panic!("expected Verbatim, got {other:?}"),
        }

        let s = iter.next_block().expect("ok").expect("Some");
        assert!(s.is_samples());
        match s {
            DecodedBlock::Samples { channel, samples } => {
                assert_eq!(channel, 0);
                assert_eq!(samples, vec![100, -50]);
            }
            other => panic!("expected Samples, got {other:?}"),
        }

        assert!(iter.next_block().expect("ok").is_none());
    }

    #[test]
    fn iterator_cursor_rotates_through_channels_in_order() {
        // 2 channels, blocksize 2, 4 DIFF0 sample blocks then QUIT.
        let mut bits = header_param_bits(5, 2, 2, 0, 0, 0);
        append_diff_block(&mut bits, 0, 3, &[1, 2]); // ch0
        append_diff_block(&mut bits, 0, 3, &[10, 20]); // ch1
        append_diff_block(&mut bits, 0, 3, &[3, 4]); // ch0
        append_diff_block(&mut bits, 0, 3, &[30, 40]); // ch1
        bits.extend(encode_uvar(4, FNSIZE));
        let buf = assemble(&bits);

        let iter = StreamDecoder::new(&buf).expect("new");
        let blocks: Vec<DecodedBlock> = iter.map(|r| r.expect("ok")).collect();
        assert_eq!(blocks.len(), 4);

        let expected_channels = [0, 1, 0, 1];
        let expected_samples = [vec![1, 2], vec![10, 20], vec![3, 4], vec![30, 40]];
        for (i, b) in blocks.iter().enumerate() {
            match b {
                DecodedBlock::Samples { channel, samples } => {
                    assert_eq!(*channel, expected_channels[i], "block {i} channel");
                    assert_eq!(*samples, expected_samples[i], "block {i} samples");
                }
                other => panic!("block {i}: expected Samples, got {other:?}"),
            }
        }
    }

    #[test]
    fn iterator_absorbs_blocksize_command_then_yields_resized_block() {
        // Default bs = 4, BLOCKSIZE -> 2 before the second block.
        let mut bits = header_param_bits(5, 1, 4, 0, 0, 0);
        append_diff_block(&mut bits, 0, 3, &[1, 2, 3, 4]);
        bits.extend(encode_uvar(5, FNSIZE)); // BLOCKSIZE = 5
        bits.extend(encode_ulong(2, 2));
        append_diff_block(&mut bits, 0, 3, &[5, 6]);
        bits.extend(encode_uvar(4, FNSIZE)); // QUIT
        let buf = assemble(&bits);

        let mut iter = StreamDecoder::new(&buf).expect("new");
        assert_eq!(iter.current_block_size(), 4);

        let b0 = iter.next_block().expect("ok").expect("Some");
        assert_eq!(b0.sample_count(), 4);
        // After block 0 the BLOCKSIZE command is absorbed in the next
        // next_block call, before the second sample block is yielded.

        let b1 = iter.next_block().expect("ok").expect("Some");
        match b1 {
            DecodedBlock::Samples { samples, .. } => assert_eq!(samples, vec![5, 6]),
            other => panic!("expected Samples, got {other:?}"),
        }
        // After the resized block has been emitted the running
        // block size reflects the override.
        assert_eq!(iter.current_block_size(), 2);
        assert!(iter.next_block().expect("ok").is_none());
    }

    #[test]
    fn iterator_absorbs_bitshift_command_and_shifts_emitted_samples() {
        // BITSHIFT bshift = 3 then DIFF1 (bs = 4). Pre-shift output
        // [1, 2, 3, 4] becomes [8, 16, 24, 32] on the wire.
        let mut bits = header_param_bits(5, 1, 4, 0, 0, 0);
        bits.extend(encode_uvar(6, FNSIZE)); // BITSHIFT = 6
        bits.extend(encode_uvar(3, 2)); // bshift = 3
        append_diff_block(&mut bits, 1, 3, &[1, 1, 1, 1]);
        bits.extend(encode_uvar(4, FNSIZE));
        let buf = assemble(&bits);

        let mut iter = StreamDecoder::new(&buf).expect("new");
        assert_eq!(iter.current_bitshift(), 0);

        let b = iter.next_block().expect("ok").expect("Some");
        match b {
            DecodedBlock::Samples { samples, .. } => assert_eq!(samples, vec![8, 16, 24, 32]),
            other => panic!("expected Samples, got {other:?}"),
        }
        // The bitshift state survives across blocks.
        assert_eq!(iter.current_bitshift(), 3);
    }

    #[test]
    fn iterator_zero_command_emits_running_mean_block() {
        // Same scenario as driver.rs's zero_command_emits_running_mean_block:
        // meanblocks = 4, DIFF0 [6,6,6,6] then ZERO yields [2,2,2,2].
        let mut bits = header_param_bits(5, 1, 4, 0, 4, 0);
        append_diff_block(&mut bits, 0, 3, &[6, 6, 6, 6]);
        bits.extend(encode_uvar(8, FNSIZE)); // ZERO
        bits.extend(encode_uvar(4, FNSIZE)); // QUIT
        let buf = assemble(&bits);

        let iter = StreamDecoder::new(&buf).expect("new");
        let blocks: Vec<DecodedBlock> = iter.map(|r| r.expect("ok")).collect();
        assert_eq!(blocks.len(), 2);
        match &blocks[0] {
            DecodedBlock::Samples { samples, .. } => assert_eq!(samples, &vec![6, 6, 6, 6]),
            other => panic!("block 0: {other:?}"),
        }
        match &blocks[1] {
            DecodedBlock::Samples { samples, .. } => assert_eq!(samples, &vec![2, 2, 2, 2]),
            other => panic!("block 1: {other:?}"),
        }
    }

    #[test]
    fn iterator_truncated_yields_error_once_then_none() {
        // DIFF0 claims bs = 4 residuals but supply only one.
        let mut bits = header_param_bits(5, 1, 4, 0, 0, 0);
        bits.extend(encode_uvar(0, FNSIZE));
        bits.extend(encode_uvar(3, ENERGYSIZE));
        bits.extend(encode_svar(1, 4));
        let buf = assemble(&bits);

        let mut iter = StreamDecoder::new(&buf).expect("new");
        let err = iter.next().expect("Some").expect_err("Err");
        assert_eq!(err, Error::Truncated);
        // After an error the iterator is exhausted.
        assert!(iter.next().is_none());
        assert!(iter.is_finished());
    }

    #[test]
    fn iterator_zero_channel_header_is_rejected_at_construction() {
        let bits = header_param_bits(5, 0, 4, 0, 0, 0);
        let buf = assemble(&bits);
        assert_eq!(StreamDecoder::new(&buf).err(), Some(Error::Truncated));
    }

    #[test]
    fn iterator_matches_whole_stream_driver_on_two_channel_diff_qlpc_mix() {
        // A multi-block stream with DIFF1 + DIFF0 across two channels;
        // collect the iterator's per-channel samples and compare them
        // to the round-7 driver's `DecodedStream::channels`. This is
        // the load-bearing equivalence test — the iterator's per-
        // block state machine must yield the same per-channel samples
        // as the all-at-once driver for any well-formed stream.
        let mut bits = header_param_bits(5, 2, 4, 0, 0, 0);
        append_diff_block(&mut bits, 1, 3, &[1, 1, 1, 1]); // ch0 DIFF1
        append_diff_block(&mut bits, 0, 3, &[10, 20, -5, 7]); // ch1 DIFF0
        append_diff_block(&mut bits, 1, 3, &[2, -1, 0, 3]); // ch0 DIFF1
        append_diff_block(&mut bits, 0, 3, &[1, -2, 4, -3]); // ch1 DIFF0
        bits.extend(encode_uvar(4, FNSIZE));
        let buf = assemble(&bits);

        // Reference: round-7 driver.
        let reference = decode_stream(&buf).expect("reference decode_stream");

        // Streaming iterator: collect per-block items, accumulate per
        // channel.
        let iter = StreamDecoder::new(&buf).expect("new");
        let mut per_channel: Vec<Vec<i32>> = vec![Vec::new(); reference.header.channels as usize];
        for item in iter {
            match item.expect("ok") {
                DecodedBlock::Samples { channel, samples } => {
                    per_channel[channel].extend_from_slice(&samples);
                }
                DecodedBlock::Verbatim { .. } => {
                    panic!("no verbatim expected in this fixture")
                }
            }
        }
        assert_eq!(per_channel, reference.channels);
    }

    #[test]
    fn iterator_header_accessor_matches_decode_stream() {
        let mut bits = header_param_bits(3, 2, 8, 0, 4, 0);
        append_diff_block(&mut bits, 0, 3, &[1, 2, 3, 4, 5, 6, 7, 8]);
        append_diff_block(&mut bits, 0, 3, &[9, 10, 11, 12, 13, 14, 15, 16]);
        bits.extend(encode_uvar(4, FNSIZE));
        let buf = assemble(&bits);

        let iter = StreamDecoder::new(&buf).expect("new");
        let from_driver = decode_stream(&buf).expect("driver");
        assert_eq!(iter.header(), &from_driver.header);
    }

    #[test]
    fn iterator_current_channel_advances_only_on_sample_commands() {
        let mut bits = header_param_bits(5, 2, 4, 0, 0, 0);
        // VERBATIM (housekeeping for cursor purposes per spec/03 §2.1).
        bits.extend(encode_uvar(9, FNSIZE));
        bits.extend(encode_uvar(2, 5));
        for b in [b'X', b'Y'] {
            bits.extend(encode_uvar(b as u32, 8));
        }
        // DIFF0 ch0 — cursor 0 -> 1 after.
        append_diff_block(&mut bits, 0, 3, &[1, 2, 3, 4]);
        // BLOCKSIZE override (housekeeping, cursor unchanged).
        bits.extend(encode_uvar(5, FNSIZE));
        bits.extend(encode_ulong(2, 2));
        // DIFF0 ch1 — cursor 1 -> 0 after.
        append_diff_block(&mut bits, 0, 3, &[10, 20]);
        bits.extend(encode_uvar(4, FNSIZE));
        let buf = assemble(&bits);

        let mut iter = StreamDecoder::new(&buf).expect("new");
        assert_eq!(iter.current_channel(), 0);

        // First yielded item: VERBATIM; cursor unchanged.
        let v = iter.next_block().expect("ok").expect("Some");
        assert!(v.is_verbatim());
        assert_eq!(iter.current_channel(), 0);

        // Second: ch0 samples; cursor advances to 1.
        let s0 = iter.next_block().expect("ok").expect("Some");
        match s0 {
            DecodedBlock::Samples { channel, .. } => assert_eq!(channel, 0),
            other => panic!("{other:?}"),
        }
        assert_eq!(iter.current_channel(), 1);

        // Third: ch1 samples (after the absorbed BLOCKSIZE); cursor
        // wraps back to 0.
        let s1 = iter.next_block().expect("ok").expect("Some");
        match s1 {
            DecodedBlock::Samples { channel, samples } => {
                assert_eq!(channel, 1);
                assert_eq!(samples.len(), 2); // BLOCKSIZE override took effect.
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(iter.current_channel(), 0);

        assert!(iter.next_block().expect("ok").is_none());
    }

    #[test]
    fn decode_stream_iter_convenience_wrapper_matches_struct_construction() {
        let mut bits = header_param_bits(5, 1, 2, 0, 0, 0);
        append_diff_block(&mut bits, 0, 3, &[7, -7]);
        bits.extend(encode_uvar(4, FNSIZE));
        let buf = assemble(&bits);

        let iter = decode_stream_iter(&buf).expect("decode_stream_iter");
        let blocks: Vec<DecodedBlock> = iter.map(|r| r.expect("ok")).collect();
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            DecodedBlock::Samples { channel, samples } => {
                assert_eq!(*channel, 0);
                assert_eq!(samples, &vec![7, -7]);
            }
            other => panic!("{other:?}"),
        }
    }
}
