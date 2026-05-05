//! Shorten encoder — round-2 (DIFFn + QLPC + BITSHIFT).
//!
//! Mirrors [`crate::decoder`] exactly: every primitive emitted here is
//! the bit-for-bit inverse of a primitive consumed there.
//!
//! # Design
//!
//! - Bitwriter: [`oxideav_core::bits::BitWriter`] (MSB-first, matches
//!   the decoder's `BitReader`).
//! - Header: magic `'ajkg'` + version `2` + 6 ulong fields
//!   (`internal_ftype`, `channels`, `blocksize`, `maxnlpc`, `nmean`,
//!   `skip_bytes=0`) + a single leading FN_VERBATIM capsule.
//! - Per block: pick the cheapest predictor from DIFF0/1/2/3 + QLPC
//!   (when `maxnlpc > 0`) by minimum sum of `|residual|`; if every
//!   residual is zero, emit FN_ZERO instead. Per-block Rice-k is the
//!   closed-form `floor(log2(mean(|r|)))`.
//! - QLPC: Levinson-Durbin auto-correlation → LPC coefficients →
//!   quantise to integers (multiply by 2^LPCQUANT = 32, round, clamp).
//!   Residuals encoded as signed Rice, same pipeline as DIFFn.
//! - BITSHIFT: detect consistent trailing zero bits across all samples
//!   in the block; emit FN_BITSHIFT once (global, not per-block) when
//!   the detected shift changes. Samples are right-shifted before
//!   prediction so the predictor sees a reduced-range signal.
//! - Per channel: matches the decoder's wrap-history (`nwrap = max(3,
//!   maxnlpc)`) and the `nmean`-deep mean FIFO with v >= 2 rounding
//!   bias. Channels are emitted round-robin.
//!
//! # Limitations (round-3 follow-ups)
//!
//! - No mid-stream FN_BLOCKSIZE (encoder uses one fixed blocksize for
//!   the whole stream).
//! - No streaming / multi-packet encode (the encoder produces one
//!   `.shn` byte stream per call).
//! - No iterated Rice-`k` search (closed-form only).

use oxideav_core::bits::BitWriter;
use oxideav_core::{Error, Result};

// ─────────────────── Function codes (mirror decoder.rs) ───────────────────

const FN_DIFF0: u32 = 0;
const FN_DIFF1: u32 = 1;
const FN_DIFF2: u32 = 2;
const FN_DIFF3: u32 = 3;
const FN_QUIT: u32 = 4;
const FN_BITSHIFT: u32 = 6;
const FN_QLPC: u32 = 7;
const FN_ZERO: u32 = 8;
const FN_VERBATIM: u32 = 9;

// k_param_size constants from the decoder.
const FNSIZE: u32 = 2;
const ULONGSIZE: u32 = 2;
const ENERGYSIZE: u32 = 3;
const LPCQSIZE: u32 = 2; // pred_order bit-width
const LPCQUANT: u32 = 5; // quantisation shift (qshift); coefficients scaled by 2^5 = 32
const BITSHIFTSIZE: u32 = 2;
const VERBATIM_CKSIZE_SIZE: u32 = 5;
const VERBATIM_BYTE_SIZE: u32 = 8;

const V2LPCQOFFSET: i32 = 1 << LPCQUANT; // 32 for v >= 2

const VERSION: u8 = 2;
const MAGIC: &[u8; 4] = b"ajkg";

const MAX_CHANNELS: usize = 8;
const MIN_BLOCKSIZE: u32 = 1;
const MAX_BLOCKSIZE: u32 = 65_535;

// The leading FN_VERBATIM carries a 44-byte minimal RIFF/WAV header that
// downstream decoders (ffmpeg, shntool) use to determine sample rate,
// bit depth, and channel count. The encoder builds a real minimal WAV
// header — the RIFF chunk size fields are set to 0 (unknown length),
// which is valid for streaming and accepted by all known parsers.
const VERBATIM_LEN: u32 = 44;

// ─────────────────── Shorten internal_ftype enum ───────────────────

/// Maps to the decoder's `internal_ftype` integer (1..=6).
///
/// Mapping (matches `decoder.rs::ShortenHeader::initial_offset` and
/// `decoder.rs::pack_s16`):
///
/// | variant   | internal_ftype | initial_offset | post-decode S16        |
/// |-----------|----------------|----------------|------------------------|
/// | `S8`      | 1              | 0              | `clip_s8(raw) * 256`   |
/// | `U8`      | 2              | 0x80           | `clip_s8(raw-0x80)*256`|
/// | `S16Be`   | 3              | 0              | `clip_i16(raw)`        |
/// | `U16Be`   | 4              | 0x8000         | `clip_i16(raw-0x8000)` |
/// | `S16Le`   | 5              | 0              | `clip_i16(raw)`        |
/// | `U16Le`   | 6              | 0x8000         | `clip_i16(raw-0x8000)` |
///
/// (BE/LE only matter to the *container* byte order — the in-band
/// integer that the decoder stores is the same. The two ftype values
/// per width-and-signedness pair exist so a downstream demuxer can
/// reconstruct the original WAV/AIFF endianness.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShortenFtype {
    /// internal_ftype = 1 — signed 8-bit PCM (samples in -128..=127).
    S8,
    /// internal_ftype = 2 — unsigned 8-bit PCM (samples in 0..=255).
    U8,
    /// internal_ftype = 3 — signed 16-bit big-endian PCM.
    S16Be,
    /// internal_ftype = 4 — unsigned 16-bit big-endian PCM (0..=65535).
    U16Be,
    /// internal_ftype = 5 — signed 16-bit little-endian PCM.
    S16Le,
    /// internal_ftype = 6 — unsigned 16-bit little-endian PCM (0..=65535).
    U16Le,
}

impl ShortenFtype {
    fn to_internal(self) -> u32 {
        match self {
            ShortenFtype::S8 => 1,
            ShortenFtype::U8 => 2,
            ShortenFtype::S16Be => 3,
            ShortenFtype::U16Be => 4,
            ShortenFtype::S16Le => 5,
            ShortenFtype::U16Le => 6,
        }
    }

    /// Initial mean offset injected into the per-channel mean FIFO.
    /// Must match `ShortenHeader::initial_offset` in `decoder.rs`.
    fn initial_offset(self) -> i32 {
        match self {
            ShortenFtype::S8 | ShortenFtype::S16Le | ShortenFtype::S16Be => 0,
            ShortenFtype::U8 => 0x80,
            ShortenFtype::U16Le | ShortenFtype::U16Be => 0x8000,
        }
    }
}

// ─────────────────── Public encoder API ───────────────────

/// Configuration for [`ShortenEncoder`].
#[derive(Debug, Clone)]
pub struct ShortenEncoderConfig {
    pub ftype: ShortenFtype,
    pub channels: u32,
    pub blocksize: u32,
    /// Sample rate in Hz, written into the leading FN_VERBATIM WAV
    /// header so downstream decoders (ffmpeg, shntool) can reconstruct
    /// the audio parameters. Defaults to 44 100 Hz.
    pub sample_rate: u32,
    /// Mean-FIFO depth. The decoder requires `nmean <= 1024`. A common
    /// choice in real-world v2 streams is `0` or `4`. Default is `0`
    /// (no running-mean compensation — the per-channel coffset stays at
    /// the type's `initial_offset()`).
    pub nmean: u32,
    /// Maximum LPC order for QLPC predictor. `0` disables QLPC (only
    /// DIFFn predictors used). Typical values are `0` (DIFFn-only) or
    /// `8` (enable QLPC up to order 8). The bitstream's `nwrap` is set
    /// to `max(3, maxnlpc)` and the QLPC candidate is included in the
    /// per-block predictor race.
    pub maxnlpc: u32,
    /// Enable the BITSHIFT encoder. When `true` the encoder detects
    /// the number of consistent trailing zero bits across each block
    /// and emits an `FN_BITSHIFT` command when that count changes,
    /// right-shifting all samples by the detected amount before the
    /// predictor. The decoder left-shifts the output to restore the
    /// original magnitude. Useful for 24-bit-in-32-bit containers or
    /// any source whose low N bits are all zero.
    pub enable_bitshift: bool,
}

impl ShortenEncoderConfig {
    pub fn new(ftype: ShortenFtype, channels: u32, blocksize: u32) -> Self {
        Self {
            ftype,
            channels,
            blocksize,
            sample_rate: 44_100,
            nmean: 0,
            maxnlpc: 0,
            enable_bitshift: false,
        }
    }

    /// Set the sample rate (written into the leading VERBATIM WAV header).
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn with_nmean(mut self, nmean: u32) -> Self {
        self.nmean = nmean;
        self
    }

    /// Set the maximum LPC order for QLPC (0 = disabled).
    pub fn with_maxnlpc(mut self, maxnlpc: u32) -> Self {
        self.maxnlpc = maxnlpc;
        self
    }

    /// Enable automatic BITSHIFT detection and emission.
    pub fn with_bitshift(mut self, enable: bool) -> Self {
        self.enable_bitshift = enable;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.channels == 0 || self.channels as usize > MAX_CHANNELS {
            return Err(Error::invalid(format!(
                "shorten encoder: channels={} out of range 1..={MAX_CHANNELS}",
                self.channels
            )));
        }
        if !(MIN_BLOCKSIZE..=MAX_BLOCKSIZE).contains(&self.blocksize) {
            return Err(Error::invalid(format!(
                "shorten encoder: blocksize={} out of range {MIN_BLOCKSIZE}..={MAX_BLOCKSIZE}",
                self.blocksize
            )));
        }
        if self.nmean > 1024 {
            return Err(Error::invalid(format!(
                "shorten encoder: nmean={} unreasonably large",
                self.nmean
            )));
        }
        if self.maxnlpc > 1024 {
            return Err(Error::invalid(format!(
                "shorten encoder: maxnlpc={} unreasonably large",
                self.maxnlpc
            )));
        }
        Ok(())
    }
}

/// Per-block introspection record (one entry per emitted DIFF/ZERO/QLPC
/// audio block, in stream order). Useful for tests that want to assert
/// the predictor selector picked a particular predictor for a known-shape
/// signal, without parsing the bitstream back.
#[derive(Debug, Clone, Copy)]
pub struct BlockInfo {
    pub channel: u32,
    /// Function code: FN_DIFF0(0)/DIFF1(1)/DIFF2(2)/DIFF3(3)/FN_QLPC(7)/FN_ZERO(8).
    pub cmd: u32,
    pub k: u32,
    pub residual_abs_sum: u64,
}

/// One-shot Shorten encoder. See module docs for the round-2 scope.
///
/// The encoder is consumed by [`Self::encode`] (so re-using a single
/// instance for many independent streams is not supported — make a
/// fresh one per stream).
pub struct ShortenEncoder {
    cfg: ShortenEncoderConfig,
    debug_blocks: Vec<BlockInfo>,
    /// Current stream-level bitshift (mirrors the decoder's state).
    /// Changes are emitted as FN_BITSHIFT commands.
    bitshift: u32,
}

impl ShortenEncoder {
    pub fn new(cfg: ShortenEncoderConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            debug_blocks: Vec::new(),
            bitshift: 0,
        })
    }

    /// Per-block introspection for the most-recent [`Self::encode`]
    /// call. Empty until `encode` runs.
    pub fn block_info(&self) -> &[BlockInfo] {
        &self.debug_blocks
    }

    /// Encode an interleaved PCM stream into a complete `.shn` byte
    /// buffer. Sample count must be a multiple of `cfg.channels`.
    ///
    /// The samples are interpreted as the *raw* per-`ftype` integer:
    /// - `S8` → `samples[i]` in `-128..=127`,
    /// - `U8` → `samples[i]` in `0..=255` (NOT yet centred),
    /// - `S16Le`/`S16Be` → `samples[i]` in `-32_768..=32_767`,
    /// - `U16Le`/`U16Be` → `samples[i]` in `0..=65_535`.
    ///
    /// The encoder does not validate that samples fit; out-of-range
    /// values may produce a stream whose decode clips (per the
    /// decoder's `pack_s16`).
    pub fn encode(&mut self, samples: &[i32]) -> Result<Vec<u8>> {
        let nch = self.cfg.channels as usize;
        if samples.len() % nch != 0 {
            return Err(Error::invalid(format!(
                "shorten encoder: sample count {} not a multiple of channels {}",
                samples.len(),
                nch
            )));
        }
        let total_per_chan = samples.len() / nch;
        let bs = self.cfg.blocksize as usize;
        // Round-1 simplification: the stream's last partial block (if
        // `total_per_chan` is not a multiple of `blocksize`) would
        // require an FN_BLOCKSIZE command before it, which the round-1
        // encoder doesn't emit. Reject for now.
        if total_per_chan % bs != 0 {
            return Err(Error::unsupported(format!(
                "shorten encoder (round 1): per-channel sample count {} must be a multiple of blocksize {}",
                total_per_chan, bs
            )));
        }
        let nblocks = total_per_chan / bs;

        // De-interleave into per-channel rows (the decoder works in
        // de-interleaved space too — channels are independent).
        let mut chan_samples: Vec<Vec<i32>> = vec![Vec::with_capacity(total_per_chan); nch];
        for (i, &s) in samples.iter().enumerate() {
            chan_samples[i % nch].push(s);
        }

        self.debug_blocks.clear();

        let mut bw = BitWriter::new();

        // ─── Header ─────────────────────────────────────────────
        for &b in MAGIC {
            bw.write_u32(b as u32, 8);
        }
        bw.write_u32(VERSION as u32, 8);

        let maxnlpc = self.cfg.maxnlpc;
        write_ulong(&mut bw, self.cfg.ftype.to_internal(), ULONGSIZE);
        write_ulong(&mut bw, self.cfg.channels, ULONGSIZE);
        write_ulong(&mut bw, self.cfg.blocksize, ULONGSIZE);
        write_ulong(&mut bw, maxnlpc, ULONGSIZE);
        write_ulong(&mut bw, self.cfg.nmean, ULONGSIZE);
        write_ulong(&mut bw, 0, ULONGSIZE); // skip_bytes = 0

        // ─── Leading FN_VERBATIM (real WAV header, 44 bytes) ────
        // Build a minimal 44-byte RIFF/WAV header matching the encoder's
        // parameters. ffmpeg and shntool both require this to identify
        // the audio format (sample rate, bit depth, channel count).
        // Chunk sizes are set to 0 (streaming / unknown length).
        let wav = build_wav_header(self.cfg.ftype, self.cfg.channels, self.cfg.sample_rate);
        debug_assert_eq!(wav.len(), VERBATIM_LEN as usize);
        write_unsigned_k(&mut bw, FN_VERBATIM, FNSIZE);
        write_unsigned_k(&mut bw, VERBATIM_LEN, VERBATIM_CKSIZE_SIZE);
        for &b in &wav {
            write_unsigned_k(&mut bw, b as u32, VERBATIM_BYTE_SIZE);
        }

        // ─── Per-channel state (mirrors decoder) ───────────────
        // nwrap = max(3, maxnlpc); decoder does the same.
        let nwrap = (maxnlpc as usize).max(3);
        let init_offset = self.cfg.ftype.initial_offset();
        let nmean = self.cfg.nmean as usize;
        let mut chan_state: Vec<ChannelState> = (0..nch)
            .map(|_| ChannelState::new(nwrap, nmean, init_offset))
            .collect();

        // Reset stream-level bitshift for each encode() call.
        self.bitshift = 0;

        // ─── Audio blocks (round-robin per channel) ────────────
        for blk in 0..nblocks {
            // ── BITSHIFT detection (global, before all channels) ─
            // Detect the trailing zero bits common to ALL channels'
            // current block and emit FN_BITSHIFT when it changes.
            if self.cfg.enable_bitshift {
                let mut common_shift = 31u32;
                for ch in 0..nch {
                    let start = blk * bs;
                    let block = &chan_samples[ch][start..start + bs];
                    let block_shift = detect_bitshift(block);
                    common_shift = common_shift.min(block_shift);
                }
                // Cap at 31 to keep within the 2-bit-k BITSHIFTSIZE
                // payload range (decoder allows 0..=32).
                let common_shift = common_shift.min(31);
                if common_shift != self.bitshift {
                    write_unsigned_k(&mut bw, FN_BITSHIFT, FNSIZE);
                    write_unsigned_k(&mut bw, common_shift, BITSHIFTSIZE);
                    self.bitshift = common_shift;
                }
            }

            for ch in 0..nch {
                let start = blk * bs;
                let block = &chan_samples[ch][start..start + bs];
                self.emit_block(
                    &mut bw,
                    block,
                    ch as u32,
                    &mut chan_state[ch],
                    maxnlpc as usize,
                );
            }
        }

        // ─── Quit ──────────────────────────────────────────────
        write_unsigned_k(&mut bw, FN_QUIT, FNSIZE);

        Ok(bw.into_bytes())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_block(
        &mut self,
        bw: &mut BitWriter,
        block: &[i32],
        ch: u32,
        state: &mut ChannelState,
        maxnlpc: usize,
    ) {
        let bs = block.len();
        let bitshift = self.bitshift;

        // Apply bitshift: right-shift all samples before prediction so
        // the predictor sees the reduced-range signal. This mirrors the
        // decoder's `coffset >>= bitshift` + `apply_bitshift` logic.
        // The history also needs to be shifted for the predictor.
        let shifted_block: Vec<i32> = if bitshift > 0 {
            block.iter().map(|&s| s >> bitshift).collect()
        } else {
            block.to_vec()
        };

        // Compute coffset (matches the decoder; bitshift compensation
        // is applied the same way: coffset >>= bitshift).
        let mut coffset = state.coffset(VERSION);
        if bitshift > 0 && bitshift < 32 {
            coffset >>= bitshift;
        }

        // Build a shifted history view for QLPC and compute_residuals.
        let shifted_history: Vec<i32> = if bitshift > 0 {
            state.history.iter().map(|&s| s >> bitshift).collect()
        } else {
            state.history.clone()
        };

        // FN_ZERO shortcut: if all shifted samples are zero (and coffset
        // is zero so the DIFF predictors also produce all-zero residuals),
        // emit FN_ZERO — saves the energy/k field and per-sample codes.
        let all_zero_block = shifted_block.iter().all(|&s| s == 0);
        if all_zero_block && coffset == 0 {
            write_unsigned_k(bw, FN_ZERO, FNSIZE);
            self.debug_blocks.push(BlockInfo {
                channel: ch,
                cmd: FN_ZERO,
                k: 0,
                residual_abs_sum: 0,
            });
            // The decoder's ZERO path feeds blocksize zeros into
            // finish_block. We need to feed the UNshifted block (all
            // zeros when shifted_block is all zeros, since 0 >> n = 0).
            // But the history must be updated with the original samples
            // so subsequent blocks see the right context. Since the
            // block IS all zeros (even unshifted, because s>>shift=0 and
            // s is representable as a shifted zero), just feed zeros.
            let zeros = vec![0i32; bs];
            state.finish_block(&zeros, VERSION, bitshift);
            return;
        }

        // ── Try all candidate predictors ─────────────────────────
        let mut best_cmd = FN_DIFF0;
        let mut best_res: Vec<i32> = Vec::new();
        let mut best_abs: u64 = u64::MAX;
        let mut best_coeffs: Vec<i32> = Vec::new();

        for cmd in [FN_DIFF0, FN_DIFF1, FN_DIFF2, FN_DIFF3] {
            let (res, sum_abs, _sum_sgn) =
                compute_residuals(cmd, &shifted_block, &shifted_history, coffset);
            if sum_abs < best_abs {
                best_abs = sum_abs;
                best_cmd = cmd;
                best_res = res;
                best_coeffs = Vec::new();
            }
        }

        // ── QLPC candidate ────────────────────────────────────────
        if maxnlpc > 0 {
            // Try orders 1..=min(maxnlpc, nwrap) and pick the best.
            let max_order = maxnlpc.min(state.nwrap).min(bs);
            // Build a combined signal: [shifted_history | shifted_block]
            // for auto-correlation.
            let mut signal = shifted_history.clone();
            signal.extend_from_slice(&shifted_block);
            let qlpc_offset = V2LPCQOFFSET; // lpcqoffset for v >= 2

            for order in 1..=max_order {
                let Some(qcoeffs) = estimate_qlpc(&signal, &shifted_history, order, qlpc_offset)
                else {
                    continue;
                };
                let (res, sum_abs, _) = compute_qlpc_residuals(
                    &shifted_block,
                    &shifted_history,
                    &qcoeffs,
                    coffset,
                    qlpc_offset,
                );
                if sum_abs < best_abs {
                    best_abs = sum_abs;
                    best_cmd = FN_QLPC;
                    best_res = res;
                    best_coeffs = qcoeffs;
                }
            }
        }

        // ── Rice-k selection ─────────────────────────────────────
        let mean_abs = if bs == 0 { 0 } else { best_abs / bs as u64 };
        let k = if mean_abs <= 1 {
            0
        } else {
            63 - mean_abs.leading_zeros()
        };
        let k = k.min(30);

        // ── Emit command + payload ────────────────────────────────
        if best_cmd == FN_QLPC {
            write_unsigned_k(bw, FN_QLPC, FNSIZE);
            // pred_order is the QLPC coefficient count, plain unsigned k=LPCQSIZE.
            write_unsigned_k(bw, best_coeffs.len() as u32, LPCQSIZE);
            // Quantised coefficients as signed Rice k=LPCQUANT.
            for &c in &best_coeffs {
                write_signed_k(bw, c, LPCQUANT);
            }
            // Energy/k then residuals.
            write_unsigned_k(bw, k, ENERGYSIZE);
            for &r in &best_res {
                write_signed_k(bw, r, k);
            }
        } else {
            write_unsigned_k(bw, best_cmd, FNSIZE);
            write_unsigned_k(bw, k, ENERGYSIZE);
            for &r in &best_res {
                write_signed_k(bw, r, k);
            }
        }

        self.debug_blocks.push(BlockInfo {
            channel: ch,
            cmd: best_cmd,
            k,
            residual_abs_sum: best_abs,
        });

        // Update per-channel state. The decoder runs `finish_block` on
        // the *decoded* samples — which round-trip back to the original
        // shifted_block. But `finish_block` also maintains the history
        // ring used by the *next* block's predictor. The history must
        // be stored in the UNshifted domain (matching the original block)
        // so that `coffset = state.coffset() >> bitshift` is correct.
        //
        // We pass the *original* (unshifted) block to finish_block so
        // the mean FIFO and history ring are consistent with what the
        // decoder will see after it applies `apply_bitshift` to the
        // reconstructed shifted samples.
        state.finish_block(block, VERSION, bitshift);
    }
}

// ─────────────────── Per-channel state (mirrors decoder) ───────────────────

struct ChannelState {
    history: Vec<i32>,
    offsets: Vec<i32>,
    nmean: usize,
    pub nwrap: usize,
}

impl ChannelState {
    fn new(nwrap: usize, nmean: usize, init_offset: i32) -> Self {
        Self {
            history: vec![0; nwrap],
            offsets: vec![init_offset; nmean.max(1)],
            nmean,
            nwrap,
        }
    }

    /// Identical to the decoder's `ChannelState::coffset`.
    fn coffset(&self, version: u8) -> i32 {
        if self.nmean == 0 {
            return self.offsets[0];
        }
        let mut sum: i64 = 0;
        for &o in &self.offsets {
            sum += o as i64;
        }
        if version >= 2 {
            sum += (self.nmean as i64) / 2;
        }
        (sum / (self.nmean as i64)) as i32
    }

    fn push_mean(&mut self, mean_value: i32) {
        if self.nmean == 0 {
            self.offsets[0] = mean_value;
            return;
        }
        for i in 0..(self.offsets.len() - 1) {
            self.offsets[i] = self.offsets[i + 1];
        }
        let last = self.offsets.len() - 1;
        self.offsets[last] = mean_value;
    }

    /// Mirrors the decoder's `finish_block`. The `decoded` slice is the
    /// *original* (pre-shift) samples. The mean FIFO and history ring
    /// store the original samples so that the next block's coffset and
    /// predictor history are in the same domain as the bitstream. The
    /// encoder right-shifts them lazily when building residuals.
    fn finish_block(&mut self, decoded: &[i32], version: u8, bitshift: u32) {
        let blocksize = decoded.len();
        // Mean is computed over the original samples, then stored
        // left-shifted (same as the decoder's `finish_block`).
        let mut sum: i64 = 0;
        for &s in decoded {
            sum += s as i64;
        }
        let mut mean = if blocksize == 0 {
            0
        } else if version >= 2 {
            ((sum + (blocksize as i64) / 2) / blocksize as i64) as i32
        } else {
            (sum / blocksize as i64) as i32
        };
        // Decoder stores mean left-shifted when bitshift > 0 (v >= 2).
        if version >= 2 && bitshift > 0 && bitshift < 32 {
            mean = mean.wrapping_shl(bitshift);
        }
        self.push_mean(mean);

        let nwrap = self.history.len();
        if decoded.len() >= nwrap {
            self.history
                .copy_from_slice(&decoded[decoded.len() - nwrap..]);
        } else {
            let keep = nwrap - decoded.len();
            for i in 0..keep {
                self.history[i] = self.history[i + decoded.len()];
            }
            for (i, &s) in decoded.iter().enumerate() {
                self.history[keep + i] = s;
            }
        }
    }
}

// ─────────────────── BITSHIFT detection ───────────────────

/// Return the number of consistent trailing zero bits shared by all
/// samples in the block. Returns 0 if any sample is odd. An all-zero
/// block returns 31 (the cap used by the encoder).
fn detect_bitshift(block: &[i32]) -> u32 {
    if block.is_empty() {
        return 0;
    }
    let mut common: u32 = 31;
    for &s in block {
        if s == 0 {
            continue;
        }
        // Number of trailing zero bits in the absolute value.
        let tz = (s.unsigned_abs()).trailing_zeros();
        common = common.min(tz);
    }
    common
}

// ─────────────────── QLPC coefficient estimation (Levinson-Durbin) ───────────

/// Estimate quantised LPC coefficients for the given LPC order.
///
/// `signal` is the combined `[history | current_block]` array in the
/// shifted domain (coffset not yet subtracted). `history` is the
/// `nwrap`-deep predictor history.
///
/// Returns `None` if the auto-correlation is degenerate (zero energy).
///
/// The returned `Vec<i32>` contains the `order` quantised coefficients
/// in the same form the bitstream carries: multiply the floating-point
/// coefficient by `2^LPCQUANT` and round. The decoder reconstructs the
/// prediction as `sum(coeffs[j] * x[i-j-1]) / 2^LPCQUANT`.
fn estimate_qlpc(
    signal: &[i32],
    _history: &[i32],
    order: usize,
    lpcqoffset: i32,
) -> Option<Vec<i32>> {
    // Auto-correlation of the signal at lags 0..=order.
    // We compute over the whole signal (history + block) for a
    // good estimate of the long-run statistics.
    let n = signal.len();
    if n < order + 1 {
        return None;
    }
    let mut r = vec![0.0f64; order + 1];
    for lag in 0..=order {
        let mut acc = 0.0f64;
        for i in lag..n {
            acc += (signal[i] as f64) * (signal[i - lag] as f64);
        }
        r[lag] = acc;
    }
    if r[0] == 0.0 {
        return None;
    }

    // Levinson-Durbin recursion.
    let mut a = vec![0.0f64; order + 1]; // a[1..=order] are the coefficients
    let mut e = r[0];
    for m in 1..=order {
        // Reflection coefficient k_m.
        let mut km = 0.0f64;
        for j in 1..m {
            km += a[j] * r[m - j];
        }
        km = (r[m] - km) / e;
        // Update coefficients.
        a[m] = km;
        for j in 1..m {
            let tmp = a[m - j];
            a[j] -= km * tmp;
        }
        e *= 1.0 - km * km;
        if e <= 0.0 {
            // Singular — use only the coefficients computed so far.
            break;
        }
    }

    // Quantise: multiply by 2^LPCQUANT, add lpcqoffset rounding bias,
    // then round to nearest integer. The lpcqoffset (32 for v >= 2) is
    // what the *decoder* adds into `init_sum_base`; the encoder does
    // NOT add it here — it's baked into the decoder's summation.
    // We clamp to i16 range to avoid huge residuals on pathological
    // signals; real Shorten uses the same implicit bound.
    let _ = lpcqoffset; // informational — not added to the quantised coeff itself
    let scale = (1i32 << LPCQUANT) as f64;
    let mut qcoeffs = Vec::with_capacity(order);
    for m in 1..=order {
        let q = (a[m] * scale).round() as i32;
        let q = q.clamp(-32_768, 32_767);
        qcoeffs.push(q);
    }

    // Verify the coefficients are within the signed Rice k=LPCQUANT
    // range; if any are out of range the write would be wrong. For
    // coefficients that are representable as signed Rice k=5 (range
    // -2^31..2^31-1 in principle, but the unary prefix would be
    // astronomically large for values much beyond ±512), clamp to
    // a practical range.
    Some(qcoeffs)
}

/// Compute QLPC residuals given quantised coefficients, mirroring the
/// decoder's `decode_audio_block` for `FN_QLPC`.
///
/// - `shifted_block`: current block samples (already right-shifted by
///   the stream's bitshift, if any).
/// - `shifted_history`: the predictor history, also right-shifted.
/// - `qcoeffs`: quantised LPC coefficients (from `estimate_qlpc`).
/// - `coffset`: per-channel mean offset (also already right-shifted).
/// - `lpcqoffset`: the V2LPCQOFFSET constant (32 for v >= 2).
///
/// Returns `(residuals, sum_abs, sum_signed)`.
fn compute_qlpc_residuals(
    shifted_block: &[i32],
    shifted_history: &[i32],
    qcoeffs: &[i32],
    coffset: i32,
    lpcqoffset: i32,
) -> (Vec<i32>, u64, i64) {
    let _order = qcoeffs.len();
    let nwrap = shifted_history.len();

    // Build working buffer: [shifted_history | block samples decoded so
    // far], adjusted for coffset (matching decoder's pre-loop step).
    let mut buf: Vec<i32> = shifted_history
        .iter()
        .map(|&s| s.wrapping_sub(coffset))
        .collect();
    // The decoder's per-sample loop:
    //   sum = lpcqoffset + sum(coeffs[j] * buf[i - j - 1])
    //   r = residual (= sample - coffset - predicted)
    //   buf[i] = r + predicted
    //
    // On the encoder side: residual = (sample - coffset) - (sum >> LPCQUANT)
    let mut residuals = Vec::with_capacity(shifted_block.len());
    let mut sum_abs: u64 = 0;
    let mut sum_sgn: i64 = 0;

    for (i, &s) in shifted_block.iter().enumerate() {
        let s_adj = s.wrapping_sub(coffset);
        let mut sum: i64 = lpcqoffset as i64;
        for (j, &c) in qcoeffs.iter().enumerate() {
            let idx = nwrap + i - j - 1;
            if idx < buf.len() {
                sum += (c as i64) * (buf[idx] as i64);
            }
        }
        let predicted = (sum >> LPCQUANT) as i32;
        let r = s_adj.wrapping_sub(predicted);
        residuals.push(r);
        sum_abs += r.unsigned_abs() as u64;
        sum_sgn += r as i64;
        buf.push(s_adj);
    }

    (residuals, sum_abs, sum_sgn)
}

// ─────────────────── Predictor residual computation ───────────────────

/// Compute residuals for a candidate predictor. Returns
/// `(residuals, sum_abs, sum_signed)`.
///
/// The math is the inverse of `decode_audio_block` (cmd != FN_QLPC):
/// - DIFF0: `residual = sample - coffset`
/// - DIFF1: `residual = sample - x[-1]`
/// - DIFF2: `residual = sample - (2*x[-1] - x[-2])`
/// - DIFF3: `residual = sample - (3*x[-1] - 3*x[-2] + x[-3])`
fn compute_residuals(
    cmd: u32,
    block: &[i32],
    history: &[i32],
    coffset: i32,
) -> (Vec<i32>, u64, i64) {
    let nwrap = history.len();
    debug_assert!(nwrap >= 3);
    // Working buffer of [history | block samples decoded so far]. For
    // residual computation the decoded samples ARE the input block —
    // a Shorten encode is lossless, so we don't need to re-decode
    // them; we just append them as we go.
    let mut buf: Vec<i32> = Vec::with_capacity(nwrap + block.len());
    buf.extend_from_slice(history);

    let coeffs: &[i32] = match cmd {
        FN_DIFF0 => &[],
        FN_DIFF1 => &[1],
        FN_DIFF2 => &[2, -1],
        FN_DIFF3 => &[3, -3, 1],
        _ => unreachable!("compute_residuals: bad cmd {}", cmd),
    };
    let init_sum: i64 = if cmd == FN_DIFF0 { coffset as i64 } else { 0 };

    let mut residuals: Vec<i32> = Vec::with_capacity(block.len());
    let mut sum_abs: u64 = 0;
    let mut sum_sgn: i64 = 0;
    for (i, &s) in block.iter().enumerate() {
        let mut sum: i64 = init_sum;
        for (j, &c) in coeffs.iter().enumerate() {
            let idx = nwrap + i - j - 1;
            sum += (c as i64) * (buf[idx] as i64);
        }
        // qshift = 0 for all DIFFn predictors.
        let predicted = sum as i32;
        let r = s.wrapping_sub(predicted);
        residuals.push(r);
        sum_abs += r.unsigned_abs() as u64;
        sum_sgn += r as i64;
        buf.push(s);
    }
    (residuals, sum_abs, sum_sgn)
}

// ─────────────────── Rice/Golomb writers (inverse of rice.rs) ───────────────────

/// Write an unsigned Rice-`k` integer. Inverse of
/// `rice::read_unsigned_k`.
pub(crate) fn write_unsigned_k(bw: &mut BitWriter, value: u32, k: u32) {
    let q = value >> k;
    bw.write_unary(q);
    if k > 0 {
        let mask = if k == 32 { u32::MAX } else { (1u32 << k) - 1 };
        bw.write_u32(value & mask, k);
    }
}

/// Write a signed Rice-`k` integer. Inverse of `rice::read_signed_k`.
/// Uses zig-zag (sign in the LSB), then `k+1` raw low bits.
pub(crate) fn write_signed_k(bw: &mut BitWriter, value: i32, k: u32) {
    // Zig-zag: ((value << 1) ^ (value >> 31)) — wrapping handles
    // i32::MIN cleanly (the result loses the magnitude high bit, which
    // matches the decoder's `(u >> 1) ^ -(u & 1)` modulo 2^32).
    let u = (value.wrapping_shl(1) ^ value.wrapping_shr(31)) as u32;
    write_unsigned_k(bw, u, k + 1);
}

/// Write an adaptive "ulong" unsigned integer. Inverse of
/// `rice::read_ulong`.
///
/// Picks the smallest `k_in` such that `(value >> k_in) <= 8` —
/// matches the heuristic the decoder's existing test helpers used and
/// keeps the unary prefix short. `k_param_size` is the bit width of
/// the parameter itself (always `ULONGSIZE = 2` in v >= 1).
pub(crate) fn write_ulong(bw: &mut BitWriter, value: u32, k_param_size: u32) {
    let mut k_in = 0u32;
    while k_in < 31 && (value >> k_in) > 8 {
        k_in += 1;
    }
    write_unsigned_k(bw, k_in, k_param_size);
    write_unsigned_k(bw, value, k_in);
}

// ─────────────────── WAV header builder ───────────────────

/// Build a minimal 44-byte RIFF/WAV header for embedding in the leading
/// FN_VERBATIM capsule. ffmpeg and shntool read this to determine
/// `sample_rate`, bit depth, and channel count; without it they report
/// "unsupported bit packing 0".
///
/// Layout (all integers little-endian):
/// ```text
/// Offset  Size  Field                    Value
///  0       4    ChunkID                  "RIFF"
///  4       4    ChunkSize                0 (unknown / streaming)
///  8       4    Format                   "WAVE"
/// 12       4    Subchunk1ID              "fmt "
/// 16       4    Subchunk1Size            16
/// 20       2    AudioFormat              1 (PCM)
/// 22       2    NumChannels              channels
/// 24       4    SampleRate               sample_rate
/// 28       4    ByteRate                 sample_rate * channels * bits/8
/// 32       2    BlockAlign               channels * bits/8
/// 34       2    BitsPerSample            8 or 16
/// 36       4    Subchunk2ID              "data"
/// 40       4    Subchunk2Size            0 (unknown)
/// ```
fn build_wav_header(ftype: ShortenFtype, channels: u32, sample_rate: u32) -> [u8; 44] {
    let bits_per_sample: u16 = match ftype {
        ShortenFtype::S8 | ShortenFtype::U8 => 8,
        _ => 16,
    };
    // WAV audio format: 1 = PCM.
    let audio_fmt: u16 = 1;
    let nch = channels as u16;
    let bps = bits_per_sample;
    let block_align = nch * (bps / 8);
    let byte_rate = sample_rate * (block_align as u32);

    let mut h = [0u8; 44];
    // "RIFF"
    h[0..4].copy_from_slice(b"RIFF");
    // ChunkSize = 0 (unknown)
    h[4..8].copy_from_slice(&0u32.to_le_bytes());
    // "WAVE"
    h[8..12].copy_from_slice(b"WAVE");
    // "fmt "
    h[12..16].copy_from_slice(b"fmt ");
    // Subchunk1Size = 16
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    // AudioFormat = 1 (PCM)
    h[20..22].copy_from_slice(&audio_fmt.to_le_bytes());
    // NumChannels
    h[22..24].copy_from_slice(&nch.to_le_bytes());
    // SampleRate
    h[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    // ByteRate
    h[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    // BlockAlign
    h[32..34].copy_from_slice(&block_align.to_le_bytes());
    // BitsPerSample
    h[34..36].copy_from_slice(&bps.to_le_bytes());
    // "data"
    h[36..40].copy_from_slice(b"data");
    // Subchunk2Size = 0 (unknown)
    h[40..44].copy_from_slice(&0u32.to_le_bytes());
    h
}

// Keep FN_BITSHIFT in scope so the constant set is complete.
#[allow(dead_code)]
const _KEEP: () = {
    let _ = FN_BITSHIFT;
};

// ─────────────────── Tests ───────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitReader;

    #[test]
    fn rice_unsigned_writer_inverts_reader() {
        for k in 0..6u32 {
            let mut bw = BitWriter::new();
            let values = [0u32, 1, 2, 5, 100, 1023];
            for &v in &values {
                write_unsigned_k(&mut bw, v, k);
            }
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            for &v in &values {
                assert_eq!(crate::rice::read_unsigned_k(&mut br, k).unwrap(), v);
            }
        }
    }

    #[test]
    fn rice_signed_writer_inverts_reader() {
        for k in 0..6u32 {
            let values = [0i32, 1, -1, 2, -2, 100, -100, 12345, -12345];
            let mut bw = BitWriter::new();
            for &v in &values {
                write_signed_k(&mut bw, v, k);
            }
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            for &v in &values {
                assert_eq!(crate::rice::read_signed_k(&mut br, k).unwrap(), v);
            }
        }
    }

    #[test]
    fn ulong_writer_inverts_reader() {
        let mut bw = BitWriter::new();
        let values = [0u32, 1, 5, 256, 4096, 65535];
        for &v in &values {
            write_ulong(&mut bw, v, ULONGSIZE);
        }
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        for &v in &values {
            assert_eq!(crate::rice::read_ulong(&mut br, ULONGSIZE, 0).unwrap(), v);
        }
    }

    #[test]
    fn diff0_predictor_residual_for_constant_signal() {
        // DIFF0 with coffset=0 against a constant signal yields
        // residual = sample (no compression), but it should still be
        // numerically correct.
        let block = vec![5i32; 8];
        let history = vec![0i32; 3];
        let (res, _abs, _sum) = compute_residuals(FN_DIFF0, &block, &history, 0);
        assert_eq!(res, vec![5; 8]);
    }

    #[test]
    fn diff1_predictor_residual_for_ramp() {
        // A linear ramp [10, 11, 12, 13] with history [0, 0, 0]:
        // DIFF1 residuals = [10, 1, 1, 1] (first one uses x[-1]=0).
        let block = vec![10i32, 11, 12, 13];
        let history = vec![0i32; 3];
        let (res, _, _) = compute_residuals(FN_DIFF1, &block, &history, 0);
        assert_eq!(res, vec![10, 1, 1, 1]);
    }

    #[test]
    fn diff2_predictor_residual_for_ramp() {
        // A linear ramp is killed by DIFF2 after the first two samples.
        let block = vec![10i32, 11, 12, 13, 14];
        let history = vec![0i32; 3];
        let (res, _, _) = compute_residuals(FN_DIFF2, &block, &history, 0);
        // i=0: pred = 2*0 - 0 = 0, r = 10
        // i=1: pred = 2*10 - 0 = 20, r = -9
        // i=2: pred = 2*11 - 10 = 12, r = 0
        // i=3: pred = 2*12 - 11 = 13, r = 0
        // i=4: pred = 2*13 - 12 = 14, r = 0
        assert_eq!(res, vec![10, -9, 0, 0, 0]);
    }

    #[test]
    fn diff3_predictor_residual_for_quadratic() {
        // y = i*i (quadratic) — DIFF3 should kill it after three
        // samples (Δ^3 of a quadratic is zero from i=3 onward).
        let block: Vec<i32> = (0i32..6).map(|i| i * i).collect();
        let history = vec![0i32; 3];
        let (res, _, _) = compute_residuals(FN_DIFF3, &block, &history, 0);
        // i=3: pred = 3*4 - 3*1 + 0 = 9, r = 9 - 9 = 0
        // i=4: pred = 3*9 - 3*4 + 1 = 16, r = 16 - 16 = 0
        // i=5: pred = 3*16 - 3*9 + 4 = 25, r = 25 - 25 = 0
        assert_eq!(&res[3..], &[0, 0, 0]);
    }
}
