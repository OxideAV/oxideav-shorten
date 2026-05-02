//! Shorten encoder — round-1 (DIFFn predictors only, no QLPC).
//!
//! Mirrors [`crate::decoder`] exactly: every primitive emitted here is
//! the bit-for-bit inverse of a primitive consumed there. The encoder
//! is deliberately restricted to the fixed forward-difference
//! predictors (DIFF0/1/2/3) plus the FN_ZERO silence shortcut for
//! round 1; QLPC, BITSHIFT, and BLOCKSIZE-mid-stream are documented as
//! round-2 follow-ups in the crate CHANGELOG.
//!
//! # Round-1 design
//!
//! - Bitwriter: [`oxideav_core::bits::BitWriter`] (MSB-first, matches
//!   the decoder's `BitReader`).
//! - Header: magic `'ajkg'` + version `2` + 6 ulong fields
//!   (`internal_ftype`, `channels`, `blocksize`, `maxnlpc=0`, `nmean`,
//!   `skip_bytes=0`) + a single leading FN_VERBATIM capsule.
//! - Per block: pick the cheapest of DIFF0..3 by minimum sum of
//!   `|residual|`; if every residual is zero, emit FN_ZERO instead.
//!   Per-block Rice-k is the closed-form `floor(log2(mean(|r|)))`.
//! - Per channel: matches the decoder's wrap-history (`nwrap = 3`
//!   when `maxnlpc = 0`) and the `nmean`-deep mean FIFO with v >= 2
//!   rounding bias. Channels are emitted round-robin.
//!
//! # Limitations (round-2 follow-ups)
//!
//! - No QLPC (would need Levinson-Durbin + coefficient quantisation).
//! - No BITSHIFT (would let 24-bit-in-32 streams compress better).
//! - No mid-stream FN_BLOCKSIZE (encoder uses one fixed blocksize for
//!   the whole stream).
//! - No streaming / multi-packet encode (the encoder produces one
//!   `.shn` byte stream per call).

use oxideav_core::bits::BitWriter;
use oxideav_core::{Error, Result};

// ─────────────────── Function codes (mirror decoder.rs) ───────────────────

const FN_DIFF0: u32 = 0;
const FN_DIFF1: u32 = 1;
const FN_DIFF2: u32 = 2;
const FN_DIFF3: u32 = 3;
const FN_QUIT: u32 = 4;
const FN_QLPC: u32 = 7;
const FN_ZERO: u32 = 8;
const FN_VERBATIM: u32 = 9;

// k_param_size constants from the decoder.
const FNSIZE: u32 = 2;
const ULONGSIZE: u32 = 2;
const ENERGYSIZE: u32 = 3;
const VERBATIM_CKSIZE_SIZE: u32 = 5;
const VERBATIM_BYTE_SIZE: u32 = 8;

const VERSION: u8 = 2;
const MAGIC: &[u8; 4] = b"ajkg";

const MAX_CHANNELS: usize = 8;
const MIN_BLOCKSIZE: u32 = 1;
const MAX_BLOCKSIZE: u32 = 65_535;

// Canonical RIFF/AIFF header size used by the leading FN_VERBATIM. The
// decoder enforces `len >= 44`, so we use exactly 44 zero bytes — the
// caller's downstream demuxer is expected to ignore the placeholder
// (round-2 work would synthesise a real WAV header here).
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
    /// Mean-FIFO depth. The decoder requires `nmean <= 1024`. A common
    /// choice in real-world v2 streams is `0` or `4`. Round-1 default
    /// is `0` (no running-mean compensation — the per-channel coffset
    /// stays at the type's `initial_offset()`).
    pub nmean: u32,
}

impl ShortenEncoderConfig {
    pub fn new(ftype: ShortenFtype, channels: u32, blocksize: u32) -> Self {
        Self {
            ftype,
            channels,
            blocksize,
            nmean: 0,
        }
    }

    pub fn with_nmean(mut self, nmean: u32) -> Self {
        self.nmean = nmean;
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
        Ok(())
    }
}

/// Per-block introspection record (one entry per emitted DIFF/ZERO
/// audio block, in stream order). Useful for tests that want to assert
/// the predictor selector picked a particular DIFFn for a known-shape
/// signal, without parsing the bitstream back.
#[derive(Debug, Clone, Copy)]
pub struct BlockInfo {
    pub channel: u32,
    pub cmd: u32, // FN_DIFFn or FN_ZERO
    pub k: u32,
    pub residual_abs_sum: u64,
}

/// One-shot Shorten encoder. See module docs for the round-1 scope.
///
/// The encoder is consumed by [`Self::encode`] (so re-using a single
/// instance for many independent streams is not supported — make a
/// fresh one per stream).
pub struct ShortenEncoder {
    cfg: ShortenEncoderConfig,
    debug_blocks: Vec<BlockInfo>,
}

impl ShortenEncoder {
    pub fn new(cfg: ShortenEncoderConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            debug_blocks: Vec::new(),
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

        write_ulong(&mut bw, self.cfg.ftype.to_internal(), ULONGSIZE);
        write_ulong(&mut bw, self.cfg.channels, ULONGSIZE);
        write_ulong(&mut bw, self.cfg.blocksize, ULONGSIZE);
        write_ulong(&mut bw, 0, ULONGSIZE); // maxnlpc = 0
        write_ulong(&mut bw, self.cfg.nmean, ULONGSIZE);
        write_ulong(&mut bw, 0, ULONGSIZE); // skip_bytes = 0

        // ─── Leading FN_VERBATIM (placeholder, 44 zero bytes) ───
        write_unsigned_k(&mut bw, FN_VERBATIM, FNSIZE);
        write_unsigned_k(&mut bw, VERBATIM_LEN, VERBATIM_CKSIZE_SIZE);
        for _ in 0..VERBATIM_LEN {
            write_unsigned_k(&mut bw, 0, VERBATIM_BYTE_SIZE);
        }

        // ─── Per-channel state (mirrors decoder) ───────────────
        let nwrap = 3usize; // maxnlpc = 0 ⇒ nwrap = max(3, 0) = 3
        let init_offset = self.cfg.ftype.initial_offset();
        let nmean = self.cfg.nmean as usize;
        let mut chan_state: Vec<ChannelState> = (0..nch)
            .map(|_| ChannelState::new(nwrap, nmean, init_offset))
            .collect();

        // ─── Audio blocks (round-robin per channel) ────────────
        for blk in 0..nblocks {
            for ch in 0..nch {
                let start = blk * bs;
                let block = &chan_samples[ch][start..start + bs];
                self.emit_block(&mut bw, block, ch as u32, &mut chan_state[ch]);
            }
        }

        // ─── Quit ──────────────────────────────────────────────
        write_unsigned_k(&mut bw, FN_QUIT, FNSIZE);

        Ok(bw.into_bytes())
    }

    fn emit_block(&mut self, bw: &mut BitWriter, block: &[i32], ch: u32, state: &mut ChannelState) {
        let bs = block.len();
        // Compute coffset (matches the decoder; bitshift is 0 in
        // round 1 so no >>bitshift fixup applies).
        let coffset = state.coffset(VERSION);

        // Try each predictor; pick the one with the smallest
        // sum-of-absolute-residuals.
        let mut best_cmd = FN_DIFF0;
        let mut best_res: Vec<i32> = Vec::new();
        let mut best_abs: u64 = u64::MAX;
        let mut best_sum: i64 = 0;

        for cmd in [FN_DIFF0, FN_DIFF1, FN_DIFF2, FN_DIFF3] {
            let (res, sum_abs, sum_sgn) = compute_residuals(cmd, block, &state.history, coffset);
            if sum_abs < best_abs {
                best_abs = sum_abs;
                best_cmd = cmd;
                best_res = res;
                best_sum = sum_sgn;
            }
        }

        // FN_ZERO shortcut: if all residuals are zero (regardless of
        // predictor), emit FN_ZERO instead — saves the energy/k field
        // and the per-sample residual codes.
        let all_zero_block = block.iter().all(|&s| s == 0);
        if all_zero_block && coffset == 0 {
            // Decoder's ZERO path produces blocksize zeros and feeds
            // them to finish_block. Mirror exactly.
            write_unsigned_k(bw, FN_ZERO, FNSIZE);
            self.debug_blocks.push(BlockInfo {
                channel: ch,
                cmd: FN_ZERO,
                k: 0,
                residual_abs_sum: 0,
            });
            let zeros = vec![0i32; bs];
            state.finish_block(&zeros, VERSION);
            return;
        }

        // Pick Rice-k from the mean of |residual|.
        // Closed form: k = max(0, floor(log2(max(1, mean_abs)))).
        // mean_abs uses signed-Rice's natural pop count (we take 1
        // raw "magnitude" bit per residual implicitly via the +1).
        let mean_abs = if bs == 0 { 0 } else { best_abs / bs as u64 };
        let k = if mean_abs <= 1 {
            0
        } else {
            // floor(log2(mean_abs))
            63 - mean_abs.leading_zeros()
        };
        let k = k.min(30);

        let cmd = best_cmd;
        let _ = best_sum; // currently unused; kept for round-2 mean accuracy work

        // Emit FN_DIFFn + energy/k + signed-Rice residuals.
        write_unsigned_k(bw, cmd, FNSIZE);
        write_unsigned_k(bw, k, ENERGYSIZE);
        for &r in &best_res {
            write_signed_k(bw, r, k);
        }

        self.debug_blocks.push(BlockInfo {
            channel: ch,
            cmd,
            k,
            residual_abs_sum: best_abs,
        });

        // Update per-channel state — the decoder runs `finish_block`
        // on the *decoded* samples, which round-trip back to the
        // original `block`. So we feed `block` directly.
        state.finish_block(block, VERSION);
    }
}

// ─────────────────── Per-channel state (mirrors decoder) ───────────────────

struct ChannelState {
    history: Vec<i32>,
    offsets: Vec<i32>,
    nmean: usize,
}

impl ChannelState {
    fn new(nwrap: usize, nmean: usize, init_offset: i32) -> Self {
        Self {
            history: vec![0; nwrap],
            offsets: vec![init_offset; nmean.max(1)],
            nmean,
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

    /// Mirrors the decoder's `finish_block` (with bitshift = 0).
    fn finish_block(&mut self, decoded: &[i32], version: u8) {
        let blocksize = decoded.len();
        let mut sum: i64 = 0;
        for &s in decoded {
            sum += s as i64;
        }
        let mean = if blocksize == 0 {
            0
        } else if version >= 2 {
            ((sum + (blocksize as i64) / 2) / blocksize as i64) as i32
        } else {
            (sum / blocksize as i64) as i32
        };
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

// Suppress dead_code on FN_QLPC — kept around because the constant set
// is shared with the decoder and round-2 will use it.
#[allow(dead_code)]
const _: u32 = FN_QLPC;

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
