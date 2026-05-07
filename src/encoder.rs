//! Test-only minimal Shorten encoder.
//!
//! Used purely to construct valid round-1 bit streams for self-
//! roundtrip tests. The encoder is a thin one (no predictor search,
//! no rate-distortion optimisation) — it picks a single per-block
//! predictor (caller-supplied or DIFF1) and writes the residuals at
//! a width chosen to fit the largest residual in the block.
//!
//! This module is `#[cfg(test)]`-gated; it is *not* part of the
//! crate's public API for round 1 (decode-only). A production
//! encoder belongs in a later round.

#![cfg(test)]

use crate::header::{Filetype, MAGIC};
use crate::varint::{
    signed_to_unsigned, BITSHIFTSIZE, ENERGYSIZE, FNSIZE, LPCQSIZE, LPCQUANT, ULONGSIZE,
    VERBATIM_BYTE_SIZE, VERBATIM_CHUNK_SIZE,
};

/// Bit writer — packs MSB-first per byte.
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    /// Current byte being filled (high bits are written first).
    cur: u8,
    /// Number of bits already deposited in `cur` (0..8).
    cur_bits: u8,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            bytes: Vec::new(),
            cur: 0,
            cur_bits: 0,
        }
    }

    pub fn write_bit(&mut self, b: u32) {
        self.cur = (self.cur << 1) | ((b & 1) as u8);
        self.cur_bits += 1;
        if self.cur_bits == 8 {
            self.bytes.push(self.cur);
            self.cur = 0;
            self.cur_bits = 0;
        }
    }

    pub fn write_bits(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.write_bit((value >> i) & 1);
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        if self.cur_bits > 0 {
            // Zero-pad to the next byte boundary.
            self.cur <<= 8 - self.cur_bits;
            self.bytes.push(self.cur);
        }
        self.bytes
    }

    /// Write a `uvar(n)` value: zero-prefix + terminator + n-bit
    /// mantissa.
    pub fn write_uvar(&mut self, value: u32, n: u32) {
        let high = value >> n;
        let mantissa = if n == 0 { 0 } else { value & ((1u32 << n) - 1) };
        for _ in 0..high {
            self.write_bit(0);
        }
        self.write_bit(1);
        self.write_bits(mantissa, n);
    }

    pub fn write_svar(&mut self, s: i32, n: u32) {
        let u = signed_to_unsigned(s);
        self.write_uvar(u, n);
    }

    /// Write a `ulong()` value with the smallest width that fits it.
    pub fn write_ulong(&mut self, value: u32) {
        // Smallest `w` such that `value < 2^w * (some prefix budget)`.
        // For simplicity we pick the smallest w such that
        // `value >> w` is small (uses few prefix zeros). A simple
        // heuristic: w = max(0, 32 - leading_zeros(value | 1) - 1)
        // and clamp to 0..=16. Test scaffolding only.
        let mut w: u32 = 0;
        if value > 0 {
            // Choose w such that `value >> w` is small (≤ 4 zeros of prefix).
            let bits_needed = 32 - value.leading_zeros();
            w = bits_needed.saturating_sub(2);
        }
        if w > 16 {
            w = 16;
        }
        self.write_uvar(w, ULONGSIZE);
        self.write_uvar(value, w);
    }
}

/// Encode a tiny `.shn` stream from i32-lane PCM samples.
///
/// `samples` is interleaved (`c0_s0, c1_s0, c0_s1, ...`).  The encoder
/// uses `BLOCK_FN_VERBATIM` for an optional prefix, then round-robins
/// `predictor` blocks across the channels until samples are exhausted.
/// A final partial block is emitted with `BLOCK_FN_BLOCKSIZE` if
/// needed.  The encoder closes with `BLOCK_FN_QUIT`.
///
/// This helper exists for **tests only**; it does not exercise the
/// running-mean estimator, the LPC predictor, or the bit-shift
/// command. Round-2 work can grow it into a production encoder.
pub fn encode_minimal(
    filetype: Filetype,
    channels: u16,
    blocksize: u32,
    samples: &[i32],
    predictor: u32,
    verbatim: &[u8],
) -> Vec<u8> {
    assert!(
        samples.len() % channels as usize == 0,
        "samples must be channel-aligned"
    );
    let nch = channels as usize;
    let total_per_ch = samples.len() / nch;

    // 5-byte byte-aligned prefix.
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.push(2u8); // version

    let mut bw = BitWriter::new();
    // Header fields: filetype, channels, blocksize, max_lpc_order=0,
    // mean_blocks=0 (disable so the decoder can ignore the running
    // mean for these tests), skip_bytes=0.
    bw.write_ulong(filetype.to_code());
    bw.write_ulong(channels as u32);
    bw.write_ulong(blocksize);
    bw.write_ulong(0); // max_lpc_order
    bw.write_ulong(0); // mean_blocks
    bw.write_ulong(0); // skip_bytes

    // VERBATIM chunk if any.
    if !verbatim.is_empty() {
        assert!(
            verbatim.len() < (1u32 << 24) as usize,
            "verbatim too long for the test encoder"
        );
        bw.write_uvar(crate::decoder::fn_code::VERBATIM, FNSIZE);
        bw.write_uvar(verbatim.len() as u32, VERBATIM_CHUNK_SIZE);
        for &b in verbatim {
            bw.write_uvar(b as u32, VERBATIM_BYTE_SIZE);
        }
    }

    // De-interleave samples for per-channel predictor application.
    let mut per_ch: Vec<Vec<i32>> = (0..nch).map(|_| Vec::with_capacity(total_per_ch)).collect();
    for i in 0..total_per_ch {
        for ch in 0..nch {
            per_ch[ch].push(samples[i * nch + ch]);
        }
    }

    // Round-robin emit blocks of `blocksize` samples per channel.
    let mut history: Vec<Vec<i32>> = (0..nch).map(|_| vec![0i32; 3]).collect();
    let mut written: Vec<usize> = vec![0; nch];

    let mut current_bs = blocksize as usize;
    let mut emitted_partial = false;

    'outer: loop {
        for ch in 0..nch {
            let remaining = total_per_ch - written[ch];
            if remaining == 0 {
                break 'outer;
            }
            let take = remaining.min(current_bs);
            if take < current_bs && !emitted_partial {
                // Emit a BLOCKSIZE override before this block.
                bw.write_uvar(crate::decoder::fn_code::BLOCKSIZE, FNSIZE);
                bw.write_ulong(take as u32);
                current_bs = take;
                emitted_partial = true;
            }
            let block = &per_ch[ch][written[ch]..written[ch] + take];
            // Compute residuals under the chosen predictor.
            let residuals = compute_residuals(predictor, &history[ch], block);
            // Pick the smallest mantissa width that fits all residuals.
            let max_abs = residuals
                .iter()
                .map(|r| r.unsigned_abs())
                .max()
                .unwrap_or(0);
            let needed_unsigned = if max_abs == 0 {
                1
            } else {
                // Largest fold(u)= 2*max_abs (or 2*max_abs - 1 for negatives) — round up.
                let folded: u32 = max_abs.saturating_mul(2) | 1;
                32 - folded.leading_zeros()
            };
            // residual width = field + 1 → field = needed - 1, but
            // field must fit in uvar(ENERGYSIZE)+ prefix-zeros so any
            // value works.  needed_unsigned is the number of bits the
            // largest folded magnitude takes.  We want a residual
            // mantissa width that keeps the prefix small: aim for
            // about `bits_for_folded - 1` so half the values fit in
            // the mantissa.  For the round-1 tests we use the simple
            // width = max(1, ceil(log2(folded+1))) heuristic; but more
            // importantly we just pick a width such that everything
            // fits, so the prefix stays bounded.
            let width: u32 = needed_unsigned.max(1);
            // The energy field is width - 1 (encoded as uvar(3)).
            let field = width - 1;
            bw.write_uvar(predictor, FNSIZE);
            bw.write_uvar(field, ENERGYSIZE);
            for &r in &residuals {
                bw.write_svar(r, width);
            }
            // Update the per-channel history for the next block.
            update_history(&mut history[ch], block);
            written[ch] += take;
        }
    }

    // QUIT.
    bw.write_uvar(crate::decoder::fn_code::QUIT, FNSIZE);

    out.extend_from_slice(&bw.finish());
    out
}

/// Compute predictor residuals for a single channel block.
fn compute_residuals(predictor: u32, history: &[i32], block: &[i32]) -> Vec<i32> {
    // history[0] = s(t-1), history[1] = s(t-2), history[2] = s(t-3).
    let mut s1 = history[0];
    let mut s2 = history[1];
    let mut s3 = history[2];
    let mut out = Vec::with_capacity(block.len());
    for &s in block {
        let pred = match predictor {
            0 => 0i32, // DIFF0
            1 => s1,
            2 => s1.wrapping_mul(2).wrapping_sub(s2),
            3 => s1
                .wrapping_mul(3)
                .wrapping_sub(s2.wrapping_mul(3))
                .wrapping_add(s3),
            _ => panic!("test encoder only supports DIFF0..3"),
        };
        out.push(s.wrapping_sub(pred));
        s3 = s2;
        s2 = s1;
        s1 = s;
    }
    out
}

fn update_history(history: &mut [i32], block: &[i32]) {
    let bs = block.len();
    if bs >= history.len() {
        for i in 0..history.len() {
            history[i] = block[bs - 1 - i];
        }
    } else {
        // Shift older history right by `bs`.
        let hist_len = history.len();
        let mut new_hist = vec![0i32; hist_len];
        for (i, slot) in new_hist.iter_mut().enumerate().take(bs) {
            *slot = block[bs - 1 - i];
        }
        new_hist[bs..hist_len].copy_from_slice(&history[..(hist_len - bs)]);
        history.copy_from_slice(&new_hist);
    }
}

/// Build a hand-crafted bit stream that exercises `BLOCK_FN_BITSHIFT`
/// and `BLOCK_FN_ZERO` in addition to a trailing DIFF1 block. Used by
/// the round-1 tests to exercise the housekeeping paths without
/// growing [`encode_minimal`] beyond its scaffold scope.
pub fn encode_with_bitshift_and_zero(
    filetype: Filetype,
    channels: u16,
    blocksize: u32,
    bshift: u32,
    zero_blocks: u32,
    diff_block: &[i32],
) -> Vec<u8> {
    assert_eq!(
        diff_block.len(),
        blocksize as usize,
        "diff_block must equal blocksize"
    );
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.push(2);

    let mut bw = BitWriter::new();
    bw.write_ulong(filetype.to_code());
    bw.write_ulong(channels as u32);
    bw.write_ulong(blocksize);
    bw.write_ulong(0);
    bw.write_ulong(0); // mean_blocks
    bw.write_ulong(0); // skip_bytes

    // Optional bit-shift command.
    if bshift > 0 {
        bw.write_uvar(crate::decoder::fn_code::BITSHIFT, FNSIZE);
        bw.write_uvar(bshift, BITSHIFTSIZE);
    }

    let nch = channels as usize;
    // Emit `zero_blocks` total ZERO commands (round-robin).
    for _ in 0..zero_blocks {
        for _ in 0..nch {
            bw.write_uvar(crate::decoder::fn_code::ZERO, FNSIZE);
        }
    }

    // Emit one DIFF1 block per channel using the same residual stream
    // (test scaffolding only).
    for _ in 0..nch {
        let max_abs = diff_block
            .iter()
            .map(|r| r.unsigned_abs())
            .max()
            .unwrap_or(0);
        let folded: u32 = max_abs.saturating_mul(2) | 1;
        let needed = 32 - folded.leading_zeros();
        let width = needed.max(1);
        let field = width - 1;
        bw.write_uvar(crate::decoder::fn_code::DIFF1, FNSIZE);
        bw.write_uvar(field, ENERGYSIZE);
        // residuals = block samples (zero history).
        let mut prev: i32 = 0;
        for &s in diff_block {
            let r = s.wrapping_sub(prev);
            bw.write_svar(r, width);
            prev = s;
        }
    }

    bw.write_uvar(crate::decoder::fn_code::QUIT, FNSIZE);
    out.extend_from_slice(&bw.finish());
    out
}

/// Build a hand-crafted stream that exercises `BLOCK_FN_QLPC` with an
/// LPC order >= 1.
pub fn encode_qlpc_block(
    filetype: Filetype,
    channels: u16,
    blocksize: u32,
    max_lpc_order: u32,
    coefs: &[i32],
    samples_per_channel: &[Vec<i32>],
) -> Vec<u8> {
    assert!(coefs.len() as u32 <= max_lpc_order);
    assert_eq!(samples_per_channel.len(), channels as usize);

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.push(2);

    let mut bw = BitWriter::new();
    bw.write_ulong(filetype.to_code());
    bw.write_ulong(channels as u32);
    bw.write_ulong(blocksize);
    bw.write_ulong(max_lpc_order);
    bw.write_ulong(0); // mean_blocks
    bw.write_ulong(0); // skip_bytes

    let nch = channels as usize;
    let order = coefs.len();
    // Emit one QLPC block per channel.
    for block in samples_per_channel.iter().take(nch) {
        assert_eq!(block.len(), blocksize as usize);
        // Compute residuals: r(t) = s(t) - sum coefs[i] * s(t-1-i).
        let mut hist = vec![0i32; order];
        let mut residuals = Vec::with_capacity(block.len());
        for &s in block {
            let mut pred: i64 = 0;
            for i in 0..order {
                pred = pred.wrapping_add((coefs[i] as i64).wrapping_mul(hist[i] as i64));
            }
            let r = s.wrapping_sub(pred as i32);
            residuals.push(r);
            for i in (1..order).rev() {
                hist[i] = hist[i - 1];
            }
            if order > 0 {
                hist[0] = s;
            }
        }
        let max_abs = residuals
            .iter()
            .map(|r| r.unsigned_abs())
            .max()
            .unwrap_or(0);
        let folded: u32 = max_abs.saturating_mul(2) | 1;
        let needed = 32 - folded.leading_zeros();
        let width = needed.max(1);
        let field = width - 1;
        bw.write_uvar(crate::decoder::fn_code::QLPC, FNSIZE);
        bw.write_uvar(order as u32, LPCQSIZE);
        for &c in coefs {
            bw.write_svar(c, LPCQUANT);
        }
        bw.write_uvar(field, ENERGYSIZE);
        for &r in &residuals {
            bw.write_svar(r, width);
        }
    }

    bw.write_uvar(crate::decoder::fn_code::QUIT, FNSIZE);
    out.extend_from_slice(&bw.finish());
    out
}
