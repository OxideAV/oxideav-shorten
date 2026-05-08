//! Per-block decoder.
//!
//! Consumes the bit stream from the end of the parameter block (see
//! [`crate::header::parse_header`]) through to `BLOCK_FN_QUIT`. Each
//! per-block command is a `uvar(FNSIZE)` function code optionally
//! followed by a command-specific parameter list and payload, per
//! `spec/03-block-and-predictor.md` §3.
//!
//! The decoder maintains:
//!
//! * one [`ChannelState`] per channel, holding the per-channel
//!   sample-history carry, the running-mean estimator window, and a
//!   per-channel block-mean accumulator (reset at block start);
//! * a `current_block_size`, defaulting to `H_blocksize`, mutated by
//!   `BLOCK_FN_BLOCKSIZE`;
//! * a per-stream `bshift` count, mutated by `BLOCK_FN_BITSHIFT`;
//! * a verbatim-prefix byte buffer that captures `BLOCK_FN_VERBATIM`
//!   payloads;
//! * an implicit channel cursor that advances modulo
//!   `H_channels` after every sample-producing command.

use crate::bitreader::BitReader;
use crate::bitstream64::Bitstream64;
use crate::header::StreamHeader;
use crate::varint::{
    read_svar, read_ulong, read_uvar, BITSHIFTSIZE, ENERGYSIZE, FNSIZE, LPCQSIZE, LPCQUANT,
    RESIDUAL_WIDTH_CAP, VERBATIM_BYTE_SIZE, VERBATIM_CHUNK_SIZE,
};
use crate::{Error, Result};

/// Cap on the number of zero bits permitted in any `uvar` prefix —
/// matches the private `UVAR_MAX_ZEROS` in `varint.rs`. Duplicated
/// here to keep the [`Bitstream64`] residual-batch path standalone.
const RESIDUAL_UVAR_MAX_ZEROS: u32 = 64;

/// Function-code values pinned in `spec/04-function-code-resolution.md`.
pub(crate) mod fn_code {
    pub const DIFF0: u32 = 0;
    pub const DIFF1: u32 = 1;
    pub const DIFF2: u32 = 2;
    pub const DIFF3: u32 = 3;
    pub const QUIT: u32 = 4;
    pub const BLOCKSIZE: u32 = 5;
    pub const BITSHIFT: u32 = 6;
    pub const QLPC: u32 = 7;
    pub const ZERO: u32 = 8;
    pub const VERBATIM: u32 = 9;
}

/// Decoded stream — verbatim-prefix bytes plus per-channel samples.
#[derive(Debug, Clone)]
pub struct DecodedStream {
    /// The parsed stream header.
    pub header: StreamHeader,
    /// Concatenated `BLOCK_FN_VERBATIM` payload bytes, in encode
    /// order. For RIFF / WAVE / AIFF fixtures this is the
    /// reconstructed container preamble.
    pub verbatim_prefix: Vec<u8>,
    /// Decoded PCM samples, **interleaved** in channel-major order:
    /// `c0_s0, c1_s0, c0_s1, c1_s1, ...` for stereo, mono is one
    /// sample per index. Values are post-`bshift`-applied i32 lanes.
    pub samples: Vec<i32>,
    /// Total number of samples per channel.
    pub samples_per_channel: usize,
    /// Final per-stream `bshift` value at end-of-stream. Useful for
    /// callers diagnosing lossy streams.
    pub final_bshift: u32,
}

/// Per-channel decoder state.
struct ChannelState {
    /// Sample-history carry buffer. `carry[0]` is `s(t-1)`,
    /// `carry[1]` is `s(t-2)`, etc.; length = `max(3, max_lpc_order)`.
    /// Initialised to zero; refreshed on every sample-producing block.
    carry: Vec<i32>,
    /// Running-mean estimator window — `mean_blocks` per-block means.
    /// All zeros at stream start; oldest slot is evicted when a new
    /// per-block mean is appended (`spec/05` §2.5).
    mean_buf: Vec<i32>,
}

impl ChannelState {
    fn new(carry_len: usize, mean_blocks: u32) -> Self {
        Self {
            carry: vec![0i32; carry_len],
            mean_buf: vec![0i32; mean_blocks as usize],
        }
    }

    /// Running-mean estimator value at block start
    /// (`spec/05` §2.5 step 3).
    fn running_mean(&self, mean_blocks: u32) -> i32 {
        if mean_blocks == 0 || self.mean_buf.is_empty() {
            return 0;
        }
        let divisor = mean_blocks as i64;
        let sum: i64 = self.mean_buf.iter().map(|&v| v as i64).sum();
        // Always-positive bias `+divisor/2` regardless of sign of
        // sum, then C-style truncation toward zero.
        let bias = divisor / 2;
        let numerator = sum + bias;
        // Rust's integer division truncates toward zero already.
        (numerator / divisor) as i32
    }

    /// Append a per-block mean to the running-mean window
    /// (`spec/05` §2.5 step 4): evict the oldest slot, push the new.
    fn push_block_mean(&mut self, mu_blk: i32) {
        if self.mean_buf.is_empty() {
            return;
        }
        // Sliding window: shift left by one and write at the tail.
        let n = self.mean_buf.len();
        for i in 0..n - 1 {
            self.mean_buf[i] = self.mean_buf[i + 1];
        }
        self.mean_buf[n - 1] = mu_blk;
    }

    /// Update the carry buffer after producing a block of `bs` samples
    /// (`spec/05` §1, step 3). For `bs >= carry_len` the carry is
    /// fully refreshed from the new block; for shorter blocks the
    /// older history is retained.
    fn update_carry(&mut self, block: &[i32]) {
        let carry_len = self.carry.len();
        let bs = block.len();
        if carry_len == 0 {
            return;
        }
        if bs >= carry_len {
            for i in 0..carry_len {
                self.carry[i] = block[bs - 1 - i];
            }
        } else {
            // Shift older history right by `bs`, then write the new
            // block at the most-recent slots.
            // First copy old carry into a temporary so we don't
            // clobber it before reading.
            let mut new_carry = vec![0i32; carry_len];
            for (i, slot) in new_carry.iter_mut().enumerate().take(bs) {
                *slot = block[bs - 1 - i];
            }
            new_carry[bs..carry_len].copy_from_slice(&self.carry[..(carry_len - bs)]);
            self.carry.copy_from_slice(&new_carry);
        }
    }
}

/// Advance the reader by `target_bit` bits without exceeding the
/// `read_bits(<= 32)` bound on a single call.
fn skip_to_bit(br: &mut BitReader<'_>, target_bit: usize) -> Result<()> {
    while br.bit_pos() < target_bit {
        let remaining = target_bit - br.bit_pos();
        let chunk = remaining.min(32) as u32;
        let _ = br.read_bits(chunk)?;
    }
    Ok(())
}

/// Per-block-mean computation per `spec/05` §2.5 step 1:
/// `mu_blk = (sum + bs/2) / bs` with the bias added regardless of
/// sign-of-sum and C-style truncation toward zero.
fn block_mean(block: &[i32]) -> i32 {
    let bs = block.len() as i64;
    if bs == 0 {
        return 0;
    }
    let sum: i64 = block.iter().map(|&v| v as i64).sum();
    let bias = bs / 2;
    let numerator = sum + bias;
    (numerator / bs) as i32
}

/// One-shot decoder over an entire `.shn` byte buffer.
pub fn decode(file_bytes: &[u8]) -> Result<DecodedStream> {
    let header = crate::header::parse_header(file_bytes)?;

    // Step the BitReader past the header parameter block. The
    // bit-count can exceed 32 so we re-decode the header fields
    // through a fresh reader rather than trying to skip in one
    // read_bits call (which is bounded to 32 bits).
    let mut br = BitReader::new(&file_bytes[5..]);
    skip_to_bit(&mut br, header.header_end_bit)?;

    let nch = header.channels as usize;
    let carry_len = header.carry_len();
    let mut channels: Vec<ChannelState> = (0..nch)
        .map(|_| ChannelState::new(carry_len, header.mean_blocks))
        .collect();

    // Per-channel output — accumulate flat blocks, then interleave at
    // the end. We can size the buffer dynamically since we don't know
    // the final sample count up-front.
    let mut per_channel_samples: Vec<Vec<i32>> = (0..nch).map(|_| Vec::new()).collect();

    let mut current_block_size: u32 = header.blocksize;
    let mut bshift: u32 = 0;
    let mut channel_cursor: usize = 0;
    let mut verbatim_prefix: Vec<u8> = Vec::new();

    loop {
        let fn_code = read_uvar(&mut br, FNSIZE)?;
        match fn_code {
            fn_code::DIFF0 | fn_code::DIFF1 | fn_code::DIFF2 | fn_code::DIFF3 => {
                let block = decode_diff_block(
                    &mut br,
                    fn_code,
                    current_block_size as usize,
                    &channels[channel_cursor],
                    header.mean_blocks,
                )?;
                // Update mean buffer & carry, then commit.
                let mu_blk = block_mean(&block);
                let cs = &mut channels[channel_cursor];
                cs.push_block_mean(mu_blk);
                cs.update_carry(&block);
                per_channel_samples[channel_cursor].extend_from_slice(&block);
                channel_cursor = (channel_cursor + 1) % nch;
            }
            fn_code::QLPC => {
                let block = decode_qlpc_block(
                    &mut br,
                    current_block_size as usize,
                    header.max_lpc_order,
                    &channels[channel_cursor],
                    header.mean_blocks,
                )?;
                let mu_blk = block_mean(&block);
                let cs = &mut channels[channel_cursor];
                cs.push_block_mean(mu_blk);
                cs.update_carry(&block);
                per_channel_samples[channel_cursor].extend_from_slice(&block);
                channel_cursor = (channel_cursor + 1) % nch;
            }
            fn_code::ZERO => {
                // Zero block: emit `bs` samples = running mean.
                let cs = &channels[channel_cursor];
                let mu = cs.running_mean(header.mean_blocks);
                let bs = current_block_size as usize;
                let block = vec![mu; bs];
                let mu_blk = mu;
                let cs = &mut channels[channel_cursor];
                cs.push_block_mean(mu_blk);
                cs.update_carry(&block);
                per_channel_samples[channel_cursor].extend_from_slice(&block);
                channel_cursor = (channel_cursor + 1) % nch;
            }
            fn_code::BLOCKSIZE => {
                let new_bs = read_ulong(&mut br)?;
                if new_bs == 0 || new_bs > crate::MAX_BLOCKSIZE {
                    return Err(Error::UnsupportedBlockSize(new_bs));
                }
                current_block_size = new_bs;
            }
            fn_code::BITSHIFT => {
                let new_shift = read_uvar(&mut br, BITSHIFTSIZE)?;
                if new_shift >= 32 {
                    return Err(Error::BitShiftOverflow(new_shift));
                }
                bshift = new_shift;
            }
            fn_code::VERBATIM => {
                let length = read_uvar(&mut br, VERBATIM_CHUNK_SIZE)?;
                let mut bytes = Vec::with_capacity(length as usize);
                for _ in 0..length {
                    let b = read_uvar(&mut br, VERBATIM_BYTE_SIZE)?;
                    bytes.push((b & 0xFF) as u8);
                }
                verbatim_prefix.extend_from_slice(&bytes);
            }
            fn_code::QUIT => {
                break;
            }
            other => return Err(Error::UnknownFunctionCode(other)),
        }
    }

    // Apply per-stream bshift and assemble the interleaved output.
    let samples_per_channel = per_channel_samples.iter().map(Vec::len).min().unwrap_or(0);
    // Round-robin per spec — every channel should have the same
    // sample count if the stream is well-formed. If they differ
    // we truncate to the shortest, which is the conservative
    // behaviour for partial-block tails.
    let total = samples_per_channel * nch;
    let mut interleaved = Vec::with_capacity(total);
    for i in 0..samples_per_channel {
        for ch_samples in per_channel_samples.iter().take(nch) {
            // Samples in carry/internal lane are pre-shift; output
            // is the lane left-shifted by the current bshift.
            let s = ch_samples[i];
            let out = if bshift == 0 {
                s
            } else {
                s.wrapping_shl(bshift)
            };
            interleaved.push(out);
        }
    }

    Ok(DecodedStream {
        header,
        verbatim_prefix,
        samples: interleaved,
        samples_per_channel,
        final_bshift: bshift,
    })
}

/// Bulk-decode `bs` `svar(width)` residuals into `out` using a 64-bit
/// reservoir reader. Round-6 hot path: the reservoir keeps the next
/// 32–64 unread bits in-register and resolves prefix scans with
/// `u64::leading_zeros` (hardware `lzcnt`/`clz`), avoiding the
/// byte-LUT + per-bit refill overhead of
/// [`BitReader::read_uvar_prefix`] on the long blocks that dominate
/// decode time.
///
/// On return the underlying [`BitReader`] cursor is advanced past the
/// last residual consumed. Returns [`Error::UnexpectedEof`] if the
/// stream ends before `bs` residuals have been read.
fn read_residuals_into(
    br: &mut BitReader<'_>,
    width: u32,
    bs: usize,
    out: &mut Vec<i32>,
) -> Result<()> {
    out.clear();
    out.reserve(bs);
    let mut bs64 = Bitstream64::from_bit_reader(br);
    for _ in 0..bs {
        let s = bs64.read_svar(width, RESIDUAL_UVAR_MAX_ZEROS)?;
        out.push(s);
    }
    bs64.finalize_into(br)?;
    Ok(())
}

/// Decode one polynomial-difference predictor block (DIFF0..DIFF3).
fn decode_diff_block(
    br: &mut BitReader<'_>,
    fn_code: u32,
    bs: usize,
    cs: &ChannelState,
    mean_blocks: u32,
) -> Result<Vec<i32>> {
    let energy_field = read_uvar(br, ENERGYSIZE)?;
    let width = energy_field
        .checked_add(1)
        .ok_or(Error::ResidualWidthOverflow(energy_field))?;
    if width > RESIDUAL_WIDTH_CAP {
        return Err(Error::ResidualWidthOverflow(width));
    }

    // Round-6: bulk-decode all residuals into a scratch buffer, then
    // run the predictor recurrence on the buffer. Separating I/O
    // (variable-length bit reads) from arithmetic (predictor
    // recurrence) lets the inner residual loop run from a 64-bit
    // bit reservoir rather than per-bit through the `BitReader` API.
    let mut residuals: Vec<i32> = Vec::with_capacity(bs);
    read_residuals_into(br, width, bs, &mut residuals)?;

    let mut samples = Vec::with_capacity(bs);
    let mu = cs.running_mean(mean_blocks);

    match fn_code {
        // DIFF0: s = r + mu (mean estimator only on order-0).
        fn_code::DIFF0 => {
            for &r in &residuals {
                samples.push(r.wrapping_add(mu));
            }
        }
        // DIFF1: s = s(t-1) + r.
        fn_code::DIFF1 => {
            let mut s1 = cs.carry.first().copied().unwrap_or(0);
            for &r in &residuals {
                let s = s1.wrapping_add(r);
                samples.push(s);
                s1 = s;
            }
        }
        // DIFF2: s = 2 s(t-1) - s(t-2) + r.
        fn_code::DIFF2 => {
            let mut s1 = cs.carry.first().copied().unwrap_or(0);
            let mut s2 = cs.carry.get(1).copied().unwrap_or(0);
            for &r in &residuals {
                let s = s1.wrapping_mul(2).wrapping_sub(s2).wrapping_add(r);
                samples.push(s);
                s2 = s1;
                s1 = s;
            }
        }
        // DIFF3: s = 3 s(t-1) - 3 s(t-2) + s(t-3) + r.
        fn_code::DIFF3 => {
            let mut s1 = cs.carry.first().copied().unwrap_or(0);
            let mut s2 = cs.carry.get(1).copied().unwrap_or(0);
            let mut s3 = cs.carry.get(2).copied().unwrap_or(0);
            for &r in &residuals {
                let s = s1
                    .wrapping_mul(3)
                    .wrapping_sub(s2.wrapping_mul(3))
                    .wrapping_add(s3)
                    .wrapping_add(r);
                samples.push(s);
                s3 = s2;
                s2 = s1;
                s1 = s;
            }
        }
        _ => unreachable!("decode_diff_block called with non-DIFF code"),
    }
    Ok(samples)
}

/// Decode one `BLOCK_FN_QLPC` quantised-LPC predictor block.
fn decode_qlpc_block(
    br: &mut BitReader<'_>,
    bs: usize,
    header_max_order: u32,
    cs: &ChannelState,
    _mean_blocks: u32,
) -> Result<Vec<i32>> {
    let order = read_uvar(br, LPCQSIZE)?;
    if order > header_max_order {
        return Err(Error::LpcOrderExceedsHeader {
            block_order: order,
            header_max: header_max_order,
        });
    }
    let mut coefs = Vec::with_capacity(order as usize);
    for _ in 0..order {
        let c = read_svar(br, LPCQUANT)?;
        // Bound coefficient magnitude to keep the predictor in i32.
        if !(-(1 << 16)..=(1 << 16)).contains(&c) {
            return Err(Error::LpcCoefOutOfRange(c));
        }
        coefs.push(c);
    }
    let energy_field = read_uvar(br, ENERGYSIZE)?;
    let width = energy_field
        .checked_add(1)
        .ok_or(Error::ResidualWidthOverflow(energy_field))?;
    if width > RESIDUAL_WIDTH_CAP {
        return Err(Error::ResidualWidthOverflow(width));
    }

    let order_us = order as usize;
    let carry_len = cs.carry.len();
    // Build a working history: index 0 = s(t-1), 1 = s(t-2), …
    // We'll read this many entries from the carry, and roll new
    // samples into it as we produce them.
    let mut hist: Vec<i32> = vec![0; order_us.max(carry_len)];
    for (i, slot) in hist.iter_mut().enumerate().take(order_us) {
        *slot = cs.carry.get(i).copied().unwrap_or(0);
    }

    // Round-6: bulk-decode residuals first, then run the LPC
    // predictor recurrence on the buffer. Same rationale as
    // [`decode_diff_block`].
    let mut residuals: Vec<i32> = Vec::with_capacity(bs);
    read_residuals_into(br, width, bs, &mut residuals)?;

    let mut samples = Vec::with_capacity(bs);
    for &r in &residuals {
        // Predict: ŝ(t) = sum_{i=1..=order} a_i * s(t - i)
        //                = sum_{i=0..order} coefs[i] * hist[i]
        let mut pred: i64 = 0;
        for i in 0..order_us {
            pred = pred.wrapping_add((coefs[i] as i64).wrapping_mul(hist[i] as i64));
        }
        let s = (pred as i32).wrapping_add(r);
        samples.push(s);
        // Roll history: shift right by 1, write s at index 0.
        for i in (1..hist.len()).rev() {
            hist[i] = hist[i - 1];
        }
        if !hist.is_empty() {
            hist[0] = s;
        }
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_mean_computes_with_bias() {
        // sum = 0, bs = 4 → mu = (0 + 2)/4 = 0.
        assert_eq!(block_mean(&[0, 0, 0, 0]), 0);
        // sum = 10, bs = 4 → (10 + 2)/4 = 3.
        assert_eq!(block_mean(&[1, 2, 3, 4]), 3);
        // sum = -10, bs = 4 → (-10 + 2)/4 = -8/4 = -2.
        assert_eq!(block_mean(&[-1, -2, -3, -4]), -2);
    }

    #[test]
    fn carry_fully_refreshes_for_long_block() {
        let mut cs = ChannelState::new(3, 0);
        cs.update_carry(&[1, 2, 3, 4, 5]);
        assert_eq!(cs.carry, vec![5, 4, 3]);
    }

    #[test]
    fn carry_partial_refresh_for_short_block() {
        let mut cs = ChannelState::new(3, 0);
        cs.carry.copy_from_slice(&[10, 20, 30]);
        // bs = 1 < carry_len = 3.
        cs.update_carry(&[7]);
        // After a 1-sample block, the most-recent entry is 7, and the
        // older two entries shift right by 1 (they were 10, 20, 30 —
        // s(t-1), s(t-2), s(t-3) — and the new s becomes s(t-1), the
        // old s(t-1)=10 becomes s(t-2), old s(t-2)=20 becomes s(t-3)).
        assert_eq!(cs.carry, vec![7, 10, 20]);
    }
}
