//! Shorten polynomial-difference predictor kernels — orders 0..3 +
//! per-channel running mean estimator.
//!
//! Implements the `BLOCK_FN_DIFF0..3` reconstruction recurrences of
//! `docs/audio/shorten/spec/03-block-and-predictor.md` §3.1..§3.4
//! plus the per-channel sample-history carry of `spec/03` §3.11 /
//! `spec/05-state-and-quirks.md` §1, with residuals decoded under the
//! `svar(e + 1)` rule of `spec/05` §3 ("encoded value plus one"), and
//! the per-channel running mean estimator of `spec/05` §2 + §2.5.
//!
//! ## Predictor recurrences
//!
//! From `spec/03` §3.1..§3.4 (TR.156 equations 3..10):
//!
//! - **Order 0** (`BLOCK_FN_DIFF0`):
//!   `ŝ₀(t) = mu_chan`; reconstruction `s(t) = e₀(t) + mu_chan` where
//!   `mu_chan` is the per-channel running mean from the estimator of
//!   `spec/05` §2.3 (the buffer's value at the *start* of this block).
//! - **Order 1** (`BLOCK_FN_DIFF1`):
//!   `ŝ₁(t) = s(t-1)`; reconstruction `s(t) = s(t-1) + e₁(t)`.
//! - **Order 2** (`BLOCK_FN_DIFF2`):
//!   `ŝ₂(t) = 2·s(t-1) − s(t-2)`; reconstruction
//!   `s(t) = 2·s(t-1) − s(t-2) + e₂(t)`.
//! - **Order 3** (`BLOCK_FN_DIFF3`):
//!   `ŝ₃(t) = 3·s(t-1) − 3·s(t-2) + s(t-3)`; reconstruction
//!   `s(t) = 3·s(t-1) − 3·s(t-2) + s(t-3) + e₃(t)`.
//!
//! The carry buffer per `spec/05` §1.1 indexes `carry[0] = s(t-1)`,
//! `carry[1] = s(t-2)`, `carry[2] = s(t-3)` (most-recent-first), and is
//! initialised to zero at stream start. After each block completes for
//! channel `ch`, the carry is refreshed from the just-emitted block per
//! `spec/05` §1.3 — for a `bs ≥ 3` block (the common case) the carry
//! becomes `[blk[bs-1], blk[bs-2], blk[bs-3]]`.
//!
//! ## Residual mantissa width
//!
//! `spec/05` §3 pins the energy field's "encoded value plus one"
//! convention: the wire reads `e = uvar(ENERGYSIZE = 3)` and the
//! subsequent `bs` residuals are read as `svar(e + 1)`. The
//! `signed-zero` width-zero case is degenerate per `spec/05` §3 and
//! never selected by an optimising encoder, but the implementation
//! handles it (reading would yield only the `uvar(0)` unary part as a
//! signed-fold integer).
//!
//! ## Mean estimator (`spec/05` §2 + §2.5)
//!
//! [`MeanEstimator`] holds a per-channel sliding window of length
//! `H_meanblocks`, initialised to all zeros at stream start
//! (`spec/05` §2.1). After every sample-producing block (a DIFFn /
//! QLPC / ZERO command) the estimator's `record_block()` computes the
//! per-block mean
//!
//! ```text
//!   mu_blk = trunc_div(sum_of_block + bs/2, bs)
//! ```
//!
//! and slides it into the window's most-recent slot, evicting the
//! oldest. `mu_chan()` returns the channel-wide running mean
//!
//! ```text
//!   mu_chan = trunc_div(sum_of_slots + H_meanblocks/2, H_meanblocks)
//! ```
//!
//! Both divisions truncate toward zero (C semantics) and apply the
//! `+ divisor/2` bias regardless of the sign of the numerator — pinned
//! by `spec/05` §2.5 (validation-corrected). The mean estimator
//! contributes only to `BLOCK_FN_DIFF0`'s reconstruction (the higher-
//! order polynomial-difference predictors and the LPC predictor are
//! mean-invariant) and to `BLOCK_FN_ZERO`'s emitted-sample value
//! ([`fill_zero_block`]).
//!
//! ### `H_meanblocks = 0`
//!
//! When `H_meanblocks = 0` the estimator is disabled (`spec/01` §3.5):
//! `mu_chan()` is always zero, `record_block()` is a no-op, and the
//! DIFF0 reconstruction reduces to `s(t) = e₀(t)`. Fixture `F1`'s
//! header carries `H_meanblocks = 4`, so the disabled branch is not
//! exercised by `F1`-style integration tests but is unit-tested
//! independently.
//!
//! ## Arithmetic width
//!
//! Reconstruction is performed in `i64` to avoid intermediate overflow
//! for the order-3 case (a worst-case `3·s(t-1) − 3·s(t-2) + s(t-3) +
//! e₃(t)` exceeds `i32` range only for pathological inputs but the
//! `i64` headroom is free). Samples are stored in the public carry
//! buffer as `i32` since `spec/05` §6 names `s16hl` / `s16lh` (16-bit
//! signed) as the largest pinned sample format. The narrowing checks
//! at the `i64 -> i32` boundary surface as
//! [`crate::Error::SampleOverflow`].

use crate::bitreader::BitReader;
use crate::block::FunctionCode;
use crate::error::{Error, Result};

/// Width of the per-block energy parameter on the wire. Pinned by
/// `spec/02` §4.2 (`ENERGYSIZE = 3`).
pub const ENERGYSIZE: u32 = 3;

/// Minimum sample-history carry buffer length. `spec/05` §1 / `spec/03`
/// §3.11 pin `CARRY_LEN = max(3, H_maxlpcorder)`; for streams with
/// `H_maxlpcorder = 0` (every fixture observed against real predictor
/// commands so far — see [`crate::ShortenStreamHeader`]'s
/// `sample_history_carry_len`), the buffer is exactly 3 samples.
pub const CARRY_LEN_FLOOR: usize = 3;

/// Implementation-side safety cap on the residual mantissa width. The
/// per-block energy field is `uvar(ENERGYSIZE = 3)` (encoded value 0..7
/// plus rare prefix-zero extensions). After the `+1` offset of `spec/05`
/// §3 the residual width is at most 8 for any encoder-emitted block;
/// values beyond 30 would push `svar(n)`'s mantissa beyond what the
/// `u32`-backed `read_uvar` can return without overflow. We cap at 30
/// to leave a one-bit margin against `u32` mantissa overflow inside
/// `read_uvar` while still admitting every legitimate encoder choice.
const MAX_RESIDUAL_WIDTH: u32 = 30;

/// Implementation-side safety cap on the block-size argument the
/// predictor decoder will accept. `H_blocksize` is encoded as a
/// `uvar(9)` in the only fixture observed (`F1`); large encoder
/// choices are bounded by TR.156 at ~256 samples per block. Allowing
/// up to 1 MiB samples per block keeps the cap generous while
/// preventing a malformed stream from triggering a multi-gigabyte
/// allocation through `Vec::with_capacity`.
const MAX_BLOCK_SAMPLES: u32 = 1024 * 1024;

/// The per-channel sample-history carry buffer of `spec/05` §1 — a
/// most-recent-first ring of the last `CARRY_LEN` samples emitted for
/// the channel.
///
/// Layout:
///
/// * `samples[0]` = `s(t - 1)` (most-recent past sample)
/// * `samples[1]` = `s(t - 2)`
/// * `samples[2]` = `s(t - 3)`
/// * `samples[i]` = `s(t - 1 - i)`
///
/// Initialised to all zeros at stream start (`spec/05` §1.2). After
/// each sample-producing block completes, [`Self::update_after_block`]
/// refreshes the buffer from the block's tail samples per `spec/05`
/// §1.3.
///
/// The carry length is fixed at construction time to
/// `max(CARRY_LEN_FLOOR, H_maxlpcorder)`; for the `H_maxlpcorder = 0`
/// fixture (`F1`) the buffer is exactly 3 samples long, satisfying the
/// polynomial-difference predictors of orders 1..3.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelCarry {
    /// Most-recent-first sample history. Length is fixed at
    /// construction.
    samples: Vec<i32>,
}

impl ChannelCarry {
    /// Construct a fresh zero-initialised carry of `len` samples.
    /// `len` must be at least [`CARRY_LEN_FLOOR`]; smaller values
    /// would fail to supply enough history for order-1..3 predictors.
    pub fn new(len: usize) -> Self {
        let len = core::cmp::max(len, CARRY_LEN_FLOOR);
        Self {
            samples: vec![0i32; len],
        }
    }

    /// Length of the carry buffer.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// True if the carry has no entries — only reachable in the
    /// degenerate `Self::new(0)` case once the floor below `CARRY_LEN_FLOOR`
    /// is applied (and never returns true in practice).
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Read the `i`-th most-recent past sample. `at(0)` returns
    /// `s(t-1)`, `at(1)` returns `s(t-2)`, etc.
    pub fn at(&self, i: usize) -> i32 {
        self.samples[i]
    }

    /// Refresh the carry from the just-emitted block per `spec/05` §1.3.
    ///
    /// For an emitted block `blk[0..bs)`:
    ///
    /// ```text
    /// carry[i] = blk[bs - 1 - i]              if  bs - 1 - i >= 0
    ///          = previous carry[i - bs]       otherwise
    /// ```
    ///
    /// The second clause is exercised only when a sub-block-size
    /// override (`spec/03` §3.6) has reduced `bs` below the carry
    /// length. For typical blocks (`bs >= CARRY_LEN`) the buffer is
    /// fully refreshed from the new block alone.
    pub fn update_after_block(&mut self, block: &[i32]) {
        let bs = block.len();
        let len = self.samples.len();
        let mut next = vec![0i32; len];
        for (i, slot) in next.iter_mut().enumerate() {
            if bs > i {
                *slot = block[bs - 1 - i];
            } else {
                // Carry over from the older portion of the existing
                // buffer.
                let old_idx = i - bs;
                *slot = self.samples[old_idx];
            }
        }
        self.samples = next;
    }
}

/// Truncate-toward-zero integer division (C `/` semantics on signed
/// operands), used by the mean-estimator arithmetic of `spec/05` §2.5.
///
/// Rust's `i64::div_euclid` rounds toward negative infinity while
/// Rust's `/` operator already truncates toward zero, so this helper
/// is technically just `numerator / divisor`. It exists as a named
/// helper so the call sites read against the spec wording
/// ("truncates toward zero (C semantics, not Python floor)" — §2.5).
///
/// Panics on `divisor == 0`; callers gate the disabled-estimator path
/// (`H_meanblocks = 0`) before reaching this helper.
fn trunc_div(numerator: i64, divisor: i64) -> i64 {
    debug_assert!(divisor != 0, "trunc_div by zero");
    numerator / divisor
}

/// Per-channel running mean estimator of `spec/05` §2 + §2.5.
///
/// Holds a sliding window of length `H_meanblocks` storing the
/// per-block means of the most recent sample-producing blocks for one
/// channel. The buffer is initialised to all zeros at stream start
/// (`spec/05` §2.1). After every sample-producing block the channel's
/// estimator advances by one slot via [`Self::record_block`].
///
/// ## Disabled branch (`H_meanblocks = 0`)
///
/// `spec/01` §3.5 allows the encoder to disable the mean estimator by
/// setting `H_meanblocks = 0`. In that case the channel's [`mu_chan`]
/// is always zero and [`record_block`] is a no-op. The buffer is empty
/// (`slots.is_empty()`).
///
/// [`mu_chan`]: Self::mu_chan
/// [`record_block`]: Self::record_block
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeanEstimator {
    /// Sliding window of per-block means, oldest at index 0 and
    /// newest at index `slots.len() - 1`. Length is fixed to
    /// `H_meanblocks` at construction; empty if `H_meanblocks = 0`.
    slots: Vec<i64>,
}

impl MeanEstimator {
    /// Construct a fresh zero-initialised estimator with a window of
    /// `meanblocks` slots. When `meanblocks = 0` the estimator is
    /// disabled (mu_chan always returns zero per `spec/01` §3.5 +
    /// `spec/05` §2.1).
    pub fn new(meanblocks: u32) -> Self {
        Self {
            slots: vec![0i64; meanblocks as usize],
        }
    }

    /// Length of the sliding window. Equals `H_meanblocks`. Zero when
    /// the estimator is disabled.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// True if the estimator is disabled (`H_meanblocks = 0`).
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The channel's running mean at the start of the next block
    /// (`spec/05` §2.3 / §2.5):
    ///
    /// ```text
    ///   mu_chan = trunc_div(sum_of_slots + H_meanblocks/2, H_meanblocks)
    /// ```
    ///
    /// with truncation toward zero. The `+ H_meanblocks/2` bias is
    /// added regardless of the numerator's sign (validation-corrected
    /// per `spec/05` §2.5).
    ///
    /// Returns 0 when the estimator is disabled (`H_meanblocks = 0`).
    pub fn mu_chan(&self) -> i64 {
        let n = self.slots.len() as i64;
        if n == 0 {
            return 0;
        }
        let sum: i64 = self.slots.iter().sum();
        trunc_div(sum + n / 2, n)
    }

    /// Record the per-block mean of `block` into the sliding window
    /// per `spec/05` §2.5. The per-block mean is
    ///
    /// ```text
    ///   mu_blk = trunc_div(sum_of_block + bs/2, bs)
    /// ```
    ///
    /// with truncation toward zero and the always-positive bias
    /// `+ bs/2`. The oldest slot (`slots[0]`) is evicted and the new
    /// mean is appended at the most-recent slot (`slots[len - 1]`).
    ///
    /// No-op when the estimator is disabled (`H_meanblocks = 0`) or
    /// the block is empty (`bs = 0`). Empty-block guard is defensive;
    /// real DIFFn / QLPC / ZERO blocks always have `bs >= 1`.
    pub fn record_block(&mut self, block: &[i32]) {
        if self.slots.is_empty() {
            return;
        }
        let bs = block.len() as i64;
        if bs == 0 {
            return;
        }
        let sum: i64 = block.iter().map(|&x| x as i64).sum();
        let mu_blk = trunc_div(sum + bs / 2, bs);
        // Slide: evict slots[0], shift left, append mu_blk at tail.
        let len = self.slots.len();
        for i in 0..(len - 1) {
            self.slots[i] = self.slots[i + 1];
        }
        self.slots[len - 1] = mu_blk;
    }
}

/// Polynomial-difference predictor order (0..3) — the predictor's
/// numeric subscript in TR.156 §3.2 equations 3..10.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolyOrder {
    /// `BLOCK_FN_DIFF0` — zero-prediction.
    Order0,
    /// `BLOCK_FN_DIFF1` — predict the previous sample.
    Order1,
    /// `BLOCK_FN_DIFF2` — predict by line-fit through the last two
    /// samples.
    Order2,
    /// `BLOCK_FN_DIFF3` — predict by quadratic-fit through the last
    /// three samples.
    Order3,
}

impl PolyOrder {
    /// The polynomial-difference predictor order associated with a
    /// function code. Returns `None` for non-DIFFn codes.
    pub fn from_function_code(code: FunctionCode) -> Option<Self> {
        Some(match code {
            FunctionCode::Diff0 => PolyOrder::Order0,
            FunctionCode::Diff1 => PolyOrder::Order1,
            FunctionCode::Diff2 => PolyOrder::Order2,
            FunctionCode::Diff3 => PolyOrder::Order3,
            _ => return None,
        })
    }

    /// Numeric order (0..3).
    pub fn order(self) -> u32 {
        match self {
            PolyOrder::Order0 => 0,
            PolyOrder::Order1 => 1,
            PolyOrder::Order2 => 2,
            PolyOrder::Order3 => 3,
        }
    }
}

/// Read a `BLOCK_FN_DIFFn` payload after its function-code field has
/// already been consumed and reconstruct the channel block.
///
/// Layout per `spec/03` §3.1..§3.4:
///
/// ```text
/// <DIFFn block> ::= <energy> <residual>×bs
///   energy:   uvar(ENERGYSIZE = 3)            (encoded value e = n - 1)
///   residual: svar(e + 1)                       (mantissa width = e + 1)
/// ```
///
/// `bs` is the current sub-block size (default `H_blocksize`, may be
/// overridden per `spec/03` §3.6 — not yet wired up).
/// `carry` supplies the per-channel sample history per `spec/05` §1;
/// `mu_chan` is the per-channel running mean from
/// [`MeanEstimator::mu_chan`] at the *start* of this block per
/// `spec/05` §2.3 — only consumed by `PolyOrder::Order0`; the higher
/// orders are mean-invariant per `spec/05` §2 introductory paragraph.
/// On return, both `carry.update_after_block(&block)` and
/// `mean.record_block(&block)` should be called by the higher-level
/// command-dispatch layer to refresh state from the just-decoded
/// block. (This function does **not** mutate `carry` or any mean
/// estimator — the caller owns the state-update step so a decoder
/// that rejects a block mid-decode does not corrupt the carry.)
///
/// Returns the decoded block as `Vec<i32>` of length exactly `bs`.
///
/// Errors:
///
/// * [`Error::EnergyTooLarge`] — the encoded energy plus one exceeds
///   the implementation cap [`MAX_RESIDUAL_WIDTH`].
/// * [`Error::BlockTooLarge`] — `bs` exceeds the implementation
///   safety cap.
/// * [`Error::SampleOverflow`] — a reconstructed sample fell outside
///   `i32` range.
/// * [`Error::Truncated`] / [`Error::OverflowingUvar`] — bit-stream
///   exhaustion or pathological `uvar` prefix.
pub fn decode_diff_block(
    reader: &mut BitReader<'_>,
    order: PolyOrder,
    bs: u32,
    carry: &ChannelCarry,
    mu_chan: i64,
) -> Result<Vec<i32>> {
    if bs > MAX_BLOCK_SAMPLES {
        return Err(Error::BlockTooLarge(bs));
    }
    let energy_encoded = reader.read_uvar(ENERGYSIZE)?;
    // Per spec/05 §3: the residual mantissa width is the encoded value
    // plus one.
    let width = energy_encoded
        .checked_add(1)
        .ok_or(Error::EnergyTooLarge(energy_encoded))?;
    if width > MAX_RESIDUAL_WIDTH {
        return Err(Error::EnergyTooLarge(energy_encoded));
    }

    // Pre-stage previous samples in `i64` for the recurrence. carry[0]
    // = s(t-1), carry[1] = s(t-2), carry[2] = s(t-3).
    let mut s_m1: i64 = carry.at(0) as i64;
    let mut s_m2: i64 = if carry.len() > 1 {
        carry.at(1) as i64
    } else {
        0
    };
    let mut s_m3: i64 = if carry.len() > 2 {
        carry.at(2) as i64
    } else {
        0
    };

    let mut out: Vec<i32> = Vec::with_capacity(bs as usize);
    for _ in 0..bs {
        let residual: i64 = reader.read_svar(width)?;
        let predicted: i64 = match order {
            // spec/05 §2.3: DIFF0 adds the channel running mean.
            PolyOrder::Order0 => mu_chan,
            PolyOrder::Order1 => s_m1,
            PolyOrder::Order2 => 2 * s_m1 - s_m2,
            PolyOrder::Order3 => 3 * s_m1 - 3 * s_m2 + s_m3,
        };
        let s = predicted
            .checked_add(residual)
            .ok_or(Error::SampleOverflow)?;
        // Narrow to i32 — the sample-storage type.
        let s_i32: i32 = s.try_into().map_err(|_| Error::SampleOverflow)?;
        out.push(s_i32);
        // Slide the recurrence window.
        s_m3 = s_m2;
        s_m2 = s_m1;
        s_m1 = s;
    }
    Ok(out)
}

/// Fill a `BLOCK_FN_ZERO` payload with `bs` samples all equal to
/// `mu_chan` per `spec/05` §2.4. The command carries no further wire
/// payload after its function-code field — the function takes no
/// `BitReader` argument and emits the block synthetically.
///
/// As with [`decode_diff_block`] the caller owns the state-update
/// step: after dispatching this block the higher-level driver must
/// call `carry.update_after_block(&block)` and
/// `mean.record_block(&block)` to refresh the channel state.
///
/// Errors:
///
/// * [`Error::BlockTooLarge`] — `bs` exceeds the implementation
///   safety cap.
/// * [`Error::SampleOverflow`] — `mu_chan` falls outside `i32` range.
///   (Realistic encoders never produce such values; `mu_chan` is
///   bounded by the sample range as a long-window average. The
///   `i64 -> i32` boundary is guarded for defence in depth.)
pub fn fill_zero_block(bs: u32, mu_chan: i64) -> Result<Vec<i32>> {
    if bs > MAX_BLOCK_SAMPLES {
        return Err(Error::BlockTooLarge(bs));
    }
    let mu_i32: i32 = mu_chan.try_into().map_err(|_| Error::SampleOverflow)?;
    Ok(vec![mu_i32; bs as usize])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::FNSIZE;

    /// MSB-first bit packer used to build synthetic predictor-block
    /// fixtures (same helper as in `block::tests` / the integration
    /// test).
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

    /// `uvar(n)` encoder, mirror of the per-spec helper used by the
    /// block dispatch test module.
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

    /// `svar(n)` encoder using the same one's-complement folding the
    /// `BitReader::read_svar` decoder inverts. For `s >= 0` the folded
    /// `u = s << 1`; for `s < 0` the folded `u = ((!s) << 1) | 1`.
    fn encode_svar(value: i64, n: u32) -> Vec<u32> {
        let u: u64 = if value >= 0 {
            (value as u64) << 1
        } else {
            (((!value) as u64) << 1) | 1
        };
        // Constrained to u32 for the test-side encoder.
        let u32_val = u32::try_from(u).expect("svar fits in u32 for these tests");
        encode_uvar(u32_val, n)
    }

    /// Assemble a synthetic DIFFn block: function-code-already-consumed
    /// energy + bs residuals. Returns the byte buffer to feed a
    /// `BitReader`.
    fn synth_diff_payload(energy_encoded: u32, residuals: &[i64]) -> Vec<u8> {
        let width = energy_encoded + 1;
        let mut bits = Vec::new();
        bits.extend(encode_uvar(energy_encoded, ENERGYSIZE));
        for &r in residuals {
            bits.extend(encode_svar(r, width));
        }
        pack_bits_msb_first(&bits)
    }

    #[test]
    fn order_from_function_code_maps_diff_codes_only() {
        assert_eq!(
            PolyOrder::from_function_code(FunctionCode::Diff0),
            Some(PolyOrder::Order0)
        );
        assert_eq!(
            PolyOrder::from_function_code(FunctionCode::Diff1),
            Some(PolyOrder::Order1)
        );
        assert_eq!(
            PolyOrder::from_function_code(FunctionCode::Diff2),
            Some(PolyOrder::Order2)
        );
        assert_eq!(
            PolyOrder::from_function_code(FunctionCode::Diff3),
            Some(PolyOrder::Order3)
        );
        assert_eq!(PolyOrder::from_function_code(FunctionCode::Qlpc), None);
        assert_eq!(PolyOrder::from_function_code(FunctionCode::Verbatim), None);
    }

    #[test]
    fn carry_construction_floor_is_three() {
        let c = ChannelCarry::new(0);
        assert_eq!(c.len(), 3);
        let c2 = ChannelCarry::new(1);
        assert_eq!(c2.len(), 3);
        let c3 = ChannelCarry::new(5);
        assert_eq!(c3.len(), 5);
        // Zero-initialised per spec/05 §1.2.
        assert_eq!(c.at(0), 0);
        assert_eq!(c.at(1), 0);
        assert_eq!(c.at(2), 0);
    }

    #[test]
    fn carry_update_refreshes_from_block_tail() {
        // spec/05 §1.3: for a block [10, 20, 30, 40, 50] with a
        // 3-slot carry, the carry becomes [50, 40, 30].
        let mut c = ChannelCarry::new(3);
        c.update_after_block(&[10, 20, 30, 40, 50]);
        assert_eq!(c.at(0), 50);
        assert_eq!(c.at(1), 40);
        assert_eq!(c.at(2), 30);
    }

    #[test]
    fn carry_update_short_block_retains_older_history() {
        // spec/05 §1.3 second clause: a short block (bs < CARRY_LEN)
        // retains the older portion of the prior carry.
        //
        // Initial carry: [100, 200, 300]. Block of length 2 = [40, 50].
        // New carry should be [50, 40, 100] — the two new samples plus
        // the most-recent slot of the prior carry (since bs = 2 means
        // i = 0,1 take from new block; i = 2 takes from old carry[0]
        // = 100).
        let mut c = ChannelCarry::new(3);
        c.samples = vec![100, 200, 300];
        c.update_after_block(&[40, 50]);
        assert_eq!(c.at(0), 50);
        assert_eq!(c.at(1), 40);
        assert_eq!(c.at(2), 100);
    }

    #[test]
    fn decode_diff0_reduces_to_residual_with_zero_mean() {
        // DIFF0 with mean estimator disabled: s(t) = e0(t).
        // Energy encoded = 0 -> width = 1; residuals fit in 1 bit's
        // signed range under one's-complement folding, so build a
        // larger-width example: encoded = 3 -> width = 4.
        let residuals: Vec<i64> = vec![0, 1, -1, 7, -7, 3, -3, 5];
        let buf = synth_diff_payload(3, &residuals);
        let mut r = BitReader::new(&buf);
        let carry = ChannelCarry::new(3);
        let block = decode_diff_block(&mut r, PolyOrder::Order0, residuals.len() as u32, &carry, 0)
            .expect("DIFF0 must decode");
        let expected: Vec<i32> = residuals.iter().map(|&x| x as i32).collect();
        assert_eq!(block, expected);
    }

    #[test]
    fn decode_diff1_reproduces_cumulative_sum() {
        // DIFF1 reconstruction: s(t) = s(t-1) + e1(t), s(-1) = 0.
        let residuals: Vec<i64> = vec![4, 0, -26, 42, -17, -14];
        // Expected cumulative sum: 4, 4, -22, 20, 3, -11.
        let expected: Vec<i32> = vec![4, 4, -22, 20, 3, -11];
        let buf = synth_diff_payload(3, &residuals); // width 4 -> fits all
        let mut r = BitReader::new(&buf);
        let carry = ChannelCarry::new(3);
        let block = decode_diff_block(&mut r, PolyOrder::Order1, residuals.len() as u32, &carry, 0)
            .expect("DIFF1 must decode");
        assert_eq!(block, expected);
    }

    #[test]
    fn decode_diff2_uses_line_fit_predictor() {
        // DIFF2 reconstruction: s(t) = 2*s(t-1) - s(t-2) + e2(t).
        // Starting from carry = [0, 0]:
        //   s(0) = 0 - 0 + r0 = r0
        //   s(1) = 2*s(0) - 0 + r1 = 2*r0 + r1
        //   s(2) = 2*s(1) - s(0) + r2 = 4*r0 + 2*r1 - r0 + r2 = 3*r0 + 2*r1 + r2
        //   ...
        // For residuals [1, 0, 0, 0, 0]:
        //   s(0)=1, s(1)=2, s(2)=3, s(3)=4, s(4)=5
        let residuals: Vec<i64> = vec![1, 0, 0, 0, 0];
        let expected: Vec<i32> = vec![1, 2, 3, 4, 5];
        let buf = synth_diff_payload(2, &residuals); // width 3
        let mut r = BitReader::new(&buf);
        let carry = ChannelCarry::new(3);
        let block = decode_diff_block(&mut r, PolyOrder::Order2, residuals.len() as u32, &carry, 0)
            .expect("DIFF2 must decode");
        assert_eq!(block, expected);
    }

    #[test]
    fn decode_diff3_uses_quadratic_fit_predictor() {
        // DIFF3 reconstruction: s(t) = 3*s(t-1) - 3*s(t-2) + s(t-3) + e3(t).
        // Starting from carry = [0, 0, 0]:
        //   s(0) = 0 + r0 = r0
        //   s(1) = 3*r0 + 0 + 0 + r1 = 3*r0 + r1
        //   s(2) = 3*s(1) - 3*r0 + 0 + r2 = 3*(3*r0 + r1) - 3*r0 + r2
        //        = 9*r0 + 3*r1 - 3*r0 + r2 = 6*r0 + 3*r1 + r2
        // For r = [1, 0, 0, 0]:
        //   s(0)=1, s(1)=3, s(2)=6,
        //   s(3) = 3*s(2) - 3*s(1) + s(0) + r3
        //        = 3*6 - 3*3 + 1 + 0 = 18 - 9 + 1 = 10
        let residuals: Vec<i64> = vec![1, 0, 0, 0];
        let expected: Vec<i32> = vec![1, 3, 6, 10];
        let buf = synth_diff_payload(2, &residuals); // width 3
        let mut r = BitReader::new(&buf);
        let carry = ChannelCarry::new(3);
        let block = decode_diff_block(&mut r, PolyOrder::Order3, residuals.len() as u32, &carry, 0)
            .expect("DIFF3 must decode");
        assert_eq!(block, expected);
    }

    #[test]
    fn decode_diff1_with_nonzero_carry_continues_recurrence() {
        // Two consecutive DIFF1 blocks for the same channel — verify
        // the carry hands `s(t-1)` from the first block to the second.
        //
        // Block A residuals [10, 5, -3] -> samples [10, 15, 12] from
        //   zero carry.
        // After block A, carry[0] = 12.
        // Block B residuals [7, -2, -5] under carry[0] = 12 ->
        //   samples [12 + 7 = 19, 19 - 2 = 17, 17 - 5 = 12].
        let buf_a = synth_diff_payload(3, &[10, 5, -3]); // width 4
        let mut r_a = BitReader::new(&buf_a);
        let mut carry = ChannelCarry::new(3);
        let block_a = decode_diff_block(&mut r_a, PolyOrder::Order1, 3, &carry, 0).unwrap();
        assert_eq!(block_a, vec![10, 15, 12]);
        carry.update_after_block(&block_a);
        // carry now [12, 15, 10] (most-recent-first).
        assert_eq!(carry.at(0), 12);
        assert_eq!(carry.at(1), 15);
        assert_eq!(carry.at(2), 10);

        let buf_b = synth_diff_payload(3, &[7, -2, -5]);
        let mut r_b = BitReader::new(&buf_b);
        let block_b = decode_diff_block(&mut r_b, PolyOrder::Order1, 3, &carry, 0).unwrap();
        assert_eq!(block_b, vec![19, 17, 12]);
    }

    #[test]
    fn decode_diff_block_rejects_oversized_block() {
        let residuals: Vec<i64> = vec![0; 4];
        let buf = synth_diff_payload(0, &residuals);
        let mut r = BitReader::new(&buf);
        let carry = ChannelCarry::new(3);
        // bs above the cap.
        let result = decode_diff_block(&mut r, PolyOrder::Order0, MAX_BLOCK_SAMPLES + 1, &carry, 0);
        assert_eq!(result, Err(Error::BlockTooLarge(MAX_BLOCK_SAMPLES + 1)));
    }

    #[test]
    fn decode_diff_block_truncated_input_returns_truncated() {
        // Provide only the energy, no residuals — the first svar read
        // exhausts the buffer.
        let mut bits = Vec::new();
        bits.extend(encode_uvar(3, ENERGYSIZE));
        let buf = pack_bits_msb_first(&bits);
        let mut r = BitReader::new(&buf);
        let carry = ChannelCarry::new(3);
        let result = decode_diff_block(&mut r, PolyOrder::Order0, 3, &carry, 0);
        assert_eq!(result, Err(Error::Truncated));
    }

    #[test]
    fn decode_diff0_energy_plus_one_anchor_spec05_section_3() {
        // spec/05 §3 + T15: the residual mantissa width is
        // (energy_encoded + 1). With energy_encoded = 3 -> width = 4,
        // the svar reads handle residuals in the range [-2^31, 2^31)
        // at minimum.
        //
        // Encode three residuals at width 4 (encoded energy = 3):
        // r = [4, 4, -22]. Decoding under DIFF1 with zero carry
        // yields: s(0) = 0 + 4 = 4, s(1) = 4 + 4 = 8 (wait — let me
        // redo: this is DIFF1 cumulative from zero carry).
        let residuals: Vec<i64> = vec![4, 4, -22];
        let expected: Vec<i32> = vec![4, 8, -14];
        let buf = synth_diff_payload(3, &residuals);
        let mut r = BitReader::new(&buf);
        let carry = ChannelCarry::new(3);
        let block = decode_diff_block(&mut r, PolyOrder::Order1, residuals.len() as u32, &carry, 0)
            .unwrap();
        assert_eq!(block, expected);
    }

    #[test]
    fn carry_update_idempotent_for_full_block_overwrite() {
        // Update once, update again with the same block — the result is
        // identical (no state leakage between consecutive calls).
        let mut c = ChannelCarry::new(3);
        c.update_after_block(&[1, 2, 3, 4, 5]);
        let snapshot_a = c.samples.clone();
        c.update_after_block(&[1, 2, 3, 4, 5]);
        assert_eq!(c.samples, snapshot_a);
    }

    // Make sure the spec-pinned constant agrees with `FNSIZE` to avoid
    // accidental drift between the bit reader and the higher-level
    // block dispatch.
    #[test]
    fn fnsize_is_two_per_spec02_section_4_1() {
        assert_eq!(FNSIZE, 2);
    }

    // -------------------------------------------------------------
    // Mean estimator + DIFF0/ZERO consumer tests (round 4).
    // -------------------------------------------------------------

    #[test]
    fn mean_estimator_zero_init_returns_zero_mu_chan() {
        // spec/05 §2.1 — every channel's mean buffer is zero-initialised.
        // Any positive window length should report mu_chan = 0 before the
        // first block records.
        for len in [1u32, 3, 4, 7, 16] {
            let m = MeanEstimator::new(len);
            assert_eq!(m.len() as u32, len);
            assert!(!m.is_empty());
            assert_eq!(m.mu_chan(), 0);
        }
    }

    #[test]
    fn mean_estimator_disabled_branch_when_meanblocks_zero() {
        // spec/01 §3.5 / spec/05 §2.1 — when H_meanblocks = 0 the
        // estimator is disabled. mu_chan stays at zero across any number
        // of recorded blocks.
        let mut m = MeanEstimator::new(0);
        assert!(m.is_empty());
        assert_eq!(m.mu_chan(), 0);
        m.record_block(&[1000, 2000, 3000]);
        assert_eq!(m.mu_chan(), 0);
        m.record_block(&[-500, -600]);
        assert_eq!(m.mu_chan(), 0);
    }

    #[test]
    fn mean_estimator_per_block_mean_uses_plus_half_bs_bias() {
        // spec/05 §2.5 case 1: mu_blk = trunc_div(sum + bs/2, bs).
        // For block [1, 2, 3] with bs = 3: sum = 6, bs/2 = 1, mu_blk =
        //   trunc_div(6 + 1, 3) = trunc_div(7, 3) = 2.
        // Window length 1 to surface the per-block mean directly via
        // mu_chan after a single record_block.
        let mut m = MeanEstimator::new(1);
        m.record_block(&[1, 2, 3]);
        // sum_of_slots = 2, divisor = 1, +0 bias.
        // mu_chan = trunc_div(2 + 0, 1) = 2.
        assert_eq!(m.mu_chan(), 2);
    }

    #[test]
    fn mean_estimator_per_block_mean_truncates_toward_zero_for_negatives() {
        // spec/05 §2.5 — truncation toward zero (C semantics, not Python
        // floor). The +bs/2 bias is always positive regardless of sign.
        //
        // Block [-1, -2, -3] with bs = 3: sum = -6, bias = 1, num = -5,
        // trunc_div(-5, 3) = -1 (Rust `/` truncates toward zero;
        // div_euclid would give -2).
        let mut m = MeanEstimator::new(1);
        m.record_block(&[-1, -2, -3]);
        assert_eq!(m.mu_chan(), -1);
    }

    #[test]
    fn mean_estimator_sliding_window_evicts_oldest() {
        // Window length 3. Record three distinct blocks then a fourth;
        // the first slot should have been evicted before the fourth is
        // appended.
        //
        // Per-block means under bs=1 (so mu_blk = sum + 0 = sample):
        //   block [10]    -> mu_blk = 10
        //   block [20]    -> mu_blk = 20
        //   block [30]    -> mu_blk = 30
        //   block [40]    -> mu_blk = 40 (evicts 10)
        let mut m = MeanEstimator::new(3);
        m.record_block(&[10]);
        // slots = [0, 0, 10]; sum=10; mu_chan = trunc_div(10+1, 3) = 3.
        assert_eq!(m.mu_chan(), 3);
        m.record_block(&[20]);
        // slots = [0, 10, 20]; sum=30; mu_chan = trunc_div(30+1,3) = 10.
        assert_eq!(m.mu_chan(), 10);
        m.record_block(&[30]);
        // slots = [10, 20, 30]; sum=60; mu_chan = trunc_div(60+1,3)=20.
        assert_eq!(m.mu_chan(), 20);
        m.record_block(&[40]);
        // slots = [20, 30, 40]; sum=90; mu_chan = trunc_div(90+1,3)=30.
        assert_eq!(m.mu_chan(), 30);
    }

    #[test]
    fn mean_estimator_running_mean_uses_plus_half_window_bias() {
        // spec/05 §2.5 case 3 — running mean adds H_meanblocks/2 bias
        // regardless of sign of numerator.
        //
        // Window length 4 with all slots = -3 (sum = -12):
        //   mu_chan = trunc_div(-12 + 2, 4) = trunc_div(-10, 4) = -2.
        // (Floor division would give -3; truncation gives -2.)
        let mut m = MeanEstimator::new(4);
        // Set up the slots by recording four blocks of -3:
        for _ in 0..4 {
            m.record_block(&[-3]);
        }
        assert_eq!(m.mu_chan(), -2);
    }

    #[test]
    fn decode_diff0_with_nonzero_mu_chan_offsets_reconstruction() {
        // spec/05 §2.3 — DIFF0 reconstruction is s(t) = e0(t) + mu_chan.
        // With mu_chan = 100 and residuals [0, 5, -5, 10], the output
        // samples are [100, 105, 95, 110].
        let residuals: Vec<i64> = vec![0, 5, -5, 10];
        let expected: Vec<i32> = vec![100, 105, 95, 110];
        let buf = synth_diff_payload(3, &residuals);
        let mut r = BitReader::new(&buf);
        let carry = ChannelCarry::new(3);
        let block = decode_diff_block(
            &mut r,
            PolyOrder::Order0,
            residuals.len() as u32,
            &carry,
            100,
        )
        .expect("DIFF0 with mu_chan = 100 must decode");
        assert_eq!(block, expected);
    }

    #[test]
    fn decode_diff1_is_mean_invariant_regardless_of_mu_chan() {
        // spec/05 §2 introductory paragraph — only DIFF0 (and ZERO) sees
        // mu_chan; higher-order polynomial-difference predictors are
        // mean-invariant. The same residuals + carry must produce
        // identical output for mu_chan = 0 and mu_chan = 12345.
        let residuals: Vec<i64> = vec![4, 0, -26, 42, -17, -14];
        let buf = synth_diff_payload(3, &residuals);

        let carry = ChannelCarry::new(3);
        let mut r0 = BitReader::new(&buf);
        let blk_zero = decode_diff_block(
            &mut r0,
            PolyOrder::Order1,
            residuals.len() as u32,
            &carry,
            0,
        )
        .unwrap();
        let mut r1 = BitReader::new(&buf);
        let blk_nonzero = decode_diff_block(
            &mut r1,
            PolyOrder::Order1,
            residuals.len() as u32,
            &carry,
            12_345,
        )
        .unwrap();
        assert_eq!(blk_zero, blk_nonzero);
    }

    #[test]
    fn fill_zero_block_emits_mu_chan_replicated() {
        // spec/05 §2.4 — BLOCK_FN_ZERO emits `bs` samples all equal to
        // mu_chan at block start.
        let blk = fill_zero_block(8, 0).expect("zero-mu block must build");
        assert_eq!(blk, vec![0i32; 8]);
        let blk2 = fill_zero_block(5, -42).expect("negative-mu block must build");
        assert_eq!(blk2, vec![-42i32; 5]);
        let blk3 = fill_zero_block(3, 12_345).expect("large positive-mu block must build");
        assert_eq!(blk3, vec![12_345i32; 3]);
    }

    #[test]
    fn fill_zero_block_rejects_oversized_block() {
        let result = fill_zero_block(MAX_BLOCK_SAMPLES + 1, 0);
        assert_eq!(result, Err(Error::BlockTooLarge(MAX_BLOCK_SAMPLES + 1)));
    }

    #[test]
    fn fill_zero_block_rejects_mu_chan_out_of_i32_range() {
        // mu_chan above i32::MAX should narrow-fail. Realistic encoders
        // never produce such values; the boundary is guarded defensively.
        let result = fill_zero_block(4, (i32::MAX as i64) + 1);
        assert_eq!(result, Err(Error::SampleOverflow));
        let result_neg = fill_zero_block(4, (i32::MIN as i64) - 1);
        assert_eq!(result_neg, Err(Error::SampleOverflow));
    }

    #[test]
    fn mean_estimator_records_carries_into_subsequent_diff0() {
        // Two consecutive DIFF0 blocks for the same channel with a
        // window-1 mean estimator. The first block's per-block mean is
        // recorded into the estimator; the second block reads that mean
        // as mu_chan and offsets its residuals by it.
        //
        // Block A residuals [10, 20, 30, 40, 50] under DIFF0 with
        // initial mu_chan = 0 yields samples [10, 20, 30, 40, 50] (sum
        // = 150, bs = 5, per-block mean = trunc_div(150 + 2, 5) =
        // trunc_div(152, 5) = 30).
        //
        // After record_block(blk_a) the window holds [30].
        // mu_chan = trunc_div(30 + 0, 1) = 30.
        //
        // Block B residuals [0, 1, -1, 2, -2] under DIFF0 with
        // mu_chan = 30 yields samples [30, 31, 29, 32, 28].
        let bs = 5u32;
        let residuals_a: Vec<i64> = vec![10, 20, 30, 40, 50];
        let residuals_b: Vec<i64> = vec![0, 1, -1, 2, -2];

        let mut estimator = MeanEstimator::new(1);
        let carry = ChannelCarry::new(3);

        let buf_a = synth_diff_payload(5, &residuals_a); // width 6, fits 50.
        let mut r_a = BitReader::new(&buf_a);
        let blk_a = decode_diff_block(&mut r_a, PolyOrder::Order0, bs, &carry, estimator.mu_chan())
            .expect("DIFF0 block A");
        assert_eq!(blk_a, vec![10, 20, 30, 40, 50]);
        estimator.record_block(&blk_a);
        assert_eq!(estimator.mu_chan(), 30);

        let buf_b = synth_diff_payload(3, &residuals_b); // width 4.
        let mut r_b = BitReader::new(&buf_b);
        let blk_b = decode_diff_block(&mut r_b, PolyOrder::Order0, bs, &carry, estimator.mu_chan())
            .expect("DIFF0 block B");
        assert_eq!(blk_b, vec![30, 31, 29, 32, 28]);
    }

    #[test]
    fn trunc_div_helper_truncates_toward_zero() {
        // Sanity-check the helper directly. Rust's `/` operator already
        // truncates toward zero, but the helper makes the intent explicit
        // at call sites.
        assert_eq!(trunc_div(7, 3), 2);
        assert_eq!(trunc_div(-7, 3), -2);
        assert_eq!(trunc_div(7, -3), -2);
        assert_eq!(trunc_div(-7, -3), 2);
        assert_eq!(trunc_div(0, 5), 0);
        assert_eq!(trunc_div(5, 1), 5);
    }
}
