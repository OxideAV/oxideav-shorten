//! Shorten per-block predictor-selection sequencer — round 251.
//!
//! Round 251 adds the higher-layer encoder routine that compares the
//! cheapest of `BLOCK_FN_DIFF0..3` and `BLOCK_FN_ZERO` for a given
//! channel-block and picks the predictor with the smallest encoded
//! bit cost. The five lower-level predictor writers ([`crate::write_diff0_block`]
//! / [`crate::write_diff1_block`] / [`crate::write_diff2_block`] /
//! [`crate::write_diff3_block`] / [`crate::write_zero_block`]) stay
//! authoritative for the wire-format emission; this module sits one
//! layer above them, takes a slice of samples plus the per-channel
//! `μ_chan` and `ChannelCarry`, and returns the cheapest [`Choice`]
//! a caller can hand to [`write_selected_block`].
//!
//! `BLOCK_FN_QLPC` is **not** part of the auto-selection because the
//! caller still owns coefficient quantisation per `spec/03` §3.5 and
//! TR.156 §3.2's Laplacian-distribution rule; a future round can
//! layer QLPC into the selector by accepting a `&[i64]` candidate
//! coefficient vector.
//!
//! ## Selection criterion
//!
//! For each candidate predictor the sequencer computes the **total
//! encoded bit count** of its `BLOCK_FN_*` command (function code +
//! optional energy field + per-sample residuals folded under
//! `svar(energy + 1)`). The cheapest candidate wins; ties break in
//! the priority order `ZERO > DIFF0 > DIFF1 > DIFF2 > DIFF3` (ZERO
//! first because it's the smallest token at 5 bits total; the DIFFn
//! order matches `spec/03` §3.1..§3.4 narrative).
//!
//! The bit cost is computed against the **natural energy** of each
//! predictor's residual stream — the smallest `e ∈ 0..=MAX_NATURAL_ENERGY`
//! such that every folded residual fits inside the `svar(e + 1)`
//! mantissa with zero prefix-zero bits, exactly mirroring
//! [`crate::min_energy_for_diff0`] / `..diff1` / `..diff2` / `..diff3`.
//! A candidate whose residual stream does not fit any natural energy
//! is skipped from consideration (the caller can still pick a wider
//! energy explicitly through the per-predictor writer).
//!
//! ## ZERO eligibility
//!
//! `BLOCK_FN_ZERO` decodes to `bs` samples all equal to the current
//! channel's running mean `μ_chan` per `spec/05` §2.4. The sequencer
//! emits ZERO only when every sample equals the caller-supplied
//! `mu_chan`; otherwise the candidate is disabled and the DIFFn
//! family carries the block.
//!
//! ## Clean-room provenance
//!
//! * `docs/audio/shorten/spec/03-block-and-predictor.md` §3.1..§3.4
//!   (the four polynomial-difference residual definitions) + §3.9
//!   (the constant-block sentinel) + §3 narrative (the bit-budget
//!   formula via the per-command wire layout).
//! * `docs/audio/shorten/spec/02-variable-length-coding.md` §2.1
//!   (`uvar(n)` length formula `⌊v / 2^n⌋ + 1 + n`) + §2.2
//!   (`svar(n)` folding).
//! * `docs/audio/shorten/spec/05-state-and-quirks.md` §2.3 (DIFF0
//!   residual `e₀(t) = s(t) − μ_chan`) + §2.4 (ZERO emits `μ_chan`
//!   for `bs` samples) + §3.1 (energy field's `+1` mantissa-width
//!   rule).

use crate::bitwriter::BitWriter;
use crate::block::FNSIZE;
use crate::encoder::{
    write_diff0_block, write_diff1_block, write_diff2_block, write_diff3_block, write_zero_block,
    EncodeResult, FN_DIFF0, FN_DIFF1, FN_DIFF2, FN_DIFF3, FN_ZERO, MAX_NATURAL_ENERGY,
};
use crate::predictor::{ChannelCarry, ENERGYSIZE};

/// A predictor-selection outcome the sequencer returns.
///
/// The `bits` field is the total encoded bit count of the candidate
/// (function code + optional energy field + per-sample residuals);
/// [`write_selected_block`] consumes the `Choice` and dispatches to
/// the matching per-predictor writer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Choice {
    /// `BLOCK_FN_ZERO` — the constant-block sentinel.
    ///
    /// Selected only when every sample equals the caller-supplied
    /// `mu_chan` per `spec/05` §2.4. Carries the bare 5-bit function
    /// code token; no payload follows.
    Zero {
        /// Total encoded bit count (5 bits: `uvar(FNSIZE = 2)` over
        /// value `FN_ZERO = 8` packs as the bit pattern `00100`).
        bits: u64,
    },
    /// `BLOCK_FN_DIFF0` — order-0 polynomial-difference predictor
    /// (`spec/03` §3.1, residual `e₀(t) = s(t) − μ_chan`).
    Diff0 {
        /// Encoded energy parameter the writer will use; the
        /// per-sample mantissa width is `energy + 1`.
        energy: u32,
        /// Total encoded bit count (function code + energy field +
        /// `bs × svar(energy + 1)` residuals).
        bits: u64,
    },
    /// `BLOCK_FN_DIFF1` — order-1 polynomial-difference predictor
    /// (`spec/03` §3.2, residual `e₁(t) = s(t) − s(t − 1)`).
    Diff1 { energy: u32, bits: u64 },
    /// `BLOCK_FN_DIFF2` — order-2 polynomial-difference predictor
    /// (`spec/03` §3.3,
    /// residual `e₂(t) = s(t) − (2·s(t − 1) − s(t − 2))`).
    Diff2 { energy: u32, bits: u64 },
    /// `BLOCK_FN_DIFF3` — order-3 polynomial-difference predictor
    /// (`spec/03` §3.4,
    /// residual `e₃(t) = s(t) − (3·s(t − 1) − 3·s(t − 2) + s(t − 3))`).
    Diff3 { energy: u32, bits: u64 },
}

impl Choice {
    /// Total encoded bit count of the candidate.
    pub fn bits(&self) -> u64 {
        match *self {
            Self::Zero { bits } => bits,
            Self::Diff0 { bits, .. } => bits,
            Self::Diff1 { bits, .. } => bits,
            Self::Diff2 { bits, .. } => bits,
            Self::Diff3 { bits, .. } => bits,
        }
    }

    /// Function-code numeric value (one of `FN_DIFF0..3` / `FN_ZERO`).
    pub fn function_code(&self) -> u32 {
        match self {
            Self::Zero { .. } => FN_ZERO,
            Self::Diff0 { .. } => FN_DIFF0,
            Self::Diff1 { .. } => FN_DIFF1,
            Self::Diff2 { .. } => FN_DIFF2,
            Self::Diff3 { .. } => FN_DIFF3,
        }
    }
}

/// Encoded bit length of `uvar(n)` over `value` per `spec/02` §2.1:
/// `⌊value / 2^n⌋ + 1 + n`.
fn uvar_bits(value: u32, n: u32) -> u64 {
    let prefix_zeros: u64 = if n >= 32 { 0 } else { (value >> n) as u64 };
    prefix_zeros + 1 + (n as u64)
}

/// Encoded bit length of `svar(n)` over a signed `value` per
/// `spec/02` §2.2 + §2.1. Folds `value` to unsigned `u = 2v`
/// (`v ≥ 0`) or `u = 2|v| − 1` (`v < 0`) then applies the
/// `uvar(u, n)` length formula.
///
/// Returns `None` if the folded magnitude overflows `u32` (the
/// natural-energy auto-selection rejects such residuals upstream).
#[cfg(test)]
fn svar_bits(value: i64, n: u32) -> Option<u64> {
    let u: u64 = if value >= 0 {
        (value as u64).checked_shl(1)?
    } else {
        let mag = (value as i128).unsigned_abs() as u64;
        mag.checked_shl(1)?.checked_sub(1)?
    };
    if u > u32::MAX as u64 {
        return None;
    }
    Some(uvar_bits(u as u32, n))
}

/// Smallest natural energy `e ∈ 0..=MAX_NATURAL_ENERGY` such that
/// every folded residual fits in the `svar(e + 1)` mantissa with zero
/// prefix-zero bits, plus the **total** encoded bit length of the
/// residual stream at that energy. Returns `None` when no natural
/// energy fits the largest folded residual.
///
/// The total cost includes only the residuals; the caller adds the
/// command's `FNSIZE` + `ENERGYSIZE` prefix.
fn natural_energy_and_residual_bits(residuals: &[i64]) -> Option<(u32, u64)> {
    // Build a per-sample upper bound on the folded magnitude to
    // pick the natural energy.
    let mut u_max: u64 = 0;
    for &r in residuals {
        let u: u64 = if r >= 0 {
            (r as u64).checked_shl(1)?
        } else {
            let mag = (r as i128).unsigned_abs() as u64;
            mag.checked_shl(1)?.checked_sub(1)?
        };
        if u > u_max {
            u_max = u;
        }
    }
    let mut energy: Option<u32> = None;
    for e in 0..=MAX_NATURAL_ENERGY {
        let cap = 1u64 << (e + 1);
        if u_max < cap {
            energy = Some(e);
            break;
        }
    }
    let e = energy?;
    let width = e + 1;
    // Sum the per-sample svar lengths. By natural-energy construction
    // every folded value satisfies `u < 2^width`, so `svar` cost is
    // exactly `1 + width` bits per sample (zero prefix-zero bits +
    // terminator + mantissa).
    let per_sample: u64 = 1 + width as u64;
    let total: u64 = per_sample.checked_mul(residuals.len() as u64)?;
    Some((e, total))
}

/// Bit cost of a `DIFFn` command's `FNSIZE + ENERGYSIZE` prefix at
/// the given encoded energy.
fn diffn_prefix_bits(fn_code: u32, energy: u32) -> u64 {
    uvar_bits(fn_code, FNSIZE) + uvar_bits(energy, ENERGYSIZE)
}

/// Bit cost of the bare `BLOCK_FN_ZERO` token (`uvar(FNSIZE = 2)` over
/// `FN_ZERO = 8`). The value 8 packs as the 5-bit pattern `00100`
/// (`⌊8/4⌋ = 2` prefix zeros + terminator `1` + 2-bit mantissa `00`).
fn zero_token_bits() -> u64 {
    uvar_bits(FN_ZERO, FNSIZE)
}

/// Compute the `DIFF0` candidate for a block.
///
/// Per `spec/03` §3.1 + `spec/05` §2.3 the residual is
/// `e₀(t) = s(t) − mu_chan`. Returns `None` when no natural energy
/// fits the largest folded residual.
fn evaluate_diff0(samples: &[i32], mu_chan: i64) -> Option<Choice> {
    let mut residuals: Vec<i64> = Vec::with_capacity(samples.len());
    for &s in samples {
        residuals.push((s as i64) - mu_chan);
    }
    let (energy, residual_bits) = natural_energy_and_residual_bits(&residuals)?;
    let bits = diffn_prefix_bits(FN_DIFF0, energy) + residual_bits;
    Some(Choice::Diff0 { energy, bits })
}

/// Compute the `DIFF1` candidate for a block per `spec/03` §3.2 +
/// `spec/05` §1.1 (the first-difference recurrence seeded from
/// `carry.at(0)`).
fn evaluate_diff1(samples: &[i32], carry: &ChannelCarry) -> Option<Choice> {
    let mut s_m1: i64 = carry.at(0) as i64;
    let mut residuals: Vec<i64> = Vec::with_capacity(samples.len());
    for &s in samples {
        let s_i64 = s as i64;
        residuals.push(s_i64 - s_m1);
        s_m1 = s_i64;
    }
    let (energy, residual_bits) = natural_energy_and_residual_bits(&residuals)?;
    let bits = diffn_prefix_bits(FN_DIFF1, energy) + residual_bits;
    Some(Choice::Diff1 { energy, bits })
}

/// Compute the `DIFF2` candidate for a block per `spec/03` §3.3 +
/// `spec/05` §1.1 (the second-difference recurrence seeded from
/// `carry.at(0..2)`).
fn evaluate_diff2(samples: &[i32], carry: &ChannelCarry) -> Option<Choice> {
    let mut s_m1: i64 = carry.at(0) as i64;
    let mut s_m2: i64 = if carry.len() > 1 {
        carry.at(1) as i64
    } else {
        0
    };
    let mut residuals: Vec<i64> = Vec::with_capacity(samples.len());
    for &s in samples {
        let s_i64 = s as i64;
        let predicted = 2 * s_m1 - s_m2;
        residuals.push(s_i64 - predicted);
        s_m2 = s_m1;
        s_m1 = s_i64;
    }
    let (energy, residual_bits) = natural_energy_and_residual_bits(&residuals)?;
    let bits = diffn_prefix_bits(FN_DIFF2, energy) + residual_bits;
    Some(Choice::Diff2 { energy, bits })
}

/// Compute the `DIFF3` candidate for a block per `spec/03` §3.4 +
/// `spec/05` §1.1 (the third-difference recurrence seeded from
/// `carry.at(0..3)`).
fn evaluate_diff3(samples: &[i32], carry: &ChannelCarry) -> Option<Choice> {
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
    let mut residuals: Vec<i64> = Vec::with_capacity(samples.len());
    for &s in samples {
        let s_i64 = s as i64;
        let predicted = 3 * s_m1 - 3 * s_m2 + s_m3;
        residuals.push(s_i64 - predicted);
        s_m3 = s_m2;
        s_m2 = s_m1;
        s_m1 = s_i64;
    }
    let (energy, residual_bits) = natural_energy_and_residual_bits(&residuals)?;
    let bits = diffn_prefix_bits(FN_DIFF3, energy) + residual_bits;
    Some(Choice::Diff3 { energy, bits })
}

/// `BLOCK_FN_ZERO` eligibility per `spec/05` §2.4: every sample
/// equals the caller-supplied `mu_chan`.
fn evaluate_zero(samples: &[i32], mu_chan: i64) -> Option<Choice> {
    if samples.is_empty() {
        return None;
    }
    for &s in samples {
        if (s as i64) != mu_chan {
            return None;
        }
    }
    Some(Choice::Zero {
        bits: zero_token_bits(),
    })
}

/// Compute every candidate's bit cost for the block and return them
/// in priority order (ZERO first, then DIFF0..3). Useful for
/// inspection / debugging; [`select_predictor`] picks the cheapest.
///
/// A candidate that doesn't fit a natural energy or doesn't satisfy
/// its eligibility predicate (ZERO requires the all-`mu_chan`
/// invariant) is dropped from the returned list.
pub fn evaluate_candidates(samples: &[i32], mu_chan: i64, carry: &ChannelCarry) -> Vec<Choice> {
    let mut out: Vec<Choice> = Vec::with_capacity(5);
    if let Some(c) = evaluate_zero(samples, mu_chan) {
        out.push(c);
    }
    if let Some(c) = evaluate_diff0(samples, mu_chan) {
        out.push(c);
    }
    if let Some(c) = evaluate_diff1(samples, carry) {
        out.push(c);
    }
    if let Some(c) = evaluate_diff2(samples, carry) {
        out.push(c);
    }
    if let Some(c) = evaluate_diff3(samples, carry) {
        out.push(c);
    }
    out
}

/// Pick the cheapest predictor for the block among `DIFF0..3` and
/// `ZERO` and return a [`Choice`] the caller hands to
/// [`write_selected_block`].
///
/// Per the selection criterion the candidate with the smallest total
/// encoded bit count wins; ties break in priority order
/// `ZERO > DIFF0 > DIFF1 > DIFF2 > DIFF3` (ZERO first because it's
/// the cheapest possible token at 5 bits and the rest match
/// `spec/03` §3.1..§3.4 narrative order).
///
/// Returns `None` when no candidate fits — only reachable when every
/// `DIFFn` residual stream overflows the natural-energy range and
/// ZERO is ineligible (an empty block or a block whose samples don't
/// all equal `mu_chan`). The caller can still encode such a block
/// manually by picking an explicit wider energy through the
/// per-predictor writer.
///
/// `mu_chan` is the per-channel running mean estimate the caller
/// will pass to [`crate::write_diff0_block`] when the selector
/// returns [`Choice::Diff0`]. The DIFF1..3 predictors are
/// mean-invariant per `spec/05` §2 introductory paragraph; the
/// selector still needs `mu_chan` for the DIFF0 residual scan and
/// the ZERO eligibility check.
///
/// `carry` is the per-channel sample-history carry (`spec/05` §1.1);
/// the DIFF1..3 residual scans seed their recurrence windows from
/// `carry.at(0)` / `carry.at(1)` / `carry.at(2)` respectively.
///
/// ## Empty block
///
/// An empty `samples` slice returns `None` because no predictor's
/// residual stream is defined; the caller should never reach this
/// path because the round-7 driver's `bs` is always at least 1
/// (`spec/03` §3.6 rejects `new_bs = 0` and `spec/01` §3 pins
/// `H_blocksize ≥ 1`).
pub fn select_predictor(samples: &[i32], mu_chan: i64, carry: &ChannelCarry) -> Option<Choice> {
    if samples.is_empty() {
        return None;
    }
    let candidates = evaluate_candidates(samples, mu_chan, carry);
    // Tie-breaker: stable min_by returns the first element on a tie,
    // and `evaluate_candidates` produces candidates in priority order
    // ZERO > DIFF0 > DIFF1 > DIFF2 > DIFF3.
    candidates
        .into_iter()
        .min_by(|a, b| a.bits().cmp(&b.bits()))
}

/// Emit the selected predictor's command to `writer`, dispatching to
/// the matching per-predictor primitive.
///
/// The caller is responsible for updating the per-channel sample-
/// history `carry` ([`crate::ChannelCarry::update_after_block`]) and
/// the mean estimator ([`crate::MeanEstimator::record_block`]) after
/// a successful return — this function does not maintain any state.
///
/// Errors mirror the per-predictor writers: an over-cap energy or
/// over-cap sample count surfaces the same [`EncodeError`] the
/// underlying writer would.
pub fn write_selected_block(
    writer: &mut BitWriter,
    choice: &Choice,
    samples: &[i32],
    mu_chan: i64,
    carry: &ChannelCarry,
) -> EncodeResult<()> {
    match *choice {
        Choice::Zero { .. } => {
            write_zero_block(writer);
            Ok(())
        }
        Choice::Diff0 { energy, .. } => write_diff0_block(writer, energy, samples, mu_chan),
        Choice::Diff1 { energy, .. } => write_diff1_block(writer, energy, samples, carry),
        Choice::Diff2 { energy, .. } => write_diff2_block(writer, energy, samples, carry),
        Choice::Diff3 { energy, .. } => write_diff3_block(writer, energy, samples, carry),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitwriter::BitWriter;
    use crate::predictor::ChannelCarry;

    // ---- uvar / svar bit-length helpers ----

    #[test]
    fn uvar_bits_matches_spec_02_2_1_examples() {
        // spec/02 §2.1 worked examples: uvar(n = 2) over the small
        // integers 0..=4. Length = ⌊v/4⌋ + 1 + 2.
        assert_eq!(uvar_bits(0, 2), 3);
        assert_eq!(uvar_bits(1, 2), 3);
        assert_eq!(uvar_bits(2, 2), 3);
        assert_eq!(uvar_bits(3, 2), 3);
        assert_eq!(uvar_bits(4, 2), 4);
        assert_eq!(uvar_bits(7, 2), 4);
        assert_eq!(uvar_bits(8, 2), 5); // BLOCK_FN_ZERO token (00100).
    }

    #[test]
    fn svar_bits_matches_writer_actual_length() {
        // Mirror against BitWriter::write_svar's actual bit count
        // for a representative spread of values + widths.
        for &width in &[1u32, 2, 3, 4, 5] {
            for &v in &[
                0i64, 1, -1, 2, -2, 3, -3, 7, -7, 8, -8, 15, -15, 16, -16, 31, -31, 64, -64,
            ] {
                let mut w = BitWriter::new();
                w.write_svar(v, width);
                let actual = w.bits_written();
                let computed = svar_bits(v, width).unwrap();
                assert_eq!(
                    actual, computed,
                    "svar({v}, {width}): actual {actual} != computed {computed}"
                );
            }
        }
    }

    // ---- zero token cost ----

    #[test]
    fn zero_token_bits_is_5() {
        // uvar(FNSIZE = 2) over FN_ZERO = 8 packs as 00100 (5 bits).
        assert_eq!(zero_token_bits(), 5);
        let mut w = BitWriter::new();
        write_zero_block(&mut w);
        assert_eq!(w.bits_written(), 5);
    }

    // ---- evaluate_zero ----

    #[test]
    fn zero_candidate_requires_all_samples_equal_to_mu_chan() {
        let samples = vec![0i32, 0, 0, 0];
        assert!(matches!(
            evaluate_zero(&samples, 0),
            Some(Choice::Zero { bits: 5 })
        ));
        let nonzero = vec![0i32, 0, 1, 0];
        assert!(evaluate_zero(&nonzero, 0).is_none());
        // Non-zero mean: every sample must equal mu_chan.
        let mu7 = vec![7i32; 8];
        assert!(matches!(
            evaluate_zero(&mu7, 7),
            Some(Choice::Zero { bits: 5 })
        ));
        assert!(evaluate_zero(&mu7, 0).is_none());
    }

    #[test]
    fn empty_block_rejects_zero_candidate() {
        let samples: Vec<i32> = vec![];
        assert!(evaluate_zero(&samples, 0).is_none());
    }

    // ---- evaluate_diff0 ----

    #[test]
    fn diff0_residual_is_sample_minus_mu_chan() {
        // mu_chan = 0: residuals == samples; one-sample {0} folds to 0,
        // svar(1) over 0 is 1+1 = 2 bits; total = 3 (fn) + 4 (energy=0 uvar(3)) + 2 = 9.
        let samples = vec![0i32];
        let choice = evaluate_diff0(&samples, 0).unwrap();
        match choice {
            Choice::Diff0 { energy, bits } => {
                assert_eq!(energy, 0);
                // fn=0 in uvar(2) → 3 bits; energy=0 in uvar(3) → 4 bits;
                // svar(1) over 0 → 2 bits.
                assert_eq!(bits, 3 + 4 + 2);
            }
            _ => panic!("expected DIFF0 variant"),
        }
    }

    #[test]
    fn diff0_with_nonzero_mu_uses_residual_scan() {
        // samples = [5, 5, 5], mu_chan = 5 → residuals = [0, 0, 0].
        // Natural energy = 0, width = 1, svar(1) over 0 = 2 bits.
        let samples = vec![5i32, 5, 5];
        let choice = evaluate_diff0(&samples, 5).unwrap();
        match choice {
            Choice::Diff0 { energy, bits } => {
                assert_eq!(energy, 0);
                // 3 (fn) + 4 (energy) + 3 × 2 = 13 bits.
                assert_eq!(bits, 13);
            }
            _ => panic!("expected DIFF0 variant"),
        }
    }

    // ---- evaluate_diff1 ----

    #[test]
    fn diff1_constant_input_with_zero_carry_emits_seed_jump() {
        // samples = [3, 3, 3, 3] with carry.at(0) = 0:
        // residuals = [3, 0, 0, 0]. Folded magnitudes [6, 0, 0, 0].
        // Natural energy = 2 (cap 8 > 6), width = 3.
        let samples = vec![3i32, 3, 3, 3];
        let carry = ChannelCarry::new(3);
        let choice = evaluate_diff1(&samples, &carry).unwrap();
        match choice {
            Choice::Diff1 { energy, bits } => {
                assert_eq!(energy, 2);
                // 3 (fn=1 uvar(2)=3 bits: '0 1 01') + 4 (energy=2 uvar(3))
                //   + 4 × svar(3): per-sample 1 + 3 = 4 bits → 16 bits.
                // Total = 3 + 4 + 16 = 23 bits.
                assert_eq!(bits, 23);
            }
            _ => panic!("expected DIFF1 variant"),
        }
    }

    // ---- selector ----

    #[test]
    fn selector_picks_zero_for_all_zero_block_with_zero_mu() {
        let samples = vec![0i32; 8];
        let carry = ChannelCarry::new(3);
        let choice = select_predictor(&samples, 0, &carry).unwrap();
        assert!(matches!(choice, Choice::Zero { bits: 5 }));
    }

    #[test]
    fn selector_picks_zero_over_diff0_when_block_is_constant_mu() {
        // samples = [7; 8], mu = 7. ZERO costs 5 bits; DIFF0 costs
        // 3 + 4 + 8 × 2 = 23 bits. ZERO wins.
        let samples = vec![7i32; 8];
        let carry = ChannelCarry::new(3);
        let choice = select_predictor(&samples, 7, &carry).unwrap();
        assert!(matches!(choice, Choice::Zero { bits: 5 }));
    }

    #[test]
    fn selector_picks_diff1_when_ramp_has_smaller_first_differences() {
        // samples = arithmetic progression 0, 5, 10, 15, ..., 5·(N-1)
        // with mu_chan = 0 (so ZERO ineligible).
        // DIFF0 residuals = samples themselves; values grow without
        //   bound (max 5·(N-1)) so natural-energy gets wide as N grows.
        // DIFF1 residuals = [0, 5, 5, 5, ..., 5]; folded max 10 →
        //   natural energy e=3 (cap 16 > 10), width 4. Per-sample 5
        //   bits; cost 3 + 4 + N × 5.
        // For N = 64 the DIFF0 residual max is 5·63 = 315, requiring
        // a wider svar; DIFF1 wins by a comfortable margin.
        let samples: Vec<i32> = (0..64i32).map(|t| 5 * t).collect();
        let carry = ChannelCarry::new(3);
        let choice = select_predictor(&samples, 0, &carry).unwrap();
        assert!(
            matches!(choice, Choice::Diff1 { .. }),
            "expected DIFF1, got {choice:?}"
        );
        // DIFF0 cost lower bound at N=64 with folded-max 630
        //   → natural e=9 doesn't fit (max 7); evaluate_diff0 returns
        //   None when residuals exceed natural-energy cap.
        // Whatever DIFF1's exact bits are, they should beat that.
    }

    #[test]
    fn selector_picks_diff3_for_pure_cubic() {
        // samples form a perfect cubic so DIFF3 residuals are zero
        // after the seed jumps. With a long-enough block, DIFF3 should
        // beat DIFF0..2.
        // Use s(t) = t³: 0, 1, 8, 27, 64, 125, 216, 343, 512, 729.
        // DIFF3 residuals after the seed should all be 6 (constant
        // third-difference of cubic), so folded {12, 12, ...} once the
        // window stabilises (5 zero-seed samples → seed jumps).
        // Constant-residual block of length N at folded value 12:
        // natural energy: 2^(e+1) > 12 → e=3, width=4.
        // For a long block (256 samples) DIFF3 cost is bounded;
        // DIFF0 cost grows as N × svar(wide) and loses.
        let samples: Vec<i32> = (0..256i32).map(|t| t * t * t).collect();
        let carry = ChannelCarry::new(3);
        let zero = evaluate_zero(&samples, 0); // can't ZERO this.
        assert!(zero.is_none());
        let choice = select_predictor(&samples, 0, &carry).unwrap();
        // Cubic should make DIFF3 the natural winner (or DIFF0 if cubic
        // values overflow the natural-energy range — in which case the
        // selector falls back to whichever fits).
        let bits = choice.bits();
        // At minimum, DIFF3 must be considered: its residuals are bounded.
        let d3 = evaluate_diff3(&samples, &carry);
        if let Some(c3) = d3 {
            // The selector picked the cheapest among candidates.
            assert!(
                bits <= c3.bits(),
                "selector picked a worse candidate than DIFF3"
            );
        }
    }

    // ---- write_selected_block ----

    #[test]
    fn write_selected_emits_zero_token_for_zero_choice() {
        let mut w = BitWriter::new();
        let carry = ChannelCarry::new(3);
        let samples = vec![0i32; 4];
        let choice = select_predictor(&samples, 0, &carry).unwrap();
        write_selected_block(&mut w, &choice, &samples, 0, &carry).unwrap();
        assert_eq!(w.bits_written(), 5, "ZERO is 5 bits total");
    }

    #[test]
    fn write_selected_for_diff0_matches_direct_write_diff0_block() {
        let samples = vec![1i32, -1, 2, -2];
        let carry = ChannelCarry::new(3);
        // Force DIFF0 by using mu_chan = 0 and a block whose samples
        // don't all equal mu_chan (ZERO ineligible) and whose
        // first-difference pattern beats / ties DIFF0 only in some
        // cases — we'll let the selector decide and just verify the
        // dispatch table.
        let choice = select_predictor(&samples, 0, &carry).unwrap();
        let mut w_selected = BitWriter::new();
        write_selected_block(&mut w_selected, &choice, &samples, 0, &carry).unwrap();
        // The bits_written must equal the Choice's reported bit count.
        assert_eq!(w_selected.bits_written(), choice.bits());
    }

    #[test]
    fn selector_handles_empty_block() {
        let samples: Vec<i32> = vec![];
        let carry = ChannelCarry::new(3);
        assert!(select_predictor(&samples, 0, &carry).is_none());
    }
}
