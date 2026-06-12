//! Shorten per-block predictor-selection sequencer — rounds 251 + 254
//! + 266 + 282.
//!
//! Round 251 added the higher-layer encoder routine that compares the
//! cheapest of `BLOCK_FN_DIFF0..3` and `BLOCK_FN_ZERO` for a given
//! channel-block and picks the predictor with the smallest encoded
//! bit cost. Round 254 lifts the per-predictor cost from the natural
//! energy (zero prefix-zero bits) to the **Rice-n statistical optimum**
//! of TR.156 §3.3, sweeping every encoded energy in `0..=MAX_NATURAL_ENERGY`
//! and picking whichever minimises the residual stream's total
//! `⌊u / 2^width⌋ + 1 + width` per-sample cost. Round 266 extends the
//! selector with optional **`BLOCK_FN_QLPC`** auto-selection via the
//! new entry points [`select_predictor_with_qlpc`] and
//! [`evaluate_candidates_with_qlpc`]: a caller supplies a candidate
//! quantised-coefficient vector and the selector scores it under the
//! same Rice-n cost metric as the DIFFn family, picking QLPC when its
//! residual stream wins on bit cost. The six lower-level predictor
//! writers ([`crate::write_diff0_block`] /
//! [`crate::write_diff1_block`] / [`crate::write_diff2_block`] /
//! [`crate::write_diff3_block`] / [`crate::write_zero_block`] /
//! [`crate::write_qlpc_block`]) stay authoritative for the wire-format
//! emission; this module sits one layer above them, takes a slice of
//! samples plus the per-channel `μ_chan` and `ChannelCarry` (plus the
//! optional QLPC candidate), and returns the cheapest [`Choice`] a
//! caller can hand to [`write_selected_block`].
//!
//! Round 282 closes the remaining gap: **QLPC auto-derivation**. The
//! new entry points [`select_predictor_auto`] /
//! [`evaluate_candidates_auto`] no longer need a caller-supplied
//! coefficient vector — [`derive_qlpc_coefs`] derives the quantised
//! per-block coefficients itself (least-squares normal equations over
//! the block's carry-seeded prediction contexts, rounded into the
//! plain-signed-integer coefficient domain `spec/03` §3.5 pins for the
//! wire), and [`derive_qlpc_candidate`] runs the per-block **order
//! search** over `0..=max_lpc_order` (zero-anchored per `spec/02`
//! §4.3's TR.156 anchor "for version 2 and above, the search is
//! started at zero order"), scoring every order's candidate under the
//! full cost model — function code + `uvar(LPCQSIZE)` order field +
//! `order × svar(LPCQUANT)` coefficient-transmission overhead +
//! energy field + Rice-n-optimal residual stream — so the selector
//! picks QLPC only when it is genuinely cheapest end-to-end.
//!
//! ## Selection criterion
//!
//! For each candidate predictor the sequencer computes the **total
//! encoded bit count** of its `BLOCK_FN_*` command (function code +
//! optional `order` + `coefs` fields + optional energy field + per-
//! sample residuals folded under `svar(energy + 1)`). The cheapest
//! candidate wins; ties break in the priority order
//! `ZERO > DIFF0 > DIFF1 > DIFF2 > DIFF3 > QLPC` (ZERO first because
//! it's the smallest token at 5 bits total; the DIFFn order matches
//! `spec/03` §3.1..§3.4 narrative; QLPC last because of its larger
//! fixed overhead — order field + coefficient field — so on an exact
//! cost tie a smaller-fixed-overhead command wins).
//!
//! The per-predictor cost is computed at the **Rice-n statistical
//! optimum** of TR.156 §3.3 over `e ∈ 0..=MAX_NATURAL_ENERGY`. The
//! sequencer ranges every encoded energy in the natural band and picks
//! whichever minimises the residual stream's `⌊u / 2^width⌋ + 1 +
//! width` per-sample cost summed across the block. For tightly-
//! distributed residual streams the optimum coincides with the
//! zero-prefix-bits natural energy of [`crate::min_energy_for_diff0`]
//! / `..diff1` / `..diff2` / `..diff3`; for streams with a single
//! outlier (e.g. a fresh-carry seed jump) the optimum picks a tighter
//! mantissa and lets the outlier eat the prefix-zero cost because the
//! savings on the other samples dominate.
//!
//! A candidate whose folded magnitudes overflow `u64` (`v` near
//! `i64::MIN`'s magnitude) is skipped; the caller can still pick a
//! wider energy explicitly through the per-predictor writer.
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
//!   (the four polynomial-difference residual definitions) + §3.5
//!   (the `BLOCK_FN_QLPC` wire layout — order + coefficients +
//!   energy + residuals; the predictor recurrence
//!   `s(t) = Σᵢ aᵢ · s(t − i) + e_QLPC(t)`) + §3.9 (the constant-
//!   block sentinel) + §3 narrative (the bit-budget formula via the
//!   per-command wire layout).
//! * `docs/audio/shorten/spec/02-variable-length-coding.md` §2.1
//!   (`uvar(n)` length formula `⌊v / 2^n⌋ + 1 + n` — the source of
//!   the Rice-n cost metric) + §2.2 (`svar(n)` folding) + §4.2
//!   (per-block Rice-parameter width's TR.156 §3.3 anchor) + §4.3
//!   (`LPCQSIZE = 2`, the per-block-order field width) + §4.4
//!   (`LPCQUANT = 2`, the per-coefficient field width).
//! * `docs/audio/shorten/spec/05-state-and-quirks.md` §2 (the
//!   mean-invariance argument that anchors QLPC's no-`mu_chan`
//!   signature alongside DIFF1..3) + §2.3 (DIFF0 residual
//!   `e₀(t) = s(t) − μ_chan`) + §2.4 (ZERO emits `μ_chan` for `bs`
//!   samples) + §3.1 (energy field's `+1` mantissa-width rule and the
//!   consistency note that the smallest sensible width is 1 not 0,
//!   anchoring the search at `e = 0`).
//!
//! Round-282 auto-derivation anchors:
//!
//! * `spec/03` §3.5 — the predictor model
//!   `ŝ(t) = Σᵢ₌₁..order aᵢ · s(t − i)` the least-squares objective
//!   minimises; the integer quantisation domain ("each `coef` is a
//!   small signed integer that the decoder applies in the predictor
//!   recurrence without scaling") that makes nearest-integer rounding
//!   the projection onto the wire's representable coefficient set;
//!   the `0..H_maxlpcorder` per-block order range; the
//!   `H_maxlpcorder = 0` ⇒ no-QLPC rule.
//! * `spec/02` §4.3 — the zero-anchored order search ("for version 2
//!   and above, the search is started at zero order") + §4.4 (the
//!   `svar(LPCQUANT = 2)` coefficient-transmission cost the order
//!   search charges each candidate).
//! * `spec/05` §1.1 — the carry-seeded prediction contexts the
//!   normal-equation accumulation mirrors from [`qlpc_residuals`].

use crate::bitwriter::BitWriter;
use crate::block::FNSIZE;
use crate::encoder::{
    optimal_energy_for_residuals, qlpc_residuals, residual_bits_at_energy, write_diff0_block,
    write_diff1_block, write_diff2_block, write_diff3_block, write_qlpc_block, write_zero_block,
    EncodeResult, FN_DIFF0, FN_DIFF1, FN_DIFF2, FN_DIFF3, FN_QLPC, FN_ZERO, MAX_QLPC_ORDER,
};
use crate::predictor::{ChannelCarry, ENERGYSIZE, LPCQSIZE, LPCQUANT};

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
    /// `BLOCK_FN_QLPC` — quantised-LPC predictor (`spec/03` §3.5,
    /// residual `e_QLPC(t) = s(t) − Σᵢ coefs[i] · s(t − i − 1)`).
    ///
    /// The variant carries the caller-supplied quantised coefficient
    /// vector so [`write_selected_block`] can dispatch the command
    /// without a separate coefficient parameter. The variant's
    /// `energy` is the Rice-n statistical optimum over the QLPC
    /// residual stream (TR.156 §3.3), matching the DIFFn family's
    /// `energy` semantics.
    ///
    /// Selected only when [`select_predictor_with_qlpc`] or
    /// [`evaluate_candidates_with_qlpc`] is called with
    /// `qlpc_candidate = Some(coefs)`; the round-251 entry points
    /// [`select_predictor`] / [`evaluate_candidates`] never produce
    /// this variant.
    Qlpc {
        /// Per-block quantised LPC coefficients (`coefs[i] = aᵢ₊₁`,
        /// applied to `s(t − i − 1)` per `spec/03` §3.5). Carried by
        /// value so the dispatch path is self-contained — the caller
        /// supplies the vector once to the selector, and
        /// [`write_selected_block`] consumes it from the variant
        /// without a second hand-off.
        coefs: Vec<i64>,
        /// Encoded energy parameter the writer will use; the
        /// per-sample mantissa width is `energy + 1`.
        energy: u32,
        /// Total encoded bit count (function code + per-block
        /// `uvar(LPCQSIZE)` order field + `order × svar(LPCQUANT)`
        /// coefficient fields + `uvar(ENERGYSIZE)` energy field +
        /// `bs × svar(energy + 1)` residuals).
        bits: u64,
    },
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
            Self::Qlpc { bits, .. } => bits,
        }
    }

    /// Function-code numeric value (one of `FN_DIFF0..3` / `FN_ZERO`
    /// / `FN_QLPC`).
    pub fn function_code(&self) -> u32 {
        match self {
            Self::Zero { .. } => FN_ZERO,
            Self::Diff0 { .. } => FN_DIFF0,
            Self::Diff1 { .. } => FN_DIFF1,
            Self::Diff2 { .. } => FN_DIFF2,
            Self::Diff3 { .. } => FN_DIFF3,
            Self::Qlpc { .. } => FN_QLPC,
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
/// natural-energy auto-selection rejects such residuals upstream) or
/// if the code's prefix-zero run would exceed the decoder's `uvar`
/// prefix cap (32 leading zeros — the decoder's `read_uvar` rejects
/// longer runs, so the value is unrepresentable on a decodable wire
/// at this width and the candidate carrying it must be dropped).
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
    let prefix = if n >= 32 { 0 } else { u >> n };
    if prefix > crate::bitreader::UVAR_PREFIX_CAP as u64 {
        return None;
    }
    Some(uvar_bits(u as u32, n))
}

/// Optimal Rice energy `e ∈ 0..=MAX_NATURAL_ENERGY` for the residual
/// stream per TR.156 §3.3's optimal-`n` rate metric, plus the **total**
/// encoded bit length of the residual stream at that energy. Returns
/// `None` when no `e` in the natural band yields a finite cost (every
/// folded value overflows) or when `residuals` is empty.
///
/// The cost metric is the same `⌊u / 2^width⌋ + 1 + width` per-sample
/// formula `svar(width)` uses, summed across the block; the optimum
/// trades a wider mantissa against the prefix-zero count. For tightly-
/// distributed residual streams (most blocks the encoder will see) the
/// optimum coincides with the natural energy of
/// [`crate::min_energy_for_diff0`] / `..diff1` / `..diff2` / `..diff3`;
/// for sparse streams with a single outlier — e.g. the seed-jump
/// residual at a fresh DIFF1/2/3 carry — the optimum chooses a
/// tighter mantissa and lets the outlier eat the prefix-zero cost
/// because the savings on the other samples dominate.
///
/// The total cost includes only the residuals; the caller adds the
/// command's `FNSIZE` + `ENERGYSIZE` prefix.
fn optimal_energy_and_residual_bits(residuals: &[i64]) -> Option<(u32, u64)> {
    let e = optimal_energy_for_residuals(residuals)?;
    let cost = residual_bits_at_energy(residuals, e)?;
    Some((e, cost))
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
    let (energy, residual_bits) = optimal_energy_and_residual_bits(&residuals)?;
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
    let (energy, residual_bits) = optimal_energy_and_residual_bits(&residuals)?;
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
    let (energy, residual_bits) = optimal_energy_and_residual_bits(&residuals)?;
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
    let (energy, residual_bits) = optimal_energy_and_residual_bits(&residuals)?;
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

/// Compute the `QLPC` candidate for a block per `spec/03` §3.5.
///
/// Wire layout (matches [`crate::write_qlpc_block`]): `uvar(FNSIZE)`
/// over `FN_QLPC` + `uvar(LPCQSIZE)` over `coefs.len()` +
/// `coefs.len() × svar(LPCQUANT)` over each quantised coefficient +
/// `uvar(ENERGYSIZE)` over the chosen energy + `bs × svar(energy + 1)`
/// over the residuals. The energy is the Rice-n statistical optimum
/// over the residual stream (TR.156 §3.3, identical to the DIFFn
/// family's selection rule).
///
/// Returns `None` when:
/// * `samples.is_empty()` — no residuals to score.
/// * `coefs.len()` exceeds [`MAX_QLPC_ORDER`] or `carry.len()` (matches
///   [`crate::write_qlpc_block`]'s `LpcOrderTooLarge` precondition).
/// * Any quantised coefficient overflows the `svar(LPCQUANT)` folding
///   (the writer would reject it; the sequencer must skip the
///   candidate so the selector falls back to DIFFn).
/// * Computing the residual stream surfaces an error from
///   [`qlpc_residuals`] (mirrors the writer's eligibility).
/// * No `e ∈ 0..=MAX_NATURAL_ENERGY` yields a finite residual bit
///   count (every folded residual would overflow).
///
/// `coefs` is cloned into the returned `Choice::Qlpc` so the variant
/// is self-contained for the [`write_selected_block`] dispatch path —
/// the caller hands the vector to the selector once, and the writer
/// reads it back from the variant.
fn evaluate_qlpc(samples: &[i32], coefs: &[i64], carry: &ChannelCarry) -> Option<Choice> {
    if samples.is_empty() {
        return None;
    }
    let order = coefs.len();
    if order > MAX_QLPC_ORDER as usize || order > carry.len() {
        return None;
    }
    // Cost of the per-block-order field: uvar(LPCQSIZE = 2) over
    // `order`. The decoder reads it before the coefficient stream, so
    // an order outside the field's representable range would also be
    // a wire-format violation — but the MAX_QLPC_ORDER cap above
    // pre-emptively bounds us well under the field's natural
    // representation.
    let order_field_bits = uvar_bits(order as u32, LPCQSIZE);
    // Cost of the coefficient stream: each coefficient is svar(LPCQUANT).
    // A coefficient whose folded magnitude overflows u32 is skipped —
    // the writer would surface CoefficientOutOfRange; the sequencer
    // mirrors that by dropping the candidate so the selector falls
    // back to DIFFn.
    let mut coef_bits: u64 = 0;
    for &c in coefs {
        coef_bits = coef_bits.checked_add(svar_bits(c, LPCQUANT)?)?;
    }
    // Residual stream: e_QLPC(t) = s(t) − Σᵢ coefs[i] · s(t − i − 1).
    let residuals = qlpc_residuals(samples, coefs, carry).ok()?;
    let (energy, residual_bits) = optimal_energy_and_residual_bits(&residuals)?;
    // Function-code + energy prefix matches the DIFFn family.
    let fn_prefix_bits = uvar_bits(FN_QLPC, FNSIZE);
    let energy_field_bits = uvar_bits(energy, ENERGYSIZE);
    let bits = fn_prefix_bits
        .checked_add(order_field_bits)?
        .checked_add(coef_bits)?
        .checked_add(energy_field_bits)?
        .checked_add(residual_bits)?;
    Some(Choice::Qlpc {
        coefs: coefs.to_vec(),
        energy,
        bits,
    })
}

/// Upper bound on the order the **auto-derivation** order search of
/// [`derive_qlpc_candidate`] will visit, independent of the caller's
/// `max_lpc_order`.
///
/// The wire format itself admits orders up to the stream header's
/// `H_maxlpcorder` (`spec/03` §3.5, bounded implementation-side by
/// [`MAX_QLPC_ORDER`]), and the explicit
/// [`select_predictor_with_qlpc`] path accepts any caller-derived
/// vector up to that cap. The auto-derivation path additionally caps
/// its search because the normal-equation solve is `O(order³)` per
/// order — a deliberately bounded encoder-side budget, not a format
/// limit. TR.156's man-page bounds practical LPC orders to small
/// integers (Appendix §"-p prediction order": "transmitting each
/// filter coefficient requires about 7 bits", so high orders pay a
/// linearly growing per-block overhead the residual savings rarely
/// recover).
pub const MAX_QLPC_AUTO_ORDER: u32 = 32;

/// Past sample `s(t − i)` (`i ≥ 1`) for position `t` of the block,
/// seeded from the per-channel carry for the pre-block window exactly
/// as [`qlpc_residuals`] seeds it: `carry.at(k)` is `s(−1 − k)` per
/// `spec/05` §1.1.
fn qlpc_past_sample(samples: &[i32], carry: &ChannelCarry, t: usize, i: usize) -> f64 {
    debug_assert!(i >= 1);
    if t >= i {
        samples[t - i] as f64
    } else {
        carry.at(i - t - 1) as f64
    }
}

/// Derive a quantised per-block LPC coefficient vector of exactly
/// `order` coefficients for the block, per `spec/03` §3.5's predictor
/// model `ŝ(t) = Σᵢ₌₁..order aᵢ · s(t − i)`.
///
/// The derivation minimises the encode-side residual energy
/// `Σₜ (s(t) − Σᵢ aᵢ · s(t − i))²` over the **actual** prediction
/// contexts the residual computation uses — the per-sample history
/// windows seeded from the channel carry (`spec/05` §1.1) and slid
/// against the block's own samples, exactly mirroring
/// [`qlpc_residuals`] — by solving the least-squares normal equations
/// with Gaussian elimination (partial pivoting). The real-valued
/// solution is then **quantised** by rounding each coefficient to the
/// nearest integer: `spec/03` §3.5 pins the wire's quantisation domain
/// as plain small signed integers "that the decoder applies in the
/// predictor recurrence without scaling", so nearest-integer rounding
/// is the projection onto the representable coefficient set. (The
/// integer vector is what the residuals are recomputed against, so
/// any rounding loss shows up only as residual cost — never as a
/// reconstruction error; the scheme stays lossless by construction.)
///
/// Returns `None` when the block cannot support the derivation:
///
/// * `order == 0` (nothing to derive; the order-0 QLPC candidate has
///   an empty coefficient vector and needs no derivation) or
///   `samples.len() < order` (fewer equations than unknowns).
/// * `order` exceeds [`MAX_QLPC_ORDER`] or `carry.len()` (the
///   candidate could not be written anyway — mirrors
///   [`crate::write_qlpc_block`]'s `LpcOrderTooLarge` precondition).
/// * The normal-equation system is singular or numerically degenerate
///   (e.g. an all-zero block, or a block whose past-sample lag
///   vectors are linearly dependent because a lower order already
///   models it exactly — the order search visits that lower order
///   separately).
/// * A solved coefficient is non-finite or too large to be a sane
///   integer coefficient (the cost model would reject it regardless).
pub fn derive_qlpc_coefs(samples: &[i32], carry: &ChannelCarry, order: usize) -> Option<Vec<i64>> {
    if order == 0 || samples.len() < order {
        return None;
    }
    if order > MAX_QLPC_ORDER as usize || order > carry.len() {
        return None;
    }
    // Accumulate the normal equations M·a = b with
    //   M[i][j] = Σₜ s(t − i − 1) · s(t − j − 1)
    //   b[i]    = Σₜ s(t) · s(t − i − 1)
    // over every block position t, history seeded from the carry.
    let n = samples.len();
    let mut m = vec![vec![0.0f64; order]; order];
    let mut b = vec![0.0f64; order];
    let mut row = vec![0.0f64; order];
    for t in 0..n {
        for (i, slot) in row.iter_mut().enumerate() {
            *slot = qlpc_past_sample(samples, carry, t, i + 1);
        }
        let s_t = samples[t] as f64;
        for i in 0..order {
            b[i] += s_t * row[i];
            for j in i..order {
                m[i][j] += row[i] * row[j];
            }
        }
    }
    // Mirror the upper triangle (the accumulation above only filled
    // j ≥ i; M is symmetric by construction).
    for i in 1..order {
        let (head, tail) = m.split_at_mut(i);
        let row_i = &mut tail[0];
        for (j, head_row) in head.iter().enumerate() {
            row_i[j] = head_row[i];
        }
    }
    // Scale-relative singularity threshold.
    let mut scale = 0.0f64;
    for r in &m {
        for &v in r {
            if v.abs() > scale {
                scale = v.abs();
            }
        }
    }
    if scale == 0.0 {
        return None; // all-zero contexts; ZERO / DIFF0 carry the block.
    }
    let pivot_eps = scale * 1e-12;
    // Gaussian elimination with partial pivoting on [M | b].
    for col in 0..order {
        let mut pivot_row = col;
        let mut pivot_mag = m[col][col].abs();
        for (r, mr) in m.iter().enumerate().skip(col + 1) {
            if mr[col].abs() > pivot_mag {
                pivot_mag = mr[col].abs();
                pivot_row = r;
            }
        }
        if pivot_mag <= pivot_eps {
            return None; // singular / degenerate at this order.
        }
        if pivot_row != col {
            m.swap(col, pivot_row);
            b.swap(col, pivot_row);
        }
        for r in (col + 1)..order {
            // `r > col`, so splitting at `r` puts the pivot row in
            // the head and the target row at the start of the tail.
            let (head, tail) = m.split_at_mut(r);
            let pivot_row = &head[col];
            let target_row = &mut tail[0];
            let factor = target_row[col] / pivot_row[col];
            if factor != 0.0 {
                for (t, &p) in target_row.iter_mut().zip(pivot_row.iter()).skip(col) {
                    *t -= factor * p;
                }
                b[r] -= factor * b[col];
            }
        }
    }
    // Back-substitution.
    let mut a = vec![0.0f64; order];
    for i in (0..order).rev() {
        let mut acc = b[i];
        for j in (i + 1)..order {
            acc -= m[i][j] * a[j];
        }
        a[i] = acc / m[i][i];
    }
    // Quantise into the spec/03 §3.5 integer coefficient domain.
    let mut coefs: Vec<i64> = Vec::with_capacity(order);
    for &v in &a {
        if !v.is_finite() || v.abs() > 1e15 {
            return None;
        }
        coefs.push(v.round() as i64);
    }
    Some(coefs)
}

/// Run the per-block **QLPC order search** and return the cheapest
/// auto-derived `BLOCK_FN_QLPC` candidate, or `None` when no order in
/// the search band yields a well-formed candidate.
///
/// The search visits every order in `0..=cap` where `cap` is the
/// minimum of the caller's `max_lpc_order` (normally the stream
/// header's `H_maxlpcorder`, which `spec/03` §3.5 pins as the
/// per-block order's upper bound), `carry.len()`,
/// [`MAX_QLPC_ORDER`], and the auto-derivation budget
/// [`MAX_QLPC_AUTO_ORDER`]. The search is **zero-anchored** per
/// `spec/02` §4.3's TR.156 anchor ("for version 2 and above, the
/// search is started at zero order"): the order-0 candidate carries
/// an empty coefficient vector (it predicts zero without consulting
/// the mean estimator, so it is *not* residual-identical to DIFF0
/// when `μ_chan ≠ 0`). Orders ≥ 1 derive their quantised coefficient
/// vector through [`derive_qlpc_coefs`].
///
/// Every candidate is scored under the full per-command cost model of
/// `evaluate_qlpc` — function code + `uvar(LPCQSIZE)` order field +
/// `order × svar(LPCQUANT)` coefficient-transmission overhead +
/// `uvar(ENERGYSIZE)` energy field + Rice-n-optimal residual stream
/// (TR.156 §3.3) — so a higher order wins only when its residual
/// savings genuinely repay the extra coefficient bits. Cost ties
/// break toward the **lower** order (smaller fixed overhead is the
/// established tie-break direction of the round-266 priority order).
///
/// `max_lpc_order = 0` returns `None`: per `spec/03` §3.5 a
/// `BLOCK_FN_QLPC` block is well-formed only when `H_maxlpcorder > 0`,
/// so a polynomial-difference-only stream must never grow a QLPC
/// command.
pub fn derive_qlpc_candidate(
    samples: &[i32],
    carry: &ChannelCarry,
    max_lpc_order: u32,
) -> Option<Choice> {
    if samples.is_empty() || max_lpc_order == 0 {
        return None;
    }
    let cap = max_lpc_order
        .min(MAX_QLPC_AUTO_ORDER)
        .min(MAX_QLPC_ORDER)
        .min(carry.len() as u32);
    let mut best: Option<Choice> = None;
    for order in 0..=cap {
        let candidate = if order == 0 {
            evaluate_qlpc(samples, &[], carry)
        } else {
            match derive_qlpc_coefs(samples, carry, order as usize) {
                Some(coefs) => evaluate_qlpc(samples, &coefs, carry),
                None => None,
            }
        };
        if let Some(c) = candidate {
            let improves = match &best {
                Some(current) => c.bits() < current.bits(),
                None => true,
            };
            if improves {
                best = Some(c);
            }
        }
    }
    best
}

/// Compute every candidate's bit cost for the block and return them
/// in priority order (ZERO first, then DIFF0..3). Useful for
/// inspection / debugging; [`select_predictor`] picks the cheapest.
///
/// A candidate that doesn't fit a natural energy or doesn't satisfy
/// its eligibility predicate (ZERO requires the all-`mu_chan`
/// invariant) is dropped from the returned list.
///
/// This entry point does **not** evaluate `BLOCK_FN_QLPC`. Use
/// [`evaluate_candidates_with_qlpc`] (passing a candidate coefficient
/// vector) to add QLPC to the listing.
pub fn evaluate_candidates(samples: &[i32], mu_chan: i64, carry: &ChannelCarry) -> Vec<Choice> {
    evaluate_candidates_with_qlpc(samples, mu_chan, carry, None)
}

/// Compute every candidate's bit cost for the block, including the
/// optional `BLOCK_FN_QLPC` candidate, and return them in priority
/// order `ZERO > DIFF0 > DIFF1 > DIFF2 > DIFF3 > QLPC`.
///
/// When `qlpc_candidate` is `Some(coefs)`, the sequencer scores the
/// QLPC command at the Rice-n optimum over the residual stream
/// `e_QLPC(t) = s(t) − Σᵢ coefs[i] · s(t − i − 1)` (TR.156 §3.3 +
/// `spec/03` §3.5) and appends it to the candidate list. The caller
/// retains ownership of coefficient quantisation per `spec/03` §3.5
/// and TR.156 §3.2; the sequencer treats `coefs` as opaque.
///
/// When `qlpc_candidate` is `None`, the behaviour is identical to
/// [`evaluate_candidates`] — a load-bearing backward-compat invariant
/// the in-module test
/// `select_with_qlpc_none_matches_legacy_select_predictor` pins
/// directly.
///
/// A QLPC candidate is dropped from the list (and the selector falls
/// back to DIFFn) when any of:
/// * `coefs.len() > MAX_QLPC_ORDER` ([`crate::write_qlpc_block`]'s
///   cap).
/// * `coefs.len() > carry.len()` (the predictor cannot seed enough
///   past samples; mirrors
///   [`crate::EncodeError::LpcOrderTooLarge`]).
/// * Any coefficient overflows the `svar(LPCQUANT)` folding.
/// * The residual stream contains a value the `svar` folding
///   overflows.
/// * No `e ∈ 0..=MAX_NATURAL_ENERGY` fits the residual stream.
pub fn evaluate_candidates_with_qlpc(
    samples: &[i32],
    mu_chan: i64,
    carry: &ChannelCarry,
    qlpc_candidate: Option<&[i64]>,
) -> Vec<Choice> {
    let mut out: Vec<Choice> = Vec::with_capacity(6);
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
    if let Some(coefs) = qlpc_candidate {
        if let Some(c) = evaluate_qlpc(samples, coefs, carry) {
            out.push(c);
        }
    }
    out
}

/// Compute every candidate's bit cost for the block — `ZERO`,
/// `DIFF0..3`, and the **auto-derived** `BLOCK_FN_QLPC` candidate of
/// [`derive_qlpc_candidate`] — in priority order
/// `ZERO > DIFF0 > DIFF1 > DIFF2 > DIFF3 > QLPC`.
///
/// Unlike [`evaluate_candidates_with_qlpc`], no caller-supplied
/// coefficient vector is needed: the sequencer runs the zero-anchored
/// order search over `0..=max_lpc_order` itself and appends the
/// cheapest derived candidate (if any order yields one). Passing
/// `max_lpc_order = 0` reproduces [`evaluate_candidates`] exactly —
/// per `spec/03` §3.5 a stream with `H_maxlpcorder = 0` never carries
/// a `BLOCK_FN_QLPC` command.
pub fn evaluate_candidates_auto(
    samples: &[i32],
    mu_chan: i64,
    carry: &ChannelCarry,
    max_lpc_order: u32,
) -> Vec<Choice> {
    let mut out = evaluate_candidates_with_qlpc(samples, mu_chan, carry, None);
    if let Some(c) = derive_qlpc_candidate(samples, carry, max_lpc_order) {
        out.push(c);
    }
    out
}

/// Pick the cheapest predictor for the block among `DIFF0..3`,
/// `ZERO`, and the **auto-derived** `BLOCK_FN_QLPC` candidate, and
/// return a [`Choice`] the caller hands to [`write_selected_block`].
///
/// This is the round-282 closure of the selection sequencer: where
/// [`select_predictor_with_qlpc`] required the caller to derive and
/// quantise the per-block coefficient vector itself, this entry point
/// owns the whole QLPC pipeline — order search over
/// `0..=max_lpc_order` (zero-anchored per `spec/02` §4.3's TR.156
/// anchor), least-squares coefficient derivation + integer
/// quantisation ([`derive_qlpc_coefs`], `spec/03` §3.5's unscaled
/// signed-integer coefficient domain), and the full cost model
/// including the `uvar(LPCQSIZE)` order field and the
/// `order × svar(LPCQUANT)` coefficient-transmission overhead — so
/// `BLOCK_FN_QLPC` is selected exactly when it is genuinely cheapest.
///
/// `max_lpc_order` is normally the stream header's `H_maxlpcorder`;
/// passing `0` reproduces [`select_predictor`] exactly (the
/// backward-compat invariant the in-module test
/// `select_auto_with_zero_max_order_matches_legacy` pins), because a
/// `H_maxlpcorder = 0` stream is polynomial-difference-only per
/// `spec/03` §3.5.
///
/// Ties break in the round-266 priority order
/// `ZERO > DIFF0 > DIFF1 > DIFF2 > DIFF3 > QLPC`; within the QLPC
/// order search, ties break toward the lower order.
///
/// Returns `None` when no candidate fits (empty block, or every
/// candidate's residual stream overflows the natural-energy band).
pub fn select_predictor_auto(
    samples: &[i32],
    mu_chan: i64,
    carry: &ChannelCarry,
    max_lpc_order: u32,
) -> Option<Choice> {
    if samples.is_empty() {
        return None;
    }
    evaluate_candidates_auto(samples, mu_chan, carry, max_lpc_order)
        .into_iter()
        .min_by(|a, b| a.bits().cmp(&b.bits()))
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
/// This entry point does **not** evaluate `BLOCK_FN_QLPC`. Use
/// [`select_predictor_with_qlpc`] (passing a candidate coefficient
/// vector) to add QLPC to the selection.
///
/// ## Empty block
///
/// An empty `samples` slice returns `None` because no predictor's
/// residual stream is defined; the caller should never reach this
/// path because the round-7 driver's `bs` is always at least 1
/// (`spec/03` §3.6 rejects `new_bs = 0` and `spec/01` §3 pins
/// `H_blocksize ≥ 1`).
pub fn select_predictor(samples: &[i32], mu_chan: i64, carry: &ChannelCarry) -> Option<Choice> {
    select_predictor_with_qlpc(samples, mu_chan, carry, None)
}

/// Pick the cheapest predictor for the block among `DIFF0..3`,
/// `ZERO`, and (when `qlpc_candidate` is `Some`) `QLPC`, and return a
/// [`Choice`] the caller hands to [`write_selected_block`].
///
/// Per the selection criterion the candidate with the smallest total
/// encoded bit count wins; ties break in priority order
/// `ZERO > DIFF0 > DIFF1 > DIFF2 > DIFF3 > QLPC`. QLPC sits last
/// because of its larger fixed overhead — the `uvar(LPCQSIZE)` order
/// field plus the `order × svar(LPCQUANT)` coefficient field add a
/// per-block constant the DIFFn family doesn't pay — so a tied total
/// cost is broken in favour of the simpler command.
///
/// `qlpc_candidate` carries the caller's already-quantised per-block
/// coefficient vector (`coefs[i] = aᵢ₊₁`, applied to `s(t − i − 1)`
/// per `spec/03` §3.5 / TR.156 §3.2 first equation). The selector
/// treats the vector as opaque: deriving the optimal quantised
/// coefficients from raw samples is a separate, higher-layer concern
/// per TR.156 §3.2's Laplacian-distribution rule.
///
/// When `qlpc_candidate` is `None`, the behaviour is identical to
/// [`select_predictor`] — a load-bearing backward-compat invariant
/// the in-module test
/// `select_with_qlpc_none_matches_legacy_select_predictor` pins
/// directly.
///
/// Returns `None` when no candidate fits.
pub fn select_predictor_with_qlpc(
    samples: &[i32],
    mu_chan: i64,
    carry: &ChannelCarry,
    qlpc_candidate: Option<&[i64]>,
) -> Option<Choice> {
    if samples.is_empty() {
        return None;
    }
    let candidates = evaluate_candidates_with_qlpc(samples, mu_chan, carry, qlpc_candidate);
    // Tie-breaker: stable min_by returns the first element on a tie,
    // and `evaluate_candidates_with_qlpc` produces candidates in
    // priority order ZERO > DIFF0 > DIFF1 > DIFF2 > DIFF3 > QLPC.
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
    match choice {
        Choice::Zero { .. } => {
            write_zero_block(writer);
            Ok(())
        }
        Choice::Diff0 { energy, .. } => write_diff0_block(writer, *energy, samples, mu_chan),
        Choice::Diff1 { energy, .. } => write_diff1_block(writer, *energy, samples, carry),
        Choice::Diff2 { energy, .. } => write_diff2_block(writer, *energy, samples, carry),
        Choice::Diff3 { energy, .. } => write_diff3_block(writer, *energy, samples, carry),
        Choice::Qlpc { coefs, energy, .. } => {
            write_qlpc_block(writer, *energy, coefs, samples, carry)
        }
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
        // for a representative spread of values + widths. Values
        // whose prefix-zero run at the given width exceeds the
        // decoder's read_uvar cap (32) are unrepresentable on a
        // decodable wire: svar_bits must return None for them.
        for &width in &[1u32, 2, 3, 4, 5] {
            for &v in &[
                0i64, 1, -1, 2, -2, 3, -3, 7, -7, 8, -8, 15, -15, 16, -16, 31, -31, 64, -64,
            ] {
                let folded: u64 = if v >= 0 {
                    (v as u64) << 1
                } else {
                    ((v.unsigned_abs()) << 1) - 1
                };
                let computed = svar_bits(v, width);
                if (folded >> width) > 32 {
                    assert_eq!(
                        computed, None,
                        "svar({v}, {width}) must reject prefix runs beyond the decoder cap"
                    );
                    continue;
                }
                let mut w = BitWriter::new();
                w.write_svar(v, width);
                let actual = w.bits_written();
                let computed = computed.unwrap();
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
    fn diff1_constant_input_with_zero_carry_picks_tight_mantissa_per_tr156_3_3() {
        // samples = [3, 3, 3, 3] with carry.at(0) = 0:
        // residuals = [3, 0, 0, 0]. Folded magnitudes [6, 0, 0, 0].
        // Round-251's natural-energy rule picked e=2 (cap 8 > 6) for a
        // 23-bit total. Round-254's TR.156 §3.3 statistical optimum
        // recognises that three of the four samples are zero — a tighter
        // mantissa width pays prefix bits on the seed-jump outlier but
        // saves mantissa bits on the rest. Sweeping `e ∈ 0..=7`:
        //   e=0 (w=1): per-sample fixed 2 bits; prefix ⌊6/2⌋ = 3 once.
        //              total = 4·2 + 3 = 11 bits.
        //   e=1 (w=2): per-sample 3; prefix ⌊6/4⌋ = 1. total = 13.
        //   e=2 (w=3): per-sample 4; prefix 0. total = 16. ← old pick.
        //   e≥3: monotonically worse.
        // Optimum is e=0 at 11 residual bits; the encoder emits the
        // command in 3 (fn) + 4 (energy uvar(3) over 0) + 11 = 18 bits.
        let samples = vec![3i32, 3, 3, 3];
        let carry = ChannelCarry::new(3);
        let choice = evaluate_diff1(&samples, &carry).unwrap();
        match choice {
            Choice::Diff1 { energy, bits } => {
                assert_eq!(energy, 0, "TR.156 §3.3 optimum is e=0 for this stream");
                assert_eq!(bits, 3 + 4 + 11);
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
    fn selector_picks_diff2_for_arithmetic_ramp_under_rice_n_optimum() {
        // samples = arithmetic progression 0, 5, 10, 15, ..., 5·(N-1)
        // with mu_chan = 0 (so ZERO ineligible).
        //
        // Round 251's natural-energy rule would have picked DIFF1 — the
        // first-differences are [0, 5, 5, 5, ..., 5] (folded max 10,
        // natural e=3, per-sample 5 bits → ~3 + 4 + 64·5 = 327 bits at
        // N=64). Round 254's TR.156 §3.3 statistical optimum sees the
        // second-differences of an arithmetic ramp are sparse:
        //   DIFF2 residuals = [0, 5, 0, 0, ..., 0] (only the second
        //   sample is non-zero — the recurrence stabilises at zero
        //   after the seed window winds down).
        // Folded magnitudes = [0, 10, 0, 0, ..., 0]. Sweeping `e`:
        //   e=0 (w=1): per-sample 2; prefix ⌊10/2⌋ = 5 once. cost = 133.
        //   e=1 (w=2): per-sample 3; prefix 2. cost = 194. monotone up.
        // Plus 3 + 4 prefix = 140 bits — beats DIFF1's 327 comfortably.
        // The optimum correctly chooses DIFF2.
        let samples: Vec<i32> = (0..64i32).map(|t| 5 * t).collect();
        let carry = ChannelCarry::new(3);
        let choice = select_predictor(&samples, 0, &carry).unwrap();
        match choice {
            Choice::Diff2 { energy, bits } => {
                assert_eq!(
                    energy, 0,
                    "TR.156 §3.3 optimum is e=0 for this sparse stream"
                );
                assert_eq!(bits, 140);
            }
            _ => panic!("expected DIFF2 (sparse second-diff stream), got {choice:?}"),
        }
        // Sanity: DIFF0 is cheapest only when the raw samples are
        // already tight; on the long-range ramp it cannot fit any
        // natural energy and is dropped from the candidate set.
        let cands = evaluate_candidates(&samples, 0, &carry);
        assert!(
            cands.iter().any(|c| matches!(c, Choice::Diff2 { .. })),
            "DIFF2 must be in the candidate set"
        );
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

    // ---- round 266: QLPC auto-selection ----

    #[test]
    fn select_with_qlpc_none_matches_legacy_select_predictor() {
        // Backward-compat invariant: passing `None` to
        // select_predictor_with_qlpc must produce a Choice equal to
        // the legacy select_predictor on the same inputs across a
        // representative spread of blocks.
        let carry = ChannelCarry::new(3);
        let test_blocks: Vec<Vec<i32>> = vec![
            vec![0; 8],                          // ZERO-eligible
            vec![7; 6],                          // DIFF0 only (mu = 0)
            vec![0, 1, 2, 3, 4, 5, 6, 7],        // small ramp
            (0..32i32).map(|t| 5 * t).collect(), // arithmetic ramp
            vec![3, 0, 0, 0],                    // sparse seed-jump
            vec![1, -1, 2, -2, 3, -3, 4, -4],    // alternating
        ];
        for (i, samples) in test_blocks.iter().enumerate() {
            for &mu in &[0i64, 3, 7] {
                let legacy = select_predictor(samples, mu, &carry);
                let with_none = select_predictor_with_qlpc(samples, mu, &carry, None);
                assert_eq!(
                    legacy, with_none,
                    "block {i} mu {mu}: select_predictor != select_predictor_with_qlpc(.., None)"
                );
            }
        }
    }

    #[test]
    fn qlpc_candidate_skipped_when_order_exceeds_carry_len() {
        // carry has 3 slots; a 4-element coefficient vector cannot
        // seed the predictor. evaluate_candidates_with_qlpc must drop
        // the QLPC candidate so the selector falls back to DIFFn.
        let carry = ChannelCarry::new(3);
        let samples: Vec<i32> = (0..16i32).collect();
        let coefs: Vec<i64> = vec![1, 0, 0, 0]; // 4-element vector
        let cands = evaluate_candidates_with_qlpc(&samples, 0, &carry, Some(&coefs));
        assert!(
            !cands.iter().any(|c| matches!(c, Choice::Qlpc { .. })),
            "QLPC candidate must be skipped when coefs.len() > carry.len(); got {cands:?}"
        );
        // Selector must still produce a non-QLPC choice.
        let choice = select_predictor_with_qlpc(&samples, 0, &carry, Some(&coefs)).unwrap();
        assert!(!matches!(choice, Choice::Qlpc { .. }));
    }

    #[test]
    fn qlpc_candidate_skipped_when_order_exceeds_max_qlpc_order() {
        // A coefs vector longer than MAX_QLPC_ORDER would be rejected
        // by write_qlpc_block; evaluate_qlpc mirrors that rule and
        // drops the candidate. Use a carry large enough that the
        // carry-length check doesn't fire first.
        let carry = ChannelCarry::new((MAX_QLPC_ORDER as usize) + 2);
        let samples: Vec<i32> = vec![0; 16];
        let coefs: Vec<i64> = vec![0; (MAX_QLPC_ORDER as usize) + 1];
        let cands = evaluate_candidates_with_qlpc(&samples, 0, &carry, Some(&coefs));
        assert!(
            !cands.iter().any(|c| matches!(c, Choice::Qlpc { .. })),
            "QLPC candidate must be skipped when coefs.len() > MAX_QLPC_ORDER"
        );
    }

    #[test]
    fn qlpc_candidate_skipped_on_empty_samples() {
        let carry = ChannelCarry::new(3);
        let samples: Vec<i32> = vec![];
        let coefs: Vec<i64> = vec![1, -1];
        assert!(evaluate_qlpc(&samples, &coefs, &carry).is_none());
        // Selector still returns None overall.
        assert!(select_predictor_with_qlpc(&samples, 0, &carry, Some(&coefs)).is_none());
    }

    #[test]
    fn zero_still_wins_over_qlpc_on_constant_mu_block() {
        // A constant-mu_chan block is ZERO-eligible; even with a
        // tempting QLPC candidate the 5-bit ZERO token must win on
        // total bit cost.
        let carry = ChannelCarry::new(3);
        let samples = vec![7i32; 8];
        let coefs: Vec<i64> = vec![1, 0]; // any well-formed candidate
        let choice = select_predictor_with_qlpc(&samples, 7, &carry, Some(&coefs)).unwrap();
        assert!(matches!(choice, Choice::Zero { bits: 5 }));
    }

    #[test]
    fn qlpc_chosen_when_residuals_beat_diff_family() {
        // Construct a block where the QLPC candidate matches the
        // underlying generating recurrence and produces all-zero
        // residuals — the cheapest possible residual stream. The
        // DIFFn family must NOT also match the recurrence, or it
        // would win on its lower fixed overhead.
        //
        // Use a Fibonacci-style recurrence s(t) = s(t-1) + s(t-2)
        // (coefs [1, 1]). With carry seed [10, 5] the stream is
        //   s(0) = 10 + 5 = 15
        //   s(1) = 15 + 10 = 25
        //   s(2) = 25 + 15 = 40
        //   s(3) = 40 + 25 = 65
        //   s(4) = 65 + 40 = 105
        //   ...
        // QLPC residuals under coefs [1, 1] are zero by construction;
        // DIFFn residuals (first / second / third differences of a
        // Fibonacci-like stream) are themselves Fibonacci-like and
        // grow without bound — not sparse enough to beat QLPC's
        // all-zero residual stream even at QLPC's larger fixed
        // overhead.
        let mut carry = ChannelCarry::new(3);
        carry.update_after_block(&[5, 10]); // carry.at(0)=10, at(1)=5
        let mut samples: Vec<i32> = Vec::new();
        let mut a = 5i32;
        let mut b = 10i32;
        for _ in 0..16 {
            let next = a + b;
            samples.push(next);
            a = b;
            b = next;
        }
        let coefs: Vec<i64> = vec![1, 1];

        // Verify the QLPC residuals are all zero under the chosen coefs.
        let residuals = qlpc_residuals(&samples, &coefs, &carry).unwrap();
        assert!(
            residuals.iter().all(|&r| r == 0),
            "test setup: expected all-zero QLPC residuals, got {residuals:?}"
        );

        let choice = select_predictor_with_qlpc(&samples, 0, &carry, Some(&coefs)).unwrap();
        match &choice {
            Choice::Qlpc {
                coefs: chosen,
                energy,
                bits,
            } => {
                assert_eq!(chosen, &coefs);
                assert_eq!(*energy, 0);
                // Residual cost at e=0 with all-zero folded values:
                // svar(1) over 0 is 2 bits per sample.
                let residual_bits = (samples.len() as u64) * 2;
                // Fixed cost: fn(uvar(2) over 7) + order_field
                // (uvar(2) over 2) + coef_stream (2 × svar(2) over
                // signed coefficient 1) + energy_field (uvar(3)
                // over 0).
                let order_field_bits = uvar_bits(coefs.len() as u32, LPCQSIZE);
                let coef_stream_bits: u64 =
                    coefs.iter().map(|&c| svar_bits(c, LPCQUANT).unwrap()).sum();
                let fn_prefix_bits = uvar_bits(FN_QLPC, FNSIZE);
                let energy_field_bits = uvar_bits(0, ENERGYSIZE);
                let expected = fn_prefix_bits
                    + order_field_bits
                    + coef_stream_bits
                    + energy_field_bits
                    + residual_bits;
                assert_eq!(*bits, expected);
            }
            other => panic!("expected QLPC variant, got {other:?}"),
        }
    }

    #[test]
    fn diff_wins_over_qlpc_when_qlpc_overhead_dominates() {
        // A QLPC candidate that doesn't help (zero-coefficient vector
        // → QLPC residuals equal the samples themselves) must not
        // displace a cheaper DIFFn candidate. The exact DIFFn winner
        // (DIFF1 / DIFF2 / DIFF3) depends on which polynomial order
        // the sample stream collapses, but it must always beat QLPC's
        // higher fixed overhead on the same residual stream.
        let carry = ChannelCarry::new(3);
        let samples: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let coefs: Vec<i64> = vec![0, 0]; // zero predictor → QLPC residuals == samples

        let cands = evaluate_candidates_with_qlpc(&samples, 0, &carry, Some(&coefs));
        // QLPC must be in the candidate list (well-formed).
        let qlpc_cost = cands
            .iter()
            .find_map(|c| {
                if let Choice::Qlpc { bits, .. } = c {
                    Some(*bits)
                } else {
                    None
                }
            })
            .expect("QLPC candidate present");
        // Find the cheapest DIFFn candidate.
        let cheapest_diff_cost = cands
            .iter()
            .filter_map(|c| match c {
                Choice::Diff0 { bits, .. }
                | Choice::Diff1 { bits, .. }
                | Choice::Diff2 { bits, .. }
                | Choice::Diff3 { bits, .. } => Some(*bits),
                _ => None,
            })
            .min()
            .expect("at least one DIFFn candidate present");
        assert!(
            cheapest_diff_cost < qlpc_cost,
            "cheapest DIFFn ({cheapest_diff_cost} bits) must beat zero-predictor QLPC ({qlpc_cost} bits)"
        );
        let choice = select_predictor_with_qlpc(&samples, 0, &carry, Some(&coefs)).unwrap();
        // The selector must pick a DIFFn, not the zero-predictor QLPC.
        assert!(
            !matches!(choice, Choice::Qlpc { .. }),
            "selector must avoid the zero-predictor QLPC, got {choice:?}\ncandidates = {cands:?}"
        );
    }

    #[test]
    fn qlpc_tie_breaks_below_diff3() {
        // Construct a scenario where DIFF3 and QLPC report identical
        // total bit costs; the priority order requires DIFF3 to win.
        // We don't need a natural construction — we can directly
        // synthesise two candidates with equal bits and check the
        // tie-break by feeding evaluate_candidates' output through
        // min_by directly.
        let d3 = Choice::Diff3 {
            energy: 1,
            bits: 100,
        };
        let q = Choice::Qlpc {
            coefs: vec![1, -1],
            energy: 1,
            bits: 100,
        };
        // The candidate list as evaluate_candidates_with_qlpc would
        // emit it (priority order).
        let candidates = vec![d3.clone(), q.clone()];
        let winner = candidates
            .into_iter()
            .min_by(|a, b| a.bits().cmp(&b.bits()))
            .unwrap();
        assert_eq!(
            winner, d3,
            "tie-break must favour DIFF3 (priority) over QLPC"
        );
    }

    #[test]
    fn write_selected_for_qlpc_emits_choice_bits_count() {
        // After a QLPC selection, the actual writer bit count must
        // equal Choice::bits() — load-bearing for higher-layer rate
        // planners (round-251 precedent extended to QLPC).
        //
        // Use the same Fibonacci-style construction as
        // qlpc_chosen_when_residuals_beat_diff_family so the
        // selector actually picks QLPC.
        let mut carry = ChannelCarry::new(3);
        carry.update_after_block(&[5, 10]); // seed carry to [10, 5]
        let mut samples: Vec<i32> = Vec::new();
        let mut a = 5i32;
        let mut b = 10i32;
        for _ in 0..16 {
            let next = a + b;
            samples.push(next);
            a = b;
            b = next;
        }
        let coefs: Vec<i64> = vec![1, 1]; // matches the recurrence

        let choice = select_predictor_with_qlpc(&samples, 0, &carry, Some(&coefs)).unwrap();
        assert!(matches!(choice, Choice::Qlpc { .. }));
        let mut writer = BitWriter::new();
        let before = writer.bits_written();
        write_selected_block(&mut writer, &choice, &samples, 0, &carry).unwrap();
        let after = writer.bits_written();
        assert_eq!(after - before, choice.bits());
    }

    #[test]
    fn evaluate_candidates_with_qlpc_includes_qlpc_when_eligible() {
        // The QLPC candidate must appear in the listing when supplied
        // and eligible; positioned last per priority order.
        let mut carry = ChannelCarry::new(3);
        carry.update_after_block(&[5, 10]);
        let samples: Vec<i32> = (0..16i32).map(|t| 15 + 5 * t).collect();
        let coefs: Vec<i64> = vec![2, -1];

        let cands = evaluate_candidates_with_qlpc(&samples, 0, &carry, Some(&coefs));
        // The last candidate must be QLPC (priority order ends with it).
        match cands.last().expect("non-empty candidate list") {
            Choice::Qlpc { .. } => {}
            other => panic!("expected QLPC last, got {other:?}"),
        }
    }

    // ---- round 282: QLPC auto-derivation (order search + quantised
    //      coefficient derivation) ----

    /// Generate a block from an exact integer linear recurrence
    /// `s(t) = Σᵢ coefs[i] · s(t − i − 1)`, history seeded from
    /// `carry` exactly as the encoder's residual computation seeds it.
    fn recurrence_block(coefs: &[i64], carry: &ChannelCarry, len: usize) -> Vec<i32> {
        let order = coefs.len();
        let mut hist: Vec<i64> = (0..order).map(|i| carry.at(i) as i64).collect();
        let mut out: Vec<i32> = Vec::with_capacity(len);
        for _ in 0..len {
            let mut next: i64 = 0;
            for (c, h) in coefs.iter().zip(hist.iter()) {
                next += c * h;
            }
            out.push(i32::try_from(next).expect("recurrence stays in i32"));
            if order > 0 {
                for i in (1..order).rev() {
                    hist[i] = hist[i - 1];
                }
                hist[0] = next;
            }
        }
        out
    }

    #[test]
    fn derive_recovers_exact_two_term_recurrence() {
        // s(t) = s(t − 1) − s(t − 2) is a bounded period-6 oscillation
        // outside the DIFFn family. The least-squares solve over the
        // carry-seeded contexts recovers [1, −1] exactly (the residual
        // energy minimum is 0 there) and integer rounding keeps it.
        let mut carry = ChannelCarry::new(3);
        carry.update_after_block(&[50, 100]); // carry.at(0)=100, at(1)=50
        let samples = recurrence_block(&[1, -1], &carry, 48);
        let coefs = derive_qlpc_coefs(&samples, &carry, 2).expect("derivation");
        assert_eq!(coefs, vec![1, -1]);
        // Residuals under the derived vector are exactly zero.
        let residuals = qlpc_residuals(&samples, &coefs, &carry).unwrap();
        assert!(residuals.iter().all(|&r| r == 0), "got {residuals:?}");
    }

    #[test]
    fn derive_recovers_exact_first_order_recurrence() {
        // s(t) = 3·s(t − 1) from seed 1 stays in i32 for a short block.
        let mut carry = ChannelCarry::new(3);
        carry.update_after_block(&[1]); // carry.at(0) = 1
        let samples = recurrence_block(&[3], &carry, 12);
        let coefs = derive_qlpc_coefs(&samples, &carry, 1).expect("derivation");
        assert_eq!(coefs, vec![3]);
    }

    #[test]
    fn derive_rejects_degenerate_inputs() {
        let carry = ChannelCarry::new(3);
        // All-zero block: normal-equation matrix is all-zero.
        assert!(derive_qlpc_coefs(&[0i32; 16], &carry, 1).is_none());
        // Order 0: nothing to derive.
        assert!(derive_qlpc_coefs(&[1i32, 2, 3], &carry, 0).is_none());
        // Fewer equations than unknowns.
        assert!(derive_qlpc_coefs(&[1i32, 2], &carry, 3).is_none());
        // Order beyond the carry length.
        assert!(derive_qlpc_coefs(&(0..16i32).collect::<Vec<_>>(), &carry, 4).is_none());
        // Empty block.
        assert!(derive_qlpc_coefs(&[], &carry, 1).is_none());
    }

    #[test]
    fn order_search_prefers_lowest_adequate_order() {
        // A pure first-order recurrence: order 1 already drives the
        // residuals to zero; order 2 can only tie on residual cost and
        // pays an extra svar(LPCQUANT) coefficient field, so the
        // strict-< order search keeps order 1.
        let mut carry = ChannelCarry::new(3);
        carry.update_after_block(&[1]);
        let samples = recurrence_block(&[3], &carry, 12);
        let choice = derive_qlpc_candidate(&samples, &carry, 3).expect("candidate");
        match &choice {
            Choice::Qlpc { coefs, .. } => assert_eq!(coefs, &vec![3]),
            other => panic!("expected QLPC candidate, got {other:?}"),
        }
    }

    #[test]
    fn order_search_is_zero_anchored_per_spec_02_4_3() {
        // spec/02 §4.3: "for version 2 and above, the search is
        // started at zero order". Order-0 QLPC predicts zero WITHOUT
        // consulting the mean estimator, so on an all-zero block with
        // a non-zero μ_chan (ZERO ineligible, DIFF0 residuals all
        // −μ_chan) the empty-coefficient candidate is the cheapest
        // QLPC order and must be visited by the search.
        let carry = ChannelCarry::new(3);
        let samples = vec![0i32; 32];
        let choice = derive_qlpc_candidate(&samples, &carry, 3).expect("candidate");
        match &choice {
            Choice::Qlpc { coefs, energy, .. } => {
                assert!(coefs.is_empty(), "expected order-0, got {coefs:?}");
                assert_eq!(*energy, 0);
            }
            other => panic!("expected QLPC candidate, got {other:?}"),
        }
    }

    #[test]
    fn derive_candidate_respects_max_order_zero() {
        // spec/03 §3.5: H_maxlpcorder = 0 streams never carry QLPC.
        let mut carry = ChannelCarry::new(3);
        carry.update_after_block(&[50, 100]);
        let samples = recurrence_block(&[1, -1], &carry, 48);
        assert!(derive_qlpc_candidate(&samples, &carry, 0).is_none());
    }

    #[test]
    fn select_auto_with_zero_max_order_matches_legacy() {
        // Backward-compat invariant: max_lpc_order = 0 must reproduce
        // the legacy selector exactly across a representative spread.
        let carry = ChannelCarry::new(3);
        let test_blocks: Vec<Vec<i32>> = vec![
            vec![0; 8],
            vec![7; 6],
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            (0..32i32).map(|t| 5 * t).collect(),
            vec![3, 0, 0, 0],
            vec![1, -1, 2, -2, 3, -3, 4, -4],
        ];
        for (i, samples) in test_blocks.iter().enumerate() {
            for &mu in &[0i64, 3, 7] {
                let legacy = select_predictor(samples, mu, &carry);
                let auto = select_predictor_auto(samples, mu, &carry, 0);
                assert_eq!(
                    legacy, auto,
                    "block {i} mu {mu}: select_predictor != select_predictor_auto(.., 0)"
                );
            }
        }
    }

    #[test]
    fn select_auto_picks_qlpc_on_non_polynomial_recurrence() {
        // The period-6 oscillation s(t) = s(t − 1) − s(t − 2) defeats
        // every DIFFn predictor (its differences stay full-amplitude
        // oscillations) but the auto-derived [1, −1] QLPC candidate
        // drives the residuals to zero — 2 bits per sample at e = 0
        // plus the fixed QLPC overhead, the genuine cheapest pick.
        let mut carry = ChannelCarry::new(3);
        carry.update_after_block(&[50, 100]);
        let samples = recurrence_block(&[1, -1], &carry, 64);
        let choice = select_predictor_auto(&samples, 0, &carry, 2).expect("choice");
        match &choice {
            Choice::Qlpc { coefs, energy, .. } => {
                assert_eq!(coefs, &vec![1, -1]);
                assert_eq!(*energy, 0);
            }
            other => panic!("expected QLPC, got {other:?}"),
        }
        // The auto pick must be genuinely cheapest: strictly cheaper
        // than every DIFFn / ZERO candidate on the same block.
        let base = evaluate_candidates(&samples, 0, &carry);
        for cand in &base {
            assert!(
                choice.bits() < cand.bits(),
                "QLPC ({} bits) must beat {cand:?}",
                choice.bits()
            );
        }
    }

    #[test]
    fn select_auto_falls_back_to_diffn_when_overhead_dominates() {
        // An arithmetic ramp is an exact second-order polynomial: the
        // auto-derived order-2 vector is [2, −1] — residual-identical
        // to DIFF2 (same carry seeding) — so QLPC can only tie DIFF2's
        // residual stream while paying the order + coefficient fields.
        // The selector must keep DIFF2.
        let carry = ChannelCarry::new(3);
        let samples: Vec<i32> = (0..64i32).map(|t| 5 * t).collect();
        let choice = select_predictor_auto(&samples, 0, &carry, 3).expect("choice");
        assert!(
            matches!(choice, Choice::Diff2 { .. }),
            "DIFF2 must win on overhead, got {choice:?}"
        );
        // And it must equal the legacy pick bit-for-bit.
        assert_eq!(choice, select_predictor(&samples, 0, &carry).unwrap());
    }

    #[test]
    fn select_auto_zero_still_wins_on_constant_mu_block() {
        let carry = ChannelCarry::new(3);
        let samples = vec![7i32; 16];
        let choice = select_predictor_auto(&samples, 7, &carry, 3).expect("choice");
        assert!(matches!(choice, Choice::Zero { bits: 5 }));
    }

    #[test]
    fn select_auto_handles_empty_block() {
        let carry = ChannelCarry::new(3);
        assert!(select_predictor_auto(&[], 0, &carry, 3).is_none());
    }

    #[test]
    fn evaluate_candidates_auto_appends_qlpc_last() {
        let mut carry = ChannelCarry::new(3);
        carry.update_after_block(&[50, 100]);
        let samples = recurrence_block(&[1, -1], &carry, 32);
        let cands = evaluate_candidates_auto(&samples, 0, &carry, 2);
        match cands.last().expect("non-empty candidate list") {
            Choice::Qlpc { .. } => {}
            other => panic!("expected auto QLPC last, got {other:?}"),
        }
        // Base (max 0) listing must carry no QLPC entry.
        let base = evaluate_candidates_auto(&samples, 0, &carry, 0);
        assert!(!base.iter().any(|c| matches!(c, Choice::Qlpc { .. })));
    }

    #[test]
    fn write_selected_for_auto_qlpc_emits_choice_bits_count() {
        // Round-251 invariant extended through the auto path: the
        // writer's actual bit delta equals Choice::bits() exactly.
        let mut carry = ChannelCarry::new(3);
        carry.update_after_block(&[50, 100]);
        let samples = recurrence_block(&[1, -1], &carry, 64);
        let choice = select_predictor_auto(&samples, 0, &carry, 2).expect("choice");
        assert!(matches!(choice, Choice::Qlpc { .. }));
        let mut writer = BitWriter::new();
        write_selected_block(&mut writer, &choice, &samples, 0, &carry).unwrap();
        assert_eq!(writer.bits_written(), choice.bits());
    }

    #[test]
    fn auto_order_search_caps_at_carry_length() {
        // carry.len() = 3 caps the search even when the caller allows
        // more; the search must not produce a vector longer than the
        // carry can seed (write_qlpc_block would reject it).
        let mut carry = ChannelCarry::new(3);
        carry.update_after_block(&[50, 100]);
        let samples = recurrence_block(&[1, -1], &carry, 32);
        let choice = derive_qlpc_candidate(&samples, &carry, 100).expect("candidate");
        match &choice {
            Choice::Qlpc { coefs, .. } => assert!(coefs.len() <= 3),
            other => panic!("expected QLPC candidate, got {other:?}"),
        }
    }
}
