//! Shorten encoder.
//!
//! Round 2 promotes the encoder from the round-1 test scaffold into a
//! production-grade implementation:
//!
//! * **Predictor search.** Each block is tried against `DIFF0..DIFF3`
//!   and (when `max_lpc_order > 0`) `QLPC` at orders `1..=max_lpc_order`,
//!   with the predictor that produces the smallest residual sum-of-
//!   absolute-values selected for emission.
//! * **Energy-width optimisation.** For every candidate predictor the
//!   encoder computes the Rice-coded mantissa width that minimises
//!   the bit cost of the block's residual stream. The chosen width is
//!   the smallest `n` such that the prefix-plus-mantissa cost of every
//!   residual fits the encoder's bit budget — a direct application of
//!   TR.156 §3.3 equation 21 (`n ≈ log2(log(2) · E(|x|))`) with the
//!   `+1` energy-field offset of `spec/05` §3.
//! * **All eleven filetypes.** Round 2 handles the `Filetype` enum's
//!   eleven labels; the eight numeric codes the spec set leaves
//!   unpinned roundtrip through this crate's encoder + decoder pair.
//!
//! The encoder is byte-aligned at start-of-stream (the magic + version
//! byte) and finishes with `BLOCK_FN_QUIT` plus zero-padding to the
//! next byte boundary per `spec/05` §4. The output is a complete
//! `.shn` byte buffer.
//!
//! ## What this encoder does **not** do
//!
//! * Running-mean-estimator (`mean_blocks > 0`) on the encode side.
//!   The encoder always writes `H_meanblocks = 0`, sidestepping the
//!   ±1 sub-bit-precision drift documented in `audit/01` §8.1. The
//!   decoder still handles `mean_blocks > 0` for streams produced
//!   externally.
//! * Bit-shift mode (`BLOCK_FN_BITSHIFT`). The encoder emits a
//!   `bshift = 0` stream — i.e. a fully lossless encode of the input
//!   `i32` lanes. Lossy bshift is a future enhancement.
//! * Skip-bytes (`H_skipbytes`). Verbatim-prefix bytes are emitted
//!   via `BLOCK_FN_VERBATIM` instead.

use crate::decoder::fn_code;
use crate::header::{Filetype, MAGIC};
#[cfg(test)]
use crate::varint::BITSHIFTSIZE;
use crate::varint::{
    signed_to_unsigned, ENERGYSIZE, FNSIZE, LPCQSIZE, LPCQUANT, ULONGSIZE, VERBATIM_BYTE_SIZE,
    VERBATIM_CHUNK_SIZE,
};

/// Encoder configuration.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Output filetype written into the `H_filetype` field. Round 2
    /// supports every TR.156 label.
    pub filetype: Filetype,
    /// Channel count.
    pub channels: u16,
    /// Default block size in samples per channel. TR.156's default is
    /// 256.
    pub blocksize: u32,
    /// Maximum LPC order considered by the predictor search. `0`
    /// disables `BLOCK_FN_QLPC` and restricts the search to the
    /// fixed polynomial predictors.
    pub max_lpc_order: u32,
    /// Verbatim-prefix bytes written via `BLOCK_FN_VERBATIM`. Used to
    /// preserve a host-format header (e.g. RIFF/WAVE preamble) at the
    /// front of the stream.
    pub verbatim: Vec<u8>,
}

impl EncoderConfig {
    /// Construct a minimal config for `filetype` and `channels`. The
    /// default block size is TR.156's recommended 256 samples.
    pub fn new(filetype: Filetype, channels: u16) -> Self {
        Self {
            filetype,
            channels,
            blocksize: 256,
            max_lpc_order: 0,
            verbatim: Vec::new(),
        }
    }

    /// Set the default block size.
    pub fn with_blocksize(mut self, blocksize: u32) -> Self {
        self.blocksize = blocksize;
        self
    }

    /// Enable LPC up to `max_order`. `0` disables.
    pub fn with_max_lpc_order(mut self, max_order: u32) -> Self {
        self.max_lpc_order = max_order;
        self
    }

    /// Set the verbatim prefix.
    pub fn with_verbatim(mut self, verbatim: Vec<u8>) -> Self {
        self.verbatim = verbatim;
        self
    }
}

/// Encoder errors.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// `samples.len()` is not a multiple of `channels`.
    SamplesNotChannelAligned { samples: usize, channels: u16 },
    /// `channels`, `blocksize`, or `max_lpc_order` outside accepted bounds.
    InvalidConfig(&'static str),
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SamplesNotChannelAligned { samples, channels } => write!(
                f,
                "oxideav-shorten encode: samples ({samples}) not a multiple of channels ({channels})"
            ),
            Self::InvalidConfig(msg) => write!(f, "oxideav-shorten encode: {msg}"),
        }
    }
}

impl std::error::Error for EncodeError {}

/// Encode interleaved `i32` PCM lanes into a complete `.shn` byte
/// buffer.
///
/// `samples` is interleaved (`c0_s0, c1_s0, c0_s1, ...`); its length
/// must be a multiple of `cfg.channels`. Returns the encoded bytes
/// (the file format starts with the byte-aligned `ajkg` magic + the
/// `0x02` version byte).
///
/// ## Predictor search and width optimisation
///
/// For every block the encoder evaluates the candidate predictors
/// (`DIFF0..3`, plus `QLPC` at orders `1..=max_lpc_order` when
/// enabled) and picks the predictor + width pair that minimises the
/// total bit cost of the block:
///
/// * For each predictor it computes the residual sequence and chooses
///   the energy-field width minimising
///   `sum_over_residuals(width + (folded_residual >> width) + 1)`,
///   bounded by `RESIDUAL_WIDTH_CAP - 1` (round 2 keeps the same cap
///   the decoder enforces, leaving headroom for the `+1` decoder
///   offset).
/// * The candidate with the lowest total cost wins; ties favour
///   lower-order predictors (smaller wire overhead).
pub fn encode(cfg: &EncoderConfig, samples: &[i32]) -> Result<Vec<u8>, EncodeError> {
    if cfg.channels == 0 {
        return Err(EncodeError::InvalidConfig("channels must be at least 1"));
    }
    if cfg.channels > crate::MAX_CHANNELS {
        return Err(EncodeError::InvalidConfig("channels exceeds MAX_CHANNELS"));
    }
    if cfg.blocksize == 0 || cfg.blocksize > crate::MAX_BLOCKSIZE {
        return Err(EncodeError::InvalidConfig("blocksize out of range"));
    }
    if cfg.max_lpc_order > crate::MAX_LPC_ORDER {
        return Err(EncodeError::InvalidConfig(
            "max_lpc_order exceeds MAX_LPC_ORDER",
        ));
    }
    if samples.len() % cfg.channels as usize != 0 {
        return Err(EncodeError::SamplesNotChannelAligned {
            samples: samples.len(),
            channels: cfg.channels,
        });
    }

    let nch = cfg.channels as usize;
    let total_per_ch = samples.len() / nch;

    // 5-byte byte-aligned prefix: magic + version. Round 2 emits v2
    // (the only version with a reachable fixture per spec/05 §7).
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.push(2u8);

    let mut bw = BitWriter::new();

    // Header parameter block.
    bw.write_ulong(cfg.filetype.to_code());
    bw.write_ulong(cfg.channels as u32);
    bw.write_ulong(cfg.blocksize);
    bw.write_ulong(cfg.max_lpc_order);
    bw.write_ulong(0); // mean_blocks — encoder side disables the running mean.
    bw.write_ulong(0); // skip_bytes — verbatim is emitted via BLOCK_FN_VERBATIM.

    // Verbatim prefix.
    if !cfg.verbatim.is_empty() {
        write_verbatim(&mut bw, &cfg.verbatim);
    }

    // De-interleave for per-channel predictor application.
    let mut per_ch: Vec<Vec<i32>> = (0..nch).map(|_| Vec::with_capacity(total_per_ch)).collect();
    for i in 0..total_per_ch {
        for ch in 0..nch {
            per_ch[ch].push(samples[i * nch + ch]);
        }
    }

    // Carry buffers for each channel: index 0 = s(t-1), 1 = s(t-2), …
    // Length matches the decoder's `max(3, max_lpc_order)` convention.
    let carry_len = core::cmp::max(3, cfg.max_lpc_order as usize);
    let mut carry: Vec<Vec<i32>> = (0..nch).map(|_| vec![0i32; carry_len]).collect();
    let mut written: Vec<usize> = vec![0; nch];

    let mut current_bs = cfg.blocksize as usize;

    'outer: loop {
        for ch in 0..nch {
            let remaining = total_per_ch - written[ch];
            if remaining == 0 {
                break 'outer;
            }
            let take = remaining.min(current_bs);
            if take != current_bs {
                // Emit a BLOCKSIZE override before this block.
                bw.write_uvar(fn_code::BLOCKSIZE, FNSIZE);
                bw.write_ulong(take as u32);
                current_bs = take;
            }
            let block = &per_ch[ch][written[ch]..written[ch] + take];
            // Search for the best (predictor, width) pair.
            let choice = best_predictor(block, &carry[ch], cfg.max_lpc_order);
            // Emit the chosen command + payload.
            emit_block(&mut bw, &choice);
            // Update the per-channel carry.
            update_carry(&mut carry[ch], block);
            written[ch] += take;
        }
    }

    // QUIT — the decoder pads to the next byte boundary on completion.
    bw.write_uvar(fn_code::QUIT, FNSIZE);

    out.extend_from_slice(&bw.finish());
    Ok(out)
}

/// One predictor candidate's evaluated state. Held by the search to
/// pick the best one.
struct PredictorChoice {
    /// Function-code for emission (`fn_code::DIFF*` or `QLPC`).
    fn_code: u32,
    /// LPC order — only meaningful when `fn_code == fn_code::QLPC`.
    /// Zero for the polynomial predictors.
    lpc_order: u32,
    /// LPC coefficients (length `lpc_order`).
    lpc_coefs: Vec<i32>,
    /// Encoder-side mantissa width (`n` of TR.156 §3.3 eq. 21). The
    /// energy field on the wire is `n - 1`.
    width: u32,
    /// The residual sequence.
    residuals: Vec<i32>,
}

/// Search for the predictor + width pair that minimises the per-block
/// bit cost.
fn best_predictor(block: &[i32], carry: &[i32], max_lpc_order: u32) -> PredictorChoice {
    let mut best: Option<(u64, PredictorChoice)> = None;

    // Polynomial-difference candidates: DIFF0..DIFF3.
    for poly_order in 0u32..=3 {
        let residuals = compute_diff_residuals(poly_order, carry, block);
        let width = pick_width(&residuals);
        let cost = cost_of_residuals(&residuals, width);
        // Add a small fixed overhead for the command + energy field
        // (FNSIZE prefix + 3-bit ENERGYSIZE mantissa, ~5 bits) — the
        // overhead is identical across DIFF0..3 so it doesn't change
        // the ordering, but we keep it to allow comparing against
        // QLPC fairly.
        let overhead = 1 + FNSIZE as u64 + 1 + ENERGYSIZE as u64;
        let total = cost.saturating_add(overhead);
        if best.as_ref().map_or(true, |(b, _)| total < *b) {
            best = Some((
                total,
                PredictorChoice {
                    fn_code: match poly_order {
                        0 => fn_code::DIFF0,
                        1 => fn_code::DIFF1,
                        2 => fn_code::DIFF2,
                        3 => fn_code::DIFF3,
                        _ => unreachable!(),
                    },
                    lpc_order: 0,
                    lpc_coefs: Vec::new(),
                    width,
                    residuals,
                },
            ));
        }
    }

    // QLPC candidates: orders 1..=max_lpc_order. The encoder uses a
    // simple identity-style coefficient set (e.g. order=1 → coefs=[1]
    // ≡ DIFF1) for now; a full Levinson-Durbin search is a future
    // enhancement. We still consider them so a stream advertised
    // with `max_lpc_order > 0` actually exercises the QLPC command.
    for order in 1..=max_lpc_order.min(crate::MAX_LPC_ORDER) {
        let coefs = identity_lpc_coefs(order as usize);
        let residuals = compute_qlpc_residuals(&coefs, carry, block);
        let width = pick_width(&residuals);
        let cost = cost_of_residuals(&residuals, width);
        let coef_overhead = 1 + LPCQSIZE as u64 + (order as u64) * (1 + LPCQUANT as u64);
        let total = cost.saturating_add(1 + FNSIZE as u64 + coef_overhead + 1 + ENERGYSIZE as u64);
        if best.as_ref().map_or(true, |(b, _)| total < *b) {
            best = Some((
                total,
                PredictorChoice {
                    fn_code: fn_code::QLPC,
                    lpc_order: order,
                    lpc_coefs: coefs,
                    width,
                    residuals,
                },
            ));
        }
    }

    best.expect("at least DIFF0 is always evaluated").1
}

/// Identity LPC coefficients for an `order`-tap predictor that
/// reproduces DIFF1 / DIFF2 / DIFF3 / shifted-DIFF behaviour.
///
/// * order = 1 → `[1]`           (predicts s(t-1))
/// * order = 2 → `[2, -1]`       (DIFF2 polynomial)
/// * order = 3 → `[3, -3, 1]`    (DIFF3 polynomial)
/// * order ≥ 4 → `[3, -3, 1, 0, 0, …]`
///
/// The QLPC residual stream computed under these coefficients matches
/// the corresponding DIFF predictor's residuals exactly; the QLPC
/// command's only added cost is the coefficient encoding.
fn identity_lpc_coefs(order: usize) -> Vec<i32> {
    let mut coefs = vec![0i32; order];
    let template: [i32; 3] = [3, -3, 1];
    for (i, slot) in coefs.iter_mut().enumerate().take(template.len().min(order)) {
        *slot = template[i];
    }
    if order == 1 {
        coefs[0] = 1;
    } else if order == 2 {
        coefs[0] = 2;
        coefs[1] = -1;
    }
    coefs
}

/// Residuals for the `order`-th polynomial-difference predictor.
fn compute_diff_residuals(order: u32, carry: &[i32], block: &[i32]) -> Vec<i32> {
    let mut s1 = carry.first().copied().unwrap_or(0);
    let mut s2 = carry.get(1).copied().unwrap_or(0);
    let mut s3 = carry.get(2).copied().unwrap_or(0);
    let mut out = Vec::with_capacity(block.len());
    for &s in block {
        let pred = match order {
            0 => 0i32,
            1 => s1,
            2 => s1.wrapping_mul(2).wrapping_sub(s2),
            3 => s1
                .wrapping_mul(3)
                .wrapping_sub(s2.wrapping_mul(3))
                .wrapping_add(s3),
            _ => unreachable!("compute_diff_residuals: order in 0..=3"),
        };
        out.push(s.wrapping_sub(pred));
        s3 = s2;
        s2 = s1;
        s1 = s;
    }
    out
}

/// Residuals for an `order`-tap QLPC predictor with given coefficients.
fn compute_qlpc_residuals(coefs: &[i32], carry: &[i32], block: &[i32]) -> Vec<i32> {
    let order = coefs.len();
    let mut hist = vec![0i32; order];
    for (i, slot) in hist.iter_mut().enumerate().take(order) {
        *slot = carry.get(i).copied().unwrap_or(0);
    }
    let mut out = Vec::with_capacity(block.len());
    for &s in block {
        let mut pred: i64 = 0;
        for i in 0..order {
            pred = pred.wrapping_add((coefs[i] as i64).wrapping_mul(hist[i] as i64));
        }
        out.push(s.wrapping_sub(pred as i32));
        for i in (1..hist.len()).rev() {
            hist[i] = hist[i - 1];
        }
        if !hist.is_empty() {
            hist[0] = s;
        }
    }
    out
}

/// Pick the per-block residual mantissa width minimising the
/// Rice-coding bit cost.
///
/// TR.156 §3.3 eq. 21 gives `n ≈ log2(log(2) · E(|x|))`; in practice
/// for integer-arithmetic implementations the optimum is the smallest
/// `n` such that `(2^n) · L_n` minimises `n · L_n + sum(|r| >> n)`
/// where `L_n` is the per-residual cost. We compute the mean-absolute-
/// residual and bracket the search around `floor(log2(mean_abs))`,
/// scanning ±2 widths to land on the true optimum without paying for a
/// full O(width_cap) scan.
fn pick_width(residuals: &[i32]) -> u32 {
    if residuals.is_empty() {
        return 1;
    }
    // Width cap: 1 less than RESIDUAL_WIDTH_CAP to leave room for the
    // decoder's `+1` offset. Width floor is 1 — the energy field
    // encodes `width - 1` as an unsigned value, so width=0 is not
    // representable on the wire.
    let cap: u32 = crate::varint::RESIDUAL_WIDTH_CAP - 1;
    // Mean absolute folded residual.
    let mut sum_abs: u64 = 0;
    for &r in residuals {
        sum_abs += signed_to_unsigned(r) as u64;
    }
    let mean_folded = (sum_abs / residuals.len() as u64).max(1);
    // Approximate log2.
    let approx = (64 - mean_folded.leading_zeros()).saturating_sub(1);
    let lo = approx.saturating_sub(2).max(1);
    let hi = (approx + 2).clamp(1, cap);
    let mut best: Option<(u64, u32)> = None;
    for n in lo..=hi {
        let c = cost_of_residuals(residuals, n);
        if best.as_ref().map_or(true, |(b, _)| c < *b) {
            best = Some((c, n));
        }
    }
    best.expect("scan range is non-empty").1
}

/// Bit cost of encoding the residual sequence at mantissa width `n`.
///
/// Per-residual cost = `n + (folded >> n) + 1` (mantissa + prefix
/// zeros + terminator bit). The cost summed over the block is the
/// quantity to minimise.
fn cost_of_residuals(residuals: &[i32], n: u32) -> u64 {
    if n >= 32 {
        return u64::MAX; // sentinel: unencodable.
    }
    let mut total: u64 = 0;
    for &r in residuals {
        let u = signed_to_unsigned(r);
        let high = (u >> n) as u64;
        // Cap the prefix length to avoid runaway costs on outlier
        // residuals — choose the next-higher width if a prefix
        // explodes past 64 zeros.
        if high > 64 {
            return u64::MAX;
        }
        total = total.saturating_add(high + 1 + n as u64);
    }
    total
}

/// Emit one predictor block (header + residuals) using the chosen
/// candidate.
fn emit_block(bw: &mut BitWriter, choice: &PredictorChoice) {
    bw.write_uvar(choice.fn_code, FNSIZE);
    if choice.fn_code == fn_code::QLPC {
        bw.write_uvar(choice.lpc_order, LPCQSIZE);
        for &c in &choice.lpc_coefs {
            bw.write_svar(c, LPCQUANT);
        }
    }
    // Energy field = width - 1.
    let field = choice.width.saturating_sub(1);
    bw.write_uvar(field, ENERGYSIZE);
    for &r in &choice.residuals {
        bw.write_svar(r, choice.width);
    }
}

/// Append a verbatim chunk to `bw`.
fn write_verbatim(bw: &mut BitWriter, bytes: &[u8]) {
    let max_chunk = (1u32 << VERBATIM_CHUNK_SIZE) - 1; // mantissa cap; prefix can grow.
                                                       // Even with a wider chunk-length field we keep emission simple by
                                                       // chunking into sub-pieces that fit a moderate prefix budget.
    let chunk_size = max_chunk.saturating_mul(8) as usize; // generous chunk
    let mut idx = 0;
    while idx < bytes.len() {
        let end = (idx + chunk_size).min(bytes.len());
        bw.write_uvar(fn_code::VERBATIM, FNSIZE);
        bw.write_uvar((end - idx) as u32, VERBATIM_CHUNK_SIZE);
        for &b in &bytes[idx..end] {
            bw.write_uvar(b as u32, VERBATIM_BYTE_SIZE);
        }
        idx = end;
    }
}

/// Carry-buffer update — mirrors the decoder's `ChannelState::update_carry`.
fn update_carry(carry: &mut [i32], block: &[i32]) {
    let carry_len = carry.len();
    let bs = block.len();
    if carry_len == 0 {
        return;
    }
    if bs >= carry_len {
        for i in 0..carry_len {
            carry[i] = block[bs - 1 - i];
        }
    } else {
        let mut new_carry = vec![0i32; carry_len];
        for (i, slot) in new_carry.iter_mut().enumerate().take(bs) {
            *slot = block[bs - 1 - i];
        }
        new_carry[bs..carry_len].copy_from_slice(&carry[..(carry_len - bs)]);
        carry.copy_from_slice(&new_carry);
    }
}

// ───────────────────────── BitWriter ─────────────────────────

/// MSB-first bit packer. Public-within-crate so encoder helpers can
/// reuse it; not exported.
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    cur: u8,
    cur_bits: u8,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        Self {
            bytes: Vec::new(),
            cur: 0,
            cur_bits: 0,
        }
    }

    pub(crate) fn write_bit(&mut self, b: u32) {
        self.cur = (self.cur << 1) | ((b & 1) as u8);
        self.cur_bits += 1;
        if self.cur_bits == 8 {
            self.bytes.push(self.cur);
            self.cur = 0;
            self.cur_bits = 0;
        }
    }

    pub(crate) fn write_bits(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.write_bit((value >> i) & 1);
        }
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        if self.cur_bits > 0 {
            self.cur <<= 8 - self.cur_bits;
            self.bytes.push(self.cur);
        }
        self.bytes
    }

    pub(crate) fn write_uvar(&mut self, value: u32, n: u32) {
        let high = value >> n;
        let mantissa = if n == 0 { 0 } else { value & ((1u32 << n) - 1) };
        for _ in 0..high {
            self.write_bit(0);
        }
        self.write_bit(1);
        self.write_bits(mantissa, n);
    }

    pub(crate) fn write_svar(&mut self, s: i32, n: u32) {
        let u = signed_to_unsigned(s);
        self.write_uvar(u, n);
    }

    pub(crate) fn write_ulong(&mut self, value: u32) {
        // `ulong()` is a two-stage `uvar(uvar(ULONGSIZE), w)` — pick
        // the smallest `w` such that the total bit cost is minimised.
        // The TR.156 wiki convention is the smallest `w` where
        // `value < 2^(some_prefix_budget) << w`; the simplest and
        // adequate heuristic is `w = max(0, bits_needed - 2)` which
        // keeps the prefix to ≤ 3 zero bits across the value range
        // we care about (header fields).
        let mut w: u32 = 0;
        if value > 0 {
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

// ─────────────────── Test-only round-1 helpers ───────────────────
//
// Round 1 used these to construct hand-crafted streams for the
// round-1 self-roundtrip suite. Round 2's production encoder
// supersedes them, but they remain compiled under `#[cfg(test)]` so
// the round-1 test corpus continues to pass.

#[cfg(test)]
pub(crate) fn encode_minimal(
    filetype: Filetype,
    channels: u16,
    blocksize: u32,
    samples: &[i32],
    predictor: u32,
    verbatim: &[u8],
) -> Vec<u8> {
    assert!(samples.len() % channels as usize == 0);
    let nch = channels as usize;
    let total_per_ch = samples.len() / nch;

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.push(2u8);

    let mut bw = BitWriter::new();
    bw.write_ulong(filetype.to_code());
    bw.write_ulong(channels as u32);
    bw.write_ulong(blocksize);
    bw.write_ulong(0);
    bw.write_ulong(0);
    bw.write_ulong(0);

    if !verbatim.is_empty() {
        write_verbatim(&mut bw, verbatim);
    }

    let mut per_ch: Vec<Vec<i32>> = (0..nch).map(|_| Vec::with_capacity(total_per_ch)).collect();
    for i in 0..total_per_ch {
        for ch in 0..nch {
            per_ch[ch].push(samples[i * nch + ch]);
        }
    }

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
                bw.write_uvar(fn_code::BLOCKSIZE, FNSIZE);
                bw.write_ulong(take as u32);
                current_bs = take;
                emitted_partial = true;
            }
            let block = &per_ch[ch][written[ch]..written[ch] + take];
            let residuals = compute_diff_residuals(predictor, &history[ch], block);
            let max_abs = residuals
                .iter()
                .map(|r| r.unsigned_abs())
                .max()
                .unwrap_or(0);
            let needed_unsigned = if max_abs == 0 {
                1
            } else {
                let folded: u32 = max_abs.saturating_mul(2) | 1;
                32 - folded.leading_zeros()
            };
            let width: u32 = needed_unsigned.max(1);
            let field = width - 1;
            bw.write_uvar(predictor, FNSIZE);
            bw.write_uvar(field, ENERGYSIZE);
            for &r in &residuals {
                bw.write_svar(r, width);
            }
            update_carry(&mut history[ch], block);
            written[ch] += take;
        }
    }

    bw.write_uvar(fn_code::QUIT, FNSIZE);
    out.extend_from_slice(&bw.finish());
    out
}

#[cfg(test)]
pub(crate) fn encode_with_bitshift_and_zero(
    filetype: Filetype,
    channels: u16,
    blocksize: u32,
    bshift: u32,
    zero_blocks: u32,
    diff_block: &[i32],
) -> Vec<u8> {
    assert_eq!(diff_block.len(), blocksize as usize);
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.push(2);

    let mut bw = BitWriter::new();
    bw.write_ulong(filetype.to_code());
    bw.write_ulong(channels as u32);
    bw.write_ulong(blocksize);
    bw.write_ulong(0);
    bw.write_ulong(0);
    bw.write_ulong(0);

    if bshift > 0 {
        bw.write_uvar(fn_code::BITSHIFT, FNSIZE);
        bw.write_uvar(bshift, BITSHIFTSIZE);
    }

    let nch = channels as usize;
    for _ in 0..zero_blocks {
        for _ in 0..nch {
            bw.write_uvar(fn_code::ZERO, FNSIZE);
        }
    }

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
        bw.write_uvar(fn_code::DIFF1, FNSIZE);
        bw.write_uvar(field, ENERGYSIZE);
        let mut prev: i32 = 0;
        for &s in diff_block {
            let r = s.wrapping_sub(prev);
            bw.write_svar(r, width);
            prev = s;
        }
    }

    bw.write_uvar(fn_code::QUIT, FNSIZE);
    out.extend_from_slice(&bw.finish());
    out
}

#[cfg(test)]
pub(crate) fn encode_qlpc_block(
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
    bw.write_ulong(0);
    bw.write_ulong(0);

    let nch = channels as usize;
    let order = coefs.len();
    for block in samples_per_channel.iter().take(nch) {
        assert_eq!(block.len(), blocksize as usize);
        let residuals = compute_qlpc_residuals(coefs, &[0i32; 0], block);
        let max_abs = residuals
            .iter()
            .map(|r| r.unsigned_abs())
            .max()
            .unwrap_or(0);
        let folded: u32 = max_abs.saturating_mul(2) | 1;
        let needed = 32 - folded.leading_zeros();
        let width = needed.max(1);
        let field = width - 1;
        bw.write_uvar(fn_code::QLPC, FNSIZE);
        bw.write_uvar(order as u32, LPCQSIZE);
        for &c in coefs {
            bw.write_svar(c, LPCQUANT);
        }
        bw.write_uvar(field, ENERGYSIZE);
        for &r in &residuals {
            bw.write_svar(r, width);
        }
    }

    bw.write_uvar(fn_code::QUIT, FNSIZE);
    out.extend_from_slice(&bw.finish());
    out
}
