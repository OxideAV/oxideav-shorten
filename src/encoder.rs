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
//! ## Round 3 additions
//!
//! * **Levinson–Durbin LPC coefficient search.** When `max_lpc_order >
//!   0`, the encoder solves the Yule-Walker system on the per-block
//!   autocorrelation to derive LPC coefficients (TR.156 §3.5 narrative;
//!   the standard textbook recursion). The float coefficients are
//!   then rounded to integer for the spec's coefficient-without-scaling
//!   recurrence. The search keeps the integer-rounded set if it beats
//!   the polynomial-equivalent baseline; otherwise the latter wins on a
//!   tie. Composes with the existing `DIFF0..3` candidates within the
//!   same per-block predictor search.
//! * **`BLOCK_FN_BITSHIFT` lossy mode.** When [`EncoderConfig::bshift`]
//!   is non-zero, the encoder emits a `BLOCK_FN_BITSHIFT` command at
//!   the start of the per-block stream and right-shifts every input
//!   sample by `bshift` before encoding. The decoder applies the
//!   inverse left-shift, so the recovered samples have the lower
//!   `bshift` bits zeroed. Per `spec/04` §3 the BITSHIFT command may
//!   appear anywhere in the stream; round 3 emits it once at stream
//!   start.
//!
//! ## Round 4 additions
//!
//! * **Running-mean estimator on encode side** ([`EncoderConfig::with_mean_blocks`]).
//!   When `mean_blocks > 0` the encoder maintains a per-channel
//!   `mean_blocks`-slot ring buffer of past per-block means, computes
//!   the at-block-start running mean `mu_chan` with the same C-style
//!   `trunc_div(sum + divisor/2, divisor)` rule as the decoder, and
//!   feeds it into the `BLOCK_FN_DIFF0` predictor's residuals
//!   (`r = s - mu_chan`). On constant-`mu_chan` blocks the encoder
//!   may emit `BLOCK_FN_ZERO` instead of a DIFF0 stream of all-zero
//!   residuals — saving the energy + per-residual cost. Composes with
//!   the LPC search and lossy `bshift > 0` mode, closing
//!   `audit/01` §8.1's ±1 drift on `bshift > 0` lossy fixtures
//!   because the encoder's DIFF0 residuals are now produced under the
//!   same `mu_chan` the decoder will subtract back. The default is
//!   still `mean_blocks = 0` to preserve the round-3 wire format
//!   exactly.
//!
//! ## Round 5 additions
//!
//! * **Bit-budget `-n N` lossy mode** ([`EncoderConfig::with_bit_budget`])
//!   and **bit-rate `-r N` lossy mode** ([`EncoderConfig::with_bit_rate`]).
//!   Both compute an effective `bshift` such that the resulting
//!   per-sample residual bit cost satisfies the target. The encoder
//!   probes candidate `bshift` values starting at 0, encodes a
//!   per-channel sample of the input under each, measures the actual
//!   bits-per-sample of the residual stream, and selects the smallest
//!   `bshift` whose output meets the target. When a target is
//!   unreachable even at `bshift = BITSHIFT_MAX`, the encoder caps at
//!   the maximum (the result is the lowest-rate stream it can
//!   produce). Composes with `with_max_lpc_order` and the running-mean
//!   estimator. Closes `audit/01` `F10`/`F11`/`F14`/`F15` lossy-mode
//!   coverage at the encoder side; matching ffmpeg / Tony Robinson's
//!   reference encoder on the same input remains §9.4-blocked because
//!   the exact bit-budget search heuristic the reference uses is not
//!   pinned in the spec set.
//! * **Speed: table-driven `uvar` prefix decode.** A 256-entry LUT
//!   indexes every 8-bit chunk to its leading-zero prefix length and
//!   the position of the terminating 1-bit, accelerating the
//!   [`crate::bitreader::BitReader::read_uvar_prefix`] hot path on
//!   long predictor-block residual streams. See
//!   `bitreader::UVAR_PREFIX_LUT`.
//!
//! ## What this encoder does **not** do
//!
//! * Skip-bytes (`H_skipbytes`). Verbatim-prefix bytes are emitted
//!   via `BLOCK_FN_VERBATIM` instead.

use crate::decoder::fn_code;
use crate::header::{Filetype, MAGIC};
use crate::varint::{
    signed_to_unsigned, BITSHIFTSIZE, ENERGYSIZE, FNSIZE, LPCQSIZE, LPCQUANT, ULONGSIZE,
    VERBATIM_BYTE_SIZE, VERBATIM_CHUNK_SIZE,
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
    /// Lossy bit-shift count emitted via a leading `BLOCK_FN_BITSHIFT`
    /// command and applied as a right-shift to every input sample
    /// prior to predictor application. `0` disables (lossless encode).
    /// Capped at `BITSHIFT_MAX` (`(1 << BITSHIFTSIZE) - 1` plus the
    /// uvar-prefix headroom; the decoder rejects any `bshift >= 32`).
    pub bshift: u32,
    /// `H_meanblocks` written into the header. When > 0 the encoder
    /// maintains a per-channel running-mean estimator with the same
    /// arithmetic the decoder applies (`spec/05` §2.5; the
    /// Validator-pinned C-truncation rule), and the
    /// `BLOCK_FN_DIFF0` predictor's residuals are produced relative
    /// to `mu_chan` rather than zero. Default `0` — the round-3 wire
    /// format. Capped at `MEAN_BLOCKS_MAX`.
    pub mean_blocks: u32,
    /// Round-5 lossy bit-budget target (`-n N` mode of TR.156's
    /// reference encoder). `Some(n)` instructs the encoder to pick the
    /// smallest `bshift` such that the per-sample post-Rice residual
    /// cost is `<= n` bits per sample. `None` (the default) disables
    /// the heuristic and uses the explicit [`Self::bshift`] field
    /// instead. Setting this field at the same time as a non-zero
    /// `bshift` returns [`EncodeError::BothBshiftAndBudget`].
    pub bit_budget: Option<u32>,
    /// Round-5 lossy bit-rate target (`-r N` mode of TR.156's
    /// reference encoder). `Some(r)` instructs the encoder to pick
    /// the smallest `bshift` such that the per-sample post-Rice
    /// residual cost is `<= r` bits per sample. Same semantics as
    /// [`Self::bit_budget`] but expressed as a real number; `r =
    /// 2.5` is fixture `F14`'s `-r 2_5`, `r = 4.0` is `F15`'s `-r 4`.
    /// Setting this field at the same time as a non-zero `bshift` or
    /// a `bit_budget` returns [`EncodeError::BothBshiftAndBudget`].
    pub bit_rate: Option<f32>,
}

/// Maximum running-mean window size accepted by the encoder. The
/// decoder accepts arbitrary `H_meanblocks` from the header but the
/// encoder caps to a reasonable bound so the per-channel buffer
/// allocation stays small and predictable. TR.156's typical default
/// is 4; the reference implementation's documented options range over
/// 0..=16.
pub const MEAN_BLOCKS_MAX: u32 = 64;

/// Maximum lossy bit-shift the encoder accepts. The decoder rejects
/// `bshift >= 32` (per `spec/04` §3 + `decoder::Error::BitShiftOverflow`),
/// so the encoder caps below that bound.
pub const BITSHIFT_MAX: u32 = 31;

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
            bshift: 0,
            mean_blocks: 0,
            bit_budget: None,
            bit_rate: None,
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

    /// Enable lossy `BLOCK_FN_BITSHIFT` mode at the given shift count.
    /// `0` is lossless. See [`EncoderConfig::bshift`].
    pub fn with_bshift(mut self, bshift: u32) -> Self {
        self.bshift = bshift;
        self
    }

    /// Enable the running-mean estimator with a window of `n` per-block
    /// means. `0` (the default) disables. The reference Shorten
    /// encoder's typical value is 4; the spec set has not pinned a
    /// preferred default for round 4. See [`EncoderConfig::mean_blocks`].
    pub fn with_mean_blocks(mut self, n: u32) -> Self {
        self.mean_blocks = n;
        self
    }

    /// Enable round-5 lossy `-n N` bit-budget mode at `n` bits per
    /// sample. The encoder picks the smallest `bshift` such that the
    /// per-sample post-Rice residual cost is `<= n`. See
    /// [`Self::bit_budget`].
    pub fn with_bit_budget(mut self, n: u32) -> Self {
        self.bit_budget = Some(n);
        self
    }

    /// Enable round-5 lossy `-r N` bit-rate mode at `r` bits per
    /// sample (real-valued). Mirrors `with_bit_budget` but supports
    /// fractional targets — `r = 2.5` is fixture `F14`. See
    /// [`Self::bit_rate`].
    pub fn with_bit_rate(mut self, r: f32) -> Self {
        self.bit_rate = Some(r);
        self
    }
}

/// Encoder errors.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EncodeError {
    /// `samples.len()` is not a multiple of `channels`.
    SamplesNotChannelAligned { samples: usize, channels: u16 },
    /// `channels`, `blocksize`, or `max_lpc_order` outside accepted bounds.
    InvalidConfig(&'static str),
    /// Both `bit_budget`/`bit_rate` and a non-zero `bshift` were set.
    /// The two modes are mutually exclusive — a fixed `bshift` is the
    /// `-q N` mode, while `bit_budget` / `bit_rate` are the
    /// auto-`bshift` modes that pick a shift internally.
    BothBshiftAndBudget,
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SamplesNotChannelAligned { samples, channels } => write!(
                f,
                "oxideav-shorten encode: samples ({samples}) not a multiple of channels ({channels})"
            ),
            Self::InvalidConfig(msg) => write!(f, "oxideav-shorten encode: {msg}"),
            Self::BothBshiftAndBudget => write!(
                f,
                "oxideav-shorten encode: bit_budget / bit_rate are mutually exclusive with a non-zero bshift; pick one mode"
            ),
        }
    }
}

impl Eq for EncodeError {}

impl std::error::Error for EncodeError {}

/// Compute the effective `bshift` for the given config + input.
///
/// * If neither [`EncoderConfig::bit_budget`] nor
///   [`EncoderConfig::bit_rate`] is set, returns `cfg.bshift`
///   unchanged.
/// * Otherwise probes candidate shifts `0..=BITSHIFT_MAX`, encoding
///   a representative slice of the input under each shift and
///   measuring the resulting per-sample post-Rice residual bit cost.
///   Returns the smallest shift whose measured cost is `<= target`,
///   or `BITSHIFT_MAX` if no shift in the range can hit the target
///   (the lowest-rate stream the encoder can produce).
///
/// The probe encodes one block per channel of size up to
/// [`EncoderConfig::blocksize`] from the *front* of the input —
/// representative for the typical short-correlation-length signals
/// Shorten targets, and bounded so the probe overhead stays small
/// relative to the full encode.
fn compute_effective_bshift(cfg: &EncoderConfig, samples: &[i32]) -> Result<u32, EncodeError> {
    // Translate the public config to a single target bits/sample
    // value, preferring `bit_budget` over `bit_rate` when both are
    // set (the validation upstream rejects that case via
    // BothBshiftAndBudget; the precedence here is defensive).
    let target_bps: f32 = match (cfg.bit_budget, cfg.bit_rate) {
        (Some(n), _) => n as f32,
        (None, Some(r)) => r,
        (None, None) => return Ok(cfg.bshift),
    };
    // bit_budget = 0 is accepted and treated as "unreachable", which
    // caps the search at BITSHIFT_MAX. bit_rate < 0 / NaN is rejected
    // upstream of this helper.
    // Probe each candidate shift. We measure on a per-channel slice
    // of up to one default block; the probe runs the same predictor
    // search the full encode would, so the measured per-residual
    // cost is exactly what the full encode would produce on this
    // probe block.
    let nch = cfg.channels as usize;
    let total_per_ch = samples.len() / nch;
    if total_per_ch == 0 {
        return Ok(0);
    }
    let probe_len = (cfg.blocksize as usize).min(total_per_ch);
    // De-interleave just the probe slice.
    let mut per_ch: Vec<Vec<i32>> = (0..nch).map(|_| Vec::with_capacity(probe_len)).collect();
    for i in 0..probe_len {
        for (ch, lane) in per_ch.iter_mut().enumerate().take(nch) {
            lane.push(samples[i * nch + ch]);
        }
    }
    let carry_len = core::cmp::max(3, cfg.max_lpc_order as usize);
    for shift in 0..=BITSHIFT_MAX {
        let bps = measure_bps_for_shift(&per_ch, cfg, carry_len, shift);
        if bps <= target_bps {
            return Ok(shift);
        }
    }
    Ok(BITSHIFT_MAX)
}

/// Measure per-sample bits for the probe slice at a candidate
/// `bshift`. The cost includes only the residual stream itself
/// (mantissa + Golomb-prefix + terminator) — block headers /
/// LPC-coefficient overhead amortise across the block and would
/// flatten the signal between candidates.
fn measure_bps_for_shift(
    per_ch: &[Vec<i32>],
    cfg: &EncoderConfig,
    carry_len: usize,
    shift: u32,
) -> f32 {
    let mut total_bits: u64 = 0;
    let mut total_residuals: u64 = 0;
    for lane in per_ch {
        let shifted: Vec<i32> = if shift == 0 {
            lane.clone()
        } else {
            lane.iter().map(|&s| s >> shift).collect()
        };
        let carry = vec![0i32; carry_len];
        let choice = best_predictor(&shifted, &carry, cfg.max_lpc_order);
        if !choice.residuals.is_empty() {
            total_bits += cost_of_residuals(&choice.residuals, choice.width);
            total_residuals += choice.residuals.len() as u64;
        }
    }
    if total_residuals == 0 {
        return 0.0;
    }
    total_bits as f32 / total_residuals as f32
}

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
    if cfg.bshift > BITSHIFT_MAX {
        return Err(EncodeError::InvalidConfig("bshift exceeds BITSHIFT_MAX"));
    }
    if cfg.mean_blocks > MEAN_BLOCKS_MAX {
        return Err(EncodeError::InvalidConfig(
            "mean_blocks exceeds MEAN_BLOCKS_MAX",
        ));
    }
    if (cfg.bit_budget.is_some() || cfg.bit_rate.is_some()) && cfg.bshift != 0 {
        return Err(EncodeError::BothBshiftAndBudget);
    }
    if let Some(r) = cfg.bit_rate {
        if !r.is_finite() || r <= 0.0 {
            return Err(EncodeError::InvalidConfig(
                "bit_rate must be finite and > 0",
            ));
        }
    }
    if samples.len() % cfg.channels as usize != 0 {
        return Err(EncodeError::SamplesNotChannelAligned {
            samples: samples.len(),
            channels: cfg.channels,
        });
    }

    // Round-5 `-n N` / `-r N` modes: pick an effective bshift before
    // entering the predictor search. The target is per-sample
    // post-Rice residual bits; we probe candidate shifts and pick the
    // smallest whose measured cost meets the target. For
    // `bit_budget = None && bit_rate = None` this evaluates to the
    // configured (possibly zero) `cfg.bshift` unchanged.
    let effective_bshift = compute_effective_bshift(cfg, samples)?;

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
    bw.write_ulong(cfg.mean_blocks); // running-mean window (round 4).
    bw.write_ulong(0); // skip_bytes — verbatim is emitted via BLOCK_FN_VERBATIM.

    // Verbatim prefix.
    if !cfg.verbatim.is_empty() {
        write_verbatim(&mut bw, &cfg.verbatim);
    }

    // BITSHIFT command — emitted once at stream start when the
    // encoder is configured for lossy bshift > 0. The decoder
    // applies the inverse left-shift on emitted samples. Round 5:
    // `effective_bshift` resolves either the explicit `cfg.bshift`
    // or the auto-shift computed from `bit_budget` / `bit_rate`.
    if effective_bshift > 0 {
        bw.write_uvar(fn_code::BITSHIFT, FNSIZE);
        bw.write_uvar(effective_bshift, BITSHIFTSIZE);
    }

    // De-interleave for per-channel predictor application. Apply the
    // configured `bshift` (right-shift, signed-arithmetic) per sample
    // before queuing — the decoder restores the lower zero bits via
    // its own left-shift on emission.
    let mut per_ch: Vec<Vec<i32>> = (0..nch).map(|_| Vec::with_capacity(total_per_ch)).collect();
    let bshift = effective_bshift;
    for i in 0..total_per_ch {
        for ch in 0..nch {
            let s = samples[i * nch + ch];
            let shifted = if bshift == 0 { s } else { s >> bshift };
            per_ch[ch].push(shifted);
        }
    }

    // Carry buffers for each channel: index 0 = s(t-1), 1 = s(t-2), …
    // Length matches the decoder's `max(3, max_lpc_order)` convention.
    let carry_len = core::cmp::max(3, cfg.max_lpc_order as usize);
    let mut carry: Vec<Vec<i32>> = (0..nch).map(|_| vec![0i32; carry_len]).collect();
    // Running-mean window — `mean_blocks` slots per channel, all zero
    // at stream start. Mirrors the decoder's `ChannelState::mean_buf`
    // (`spec/05` §2.5).
    let mut mean_buf: Vec<Vec<i32>> = (0..nch)
        .map(|_| vec![0i32; cfg.mean_blocks as usize])
        .collect();
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
            // Compute the at-block-start running mean. For
            // `mean_blocks == 0` this is always zero (matching the
            // round-3 wire format).
            let mu_chan = running_mean(&mean_buf[ch], cfg.mean_blocks);
            // Search for the best (predictor, width) pair, including
            // the BLOCK_FN_ZERO short-circuit when the block is
            // constant `mu_chan`.
            let choice = best_predictor_with_mean(
                block,
                &carry[ch],
                cfg.max_lpc_order,
                mu_chan,
                cfg.mean_blocks,
            );
            // Emit the chosen command + payload.
            emit_block(&mut bw, &choice);
            // Update the per-channel carry + mean buffer.
            update_carry(&mut carry[ch], block);
            push_block_mean(&mut mean_buf[ch], block_mean_of(block));
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
    /// Function-code for emission (`fn_code::DIFF*`, `QLPC`, or
    /// `ZERO`).
    fn_code: u32,
    /// LPC order — only meaningful when `fn_code == fn_code::QLPC`.
    /// Zero for the polynomial predictors.
    lpc_order: u32,
    /// LPC coefficients (length `lpc_order`).
    lpc_coefs: Vec<i32>,
    /// Encoder-side mantissa width (`n` of TR.156 §3.3 eq. 21). The
    /// energy field on the wire is `n - 1`. Unused for `ZERO`.
    width: u32,
    /// The residual sequence. Empty for `ZERO`.
    residuals: Vec<i32>,
}

/// Per-block mean — mirrors `decoder::block_mean` (the
/// Validator-pinned C-truncation rule of `spec/05` §2.5 step 1).
pub(crate) fn block_mean_of(block: &[i32]) -> i32 {
    let bs = block.len() as i64;
    if bs == 0 {
        return 0;
    }
    let sum: i64 = block.iter().map(|&v| v as i64).sum();
    let bias = bs / 2;
    let numerator = sum + bias;
    (numerator / bs) as i32
}

/// Running-mean estimator value at block start — mirrors
/// `decoder::ChannelState::running_mean`.
pub(crate) fn running_mean(buf: &[i32], mean_blocks: u32) -> i32 {
    if mean_blocks == 0 || buf.is_empty() {
        return 0;
    }
    let divisor = mean_blocks as i64;
    let sum: i64 = buf.iter().map(|&v| v as i64).sum();
    let bias = divisor / 2;
    let numerator = sum + bias;
    (numerator / divisor) as i32
}

/// Sliding-window update — mirrors
/// `decoder::ChannelState::push_block_mean`.
pub(crate) fn push_block_mean(buf: &mut [i32], mu_blk: i32) {
    let n = buf.len();
    if n == 0 {
        return;
    }
    for i in 0..n - 1 {
        buf[i] = buf[i + 1];
    }
    buf[n - 1] = mu_blk;
}

/// Search wrapper that augments the round-3 polynomial+QLPC search
/// with the round-4 mean-aware DIFF0 candidate (residual `s - mu_chan`)
/// and a `BLOCK_FN_ZERO` short-circuit when the entire block equals
/// `mu_chan`.
fn best_predictor_with_mean(
    block: &[i32],
    carry: &[i32],
    max_lpc_order: u32,
    mu_chan: i32,
    mean_blocks: u32,
) -> PredictorChoice {
    // ZERO short-circuit: when the running-mean estimator is active
    // and every sample of the block equals `mu_chan`, emit ZERO. The
    // decoder emits `bs` samples = `mu_chan` for ZERO blocks
    // (`spec/05` §2.4). The block-mean-of `mu_chan` is `mu_chan` so
    // the running-mean state evolves identically on both sides.
    if mean_blocks > 0 && !block.is_empty() && block.iter().all(|&s| s == mu_chan) {
        return PredictorChoice {
            fn_code: fn_code::ZERO,
            lpc_order: 0,
            lpc_coefs: Vec::new(),
            width: 1, // unused; stay non-zero for cost accounting.
            residuals: Vec::new(),
        };
    }

    let mut best = best_predictor(block, carry, max_lpc_order);

    // Recompute DIFF0 candidate with the running-mean offset. For
    // `mean_blocks == 0` `mu_chan` is always 0 and this re-evaluates
    // to exactly the round-3 DIFF0 candidate; for `mean_blocks > 0`
    // it may beat the round-3 baseline because the residual stream
    // is `s - mu_chan` which is centred near zero on DC-active
    // signals.
    if mean_blocks > 0 {
        let residuals: Vec<i32> = block.iter().map(|&s| s.wrapping_sub(mu_chan)).collect();
        let width = pick_width(&residuals);
        let cost = cost_of_residuals(&residuals, width);
        // Match the overhead accounting in `best_predictor`.
        let overhead = 1 + FNSIZE as u64 + 1 + ENERGYSIZE as u64;
        let total = cost.saturating_add(overhead);
        let best_cost = predictor_cost(&best);
        if total < best_cost {
            best = PredictorChoice {
                fn_code: fn_code::DIFF0,
                lpc_order: 0,
                lpc_coefs: Vec::new(),
                width,
                residuals,
            };
        }
    }
    best
}

/// Back-compute a candidate's emission cost — keeps the round-4
/// short-circuit comparisons honest against the round-3 search's
/// best.
fn predictor_cost(choice: &PredictorChoice) -> u64 {
    if choice.fn_code == fn_code::ZERO {
        return (1 + FNSIZE) as u64;
    }
    let body = cost_of_residuals(&choice.residuals, choice.width);
    let mut overhead = 1 + FNSIZE as u64 + 1 + ENERGYSIZE as u64;
    if choice.fn_code == fn_code::QLPC {
        overhead += 1
            + LPCQSIZE as u64
            + choice
                .lpc_coefs
                .iter()
                .map(|&c| svar_cost(c, LPCQUANT))
                .sum::<u64>();
    }
    body.saturating_add(overhead)
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

    // QLPC candidates: orders 1..=max_lpc_order. Round 3 derives
    // the coefficients via Levinson–Durbin recursion on the per-block
    // (with carry) autocorrelation, then quantises the float
    // coefficients to small signed integers — Shorten's QLPC predictor
    // applies coefficients without scaling, so the integer-valued
    // estimate that the recursion produces is the wire form. For
    // many natural-audio blocks this lands close to the polynomial
    // DIFF predictors at low orders ([1] for order 1, [2,-1] for
    // order 2, [3,-3,1] for order 3), but signals with non-trivial
    // resonance prefer different integer coefficients. For each order
    // we additionally consider the polynomial-equivalent baseline so
    // a regression on a flat-spectrum block falls back to the
    // identity coefficient set rather than to the (possibly inferior)
    // Levinson estimate.
    for order in 1..=max_lpc_order.min(crate::MAX_LPC_ORDER) {
        for coefs in lpc_candidate_coefs(order as usize, carry, block) {
            let residuals = compute_qlpc_residuals(&coefs, carry, block);
            let width = pick_width(&residuals);
            let cost = cost_of_residuals(&residuals, width);
            let coef_overhead =
                1 + LPCQSIZE as u64 + coefs.iter().map(|&c| svar_cost(c, LPCQUANT)).sum::<u64>();
            let total =
                cost.saturating_add(1 + FNSIZE as u64 + coef_overhead + 1 + ENERGYSIZE as u64);
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
    }

    best.expect("at least DIFF0 is always evaluated").1
}

/// Bit cost of the `svar(n)` encoding of `s` — folds to unsigned
/// then computes the same prefix + mantissa + terminator cost as
/// [`cost_of_residuals`] for a single value.
fn svar_cost(s: i32, n: u32) -> u64 {
    if n >= 32 {
        return u64::MAX;
    }
    let u = signed_to_unsigned(s);
    let high = (u >> n) as u64;
    high + 1 + n as u64
}

/// Identity (polynomial-equivalent) LPC coefficient set for an
/// `order`-tap predictor. These are the integer coefficients that
/// reproduce DIFF1 / DIFF2 / DIFF3 / DIFF3-padded behaviour:
///
/// * order = 1 → `[1]`
/// * order = 2 → `[2, -1]`
/// * order = 3 → `[3, -3, 1]`
/// * order ≥ 4 → `[3, -3, 1, 0, 0, …]`
fn identity_lpc_coefs(order: usize) -> Vec<i32> {
    let mut coefs = vec![0i32; order];
    if order >= 1 {
        coefs[0] = 1;
    }
    if order == 2 {
        coefs[0] = 2;
        coefs[1] = -1;
    } else if order >= 3 {
        coefs[0] = 3;
        coefs[1] = -3;
        coefs[2] = 1;
    }
    coefs
}

/// Levinson–Durbin recursion on the per-block (with carry)
/// autocorrelation, returning a quantised integer coefficient set of
/// length `order`.
///
/// The autocorrelation is taken over the concatenation of the carry
/// (most-recent first) and the block — `[carry[order-1], …, carry[0],
/// block[0], block[1], …]`. This matches the predictor's view: when
/// computing `ŝ(t) = Σ a_i · s(t − i)`, the index `t − i` reaches into
/// the carry for the first `order` samples of the block.
///
/// The recursion produces float reflection coefficients `k_i` and the
/// derived direct-form `a_i`. Each `a_i` is rounded to the nearest
/// signed integer for emission — Shorten's QLPC predictor applies
/// coefficients without an implicit shift, so any fractional precision
/// is lost on the wire. Where the float estimate rounds to a value
/// the polynomial-DIFF predictor already covers, the search will
/// re-discover the polynomial equivalent (and the identity baseline
/// guarantees a tie-break in that direction).
fn levinson_durbin(order: usize, carry: &[i32], block: &[i32]) -> Vec<i32> {
    if order == 0 {
        return Vec::new();
    }
    // Build the windowed signal `[carry[order-1], …, carry[0], block...]`.
    let mut windowed: Vec<f64> = Vec::with_capacity(order + block.len());
    for i in (0..order).rev() {
        windowed.push(carry.get(i).copied().unwrap_or(0) as f64);
    }
    for &s in block {
        windowed.push(s as f64);
    }

    // Autocorrelation r[0..=order] over the windowed signal.
    let n = windowed.len();
    let mut r = vec![0.0f64; order + 1];
    for lag in 0..=order {
        let mut acc = 0.0f64;
        if n > lag {
            for i in 0..(n - lag) {
                acc += windowed[i] * windowed[i + lag];
            }
        }
        r[lag] = acc;
    }

    // Degenerate / silent block — every coefficient stays zero. The
    // residuals will equal the input samples themselves.
    if r[0] <= 0.0 {
        return vec![0i32; order];
    }

    // Levinson–Durbin proper. `a` accumulates the direct-form
    // coefficients across iterations; `e` is the prediction-error
    // energy of the current order's predictor.
    let mut a = vec![0.0f64; order + 1];
    let mut e = r[0];
    a[0] = 1.0;
    for i in 1..=order {
        // Reflection coefficient k_i.
        let mut acc = 0.0f64;
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = -(r[i] + acc) / e;
        if !k.is_finite() {
            // Numerical breakdown — bail out with zero coefficients,
            // which is a valid (non-predictive) fallback.
            return vec![0i32; order];
        }
        // Update direct-form coefficients in place.
        let half = i / 2;
        for j in 1..=half {
            let aj = a[j];
            let ai_minus_j = a[i - j];
            a[j] = aj + k * ai_minus_j;
            a[i - j] = ai_minus_j + k * aj;
        }
        if i % 2 != 0 {
            let mid = i / 2 + 1;
            a[mid] += k * a[i - mid];
        }
        a[i] = k;
        e *= 1.0 - k * k;
        if e <= 0.0 {
            // Degenerate (rank-deficient) — stop expanding. Lower-
            // order coefficients remain valid; remaining slots stay 0.
            break;
        }
    }

    // Round the predictor coefficients (the negation of `a[1..=order]`,
    // since LPC convention represents the prediction as `s(t) +
    // Σ a_i · s(t-i) ≈ 0`, while Shorten's emitted coefficient set
    // represents `ŝ(t) = Σ c_i · s(t-i)` directly: c_i = -a_i).
    (1..=order).map(|i| (-a[i]).round() as i32).collect()
}

/// Candidate integer LPC coefficient sets the search evaluates for a
/// given block + order: the polynomial-equivalent identity set plus
/// the Levinson–Durbin estimate. A coefficient set rejected by the
/// `svar(LPCQUANT)` magnitude budget (i.e. an outlier with a
/// runaway prefix cost) is filtered before reaching the search.
fn lpc_candidate_coefs(order: usize, carry: &[i32], block: &[i32]) -> Vec<Vec<i32>> {
    let mut out = Vec::with_capacity(2);
    let identity = identity_lpc_coefs(order);
    out.push(identity.clone());
    let levinson = levinson_durbin(order, carry, block);
    if levinson != identity && levinson.iter().all(|&c| c.unsigned_abs() <= 1 << 16) {
        out.push(levinson);
    }
    out
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
    // BLOCK_FN_ZERO is parameter-less — emit just the function code.
    if choice.fn_code == fn_code::ZERO {
        return;
    }
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
