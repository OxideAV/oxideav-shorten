# oxideav-shorten

Pure-Rust Shorten lossless audio codec (decoder + production encoder)
for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

## Status

**Round 7 — clean-room rebuild from `docs/audio/shorten/`.** Decodes
v2/v3 wire format pinned in `spec/00..05` (with v1 syntactically
accepted; no v1 fixture is reachable to confirm the v1 layout).
Encodes via a predictor-search + energy-width-optimised production
encoder with Levinson–Durbin LPC, lossy bit-shift, running-mean
estimator, and round-5 bit-budget / bit-rate target lossy modes.
Round-6 64-bit-reservoir residual unpack lands a 2.13× decode
speed-up on long blocks; round-7 fused SoA stereo write lands an
additional 1.16× speed-up on the most common 2-channel shape.
Mono / stereo / multi-channel PCM I/O across all eleven TR.156
filetype labels.

| Round | Adds                                                                                   |
| ----- | -------------------------------------------------------------------------------------- |
| 1     | Decoder + 10 function codes + Rice/Golomb residuals + carry + mean estimator + 3 pinned filetypes. |
| 2     | Production encoder (`encode` + `EncoderConfig`); all 11 TR.156 filetypes; container demuxer + `ajkg` probe. |
| 3     | Levinson–Durbin LPC search; lossy `BLOCK_FN_BITSHIFT` encode; F1..F18 corpus-style structural tests. |
| 4     | Running-mean estimator on encode side (`with_mean_blocks`); closes audit/01 §8.1 ±1 drift on `bshift > 0`. |
| 5     | Bit-budget (`-n N`) / bit-rate (`-r N`) lossy encoder modes; LUT-driven `uvar` prefix decode. |
| 6     | 64-bit-reservoir residual unpack (`u64::leading_zeros`); 2.13× decode throughput on 4 KB+ blocks. |
| 7     | Fused SoA stereo decode (strided write, no end-of-stream interleave pass); 1.16× decode throughput on stereo 4 KB+ blocks. |

## What round 7 lands

- **Fused SoA stereo / multi-channel decode.** The decoder no longer
  accumulates per-channel `Vec<i32>` buffers and runs an end-of-stream
  interleave pass. Each block is written **directly into the
  interleaved output** at strided positions
  (`interleaved[(t + i) * nch + c]`), with the per-stream `bshift`
  left-shift applied inline. A single reusable scratch block buffer
  is allocated once per `decode` call and re-used across every
  block — eliminating the per-block `Vec<i32>` allocation that the
  round-6 path performed.
- **Throughput delta**: 4 KB-block × 64-block × 2-channel decode is
  **1.16× faster** than round 6 master (3.05 ms → 2.63 ms best-of-5
  on an M-series macOS host for 524288 samples). The mono path
  inherits a smaller speed-up (1.46 ms → 1.35 ms) from the same
  reusable-scratch + no-final-pass changes.
- **6 round-7 tests** covering bit-exact stereo 4 KB-block roundtrip,
  4-channel 4 KB-block roundtrip, stereo `bshift = 2` lossy mode,
  stereo many-blocks roundtrip, and stereo throughput-floor /
  best-of-5 print. The crate ships **164 tests total** (up from 158).

## What round 6 lands

- **64-bit-reservoir residual unpack** (`bitstream64::Bitstream64`).
  The decoder's per-block residual loop drops down into an in-register
  `u64` reader that resolves `uvar` prefix scans with
  `u64::leading_zeros` (lowering to hardware `lzcnt` on x86-64 / `clz`
  on aarch64), eliminating the round-5 byte-LUT lookup + per-bit
  refill on the long blocks that dominate decode time. The reservoir
  refills 8 bytes at a time via a single big-endian
  `u64::from_be_bytes` load. The decoder also bulk-decodes residuals
  into a scratch `Vec<i32>` first and runs the predictor recurrence on
  the buffer, separating variable-length bit reads from arithmetic.
- **Throughput delta**: 4 KB-block × 64-block, 16-bit synthetic decode
  is **2.13× faster** than master (3.30 ms → 1.55 ms best-of-5 on an
  M-series macOS host). On the prefix-scan-only synthetic stream the
  reservoir kernel is **2.63× faster** than the round-5 byte-LUT path
  (1.82 ms → 0.69 ms on 200 000 `uvar(2)` codes). Default-on; no
  Cargo feature gate.
- **16 round-6 tests** covering reservoir construction edge cases
  (sub-byte start, multi-refill long-zero runs, EOF-on-overrun),
  4 KB-block DIFF1 / stereo / QLPC roundtrips, mixed-prefix walks,
  and a side-by-side reservoir-vs-byte-LUT benchmark with a 1.2× speed
  floor. The crate ships **158 tests total** (up from 142).

## What round 5 lands

- **Bit-budget `-n N` and bit-rate `-r N` lossy encoder modes**
  (`EncoderConfig::with_bit_budget` / `EncoderConfig::with_bit_rate`).
  Both compute an effective `bshift` such that the per-sample
  post-Rice residual bit cost is `<= target`. The encoder probes
  candidate shifts `0..=BITSHIFT_MAX` against the leading
  per-channel block, runs the same predictor search the full
  encode would, and picks the smallest shift whose measured cost
  meets the target. Closes audit/01 §2 fixtures `F10` (`-n 8`),
  `F11` (`-n 16`), `F14` (`-r 2.5`), `F15` (`-r 4`) at the
  encoder side. Composes with `with_max_lpc_order` and
  `with_mean_blocks`. Mutually exclusive with an explicit non-zero
  `bshift` (returns `EncodeError::BothBshiftAndBudget`).
- **Speed: LUT-driven `uvar` prefix decode.** A 256-entry
  leading-zero table indexes each MSB-first byte to its position
  of first set bit, letting `BitReader::read_uvar_prefix` consume
  up to a full byte of `uvar` zero-prefix per LUT lookup when
  byte-aligned. `varint::read_uvar` routes its prefix scan through
  the new helper, replacing the round-1 bit-by-bit loop on the
  hot residual-decode path.
- **17 round-5 tests** covering bit-budget / bit-rate
  effective-shift selection across F10/F11/F12/F14/F15,
  monotonicity in target, mutual exclusion with explicit `bshift`,
  invalid-rate rejection (NaN, zero, negative), composition with
  LPC + running-mean, stereo roundtrip, and a hand-crafted
  prefix-decode matrix walking every zero-count 0..=15 across each
  byte-alignment 0..=7. The crate ships 142 tests total.

## What round 4 lands

- **Running-mean estimator on the encode side**
  (`EncoderConfig::with_mean_blocks`, capped at `MEAN_BLOCKS_MAX =
  64`). The encoder mirrors the decoder's per-channel
  `mean_blocks`-slot ring buffer with the Validator-pinned C-style
  `trunc_div(sum + divisor/2, divisor)` arithmetic of `spec/05` §2.5
  + `audit/01` §6.1, computes `mu_chan` at block start, and produces
  `BLOCK_FN_DIFF0` residuals as `s - mu_chan`. Constant-`mu_chan`
  blocks short-circuit to a parameter-less `BLOCK_FN_ZERO`. This
  closes `audit/01` §8.1's ±1 drift on `bshift > 0` lossy fixtures
  by lock-stepping encoder and decoder on identical `mu_chan`
  arithmetic. Default remains `mean_blocks = 0` (round-3 wire
  format). Composes with `with_max_lpc_order` and `with_bshift`.
- **20 round-4 tests** covering lossless roundtrip across
  `mean_blocks ∈ {0,1,4,8,16}` (mono + stereo), drift closure on
  `bshift ∈ {1,4,8,12}`, the `BLOCK_FN_ZERO` short-circuit, and
  composition with the round-3 LPC + bshift features. The crate
  ships 125 tests total.

## What round 3 lands

- **Levinson–Durbin LPC coefficient search.** When `max_lpc_order >
  0` the encoder solves the Yule-Walker system on the per-block-plus-
  carry autocorrelation to derive LPC coefficients, then rounds them
  to integers (Shorten's QLPC predictor applies coefficients without
  scaling per `spec/03` §3.5). The polynomial-equivalent identity
  baseline is retained as a candidate so a regression on a flat-
  spectrum block falls back to the polynomial predictor. On
  resonant signals the search yields meaningful compression gains
  (alternating-sign signal: 76% size reduction vs the polynomial
  baseline); on natural audio whose dynamics fit DIFF1..3 the search
  ties without regressing.
- **`BLOCK_FN_BITSHIFT` lossy encode mode** via
  `EncoderConfig::with_bshift`. The encoder emits a leading
  `BITSHIFT` command and right-shifts every input sample by `bshift`
  before predictor application; the decoder restores the bottom
  `bshift` bits as zeros via its existing left-shift on emission.
  Cap is `BITSHIFT_MAX = 31` to satisfy the decoder's `< 32` rule.
- **F1..F18 corpus-style tests.** 14 hand-built fixtures matching
  the audit/01 §2 enumeration's filetype + channel + bshift
  parameters (the `.shn` binaries themselves are not in the docs
  tree). Each asserts the magic + version byte + filetype + channel
  count + bshift round-trip + sample identity / shifted-identity.

## What round 2 lands

- **Production encoder** — predictor search across `DIFF0..3` and
  `QLPC` (when `max_lpc_order > 0`), energy-width optimisation per
  TR.156 §3.3 eq. 21 with the `+1` offset of `spec/05` §3.
- **All eleven filetypes** of TR.156's `-t` enumeration: `ulaw`, `s8`,
  `u8`, `s16hl`, `u16hl`, `s16lh`, `u16lh`, `s16`, `u16`, `s16x`,
  `u16x`. The eight numeric codes the spec set leaves unpinned are
  chosen by this implementation per the `Filetype` rustdoc; cross-
  implementation roundtrip on those codes is not guaranteed until
  `spec/05` §6 pins them.
- **Container demuxer + `ajkg` probe** — registers a `"shorten"`
  demuxer + a content-based probe on `oxideav-core`'s
  `ContainerRegistry`. Probe scores 100 on byte streams whose first
  four bytes are `ajkg` *and* whose version byte is in `1..=3`. The
  demuxer reads the entire input into a single packet — Shorten has no
  internal framing — and lets the codec decoder unpack the bit stream.

## What's not yet implemented (round 8+ candidates)

- **Bit-stream-corpus byte-exact decode** of the public `.shn` fixture
  set. Round 4 closes audit/01 §8.1's ±1 drift in the *internal*
  encode/decode pair (synthetic round-trips); the residual gap
  against ffmpeg on public-fixture pipeline-A still requires §9.4
  binary procurement to fully pin. The round-5 `-n N` / `-r N`
  modes pick a defensible bshift heuristic; matching the reference
  encoder's exact heuristic on the same input is also §9.4-blocked.
- **Format-version 1 / 3 wire-format deltas.** No v1 or v3 fixture is
  reachable in the docs corpus; v1 is syntactically accepted but not
  behaviourally pinned.
- **SIMD residual unpacking.** Round 5 lands the LUT-driven uvar
  prefix decode; SIMD-batched residual mantissa reads are the next
  throughput tier.
- **`bshift` left-shift loop SIMD-isation.** The per-block strided
  write currently applies `wrapping_shl(bshift)` scalar-style; a
  portable-SIMD lane-wise shift would chew through long stereo
  blocks faster but requires Rust 1.79+ MSRV.
- **Vendor-specific CLZ intrinsics** (`_lzcnt_u64` / `__clz`) — only
  if profiling shows `u64::leading_zeros` not lowering cleanly on
  some target.

## Cargo features

- **`registry`** (default): wire the crate's `register(ctx)` entry
  point into `oxideav-core`'s codec + container registries. Disable
  for standalone builds that want the codec without the framework.

## Public API

- [`encode`] / [`EncoderConfig`] / [`EncodeError`] — production
  encoder; takes interleaved `i32` PCM lanes and emits a `.shn` byte
  buffer.
- [`decode`] / [`DecodedStream`] — single-shot decode of a complete
  `.shn` byte buffer.
- [`parse_header`] — parse the byte-aligned + variable-length-integer
  header in isolation.
- [`StreamHeader`], [`Filetype`], [`MAGIC`] — output / format types.
- [`Error`], [`Result`] — crate-local error type.
- (with `registry`) [`register`], [`register_codecs`],
  [`container_register`], [`shorten_probe`], [`CODEC_ID_STR`],
  [`CONTAINER_NAME`].

The crate `forbid`s `unsafe`.

## Provenance

This implementation is a clean-room rebuild driven exclusively from:

- `docs/audio/shorten/spec/` — natural-language wire description
  (TR.156 academic source + multimedia.cx wiki + Hydrogenaudio +
  LOC FDD + black-box behavioural observation of FFmpeg 7.1.2's
  decoder over public `.shn` fixtures).
- `docs/audio/shorten/tables/` — extracted constants.
- `docs/audio/shorten/audit/` — validation reports.

No FFmpeg / libavcodec source, no Tony Robinson reference C source,
no third-party Shorten implementation, and no `old` branch content
were consulted.
