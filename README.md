# oxideav-shorten

Pure-Rust Shorten lossless audio codec (decoder + production encoder)
for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

## Status

**Round 3 — clean-room rebuild from `docs/audio/shorten/`.** Decodes
v2/v3 wire format pinned in `spec/00..05` (with v1 syntactically
accepted; no v1 fixture is reachable to confirm the v1 layout).
Encodes via a predictor-search + energy-width-optimised production
encoder with Levinson–Durbin LPC + lossy bit-shift modes. Mono /
stereo / multi-channel PCM I/O across all eleven TR.156 filetype
labels.

| Round | Adds                                                                                   |
| ----- | -------------------------------------------------------------------------------------- |
| 1     | Decoder + 10 function codes + Rice/Golomb residuals + carry + mean estimator + 3 pinned filetypes. |
| 2     | Production encoder (`encode` + `EncoderConfig`); all 11 TR.156 filetypes; container demuxer + `ajkg` probe. |
| 3     | Levinson–Durbin LPC search; lossy `BLOCK_FN_BITSHIFT` encode; F1..F18 corpus-style structural tests. |

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

## What's not yet implemented (round 4+ candidates)

- **Bit-stream-corpus byte-exact decode** of the public `.shn` fixture
  set. The mean-estimator residual ±1 drift documented in
  `audit/01-validation-report.md §8.1` remains the known-bounded gap.
- **Mean estimator on the encode side.** Round 3 still writes
  `H_meanblocks = 0`, sidestepping the ±1 drift.
- **Format-version 1 / 3 wire-format deltas.** No v1 or v3 fixture is
  reachable in the docs corpus; v1 is syntactically accepted but not
  behaviourally pinned.
- **High-throughput optimisations.** Table-driven `uvar` prefix
  decode + SIMD residual unpacking are not in scope yet.

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
