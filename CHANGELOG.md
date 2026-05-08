# Changelog

All notable changes to this crate are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Round-4 running-mean estimator on the encode side
  (`EncoderConfig::with_mean_blocks`, capped at `MEAN_BLOCKS_MAX =
  64`). The encoder mirrors the decoder's per-channel
  `mean_blocks`-slot ring buffer with the Validator-pinned C-style
  `trunc_div(sum + divisor/2, divisor)` arithmetic of `spec/05` §2.5
  + `audit/01` §6.1, computes `mu_chan` at block start, and produces
  `BLOCK_FN_DIFF0` residuals as `s - mu_chan` rather than `s - 0`.
  Constant-`mu_chan` blocks short-circuit to a parameter-less
  `BLOCK_FN_ZERO` emission. This closes `audit/01` §8.1's ±1 drift
  on `bshift > 0` lossy fixtures by lock-stepping encoder and
  decoder on the same `mu_chan`. Default remains `mean_blocks = 0`
  (round-3 wire format). Composes with `with_max_lpc_order` and
  `with_bshift`. Adds 20 round-4 self-roundtrip + drift-closure
  tests, bringing the suite to 125.
- Round-3 Levinson–Durbin LPC coefficient search inside the encoder's
  per-block predictor candidates. When `max_lpc_order > 0` the
  encoder runs the standard recursion over the per-block-plus-carry
  autocorrelation, rounds the float direct-form coefficients to
  integers (Shorten's QLPC predictor applies coefficients without
  an implicit shift per `spec/03` §3.5), and adopts the result
  whenever it beats the polynomial-equivalent identity baseline.
  See `levinson_durbin` and `lpc_candidate_coefs` in `encoder.rs`.
- Round-3 `BLOCK_FN_BITSHIFT` lossy encode mode via
  `EncoderConfig::with_bshift` (capped at `BITSHIFT_MAX = 31`).
  The encoder emits a leading `BITSHIFT` command and right-shifts
  every input sample by `bshift` before predictor application; the
  decoder restores the bottom `bshift` bits as zeros via its
  existing left-shift on emission. Round-3 emits the command once
  at stream start; per `spec/04` §3 it may also appear later, but
  there is no encoder use-case that requires that yet.
- Round-3 corpus-style structural tests covering F1, F2, F3, F4,
  F5, F6, F7, F8, F9, F12, F13, F16, F17, F18 (14 fixtures from the
  audit/01 §2 enumeration). Each reproduces the fixture's
  filetype + channel count + bshift via the production encoder,
  decodes the round-tripped stream, and asserts the magic bytes
  (`ajkg`), version byte (`0x02`), filetype, channel count, and
  the bshift round-trip property `recovered = (input >> bshift) <<
  bshift`. The `.shn` binaries themselves are not in the docs tree.
- Round-1 decoder: header parse, function codes 0..=9 (`DIFF0..3`,
  `QUIT`, `BLOCKSIZE`, `BITSHIFT`, `QLPC`, `ZERO`, `VERBATIM`),
  Rice / Golomb residual coding (`uvar` / `svar` / `ulong`),
  per-channel sample-history carry, running-mean estimator,
  pinned filetypes `u8 = 2`, `s16hl = 3`, `s16lh = 5`, mono and
  stereo i32-lane PCM output.
- `oxideav-core` framework integration (default-on `registry`
  feature) — codec id `shorten`, `register(ctx)` entry point.
- Round-2 production encoder (`encode` + `EncoderConfig` +
  `EncodeError`): predictor search across `DIFF0..3` plus `QLPC`
  (when `max_lpc_order > 0`); energy-width optimisation minimising
  the per-block Rice-coding bit cost (TR.156 §3.3 eq. 21 with the
  `+1` offset of `spec/05` §3).
- Round-2 expansion of `Filetype` to all eleven TR.156 labels
  (`Ulaw`, `S8`, `U8`, `S16Be`, `U16Be`, `S16Le`, `U16Le`,
  `S16Native`, `U16Native`, `S16Swapped`, `U16Swapped`). The eight
  numeric codes the spec set leaves unpinned are assigned by this
  implementation; the three pinned codes (`u8 = 2`, `s16hl = 3`,
  `s16lh = 5`) match `spec/05` §6.
- Round-2 container demuxer + `'ajkg'` content-based probe
  registered with `oxideav-core`'s `ContainerRegistry` under the
  name `"shorten"`. The `.shn` extension is also registered.
- Round-2 PCM packer covers all 11 filetypes (signed and unsigned
  8/16-bit; explicit, native, and byte-swapped 16-bit endianness;
  µ-law passthrough).
- 75 unit + roundtrip tests covering every function-code path,
  every filetype, encoder error handling, and the demuxer's
  end-to-end packet emission. Round 3 brings the suite to 105
  with the LPC + bitshift + corpus additions above.

### Changed

- Clean-room rebuild from a fresh orphan `master`. The previous
  implementation was retired by the OxideAV docs audit dated
  2026-05-06; the prior history is preserved on the `old` branch.
  See `README.md` for the rebuild scope and the strict-isolation
  workspace the Implementer rounds drew from.
