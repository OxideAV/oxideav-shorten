# Changelog

All notable changes to this crate are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Round-7 fused SoA stereo / multi-channel decode path. The decoder
  no longer accumulates per-channel `Vec<i32>` buffers and runs an
  end-of-stream interleave pass; each per-channel block is written
  directly into the interleaved output at strided positions
  (`interleaved[(t + i) * nch + c]`), with the per-stream `bshift`
  left-shift applied inline. A single reusable scratch block buffer
  is allocated once per `decode` call and re-used across every block,
  eliminating the per-block `Vec<i32>` allocation that the round-6
  path performed. **Throughput delta**: 4 KB-block × 64-block ×
  2-channel decode is **1.16× faster** than round-6 master (3.05 ms
  → 2.63 ms, best-of-5, M-series macOS host, 524288 samples). The
  mono path inherits a smaller speed-up (1.46 ms → 1.35 ms) from the
  same reusable-scratch + no-final-pass changes. As a side-effect
  the variable-`BLOCK_FN_BITSHIFT`-stream behaviour becomes more
  correct: the bshift in effect at each block's commit is applied,
  not the final stream-end bshift; existing fixtures (which emit
  `BITSHIFT` at most once at start) are unaffected.
- 6 round-7 tests: bit-exact stereo 4 KB-block roundtrip, 4-channel
  4 KB-block roundtrip, stereo `bshift = 2` lossy-mode roundtrip,
  stereo many-blocks (32 blocks/channel × 128 samples) roundtrip,
  and stereo throughput-floor + best-of-5 print. Crate ships
  **164 tests total** (up from 158).

- Round-6 64-bit bit-reservoir residual unpack
  (`bitstream64::Bitstream64`). The per-block residual decode loop now
  drops down into an in-register `u64` reader that resolves `uvar`
  prefix scans with `u64::leading_zeros` (lowering to hardware `lzcnt`
  on x86-64 / `clz` on aarch64), eliminating the round-5 byte-LUT
  lookup + per-bit refill on the long blocks that dominate decode
  time. The reservoir refills 8 bytes at a time via a single
  big-endian `u64::from_be_bytes` load. The decoder also bulk-decodes
  residuals into a scratch `Vec<i32>` first and runs the predictor
  recurrence on the buffer, separating variable-length bit reads from
  arithmetic. **Throughput delta**: 4 KB-block, 64-block synthetic
  16-bit decode is **2.13× faster** than master (3.30 ms → 1.55 ms,
  best-of-5, M-series macOS host). On the synthetic `uvar(2)` stream
  the prefix-scan-only kernel is **2.63× faster** (1.82 ms → 0.69 ms
  on 200 000 codes). Default-on; no Cargo feature gate.
- 16 round-6 tests: reservoir construction edge cases (sub-byte
  start, multi-refill long-zero runs, EOF-on-overrun), 4 KB-block
  DIFF1 / stereo / QLPC roundtrips, mixed-prefix walk over
  `uvar(0)` codes 0..=7, decode-throughput floor (asserts < 1 s for
  64×4096 samples), and a side-by-side reservoir-vs-byte-LUT
  benchmark with a 1.2× speed floor. Crate ships **158 tests total**
  (up from 142).
- Round-5 lossy bit-budget (`-n N`) and bit-rate (`-r N`) target
  encoder modes (`EncoderConfig::with_bit_budget` /
  `EncoderConfig::with_bit_rate`). Both compute an effective
  `bshift` such that the per-sample post-Rice residual bit cost
  is `<= target`. The encoder probes candidate shifts
  `0..=BITSHIFT_MAX` against the leading per-channel block, runs
  the same predictor search the full encode would, and picks the
  smallest shift whose measured cost meets the target. Targets so
  tight nothing in the range hits them cap at `BITSHIFT_MAX`
  rather than failing. Closes audit/01 §2's `F10` (`-n 8`),
  `F11` (`-n 16`), `F14` (`-r 2.5`), and `F15` (`-r 4`) lossy-mode
  encoder coverage. Returns `EncodeError::BothBshiftAndBudget` when
  combined with a non-zero `bshift`. Composes with
  `with_max_lpc_order` and `with_mean_blocks`.
- Round-5 speed: 256-entry leading-zero LUT
  (`bitreader::UVAR_PREFIX_LUT`) backing
  `BitReader::read_uvar_prefix`. The hot residual-decode path now
  consumes up to a full byte of zero-prefix per LUT lookup when
  byte-aligned, replacing the round-1 bit-by-bit scan loop.
  `varint::read_uvar` routes its prefix scan through the new
  helper, accelerating decode of long predictor blocks. The
  bit-by-bit fallback handles cursor positions straddling a byte
  boundary.
- 17 round-5 tests: bit-budget / bit-rate effective-shift
  selection (F10/F11/F14/F15 + lossless F12 corner), monotonicity
  in target, mutual exclusion with explicit `bshift`, invalid-rate
  rejection (NaN, zero, negative), composition with LPC and
  running-mean, stereo roundtrip, and a hand-crafted prefix-decode
  matrix that walks every leading-zero count 0..=15 across each
  byte-alignment offset 0..=7. Crate ships 142 tests total
  (up from 125).
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
