# Changelog

All notable changes to this crate are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Round-1 decoder: header parse, function codes 0..=9 (`DIFF0..3`,
  `QUIT`, `BLOCKSIZE`, `BITSHIFT`, `QLPC`, `ZERO`, `VERBATIM`),
  Rice / Golomb residual coding (`uvar` / `svar` / `ulong`),
  per-channel sample-history carry, running-mean estimator,
  pinned filetypes `u8 = 2`, `s16hl = 3`, `s16lh = 5`, mono and
  stereo i32-lane PCM output.
- `oxideav-core` framework integration (default-on `registry`
  feature) — codec id `shorten`, `register(ctx)` entry point.
- 38 unit + roundtrip tests covering every implemented function
  code path.

### Changed

- Clean-room rebuild from a fresh orphan `master`. The previous
  implementation was retired by the OxideAV docs audit dated
  2026-05-06; the prior history is preserved on the `old` branch.
  See `README.md` for the rebuild scope and the strict-isolation
  workspace the Implementer rounds drew from.
