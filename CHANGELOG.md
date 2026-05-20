# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Round 1 clean-room rebuild.** File-header parser landed against
  `docs/audio/shorten/spec/01-stream-header.md` + `spec/02`:
  - `parse_stream_header(bytes)` returns the parsed
    [`ShortenStreamHeader`] (`version`, `filetype`, `channels`,
    `blocksize`, `maxlpcorder`, `meanblocks`, `skipbytes`) and the
    bit-stream position at which the per-block command stream
    begins.
  - MSB-first `BitReader` implementing `uvar(n)` and the two-stage
    `ulong()` form (`ULONGSIZE = 2` per `spec/02` §3 + §5).
  - 13 unit tests, including byte-exact verification on fixture
    `F1`'s first eleven bytes (`61 6A 6B 67 02 FB B1 70 09 F9 25`),
    rejection of short / bad-magic / unsupported-version buffers,
    and the `spec/01` §4 sample-history carry floor.

### Erased

- Prior master history was force-erased on **2026-05-18** under
  Hat-3 cold enforcement of the workspace clean-room policy
  (`docs/IMPLEMENTOR_ROUND.md`).

### Reset

- Crate reduced to a minimal `oxideav_core::register!` stub on
  2026-05-18. Round 1 (above) restores a real but minimal parse path.
  The crates.io version (`0.0.2`) is preserved on the new master to
  avoid breaking downstream version pins; the published versions on
  crates.io will be yanked by the maintainer.

### Next

- `svar(n)` signed reader + per-block command stream (`spec/03`).
- Function-code resolution against the nine-fixture corpus
  (`spec/04`).
- Predictor + Rice residual decode (`spec/03` + `spec/05`).
- `oxideav-core` `Decoder` / `Encoder` integration (registry wiring).
