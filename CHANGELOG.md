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
  end-to-end packet emission.

### Changed

- Clean-room rebuild from a fresh orphan `master`. The previous
  implementation was retired by the OxideAV docs audit dated
  2026-05-06; the prior history is preserved on the `old` branch.
  See `README.md` for the rebuild scope and the strict-isolation
  workspace the Implementer rounds drew from.
