# oxideav-shorten

Pure-Rust Shorten lossless audio decoder for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

## Status

**Round 1 — clean-room rebuild from `docs/audio/shorten/`.** Decodes
v2/v3 wire format pinned in `spec/00..05` (with v1 syntactically
accepted; no v1 fixture is reachable to confirm the v1 layout). Mono
and stereo PCM output for the three pinned filetypes (`u8 = 2`,
`s16hl = 3`, `s16lh = 5`). All ten function codes (`DIFF0..3`,
`QUIT`, `BLOCKSIZE`, `BITSHIFT`, `QLPC`, `ZERO`, `VERBATIM`) are
implemented per `spec/04`. Self-roundtrip tests exercise DIFF0..3,
QLPC orders 1 and 2, BITSHIFT, ZERO, VERBATIM, and BLOCKSIZE
sub-block-size overrides.

The crate is a fresh orphan `master`; the previous implementation was
retired alongside the OxideAV docs audit dated 2026-05-06. The prior
history is preserved on the `old` branch for archival and is forbidden
input for the rebuild.

## What's not yet implemented (round 2+ candidates)

- **Bit-stream-corpus byte-exact decode** of the public `.shn` fixture
  set (`F1..F18`). The mean-estimator residual ±1 drift documented in
  `audit/01-validation-report.md §8.1` is the known-bounded gap.
- **Production encoder.** The crate ships a `#[cfg(test)]` minimal
  encoder for self-roundtrip; a real encoder belongs in round 2.
- **Container demuxer.** `.shn` files have no container per se (the
  bit stream is the file). A simple raw-file demuxer can be added in
  round 2 alongside the production encoder.

## Cargo features

- **`registry`** (default): wire the crate's `register(ctx)` entry
  point into `oxideav-core`'s codec registry. Disable for standalone
  builds that want the decoder without the framework dependency.

## Public API

- [`decode`](https://docs.rs/oxideav-shorten) — single-shot decode of
  a complete `.shn` byte buffer into interleaved `i32` samples plus
  the parsed [`StreamHeader`].
- [`parse_header`] — parse the byte-aligned + variable-length-integer
  header in isolation.
- [`StreamHeader`], [`Filetype`], [`DecodedStream`] — output types.
- [`Error`], [`Result`] — crate-local error type.

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
