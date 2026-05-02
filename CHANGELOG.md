# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Round-2 backlog

- Streaming decode over multiple packets (today each packet is a
  self-contained `.shn` byte stream).
- Native S8 / S32 output instead of always-S16 widening.
- `.shn` demuxer + probe in `oxideav-container`.
- Pure-Rust encoder.
- Parse the leading FN_VERBATIM capsule for sample-rate /
  bits-per-coded-sample.
- Public-domain `.shn` fixture coverage in `tests/data/`.

## [0.0.1] - 2026-05-02

### Added

- Initial reverse-engineered clean-room Shorten decoder bootstrapped
  from `docs/audio/shorten/shorten-trace-reverse-engineering.md`
  (no upstream-source quotes consulted).
- Header parse (magic + version 0/1/2 + 6 ulong fields).
- All fixed DIFFn predictors (orders 0..3) and the QLPC predictor
  with quantised coefficients (`maxnlpc <= 8`).
- ZERO silence shortcut and the BLOCKSIZE / BITSHIFT / VERBATIM /
  QUIT non-audio commands.
- Per-channel `nmean`-deep running-mean FIFO with v >= 2 rounding
  bias and bitshift compensation.
- All `internal_ftype` values 1..6 normalised to packed S16
  interleaved output.
- 5 unit tests + 5 integration tests, all hand-built fixtures.
