# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.2](https://github.com/OxideAV/oxideav-shorten/compare/v0.0.1...v0.0.2) - 2026-05-03

### Other

- drop duplicate semver_check key
- replace never-match regex with semver_check = false
- clean the comment+block remnant from the prior bad sed
- drop nested [workspace] block
- Add round-1 Shorten encoder (DIFFn predictors only, no QLPC yet)

### Added

- Pure-Rust **encoder** (round 1): DIFFn (orders 0..3) predictors
  only, FN_ZERO silence shortcut, per-block predictor selection by
  minimum sum-of-`|residual|`, per-block Rice-`k` from
  `floor(log2(mean(|r|)))`, per-channel state mirroring the decoder
  (3-deep history ring + `nmean`-deep mean FIFO), round-robin channel
  interleave. Mirrors `decoder.rs` exactly so the encode→decode
  bit-exact round-trip is the binding correctness guard.
- Bit-exact roundtrip integration suite: mono + stereo S16LE, mono
  S16BE, mono U8/S8, mono U16LE, multi-block, `nmean=4`, predictor
  selector spot-checks, FN_ZERO emission, error-path coverage.
- Public types: `ShortenEncoder`, `ShortenEncoderConfig`,
  `ShortenFtype`, `BlockInfo`.

### Round-2 backlog

- **QLPC encoder** — needs Levinson-Durbin coefficient estimation +
  quantisation at `qshift = LPCQUANT = 5` to match the decoder. Once
  in, the round-trip suite gains nontrivial-LPC test coverage and
  the encoder's compression ratio on tonal content jumps.
- **BITSHIFT encoder** — for streams with consistent low-zero bits
  (24-bit-in-32-bit containers, etc.); detect via per-block bitwise
  AND of the residuals and emit FN_BITSHIFT before the affected blocks.
- **Mid-stream FN_BLOCKSIFT** — would let the encoder accept
  arbitrary sample counts (today it requires the per-channel sample
  count to be a multiple of `blocksize`).
- **Iterated Rice-`k` search** — the closed-form
  `floor(log2(mean(|r|)))` is good but not optimal; iterate
  `k-1 / k / k+1` and pick the actual minimum encoded length.
- **Streaming / multi-packet encode** — produce one block per
  `Encoder::send_frame` call, preserving inter-block state across
  call boundaries.
- **Synthesised RIFF/AIFF leading FN_VERBATIM** — today the encoder
  emits a 44-byte zero placeholder. A real WAV header (with the
  caller's sample rate + bit depth) would let downstream demuxers
  reconstruct the original container without external metadata.
- Streaming decode over multiple packets (today each packet is a
  self-contained `.shn` byte stream).
- Native S8 / S32 output instead of always-S16 widening.
- `.shn` demuxer + probe in `oxideav-container`.
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
