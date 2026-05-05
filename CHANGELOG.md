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

- **QLPC encoder** (round 2): Levinson-Durbin auto-correlation over
  the combined history+block signal → floating-point LPC coefficients
  → quantised integers at `qshift = 5` (multiply by 2^5 = 32, round,
  clamp to i16 range). The QLPC candidate is included in the
  per-block predictor race (DIFF0/1/2/3 + QLPC orders 1..=maxnlpc)
  and wins when it yields a smaller sum of `|residual|`. Enabled via
  `ShortenEncoderConfig::with_maxnlpc(N)`.
- **BITSHIFT encoder** (round 2): detects consistent trailing zero
  bits across all channels' current block using
  `samples[i].trailing_zeros()`; emits FN_BITSHIFT when the shift
  count changes; right-shifts samples before prediction and the
  decoder left-shifts the output to restore the original magnitude.
  Enabled via `ShortenEncoderConfig::with_bitshift(true)`.
- **Real WAV leading FN_VERBATIM**: encoder now builds a proper
  44-byte RIFF/WAV header (sample rate, bit depth, channel count)
  instead of the previous 44-byte zero placeholder. ffmpeg and
  shntool require this to identify the audio format.
- `ShortenEncoderConfig::with_sample_rate(u32)` — sets the sample
  rate written into the VERBATIM WAV header (default 44 100 Hz).
- `ShortenEncoderConfig::with_maxnlpc(u32)` — sets the maximum QLPC
  order (0 = disabled, default 0).
- `ShortenEncoderConfig::with_bitshift(bool)` — enables automatic
  BITSHIFT detection (default false).
- **ffmpeg cross-decode tests** (4 new): DIFFn, QLPC, BITSHIFT, and
  QLPC+BITSHIFT+stereo encoder outputs all pass ffmpeg `shorten`
  decode bit-exactly on this machine (ffmpeg 8.1).
- 9 new roundtrip tests exercising QLPC (mono, stereo, with nmean)
  and BITSHIFT (consistent trailing zeros, size reduction, combined
  with QLPC, zero-block shortcut, stereo).

### Round-3 backlog

- **Mid-stream FN_BLOCKSIZE** — would let the encoder accept
  arbitrary sample counts (today it requires the per-channel sample
  count to be a multiple of `blocksize`).
- **Iterated Rice-`k` search** — the closed-form
  `floor(log2(mean(|r|)))` is good but not optimal; iterate
  `k-1 / k / k+1` and pick the actual minimum encoded length.
- **Streaming / multi-packet encode** — produce one block per
  `Encoder::send_frame` call, preserving inter-block state across
  call boundaries.
- Streaming decode over multiple packets (today each packet is a
  self-contained `.shn` byte stream).
- Native S8 / S32 output instead of always-S16 widening.
- `.shn` demuxer + probe in `oxideav-container`.
- Parse the leading FN_VERBATIM capsule for sample-rate /
  bits-per-coded-sample (decoder side).
- Public-domain `.shn` fixture coverage in `tests/data/`.

### Changed (round 2)

- `ShortenEncoderConfig` gains `sample_rate: u32`, `maxnlpc: u32`,
  and `enable_bitshift: bool` fields (all with backward-compatible
  defaults).
- The header `maxnlpc` ulong field now reflects the actual `maxnlpc`
  config value (previously hardcoded to 0); `nwrap` is set to
  `max(3, maxnlpc)` in both encoder and decoder paths.
- `ChannelState::finish_block` now takes a `bitshift: u32` parameter
  to mirror the decoder's mean-FIFO bitshift compensation exactly.

### Round-1 additions (documented in 0.0.1)

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
