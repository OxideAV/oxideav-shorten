# oxideav-shorten

Pure-Rust **Shorten** (`.shn`) lossless audio decoder. Zero C
dependencies, no FFI, no `*-sys` crates.

Shorten is Tony Robinson's 1994 lossless waveform compressor (CUED
technical report TR.156, *"SHORTEN: Simple lossless and near-lossless
waveform compression"*) — the historical predecessor to FLAC. The
bitstream uses fixed polynomial predictors (forward-difference orders
0..3) and quantised LPC, with all integers Golomb-Rice coded.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-shorten = "0.0"
```

## Quick use

The decoder exposes itself through the `oxideav-core` codec registry.
Register, build a decoder for the `shorten` codec id, then feed
packets:

```rust,no_run
use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, SampleFormat, TimeBase};

let mut codecs = CodecRegistry::new();
oxideav_shorten::register_codecs(&mut codecs);

let mut params = CodecParameters::audio(CodecId::new("shorten"));
params.sample_rate = Some(44_100);
params.channels = Some(2);
params.sample_format = Some(SampleFormat::S16);

let bytes = std::fs::read("song.shn").unwrap();
let mut dec = codecs.make_decoder(&params).unwrap();
let pkt = Packet::new(0, TimeBase::new(1, 44_100), bytes);
dec.send_packet(&pkt).unwrap();
let Frame::Audio(af) = dec.receive_frame().unwrap() else { return };
println!("decoded {} samples per channel", af.samples);
```

## Round-1 coverage

- Versions 0/1/2 (the entire wild + canonical encoder universe; v2
  rounding biases applied where required).
- All fixed DIFFn predictors (orders 0..3).
- QLPC predictor with quantised coefficients, `maxnlpc <= 8`.
- ZERO blocks (silence shortcut).
- BLOCKSIZE / BITSHIFT / VERBATIM / QUIT non-audio commands.
- All `internal_ftype` values 1..6 (S8 / U8 / S16HL / U16HL / S16LH /
  U16LH).
- Mono and multi-channel decode (channels independent — Shorten does
  no inter-channel decorrelation).
- Per-channel `nmean`-deep mean FIFO with v >= 2 rounding bias and
  bitshift compensation.
- Output is always packed S16 interleaved; 8-bit and U16 inputs are
  normalised to S16 before emission.

## Round-2 work (roadmap)

The following are not yet implemented but planned:

- **Streaming over multiple packets.** Today each packet is treated
  as a self-contained `.shn` byte stream. A real `.shn` demuxer and
  packetiser would split per audio block and let the decoder consume
  one block at a time, preserving inter-block predictor state across
  packet boundaries.
- **Native bit-depth output.** S8 / S32 outputs would preserve the
  source's full dynamic range without the lossy widen-to-S16
  normalisation step.
- **Demuxer / probe registration in `oxideav-container`.** Today
  callers must hand the decoder a packet directly with the
  appropriate `CodecParameters`.
- **Encoder.** No upstream pure-Rust Shorten encoder exists; one
  would close the round-trip loop.
- **Sample-rate / bits-per-coded-sample inference from VERBATIM.**
  The leading FN_VERBATIM capsule carries the original RIFF/AIFF
  header; today it is read and discarded. Parsing it would let
  the decoder derive `sample_rate` and `bits_per_coded_sample`
  from the bitstream itself instead of relying on
  `CodecParameters` from the caller.
- **Public-domain `.shn` fixture coverage.** Today the integration
  tests are hand-built bitstreams; a real-world fixture would
  exercise the end-to-end path against decoder output captured by
  ffmpeg's reference decoder.

## License

[MIT](LICENSE) — Karpelès Lab Inc., 2026.
