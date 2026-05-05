# oxideav-shorten

Pure-Rust **Shorten** (`.shn`) lossless audio codec — full decoder
plus a round-1 DIFFn encoder. Zero C dependencies, no FFI, no
`*-sys` crates.

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

## Round-1 decoder coverage

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

## Encoder coverage

- Bitwriter primitive (inverse of the decoder's `BitReader`) and
  Rice/Golomb writers (unsigned, signed-zig-zag, adaptive ulong).
- Per-block predictor selection across **DIFF0/1/2/3 + QLPC** (greedy:
  pick the predictor with the smallest sum of `|residual|`).
- **QLPC** (Quantised Linear Predictor): Levinson-Durbin
  auto-correlation → floating-point LPC coefficients → quantised
  integers at `qshift = 5` (multiply by 2^5 = 32, round, clamp to
  i16). Enabled via `with_maxnlpc(N)`; the QLPC candidate is included
  in the per-block predictor race for orders 1..=N.
- **BITSHIFT**: detects consistent trailing zero bits across all
  channels' current block; emits FN_BITSHIFT when the shift count
  changes; samples are right-shifted before prediction and the decoder
  left-shifts the output to restore the original magnitude. Enabled
  via `with_bitshift(true)`.
- FN_ZERO silence shortcut for all-zero blocks.
- Per-block Rice-`k` selection from
  `max(0, floor(log2(mean(|residual|))))`.
- Per-channel state mirroring the decoder: `nwrap = max(3, maxnlpc)`
  history ring, `nmean`-deep mean FIFO with v >= 2 rounding bias.
- Round-robin channel interleave matching the decoder's emission order.
- All `internal_ftype` values 1..6 mapped from
  [`encoder::ShortenFtype`].
- Header magic + version `2` + 6 ulong fields + real 44-byte WAV
  leading FN_VERBATIM (sample rate, bit depth, channel count) +
  trailing FN_QUIT.
- ffmpeg cross-decode verified: all encoder outputs (DIFFn, QLPC,
  BITSHIFT, QLPC+BITSHIFT stereo) pass ffmpeg `shorten` decode
  bit-exactly.

```rust,no_run
use oxideav_shorten::{ShortenEncoder, ShortenEncoderConfig, ShortenFtype};

// DIFFn-only encoder (simple use case):
let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, /*channels*/ 2, /*blocksize*/ 256);
let mut enc = ShortenEncoder::new(cfg).unwrap();
let pcm: Vec<i32> = vec![/* interleaved S16 samples */];
let bytes = enc.encode(&pcm).unwrap();
std::fs::write("song.shn", bytes).unwrap();

// QLPC + BITSHIFT encoder (better compression):
let cfg = ShortenEncoderConfig::new(ShortenFtype::S16Le, 2, 256)
    .with_maxnlpc(8)        // enable QLPC up to order 8
    .with_bitshift(true)    // auto-detect trailing-zero bits
    .with_sample_rate(44_100);
let mut enc = ShortenEncoder::new(cfg).unwrap();
let bytes = enc.encode(&pcm).unwrap();
```

## Round-3 work (roadmap)

The following are not yet implemented but planned:

### Encoder

- **Mid-stream FN_BLOCKSIZE.** Today the encoder requires the
  per-channel sample count to be a multiple of `blocksize`; emitting
  FN_BLOCKSIZE before the trailing partial block would lift this.
- **Iterated Rice-`k` search.** The closed-form
  `floor(log2(mean(|r|)))` is cheap and good but not optimal —
  iterate `k-1 / k / k+1` and pick the actual minimum encoded length.
- **Streaming / multi-packet encode.** Produce one block per
  `Encoder::send_frame` call, preserving inter-block state.

### Decoder

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
