# oxideav-shorten

A pure-Rust Shorten (`.shn`) lossless audio codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework. Implemented
from the in-tree clean-room specification at
[`docs/audio/shorten/`](../../docs/audio/shorten/).

## Capabilities

Both the decode and encode sides are complete for the integer-PCM
v2/v3 path:

* **Decoder** — full-stream decode (`decode_stream`) plus a streaming
  iterator (`StreamDecoder` / `decode_stream_iter`) that retains only
  the per-channel state and so decodes arbitrarily long files in
  constant memory. Handles the `ajkg` magic + version prefix, the
  six-field variable-length parameter block, all ten function codes
  (`DIFF0..3`, `QLPC`, `ZERO`, `VERBATIM`, `QUIT`, `BLOCKSIZE`,
  `BITSHIFT`), the per-channel sample-history carry, the running-mean
  estimator, and the per-stream bit-shift. On `QUIT` it consumes the
  zero-padding to the next byte boundary and exposes the byte-exact
  end of the SHN stream proper as `DecodedStream::stream_proper_len`
  (`spec/04` §2.1) — the authoritative split point for any out-of-band
  seek-table sidecar. It also **observes** those QUIT padding bits and
  exposes them as `DecodedStream::quit_padding` (a `BytePadding`):
  `spec/05` §4 pins the padding to all-zero with a 0..7 count, and
  `BytePadding::is_spec_conformant` checks that rule — while decode
  stays lenient and still accepts a stream whose padding is non-zero
  (matching FFmpeg, `spec/05` §5.2).
* **Encoder** — the whole-stream encode driver (`encode_stream`) takes
  an interleaved `&[i32]` PCM buffer and produces a `.shn` byte stream
  that `decode_stream` reconstructs sample-exact. A per-block selector
  (`select_predictor_auto`) derives and quantises QLPC coefficients,
  runs an order search, and scores every candidate predictor at the
  Rice-optimal residual energy across the full decoder-accepted band,
  choosing the cheapest per block. When the stream header sets
  `H_maxlpcorder > 0` the driver emits a genuine `BLOCK_FN_QLPC`
  command (function code `7`, `spec/04` §5) for blocks where the
  quantised-LPC predictor is cheapest — verified end-to-end by walking
  the produced bit stream and counting the emitted QLPC commands.
* **Framework wiring** (default `registry` feature) — `ShortenDecoder`
  / `ShortenStreamingDecoder` / `ShortenEncoder` implement the
  `oxideav_core::Decoder` / `Encoder` traits. The crate also exposes
  the direct `make_decoder` / `make_encoder` factories and a
  `register(ctx)` installer (codec ids `"shorten"` and
  `"shorten-streaming"`).
* **SHNAMPSK seek-table trailer** — `detect_shnampsk_trailer` /
  `split_off_shnampsk_trailer` separate the SHN-stream-proper bytes
  from the non-standard sidecar some distributed `.shn` files append.

Three `H_filetype` sample-format codes are pinned and packed:
`2` (`u8`, `U8P`), `3` (`s16hl`, big-endian `S16P`), and
`5` (`s16lh`, little-endian `S16P`).

## Not yet supported

* The eight `H_filetype` labels whose numeric codes the spec leaves
  unpinned (`ulaw`, `s8`, `s16`, `u16`, `s16x`, `u16x`, `u16hl`,
  `u16lh`). Unblocking needs additional fixtures.
* The seek-record internal schema of the SHNAMPSK sidecar (the trailer
  boundary is detected; the records themselves are out of scope).

## Usage

```toml
[dependencies]
oxideav-shorten = "0.1"
```

```rust,no_run
use oxideav_shorten::decode_stream;

let shn_bytes: Vec<u8> = std::fs::read("input.shn").unwrap();
let decoded = decode_stream(&shn_bytes).unwrap();
// decoded.channels: Vec<Vec<i32>>, decoded.verbatim: Vec<u8>
```

Disable default features for a standalone decoder build without the
framework dependency.

## License

MIT — see [LICENSE](./LICENSE).
