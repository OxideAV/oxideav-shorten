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
  (matching lenient real-world decoders, `spec/05` §5.2).
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
* **Lossy `-q N` encode** (`encode_stream_lossy`) — the encoder-side
  dual of the decoder's `BLOCK_FN_BITSHIFT` handling (`spec/03` §3.7 /
  `spec/04` §3). It emits a `BLOCK_FN_BITSHIFT` command and discards the
  `bshift` low-order bits of every sample before prediction — exactly
  the form TR.156's `-q quantisation level` option produces (the
  `F5..F8` `-q ∈ {1, 4, 8, 12}` anchors of `spec/04` §3.1). The decode
  reconstructs `(s >> bshift) << bshift`, so the round-trip is
  near-lossless; `bshift = 0` is byte-identical to `encode_stream`.
* **Framework wiring** (default `registry` feature) — `ShortenDecoder`
  / `ShortenStreamingDecoder` / `ShortenEncoder` implement the
  `oxideav_core::Decoder` / `Encoder` traits. The crate also exposes
  the direct `make_decoder` / `make_encoder` factories and a
  `register(ctx)` installer (codec ids `"shorten"` and
  `"shorten-streaming"`). `ShortenEncoder` accepts a `"bitshift"`
  option selecting the lossy `-q N` quantisation level (default `0`,
  lossless).
* **SHNAMPSK seek-table trailer** — `detect_shnampsk_trailer` /
  `split_off_shnampsk_trailer` separate the SHN-stream-proper bytes
  from the non-standard sidecar some distributed `.shn` files append.

Three `H_filetype` sample-format codes are pinned and packed:
`2` (`u8`, `U8P`), `3` (`s16hl`, big-endian `S16P`), and
`5` (`s16lh`, little-endian `S16P`).

## Not yet supported

Both remaining gaps are **blocked on the spec, not on this crate** — the
references in the clean-room allow-list (`docs/audio/shorten/spec/`) do
not pin them, and resolving either is a `spec/05` §8 `§9.4`-escalation
item (additional public `.shn` fixtures or Tony Robinson's reference
encoder *binary*):

* The eight `H_filetype` labels whose numeric codes the spec leaves
  unpinned (`ulaw`, `s8`, `s16`, `u16`, `s16x`, `u16x`, `u16hl`,
  `u16lh`). Only `2`/`u8`, `3`/`s16hl`, `5`/`s16lh` are forced by the
  fixture corpus (`spec/05` §6); the other eight codes' numeric values
  are not derivable from the allow-listed references (`spec/01` §7
  item 1, `spec/05` §8 item 3). Unblocking needs one fixture per label.
* The seek-record internal schema of the SHNAMPSK sidecar (the trailer
  boundary is detected; the records themselves are explicitly out of
  scope per `spec/00` and `spec/05` §8 item 5).
* Format-version 1 and version 3 wire-format deltas — every reachable
  fixture is `v2` (`spec/05` §7); the v1/v3 deltas are not pinned
  (`spec/05` §8 item 4). The decoder accepts the `v1`/`v3` version
  byte but the layout differences are unverified.

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
