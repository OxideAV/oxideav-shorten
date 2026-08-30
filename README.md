# oxideav-shorten

[![CI](https://github.com/OxideAV/oxideav-shorten/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-shorten/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-shorten.svg)](https://crates.io/crates/oxideav-shorten) [![docs.rs](https://docs.rs/oxideav-shorten/badge.svg)](https://docs.rs/oxideav-shorten) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

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
  (matching lenient real-world decoders, `spec/05` §5.2). The same QUIT
  byte-boundary observation is exposed across **all three API layers**:
  the streaming `StreamDecoder` surfaces it via
  `stream_proper_len()` / `quit_padding()` / `trailer_len()` (the last
  being the appended-sidecar size), and both `oxideav_core::Decoder`
  trait wrappers (`ShortenDecoder` whole-stream + `ShortenStreamingDecoder`
  chop-anywhere) expose `stream_proper_len()` / `quit_padding()` — the
  streaming wrapper computing the byte-exact boundary even when the
  stream is delivered one byte per `send_packet`. The boundary computed
  bottom-up from the QUIT alignment is cross-validated against the
  `SHNAMPSK` detector's top-down `sidecar_start`: a dedicated test pins
  the three-way agreement (`spec/05` §5.2).
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
* **Decode-time resource bounds** — the three header `ulong()` fields
  that linearly size decoder allocations (`H_channels`,
  `H_maxlpcorder`, `H_meanblocks`) are validated against
  implementation-side safety caps (`MAX_CHANNELS` = 1024,
  `MAX_LPC_ORDER` = 1024, `MAX_MEANBLOCKS` = 4096) before any buffer is
  allocated, in both `decode_stream` and `StreamDecoder::new`. The spec
  pins no maximum on these fields, so a crafted ~20-byte header could
  otherwise force a multi-gigabyte allocation; an over-cap value now
  returns `Error::HeaderResourceTooLarge` (same spirit as the existing
  `BLOCKSIZE_MAX` / `MAX_COMMANDS` caps). No real Shorten stream
  approaches these caps.

Three `H_filetype` sample-format codes are pinned and packed:
`2` (`u8`, `U8P`), `3` (`s16hl`, big-endian `S16P`), and
`5` (`s16lh`, little-endian `S16P`).

## Fixture-anchored conformance tests

Beyond the encode/decode self-roundtrip suite, the decode path is
pinned against **external ground truth** — the reference encoder's decoded PCM
for the clean-room spec's fixture corpus, as transcribed into the spec's
behavioural footnotes. Each test reconstructs the exact spec-pinned wire
residual stream, drives it through the public `decode_stream` API, and
asserts the reference-encoder byte-exact output (not the crate's own encoder
output):

* `tests/f1_diff1_byte_exact.rs` — fixture `F1`'s first
  `BLOCK_FN_DIFF1` block (`spec/05` §3.1 / `T15`): energy-field encoded
  value `3` read at the `+1`-incremented width `svar(4)` →
  residuals `[4, 0, -26, 42, -17, -14]` → ch0 PCM
  `[4, 4, -22, 20, 3, -11]`. Also pins the negative half (the same bits
  read at `svar(3)` give the documented wrong stream).
* `tests/f4_zero_diff0_leading_run.rs` — fixture `F4`'s leading
  `BLOCK_FN_ZERO` + `BLOCK_FN_DIFF0` zero run (`spec/04` §6.1 / `T13`,
  `spec/05` §2): `ZERO` emits `mu_chan = 0` and a following `DIFF0` with
  zero residuals stays zero under the zero-initialised mean buffer.
* `tests/f2_bitshift_diff1_u8.rs` — fixture `F2`'s
  `VERBATIM → BITSHIFT(7) → DIFF1` `u8p` opening chain
  (`spec/04` §3.2 / `T11`): first non-zero ch0 sample at index 4 is
  `128 = 1 << 7`, exercising the verbatim-prefix carry, the
  `BLOCK_FN_BITSHIFT` state, and the `sample << bshift` emit together.

Beyond the fixture-anchored anchors, a set of hand-built bitstreams
drives every block-command kernel **byte-exact through the public
`decode_stream` driver** (the whole-stream path with the driver-owned
round-robin channel cursor + per-channel state), closing the
decoder-direct combinations that were previously only exercised at the
predictor/`MeanEstimator` level or via the encoder round-trip suite
(`tests/decode_stream_driver.rs`):

* `full_stream_diff2_diff3_carry_handoff_through_driver` — the order-2
  line-fit and order-3 quadratic-fit predictors with two-block
  per-channel carry hand-off (`spec/03` §3.3–§3.4).
* `full_stream_mean_estimator_diff0_and_zero_through_driver` — the
  non-zero running-mean estimator: `DIFF0` reconstructs `e0(t) + μ` and
  `ZERO` emits `μ` while the one-slot sliding window updates across five
  blocks, exercising the `spec/05` §2.5 integer arithmetic
  (`trunc_div(Σ + divisor/2, divisor)`) at the public API.
* `full_stream_qlpc_carry_handoff_through_driver` — the quantised-LPC
  predictor at `H_maxlpcorder = 3` with two-block carry hand-off,
  interleaved with `DIFF1` on the other channel (`spec/03` §3.5).
* `full_stream_partial_block_carry_retention_through_driver` — the
  `bs < CARRY_LEN` carry-retention path (`spec/05` §1.3 second clause):
  a `BLOCK_FN_BLOCKSIZE` override to `new_bs = 1` makes each size-1
  `DIFF3` block keep the older history that its `s(t-2)`/`s(t-3)`
  prediction reads.

## Malformed-input robustness

`tests/decode_robustness_fuzz.rs` pins the decoder's hostile-input
contract: both public entry points (`decode_stream` and
`StreamDecoder`) must return a `Result` and never panic, abort, or hang
on arbitrary bytes. A 62k-case deterministic corpus — truncation
prefixes, single-bit flips, PRNG byte substitutions, random
`ajkg`+version headers, and pure noise — is driven through both entry
points inside `catch_unwind` and asserted panic-free. The corpus is
generated by an in-test xorshift PRNG over baselines produced by the
crate's own `encode_stream` (no external data). This also exercises the
decode-time resource bounds under fuzzing (a crafted header cannot OOM
the decoder).

## Fuzzing

`fuzz/` is a cargo-fuzz sub-crate with five structure-aware targets,
run daily (and on demand) by the `Fuzz` workflow via the org-shared
`crate-fuzz.yml` (30-minute budget split across targets, corpus cached
between runs; a crash uploads the offending input and fails the job):

* **`parse_header`** — panic-free `parse_stream_header` on arbitrary
  bytes plus writer/parser equivalence: every accepted header must
  re-serialise through `write_stream_header` and re-parse identically.
* **`decode`** — the two public decode paths (`decode_stream` vs
  `StreamDecoder`) as a differential: identical samples / verbatim /
  QUIT boundary / padding on success, the identical `Error` on
  failure. Output is bounded target-side (documented sample/block
  caps) — a `BLOCK_FN_ZERO` run with a huge sub-block size expands
  legitimately, so the decoder itself is never truncated.
* **`encode_roundtrip`** — structure-aware: fuzz-derived headers
  (v2/v3, 1..=8 channels, block sizes 1..=2048, LPC orders 0..=40,
  mean windows 0..=8), bit depths 1..=24 and bshift 0..=31 are all
  mapped into the encoder's contract-valid domain, so `encode_stream`
  / `encode_stream_lossy` MUST succeed and both decode paths MUST
  reconstruct `(s >> bshift) << bshift` sample-exact — an `expect`
  fires only on a real defect.
* **`trailer`** — SHNAMPSK envelope rules on arbitrary tails
  (signature present ⇒ never "no trailer"; detect/split agreement)
  plus valid-stream + fuzz-shaped-sidecar composites pinning the
  top-down `len_u32` boundary against the decoder's bottom-up QUIT
  boundary.
* **`packet_chunking`** — the two `oxideav_core::Decoder` wrappers fed
  the same bytes across fuzz-chosen packet boundaries must agree with
  `decode_stream` (packed planes, `stream_proper_len`, termination).

The round-453 deep runs (≥ 25 min per target, ASAN) surfaced five
defects, all fixed with regression pins in
`tests/fuzz_regressions_r453.rs` (details in `CHANGELOG.md`): a
`BitWriter::write_uvar` width-32 shift overflow; `encode_stream`
accepting `version: 1` (whose layout is unpinned and whose streams the
crate's own decoder rejects); `flush()` reporting a clean `Eof` on a
truncated stream; `ShortenStreamingDecoder` missing the decode-time
resource bounds (an 87-byte header claiming 1.7 M channels reserved
> 8 GiB); and the streaming wrapper silently dropping a partial
channel-round at `BLOCK_FN_QUIT` where the whole-stream wrapper
rejects it.

## Not yet supported

Both remaining gaps are **blocked on the spec, not on this crate** — the
references in the clean-room allow-list (`docs/audio/shorten/spec/`) do
not pin them, and resolving either is a `spec/05` §8 `§9.4`-escalation
item (additional public `.shn` fixtures or a reference Shorten encoder
*binary*):

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
