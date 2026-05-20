# oxideav-shorten

A pure-Rust Shorten (`.shn`) lossless audio codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 1 (2026-05-20).** The crate was
orphan-rebuilt on 2026-05-18 after a workspace audit found the prior
implementation derived from an external reference codebase.
Re-implementation is proceeding strictly against the in-tree clean-room
spec at [`docs/audio/shorten/`](../../docs/audio/shorten/) — the
TR.156-anchored chapter set `spec/00..05` and the tables it pins.

### What's wired up

Round 1 lands the **file-header parser** of `spec/01-stream-header.md`:

* `ajkg` magic + one-byte format version at file offsets `0x00..=0x04`
  (`spec/01` §1).
* The six v2/v3 variable-length-integer parameter-block fields of
  `spec/01` §3 — `H_filetype`, `H_channels`, `H_blocksize`,
  `H_maxlpcorder`, `H_meanblocks`, `H_skipbytes` — read MSB-first
  under the `ulong()` two-stage form of `spec/02` §3.

The parser is exercised by 13 unit tests, including a byte-exact
decode of fixture `F1`'s first eleven bytes
(`61 6A 6B 67 02 FB B1 70 09 F9 25`) against the field-level values
the spec asserts in `spec/02` §6, and rejection paths for short
buffers, bad magic, unsupported versions, and v1 (whose parameter
block layout differs from v2/v3 and is open §9.4 candidate #2).

### What's not yet here

* Variable-length signed (`svar(n)`) reader, per-block command stream
  (`spec/03`), function-code resolution (`spec/04`), per-channel
  predictor state (`spec/05`).
* `oxideav-core` `Decoder` / `Encoder` integration — `register(ctx)`
  remains a no-op until the per-block decode lands.

## License

MIT — see [LICENSE](./LICENSE).
