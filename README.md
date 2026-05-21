# oxideav-shorten

A pure-Rust Shorten (`.shn`) lossless audio codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 2 (2026-05-22).** The crate was
orphan-rebuilt on 2026-05-18 after a workspace audit found the prior
implementation derived from an external reference codebase.
Re-implementation is proceeding strictly against the in-tree clean-room
spec at [`docs/audio/shorten/`](../../docs/audio/shorten/) — the
TR.156-anchored chapter set `spec/00..05` and the tables it pins.

### What's wired up

Rounds 1+2 land the **file-header parser** plus the first slice of
the **per-block command stream**:

* Round 1 (`spec/01` + `spec/02`):
  * `ajkg` magic + one-byte format version at file offsets
    `0x00..=0x04`.
  * The six v2/v3 variable-length-integer parameter-block fields
    `H_filetype`, `H_channels`, `H_blocksize`, `H_maxlpcorder`,
    `H_meanblocks`, `H_skipbytes` read MSB-first under the `ulong()`
    two-stage form (`ULONGSIZE = 2`).
* Round 2 (`spec/02` §2.2 + `spec/03` + `spec/04`):
  * Signed `svar(n)` primitive — one's-complement folding (even
    unsigned → non-negative, odd unsigned → negative).
  * `FunctionCode` enum naming all ten v2/v3 codes 0..=9 plus
    cursor-advancement classification matching `spec/03` §3 per-
    command clauses.
  * `read_function_code()` — read + classify one command-header
    `uvar(FNSIZE = 2)`.
  * `read_verbatim_payload()` — full payload decode for
    `BLOCK_FN_VERBATIM` (length `uvar(5)` + `length × uvar(8)`
    opaque bytes), the command that recovers fixture `F1`'s 44-byte
    WAV preamble per `spec/02` §4.1 + §4.5 test `T6`.
  * `BLOCK_FN_QUIT` sentinel — bare function-code field, no payload
    (`spec/03` §3.8 + `spec/04` §2).

The combined surface is exercised by 26 tests (25 unit + 1
integration). The integration test composes the header parse with
the per-block dispatch to verify the post-header bit alignment
(43 bits relative to byte `0x05`, per `spec/02` §6.7) and reads a
synthetic VERBATIM-then-QUIT command pair end-to-end.

### What's not yet here

* Predictor commands `BLOCK_FN_DIFF0..3` and `BLOCK_FN_QLPC`
  (`spec/03` §3.1..3.5) — Rice residual decode + per-channel
  sample-history carry + running mean estimator.
* Housekeeping commands `BLOCK_FN_BLOCKSIZE`, `BLOCK_FN_BITSHIFT`,
  `BLOCK_FN_ZERO` payload state mutation (`spec/03` §3.6 / §3.7 /
  §3.9). The function-code classification surfaces them as
  `Error::BlockCommandNotImplemented` for now.
* `oxideav-core` `Decoder` / `Encoder` integration — `register(ctx)`
  remains a no-op until the predictor commands land.

## License

MIT — see [LICENSE](./LICENSE).
