# oxideav-shorten

A pure-Rust Shorten (`.shn`) lossless audio codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 3 (2026-05-22).** The crate was
orphan-rebuilt on 2026-05-18 after a workspace audit found the prior
implementation derived from an external reference codebase.
Re-implementation is proceeding strictly against the in-tree clean-room
spec at [`docs/audio/shorten/`](../../docs/audio/shorten/) — the
TR.156-anchored chapter set `spec/00..05` and the tables it pins.

### What's wired up

Rounds 1+2+3 land the **file-header parser**, the **per-block command
dispatch**, and the **polynomial-difference predictor kernels** of the
integer-PCM decode path:

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
* Round 3 (`spec/03` §3.1..§3.4 + `spec/05` §1 + §3):
  * `PolyOrder` enum naming the four `BLOCK_FN_DIFF0..3` polynomial-
    difference predictor orders + the `FunctionCode -> PolyOrder`
    mapping for command-dispatch consumers.
  * `ChannelCarry` — most-recent-first per-channel sample-history
    buffer (`carry[0] = s(t-1)`, `carry[1] = s(t-2)`, …) with the
    `spec/05` §1.3 update rule (short-block second-clause included).
  * `ENERGYSIZE = 3` exposed plus the `+1`-adjusted residual mantissa
    width pinned in `spec/05` §3 / test `T15`.
  * `decode_diff_block()` — full payload decode for `BLOCK_FN_DIFFn`:
    energy (`uvar(3)`) + `bs × svar(energy + 1)` residuals; applies
    the order-`n` polynomial-difference reconstruction recurrence
    (`spec/03` §3.1..§3.4 / TR.156 equations 3..10) in `i64` with
    `SampleOverflow` narrowing checks on the `i64 -> i32` boundary.
  * The running mean estimator of `spec/05` §2 / §2.5 is **not**
    wired up yet — DIFF0 reconstruction is `s(t) = e₀(t) + 0`, which
    is byte-exact for the very first block of each channel since
    `mu_chan` is initialised to zero per §2.1.

The combined surface is exercised by **41 tests** (39 unit + 2
integration). The two integration tests compose the header parse with
the per-block dispatch: one verifies the VERBATIM-then-QUIT path
(round 2), the other verifies a multi-channel DIFF1-DIFF1-DIFF1-QUIT
sequence with the round-robin channel cursor of `spec/03` §2 and the
per-channel carry hand-off between consecutive same-channel blocks
(round 3).

### What's not yet here

* Running mean estimator (`spec/05` §2.5) — required for the DIFF0
  predictor to match streams whose first-of-channel block is not a
  DIFF0 block (since later DIFF0 blocks need a non-zero `mu_chan`).
* `BLOCK_FN_QLPC` quantised LPC predictor (`spec/03` §3.5) — the
  command-code dispatch surfaces it as
  `Error::BlockCommandNotImplemented`.
* Housekeeping commands `BLOCK_FN_BLOCKSIZE`, `BLOCK_FN_BITSHIFT`,
  `BLOCK_FN_ZERO` payload state mutation (`spec/03` §3.6 / §3.7 /
  §3.9). The function-code classification surfaces them as
  `Error::BlockCommandNotImplemented` for now.
* `oxideav-core` `Decoder` / `Encoder` integration — `register(ctx)`
  remains a no-op until the housekeeping commands + mean estimator
  land and a real full-fixture decode pipeline is wired up.

## License

MIT — see [LICENSE](./LICENSE).
