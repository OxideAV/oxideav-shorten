# oxideav-shorten

A pure-Rust Shorten (`.shn`) lossless audio codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 6 (2026-05-24).** The crate was
orphan-rebuilt on 2026-05-18 after a workspace audit found the prior
implementation derived from an external reference codebase.
Re-implementation is proceeding strictly against the in-tree clean-room
spec at [`docs/audio/shorten/`](../../docs/audio/shorten/) — the
TR.156-anchored chapter set `spec/00..05` and the tables it pins.

### What's wired up

Rounds 1+2+3+4+5+6 land the **file-header parser**, the **per-block
command dispatch** (every code 0..=9 now has a payload decoder), the
**polynomial-difference predictor kernels**, the **per-channel running
mean estimator**, the **quantised-LPC predictor**, and the
**`BLOCK_FN_BLOCKSIZE` / `BLOCK_FN_BITSHIFT` housekeeping commands**
of the integer-PCM decode path:

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
* Round 4 (`spec/05` §2 + §2.4 + §2.5):
  * `MeanEstimator` — per-channel sliding-window mean estimator of
    length `H_meanblocks`, zero-initialised per `spec/05` §2.1.
    `mu_chan()` returns the channel-wide running mean and
    `record_block(&block)` slides one per-block mean into the window,
    both using the validation-corrected arithmetic of `spec/05` §2.5:
    `mu_blk = trunc_div(sum + bs/2, bs)` and
    `mu_chan = trunc_div(sum_of_slots + H_meanblocks/2, H_meanblocks)`
    with truncation toward zero (C semantics) and the always-positive
    `+ divisor/2` bias regardless of numerator sign.
  * `decode_diff_block()` now takes a `mu_chan: i64` parameter,
    consumed by `PolyOrder::Order0` per `spec/05` §2.3 (`s(t) =
    e₀(t) + mu_chan`) and ignored by the mean-invariant orders 1..3.
  * `fill_zero_block(bs, mu_chan)` — synthesises the `BLOCK_FN_ZERO`
    payload of `spec/05` §2.4: `bs` samples all equal to `mu_chan`.
    The dispatch layer routes the `FunctionCode::Zero` command into
    this helper (no wire payload follows the function-code field).
  * `H_meanblocks = 0` disabled branch: the estimator's `mu_chan()`
    stays at zero and `record_block()` is a no-op, reducing DIFF0
    to `s(t) = e₀(t)` as `spec/01` §3.5 specifies.
* Round 5 (`spec/03` §3.5 + `spec/02` §4.3/§4.4 + `spec/05` §3.2):
  * `LPCQSIZE = 2` / `LPCQUANT = 2` exposed — the per-block LPC-order
    field width and the signed quantised-coefficient width pinned in
    `spec/02` §4.3 / §4.4.
  * `decode_qlpc_block()` — full payload decode for `BLOCK_FN_QLPC`:
    order (`uvar(2)`) + `order` quantised coefficients (`svar(2)`) +
    energy (`uvar(3)`) + `bs × svar(energy + 1)` residuals; applies
    the general LPC reconstruction `s(t) = Σᵢ aᵢ·s(t-i) + e(t)` of
    `spec/03` §3.5 (TR.156 §3.2 eq. 1/2) in `i64` headroom with
    `SampleOverflow` narrowing on the `i64 -> i32` boundary.
    Coefficients are applied **without scaling** per `spec/03` §3.5;
    the predictor reads its `s(t-1)..s(t-order)` history from the
    per-channel carry (`spec/05` §1) for the leading samples and
    feeds each just-emitted sample back as the next prediction's
    most-recent past sample.
  * QLPC is **mean-invariant** (`spec/03` §3.12 / `spec/05` §2): its
    prediction is a linear function of past samples that already carry
    the channel mean, so `decode_qlpc_block` takes no `mu_chan`
    argument.
  * `Error::LpcOrderTooLarge` surfaces when a per-block order exceeds
    the carry length — `spec/03` §3.11 pins the carry at
    `max(3, H_maxlpcorder)` and §3.5 bounds the order at
    `H_maxlpcorder`, so a well-formed stream always satisfies
    `order ≤ carry.len()`.
* Round 6 (`spec/03` §3.6 + §3.7 + `spec/02` §4.6 + `spec/04` §3 + §4):
  * `BITSHIFTSIZE = 2` exposed — the per-stream bit-shift field width
    pinned in `spec/02` §4.6.
  * `BITSHIFT_MAX = 31` / `BLOCKSIZE_MAX = 1 MiB` — implementation
    safety caps; the housekeeping decoders surface
    `Error::BitshiftTooLarge` / `Error::BlockTooLarge` for over-cap
    parameters and `Error::ZeroBlockSize` for the degenerate
    `new_bs = 0` case the encoder never emits.
  * `read_blocksize_payload()` — full payload decode for
    `BLOCK_FN_BLOCKSIZE`: a single `ulong()`-encoded `new_bs` that
    the higher-level driver swaps into its running sub-block-size
    state per `spec/03` §3.6 (`F2`'s tail-block override at command
    11,377 carries `new_bs = 155` per `T12`).
  * `read_bitshift_payload()` — full payload decode for
    `BLOCK_FN_BITSHIFT`: a single `uvar(BITSHIFTSIZE)` bit-shift
    amount that the higher-level driver swaps into its running
    per-stream bit-shift state per `spec/03` §3.7 (`F5..F8`'s first
    `BLOCK_FN_BITSHIFT` parameter values are 1/4/8/12, matching the
    encoder's `-q N` invocation per `T10`). The carry stays in
    pre-shift form per `spec/05` §1.4; the driver applies the shift
    on emission, not into the carry.
  * Neither housekeeping command advances the channel cursor; the
    per-channel dispatch resumes on the same channel after the state
    update.

The combined surface is exercised by **81 tests** (76 unit + 5
integration). The five integration tests compose the header parse
with the per-block dispatch: VERBATIM-then-QUIT (round 2), multi-
channel DIFF1-DIFF1-DIFF1-QUIT with the round-robin cursor of
`spec/03` §2 (round 3), DIFF0-ZERO-DIFF0-QUIT exercising the
running-mean estimator's sliding-window update + `BLOCK_FN_ZERO`'s
`mu_chan` fill (round 4), QLPC-QLPC-QUIT exercising an order-2 LPC
predictor with the per-channel carry hand-off across two blocks
(round 5), and BITSHIFT-DIFF1-BLOCKSIZE-DIFF1-QUIT exercising the
two housekeeping commands' state updates plus the carry hand-off
across the BLOCKSIZE-override boundary (round 6).

### What's not yet here

* Full-fixture decode driver (header + post-header bit alignment +
  block-stream loop carrying the round-robin channel cursor + running
  sub-block-size + running bit-shift + per-channel carries + per-
  channel mean estimators across the entire block stream until
  `BLOCK_FN_QUIT`).
* `oxideav-core` `Decoder` / `Encoder` integration — `register(ctx)`
  remains a no-op until that decode driver lands; every per-block
  command 0..=9 now has a payload decoder, so the missing piece is
  the orchestration loop, not any individual command's parser.

## License

MIT — see [LICENSE](./LICENSE).
