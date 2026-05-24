# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Round 7 clean-room rebuild.** Full-stream decode driver landed
  against `docs/audio/shorten/spec/03-block-and-predictor.md` §2 +
  §3.6 + §3.7 + §3.8 + `spec/05-state-and-quirks.md` §1 + §1.4 + §2 +
  `spec/01-stream-header.md`:
  - `decode_stream(bytes) -> Result<DecodedStream>` — the orchestration
    loop the prior six rounds were building toward. Parses the header,
    seeks a fresh `BitReader` past the byte-aligned magic + version
    prefix and the variable-length parameter block, then dispatches
    per-block commands until `BLOCK_FN_QUIT`. It carries the
    round-robin channel cursor (`spec/03` §2; advanced modulo
    `H_channels` by sample-producing commands, unchanged by
    housekeeping commands), the running sub-block size (default
    `H_blocksize`, overridden by `BLOCK_FN_BLOCKSIZE` per §3.6), the
    running per-stream bit-shift (set by `BLOCK_FN_BITSHIFT` per §3.7),
    the per-channel sample-history carries (`spec/05` §1), and the
    per-channel running mean estimators (`spec/05` §2) across the whole
    stream.
  - `DecodedStream` — the driver's output struct: the parsed `header`,
    the `verbatim` byte prefix (`BLOCK_FN_VERBATIM` payloads in
    encounter order, `spec/03` §3.10), and `channels: Vec<Vec<i32>>`
    (one time-ordered sample vector per channel). A `channel_len`
    accessor is provided. Per `spec/05` §1.4 the per-channel carry
    stores the pre-shift sample form (so the predictor recurrences stay
    bit-shift invariant) and the driver applies the left-shift on
    emission only, guarding the `i64 -> i32` boundary with
    `SampleOverflow`.
  - `BitReader::skip_bits(n)` — unbounded MSB-first bit-skip that loops
    past the 32-bit `read_bits` single-call cap; used to position the
    driver's reader at the first per-block command after the header.
  - `MAX_COMMANDS` (~1 billion) — a command-count safety cap that
    bounds a malformed never-terminating stream; well below it sits
    fixture `F2`'s entire 11,380-command stream, so no realistic stream
    is rejected. Reaching the cap surfaces
    `BlockCommandNotImplemented`.
  - A zero-channel header is rejected (`Error::Truncated`) — the
    round-robin cursor is undefined for `H_channels = 0`.
  - New integration test (`tests/decode_stream_driver.rs`) decoding a
    single stereo v2 stream end-to-end through the public driver:
    `VERBATIM` (4-byte prefix) → `BITSHIFT(bshift=2)` → `DIFF1` ch0 →
    `DIFF0` ch1 → `BLOCKSIZE(new_bs=2)` → `DIFF1` ch0 (bs=2) → `QUIT`.
    It asserts the verbatim collection, the two-channel interleave, the
    bit-shift-on-emission / pre-shift-carry split, and the
    blocksize-override length change. Nine driver unit tests plus three
    `skip_bits` unit tests cover the per-command driver paths and edge
    cases (empty `QUIT`-only stream, truncated block stream, zero
    channels).
  - Total test count: 81 -> 94 (88 unit + 6 integration; +12 unit +1
    integration relative to round 6). With round 7 the integer-PCM
    decode path is end-to-end — every per-block command 0..=9 is
    dispatched by the driver.

- **Round 6 clean-room rebuild.** Housekeeping commands
  `BLOCK_FN_BLOCKSIZE` and `BLOCK_FN_BITSHIFT` landed against
  `docs/audio/shorten/spec/03-block-and-predictor.md` §3.6 + §3.7 +
  `spec/02-variable-length-coding.md` §4.6 +
  `spec/04-function-code-resolution.md` §3 + §4:
  - `BITSHIFTSIZE = 2` (per-stream bit-shift field width, `spec/02`
    §4.6), `BITSHIFT_MAX = 31` and `BLOCKSIZE_MAX = 1 MiB`
    (implementation safety caps) exposed as public constants.
    Compile-time `const _: () = assert!(...)` bounds keep the caps in
    sync with the housekeeping decoders' assumptions.
  - `read_blocksize_payload()` — full payload decode for
    `BLOCK_FN_BLOCKSIZE`: a single `ulong()`-encoded `new_bs` per
    `spec/03` §3.6. The driver swaps the returned value into its
    running sub-block-size state; subsequent predictor commands
    produce blocks of `new_bs` samples per channel until another
    `BLOCK_FN_BLOCKSIZE` or end of stream. Surfaces
    `Error::ZeroBlockSize` for `new_bs == 0` (the encoder never emits
    this) and `Error::BlockTooLarge` for over-cap values. `F2`'s
    tail-block override at command 11,377 carrying `new_bs = 155`
    (per `T12`) is the behavioural anchor.
  - `read_bitshift_payload()` — full payload decode for
    `BLOCK_FN_BITSHIFT`: a single `uvar(BITSHIFTSIZE)` per-stream
    bit-shift amount per `spec/03` §3.7. The driver swaps the
    returned value into its running bit-shift state; subsequent
    samples emitted by predictor commands are left-shifted by this
    amount before delivery to the PCM sink. Surfaces
    `Error::BitshiftTooLarge` for over-cap values. The per-channel
    carry stores the pre-shift form per `spec/05` §1.4 so the
    predictor recurrences continue to see the same integer
    relationships across the BITSHIFT boundary. `F5..F8`'s first
    `BLOCK_FN_BITSHIFT` parameter values 1/4/8/12 matching the
    encoder's `-q N` invocation (per `T10`) are the behavioural
    anchors.
  - Neither housekeeping command advances the channel cursor
    (`FunctionCode::advances_channel_cursor()` already returns
    `false` for both per round 2); the per-channel dispatch resumes
    on the same channel after the state update.
  - `Error::ZeroBlockSize` and `Error::BitshiftTooLarge(u32)` —
    new variants surfacing the two payload-rejection cases above.
  - New integration test (`tests/housekeeping_pipeline.rs`)
    composing the header parse with a five-command sequence:
    `BITSHIFT(bshift=4)` → `DIFF1` (default `H_blocksize = 8`,
    cumulative sum from zero carry) → `BLOCKSIZE(new_bs=4)` →
    `DIFF1` (4 samples at the new sub-block size, with predictor
    history supplied by the prior block's carry) → `QUIT`. Verifies
    the running sub-block-size + running bit-shift state cells, the
    cursor non-advancement of the housekeeping commands, and the
    carry hand-off across the BLOCKSIZE-override boundary.
  - Total test count: 68 -> 81 (76 unit + 5 integration; +12 unit +1
    integration relative to round 5). With round 6 every per-block
    command 0..=9 has a payload decoder.

- **Round 5 clean-room rebuild.** Quantised-LPC predictor landed
  against `docs/audio/shorten/spec/03-block-and-predictor.md` §3.5 +
  `spec/02-variable-length-coding.md` §4.3 / §4.4 +
  `spec/05-state-and-quirks.md` §3.2:
  - `LPCQSIZE = 2` (per-block LPC-order field width, `spec/02` §4.3)
    and `LPCQUANT = 2` (signed quantised-coefficient width, `spec/02`
    §4.4) exposed as public constants.
  - `decode_qlpc_block()` — full payload decode for `BLOCK_FN_QLPC`:
    order (`uvar(LPCQSIZE)`) + `order` quantised coefficients
    (`svar(LPCQUANT)`) + energy (`uvar(ENERGYSIZE)`) + `bs ×
    svar(energy + 1)` residuals, applying the general LPC
    reconstruction `s(t) = Σᵢ aᵢ·s(t-i) + e(t)` of `spec/03` §3.5
    (TR.156 §3.2 eq. 1/2) in `i64` headroom with `SampleOverflow`
    narrowing on the `i64 -> i32` boundary. Coefficients are applied
    **without scaling** per `spec/03` §3.5 (each `coef` is a small
    signed integer the decoder uses directly). The history window
    `s(t-1)..s(t-order)` is read from the per-channel carry
    (`spec/05` §1) for the leading block samples and each just-emitted
    sample feeds back as the next prediction's most-recent past
    sample. The predictor is mean-invariant (`spec/03` §3.12 /
    `spec/05` §2), so `decode_qlpc_block` takes no `mu_chan` argument.
  - `Error::LpcOrderTooLarge { order, carry_len }` — surfaces when a
    per-block order exceeds the carry length (the carry cannot supply
    the required history) or the implementation cap. `spec/03` §3.11
    pins the carry length at `max(3, H_maxlpcorder)` and §3.5 bounds
    the per-block order at `H_maxlpcorder`, so a well-formed stream
    always satisfies `order <= carry.len()`.
  - New integration test (`tests/qlpc_block_pipeline.rs`) exercising a
    single-channel QLPC-QLPC-QUIT sequence with `H_maxlpcorder = 2`,
    an order-2 predictor (`a1 = 2, a2 = -1`, mirroring the DIFF2
    line-fit), and verifying the per-channel carry hands the first
    block's tail samples into the second block's predictor history.
  - Total test count: 55 -> 68 (64 unit + 4 integration; +12 unit +1
    integration relative to round 4).

- **Round 4 clean-room rebuild.** Per-channel running mean estimator
  landed against `docs/audio/shorten/spec/05-state-and-quirks.md` §2 +
  §2.5 + §2.3 + §2.4:
  - `MeanEstimator` — sliding-window per-channel mean buffer of length
    `H_meanblocks`, zero-initialised at construction per `spec/05`
    §2.1; `record_block(&block)` appends per-block mean and evicts the
    oldest slot; `mu_chan()` returns the running mean of all slots.
    Both per-block mean and running mean use the validation-corrected
    arithmetic of `spec/05` §2.5 — `trunc_div(numerator + divisor/2,
    divisor)` with C-semantics truncation toward zero and the
    always-positive `+ divisor/2` bias regardless of numerator sign.
  - `H_meanblocks = 0` disabled branch — `mu_chan()` always returns
    zero, `record_block()` is a no-op (`spec/01` §3.5 / `spec/05`
    §2.1).
  - `decode_diff_block()` now takes a `mu_chan: i64` parameter,
    consumed by `PolyOrder::Order0` per `spec/05` §2.3
    (`s(t) = e₀(t) + mu_chan`) and ignored by the mean-invariant
    orders 1..3 per `spec/05` §2 introductory paragraph.
  - `fill_zero_block(bs, mu_chan)` — `BLOCK_FN_ZERO` payload helper
    per `spec/05` §2.4: emits `bs` samples all equal to `mu_chan`.
    The command carries no further wire bits after its function-code
    field; the helper is called directly by the dispatch layer.
  - New integration test (`tests/mean_estimator_pipeline.rs`)
    exercising a single-channel DIFF0-ZERO-DIFF0-QUIT sequence with
    `H_meanblocks = 4` and verifying that the running mean after each
    block (6, then 8 after DIFF0+ZERO) drives the subsequent block's
    samples bit-exactly.
  - Total test count: 41 → 55 (52 unit + 3 integration; +13 unit +1
    integration relative to round 3).

- **Round 3 clean-room rebuild.** Polynomial-difference predictor
  kernels of orders 0..3 landed against
  `docs/audio/shorten/spec/03-block-and-predictor.md` §3.1..§3.4 +
  `spec/05-state-and-quirks.md` §1 + §3:
  - `PolyOrder` enum naming the four `BLOCK_FN_DIFF0..3` orders +
    `from_function_code()` mapping for command-dispatch consumers.
  - `ChannelCarry` — per-channel sample-history buffer with most-
    recent-first indexing (`carry[0] = s(t-1)`, `carry[1] = s(t-2)`,
    `carry[2] = s(t-3)`) per `spec/05` §1.1; zero-initialised at
    construction per `spec/05` §1.2; `update_after_block()` refresh
    rule per `spec/05` §1.3 (with the short-block second-clause
    covered for sub-block-size override scenarios).
  - `ENERGYSIZE = 3` (per `spec/02` §4.2) exposed; the per-block
    energy is read as `uvar(3)` and the residuals as `svar(energy + 1)`
    per the `spec/05` §3 / `T15` "encoded value plus one" rule.
  - `decode_diff_block()` — full payload decode: energy +
    `bs × svar(width)` residuals, with the order-`n` polynomial-
    difference reconstruction recurrence (`s(t) = ŝ_n(t) + e_n(t)`
    per TR.156 equations 3..10) applied in `i64` headroom and
    narrowed back to `i32` with `SampleOverflow` checks on the
    boundary.
  - The running mean estimator of `spec/05` §2 is deliberately left
    at its initial-zero state for round 3; DIFF0 reconstruction is
    `s(t) = e₀(t) + 0` (byte-exact for the very first block of each
    channel since `mu_chan` is initialised to zero per `spec/05`
    §2.1). The sliding-window update of `spec/05` §2.5 lands in a
    later round.
  - `Error::EnergyTooLarge` / `Error::BlockTooLarge` /
    `Error::SampleOverflow` surface for over-cap energy widths,
    over-cap block sizes, and pathological reconstruction overflows.
  - New integration test (`tests/diff_block_pipeline.rs`) composing
    header parse + post-header bit alignment + three consecutive
    DIFF1 blocks with the round-robin channel cursor of `spec/03` §2
    + per-channel carry hand-off between channel-0's two blocks +
    a terminal QUIT sentinel.
  - Total test count: 26 → 41 (39 unit + 2 integration; +14 unit +1
    integration relative to round 2).

- **Round 2 clean-room rebuild.** Per-block command stream — first
  slice — landed against `docs/audio/shorten/spec/02` §2.2 / §4.1 /
  §4.5 + `spec/03` §3.8 / §3.10 + `spec/04` §2:
  - `BitReader::read_svar(n)` — signed variable-length reader using
    the one's-complement folding pinned in `spec/02` §2.2 (even
    unsigned → non-negative, odd unsigned → negative). Verified
    against the §2.2 example sequence and the n=0/n=2 boundary
    cases.
  - `FunctionCode` enum naming all ten v2/v3 codes 0..=9
    (`BLOCK_FN_DIFF0..3`, `BLOCK_FN_QUIT`, `BLOCK_FN_BLOCKSIZE`,
    `BLOCK_FN_BITSHIFT`, `BLOCK_FN_QLPC`, `BLOCK_FN_ZERO`,
    `BLOCK_FN_VERBATIM`) + `advances_channel_cursor()` matching the
    `spec/03` §3 per-command clauses.
  - `read_function_code()` reads + classifies one command-header
    field (`uvar(FNSIZE = 2)`).
  - `read_verbatim_payload()` — full payload decode for
    `BLOCK_FN_VERBATIM`: length `uvar(VERBATIM_CHUNK_SIZE = 5)` +
    `length × uvar(VERBATIM_BYTE_SIZE = 8)` opaque bytes. Embeds
    fixture `F1`'s 44-byte verbatim payload as a structural anchor
    (the spec/02 §4.1 / §4.5 `T6` decode) and round-trips it.
  - `Error::UnknownFunctionCode` / `Error::BlockCommandNotImplemented`
    surface for out-of-range or not-yet-implemented codes.
  - One integration test (`tests/header_then_verbatim.rs`)
    composing the header parse + post-header bit-alignment +
    VERBATIM + QUIT command sequence end-to-end.
  - Total test count: 13 → 26 (25 unit + 1 integration).

- **Round 1 clean-room rebuild.** File-header parser landed against
  `docs/audio/shorten/spec/01-stream-header.md` + `spec/02`:
  - `parse_stream_header(bytes)` returns the parsed
    [`ShortenStreamHeader`] (`version`, `filetype`, `channels`,
    `blocksize`, `maxlpcorder`, `meanblocks`, `skipbytes`) and the
    bit-stream position at which the per-block command stream
    begins.
  - MSB-first `BitReader` implementing `uvar(n)` and the two-stage
    `ulong()` form (`ULONGSIZE = 2` per `spec/02` §3 + §5).
  - 13 unit tests, including byte-exact verification on fixture
    `F1`'s first eleven bytes (`61 6A 6B 67 02 FB B1 70 09 F9 25`),
    rejection of short / bad-magic / unsupported-version buffers,
    and the `spec/01` §4 sample-history carry floor.

### Erased

- Prior master history was force-erased on **2026-05-18** under
  Hat-3 cold enforcement of the workspace clean-room policy
  (`docs/IMPLEMENTOR_ROUND.md`).

### Reset

- Crate reduced to a minimal `oxideav_core::register!` stub on
  2026-05-18. Round 1 (above) restores a real but minimal parse path.
  The crates.io version (`0.0.2`) is preserved on the new master to
  avoid breaking downstream version pins; the published versions on
  crates.io will be yanked by the maintainer.

### Next

- `oxideav-core` `Decoder` impl + `register(ctx)` registry wiring. The
  full-stream decode driver (`decode_stream`) landed in round 7, so the
  remaining gap is the sample-format byte-packing layer (the `spec/05`
  §6 file-type table mapping the reconstructed `i32` channel samples +
  the verbatim host-format envelope into the container the registry
  expects) and the `Decoder` trait surface itself.
- DIFFn / QLPC encoder path (the crate description advertises a
  "DIFFn encoder"); only the decode direction exists so far.
