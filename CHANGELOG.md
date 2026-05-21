# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

- Running mean estimator sliding-window update (`spec/03` §3.12 /
  `spec/05` §2.5) — required for the DIFF0 reconstruction to match
  streams whose later DIFF0 blocks need a non-zero `mu_chan`.
- `BLOCK_FN_QLPC` quantised LPC predictor (`spec/03` §3.5) — order +
  coefficients (`svar(LPCQUANT = 2)` × `order`) + energy + residuals.
- Housekeeping commands `BLOCK_FN_BLOCKSIZE` / `BLOCK_FN_BITSHIFT`
  / `BLOCK_FN_ZERO` payload state mutation (`spec/03` §3.6 / §3.7 /
  §3.9).
- Full per-fixture decode driver (header + block-stream loop +
  channel cursor + bit-shift sample reformatting + verbatim-prefix
  emission) → an `oxideav-core` `Decoder` impl + `register(ctx)`
  registry wiring.
