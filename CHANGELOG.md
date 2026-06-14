# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Round 304 clean-room rebuild.** `oxideav_core::Encoder` trait wiring —
  the frame-in / packet-out mirror of the round-8 `ShortenDecoder`
  (`spec/05` §6 + `spec/03` §2 + `spec/01` §3 + the `oxideav_core::Encoder`
  trait contract):
  - New `ShortenEncoder` adaptor: `send_frame` unpacks each planar
    `AudioFrame` into per-channel `i32` accumulators (reversing the
    `spec/05` §6 host-byte packing the decoder applies on emission);
    `flush` re-interleaves and runs `encode_stream` once, queuing exactly
    one `.shn` `Packet`; `receive_packet` drains it then returns `Eof`.
  - `H_filetype` derived from `params.sample_format` (`U8P` → `2`/`u8`,
    `S16P` → `5`/`s16lh` by default); a `"filetype"` codec option overrides
    to one of the three `spec/05` §6 pinned codes (`2`/`3`/`5`).
    `"blocksize"` / `"maxlpcorder"` / `"meanblocks"` options tune the
    remaining header fields with spec-default fallbacks (256 / 0 / 0).
  - `make_encoder(params)` direct factory + `register_encoder(reg)`
    installer (registered under the existing `"shorten"` codec id alongside
    the decoder factory); `register(ctx)` now wires both encode + decode.
  - Public constants `DEFAULT_BLOCKSIZE = 256` and
    `ENCODER_HEADER_VERSION = 2`.
  - +15 in-module unit tests + 2 integration tests
    (`tests/encoder_trait_roundtrip.rs`): registry-resolved encode → decode
    round-trip sample-exact (mono + multi-frame stereo + `u8`), the
    decoder-output → encoder re-encode byte-identity property, factory
    parameter validation, and `filetype` option override.

- **Round 297 clean-room rebuild.** Wide energy sweep up to the decoder's
  residual-width cap (`spec/02` §4.2 + `spec/05` §3), closing round 290's
  energy-sweep follow-up:
  - New `pub const MAX_ENERGY = 29` — the largest encoded energy whose
    residual mantissa width `e + 1` reaches the decoder's
    `MAX_RESIDUAL_WIDTH = 30` cap. `spec/02` §4.2 places no upper bound on
    the `uvar(ENERGYSIZE = 3)` energy-field value, so emitting an energy in
    `8..=29` is not a wire-format change.
  - New public helper `optimal_energy_for_residuals_wide(residuals)` — the
    rate-optimum scan over the full band `e ∈ 0..=MAX_ENERGY`, alongside
    the natural-band `optimal_energy_for_residuals`. The per-block
    sequencer now scores candidates against the wide variant, so a
    full-scale cold-carry block selects a wider energy and encodes
    sample-exact instead of surfacing `EncodeError::NoPredictorFits`.
  - The five per-block writers (`write_diff0..3_block`, `write_qlpc_block`)
    now accept `energy_encoded` up to `MAX_ENERGY` (formerly capped at
    `MAX_NATURAL_ENERGY = 7`), still rejecting values above `MAX_ENERGY`
    with `EncodeError::EnergyOutOfRange`.

- **Round 290 clean-room rebuild.** Whole-stream encode driver
  `encode_stream(header, samples, verbatim_prefix)` — the encoder mirror
  of `decode_stream` (`spec/03` §2 + §3.6 + §3.8 + §3.10 + `spec/04`
  §4.1 + `spec/05` §1 + §2 + §4):
  - Deinterleaves an interleaved `&[i32]` PCM buffer (`a(0), b(0),
    a(1), b(1), …` per `spec/03` §2 / TR.156 §3.1) into `H_channels`
    per-channel planes, partitions each into `H_blocksize` blocks, and
    emits them in the round-robin cursor order `decode_stream` consumes.
  - Carries the per-channel `ChannelCarry` (`spec/05` §1) and
    `MeanEstimator` (`spec/05` §2) across blocks, updating each from the
    pre-shift block in the same order as `decode_stream`'s
    `commit_block`, so the produced stream is reconstructed
    **sample-exact**.
  - Trailing partial block handled by a single `BLOCK_FN_BLOCKSIZE`
    override at the head of the tail round, mirroring fixture `F2`'s
    lone tail override (`new_bs = 155`, `spec/04` §4.1 `T12`).
  - Per-block predictor chosen by `select_predictor_auto` (ZERO /
    DIFF0..3 / auto-derived QLPC up to `H_maxlpcorder`); envelope =
    magic + version + parameter block + optional `BLOCK_FN_VERBATIM`
    prefix (`spec/03` §3.10), terminated by `BLOCK_FN_QUIT` + byte-align
    padding (`spec/05` §4).
  - New `EncodeError` variants `RaggedInterleave { samples, channels }`,
    `NoPredictorFits`, and `ZeroChannels`. `NoPredictorFits` surfaces
    when a block's best-predictor residual stream overflows the
    natural-energy band (`e ∈ 0..=MAX_NATURAL_ENERGY`) the selector
    scores — a documented limitation; widening the energy sweep up to
    the decoder's `MAX_RESIDUAL_WIDTH = 30` cap is a sequencer-layer
    refinement that does not change the wire format.
  - 14 in-module unit tests + 7 integration tests
    (`tests/encode_stream_pipeline.rs`): mono / stereo / 3-/5-channel
    round-robin, tail-block override, verbatim splice, constant-signal
    ZERO eligibility, LPC-recurrence material under `H_maxlpcorder = 2`,
    auto-QLPC never larger than DIFFn-only on LPC material, empty
    buffer, ragged / zero-channel rejection — all round-trip
    sample-exact through `decode_stream`.

- **Round 282 clean-room rebuild.** QLPC auto-derivation in the
  per-block predictor-selection sequencer (`spec/03` §3.5 + `spec/02`
  §4.3 + §4.4 + `spec/05` §1.1):
  - New public `select_predictor_auto(samples, mu_chan, carry,
    max_lpc_order)` / `evaluate_candidates_auto(..)`: the sequencer
    now derives and quantises the per-block LPC coefficient vector
    itself and runs the per-block order search, so `BLOCK_FN_QLPC`
    is auto-selected exactly when it is genuinely cheapest under the
    full cost model (function code + `uvar(LPCQSIZE)` order field +
    `order × svar(LPCQUANT)` coefficient-transmission overhead +
    energy field + Rice-n-optimal residuals). `max_lpc_order = 0`
    reproduces `select_predictor` bit-for-bit per `spec/03` §3.5's
    `H_maxlpcorder = 0` ⇒ polynomial-difference-only rule.
  - New public `derive_qlpc_coefs(samples, carry, order)`:
    least-squares normal equations over the block's carry-seeded
    prediction contexts (Gaussian elimination, partial pivoting),
    rounded to nearest integer — the projection onto `spec/03`
    §3.5's unscaled signed-integer coefficient domain. Degenerate
    systems drop the candidate.
  - New public `derive_qlpc_candidate(samples, carry,
    max_lpc_order)`: zero-anchored order search (`spec/02` §4.3's
    TR.156 anchor) over `0..=min(max_lpc_order, carry.len(),
    MAX_QLPC_ORDER, MAX_QLPC_AUTO_ORDER = 32)`; cost ties break
    toward the lower order. New public `MAX_QLPC_AUTO_ORDER`.
  - Measured on a 16-block period-6 integer-recurrence signal:
    8,768 bits via auto-QLPC vs 34,240 bits via the DIFFn-only
    selector (74 % saving, QLPC on 16/16 blocks); never worse than
    the legacy pick on any block (superset candidate set).
  - +13 in-module unit tests, +5 integration tests
    (`tests/encoder_qlpc_autoselect_pipeline.rs`) including
    bit-exact decode parity of auto-selected QLPC streams through
    `decode_stream` and byte-identical legacy reproduction at
    `max_lpc_order = 0`. Combined surface now 394 tests (was 376).

### Fixed

- **Rice-n energy sweep could select codes the decoder rejects**
  (latent since round 254, surfaced by the round-282 QLPC work but
  reachable through plain DIFFn selection): a sparse residual stream
  with one large outlier (e.g. a constant-50 block over a cold carry
  → DIFF1 residuals `[50, 0, …]`) drove the sweep to `e = 0`, whose
  outlier code needs a prefix-zero run beyond the decoder's 32-zero
  `uvar` cap — the emitted stream failed to decode with
  `OverflowingUvar`. `residual_bits_at_energy` and the sequencer's
  coefficient costing now treat over-cap codes as unrepresentable
  (`None`), so the sweep settles on a decodable energy; regression
  pinned by
  `rice_optimum_respects_decoder_prefix_cap_regression` with a full
  encode → decode roundtrip.

- **`BLOCK_FN_QUIT` encoding documentation aligned with the resolved
  `spec/04` §2 errata.** The function-code field `uvar(FNSIZE = 2)`
  of the QUIT value 4 is the **4-bit** pattern `0100`, not the 5-bit
  `00100` (which `spec/02` §2.1 decodes to value 8 = `BLOCK_FN_ZERO`).
  An earlier revision of `spec/04` §2 carried that arithmetic typo;
  the in-tree `write_uvar` / `read_uvar` primitives were always
  correct (they compute `(k << n) + m` from the worked-example
  algorithm), so the decode and round-trip behaviour are unchanged —
  only stale "5-bit `00100`" QUIT comments in `bitwriter.rs`,
  `block.rs`, and `encoder.rs` were corrected, and the prior
  `quit_command_at_byte_boundary_pads_to_zeroes` disclaimer that
  flagged the contradiction now records its resolution. The `ZERO`
  (value 8 = `00100`, 5 bits) references are left untouched — those
  were and remain correct.
- New behavioural unit tests pin the corrected encoding directly:
  - `write_quit_command_emits_uvar_of_four_four_bits` — asserts the
    bare QUIT field is exactly 4 bits and, at a byte boundary, packs
    to `0100 0000`.
  - `write_quit_command_at_nonzero_bit_offset_matches_spec04_f9_byte`
    — reproduces the `spec/04` §2.1 fixture-`F9` final byte `0x20`
    (`0010 0000`): a prior-residual trailing bit at byte position 0,
    the `0100` QUIT field at positions 1..4, three zero padding bits.
  - `write_quit_block_roundtrips_through_read_function_code` — the
    4-bit pattern the writer emits dispatches to `FunctionCode::Quit`
    on decode, confirming writer/reader agreement on the errata form.

### Added

- **Round 266 clean-room rebuild.** `BLOCK_FN_QLPC` auto-selection in
  the per-block predictor-selection sequencer (`spec/03` §3.5 +
  `spec/02` §4.3 + §4.4 + `spec/05` §3.1 + TR.156 §3.2):
  - New `Choice::Qlpc { coefs, energy, bits }` variant. The variant
    carries the caller-supplied quantised coefficient vector inside
    itself so `write_selected_block` dispatches the QLPC command
    without a separate coefficient hand-off.
  - New entry points `select_predictor_with_qlpc(samples, mu_chan,
    carry, qlpc_candidate: Option<&[i64]>) -> Option<Choice>` and
    `evaluate_candidates_with_qlpc(samples, mu_chan, carry,
    qlpc_candidate)`. The existing `select_predictor` and
    `evaluate_candidates` become thin wrappers that delegate to the
    new entry points with `qlpc_candidate = None`. Passing `None`
    reproduces the legacy behaviour byte-for-byte (a load-bearing
    backward-compat invariant the new test
    `select_with_qlpc_none_matches_legacy_select_predictor` pins).
  - QLPC cost metric: the same Rice-`n` statistical optimum as the
    DIFFn family (round 254 `optimal_energy_for_qlpc` +
    `residual_bits_at_energy`). Total command cost is
    `uvar(FNSIZE, FN_QLPC)` + `uvar(LPCQSIZE, order)` +
    `order × svar(LPCQUANT, coef)` + `uvar(ENERGYSIZE, energy)` +
    `bs × svar(energy + 1, residual)`.
  - QLPC candidate skipped (selector falls back to DIFFn) when
    `coefs.len() > MAX_QLPC_ORDER` or `coefs.len() > carry.len()`
    (mirrors `EncodeError::LpcOrderTooLarge`), any coefficient
    overflows `svar(LPCQUANT)` folding, the residual stream
    overflows `svar`, no natural energy fits the residuals, or
    `samples` is empty.
  - Tie-break priority extended to
    `ZERO > DIFF0 > DIFF1 > DIFF2 > DIFF3 > QLPC`. QLPC sits last
    because of its larger fixed overhead — the order + coefficient
    fields add a per-block constant the DIFFn family doesn't pay.
  - `write_selected_block` dispatches `Choice::Qlpc` to
    `write_qlpc_block`, reading the coefficient vector and energy
    from the variant. `bits_written` after the call equals
    `Choice::bits()` exactly (round-251 invariant extended to QLPC).
  - Tests: 10 new in-module unit tests in `sequencer.rs` +
    3 new integration tests in `tests/encoder_sequencer_pipeline.rs`.
    Total test count: 359 → 372 (281 in-module unit + 91 integration).
  - **Scope.** Round 266 closes the QLPC auto-selection follow-up
    called out in round 251 / 254. The remaining encoder-side QLPC
    gap is coefficient quantisation; the selector takes the
    candidate as `Option<&[i64]>` and is otherwise opaque to it.

- **Round 254 clean-room rebuild.** Rice-`n` statistical-optimum
  energy selection inside the per-block predictor-selection sequencer
  (`spec/02` §2.1 + §4.2 + `spec/05` §3 + TR.156 §3.3):
  - New public encoder helpers
    `residual_bits_at_energy(residuals, encoded_energy) -> Option<u64>`
    and `optimal_energy_for_residuals(residuals) -> Option<u32>`. The
    bit-count helper sums the per-sample `⌊u / 2^width⌋ + 1 + width`
    `svar` cost of `spec/02` §2.1 across a residual stream at any
    encoded energy `e` such that `e + 1 ≤ MAX_RESIDUAL_WIDTH`. The
    optimum helper sweeps `e ∈ 0..=MAX_NATURAL_ENERGY` and returns
    whichever minimises the bit count — the empirical equivalent of
    TR.156 §3.3 equation 21's `n ≈ log₂(log(2) · E(|x|))` over the
    natural-band of encoded energies the `uvar(ENERGYSIZE = 3)` field
    of `spec/02` §4.2 admits at its natural 4-bit width.
  - Per-predictor wrappers
    `optimal_energy_for_diff0` / `..diff1` / `..diff2` / `..diff3` /
    `..qlpc` mirror the existing `min_energy_for_*` family for symmetry
    at the call site; each one delegates to the shared scan.
  - `select_predictor` (and `evaluate_candidates`) now score every
    `BLOCK_FN_DIFFn` candidate at the Rice-`n` optimum rather than at
    the natural energy. Observed effects on the new test corpus:
    - Sparse seed-jump streams (e.g. `[3, 0, 0, 0]` from DIFF1 with
      a fresh carry) move from `e = 2` / 23 bits to `e = 0` / 18
      bits — a 22 % saving on this single block, equally for any
      DIFFn block whose residual stream has a single outlier and a
      long zero tail.
    - Arithmetic-progression streams (the standard "linear ramp"
      test case) move from `Diff1 { e = 3 }` (327-bit cost at
      `N = 64`) to `Diff2 { e = 0 }` (140-bit cost) — DIFF2's
      second-difference is non-zero in exactly one position, and
      under Rice-`n` the optimum prefers the order-2 predictor's
      sparser residual stream over DIFF1's constant non-zero one.
  - `Choice::bits()` continues to reflect the **emitted** bit count
    of the chosen `(predictor, energy)` pair; the writer side
    consumes the choice unchanged. Higher-layer rate planners can
    therefore continue to call `Choice::bits()` to budget per-block
    cost without re-deriving the metric.
  - Tests: 6 new in-module unit tests in `encoder.rs`
    (`residual_bits_at_energy_matches_bitwriter_actual_count` —
    crosschecks the cost helper against the BitWriter's actual
    `write_svar` output across 7 streams × 8 energies; energy-cap
    rejection; natural-coincidence on tight streams; sparse-stream
    optimum strictly smaller than natural; per-predictor-wrapper
    agreement; empty-stream rejection). Sequencer tests rewritten to
    pin the new optimum behaviour explicitly: the seed-jump unit
    test moves from natural to optimum cost; the arithmetic-ramp
    selector test now asserts DIFF2 instead of DIFF1 with the
    Rice-`n` rationale spelled out. The integration test
    `mono_sequencer_picks_diff_for_arithmetic_ramp` is similarly
    updated to assert DIFF2 and round-trips end-to-end through
    `decode_stream`. The QLPC auto-selection follow-up (which
    requires a candidate coefficient-quantisation pass) stays
    deferred per the unchanged sequencer module-level note.

- **Round 251 clean-room rebuild.** Per-block predictor-selection
  sequencer (`spec/03` §3.1..§3.4 + §3.9 + `spec/02` §2.1 + §2.2 +
  `spec/05` §2.3 + §2.4):
  - New `select_predictor(samples, mu_chan, carry) -> Option<Choice>`
    higher-layer entry point. Computes the natural-energy total
    encoded bit cost of every eligible candidate among
    `BLOCK_FN_DIFF0..3` and `BLOCK_FN_ZERO`, picks the cheapest, and
    returns a `Choice` enum the caller hands to
    `write_selected_block`. Ties break in priority order
    `ZERO > DIFF0 > DIFF1 > DIFF2 > DIFF3`. Returns `None` for an
    empty block and for blocks whose residuals overflow every natural
    energy while ZERO is ineligible (the caller can still encode such
    a block manually by picking an explicit wider energy through the
    per-predictor writer).
  - New public `Choice` enum with variants `Zero { bits }`,
    `Diff0 { energy, bits }`, `Diff1 { energy, bits }`,
    `Diff2 { energy, bits }`, `Diff3 { energy, bits }`. Each variant
    carries the natural energy parameter the writer will use plus
    the total encoded bit count of the command. `Choice::bits()` and
    `Choice::function_code()` accessors.
  - New `write_selected_block(writer, choice, samples, mu_chan,
    carry)` — dispatches the `Choice` to the matching per-predictor
    writer (`write_zero_block` / `write_diff0_block` /
    `write_diff1_block` / `write_diff2_block` / `write_diff3_block`).
  - New `evaluate_candidates(samples, mu_chan, carry) -> Vec<Choice>`
    accessor returning every eligible candidate in priority order;
    useful for inspecting why a particular block selects a
    particular predictor.
  - `BLOCK_FN_QLPC` is **not** part of auto-selection because the
    caller still owns coefficient quantisation per `spec/03` §3.5
    and TR.156 §3.2's Laplacian-distribution rule; a future round
    can extend the selector to accept a candidate coefficient
    vector.
  - Tests: 15 new in-module unit tests + 7 new integration tests
    (`tests/encoder_sequencer_pipeline.rs`) cover: `uvar_bits` /
    `svar_bits` length formulas against `spec/02` §2.1 worked
    examples and `BitWriter`'s actual emitted bit count; ZERO-token
    5-bit cost; ZERO eligibility against `mu_chan`; tie-break
    priority (ZERO wins over DIFF0 when both fit); selector picks
    DIFF1 for an arithmetic-progression input where the first-
    difference stream is small; full round-trip through
    `decode_stream` for mono / stereo / back-to-back blocks / mixed
    predictor streams; bits-written delta equals `Choice::bits()`
    exactly (load-bearing for higher-layer rate planners).

- **Round 244 clean-room rebuild.** Encoder-side `BLOCK_FN_BLOCKSIZE`
  housekeeping writer (`spec/03` §3.6 + `spec/04` §4):
  - New `write_blocksize_command(writer, new_bs)` primitive emits the
    full `<fn=5> <new_bs>` command. The function code is written as
    `uvar(FNSIZE = 2)` over `FN_BLOCKSIZE = 5`; the `new_bs` payload
    is the two-stage `ulong()` form of `spec/02` §3
    (`uvar(ULONGSIZE = 2)` over the per-value mantissa width followed
    by `uvar(width)` over the value), with the mantissa width chosen
    by the same `natural_ulong_width` minimum-width rule
    `write_parameter_block` applies to the six header fields.
  - New public constant `FN_BLOCKSIZE: u32 = 5` (matching the
    existing per-command numeric `FN_BITSHIFT` / `FN_QUIT` /
    `FN_DIFF0..3` / `FN_QLPC` / `FN_VERBATIM` / `FN_ZERO`).
  - New `EncodeError` variants `ZeroBlocksize` (rejection mirror of
    the decoder-side `Error::ZeroBlockSize` per `spec/03` §3.6's
    "encoder never emits zero-sample overrides") and
    `BlocksizeOutOfRange(u32)` (rejection mirror of the decoder-side
    `Error::BlockTooLarge` at the `BLOCKSIZE_MAX = 1 MiB`
    implementation safety cap). A rejected writer call surfaces the
    error without committing any partial command bytes.
  - Decode semantics on the round-7 driver side: the decoder installs
    the returned value as its running sub-block size; subsequent
    predictor commands (`DIFF0..3` / `QLPC` / `ZERO`) produce blocks
    of `new_bs` samples per channel until the next
    `BLOCK_FN_BLOCKSIZE` command or end-of-stream. The command does
    **not** advance the channel cursor; the per-channel dispatch
    resumes on the same channel after the override takes effect
    (`spec/03` §3.6).
  - Tests: 6 new in-module unit tests + 5 new integration tests
    (`tests/encoder_blocksize_pipeline.rs`) cover: function-code
    constant matches `spec/04` §4 (`FN_BLOCKSIZE = 5`); bit counts
    match the `spec/02` §2.1 / §3 length formulas for the F2 T12
    anchor value (`new_bs = 155` → 18 bits total), the default
    `H_blocksize = 256` (19 bits), and the minimum `new_bs = 1`
    (9 bits); a representative spread (1, 2, 64, 155, 256, 1024,
    65536, `BLOCKSIZE_MAX`) round-trips through `read_function_code`
    + `read_blocksize_payload`; `new_bs = 0` rejection produces
    `ZeroBlocksize` with zero bits committed; `BLOCKSIZE_MAX + 1`
    rejection produces `BlocksizeOutOfRange` with zero bits
    committed; an exact byte-pattern pin (`new_bs = 1` → `0x5B 0x80`)
    corroborates the `uvar(FNSIZE = 2)` + `ulong()` MSB-first packing
    against `spec/02` §2.1's `0101` function-code prefix
    decomposition. Integration: mono shrinking override (default
    `H_blocksize = 8` → `new_bs = 3` tail block) round-trips
    byte-exact through `decode_stream`; stereo round-robin override
    exercises the channel-cursor non-advancement rule of `spec/03`
    §3.6; an override that resets `new_bs` to the same value as
    `H_blocksize` is admissible and round-trips; the exact `spec/04`
    §4.1 T12 anchor value `new_bs = 155` round-trips end-to-end;
    the degenerate single-sample override `new_bs = 1` round-trips.

- **Round 241 clean-room rebuild.** Typed `H_filetype` accessor
  surfacing the three numeric codes `spec/05` §6 pins behaviourally:
  - New public `Filetype` enum with variants `U8` (wire value 2 /
    TR.156 label `u8` / fixture `F2`), `S16HL` (wire value 3 /
    `s16hl` / fixture `F3`), and `S16LH` (wire value 5 / `s16lh` /
    fixture `F1`). The enum is `#[non_exhaustive]` so a later round
    can add the remaining eight TR.156 labels (`ulaw`, `s8`, `s16`,
    `u16`, `s16x`, `u16x`, `u16hl`, `u16lh`) once `spec/05` §8 open
    §9.4 candidate #3 is closed — without breaking callers.
  - New `ShortenStreamHeader::filetype_pinned() -> Option<Filetype>`
    instance method that returns `Some(variant)` when the raw
    `H_filetype` field falls in the `{2, 3, 5}` pinned-by-fixture
    set and `None` for every other numeric value (no guessing for
    unpinned labels). Round-trip helpers `Filetype::from_wire(u32)`
    + `Filetype::wire_value()` plus typed `label()`,
    `bytes_per_sample()`, `is_signed()`, and `is_little_endian()`
    accessors mirror the `spec/05` §6 table row-by-row (`u8` → 1
    byte / unsigned / byte-order-irrelevant; `s16hl` → 2 bytes /
    signed / big-endian; `s16lh` → 2 bytes / signed /
    little-endian).
  - Tests: 8 new in-crate unit tests in `header.rs` covering the
    real fixture-`F1` byte sequence, the three pinned-code positive
    paths, the unpinned-code negative path (eleven sampled
    `H_filetype` numeric values outside `{2, 3, 5}`), the
    `from_wire ↔ wire_value` round-trip on each pinned variant, the
    TR.156 label match, the per-variant `bytes_per_sample` width,
    the `is_signed` sign convention, and the byte-order accessor's
    `None`-for-u8 convention. Plus 6 integration tests in
    `tests/filetype_pinned_accessor.rs` re-exercising the same
    contract through `parse_stream_header()` against (a) the real
    fixture-`F1` 11-byte prefix and (b) synthetic v2 headers
    stamped with each pinned numeric code plus a sweep of unpinned
    ones.
  - No wire-format change. The accessor is a typed window on top of
    the existing raw `filetype: u32` field, which remains the
    source of truth and is unchanged.

- **Round 238 clean-room rebuild.** `BLOCK_FN_BITSHIFT` housekeeping
  encoder (`spec/03` §3.7 + `spec/04` §3 + `spec/05` §1.4) — the
  per-stream left-shift command that applies `sample << bshift` to
  every subsequent decoded sample:
  - `write_bitshift_command(writer, bshift)` — emits a complete
    `BLOCK_FN_BITSHIFT` command as `uvar(FNSIZE = 2)` over value 6
    followed by `uvar(BITSHIFTSIZE = 2)` over the `bshift` payload.
    The encoder caps `bshift` at `BITSHIFT_MAX = 31` (the decoder
    side's `Error::BitshiftTooLarge` dual); over-cap values surface
    a new `EncodeError::BitshiftOutOfRange(b)` variant without
    emitting any partial bytes onto the writer.
  - `FN_BITSHIFT = 6` constant added to the encoder's public surface,
    matching the existing `FN_DIFF0..3`, `FN_QUIT`, `FN_QLPC`,
    `FN_VERBATIM`, `FN_ZERO` constants for the other commands.
  - Tests: 4 new in-crate unit tests covering (a) the
    `FN_BITSHIFT = 6` function-code numeric value, (b) the exact
    bit-length budget for the four anchor-fixture `bshift` values
    1 / 4 / 8 / 12 (totals 7 / 8 / 9 / 10 bits per `spec/02` §2.1
    length formula), (c) round-trip through `read_function_code` +
    `read_bitshift_payload` for `bshift ∈ {0, 1, 4, 7, 8, 12,
    BITSHIFT_MAX}` (covering the explicit no-op edge, the four
    anchor-fixture values, the `F2` cross-fixture corroborator
    `bshift = 7`, and the cap-edge), (d) `BITSHIFT_MAX + 1`
    over-cap rejection without partial writer state.

- **Round 18 clean-room rebuild.** `BLOCK_FN_ZERO` sentinel encoder
  (`spec/03` §3.9 + `spec/04` §6 + `spec/05` §2.4) — the constant-
  block command, extending the round-13..17 predictor-encoder push
  with the format's cheapest sample-producing command:
  - `write_zero_block(writer)` — emits a bare `BLOCK_FN_ZERO`
    command (function code 8) as `uvar(FNSIZE = 2)` over value 8;
    no further wire fields follow. Total encoded bit count is
    five bits — the cheapest sample-producing command in the
    format, packing four ZERO commands tightly into 20 bits.
  - The decoder side (`fill_zero_block` + the round-7 driver's
    `FunctionCode::Zero` arm) emits `bs` samples all equal to the
    channel's current running-mean estimate `μ_chan` per `spec/05`
    §2.4 — zero when `H_meanblocks = 0`, the running mean of the
    last `H_meanblocks` per-channel block-means otherwise. The
    block advances the channel cursor per `spec/03` §3.9.
  - Caller responsibility: emit ZERO only when the source block
    consists of exactly `bs` copies of the current `μ_chan` value
    at the producer end. The writer does not verify this — the
    `μ_chan` value is the higher-level sequencer's knowledge, not
    the writer primitive's. (`MeanEstimator::mu_chan` is the
    matching read accessor on the decoder side.)
  - `FN_ZERO = 8` constant added to the encoder's public surface,
    matching the existing `FN_DIFF0..3`, `FN_QUIT`, `FN_QLPC`,
    `FN_VERBATIM` constants for the other commands.
  - Tests: 6 new in-crate unit tests covering (a) the `FN_ZERO = 8`
    function-code numeric value, (b) the 5-bit emitted size and
    the `0b00100` bit pattern, (c) round-trip through
    `read_function_code` returning `FunctionCode::Zero`, (d) the
    mono / stereo / mixed-with-DIFF0 / three-consecutive shapes
    end-to-end through `decode_stream`. Plus 7 new integration
    tests in `tests/encoder_zero_pipeline.rs` covering single-
    block, multi-channel round-robin, many-consecutive-ZERO,
    ZERO-then-DIFF0 continuity, ZERO with VERBATIM envelope
    splice, three-channel ZERO + DIFF0 + ZERO interleave, and
    per-command 5-bit packing density verification.

### Changed

- **Round 203 in-place scrub.** Removed pre-existing enumerated denials
  in `src/lib.rs`, `src/driver.rs`, and `src/codec.rs` clean-room
  provenance prose. The wall-respected statement belongs in the agent
  final report, not in committed source comments. The remaining
  provenance prose keeps positive `spec/` attribution intact.

### Added

- **Round 17 clean-room rebuild.** `BLOCK_FN_QLPC` predictor encoder
  (`spec/03` §3.5 + `spec/02` §4.3 + §4.4 + `spec/05` §1 + §3.1) — the
  general quantised-LPC predictor encoder, closing the predictor
  encoder family on the encoder side (DIFF0..3 from rounds 13..16 plus
  QLPC here):
  - `write_qlpc_block(writer, energy_encoded, coefs, samples, carry)`
    — emits a full `BLOCK_FN_QLPC` command (function code 7 +
    `uvar(LPCQSIZE = 2)` per-block order + `order` quantised
    coefficients as `svar(LPCQUANT = 2)` + `uvar(ENERGYSIZE = 3)`
    energy + `bs × svar(energy + 1)` per-sample residuals) per
    `spec/03` §3.5. The encoder seeds the predictor history window
    from `carry.at(0)..carry.at(order − 1)` per `spec/05` §1.1,
    then slides the window to each just-emitted sample as the
    recurrence advances. The per-sample residual is
    `e_QLPC(t) = s(t) − Σᵢ coefs[i] · s(t − i − 1)` (TR.156 §3.2
    first equation). Coefficients are applied **without scaling**
    per `spec/03` §3.5. The decoder's `decode_qlpc_block`
    reconstructs `s(t) = Σᵢ aᵢ · s(t − i) + e_QLPC(t)`. Output
    round-trips losslessly through `decode_stream`.
  - QLPC is **mean-invariant** per `spec/03` §3.12 / `spec/05` §2
    introductory paragraph (the prediction is a linear function of
    past *samples* which already incorporate the channel mean), so
    `write_qlpc_block` takes no `mu_chan` parameter — matches the
    DIFF1 / DIFF2 / DIFF3 signature shape, distinct from DIFF0.
  - `qlpc_residuals(samples, coefs, carry)` — public helper
    computing the per-sample QLPC residual stream so callers
    driving `min_energy_for_qlpc` don't have to re-derive the
    per-sample scan at the call site.
  - `min_energy_for_qlpc(residuals)` — picks the smallest encoded
    energy `e ∈ 0..=7` such that every folded QLPC residual fits
    inside the `svar(e + 1)` mantissa with zero prefix-zero bits
    (the "natural" width per `spec/05` §3.1's "smallest sensible
    `n` is 1" floor). Returns `None` when no natural width fits
    the largest folded residual. Shares the same private scan as
    the four `min_energy_for_diff*` helpers.
  - `FN_QLPC = 7` (`spec/03` §3 + `spec/04` §5) and
    `MAX_QLPC_ORDER = 1024` (encoder safety cap matching the
    decoder's `MAX_LPC_ORDER`) — new public constants.
  - `EncodeError::LpcOrderTooLarge { order, carry_len }` and the
    `#[doc(hidden)]` `EncodeError::CoefficientOutOfRange(i64)` /
    `EncodeError::QlpcOrderCoefCountMismatch { order, coefs }`
    variants — new error states surfaced when `coefs.len()`
    exceeds the safety cap or the supplied `ChannelCarry`'s length.
  - 15 new in-module unit tests (`encoder::tests::*`) + 9 new
    integration tests (`tests/encoder_qlpc_pipeline.rs`) confirm:
    function-code constant matches `spec/03` §3; order-0 / 1 / 2 /
    3 residual scans against hand-computed expectations (order 0
    reduces to the input; `a₁ = 1` reduces to first-differences;
    `a₁ = 2, a₂ = -1` is the order-2 polynomial-difference
    predictor wire-encoded as QLPC; `a₁ = 3, a₂ = -3, a₃ = 1` is
    the order-3 polynomial-difference predictor on a pure-cubic
    ramp); minimum-energy selection (zero residuals → energy 0;
    cross-helper consistency vs. `min_energy_for_diff0`);
    energy-out-of-range / order-out-of-range / over-cap order
    rejection; bit-count correctness for the encoded fn + order +
    energy + residual layout; non-zero carry seed for cross-block
    continuity; full-stream round-trip via `decode_stream` at
    `H_maxlpcorder = 2` (mono single block) and `H_maxlpcorder = 1`
    (stereo two-block round-robin); a three-channel two-blocks-each
    round-robin stress with three distinct per-channel coefficient
    vectors; VERBATIM splice; silent block; carry continuity across
    two consecutive QLPC blocks on the same channel.

- **Round 16 clean-room rebuild.** `BLOCK_FN_DIFF3` predictor encoder
  (`spec/03` §3.4 + `spec/05` §1 + §3.1) — the order-3 polynomial-
  difference predictor encoder, closing the polynomial-difference
  predictor family on the encoder side:
  - `write_diff3_block(writer, energy_encoded, samples, carry)` —
    emits a full `BLOCK_FN_DIFF3` command (function code 3 +
    `uvar(ENERGYSIZE = 3)` energy field + `bs × svar(energy + 1)`
    per-sample third-differences) per `spec/03` §3.4. The encoder
    seeds `s(t − 1)` from `carry.at(0)`, `s(t − 2)` from
    `carry.at(1)`, and `s(t − 3)` from `carry.at(2)` (`spec/05`
    §1.1), then slides the rolling `s_m1` / `s_m2` / `s_m3` window
    to each just-emitted sample. The per-sample residual is
    `e₃(t) = s(t) − (3·s(t − 1) − 3·s(t − 2) + s(t − 3))`
    (TR.156 §3.2 eq. 6). The decoder's `decode_diff_block` with
    `PolyOrder::Order3` reconstructs
    `s(t) = 3·s(t − 1) − 3·s(t − 2) + s(t − 3) + e₃(t)` per
    `spec/03` §3.4. Output round-trips losslessly through
    `decode_stream`.
  - DIFF3 is mean-invariant per `spec/05` §2 introductory paragraph
    (the running mean cancels in the third-difference form), so the
    encoder takes no `mu_chan` parameter — matches the DIFF1 /
    DIFF2 signature shape, distinct from DIFF0.
  - `min_energy_for_diff3(residuals)` — picks the smallest encoded
    energy `e ∈ 0..=7` such that every folded third-difference fits
    inside the `svar(e + 1)` mantissa with zero prefix-zero bits
    (the "natural" width per `spec/05` §3.1's "smallest sensible
    `n` is 1" floor). Returns `None` when no natural width fits the
    largest folded residual.
  - `FN_DIFF3 = 3` — new wire-format numeric constant
    (`spec/04` §7).
  - 8 new in-module unit tests (`encoder::tests::*`) + 9 new
    integration tests (`tests/encoder_diff3_pipeline.rs`) confirm:
    minimum-energy selection across `e ∈ 0..=7`, function-code +
    bit-count correctness for the small encoder output, full
    round-trip through `decode_diff_block` with zero-seeded and
    non-zero-seeded `ChannelCarry` (three-sample window), mono
    single-block + stereo two-block round-robin round-trip via the
    full `decode_stream` driver, cross-block carry continuity for
    consecutive blocks on the same channel (load-bearing for
    `spec/05` §1.3's update rule — three-sample window seeding),
    VERBATIM-prefix splice decoded end-to-end, silent block
    (all-zero samples) at minimum energy, pure-quadratic collapse
    (third-differences `[seed₀, seed₁, seed₂, 0, 0, …]` selecting
    an explicit width for the seed jumps), ±127 max-natural
    third-difference edge, and a three-channel three-blocks-each
    round-robin stress.

- **Round 15 clean-room rebuild.** `BLOCK_FN_DIFF2` predictor encoder
  (`spec/03` §3.3 + `spec/05` §1 + §3.1) — the order-2 polynomial-
  difference predictor encoder, next step after round 14's `DIFF1`:
  - `write_diff2_block(writer, energy_encoded, samples, carry)` —
    emits a full `BLOCK_FN_DIFF2` command (function code 2 +
    `uvar(ENERGYSIZE = 3)` energy field + `bs × svar(energy + 1)`
    per-sample second-differences) per `spec/03` §3.3. The encoder
    seeds `s(t − 1)` from `carry.at(0)` and `s(t − 2)` from
    `carry.at(1)` (`spec/05` §1.1), then slides the rolling
    `s_m1` / `s_m2` window to each just-emitted sample. The
    per-sample residual is
    `e₂(t) = s(t) − (2·s(t − 1) − s(t − 2))`. The decoder's
    `decode_diff_block` with `PolyOrder::Order2` reconstructs
    `s(t) = 2·s(t − 1) − s(t − 2) + e₂(t)` per `spec/03` §3.3.
    Output round-trips losslessly through `decode_stream`.
  - DIFF2 is mean-invariant per `spec/05` §2 introductory paragraph
    (the running mean cancels in the second-difference), so the
    encoder takes no `mu_chan` parameter — matches the DIFF1
    signature shape, distinct from DIFF0.
  - `min_energy_for_diff2(residuals)` — picks the smallest encoded
    energy `e ∈ 0..=7` such that every folded second-difference
    fits inside the `svar(e + 1)` mantissa with zero prefix-zero
    bits, under the same scan rule as `min_energy_for_diff0` /
    `min_energy_for_diff1`. The three helpers share a private
    scan; the per-predictor entry points exist to make the
    call-site intent explicit. Returns `None` if the largest folded
    residual exceeds the natural 8-bit width.
  - `FN_DIFF2 = 2` — new wire-format constant the encoder emits.
  - 8 new in-module unit tests + 9 new integration tests
    (`tests/encoder_diff2_pipeline.rs`) covering: minimum-energy
    selection across the `e ∈ 0..=7` range, function-code emission
    + bit-count correctness, round-trip through `decode_diff_block`
    at all natural widths with zero-seeded and non-zero-seeded
    carry (two-sample window), multi-channel round-robin dispatch
    with per-channel carry update, cross-block carry continuity
    (the load-bearing two-sample seed test), VERBATIM-prefix splice
    with DIFF2 block, silent block (all-zero samples → all-zero
    residuals → energy 0), pure-ramp collapse (second-difference
    `[seed, 0, 0, …]`), `±127` max-natural second-difference edge,
    and three-channel three-blocks-each round-robin stress.

  Test counts: 188 → 199 in-module tests (+8 DIFF2 plus 3 inline
  helper unit tests on `diff2_residuals` and the carry-pair seeding),
  35 → 44 integration tests (+9 DIFF2 pipeline tests) — totals
  224 → 244 (+20 net). The `DIFF3` and `QLPC` predictor encoders +
  the per-block channel-round sequencer + the Rice-parameter
  optimal-selection of TR.156 §3.3 remain pending.

- **Round 14 clean-room rebuild.** `BLOCK_FN_DIFF1` predictor encoder
  (`spec/03` §3.2 + `spec/05` §1 + §3.1) — the order-1 polynomial-
  difference predictor encoder, next step after round 13's `DIFF0`:
  - `write_diff1_block(writer, energy_encoded, samples, carry)` —
    emits a full `BLOCK_FN_DIFF1` command (function code 1 +
    `uvar(ENERGYSIZE = 3)` energy field + `bs × svar(energy + 1)`
    per-sample first-differences) per `spec/03` §3.2. The encoder
    seeds `s(t − 1)` from `carry.at(0)` (`spec/05` §1.1: index 0 is
    the most-recent past sample, zero-initialised at stream start),
    then slides the rolling `s_m1` to each just-emitted sample.
    The decoder's `decode_diff_block` with `PolyOrder::Order1`
    reconstructs `s(t) = s(t − 1) + e₁(t)` per `spec/03` §3.2.
    Output round-trips losslessly through `decode_stream`.
  - DIFF1 is mean-invariant per `spec/05` §2 introductory paragraph
    (the running mean cancels in the difference), so the encoder
    takes no `mu_chan` parameter — distinct from `write_diff0_block`.
  - `min_energy_for_diff1(residuals)` — picks the smallest encoded
    energy `e ∈ 0..=7` such that every folded first-difference
    fits inside the `svar(e + 1)` mantissa with zero prefix-zero
    bits, under the same scan rule as `min_energy_for_diff0`. The
    two helpers share a private scan; the per-predictor entry
    points exist to make the call-site intent explicit. Returns
    `None` if the largest folded residual exceeds the natural
    8-bit width.
  - `FN_DIFF1 = 1` — new wire-format constant the encoder emits.
  - 8 new in-module unit tests + 9 new integration tests
    (`tests/encoder_diff1_pipeline.rs`) covering: minimum-energy
    selection across the `e ∈ 0..=7` range, function-code emission
    + bit-count correctness, round-trip through `decode_diff_block`
    at all natural widths with zero-seeded and non-zero-seeded
    carry, multi-channel round-robin dispatch with per-channel
    carry update, cross-block carry continuity (`spec/05` §1.3
    update rule), VERBATIM-prefix splice with DIFF1 block, silent
    block (all-zero samples → all-zero residuals → energy 0),
    constant-signal collapse (first-difference [seed, 0, 0, …]),
    ±127 max-natural first-difference edge, and three-channel
    three-blocks-each round-robin stress.

  Test counts: 177 → 188 in-module tests (+8 DIFF1 plus 3 inline
  helper unit tests on `diff1_residuals` and the new
  `ChannelCarry::update_after_block` interaction), 26 → 35
  integration tests (+9 DIFF1 pipeline tests) — totals 203 → 224
  (+20 net). The `DIFF2..3` and `QLPC` predictor encoders + the
  per-block channel-round sequencer + the Rice-parameter optimal-
  selection of TR.156 §3.3 remain pending.

- **Round 13 clean-room rebuild.** `BLOCK_FN_DIFF0` predictor encoder
  (`spec/03` §3.1 + `spec/05` §3.1) — the first predictor-side
  encoder, building on round 12's envelope surface:
  - `write_diff0_block(writer, energy_encoded, samples, mu_chan)` —
    emits a full `BLOCK_FN_DIFF0` command (function code 0 +
    `uvar(ENERGYSIZE = 3)` energy field + `bs × svar(energy + 1)`
    residuals) per `spec/03` §3.1. Residuals are computed
    encode-side as `e₀(t) = s(t) − μ_chan`; the decoder's
    `decode_diff_block` with `PolyOrder::Order0` reconstructs
    `s(t) = e₀(t) + μ_chan` per `spec/05` §2.3. Output round-trips
    losslessly through `decode_stream` and the streaming decoder.
  - `min_energy_for_diff0(residuals)` — picks the smallest encoded
    energy `e ∈ 0..=7` such that every folded residual fits inside
    the `svar(e + 1)` mantissa with zero prefix-zero bits — matching
    `spec/05` §3.1's "smallest sensible n is 1" floor. Returns
    `None` if the largest folded residual exceeds the natural
    8-bit width; callers may either accept a prefix-zero blow-up
    by passing `MAX_NATURAL_ENERGY = 7` explicitly or fall back to
    a wider non-natural width up to the decoder's
    `MAX_RESIDUAL_WIDTH = 30` cap.
  - `FN_DIFF0 = 0` / `MAX_NATURAL_ENERGY = 7` — new wire-format
    constants the encoder emits.
  - `EncodeError::EnergyOutOfRange` / `EncodeError::ResidualOutOfRange`
    / `EncodeError::BlockTooLong` — three new encoder-side error
    variants surfacing per-block parameter validation failures.
  - 9 new in-module unit tests + 6 new integration tests
    (`tests/encoder_diff0_pipeline.rs`) covering: minimum-energy
    selection across the `e ∈ 0..=7` range, function-code emission
    + bit-count correctness, round-trip through `decode_diff_block`
    at all natural widths, full envelope-encoder + DIFF0 splice
    decoding via the round-7 `decode_stream` driver, multi-block
    round-robin stereo round-trip, silent-block (all-zero) encoding,
    and encode/decode mean-directionality consistency.

  Test counts: 168 → 177 library tests (+9), 20 → 26 integration
  tests (+6) — totals 197 → 203 (+14). The `DIFF1..3` and `QLPC`
  predictor encoders + the per-block channel-round sequencer + the
  Rice-parameter optimal-selection of TR.156 §3.3 remain pending.

- **Round 12 clean-room rebuild.** Encoder-side envelope primitives
  + `BitWriter` foundation — first encoder-side surface since the
  2026-05-18 orphan rebuild (`spec/01` §1 + §3 + `spec/02` §1..§3 +
  `spec/03` §3.8 + §3.10 + `spec/04` §2 + §7 + `spec/05` §4):
  - `BitWriter` — MSB-first bit writer (encode-side counterpart of
    `BitReader`). Exposes `write_bit` / `write_bits` / `write_uvar`
    / `write_svar` / `write_ulong` (mirroring `spec/02` §2.1 / §2.2
    / §3) plus `pad_to_byte` for the post-QUIT zero-padding rule of
    `spec/05` §4 and `snapshot_bytes` / `into_bytes` finalisers.
    The roundtrip `read_*(write_*(v)) == v` is verified across a
    representative spread of widths and values, including TR.156's
    worked `uvar(2)` examples (0..16) from `spec/02` §2.1.
  - `natural_ulong_width(v)` — the encoder's minimum-width rule
    for the `ulong()` two-stage form of `spec/02` §3.
  - `encode_envelope_stream(header, verbatim_prefix)` — high-
    level envelope encoder that builds a syntactically-valid
    Shorten byte stream out of `(ShortenStreamHeader, &[u8])`:
    byte-aligned magic + version (`spec/01` §1) + the six-field
    parameter block (`spec/01` §3) + optional `BLOCK_FN_VERBATIM`
    (`spec/03` §3.10) + `BLOCK_FN_QUIT` (`spec/03` §3.8 /
    `spec/04` §2) + zero-pad to next byte boundary (`spec/05` §4).
    Output round-trips losslessly through `decode_stream`.
  - `write_byte_aligned_prefix` / `write_stream_header` /
    `write_parameter_block` / `write_verbatim_block` /
    `write_quit_command` — lower-level primitives the envelope
    driver composes, exposed for callers that need finer-grained
    control over command sequencing.
  - `FN_VERBATIM = 9` / `FN_QUIT = 4` / `ENCODER_VERSION = 2` —
    wire-format numeric constants the encoder emits.
  - `EncodeError` — encoder-side error enum
    (`UnsupportedVersion`, `VerbatimTooLong`).
  - 9 new integration tests in
    `tests/encoder_envelope_roundtrip.rs` exercise header roundtrip
    across the three pinned `H_filetype` codes, the F1 + F3 + F4
    verbatim-prefix shapes, the §9.4-candidate extreme-width
    parameter values, and equivalence between
    `encode_envelope_stream` and primitive-level composition.
  - 21 new unit tests in `src/bitwriter.rs` + `src/encoder.rs`
    cover bit-level primitives, TR.156 `uvar(2)` worked examples,
    signed/unsigned roundtrip across mantissa widths, header
    roundtrip via `parse_stream_header`, and version rejection.
  - Total test count: 189 (168 unit + 21 integration), up from 152
    in round 11.
- **Spec gap noted.** `docs/audio/shorten/spec/04-function-code-resolution.md`
  §2's narrative describes `BLOCK_FN_QUIT = 4` as the 5-bit
  `uvar(2)` pattern `00100`, but per `spec/02` §2.1's worked
  examples `00100` is the encoding of value 8 (`BLOCK_FN_ZERO`).
  The encoding of value 4 in `uvar(2)` is the 4-bit pattern
  `0100`. The decoder side (which maps numeric 4 → `Quit` in
  `block.rs`) is consistent with the new encoder; the gap is in
  §2's narrative.

- **Round 11 clean-room rebuild.** Streaming `oxideav_core::Decoder`
  trait wiring built on the same per-block dispatch as
  `StreamDecoder` (round 10) — closes the README "lacks" tail
  "*A streaming variant of the `oxideav_core::Decoder` trait wiring
  built on `StreamDecoder` (round 10's iterator is the pure-rust
  surface; the framework adaptor in `codec.rs` still buffers the
  full file before emitting one frame).*" (`spec/03` §2 + §3.6 +
  §3.7 + §3.8 + §3.10 + `spec/05` §1.4 + §2 + §6):
  - `ShortenStreamingDecoder` — `oxideav_core::Decoder`
    implementation that walks the per-block command loop
    incrementally and emits **one planar `AudioFrame` per full
    channel round** (one block per channel packed across all
    channels), rather than the whole-stream `ShortenDecoder`'s
    "buffer-the-whole-file, emit one frame" shape. Per `spec/05`
    §1.4 the per-channel carry stores pre-shift samples; the
    left-shift is applied on emission only. `BLOCK_FN_VERBATIM`
    payloads accumulate incrementally onto a
    `verbatim_prefix()` accessor without producing a frame (the
    envelope is part of the host-format wrapper, not the sample
    stream). `BLOCK_FN_BLOCKSIZE` and `BLOCK_FN_BITSHIFT` are
    absorbed silently per `spec/03` §3.6/§3.7; a mid-round
    BLOCKSIZE change is surfaced as `Error::invalid` because the
    planar frame shape requires every plane in a frame to have
    the same sample count.
  - `make_streaming_decoder(params)` — factory returning a
    boxed `Decoder` over the streaming adaptor.
  - `register_streaming_codecs(reg)` — installs the streaming
    decoder factory under the new codec id
    `"shorten-streaming"` (distinct from `"shorten"`, which the
    whole-stream wrapper continues to claim). A caller picks
    the emission shape explicitly by codec id.
  - `STREAMING_CODEC_ID_STR` — public string form of the
    streaming codec id.
  - `register(ctx)` — the framework entry now installs **both**
    factories into the runtime context's registry, so a
    `RuntimeContext` resolves either shape on demand.
  - Memory characteristic: `O(buffered_bytes + n_channels ×
    current_block_size)` plus the per-channel carries / mean
    estimators the iterator already needs — bounded by the
    header parameters and independent of stream length, in
    contrast to the whole-stream wrapper which buffers
    `O(stream_length)` of decoded samples before emitting.
  - 10 new unit tests in `src/codec.rs::tests` cover:
    one-block single-channel emission + EOF, two-channel
    two-round multi-plane frames, parity with the whole-stream
    driver on a complex DIFFn / VERBATIM / BITSHIFT / BLOCKSIZE
    / ZERO fixture, split-packet chop-anywhere streaming
    equivalence to whole-buffer delivery, verbatim-prefix
    incremental accumulation across two consecutive
    `BLOCK_FN_VERBATIM` commands, mid-round BLOCKSIZE rejection
    as `Error::invalid`, `register_streaming_codecs` isolation
    from `register_codecs`, `reset()` allowing a fresh
    redecoding after partial state, on-demand stop-at-frame-
    boundary pulls, and the framework `register(ctx)` entry
    installing both decoder shapes into the same registry.

- **Round 10 clean-room rebuild.** Block-by-block streaming decode
  iterator (`docs/audio/shorten/spec/03-block-and-predictor.md` §2 +
  §3.6/§3.7/§3.8/§3.10 + `spec/05-state-and-quirks.md` §1.4 + §2):
  - `StreamDecoder<'a>` — a `Iterator<Item = Result<DecodedBlock>>`
    over the per-block command stream. Parses the file header
    eagerly at construction (so `header()` / `current_block_size()`
    / `current_bitshift()` / `current_channel()` are observable
    before the first pull); subsequent `next_block` / `next` calls
    walk the round-robin command loop one block at a time, yielding
    each sample-producing block (`DIFFn` / `QLPC` / `ZERO`) or
    `VERBATIM` envelope payload as a `DecodedBlock` item. The
    housekeeping commands `BLOCK_FN_BLOCKSIZE` (§3.6) and
    `BLOCK_FN_BITSHIFT` (§3.7) are absorbed silently into the
    iterator's state and surface only through the running-state
    accessors. The iterator returns `None` after `BLOCK_FN_QUIT`
    (§3.8) and short-circuits to `None` after yielding any error
    once (standard `Iterator<Item = Result<_, _>>` convention).
  - `DecodedBlock { Samples { channel, samples } | Verbatim
    { bytes } }` — the per-block item shape, with `sample_count`
    / `is_samples` / `is_verbatim` convenience accessors.
  - `decode_stream_iter(bytes)` — convenience free function
    equivalent to `StreamDecoder::new(bytes)`.
  - Memory characteristic: the iterator retains only the per-channel
    carries (`spec/05` §1) + mean estimators (`spec/05` §2) + the
    `BitReader`'s small cache across pulls, so memory reaches
    `O(n_channels × max(3, H_maxlpcorder + H_meanblocks))`
    independent of stream length — the load-bearing improvement
    over the round-7 whole-stream driver `decode_stream`, which
    accumulates every decoded sample into `DecodedStream::channels`
    ahead of the caller.
  - 12 unit tests in `src/stream_iter.rs::tests` cover: single-block
    Quit termination, VERBATIM-then-samples sequencing, two-channel
    round-robin cursor rotation, BLOCKSIZE / BITSHIFT housekeeping
    absorption + observable state mutation, ZERO running-mean
    emission parity with the round-7 driver, truncated-stream
    error short-circuit + post-error exhaustion, zero-channel
    header rejection at construction, multi-block multi-channel
    equivalence to the whole-stream driver, header-accessor
    equivalence to `decode_stream`, cursor-advancement guard
    (housekeeping vs sample-producing), and `decode_stream_iter`
    free-function wrapper.
  - 2 integration tests in `tests/streaming_iterator_pipeline.rs`
    compose every command category (VERBATIM, BITSHIFT, two DIFF1
    + one DIFF0 block, BLOCKSIZE override, second VERBATIM, QUIT)
    into one fixture and assert (a) iterator-vs-driver per-channel
    sample equality + concatenated-verbatim equality + item-count
    correctness + observable-state mutation, and (b) on-demand
    decoding (pulling the first sample block doesn't decode the
    second block's bytes).
  - No new wire-format behaviour, no new reconstruction
    recurrences — a pure API re-shaping over the round-7 driver's
    orchestration loop.

- **Round 9 clean-room rebuild.** `SHNAMPSK`-tagged seek-table
  trailer detector landed against
  `docs/audio/shorten/spec/05-state-and-quirks.md` §5.1
  (trailer layout), §5.2 (decoder behaviour), §5.3 (fixture
  verification anchors `F1..F8` carry the trailer; `F9` /
  `Choppy.shn` do not):
  - `detect_shnampsk_trailer(bytes) -> Result<Option<ShnampskTrailer>>`
    — identifies the 12-byte tail layout (4-byte little-endian
    `len_u32` sidecar length + 8-byte `SHNAMPSK` ASCII signature)
    pinned by `spec/05` §5.1 and verifies the `SEEK` magic anchor at
    the computed `len(file) − len_u32` sidecar start. Returns
    `Ok(None)` when the signature is absent (matches fixture `F9` /
    `Choppy.shn`); returns `Ok(Some(t))` when both the signature and
    anchor check pass; surfaces `Error::MalformedShnampskTrailer`
    when the signature is present but the length field is below
    `MIN_SIDECAR_LEN`, above the implementation safety cap, or
    larger than the file itself, or when the bytes at the computed
    sidecar start do not begin with `SEEK`.
  - `split_off_shnampsk_trailer(bytes) -> Result<(&[u8],
    Option<&[u8]>)>` — convenience wrapper that returns
    `(shn_proper, sidecar_opt)` so callers can hand the
    SHN-stream-proper slice directly to `decode_stream` without
    computing the slice indices themselves.
  - `ShnampskTrailer { sidecar_start, sidecar_len }` — the parsed
    trailer record. `sidecar_start` is the byte offset within the
    original file at which the sidecar begins (the `S` of `SEEK`);
    `sidecar_len` equals `bytes.len() - sidecar_start` for a
    well-formed trailer.
  - Public constants: `SHNAMPSK_SIGNATURE` (the 8-byte
    `b"SHNAMPSK"`), `SEEK_MAGIC` (the 4-byte `b"SEEK"`),
    `TRAILER_TAIL_LEN = 12` (`spec/05` §5.1), `MIN_SIDECAR_LEN = 16`
    (sidecar must hold at least the `SEEK` magic and the
    `TRAILER_TAIL_LEN` tail), and `SIDECAR_LEN_CAP = 16 MiB`
    (implementation safety cap on the reported sidecar length to
    avoid mis-interpreting an opaque byte sequence whose tail
    coincidentally spells `SHNAMPSK`).
  - `Error::MalformedShnampskTrailer` — new variant surfacing the
    rejection cases above. Per `spec/05` §5.2 the decoder is free to
    either ignore or surface the trailer; this crate's detector
    surfaces it so callers that want the sidecar are told of any
    structural inconsistency rather than silently receiving "no
    trailer present".
  - The detector does not run `decode_stream`; the existing
    whole-stream driver of `spec/03` §3.8 already terminates at
    `BLOCK_FN_QUIT`'s zero-bit padding (`spec/05` §4) and so never
    reads into the trailer. Callers compose the two:
    `decode_stream(split_off_shnampsk_trailer(bytes)?.0)`.
  - New integration test (`tests/shnampsk_trailer_strip_then_decode.rs`)
    exercises the composition end-to-end: a synthetic single-channel
    `s16lh` stream with `VERBATIM → DIFF0 → QUIT` decodes identically
    with and without an appended well-formed `SHNAMPSK` trailer; the
    detector reports the correct `sidecar_start` / `sidecar_len` for
    the trailer-present case and `None` for the no-trailer case.
  - 18 unit tests plus 2 integration tests cover the present-trailer
    path, the absent-trailer path, every rejection case (too-small
    `len_u32`, too-large `len_u32` vs. file, over-cap `len_u32`,
    missing `SEEK` anchor at the computed offset), the constant
    invariants (signature ASCII bytes, magic ASCII bytes, layout
    arithmetic), `split_off` error propagation, and a coincidental
    `SHNAMPSK`-at-tail trap (signature present at tail, no `SEEK`
    anchor — surfaces as malformed rather than silently chopping the
    "trailer" off).
  - Total test count: 107 -> 127 (118 unit + 9 integration; +18 unit
    +1 integration test file with 2 tests relative to round 8). The
    README "lacks" tail's reference to publicly-distributed sidecar
    handling is closed; the remaining tail item is the DIFFn / QLPC
    encoder path.


## [0.0.2](https://github.com/OxideAV/oxideav-shorten/releases/tag/v0.0.2) - 2026-05-24

### Other

- round 7: full-stream decode driver (decode_stream)
- round 6 — Implementer: BLOCK_FN_BLOCKSIZE + BLOCK_FN_BITSHIFT
- round 5 — Implementer: BLOCK_FN_QLPC quantised-LPC predictor
- round 4 clean-room rebuild: running mean estimator + DIFF0 / ZERO mu_chan consumers
- round 3 clean-room rebuild: DIFFn predictors + Rice residuals + per-channel carry
- round 2 clean-room rebuild: svar reader + per-block command dispatch (verbatim + quit)
- round 1 clean-room rebuild: file-header parser
- orphan rebuild: clean-room scaffold post 2026-05-18 audit

### Added

- **Round 8 clean-room rebuild.** `oxideav_core::Decoder` trait
  wiring landed against `docs/audio/shorten/spec/05-state-and-quirks.md`
  §6 (file-type-code numeric mapping), `spec/03` §3.10 (verbatim
  prefix collection), and the public `oxideav-core` `Decoder` trait
  contract:
  - `ShortenDecoder` — packet-in / frame-out adaptor wrapping the
    round-7 `decode_stream` driver. `send_packet` appends incoming
    `Packet` bytes to an internal buffer and eagerly tries
    `decode_stream`; on `Truncated` (or a short-buffer `InvalidMagic`
    before the 5-byte prefix has arrived) the wrapper leaves the
    buffer in place so the caller can deliver more bytes; on success
    it queues exactly one `AudioFrame` and sets EOF.
  - `pack_decoded_stream_to_frame` (internal) — sample-format byte
    packing per `spec/05` §6: filetype `2`/`u8` packs each `i32`
    channel sample as one `u8` per plane byte; filetype `3`/`s16hl`
    packs as 16-bit signed big-endian; filetype `5`/`s16lh` packs as
    16-bit signed little-endian. The eight unpinned TR.156 labels
    (`ulaw`, `s8`, `s16`, `u16`, `s16x`, `u16x`, `u16hl`, `u16lh`)
    surface `oxideav_core::Error::Unsupported`.
  - `verbatim_prefix()` accessor on the concrete `ShortenDecoder`
    type exposes the host-format envelope bytes collected from
    `BLOCK_FN_VERBATIM` commands; the boxed-trait surface returns
    only `AudioFrame`.
  - `make_decoder(params) -> Result<Box<dyn Decoder>>` and
    `register_codecs(reg)` exposed at the crate root (gated on the
    `registry` feature, default-on); the crate's `register(ctx)`
    entry point now installs the decoder factory into the runtime
    context's codec registry under codec id `"shorten"`.
  - Public constants: `CODEC_ID_STR = "shorten"`, `FILETYPE_U8 = 2`,
    `FILETYPE_S16HL = 3`, `FILETYPE_S16LH = 5` — pinned by
    `spec/05` §6 against fixtures `F2` / `F3` / `F1` respectively.
  - Reset semantics: `Decoder::reset` drops the buffer, the queued
    frame, the verbatim prefix, and the `decoded`/`eof` flags so the
    next `send_packet` starts as if no prior packets had been
    processed. `Decoder::flush` makes a final decode attempt against
    the buffered bytes before setting EOF.
  - Split-packet streaming: a caller may deliver the `.shn` file in
    multiple slices across separate `send_packet` calls; the
    integration test `split_packet_streaming_assembles_to_one_frame`
    exercises this with a mid-stream split.
  - New integration test (`tests/decoder_trait_roundtrip.rs`,
    `registry`-feature-gated) decodes a synthetic two-channel
    `s16lh` stream through the registered factory resolved out of
    `CodecRegistry::first_decoder` and asserts the emitted plane
    bytes match the direct `decode_stream` output byte-for-byte
    after hand-packing each channel's `i32` samples to little-endian
    `i16`. A second integration test pins the registration surface
    independently (`register_codecs` + `has_decoder` +
    `first_decoder` path).
  - Total test count: 94 -> 107 (100 unit + 7 integration; +12 unit
    + 1 integration test file with 2 tests relative to round 7).
    The README "lacks" tail's `oxideav-core Decoder wiring` item is
    closed; the remaining tail item is the DIFFn / QLPC encoder
    path.

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
