# oxideav-shorten

A pure-Rust Shorten (`.shn`) lossless audio codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 254 (2026-06-08).** The crate was
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
of the integer-PCM decode path. Round 7 ties them together into the
**full-stream decode driver** ([`decode_stream`]). Round 8 wires the
driver into the framework's [`oxideav_core::Decoder`] trait via the
**[`ShortenDecoder`]** adaptor + **`register_codecs`** factory. Round 9
adds the **`SHNAMPSK`-tagged seek-table trailer detector**
([`detect_shnampsk_trailer`] / [`split_off_shnampsk_trailer`]) so
callers can separate a publicly-distributed `.shn` file's
SHN-stream-proper bytes from the non-standard sidecar appended by
Wayne Stielau's seek-table utility:

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
* Round 7 (`spec/03` §2 + §3.6/§3.7 + §3.8 + `spec/05` §1 + §1.4 + §2):
  * `decode_stream(bytes)` — the full-stream decode driver. Parses the
    header, seeks a fresh `BitReader` past the byte-aligned magic +
    version prefix and the variable-length parameter block (via the new
    `BitReader::skip_bits`), then runs the per-block command loop until
    `BLOCK_FN_QUIT`. It carries the round-robin **channel cursor**
    (`spec/03` §2; advanced modulo `H_channels` by the sample-producing
    commands, left unchanged by housekeeping commands), the running
    **sub-block size** (default `H_blocksize`, overridden by
    `BLOCK_FN_BLOCKSIZE`), the running **bit-shift** (set by
    `BLOCK_FN_BITSHIFT`), the per-channel **sample-history carries**,
    and the per-channel **running mean estimators** across the whole
    stream.
  * `DecodedStream` — the driver's output: the parsed `header`, the
    `verbatim` byte prefix (collected from `BLOCK_FN_VERBATIM` in
    encounter order), and `channels: Vec<Vec<i32>>` (one time-ordered
    sample vector per channel). Per `spec/05` §1.4 the per-channel
    carry stores the **pre-shift** sample form so the predictor
    recurrences stay bit-shift invariant; the driver applies the
    left-shift on **emission** only, guarding the `i64 -> i32` boundary
    with `SampleOverflow`.
  * `BitReader::skip_bits(n)` — unbounded MSB-first bit-skip (chunked
    past the 32-bit `read_bits` cap) used to position the driver's
    reader at the first per-block command.
  * `MAX_COMMANDS` — a generous command-count safety cap (~1 billion,
    vs. fixture `F2`'s 11,380-command stream) that bounds a malformed
    never-terminating stream.

* Round 8 (`spec/05` §6 + `spec/03` §3.10 + `oxideav-core`
  `Decoder` trait):
  * `ShortenDecoder` — packet-in / frame-out adaptor that buffers
    incoming `Packet` bytes, runs `decode_stream` once enough bytes
    are present, and emits one planar `AudioFrame` packed per the
    `H_filetype` byte-order rule of `spec/05` §6. Filetype `2`
    (`u8`) packs as `U8P` (one byte per sample); filetype `3`
    (`s16hl`) packs as `S16P` big-endian; filetype `5` (`s16lh`)
    packs as `S16P` little-endian. The eight unpinned TR.156 labels
    surface `Error::Unsupported`.
  * `verbatim_prefix()` accessor on the concrete `ShortenDecoder`
    type exposes the host-format envelope bytes collected from
    `BLOCK_FN_VERBATIM` commands.
  * `make_decoder` / `register_codecs` — the historical direct
    factory shape plus the `CodecRegistry` installer. The crate's
    `register(ctx)` entry point now wires the decoder factory into a
    `RuntimeContext`'s codec registry under codec id `"shorten"`.
  * Split-packet streaming: a caller may deliver the `.shn` file in
    arbitrary slices across multiple `send_packet` calls; the
    wrapper buffers until `decode_stream` produces a frame.

* Round 10 (`spec/03` §2 + §3.6 + §3.7 + §3.8 + §3.10 +
  `spec/05` §1.4 + §2):
  * `StreamDecoder<'a>` — an `Iterator<Item = Result<DecodedBlock>>`
    surface over the per-block command stream. Parses the file
    header eagerly at construction; subsequent `next_block` /
    `next` calls walk the round-robin command loop one block at a
    time, yielding each sample-producing block (`DIFFn` / `QLPC` /
    `ZERO`) or `VERBATIM` envelope payload as a [`DecodedBlock`]
    item. The housekeeping commands `BLOCK_FN_BLOCKSIZE`
    (`spec/03` §3.6) and `BLOCK_FN_BITSHIFT` (`spec/03` §3.7) are
    absorbed silently into iterator state; they surface only via
    the `current_block_size` / `current_bitshift` accessors.
    `BLOCK_FN_QUIT` (`spec/03` §3.8) returns `None`. Errors are
    yielded once then exhaust the iterator (standard
    `Iterator<Item = Result<_, _>>` short-circuit convention).
  * `DecodedBlock { Samples { channel, samples } | Verbatim
    { bytes } }` — the per-block item shape, with
    `sample_count`/`is_samples`/`is_verbatim` accessors. The
    `Samples::samples` vector carries the **emitted** PCM
    (left-shifted by the running bit-shift per `spec/05` §1.4);
    the per-channel carry buffers continue to store the pre-shift
    form internally so the predictor recurrences stay bit-shift
    invariant.
  * `decode_stream_iter(bytes)` — free-function form of
    `StreamDecoder::new`.
  * **Memory characteristic:** unlike `decode_stream`, which
    accumulates every decoded sample into
    `DecodedStream::channels: Vec<Vec<i32>>` ahead of the caller,
    the iterator retains only the per-channel carries (`spec/05`
    §1, `n_channels × max(3, H_maxlpcorder)` samples), the
    per-channel mean estimators (`spec/05` §2, `n_channels ×
    H_meanblocks` slots), plus the `BitReader`'s small internal
    cache across pulls — memory reaches
    `O(n_channels × max(3, H_maxlpcorder + H_meanblocks))`
    independent of stream length. This is the load-bearing
    improvement: a 10.6 MB lossless fixture (the size order of
    `F1` per `spec/05` §2.5) decoded through the iterator never
    materialises the full per-channel sample population at once.

* Round 11 (`spec/03` §2 + §3.6 + §3.7 + §3.8 + §3.10 +
  `spec/05` §1.4 + §2 + §6):
  * `ShortenStreamingDecoder` — streaming
    `oxideav_core::Decoder` adaptor built on the same per-block
    dispatch as round 10's `StreamDecoder`. Unlike round 8's
    whole-stream `ShortenDecoder` (which buffers the entire file
    before emitting a single `AudioFrame`), the streaming wrapper
    emits **one planar `AudioFrame` per full channel round** —
    one block per channel packed across all channels per `spec/05`
    §6's file-type table. `BLOCK_FN_VERBATIM` payloads accumulate
    incrementally onto `verbatim_prefix()` without producing a
    frame; `BLOCK_FN_BLOCKSIZE` and `BLOCK_FN_BITSHIFT` are
    absorbed silently per `spec/03` §3.6/§3.7. Mid-round
    BLOCKSIZE changes are rejected because the planar frame shape
    requires per-plane sample counts to match.
  * `make_streaming_decoder(params)` — factory returning a boxed
    `Decoder` over the streaming adaptor.
  * `register_streaming_codecs(reg)` — installs the streaming
    decoder factory under the new codec id `"shorten-streaming"`
    (kept distinct from `"shorten"`, which the whole-stream
    wrapper continues to claim). A caller picks the emission
    shape explicitly by codec id; `register(ctx)` now installs
    **both** factories into the runtime context's registry.
  * Memory characteristic: `O(buffered_bytes + n_channels ×
    current_block_size)` plus the per-channel carries / mean
    estimators the iterator already needs — bounded by the
    header parameters and independent of stream length, in
    contrast to `ShortenDecoder` which buffers `O(stream_length)`
    of decoded samples before emitting.

* Round 9 (`spec/05` §5.1 + §5.2 + §5.3):
  * `detect_shnampsk_trailer(bytes) -> Result<Option<ShnampskTrailer>>`
    — identifies the 12-byte trailer tail layout pinned by
    `spec/05` §5.1 (4-byte little-endian `len_u32` sidecar length +
    8-byte ASCII `SHNAMPSK` signature) and verifies the `SEEK`
    magic anchor at the computed sidecar-start offset
    `len(file) − len_u32`. Returns `None` when the signature is
    absent (matches fixture `F9` / `Choppy.shn` per §5.3);
    surfaces `Error::MalformedShnampskTrailer` when the signature
    is present but the `len_u32` field or `SEEK` anchor is
    inconsistent.
  * `split_off_shnampsk_trailer(bytes)` — convenience wrapper
    returning `(shn_proper, sidecar_opt)` so callers can hand the
    SHN-stream-proper slice directly to `decode_stream` per
    `spec/05` §5.2 (the wire format itself terminates at
    `BLOCK_FN_QUIT`'s zero-bit padding; bytes after that are out
    of scope and the decoder may ignore them).
  * Public constants `SHNAMPSK_SIGNATURE` (`b"SHNAMPSK"`),
    `SEEK_MAGIC` (`b"SEEK"`), `TRAILER_TAIL_LEN = 12`,
    `MIN_SIDECAR_LEN = 16`, and `SIDECAR_LEN_CAP = 16 MiB`
    (implementation safety cap; the `spec/05` §5 narrative does
    not pin a numeric cap).

* Round 12 (`spec/01` §1 + §3 + `spec/02` §1..§3 + `spec/03` §3.8 +
  §3.10 + `spec/04` §2 + §7 + `spec/05` §4):
  * `BitWriter` — MSB-first encode-side counterpart of `BitReader`,
    the bit-level primitive every encoder routine sits on. Exposes
    `write_bit` / `write_bits` / `write_uvar` / `write_svar` /
    `write_ulong` (mirroring the read-side methods of `spec/02`
    §2.1 / §2.2 / §3) plus `pad_to_byte` for the post-`BLOCK_FN_QUIT`
    zero-padding rule of `spec/05` §4 and `snapshot_bytes` /
    `into_bytes` finalisers. The roundtrip
    `read_*(write_*(v)) == v` is verified across a representative
    spread of widths and values, including TR.156's worked
    `uvar(2)` examples (0..16 from `spec/02` §2.1).
  * `natural_ulong_width(v)` — the encoder's minimum-width
    rule for the `ulong()` two-stage form of `spec/02` §3,
    returning the smallest `width` such that `v < 2^width` (`0`
    for `v == 0`).
  * `encode_envelope_stream(header, verbatim_prefix)` — high-level
    encoder driver that builds a syntactically-valid Shorten byte
    stream out of a `(ShortenStreamHeader, &[u8])` pair: byte-
    aligned magic + version (`spec/01` §1) + the six-field
    parameter block (`spec/01` §3) + optional `BLOCK_FN_VERBATIM`
    command (`spec/03` §3.10) + `BLOCK_FN_QUIT` (`spec/03` §3.8 /
    `spec/04` §2) + zero-pad to the next byte boundary (`spec/05`
    §4). The output round-trips losslessly through
    [`decode_stream`]: the recovered header equals the encoded
    header, the verbatim prefix equals the encoded prefix, and the
    per-channel sample vectors are empty (no predictor commands
    were emitted).
  * `write_byte_aligned_prefix` / `write_stream_header` /
    `write_parameter_block` / `write_verbatim_block` /
    `write_quit_command` — the lower-level primitives the high-
    level driver composes, exposed so callers can build streams
    with custom per-block command sequences before the predictor
    encoder lands.
  * `FN_VERBATIM = 9` / `FN_QUIT = 4` / `ENCODER_VERSION = 2` —
    the wire-format numeric constants the encoder emits.
  * `EncodeError` — encoder-side error enum (`UnsupportedVersion`
    for a header version outside `{1, 2, 3}` per `spec/00`;
    `VerbatimTooLong` for a payload exceeding
    `VERBATIM_MAX_LEN` per `spec/02` §4.5).
  * **Scope.** Round 12 stops at the envelope. Round 13 (below)
    adds `BLOCK_FN_DIFF0` — the simplest predictor encoder. The
    other predictor + housekeeping encoder paths (`BLOCK_FN_DIFF1..3`
    / `BLOCK_FN_QLPC` / `BLOCK_FN_BLOCKSIZE` / `BLOCK_FN_BITSHIFT` /
    `BLOCK_FN_ZERO`) still belong to later rounds —
    `BLOCK_FN_DIFFn` for `n > 0` needs the per-channel sample-history
    carry across blocks and the higher orders' recurrence-form
    residual computation; `BLOCK_FN_QLPC` needs the coefficient
    quantisation of `spec/03` §3.5; and a full per-block channel-
    round sequencer with the Rice-`n` optimisation TR.156 §3.3
    derives sits above all of them.

- **Round 13** — the `BLOCK_FN_DIFF0` predictor encoder
  (`spec/03` §3.1 + `spec/05` §3.1):
  * `write_diff0_block(writer, energy_encoded, samples, mu_chan)`
    — emits the full `<fn=0> <energy> <residual>×bs` command,
    computing residuals encode-side as `e₀(t) = s(t) − μ_chan`.
    The decoder's `decode_diff_block` with `PolyOrder::Order0`
    reconstructs `s(t) = e₀(t) + μ_chan` per `spec/05` §2.3.
  * `min_energy_for_diff0(residuals)` — picks the smallest
    encoded energy `e ∈ 0..=7` such that every folded residual
    fits inside the `svar(e + 1)` mantissa with zero prefix-zero
    bits (the "natural" width per `spec/05` §3.1's "smallest
    sensible n is 1" floor). Returns `None` when no natural
    width fits the largest folded residual.
  * `FN_DIFF0 = 0` / `MAX_NATURAL_ENERGY = 7` — new wire-format
    numeric constants.
  * 6 new integration tests (`tests/encoder_diff0_pipeline.rs`)
    confirm: mono single-block round-trip, stereo round-robin
    multi-block round-trip, DIFF0 + envelope-VERBATIM splice
    decoded through `decode_stream`, all-zero silent block at
    minimum energy, ±100 residuals at maximum natural energy,
    and encode/decode mean-directionality consistency.
  * **Scope.** Round 13 is `DIFF0` only — the simplest of the
    four polynomial-difference predictors, with no past-sample
    carry on the encoder side. `BLOCK_FN_DIFF1..3` need the
    recurrence-form residual computation off the per-channel
    carry; `BLOCK_FN_QLPC` needs coefficient quantisation. The
    Rice-`n` selection helper is a natural-width-only heuristic;
    the statistical optimum of TR.156 §3.3 (which minimises
    total encoded bit count, not just prefix bits) is a future
    refinement that doesn't change the wire format.

- **Round 14** — the `BLOCK_FN_DIFF1` predictor encoder
  (`spec/03` §3.2 + `spec/05` §1 + §3.1):
  * `write_diff1_block(writer, energy_encoded, samples, carry)`
    — emits the full `<fn=1> <energy> <residual>×bs` command,
    computing the per-sample first-differences
    `e₁(t) = s(t) − s(t − 1)`. The initial `s(t − 1)` comes
    from `carry.at(0)` per `spec/05` §1.1 (index 0 is the
    most-recent past sample, zero-initialised at stream start);
    the rolling `s_m1` slides to each just-emitted sample as
    the recurrence advances. The decoder's `decode_diff_block`
    with `PolyOrder::Order1` reconstructs
    `s(t) = s(t − 1) + e₁(t)` per `spec/03` §3.2.
  * DIFF1 is mean-invariant per `spec/05` §2 introductory
    paragraph (the channel running mean cancels in the
    difference form), so the encoder takes no `mu_chan`
    parameter — a deliberate signature divergence from
    `write_diff0_block`.
  * `min_energy_for_diff1(residuals)` — picks the smallest
    encoded energy `e ∈ 0..=7` under the same svar-prefix-zero
    rule as `min_energy_for_diff0` (sharing a private scan
    helper); the per-predictor entry points exist to make the
    call-site intent explicit and to document which predictor
    definition the caller has applied to the input samples.
  * `FN_DIFF1 = 1` — new wire-format numeric constant.
  * 8 new in-module unit tests + 9 new integration tests
    (`tests/encoder_diff1_pipeline.rs`) confirm: minimum-energy
    selection across `e ∈ 0..=7`, function-code + bit-count
    correctness for the small encoder output, full round-trip
    through `decode_diff_block` with zero-seeded and non-zero-
    seeded `ChannelCarry`, mono single-block + stereo two-
    block round-robin round-trip via the full
    `decode_stream` driver, cross-block carry continuity for
    consecutive blocks on the same channel (load-bearing for
    `spec/05` §1.3's update rule), VERBATIM-prefix splice
    decoded end-to-end, silent block (all-zero samples) at
    minimum energy, constant-signal collapse (first-difference
    `[seed, 0, 0, …]`) selecting an explicit width for the seed
    jump, ±127 max-natural first-difference edge, and a
    three-channel three-blocks-each round-robin stress.
  * **Scope.** Round 14 adds `DIFF1`. `BLOCK_FN_DIFF2..3` need
    the order-2 / order-3 recurrence-form residual computation
    that reads two and three past samples from the carry; the
    Rice-`n` selection is still a natural-width-only heuristic
    and the statistical optimum of TR.156 §3.3 remains pending.

- **Round 15** — the `BLOCK_FN_DIFF2` predictor encoder
  (`spec/03` §3.3 + `spec/05` §1 + §3.1):
  * `write_diff2_block(writer, energy_encoded, samples, carry)`
    — emits the full `<fn=2> <energy> <residual>×bs` command,
    computing the per-sample second-differences
    `e₂(t) = s(t) − (2·s(t − 1) − s(t − 2))`. The initial
    `s(t − 1)` and `s(t − 2)` come from `carry.at(0)` and
    `carry.at(1)` per `spec/05` §1.1 (both zero-initialised at
    stream start); the rolling `s_m1` / `s_m2` window slides
    to each just-emitted sample as the recurrence advances.
    The decoder's `decode_diff_block` with `PolyOrder::Order2`
    reconstructs `s(t) = 2·s(t − 1) − s(t − 2) + e₂(t)` per
    `spec/03` §3.3.
  * DIFF2 is mean-invariant per `spec/05` §2 introductory
    paragraph (the channel running mean cancels in the
    second-difference form), so the encoder takes no `mu_chan`
    parameter — matches the DIFF1 signature shape.
  * `min_energy_for_diff2(residuals)` — picks the smallest
    encoded energy `e ∈ 0..=7` under the same svar-prefix-zero
    rule as `min_energy_for_diff0` / `min_energy_for_diff1`
    (sharing the same private scan helper); the per-predictor
    entry points exist to make the call-site intent explicit
    and to document which predictor definition the caller has
    applied to the input samples.
  * `FN_DIFF2 = 2` — new wire-format numeric constant.
  * 8 new in-module unit tests + 9 new integration tests
    (`tests/encoder_diff2_pipeline.rs`) confirm: minimum-energy
    selection across `e ∈ 0..=7`, function-code + bit-count
    correctness for the small encoder output, full round-trip
    through `decode_diff_block` with zero-seeded and non-zero-
    seeded `ChannelCarry` (two-sample window), mono single-block
    + stereo two-block round-robin round-trip via the full
    `decode_stream` driver, cross-block carry continuity for
    consecutive blocks on the same channel (load-bearing for
    `spec/05` §1.3's update rule — two-sample window seeding),
    VERBATIM-prefix splice decoded end-to-end, silent block
    (all-zero samples) at minimum energy, pure-ramp collapse
    (second-differences `[seed, 0, 0, …]` selecting an explicit
    width for the seed jump), ±127 max-natural second-difference
    edge, and a three-channel three-blocks-each round-robin
    stress.
  * **Scope.** Round 15 adds `DIFF2`. `BLOCK_FN_DIFF3` still
    needs the order-3 recurrence-form residual computation that
    reads three past samples from the carry; the Rice-`n`
    selection is still a natural-width-only heuristic and the
    statistical optimum of TR.156 §3.3 remains pending.

- **Round 16** — the `BLOCK_FN_DIFF3` predictor encoder
  (`spec/03` §3.4 + `spec/05` §1 + §3.1):
  * `write_diff3_block(writer, energy_encoded, samples, carry)`
    — emits the full `<fn=3> <energy> <residual>×bs` command,
    computing the per-sample third-differences
    `e₃(t) = s(t) − (3·s(t − 1) − 3·s(t − 2) + s(t − 3))`
    (TR.156 §3.2 eq. 6). The initial `s(t − 1)`, `s(t − 2)`,
    and `s(t − 3)` come from `carry.at(0)`, `carry.at(1)`, and
    `carry.at(2)` per `spec/05` §1.1 (all zero-initialised at
    stream start); the rolling `s_m1` / `s_m2` / `s_m3` window
    slides to each just-emitted sample as the recurrence
    advances. The decoder's `decode_diff_block` with
    `PolyOrder::Order3` reconstructs
    `s(t) = 3·s(t − 1) − 3·s(t − 2) + s(t − 3) + e₃(t)` per
    `spec/03` §3.4.
  * DIFF3 is mean-invariant per `spec/05` §2 introductory
    paragraph (the running mean cancels in the third-difference
    form), so the encoder takes no `mu_chan` parameter —
    matches the DIFF1 / DIFF2 signature shape.
  * `min_energy_for_diff3(residuals)` — picks the smallest
    encoded energy `e ∈ 0..=7` under the same svar-prefix-zero
    rule as `min_energy_for_diff0` / `min_energy_for_diff1` /
    `min_energy_for_diff2` (sharing the same private scan
    helper); the per-predictor entry points exist to make the
    call-site intent explicit and to document which predictor
    definition the caller has applied to the input samples.
  * `FN_DIFF3 = 3` — new wire-format numeric constant.
  * 8 new in-module unit tests + 9 new integration tests
    (`tests/encoder_diff3_pipeline.rs`) confirm: minimum-energy
    selection across `e ∈ 0..=7`, function-code + bit-count
    correctness for the small encoder output, full round-trip
    through `decode_diff_block` with zero-seeded and non-zero-
    seeded `ChannelCarry` (three-sample window), mono single-block
    + stereo two-block round-robin round-trip via the full
    `decode_stream` driver, cross-block carry continuity for
    consecutive blocks on the same channel (load-bearing for
    `spec/05` §1.3's update rule — three-sample window seeding),
    VERBATIM-prefix splice decoded end-to-end, silent block
    (all-zero samples) at minimum energy, pure-quadratic collapse
    (third-differences `[seed₀, seed₁, seed₂, 0, 0, …]` selecting
    an explicit width for the seed jumps), ±127 max-natural
    third-difference edge, and a three-channel three-blocks-each
    round-robin stress.
  * **Scope.** Round 16 closes the polynomial-difference predictor
    family (DIFF0..3) on the encoder side. `BLOCK_FN_QLPC`'s
    encoder is added by round 17 (below); the Rice-`n` selection
    is still a natural-width-only heuristic and the statistical
    optimum of TR.156 §3.3 remains pending.
  * **Spec gap (docs/audio/shorten/spec/04 §2):** the §2 narrative
    incorrectly describes `BLOCK_FN_QUIT = 4`'s encoding as the
    5-bit `uvar(2)` pattern `00100`, but per `spec/02` §2.1's
    worked examples `00100` is the encoding of value 8
    (`BLOCK_FN_ZERO`), not 4. The encoding of value 4 in `uvar(2)`
    is the 4-bit pattern `0100` (one leading zero + terminator +
    2-bit mantissa `00`; `k = 1` per `value = (k << n) + m`). The
    decoder side (which maps numeric 4 → `Quit` in `block.rs`) is
    consistent with the new encoder; the spec gap is on the
    narrative side of §2. Same wording slip likely on `spec/04`
    §2's F9 / F1 trace-position arithmetic; the F9 "last byte
    0x20 = 0010 0000" is consistent with a 4-bit QUIT at bit
    position 4 plus 4 zero padding bits OR a 5-bit ZERO pattern
    starting at the byte boundary (the parse hinges on whether
    the preceding command terminated at bit position 0 or bit
    position 4 of the last byte; the §2 narrative conflates the
    two cases).

- **Round 17** — the `BLOCK_FN_QLPC` predictor encoder
  (`spec/03` §3.5 + `spec/02` §4.3 + §4.4 + `spec/05` §1 + §3.1):
  * `write_qlpc_block(writer, energy_encoded, coefs, samples, carry)`
    — emits a full `<fn=7> <order> <coef>×order <energy>
    <residual>×bs` command, computing per-sample residuals
    `e_QLPC(t) = s(t) − Σᵢ coefs[i] · s(t − i − 1)` (TR.156 §3.2
    first equation). The first `order` past samples come from the
    per-channel sample-history carry (`carry.at(0) = s(t − 1)`,
    …, `carry.at(order − 1) = s(t − order)` per `spec/05` §1.1);
    the rolling history window slides to each just-emitted sample
    as the recurrence advances. Coefficients are applied
    **without scaling** per `spec/03` §3.5 — `coef[i] = aᵢ₊₁` is
    paired with `s(t − i − 1)` in the history window. The
    decoder's `decode_qlpc_block` reconstructs
    `s(t) = Σᵢ aᵢ · s(t − i) + e_QLPC(t)` per `spec/03` §3.5.
    Output round-trips losslessly through `decode_stream`.
  * QLPC is **mean-invariant** per `spec/03` §3.12 / `spec/05` §2
    introductory paragraph (the prediction is a linear function of
    past *samples* which already carry the channel mean), so
    `write_qlpc_block` takes no `mu_chan` parameter.
  * `qlpc_residuals(samples, coefs, carry)` — public helper that
    computes the per-sample residual stream a caller passes to
    `min_energy_for_qlpc` (and would otherwise have to re-derive
    at the call site).
  * `min_energy_for_qlpc(residuals)` — picks the smallest encoded
    energy `e ∈ 0..=7` such that every folded QLPC residual fits
    inside the `svar(e + 1)` mantissa with zero prefix-zero bits
    (the "natural" width per `spec/05` §3.1's "smallest sensible
    `n` is 1" floor). Shares the same private scan as the four
    `min_energy_for_diff*` helpers.
  * `FN_QLPC = 7` (`spec/03` §3 + `spec/04` §5) and
    `MAX_QLPC_ORDER = 1024` (encoder safety cap matching the
    decoder's `MAX_LPC_ORDER`) — new public constants.
  * `EncodeError::LpcOrderTooLarge { order, carry_len }` — new
    variant surfaced when `coefs.len()` exceeds either the safety
    cap or the supplied `ChannelCarry`'s length (the encoder
    cannot seed `order` past samples from a shorter carry).
  * 15 new in-module unit tests + 9 new integration tests
    (`tests/encoder_qlpc_pipeline.rs`) confirm: function-code
    constant matches `spec/03` §3; order-0 residual scan equals
    the input; order-1 with `a₁ = 1` and zero carry reduces to
    the first-difference stream; order-2 / order-3 residual scans
    against hand-computed expectations; over-cap order rejection;
    minimum-energy selection (zero residuals → energy 0; cross-
    helper consistency vs. `min_energy_for_diff0`);
    energy-out-of-range / order-out-of-range rejections; bit-
    count correctness for the encoded fn + order + energy +
    residual layout; round-trips at orders 0 / 1 / 2 / 3 (the
    `a₁ = 1, a₂ = -1` configuration is the order-2 polynomial-
    difference predictor encoded via QLPC; `a₁ = 3, a₂ = -3,
    a₃ = 1` is the order-3 polynomial-difference predictor on a
    pure-cubic ramp); non-zero carry seed for cross-block
    continuity; full-stream round-trip via `decode_stream` at
    `H_maxlpcorder = 2` (mono) and `H_maxlpcorder = 1` (stereo
    round-robin); a three-channel two-blocks-each round-robin
    stress with three different per-channel coefficient vectors;
    VERBATIM splice; silent block; carry continuity across two
    consecutive QLPC blocks on the same channel.
  * **Scope.** Round 17 closes the predictor encoder family on
    the encoder side (DIFF0..3 + QLPC). The natural-width
    `min_energy_for_qlpc` helper is **not** the optimal Rice
    parameter of TR.156 §3.3 (which minimises total encoded bit
    count); a future round can layer the statistical optimum on
    top without changing the wire format. The encoder-side
    coefficient quantisation (the encoder takes the caller's
    quantised coefficients as `&[i64]`; deriving the optimal
    coefficients from input samples is a separate, higher-layer
    concern per TR.156 §3.2's quantisation narrative). The
    per-block channel-round sequencer that picks
    `BLOCK_FN_DIFFn` / `BLOCK_FN_QLPC` from a per-block
    statistical objective likewise remains pending.

- **Round 18** — the `BLOCK_FN_ZERO` sentinel encoder
  (`spec/03` §3.9 + `spec/04` §6 + `spec/05` §2.4):
  * `write_zero_block(writer)` — emits the bare constant-block
    command as `uvar(FNSIZE = 2)` over `FN_ZERO = 8`. No further
    wire fields follow the function code; the command is a single
    5-bit token in the bit stream (prefix `00` + terminator `1` +
    mantissa `00`). Four ZERO commands pack tightly into 20 bits;
    five ZERO commands plus a single `BLOCK_FN_QUIT` plus its
    byte-alignment padding fit inside four bytes total. This is
    the cheapest sample-producing command in the format.
  * Decode semantics. The decoder side (`fill_zero_block` + the
    round-7 driver's `FunctionCode::Zero` arm) emits `bs` samples
    all equal to the channel's current running-mean estimate
    `μ_chan` per `spec/05` §2.4 — zero when `H_meanblocks = 0`,
    the running mean of the last `H_meanblocks` per-channel
    block-means otherwise. The block advances the channel cursor
    per `spec/03` §3.9, so a multi-channel stream's per-channel
    state advances on every ZERO.
  * Caller responsibility. The writer does not verify that the
    source block consists of `bs` copies of the current `μ_chan`
    at the producer end — the `μ_chan` value is the higher-level
    sequencer's knowledge, not the writer primitive's. An over-
    eager ZERO would produce wrong samples on decode (`bs`
    `μ_chan` values where the producer intended something else).
  * `FN_ZERO = 8` (`spec/03` §3 + `spec/04` §6) — new public
    encoder constant, matching the existing per-command numeric
    `FN_DIFF0..3 / FN_QUIT / FN_QLPC / FN_VERBATIM`.
  * 6 new in-module unit tests + 7 new integration tests
    (`tests/encoder_zero_pipeline.rs`) confirm: function-code
    constant matches `spec/04` §6; emitted bit pattern is `0b00100`
    over 5 bits; round-trip through `read_function_code` returns
    `FunctionCode::Zero`; mono / stereo / three-channel
    interleaved-with-DIFF0 round-trips through `decode_stream`;
    many-consecutive-ZERO mass test (16 back-to-back ZERO blocks
    decoding to 128 zeros via `bs = 8`); ZERO with VERBATIM
    envelope splice; ZERO-then-DIFF0 sample-history-carry
    continuity (the DIFF0 block sees `μ_chan = 0` after a ZERO
    block on the same channel under `H_meanblocks = 0`); per-
    command 5-bit packing-density verification using
    `BitWriter::bits_written()` deltas.
  * **Scope.** Round 18 lands the constant-block primitive. The
    remaining unwritten encoder branches are the two housekeeping
    commands `BLOCK_FN_BITSHIFT` (`spec/03` §3.7) and
    `BLOCK_FN_BLOCKSIZE` (`spec/03` §3.6) — both per-stream state
    mutators with no payload beyond a single `uvar` field each —
    and the higher-layer per-block channel-round sequencer that
    picks the cheapest of DIFF0..3 / QLPC / ZERO per block under a
    statistical objective.

- **Round 238** — the `BLOCK_FN_BITSHIFT` housekeeping encoder
  (`spec/03` §3.7 + `spec/04` §3 + `spec/05` §1.4):
  * `write_bitshift_command(writer, bshift)` — emits a complete
    `BLOCK_FN_BITSHIFT` command as `uvar(FNSIZE = 2)` over
    `FN_BITSHIFT = 6` followed by `uvar(BITSHIFTSIZE = 2)` over the
    `bshift` payload. The encoder caps `bshift` at `BITSHIFT_MAX
    = 31` (the decoder side's `Error::BitshiftTooLarge` dual);
    over-cap values surface a new
    `EncodeError::BitshiftOutOfRange(b)` without emitting any
    partial bytes onto the writer.
  * Bit-budget. Per `spec/02` §2.1 the function-code prefix is a
    4-bit pattern (`uvar(FNSIZE = 2)` over value 6 is `0110`); the
    payload `uvar(BITSHIFTSIZE = 2)` over `bshift` adds
    `⌊bshift / 4⌋ + 1 + 2` bits. The four anchor-fixture `bshift`
    values 1 / 4 / 8 / 12 (the `F5..F8` `-q N` invocations pinned by
    `spec/04` §3.1 test T10) pack to 7 / 8 / 9 / 10 bits total
    respectively; the `bshift = 0` no-op form is 7 bits.
  * Decode semantics. The decoder applies `sample << bshift` to
    every subsequent decoded sample emitted to the output stream
    (`spec/03` §3.7 + `spec/05` §1.4); the reconstruction
    recurrences and the channel-cursor / running-mean machinery
    remain in the pre-shift sample domain.
  * `FN_BITSHIFT = 6` (`spec/03` §3 + `spec/04` §3) — new public
    encoder constant, matching the existing per-command numeric
    `FN_DIFF0..3 / FN_QUIT / FN_QLPC / FN_VERBATIM / FN_ZERO`.
  * 4 new in-module unit tests confirm: function-code constant
    matches `spec/04` §3; emitted bit counts match the `spec/02`
    §2.1 length formula for the four anchor-fixture `bshift`
    values; round-trip through `read_function_code` +
    `read_bitshift_payload` recovers the original `bshift` for
    `{0, 1, 4, 7, 8, 12, BITSHIFT_MAX}` (the explicit no-op edge,
    the four `F5..F8` anchor values, the `F2` 8-bit cross-fixture
    corroborator `bshift = 7`, and the cap-edge); `BITSHIFT_MAX
    + 1` over-cap rejection produces `EncodeError::BitshiftOutOfRange`
    without partial writer state.
  * **Scope.** Round 238 lands the `BLOCK_FN_BITSHIFT` writer
    primitive only. The remaining unwritten encoder branch is the
    `BLOCK_FN_BLOCKSIZE` housekeeping writer (`spec/03` §3.6 +
    `spec/04` §4) — a per-stream block-size override carrying a
    single `ulong()` payload — plus the higher-layer per-block
    channel-round sequencer.

- **Round 241** — typed `H_filetype` accessor surfacing the three
  numeric codes `spec/05` §6 pins behaviourally:
  * New `Filetype` enum with variants `U8` (wire value 2 / TR.156
    label `u8` / fixture `F2`), `S16HL` (wire value 3 / `s16hl` /
    fixture `F3`), and `S16LH` (wire value 5 / `s16lh` / fixture
    `F1`). Marked `#[non_exhaustive]` so the remaining eight TR.156
    labels (`ulaw`, `s8`, `s16`, `u16`, `s16x`, `u16x`, `u16hl`,
    `u16lh`) can be added later — `spec/05` §8 candidate #3 — without
    breaking callers.
  * `ShortenStreamHeader::filetype_pinned() -> Option<Filetype>` —
    instance method returning `Some(variant)` when the raw
    `H_filetype: u32` field falls in the `{2, 3, 5}` pinned-by-fixture
    set and `None` for every other numeric value. The accessor never
    guesses a label for an unpinned code; the raw `filetype: u32`
    field stays the source of truth.
  * Per-variant accessors: `wire_value`, `label` (TR.156 textual
    name `u8` / `s16hl` / `s16lh`), `bytes_per_sample` (1 / 2 / 2),
    `is_signed` (false / true / true), and `is_little_endian`
    (`None` for `U8`, `Some(false)` for `S16HL`, `Some(true)` for
    `S16LH`). Plus the round-trip `Filetype::from_wire(u32) ->
    Option<Self>` constructor.
  * No wire-format change. The accessor is a typed window on top of
    the existing raw `filetype: u32` field; the bit-level header
    parse, the trait-side PCM-plane packing path, and the registered
    `FILETYPE_U8 / FILETYPE_S16HL / FILETYPE_S16LH` constants are
    unchanged.
  * 8 new in-module unit tests in `header.rs` (real fixture-`F1`
    byte sequence pinning the typed accessor to `Filetype::S16LH`;
    the three pinned-code positive paths; the unpinned-code
    negative path sweep `[0, 1, 4, 6, 7, 8, 9, 10, 11, 12, 255]`;
    `from_wire ↔ wire_value` round-trip on each pinned variant;
    TR.156 label match; per-variant `bytes_per_sample` width;
    per-variant `is_signed`; per-variant `is_little_endian`) plus
    6 integration tests in `tests/filetype_pinned_accessor.rs` that
    re-exercise the same contract through `parse_stream_header()`
    against (a) the real fixture-`F1` 11-byte prefix and (b)
    synthetic v2 headers stamped with each pinned numeric code plus
    five unpinned ones.

- **Round 244** — the `BLOCK_FN_BLOCKSIZE` housekeeping encoder
  (`spec/03` §3.6 + `spec/04` §4 + `spec/02` §3):
  * `write_blocksize_command(writer, new_bs)` emits the full
    `<fn=5> <new_bs>` command. The function code is written as
    `uvar(FNSIZE = 2)` over `FN_BLOCKSIZE = 5` (the 4-bit pattern
    `0101`); the `new_bs` payload is the two-stage `ulong()` form
    of `spec/02` §3 — `uvar(ULONGSIZE = 2)` over the per-value
    mantissa width followed by `uvar(width)` over the value —
    with the mantissa width chosen by the same `natural_ulong_width`
    minimum-width rule [`write_parameter_block`] applies to the six
    header fields.
  * Decode semantics. The decoder installs the returned value as
    its running sub-block size; subsequent predictor commands
    (`DIFF0..3` / `QLPC` / `ZERO`) produce blocks of `new_bs`
    samples per channel until the next `BLOCK_FN_BLOCKSIZE` command
    or end-of-stream (`spec/03` §3.6). The command does **not**
    advance the channel cursor; the per-channel dispatch resumes on
    the same channel after the override takes effect.
  * `FN_BLOCKSIZE = 5` (`spec/03` §3 + `spec/04` §4) — new public
    encoder constant, matching the existing per-command numeric
    `FN_DIFF0..3 / FN_QUIT / FN_QLPC / FN_BITSHIFT / FN_VERBATIM /
    FN_ZERO`.
  * New `EncodeError` variants `ZeroBlocksize` (rejection mirror of
    the decoder-side `Error::ZeroBlockSize`; the wire layout admits
    `new_bs = 0` in principle but `spec/03` §3.6 pins that the
    reference encoder never emits it — the residual loop would be
    empty, defeating the override's purpose) and
    `BlocksizeOutOfRange(u32)` (rejection mirror of the decoder-side
    `Error::BlockTooLarge` at the `BLOCKSIZE_MAX = 1 MiB`
    implementation safety cap). A rejected writer call surfaces the
    error without committing any partial command bytes.
  * 6 new in-module unit tests + 5 new integration tests
    (`tests/encoder_blocksize_pipeline.rs`) confirm: function-code
    constant matches `spec/04` §4 / T12; emitted bit counts match
    the `spec/02` §2.1 + §3 length formulas for the F2 anchor
    `new_bs = 155` (18 bits), the default `H_blocksize = 256`
    (19 bits), and the minimum `new_bs = 1` (9 bits); round-trip
    through `read_function_code` + `read_blocksize_payload`
    recovers the original `new_bs` for `{1, 2, 64, 155, 256, 1024,
    65536, BLOCKSIZE_MAX}`; `new_bs = 0` rejection produces
    `ZeroBlocksize` with zero bits committed; `BLOCKSIZE_MAX + 1`
    rejection produces `BlocksizeOutOfRange` with zero bits
    committed; an exact byte-pattern pin (`new_bs = 1` → `0x5B
    0x80`) corroborates the MSB-first packing. Integration: mono
    shrinking override (default `H_blocksize = 8` → `new_bs = 3`
    tail block) round-trips byte-exact through `decode_stream`;
    stereo round-robin override exercises the channel-cursor
    non-advancement rule of `spec/03` §3.6; an override that
    resets `new_bs` to the same value as `H_blocksize` is
    admissible and round-trips; the exact `spec/04` §4.1 T12
    anchor value `new_bs = 155` round-trips end-to-end; the
    degenerate single-sample override `new_bs = 1` round-trips.
  * **Scope.** Round 244 lands the `BLOCK_FN_BLOCKSIZE` writer
    primitive only. The remaining encoder gap is the per-block
    channel-round command sequencer (the higher layer that
    compares the cheapest of `DIFF0..3` / `QLPC` / `ZERO` per
    block and picks the predictor for each).

- **Round 251** — the **per-block predictor-selection sequencer**
  (`spec/03` §3.1..§3.4 + §3.9 + `spec/02` §2.1 + §2.2 +
  `spec/05` §2.3 + §2.4):
  * `select_predictor(samples, mu_chan, carry) -> Option<Choice>`
    — compares the natural-energy total encoded bit cost of every
    eligible candidate among `BLOCK_FN_DIFF0..3` and `BLOCK_FN_ZERO`
    and returns the cheapest. The bit cost is computed using the
    `spec/02` §2.1 `uvar(n)` length formula `⌊v / 2^n⌋ + 1 + n`
    plus the `spec/02` §2.2 svar folding; with natural-energy
    selection every residual lands at zero prefix-zero bits so each
    `svar(width)` costs exactly `1 + width` bits. Ties break in
    priority order `ZERO > DIFF0 > DIFF1 > DIFF2 > DIFF3`. Returns
    `None` for an empty block and for blocks whose residuals
    overflow every natural-energy width while ZERO is ineligible.
  * `Choice` enum — variants `Zero { bits }`, `Diff0 { energy,
    bits }`, `Diff1 { energy, bits }`, `Diff2 { energy, bits }`,
    `Diff3 { energy, bits }`. Each variant carries the natural
    energy parameter the writer will use plus the total encoded
    bit count of the command.
  * `write_selected_block(writer, choice, samples, mu_chan,
    carry)` — dispatches the [`Choice`] to the matching per-
    predictor writer ([`write_zero_block`] / [`write_diff0_block`]
    / [`write_diff1_block`] / [`write_diff2_block`] /
    [`write_diff3_block`]). The integration test
    `sequencer_emission_cost_matches_choice_bits` pins
    `bits_written` delta to `Choice::bits()` exactly — load-
    bearing for any higher-layer rate planner.
  * `evaluate_candidates` — debug-friendly accessor returning every
    eligible candidate in priority order, useful for inspecting
    why a particular block selects a particular predictor.
  * `BLOCK_FN_QLPC` is **not** auto-selected; the caller still
    owns coefficient quantisation per `spec/03` §3.5 and TR.156
    §3.2's Laplacian-distribution rule. A future round can layer
    QLPC into the selector by accepting a candidate coefficient
    vector and computing the same residual + natural-energy scan.
  * 15 new in-module unit tests + 7 new integration tests
    (`tests/encoder_sequencer_pipeline.rs`) confirm: `uvar_bits`
    matches `spec/02` §2.1 worked examples; `svar_bits` matches
    [`BitWriter::write_svar`]'s actual emitted bit count across
    a spread of widths + values; the ZERO token costs exactly 5
    bits; ZERO eligibility requires every sample to equal
    `mu_chan`; selector tie-break priority; DIFF0 / DIFF1 / DIFF3
    candidate evaluation; selector picks `Choice::Zero` for an
    all-zero block (with `mu_chan = 0`); selector picks
    `Choice::Diff1` for an arithmetic-progression input where the
    first-difference stream is small; full round-trip through
    `decode_stream` for mono / stereo / back-to-back blocks /
    mixed predictor streams; cost-equality property.
  * **Scope.** Round 251 lands the auto-selection layer above the
    DIFF0..3 + ZERO writers. The remaining gaps are QLPC auto-
    selection (needs coefficient quantisation) and — closed in
    round 254 below — the Rice-`n` statistical optimum of
    TR.156 §3.3.

- **Round 254** — **Rice-`n` statistical-optimum energy selection**
  inside the per-block predictor-selection sequencer (`spec/02` §2.1
  + §4.2 + `spec/05` §3 + TR.156 §3.3):
  * New public encoder helpers
    `residual_bits_at_energy(residuals, encoded_energy) -> Option<u64>`
    and `optimal_energy_for_residuals(residuals) -> Option<u32>`. The
    bit-count helper sums the per-sample
    `⌊u / 2^width⌋ + 1 + width` `svar` cost of `spec/02` §2.1 across
    a residual stream at any encoded energy whose mantissa width
    fits the decoder cap (`MAX_RESIDUAL_WIDTH = 30`). The optimum
    helper sweeps `e ∈ 0..=MAX_NATURAL_ENERGY` and returns whichever
    minimises the bit count — the empirical equivalent of TR.156
    §3.3 equation 21's `n ≈ log₂(log(2) · E(|x|))` over the
    natural-band of encoded energies the `uvar(ENERGYSIZE = 3)`
    field of `spec/02` §4.2 admits at its natural 4-bit width.
  * Per-predictor wrappers
    `optimal_energy_for_diff0` / `..diff1` / `..diff2` / `..diff3` /
    `..qlpc` mirror the existing `min_energy_for_*` family so the
    call-site intent stays explicit.
  * `select_predictor` (and `evaluate_candidates`) now score every
    `BLOCK_FN_DIFFn` candidate at the Rice-`n` optimum rather than
    at the natural energy. Observed effects:
    - Sparse seed-jump streams (e.g. `[3, 0, 0, 0]` from DIFF1 with
      a fresh carry) move from `e = 2` / 23 bits to `e = 0` / 18
      bits — a 22 % saving on this single block.
    - Arithmetic-progression streams move from `Diff1 { e = 3 }`
      (~327-bit cost at `N = 64`) to `Diff2 { e = 0 }` (~140-bit
      cost) — DIFF2's second-difference is non-zero in exactly
      one position, and under Rice-`n` the optimum prefers the
      order-2 predictor's sparser residual stream over DIFF1's
      constant non-zero one.
  * `Choice::bits()` continues to reflect the **emitted** bit count
    of the chosen `(predictor, energy)` pair; the writer side
    consumes the choice unchanged. Higher-layer rate planners can
    therefore continue to call `Choice::bits()` to budget per-block
    cost without re-deriving the metric.
  * 6 new in-module unit tests in `encoder.rs`
    (`residual_bits_at_energy_matches_bitwriter_actual_count` —
    crosschecks the cost helper against the BitWriter's actual
    `write_svar` output across 7 streams × 8 energies;
    `..rejects_width_above_residual_cap`;
    `optimal_energy_matches_natural_when_residuals_are_tight`;
    `optimal_energy_picks_smaller_e_on_sparse_seed_jump_streams`;
    `optimal_energy_per_predictor_wrappers_agree_with_shared_scan`;
    `optimal_energy_rejects_empty_residual_stream`). Sequencer unit
    tests and the
    `tests/encoder_sequencer_pipeline.rs::mono_sequencer_picks_diff_for_arithmetic_ramp`
    integration test are rewritten in the same commit to assert the
    new optimum behaviour with the TR.156 §3.3 rationale spelled
    out; the round-trips through `decode_stream` continue to be
    byte-exact.
  * **Scope.** Round 254 closes the Rice-`n` statistical-optimum
    follow-up of round 251. The other round-251 follow-up — QLPC
    auto-selection — still requires a candidate coefficient-
    quantisation pass and remains future work.

The combined surface is exercised by **359 tests** (271 in-module
unit + 88 integration tests across 19 integration binaries). The
integration suite composes the header parse
with the per-block dispatch: VERBATIM-then-QUIT (round 2), multi-
channel DIFF1-DIFF1-DIFF1-QUIT with the round-robin cursor of
`spec/03` §2 (round 3), DIFF0-ZERO-DIFF0-QUIT exercising the
running-mean estimator's sliding-window update + `BLOCK_FN_ZERO`'s
`mu_chan` fill (round 4), QLPC-QLPC-QUIT exercising an order-2 LPC
predictor with the per-channel carry hand-off across two blocks
(round 5), BITSHIFT-DIFF1-BLOCKSIZE-DIFF1-QUIT exercising the
two housekeeping commands' state updates plus the carry hand-off
across the BLOCKSIZE-override boundary (round 6), a full
VERBATIM-BITSHIFT-DIFF1-DIFF0-BLOCKSIZE-DIFF1-QUIT stream decoded
end-to-end through the public `decode_stream` driver (round 7),
a trait-driven decode of a synthetic two-channel `s16lh` stream
resolved out of `CodecRegistry::first_decoder` whose `AudioFrame`
plane bytes match the direct `decode_stream` output byte-for-byte
(round 8), two SHNAMPSK-trailer composition tests verifying that
a synthetic `s16lh` stream decodes identically with and without a
well-formed trailer appended after `BLOCK_FN_QUIT`'s zero-bit
padding (round 9), and two streaming-iterator pipeline tests that
(a) feed a multi-command fixture (VERBATIM-BITSHIFT-DIFF1-DIFF0-
BLOCKSIZE-DIFF1-VERBATIM-QUIT) through both `decode_stream` and
the new `StreamDecoder` and assert per-channel sample equality +
concatenated verbatim equality + iterator-state mutation, and
(b) verify on-demand pulling (decoding the first sample block
leaves the second block undecoded until the next pull) (round 10).

### What's not yet here

* Encoder-side **coefficient quantisation** for QLPC. The round-17
  `write_qlpc_block` entry point accepts the caller's already-
  quantised coefficient vector as `&[i64]`; the optimal-quantisation
  derivation from raw input samples per TR.156 §3.2 (the
  Laplacian-distribution rule) remains a higher-layer concern.
* **QLPC auto-selection** — the round-251 sequencer
  (`select_predictor`) covers `BLOCK_FN_DIFF0..3` and
  `BLOCK_FN_ZERO` but not `BLOCK_FN_QLPC`, because the caller
  still owns coefficient quantisation per `spec/03` §3.5 and
  TR.156 §3.2. A future round can extend the selector to compare a
  caller-supplied candidate coefficient vector against the DIFFn
  family and pick QLPC when its residual stream wins on bit cost
  under the round-254 Rice-`n` statistical-optimum metric. The
  encoder helper `optimal_energy_for_qlpc` already exposes the
  metric so the QLPC-side change is purely the addition of a new
  `Choice::Qlpc` variant and its candidate-evaluation path inside
  the sequencer.
* Sample-format byte-packing for the eight TR.156 labels that
  `spec/05` §6 leaves with unpinned numeric codes (`ulaw`, `s8`,
  `s16`, `u16`, `s16x`, `u16x`, `u16hl`, `u16lh`); unblocking
  requires either additional fixtures or the reference encoder's
  output observed under each `-t` invocation.

## License

MIT — see [LICENSE](./LICENSE).
