# oxideav-shorten

A pure-Rust Shorten (`.shn`) lossless audio codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Clean-room rebuild — round 12 (2026-05-30).** The crate was
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

The combined surface is exercised by **203 tests** (177 unit + 26
integration). The integration suite composes the header parse
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

* DIFF1..3 / QLPC **predictor encoder** path: round 13 lands
  `BLOCK_FN_DIFF0`, but the higher-order polynomial-difference
  predictors (`BLOCK_FN_DIFF1..3`) and the general LPC predictor
  (`BLOCK_FN_QLPC`) still need encoder-side residual computation
  off the per-channel sample-history carry (orders 1..3) and
  coefficient quantisation per `spec/03` §3.5 (QLPC), plus the
  Rice-`n` optimal-selection of TR.156 §3.3 and the channel-round
  command sequencer.
* Sample-format byte-packing for the eight TR.156 labels that
  `spec/05` §6 leaves with unpinned numeric codes (`ulaw`, `s8`,
  `s16`, `u16`, `s16x`, `u16x`, `u16hl`, `u16lh`); unblocking
  requires either additional fixtures or the reference encoder's
  output observed under each `-t` invocation.

## License

MIT — see [LICENSE](./LICENSE).
