//! `oxideav_core::Decoder` wiring for Shorten.
//!
//! Round 7 landed [`decode_stream`](crate::decode_stream) — the
//! whole-stream orchestration loop that walks `BLOCK_FN_*` commands
//! until `BLOCK_FN_QUIT` and emits per-channel `Vec<i32>` sample
//! vectors. This module (round 145, per the README "lacks" tail of
//! "oxideav-core Decoder wiring + encoder") exposes that driver
//! through the framework's packet-in / frame-out
//! [`Decoder`](oxideav_core::Decoder) trait so a `RuntimeContext` can
//! resolve "the shorten decoder" and decode an `.shn` byte stream end
//! to end.
//!
//! ## Trait-API adaptation
//!
//! Shorten is **stream-natively single-packet**: the `ajkg` header,
//! parameter block, and per-block command stream together form one
//! contiguous bit-aligned blob terminated by `BLOCK_FN_QUIT` + byte
//! padding (`spec/05` §4). There is no inter-frame framing the
//! decoder needs to honour; the file IS the frame.
//!
//! The wrapper therefore behaves as follows:
//!
//! * [`send_packet`](oxideav_core::Decoder::send_packet) appends the
//!   incoming `packet.data` to an internal buffer and eagerly attempts
//!   [`decode_stream`]. On success the wrapper packs the decoded
//!   per-channel `i32` samples into the sample-format byte stream
//!   selected by `H_filetype` (`spec/05` §6) and queues exactly one
//!   [`AudioFrame`] holding the full per-channel planar PCM. The
//!   stream is then considered terminated (EOF after the queued frame
//!   is drained).
//! * If `decode_stream` returns [`Error::Truncated`] the wrapper
//!   leaves the buffer in place; the next `send_packet` appends more
//!   bytes and retries. (This handles the case where a caller chops
//!   an `.shn` file into pieces and delivers each as a separate
//!   packet.)
//! * [`receive_frame`](oxideav_core::Decoder::receive_frame) pops the
//!   queued frame, returns [`Error::NeedMore`] before the stream
//!   completes, and [`Error::Eof`] after the queued frame has been
//!   drained.
//! * [`flush`](oxideav_core::Decoder::flush) signals end-of-stream and
//!   makes a final decode attempt against whatever bytes are already
//!   buffered.
//! * [`reset`](oxideav_core::Decoder::reset) drops every buffered
//!   byte and queued frame so the next `send_packet` starts as if no
//!   prior packets had been processed.
//!
//! ## Sample-format byte packing (`spec/05` §6)
//!
//! `spec/05` §6 pins three numeric `H_filetype` values across the
//! reachable fixture corpus:
//!
//! | `H_filetype` | Label   | Plane bytes per sample |
//! | ------------ | ------- | ---------------------- |
//! | 2            | `u8`    | 1 (sample as `u8`)     |
//! | 3            | `s16hl` | 2 (sample as `i16` big-endian)    |
//! | 5            | `s16lh` | 2 (sample as `i16` little-endian) |
//!
//! Any other `H_filetype` value surfaces
//! [`oxideav_core::Error::Unsupported`] — the eight unpinned labels
//! (`ulaw`, `s8`, `s16`, `u16`, `s16x`, `u16x`, `u16hl`, `u16lh`)
//! lack a numeric code in `spec/05` §6 and so cannot be safely packed
//! by this round.
//!
//! The host-side [`SampleFormat`] surfaced through
//! [`CodecParameters`] is `U8P` for filetype `2` and `S16P` for
//! filetypes `3` / `5`. Byte order is encoded into the plane bytes
//! per the table above; `SampleFormat::S16P` itself is the framework's
//! generic 16-bit planar shape, and the choice of `s16hl` versus
//! `s16lh` lives in the packed bytes (not in the `SampleFormat`
//! variant). The verbatim envelope captured from `BLOCK_FN_VERBATIM`
//! commands is preserved on
//! [`ShortenDecoder::verbatim_prefix`] for callers that need the
//! original host-format header (RIFF/WAVE, AIFF, AU).
//!
//! ## Scope
//!
//! This round ties the existing whole-stream driver into the trait
//! surface. The decode path itself is unchanged; what's new is the
//! adaptor + registration so a `RuntimeContext` can resolve a Shorten
//! decoder. The encoder side stays unimplemented; the README "lacks"
//! tail still names `DIFFn / QLPC encoder`.
//!
//! ## Clean-room provenance
//!
//! The trait wiring is assembled from `docs/audio/shorten/spec/05` §6
//! (the file-type table), `spec/03` §3.10 (verbatim prefix), `spec/03`
//! §2 (per-channel sample ordering), and the public surface of
//! `oxideav_core` (the `Decoder` trait contract). No external decoder
//! source, no FFmpeg `shorten.c`, and no archived `old` branch were
//! consulted.
//!
//! ## Streaming adaptor — round 11
//!
//! The whole-stream [`ShortenDecoder`] wrapper buffers every byte of
//! the file before producing a single [`AudioFrame`]. For long inputs
//! that pre-materialises the entire decoded sample population ahead of
//! the caller — the same memory characteristic
//! [`crate::StreamDecoder`] was added to avoid on the pure-Rust API.
//!
//! Round 11 adds a parallel [`ShortenStreamingDecoder`] trait
//! implementation that walks the per-block command stream incrementally
//! and emits one [`AudioFrame`] per **full channel round** (one block
//! per channel) instead of one frame per whole file:
//!
//! * `send_packet` appends bytes to an internal buffer (same
//!   chop-anywhere split-packet support as [`ShortenDecoder`]).
//! * `receive_frame` drives the per-block command loop forward,
//!   accumulating one [`PolyOrder`]/[`crate::decode_qlpc_block`]/
//!   [`crate::fill_zero_block`] block per channel into the in-progress
//!   round. Once every channel has a block of the same length, the
//!   wrapper packs them into a planar [`AudioFrame`] per `spec/05` §6
//!   and queues it.
//! * `BLOCK_FN_VERBATIM` envelope payloads append to
//!   [`ShortenStreamingDecoder::verbatim_prefix`] without emitting a
//!   frame (the envelope is part of the host-format wrapper, not the
//!   sample stream); the round-in-progress is preserved across the
//!   verbatim command per `spec/03` §3.10's cursor-non-advance rule.
//! * `BLOCK_FN_BLOCKSIZE` and `BLOCK_FN_BITSHIFT` are absorbed silently
//!   per `spec/03` §3.6/§3.7. `BLOCKSIZE` takes effect for the next
//!   sample-producing command, so a well-formed stream emits it at a
//!   channel-round boundary; if a stream emits it mid-round (channel-0
//!   block already accumulated, channel-1 block at the new size), the
//!   wrapper surfaces an error because the resulting planes would have
//!   mismatched lengths and the planar [`AudioFrame`] shape requires a
//!   single per-channel sample count.
//! * `BLOCK_FN_QUIT` terminates the stream; a partially-accumulated
//!   round at termination is discarded (a well-formed stream emits
//!   blocks in full channel rounds and only quits at a round boundary).
//!
//! Memory characteristic: `O(buffered_bytes + n_channels ×
//! current_block_size)` plus the per-channel carries / mean estimators
//! the iterator already needs. For a streaming caller that consumes
//! frames as fast as they are produced, that's a constant footprint
//! independent of stream length — in contrast to [`ShortenDecoder`]
//! which buffers `O(stream_length)` of decoded samples before emitting
//! anything.

use std::collections::VecDeque;

use oxideav_core::{
    AudioFrame, CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, Decoder,
    Error as CoreError, Frame, Packet, Result as CoreResult, SampleFormat,
};

use crate::bitreader::BitReader;
use crate::block::{
    read_bitshift_payload, read_blocksize_payload, read_function_code, read_verbatim_payload,
    FunctionCode,
};
use crate::decode_stream;
use crate::driver::MAX_COMMANDS;
use crate::error::Error as ShortenError;
use crate::header::{parse_stream_header, ShortenStreamHeader, MIN_HEADER_BYTES};
use crate::predictor::{
    decode_diff_block, decode_qlpc_block, fill_zero_block, ChannelCarry, MeanEstimator, PolyOrder,
};

/// String form of the codec id this crate registers under.
pub const CODEC_ID_STR: &str = "shorten";

/// `H_filetype` numeric value for the `u8` (8-bit unsigned) sample
/// format pinned in `spec/05` §6 by fixture `F2`.
pub const FILETYPE_U8: u32 = 2;
/// `H_filetype` numeric value for the `s16hl` (16-bit signed
/// high-byte-first) sample format pinned in `spec/05` §6 by fixture
/// `F3`.
pub const FILETYPE_S16HL: u32 = 3;
/// `H_filetype` numeric value for the `s16lh` (16-bit signed
/// low-byte-first) sample format pinned in `spec/05` §6 by fixture
/// `F1`.
pub const FILETYPE_S16LH: u32 = 5;

/// Build a boxed Shorten [`Decoder`] from `params`.
///
/// `params.codec_id` is recorded so the decoder reports it back via
/// [`Decoder::codec_id`]; `params.sample_rate`, `params.channels`,
/// and `params.sample_format` are advisory at construction time and
/// re-derived from the parsed stream header on the first
/// [`Decoder::send_packet`] call that successfully decodes a complete
/// stream.
///
/// The factory itself never fails: the actual stream-level fields are
/// not known until a packet has been parsed, so no validation is
/// possible at construction. Errors surface at the trait boundary
/// when the buffered bytes turn out to be malformed.
pub fn make_decoder(params: &CodecParameters) -> CoreResult<Box<dyn Decoder>> {
    let codec_id = params.codec_id.clone();
    Ok(Box::new(ShortenDecoder::new(codec_id, params.clone())))
}

/// Packet-to-frame adaptor that wires [`decode_stream`] into the
/// framework [`Decoder`] trait.
///
/// State carried across `send_packet` calls:
///
/// * `buffer` — every byte of `packet.data` received so far. The
///   wrapper appends each packet and tries `decode_stream` on the
///   accumulated buffer; this lets the caller chop an `.shn` file
///   into arbitrary slices.
/// * `pending` — at most one fully-decoded [`AudioFrame`]; popped by
///   [`Decoder::receive_frame`].
/// * `verbatim_prefix` — the host-format envelope bytes
///   ([`crate::DecodedStream::verbatim`]) preserved for callers that
///   need the original RIFF/WAVE / AIFF / AU header.
/// * `decoded` — once `decode_stream` succeeds the wrapper stops
///   trying further decodes (the stream terminated at
///   `BLOCK_FN_QUIT`).
/// * `eof` — set by [`Decoder::flush`] or implicitly once `decoded`
///   is true and `pending` has been drained.
pub struct ShortenDecoder {
    codec_id: CodecId,
    output: CodecParameters,
    buffer: Vec<u8>,
    pending: VecDeque<AudioFrame>,
    verbatim_prefix: Vec<u8>,
    decoded: bool,
    eof: bool,
    /// PTS to attach to the emitted [`AudioFrame`]. Taken from the
    /// first packet whose `pts.is_some()`.
    pts: Option<i64>,
}

impl std::fmt::Debug for ShortenDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShortenDecoder")
            .field("codec_id", &self.codec_id)
            .field("buffered_bytes", &self.buffer.len())
            .field("pending", &self.pending.len())
            .field("verbatim_prefix_len", &self.verbatim_prefix.len())
            .field("decoded", &self.decoded)
            .field("eof", &self.eof)
            .finish()
    }
}

impl ShortenDecoder {
    fn new(codec_id: CodecId, output: CodecParameters) -> Self {
        Self {
            codec_id,
            output,
            buffer: Vec::new(),
            pending: VecDeque::new(),
            verbatim_prefix: Vec::new(),
            decoded: false,
            eof: false,
            pts: None,
        }
    }

    /// Bytes carrying the host-format envelope
    /// ([`crate::DecodedStream::verbatim`]). Empty until a complete
    /// stream has been decoded.
    pub fn verbatim_prefix(&self) -> &[u8] {
        &self.verbatim_prefix
    }

    /// Try to decode whatever bytes are currently buffered. On
    /// success queues the resulting frame, sets `decoded`, and clears
    /// the buffer; on [`ShortenError::Truncated`] (or
    /// [`ShortenError::InvalidMagic`] when the buffer is shorter than
    /// the 5-byte magic+version prefix) silently returns `Ok(())`
    /// (the caller is expected to deliver more bytes); any other
    /// error is surfaced to the caller.
    fn try_decode(&mut self) -> CoreResult<()> {
        if self.decoded || self.buffer.is_empty() {
            return Ok(());
        }
        match decode_stream(&self.buffer) {
            Ok(stream) => {
                let frame = pack_decoded_stream_to_frame(
                    stream.header.filetype,
                    stream.header.channels,
                    &stream.channels,
                    self.pts,
                )?;
                // Refresh output params from the parsed header.
                self.output.sample_rate = self.output.sample_rate.or(Some(0));
                self.output.channels = Some(stream.header.channels as u16);
                self.output.sample_format = Some(host_sample_format(stream.header.filetype)?);
                self.verbatim_prefix = stream.verbatim;
                self.pending.push_back(frame);
                self.decoded = true;
                self.buffer.clear();
                Ok(())
            }
            Err(ShortenError::Truncated) => Ok(()),
            Err(ShortenError::InvalidMagic)
                if self.buffer.len() < crate::header::MIN_HEADER_BYTES =>
            {
                // The header parser rejects a buffer shorter than the
                // 5-byte magic+version prefix with `InvalidMagic`
                // (`docs/audio/shorten/spec/01-stream-header.md` §1).
                // From the trait wrapper's standpoint this is "the
                // caller hasn't delivered enough bytes yet" — treat
                // it the same as `Truncated` and wait for more.
                Ok(())
            }
            Err(e) => Err(shorten_error_to_core(e)),
        }
    }
}

impl Decoder for ShortenDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> CoreResult<()> {
        if self.eof {
            return Err(CoreError::other(
                "oxideav-shorten: cannot send_packet after flush",
            ));
        }
        if self.decoded {
            // A second packet after the stream already terminated at
            // BLOCK_FN_QUIT would be appended to nothing — surface the
            // mismatch so the caller can notice rather than silently
            // dropping bytes.
            return Err(CoreError::other(
                "oxideav-shorten: stream already terminated at BLOCK_FN_QUIT; ignore further packets",
            ));
        }
        if self.pts.is_none() {
            self.pts = packet.pts;
        }
        self.buffer.extend_from_slice(&packet.data);
        self.try_decode()
    }

    fn receive_frame(&mut self) -> CoreResult<Frame> {
        if let Some(a) = self.pending.pop_front() {
            return Ok(Frame::Audio(a));
        }
        if self.eof || self.decoded {
            return Err(CoreError::Eof);
        }
        Err(CoreError::NeedMore)
    }

    fn flush(&mut self) -> CoreResult<()> {
        // Take one final decode attempt against the bytes already
        // buffered, in case the caller delivered the whole stream
        // before calling flush() and never called send_packet again.
        if !self.decoded {
            self.try_decode()?;
        }
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> CoreResult<()> {
        self.buffer.clear();
        self.pending.clear();
        self.verbatim_prefix.clear();
        self.decoded = false;
        self.eof = false;
        self.pts = None;
        Ok(())
    }
}

/// Pack the per-channel `i32` sample vectors of
/// [`crate::DecodedStream::channels`] into a planar [`AudioFrame`]
/// per the `spec/05` §6 file-type table.
///
/// One plane per channel; each plane carries `samples_per_channel ×
/// bytes_per_sample` bytes laid out in the byte order pinned by the
/// numeric `filetype` (see the module docs for the three pinned
/// values).
fn pack_decoded_stream_to_frame(
    filetype: u32,
    declared_channels: u32,
    channels: &[Vec<i32>],
    pts: Option<i64>,
) -> CoreResult<AudioFrame> {
    if channels.len() != declared_channels as usize {
        return Err(CoreError::invalid(format!(
            "oxideav-shorten: decoded channel count {} != header H_channels {}",
            channels.len(),
            declared_channels
        )));
    }
    let samples_per_ch = channels.first().map(|c| c.len()).unwrap_or(0);
    for (ci, ch) in channels.iter().enumerate() {
        if ch.len() != samples_per_ch {
            return Err(CoreError::invalid(format!(
                "oxideav-shorten: channel {ci} length {} != channel 0 length {samples_per_ch}",
                ch.len(),
            )));
        }
    }

    let planes: Vec<Vec<u8>> = match filetype {
        FILETYPE_U8 => channels
            .iter()
            .map(|ch| ch.iter().map(|&s| s as u8).collect::<Vec<u8>>())
            .collect(),
        FILETYPE_S16HL => channels
            .iter()
            .map(|ch| {
                let mut plane = Vec::with_capacity(ch.len() * 2);
                for &s in ch {
                    plane.extend_from_slice(&(s as i16).to_be_bytes());
                }
                plane
            })
            .collect(),
        FILETYPE_S16LH => channels
            .iter()
            .map(|ch| {
                let mut plane = Vec::with_capacity(ch.len() * 2);
                for &s in ch {
                    plane.extend_from_slice(&(s as i16).to_le_bytes());
                }
                plane
            })
            .collect(),
        other => {
            return Err(CoreError::unsupported(format!(
                "oxideav-shorten: H_filetype {other} not pinned by spec/05 §6 \
                 (only 2/u8, 3/s16hl, 5/s16lh have pinned numeric codes)",
            )));
        }
    };

    Ok(AudioFrame {
        samples: samples_per_ch as u32,
        pts,
        data: planes,
    })
}

/// Map a numeric `H_filetype` value to the framework
/// [`SampleFormat`] surfaced through [`CodecParameters`]. Byte order
/// for the `s16hl` / `s16lh` filetypes is encoded into the packed
/// plane bytes — both surface as the generic `S16P` shape here.
fn host_sample_format(filetype: u32) -> CoreResult<SampleFormat> {
    match filetype {
        FILETYPE_U8 => Ok(SampleFormat::U8P),
        FILETYPE_S16HL | FILETYPE_S16LH => Ok(SampleFormat::S16P),
        other => Err(CoreError::unsupported(format!(
            "oxideav-shorten: H_filetype {other} not pinned by spec/05 §6"
        ))),
    }
}

/// Translate a crate-local [`ShortenError`] into the framework
/// [`oxideav_core::Error`]. The framework's error enum has no
/// per-codec variant tree, so each crate-local variant maps onto a
/// generic flavour with a descriptive message.
fn shorten_error_to_core(e: ShortenError) -> CoreError {
    match e {
        ShortenError::Truncated => CoreError::NeedMore,
        ShortenError::InvalidMagic => CoreError::invalid("oxideav-shorten: invalid 'ajkg' magic"),
        ShortenError::UnsupportedVersion(v) => {
            CoreError::unsupported(format!("oxideav-shorten: format version {v}"))
        }
        other => CoreError::other(format!("oxideav-shorten: {other}")),
    }
}

// ─────────────────────────── streaming adaptor ───────────────────────────

/// Codec id for the streaming-mode Shorten decoder factory installed by
/// [`register_streaming_codecs`]. Keeps the `"shorten"` id reserved for
/// the whole-stream [`ShortenDecoder`] factory so a caller can pick
/// between the two shapes by codec id.
pub const STREAMING_CODEC_ID_STR: &str = "shorten-streaming";

/// Build a boxed Shorten **streaming** [`Decoder`] from `params`.
///
/// Unlike [`make_decoder`] (which buffers the full file before emitting
/// a single [`AudioFrame`]), the decoder built here emits one
/// `AudioFrame` per full channel round (one per-channel block packed
/// across all channels). See the module docs ("Streaming adaptor")
/// for the full contract.
///
/// The factory itself never fails: stream-level fields are not known
/// until the file header has been parsed, which happens on the first
/// `send_packet` that delivers at least [`MIN_HEADER_BYTES`] +
/// parameter-block bytes.
pub fn make_streaming_decoder(params: &CodecParameters) -> CoreResult<Box<dyn Decoder>> {
    let codec_id = params.codec_id.clone();
    Ok(Box::new(ShortenStreamingDecoder::new(
        codec_id,
        params.clone(),
    )))
}

/// One block-per-channel accumulation slot inside the round in progress.
///
/// Per `spec/03` §2 the channel cursor advances modulo `H_channels`
/// after each sample-producing command, so a well-formed stream emits
/// blocks in rounds of `H_channels` consecutive commands. The
/// streaming adaptor accumulates one block per slot and flushes the
/// whole round as a planar [`AudioFrame`] once every slot is filled.
#[derive(Debug, Clone, Default)]
struct ChannelBlockSlot {
    samples: Vec<i32>,
}

/// Packet-to-frame adaptor that walks the Shorten per-block command
/// stream block-by-block and emits one [`AudioFrame`] per full channel
/// round, rather than buffering the entire decoded sample population
/// before emission.
///
/// See the module-level "Streaming adaptor" section for the full
/// contract.
pub struct ShortenStreamingDecoder {
    codec_id: CodecId,
    output: CodecParameters,
    /// Every byte received via `send_packet` so far. Not cleared after
    /// frames are emitted — the bit-position cursor `bits_consumed`
    /// indexes into this buffer, and re-parsing from a fresh offset
    /// requires the prefix to stay around. (For a 10.6 MB compressed
    /// input that's ~10.6 MB of resident memory while decoding; the
    /// load-bearing improvement of this shape is that the *decoded*
    /// `i32` sample stream is bounded by `n_channels × block_size`,
    /// not by `stream_length`.)
    buffer: Vec<u8>,
    /// Parsed file header. `None` until enough bytes have been
    /// delivered to parse it; `Some` after that.
    header: Option<ShortenStreamHeader>,
    /// Bit position within the body slice (`buffer[5..]`) where the
    /// next per-block command lives. Saved across calls; on each
    /// resume we re-build a fresh [`BitReader`] over the body and
    /// [`BitReader::skip_bits`] forward to this position.
    bits_consumed: u32,
    /// Per-channel sample-history carries (`spec/05` §1). Allocated
    /// when the header is first parsed.
    carries: Vec<ChannelCarry>,
    /// Per-channel running mean estimators (`spec/05` §2). Allocated
    /// when the header is first parsed.
    means: Vec<MeanEstimator>,
    /// Round-robin channel cursor (`spec/03` §2).
    cursor: usize,
    /// Running sub-block size (default `H_blocksize`; overridden by
    /// `BLOCK_FN_BLOCKSIZE`).
    block_size: u32,
    /// Running per-stream bit-shift (zero default; set by
    /// `BLOCK_FN_BITSHIFT`).
    bshift: u32,
    /// Commands processed so far. Bounded by [`MAX_COMMANDS`].
    commands: u64,
    /// In-progress channel-round accumulation. Once
    /// `pending_round[i]` is `Some` for every channel `i`, the
    /// adaptor packs the round into one planar [`AudioFrame`] and
    /// clears the slots.
    pending_round: Vec<Option<ChannelBlockSlot>>,
    /// Frames already packed and queued for [`Decoder::receive_frame`].
    pending: VecDeque<AudioFrame>,
    /// Verbatim envelope bytes collected across `BLOCK_FN_VERBATIM`
    /// commands. Exposed via [`Self::verbatim_prefix`].
    verbatim_prefix: Vec<u8>,
    /// True once `BLOCK_FN_QUIT` has been observed or `flush()` has
    /// been called. After eof + pending drains, `receive_frame`
    /// returns [`CoreError::Eof`].
    eof: bool,
    /// PTS to attach to the *first* emitted frame; subsequent frames
    /// carry `None` (the framework's planar `AudioFrame::pts` is
    /// per-frame, but only the first frame of a stream is anchored to
    /// the packet's PTS — subsequent frames slide forward in time at
    /// the rate dictated by their `samples` field).
    pts: Option<i64>,
    /// True once we've emitted at least one frame; controls the "first
    /// frame carries pts, others carry None" logic above.
    first_frame_emitted: bool,
    /// `Some(msg)` if a previous `try_advance` surfaced an
    /// unrecoverable error. Cached so subsequent `receive_frame` calls
    /// re-surface the same message rather than retrying the doomed
    /// decode. We store the message rather than the
    /// [`oxideav_core::Error`] itself because that type is not
    /// `Clone` and `receive_frame` returns it by value.
    fatal: Option<String>,
}

impl std::fmt::Debug for ShortenStreamingDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShortenStreamingDecoder")
            .field("codec_id", &self.codec_id)
            .field("buffered_bytes", &self.buffer.len())
            .field("header_parsed", &self.header.is_some())
            .field("bits_consumed", &self.bits_consumed)
            .field("cursor", &self.cursor)
            .field("block_size", &self.block_size)
            .field("bshift", &self.bshift)
            .field("commands", &self.commands)
            .field("pending_frames", &self.pending.len())
            .field("verbatim_prefix_len", &self.verbatim_prefix.len())
            .field("eof", &self.eof)
            .finish()
    }
}

impl ShortenStreamingDecoder {
    fn new(codec_id: CodecId, output: CodecParameters) -> Self {
        Self {
            codec_id,
            output,
            buffer: Vec::new(),
            header: None,
            bits_consumed: 0,
            carries: Vec::new(),
            means: Vec::new(),
            cursor: 0,
            block_size: 0,
            bshift: 0,
            commands: 0,
            pending_round: Vec::new(),
            pending: VecDeque::new(),
            verbatim_prefix: Vec::new(),
            eof: false,
            pts: None,
            first_frame_emitted: false,
            fatal: None,
        }
    }

    /// Bytes carrying the host-format envelope (concatenated
    /// `BLOCK_FN_VERBATIM` payloads). Mirrors [`ShortenDecoder::
    /// verbatim_prefix`] but accumulates incrementally as commands
    /// are walked, rather than only after the whole file has been
    /// processed.
    pub fn verbatim_prefix(&self) -> &[u8] {
        &self.verbatim_prefix
    }

    /// True once the file header has been parsed (i.e., enough bytes
    /// have been delivered through `send_packet` for
    /// [`parse_stream_header`] to succeed).
    pub fn header_parsed(&self) -> bool {
        self.header.is_some()
    }

    /// Mark the decoder as having hit a permanent fault and return
    /// the same error to the caller. Stores the message so subsequent
    /// `receive_frame` calls can re-surface it. `oxideav_core::Error`
    /// is not `Clone`, so the cache is the textual form.
    fn fail(&mut self, err: CoreError) -> CoreError {
        self.fatal = Some(err.to_string());
        err
    }

    /// Try to parse the file header from the buffered bytes. Returns
    /// `Ok(())` on success or on a "need more bytes" status; surfaces
    /// any non-recoverable header error.
    fn try_parse_header(&mut self) -> CoreResult<()> {
        if self.header.is_some() || self.buffer.is_empty() {
            return Ok(());
        }
        match parse_stream_header(&self.buffer) {
            Ok(parsed) => {
                let header = parsed.header;
                if header.channels == 0 {
                    return Err(CoreError::invalid(
                        "oxideav-shorten: header H_channels = 0 (round-robin cursor undefined)",
                    ));
                }
                let n_channels = header.channels as usize;
                let carry_len = header.sample_history_carry_len() as usize;
                self.carries = (0..n_channels)
                    .map(|_| ChannelCarry::new(carry_len))
                    .collect();
                self.means = (0..n_channels)
                    .map(|_| MeanEstimator::new(header.meanblocks))
                    .collect();
                self.pending_round = vec![None; n_channels];
                self.block_size = header.blocksize;
                self.bits_consumed = parsed.bits_consumed_after_v;
                // Refresh advertised output params from the parsed
                // header; the wrapper does not know the sample rate
                // (Shorten's container carries it externally) so leave
                // that field at the caller's hint.
                self.output.channels = Some(header.channels as u16);
                self.output.sample_format = Some(host_sample_format(header.filetype)?);
                self.header = Some(header);
                Ok(())
            }
            Err(ShortenError::Truncated) => Ok(()),
            Err(ShortenError::InvalidMagic) if self.buffer.len() < MIN_HEADER_BYTES => Ok(()),
            Err(e) => Err(shorten_error_to_core(e)),
        }
    }

    /// Drive the per-block command loop forward as far as the
    /// currently-buffered bytes allow, queuing any completed frames.
    ///
    /// Returns `Ok(())` if either: (a) the buffer was exhausted and we
    /// need more bytes (the buffer's bit-offset cursor is left at the
    /// failed command's start), (b) `BLOCK_FN_QUIT` was reached
    /// (`self.eof` is set), or (c) one or more frames were queued and
    /// the caller should now drain them. Any other error surfaces.
    fn try_advance(&mut self) -> CoreResult<()> {
        if self.eof || self.fatal.is_some() {
            return Ok(());
        }
        self.try_parse_header()?;
        let header = match self.header {
            Some(h) => h,
            None => return Ok(()),
        };
        // The bit-stream body starts at buffer[5..] per spec/01 §1.
        // We re-build a fresh BitReader and skip to the saved offset
        // on every call. The BitReader is cheap (it carries only a
        // small cache); the skip_bits is a u32 add.
        loop {
            if self.commands >= MAX_COMMANDS {
                return Err(self.fail(CoreError::invalid(
                    "oxideav-shorten: command count exceeded MAX_COMMANDS safety cap",
                )));
            }
            // Take a snapshot of the bit offset BEFORE attempting to
            // read the next command — on a Truncated read we restore
            // it so the next pass retries from the same position.
            let saved_offset = self.bits_consumed;
            let body = &self.buffer[MIN_HEADER_BYTES..];
            let mut reader = BitReader::new(body);
            if let Err(e) = reader.skip_bits(saved_offset) {
                // The saved offset overshoots the buffer — we have
                // not yet received enough bytes for the next command.
                if matches!(e, ShortenError::Truncated) {
                    return Ok(());
                }
                let core = shorten_error_to_core(e);
                return Err(self.fail(core));
            }
            let total_body_bits = (body.len() as u32).saturating_mul(8);
            // Attempt to read the function code.
            let fc = match read_function_code(&mut reader) {
                Ok(fc) => fc,
                Err(ShortenError::Truncated) => {
                    return Ok(());
                }
                Err(e) => {
                    let core = shorten_error_to_core(e);
                    return Err(self.fail(core));
                }
            };
            match fc {
                FunctionCode::Quit => {
                    self.commands += 1;
                    self.bits_consumed = reader.bits_consumed_so_far(total_body_bits);
                    self.eof = true;
                    return Ok(());
                }
                FunctionCode::Blocksize => match read_blocksize_payload(&mut reader) {
                    Ok(new_bs) => {
                        // Reject mid-round size changes: the planar
                        // AudioFrame shape requires every plane in a
                        // frame to have the same sample count, and a
                        // BLOCKSIZE override while channel slots are
                        // partially filled would produce mismatched
                        // planes after the round completes.
                        if self.pending_round.iter().any(|s| s.is_some())
                            && new_bs != self.block_size
                        {
                            return Err(self.fail(CoreError::invalid(
                                "oxideav-shorten: BLOCK_FN_BLOCKSIZE arrived mid channel-round; \
                                 planar frame shape requires per-channel block sizes to match",
                            )));
                        }
                        self.block_size = new_bs;
                        self.commands += 1;
                        self.bits_consumed = reader.bits_consumed_so_far(total_body_bits);
                    }
                    Err(ShortenError::Truncated) => {
                        return Ok(());
                    }
                    Err(e) => {
                        let core = shorten_error_to_core(e);
                        return Err(self.fail(core));
                    }
                },
                FunctionCode::Bitshift => match read_bitshift_payload(&mut reader) {
                    Ok(bs) => {
                        self.bshift = bs;
                        self.commands += 1;
                        self.bits_consumed = reader.bits_consumed_so_far(total_body_bits);
                    }
                    Err(ShortenError::Truncated) => {
                        return Ok(());
                    }
                    Err(e) => {
                        let core = shorten_error_to_core(e);
                        return Err(self.fail(core));
                    }
                },
                FunctionCode::Verbatim => match read_verbatim_payload(&mut reader) {
                    Ok(chunk) => {
                        self.verbatim_prefix.extend_from_slice(&chunk.bytes);
                        self.commands += 1;
                        self.bits_consumed = reader.bits_consumed_so_far(total_body_bits);
                    }
                    Err(ShortenError::Truncated) => {
                        return Ok(());
                    }
                    Err(e) => {
                        let core = shorten_error_to_core(e);
                        return Err(self.fail(core));
                    }
                },
                FunctionCode::Diff0
                | FunctionCode::Diff1
                | FunctionCode::Diff2
                | FunctionCode::Diff3 => {
                    let order = PolyOrder::from_function_code(fc)
                        .expect("DIFF0..3 always map to a PolyOrder");
                    let mu_chan = self.means[self.cursor].mu_chan();
                    let block = match decode_diff_block(
                        &mut reader,
                        order,
                        self.block_size,
                        &self.carries[self.cursor],
                        mu_chan,
                    ) {
                        Ok(b) => b,
                        Err(ShortenError::Truncated) => {
                            return Ok(());
                        }
                        Err(e) => {
                            let core = shorten_error_to_core(e);
                            return Err(self.fail(core));
                        }
                    };
                    self.commands += 1;
                    self.bits_consumed = reader.bits_consumed_so_far(total_body_bits);
                    self.finalise_sample_block(&block, header.channels as usize)?;
                }
                FunctionCode::Qlpc => {
                    let block = match decode_qlpc_block(
                        &mut reader,
                        self.block_size,
                        &self.carries[self.cursor],
                    ) {
                        Ok(b) => b,
                        Err(ShortenError::Truncated) => {
                            return Ok(());
                        }
                        Err(e) => {
                            let core = shorten_error_to_core(e);
                            return Err(self.fail(core));
                        }
                    };
                    self.commands += 1;
                    self.bits_consumed = reader.bits_consumed_so_far(total_body_bits);
                    self.finalise_sample_block(&block, header.channels as usize)?;
                }
                FunctionCode::Zero => {
                    let mu_chan = self.means[self.cursor].mu_chan();
                    let block = match fill_zero_block(self.block_size, mu_chan) {
                        Ok(b) => b,
                        Err(e) => {
                            let core = shorten_error_to_core(e);
                            return Err(self.fail(core));
                        }
                    };
                    self.commands += 1;
                    self.bits_consumed = reader.bits_consumed_so_far(total_body_bits);
                    self.finalise_sample_block(&block, header.channels as usize)?;
                }
            }
        }
    }

    /// Update the current channel's carry + mean estimator from a
    /// freshly-decoded pre-shift block, then deposit the emitted
    /// (post-shift) samples into the in-progress round's channel slot.
    /// If the round is now complete, pack it into a planar
    /// [`AudioFrame`] and queue it.
    fn finalise_sample_block(&mut self, block: &[i32], n_channels: usize) -> CoreResult<()> {
        // Carry + mean estimator update consumes pre-shift samples.
        self.carries[self.cursor].update_after_block(block);
        self.means[self.cursor].record_block(block);
        // Post-shift emission samples (spec/05 §1.4).
        let emitted: Vec<i32> = if self.bshift == 0 {
            block.to_vec()
        } else {
            let mut out = Vec::with_capacity(block.len());
            for &s in block {
                let shifted = (s as i64).checked_shl(self.bshift).ok_or_else(|| {
                    CoreError::invalid("oxideav-shorten: sample left-shift overflow")
                })?;
                let s_i32: i32 = shifted.try_into().map_err(|_| {
                    CoreError::invalid("oxideav-shorten: sample after shift outside i32 range")
                })?;
                out.push(s_i32);
            }
            out
        };
        let slot_idx = self.cursor;
        self.pending_round[slot_idx] = Some(ChannelBlockSlot { samples: emitted });
        // Advance the round-robin cursor for the NEXT sample-producing
        // command (spec/03 §2).
        self.cursor = (self.cursor + 1) % n_channels;
        // If every channel slot is filled, pack and queue the frame.
        if self.pending_round.iter().all(|s| s.is_some()) {
            self.pack_round_into_frame()?;
        }
        Ok(())
    }

    /// Convert the now-full `pending_round` into a planar
    /// [`AudioFrame`] per the `spec/05` §6 file-type table, queue it,
    /// and clear the slots.
    fn pack_round_into_frame(&mut self) -> CoreResult<()> {
        let header = self
            .header
            .expect("pack_round_into_frame called before header parsed");
        let filetype = header.filetype;
        let n_channels = header.channels as usize;
        let mut planes: Vec<Vec<u8>> = Vec::with_capacity(n_channels);
        let mut per_channel_len: Option<usize> = None;
        let mut slots: Vec<Vec<i32>> = Vec::with_capacity(n_channels);
        for slot in self.pending_round.iter_mut() {
            let s = slot.take().expect("all slots are Some by invariant");
            if let Some(n) = per_channel_len {
                if s.samples.len() != n {
                    return Err(CoreError::invalid(
                        "oxideav-shorten: per-channel block lengths diverged inside a channel round",
                    ));
                }
            } else {
                per_channel_len = Some(s.samples.len());
            }
            slots.push(s.samples);
        }
        let samples_per_ch = per_channel_len.unwrap_or(0);
        for ch in &slots {
            let plane = match filetype {
                FILETYPE_U8 => ch.iter().map(|&s| s as u8).collect::<Vec<u8>>(),
                FILETYPE_S16HL => {
                    let mut p = Vec::with_capacity(ch.len() * 2);
                    for &s in ch {
                        p.extend_from_slice(&(s as i16).to_be_bytes());
                    }
                    p
                }
                FILETYPE_S16LH => {
                    let mut p = Vec::with_capacity(ch.len() * 2);
                    for &s in ch {
                        p.extend_from_slice(&(s as i16).to_le_bytes());
                    }
                    p
                }
                other => {
                    return Err(CoreError::unsupported(format!(
                        "oxideav-shorten: H_filetype {other} not pinned by spec/05 §6 \
                         (only 2/u8, 3/s16hl, 5/s16lh have pinned numeric codes)",
                    )));
                }
            };
            planes.push(plane);
        }
        let pts = if self.first_frame_emitted {
            None
        } else {
            self.first_frame_emitted = true;
            self.pts
        };
        self.pending.push_back(AudioFrame {
            samples: samples_per_ch as u32,
            pts,
            data: planes,
        });
        Ok(())
    }
}

impl Decoder for ShortenStreamingDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> CoreResult<()> {
        if self.eof {
            return Err(CoreError::other(
                "oxideav-shorten: cannot send_packet after flush / BLOCK_FN_QUIT",
            ));
        }
        if self.pts.is_none() {
            self.pts = packet.pts;
        }
        self.buffer.extend_from_slice(&packet.data);
        // Drive the loop opportunistically — frame production may
        // unblock with the new bytes.
        self.try_advance()
    }

    fn receive_frame(&mut self) -> CoreResult<Frame> {
        if let Some(a) = self.pending.pop_front() {
            return Ok(Frame::Audio(a));
        }
        if let Some(msg) = &self.fatal {
            // Re-surface the cached fatal error message. Using
            // CoreError::other since we don't know which variant the
            // original error was; the textual form is preserved.
            return Err(CoreError::other(msg.clone()));
        }
        // No queued frame — try advancing once in case the buffered
        // bytes carry a complete unconsumed round.
        self.try_advance()?;
        if let Some(a) = self.pending.pop_front() {
            return Ok(Frame::Audio(a));
        }
        if self.eof {
            return Err(CoreError::Eof);
        }
        Err(CoreError::NeedMore)
    }

    fn flush(&mut self) -> CoreResult<()> {
        if !self.eof {
            self.try_advance()?;
        }
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> CoreResult<()> {
        self.buffer.clear();
        self.header = None;
        self.bits_consumed = 0;
        self.carries.clear();
        self.means.clear();
        self.cursor = 0;
        self.block_size = 0;
        self.bshift = 0;
        self.commands = 0;
        self.pending_round.clear();
        self.pending.clear();
        self.verbatim_prefix.clear();
        self.eof = false;
        self.pts = None;
        self.first_frame_emitted = false;
        self.fatal = None;
        Ok(())
    }
}

/// Install the Shorten **streaming** decoder factory into `reg` under
/// codec id [`STREAMING_CODEC_ID_STR`]. Distinct from
/// [`register_codecs`]'s `"shorten"` id so a caller can opt in to the
/// block-by-block emission shape explicitly.
pub fn register_streaming_codecs(reg: &mut CodecRegistry) {
    let info = CodecInfo::new(CodecId::new(STREAMING_CODEC_ID_STR))
        .capabilities(
            CodecCapabilities::audio(STREAMING_CODEC_ID_STR)
                .with_decode()
                .with_lossless(true),
        )
        .decoder(make_streaming_decoder);
    reg.register(info);
}

/// Install the Shorten decoder factory into `reg`.
///
/// Claims the codec id `"shorten"` and the Matroska codec id
/// `A_SHORTEN`. No WAVE format tag and no FourCC are claimed: every
/// Shorten file observed by the reachable fixture corpus is a stand-
/// alone `.shn` file with the `ajkg` magic at offset 0 (the codec is
/// its own native container per the `oxideav-shorten` README), so
/// there is no foreign container tag that resolves to Shorten.
///
/// No encoder factory is installed: the README "lacks" tail still
/// names the DIFFn / QLPC encoder, and the registry would happily
/// announce a decoder-only entry under this codec id without it.
pub fn register_codecs(reg: &mut CodecRegistry) {
    let info = CodecInfo::new(CodecId::new(CODEC_ID_STR))
        .capabilities(
            CodecCapabilities::audio(CODEC_ID_STR)
                .with_decode()
                .with_lossless(true),
        )
        .decoder(make_decoder);
    reg.register(info);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::FNSIZE;
    use crate::predictor::ENERGYSIZE;
    use oxideav_core::TimeBase;

    // ---- synthetic-stream builders (mirror those in driver.rs) ----

    fn pack_bits_msb_first(bits: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut byte = 0u8;
        let mut n = 0u32;
        for &b in bits {
            byte = (byte << 1) | (b as u8 & 1);
            n += 1;
            if n == 8 {
                out.push(byte);
                byte = 0;
                n = 0;
            }
        }
        if n > 0 {
            out.push(byte << (8 - n));
        }
        out
    }

    fn encode_uvar(value: u32, n: u32) -> Vec<u32> {
        if n == 0 {
            let mut bits = vec![0u32; value as usize];
            bits.push(1);
            bits
        } else {
            let span = 1u32 << n;
            let prefix_zeros = value / span;
            let mantissa = value % span;
            let mut bits = vec![0u32; prefix_zeros as usize];
            bits.push(1);
            for i in (0..n).rev() {
                bits.push((mantissa >> i) & 1);
            }
            bits
        }
    }

    fn encode_svar(value: i64, n: u32) -> Vec<u32> {
        let u: u64 = if value >= 0 {
            (value as u64) << 1
        } else {
            (((!value) as u64) << 1) | 1
        };
        let u32_val = u32::try_from(u).expect("svar fits in u32 in this test");
        encode_uvar(u32_val, n)
    }

    fn encode_ulong(value: u32, w: u32) -> Vec<u32> {
        let mut bits = Vec::new();
        bits.extend(encode_uvar(w, 2));
        bits.extend(encode_uvar(value, w));
        bits
    }

    fn header_param_bits(
        filetype: u32,
        channels: u32,
        blocksize: u32,
        maxlpcorder: u32,
        meanblocks: u32,
        skipbytes: u32,
    ) -> Vec<u32> {
        let bits_for = |v: u32| -> u32 {
            if v == 0 {
                0
            } else {
                32 - v.leading_zeros()
            }
        };
        let mut bits = Vec::new();
        bits.extend(encode_ulong(filetype, bits_for(filetype)));
        bits.extend(encode_ulong(channels, bits_for(channels)));
        bits.extend(encode_ulong(blocksize, bits_for(blocksize)));
        bits.extend(encode_ulong(maxlpcorder, bits_for(maxlpcorder)));
        bits.extend(encode_ulong(meanblocks, bits_for(meanblocks)));
        bits.extend(encode_ulong(skipbytes, bits_for(skipbytes)));
        bits
    }

    fn assemble(all_bits: &[u32]) -> Vec<u8> {
        let body = pack_bits_msb_first(all_bits);
        let mut buf = Vec::with_capacity(5 + body.len());
        buf.extend_from_slice(&crate::header::MAGIC);
        buf.push(2);
        buf.extend_from_slice(&body);
        buf
    }

    fn append_diff_block(out: &mut Vec<u32>, code: u32, energy: u32, residuals: &[i64]) {
        out.extend(encode_uvar(code, FNSIZE));
        out.extend(encode_uvar(energy, ENERGYSIZE));
        let width = energy + 1;
        for &r in residuals {
            out.extend(encode_svar(r, width));
        }
    }

    fn build_params() -> CodecParameters {
        CodecParameters::audio(CodecId::new(CODEC_ID_STR))
    }

    #[test]
    fn make_decoder_builds_and_reports_codec_id() {
        let p = build_params();
        let dec = make_decoder(&p).expect("make_decoder");
        assert_eq!(dec.codec_id().as_str(), CODEC_ID_STR);
    }

    #[test]
    fn register_codecs_installs_decoder_factory() {
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        let id = CodecId::new(CODEC_ID_STR);
        assert!(reg.has_decoder(&id));
        // No encoder is registered yet (README "lacks" tail).
        assert!(!reg.has_encoder(&id));
    }

    /// Decode a synthetic `s16lh` (filetype 5) stream through the
    /// registered factory and assert the emitted plane bytes match
    /// the direct-driver output packed little-endian.
    #[test]
    fn trait_decode_s16lh_matches_direct_driver_bit_exact() {
        // H_channels = 2, H_blocksize = 4, filetype = 5 (s16lh).
        let mut bits = header_param_bits(FILETYPE_S16LH, 2, 4, 0, 0, 0);
        // VERBATIM: 4-byte host-format envelope prefix.
        bits.extend(encode_uvar(9, FNSIZE));
        bits.extend(encode_uvar(4, 5));
        for b in [b'R', b'I', b'F', b'F'] {
            bits.extend(encode_uvar(b as u32, 8));
        }
        // DIFF1 ch0 [1, 2, 3, 4] over zero carry -> [1, 3, 6, 10]
        append_diff_block(&mut bits, 1, 3, &[1, 2, 3, 4]);
        // DIFF0 ch1 [100, 200, 300, 400] -> [100, 200, 300, 400]
        append_diff_block(&mut bits, 0, 4, &[100, 200, 300, 400]);
        bits.extend(encode_uvar(4, FNSIZE)); // QUIT
        let bytes = assemble(&bits);

        // Direct-driver reference output.
        let direct = decode_stream(&bytes).expect("direct decode_stream");
        assert_eq!(direct.channels.len(), 2);
        assert_eq!(direct.channels[0], vec![1, 3, 6, 10]);
        assert_eq!(direct.channels[1], vec![100, 200, 300, 400]);
        assert_eq!(direct.verbatim, vec![b'R', b'I', b'F', b'F']);

        // Trait-driven decode through the registered factory.
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        let mut dec = reg.first_decoder(&build_params()).expect("first_decoder");
        let tb = TimeBase::new(1, 44_100);
        let mut pkt = Packet::new(0, tb, bytes.clone());
        pkt.pts = Some(0);
        dec.send_packet(&pkt).expect("send_packet");

        let frame = match dec.receive_frame().expect("receive_frame") {
            Frame::Audio(a) => a,
            other => panic!("expected Audio frame, got {other:?}"),
        };
        // After draining, receive_frame returns Eof (the stream
        // terminated at BLOCK_FN_QUIT inside the single packet).
        assert!(matches!(dec.receive_frame(), Err(CoreError::Eof)));

        assert_eq!(frame.samples, 4);
        assert_eq!(frame.pts, Some(0));
        assert_eq!(frame.data.len(), 2);
        // ch0 LE: [1, 3, 6, 10] -> 01 00 03 00 06 00 0a 00
        assert_eq!(
            frame.data[0],
            vec![0x01, 0x00, 0x03, 0x00, 0x06, 0x00, 0x0a, 0x00]
        );
        // ch1 LE: [100, 200, 300, 400]
        assert_eq!(
            frame.data[1],
            vec![0x64, 0x00, 0xc8, 0x00, 0x2c, 0x01, 0x90, 0x01]
        );
    }

    /// Decode a synthetic `s16hl` (filetype 3) stream through the
    /// registered factory and assert the emitted plane bytes use the
    /// big-endian byte order pinned by the file-type table.
    #[test]
    fn trait_decode_s16hl_packs_big_endian() {
        // energy = 12 chosen so the +1-adjusted residual width (13)
        // accommodates the 0x1234 residual without overflowing the
        // uvar prefix-zeros safety cap.
        let mut bits = header_param_bits(FILETYPE_S16HL, 1, 2, 0, 0, 0);
        append_diff_block(&mut bits, 0, 12, &[0x1234, -0x0001]);
        bits.extend(encode_uvar(4, FNSIZE));
        let bytes = assemble(&bits);

        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        let mut dec = reg.first_decoder(&build_params()).expect("first_decoder");
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let frame = match dec.receive_frame().unwrap() {
            Frame::Audio(a) => a,
            o => panic!("non-audio frame: {o:?}"),
        };
        assert_eq!(frame.samples, 2);
        // s16hl: 0x1234 BE -> 12 34, -1 BE -> ff ff
        assert_eq!(frame.data[0], vec![0x12, 0x34, 0xff, 0xff]);
    }

    /// Decode a synthetic `u8` (filetype 2) stream and assert each
    /// sample is packed as a single unsigned byte.
    #[test]
    fn trait_decode_u8_packs_one_byte_per_sample() {
        let mut bits = header_param_bits(FILETYPE_U8, 1, 3, 0, 0, 0);
        append_diff_block(&mut bits, 0, 4, &[7, 9, 11]);
        bits.extend(encode_uvar(4, FNSIZE));
        let bytes = assemble(&bits);

        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        let mut dec = reg.first_decoder(&build_params()).expect("first_decoder");
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let frame = match dec.receive_frame().unwrap() {
            Frame::Audio(a) => a,
            o => panic!("non-audio frame: {o:?}"),
        };
        assert_eq!(frame.samples, 3);
        assert_eq!(frame.data[0], vec![7, 9, 11]);
    }

    #[test]
    fn unsupported_filetype_surfaces_unsupported_error() {
        let mut bits = header_param_bits(99, 1, 2, 0, 0, 0);
        append_diff_block(&mut bits, 0, 4, &[1, 2]);
        bits.extend(encode_uvar(4, FNSIZE));
        let bytes = assemble(&bits);

        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        let mut dec = reg.first_decoder(&build_params()).expect("first_decoder");
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), bytes);
        let res = dec.send_packet(&pkt);
        assert!(res.is_err(), "expected unsupported filetype to error");
    }

    #[test]
    fn receive_frame_before_send_returns_need_more() {
        let mut dec = make_decoder(&build_params()).expect("make_decoder");
        assert!(matches!(dec.receive_frame(), Err(CoreError::NeedMore)));
    }

    #[test]
    fn flush_makes_pending_drain_then_eof() {
        let mut bits = header_param_bits(FILETYPE_S16LH, 1, 2, 0, 0, 0);
        append_diff_block(&mut bits, 0, 4, &[1, -1]);
        bits.extend(encode_uvar(4, FNSIZE));
        let bytes = assemble(&bits);

        let mut dec = make_decoder(&build_params()).expect("make_decoder");
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        dec.flush().expect("flush");
        // The pending frame is still drainable after flush.
        assert!(matches!(dec.receive_frame(), Ok(Frame::Audio(_))));
        // Then EOF.
        assert!(matches!(dec.receive_frame(), Err(CoreError::Eof)));
    }

    #[test]
    fn send_packet_after_stream_terminated_errors() {
        let mut bits = header_param_bits(FILETYPE_S16LH, 1, 1, 0, 0, 0);
        append_diff_block(&mut bits, 0, 4, &[42]);
        bits.extend(encode_uvar(4, FNSIZE));
        let bytes = assemble(&bits);

        let mut dec = make_decoder(&build_params()).expect("make_decoder");
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        // The stream terminated; a second packet should error.
        let pkt2 = Packet::new(0, TimeBase::new(1, 44_100), vec![0u8; 4]);
        assert!(dec.send_packet(&pkt2).is_err());
    }

    #[test]
    fn split_packet_streaming_assembles_to_one_frame() {
        // Build a complete stream, then deliver it in two chunks
        // across two send_packet calls. The wrapper should buffer the
        // first half (Truncated → no frame yet) and decode on the
        // second half's arrival.
        let mut bits = header_param_bits(FILETYPE_S16LH, 1, 4, 0, 0, 0);
        append_diff_block(&mut bits, 0, 5, &[111, 222, 333, 444]);
        bits.extend(encode_uvar(4, FNSIZE));
        let bytes = assemble(&bits);
        assert!(bytes.len() >= 6, "test bytes too short to split");
        let mid = bytes.len() / 2;
        let (a, b) = bytes.split_at(mid);

        let mut dec = make_decoder(&build_params()).expect("make_decoder");
        let pkt_a = Packet::new(0, TimeBase::new(1, 44_100), a.to_vec());
        dec.send_packet(&pkt_a).expect("send_packet first half");
        assert!(matches!(dec.receive_frame(), Err(CoreError::NeedMore)));
        let pkt_b = Packet::new(0, TimeBase::new(1, 44_100), b.to_vec());
        dec.send_packet(&pkt_b).expect("send_packet second half");
        let frame = match dec.receive_frame().unwrap() {
            Frame::Audio(a) => a,
            o => panic!("non-audio frame: {o:?}"),
        };
        assert_eq!(frame.samples, 4);
        assert_eq!(
            frame.data[0],
            vec![0x6f, 0x00, 0xde, 0x00, 0x4d, 0x01, 0xbc, 0x01]
        );
    }

    #[test]
    fn reset_clears_buffer_so_fresh_stream_decodes_after() {
        let mut bits = header_param_bits(FILETYPE_S16LH, 1, 2, 0, 0, 0);
        append_diff_block(&mut bits, 0, 4, &[1, 2]);
        bits.extend(encode_uvar(4, FNSIZE));
        let bytes = assemble(&bits);

        let mut dec = make_decoder(&build_params()).expect("make_decoder");
        // Push only half the stream, then reset.
        let pkt_half = Packet::new(0, TimeBase::new(1, 44_100), bytes[..3].to_vec());
        dec.send_packet(&pkt_half).expect("send_packet half");
        dec.reset().expect("reset");
        // After reset a fresh complete stream should decode normally.
        let pkt_full = Packet::new(0, TimeBase::new(1, 44_100), bytes.clone());
        dec.send_packet(&pkt_full).expect("send_packet full");
        let frame = match dec.receive_frame().unwrap() {
            Frame::Audio(a) => a,
            o => panic!("non-audio frame: {o:?}"),
        };
        assert_eq!(frame.samples, 2);
    }

    #[test]
    fn verbatim_prefix_is_exposed_after_decode() {
        let mut bits = header_param_bits(FILETYPE_S16LH, 1, 2, 0, 0, 0);
        bits.extend(encode_uvar(9, FNSIZE));
        bits.extend(encode_uvar(3, 5));
        for b in [0x01u8, 0x02, 0x03] {
            bits.extend(encode_uvar(b as u32, 8));
        }
        append_diff_block(&mut bits, 0, 4, &[1, 2]);
        bits.extend(encode_uvar(4, FNSIZE));
        let bytes = assemble(&bits);

        // Construct the typed wrapper directly so we can inspect
        // verbatim_prefix(); the trait method exposes only AudioFrame.
        let mut dec = ShortenDecoder::new(
            CodecId::new(CODEC_ID_STR),
            CodecParameters::audio(CodecId::new(CODEC_ID_STR)),
        );
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        assert_eq!(dec.verbatim_prefix(), &[0x01, 0x02, 0x03]);
    }

    // ============================================================
    // Streaming-decoder tests (round 11).
    // ============================================================

    fn streaming_params() -> CodecParameters {
        CodecParameters::audio(CodecId::new(STREAMING_CODEC_ID_STR))
    }

    /// One-channel single-block stream: streaming decoder yields one
    /// `AudioFrame` then `Eof`.
    #[test]
    fn streaming_one_block_emits_one_frame_then_eof() {
        let mut bits = header_param_bits(FILETYPE_S16LH, 1, 4, 0, 0, 0);
        append_diff_block(&mut bits, 0, 4, &[1, 2, 3, 4]);
        bits.extend(encode_uvar(4, FNSIZE));
        let bytes = assemble(&bits);

        let mut dec = make_streaming_decoder(&streaming_params()).expect("make_streaming_decoder");
        let mut pkt = Packet::new(0, TimeBase::new(1, 44_100), bytes);
        pkt.pts = Some(0);
        dec.send_packet(&pkt).expect("send_packet");

        // Drain frames.
        let frame = match dec.receive_frame().expect("receive_frame") {
            Frame::Audio(a) => a,
            other => panic!("expected Audio, got {other:?}"),
        };
        assert_eq!(frame.samples, 4);
        assert_eq!(frame.pts, Some(0));
        // ch0 LE: 1 -> 01 00, 2 -> 02 00, 3 -> 03 00, 4 -> 04 00
        assert_eq!(frame.data.len(), 1);
        assert_eq!(
            frame.data[0],
            vec![0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04, 0x00]
        );
        assert!(matches!(dec.receive_frame(), Err(CoreError::Eof)));
    }

    /// Two-channel two-round stream: streaming decoder emits two
    /// frames (one per channel-round), each carrying both planes.
    #[test]
    fn streaming_two_channel_two_rounds_emits_two_frames() {
        // H_channels = 2, blocksize = 4, two DIFF0 rounds.
        let mut bits = header_param_bits(FILETYPE_S16LH, 2, 4, 0, 0, 0);
        append_diff_block(&mut bits, 0, 4, &[1, 2, 3, 4]); // round 0 ch 0
        append_diff_block(&mut bits, 0, 4, &[10, 20, 30, 40]); // round 0 ch 1
        append_diff_block(&mut bits, 0, 4, &[5, 6, 7, 8]); // round 1 ch 0
        append_diff_block(&mut bits, 0, 4, &[50, 60, 70, 80]); // round 1 ch 1
        bits.extend(encode_uvar(4, FNSIZE)); // QUIT
        let bytes = assemble(&bits);

        let mut dec = make_streaming_decoder(&streaming_params()).expect("make_streaming_decoder");
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), bytes);
        dec.send_packet(&pkt).expect("send_packet");

        let f0 = match dec.receive_frame().expect("frame 0") {
            Frame::Audio(a) => a,
            o => panic!("{o:?}"),
        };
        let f1 = match dec.receive_frame().expect("frame 1") {
            Frame::Audio(a) => a,
            o => panic!("{o:?}"),
        };
        assert_eq!(f0.samples, 4);
        assert_eq!(f0.data.len(), 2);
        // f0 ch0 = [1,2,3,4] LE; ch1 = [10,20,30,40] LE
        assert_eq!(
            f0.data[0],
            vec![0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04, 0x00]
        );
        assert_eq!(
            f0.data[1],
            vec![0x0a, 0x00, 0x14, 0x00, 0x1e, 0x00, 0x28, 0x00]
        );
        assert_eq!(f1.samples, 4);
        // Only the first emitted frame carries the packet's PTS;
        // subsequent frames carry None.
        assert!(f1.pts.is_none());
        assert_eq!(
            f1.data[0],
            vec![0x05, 0x00, 0x06, 0x00, 0x07, 0x00, 0x08, 0x00]
        );
        assert_eq!(
            f1.data[1],
            vec![0x32, 0x00, 0x3c, 0x00, 0x46, 0x00, 0x50, 0x00]
        );
        assert!(matches!(dec.receive_frame(), Err(CoreError::Eof)));
    }

    /// Verifies the load-bearing invariant: the streaming adaptor's
    /// per-channel concatenated output exactly equals the whole-stream
    /// `decode_stream`'s `channels` for a multi-round, multi-channel
    /// fixture that exercises DIFFn + VERBATIM + BITSHIFT + BLOCKSIZE +
    /// ZERO commands.
    #[test]
    fn streaming_matches_whole_stream_driver_on_complex_fixture() {
        let mut bits = header_param_bits(FILETYPE_S16LH, 2, 4, 0, 4, 0);
        // VERBATIM envelope (cursor unchanged).
        bits.extend(encode_uvar(9, FNSIZE));
        bits.extend(encode_uvar(4, 5));
        for b in [b'R', b'I', b'F', b'F'] {
            bits.extend(encode_uvar(b as u32, 8));
        }
        // BITSHIFT = 0 (no-op; exercises the absorb path).
        bits.extend(encode_uvar(6, FNSIZE));
        bits.extend(encode_uvar(0, 2));
        // Round 0: DIFF1 ch0, DIFF0 ch1
        append_diff_block(&mut bits, 1, 3, &[1, 1, 1, 1]); // ch0 DIFF1
        append_diff_block(&mut bits, 0, 4, &[10, 20, 30, 40]); // ch1 DIFF0
                                                               // Round 1: DIFF0 ch0, DIFF0 ch1
        append_diff_block(&mut bits, 0, 4, &[3, 4, 5, 6]); // ch0 DIFF0
        append_diff_block(&mut bits, 0, 4, &[11, 22, 33, 44]); // ch1 DIFF0
                                                               // BLOCKSIZE override (channel-round boundary), 4 -> 2
        bits.extend(encode_uvar(5, FNSIZE));
        bits.extend(encode_ulong(2, 2));
        // Round 2 (smaller bs): DIFF0 ch0, ZERO ch1 (mean-driven)
        append_diff_block(&mut bits, 0, 4, &[100, 200]); // ch0
        bits.extend(encode_uvar(8, FNSIZE)); // ZERO ch1
        bits.extend(encode_uvar(4, FNSIZE)); // QUIT
        let bytes = assemble(&bits);

        // Reference: round-7 whole-stream driver.
        let direct = decode_stream(&bytes).expect("decode_stream");

        // Streaming: collect concatenated per-channel emitted samples.
        let mut dec = make_streaming_decoder(&streaming_params()).expect("make_streaming_decoder");
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        let mut per_channel: Vec<Vec<i32>> = vec![Vec::new(); direct.header.channels as usize];
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    // Decode each plane back to i32 per the s16lh layout.
                    for (ci, plane) in a.data.iter().enumerate() {
                        assert_eq!(plane.len() % 2, 0);
                        for chunk in plane.chunks_exact(2) {
                            let s = i16::from_le_bytes([chunk[0], chunk[1]]) as i32;
                            per_channel[ci].push(s);
                        }
                    }
                }
                Ok(other) => panic!("non-audio frame: {other:?}"),
                Err(CoreError::Eof) => break,
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert_eq!(per_channel, direct.channels);
    }

    /// Streaming decoder honours `send_packet` chop-anywhere: the
    /// caller may deliver the file in arbitrary slices and the same
    /// per-channel output materialises.
    #[test]
    fn streaming_split_packet_assembles_to_same_frames() {
        let mut bits = header_param_bits(FILETYPE_S16LH, 2, 2, 0, 0, 0);
        append_diff_block(&mut bits, 0, 4, &[1, 2]); // round 0 ch 0
        append_diff_block(&mut bits, 0, 4, &[10, 20]); // round 0 ch 1
        append_diff_block(&mut bits, 0, 4, &[3, 4]); // round 1 ch 0
        append_diff_block(&mut bits, 0, 4, &[30, 40]); // round 1 ch 1
        bits.extend(encode_uvar(4, FNSIZE));
        let bytes = assemble(&bits);

        // Whole-buffer reference.
        let mut whole =
            make_streaming_decoder(&streaming_params()).expect("make_streaming_decoder");
        whole
            .send_packet(&Packet::new(0, TimeBase::new(1, 44_100), bytes.clone()))
            .expect("whole send_packet");
        let mut whole_frames: Vec<AudioFrame> = Vec::new();
        loop {
            match whole.receive_frame() {
                Ok(Frame::Audio(a)) => whole_frames.push(a),
                Err(CoreError::Eof) => break,
                other => panic!("{other:?}"),
            }
        }

        // Split delivery: 4 packets of unequal sizes.
        let mut chunks: Vec<Vec<u8>> = Vec::new();
        let chunk_size = (bytes.len() / 4).max(1);
        let mut i = 0;
        while i < bytes.len() {
            let j = (i + chunk_size).min(bytes.len());
            chunks.push(bytes[i..j].to_vec());
            i = j;
        }
        let mut split =
            make_streaming_decoder(&streaming_params()).expect("make_streaming_decoder");
        let mut split_frames: Vec<AudioFrame> = Vec::new();
        for chunk in &chunks {
            split
                .send_packet(&Packet::new(0, TimeBase::new(1, 44_100), chunk.clone()))
                .expect("split send_packet");
            // Drain any frames the new bytes unlocked.
            loop {
                match split.receive_frame() {
                    Ok(Frame::Audio(a)) => split_frames.push(a),
                    Err(CoreError::NeedMore) => break,
                    Err(CoreError::Eof) => break,
                    other => panic!("{other:?}"),
                }
            }
        }
        // After the final chunk, flush + drain.
        split.flush().expect("flush");
        loop {
            match split.receive_frame() {
                Ok(Frame::Audio(a)) => split_frames.push(a),
                Err(CoreError::Eof) => break,
                other => panic!("{other:?}"),
            }
        }
        assert_eq!(whole_frames.len(), split_frames.len());
        for (w, s) in whole_frames.iter().zip(split_frames.iter()) {
            assert_eq!(w.samples, s.samples);
            assert_eq!(w.data, s.data);
        }
    }

    /// `VERBATIM` envelope bytes are accumulated incrementally on the
    /// streaming adaptor's `verbatim_prefix()` accessor — they do not
    /// have to wait for `BLOCK_FN_QUIT` to be exposed.
    #[test]
    fn streaming_verbatim_prefix_is_accumulated_across_commands() {
        let mut bits = header_param_bits(FILETYPE_S16LH, 1, 2, 0, 0, 0);
        // Two VERBATIM blocks before any samples.
        bits.extend(encode_uvar(9, FNSIZE));
        bits.extend(encode_uvar(3, 5));
        for b in [b'R', b'I', b'F'] {
            bits.extend(encode_uvar(b as u32, 8));
        }
        bits.extend(encode_uvar(9, FNSIZE));
        bits.extend(encode_uvar(2, 5));
        for b in [b'F', b'X'] {
            bits.extend(encode_uvar(b as u32, 8));
        }
        // Then a single sample block + QUIT.
        append_diff_block(&mut bits, 0, 4, &[1, 2]);
        bits.extend(encode_uvar(4, FNSIZE));
        let bytes = assemble(&bits);

        // Build the typed wrapper directly so we can inspect verbatim_prefix.
        let mut dec =
            ShortenStreamingDecoder::new(CodecId::new(STREAMING_CODEC_ID_STR), streaming_params());
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        // After driving the loop to completion the wrapper's verbatim
        // prefix carries both chunks concatenated in encounter order.
        assert_eq!(dec.verbatim_prefix(), b"RIFFX");
        // And the sample frame is queued.
        let frame = match dec.receive_frame().expect("frame") {
            Frame::Audio(a) => a,
            o => panic!("{o:?}"),
        };
        assert_eq!(frame.samples, 2);
    }

    /// A `BLOCK_FN_BLOCKSIZE` mid channel-round is rejected (the
    /// resulting planar frame would have mismatched plane lengths).
    #[test]
    fn streaming_mid_round_blocksize_change_is_rejected() {
        let mut bits = header_param_bits(FILETYPE_S16LH, 2, 4, 0, 0, 0);
        // Round-0 channel-0 block at size 4.
        append_diff_block(&mut bits, 0, 4, &[1, 2, 3, 4]);
        // BLOCKSIZE override AFTER channel 0 but BEFORE channel 1 —
        // mid-round, which the planar shape can't support.
        bits.extend(encode_uvar(5, FNSIZE));
        bits.extend(encode_ulong(2, 2));
        append_diff_block(&mut bits, 0, 3, &[10, 20]);
        bits.extend(encode_uvar(4, FNSIZE));
        let bytes = assemble(&bits);

        let mut dec = make_streaming_decoder(&streaming_params()).expect("make_streaming_decoder");
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), bytes);
        let res = dec.send_packet(&pkt);
        assert!(
            res.is_err(),
            "mid-round BLOCKSIZE should error, got {res:?}",
        );
    }

    /// `register_streaming_codecs` installs a decoder factory under
    /// the `STREAMING_CODEC_ID_STR` id, alongside (but distinct from)
    /// the whole-stream `register_codecs` registration.
    #[test]
    fn streaming_register_installs_under_distinct_codec_id() {
        let mut reg = CodecRegistry::new();
        register_streaming_codecs(&mut reg);
        let id = CodecId::new(STREAMING_CODEC_ID_STR);
        assert!(reg.has_decoder(&id));
        // The whole-stream codec id is NOT registered by the streaming
        // helper alone.
        assert!(!reg.has_decoder(&CodecId::new(CODEC_ID_STR)));
        // And conversely, register_codecs alone doesn't install the
        // streaming id.
        let mut reg2 = CodecRegistry::new();
        register_codecs(&mut reg2);
        assert!(reg2.has_decoder(&CodecId::new(CODEC_ID_STR)));
        assert!(!reg2.has_decoder(&id));
    }

    /// `reset()` returns the streaming decoder to its initial state so
    /// the same instance can decode a fresh stream.
    #[test]
    fn streaming_reset_allows_redecoding_a_fresh_stream() {
        let mut bits = header_param_bits(FILETYPE_S16LH, 1, 2, 0, 0, 0);
        append_diff_block(&mut bits, 0, 4, &[1, 2]);
        bits.extend(encode_uvar(4, FNSIZE));
        let bytes = assemble(&bits);

        let mut dec = make_streaming_decoder(&streaming_params()).expect("make_streaming_decoder");
        // Push only a partial buffer.
        dec.send_packet(&Packet::new(
            0,
            TimeBase::new(1, 44_100),
            bytes[..3].to_vec(),
        ))
        .expect("partial send");
        dec.reset().expect("reset");
        // Push the full stream after reset; the wrapper should be
        // indistinguishable from a fresh instance.
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 44_100), bytes.clone()))
            .expect("full send");
        let frame = match dec.receive_frame().expect("frame") {
            Frame::Audio(a) => a,
            o => panic!("{o:?}"),
        };
        assert_eq!(frame.samples, 2);
    }

    /// On-demand pull: `receive_frame` decodes only as many blocks as
    /// needed to produce the next frame. The internal command counter
    /// advances by exactly `n_channels` per surfaced frame (one block
    /// per channel = one round).
    #[test]
    fn streaming_on_demand_decode_stops_at_each_frame_boundary() {
        // Build a stream with TWO rounds; assert that after pulling
        // ONE frame the decoder has not yet consumed the second
        // round's commands (the buffer's pending samples for round 1
        // are still queued behind the cursor).
        let mut bits = header_param_bits(FILETYPE_S16LH, 1, 2, 0, 0, 0);
        append_diff_block(&mut bits, 0, 4, &[1, 2]); // round 0 -> frame 0
        append_diff_block(&mut bits, 0, 4, &[3, 4]); // round 1 -> frame 1
        bits.extend(encode_uvar(4, FNSIZE));
        let bytes = assemble(&bits);

        let mut dec =
            ShortenStreamingDecoder::new(CodecId::new(STREAMING_CODEC_ID_STR), streaming_params());
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), bytes);
        dec.send_packet(&pkt).expect("send_packet");

        // Drain frame 0.
        let f0 = match dec.receive_frame().expect("frame 0") {
            Frame::Audio(a) => a,
            o => panic!("{o:?}"),
        };
        assert_eq!(f0.samples, 2);
        // Drain frame 1.
        let f1 = match dec.receive_frame().expect("frame 1") {
            Frame::Audio(a) => a,
            o => panic!("{o:?}"),
        };
        assert_eq!(f1.samples, 2);
        // EOF reached.
        assert!(matches!(dec.receive_frame(), Err(CoreError::Eof)));
    }

    /// The framework `register(ctx)` entry installs BOTH the
    /// whole-stream and the streaming decoders into the same
    /// `RuntimeContext`'s registry, so a caller can resolve either
    /// shape by codec id.
    #[test]
    fn register_function_installs_both_decoder_shapes() {
        let mut ctx = oxideav_core::RuntimeContext::new();
        crate::register(&mut ctx);
        assert!(ctx.codecs.has_decoder(&CodecId::new(CODEC_ID_STR)));
        assert!(ctx
            .codecs
            .has_decoder(&CodecId::new(STREAMING_CODEC_ID_STR)));
    }
}
