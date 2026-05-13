//! `oxideav-core` framework integration.
//!
//! Compiled only when the default-on `registry` Cargo feature is
//! enabled. Standalone consumers (`default-features = false`) do not
//! pull in `oxideav-core` and skip this module entirely.

#![cfg(feature = "registry")]

use std::io::{Read, SeekFrom};

use oxideav_core::{
    AudioFrame, CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry,
    CodecResolver, ContainerRegistry, Decoder as CoreDecoder, Demuxer, Error as CoreError, Frame,
    MediaType, Packet, ProbeData, ReadSeek, Result as CoreResult, RuntimeContext, SampleFormat,
    StreamInfo, TimeBase,
};

use crate::header::Filetype;

/// Canonical codec id string registered with `oxideav-core`. The
/// pipeline looks the codec up by this string when constructing a
/// decoder for a `.shn` stream.
pub const CODEC_ID_STR: &str = "shorten";

/// Container-format name registered with `oxideav-core`'s
/// `ContainerRegistry` under [`shorten_probe`].
pub const CONTAINER_NAME: &str = "shorten";

/// Register the Shorten decoder with `reg`.
///
/// The Shorten format is signalled by:
///
/// * the four-byte magic `ajkg` at file offset 0 (see
///   [`shorten_probe`] for the probe registered with the
///   container registry);
/// * the FourCC `shrn` (used by some legacy WAVE wrappers) — this
///   crate registers it for forward-compat with future container work.
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("shorten_sw")
        .with_decode()
        .with_encode()
        .with_lossless(true)
        .with_intra_only(true)
        .with_max_channels(8)
        .with_max_sample_rate(192_000);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder),
    );
}

/// Register the `.shn` container demuxer + `ajkg` probe with `reg`.
///
/// A `.shn` file has no internal framing — the entire byte buffer is a
/// single Shorten stream. The demuxer therefore reads the input to end-
/// of-stream and emits the buffer as a single packet on stream 0; the
/// codec-side decoder unpacks the bit stream and yields per-block
/// audio frames.
pub fn container_register(reg: &mut ContainerRegistry) {
    reg.register_demuxer(CONTAINER_NAME, open_demuxer);
    reg.register_extension("shn", CONTAINER_NAME);
    reg.register_probe(CONTAINER_NAME, shorten_probe);
}

/// Container probe — returns 100 for byte streams whose first four
/// bytes are the `ajkg` magic, 0 otherwise.
pub fn shorten_probe(p: &ProbeData) -> u8 {
    if p.buf.len() < 5 {
        return 0;
    }
    if p.buf[0..4] == crate::MAGIC[..] {
        // Sanity-check the version byte while we're here so a buffer
        // that happens to start `ajkg` but isn't actually Shorten
        // doesn't get misidentified.
        match p.buf[4] {
            1..=3 => 100,
            _ => 0,
        }
    } else {
        0
    }
}

/// Unified entry point invoked by the `oxideav_core::register!`
/// macro-generated `__oxideav_entry`. Installs both the codec and
/// container registrations.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
    container_register(&mut ctx.containers);
}

fn make_decoder(params: &CodecParameters) -> CoreResult<Box<dyn CoreDecoder>> {
    let output_format = params
        .sample_format
        .ok_or_else(|| CoreError::invalid("oxideav-shorten: sample_format missing on stream"))?;
    let channels = params
        .channels
        .ok_or_else(|| CoreError::invalid("oxideav-shorten: channels missing on stream"))?;
    Ok(Box::new(ShortenDecoder {
        codec_id: params.codec_id.clone(),
        output_format,
        channels,
        pending: None,
        eof: false,
    }))
}

struct ShortenDecoder {
    codec_id: CodecId,
    output_format: SampleFormat,
    channels: u16,
    pending: Option<Packet>,
    eof: bool,
}

impl CoreDecoder for ShortenDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> CoreResult<()> {
        if self.pending.is_some() {
            return Err(CoreError::other(
                "oxideav-shorten: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> CoreResult<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(CoreError::Eof)
            } else {
                Err(CoreError::NeedMore)
            };
        };
        let stream = crate::decode(&pkt.data)
            .map_err(|e| CoreError::invalid(format!("oxideav-shorten: {e}")))?;
        if stream.header.channels != self.channels {
            return Err(CoreError::invalid(format!(
                "oxideav-shorten: stream has {} channels but decoder configured for {}",
                stream.header.channels, self.channels
            )));
        }
        let bytes = pcm_pack(&stream, self.output_format)?;
        Ok(Frame::Audio(AudioFrame {
            samples: stream.samples_per_channel as u32,
            pts: pkt.pts,
            data: vec![bytes],
        }))
    }

    fn flush(&mut self) -> CoreResult<()> {
        self.eof = true;
        Ok(())
    }
}

// ───────────────────────── Demuxer ─────────────────────────

/// Test-only typed factory. Constructs a [`ShortenDemuxer`] directly
/// over the in-memory byte buffer — the same path
/// [`open_demuxer`] takes, but returning the concrete type so the
/// integration test in `tests/seek.rs` can inspect the cached frame
/// index and the scan counter. Hidden from public Rustdoc.
#[doc(hidden)]
pub fn __open_demuxer_typed(buf: Vec<u8>) -> CoreResult<ShortenDemuxer> {
    if buf.len() < 5 || buf[0..4] != crate::MAGIC[..] {
        return Err(CoreError::invalid(
            "oxideav-shorten: input is not a Shorten stream (missing 'ajkg' magic)",
        ));
    }
    let header = crate::parse_header(&buf)
        .map_err(|e| CoreError::invalid(format!("oxideav-shorten: {e}")))?;
    let sample_fmt = match header.filetype {
        Filetype::U8 => SampleFormat::U8,
        Filetype::S8 => SampleFormat::S8,
        Filetype::S16Be | Filetype::S16Le | Filetype::S16Native | Filetype::S16Swapped => {
            SampleFormat::S16
        }
        Filetype::U16Be | Filetype::U16Le | Filetype::U16Native | Filetype::U16Swapped => {
            SampleFormat::S16
        }
        Filetype::Ulaw => SampleFormat::U8,
    };
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Audio;
    params.channels = Some(header.channels);
    params.sample_format = Some(sample_fmt);
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 44_100),
        duration: None,
        start_time: None,
        params,
    };
    Ok(ShortenDemuxer {
        streams: vec![stream],
        payload: buf,
        emitted: false,
        next_pts: 0,
        frame_index: None,
        scan_count: 0,
    })
}

fn open_demuxer(
    mut input: Box<dyn ReadSeek>,
    _codecs: &dyn CodecResolver,
) -> CoreResult<Box<dyn Demuxer>> {
    input.seek(SeekFrom::Start(0))?;
    let mut buf = Vec::new();
    input.read_to_end(&mut buf)?;
    if buf.len() < 5 || buf[0..4] != crate::MAGIC[..] {
        return Err(CoreError::invalid(
            "oxideav-shorten: input is not a Shorten stream (missing 'ajkg' magic)",
        ));
    }

    // Pre-parse the header so we can declare a meaningful StreamInfo
    // up-front — the pipeline needs `sample_format` + `channels` to
    // select a decoder factory.
    let header = crate::parse_header(&buf)
        .map_err(|e| CoreError::invalid(format!("oxideav-shorten: {e}")))?;

    let sample_fmt = match header.filetype {
        Filetype::U8 => SampleFormat::U8,
        Filetype::S8 => SampleFormat::S8,
        Filetype::S16Be | Filetype::S16Le | Filetype::S16Native | Filetype::S16Swapped => {
            SampleFormat::S16
        }
        // For unsigned-16 and µ-law the Shorten stream stores samples
        // in a signed predictor lane. We surface S16 to the consumer
        // and let the codec-side packer expand to the host PCM layout
        // appropriate for the declared output filetype.
        Filetype::U16Be | Filetype::U16Le | Filetype::U16Native | Filetype::U16Swapped => {
            SampleFormat::S16
        }
        Filetype::Ulaw => SampleFormat::U8,
    };

    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.media_type = MediaType::Audio;
    params.channels = Some(header.channels);
    params.sample_format = Some(sample_fmt);
    // We don't know the sample rate from the Shorten stream itself —
    // it is conveyed by the wrapping container (RIFF/WAVE in
    // `verbatim_prefix`). Leave `sample_rate = None`; the consumer
    // can parse the verbatim prefix or default to 44100.

    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 44_100),
        duration: None,
        start_time: None,
        params,
    };

    Ok(Box::new(ShortenDemuxer {
        streams: vec![stream],
        payload: buf,
        emitted: false,
        next_pts: 0,
        frame_index: None,
        scan_count: 0,
    }))
}

/// Granularity of the lazy frame index. We sample one entry every
/// [`FRAME_INDEX_STRIDE`] per-channel block-rounds; smaller values give
/// finer seek precision at the cost of a longer index. `10` is a
/// pragmatic compromise: on TR.156's default 256-sample block the
/// index lands every 2 560 per-channel samples (~58 ms at 44 100 Hz),
/// well below the audible threshold for a missed seek target while
/// keeping a 5-minute mono file's index under ~1 KB.
const FRAME_INDEX_STRIDE: u64 = 10;

/// Cached entry in the lazy frame index. `pts` is the per-channel
/// sample index at which the indexed block starts (matches the
/// container time-base); `byte_offset` is the file-relative byte at
/// which the block stream **begins** for that pts (not used for
/// payload slicing in round-8 — see the seek semantics note in the
/// `seek_to` doc-comment, which slices on `pts` alone and accepts a
/// transient predictor-state glitch).
#[derive(Debug, Clone, Copy)]
struct FrameIndexEntry {
    pts: i64,
    /// Informational — file-relative byte at which the indexed
    /// block-round begins. Not consumed by [`ShortenDemuxer::seek_to`]
    /// in round 8 (the payload is sliced on pts alone, see the
    /// `seek_to` doc-comment) but recorded so a future round can
    /// switch to byte-aware slicing once predictor-state replay is
    /// supported. Surfaced through the test-only
    /// [`ShortenDemuxer::frame_index_byte_offsets`] helper.
    #[cfg_attr(not(test), allow(dead_code))]
    byte_offset: u64,
}

/// Concrete Shorten demuxer. Exposed publicly (with `#[doc(hidden)]`)
/// so the seek integration test in `tests/seek.rs` can inspect the
/// scan counter and the cached frame index — see
/// [`__open_demuxer_typed`]. Framework consumers should always go
/// through the [`Demuxer`] trait via the registered container.
#[doc(hidden)]
pub struct ShortenDemuxer {
    streams: Vec<StreamInfo>,
    payload: Vec<u8>,
    emitted: bool,
    /// Per-channel sample index at which the next emitted packet's
    /// pts/dts is anchored. Default 0; mutated by [`seek_to`].
    next_pts: i64,
    /// Lazy frame index — built on first [`seek_to`]. `None` until
    /// then; [`Some`] thereafter (even if empty for a degenerate
    /// stream).
    frame_index: Option<Vec<FrameIndexEntry>>,
    /// Cheap diagnostic counter — incremented each time the demuxer
    /// scans the FN command stream. Tests use this to assert that the
    /// frame index is cached across multiple [`seek_to`] calls (a
    /// scanned-once invariant means the counter stays at 1 across the
    /// 2nd and Nth seek). Not exposed through the public Demuxer
    /// trait; accessed from the test module through an internal
    /// helper [`scan_count`].
    scan_count: u32,
}

impl ShortenDemuxer {
    /// Diagnostic — number of times the FN command scan has run.
    /// Used by the seek tests to assert the index is cached across
    /// calls. `#[doc(hidden)]` since framework consumers shouldn't
    /// reach for it.
    #[doc(hidden)]
    pub fn scan_count(&self) -> u32 {
        self.scan_count
    }

    /// Diagnostic — file-relative byte offsets recorded in the lazy
    /// frame index, in the order the scan emitted them. Returns
    /// `None` if [`Self::seek_to`] has not yet built the index.
    /// `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn frame_index_byte_offsets(&self) -> Option<Vec<u64>> {
        self.frame_index
            .as_ref()
            .map(|v| v.iter().map(|e| e.byte_offset).collect())
    }

    /// Diagnostic — pts list from the lazy frame index. `None` if
    /// [`Self::seek_to`] has not yet been called.
    #[doc(hidden)]
    pub fn frame_index_pts(&self) -> Option<Vec<i64>> {
        self.frame_index
            .as_ref()
            .map(|v| v.iter().map(|e| e.pts).collect())
    }

    /// Current `next_pts` cursor — pts at which the next emitted
    /// packet will be anchored. `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn next_pts(&self) -> i64 {
        self.next_pts
    }

    /// Build or return the frame index. Memoised across calls.
    fn ensure_index(&mut self) -> CoreResult<&[FrameIndexEntry]> {
        if self.frame_index.is_none() {
            let entries = build_frame_index(&self.payload, FRAME_INDEX_STRIDE)
                .map_err(|e| CoreError::invalid(format!("oxideav-shorten: {e}")))?;
            self.scan_count = self.scan_count.saturating_add(1);
            self.frame_index = Some(entries);
        }
        Ok(self
            .frame_index
            .as_ref()
            .expect("frame_index just set")
            .as_slice())
    }
}

impl Demuxer for ShortenDemuxer {
    fn format_name(&self) -> &str {
        CONTAINER_NAME
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> CoreResult<Packet> {
        if self.emitted {
            return Err(CoreError::Eof);
        }
        self.emitted = true;
        let mut pkt = Packet::new(0, self.streams[0].time_base, self.payload.clone());
        pkt.pts = Some(self.next_pts);
        pkt.dts = Some(self.next_pts);
        pkt.flags.keyframe = true;
        Ok(pkt)
    }

    /// Seek to the nearest indexed block at or before `pts`.
    ///
    /// **Seek semantics — round 8.** Shorten has no built-in seek
    /// table: per-block predictor + carry state accumulates from byte
    /// 0 of the stream, so a clean seek requires either replaying the
    /// stream from the start (no gain) or accepting that the first
    /// block or two after the seek point will decode with a stale
    /// predictor state (one to two blocks of audible glitch on
    /// lossless content, less on lossy / smooth audio).
    ///
    /// This implementation:
    ///
    /// 1. Lazily builds a `Vec<FrameIndexEntry>` on the first call by
    ///    walking the FN command stream and recording
    ///    `(per-channel-sample-pts, byte_offset)` every
    ///    [`FRAME_INDEX_STRIDE`] per-channel block-rounds. The index
    ///    is cached on the demuxer; subsequent calls are O(log n)
    ///    over the index plus the `next_pts` write.
    /// 2. Binary-searches the index for the largest entry whose pts
    ///    is `<= target_pts`. Below the first entry it clamps to 0;
    ///    past the last entry it clamps to the last indexed pts.
    /// 3. Sets [`Self::next_pts`] to the chosen entry's pts and
    ///    arms [`Self::next_packet`] to re-emit the full payload at
    ///    that pts. The consumer is expected to drop frames whose
    ///    timestamps fall below `next_pts`, taking the
    ///    predictor-state-glitch tail as documented in this crate's
    ///    README.
    ///
    /// Returns the actual pts seeked to (which may differ from
    /// `pts` by up to one indexed-stride's worth of samples).
    fn seek_to(&mut self, stream_index: u32, pts: i64) -> CoreResult<i64> {
        if stream_index != 0 {
            return Err(CoreError::invalid(format!(
                "oxideav-shorten: stream index {stream_index} out of range (only stream 0 exists)"
            )));
        }
        let target = pts.max(0);

        // Build the index lazily.
        let entries = self.ensure_index()?;

        // Determine the landing pts. If the index is empty (degenerate
        // single-block stream) we always land at 0. Otherwise binary-
        // search for the largest entry with pts <= target.
        let landed = if entries.is_empty() {
            0
        } else {
            // `partition_point` returns the first index whose
            // predicate is false; the candidate is the entry just
            // before it.
            let idx = entries.partition_point(|e| e.pts <= target);
            if idx == 0 {
                // Target before the first indexed pts — land at 0.
                0
            } else {
                entries[idx - 1].pts
            }
        };

        self.next_pts = landed;
        self.emitted = false;
        Ok(landed)
    }
}

// ───────────────────────── Frame-index scan ─────────────────────────

/// Walk the FN command stream from byte 5 (immediately past magic +
/// version) through `BLOCK_FN_QUIT`, recording one
/// [`FrameIndexEntry`] every `stride` per-channel block-rounds.
///
/// A "per-channel block-round" is `n_channels` consecutive
/// sample-producing blocks (`DIFF0..3` / `QLPC` / `ZERO`) — exactly
/// one block per channel under the encoder's round-robin commit
/// (see [`crate::encoder::encode`]). Each round advances the
/// per-channel sample-pts cursor by the **block size in effect at
/// round start** (which a `BLOCK_FN_BLOCKSIZE` command may have
/// changed between rounds).
///
/// The byte offset recorded for each entry is the file-relative byte
/// containing the bit at which the indexed block-round begins (i.e.
/// `5 + bit_pos / 8` where `bit_pos` is the BitReader's cursor at
/// round start). This is informational only in round 8 — the seek
/// path does not slice on it, since the per-block predictor state
/// cannot be reconstructed from a mid-stream cut.
fn build_frame_index(file_bytes: &[u8], stride: u64) -> crate::Result<Vec<FrameIndexEntry>> {
    use crate::bitreader::BitReader;
    use crate::decoder::fn_code;
    use crate::varint::{
        read_svar, read_ulong, read_uvar, BITSHIFTSIZE, ENERGYSIZE, FNSIZE, LPCQSIZE, LPCQUANT,
        RESIDUAL_WIDTH_CAP, VERBATIM_BYTE_SIZE, VERBATIM_CHUNK_SIZE,
    };

    let header = crate::parse_header(file_bytes)?;
    let nch = header.channels as usize;
    if nch == 0 {
        return Ok(Vec::new());
    }

    let mut br = BitReader::new(&file_bytes[5..]);
    // Advance past the parameter block.
    while br.bit_pos() < header.header_end_bit {
        let remaining = header.header_end_bit - br.bit_pos();
        let chunk = remaining.min(32) as u32;
        let _ = br.read_bits(chunk)?;
    }

    let mut entries: Vec<FrameIndexEntry> = Vec::new();
    // Always include the start of the block stream as the first
    // indexed entry — seek-to-zero relies on this anchor.
    entries.push(FrameIndexEntry {
        pts: 0,
        byte_offset: 5 + (br.bit_pos() as u64 / 8),
    });

    let mut current_block_size: u32 = header.blocksize;
    let mut channel_cursor: usize = 0;
    // Per-channel sample-pts (cumulative). All channels advance in
    // lockstep — channel 0's count is the canonical pts.
    let mut samples_per_channel: i64 = 0;
    let mut rounds_since_index: u64 = 0;

    loop {
        let fn_code = read_uvar(&mut br, FNSIZE)?;
        match fn_code {
            fn_code::DIFF0 | fn_code::DIFF1 | fn_code::DIFF2 | fn_code::DIFF3 => {
                // Skip the residual payload of one DIFF block: an
                // ENERGYSIZE-width energy field then `bs`
                // svar(width)-encoded residuals.
                let energy = read_uvar(&mut br, ENERGYSIZE)?;
                let width = energy.checked_add(1).unwrap_or(0);
                if width > RESIDUAL_WIDTH_CAP {
                    return Err(crate::Error::ResidualWidthOverflow(width));
                }
                let bs = current_block_size as usize;
                for _ in 0..bs {
                    let _ = read_svar(&mut br, width)?;
                }
                // Advance round bookkeeping.
                channel_cursor = (channel_cursor + 1) % nch;
                if channel_cursor == 0 {
                    samples_per_channel += bs as i64;
                    rounds_since_index += 1;
                    if rounds_since_index >= stride {
                        entries.push(FrameIndexEntry {
                            pts: samples_per_channel,
                            byte_offset: 5 + (br.bit_pos() as u64 / 8),
                        });
                        rounds_since_index = 0;
                    }
                }
            }
            fn_code::QLPC => {
                let order = read_uvar(&mut br, LPCQSIZE)?;
                for _ in 0..order {
                    let _ = read_svar(&mut br, LPCQUANT)?;
                }
                let energy = read_uvar(&mut br, ENERGYSIZE)?;
                let width = energy.checked_add(1).unwrap_or(0);
                if width > RESIDUAL_WIDTH_CAP {
                    return Err(crate::Error::ResidualWidthOverflow(width));
                }
                let bs = current_block_size as usize;
                for _ in 0..bs {
                    let _ = read_svar(&mut br, width)?;
                }
                channel_cursor = (channel_cursor + 1) % nch;
                if channel_cursor == 0 {
                    samples_per_channel += bs as i64;
                    rounds_since_index += 1;
                    if rounds_since_index >= stride {
                        entries.push(FrameIndexEntry {
                            pts: samples_per_channel,
                            byte_offset: 5 + (br.bit_pos() as u64 / 8),
                        });
                        rounds_since_index = 0;
                    }
                }
            }
            fn_code::ZERO => {
                // No residual payload — emits `bs` samples = running
                // mean. Just advance the round bookkeeping.
                let bs = current_block_size as i64;
                channel_cursor = (channel_cursor + 1) % nch;
                if channel_cursor == 0 {
                    samples_per_channel += bs;
                    rounds_since_index += 1;
                    if rounds_since_index >= stride {
                        entries.push(FrameIndexEntry {
                            pts: samples_per_channel,
                            byte_offset: 5 + (br.bit_pos() as u64 / 8),
                        });
                        rounds_since_index = 0;
                    }
                }
            }
            fn_code::BLOCKSIZE => {
                let new_bs = read_ulong(&mut br)?;
                if new_bs == 0 || new_bs > crate::MAX_BLOCKSIZE {
                    return Err(crate::Error::UnsupportedBlockSize(new_bs));
                }
                current_block_size = new_bs;
            }
            fn_code::BITSHIFT => {
                let new_shift = read_uvar(&mut br, BITSHIFTSIZE)?;
                if new_shift >= 32 {
                    return Err(crate::Error::BitShiftOverflow(new_shift));
                }
                // No state we track here cares about bshift — the
                // residual layout is unaffected.
            }
            fn_code::VERBATIM => {
                let length = read_uvar(&mut br, VERBATIM_CHUNK_SIZE)?;
                for _ in 0..length {
                    let _ = read_uvar(&mut br, VERBATIM_BYTE_SIZE)?;
                }
            }
            fn_code::QUIT => {
                break;
            }
            other => return Err(crate::Error::UnknownFunctionCode(other)),
        }
    }

    Ok(entries)
}

// ───────────────────────── PCM packer ─────────────────────────

/// Pack the i32-lane samples of a [`crate::DecodedStream`] into the
/// byte layout for `output_format`.
///
/// For unsigned output filetypes (`U8`, `U16*`) the predictor lane's
/// signed samples are biased by the half-range so the resulting byte
/// payload matches the expected host PCM layout. This matches the
/// encoder side of TR.156 §"-m blocks" + §3.1 ("subtract a non-zero
/// mean by default for v2/v3 streams"); when the encoder did *not*
/// subtract a mean, the bias defaults to zero and the unsigned output
/// is the signed lane reinterpreted as unsigned.
fn pcm_pack(stream: &crate::DecodedStream, output_format: SampleFormat) -> CoreResult<Vec<u8>> {
    let bytes_per_sample = output_format.bytes_per_sample();
    let mut out = Vec::with_capacity(stream.samples.len() * bytes_per_sample);
    let ft = stream.header.filetype;
    match (output_format, ft) {
        (SampleFormat::S16, Filetype::S16Le) => {
            for &s in &stream.samples {
                out.extend_from_slice(&(s as i16).to_le_bytes());
            }
        }
        (SampleFormat::S16, Filetype::S16Be) => {
            for &s in &stream.samples {
                out.extend_from_slice(&(s as i16).to_be_bytes());
            }
        }
        (SampleFormat::S16, Filetype::S16Native) => {
            for &s in &stream.samples {
                out.extend_from_slice(&(s as i16).to_ne_bytes());
            }
        }
        (SampleFormat::S16, Filetype::S16Swapped) => {
            for &s in &stream.samples {
                let bytes = (s as i16).to_ne_bytes();
                out.extend_from_slice(&[bytes[1], bytes[0]]);
            }
        }
        (SampleFormat::S16, Filetype::U16Le) => {
            for &s in &stream.samples {
                let u = s.wrapping_add(0x8000) as u16;
                out.extend_from_slice(&u.to_le_bytes());
            }
        }
        (SampleFormat::S16, Filetype::U16Be) => {
            for &s in &stream.samples {
                let u = s.wrapping_add(0x8000) as u16;
                out.extend_from_slice(&u.to_be_bytes());
            }
        }
        (SampleFormat::S16, Filetype::U16Native) => {
            for &s in &stream.samples {
                let u = s.wrapping_add(0x8000) as u16;
                out.extend_from_slice(&u.to_ne_bytes());
            }
        }
        (SampleFormat::S16, Filetype::U16Swapped) => {
            for &s in &stream.samples {
                let u = s.wrapping_add(0x8000) as u16;
                let bytes = u.to_ne_bytes();
                out.extend_from_slice(&[bytes[1], bytes[0]]);
            }
        }
        (SampleFormat::U8, Filetype::U8) => {
            for &s in &stream.samples {
                let v = (s & 0xFF) as u8;
                out.push(v);
            }
        }
        (SampleFormat::S8, Filetype::S8) => {
            for &s in &stream.samples {
                out.push((s as i8) as u8);
            }
        }
        (SampleFormat::U8, Filetype::Ulaw) => {
            // µ-law samples on the wire are 8-bit µ-law-encoded bytes
            // packed into the signed predictor lane. The decoder
            // surfaces them as i32; we narrow to u8 here. Linear-PCM
            // expansion (per ITU-T G.711) is the consumer's job — the
            // packer's contract is "raw filetype bytes".
            for &s in &stream.samples {
                let v = (s & 0xFF) as u8;
                out.push(v);
            }
        }
        (other_fmt, other_ft) => {
            return Err(CoreError::unsupported(format!(
                "oxideav-shorten: unsupported (output_format={other_fmt:?}, filetype={other_ft:?}) combination"
            )));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_decoder_factory() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let codec_id = CodecId::new(CODEC_ID_STR);
        assert!(
            ctx.codecs.has_decoder(&codec_id),
            "register_codecs should install a decoder factory"
        );
    }

    #[test]
    fn registers_container_probe() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        // Build a minimal valid ajkg+v2 prefix and make sure the
        // probe scores it 100.
        let buf = [b'a', b'j', b'k', b'g', 0x02, 0x00, 0x00];
        let p = ProbeData {
            buf: &buf,
            ext: None,
        };
        assert_eq!(shorten_probe(&p), 100);
        // Wrong magic.
        let bad = [0u8; 8];
        let p2 = ProbeData {
            buf: &bad,
            ext: None,
        };
        assert_eq!(shorten_probe(&p2), 0);
        // Right magic but bogus version.
        let bad_ver = [b'a', b'j', b'k', b'g', 0x09];
        let p3 = ProbeData {
            buf: &bad_ver,
            ext: None,
        };
        assert_eq!(shorten_probe(&p3), 0);
    }

    #[test]
    fn demuxer_round_trips_a_simple_stream() {
        // Encode a short stream with the production encoder, hand the
        // bytes to the demuxer, and confirm the demuxer surfaces a
        // single packet whose payload is bit-identical to the input.
        use std::io::Cursor;

        let cfg = crate::EncoderConfig::new(crate::Filetype::S16Le, 1).with_blocksize(8);
        let encoded = crate::encode(&cfg, &(0i32..16).collect::<Vec<_>>()).unwrap();

        let mut ctx = RuntimeContext::new();
        register(&mut ctx);

        let cursor: Box<dyn ReadSeek> = Box::new(Cursor::new(encoded.clone()));
        let mut demux = ctx
            .containers
            .open_demuxer(CONTAINER_NAME, cursor, &oxideav_core::NullCodecResolver)
            .expect("open demuxer");
        let pkt = demux.next_packet().expect("first packet");
        assert_eq!(pkt.stream_index, 0);
        assert!(pkt.flags.keyframe);
        assert_eq!(pkt.data, encoded);
        // Second call returns EOF.
        match demux.next_packet() {
            Err(CoreError::Eof) => {}
            other => panic!("expected EOF on second packet; got {other:?}"),
        }
    }
}
