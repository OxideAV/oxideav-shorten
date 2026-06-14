//! `oxideav_core::Encoder` wiring for Shorten.
//!
//! Round 290 landed [`encode_stream`](crate::encode_stream) — the
//! whole-stream encode driver that takes an interleaved `&[i32]` PCM
//! buffer plus a [`ShortenStreamHeader`] and produces a syntactically-
//! valid `.shn` byte stream that [`decode_stream`](crate::decode_stream)
//! reconstructs sample-exact. This module (round 304, per the dual-API
//! convention that every codec crate exposes both the registry path and
//! direct `make_*` factory endpoints) exposes that driver through the
//! framework's frame-in / packet-out
//! [`Encoder`](oxideav_core::Encoder) trait, the mirror of the round-8
//! [`ShortenDecoder`](crate::ShortenDecoder) adaptor.
//!
//! ## Trait-API adaptation
//!
//! Shorten is **stream-natively single-packet** on both sides: the
//! `ajkg` header, parameter block, and per-block command stream
//! together form one contiguous bit-aligned blob terminated by
//! `BLOCK_FN_QUIT` + byte padding (`spec/05` §4). There is no inter-
//! frame framing the encoder emits; the whole stream IS the packet, and
//! the predictor / mean-estimator state runs across every block of the
//! file. The encoder therefore cannot emit anything until it has seen
//! the entire sample population.
//!
//! The wrapper behaves as follows:
//!
//! * [`send_frame`](oxideav_core::Encoder::send_frame) unpacks the
//!   incoming planar [`AudioFrame`] into per-channel `i32` sample
//!   vectors (reversing the `spec/05` §6 host-byte packing the decoder
//!   applies) and appends them to per-channel accumulators. Multiple
//!   frames concatenate into one logical stream.
//! * [`flush`](oxideav_core::Encoder::flush) re-interleaves the
//!   accumulated per-channel samples, builds the [`ShortenStreamHeader`]
//!   from the encoder's [`CodecParameters`] + options, runs
//!   [`encode_stream`] once, and queues exactly one [`Packet`] holding
//!   the full `.shn` byte stream.
//! * [`receive_packet`](oxideav_core::Encoder::receive_packet) pops the
//!   queued packet, returns [`Error::NeedMore`] before `flush` has been
//!   called, and [`Error::Eof`] after the queued packet has been
//!   drained.
//!
//! ## Header parameters
//!
//! The encoder derives [`ShortenStreamHeader`] from its
//! [`CodecParameters`]:
//!
//! * `H_channels` — `params.channels` (required; the planar frame's
//!   plane count must match).
//! * `H_filetype` — derived from `params.sample_format`
//!   ([`SampleFormat::U8P`] → `2`/`u8`; [`SampleFormat::S16P`] →
//!   `5`/`s16lh` by default, the little-endian form pinned by fixture
//!   `F1` in `spec/05` §6). A `"filetype"` codec option overrides the
//!   numeric value directly (allowing `3`/`s16hl` selection); only the
//!   three codes `spec/05` §6 pins (`2`, `3`, `5`) are accepted.
//! * `H_blocksize` — the `"blocksize"` option, default `256` (TR.156's
//!   default per `spec/01` §3.3).
//! * `H_maxlpcorder` — the `"maxlpcorder"` option, default `0` (the
//!   polynomial-difference-only candidate set; `spec/01` §3.4).
//! * `H_meanblocks` — the `"meanblocks"` option, default `0` (the
//!   running-mean estimator disabled; `spec/01` §3.5 / `spec/05` §2).
//! * `version` — `2` (the v2/v3 envelope the parameter-block writer
//!   emits; `spec/00`).
//!
//! ## Sample-format byte unpacking (`spec/05` §6)
//!
//! `send_frame` reverses the same byte packing
//! [`ShortenDecoder`](crate::ShortenDecoder) applies on emission:
//!
//! | `H_filetype` | Label   | Plane bytes per sample            |
//! | ------------ | ------- | --------------------------------- |
//! | 2            | `u8`    | 1 (`u8` widened to `i32`)         |
//! | 3            | `s16hl` | 2 (`i16` big-endian → `i32`)      |
//! | 5            | `s16lh` | 2 (`i16` little-endian → `i32`)   |
//!
//! The round-trip with [`ShortenDecoder`] is the load-bearing property:
//! a frame packed by the decoder and fed straight back into this encoder
//! re-encodes to a stream that decodes to the same samples.
//!
//! ## Clean-room provenance
//!
//! The trait wiring is assembled from `docs/audio/shorten/spec/05` §6
//! (the file-type table), `spec/03` §2 (per-channel sample ordering),
//! `spec/01` §3 (header field semantics), and the public surface of
//! `oxideav_core` (the `Encoder` trait contract). It adds no new
//! wire-format decisions: every byte it emits comes from
//! [`encode_stream`], which earlier rounds pinned against the spec.

use std::collections::VecDeque;

use oxideav_core::{
    AudioFrame, CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, Encoder,
    Error as CoreError, Frame, Packet, Result as CoreResult, SampleFormat, TimeBase,
};

use crate::codec::{CODEC_ID_STR, FILETYPE_S16HL, FILETYPE_S16LH, FILETYPE_U8};
use crate::encode_driver::encode_stream;
use crate::encoder::EncodeError;
use crate::header::ShortenStreamHeader;

/// TR.156 default per-channel block size (`spec/01` §3.3).
pub const DEFAULT_BLOCKSIZE: u32 = 256;
/// Default format version the encoder stamps into the header (`spec/00`).
pub const ENCODER_HEADER_VERSION: u8 = 2;

/// Build a boxed Shorten [`Encoder`] from `params`.
///
/// `params.channels` is required — the encoder needs the channel count
/// to deinterleave / re-interleave the sample stream and to validate
/// each frame's plane count. `params.sample_format` selects the default
/// `H_filetype` packing (overridable via the `"filetype"` option). The
/// `"blocksize"`, `"maxlpcorder"`, and `"meanblocks"` options tune the
/// remaining header fields; all have spec-default fallbacks.
///
/// The factory fails with [`CoreError::invalid`] when `params.channels`
/// is missing or zero, or when an option value is malformed. The actual
/// encode happens at [`Encoder::flush`]; per-frame errors surface at the
/// trait boundary.
pub fn make_encoder(params: &CodecParameters) -> CoreResult<Box<dyn Encoder>> {
    Ok(Box::new(ShortenEncoder::new(params)?))
}

/// Resolve the numeric `H_filetype` from the codec parameters: a
/// `"filetype"` option (if present) takes precedence, otherwise the
/// `sample_format` selects a default packing.
fn resolve_filetype(params: &CodecParameters) -> CoreResult<u32> {
    if let Some(raw) = params.options.get("filetype") {
        let v: u32 = raw.parse().map_err(|_| {
            CoreError::invalid(format!(
                "oxideav-shorten: filetype option {raw:?} is not a non-negative integer"
            ))
        })?;
        return match v {
            FILETYPE_U8 | FILETYPE_S16HL | FILETYPE_S16LH => Ok(v),
            other => Err(CoreError::unsupported(format!(
                "oxideav-shorten: filetype {other} not pinned by spec/05 §6 \
                 (only 2/u8, 3/s16hl, 5/s16lh have pinned numeric codes)"
            ))),
        };
    }
    match params.sample_format {
        Some(SampleFormat::U8P) | Some(SampleFormat::U8) => Ok(FILETYPE_U8),
        // S16P / S16 default to the little-endian s16lh packing (the
        // form fixture F1 pins in spec/05 §6). A producer that wants the
        // big-endian s16hl packing sets the "filetype" option to 3.
        Some(SampleFormat::S16P) | Some(SampleFormat::S16) => Ok(FILETYPE_S16LH),
        Some(other) => Err(CoreError::unsupported(format!(
            "oxideav-shorten: sample format {other:?} has no Shorten H_filetype mapping \
             (set the \"filetype\" option to one of 2/u8, 3/s16hl, 5/s16lh)"
        ))),
        None => Err(CoreError::invalid(
            "oxideav-shorten: encoder needs params.sample_format or a \"filetype\" option",
        )),
    }
}

/// Parse a non-negative integer codec option, falling back to `default`
/// when absent.
fn option_u32(params: &CodecParameters, key: &str, default: u32) -> CoreResult<u32> {
    match params.options.get(key) {
        Some(raw) => raw.parse().map_err(|_| {
            CoreError::invalid(format!(
                "oxideav-shorten: {key} option {raw:?} is not a non-negative integer"
            ))
        }),
        None => Ok(default),
    }
}

/// Frame-to-packet adaptor that wires [`encode_stream`] into the
/// framework [`Encoder`] trait.
///
/// State carried across `send_frame` calls:
///
/// * `planes` — one `Vec<i32>` accumulator per channel, filled by
///   unpacking each frame's planar PCM bytes per the `H_filetype` table.
/// * `pending` — at most one fully-encoded [`Packet`]; produced by
///   [`Encoder::flush`] and popped by
///   [`Encoder::receive_packet`].
/// * `flushed` — once `flush` has run the encoder stops accepting
///   frames.
/// * `pts` — taken from the first frame whose `pts.is_some()`, attached
///   to the emitted packet.
pub struct ShortenEncoder {
    codec_id: CodecId,
    output: CodecParameters,
    filetype: u32,
    blocksize: u32,
    maxlpcorder: u32,
    meanblocks: u32,
    n_channels: usize,
    planes: Vec<Vec<i32>>,
    pending: VecDeque<Packet>,
    flushed: bool,
    drained: bool,
    pts: Option<i64>,
}

impl std::fmt::Debug for ShortenEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShortenEncoder")
            .field("codec_id", &self.codec_id)
            .field("filetype", &self.filetype)
            .field("blocksize", &self.blocksize)
            .field("maxlpcorder", &self.maxlpcorder)
            .field("meanblocks", &self.meanblocks)
            .field("n_channels", &self.n_channels)
            .field(
                "accumulated_samples_per_channel",
                &self.planes.first().map(|p| p.len()).unwrap_or(0),
            )
            .field("pending", &self.pending.len())
            .field("flushed", &self.flushed)
            .finish()
    }
}

impl ShortenEncoder {
    fn new(params: &CodecParameters) -> CoreResult<Self> {
        let n_channels = match params.channels {
            Some(0) | None => {
                return Err(CoreError::invalid(
                    "oxideav-shorten: encoder needs params.channels >= 1",
                ));
            }
            Some(c) => c as usize,
        };
        let filetype = resolve_filetype(params)?;
        let blocksize = option_u32(params, "blocksize", DEFAULT_BLOCKSIZE)?;
        if blocksize == 0 {
            return Err(CoreError::invalid(
                "oxideav-shorten: blocksize option must be >= 1",
            ));
        }
        let maxlpcorder = option_u32(params, "maxlpcorder", 0)?;
        let meanblocks = option_u32(params, "meanblocks", 0)?;

        let mut output = params.clone();
        output.channels = Some(n_channels as u16);
        output.sample_format = Some(host_sample_format(filetype));

        Ok(Self {
            codec_id: params.codec_id.clone(),
            output,
            filetype,
            blocksize,
            maxlpcorder,
            meanblocks,
            n_channels,
            planes: vec![Vec::new(); n_channels],
            pending: VecDeque::new(),
            flushed: false,
            drained: false,
            pts: None,
        })
    }

    /// Number of samples accumulated per channel so far.
    pub fn accumulated_samples(&self) -> usize {
        self.planes.first().map(|p| p.len()).unwrap_or(0)
    }

    /// Unpack one planar [`AudioFrame`] into the per-channel
    /// accumulators, reversing the `spec/05` §6 host-byte packing.
    fn append_frame(&mut self, audio: &AudioFrame) -> CoreResult<()> {
        if audio.data.len() != self.n_channels {
            return Err(CoreError::invalid(format!(
                "oxideav-shorten: frame has {} planes but encoder expects {} channels",
                audio.data.len(),
                self.n_channels
            )));
        }
        let expected = audio.samples as usize;
        for (ci, plane) in audio.data.iter().enumerate() {
            let samples = unpack_plane(self.filetype, plane)?;
            if samples.len() != expected {
                return Err(CoreError::invalid(format!(
                    "oxideav-shorten: plane {ci} decodes to {} samples but \
                     AudioFrame::samples = {expected}",
                    samples.len()
                )));
            }
            self.planes[ci].extend_from_slice(&samples);
        }
        Ok(())
    }

    /// Re-interleave the accumulated per-channel samples and run
    /// [`encode_stream`], queuing the produced `.shn` bytes as one
    /// [`Packet`]. Idempotent: a second call after the packet is queued
    /// is a no-op.
    fn finish_stream(&mut self) -> CoreResult<()> {
        if self.flushed {
            return Ok(());
        }
        self.flushed = true;

        let per_channel = self.accumulated_samples();
        // Sanity: every plane must have the same length (send_frame
        // enforces this per frame, but a divergent total would corrupt
        // the interleave).
        for (ci, p) in self.planes.iter().enumerate() {
            if p.len() != per_channel {
                return Err(CoreError::invalid(format!(
                    "oxideav-shorten: channel {ci} accumulated {} samples != channel 0's {per_channel}",
                    p.len()
                )));
            }
        }

        let mut interleaved = Vec::with_capacity(per_channel * self.n_channels);
        for t in 0..per_channel {
            for c in 0..self.n_channels {
                interleaved.push(self.planes[c][t]);
            }
        }

        let header = ShortenStreamHeader {
            version: ENCODER_HEADER_VERSION,
            filetype: self.filetype,
            channels: self.n_channels as u32,
            blocksize: self.blocksize,
            maxlpcorder: self.maxlpcorder,
            meanblocks: self.meanblocks,
            skipbytes: 0,
        };

        let bytes = encode_stream(&header, &interleaved, &[]).map_err(encode_error_to_core)?;

        let mut packet = Packet::new(0, TimeBase::new(1, 1), bytes);
        packet.pts = self.pts;
        self.pending.push_back(packet);
        Ok(())
    }
}

impl Encoder for ShortenEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output
    }

    fn send_frame(&mut self, frame: &Frame) -> CoreResult<()> {
        if self.flushed {
            return Err(CoreError::other(
                "oxideav-shorten: cannot send_frame after flush",
            ));
        }
        let audio = match frame {
            Frame::Audio(a) => a,
            other => {
                return Err(CoreError::invalid(format!(
                    "oxideav-shorten: encoder accepts only audio frames, got {other:?}"
                )));
            }
        };
        if self.pts.is_none() {
            self.pts = audio.pts;
        }
        self.append_frame(audio)
    }

    fn receive_packet(&mut self) -> CoreResult<Packet> {
        if let Some(p) = self.pending.pop_front() {
            return Ok(p);
        }
        if self.flushed {
            self.drained = true;
            return Err(CoreError::Eof);
        }
        Err(CoreError::NeedMore)
    }

    fn flush(&mut self) -> CoreResult<()> {
        self.finish_stream()
    }
}

/// Map a numeric `H_filetype` value to the framework [`SampleFormat`]
/// surfaced through [`CodecParameters`]. Byte order for the
/// `s16hl` / `s16lh` filetypes is encoded into the packed plane bytes —
/// both surface as the generic `S16P` shape here, mirroring the decoder
/// side.
fn host_sample_format(filetype: u32) -> SampleFormat {
    match filetype {
        FILETYPE_U8 => SampleFormat::U8P,
        // Only the three pinned codes reach here (resolve_filetype
        // rejects everything else), so s16hl / s16lh are the only S16
        // cases.
        _ => SampleFormat::S16P,
    }
}

/// Reverse the `spec/05` §6 host-byte packing for one plane: turn the
/// packed bytes back into the `i32` sample values [`encode_stream`]
/// consumes.
fn unpack_plane(filetype: u32, plane: &[u8]) -> CoreResult<Vec<i32>> {
    match filetype {
        FILETYPE_U8 => Ok(plane.iter().map(|&b| b as i32).collect()),
        FILETYPE_S16HL => {
            if plane.len() % 2 != 0 {
                return Err(CoreError::invalid(
                    "oxideav-shorten: s16hl plane length is not a multiple of 2",
                ));
            }
            Ok(plane
                .chunks_exact(2)
                .map(|c| i16::from_be_bytes([c[0], c[1]]) as i32)
                .collect())
        }
        FILETYPE_S16LH => {
            if plane.len() % 2 != 0 {
                return Err(CoreError::invalid(
                    "oxideav-shorten: s16lh plane length is not a multiple of 2",
                ));
            }
            Ok(plane
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as i32)
                .collect())
        }
        other => Err(CoreError::unsupported(format!(
            "oxideav-shorten: H_filetype {other} not pinned by spec/05 §6"
        ))),
    }
}

/// Translate a crate-local [`EncodeError`] into the framework
/// [`oxideav_core::Error`]. The framework's error enum has no per-codec
/// variant tree, so each crate-local variant maps onto a generic
/// flavour with a descriptive message.
fn encode_error_to_core(e: EncodeError) -> CoreError {
    match e {
        EncodeError::UnsupportedVersion(v) => {
            CoreError::unsupported(format!("oxideav-shorten: encode version {v}"))
        }
        EncodeError::ZeroChannels => CoreError::invalid("oxideav-shorten: encode H_channels = 0"),
        other => CoreError::other(format!("oxideav-shorten: {other}")),
    }
}

/// Install the Shorten **encoder** factory into `reg` under the existing
/// codec id [`CODEC_ID_STR`] (`"shorten"`), alongside the decoder
/// factory [`register_codecs`](crate::register_codecs) installs.
///
/// Registered as a separate call so a caller can wire only the side it
/// needs; [`crate::register`] installs both.
pub fn register_encoder(reg: &mut CodecRegistry) {
    let info = CodecInfo::new(CodecId::new(CODEC_ID_STR))
        .capabilities(
            CodecCapabilities::audio(CODEC_ID_STR)
                .with_encode()
                .with_lossless(true),
        )
        .encoder(make_encoder);
    reg.register(info);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_stream;
    use oxideav_core::CodecOptions;

    fn params_with(channels: u16, fmt: SampleFormat) -> CodecParameters {
        let mut p = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        p.channels = Some(channels);
        p.sample_format = Some(fmt);
        p
    }

    fn audio_frame(planes: Vec<Vec<u8>>, samples: u32, pts: Option<i64>) -> Frame {
        Frame::Audio(AudioFrame {
            samples,
            pts,
            data: planes,
        })
    }

    /// Pack a per-channel i32 sample vector to s16lh (little-endian)
    /// plane bytes — the inverse of `unpack_plane` for FILETYPE_S16LH.
    fn pack_s16lh(samples: &[i32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            out.extend_from_slice(&(s as i16).to_le_bytes());
        }
        out
    }

    #[test]
    fn make_encoder_builds_and_reports_codec_id() {
        let p = params_with(1, SampleFormat::S16P);
        let enc = make_encoder(&p).expect("make_encoder");
        assert_eq!(enc.codec_id().as_str(), CODEC_ID_STR);
    }

    #[test]
    fn make_encoder_rejects_missing_channels() {
        let mut p = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        p.sample_format = Some(SampleFormat::S16P);
        assert!(make_encoder(&p).is_err());
    }

    #[test]
    fn make_encoder_rejects_zero_channels() {
        let p = params_with(0, SampleFormat::S16P);
        assert!(make_encoder(&p).is_err());
    }

    #[test]
    fn register_encoder_installs_encoder_factory() {
        let mut reg = CodecRegistry::new();
        register_encoder(&mut reg);
        let id = CodecId::new(CODEC_ID_STR);
        assert!(reg.has_encoder(&id));
    }

    #[test]
    fn resolve_filetype_defaults_from_sample_format() {
        let u8p = params_with(1, SampleFormat::U8P);
        assert_eq!(resolve_filetype(&u8p).unwrap(), FILETYPE_U8);
        let s16p = params_with(1, SampleFormat::S16P);
        assert_eq!(resolve_filetype(&s16p).unwrap(), FILETYPE_S16LH);
    }

    #[test]
    fn resolve_filetype_option_overrides_to_s16hl() {
        let mut p = params_with(1, SampleFormat::S16P);
        p.options = CodecOptions::default().set("filetype", "3");
        assert_eq!(resolve_filetype(&p).unwrap(), FILETYPE_S16HL);
    }

    #[test]
    fn resolve_filetype_rejects_unpinned_option() {
        let mut p = params_with(1, SampleFormat::S16P);
        p.options = CodecOptions::default().set("filetype", "7");
        assert!(resolve_filetype(&p).is_err());
    }

    #[test]
    fn unpack_plane_roundtrips_s16lh() {
        let samples = [-100i32, 0, 100, 32_000, -32_000];
        let packed = pack_s16lh(&samples);
        let back = unpack_plane(FILETYPE_S16LH, &packed).unwrap();
        assert_eq!(back, samples.to_vec());
    }

    #[test]
    fn unpack_plane_rejects_odd_length_s16() {
        assert!(unpack_plane(FILETYPE_S16LH, &[0x00]).is_err());
    }

    /// The headline property: a single mono frame round-trips through
    /// the encoder + `decode_stream` sample-exact.
    #[test]
    fn mono_frame_encodes_and_decodes_sample_exact() {
        let samples: Vec<i32> = (0..300).map(|t| ((t * 13) % 400) - 200).collect();
        let p = params_with(1, SampleFormat::S16P);
        let mut enc = make_encoder(&p).expect("make_encoder");
        enc.send_frame(&audio_frame(
            vec![pack_s16lh(&samples)],
            samples.len() as u32,
            Some(7),
        ))
        .expect("send_frame");
        // Before flush, no packet.
        assert!(matches!(enc.receive_packet(), Err(CoreError::NeedMore)));
        enc.flush().expect("flush");
        let packet = match enc.receive_packet() {
            Ok(p) => p,
            other => panic!("expected packet, got {other:?}"),
        };
        assert_eq!(packet.pts, Some(7));
        // After draining, Eof.
        assert!(matches!(enc.receive_packet(), Err(CoreError::Eof)));

        let dec = decode_stream(&packet.data).expect("decode_stream");
        assert_eq!(dec.header.channels, 1);
        assert_eq!(dec.channels[0], samples);
    }

    /// Stereo, delivered across two frames, round-trips sample-exact and
    /// re-interleaves correctly.
    #[test]
    fn stereo_two_frames_roundtrip_sample_exact() {
        let ch0a: Vec<i32> = (0..64).map(|t| t - 32).collect();
        let ch1a: Vec<i32> = (0..64).map(|t| 1000 - t * 3).collect();
        let ch0b: Vec<i32> = (0..40).map(|t| t * 5).collect();
        let ch1b: Vec<i32> = (0..40).map(|t| -(t * 2)).collect();

        let p = params_with(2, SampleFormat::S16P);
        let mut enc = make_encoder(&p).expect("make_encoder");
        enc.send_frame(&audio_frame(
            vec![pack_s16lh(&ch0a), pack_s16lh(&ch1a)],
            64,
            None,
        ))
        .expect("send_frame a");
        enc.send_frame(&audio_frame(
            vec![pack_s16lh(&ch0b), pack_s16lh(&ch1b)],
            40,
            None,
        ))
        .expect("send_frame b");
        enc.flush().expect("flush");
        let packet = enc.receive_packet().expect("packet");

        let dec = decode_stream(&packet.data).expect("decode_stream");
        assert_eq!(dec.header.channels, 2);
        let mut ch0 = ch0a.clone();
        ch0.extend_from_slice(&ch0b);
        let mut ch1 = ch1a.clone();
        ch1.extend_from_slice(&ch1b);
        assert_eq!(dec.channels[0], ch0);
        assert_eq!(dec.channels[1], ch1);
    }

    #[test]
    fn send_frame_rejects_wrong_plane_count() {
        let p = params_with(2, SampleFormat::S16P);
        let mut enc = make_encoder(&p).expect("make_encoder");
        // One plane for a 2-channel encoder.
        let err = enc.send_frame(&audio_frame(vec![pack_s16lh(&[1, 2])], 2, None));
        assert!(err.is_err());
    }

    #[test]
    fn send_frame_after_flush_errors() {
        let p = params_with(1, SampleFormat::S16P);
        let mut enc = make_encoder(&p).expect("make_encoder");
        enc.send_frame(&audio_frame(vec![pack_s16lh(&[1, 2, 3])], 3, None))
            .expect("send_frame");
        enc.flush().expect("flush");
        let err = enc.send_frame(&audio_frame(vec![pack_s16lh(&[4])], 1, None));
        assert!(err.is_err());
    }

    #[test]
    fn u8_filetype_roundtrips_via_encoder() {
        let samples: Vec<i32> = (0..50).map(|t| (t * 5) % 256).collect();
        let plane: Vec<u8> = samples.iter().map(|&s| s as u8).collect();
        let p = params_with(1, SampleFormat::U8P);
        let mut enc = make_encoder(&p).expect("make_encoder");
        enc.send_frame(&audio_frame(vec![plane], samples.len() as u32, None))
            .expect("send_frame");
        enc.flush().expect("flush");
        let packet = enc.receive_packet().expect("packet");
        let dec = decode_stream(&packet.data).expect("decode_stream");
        assert_eq!(dec.header.filetype, FILETYPE_U8);
        assert_eq!(dec.channels[0], samples);
    }

    #[test]
    fn empty_stream_flush_produces_decodable_envelope() {
        // No frames: flush should still emit a valid (sample-empty)
        // stream.
        let p = params_with(1, SampleFormat::S16P);
        let mut enc = make_encoder(&p).expect("make_encoder");
        enc.flush().expect("flush");
        let packet = enc.receive_packet().expect("packet");
        let dec = decode_stream(&packet.data).expect("decode_stream");
        assert_eq!(dec.header.channels, 1);
        assert!(dec.channels[0].is_empty());
    }
}
