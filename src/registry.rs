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
        payload: Some(buf),
    }))
}

struct ShortenDemuxer {
    streams: Vec<StreamInfo>,
    payload: Option<Vec<u8>>,
}

impl Demuxer for ShortenDemuxer {
    fn format_name(&self) -> &str {
        CONTAINER_NAME
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> CoreResult<Packet> {
        match self.payload.take() {
            Some(data) => {
                let mut pkt = Packet::new(0, self.streams[0].time_base, data);
                pkt.pts = Some(0);
                pkt.dts = Some(0);
                pkt.flags.keyframe = true;
                Ok(pkt)
            }
            None => Err(CoreError::Eof),
        }
    }
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
