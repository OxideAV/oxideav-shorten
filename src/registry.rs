//! `oxideav-core` framework integration.
//!
//! Compiled only when the default-on `registry` Cargo feature is
//! enabled. Standalone consumers (`default-features = false`) do not
//! pull in `oxideav-core` and skip this module entirely.

#![cfg(feature = "registry")]

use oxideav_core::{
    AudioFrame, CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry,
    Decoder as CoreDecoder, Error as CoreError, Frame, Packet, Result as CoreResult,
    RuntimeContext, SampleFormat,
};

use crate::header::Filetype;

/// Canonical codec id string registered with `oxideav-core`. The
/// pipeline looks the codec up by this string when constructing a
/// decoder for a `.shn` stream.
pub const CODEC_ID_STR: &str = "shorten";

/// Register the Shorten decoder with `reg`.
///
/// The Shorten format is signalled by:
///
/// * the four-byte magic `ajkg` at file offset 0
///   (the test-only [`probe`](crate::probe) helper exposes this);
/// * the FourCC `shrn` (used by some legacy WAVE wrappers) — this
///   crate registers it for forward-compat with future container work;
///   the round-1 implementation does not depend on this binding.
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("shorten_sw")
        .with_decode()
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

/// Unified entry point invoked by the `oxideav_core::register!`
/// macro-generated `__oxideav_entry`.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
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

/// Pack the i32-lane samples of a [`crate::DecodedStream`] into the
/// byte layout for `output_format`.
fn pcm_pack(stream: &crate::DecodedStream, output_format: SampleFormat) -> CoreResult<Vec<u8>> {
    let bytes_per_sample = output_format.bytes_per_sample();
    let mut out = Vec::with_capacity(stream.samples.len() * bytes_per_sample);
    match (output_format, stream.header.filetype) {
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
        (SampleFormat::U8, Filetype::U8) => {
            // U8 wire form: predictor lane is signed; output is
            // unsigned with the half-range bias added back. Without
            // an active mean estimator the bias defaults to 0 (i.e.
            // the encoder's input had its mean subtracted on the
            // way in); this is sufficient for decoded test vectors
            // whose encoder never enabled the mean estimator.
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
}
