//! Integration test for the round-304 `oxideav_core::Encoder` trait
//! wiring [`oxideav_shorten::ShortenEncoder`] — the frame-in / packet-out
//! mirror of the round-8 [`oxideav_shorten::ShortenDecoder`].
//!
//! The encoder buffers planar [`oxideav_core::AudioFrame`] PCM across
//! `send_frame` calls, re-interleaves on `flush`, and runs
//! [`oxideav_shorten::encode_stream`] to produce one `.shn`
//! [`oxideav_core::Packet`]. These tests resolve the encoder (and the
//! decoder) out of a [`oxideav_core::CodecRegistry`] via the crate's
//! `register_encoder` / `register_codecs` installers and assert the
//! encode → decode round-trip is sample-exact, exercising the full
//! registry path end to end.
//!
//! Clean-room provenance:
//!
//! * `docs/audio/shorten/spec/05-state-and-quirks.md` §6 (the host-byte
//!   packing the encoder reverses on `send_frame` and the decoder
//!   applies on emission).
//! * `docs/audio/shorten/spec/03-block-and-predictor.md` §2 (per-channel
//!   interleaving).
//! * the public `oxideav_core` `Encoder` / `Decoder` trait contract.

use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, CodecRegistry, Frame, Packet, SampleFormat, TimeBase,
};
use oxideav_shorten::{make_encoder, register_codecs, register_encoder, CODEC_ID_STR};

fn params(channels: u16) -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    p.channels = Some(channels);
    p.sample_format = Some(SampleFormat::S16P);
    p
}

fn pack_s16lh(samples: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        out.extend_from_slice(&(s as i16).to_le_bytes());
    }
    out
}

fn audio(planes: Vec<Vec<u8>>, samples: u32) -> Frame {
    Frame::Audio(AudioFrame {
        samples,
        pts: None,
        data: planes,
    })
}

/// Encode mono PCM through a registry-resolved encoder, then decode the
/// produced packet through a registry-resolved decoder; assert the
/// recovered samples are bit-exact.
#[test]
fn registry_encode_then_registry_decode_mono_sample_exact() {
    let samples: Vec<i32> = (0..512).map(|t| ((t * 11) % 600) - 300).collect();

    let mut reg = CodecRegistry::new();
    register_encoder(&mut reg);
    register_codecs(&mut reg);
    let id = CodecId::new(CODEC_ID_STR);
    assert!(reg.has_encoder(&id));
    assert!(reg.has_decoder(&id));

    let p = params(1);
    let mut enc = reg.first_encoder(&p).expect("first_encoder");
    enc.send_frame(&audio(vec![pack_s16lh(&samples)], samples.len() as u32))
        .expect("send_frame");
    enc.flush().expect("flush");
    let packet = enc.receive_packet().expect("packet");

    // Decode through the registry-resolved decoder.
    let mut dec = reg.first_decoder(&p).expect("first_decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 44_100), packet.data);
    dec.send_packet(&pkt).expect("send_packet");
    let frame = match dec.receive_frame().expect("receive_frame") {
        Frame::Audio(a) => a,
        other => panic!("expected Audio frame, got {other:?}"),
    };
    // Re-unpack the decoded plane (s16lh) and compare to the source.
    let decoded: Vec<i32> = frame.data[0]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as i32)
        .collect();
    assert_eq!(decoded, samples);
}

/// A frame packed by the decoder fed straight back into the encoder
/// re-encodes to a stream that decodes to the same samples — the
/// decoder ↔ encoder host-byte-packing round-trip.
#[test]
fn decoder_output_frame_reencodes_identically() {
    let ch0: Vec<i32> = (0..200).map(|t| t - 100).collect();
    let ch1: Vec<i32> = (0..200).map(|t| 500 - t * 2).collect();

    // First encode.
    let p = params(2);
    let mut enc = make_encoder(&p).expect("make_encoder");
    enc.send_frame(&audio(vec![pack_s16lh(&ch0), pack_s16lh(&ch1)], 200))
        .expect("send_frame");
    enc.flush().expect("flush");
    let first = enc.receive_packet().expect("packet");

    // Decode it, then re-encode the produced frame.
    let dec = oxideav_shorten::decode_stream(&first.data).expect("decode_stream");
    let plane0 = pack_s16lh(&dec.channels[0]);
    let plane1 = pack_s16lh(&dec.channels[1]);

    let mut enc2 = make_encoder(&p).expect("make_encoder");
    enc2.send_frame(&audio(vec![plane0, plane1], dec.channels[0].len() as u32))
        .expect("send_frame 2");
    enc2.flush().expect("flush 2");
    let second = enc2.receive_packet().expect("packet 2");

    // The two encodes are byte-identical (same selector, same input).
    assert_eq!(first.data, second.data);
}
