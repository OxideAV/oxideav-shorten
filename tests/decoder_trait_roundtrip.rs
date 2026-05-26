//! Integration test for the round-8 `oxideav_core::Decoder` trait
//! wiring.
//!
//! Gated on the `registry` cargo feature (default-on); a standalone
//! build without `oxideav-core` skips it entirely.

#![cfg(feature = "registry")]

//!
//! Drives a synthetic Shorten v2 byte stream through the registered
//! decoder factory (`make_decoder` resolved out of a freshly-
//! constructed `RuntimeContext`'s `CodecRegistry`), pulls one
//! `AudioFrame`, and asserts the per-channel plane bytes packed under
//! the `H_filetype = 5` (`s16lh`, little-endian `i16`) byte-order
//! rule of `docs/audio/shorten/spec/05-state-and-quirks.md` §6 match
//! the per-channel `i32` samples that the round-7 driver
//! `decode_stream` would produce on the same bytes, packed by hand to
//! 16-bit little-endian.
//!
//! Behavioural anchors:
//! * `spec/05` §6 — file-type-code numeric mapping
//!   (`5 → s16lh, 16-bit little-endian`).
//! * `spec/03` §2 — per-channel round-robin sample ordering driving
//!   the planar `AudioFrame::data` layout (one plane per channel).
//! * `spec/03` §3.10 — verbatim prefix collection (exposed via
//!   `ShortenDecoder::verbatim_prefix`).
//! * `spec/03` §3.8 — `BLOCK_FN_QUIT` stream terminator, after which
//!   the trait wrapper queues the single emitted `AudioFrame` and
//!   surfaces `Error::Eof` from subsequent `receive_frame` calls.

use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, Error as CoreError, Frame, Packet, TimeBase,
};
use oxideav_shorten::{
    decode_stream, register_codecs, ShortenDecoder, CODEC_ID_STR, FILETYPE_S16LH, FNSIZE, MAGIC,
};

// ---- bit-packer + uvar/svar/ulong builders mirroring driver.rs ----

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

fn bits_for(v: u32) -> u32 {
    if v == 0 {
        0
    } else {
        32 - v.leading_zeros()
    }
}

fn header_param_bits(
    filetype: u32,
    channels: u32,
    blocksize: u32,
    maxlpcorder: u32,
    meanblocks: u32,
    skipbytes: u32,
) -> Vec<u32> {
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
    buf.extend_from_slice(&MAGIC);
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

// ENERGYSIZE = 3 per `spec/02` §4.2; mirrored from the crate.
const ENERGYSIZE: u32 = 3;

/// Build a synthetic two-channel v2 stream with a verbatim prefix,
/// two DIFF blocks (one per channel), and a QUIT terminator. The
/// per-channel `i32` samples this assembles to are pinned in the
/// asserts below.
fn build_two_channel_s16lh_stream() -> Vec<u8> {
    let mut bits = header_param_bits(FILETYPE_S16LH, 2, 4, 0, 0, 0);
    // VERBATIM prefix: 4-byte "RIFF" envelope head (`spec/03` §3.10).
    bits.extend(encode_uvar(9, FNSIZE));
    bits.extend(encode_uvar(4, 5));
    for b in [b'R', b'I', b'F', b'F'] {
        bits.extend(encode_uvar(b as u32, 8));
    }
    // DIFF1 ch0 residuals [1, 2, 3, 4] over zero carry
    //   -> running cumulative sum [1, 3, 6, 10].
    append_diff_block(&mut bits, 1, 3, &[1, 2, 3, 4]);
    // DIFF0 ch1 residuals [100, 200, 300, 400] (mean-invariant since
    // H_meanblocks = 0 — `spec/01` §3.5).
    append_diff_block(&mut bits, 0, 5, &[100, 200, 300, 400]);
    bits.extend(encode_uvar(4, FNSIZE)); // QUIT
    assemble(&bits)
}

#[test]
fn registered_decoder_emits_frame_bit_exact_to_driver_output() {
    let bytes = build_two_channel_s16lh_stream();

    // Reference: drive `decode_stream` directly so the trait wrapper
    // can be byte-for-byte compared.
    let direct = decode_stream(&bytes).expect("direct decode_stream");
    assert_eq!(direct.header.filetype, FILETYPE_S16LH);
    assert_eq!(direct.header.channels, 2);
    assert_eq!(direct.channels[0], vec![1, 3, 6, 10]);
    assert_eq!(direct.channels[1], vec![100, 200, 300, 400]);
    assert_eq!(direct.verbatim, vec![b'R', b'I', b'F', b'F']);

    // Trait-driven: resolve the factory through the registry and run
    // the full send_packet -> receive_frame -> Eof cycle.
    let mut reg = CodecRegistry::new();
    register_codecs(&mut reg);
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.sample_rate = Some(44_100);
    let mut dec = reg.first_decoder(&params).expect("first_decoder");

    let tb = TimeBase::new(1, 44_100);
    let mut pkt = Packet::new(0, tb, bytes);
    pkt.pts = Some(0);
    dec.send_packet(&pkt).expect("send_packet");

    let frame = match dec.receive_frame().expect("receive_frame") {
        Frame::Audio(a) => a,
        other => panic!("expected audio frame, got {other:?}"),
    };
    // The stream terminated at BLOCK_FN_QUIT inside the single
    // packet; subsequent receive_frame surfaces Eof.
    assert!(matches!(dec.receive_frame(), Err(CoreError::Eof)));

    // Bit-exact comparison: hand-pack each direct-driver i32 channel
    // to little-endian i16 bytes and assert plane-by-plane equality.
    assert_eq!(frame.samples, 4);
    assert_eq!(frame.pts, Some(0));
    assert_eq!(frame.data.len(), 2);
    for (ci, ch) in direct.channels.iter().enumerate() {
        let mut expected = Vec::with_capacity(ch.len() * 2);
        for &s in ch {
            expected.extend_from_slice(&(s as i16).to_le_bytes());
        }
        assert_eq!(
            frame.data[ci], expected,
            "channel {ci} plane bytes mismatch trait vs direct"
        );
    }
}

#[test]
fn registry_resolves_factory_after_register_codecs() {
    // The runtime resolution path: a freshly-constructed
    // `CodecRegistry`, the crate's `register_codecs` call, then
    // `first_decoder` on the codec id resolves a working factory.
    // Pins the registry surface independently of the bit-exact body.
    let mut reg = CodecRegistry::new();
    register_codecs(&mut reg);
    let id = CodecId::new(CODEC_ID_STR);
    assert!(reg.has_decoder(&id), "decoder factory not registered");
    let params = CodecParameters::audio(id);
    let dec = reg.first_decoder(&params).expect("first_decoder");
    assert_eq!(dec.codec_id().as_str(), CODEC_ID_STR);
    // ShortenDecoder is publicly re-exported for callers that need
    // the verbatim-prefix accessor that the boxed trait surface
    // doesn't expose.
    let _ = std::any::type_name::<ShortenDecoder>();
}
