//! Regression pins for the round-453 cargo-fuzz findings.
//!
//! Each test carries the minimized shape of a defect the `fuzz/`
//! harness surfaced, driven through the same public API the fuzz
//! target exercises, so the defect stays fixed without needing the
//! fuzzer to re-discover it.

use oxideav_core::{CodecId, CodecParameters, Error as CoreError, Frame, Packet, TimeBase};
use oxideav_shorten::{
    encode_stream, make_decoder, make_streaming_decoder, parse_stream_header,
    write_byte_aligned_prefix, write_parameter_block, write_quit_command, write_stream_header,
    write_zero_block, BitWriter, EncodeError, ShortenStreamHeader,
};

/// `parse_header` fuzz target, round 453: a header field with bit 31
/// set has `natural_ulong_width == 32`, and `BitWriter::write_uvar`
/// computed its prefix-zero count as `value >> 32` — a shift-overflow
/// panic in any debug build. Fuzz artifact (24 bytes): a v2
/// header whose `H_channels` field decodes above `2^31`.
#[test]
fn write_uvar_width_32_no_shift_overflow() {
    // Direct primitive pin: width 32 puts the whole value in the
    // mantissa with an empty prefix (`spec/02` §2.1 with `⌊v/2^32⌋ = 0`).
    let mut w = BitWriter::new();
    w.write_uvar(u32::MAX, 32);
    w.write_uvar(0x8000_0000, 32);
    w.pad_to_byte();
    let bytes = w.into_bytes();
    // 1 terminator + 32 mantissa bits, twice = 66 bits -> 9 bytes.
    assert_eq!(bytes.len(), 9);

    // End-to-end pin on the fuzz artifact's header: parse -> rewrite
    // -> reparse must round-trip (the fuzz target's writer/parser
    // equivalence contract).
    let art: [u8; 24] = [
        0x61, 0x6a, 0x6b, 0x67, 0x02, 0xfb, 0x77, 0x80, 0x6a, 0x03, 0x00, 0x00, 0x00, 0xc0, 0x4c,
        0x03, 0x19, 0x00, 0x00, 0x01, 0x00, 0x00, 0x05, 0xad,
    ];
    let parsed = parse_stream_header(&art).expect("artifact header parses");
    let mut out = Vec::new();
    let bits = write_stream_header(&mut out, &parsed.header).expect("re-serialises");
    let again = parse_stream_header(&out).expect("re-parses");
    assert_eq!(again.header, parsed.header);
    assert_eq!(again.bits_consumed_after_v, bits);
}

/// `encode_roundtrip` fuzz target, round 453: `encode_stream` accepted
/// `version: 1` and emitted the v2 parameter-block layout under a v1
/// version byte — bytes the crate's own decoder rejects with
/// `UnsupportedVersion(1)` (the v1 layout is unpinned, `spec/01` §3.5).
/// The encoder must refuse up front instead of producing an
/// undecodable stream.
#[test]
fn encode_stream_rejects_version_1() {
    let header = ShortenStreamHeader {
        version: 1,
        filetype: 5,
        channels: 1,
        blocksize: 4,
        maxlpcorder: 0,
        meanblocks: 0,
        skipbytes: 0,
    };
    assert_eq!(
        encode_stream(&header, &[1, 2, 3, 4], &[]).err(),
        Some(EncodeError::UnsupportedVersion(1))
    );
}

/// `packet_chunking` fuzz target, round 453: `flush()` on a truncated
/// stream (bytes delivered, `BLOCK_FN_QUIT` never reached) returned
/// `Ok` and the subsequent `receive_frame` reported a clean `Eof` —
/// the whole-stream wrapper silently dropped the entire stream, and
/// the streaming wrapper silently dropped the unterminated tail. Both
/// wrappers now surface an error from `flush()`.
#[test]
fn flush_on_truncated_stream_errors_instead_of_silent_eof() {
    let header = ShortenStreamHeader {
        version: 2,
        filetype: 5,
        channels: 1,
        blocksize: 8,
        maxlpcorder: 0,
        meanblocks: 0,
        skipbytes: 0,
    };
    let samples: Vec<i32> = (0..64).map(|t| (t * 37 % 500) - 250).collect();
    let bytes = encode_stream(&header, &samples, &[]).expect("encodes");
    let truncated = &bytes[..bytes.len() - 3];

    let params = CodecParameters::audio(CodecId::new(oxideav_shorten::CODEC_ID_STR));
    let tb = TimeBase::new(1, 44_100);

    for (name, mut dec) in [
        ("whole", make_decoder(&params).expect("make_decoder")),
        (
            "streaming",
            make_streaming_decoder(&params).expect("make_streaming_decoder"),
        ),
    ] {
        dec.send_packet(&Packet::new(0, tb, truncated.to_vec()))
            .unwrap_or_else(|e| panic!("{name}: send_packet: {e:?}"));
        // Drain whatever partial frames the streaming shape emits.
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(_)) => {}
                Ok(other) => panic!("{name}: non-audio frame {other:?}"),
                Err(CoreError::NeedMore) | Err(CoreError::Eof) => break,
                Err(e) => panic!("{name}: pre-flush receive_frame: {e:?}"),
            }
        }
        let flushed = dec.flush();
        assert!(
            flushed.is_err(),
            "{name}: flush() on a truncated stream must error, got {flushed:?}"
        );
    }

    // The complete stream still flushes cleanly through both wrappers.
    for (name, mut dec) in [
        ("whole", make_decoder(&params).expect("make_decoder")),
        (
            "streaming",
            make_streaming_decoder(&params).expect("make_streaming_decoder"),
        ),
    ] {
        dec.send_packet(&Packet::new(0, tb, bytes.clone()))
            .unwrap_or_else(|e| panic!("{name}: send_packet: {e:?}"));
        dec.flush()
            .unwrap_or_else(|e| panic!("{name}: flush on complete stream: {e:?}"));
        let mut samples_out = 0u64;
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => samples_out += u64::from(a.samples),
                Ok(other) => panic!("{name}: non-audio frame {other:?}"),
                Err(CoreError::Eof) => break,
                Err(e) => panic!("{name}: post-flush receive_frame: {e:?}"),
            }
        }
        assert_eq!(samples_out, 64, "{name}: all samples recovered");
    }
}

/// `packet_chunking` fuzz target, round 453: `ShortenStreamingDecoder`
/// was the one decode path missing the round-398 decode-time resource
/// bounds — `try_parse_header` allocated per-channel carries, mean
/// estimators, and the pending-round table straight from the header's
/// `ulong()` fields, so an 87-byte stream whose header claimed
/// `H_channels = 1_709_129` drove more than 8 GiB of allocations
/// before a single sample byte was needed. The wrapper now applies the
/// same `check_decode_resource_bounds` guard as `decode_stream` and
/// `StreamDecoder::new`.
#[test]
fn streaming_wrapper_rejects_over_cap_header_resources() {
    // A v2 header claiming 2^21 channels, otherwise minimal. Built
    // with the crate's own writer so the test tracks the wire format.
    let header = ShortenStreamHeader {
        version: 2,
        filetype: 5,
        channels: 1 << 21,
        blocksize: 1,
        maxlpcorder: 0,
        meanblocks: 0,
        skipbytes: 0,
    };
    let mut bytes = Vec::new();
    write_stream_header(&mut bytes, &header).expect("header serialises");
    bytes.extend_from_slice(&[0u8; 16]); // command-stream padding bytes

    let params = CodecParameters::audio(CodecId::new(oxideav_shorten::CODEC_ID_STR));
    let mut dec = make_streaming_decoder(&params).expect("make_streaming_decoder");
    let err = dec
        .send_packet(&Packet::new(0, TimeBase::new(1, 44_100), bytes))
        .expect_err("over-cap H_channels must be rejected, not allocated");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("H_channels"),
        "error should name the offending field: {msg}"
    );
}

/// `packet_chunking` fuzz target, round 453: on a stream whose
/// `BLOCK_FN_QUIT` lands mid channel-round (channel 0 has one more
/// block than channel 1) the whole-stream wrapper rejected the ragged
/// planes at frame packing while the streaming wrapper silently
/// discarded the partial round and reported a clean `Eof`. Both
/// wrappers must now reject the stream.
#[test]
fn quit_mid_channel_round_rejected_by_both_wrappers() {
    let header = ShortenStreamHeader {
        version: 2,
        filetype: 5,
        channels: 2,
        blocksize: 8,
        maxlpcorder: 0,
        meanblocks: 0,
        skipbytes: 0,
    };
    // Header, one ZERO block (lands on channel 0), then QUIT: channel
    // 1 never receives its block of the round.
    let mut bytes = Vec::new();
    write_byte_aligned_prefix(&mut bytes, header.version).expect("prefix");
    let mut w = BitWriter::new();
    write_parameter_block(&mut w, &header);
    write_zero_block(&mut w);
    write_quit_command(&mut w);
    w.pad_to_byte();
    bytes.extend(w.into_bytes());

    let params = CodecParameters::audio(CodecId::new(oxideav_shorten::CODEC_ID_STR));
    let tb = TimeBase::new(1, 44_100);
    for (name, mut dec) in [
        ("whole", make_decoder(&params).expect("make_decoder")),
        (
            "streaming",
            make_streaming_decoder(&params).expect("make_streaming_decoder"),
        ),
    ] {
        let sent = dec.send_packet(&Packet::new(0, tb, bytes.clone()));
        let flushed = dec.flush();
        let received = dec.receive_frame();
        let any_err = sent.is_err()
            || flushed.is_err()
            || matches!(
                received,
                Err(CoreError::Other(_)) | Err(CoreError::InvalidData(_))
            );
        assert!(
            any_err,
            "{name}: a mid-round QUIT must surface an error, got send={sent:?} flush={flushed:?} recv={received:?}"
        );
        assert!(
            received.is_err(),
            "{name}: no frame may be produced from a ragged round"
        );
    }
}

/// `packet_chunking` fuzz target, round 453 (second wave): with
/// `H_blocksize = 0` every pending block is empty, so a
/// `BLOCK_FN_BLOCKSIZE` (or `BLOCK_FN_QUIT`) arriving "mid round"
/// cannot make the planes ragged — the whole-stream wrapper accepts
/// the stream (all channels zero-length) while the streaming wrapper's
/// mid-round guards fired on the empty slots. Both guards now ignore
/// empty pending blocks. The bytes are the fuzz artifact's stream.
#[test]
fn empty_pending_blocks_do_not_trip_mid_round_guards() {
    let bytes: [u8; 48] = [
        0x61, 0x6a, 0x6b, 0x67, 0x02, 0xf7, 0x89, 0x6a, 0x67, 0x61, 0x01, 0x20, 0xff, 0xff, 0xff,
        0xff, 0xff, 0x9f, 0x52, 0xa6, 0xd0, 0x69, 0x34, 0x6a, 0x33, 0x5f, 0x70, 0x00, 0x00, 0x01,
        0x3b, 0x00, 0xf1, 0x61, 0xc9, 0x85, 0x26, 0x0c, 0x18, 0xb0, 0xa1, 0xc1, 0x81, 0x0a, 0xff,
        0xff, 0xdc, 0x72,
    ];
    let whole = oxideav_shorten::decode_stream(&bytes).expect("artifact stream decodes");
    assert!(whole.channels.iter().all(|c| c.is_empty()));

    let params = CodecParameters::audio(CodecId::new(oxideav_shorten::CODEC_ID_STR));
    let tb = TimeBase::new(1, 44_100);
    let mut dec = make_streaming_decoder(&params).expect("make_streaming_decoder");
    dec.send_packet(&Packet::new(0, tb, bytes.to_vec()))
        .expect("streaming wrapper accepts the zero-blocksize stream");
    dec.flush().expect("flush is clean at BLOCK_FN_QUIT");
    assert!(matches!(dec.receive_frame(), Err(CoreError::Eof)));
}
