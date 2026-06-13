//! Integration test for the round-290 whole-stream encode driver
//! [`oxideav_shorten::encode_stream`] — the encoder mirror of
//! [`oxideav_shorten::decode_stream`].
//!
//! `encode_stream` composes the per-block predictor-selection sequencer
//! ([`oxideav_shorten::select_predictor_auto`] /
//! [`oxideav_shorten::write_selected_block`]) with the channel
//! deinterleaving, round-robin cursor, per-channel carry + mean state,
//! tail-block `BLOCK_FN_BLOCKSIZE` override, verbatim prefix, and
//! post-`BLOCK_FN_QUIT` byte alignment into a single end-to-end call.
//! Each test encodes an interleaved `i32` buffer, decodes the produced
//! bytes back through `decode_stream`, and asserts the recovered
//! per-channel samples / header / verbatim prefix are bit-exact with
//! the encoder input.
//!
//! Clean-room provenance:
//!
//! * `docs/audio/shorten/spec/03-block-and-predictor.md` §2 (channel
//!   interleaving + round-robin cursor) + §3.6 (sub-block-size override)
//!   + §3.8 (QUIT) + §3.10 (verbatim prefix).
//! * `docs/audio/shorten/spec/04-function-code-resolution.md` §4.1
//!   (`F2` tail-block override `T12`, `new_bs = 155`).
//! * `docs/audio/shorten/spec/05-state-and-quirks.md` §1 (carry update)
//!   + §2 (mean estimator) + §4 (post-QUIT byte alignment).
//! * `docs/audio/shorten/spec/01-stream-header.md` (header parse +
//!   carry-length derivation).

use oxideav_shorten::{decode_stream, encode_stream, ShortenStreamHeader};

fn header(channels: u32, blocksize: u32, maxlpcorder: u32, meanblocks: u32) -> ShortenStreamHeader {
    ShortenStreamHeader {
        version: 2,
        filetype: 5, // s16lh
        channels,
        blocksize,
        maxlpcorder,
        meanblocks,
        skipbytes: 0,
    }
}

fn deinterleave(samples: &[i32], n: usize) -> Vec<Vec<i32>> {
    let mut planes = vec![Vec::new(); n];
    for (i, &s) in samples.iter().enumerate() {
        planes[i % n].push(s);
    }
    planes
}

/// Encode then decode; assert sample-exact reconstruction.
fn assert_roundtrip(h: &ShortenStreamHeader, samples: &[i32], verbatim: &[u8]) -> Vec<u8> {
    let bytes = encode_stream(h, samples, verbatim).expect("encode succeeds");
    let dec = decode_stream(&bytes).expect("decode succeeds");
    assert_eq!(dec.header, *h, "header round-trips");
    assert_eq!(dec.verbatim, verbatim, "verbatim prefix round-trips");
    assert_eq!(
        dec.channels,
        deinterleave(samples, h.channels as usize),
        "per-channel samples round-trip"
    );
    bytes
}

#[test]
fn mono_many_full_blocks_roundtrips() {
    let h = header(1, 32, 0, 4);
    let samples: Vec<i32> = (0..320).map(|t| ((t * 7) % 200) - 100).collect();
    assert_roundtrip(&h, &samples, &[]);
}

#[test]
fn stereo_with_tail_and_verbatim_roundtrips() {
    // 2 channels × 73 samples each (bs = 16) → 4 full rounds + a
    // 9-sample tail round under a single BLOCKSIZE override.
    let h = header(2, 16, 0, 4);
    let mut samples = Vec::new();
    let mut a = -50i32;
    let mut b = 200i32;
    for t in 0..73i32 {
        a += (t % 5) - 2;
        b += 1 - (t % 3);
        samples.push(a);
        samples.push(b);
    }
    let verbatim = b"RIFF\x00\x00\x00\x00WAVE";
    assert_roundtrip(&h, &samples, verbatim);
}

#[test]
fn five_channels_round_robin_roundtrips() {
    let h = header(5, 8, 0, 0);
    let mut samples = Vec::new();
    for t in 0..47i32 {
        for c in 0..5i32 {
            samples.push((t * (c + 1)) % 64 - 32);
        }
    }
    assert_roundtrip(&h, &samples, &[]);
}

#[test]
fn lpc_recurrence_material_roundtrips_under_maxlpcorder() {
    // s(t) = s(t-1) - s(t-2) is an integer recurrence the auto-derived
    // QLPC vector [1, -1] models exactly; with maxlpcorder = 2 the
    // selector may pick QLPC. The decode must reconstruct it bit-exact.
    let h = header(1, 16, 2, 0);
    let mut s = vec![2i32, 5];
    for t in 2..96 {
        let v = s[t - 1] - s[t - 2];
        s.push(v);
    }
    assert_roundtrip(&h, &s, &[]);
}

#[test]
fn constant_signal_with_mean_window_roundtrips() {
    // A constant signal: once the running-mean window warms up the
    // selector should reach BLOCK_FN_ZERO eligibility; the round-trip
    // must hold whichever predictor is chosen per block.
    let h = header(1, 4, 0, 3);
    let samples = vec![9i32; 60];
    assert_roundtrip(&h, &samples, &[]);
}

#[test]
fn auto_qlpc_stream_is_smaller_than_maxlpcorder_zero_on_lpc_material() {
    // On integer-recurrence material the maxlpcorder=2 encode (QLPC
    // eligible) should not be larger than the maxlpcorder=0 encode
    // (DIFFn only) — the QLPC candidate is added to a superset of the
    // DIFFn candidate set, so it never costs more.
    let mut s = vec![1i32, 1];
    for t in 2..128 {
        let v = (s[t - 1] - s[t - 2]).rem_euclid(97) - 48;
        s.push(v);
    }
    let with_lpc = encode_stream(&header(1, 16, 2, 0), &s, &[]).expect("encode lpc");
    let without_lpc = encode_stream(&header(1, 16, 0, 0), &s, &[]).expect("encode no-lpc");
    assert!(
        with_lpc.len() <= without_lpc.len(),
        "auto-QLPC encode ({} bytes) must not exceed DIFFn-only ({} bytes)",
        with_lpc.len(),
        without_lpc.len()
    );
    // Both must reconstruct identically.
    assert_eq!(
        decode_stream(&with_lpc).unwrap().channels,
        decode_stream(&without_lpc).unwrap().channels
    );
}

#[test]
fn tail_only_single_partial_block_roundtrips() {
    // Fewer samples than one default block: the entire channel is a
    // single tail block under a BLOCKSIZE override.
    let h = header(2, 256, 0, 0);
    let mut samples = Vec::new();
    for t in 0..20i32 {
        samples.push(t - 10);
        samples.push((t - 10) * 2);
    }
    assert_roundtrip(&h, &samples, &[]);
}
