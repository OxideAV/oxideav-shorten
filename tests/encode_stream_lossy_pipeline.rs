//! Integration test for the round-345 lossy `-q N` whole-stream encode
//! path [`oxideav_shorten::encode_stream_lossy`] — the encoder-side dual
//! of the decoder's `BLOCK_FN_BITSHIFT` handling.
//!
//! `encode_stream_lossy` emits a single `BLOCK_FN_BITSHIFT` command after
//! the verbatim prefix and encodes the residuals of the **quantised**
//! sample stream (each sample arithmetic-right-shifted by `bshift`),
//! exactly the form TR.156's `-q quantisation level` option produces
//! (fixtures `F5..F8` with `-q ∈ {1, 4, 8, 12}`, `spec/04` §3.1). The
//! decoder restores the magnitude with its dual left-shift on emission,
//! so the round-trip is near-lossless: `decode(encode_lossy(s, N))`
//! equals `s` with its `N` low-order bits cleared, i.e. `(s >> N) << N`.
//!
//! Each test encodes an interleaved `i32` buffer at a given `bshift`,
//! decodes the produced bytes back through `decode_stream`, and asserts
//! the recovered per-channel samples equal the quantised expectation.
//!
//! Clean-room provenance:
//!
//! * `docs/audio/shorten/spec/03-block-and-predictor.md` §3.7
//!   (`BLOCK_FN_BITSHIFT` per-stream bit-shift; does not advance the
//!   channel cursor) + §2 (channel interleaving) + §3.6 (tail-block
//!   override) + §3.8 (QUIT).
//! * `docs/audio/shorten/spec/04-function-code-resolution.md` §3 + §3.1
//!   (`BLOCK_FN_BITSHIFT = 6`; `F5..F8` `-q N` anchors `T10`).
//! * `docs/audio/shorten/spec/05-state-and-quirks.md` §1.4 (the carry +
//!   mean estimator consume the pre-shift / quantised samples).

use oxideav_shorten::{decode_stream, encode_stream, encode_stream_lossy, ShortenStreamHeader};

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

/// `(s >> bshift) << bshift` applied per channel — the decoder's
/// reconstruction of a lossily-encoded sample (`spec/03` §3.7).
fn quantise_planes(samples: &[i32], n: usize, bshift: u32) -> Vec<Vec<i32>> {
    deinterleave(samples, n)
        .into_iter()
        .map(|p| {
            p.into_iter()
                .map(|s| {
                    if bshift == 0 {
                        s
                    } else {
                        (s >> bshift) << bshift
                    }
                })
                .collect()
        })
        .collect()
}

/// Encode lossily, decode, assert the per-channel reconstruction equals
/// the quantised expectation. Returns the encoded byte length.
fn assert_lossy_roundtrip(
    h: &ShortenStreamHeader,
    samples: &[i32],
    verbatim: &[u8],
    bshift: u32,
) -> usize {
    let bytes = encode_stream_lossy(h, samples, verbatim, bshift).expect("lossy encode succeeds");
    let dec = decode_stream(&bytes).expect("decode succeeds");
    assert_eq!(dec.header, *h, "header round-trips");
    assert_eq!(dec.verbatim, verbatim, "verbatim prefix round-trips");
    assert_eq!(
        dec.channels,
        quantise_planes(samples, h.channels as usize, bshift),
        "per-channel samples round-trip to (s >> {bshift}) << {bshift}"
    );
    bytes.len()
}

#[test]
fn mono_q1_through_q12_all_roundtrip_to_quantised() {
    let h = header(1, 32, 0, 0);
    // Material with real content in the low order bits so each shift
    // genuinely loses information.
    let samples: Vec<i32> = (0..320).map(|t| ((t * 137) % 30000) - 15000).collect();
    for bshift in [1u32, 2, 4, 8, 12] {
        assert_lossy_roundtrip(&h, &samples, &[], bshift);
    }
}

#[test]
fn stereo_with_tail_and_verbatim_lossy_roundtrips() {
    // 2 channels × 73 samples (bs = 16): 4 full rounds + a 9-sample tail
    // round under a BLOCKSIZE override, all under a q=3 bit-shift.
    let h = header(2, 16, 0, 4);
    let mut samples = Vec::new();
    let mut a = -5000i32;
    let mut b = 20000i32;
    for t in 0..73i32 {
        a += ((t * 13) % 200) - 100;
        b += 70 - ((t * 17) % 140);
        samples.push(a);
        samples.push(b);
    }
    let verbatim = b"RIFF\x00\x00\x00\x00WAVE";
    assert_lossy_roundtrip(&h, &samples, verbatim, 3);
}

#[test]
fn lossy_q0_equals_lossless_bytes_exactly() {
    // bshift = 0 must produce byte-identical output to encode_stream.
    let h = header(2, 16, 0, 3);
    let mut samples = Vec::new();
    for t in 0..120i32 {
        samples.push((t * 19) % 4000 - 2000);
        samples.push(1000 - (t * 7) % 3000);
    }
    let lossless = encode_stream(&h, &samples, b"hdr").expect("lossless");
    let lossy0 = encode_stream_lossy(&h, &samples, b"hdr", 0).expect("lossy0");
    assert_eq!(lossless, lossy0, "bshift=0 lossy output equals lossless");
}

#[test]
fn higher_bitshift_compresses_noisy_signal_more() {
    // A high-amplitude pseudo-random signal carries little inter-sample
    // correlation, so the residual entropy is dominated by the sample
    // magnitude. Dropping low-order bits via a larger bshift strictly
    // reduces the residual magnitude, so the encoded stream gets smaller
    // (or at worst equal) as bshift grows. Verify the monotone trend
    // across q ∈ {0, 2, 4, 6, 8}.
    let h = header(1, 64, 0, 0);
    let mut state: u32 = 0x1234_5678;
    let samples: Vec<i32> = (0..1024)
        .map(|_| {
            // xorshift32 — deterministic, no external deps.
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as i32 % 65536) - 32768
        })
        .collect();

    let mut prev = usize::MAX;
    for bshift in [0u32, 2, 4, 6, 8] {
        let len = assert_lossy_roundtrip(&h, &samples, &[], bshift);
        assert!(
            len <= prev,
            "q={bshift} encode ({len} bytes) must not exceed the previous-q size ({prev} bytes)"
        );
        prev = len;
    }
}

#[test]
fn lossy_with_qlpc_material_roundtrips_to_quantised() {
    // QLPC operates on the quantised (post-shift) stream; the near-lossless
    // round-trip still holds to the (s >> bshift) << bshift expectation.
    let h = header(1, 16, 2, 0);
    let mut s = vec![3000i32, 7000];
    for t in 2..96 {
        let v = (s[t - 1] - s[t - 2] / 3).clamp(-30000, 30000);
        s.push(v);
    }
    assert_lossy_roundtrip(&h, &s, &[], 4);
}

#[test]
fn lossy_negative_signal_floors_toward_negative_infinity() {
    // Arithmetic right shift floors toward -inf; verify the round-trip
    // matches that convention on a fully-negative signal.
    let h = header(1, 8, 0, 0);
    let samples: Vec<i32> = (0..64).map(|t| -((t * 91) % 12000) - 1).collect();
    assert_lossy_roundtrip(&h, &samples, &[], 5);
}
