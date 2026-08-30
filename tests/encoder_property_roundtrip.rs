//! Property tests for the whole-stream encoder: sample-exact round
//! trips over a deterministic random parameter grid.
//!
//! The fuzz harness (`fuzz/fuzz_targets/encode_roundtrip.rs`) drives
//! the same contract coverage-guided and open-ended; this suite pins a
//! reproducible slice of that parameter space in CI so a regression
//! is caught by `cargo test` without needing a fuzzer:
//!
//! * every contract-valid `(version, channels, blocksize, maxlpcorder,
//!   meanblocks, filetype, skipbytes)` header encodes successfully;
//! * `decode_stream` reconstructs the input samples exactly
//!   (`(s >> bshift) << bshift` for the lossy path), with the header
//!   and verbatim prefix intact;
//! * the constant-memory `StreamDecoder` agrees with the whole-stream
//!   driver on every stream;
//! * `stream_proper_len` equals the produced byte length and the QUIT
//!   padding is `spec/05` §4-conformant.

use oxideav_shorten::{
    decode_stream, encode_stream, encode_stream_lossy, DecodedBlock, ShortenStreamHeader,
    StreamDecoder,
};

/// Deterministic xorshift64* PRNG (same shape as the robustness
/// suite's) — reproducible grids, no external dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform-ish in `0..=n`.
    fn upto(&mut self, n: u64) -> u64 {
        self.next_u64() % (n + 1)
    }
}

/// One round trip: encode `samples` under `header` (+ optional lossy
/// `bshift`), decode through BOTH public decode paths, and assert the
/// full contract.
fn check_roundtrip(
    header: &ShortenStreamHeader,
    samples: &[i32],
    verbatim: &[u8],
    bshift: u32,
    what: &str,
) {
    let bytes = if bshift == 0 {
        encode_stream(header, samples, verbatim)
    } else {
        encode_stream_lossy(header, samples, verbatim, bshift)
    }
    .unwrap_or_else(|e| panic!("{what}: encode failed: {e:?}"));

    let n = header.channels as usize;
    let expected: Vec<Vec<i32>> = (0..n)
        .map(|c| {
            samples
                .iter()
                .skip(c)
                .step_by(n)
                .map(|&s| {
                    if bshift == 0 {
                        s
                    } else {
                        (s >> bshift) << bshift
                    }
                })
                .collect()
        })
        .collect();

    let dec = decode_stream(&bytes).unwrap_or_else(|e| panic!("{what}: decode failed: {e:?}"));
    assert_eq!(dec.header, *header, "{what}: header");
    assert_eq!(dec.verbatim, verbatim, "{what}: verbatim prefix");
    assert_eq!(dec.channels, expected, "{what}: samples");
    assert_eq!(dec.stream_proper_len, bytes.len(), "{what}: no trailer");
    assert!(
        dec.quit_padding.is_spec_conformant(),
        "{what}: QUIT padding"
    );

    let mut it = StreamDecoder::new(&bytes).unwrap_or_else(|e| panic!("{what}: iter: {e:?}"));
    let mut planes: Vec<Vec<i32>> = vec![Vec::new(); n];
    let mut vb = Vec::new();
    loop {
        match it.next_block() {
            Ok(Some(DecodedBlock::Samples { channel, samples })) => planes[channel].extend(samples),
            Ok(Some(DecodedBlock::Verbatim { bytes })) => vb.extend(bytes),
            Ok(None) => break,
            Err(e) => panic!("{what}: streaming decode failed: {e:?}"),
        }
    }
    assert_eq!(planes, expected, "{what}: streaming samples");
    assert_eq!(vb, verbatim, "{what}: streaming verbatim");
}

/// Random parameter grid, lossless. 160 configurations spanning
/// channels 1..=6, block sizes 1..=300, LPC orders 0..=12, mean
/// windows 0..=6, bit depths 4..=24, verbatim prefixes 0..=40 bytes,
/// and 0..=600 total samples of white noise at the chosen depth.
#[test]
fn random_grid_lossless_sample_exact() {
    let mut rng = Rng::new(0x5348_4f52_5445_4e01);
    for case in 0..160u32 {
        let channels = 1 + rng.upto(5) as u32;
        let header = ShortenStreamHeader {
            version: 2 + rng.upto(1) as u8,
            filetype: rng.upto(200) as u32,
            channels,
            blocksize: 1 + rng.upto(299) as u32,
            maxlpcorder: rng.upto(12) as u32,
            meanblocks: rng.upto(6) as u32,
            skipbytes: rng.upto(90) as u32,
        };
        let depth = 4 + rng.upto(20) as u32;
        let verbatim: Vec<u8> = (0..rng.upto(40)).map(|_| rng.next_u64() as u8).collect();
        let total = (rng.upto(600) as usize) / channels as usize * channels as usize;
        let samples: Vec<i32> = (0..total)
            .map(|_| {
                let masked = (rng.next_u64() as u32) & ((1u32 << depth) - 1);
                ((masked << (32 - depth)) as i32) >> (32 - depth)
            })
            .collect();
        check_roundtrip(&header, &samples, &verbatim, 0, &format!("case {case}"));
    }
}

/// Random parameter grid, lossy: bshift 1..=12 over 16-bit-ish noise
/// plus structured (ramp / sine-free polynomial) signals, asserting
/// the `(s >> bshift) << bshift` reconstruction.
#[test]
fn random_grid_lossy_quantised_exact() {
    let mut rng = Rng::new(0x5348_4f52_5445_4e02);
    for case in 0..60u32 {
        let channels = 1 + rng.upto(3) as u32;
        let header = ShortenStreamHeader {
            version: 2,
            filetype: 5,
            channels,
            blocksize: 1 + rng.upto(127) as u32,
            maxlpcorder: rng.upto(8) as u32,
            meanblocks: rng.upto(4) as u32,
            skipbytes: 0,
        };
        let bshift = 1 + rng.upto(11) as u32;
        let total = (rng.upto(400) as usize) / channels as usize * channels as usize;
        let structured = case % 3 == 0;
        let samples: Vec<i32> = (0..total)
            .map(|t| {
                if structured {
                    // Quadratic ramp: exercises DIFF2/DIFF3 selection.
                    let t = t as i64;
                    ((t * t / 40 - 3 * t) % 30_000) as i32
                } else {
                    (rng.next_u64() as u32 as i32) >> 16
                }
            })
            .collect();
        check_roundtrip(
            &header,
            &samples,
            &[],
            bshift,
            &format!("lossy case {case}"),
        );
    }
}

/// Degenerate shapes the grid may under-sample: empty streams, one
/// sample, constant blocks (ZERO-eligible), full-scale extremes at
/// every depth-boundary, and blocksize-1 streams with high LPC orders.
#[test]
fn degenerate_shapes_sample_exact() {
    let h = |channels, blocksize, maxlpcorder, meanblocks| ShortenStreamHeader {
        version: 2,
        filetype: 5,
        channels,
        blocksize,
        maxlpcorder,
        meanblocks,
        skipbytes: 0,
    };
    // Empty stream (header + QUIT only).
    check_roundtrip(&h(1, 256, 0, 0), &[], &[], 0, "empty mono");
    check_roundtrip(&h(4, 16, 3, 2), &[], b"hdr", 0, "empty quad + verbatim");
    // Single sample.
    check_roundtrip(&h(1, 256, 0, 0), &[12345], &[], 0, "one sample");
    // Constant blocks — ZERO with a zero mean, then DIFF selection
    // against a non-zero running mean.
    check_roundtrip(&h(2, 32, 0, 4), &vec![7; 256], &[], 0, "constant stereo");
    check_roundtrip(&h(1, 64, 0, 0), &vec![0; 640], &[], 0, "silence");
    // i16 extremes with sign alternation (worst-case DIFF residuals).
    let extremes: Vec<i32> = (0..128)
        .map(|t| if t % 2 == 0 { 32767 } else { -32768 })
        .collect();
    check_roundtrip(&h(1, 16, 4, 0), &extremes, &[], 0, "i16 extremes");
    // 24-bit extremes.
    let deep: Vec<i32> = (0..96)
        .map(|t| if t % 3 == 0 { 8_388_607 } else { -8_388_608 })
        .collect();
    check_roundtrip(&h(2, 8, 2, 3), &deep, &[], 0, "24-bit extremes");
    // blocksize 1 with an LPC order that exceeds the block length.
    let tiny: Vec<i32> = (0..40).map(|t| t * 31 - 600).collect();
    check_roundtrip(&h(1, 1, 8, 0), &tiny, &[], 0, "bs=1 lpc=8");
    // Lossy extremes: bshift larger than the signal amplitude.
    check_roundtrip(
        &h(1, 32, 0, 0),
        &[1, -1, 2, -2, 3, -3],
        &[],
        12,
        "bshift 12 tiny",
    );
    // Max legal bshift.
    let wide: Vec<i32> = (0..64).map(|t| (t - 32) << 20).collect();
    check_roundtrip(&h(1, 16, 0, 0), &wide, &[], 31, "bshift 31");
}
