//! Integration test for the round-244 `BLOCK_FN_BLOCKSIZE` housekeeping
//! encoder.
//!
//! Builds full Shorten files (header + DIFFn full-size blocks +
//! BLOCKSIZE override + DIFFn tail block of the new size +
//! QUIT + zero-pad to byte boundary) using only the public encoder
//! surface, then decodes them through the round-7 whole-stream driver
//! [`oxideav_shorten::decode_stream`] and confirms the recovered
//! per-channel samples are byte-exact with the encoded inputs.
//!
//! Clean-room provenance:
//!
//! * `docs/audio/shorten/spec/03-block-and-predictor.md` §3.6
//!   (`BLOCK_FN_BLOCKSIZE` wire layout `<fn=5> <new_bs>` where
//!   `new_bs` is `ulong()`; the command does not advance the channel
//!   cursor; subsequent predictor commands produce blocks of `new_bs`
//!   samples per channel until the next override or end-of-stream).
//! * `docs/audio/shorten/spec/04-function-code-resolution.md` §4
//!   (function-code numeric value 5; the `F2` T12 behavioural anchor
//!   pins the single tail-block override at command index 11,377
//!   with `new_bs = 155`).
//! * `docs/audio/shorten/spec/02-variable-length-coding.md` §3
//!   (the two-stage `ulong()` form: `uvar(ULONGSIZE = 2)` over the
//!   per-value mantissa width followed by `uvar(width)` over the
//!   value itself).

use oxideav_shorten::{
    decode_stream, min_energy_for_diff1, write_blocksize_command, write_byte_aligned_prefix,
    write_diff1_block, write_parameter_block, write_quit_command, BitWriter, ShortenStreamHeader,
};

fn synth_header(
    version: u8,
    filetype: u32,
    channels: u32,
    blocksize: u32,
    maxlpcorder: u32,
    meanblocks: u32,
    skipbytes: u32,
) -> ShortenStreamHeader {
    ShortenStreamHeader {
        version,
        filetype,
        channels,
        blocksize,
        maxlpcorder,
        meanblocks,
        skipbytes,
    }
}

/// Helper: compute first-differences of `samples`, anchored at the
/// `seed` previous-sample value. Returns the residuals stream that the
/// order-1 polynomial-difference predictor consumes.
fn diff1_residuals(samples: &[i32], seed: i32) -> Vec<i64> {
    let mut prev = seed as i64;
    samples
        .iter()
        .map(|&s| {
            let r = s as i64 - prev;
            prev = s as i64;
            r
        })
        .collect()
}

/// Compose: byte-aligned magic + version + parameter block + per-channel
/// DIFF1 block of the default `H_blocksize` + BLOCKSIZE override +
/// per-channel DIFF1 tail block of `new_bs` samples + QUIT + zero-pad.
///
/// Returns `(stream_bytes, per_channel_full_samples, per_channel_tail_samples)`.
fn build_blocksize_stream(
    header: &ShortenStreamHeader,
    new_bs: u32,
    full_samples_per_channel: Vec<Vec<i32>>,
    tail_samples_per_channel: Vec<Vec<i32>>,
) -> Vec<u8> {
    assert_eq!(full_samples_per_channel.len() as u32, header.channels);
    assert_eq!(tail_samples_per_channel.len() as u32, header.channels);
    for full in &full_samples_per_channel {
        assert_eq!(full.len() as u32, header.blocksize);
    }
    for tail in &tail_samples_per_channel {
        assert_eq!(tail.len() as u32, new_bs);
    }

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, header);

    // Round-robin per spec/03 §2: ch0 full block, ch1 full block, …
    // ch0..N gets one full-size DIFF1 block; the per-channel carry
    // seeds at zero per spec/05 §1.1 so the first residual is `s(0)`.
    let mut carries: Vec<oxideav_shorten::ChannelCarry> = (0..header.channels)
        .map(|_| oxideav_shorten::ChannelCarry::new(header.maxlpcorder as usize))
        .collect();

    for ch in 0..header.channels as usize {
        let samples = &full_samples_per_channel[ch];
        let residuals = diff1_residuals(samples, carries[ch].at(0));
        let e = min_energy_for_diff1(&residuals).expect("DIFF1 energy fits");
        write_diff1_block(&mut writer, e, samples, &carries[ch]).expect("write DIFF1 full");
        carries[ch].update_after_block(samples);
    }

    // BLOCKSIZE override — does not advance the channel cursor.
    write_blocksize_command(&mut writer, new_bs).expect("write BLOCKSIZE");

    // Per-channel tail block at the new sub-block-size.
    for ch in 0..header.channels as usize {
        let samples = &tail_samples_per_channel[ch];
        let residuals = diff1_residuals(samples, carries[ch].at(0));
        let e = min_energy_for_diff1(&residuals).expect("DIFF1 energy fits tail");
        write_diff1_block(&mut writer, e, samples, &carries[ch]).expect("write DIFF1 tail");
        carries[ch].update_after_block(samples);
    }

    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());
    out
}

#[test]
fn mono_blocksize_override_shrinks_tail_block_round_trips() {
    // Mono, default H_blocksize = 8, tail block new_bs = 3 — a
    // shrinking override that mirrors F2's T12 tail-block partition
    // shape (partial trailing block smaller than H_blocksize).
    let header = synth_header(2, 5, 1, 8, 0, 0, 0);
    let new_bs = 3u32;
    let full = vec![vec![10i32, 11, 13, 16, 20, 25, 31, 38]];
    let tail = vec![vec![42i32, 47, 53]];

    let stream = build_blocksize_stream(&header, new_bs, full.clone(), tail.clone());
    let dec = decode_stream(&stream).expect("decode");

    assert_eq!(dec.header, header);
    assert_eq!(dec.channels.len(), 1);
    let mut expected = full[0].clone();
    expected.extend_from_slice(&tail[0]);
    assert_eq!(dec.channels[0], expected);
}

#[test]
fn stereo_blocksize_override_round_robin_preserves_channel_layout() {
    // Stereo: full block per channel, BLOCKSIZE, tail block per
    // channel. Verify the channel-cursor non-advancement rule of
    // spec/03 §3.6 — the tail block resumes on ch0, exactly where
    // the second full block would have started.
    let header = synth_header(2, 3, 2, 6, 0, 0, 0);
    let new_bs = 2u32;
    let full = vec![
        vec![100i32, 102, 105, 109, 114, 120],
        vec![-50i32, -45, -39, -32, -24, -15],
    ];
    let tail = vec![vec![127i32, 135], vec![-5i32, 6]];

    let stream = build_blocksize_stream(&header, new_bs, full.clone(), tail.clone());
    let dec = decode_stream(&stream).expect("decode");

    assert_eq!(dec.channels.len(), 2);
    let mut expected_ch0 = full[0].clone();
    expected_ch0.extend_from_slice(&tail[0]);
    let mut expected_ch1 = full[1].clone();
    expected_ch1.extend_from_slice(&tail[1]);
    assert_eq!(dec.channels[0], expected_ch0);
    assert_eq!(dec.channels[1], expected_ch1);
}

#[test]
fn blocksize_override_to_default_h_blocksize_is_a_noop_in_effect() {
    // BLOCKSIZE override that resets new_bs to the same value as
    // H_blocksize is admissible per spec/03 §3.6 — it's a no-op in
    // effect but a valid wire-format command. Round-trips byte-exact.
    let header = synth_header(2, 5, 1, 4, 0, 0, 0);
    let new_bs = 4u32; // same as H_blocksize
    let full = vec![vec![1i32, 2, 3, 5]];
    let tail = vec![vec![8i32, 13, 21, 34]];

    let stream = build_blocksize_stream(&header, new_bs, full.clone(), tail.clone());
    let dec = decode_stream(&stream).expect("decode");

    let mut expected = full[0].clone();
    expected.extend_from_slice(&tail[0]);
    assert_eq!(dec.channels[0], expected);
}

#[test]
fn blocksize_anchor_value_155_round_trips_through_decode_stream() {
    // The exact spec/04 §4.1 T12 anchor value: new_bs = 155. Use
    // H_blocksize = 4 to keep the test fixture small while still
    // exercising the actual override value the F2 fixture carries
    // at command 11,377. The tail block is 155 samples of a smooth
    // ramp (so DIFF1's first differences are bounded and fit a
    // natural energy width).
    let header = synth_header(2, 5, 1, 4, 0, 0, 0);
    let new_bs = 155u32;
    let full = vec![vec![0i32, 1, 2, 3]];
    let tail: Vec<i32> = (0..155i32).map(|i| 4 + i).collect();

    let stream = build_blocksize_stream(&header, new_bs, full.clone(), vec![tail.clone()]);
    let dec = decode_stream(&stream).expect("decode");

    let mut expected = full[0].clone();
    expected.extend(tail.iter().copied());
    assert_eq!(dec.channels[0].len(), 4 + 155);
    assert_eq!(dec.channels[0], expected);
}

#[test]
fn blocksize_override_to_one_sample_blocks_round_trips() {
    // Degenerate but spec-admissible: new_bs = 1 — single-sample
    // blocks. spec/03 §3.6 does not pin a lower bound above zero;
    // BLOCKSIZE_MAX is an implementation safety cap.
    let header = synth_header(2, 5, 1, 4, 0, 0, 0);
    let new_bs = 1u32;
    let full = vec![vec![7i32, 9, 11, 13]];
    let tail = vec![vec![15i32]];

    let stream = build_blocksize_stream(&header, new_bs, full.clone(), tail.clone());
    let dec = decode_stream(&stream).expect("decode");

    let mut expected = full[0].clone();
    expected.extend_from_slice(&tail[0]);
    assert_eq!(dec.channels[0], expected);
}
