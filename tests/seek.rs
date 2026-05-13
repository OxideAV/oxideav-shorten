//! Seek integration tests for the Shorten container demuxer.
//!
//! Round 8 — adds lazy frame-index construction + binary-search
//! `seek_to` on the [`Demuxer`] trait. Shorten has no native seek
//! table (TR.156 §"Hydrogenaudio: lacks ... support for seeking"); the
//! demuxer builds a `(pts, byte_offset)` index on first seek by
//! walking the FN command stream, then caches it for subsequent
//! calls.
//!
//! These tests cover:
//!
//! * `seek_to_zero_resets_to_start` — pts=0 lands at sample 0.
//! * `seek_at_block_boundary_lands_exact` — a requested pts that
//!   matches an indexed entry returns that exact pts.
//! * `seek_mid_block_lands_at_containing_block` — a requested pts
//!   that falls between two indexed entries returns the entry just
//!   before (the largest pts ≤ target).
//! * `seek_past_end_clamps` — a requested pts beyond the last
//!   indexed entry returns the last indexed entry.
//! * `frame_index_cached_on_second_seek_call` — the FN scan runs
//!   exactly once across multiple seeks (scan_count stays at 1).
//!
//! No external fixture: the existing production encoder generates a
//! multi-block stream programmatically. ffmpeg is not required.

use oxideav_core::Demuxer;
use oxideav_shorten::{__open_demuxer_typed, encode, EncoderConfig, Filetype};

/// Build a small `.shn` fixture: 1-channel S16Le sine-like ramp,
/// `blocksize = 64`, 16 blocks worth of samples. Yields a stream
/// with predictable per-block pts boundaries (64, 128, 192, …) and a
/// non-trivial frame index across the FRAME_INDEX_STRIDE = 10
/// granularity.
fn build_fixture_mono(n_blocks: usize, blocksize: u32) -> Vec<u8> {
    let total = (blocksize as usize) * n_blocks;
    let samples: Vec<i32> = (0..total).map(|i| (i as i32 % 8192) - 4096).collect();
    let cfg = EncoderConfig::new(Filetype::S16Le, 1).with_blocksize(blocksize);
    encode(&cfg, &samples).expect("encode mono fixture")
}

/// Build a stereo fixture — exercises the per-channel block-round
/// bookkeeping in the FN scan. Each round-trip of 2 blocks advances
/// pts by `blocksize` (one block per channel).
fn build_fixture_stereo(n_rounds: usize, blocksize: u32) -> Vec<u8> {
    let total_per_ch = (blocksize as usize) * n_rounds;
    let total = total_per_ch * 2;
    // Interleaved: c0 ramps up, c1 ramps down.
    let mut samples: Vec<i32> = Vec::with_capacity(total);
    for i in 0..total_per_ch {
        samples.push((i as i32 % 4096) - 2048);
        samples.push(((total_per_ch - i) as i32 % 4096) - 2048);
    }
    let cfg = EncoderConfig::new(Filetype::S16Le, 2).with_blocksize(blocksize);
    encode(&cfg, &samples).expect("encode stereo fixture")
}

#[test]
fn seek_to_zero_resets_to_start() {
    let buf = build_fixture_mono(16, 64);
    let mut demux = __open_demuxer_typed(buf).expect("open demuxer");

    // Drain one packet to consume the stream's initial packet.
    let _ = demux.next_packet().expect("first packet");
    // EOF on the next call.
    assert!(demux.next_packet().is_err());

    // Seek back to pts=0.
    let landed = demux.seek_to(0, 0).expect("seek to 0");
    assert_eq!(landed, 0, "seek-to-zero should land at pts 0");
    assert_eq!(demux.next_pts(), 0);

    // Now next_packet should re-emit at pts=0.
    let pkt = demux.next_packet().expect("packet after seek-to-zero");
    assert_eq!(pkt.pts, Some(0));
    assert!(pkt.flags.keyframe);
}

#[test]
fn seek_at_block_boundary_lands_exact() {
    // 16 mono blocks of 64 samples → block-round boundaries at
    // 0, 64, 128, …, 960. With FRAME_INDEX_STRIDE = 10 the indexed
    // entries are at pts 0 and 640 (the round at round-count 10).
    let buf = build_fixture_mono(16, 64);
    let mut demux = __open_demuxer_typed(buf).expect("open demuxer");

    // First seek builds the index.
    let landed_first = demux.seek_to(0, 640).expect("seek");
    assert_eq!(
        landed_first, 640,
        "exact match against the indexed pts=640 should round-trip"
    );

    // The frame index should contain at least an entry at pts 0 and
    // one at pts 640.
    let indexed = demux.frame_index_pts().expect("index built");
    assert!(
        indexed.contains(&0),
        "frame index should include the start anchor pts=0, got {indexed:?}"
    );
    assert!(
        indexed.contains(&640),
        "frame index should include the round-10 anchor pts=640, got {indexed:?}"
    );
}

#[test]
fn seek_mid_block_lands_at_containing_block() {
    // With FRAME_INDEX_STRIDE = 10 + blocksize = 64 (mono), the
    // index has entries at pts 0 and 640. A seek to pts=700 (between
    // 640 and the end at 1024) should land on the containing
    // indexed entry, which is 640.
    let buf = build_fixture_mono(16, 64);
    let mut demux = __open_demuxer_typed(buf).expect("open demuxer");

    let landed = demux.seek_to(0, 700).expect("seek");
    assert_eq!(
        landed, 640,
        "mid-block seek should land at the largest indexed pts <= target (640 <= 700)"
    );

    // A seek to pts=100 (between 0 and 640) should land on 0.
    let landed2 = demux.seek_to(0, 100).expect("seek 2");
    assert_eq!(
        landed2, 0,
        "seek between pts=0 and the first stride boundary should land at 0"
    );
}

#[test]
fn seek_past_end_clamps() {
    let buf = build_fixture_mono(16, 64);
    let mut demux = __open_demuxer_typed(buf).expect("open demuxer");

    // Total per-channel samples = 16 * 64 = 1024. Seek to pts=10_000
    // — way past the end. Should clamp to the last indexed entry.
    let landed = demux.seek_to(0, 10_000).expect("seek past end");
    let indexed = demux.frame_index_pts().expect("index built");
    let last_indexed = *indexed.last().expect("non-empty index");
    assert_eq!(
        landed, last_indexed,
        "seek past end should clamp to the last indexed pts ({last_indexed})"
    );
    assert!(
        landed <= 1024,
        "landed pts must be within the stream's per-channel sample count"
    );
}

#[test]
fn seek_negative_pts_clamps_to_zero() {
    let buf = build_fixture_mono(16, 64);
    let mut demux = __open_demuxer_typed(buf).expect("open demuxer");

    // The Demuxer trait takes an i64 pts; negative values are a
    // common consumer-side encoding for "before start". We clamp.
    let landed = demux.seek_to(0, -500).expect("seek negative");
    assert_eq!(landed, 0, "negative pts should clamp to 0");
}

#[test]
fn seek_rejects_invalid_stream_index() {
    let buf = build_fixture_mono(4, 64);
    let mut demux = __open_demuxer_typed(buf).expect("open demuxer");
    let err = demux.seek_to(1, 0).unwrap_err();
    // Shouldn't trigger an index build either.
    assert!(
        demux.frame_index_pts().is_none(),
        "rejected seek must not build the index"
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("stream index 1 out of range") || msg.contains("only stream 0"),
        "error message should explain the bad stream index, got {msg:?}"
    );
}

#[test]
fn frame_index_cached_on_second_seek_call() {
    let buf = build_fixture_mono(32, 64);
    let mut demux = __open_demuxer_typed(buf).expect("open demuxer");

    // Before any seek, the scan counter is zero.
    assert_eq!(
        demux.scan_count(),
        0,
        "scan counter must be zero before the first seek_to call"
    );

    // First seek builds the index — scan counter advances to 1.
    let _ = demux.seek_to(0, 100).expect("first seek");
    assert_eq!(
        demux.scan_count(),
        1,
        "first seek_to must trigger exactly one FN-command scan"
    );

    // Second seek does NOT rescan — counter stays at 1.
    let _ = demux.seek_to(0, 1000).expect("second seek");
    assert_eq!(
        demux.scan_count(),
        1,
        "second seek_to must reuse the cached index (no rescan)"
    );

    // Third seek to a different target — also no rescan.
    let _ = demux.seek_to(0, 0).expect("third seek");
    assert_eq!(
        demux.scan_count(),
        1,
        "third seek_to must reuse the cached index (no rescan)"
    );

    // The index itself is non-trivial — we built more than just the
    // pts=0 anchor.
    let indexed = demux.frame_index_pts().expect("index built");
    assert!(
        indexed.len() >= 2,
        "32-block fixture with stride=10 should produce >= 2 index entries, got {}",
        indexed.len()
    );
    let offsets = demux.frame_index_byte_offsets().expect("offsets built");
    assert_eq!(
        indexed.len(),
        offsets.len(),
        "pts list and byte-offset list must be the same length"
    );
    // Byte offsets must be monotone non-decreasing (the scan walks
    // forward through the stream).
    for w in offsets.windows(2) {
        assert!(
            w[0] <= w[1],
            "byte offsets must be monotone non-decreasing across the index: {:?}",
            offsets
        );
    }
}

#[test]
fn stereo_seek_advances_pts_in_per_channel_units() {
    // Stereo: each "block-round" of 2 commands advances pts by
    // `blocksize`. With FRAME_INDEX_STRIDE = 10 and blocksize = 32,
    // the first indexed pts past 0 is 320.
    let buf = build_fixture_stereo(20, 32);
    let mut demux = __open_demuxer_typed(buf).expect("open demuxer");

    let landed = demux.seek_to(0, 320).expect("seek");
    assert_eq!(
        landed, 320,
        "stereo seek should land on the per-channel-sample boundary at the round-10 stride"
    );

    let indexed = demux.frame_index_pts().expect("index built");
    assert!(
        indexed.contains(&0),
        "stereo index should anchor at pts=0, got {indexed:?}"
    );
    assert!(
        indexed.contains(&320),
        "stereo index should include the round-10 stride at pts=320, got {indexed:?}"
    );
}
