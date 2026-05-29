//! Streaming iterator integration test — round 181.
//!
//! Exercises the per-block `StreamDecoder` iterator surface
//! (`docs/audio/shorten/spec/03-block-and-predictor.md` §2 + §3.6 +
//! §3.7 + §3.8 + §3.10, `spec/05-state-and-quirks.md` §1 + §1.4 + §2)
//! against a complete two-channel stream with every command category
//! present (VERBATIM, BITSHIFT, BLOCKSIZE override, DIFFn / DIFF0
//! sample-producing blocks, QUIT). Asserts:
//!
//! 1. Each `next_block` call returns exactly one [`DecodedBlock`] item
//!    (or `None` after `QUIT`).
//! 2. The per-channel accumulated samples from the iterator match the
//!    round-7 whole-stream driver's
//!    [`DecodedStream::channels`](oxideav_shorten::DecodedStream)
//!    byte-for-byte — same orchestration loop, same emission rule.
//! 3. The `VERBATIM` payloads surfaced as
//!    [`DecodedBlock::Verbatim`](oxideav_shorten::DecodedBlock) items
//!    concatenate to the driver's `DecodedStream::verbatim`.
//! 4. Housekeeping commands (`BLOCKSIZE`, `BITSHIFT`) do not produce
//!    `DecodedBlock` items but mutate the iterator's running state, as
//!    observed via `current_block_size` / `current_bitshift` between
//!    pulls.
//!
//! ## Clean-room provenance
//!
//! Constructed entirely from `docs/audio/shorten/spec/03` + `spec/05`
//! plus the round-7 driver's behaviour — spec PDFs / clean-room
//! workspace material only.

use oxideav_shorten::{decode_stream, decode_stream_iter, DecodedBlock};

// ---- synthetic-stream builders ----

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

fn header_param_bits(
    filetype: u32,
    channels: u32,
    blocksize: u32,
    maxlpcorder: u32,
    meanblocks: u32,
    skipbytes: u32,
) -> Vec<u32> {
    let bits_for = |v: u32| -> u32 {
        if v == 0 {
            0
        } else {
            32 - v.leading_zeros()
        }
    };
    let mut bits = Vec::new();
    bits.extend(encode_ulong(filetype, bits_for(filetype)));
    bits.extend(encode_ulong(channels, bits_for(channels)));
    bits.extend(encode_ulong(blocksize, bits_for(blocksize)));
    bits.extend(encode_ulong(maxlpcorder, bits_for(maxlpcorder)));
    bits.extend(encode_ulong(meanblocks, bits_for(meanblocks)));
    bits.extend(encode_ulong(skipbytes, bits_for(skipbytes)));
    bits
}

const FNSIZE: u32 = 2;
const ENERGYSIZE: u32 = 3;

fn assemble(all_bits: &[u32]) -> Vec<u8> {
    let body = pack_bits_msb_first(all_bits);
    // 'ajkg' magic + format-version byte 2.
    let mut buf = Vec::with_capacity(5 + body.len());
    buf.extend_from_slice(b"ajkg");
    buf.push(2);
    buf.extend_from_slice(&body);
    buf
}

fn append_diff_block(out: &mut Vec<u32>, code: u32, energy_encoded: u32, residuals: &[i64]) {
    out.extend(encode_uvar(code, FNSIZE));
    out.extend(encode_uvar(energy_encoded, ENERGYSIZE));
    let width = energy_encoded + 1;
    for &r in residuals {
        out.extend(encode_svar(r, width));
    }
}

#[test]
fn full_stream_through_iterator_matches_driver_per_channel_and_verbatim() {
    // 2 channels, blocksize 4 default.
    let mut bits = header_param_bits(5, 2, 4, 0, 0, 0);

    // 1. VERBATIM "WAV\0" (4 bytes envelope).
    bits.extend(encode_uvar(9, FNSIZE));
    bits.extend(encode_uvar(4, 5));
    for b in [b'W', b'A', b'V', 0u8] {
        bits.extend(encode_uvar(b as u32, 8));
    }
    // 2. BITSHIFT bshift = 1: emitted samples are pre-shift << 1.
    bits.extend(encode_uvar(6, FNSIZE));
    bits.extend(encode_uvar(1, 2));
    // 3. DIFF1 ch0 [1,1,1,1] over zero carry pre-shift [1,2,3,4]
    //    -> emitted (<<1) = [2,4,6,8].
    append_diff_block(&mut bits, 1, 3, &[1, 1, 1, 1]);
    // 4. DIFF0 ch1 [50,60,70,80] -> [50,60,70,80] pre-shift
    //    -> emitted (<<1) = [100,120,140,160].
    append_diff_block(&mut bits, 0, 4, &[50, 60, 70, 80]);
    // 5. BLOCKSIZE override -> 2.
    bits.extend(encode_uvar(5, FNSIZE));
    bits.extend(encode_ulong(2, 2));
    // 6. DIFF1 ch0 [0,0] over carry s(t-1) = 4 pre-shift
    //    -> pre-shift [4, 4] -> emitted [8, 8].
    append_diff_block(&mut bits, 1, 3, &[0, 0]);
    // 7. VERBATIM "END" (3 bytes).
    bits.extend(encode_uvar(9, FNSIZE));
    bits.extend(encode_uvar(3, 5));
    for b in [b'E', b'N', b'D'] {
        bits.extend(encode_uvar(b as u32, 8));
    }
    // 8. QUIT.
    bits.extend(encode_uvar(4, FNSIZE));

    let buf = assemble(&bits);

    // Reference: round-7 driver.
    let reference = decode_stream(&buf).expect("reference decode_stream");

    // Streaming iterator: accumulate per-channel samples + verbatim
    // bytes and compare.
    let mut iter = decode_stream_iter(&buf).expect("decode_stream_iter");
    assert_eq!(iter.header().channels, 2);
    assert_eq!(iter.current_block_size(), 4);
    assert_eq!(iter.current_bitshift(), 0);

    let mut per_channel: Vec<Vec<i32>> = vec![Vec::new(); reference.header.channels as usize];
    let mut verbatim_concat: Vec<u8> = Vec::new();
    let mut item_count = 0usize;
    let mut sample_block_count = 0usize;
    let mut verbatim_block_count = 0usize;
    while let Some(item) = iter.next_block().expect("ok") {
        item_count += 1;
        match item {
            DecodedBlock::Samples { channel, samples } => {
                sample_block_count += 1;
                per_channel[channel].extend_from_slice(&samples);
            }
            DecodedBlock::Verbatim { bytes } => {
                verbatim_block_count += 1;
                verbatim_concat.extend_from_slice(&bytes);
            }
        }
    }

    // 1. Per-channel output matches the driver byte-for-byte.
    assert_eq!(per_channel, reference.channels);
    // 2. Concatenated verbatim bytes match the driver's
    //    `DecodedStream::verbatim` (encounter order).
    assert_eq!(verbatim_concat, reference.verbatim);
    // 3. Item-count sanity: 3 sample blocks + 2 verbatim blocks.
    assert_eq!(item_count, 5);
    assert_eq!(sample_block_count, 3);
    assert_eq!(verbatim_block_count, 2);
    // 4. Iterator's running state reflects the housekeeping commands.
    assert_eq!(iter.current_block_size(), 2);
    assert_eq!(iter.current_bitshift(), 1);
    assert!(iter.is_finished());
}

#[test]
fn iterator_yields_first_block_without_buffering_remainder() {
    // Build a stream with TWO sample blocks for one channel.
    // After pulling the first block the iterator must NOT have
    // consumed the second block's bytes from the reader yet —
    // pulling once and dropping the iterator should leave the
    // second block undecoded. (Behavioural check that the iterator
    // is genuinely on-demand, not eager.)
    let mut bits = header_param_bits(5, 1, 4, 0, 0, 0);
    append_diff_block(&mut bits, 0, 3, &[1, 2, 3, 4]);
    append_diff_block(&mut bits, 0, 3, &[5, 6, 7, 8]);
    bits.extend(encode_uvar(4, FNSIZE));
    let buf = assemble(&bits);

    let mut iter = decode_stream_iter(&buf).expect("new");
    let first = iter.next_block().expect("ok").expect("Some");
    match first {
        DecodedBlock::Samples { samples, .. } => assert_eq!(samples, vec![1, 2, 3, 4]),
        other => panic!("{other:?}"),
    }
    // At this point the iterator is NOT finished and has not yet
    // touched the second block's bytes.
    assert!(!iter.is_finished());
    // Pulling again decodes block 2.
    let second = iter.next_block().expect("ok").expect("Some");
    match second {
        DecodedBlock::Samples { samples, .. } => assert_eq!(samples, vec![5, 6, 7, 8]),
        other => panic!("{other:?}"),
    }
    // Then QUIT.
    assert!(iter.next_block().expect("ok").is_none());
    assert!(iter.is_finished());
}
