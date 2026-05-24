//! Integration test for the round-7 full-stream decode driver
//! [`oxideav_shorten::decode_stream`].
//!
//! This is the end-to-end counterpart to the per-element pipeline
//! tests (`diff_block_pipeline`, `mean_estimator_pipeline`,
//! `qlpc_block_pipeline`, `housekeeping_pipeline`): instead of
//! re-implementing the orchestration loop inline, it builds a single
//! synthetic v2 stream that exercises every command class in one pass
//! and decodes it through the public driver API.
//!
//! Behavioural anchors:
//! * `docs/audio/shorten/spec/03-block-and-predictor.md` §2 — the
//!   round-robin channel cursor advances on sample-producing commands
//!   and is unchanged by housekeeping commands.
//! * `spec/03` §3.6 / §3.7 — the running sub-block-size and bit-shift
//!   state cells, set by `BLOCK_FN_BLOCKSIZE` / `BLOCK_FN_BITSHIFT`.
//! * `spec/05-state-and-quirks.md` §1.4 — the per-channel carry stores
//!   the pre-shift sample form; the driver applies the bit-shift on
//!   emission only.
//! * `spec/03` §3.10 — verbatim prefix collection.
//! * `spec/03` §3.8 — `BLOCK_FN_QUIT` termination.

use oxideav_shorten::{decode_stream, FNSIZE, MAGIC};

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

fn append_diff_block(out: &mut Vec<u32>, code: u32, energy_encoded: u32, residuals: &[i64]) {
    out.extend(encode_uvar(code, FNSIZE));
    out.extend(encode_uvar(energy_encoded, ENERGYSIZE));
    let width = energy_encoded + 1;
    for &r in residuals {
        out.extend(encode_svar(r, width));
    }
}

const ENERGYSIZE: u32 = 3;

fn assemble(all_bits: &[u32]) -> Vec<u8> {
    let body = pack_bits_msb_first(all_bits);
    let mut buf = Vec::with_capacity(5 + body.len());
    buf.extend_from_slice(&MAGIC);
    buf.push(2);
    buf.extend_from_slice(&body);
    buf
}

/// A single stereo v2 stream exercising VERBATIM + a BITSHIFT +
/// a BLOCKSIZE override + DIFF predictors interleaved across two
/// channels + QUIT, decoded end-to-end via the public driver.
#[test]
fn full_stream_two_channel_with_verbatim_bitshift_blocksize_quit() {
    // Header: filetype = 5 (s16lh), 2 channels, default blocksize = 4,
    // no LPC, no mean, no skip.
    let mut bits = header_param_bits(5, 2, 4, 0, 0, 0);

    // 1. VERBATIM (4-byte "RIFF"-ish prefix). Cursor unchanged.
    bits.extend(encode_uvar(9, FNSIZE)); // VERBATIM = 9
    bits.extend(encode_uvar(4, 5)); // length = 4
    for b in [0x52u8, 0x49, 0x46, 0x46] {
        bits.extend(encode_uvar(b as u32, 8));
    }

    // 2. BITSHIFT bshift = 2 — emitted samples left-shifted by 2,
    //    carry stays in pre-shift form. Cursor unchanged.
    bits.extend(encode_uvar(6, FNSIZE)); // BITSHIFT = 6
    bits.extend(encode_uvar(2, 2)); // bshift = 2 (uvar(BITSHIFTSIZE=2))

    // 3. DIFF1 ch0: residuals [1, 1, 1, 1] over zero carry
    //    -> pre-shift [1, 2, 3, 4] -> emitted (<<2) [4, 8, 12, 16].
    //    Cursor 0 -> 1.
    append_diff_block(&mut bits, 1, 3, &[1, 1, 1, 1]);

    // 4. DIFF0 ch1: residuals [5, 6, 7, 8] (mean = 0)
    //    -> pre-shift [5, 6, 7, 8] -> emitted (<<2) [20, 24, 28, 32].
    //    Cursor 1 -> 0.
    append_diff_block(&mut bits, 0, 3, &[5, 6, 7, 8]);

    // 5. BLOCKSIZE override new_bs = 2. Cursor unchanged (still ch0).
    bits.extend(encode_uvar(5, FNSIZE)); // BLOCKSIZE = 5
    bits.extend(encode_ulong(2, 2));

    // 6. DIFF1 ch0 second block, bs = 2: residuals [0, 0] over the
    //    pre-shift carry (carry[0] = 4 from block 3) -> pre-shift
    //    [4, 4] -> emitted (<<2) [16, 16]. Cursor 0 -> 1.
    append_diff_block(&mut bits, 1, 3, &[0, 0]);

    // 7. QUIT.
    bits.extend(encode_uvar(4, FNSIZE));

    let buf = assemble(&bits);
    let dec = decode_stream(&buf).expect("full stream decodes");

    // Header round-trip.
    assert_eq!(dec.header.channels, 2);
    assert_eq!(dec.header.blocksize, 4);
    assert_eq!(dec.header.filetype, 5);

    // Verbatim prefix collected.
    assert_eq!(dec.verbatim, vec![0x52, 0x49, 0x46, 0x46]);

    // Two channels.
    assert_eq!(dec.channels.len(), 2);
    // ch0: first DIFF1 block (<<2) then the bs=2 second DIFF1 block.
    assert_eq!(dec.channels[0], vec![4, 8, 12, 16, 16, 16]);
    // ch1: the single DIFF0 block (<<2).
    assert_eq!(dec.channels[1], vec![20, 24, 28, 32]);
    assert_eq!(dec.channel_len(0), 6);
    assert_eq!(dec.channel_len(1), 4);
}
