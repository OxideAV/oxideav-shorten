//! Integration test exercising the header parse + per-block command
//! dispatch + DIFFn predictor reconstruction as a single pipeline.
//!
//! The behavioural anchors here are
//! `docs/audio/shorten/spec/03-block-and-predictor.md` §3.1..§3.4
//! (the four polynomial-difference predictor recurrences),
//! `spec/05-state-and-quirks.md` §1 (the per-channel sample-history
//! carry's most-recent-first indexing + zero initialisation), and
//! `spec/05` §3 / `T15` (the residual mantissa width = encoded
//! energy + 1).
//!
//! The test builds a synthetic v2 stream containing:
//!   1. A v2 header matching fixture `F1`'s field choices
//!      (`H_blocksize = 256`, `H_channels = 2`, `H_maxlpcorder = 0`).
//!   2. A `BLOCK_FN_DIFF1` block (channel 0) carrying four residuals.
//!   3. A `BLOCK_FN_DIFF1` block (channel 1) carrying four residuals.
//!   4. A `BLOCK_FN_DIFF1` block (channel 0, second block) — the
//!      channel-cursor round-robins back per `spec/03` §2 — to verify
//!      the per-channel carry buffer correctly hands `s(t-1)` from
//!      channel 0's first block into its second block.
//!   5. A `BLOCK_FN_QUIT` sentinel.
//!
//! The test then parses the header, dispatches each command, applies
//! the round-robin channel cursor, and verifies the per-channel
//! samples reconstruct exactly. The integration is the smallest
//! end-to-end test that covers every round-3-landed piece (header +
//! function-code dispatch + DIFFn kernel + carry + multi-channel
//! cursor).

use oxideav_shorten::{
    decode_diff_block, parse_stream_header, read_function_code, BitReader, ChannelCarry,
    FunctionCode, PolyOrder, ENERGYSIZE, FNSIZE, MAGIC,
};

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

/// Append a synthetic `BLOCK_FN_DIFF1` command (function code + energy
/// + residuals) to `out_bits`.
///
/// `energy_encoded` is the value the energy field carries; residuals
/// are decoded as `svar(energy_encoded + 1)` per `spec/05` §3.
fn append_diff1_block(out_bits: &mut Vec<u32>, energy_encoded: u32, residuals: &[i64]) {
    out_bits.extend(encode_uvar(1, FNSIZE)); // DIFF1 = 1
    out_bits.extend(encode_uvar(energy_encoded, ENERGYSIZE));
    let width = energy_encoded + 1;
    for &r in residuals {
        out_bits.extend(encode_svar(r, width));
    }
}

#[test]
fn header_then_diff1_blocks_round_trip_with_channel_cursor() {
    // --- Stage 1: build the header bits (mirrors F1's parameters). ---
    let mut header_bits = Vec::new();
    header_bits.extend(encode_ulong(5, 3)); // H_filetype = 5
    header_bits.extend(encode_ulong(2, 2)); // H_channels = 2
    header_bits.extend(encode_ulong(256, 9)); // H_blocksize = 256
    header_bits.extend(encode_ulong(0, 0)); // H_maxlpcorder = 0
    header_bits.extend(encode_ulong(4, 3)); // H_meanblocks = 4
    header_bits.extend(encode_ulong(0, 0)); // H_skipbytes = 0
    assert_eq!(header_bits.len(), 43);

    // --- Stage 2: build the block-stream bits. ---
    // ch0 block A: residuals [10, 5, -3, 2] under DIFF1 with zero carry
    //   -> samples [10, 15, 12, 14]; carry[0] = 14 after.
    // ch1 block A: residuals [-1, -1, 2, 3] under DIFF1 with zero carry
    //   -> samples [-1, -2, 0, 3]; carry[0] = 3 after.
    // ch0 block B: residuals [7, -2, -5, 1] under DIFF1 with carry[0]=14
    //   -> samples [14+7=21, 21-2=19, 19-5=14, 14+1=15].
    let mut block_bits = Vec::new();
    append_diff1_block(&mut block_bits, 3, &[10, 5, -3, 2]);
    append_diff1_block(&mut block_bits, 3, &[-1, -1, 2, 3]);
    append_diff1_block(&mut block_bits, 3, &[7, -2, -5, 1]);
    block_bits.extend(encode_uvar(4, FNSIZE)); // QUIT = 4

    // --- Stage 3: assemble + parse the header. ---
    let mut all_bits = Vec::with_capacity(header_bits.len() + block_bits.len());
    all_bits.extend(&header_bits);
    all_bits.extend(&block_bits);
    let body = pack_bits_msb_first(&all_bits);
    let mut buf = Vec::with_capacity(5 + body.len());
    buf.extend_from_slice(&MAGIC);
    buf.push(2);
    buf.extend_from_slice(&body);

    let parsed = parse_stream_header(&buf).expect("header must parse");
    assert_eq!(parsed.header.channels, 2);
    assert_eq!(parsed.header.blocksize, 256);
    assert_eq!(parsed.header.maxlpcorder, 0);
    assert_eq!(parsed.bits_consumed_after_v, 43);

    // --- Stage 4: re-open a BitReader and skip the 43 header bits. ---
    let post_version = &buf[5..];
    let mut reader = BitReader::new(post_version);
    let _ = reader.read_bits(32).expect("burn the first 32 header bits");
    let _ = reader
        .read_bits(parsed.bits_consumed_after_v - 32)
        .expect("burn remaining header bits");

    // --- Stage 5: dispatch each block command with the round-robin
    // channel cursor per `spec/03` §2. With H_channels = 2 the cursor
    // alternates 0, 1, 0 across the three DIFF blocks.
    let mut carries: Vec<ChannelCarry> = (0..parsed.header.channels)
        .map(|_| ChannelCarry::new(parsed.header.sample_history_carry_len() as usize))
        .collect();
    let mut cursor: usize = 0;
    let bs = parsed.header.blocksize.min(4); // override locally to 4
                                             // for test fixtures.

    // We override `bs` here rather than emitting a BLOCK_FN_BLOCKSIZE
    // command (that command's payload-state mutation isn't wired up in
    // round 3); the integration test deliberately exercises only the
    // DIFFn kernel + the per-channel cursor.
    let mut decoded: Vec<Vec<i32>> = Vec::new();
    let mut decoded_channels: Vec<usize> = Vec::new();
    for _ in 0..3 {
        let fc = read_function_code(&mut reader).expect("command code must classify");
        assert_eq!(fc, FunctionCode::Diff1);
        let order = PolyOrder::from_function_code(fc).expect("Diff1 maps to order 1");
        let block = decode_diff_block(&mut reader, order, bs, &carries[cursor])
            .expect("DIFFn payload must decode");
        assert_eq!(block.len(), bs as usize);
        carries[cursor].update_after_block(&block);
        decoded.push(block);
        decoded_channels.push(cursor);
        cursor = (cursor + 1) % (parsed.header.channels as usize);
    }
    // After three DIFF blocks with 2 channels: cursor = 0, 1, 0 then
    // wraps to 1. So decoded_channels = [0, 1, 0].
    assert_eq!(decoded_channels, vec![0, 1, 0]);

    // Verify the reconstructions.
    assert_eq!(decoded[0], vec![10, 15, 12, 14]);
    assert_eq!(decoded[1], vec![-1, -2, 0, 3]);
    assert_eq!(decoded[2], vec![21, 19, 14, 15]);

    // --- Stage 6: the next command should be QUIT. ---
    let fc_quit = read_function_code(&mut reader).expect("quit code must classify");
    assert_eq!(fc_quit, FunctionCode::Quit);
}
