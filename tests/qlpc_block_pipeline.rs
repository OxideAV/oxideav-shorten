//! Integration test exercising the header parse + per-block command
//! dispatch + quantised-LPC predictor reconstruction as a single
//! pipeline.
//!
//! The behavioural anchors here are
//! `docs/audio/shorten/spec/03-block-and-predictor.md` §3.5 (the
//! `BLOCK_FN_QLPC` wire layout `<order> <coef>×order <energy>
//! <residual>×bs` and the general LPC reconstruction `s(t) =
//! Σᵢ aᵢ·s(t-i) + e(t)` applied without scaling), `spec/02` §4.3/§4.4
//! (`LPCQSIZE = 2`, `LPCQUANT = 2`), `spec/05` §3.2 (the QLPC energy
//! field follows the same `+1` residual-width convention as the DIFFn
//! blocks), and `spec/05` §1 (the per-channel sample-history carry's
//! most-recent-first indexing + zero initialisation).
//!
//! The test builds a synthetic v2 stream containing:
//!   1. A v2 header with `H_maxlpcorder = 2` (so the per-channel carry
//!      is `max(3, 2) = 3` samples long and admits an order-2 QLPC
//!      block), `H_channels = 1`, `H_blocksize = 256`.
//!   2. A `BLOCK_FN_QLPC` block (channel 0) carrying an order-2
//!      predictor (`a1 = 2, a2 = -1`, the DIFF2 line-fit), an energy
//!      field, and four residuals.
//!   3. A second `BLOCK_FN_QLPC` block (channel 0) verifying the
//!      per-channel carry hands the first block's last two samples into
//!      the second block's predictor history.
//!   4. A `BLOCK_FN_QUIT` sentinel.
//!
//! The test parses the header, dispatches each command, and verifies
//! the reconstructed samples + the carry hand-off across blocks.

use oxideav_shorten::{
    decode_qlpc_block, parse_stream_header, read_function_code, BitReader, ChannelCarry,
    FunctionCode, ENERGYSIZE, LPCQSIZE, LPCQUANT, MAGIC,
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

/// Append a synthetic `BLOCK_FN_QLPC` command (function code + order +
/// coefficients + energy + residuals) to `out_bits`.
fn append_qlpc_block(
    out_bits: &mut Vec<u32>,
    coefs: &[i64],
    energy_encoded: u32,
    residuals: &[i64],
) {
    out_bits.extend(encode_uvar(7, 2)); // QLPC = 7, FNSIZE = 2
    out_bits.extend(encode_uvar(coefs.len() as u32, LPCQSIZE));
    for &c in coefs {
        out_bits.extend(encode_svar(c, LPCQUANT));
    }
    out_bits.extend(encode_uvar(energy_encoded, ENERGYSIZE));
    let width = energy_encoded + 1;
    for &r in residuals {
        out_bits.extend(encode_svar(r, width));
    }
}

#[test]
fn header_then_qlpc_blocks_round_trip_with_carry_handoff() {
    // --- Stage 1: header bits, H_maxlpcorder = 2. ---
    let mut header_bits = Vec::new();
    header_bits.extend(encode_ulong(5, 3)); // H_filetype = 5
    header_bits.extend(encode_ulong(1, 1)); // H_channels = 1
    header_bits.extend(encode_ulong(256, 9)); // H_blocksize = 256
    header_bits.extend(encode_ulong(2, 2)); // H_maxlpcorder = 2
    header_bits.extend(encode_ulong(0, 0)); // H_meanblocks = 0 (disabled)
    header_bits.extend(encode_ulong(0, 0)); // H_skipbytes = 0

    // --- Stage 2: block-stream bits. ---
    // Block A: order 2 (a1=2, a2=-1, DIFF2 line-fit), residuals
    //   [1, 0, 0, 0] under zero carry:
    //     s0 = 2*0 - 1*0 + 1 = 1
    //     s1 = 2*1 - 1*0 + 0 = 2
    //     s2 = 2*2 - 1*1 + 0 = 3
    //     s3 = 2*3 - 1*2 + 0 = 4
    //   block A = [1, 2, 3, 4]; carry after = [4, 3, 2] (most-recent-first).
    //
    // Block B: order 2 (a1=2, a2=-1), residuals [0, 0] under carry
    //   [4, 3, 2] (s(t-1)=4, s(t-2)=3):
    //     s0 = 2*4 - 1*3 + 0 = 5
    //     s1 = 2*5 - 1*4 + 0 = 6   (s(t-1)=s0=5, s(t-2)=4)
    //   block B = [5, 6].
    let mut block_bits = Vec::new();
    append_qlpc_block(&mut block_bits, &[2, -1], 2, &[1, 0, 0, 0]); // width 3
    append_qlpc_block(&mut block_bits, &[2, -1], 0, &[0, 0]); // width 1
    block_bits.extend(encode_uvar(4, 2)); // QUIT = 4

    // --- Stage 3: assemble + parse. ---
    let mut all_bits = Vec::with_capacity(header_bits.len() + block_bits.len());
    all_bits.extend(&header_bits);
    all_bits.extend(&block_bits);
    let body = pack_bits_msb_first(&all_bits);
    let mut buf = Vec::with_capacity(5 + body.len());
    buf.extend_from_slice(&MAGIC);
    buf.push(2);
    buf.extend_from_slice(&body);

    let parsed = parse_stream_header(&buf).expect("header must parse");
    assert_eq!(parsed.header.channels, 1);
    assert_eq!(parsed.header.maxlpcorder, 2);
    assert_eq!(parsed.header.sample_history_carry_len(), 3);

    // --- Stage 4: skip the header bits via a fresh reader. ---
    let post_version = &buf[5..];
    let mut reader = BitReader::new(post_version);
    let header_bit_count = parsed.bits_consumed_after_v;
    let mut remaining = header_bit_count;
    while remaining > 0 {
        let chunk = remaining.min(32);
        let _ = reader.read_bits(chunk).expect("burn header bits");
        remaining -= chunk;
    }

    // --- Stage 5: dispatch the two QLPC blocks (single channel). ---
    let mut carry = ChannelCarry::new(parsed.header.sample_history_carry_len() as usize);
    let bs = 4u32; // override block size locally for block A.

    let fc_a = read_function_code(&mut reader).expect("block A code classifies");
    assert_eq!(fc_a, FunctionCode::Qlpc);
    let block_a = decode_qlpc_block(&mut reader, bs, &carry).expect("QLPC block A decodes");
    assert_eq!(block_a, vec![1, 2, 3, 4]);
    carry.update_after_block(&block_a);
    assert_eq!(carry.at(0), 4);
    assert_eq!(carry.at(1), 3);
    assert_eq!(carry.at(2), 2);

    let fc_b = read_function_code(&mut reader).expect("block B code classifies");
    assert_eq!(fc_b, FunctionCode::Qlpc);
    let block_b = decode_qlpc_block(&mut reader, 2, &carry).expect("QLPC block B decodes");
    assert_eq!(block_b, vec![5, 6]);

    // --- Stage 6: terminal QUIT. ---
    let fc_quit = read_function_code(&mut reader).expect("quit code classifies");
    assert_eq!(fc_quit, FunctionCode::Quit);
}
