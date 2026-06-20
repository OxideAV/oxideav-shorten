//! Fixture-anchored byte-exact decode test — fixture `F2`'s
//! `BLOCK_FN_VERBATIM` → `BLOCK_FN_BITSHIFT(7)` → `BLOCK_FN_DIFF1`
//! opening chain.
//!
//! The behavioural anchor is the reference-encoder byte-exact `u8p` PCM pinned in
//! `docs/audio/shorten/spec/04-function-code-resolution.md` §3.2 +
//! footnote `T11`. `F2` is an 8-bit-unsigned (`u8p`) AIFC-formatted
//! Shorten stream whose opening per-block chain after the verbatim
//! prefix is `BLOCK_FN_BITSHIFT` with `bshift = 7`, then a
//! `BLOCK_FN_DIFF1` block whose energy-field encoded value is `0`
//! (residual width `1`) and whose first residuals are
//! `[0, 0, 0, 0, 1, …]`. Under the DIFF1 reconstruction with a
//! zero-initialised carry the predictor-domain samples are
//! `[0, 0, 0, 0, 1, …]`; the decoder then emits each sample
//! left-shifted by `bshift = 7`, so the **first non-zero ch0 output
//! sample is at index 4 with value `128 = 1 << 7`** — byte-for-byte
//! with the reference encoder's `u8p` PCM for `F2`.
//!
//! Only the leading samples `T11` actually pins are asserted: the four
//! leading zeros and the value `128` at index 4. `T11`'s "first
//! non-zero ch0 sample at index 4 with value 128" is the firm reference encoder
//! anchor; samples past index 4 are not pinned by the footnote (it ends
//! the residual list with `…`), so the test does not assert them.
//!
//! This exercises three pieces through the public `decode_stream` path
//! that no other fixture vector in the suite covers together: the
//! verbatim-prefix carry, the `BLOCK_FN_BITSHIFT` state, and the
//! `(sample << bshift)` emit of `spec/03` §3.7 / `spec/05` §1.4.

use oxideav_shorten::{decode_stream, MAGIC};

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
    let u32_val = u32::try_from(u).expect("residual fits in u32");
    encode_uvar(u32_val, n)
}

fn encode_ulong(value: u32, w: u32) -> Vec<u32> {
    let mut bits = encode_uvar(w, 2);
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

/// F2 parameter block (`spec/04` §3.2 + `spec/05` §6: filetype
/// `u8 = 2`, single channel for the ch0 vector, blocksize 256, no LPC).
/// `H_meanblocks` is set to 0 here so the BITSHIFT+DIFF1 chain under
/// test is mean-invariant (DIFF1 is mean-invariant regardless, but a
/// zero mean buffer keeps the focus on the bitshift emit).
fn f2_header_param_bits(blocksize: u32) -> Vec<u32> {
    let mut bits = Vec::new();
    bits.extend(encode_ulong(2, bits_for(2))); // H_filetype = u8
    bits.extend(encode_ulong(1, bits_for(1))); // H_channels = 1 (F2 ch0)
    bits.extend(encode_ulong(blocksize, bits_for(blocksize)));
    bits.extend(encode_ulong(0, bits_for(0))); // H_maxlpcorder = 0
    bits.extend(encode_ulong(0, bits_for(0))); // H_meanblocks = 0
    bits.extend(encode_ulong(0, bits_for(0))); // H_skipbytes = 0
    bits
}

fn assemble(all_bits: &[u32]) -> Vec<u8> {
    let body = pack_bits_msb_first(all_bits);
    let mut buf = Vec::with_capacity(5 + body.len());
    buf.extend_from_slice(&MAGIC);
    buf.push(2);
    buf.extend_from_slice(&body);
    buf
}

const FN_DIFF1: u32 = 1;
const FN_VERBATIM: u32 = 9;
const FN_BITSHIFT: u32 = 6;
const FN_BLOCKSIZE: u32 = 5;
const FN_QUIT: u32 = 4;

/// The bshift value pinned by `T11` for `F2`.
const F2_BSHIFT: u32 = 7;

/// The leading DIFF1 residuals pinned by `T11` (energy-field 0 → width
/// 1, so each residual is 0 or 1 in the svar(1) sense; the value 1 at
/// index 4 is the one that produces the 128 output).
const F2_DIFF1_RESIDUALS: [i64; 6] = [0, 0, 0, 0, 1, 0];

#[test]
fn f2_bitshift7_diff1_first_nonzero_ch0_sample_is_128_at_index_4() {
    // Build: VERBATIM (a few opaque bytes mirroring F2's AIFC preamble,
    // value-agnostic), BITSHIFT(7), one DIFF1 block at a small blocksize
    // carrying the pinned residuals, QUIT.
    let verbatim_bytes: [u8; 4] = [0x46, 0x4f, 0x52, 0x4d]; // "FORM" — opaque to the codec
    let bs = F2_DIFF1_RESIDUALS.len() as u32;

    let mut bits = f2_header_param_bits(256);

    // VERBATIM: length uvar(5), then `length` × uvar(8) bytes.
    bits.extend(encode_uvar(FN_VERBATIM, 2));
    bits.extend(encode_uvar(verbatim_bytes.len() as u32, 5));
    for &b in &verbatim_bytes {
        bits.extend(encode_uvar(b as u32, 8));
    }

    // BITSHIFT(7): fn, then bshift as uvar(BITSHIFTSIZE = 2).
    bits.extend(encode_uvar(FN_BITSHIFT, 2));
    bits.extend(encode_uvar(F2_BSHIFT, 2));

    // BLOCKSIZE override to `bs` so the DIFF1 block carries exactly the
    // pinned residuals.
    bits.extend(encode_uvar(FN_BLOCKSIZE, 2));
    bits.extend(encode_ulong(bs, bits_for(bs)));

    // DIFF1 block: fn, energy-field encoded value 0 (residual width 1),
    // then the residuals at svar(1).
    bits.extend(encode_uvar(FN_DIFF1, 2));
    bits.extend(encode_uvar(0, 3)); // ENERGYSIZE = 3, encoded 0 → width 1
    for &r in &F2_DIFF1_RESIDUALS {
        bits.extend(encode_svar(r, 1));
    }

    bits.extend(encode_uvar(FN_QUIT, 2));
    let stream = assemble(&bits);

    let decoded = decode_stream(&stream).expect("F2 bitshift chain decodes");

    // The verbatim prefix is carried through opaquely.
    assert_eq!(
        decoded.verbatim, verbatim_bytes,
        "BLOCK_FN_VERBATIM bytes are carried opaquely through decode"
    );

    assert_eq!(decoded.channels.len(), 1);
    let ch0 = &decoded.channels[0];
    assert!(
        ch0.len() >= 5,
        "the DIFF1 block must produce at least 5 ch0 samples"
    );

    // T11's firm reference-encoder anchor: first four samples zero, sample 4 = 128.
    assert_eq!(
        &ch0[..5],
        &[0, 0, 0, 0, 128],
        "spec/04 §3.2 / T11: under BITSHIFT(7) the first non-zero ch0 \
         sample is at index 4 with value 128 = 1 << 7 (reference-encoder u8p PCM)"
    );
}

#[test]
fn f2_bshift_emit_is_left_shift_by_bshift() {
    // Direct corroboration of the `<<bshift` emit independent of the
    // DIFF1 chain: a single DIFF0 block under BITSHIFT(7) with residual
    // `1` emits `1 << 7 = 128`. (DIFF0 sample = e0 + mu_chan; mu_chan=0
    // with H_meanblocks=0.) This pins the shift direction and magnitude
    // that T11 relies on, without depending on the open post-index-4
    // residual values.
    let mut bits = f2_header_param_bits(256);
    bits.extend(encode_uvar(FN_BITSHIFT, 2));
    bits.extend(encode_uvar(F2_BSHIFT, 2));
    bits.extend(encode_uvar(FN_BLOCKSIZE, 2));
    bits.extend(encode_ulong(1, bits_for(1))); // blocksize 1
    bits.extend(encode_uvar(0, 2)); // FN_DIFF0
    bits.extend(encode_uvar(0, 3)); // energy encoded 0 → width 1
    bits.extend(encode_svar(1, 1)); // residual 1
    bits.extend(encode_uvar(FN_QUIT, 2));
    let stream = assemble(&bits);

    let decoded = decode_stream(&stream).expect("single bitshifted DIFF0 decodes");
    assert_eq!(
        decoded.channels[0],
        vec![128],
        "BITSHIFT(7) emits sample << 7: residual 1 → output 128"
    );
}
