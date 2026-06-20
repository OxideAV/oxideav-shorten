//! Fixture-anchored byte-exact decode test — fixture `F4`'s leading
//! `BLOCK_FN_ZERO` + `BLOCK_FN_DIFF0` zero-sample run.
//!
//! The behavioural anchor is the reference-encoder byte-exact leading-zero run
//! pinned in `docs/audio/shorten/spec/04-function-code-resolution.md`
//! §6.1 + footnote `T13`, cross-checked against
//! `spec/05-state-and-quirks.md` §2.1/§2.3/§2.4 (running-mean
//! initialisation and its effect on `BLOCK_FN_ZERO` / `BLOCK_FN_DIFF0`).
//!
//! `F4` is a 44.1 kHz stereo s16 stream whose decoded PCM begins with a
//! long run of zeros. Per `T13` the bit stream (after the verbatim
//! prefix) is `80` consecutive `fn = 8` (`BLOCK_FN_ZERO`) commands —
//! `40` per channel at the default blocksize `256` = `10,240` zero
//! samples per channel — followed by an `fn = 0` (`BLOCK_FN_DIFF0`)
//! block for ch0 whose first `159` residuals are zero (extending ch0's
//! zero run to `10,399`), then ch1's first DIFF0 with `351` leading
//! zero residuals (extending ch1's run to `10,591`). the reference encoder decodes all
//! of those leading samples to exactly zero.
//!
//! The two state rules this pins are:
//!   * `BLOCK_FN_ZERO` emits `bs` samples all equal to the channel's
//!     running mean `mu_chan`, which is `0` under the zero-initialised
//!     mean buffer at stream start (spec/05 §2.1/§2.4).
//!   * `BLOCK_FN_DIFF0` reconstructs `s(t) = e0(t) + mu_chan`; with
//!     `mu_chan = 0` and zero residuals the samples stay zero
//!     (spec/05 §2.3).
//!
//! The test reproduces a scaled-down but structurally identical leading
//! run (`H_meanblocks = 4` as in `F4`, a handful of ZERO blocks per
//! channel, then a DIFF0 block of all-zero residuals) and asserts every
//! decoded sample is zero — the reference-encoder ground truth for the leading
//! run. It uses a `BLOCK_FN_BLOCKSIZE` override to keep the block count
//! small while preserving the ZERO-then-DIFF0 ordering and the
//! zero-mean reconstruction semantics.

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

/// F4 parameter block (`spec/04` §6.1: filetype `s16lh = 5`, two
/// channels, blocksize 256, no LPC, `H_meanblocks = 4`, no skipbytes),
/// but with `H_channels` and `H_blocksize` caller-chosen so the leading
/// run is small. `H_meanblocks = 4` is kept exactly as `F4` so the
/// zero-mean initialisation under test matches the fixture.
fn f4_header_param_bits(channels: u32, blocksize: u32) -> Vec<u32> {
    let mut bits = Vec::new();
    bits.extend(encode_ulong(5, bits_for(5))); // H_filetype = s16lh
    bits.extend(encode_ulong(channels, bits_for(channels)));
    bits.extend(encode_ulong(blocksize, bits_for(blocksize)));
    bits.extend(encode_ulong(0, bits_for(0))); // H_maxlpcorder = 0
    bits.extend(encode_ulong(4, bits_for(4))); // H_meanblocks = 4 (as F4)
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

const FN_DIFF0: u32 = 0;
const FN_ZERO: u32 = 8;
const FN_BLOCKSIZE: u32 = 5;
const FN_QUIT: u32 = 4;

#[test]
fn f4_leading_zero_then_diff0_run_decodes_all_zero() {
    // Structurally mirror F4's leading run: per channel, several
    // BLOCK_FN_ZERO blocks then a BLOCK_FN_DIFF0 block of zero
    // residuals. Two channels (as F4), blocksize overridden to 4 so the
    // run is compact.
    let channels = 2u32;
    let bs = 4u32;
    let zero_blocks_per_channel = 3u32; // ≪ F4's 40, same semantics
    let diff0_residuals = vec![0i64; bs as usize];

    let mut bits = f4_header_param_bits(channels, bs);

    // BLOCKSIZE override to `bs` (the header default is 256; F4's
    // blocks run at the default, but we shrink it for a compact test).
    bits.extend(encode_uvar(FN_BLOCKSIZE, 2));
    bits.extend(encode_ulong(bs, bits_for(bs)));

    // Per T13: a run of ZERO commands. They round-robin through the
    // channels exactly like sample-producing blocks (each ZERO block
    // belongs to the current cursor channel and advances it), so emit
    // `zero_blocks_per_channel * channels` of them.
    for _ in 0..(zero_blocks_per_channel * channels) {
        bits.extend(encode_uvar(FN_ZERO, 2));
    }

    // Then one DIFF0 block per channel with all-zero residuals
    // (energy-field encoded value 0 → residual width 1, spec/05 §3 /
    // T13's "energy-field encoded value 0 (true mantissa width 1)").
    for _ in 0..channels {
        bits.extend(encode_uvar(FN_DIFF0, 2));
        bits.extend(encode_uvar(0, 3)); // ENERGYSIZE = 3, encoded 0 → width 1
        for &r in &diff0_residuals {
            bits.extend(encode_svar(r, 1));
        }
    }

    bits.extend(encode_uvar(FN_QUIT, 2));
    let stream = assemble(&bits);

    let decoded = decode_stream(&stream).expect("F4 leading-run synthetic stream decodes");
    assert_eq!(decoded.channels.len(), channels as usize);

    let expect_len = (zero_blocks_per_channel * bs + bs) as usize;
    for (ch, plane) in decoded.channels.iter().enumerate() {
        assert_eq!(
            plane.len(),
            expect_len,
            "channel {ch}: {zero_blocks_per_channel} ZERO blocks + one DIFF0 block at bs={bs}"
        );
        assert!(
            plane.iter().all(|&s| s == 0),
            "spec/04 §6.1 / T13 + spec/05 §2.1/§2.3/§2.4: BLOCK_FN_ZERO emits \
             mu_chan=0 and the following DIFF0 with zero residuals stays zero — \
             channel {ch} must be all-zero, got {plane:?}"
        );
    }
}

#[test]
fn f4_single_zero_block_emits_blocksize_zero_samples() {
    // Tighter pin on the ZERO command alone: one ZERO block on a single
    // channel emits exactly `bs` zero samples (mu_chan = 0 at init,
    // spec/05 §2.4) and nothing else before QUIT.
    let bs = 7u32;
    let mut bits = f4_header_param_bits(1, bs);
    bits.extend(encode_uvar(FN_BLOCKSIZE, 2));
    bits.extend(encode_ulong(bs, bits_for(bs)));
    bits.extend(encode_uvar(FN_ZERO, 2));
    bits.extend(encode_uvar(FN_QUIT, 2));
    let stream = assemble(&bits);

    let decoded = decode_stream(&stream).expect("single ZERO block decodes");
    assert_eq!(decoded.channels.len(), 1);
    assert_eq!(
        decoded.channels[0],
        vec![0i32; bs as usize],
        "a single BLOCK_FN_ZERO at stream start emits bs zero samples (mu_chan=0)"
    );
}
