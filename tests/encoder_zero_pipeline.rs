//! Integration test for the round-18 `BLOCK_FN_ZERO` sentinel encoder.
//!
//! Builds full Shorten files (header + ZERO blocks across channels +
//! optional VERBATIM splice + optional DIFFn / QLPC mixed blocks +
//! QUIT + zero-pad to byte boundary) using only the public encoder
//! surface, then decodes them through the round-7 whole-stream
//! driver [`oxideav_shorten::decode_stream`] and confirms the
//! recovered per-channel samples are byte-exact with the expected
//! reconstruction (`bs` copies of `μ_chan` per ZERO block).
//!
//! Clean-room provenance:
//!
//! * `docs/audio/shorten/spec/03-block-and-predictor.md` §3.9 +
//!   `docs/audio/shorten/spec/04-function-code-resolution.md` §6
//!   (`BLOCK_FN_ZERO` wire layout `<fn=8>` — a bare function-code
//!   field with no further payload; the command advances the channel
//!   cursor on decode).
//! * `docs/audio/shorten/spec/05-state-and-quirks.md` §2.4 (the
//!   decoder's reconstruction emits `bs` samples all equal to
//!   `μ_chan`, the channel's current running-mean estimate; zero
//!   when `H_meanblocks = 0`).
//! * `docs/audio/shorten/spec/02-variable-length-coding.md` §1 +
//!   §2.1 (MSB-first `uvar(n)` primitive; the only field the ZERO
//!   command writes).

use oxideav_shorten::{
    decode_stream, min_energy_for_diff0, write_byte_aligned_prefix, write_diff0_block,
    write_parameter_block, write_quit_command, write_verbatim_block, write_zero_block, BitWriter,
    ShortenStreamHeader,
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

#[test]
fn mono_single_zero_block_decodes_to_block_of_zeros() {
    // Mono, H_meanblocks = 0 → mu_chan = 0. One ZERO block produces
    // a single block of `bs` zero samples.
    let header = synth_header(2, 5, 1, 6, 0, 0, 0);

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);
    write_zero_block(&mut writer);
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    assert_eq!(dec.header, header);
    assert_eq!(dec.channels.len(), 1);
    assert_eq!(dec.channels[0], vec![0i32; 6]);
}

#[test]
fn stereo_one_zero_per_channel_round_robin() {
    // Stereo: each channel emits one ZERO block. Round-robin order
    // is ch0 then ch1; the decoder reconstructs `bs` zeros per
    // channel.
    let header = synth_header(2, 5, 2, 4, 0, 0, 0);

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);
    write_zero_block(&mut writer);
    write_zero_block(&mut writer);
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    assert_eq!(dec.channels.len(), 2);
    assert_eq!(dec.channels[0], vec![0i32; 4]);
    assert_eq!(dec.channels[1], vec![0i32; 4]);
}

#[test]
fn many_consecutive_zero_blocks_mono() {
    // Mono, 16 consecutive ZERO blocks → 16 * bs zero samples.
    let header = synth_header(2, 5, 1, 8, 0, 0, 0);
    let n_blocks = 16usize;

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);
    for _ in 0..n_blocks {
        write_zero_block(&mut writer);
    }
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    let expected_len = n_blocks * 8;
    assert_eq!(dec.channels[0].len(), expected_len);
    assert!(dec.channels[0].iter().all(|&s| s == 0));
}

#[test]
fn zero_block_then_diff0_continues_same_channel() {
    // Mono, ZERO followed by a DIFF0 block. Channel cursor wraps to
    // ch0 (only one channel). Decoder reconstructs `bs` zeros from
    // the ZERO block, then `bs` samples reconstructed from the DIFF0
    // residuals (mu_chan = 0 so s(t) = e0(t)).
    let header = synth_header(2, 5, 1, 4, 0, 0, 0);
    let diff0_samples: Vec<i32> = vec![5, -3, 7, -1];
    let residuals_i64: Vec<i64> = diff0_samples.iter().map(|&s| s as i64).collect();
    let energy = min_energy_for_diff0(&residuals_i64).expect("fits");

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);
    write_zero_block(&mut writer);
    write_diff0_block(&mut writer, energy, &diff0_samples, 0).expect("diff0");
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    let mut expected: Vec<i32> = vec![0i32; 4];
    expected.extend(diff0_samples.iter());
    assert_eq!(dec.channels[0], expected);
}

#[test]
fn zero_block_with_verbatim_envelope() {
    // Header + VERBATIM prefix + ZERO + QUIT. The verbatim prefix
    // should be recovered byte-for-byte, and the per-channel sample
    // stream should consist of `bs` zeros.
    let header = synth_header(2, 5, 1, 6, 0, 0, 12);
    let preamble = b"RIFF\x00\x00\x00\x00WAVE";
    assert_eq!(preamble.len(), 12);

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);
    write_verbatim_block(&mut writer, preamble).expect("verbatim");
    write_zero_block(&mut writer);
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    assert_eq!(dec.header, header);
    assert_eq!(dec.verbatim, preamble.to_vec());
    assert_eq!(dec.channels.len(), 1);
    assert_eq!(dec.channels[0], vec![0i32; 6]);
}

#[test]
fn three_channels_alternating_zero_and_diff0() {
    // Three channels: ch0 emits ZERO, ch1 emits DIFF0, ch2 emits
    // ZERO. After the round-robin completes, ch0 has `bs` zeros,
    // ch1 has the DIFF0 reconstruction, ch2 has `bs` zeros.
    let header = synth_header(2, 5, 3, 4, 0, 0, 0);
    let ch1_samples: Vec<i32> = vec![1, 2, -1, 0];
    let energy = min_energy_for_diff0(&ch1_samples.iter().map(|&s| s as i64).collect::<Vec<_>>())
        .expect("fits");

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);
    write_zero_block(&mut writer);
    write_diff0_block(&mut writer, energy, &ch1_samples, 0).expect("diff0");
    write_zero_block(&mut writer);
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    assert_eq!(dec.channels.len(), 3);
    assert_eq!(dec.channels[0], vec![0i32; 4]);
    assert_eq!(dec.channels[1], ch1_samples);
    assert_eq!(dec.channels[2], vec![0i32; 4]);
}

#[test]
fn zero_block_each_costs_five_bits_packed_tightly() {
    // Twelve consecutive ZERO commands pack into a tight bit stream:
    // each is 5 bits, total = 60 bits, which round-trips through the
    // decoder as 12 * bs zero samples per channel.
    //
    // This test verifies the bit-level packing claim from the
    // round-18 commit message and the encoder unit test: 5 bits per
    // ZERO command, no per-command padding.
    let header = synth_header(2, 5, 1, 3, 0, 0, 0);

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);
    let bits_before = writer.bits_written();
    for _ in 0..12 {
        write_zero_block(&mut writer);
    }
    let bits_after = writer.bits_written();
    assert_eq!(bits_after - bits_before, 12 * 5);

    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode");
    assert_eq!(dec.channels[0], vec![0i32; 12 * 3]);
}
