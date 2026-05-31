//! Integration test for the round-13 `BLOCK_FN_DIFF0` predictor
//! encoder.
//!
//! Builds full Shorten files (header + DIFF0 blocks across channels +
//! QUIT + zero-pad to byte boundary) using only the public encoder
//! surface, then decodes them through the round-7 whole-stream driver
//! [`oxideav_shorten::decode_stream`] and confirms the recovered
//! per-channel samples are byte-exact with the input.
//!
//! Clean-room provenance: `docs/audio/shorten/spec/03-block-and-predictor.md`
//! §3.1 (DIFF0 wire layout) + `spec/05-state-and-quirks.md` §3.1
//! (encoded-value-plus-one rule for the residual width) + `spec/02` §1
//! / §2.1 / §2.2 (MSB-first uvar / svar primitives).

use oxideav_shorten::{
    decode_stream, encode_envelope_stream, min_energy_for_diff0, write_byte_aligned_prefix,
    write_diff0_block, write_parameter_block, write_quit_command, BitWriter, ShortenStreamHeader,
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

/// Assemble a Shorten file with a sequence of per-channel sample
/// blocks emitted in round-robin order per `spec/03` §2.
///
/// `channel_blocks[c]` is a vector of per-block sample vectors for
/// channel `c`. Each channel must contribute the same number of
/// blocks (the driver dispatches blocks in round-robin order, and a
/// channel with fewer blocks would terminate the stream early). The
/// per-block sample-length must equal `header.blocksize` so the
/// round-robin decoder uses the default sub-block size throughout
/// (no `BLOCK_FN_BLOCKSIZE` override is emitted by this helper).
fn assemble_diff0_stream(
    header: &ShortenStreamHeader,
    channel_blocks: &[Vec<Vec<i32>>],
) -> Vec<u8> {
    assert_eq!(channel_blocks.len(), header.channels as usize);
    let n_blocks_per_chan = channel_blocks[0].len();
    for ch in channel_blocks {
        assert_eq!(ch.len(), n_blocks_per_chan);
        for blk in ch {
            assert_eq!(blk.len(), header.blocksize as usize);
        }
    }

    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, header);

    // Round-robin: for each block index, walk every channel before
    // moving to the next block index.
    for b in 0..n_blocks_per_chan {
        for ch_blocks in channel_blocks.iter() {
            let samples = &ch_blocks[b];
            let energy =
                min_energy_for_diff0(&samples.iter().map(|&s| s as i64).collect::<Vec<_>>())
                    .expect("samples fit in natural energy range");
            write_diff0_block(&mut writer, energy, samples, /* mu_chan = */ 0)
                .expect("write_diff0_block");
        }
    }

    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());
    out
}

#[test]
fn mono_single_block_diff0_roundtrips_byte_exact() {
    let header = synth_header(2, 5, 1, 8, 0, 0, 0);
    let samples: Vec<i32> = vec![10, -7, 3, 0, -1, 5, -4, 2];
    let blocks = vec![vec![samples.clone()]];
    let bytes = assemble_diff0_stream(&header, &blocks);

    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.header, header);
    assert_eq!(dec.channels.len(), 1);
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn stereo_two_blocks_diff0_roundtrips_per_channel() {
    let header = synth_header(2, 5, 2, 4, 0, 0, 0);
    let ch0_blocks: Vec<Vec<i32>> = vec![vec![1, -1, 2, -2], vec![10, -10, 5, -5]];
    let ch1_blocks: Vec<Vec<i32>> = vec![vec![0, 1, -1, 2], vec![-8, 8, 0, 0]];
    let blocks = vec![ch0_blocks.clone(), ch1_blocks.clone()];
    let bytes = assemble_diff0_stream(&header, &blocks);

    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.channels.len(), 2);
    let expected_ch0: Vec<i32> = ch0_blocks.into_iter().flatten().collect();
    let expected_ch1: Vec<i32> = ch1_blocks.into_iter().flatten().collect();
    assert_eq!(dec.channels[0], expected_ch0);
    assert_eq!(dec.channels[1], expected_ch1);
}

#[test]
fn diff0_with_verbatim_prefix_envelope_preserves_payload_and_samples() {
    // Build the file manually to combine the envelope encoder's
    // verbatim-prefix support with the new DIFF0 block encoder.
    let header = synth_header(2, 5, 1, 4, 0, 0, 32);
    let preamble: Vec<u8> = (0..32).collect();
    let samples: Vec<i32> = vec![3, -3, 7, -1];

    // Start with an envelope-only stream (header + VERBATIM + QUIT +
    // pad). We then re-build the stream from primitives to splice
    // the DIFF0 block in before the QUIT terminator.
    let envelope = encode_envelope_stream(&header, &preamble).expect("envelope");
    assert!(envelope.len() > 32); // sanity — at least the preamble fits.
                                  // Sanity check the envelope-only file decodes to the same header
                                  // + verbatim with no samples.
    let env_dec = decode_stream(&envelope).expect("envelope decodes");
    assert_eq!(env_dec.verbatim, preamble);
    assert_eq!(env_dec.channels[0], Vec::<i32>::new());

    // Now build the full file with a DIFF0 block inserted.
    let mut out = Vec::new();
    write_byte_aligned_prefix(&mut out, header.version).expect("prefix");
    let mut writer = BitWriter::new();
    write_parameter_block(&mut writer, &header);
    // VERBATIM command first (before the predictor block) per
    // spec/03 §3.10 — the verbatim prefix is the file-header
    // payload the encoder buffered up front.
    oxideav_shorten::write_verbatim_block(&mut writer, &preamble).expect("verbatim");
    // Then the DIFF0 block carrying the samples.
    let energy = min_energy_for_diff0(&samples.iter().map(|&s| s as i64).collect::<Vec<_>>())
        .expect("samples fit");
    write_diff0_block(&mut writer, energy, &samples, 0).expect("diff0");
    write_quit_command(&mut writer);
    writer.pad_to_byte();
    out.extend(writer.into_bytes());

    let dec = decode_stream(&out).expect("decode full stream");
    assert_eq!(dec.verbatim, preamble);
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn diff0_silent_block_uses_minimum_energy() {
    // An all-zero sample block should pick energy 0 (residual
    // width 1) — the narrowest natural width per spec/05 §3.1.
    let zeros: Vec<i64> = vec![0; 16];
    assert_eq!(min_energy_for_diff0(&zeros), Some(0));

    let header = synth_header(2, 5, 1, 16, 0, 0, 0);
    let samples: Vec<i32> = vec![0; 16];
    let blocks = vec![vec![samples.clone()]];
    let bytes = assemble_diff0_stream(&header, &blocks);
    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn diff0_block_with_max_natural_residuals_roundtrips() {
    // ±100 → folded u ≤ 200 < 256 = 2^8, so energy = 7 (width 8).
    let header = synth_header(2, 5, 1, 4, 0, 0, 0);
    let samples: Vec<i32> = vec![100, -100, 0, -50];
    let blocks = vec![vec![samples.clone()]];
    let bytes = assemble_diff0_stream(&header, &blocks);
    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.channels[0], samples);
}

#[test]
fn diff0_block_residual_directionality_matches_decoder_reconstruction() {
    // spec/05 §2.3 reconstruction is s(t) = e0(t) + mu_chan. The
    // encoder emits e0(t) = s(t) - mu_chan; the decoder adds mu_chan
    // back. Because the round-7 decode_stream driver wires its own
    // running mean estimator (initialised to zero), this test sets
    // H_meanblocks = 0 so mu_chan is identically zero on both sides
    // and exercises the directionality without coupling to the mean
    // estimator's exact update rule.
    let header = synth_header(2, 5, 1, 8, 0, 0, 0);
    let samples: Vec<i32> = vec![50, 51, 49, 50, 52, 48, 50, 51];
    let blocks = vec![vec![samples.clone()]];
    let bytes = assemble_diff0_stream(&header, &blocks);
    let dec = decode_stream(&bytes).expect("decode");
    assert_eq!(dec.channels[0], samples);
}
