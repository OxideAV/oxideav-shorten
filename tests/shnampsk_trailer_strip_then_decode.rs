//! Integration test for the round-9 `SHNAMPSK` trailer detector
//! composed with the round-7 whole-stream decoder.
//!
//! `spec/05` §5.2 states that the SHN wire format itself terminates
//! at `BLOCK_FN_QUIT`'s zero-bit padding; bytes after that boundary
//! are out of scope and the decoder may ignore them. This test
//! verifies the two compositional contracts the round-9 detector
//! offers:
//!
//! 1. A complete SHN stream with no trailer decodes via
//!    [`decode_stream`] and `split_off_shnampsk_trailer` reports
//!    `None` for the sidecar.
//! 2. The same SHN stream with a synthetic `SHNAMPSK` trailer
//!    appended is split into `(shn_proper, Some(sidecar))` by
//!    `split_off_shnampsk_trailer`; the SHN-proper slice decodes to
//!    the same per-channel samples + verbatim prefix as the no-trailer
//!    case. The sidecar bytes match the input.
//!
//! Anchor: `spec/05` §5.1 (trailer layout), §5.2 (decoder behaviour),
//! §5.3 (fixtures `F1..F8` carry the trailer; `F9` / `Choppy.shn` do
//! not). `decode_stream` is the round-7 driver of `spec/03` §2 +
//! §3.6/§3.7/§3.8 + `spec/05` §1 + §1.4 + §2.

use oxideav_shorten::{
    decode_stream, decode_stream_iter, detect_shnampsk_trailer, split_off_shnampsk_trailer,
    FunctionCode, ENERGYSIZE, FNSIZE, SEEK_MAGIC, SHNAMPSK_SIGNATURE, TRAILER_TAIL_LEN,
};

// ---- synthetic-stream builders (mirror those used elsewhere) ----

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

fn assemble(all_bits: &[u32]) -> Vec<u8> {
    let body = pack_bits_msb_first(all_bits);
    let mut buf = Vec::with_capacity(5 + body.len());
    buf.extend_from_slice(b"ajkg");
    buf.push(2);
    buf.extend_from_slice(&body);
    buf
}

fn append_diff_block(out: &mut Vec<u32>, code: u32, energy: u32, residuals: &[i64]) {
    out.extend(encode_uvar(code, FNSIZE));
    out.extend(encode_uvar(energy, ENERGYSIZE));
    let width = energy + 1;
    for &r in residuals {
        out.extend(encode_svar(r, width));
    }
}

/// Build a single-channel `s16lh` stream:
/// VERBATIM(3 bytes) → DIFF0 ch0 [10, 20, 30, 40] → QUIT.
fn synthesise_stream() -> Vec<u8> {
    let mut bits = header_param_bits(5 /* s16lh */, 1, 4, 0, 0, 0);
    bits.extend(encode_uvar(FunctionCode::Verbatim as u32, FNSIZE));
    bits.extend(encode_uvar(3, 5));
    for b in [0x01u8, 0x02, 0x03] {
        bits.extend(encode_uvar(b as u32, 8));
    }
    append_diff_block(&mut bits, FunctionCode::Diff0 as u32, 5, &[10, 20, 30, 40]);
    bits.extend(encode_uvar(FunctionCode::Quit as u32, FNSIZE));
    assemble(&bits)
}

/// Append a well-formed SHNAMPSK trailer with the given opaque body
/// bytes (between the `SEEK` magic at sidecar-start and the trailing
/// `len_u32` + signature pair).
fn append_well_formed_trailer(file: &mut Vec<u8>, body_bytes: &[u8]) {
    let sidecar_len = SEEK_MAGIC.len() + body_bytes.len() + TRAILER_TAIL_LEN;
    file.extend_from_slice(&SEEK_MAGIC);
    file.extend_from_slice(body_bytes);
    file.extend_from_slice(&(sidecar_len as u32).to_le_bytes());
    file.extend_from_slice(&SHNAMPSK_SIGNATURE);
}

#[test]
fn no_trailer_path_decodes_and_split_reports_none() {
    let shn = synthesise_stream();
    let (proper, sidecar) = split_off_shnampsk_trailer(&shn).expect("split_off");
    assert_eq!(proper.len(), shn.len());
    assert!(sidecar.is_none());

    let dec = decode_stream(proper).expect("decode_stream");
    assert_eq!(dec.header.channels, 1);
    assert_eq!(dec.channels.len(), 1);
    // DIFF0 over zero carry, mu_chan = 0: samples = residuals.
    assert_eq!(dec.channels[0], vec![10, 20, 30, 40]);
    assert_eq!(dec.verbatim, vec![0x01, 0x02, 0x03]);
}

#[test]
fn trailer_present_path_strips_then_decodes_identically() {
    let shn = synthesise_stream();
    let mut file = shn.clone();
    let body = vec![0xAAu8; 24];
    append_well_formed_trailer(&mut file, &body);
    assert!(file.len() > shn.len());

    let t = detect_shnampsk_trailer(&file)
        .expect("detect_shnampsk_trailer")
        .expect("trailer present");
    assert_eq!(t.sidecar_start, shn.len());
    // sidecar_len = 4 (SEEK) + 24 (body) + 12 (tail) = 40.
    assert_eq!(t.sidecar_len, 40);

    let (proper, sidecar) = split_off_shnampsk_trailer(&file).expect("split_off");
    assert_eq!(proper.len(), shn.len());
    assert_eq!(proper, &shn[..]);
    let sc = sidecar.expect("sidecar present");
    assert_eq!(sc.len(), 40);
    assert_eq!(&sc[..4], &SEEK_MAGIC);
    assert_eq!(&sc[sc.len() - 8..], &SHNAMPSK_SIGNATURE);

    // The stripped SHN-proper bytes decode to the same samples and
    // verbatim prefix as the no-trailer reference.
    let dec_ref = decode_stream(&shn).expect("decode_stream ref");
    let dec_trim = decode_stream(proper).expect("decode_stream trimmed");
    assert_eq!(dec_ref.channels, dec_trim.channels);
    assert_eq!(dec_ref.verbatim, dec_trim.verbatim);
    assert_eq!(dec_ref.header.channels, dec_trim.header.channels);
    assert_eq!(dec_ref.header.filetype, dec_trim.header.filetype);
}

/// The decoder's own `BLOCK_FN_QUIT` byte-alignment boundary
/// (`stream_proper_len`, computed bottom-up from the wire) and the
/// `SHNAMPSK` detector's `sidecar_start` (computed top-down from the
/// trailing length field) must agree exactly on a well-formed file — two
/// independent computations of the same split point. `spec/05` §5.2 ties
/// them together: the SHN wire format terminates at the QUIT padding, so
/// the sidecar begins precisely where `stream_proper_len` points. This
/// is the cross-validation the prior two tests did not assert; a bug in
/// either boundary computation would surface as a mismatch here.
#[test]
fn quit_boundary_and_shnampsk_sidecar_start_agree() {
    let shn = synthesise_stream();
    let mut file = shn.clone();
    let body = vec![0x5Au8; 40];
    append_well_formed_trailer(&mut file, &body);

    // Top-down: the SHNAMPSK detector reports where the sidecar starts.
    let t = detect_shnampsk_trailer(&file)
        .expect("detect")
        .expect("trailer present");

    // Bottom-up (batch): the wire decoder reports the SHN-proper end via
    // the QUIT byte alignment. Run it over the WHOLE file (trailer and
    // all) — the decoder must stop at QUIT and never read the sidecar.
    let dec_full = decode_stream(&file).expect("decode_stream over full file");
    assert_eq!(
        dec_full.stream_proper_len, t.sidecar_start,
        "QUIT boundary must equal SHNAMPSK sidecar_start"
    );
    // And the SHN-proper length equals the untrailered stream length.
    assert_eq!(dec_full.stream_proper_len, shn.len());

    // Bottom-up (streaming): the StreamDecoder must arrive at the same
    // boundary and report the trailer size = full len - proper len.
    let mut iter = decode_stream_iter(&file).expect("decode_stream_iter");
    while iter.next_block().expect("ok").is_some() {}
    assert_eq!(iter.stream_proper_len(), Some(t.sidecar_start));
    assert_eq!(iter.trailer_len(), Some(file.len() - shn.len()));
    assert_eq!(iter.trailer_len(), Some(t.sidecar_len as usize));

    // All three computations agree on the same byte offset.
    assert_eq!(
        dec_full.stream_proper_len,
        iter.stream_proper_len().unwrap()
    );
}
