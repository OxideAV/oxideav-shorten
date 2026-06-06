//! Integration test exercising the spec/05 §6 typed-filetype
//! accessor `ShortenStreamHeader::filetype_pinned()` against:
//!
//! 1. The real fixture-`F1` byte sequence (the 11-byte prefix
//!    `0x61 0x6A 0x6B 0x67 0x02 0xFB 0xB1 0x70 0x09 0xF9 0x25` whose
//!    `H_filetype` field decodes to `5` per spec/02 §6.1 / spec/05 §6
//!    behavioural anchor).
//! 2. Synthetic v2 headers stamped with each of the three pinned
//!    numeric codes (`2`, `3`, `5`) covering `u8` / `s16hl` /
//!    `s16lh` per spec/05 §6.
//! 3. Synthetic v2 headers stamped with numeric codes that spec/05
//!    §6 + §8 candidate #3 leaves unpinned (`0`, `1`, `4`, `6`, `7`,
//!    `8`, …) — the accessor must return `None` rather than guessing
//!    a label.
//!
//! This pins the typed accessor against (a) a real fixture's bytes
//! the spec set explicitly anchors and (b) the unpinned-code negative
//! contract spec/05 §6 + §8 requires.

use oxideav_shorten::{parse_stream_header, Filetype, MAGIC};

/// Bit-packer helper mirroring the shape used elsewhere in this
/// crate's integration tests: takes an MSB-first bit sequence and
/// packs it into a `Vec<u8>`, left-justifying the trailing byte.
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

/// Encode `value` as `ulong(value)` for a chosen mantissa width `w`
/// per spec/02 §3: stage 1 emits `uvar(ULONGSIZE = 2)` for `w` and
/// stage 2 emits `uvar(w)` for `value`.
fn encode_ulong(value: u32, w: u32) -> Vec<u32> {
    let mut bits = Vec::new();
    // Stage 1: uvar(2) for w.
    let prefix_zeros_w = w / 4;
    let mantissa_w = w % 4;
    bits.resize(bits.len() + prefix_zeros_w as usize, 0);
    bits.push(1);
    bits.push((mantissa_w >> 1) & 1);
    bits.push(mantissa_w & 1);

    // Stage 2: uvar(w) for value.
    if w == 0 {
        assert_eq!(value, 0, "uvar(0) can only encode 0");
        bits.push(1);
    } else {
        let span = 1u32 << w;
        let prefix_zeros_v = value / span;
        let mantissa_v = value % span;
        bits.resize(bits.len() + prefix_zeros_v as usize, 0);
        bits.push(1);
        for i in (0..w).rev() {
            bits.push((mantissa_v >> i) & 1);
        }
    }
    bits
}

/// Build a synthetic v2 stream-header byte buffer with the supplied
/// six header fields, using the per-field widths fixture `F1` uses
/// (`w = 3, 2, 9, 0, 3, 0` per spec/02 §6).
fn synth_v2_header(
    filetype: u32,
    channels: u32,
    blocksize: u32,
    maxlpcorder: u32,
    meanblocks: u32,
    skipbytes: u32,
) -> Vec<u8> {
    let mut bits = Vec::new();
    bits.extend(encode_ulong(filetype, 3));
    bits.extend(encode_ulong(channels, 2));
    bits.extend(encode_ulong(blocksize, 9));
    bits.extend(encode_ulong(maxlpcorder, 0));
    bits.extend(encode_ulong(meanblocks, 3));
    bits.extend(encode_ulong(skipbytes, 0));
    let mut out = Vec::with_capacity(5 + bits.len() / 8 + 1);
    out.extend_from_slice(&MAGIC);
    out.push(2);
    out.extend(pack_bits_msb_first(&bits));
    out
}

#[test]
fn f1_real_byte_prefix_resolves_to_s16lh() {
    // Fixture F1's 11-byte prefix per spec/02 §6: the H_filetype
    // field decodes to 5 (`s16lh`) per §6.1 / spec/05 §6 anchor.
    let buf = [
        0x61, 0x6A, 0x6B, 0x67, 0x02, 0xFB, 0xB1, 0x70, 0x09, 0xF9, 0x25,
    ];
    let parsed = parse_stream_header(&buf).expect("F1 prefix must parse");
    assert_eq!(parsed.header.filetype, 5);
    assert_eq!(parsed.header.filetype_pinned(), Some(Filetype::S16LH));

    let ft = parsed.header.filetype_pinned().unwrap();
    assert_eq!(ft.label(), "s16lh");
    assert_eq!(ft.wire_value(), 5);
    assert_eq!(ft.bytes_per_sample(), 2);
    assert!(ft.is_signed());
    assert_eq!(ft.is_little_endian(), Some(true));
}

#[test]
fn synthetic_u8_header_resolves_to_u8_variant() {
    // spec/05 §6 row: numeric 2 ↔ TR.156 label `u8` ↔ fixture F2.
    let buf = synth_v2_header(2, 1, 256, 0, 4, 0);
    let parsed = parse_stream_header(&buf).expect("synthetic u8 header must parse");
    assert_eq!(parsed.header.filetype, 2);
    assert_eq!(parsed.header.filetype_pinned(), Some(Filetype::U8));

    let ft = parsed.header.filetype_pinned().unwrap();
    assert_eq!(ft.label(), "u8");
    assert_eq!(ft.bytes_per_sample(), 1);
    assert!(!ft.is_signed());
    assert_eq!(ft.is_little_endian(), None);
}

#[test]
fn synthetic_s16hl_header_resolves_to_s16hl_variant() {
    // spec/05 §6 row: numeric 3 ↔ TR.156 label `s16hl` ↔ fixture F3.
    let buf = synth_v2_header(3, 2, 256, 0, 4, 0);
    let parsed = parse_stream_header(&buf).expect("synthetic s16hl header must parse");
    assert_eq!(parsed.header.filetype, 3);
    assert_eq!(parsed.header.filetype_pinned(), Some(Filetype::S16HL));

    let ft = parsed.header.filetype_pinned().unwrap();
    assert_eq!(ft.label(), "s16hl");
    assert_eq!(ft.bytes_per_sample(), 2);
    assert!(ft.is_signed());
    assert_eq!(ft.is_little_endian(), Some(false));
}

#[test]
fn synthetic_unpinned_filetypes_return_none() {
    // spec/05 §6 + §8 candidate #3: numeric codes outside { 2, 3, 5 }
    // are not behaviourally anchored. The typed accessor must
    // distinguish "unpinned" from "guessed" by returning None.
    for unpinned in [0u32, 1, 4, 6, 7] {
        let buf = synth_v2_header(unpinned, 1, 256, 0, 4, 0);
        let parsed = parse_stream_header(&buf).expect("synthetic header must parse");
        assert_eq!(parsed.header.filetype, unpinned);
        assert_eq!(
            parsed.header.filetype_pinned(),
            None,
            "wire value {unpinned} is unpinned by spec/05 §6 + §8",
        );
    }
}

#[test]
fn three_pinned_codes_round_trip_through_from_wire_and_wire_value() {
    // The three spec/05 §6 pinned codes each round-trip through the
    // wire ↔ variant accessor pair.
    for (variant, wire) in [
        (Filetype::U8, 2u32),
        (Filetype::S16HL, 3),
        (Filetype::S16LH, 5),
    ] {
        assert_eq!(Filetype::from_wire(wire), Some(variant));
        assert_eq!(variant.wire_value(), wire);
    }
}

#[test]
fn pinned_variant_byte_widths_match_pcm_plane_packing() {
    // The bytes-per-sample accessor must match the per-channel PCM
    // plane width the round-8 trait wrapper packs each variant into
    // (`spec/05` §6 narrative: u8 → 1 byte, s16* → 2 bytes per
    // sample). The plane width drives the host-side `AudioFrame`
    // buffer allocation, so a mismatch would surface as an
    // off-by-half truncation or doubling at the trait boundary.
    let channels = 2u32;
    let samples_per_channel = 256u32;

    for variant in [Filetype::U8, Filetype::S16HL, Filetype::S16LH] {
        let expected_plane_bytes = samples_per_channel as usize * variant.bytes_per_sample();
        let plane_bytes_per_channel = match variant {
            Filetype::U8 => samples_per_channel as usize,
            Filetype::S16HL | Filetype::S16LH => samples_per_channel as usize * 2,
            _ => unreachable!(),
        };
        assert_eq!(plane_bytes_per_channel, expected_plane_bytes);
        let _ = channels;
    }
}
