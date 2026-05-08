//! Variable-length integer codec (`uvar` / `svar` / `ulong`).
//!
//! Implements the bit-stream forms documented in
//! `spec/02-variable-length-coding.md`:
//!
//! * `uvar(n)` — Rice / Golomb prefix code: a run of zero bits
//!   followed by a bounding `1` and an `n`-bit MSB-first mantissa.
//!   Decoded value is `(zero_count << n) | mantissa`.
//! * `svar(n)` — one's-complement signed folding on top of `uvar(n)`:
//!   even unsigned values map to non-negative integers (`u >> 1`); odd
//!   values map to negative integers (`!(u >> 1)` in two's-complement,
//!   equivalently `-((u >> 1) + 1)`).
//! * `ulong()` — the two-stage form: read a width
//!   `w = uvar(ULONGSIZE = 2)`, then read the value as `uvar(w)`.

use crate::bitreader::BitReader;
use crate::{Error, Result};

/// Width-of-width for the two-stage [`read_ulong`] form.
///
/// Pinned by `spec/02-variable-length-coding.md` §3.
pub(crate) const ULONGSIZE: u32 = 2;

/// Per-block command-code mantissa width.
///
/// Pinned by `spec/02-variable-length-coding.md` §4.1.
pub(crate) const FNSIZE: u32 = 2;

/// Per-block Rice-coding energy-parameter width. The encoded value
/// is `n - 1` (the residual mantissa width is `field + 1`); see
/// `spec/05-state-and-quirks.md` §3.
pub(crate) const ENERGYSIZE: u32 = 3;

/// Per-block LPC-order field width. `BLOCK_FN_QLPC` reads
/// `uvar(LPCQSIZE)` for the per-block order.
pub(crate) const LPCQSIZE: u32 = 2;

/// Per-coefficient quantised-LPC `svar` mantissa width.
pub(crate) const LPCQUANT: u32 = 2;

/// Verbatim-prefix length-field width.
pub(crate) const VERBATIM_CHUNK_SIZE: u32 = 5;

/// Verbatim-prefix per-byte mantissa width.
pub(crate) const VERBATIM_BYTE_SIZE: u32 = 8;

/// Bit-shift command's shift-amount field width.
pub(crate) const BITSHIFTSIZE: u32 = 2;

/// Cap on the number of leading zero bits permitted in any `uvar`
/// read. This is a bit-stream-corruption guard: a wedge of zero bits
/// past the end of a real bit stream would otherwise loop forever
/// before EOF surfaces.
const UVAR_MAX_ZEROS: u32 = 64;

/// Hard cap on the residual mantissa width after the energy-field
/// `+1` adjustment. Shorten's signed-residual lanes are 32-bit; the
/// encoder never emits widths anywhere near this. A field-plus-one
/// width above this bound is treated as a structural error rather
/// than allowed to silently wrap.
pub(crate) const RESIDUAL_WIDTH_CAP: u32 = 32;

/// Decode `uvar(n)` from `br`.
///
/// Reads zero bits until a terminating `1` is encountered, then `n`
/// mantissa bits MSB-first. Returns `(zero_count << n) | mantissa`
/// as a `u32`.
///
/// `n` is checked against [`RESIDUAL_WIDTH_CAP`] so that residual
/// reads cannot demand more than the lane can hold.
///
/// Round 5 routes the prefix scan through the table-driven
/// [`crate::bitreader::BitReader::read_uvar_prefix`] LUT for the
/// hot decoder loop.
#[inline]
pub(crate) fn read_uvar(br: &mut BitReader<'_>, n: u32) -> Result<u32> {
    if n > RESIDUAL_WIDTH_CAP {
        return Err(Error::ResidualWidthOverflow(n));
    }
    let zeros = br
        .read_uvar_prefix(UVAR_MAX_ZEROS)
        .ok_or(Error::UnexpectedEof)?;
    let mantissa = br.read_bits(n)?;
    // (zeros << n) | mantissa; check for overflow above the 32-bit lane.
    let high = zeros
        .checked_shl(n)
        .ok_or(Error::ResidualWidthOverflow(n))?;
    Ok(high | mantissa)
}

/// Decode `svar(n)` from `br`.
///
/// One's-complement folding (`spec/02 §2.2`):
///   * even `u` → `s = u >> 1`,
///   * odd  `u` → `s = !(u >> 1)` (negative).
///
/// The residual lane is `i32`. The encoder is permitted to emit
/// values in the full `i32` range; widths up to 31 fit into the
/// unsigned `u32` payload before folding.
#[inline]
pub(crate) fn read_svar(br: &mut BitReader<'_>, n: u32) -> Result<i32> {
    let u = read_uvar(br, n)?;
    Ok(unsigned_to_signed(u))
}

/// Inverse of the encoder-side one's-complement folding. Exposed at
/// crate scope so the test-only encoder helpers can reuse it.
#[inline]
pub(crate) fn unsigned_to_signed(u: u32) -> i32 {
    let half = (u >> 1) as i32;
    if (u & 1) == 0 {
        half
    } else {
        // Two's-complement bitwise NOT of `half` — i.e., `-half - 1`.
        !half
    }
}

/// Forward of the one's-complement folding. Used by the production
/// encoder + the test-only round-1 helpers.
#[inline]
pub(crate) fn signed_to_unsigned(s: i32) -> u32 {
    if s >= 0 {
        (s as u32) << 1
    } else {
        // For negative s: u = ((!s) << 1) | 1 = ((-s - 1) << 1) | 1.
        let mag = (!s) as u32;
        (mag << 1) | 1
    }
}

/// Decode the two-stage `ulong()` form: width `w = uvar(ULONGSIZE)`
/// then value `v = uvar(w)`.
///
/// Pinned in `spec/02-variable-length-coding.md` §3.
#[inline]
pub(crate) fn read_ulong(br: &mut BitReader<'_>) -> Result<u32> {
    let w = read_uvar(br, ULONGSIZE)?;
    if w > RESIDUAL_WIDTH_CAP {
        return Err(Error::ResidualWidthOverflow(w));
    }
    read_uvar(br, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a [`BitReader`] from an explicit bit pattern (MSB-first
    /// per byte). Test scaffolding only.
    fn bits_from(pattern: &[u8]) -> Vec<u8> {
        // Pack the MSB-first bit list into bytes.
        let mut out = Vec::with_capacity(pattern.len().div_ceil(8));
        let mut acc: u8 = 0;
        let mut count = 0;
        for &b in pattern {
            assert!(b <= 1, "test bit pattern uses bits in {{0,1}}");
            acc = (acc << 1) | b;
            count += 1;
            if count == 8 {
                out.push(acc);
                acc = 0;
                count = 0;
            }
        }
        if count > 0 {
            acc <<= 8 - count;
            out.push(acc);
        }
        out
    }

    #[test]
    fn uvar_zero_mantissa() {
        // n=2: pattern "1 00" decodes as (0 zeros << 2) | 00 = 0.
        let bytes = bits_from(&[1, 0, 0]);
        let mut br = BitReader::new(&bytes);
        assert_eq!(read_uvar(&mut br, 2).unwrap(), 0);
    }

    #[test]
    fn uvar_tr156_examples() {
        // TR.156 examples for n=2:
        //   100 → 0; 101 → 1; 110 → 2; 111 → 3;
        //   0100 → 4; 0111 → 7; 00100 → 8; 0000100 → 16.
        let cases: &[(&[u8], u32)] = &[
            (&[1, 0, 0], 0),
            (&[1, 0, 1], 1),
            (&[1, 1, 0], 2),
            (&[1, 1, 1], 3),
            (&[0, 1, 0, 0], 4),
            (&[0, 1, 1, 1], 7),
            (&[0, 0, 1, 0, 0], 8),
            (&[0, 0, 0, 0, 1, 0, 0], 16),
        ];
        for (pat, expected) in cases {
            let bytes = bits_from(pat);
            let mut br = BitReader::new(&bytes);
            assert_eq!(read_uvar(&mut br, 2).unwrap(), *expected, "pattern {pat:?}");
        }
    }

    #[test]
    fn uvar_zero_width() {
        // n=0: only the prefix is read; value = zeros count.
        let bytes = bits_from(&[1]);
        let mut br = BitReader::new(&bytes);
        assert_eq!(read_uvar(&mut br, 0).unwrap(), 0);
        let bytes = bits_from(&[0, 0, 0, 1]);
        let mut br = BitReader::new(&bytes);
        assert_eq!(read_uvar(&mut br, 0).unwrap(), 3);
    }

    #[test]
    fn svar_folding_round_trip() {
        for s in [-5i32, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] {
            let u = signed_to_unsigned(s);
            assert_eq!(unsigned_to_signed(u), s, "s={s}, u={u}");
        }
    }

    #[test]
    fn svar_folding_specific_values() {
        // From the worked F1 trace: residuals like 4 and -26 appear.
        // 4 → u = 8; -26 → u = ((-(-26))-1) << 1 | 1 = 25 << 1 | 1 = 51.
        assert_eq!(signed_to_unsigned(4), 8);
        assert_eq!(unsigned_to_signed(8), 4);
        assert_eq!(signed_to_unsigned(-26), 51);
        assert_eq!(unsigned_to_signed(51), -26);
    }

    #[test]
    fn ulong_f1_header() {
        // F1's header parameter block, decoded bit-by-bit (spec/02 §6).
        // Bytes 0x05..0x0A = FB B1 70 09 F9 25.
        let bytes = [0xFBu8, 0xB1, 0x70, 0x09, 0xF9, 0x25];
        let mut br = BitReader::new(&bytes);
        // H_filetype = 5
        assert_eq!(read_ulong(&mut br).unwrap(), 5);
        // H_channels = 2
        assert_eq!(read_ulong(&mut br).unwrap(), 2);
        // H_blocksize = 256
        assert_eq!(read_ulong(&mut br).unwrap(), 256);
        // H_maxlpcorder = 0
        assert_eq!(read_ulong(&mut br).unwrap(), 0);
        // H_meanblocks = 4
        assert_eq!(read_ulong(&mut br).unwrap(), 4);
        // H_skipbytes = 0
        assert_eq!(read_ulong(&mut br).unwrap(), 0);
        // After 6 fields, bit cursor is at bit 43 in `bytes`.
        assert_eq!(br.bit_pos(), 43);
    }
}
