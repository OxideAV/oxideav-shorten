//! Shorten's three Rice/Golomb integer flavours.
//!
//! All three ride on top of the JPEG-LS unsigned Golomb-Rice primitive:
//! a unary-coded high part (`q` zero bits terminated by a `1`), then `k`
//! raw low bits.
//!
//! - [`read_unsigned_k`] — plain Rice-`k`: `(q << k) | low_k`.
//! - [`read_signed_k`]   — same, but `k+1` low bits with the LSB acting
//!   as a sign (zig-zag `(u >> 1) XOR -(u & 1)`). Note the underlying
//!   unsigned read uses `k+1` low bits, not `k`. This is **not** the
//!   same as FLAC's signed Rice, which uses `k` low bits and a separate
//!   even/odd zig-zag decode.
//! - [`read_ulong`]      — adaptive: a 2-bit-`k=k_param` unsigned Rice
//!   selects the parameter, then the value is read as fixed-`k`
//!   unsigned. The 2-bit `k_param` is itself the `k_param_size`
//!   constant passed in (typically 2 — `ULONGSIZE` in FFmpeg
//!   nomenclature).

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

/// Read an unsigned Rice-`k` integer.
///
/// Layout: unary prefix `q` (zero bits terminated by a `1`), followed
/// by `k` raw bits. The decoded value is `(q << k) | low`.
pub fn read_unsigned_k(br: &mut BitReader<'_>, k: u32) -> Result<u32> {
    let q = br.read_unary()?;
    if k == 0 {
        return Ok(q);
    }
    if k > 31 {
        return Err(Error::invalid("shorten rice: k > 31"));
    }
    let low = br.read_u32(k)?;
    // Guard against pathological inputs whose unary prefix would
    // overflow `q << k`. Real bitstreams stay well below this — the
    // largest residuals in the corpus fit in ~30 bits total — but a
    // hand-crafted bitstream could feed thousands of zero bits.
    let q_shifted = q
        .checked_shl(k)
        .ok_or_else(|| Error::invalid("shorten rice: unary overflow"))?;
    Ok(q_shifted | low)
}

/// Read a signed Rice-`k` integer.
///
/// Reads as unsigned Rice with `k + 1` raw low bits, then unzig-zags:
/// the LSB acts as the sign (`1 = negative, magnitude = upper bits`),
/// so `decode(u) = (u >> 1) XOR -(u & 1)` in two's complement.
pub fn read_signed_k(br: &mut BitReader<'_>, k: u32) -> Result<i32> {
    let u = read_unsigned_k(br, k + 1)?;
    // Zig-zag: sign in the LSB.
    let mag = (u >> 1) as i32;
    let sign = (u & 1) as i32;
    Ok(mag ^ -sign)
}

/// Read an adaptive "ulong" unsigned integer.
///
/// First read a 2-bit-`k` unsigned Rice parameter (`k_in`), then read the
/// value as fixed-`k_in` unsigned Rice. Used everywhere a header field
/// or BLOCKSIZE payload appears in the bitstream.
///
/// `k_param_size` is the number of bits used for the parameter itself
/// (`ULONGSIZE = 2` in v >= 1; v0 streams use the literal `size_hint`
/// directly without re-reading `k_in`).
pub fn read_ulong(br: &mut BitReader<'_>, k_param_size: u32, _size_hint: u32) -> Result<u32> {
    let k_in = read_unsigned_k(br, k_param_size)?;
    if k_in > 31 {
        return Err(Error::invalid("shorten rice: ulong k out of range"));
    }
    read_unsigned_k(br, k_in)
}

/// Read a v0-style ulong. v0 streams skip the leading parameter and
/// just read `size_hint` raw bits as Rice-`k=size_hint`. Kept for
/// completeness — the corpus contains only v2 streams, but the
/// decoder accepts v0/v1 inputs as well.
pub fn read_ulong_v0(br: &mut BitReader<'_>, size_hint: u32) -> Result<u32> {
    read_unsigned_k(br, size_hint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;

    fn write_unsigned(bw: &mut BitWriter, value: u32, k: u32) {
        // Unary prefix: q zero bits followed by a 1.
        let q = value >> k;
        for _ in 0..q {
            bw.write_u32(0, 1);
        }
        bw.write_u32(1, 1);
        if k > 0 {
            let low = value & ((1u32 << k) - 1);
            bw.write_u32(low, k);
        }
    }

    fn write_signed(bw: &mut BitWriter, value: i32, k: u32) {
        // Zig-zag in the LSB. Use wrapping to handle i32::MIN.
        let u = ((value << 1) ^ (value >> 31)) as u32;
        write_unsigned(bw, u, k + 1);
    }

    fn write_ulong(bw: &mut BitWriter, value: u32, k_param_size: u32) {
        // Pick the smallest k such that value < 2^k * (some unary
        // prefix budget). For tests we just pick the smallest k that
        // gives a non-empty unary prefix budget large enough for
        // value, with a reasonable default.
        let mut k_in = 0u32;
        while k_in < 31 && (value >> k_in) > 8 {
            k_in += 1;
        }
        write_unsigned(bw, k_in, k_param_size);
        write_unsigned(bw, value, k_in);
    }

    #[test]
    fn unsigned_k_roundtrip() {
        for k in 0..6 {
            let mut bw = BitWriter::new();
            for v in [0u32, 1, 2, 5, 100, 1023] {
                write_unsigned(&mut bw, v, k);
            }
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            for v in [0u32, 1, 2, 5, 100, 1023] {
                let got = read_unsigned_k(&mut br, k).unwrap();
                assert_eq!(got, v, "k={k} v={v}");
            }
        }
    }

    #[test]
    fn signed_k_roundtrip() {
        for k in 0..6 {
            let values = [0i32, 1, -1, 2, -2, 100, -100, 12345, -12345];
            let mut bw = BitWriter::new();
            for &v in &values {
                write_signed(&mut bw, v, k);
            }
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            for &v in &values {
                let got = read_signed_k(&mut br, k).unwrap();
                assert_eq!(got, v, "k={k} v={v}");
            }
        }
    }

    #[test]
    fn ulong_roundtrip() {
        let mut bw = BitWriter::new();
        let values = [0u32, 1, 5, 256, 4096, 65535];
        for &v in &values {
            write_ulong(&mut bw, v, 2);
        }
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        for &v in &values {
            let got = read_ulong(&mut br, 2, 0).unwrap();
            assert_eq!(got, v, "v={v}");
        }
    }
}
