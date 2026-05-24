//! MSB-first bit reader for the Shorten variable-length integer
//! scheme of `docs/audio/shorten/spec/02-variable-length-coding.md`.
//!
//! Per spec/02 §1: "Within each byte the bits are read
//! **most-significant bit first**: the first bit consumed is bit 7 of
//! the first byte (mask `0x80`), the second is bit 6 (mask `0x40`),
//! and so on, reaching bit 0 of the byte (mask `0x01`) before
//! advancing to bit 7 of the next byte." The reader is fed the bytes
//! at file offset `0x05` onward — the header parser handles the
//! byte-aligned magic + version field separately.
//!
//! Round 1 implemented the three primitives the file-header parser
//! needs:
//!
//! * [`BitReader::read_bit`] — one bit, MSB-first.
//! * [`BitReader::read_bits`] — `n` bits (with `n <= 32`),
//!   left-to-right (the high bits of the returned `u32` are zero;
//!   the next-to-consume bit lands in position `n - 1`).
//! * [`BitReader::read_uvar`] — the unsigned `uvar(n)` form of
//!   spec/02 §2.1: count leading zero bits to a terminating one,
//!   read `n`-bit mantissa, return `(k << n) + m`.
//! * [`BitReader::read_ulong`] — the two-stage `ulong()` form of
//!   spec/02 §3: read a width `w = uvar(ULONGSIZE)`, then read the
//!   value `v = uvar(w)`. `ULONGSIZE = 2` is the wire constant
//!   pinned in spec/02 §3 + §5.
//!
//! Round 2 adds:
//!
//! * [`BitReader::read_svar`] — the signed `svar(n)` form of
//!   spec/02 §2.2 (one's-complement folding: even unsigned -> non-
//!   negative `s = u >> 1`, odd unsigned -> negative `s = !(u >> 1)`).

use crate::error::{Error, Result};

/// Width of the inner `uvar` field read by the two-stage `ulong()`
/// form. `spec/02-variable-length-coding.md` §3 + §5 pins
/// `ULONGSIZE = 2`.
pub const ULONGSIZE: u32 = 2;

/// Implementation-side safety cap on the leading-zero prefix of a
/// `uvar(n)` decode. A well-formed Shorten header field is a small
/// integer (the largest field, `H_blocksize`, defaults to 256 — so
/// even with `n = 0` only 8 leading zeros would be needed); anything
/// approaching 32 leading zeros is either a malformed stream or
/// pathological input.
const UVAR_PREFIX_CAP: u32 = 32;

/// MSB-first bit reader over a borrowed byte slice. The reader
/// starts at byte `0` of the slice and reads from bit 7 of that byte
/// downward, then bit 7 of byte 1, etc.
#[derive(Debug)]
pub struct BitReader<'a> {
    bytes: &'a [u8],
    /// Next byte index to consume into the cache.
    byte_pos: usize,
    /// MSB-first cache. The valid bits live in positions
    /// `[64 - n_bits, 64)` (i.e., the highest `n_bits` of a u64). The
    /// next bit to consume is bit 63.
    cache: u64,
    /// Count of valid bits currently in `cache`.
    n_bits: u32,
}

impl<'a> BitReader<'a> {
    /// Construct a fresh reader over `bytes`. The first call to
    /// [`Self::read_bit`] returns the MSB (mask `0x80`) of `bytes[0]`.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            byte_pos: 0,
            cache: 0,
            n_bits: 0,
        }
    }

    /// Total number of bits already consumed by `read_*` calls. Used
    /// by the header parser to report the end-of-header bit position
    /// (spec/02 §6.7 — fixture `F1` ends at bit 43 relative to byte
    /// `0x05`).
    pub fn bits_consumed_so_far(&self, total_bits_in_input: u32) -> u32 {
        let cached = self.n_bits;
        let bits_read_from_input = (self.byte_pos as u32) * 8;
        // Bits surfaced to the caller = bits pulled from input - bits
        // still sitting in the cache.
        debug_assert!(bits_read_from_input >= cached);
        debug_assert!(bits_read_from_input <= total_bits_in_input + cached);
        bits_read_from_input - cached
    }

    /// Refill the cache so that it holds at least `min_bits` valid
    /// bits. Each refilled byte is shifted into the low positions of
    /// the cache window, but the read API consumes from the high end.
    fn refill_to(&mut self, min_bits: u32) -> Result<()> {
        debug_assert!(min_bits <= 64);
        while self.n_bits < min_bits {
            if self.byte_pos >= self.bytes.len() {
                return Err(Error::Truncated);
            }
            let b = self.bytes[self.byte_pos];
            self.byte_pos += 1;
            // Insert the byte just below the current valid window.
            // After the insert the valid window has grown by 8 bits;
            // the just-inserted byte's MSB sits at position
            // `64 - n_bits - 1` (i.e., one below the previous low
            // boundary).
            let shift = 64 - self.n_bits - 8;
            self.cache |= (b as u64) << shift;
            self.n_bits += 8;
        }
        Ok(())
    }

    /// Read one bit, MSB-first. Returns `0` or `1`.
    pub fn read_bit(&mut self) -> Result<u32> {
        self.refill_to(1)?;
        // Highest valid bit is at position 63.
        let v = ((self.cache >> 63) & 1) as u32;
        self.cache <<= 1;
        self.n_bits -= 1;
        Ok(v)
    }

    /// Read `n` bits MSB-first (`n <= 32`). The returned `u32` has
    /// the first-consumed bit in position `n - 1` and the
    /// last-consumed bit in position `0`.
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        debug_assert!(n <= 32);
        self.refill_to(n)?;
        // Take the top `n` bits of the cache.
        let shift = 64 - n;
        let v = (self.cache >> shift) as u32;
        // Discard them from the cache.
        if n == 64 {
            self.cache = 0;
        } else {
            self.cache <<= n;
        }
        self.n_bits -= n;
        Ok(v)
    }

    /// Skip `n` bits MSB-first without surfacing their value, draining
    /// the bit stream by exactly `n` positions. Unlike [`Self::read_bits`]
    /// the count is unbounded — the driver uses it to seek past the
    /// variable-length header parameter block (whose total width can
    /// exceed 32 bits) to the first per-block command. Returns
    /// [`Error::Truncated`] if the stream exhausts before `n` bits are
    /// drained.
    pub fn skip_bits(&mut self, mut n: u32) -> Result<()> {
        while n > 0 {
            let take = n.min(32);
            self.read_bits(take)?;
            n -= take;
        }
        Ok(())
    }

    /// `uvar(n)` per spec/02 §2.1: count zero bits to a terminating
    /// one, then read `n` mantissa bits. Returns `(k << n) + m` where
    /// `k` is the leading-zero count and `m` is the mantissa.
    pub fn read_uvar(&mut self, n: u32) -> Result<u32> {
        let mut k = 0u32;
        loop {
            let bit = self.read_bit()?;
            if bit == 1 {
                break;
            }
            k += 1;
            if k > UVAR_PREFIX_CAP {
                return Err(Error::OverflowingUvar);
            }
        }
        let m = if n == 0 { 0 } else { self.read_bits(n)? };
        // `(k << n)` cannot overflow u32 because `n <= 32` and
        // `k <= UVAR_PREFIX_CAP = 32`. Saturate just in case the cap
        // is widened in a later round.
        let high = (k as u64) << n;
        let v = high.checked_add(m as u64).ok_or(Error::OverflowingUvar)?;
        if v > u32::MAX as u64 {
            return Err(Error::OverflowingUvar);
        }
        Ok(v as u32)
    }

    /// `ulong()` per spec/02 §3: a two-stage `uvar` where the first
    /// stage names the width of the second. `w = uvar(ULONGSIZE)`,
    /// `v = uvar(w)`.
    pub fn read_ulong(&mut self) -> Result<u32> {
        let w = self.read_uvar(ULONGSIZE)?;
        // The width drives a `read_bits(w)`; cap it to a sane upper
        // bound. spec/02 §6.3 reaches `w = 9` for `H_blocksize` —
        // anything beyond 32 is meaningless for a u32 return.
        if w > 32 {
            return Err(Error::OverflowingUvar);
        }
        self.read_uvar(w)
    }

    /// `svar(n)` per spec/02 §2.2: read an unsigned `uvar(n)` value
    /// `u` then unfold it back to a signed integer.
    ///
    /// The folding spec/02 §2.2 pins (verified byte-exact on fixture
    /// `F1`'s verbatim-prefix recovery) is:
    ///
    /// * If `u` is even: `s = u >> 1` (non-negative).
    /// * If `u` is odd:  `s = !(u >> 1)` (negative; equals
    ///   `-((u >> 1) + 1)`).
    ///
    /// The folded `u` can occupy any value `u32` admits; the signed
    /// `s` is widened to `i64` so that `u = u32::MAX` (which maps to
    /// `s = -(2^31)`) does not lose precision against an `i32`
    /// surface. Per-block residuals and quantised LPC coefficients of
    /// spec/03 will narrow to `i32` at the predictor boundary; this
    /// reader only undoes the folding.
    pub fn read_svar(&mut self, n: u32) -> Result<i64> {
        let u = self.read_uvar(n)?;
        let mag = (u >> 1) as i64;
        if u & 1 == 0 {
            Ok(mag)
        } else {
            Ok(!mag)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_bit_msb_first() {
        // 0b1010_0110 — MSB-first reads should yield
        // 1, 0, 1, 0, 0, 1, 1, 0.
        let bytes = [0b1010_0110];
        let mut r = BitReader::new(&bytes);
        for &expected in &[1u32, 0, 1, 0, 0, 1, 1, 0] {
            assert_eq!(r.read_bit().unwrap(), expected);
        }
        // One more bit would exhaust the buffer.
        assert!(matches!(r.read_bit(), Err(Error::Truncated)));
    }

    #[test]
    fn read_bits_crosses_byte_boundary() {
        // 0xFB 0xB1 = 1111 1011 1011 0001. Reading 12 bits MSB-first
        // yields 0b1111_1011_1011 = 0xFBB.
        let bytes = [0xFB, 0xB1];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(12).unwrap(), 0xFBB);
        // Remaining 4 bits = 0b0001.
        assert_eq!(r.read_bits(4).unwrap(), 0b0001);
    }

    #[test]
    fn skip_bits_drains_exact_count_including_over_32() {
        // 0xFB 0xB1 0x70 0x09 0xF9 = 40 bits. Skip 12, then the next
        // 4 bits should be 0b0001 (matching read_bits_crosses_byte_boundary),
        // then skip the remaining 24 to exhaustion.
        let bytes = [0xFB, 0xB1, 0x70, 0x09, 0xF9];
        let mut r = BitReader::new(&bytes);
        r.skip_bits(12).unwrap();
        assert_eq!(r.read_bits(4).unwrap(), 0b0001);
        // Skip past the 32-bit single-call cap: 24 remaining bits in one
        // call routes through the chunked loop.
        r.skip_bits(24).unwrap();
        // Stream is exhausted; one more bit is Truncated.
        assert!(matches!(r.read_bit(), Err(Error::Truncated)));
    }

    #[test]
    fn skip_bits_zero_is_a_noop() {
        let bytes = [0xAB];
        let mut r = BitReader::new(&bytes);
        r.skip_bits(0).unwrap();
        // Still reads the first bit (MSB of 0xAB = 1).
        assert_eq!(r.read_bit().unwrap(), 1);
    }

    #[test]
    fn skip_bits_past_end_is_truncated() {
        let bytes = [0xAB];
        let mut r = BitReader::new(&bytes);
        assert!(matches!(r.skip_bits(9), Err(Error::Truncated)));
    }

    #[test]
    fn read_uvar_n2_examples_from_tr156() {
        // spec/02 §2.1 reproduces TR.156's worked examples for n = 2:
        //   "100" = 0, "101" = 1, "110" = 2, "111" = 3,
        //   "0100" = 4, "0111" = 7, "00100" = 8, "0000100" = 16.
        let cases: &[(u32, &[u32])] = &[
            (0, &[1, 0, 0]),
            (1, &[1, 0, 1]),
            (2, &[1, 1, 0]),
            (3, &[1, 1, 1]),
            (4, &[0, 1, 0, 0]),
            (7, &[0, 1, 1, 1]),
            (8, &[0, 0, 1, 0, 0]),
            (16, &[0, 0, 0, 0, 1, 0, 0]),
        ];
        for (expected, bits) in cases {
            // Pack the bit sequence into a byte buffer MSB-first.
            let mut buf = Vec::new();
            let mut byte = 0u8;
            let mut n = 0u32;
            for &b in *bits {
                byte = (byte << 1) | (b as u8);
                n += 1;
                if n == 8 {
                    buf.push(byte);
                    byte = 0;
                    n = 0;
                }
            }
            if n > 0 {
                // Left-justify the remaining bits at the MSB end of
                // the trailing byte.
                buf.push(byte << (8 - n));
            }
            let mut r = BitReader::new(&buf);
            assert_eq!(
                r.read_uvar(2).unwrap(),
                *expected,
                "uvar(2) of {:?} should be {}",
                bits,
                expected
            );
        }
    }

    #[test]
    fn read_uvar_n0_is_unary_only() {
        // `uvar(0)` consumes only the prefix-zeros + terminator. A
        // lone `1` bit decodes to 0.
        let bytes = [0b1000_0000];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_uvar(0).unwrap(), 0);
        // The next bit is 0, the bit after is 0, ...; an
        // ever-zero stream would overflow the prefix cap. A 5-zero
        // prefix followed by a terminator should decode to 5.
        let bytes2 = [0b0000_0100];
        let mut r2 = BitReader::new(&bytes2);
        assert_eq!(r2.read_uvar(0).unwrap(), 5);
    }

    #[test]
    fn read_svar_unfolds_one_complement_per_spec02_2_2() {
        // spec/02 §2.2 — the one's-complement folding:
        //   u = 0 -> s =  0
        //   u = 1 -> s = -1
        //   u = 2 -> s =  1
        //   u = 3 -> s = -2
        //   u = 4 -> s =  2
        //   u = 5 -> s = -3
        //
        // Build a buffer of consecutive `uvar(0)` codes (each = unary
        // count + terminator), then read them back as `svar(0)`. The
        // n=0 form is the smallest-width case and the easiest to
        // hand-pack:
        //   u = 0 -> bits "1"
        //   u = 1 -> bits "01"
        //   u = 2 -> bits "001"
        //   u = 3 -> bits "0001"
        //   u = 4 -> bits "00001"
        //   u = 5 -> bits "000001"
        //
        // Concatenated: 1 01 001 0001 00001 000001 = 21 bits, packed
        // MSB-first into 3 bytes.
        let bits = [
            1, // u=0
            0, 1, // u=1
            0, 0, 1, // u=2
            0, 0, 0, 1, // u=3
            0, 0, 0, 0, 1, // u=4
            0, 0, 0, 0, 0, 1, // u=5
        ];
        let mut buf = Vec::new();
        let mut byte = 0u8;
        let mut n = 0u32;
        for &b in &bits {
            byte = (byte << 1) | (b as u8);
            n += 1;
            if n == 8 {
                buf.push(byte);
                byte = 0;
                n = 0;
            }
        }
        if n > 0 {
            buf.push(byte << (8 - n));
        }
        let mut r = BitReader::new(&buf);
        for &expected in &[0i64, -1, 1, -2, 2, -3] {
            assert_eq!(r.read_svar(0).unwrap(), expected);
        }
    }

    #[test]
    fn read_svar_n2_examples() {
        // Pack uvar(2) codes for u = 0..=4, decode as svar(2):
        //   u=0 -> "1 00" -> s = 0
        //   u=1 -> "1 01" -> s = -1
        //   u=2 -> "1 10" -> s = 1
        //   u=3 -> "1 11" -> s = -2
        //   u=4 -> "0 1 00" -> s = 2
        let bits = [
            1, 0, 0, // u=0
            1, 0, 1, // u=1
            1, 1, 0, // u=2
            1, 1, 1, // u=3
            0, 1, 0, 0, // u=4
        ];
        let mut buf = Vec::new();
        let mut byte = 0u8;
        let mut n = 0u32;
        for &b in &bits {
            byte = (byte << 1) | (b as u8);
            n += 1;
            if n == 8 {
                buf.push(byte);
                byte = 0;
                n = 0;
            }
        }
        if n > 0 {
            buf.push(byte << (8 - n));
        }
        let mut r = BitReader::new(&buf);
        for &expected in &[0i64, -1, 1, -2, 2] {
            assert_eq!(r.read_svar(2).unwrap(), expected);
        }
    }

    #[test]
    fn read_ulong_matches_spec02_section_6_1() {
        // spec/02 §6.1: decoding `H_filetype` from the start of byte
        // `0x05` yields `H_filetype = 5`. The first 7 bits are
        // `1 1 1 1 1 0 1` (top 7 bits of 0xFB).
        //
        //  ulong():
        //    uvar(2) for w:  bit0=1 (terminator), mantissa "11" = 3 → w = 3.
        //    uvar(3) for v:  bit3=1 (terminator), mantissa "101" = 5 → v = 5.
        //
        // We construct a 1-byte buffer holding exactly those 7 bits
        // left-padded into byte `0xFA` (`1111_1010`). Reading the 8th
        // bit isn't required for the 7-bit decode.
        //
        // 0xFB = 1111_1011 — we use the actual fixture byte and stop
        // at bit 7.
        let bytes = [0xFB];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_ulong().unwrap(), 5);
    }
}
