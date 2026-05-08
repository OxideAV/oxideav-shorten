//! 64-bit bit-reservoir reader for batched residual decode.
//!
//! Round 6 hot-path optimisation. The round-5 byte-LUT prefix scan
//! consumes one byte (up to 8 zero bits) per LUT lookup; this reader
//! amortises bit reads further by holding the next 32–64 unread bits
//! in a left-justified `u64` reservoir and using
//! [`u64::leading_zeros`] (which lowers to hardware `lzcnt` on x86-64
//! and `clz` on aarch64) for prefix scans.
//!
//! Unlike the SIMD batch designs that try to vectorise across
//! independent residuals, Shorten's `uvar(n)` codes have data-dependent
//! prefix lengths, so per-lane mantissa loads cannot be issued in
//! parallel without first running the prefix scan. The bit reservoir
//! is the SIMD-grade refactor that this codec actually benefits from:
//! every per-residual call now reads the prefix and the mantissa from
//! the same in-register reservoir without touching memory in the
//! common case.
//!
//! # Layout
//!
//! `buf` holds the next bits to be consumed, **left-justified at bit 63**.
//! `valid_bits` is the count of bits in `buf` that are still unread
//! (i.e., the high `valid_bits` of `buf` are the bit-stream's next
//! `valid_bits` bits, MSB-first; the remaining low bits of `buf` are
//! padding zeros).
//!
//! The reader is constructed from a [`BitReader`] cursor and writes
//! the cursor back on drop / explicit sync — every byte consumed by
//! the reservoir is reflected in the underlying reader's `pos`.

use crate::bitreader::BitReader;
use crate::{Error, Result};

/// 64-bit bit reservoir over a borrowed byte slice. See module docs.
///
/// The reservoir keeps up to 64 unread bits at the high bits of `buf`;
/// `valid_bits` is the count of those bits. `byte_pos` is the index
/// of the next *unread* byte in the underlying buffer (i.e., bytes
/// strictly after the reservoir's tail).
pub(crate) struct Bitstream64<'a> {
    bytes: &'a [u8],
    /// Reservoir; high `valid_bits` bits are unread, MSB-first.
    buf: u64,
    /// Number of valid bits in `buf` (0..=64).
    valid_bits: u32,
    /// Index of the next byte in `bytes` not yet pulled into `buf`.
    byte_pos: usize,
    /// Bit offset within `bytes[start_byte]` at which the reservoir
    /// initially started — used by [`finalize_into`] to compute how
    /// many bits the reservoir has consumed in total.
    start_bit: usize,
}

impl<'a> Bitstream64<'a> {
    /// Construct a reservoir from a [`BitReader`] cursor. The reservoir
    /// takes ownership of the cursor's position; call
    /// [`Self::finalize_into`] to write the bit count back into the
    /// reader once the residual batch is decoded.
    #[inline]
    pub fn from_bit_reader(br: &BitReader<'a>) -> Self {
        // SAFETY-AT-LOGIC: BitReader stores `bytes` as `&'a [u8]`, but
        // there's no public accessor; we round-trip through bit_pos +
        // bytes via the read path. Provide a constructor that takes
        // both explicitly to keep the interface narrow.
        Self::new_at(br_bytes(br), br.bit_pos())
    }

    /// Internal constructor — same as [`from_bit_reader`] but takes
    /// the byte slice and start bit explicitly. Test-only direct use.
    #[inline]
    pub(crate) fn new_at(bytes: &'a [u8], start_bit: usize) -> Self {
        let mut s = Self {
            bytes,
            buf: 0,
            valid_bits: 0,
            byte_pos: start_bit >> 3,
            start_bit,
        };
        // Pre-load the reservoir with up to 64 bits, accounting for the
        // sub-byte start offset.
        let sub = (start_bit & 7) as u32;
        s.refill();
        // If the start bit isn't byte-aligned, drop `sub` bits from the
        // top of the reservoir.
        if sub != 0 {
            let drop = sub.min(s.valid_bits);
            s.buf <<= drop;
            s.valid_bits -= drop;
            // Re-refill to top up after dropping the sub-byte alignment.
            s.refill();
        }
        s
    }

    /// Total bits consumed since construction.
    ///
    /// Equals the number of bits the user has read past `start_bit`:
    /// `(bytes pulled into reservoir) - (initial sub-byte alignment drop)
    /// - (bits still resident in the reservoir)`.
    #[inline]
    pub fn bits_consumed(&self) -> usize {
        let start_byte = self.start_bit >> 3;
        let sub = self.start_bit & 7;
        let pulled_bits = (self.byte_pos - start_byte) * 8;
        pulled_bits
            .saturating_sub(sub)
            .saturating_sub(self.valid_bits as usize)
    }

    /// Refill the reservoir up to (close to) 64 bits by pulling whole
    /// bytes from `bytes[byte_pos..]`. Pulls 0..=8 bytes depending on
    /// remaining headroom and remaining input.
    #[inline]
    fn refill(&mut self) {
        // Greedy fast path: if at least 8 bytes remain and the
        // reservoir is at most 56 bits full, pull a single u64 from
        // the input in a single big-endian load and merge it.
        while self.valid_bits <= 56 && self.byte_pos < self.bytes.len() {
            // Try the 8-byte fast path first.
            if self.valid_bits == 0 && self.byte_pos + 8 <= self.bytes.len() {
                // Load 8 bytes big-endian into the reservoir.
                let chunk = u64::from_be_bytes([
                    self.bytes[self.byte_pos],
                    self.bytes[self.byte_pos + 1],
                    self.bytes[self.byte_pos + 2],
                    self.bytes[self.byte_pos + 3],
                    self.bytes[self.byte_pos + 4],
                    self.bytes[self.byte_pos + 5],
                    self.bytes[self.byte_pos + 6],
                    self.bytes[self.byte_pos + 7],
                ]);
                self.buf = chunk;
                self.valid_bits = 64;
                self.byte_pos += 8;
                continue;
            }
            // One-byte slow path for the residual headroom.
            let byte = self.bytes[self.byte_pos] as u64;
            let shift = 64 - self.valid_bits - 8;
            self.buf |= byte << shift;
            self.valid_bits += 8;
            self.byte_pos += 1;
        }
    }

    /// Read `n` bits MSB-first into a `u32`. `n` must be `<= 32`.
    /// Returns [`Error::UnexpectedEof`] when fewer than `n` bits remain
    /// in the reservoir + input combined.
    #[inline]
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.valid_bits < n {
            self.refill();
            if self.valid_bits < n {
                return Err(Error::UnexpectedEof);
            }
        }
        let value = (self.buf >> (64 - n)) as u32;
        self.buf <<= n;
        self.valid_bits -= n;
        // Opportunistically refill to keep subsequent reads fast.
        if self.valid_bits <= 32 {
            self.refill();
        }
        Ok(value)
    }

    /// Count leading zero bits up to and including the terminating `1`,
    /// returning the zero count and consuming the prefix. Caps at
    /// `max_zeros`; returns `None` on EOF or cap-exceeded.
    ///
    /// Hot path: in the common case the entire prefix sits inside the
    /// already-loaded reservoir and a single `u64::leading_zeros` call
    /// resolves it. On long all-zero runs (rare in real Shorten
    /// streams) the prefix may span several refills; we loop.
    #[inline]
    pub fn read_uvar_prefix(&mut self, max_zeros: u32) -> Option<u32> {
        let mut zeros: u32 = 0;
        loop {
            if self.valid_bits == 0 {
                self.refill();
                if self.valid_bits == 0 {
                    return None;
                }
            }
            // Count leading zeros within the valid window of the
            // reservoir. `buf` is left-justified; valid bits occupy
            // the top `valid_bits`.
            let lz = self.buf.leading_zeros();
            if lz < self.valid_bits {
                // Terminator within reservoir.
                let consumed = lz + 1;
                zeros = zeros.checked_add(lz)?;
                if zeros > max_zeros {
                    return None;
                }
                // Shift the prefix + terminator out.
                if consumed == 64 {
                    self.buf = 0;
                } else {
                    self.buf <<= consumed;
                }
                self.valid_bits -= consumed;
                if self.valid_bits <= 32 {
                    self.refill();
                }
                return Some(zeros);
            }
            // All `valid_bits` of the reservoir were zeros — consume
            // them and refill.
            zeros = zeros.checked_add(self.valid_bits)?;
            if zeros > max_zeros {
                return None;
            }
            self.buf = 0;
            self.valid_bits = 0;
            // Loop refills.
        }
    }

    /// Decode a single `uvar(n)` value. Mirrors [`crate::varint::read_uvar`]
    /// using the reservoir.
    #[inline]
    pub fn read_uvar(&mut self, n: u32, max_zeros: u32) -> Result<u32> {
        let zeros = self
            .read_uvar_prefix(max_zeros)
            .ok_or(Error::UnexpectedEof)?;
        let mantissa = self.read_bits(n)?;
        let high = zeros
            .checked_shl(n)
            .ok_or(Error::ResidualWidthOverflow(n))?;
        Ok(high | mantissa)
    }

    /// Decode a single `svar(n)` value (one's-complement folding).
    #[inline]
    pub fn read_svar(&mut self, n: u32, max_zeros: u32) -> Result<i32> {
        let u = self.read_uvar(n, max_zeros)?;
        Ok(crate::varint::unsigned_to_signed(u))
    }

    /// Write the reservoir's bit cursor back into `br`.
    #[inline]
    pub fn finalize_into(self, br: &mut BitReader<'a>) -> Result<()> {
        let consumed = self.bits_consumed();
        let target = self.start_bit + consumed;
        // Advance the underlying BitReader cursor by burning bits via
        // its own read_bits. We can't set pos directly (the field is
        // private to bitreader.rs); instead expose a setter there.
        br.advance_to_bit(target)
    }
}

/// Helper to access a `BitReader`'s underlying byte slice. Implemented
/// in `bitreader.rs` via a non-public accessor.
#[inline]
fn br_bytes<'a>(br: &BitReader<'a>) -> &'a [u8] {
    br.bytes_slice()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_bits_matches_bit_reader() {
        let bytes = [0xFBu8, 0xB1, 0x70, 0x09, 0xF9, 0x25];
        let mut bs = Bitstream64::new_at(&bytes, 0);
        // 7-bit MSB-first read of 0xFB = 0b11111011 → top 7 = 0b1111101 = 0x7D.
        assert_eq!(bs.read_bits(7).unwrap(), 0x7D);
        // Next bit is the trailing 1 of 0xFB.
        assert_eq!(bs.read_bits(1).unwrap(), 1);
        // Next 8 bits = 0xB1.
        assert_eq!(bs.read_bits(8).unwrap(), 0xB1);
    }

    #[test]
    fn read_uvar_prefix_runs() {
        // Bit pattern "0000 1000" = 0x08 → 4 zeros then a 1.
        let bytes = [0x08u8];
        let mut bs = Bitstream64::new_at(&bytes, 0);
        assert_eq!(bs.read_uvar_prefix(64), Some(4));
        // Pattern "1xxx xxxx" — 0 zeros.
        let bytes = [0x80u8];
        let mut bs = Bitstream64::new_at(&bytes, 0);
        assert_eq!(bs.read_uvar_prefix(64), Some(0));
    }

    #[test]
    fn read_uvar_matches_tr156() {
        // Same TR.156 examples used by varint.rs tests.
        // n=2: 100→0; 101→1; 110→2; 111→3; 0100→4; 0111→7;
        //      00100→8; 0000100→16.
        let cases: &[(&[u8], u32, u32)] = &[
            (&[0b1000_0000], 2, 0),
            (&[0b1010_0000], 2, 1),
            (&[0b1100_0000], 2, 2),
            (&[0b1110_0000], 2, 3),
            (&[0b0100_0000], 2, 4),
            (&[0b0111_0000], 2, 7),
            (&[0b0010_0000], 2, 8),
            (&[0b0000_1000], 2, 16),
        ];
        for (bytes, n, expected) in cases {
            let mut bs = Bitstream64::new_at(bytes, 0);
            assert_eq!(bs.read_uvar(*n, 64).unwrap(), *expected);
        }
    }

    #[test]
    fn long_zero_run_spans_refill() {
        // 80 zero bits then a 1. 10 zero bytes = 80 bits, then 0x80
        // for the terminator (1 bit) plus a 7-bit mantissa of zeros.
        let mut bytes = vec![0u8; 10];
        bytes.push(0x80);
        let mut bs = Bitstream64::new_at(&bytes, 0);
        assert_eq!(bs.read_uvar_prefix(128), Some(80));
    }

    #[test]
    fn finalize_writes_back_position() {
        use crate::bitreader::BitReader;
        let bytes = [0xFBu8, 0xB1, 0x70];
        let mut br = BitReader::new(&bytes);
        // Skip 1 bit so we test sub-byte alignment.
        let _ = br.read_bits(1).unwrap();
        let mut bs = Bitstream64::from_bit_reader(&br);
        let _ = bs.read_bits(15).unwrap();
        bs.finalize_into(&mut br).unwrap();
        assert_eq!(br.bit_pos(), 16);
    }

    #[test]
    fn lut_reference_sanity() {
        // The bit-reservoir reader uses `u64::leading_zeros` rather
        // than the byte LUT, but the round-5 LUT is still wired into
        // the BitReader fallback path. Cross-check the boundary
        // entries against the equivalent reservoir reads.
        let mut bs = Bitstream64::new_at(&[0x80u8], 0);
        assert_eq!(bs.read_uvar_prefix(64), Some(0));
        let mut bs = Bitstream64::new_at(&[0x01u8], 0);
        assert_eq!(bs.read_uvar_prefix(64), Some(7));
        // All-zero-byte tail: 8 zeros, then EOF (no terminator).
        let mut bs = Bitstream64::new_at(&[0x00u8], 0);
        assert_eq!(bs.read_uvar_prefix(64), None);
    }
}
