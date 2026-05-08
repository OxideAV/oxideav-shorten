//! MSB-first bit reader.
//!
//! From the byte at file offset `0x05` onward, every Shorten field is
//! consumed under a single bit-stream variable-length integer scheme
//! (see `spec/02-variable-length-coding.md` §1). Within each byte the
//! bits are read **most-significant bit first**: the first bit consumed
//! is bit 7 (mask `0x80`), the last is bit 0 (mask `0x01`).
//!
//! The reader exposed here owns a byte slice and a bit cursor measured
//! in bits from the start of the slice. Reads beyond end-of-buffer
//! return [`crate::Error::UnexpectedEof`].

use crate::{Error, Result};

/// MSB-first bit cursor over a borrowed byte slice.
///
/// All reads advance the cursor; there is no peek-and-restore. Reads
/// at or past `bytes.len() * 8` return [`Error::UnexpectedEof`].
#[derive(Debug, Clone)]
pub(crate) struct BitReader<'a> {
    bytes: &'a [u8],
    /// Bit cursor measured from the start of `bytes`. Bit 0 is MSB of
    /// `bytes[0]`; bit 7 is LSB of `bytes[0]`; bit 8 is MSB of
    /// `bytes[1]`, etc.
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// Construct a reader positioned at bit 0 of `bytes`.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Total bit length of the underlying buffer.
    pub fn total_bits(&self) -> usize {
        self.bytes.len().saturating_mul(8)
    }

    /// Current bit cursor (in bits from buffer start).
    pub fn bit_pos(&self) -> usize {
        self.pos
    }

    /// Number of bits still readable without raising EOF.
    pub fn bits_remaining(&self) -> usize {
        self.total_bits().saturating_sub(self.pos)
    }

    /// Borrow the underlying byte slice. Used by
    /// [`crate::bitstream64::Bitstream64`] to construct a reservoir
    /// reader that decodes residuals in batch from the same buffer.
    #[inline]
    pub(crate) fn bytes_slice(&self) -> &'a [u8] {
        self.bytes
    }

    /// Advance the cursor to `target_bit` (which must be `>= bit_pos`).
    /// Used to write back the bit count consumed by a transient
    /// reservoir reader (see [`crate::bitstream64::Bitstream64`]).
    /// Returns [`Error::UnexpectedEof`] if `target_bit` exceeds
    /// `total_bits()`.
    #[inline]
    pub(crate) fn advance_to_bit(&mut self, target_bit: usize) -> Result<()> {
        debug_assert!(target_bit >= self.pos);
        if target_bit > self.total_bits() {
            return Err(Error::UnexpectedEof);
        }
        self.pos = target_bit;
        Ok(())
    }

    /// Read a single bit, advancing the cursor.
    #[allow(dead_code)]
    pub fn read_bit(&mut self) -> Result<u32> {
        if self.pos >= self.total_bits() {
            return Err(Error::UnexpectedEof);
        }
        let byte = self.bytes[self.pos >> 3];
        // Bit 0 of `pos % 8` selects the most significant in-byte bit.
        let shift = 7 - (self.pos & 7);
        self.pos += 1;
        Ok(((byte >> shift) & 1) as u32)
    }

    /// Read `n` bits MSB-first into a `u32`. `n` must be `<= 32`.
    ///
    /// Returns [`Error::UnexpectedEof`] if fewer than `n` bits remain
    /// in the buffer; on error the cursor's position is unspecified
    /// (the decoder treats EOF as terminal and does not retry).
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if (n as usize) > self.bits_remaining() {
            return Err(Error::UnexpectedEof);
        }
        let mut value: u32 = 0;
        for _ in 0..n {
            // Inline read_bit body — saves a branch per bit on hot
            // residual reads.
            let byte = self.bytes[self.pos >> 3];
            let shift = 7 - (self.pos & 7);
            value = (value << 1) | (((byte >> shift) & 1) as u32);
            self.pos += 1;
        }
        Ok(value)
    }

    /// Skip the bits up to the next byte boundary. Used after the
    /// `BLOCK_FN_QUIT` command (`spec/05-state-and-quirks.md` §4) to
    /// align the cursor to the file's last byte boundary.
    ///
    /// Returns the number of bits skipped (0..=7).
    #[allow(dead_code)]
    pub fn align_to_byte(&mut self) -> u32 {
        let pad = (8 - (self.pos & 7)) & 7;
        self.pos += pad;
        pad as u32
    }

    /// Round-5 hot-path helper: count the leading zero bits up to and
    /// including the terminating `1` for a `uvar(n)` prefix. Returns
    /// the zero count (cursor advanced past the terminator). Uses
    /// the 256-entry [`UVAR_PREFIX_LUT`] to consume a full byte of
    /// zeros at a time when the cursor is byte-aligned, and falls
    /// back to a bit-by-bit scan when straddling byte boundaries.
    ///
    /// Returns `None` on EOF (the caller maps that to
    /// [`crate::Error::UnexpectedEof`]).
    #[inline]
    pub fn read_uvar_prefix(&mut self, max_zeros: u32) -> Option<u32> {
        let mut zeros: u32 = 0;
        // Fast path: consume entire bytes of zeros while byte-aligned.
        // The LUT lookup yields the position of the highest set bit
        // (0..=7 from the MSB; 8 if the byte is zero).
        loop {
            let bit_offset = self.pos & 7;
            if bit_offset != 0 {
                break;
            }
            let byte_idx = self.pos >> 3;
            if byte_idx >= self.bytes.len() {
                return None;
            }
            let byte = self.bytes[byte_idx];
            if byte == 0 {
                // Whole byte is zeros — consume it.
                zeros = zeros.checked_add(8)?;
                if zeros > max_zeros {
                    return None;
                }
                self.pos += 8;
                continue;
            }
            // The byte has a 1 somewhere; the LUT entry is the index
            // (0..=7 from the MSB) of the first 1 — i.e. the count of
            // leading zeros within this byte.
            let leading = UVAR_PREFIX_LUT[byte as usize] as u32;
            zeros = zeros.checked_add(leading)?;
            if zeros > max_zeros {
                return None;
            }
            // Consume the leading zeros and the terminating `1` bit.
            self.pos += leading as usize + 1;
            return Some(zeros);
        }
        // Straddling fallback: read bits one at a time.
        loop {
            if self.pos >= self.total_bits() {
                return None;
            }
            let byte = self.bytes[self.pos >> 3];
            let shift = 7 - (self.pos & 7);
            self.pos += 1;
            let bit = (byte >> shift) & 1;
            if bit == 1 {
                return Some(zeros);
            }
            zeros = zeros.checked_add(1)?;
            if zeros > max_zeros {
                return None;
            }
            // After consuming one bit, if we're now byte-aligned, jump
            // back into the fast path.
            if self.pos & 7 == 0 {
                loop {
                    let byte_idx = self.pos >> 3;
                    if byte_idx >= self.bytes.len() {
                        return None;
                    }
                    let b = self.bytes[byte_idx];
                    if b == 0 {
                        zeros = zeros.checked_add(8)?;
                        if zeros > max_zeros {
                            return None;
                        }
                        self.pos += 8;
                        continue;
                    }
                    let leading = UVAR_PREFIX_LUT[b as usize] as u32;
                    zeros = zeros.checked_add(leading)?;
                    if zeros > max_zeros {
                        return None;
                    }
                    self.pos += leading as usize + 1;
                    return Some(zeros);
                }
            }
        }
    }
}

/// 256-entry leading-zero table for an MSB-first byte. Entry `b` is
/// the index (0..=7) of the most-significant set bit, counted from
/// the MSB. For `b = 0` the entry is `8` (no set bit). Used by the
/// round-5 [`BitReader::read_uvar_prefix`] fast path to consume up to
/// 8 zero bits per byte in a single LUT lookup, replacing the
/// bit-by-bit scan loop the round-1 `read_uvar` used.
const fn make_uvar_prefix_lut() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut leading = 0u8;
        let mut bit = 7i32;
        while bit >= 0 {
            if ((i >> bit) & 1) == 1 {
                break;
            }
            leading += 1;
            bit -= 1;
        }
        t[i] = leading;
        i += 1;
    }
    t
}

pub(crate) const UVAR_PREFIX_LUT: [u8; 256] = make_uvar_prefix_lut();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msb_first_single_bits() {
        // 0b1010_0101 = 0xA5 → bit 0 = 1, bit 1 = 0, bit 2 = 1, ...
        let bytes = [0xA5u8];
        let mut br = BitReader::new(&bytes);
        let expected = [1, 0, 1, 0, 0, 1, 0, 1];
        for &exp in &expected {
            assert_eq!(br.read_bit().unwrap(), exp);
        }
        // Ninth read should EOF.
        assert!(matches!(br.read_bit(), Err(Error::UnexpectedEof)));
    }

    #[test]
    fn read_bits_msb_first() {
        // F1 byte 0x05 = 0xFB = 0b1111_1011.
        // First 7 bits MSB-first: 1, 1, 1, 1, 1, 0, 1 → as u32 = 0b1111101 = 0x7D.
        let bytes = [0xFBu8];
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_bits(7).unwrap(), 0x7D);
        assert_eq!(br.bit_pos(), 7);
    }

    #[test]
    fn read_bits_crosses_bytes() {
        // F1 bytes 0x05..0x07 = FB B1 70.
        // bits 0..16 are 1111 1011 1011 0001
        // bits 0..16 as u32 = 0xFBB1.
        let bytes = [0xFBu8, 0xB1, 0x70];
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_bits(16).unwrap(), 0xFBB1);
    }

    #[test]
    fn read_bits_zero_returns_zero() {
        let bytes = [0xFFu8];
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_bits(0).unwrap(), 0);
        assert_eq!(br.bit_pos(), 0);
    }

    #[test]
    fn align_to_byte_pads_with_count() {
        let bytes = [0xFFu8, 0x00];
        let mut br = BitReader::new(&bytes);
        let _ = br.read_bits(3).unwrap();
        let pad = br.align_to_byte();
        assert_eq!(pad, 5);
        assert_eq!(br.bit_pos(), 8);
    }
}
