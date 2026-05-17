//! 64-bit bit-reservoir writer for batched residual encode.
//!
//! Round 9 hot-path optimisation, symmetric to round 6's
//! [`crate::bitstream64::Bitstream64`] reader. The round-2 `BitWriter`
//! shifted into a single-byte accumulator one bit at a time even when
//! `write_bits(value, n)` was called with `n > 1`, paying a per-bit
//! load-mask-store-branch cost on every mantissa bit emitted. For a
//! 64-block × 4096-sample × 2-channel stereo encode the residual
//! stream is several million bits long, and the per-bit cost
//! dominates encode wallclock.
//!
//! This reservoir writer keeps the unfinished output in a left-justified
//! `u64` accumulator and flushes complete bytes out the high end as the
//! accumulator fills, in a single `to_be_bytes` store per 8 bytes. The
//! mantissa bits of any single `uvar(n)` call (with `n <= 32`) and the
//! terminating `1` bit of any prefix run that fits in 64 zeros are
//! emitted with **two** writes — one for the `n + 1` post-prefix bits
//! and one (only when needed) for an over-long prefix. Long zero
//! prefixes flush whole `u64` zeros in 8-byte chunks.
//!
//! # Layout
//!
//! `buf` holds the next bits to be emitted, **left-justified at bit 63**.
//! `filled_bits` is the count of bits already written into the high
//! `filled_bits` of `buf` (i.e., the high `filled_bits` of `buf` are
//! the next bits in the bit stream, MSB-first; the remaining low bits
//! of `buf` are placeholders for future writes).
//!
//! `finish()` (or `flush_into()`) drains the accumulator and pads the
//! final partial byte with zero bits up to a byte boundary — exactly
//! what the round-2 `BitWriter::finish` did.

/// 64-bit bit reservoir writer. See module docs.
pub(crate) struct BitWriter64 {
    /// Completed bytes already evicted from the reservoir.
    bytes: Vec<u8>,
    /// Reservoir; high `filled_bits` bits are pending output, MSB-first.
    buf: u64,
    /// Number of valid bits in `buf` (0..=64).
    filled_bits: u32,
}

impl BitWriter64 {
    /// Construct an empty writer.
    #[cfg(test)]
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            bytes: Vec::new(),
            buf: 0,
            filled_bits: 0,
        }
    }

    /// Construct an empty writer with capacity hint for the output
    /// `Vec<u8>`.
    #[inline]
    pub(crate) fn with_capacity(cap: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(cap),
            buf: 0,
            filled_bits: 0,
        }
    }

    /// Append a single bit (LSB of `b`).
    #[inline]
    pub(crate) fn write_bit(&mut self, b: u32) {
        // Place the bit at position 63 - filled_bits.
        // SAFETY-AT-LOGIC: filled_bits < 64 between calls; the shift
        // amount `63 - filled_bits` is in `0..=63`.
        self.buf |= ((b as u64) & 1) << (63 - self.filled_bits);
        self.filled_bits += 1;
        if self.filled_bits >= 64 {
            self.drain_full();
        }
    }

    /// Append `n` MSB-first bits from `value` (the high `32 - n` bits
    /// of `value` are ignored). `n <= 32`.
    #[inline]
    pub(crate) fn write_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        if n == 0 {
            return;
        }
        // Mask `value` to its low `n` bits — defensive; callers should
        // pass values already in range, but a stray high bit would
        // smear into the next pending field.
        let mask: u64 = if n == 32 {
            u64::from(u32::MAX)
        } else {
            (1u64 << n) - 1
        };
        let v = (value as u64) & mask;
        // Total bits in the reservoir after this insert.
        let target = self.filled_bits + n;
        if target <= 64 {
            // Place `v` so its high bit lands at slot `63 - filled_bits`.
            let shift = 64 - target;
            self.buf |= v << shift;
            self.filled_bits = target;
            if self.filled_bits >= 64 {
                self.drain_full();
            }
            return;
        }
        // Spans the reservoir boundary. Place the high `top` bits of
        // `v` to fill the reservoir to 64 then drain; the low `n - top`
        // bits become the start of the next reservoir window.
        let top = 64 - self.filled_bits;
        let low = n - top;
        // High `top` bits of `v` are `v >> low`.
        let high_part = v >> low;
        self.buf |= high_part;
        self.filled_bits = 64;
        self.drain_full();
        // Remaining `low` bits.
        let low_mask: u64 = (1u64 << low) - 1;
        let rest = v & low_mask;
        // Place at slot 63 - 0 - (low - 1) = 64 - low.
        let shift = 64 - low;
        self.buf |= rest << shift;
        self.filled_bits = low;
    }

    /// Append a `uvar(n)` code: `zeros = value >> n` leading zero bits,
    /// a terminating `1` bit, then `n` mantissa bits MSB-first.
    ///
    /// This is the hot path for residual encode. The `n` mantissa bits
    /// plus the terminator are written together when the prefix is
    /// short (the common case for narrow `uvar` codes); long prefixes
    /// drain whole 8-byte zero chunks before the mantissa write.
    #[inline]
    pub(crate) fn write_uvar(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        let high = value >> n;
        let mantissa = if n == 0 {
            0u32
        } else {
            value & ((1u32 << n) - 1)
        };
        // Emit `high` zero bits — flush in 8-byte chunks when possible.
        if high > 0 {
            self.write_zeros(high);
        }
        // Terminator + mantissa share one combined write of `1 + n`
        // bits when `n <= 31`. For the `n == 32` edge case (rare;
        // decoder rejects width > 31 in practice) we split into two
        // writes to avoid the `1u32 << 32` UB.
        if n == 32 {
            self.write_bit(1);
            self.write_bits(mantissa, 32);
            return;
        }
        let combined: u32 = (1u32 << n) | mantissa;
        self.write_bits(combined, n + 1);
    }

    /// Emit `count` zero bits efficiently. Whole 64-bit reservoir
    /// fills are flushed as zeros via `drain_full`; the residual is
    /// re-routed into the bit accumulator via the normal path.
    #[inline]
    fn write_zeros(&mut self, count: u32) {
        let mut remaining = count;
        // Fill the current reservoir to 64 with zeros (filled_bits
        // advances, buf untouched since the bits are already 0).
        let to_fill = (64 - self.filled_bits).min(remaining);
        self.filled_bits += to_fill;
        remaining -= to_fill;
        if self.filled_bits >= 64 {
            self.drain_full();
        }
        // Drain full 64-bit zero reservoirs.
        while remaining >= 64 {
            // buf is currently 0 and filled_bits == 0; emit 8 zero
            // bytes.
            debug_assert_eq!(self.buf, 0);
            debug_assert_eq!(self.filled_bits, 0);
            self.bytes.extend_from_slice(&[0u8; 8]);
            remaining -= 64;
        }
        // The tail < 64 zero bits stays as filled_bits advance.
        self.filled_bits += remaining;
        if self.filled_bits >= 64 {
            self.drain_full();
        }
    }

    /// Drain the full 64-bit reservoir into `bytes`. Caller has
    /// established `filled_bits >= 64`.
    #[inline]
    fn drain_full(&mut self) {
        debug_assert!(self.filled_bits >= 64);
        self.bytes.extend_from_slice(&self.buf.to_be_bytes());
        self.buf = 0;
        self.filled_bits = 0;
    }

    /// Append a `svar(n)` code — one's-complement folding identical to
    /// the round-2 [`crate::encoder::BitWriter::write_svar`].
    #[inline]
    pub(crate) fn write_svar(&mut self, s: i32, n: u32) {
        let u = crate::varint::signed_to_unsigned(s);
        self.write_uvar(u, n);
    }

    /// Append a `ulong()` two-stage code matching the round-2 writer's
    /// width-selection heuristic.
    #[inline]
    pub(crate) fn write_ulong(&mut self, value: u32) {
        let mut w: u32 = 0;
        if value > 0 {
            let bits_needed = 32 - value.leading_zeros();
            w = bits_needed.saturating_sub(2);
        }
        if w > 16 {
            w = 16;
        }
        self.write_uvar(w, crate::varint::ULONGSIZE);
        self.write_uvar(value, w);
    }

    /// Finish and return the produced byte buffer. The final partial
    /// byte (if any) is padded with zero bits on the LSB side. Mirrors
    /// the round-2 `BitWriter::finish` semantics.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        // Emit any complete bytes from the reservoir.
        let full_bytes = (self.filled_bits / 8) as usize;
        if full_bytes > 0 {
            let be = self.buf.to_be_bytes();
            self.bytes.extend_from_slice(&be[..full_bytes]);
            // Clear out the bits we just flushed.
            let cleared_bits = (full_bytes * 8) as u32;
            self.filled_bits -= cleared_bits;
            self.buf <<= cleared_bits;
        }
        // Handle a residual sub-byte (0..=7 bits): pad with zeros on
        // the LSB side.
        if self.filled_bits > 0 {
            // The high `filled_bits` of `buf` are the pending bits;
            // since buf is left-justified at bit 63, the byte we want
            // is the top byte of `buf` shifted so the pending bits
            // occupy the top of one byte.
            let byte = (self.buf >> 56) as u8;
            self.bytes.push(byte);
        }
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the reservoir writer produces the same byte stream
    /// as the round-2 single-bit `BitWriter` over a wide range of
    /// per-bit and multi-bit field widths.
    fn ref_bytes(ops: &[(u32, u32)]) -> Vec<u8> {
        // ops are `(value, n)` tuples to feed write_bits.
        // Mirrors the round-2 `BitWriter::write_bits` semantics
        // bit-by-bit so we can verify against a known-correct
        // baseline.
        let mut out: Vec<u8> = Vec::new();
        let mut cur: u8 = 0;
        let mut cur_bits: u32 = 0;
        for &(value, n) in ops {
            for i in (0..n).rev() {
                let bit = (value >> i) & 1;
                cur = (cur << 1) | (bit as u8);
                cur_bits += 1;
                if cur_bits == 8 {
                    out.push(cur);
                    cur = 0;
                    cur_bits = 0;
                }
            }
        }
        if cur_bits > 0 {
            cur <<= 8 - cur_bits;
            out.push(cur);
        }
        out
    }

    #[test]
    fn write_bits_matches_reference_short() {
        // Walking pattern: 3-bit, 5-bit, 7-bit fields.
        let ops: &[(u32, u32)] = &[(0b101, 3), (0b10110, 5), (0b1010111, 7)];
        let mut w = BitWriter64::new();
        for &(v, n) in ops {
            w.write_bits(v, n);
        }
        assert_eq!(w.finish(), ref_bytes(ops));
    }

    #[test]
    fn write_bits_long_run_matches_reference() {
        let mut ops: Vec<(u32, u32)> = Vec::new();
        let mut x: u32 = 0xDEAD_BEEF;
        for _ in 0..200 {
            x = x.wrapping_mul(1103515245).wrapping_add(12345);
            let n = ((x >> 28) & 0xF).max(1); // n in 1..=15
            let mask = if n == 0 { 0 } else { (1u32 << n) - 1 };
            ops.push((x & mask, n));
        }
        let mut w = BitWriter64::new();
        for &(v, n) in &ops {
            w.write_bits(v, n);
        }
        assert_eq!(w.finish(), ref_bytes(&ops));
    }

    #[test]
    fn write_bit_matches_reference() {
        let bits: &[u32] = &[1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1];
        let mut w = BitWriter64::new();
        for &b in bits {
            w.write_bit(b);
        }
        let bytes = w.finish();
        // Expected: 1010_0111 0100_1010 0... = 0xA7 0x4A 0x00 (3 trailing 0 pad bits).
        assert_eq!(bytes, vec![0xA7, 0x4A]);
    }

    #[test]
    fn write_uvar_matches_round2_writer() {
        // Cover the TR.156 examples and a wide value range.
        let cases: &[(u32, u32)] = &[
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
            (7, 2),
            (8, 2),
            (16, 2),
            (0, 0),
            (5, 0),
            (255, 8),
            (0x1234, 12),
        ];
        for &(v, n) in cases {
            let mut w = BitWriter64::new();
            w.write_uvar(v, n);
            let observed = w.finish();
            let mut ref_w = crate::encoder::BitWriter::new();
            ref_w.write_uvar(v, n);
            let expected = ref_w.finish();
            assert_eq!(
                observed, expected,
                "uvar({v}, {n}) mismatch: {observed:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn write_uvar_long_run_matches_round2_writer() {
        let mut state: u32 = 0xC0FE_BABE;
        let mut w64 = BitWriter64::new();
        let mut w2 = crate::encoder::BitWriter::new();
        for _ in 0..500 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let n = ((state >> 28) & 0xF).max(1); // n in 1..=15
            let v = state & ((1u32 << n.min(31)) - 1);
            w64.write_uvar(v, n);
            w2.write_uvar(v, n);
        }
        assert_eq!(w64.finish(), w2.finish());
    }

    #[test]
    fn write_uvar_high_prefix_drains_zero_bytes() {
        // value = 1 << 10, n = 0 → 1024 zero bits then a 1.
        let mut w = BitWriter64::new();
        w.write_uvar(1024, 0);
        let bytes = w.finish();
        // 1024 zeros + 1 = 1025 bits = 128 zero bytes + 0x80 + 7 pad zeros.
        assert_eq!(bytes.len(), 129);
        assert!(bytes[..128].iter().all(|&b| b == 0));
        assert_eq!(bytes[128], 0x80);
    }

    #[test]
    fn round_trip_via_bit_reader() {
        // Write a known stream and read it back via the production
        // BitReader to verify MSB-first ordering is preserved.
        use crate::bitreader::BitReader;
        let mut w = BitWriter64::new();
        w.write_bits(0b1010, 4);
        w.write_bits(0xFF, 8);
        w.write_bits(0x1234, 16);
        let bytes = w.finish();
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_bits(4).unwrap(), 0b1010);
        assert_eq!(br.read_bits(8).unwrap(), 0xFF);
        assert_eq!(br.read_bits(16).unwrap(), 0x1234);
    }

    #[test]
    fn finalize_pads_partial_byte_with_zeros() {
        // Write 11 bits, last 5 of the final byte should be zero.
        let mut w = BitWriter64::new();
        w.write_bits(0b101_0101_0111, 11);
        let bytes = w.finish();
        assert_eq!(bytes.len(), 2);
        assert_eq!(bytes[0], 0b1010_1010);
        // Top 3 bits of byte[1] are 111; low 5 bits zero.
        assert_eq!(bytes[1], 0b1110_0000);
    }
}
