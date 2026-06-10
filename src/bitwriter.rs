//! MSB-first bit writer — the encode-side counterpart of
//! [`crate::BitReader`].
//!
//! `docs/audio/shorten/spec/02-variable-length-coding.md` §1 pins the
//! bit-stream's reading order ("Within each byte the bits are read
//! **most-significant bit first**: the first bit consumed is bit 7 of
//! the first byte (mask `0x80`), the second is bit 6 (mask `0x40`), and
//! so on…"). The encode-side bit packing is the dual: the first bit
//! written lands in bit 7 of the first byte, the second in bit 6, and
//! so on; the active byte fills MSB-first until eight bits have been
//! deposited, at which point the byte is flushed and the cursor
//! advances.
//!
//! This module exposes the primitive [`BitWriter`] and three helpers
//! the encoder side needs to emit syntactically-valid Shorten wire
//! format:
//!
//! * [`BitWriter::write_uvar`] — the unsigned `uvar(n)` form of
//!   `spec/02` §2.1: emit `⌊v / 2^n⌋` zero bits, a terminating one,
//!   then the value's low `n` bits MSB-first.
//! * [`BitWriter::write_svar`] — the signed `svar(n)` form of
//!   `spec/02` §2.2 (one's-complement folding): non-negative `s` folds
//!   to `u = s << 1`; negative `s` folds to `u = ((~s) << 1) | 1`.
//! * [`BitWriter::write_ulong`] — the two-stage `ulong()` form of
//!   `spec/02` §3: `uvar(ULONGSIZE = 2)` width prefix followed by
//!   `uvar(width)` value.
//!
//! The writer guarantees the invariant that the byte buffer it
//! exposes ([`BitWriter::into_bytes`]) decodes byte-for-byte through
//! [`crate::BitReader`] back to the exact integers it was fed. This is
//! exercised by an in-tree roundtrip test on every `uvar`/`svar`
//! width-class pair the encoder uses and by the higher-level
//! [`crate::encoder::encode_stream_header`] roundtrip against
//! [`crate::header::parse_stream_header`].
//!
//! ## Trailing-bit alignment to the next byte boundary
//!
//! `spec/05` §4 pins the post-`BLOCK_FN_QUIT` byte alignment: the
//! encoder appends zero bits up to the next byte boundary after the
//! 4-bit `BLOCK_FN_QUIT` pattern (`uvar(2)` of value 4 = `0100`) is
//! emitted. [`BitWriter::pad_to_byte`]
//! exposes this rule explicitly; the residual count of padding bits is
//! always in `0..=7`, matching the verification on fixture `F9` (3
//! padding bits) and the §4.1 observation across `F1`, `F4`, `F9`.

use crate::bitreader::ULONGSIZE;

/// MSB-first bit writer over an internal byte vector.
///
/// The active byte fills from bit 7 down to bit 0; once 8 bits have
/// been deposited the byte is flushed onto the output and a fresh
/// active byte is started.
#[derive(Debug, Default)]
pub struct BitWriter {
    out: Vec<u8>,
    /// In-progress byte. Lower-numbered bit positions (closer to
    /// `n_bits`) are yet-unwritten; the most-significant `n_bits` of
    /// this u8 hold the bits already deposited.
    cur: u8,
    /// Count of bits already deposited into `cur`. Always in `0..=7`.
    n_bits: u32,
}

impl BitWriter {
    /// Construct an empty bit writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a writer with capacity reserved for `cap` bytes of
    /// output.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            out: Vec::with_capacity(cap),
            cur: 0,
            n_bits: 0,
        }
    }

    /// Total bits already emitted (whole bytes in `out` × 8 + bits
    /// pending in the active byte).
    pub fn bits_written(&self) -> u64 {
        self.out.len() as u64 * 8 + self.n_bits as u64
    }

    /// Pending-byte fill count: number of bits in the active byte not
    /// yet flushed to `out`. Always in `0..=7`. When zero, the writer
    /// is byte-aligned.
    pub fn pending_bits(&self) -> u32 {
        self.n_bits
    }

    /// True when no bits remain pending in the active byte (the next
    /// write begins at a fresh byte boundary).
    pub fn is_byte_aligned(&self) -> bool {
        self.n_bits == 0
    }

    /// Emit a single bit. `bit & 1` is taken as the value.
    pub fn write_bit(&mut self, bit: u32) {
        self.cur = (self.cur << 1) | ((bit & 1) as u8);
        self.n_bits += 1;
        if self.n_bits == 8 {
            self.out.push(self.cur);
            self.cur = 0;
            self.n_bits = 0;
        }
    }

    /// Emit the low `n` bits of `value` MSB-first. `n` must be in
    /// `0..=32`; `n == 0` is a no-op.
    pub fn write_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32, "write_bits n exceeds u32 width");
        for i in (0..n).rev() {
            self.write_bit((value >> i) & 1);
        }
    }

    /// Emit `value` as `uvar(n)` per `spec/02` §2.1.
    ///
    /// The encoding emits `⌊value / 2^n⌋` zero bits, a terminating one
    /// bit, then the low `n` bits of `value` MSB-first. With `n == 0`
    /// the mantissa is empty and only the prefix-plus-terminator is
    /// emitted (value `v` → `v` zero bits then a single `1`).
    ///
    /// The encoded bit length is `⌊value / 2^n⌋ + 1 + n` bits, matching
    /// the length formula in `spec/02` §2.1.
    pub fn write_uvar(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32, "write_uvar n exceeds u32 width");
        let prefix_zeros = if n == 0 { value } else { value >> n };
        for _ in 0..prefix_zeros {
            self.write_bit(0);
        }
        self.write_bit(1);
        if n > 0 {
            // Mantissa lives in the low `n` bits.
            let mantissa = value & ((1u32 << n) - 1);
            self.write_bits(mantissa, n);
        }
    }

    /// Emit `value` as `svar(n)` per `spec/02` §2.2 (one's-complement
    /// folding).
    ///
    /// Non-negative `s` folds to `u = s << 1`; negative `s` folds to
    /// `u = ((~s) << 1) | 1`. The folded `u` is then emitted as
    /// `uvar(n)`.
    pub fn write_svar(&mut self, value: i64, n: u32) {
        let u: u64 = if value >= 0 {
            (value as u64) << 1
        } else {
            (((!value) as u64) << 1) | 1
        };
        // `spec/02` §2 says the folded form fits in the same width
        // class as `uvar`; in practice the per-block residual widths
        // pinned in `spec/02` §4.2 / §5 / `spec/05` §3 are well below
        // 32 bits. We cap at u32 here because every caller that
        // produces svar payloads bounds its value range upstream.
        debug_assert!(u <= u32::MAX as u64, "svar folded value exceeds u32");
        self.write_uvar(u as u32, n);
    }

    /// Emit `value` as the two-stage `ulong()` form of `spec/02` §3.
    ///
    /// The width parameter `width` is the per-encoded-value natural
    /// mantissa width the encoder chooses for `value`; the wire layout
    /// is `uvar(ULONGSIZE = 2)` over `width` followed by `uvar(width)`
    /// over `value`.
    ///
    /// Per `spec/02` §3 the encoder is free to pick any `width >=
    /// bits_for(value)` (where `bits_for(v) = ⌊log2(v)⌋ + 1` for
    /// `v > 0`, `0` for `v == 0`); a wider choice merely wastes bits.
    /// [`encode_ulong_natural`] applies the minimum-width rule used by
    /// the test fixtures in `src/driver.rs`.
    pub fn write_ulong(&mut self, value: u32, width: u32) {
        self.write_uvar(width, ULONGSIZE);
        self.write_uvar(value, width);
    }

    /// Pad to the next byte boundary by emitting zero bits per
    /// `spec/05` §4. The number of padding bits emitted is in `0..=7`
    /// (zero when the writer is already byte-aligned). Returns the
    /// number of padding bits written.
    pub fn pad_to_byte(&mut self) -> u32 {
        if self.n_bits == 0 {
            return 0;
        }
        let pad = 8 - self.n_bits;
        for _ in 0..pad {
            self.write_bit(0);
        }
        pad
    }

    /// Finalise the writer and return the assembled byte buffer.
    ///
    /// If a partial byte is still pending, it is zero-padded to the
    /// next byte boundary first (per `spec/05` §4). The returned
    /// vector contains every flushed byte in emission order.
    pub fn into_bytes(mut self) -> Vec<u8> {
        if self.n_bits > 0 {
            let pad = 8 - self.n_bits;
            self.cur <<= pad;
            self.out.push(self.cur);
        }
        self.out
    }

    /// Snapshot the assembled byte buffer without consuming the writer.
    ///
    /// If a partial byte is pending it is included as a zero-padded
    /// final byte at the end of the returned vector; the writer's
    /// internal state is unchanged so further writes resume at the
    /// same bit offset.
    pub fn snapshot_bytes(&self) -> Vec<u8> {
        let mut buf = self.out.clone();
        if self.n_bits > 0 {
            let pad = 8 - self.n_bits;
            buf.push(self.cur << pad);
        }
        buf
    }
}

/// Compute the smallest natural mantissa width capable of representing
/// `value` as `uvar(width)`.
///
/// Returns `0` when `value == 0` (a single terminator bit suffices) and
/// `⌊log2(value)⌋ + 1` otherwise.
///
/// `spec/02` §3 specifies the encoder picks the width per-value; the
/// test helpers in `src/driver.rs` use exactly this minimum-width rule,
/// and the same rule is applied by [`crate::encoder::write_stream_header`]
/// for every `ulong()` field of the header parameter block.
pub fn natural_ulong_width(value: u32) -> u32 {
    if value == 0 {
        0
    } else {
        32 - value.leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;

    // ---- elementary primitives ----

    #[test]
    fn writes_empty_buffer_when_nothing_written() {
        let w = BitWriter::new();
        assert_eq!(w.into_bytes(), Vec::<u8>::new());
    }

    #[test]
    fn writes_single_bit_padded_to_byte() {
        let mut w = BitWriter::new();
        w.write_bit(1);
        // MSB-first → bit 7 of the first byte → 0x80.
        assert_eq!(w.into_bytes(), vec![0x80]);
    }

    #[test]
    fn write_bits_msb_first_order() {
        let mut w = BitWriter::new();
        w.write_bits(0b1011_0010, 8);
        assert_eq!(w.into_bytes(), vec![0b1011_0010]);
    }

    #[test]
    fn write_bits_spans_byte_boundary() {
        let mut w = BitWriter::new();
        w.write_bits(0b1010_1010, 8);
        w.write_bits(0b1100, 4);
        // Active byte after 4-bit write: 0b1100 → padded to 0b1100_0000.
        assert_eq!(w.into_bytes(), vec![0b1010_1010, 0b1100_0000]);
    }

    #[test]
    fn pending_bits_tracks_byte_fill() {
        let mut w = BitWriter::new();
        assert_eq!(w.pending_bits(), 0);
        assert!(w.is_byte_aligned());
        w.write_bits(0b101, 3);
        assert_eq!(w.pending_bits(), 3);
        assert!(!w.is_byte_aligned());
        w.write_bits(0b11111, 5);
        assert_eq!(w.pending_bits(), 0);
        assert!(w.is_byte_aligned());
    }

    #[test]
    fn bits_written_includes_pending_active_byte() {
        let mut w = BitWriter::new();
        w.write_bits(0xFF, 8);
        w.write_bits(0b101, 3);
        assert_eq!(w.bits_written(), 11);
    }

    // ---- uvar / svar ----

    #[test]
    fn write_uvar_tr156_examples_n_eq_2() {
        // `spec/02` §2.1 lists examples for n = 2:
        // value -> encoded bit string (and bit count):
        //   0 -> "100"     (3 bits)
        //   1 -> "101"     (3 bits)
        //   2 -> "110"     (3 bits)
        //   3 -> "111"     (3 bits)
        //   4 -> "0100"    (4 bits)
        //   7 -> "0111"    (4 bits)
        //   8 -> "00100"   (5 bits)
        //  16 -> "0000100" (7 bits)
        for &(v, n_bits) in &[
            (0u32, 3u64),
            (1, 3),
            (2, 3),
            (3, 3),
            (4, 4),
            (7, 4),
            (8, 5),
            (16, 7),
        ] {
            let mut w = BitWriter::new();
            w.write_uvar(v, 2);
            assert_eq!(w.bits_written(), n_bits, "value={v}");
            // Read it back through the bit reader to confirm the
            // round-trip.
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.read_uvar(2).expect("read_uvar"), v, "roundtrip v={v}");
        }
    }

    #[test]
    fn write_uvar_n_zero_emits_unary_alpha() {
        // n = 0: value v → v zero bits then a terminating 1.
        // value 0 → "1", value 3 → "0001", value 7 → "00000001".
        let mut w = BitWriter::new();
        w.write_uvar(3, 0);
        assert_eq!(w.bits_written(), 4);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_uvar(0).expect("read_uvar"), 3);
    }

    #[test]
    fn write_svar_roundtrip_signed_range() {
        // Roundtrip a representative spread of signed values through
        // svar(n) for several mantissa widths. The cases pair each
        // mantissa width with values whose folded prefix-zero count
        // stays below the reader's leading-zero safety cap.
        let cases: &[(u32, &[i64])] = &[
            // n=0 has the unary form; prefix-zero count equals the
            // folded unsigned value, which is bounded by the reader's
            // UVAR_PREFIX_CAP. Stay well below.
            (0, &[-2, -1, 0, 1, 2, 5]),
            (1, &[-7, -3, -1, 0, 1, 5, 12]),
            (2, &[-19, -3, -1, 0, 1, 5, 42, 60]),
            (3, &[-19, -3, -1, 0, 1, 5, 42, 127]),
            (4, &[-19, -3, -1, 0, 1, 5, 42, 127]),
            (7, &[-19, -3, -1, 0, 1, 5, 42, 127]),
        ];
        for (n, vs) in cases {
            for &v in *vs {
                let mut w = BitWriter::new();
                w.write_svar(v, *n);
                let bytes = w.into_bytes();
                let mut r = BitReader::new(&bytes);
                let got = r.read_svar(*n).expect("read_svar");
                assert_eq!(got, v, "n={n} v={v}");
            }
        }
    }

    // ---- ulong ----

    #[test]
    fn write_ulong_minimum_width_roundtrip() {
        for v in [0u32, 1, 2, 5, 256, 44_100, 1_731_745] {
            let w_width = natural_ulong_width(v);
            let mut w = BitWriter::new();
            w.write_ulong(v, w_width);
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.read_ulong().expect("read_ulong"), v, "v={v}");
        }
    }

    #[test]
    fn write_ulong_accepts_wider_than_minimum() {
        // Per `spec/02` §3 the encoder may pick any width >= the
        // minimum; the decoder is width-agnostic. Confirm the reader
        // accepts a non-minimum width choice and recovers the value.
        let mut w = BitWriter::new();
        w.write_ulong(5, 6); // value 5, width 6 (minimum is 3).
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_ulong().expect("read_ulong"), 5);
    }

    // ---- byte alignment / pad_to_byte ----

    #[test]
    fn pad_to_byte_zero_padding_at_aligned_boundary() {
        let mut w = BitWriter::new();
        w.write_bits(0xAB, 8); // already byte-aligned.
        assert_eq!(w.pad_to_byte(), 0);
        assert_eq!(w.into_bytes(), vec![0xAB]);
    }

    #[test]
    fn pad_to_byte_writes_correct_pad_count() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        // 5 padding bits to reach the next byte boundary.
        assert_eq!(w.pad_to_byte(), 5);
        assert_eq!(w.bits_written(), 8);
        assert_eq!(w.into_bytes(), vec![0b1010_0000]);
    }

    #[test]
    fn quit_command_at_byte_boundary_pads_to_zeroes() {
        // Per `spec/02` §2.1 worked examples, `uvar(2)` of value 4 is
        // the 4-bit pattern `0100` (one leading zero + terminator +
        // 2-bit mantissa `00`; `k = 1` per `value = (k << n) + m`).
        // The `BLOCK_FN_QUIT` function code is numerically `4`
        // (`spec/04` §2 + the in-tree `FunctionCode::Quit` dispatch at
        // `crate::block::read_function_code`), so the encoded QUIT
        // command at a fresh byte boundary occupies bits 7..4 of the
        // first byte, leaving bits 3..0 to be zero-filled per
        // `spec/05` §4. Final byte: `0100 0000 = 0x40`.
        //
        // `spec/04` §2 now pins this 4-bit `0100` encoding directly,
        // consistent with `spec/02` §2.1's worked-example list (`0100`
        // = 4, `00100` = 8). An earlier revision of `spec/04` §2 had
        // described QUIT as a "5-bit pattern `00100`", which `spec/02`
        // §2.1 decodes to value 8 (= `BLOCK_FN_ZERO`), not 4; the
        // decoder — which dispatches numeric `4` to Quit and `8` to
        // Zero — was the source of truth, and the spec was corrected
        // to match.
        let mut w = BitWriter::new();
        w.write_uvar(4, 2);
        assert_eq!(w.bits_written(), 4);
        assert_eq!(w.pad_to_byte(), 4);
        assert_eq!(w.into_bytes(), vec![0x40]);
    }

    // ---- snapshot_bytes ----

    #[test]
    fn snapshot_bytes_pads_pending_without_consuming_state() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        let snap = w.snapshot_bytes();
        assert_eq!(snap, vec![0b1010_0000]);
        // Writer's state is unchanged: a subsequent write resumes at
        // the same bit offset (the next 5 bits land in the active
        // byte's bits 4..0).
        w.write_bits(0b11111, 5);
        assert_eq!(w.into_bytes(), vec![0b1011_1111]);
    }

    // ---- natural_ulong_width ----

    #[test]
    fn natural_ulong_width_edge_cases() {
        assert_eq!(natural_ulong_width(0), 0);
        assert_eq!(natural_ulong_width(1), 1);
        assert_eq!(natural_ulong_width(2), 2);
        assert_eq!(natural_ulong_width(3), 2);
        assert_eq!(natural_ulong_width(4), 3);
        assert_eq!(natural_ulong_width(255), 8);
        assert_eq!(natural_ulong_width(256), 9);
        assert_eq!(natural_ulong_width(u32::MAX), 32);
    }
}
