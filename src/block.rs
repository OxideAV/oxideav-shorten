//! Shorten per-block command stream — function-code dispatch.
//!
//! After the file-header parse of [`crate::header`], the rest of the
//! Shorten bit stream is a sequence of **per-block commands**. Each
//! command starts with a function-code field read as
//! `uvar(FNSIZE = 2)` per
//! `docs/audio/shorten/spec/02-variable-length-coding.md` §4.1. The
//! ten numeric values 0..=9 enumerate the per-block operations
//! (`docs/audio/shorten/spec/03-block-and-predictor.md` §3 +
//! `spec/04`).
//!
//! Round 2 lands:
//!
//! * The [`FunctionCode`] enum naming every code 0..=9.
//! * The [`read_function_code`] helper that reads + classifies one
//!   command-header.
//! * Full payload decode for the two simplest commands:
//!   - [`BLOCK_FN_VERBATIM`][FunctionCode::Verbatim] — `length`
//!     (`uvar(VERBATIM_CHUNK_SIZE = 5)`) + `length × uvar(8)` opaque
//!     bytes (`spec/03` §3.10 + `spec/02` §4.5).
//!   - [`BLOCK_FN_QUIT`][FunctionCode::Quit] — bare sentinel
//!     (`spec/03` §3.8 / `spec/04` §2).
//!
//! The predictor commands (`DIFF0..3`, `QLPC`) and the housekeeping
//! commands that mutate per-stream state (`BLOCKSIZE`, `BITSHIFT`,
//! `ZERO`) parse their function-code field correctly but return
//! [`Error::BlockCommandNotImplemented`] for the payload — those land
//! in later rounds along with the per-channel sample-history carry,
//! the Rice residual decode, and the running mean estimator.
//!
//! The clean-room provenance for the function-code numeric assignment
//! is the verification table in `spec/04` (which pins codes 4..=8
//! against fixtures `F2..F9` plus structural anchors in `F1`) and the
//! header-decode of `spec/02` §4.1 + §4.5 (which pins code 9 against
//! fixture `F1`'s 44-byte verbatim prefix and codes 0..3 against the
//! polynomial-difference predictor stream that follows).

use crate::bitreader::BitReader;
use crate::error::{Error, Result};

/// Width of the function-code field on the wire. Pinned by
/// `spec/02` §4.1 ("FNSIZE = 2").
pub const FNSIZE: u32 = 2;

/// Width of the verbatim-prefix length field on the wire. Pinned by
/// `spec/02` §4.5.
pub const VERBATIM_CHUNK_SIZE: u32 = 5;

/// Width of each verbatim-prefix byte on the wire. Pinned by
/// `spec/02` §4.5.
pub const VERBATIM_BYTE_SIZE: u32 = 8;

/// Implementation-side safety cap on the byte-count carried by a
/// single `BLOCK_FN_VERBATIM` command. The encoder uses verbatim
/// chunks to preserve a host-format file header (typically a 44-byte
/// RIFF/WAVE preamble); a length above this cap is either stream
/// corruption or pathological encoder behaviour. The cap is generous
/// (64 KiB) so that any plausible host-format header fits.
pub const VERBATIM_MAX_LEN: u32 = 64 * 1024;

/// The ten per-block function codes enumerated by `spec/03` §3 /
/// `spec/04`. Codes outside 0..=9 are not emitted by any v2/v3
/// encoder per `spec/04` §1; the parser surfaces them as
/// [`Error::UnknownFunctionCode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FunctionCode {
    /// `0` — order-0 polynomial-difference predictor (`spec/03` §3.1).
    Diff0,
    /// `1` — order-1 polynomial-difference predictor (`spec/03` §3.2).
    Diff1,
    /// `2` — order-2 polynomial-difference predictor (`spec/03` §3.3).
    Diff2,
    /// `3` — order-3 polynomial-difference predictor (`spec/03` §3.4).
    Diff3,
    /// `4` — end-of-stream sentinel (`spec/03` §3.8 / `spec/04` §2).
    Quit,
    /// `5` — sub-block-size override (`spec/03` §3.6 / `spec/04` §4).
    Blocksize,
    /// `6` — per-stream bit-shift adjustment (`spec/03` §3.7 /
    /// `spec/04` §3).
    Bitshift,
    /// `7` — quantised LPC predictor (`spec/03` §3.5 / `spec/04` §5).
    Qlpc,
    /// `8` — constant-zero block for the current channel (`spec/03`
    /// §3.9 / `spec/04` §6).
    Zero,
    /// `9` — inline verbatim byte payload (`spec/03` §3.10 / `spec/02`
    /// §4.1).
    Verbatim,
}

impl FunctionCode {
    /// Decode a wire-level function-code numeric value (0..=9) into
    /// the named enum variant.
    pub fn from_wire(code: u32) -> Result<Self> {
        Ok(match code {
            0 => FunctionCode::Diff0,
            1 => FunctionCode::Diff1,
            2 => FunctionCode::Diff2,
            3 => FunctionCode::Diff3,
            4 => FunctionCode::Quit,
            5 => FunctionCode::Blocksize,
            6 => FunctionCode::Bitshift,
            7 => FunctionCode::Qlpc,
            8 => FunctionCode::Zero,
            9 => FunctionCode::Verbatim,
            other => return Err(Error::UnknownFunctionCode(other)),
        })
    }

    /// Wire-level numeric value of this function code.
    pub fn wire_value(self) -> u32 {
        match self {
            FunctionCode::Diff0 => 0,
            FunctionCode::Diff1 => 1,
            FunctionCode::Diff2 => 2,
            FunctionCode::Diff3 => 3,
            FunctionCode::Quit => 4,
            FunctionCode::Blocksize => 5,
            FunctionCode::Bitshift => 6,
            FunctionCode::Qlpc => 7,
            FunctionCode::Zero => 8,
            FunctionCode::Verbatim => 9,
        }
    }

    /// Whether the command advances the implicit channel cursor of
    /// `spec/03` §2 (i.e., whether it produces a block of samples for
    /// the current channel). Predictor commands and `BLOCK_FN_ZERO`
    /// do; housekeeping commands (`BLOCKSIZE`, `BITSHIFT`, `VERBATIM`)
    /// and the terminal `QUIT` do not.
    pub fn advances_channel_cursor(self) -> bool {
        matches!(
            self,
            FunctionCode::Diff0
                | FunctionCode::Diff1
                | FunctionCode::Diff2
                | FunctionCode::Diff3
                | FunctionCode::Qlpc
                | FunctionCode::Zero
        )
    }
}

/// Read one function-code field from `reader` and resolve it to the
/// named [`FunctionCode`]. Does **not** consume the payload that
/// follows; the caller dispatches on the returned variant.
pub fn read_function_code(reader: &mut BitReader<'_>) -> Result<FunctionCode> {
    let raw = reader.read_uvar(FNSIZE)?;
    FunctionCode::from_wire(raw)
}

/// The fully-decoded payload of a `BLOCK_FN_VERBATIM` command
/// (`spec/03` §3.10). The codec stores the bytes opaquely; the host
/// application interprets them (typically as a WAV / RIFF / AIFF
/// preamble — see `spec/02` §4.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerbatimChunk {
    /// The opaque byte payload, in the order it appeared in the
    /// encoder's input. `bytes.len() == length` is invariant; the
    /// public field carries the bytes directly for ergonomic
    /// consumption.
    pub bytes: Vec<u8>,
}

/// Read the payload of a `BLOCK_FN_VERBATIM` command after its
/// function-code field has already been consumed. The layout
/// (`spec/03` §3.10 + `spec/02` §4.5) is:
///
/// ```text
/// length: uvar(VERBATIM_CHUNK_SIZE = 5)
/// byte:   uvar(VERBATIM_BYTE_SIZE  = 8)   × length
/// ```
///
/// Rejects:
///
/// * `length > VERBATIM_MAX_LEN` ([`Error::OverflowingUvar`] — the
///   implementation-side safety cap of 64 KiB).
/// * A `uvar(8)` byte field that decodes to a value `>= 256`
///   ([`Error::OverflowingUvar`]).
/// * Bit-stream truncation mid-payload ([`Error::Truncated`]).
pub fn read_verbatim_payload(reader: &mut BitReader<'_>) -> Result<VerbatimChunk> {
    let length = reader.read_uvar(VERBATIM_CHUNK_SIZE)?;
    if length > VERBATIM_MAX_LEN {
        return Err(Error::OverflowingUvar);
    }
    let mut bytes = Vec::with_capacity(length as usize);
    for _ in 0..length {
        let b = reader.read_uvar(VERBATIM_BYTE_SIZE)?;
        if b > u8::MAX as u32 {
            return Err(Error::OverflowingUvar);
        }
        bytes.push(b as u8);
    }
    Ok(VerbatimChunk { bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MSB-first bit packer used to build synthetic block-stream
    /// fixtures.
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

    /// Encode a `uvar(n)` value into a bit sequence. The encoder
    /// chooses the smallest legal prefix (zero leading zeros when
    /// `value < 2^n`; one or more leading zeros otherwise).
    fn encode_uvar(value: u32, n: u32) -> Vec<u32> {
        if n == 0 {
            // Pure unary; value zeros + a terminator.
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

    #[test]
    fn function_code_round_trips_wire_values_0_to_9() {
        for code in 0u32..=9 {
            let fc = FunctionCode::from_wire(code).expect("0..=9 must decode");
            assert_eq!(fc.wire_value(), code);
        }
    }

    #[test]
    fn function_code_rejects_out_of_range_wire_value() {
        for code in [10u32, 11, 15, 16, 100, u32::MAX] {
            assert_eq!(
                FunctionCode::from_wire(code),
                Err(Error::UnknownFunctionCode(code))
            );
        }
    }

    #[test]
    fn cursor_advancement_matches_spec03_per_command_clauses() {
        for (code, advances) in [
            (FunctionCode::Diff0, true),
            (FunctionCode::Diff1, true),
            (FunctionCode::Diff2, true),
            (FunctionCode::Diff3, true),
            (FunctionCode::Quit, false),
            (FunctionCode::Blocksize, false),
            (FunctionCode::Bitshift, false),
            (FunctionCode::Qlpc, true),
            (FunctionCode::Zero, true),
            (FunctionCode::Verbatim, false),
        ] {
            assert_eq!(
                code.advances_channel_cursor(),
                advances,
                "cursor advancement for {code:?}"
            );
        }
    }

    #[test]
    fn read_function_code_classifies_each_wire_value() {
        // Encode each `uvar(FNSIZE=2)` code value and verify
        // `read_function_code` resolves it.
        for code in 0u32..=9 {
            let bits = encode_uvar(code, FNSIZE);
            let buf = pack_bits_msb_first(&bits);
            let mut r = BitReader::new(&buf);
            let fc = read_function_code(&mut r).expect("0..=9 must classify");
            assert_eq!(fc.wire_value(), code);
        }
    }

    #[test]
    fn read_function_code_rejects_invalid_wire_value() {
        // uvar(2) = 10 has prefix "00" then terminator "1" then
        // mantissa "10" = 6 bits total: 0010_10..
        let bits = encode_uvar(10, FNSIZE);
        let buf = pack_bits_msb_first(&bits);
        let mut r = BitReader::new(&buf);
        assert_eq!(
            read_function_code(&mut r),
            Err(Error::UnknownFunctionCode(10))
        );
    }

    #[test]
    fn read_verbatim_payload_recovers_simple_chunk() {
        // length = 3, bytes = [0x12, 0x34, 0x56].
        let mut bits = Vec::new();
        bits.extend(encode_uvar(3, VERBATIM_CHUNK_SIZE));
        for b in [0x12u8, 0x34, 0x56] {
            bits.extend(encode_uvar(b as u32, VERBATIM_BYTE_SIZE));
        }
        let buf = pack_bits_msb_first(&bits);
        let mut r = BitReader::new(&buf);
        let chunk = read_verbatim_payload(&mut r).unwrap();
        assert_eq!(chunk.bytes, vec![0x12, 0x34, 0x56]);
    }

    #[test]
    fn read_verbatim_payload_handles_zero_length() {
        // length = 0: just the length field, no byte reads.
        let bits = encode_uvar(0, VERBATIM_CHUNK_SIZE);
        let buf = pack_bits_msb_first(&bits);
        let mut r = BitReader::new(&buf);
        let chunk = read_verbatim_payload(&mut r).unwrap();
        assert!(chunk.bytes.is_empty());
    }

    #[test]
    fn read_verbatim_payload_truncation_returns_truncated() {
        // length = 3 but only one byte's worth of bits supplied.
        let mut bits = Vec::new();
        bits.extend(encode_uvar(3, VERBATIM_CHUNK_SIZE));
        bits.extend(encode_uvar(0x12, VERBATIM_BYTE_SIZE));
        let buf = pack_bits_msb_first(&bits);
        let mut r = BitReader::new(&buf);
        assert_eq!(read_verbatim_payload(&mut r), Err(Error::Truncated));
    }

    /// Behavioural anchor — `spec/02` §4.1 / §4.5 + `spec/02` §6.7.
    /// Starting from the post-header bit position on fixture `F1`
    /// (bit 43 relative to byte `0x05`, which is byte `0x0A` bit 5
    /// of the file), the first per-block command decodes as
    /// `uvar(2) = 9` (VERBATIM) + `uvar(5) = 44` + 44 × `uvar(8)`
    /// yielding 44 bytes matching `F1`'s input WAV file's leading
    /// 44 bytes byte-for-byte. The recovered bytes embed
    /// `0x0000AC44 = 44100` (sample rate) and `0x0002` (channel
    /// count), matching `ffprobe`'s report on the same fixture
    /// (test `T6`).
    ///
    /// We reconstruct the synthetic-byte input via the same encoder
    /// the spec describes (the same `uvar` packing tested above)
    /// and verify the round-trip end-to-end. The literal F1 fixture
    /// bytes are not committed to the test (the fixture is a
    /// 7,187,365-byte file from `samples.mplayerhq.hu` — too large
    /// to embed) but the structural test below is the strongest
    /// in-tree anchor we can build without that fixture.
    #[test]
    fn fixture_f1_t6_synthetic_verbatim_chunk() {
        // The 44 verbatim bytes spec/02 §4.1 / §4.5 footnote
        // `T6` quotes for fixture F1. We embed them here as the
        // spec's own observation table; the test then encodes them
        // via our `uvar(8)` packer and round-trips them through
        // `read_verbatim_payload`.
        let f1_verbatim: [u8; 44] = [
            0x52, 0x49, 0x46, 0x46, 0xa4, 0xca, 0xa2, 0x00, 0x57, 0x41, 0x56, 0x45, 0x66, 0x6d,
            0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x44, 0xac, 0x00, 0x00,
            0x10, 0xb1, 0x02, 0x00, 0x04, 0x00, 0x10, 0x00, 0x64, 0x61, 0x74, 0x61, 0x80, 0xca,
            0xa2, 0x00,
        ];

        // Embedded fields the spec's T6 footnote cites as
        // independent confirmation against `ffprobe`'s report:
        //   sample rate = 0x0000_AC44 = 44100 (bytes 24..28 LE)
        //   channel count = 0x0002    = 2     (bytes 22..24 LE)
        let sample_rate = u32::from_le_bytes([
            f1_verbatim[24],
            f1_verbatim[25],
            f1_verbatim[26],
            f1_verbatim[27],
        ]);
        let channels = u16::from_le_bytes([f1_verbatim[22], f1_verbatim[23]]);
        assert_eq!(sample_rate, 44100);
        assert_eq!(channels, 2);

        // Build the full command (function code 9 then payload)
        // and roundtrip it.
        let mut bits = Vec::new();
        bits.extend(encode_uvar(9, FNSIZE)); // VERBATIM
        bits.extend(encode_uvar(f1_verbatim.len() as u32, VERBATIM_CHUNK_SIZE));
        for &b in f1_verbatim.iter() {
            bits.extend(encode_uvar(b as u32, VERBATIM_BYTE_SIZE));
        }
        let buf = pack_bits_msb_first(&bits);

        let mut r = BitReader::new(&buf);
        let fc = read_function_code(&mut r).expect("first code must classify");
        assert_eq!(fc, FunctionCode::Verbatim);
        let chunk = read_verbatim_payload(&mut r).expect("payload must decode");
        assert_eq!(chunk.bytes.len(), 44);
        assert_eq!(chunk.bytes.as_slice(), f1_verbatim.as_slice());
    }

    #[test]
    fn quit_function_code_carries_no_payload() {
        // spec/03 §3.8 / spec/04 §2: `BLOCK_FN_QUIT` is the bare
        // 5-bit pattern `00100`. The decoder reads the function code
        // and is done.
        let bits = encode_uvar(4, FNSIZE);
        let buf = pack_bits_msb_first(&bits);
        let mut r = BitReader::new(&buf);
        assert_eq!(read_function_code(&mut r).unwrap(), FunctionCode::Quit);
    }
}
