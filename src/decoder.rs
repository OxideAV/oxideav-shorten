//! Shorten decoder — header parse + per-block sample reconstruction.
//!
//! Wired into the [`oxideav_core::Decoder`] trait. The current
//! implementation treats every packet as a self-contained `.shn` byte
//! stream (parses magic + header + the rest), suitable for the common
//! "decode the whole file as one packet" use case. Round-2 work will
//! split this into per-frame packets driven by a `.shn` demuxer.

use oxideav_core::bits::BitReader;
use oxideav_core::Decoder;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, Result};

use crate::rice::{read_signed_k, read_ulong, read_ulong_v0, read_unsigned_k};

// ────────────────────────── Constants from §5 of the trace doc ──────────────────────────

const MAGIC: [u8; 4] = *b"ajkg";

const FNSIZE: u32 = 2; // function code k_param_size
const ULONGSIZE: u32 = 2; // header field k_param_size
const TYPESIZE_HINT: u32 = 4;
const CHANSIZE_HINT: u32 = 0;
const BLOCKSIZE_HINT: u32 = 8; // log2(256)
const LPCQSIZE: u32 = 2;
const NMEAN_HINT: u32 = 0;
const NSKIPSIZE: u32 = 1;
const ENERGYSIZE: u32 = 3;
const BITSHIFTSIZE: u32 = 2;
const LPCQUANT: u32 = 5; // QLPC coefficients are signed Rice-k=5; qshift = 5
const VERBATIM_CKSIZE_SIZE: u32 = 5;
const VERBATIM_BYTE_SIZE: u32 = 8;

const V2LPCQOFFSET: i32 = 1 << LPCQUANT; // 32 for v >= 2

const FN_DIFF0: u32 = 0;
const FN_DIFF1: u32 = 1;
const FN_DIFF2: u32 = 2;
const FN_DIFF3: u32 = 3;
const FN_QUIT: u32 = 4;
const FN_BLOCKSIZE: u32 = 5;
const FN_BITSHIFT: u32 = 6;
const FN_QLPC: u32 = 7;
const FN_ZERO: u32 = 8;
const FN_VERBATIM: u32 = 9;

const MAX_CHANNELS: usize = 8;
const CANONICAL_HEADER_SIZE: u32 = 44;
const OUT_BUFFER_SIZE: u32 = 16_384;
const MAX_BLOCKSIZE: u32 = 65_535;

// ───────────────────────────── public factory ─────────────────────────────

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(ShortenDecoder {
        codec_id: params.codec_id.clone(),
        sample_rate: params.sample_rate,
        channels_hint: params.channels.map(|c| c as u32),
        pending: None,
        eof: false,
    }))
}

// ───────────────────────────── decoder state ─────────────────────────────

struct ShortenDecoder {
    codec_id: CodecId,
    sample_rate: Option<u32>,
    channels_hint: Option<u32>,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for ShortenDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "shorten: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        decode_full_stream(&pkt.data, pkt.pts, self.sample_rate, self.channels_hint)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

// ───────────────────────────── header ─────────────────────────────

#[derive(Debug, Clone)]
pub struct ShortenHeader {
    pub version: u8,
    pub internal_ftype: u32,
    pub channels: u32,
    pub blocksize: u32,
    pub maxnlpc: u32,
    pub nmean: u32,
    pub _skip_bytes: u32,
}

impl ShortenHeader {
    /// Parse the leading magic + version + 6 ulong header fields.
    /// Leaves the bitreader positioned just past the `skip_bytes`-byte
    /// hint section (i.e. ready to read the first FN_VERBATIM command).
    pub fn parse(br: &mut BitReader<'_>) -> Result<Self> {
        let m0 = br.read_u32(8)? as u8;
        let m1 = br.read_u32(8)? as u8;
        let m2 = br.read_u32(8)? as u8;
        let m3 = br.read_u32(8)? as u8;
        if [m0, m1, m2, m3] != MAGIC {
            return Err(Error::invalid("shorten: missing 'ajkg' magic"));
        }
        let version = br.read_u32(8)? as u8;
        if version > 2 {
            return Err(Error::unsupported(format!(
                "shorten: unknown stream version {version}"
            )));
        }
        // v0 streams use raw fixed-k reads with the size_hint as k;
        // v >= 1 uses adaptive ulong (read k, then read value).
        let read_field = |br: &mut BitReader<'_>, hint: u32| -> Result<u32> {
            if version == 0 {
                read_ulong_v0(br, hint)
            } else {
                read_ulong(br, ULONGSIZE, hint)
            }
        };
        let internal_ftype = read_field(br, TYPESIZE_HINT)?;
        let channels = read_field(br, CHANSIZE_HINT)?;
        if channels == 0 || channels as usize > MAX_CHANNELS {
            return Err(Error::invalid(format!(
                "shorten: channels={channels} out of range 1..={MAX_CHANNELS}"
            )));
        }
        let blocksize = read_field(br, BLOCKSIZE_HINT)?;
        if blocksize == 0 || blocksize > MAX_BLOCKSIZE {
            return Err(Error::invalid(format!(
                "shorten: blocksize={blocksize} out of range"
            )));
        }
        let maxnlpc = read_field(br, LPCQSIZE)?;
        if maxnlpc > 1024 {
            return Err(Error::invalid(format!(
                "shorten: maxnlpc={maxnlpc} unreasonably large"
            )));
        }
        let nmean = read_field(br, NMEAN_HINT)?;
        if nmean > 1024 {
            return Err(Error::invalid(format!(
                "shorten: nmean={nmean} unreasonably large"
            )));
        }
        let skip_bytes = read_field(br, NSKIPSIZE)?;
        // The skip_bytes "seek hint" payload — read and discard byte by
        // byte using the same 8-bit-k unsigned Rice as VERBATIM bytes.
        for _ in 0..skip_bytes {
            let _ = read_unsigned_k(br, 8)?;
        }
        Ok(Self {
            version,
            internal_ftype,
            channels,
            blocksize,
            maxnlpc,
            nmean,
            _skip_bytes: skip_bytes,
        })
    }

    /// Initial offset for the running mean ring: 0 for signed types,
    /// 0x80 for U8 (the silence midpoint of unsigned 8-bit PCM). U16
    /// types get 0x8000.
    fn initial_offset(&self) -> i32 {
        match self.internal_ftype {
            // 1 = S8 → 0
            1 => 0,
            // 2 = U8 → 0x80
            2 => 0x80,
            // 3 = S16HL, 5 = S16LH → 0
            3 | 5 => 0,
            // 4 = U16HL, 6 = U16LH → 0x8000
            4 | 6 => 0x8000,
            _ => 0,
        }
    }

    fn lpcqoffset(&self) -> i32 {
        if self.version >= 2 {
            V2LPCQOFFSET
        } else {
            0
        }
    }
}

// ───────────────────────────── full-packet decode ─────────────────────────────

fn decode_full_stream(
    data: &[u8],
    pts: Option<i64>,
    _sample_rate_hint: Option<u32>,
    channels_hint: Option<u32>,
) -> Result<Frame> {
    let mut br = BitReader::new(data);
    let hdr = ShortenHeader::parse(&mut br)?;
    if let Some(c) = channels_hint {
        if c != hdr.channels {
            return Err(Error::invalid(format!(
                "shorten: header channels={} != params channels={}",
                hdr.channels, c
            )));
        }
    }

    let nch = hdr.channels as usize;
    let nwrap = hdr.maxnlpc.max(3) as usize;
    let nmean = hdr.nmean as usize;
    let lpcqoffset = hdr.lpcqoffset();
    let init_offset = hdr.initial_offset();

    // Per-channel state.
    let mut chan = Vec::with_capacity(nch);
    for _ in 0..nch {
        chan.push(ChannelState::new(nwrap, nmean, init_offset));
    }
    // Stream-level state.
    let mut blocksize = hdr.blocksize as usize;
    let mut bitshift: u32 = 0;
    let mut cur_chan = 0usize;
    let mut quit = false;

    // Output is interleaved S16 — one i16 per (sample, channel). We
    // accumulate into a frame-major buffer, in source order: channel 0
    // then channel 1 etc. fill `cur_chan`'s row, and when all channels
    // are filled, interleave into the output.
    let mut interleaved: Vec<i16> = Vec::new();
    // Per-channel scratch for the most recently decoded block, so we
    // can interleave once `cur_chan` wraps.
    let mut block_scratch: Vec<Vec<i32>> = vec![Vec::new(); nch];

    while !quit {
        // Function codes are plain unsigned fixed-k=2 per §5 of the
        // trace doc, NOT adaptive ulong — even though §3's prose
        // calls them "2-bit-k ulong", what's actually packed in the
        // bitstream is `uvar_get(FNSIZE)` (plain Rice).
        let cmd = read_unsigned_k(&mut br, FNSIZE)?;
        match cmd {
            FN_QUIT => {
                quit = true;
            }
            FN_BLOCKSIZE => {
                let new_bs = read_ulong(&mut br, ULONGSIZE, log2_floor(blocksize as u32))?;
                if new_bs == 0 || new_bs as usize > blocksize {
                    return Err(Error::invalid(format!(
                        "shorten: BLOCKSIZE {new_bs} not in 1..={blocksize}"
                    )));
                }
                blocksize = new_bs as usize;
            }
            FN_BITSHIFT => {
                // BITSHIFT payload is plain unsigned fixed-k=2 (not
                // adaptive ulong) per §5.
                let bs = read_unsigned_k(&mut br, BITSHIFTSIZE)?;
                if bs > 32 {
                    return Err(Error::invalid(format!("shorten: BITSHIFT {bs} > 32")));
                }
                bitshift = bs;
            }
            FN_VERBATIM => {
                // VERBATIM length and per-byte payload are plain
                // unsigned fixed-k Rice (NOT adaptive ulong) — see §5
                // table.
                let len = read_unsigned_k(&mut br, VERBATIM_CKSIZE_SIZE)?;
                if !(CANONICAL_HEADER_SIZE..=OUT_BUFFER_SIZE).contains(&len) {
                    return Err(Error::invalid(format!(
                        "shorten: VERBATIM length {len} out of {CANONICAL_HEADER_SIZE}..={OUT_BUFFER_SIZE}"
                    )));
                }
                for _ in 0..len {
                    let _ = read_unsigned_k(&mut br, VERBATIM_BYTE_SIZE)?;
                }
            }
            FN_DIFF0 | FN_DIFF1 | FN_DIFF2 | FN_DIFF3 | FN_QLPC | FN_ZERO => {
                let c = &mut chan[cur_chan];
                let block = decode_audio_block(
                    &mut br,
                    cmd,
                    blocksize,
                    bitshift,
                    nmean,
                    nwrap,
                    hdr.maxnlpc,
                    hdr.version,
                    lpcqoffset,
                    c,
                )?;
                block_scratch[cur_chan] = block;
                cur_chan += 1;
                if cur_chan == nch {
                    // Emit one frame's worth of interleaved samples.
                    interleave_into_s16(
                        &block_scratch,
                        blocksize,
                        hdr.internal_ftype,
                        &mut interleaved,
                    );
                    cur_chan = 0;
                }
            }
            other => {
                return Err(Error::invalid(format!(
                    "shorten: unknown function code {other}"
                )));
            }
        }
    }

    // Build the audio frame.
    let mut bytes = Vec::with_capacity(interleaved.len() * 2);
    for s in &interleaved {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    let frame = AudioFrame {
        samples: (interleaved.len() / nch) as u32,
        pts,
        data: vec![bytes],
    };
    Ok(Frame::Audio(frame))
}

// ───────────────────────────── per-channel state ─────────────────────────────

struct ChannelState {
    /// Predictor history ring (last `nwrap` decoded samples). The
    /// history is stored at the end of the buffer; the current block's
    /// decoded samples are appended after.
    history: Vec<i32>,
    /// FIFO of past block means, capacity `nmean`.
    offsets: Vec<i32>,
    nwrap: usize,
    nmean: usize,
}

impl ChannelState {
    fn new(nwrap: usize, nmean: usize, init_offset: i32) -> Self {
        Self {
            history: vec![0; nwrap],
            offsets: vec![init_offset; nmean.max(1)],
            nwrap,
            nmean,
        }
    }

    /// Mean offset to bias the current block by, computed from the
    /// past `nmean` block means. For v >= 2, includes the half-`nmean`
    /// rounding bias; the right-shift by `bitshift` to compensate for
    /// the FN_BITSHIFT scaling is applied by the caller.
    fn coffset(&self, version: u8) -> i32 {
        if self.nmean == 0 {
            // Use the persistent slot 0 (initialised to init_offset).
            return self.offsets[0];
        }
        let mut sum: i64 = 0;
        for &o in &self.offsets {
            sum += o as i64;
        }
        if version >= 2 {
            sum += (self.nmean as i64) / 2;
        }
        (sum / (self.nmean as i64)) as i32
    }

    /// Push a new mean into the FIFO. The `mean_value` is the raw
    /// (sum_of_block + blocksize/2) / blocksize for v >= 2, or just
    /// sum/blocksize for v < 2; if `bitshift > 0` and v >= 2, the
    /// stored mean is left-shifted by `bitshift`.
    fn push_mean(&mut self, mean_value: i32) {
        if self.nmean == 0 {
            // Persistent slot 0 — keep the initial value untouched.
            // (The encoder side never reads back the FIFO when nmean=0.)
            self.offsets[0] = mean_value;
            return;
        }
        // Shift FIFO left, push new at the end.
        for i in 0..(self.offsets.len() - 1) {
            self.offsets[i] = self.offsets[i + 1];
        }
        let last = self.offsets.len() - 1;
        self.offsets[last] = mean_value;
    }
}

// ───────────────────────────── audio block decode ─────────────────────────────

#[allow(clippy::too_many_arguments)]
fn decode_audio_block(
    br: &mut BitReader<'_>,
    cmd: u32,
    blocksize: usize,
    bitshift: u32,
    nmean: usize,
    nwrap: usize,
    maxnlpc: u32,
    version: u8,
    lpcqoffset: i32,
    state: &mut ChannelState,
) -> Result<Vec<i32>> {
    debug_assert_eq!(nwrap, state.nwrap);
    let _ = (nmean, maxnlpc);

    // ZERO is the silence shortcut: emit blocksize zeros, mean update
    // sees a zero block, history wraps zeros.
    if cmd == FN_ZERO {
        let block = vec![0i32; blocksize];
        finish_block(&block, state, blocksize, bitshift, version);
        // Apply bitshift fixup to the *output* (not to history).
        return Ok(apply_bitshift(&block, bitshift));
    }

    // Energy / Rice-k parameter for the residuals. Plain unsigned
    // fixed-k=3 (NOT adaptive ulong) per §5. For v0 streams the value
    // is decremented by 1 after the read (a historical wart).
    let mut k = read_unsigned_k(br, ENERGYSIZE)?;
    if version == 0 && k > 0 {
        k -= 1;
    }
    if k > 31 {
        return Err(Error::invalid(format!("shorten: residual k={k} > 31")));
    }

    // Compute coffset (mean offset for this block) and apply bitshift
    // compensation (the encoder stored shifted means, so we right-
    // shift here to undo it for the prediction step).
    let mut coffset = state.coffset(version);
    if bitshift > 0 && bitshift < 32 {
        coffset >>= bitshift;
    }

    // Predictor configuration.
    let (qshift, init_sum_base, coeffs): (u32, i32, Vec<i32>) = match cmd {
        FN_DIFF0 => (0, coffset, vec![]),
        FN_DIFF1 => (0, 0, vec![1]),
        FN_DIFF2 => (0, 0, vec![2, -1]),
        FN_DIFF3 => (0, 0, vec![3, -3, 1]),
        FN_QLPC => {
            // QLPC pred_order is plain unsigned fixed-k=2 per §5.
            let m = read_unsigned_k(br, LPCQSIZE)?;
            if m as usize > nwrap {
                return Err(Error::invalid(format!(
                    "shorten: QLPC pred_order {m} > nwrap {nwrap}"
                )));
            }
            let mut cs = Vec::with_capacity(m as usize);
            for _ in 0..m {
                cs.push(read_signed_k(br, LPCQUANT)?);
            }
            (LPCQUANT, lpcqoffset, cs)
        }
        _ => unreachable!("non-audio command in decode_audio_block"),
    };

    // For QLPC, subtract coffset from the history before the loop
    // (and re-add after) so the LPC predictor sees a roughly zero-mean
    // signal. For DIFFn predictors the coefficient sum is exactly
    // zero (Δ^n applied to a constant), so the constant offset just
    // passes through; we still keep it around to add it back to the
    // emitted samples.
    let history_save: Vec<i32> = state.history.clone();
    if cmd == FN_QLPC {
        for s in state.history.iter_mut() {
            *s -= coffset;
        }
    }

    // Decode the block. We work in a buffer of length `nwrap +
    // blocksize` so past-history indexing is uniform.
    let mut buf: Vec<i32> = Vec::with_capacity(nwrap + blocksize);
    buf.extend_from_slice(&state.history);
    for i in 0..blocksize {
        let mut sum: i64 = init_sum_base as i64;
        // sum += coeffs[j] * decoded[i - j - 1]
        for (j, &c) in coeffs.iter().enumerate() {
            let idx = nwrap + i - j - 1;
            sum += (c as i64) * (buf[idx] as i64);
        }
        let r = read_signed_k(br, k)?;
        let predicted = (sum >> qshift) as i32;
        buf.push(r.wrapping_add(predicted));
    }

    let mut decoded: Vec<i32> = buf.split_off(nwrap);

    if cmd == FN_QLPC {
        // Re-add coffset to the freshly-decoded samples and restore
        // the pre-block history (encoder subtracted only conceptually).
        for s in decoded.iter_mut() {
            *s = s.wrapping_add(coffset);
        }
        // History got mutated above; restore it before push_back so
        // mean update + bitshift fixup see the right pre-state.
        state.history = history_save;
    }

    finish_block(&decoded, state, blocksize, bitshift, version);
    Ok(apply_bitshift(&decoded, bitshift))
}

/// Push the new block's mean into the FIFO and copy the last `nwrap`
/// samples into the per-channel history ring.
fn finish_block(
    decoded: &[i32],
    state_ref: &mut ChannelState,
    blocksize: usize,
    bitshift: u32,
    version: u8,
) {
    // ── Mean update ─────────────────────────────────────────────
    let mut sum: i64 = 0;
    for &s in decoded {
        sum += s as i64;
    }
    let mut mean = if blocksize > 0 {
        if version >= 2 {
            ((sum + (blocksize as i64) / 2) / blocksize as i64) as i32
        } else {
            (sum / blocksize as i64) as i32
        }
    } else {
        0
    };
    // For v >= 2 and bitshift > 0, the encoder stores the mean
    // left-shifted so the next block's coffset compensates correctly.
    if version >= 2 && bitshift > 0 && bitshift < 32 {
        mean = mean.wrapping_shl(bitshift);
    }
    // Push into the channel's FIFO (state_ref.offsets).
    state_ref.push_mean(mean);

    // ── History wrap ────────────────────────────────────────────
    let nwrap = state_ref.history.len();
    if decoded.len() >= nwrap {
        state_ref
            .history
            .copy_from_slice(&decoded[decoded.len() - nwrap..]);
    } else {
        // Shift left, append.
        let keep = nwrap - decoded.len();
        for i in 0..keep {
            state_ref.history[i] = state_ref.history[i + decoded.len()];
        }
        for (i, &s) in decoded.iter().enumerate() {
            state_ref.history[keep + i] = s;
        }
    }
}

/// Apply the BITSHIFT fixup to the output samples. This is the final
/// per-sample multiplication by `1 << bitshift`. `bitshift == 32` is
/// the "all zeros" sentinel.
fn apply_bitshift(samples: &[i32], bitshift: u32) -> Vec<i32> {
    if bitshift == 0 {
        return samples.to_vec();
    }
    if bitshift >= 32 {
        return vec![0; samples.len()];
    }
    samples.iter().map(|&s| s.wrapping_shl(bitshift)).collect()
}

// ───────────────────────────── output format ─────────────────────────────

/// Convert per-channel decoded i32 blocks into one interleaved-S16
/// run, normalising 8-bit and U16 input types into the S16 range.
fn interleave_into_s16(
    blocks: &[Vec<i32>],
    blocksize: usize,
    internal_ftype: u32,
    out: &mut Vec<i16>,
) {
    let nch = blocks.len();
    out.reserve(blocksize * nch);
    for i in 0..blocksize {
        for ch in 0..nch {
            let raw = blocks[ch].get(i).copied().unwrap_or(0);
            let s16 = pack_s16(raw, internal_ftype);
            out.push(s16);
        }
    }
}

/// Map a decoded sample to S16 according to its `internal_ftype`.
///
/// - `1` (S8): clip to i8 range, then sign-extend to i16, then shift
///   left 8 to fill the S16 dynamic range.
/// - `2` (U8): subtract 0x80 (the U8 silence midpoint), clip to i8
///   range, then shift left 8.
/// - `3`/`5` (S16HL/S16LH): clip to i16 range. Endianness only affects
///   the *container* byte order — internally Shorten already produces
///   little-endian-equivalent integers.
/// - `4`/`6` (U16HL/U16LH): subtract 0x8000, clip to i16 range.
fn pack_s16(raw: i32, internal_ftype: u32) -> i16 {
    match internal_ftype {
        1 => clip_s8(raw).saturating_mul(256) as i16,
        2 => clip_s8(raw - 0x80).saturating_mul(256) as i16,
        4 | 6 => clip_i16(raw - 0x8000) as i16,
        _ => clip_i16(raw) as i16,
    }
}

fn clip_s8(v: i32) -> i32 {
    v.clamp(-128, 127)
}

fn clip_i16(v: i32) -> i32 {
    v.clamp(-32_768, 32_767)
}

fn log2_floor(v: u32) -> u32 {
    if v == 0 {
        0
    } else {
        31 - v.leading_zeros()
    }
}

// ───────────────────────────── tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::bits::BitWriter;
    use oxideav_core::SampleFormat;

    /// Hand-build the smallest possible Shorten v2 mono S16 stream: a
    /// header, a leading FN_VERBATIM with a 44-byte canonical header,
    /// one DIFF0 block of 4 zero residuals at k=0, then FN_QUIT.
    fn build_minimal_stream() -> Vec<u8> {
        let mut bw = BitWriter::new();
        // Magic 'ajkg' (4 raw bytes)
        for &b in b"ajkg" {
            bw.write_u32(b as u32, 8);
        }
        // Version = 2
        bw.write_u32(2, 8);
        // Helper: write an unsigned Rice-k integer.
        fn write_unsigned(bw: &mut BitWriter, value: u32, k: u32) {
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
            let u = ((value << 1) ^ (value >> 31)) as u32;
            write_unsigned(bw, u, k + 1);
        }
        // ulong = 2-bit-k unsigned then unsigned-k. Pick the smallest
        // k that gives a non-empty unary prefix budget.
        fn write_ulong(bw: &mut BitWriter, value: u32) {
            let mut k = 0u32;
            while k < 31 && (value >> k) > 8 {
                k += 1;
            }
            write_unsigned(bw, k, 2);
            write_unsigned(bw, value, k);
        }
        // 6 header fields: internal_ftype=5 (S16LH), channels=1,
        // blocksize=4, maxnlpc=0, nmean=0, skip_bytes=0.
        write_ulong(&mut bw, 5);
        write_ulong(&mut bw, 1);
        write_ulong(&mut bw, 4);
        write_ulong(&mut bw, 0);
        write_ulong(&mut bw, 0);
        write_ulong(&mut bw, 0);
        // FN_VERBATIM (cmd=9 — plain unsigned fixed-k=2), then length
        // (5-bit-k unsigned Rice) = 44, then 44 zero bytes (8-bit-k
        // unsigned Rice each).
        write_unsigned(&mut bw, FN_VERBATIM, 2);
        write_unsigned(&mut bw, 44, 5);
        for _ in 0..44 {
            write_unsigned(&mut bw, 0, 8);
        }
        // One FN_DIFF0 block: cmd=0 (k=2), energy/k=0 (3-bit-k unsigned),
        // then 4 signed-Rice residuals at k=0 — small values for sanity.
        write_unsigned(&mut bw, FN_DIFF0, 2);
        write_unsigned(&mut bw, 0, 3); // energy = 0
        for v in &[0i32, 1, -1, 2] {
            write_signed(&mut bw, *v, 0);
        }
        // FN_QUIT (k=2).
        write_unsigned(&mut bw, FN_QUIT, 2);
        bw.into_bytes()
    }

    #[test]
    fn parse_minimal_header() {
        let bytes = build_minimal_stream();
        let mut br = BitReader::new(&bytes);
        let hdr = ShortenHeader::parse(&mut br).expect("header parse");
        assert_eq!(hdr.version, 2);
        assert_eq!(hdr.internal_ftype, 5);
        assert_eq!(hdr.channels, 1);
        assert_eq!(hdr.blocksize, 4);
        assert_eq!(hdr.maxnlpc, 0);
        assert_eq!(hdr.nmean, 0);
    }

    #[test]
    fn decode_minimal_stream() {
        let bytes = build_minimal_stream();
        let mut params = CodecParameters::audio(CodecId::new(super::super::CODEC_ID_STR));
        params.sample_rate = Some(44_100);
        params.channels = Some(1);
        params.sample_format = Some(SampleFormat::S16);
        let mut dec = make_decoder(&params).expect("make_decoder");
        let pkt = Packet::new(0, oxideav_core::TimeBase::new(1, 44_100), bytes);
        dec.send_packet(&pkt).unwrap();
        let frame = dec.receive_frame().expect("decode");
        let Frame::Audio(af) = frame else {
            panic!("expected audio");
        };
        assert_eq!(af.samples, 4);
        assert_eq!(af.data.len(), 1);
        assert_eq!(af.data[0].len(), 8); // 4 samples * 2 bytes
                                         // DIFF0 with coffset = 0 (initial offset is 0 for S16): the
                                         // residuals 0, 1, -1, 2 decode to themselves.
        let s: Vec<i16> = af.data[0]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(s, vec![0, 1, -1, 2]);
    }
}
