#![no_main]

//! Parse arbitrary fuzz-supplied bytes through the file-header parser
//! and pin the writer/parser equivalence.
//!
//! The header is the first thing an attacker controls: a byte-aligned
//! `ajkg` magic + version byte (`spec/01` §1) followed by six
//! `ulong()` fields (`spec/01` §3 / `spec/02` §3) whose prefix-zero
//! runs are unbounded on the wire. The contract under test:
//!
//! 1. [`oxideav_shorten::parse_stream_header`] always *returns* —
//!    `Err(Error::…)` for a malformed prefix, `Ok(ParsedHeader)`
//!    otherwise; no panic, no overflow, no unbounded loop on a long
//!    prefix-zero run (`Error::OverflowingUvar` caps it).
//! 2. Every accepted header survives the typed accessors
//!    (`filetype_pinned`, `check_decode_resource_bounds`,
//!    `sample_history_carry_len`) without panicking, and the carry
//!    length honours the `max(3, H_maxlpcorder)` rule of `spec/03`
//!    §3.11 / `spec/05` §1.
//! 3. **Writer/parser equivalence**: re-serialising an accepted header
//!    through [`oxideav_shorten::write_stream_header`] and parsing the
//!    result yields the identical `ShortenStreamHeader` and the same
//!    `bits_consumed_after_v` count as the writer reports.

use libfuzzer_sys::fuzz_target;
use oxideav_shorten::{parse_stream_header, write_stream_header, Error, CARRY_LEN_FLOOR};

fuzz_target!(|data: &[u8]| {
    let parsed = match parse_stream_header(data) {
        Ok(p) => p,
        Err(Error::InvalidMagic) => {
            assert!(
                data.len() < 5 || data[..4] != *b"ajkg",
                "InvalidMagic reported on a buffer that carries the magic"
            );
            return;
        }
        Err(_) => return,
    };
    let h = parsed.header;

    // Typed accessors never panic.
    let _ = h.filetype_pinned();
    let _ = h.check_decode_resource_bounds();
    let carry = h.sample_history_carry_len();
    assert!(carry as usize >= CARRY_LEN_FLOOR);
    assert!(carry >= h.maxlpcorder);
    assert!(matches!(h.version, 1..=3));

    // Writer/parser equivalence.
    let mut out = Vec::new();
    let bits = write_stream_header(&mut out, &h).expect("accepted header re-serialises");
    let again = parse_stream_header(&out).expect("re-serialised header parses");
    assert_eq!(again.header, h, "header round-trips through write_stream_header");
    assert_eq!(
        again.bits_consumed_after_v, bits,
        "writer-reported bit count matches the parser's"
    );
});
