//! Shorten codec registration.

use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Result};
use oxideav_core::{CodecInfo, CodecRegistry, Decoder};

pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("shorten_sw")
        .with_lossless(true)
        .with_intra_only(true)
        .with_max_channels(8);
    reg.register(
        CodecInfo::new(CodecId::new(super::CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder),
    );
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    super::decoder::make_decoder(params)
}
