# oxideav-shorten

A pure-Rust Shorten (`.shn`) audio codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Orphan-rebuild scaffold (2026-05-18).** The prior implementation was
retired under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav/blob/master/docs/IMPLEMENTOR_ROUND.md):
a top-level source comment in the prior tree declared the implementation
derived from an external library codebase, and that admission could not
be reconciled with the clean-room provenance requirement. Master history
was fully erased per the Hat-3 cold-enforcement procedure.

The implementation will be re-built against the published Shorten
specification (Tony Robinson, 1994 — "Shorten: Simple lossless and
near-lossless waveform compression", CUED/F-INFENG/TR.156) and the
in-tree clean-room behavioural references in `docs/audio/shorten/`.

## License

MIT — see [LICENSE](./LICENSE).
