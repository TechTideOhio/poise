# Agent Guidelines for poise

Follow the canonical workspace instructions in `AGENTS.md` at workspace root.

## Specific Crate Directives
- Pure Rust, zero unsafe code (`#![forbid(unsafe_code)]`).
- All subcommands support `--json` and output `HarnessEnvelope<T>`.
- Strict typography: zero em-dashes and zero en-dashes across all files.
- Validate changes via `node Tools/agent-harness/q.mjs -- cargo test --manifest-path Tools/poise/Cargo.toml`.
