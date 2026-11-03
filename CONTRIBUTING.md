# Contributing to poise

Thank you for contributing to `poise`, part of The Complete TideOS Eval Kit.

## Standards

1. Pure Rust: No C-FFI or unpinned unsafe code (`#![forbid(unsafe_code)]`).
2. Determinism: All sampling must use explicit seeds and deterministic PRNGs (`SplitMix64`).
3. Gate check: Run formatting, clippy with denied warnings, and all test targets before opening PRs:
   ```bash
   cargo fmt --all -- --check
   cargo clippy --all-targets -- -D warnings
   cargo test --all-targets
   ```
4. Typography: Enforce zero em-dashes and zero en-dashes across all docs, code, and test packs.
