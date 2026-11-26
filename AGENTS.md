# AGENTS.md

Coding-agent and contributor guide for **nagaya** — an exokernel for GPU megakernels, its Rust reference implementation, and the council v2 server. @tobert owns the project and makes final decisions.

## Where to look

- [`README.md`](README.md) — Architecture overview, hardware targets, and code reading order.
- [`Cargo.toml`](Cargo.toml) — All 30 workspace crates (`nagaya-<kind>-<model>-<target>` across `gfx1151`, `bmg`, `sm121`). Keep pure logic out of engine crates and `crates/nagaya`.
- [`docs/spec.md`](docs/spec.md) & [`docs/engine-abi.md`](docs/engine-abi.md) — Normative specification and `Engine` trait contract (D1–D11).
- [`docs/ontology.md`](docs/ontology.md) — Canonical terms (`Quantum`, `Ledger`, `Recipe`, `ContentId`). Use them in prose and code.
- [`docs/deploy.md`](docs/deploy.md) — Release builds, systemd user units, device release/reclaim, and rollback.
- [`docs/new-engine.md`](docs/new-engine.md) & [`docs/milestones.md`](docs/milestones.md) — Checklists for adding engines/targets and open work.

## Non-negotiable rules

- **Public repo hygiene:** Never commit old codenames (`mkviolet`, `gmk`, `diffusepal`, `/mk/v1`), real hostnames, real IPs, or home paths (`mk` stays in `nagaya-mk-*` crate names and internal kernel symbols). Use RFC 5737 IPs (`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`) and hardware placeholders (`gfx1151host`, `b70host`, `sm121host`).
- **Clarity and high information density:** In docs, comments, error messages, and commit messages. Comments explain *what* or *why*; error messages say what failed, why, and what the caller can do.
- **Spec is contract:** If code and spec disagree, stop and report. If a doc leaves a choice open, commit the simplest choice marked **Proposed** with your rationale, then proceed.
- **Bit-exact determinism & fail-fast (`spec 8.5`):** Same engine identity + inputs = identical bits. Any change that moves bits must update engine identity and note it in the commit message. Never add silent fallbacks or risk corrupting state.
- **Test-first & mutation testing:** Write the test first, watch it fail, implement, then mutate the code under test to verify the test catches it, restore, and record the tested mutation in the commit message. **Never mutate a pointer or size read by a GPU kernel.**
- **GPU safety (`spec 3.1`):** Single compute client per GPU. Check `systemctl --user is-active nagaya.service` (`--user` is required) and `nvidia-smi` (on `sm121`) before GPU tests. Run GPU tests with `--test-threads=1`, `NAGAYA_TEST_CARDS` for free cards only, and a separate `CARGO_TARGET_DIR` (the service runs `target/release/nagaya`). Never touch an unrelated GPU process or the display iGPU, and leave `nagaya.service` active.
- **Explicit reporting:** Report real counts (tests run, passed, skipped, and anything intentionally not run), verify `file:line` citations, and report sandbox/tool blockers clearly.
- **Isolation & external repos:** Use git worktrees (`~/src/wt/nagaya-<topic>`) with separate `CARGO_TARGET_DIR` for parallel lanes. Do not edit other repos except `gogi` (edit on a branch, merge to `main`, push, and bump `rev` in `Cargo.toml`).

## Checks & git

Gate commits directly on exit codes (never pipe through `tail` or `tee`):

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --tests
cargo test --workspace --tests --no-default-features
```

Also lint GPU features (`--features hip|xe|cuda`); without `ocloc` or `nvcc`, prepend `tools/ocloc-stub` or `tools/nvcc-stub` to `PATH` (stub builds are for type-checking/linting only). Host tests requiring local weights skip when their env var is unset—set `NAGAYA_REQUIRE_MODELS=1` on hosts with weights before merging changes to those crates.

- Commit small, single-idea steps; stage files by name (never `git add -A`/`.`). Never push, amend, rebase, reset, or force-push unless asked.
- Commit format: `<crate>: <what changed>` (or `nagaya: ...`), *why* and mutations run in the body, `Co-Authored-By:` / `Reviewed-by:` trailers. Run kaibo review rounds with another model family near milestone completion (`docs/reviews.md`).
