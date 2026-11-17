# The scrub for a public repository

This file is the work list of milestone 31 (`AGENTS.md`). It holds the inventory of step 1, the table of names, and
the decisions that steer steps 2 to 8. It names the old identifiers, because step 2 removes them. Remove this file
before the new root commit of step 8.

The counts are from `git grep` on `main` at `22f5804`, on 2026-11-17, without `docs/history.md`. A line can hold
more than one class. The script is `git grep -c -P <pattern>`, summed over files.

## Decisions

The owner decided these on 2026-11-17, by interview.

- **The `MK_` macros and the kernel entry stay.** `MK_LFM3`, `MK_SDXL`, `MK_QWEN3TTS`, `MK_EMBEDDINGGEMMA2` and
  `mk_persistent_interpreter` are internal names of a megakernel. Here `mk` means "megakernel", as in a crate name.
  The interface of the exokernel shows none of them. Thus no kernel binary changes, and no engine identity changes.
- **The reference code comes into the repository.** Each oracle tool reads its reference Python from `tools/`, with
  the notice of its source. No tool reads a tree outside the repository. The Gemma 4 checkpoint moves out of the
  old practice tree, into the Hugging Face cache, as the other models are.
- **The example machines get generic names.** The profiles in `deploy/` take names by their hardware, not by the
  host. Tests use made-up host names. An address in an example is a placeholder.
- **The model names stay.** The steering docs describe the mechanism, control vectors, and not one use of it.
- **A rule names no author.** A note is "Proposed" or "Decided <date>". It names no person, no agent and no model.
- **The web app and the swarm driver go public**, with their hosts from flags or config.
- **After the post, the work stays as it is now:** small commits to `main`, and kaibo rounds at the end of a
  milestone.
- **`docs/history.md` stays private.** The public tree starts with the spec as its record. No public file cites
  `history.md`, a milestone of it, or a commit of this archive.
- **The docs become fewer.** The spec is the main document, because readers will build their own engines from it.
  A supporting doc folds into the spec, or it comes along as it is. A later pass decides each one.

## The inventory

| Class | Files | Lines | Where | Step |
|---|---|---|---|---|
| `mkviolet` | 54 | 337 | docs 199, the Gemma 4 crates 75, the Qwen3.8 text crate 15, tools 10, deploy 9 | 2, 3 |
| `gmk` | 7 | 13 | docs 7, `tools/lfm3_jlens_fit.py` 3, `AGENTS.md`, `README.md` | 2, 3 |
| `mk_api` | 5 | 12 | the spec 8, engine-abi 1, `nagaya-text-gemma4` 3 | 2, 3 |
| the old Qwen3.8 repository path | 10 | 73 | the build fixtures of the Qwen3.8 crate 60, tools 9 | 2, 5 |
| `mkengine` | 2 | 13 | the build fixture of the Qwen3.8 crate 12 | 2, 5 |
| `megakernel.service`, `mkviolet.service` | 9 | 17 | deploy, `tools/op_mutate.py`, a Qwen3.8 test, docs | 2 |
| `diffusepal` | 4 | 15 | the spec 9, the Qwen-Image tools 5 | 2, 3 |
| `lfm2d`, `/mk/v1` | 2 | 2 | `README.md`, `AGENTS.md` | 3 |
| the owner by name | 41 | 559 | docs 472, deploy 7, a comment in 8 crates | 3 |
| the agent by name | 11 | 146 | docs 139 | 3 |
| a Claude model by name | 31 | 204 | docs 179, comments 19, tools 5 | 2, 3 |
| `gfx1151host` | 41 | 212 | docs 76, `crates/nagaya` 56, tools 30, deploy 11 | 4 |
| `b70host` | 45 | 202 | docs 93, `crates/nagaya` 50, tools 26, deploy 13 | 4 |
| `sm121host` | 14 | 32 | `crates/nagaya` 13 (a fake peer in tests), docs 15 | 4 |
| a tailnet address | 15 | 47 | docs 14, deploy 12, `crates/nagaya` 11, tools 8 | 4 |
| a home path | 10 | 79 | the Qwen3.8 build fixtures 60, deploy 9, `Cargo.toml` and `Cargo.lock` 8 | 4, 5 |
| a path under the home `src` | 19 | 55 | docs 19, deploy 12, the Qwen3.8 text crate 10, `crates/nagaya` 6 | 4, 5 |
| a commit hash in a comment | 23 | 27 | the comments of the crates and tools | 3 |
| a commit hash in a doc | 7 | 38 | docs | 3 |
| a citation of "`AGENTS.md` milestone N" | 11 | 12 | the comments of the crates | 3 |
| a citation of `history.md` | 3 | 8 | docs | 3 |
| the steering docs by use | 10 | 64 | docs, tools | 3 |
| an SPDX line | 50 | 50 | the Qwen3.8 crate 39, tools 11 | 6 |

Other details:

- gogi is a git dependency by a `file://` path (`Cargo.toml:35-38`). gogi is public on GitHub.
- No LICENSE file is present.
- `README.md` is stale. It says that milestone 23 is next.
- The HTTP routes have no authentication. The README must say that the nagaya trusts its network.
- The Gemma 4 golden tools run a copy of the reference Python outside the repository. That copy and
  `tools/gemma4_reference.py` differ from line 24.
- The Qwen3.8 oracle tool (`tools/oracle_fixture.py`) imports its reference from the old Qwen3.8 repository. That
  repository is on the Strix Halo machine only.

## The table of names

Step 2 does each row. A row changes no bits unless its last column says so.

| Old identifier | Where | New name or action | Cost |
|---|---|---|---|
| `MK_LFM3`, `MK_SDXL`, `MK_QWEN3TTS`, `MK_EMBEDDINGGEMMA2` | the kernels of `nagaya-mk-bmg` | keep: `mk` is "megakernel" | none |
| `mk_persistent_interpreter` | `megakernel.cl`, `KERNEL_ENTRY` | keep: `mk` is "megakernel" | none |
| `// mkviolet: per-card task interpreter` | `megakernel.cl:1` | "The task interpreter of a B70 card" | none: no binary holds it |
| `MKVSNAP1`, `KvSnapshot::write_to`, `read_from` | `nagaya-mk-gemma4-bmg/src/gemma4.rs` | remove: only a test calls them | none |
| `a_fit_follows_the_join_of_mkviolet` | `nagaya-mk-gemma4-bmg/src/spec.rs` | `a_fit_follows_the_join_of_the_held_rows` | none |
| `a_plain_read_has_the_layout_of_mkviolet` | `nagaya-text-gemma4/tests/council.rs` | `a_plain_read_has_the_council_layout` | none |
| `a_plain_read_has_the_tokens_of_the_layout_of_mkviolet` | `nagaya-text-qwen38/tests/council.rs` | `a_plain_read_has_the_tokens_of_the_council_layout` | none |
| `the_b70host_config_binds_mkviolets_address_and_ignores_the_igpu` | `crates/nagaya/tests/deploy.rs` | `the_b70_config_binds_its_address_and_ignores_the_igpu` | none |
| `--mkviolet`, `DEFAULT_MKVIOLET` | `tools/gemma4_oracle_fixture.py` | remove: the tool reads `tools/gemma4_reference.py` | none |
| `QWEN38_REFERENCE_REPO` and its default | `tools/oracle_fixture.py` | the reference in `tools/`; the variable goes | none |
| the default of `GEMMA4_MODEL_DIR` | the unit, `deploy.rs`, two tools | no default, as each other model folder | none |
| `mkv-letters-1` | comments, the spec | remove the citation | none |
| `mk_api.rs`, `council_server.rs`, `golden.rs` of mkviolet | comments | "the first Gemma 4 engine", or no citation | none |
| `tools/mk_dequant_digests.cpp` | comments, two dequant fixtures | "the digest tool" and its method, with no commit | none |
| the old Qwen3.8 path, `mkengine` | `tests/fixtures/build/*.json` | paths relative to the crate; write the fixtures again | none |
| `megakernel.service` | `deploy/nagaya.service`, `tools/op_mutate.py`, a test | remove: the unit is gone | none |
| `mkviolet.service` and its address story | `deploy/b70host/` | remove | none |
| `diffusepal` | the Qwen-Image tools, the spec | "the diffusers reference" | none |
| `gmk` | `tools/lfm3_jlens_fit.py`, docs | "the first J-Lens fit, of Gemma 4 12B" | none |
| "How mkviolet maps", "mkviolet's limits", "mkviolet's parts" | engine-abi headings, and the comments that cite them | "How the Gemma 4 engine maps", and so on | none |
| `deploy/gfx1151host/`, `deploy/b70host/` | deploy, tests, docs | names by hardware, for example `deploy/strix-halo/` and `deploy/b70-pair/` | a profile copy at the next deploy |
| `sm121host` | the cluster tests | a made-up peer, for example `node-c` | none |

## Open

- **The checkout path is in four kernel binaries.** The binaries of Qwen3-TTS, SDXL, EmbeddingGemma 2 and LFM3
  hold the `-I` path of `build.rs`. Thus the identity of each engine depends on the folder of the checkout. A clone in
  another folder builds the same bits under other identities. Step 5 gives `ocloc` a path relative to its working
  folder. That changes four identities once, so it waits for a deploy window. The test
  `the_other_binaries_keep_their_bytes` fails in a worktree for this cause.
- **The names of the example profiles.** The two names in the table are a proposal.
- **The Qwen3.8 reference.** Its source is the old Qwen3.8 repository on the Strix Halo machine. The session there
  copies the parts that `tools/oracle_fixture.py` imports.
