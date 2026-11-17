# Deploy on gfx1151host and b70host

This doc tells how to build the nagaya, the process of `docs/spec.md`, and install its systemd user unit on gfx1151host.
It tells how to serve council v2 on port 8090 from the nagaya. The rules are in
`docs/spec.md` 2.1, 2.4, 3.3 and 10.1-10.4. The section "b70host" at the end does the same for the Gemma 4 engine
on b70host.

`nagaya.service` serves council v2 on port 8090 on gfx1151. Since 2026-11-01, `qwen-image-fusal` is resident and the
Qwen3.8 engine is on demand ("The layout of gfx1151host").

Run each command as `atobey` on gfx1151host. The unit is a user unit, so every `systemctl` command has `--user`. A command
without `--user` talks to the system manager and says "inactive" for a user unit.

## The files

| File | What it is |
|---|---|
| `deploy/nagaya.service` | The systemd user unit. `Type=notify` (spec 10.1). |
| `deploy/gfx1151host/nagaya.toml` | The bootstrap config (spec 2.1). The unit sets `NAGAYA_CONFIG` to it. |
| `deploy/gfx1151host/profile.toml` | The placement profile (spec 2.4). A copy goes in the store directory. |
| `deploy/wait-address.sh` | The wait for the tailnet address. Both units run it before the process starts. |

The test `crates/nagaya/tests/deploy.rs` reads all three with the real loaders. A change to a schema breaks it.

The bootstrap config binds HTTP to `198.51.100.20:8090`, the address of port 8090 on gfx1151host (Amy, 2026-10-18). It binds
gossip to UDP port 8091 on the same address. Its seed peer is b70host's gossip address, `198.51.100.10:8092`, so the
two machines form one cluster (milestone 23). The store directory is `/home/atobey/.local/share/nagaya/store`.

The unit sets `OTEL_EXPORTER_OTLP_ENDPOINT` to `http://127.0.0.1:4317`, the host collector. The nagaya exports OTLP
over gRPC (spec 9.1). Without a collector, the log still works.

The unit waits for the tailnet address before it starts the process. A user unit cannot order itself after the
system unit `tailscaled.service`, so at boot the address can be missing. A bind to a missing address exits 3, and
the unit stays down (spec 10.4). On 2026-10-24 b70host booted after a power loss, and this kept its nagaya
down. `deploy/wait-address.sh` checks `ip -o addr show` once a second for 120 s. Then it fails with the address in
the message, and the unit restarts after 30 s.

## Before you start

The gaps of 2026-10-20 night are closed. Check these items before each start, because a start fails when one is
wrong.

- Build the `nagaya` binary with `--features hip` (see "Build"). The feature registers the kinds
  `mk-qwen38-gfx1151` and `qwenimage21`. A build without it exits 3 with the message "this build has no engine".
- The binary holds the Qwen3.8 engine. The build compiles it from `crates/nagaya-mk-qwen38-gfx1151/hip/` and needs ROCm
  at `/opt/rocm`. No separate engine library is needed (AGENTS.md milestone 15, Amy 2026-10-22).
- The profile must hold the real size of the Qwen3.8 engine's weight arena. The file in `deploy/gfx1151host/` holds the
  measured size. A new build of the engine or new weights can change it. Read "The weights size" below.
- The context pool is 12 contexts of 4,096 tokens and 4 of 32,768. The binding fixes it, and the profile names no
  context count. Read "The memory budgets" below when the Qwen3.8 engine or the pool changes.
- Nothing else may hold the GPU (see "Rules for every restart").

## Build

Build the release binary in the repository of the nagaya:

```
cd ~/src/nagaya
nice -n 10 cargo build --release -p nagaya --features hip
ls -l target/release/nagaya
```

The unit runs this file. Build it again before each restart, or the unit runs the old code. The feature `hip` adds
the Qwen3.8 engine. Without it, the binary has no engine and exits 3. The build compiles the Qwen3.8 engine's HIP code
with ROCm's `clang++` and links it statically, so the build takes about one minute more. It does not open the GPU.

The release profile needs no setting beyond the default (`opt-level = 3`). The `[profile.dev]` table is not part of
the release build.

At start, the binary reads the inventory of the machine from files under `/` (spec 2.1). It opens no GPU runtime for
this. A probe error exits 3 with the message "cannot read the inventory of this machine".

## Install

Link the unit, copy the profile, and ask systemd to read the unit. Run these steps once.

```
ln -s ~/src/nagaya/deploy/nagaya.service ~/.config/systemd/user/nagaya.service
mkdir -p ~/.local/share/nagaya/store
cp -n ~/src/nagaya/deploy/gfx1151host/profile.toml ~/.local/share/nagaya/store/profile.toml
systemctl --user daemon-reload
systemd-analyze --user verify nagaya.service
```

Copy the profile. Do not link it. The nagaya writes `profile.toml` again when a client swaps the profile (spec 2.3).
The `-n` flag keeps a profile that is already there. After a swap, that file is the active profile, and the file in
`deploy/gfx1151host/` is only the first one.

Without a `profile.toml`, the process starts with an empty profile and sends `READY=1` (spec 10.1). It then serves
no engine. Check the log line "ready with 1 engines" after each start.

## Rules for every restart

- Build the release binary again before each restart, with `--features hip`. A restart runs `target/release/nagaya`
  as it is.
- Nothing else may hold the GPU while the nagaya runs (spec 3.1). A tool that needs the GPU stops the unit first:
  `systemctl --user stop nagaya.service`. It starts the unit again when it ends.
- Never start a second process that uses the GPU. A second process on the GPU can stop the SMU (spec 3.1).
- A new profile that loads or reloads an engine makes the process exit with code 4 and restart (spec 3.3, "Which
  actions run in place", 10.4). systemd counts that start. The unit allows 3 starts in 600 s, so a fourth profile
  swap in 600 s fails and the unit stays down.
- Exit code 3 is a refusal to start: a bad config, a profile that does not fit, a canary that differs from its record.
  The unit does not restart. Read the log, fix the cause, and run `systemctl --user reset-failed nagaya.service`.
  Then start it.

## The canary of the Qwen3.8 engine

The Qwen3.8 engine runs a canary at each load of its weights (spec 8.1). It runs fixed token ids through the engine and
hashes the logits. The open fails when the hash differs from the record in the table of `src/canary.rs` of the engine
crate. At start the process exits with code 3, and the unit does not restart.

The canary adds time to a load. Measured on gfx1151host on 2026-11-12, in 5 processes each: about 1.9 s on the fast path and
about 4.2 s on the exact path. A load takes about 28 s. The log has one line at each load: the canary hash, its
milliseconds and the facts of the build.

When the canary refuses an open, for example after an update of ROCm or a change of the kernels:

1. Read the message. It gives the key, the record, the measured hash and the 19 row hashes.
2. Decide if the change of bits is meant. If it is not, the build or the driver changed the result. Do not serve it.
3. If it is meant, replace the value in the table in a commit that says that the bits moved. The identity digest then
   changes, and each held context of the engine fills again once.
4. A checkpoint or a setting with no record gives an error that names the key. Measure it with the second form of the
   open: `cargo test -p nagaya-mk-qwen38-gfx1151 --features hip --test hip_canary -- --nocapture`, with `QWEN38_SHARDS`
   set and `nagaya.service` stopped. Add the line that it prints to the table, in a commit that records the bits of
   that checkpoint.

The card tests of a deploy open the engine in the checked form, so a deploy finds a mismatch before the service does.
A mismatch at a swap after the start stops the process, as each failed open of a swap does.

## The kaijutsu gate

The live gate council of kaijutsu reads `~/.config/kaijutsu/config/kernel/gate.toml`. Its `[council]` table has
`server = "http://198.51.100.20:8090"`. It talks to whatever serves port 8090. That is the
nagaya.

A threshold is fitted for one identity (gogi `docs/council-api.md`). Since 2026-10-21, the file's threshold
names its spec alone, and no identity fields. Its comment says: "a megakernel redeploy no longer turns decisions into
misses."

A new engine identity thus does not remove the threshold. Do these steps after each deploy:

1. Read the identity from `GET /council/v2/identity`.
2. If a `[[council.threshold]]` entry names `weight_hash`, `engine`, `tokenizer_hash` or `template`, compare them.
3. If one differs, the gate has no threshold for that identity. Ask Amy before you change the file.

On 2026-10-21, a throwaway kaijutsu kernel drove a generation and gate council decisions through the nagaya.

## The memory budgets

The profile has one pool for `device(0)` and `host` (spec 4.1): 112 GiB, or 120,259,084,288 bytes. The host has
about 125.1 GiB. The host safety floor is 12 GiB (spec 4.7). That leaves 113.1 GiB, and the pool is a whole count
below it. The store takes 2 GiB of host memory from the pool, and 50 GiB of NVMe ("Two engines on gfx1151host"). Until
milestone 22, the host budget was 16 GiB, the Qwen3.8 engine's default for its parked contexts.

The Qwen3.8 engine declares its terms in `limits().memory`. The terms are these:

- the weights: 65,490,227,200 bytes (`weights_bytes` in the profile);
- the context pool of 12 contexts of 4,096 tokens and 4 of 32,768: 12,198,215,680 bytes;
- the pass scratch: 324,009,984 bytes.

The sum is 78,012,452,864 bytes. With the Qwen3.8 engine alone and the old store budget of 17,179,869,184 bytes, it was
95,192,322,048 bytes. The pool has 120,259,084,288 bytes, so about 25.1 GB stayed free. A new context and a staging
copy (spec 4.2) take their bytes from that room. A copy of a 32,768-token context is about 1.93 GB.

If a change makes the sum larger than the pool, the process refuses the profile and exits 3. The log gives the
shortfall in bytes. Then raise the pool in `profile.toml`, or lower `store.host`. Keep the pool 12 GiB below
`MemTotal`.

**Two engines on gfx1151host.** Reason: Amy chose to keep Qwen3.8 and Qwen-Image loaded while gfx1151host only serves, and the
measures below leave no room for the old store budget.

- The pool stays 112 GiB, and the floor of spec 4.7 stays 12 GiB.
- The store's host budget goes from 16 GiB to 2 GiB. The NVMe budget stays 50 GiB, so a parked context still has a
  place. A host victim moves to the NVMe tier (spec 6.9, Decided 2026-11-01), so the Qwen3.8 engine's snapshots stay in
  the store. A conditioning of Qwen-Image (spec 6.16) takes at most 7.9 MiB, and the prompt of the oracle 176 KiB.
- Qwen-Image declares 29,873,193,256 bytes of weights and 7 GiB of scratch. `examples/measure.rs` measured the
  scratch on 2026-11-01. A job of 1440 by 1440 peaked at 6.41 GiB over the loaded engine, with the picks of the
  GEMMs. With the picks in the plan table, it peaked at 3.63 GiB. The open loads 28.11 GiB in 9.5 s.
- The Qwen3.8 engine declares 78,012,452,864 bytes. Qwen-Image declares 37,389,386,024, and the store 2,147,483,648. The
  sum is 117,549,322,536 bytes. The pool has 120,259,084,288, so 2.71 GB stay free. A staging copy of a context of
  32,768 tokens is about 1.93 GB, so it still fits.
- A store budget of 4 GiB would leave 0.56 GB, too little for that copy. A pool above 113.1 GiB breaks the floor.
- Before a large build on gfx1151host, unload `qwen-image-fusal` with `POST /nagaya/v1/engines/qwen-image-fusal/unload`
  (spec 3.3b), or stop the nagaya.

**The layout of gfx1151host.** Qwen-Image is the resident engine, and the Qwen3.8
engine is on demand. Amy: "current goal for nagaya is to have loaded what it's instructed to have loaded. so if we load
both on gfx1151host, it's intentional. if we can't do it we can't do it, and only one stays". The API manages the swaps.

- The start loads `qwen-image-fusal` only. A client swaps to Qwen3.8 with `POST /nagaya/v1/engines/qwen38/load` and the
  body `{"unload": ["qwen-image-fusal"]}`, and back in the same way.
- The Qwen3.8 engine's open needs the bytes of its shards and its floor in MemAvailable: 93.68 GB and 12.88 GB. With
  Qwen-Image loaded, gfx1151host has less, so a load of the Qwen3.8 engine beside it fails. Load the Qwen3.8 engine first to
  have both.
- With both loaded, MemAvailable is about 16 GiB. The engine stops a pass below 12 GiB. KFD stops its
  queues when the kernel moves the pages of its arena. On 2026-11-01 a `cargo test` of
  another session stopped the nagaya three times this way. So do no build on gfx1151host while both are loaded.

## The weights size

The engine config in `profile.toml` has one key, `weights_bytes`: the size of the Qwen3.8 engine's weight arena (spec
2.4, `docs/engine-abi.md` "v0 binding choices"). The plan reads no weights, so the profile must give the size. The
ledger holds these bytes for the weights term.

The file in `deploy/gfx1151host/` holds the measured size: 65,490,227,200 bytes, or 60.99 GiB. The first start on gfx1151
gave it on 2026-10-20. After a new build of the engine or new weights, a start can exit 3 with another size. Do these
steps then:

1. Start the unit. The Qwen3.8 engine loads its weights, which takes about 35 s.
2. The process finds another arena size and exits 3. The log says: "the weight arena of the Qwen3.8 engine is N bytes,
   and the profile plans M bytes; set `weights_bytes = N` in the config of the engine".
3. Write N in both profiles: `deploy/gfx1151host/profile.toml` and `~/.local/share/nagaya/store/profile.toml`.
4. Run `systemctl --user reset-failed nagaya.service`, then start the unit again.

A change to the HIP code of the engine can change the arena size, and the same exit 3 follows. Do the same steps
then.

## Settings left unset

The unit leaves one setting of the Qwen3.8 engine unset. Set it with a drop-in file:
`systemctl --user edit nagaya.service`.

- `QWEN38_FAST_QMM=0` selects the exact path. Unset means the fast path, an approximation. The identity carries the
  setting (`docs/engine-abi.md`). A change of it changes the identity.

## Arenas through a restart

On gfx1151host, the unit does not keep arenas through a restart yet (spec 10.2). Each restart loads the weights again.
Later, the unit sets `FileDescriptorStoreMax=512` and `FileDescriptorStorePreserve=restart`, and the process
stores its fds. The unit file has a comment at that place.

## Decided by senpai 2026-10-20 night, accepted by Amy 2026-11-02

These rules fill gaps in the spec. Each one is the simplest choice that works.

- The engine kind in the profile is `mk-qwen38-gfx1151`, the name of the engine crate. The engine is named `qwen38`.
- `TimeoutStopSec` is 60 s. A running quantum takes at most 10 s (`max_quantum_ms`), and the park takes 10 s.
- The unit makes the store directory with `ExecStartPre`. The process reads its profile from there before it opens
  the store.
- The gossip port is 8091, next to the HTTP port 8090.
- The pool is 112 GiB, the store's host budget is 2 GiB and its NVMe budget is 50 GiB (see "The memory budgets").
- `interval_copies` is 0 (spec 6.13), so the scheduler makes no interval copy. A larger value turns the copies on.
  A change of it restarts the process (spec 2.3).
- `QWEN38_FAST_QMM` stays unset. The library pin was unset too. Milestone 15 removed it, because the binary links its
  engine (Amy, 2026-10-22).
- The profile is copied into the store directory once. It is not a link.
- The release binary is built with `--features hip`. The plain build has no engine and exits 3.
- `weights_bytes` in the engine config is the size of the Qwen3.8 engine's weight arena. The file holds the measured
  size (see "The weights size"). A wrong size exits 3.
- A probe error at start exits 3 (spec 2.1, 10.4).

## b70host

The nagaya replaces mkviolet on b70host (Amy, 2026-10-21: "our goal is to replace it"). It serves Gemma 4 on the
two B70 cards behind council v2 on `198.51.100.10:8091`, mkviolet's address. It also serves speech, image and embedding
engines on one card each (milestones 17 to 20). Run each command as `atobey` on b70host.
Every `systemctl` command has `--user`.

### The files

| File | What it is |
|---|---|
| `deploy/b70host/nagaya.service` | The systemd user unit (spec 10.1, "One unit file for each machine"). |
| `deploy/b70host/nagaya.toml` | The bootstrap config (spec 2.1). The unit sets `NAGAYA_CONFIG` to it. |
| `deploy/b70host/profile.toml` | The placement profile (spec 2.4). A copy goes in the store directory. |
| `deploy/wait-address.sh` | The wait for `198.51.100.10`, as on gfx1151host ("The files"). |

The test `crates/nagaya/tests/deploy.rs` reads all three with the real loaders. Its admission test plans from the
checkpoint, so run it with `GEMMA4_MODEL_DIR` set.

- The config binds HTTP to `198.51.100.10:8091` and gossip to UDP port 8092 on the same address. Its
  seed peer is gfx1151host's gossip address, `198.51.100.20:8091` (milestone 23).
- The config ignores the Radeon 610M at `0000:80:00.0`. The iGPU drives the display, and it gets no work (spec 2.1).
- The store directory is `/home/atobey/.local/share/nagaya/store`, on the NVMe disk.
- The profile puts the engine `gemma` of kind `mk-gemma4-bmg` on devices 0 and 1, the cards at `0000:03:00.0`
  and `0000:08:00.0`. It is on demand since milestone 24, and it loads only through the load routes (spec 3.3b).
- The profile puts `lfm3-omni` and `lfm3` of kind `mk-lfm3-bmg` on device 0. Both are resident (spec 11.11). They
  plan 24,742,512,648 bytes of card 0 together, with eight slots each. Eight slots let a batched decode pass hold
  eight contexts (milestone 27); with two, they planned 14,255,941,640 bytes.
- The entry of `lfm3` names its J-Lens in `[engines.lab]`: `~/.local/share/nagaya/lens/lfm3.jlens.safetensors`, in
  the profile as an absolute path (`docs/lab.md`, "The rules of the J-Lens"). `tools/lfm3_jlens_fit.py` fitted it on
  the CPU of b70host over 100 prompts, in 16.7 hours. The lab reads it at its first use, outside the ledger.
- The slots are mkviolet's: one of 4,096 positions and one of 65,536. A block holds 512 tokens.
- The profile puts the speech engine `voice` of kind `qwen3tts-bmg` on device 1, on demand (spec 3.3a, 11.8). It serves
  the council model `Qwen3-TTS-12Hz-1.7B-VoiceDesign` (spec 11.8a). Until M18 step 3 it also served
  `POST /v1/audio/speech`.
- The profile puts six image engines of kind `mk-sdxl-bmg` on one card each, on demand (spec 11.9). `sdxl-0`,
  `pony-0` and `lightning-0` are on device 0. `sdxl-1`, `autismmix-1` and `inpaint-0` are on device 1. `inpaint-0`
  has the 9-channel UNet (spec 11.9b).
- The profile puts `embeddinggemma-2` of kind `mk-embeddinggemma2-bmg` on device 1. It is resident and pinned (spec
  11.10).
- The unit sets `QWEN3TTS_MODEL_DIR` to the Qwen3-TTS checkpoint in the Hugging Face cache.
- The unit sets `GEMMA4_MODEL_DIR` to `~/models/gemma-4-26B-A4B`. The engine and the family read
  the checkpoint there.
- The unit sets no Level Zero variable. `ZE_AFFINITY_MASK` would hide a card, and the engine finds its cards by PCI
  address.
- The unit keeps the 31 Gemma 4 slabs through a restart (spec 10.2). `systemctl --user restart` reaches `READY=1`
  in about 1.2 s. `systemctl --user stop` clears the store, so the next start loads cold in about 7.1 s.
- The OTLP line of the unit is a comment. No collector listened on b70host on 2026-10-21. Remove the `#` when one
  runs on port 4317.

On 2026-10-21, `gemma` planned 29,964,945,932 bytes of terms on device 0 and 31,889,794,060 bytes on device 1.
Each card has 32 GiB (34,359,738,368 bytes), so card 1 had about 2.3 GB left. A larger slot does not fit. The host
budget is 96 GiB, and the store takes 16 GiB of it and 50 GiB of NVMe.

The speech engine plans 5,645,096,644 bytes on card 1, and card 1 has about 2.3 GB beside `gemma`. Thus it does not
load beside `gemma` (engine-abi "How a group loads both"). A client loads it with the load route and names `gemma` in
`unload`, and later it loads `gemma` again (spec 3.3b). Gemma 4 does not load beside the LFM3 models either, so its
load names `lfm3-omni` and `lfm3` in `unload`. A load of an image engine on card 0 names them too. No request starts
a load, and no engine swaps back by itself (spec 3.3, "Explicit loads", and 3.3a, "Explicit layout").

### Change the profile

The process reads the copy in the store directory, so a new file in `deploy/b70host/` changes nothing alone.

1. Compare the copy with the file in the repository. A profile swap of a client writes the copy (spec 2.3), so a
   difference that you did not make is a swap. Stop and ask Amy about it.

   ```
   diff ~/.local/share/nagaya/store/profile.toml <(git -C ~/src/nagaya show HEAD~1:deploy/b70host/profile.toml)
   ```

2. Stop the unit, copy the new file, and start the unit. The stop clears the stored arenas, so `gemma` loads cold.

   ```
   systemctl --user stop nagaya.service
   cp ~/src/nagaya/deploy/b70host/profile.toml ~/.local/share/nagaya/store/profile.toml
   systemctl --user start nagaya.service
   ```

### Build and install

Build the release binary with the features `xe` and `video`. The feature `xe` registers the kinds `mk-gemma4-bmg`,
`qwen3tts-bmg`, `mk-sdxl-bmg`, `mk-embeddinggemma2-bmg` and `mk-lfm3-bmg`, and `build.rs` compiles the kernels with
`ocloc`. A build without it has no engine and exits 3. The feature `video` lets `lfm3-omni` read video parts with
ffmpeg's libraries (spec 11.11), so the machine needs the package `ffmpeg` and libclang.

```
cd ~/src/nagaya
nice -n 10 cargo build --release -p nagaya --features xe,video
ls -l target/release/nagaya
```

Link the unit, copy the profile, and ask systemd to read the unit. Run these steps once.

```
ln -s ~/src/nagaya/deploy/b70host/nagaya.service ~/.config/systemd/user/nagaya.service
mkdir -p ~/.local/share/nagaya/store
cp -n ~/src/nagaya/deploy/b70host/profile.toml ~/.local/share/nagaya/store/profile.toml
systemctl --user daemon-reload
systemd-analyze --user verify nagaya.service
```

Copy the profile. Do not link it, because a profile swap writes the file again (spec 2.3).

### Cut over from mkviolet

Do these steps in this order. Stop at the first step that fails, and do the rollback below.

1. Check that mkviolet does not run, and that nothing else holds the cards (spec 3.1).

   ```
   systemctl --user is-active mkviolet.service nagaya.service
   fuser -v /dev/dri/renderD* 2>&1
   ```

   Both units must say `inactive`. Amy stopped and disabled `mkviolet.service` on 2026-10-21. The second command
   lists the processes that hold a card. The display server holds the iGPU, and that is correct. Stop any job of
   ours that holds a B70 card. Do not stop a process that you do not know.

2. Enable and start the nagaya.

   ```
   systemctl --user enable --now nagaya.service
   ```

   The command returns when the process sends `READY=1`. The cutover of 2026-10-23 was ready after 7 s.

3. Check the state of the unit and its log.

   ```
   systemctl --user status nagaya.service
   journalctl --user -u nagaya.service -b --no-pager | tail -40
   ```

   The log has the lines "inventory: device 2 `gfx1036` at 0000:80:00.0 ignored" and "ready with 1 engines". With
   the profile of 2026-11-02, the second line is "ready with 2 engines": `gemma` and `embeddinggemma-2`.

4. Ask for the identity, and run gogi's conformance suite.

   ```
   curl -s http://198.51.100.10:8091/council/v2/identity
   cd ~/src/gogi
   nice -n 10 cargo run --release -p gogi-conformance -- http://198.51.100.10:8091
   ```

   The cutover of 2026-10-23 gave `266 ok, 0 FAIL, 11 skip`, with gogi `a07d85e`. The skips include `ride`,
   which the nagaya does not declare, and the restart check.

   The restart check runs with `--allow-restart 'systemctl --user restart nagaya.service'`. On 2026-10-23, from
   `47f2191`, it gave `273 ok, 0 FAIL, 10 skip`. The stop parked 1 pinned entry of 38,309,888 bytes (spec 6.11). The
   restart attached the 31 stored arena fds. It sent `READY=1` 1.3 s after the old process exited.

   The HTTP server listens before the engines open, and it answers 503 `unavailable` until they are ready (spec
   10.1). The process sends `READY=1` after the routes attach, so the suite can run when `restart` returns.

5. Look for card errors in the kernel log.

   ```
   journalctl -k -b --no-pager | grep -i -E 'xe .*(hang|reset|timeout|wedge)|drm.*(hang|reset)' | tail -20
   ```

   Any hang or reset after the start is a failed gate. Do the rollback.

6. Check the kaijutsu gate as in "The kaijutsu gate", with the address `http://198.51.100.10:8091`.

### Roll back to mkviolet

Use this when a gate fails, or when Amy asks for mkviolet. mkviolet is an archive since milestone 15s
(`docs/ontology.md`), and the rollback needs its unit file and its source. Its unit runs its release binary, so
build it first.

```
systemctl --user disable --now nagaya.service
cd ~/src/mkviolet
nice -n 10 cargo build --release
systemctl --user link ~/src/mkviolet/systemd/mkviolet.service
systemctl --user enable --now mkviolet.service
systemctl --user is-active mkviolet.service
curl -s http://198.51.100.10:8091/council/v2/identity
```

The `disable` removes the nagaya's link. The `link` puts mkviolet's unit back in systemd's search path. The last line
answers when mkviolet has loaded its weights. Nothing in the store directory needs a change for the rollback.

### Rules for every restart on b70host

- Build the release binary again before each restart, with `--features xe`.
- `mkviolet.service` and the nagaya's unit use the same address. Never start both.
- Do not run work on the iGPU. It drives the display.
- The unit waits up to 120 s for `198.51.100.10` before the process starts. A unit that waits at boot is
  waiting for tailscaled. Check it with `tailscale ip`.
- Exit code 3 is a refusal to start. Read the log, fix the cause, and run
  `systemctl --user reset-failed nagaya.service`. Then start it.
- The unit keeps the 31 Gemma 4 slabs through a restart (spec 10.2): it sets `FileDescriptorStoreMax=512` and
  `FileDescriptorStorePreserve=restart`. `systemctl --user restart` attaches them in about 1.2 s. `systemctl --user
  stop` clears the store, so the next start loads cold.

