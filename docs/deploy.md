# Deploy

This doc tells how to build the nagaya and install it as a systemd user unit. The rules are in `docs/spec.md` 2.1,
2.4, 3.3 and 10.1-10.4. The directory `deploy/` holds three examples:

- `deploy/strix-halo/`: an AMD Strix Halo machine, with one gfx1151 GPU and 128 GB of unified memory. Qwen-Image 2.1
  is resident, and the Qwen3.8 engine is on demand.
- `deploy/b70-pair/`: a machine with two Intel Arc Pro B70 cards and an iGPU that drives the display. LFM3, LFM3-Omni,
  SDXL and EmbeddingGemma 2 are resident. Gemma 4 is on demand on both cards, and Inkling 2 Mini on card 0.
- `deploy/dgx-spark/`: an NVIDIA DGX Spark, with one GB10 GPU and 128 GB of unified memory. One engine,
  `inkling2-mini-gb10`, is resident: Inkling 2 Mini in 4 slots of 131,072 positions, with the `e4m3` cache.

The first two examples form one cluster: each names the gossip address of the other as its seed peer. The DGX Spark
example has no seed peer, so its first start is a cluster of one. To join, write the gossip address of a member in
`seed_peers`. The addresses are of RFC 5737, and the paths under `/srv` are examples. Change them for your machine.

## The files

Each example directory holds three files:

| File | What it is |
|---|---|
| `nagaya.service` | The systemd user unit. `Type=notify` (spec 10.1). |
| `nagaya.toml` | The bootstrap config (spec 2.1). The unit sets `NAGAYA_CONFIG` to `~/.config/nagaya/nagaya.toml`. |
| `profile.toml` | The placement profile (spec 2.4). The process reads its copy in the store directory. |

`deploy/wait-address.sh` waits for the address of `http_bind`. Each unit runs it before the process starts.

The test `crates/nagaya/tests/deploy.rs` reads the example files with the real loaders. A change to a schema breaks
it.

The unit is a user unit, so each `systemctl` command has `--user`. A command without `--user` talks to the system
manager, and it says "inactive" for a user unit. Check the unit with `systemctl --user is-active nagaya.service`.

## The address

The routes have no authentication. Bind `http_bind` and `gossip_bind` to an address that only your clients can
reach. A VPN, such as a tailnet, is one way.

- The gossip address is a UDP port on the same address as HTTP. The loader refuses `0.0.0.0`, because no peer can
  reach it.
- A user unit cannot order itself after a system unit, such as the unit of a VPN. Thus at boot the address can be
  missing.
- A bind to a missing address exits 3, and the unit stays down (spec 10.4). Thus the unit runs
  `wait-address.sh <address> 120` first.
- The script checks `ip -o addr show` once a second for 120 s. Then it fails with the address in the message, and
  the unit starts again after 30 s.

## Build

Build the release binary in the clone of the repository. The unit runs `%h/src/nagaya/target/release/nagaya`, so put
the clone at `~/src/nagaya`, or change the paths of the unit.

For the Strix Halo example:

```
cd ~/src/nagaya
nice -n 10 cargo build --release -p nagaya --features hip
```

For the B70 example:

```
cd ~/src/nagaya
nice -n 10 cargo build --release -p nagaya --features xe,video
```

For the DGX Spark example:

```
cd ~/src/nagaya
PATH=/usr/local/cuda/bin:$PATH nice -n 10 cargo build --release -p nagaya --features cuda
```

- The feature `hip` registers the kinds `mk-qwen38-gfx1151` and `qwenimage21`. The build compiles the HIP code of
  the Qwen3.8 engine with ROCm's `clang++` and links it statically. It needs ROCm at `/opt/rocm`.
- The feature `xe` registers the kinds `mk-gemma4-bmg`, `qwen3tts-bmg`, `mk-sdxl-bmg`, `mk-embeddinggemma2-bmg`,
  `mk-lfm3-bmg` and `mk-inkling2-bmg`. `build.rs` compiles the kernels with `ocloc`.
- The feature `cuda` registers the kind `mk-inkling2-sm121`. `build.rs` compiles the kernels with `nvcc` 13.2 for
  `sm_121a`, and the build needs the CUDA Driver API library `libcuda`. The run needs no `nvcc`.
- The feature `video` lets `lfm3-omni` read video parts with ffmpeg's libraries (spec 11.11). The machine needs
  ffmpeg and libclang.
- A build without an engine feature has no engine. It exits 3 with the message "this build has no engine".
- The release profile needs no setting beyond the default (`opt-level = 3`).

A restart runs the file in `target/release` as it is. Build it again before each restart, or the unit runs the old
code. Build tests into another target directory (`CARGO_TARGET_DIR`), because a test build replaces that file.

At start, the binary reads the inventory of the machine from files under `/` (spec 2.1). It opens no GPU runtime for
this. A probe error exits 3 with the message "cannot read the inventory of this machine".

## Install

Do these steps once. The example commands are for `strix-halo`; use `b70-pair` or `dgx-spark` for the others.

1. Copy the bootstrap config. Change `http_bind`, `gossip_bind`, `store_dir` and `seed_peers`. On the B70 example,
   change `ignore_devices` to the PCI address of your display GPU. The loader does not expand `~`, so `store_dir`
   is an absolute path. Put the store directory on an NVMe disk.

   ```
   mkdir -p ~/.config/nagaya
   cp ~/src/nagaya/deploy/strix-halo/nagaya.toml ~/.config/nagaya/nagaya.toml
   ```

2. Copy the profile into the store directory, and change the model paths.

   ```
   mkdir -p <store_dir>
   cp -n ~/src/nagaya/deploy/strix-halo/profile.toml <store_dir>/profile.toml
   ```

3. Copy the unit. Change the paths of the models, the store directory and the address of `wait-address.sh`. Then
   ask systemd to read it, and start it.

   ```
   cp ~/src/nagaya/deploy/strix-halo/nagaya.service ~/.config/systemd/user/nagaya.service
   systemctl --user daemon-reload
   systemd-analyze --user verify nagaya.service
   systemctl --user enable --now nagaya.service
   ```

Copy each file. Do not link it.

- The unit and the config are copies, so a pull of the repository never changes a running machine.
- The nagaya writes `profile.toml` again when a client swaps the profile (spec 2.3). After a swap, that file is the
  active profile, and the example is only the first one. The `-n` flag keeps a profile that is already there.

Without a `profile.toml`, the process starts with an empty profile and sends `READY=1` (spec 10.1). It then serves
no engine.

The Strix Halo unit sets `OTEL_EXPORTER_OTLP_ENDPOINT` to `http://127.0.0.1:4317`, for a collector on the machine. The
nagaya exports OTLP over gRPC (spec 9.1). Without a collector, the exporter logs a transport error every 5 s, so
remove the line. In the B70 unit the line is a comment. Remove the `#` when a collector runs.

## The web app

`deploy/nagaya-web.service` runs `tools/nagaya_web.py` on `127.0.0.1:8096` with `--auth tailscale-serve`.
`tailscale serve` on the same machine gives it an HTTPS address on the tailnet and sets `Tailscale-User-Login` on each
request. The app answers 401 to a request without it (the docstring of `tools/nagaya_web.py`). One machine of the
cluster is enough: the app finds the other nagayas from the cluster route.

```
printf 'NAGAYA_URLS=http://<http_bind>\n' > ~/.config/nagaya/web.env
cp ~/src/nagaya/deploy/nagaya-web.service ~/.config/systemd/user/nagaya-web.service
systemctl --user daemon-reload
systemctl --user enable --now nagaya-web.service
tailscale serve --bg --https=443 http://127.0.0.1:8096
```

- `tailscale serve` needs root or the tailscale operator: run `sudo tailscale set --operator=$USER` once. The serve
  config persists in tailscaled through a reboot, so it needs no unit.
- The page is then at `https://<machine>.<tailnet>.ts.net/`. `tailscale serve status` shows it.
- The login gates the web app only. The nagaya ports still have no authentication, so limit who reaches them with
  tailnet ACLs.
- The app logs to the journal: `journalctl --user -u nagaya-web.service`.

## Check a start

1. Read the state of the unit and its log.

   ```
   systemctl --user status nagaya.service
   journalctl --user -u nagaya.service -b --no-pager | tail -40
   ```

   The log has the line "ready with N engines", with the count of resident engines. On the B70 example, it also has
   a line for the ignored iGPU.

2. Ask for the identity, and run gogi's conformance suite.

   ```
   curl -s http://<address>:<port>/council/v2/identity
   cargo run --release -p gogi-conformance -- http://<address>:<port>
   ```

   Run the restart check of the suite with `--allow-restart 'systemctl --user restart nagaya.service'`. The HTTP
   server listens before the engines open, and it answers 503 `unavailable` until they are ready (spec 10.1). The
   process sends `READY=1` after the routes attach, so the suite can run when `restart` returns.

3. On the B70 example, look for card errors in the kernel log. Any hang or reset after the start is a failed check.

   ```
   journalctl -k -b --no-pager | grep -i -E 'xe .*(hang|reset|timeout|wedge)|drm.*(hang|reset)' | tail -20
   ```

## Rules for every restart

- Build the release binary again before each restart, with the features of the machine.
- Nothing else may hold the GPU while the nagaya runs (spec 3.1). A tool that needs the GPU stops the unit first:
  `systemctl --user stop nagaya.service`. It starts the unit again when it ends.
- Never start a second process that uses the GPU. On the Strix Halo machine, a second process on the GPU can stop the
  SMU (spec 3.1).
- On the B70 example, do not run work on the iGPU. It drives the display.
- On the DGX Spark example, the nagaya is the only compute client of the GB10 (spec 3.1). Check
  `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` before a start. Stop the unit before
  each GPU test.
- A new profile that loads or reloads an engine makes the process exit with code 4 and restart (spec 3.3, "Which
  actions run in place", 10.4). systemd counts that start.
- systemd allows 3 starts of the unit in 600 s. A fourth start in 600 s fails, and the unit stays down.
- Exit code 3 is a refusal to start: a bad config, a profile that does not fit, or a canary that differs from its
  record. The unit does not restart. Read the log, and fix the cause. Run
  `systemctl --user reset-failed nagaya.service`, then start the unit.
- A unit that waits at boot waits for its address. Check that the VPN gives the machine its address.

## The canary of the Qwen3.8 engine

The Qwen3.8 engine runs a canary at each load of its weights (spec 8.1). It runs fixed token ids through the engine
and hashes the logits. The open fails when the hash differs from the record in the table of `src/canary.rs` of the
engine crate. At start the process exits with code 3, and the unit does not restart.

The canary adds about 1.9 s to a load on the fast path and about 4.2 s on the exact path. A load takes about 28 s.
The log has one line at each load: the canary hash, its milliseconds and the facts of the build.

When the canary refuses an open, for example after an update of ROCm or a change of the kernels:

1. Read the message. It gives the key, the record, the measured hash and the 19 row hashes.
2. Decide if the change of bits is intended. If it is not, the build or the driver changed the result. Do not serve
   it.
3. If it is intended, replace the value in the table in a commit that says that the bits moved. The identity digest
   then changes, and each held context of the engine fills again once.
4. A checkpoint or a setting with no record gives an error that names the key. Measure it with the second form of the
   open: `cargo test -p nagaya-mk-qwen38-gfx1151 --features hip --test hip_canary -- --nocapture`, with
   `QWEN38_SHARDS` set and `nagaya.service` stopped. Add the line that it prints to the table, in a commit that
   records the bits of that checkpoint.

The card tests of a deploy open the engine in the checked form, so a deploy finds a mismatch before the service does.
A mismatch at a swap after the start stops the process, as each failed open of a swap does.

## The kaijutsu gate

kaijutsu is a client whose gate council reads decisions from a nagaya. Its gate config names the address of the
nagaya in its `[council]` table, for example `server = "http://<address>:8090"`. Each client that holds a threshold
does the same check.

A threshold is fitted for one identity (gogi `docs/council-api.md`). A threshold that names only its spec, and no
identity fields, stays valid after a new engine identity. Do these steps after each deploy:

1. Read the identity from `GET /council/v2/identity`.
2. If a `[[council.threshold]]` entry names `weight_hash`, `engine`, `tokenizer_hash` or `template`, compare them.
3. If one differs, the gate has no threshold for that identity. Fit the threshold again, or remove the entry.

## The memory budgets

These numbers are of the Strix Halo example. The profile has one pool for `device(0)` and `host` (spec 4.1): 112 GiB,
or 120,259,084,288 bytes. The host has about 125.1 GiB. The host safety floor is 12 GiB (spec 4.7). That leaves
113.1 GiB, and the pool is a whole count of GiB below it. The store takes 2 GiB of host memory from the pool, and
50 GiB of NVMe.

The Qwen3.8 engine declares its terms in `limits().memory`. The terms are these:

- the weights: 65,490,227,200 bytes (`weights_bytes` in the profile);
- the context pool of 12 contexts of 4,096 tokens and 4 of 32,768: 12,198,215,680 bytes;
- the pass scratch: 324,009,984 bytes.

The sum is 78,012,452,864 bytes. The binding fixes the context pool, so the profile names no context count. A new
context and a staging copy (spec 4.2) take their bytes from the free room of the pool. A copy of a 32,768-token
context is about 1.93 GB.

Qwen-Image declares 29,873,193,256 bytes of weights and 7 GiB of scratch, 37,389,386,024 bytes in all.
`examples/measure.rs` of its crate measures the scratch. A job of 1440 by 1440 peaks at 3.63 GiB over the loaded
engine, with the picks of the GEMMs in the plan table. The open loads 28.11 GiB in about 9.5 s.

The sum of both engines and the store is 117,549,322,536 bytes. The pool has 120,259,084,288, so 2.71 GB stay free.
A staging copy of a context of 32,768 tokens still fits.

- A store budget of 4 GiB leaves 0.56 GB, too little for that copy. A pool above 113.1 GiB breaks the floor.
- The NVMe budget of the store is 50 GiB, so a parked context still has a place. A host victim moves to the NVMe
  tier (spec 6.9), so the snapshots of the Qwen3.8 engine stay in the store.
- A conditioning of Qwen-Image (spec 6.16) takes at most 7.9 MiB.

If a change makes the sum larger than the pool, the process refuses the profile and exits 3. The log gives the
shortfall in bytes. Then raise the pool in `profile.toml`, or lower `store.host`. Keep the pool 12 GiB below
`MemTotal`.

## Two engines on one GPU

The Qwen3.8 engine and Qwen-Image share the unified memory of the Strix Halo machine. The profile can load both, and
the API manages the swaps. The example keeps one resident and one on demand.

- The start loads `qwen-image-fusal` only. A client swaps to the Qwen3.8 engine with
  `POST /nagaya/v1/engines/qwen38/load` and the body `{"unload": ["qwen-image-fusal"]}`. The way back is the same.
- The open of the Qwen3.8 engine needs the bytes of its shards and its floor in MemAvailable: 93.68 GB and 12.88 GB.
  With Qwen-Image loaded, the machine has less, so a load of the Qwen3.8 engine beside it fails. To have both, load
  the Qwen3.8 engine first.
- With both loaded, MemAvailable is about 16 GiB. The engine stops a pass below 12 GiB. KFD stops its queues when
  the kernel moves the pages of its arena. A large build on the machine can do this, so do no build while both are
  loaded.
- Before a large build, unload `qwen-image-fusal` with `POST /nagaya/v1/engines/qwen-image-fusal/unload`
  (spec 3.3b), or stop the nagaya.

## The weights size

The engine config of the Qwen3.8 engine in `profile.toml` has one key, `weights_bytes`: the size of its weight arena
(spec 2.4, `docs/engine-abi.md` "v0 binding choices"). The plan reads no weights, so the profile must give the size.
The ledger holds these bytes for the weights term.

The Strix Halo example holds the size of the UD-IQ4_XS shards: 65,490,227,200 bytes, or 60.99 GiB. After a new build
of the engine or new weights, a start can exit 3 with another size. Do these steps then:

1. Start the unit. The Qwen3.8 engine loads its weights, which takes about 35 s.
2. The process finds another arena size and exits 3. The log says: "the weight arena of the Qwen3.8 engine is N bytes,
   and the profile plans M bytes; set `weights_bytes = N` in the config of the engine".
3. Write N in the copy of the profile in the store directory, `<store_dir>/profile.toml`.
4. Run `systemctl --user reset-failed nagaya.service`, then start the unit again.

A change to the HIP code of the engine can change the arena size, and the same exit 3 follows. Do the same steps
then.

## Settings left unset

The unit leaves one setting of the Qwen3.8 engine unset. Set it with a drop-in file:
`systemctl --user edit nagaya.service`.

- `QWEN38_FAST_QMM=0` selects the exact path. Unset means the fast path, an approximation. The identity carries the
  setting (`docs/engine-abi.md`). A change of it changes the identity.

The B70 unit sets no Level Zero variable. `ZE_AFFINITY_MASK` would hide a card, and the engine finds its cards by PCI
address.

## Arenas through a restart

The B70 unit keeps the arenas of the engines through a restart (spec 10.2). It sets `FileDescriptorStoreMax=512` and
`FileDescriptorStorePreserve=restart`, and the process stores its fds.

- `systemctl --user restart` attaches the stored arenas, for example the 31 slabs of Gemma 4, in about 1.2 s.
- `systemctl --user stop` clears the fd store, so the next start loads cold.

The Qwen3.8 engine exports no arena, so the Strix Halo unit sets no fd store. Each restart loads the weights again.

## The B70 example

- Device 0 is card 0, and device 1 is card 1. The engines find their cards by PCI address. Device 2 is the ignored
  iGPU.
- Card 1 holds the image engine `autismmix-1`, `embeddinggemma-2` and `lfm3-omni`. Card 0 holds `lfm3`. Each is
  resident, so each start loads it. `embeddinggemma-2` is pinned too (spec 11.10).
- `lfm3` has eight slots of 32,768 positions, so a batched decode pass holds eight contexts. `lfm3-omni` has four.
- The entry of `lfm3` names its J-Lens in `[engines.lab]`, as an absolute path (`docs/lab.md`, "The rules of the
  J-Lens"). `tools/lfm3_jlens_fit.py` fits it on the CPU. The lab reads it at its first use, outside the ledger.
- The unit gives the checkpoint folders of Gemma 4, Qwen3-TTS, SDXL and EmbeddingGemma 2 in its environment. It
  also gives a folder of Inkling 2 for the family of an Inkling 2 engine on demand (spec 11.18).
- Each card has 32 GiB (34,359,738,368 bytes), and the budget of a card is all of it. The host budget is 96 GiB. The
  store takes 64 GiB of it and 300 GiB of NVMe.

`gemma` of kind `mk-gemma4-bmg` has stage 0 on device 0 and stage 1 on device 1. It has one slot of 4,096 positions
and one of 65,536, and a block holds 512 tokens. It is on demand, and it loads only through the load routes
(spec 3.3b). It does not fit beside the other engines that are not pinned, so its load names them in `unload`:

```
curl -s -X POST http://<address>:8091/nagaya/v1/engines/gemma/load \
  -d '{"unload": ["autismmix-1", "lfm3-omni", "lfm3"]}'
```

The way back is a load of `lfm3` with `{"unload": ["gemma"]}`, then a load of each of the other two. A load that does
not fit is refused, and the refusal names each engine in the way. A swap waits for the jobs that run on its engines.
No request starts a load, and no engine swaps back by itself (spec 3.3, "Explicit loads", and 3.3a, "Explicit
layout").

`inkling2-mini` of kind `mk-inkling2-bmg` serves Inkling 2 Mini on device 0 (spec 11.18). It has two slots of 32,768
positions and the reserve of 3 cycles, so it takes 25,543,784,132 bytes of its card. It is on demand, and it does not
fit beside `lfm3`:

```
curl -s -X POST http://<address>:8091/nagaya/v1/engines/inkling2-mini/load -d '{"unload": ["lfm3"]}'
```

The way back is a load of `lfm3` with `{"unload": ["inkling2-mini"]}`. The start describes the engine with no load, so
the council serves it as `unloaded` (spec 3.3a). The description reads each tensor of the checkpoint for its digest,
about 20 GB at each start. The research model is a second engine of the same kind, with its own folder. A reserve of
48 runs (`extra_runs = 48`) lets a context run 1 to 5 cycles, and it adds 1,578,975,232 bytes to each slot.

## The DGX Spark example

The profile has one engine, `inkling2-mini-gb10`, of the kind `mk-inkling2-sm121`. It is resident, so each start loads
the weights. The name differs from `inkling2-mini` of the B70 example, so both can serve in one cluster.

- The slots are a list: `slots = [131072, 131072, 131072, 131072]` gives 4 slots of 128K positions. A count of slots
  gives slots of 32,768 positions.
- The cache is `e4m3`. The slots take 9,108,586,496 bytes. An `f32` cache takes about four times as many.
- The plan is 28,303,621,732 bytes. It has four terms:
  - the weights: 18,889,360,896 bytes;
  - the slots: 9,108,586,496 bytes;
  - the work of a pass: 238,297,188 bytes;
  - pinned host memory: 67,377,152 bytes.
- The pool of `device(0)` and `host` is 109 GiB, or 117,037,858,816 bytes. The host has 127,598,640 kB, and the safety
  floor is 12 GiB (spec 4.7). The store takes 4 GiB of host memory from the pool.
- The profile names the model folder twice: in `model_dir` of the engine, and in the family table. The unit sets no
  model variable.
- The unit has no fd store, because the engine exports no arena. The unit stops in 60 s at most.

The test `the_dgx_spark_profile_admits_on_the_probed_gb10_machine` plans the profile on a recorded GB10 machine. It
reads the model folder from `INKLING2_MODEL_DIR`, and it skips when the variable is unset.

## A card test beside the service

A card test needs a card that no loaded engine uses (spec 3.1). Release the card, run the test, and reclaim it:

1. `POST /nagaya/v1/devices/0/release` unloads each engine that names device 0 (spec 3.3b, "The release of a
   device").
2. Run the test with `NAGAYA_TEST_CARDS` set to the PCI address of that card only.
3. `POST /nagaya/v1/devices/0/reclaim` loads the resident engines of the device again.

A pinned engine blocks a release. The release is not stored, so a start of the unit ends it.

## Change the profile

The process reads the copy in the store directory, so a new example file in `deploy/` changes nothing alone. Use one
of these two ways:

- Send the profile as JSON with `PUT /nagaya/v1/profile` (spec 2.3). The process writes the new copy. A change that
  loads or reloads an engine restarts the process with exit 4.
- Stop the unit, copy the new file, and start the unit. On the B70 example, the stop clears the stored arenas, so
  `gemma` loads cold.

  ```
  systemctl --user stop nagaya.service
  cp <new profile> <store_dir>/profile.toml
  systemctl --user start nagaya.service
  ```

Before a copy, compare the copy in the store directory with your last file. A profile swap of a client writes the copy
(spec 2.3), so a difference that you did not make is a swap. Keep it, or replace it on purpose.

## Roll back

Keep the previous binary and the previous profile before each deploy:

```
cp ~/src/nagaya/target/release/nagaya ~/src/nagaya/target/nagaya.prev
cp <store_dir>/profile.toml <store_dir>/profile.toml.prev
```

To roll back, stop the unit, put both files back, and start the unit:

```
systemctl --user stop nagaya.service
cp ~/src/nagaya/target/nagaya.prev ~/src/nagaya/target/release/nagaya
cp <store_dir>/profile.toml.prev <store_dir>/profile.toml
systemctl --user reset-failed nagaya.service
systemctl --user start nagaya.service
```

Then do the steps of "Check a start".
