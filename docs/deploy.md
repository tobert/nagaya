# Deploy on gfx1151host

This doc tells how to build the nagaya and install its systemd user unit on gfx1151host. It tells how to move the service
on port 8090 from mk to the nagaya, and how to go back. The rules are in `docs/spec.md` 2.1, 2.4, 3.3 and 10.1-10.4.

The cutover ran at 00:24 on 2026-10-21. `nagaya.service` serves council v2 on port 8090 with mk on gfx1151, and
`megakernel.service` is stopped and disabled. Use "Cut over from mk" again only after a rollback.

Run each command as `atobey` on gfx1151host. The unit is a user unit, so every `systemctl` command has `--user`. A command
without `--user` talks to the system manager and says "inactive" for a user unit.

## The files

| File | What it is |
|---|---|
| `deploy/nagaya.service` | The systemd user unit. `Type=notify` (spec 10.1). |
| `deploy/gfx1151host/nagaya.toml` | The bootstrap config (spec 2.1). The unit sets `NAGAYA_CONFIG` to it. |
| `deploy/gfx1151host/profile.toml` | The placement profile (spec 2.4). A copy goes in the store directory. |

The test `crates/nagaya/tests/deploy.rs` reads all three with the real loaders. A change to a schema breaks it.

The bootstrap config binds HTTP to `198.51.100.20:8090`, the address of mk's service (Amy, 2026-10-18). It binds
gossip to UDP port 8091 on the same address. It has no seed peers, because gfx1151host is one machine. The store directory
is `/home/atobey/.local/share/nagaya/store`.

The unit sets `OTEL_EXPORTER_OTLP_ENDPOINT` to `http://127.0.0.1:4317`, the host collector. The nagaya exports OTLP
over gRPC (spec 9.1). Without a collector, the log still works.

## Before you start

The gaps of 2026-10-20 night are closed. Check these items before each start, because a start fails when one is
wrong.

- Build the `nagaya` binary with `--features hip` (see "Build"). The feature registers the kind
  `mk-qwen38-gfx1151`. A build without it exits 3 with the message "this build has no engine".
- Build the binary from a commit at or after `9a79c3a`. Before that merge, mk did not declare `slots` and
  `coverage`. The old binary logs "cannot serve engine `mk`" and exits 3 (spec 11.7).
- mk's library must exist and be newer than mk's last commit (AGENTS.md "mk on gfx1151host"). Amy builds it.
- The profile must hold the real size of mk's weight arena. The file in `deploy/gfx1151host/` holds the measured size.
  A new mk or new weights can change it. Read "The weights size" below.
- The context pool is 12 contexts of 4,096 tokens and 4 of 32,768. The binding fixes it, and the profile names no
  context count. Read "The memory budgets" below when mk or the pool changes.
- Nothing else may hold the GPU, and `megakernel.service` must stay inactive (see "Rules for every restart").

## Build

Build the release binary in the repository of the nagaya:

```
cd ~/src/nagaya
nice -n 10 cargo build --release -p nagaya --features hip
ls -l target/release/nagaya
```

The unit runs this file. Build it again before each restart, or the unit runs the old code. The feature `hip` adds
the mk engine. Without it, the binary has no engine and exits 3. The build does not need mk, and it does not open the
GPU: the process loads mk's library at run time from `MK_ENGINE_LIB`.

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

## Cut over from mk

Do these steps in this order. Stop at the first step that fails, and do the rollback below.

1. Check that nothing else is a GPU client (spec 3.1). The nagaya is the only client.

   ```
   systemctl --user is-active megakernel.service
   fuser -v /dev/kfd /dev/dri/renderD* 2>&1
   ```

   The first command says `active` while mk runs. The second command lists the processes that hold the GPU. Stop any
   job of ours that it lists. Do not stop a process that you do not know.

2. Stop and disable mk. The stop frees mk's weights and its GPU queues.

   ```
   systemctl --user stop megakernel.service
   systemctl --user disable megakernel.service
   ```

3. Enable and start the nagaya.

   ```
   systemctl --user enable --now nagaya.service
   ```

   The command returns when the process sends `READY=1`, or when `TimeoutStartSec` (300 s) ends. While mk opens, the
   process sends `EXTEND_TIMEOUT_USEC=` every 5 s, so a slow load does not time the start out. The process ends a
   hung open itself after 300 s and exits 1.

4. Check the state of the unit and its log.

   ```
   systemctl --user status nagaya.service
   journalctl --user -u nagaya.service -b --no-pager | tail -40
   journalctl --user -u nagaya.service -f
   ```

   `status` shows `Active: active (running)` and a `Status:` line with the state of each engine. The log has the
   line "ready with 1 engines". Use `journalctl -f` to follow the log. Stop it with Ctrl-C.

5. Ask for the identity of the served models.

   ```
   curl -s http://198.51.100.20:8090/council/v2/identity
   ```

   The reply is JSON. It lists the served model with its identity, limits, capabilities and status.

6. Run gogi's conformance suite against the process. It needs no restart.

   ```
   cd ~/src/gogi
   nice -n 10 cargo run --release -p gogi-conformance -- http://198.51.100.20:8090
   ```

   The last line counts the checks. On 2026-10-21 it was `215 ok, 0 FAIL, 5 skip`. Each `FAIL` line names a check. Add
   `--model <name>` to check one served model. The flag `--allow-restart <cmd>` also checks a restart, but pinned
   snapshots do not survive one yet (spec 6.11), so leave it out.

7. Look for GPU errors in the kernel log.

   ```
   journalctl -k -b --no-pager | grep -i -E 'amdgpu|kfd' | tail -20
   ```

   Any `amdgpu` error after the start is a failed gate. Do the rollback.

## Roll back to mk

Use this when a gate fails, or when Amy asks for mk.

```
systemctl --user disable --now nagaya.service
cmp ~/src/megakernel-qwen38-flashnext-strixhalo/deploy/megakernel.service ~/.config/systemd/user/megakernel.service
systemctl --user enable --now megakernel.service
systemctl --user is-active megakernel.service
curl -s http://198.51.100.20:8090/mk/v1/health
```

`systemctl --user disable` removes a unit that is a link into a repository. The cutover removed mk's link that way
(seen on 2026-10-21). Thus mk's unit in `~/.config/systemd/user/` is now a copy of the file in mk's repository, and
`is-enabled` says `disabled`. The second line checks that the copy is the same as mk's file. If `cmp` reports a
difference, copy mk's file again and run `systemctl --user daemon-reload`.

The third line starts mk, which loads its weights. The fourth line says `active`. The `curl` answers when
the load ends. mk serves its own route, `/mk/v1`, on port 8090. Its answer proves that mk is back.

To go back to the nagaya later, link its unit again first, as in "Install".

Nothing in the store directory needs a change for the rollback. The nagaya leaves its store and its profile there.

## Rules for every restart

- Build the release binary again before each restart, with `--features hip`. A restart runs `target/release/nagaya`
  as it is.
- Nothing else may hold the GPU while the nagaya runs (spec 3.1). A tool that needs the GPU stops the unit first:
  `systemctl --user stop nagaya.service`. It starts the unit again when it ends.
- mk's unit and the nagaya's unit use the same address. Never start both. A second process on the GPU can stop the
  SMU (spec 3.1).
- A new profile from a client makes the process exit with code 4 and restart (spec 3.3, 10.4). systemd counts that
  start. The unit allows 3 starts in 600 s, so a fourth profile swap in 600 s fails and the unit stays down.
- Exit code 3 is a refusal to start: a bad config, a profile that does not fit, a pin that differs. The unit does
  not restart. Read the log, fix the cause, and run `systemctl --user reset-failed nagaya.service`. Then start it.

## The kaijutsu gate

The live gate council of kaijutsu reads `~/.config/kaijutsu/config/kernel/gate.toml`. Its `[council]` table has
`server = "http://198.51.100.20:8090"`. It talks to whatever serves port 8090. After the cutover, that is the
nagaya, and the thresholds in the file must match the identity that the nagaya serves. A threshold is fitted for one
identity (gogi `docs/council-api.md`). Do these steps after each cutover and each rollback:

1. Read the identity from `GET /council/v2/identity`.
2. Compare its `weight_hash`, `engine`, `tokenizer_hash` and `template` with each `[[council.threshold]]` entry.
3. If one differs, the gate has no threshold for that identity. Ask Amy before you change the file.

On 2026-10-21, a throwaway kaijutsu kernel drove a generation and gate council decisions through the nagaya.

## The memory budgets

The profile has one pool for `device(0)` and `host` (spec 4.1): 112 GiB, or 120,259,084,288 bytes. The host has
about 125.1 GiB. The host safety floor is 12 GiB (spec 4.7). That leaves 113.1 GiB, and the pool is a whole count
below it. The store takes 16 GiB of host memory from the pool, and 50 GiB of NVMe. These are mk's defaults for its
parked contexts.

The mk engine declares its terms in `limits().memory`. The terms are these:

- the weights: 65,490,227,200 bytes (`weights_bytes` in the profile);
- the context pool of 12 contexts of 4,096 tokens and 4 of 32,768: 12,198,215,680 bytes;
- the pass scratch: 324,009,984 bytes.

The sum is 78,012,452,864 bytes. With the store's 17,179,869,184 bytes of host memory, it is 95,192,322,048 bytes.
The pool has 120,259,084,288 bytes, so about 25.1 GB stay free. A new context and a staging copy (spec 4.2) take
their bytes from that room. A copy of a 32,768-token context is about 1.93 GB.

If a change makes the sum larger than the pool, the process refuses the profile and exits 3. The log gives the
shortfall in bytes. Then raise the pool in `profile.toml`, or lower `store.host`. Keep the pool 12 GiB below
`MemTotal`.

## The weights size

The engine config in `profile.toml` has one key, `weights_bytes`: the size of mk's weight arena (spec 2.4,
`docs/engine-abi.md` "v0 mk binding choices"). The plan reads no weights, so the profile must give the size. The
ledger holds these bytes for the weights term.

The file in `deploy/gfx1151host/` holds the measured size: 65,490,227,200 bytes, or 60.99 GiB. The first start on gfx1151
gave it on 2026-10-20. After a new mk or new weights, a start can exit 3 with another size. Do these steps then:

1. Start the unit. mk loads its weights, which takes about 35 s.
2. The process finds another arena size and exits 3. The log says: "mk's weight arena is N bytes, and the profile
   plans M bytes; set `weights_bytes = N`".
3. Write N in both profiles: `deploy/gfx1151host/profile.toml` and `~/.local/share/nagaya/store/profile.toml`.
4. Run `systemctl --user reset-failed nagaya.service`, then start the unit again.

A rebuild of mk can change the arena size, and the same exit 3 follows. Do the same steps then.

## Settings left unset

The unit leaves two mk settings unset. Set one with a drop-in file: `systemctl --user edit nagaya.service`.

- `MK_ENGINE_LIB_SHA256` is the pin of the library. When it is set, the binding refuses a library with another
  SHA-256, and the process exits 3. Get the value with `sha256sum` on `libmkengine.so`. The pin makes a rebuild of
  mk fail the start, so set it only when Amy asks.
- `MK_FAST_QMM=0` selects the exact path. Unset means the fast path, an approximation. The identity carries the
  setting (`docs/engine-abi.md`). A change of it changes the identity.

## Arenas through a restart

The unit does not keep arenas through a restart yet (spec 10.2). Each restart loads the weights again. Later, the
unit sets `FileDescriptorStoreMax=512` and `FileDescriptorStorePreserve=restart`, and the process stores its fds.
The unit file has a comment at that place.

## Decided by senpai 2026-10-20 night (vetoable)

These rules fill gaps in the spec. Each one is the simplest choice that works.

- The engine kind in the profile is `mk-qwen38-gfx1151`, the name of the engine crate. The engine is named `mk`.
- `TimeoutStopSec` is 60 s. A running quantum takes at most 10 s (`max_quantum_ms`), and the park takes 10 s.
- The unit makes the store directory with `ExecStartPre`. The process reads its profile from there before it opens
  the store.
- The gossip port is 8091, next to the HTTP port 8090.
- The pool is 112 GiB, the store's host budget is 16 GiB and its NVMe budget is 50 GiB (see "The memory budgets").
- `interval_copies` is 0 (spec 6.13), so the scheduler makes no interval copy. A larger value turns the copies on.
  A change of it restarts the process (spec 2.3).
- The library pin and `MK_FAST_QMM` stay unset.
- The profile is copied into the store directory once. It is not a link.
- The release binary is built with `--features hip`. The plain build has no engine and exits 3.
- `weights_bytes` in the engine config is the size of mk's weight arena. The file holds the measured size (see "The
  weights size"). A wrong size exits 3.
- A probe error at start exits 3 (spec 2.1, 10.4).
