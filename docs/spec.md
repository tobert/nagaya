# nagaya spec v0 (draft)

This spec says what the nagaya is and what it does, one rule at a time. It is written in Simplified Technical
English (`AGENTS.md`).

- `docs/ontology.md` defines each term once.
- `docs/engine-abi.md` defines the calls and the decisions D1-D10. The spec refers to them by number.
- `docs/prior-art.md`, `docs/deepinfer.md` and `docs/reviews.md` record where the rules came from.

When this spec and another doc do not agree, correct one of them. Neither one wins without a decision.

Status: draft, 2026-10-21. The sources are the design conversations with Amy and inventories of mk
(`~/src/megakernel-qwen38-flashnext-strixhalo`) and mkviolet (`~/src/mkviolet`). The code follows `AGENTS.md`. Each open
question has the mark **Open**, and section 15 lists all of them.

We change this spec before we change the code (Amy, 2026-10-20: "I want to keep updating the spec before the code so
we could redo it all again if we decide to"). The mark "Decided 2026-10-20 (reading)" is on a rule that began as a
**Proposed** reading of the text. Senpai wrote it as a rule, and Amy can veto it.

The mark "Decided by senpai 2026-10-20 night (vetoable)" is on a rule that senpai chose where the spec left a choice
open. Amy can veto it. The mark "Decided by senpai 2026-10-22 (vetoable)" is on a **Proposed** note that senpai
accepted (Amy, 2026-10-22: "if those decisions seem ok to you let's call them ok and melt them in"). Amy reads them
in her review of the docs.

## 1. Scope

The nagaya is one process on each machine. It owns the machine's GPUs, and engines run inside it. It decides three
things: where engines and state live, what runs next on each device, and what the store keeps. It does not decide
what a client asks a model.

| Layer | Owns | Does not own |
|---|---|---|
| kaijutsu (the agent) | coordination: casts, seats, judges, bumps, which model answers what | devices, memory, scheduling |
| nagaya | placement, the ledger, scheduling, the store, IO, machine telemetry, lifecycle, the client API | what a request means; kernels |
| engine | a model's code and weights, its kernels, its private layouts and caches | anything that another engine uses |

Amy, 2026-10-17: "Coordination belongs in kaijutsu, yeah. It's the agent."

Goals for v0:

- Several engines run on one machine at the same time: inference endpoints with different models, image batch, and
  training as filler. All three kinds of engine are in v0 (Amy, 2026-10-17). Thus the ABI must cover image steps
  and training before v0 is done.
- Each engine gives the same bits for the same identity and inputs.
- Nothing is lost or corrupted without a report.

Not in v0:

- One model split across machines. The links are gigabit today (README, "Between machines").
- Engines that load and unload in place while the process runs. In v0, a change of engines restarts the process
  (3.3).
- Cooperative yields inside a pass (D8).
- Tensor transfer between engines (13.3).

## 2. Machines, gossip and placement

**2.1 Inventory.** At start, the nagaya probes its own machine: its devices (amdgpu KFD, Intel Xe, or NVIDIA NVML),
its host memory, and its NVMe store path. It needs only a tiny bootstrap config (`nagaya.toml`) with its
bind addresses, store directory, and seed peer addresses.

| Machine | Devices | Memory |
|---|---|---|
| gfx1151host | Strix Halo, one gfx1151 GPU | 128 GB unified |
| b70host | two Intel Arc Pro B70, and an AMD Radeon 610M for the display | 64 GB GDDR6, 128 GB DDR5, 4 TB NVMe |
| sm121host | DGX Spark, one Blackwell GPU | 128 GB unified |

On 2026-10-21, a capture of b70host's tree showed these facts:

- The CPU is a Ryzen 9 9950X with 32 online CPUs. Each CCD has one L3 domain: CPUs 0-7 and 16-23, then 8-15 and
  24-31 (`cpu*/cache/index3/shared_cpu_list`).
- The B70 cards are at PCI 0000:03:00.0 and 0000:08:00.0. The iGPU is at 0000:80:00.0 (`1002:13c0`, gfx1036). The
  firmware makes the iGPU the primary display.
- No BlueField-2 is in the machine. The NICs are `enp17s0` (igc), `enp18s0` (atlantic, 1 Gb/s) and a Wi-Fi card.
- Both cards have `local_cpulist` 0-31 and `numa_node` -1. Thus the machine has no NUMA locality to use (3.2b).

Telemetry takes hardware attributes from the probed inventory, not from constants in code. (mkviolet puts B70 values
in `telemetry.rs` as constants.) A CPU that runs work is a device in the inventory too, with host memory as its tier.
A machine with no GPU engine can also join the cluster to serve as a coordinator, run CPU devices (13.2), or hold a
store tier.

**The probe reads files, not a GPU runtime.** The inventory probe reads sysfs, procfs and `/dev` under a root path.
It does not open a GPU runtime to count devices. A call into a vendor library is allowed only when it opens no device
context. (DeepInfer reads NVML through `ctypes`.) The tests run the probe against fixed trees for gfx1151host, b70host
and sm121host. Decided 2026-10-20 (`docs/deepinfer.md` 5).

**What the file probe reads.** `FileProbe` in `crates/nagaya/src/probe.rs` reads these files under its root path. The
root path is `/` on a machine, and a fixed tree in a test (`crates/nagaya/tests/fixtures/probe/`).

- Host memory: `MemTotal` in `proc/meminfo`.
- CPUs: the list in `sys/devices/system/cpu/online`. Each CPU number is below 1024, the CPUs that one `cpu_set_t` mask
  holds (3.2a). A list with a larger number is a probe error that names the file. Decided by senpai 2026-10-20 night
  (vetoable).
- AMD GPUs: the nodes of `sys/class/kfd/kfd/topology/nodes`. A node with `gpu_id` 0 is a CPU node. The device name
  comes from `gfx_target_version`, so 110501 gives `gfx1151`. The PCI address comes from `location_id` and `domain`.
- Intel GPUs: each `sys/class/drm/card<N>` whose `device/uevent` says `DRIVER=xe`.
- NVIDIA GPUs: each directory of `proc/driver/nvidia/gpus`, and its `information` file.
- NVMe: each `nvme<C>n<N>` in `sys/block`. The size is the `size` file in 512-byte sectors.
- Network: each interface in `sys/class/net` that has a `device` directory. Virtual interfaces have none.

A device id is the position in the list, in the order that the next paragraph decides.

Decided 2026-10-21 (Amy, by interview): the device ids follow the PCI address, in the order of domain, bus, device
and function, for every vendor. Reason: on b70host the vendor order makes the display iGPU device 0, and the
B70 cards devices 1 and 2. The PCI order gives the cards 0 and 1, and the iGPU 2. A device id then does not change
when a driver loads in another order. gfx1151host has one device, so it keeps id 0.

**Ignored devices.** Decided 2026-10-21 (Amy: "I think we should keep visibility of igpus, but mostly ignore them,
maybe a config sets them as ignored/disabled"). The probe lists every GPU that it can size, also a GPU that drives
the display, such as b70host's iGPU. The bootstrap config key `ignore_devices` lists PCI addresses (2.1, "The
bootstrap config"). These rules apply to an ignored device:

- It keeps its device id, so the ids of the other devices do not change.
- The inventory and the gossip key `inventory` keep it, marked `ignored: true`. Thus a person still sees it.
- It gets no CPU set (3.2b), no budget, no group thread and no work.
- A profile that names it is refused, and the message names its PCI address.
- An address in `ignore_devices` that the probe does not find stops the process with exit 3. Reason: a stale or
  misspelled address must not hide a device silently.

A device that is not ignored and that no engine of the profile names also gets no budget, no group thread and no
work. The nagaya is the only compute client of each device that its profile names (3.1).

A missing directory that lists one kind of GPU means that the machine has no GPU of that kind. This is how a
coordinator machine joins with no GPU (2.1). Any other missing or malformed file is an error that names the file
(8.5).

A probe error at start refuses the start and exits 3 (10.4). Reason: the machine is wrong, not a config, so a
restart cannot help. A build without the fake engine runs `FileProbe::new("/")` through the `Probe` trait when it
starts. Decided by senpai 2026-10-20 night (vetoable).

**What the file probe cannot see.** A probe that opens no GPU runtime has these limits.

- NVIDIA: `proc/driver/nvidia` gives the model, the bus location and the minor number. It does not give the memory
  size. NVML gives it. Thus the probe accepts only the models in a short list whose memory is unified, and it reads
  their memory as the host memory. Any other model is an error.
- Xe: sysfs has no file for the size of the device memory, and `xe` has no `tile0/vram0` size file. The probe reads
  the largest prefetchable memory BAR of the standard BARs 0 to 5, which are the first six lines of the `resource`
  file. The later lines are the ROM and the SR-IOV BARs of the virtual functions. With resizable BAR off, the BAR is
  small, so the probe refuses the card. Resizable BAR is a deploy requirement on b70host.

  On b70host, line 9 of `resource` holds 56 GiB: 7 virtual functions of 8 GiB. BAR 2 holds the 32 GiB of the
  card. A probe that reads every line takes the 56 GiB.
- AMD: the probe cannot tell unified memory from dedicated memory by one flag. It compares the KFD framebuffer
  memory with `mem_info_gtt_total` and `mem_info_vram_total` of the PCI device. Equal to the first means unified.
  Equal to the second means dedicated. Equal to neither is an error.
- Free memory, device clocks and temperatures. Telemetry reads those (section 9).
- The measured edges of 2.2. The probe gives devices and sizes only.
- The NVMe namespace that holds `store_dir`. The probe lists all namespaces.
- Intel cards with the `i915` driver, and any other driver. They are not devices of the nagaya.

Decided by senpai 2026-10-20 night (vetoable): the rules of "What the file probe reads" and "What the file probe
cannot see". Reason: the probe must give a true size or stop. A heuristic that guesses wrong without a sign would
corrupt the ledger (8.5).

Decided by senpai 2026-10-20 night (vetoable): the tree for sm121host is made by hand, not captured. The coder could not
read that machine. The README in the fixture directory marks each guessed value. Capture the real tree when we port.

The tree for b70host is a capture of 2026-10-21. A Xe card reports the largest prefetchable range among its six
standard BARs. The ranges after them hold the ROM and the SR-IOV functions, and they are not memory of the card.

Decided by senpai 2026-10-20 night (vetoable): a build with the feature `fake` keeps a made-up inventory. It has one
device and 1 GiB of host memory. Reason: the host tests need no real machine. A build with none of the features
`fake`, `hip` and `xe` has no engine, so it refuses the start (exit 3, `engine_registry` in `main.rs`).

**The bootstrap config.** The bootstrap config is one TOML file, `nagaya.toml`. Its keys are `http_bind`,
`gossip_bind`, `store_dir`, `seed_peers`, `park_budget_secs` and `ignore_devices`. The first three keys are required.
By default, `seed_peers` is an empty list, `park_budget_secs` is 10 (10.3, ds4-recipe), and `ignore_devices` is an
empty list of PCI addresses (2.1, "Ignored devices"). b70host's config sets `ignore_devices = ["0000:80:00.0"]`.

An unknown key is an error, so a misspelled key stops the process. There are no flags for these values, and the file
is the only source. Reason: two sources for one value can disagree. Decided 2026-10-20 (reading).

The node advertises `gossip_bind` to its peers (2.3). An unspecified address, `0.0.0.0` or `::`, is an address that no
peer can reach. The process refuses it and exits 3, and the message names `gossip_bind`. Decided 2026-10-19.

The bootstrap config has one more key, `node_name`. It is optional. Its default is the machine's host name. The name
is the `<host>` of the gossip keys (2.3), and gossip uses it as the node id. Each machine in a cluster needs its own
name.

A name has at most 128 bytes, so each gossip key that holds it fits 1024 bytes (2.3). Decided 2026-10-20
(reading).

A name has letters, digits, `.`, `_` and `-`, and at least one character. Any other name stops the process with
exit 3.

Reason for `node_name`: a profile for `<host>` needs a name that a client knows. A test that runs two processes on
one machine sets two names. Decided 2026-10-19.

The process reads the bootstrap config from the path in the environment variable `NAGAYA_CONFIG`. When the variable
is unset, it reads `nagaya.toml` in the working directory. The systemd unit sets the variable, and a test points it
at a temporary file. A file that does not read or parse stops the process with exit 3, and the message names the
path. Decided 2026-10-20 (reading).

**2.2 Tier edges.** An edge records more than a bandwidth (`docs/prior-art.md` 5):

- the rate for each read size and queue depth;
- whether direct IO works into the destination;
- the alignment that the edge needs.

The nagaya measures edges at install, stores the result in its local store index, and can measure them again. Each
measurement is data with a date.

**The edges of a machine with discrete cards.** Decided 2026-10-22 (Amy: "accepted"). b70host has these
edges, and the install measures each one:

- host to `device(i)`, and `device(i)` to host. Each B70 links at PCIe 5.0 x8. mkviolet measured about 21.6 GB/s for
  a copy from host to device (mkviolet `docs/07-long-context.md`).
- `device(i)` to `device(j)`. mkviolet copies the residual through host memory (`run_card0`, `run_card1`). A peer
  copy is a separate edge with its own measurement.
- NVMe to `device(i)`, through host memory or direct. The edge records which path works (7.2).

The probe reads the link of a card from the upstream port of the card's own PCIe switch. On b70host the ports are
01:00.0 and 06:00.0, and each reports 32 GT/s x8. The card's endpoint reports 2.5 GT/s x1, because it sits behind
that switch. Thus a probe that reads the endpoint gives a false rate.

**Pinned host memory.** Decided 2026-10-22 (Amy: "accepted"). A copy over PCIe is fastest from pinned host
memory. Thus a mandatory host term may set `pinned: true`. The ledger charges its bytes in the host budget, as any
host bytes. `nagaya.ledger.bytes` reports the term, so the pinned bytes are visible. mkviolet pins a staging buffer
of 4 MiB for uploads (`STAGE_FLOATS`, `gemma4.rs:172`), and host buffers for the routes (`gemma4.rs:521`).

Not built: the term has no member `pinned`. The Gemma 4 binding names its term `pinned.<i>`
(`crates/nagaya-mk-gemma4-bmg/src/spec.rs:197`), and the ledger charges it as any host term.

Not built. The nagaya measures no edge yet.

**2.3 Gossip (`chitchat`).** The nagaya processes form a cluster over UDP with `chitchat` (Scuttlebutt reconciliation
and phi-accrual failure detection). Gossip data stays small and write-light. Each machine writes only its own node
state:

- `inventory`: its probed devices, tiers and measured edges;
- `profile:active`: the placement profile that it currently runs;
- `profile:desired:<host>`: a target profile for `<host>` with a version, set when a client posts a profile to any
  machine;
- `engine:<name>`: the engine's `identity()`, its served models, its state (`loading`, `ready`, `draining`,
  `unloaded`, `fatal`), and its HTTP address.

Pins and snapshot entries never go in gossip.

Each gossip value is compact JSON, and it has at most 8192 bytes. The nagaya refuses to publish a larger value. The
refusal names the key and the size. Reason: `chitchat` sends a delta in one UDP datagram of at most 65,507 bytes. A
node writes a few keys, plus one key for each engine.

A profile that does not fit in 8192 bytes needs a different path, which is not in v0. Decided 2026-10-19.

A gossip key has at most 1024 bytes. The nagaya refuses to publish a longer key, and the refusal names the key.
Reason: `chitchat` 0.13.0 writes the length of a key in 16 bits, and a delta must fit in one datagram. Decided
2026-10-20.

The value of `profile:desired:<host>` is JSON with two fields. `version` is an integer that the client picks.
`profile` is the placement profile (2.4) as a JSON object. The machine that takes a `PUT` writes the key under its
own node.

A node reads the key for its own name (2.1 note) from every live peer. It takes the highest `version`. It ignores a
version that is not higher than the last version that it handled.

The node writes the handled version to `<store_dir>/profile.version`, after it writes `profile.toml`. A restart keeps
the order. A missing file means version 0.

The node handles a version in this order: admit the profile with the ledger, persist it, then act on it (3.3). A
profile that differs from the active profile in any field is admitted and persisted. The node then stops through the
drain and the park (10.3) and exits 4. A profile that equals the active profile changes nothing: the node does not
ask the ledger and does not rewrite `profile.toml`. It records the handled version and logs that nothing changed.
Reason: a profile that the ledger admits must match the engines that run.

A node refuses a profile that does not parse or does not fit. It keeps the active profile and the process runs on.
It logs the reason, and the shortfall in bytes for each tier.

The node writes the key `profile:refused` under its own node. Its value is JSON with `version`, `reason` and
`shortfall`. Each `shortfall` entry has `tiers` and `bytes`. The node removes the key when it handles a newer
version. Decided 2026-10-19.

A refused version counts as handled. The node writes the version to `profile.version`, and it keeps the old
`profile.toml`. After a restart, the node does not handle that version again, and it ignores each lower version.
Decided 2026-10-20.

**A duplicate node name.** Two live nodes can have one name. A chitchat id has three fields: the name, a generation
and the gossip address. A live peer is a duplicate when it has this node's name and is not this node's earlier run.
An earlier run has this node's gossip address and a lower generation. Thus a peer with another address is a
duplicate. A peer with this address and a generation that is not lower is a duplicate too.

For each duplicate, the node logs one error that names both ids as `name:generation:address`. It also writes the key
`name:conflict` under its own node, with the JSON value `{"own": "<id>", "peers": ["<id>", ...]}`. `peers` holds the
ids of the live duplicates, in sorted order. The node removes the key when no duplicate is live. It takes no other
action. Decided 2026-10-19 (Amy: log it and publish a key), and written as a rule 2026-10-20.

**2.4 Placement profile.** A placement profile names a machine's engines, their devices, their config, each tier's
budget (4.1), each engine's share (D9), and pinned identities (12.1). A client reads or updates a machine's profile
through `GET` and `PUT /nagaya/v1/profile` on any machine in the cluster. The target machine persists its accepted
profile in its store directory, so it starts with that profile after a restart.

Before the nagaya applies a profile, it calls `plan(config)` for each engine, which gives the engine's limits without
a read of the weights. The nagaya adds all mandatory terms and the minimum of each elastic term. If the sum does not
fit the budgets (section 4), the nagaya refuses the profile before it opens an engine. The refusal gives the
shortfall in bytes for each tier. At cold start, a refused profile exits 3 (10.4). At runtime, a refused `PUT`
returns 507 and keeps the active profile.

A refusal has one entry for each budget that is short, with the tiers that share that budget. Thus a caller that
adds the entries does not count a pooled budget twice. On gfx1151host, `device(0)` and `host` are one pool (4.2). Decided
2026-10-20 (reading).

The placement profile is one TOML file. Each `[[engines]]` entry has `name`, `kind`, `devices`, `config`, `share`
and `pin_identity`. Each `[[budgets]]` entry has `tiers` and `bytes`, and a tier is `host` or `device(i)`. The
`[store]` table has `nvme`, an optional `host` (4.1a), and an optional `interval_copies` (6.13), 0 by default.
Decided 2026-10-20 (reading).

**One name for each device.** Decided 2026-10-22 (Amy: "all good"). The spec names a device in four places.
Each place uses one of two names, and this map joins them.

- A **machine device id** is the id of 2.1. The profile uses it: `devices = [0, 1]`, and a budget tier
  `device(0)`. The metric attribute `tier` writes it as `device0` (9.4). The fd names of 10.2 use it.
- A **group index** is a position in the profile's `devices` list of the engine. The ABI uses it: a term tier
  `device(i)` (4.2), `limits().stages[].device`, and the devices that `open` gets (engine-abi "Core").
- The ledger maps a term tier `device(i)` to the machine device `devices[i]` of the engine. Then it finds the budget
  of that machine device. Thus an engine never sees a machine device id, and a profile never sees a group index.
- On gfx1151host, both names of the one device are 0.

Decided 2026-10-22 (Amy: "all good"): `limits().devices` holds group indexes, `0` to `n - 1` in order. The
process checks that `n` is the length of the profile's `devices` list, as `check_devices` checks the list today
(`crates/nagaya/src/placement.rs`). The engine finds each physical card by the PCI address in the devices that
`plan` and `open` get. Reason: today `limits().devices` holds machine ids, so an engine on machine device 1 must know
its machine id. That breaks the rule above, and an engine plan would differ between machines for the same config.

An engine name has letters, digits, `.`, `_` and `-`, and at most 128 bytes. A profile with another name does not
parse. Reason: the name is part of the gossip key `engine:<name>` (2.3). Decided 2026-10-20.

The process writes an accepted profile to `<store_dir>/profile.toml`, atomically: a temporary file, then a rename.
The file has the same shape as the input, so a read gives an equal profile. An unknown kind or key refuses the
profile before any engine opens. Reason: TOML is the engine config format, so one format is easier to read and to
check.

## 3. Process

**3.1 One process per machine.** The nagaya owns every GPU on the machine (Amy: "we'd do 1 process per host that
owns the gpus"). The nagaya is the only GPU client. An ignored device, such as a display GPU, is the one exception
(2.1, "Ignored devices"). A tool that needs a device sends its work through the nagaya, or
stops the nagaya first. (ds4-recipe: a shader fault with a second GPU client made the SMU stop responding.)

**3.2 Threads and CPU isolation.** The nagaya isolates control work from compute work with `sched_setaffinity` and OS
thread priorities:

- **Control runtime (`tokio`).** Runs `chitchat` gossip and the HTTP coordinator on reserved housekeeping cores
  (`sched_setaffinity`) at elevated priority. It never runs tokenization, hashing, safetensors conversion, or engine
  passes directly.
- **One ledger thread.** Runs on the housekeeping cores and owns every budget: device memory, host memory and NVMe.
  Group threads and the store send admission requests to it. No other thread writes the ledger.
- **One watchdog thread.** Runs on the housekeeping cores and watches each group thread from outside, because a pass
  blocks the thread that it watches (7.4).
- **One thread per GPU device group.** A device group is the set of devices that one or more engines use. Most groups
  have one device, and mkviolet's group has both B70 cards. Engines whose device sets share a device, also through
  a chain of shared devices, are in one group with the union of their devices. A group has one thread and one
  scheduler. Every engine call for a group comes from its thread, and the call
  is synchronous.
- **One stage thread per device of a staged engine.** An engine that declares `stages` gets one more thread for
  each device of its stages (D1). The group thread runs the scheduler and sends each stage quantum to the thread of
  its device. Each call that touches a device comes from the stage thread of that device. Decided 2026-10-21 (Amy:
  "move stage_pass up"). The thread layout is in engine-abi D1.
- **A bounded CPU worker pool.** Rendering, tokenization, hashing, safetensors conversion, and CPU device passes
  (13.2) run on a bounded pool pinned to the remaining cores at `SCHED_BATCH`. Long CPU work yields between chunks
  to obey `max_quantum_ms`. On unified-memory machines (`gfx1151host` and `sm121host`), the scheduler pauses background CPU
  memory work during a GPU pass to protect GPU memory bandwidth.
- **No pause on a discrete card.** A machine whose cards have their own memory, such as b70host, has no pause
  rule. A pass there reads GDDR6, not host memory. The host traffic of a
  pass is the residual and the uploads, about 11 MB for a block of mkviolet. A measurement under load can add a rule.
  Decided by senpai 2026-10-22 (vetoable).

**3.2a Control runtime first.** The control runtime (`tokio`) arrived with gossip, in milestone 10 step f. The HTTP
server joined the runtime in milestone 12 step 6. The group threads, the ledger thread and the watchdog thread stay
std threads. They do not use `tokio`. Milestone 14 applied the CPU sets (below).

Milestone 10 did not apply `sched_setaffinity`, because the bootstrap config names no core sets. The isolation has a
test. The test reads the CPU mask of each thread with `sched_getaffinity`. It checks that the control threads do not
use the group cores. Decided 2026-10-20 (reading).

**CPU sets (milestone 14).** The nagaya splits the online CPUs into a housekeeping set and a pool set. The bootstrap
config names no core sets, so the split is one fixed rule.

- The housekeeping set holds the two lowest-numbered online CPUs.
- The pool set holds every other online CPU.
- The two sets do not overlap. Together they hold every online CPU.
- The pool size is the number of CPUs in the pool set. The pool has one thread for each CPU in the set.
- Offline CPUs are in neither set. CPU numbers can have holes, so the nagaya does not assume `0` to `N-1`.
- A machine with fewer than three online CPUs cannot split the sets. The process exits 3 (10.4) when it
  applies the split.
- The online CPU list must be sorted and strictly increasing. Another list is an error.

Decided by senpai 2026-10-20 night (vetoable): the rules of the two sets above.

**Where each thread pins (milestone 14).** Each thread pins itself to its set when it starts. A thread that cannot pin
is an error: a worker thread panics, and the process exits 1 (8.5). A thread of the control runtime aborts the
process. Reason: a hook of the runtime cannot return an error, and a thread that runs on the wrong cores breaks the
isolation.

- The control runtime (every worker and blocking thread, `nagaya-control`), the ledger thread (`nagaya-ledger`) and the
  watchdog thread (`nagaya-watchdog`) pin to the housekeeping set.
- The group threads (`nagaya-group-<index>`) pin to the pool set, with the normal scheduling policy.
- The store thread (`nagaya-store`) pins to the pool set at `SCHED_BATCH`. It hashes and converts bytes, which is pool
  work.
- A start with no CPU sets pins no thread. Only a test starts that way. The binary always makes the sets (below).
- An engine's own threads keep the mask of the group thread that starts them. Linux gives a new thread the mask of
  the thread that starts it. mk starts its worker thread (`mk-engine`) in `open`, on the group thread, so the worker
  runs on the pool set. Decided by senpai 2026-10-20 night (vetoable). Reason: the process crate makes the sets, and
  the group thread is pinned before it opens an engine. Thus the factory needs no CPU argument.

The binary reads the online CPUs through the file probe (2.1) and makes the sets before it starts gossip. A probe
error or a split error is a refusal to start: exit 3 (10.4).

Decided by senpai 2026-10-20 night (vetoable): the group threads run on the pool set. A group thread drives the GPU,
and it does host work between passes: it builds segments, copies bytes and reads taps. The host work is the same kind
of work as the pool does, and it needs whole cores. A group thread on the housekeeping set would compete with gossip
and the HTTP coordinator. A stalled gossip round is worse than a slow pass.

The pool has one thread for each pool CPU, and a group thread uses one more. This oversubscription is small. A
measurement on gfx1151host can change it.

Decided by senpai 2026-10-20 night (vetoable): the sets use logical CPUs, and the split does not know SMT siblings.
A housekeeping CPU can share a physical core with a pool CPU. Reason: on gfx1151host, CPU 0 pairs with 16 and CPU 1 with
17 (`thread_siblings_list`), so the two housekeeping CPUs sit on two physical cores. A rule for other machines would
only guess before a measurement. The bootstrap config can name the sets later.

**Open:** on gfx1151 on 2026-10-20, two of the four `mk-engine` threads had the mask 0-31. The worker and one other
thread had the pool set, 2-31. The HIP runtime starts the two other threads and sets their masks itself. The nagaya
does not pin threads that a runtime starts.

**Open:** 3.2 asks for elevated priority on the control threads. Milestone 14 does not apply it. A raised priority
needs `CAP_SYS_NICE` or a setting in the unit.

**3.2b CPU sets for each device.** Decided 2026-10-22 (Amy: "accepted"). 3.2a puts every group thread in one
pool set. A stage thread (3.2) drives one card, so it gets a set of its own.

- A stage thread pins to the CPUs of the card's `local_cpulist` that are in the pool set.
- A card whose `local_cpulist` holds every online CPU has no locality. Then the nagaya gives each such card one L3
  domain (`cpu*/cache/index3/shared_cpu_list`), in the order of the device ids. With more cards than domains, the
  cards take the domains in turn.
- On b70host, card 0 gets CPUs 2-7 and 16-23, and card 1 gets CPUs 8-15 and 24-31. CPUs 0 and 1 are the
  housekeeping set.
- The group thread, the CPU pool and the store thread keep the pool set (3.2a).
- A machine with one card has nothing to separate. Its card keeps the pool set, and the nagaya reads no CPU list.
  Reason: on gfx1151host the behavior of 3.2a stays as it was.
- A card with a `local_cpulist` that holds a part of the online CPUs takes that part, less the housekeeping set.
- A card whose set holds no pool CPU is an error. The start is refused: exit 3 (10.4).
  Reason: a stage thread on the wrong CPUs breaks the isolation.
- The L3 domains come in the order of their lowest CPU.

Reason: the two CCDs of the 9950X have their own L3. A stage thread that stays on one CCD keeps its host buffers in
one cache.

The Level Zero runtime and IGC start threads of their own, as the HIP runtime does (15 item 14). Linux gives a new
thread the mask of the thread that starts it. Decided 2026-10-22 (Amy: "accepted"): the engine calls `zeInit` on
the group thread, so those threads start with the pool set. The nagaya does not pin a thread that a runtime starts.
**Open:** measure the masks of the Level Zero threads on b70host, as on gfx1151.

**3.3 Engine lifecycle and profile swaps.** The nagaya opens each engine at start, from the profile. mk cannot close
inside a process today: "the HIP runtime's own teardown segfaults" (mk `service/__main__.py`). Thus in v0, when
`PUT /nagaya/v1/profile` changes the profile, the nagaya persists the profile and stops as in 10.3. It keeps its
arenas in the file-descriptor store (10.2) and exits with code 4, and systemd starts it again. Later, an engine whose
`close()` is safe loads and unloads in place (README door 1).

**Open:** Do the GPU queues of an idle engine slow the other engines on its device? mk has a note that its idle
queues slow another process, but we did not find the source. A measurement answers the question.

## 4. Ledger

The ledger is the machine's account of memory (ontology "Ledger", D9, ds4-recipe `05-memory-ledger.md`). The
ledger thread owns it (3.2).

**4.1 Budgets.** Each tier has one budget. A budget is a byte target in the placement profile (2.4). It is not a
measurement of free memory.

- gfx1151host and sm121host: one shared pool for device and host memory.
- b70host: one budget for each card's device memory, and one for host memory.
- Every machine: one budget for the store's host-memory tier, and one for the store on NVMe.

**4.1a The store's budgets.** The placement profile names the store's NVMe budget in bytes. The profile may name the
store's host budget. Without it, the host budget is the host-only budget's unassigned bytes (4.4 step 4). A profile
with an engine that names no NVMe budget is refused at start (10.4, exit 3). A profile with no engine may omit the
store section: it parks nothing.

Reason: a store write must not take bytes that an engine needs. An unbounded NVMe budget can fill the disk with no
warning (Amy, 2026-10-19). The owner quota is unbounded, and the chain limit is open.

**4.2 Terms.** Each engine declares its terms in `limits().memory`. `plan(config)` gives the same terms before
`open` (2.4).

- Mandatory terms `{name, tier, bytes, lifetime, kinds}`. The lifetime is per engine, per context, or loaned (one
  holder at a time). mk declares its whole pool as one term with lifetime `engine`. The ledger reads the lifetime
  like this. Decided 2026-10-20 (reading, from a note by Sonnet 5.5 in milestone 12).
  - `open` charges each `engine` term and each `loan` term once. A loan has one holder, so one copy is enough.
  - `open` does not charge a `context` term. Each `ctx_new` charges the bytes of every `context` term of its
    engine, in the tier of the term.
  - `ctx_new` takes the bytes from the unassigned bytes of the budget (4.4 step 4). If they are too few, the
    ledger refuses and changes nothing (4.3). The reclaim of 4.6 is the caller's step.
  - `ctx_fork` and `ctx_copy` make a context too, so each charges as `ctx_new` does.
  - The end of a context releases its bytes.

  Reason: this is the smallest rule that gives the lifetime a meaning. No engine declares a `context` term today,
  so mk does not change.
- Elastic terms `{name, tier, unit_bytes, minimum, maximum, kinds}`. The minimum and maximum are counts of units.
- A term's tier is `device(i)` or `host`, where `i` is an index into the engine's device group. The ledger maps each
  tier to a budget (4.1). On gfx1151host and sm121host, `device(0)` and `host` map to one pool. NVMe belongs to the store.
- Each term has a key by kind of quantum, where the bytes change by kind. A prefill's scratch is not a decode's
  scratch.

The nagaya declares its own terms in the ledger: staging, store buffers and IO. The ledger charges every allocation.
Bytes without a charge are a defect. (ds4-recipe: vision weights without a charge "show up as untracked bytes".)

**Staging.** The ledger makes a staging charge for each copy of engine state in host memory. Two copies get the
charge. One is the copy that a read or a decision keeps, so that its later questions start from it. The other is the
buffer that holds a snapshot until the store takes it. Decided by senpai 2026-10-20 night (vetoable).

- The size is the stored bytes of one slot. A fixed part counts whole. A cells part and a ring part count their rows
  up to the highest row of the slot. A 32,768-token mk context is about 1.93 GB.
- The job asks the ledger for the bytes before it reads the state. The tier is `host`. The bytes come from the
  unassigned bytes of its budget (4.4 step 4), as the bytes of a new context do.
- A refusal reads nothing and changes nothing (4.3). The client gets 429 `busy` (11.5), with the shortfall in bytes.
- The charge ends when the copy drops. A job drops its copy before its caller hears the result.
- The store's own bytes are in the store's budget (4.1a). The charge covers the transient buffer only.
- A release that the ledger refuses leaves the ledger wrong. This holds for a staging charge and for a context charge.
  The service records a failure, the main loop sees it, and the process stops with exit 1 (8.5). A release in a
  stopping process is no failure, because the ledger has no charge left.

Reason: the copy is the largest transient allocation of the nagaya, and no term charged it (kaibo review,
2026-10-20, finding 3).

**4.3 Admission.** The ledger thread admits each call that allocates before the call runs. These calls are `open`,
`attach`, `ctx_new`, `ctx_fork`, `ctx_copy`, `part_write`, and each resize. The check includes transient peaks. A
refusal charges nothing and changes nothing.

`pass` never allocates. Its scratch for each kind is a mandatory term, which the ledger charges at `open`.

Admission of a profile keeps the charges that live now. Decided by senpai 2026-10-20 night (vetoable).

- A live context and a live staging copy stay charged when the ledger admits a profile. A later release finds its
  charge.
- The ledger takes the live charges from the unassigned bytes of the new budgets. It charges a live context with the
  context terms of the new profile.
- If the live charges do not fit the new budgets, the ledger refuses the profile and changes nothing. The shortfall
  is in bytes (4.3).

Reason: a profile swap (3.3) admits a profile while requests run. A reset of the charges made a later release fail.
The failed release stops the process (4.2), so a planned restart would have become a crash. The alternative, a
refusal of every admission while a charge lives, would make a transient request refuse a profile for good (2.3).

When the ledger refuses a quantum's call, the quantum waits in the scheduler and keeps its place. A request that can
never fit, even with all reclaim done (4.6), gets 413 `too_large`. Its `param` names the limit that it is over.

**4.4 Spare memory.** The ledger first charges all mandatory terms and the minimum of each elastic term. If they do
not fit, the nagaya refuses the profile (2.4). The ledger then divides what remains in each budget like this:

1. Give each engine a part of the spare bytes in proportion to its share (D9).
2. Inside an engine, split its part among its elastic terms in proportion to each term's minimum bytes. When every
   minimum is 0, the elastic terms get no spare bytes.
3. Round each term down to whole units, and cap it at its maximum.
4. Leave the bytes that remain unassigned, and report them.

Counts come from this arithmetic, never from free memory. Decided 2026-10-18.

Some engines have bits that depend on the size of a cache. Such an engine declares that cache as a mandatory term,
not an elastic one. Thus reclaim and resize cannot change its bits, and each start gives it the same size.

**4.5 Resize.** An elastic term changes size only between quanta, through `resize(term, units)`. It does not shrink
while a lease refers to what would go. (A pin is a lease too.)

**4.6 Reclaim.** When admission needs bytes, the ledger tries these steps in this order:

1. Park idle contexts that have no lease to a lower tier, the oldest first (D7).
2. Shrink the elastic terms of idle engines to their minimums.
3. Refuse the request.

A park out of a slot uses `part_read`, which changes nothing in the slot. The nagaya frees the device copy only
after the copy in the lower tier is durable. Reclaim never seals. It never discards state that has no other copy.

**4.7 Host safety floor.** mk keeps 12 GiB of MemAvailable free (mk `service/contexts.py`, `FLOOR_BYTES`). In the
nagaya, a safety floor is a check for what the ledger cannot see: other processes, tmpfs and swap. It does not set
any size. When free memory goes below the floor, the ledger refuses new admissions, resizes nothing, and reports
the event.

On machines with unified memory, RSS and the OOM scorer do not show GTT. Thus the nagaya sets its own OOM score.

Not built. The ledger checks no floor, and the process sets no OOM score. The mk binding checks the floor when it
opens (engine-abi, "The context term holds the pool").

## 5. Scheduling

**5.1 Units.** A job is the quanta of one request, in order, with a class (ontology). A quantum is one pass or one
stage. It runs to its end, and the nagaya chooses what runs next only between quanta.

For an engine with stages, the nagaya chooses between quanta on each device apart. A stage quantum of a block is
ready after the stage before it ends. The job's lease keeps every other job from the slot until the job ends
(engine-abi D1, "The hold of the slot"). Decided 2026-10-21.

An engine meets its `max_quantum_ms` (D8) when it chooses the size of each unit of work. The engine knows its own
speed, so it picks the size from its own measurements. (mk splits a prefill into 256-token chunks in its Python
service, and that split moves into mk's engine wrapper.) The watchdog reports each quantum that runs past the bound
(7.4).

**5.2 Kinds.** Each quantum has a kind: `read`, `prefill`, `decode`, `embed`, `copy`, `step` (an image step), or
`train` (a micro-batch). Metrics and the cost model use kinds. A kind does not set urgency (mk `mk.gpu.kind`).

**5.3 Classes.** The classes are `interactive`, `batch` and `filler`. A client sends `interactive` or `batch` in
council v2's `class` field (gogi). Only jobs that the nagaya or a profile starts, such as training and the audit, are
`filler`. Between quanta, the scheduler always runs a
ready quantum of a higher class first. A quantum is ready when its job has no quantum that runs, its inputs are
present, and the ledger admitted its calls (4.3).

A council decision is interactive. A job's class starts to apply at its first quantum, not when the job arrives.
mkviolet's `lazy_high` takes the card at the council's first GPU step. That change cut decisions beside a prefill
from 383-702 ms to 300-390 ms.

**5.4 Order in a class.** The classes do not schedule the same way (Amy, 2026-10-18: "batch/filler can be fifo-ish,
while interactive is more cooperative multitasking"):

- In `batch` and `filler`, the scheduler orders jobs by integer priority (0 by default) and then time of arrival.
  One job runs its quanta in flow until it finishes, or until a higher class or higher priority job becomes ready.
  The scheduler does not interleave jobs of equal priority in `batch` or `filler`.
- In `interactive`, the scheduler interleaves ready jobs by three keys: integer priority (0 by default), owed
  seconds, and time of arrival. Owed seconds is the device time that a job is owed under equal sharing among the
  present jobs in `interactive`, clamped to ±30 s. A job that leaves takes only its own balance (ds4-recipe
  `11-multi-session-serving.md` section 6).

gogi settles the first key: "a higher priority runs first within its class" (`~/src/gogi/docs/council-api.md:159-160`).
Among equal priorities, the higher owed balance runs first, and then the earlier arrival.

When an interactive pass has several segments, each segment gets an equal part of the pass time. The rounding
remainder goes to the segment with the earliest arrival. Decided 2026-10-20 (reading).

The scheduler charges each segment on its own. A segment of job `j` with a share of `c` ms, among `n` present
interactive jobs, changes the balances like this:

1. the balance of `j` falls by `c*(n-1)/n`, rounded down;
2. the balance of each other present job rises by `c/n`, rounded down;
3. the rounding remainder goes to the present job with the earliest arrival, so the changes sum to zero;
4. each balance then stays between -30 s and +30 s.

A job that leaves removes its own balance. Step 4 forgives the excess, so the balances sum to zero only while no
balance is at a bound. Decided 2026-10-20 (reading, after the audit found two readings of the old text).

**5.5 Filler.** A filler job declares a checkpoint interval: the time between its checkpoints. It must be resumable:
its state parks when it gives way (Ai2's "unallocated" time). It runs only when no interactive or batch quantum is
ready on its device group.

One scheduler serves one device group (spec 3.2, D1: one thread per device group). Two engines that use one device
share one scheduler. The filler gate is per group. Decided 2026-10-18.

**5.6 No preemption.** Nothing interrupts a quantum that runs. Priority orders the work that waits. Priority never
moves a continuation off the context that holds it. The job holds that context with a lease.

**5.7 Segments.** Quanta from several jobs on one engine can share a pass as segments. The scheduler permits this
only when the engine declares `segments_exact` (`docs/prior-art.md` 1). Only segments of one class share a pass, and
the quantum takes that class.

Each class has a join window: the time that the scheduler waits to add more ready segments to a pass. It never
waits for jobs that are not ready. In v0 each window is 0 ms, so a pass takes only the segments that are ready when
the scheduler chooses. Decided 2026-10-18.

In v0 the windows have no key in the bootstrap config. A measurement that
changes them adds the key (2.1). Decided 2026-10-20 (reading).

An engine with a feed rule (6.14) declares no `segments_exact`. The scheduler joins its segments only when no
segment writes rows that the store keeps, as for reads and branch blocks. Decided 2026-10-21 (engine-abi D11).

A pass holds segments of one kind only (5.2), and at most the engine's `max_segments`.
The scheduler joins a ready job only when its next quantum has the
kind of the first segment. Decided 2026-10-20 (reading).

**What `segments_exact` promises.** An engine that declares `segments_exact` promises that the order of each sum
depends only on the shape of the problem. It never depends on the segment count, the row count of a pass, or the
count of compute units. The split of a segment's tokens into passes does not change bits either. Thus the engine does
not choose a split of a reduction by the size of a pass, and it adds
no floats with atomics. Decided 2026-10-20 (`docs/deepinfer.md` 2). DeepInfer's batch-invariant mode keeps this
rule: its gate GEMM never splits K, and its other splits have a fixed count.

**5.8 Deadlines.** A job can have a deadline (council v2 `timeout_ms`). The scheduler checks it between quanta. A
job at or after its deadline goes to the terminal state "expired", and its segments do not run again. On a status code,
an expired job is a 504 `timeout`. On a stream that already sent its first event, the nagaya sends a terminal
`error` event of type `timeout` (gogi).

**5.9 Cancellation.** A client cancels one job. The scheduler drops the job's next segments, and the quantum that
runs goes to its end. Cancellation is not failure: a cancelled job keeps its source snapshots. (ds4-recipe: the
loss of one such snapshot cost a cold prefill of 35 minutes.)

Every terminal state of a job releases its loans and leases. The terminal states are done, cancelled, failed and
expired. A release does not delete a snapshot. Only eviction (6.9) removes a stored entry.

**5.10 The door limit.** The client API accepts a limited number of requests in flight for each engine (mk:
16). Above that limit it answers 429 with `Retry-After`.

Milestone 12 step 6 limits the calls in another way (11.7). It allows 64 calls at once for the process, and it answers
429 `busy` with no `Retry-After`. The limit for each engine is not built.

**5.11 Cost model (D4).** The nagaya fits a robust affine cost from production quanta, keyed by engine, kind,
segment count and context depth. Quanta that ran beside other work are not in the fit. The nagaya uses the fit for
store eviction value-per-byte (6.9) and deadline checks between quanta (5.8). An engine can have its own private
planner, which stays in the engine (mkviolet `council_server.rs`).

**A measured prior.** At the first start of an engine identity, the nagaya times a short curve for each kind. Each
point is one warmup run and the trimmed mean of five runs. The curve goes into the store's index under the identity
digest, and later starts read it. The fit starts from the curve and corrects it with production quanta. Decided
2026-10-20 (`docs/deepinfer.md` 4). DeepInfer measures its decode curve in this way
(`deepinfer/worker/model_runner.py:586-619`).

Not built. The nagaya fits no cost curve yet, and it times no measured prior.

## 6. State and the store

**6.1 One store per machine.** One content-addressed store holds the snapshots of every engine (D3). The store has
tiers (host memory, NVMe) and gets its budgets from the ledger. It replaces these stores:

- mk's parked files (`.mkctx`, a 50 GiB budget);
- mkviolet's `council_server` snapshots, its `held::Store`, and the prefix cache in `mk_api`. These are three
  formats and three SHA-256 id schemes on one host.

**The store tier is `host` or `nvme`.** An entry's tier is `host` or `nvme`, apart from the ABI's memory tier. The
ABI's memory tier, `device(i)` or `host`, names where an engine term lives. The store does not use it. Decided
2026-10-18.

Reason: 6.1 gives the store its own tiers. An entry is host memory or a file, never device memory.

**6.2 Keys.** A snapshot's content id is the BLAKE3 hash of its sealed bytes. When the snapshot is a piece of a
chain, the hash includes its parent's content id. A snapshot's recipe is a hash of these items:

- the engine identity, which includes the tokenizer and the template;
- the logical inputs: the tokens, or the rows as bytes;
- the boundaries;
- the engine's `geometry_key`, if the engine declares one (D3);
- the tags of each approximation;
- lab state that changes bits (6.12).

The recipe is BLAKE3 in derive-key mode, with the context string `nagaya recipe v1`. Each item is a `u16` tag, a
`u64` length and its bytes, all little-endian, in a fixed tag order. Tokens are `u32` little-endian. Recipes chain at
boundaries:

- The root recipe covers the identity digest, the `geometry_key`, the approximation tags, and the lab state.
- The recipe at boundary k covers the recipe at boundary k-1, the tokens since that boundary, and the position k.

Thus a prefix lookup hashes only the tokens since the last boundary. The identity digest is the same tagged hash over
the fields of `identity()` (12.1). The inputs are tokens, not text: the tokenizer and template are in the identity.
Decided 2026-10-18.

The tag numbers are in this table. Decided 2026-10-20 (reading).

| Tag | Item |
|---|---|
| 1 | the identity digest (12.1), which the caller gives |
| 2 | the geometry key |
| 3 | the approximation tags: a `u64` count, then each tag as a `u64` length and its bytes |
| 4 | the lab state |
| 5 | the previous recipe |
| 6 | the tokens |
| 7 | the boundary position |
| 8 | "not sealed", with no bytes: only on an interval copy (6.13). It follows tag 5, which holds the sealed recipe at the same position |

**The park recipe.** The park knows the bytes of a state in a slot. It does not know the inputs that made the state.
Thus the park recipe names the bytes. 10.3 gives the one exception. The park recipe is a root recipe with four items:

- tag 1, the identity digest;
- tag 2, the fixed key `nagaya park v1`;
- tag 3, no approximations;
- tag 4, the 32 bytes of the snapshot's content id.

Equal bytes give one entry, and the tripwire fires only on a real duplicate. Decided 2026-10-20 (reading, kaibo
job-2 finding 2).

mk's `context_id` and mkviolet's `snap:<sha256>` are the input part of a recipe. They do not include the engine
identity. mk's id is a SHA-256 of the tokens and boundaries with `|gdn-bf16-v1|`.

**6.3 Sealed bytes are canonical.** In the store, each snapshot is a canonical safetensors file (D10). Each cells
part and ring part has the shape of its coverage: the rows that it holds, in position order. Thus two slots of
different sizes that hold the same rows make the same file.

Padding between tensors is zero. The header pads with spaces (0x20), so that the data starts on a 4 KiB
boundary (D10 rule 5). The word "zero" names the gap between tensors, not the header padding. Decided 2026-10-18.

Times, hit counts, owners and reasons go in the store's index, never in the sealed bytes. If they go in the sealed
bytes, two copies of one state get different content ids. Then the store keeps both copies, because it cannot see
that they are the same state.

**6.4 Entries.** Each entry in the index records:

- the content id, the recipe, the parent, and the coverage;
- the tier, the size, and the owner. The owner is the client principal that created the entry;
- the times of creation and last use;
- the reason for the store: anchor, interval, prefill done, evict, reclaim, or shutdown;
- the provenance of an approximate entry;
- the strength of the store's last check of the entry: `full` (every byte hashed), `spot` (a sample hashed) or `none`
  yet;
- the engine's rules for the entry: its determinism class, its bit dependence and its chain limits (D3, 6.5). The
  put carries them, because one store serves every engine. The tripwire (6.7) and the chain checks read them from
  the entry. Decided 2026-10-20.

The index also holds two kinds of record that are not entries. They are the expected identity of each engine
(12.1), and the cost curve of each engine identity (5.11). Decided 2026-10-20.

**One `Check` enum.** The store's `Check` is `full | spot | none`, the same enum that the ABI uses for arenas and
store entries (engine-abi "Capabilities"). Decided 2026-10-18.

The store sets `none` after a restart, before it hashes the blob. It sets `full` after it hashes the blob. It does
not produce `spot` in v0. Reason: `spot` now names mk's digest, and the store does a full hash of each entry.
Decided 2026-10-18 evening.

**6.5 Chains.** Each engine declares a maximum chain depth and a size fraction. (ds4-recipe: depth 1, and a delta
only when it is at most a quarter of a full snapshot.) A delta holds, for each cells part and ring part, the rows
after its parent's end. It holds each fixed part whole. The store never evicts a base while a delta needs it.

A ring part can wrap past its parent's end (kaibo, 2026-10-20). Then the rows after the parent's end are gone, and
check 4 refuses the delta. The refusal changes nothing, so nothing is corrupt.

Decided 2026-10-22 (Amy: "all good"): a ring part can start after the end of the parent. Then the build puts a
whole snapshot with no parent. Check 4 stays as it is. Reason: the store needs no new rule. A ring held whole in the
delta holds 4,096 rows of each sliding layer of mkviolet. That is close to the size of a whole snapshot.

A delta's `put` names its parent content id and its previous recipe. The store checks these facts:

1. The parent is live.
2. The delta is no deeper than the chain limit.
3. The size of the delta is within the delta fraction of the chain's base, which is a full snapshot.
4. The coverage of the delta starts at the end of the parent.
5. The previous recipe equals the recipe of the parent entry, and the recipe of the delta differs from it.

The store does not have the tokens, so it cannot compute the recipe again. The caller holds the inputs that prove
it. Decided 2026-10-20 (reading).

**`max_depth` counts deltas, not entries.** A chain with a base and one delta has depth 1. A store with
`max_depth: 0` refuses every delta. Reason: 6.5 gives ds4-recipe depth 1 with one delta. Decided 2026-10-18
(senpai).

**6.6 Lookup.** A lookup is by recipe. It returns the chain of entries that covers the request, from its base to
its tip. The store keeps only the chains that fit the slot (coverage) and the served model's tolerance (ontology).

The store picks the deepest exact chain. The deepest chain is the one whose furthest covered position is highest in
every part. The store gives an approximate chain only to a served model whose tolerance names that method.

**The lookup refuses incomparable chains.** Two chains can be incomparable under the deepest rule: each is higher in
a different part. The store refuses the lookup with an error that names both chains. It does not pick one by a
tie-break. The process treats that refusal as a miss and recomputes. Decided 2026-10-18.

Reason: a silent choice would not be the deepest in every part.

**An equal chain goes to the newest entry.** Two chains can be equal: the same furthest covered position in every
part. A tie is not an incomparable pair. The store serves the newest entry by `created_ms`. If two entries have the
same `created_ms`, the store serves the smaller content id. Decided 2026-10-18.

Reason: a fixed tie-break keeps a put that the store allowed from failing at lookup.

Note: D3 says a per-process engine refuses only within one process, and 6.7 lists only `always` and `pinned`. A
per-process store can hold two equal-coverage entries under one recipe.

**6.7 The tripwire.** Some engines declare `always` or `pinned` (D3). For those engines, two content ids under one
recipe with the same coverage are a defect. The store quarantines both entries, refuses that recipe, and reports
it. A person clears the refusal after the cause is known.

Decided 2026-10-18 (Amy). Clearing a tripwire refusal removes the recipe's quarantined entries from the index and
moves their blobs to the store's `quarantine/` directory. The blobs stay there until a person removes them. No budget
counts that directory, and no lookup, restore or eviction reads it. It does not clear a content-id quarantine;
`clear_entry` does that. Reason: 6.8 says that a failure never deletes content, and a person can find the cause only
while the blobs exist.

For other engines, the audit job (8.4) does the check.

**6.8 Restore.** The nagaya checks the content id, the identity and the fit before it calls the engine.
`part_write` and `cells_write` return one of three results:

- ok;
- refused, and nothing changed;
- failed after a change. The state is poisoned (D6).

A restore stops at the first write that is not ok, and it names the part that stopped it. A refusal of the first
write is a refused restore: nothing changed. A refusal after an earlier write changed the state is a failed
restore, and the state is poisoned (8.5). Decided 2026-10-20 (reading).

The store quarantines an entry only when its content id or its identity fails verification. A blob whose hash
matches but whose bytes do not decode fails its content id too. So does a blob whose parent or coverage differs
from its index record: the store cannot tell which one is right, so it serves neither. Decided 2026-10-20
(reading). A refusal leaves the entry as it is. A failure never deletes content.

A read error of a blob is not a quarantine. The lookup or the
restore treats it as a miss, and the caller computes the state again. The entry stays in the index, the store logs
the error, and the next use checks the entry again. Decided 2026-10-20.

**6.9 Eviction.** The store chooses victims in this order:

1. Interval copies go first, the least recently used first (6.13).
2. Pinned entries are exempt.
3. Thin the chains: interiors go before tips and roots.
4. Remove the remaining entries by value per byte: recency, hits, and the cost to build them again. Tips, roots and
   entries without a chain are in one ranking. Decided 2026-10-20 (reading).

A chain goes as a group: a root takes its chain with it, and the group's value is the value of its first entry.
The formula for value per byte is a choice of the implementation. It never changes bits. The store chooses all
victims before it deletes one. It also chooses them before it writes the new entry.

Each owner has a quota, so one long conversation cannot remove the others (a ds4-recipe incident). If the entries
of one pin are larger than the tier's budget, the pin loses its protection, so eviction cannot deadlock.

**An owner's quota is per store tier.** A put evicts the owner's entries in its own tier until the owner's total
fits the quota and the tier's total fits the budget. Reason: memory and NVMe are separate budgets in 6.1. A byte in
one tier does not make room in the other. Decided 2026-10-18.

**6.10 Pins.** A pin is a lease on a snapshot, with an owner and an expiry (mkviolet v2: one day at most). A request
pin holds a snapshot from its lookup to the end of its load. Pins stay through a restart (mkviolet `pins.json`).

**Many owners can pin one entry.** Each owner holds its own pin, with its own expiry. An entry is pinned while any
pin on it is live. An unpin removes only the caller's own pin. A pin on a tip also keeps the entries under it,
because eviction takes a chain as a group (6.9). Decided 2026-10-20.

**The one-day maximum is for every pin.** The one-day maximum is a nagaya rule for every engine, not only for
mkviolet. The store has no clock, so `pin` takes `now_ms` from the caller. It refuses an expiry more than one day
(86,400,000 ms) after `now_ms` with `Invalid`. A refused pin changes nothing. Decided 2026-10-18.

Reason: one store holds every engine, pins stay through a restart, and a lease with no bound can hold bytes
forever. The mkviolet rule is the origin, not the scope.

**6.11 Restart.** At start, the store loads its index again. It checks each entry when a lookup first uses it.
Entries in the host tier do not survive a restart, so the store removes them from the index at start. Decided
2026-10-20 (reading).

The hits and the last use of each entry are in memory only, and a restart starts them again at zero. **Open:** a
flush at a stop or at an interval keeps them. A write at each use is too slow.

At start, the store also wakes pinned entries into slots when the profile asks for them. (mk wakes its pinned
parked contexts before `READY=1`.) Not built: the nagaya wakes no pinned entry at start.

**Open:** A restart loses an entry of the host tier, also when a pin holds it. The index drops the entry and its pin
at start. The stop parks the states of the slots, and it does not park the pinned entries of the host tier. Thus a
pinned snapshot does not survive a restart. The restart check of `gogi-conformance` (`--allow-restart`) needs a
pinned snapshot to survive. The stop must park the pinned entries of the host tier to NVMe for that check to pass.

**6.12 Steering and lab inputs.** Some state changes bits but is not an input, for example mk's attention cell mask
or a steering vector. Such state goes into the recipe. Thus a steered or masked result never has a plain recipe.

**6.13 Interval copies.** An engine can declare a checkpoint grain in `limits()`, in tokens. Then, between quanta,
the scheduler makes an interval copy (ontology) of each context that passed a multiple of the grain. Milestone 11.5
built the store's part. Milestone 12 step 7 built the scheduler's part on the job path.
Decided 2026-10-20 (`docs/deepinfer.md` 3), and reworked the same day after kaibo's review.

mk declares a grain of 256 tokens. Decided by senpai 2026-10-20 night (vetoable). Two GPU tests in
`crates/nagaya-mk-qwen38-gfx1151/tests/hip.rs` measured it on gfx1151:

- `an_interval_copy_restores_and_replays_to_the_bits_of_the_live_context`: a copy at 512 tokens left the live
  bits as they were. A context of 32,768 tokens restored the copy and replayed 188 tokens. Every part and all
  248,320 logits were equal to the live context.
- `a_split_across_256_does_not_change_the_bits_in_either_class`: four splits of 600 tokens crossed 256 and 512 at
  different tokens. One split crossed 256 one token at a time. A context of 4,096 tokens and a context of 32,768
  tokens gave equal parts and logits.

- **A store entry that is not sealed.** The copy is a store put like any other: it has a content id, a recipe and
  an entry (6.4) with the reason `interval`. Its recipe adds the item "not sealed" (6.2 tag 8), so a lookup for a
  sealed snapshot never finds it.
- **No seal.** The copy does not seal, so the live context keeps its bits. A seal would change every later token
  of every generation.
- **Whole parts.** The copy holds every part from position 0. It has no parent. A later step can make a copy a delta
  of the context's last boundary entry.
- **Never a parent.** No entry names an interval copy as its parent. The store refuses such a put. Thus the cap and
  eviction can remove a copy without cutting a chain. Decided 2026-10-20 (reading, from the store fix step).
- **Host memory only.** The copy goes to the store's host tier, and the store makes room for it as for any put
  (6.9). It never goes to NVMe. A restart loses it (6.11).
- **The latest copy only.** When a new copy of a context is in the store, the scheduler removes that context's
  previous copy.
- **A cap and an order.** The `[store]` table of the profile has `interval_copies`, the most copies that the store
  keeps (2.4). Above the cap, the store removes the least recently used copy at once. When a put needs bytes, copies
  go first, the least recently used first (6.9).
- **How the store knows a copy.** An interval put names the sealed recipe at its position. The store computes the
  copy's recipe from it (6.2 tag 8) and refuses a put whose recipe differs. It also refuses an interval put to
  NVMe, a copy with a parent, and any interval put while the cap is 0.
- **Pins and the cap.** A pinned copy stays above the cap. When pins block the cap, the put fails with no space and
  changes nothing. `remove` takes only an interval copy, and it returns false for an id that is not in the store.
- **Owners.** Copies count toward their owner's quota in the host tier. Eviction step 1 takes the copies of the
  owner whose put needs bytes, and the cap counts the copies of every owner.
- **The same checks.** The tripwire (6.7) applies to copies as to other entries. A change of `interval_copies` is
  a new profile, so it restarts the process (2.3). An NVMe delta whose parent is in the host tier is refused,
  because a restart would cut its chain.
- **Any split replays.** An engine that declares a checkpoint grain must also declare `segments_exact`. Thus the row
  count of a pass does not change its bits (5.7), and a replay can use any split into passes.

**The scheduler's part.** Decided 2026-10-20 (reading, from a note by Sonnet 5.5 in milestone 12).

- **When a copy is due.** A copy is due after a quantum that moves a context across a multiple of the grain. The
  copy holds the state at the end of that quantum, so its position can be above the multiple. The scheduler does
  not split a quantum to end on the multiple. A split would put the copies of contexts with one prefix at one
  position, so they could share a copy. It costs more passes.
- **One try for each multiple.** The scheduler makes one attempt for each multiple that a context crosses. If the
  put fails (for example, pins block the cap), the scheduler waits for the next multiple. It does not retry at once,
  because each attempt reads about 112 MiB on mk.
- **The previous copy.** When the store holds the new copy, the scheduler removes the previous copy of that context.
  The scheduler does not remove a copy when its context ends. The cap and eviction remove it later.
- **Bad limits.** A grain of 0, or a grain on an engine without `segments_exact`, is an error.
  A cap of 0 means no copies, so the scheduler makes none.
- **Class and kind.** The copy is a quantum of the kind `copy` (5.2). It has the class of its job (see "The job
  path" below).

Reason: these are the smallest rules that keep the text above true.

**The job path.** Decided by senpai 2026-10-20 night (vetoable). The council jobs of milestone 12 step 5 make and
use the copies (`crates/nagaya/src/council/build.rs`):

- **A context is a slot.** The tracker of an engine keys each context by the slot of its job. A job tracks its slot
  from its start to its end. A rewind moves the rows back, so the job tracks the slot again from the new rows. The
  context keeps its copy through the rewind.
- **The copy is the next quantum.** The copy is a quantum of the kind `copy` in the class of the job. It runs before
  the job's other quanta, so it holds the state at the end of the pass (`Ran::First`).
- **No copy on a boundary.** When a pass of a build ends on a boundary, the snapshot of that boundary is the next
  quantum. That snapshot holds the state, so the job makes no copy there.
- **The recipe.** The sealed recipe at a copy's position chains from the last seal of the state (6.2). Its items are
  the recipe at the last seal, the tokens since that seal, and the position. A seal is a boundary snapshot or the
  head of a reply.
- **A build looks for a copy.** A build takes the live copies of its engine. It computes the recipe of a copy at each
  copy's position from its own tokens. It restores the longest copy that matches above its highest held boundary,
  and it feeds from there. A copy on a boundary position is never used. The build names the copy in `from`.
- **A skip is not an error.** A charge that the ledger refuses and a put with no space skip the copy, with a warning
  in the log. Any other error of the copy ends the job.
- **No client pin on a copy.** The pin route reports an interval copy as not held, and it pins nothing. A pin would
  stop the park (10.3) and the next copy from removing the copy. The store keeps its rules for a pinned copy, for
  other callers.

A restore from a copy writes its parts, then replays the tokens after it, with the same boundaries. The result has
the bits of the first run. The lookup is by recipe with tag 8, so any context with the same prefix can use a copy.
The copy serves three uses: a branch from an earlier position, a truncate, and a recovery after the live state is
lost. Amy, 2026-10-20: "I suspect we only want a few copies online and won't benefit as much as a commerical service
does, but it is cool and we'll do it."

For mk, the fixed parts of a copy are about 112 MiB. They are the GDN state in f32 (108 MiB) and the conv state
(4.3 MiB, mk `DESIGN.md`). The K/V, the index keys and the pooled keys add 55.5 KiB (56,832 bytes) for each
token. With a grain of 256 tokens, a decode makes a copy every 10-15 s.

On mk, the split of tokens into passes does not change bits. The probe `examples/split_bits.rs` in the mk crate
(`427fdc3`) compared one pass, one pass for each token, and uneven chunks, at 64 and 200 tokens. Every part and all
248,320 logits were equal. One pass of 200 tokens took 424 ms, and 200 passes of one token took 10,079 ms. Thus a
replay of up to 255 tokens takes about 0.5 s on mk.

**6.14 The feed rule.** Decided 2026-10-21 (Amy, engine-abi D11). An engine can declare a feed rule, `{block}`. The
scheduler feeds each span between boundaries in passes of `block` tokens from the start of the span. mkviolet
declares it, because its bits depend on the token count of a pass.

This is the reading of 6.2 for such an engine. A recipe names the identity, the tokens and the boundaries. The bits
of a state follow from them only when the rule made every row. Thus the store keeps a state under a recipe only when
the rule made each of its rows. These rules follow:

- **Decode.** A decode quantum feeds one token, or one token and its drafts. Its rows do not follow the rule. The job
  path never puts a state with decode rows under a recipe of 6.2. The park recipe (6.2) names bytes, so a park of
  such a state is still exact.
- **Drafts.** A draft test feeds one token and k drafts in one pass, with k + 1 logit rows (mkviolet
  `mk_api.rs:1187-1222`). The rows of a rejected draft stay past the token count, and the next pass writes over them.
  The logits fall under the tolerance of the served model.
- **Branch blocks.** A branch block is a read. It feeds the tail of a case and each branch in one pass with
  per-token `meta` (`branch_block`, `gemma4.rs:1521-1554`). No row of it is kept, also the rows of the tail. No build
  continues from the slot after it. The job restores a held snapshot, or feeds again from a boundary.
- **The bits of a branch.** On 2026-10-21, a branch in a branch block gave the logits of a plain pass of the prefix
  and that branch. The difference was 0 (`golden_branches_match_separate_passes`, mkviolet `golden.rs:264-287`).
  The rows stay unkept all the same, because the rule did not make them.
- **Reads.** The reads of a decision keep no state. mkviolet splits a case into chunks that it picks for speed
  (`council_server.rs:822-848`). The nagaya may do the same for a read on such an engine.
- **A continued span.** The last pass of a span can hold fewer than `block` tokens. Then a job that continues the
  span feeds that pass again (engine-abi D11 rule 7).

Decided 2026-10-21 (Amy, by interview): a generation on an engine with a feed rule holds no head after its reply.
Its `head` is null (11.3c). The reply rows come from decode quanta. The next build finds the last held boundary of the
prompt, and it feeds the reply by the rule. Reason: a head would need a second feed of the reply now. The next build
does that feed only when a client asks for it.

**The job path.** The council jobs take passes of `block` for an engine with a feed rule (`council/mod.rs`, the
`chunk` of a build). A build starts each span at a held boundary or at 0, and a generation holds no head. Thus no job
continues a span today, and rule 7 has no code yet. The tests are in `crates/nagaya/tests/council_feed.rs`.

**Proposed** (claude-opus-5-5, 2026-10-21): drafts are a setting of the served model, not a bit setting of the
engine. The nagaya's generation job makes the prompt-lookup drafts, as mkviolet does (`draft`, `mk_api.rs:336-347`,
keys of 4 and 3 tokens, at most 7 drafts). Reason: drafts change no row that the store keeps. Thus a change of the
setting must not change the identity and lose the store.

**Open** (Amy, 2026-10-22: "leave open for testing on b70host"): do drafts change the bits? A drafted verify pass
has a shape that a plain decode does not have. Gemma 4 declares `segments_exact: false`, and its feed rule (6.14)
exists because its bits depend on the split of a pass. The test runs on the B70 cards:

- Decode the same prompt greedily with drafts on and with drafts off. Compare the logits of each step, bit for bit.
- Compare the rows that the slot holds after each run, bit for bit.
- Equal bits accept the note above. Other bits make drafts a bit setting, or a rule that a verify pass follows.

gfx1151host cannot run this test. mk declares `segments_exact: true` (`crates/nagaya-mk-qwen38-gfx1151/src/spec.rs:463`),
so the test cannot fail there for this reason.

## 7. IO

**7.1 One IO service per machine.** Engines (through the `io` capability) and the store send IO requests
`(file, offset, len, dest)` with a class to the IO service. The service owns these jobs:

- the rate on each edge and the queue depth;
- hedged reads across copies on different drives;
- the loss of a drive, and its return.

An IO request holds a lease on its destination until its final wait ends. This is also true after an error.

Not built: the IO service of 7.1 to 7.3. The store does its own file IO.

**7.2 Direct and staged bytes.** The service counts direct bytes and staged bytes separately. Thus a change from
direct IO to staged IO is visible.

**7.3 Replicas.** A store on several NVMe drives names each copy by content id. "Same size is not same bytes". A
copy that has no verification gives no reads.

**7.4 Watchdog.** A drive that hangs can hold a quantum until the kernel times out (about 30 s in ds4-recipe).
The watchdog thread (3.2) reports each quantum that is past `max_quantum_ms` by a margin. It runs on its own thread,
because the group thread is blocked in the pass.

A device group that stops its heartbeat is a failure. The process exits with code 1, the crash code of 10.4, and
systemd restarts it. Reason: a stalled engine can hold a device, and the process stops before it continues in a bad
state (8.5).

A group with stages has one heartbeat for each stage thread, and the watchdog watches each one (engine-abi D1). A
stall of one stage thread is a stall of the group.

The timeout is 30 s, three times the longest declared quantum bound (mk: 10 s). It has no key in the
bootstrap config. The code takes it as a start parameter, so the tests use a short value.
Decided 2026-10-20 (reading).

**Open:** The group thread sends no beat while a quantum runs. On a slow NVMe, a copy quantum could run past the 30 s
timeout, and the watchdog would end the process. We have not measured it (kaibo review job-1, finding 4).

**The watchdog during an open.** An engine's open can take longer than the timeout: mk loaded in 24-50 s on
2026-10-20. A group beats its heartbeat while its engine opens, so a long open is not a stall. Amy, 2026-10-20: the
heartbeat during an open is part of milestone 14. The group thread is blocked in the open. Thus a helper thread sends
the beats, at the heartbeat interval, until the open ends. The helper stops when the group thread leaves the open
for any reason.

The beat hides a hung open from the watchdog. The wait for the opens is the backstop (10.1). It ends a hung open
after 300 s, and the process exits 1 (10.4). The 300 s equals `TimeoutStartSec` in the unit.

Decided by senpai 2026-10-20 night (vetoable): the wait for the opens is 300 s. Reason: mk waits up to 60 s for free
memory, then loads for about 35 s. The old wait of 60 s ended a slow open that was not hung.

## 8. Correctness at runtime

**8.1 Canary.** The nagaya records a canary for each engine: fixed inputs and the exact outputs that the engine gave
for them. The nagaya runs the canary through the engine, with its own sampler and decision code. mk's canary has
pilot reads and an 8-step greedy decode (mk `service/canary.py`). It keeps letter logits as f32 bit patterns, the
top 5, and a SHA-256 of all logits. The canary's hash is part of the engine identity.

The nagaya runs the canary again before the engine is ready, with no tolerance. At start, a mismatch makes the
nagaya exit with code 3, "do not restart" (10.4). After start, a mismatch marks the engine fatal (8.2), and the
process continues with its other engines.

Not built. The process runs no canary, and the mk binding gives an empty `canary_hash` (engine-abi "v0 mk binding
choices"). A start checks no canary.

**8.2 Pass failures.** A pass returns ok, poison (D6), or timeout. The nagaya treats outputs that are not finite as
poison. After a poison or a timeout, the nagaya does these steps:

1. Fail the jobs in that pass. Free their contexts, or mark their slots dirty until a job clears them (`slot_clear`),
   overwrites them from position 0, or writes them through `part_write`.
2. Run the canary again on the group thread, before any other work.
3. If the canary does not match, or if this is the engine's fourth timeout in the last 600 s, mark the engine fatal.
   Refuse every later job for it.

(mk `service/gpu.py` does this.)

**8.3 Conformance.** Each engine has a conformance suite in the nagaya's tests. It checks these properties:

- park and restore keep every bit;
- seal runs only at boundaries;
- the quantum conditions hold (ontology "Quantum");
- an admission refusal changes nothing;
- for an engine that declares `segments_exact`, a segment gives the same bits in passes with one, two and the
  maximum count of segments (5.7);
- for an engine with stages, `Engine::pass` and the stage path give the same outputs and parts (engine-abi D1);
- for an engine with a feed rule, a span fed by the rule after a restore gives the bits of one job (6.14).

**One oracle for each family.** Each model family has one reference: today, fixed outputs of mk's Python. Every
engine of that family passes its conformance against that reference, on any vendor's device. The reference runs on
the CPU, so it does not give the bits of a GPU. Thus the oracle has a tolerance, and a spike measures the gap first.

**The Qwen3.8 oracle.** mk's Python reference (`tools/oracle_fixture.py`) writes the fixture. It holds five prompts
of 5 to 596 tokens, with the top 64 logits and the logsumexp of the last position. The bench `oracle` in the mk
crate compares mk with it. mk's exact path (`MK_FAST_QMM=0`) is the gate. Its largest gap was a logit difference
of 1.7e-5 on 2026-10-20. Decided 2026-10-20 (reading).

- On the exact path, the argmax agrees when the reference's top two differ by more than 0.25. The reference's top 8
  are in the engine's top 16. The largest logit difference is at most 1e-3, the KL at most 1e-8, and the logsumexp
  difference at most 1e-3.
- The fast path (`MK_FAST_QMM` unset) is an approximation (12.1). It keeps only the argmax and top-8 rules, and the
  bench reports its other numbers. At 249 tokens its largest logit difference was 1.03.

The fixture covers prompts of up to 596 tokens. Past about 640 tokens, the gap between mk and the reference is not
explained (15 item 8). The oracle does not gate a longer prompt.

A family with no reference has no oracle yet: the first port of mkviolet builds the Gemma reference.

**The Gemma 4 oracle.** Decided 2026-10-22 (Amy: "accepted"). mkviolet's `tools/gemma4_reference.py` runs HF
transformers' own Gemma 4 layers in f32 on the CPU. `tools/gemma4_render_reference.py` renders the chat template with
Python's jinja2. The port makes the fixture with them, as mk's `tools/oracle_fixture.py` makes the Qwen3.8 fixture.

The tool is `tools/gemma4_oracle_fixture.py`, and the fixture is `crates/nagaya-text-gemma4/tests/fixtures/oracle.json`.
It holds five prompts of 6 to 586 tokens, in the format of the Qwen3.8 fixture. It took 94 s on b70host on
2026-10-21. The fixture belongs to the family, so each engine of Gemma 4 reads the same file.

- mkviolet measured these gaps (mkviolet `docs/README.md`). The full pass had a largest logit difference of 5.8e-5.
  A prompt of 1,270 tokens, past the sliding window of 1,024, had 7.2e-3 and the same top token.
- At 1,270 tokens, the f32 order of sums changed 123 of 38,100 top-8 expert picks. Each change began on a near tie in
  HF's own router.
- The gates are the argmax and top-8 rules of the Qwen3.8 oracle. The largest logit difference is at most 1e-2. The
  bench reports the other numbers. A spike measures the gap at longer prompts first. Decided 2026-10-20
(`docs/deepinfer.md` 6). The engine crates stay one for each vendor and device (Amy: "we generate a megakernel per
vendor/device and they get their own whole crate").

A gate that restores state restores it through `Store::restore`, the path that production uses. Thus a gate and
the store cannot drift apart. Decided 2026-10-20 (reading, kaibo job-2 finding 1).

The suite compares values by integer bit pattern. NaN or Inf fails, even when the bit patterns are equal. Counts
must also match. Each gate keeps a broken variant that it must fail (ds4-recipe `01-correctness-and-measurement.md`).

**8.4 Audit.** For some engines the tripwire is not reliable, because of their determinism class. For them, the
nagaya runs an audit after each start, as a filler job in the same process. The audit builds a sample of stored
recipes again and compares the content ids (D3). A restart is the fresh process that the audit needs, so the audit
needs no second GPU client.

Not built. The nagaya runs no audit yet.

**8.5 Crash over corruption.** An unexpected failure in the nagaya stops the process (exit 1). The process does not
continue in a bad state. An event that does not fit the scheduler's state, such as a cancel of an unknown job, is
an error that its caller sees. A cancel of a job that already ended is not an error, because a client can race the
end of a job. Decided 2026-10-20 (reading).

## 9. Observability

The nagaya follows kaijutsu's conventions (`~/src/kaijutsu/docs/telemetry.md`). mkviolet already follows them.

**9.1 Activation.** The nagaya always logs to stderr. It also exports in these cases:

- OTLP over gRPC to the machine's collector, when `OTEL_EXPORTER_OTLP_ENDPOINT` is set;
- OTLP JSON-lines files, when `NAGAYA_OTEL_FILE_DIR` or `KAIJUTSU_OTEL_FILE_DIR` is set.

`OTEL_SDK_DISABLED=true` stops all export. The `service.name` is `nagaya`. The resource attributes carry
`host.name` and the hardware from the inventory.

Built: the log, and the OTLP gRPC export of spans and metrics. Not built: the JSON-lines file export and the OTLP
export of logs. The nagaya warns at the start when `NAGAYA_OTEL_FILE_DIR` or `KAIJUTSU_OTEL_FILE_DIR` is set, so that
a missing export is never silent. The code is in `crates/nagaya-telemetry`.

Decided by senpai 2026-10-20 night (vetoable):

- A log line is the message, then the fields as `key=value`. A warning or an error ends with `level=warn` or
  `level=error`. A line has no time, no level prefix and no span name. The journal adds the time. The tests of the
  process read a line by the start of its message.
- `RUST_LOG` sets the log filter. The default is `info`. A filter that does not parse refuses the start (exit 3).
  The filter does not change the export.
- The nagaya starts the telemetry before it reads the config. Thus a config error reaches the log. The inventory
  comes first, because the resource carries the hardware.
- The resource attributes are `host.name`, `nagaya.device.count`, `nagaya.device.names`, `nagaya.device.memory_bytes`
  and `nagaya.host.memory_bytes`. The nagaya leaves out an attribute that it cannot read, and it warns.
- The process drops the telemetry before it exits, because `std::process::exit` runs no destructor. The drop waits at
  most 2 s for each signal. A collector that is gone does not stop the process.
- The exporters run on a Tokio runtime of their own with one worker. The telemetry owns it, and the nagaya does not
  leak it (compare mkviolet, section 14).

**9.2 Traces.** When a request has a W3C `traceparent` and `tracestate`, its span is a child of the caller's span.
The nagaya records an invalid header as an attribute, not an error. The spans nest in this order:

1. the HTTP server span, `<METHOD> <route template>`;
2. `nagaya.job`, with engine, class and request kind;
3. `nagaya.quantum` for each quantum, with engine, device group, kind, class, segments, and wait and run times in ms.

Decode quanta make metrics only: "one span per generated token would bury a generation's trace" (mk). The trace
context goes to the group thread with the job. The nagaya gives it to the engine through the telemetry capability,
and the engine adds its own spans under the quantum. Today, neither engine carries it onto GPU work.

Built: the spans `nagaya.process` (the run, with `exit_code`), `nagaya.start` and `nagaya.stop` (with `engines` and
`outcome`), `nagaya.job` and `nagaya.quantum`. The group thread opens the last two (`GroupJobs` in
`crates/nagaya/src/jobs.rs`, through `ObservedScheduler`). A failure marks its span as an error. The function
`adopt_parent` makes a span the child of the caller's span from `traceparent` and `tracestate`. It records an invalid
header in the attribute `nagaya.traceparent.invalid`.

The HTTP server span is built in `observe` (`crates/nagaya/src/http/mod.rs`). It calls `adopt_parent` for each
request.

The group thread records each quantum that it runs. `GroupJobs` holds an `ObservedScheduler` and steps it for every
event. The span `nagaya.job` has the attributes `engine`, `class` and `request`, and `outcome` when the job ends. It
is the parent of the quantum spans of the job. The body of a quantum runs inside the span of the quantum, so the spans
of the engine nest under it.

A new thread enters the telemetry of its starter (`Inherit`). Thus a test on its own thread reads what the group
thread records.

**Open:** the span `nagaya.job` is the child of the span that is current when the request submits the job
(`Ports::submit_job` reads `Span::current()`). The council routes call the service on the council's own thread
(`council.call` in `crates/nagaya/src/http/routes.rs`), and that thread does not carry the span of the request. Thus a
job from an HTTP request starts a trace of its own today. A test shows the parent when the caller's span is current
(`crates/nagaya/tests/quantum_telemetry.rs`).

Decided by senpai 2026-10-20 night (vetoable):

- The attribute `engine` of a group is the names of its engines in profile order, joined with `+`. A group with one
  engine has the name of the engine.
- The attribute `request` of `nagaya.job` has one of these values: `attach`, `build`, `dry_run`, `read`, `decision`,
  `generation` or `job`. A body names its value with `JobBody::request`. The value `job` is for a body that gives none.
- The job span ends when the job reaches a terminal state, or when a stop cancels it. Its `outcome` is the job state.

Decided by senpai 2026-10-20 night (vetoable). The route template is axum's, such as `/council/v2/pins/{owner}`. A
request that matches no route has the span `<METHOD>` alone, and the `http.route` value `unmatched`. The span has the
attributes `http.request.method`, `http.route`, `http.response.status_code` and `request_id`. A status of 500 or more
marks the span as an error. The span name has no `nagaya.` prefix, so it has the default rate of 9.3.

**9.3 Sampling.** The nagaya uses kaijutsu's rule. It always keeps errors and spans whose parent is sampled. It
keeps 100% of the spans in some dot-qualified prefixes and 10% of the others. `nagaya.` and `council.` are 100%
prefixes, so `nagaya.job` and `nagaya.quantum` are always kept. Decode quanta make no spans, so the volume stays low.

A prefix carries its dot, so a span named `nagaya_x` has the default rate. The error test reads the attributes at the
start of a span. Thus a span that fails later is kept only through its prefix or its parent.

**9.4 Metrics.** The nagaya keeps each metric attribute to a bounded set of values. Context, trace and job ids are
never attributes.

| Metric | Unit | Attributes |
|---|---|---|
| `nagaya.quantum.wait.duration`, `nagaya.quantum.run.duration` | ms | engine, device_group, kind, class |
| `nagaya.quantum.tokens` | {token} | engine, kind |
| `nagaya.ledger.bytes` | By | tier, engine, term |
| `nagaya.ledger.refusals` | {refusal} | tier, engine, gate (budget, floor, resize) |
| `nagaya.store.bytes`, `nagaya.store.entries` | By, {entry} | tier, engine |
| `nagaya.store.lookups` | {lookup} | engine, outcome (exact, approx, miss, refused) |
| `nagaya.store.evictions` | {entry} | tier, reason (evict, put_room, interval_cap, interval_replaced) |
| `nagaya.store.restores` | {restore} | engine, result (done, refused, failed, error) |
| `nagaya.io.bytes` | By | edge, path (direct, staged), class |
| `nagaya.arena.handoff` | {handoff} | engine, status (hot, cold) |
| `nagaya.engine.state` | 1 | engine, state (loading, ready, draining, unloaded, fatal) |
| `http.server.request.duration` | s | http.route, status |

Decided by senpai 2026-10-20 night (vetoable). These rules fix the labels of the metrics that exist:

- The attribute `device_group` holds the id of the device group. The attribute `tier` holds `host`, `nvme` or
  `device<id>`. The `<id>` is the machine device id (2.4, "One name for each device").
  A budget that covers several tiers has the sorted tiers joined with `+`, for example `device0+host`.
- The wait of a quantum runs from the moment its lead segment became ready to the moment the quantum started. The lead
  segment is the first one. A job is ready when it arrives, and again when a quantum of it ends and more remain. A job
  that the ledger has not admitted yet is ready, so its wait includes the time until the admission.
- `nagaya.ledger.bytes` is a gauge with one point for each engine term. It holds the bytes of the last admission. A term
  of the lifetime `context` has no point, because the ledger charges it at `ctx_new`.
- `nagaya.ledger.refusals` counts one refusal for each short budget. The engine is `all` for a refused profile, and the
  engine name for a refused context. The ledger has the gate `budget` only. The gates `floor` and `resize` come with the
  code that refuses at them.
- `nagaya.store.bytes` and `nagaya.store.entries` are gauges. The store reports them after `open`, `put`, `evict` and
  `remove`. A series that empties reports zero once. The engine is the name in the entry.
- `nagaya.store.lookups` counts every `lookup`. The engine is the model name of the engine that asks, because the store
  does not know the profile name. An outcome is `approx` when an entry of the chain has a provenance. A lookup that
  returns an error counts as `refused`.
- This table adds `nagaya.store.restores`. It counts every `restore`. The engine is the model name. The result
  `error` means that the store refused before any write.
- `nagaya.engine.state` is a gauge. A change of state records one for the new state and zero for the other four.

Each GPU vendor has a sensor provider: amdgpu (the SMU `gpu_metrics` table and KFD counters), Intel Xe, and NVIDIA
NVML. Each provider maps its values onto shared names:

| Metric | Unit | Attributes |
|---|---|---|
| `nagaya.device.clock` | MHz | device, clock |
| `nagaya.device.power` | mW | device, rail |
| `nagaya.device.temperature` | Cel | device, sensor |
| `nagaya.device.activity` | % | device |
| `nagaya.device.memory.used` | By | device, domain |
| `nagaya.device.throttle` | {event} | device, reason |
| `nagaya.device.evicted` | ms | device (amdgpu KFD only) |

**The Xe provider.** Decided 2026-10-22 (Amy: "accepted"). Each card has these files under
`/sys/class/drm/card<N>/device`, as the b70host tree shows:

- clock: `tile0/gt<N>/freq0/act_freq` and `cur_freq`;
- power: the change of `hwmon/hwmon*/energy1_input` over time, and `power1_cap`;
- temperature: `hwmon/hwmon*/temp*_input`, with each `temp*_label` as the sensor;
- activity: the change of `tile0/gt<N>/gtidle/idle_residency_ms` over time;
- throttle: `tile0/gt<N>/freq0/throttle/reason_*`, with the reason in the file name.

`nagaya.device.evicted` stays amdgpu only.

mk's `mk.gpu.*` names and mkviolet's `mkviolet.*` names map onto these names. Council decision metrics (count,
spread, confidence) belong to the council layer (11.1).

**9.5 Each instrument has a call site under test.** mkviolet defines `record_gpu_pass` but calls it only from its
smoke test. It calls `record_council_decision` only in tests. Its docs name instruments for routing entropy and
J-Lens drift that do not exist. Thus a test shows that each instrument records on the path that it describes.

A test runs the real call site and reads the points through an in-memory exporter (`nagaya_telemetry::testing`). The
code defines an instrument in the commit that adds its call site. The table shows each instrument that exists.

| Instrument | Call site | Test |
|---|---|---|
| `nagaya.quantum.wait.duration`, `nagaya.quantum.run.duration` | `ObservedScheduler::step`, called by `GroupJobs::run_one` on the group thread (`crates/nagaya-sched/src/observed.rs`, `crates/nagaya/src/jobs.rs`) | `crates/nagaya-sched/tests/observed.rs`, `crates/nagaya/tests/quantum_telemetry.rs` |
| `nagaya.ledger.bytes`, `nagaya.ledger.refusals` | `Ledger::admit` and `Ledger::admit_context` in `crates/nagaya-ledger/src/lib.rs` | `crates/nagaya-ledger/tests/telemetry.rs` |
| `nagaya.store.bytes`, `nagaya.store.entries`, `nagaya.store.lookups`, `nagaya.store.evictions`, `nagaya.store.restores` | `Store::open`, `put`, `evict`, `remove`, `lookup` and `restore` in `crates/nagaya-store/src/lib.rs` | `crates/nagaya-store/tests/telemetry.rs` |
| `nagaya.engine.state` | `lifecycle::start` and `lifecycle::stop` in `crates/nagaya/src/lifecycle.rs` | `crates/nagaya/tests/telemetry.rs` |
| `nagaya.arena.handoff` | `lifecycle::start` in `crates/nagaya/src/lifecycle.rs`, from the handoff of each opened engine | `crates/nagaya/tests/arena_start.rs` |
| `http.server.request.duration` | `observe` in `crates/nagaya/src/http/mod.rs` | `crates/nagaya/tests/http_telemetry.rs` |

These instruments have no definition yet, because they have no call site: `nagaya.quantum.tokens`, `nagaya.io.bytes`
and the device sensors. Each one comes with its call site.

The first row has its call site on the path that it describes. The test `quantum_telemetry` runs a council build and a
generation on the fake engine through the process. It reads the spans and both durations from memory.

`http.server.request.duration` runs from the start of a request to its response head. Its `status` is the HTTP status
as a number. A stream ends later than its response head, so the point does not cover the reply. Decided by senpai
2026-10-20 night (vetoable).

**9.6 Evidence on failure.** Before and after a quantum that fails, the nagaya reads the machine's counters. It logs
the change with the failure. mk reads these counters: KFD evicted ms, MemAvailable, GTT, and from `/proc/vmstat`
compaction stalls, migrations, swap-outs and direct scans.

Built as `nagaya::evidence` in `crates/nagaya/src/evidence.rs`, with fixed trees in the tests. The code that runs a
quantum calls `Machine::watch` before the quantum and `Watch::failed` in its error branch. `GroupJobs::run_one` does
this on the group thread. The tests are `crates/nagaya/tests/job_evidence.rs` and
`crates/nagaya/tests/process_evidence.rs`.
A read costs four files, so the group thread skips `watch` for decode quanta.

Decided by senpai 2026-10-20 night (vetoable). A decode quantum that fails still logs the failure. The line says
`evidence unavailable (not read before a decode quantum)` (`evidence::failed_unwatched`). The failure text names the
kind of the quantum, the kind of request and the engine. `Start::machine` gives the process a machine with a fixed tree
of files for a test.

Evidence never hides its failure. A counter that cannot be read, or that went back, is named in the line as
`evidence unavailable`. The failure is logged all the same. The sources are `/proc/meminfo`, `/proc/vmstat`,
`/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms` and `/sys/class/drm/card*/device/mem_info_gtt_used`. The GTT is
summed over the devices, and it is absent on a machine with no amdgpu device.

**Evidence on Xe.** Decided 2026-10-22 (Amy: "accepted"). Xe has no counter of evicted ms, and no GTT in the
amdgpu sense. Thus a Xe machine reads these sources in their place:

- the memory of the nagaya's own DRM clients: the `drm-total-*` and `drm-resident-*` keys in `/proc/self/fdinfo/<fd>`
  of each `xe` fd;
- the throttle state of each GT: `device/tile0/gt<N>/freq0/throttle/status` and `reasons`;
- `MemAvailable` and `/proc/vmstat`, as on any machine.

The exact fdinfo keys need a check on b70host before the code reads them. A source that is absent is
`evidence unavailable`, as above.

**9.7 Status feed.** Each machine has one SSE status feed (mk `/mk/v1/status`, mkviolet `PassBus`):

- a snapshot when a client connects;
- one `quantum` event for each quantum, with engine, device group, kind, ms, and queue length by class;
- a keepalive every 15 s.

The feed disconnects a slow subscriber. It does not skip events for it. A subscriber that wants less volume asks for
a filter by engine or kind when it connects.

Not built. The HTTP server exists (11.7), and it has no status route. `ObservedScheduler` sees every quantum, and
the job path uses it now (9.2), so it can feed the bus.

**9.7a The events.** The route is `GET /nagaya/v1/status` (11.4a). The feed is `text/event-stream`. Each event has an
`event:` line and a `data:` line of one JSON object. These rules are **Proposed (for Amy)**, by Claude Sonnet 5.5 on
2026-10-21.

| Event | When | `data` |
|---|---|---|
| `snapshot` | First, once, when a client connects. | The state of the machine, below. |
| `quantum` | After each quantum that a group thread runs. | The quantum, below. |

The keepalive is a comment line, `: keepalive`. It is not an event.

The `data` of `snapshot`:

| Member | Type | Meaning |
|---|---|---|
| `t_ms` | integer | Milliseconds since the process started. |
| `machine` | string | The `host.name` resource attribute (9.1). |
| `engines` | array | One entry for each engine group on the machine. |
| `engines[].engine` | string | The `engine` attribute of 9.2: the engine names of the group, joined with `+`. |
| `engines[].device_group` | integer | The id of the device group. |
| `engines[].state` | string | `loading`, `ready`, `draining`, `unloaded` or `fatal`. It is `EngineState` of `nagaya-telemetry`. |
| `engines[].queue` | object | The jobs that wait, by class: `interactive`, `batch` and `filler`. |
| `engines[].jobs` | integer | The jobs that have not ended, with the one that runs. |

The `data` of `quantum`:

| Member | Type | Meaning |
|---|---|---|
| `t_ms` | integer | Milliseconds since the process started, at the end of the quantum. |
| `engine` | string | As in the snapshot. |
| `device_group` | integer | As in the snapshot. |
| `kind` | string | `read`, `prefill`, `decode`, `embed`, `copy`, `step` or `train`. |
| `class` | string | `interactive`, `batch` or `filler`. |
| `request` | string | The kind of request: the `request` attribute of `nagaya.job` (9.2). |
| `ms` | integer | The run time of the quantum. It is the value of `nagaya.quantum.run.duration` (9.4). |
| `outcome` | string | `done` or `failed`. It uses the words of the `outcome` of the span `nagaya.quantum`. |
| `queue` | object | As in the snapshot. It counts the jobs that are `Ready` when the quantum ends. |
| `evidence` | string | Only when `outcome` is `failed`. It is the line that 9.6 logs, with the counters. |

```
event: snapshot
data: {"t_ms": 91234, "machine": "gfx1151host", "engines": [{"engine": "mk-qwen38-gfx1151", "device_group": 0,
  "state": "ready", "queue": {"interactive": 0, "batch": 0, "filler": 0}, "jobs": 0}]}

event: quantum
data: {"t_ms": 91790, "engine": "mk-qwen38-gfx1151", "device_group": 0, "kind": "prefill", "class": "interactive",
  "request": "generation", "ms": 412, "outcome": "done", "queue": {"interactive": 1, "batch": 0, "filler": 0}}

: keepalive
```

In a real event, the `data` is one line.

- **Proposed (for Amy).** A decode quantum makes an event too, as mk's `pass` does. Decode makes metrics only in 9.2,
  but a feed is for a person, and a client that wants less volume filters by `kind`.
- **Proposed (for Amy).** The events of one group are in order. The events of two groups have no order, and `t_ms`
  tells the client which came first.
- **Proposed (for Amy).** A filter is the query `engine` and the query `kind` of the connect request. Each takes
  values joined with commas. A name that is not an engine of the machine, or a kind that does not exist, is a 400
  `invalid_request` that names the query. An empty result from a typo would hide the fault.
- **Proposed (for Amy).** The `kind` filter drops quantum events only. The `engine` filter also drops entries of the
  snapshot.
- **Proposed (for Amy).** The keepalive comes every 15 s without an event, as in mk. A machine with no engine sends a
  snapshot with an empty `engines`, and then the keepalive.
- **Proposed (for Amy).** At most 16 feeds are open at once. A request past that is a 429 `busy`. A feed does not
  count toward the 64 calls of 11.7, because it holds no call thread.
- **Proposed (for Amy).** A stop ends each feed after the `draining` state (10.3). A client reconnects and gets a new
  snapshot.

**9.7b Where the events come from.** **Proposed (for Amy).** The group thread produces them, and a bus delivers them.
The code from the obs-wire merge (532eb28) already holds each value.

- A new type `StatusBus` in `crates/nagaya/src/status.rs` holds the subscribers and the last state of each group. The
  process makes one bus and gives it to the council and to each group.
- `GroupJobs::run_one` (`crates/nagaya/src/jobs.rs`) sends the `quantum` event after the outcome is known. A builder
  method `GroupJobs::status`, next to `GroupJobs::machine`, gives the group its bus. `GroupJobs::add_under` and
  `GroupJobs::cancel_all` update the state of the group, because they change the queue.
- `engine` and `device_group` are fields of `GroupJobs`. `request` is `JobBody::request`. `kind` and `class` come from
  the `Quantum` in `ObservedScheduler::state().running`, which `run_one` already clones. The function `kind_label` in
  `crates/nagaya-sched/src/observed.rs` becomes public, so that spans and events use one spelling.
- `ms` is the `duration_ms` that `run_one` measures. It is the value that `ObservedScheduler::step` records as
  `nagaya.quantum.run.duration` when it hears the `Complete` event. Thus the feed and the metric agree.
- `queue` counts the jobs of `ObservedScheduler::state` whose state is `Ready`, by `Class`. `jobs` counts the jobs that
  are not terminal.
- `evidence` is the line that `Watch::failed` returns (`crates/nagaya/src/evidence.rs`). For a decode quantum it is
  the line of `failed_unwatched`. Both functions already give the line, so no new read of the counters happens.
- The `state` of an engine comes from the call that records `nagaya.engine.state` in
  `crates/nagaya/src/lifecycle.rs`. That call also updates the bus.
- Each subscriber has a `tokio::sync::mpsc` channel for 256 events. The bus sends with `try_send` from the group
  thread, so a slow subscriber never blocks a group. A full channel removes the subscriber, and its stream ends. The
  nagaya logs `status feed dropped a slow subscriber` as a warning.
- The route is a handler in `crates/nagaya/src/http/extensions.rs`. It subscribes before it reads the snapshot, so no
  quantum falls between them (as mk's `gpu.subscribe` does).

Tests that can fail: a fake engine runs a build and a failing quantum, and the test reads the events from a
subscriber. It checks each member of 9.7a against the metric and the span. A subscriber that never reads must be
dropped, and the group must go on.

Other option for the source: emit from inside `ObservedScheduler::record_end`, so that no caller can forget an event.
That type then needs a sink, and it does not know the request kind or the evidence. The pick keeps the scheduler a pure
recorder of metrics and spans.

## 10. Lifecycle

**10.1 systemd.** The unit is `Type=notify`. The process sends `READY=1` when every engine in the profile has
opened, passed its canary, and woken its pinned state. `STATUS=` gives the state of each engine.

**One unit file for each machine.** Decided 2026-10-22 (Amy: "all good"). gfx1151host's unit stays at
`deploy/nagaya.service`. b70host's unit is `deploy/b70host/nagaya.service`, next to its config and profile.
Reason: the units differ in the engine variables, the cargo feature of the build and the OTLP line. One shared unit
needs a drop-in file on each machine, and a drop-in is not in the repository.

When `<store_dir>/profile.toml` does not exist, the process starts with an empty profile. An empty profile is valid,
because a machine with no engine can serve as a coordinator or hold a store tier (2.1). A process with an empty
profile sends `READY=1` when admission passes. Decided 2026-10-20 (reading).

Not built: the canary and the wake of pinned state (8.1, 6.11). Today `READY=1` follows the open of every engine.

While the engines open, the process tells systemd that the start goes on. Every 5 s (the heartbeat interval), it
sends one datagram. The datagram holds `EXTEND_TIMEOUT_USEC=` and a `STATUS=` line that says `loading` for each
engine. The extension is 30 s, the watchdog timeout (7.4). systemd sets the start deadline to 30 s after each
datagram, so a slow open does not time the start out. A process that stops sending the datagram times out within 30 s.

Decided by senpai 2026-10-20 night (vetoable).

When the wait for the opens ends (7.4), the process sends `STATUS=` with the name of each engine that did not open.
Then it exits 1.

**10.2 Arenas through a restart.** systemd's file-descriptor store keeps the arenas (`FDSTORE=1`). Each fd has the
name `<engine>/<device>/<part>`. systemd reads `FileDescriptorStoreMax` from the unit file before the process starts.
The unit file sets a generous ceiling (`FileDescriptorStoreMax=512`), and profile admission (2.4) refuses a profile
that needs more arena fds than the ceiling. (mkviolet keeps 30 slab fds under a limit of 64,
`systemd/mkviolet.service`.)

At start, the nagaya does these steps for each engine:

1. Check each stored arena against the engine's build, and record the strength of the check.
2. If an arena does not match, make sure that a cold load fits the ledger first (2.4). Then drop that engine's
   stored fds and load it cold. If a cold load does not fit, keep the fds and exit 3.
3. Duplicate each fd before use. (In mkviolet on 2026-10-16, an import closed the stored fd and caused a crash
   loop.)

`systemctl stop` clears the store. A restart keeps it.

**The names of the fds.** Decided 2026-10-22 (Amy: "accepted").

- `<engine>` is the engine name in the profile. `<device>` is `device<id>`, with the machine device id (2.4).
  `<part>` is a name that the engine gives in `arena_export`.
- mkviolet's parts are `w<ll>` for the slab of layer `ll`, and `head` for the head on card 1. Thus its arena has 31
  fds, such as `gemma/device1/w15`.
- mkviolet stores no head today, so a restart uploads the head again. The head is about 1.5 GB: 262,144 rows of 2,816
  BF16 values. The port gives it a slab.
- The engine opens its stored fds through `Factory::attach` (engine-abi "The factory gets its devices"). The check is
  `spot`: the first and last 64 bytes of each tensor (`verify_stored_slab`, `gemma4.rs:287-307`).

**The ceiling and the start steps.** Decided 2026-10-22 (Amy: "all good").

- The process reads the ceiling from `$FDSTORE`. systemd sets it to `FileDescriptorStoreMax` (systemd.exec(5),
  version 254 and later).
- When `$FDSTORE` is unset or 0, the process keeps no arena. It sends no `FDSTORE=1`, and admission counts no fd.
  systemd ignores `FDSTORE=1` from such a unit (sd_notify(3)).
- Each factory gives the names of its arena fds before the open, with `arena_parts` (engine-abi "Arena calls in
  v0"). Admission adds the counts of the profile. A sum above the ceiling refuses the profile, and the process exits 3.
- The process reads `$LISTEN_PID`, `$LISTEN_FDS` and `$LISTEN_FDNAMES` once, before it starts a thread. It obeys
  sd_listen_fds(3). The fds start at 3, `$LISTEN_PID` must be its own pid, and it removes the variables after it
  reads them.
- When `$LISTEN_PIDFDID` is set, it must be the inode of the pidfd of the process. Else the process uses no fd.
- A cold load fits the ledger when the admission of the profile passes (step 2). Thus the process drops no fd before
  admission passes. A refused profile exits 3 and keeps each fd.
- After admission, the process drops each fd whose name breaks the name rule or names an engine that is not in the
  profile. It logs one warning for each. Reason: such an fd holds device memory, and no engine can use it.
- The fds of an engine go to `attach` only when their names equal the names of its `arena_parts`. Else the process
  drops them and loads the engine cold.
- After a cold load, the process exports the arena and sends each fd with `FDSTORE=1` and its `FDNAME=`. After an
  attach, it sends nothing, because systemd still holds the fds.
- `FDNAME=` takes ASCII with no control character and no `:`, at most 255 characters (sd_notify(3)). Thus the `/` of
  the name rule is legal.

Reason: the unit file is the one source of the ceiling. A bootstrap key can differ from the unit.

mkviolet restarts in 0.9 to 1.2 s from its stored slabs, and a cold start takes 7.4 s (mkviolet `docs/README.md`).
Thus 10.2 moves into the b70host milestone (`AGENTS.md`), because mkviolet has it now and the port must not lose
it.

Built: the process side (`crates/nagaya/src/arena.rs`, `crates/nagaya/src/process/handoff.rs`) and the arena of the
Gemma 4 engine, with its head slab (`Loaded::attach`). On b70host on 2026-10-21, a cold `Loaded::load` took 7.1 s,
and an attach from the 31 slabs took 1.2 s (`tests/xe_arena.rs`, release build). A process restart took 7.1 s to
`READY=1` cold and 1.2 s hot (`crates/nagaya/tests/xe_restart.rs`). The b70host unit sets
`FileDescriptorStoreMax=512` (`deploy/b70host/nagaya.service`). The mk engine on gfx1151host exports no arena yet.

**10.3 Stop.** On SIGTERM, the nagaya does these steps:

1. Accept no new jobs.
2. Let each running quantum go to its end.
3. Park pinned and wanted contexts to NVMe in a time budget (ds4-recipe: 10 s). The budget is in the bootstrap config
   (2.1).
4. Exit.

A second signal makes the process exit at once. Cancellation of running work follows 5.9. **Open:** the conversion
to canonical safetensors (D10) is not yet measured, so we do not know how many contexts fit in the budget.

**Queued quanta and the park budget.** A stop cancels each queued quantum that has not started, and the submitter
gets a cancelled result (5.9). Only the running quantum goes to its end. The park budget of a group starts when the
drain of that group ends. A drain that does not end is a stall, and the process exits 1 (7.4). When a second signal
interrupts a stop, the process exits 1, unless every state was parked. Decided 2026-10-20.

A park that runs past the park budget makes the process exit 1. A SIGTERM during a planned restart stops the
process with exit 0. Decided 2026-10-20 (reading).

**Which states the park keeps.** Decided by senpai 2026-10-20 night (vetoable).

- The park stores each state that holds tokens. It skips a state with no token, because that state has nothing to
  keep.
- The park reads the row count of a state with `slot_high` (engine-abi "Capabilities"). A count of 0 means no token.
- An engine with a cells or ring part needs the `slots` capability. Without it, the park cannot count the rows, so
  the stop fails and the process exits 1.
- An engine with no cells or ring part has no row count. The park keeps each of its states.

Reason: step 3 parks pinned and wanted contexts. The process does not track pins or wants on live states yet, so
it parks each state with tokens. An empty state of mk is 118,038,528 bytes of fixed parts ("v0 mk binding choices").
A park of mk's 16 empty states would write about 1.9 GB and use most of the 10 s budget.

**A park of bytes that the store holds.** Decided by senpai 2026-10-20 night (vetoable).

- The index has one entry for each content id, so a content id has one recipe. The store refuses a put that gives a
  held content id another recipe. That refusal is not the tripwire, which is two content ids under one recipe (6.7).
- A slot can hold the bytes of a held entry, for example the head that a build put in the host tier (11.3c). Then
  the park does not put. It parks that entry to NVMe (`Store::park`), and the entry keeps its recipe, owner and reason.
- Thus a lookup after a restart finds the parked state by the recipe of the build, and the build restores it.
- A slot can hold the bytes of an interval copy. A copy never goes to NVMe (6.13). Thus the park removes the copy, and
  it puts the state under the park recipe.
- An entry that is on NVMe already stays as it is.
- A park of an entry makes room on NVMe as a put does (6.9). A delta parks only when its parent is on NVMe.
- A quarantined entry with the same bytes makes the park fail, and the process exits 1.

Reason: on gfx1151 on 2026-10-21 at 00:44, the park of one mk state failed with "the content id already has another
recipe". The stop exited 1. The recipe of a build names the inputs, so it is a better key than the park recipe.

**Open:** the park's copy of a state in host memory has no staging charge (4.2 "Staging"). The park holds one copy
at a time, on the group thread. A copy of a 32,768-token mk context is about 1.93 GB. The charge path
(`Ports::admit_staging`) refuses while the process stops, and the group thread holds no `Ports`. A charge for the park
needs a ledger path that the stop keeps open.

The process catches SIGTERM with a handler that adds one to a static atomic counter. The handler is `signal(2)`,
declared in `extern "C"`, because std has no signal API. The main loop reads the counter every 50 ms. Reason: this
needs no new dependency. A later step can use a signal stream from `tokio` instead. Decided 2026-10-20 (reading).

**10.4 Exit codes.**

| Code | Meaning | systemd |
|---|---|---|
| 0 | a clean stop | no restart |
| 1 | a crash | restart |
| 3 | a refusal to start: a bad bootstrap config (2.1), a probe error (2.1), a profile that does not fit, a canary mismatch at start, a mismatch of a pinned identity (12.1), of a part size or of the weight arena size, a store that does not open, a gossip bind or an HTTP bind that fails, or a council service that does not attach (11.7) | `RestartPreventExitStatus=3` |
| 4 | a planned restart: a new profile (2.3, 3.3) | `RestartForceExitStatus=4` |

The restart policy follows mk's unit: `Restart=on-failure`, `RestartSec=30`, and at most 3 starts in 600 s.
systemd counts each start against that limit, also a start after code 4. Thus a fourth profile swap in 600 s fails.

An engine that does not open at start stops the process. A mismatch of a pinned identity (12.1), a part size or the
weight arena size is a refusal to start: exit 3. Any other failure to open is a crash. The process exits 1, and
systemd restarts it within its start limit. A message on stderr names the engine and the reason. Decided 2026-10-20
(reading).

The library pin was a refusal to start too. It is gone, because the binary links mk's engine. Amy decided this on
2026-10-22 (AGENTS.md milestone 15).

**One exit path.** Decided by senpai 2026-10-20 night (vetoable).

- The binary ends through one function, `nagaya::exit::leave`, for every code.
- First the process drops the telemetry, so the OTLP batches flush (9.1). Then `leave` flushes stdout, stderr and the
  C output streams.
- Then `leave` calls `_exit(2)`. It runs no `atexit` handler and no static destructor.
- The fake build takes the same path, so the host tests run the path of the GPU build.

Reason: mk's HIP runtime stops the program in its static destructors, as its teardown does (3.3). On gfx1151 on
2026-10-20, a clean stop with `exit(3)` printed "pure virtual method called" and ended with SIGABRT (status 134) and
a core dump. Nothing in the process needs an `atexit` handler.

## 11. Client API

**11.1 Council and coordinator routing.** The nagaya serves council v2 for all engines in the cluster. It does not
serve v1, because no client speaks v1. The contract is gogi (`~/src/gogi`): `docs/council-api.md`, the types in
`gogi-proto`, and the suite in `gogi-conformance`. mkviolet and kaijutsu move
to the same gogi before the nagaya serves them (Amy, 2026-10-18).

Any machine can receive any request and act as its coordinator (Cassandra style). Clients prefer a coordinator with
separate host memory (`b70host` or a CPU rig) to save unified memory bandwidth on `gfx1151host` and `sm121host` (3.2). The
coordinator finds the target engine from the request's `model` field and the `chitchat` cluster state (2.3). Because
the coordinator has the family libraries (11.3) and the engine's `identity()` from gossip, it can render, tokenize
and compute the recipe locally. It then schedules the job locally or forwards the work to the machine that holds the
engine.

Routing to the machine that holds the deepest chain is deferred. Amy, 2026-10-20: "it's more likely folks like me
and my friend Gavin have a heterogeneous fleet, and will typically want sticky sessions with each model/gpu rig,
and it's okay those can't move around easily" (`docs/deepinfer.md` 7).

The API has contexts, specs, decisions, snapshots, pins, reads, and generations with riding decisions. In council v2
responses, a snapshot `id` is its BLAKE3 content id (`blake3:<hex>`), and the store looks up prefixes by recipe
(6.6). gogi sets these request rules:

- Each request names a served model (ontology) in `model`. A request without it is a 400.
- `identity.models` lists every served model in the cluster, with its status from gossip (2.3).
- The statuses are `ready`, `loading`, `draining` and `unloaded`. A fatal engine's served models report `unloaded`.
- A served model that is not `ready` gets 503 `model_unavailable`.
- `class` is `interactive` (the default) or `batch` (5.3). `priority` is an `i32`, 0 by default (5.4).
- A request carries no tolerance. Its served model sets the tolerance (6.6).

**Capabilities.** Milestone 12 declares `reads`, `generate`, `dry_run` and `leave_one_out` for each served model. It
declares `pins` and `classes` for the server. Decided 2026-10-20 (Amy chose the core set). `describe`, `think`,
`ride` and `tiers` come in a later milestone. A request that needs one of them is a 400 that names it (gogi
"Errors").

Pooling and the decision numbers come from `gogi_proto::math`, which all engines share. The council
`identity.engine` covers two things: the engine identity, and the nagaya code that turns logits into decisions.
(mk: `libmkengine:<sha>+numbers:<sha>`.)

**11.2 Generation.** The nagaya runs one sampler for all engines. It applies top_k, top_p, min_p, then temperature.
The draw is seeded, and the response reports the seed. Each engine has its own defaults. All engines share the SSE
framing.

**The draw keeps no state.** The random value for a token depends on three things: the seed, the position of the
token, and the purpose of the draw. The purposes are a sample, a resample and the test of a draft.
The sampler keeps no random state between tokens. A fork gets a child seed, a hash of the parent's seed and a branch
number. Decided 2026-10-20 (`docs/deepinfer.md` 1).

Reason: a park, a restore or a fork then keeps the sampler's bits, with nothing to save. A client that sends a
request again with the same seed gets the same tokens. A speculative draft (mkviolet's prompt-lookup drafts) has
its own purpose, so it never moves a later draw. Thus the reported seed does not change when drafts are on.

**The draw function.** These rules give the bits. Decided 2026-10-20 (reading).

- The position is the token's position in the context, from 0. It is not its index in the reply.
- The purposes are numbers: a sample is 0, a resample is 1, and the test of a draft is 2.
- The value is the BLAKE3 hash of `nagaya draw v1`, then the seed, the position and the purpose. The seed and the
  position are 8 bytes each, little-endian. The purpose is 1 byte.
- The uniform value is the first 8 bytes of the hash, little-endian, shifted right by 11 and times 2^-53. It is in
  [0, 1).
- A child seed is the first 8 bytes of a BLAKE3 hash, little-endian. The hash is of `nagaya fork v1`, the parent
  seed and the branch number, each 8 bytes little-endian.
- The pick is mk's inverse CDF: the survivors in order of logit down and id up, in float64.

Thus mk's own uniform value, put into the nagaya's pick, gives mk's token. The sampler's tests use that.

mk's sampler draws from one PCG64 stream for each request (mk `service/sampling.py:106`, `:153`). The nagaya does
not match those bits. It matches mk's filter order and its probabilities (milestone 12 step 2).

**11.3 Text front-end.** Each model family has its own text rules. They are a tokenizer, a template, escapes for
special tokens in client text, and parsers for tool calls and thinking. The nagaya holds one library per family
(ontology "Family"), and the engine identity selects it by its `tokenizer_hash` and `template`. Engines never see
text. They take tokens, rows, or both (D2).

Decided 2026-10-17 (Amy: "oh, yes, b for sure, plus that can support the tokenize apis nicely for when we bring back
kaijutsu's semantic search and stuff"). Thus one renderer serves every engine of a family, on any vendor's card.
The rendering rules in gogi, such as where a boundary exists and where thinking renders, get one implementation and
one set of conformance tests. mk's C ABI does not need a tokenizer.

**The Qwen3.8 library.** The library follows these rules. Decided 2026-10-20 (Amy accepted the note from
milestones 11 and 11b).

- The library reads the template from the GGUF key `tokenizer.chat_template` and runs it with a Jinja interpreter.
  It does not hold a hand-written copy. The template is part of the identity, so a new template in a GGUF needs no
  new code.
- The interpreter matches mk's Python on four points. `trim` strips what Python's `str.strip` strips, which includes
  U+001C to U+001F. `tojson` writes what Python's `json.dumps` writes, with `ensure_ascii` off. A map
  key that is an integer, a float, a bool or none becomes a string as in Python.
  A map keeps its key order.
- A value in `{{ }}` and the `string` filter follow Python's `str`. Thus none is `None` and a bool is `True`. A float
  is its `repr`, and a string in a list has quotes.
- A render refuses the same input as mk. A template error is an error. A message that holds a Unicode noncharacter is
  an error, because mk keeps those for its escapes (`harness/tokenizer.py`, `RESERVED`). The error names the
  character and its offset in characters, as mk does. A message of `raise_exception` reaches the error.
- mk lets some Python exceptions pass, such as a `TypeError`. The library gives a template error for those.
- **String methods.** The engine has no Python methods. The library holds these methods of `str`, with the results of
  Python 3.14:
  - `startswith` and `endswith`, also with a tuple, `start` and `end`;
  - `strip`, `lstrip` and `rstrip`, with optional characters;
  - `lower`, `upper`, `title` and `capitalize`;
  - `replace`, `split` (with `sep` and `maxsplit`, also by name) and `join`;
  - `count`, `find`, `removeprefix` and `removesuffix`;
  - `isdigit`, `isalpha` and `isspace`.

  The real template calls `startswith` and `endswith` only. A template that needs another method fails when it
  renders. The error names the method and lists the methods that exist. The Unicode tables come from Python 3.14.8
  (Unicode 16.0.0), because the tables of Rust are of another version.
- **Functions and filters.** `trim` takes the optional characters. `raise_exception` takes any value, by position or
  as `msg=`, and writes it as Python's `str` does. `length` and `items` accept the undefined value.
- **Tests and the undefined value.** The tests `sequence`, `number`, `string`, `mapping`, `iterable`, `defined`,
  `undefined`, `none`, `true`, `false`, `boolean`, `integer`, `float` and `callable` give the answers of jinja2.
  A string is a sequence, and a bool is a number. The undefined value writes nothing, it has no items, and a loop
  over it is empty. An attribute of it is an error. This is the default `Undefined` of jinja2.
- **Messages.** The library refuses a `messages` value that is not an array. This is a deliberate difference. mk
  passes any value to jinja2. For a map or a string, the loop runs over keys or characters by accident. The API
  takes a list of messages. A caller that sends another type has a bug, and the refusal shows it (8.5).
- **Differences that stay.** Each one has a test that pins it. The chat template uses none of them.
  - The operator `~` writes a float with an exponent, a list and a map in the form of the engine.
  - The engine has no tuple. A tuple in a template is a list. Thus `startswith` takes a list, where Python refuses
    it, and `(1, 2)` writes as `[1, 2]`.
  - The tests `lower`, `upper`, `odd`, `even`, `lt`, `ge`, `divisibleby`, `filter` and `test` give the answers of the
    engine.
  - `nothing|int`, `nothing|float` and `nothing < 1` give a value, where Python raises. `nothing|first` raises,
    where Python gives an undefined value.
  - A key with the name of a method of a dict, such as `items`, reads as the key. Python reads the method.
- The library takes JSON in its own type that keeps key order. It refuses `NaN`, `Infinity` and an integer outside
  the range of a signed 128-bit integer. Python accepts those, so this is stricter than mk.
  It refuses nesting of more than 512 levels.
  Python stops near 1000 levels.
- The GGUF reader bounds its allocations. A string of more than 16 MiB is an error, and a string is read in chunks.
  An array grows with the values that the file gives. A key twice in one file is an error.
- The tokenizer reads `tokenizer.ggml.token_type` of any integer type. A value that does not fit a signed 64-bit
  integer, and any float, is an error. A special token with an empty text is an error. mk builds an empty
  alternative that splits the text at every position, so a file like that is a mistake.
- Milestone 11 built the tokenizer, the template and the detokenizer. Milestone 12 step 4 built the parsers for tool
  calls and thinking. They match mk's `service/chatfmt.py` (`ReplyTracker`), and fixed outputs of mk's Python test
  them.
- **The finish.** mk gives only `stop` and `length`. The parser maps a stop token with one or more entries in
  `tool_calls` to `tool_use`, and a stop token with none to `end_turn`. An entry that did not parse counts as a call,
  because the model tried one. The caller says whether a cut came from `max_tokens` or from a full context. The
  parser makes no call ids: the server names each call. Decided 2026-10-20 (reading, from a note by Sonnet 5.5).
- **The pieces of a stream.** mk's `piece` is the text of the token. It holds `</think>` in the reasoning pieces, and
  the letters of a marker that arrive in separate tokens. Thus gogi's rule for pieces fails on mk's pieces in 64 of
  211 cases of the fixture. The rule says that joined pieces equal the part, up to `strip()`.

  The parser also gives the text that mk's tracker released for each token (`released`). That text keeps the rule in
  every case, with the end of thinking left out. The reply parser of the family library writes the pieces of a step
  from `released`. Decided 2026-10-20 (reading, from a note by Sonnet 5.5).
- One token can release text in two phases, but a gogi `token` event has one phase. This is rare, because each
  Qwen3.8 marker is one token. 11.3a has the rule for the step of such a token.

**The Gemma 4 library.** Decided 2026-10-22 (Amy: "accepted"). The crate `nagaya-text-gemma4` holds the Gemma
4 family. It ports mkviolet's text code, which already serves council v2 and `/mk/v1` on b70host.

- **Tokenizer.** The library reads the `tokenizer.json` of the checkpoint with the `tokenizers` crate, as mkviolet
  does (`tokenizer.rs:27-48`). The tokenizer hash is of that file (12.3).
- **Template.** The library runs Google's chat template (`assets/gemma4_chat_template.jinja` in mkviolet) with the
  nagaya's own Jinja interpreter, with `trim_blocks` and `lstrip_blocks`, as `chat.rs` does. Decided 2026-10-21
  (Amy, by interview): not `minijinja`. The identity's `template` is the text of that
  file. The variables that the library sets, such as `PRESERVE_THINKING`, are code, so the test renders of 11.3a
  cover them.
- **Escapes.** Client text that spells a marker is content. mkviolet swaps each marker text for a sentinel of
  Unicode noncharacters before the render, and it refuses a noncharacter in client text (`chat.rs:1-13`).
- **Council rendering.** The library ports `src/template.rs` (`mkv-letters-1`), the origin of 11.3a, with its Gemma 4
  layout. The template id follows 11.3a.
- **Reply parser.** The library ports `parse_tool_call` and the thought channel of `chat.rs`.
- **The method `get` of a dict.** Google's template calls `message.get(...)`. The interpreter holds `get` with the
  results of Python: a missing key gives the default, which is none. It refuses a keyword argument, as Python does.
  The other methods of a dict stay refused. Decided 2026-10-22 (Amy: "accepted"). Reason: the template needs
  it, and transformers renders it so (`crates/nagaya-jinja/tests/dict_get.rs`).

**The Gemma 4 choices.** Decided 2026-10-22 (Amy: "yeah refuse on both/all"). The port found these places where
mkviolet's rules do not carry over, or where the rules above leave a choice open:

- **A failed tool result.** mkviolet gives the template the map `{"error": content}` (`held.rs:48`). In Python's
  jinja2 a map is a sequence, so Google's template loops over its keys and fails. The library writes `Error: ` before
  the content, as the Qwen3.8 library does. Reason: one form for both families, and a render that Python gives too.
- **One form in every family.** A family library gives a failed tool result the form `Error: ` and the content. It
  uses another form only when its model's template needs that form. Amy, 2026-10-22: "seems okay if it needs to
  follow a model's template, but otherwise we should try to be consistent across models and kernels."
- **Noncharacters.** mkviolet refuses only U+FDD0 to U+FDEF. The library refuses all 66 noncharacters, and the error
  names the character and its offset, as the Qwen3.8 library does. Reason: the words of the rule above.
- **The reply parser** ports the reply splitter of `mk_api.rs` (`Splitter`, council wire), because `chat.rs` has
  only `parse_tool_call`. Its stop tokens are `<turn|>`, `<eos>`, `<|tool_response>` and `<|turn>` (`mk_api.rs:654`).
  A marker token gives an empty piece.
- **Boundaries.** A marked message is a boundary when the tokens of its rendering begin the tokens of the whole, as
  in the Qwen3.8 library. Two boundaries with the same token count keep the first.
- **Reads.** A plain read and a queue read keep mkviolet's text and its tokenization piece by piece. A read hashes the
  whole text that it was conditioned on (gogi), as the Qwen3.8 library does. mkviolet hashes the id of the head.
- **Detokenize.** Every family library refuses tokens whose bytes end inside a character. The error names `tokens`
  (11.4d). No library puts U+FFFD in their place (`crates/nagaya-text-gemma4/src/council/mod.rs`,
  `decode_whole`). A piece of one token is not a detokenize, so `pieces` keeps U+FFFD (`Family::pieces`).
- **Host tests** read the checkpoint directory from `GEMMA4_MODEL_DIR`, with no default path. With the variable unset,
  a test skips. With `NAGAYA_REQUIRE_MODELS=1`, it fails.

Reason for `tokenizers`: the port copies code that its Python fixtures already check.

Decided 2026-10-22 (Amy: "accepted"): the Jinja interpreter moves out of `nagaya-text-qwen38` into a crate
`nagaya-jinja`, which both family libraries use. The move changes no behavior, and the Qwen3.8 fixtures stay green
through it. A feature that Gemma 4's template needs and the interpreter lacks is a step with a test that fails first.
Reason: one template engine for every family, as Amy chose, with no second copy of it.

**The engine registry.** Decided 2026-10-22 (Amy: "accepted"). Before this rule, a hard map from the engine kind
picked the family (`if kind == crate::mk::KIND` in `crates/nagaya/src/serving.rs`). These rules replace it:

- The identity gets one more field, `family`: `toy`, `qwen38` or `gemma4`. The engine sets it.
- The process holds a registry of family libraries by that name. A family library opens from the identity and the
  table `[engines.family]` of the profile, which names its files.
- The opened library must give the `tokenizer_hash` of the identity, as 11.7 says now. Another hash refuses the
  start (exit 3).
- `context_tokens` of the served model is the largest slot of the engine, in tokens. `SlotLimit` gets a member
  `tokens` for it. This replaces the 4096 and mk's `MAX_CONTEXT` of 11.7.

Reason: the identity names what the engine computes, so it selects the text rules too (11.3). A new engine kind then
needs no change in `serving.rs`. mkviolet gives `context_tokens` as the slot size less one (`council_server.rs:334`).
The nagaya does not copy that.

**Registry choices.** These choices fill what the rules above leave open. Decided 2026-10-21 (Amy: "seems fine",
"good", "ok", for the digest, `context_tokens` and the key `gguf`):

- The identity digest (12.1) does not cover `family`. The family follows from the model, the tokenizer hash and the
  template, which the digest already covers. A new digest would make every held entry of gfx1151host unusable.
- An engine whose family has no library is not a served model, and the log says so. This is the rule of 11.7 for
  an unknown kind, now keyed by the family.
- The key `gguf` of `[engines.family]` names the GGUF file of the `qwen38` family. With no key, the library reads the
  first shard of `MK_SHARDS`, as before. The `toy` family needs no file and no tokenizer hash.
- The registry of engine factories holds each kind that the build has. The feature `hip` gives `mk-qwen38-gfx1151`,
  `xe` gives `mk-gemma4-bmg`, and `fake` gives the fake kinds. A build with no kind refuses the start (exit 3).
- The key `model_dir` of `[engines.family]` names the checkpoint directory of the `gemma4` family. With no key, the
  library reads `GEMMA4_MODEL_DIR`. The engine config reads the checkpoint in the same way (engine-abi "The first
  Gemma 4 binding").
- `context_tokens` is `SlotLimit.tokens` of the largest slot, with no "less one". mk declares 32,768, the value
  that `MAX_CONTEXT` had.

**11.3a Council rendering.** A decision and a raw read put a spec, a case and a letter menu after each context. The
nagaya renders them with mkviolet's rules (mkviolet `src/template.rs` and its council code), ported to each family.
The family library holds the port, because the tokens belong to the family. Decided 2026-10-20 (Amy: "mkviolet is
the most recent; mk is a bit behind. so port mkviolet's rules").

A served model's `template` in council identity has two parts: `nagaya-council:<hash>+<family>-chat:<hash>`.

- The first hash is over the tokens of a fixed set of test renders. The family library keeps the set: specs, cases
  and contexts. Thus the id changes only when the rendered tokens change.
- An edit to a comment or a change in the code structure keeps the id. Thus it keeps the thresholds that clients
  fitted (gogi rule 6).
- The id has no hand version, such as the `-1` of mkviolet's `mkv-letters-1`. The hash does that work.
- The second hash is the digest of the chat template text (12.1).
- Each hash is the first 16 hex digits of a SHA-256.

Decided 2026-10-20 (Amy chose the name `nagaya-council`).

**Open:** the first hash covers the tokens of the test renders and nothing else. A change in how the family parses a
reply does not change the id.

**The Qwen3.8 rendering.** The spec and gogi leave these choices open for Qwen3.8. Sonnet 5.5 proposed them on
2026-10-20. Each rule has the mark "Decided by senpai 2026-10-20 night (vetoable)", and Amy can veto it.

- Every render uses the reasoning effort `medium`, as mk's council does. The template then writes no effort line.
  Decided by senpai 2026-10-20 night (vetoable).
- Thinking follows the context: the render uses `thinks()` of the context. A context renders the same with thinking on
  and off, except for the generation prompt. A past assistant message keeps its reasoning in both. A test compares the
  two renders.
  Decided by senpai 2026-10-20 night (vetoable).
- A read closes the thinking. The case turn ends with the template's prompt for thinking off:
  `<think>\n\n</think>\n\n`.
  Decided by senpai 2026-10-20 night (vetoable).
- A plain read keeps mkviolet's layout: spec turn, case turn, `Answer to question`, then ` N (stem):`. The letter
  token has a leading space, as in mkviolet. A raw read uses the letters that the client sends.
  Decided by senpai 2026-10-20 night (vetoable).
- A plain read tokenizes the template text and the client text in one pass, as mk renders a chat. mkviolet tokenizes
  the pieces one by one, and the two ways can differ at a join. The library checks that the tokens of the context
  begin the tokens of the read. It also checks that each step begins with the tokens before it. A failed check is an
  internal error (`council/read.rs`).
  Decided by senpai 2026-10-20 night (vetoable).
- A queue read writes the call in Qwen3.8's form, with the parameter `answers` as one JSON object. It reads a bare
  letter after the opening quote of the value.
  Decided by senpai 2026-10-20 night (vetoable).
- A queue read asks the questions in the order of the properties of `answers` in the queue tool's schema. Qwen3.8's
  template writes the keys in the order it gets them. mkviolet sorts by question id, because Gemma 4's template sorts.
  A test pins the order.
  Decided by senpai 2026-10-20 night (vetoable).
- The workspace turns on the `serde_json` features `preserve_order` and `float_roundtrip` for all its crates, because
  Cargo joins the features of one build. The queue read needs `preserve_order` to read the order of the schema.
  Without it, the map sorts its keys, and the questions come in the wrong order. `float_roundtrip` keeps a float of a
  schema or a call equal to the float that was sent. A build of one crate alone may lack both.
  Decided by senpai 2026-10-20 night (vetoable).
- The template has no form for a failed result. The library writes `Error: ` before the content of a tool message with
  `is_error`.
  Decided by senpai 2026-10-20 night (vetoable).
- A system message after a message of another role renders as a system turn, in its place. gogi allows it
  (`docs/council-api.md`: "A system message may come later"), and `gogi-conformance` sends one. The template refuses
  such a message, so the library gives it a user message with stand-in text. Then the library swaps that user turn
  for a system turn with the trimmed content. The template merges the system messages at the start into one turn.
  Decided by senpai 2026-10-20 night (vetoable).
- A marked message is a boundary when the tokens of the rendering up to it begin the whole rendering. A tool message
  inside a run of tool messages is not a boundary, because the template puts one user turn around the run. A later
  system message ends the run.
  Decided by senpai 2026-10-20 night (vetoable).
- The template keeps the reasoning of every assistant message. No marked message loses its boundary for that reason.
  Decided by senpai 2026-10-20 night (vetoable).
- `rendered_sha256` is the SHA-256 of the text that the read was conditioned on (gogi's words). mkviolet hashes the
  head's snapshot id. A plain read hashes the text through `Answer to question`, which all its questions share. A
  queue read hashes its whole text at the end: the call through the last letter that the read chose, and its closing
  quote.
  Decided by senpai 2026-10-20 night (vetoable).
- The reply parser follows these rules.
  - It takes the stop token as a pushed token (`<|im_end|>` or `<|endoftext|>`). It adds no text for that token.
  - A token has the phase of the last text that it released. The step of the token holds only the text of that phase.
  - A token can release text in an earlier phase too. That text is in the fields of the whole reply, and in no step.
  - The end of the thinking is not part of the reasoning.
  - A call that parses but holds an integer of more than 64 bits becomes a malformed call. Its text stays as written.

  Decided by senpai 2026-10-20 night (vetoable).

**11.3b The toy family.** The fake engine has no real tokenizer. For tests, it serves the toy family of gogi's
`gogi-toy` crate. Its tokens are the 256 byte values and 14 template markers, and its template is the template of
`gogi-fake`. The family library `nagaya-text-toy` puts the nagaya's family interface on `gogi-toy`, and the fake
engine gives logits over the toy vocabulary. Thus `gogi-conformance` runs against the process with the fake engine, and
needs no GPU and no GGUF. Decided 2026-10-20 (Amy: "extend gogi to have what we need").

**11.3c The job path of a build.** A build renders a context, finds the longest held prefix, restores it, feeds the
rest, and holds a snapshot at each boundary. These notes fix the choices that the text above leaves open.

- **The store and the group thread.** `lookup` and `restore` take the engine, and the engine lives on its group
  thread (3.2). Thus a build runs them on the group thread. The store sits behind a mutex that the group threads and
  the park thread share. The ledger stays on its own thread, and a build asks it by message.
  Decided by senpai 2026-10-22 (vetoable).
- **The quanta of a build.** The first quantum has the kind `copy`. It chooses a slot, looks up the prefixes and
  restores the longest one. It then adds the other quanta to the job. Decided by senpai 2026-10-22 (vetoable).
- **The kinds of the later quanta.** A feed of at most `max_pass_inputs` tokens is `prefill`. An engine with a feed
  rule feeds one block in each quantum (6.14). A seal and a put at a boundary is `copy`. A feed never crosses a
  boundary. Decided by senpai 2026-10-22 (vetoable).
- **The seal.** A build calls `seal` at each boundary that it passes, before it reads the parts. A snapshot is a
  context sealed at a boundary (ontology). An engine without `seal` skips the call.
  Decided by senpai 2026-10-22 (vetoable).
- **The slot.** A build takes the lowest free slot where the prompt fits (`fits` on each cells part). A build that
  fits no slot is `too_large`. A build with no free slot is `busy`. Decided by senpai 2026-10-22 (vetoable).
- **What a build lists.** The response lists the boundaries that the store holds when the build ends. A boundary
  that eviction removed, below the start of the build, is not made again. A dry run feeds nothing and lists no
  snapshot. Decided by senpai 2026-10-22 (vetoable).
- The head of a dry run. A dry run reports the held head when the store holds it. Otherwise the head is `recipe:` and
  the hex of the head's recipe. It names the state that a build would make. No store entry has the id, so a snapshot
  read or a pin of it finds nothing.

  Reason: gogi types `head` as required, and gogi is the contract. This settles a note of Claude Sonnet 5.5.
  Decided by senpai 2026-10-20 night (vetoable).
- **The pin owner.** gogi names an owner with text, and the store names it with a `u64`. The service keeps the
  text and gives the store the first 8 bytes of the BLAKE3 hash of the text. A restart keeps the text in `pins.json`
  in the store directory. Entries that builds put have the owner 1. Decided by senpai 2026-10-22 (vetoable).

**Reads and generations on the job path.** The rules below are each "Decided by senpai 2026-10-20 night (vetoable)".

- A raw read, a decision read and a generation are each one job for each context. The job builds the context, as a
  build does, and its work goes on in the same slot. When the store holds the head, the job restores it.
- A decision read feeds the prefix of its plan once. The first question that starts after the prefix keeps a copy of
  the state in memory. Each later question of that kind writes the copy back into the slot first. A raw read does
  the same for each suffix after the first.
- The usage of a read counts the tokens that it fed: the shared prefix once, and no letter after the last question.
  gogi-fake counts the prefix once for each question, and the last letter of a queue read. Both keep
  `kept + fed = prompt`.
- The reads of a decision run one after another, in request order. The service checks `timeout_ms` before each
  read, and the scheduler checks it between quanta (5.8). Either one ends the decision with 504 `timeout`.
- A raw read lists 8 top tokens when the request names no `top`, and at most 64. A larger `top` is a 400.
- A build of more than `context_tokens` tokens is a 413 `too_large` that names `context_tokens`. So is a raw read
  whose context and longest suffix have more. A context of exactly `context_tokens` tokens builds. `context_full` is
  only for the prompt of a generation (gogi "Errors", gogi `a07d85e`).
- The text of a top token is the detokenized token. A token that is not whole text alone, such as one byte of a
  longer character, has the text U+FFFD. Its `id` names it exactly.
- The council identity's `engine` is `nagaya:` and the first 16 hex digits of the identity digest (6.2). The digest
  covers each field of `identity()`, `numbers` too.
- A generation holds its head after the reply when both of these facts are true. Otherwise `head` is null.
  - The rendering of the context with the reply has the boundaries of the prompt before the reply.
  - The slot holds a prefix of that rendering, at most 8 tokens short.

  Then the job feeds the rest and puts the snapshot. The head of a reply that ends with `max_tokens` is the seam
  of its open rendering (see continuations below).
- A build can start at a cut. A cut is a held snapshot that is not at a boundary of the context. Examples are the end
  of a message that the client did not mark, and the head of a cut reply. The build starts at a cut when the store holds
  no higher boundary. An earlier context made that snapshot at a seal of its own. Reason: gogi's build starts from
  the longest held snapshot whose tokens begin the context, and the next request sends the reply unmarked.
- A seal changes bits. So a build from a cut keeps the cut and each seal of its chain among the boundaries of its
  state (`Outcome::inserted`). Its later snapshots have recipes with them, and the response lists no snapshot at
  them. Thus a snapshot never holds a seal that its recipe does not name (gogi "Boundaries are content").
- The service finds the cuts in the store's index by `chains_from`, the recipe of the seal that a snapshot's recipe
  chains from (`Service::starts`, `Store::chained_from`). From each seal, starting at the root, it goes to the next
  boundary of the plan. It also goes to each held entry that chains from the seal, when the context's tokens give
  its recipe. It reads only the
  entries that chain from the context, and it renders nothing. The store makes the index again from its entries at
  a load (`Index::rechain`).
- A reply parser can release text after the last token, such as text that it held back as a possible marker. The
  service adds that text to the piece of the last token event. Thus the pieces of a part join to its text in
  `done` (gogi "Streams"). The text must have the phase of the last token. Text of another phase is an internal error,
  because each marker of a family is one token.
- The service names the calls of a reply `call_<16 hex digits>_<index>`. The digits come from a BLAKE3 hash of the
  prompt and the reply. Thus a greedy replay gives the same ids.
- A generation that sends no seed takes one from the hash keys of the Rust standard library, which the operating
  system seeds.
- The sample defaults of a served model are temperature 1 with no filter, until the profile names others.
- A reply ends with `context_full` when the prompt and the reply reach `context_tokens`. When it also reaches
  `max_tokens`, it ends with `context_full` (gogi "Generations", `finish`, gogi `a07d85e`).
- The decision limits are mkviolet's until the profile names others: `decision_context_tokens` 3072,
  `contexts_per_decision` 8, `case_bytes` 65536 and `default_timeout_ms` 30000.

Continuations follow gogi's rules (`docs/council-api.md`, "Continuations", gogi `dc94cb3`). The nagaya uses them in
these ways. Each served model declares the capability `continuation` (`crates/nagaya/src/http/mod.rs`).

- **The rendering.** `Family::render` renders an open last message as a reply up to its open text, without its end
  and without a generation prompt. Thus `end` is the length of the tokens (`crates/nagaya-council/src/family.rs`).
  The toy family takes the rendering of `gogi-toy`.
- **The Qwen3.8 rendering.** The chat template renders a closed assistant message as `<|im_start|>assistant\n`,
  `<think>\n`, the trimmed reasoning, `\n</think>\n\n`, the trimmed content and `<|im_end|>\n`.
  `Qwen38Family::run_open` renders the message closed with stand-in text in its open field, and cuts at the
  stand-in. Then it adds the open field as sent. Thus the text before the field is the template's own, and the field
  keeps its white space (`crates/nagaya-text-qwen38/src/council/mod.rs`).
- **The prefix.** The tokens of a Qwen3.8 open rendering begin the tokens of its closed rendering, unless the open
  field ends with white space. The template trims that white space. The tokenizer keeps a newline apart from the text
  before it, so the newline before `</think>` merges with no last token
  (`crates/nagaya-text-qwen38/tests/continuation.rs`).
- **The Gemma 4 rendering.** Decided 2026-10-22 (Amy: "accept."). Google's template renders a closed
  assistant message in a form that differs from its generation prompt. With thinking off, the prompt ends with an
  empty thought channel, `<|channel>thought\n<channel|>`, and a closed message has none. With thinking on, a closed
  message puts `\n` before `<channel|>`, and the model writes no such newline. So `Gemma4Family` renders an open
  message as the generation prompt of the context before it. Then it adds the open text as the model writes it:
  - the reasoning as sent;
  - `<channel|>`, when thinking is on and the message has a `content` member;
  - the content as sent.

  The library tokenizes this text as one string, with the escapes of 11.3. Reason: these are the tokens that the
  model saw when it sampled the reply, so a cut reply goes on from the same tokens. The closed rendering's tokens
  do not begin with them, so a cut reply that the client sends back closed starts from the last boundary.
- **The seam check.** The family checks each open field in `render`, with `gogi_proto::continuation::check_seam`. A
  field that does not tokenize on its own is a 400 that names `continue`. This is the prefix check of mk's
  `prefill` (`service/generate.py`, `_with_prefill`).
- **The parse.** The reply parser starts in the phase that the open message sets (`ReplyStart::of`). It reads only
  the sampled tokens. After an open message, the Qwen3.8 parser keeps the white space at the start of its first part.
  At `max_tokens`, it keeps the white space at the end of its last part (`Keep` in
  `crates/nagaya-text-qwen38/src/reply.rs`). The toy parser and the Gemma 4 parser trim nothing.
- **The reply.** `done` joins the open text as sent and the new text. `usage.prompt` counts the open text. `n` and
  `usage.completion` count only the sampled tokens (`GenerateThen::finish`, `crates/nagaya/src/council/generations.rs`).
- The build plan: a continuation builds the context before its open message (`Service::open_plan`,
  `nagaya_council::continuation::open_plan`). Then it feeds the open message as a generation feeds its generation
  prompt, with no seal. Reason: a seal changes bits (mk rounds the GDN state to bf16). A seal inside the open text
  makes a continuation differ from the reply that it continues. The next request starts from the head after the
  reply. Decided by Amy 2026-10-21.
- The head of a cut reply: a reply that ends with `max_tokens` seals a snapshot of its open rendering. It gives
  that snapshot as `head`, listed last in `snapshots` (`GenerateThen::cut_head`). A continuation of that reply
  starts from the head: the longest held snapshot wins, as for any build. It may differ from the uncut reply, which
  gogi allows (boundaries are content). Decided by Amy 2026-10-21 ("Sealed head").
  - The build plan above stays. The build also looks up the seam of the open text after the plan
    (`build::Extension`, `ContextBody::restore_seam`). When the store holds it, the build restores it and the
    generation feeds the last token. The state keeps the seal of the seam, so the head after that reply has the seam
    among the boundaries of its recipe (`recipes_with_cuts`).
  - A cut reply that the client sends back closed starts from the head too, in any form. The open rendering's
    tokens up to the seam must begin the closed rendering's tokens. The search of the store's index finds the head
    (see the cuts above). So does a context that sends back the reply of a continuation from a head: that reply's
    head chains through the seam.
  - **Proposed (for Amy)** (Claude Opus 5.5, 2026-10-21), the cuts from the index. A build tries each sealed
    snapshot that the search of the index finds, not only the ends of messages. Reason: the first rule tried the
    open forms of the last assistant message. It missed a reply cut inside its thinking and sent back with
    `content: ""`, and it cost three renders of the prefix. The search also replaces the second render of the cuts at
    message ends.
  - The timings of a dry run come from a release host test (`crates/nagaya/tests/council_heads.rs`). For 18026
    tokens with an assistant message: 103 ms with the open forms, then 38 ms with a pass over the index. It is 16 ms
    with one render. The same context without the message takes 16 ms. With 10,000 unrelated entries in the store: 85 ms
    with a pass over the index, then 0.22 ms (0.16 ms when empty).
  - A dry run reports the interval copy that a build would restore. For a continuation, it asks for a slot of the
    open rendering, as the generation does.
  - A continuation whose reply's closed tokens differ from its prompt and sampled tokens holds no head after the
    reply. Two causes: the template trims white space at the end of the reply, and a token at the seam merges in
    the closed rendering. The slot holds the other tokens, so no head can name them.
  - The seam. The head is the seam of the open rendering, the position before the last sampled token, not its end.
    Reason: a snapshot holds no logit row. A continuation from a head at the end would have no first logit row, and
    it cannot feed the last token again. At the seam it feeds that token (`fed` 1), which gogi allows. Proposed by
    Claude Opus 5.5; decided by Amy 2026-10-21.
- The dry run: a build with an open message and `dry_run: true` looks up the plan of the continuation and its seam.
  Its `kept` and `fed` are what the generation keeps and feeds. Its `head` is `recipe:` and the recipe of the open
  rendering, which the store never holds. Decided by Amy 2026-10-21.
- The template id: the test sets of the families have no open message, so the open rendering is not in the template
  id (11.3a). Adding one would change the id of each family once, and a fitted threshold does not carry over.
  Decided by Amy 2026-10-21.
- A riding `before` is not built, because the nagaya does not serve `ride`. A generation with `decisions` is a 400
  that names `decisions`.

**11.4 `/mk/v1`.** kaijutsu's mk provider uses `model`, `render` and `generate` (mkviolet `mk_api.rs`).

Decided 2026-10-17: the nagaya does not serve `/mk/v1`. kaijutsu moves to council v2 and uses the gogi reference
(`~/src/gogi`). Amy: "kaijutsu will move to council v2 and use the new gogi (~/src/gogi) reference. it's happening
now, on b70host". In council v2 the server renders messages with its model's own template, and the template is
part of identity (gogi `docs/council-api.md`, "Terms").

**11.4a Extensions.** gogi puts model-specific routes under the server's own prefix, listed in
`identity.extensions` (gogi `docs/council-api.md`, "Extensions"). The nagaya's prefix is `/nagaya/v1`:

- `profile` and `cluster`: dynamic placement profiles and the `chitchat` cluster view (2.3, 2.4);
- `cas/{content_id}`: plain HTTP `GET` and `HEAD` for content-addressed snapshots and tap artifacts across machines;
- `render`, `tokenize` and `detokenize`: served by the family library for bench tools and semantic search;
- `embeddings`: served by the engine through a pass with a hidden-state output;
- `status` (9.7) and `traces` (13.1).

Built on 2026-10-21: `render`, `tokenize` and `detokenize` (11.4b to 11.4d), and the prefix in `identity` (11.4f). The
other routes in this list are not built.

**11.4b The text routes.** `render`, `tokenize` and `detokenize` use the family library (11.3). They share the rules
below. These rules are **Proposed (for Amy)**, by Claude Sonnet 5.5 on 2026-10-21. Amy decides, and senpai writes an
accepted rule as "Decided <date>".

Their sources are mk (`service/v1.py`, `service/openapi.json`) and mkviolet (`src/mk_api.rs`). Bench tools use them to
see what a council request feeds. The semantic search of kaijutsu uses them to count and to cut text by tokens.

- **Proposed (for Amy).** Each body has a required `model`, as a council request has. A body without it, or with a name
  that no served model has, is a 400 `invalid_request` that names `model`. The routes never choose a model for a
  client (gogi "Served models").
- **Proposed (for Amy).** A served model that is not `ready` gets a 503 `model_unavailable`. The route calls
  `Council::route` (`crates/nagaya/src/http/mod.rs`), as the council routes do.
  Other option: serve the text routes while the model loads, because they need only the family. The pick keeps one
  rule for all routes, and `identity` tells a client when the model is `ready`.
- **Proposed (for Amy).** The routes use no engine, no ledger charge and no job. The call runs on a call thread and
  counts toward the 64 calls (11.7). A body has the limit of 16 MiB (11.7).
- **Proposed (for Amy).** A body with an unknown member is a 400 that names it, through `parse`
  (`crates/nagaya/src/http/errors.rs`). Thus `class` and `priority` are 400, because the routes do not schedule.
- **Proposed (for Amy).** A response names the served model by its `model`, never by an alias. It carries the
  `request-id` header, as every response does.
- **Proposed (for Amy).** The HTTP server span of 9.2 covers each call, and its `http.route` is the path, such as
  `/nagaya/v1/render`. The routes make no `nagaya.job` span and no metric of their own.

Built on 2026-10-21. The code is in `crates/nagaya/src/http/extensions.rs`. The function `routes` in
`crates/nagaya/src/http/routes.rs` merges its routes, so `router` and the running server both serve them. `Service`
has one accessor, `Service::family` (`crates/nagaya/src/council/mod.rs`), because its field `family` is private. The
tests are `crates/nagaya/tests/extensions.rs` (the toy family), `extensions_qwen38.rs` (the real tokenizer and
template) and `http_telemetry_extensions.rs` (the span and the duration point).

A `FamilyError::Client` becomes a 400. The existing `From<FamilyError>` for `ServiceError`
(`crates/nagaya/src/council/error.rs`) does this. A `FamilyError::Internal` is a 500, and the process stops (8.5).

**Proposed (for Amy).** A family error can have no `param`. Then the route sets it to the request member at fault:
`text` for `tokenize`, `tokens` for `detokenize`, and `context` for `render`. mk and mkviolet name only the message.
A family error that has a `param`, such as `continue`, keeps it (`at` in `extensions.rs`).

**11.4c Render.** `POST /nagaya/v1/render` gives the text and the tokens of a context, as a council request feeds
them. **Proposed (for Amy).** It is for bench tools and for the debug of a template.

| Request member | Type | Meaning |
|---|---|---|
| `model` | string | Required. A served model or one of its aliases (11.4b). |
| `context` | object | Required. The gogi context object: `messages`, `tools`, `thinking` and `queue` (gogi "Contexts"). |
| `generation_prompt` | bool | Optional, `false` by default. When `true`, the tokens end with the start of the reply. |

| Response member | Type | Meaning |
|---|---|---|
| `model` | string | The `model` of the served model. |
| `template` | string | The template id (11.3a), from `Family::template_id`. |
| `text` | string | The rendering, for reading. Each marker of the template is its own text. |
| `tokens` | array of integers | The token ids. |
| `n_tokens` | integer | The length of `tokens`. |
| `end` | integer | The token count before the generation prompt. It equals `n_tokens` when there is none. |
| `boundaries` | array | The boundaries that the rendering keeps: `{message, tokens}`, in order (11.3c). |
| `signals.control_text` | array | The hits of control text: `{where, token}`. `where` is `message:<i>`. |

```json
POST /nagaya/v1/render
{"model": "qwen-3.8-flash-next",
 "context": {"thinking": false,
             "messages": [{"role": "system", "content": "You are brief."},
                          {"role": "user", "content": "Say hi.", "snap": true}]},
 "generation_prompt": false}
-> {"model": "qwen-3.8-flash-next", "template": "nagaya-council:ab12…+qwen38-chat:cd34…",
    "text": "<|im_start|>system\nYou are brief.<|im_end|>\n<|im_start|>user\nSay hi.<|im_end|>\n",
    "tokens": [151644, 8948, …], "n_tokens": 19, "end": 19,
    "boundaries": [{"message": 1, "tokens": 19}], "signals": {"control_text": []}}
```

The numbers in the example show the shape only.

- **Proposed (for Amy).** The route calls `Family::render(&context, generation_prompt)`
  (`crates/nagaya-council/src/family.rs`). The type `Context` is `gogi_proto::context::Context`, so the route
  decodes it as `POST /council/v2/contexts` does. The implementations are `Qwen38Family::render`
  (`crates/nagaya-text-qwen38/src/council/mod.rs`) and `ToyFamily::render` (`crates/nagaya-text-toy/src/lib.rs`).
- **Proposed (for Amy).** `text` is `Family::detokenize(&rendering.tokens)`. For Qwen3.8 it equals `Rendered.text` of
  `Qwen38::render` (`crates/nagaya-text-qwen38/src/family.rs`). A test pins this equality. `Rendering` has no text
  member, so this needs no change to the trait.
- **Proposed (for Amy).** `tokens`, `end`, `boundaries` and `signals` come from the `Rendering` as they are. The
  response never reports a boundary that a build would not keep.
- **Proposed (for Amy).** Content that spells a marker is plain text in `tokens` (gogi rule 2). In `text` it looks
  like the real marker. Thus `control_text` is the way to tell them apart. A client never sends `text` back as a
  prompt. It sends the context.
- **Proposed (for Amy).** Unlike the council responses (11.7), `render` reports the hits of `message:<i>`. gogi names
  no such place for the council routes, but this route is a debug tool, and the place is the answer that it gives.
- **Proposed (for Amy).** `render` does not check `context_tokens`. A client compares `n_tokens` with the limit in
  `identity`. Thus the route needs no code for `context_full`.
- **Proposed (for Amy).** The route shows the context only. A decision also renders a spec, a case and a letter menu
  after it (11.3a). A later route can show them through `ReadPlan::prefix`. It is not in the first set.

Choices between options, each one made for Amy to change:

- **Proposed (for Amy).** `generation_prompt` is `false` by default. This matches a build and a decision, which render
  with no generation prompt, and so `boundaries` match a build. mk and mkviolet default to `true`. A client that
  wants the prompt of a generation sends `true`.
- **Proposed (for Amy).** The context is one member, `context`. The same object then goes to `/council/v2/contexts`.
  The other option is mk's flat body, with `messages`, `tools` and `thinking` at the top. It saves one level, and a
  client copies less.
- **Proposed (for Amy).** The route has no `reasoning_effort`. gogi's context has none, and the Qwen3.8 library fixes
  it to `medium` (`EFFORT` in `crates/nagaya-text-qwen38/src/council/mod.rs`). A body with it is a 400.

A message that the template refuses, a context that fails `Context::check`, and a Unicode noncharacter in a string
are each a 400 `invalid_request`. The message names the fault, as mk does. The `param` is `context`, unless the fault
is in a member that has its own name, such as an unknown member of the context.

The route detokenizes the tokens of the rendering to make `text`. These tokens come from valid text, so a failure
there is a defect: a 500 `internal`, and the process stops (8.5).

**11.4d Tokenize and detokenize.** `POST /nagaya/v1/tokenize` turns text into token ids. `POST /nagaya/v1/detokenize`
turns token ids into text. **Proposed (for Amy).** They are for the semantic search, for a count of tokens, and for a
look at the pieces of a rendering.

| Request member | Type | Meaning |
|---|---|---|
| `model` | string | Required (11.4b). |
| `text` | string | Required for `tokenize`. |
| `special` | bool | Optional for `tokenize`, `false` by default. When `true`, the text of a marker is its one token. |
| `pieces` | bool | Optional for `tokenize`, `false` by default. When `true`, the response lists the text of each token. |
| `tokens` | array of integers | Required for `detokenize`. Each id is an unsigned 32-bit integer. |

| Response member | Type | Meaning |
|---|---|---|
| `model` | string | The `model` of the served model. |
| `tokens` | array of integers | `tokenize` only. The token ids. |
| `n_tokens` | integer | `tokenize` only. The length of `tokens`. |
| `pieces` | array of strings | `tokenize` only, and only when asked. One string for each token. |
| `text` | string | `detokenize` only. The text of the tokens. |

```json
POST /nagaya/v1/tokenize
{"model": "qwen-3.8-flash-next", "text": "Hello world", "pieces": true}
-> {"model": "qwen-3.8-flash-next", "tokens": [9707, 1879], "n_tokens": 2, "pieces": ["Hello", " world"]}

POST /nagaya/v1/detokenize
{"model": "qwen-3.8-flash-next", "tokens": [9707, 1879]}
-> {"model": "qwen-3.8-flash-next", "text": "Hello world"}
```

- **Proposed (for Amy).** With `special` off, `tokenize` calls `Family::tokenize`. The text of a marker stays text
  (gogi rule 2). For Qwen3.8 this is `Qwen38Family::tokenize`: escape, then `encode_escaped`.
- **Proposed (for Amy).** `detokenize` calls `Family::detokenize`. A token outside the vocabulary is a 400 with
  `param` `tokens`. The message of `Error::TokenId` already gives the vocabulary size.
- **Proposed (for Amy).** Tokens that do not make whole UTF-8 text are a 400 with `param` `tokens`. mk and mkviolet
  replace the bad bytes. Here the route follows `Family::detokenize`, and a silent replacement would hide a split
  character. Other option: add a member `lossy` that replaces the bytes.
- **Proposed (for Amy).** An empty `text` gives an empty `tokens`. An empty `tokens` gives an empty `text`. Neither is
  an error.
- **Proposed (for Amy).** `pieces` are lossy by design. A piece is the bytes of one token, and a token can end inside
  a character. The route replaces bad bytes with U+FFFD in each piece (`String::from_utf8_lossy`). The piece of a
  marker is its text.
- **Proposed (for Amy).** Code found a difference from mk. mk's `piece_text` drops a cut-off character at the end of a
  piece. The route keeps a U+FFFD, so that a client sees the cut. A token of several stray bytes can give more than
  one U+FFFD, and a top token in a raw read (11.3c) always gives one.
- **Proposed (for Amy).** Each route has its own members. `special` and `pieces` in a `detokenize` body, and `tokens` in
  a `tokenize` body, are each a 400 that names the member. A `tokens` member must be an array of unsigned 32-bit
  integers. Any other value is a 400 with `param` `tokens`, and the message names the place of the bad id.

Two small additions to the `Family` trait (`crates/nagaya-council/src/family.rs`) serve `special` and `pieces`.

- **Proposed (for Amy).** Built on 2026-10-21. `Family::tokenize_special(&self, text: &str) -> Result<Vec<u32>,
  FamilyError>`. Qwen3.8 calls `Tokenizer::encode`. This is the rule of a raw-read suffix
  (gogi "Raw reads"), where a marker is one token.
- **Proposed (for Amy).** Built on 2026-10-21. `Family::pieces(&self, tokens: &[u32]) -> Result<Vec<String>,
  FamilyError>`. Qwen3.8 calls `Tokenizer::token_piece(id, true)` for each id, then converts with lossy UTF-8
  (`crates/nagaya-text-qwen38/src/council/mod.rs`). `ToyFamily` and the test family `Holding`
  (`crates/nagaya/tests/council_generations.rs`) implement both. Both methods are required, not defaults, so a family
  that does not implement them does not compile.
- **Proposed (for Amy).** The other option was to build the first set with neither `special` nor `pieces`. `tokenize`
  then needs no change to the trait. The cost is a bench tool that cannot tokenize a raw suffix. The build took the
  pick above.

Tests that can fail (all built, in `crates/nagaya/tests/extensions.rs` and `extensions_qwen38.rs`):

- `detokenize(render.tokens)` equals `render.text`, for each fixture of the family.
- `tokenize` with `special` on, applied to a `render.text` with no `control_text`, gives `render.tokens`.
- The same call, applied to a text with `control_text`, gives other tokens. This pins the mk behavior in its
  OpenAPI text.
- A noncharacter, an id past the vocabulary, and a split character each give a 400 with the `param`.

**11.4e Model and health.** **Proposed (for Amy).** The first set has no `model` route and no `health` route. Both
exist in mk, and `GET /council/v2/identity` covers most of what they give.

- **`model`.** mk's object has `id`, `weight_hash`, `max_context`, `build.engine_sha256` and `device`. The council
  identity has `model`, `weight_hash`, `engine`, `limits.context_tokens` and `device` for each served model. The
  members `hidden_size` and `vocab_size` are not there. A client needs them for a vector space, so they belong to the
  `embeddings` response, which this section does not cover. `Family::vocab_size` already holds the vocabulary size.
- **`health`.** mk answers 200 while it serves and 503 after a failure. The nagaya stops the process at a failure
  (8.5), so it has no state that serves and is unhealthy. A probe uses `identity`: a 200 with each model `ready` is
  healthy. A fatal engine reports `unloaded` there (11.1). systemd sees `READY=1` when the engines open (10.1).
  The watchdog thread ends a hung process (7.4).
- **`weights`.** mk's `health` tells `attached` or `loaded`. The nagaya counts this as `nagaya.arena.handoff` (9.4),
  with the status `hot` or `cold`. No client route is needed.

The other option is a `GET /nagaya/v1/health` that gives only `{"status": "ok"}`, for a load balancer that cannot read
`identity`. It costs one route and no new state.

**11.4f Extensions in identity.** **Proposed (for Amy).** `identity.extensions` lists the prefix `/nagaya/v1` once.
gogi's example lists `/mk/v1` this way, and it lists no route (gogi "Extensions").

- **Proposed (for Amy).** `Council::identity` returns `["/nagaya/v1"]` as soon as the process serves one route under
  the prefix. The text routes need a family. The list holds the prefix when the council has a served model, and it is
  empty otherwise (`crates/nagaya/src/http/mod.rs`). Built on 2026-10-21.
- **Proposed (for Amy).** The list claims only routes that exist. A route that is not built is a 404 `not_found`
  (11.7), also under the prefix. Thus a client probes a route by calling it.
- **Proposed (for Amy).** Each machine lists the prefix for its own routes. `status` (9.7) is per machine, and the
  text routes need a family that the machine holds (11.1).
- **Proposed (for Amy).** A test calls `identity` and checks that the prefix is present if and only if a route under it
  answers. A test for the empty list fails when the first route is merged without the change.

The other option is a route `GET /nagaya/v1` that lists the routes by name and version. It is more work, and gogi
does not ask for it.

**11.5 Errors.** All routes use gogi's error body, `{"error": {"type", "message", "param"?}, "request_id"}`, and the
`request-id` header. Council routes use only gogi's error table (gogi `docs/council-api.md`, "Errors"). The
`/nagaya/v1` routes add two codes:

| Code | `error.type` | When |
|---|---|---|
| 409 | `conflict` | a profile `PUT` with an old version (2.3) |
| 507 | `no_memory` | a profile that does not fit its budgets (2.4) |

An unexpected failure is a 500 `internal`, and the process stops (8.5).

**Proposed (for Amy).** The routes of 11.4b to 11.4d, and the status feed of 9.7a, need no new code in this table.
They use the codes that the table and gogi's table already hold:

| Code | `error.type` | When, on these routes |
|---|---|---|
| 400 | `invalid_request` | A missing `model`, an unknown member, a bad context, text or tokens, or a bad `engine` or `kind` query. |
| 413 | `too_large` | A body past 16 MiB (11.7). |
| 429 | `busy` | More than 64 calls (11.7), or more than 16 open feeds (9.7a). |
| 500 | `internal` | A family library defect. The process stops (8.5). |
| 503 | `model_unavailable` | The served model is not `ready`. The text routes only. |
| 503 | `unavailable` | The process is stopping. |

**11.6 Binding.** The HTTP and gossip listen addresses and seed peers are in the bootstrap config (2.1). (mkviolet
has a tailnet address in its code.)

**11.7 The HTTP routes.** The process serves the council routes with `axum` on the control runtime
(`crates/nagaya/src/http`). It starts the server after `READY=1`, and it logs `http listening on <address>`. These
rules are each "Decided by senpai 2026-10-20 night (vetoable)".

- The engine registry of 11.3 replaces the first two rules below. The code builds it
  (`crates/nagaya/src/serving.rs`, `FamilyRegistry`).
- The process serves an opened engine when its kind has a family library. The kind `fake-toy` has the toy family
  (11.3b). The kind `mk-qwen38-gfx1151` has the Qwen3.8 family, from the first shard of `MK_SHARDS`. The tokenizer
  hash of that family must equal the `tokenizer_hash` of the engine identity. Thus the identity selects the library
  (11.3). Another kind is not a served model, and the log says so.
- The served model is the `model` of the engine identity, as gossip publishes it (2.3). It has no alias and no
  `device` yet. Its `context_tokens` is 4096, or mk's `MAX_CONTEXT` for mk.
- A process serves at most one model in milestone 12. The specs and the pins of a service are files in the store
  directory, and two services would write one file. A profile with two served engines refuses the start (exit 3).
- Specs, pins and snapshots belong to the server. Their routes use the one service. With no served model, they are
  a 503 `unavailable`.
- A service call blocks, so each call runs on a call thread of its own, on the pool CPUs (3.2a). At most 64 calls
  run at once. A request past that is a 429 `busy`. A panic in a call is a 500 `internal`.
  A call thread that cannot pin panics before it runs the call, which is also a 500 `internal` (3.2a).
- A request body has at most 16 MiB. A larger body is a 413 `too_large` that names `body`.
  A body that does not arrive whole is a 400 `invalid_request` that names `body`.
- A route that does not exist is a 404 `not_found`, also for a known path with another method.
- A request id is `req_<start time in hex>_<count>`. It differs between requests and between runs.
- A ledger that refuses the charge of a context or of a staging copy (4.2) is a 429 `busy`. A job that a stop
  cancelled is a 503 `unavailable`.
- An internal error is a 500. The process then stops with exit 1 (8.5). This holds for an error that comes while the
  process stops: the stop path reads the failure again, and a clean stop becomes exit 1.
- A stream waits for its first event before it sends the status. Thus a request that fails before its first token
  gets a status response. A later failure is an `error` event. At most 256 events wait for a slow client. A full
  queue holds the generation back, and it drops no event.
- A closed stream ends the forward thread, so the generation stops at its next token and holds no head.
- A council response does not report a control-text hit in a message of a context (`message:<i>`). gogi names only
  the places `case`, `spec` and `suffix:<i>` (gogi rule 2). The response of `render` does report it (11.4c).
- A stop first sets every served model to `draining`, so a new request is a 503 `model_unavailable`. Then the
  engines stop (10.3), and the server waits at most 2 s for its open requests.
- The routes `render`, `tokenize` and `detokenize` under `/nagaya/v1` are built (11.4b to 11.4d). The other routes of
  11.4a and the status feed (9.7) are not built.
- Continuations are built (11.3c). `gogi-conformance` against the process with the fake engine gave 271 to 274 ok,
  0 FAIL and 8 skip on 2026-10-21, at gogi `dc94cb3`. The number of checks depends on the reply to a random nonce.

## 12. Identity

**12.1 Fields.** An engine identity has the fields of `identity()` in the ABI:

- `model`;
- `checkpoint_digest`, the first of the two weight ids (12.2);
- `tokenizer_hash`, the hash of the tokenizer that the engine uses: of its tokenizer file, or of the tokenizer keys
  of its GGUF (engine-abi "v0 mk binding choices");
- `template`, the text of the chat template, for example the GGUF key `tokenizer.chat_template`. The identity digest
  covers it, so a new template is a new recipe. Gossip carries the digest, not the text (2.3). Decided 2026-10-20
  (reading, kaibo job-2 finding 7);
- `family`, the name of the family library that the engine needs (11.3, "The engine registry");
- `numbers`: the engine binary, and the nagaya code that turns its outputs into answers;
- the lineage of the plan table;
- the bit settings, each with its class (`docs/prior-art.md` 1);
- the canary hash.

The nagaya keeps each engine's expected identity in the store's index. At start, it compares the identity from
`identity()` with the expected one. A mismatch makes the old entries of that engine unusable. It is a refusal to
start only when the profile pins the identity.

**The canary decides for a new build.** Decided by senpai 2026-10-22 (vetoable). This replaces the note "A GPU
binary in the identity" of 2026-10-21. That note put the compiler and the driver in the identity digest. Then each
driver update makes every held entry unusable, also when the bits do not change. Amy, 2026-10-22: "if I have a long
running context and upgrade I don't want to have to refill it just because I built a new binary that is not very
different most of the time."

- The identity gets a member `build`, with the facts of the build. It holds the SHA-256 of each kernel binary and
  engine library, and the compiler version. It also holds the driver or runtime version, and the PCI ids of the card.
- The identity digest does not cover `build`. It covers the other fields of 12.1, also the canary hash.
- `numbers` keeps the values that change bits by choice, such as mk's `fast_qmm` and `tiles`. A file path is not a
  number. mk's `engine_lib` path is gone, because the binary links mk's engine (Amy, 2026-10-22, AGENTS.md milestone
  15). Until `build` exists, `numbers` holds mk's build facts: `engine_build_sha256`, `engine_compiler` and
  `engine_hip` (`crates/nagaya-mk-qwen38-gfx1151/src/spec.rs:515-517`).
- Thus a new build with the same canary bits has the same digest. Its held entries stay usable, and the gate
  thresholds still fit. A new build with other canary bits has a new digest, as now.
- Each entry records the `build` that made it. A context with rows of two builds is then visible.
- A Level Zero engine reads the driver version with `zeDriverGetProperties`. A HIP engine reads the HIP runtime
  version. A kernel that compiles at load is hashed after the compile.
- The canary covers the shapes of the oracle fixture (8.3): a short prompt, a long prompt and a greedy decode. A
  staged engine runs each stage, and a ring part wraps.
- Order: build the canary (8.1) first. Until it runs at each start, `build` stays in the digest, as now.

Reason: the bits decide if a build changed the numbers, not the facts of the build. A canary covers only the paths
that it runs, so its inputs must run each path that the limits allow.

**12.2 Two weight ids.** The checkpoint digest is logical. It hashes the canonical config and each tensor's payload,
so a new packing does not change it. The arena id is the content id of the packed bytes that the engine runs.
`arena_export` returns it.

The identity carries the checkpoint digest. The arena records its own id and the strength of its check. mk's
`weight_hash` and mkviolet's `header_hash` hash only the shard headers and sizes. They are spot checks, and the
identity labels them as spot checks until we replace them.

**12.3 Identity defects to correct in the port.**

- mkviolet loads EmbeddingGemma 2's `tokenizer.json` for Gemma 4 (`main.rs:132`). The file "differs ... by a few
  hundred bytes; which entries differ is not yet checked" (`tokenizer.rs:4-5`). Thus its tokenizer hash names the
  wrong file. The port loads the `tokenizer.json` of the Gemma 4 checkpoint (`checkpoints/gemma-4-26B-A4B/`).

  The two files differ only in the post-processor (checked 2026-10-21). EmbeddingGemma 2's adds `<bos>` and `<eos>`.
  mkviolet encodes with no post-processor, so the ids are the same, and only the hash differs.
- mkviolet's `tokenizer_hash` is the first 16 hex digits of the SHA-256 of the file (`tokenizer.rs:22`). Decided
  2026-10-22 (Amy: "accepted"): the port hashes the whole file and writes `sha256:` and 64 hex digits, as the Qwen3.8
  library does. Reason: one form for every family.
- mkviolet's `weight_hash` hashes the shard headers only (`main.rs:216`). The port labels it `spot-sha256:`, as the mk
  binding does (12.2).
- mkviolet's metrics label card 1 `google/embeddinggemma-2` (`telemetry.rs:492`, `:523`). The port takes the label
  from the identity.

## 13. Taps, lenses and composition

**13.1 Taps.** An engine declares its taps in `limits()`, with the conditions that each tap needs (mk: unfused, one
segment). A tap gives one of these outputs: a residual at a layer (rows), routes, or gates. The store keeps large
tap outputs as content-addressed artifacts, with an owner and a budget. (mk keeps traces as per-layer `.npy` files.)

**mkviolet's taps, for later.** These taps are not in the first port. They are noted here so that the port does not
close a door.

- **Routes.** mkviolet writes each layer's expert picks (`MoERouteDecision`, 128 bytes) into host memory
  (`gemma4.rs:521`), and `count_routes` reads them after each stage (`gemma4.rs:1219-1230`). A route tap costs no
  copy.
- **Residuals.** A J-Lens reads the residual at layer l (13.2). The residual after layer 14 already crosses host
  memory between the stages (D1).
- **Steering.** A steering vector is an input that changes bits. Thus it goes into the recipe (6.12), and a steered
  state never has a plain recipe.

**13.2 Lenses.** A lens (mkviolet J-Lens) is a state object of its own. It belongs to one model, has a content id,
and runs on a CPU device (2.1). Two models cannot share a lens. Models are compared in the lens's concept
coordinates, not in raw residuals.

**13.3 Composition between models.** kaijutsu composes models through the council API, for example a judge that
follows a coder. The nagaya keeps the engines loaded side by side and schedules their classes. Tensor transfer
between engines is out of v0. (mkviolet `docs/02` designs KV bridges and residual splices.) If it comes, these
rules apply:

- it stays on one machine;
- it uses an `inject` capability;
- each adapter is a state object with its own identity.

## 14. Porting notes

The inventories found these items. We correct each one when its code moves. We do not copy them.

- mk splits a prefill in its Python service (`contexts._feed`). The split moves under `max_quantum_ms` (5.1).
- mk's `prefill` field of `/mk/v1/generate` (`service/generate.py`, `_with_prefill`) becomes a gogi
  continuation. Its prefix check at the seam moved with it (11.3c, the seam check). The new name keeps "prefill" for
  the prompt pass.
- mk orders jobs first in, first out in its one high tier. The nagaya uses the order in 5.4.
- mk's "park" writes through to disk and seals at each "hold". In nagaya terms, a hold is a declared boundary, and
  a write to disk is a park to NVMe (D7).
- mk's MemAvailable floor becomes the safety floor in 4.7. It does not set any size.
- mkviolet's three caches merge into the store (6.1). Its fixed budgets become ledger terms.
- mkviolet's telemetry setup leaks a runtime (`Box::leak`). Its HTTP server is hand-written, and its bind address is
  in its code.
- mkviolet has instruments without call sites (9.5) and wrong model labels (12.3).
- mkviolet's README gives `GEN_SEQ` 16384 and its `main.rs` gives 65536. Its `docs/07` is out of date.
- mkviolet declares `always` with a feed rule (6.14, engine-abi D11). The first port runs one recipe in two
  processes and compares the content ids. The test shows if mkviolet meets the class.

## 15. Open questions

1. Do an idle engine's GPU queues slow the other engines on its device (3.3)? A measurement answers this.
2. What makes `close()` safe for mk, so that engines can load and unload at runtime (3.3)?
3. How long does the conversion to canonical safetensors take on mk and mkviolet (10.3, D10)?
4. Answered 2026-10-20: on mk, one pass of many tokens gives the same bits as one pass for each token (6.13).
   On 2026-10-21, splits across 256 and 512 tokens gave the same bits in both classes of context (6.13).
5. How does a profile `PUT` work across machines (2.3, 2.4, 11.5)? The machine that takes the `PUT` cannot
   return the target's 507, and 2.3 ignores an old version where 11.5 answers 409 (kaibo, 2026-10-20). The route
   `PUT /nagaya/v1/profile` is not built, so the question stays open.
6. The template engine differs from jinja2 on integers past 2^127 and on a float key that is not finite (milestone
   11.5 text step). The chat template uses none of them. Which ones do we match?
7. How far does mk's fast path (`MK_FAST_QMM` unset) move council decisions? At 249 tokens it moved logits by up to
   1.03, and the argmax and the top 8 held (8.3). Amy, 2026-10-20: "we'll measure later".
8. What causes the oracle gap past about 640 tokens, and how large is it at 2,000 tokens and more? The contexts
   hold 4,096 and 32,768 tokens now, and a long prompt takes several passes. On the exact path, the findings are
   these:
   - mk agrees with the oracle up to 630 tokens. The largest logit difference is 5.2e-5 at 478 tokens, 1.9e-5 at
     604 and 1.5e-5 at 630 (diagnostic prompts).
   - At 655 tokens the difference is 0.29. At 684 it is 0.16. At 721 it is 0.23.
   - The split into passes does not change mk's bits, so the pass split is not the cause.
   - The cause is not known: mk or the reference. The indexer is dense below about 2,050 tokens in both.

   Amy, 2026-10-20: "we can save and do the measurement later".
9. Should the control threads run at elevated priority (3.2, 3.2a)? A raised priority needs `CAP_SYS_NICE` or a
   setting in the unit, such as `Nice=`. A measurement under load shows whether the pinned cores are enough.

10. Can a restart keep a pinned entry of the host tier (6.11)? Today the index drops the entry and its pin at start.
    The restart check of `gogi-conformance` needs the stop to park such an entry to NVMe.
11. How does the park charge the host copy of a state (10.3 "Open", 4.2 "Staging")? The copy has no staging charge
    today. The charge path refuses while the process stops.
12. Can a copy quantum run past the 30 s watchdog timeout on a slow NVMe (7.4)? We have not measured it (kaibo review
    job-1, finding 4).
13. How does mk reset a context without a new allocation (engine-abi "v0 mk binding choices")? `slot_clear` frees the
    context and allocates a new one. A `hipMalloc` that fails stops the process. mk needs a reset call.
14. Can the nagaya pin the two HIP runtime threads of mk (3.2a)? The HIP runtime starts them and sets their masks. The
    mask of both is 0-31 on gfx1151.
15. Should the template id (11.3a) cover more than the tokens of the test renders? A change in how the family parses a
    reply keeps the id today.
16. Answered 2026-10-21: the job path uses `ObservedScheduler` and the evidence module (9.2, 9.6). Two things stay
    open (9.2). The council thread loses the parent of `nagaya.job` for an HTTP request. The engine receives no trace
    context.
17. Which masks do the threads of the Level Zero runtime and IGC get on b70host (3.2b)? The nagaya does not pin
    them, so a measurement shows whether they leave the pool set.

## 16. Decided by senpai

Senpai chose each rule below where the spec left a choice open. Each rule has the mark "Decided by senpai 2026-10-20
night (vetoable)" or "Decided by senpai 2026-10-22 (vetoable)" in its section. Amy can veto any of them. Each line
gives the section and the rule in a few words. The lists before the last one are from 2026-10-20 night.

**In this spec**

- 2.1: a CPU number of 1024 or more is a probe error.
- 2.1: the probe gives a true size or stops, and it refuses a model or a driver that it cannot size.
- 2.1: the file tree for sm121host is made by hand, and each guessed value is marked. The tree for b70host is real.
- 2.1: a build with the feature `fake` keeps a made-up inventory.
- 2.1: a probe error at start exits 3, and a build with no `fake` probes the files of `/`.
- 3.2a: the housekeeping set is the two lowest online CPUs, and the pool set is the others.
- 3.2a: mk's worker thread keeps the pool set of its group thread.
- 3.2a: the group threads run on the pool set.
- 3.2a: the sets use logical CPUs and do not know SMT siblings.
- 4.2: a staging charge covers each host copy of engine state, up to one slot.
- 4.3: the admission of a profile keeps the live context and staging charges, or it refuses with the shortfall.
- 6.13: mk declares a checkpoint grain of 256 tokens.
- 6.13: the job path makes the interval copy as the next quantum, and a build restores the longest copy.
- 6.13: a refused charge or a put with no space skips the copy, and a client cannot pin a copy.
- 7.4: the wait for the opens is 300 s.
- 9.1: a log line is the message and `key=value` fields, with no time and no prefix.
- 9.1: `RUST_LOG` sets the log filter, and a filter that does not parse refuses the start.
- 9.1: the telemetry starts before the config is read, and after the inventory.
- 9.1: the resource attributes are five fixed names, and an unreadable one is left out with a warning.
- 9.1: the process drops the telemetry before it exits, and the drop waits at most 2 s for each signal.
- 9.1: the exporters run on a runtime of their own with one worker.
- 9.2: the HTTP span is named by axum's route template, and a status of 500 or more marks an error.
- 9.2: the `engine` of a group with several engines is the names joined with `+`, and `request` has a fixed set.
- 9.6: a decode quantum is not watched, and its failure line says so.
- 9.4: the labels of the metrics that exist (device group, tier, wait, gauges, lookups, restores, state).
- 9.5: `http.server.request.duration` runs from the request to the response head.
- 10.1: during the open, the process extends the start timeout by 30 s every 5 s.
- 10.3: the stop parks each state that holds tokens, and it needs the `slots` capability for rows.
- 10.3: the stop parks a held entry to NVMe with its own recipe, and it replaces an interval copy.
- 10.4: the binary ends through one function that calls `_exit(2)` after the flushes.
- 11.3a: every render uses the reasoning effort `medium`.
- 11.3a: thinking follows the context, and the renders differ only in the generation prompt.
- 11.3a: a read closes the thinking with the template's prompt for thinking off.
- 11.3a: a plain read keeps mkviolet's layout, with a leading space on the letter.
- 11.3a: a plain read tokenizes in one pass and checks that the tokens begin as the context does.
- 11.3a: a queue read writes the call in Qwen3.8's form with one `answers` object.
- 11.3a: a queue read asks the questions in the order of the schema.
- 11.3a: the workspace turns on `preserve_order` and `float_roundtrip` of `serde_json`.
- 11.3a: a failed tool result gets the prefix `Error: `.
- 11.3a: a system message after another role renders as a system turn.
- 11.3a: a marked message is a boundary, but a tool message inside a run is not.
- 11.3a: the template keeps the reasoning of every assistant message.
- 11.3a: `rendered_sha256` hashes the text that the read was conditioned on.
- 11.3a: the reply parser takes the stop token as a pushed token and puts early-phase text in no step.
- 11.3c: a dry run reports `recipe:` and the hex of the head's recipe when the store does not hold the head.
- 11.3c: a raw read, a decision read and a generation are each one job for each context.
- 11.3c: a decision read feeds its prefix once and writes a kept copy back for each later question.
- 11.3c: the usage of a read counts the shared prefix once.
- 11.3c: the reads of a decision run in request order, and `timeout_ms` is checked before each.
- 11.3c: a raw read lists 8 top tokens by default and at most 64.
- 11.3c: the text of a top token is its detokenized text, or U+FFFD for part of a character.
- 11.3c: the council `engine` is `nagaya:` and 16 hex digits of the identity digest.
- 11.3c: a generation holds its head after the reply when the slot is at most 8 tokens short.
- 11.3c: a build can start at a cut, and the cut stays among the boundaries of its state.
- 11.3c: the service adds the text a parser held back to the piece of the last token.
- 11.3c: the call ids are `call_<16 hex digits>_<index>`, from a hash of the prompt and the reply.
- 11.3c: a generation with no seed takes one from the operating system.
- 11.3c: the sample defaults are temperature 1 with no filter.
- 11.3c: a reply that reaches the context size ends with `context_full`, also when it reached `max_tokens`. gogi
  `a07d85e` changed this order on 2026-10-21.
- 11.3c: the decision limits are mkviolet's until the profile names others.
- 11.7: the process serves an engine only when its kind has a family library with the same tokenizer hash.
- 11.7: the served model is the engine's `model`, with 4096 tokens of context, or mk's maximum.
- 11.7: a process serves at most one model.
- 11.7: the specs, pins and snapshots routes use the one service, or give 503.
- 11.7: each service call runs on its own thread, at most 64 at once, and the 65th gets 429.
- 11.7: a request body has at most 16 MiB.
- 11.7: an unknown route is a 404, also for a known path with another method.
- 11.7: a request id is `req_<start time in hex>_<count>`.
- 11.7: a refused charge is 429 `busy`, and a job that a stop cancelled is 503.
- 11.7: an internal error is a 500, and the process stops with exit 1.
- 11.7: a stream waits for its first event before the status, and its queue holds 256 events.
- 11.7: a closed stream stops the generation, which holds no head.
- 11.7: a response does not report a control-text hit in a message of a context.
- 11.7: a stop sets every served model to `draining`, and the server waits at most 2 s for open requests.

**In gogi (`~/src/gogi/docs/council-api.md`, `thinking`)**

- A template may render a context the same with thinking on and off. Then both contexts have one head. gogi
  `c98fe1f` changes the conformance check to match.

**In `docs/engine-abi.md`**

- "v0 mk binding choices": mk declares `slots` and `coverage`, and `slot_clear` frees and allocates a context.
- "v0 mk binding choices": the factory plans from the config key `weights_bytes`, and a wrong size exits 3.
- "v0 mk binding choices": the pool is 12 contexts of 4,096 tokens and 4 of 32,768.
- "v0 mk binding choices": mk declares a checkpoint grain of 256 tokens.

**In `docs/deploy.md`, section "Decided by senpai 2026-10-20 night"**

- The engine kind in the profile is `mk-qwen38-gfx1151`, and the engine is named `mk`.
- `TimeoutStopSec` is 60 s.
- The unit makes the store directory with `ExecStartPre`.
- The gossip port is 8091, next to the HTTP port 8090.
- The pool is 112 GiB, the store's host budget is 16 GiB, and its NVMe budget is 50 GiB.
- `interval_copies` is 0, so the scheduler makes no interval copy.
- `MK_FAST_QMM` stays unset. Milestone 15 removed the library pin.
- The profile is copied into the store directory once, and it is not a link.
- The release binary is built with `--features hip`.
- `weights_bytes` is the size of mk's weight arena, and a wrong size exits 3.
- A probe error at start exits 3.

**On 2026-10-22, from Proposed notes**

- 3.2: a machine whose cards have their own memory has no pause rule for background CPU work.
- 11.3c: a build runs `lookup` and `restore` on the group thread, and the store sits behind a mutex.
- 11.3c: the first quantum of a build is a `copy` that chooses the slot and restores the longest prefix.
- 11.3c: a feed is `prefill` of at most `max_pass_inputs` tokens, or one block of a feed rule.
- 11.3c: a build seals each boundary that it passes, before it reads the parts.
- 11.3c: a build takes the lowest free slot that fits, or it is `too_large` or `busy`.
- 11.3c: a build lists the boundaries that the store holds when it ends.
- 11.3c: the store's owner key is the first 8 bytes of the BLAKE3 hash of the owner text.
- 12.1: the canary decides for a new build, and the identity digest does not cover `build`.
- engine-abi D1: the stage thread calls `wake` after 3 ms with no quantum, only while a job holds a lease.
