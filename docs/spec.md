# nagaya spec v0 (draft)

This spec says what the nagaya is and what it does, one rule at a time. It is written in Simplified Technical
English (`AGENTS.md`).

- `docs/ontology.md` defines each term once.
- `docs/engine-abi.md` defines the calls and the decisions D1-D11. The spec refers to them by number.
- `docs/lab.md` defines the lab routes, which look inside a model.

When this spec and another doc do not agree, correct one of them. Neither one wins without a decision.

Each rule here is in force. A rule that is not decided yet has the mark **Proposed**. An open question has the mark
**Open**, and section 15 lists all of them. The spec changes before the code changes. Git holds the history of each
rule.

## 1. Scope

The nagaya is an exokernel for megakernels. This repository holds its spec (this doc and `docs/engine-abi.md`), its
reference implementation (`crates/`), and the council v2 API that it serves (section 11).

The nagaya is one process on each machine. It owns the machine's GPUs, and engines run inside it. It decides three
things: where engines and state live, what runs next on each device, and what the store keeps. It does not decide
what a client asks a model.

| Layer | Owns | Does not own |
|---|---|---|
| kaijutsu (the agent) | coordination: casts, seats, judges, bumps, which model answers what | devices, memory, scheduling |
| nagaya | placement, the ledger, scheduling, the store, IO, machine telemetry, lifecycle, the client API | what a request means; kernels |
| engine | a model's code and weights, its kernels, its private layouts and caches | anything that another engine uses |

Goals for v0:

- Several engines run on one machine at the same time: inference endpoints with different models, image batch, and
  training as filler. All three kinds of engine are in v0. Thus the ABI must cover image steps
  and training before v0 is done.
- Each engine gives the same bits for the same identity and inputs.
- Nothing is lost or corrupted without a report.

Not in v0:

- One model split across machines. The links are gigabit today (README, "Between machines").
- Engines that load in place for a profile `PUT`. A `PUT` that adds or changes an engine still restarts the process
  (3.3, "Which actions run in place"). The load routes start a swap in place (3.3b).
- Cooperative yields inside a pass (D8).
- Tensor transfer between engines (13.3).

## 2. Machines, gossip and placement

**2.1 Inventory.** At start, the nagaya probes its own machine: its devices (amdgpu KFD, Intel Xe, or NVIDIA NVML),
its host memory, and its NVMe store path. It needs only a tiny bootstrap config (`nagaya.toml`) with its
bind addresses, store directory, and seed peer addresses.

| Machine | Devices | Memory |
|---|---|---|
| the Strix Halo machine | Strix Halo, one gfx1151 GPU | 128 GB unified |
| the B70 machine | two Intel Arc Pro B70, and an AMD Radeon 610M for the display | 64 GB GDDR6, 128 GB DDR5, 4 TB NVMe |
| a machine with unified memory | DGX Spark, one Blackwell GPU | 128 GB unified |

A capture of the tree of the B70 machine shows these facts:

- The CPU is a Ryzen 9 9950X with 32 online CPUs. Each CCD has one L3 domain: CPUs 0-7 and 16-23, then 8-15 and
  24-31 (`cpu*/cache/index3/shared_cpu_list`).
- The B70 cards are at PCI 0000:03:00.0 and 0000:08:00.0. The iGPU is at 0000:80:00.0 (`1002:13c0`, gfx1036). The
  firmware makes the iGPU the primary display.
- No BlueField-2 is in the machine. The NICs are `enp17s0` (igc), `enp18s0` (atlantic, 1 Gb/s) and a Wi-Fi card.
- Both cards have `local_cpulist` 0-31 and `numa_node` -1. Thus the machine has no NUMA locality to use (3.2b).

Telemetry takes hardware attributes from the probed inventory, not from constants in code. A CPU that runs work is a
device in the inventory too, with host memory as its tier. A machine with no GPU engine can also join the cluster to
serve as a coordinator, run CPU devices (13.2), or hold a store tier.

**The probe reads files, not a GPU runtime.** The inventory probe reads sysfs, procfs and `/dev` under a root path.
It does not open a GPU runtime to count devices. A call into a vendor library is allowed only when it opens no device
context. (DeepInfer reads NVML through `ctypes`.) The tests run the probe against fixed trees of the Strix Halo
machine, the B70 machine and a machine with unified memory.

**What the file probe reads.** `FileProbe` in `crates/nagaya/src/probe.rs` reads these files under its root path. The
root path is `/` on a machine, and a fixed tree in a test (`crates/nagaya/tests/fixtures/probe/`).

- Host memory: `MemTotal` in `proc/meminfo`.
- CPUs: the list in `sys/devices/system/cpu/online`. Each CPU number is below 1024, the CPUs that one `cpu_set_t` mask
  holds (3.2a). A list with a larger number is a probe error that names the file.
- AMD GPUs: the nodes of `sys/class/kfd/kfd/topology/nodes`. A node with `gpu_id` 0 is a CPU node. The device name
  comes from `gfx_target_version`, so 110501 gives `gfx1151`. The PCI address comes from `location_id` and `domain`.
- Intel GPUs: each `sys/class/drm/card<N>` whose `device/uevent` says `DRIVER=xe`.
- NVIDIA GPUs: each directory of `proc/driver/nvidia/gpus`, and its `information` file.
- NVMe: each `nvme<C>n<N>` in `sys/block`. The size is the `size` file in 512-byte sectors.
- Network: each interface in `sys/class/net` that has a `device` directory. Virtual interfaces have none.

A device id is the position in the list, in the order that the next paragraph decides.

The device ids follow the PCI address, in the order of domain, bus, device
and function, for every vendor. Reason: on the B70 machine the vendor order makes the display iGPU device 0, and the
B70 cards devices 1 and 2. The PCI order gives the cards 0 and 1, and the iGPU 2. A device id then does not change
when a driver loads in another order. The Strix Halo machine has one device, so it keeps id 0.

**Ignored devices.** The probe lists every GPU that it can size, also a GPU that drives the display, such as the iGPU of
the B70 machine. The bootstrap config key `ignore_devices` lists PCI addresses (2.1, "The bootstrap config"). These
rules apply to an ignored device:

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
starts.

**What the file probe cannot see.** A probe that opens no GPU runtime has these limits.

- NVIDIA: `proc/driver/nvidia` gives the model, the bus location and the minor number. It does not give the memory
  size. NVML gives it. Thus the probe accepts only the models in a short list whose memory is unified, and it reads
  their memory as the host memory. Any other model is an error.

  **Proposed (Gemini 4 Neon, 2026-11-22):** on GB10 hardware, the driver writes `Model: Unknown` in
  `proc/driver/nvidia/gpus/<addr>/information`. When `Model` is `Unknown`, the probe reads the PCI vendor and device id
  from `sys/bus/pci/devices/<addr>`. It accepts unified models in a short PCI-id list (`10de:2e12` maps to
  `NVIDIA GB10`) and takes their memory as the host memory. Any other model name or PCI id is an error that names both.
- Xe: sysfs has no file for the size of the device memory, and `xe` has no `tile0/vram0` size file. The probe reads
  the largest prefetchable memory BAR of the standard BARs 0 to 5, which are the first six lines of the `resource`
  file. The later lines are the ROM and the SR-IOV BARs of the virtual functions. With resizable BAR off, the BAR is
  small, so the probe refuses the card. Resizable BAR is a deploy requirement on the B70 machine.

  On the B70 machine, line 9 of `resource` holds 56 GiB: 7 virtual functions of 8 GiB. BAR 2 holds the 32 GiB of the
  card. A probe that reads every line takes the 56 GiB.
- AMD: the probe cannot tell unified memory from dedicated memory by one flag. It compares the KFD framebuffer
  memory with `mem_info_gtt_total` and `mem_info_vram_total` of the PCI device. Equal to the first means unified.
  Equal to the second means dedicated. Equal to neither is an error.
- Free memory, device clocks and temperatures. Telemetry reads those (section 9).
- The measured edges of 2.2. The probe gives devices and sizes only.
- The NVMe namespace that holds `store_dir`. The probe lists all namespaces.
- Intel cards with the `i915` driver, and any other driver. They are not devices of the nagaya.

Reason for the rules of "What the file probe reads" and "What the file probe cannot see": the probe must give a true
size or stop. A heuristic that guesses wrong without a sign
would corrupt the ledger (8.5).

The `dgx-spark` tree is captured from a real GB10 machine. Its `proc/driver/nvidia` file writes `Model: Unknown`,
with the GPU UUID zeroed.

The tree of the B70 machine is a capture. A Xe card reports the largest prefetchable range among its six
standard BARs. The ranges after them hold the ROM and the SR-IOV functions, and they are not memory of the card.

A build with the feature `fake` keeps a made-up
inventory. It has one device and 1 GiB of host memory. Reason: the host tests need no real machine. A build with none of
the features `fake`, `hip` and `xe` has no engine, so it refuses the start (exit 3, `engine_registry` in `main.rs`).

**The bootstrap config.** The bootstrap config is one TOML file, `nagaya.toml`. Its keys are `http_bind`,
`gossip_bind`, `store_dir`, `seed_peers`, `park_budget_secs` and `ignore_devices`. The first three keys are required.
By default, `seed_peers` is an empty list, `park_budget_secs` is 10 (10.3, ds4-recipe), and `ignore_devices` is an
empty list of PCI addresses (2.1, "Ignored devices"). The config of the B70 machine sets
`ignore_devices = ["0000:80:00.0"]`.

An unknown key is an error, so a misspelled key stops the process. There are no flags for these values, and the file
is the only source. Reason: two sources for one value can disagree.

The node advertises `gossip_bind` to its peers (2.3). An unspecified address, `0.0.0.0` or `::`, is an address that no
peer can reach. The process refuses it and exits 3, and the message names `gossip_bind`.

**An unspecified HTTP address.** Reason: the node publishes `http_bind` in
`server` and in `engine:<name>` (2.3), and a peer sends each call of the cluster to it (11.1a). The process refuses an
unspecified `http_bind`, `0.0.0.0` or `::`, as it refuses one for `gossip_bind`. It exits 3, and the message names
`http_bind`.

The bootstrap config has one more key, `node_name`. It is optional. Its default is the machine's host name. The name
is the `<host>` of the gossip keys (2.3), and gossip uses it as the node id. Each machine in a cluster needs its own
name.

A name has at most 128 bytes, so each gossip key that holds it fits 1024 bytes (2.3).

A name has letters, digits, `.`, `_` and `-`, and at least one character. Any other name stops the process with
exit 3.

Reason for `node_name`: a profile for `<host>` needs a name that a client knows. A test that runs two processes on
one machine sets two names.

The process reads the bootstrap config from the path in the environment variable `NAGAYA_CONFIG`. When the variable
is unset, it reads `nagaya.toml` in the working directory. The systemd unit sets the variable, and a test points it
at a temporary file. A file that does not read or parse stops the process with exit 3, and the message names the
path.

**2.2 Tier edges.** An edge records more than a bandwidth (after ds4-recipe):

- the rate for each read size and queue depth;
- whether direct IO works into the destination;
- the alignment that the edge needs.

The nagaya measures edges at install, stores the result in its local store index, and can measure them again. Each
measurement is data with a date.

**The edges of a machine with discrete cards.** The B70 machine has these edges, and the install measures each one:

- host to `device(i)`, and `device(i)` to host. Each B70 links at PCIe 5.0 x8. The first Gemma 4 engine measured about
  21.6 GB/s for a copy from host to device.
- `device(i)` to `device(j)`. The first Gemma 4 engine copies the residual through host memory. A peer
  copy is a separate edge with its own measurement.
- NVMe to `device(i)`, through host memory or direct. The edge records which path works (7.2).

The probe reads the link of a card from the upstream port of the card's own PCIe switch. On the B70 machine the ports
are 01:00.0 and 06:00.0, and each reports 32 GT/s x8. The card's endpoint reports 2.5 GT/s x1, because it sits behind
that switch. Thus a probe that reads the endpoint gives a false rate.

**Pinned host memory.** A copy over PCIe is fastest from pinned host memory. Thus a mandatory host term may set `pinned:
true`. The ledger charges its bytes in the host budget, as any host bytes. `nagaya.ledger.bytes` reports the term, so
the pinned bytes are visible. The first Gemma 4 engine pins a staging buffer of 4 MiB for uploads, and host buffers for
the routes.

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

**The state of an engine in gossip.** Reason: a swap changes the state of an engine while the process runs (3.3a), and a
client reads the state of the fleet from gossip. A loop of loads and unloads must not flood gossip. `chitchat` 0.13.0
has no control for a key that changes often: its failure detector watches nodes, not keys.

- The main loop compares the state of each engine with the state in `engine:<name>`, and it writes a changed state.
  Thus gossip shows `loading`, `ready`, `draining` and `unloaded` for each swap.
- It writes the key of one engine at most once each second. The newest state wins, so a state that changes and changes
  back within the second makes one write or none.
- It counts the writes of each engine in the last 60 seconds. At more than 10, it logs one warning that names the
  engine. Reason: this count is where a later rule for flaps starts. It takes no other action now.
- The start and the stop write their states at once (10.1, 10.3).

Each gossip value is compact JSON, and it has at most 8192 bytes. The nagaya refuses to publish a larger value. The
refusal names the key and the size. Reason: `chitchat` sends a delta in one UDP datagram of at most 65,507 bytes. A
node writes a few keys, plus one key for each engine.

A profile that does not fit in 8192 bytes needs a different path, which is not in v0.

A gossip key has at most 1024 bytes. The nagaya refuses to publish a longer key, and the refusal names the key.
Reason: `chitchat` 0.13.0 writes the length of a key in 16 bits, and a delta must fit in one datagram.

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
version.

A refused version counts as handled. The node writes the version to `profile.version`, and it keeps the old
`profile.toml`. After a restart, the node does not handle that version again, and it ignores each lower version.

**A duplicate node name.** Two live nodes can have one name. A chitchat id has three fields: the name, a generation
and the gossip address. A live peer is a duplicate when it has this node's name and is not this node's earlier run.
An earlier run has this node's gossip address and a lower generation. Thus a peer with another address is a
duplicate. A peer with this address and a generation that is not lower is a duplicate too.

For each duplicate, the node logs one error that names both ids as `name:generation:address`. It also writes the key
`name:conflict` under its own node, with the JSON value `{"own": "<id>", "peers": ["<id>", ...]}`. `peers` holds the
ids of the live duplicates, in sorted order. The node removes the key when no duplicate is live. It takes no other
action.

**The cluster of the two machines.** The Strix Halo machine and the B70 machine join one cluster, and each machine
serves the models of both. Each machine lists the gossip address of
the other in `seed_peers` (2.1). Each node publishes the HTTP address of each engine in `engine:<name>`, as the list
above says. 11.1a gives the rules of the forward.

**What a node publishes for the cluster.** Reason: a coordinator builds the identity of the cluster and forwards from
gossip alone (11.1a). Thus gossip must hold each served model, the HTTP address of each node, and the server limits.

- **The seeds.** A node that reaches no seed serves alone. It joins when a seed answers. A seed is only an address,
  and a node does not wait for its seeds before it serves.
- **`served`.** The value of `engine:<name>` gets the member `served`. It lists the served models of the engine, each
  in the shape of a gogi identity entry, without `status`. The state of the engine gives the status (11.1).
- An engine that is not a served model (11.7) has an empty `served` list. An engine on demand that its factory
  describes (11.1, "A text engine on demand") lists its model from the description.
- **`server`.** A new key `server` holds `{"http", "limits", "capabilities"}`. They are the HTTP address of the node,
  and the server limits and capabilities of its own identity. A node with no engine writes it too, so peers can reach
  its store.
- **The sizes.** An identity of the B70 machine with nine served models has 4,524 bytes. The largest entry has 738
  bytes (`embeddinggemma-2`). Each engine has its own key, so a key holds one entry, far below 8192 bytes.
- The largest value of a node is `profile:active`. The profile of the B70 machine, with nine engines, is 1,791 bytes as
  compact JSON (`deploy/b70-pair/profile.toml`, without its comments).
- **A value that does not fit.** At the start, a value of `engine:<name>` or `server` past 8192 bytes refuses the
  start (exit 3). The message names the key and its size. After the start, it stops the process (8.5).
- Reason for the stop: a served model that no peer sees is a silent fallback.
- Not built: `published` in `crates/nagaya/src/main.rs` only logs a warning, for any key.

**A served model name in the cluster.** A served model name is the id of a
model in the whole cluster, so two served models never share a name. A name is a `model` or an alias.

- A profile that adds a name that a live peer serves is refused, as a profile that does not fit is refused (2.4,
  3.3). The refusal names the model and the peer. The start of a node whose profile has such a name exits 3 (10.4).
- A clash can still happen, for example when two nodes join with profiles that were written apart. Then it is a
  configuration error. Each node logs one error that names the model and both nodes.
- Until a client renames one, the node with the lower node name, in byte order, serves the name. Reason: a node name
  does not change at a restart, and each node finds the same keeper from its own view of gossip.
- The other node writes the key `model:conflict` under its own node, with the value `{"models": [{"model",
  "kept_by"}]}`, sorted by `model`. It removes an entry when the clash ends, and the key when no entry is left.
- It does not list its copy, and it forwards each request for the name to the keeper (11.1a). Its engine stays as it
  is, and its `engine:<name>` key stays.
- Replicas behind one name are a later design.
- An engine name is unique in one profile only (2.4). Two nodes can each have an engine `gemma` (3.3b).

**The check of a name.** Reason: the ledger knows no served model name
before an engine opens, and the rule above needs the names of a new profile.

- A factory gives the served model names of an entry with no open and no read of a file (`Factory::served_names`). An
  image, speech or embedding kind gives the engine name or its fixed model. The kind `fake-toy` gives its key `model`.
- A kind that cannot give them gives none. Today the Qwen3.8 engine and Qwen-Image give none, so the check of a profile
  skips their entries. Only the start check finds a clash of their names.
- The check of a profile reads the names with the plans of the ledger check (2.4). A name is added when no entry of the
  active profile gives it.
- A refused local `PUT` is a 400 `invalid_request` with the param `profile`, because the profile breaks a rule of 2.3.
  A refused desired profile writes `profile:refused` with that reason and no shortfall.
- The start check compares the served models of the council with the live peers, before the node publishes them. The
  node does not wait for its seeds, so a peer that it does not see yet makes a clash of the rules above.
- A clash is not a 507, as for a profile that does not fit. A 507 names a shortfall in bytes, and a clash has
  none.

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
adds the entries does not count a pooled budget twice. On the Strix Halo machine, `device(0)` and `host` are one pool
(4.2).

The placement profile is one TOML file. Each `[[engines]]` entry has `name`, `kind`, `devices`, `config`, an optional
`family` table (11.3) and `share`. An entry can also have `residency` and `pinned` (3.3a).
Each `[[budgets]]` entry has `tiers` and `bytes`, and a tier is
`host` or `device(i)`. The `[store]` table has `nvme`, an optional `host` (4.1a), an optional `interval_copies`
(6.13), 0 by default, and an optional `objects` (6.15).

The key `pin_identity` is gone. A profile that names it does not parse, because the canary holds the bits of an engine
(8.1).

**One name for each device.** The spec names a device in four places.
Each place uses one of two names, and this map joins them.

- A **machine device id** is the id of 2.1. The profile uses it: `devices = [0, 1]`, and a budget tier
  `device(0)`. The metric attribute `tier` writes it as `device0` (9.4). The fd names of 10.2 use it.
- A **group index** is a position in the profile's `devices` list of the engine. The ABI uses it: a term tier
  `device(i)` (4.2), `limits().stages[].device`, and the devices that `open` gets (engine-abi "Core").
- The ledger maps a term tier `device(i)` to the machine device `devices[i]` of the engine. Then it finds the budget
  of that machine device. Thus an engine never sees a machine device id, and a profile never sees a group index.
- The list can have any order. With `devices = [1, 0]`, group index 0 is machine device 1. Thus stage 0 of an engine
  with stages runs on machine device 1. It uses the device thread and the CPU set of that card (3.2, 3.2b).
- On the Strix Halo machine, both names of the one device are 0.

`limits().devices` holds group indexes, `0` to `n - 1` in order. The
process checks that `n` is the length of the profile's `devices` list, as `check_devices` checks the list today
(`crates/nagaya/src/placement.rs`). The engine finds each physical card by the PCI address in the devices that
`plan` and `open` get. Reason: today `limits().devices` holds machine ids, so an engine on machine device 1 must know
its machine id. That breaks the rule above, and an engine plan would differ between machines for the same config.

An engine name has letters, digits, `.`, `_` and `-`, and at most 128 bytes. A profile with another name does not
parse. Reason: the name is part of the gossip key `engine:<name>` (2.3).

The process writes an accepted profile to `<store_dir>/profile.toml`, atomically: a temporary file, then a rename.
The file has the same shape as the input, so a read gives an equal profile. An unknown kind or key refuses the
profile before any engine opens. Reason: TOML is the engine config format, so one format is easier to read and to
check.

**A profile for another host.** A profile `PUT` for another host publishes
`profile:desired:<host>` under the node that takes the `PUT`. The target handles the key by the rules of 2.3.

**The routes of a profile for another host.** Reason: the path is gossip, so the coordinator cannot wait for the
answer of the target.

- The query `host` names the target: `PUT /nagaya/v1/profile?host=<name>&version=7`. The body is the profile as JSON,
  as for a local `PUT` (3.3).
- A `host` equal to the name of this node is a local `PUT`. A `version` with it is a 400 `invalid_request` that names
  `version`.
- With another host, `version` is required, a whole number above 0. A missing or bad `version` is a 400 that names
  `version`. A `host` that no live node has is a 404 `not_found` that names `host`.
- The coordinator checks only that the body is a JSON object. Reason: the coordinator may lack the engine kinds of the
  target, so the target checks the rest.
- A `version` that is not higher than the version of each live `profile:desired:<host>` is a 409 `conflict` (11.5).
- Otherwise the coordinator writes the key and gives a 202 `{"host", "version"}`. The result comes later. The
  target's `profile:active` changes, or its `profile:refused` names the version (11.4a, "The cluster route").
- The target handles a desired profile with the actions of 3.3, as for a local `PUT`. A version that comes while a
  swap waits is refused with the reason `busy`. It counts as handled, so the client sends a higher version.
- This rule replaces the stop on each change of 2.3, which `desired_profile` in `crates/nagaya/src/main.rs` does today.
- `GET /nagaya/v1/profile?host=<name>` gives the `profile:active` of that live node, from gossip. A `host` that no live
  node has is a 404 that names `host`.

## 3. Process

**3.1 One process per machine.** The nagaya owns every GPU on the machine. The nagaya is the only
GPU client. An ignored device, such as a display GPU, is the one exception
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
  have one device, and the group of the Gemma 4 engine has both B70 cards. Engines whose device sets share a device,
  also through a chain of shared devices, are in one group with the union of their devices. A group has one thread and
  one scheduler. Every engine call for a group comes from its thread, and the call is synchronous.
- **One device thread per device of a group with more than one device.** An engine with one device opens on the thread
  of its device, and its quanta run there. Thus engines on separate cards of one group run at once. Reason: Gemma 4
  spans both B70 cards, so each engine of the B70 machine is in its group. The image engines on the two cards must run
  in parallel (11.9). This extends the rule of engine-abi "How a group loads both" to a group with no loaded engine with
  stages.
- **One stage thread per device of a staged engine.** An engine that declares `stages` runs each stage on the device
  thread of its device (D1, and the next rule). The group thread runs the scheduler and sends each stage quantum to the
  thread of its device. Each call that touches a device comes from the stage thread of that device. The thread layout is
  in engine-abi D1.
- **The device threads live as long as the group.** A
  group of more than one device starts its device threads once, and they end when the group stops. At its open, the
  engine with stages gives each stage handle to the thread of its device. Its unload takes the handles back. Thus the
  stage thread of a device is its device thread. Reason: an engine with one device then stays on its thread through each
  swap of the engine with stages (3.3a, "Explicit layout").
- **A bounded CPU worker pool.** Rendering, tokenization, hashing, safetensors conversion, and CPU device passes (13.2)
  run on a bounded pool pinned to the remaining cores at `SCHED_BATCH`. Long CPU work yields between chunks to obey
  `max_quantum_ms`. On unified-memory machines, such as the Strix Halo machine, the scheduler pauses background CPU
  memory work during a GPU pass to protect GPU memory bandwidth.
- **The row threads.** After a pass that several
  jobs share, each row has host work that uses no engine. That work is the read of its logit row and the draw of its
  token. The thread that ran the pass gives that work to its row threads, and it waits for all of them. Then it ends
  each job in the order of the segments, as before.
  - A thread that runs shared passes has 8 row threads (`nagaya-row-<index>`). They start at its first shared pass.
  - A row thread keeps the CPU mask of the thread that started it. Thus the row threads of a device thread stay on
    the CPUs of its card (3.2b).
  - A row thread has the normal scheduling policy, not `SCHED_BATCH`. Reason: the card waits for this work.
  - The draw of a row is the same function of the same inputs on each thread (11.2). Thus the bits do not change.
  - Each call of an engine stays on the thread of its device.

  Reason: when one thread draws each row in turn, the card waits. With 8 rows and no filter, that is 22 ms of each
  step of 46 ms.
- **No pause on a discrete card.** A machine whose cards have their own memory, such as the B70 machine, has no pause
  rule. A pass there reads GDDR6, not host memory. The host traffic of a pass is the residual and the uploads, about
  11 MB for a block of the first Gemma 4 engine. A measurement under load can add a rule.

**3.2a Control runtime first.** The control runtime (`tokio`) runs gossip and the HTTP server. The group threads, the
ledger thread and the watchdog thread stay std threads. They do not use `tokio`.

The isolation has a test. The test reads the CPU mask of each thread with `sched_getaffinity`. It checks that the
control threads do not use the group cores.

**CPU sets.** The nagaya splits the online CPUs into a housekeeping set and a pool set. The bootstrap
config names no core sets, so the split is one fixed rule.

- The housekeeping set holds the two lowest-numbered online CPUs.
- The pool set holds every other online CPU.
- The two sets do not overlap. Together they hold every online CPU.
- The pool size is the number of CPUs in the pool set. The pool has one thread for each CPU in the set.
- Offline CPUs are in neither set. CPU numbers can have holes, so the nagaya does not assume `0` to `N-1`.
- A machine with fewer than three online CPUs cannot split the sets. The process exits 3 (10.4) when it
  applies the split.
- The online CPU list must be sorted and strictly increasing. Another list is an error.

**Where each thread pins.** Each thread pins itself to its set when it starts. A thread that cannot pin
is an error: a worker thread panics, and the process exits 1 (8.5). A thread of the control runtime aborts the
process. Reason: a hook of the runtime cannot return an error, and a thread that runs on the wrong cores breaks the
isolation.

- The control runtime (every worker and blocking thread, `nagaya-control`), the ledger thread (`nagaya-ledger`) and the
  watchdog thread (`nagaya-watchdog`) pin to the housekeeping set. The telemetry exporter thread (`nagaya-otlp`, 9.1)
  starts before the CPU sets are built, so it keeps the process's initial CPU mask.
- The group threads (`nagaya-group-<index>`) pin to the pool set, with the normal scheduling policy.
- The store thread (`nagaya-store`) pins to the pool set at `SCHED_BATCH`. It hashes and converts bytes, which is pool
  work.
- A start with no CPU sets pins no thread. Only a test starts that way. The binary always makes the sets (below).
- An engine's own threads keep the mask of the group thread that starts them. Linux gives a new thread the mask of the
  thread that starts it. The Qwen3.8 engine starts its worker thread (`qwen38-engine`) in `open`, on the group thread,
  so the worker runs on the pool set. Reason: the
  process crate makes the sets, and the group thread is pinned before it opens an engine. Thus the factory needs no CPU
  argument.

The binary reads the online CPUs through the file probe (2.1) and makes the sets before it starts gossip. A probe
error or a split error is a refusal to start: exit 3 (10.4).

The group threads run on the pool set. A group thread
drives the GPU, and it does host work between passes: it builds segments, copies bytes and reads taps. The host work is
the same kind of work as the pool does, and it needs whole cores. A group thread on the housekeeping set would compete
with gossip and the HTTP coordinator. A stalled gossip round is worse than a slow pass.

The pool has one thread for each pool CPU, and a group thread uses one more. This oversubscription is small. A
measurement on the Strix Halo machine can change it.

The sets use logical CPUs, and the split does not know SMT siblings. A housekeeping CPU can share a physical core with a
pool CPU. Reason: on the Strix Halo machine, CPU 0 pairs with 16 and CPU 1 with 17 (`thread_siblings_list`). Thus the
two housekeeping CPUs sit on two physical cores. A rule for other machines would only guess before a measurement. The
bootstrap config can name the sets later.

**Open:** on gfx1151, a measurement showed that two of the four `qwen38-engine` threads had the mask 0-31. The worker
and one other thread had the pool set, 2-31. The HIP runtime starts the two other threads and sets their masks itself.
The nagaya does not pin threads that a runtime starts.

**Open:** 3.2 asks for elevated priority on the control threads. The nagaya does not apply it yet. A raised priority
needs `CAP_SYS_NICE` or a setting in the unit.

**3.2b CPU sets for each device.** 3.2a puts every group thread in one
pool set. A stage thread (3.2) drives one card, so it gets a set of its own.

- A stage thread pins to the CPUs of the card's `local_cpulist` that are in the pool set.
- A card whose `local_cpulist` holds every online CPU has no locality. Then the nagaya gives each such card one L3
  domain (`cpu*/cache/index3/shared_cpu_list`), in the order of the device ids. With more cards than domains, the
  cards take the domains in turn.
- On the B70 machine, card 0 gets CPUs 2-7 and 16-23, and card 1 gets CPUs 8-15 and 24-31. CPUs 0 and 1 are the
  housekeeping set.
- The group thread, the CPU pool and the store thread keep the pool set (3.2a).
- A machine with one card has nothing to separate. Its card keeps the pool set, and the nagaya reads no CPU list.
  Reason: the Strix Halo machine keeps the behavior of 3.2a.
- A card with a `local_cpulist` that holds a part of the online CPUs takes that part, less the housekeeping set.
- A card whose set holds no pool CPU is an error. The start is refused: exit 3 (10.4).
  Reason: a stage thread on the wrong CPUs breaks the isolation.
- The L3 domains come in the order of their lowest CPU.

Reason: the two CCDs of the 9950X have their own L3. A stage thread that stays on one CCD keeps its host buffers in
one cache.

The Level Zero runtime and IGC start threads of their own, as the HIP runtime does (15 item 14). Linux gives a new
thread the mask of the thread that starts it. The engine calls `zeInit` on
the group thread, so those threads start with the pool set. The nagaya does not pin a thread that a runtime starts.
**Open:** measure the masks of the Level Zero threads on the B70 machine, as on gfx1151.

**3.3 Engine lifecycle and profile swaps.** The nagaya opens each engine at start, from the profile. The HIP runtime
cannot end inside a process: its teardown segfaults. Thus a profile change that must free such an engine persists the
profile and stops as in 10.3. The nagaya keeps its arenas in the file-descriptor store (10.2) and exits with code 4, and
systemd starts it again. An engine whose `close()` is safe loads and unloads in place (README door 1, and the rules
below). The Qwen3.8 engine has a `close`: it frees its weights and keeps the HIP runtime (engine-abi "The close of the
Qwen3.8 engine").

**A profile change in place.** Reason: a client sets the work of the fleet with one `PUT /nagaya/v1/profile` to each
machine. A named set of profiles, such as agent time or live performance,
is the client's policy. The nagaya keeps no such name.

- **Admission first.** The nagaya admits the new profile as a whole (2.4). A refusal is a 507 with the shortfall,
  and nothing changes.
- **Compare by name.** The nagaya compares each `[[engines]]` entry of the new profile with the active entry of the
  same name.
- **The actions.** Each engine gets one action:
  - `keep`: the entry is the same, or only its `share` changes. A new share takes effect through admission (4.4).
  - `residency`: only `residency` or `pinned` changes. A resident engine that is not loaded loads when it fits beside
    the loaded engines (3.3a, "Explicit layout"). An engine that becomes on demand stays loaded until an unload.
  - **One try.** The change tries the load
    of a resident engine once. When it does not fit, the engine stays unloaded and the log names the shortfall. A client
    loads it later with the load route (3.3b). Reason: a retry by itself is a swap that no client asked for.
  - `load`: the entry is new. It loads if it is resident.
  - `unload`: the entry is gone.
  - `reload`: `kind`, `devices`, `config` or `family` changes. The engine unloads, then it loads with the new entry.
- **Swaps do the work.** Each load and unload is a swap of 3.3a, on the group thread of the engine. Each job keeps
  its deadline.
- **A restart only when it must.** When an action unloads an engine without `close`, the nagaya persists the profile
  and restarts as above (exit 4). A change of the `[store]` table also restarts.
- **The response** is 202 after admission, with each engine and its action. The swaps run after the response, and
  the status feed shows each state (9.4). The status feed is not built yet (9.7), so the identity (11.1) and
  gossip show the states today.

**Which actions run in place.** `keep`, `residency` and `unload` run in place. `load` and
`reload` restart the process (exit 4). Reason: the engines of a device group are fixed at start, so a new entry has no
group thread.

**The profile routes.**
Reason: a client needs a fixed shape for the change of 3.3, and for a read of the active profile.

- `PUT /nagaya/v1/profile` takes the profile as JSON, in the shape of `profile.toml`. It changes only this machine.
  A `PUT` for another machine follows 2.4, "The routes of a profile for another host".
- A profile that does not parse, or that breaks a rule of 2.4 or 3.3a, is a 400 `invalid_request` with the param
  `profile`. A profile that does not fit is a 507 with the type `too_large`, the param `profile` and `shortfall`.
- A profile equal to the active profile is a 200, with the action `keep` for each engine.
- An accepted profile is a 202 with `{"actions": [{"engine", "action"}], "restart"}`. With `restart` true, the
  process persists the profile and stops with exit 4.
- A change of `[[budgets]]` runs in place: the ledger admits the loaded engines with the new budgets.
- An engine that unloads in place stays in its group as a retired engine until the next start. A load of it is a 404,
  and its served model stays in identity with the status `unloaded`.
- A `PUT` while a change in place runs is a 409 `invalid_request` with the param `profile`.
- `GET /nagaya/v1/profile` gives the active profile as JSON.

**Swaps in place.** An engine whose `close()` is safe loads and unloads in
place, with no restart. The placement profile chooses the devices of each engine.

**Explicit loads.**

- A request never starts a load. A request for a served model that is not `ready` gets a 503 `model_unavailable`, with
  `model` and `status` (gogi).
- A client loads and unloads an engine in two ways: a profile `PUT` (above), and the load routes (3.3b).
- The nagaya never loads or unloads an engine by itself after the start (3.3a, "Explicit layout").
- The route `POST /v1/audio/speech` goes, because speech is a council served model (11.8a).

Reason: a load takes seconds, and it can unload another engine of the group. When requests start loads, the cards can
churn between engines. The client knows its plan, so the client loads.

Reason for swaps in place: voice and image engines can use the B70 cards when Gemma 4 is idle.

**3.3a The swap in place.** Reason: these rules make each swap one choice on one thread.

- **Residency.** Each engine in the profile has `residency`: `resident` or `on_demand`. The default is `resident`.
  At start the process opens each resident engine, and an on-demand engine stays unloaded.
- **Pins.** An entry can have `pinned = true`. A pinned engine is resident: admission refuses a pin on an on-demand
  entry. No load and no unload removes a pinned engine. Only a profile `PUT` does (3.3a, "Explicit layout").
- **Only engines that close.** A profile refuses an on-demand engine unless each engine of its device group declares
  `close` (engine-abi "Capabilities"). Reason: a swap frees an engine in the process, and only `close` makes that
  safe.
- **Admission.** The resident engines must fit together (2.4). Each on-demand engine must also fit with the resident
  engines of the other device groups. Reason: in the worst case, a swap unloads every other engine of the group.
- **One thread decides.** The group thread of an engine runs its swaps. A load (3.3) names its engine, and the group
  thread starts the swap.
- **The engines that unload.** A swap unloads only the engines that its load names (3.3b). The group asks the ledger
  whether the engine fits beside the engines that stay.
  When it does not fit, the load is a 409, and nothing changes. Engines of other groups never unload for a swap.
- The ledger takes a load when the group plans it, before any engine unloads. Until the swap ends, a later context or
  staging charge must fit the ledger now and the ledger after the swap. The engines that unload keep their bytes until
  their drop, and the ledger charges the swap after the drop. A profile change waits for the drop of each engine that it
  retires. Reason: a late charge can stop the process
  after a 202, and bytes freed too early let the staying engines over-commit.
- Swaps of two groups can wait at one time, and they can end in either order. A swap is charged after its own unloads,
  beside any set of other swaps that ended first, each whole with its load and unload. Each such state must fit. A swap
  that only unloads is never refused. Reason: a check of the end state can let the first swap find no room. A
  load counted without its unload can refuse an unload.
- A profile `PUT` while any swap waits or runs is a 409 busy, and nothing changes. The client sends it again later.
  Reason: a profile checked
  beside a waiting swap could fail after that swap.
- After the nagaya accepts a change, a load is a 409 of the same kind until the ledger takes the new profile. The ledger
  takes it when no swap waits, and the engines that the change removes dropped and hold no context. Reason: a load in
  that time could break the new profile, and a late context can stop the process.
- Until the ledger takes the new profile, each context and staging charge must also fit it. A charge for an engine that
  the change removes is refused as not ready, a 503 `model_unavailable`. The log names the wait of the change at its
  start and every 10 s. Reason: a charge that fits
  only the old profile can stop the process, and steady traffic to a removed engine can hold the change off.
- Known limit: a swap that only unloads is charged with no check. Fewer engines fit, but with elastic terms the spare of
  the staying engines can shrink under live contexts. No engine declares elastic terms today.
- **The wait before an unload.** An engine that must unload takes no new job. Its jobs that the group holds run to
  their end. Jobs of engines that stay loaded go on. A swap of an engine with stages is the one exception: new jobs
  of the staying engines wait for it (see "No moves" below).
- **Single-card swaps.** A swap of an engine on one device runs on that device's
  thread, not on the group thread. The ledger still decides each swap, on its one thread. The group thread keeps the
  swaps of an engine that spans devices, such as Gemma 4 on both cards, and that stall is accepted. Reason: a 244 ms
  image swap then no longer waits on the other card. The cost: two threads run swaps, so the ledger check orders them.

**A job for an engine that is not ready.** Reason: the explicit loads of 3.3 remove the wait for a load.

- The group thread decides. A job for an engine that is not `ready` ends at once, and nothing runs. Its request gets a
  503 `model_unavailable`.
- An engine is not ready when it is unloaded, or when a planned swap loads or unloads it.
- A request checks the state of its served model before it charges a context (3.3). The check of the group catches a
  swap that the group planned after that check.

**The steps of a swap.**

1. Unload each engine that must go. Park its states, as the stop does (10.3 step 3), in the park budget.
2. Release the charges of its contexts. Remove its arena fds from systemd's store (`FDSTOREREMOVE=1`, 10.2).
3. Drop the engine, which closes it. Admit the profile of the engines that stay loaded, so their shares grow (4.4).
4. Admit the profile of the loaded engines with the new engine. Open the new engine cold, and export its arena (10.2).
5. Set the new engine to `ready`. Its served models then take requests.

An open that fails during a swap is a crash (8.5): the process exits 1, and systemd starts it with the resident
engines. Reason: the displaced engines are unloaded, and no rule says which engine serves next. A crash restarts in
a known state.

Reason for the fd removal: an fd in systemd's store holds the device memory of its arena. An engine that closes while
its fds stay in the store frees no device memory.

**The ledger of a swap.**

- The ledger admits the set of loaded engines, and the ledger thread keeps that set.
- An engine that is not loaded stays in the ledger with its context terms only, and with a share of 0. It takes no
  memory and no part of the spare memory (4.4).
- Thus the terms of its contexts stay admitted while it is unloaded (4.2).

Reason: a council request charges its context on the request thread, before it submits its job (11.3c). Without this
rule, a generation for an unloaded Gemma 4 fails with "engine gemma is not admitted".

**Explicit layout.**

- Engines load and unload only through the API: the load routes (3.3b) and a profile `PUT` (3.3). The start opens
  the resident engines (above).
- No idle time unloads an engine, and no unload loads another engine. A profile that names `idle_unload_s` is a 400
  that says the key is gone.
- A load that does not fit beside the loaded engines is a 409. It names the shortfall in bytes for each short budget,
  the loaded engines in the way, and each pinned engine. The nagaya never chooses an engine to unload.
- A load can name the engines to unload first: `{"unload": [names]}`. The group checks the whole swap before it
  unloads any engine. A swap that still does not fit is a 409, and nothing changes.
- A pinned engine never unloads for a load or by the unload route. A load that needs its memory is a 409 that names
  the pin.
- **No moves.** A swap of an engine with stages unloads only the engines that its load names, as every swap does. Each
  other engine of the group stays loaded on its device thread (3.2, "The device threads live as long as the group"). A
  load of an engine with stages that does not fit is a 409, as for any load. Reason: a move can take a pinned engine
  away for seconds, and a later swap can unload it until a restart.
- The group thread opens and closes the engine with stages. It waits until the group holds no job, and the jobs of
  the other engines wait during the open. Reason: the scheduler of the group runs on that thread.
- While such a swap waits, a new job of an engine that stays waits too. It runs after the swap, and it never fails for
  the swap. Thus the jobs in flight end, and a steady stream of jobs cannot hold off the swap. Reason: steady embeddings
  can keep the group busy, so that a load of Gemma 4 never runs.
- A resident engine can share a device group with a resident engine with stages. At start, it loads beside that
  engine after its open, as a load does.
- A profile change that makes an engine resident loads it when it fits beside the loaded engines (3.3). Otherwise the
  engine stays `unloaded`, the log names the shortfall, and a client loads it with an `unload` list.

Reason: a client agent knows its plan, so it chooses the layout. A layout that changes by itself surprises it.

**What a swap does not change.** A swap changes no device group, group thread or route. An unloaded engine is in its
group, with the state `unloaded` (9.4, 11.1).

**Council models on demand.**

- Any served model can be on demand. Identity lists an unloaded model with the status `unloaded`.
- A request for it is a 503 `model_unavailable` (3.3).
- Its `weight_hash` and `tokenizer_hash` come from its first open since the start, and they are empty before (11.8a).

**3.3b The load routes.**

- `POST /nagaya/v1/engines/{name}/load` loads the engine of the profile with that name, by a swap (3.3a). The reply is a
  202 when the swap starts, and it names the engines that unload. A loaded or loading engine gives a 200.
- The body of a load is empty, or `{"unload": [names]}` (3.3a, "Explicit layout"). A name of another device group, the
  engine itself, or a malformed body is a 400 `invalid_request` with the param `unload`. An unknown name is a 404 that
  names it.
- A name that comes twice in `unload` is a 400 with the param `unload` too. The checks of the list run before the 200 of
  a loaded engine. Reason: a repeat can crash the
  ledger thread, and a loaded engine can hide a bad list behind a 200.
- A load that does not fit is a 409 that names the shortfall, the engines in the way and the pins (3.3a).
- **The memory check of an open.** The ledger holds byte targets of
  the profile, not a measurement (4.1). An engine can also check the free memory of the machine at its open. A load
  that such a check refuses is a 409 of the same kind, and the group answers it before the swap starts.
  - The factory of an engine kind can answer "the problem of an open now, if any". It gets the bytes that the unload
    list frees. The default answer is none.
  - The group asks it on its own thread, before it gives the swap to the ledger. A problem is a refusal that
    changes nothing. It names the bytes needed, the bytes available, the floor and the engines in the way.
  - The freed bytes are the sum of the mandatory terms and the minimum elastic terms of the engines that unload,
    in every tier. The group cannot know the real bytes that an unload returns to the machine.
  - The sum is an upper bound, so the check never refuses a load that would have worked. A load can pass the
    check and still fail at its open, for example when another process takes memory in between. That open still
    panics (8.5), because a silent fallback after an unload would hide a corrupt state.
  - Reason: a load with an empty `unload` list can open the Qwen3.8 engine beside a resident engine. Then the
    open finds less memory than it needs with its floor, the process exits, and systemd starts it again.
- `POST /nagaya/v1/engines/{name}/unload` unloads a loaded engine that is not pinned, resident or on demand. No other
  engine loads. An engine that is not loaded gives a 200.
- An unknown name is a 404 `not_found` that names `name`. A pinned engine, or an engine without `close`, is a 409 that
  names `name`. A pinned engine leaves only by a profile `PUT`.
- A load does not change the profile. The next start of the process loads the resident engines only.
- A load or an unload that arrives while the group has a planned swap waits for the end of that swap. Then the group
  plans it.
- The body of a 202 is `{"engine", "unload", "load"}`: the name, then the engines that unload and load, in the order
  of the swap. The body of a 200 is `{"engine", "status"}`, with the state of the engine.
- A 409 has the type `invalid_request` and the param `name`, because gogi has no type for a conflict. A stop gives a
  503 `unavailable`.
- The status feed and the identity show each state: `loading`, `ready` and `unloaded` (9.4, 11.1). The status feed
  is not built yet (9.7), so only the identity and gossip show them today.
- **Any node.** A load or an unload of an engine on any node goes to the node
  that holds the engine.

**The release of a device.** A route unloads a card, so a card test
runs on it while the other cards serve. Reason for each choice: it is the simplest one that
keeps the rules of "Explicit layout" and adds no key to the profile.

- `POST /nagaya/v1/devices/{id}/release` unloads each loaded engine whose device list names the device. No engine
  loads. An engine with stages on two devices unloads whole. The reply is a 202 with `{"device", "unload"}`: the id,
  then the engines that unload, in the order of the swap. A device with no loaded engine gives a 200.
- A released device takes no load. A load of an engine that names it is a 409 that names the device, until a
  reclaim. The device keeps its budget and its thread, and it runs no work.
- `POST /nagaya/v1/devices/{id}/reclaim` ends the release, and it loads each resident engine that names the device.
  The reply is a 202 with `{"device", "load"}`. A device that is not released gives a 200.
- A reclaim whose load is refused leaves the device released. The next reclaim tries the load again.
- A release refuses a device that holds a pinned engine, or an engine without `close`: a 409 that names the engine.
  A pinned engine leaves only by a profile `PUT`, as before.
- The release is not stored. The next start of the process loads the resident engines, as the profile says. Thus a
  crash does not leave a device empty, and the profile gets no new key.
- A device that no engine of the profile names is a 404 `not_found` that names `id`. An ignored device is such a
  device (2.1).
- The identity shows each engine of the device as `unloaded`.
- The release holds the engines that the profile names on the device at that time. An engine that a later profile
  `PUT` adds to the device is not held.
- Not built: the query `host`, and a mark of the release in the gossip key `inventory`. The routes act on the node
  that gets the request. These routes get no forward. A client reads the HTTP address of each
  node from `GET /nagaya/v1/cluster`, which gossip fills, and it calls the node that holds the device.
- A card test names its cards in `NAGAYA_TEST_CARDS` (`AGENTS.md`). It does not read the release.

**The node of a load route.** Reason:
an engine name is unique only in one profile (2.4), so two nodes can each have an engine `gemma`. A load must name one
engine.

- The query `host` names the node: `POST /nagaya/v1/engines/gemma/load?host=<name>`. A `host` that no live node
  has is a 404 `not_found` that names `host`.
- Without `host`, the route takes the engine of the local profile. Without one there, it takes the one live peer that
  publishes `engine:<name>`.
- Two or more such peers is a 400 `invalid_request` that names `host`, and the message lists the nodes. No such peer
  is the 404 of an unknown name.
- The coordinator forwards the request to that node by the rules of 11.1a, with its body. The response comes back as
  it is.
- The names in an `unload` list are engines of the target node.
- A `host` equal to the name of this node is a local load, with no forward.

**Open:** Do the GPU queues of an idle engine slow the other engines on its device? The Qwen3.8 engine has a note that
its idle queues slow another process. The source of that note is unknown. A measurement answers the question.

## 4. Ledger

The ledger is the machine's account of memory (ontology "Ledger", D9, ds4-recipe `05-memory-ledger.md`). The
ledger thread owns it (3.2).

**4.1 Budgets.** Each tier has one budget. A budget is a byte target in the placement profile (2.4). It is not a
measurement of free memory.

- A machine with unified memory, such as the Strix Halo machine: one shared pool for device and host memory.
- The B70 machine: one budget for each card's device memory, and one for host memory.
- Every machine: one budget for the store's host-memory tier, and one for the store on NVMe.

**4.1a The store's budgets.** The placement profile names the store's NVMe budget in bytes. The profile may name the
store's host budget. Without it, the host budget is the host-only budget's unassigned bytes (4.4 step 4). A profile
with an engine that names no NVMe budget is refused at start (10.4, exit 3). A profile with no engine may omit the
store section: it parks nothing.

Reason: a store write must not take bytes that an engine needs. An unbounded NVMe budget can fill the disk with no
warning. The owner quota is unbounded, and the chain limit is open.

**4.2 Terms.** Each engine declares its terms in `limits().memory`. `plan(config)` gives the same terms before
`open` (2.4).

- Mandatory terms `{name, tier, bytes, lifetime, kinds}`. The lifetime is per engine, per context, or loaned (one
  holder at a time). The Qwen3.8 engine declares its whole pool as one term with lifetime `engine`. The ledger reads the
  lifetime like this.
  - `open` charges each `engine` term and each `loan` term once. A loan has one holder, so one copy is enough.
  - `open` does not charge a `context` term. Each `ctx_new` charges the bytes of every `context` term of its
    engine, in the tier of the term.
  - `ctx_new` takes the bytes from the unassigned bytes of the budget (4.4 step 4). If they are too few, the
    ledger refuses and changes nothing (4.3). The reclaim of 4.6 is the caller's step.
  - `ctx_fork` and `ctx_copy` make a context too, so each charges as `ctx_new` does.
  - The end of a context releases its bytes.

  Reason: this is the smallest rule that gives the lifetime a meaning. No engine declares a `context` term today,
  so the Qwen3.8 engine does not change.
- Elastic terms `{name, tier, unit_bytes, minimum, maximum, kinds}`. The minimum and maximum are counts of units.
- A term's tier is `device(i)` or `host`, where `i` is an index into the engine's device group. The ledger maps each
  tier to a budget (4.1). On a machine with unified memory, `device(0)` and `host` map to one pool. NVMe belongs to
  the store.
- Each term has a key by kind of quantum, where the bytes change by kind. A prefill's scratch is not a decode's
  scratch.

The nagaya declares its own terms in the ledger: staging, store buffers and IO. The ledger charges every allocation.
Bytes without a charge are a defect. (ds4-recipe: vision weights without a charge "show up as untracked bytes".)

**Staging.** The ledger makes a staging charge for each copy of engine state in host memory. Two copies get the
charge. One is the copy that a read or a decision keeps, so that its later questions start from it. The other is the
buffer that holds a snapshot until the store takes it.

- The size is the stored bytes of one slot. A fixed part counts whole. A cells part and a ring part count their rows
  up to the highest row of the slot. A 32,768-token Qwen3.8 context is about 1.93 GB.
- The job asks the ledger for the bytes before it reads the state. The tier is `host`. The bytes come from the
  unassigned bytes of its budget (4.4 step 4), as the bytes of a new context do.
- A refusal reads nothing and changes nothing (4.3). The client gets 429 `busy` (11.5), with the shortfall in bytes.
- The charge ends when the copy drops. A job drops its copy before its caller hears the result.
- The store's own bytes are in the store's budget (4.1a). The charge covers the transient buffer only.
- A release that the ledger refuses leaves the ledger wrong. This holds for a staging charge and for a context charge.
  The service records a failure, the main loop sees it, and the process stops with exit 1 (8.5). A release in a
  stopping process is no failure, because the ledger has no charge left.

Reason: the copy is the largest transient allocation of the nagaya, and no other term charges it.

**4.3 Admission.** The ledger thread admits each call that allocates before the call runs. These calls are `open`,
`attach`, `ctx_new`, `ctx_fork`, `ctx_copy`, `part_write`, and each resize. The check includes transient peaks. A
refusal charges nothing and changes nothing.

`pass` never allocates. Its scratch for each kind is a mandatory term, which the ledger charges at `open`.

Admission of a profile keeps the charges that live now.

- A live context and a live staging copy stay charged when the ledger admits a profile. A later release finds its
  charge.
- The ledger takes the live charges from the unassigned bytes of the new budgets. It charges a live context with the
  context terms of the new profile.
- If the live charges do not fit the new budgets, the ledger refuses the profile and changes nothing. The shortfall
  is in bytes (4.3).

Reason: a profile swap (3.3) admits a profile while requests run. A reset of the charges makes a later release fail.
The failed release stops the process (4.2), so a planned restart becomes a crash. The alternative, a
refusal of every admission while a charge lives, would make a transient request refuse a profile for good (2.3).

When the ledger refuses a quantum's call, the quantum waits in the scheduler and keeps its place. A request that can
never fit, even with all reclaim done (4.6), gets 413 `too_large`. Its `param` names the limit that it is over.

**4.4 Spare memory.** The ledger first charges all mandatory terms and the minimum of each elastic term. If they do
not fit, the nagaya refuses the profile (2.4). The ledger then divides what remains in each budget like this:

1. Give each engine in the profile a part of the spare bytes in proportion to its share (D9). An engine with no
   elastic term in that budget leaves its part unassigned.
2. Inside an engine, split its part among its elastic terms in proportion to each term's minimum bytes. When every
   minimum is 0, the elastic terms get no spare bytes.
3. Round each term down to whole units, and cap it at its maximum.
4. Leave the bytes that remain unassigned, and report them.

Counts come from this arithmetic, never from free memory.

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

**4.7 Host safety floor.** The first Qwen3.8 engine keeps 12 GiB of MemAvailable free. In the nagaya, a safety floor
is a check for what the ledger cannot see: other processes, tmpfs and swap. It does not set any size. When free
memory goes below the floor, the ledger refuses new admissions, resizes nothing, and reports the event.

On machines with unified memory, RSS and the OOM scorer do not show GTT. Thus the nagaya sets its own OOM score.

Not built. The ledger checks no floor, and the process sets no OOM score. The Qwen3.8 binding checks the floor when it
opens (engine-abi, "The context term holds the pool").

## 5. Scheduling

**5.1 Units.** A job is the quanta of one request, in order, with a class (ontology). A quantum is one pass or one
stage. It runs to its end, and the nagaya chooses what runs next only between quanta.

For an engine with stages, the nagaya chooses between quanta on each device apart. A stage quantum of a block is
ready after the stage before it ends. The job's lease keeps every other job from the slot until the job ends
(engine-abi D1, "The hold of the slot").

An engine meets its `max_quantum_ms` (D8) when it chooses the size of each unit of work. The engine knows its own
speed, so it picks the size from its own measurements. (The first Qwen3.8 engine splits a prefill into 256-token chunks
in its Python service, and that split moves into the wrapper of the Qwen3.8 engine.) The watchdog reports each quantum
that runs past the bound (7.4).

**5.2 Kinds.** Each quantum has a kind: `read`, `prefill`, `decode`, `embed`, `copy`, `step` (an image step), or
`train` (a micro-batch). Metrics and the cost model use kinds. A kind does not set urgency (the first Qwen3.8 engine did
the same). An image job runs one `embed` for its text encoders, one `step` for each denoising step, and one `step` for
the VAE decode (11.9). An embedding job runs one `embed` for each input (11.10).

**5.3 Classes.** The classes are `interactive`, `batch` and `filler`. A client sends `interactive` or `batch` in
council v2's `class` field (gogi). Only jobs that the nagaya or a profile starts, such as training and the audit, are
`filler`. Between quanta, the scheduler always runs a
ready quantum of a higher class first. A quantum is ready when its job has no quantum that runs, its inputs are
present, and the ledger admitted its calls (4.3).

A council decision is interactive. A job's class starts to apply at its first quantum, not when the job arrives.
The first Gemma 4 engine takes the card at the council's first GPU step (`lazy_high`). There, decisions beside a
prefill take 300-390 ms, and 383-702 ms without it.

**5.4 Order in a class.** The classes do not schedule the same way:

- In `batch` and `filler`, the scheduler orders jobs by integer priority (0 by default) and then time of arrival.
  One job runs its quanta in flow until it finishes, or until a higher class or higher priority job becomes ready.
  The scheduler does not interleave jobs of equal priority in `batch` or `filler`.
- In `interactive`, the scheduler interleaves ready jobs by three keys: integer priority (0 by default), owed
  seconds, and time of arrival. Owed seconds is the device time that a job is owed under equal sharing among the
  present jobs in `interactive`, clamped to ±30 s. A job that leaves takes only its own balance (ds4-recipe
  `11-multi-session-serving.md` section 6).

gogi settles the first key: "a higher priority runs first within its class" (gogi `docs/council-api.md`, "Classes").
Among equal priorities, the higher owed balance runs first, and then the earlier arrival.

When an interactive pass has several segments, each segment gets an equal part of the pass time. The rounding
remainder goes to the segment with the earliest arrival.

The scheduler charges each segment on its own. A segment of job `j` with a share of `c` ms, among `n` present
interactive jobs, changes the balances like this:

1. the balance of `j` falls by `c*(n-1)/n`, rounded down;
2. the balance of each other present job rises by `c/n`, rounded down;
3. the rounding remainder goes to the present job with the earliest arrival, so the changes sum to zero;
4. each balance then stays between -30 s and +30 s.

A job that leaves removes its own balance. Step 4 forgives the excess, so the balances sum to zero only while no
balance is at a bound.

**5.5 Filler.** A filler job declares a checkpoint interval: the time between its checkpoints. It must be resumable:
its state parks when it gives way (Ai2's "unallocated" time). It runs only when no interactive or batch quantum is
ready on its device group.

One scheduler serves one device group (spec 3.2, D1: one thread per device group). Two engines that use one device
share one scheduler. The filler gate is per group.

**5.6 No preemption.** Nothing interrupts a quantum that runs. Priority orders the work that waits. Priority never
moves a continuation off the context that holds it. The job holds that context with a lease.

**5.7 Segments.** Quanta from several jobs on one engine can share a pass as segments. The scheduler permits this
only when the engine declares `segments_exact` (after ds4-recipe). Only segments of one class share a pass, and
the quantum takes that class.

Each class has a join window: the time that the scheduler waits to add more ready segments to a pass. It never
waits for jobs that are not ready. In v0 each window is 0 ms, so a pass takes only the segments that are ready when
the scheduler chooses.

In v0 the windows have no key in the bootstrap config. A measurement that
changes them adds the key (2.1).

An engine with a feed rule (6.14) declares no `segments_exact`. The scheduler joins its segments only when no
segment writes rows that the store keeps, as for reads and branch blocks.

**The first joins of an engine with `segments_exact`.** The process joins only the decode quanta of such an engine at
first, as it does for a feed rule (engine-abi "Batched passes"):

- A decode segment is one token with its logit row, no media and no tap. Other segments run one in a pass.
- Segments join only when their control lists are equal.
- A pass holds at most the engine's `max_segments` segments. For the Qwen3.8 engine, that is 16.

Reason: this rule permits more, such as the prefills of several jobs in one pass. But the drivers build only joined
decode quanta today. The card gate `tests/hip_joined.rs` of the Qwen3.8 crate checks the Qwen3.8 engine's decode
rows at deep positions, alone and joined, and beside a prefill.

A pass holds segments of one kind only (5.2), and at most the engine's `max_segments`.
The scheduler joins a ready job only when its next quantum has the
kind of the first segment.

**What `segments_exact` promises.** An engine that declares `segments_exact` promises that the order of each sum
depends only on the shape of the problem. It never depends on the segment count, the row count of a pass, or the
count of compute units. The split of a segment's tokens into passes does not change bits either. Thus the engine does
not choose a split of a reduction by the size of a pass, and it adds
no floats with atomics. DeepInfer's batch-invariant mode keeps this
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

**5.10 The door limit.** The client API accepts a limited number of requests in flight for each engine (the Qwen3.8
engine: 16). Above that limit it answers 429 with `Retry-After`.

The process limits the calls in another way too (11.7). It allows 64 calls at once for the process, and it answers
429 `busy` with no `Retry-After`. The limit for each engine is not built.

**5.11 Cost model (D4).** The nagaya fits a robust affine cost from production quanta, keyed by engine, kind,
segment count and context depth. Quanta that ran beside other work are not in the fit. The nagaya uses the fit for
store eviction value-per-byte (6.9) and deadline checks between quanta (5.8). An engine can have its own private
planner, which stays in the engine.

**A measured prior.** At the first start of an engine identity, the nagaya times a short curve for each kind. Each point
is one warmup run and the trimmed mean of five runs. The curve goes into the store's index under the identity digest,
and later starts read it. The fit starts from the curve and corrects it with production quanta. DeepInfer measures its
decode curve in this way (`deepinfer/worker/model_runner.py:586-619`).

Not built. The nagaya fits no cost curve yet, and it times no measured prior.

## 6. State and the store

**6.1 One store per machine.** One content-addressed store holds the snapshots of every engine (D3). The store has
tiers (host memory, NVMe) and gets its budgets from the ledger. It replaces these stores:

- The first Qwen3.8 engine's parked files (a 50 GiB budget);
- the first Gemma 4 engine's snapshots, its held store, and its prefix cache. These are three
  formats and three SHA-256 id schemes on one host.

**The store tier is `host` or `nvme`.** An entry's tier is `host` or `nvme`, apart from the ABI's memory tier. The
ABI's memory tier, `device(i)` or `host`, names where an engine term lives. The store does not use it.

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

The identity digest uses derive-key mode with the same context string.

The tag numbers are in this table.

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
| 9 | a media part (11.11): its media type, its content id and its count of positions. Only on a decoded head (6.14). It follows tags 5, 6 and 7. Tag 5 holds the recipe of the prompt as the rule fed it. Tag 6 holds the tokens after the prompt. |

**The park recipe.** The park knows the bytes of a state in a slot. It does not know the inputs that made the state.
Thus the park recipe names the bytes. 10.3 gives the one exception. The park recipe is a root recipe with four items:

- tag 1, the identity digest;
- tag 2, the fixed key `nagaya park v1`;
- tag 3, no approximations;
- tag 4, the 32 bytes of the snapshot's content id.

Equal bytes give one entry, and the tripwire fires only on a real duplicate.

The Qwen3.8 engine's `context_id` and the first Gemma 4 engine's `snap:<sha256>` are the input part of a recipe.
They do not include the engine identity. The Qwen3.8 engine's id is a SHA-256 of the tokens and boundaries with
`|gdn-bf16-v1|`.

**The printed form of a content id.**

- A content id is the whole 256-bit BLAKE3 hash. Its printed form is `blake3:` and 64 lowercase hex digits.
- Each API, log, file and gossip value uses the whole printed form.
- A display can show a short form. No input accepts one, and no input resolves a prefix.
- A request that names a short form or a prefix is a 400 `invalid_request`. Its `param` is `id` for a path, or the
  member of the body. The message quotes the id (gogi "Content ids"). A whole id that the server does not
  hold is a 404.

Reason for the whole form: a person copies and pastes what they see. In a cluster that copies objects (6.15), a
prefix that is unique today can name two objects tomorrow.

Reason for BLAKE3, not SHA-256:

- BLAKE3 is a tree hash. A reader can check one range of an object against its id, without the rest.
- A BLAKE3 hash of one file runs on many cores. A SHA-256 hash of one file runs on one core.
- The store, the recipes, the identities and kaijutsu's store already use BLAKE3.

A SHA-256 of a source, for example the object id of a Hugging Face file, can be metadata of an entry. It is never
an address.

**6.3 Sealed bytes are canonical.** In the store, each snapshot is a canonical safetensors file (D10). Each cells
part and ring part has the shape of its coverage: the rows that it holds, in position order. Thus two slots of
different sizes that hold the same rows make the same file.

Padding between tensors is zero. The header pads with spaces (0x20), so that the data starts on a 4 KiB
boundary (D10 rule 5). The word "zero" names the gap between tensors, not the header padding.

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
  the entry.

The index also holds a record that is not an entry: the cost curve of each engine identity (5.11). A second record,
the expected identity of each engine, is not built (12.1).

**One `Check` enum.** The store's `Check` is `full | spot | none`, the same enum that the ABI uses for arenas and
store entries (engine-abi "Capabilities").

The store sets `none` after a restart, before it hashes the blob. It sets `full` after it hashes the blob. It does
not produce `spot` in v0. Reason: `spot` now names the Qwen3.8 engine's digest, and the store does a full hash of each
entry.

**6.5 Chains.** Each engine declares a maximum chain depth and a size fraction. (ds4-recipe: depth 1, and a delta
only when it is at most a quarter of a full snapshot.) A delta holds, for each cells part and ring part, the rows
after its parent's end. It holds each fixed part whole. The store never evicts a base while a delta needs it.

A ring part can wrap past its parent's end. Then the rows after the parent's end are gone, and
check 4 refuses the delta. The refusal changes nothing, so nothing is corrupt.

A ring part can start after the end of the parent. Then the build puts a
whole snapshot with no parent. Check 4 stays as it is. Reason: the store needs no new rule. A ring held whole in the
delta holds 4,096 rows of each sliding layer of the first Gemma 4 engine. That is close to the size of a whole snapshot.

A delta's `put` names its parent content id and its previous recipe. The store checks these facts:

1. The parent is live.
2. The delta is no deeper than the chain limit.
3. The size of the delta is within the delta fraction of the chain's base, which is a full snapshot.
4. The coverage of the delta starts at the end of the parent.
5. The previous recipe equals the recipe of the parent entry, and the recipe of the delta differs from it.

The store does not have the tokens, so it cannot compute the recipe again. The caller holds the inputs that prove
it.

**`max_depth` counts deltas, not entries.** A chain with a base and one delta has depth 1. A store with
`max_depth: 0` refuses every delta. Reason: 6.5 gives ds4-recipe depth 1 with one delta.

**6.6 Lookup.** A lookup is by recipe. It returns the chain of entries that covers the request, from its base to
its tip. The store keeps only the chains that fit the slot (coverage) and the served model's tolerance (ontology).

The store picks the deepest exact chain. The deepest chain is the one whose furthest covered position is highest in
every part. The store gives an approximate chain only to a served model whose tolerance names that method.

**The lookup refuses incomparable chains.** Two chains can be incomparable under the deepest rule: each is higher in
a different part. The store refuses the lookup with an error that names both chains. It does not pick one by a
tie-break. The process treats that refusal as a miss and recomputes.

Reason: a silent choice would not be the deepest in every part.

**An equal chain goes to the newest entry.** Two chains can be equal: the same furthest covered position in every
part. A tie is not an incomparable pair. The store serves the newest entry by `created_ms`. If two entries have the
same `created_ms`, the store serves the smaller content id.

Reason: a fixed tie-break keeps a put that the store allowed from failing at lookup.

Note: D3 says a per-process engine refuses only within one process, and 6.7 lists only `always` and `pinned`. A
per-process store can hold two equal-coverage entries under one recipe.

**6.7 The tripwire.** Some engines declare `always` or `pinned` (D3). For those engines, two content ids under one
recipe with the same coverage are a defect. The store quarantines both entries, refuses that recipe, and reports
it. A person clears the refusal after the cause is known.

Clearing a tripwire refusal removes the recipe's quarantined entries from the index and
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
restore, and the state is poisoned (8.5).

The store quarantines an entry only when its content id or its identity fails verification. A blob whose hash matches
but whose bytes do not decode fails its content id too. So does a blob whose parent or coverage differs from its index
record: the store cannot tell which one is right, so it serves neither. A refusal leaves the entry as it is. A failure
never deletes content.

A read error of a blob is not a quarantine. The lookup or the
restore treats it as a miss, and the caller computes the state again. The entry stays in the index, the store logs
the error, and the next use checks the entry again.

**6.9 Eviction.** The store chooses victims in this order:

1. Interval copies go first, the least recently used first (6.13).
2. Pinned entries are exempt.
3. Thin the chains: interiors go before tips and roots.
4. Remove the remaining entries by value per byte: recency, hits, and the cost to build them again. Tips, roots and
   entries without a chain are in one ranking.

A chain goes as a group: a root takes its chain with it, and the group's value is the value of its first entry.
The formula for value per byte is a choice of the implementation. It never changes bits. The store chooses all
victims before it deletes one. It also chooses them before it writes the new entry.

Each owner has a quota, so one long conversation cannot remove the others (a ds4-recipe incident). If the entries
of one pin are larger than the tier's budget, the pin loses its protection, so eviction cannot deadlock.

**An owner's quota is per store tier.** A put evicts the owner's entries in its own tier until the owner's total
fits the quota and the tier's total fits the budget. Reason: memory and NVMe are separate budgets in 6.1. A byte in
one tier does not make room in the other.

**A host victim moves to NVMe.** When a put to the host tier needs room,
the store first parks the victims to the NVMe tier (10.3, ontology "Park"). It parks a parent before its children.
The NVMe tier makes room by its own rules. The put removes a victim that a park refuses:

- an interval copy, which never goes to NVMe (6.13);
- a child whose parent stays in the host tier, because a restart would cut the chain (6.11);
- a victim with no room on NVMe.

Reason: with two engines on the Strix Halo machine, the host tier is 2 GiB (`docs/deploy.md`, "Two engines on
one GPU"). The Qwen3.8 engine's snapshots of about 135 MB can leave the tier before a client reads them again. The
NVMe tier has 50 GiB. A refused put still changes nothing: the store checks the room before it parks a
victim.

**A host put of bytes on NVMe is a hit.** A put for the host tier can name
bytes that the store holds live on NVMe, with the same recipe and parent. Then the put gives the id of that entry, which
stays on NVMe. Reason: NVMe holds the bytes at least as long as the host tier (6.11). A put for NVMe of bytes in the
host tier stays refused, because an Ok would promise more than the store holds.

Reason: two builds of one prefix can both miss. Then the first put moves to NVMe as a victim, and the second put
asks for the host tier. A refusal of that put stops the process (8.5).

**6.10 Pins.** A pin is a lease on a snapshot, with an owner and an expiry of one day at most. A request
pin holds a snapshot from its lookup to the end of its load. Pins stay through a restart, in `pins.json`.

**Many owners can pin one entry.** Each owner holds its own pin, with its own expiry. An entry is pinned while any
pin on it is live. An unpin removes only the caller's own pin. A pin on a tip also keeps the entries under it,
because eviction takes a chain as a group (6.9).

**The one-day maximum is for every pin.** The one-day maximum is a nagaya rule for every engine. The
store has no clock, so `pin` takes `now_ms` from the caller. It refuses an expiry more than one day (86,400,000 ms)
after `now_ms` with `Invalid`. A refused pin changes nothing.

Reason: one store holds every engine, pins stay through a restart, and a lease with no bound can hold bytes
forever.

**6.11 Restart.** At start, the store loads its index again. It checks each entry when a lookup first uses it.
Entries in the host tier do not survive a restart, so the store removes them from the index at start. (The stop
parks the pinned ones first, below.)

`index.json` stores `hits` and `last_used_ms`, so a restart keeps them for every NVMe entry. A lookup updates them in
memory, and the next index write saves them to disk.

At start, the store also wakes pinned entries into slots when the profile asks for them. (The Qwen3.8 engine wakes its
pinned parked contexts before `READY=1`.) Not built: the nagaya wakes no pinned entry at start.

**The stop parks the pinned entries of the host tier.** After the groups park their states (10.3 step 3), the stop
parks each live entry of the host tier that a live pin holds. It parks the entries under it first, because a park
needs its parent on NVMe. Each parked entry keeps its recipe, its owner and its pins. Thus a pinned snapshot and its
pin stay through a clean restart, as 6.10 says. A crash does not stop, so a crash loses them.

A park of a pinned entry that fails is a failure of the stop, as for a state (10.3).

**Two rules for the park of pins.**

- The stop does not park a pinned interval copy, and its pin ends at the restart. Reason: an interval copy never goes
  to NVMe (6.13), and a copy is not a parent, so no chain needs it.
- The stop waits for the park of pinned entries for at most one park budget (2.1). A park past it makes the process
  exit 1, as for a group (10.3). Reason: the budget bounds the whole stop, and a new budget needs a config field.

**6.12 Steering and lab inputs.** Some state changes bits but is not an input, for example the Qwen3.8 engine's
attention cell mask or a control vector (11.12). Such state goes into the recipe. Thus a steered or masked result never
has a plain recipe. The bytes of a control list are in 11.12.

**6.13 Interval copies.** An engine can declare a checkpoint grain in `limits()`, in tokens. Then, between quanta,
the scheduler makes an interval copy (ontology) of each context that passed a multiple of the grain.

The Qwen3.8 engine declares a grain of 256 tokens. Two
GPU tests in `crates/nagaya-mk-qwen38-gfx1151/tests/hip.rs` measured it on gfx1151:

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
  eviction can remove a copy without cutting a chain.
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

**The scheduler's part.**

- **When a copy is due.** A copy is due after a quantum that moves a context across a multiple of the grain. The
  copy holds the state at the end of that quantum, so its position can be above the multiple. The scheduler does
  not split a quantum to end on the multiple. A split would put the copies of contexts with one prefix at one
  position, so they could share a copy. It costs more passes.
- **One try for each multiple.** The scheduler makes one attempt for each multiple that a context crosses. If the
  put fails (for example, pins block the cap), the scheduler waits for the next multiple. It does not retry at once,
  because each attempt reads about 112 MiB on the Qwen3.8 engine.
- **The previous copy.** When the store holds the new copy, the scheduler removes the previous copy of that context.
  The scheduler does not remove a copy when its context ends. The cap and eviction remove it later.
- **Bad limits.** A grain of 0, or a grain on an engine without `segments_exact`, is an error.
  A cap of 0 means no copies, so the scheduler makes none.
- **Class and kind.** The copy is a quantum of the kind `copy` (5.2). It has the class of its job (see "The job
  path" below).

Reason: these are the smallest rules that keep the text above true.

**The job path.** The council jobs make and use the copies (`crates/nagaya/src/council/build.rs`):

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
lost.

For the Qwen3.8 engine, the fixed parts of a copy are about 112 MiB. They are the GDN state in f32 (108 MiB) and the
conv state (4.3 MiB). The K/V, the index keys and the pooled keys add 55.5 KiB (56,832 bytes) for each
token. With a grain of 256 tokens, a decode makes a copy every 10-15 s.

On the Qwen3.8 engine, the split of tokens into passes does not change bits. The probe `examples/split_bits.rs` in the
Qwen3.8 crate compared one pass, one pass for each token, and uneven chunks, at 64 and 200 tokens. Every
part and all 248,320 logits were equal. One pass of 200 tokens took 424 ms, and 200 passes of one token took 10,079 ms.
Thus a replay of up to 255 tokens takes about 0.5 s on the Qwen3.8 engine.

**6.14 The feed rule.** An engine can declare a feed rule, `{block}`. The
scheduler feeds each span between boundaries in passes of `block` tokens from the start of the span. The Gemma 4 engine
declares it, because its bits depend on the token count of a pass.

This is the reading of 6.2 for such an engine. A recipe names the identity, the tokens and the boundaries. The bits
of a state follow from them only when the rule made every row. Thus the store keeps a state under a recipe only when
the rule made each of its rows. These rules follow:

- **Decode.** A decode quantum feeds one token, or one token and its drafts. Its rows do not follow the rule. The job
  path puts a state with decode rows under no recipe of 6.2 but the recipe of a decoded head (below). The park recipe
  (6.2) names bytes, so a park of such a state is still exact.
- **Drafts.** A draft test feeds one token and k drafts in one pass, with k + 1 logit rows. The rows of a
  rejected draft stay past the token count, and the next pass writes over them. The logits fall
  under the tolerance of the served model.
- **Branch blocks.** A branch block is a read. It feeds the tail of a case and each branch in one pass with
  per-token `meta` (`branch_block`, `gemma4.rs:1521-1554`). No row of it is kept, also the rows of the tail. No build
  continues from the slot after it. The job restores a held snapshot, or feeds again from a boundary.
- **The bits of a branch.** A branch in a branch block gives the logits of a plain pass of the prefix and that
  branch. The difference is 0.
  The rows stay unkept all the same, because the rule did not make them.
- **Reads.** The reads of a decision keep no state. The first Gemma 4 engine splits a case into chunks that it picks
  for speed. The nagaya may do the same for a read on such an engine.
- **A continued span.** The last pass of a span can hold fewer than `block` tokens. Then a job that continues the
  span feeds that pass again (engine-abi D11 rule 7).

**The decoded head.** A generation on an engine with a feed rule holds a head
after its reply. The head keeps the decode rows of the reply. A decoded head needs no second feed of the reply: the
engine has the state when the reply ends.

The goal is that the nagaya does not run again what it has run. These rules follow:

- **Its recipe names its rows.** The recipe of a decoded head names the last boundary of the prompt and the sampled
  tokens. It also says that a decode quantum made each of their rows. It is another recipe than the recipe of the
  same tokens fed by the rule. Thus the bits of each state still follow from its recipe, and the tripwire (6.7)
  stays whole.
- **The record names the head.** The context record of a reply (gogi "The reply in a record") keeps the id of its
  decoded head. A build of that context starts from the head on two conditions. The store holds the head, and
  the rendering of the context starts with its tokens. It makes no difference whether the request names the record
  as its `parent` or sends the context whole.
- **The bits differ a little.** A request that starts from a decoded head gives other bits than the same context
  built from its text. The rule made the rows of that build. Both are valid readings of the context. The
  same request from the same head gives the same bits. The council gives no logits, so the card gate `xe_lfm3_head`
  measures the next-token log probabilities. On `lfm3`, they differ by 7.2e-6 at most, over the 68
  tokens that both reads list.
- **No head when the tokens differ.** The family can render the reply with other tokens than the model sampled. For
  example, the reply parser of LFM3 removes white space at the ends of the content. Then the context has no decoded
  head, and the next build feeds the reply by the rule, as before.
- **No head from a draft.** A reply whose rows came from a pass with drafts has no decoded head, because those rows
  depend on the drafts. The drafts are not built (below), so this rule has no code yet.
- **The closing tokens.** The template ends the reply with tokens that the model did not sample, such as a newline
  after the end token. The job feeds them after the reply, at most 8 tokens (11.3c), and then it holds the head.
- **The items of the recipe.** The recipe has the items of tags 5, 6, 7 and 10 (6.2). Tag 5 is the recipe that the
  prompt has as a span that the rule fed from the last seal, with its media parts. No seal is at the end of the
  prompt. Tag 6 holds the sampled tokens and the closing tokens. Tag 7 is the position of the head.
- **One token in each pass.** Each token of tag 6 had a pass of one token alone. The job feeds the last sampled
  token and the closing tokens in that way too, as decode quanta with no logit row.
- **No head from a shared pass.** A reply holds no decoded head when one of its decode rows shared a pass with a
  row of another job. A reply of an ensemble (`docs/lab.md`) holds none. Reason: with a feed rule, the bits can
  depend on the token count of the pass (engine-abi D11). The ABI has no declaration that they do not. Thus the
  rule of 5.7 stays whole: no joined segment writes a row that the store keeps.
- **LFM3 can do more later.** The bits of LFM3 do not depend on the other segments of a pass (engine-abi "Batched
  passes"). A later declaration in the ABI can let it keep the head of a shared pass.
- **The span after the head.** A decoded head is a seal, as a boundary is. A build that starts from it feeds the
  rest by the rule, as a new span from the head. Each later boundary chains from the recipe of the head. The
  response counts the tokens of the head in `usage.kept`.
- **The search.** A build reads the record of the prefix that ends at each `assistant` message, from the last one
  down. It reads at most 4 records. It reads the record id of that prefix with `snap: true` on its last message, as
  gogi's `with_reply` writes it. The search stops at the first record that has a head note (6.17).
- **The check of a note.** The build computes the recipe of the note again from its own rendering. A recipe that
  differs is a miss, and so is a head that the store does not hold. After a miss, the build does what it does with
  no decoded head. A read, a decision and a speech job make the same search, because each starts with a build.
- **A cut reply.** A reply that ends with `max_tokens` holds a decoded head at the seam of its open rendering
  (11.3c). The record of the context with the open reply names it, and a continuation of that context starts from
  it. A cut reply that the client sends back closed has another record, so its build finds no note. It feeds the
  reply by the rule. Thus the rule "in any form" of 11.3c holds only for an engine with no feed rule.
- **A dry run.** A dry run names a head that the store does not hold by the recipe of the rule (11.3c). It does so
  also when a build would start from a decoded head.
- **No head after latent steps.** A latent row is not a token (`docs/lab.md`, "No head").

The code of these rules:

- `GenerateThen::decoded_head` in `council/generations.rs` holds the head, and it writes the head note.
- `Service::noted_head` in `council/mod.rs` makes the search. `council/records.rs` keeps the notes.
- `JobBody::shared_pass` in `jobs.rs` tells a job that its decode row shared a pass.
- `crates/nagaya/tests/council_feed.rs` has the tests, with the fake engine that has a feed rule.
- `crates/nagaya/tests/http_conformance_feed.rs` runs gogi's suite on that engine, with its head checks.
- `crates/nagaya/tests/xe_lfm3_head.rs` is the card gate of LFM3.

**The job path.** The council jobs take passes of `block` for an engine with a feed rule (`council/mod.rs`, the
`chunk` of a build). A build starts each span at a held boundary, at a decoded head or at 0. A decoded head is a seal.
Thus no job continues a span today, and rule 7 has no code yet. The tests are in
`crates/nagaya/tests/council_feed.rs`.

Drafts are a setting of the served model, not a bit setting of the engine. The nagaya's generation job makes the
prompt-lookup drafts. The first Gemma 4 engine makes them in the same way: keys of 4 and 3 tokens, and at most 7
drafts. Reason: drafts change no row that the store keeps. Thus a change of the setting must not change the identity
and lose the store. A test on the cards must still show that drafts change no bit of a kept row
(`docs/new-engine.md`).

The drafts are not built yet. They come first to the Qwen3.8 engine on the Strix Halo machine.

**Open:** what does a seed promise when drafts change the text of a sampled reply? The work that builds the drafts
answers it.

**6.15 Objects.** Council requests can carry images, audio, video and MIDI. gogi's contract holds the text of the
API: "Objects and keeps", "Outputs" and "Contexts". This section is how the nagaya implements it.

- **Object.** An object is bytes and a media type, named by its content id (6.2). The id is the BLAKE3 hash of the
  bytes, with no canonical form. A message, a reply or a case names an object by `{"id", "mime"}`.
- **The routes.** `PUT /council/v2/objects/{id}` sends the bytes. The server hashes them as they arrive. A
  mismatch is a 400 that names both ids, and the server keeps nothing. A put of a held object takes no new disk space.
  The server still reads and hashes its body, so a client sends `HEAD` first to skip the transfer (gogi).
  `GET` and `HEAD` on the same path read the object, and `GET` accepts `Range`.
- **Two classes.** An object is `cached` or `kept`. A cached object is evicted as an entry with no pin is (6.9).
  A kept object has a keep: a lease with an owner and an expiry, as a pin has (6.10). The store never evicts an
  object while a keep on it is live.
- **The term of a keep.** A keep lasts at most 30 days, and a client renews it. Reason: the one-day maximum of a
  pin (6.10) is for model state that the server can build again. A client cannot always make an object again.
- **A miss.** A request that names an object that the cluster does not hold is a 404 `not_found`. Its `param` names
  the part, and the message quotes the id (gogi). The client sends the object again. Thus the rule of the contract
  stays: an eviction costs a send, never a wrong answer.
- **Home machines.** Each kept object has two home machines, by rendezvous hash of its id over the machines in gossip
  that hold a store (2.3). A machine that receives a kept object tells its homes, and each home pulls a copy.
  Gossip carries which machines hold a store, never a list of objects.
- **A pull.** A machine that does not hold an object asks its homes first, then the other machines. It checks each
  range of the bytes against the id with the BLAKE3 tree (the Bao encoding). Thus parts of one large object can
  come from several machines.
- **Outputs.** An image, audio or other output of a generation is a cached object. A client that keeps the result
  puts a keep on it.
- **On disk.** The reference store keeps an object as a file under the store directory, at
  `objects/<the first 2 hex digits>/<the other 62>`. Its media type and keeps are in the index, not in the file.

**The budget of objects.**

- Objects have their own budget, the key `objects` of the `[store]` table, in bytes. It is not part of `nvme`. Reason:
  a batch of images must not evict model state, and model state must not evict images. The profile of the B70
  machine sets 200 GiB.
- A profile without the key declares no capability `objects`, and the routes of objects are a 404.
- `object_bytes` is 8 GiB (8,589,934,592 bytes), or half the budget when that is less. `keep_bytes` is half the
  budget. Thus a put always fits after the store evicts cached objects, and full keeps never stop a generation.
- The store evicts the cached object that was used least recently first.
- At most four puts stream at once, so staged bytes stay below four times `object_bytes`. A fifth is a 429 `busy`.
- The first read of an object after a start hashes its file. A file that does not hash to its id drops the object,
  as an eviction does (7.3: same size is not same bytes). An index that does not read moves aside, and the store
  starts empty. Objects are a cache of what clients sent, so a start never stops for them.
- Replication waits. The first step is the store on one machine, its routes and the conformance
  checks. Home machines and pulls come when a second machine speaks this contract.

**A read on a miss.** A `GET` of an object that this node lacks asks each live
peer, and it gives the first hit. A forwarded request takes its objects to the target first (11.1a). This is the
first step of "A pull" above. The home machines, the copies and the checks of each range wait.

**Objects in the cluster.** These rules fill what "A read on a miss" leaves open: the order of the peers, the
pending ids, and the copies.

- **The read.** The coordinator sends a `HEAD` of the same path to each live peer at once, as a broadcast read of the
  cluster call (11.1a).
- **The answer.** The first peer that does not give a 404 gets the `GET` or the `HEAD` of the client, with its
  `Range`. The status, the headers and the body of the peer come back as they are: 200, 206,
  308 or 416.
- **No copy.** The coordinator keeps no copy of the bytes. Reason: the pull and the home machines above replace this
  rule later, with a check of each range.
- **Pending ids.** A pending id follows the same rule, so a client can read the output of a forwarded generation
  through the coordinator. After the seal, the 308 of the target names the content id. Its `Location` is a path with
  no host (`crates/nagaya/src/http/objects.rs:283`), so the client comes back to the coordinator. A `GET` of that id
  follows the rule again.
- **A put.** `PUT /council/v2/objects/{id}` holds the object on the coordinator only.
- **The objects of a forward.** Before a forward, the coordinator collects the ids that the body names. They are the
  object parts of each message and of each embedding input, and each entry of `case_media` (gogi "Contexts").
- For each id, the coordinator sends a `HEAD` to the target. On a 404, it reads the object, locally or by the rule
  above, and puts it to the target. The puts end before the forward starts.
- An object that no node holds does not stop the forward. The target's 404 names the part. A put that the target
  refuses, such as a 413 `too_large`, is the response, and nothing forwards.
- A peer that does not answer in 2 s counts as a 404, and the log names it.

Snapshots stay entries (6.4) for now. They can become objects of their own media type later, with their own order of
eviction.

Some names in an identity are short today: the council `engine` (11.3c), and the template and tokenizer hashes
(12.1). Each has 16 hex digits. They are names that a client compares, not addresses.

**Open:** do these names become whole at the next change of identity?

**Open:** do drafts change the bits? A drafted verify pass
has a shape that a plain decode does not have. Gemma 4 declares `segments_exact: false`, and its feed rule (6.14)
exists because its bits depend on the split of a pass. The test runs on the B70 cards:

- Decode the same prompt greedily with drafts on and with drafts off. Compare the logits of each step, bit for bit.
- Compare the rows that the slot holds after each run, bit for bit.
- Equal bits accept the note above. Other bits make drafts a bit setting, or a rule that a verify pass follows.

The Strix Halo machine cannot run this test. The Qwen3.8 engine declares `segments_exact: true`
(`crates/nagaya-mk-qwen38-gfx1151/src/spec.rs:463`), so the test cannot fail there for this reason.

**6.16 Conditionings.** The store keeps the encoder output of each prompt. Variants with the same encoder share their
conditionings. Reason: a conditioning is
sealed state with a recipe. Thus the rules of 6.2 to 6.11 give its key, its lookup, its eviction and its tripwire.

- A **conditioning** is the output of an image engine's text encoder for one prompt. It has one row for each id after
  the cut (11.9c). Qwen-Image 2.1 gives rows of 4096 bf16 values, 8 KiB each.
- The store keeps it as a snapshot of one cells part, `conditioning`. The canonical file (D10) holds that tensor, of
  shape `[rows, 4096]` in BF16. Its coverage is the positions from the cut to the last id.
- Its recipe is a root recipe with the encoder digest and the fixed key `nagaya conditioning v1` as tag 2. It has no
  approximation tags and no lab state. Reason: the park recipe of 6.2 uses a fixed key in this way.
- The **encoder digest** takes the place of the identity digest. It is the tagged hash of the encoder's part of the
  identity (12.1). That part is the checkpoint digest of the text tower, the tokenizer hash, the template and the
  cut. It also holds the numbers of the encoder and the lineage of the plan table. Until the canary of that
  engine runs (12.1), the digest also covers `build`.
- The transformer and its LoRA are not in the encoder digest. A LoRA never changes the encoder (11.9c, "The LoRA
  fold"). Thus two variants with the same encoder, tokenizer, template and build share each conditioning.
- The engine gives the encoder digest in its `conditioning` capability (engine-abi "Image").
- One boundary follows at the end of the ids. Its tag 5 is the root recipe, tag 6 the template ids, and tag 7 the id
  count. Thus the recipe covers the weights, the tokenizer, the template and the ids.
- The nagaya looks up the recipe before `image_begin`. On a hit, the job carries the rows, and the engine skips the
  encoder (engine-abi "Image"). On a miss, `image_begin` gives the rows, and the nagaya puts the entry.
- The reason of the entry is `prefill done` (6.4). Its owner is the owner of the request. It lives in the host
  tier, until a full tier parks it (6.9). The engine declares `max_depth: 0` (6.5), so eviction ranks it as an
  entry with no chain (6.9).
- The engine declares the determinism `pinned` (12.1, "The plan table of a HIP engine"). Thus two content ids under
  one recipe fire the tripwire (6.7).
- A hit and a miss give the same image bits, because the engine reads the same rows in both. The conformance suite
  checks this (8.3).
- No route reads, pins or lists a conditioning in v0. The `snapshot_bytes` of the served model stays 0 (11.9a),
  because a client holds no context of it.
- A template of 1024 ids gives at most 1010 rows, about 7.9 MiB. The prompt of the oracle gives 22 rows, 176 KiB.

One choice is left for later. It is a design, and nothing builds it now.

- The keys and values of the conditioning rows in the 32 blocks take 512 KiB for each row. They depend on the
  transformer and its LoRA. A later rule can keep them as entries too, so that a hit skips all of `image_begin`.

**6.17 Context records.** gogi "Context
records" says what a record is, and it leaves the choices below to the server. Reason: each choice is the simplest
one that keeps the contract. Each follows a rule that specs or pins have now.

- **A record is a server file.** It is the file `contexts/<context id>.json` in the store directory, beside `specs/`
  and `pins.json`. The file holds the record text of gogi's `record_text`, and nothing else.
- **A record is not an entry.** The index of 6.4 does not list it, and it has no recipe, no tier and no chain. It
  takes no bytes of a budget of 4.1a, and eviction (6.9) never reads it. Reason: a record is text of the client, not
  model state (gogi "Context records").
- **One book for the server.** The server opens the records once, with the specs and the pin book (11.7, "Several
  text models in one process"). Each text service has a handle to the same book. A record names no served model.
- **A node with no home holds no record.** A node that serves no text model has no specs and no pins (11.1a), and it
  has no records. A `parent` on it is the 404 of gogi, and its generations name no record.
- **The limit is a count.** `limits.context_records` is 10,000, a constant in the code. Identity names it when the
  node has a home. Reason: gogi's limit is a count, and the fake holds 10,000.
- **No byte limit.** The body limit of 16 MiB (11.7) bounds what one request adds. A context past `context_tokens`
  fails its checks, so it holds no record. A byte limit can follow if the files grow too large.
- **Past the limit.** After each hold, the book drops records until it has 10,000. It drops the least recently named
  record first. It never drops a record that a live pin keeps, and it never drops the record of that hold. Thus
  pinned records can take the book past the limit.
- **What names a record.** A hold names it. So does a request that names it as a `parent`, and so does a `GET` of
  it. A pin does not name it.
- **The order is in memory.** No request writes the order to disk. At start, the records take the order of their
  file times, the oldest first, and then the byte order of their ids. Reason: a read must not write a file.
- **The book keeps ids in memory.** It reads the text of a record from its file at each name. Reason: 10,000 long
  contexts do not belong in host memory outside the ledger (4.1).
- **A hold writes once.** A new record goes to a temporary file, then `fsync`, then a rename, as a spec does. A hold
  of a record that the book has writes nothing. Thus a crash leaves no half record.
- **A drop removes the file.**
- **Restart.** At start, the book lists the directory. It removes each temporary file. A file whose name is not
  `<context id>.json` stops the start, as a bad spec file does (8.5). The message says to remove it. The stop of
  10.3 writes nothing for records.
- **A check at each read.** The book checks a record when a request reads it, as the store checks an entry (6.11).
  The context id of the file's text must be its name. A file that fails is removed, the log has a warning, and the
  read is a miss. Reason: gogi lets a server drop a record at any time. A change of gogi's record text makes each
  old file such a file, and a miss costs the client one whole context.
- **The head note.** The record of a reply
  can have a head note: the file `context-heads/<context id>.json` in the store directory. The note holds the
  recipe of the decoded head (6.14), its token count, and each seal of its state in order. A seal is a token count.
  The seal of a decoded head also has the token count of its prompt. Reason: the build computes the recipe again
  from these counts and its own tokens, so it trusts no token of the note.
  - The note lives with its record. A drop of the record removes the note. At start, the book removes each note
    that has no record. A file whose name is not `<context id>.json` stops the start.
  - A note goes to a temporary file, then `fsync`, then a rename, as a record does. Thus a note stays through a
    restart. The store decides if the head stays (6.9, 10.2).
  - A note that is not valid text is removed, the log has a warning, and the read is a miss.
  - A note is not an entry, and it keeps no entry in the store. A note whose head the store does not hold is a miss.
  - The read of a note does not name its record.
- **Pins.** The pin book keeps the record ids of each owner beside its snapshots (6.10), in `pins.json`. A pin keeps
  the records that the book held at the `PUT`. A `PUT` with held records and no held snapshot keeps its owner. The
  records of an owner end with its pins. At start, the pin book forgets each record whose file is gone.
- **Lock order.** A method that needs several locks takes the pin book first, then the store, then the records.

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

The timeout is 30 s, three times the longest declared quantum bound (the Qwen3.8 engine: 10 s). It has no key in the
bootstrap config. The code takes it as a start parameter, so the tests use a short value.

**Open:** The group thread sends no beat while a quantum runs. On a slow NVMe, a copy quantum could run past the 30 s
timeout, and the watchdog would end the process. This is not measured.

**The watchdog during an open.** An engine's open can take longer than the timeout: the Qwen3.8 engine loads in 24 to
50 s. A group beats its heartbeat while its engine opens, so a long open is not a stall. The group thread is blocked
in the open. Thus a helper thread sends the beats, at the heartbeat interval, until the open ends. The helper stops
when the group thread leaves the open for any reason.

The beat hides a hung open from the watchdog. The wait for the opens is the backstop (10.1). It ends a hung open
after 300 s, and the process exits 1 (10.4). The 300 s equals `TimeoutStartSec` in the unit.

The wait for the opens is 300 s. Reason: the Qwen3.8 engine waits up to 60 s for free memory, then loads for about
35 s. A wait of 60 s ends a slow open that is not hung.

## 8. Correctness at runtime

**8.1 Canary.** The nagaya records a canary for each engine: fixed inputs and the exact outputs that the engine gave
for them. The nagaya runs the canary through the engine, with its own sampler and decision code. The Qwen3.8 engine's
canary has pilot reads and an 8-step greedy decode. It keeps letter
logits as f32 bit patterns, the top 5, and a SHA-256 of all logits. The canary's hash is part of the engine identity.

The nagaya runs the canary again before the engine is ready, with no tolerance. At start, a mismatch makes the
nagaya exit with code 3, "do not restart" (10.4). After start, a mismatch marks the engine fatal (8.2), and the
process continues with its other engines.

Built for the Qwen3.8 engine only. The engine runs its canary at each load and compares it with a table of records.
Any other engine gives an empty `canary_hash` and runs no canary. The process runs no canary through the `Engine`
trait.

The canary adds time to each load. Over 5 processes for each path, it takes about 1.9 s on the fast path and about
4.2 s on the exact path. A load takes about 28 s.

The Qwen3.8 engine runs its canary itself. It runs on the engine's worker thread, in two contexts that it makes and
frees, and it uses the same pass path as a client. The host takes the
argmax of each row. Reason: the engine crate holds the worker thread, the passes and the fixed token ids, and it needs
no tokenizer. The form through the `Engine` trait comes with the canary of the other engines.

The content of the canary of the Qwen3.8 engine is fixed.

- Context A takes 5 tokens in one pass. Context B takes 300 tokens in two passes of 256 and 44 tokens.
- Then 8 greedy steps follow. Each step is one joined pass of 2 segments, A then B, with 1 token each.
- Each pass gives the logits row of each segment. That makes 19 rows: 1, then 2, then 16.
- A row hash is the SHA-256 of the row's f32 values as little-endian bytes, with no change to a value.
- The canary hash covers a text tag, the input ids and the 19 row hashes (`canary.rs` gives the byte layout).
- The canary does not reach the indexer path. That path starts past 2051 cells, and it needs about 2,300 tokens.

Reason: this content covers a short prefill, a long prefill in two passes and a joined decode, in about 3 s. The card
gates keep the cases that it does not cover.

The record of a canary is a table in the crate of the Qwen3.8 engine.

- The key of an entry is the checkpoint digest, `fast_qmm` and `tiles`. The value is the canary hash.
- The identity takes `canary_hash` from the table. Thus a description gives it with no GPU and no load.
- A set of key values with no entry does not open. The error names the three key values and what to do.

Reason: the council takes the digest and the root recipe of an engine on demand from its description, before any
load (11.1). A canary hash that only an open measures would come too late for that.

A mismatch of the canary refuses the open, with no tolerance.

- The engine runs the canary at each load of its weights, after the scratch probe. It compares the canary hash with
  the table.
- A build with other bits does not open until a commit records its hash. The commit says that the bits moved. Then
  the engine has a new digest.
- At start, a mismatch makes the nagaya exit with code 3. A mismatch at a later load stops the process today, as
  each failed open of a swap does.
- The message of the error gives the table entry, the measured canary hash and the 19 measured row hashes.
- A canary pass that fails for another reason, such as a stop on the spin bound, is an ordinary open error. It is
  not a mismatch.
- Not built: the fatal state and the run after a pass failure of 8.2.
- Built: the table, the check at each load, the exit code 3 at start, and the second form of the open.

Reason: a build that gives other bits must not serve under the old digest. A load is rare, so a check at each load
can cost its time.

A checkpoint that the table does not hold.

- A set of key values with no entry is an error in the description and in the open. The start refuses with exit 3, as
  each other error of a description does.
- The service cannot measure the hash then, because it does not start. The crate gives the open a second form that runs
  the canary, reports it and compares nothing. The test `hip_canary` uses that form and prints the measured hash.
- The person adds one line to the table, in a commit that records the bits of that checkpoint.
- The process always uses the checked form. A card test of a deploy opens the engine in the checked form too.

Reason: the table is the simplest record that needs no new file. A stranger needs one line for each checkpoint.

**8.2 Pass failures.** A pass returns ok, poison (D6), or timeout. The nagaya treats outputs that are not finite as
poison. After a poison or a timeout, the nagaya does these steps:

1. Fail the jobs in that pass. Free their contexts, or mark their slots dirty until a job clears them (`slot_clear`),
   overwrites them from position 0, or writes them through `part_write`.
2. Run the canary again on the group thread, before any other work.
3. If the canary does not match, or if this is the engine's fourth timeout in the last 600 s, mark the engine fatal.
   Refuse every later job for it.

**8.3 Conformance.** The crate `nagaya-conformance` holds one suite for every engine, generic over the `Engine`
trait. The GPU tests of each engine crate run it on the real device. They are `tests/hip.rs` in the Qwen3.8 crate and
`tests/xe.rs` in the Gemma 4 crate. The fake engines run it with no GPU. It checks these properties:

- park and restore keep every bit;
- seal runs only at boundaries;
- the quantum conditions hold (ontology "Quantum");
- an admission refusal changes nothing;
- for an engine that declares `segments_exact`, a segment gives the same bits in passes with one, two and the
  maximum count of segments (5.7);
- for an engine with stages, `Engine::pass` and the stage path give the same outputs and parts (engine-abi D1);
- for an engine with a feed rule, a span fed by the rule after a restore gives the bits of one job (6.14).

**One oracle for each family.** Each model family has one reference: fixed outputs of the model's own Python code
on the CPU, kept as a fixture in this repository. Every engine of that family passes its conformance against that
reference, on any vendor's device. The reference runs on the CPU, so it does not give the bits of a GPU. Thus the
oracle has a tolerance, and a spike measures the gap first.

**The Qwen3.8 oracle.** The tool `tools/oracle_fixture.py` writes the fixture. It runs the model's transformers code
from the reference repository of the Qwen3.8 engine, with its venv (`AGENTS.md`, "Checks").

The fixture holds five prompts of 5 to 596 tokens, with the top 64 logits and the logsumexp of the last position.
The bench `oracle` in the Qwen3.8 crate compares the Qwen3.8 engine with it.
The Qwen3.8 engine's exact path (`QWEN38_FAST_QMM=0`) is the gate. Its largest gap is a logit difference of 1.7e-5.

- On the exact path, the argmax agrees when the reference's top two differ by more than 0.25. The reference's top 8
  are in the engine's top 16. The largest logit difference is at most 1e-3, the KL at most 1e-8, and the logsumexp
  difference at most 1e-3.
- The fast path (`QWEN38_FAST_QMM` unset) is an approximation (12.1). It keeps only the argmax and top-8 rules, and the
  bench reports its other numbers. At 249 tokens its largest logit difference is 1.03.

The fixture covers prompts of up to 596 tokens. Past about 640 tokens, the gap between the Qwen3.8 engine and the
reference is not explained (15 item 8). The oracle does not gate a longer prompt.

A family with no reference has no oracle. Both families of v0 have one: the Qwen3.8 oracle above, and the Gemma 4
oracle below.

**The Gemma 4 oracle.** `tools/gemma4_reference.py` runs HF transformers' own Gemma 4 layers in f32 on the CPU
(copied from the first Gemma 4 engine). `tools/gemma4_oracle_fixture.py` makes the fixture with it, as
`tools/oracle_fixture.py` makes the Qwen3.8 fixture. The reference Python runs in a venv of its own.

The fixture is `crates/nagaya-text-gemma4/tests/fixtures/oracle.json`, and `crates/nagaya-text-gemma4/tests/oracle.rs`
reads it. It holds five prompts of 6 to 586 tokens, in the format of the Qwen3.8 fixture. The fixture belongs to the
family, so each engine of Gemma 4 reads the same file.

- The first Gemma 4 engine measured these gaps. The full pass had a largest logit difference of 5.8e-5.
  A prompt of 1,270 tokens, past the sliding window of 1,024, had 7.2e-3 and the same top token.
- At 1,270 tokens, the f32 order of sums changed 123 of 38,100 top-8 expert picks. Each change began on a near tie in
  HF's own router.
- The gates are the argmax and top-8 rules of the Qwen3.8 oracle. The largest logit difference is at most 1e-2. The
  bench reports the other numbers. A spike measures the gap at longer prompts first. Each vendor and device has an
  engine crate of its own.

A gate that restores state restores it through `Store::restore`, the path that production uses. Thus a gate and
the store cannot drift apart.

The suite compares values by integer bit pattern. NaN or Inf fails, even when the bit patterns are equal. Counts
must also match. Each gate keeps a broken variant that it must fail (ds4-recipe `01-correctness-and-measurement.md`).

**The speech oracle.** The random stream of the card differs from PyTorch's, so the oracle of a speech engine forces
the codes. The fixture holds a prompt, the codes that the reference sampled for it, and the reference outputs:

- the logits of codebook 0 at each frame, and the logits of the code predictor for codebooks 1 to 15;
- the audio that the codec decoder makes from those codes.

The engine runs with the codes of the fixture in place of its samples, and each output must be within its tolerance.
Python writes the fixture once, on the CPU; no Python runs at test time (`AGENTS.md` "Checks").

**The embedding oracle.** `tools/embeddinggemma2_golden.sh` runs sentence-transformers 6.1.0 on the CPU, in f32 from the
bf16 weights, with eager attention. It runs in a venv of its own, and its header names each version. It writes
`cases.json` and `golden.safetensors` into `target/embeddinggemma2-golden` or `EMBEDDINGGEMMA2_GOLDEN_DIR`. Neither file
is committed.

- The files hold 8 texts and 8 tasks. For each pair, they hold the ids, the vector, and its cuts to 512, 256 and 128.
  They also hold a bf16 vector for `query` and `document`, and the hidden states of one input.
- The texts include CJK, emoji, an empty text, a text past the window of 512 positions, and one of 2040 tokens.
- The gate of the family: its ids equal the oracle's ids for each text and task.
- The gate of the engine: after the nagaya normalizes, each vector differs from the f32 vector by at most 1e-4 in
  each value. The first run on the card measures the gap first, as for every oracle.
- Reason for 1e-4: the bf16 run differs by up to 3.7e-3, with a cosine of 0.99957. A gate at that level lets a wrong
  layer pass. An engine with f32 math from the bf16 weights stays far below 1e-4.

**The Qwen-Image oracle.** Reason: this follows the oracles of SDXL and EmbeddingGemma 2, with one gate for each
stage of the engine.

- `tools/qwenimage21_golden.sh` runs diffusers 0.41.0.dev0 in a venv of its own, on the CPU only. It computes in f32
  from the bf16 weights, and loads one model at a time. Its peak is about 33 GiB of RAM.
- Run it with `nagaya.service` stopped and no GPU job. It writes `encode.safetensors`, `steps.safetensors` and
  `parts.safetensors` into `target/qwenimage21-golden` or `QWENIMAGE21_GOLDEN_DIR`. No file is committed.
- The request: the prompt `fusal style. a red fox in a snowy forest, watercolor`, 256 by 256, 4 steps, seed 1234. It
  uses CFG 1.0, the Heretic encoder and the FUSAL LoRA at 1.0. The size is below the request minimum, as for SDXL.
- The start latent comes from `torch.Generator().manual_seed(1234)`, and the engine takes it in place of its own
  stream. The template has 36 ids, the cut is 14, and 22 conditioning rows follow.
- The sigmas are 1.0, 0.7446, 0.4267, 0.02 and 0. The reference gives the LoRA to diffusers as an adapter, with no
  fold. It also records step 0 with the LoRA at weight 0.
- The metadata `template.hidden_states_last` records what `hidden_states[-1]` was in its transformers version. The
  conditioning is the input of the final norm in either case (11.9c, "The load").

The gates. A reference in f32 does not give the bits of an engine in bf16. 11.9c also rounds in bf16 where the
pipeline in bf16 rounds, and the reference does not. Thus each gate has a tolerance. The first run measures each gap,
as for every oracle.

| Gate | What the engine runs | Compares with | Cap |
|---|---|---|---|
| ids | the family on the prompt | `ids` and `drop` | equal |
| conditioning | the encoder on `ids` | `prompt_embeds` | cosine 0.9999 for each row |
| step | one step from each `latents[k]` | `transformer_out[k]` | cosine 0.999 |
| run | 4 steps from `start` | `final_latent` | cosine 0.999 |
| decode | the decode of `vae.in` | `rgba` | 4 in each byte |
| image | the whole job | `rgba` | 16 in each byte, 2 on average |

- The LoRA gate: the engine with no LoRA runs step 0 against `tr.step0_no_lora`. The step gate must fail when the
  engine drops the LoRA. A LoRA that changes less than the tolerance is a defect of the gate, to report.
- The cache gate: a job on a hit and a job on a miss give equal bits (6.16). Two runs of a job give equal bits. Image
  0 of a job with `count` 2 has the bits of the same job with `count` 1.
- Each gate keeps a broken variant that it must fail (8.3, above). The variants are the final norm applied, no LoRA,
  the timestep in place of 0 for the conditioning rows, and the bytes in BGRA order.
- `parts.safetensors` holds the encoder layers, the transformer modules of step 0 and the VAE stages. They are
  references for kernel bring-up, not gates.

**The bf16 baseline of each gate.** Reason: no bf16 encoder reaches the cap of the conditioning gate. diffusers in
bf16, as the diffusers reference runs it, does not reach it either.

- The oracle also runs each stage in bf16 under torch, from the same inputs: `prompt_embeds_bf16`,
  `transformer_out_bf16`, `latents_bf16` and `rgba_bf16` (`tools/qwenimage21_golden.sh`). These are the baselines.
- The gap of the engine and the gap of the baseline are each measured against the f32 reference. A gate passes when
  the engine's gap is at most 4 times the gap of the baseline. Thus the engine may differ from f32 as much as
  diffusers in bf16 does, with a margin of 4, and no more. A floor of 1e-5 keeps a gate from failing on noise.
- The caps of the table do not decide. Reason: the looser of the cap and 4 times the baseline lets a broken variant
  pass. A variant with the timestep 0.5 in its condition rows passes the step gate under the cap of 1e-3. 4 times the
  baseline catches it.
- The conditioning gate applies the rule to each row, with the gap 1 minus the cosine. diffusers in bf16 has cosines
  from 0.967 to 0.99991.
- The decode gate measures the mean of the absolute difference of each byte, with a floor of 0.25.
- The oracle holds no run of the whole job in bf16, so the image gate keeps its cap: 2 on average in each byte.

**8.4 Audit.** For some engines the tripwire is not reliable, because of their determinism class. For them, the
nagaya runs an audit after each start, as a filler job in the same process. The audit builds a sample of stored
recipes again and compares the content ids (D3). A restart is the fresh process that the audit needs, so the audit
needs no second GPU client.

Not built. The nagaya runs no audit yet.

**8.5 Crash over corruption.** An unexpected failure in the nagaya stops the process (exit 1). The process does not
continue in a bad state. An event that does not fit the scheduler's state, such as a cancel of an unknown job, is
an error that its caller sees. A cancel of a job that already ended is not an error, because a client can race the
end of a job.

## 9. Observability

The nagaya follows kaijutsu's conventions (kaijutsu `docs/telemetry.md`).

**9.1 Activation.** The nagaya always logs to stderr. It also exports in these cases:

- OTLP over gRPC to the machine's collector, when `OTEL_EXPORTER_OTLP_ENDPOINT` is set;
- OTLP JSON-lines files, when `NAGAYA_OTEL_FILE_DIR` or `KAIJUTSU_OTEL_FILE_DIR` is set.

`OTEL_SDK_DISABLED=true` stops all export. The `service.name` is `nagaya`. The resource attributes carry
`host.name` and the hardware from the inventory.

Built: the log, and the OTLP gRPC export of spans and metrics. Not built: the JSON-lines file export and the OTLP
export of logs. The nagaya warns at the start when `NAGAYA_OTEL_FILE_DIR` or `KAIJUTSU_OTEL_FILE_DIR` is set, so that
a missing export is never silent. The code is in `crates/nagaya-telemetry`.

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
  leak it.

**9.2 Traces.** When a request has a W3C `traceparent` and `tracestate`, its span is a child of the caller's span.
The nagaya records an invalid header as an attribute, not an error. The spans nest in this order:

1. the HTTP server span, `<METHOD> <route template>`;
2. `nagaya.job`, with engine, class and request kind;
3. `nagaya.quantum` for each quantum, with engine, device group, kind, class, segments, and wait and run times in ms.

Decode quanta make metrics only, because one span for each generated token would bury the trace of a generation.
The trace context goes to the group thread with the job. The nagaya gives it to the engine through the telemetry
capability, and the engine adds its own spans under the quantum. No engine carries it onto GPU work: none declares the
capability `telemetry` (`Capability::Telemetry`, `crates/nagaya-abi/src/lib.rs:346`).

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

- The attribute `engine` of a group is the names of its engines in profile order, joined with `+`. A group with one
  engine has the name of the engine.
- The attribute `request` of `nagaya.job` has one of these values: `attach`, `build`, `dry_run`, `read`, `decision`,
  `generation` or `job`. A body names its value with `JobBody::request`. The value `job` is for a body that gives none.
- The job span ends when the job reaches a terminal state, or when a stop cancels it. Its `outcome` is the job state.

**The engine of a job.**

- The attribute `engine` of the span `nagaya.job`, of its quanta, and of the quantum metrics is the name of the job's
  engine. `GroupJobs` holds the engine of each job, so the name is known.
- The joined name above stays for what belongs to the whole group, such as the heartbeat of the group thread.
- Reason: a group can hold many engines. Group 0 of the B70 machine has nine engines, from `gemma` to `inpaint-0`
  (`deploy/b70-pair/profile.toml`). With the joined name, each job has the same long name, and a metric cannot tell
  the engines apart.

Built. `GroupJobs` and the jobs of a staged group take the names of the engines in profile order. A job
span names the engine of its job. `ObservedScheduler::set_job_engine` gives that name to the quanta of the job. The
heartbeat of a group carries no attribute `engine` today. Thus no signal uses the joined name now.

The route template is axum's, such as `/council/v2/pins/{owner}`. A request that matches no route has the span
`<METHOD>` alone, and the `http.route` value `unmatched`. The span has the attributes `http.request.method`,
`http.route`, `http.response.status_code` and `request_id`. A status of 500 or more marks the span as an error. The span
name has no `nagaya.` prefix, so it has the default rate of 9.3.

**A forwarded request.** Reason: a forward (11.1a) crosses two nodes, and one trace must show both.

- The coordinator opens the span `nagaya.forward` under its HTTP server span. Its attributes are `nagaya.peer`, the
  name of the target node, and `http.response.status_code`.
- It also has `outcome`: `done`, `refused` (no connection), `lost` (the target left the live set) or `cancelled` (the
  client closed the stream). `refused` and `lost` mark the span as an error.
- The forward sends the `traceparent` of `nagaya.forward`. Thus the HTTP server span of the target is its child, and
  the spans of the job nest under that.
- A forward that ends `refused` or `lost` logs one warning. It names the node, the route and the request id. A forward
  that ends `done` or `cancelled` logs nothing, because the log has no line for each request.
- The forward adds no metric now. `http.server.request.duration` on each node counts the request (9.4).
- The text routes that run locally (11.1a) make no `nagaya.forward` span.

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

These rules fix the labels of the metrics that exist:

- The attribute `device_group` holds the id of the device group. The attribute `tier` holds `host`, `nvme` or
  `device<id>`. The `<id>` is the machine device id (2.4, "One name for each device").
  A budget that covers several tiers has the sorted tiers joined with `+`, for example `device0+host`.
- The wait of a quantum runs from the moment its lead segment became ready to the moment the quantum started. The lead
  segment is the first one. A job is ready when it arrives, and again when a quantum of it ends and more remain. A job
  that the ledger has not admitted yet is ready, so its wait includes the time until the admission.
- `nagaya.ledger.bytes` is a gauge with one point for each engine term. `admit` records the bytes of each mandatory and
  elastic term of the admission, and omits `context` terms because 0 contexts exist at `open`. `admit_context` and
  `release_context` record the live bytes of each `context` term.
- `nagaya.ledger.refusals` counts one refusal for each short budget. The engine is `all` for a refused profile, and the
  engine name for a refused context. The ledger has the gate `budget` only. The gates `floor` and `resize` come with the
  code that refuses at them.
- `nagaya.store.bytes` and `nagaya.store.entries` are gauges. The store reports them after `open`, `put`, `evict` and
  `remove`. A series that empties reports zero once. The engine is the name in the entry.
- `nagaya.store.lookups` counts every `lookup`. The engine is the model name of the engine that asks, because the store
  does not know the profile name. An outcome is `approx` when an entry of the chain has a provenance. A lookup that
  returns an error counts as `refused`.
- `nagaya.store.restores` counts every `restore`. The engine is the model name. The result
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

**The Xe provider.** Each card has these files under `/sys/class/drm/card<N>/device`, as the B70 machine shows:

- clock: `tile0/gt<N>/freq0/act_freq` and `cur_freq`;
- power: the change of `hwmon/hwmon*/energy1_input` over time, and `power1_cap`;
- temperature: `hwmon/hwmon*/temp*_input`, with each `temp*_label` as the sensor;
- activity: the change of `tile0/gt<N>/gtidle/idle_residency_ms` over time;
- throttle: `tile0/gt<N>/freq0/throttle/reason_*`, with the reason in the file name.

`nagaya.device.evicted` stays amdgpu only.

Council decision metrics (count, spread, confidence) belong to the council layer (11.1).

**9.5 Each instrument has a call site under test.** The first Gemma 4 engine called some instruments only from its
tests, and its docs named instruments that did not exist. Thus a test shows that each instrument records on the path
that it describes.

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
as a number. A stream ends later than its response head, so the point does not cover the reply.

**9.6 Evidence on failure.** Before and after a quantum that fails, the nagaya reads the machine's counters. It logs
the change with the failure. The Qwen3.8 engine reads these counters: KFD evicted ms, MemAvailable, GTT, and from
`/proc/vmstat` compaction stalls, migrations, swap-outs and direct scans.

Built as `nagaya::evidence` in `crates/nagaya/src/evidence.rs`, with fixed trees in the tests. The code that runs a
quantum calls `Machine::watch` before the quantum and `Watch::failed` in its error branch. `GroupJobs::run_one` does
this on the group thread. The tests are `crates/nagaya/tests/job_evidence.rs` and
`crates/nagaya/tests/process_evidence.rs`. A read costs four files, so the group thread skips `watch` for decode
quanta.

A decode quantum that fails still logs the failure. The line says `evidence unavailable (not read before a decode
quantum)` (`evidence::failed_unwatched`). The failure text names the kind of the quantum, the kind of request and the
engine. `Start::machine` gives the process a machine with a fixed tree of files for a test.

**A refused job is not a failed quantum.** A job that finds no free slot ends in its first quantum, and its caller gets
`busy` (11.3c). A request of several replies then runs that reply again (11.15). The engine and the machine did nothing
wrong, so the group thread logs no failure line and reads no evidence for it. It logs one line at the level `info` that
starts with `job refused` (`evidence::refused`).

The body reports `Ran::Refused`, and the job ends as failed. Its quantum span has the outcome `refused` and no error
status.

Reason: a request of 8 forks on an engine of 4 slots gives 8 replies. A failure line for each refusal would put 4
lines at the level `error` in the log for each such request.

Evidence never hides its failure. A counter that cannot be read, or that went back, is named in the line as
`evidence unavailable`. The failure is logged all the same. The sources are `/proc/meminfo`, `/proc/vmstat`,
`/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms` and `/sys/class/drm/card*/device/mem_info_gtt_used`. The GTT is
summed over the devices, and it is absent on a machine with no amdgpu device.

**Evidence on Xe.** Xe has no counter of evicted ms, and no GTT in the amdgpu sense. Thus a Xe machine reads these
sources in their place:

- the memory of the nagaya's own DRM clients: the `drm-total-*` and `drm-resident-*` keys in `/proc/self/fdinfo/<fd>`
  of each `xe` fd;
- the throttle state of each GT: `device/tile0/gt<N>/freq0/throttle/status` and `reasons`;
- `MemAvailable` and `/proc/vmstat`, as on any machine.

The fdinfo reader sums the keys once per `drm-client-id` across `/proc/<pid>/fdinfo`, and the GT reader reads each
`card*/device/tile*/gt*/freq0/throttle` entry. A source that is absent is `evidence unavailable`, as above.

**9.7 Status feed.** Each machine has one SSE status feed:

- a snapshot when a client connects;
- one `quantum` event for each quantum, with engine, device group, kind, ms, and queue length by class;
- a keepalive every 15 s.

The feed disconnects a slow subscriber. It does not skip events for it. A subscriber that wants less volume asks for
a filter by engine or kind when it connects.

Not built. The HTTP server exists (11.7), and it has no status route. `ObservedScheduler` sees every quantum, and
the job path uses it now (9.2), so it can feed the bus.

**9.7a The events.** The route is `GET /nagaya/v1/status` (11.4a). The feed is `text/event-stream`. Each event has an
`event:` line and a `data:` line of one JSON object. Not built yet.

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
data: {"t_ms": 91234, "machine": "node-a", "engines": [{"engine": "mk-qwen38-gfx1151", "device_group": 0,
  "state": "ready", "queue": {"interactive": 0, "batch": 0, "filler": 0}, "jobs": 0}]}

event: quantum
data: {"t_ms": 91790, "engine": "mk-qwen38-gfx1151", "device_group": 0, "kind": "prefill", "class": "interactive",
  "request": "generation", "ms": 412, "outcome": "done", "queue": {"interactive": 1, "batch": 0, "filler": 0}}

: keepalive
```

In a real event, the `data` is one line.

- **Decode events.** A decode quantum makes an event too. Decode makes metrics only in 9.2, but a feed is for a
  person. A client that wants less volume filters by `kind`.
- **Order.** The events of one group are in order. The events of two groups have no order. `t_ms` tells the client
  which came first.
- **Filters.** A filter is the query `engine` or the query `kind` of the connect request. Each takes values joined
  with commas. A name that is not an engine of the machine, or a kind that does not exist, is a 400 `invalid_request`
  that names the query. Reason: an empty result from a typo would hide the fault.
- **What a filter drops.** The `kind` filter drops quantum events only. The `engine` filter also drops entries of
  the snapshot.
- **The keepalive.** The keepalive comes every 15 s without an event. A machine with no engine sends a snapshot with an
  empty `engines`, and then the keepalive.
- **Open feeds.** At most 16 feeds are open at once. A request past that is a 429 `busy`. A feed does not count
  toward the 64 calls of 11.7, because it holds no call thread.
- **A stop.** A stop ends each feed after the `draining` state (10.3). A client reconnects and gets a new snapshot.

**9.7b Where the events come from.** Not built yet. The group thread produces the events, and a bus delivers them.
The code already holds each value.

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
  quantum falls between them.

Tests that can fail: a fake engine runs a build and a failing quantum, and the test reads the events from a
subscriber. It checks each member of 9.7a against the metric and the span. A subscriber that never reads must be
dropped, and the group must go on.

Reason for the group thread as the source: `ObservedScheduler` does not know the request kind or the evidence. Thus
the scheduler stays a pure recorder of metrics and spans.

## 10. Lifecycle

**10.1 systemd.** The unit is `Type=notify`. The process sends `READY=1` when every engine in the profile has
opened, passed its canary, and woken its pinned state. `STATUS=` gives the state of each engine.

**One unit file for each machine.** The unit of the Strix Halo example is `deploy/strix-halo/nagaya.service`. The unit
of the B70 example is `deploy/b70-pair/nagaya.service`, next to its config and profile. Reason: the units differ in the
engine variables, the cargo feature of the build and the OTLP line. One shared unit needs a drop-in file on each
machine, and a drop-in is not in the repository.

When `<store_dir>/profile.toml` does not exist, the process starts with an empty profile. An empty profile is valid,
because a machine with no engine can serve as a coordinator or hold a store tier (2.1). A process with an empty
profile sends `READY=1` when admission passes.

Not built: the canary of the other engines, and the wake of pinned state (8.1, 6.11). The Qwen3.8 engine runs its
canary inside its open, so `READY=1` follows it. `READY=1` follows the open of every resident engine and the attach
of the council routes. An engine on demand does not open at start (3.3a, `crates/nagaya/src/process.rs:1149-1154`).

**`READY=1` follows the HTTP bind.**

- The process binds the HTTP address before the engines open. A bind that fails refuses the start (exit 3, 10.4).
- Until the council routes attach, each request gets a 503 `unavailable` that says the process is starting.
- The process sends `READY=1` after the council routes attach. Thus a client that connects at `READY=1` gets an answer.

Reason: if the server listens after `READY=1`, a client that connects when `systemctl restart` returns gets a refused
connection.

A request during the start gets `unavailable`, not `model_unavailable`. Reason: `model_unavailable` names the model
and its status, and the identity is known only after the engines open. gogi lists `unavailable` for a server that
starts or stops. A client tries again later.

While the engines open, the process tells systemd that the start goes on. Every 5 s (the heartbeat interval), it
sends one datagram. The datagram holds `EXTEND_TIMEOUT_USEC=` and a `STATUS=` line that says `loading` for each
engine. The extension is 30 s, the watchdog timeout (7.4). systemd sets the start deadline to 30 s after each
datagram, so a slow open does not time the start out. A process that stops sending the datagram times out within 30 s.

When the wait for the opens ends (7.4), the process sends `STATUS=` with the name of each engine that did not open.
Then it exits 1.

**10.2 Arenas through a restart.** systemd's file-descriptor store keeps the arenas (`FDSTORE=1`). Each fd has the
name `<engine>/<device>/<part>`. systemd reads `FileDescriptorStoreMax` from the unit file before the process starts.
The unit file sets a generous ceiling (`FileDescriptorStoreMax=512`), and profile admission (2.4) refuses a profile
that needs more arena fds than the ceiling.

At start, the nagaya does these steps for each engine:

1. Check each stored arena against the engine's build, and record the strength of the check.
2. If an arena does not match, make sure that a cold load fits the ledger first (2.4). Then drop that engine's
   stored fds and load it cold. If a cold load does not fit, keep the fds and exit 3.
3. Duplicate each fd before use. Reason: an import can close the stored fd, and that causes a crash loop.

`systemctl stop` clears the store. A restart keeps it.

**The names of the fds.**

- `<engine>` is the engine name in the profile. `<device>` is `device<id>`, with the machine device id (2.4).
  `<part>` is a name that the engine gives in `arena_export`.
- The parts of the Gemma 4 engine are `w<ll>` for the slab of layer `ll`, and `head` for the head on card 1. Thus its
  arena has 31 fds, such as `gemma/device1/w15`.
- The head is about 1.5 GB: 262,144 rows of 2,816 BF16 values. It has a slab, so a restart does not upload it again.
- The engine opens its stored fds through `Factory::attach` (engine-abi "The factory gets its devices"). The check is
  `spot`: the first and last 64 bytes of each tensor (`verify_stored_slab` in
  `crates/nagaya-mk-gemma4-bmg/src/gemma4.rs`).

**The ceiling and the start steps.**

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

Built: the process side (`crates/nagaya/src/arena.rs`, `crates/nagaya/src/process/handoff.rs`) and the arena of the
Gemma 4 engine, with its head slab (`Loaded::attach`). A cold `Loaded::load` takes about 7.1 s, and an attach from
the 31 slabs takes about 1.2 s (`tests/xe_arena.rs`, release build). A process restart takes about 7.1 s to
`READY=1` cold and 1.2 s hot (`crates/nagaya/tests/xe_restart.rs`). The unit of the B70 machine sets
`FileDescriptorStoreMax=512` (`deploy/b70-pair/nagaya.service`). The Qwen3.8 engine exports no arena yet.

**10.3 Stop.** On SIGTERM, the nagaya does these steps:

1. Accept no new jobs.
2. Let each running quantum go to its end.
3. Park pinned and wanted contexts to NVMe in a time budget (ds4-recipe: 10 s). The budget is in the bootstrap config
   (2.1).
4. Exit.

A second signal makes the process exit at once. Cancellation of running work follows 5.9. **Open:** the conversion
to canonical safetensors (D10) is not yet measured, so the count of contexts that fit in the budget is not known.

**Queued quanta and the park budget.** A stop cancels each queued quantum that has not started, and the submitter
gets a cancelled result (5.9). Only the running quantum goes to its end. The park budget of a group starts when the
drain of that group ends. A drain that does not end is a stall, and the process exits 1 (7.4). When a second signal
interrupts a stop, the process exits 1, unless every state was parked.

A park that runs past the park budget makes the process exit 1. A SIGTERM during a planned restart stops the
process with exit 0.

**Which states the park keeps.**

- The park stores each state that holds tokens. It skips a state with no token, because that state has nothing to
  keep.
- The park reads the row count of a state with `slot_high` (engine-abi "Capabilities"). A count of 0 means no token.
- An engine with a cells or ring part needs the `slots` capability. Without it, the park cannot count the rows, so
  the stop fails and the process exits 1.
- An engine with no cells or ring part has no row count. The park keeps each of its states.

Reason: step 3 parks pinned and wanted contexts. The process does not track pins or wants on live states yet, so
it parks each state with tokens. An empty state of the Qwen3.8 engine is 118,038,528 bytes of fixed parts ("v0 binding
choices"). A park of the Qwen3.8 engine's 16 empty states would write about 1.9 GB and use most of the 10 s budget.

**A park of bytes that the store holds.**

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

Reason: a put under the park recipe of bytes that the store holds fails with "the content id already has another
recipe", and the stop exits 1. The recipe of a build names the inputs, so it is a better key than the park recipe.

**Open:** the park's copy of a state in host memory has no staging charge (4.2 "Staging"). The park holds one copy
at a time, on the group thread. A copy of a 32,768-token Qwen3.8 context is about 1.93 GB. The charge path
(`Ports::admit_staging`) refuses while the process stops, and the group thread holds no `Ports`. A charge for the park
needs a ledger path that the stop keeps open.

The process catches SIGTERM with a handler that adds one to a static atomic counter. The handler is `signal(2)`,
declared in `extern "C"`, because std has no signal API. The main loop reads the counter every 50 ms. Reason: this
needs no new dependency. A later step can use a signal stream from `tokio` instead.

**10.4 Exit codes.**

| Code | Meaning | systemd |
|---|---|---|
| 0 | a clean stop | no restart |
| 1 | a crash | restart |
| 3 | a refusal to start: a bad bootstrap config (2.1), a probe error (2.1), a profile that does not fit, a canary mismatch at start, a mismatch of a part size or of the weight arena size, a store that does not open, a gossip bind or an HTTP bind that fails, or a council service that does not attach (11.7) | `RestartPreventExitStatus=3` |
| 4 | a planned restart: a new profile (2.3, 3.3) | `RestartForceExitStatus=4` |

The restart policy is `Restart=on-failure`, `RestartSec=30`, and at most 3 starts in 600 s. systemd counts each start
against that limit, also a start after code 4. Thus a fourth profile swap in 600 s fails.

An engine that does not open at start stops the process. A mismatch of a part size or the weight arena size is a
refusal to start: exit 3. Any other failure to open is a crash. The process exits 1, and systemd restarts it within
its start limit. A message on stderr names the engine and the reason.

**One exit path.**

- The binary ends through one function, `nagaya::exit::leave`, for every code.
- First the process drops the telemetry, so the OTLP batches flush (9.1). Then `leave` flushes stdout, stderr and the
  C output streams.
- Then `leave` calls `_exit(2)`. It runs no `atexit` handler and no static destructor.
- The fake build takes the same path, so the host tests run the path of the GPU build.

Reason: the Qwen3.8 engine's HIP runtime stops the program in its static destructors, as its teardown does (3.3). A
clean stop with `exit(3)` prints "pure virtual method called" and ends with SIGABRT (status 134) and a core dump.
Nothing in the process needs an `atexit` handler.

## 11. Client API

**11.1 Council and coordinator routing.** The nagaya serves council v2 for all engines in the cluster. It does not
serve v1, because no client speaks v1. The contract is gogi (ontology, "The other projects"): the text
`docs/council-api.md`, the types in `gogi-proto`, and the suite in `gogi-conformance`.

gogi is normative. This spec says how the nagaya serves the API, and it never restates the API. When this
spec and gogi differ about the API, gogi is the rule, and this spec is corrected.

**Text only.** Each served model of the nagaya reads and writes text only, until a family renders an object. Its
identity declares no `media`. The server declares `objects` when its profile names an `objects` budget (6.15,
`crates/nagaya/src/http/mod.rs:306`). The routes refuse an object part, `case_media` and `output` with a 400 that names
the member (gogi "Contexts", "Outputs"). A content of text parts is the context of their string.

**The text stays in gogi.** gogi keeps the text, the types, the suite and the toy family. This spec keeps only how the
nagaya implements the API.

Any machine can receive any request and act as its coordinator (Cassandra style). Clients prefer a coordinator with
separate host memory (the B70 machine or a CPU rig). Thus a machine with unified memory saves its bandwidth (3.2). The
coordinator finds the target engine from the request's `model` field and the `chitchat` cluster state (2.3). Because
the coordinator has the family libraries (11.3) and the engine's `identity()` from gossip, it can render, tokenize
and compute the recipe locally. It then schedules the job locally or forwards the work to the machine that holds the
engine.

The forward of 11.1a replaces the local render and recipe of the last paragraph for each job route. A coordinator
forwards the whole request, and the node of the engine renders it. The local render stays the later design of a job
split across nodes.

Routing to the machine that holds the deepest chain is deferred. Reason: a typical fleet is heterogeneous, and its
clients want sticky sessions with each model and machine (after DeepSeek's DeepInfer).

The API has contexts, specs, decisions, snapshots, pins, reads, and generations with riding decisions. In council v2
responses, a snapshot `id` is its BLAKE3 content id (`blake3:<hex>`), and the store looks up prefixes by recipe
(6.6). gogi sets these request rules:

- Each request names a served model (ontology) in `model`. A request without it is a 400.
- `identity.models` lists every served model in the cluster, with its status from gossip (2.3).
- The statuses are `ready`, `loading`, `draining` and `unloaded`. A fatal engine's served models report `unloaded`.
- A served model that is not `ready` gets 503 `model_unavailable`.
- `class` is `interactive` (the default) or `batch` (5.3). `priority` is an `i32`, 0 by default (5.4).
- A request carries no tolerance. Its served model sets the tolerance (6.6).

**Capabilities.** Each served model declares `reads`, `generate`, `dry_run` and `leave_one_out`. The server declares
`pins` and `classes`. Each text model also declares `continuation` (11.3c). `describe`, `think`, `ride` and `tiers`
are not built. A request that needs one of them is a 400 that names it (gogi "Errors").

**Thinking is required.** Each context names `thinking`, `true` or `false`, and the nagaya has no default (gogi
"Contexts"). A text model declares `thinking` when its family thinks (`Family::thinks`). Then its contexts take
`thinking: true`. A context with `thinking: true` for another served model is a 400 that names `thinking`, or
`contexts[i].thinking` in a decision or a raw read. The route checks it before the job, also for an image or a
speech model.

Pooling and the decision numbers come from `gogi_proto::math`, which all engines share. The council
`identity.engine` covers two things: the engine identity, and the nagaya code that turns logits into decisions.
(The Qwen3.8 engine puts the SHA-256 of `libqwen38engine.a` in `build`, as `engine_build_sha256`. The digest does
not cover it.)

**A text engine on demand.** Reason: an engine on demand, such as the Qwen3.8 engine (`docs/deploy.md`), does not
open at the start. Without these rules it is no served model, and gogi finds no ready model.

- A factory can describe an engine: the identity and the limits that its open will report, with no GPU and no load
  (`Factory::describe`). The Qwen3.8 engine describes itself from its shards, its build and its knobs.
- At the start, the council serves each engine on demand that its factory describes, from the description. The model
  is `unloaded` until a load, and a request for it gets a 503 `model_unavailable` (3.3).
- An open that reports another identity than the description is an open error. The council's recipes would have
  the wrong root (6.2).
- A kind that cannot describe its engine is not a text model while its engine is on demand.

**The description of Gemma 4.** Reason: a profile can put Gemma 4 on demand, and without a description the server
serves no `gemma-4-26b-a4b`. Its identity names the Level Zero driver, which no file holds. A description is cheap,
but it can read the checkpoint on the CPU for fidelity.

- `Factory::describe` gets the devices of the group, as `plan` does. The Qwen3.8 engine does not read them.
- `mk-gemma4-bmg` describes itself from the checkpoint, sysfs and the version of the Level Zero driver. It asks the
  driver for its version, and it opens no card: no context, no queue and no module.
- The load reads the vendor, the device and the revision of each card from sysfs too. It refuses a card whose Level
  Zero ids are not the sysfs ids.
- The factory keeps each identity that it described. An open of the same checkpoint, cards and block that reports
  another identity is an open error. The check needs no second hash of the checkpoint.

**11.1a The cluster.** A node does the work that it can do with what it holds, and it forwards the rest. For
example, a node that holds the tokenizer tokenizes the text.

- The routes `tokenize`, `detokenize` and `render` (11.4b) run on the node that takes them when it holds the family of
  the model. Otherwise they forward.
- A job route forwards the whole request to the node that holds the engine. The response or the stream comes back as
  it is.
- A job split across nodes comes later.
- A `GET` of an object that this node lacks asks each live peer (6.15, "A read on a miss").
- `GET /nagaya/v1/cluster` gives the gossip view (11.4a).

**The identity of the cluster.**
Reason: gogi lists every served model of "the server", and through any node the cluster is one server. Each node must
give the same list.

- `identity.models` holds the served models of this node and of each live peer, in byte order of `model`.
- Each field of a peer's model comes from its `served` entry in gossip (2.3). The status comes from the state of its
  engine, and `fatal` gives `unloaded` (11.1).
- A model whose node leaves the live set stays in the list, with the status `unloaded`. The coordinator keeps the
  last entry that it saw since its start.
- Reason: a request for that model then gets the 503 `model_unavailable` of a model that comes back. It does not get
  the 400 of a name that the server never serves (gogi "Served models"). The other choice was to drop the model.
- A coordinator that starts while a peer is down does not list the models of that peer.
- A served model that loses its name to another node (2.3) is not in the list.
- The server `limits`, `capabilities` and `extensions` are those of the coordinator (11.1, 11.4f), with one change.
  The coordinator declares `pins` when any live node declares it. `pin_bytes` is then the smallest value among those
  nodes.
- Reason: objects and specs go to the store of the node that takes the put (6.15, 11.1a). Thus their limits are its
  own. Pins go to each node (11.1a, "The routes with no model"), so their limit must fit each node.
- `extensions` holds `/nagaya/v1` when `identity.models` is not empty, because the text routes then forward
  (11.4f).

**The cluster call.** Each call from a node to its peers goes through one layer of the code, in two forms:

- **Routed.** The node knows the peer from gossip: the holder of a served model (2.3) or of an engine (3.3b). The call
  goes to that peer only ("The forward" below).
- **Broadcast.** The node does not know which peer holds the thing, such as an object, a snapshot or a spec by id. The
  call goes to each live peer at once.
- A broadcast read takes the first answer that is not a 404, and the rest are dropped. A 404 comes only when each peer
  gave a 404. Content ids name the same bytes on each node, so any hit is correct (6.2).
- A broadcast write goes to each live peer, and the route says how it joins the answers. A 4xx or a 5xx of a peer is
  its answer. The layer does not hide it and does not try again.
- Each call carries `nagaya-forwarded` (below), so a peer answers from its own store and calls no other node. A peer
  that does not connect in 2 s is left out, and the log names it.
- Later, gossip can carry a bloom filter of the ids that each node holds, or a table of routes. Then a broadcast
  becomes a routed call. The routes keep their rules.

**The forward.** Reason: the rules of 11.1a name what forwards. They do not say how a coordinator finds the node, what
it sends, or how a forward fails.

- **The node of a model.** The coordinator finds the live node whose `served` entries name `model`, as a `model` or
  an alias (2.3). A model of this node never forwards, unless another node keeps its name (2.3).
- **Not ready.** The coordinator reads the status of the model from gossip first. A model that is not `ready` gets
  the 503 `model_unavailable` at once, as a local model does (3.3).
- **The job routes.** These routes carry `model`, and they forward: `POST /council/v2/contexts`, `decisions`,
  `reads`, `generations` and `embeddings`.
- **The text routes.** `render`, `tokenize` and `detokenize` run locally when a local family has the tokenizer hash
  and the template of the model in gossip. Otherwise they forward as a job route does. 11.4b still applies.
- **The same bytes.** A forward sends the method, the path, the query and the body bytes as they came. It sends the
  headers `content-type`, `accept` and `tracestate`, a new `traceparent` (9.2), and the two headers below.
- **The response.** The coordinator sends the status, the headers and the body of the target as they come. It does
  not parse or change them. A stream goes on event by event, and a closed client stream closes the forward.
- **No second forward.** A forward carries the header `nagaya-forwarded`, with the name of the coordinator. A node
  never forwards a request that carries it.
- A node that takes such a request for a model that it does not serve gives a 503 `model_unavailable` with the status
  `unloaded`. Reason: gossip can lag, and two nodes with old views must not send a request back and forth.
- **One request id.** A forward carries the header `nagaya-request-id`, with the request id of the coordinator. The
  target uses it as its own request id (11.7). Thus the client, both logs and both spans name one id.
- **A peer that does not answer.** A connection that does not open in 2 s gets a 503 `model_unavailable` with the
  status `unloaded`. The message names the node.
- A target that leaves the live set ends each forward to it. Before the status, the response is the same 503. In a
  stream, it is an `error` event with that body. Otherwise the connection closes.
- Reason for `model_unavailable`: gogi keeps `unavailable` for a server that starts or stops (gogi "Errors"). Here the
  coordinator runs, and one model is not reachable.
- **No other timeout.** A forward has no read timeout. The target keeps its own deadlines, such as `timeout_ms` (5.8).
  Reason: an image of 40 steps takes 176 s on the Strix Halo machine.
- **A duplicate node name.** A coordinator never forwards to a node name that has a live duplicate (2.3). The request
  gets the 503 above, and the message names the conflict.
- **The limits.** A forward runs on the control runtime, not on a call thread. It does not count toward the 64 calls
  of 11.7, and the target counts it. The body limit of 16 MiB holds on both nodes.
- **A stop.** A stopping coordinator answers each new request with the 503 `unavailable`, also for a model of a peer.
  An open forward gets the wait of 2 s of 11.7, as a local request does.

**The open points of the forward.** These rules fill the choices that the rules above leave open.

- **No trace.** A coordinator with no trace layer has no span context to send. Then the `traceparent` of the client
  goes on as it came, so the target can still join the trace of the client.
- **A broken connection.** A connection to the target can break while the target is still live in gossip. The forward
  ends as `lost`, as when the target leaves: a stream gets the `error` event, and another body closes.
- **Framing headers.** The response keeps each header of the target, except `transfer-encoding`, `connection` and
  `keep-alive`. They describe one connection, and the coordinator frames its own.
- **The request id.** A target takes `nagaya-request-id` only from a request that also carries `nagaya-forwarded`, and
  only a valid id (gogi "Request ids and tracing"). Other requests get a new id.
- **A bad value of a peer.** A peer value in gossip that does not read is logged once as an error. Its models are left
  out of the identity, and the node runs on. Reason: the value of one peer must not stop another node (2.3, "Trust").
- **The local family.** A local family has a model of a peer when a local served model has the same `tokenizer_hash`
  and `template` in its identity entry. The response of a text route then names the `model` of the request.

**The routes with no model.** Reason:
gogi says that snapshots, pins and specs belong to the server, and here the server is the cluster. A client must get the
same answer from each node.

- **Reads by id.** `GET /council/v2/snapshots/{id}` and `GET /council/v2/specs/{spec_id}` read the local store first.
  On a miss, they make a broadcast read of the cluster call (11.1a), as objects do (6.15). A 404 comes only when no
  node holds the id.
- **Specs.** `POST /council/v2/specs` holds the spec on the coordinator only. `GET /council/v2/specs` lists the local
  specs only.
- A forwarded decision takes its spec to the target first. The coordinator posts the spec that it holds, and the
  target gives the same id. A spec that the coordinator lacks does not stop the forward: the target's 404 names it.
- `DELETE /council/v2/specs/{spec_id}` goes to each live node. It is a 204 when any node held the spec, and a 404 when
  no node did.
- **Pins.** `PUT`, `GET` and `DELETE /council/v2/pins/{owner}` go to each live node that declares `pins`. Each node
  pins the snapshots that it holds.
- A snapshot is `held` when any node holds it. A `GET` joins the sets of the nodes, and `expires_in_s` is the
  smallest. An owner that no node knows is a 404.
- A 413 from a node is the response, and the sets of the other nodes stay as the `PUT` made them. Known limit: gogi
  says that a `PUT` past the limit pins nothing. The client sends a smaller set, which replaces the set on each node.
- **Keeps.** `/council/v2/keeps/{owner}` follows the rules of pins, over the live nodes that declare `objects`
  (6.15).
- **The calls to peers.** Each call to a peer carries `nagaya-forwarded` (above), so a peer answers from its own store
  and asks no other node. A peer that does not answer in 2 s is left out, and the log names it.
- The other choice was to keep these routes on the coordinator only. Then a pin through one node could not hold a
  snapshot of a model of another node.

**The open points of the routes with no model.** These rules fill the choices that the rules above, 3.3b, 2.4 and
6.15 leave open.

- **The join of a broadcast write.** A route of pins, keeps or specs gives the first answer that is not a 2xx or a
  404. The order is this node, then the peers in byte order of the name. A 413 is one such answer.
- **The join of the sets.** The `held` of an id is true when any node held it. The `expires_in_s` of a `PUT` is the
  smallest that a node gave, and `null` when no node held an id.
- **A node with no home.** A node that serves no text model holds no spec, pin or snapshot. A read by id on it asks
  the peers. When no peer holds the id, its own 503 stays the answer.
- **The spec of a forward.** A post of the spec that the target refuses is the response, and nothing forwards. This is
  the rule for the put of an object (6.15).
- **A target that does not connect.** A `HEAD`, a put or a post before a forward that does not connect in 2 s gives the
  503 `model_unavailable` of the forward.
- **The forward of a load.** A peer that does not connect in 2 s gives a 503 `unavailable`, and the message names the
  node. A load names no served model, so the 503 has no `model`.
- **A desired profile past the bound.** A value of `profile:desired:<host>` past 8192 bytes (2.3) is a 413
  `too_large` with the param `profile`.
- **The order of a desired profile.** The target admits it, checks its names, persists it and acts on it, through the
  code of a local `PUT`. It records the version after that code answers, also for a refusal. It writes
  `profile:active` again after each accepted change.

**Context records in the cluster.** Reason: the forward sends the same bytes (above), and a record is a file of one
node (6.17).

- **Records belong to a node.** Each node holds the records of the contexts that it reads. Gossip carries no record
  and no record id.
- **A forward sends `parent` as it came.** The coordinator does not make the context whole. The target makes it whole
  from its own records. A target that does not hold the parent gives gogi's 404, and the client sends the whole
  context. Reason: the response of a forwarded build or generation names a record of the target.
- **The objects of a forward.** The coordinator takes to the target the objects that the body names (6.15). It does
  not read the parent, so it takes no object of the parent. The target got those objects when it first read the
  parent. If it evicted one, its 404 names the message of the parent, as for any object.
- **The read by id.** `GET /council/v2/contexts/{id}` reads the local records first. On a miss, it makes a broadcast
  read to the live nodes that declare `pins`, as a spec read does. A 404 comes only when no node holds the id.
- **Pins.** A `PUT` of pins carries `contexts` to each live node that declares `pins`, and each node pins the records
  that it holds. The join gives `contexts` the rules of `snapshots`: a record is `held` when any node holds it, and a
  `GET` joins the lists.
- **The limit.** `limits.context_records` in identity is the limit of this node.
- Known limit: a record that node A holds is not a parent for a model of node B. The client gets the 404 and sends
  the whole context once. The other choice was that the coordinator makes the context whole. Then the two forms give
  ids on two nodes, and the coordinator must parse and write each forwarded body.

**Trust.** The cluster trusts the
tailnet. Each node binds its HTTP and gossip addresses on the tailnet (2.1, 11.6). It trusts the gossip of each peer and
each request with `nagaya-forwarded`, and it checks no credential in v0. gogi has no authentication either (gogi "Served
models").

**11.2 Generation.** The nagaya runs one sampler for all engines. It applies top_k, top_p, min_p, then temperature.
The draw is seeded, and the response reports the seed. Each engine has its own defaults. All engines share the SSE
framing.

**A speech engine samples on the card.** The one sampler is for text. A speech engine makes 16 samples for each 80 ms
frame, each from the one before (engine-abi "Speech"). The engine draws them on the card, from the seed of the request.
Its draw keeps no state either: the random value depends on the seed, the frame and the codebook. Reason: 16 host round
trips for each frame would cost more than the frame.

**The draw keeps no state.** The random value for a token depends on three things: the seed, the position of the
token, and the purpose of the draw. The purposes are a sample, a resample and the test of a draft.
The sampler keeps no random state between tokens. A fork gets a child seed, a hash of the parent's seed and a branch
number.

Reason: a park, a restore or a fork then keeps the sampler's bits, with nothing to save. A client that sends a
request again with the same seed gets the same tokens. A speculative draft (such as a prompt-lookup draft) has
its own purpose, so it never moves a later draw. Thus the reported seed does not change when drafts are on.

**The draw function.** These rules give the bits.

- The position is the token's position in the context, from 0. It is not its index in the reply.
- The purposes are numbers: a sample is 0, a resample is 1, and the test of a draft is 2.
- The value is the BLAKE3 hash of `nagaya draw v1`, then the seed, the position and the purpose. The seed and the
  position are 8 bytes each, little-endian. The purpose is 1 byte.
- The uniform value is the first 8 bytes of the hash, little-endian, shifted right by 11 and times 2^-53. It is in
  [0, 1).
- A child seed is the first 8 bytes of a BLAKE3 hash, little-endian. The hash is of `nagaya fork v1`, the parent
  seed and the branch number, each 8 bytes little-endian.
- The pick is the first Qwen3.8 engine's inverse CDF: the survivors in order of logit down and id up, in float64.

Thus the first Qwen3.8 engine's own uniform value, put into the nagaya's pick, gives the first Qwen3.8 engine's token.
The sampler's tests use that.

The first Qwen3.8 engine's sampler draws from one PCG64 stream for each request. The nagaya does not match those
bits. It matches the first Qwen3.8 engine's filter order and its probabilities.

**11.3 Text front-end.** Each model family has its own text rules. They are a tokenizer, a template, escapes for
special tokens in client text, and parsers for tool calls and thinking. The nagaya holds one library per family
(ontology "Family"), and the engine identity selects it by its `tokenizer_hash` and `template`. Engines never see
text. They take tokens, rows, or both (D2).

Thus one renderer serves every engine of a family, on any vendor's card.
The rendering rules in gogi, such as where a boundary exists and where thinking renders, get one implementation and
one set of conformance tests. The first Qwen3.8 engine's C ABI does not need a tokenizer.

**The Qwen3.8 library.** The library follows these rules.

- The library reads the template from the GGUF key `tokenizer.chat_template` and runs it with a Jinja interpreter.
  It does not hold a hand-written copy. The template is part of the identity, so a new template in a GGUF needs no
  new code.
- The interpreter matches the first Qwen3.8 engine's Python on four points. `trim` strips what Python's `str.strip`
  strips, which includes U+001C to U+001F. `tojson` writes what Python's `json.dumps` writes, with `ensure_ascii` off. A
  map key that is an integer, a float, a bool or none becomes a string as in Python.
  A map keeps its key order.
- A value in `{{ }}` and the `string` filter follow Python's `str`. Thus none is `None` and a bool is `True`. A float
  is its `repr`, and a string in a list has quotes.
- A render refuses the same input as the first Qwen3.8 engine. A template error is an error. A message that holds a
  Unicode noncharacter is an error, because the first Qwen3.8 engine keeps those for its escapes. The error names the
  character and its offset in characters, as the first Qwen3.8 engine does. A message of `raise_exception` reaches the
  error.
- The first Qwen3.8 engine lets some Python exceptions pass, such as a `TypeError`. The library gives a template error
  for those.
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
- **Messages.** The library refuses a `messages` value that is not an array. This is a deliberate difference. The first
  Qwen3.8 engine passes any value to jinja2. For a map or a string, the loop runs over keys or characters by accident.
  The API takes a list of messages. A caller that sends another type has a bug, and the refusal shows it (8.5).
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
  the range of a signed 128-bit integer. Python accepts those, so this is stricter than the first Qwen3.8 engine.
  It refuses nesting of more than 512 levels.
  Python stops near 1000 levels.
- The GGUF reader bounds its allocations. A string of more than 16 MiB is an error, and a string is read in chunks.
  An array grows with the values that the file gives. A key twice in one file is an error.
- The tokenizer reads `tokenizer.ggml.token_type` of any integer type. A value that does not fit a signed 64-bit
  integer, and any float, is an error. A special token with an empty text is an error. The first Qwen3.8 engine builds
  an empty alternative that splits the text at every position, so a file like that is a mistake.
- The parsers for tool calls and thinking match the reply tracker of the first Qwen3.8 engine. Fixed outputs of its
  Python test them.
- **The finish.** The first Qwen3.8 engine gives only `stop` and `length`. The parser maps a stop token with one or more
  entries in `tool_calls` to `tool_use`, and a stop token with none to `end_turn`. An entry that did not parse counts as
  a call, because the model tried one. The caller says whether a cut came from `max_tokens` or from a full context. The
  parser makes no call ids: the server names each call.
- **The pieces of a stream.** The first Qwen3.8 engine's `piece` is the text of the token. It holds `</think>` in the
  reasoning pieces, and the letters of a marker that arrive in separate tokens. Thus gogi's rule for pieces fails on the
  first Qwen3.8 engine's pieces in 64 of 211 cases of the fixture. The rule says that joined pieces equal the part, up
  to `strip()`.

  The parser also gives the text that the first Qwen3.8 engine's tracker released for each token (`released`). That text
  keeps the rule in every case, with the end of thinking left out. The reply parser of the family library writes the
  pieces of a step from `released`.
- One token can release text in two phases, but a gogi `token` event has one phase. This is rare, because each
  Qwen3.8 marker is one token. 11.3a has the rule for the step of such a token.

**The Gemma 4 library.** The crate `nagaya-text-gemma4` holds the Gemma 4 family. It ports the text code of the first
Gemma 4 engine, which served council v2.

- **Tokenizer.** The library reads the `tokenizer.json` of the checkpoint with the `tokenizers` crate, as the first
  Gemma 4 engine did. The tokenizer hash is of that file (12.3).
- **Template.** The library runs Google's chat template with the nagaya's own Jinja interpreter, with `trim_blocks` and
  `lstrip_blocks`, as the first Gemma 4 engine did. The template is the crate's own copy,
  `crates/nagaya-text-gemma4/assets/gemma4_chat_template.jinja`, copied from the first Gemma 4 engine. Not
  `minijinja`. The identity's `template` is the text of that file. The variables that the library sets, such as
  `PRESERVE_THINKING`, are code, so the test renders of 11.3a cover them.
- **Escapes.** Client text that spells a marker is content. Before the render, the first Gemma 4 engine swaps each
  marker text for a sentinel of Unicode noncharacters. It refuses a noncharacter in client text.
- **Council rendering.** The library ports the council rendering of the first Gemma 4 engine, the origin of 11.3a,
  with its Gemma 4 layout. The template id follows 11.3a.
- **Reply parser.** The library ports the tool call parser and the thought channel of the first Gemma 4 engine.
- **The method `get` of a dict.** Google's template calls `message.get(...)`. The interpreter holds `get` with the
  results of Python: a missing key gives the default, which is none. It refuses a keyword argument, as Python does.
  The other methods of a dict stay refused. Reason: the template needs
  it, and transformers renders it so (`crates/nagaya-jinja/tests/dict_get.rs`).

**The Gemma 4 choices.** These rules apply where the rules of the first Gemma 4 engine do not carry over, or where the
rules above leave a choice open:

- **A failed tool result.** The first Gemma 4 engine gives the template the map `{"error": content}`. In Python's
  jinja2 a map is a sequence, so Google's template loops over its keys and fails. The library writes `Error: ` before
  the content, as the Qwen3.8 library does. Reason: one form for both families, and a render that Python gives too.
- **One form in every family.** A family library gives a failed tool result the form `Error: ` and the content. It
  uses another form only when its model's template needs that form.
- **Noncharacters.** The first Gemma 4 engine refuses only U+FDD0 to U+FDEF. The library refuses all 66
  noncharacters, and the error names the character and its offset, as the Qwen3.8 library does. Reason: the words of
  the rule above.
- **The reply parser** ports the reply splitter of the council wire of the first Gemma 4 engine, because its chat code
  parses only tool calls. Its stop tokens are `<turn|>`, `<eos>`, `<|tool_response>` and `<|turn>`. A marker token
  gives an empty piece.
- **A character that a run of text does not finish.** At `<|channel>`, `<channel|>`, `<|tool_call>` or `<tool_call|>`,
  the parser drops its bytes, as the first Gemma 4 engine does. At a stop token or a cut, `finish` gives them as U+FFFD
  in the tail, in the phase of the last token. Reason: a marker gives an empty piece, and the pieces of a part join to
  its text.
- **Boundaries.** A marked message is a boundary when the tokens of its rendering begin the tokens of the whole, as
  in the Qwen3.8 library. Two boundaries with the same token count keep the first.
- **Reads.** A plain read and a queue read keep the text of the first Gemma 4 engine and its tokenization piece by
  piece. A read hashes the whole text that it was conditioned on (gogi), as the Qwen3.8 library does. The first Gemma 4
  engine hashes the id of the head.
- **Detokenize.** Every family library refuses tokens whose bytes end inside a character. The error names `tokens`
  (11.4d). No library puts U+FFFD in their place (`crates/nagaya-text-gemma4/src/council/mod.rs`,
  `decode_whole`). A piece of one token is not a detokenize, so `pieces` keeps U+FFFD (`Family::pieces`).
- **Host tests** read the checkpoint directory from `GEMMA4_MODEL_DIR`, with no default path. With the variable unset,
  a test skips. With `NAGAYA_REQUIRE_MODELS=1`, it fails.

Reason for `tokenizers`: the port copies code that its Python fixtures already check.

The Jinja interpreter moves out of `nagaya-text-qwen38` into a crate
`nagaya-jinja`, which both family libraries use. The move changes no behavior, and the Qwen3.8 fixtures stay green
through it. A feature that Gemma 4's template needs and the interpreter lacks is a step with a test that fails first.
Reason: one template engine for every family, with no second copy of it.

**The engine registry.** The engine identity selects the family library, not the engine kind. These rules apply:

- The identity gets one more field, `family`: `toy`, `qwen38` or `gemma4`. The engine sets it.
- The process holds a registry of family libraries by that name. A family library opens from the identity and the
  table `[engines.family]` of the profile, which names its files.
- The opened library must give the `tokenizer_hash` of the identity, as 11.7 says now. Another hash refuses the
  start (exit 3).
- `context_tokens` of the served model is the largest slot of the engine, in tokens. `SlotLimit` has a member
  `tokens` for it.

Reason: the identity names what the engine computes, so it selects the text rules too (11.3). A new engine kind then
needs no change in `serving.rs`.

**Registry choices.** These choices fill what the rules above leave open.

- The identity digest (12.1) does not cover `family`. The family follows from the model, the tokenizer hash and the
  template, which the digest already covers. A new digest would make every held entry unusable.
- An engine whose family has no library is not a served model, and the log says so. This is the rule of 11.7 for
  an unknown kind, now keyed by the family.
- The key `gguf` of `[engines.family]` names the GGUF file of the `qwen38` family. With no key, the library reads the
  first shard of `QWEN38_SHARDS`. The `toy` family needs no file and no tokenizer hash.
- The registry of engine factories holds each kind that the build has. The feature `hip` gives `mk-qwen38-gfx1151`
  and `qwenimage21`. `xe` gives `mk-gemma4-bmg`, `qwen3tts-bmg`, `mk-sdxl-bmg` and `mk-embeddinggemma2-bmg`. `fake`
  gives the fake kinds (`crates/nagaya/src/admission.rs:196-219`). A build with no kind refuses the start (exit 3).
- The key `model_dir` of `[engines.family]` names the checkpoint directory of the `gemma4` family. With no key, the
  library reads `GEMMA4_MODEL_DIR`. The engine config reads the checkpoint in the same way (engine-abi "The first
  Gemma 4 binding").
- `context_tokens` is `SlotLimit.tokens` of the largest slot, not that value less one. The Qwen3.8 engine declares
  32,768.

**11.3a Council rendering.** A decision and a raw read put a spec, a case and a letter menu after each context. The
nagaya renders them with the rules of the first Gemma 4 engine, ported to each family.
The family library holds the port, because the tokens belong to the family.

A served model's `template` in council identity has two parts: `nagaya-council:<hash>+<family>-chat:<hash>`.

- The first hash is over the tokens of a fixed set of test renders. The family library keeps the set: specs, cases
  and contexts. Thus the id changes only when the rendered tokens change.
- An edit to a comment or a change in the code structure keeps the id. Thus it keeps the thresholds that clients
  fitted (gogi rule 6).
- The id has no hand version number. The hash does that work.
- The second hash is the digest of the chat template text (12.1).
- Each hash is the first 16 hex digits of a SHA-256.


**Open:** the first hash covers the tokens of the test renders and nothing else. A change in how the family parses a
reply does not change the id.

**The Qwen3.8 rendering.** These rules fill the choices that the spec and gogi leave open for Qwen3.8.

- Every render uses the reasoning effort `medium`, as the Qwen3.8 engine's council does. The template then writes no
  effort line.
- Thinking follows the context: the render uses `thinking` of the context. A context renders the same with thinking on
  and off, except for the generation prompt. A past assistant message keeps its reasoning in both. A test compares the
  two renders.
- A read closes the thinking. The case turn ends with the template's prompt for thinking off:
  `<think>\n\n</think>\n\n`.
- A plain read keeps the layout of the first Gemma 4 engine: spec turn, case turn, `Answer to question`, then
  ` N (stem):`. The letter token has a leading space, as in that engine. A raw read uses the letters that the client
  sends.
- A plain read tokenizes the template text and the client text in one pass, as the Qwen3.8 engine renders a chat. The
  first Gemma 4 engine tokenizes the pieces one by one, and the two ways can differ at a join. The library checks that
  the tokens of the context begin the tokens of the read. It also checks that each step begins with the tokens before
  it. A failed check is an internal error (`council/read.rs`).
- A queue read writes the call in Qwen3.8's form, with the parameter `answers` as one JSON object. It reads a bare
  letter after the opening quote of the value.
- A queue read asks the questions in the order of the properties of `answers` in the queue tool's schema. Qwen3.8's
  template writes the keys in the order it gets them. The first Gemma 4 engine sorts by question id, because Gemma
  4's template sorts.
  A test pins the order.
- The workspace turns on the `serde_json` features `preserve_order` and `float_roundtrip` for all its crates, because
  Cargo joins the features of one build. The queue read needs `preserve_order` to read the order of the schema.
  Without it, the map sorts its keys, and the questions come in the wrong order. `float_roundtrip` keeps a float of a
  schema or a call equal to the float that was sent. A build of one crate alone may lack both.
- The template has no form for a failed result. The library writes `Error: ` before the content of a tool message with
  `is_error`.
- A system message after a message of another role renders as a system turn, in its place. gogi allows it
  (`docs/council-api.md`: "A system message may come later"), and `gogi-conformance` sends one. The template refuses
  such a message, so the library gives it a user message with stand-in text. Then the library swaps that user turn
  for a system turn with the trimmed content. The template merges the system messages at the start into one turn.
- A marked message is a boundary when the tokens of the rendering up to it begin the whole rendering. A tool message
  inside a run of tool messages is not a boundary, because the template puts one user turn around the run. A later
  system message ends the run.
- The template keeps the reasoning of every assistant message. No marked message loses its boundary for that reason.
- `rendered_hash` is the content id of the text that the read was conditioned on (gogi's words). The first Gemma 4
  engine hashes the head's snapshot id. A plain read hashes the text through `Answer to question`, which all its
  questions share. A queue read hashes its whole text at the end: the call through the last letter that the read chose,
  and its closing quote.
- The reply parser follows these rules.
  - It takes the stop token as a pushed token (`<|im_end|>` or `<|endoftext|>`). It adds no text for that token.
  - A token has the phase of the last text that it released. The step of the token holds only the text of that phase.
  - A token can release text in an earlier phase too. That text is in the fields of the whole reply, and in no step.
  - The end of the thinking is not part of the reasoning.
  - A call that parses but holds an integer of more than 64 bits becomes a malformed call. Its text stays as written.

**11.3b The toy family.** The fake engine has no real tokenizer. For tests, it serves the toy family of gogi's
`gogi-toy` crate. Its tokens are the 256 byte values and 14 template markers, and its template is the template of
`gogi-fake`. The family library `nagaya-text-toy` puts the nagaya's family interface on `gogi-toy`, and the fake
engine gives logits over the toy vocabulary. Thus `gogi-conformance` runs against the process with the fake engine, and
needs no GPU and no GGUF.

**11.3c The job path of a build.** A build renders a context, finds the longest held prefix, restores it, feeds the
rest, and holds a snapshot at each boundary. These notes fix the choices that the text above leaves open.

- **The store and the group thread.** `lookup` and `restore` take the engine, and the engine lives on its group
  thread (3.2). Thus a build runs them on the group thread. The store sits behind a mutex that the group threads and
  the park thread share. The ledger stays on its own thread, and a build asks it by message.
- **The quanta of a build.** The first quantum has the kind `copy`. It chooses a slot, looks up the prefixes and
  restores the longest one. It then adds the other quanta to the job.
- **The kinds of the later quanta.** A feed of at most `max_pass_inputs` tokens is `prefill`. An engine with a feed
  rule feeds one block in each quantum (6.14). A seal and a put at a boundary is `copy`. A feed never crosses a
  boundary.
- **The seal.** A build calls `seal` at each boundary that it passes, before it reads the parts. A snapshot is a
  context sealed at a boundary (ontology). An engine without `seal` skips the call.
- **The slot.** A build takes the lowest free slot where the prompt fits (`fits` on each cells part). A build that fits
  no slot is `too_large`. A build with no free slot is `busy`.
- **What a build lists.** The response lists the boundaries that the store holds when the build ends. A boundary
  that eviction removed, below the start of the build, is not made again. A dry run feeds nothing and lists no
  snapshot.
- The head of a dry run. A dry run reports the held head when the store holds it. Otherwise the head is `blake3:` and
  the hex of the head's recipe. It names the state that a build would make. No store entry has the id, so a read or a
  pin of it is a 404 or `held: false`. The id does not show that it is a recipe: a client tells that from
  `dry_run: true`.

  Reason: gogi types `head` as required, and gogi is the contract. gogi allows only `blake3:` and 64 digits as a
  snapshot id (6.2). A recipe is a whole BLAKE3 hash already (6.2).
- **The pin owner.** gogi names an owner with text, and the store names it with a `u64`. The service keeps the text and
  gives the store the first 8 bytes of the BLAKE3 hash of the text. A restart keeps the text in `pins.json` in the store
  directory. Entries that builds put have the owner 1.

**Reads and generations on the job path.** These rules apply to reads and generations.

- A raw read, a decision read and a generation are each one job for each context. The job builds the context, as a
  build does, and its work goes on in the same slot. When the store holds the head, the job restores it.
- A decision read feeds the prefix of its plan once. A plan can say that its prefix ends at a boundary. Then the
  job builds the prefix with the context, and the store holds its head (11.11, "The plain read of the LFM3
  family"). The first question that starts after the prefix keeps a copy of the state in memory. Each later
  question of that kind writes the copy back into the slot first. A raw read does the same for each suffix after
  the first.
- The usage of a read counts the tokens that it fed: the shared prefix once, and no letter after the last question.
  gogi-fake counts the prefix once for each question, and the last letter of a queue read. Both keep
  `kept + fed = prompt`.
- The reads of a decision run one after another, in request order. The service checks `timeout_ms` before each
  read, and the scheduler checks it between quanta (5.8). Either one ends the decision with 504 `timeout`.
- A raw read lists 8 top tokens when the request names no `top`, and at most 64. A larger `top` is a 400.
- A build of more than `context_tokens` tokens is a 413 `too_large` that names `context_tokens`. So is a raw read
  whose context and longest suffix have more. A context of exactly `context_tokens` tokens builds. `context_full` is
  only for the prompt of a generation (gogi "Errors").
- The text of a top token is the detokenized token. A token that is not whole text alone, such as one byte of a
  longer character, has the text U+FFFD. Its `id` names it exactly.
- The council identity's `engine` is `nagaya:` and the first 16 hex digits of the identity digest (6.2). The digest
  covers each field of `identity()`, `numbers` too.
- A generation holds its head after the reply when both of these facts are true. Otherwise `head` is null.
  - The rendering of the context with the reply has the boundaries of the prompt before the reply.
  - The slot holds a prefix of that rendering, at most 8 tokens short.

  Then the job feeds the rest and puts the snapshot. The head of a reply that ends with `max_tokens` is the seam
  of its open rendering (see continuations below). On an engine with a feed rule, the head is a decoded head, and
  6.14 has its rules.
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
  `max_tokens`, it ends with `context_full` (gogi "Generations", `finish`).
- `decision_context_tokens` is `context_tokens`: a decision reads its context in the slots of a build, and the job
  takes the lowest slot that fits. **Proposed** (Claude Fable 5.2). Reason: the owner said on 2026-11-17 that a
  decision of LFM3 Omni uses its slots of 32,768 positions. The limit was 3,072, and one large image has 1,875
  positions. The same rule for each engine is the simplest one.
- The other decision limits are those of the first Gemma 4 engine: `contexts_per_decision` 8, `case_bytes` 65536
  and `default_timeout_ms` 30000.

Continuations follow gogi's rules (`docs/council-api.md`, "Continuations"). The nagaya uses them in
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
- **The Gemma 4 rendering.** Google's template renders a closed
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
  field that does not tokenize on its own is a 400 that names `continue`. This is the prefix check of the prefill of the
  first Qwen3.8 engine.
- **The parse.** The reply parser starts in the phase that the open message sets (`ReplyStart::of`). It reads only
  the sampled tokens. After an open message, the Qwen3.8 parser keeps the white space at the start of its first part.
  At `max_tokens`, it keeps the white space at the end of its last part (`Keep` in
  `crates/nagaya-text-qwen38/src/reply.rs`). The toy parser and the Gemma 4 parser trim nothing.
- **The reply.** `done` joins the open text as sent and the new text. `usage.prompt` counts the open text. `n` and
  `usage.completion` count only the sampled tokens (`GenerateThen::finish`, `crates/nagaya/src/council/generations.rs`).
- The build plan: a continuation builds the context before its open message (`Service::open_plan`,
  `nagaya_council::continuation::open_plan`). Then it feeds the open message as a generation feeds its generation
  prompt, with no seal. Reason: a seal changes bits (the Qwen3.8 engine rounds the GDN state to bf16). A seal inside the
  open text makes a continuation differ from the reply that it continues. The next request starts from the head after
  the reply.
- The head of a cut reply: a reply that ends with `max_tokens` seals a snapshot of its open rendering. It gives
  that snapshot as `head`, listed last in `snapshots` (`GenerateThen::cut_head`). A continuation of that reply
  starts from the head: the longest held snapshot wins, as for any build. It may differ from the uncut reply, which
  gogi allows (boundaries are content).
  - The build plan above stays. The build also looks up the seam of the open text after the plan
    (`build::Extension`, `ContextBody::restore_seam`). When the store holds it, the build restores it and the
    generation feeds the last token. The state keeps the seal of the seam, so the head after that reply has the seam
    among the boundaries of its recipe (`recipes_with_cuts`).
  - A cut reply that the client sends back closed starts from the head too, in any form. The open rendering's
    tokens up to the seam must begin the closed rendering's tokens. The search of the store's index finds the head
    (see the cuts above). So does a context that sends back the reply of a continuation from a head: that reply's
    head chains through the seam.
  - The cuts from the index. A build tries each sealed snapshot that the search of the index finds, not only the
    ends of messages. Reason: a try of the open forms of the last assistant message misses a reply cut inside its
    thinking and sent back with `content: ""`. It also costs three renders of the prefix. The search also takes the
    place of a second render of the cuts at message ends.

    **Open:** can the index hold a few tokens of prefix or suffix, so that the search is more targeted? Section 15
    lists the question.
  - The timings of a dry run come from a release host test (`crates/nagaya/tests/council_heads.rs`). For 18026
    tokens with an assistant message, a pass over the index takes 38 ms. It is 16 ms with one render. The same context
    without the message takes 16 ms. With 10,000 unrelated entries in the store, it takes 0.22 ms (0.16 ms when
    empty).
  - A dry run reports the interval copy that a build would restore. For a continuation, it asks for a slot of the
    open rendering, as the generation does.
  - A continuation whose reply's closed tokens differ from its prompt and sampled tokens holds no head after the
    reply. Two causes: the template trims white space at the end of the reply, and a token at the seam merges in
    the closed rendering. The slot holds the other tokens, so no head can name them.
  - The seam. The head is the seam of the open rendering, the position before the last sampled token, not its end.
    Reason: a snapshot holds no logit row. A continuation from a head at the end would have no first logit row, and
    it cannot feed the last token again. At the seam it feeds that token (`fed` 1), which gogi allows.
- The dry run: a build with an open message and `dry_run: true` looks up the plan of the continuation and its seam.
  Its `kept` and `fed` are what the generation keeps and feeds. Its `head` is `blake3:` and the recipe of the open
  rendering, which the store never holds (see the head of a dry run above).
- The template id: the test sets of the families have no open message, so the open rendering is not in the template
  id (11.3a). Adding one would change the id of each family once, and a fitted threshold does not carry over.
- A riding `before` is not built, because the nagaya does not serve `ride`. A generation with `decisions` is a 400
  that names `decisions`.

**11.4 The routes of the first Qwen3.8 engine.** The nagaya does not serve the route set of the first Qwen3.8 engine:
`model`, `render` and `generate`. kaijutsu uses council v2 and the gogi reference. In council v2 the server renders
messages with its model's own template, and the template is part of identity (gogi `docs/council-api.md`, "Terms").

**11.4a Extensions.** gogi puts model-specific routes under the server's own prefix, listed in
`identity.extensions` (gogi `docs/council-api.md`, "Extensions"). The nagaya's prefix is `/nagaya/v1`:

- `profile` and `cluster`: dynamic placement profiles and the `chitchat` cluster view (2.3, 2.4);
- `cas/{content_id}`: plain HTTP `GET` and `HEAD` for content-addressed snapshots and tap artifacts across machines;
- `render`, `tokenize` and `detokenize`: served by the family library for bench tools and semantic search;
- `status` (9.7) and `traces` (13.1).

`render`, `tokenize` and `detokenize` (11.4b to 11.4d), the prefix in `identity` (11.4f), `profile` (3.3) and the
load routes of 3.3b are built. The other routes in this list are not built. Embeddings are not in this list: gogi's
contract has them as `POST /council/v2/embeddings` (11.10).

**The cluster route.** Reason: the route gives the gossip view (11.1a), and the web app finds its hosts there. It
needs a fixed shape.

```json
GET /nagaya/v1/cluster
-> {"node": "node-a",
    "nodes": [{"name": "node-a", "id": "node-a:1793577600000:<address>:8092", "self": true,
               "gossip": "<address>:8092", "http": "<address>:8091",
               "engines": [{"name": "gemma", "state": "ready", "models": ["gemma-4-26b-a4b"]}],
               "conflicts": {"name": null, "models": []}, "refused": null}],
    "dead": ["node-c"]}
```

- `node` is the name of the node that answers. `nodes` holds the live nodes, this node too, in byte order of the name.
  `dead` holds the names that the failure detector marked dead (2.3).
- `id` is the chitchat id, `name:generation:address` (2.3). `gossip` is the gossip address. `http` comes from the key
  `server` (2.3).
- `engines` comes from the keys `engine:<name>`, in byte order of the name. `models` lists the `model` of each
  `served` entry.
- `conflicts.name` is the value of `name:conflict`, and `conflicts.models` holds the entries of `model:conflict`.
  `refused` is the value of `profile:refused` (2.3).
- A key that the node does not write gives `null`, or an empty list for a list.
- The route takes no body and no query, and it never forwards. A view from one node can lag the view of another by a
  few gossip rounds.
- The other choice was to give each key of each node as raw JSON text. It is the whole view, but each client must
  then parse the keys itself.

**11.4b The text routes.** `render`, `tokenize` and `detokenize` use the family library (11.3). They share the rules
below.

Bench tools use them to see what a council request feeds. The semantic search of kaijutsu uses them to count and to cut
text by tokens.

- **The model.** Each body has a required `model`, as a council request has. A body without it, or with a name that
  no served model has, is a 400 `invalid_request` that names `model`. The routes never choose a model for a client
  (gogi "Served models").
- **A model that is not ready.** A served model that is not `ready` gets a 503 `model_unavailable`. The route calls
  `Council::route` (`crates/nagaya/src/http/mod.rs`), as the council routes do. The other option was to serve the
  text routes while the model loads, because they need only the family. This rule keeps one rule for all routes, and
  `identity` tells a client when the model is `ready`.
- **No engine.** The routes use no engine, no ledger charge and no job. The call runs on a call thread, and it counts
  toward the 64 calls (11.7). A body has the limit of 16 MiB (11.7).
- **Unknown members.** A body with an unknown member is a 400 that names it, through `parse`
  (`crates/nagaya/src/http/errors.rs`). Thus `class` and `priority` are 400, because the routes do not schedule.
- **The response.** A response names the served model by its `model`, never by an alias. It carries the `request-id`
  header, as every response does.
- **The span.** The HTTP server span of 9.2 covers each call, and its `http.route` is the path, such as
  `/nagaya/v1/render`. The routes make no `nagaya.job` span and no metric of their own.

The code is in `crates/nagaya/src/http/extensions.rs`. The function `routes` in
`crates/nagaya/src/http/routes.rs` merges its routes, so `router` and the running server both serve them. `Service`
has one accessor, `Service::family` (`crates/nagaya/src/council/mod.rs`), because its field `family` is private. The
tests are `crates/nagaya/tests/extensions.rs` (the toy family), `extensions_qwen38.rs` (the real tokenizer and
template) and `http_telemetry_extensions.rs` (the span and the duration point).

A `FamilyError::Client` becomes a 400. The existing `From<FamilyError>` for `ServiceError`
(`crates/nagaya/src/council/error.rs`) does this. A `FamilyError::Internal` is a 500, and the process stops (8.5).

**The `param` of a family error.** A family error can have no `param`. Then the route sets it to the request member at
fault: `text` for `tokenize`, `tokens` for `detokenize`, and `context` for `render`. A family error that has a `param`,
such as `continue`, keeps it (`at` in `extensions.rs`).

**11.4c Render.** `POST /nagaya/v1/render` gives the text and the tokens of a context, as a council request feeds
them. It is for bench tools and for the debug of a template.

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

- **The render call.** The route calls `Family::render(&context, generation_prompt)`
  (`crates/nagaya-council/src/family.rs`). The type `Context` is `gogi_proto::context::Context`, so the route
  decodes it as `POST /council/v2/contexts` does. The implementations are `Qwen38Family::render`
  (`crates/nagaya-text-qwen38/src/council/mod.rs`) and `ToyFamily::render` (`crates/nagaya-text-toy/src/lib.rs`).
- **The text.** `text` is `Family::detokenize(&rendering.tokens)`. For Qwen3.8 it equals `Rendered.text` of
  `Qwen38::render` (`crates/nagaya-text-qwen38/src/family.rs`). A test pins this equality. `Rendering` has no text
  member, so this needs no change to the trait.
- **The tokens and the boundaries.** `tokens`, `end`, `boundaries` and `signals` come from the `Rendering` as they
  are. The response never reports a boundary that a build would not keep.
- **Control text.** Content that spells a marker is plain text in `tokens` (gogi rule 2). In `text` it looks like
  the real marker. Thus `control_text` is the way to tell them apart. A client never sends `text` back as a prompt.
  It sends the context.
- **The place of a hit.** Unlike the council responses (11.7), `render` reports the hits of `message:<i>`. gogi
  names no such place for the council routes. This route is a debug tool, and the place is the answer that it gives.
- **No context limit.** `render` does not check `context_tokens`. A client compares `n_tokens` with the limit in
  `identity`. Thus the route needs no code for `context_full`.
- **The context only.** The route shows the context only. A decision also renders a spec, a case and a letter menu
  after it (11.3a). A later route can show them through `ReadPlan::prefix`. It is not in the first set.

Each of these three rules chooses one of two options.

- **The generation prompt.** `generation_prompt` is `false` by default. This matches a build and a decision, which
  render with no generation prompt, and so `boundaries` match a build. A client that wants the prompt of a generation
  sends `true`.
- **One member for the context.** The context is one member, `context`. The same object then goes to
  `/council/v2/contexts`. The other option was the first Qwen3.8 engine's flat body, with `messages`, `tools` and
  `thinking` at the top. It saves one level, and a client copies less.
- **No reasoning effort.** The route has no `reasoning_effort`. gogi's context has none, and the Qwen3.8 library
  fixes it to `medium` (`EFFORT` in `crates/nagaya-text-qwen38/src/council/mod.rs`). A body with it is a 400.

A message that the template refuses, a context that fails `Context::check`, and a Unicode noncharacter in a string
are each a 400 `invalid_request`. The message names the fault, as the first Qwen3.8 engine does. The `param` is
`context`, unless the fault is in a member that has its own name, such as an unknown member of the context.

The route detokenizes the tokens of the rendering to make `text`. These tokens come from valid text, so a failure
there is a defect: a 500 `internal`, and the process stops (8.5).

**11.4d Tokenize and detokenize.** `POST /nagaya/v1/tokenize` turns text into token ids. `POST /nagaya/v1/detokenize`
turns token ids into text. They are for the semantic search, for a count of tokens, and for a look at the pieces of
a rendering.

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

- **Tokenize.** With `special` off, `tokenize` calls `Family::tokenize`. The text of a marker stays text (gogi rule
  2). For Qwen3.8 this is `Qwen38Family::tokenize`: it escapes the text as content, then it calls `encode_escaped`.
- **Detokenize.** `detokenize` calls `Family::detokenize`. A token outside the vocabulary is a 400 with `param`
  `tokens`. The message of `Error::TokenId` already gives the vocabulary size.
- **Split characters.** Tokens that do not make whole UTF-8 text are a 400 with `param` `tokens`. The route follows
  `Family::detokenize`, because a silent replacement would hide a split character. The other option was a member `lossy`
  that replaces the bytes.
- **Empty input.** An empty `text` gives an empty `tokens`. An empty `tokens` gives an empty `text`. Neither is an
  error.
- **Pieces.** `pieces` are lossy by design. A piece is the bytes of one token, and a token can end inside a
  character. The route replaces bad bytes with U+FFFD in each piece (`String::from_utf8_lossy`). The piece of a
  marker is its text.
- **A cut at the end of a piece.** The first Qwen3.8 engine drops a cut-off character at the end of a piece. The
  route keeps a U+FFFD, so that a client sees the
  cut. A token of several stray bytes can give more than one U+FFFD, and a top token in a raw read (11.3c) always gives
  one.
- **The members of each route.** Each route has its own members. `special` and `pieces` in a `detokenize` body, and
  `tokens` in a `tokenize` body, are each a 400 that names the member. A `tokens` member must be an array of unsigned
  32-bit integers. Any other value is a 400 with `param` `tokens`, and the message names the place of the bad id.

Two small additions to the `Family` trait (`crates/nagaya-council/src/family.rs`) serve `special` and `pieces`.

- **`tokenize_special`.** `Family::tokenize_special(&self, text: &str) -> Result<Vec<u32>, FamilyError>`. Qwen3.8
  calls `Tokenizer::encode`. This is the rule of a raw-read suffix (gogi "Raw reads"), where a marker is one token.
- **`pieces`.** `Family::pieces(&self, tokens: &[u32]) -> Result<Vec<String>, FamilyError>`. Qwen3.8 calls
  `Tokenizer::token_piece(id, true)` for each id, then converts with lossy UTF-8
  (`crates/nagaya-text-qwen38/src/council/mod.rs`). `ToyFamily` and the test family `Holding`
  (`crates/nagaya/tests/council_generations.rs`) implement both. Both methods are required, not defaults, so a family
  that does not implement them does not compile.
- **The other option** was to build the first set with neither `special` nor `pieces`. `tokenize` then needs no
  change to the trait. The cost is a bench tool that cannot tokenize a raw suffix.

Tests that can fail (in `crates/nagaya/tests/extensions.rs` and `extensions_qwen38.rs`):

- `detokenize(render.tokens)` equals `render.text`, for each fixture of the family.
- `tokenize` with `special` on, applied to a `render.text` with no `control_text`, gives `render.tokens`.
- The same call, applied to a text with `control_text`, gives other tokens. This pins the behavior of the first Qwen3.8
  engine in its OpenAPI text.
- A noncharacter, an id past the vocabulary, and a split character each give a 400 with the `param`.

**11.4e Model and health.** The first set has no `model` route and no `health` route. Both existed in the first Qwen3.8
engine, and `GET /council/v2/identity` covers most of what they gave.

- **`model`.** The first Qwen3.8 engine's object had `id`, `weight_hash`, `max_context`, `build.engine_sha256` and
  `device`. The council identity has `model`, `weight_hash`, `engine`, `limits.context_tokens` and `device` for each
  served model. The members `hidden_size` and `vocab_size` are not there. A client needs them for a vector space, so
  they belong to the `embeddings` response, which this section does not cover. `Family::vocab_size` already holds the
  vocabulary size.
- **`health`.** The first Qwen3.8 engine answered 200 while it served and 503 after a failure. The nagaya stops the
  process at a failure (8.5), so it has no state that serves and is unhealthy. A probe uses `identity`: a 200 with each
  model `ready` is healthy. A fatal engine reports `unloaded` there (11.1). systemd sees `READY=1` when the routes serve
  (10.1). The watchdog thread ends a hung process (7.4).
- **`weights`.** The first Qwen3.8 engine's `health` told `attached` or `loaded`. The nagaya counts this as
  `nagaya.arena.handoff` (9.4), with the status `hot` or `cold`. No client route is needed.

The other option was a `GET /nagaya/v1/health` that gives only `{"status": "ok"}`, for a load balancer that cannot
read `identity`. It costs one route and no new state.

**11.4f Extensions in identity.** `identity.extensions` lists the prefix `/nagaya/v1` once. gogi's example lists
a route prefix this way, and it lists no route (gogi "Extensions").

- **When the list holds the prefix.** `Council::identity` returns `["/nagaya/v1"]` as soon as the process serves one
  route under the prefix. The text routes need a family. Thus the list holds the prefix when the council has a served
  model, and it is empty otherwise (`crates/nagaya/src/http/mod.rs`).
- **Only routes that exist.** The list claims only routes that exist. A route that is not built is a 404 `not_found`
  (11.7), also under the prefix. Thus a client probes a route by calling it.
- **One machine.** Each machine lists the prefix for its own routes. `status` (9.7) is per machine, and the text
  routes need a family that the machine holds (11.1).
- **The test.** A test calls `identity` and checks that the prefix is present if and only if a route under it
  answers. A test for the empty list fails when the first route is merged without the change.

The other option was a route `GET /nagaya/v1` that lists the routes by name and version. It is more work, and gogi
does not ask for it.

**11.5 Errors.** All routes use gogi's error body, `{"error": {"type", "message", "param"?}, "request_id"}`, and the
`request-id` header. Council routes use only gogi's error table (gogi `docs/council-api.md`, "Errors"). The
`/nagaya/v1` routes add two codes:

| Code | `error.type` | When |
|---|---|---|
| 409 | `conflict` | a profile `PUT` with an old version (2.3) |
| 507 | `no_memory` | a profile that does not fit its budgets (2.4) |

An unexpected failure is a 500 `internal`, and the process stops (8.5).

The routes of 11.4b to 11.4d, and the status feed of 9.7a, need no new code in this table. They use the codes that
the table and gogi's table already hold.

| Code | `error.type` | When, on these routes |
|---|---|---|
| 400 | `invalid_request` | A missing `model`, an unknown member, a bad context, text or tokens, or a bad `engine` or `kind` query. |
| 413 | `too_large` | A body past 16 MiB (11.7). |
| 429 | `busy` | More than 64 calls (11.7), or more than 16 open feeds (9.7a). |
| 500 | `internal` | A family library defect. The process stops (8.5). |
| 503 | `model_unavailable` | The served model is not `ready`. The text routes only. |
| 503 | `unavailable` | The process is starting or stopping. |

**11.6 Binding.** The HTTP and gossip listen addresses and seed peers are in the bootstrap config (2.1).

**11.7 The HTTP routes.** The process serves the council routes with `axum` on the control runtime
(`crates/nagaya/src/http`). It starts the server before the engines open, and it logs `http listening on <address>`
(10.1).

- The engine registry of 11.3 selects the family library of each engine (`crates/nagaya/src/serving.rs`,
  `FamilyRegistry`). "Several text models in one process" below has the rules of more than one text model.
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
- The routes `render`, `tokenize` and `detokenize` under `/nagaya/v1` are built (11.4b to 11.4d). So are `profile`
  and the load routes (3.3, 3.3b). `cluster`, `cas`, `traces` and the status feed (9.7) are not built.
- Continuations are built (11.3c). `gogi-conformance` runs against the process with the fake engine. The number of
  its checks depends on the reply to a random nonce.

**Several text models in one process.** Reason: one process can serve several text models, such as `lfm3`,
`lfm3-omni` and Gemma 4 on the B70 machine. gogi "Served models" says that specs, pins and
snapshots belong to the server, not to a served model. Thus one set of files serves all the text models.

- A process serves each text engine whose kind has a family library, opened or described. A profile can have
  several. Their served model names differ (2.3).
- The server opens the specs and the pin book once, in the store directory. Each text service gets a shared handle
  to them, so two services never write one file.
- The home is the first text service that the process attaches. The opened engines attach first, then the engines
  on demand. The routes of specs, pins and snapshots use the home. A decision of any text model reads the specs
  through the same handle.
- The home reads and pins the snapshots of each text engine and each speech engine of the process. A snapshot names
  the served model of the engine that made it.
- A pin names no served model. It keeps a snapshot of model A while a request names model B, and while A is
  `unloaded` (gogi "Served models").
- The limits of the server come from the config of the home: `pin_bytes`, `case_bytes`, `choice_options` and
  `questions_per_spec`. The limits of each served model stay its own.
- A request names its served model by `model`. The 503 of a model that is not `ready` names that model and its
  status. The load routes name engines (3.3b), so they do not change.
- The other choice was one book of specs and pins for each served model. gogi forbids it: a pin or a spec does not
  name a served model.

**11.8 Speech.**

- One request is a line or a paragraph. The agent orders the requests and paces them. The nagaya has no stream for
  speech, and a speech job can wait behind other work.
- The reply sends the audio in chunks while the frames decode.
- A council generation of the speech model speaks a line (11.8a). The nagaya has no route `/v1/audio/speech`.
- The engine is Qwen3-TTS VoiceDesign. A text prompt designs each voice, so the voice drifts a little between
  requests.
- The speech engine is on demand (3.3a), and the placement profile chooses its card.

**The sampler of a line.** The default sampler is the generation config of the checkpoint: `temperature` 0.9, `top_k`
50, `top_p` 1.0, and `repetition_penalty` 1.05 for codebook 0. A generation can set `temperature` and `top_k` (11.8a).

**The speech family.** The
crate `nagaya-text-qwen3tts` holds the family library of Qwen3-TTS (11.3):

- the tokenizer of the text vocabulary, from `vocab.json` and `merges.txt` of the checkpoint;
- the prompt of a request, as pairs of a text id and a codec id (engine-abi "Speech"), with the template of the
  reference `generate_voice_design`;
- the codec ids of the prompt: the think tokens, the language id, the pad and the start of speech.

The family has no detokenizer and no reply parser, because the output is audio.

**The split of the speech tokenizer.** The tokenizer splits text with Mistral's pattern, as the reference runs.
`Qwen3TTSModel.from_pretrained` passes `fix_mistral_regex=True`, and transformers 4.57.3 applies it to this checkpoint,
whose config names 4.57.3. Reason: the oracle compares with the reference as it runs. Of 28 fixture texts, only text in
camel case gave other ids under Qwen2's pattern.

The family keeps text that spells a special token as that token, as the reference does. Spec 11.3 gives each family
its escapes; the speech route has none in v0.

**11.8a Speech in the council API.**

- A speech engine is a council served model too. A generation with an `output` of `audio/wav` speaks a line.
- A speech context renders its input as it is, and the generation speaks the most recent message.
- A speech context holds state as a chat holds it: a snapshot of the voice, and the head after each spoken line. The
  next line then continues with the audio before it in the model state.
- The frame is built whether or not Qwen3-TTS keeps its voice better with it. A later model may use it better.

**The speech context.**

- The `system` message is the voice design: the text that `generate_voice_design` takes as `instruct`. A context has
  one `system` message, first, or none.
- Each `user` message is a line to speak, rendered as it is.
- An `assistant` message after a line holds that line's audio: an object part of `audio/wav` that this served model
  wrote. Its text, when it has some, is not rendered.
- The last message is a `user` line. A generation speaks it, with `output` of `audio/wav`. Any other last message is
  a 400 that names `messages`.
- `language` is `output.options.language`, `auto` by default. The seed is `sample.seed`. The sampler is the
  checkpoint's, as in 11.8.
- One language renders each line of a context. Reason: a line in the context has
  no language of its own. A generation in another language renders another context, and feeds it.

**The rendering of a speech context.**

1. The voice design: `<|im_start|>user\n{system}<|im_end|>\n`, text positions only.
2. For each line: the assistant role, the codec prefill of its language, its text with the codec pad, `tts_eos`, and
   `tts_pad` with the codec bos. These are the positions of one request today (`nagaya-text-qwen3tts`).
3. After a line with audio: one frame position for each frame of the audio. A frame position is the 16 codes of the
   frame and the trailing text `tts_pad`, as the engine feeds a frame that it made.
4. Before a line that follows audio: one text position of `tts_pad` with the codec end. Reason: the head after a line
   then holds the rows that the engine holds after the line. Thus a generation holds its head with no feed.

A boundary can follow the voice design and each message, by `snap: true`. The end of the context is a boundary.

**Each message ends a recipe.**

- The recipe of a speech context chains at the end of each message, not only at a boundary. A position count ends it.
- `snap: true` marks where a build holds a snapshot. A lookup tries the end of each message before the last line.
- The head of a generation is at the end of the reply. The next context has the reply as a message, so its end has
  the same recipe.

Reason: a text build finds the head of a reply through the seals of its chain (11.3c). A speech context needs none
of that machinery, because each message end is in the chain.

**The codes of an output.**

- The audio of a line is a WAV, but the model state needs its codes. The service keeps the codes of each output that
  it wrote, beside the object, under its content id.
- An `assistant` part names an output of this served model. Reason: the engine has no codec encoder, so it cannot
  make codes from audio.
- The server drops the codes of an output when the object store drops the object.
- Audio whose codes the server does not hold is a 404 that names the part, as an object that it does not hold. The
  client speaks that line again. Reason: the server cannot
  tell that audio from audio whose codes it dropped, and the client does the same for both.

**The WAV of an output.**
gogi leaves the bytes of a stream open.

- The sealed object is a whole WAV: PCM, mono, 16 bits, 24 kHz, with the true sizes in its header.
- A `GET` of the pending id streams the same bytes, except the two size fields of the header. They are `0xFFFFFFFF`,
  because the length is not known while the frames decode.

Reason: a reader of the object gets a valid WAV, and a reader of the stream gets audio before the end.

**The served speech model.**

- Identity lists the speech model with `media.outputs` of `audio/wav`, no inputs, and the capability `generate`. Its
  `context_tokens` is the positions of a slot.
- Its `weight_hash` and `tokenizer_hash` are the engine's after its first open since the start, and empty before.
  Reason: an engine on demand does not open at the start, and a hash of the weights reads 4 GB.
- `sample` takes `seed`, `temperature` and `top_k`. Temperature 0 is greedy: top_k 1. A `top_p` other than 1, a
  `min_p` other than 0, and a `top_k` of 0 are each a 400 that names it. Reason: the engine samples with a
  temperature and top_k, and the gogi suite sends temperature 0. `output.options` takes `language` only.
- Identity marks the model with `media.text_replies: false`. A build, a decision or
  a raw read that names it is a 400 that names `model`. Its `snapshot_bytes` is one row of `talker_kv`.
- The reply has no text, `finish` is `end_turn`, and `usage` counts positions. `snapshots` names the held ends and the
  head.

**The held state of a speech context.**

- A speech snapshot holds the talker KV of each position, as the part `talker_kv` (D10). The store keeps it as any
  snapshot. Each line starts its sampler and its codec again (engine-abi "Held speech state"), so they hold nothing.
- The recipe of a speech context hashes its positions as the recipe of a text context hashes its tokens (6.2). A text
  position is its text id and its codec id; a frame position is its 16 codes.
- A generation holds its head after the line's frames, so the next line feeds only its own positions. Pins keep a
  voice, as they keep a chat.
- A boundary is restorable only before the last line. A generation feeds at least the last line, because the feed
  that starts a line samples its first code.
- The random value of a code depends on the seed, the frame within its line, and the codebook (11.2). Thus the same
  context and seed give the same codes, held or fed.

**A slow reader of speech.**

- A slow reader of a speech stream, or of its pending id, never holds the group thread.
- The job gives each chunk of audio to the call thread through a queue of 64 chunks (`crates/nagaya/src/speech/mod.rs`,
  `QUEUE`). The call thread copies each chunk into the pending output, and it does not wait for a reader.
- The stream itself carries the `output` event and `done` only. The audio goes through the pending id, which each
  reader reads at its own pace. Thus no queue fills because of a client.
- The queue fills only when the call thread stops. Then the group thread waits after its quantum, as today. That wait
  is a defect of the call thread, not of the client.
- A closed stream cancels the line at its next chunk, and the pending id is then a `404` (gogi "Outputs"). The job
  gives its slot back. The client speaks the line again.
- Reason: the rule for images (11.9a) is for a stream that carries events at each step. The audio of a line is an
  object, and its stream holds no audio. Thus the rule for images does not apply to speech.

**11.9 Images.**

- The first image model is SDXL base 1.0. Pony Diffusion V6 and its merges come later, as other weights on the same
  engine.
- The loader reads the diffusers folders and the single file with Stability's tensor names. Thus a Civitai
  checkpoint loads with no new code.
- There is no refiner. The VAE is the VAE that the checkpoint ships, in f32. Thus the inpainting checkpoint of 11.9b
  uses its own `sdxl-vae-fp16-fix`. A VAE in f16 is a choice for the work on image speed.
- The route is council v2 only. An image is an output object, with a pending id while it decodes (gogi "Outputs").
- The family parses A1111 weights, such as `(word:1.2)` and `[word]`. Thus a prompt from Civitai works as it is.
- A request can ask for a preview at each step. First a linear map from latent to RGB, and TAESD later.
- The samplers are Euler, Euler a, DPM++ 2M Karras and DPM++ 2M SDE Karras.
- A request asks for at most 4 images, and each is its own output object.
- An image engine has one card. Each engine is its own served model with its own name.
- Engines on separate cards of one group run their quanta at once (3.2).
- A request never starts a load (3.3).

- The last `user` message is the prompt. The negative prompt is `output.options.negative`.
- The output is `image/png` with an A1111 `parameters` text chunk, so A1111 and Civitai tools read the settings.
- The engine makes the noise from `sample.seed` with its own random stream. A seed from another tool does not give
  its image. Reason: the stream is simpler.
- The engine makes its stream and runs its sampler on the host, not on the card (`noise.rs` and `sampler.rs` of
  `nagaya-mk-sdxl-bmg`). Reason: the latents of 4 images of 1.5 MP are about
  6 MB, so a step reads and writes them cheaply. On the host, the sampler follows the f32 math of diffusers step by
  step, and its test needs no card.
- A preview is a stream event, `preview`, that names a small PNG object. The server caches it and does not keep it.
- Width and height are multiples of 64 from 512 to 2048, with at most 1,572,864 pixels. The default is 1024 by 1024.
- The quanta are one `embed` for the text encoders, one `step` for each step, and one `step` for the VAE decode. The
  images of a request share each quantum (5.2).
- The defaults are 30 steps, a CFG scale of 7, and the sampler `dpm++ 2m karras`. A request asks for 1 image by
  default and previews are off.
- `clip_skip` is an option now. The family config of the served model gives its default.
- An image job holds no state in the store. Each request starts with empty latents.
- A hires pass is not built, and the design leaves room for it. LoRAs are built (11.9a). img2img and inpainting are
  built (11.9b).

**11.9a An image request.** Reason: the rules of 11.9 need names, ranges and errors before the family and the engine can
test them.

- A generation names the image model and has `output` with `mime` `image/png`. A generation with no `output` is a 400
  that names `output`, as for any model with `media.text_replies: false`.
- The context has one `user` message, or a `user` message after other messages that the server does not render. The
  text of the last `user` message is the prompt. A `system` message is a 400 that names `messages`. Reason: a later
  rule can give it a meaning.
- `output.options` takes these members. A member out of range, or an unknown member, is a 400 that names it, such
  as `output.options.steps`.

| Member | Range | Default |
|---|---|---|
| `negative` | a string | `""` |
| `width`, `height` | multiples of 64 from 512 to 2048; at most 1,572,864 pixels together | 1024 |
| `steps` | 1 to 150 | 30 |
| `cfg` | 1.0 to 30.0 | 7.0 |
| `sampler` | `euler`, `euler a`, `dpm++ 2m karras`, `dpm++ 2m sde karras` | `dpm++ 2m karras` |
| `count` | 1 to 4 | 1 |
| `clip_skip` | 2 or 3 | from the family config of the served model; 2 for SDXL base |
| `spacing` | `leading` or `trailing` | `leading` |
| `loras` | a list of at most 8 `{"id", "weight"}` | an empty list |

- `clip_skip` counts the layers of the text encoders as A1111 does. `clip_skip` n takes the hidden state of layer -n,
  so 2 is the penultimate layer. Reason: a recipe
  from Civitai then works as written. The `clip_skip` of diffusers is 1 less. The last layer, A1111's 1, is not built.

- `spacing` sets the timesteps of `euler` and `euler a`, as diffusers' `timestep_spacing` does. `leading` is the spacing
  of SDXL base. `trailing` gives the steps that a few-step checkpoint, such as SDXL-Lightning, needs. The DPM++ samplers
  take their timesteps from their sigmas, so `spacing` with them is a 400 that names `spacing`. `spacing` is a request
  option, not a setting of the served model. Reason: one served model can run both kinds of weights.
- `loras` lists objects of safetensors weights, each with its `weight` from -2.0 to 2.0. Each `id` is a `blake3:` id of
  an object (6.15). An id that the store does not hold is a 404 `not_found`, as for any object (6.15).
  A request has at most 8 entries, in the weight range above. Reason: the base weights never change, and each job
  picks its own set.
  - The scale of a LoRA is folded into its up matrix on upload. Reason: the
    engine adds each delta with no further scale, and the base weights never change.
  - A LoRA that the engine refuses makes the job a 400. The message names `output.options.loras` and the key. Reason:
    nothing ran, so it follows the other refusals of an image job.
  - A LoRA key is `lora_unet_` or `lora_te1_` or `lora_te2_` with the kohya names, or `unet.`, `text_encoder.` or
    `text_encoder_2.` with the diffusers names. Each module has `lora_down` and `lora_up`, or `lora_A` and `lora_B`,
    and it may have `alpha`. A key that names no module of SDXL base is a 400 that names the key. Reason: a silent
    skip would give an image with less of the LoRA than the recipe says.
  - The scale of a module is `weight * alpha / rank`. Its rank is the first dimension of `lora_down`. Without
    `alpha`, the scale is `weight`.
  - The rank of a module is at most 128. A larger rank is a 400 that names the key and the cap. Reason: the job's
    scratch memory grows with the largest rank.
  - Each targeted linear and convolution adds `scale * up(down(x))` to its output. The base weights are not changed,
    so a job with no LoRA gives the same bits as before. Reason: a removal by subtraction leaves rounding drift.
  - The text encoders take their LoRAs in the same way, on each encoder's linears.
  - The A1111 chunk writes the ids and the weights in the field `Lora ids`, in the order of `loras`. Each entry is
    `<id> <weight>`, and the list is in double quotes, as A1111 writes its lists.

- `sample` takes `seed` only. Image `i` of a request uses the seed plus `i`, as A1111 does for a batch. A request with
  no seed gets a random seed, and the A1111 chunk records it. Any other member of `sample` is a 400 that names it.
- `max_tokens` bounds no image. The reply has no text, `finish` is `end_turn`, and `completion` is 0.
- `done.outputs` holds one PNG for each image, in index order. `usage.media.written` counts them and their bytes.
- A full queue of a stream ends that stream, and the job runs on. Each image is sealed into the store by the job, so its
  pending id still gives the PNG. The encode of a PNG runs on a helper thread, not on the group thread. Reason: a slow
  client then never stalls another engine.
  - Speech has its own rule (11.8a). A slow speech reader never holds the group thread.
  - The queue of a stream holds 8 events. The helper thread is the worker of the route, which encodes each PNG. The
    worker never waits on a client, and the job waits only on the worker. A client that goes away still cancels the
    job (11.7). A full queue alone lets the job run on.
  - The queue keeps its last two places for the `output` events and the `done` event. A full queue drops each later
    `progress` and `preview` event, and the stream stays open. Thus each client gets the pending ids and the sealed ids,
    and the worker never waits. Reason: a cut of the stream can drop the `output` event, and then the client cannot
    find its images.
- A stream sends an `output` event for each image when its decode starts. `GET` of a pending id gives the PNG when it
  is sealed, because a PNG has no useful part before its end.
- A stream sends the events `progress` and `preview` of gogi's council API ("Steps and previews"). Reason: other models
  that run in steps use the same events, such as a text diffusion model.
  - After each step, a stream sends `progress`, `{"index", "step", "of"}`, for each image. `of` is the step count of
    the request. The VAE decode is not a step of `progress`.
  - The request member `previews: true` asks for previews, and needs `stream: true`. It is not an option of
    `output`. Without `stream`, it is a 400 that names `previews`.
  - With `previews`, the `progress` event of each image has its `preview` after it: `{"index", "step", "of", "id",
    "mime"}`. `id` names a PNG at one eighth of the size of the image. The store caches it as any object, and no
    keep holds it. `usage.media` does not count it.
- The A1111 chunk is the text `parameters`, in A1111's form: the prompt, then `Negative prompt: ...`, then `Steps,
  Sampler, CFG scale, Seed, Size, Model, Clip skip` on one line. The model is the served model name.
- Two changes to the rules above, so that the gogi suite passes.
  - A `system` message is not a 400. The server does not render it, as it does not render the other messages before
    the last `user` message. Reason: the suite opens the context of its output check with a `system` message
    (gogi-conformance `checks/mod.rs`).
  - `sample.temperature` is not a 400, and it has no effect. `top_p`, `top_k` and `min_p` stay a 400. Reason: the
    suite sends `temperature` 0 with its output check (`checks/outputs.rs:36`).

**The served image model.**

- Identity lists it with `media.outputs` of `image/png`, no inputs, `media.text_replies: false`, and the capability
  `generate`. A build, a decision or a raw read that names it is a 400 that names `model`.
- Its `context_tokens` is the tokens of the prompt that the family takes: 3 chunks of 75 tokens. Its `snapshot_bytes`
  is 0, because it holds no state.
- Its `weight_hash` and `tokenizer_hash` are the engine's after its first open since the start, and empty before
  (3.3a "Council models on demand").
- The name of the served image model is the name of its engine in the profile, such as `sdxl-0`. The A1111 chunk names
  it too. Reason: 11.9 gives each engine its own name, and the checkpoint name is the same for two engines of one
  checkpoint. The profile already keeps engine names apart.
- The engine kind is `mk-sdxl-bmg`, in a build with `xe`. Its
  config and `[engines.family]` take the key `model_dir`, and read `SDXL_MODEL_DIR` with no key, as `gemma4` does.
  `[engines.family]` also takes `clip_skip`, 2 or 3, with the default 2. Reason: these follow the rules of the other
  engine kinds of the B70 cards.
- The single file of 11.9. Reason: a Civitai checkpoint is
  one file with Stability's tensor names, and it holds no tokenizer.
  - The key `checkpoint` of `[engines.config]` names the file. The engine then reads its weights from the file only.
    `model_dir` still names a diffusers folder, for the CLIP tokenizer in `tokenizer/`, which every SDXL checkpoint
    shares. The family reads the tokenizer from it, as before.
  - The map from Stability's names to the names of diffusers is the map of diffusers 0.42.1
    (`loaders/single_file_utils.py`). It splits the fused query, key and value of the second text encoder, and it
    transposes its text projection. It drops `position_ids` and `logit_scale`, which no op reads.
  - A tensor that the map does not know, or a tensor that the engine needs and the file lacks, refuses the plan and
    the open. A file with F16 weights of the VAE gives F32 at the load, because the VAE runs in F32. F16 to F32 is
    exact.
  - The identity names the file: `model` is its name with no `.safetensors`, and `checkpoint_digest` covers it.

**The family of SDXL.**
The crate `nagaya-text-sdxl` holds it (11.3).

- The CLIP tokenizer of the checkpoint, from `tokenizer/vocab.json` and `tokenizer/merges.txt`. Both encoders take its
  ids. The first encoder pads with the end id 49407, and the second pads with 0, as A1111 does. Reason: in diffusers,
  `tokenizer_2` has the pad token `!`, so it reads each `!` of a prompt as id
  0. Thus `!!!` would be 1 token for one encoder and 3 for the other, and the chunks would not align. The oracle
  prompts have no `!` (`tests/fixtures/ids.json` of `nagaya-text-sdxl` shows the difference).
- The A1111 prompt: `(text)` multiplies the weight of its tokens by 1.1, `[text]` divides it by 1.1, and `(text:w)`
  sets it to `w`. `\(` and `\)` are plain brackets. `BREAK` starts a new chunk.
- A chunk is 75 tokens with a start token and an end token. A longer prompt makes up to 3 chunks, and the encoder
  output of the chunks joins along the tokens, as A1111 does. A longer prompt is a 400 that names `messages`.
- The family gives the engine the ids and the weight of each token. The engine scales the encoder output by the
  weights, and keeps its mean, as A1111 does.

**The design of later image work.** img2img and inpainting are built from it (11.9b), and LoRAs are built. The hires
pass is not.

- **img2img.** The last `user` message also holds an `image/png` object part. The engine encodes it with the VAE
  into the start latent, and `output.options.strength` sets how many steps run. This needs a VAE encode quantum.
- **Inpainting.** A second object part is the mask. The engine keeps the latent outside the mask at each step.
- **LoRAs.** Built, as `output.options.loras` (11.9a). The engine adds each delta to the output of its module, and the
  base weights stay as they are.
- **Hires pass.** `output.options.hires` names a scale and a step count. After the steps, the engine scales the
  latent up and runs img2img steps on it. Each is one more quantum of kind `step`.
- The engine calls (engine-abi "Image") take these inputs as optional members. Thus each one adds members and no call.

**11.9b Image inputs.** img2img and inpainting follow "The design of later image work" (11.9a).

- The image and the mask are object parts of the request.
- The blend of the latent works with any checkpoint. An inpainting UNet with 9 input channels, such as diffusers'
  `stable-diffusion-xl-1.0-inpainting-0.1`, loads as other weights.
- With no size in the request, the input image sets it. An image that breaks the size rules is a 400 that names the
  part. A request size that does not match the image is a 400 too.

The notes below fill the choices that these rules leave open. Each one follows diffusers 0.42.1, the version of the
oracle, unless it says otherwise. `img2img.py` is its `pipeline_stable_diffusion_xl_img2img.py`, and `inpaint.py` is
its `pipeline_stable_diffusion_xl_inpaint.py`. `image_processor.py`, `vae.py` and the schedulers are files of the same
package.

**The request.** Reason: the family, the route and
the oracle need the names, the ranges and the errors before they can test them.

- The object parts of the last `user` message are the inputs. The first is the image, and the second is the mask.
  Each has the `mime` `image/png`. A third object part is a 400 that names it, such as `messages[0].content[3]`.
- The text parts of the last `user` message, joined in order, are the prompt.
- The other messages are checked by the rules of gogi's council API ("A server checks each object part"). The server
  does not read their objects, as it does not render their text.
- A job with an image and no mask is img2img. A job with an image and a mask is inpainting.
- The served image model lists `media.inputs` as `["image/png"]`. This replaces "no inputs" of "The served image
  model" (11.9a). Reason: gogi refuses an object part of a media type that `media.inputs` does not list.
- `usage.media.read` counts the image and the mask, as for any object that a request reads.

`output.options` takes two more members. The rules of 11.9a for a member out of range stay.

| Member | Range | Default |
|---|---|---|
| `strength` | above 0.0, at most 1.0 | 0.75 for img2img; 0.99 for inpainting |
| `mask_blur` | an integer from 0 to 64, in pixels | 4 |

- `strength` without an image is a 400 that names `output.options.strength`. `mask_blur` without a mask is a 400 that
  names `output.options.mask_blur`. Reason: an option with no effect hides a fault of the client.
- 0.75 and 4 are A1111's defaults for img2img. 0.99 is below 1.0, as diffusers' default of 0.9999 for inpainting is
  (`inpaint.py:1092`). Thus the start latent of an inpainting job keeps a trace of the image.
- `count`, `seed`, `loras`, `cfg`, `sampler` and `clip_skip` work with an image as without one. Image `i` still uses
  the seed plus `i`.

**The usage of a job with an image.** Reason: "The request" says that the server does not read the objects of the other
messages. gogi's council API says that `read` counts each distinct object that the request names (`usage.media`). The
two rules differ for a context with earlier images, and 11.9a gives no rule for the steps of a job with an image.

- `usage.media.read` counts each distinct object that the request names, once, in any message. Thus it counts the
  image, the mask, and the objects of the other messages. Reason: a client of gogi's contract expects this count.
- `usage.media.written.steps` is `n`, the count of steps that ran, as `progress.of` is. Without an image, it stays the
  `steps` of the request.

**The size of a job with an image.** Reason: a resize would change the pixels that the client sent, and it would move
the edge of the mask.

- The image sets the width and the height. A request with no `width` and no `height` takes the size of the image.
- The image must keep the size rules of 11.9a: multiples of 64 from 512 to 2048, with at most 1,572,864 pixels. An
  image that breaks them is a 400 that names its part, and the message gives its size.
- A `width` or a `height` that differs from the image is a 400 that names that member, such as
  `output.options.width`.
- A mask of another size than the image is a 400 that names the part of the mask.
- The nagaya never resizes an input. diffusers resizes the image and the mask to the job size
  (`inpaint.py:1440-1447`), so the oracle gives inputs of the job size.

**The input PNG.** Reason: the route decodes each
input before the job starts, so a bad input is a 400, not a failed job.

- An input is a PNG with 8 bits in each channel: gray, gray with alpha, RGB, RGBA, or a palette. Other bytes, or 16
  bits in a channel, are a 400 that names the part. Reason: Pillow reads 16 bits in another mode, and the oracle has
  no such case.
- The nagaya ignores an alpha channel, as Pillow's `convert("RGB")` does (`image_processor.py:250`). diffusers takes
  an RGB image.
- A PNG with 1, 2 or 4 bits in a sample is a 400 that names the part, as 16 bits is. A `tRNS` chunk has no effect, as in
  Pillow's `convert("RGB")` and `convert("L")`. Reason: the refusal is the simplest choice. Its cost: Pillow writes a
  mask of mode `1` with 1 bit, and a palette of at most 16 colors with fewer than 8 bits. Pillow reads such a file as
  8-bit values, so a later rule can accept it.
- Each channel `v` of the image becomes `2 * v / 255 - 1` in f32 (`image_processor.py:167` and `:220`).

**The schedule of a job with an image.** Reason:
diffusers and A1111 both run the last part of the schedule, and the oracle compares each step.

- The sampler makes its schedule of `steps` timesteps as before. It runs only the last `n` of them, with
  `n = floor(steps * strength)` in f64 (`img2img.py:652-659`, `inpaint.py:913`).
- Python computes `int(num_inference_steps * strength)` in f64. Thus `steps` 90 and `strength` 0.7 run 62 steps, not
  63. The nagaya computes `n` in the same way, so it runs the same steps.
- An `n` of 0 is a 400 that names `output.options.strength`. diffusers refuses it too (`inpaint.py:1421-1424`).
- The history of a DPM++ sampler starts empty at the first step that runs. Each of the four samplers has `order` 1 in
  diffusers, and `set_begin_index` starts it there (`img2img.py:655-657`).
- The start latent of img2img is the image latent with the noise of the first timestep that runs. The scheduler's
  `add_noise` gives it, at a strength of 1.0 too (`img2img.py:764-768`).
- An inpainting job with a strength of exactly 1.0 starts from the noise times `init_noise_sigma`, as text to image
  does (`inpaint.py:813-815`). With less strength, it starts as img2img does.
- `add_noise` of Euler and Euler a is `x + sigma * noise` (`scheduling_euler_discrete.py:847`,
  `scheduling_euler_ancestral_discrete.py:524`). That of the DPM++ samplers is `alpha_t * x + sigma_t * noise`
  (`scheduling_dpmsolver_multistep.py:1349`).

**Steps and the A1111 chunk of a job with an image.**
Reason: gogi says that `step` rises by 1 up to `of` (council API, "Steps and previews"), and A1111 tools read the chunk
by its keys.

- `progress.of` is `n`, the count of steps that run. This changes "`of` is the step count of the request" of 11.9a
  for a job with an image.
- `Steps` in the A1111 chunk stays the `steps` of the request, as A1111 writes it.
- The chunk adds `Denoising strength: <strength>` after `Clip skip`. With a mask, it then adds `Mask blur: <n>`.

**The image latent.** Reason: diffusers draws the
latent from the distribution of the encoder, and A1111 does too. The mean alone would give other bits than the oracle.

- The VAE encoder gives a mean and a log variance for each latent value. The engine draws the latent
  `mean + exp(0.5 * clamp(logvar, -30, 20)) * z` (`vae.py:694-696` and `:703`).
- diffusers calls this `latent_dist.sample` (`img2img.py:115-119`). Then the engine multiplies the latent by the
  `scaling_factor` of the VAE, 0.13025 for SDXL base (`img2img.py:751`).
- `z` comes from the random stream of the engine, with the purpose `encode`. Image `i` of a request draws its own
  `z` from the seed plus `i`. Reason: the image of a seed must not depend on `count` (engine-abi "Image").
- The encoder runs once for a request, because each image has the same input. Only the draws differ.

**The mask.** Reason: these rules give the bits of
diffusers' `StableDiffusionXLInpaintPipeline` and the soft edge of A1111.

- The mask becomes gray, as Pillow's `convert("L")` makes it (`image_processor.py:267`). For RGB, the gray value is
  `(19595 R + 38470 G + 7471 B + 32768) >> 16`. The nagaya ignores an alpha channel of the mask.
- White repaints, and black keeps. diffusers sets a repainted pixel to 1 (`image_processor.py:536-537`), and it
  keeps the image where the mask is below 0.5 (`inpaint.py:1455`).
- The nagaya blurs the gray mask as Pillow's `ImageFilter.GaussianBlur(mask_blur)` does. diffusers' `blur` calls it
  (`image_processor.py:284`), and A1111 blurs its mask too. A `mask_blur` of 0 gives the mask with no change.
- The binary mask repaints a pixel whose blurred value is 128 or more. Reason: diffusers binarizes the mask at 0.5
  of 255 (`inpaint.py:322-324`, `image_processor.py:536-537`).
- The latent mask takes the binary mask at pixel `(8x, 8y)` for latent `(x, y)`. torch's `interpolate` in its mode
  `nearest` does this (`inpaint.py:862-864`).

**The blend.** Reason: this is the latent blend of
diffusers for a UNet with 4 input channels, so it works with any checkpoint.

- After each step, the latent becomes `(1 - m) * known + m * latent`, with `m` the latent mask
  (`inpaint.py:1648-1661`).
- `known` is the image latent with the noise of the next timestep, from the scheduler's `add_noise`. After the last
  step, `known` is the image latent with no noise.
- The noise of `known` is the start noise of that image, not a new draw (`inpaint.py:811` and `:1657`).
- A preview shows the latent after the blend.

**The paste after the decode.** Reason: the VAE changes the kept pixels a little on its way through. A1111 pastes the
kept pixels back by default.

- After the decode, the nagaya pastes the image back where the mask keeps it. It uses the blurred mask `a`, from 0 to
  255, so the edge is soft.
- Each channel of each pixel becomes `(a * new + (255 - a) * old + 127) / 255` in integers. `new` is the decoded
  pixel, and `old` is the pixel of the input image.
- The paste runs on the worker of the route, before the PNG encode. The engine gives its decode, as before.
- diffusers pastes only with `padding_mask_crop` (`inpaint.py:1727-1728`), and its `apply_overlay` rounds in another
  way (`image_processor.py:817`). Thus the oracle compares the decode before the paste. A host test checks the paste.
- img2img has no paste. A preview has no paste.

**The VAE encode quantum.** Reason: engine-abi
"Image" makes the encode one more quantum of kind `step`. "The design of later image work" adds members and no call.

- A job with an image runs its quanta in this order. First is the `embed` of the prompts, then one `step` for the VAE
  encode. Then come the `n` steps and one `step` for the decode.
- The encode quantum encodes each input that the job needs: the image, and the masked image of a 9-channel UNet. The
  images of a request share it, as they share each quantum (5.2).
- The encode is not a step of `progress`, as the decode is not.
- The engine declares the most pixels of a slot as before. That bound covers the encode quantum too.

**The 9-channel UNet.** Reason: this is the input of
diffusers' inpainting UNet, and the profile then needs no new key.

- The weights are diffusers' `stable-diffusion-xl-1.0-inpainting-0.1`, snapshot `115134f`. Its
  `unet/diffusion_pytorch_model.fp16.safetensors` has `conv_in.weight` of shape `[320, 9, 3, 3]` in F16. Its
  `add_embedding.linear_1.weight` is `[1280, 2816]`, as in SDXL base, so it takes the time ids of base.
- The engine does not trust the snapshot's `config.json` for this, because a download can lack it. Its VAE is
  `madebyollin/sdxl-vae-fp16-fix`. The engine reads the input channels from `conv_in.weight`: 4 or 9. Another count
  refuses the plan and the open.
- The input of each step has 9 channels in this order: the scaled latent, the latent mask, and the masked image latent
  (`inpaint.py:1612-1615`). Both halves of CFG take the same mask and masked latent (`inpaint.py:877` and `:900-902`).
- The masked image is the image with 0 at each repainted pixel, after the map to -1 to 1 (`inpaint.py:1455`). The
  engine encodes it as the image latent, with its own draw, the purpose `encode masked` (`inpaint.py:887`).
- A 9-channel UNet does no blend (`inpaint.py:1648`). The paste after the decode applies to it, as to any inpainting.
- A 9-channel UNet needs an image and a mask. A request without both is a 400 that names the content of the last
  `user` message, such as `messages[0].content`. Reason: the UNet has no input for a job without them.

The profile names the weights as any SDXL checkpoint. A single file with Stability's names and 9 input channels loads
the same way.

- The served model is `inpaint-0`, an engine of kind `mk-sdxl-bmg`. The key `model_dir` of `[engines.config]` names
  the inpainting folder.
- Its family table has no `model_dir`, so the family reads the tokenizer from `SDXL_MODEL_DIR` (11.9a).
- LoRAs work on it, as on any served image model. A LoRA module whose shape does not match is a 400 that names the key
  (11.9a).

**The errors of a job with an image.** Each one is a
400 that names the part or the member, and its message says what the client can change.

- A third object part in the last `user` message names that part.
- Bytes that are not a PNG, or a PNG with 16 bits in a channel, name the part.
- An image that breaks the size rules names its part, and gives its size and the rules.
- A mask of another size than the image names the part of the mask, and gives both sizes.
- A `width` or a `height` that differs from the image names that member, and gives the size of the image.
- `strength` without an image, `mask_blur` without a mask, or a `strength` that runs no step names that member.
- A 9-channel UNet without an image and a mask names the content of the last `user` message.

**11.9c Qwen-Image 2.1.**

- The model is Qwen-Image 2.1 (`Qwen/Qwen-Image-2.1`) with the FUSAL Anime LoRA, text to image only. Other
  variants can come later, so the design keeps room for them.
- The engine runs on the gfx1151.
- The text encoder is the Heretic Qwen3-VL-8B of FUSAL, in bf16. FUSAL trained its LoRA against
  the output of this encoder.
- The store keeps the encoder output of each prompt. 6.16 gives the rules.
- hipBLASLt does the large bf16 matrix multiplies. A change of hipBLASLt upstream can change the behavior a little,
  and that is accepted. 12.1, "The plan table of a HIP engine", gives the rules.
- Qwen3.8 and Qwen-Image are both loaded while the Strix Halo machine only serves. Before a large build, a client
  unloads Qwen-Image through the load routes (3.3b).
- The base model and the LoRA are under the Qwen Research License: non-commercial research and evaluation only. The
  Heretic encoder keeps its Apache 2.0 license (FUSAL `text_encoders/ORIGIN.md`).

The notes below fill the choices that these rules leave open. Each one follows diffusers 0.41.0.dev0 in the venv of
the diffusers reference, unless it says otherwise. `pipeline.py` is its `pipelines/qwenimage21/pipeline_qwenimage21.py`,
and `transformer.py` is its `models/transformers/transformer_qwenimage21.py`. `vae.py` is its
`models/autoencoders/autoencoder_kl_qwenimage21.py`, and `scheduler.py` is its
`schedulers/scheduling_flow_match_euler_discrete.py`.

**The engine and its weights.** Reason: a variant is other weights (above). Thus the profile names each file, and
the code names none.

- The engine kind is `mk-qwenimage21-gfx1151`, in a build with `hip`. Its crate is `nagaya-mk-qwenimage21-gfx1151`.
- The key `model_dir` of `[engines.config]` names the base folder of diffusers. With no key, the engine reads
  `QWENIMAGE21_MODEL_DIR`, as `mk-sdxl-bmg` reads `SDXL_MODEL_DIR`.
- The key `encoder` names one safetensors file of the text encoder. With no key, the engine reads the base folder's
  `text_encoder/`. FUSAL's profile names `text_encoders/qwen3-vl-8b-heretic-1.3.0_bf16.safetensors`.
- The key `loras` is a list of `{file, strength}`. FUSAL's profile has one entry: its rank 56 file with strength 1.0.
  An empty list gives the base model.
- The served model name is the name of the engine in the profile, as for SDXL (11.9a). FUSAL's engine is
  `qwen-image-fusal`.
- The checkpoint digest covers the transformer, the VAE decoder, the encoder file, and each LoRA file with its
  strength. Thus each variant has its own identity.

**The load.** Reason: text to image reads only some of the files. A strict load catches a wrong file before a job runs,
as the diffusers reference does.

- The engine loads the text tower of the encoder: `embed_tokens` and the 36 layers. It does not load the vision tower
  (`visual`), `lm_head`, or the final norm. Thus the encoder holds about 14.10 GiB of the 16.33 GiB file.
- The engine skips the final RMSNorm of the text tower. The conditioning is the input of that norm, not its output
  (`pipeline.py:303-316`). Reason: the transformer trained on the input. transformers 5.x returns the output of the
  norm in `hidden_states[-1]`, so the pipeline neutralizes the norm with a hook.
- The Heretic file has the names of ComfyUI: `model.layers.*`, `model.embed_tokens.*` and `model.norm.*`. The base
  folder has the names of transformers, with `model.language_model.` in their place. The engine reads both forms.
- A tensor that the engine needs and the file lacks refuses the plan and the open. So does a tensor that is in no
  list of the engine, whether it loads it or skips it.
- The engine loads the VAE decoder and `post_quant_conv` only. The VAE encoder comes with img2img (11.9b).
- The VAE file holds F32. The engine rounds it to bf16 at the open, because the diffusers reference runs the VAE
  in bf16.
- The plan counts the weights from the file headers, with no load: the transformer is 13.25 GiB and the text tower
  14.10 GiB. The VAE decoder adds less than 1 GiB in bf16. The scratch of a slot is a mandatory term (engine-abi
  "Core").
- The scratch of the largest slot is 7 GiB. `examples/measure.rs` measured it on the Strix Halo machine
  (`crates/nagaya-mk-qwenimage21-gfx1151/src/plan.rs:136-140`). `docs/deploy.md`, "Two engines on one GPU", gives the
  pool.

**The LoRA fold.** Reason: a LoRA of the profile is part of the weights of the variant. A fold costs nothing at each
step and needs no more memory than the base.

- At the open, each module `W` of a LoRA becomes `W + strength * scale * B A`, with one rounding to bf16. The engine
  computes the sum in f32. `A` is `lora_A` (rank by input), and `B` is `lora_B` (output by rank).
- `scale` is `alpha / rank` when the file has `alpha`, and 1 when it has none. FUSAL has no `alpha` tensor. Its
  metadata gives `ss_network_alpha` 56 and rank 56, so its scale is 1 either way.
- A key is `transformer.transformer_blocks.<i>.<module>.lora_A.weight` or `.lora_B.weight`. The modules are `attn.to_q`,
  `attn.to_k`, `attn.to_v`, `attn.to_out.0`, `img_mlp.proj`, `img_mlp.gate_layer` and `img_mlp.out`.
- A key that names no module of the transformer, or a shape that does not match, refuses the plan and the open.
- diffusers adds `B(A(x))` to the output at each step and does not fold. The fold gives other bits at the size of one
  bf16 rounding of each weight. The oracle tolerance holds this (8.3).
- The LoRAs of a request (11.9a) stay a design for this engine, and the engine refuses them now.

**The family of Qwen-Image 2.1.** The crate `nagaya-text-qwenimage21` holds
it (11.3). Reason: these rules give the ids of the processor that FUSAL trained with.

- The tokenizer is the base folder's `processor/tokenizer.json`, not a file of the Heretic repository (FUSAL README).
  Its `tokenizer_hash` is `sha256:` and the hex digits of the SHA-256 of that file.
- The prompt is the text of the last `user` message, as in 11.9a. An empty prompt becomes one space, because Qwen has
  no start token (`pipeline.py:249`).
- The family puts the prompt into this template. `\n` is a newline, and `{}` is the prompt (`pipeline.py:217-221`):
  `<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n`
  then `<|im_start|>assistant\n`.
- The family tokenizes the whole template with no added start token. The text of each special token in the template
  gives its special id.
- A prompt that holds the text of a special token is a 400 that names `messages`. Reason: diffusers would read it as a
  special id. Then a prompt could change the frame of the template.
- The cut is 14 ids: the system message, `<|im_start|>system\n...<|im_end|>\n` (`pipeline.py:228-231`). The family
  computes the cut from the tokenizer, and a test checks that it is 14.
- The template has at most 1024 ids. A longer prompt is a 400 that names `messages`. Reason: the bound fixes the
  prefix rows of a slot (engine-abi "Image").
- The family does not parse A1111 weights, such as `(word:1.2)`. Brackets are plain text. Reason: this encoder has no
  weight for each token, and diffusers gives it the text as it is.
- The family does not add the trigger phrase `fusal style.` The client writes it, as the diffusers reference and the
  FUSAL README do. Reason: the family never changes a prompt, and the A1111 chunk then gives the request's prompt.
- The identity's `template` is the template text above. Thus a new template is a new recipe (12.1).

**The served Qwen-Image model.** Reason: these rules follow the served
image model of 11.9a, and they refuse each member that this engine cannot run.

- Identity lists it as 11.9a lists a served image model, with no inputs. Its `context_tokens` is 1024.
- `output.options` takes these members. A member out of range is a 400 that names it, as in 11.9a.

| Member | Range | Default |
|---|---|---|
| `negative` | `""` only | `""` |
| `width`, `height` | multiples of 32 from 512 to 2048; at most 2,097,152 pixels together | 1024 |
| `steps` | 1 to 150 | 40 |
| `cfg` | 1.0 only | 1.0 |
| `sampler` | `euler` only | `euler` |
| `count` | 1 to 4 | 1 |

- `clip_skip`, `spacing`, `loras`, `strength` and `mask_blur` are a 400 that names the member. An object part in a
  message is a 400 that names it, because the model lists no inputs.
- `previews: true` gives the `progress` events and no `preview`. The latent has 64 channels, and no map to RGB is fitted
  yet. Reason: gogi's contract lets a served model send no
  preview, and that is not an error ("Steps and previews").
- The sizes are multiples of 32. The VAE scales by 16, and the transformer reads the latent in groups of 2 by 2
  (`pipeline.py:396`, `:752-754`). 2,097,152 pixels is 8192 latent tokens, the `max_image_seq_len` of the scheduler.
- The default of 40 steps is the default of `pipeline.py:519`. The FUSAL README asks for 30 to 40.
- `cfg` above 1.0 and a `negative` prompt are refused now. FUSAL and the pipeline both use 1.0 (`pipeline.py:550`).
  True CFG is a design for a later variant, and nothing builds it now. It runs a
  second pass of the transformer for each step, with the negative prompt as its condition.
- The PNG is RGBA. The nagaya does not flatten it to RGB.
- The output is `image/png` in RGBA, because the VAE gives 4 channels (`vae.py` config `out_channels` 4). The
  diffusers reference writes RGBA PNGs too.
- The A1111 chunk gives `Steps`, `Sampler: Euler`, `CFG scale: 1`, `Seed`, `Size` and `Model`, with no `Clip skip`.

**The schedule.** Reason: these rules give the sigmas of `pipeline.py:733-746` and `scheduler.py`, bit for bit. The
engine computes them on the host, as for SDXL (11.9).

- `n` is the step count. The sigmas start as `linspace(1, 1/n, n)` in f64, then round to f32 (`pipeline.py:735`).
- `seq` is the latent token count, `(height / 16) * (width / 16)`. With `m = (0.9 - 0.5) / (8192 - 256)` and
  `b = 0.5 - 256 m`, `mu = seq m + b` in f64. The scheduler config gives 256, 8192, 0.5 and 0.9.
- Each sigma becomes `e^mu / (e^mu + (1 / sigma - 1))` in f32 (`scheduler.py:655-656`). numpy 2 rounds `e^mu`
  to f32 before each operation.
- Then the sigmas stretch to the terminal 0.02 in f32: with `z = 1 - sigma`, each sigma becomes
  `1 - z / (z_last / 0.98)` (`scheduler.py:262-281`). A last sigma of 0 follows.
- The timestep of step `k` is `1000 * sigma_k` in f32. The transformer reads it rounded to bf16, then divided by
  1000 in bf16. It then multiplies by 1000 in f32 (`pipeline.py:782`, `:786`, and `transformer.py:149`).
- A step makes `x + (sigma_k+1 - sigma_k) v` in f32 from the bf16 latent `x` and the bf16 output `v`. It rounds the
  result to bf16 (`scheduler.py:493-526`).

**The steps.** Reason: these rules give the math of `transformer.py` for text to image. The cache of the condition keys
and values is exact, so it is a rule and not an approximation.

- The start latent of image `i` comes from the random stream of the SDXL engine (`noise.rs`), with the purpose of the
  start latent. It fills the shape `[64, height / 16, width / 16]` in order, and each value rounds to bf16.
- The sequence of a step is the conditioning rows, then the latent tokens in row order (`transformer.py:931-939`).
- A conditioning row attends to each conditioning row up to itself. A latent token attends to every row
  (`transformer.py:295`). Thus no conditioning row reads the latent.
- The conditioning rows take their modulation from the timestep 0 (`transformer.py:238-252`, `:948`). Thus their
  keys and values are the same at each step. `image_begin` computes them once, and the slot keeps them for each step
  (`pipeline.py:759-762`).
- The rotary positions have three axes of 16, 56 and 56 dimensions, with theta 10000 (`transformer.py:655-713`).
  Conditioning row `j` has position `j` on each axis.
- Latent token `(y, x)` has the position `L` on the first axis, with `L` the conditioning row count. It has
  `y - (h - h / 2)` and `x - (w - w / 2)` on the other two, with `h` and `w` the latent height and width, and integer
  division (`transformer.py:695-696`). A size is a multiple of 32, so a latent side is even, and this is `y - h / 2`.
  The oracle size of 256 also gives an even side.
- Each step gives the output of the latent tokens only (`pipeline.py:796`).

**The attention.** Reason: a balance of accuracy and speed. The engine runs attention in a fused flash kernel
(`nagaya-mk-hip`, `hip/flash.hip`). The scores stay in registers, with a running max and sum for each row, and the
probabilities round to bf16 before the values.

- The other path runs the scores as a GEMM in f32, a softmax kernel, and a GEMM with the values. The profile key
  `attention = "gemm"` selects it. Both paths pass every gate of 8.3.
- On the Strix Halo machine, a step of 1024 by 1024 takes 5.8 s on the GEMM path and 4.2 s with flash. A step of
  1440 by 1440 takes 18.3 s and 11.0 s. The GEMM path moves about 200 GB of scores through memory for each step.
- The gaps are nearly the same. The step gate gives 2.24e-5 and 2.51e-5 at step 0. The run gives 5.14e-5 and
  5.58e-5. The image gate gives mean byte gaps of 0.721 and 0.680.
- The flash kernel takes heads of 128 values. The single head of the VAE's attention, of 1152 values, keeps the GEMM
  path.

**The decode.** Reason: these rules give the bits of the pipeline's decode and of diffusers' `postprocess`, which round
in bf16.

- The latent becomes `x * std + mean` in bf16, with the 64 values of `latents_std` and `latents_mean` of the VAE
  config (`pipeline.py:842-852`).
- The VAE decoder runs in bf16. Its norms and its up samplers compute in f32 and round to bf16, as `vae.py:187-235`
  does. The output clamps to -1 to 1 (`vae.py:1328`).
- Each channel `x` becomes `u = bf16(x * 0.5 + 0.5)`, clamped to 0 to 1 (`image_processor.py:223-235`). The byte is
  `u * 255` in f32, rounded half to even, as numpy's `round` does (`image_processor.py:143`).
- The engine gives 8-bit RGBA. The nagaya makes the PNG and its chunk, as for SDXL.

**The quanta.** Reason: these follow the quanta of 11.9, with the conditioning cache of 6.16.

- `image_begin` is one quantum of kind `embed`. On a miss, it runs the encoder over the template ids. Then it computes
  the keys and values of the conditioning rows in each of the 32 blocks. On a hit, it does only the second part.
- Each `image_step` is one quantum of kind `step`. The decode is one quantum of kind `step`. The images of a request
  share each quantum (5.2).
- A step of 1024 by 1024 runs 4096 latent tokens through 7.1 billion parameters. `max_quantum_ms` is 60000 until a
  measurement sets it, as for SDXL (`crates/nagaya-mk-qwenimage21-gfx1151/src/plan.rs:22-23`).
- Thus a step of Qwen-Image can hold the device for seconds. A Qwen3.8 job then waits up to one quantum (D8).
- A step never splits into quanta of some blocks each. A chat job on the same device can wait up to one step while an
  image renders.

**11.10 Embeddings.** gogi's contract is "Embeddings" in gogi `docs/council-api.md`.

- The first embedding model is EmbeddingGemma 2 (`google/embeddinggemma-2`), for semantic search.
- The route is `POST /council/v2/embeddings`. It is native to the council API, not an extension (11.4a).
- The request names its `task`. The family renders the prompt of the task, and a client never writes it.
- `dimensions` asks for a shorter vector: 768, 512, 256 or 128. The nagaya cuts the vector and normalizes it again.
- The inputs are text now. The model also reads audio and images, and those come later.
- The engine runs on a B70 card on the shared interpreter. It is resident beside Gemma 4.

The route, the served embedding model, its job and the cut on the host are built.

**The served embedding model.**
Reason: these rules follow the served image model (11.9a) and gogi's rules for a model that only embeds.

- The name of the served model is the name of its engine in the profile, such as `embeddinggemma-2`. Reason: the
  rule of the served image model.
- Identity lists it with the capabilities `embed` and `dry_run`, and with no `generate`. Its `media` has no inputs,
  no outputs and `text_replies: false`. A build, a decision, a raw read or a generation that names it is a 400 that
  names `model` (gogi).
- Its `embeddings` member is `{"tasks": [...], "dimensions": [768, 512, 256, 128], "inputs_per_request": 64}`. The
  tasks are those of the table below.
- Its `context_tokens` is 2048: the most tokens of one input, with the task prompt and the special tokens. Reason:
  the oracle checks inputs up to 2040 tokens (8.3), and a longer input has no reference. The cost of the full
  layers grows with the square of the length.
- Its decision limits and its `snapshot_bytes` are 0, because it holds no state.
- Its `inputs_per_request` is 64. Reason: gogi's example gives 64. Each input is one quantum, so a request of 64
  inputs of 2048 tokens holds its group for 64 quanta at most.
- Its `weight_hash` and `tokenizer_hash` are the engine's after its first open since the start, and empty before
  (3.3a "Council models on demand").
- Its `template` is `embeddinggemma2-tasks:<hash>`. The hash is the first 16 hex digits of the SHA-256 of the task
  text of the family. Reason: the vectors depend on the prompts, and gogi puts the prompts in `template`.
- The task text has one line for each task, in the order of the table: the task, a tab, the prompt, and a newline.
  Its last line is `frame`, a tab, `<bos>{prompt}{text}<eos>`, and a newline. Thus the id changes only when a prompt
  or the frame changes.
- The engine kind is `mk-embeddinggemma2-bmg`, in a build with `xe`. Its config and `[engines.family]` take the key
  `model_dir`, and read `EMBEDDINGGEMMA2_MODEL_DIR` with no key, as `mk-sdxl-bmg` does.

**The tasks.** Reason: gogi asks a
model for search to name its tasks `query` and `document`. The other six are the other prompts of the model, and the
oracle checks all eight. The model's `config_sentence_transformers.json` holds the prompts. Its other names give one of
these prompts too.

| Task | Name in the model's prompts | Prompt |
|---|---|---|
| `query` | `query` | `task: search result \| query: ` |
| `document` | `document` | `title: none \| text: ` |
| `classification` | `Classification` | `task: classification \| query: ` |
| `clustering` | `Clustering` | `task: clustering \| query: ` |
| `similarity` | `SentenceSimilarity` | `task: sentence similarity \| query: ` |
| `code_retrieval` | `CodeRetrieval` | `task: code retrieval \| query: ` |
| `question_answering` | `QuestionAnswering` | `task: question answering \| query: ` |
| `fact_checking` | `FactChecking` | `task: fact checking \| query: ` |

**The family of EmbeddingGemma 2.**
The crate `nagaya-text-embeddinggemma2` holds it (11.3). Reason: these rules give the ids of sentence-transformers
6.1.0, which the oracle recorded.

- The tokenizer is the checkpoint's `tokenizer.json`. Its `tokenizer_hash` is `sha256:` and the hex digits of the
  SHA-256 of the file, as for Gemma 4 (12.3).
- The prompts come from `config_sentence_transformers.json`. A task of the table whose prompt the file lacks is an
  error at the open.
- The ids of an input are `<bos>` (2), the ids of the prompt and the text as one string, and `<eos>` (1). The prompt
  ends with a space, and the tokenizer reads that space with the first word of the text.
- The text is content. A special token that the text spells stays text. Reason: gogi never lets content be control.
  sentence-transformers reads such text as the special token, so its ids differ for this text only.
- The token count of an input is the count of its ids. A dry run gives it.

**The job of an embedding request.**
Reason: the image job (11.9a) gives the shape, and an embedding holds no state.

1. The request thread decodes the body and checks it with gogi's rules: the model, the task, the dimensions and the
   inputs.
2. A served model that is not `ready` gives a 503 `model_unavailable` (3.3a).
3. The family gives the ids of each input. An input past `context_tokens` is a 413 `too_large`, as gogi says. A dry
   run skips this check.
4. A dry run answers with the token count of each input, and no job runs. `usage.fed` is 0.
5. Otherwise the route submits one job to the group of the engine, with the class and the priority of the request
   (5.3). The job has one quantum of kind `embed` for each input, in input order.
6. Each quantum calls the engine's `embed` with the ids of one input (engine-abi "Embedding").
7. The nagaya cuts each vector to `dimensions` and normalizes it. It sums the squares in f64 in index order, and
   divides each value by the root in f64. It rounds each result to f32.
8. The response lists the vectors in input order. A client that closes its connection cancels the job between
   quanta.

- The nagaya normalizes each vector, also at the full length. Reason: one rule in one place, on the host, with a
  test that needs no card.
- A cut vector of zeros has no direction, so it cannot be normalized. The request then fails with a 500 `internal` that
  names the input, and no vector goes out. Reason:
  the division gives NaN, and a 200 with NaN values hides a defect of the engine.
- A job charges no context and stores nothing. The plan of the engine holds its weights and the work of one input of
  `context_tokens`.

**The residency of the embedding engine.** Reason: an embedding engine is always loaded somewhere.

- The embedding engine is resident and pinned (3.3a, "Explicit layout"). No load removes it, and no idle time unloads
  it. A swap of Gemma 4 moves it for the time of the swap, so its requests can get a 503 for a few seconds.
- Memory: card 1 has 2,469,944,308 bytes beside gemma. Card 0 has 4,394,792,436 (the profile of the B70 machine,
  budgets). The text tower of EmbeddingGemma 2 is 542,005,296 bytes in bf16 (the header of `model.safetensors`).
- The placement is card 1. Reason: card 0 has four on-demand engines (`voice`, `sdxl-0`,
  `pony-0` and `lightning-0`), and card 1 has two.

**11.11 LFM3.**

- Two served models run on one engine kind, `mk-lfm3-bmg`: `LFM3-Omni-2B` and `LFM3-1.7B`. Each is other weights.
- Omni reads text, images and audio. It writes text, speech, or text and speech in turn. Video comes later.
- Both models are resident on card 0 of the B70 machine. Gemma 4 is on demand, and only the load routes load it (3.3b).
- Qwen3-TTS VoiceDesign stays the speech engine (11.8). Omni's speech is a second choice, with its own voices.
- The nagaya does not check a model license. The user accepts it at the download.

The checkpoints are LiquidAI `LFM3-Omni-2B` at `21d75123f86c1f7ffb4bef1ea202710c4e7ebe67` and `LFM3-1.7B` at
`0dc4268b561499c648c7ba7c9466bf3d8d01517a`.

**The served LFM3 models.** Reason: these rules follow the served text model
(11.1) and the speech model (11.8a), and they add no member to gogi.

- The served names are `lfm3-omni` and `lfm3`. The engine names in the profile are the same.
- `lfm3` lists no `media`. Its replies are text, and it serves builds, decisions, raw reads and generations.
- `lfm3-omni` lists `media.inputs` of `image/png`, `image/jpeg` and `audio/wav`, and `media.outputs` of `audio/wav`.
  Its replies hold text, so it serves builds, decisions and raw reads too.
- Each served model gets its own `tokenizer_hash` and `template`, because the two checkpoints ship different files.
- A slot holds 32,768 positions by default, the context of Omni's model card. A profile can list the positions of
  each slot, up to 262,144 (engine-abi "Slot sizes"). `context_tokens` is the longest slot.

**The served name of an LFM3 engine.** Reason: a profile can have two engines of `LFM3-1.7B`, such as `lfm3` and
`lfm3-1`. The rule of 11.1 gives both the served name `lfm3`, because both identities have the `model` `lfm3`. A
served name is unique (2.3), so the start exits with 3.

- The served model of an engine of the kind `mk-lfm3-bmg` is its engine name in the profile. This replaces the rule
  of 11.1 for this kind only.
- The `model` of the engine identity stays `lfm3` or `lfm3-omni`. The family checks its folder against it.
- The kind gives its served name before the open (`Factory::served_names`, 2.3, "The check of a name").
- The engines `lfm3` and `lfm3-omni` keep their served names, because their engine names are the same.
- Other text kinds keep the rule of 11.1. Gemma 4 and the Qwen3.8 engine serve the `model` of their identity.
- A served name in the config of the engine is not used. It adds a key, and the engine name is already unique.

**The LFM3 family.** The crate `nagaya-text-lfm3` holds the family library
of both models (11.3).

- The tokenizer is `tokenizer.json` of the model folder, and the template is `chat_template.jinja` of the same
  folder. The library reads both from the folder that the profile names, so one library serves both models.
- Tool calls are the template's XML form: `<tool_call>`, `<function=NAME>`, `<parameter=NAME>`. The reply parser
  reads the same form.
- Thinking follows the context (11.3a). `lfm3` renders with `enable_thinking` from the context. Omni is a model with
  no reasoning, so a context of `lfm3-omni` with `thinking: true` is a 400 that names `thinking`.
- An image part renders as `<|vision_start|><|image|><|vision_end|>`, and an audio part as
  `<|vision_start|><|audio|><|vision_end|>`, as the template writes them. The family then puts one placeholder
  position for each row of the encoder in place of `<|image|>` or `<|audio|>`.
- An audio part of an `assistant` message renders as `<|audio_out|>`, one position for each frame of the output and
  one for the end frame. The reference appends the end frame to a clip in a prompt (`processing_lfm3_omni.py:408`).
- An image takes `grid_t × grid_h × grid_w / 4` positions, from 64 to 2,048 (`processing_lfm3_omni.py:755`). A clip
  takes `ceil(frames / 8)` positions, 12.5 each second (`modeling_lfm3_omni.py:408`).

**The choices of the LFM3 family.** Reason: the notes
above leave these choices open, and each one is the simplest choice that the reference can check.

- **Interpreter.** The interpreter holds the block `{% generation %}` of transformers and the method `items` of a
  dict, with the results of jinja2. Both LFM3 templates use them. The other methods of a dict stay refused.
- **Kept thinking.** The family renders with `preserve_thinking` on. Thus an assistant message keeps its reasoning
  after a later user message, and a held prefix stays a prefix. The Gemma 4 library does the same.
- **The thinking of Omni.** `lfm3-omni` does not reason, so its family does not think, and its served model does not
  declare `thinking`. A context with `thinking: true` is a 400 that names `thinking`, as the first rule said. The
  reply parser of Omni starts in the content. `lfm3` declares `thinking`. gogi requires
  `thinking`, and its suite sends `thinking: true` only to a model that declares it.
- **Escapes.** Each of the 140 added tokens of `tokenizer.json` is a marker. This includes the tokens that the file
  does not mark special, such as `<think>` and `<tool_call>`. Client text that spells a marker is content.
- **Reads.** A read closes the thinking with `<think></think>`, the prompt of the template for thinking off. A read
  with no context starts with `<|startoftext|>`. A queue read writes the call in the XML form of the template, as the
  Qwen3.8 library does.
- **Stop tokens.** The stop tokens are `eos_token_id` of `generation_config.json`. They are `<|im_end|>` for `lfm3`,
  and `<|im_end|>` and `<|text_end|>` for Omni.
- **Vocabulary.** `vocab_size` is the token count of `tokenizer.json`, 125,033. The embedding table has 128,000 rows.
  The engine gives logit rows of 125,033 values. Reason: the sampler then never picks a padding row, which has no
  text.
- **The reply parser.** Its reference is `parse_response` of transformers 5.20.2, with the `response_template` of
  `LFM3-1.7B` and the tools of the request. Omni's files have no `response_template`, and its template writes the
  same form. The parser takes the types of each parameter from the tool schema, as the reference does.
- **Differences of the parser.** Each one has a test.
  - A call that does not close is malformed. The reference reads it.
  - A call with no `<function=NAME>` is malformed. The reference keeps its text as content.
  - The content is all the text outside the thinking and the calls. The reference keeps only the last run of content
    text.
  - An integer that does not fit in 64 bits makes the call malformed, as in the Qwen3.8 library.
  - `<think>` opens the reasoning only before any content or call. After them it is text of the content. The
    reference opens the reasoning at any place in the content.
  - The parser reads the markers as tokens. The reference reads the decoded text, so it also finds a marker that the
    model spells in pieces.
- **Media.** `Family::render` reads each media part through the object reader of the process. The header of the
  part gives its count of positions. `Lfm3Family::prepare` also gives the rows of each part for the engine. The
  family writes the placeholder token that many times between `<|vision_start|>` and `<|vision_end|>`, as Liquid's
  processor does. `lfm3` refuses each media part.
- **The marker of a mode.** `<|audio_start|>` or `<|mixed_start|>` follows the whole generation prompt, after
  `<think></think>`. transformers appends it there (`generate` in `modeling_lfm3_omni.py`).

**The media of a request.** Reason: the simplest choice that the oracle
can check, with no resampler and no image decoder of another library to match.

- The family decodes a PNG or a JPEG to RGB in 8 bits, and it resizes the image as the reference processor does.
  The processor config gives the bounds: from 65,536 to 2,097,152 pixels, patches of 16, and a merge of 2 by 2.
- A WAV of `audio/wav` is PCM in 16 bits, mono, at 16 kHz. Any other WAV is a 400 that names the part. The family
  computes the 128 mel bins of the reference's feature extractor.
- The resize, the mel bins and the rows that each media part takes have a family test against the reference.
- The encoder of each media part runs on the card, in the quantum of the feed that reads it.

**The choices of the media preprocessing.** Reason:
the note above leaves these choices open, and each one is the simplest choice that the reference can check.

- **Objects.** The process gives the family a reader of the objects of this machine. The family reads the bytes of
  each media part through it. A family with no reader refuses a media part with a 400 that names the part.
- **Depth.** A PNG with 16 bits in a sample keeps the high byte of each sample. Reason: common tools write such a
  PNG, and the owner accepts a conversion on the CPU (2026-11-17). Pillow clips each value of a gray of 16 bits,
  and the family does not copy that. The SDXL family refuses such a PNG (11.9b). A PNG of 1, 2 or 4 bits expands
  to 8 bits.
- **JPEG.** The decoder is the crate `zune-jpeg`, not libjpeg-turbo of Pillow. On the fixture, 305 of 9,216 bytes
  differ, each by 2 or less. Thus a JPEG has no gate of bits, and a PNG has one.
- **Orientation.** The family does not turn an image by its EXIF orientation. The reference processor does not
  turn it either.
- **Resize.** The resize is torch's integer arithmetic with weights in 16 bits. It gives the bits of the reference
  for each size of the fixture, from 37 by 53 to 3,000 by 4,000 pixels.
- **Mel rows.** The family computes the spectrum and the mel sums in f64, and it rounds once to f32. The reference
  computes in f32, so its bins of low power hold the noise of that rounding. The test bounds the gap at each level.
- **Assistant audio.** The rule of "The generation
  modes" below holds. An `assistant` audio part names an output of this served model, and its positions are the
  frames of the codes that the service keeps. Until the nagaya builds it, the part is a 400 that names it.

**Video, and audio at any rate.** These rules
replace the rule above that a WAV must be 16 kHz mono. A build with no feature `video` keeps the rule above.

- A `user` part can be a video object, such as `video/mp4` or `video/webm`. The family decodes it on the CPU with
  ffmpeg's libraries. Reason: the reference loads a video through ffmpeg, so its frames are the closest to the oracle.
- The decoder is behind a cargo feature. A build with no feature refuses a video part with a 400 that names the part,
  and `media.inputs` lists no video type.
- The recipe item of a video part names the versions of the ffmpeg libraries ("The video of a request", below).
  Reason: a new decoder can give other frames from the same bytes.
- The frames follow the video processor of the checkpoint: 2 frames each second, from 4 to 384 frames, and at most
  768 video tokens. Each frame goes through the vision tower.
- The sound track of a video becomes an audio part beside its frames, in the order of the reference processor.
- The resampler of the decoder gives 16 kHz mono to the sound track and to any `audio/wav`. A WAV of 16 kHz mono in
  16 bits skips it, so its rows do not change. The recipe item of a resampled WAV names the version of the
  resampler, as a video names its decoder.
- Live input through PipeWire, from a microphone or a camera, is a later design.

**The video of a request.** Reason: the note
above leaves these choices open. Each follows the checkpoint's own processors (`video_processing_lfm3_vl.py`,
`processing_lfm3_omni.py`), so that the oracle can check it.

- **A reading of the note above.** The processor caps a frame at 768 video tokens only with `cap_pixels_per_frame`,
  and the checkpoint sets it false. So the bound is a pixel budget: all the frames together hold from 4,096 to
  25,165,824 pixels.
- **The decoder version.** The media type in the recipe item of a video part (tag 9) names the versions of
  libavformat, libavcodec, libswscale and libswresample. Thus an update of ffmpeg changes only the recipes of video
  parts. Reason: the build facts are in the identity digest (12.1). With the versions in the identity, an update
  would make each held context fill again, text contexts too.
- **Sampling.** The frame count is the floor of the frames of the stream divided by its rate and multiplied by 2. It
  is then at least 4, at most 384, and at most the frames of the stream. The indices are evenly spaced from the first
  frame to the last, rounded half to even.
- **Pairs.** Two frames are one patch in time. An odd count repeats the last frame. A stream of fewer than 2 frames is
  a 400 that names the part.
- **The size.** Every frame of a video resizes to one size, by the rule of the image (above) with a factor of 32 and
  the pixel budget. The arithmetic of the resize is the arithmetic of an image.
- **The text.** The video is one pair of vision markers. Inside, each pair of frames writes its time, as
  `<t seconds>` with one decimal, then its video tokens in their own vision markers. The time is the mean of the
  indices of its two frames, divided by the stream's rate. The rate is rounded to 6 decimals first, as the
  processor rounds it.
- **The tower.** Each pair of frames goes through the vision tower as an image of one frame in time. Thus each pair
  is a span of the kind of an image, at the positions of its video tokens. The engine needs no new kind for frames.
- **The sound.** The sound track resamples to 16 kHz mono, and its mel rows go through the audio encoder once. Its
  encoder rows are split at the time of the first frame of each pair, as the reference splits them. Each part follows
  the video tokens of its pair. This needs a new span kind in the engine: a run of the rows of one clip.
- **The decode gate.** The frames come from ffmpeg in the family, and from torchcodec in the reference. Both use
  ffmpeg's decoder, its color conversion and its resampler. With ffmpeg 9.0.2, both give the same bits
  of the frames and of the sound. So the decode gate is a gate of bits with the same ffmpeg build.
- **Limits.** A pair of more than 2,048 positions is a 400 that names the part. So is a sound of more than 4,096 mel
  frames, about 41 s. These are the limits of the engine's tower and encoder. The family checks them before its work.
- **Types.** `media.inputs` lists `video/mp4` and `video/webm` in a build with the feature. A video with no video
  stream is a 400 that names the part. A video with no sound track gives frames only.

**The recipe of a media part.** Reason: the rows of a part are a function
of the engine identity and the bytes of the object. The object's content id names the bytes.

- A media part enters the recipe as tag 9 (6.2): its media type, its content id, and its count of positions.
- The placeholder tokens stay in the tokens of the recipe, so a boundary after a media part is a boundary as any.
- The codes of an `assistant` audio part enter the recipe as the speech positions do (11.8a). Each frame gives 12
  codes, and the end frame is one more frame.

**Media through the council.** Reason: the notes
above leave these choices open, and each one is the simplest choice that a fake engine can check.

- **The checks.** A text service checks each object part by the rules 1 to 4 of gogi "Contexts", in that order. Rule
  2 reads the media types of the family. Rules 3 and 4 read the object store of the process.
- **The rendering.** A rendering lists its media spans. A span has its first position, its count of positions, and
  the media type and the content id of its part. The rendering of a text family lists none.
- **The rows.** The family gives the rows and the `meta` record of each span. The service gets them before the job
  starts, for each span of the context. Reason: a decode on the group thread would hold the engine. The cost is a
  decode of each part, also of a part in a held prefix.
- **The passes.** A pass of a build whose positions hold a part of a span gets `TokensAndRows`. Its `meta` holds the
  record of each such span, and its rows are the rows of each span whole (engine-abi "The LFM3 vision binding"). A
  pass with no span gets `Tokens`, as before.
- **The item.** The item of tag 9 starts with the media type, as a `u64` length and its bytes. The 32 bytes of the
  content id follow, then the count of positions as a `u64`.
- **Its place.** The recipe of the tokens from position a to position b holds one item for each span whose first
  position is in that range. The items follow tag 7, in the order of the positions. A recipe with no span is the
  same as before, so a text model keeps its recipes.
- **Inside a span.** A boundary never falls inside a span, because a part is inside a message. An interval copy
  (6.13) inside a span names the span, so two images of one size never give one recipe.

**Media in a decision.**
Reason: gogi reads `case_media` "with the case in order", and a decision context is a context. Each choice below is
the simplest one that the suite and a fake can check.

- **The case.** In a plain read, the case turn is one user message. It holds each object of `case_media` in its
  order, as object parts, then the label and the case as one text part ("The plain read of the LFM3 family").
- **The context.** A decision context may hold media parts. The build reads them as a build of that context does.
- **The rows.** The family prepares the rows of each span of the read's text, and the job feeds them by position, as
  a build does. A count of positions comes from the header of the object, as in a rendering.
- **The hash.** `rendered_hash` covers the text of the read. With media, it covers the text, then the id of each
  object of the read in order. A read with no media keeps the hash of before.
- **A queue read.** Its case is the result of a call, so it holds no media. A `case_media` of a queue read, or a media
  part in its context, is a 400 that names it.
- A family with no media refuses `case_media` with a 400 that names `case_media[0]`, as the check of objects does.

**The plain read of the LFM3 family.** **Proposed** (Claude Fable 5.2).
Reason: a trial of 2026-11-17 on both LFM3 models. The layout of the first Gemma 4 engine puts every question in one
spec turn. With it, a spec of 4 questions gave 236 right answers of 324 on 9 images, and 229 of 324 on 9 text cases.
One question in each read gave 324 of 324 and 321 of 324.

The owner chose the shape: the case first, one question in each read, and a held head after the case. The words of
each turn are the proposal.

- **The case turn.** The read puts one user message after the context. It holds each object of `case_media` in its
  order, then one text part: the label, `:`, a line feed and the case.
- **The head of the case.** The end of the case turn is a boundary. The job builds through it, as a build does, so
  the store holds a head there. A later decision with the same context and the same case starts from that head. This
  is true for each spec and each question, because the case turn holds no text of the spec.
- **The question turn.** Each asked question is a user message of its own after the case turn. It holds the
  instructions of the spec and a blank line, when the spec has instructions. Then it holds this line, a blank line,
  and the question with its lettered options:
  `Answer this question about the <label> with the letter of one option, as "Answer: X".`
- **The read.** The question turn ends with the prompt of the template for thinking off, then `Answer:`. The letter
  token has a leading space.
- **Each question alone.** Each question starts from the head of the case. No question sees another question or its
  answer. A narrowed question keeps its letters, as before.
- **The hash.** `rendered_hash` covers the text through the case turn, then the text of each question turn in the
  order of the asked questions. With media, the id of each object of the read follows, as before.
- **The room.** The job asks for a slot that holds the context, the case turn and the longest question turn.
- **A queue read** does not change.
- The Qwen3.8 library and the Gemma 4 library keep the old layout. The trial did not run on them.

**Text questions of the LFM3 family.** **Proposed** (Claude Fable 5.2).
Reason: the owner asked for the capability `describe` on 2026-11-17. gogi gives the contract ("Specs",
"Decisions"), but no server and no suite check has it. Each rule below is the simplest one that keeps gogi's words.

- **Which.** A read writes each `text` question of the spec, in the order of the spec. `ask` names no `text`
  question, and it removes none.
- **The turn.** A `text` question is a user message of its own. It holds the instructions of the spec and a blank
  line, when the spec has instructions. Then it holds the line `Answer this question about the <label> in plain
  text.`, a blank line, and the instructions of the question. The turn ends with the prompt for thinking off.
- **The answer.** The model writes the answer after that prompt. Each token is the most likely token of its logit
  row, and the lowest id wins a tie. The marker tokens that a text reply never picks are not candidates (11.11,
  "The generation modes").
- **The end.** A stop token of the family ends the answer, and it is not part of the text. `max_tokens` of the
  question ends it too. `text_stop` in identity is `stop_token`.
- **What follows.** The job then feeds the end of the assistant turn. That state is the start of each later
  question of the spec, until the next `text` question. A question before the first `text` question starts from the
  head of the case. No answer of a `choice`, `score` or `noul` question enters the state.
- **The response.** `described` of a read holds each written text by its question id, in the order of the spec. A
  read of a spec with no `text` question has no `described`. `usage.completion` counts the written tokens.
- **The hash.** `rendered_hash` covers the text of each turn of the read in order, with each written answer.
- **A queue read** writes no text. A decision with a queue context and a spec with a `text` question is a 400 that
  names `spec_id`.
- **The limit.** `text_tokens` is 256.
- Only the LFM3 models declare `describe`. A decision of another served model with such a spec is a 400 that names
  `spec_id`. The server holds a spec with a `text` question when one of its served models declares `describe`.

**The generation modes.** Reason: gogi's `output` already names a reply
with an object (gogi "Outputs"), so no new member is needed.

- A generation with no `output` is text. The sampler never picks `<|audio_start|>` or `<|mixed_start|>`.
- A generation with an `output` of `audio/wav` speaks. `output.options.mode` is `interleaved` by default, or
  `speech`. The family adds the opening marker: `<|mixed_start|>` or `<|audio_start|>`.
- The client writes the voice and the language in the `system` message, in the words of the model card. An example
  is "Perform TTS in American English using a female voice." The nagaya does not write a system prompt.
- In `interleaved` mode the reply has text in blocks of 6 tokens, then 12 frames of audio, in turn (`config.json`,
  `interleaved_n_text` and `interleaved_n_audio`). After `<|text_end|>` the rest of the reply is audio.
- The reply holds its text and one `audio/wav` output with all its frames, as 11.8a writes a line. The WAV is PCM, mono,
  16 bits, at 24 kHz, and each frame is 1,920 samples.
- An `assistant` audio part must name an output of this served model, as in 11.8a. The service keeps the codes of each
  output, and audio whose codes it does not hold is a 404 that names the part. Each frame of the codes is one position.
  Its row is the frame's summed codebook embedding, as the speech loop feeds a frame. The codes enter the recipe as the
  speech positions do (11.8a).

**The speech loop of a generation.** Reason: the
notes above leave these choices open. Each one is the simplest choice that follows gogi and the reference loop
(`modeling_lfm3_omni.py`, `_generate`).

- **Limits.** `max_tokens` bounds the text tokens of the reply, as gogi "Outputs" writes. `usage.completion` counts
  the text tokens only. The reference counts a frame as one step of `max_new_tokens`.
- **The frame limit.** A frame takes one position of the slot. A full slot ends the reply with `context_full`. Thus
  the slot bounds the frames, as for Qwen3-TTS (11.8a).
- **After the end frame.** The reply goes on in text until a stop token, as the reference does. In `interleaved` mode
  after `<|text_end|>`, the end frame ends the reply. The reference samples one more token there, `<|im_end|>` in the
  oracle. The nagaya does not, because its reply parser ends at `<|text_end|>`.
- **A stop before the audio ends.** A stop token, `max_tokens` or a full slot ends the reply and its audio. The WAV
  holds the frames so far, and the codec gives its last samples.
- **One output.** The text sampler never picks `<|audio_start|>` or `<|mixed_start|>`, in a speech reply too. Thus a
  reply holds one output. The reference lets a marker open a second clip.
- **Draws.** A text token draws at its position in the context, and each frame takes a position (11.2). A frame draws
  with the seed of the request and its index in the output, from 0.
- **Logit rows.** Each frame asks the engine for its logit row, because a text token can follow any frame.
- **Options.** `output.options` takes `mode` only. Another member, or a mode other than `interleaved` and `speech`,
  is a 400 that names it.
- **Continuations.** A generation with an `output` and an open message is a 400 that names `continue`. Reason: an
  open message holds text only (gogi "Continuations").
- **No head.** A reply with audio holds no head. Reason: the family refuses an `assistant` audio part ("Assistant
  audio" above).
- **Codes.** The service keeps the codes of each output in host memory, under its content id. A restart drops them,
  and 11.8a gives the 404 for codes that the service does not hold.
- **PCM.** A sample is clamped to [-1, 1], then rounded to `round(v × 32767)` with ties to even. The codec op of
  Qwen3-TTS uses the same rule (`qwen3tts_ops.cl`, `OP_PCM`).
- **The stream.** The `output` event comes after the checks of the request, before the first token. A failure after
  it is an `error` event, as for a speech model (11.8a).

**The sampler of the audio codes.** Reason: the checkpoint's generation
config is the sampler that Liquid tested (`generation_config.json`, model card "Sampling parameters").

- The text tokens use the sampler of 11.2. The defaults are each checkpoint's `generation_config.json`. Omni's text is
  greedy. `lfm3` samples with temperature 0.1 and top_k 50.
- The checkpoint of `lfm3` also names a repetition penalty of 1.1. The sampler of 11.2 has no repetition penalty, so
  v0 leaves it out.
- The audio codes use temperature 0.4, no top-k, and a repetition penalty of 1.2 over the last 64 frames of each
  codebook. A request cannot change them in v0.
- The depth decoder reveals the 12 codebooks of a frame in three passes of 4. The order of the codebooks is random
  in the reference, so the nagaya takes it from the seed and the frame (11.2).
- A frame in which any codebook gives the end code, 2,016, ends the audio. It is not part of the WAV.

**The held state of LFM3.** Reason: these parts follow the layer types of
the backbone. A ring keeps the window of a sliding layer, as Gemma 4 does (engine-abi "The parts of the Gemma 4
engine").

- The 6 full attention layers keep their keys and values for each position, as cells.
- The 5 sliding attention layers keep their keys and values in rings. A ring holds the last 2,048 positions, and a
  snapshot holds the positions that its ring holds. A restore needs the last 256. Reason: the feed gate of 8.3 reads
  each row part from position 0 (engine-abi "The first LFM3 binding").
- The 13 convolution layers keep a state of the last 6 inputs of each channel. It is a fixed part of the slot, not a
  row part, so a snapshot holds it whole at each boundary.
- The encoders of the media and the depth decoder hold no state between quanta.

**The LFM3 oracle.** Reason: each stage has its own gate, so a defect shows in the stage that has it.

- `tools/lfm3_golden.sh` writes the oracle from transformers on the CPU, in f32, with the pinned revisions.
- The gates are the vision rows, the audio rows and the logits of the backbone. Two more are the depth logits with
  forced codes, and the decoded audio of fixed codes.
- The codes of a reply are random, so the gate of the depth decoder forces the codes and compares the logits. The
  oracle of Qwen3-TTS forces its codes in the same way.
- The tolerance of each gate is measured first, as for each family (8.3).

**The tolerances of the LFM3 gates.** Reason: a card run measured each gate. These bounds keep the margin of Gemma 4's
logit gate, 1e-3 over a measured 3.4e-5.

- A stage asserts a relative error below 1e-4. The worst stage was 6.8e-6, the projector of the vision tower. The
  layers of the backbone gave 7.3e-7, the audio rows 2.0e-6, the depth logits 7.0e-7 and the codec 5.7e-6.
- A logit asserts a gap below 1e-3. The worst gap against transformers was 3.6e-5, and between two splits of one
  feed 6.4e-5.
- A wrong rotation, window or gate gave a relative error of 0.09 or more, and a wrong audio op 1.3e-2 or more.
- The constants are `STAGE_TOL` and `LOGIT_TOL` in `crates/nagaya-mk-lfm3-bmg/tests/common/mod.rs`.

**11.12 Control vectors.** A control vector changes the residual of a model while it runs (ontology "Control
vector"). The residual is the stream of rows that each layer reads and adds to. The vector holds one row for each
layer that it touches. An item of a context's list gives the vector an operation and a strength (engine-abi "Control
vectors"). The op acts only after a layer, so it never changes the output of the embedding.

gogi's contract is "Control vectors", and a context names its list in `control` (gogi "Contexts"). The engines of
LFM3 and Qwen3.8 declare `control`.

Reason for a vector at runtime: the same change can go into the weights, but a vector at runtime gives three things.

- One copy of the weights serves each strength, with no new load.
- An `add` or an `ablate` of strength 0 gives the plain model, so a test compares both in one process.
- `add` and `clamp` have no projection of the weights that gives the same result.

The cost is the prefix cache. A vector changes each row after its first layer, so it changes the keys and values.
Thus a snapshot with one list never serves a context with another list.

A control vector and the residual tap (13.1) use the same point of the pass. The tap reads the residual after a
layer, and the vector writes it. Thus a lens (13.2) and a control vector share one format, a vector for each layer.
A J-Lens finds a concept at a layer, and its Jacobian gives the vector that pushes that concept.

- **The capability.** A served model declares `control_vectors` when its engine declares `control` (engine-abi
  "Control vectors"). A non-empty `control` for another served model is a 400 that names `control`. The route
  checks it before the job, as it checks `thinking`.
- **The vector object.** A control vector is an object of the store (6.15), with the media type
  `application/x-safetensors`. Its header names the served model and the weight hash. A vector of other weights,
  or of another width, is a 400 that names `control[i].vector`.
- **The order.** The items apply in their order in the list. An item changes the residual after each layer that
  the vector holds, before the next layer reads it. The prompt pass and each decode step get the same items.
- **Strengths.** The client sets each strength. The nagaya searches for no strength.
- **Omni.** Omni's vector acts on its text passes only. The prompt pass and each text step get the items, also over the
  rows of a media part. A speech frame is a `frame` call (engine-abi "LFM3"), and it takes no control list. Its position
  still holds the rows that the steered passes before it gave. The control list is in the root recipe, so the frames of
  a steered reply have its recipe.

**The recipe of a control list.** A non-empty list goes into the root
recipe as the lab state, tag 4 (6.2). Its bytes are these, little-endian:

- a `u64` count of items;
- for each item, the 32 bytes of the vector's content id, one byte for the operation, and the `f64` strength.

The operation byte is 1 for `add`, 2 for `ablate` and 3 for `clamp`. An empty list gives tag 4 with no bytes, as
each plain context has (`recipe_root`, `crates/nagaya-store/src/lib.rs:478`). Thus the recipe of a plain context
does not change. A list of `add` and `ablate` items with strength 0 gives the bits of the plain
model, but it still has its own recipe. Reason: one rule with no special case, and the cost is one fill of the
cache for each such list.

**The strength of each layer.** The strength by layer is a member of the control item. Reason for the shape: an agent
sets one strength and a shape over the layers. A layer that it leaves out stays untouched, also for `clamp`.

- **The member.** An item can have `layers`: an object from a layer number, in decimal, to a finite number. With
  `layers`, the item acts only on the listed layers, at `strength` times the number of each. Without it, the item acts
  on each layer of its vector at `strength`, as before.
- **The checks.** An object with no member is a 400 that names `control[i].layers`. So is a key that is not a decimal
  layer number, and a number that is not finite. A layer that the vector does not hold is untouched.
- **The recipe.** An item with `layers` sets the bit `0x80` of its operation byte. A `u32` count of layers follows
  its strength, then each layer as a `u32` with its `f64` number, in rising layer order. An item without `layers`
  keeps its bytes, so the recipes of lists with no `layers` do not change.
- **The engine.** The control item of the engine ABI gets the strength of each listed layer (engine-abi "Control
  vectors"). An `add` or an `ablate` of strength 0 at a layer gets no task there, as for a whole item.

**The route that makes a vector.** `POST /council/v2/control-vectors` is
a job of the served model. Its class is `batch` unless the request names another class.

- The family renders each context of `positive` and of `negative`, with the generation prompt.
- Each context is a prompt pass with the residual tap (13.1), and no snapshot. The tap reads the row of the last
  prompt token.
- The job sums the rows of each set in `f64`, in the order of the request. Each layer's tensor is the mean of
  `positive` less the mean of `negative`, as `f32`.
- The job writes canonical safetensors (D10), with one tensor for each layer, `layers.<l>`. The store holds the
  bytes as an object. No keep holds it, so the client keeps it.

Reason: a fixed order of the sums gives the same vector on each run (engine-abi D3). A prompt pass with no
snapshot leaves no state in the store.

**The served model of a control vector.** The
choices below are open in the rules above and in gogi.

- A served model declares `control_vectors` when its engine declares `control` and the tap `residual`, and the process
  has an object store (6.15). The route needs the tap, and a vector is an object. Without either, the capability
  would promise a route that cannot run.
- `limits.control_vector_contexts` is 64. The job keeps the residual row of each context in host memory, outside the
  ledger. A row of `LFM3-1.7B` is 24 layers of 2,048 `f32` values, 192 KiB. Two sides of 64 rows hold 24 MiB.
- The job also makes no interval copy (6.13), so it leaves no entry in the store.
- A `strength` that is not finite as `f32` is a 400 that names it, such as `control[0].strength`. The engine takes
  the strength as `f32`, and the recipe keeps the `f64` that the client sent.
- A list that names more vectors than the table of the engine holds is a 400 that names `control`. The check runs
  before the job.

Reason: each choice is the smallest one that keeps every promise of the capability true.

**11.13 Context records.** gogi "Context records" has the rules. 6.17 says where the nagaya holds a record, and
11.1a says what the cluster does. The notes below fill the choices that gogi leaves to the server. Reason: each is
the simplest one.

- **Whole before the decode.** The routes `contexts`, `decisions`, `reads`, `generations` and `control-vectors` call
  `gogi_proto::resolve_request` on the body as JSON, before they decode it. Only a generation reads an open message
  of a parent as open. Thus each check and each job reads a whole context, and the two forms cannot differ.
- **A body with no `parent` goes on as its bytes came.** The route does not write it again.
- **The read of a parent runs on a call thread.** It reads a file, and the control runtime runs no blocking call
  (3.2). It counts as one of the 64 calls of 11.7.
- **Each model kind.** A generation of a speech model or of an image model takes `parent` too. The route makes the
  context whole before it chooses a service.
- **The record of a build.** A build that is not a dry run holds the record of its context when it succeeds, and
  `context` names it.
- **The records of a generation.** A generation holds the record of its context after its checks pass, before its
  job starts. It holds the record with the reply when the reply is complete, and `context` of the response or of
  `done` names it. A reply with an output is complete after its output is sealed.
- **A generation that fails** keeps the record of its context, and it holds no record of a reply (gogi).
- **A decision, a raw read and a control-vector request** hold the record of each of their contexts when the
  request succeeds. They name none. Reason: gogi's reference server holds them at the 200.
- **The text routes take a whole context.** `render`, `tokenize` and `detokenize` (11.4b) are not council routes. A
  `parent` in their context is the 400 of an unknown member. A client reads the record with the `GET` first.
- **`GET /council/v2/contexts/{id}`** gives `{id, context}`. An id that is not whole is a 400 that names `id`, as for
  a spec. A node with no home gives its 503 when no peer holds the id (11.1a, "A node with no home").
- **A pin `GET`** always has the member `contexts`, and an owner with no record has an empty list.

**11.14 Requests in flight share a prefix.**
Two requests can be in flight together with contexts that begin with the same tokens, while the store holds no
snapshot of them. Then the nagaya feeds those tokens once. Each note below is the simplest way to keep the rule:

- **A book for each engine.** The book holds the recipe of each boundary that a job in flight will hold
  (`council/in_flight.rs`). A build and a generation enter the book before they take a charge or a slot.
- **The wait.** A request waits on its call thread while the book holds a recipe of one of its boundaries. It counts
  as one of the 64 calls of 11.7. A request that waits has no claim, so two requests never wait for each other.
- **The claim.** Then the request claims each of its boundaries that the store does not hold. A boundary leaves the
  book when the job of its claim holds its snapshot, or when that job ends. Thus a job that fails releases each
  request that waits for it.
- **The numbers do not change.** The later job looks up its boundaries when it starts, as each job does (6.6). It
  finds the snapshot, and it gives the numbers that it gives alone.
- **Not yet.** A decision, a raw read, a control-vector request and the lab routes do not enter the book.
- **The reason.** Take 8 generations at once on `lfm3`, each with the same new head of 217 tokens. Without the book,
  each one feeds the head: 1,736 tokens, where 217 are needed.

The tests are in `crates/nagaya/tests/council_in_flight.rs`.

**11.15 Several replies.** gogi "Several replies" has the rules: a generation with `forks` or `contexts` gives one
reply for each. The route makes each `parent` whole first (11.13). Then
`gogi_proto::replies::split` gives one plain generation body for each reply. The route runs each body as a plain
generation and joins the results with `gogi_proto::replies::join`.

The notes below fill the choices that gogi leaves to the server. Reason: each is the simplest one.

- **The replies run at the same time.** Each reply is a call of 11.7 with its own call thread and its own job. Thus
  a request with 8 replies takes 8 of the 64 calls. Reason: the book of 11.14 then feeds a shared prompt once. Also,
  the decode quanta of the replies join in one pass on an engine that joins (engine-abi D11).
- **A reply waits for a slot of its own request.** A reply takes a slot as any job does, and a reply with no free
  slot is `busy` (11.3c). Then the reply waits for the replies of its request that run. When one or more of them
  end, the route runs the waiting replies again. Thus a request for 8 replies on an engine with 4 free slots runs
  4 replies, then 4 more. When no reply of the request has ended, the request has nothing to wait for, and it is a 429,
  as a plain generation is.
- **The reason for the wait:** a client must not count the free slots of an engine
  before it asks.
- **The limit.** `replies_per_generation` is 8 for each served model that declares `replies`. It does not depend on
  the slot count of the engine, because a reply can wait. Reason: the LFM3 engine joins 8 segments in one pass.
- **Which served models declare `replies`.** Each text served model declares it: its replies hold text. A speech
  model, an image model and an embedding model do not declare it. There, `forks` or `contexts` in a generation is a
  400 that names the member. Only the check of `model` runs before it.
- **A reply with an output.** `split` refuses `output` beside `forks` or `contexts`. Thus a served model that also
  speaks, such as `lfm3-omni`, writes text replies only in such a request.
- **The seed.** A request with no seed gets the seed of reply 0 from the same source as a plain generation. The
  route chooses it once, and `split` gives reply `i` that seed plus `i`.
- **The error.** The route waits for every reply. When one or more replies fail, the response is the error of the
  failed reply with the lowest index. With `contexts`, `gogi_proto::replies::param_in` renames its `param`. The
  records of the other replies stay held.
- **The `usage` of forks.** The replies start together, so the reply that feeds the prompt is not always reply 0.
  The top `kept` and `fed` are those of the fork that fed the most tokens. The other forks kept what it fed.
  Each of them feeds the last token of the prompt again for its logits, and the top `usage` does not count that.
- **The forward.** A request for a served model of a peer goes to that peer as it came, with `forks` or `contexts`
  in it (11.1a). The target splits it. The coordinator carries the objects of each context of `contexts`.
- **The lab routes write one reply.** `trace` decodes one generation body, so `forks`, `contexts` or `replies` there
  is the 400 of an unknown member. `ensemble` joins its own `contexts` into one reply (`docs/lab.md`). It does not
  read `forks` or `replies`.

The tests are in `crates/nagaya/tests/http_replies.rs` and `crates/nagaya/tests/http_conformance.rs`. The test of a
speech model is in `crates/nagaya/tests/speech_context.rs`.

**11.16 Layer loops.** gogi "Layer loops" has the rules. A context's `loop` runs a span of layers `count` times
(ontology "Run"). Each run of a layer has state of its own: its keys and values, or its recurrent state. A loop is a
member of each context, in the contract (decided 2026-11-18).

The notes below fill the choices that gogi leaves to the server. Each note is **Proposed** (Claude Opus 5.5,
2026-11-18). Reason: each is the simplest one.

- **The capability.** A served model declares `layer_loops` when its engine declares `layer_loops` (engine-abi
  "Layer loops"). Its `layer_loops` member copies `limits().layer_loops` of the engine. A `loop` for another served
  model is a 400 that names `loop`. The route checks it before the job, as it checks `control` (11.12).
- **The recipe.** The loop goes into the root recipe as the lab state, tag 4 (6.2), after the bytes of the control
  list. Its bytes are these, little-endian: the `u64` value `0xFFFF_FFFF_FFFF_FFFF`, then `start`, `end` and `count`,
  each a `u32`. The first value is never the count of a control list, so the bytes have one reading. A context with
  no `loop` adds no bytes, so the recipe of each plain context does not change. A loop of `count` 1 has its own
  recipe, as gogi asks.
- **The slot.** Each slot of an engine with `layer_loops` holds the state of `extra_runs` extra runs. Thus a loop
  never allocates, and the ledger charges a slot one size. The profile sets `extra_runs` for each engine. With 0, the
  engine declares no `layer_loops`.
- **Joins.** The worker joins the decode rows of contexts with an equal loop only (engine-abi D11). Contexts with
  other loops run in passes of their own. Reason: the weight ops of a joined pass run once for all rows.
- **Heads.** A looped context holds snapshots and a head as a plain context does. The parts of its state include the
  extra runs, so a snapshot holds them too.
- **Control.** A control item acts after each run of its layer. A control vector holds one row for each layer, not
  for each run.
- **Taps.** The tap `residual` gives one row for each layer, after its last run. Reason: the tap keeps its width and
  its count of rows, so a lens and a control vector keep one format (13.1).
- **Latent steps and soft inputs** run under the loop of their context (`docs/lab.md`).

`docs/lab.md`, "Layer loops", holds what other runtimes do and what to expect of a loop on a model that was not
trained with one.

**11.17 Cycles in the council.** gogi "Cycles" has the rules. A context's `cycles` is the count of cycles that a
model built to cycle runs for each position (ontology "Cycle"). A context with no `cycles` runs the served model's
`default`. A cycle is not a layer loop (decided 2026-11-21).

The notes below fill the choices that gogi leaves to the server. Each note is **Proposed** (Claude Opus 5.5,
2026-11-21). Reason: each is the simplest one, and each follows the rule of the loop in 11.16.

- **The capability.** A served model declares `cycles` when its engine declares `cycles` (engine-abi "Inkling 2").
  Its `cycles` member copies `limits().cycles` of the engine: `min`, `max`, `default`, `start` and `end`.
- **The checks.** The route checks `cycles` before the job, after `thinking`, `control` and `loop`. The check of
  `loop` and `cycles` together comes last (`check_members` of gogi-proto). The context is whole at this time, so a
  member of its parent counts.
- **The count that runs.** The council resolves the count: the context's `cycles`, or the `default` of the engine.
  Each segment of the job carries that count in `segment.cycles`. A segment of an engine with no `cycles` carries
  none. A context with a `loop` carries none too, because a segment names a loop or cycles, and not both (engine-abi
  "Inkling 2"). Such a context runs the default of an engine that declares both.
- **The recipe.** A count other than `default` goes into the root recipe as the lab state, tag 4 (6.2), after the
  bytes of the control list. Its bytes are these, little-endian: the `u64` value `0xFFFF_FFFF_FFFF_FFFE`, then the
  count as a `u32`. The first value is never the count of a control list, and it is not the marker of a loop.
- **The default count adds no bytes.** Thus a context with `cycles` equal to `default` and the same context with
  none have one root, and they share model state, as gogi allows. The identity of the engine fixes `default`, so the
  root still names the computation. The root of each context with no `cycles` does not change.
- **Joins.** The worker joins decode rows with an equal count only. Reason: the segments of one pass run one count
  (`Limits::check_segment_cycles`).
- **A reply with audio** runs at the default count only. A frame feeds its position as a plain pass. Another count
  is a 400 that names `cycles`.
- **`ensemble`** takes contexts with other counts of cycles. The rows of each count run in a pass of their own, and
  the job combines the rows of all counts (`docs/lab.md`, "`ensemble`: joined forks"). **Proposed** (Claude Opus 5.5,
  2026-11-21). Reason: the segments of one pass run one count, and each row of a step reads only its own context.
- **Rows of layers.** The tap `residual` of an engine with cycles gives the row after each run (engine-abi "Inkling
  2"). A control vector and a trace read the row of each layer after its last run, as for a loop (11.16).
  **Proposed** (Claude Opus 5.5, 2026-11-21). Reason: a vector and a trace keep one row for each layer.

The tests are in `crates/nagaya/tests/council_cycles.rs` and `crates/nagaya/tests/http_conformance.rs`.

**11.18 The served Inkling 2 models.** The kind `mk-inkling2-bmg` serves `Inkling-2-Mini` and the variable-loop
research model (milestone 35, engine-abi "Inkling 2"). Each rule below is **Proposed** (Claude Opus 5.5,
2026-11-21). Reason: each is the rule of the LFM3 kind (11.11), and the docs leave the choice open.

- **The served name.** The served model of an engine of this kind is its engine name in the profile. One kind
  serves two models, and a profile can hold two engines of one model, so the `model` of the identity is not unique.
- The `model` of the identity stays `inkling2-mini` or `inkling2-varloop`, from `loop_config.is_varloop` of the folder.
- The kind gives its served name before the open (`Factory::served_names`, 2.3, "The check of a name").
- The example profile names the engine `inkling2-mini` (`deploy/b70-pair/profile.toml`).
- **The family.** One library, `nagaya-text-inkling2`, serves both models, because the two folders share every file
  of the family. Its opener reads the key `model_dir` of `[engines.family]`. With no key, it reads
  `INKLING2_MODEL_DIR`. Thus an engine on demand, which has no family table (3.3a), still gets its family.
  - The family can read another folder than the engine. Thus the opener refuses a family whose chat template is not
    the template of the identity, and the registry compares the tokenizer hash (11.3).
- The registry compares the tokenizer hash of the family with the identity, as for each family (11.3). The engine
  takes the hash from the same library, so the two agree.
- **The generation prompt.** The template of Inkling 2 has none: the model writes `<|message_model|>` first. The
  council needs a generation prompt for the first logit row, so the family gives that token. After the prompt "chat",
  the reference writes it with a probability of 1 - 6e-12 (`inkling2-mini.decode.safetensors`). The reply parser
  then starts in the header of the message, so the name of a tool call is not content.
- **On demand.** The factory describes an engine with no load (11.1): the identity and the limits come from the
  folder, sysfs and the Level Zero driver. The checkpoint digest reads every tensor, so the description of each start
  reads about 20 GB. An open that reports another identity than its description is an error.
- **Images.** The served model lists `media.inputs` of `image/png` and `image/jpeg`. The process gives the family its
  reader of objects, as for LFM3 (11.11, "Objects"). The engine runs the tower inside the pass (engine-abi "The images
  of Inkling 2").
- **Reads.** A plain read follows "The plain read of the LFM3 family" (11.11). Reason: gogi puts the case in a user
  message of its own, and the template refuses an image in a message of the model. These rules fill the choices of
  the template:
  - The template renders the context, the case turn and each question turn with the effort of the context. A read
    with no context renders with thinking off. Reason: the template writes the effort line before the first message
    that is not a system message. Thus another effort changes the tokens of the context.
  - The question turn ends with `<|message_model|><|content_text|>Answer:`. The letter token has a leading space.
  - Each object of `case_media` is an image part of the case turn ("Media in a decision"). The family expands its
    placeholder, and it gives the rows and the record of each image, as for an image of a context.
  - A queue read holds no image. An image part of its context, or a `case_media`, is a 400 that names it.
- **Cycles.** The served model declares `cycles` with the limit of the engine (11.17). The research model is trained
  at 1 to 5 cycles. Mini is post-trained at 3, so another count on Mini runs outside its training.

## 12. Identity

**12.1 Fields.** An engine identity has the fields of `identity()` in the ABI:

- `model`;
- `checkpoint_digest`, the first of the two weight ids (12.2);
- `tokenizer_hash`, the hash of the tokenizer that the engine uses: of its tokenizer file, or of the tokenizer keys
  of its GGUF (engine-abi "v0 binding choices");
- `template`, the text of the chat template, for example the GGUF key `tokenizer.chat_template`. The identity digest
  covers it, so a new template is a new recipe. Gossip carries the digest, not the text (2.3). The engine binary, and
  the nagaya code that turns its outputs into answers;
- the lineage of the plan table;
- the bit settings, each with its class (after ds4-recipe);
- the canary hash.

The identity digest names each entry of the store (6.2). An engine with a new identity digest thus cannot read the
old entries of the old digest. The profile has no key that pins an identity: the canary checks the bits (8.1).

**The canary decides for a new build.** The identity digest does not cover the compiler or the driver. Reason: a
driver update must not make each held entry unusable when the bits do not change.

- The identity has a member `build`, with the facts of the build. It holds the SHA-256 of each kernel binary and
  engine library, and the compiler version. It also holds the driver or runtime version, and the PCI ids of the card.
- The identity digest does not cover `build`. It covers the other fields of 12.1, also the canary hash.
- `numbers` keeps the values that change bits by choice, such as the `fast_qmm` and `tiles` of the Qwen3.8 engine. A
  file path is not a number. The identity has no `engine_lib` path, because the binary links the Qwen3.8 engine. The
  Qwen3.8 engine keeps its build facts in `build`: `engine_build_sha256`,
  `engine_compiler` and `engine_hip` (`crates/nagaya-mk-qwen38-gfx1151/src/spec.rs`, `identity`). Another engine
  keeps them in `numbers`, and in the digest, until it has a canary.
- Thus a new build with the same canary bits has the same digest. Its held entries stay usable, and the gate
  thresholds still fit. A new build with other canary bits has a new digest.
- Each entry records the `build` that made it. A context with rows of two builds is then visible.
- A Level Zero engine reads the driver version with `zeDriverGetProperties`. A HIP engine reads the HIP runtime
  version. A kernel that compiles at load is hashed after the compile.
- The canary covers the shapes of the oracle fixture (8.3): a short prompt, a long prompt and a greedy decode. A
  staged engine runs each stage, and a ring part wraps.
- The Qwen3.8 engine has its canary (8.1), so `build` is out of its digest. Another engine keeps `build` in its digest
  until its own canary runs.

Reason: the bits decide if a build changed the numbers, not the facts of the build. A canary covers only the paths
that it runs, so its inputs must run each path that the limits allow.

These parts are not built yet:

- The `build` of an entry. An entry does not yet record the build that made it.
- The fatal state of 8.2.

Reason: the canary needs neither. Each one is a larger change, so it is a later step.

**The plan table of a HIP engine.** hipBLASLt does the large bf16 matrix multiplies of Qwen-Image (11.9c). A change of
hipBLASLt upstream can change the bits, and the nagaya accepts that. The plan table of the ontology holds such choices
(ontology "Plan table"). The Qwen3.8 engine has no plan table.

- The engine declares the determinism `pinned` (D3). Each GEMM runs the hipBLASLt algorithm that the plan table names
  for its shape.
- A shape is M, N and K, the data types, the transposes and the epilogue. The table maps each shape to one solution
  index of hipBLASLt.
- The first GEMM of a new shape picks its algorithm. The engine takes the heuristic candidates of hipBLASLt in order.
  It keeps the first one that gives equal bits in two runs on the same inputs.
- The engine also refuses a candidate that splits the sum over K with atomic adds, because its bits can change from
  run to run. Tensile writes the split into the kernel name. `GSUAMB` sums the parts of each buffer in order, with no
  atomics. `GSUASB` adds the parts with atomics, and `SK<n>` with `n` other than 0 is stream-K. The engine refuses
  the last two (`nagaya-mk-hip` `plan::refused`). hipBLASLt 1.2.2 on gfx1151 gives 128 candidates for 8 shapes of
  Qwen-Image, each `GSUAMB` with `SK0`.
- The table only grows: a shape keeps its pick. It is a file in the store directory, one for each lineage. The engine
  writes a pick to the file before a job uses it. Thus a restart keeps each pick.
- Each image of a slot runs its own GEMMs, so `count` changes no shape. Thus image `i` of a request does not depend
  on the other images (engine-abi "Image").
- The lineage is the hipBLASLt version, the HIP runtime version, the gfx target and the PCI ids of the device. A new
  lineage is a new identity (ontology "Plan table").
- Thus an update of ROCm can give new bits and a new digest. Each held conditioning (6.16) then fills again once.
- The pick stays the first candidate with equal bits, not the fastest one. A table that picks by time can pick another
  kernel when it fills again under the same lineage. Then the same identity gives other bits.
- A pick by time gains little. `examples/gemm_timing.rs` times each candidate of the large GEMMs of a step alone.
  There, the fastest picks take 21% less time than the first picks. In the engine, a step of 1024 by 1024 takes 4.06 s,
  against 4.2 s. A step of 1440 by 1440 takes 10.9 s, against 11.0 s. The timing adds 20 to 50 s to the first job
  of a new size.
- The `build` of the identity holds the hipBLASLt version too. The oracle tolerance (8.3) holds the change of bits
  from a new lineage. A new lineage that breaks a gate is a defect to report.

**12.2 Two weight ids.** The checkpoint digest is logical. It hashes the canonical config and each tensor's payload,
so a new packing does not change it. The arena id is the content id of the packed bytes that the engine runs.
`arena_export` returns it.

The identity carries the checkpoint digest. The arena records its own id and the strength of its check. The Qwen3.8
engine's `weight_hash` and the Gemma 4 engine's `header_hash` hash only the shard headers and sizes. They are spot
checks, and the identity labels them as spot checks until we replace them.

**12.3 Identity defects to correct in the port.**

- The first Gemma 4 engine loads EmbeddingGemma 2's `tokenizer.json` for Gemma 4. Thus its tokenizer hash names the
  wrong file. The port loads the `tokenizer.json` of the Gemma 4 checkpoint.

  The two files differ only in the post-processor. EmbeddingGemma 2's adds `<bos>` and `<eos>`. The first Gemma 4
  engine encodes with no post-processor, so the ids are the same, and only the hash differs.
- The first Gemma 4 engine's `tokenizer_hash` is the first 16 hex digits of the SHA-256 of the file. The port hashes
  the whole file and writes `sha256:` and 64 hex digits, as the Qwen3.8 library does. Reason: one form for every family.
- The first Gemma 4 engine's `weight_hash` hashes the shard headers only. The port labels it `spot-sha256:`, as the
  Qwen3.8 engine binding does (12.2).
- The metrics of the first Gemma 4 engine label card 1 `google/embeddinggemma-2`. The port takes the label from the
  identity.

## 13. Taps, lenses and composition

**13.1 Taps.** An engine declares its taps in `limits()`, with the conditions that each tap needs. A tap gives one of
these outputs: a residual at a layer (rows), routes, or gates. The store keeps large tap outputs as content-addressed
artifacts, with an owner and a budget. (The Qwen3.8 engine keeps traces as per-layer `.npy` files.)

**The taps of the Gemma 4 engine, for later.** These taps are not in the port. They are noted here so that the port
does not close a door.

- **Routes.** The Gemma 4 engine writes each layer's expert picks (`MoERouteDecision`, 128 bytes) into host memory,
  and `count_routes` reads them after each stage. A route tap costs no copy.
- **Residuals.** A J-Lens reads the residual at layer l (13.2). The residual after layer 14 already crosses host
  memory between the stages (D1).
- **Steering.** A control vector (11.12) is an input that changes bits. Thus it goes into the recipe (6.12), and a
  steered state never has a plain recipe.

**The residual tap.** An engine declares the tap
`residual` in `limits().taps`. A segment that names it gets the output `residual`. It holds, for each logit row of
the segment, the residual after each layer, as `f32`. The rows are in layer order, and each row has the width of
the residual. The residual is the stream that each layer adds to, before the norm of the next layer.

**The lens tap.** An engine declares the tap `lens` in `limits().taps`, with the condition `one logit row`. A segment
that names it gets the output `lens`. For each layer, it holds the logit row of the residual after that layer, through
the final norm and the head. The rows are `f32` in layer order, and each has the width of a logit row. A pass that names
`lens` and not one logit row is refused, because the `lens` of one row of LFM3-1.7B is 12 MiB.

**13.2 Lenses.** A lens (a J-Lens) is a state object of its own. It belongs to one model, has a content id,
and runs on a CPU device (2.1). Two models cannot share a lens. Models are compared in the lens's concept
coordinates, not in raw residuals.

The J-Lens exists outside the nagaya: the first J-Lens fit, of Gemma 4 12B, used Anthropic's `jacobian-lens` code.
The nagaya has no port of it yet. `docs/lab.md` designs the fit, the lens object and the routes
`project` and `trace` that read through it.

**13.3 Composition between models.** kaijutsu composes models through the council API, for example a judge that
follows a coder. The nagaya keeps the engines loaded side by side and schedules their classes. Tensor transfer
between engines is out of v0. If it comes, these rules apply:

- it stays on one machine;
- it uses an `inject` capability;
- each adapter is a state object with its own identity.

## 15. Open questions

Each question keeps its number, because other docs cite it. A number that is not here belongs to an answered question.

1\. Do an idle engine's GPU queues slow the other engines on its device (3.3)? A measurement answers this.

3\. How long does the conversion to canonical safetensors take on the Qwen3.8 engine and the Gemma 4 engine (10.3,
D10)?

6\. The template engine differs from jinja2 on integers past 2^127 and on a float key that is not finite. The chat
template uses none of them. Which ones do we match?

7\. How far does the Qwen3.8 engine's fast path (`QWEN38_FAST_QMM` unset) move council decisions? At 249 tokens it
moved logits by up to 1.03, and the argmax and the top 8 held (8.3).

8\. What causes the oracle gap past about 640 tokens, and how large is it at 2,000 tokens and more? The contexts
hold 4,096 and 32,768 tokens, and a long prompt takes several passes. On the exact path, the findings are these:

- The Qwen3.8 engine agrees with the oracle up to 630 tokens. The largest logit difference is 5.2e-5 at 478 tokens,
  1.9e-5 at 604 and 1.5e-5 at 630 (diagnostic prompts).
- At 655 tokens the difference is 0.29. At 684 it is 0.16. At 721 it is 0.23.
- The split into passes does not change the Qwen3.8 engine's bits, so the pass split is not the cause.
- The cause is not known: the Qwen3.8 engine or the reference. The indexer is dense below about 2,050 tokens in both.

9\. Should the control threads run at elevated priority (3.2, 3.2a)? A raised priority needs `CAP_SYS_NICE` or a
setting in the unit, such as `Nice=`. A measurement under load shows whether the pinned cores are enough.

11\. How does the park charge the host copy of a state (10.3 "Open", 4.2 "Staging")? The copy has no staging charge.
The charge path refuses while the process stops.

12\. Can a copy quantum run past the 30 s watchdog timeout on a slow NVMe (7.4)? It is not measured.

13\. How does the Qwen3.8 engine reset a context without a new allocation (engine-abi "v0 binding choices")?
`slot_clear` frees the context and allocates a new one. A `hipMalloc` that fails stops the process. The Qwen3.8
engine needs a reset call.

14\. Can the nagaya pin the two HIP runtime threads of the Qwen3.8 engine (3.2a)? The HIP runtime starts them and
sets their masks. The mask of both is 0-31 on gfx1151.

15\. Should the template id (11.3a) cover more than the tokens of the test renders? A change in how the family parses
a reply keeps the id.

16\. The job path uses `ObservedScheduler` and the evidence module (9.2, 9.6). Two things stay open (9.2). The
council thread loses the parent of `nagaya.job` for an HTTP request. The engine receives no trace context.

17\. Which masks do the threads of the Level Zero runtime and IGC get on the B70 machine (3.2b)? The nagaya does not
pin them, so a measurement shows whether they leave the pool set.

18\. How does a build find a reply that a client cut and sent back (11.3c, "the cuts from the index")? A build tries
each sealed snapshot that a pass over the index finds. **Open:** can the index hold a few tokens of prefix and suffix,
so that the search is more targeted?
