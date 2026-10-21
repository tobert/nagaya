# nagaya spec v0 (draft)

This spec says what the nagaya is and what it does, one rule at a time. It is written in Simplified Technical
English (`AGENTS.md`).

- `docs/ontology.md` defines each term once.
- `docs/engine-abi.md` defines the calls and the decisions D1-D10. The spec refers to them by number.
- `docs/prior-art.md`, `docs/deepinfer.md` and `docs/reviews.md` record where the rules came from.

When this spec and another doc do not agree, correct one of them. Neither one wins without a decision.

Status: draft, 2026-10-20. The sources are the design conversations with Amy and inventories of mk
(`~/src/megakernel-qwen38-flashnext-strixhalo`) and mkviolet (`~/src/mkviolet`). The code follows `AGENTS.md`. Each open
question has the mark **Open**, and section 15 lists all of them.

We change this spec before we change the code (Amy, 2026-10-20: "I want to keep updating the spec before the code so
we could redo it all again if we decide to"). The mark "Decided 2026-10-20 (reading)" is on a rule that began as a
**Proposed** reading of the text. Senpai wrote it as a rule, and Amy can veto it.
The mark "Decided by senpai 2026-10-20 night (vetoable)" is on a rule that senpai chose where the spec left a choice
open. Amy can veto it.

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
| b70host | two Intel Arc Pro B70, a BlueField-2 | 64 GB GDDR6, 128 GB DDR5, 4 TB NVMe |
| sm121host | DGX Spark, one Blackwell GPU | 128 GB unified |

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
- CPUs: the list in `sys/devices/system/cpu/online`.
- AMD GPUs: the nodes of `sys/class/kfd/kfd/topology/nodes`. A node with `gpu_id` 0 is a CPU node. The device name
  comes from `gfx_target_version`, so 110501 gives `gfx1151`. The PCI address comes from `location_id` and `domain`.
- Intel GPUs: each `sys/class/drm/card<N>` whose `device/uevent` says `DRIVER=xe`.
- NVIDIA GPUs: each directory of `proc/driver/nvidia/gpus`, and its `information` file.
- NVMe: each `nvme<C>n<N>` in `sys/block`. The size is the `size` file in 512-byte sectors.
- Network: each interface in `sys/class/net` that has a `device` directory. Virtual interfaces have none.

A device id is the position in the list: AMD GPUs first, then Intel, then NVIDIA, each in the order of the kernel
number. A missing directory that lists one kind of GPU means that the machine has no GPU of that kind. This is how a
coordinator machine joins with no GPU (2.1). Any other missing or malformed file is an error that names the file
(8.5).

A probe error at start refuses the start and exits 3 (10.4). Reason: the machine, and not a config, is wrong, so a
restart cannot help. A build without the fake engine runs `FileProbe::new("/")` through the `Probe` trait when it
starts. Decided by senpai 2026-10-20 night (vetoable).

**What the file probe cannot see.** A probe that opens no GPU runtime has these limits.

- NVIDIA: `proc/driver/nvidia` gives the model, the bus location and the minor number. It does not give the memory
  size. NVML gives it. Thus the probe accepts only the models in a short list whose memory is unified, and it reads
  their memory as the host memory. Any other model is an error.
- Xe: sysfs has no file for the size of the device memory. The probe reads the largest prefetchable memory BAR in
  the `resource` file. With resizable BAR off, the BAR is small, so the probe refuses the card.
- AMD: the probe cannot tell unified memory from dedicated memory by one flag. It compares the KFD framebuffer
  memory with `mem_info_gtt_total` and `mem_info_vram_total` of the PCI device. Equal to the first means unified.
  Equal to the second means dedicated. Equal to neither is an error.
- Free memory, device clocks and temperatures. Telemetry reads those (section 9).
- The measured edges of 2.2. The probe gives devices and sizes only.
- The NVMe namespace that holds `store_dir`. The probe lists all namespaces.
- Intel cards with the `i915` driver, and any other driver. They are not devices of the nagaya.

Decided by senpai 2026-10-20 night (vetoable): the rules of the two lists above. Reason: the probe must give a true
size or stop. A guess from a heuristic that fails without a sign would corrupt the ledger (8.5).

Decided by senpai 2026-10-20 night (vetoable): the trees for b70host and sm121host are made by hand, not captured.
The coder could not read those machines. The README in the fixture directory marks each guessed value. Capture the
real trees when we port to those machines.

Decided by senpai 2026-10-20 night (vetoable): the probe is not yet in the start path. `main.rs` still uses the fake
inventory, and the real wiring is one call, `FileProbe::new("/").probe()`. Reason: the start path takes its engines
from the same function (milestone 14), so the wiring belongs with the mk engine.

**The bootstrap config.** The bootstrap config is one TOML file, `nagaya.toml`. Its keys are `http_bind`,
`gossip_bind`, `store_dir`, `seed_peers` and `park_budget_secs`. The first three keys are required. By default,
`seed_peers` is an empty list, and `park_budget_secs` is 10 (10.3, ds4-recipe).

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

An engine name has letters, digits, `.`, `_` and `-`, and at most 128 bytes. A profile with another name does not
parse. Reason: the name is part of the gossip key `engine:<name>` (2.3). Decided 2026-10-20.

The process writes an accepted profile to `<store_dir>/profile.toml`, atomically: a temporary file, then a rename.
The file has the same shape as the input, so a read gives an equal profile. An unknown kind or key refuses the
profile before any engine opens. Reason: TOML is the engine config format, so one format is easier to read and to
check.

## 3. Process

**3.1 One process per machine.** The nagaya owns every GPU on the machine (Amy: "we'd do 1 process per host that
owns the gpus"). The nagaya is the only GPU client. A tool that needs a device sends its work through the nagaya, or
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
  is synchronous. Inside a pass, the engine runs its own work on each of its devices (D1).
- **A bounded CPU worker pool.** Rendering, tokenization, hashing, safetensors conversion, and CPU device passes
  (13.2) run on a bounded pool pinned to the remaining cores at `SCHED_BATCH`. Long CPU work yields between chunks
  to obey `max_quantum_ms`. On unified-memory machines (`gfx1151host` and `sm121host`), the scheduler pauses background CPU
  memory work during a GPU pass to protect GPU memory bandwidth.

**3.2a Control runtime first.** The control runtime (`tokio`) arrives with gossip, in milestone 10 step f. Gossip
comes before the HTTP coordinator, which joins the runtime in milestone 12. The group threads, the ledger thread
and the watchdog thread stay std threads. They do not use `tokio`. Milestone 14 applies the CPU sets (below).

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

Decided by senpai 2026-10-20 night (vetoable).

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

**Open:** 3.2 asks for elevated priority on the control threads. Milestone 14 does not apply it. A raised priority
needs `CAP_SYS_NICE` or a setting in the unit.

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

**4.3 Admission.** The ledger thread admits each call that allocates before the call runs. These calls are `open`,
`attach`, `ctx_new`, `ctx_fork`, `ctx_copy`, `part_write`, and each resize. The check includes transient peaks. A
refusal charges nothing and changes nothing.

`pass` never allocates. Its scratch for each kind is a mandatory term, which the ledger charges at `open`.

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

## 5. Scheduling

**5.1 Units.** A job is the quanta of one request, in order, with a class (ontology). A quantum is one pass or one
stage. It runs to its end, and the nagaya chooses what runs next only between quanta.

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

**5.11 Cost model (D4).** The nagaya fits a robust affine cost from production quanta, keyed by engine, kind,
segment count and context depth. Quanta that ran beside other work are not in the fit. The nagaya uses the fit for
store eviction value-per-byte (6.9) and deadline checks between quanta (5.8). An engine can have its own private
planner, which stays in the engine (mkviolet `council_server.rs`).

**A measured prior.** At the first start of an engine identity, the nagaya times a short curve for each kind. Each
point is one warmup run and the trimmed mean of five runs. The curve goes into the store's index under the identity
digest, and later starts read it. The fit starts from the curve and corrects it with production quanta. Decided
2026-10-20 (`docs/deepinfer.md` 4). DeepInfer measures its decode curve in this way
(`deepinfer/worker/model_runner.py:586-619`).

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

**The park recipe.** Until jobs track the inputs of a context (milestone 12), a park does not know them. Thus the
park recipe names the bytes. It is a root recipe with four items:

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

**Open** (kaibo, 2026-10-20): a ring part can wrap past its parent's end. Then the rows after the parent's end are
gone, and check 4 refuses the delta. The refusal changes nothing, so nothing is corrupt. A rule for a wrapped ring,
for example a ring part held whole, comes with the mkviolet port.

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
parked contexts before `READY=1`.)

**6.12 Steering and lab inputs.** Some state changes bits but is not an input, for example mk's attention cell mask
or a steering vector. Such state goes into the recipe. Thus a steered or masked result never has a plain recipe.

**6.13 Interval copies.** An engine can declare a checkpoint grain in `limits()`, in tokens. Then, between quanta,
the scheduler makes an interval copy (ontology) of each context that passed a multiple of the grain. Milestone 11.5
built the store's part. The scheduler's part needs the recipe of each live context, so it comes with milestone 12.
Until then, mk declares no grain. Decided
2026-10-20 (`docs/deepinfer.md` 3), and reworked the same day after kaibo's review.

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
- **Class and kind.** The copy is a quantum of the kind `copy` (5.2). This note does not choose its class. The job
  path of milestone 12 step 5 chooses it.

Reason: these are the smallest rules that keep the text above true.

A restore from a copy writes its parts, then replays the tokens after it, with the same boundaries. The result has
the bits of the first run. The lookup is by recipe with tag 8, so any context with the same prefix can use a copy.
The copy serves three uses: a branch from an earlier position, a truncate, and a recovery after the live state is
lost. Amy, 2026-10-20: "I suspect we only want a few copies online and won't benefit as much as a commerical service
does, but it is cool and we'll do it."

For mk, the fixed parts of a copy are about 112 MiB. They are the GDN state in f32 (108 MiB) and the conv state
(4.3 MiB, mk `DESIGN.md`). The K/V adds about 27 KiB for each token. With a grain of 256 tokens, a decode makes a
copy every 10-15 s.

On mk, the split of tokens into passes does not change bits. The probe `examples/split_bits.rs` in the mk crate
(`427fdc3`) compared one pass, one pass for each token, and uneven chunks, at 64 and 200 tokens. Every part and all
248,320 logits were equal. One pass of 200 tokens took 424 ms, and 200 passes of one token took 10,079 ms. Thus a
replay of up to 255 tokens takes about 0.5 s on mk.

## 7. IO

**7.1 One IO service per machine.** Engines (through the `io` capability) and the store send IO requests
`(file, offset, len, dest)` with a class to the IO service. The service owns these jobs:

- the rate on each edge and the queue depth;
- hedged reads across copies on different drives;
- the loss of a drive, and its return.

An IO request holds a lease on its destination until its final wait ends. This is also true after an error.

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

The timeout is 30 s, three times the longest declared quantum bound (mk: 10 s). It has no key in the
bootstrap config. The code takes it as a start parameter, so the tests use a short value.
Decided 2026-10-20 (reading).

**The watchdog during an open.** An engine's open can take longer than the timeout: mk loaded in 24-50 s on
2026-10-20. A group beats its heartbeat while its engine opens, so a long open is not a stall. Amy, 2026-10-20: the
heartbeat during an open is part of milestone 14. The group thread is blocked in the open. Thus a helper thread sends
the beats, at the heartbeat interval, until the open ends. The helper stops when the group thread leaves the open
for any reason.

The beat hides a hung open from the watchdog. The wait for the opens is the backstop (10.1). It ends a hung open
after 300 s, and the process exits 1 (10.4). The 300 s equals `TimeoutStartSec` in the unit.

Decided by senpai 2026-10-20 night (vetoable). Reason: mk waits up to 60 s for free memory, then loads for about
35 s. The old wait of 60 s ended a slow open that was not hung.

## 8. Correctness at runtime

**8.1 Canary.** The nagaya records a canary for each engine: fixed inputs and the exact outputs that the engine gave
for them. The nagaya runs the canary through the engine, with its own sampler and decision code. mk's canary has
pilot reads and an 8-step greedy decode (mk `service/canary.py`). It keeps letter logits as f32 bit patterns, the
top 5, and a SHA-256 of all logits. The canary's hash is part of the engine identity.

The nagaya runs the canary again before the engine is ready, with no tolerance. At start, a mismatch makes the
nagaya exit with code 3, "do not restart" (10.4). After start, a mismatch marks the engine fatal (8.2), and the
process continues with its other engines.

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
  maximum count of segments (5.7).

**One oracle for each family.** Each model family has one reference: today, fixed outputs of mk's Python. Every
engine of that family passes its conformance against that reference, on any vendor's device. The reference runs on
the CPU, so it does not give the bits of a GPU. Thus the oracle has a tolerance, and a spike measures the gap first.

**The Qwen3.8 oracle.** mk's Python reference (`tools/oracle_fixture.py`) writes the fixture. It holds four prompts
of 5 to 249 tokens, with the top 64 logits and the logsumexp of the last position. The bench `oracle` in the mk
crate compares mk with it. mk's exact path (`MK_FAST_QMM=0`) is the gate. Its largest gap was a logit difference
of 1.7e-5 on 2026-10-20. Decided 2026-10-20 (reading).

- On the exact path, the argmax agrees when the reference's top two differ by more than 0.25. The reference's top 8
  are in the engine's top 16. The largest logit difference is at most 1e-3, the KL at most 1e-8, and the logsumexp
  difference at most 1e-3.
- The fast path (`MK_FAST_QMM` unset) is an approximation (12.1). It keeps only the argmax and top-8 rules, and the
  bench reports its other numbers. At 249 tokens its largest logit difference was 1.03.

A family with no reference has no oracle yet: the first port of mkviolet builds the Gemma reference. Decided 2026-10-20
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
- The nagaya starts the telemetry before it reads the config, so that a config error reaches the log. It reads the
  inventory first, because the resource carries the hardware.
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
`outcome`), and `nagaya.quantum` (from `ObservedScheduler`). A failure marks its span as an error. The function
`adopt_parent` makes a span the child of the caller's span from `traceparent` and `tracestate`. It records an invalid
header in the attribute `nagaya.traceparent.invalid`. Not built: the HTTP server span and `nagaya.job`. The HTTP code
and the job path add them, and they call `adopt_parent` for each request.

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
  `device<id>`. A budget that covers several tiers has the sorted tiers joined with `+`, for example `device0+host`.
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
- `nagaya.store.restores` is new in this table. It counts every `restore`. The engine is the model name. The result
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

mk's `mk.gpu.*` names and mkviolet's `mkviolet.*` names map onto these names. Council decision metrics (count,
spread, confidence) belong to the council layer (11.1).

**9.5 Each instrument has a call site under test.** mkviolet defines `record_gpu_pass` but calls it only from its
smoke test. It calls `record_council_decision` only in tests. Its docs name instruments for routing entropy and
J-Lens drift that do not exist. Thus a test shows that each instrument records on the path that it describes.

A test runs the real call site and reads the points through an in-memory exporter (`nagaya_telemetry::testing`). The
code defines an instrument in the commit that adds its call site. The table shows each instrument that exists.

| Instrument | Call site | Test |
|---|---|---|
| `nagaya.quantum.wait.duration`, `nagaya.quantum.run.duration` | `ObservedScheduler::step` in `crates/nagaya-sched/src/observed.rs` | `crates/nagaya-sched/tests/observed.rs` |
| `nagaya.ledger.bytes`, `nagaya.ledger.refusals` | `Ledger::admit` and `Ledger::admit_context` in `crates/nagaya-ledger/src/lib.rs` | `crates/nagaya-ledger/tests/telemetry.rs` |
| `nagaya.store.bytes`, `nagaya.store.entries`, `nagaya.store.lookups`, `nagaya.store.evictions`, `nagaya.store.restores` | `Store::open`, `put`, `evict`, `remove`, `lookup` and `restore` in `crates/nagaya-store/src/lib.rs` | `crates/nagaya-store/tests/telemetry.rs` |
| `nagaya.engine.state` | `lifecycle::start` and `lifecycle::stop` in `crates/nagaya/src/lifecycle.rs` | `crates/nagaya/tests/telemetry.rs` |

These instruments have no definition yet, because they have no call site: `nagaya.quantum.tokens`, `nagaya.io.bytes`,
`nagaya.arena.handoff`, `http.server.request.duration` and the device sensors. The job path, the HTTP code and the
IO code add each one with its call site.

**9.6 Evidence on failure.** Before and after a quantum that fails, the nagaya reads the machine's counters. It logs
the change with the failure. mk reads these counters: KFD evicted ms, MemAvailable, GTT, and from `/proc/vmstat`
compaction stalls, migrations, swap-outs and direct scans.

Built as `nagaya::evidence` in `crates/nagaya/src/evidence.rs`, with fixed trees in the tests. The code that runs a
quantum calls `Machine::watch` before the quantum and `Watch::failed` in its error branch. Until the job path exists,
this module has no call site in the process. A read costs four files, so the caller may skip `watch` for decode quanta.

Evidence never hides its failure. A counter that cannot be read, or that went back, is named in the line as
`evidence unavailable`. The failure is logged all the same. The sources are `/proc/meminfo`, `/proc/vmstat`,
`/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms` and `/sys/class/drm/card*/device/mem_info_gtt_used`. The GTT is
summed over the devices, and it is absent on a machine with no amdgpu device.

**9.7 Status feed.** Each machine has one SSE status feed (mk `/mk/v1/status`, mkviolet `PassBus`):

- a snapshot when a client connects;
- one `quantum` event for each quantum, with engine, device group, kind, ms, and queue length by class;
- a keepalive every 15 s.

The feed disconnects a slow subscriber. It does not skip events for it. A subscriber that wants less volume asks for
a filter by engine or kind when it connects.

Not built. The feed needs the HTTP server (milestone 12 step 6). The scheduler side is ready: `ObservedScheduler` sees
every quantum, so it can feed the bus.

## 10. Lifecycle

**10.1 systemd.** The unit is `Type=notify`. The process sends `READY=1` when every engine in the profile has
opened, passed its canary, and woken its pinned state. `STATUS=` gives the state of each engine.

When `<store_dir>/profile.toml` does not exist, the process starts with an empty profile. An empty profile is valid,
because a machine with no engine can serve as a coordinator or hold a store tier (2.1). A process with an empty
profile sends `READY=1` when admission passes. Decided 2026-10-20 (reading).

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
that needs more arena fds than the ceiling. (mkviolet keeps 61 fds under a limit of 64, `systemd/mkviolet.service`.)

At start, the nagaya does these steps for each engine:

1. Check each stored arena against the engine's build, and record the strength of the check.
2. If an arena does not match, make sure that a cold load fits the ledger first (2.4). Then drop that engine's
   stored fds and load it cold. If a cold load does not fit, keep the fds and exit 3.
3. Duplicate each fd before use. (In mkviolet on 2026-10-16, an import closed the stored fd and caused a crash
   loop.)

`systemctl stop` clears the store. A restart keeps it.

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

The process catches SIGTERM with a handler that adds one to a static atomic counter. The handler is `signal(2)`,
declared in `extern "C"`, because std has no signal API. The main loop reads the counter every 50 ms. Reason: this
needs no new dependency. A later step can use a signal stream from `tokio` instead. Decided 2026-10-20 (reading).

**10.4 Exit codes.**

| Code | Meaning | systemd |
|---|---|---|
| 0 | a clean stop | no restart |
| 1 | a crash | restart |
| 3 | a refusal to start: a bad bootstrap config (2.1), a probe error (2.1), a profile that does not fit, a canary mismatch at start, a mismatch of a pinned identity (12.1), of the library pin, of a part size or of the weight arena size, a store that does not open, or a gossip bind that fails | `RestartPreventExitStatus=3` |
| 4 | a planned restart: a new profile (2.3, 3.3) | `RestartForceExitStatus=4` |

The restart policy follows mk's unit: `Restart=on-failure`, `RestartSec=30`, and at most 3 starts in 600 s.
systemd counts each start against that limit, also a start after code 4. Thus a fourth profile swap in 600 s fails.

An engine that does not open at start stops the process. A mismatch of a pinned identity (12.1), the library pin,
a part size or the weight arena size is a refusal to start: exit 3. Any other failure to open is a crash. The
process exits 1, and systemd restarts it within its start limit. A message on stderr names the engine and the
reason. Decided 2026-10-20 (reading).

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
not match those bits. It matches mk's filter order and its probabilities (milestone 12).

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
- The parsers for tool calls and thinking come in milestone 12. They match mk's `service/chatfmt.py`
  (`ReplyTracker`), and fixed outputs of mk's Python test them. Milestone 11 built the tokenizer, the template and
  the detokenizer.
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
- A system message after a message of another role renders in place as a system turn. gogi's `docs/council-api.md`
  says "A system message may come later", and `gogi-conformance` sends one. The template refuses such a message, so
  the library gives it a user message with stand-in text. Then the library swaps the user turn for a system turn with
  the trimmed content. System messages at the start stay with the template, which merges them into one turn.
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
- The reply parser takes the stop token as a pushed token (`<|im_end|>` or `<|endoftext|>`), and it adds no text for
  it. A token has the phase of the last text that it released, and the text it released in that phase. The text that
  the same token released in an earlier phase is in the fields of the whole reply, and in no step. The end of the
  thinking is not part of the reasoning. A call that parses but holds an integer of more than 64 bits becomes a
  malformed call with its text as written.
  Decided by senpai 2026-10-20 night (vetoable).

**11.3b The toy family.** The fake engine has no real tokenizer. For tests, it serves the toy family of gogi's
`gogi-toy` crate. Its tokens are the 256 byte values and 14 template markers, and its template is the template of
`gogi-fake`. The family library `nagaya-text-toy` puts the nagaya's family interface on `gogi-toy`, and the fake
engine gives logits over the toy vocabulary. Thus `gogi-conformance` runs against the process with the fake engine, and
needs no GPU and no GGUF. Decided 2026-10-20 (Amy: "extend gogi to have what we need").

**11.3c The job path of a build.** A build renders a context, finds the longest held prefix, restores it, feeds the
rest, and holds a snapshot at each boundary. These notes fix the choices that the text above leaves open.

- **Proposed** (Claude Sonnet 5.5, 2026-10-20), the store and the group thread. `lookup` and `restore` take the
  engine, and the engine lives on its group thread (3.2). Thus a build runs them on the group thread, and the store
  sits behind a mutex that the group threads and the park thread share. The ledger stays on its own thread. A build
  asks it by message.
- **Proposed** (Claude Sonnet 5.5, 2026-10-20), the quanta of a build. The first quantum has the kind `copy`. It
  chooses a slot, looks up the prefixes and restores the longest one. It then adds the other quanta to the job.
- **Proposed** (Claude Sonnet 5.5, 2026-10-20), the same. A feed of at most `max_pass_inputs` tokens is `prefill`.
  A seal and a put at a boundary is `copy`. A feed never crosses a boundary.
- **Proposed** (Claude Sonnet 5.5, 2026-10-20), the seal. A build calls `seal` at each boundary that it passes,
  before it reads the parts. A snapshot is a context sealed at a boundary (ontology). An engine without `seal` skips
  the call.
- **Proposed** (Claude Sonnet 5.5, 2026-10-20), the slot. A build takes the lowest free slot where the prompt fits
  (`fits` on each cells part). A build that fits no slot is `too_large`. A build with no free slot is `busy`.
- **Proposed** (Claude Sonnet 5.5, 2026-10-20), what a build lists. The response lists the boundaries that the
  store holds when the build ends. A boundary that eviction removed, below the start of the build, is not made
  again. A dry run feeds nothing and lists no snapshot.
- **Proposed** (Claude Sonnet 5.5, 2026-10-20), the head of a dry run. A dry run reports the head only when the
  store holds it. Otherwise the head is null. gogi types `head` as required, so step 6 needs Amy to choose: change
  gogi, or name the recipe.
- **Proposed** (Claude Sonnet 5.5, 2026-10-20), the pin owner. gogi names an owner with text, and the store names
  it with a `u64`. The service keeps the text and gives the store the first 8 bytes of the BLAKE3 hash of the text. A
  restart keeps the text in `pins.json` in the store directory. Entries that builds put have the owner 1.

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
- A raw read lists 8 top tokens when the request names no `top`, and at most 64. A larger `top` is a 400. A context
  and its longest suffix that reach `context_tokens` are a 413 `context_full`, as in gogi-fake.
- The text of a top token is the detokenized token. A token that is not whole text alone, such as one byte of a
  longer character, has the text U+FFFD. Its `id` names it exactly.
- The council identity's `engine` is `nagaya:` and the first 16 hex digits of the identity digest (6.2). The digest
  covers each field of `identity()`, `numbers` too.
- A generation holds its head after the reply when two facts hold. The rendering of the context with the reply has
  the boundaries of the prompt before the reply. The slot holds a prefix of that rendering, at most 8 tokens short.
  Then the job feeds the rest and puts the snapshot. Otherwise `head` is null.
- A build can start at a cut: the end of a message that the client did not mark, where the rendering keeps it. It
  starts there when the store holds a snapshot at the cut and at no higher boundary. An earlier context made that
  snapshot at a boundary of its own, such as the head of a generation after its reply. Reason: gogi's build starts
  from the longest held snapshot whose tokens begin the context, and the next request sends the reply unmarked.
- A seal changes bits, so a build from a cut keeps the cut among the boundaries of its state. Its later snapshots
  have recipes with the cut, and the response lists no snapshot at the cut. Thus a snapshot never holds a seal
  that its recipe does not name (gogi "Boundaries are content"). The service finds the cuts with a second render,
  with every message marked.
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
  `max_tokens`, it ends with `max_tokens`.
- The decision limits are mkviolet's until the profile names others: `decision_context_tokens` 3072,
  `contexts_per_decision` 8, `case_bytes` 65536 and `default_timeout_ms` 30000.

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

**11.5 Errors.** All routes use gogi's error body, `{"error": {"type", "message", "param"?}, "request_id"}`, and the
`request-id` header. Council routes use only gogi's error table (gogi `docs/council-api.md`, "Errors"). The
`/nagaya/v1` routes add two codes:

| Code | `error.type` | When |
|---|---|---|
| 409 | `conflict` | a profile `PUT` with an old version (2.3) |
| 507 | `no_memory` | a profile that does not fit its budgets (2.4) |

An unexpected failure is a 500 `internal`, and the process stops (8.5).

**11.6 Binding.** The HTTP and gossip listen addresses and seed peers are in the bootstrap config (2.1). (mkviolet
has a tailnet address in its code.)

## 12. Identity

**12.1 Fields.** An engine identity has the fields of `identity()` in the ABI:

- `model`;
- `checkpoint_digest`, the first of the two weight ids (12.2);
- `tokenizer_hash`, the hash of the tokenizer that the engine uses: of its tokenizer file, or of the tokenizer keys
  of its GGUF (engine-abi "v0 mk binding choices");
- `template`, the text of the chat template, for example the GGUF key `tokenizer.chat_template`. The identity digest
  covers it, so a new template is a new recipe. Gossip carries the digest, not the text (2.3). Decided 2026-10-20
  (reading, kaibo job-2 finding 7);
- `numbers`: the engine binary, and the nagaya code that turns its outputs into answers;
- the lineage of the plan table;
- the bit settings, each with its class (`docs/prior-art.md` 1);
- the canary hash.

The nagaya keeps each engine's expected identity in the store's index. At start, it compares the identity from
`identity()` with the expected one. A mismatch makes the old entries of that engine unusable. It is a refusal to
start only when the profile pins the identity.

**12.2 Two weight ids.** The checkpoint digest is logical. It hashes the canonical config and each tensor's payload,
so a new packing does not change it. The arena id is the content id of the packed bytes that the engine runs.
`arena_export` returns it.

The identity carries the checkpoint digest. The arena records its own id and the strength of its check. mk's
`weight_hash` and mkviolet's `header_hash` hash only the shard headers and sizes. They are spot checks, and the
identity labels them as spot checks until we replace them.

**12.3 Identity defects to correct in the port.**

- mkviolet loads EmbeddingGemma 2's `tokenizer.json` for Gemma 4. The file "differs ... by a few hundred bytes;
  which entries differ is not yet checked" (`tokenizer.rs`). Thus its tokenizer hash names the wrong file.
- mkviolet's metrics label card 1 `google/embeddinggemma-2`.

## 13. Taps, lenses and composition

**13.1 Taps.** An engine declares its taps in `limits()`, with the conditions that each tap needs (mk: unfused, one
segment). A tap gives one of these outputs: a residual at a layer (rows), routes, or gates. The store keeps large
tap outputs as content-addressed artifacts, with an owner and a budget. (mk keeps traces as per-layer `.npy` files.)

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
- mk orders jobs first in, first out in its one high tier. The nagaya uses the order in 5.4.
- mk's "park" writes through to disk and seals at each "hold". In nagaya terms, a hold is a declared boundary, and
  a write to disk is a park to NVMe (D7).
- mk's MemAvailable floor becomes the safety floor in 4.7. It does not set any size.
- mkviolet's three caches merge into the store (6.1). Its fixed budgets become ledger terms.
- mkviolet's telemetry setup leaks a runtime (`Box::leak`). Its HTTP server is hand-written, and its bind address is
  in its code.
- mkviolet has instruments without call sites (9.5) and wrong model labels (12.3).
- mkviolet's README gives `GEN_SEQ` 16384 and its `main.rs` gives 65536. Its `docs/07` is out of date.
- mkviolet's determinism class is not known. The first port runs one recipe in two processes and compares the
  content ids (engine-abi, "How mkviolet maps"). The rule that our engines are `always` or `pinned` (README) is a
  requirement on the port. The test shows if mkviolet meets it.

## 15. Open questions

1. Do an idle engine's GPU queues slow the other engines on its device (3.3)? A measurement answers this.
2. What makes `close()` safe for mk, so that engines can load and unload at runtime (3.3)?
3. How long does the conversion to canonical safetensors take on mk and mkviolet (10.3, D10)?
4. Answered 2026-10-20: on mk, one pass of many tokens gives the same bits as one pass for each token (6.13).
5. How does a profile `PUT` work across machines (2.3, 2.4, 11.5)? The machine that takes the `PUT` cannot
   return the target's 507, and 2.3 ignores an old version where 11.5 answers 409 (kaibo, 2026-10-20). Milestone 12
   settles it with the routes.
6. The template engine differs from jinja2 on integers past 2^127 and on a float key that is not finite (milestone
   11.5 text step). The chat template uses none of them. Which ones do we match?
7. How far does mk's fast path (`MK_FAST_QMM` unset) move council decisions? At 249 tokens it moved logits by up to
   1.03, and the argmax and the top 8 held (8.3). Amy, 2026-10-20: "we'll measure later".
8. What is the oracle gap at 2,000 tokens and more? The binding's contexts hold 256 tokens today, and mk allows
   32,768. The measurement needs larger contexts in the binding and a loop of passes (8.3). Amy, 2026-10-20: "we can
   save and do the measurement later".
9. Should the control threads run at elevated priority (3.2, 3.2a)? A raised priority needs `CAP_SYS_NICE` or a
   setting in the unit, such as `Nice=`. A measurement under load shows whether the pinned cores are enough.
