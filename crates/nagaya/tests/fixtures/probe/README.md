# Fixed trees for the file probe

Each directory is the root path of one machine. `FileProbe::new(<directory>)` reads it as it reads `/` (spec 2.1).
A tree holds only the files that the probe reads.

| Tree | Machine | Source |
|---|---|---|
| `strix-halo` | An AMD Strix Halo machine: one gfx1151 GPU with unified memory. | Captured (sysfs and procfs), then trimmed. |
| `b70-pair` | Two Intel Arc Pro B70 cards and an AMD iGPU. | Captured (sysfs and procfs), then trimmed. |
| `dgx-spark` | An NVIDIA GB10 with unified memory. | Captured (sysfs and procfs), then trimmed. |

The `b70-pair` tree is real. The capture copied only the files that the probe reads. `proc/meminfo` keeps its first
three lines. Each symlink was resolved, and the files sit at the path of the link. A network speed that the kernel
refuses to read (a down link) is stored as `-1`. The tree holds the two Xe cards, the AMD display GPU and three NICs.

The `b70-pair` tree also holds the files of the device CPU sets (spec 3.2b): `local_cpulist` of the three GPUs, and
`cache/index3/shared_cpu_list` of each CPU. The capture copied them from the machine. The `strix-halo` and
`dgx-spark` trees have none of them, because a machine with one card reads no CPU list.

The `dgx-spark` tree is real. `proc/driver/nvidia/gpus/000f:01:00.0/information` keeps the real `Model: Unknown` line
with the GPU UUID zeroed. A Wi-Fi interface whose `speed` file returns `EINVAL` is stored as `-1`.

Keep all trees well under 200 KB in total. A test checks this.
