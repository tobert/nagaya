# Fixed trees for the file probe

Each directory is the root path of one machine. `FileProbe::new(<directory>)` reads it as it reads `/` (spec 2.1).
A tree holds only the files that the probe reads.

| Tree | Source |
|---|---|
| `gfx1151host` | Captured from gfx1151host on 2026-10-20 (sysfs and procfs), then trimmed. |
| `b70host` | Captured from b70host on 2026-10-21 (sysfs and procfs), then trimmed. |
| `sm121host` | Made by hand, not captured. Capture the real tree when we port. |

The b70host tree is real. The capture copied only the files that the probe reads. `proc/meminfo` keeps its first
three lines. Each symlink was resolved, and the files sit at the path of the link. A network speed that the kernel
refuses to read (a down link) is stored as `-1`. The tree holds the two Xe cards, the AMD display GPU and three NICs.

The b70host tree also holds the files of the device CPU sets (spec 3.2b): `local_cpulist` of the three GPUs, and
`cache/index3/shared_cpu_list` of each CPU. The capture copied them from the machine. The gfx1151host and sm121host trees
have none of them, because a machine with one card reads no CPU list.

The tree for sm121host is made by hand. The coder could not read that machine. The values come from the kernel docs for
the NVIDIA layout, and from the spec table (2.1). Treat each number in it as a guess. These are guesses:

- sm121host: the PCI id of the GB10 (`10de:2e12`), the PCI domain `000f`, the CPU count (20), the NVMe model, the
  NIC ids and the host memory. The format of `/proc/driver/nvidia/gpus/*/information` is from the driver docs.

When a real tree replaces a hand-made one, the test for that machine shows each value that was a guess.

Keep all trees well under 200 KB in total. A test checks this.
