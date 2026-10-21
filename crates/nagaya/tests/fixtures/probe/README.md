# Fixed trees for the file probe

Each directory is the root path of one machine. `FileProbe::new(<directory>)` reads it as it reads `/` (spec 2.1).
A tree holds only the files that the probe reads.

| Tree | Source |
|---|---|
| `gfx1151host` | Captured from gfx1151host on 2026-10-20 (sysfs and procfs), then trimmed. |
| `b70host` | Made by hand, not captured. Capture the real tree when we port. |
| `sm121host` | Made by hand, not captured. Capture the real tree when we port. |

Decided by senpai 2026-10-20 night (vetoable): the trees for b70host and sm121host are made by hand. The coder could
not read those machines. The values come from the kernel docs for the Xe and NVIDIA layouts, and from the spec table
(2.1). Treat each number in these two trees as a guess. These are guesses:

- b70host: the PCI ids (`8086:e223` for the cards, `15b3:a2d6` for the BlueField-2), the BAR layout, the CPU count
  (24), the NVMe model and the host memory.
- sm121host: the PCI id of the GB10 (`10de:2e12`), the PCI domain `000f`, the CPU count (20), the NVMe model, the
  NIC ids and the host memory. The format of `/proc/driver/nvidia/gpus/*/information` is from the driver docs.

When a real tree replaces a hand-made one, the test for that machine shows each value that was a guess.

Keep all trees well under 200 KB in total. A test checks this.
