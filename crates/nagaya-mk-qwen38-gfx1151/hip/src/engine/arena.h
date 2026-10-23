// SPDX-License-Identifier: MIT
// The weight arena (Amy, 2026-09-30: "2, stop processing and chill, leaving the model ... for iterating on our
// kernel, saving a load, if that works"). The engine keeps every weight it uploads (all but the PLE table, which
// stays in the GGUF mapping) in one memfd, laid out as the kernels read it, and seals it once filled
// (F_SEAL_SEAL|SHRINK|GROW|FUTURE_WRITE): nothing can map it writable again or resize it. A second process opens it by
// /proc/<pid>/fd/<n>, checks it against its own plan and conversion, maps it read-only and registers it, and runs on
// the same pages without a load (Engine(shards, attach)). probes/RESULTS-2026-09-26.md 2026-09-30: a read-only
// registration of a sealed memfd reads at 238 GB/s on gfx1151.
//
// Little-endian. At 0 the Header (4096 bytes reserved); at manifest_off, n_tensors Entry records in offset order;
// tensor data from data_off (2 MiB aligned, for THP), each tensor at a kAlign-aligned offset. harness/arena.py reads
// the same layout (tests/engine/test_arena.py checks the two against a real arena).
#pragma once

#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <string>

namespace mk::arena {

constexpr char kMagic[8] = {'M', 'K', 'A', 'R', 'E', 'N', 'A', '\0'};
// Bump with any change to this layout or to a tile format (tiles.h) the manifest names. A forgotten bump is caught
// at attach anyway: the attacher converts the smallest tensor of every type itself and compares the bytes.
constexpr uint32_t kFormat = 1;
constexpr uint64_t kAlign = 4096, kHeaderBytes = 4096, kDataAlign = 2ull << 20;
constexpr unsigned kSeals = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_FUTURE_WRITE;
constexpr uint32_t kMaxShards = 4;

struct ShardId {  // a GGUF shard as the creator saw it: realpath, size and mtime must match at attach
  uint64_t size;
  int64_t mtime_ns;
  char path[240];
};

struct Header {
  char magic[8];
  uint32_t format, tiles, n_tensors, n_shards;
  uint64_t bytes, manifest_off, data_off;
  ShardId shards[kMaxShards];
};

struct Entry {
  char name[64];
  uint32_t type, pad;
  uint64_t ne[3];
  uint64_t row_bytes, offset, bytes, reserved;
};

static_assert(sizeof(ShardId) == 256 && offsetof(Header, shards) == 48 && sizeof(Header) <= kHeaderBytes);
static_assert(sizeof(Entry) == 128);

// The shard's identity now (dies if it can't be read).
ShardId shard_id(const std::string &path);

}  // namespace mk::arena
