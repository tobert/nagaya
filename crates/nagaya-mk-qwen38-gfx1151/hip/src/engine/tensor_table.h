// SPDX-License-Identifier: MIT
// The tensor table: the tensors of the GGUF shards, as the Rust reader found them (src/gguf.rs, src/table.rs).
// The engine parses no GGUF (milestone 15 step 3). The tensor data stays in the reader's read-only maps of the shards,
// and the reader keeps them for the engine's life: the host gathers PLE rows from them (Engine::ple_rows).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "rt.h"

extern "C" {
// One tensor record. ne[0] is the innermost dimension; a row is ne[0] values. data points into the shard's map.
struct mk_tensor {
  const char *name;
  uint32_t type;  // the ggml type id
  uint32_t n_dims;
  uint64_t ne[4];
  uint64_t n_elements;
  uint64_t n_bytes;
  const uint8_t *data;
};

// The table that the Rust reader gives to mk_engine_open, mk_engine_attach and mk_arena_check. Every pointer stays
// valid for the call; the tensor data and the reader stay valid for the engine's life.
struct mk_tensor_table {
  uint64_t n_tensors;
  const mk_tensor *tensors;  // in file order, shard after shard
  uint64_t n_shards;
  const char *const *shards;  // the shard paths, in split order (the arena records their identity)
  int64_t ple_eos_token_id;   // qwen4exp.ple.eos_token_id
  uint64_t n_ple_layer_multipliers;
  const int64_t *ple_layer_multipliers;  // qwen4exp.ple.layer_multipliers
  uint64_t n_ple_head_offsets;
  const int64_t *ple_head_offsets;  // qwen4exp.ple.head_offsets
  uint64_t n_ple_head_vocab_sizes;
  const int64_t *ple_head_vocab_sizes;  // qwen4exp.ple.head_vocab_sizes
  const void *reader;
  // Drops the page-cache pages wholly inside [p, p + n), which lies in one shard's map. Stops the program on an error.
  void (*drop_cached)(const void *reader, const uint8_t *p, uint64_t n);
  // Dequantizes n values of ggml type `type` at src into dst, bit for bit as ggml. Stops the program on an error.
  void (*dequant_f32)(uint32_t type, const void *src, int64_t n, float *dst);
};
}

namespace mk {

// One tensor of the table, with its name copied.
struct GgufTensor {
  std::string name;
  uint32_t type = 0;
  uint32_t n_dims = 0;
  uint64_t ne[4] = {1, 1, 1, 1};
  uint64_t n_elements = 0;
  uint64_t n_bytes = 0;
  const uint8_t *data = nullptr;  // borrowed from the reader's map
};

// The engine's copy of a mk_tensor_table. A null or empty field stops the program.
class TensorTable {
 public:
  explicit TensorTable(const mk_tensor_table *t) {
    if (!t) MK_DIE("the tensor table is null");
    if (!t->tensors || t->n_tensors == 0) MK_DIE("the tensor table holds no tensors");
    if (!t->shards || t->n_shards == 0) MK_DIE("the tensor table names no shards");
    if (!t->reader || !t->drop_cached || !t->dequant_f32) MK_DIE("the tensor table has no reader calls");
    tensors_.resize(t->n_tensors);
    for (uint64_t i = 0; i < t->n_tensors; ++i) {
      const mk_tensor &s = t->tensors[i];
      if (!s.name || !s.data) MK_DIE("tensor %lu of the table has no name or no data", (unsigned long)i);
      GgufTensor &d = tensors_[i];
      d.name = s.name;
      d.type = s.type;
      d.n_dims = s.n_dims;
      for (int j = 0; j < 4; ++j) d.ne[j] = s.ne[j];
      d.n_elements = s.n_elements;
      d.n_bytes = s.n_bytes;
      d.data = s.data;
      if (!index_.emplace(d.name, i).second) MK_DIE("the tensor table holds %s twice", d.name.c_str());
    }
    for (uint64_t i = 0; i < t->n_shards; ++i) {
      if (!t->shards[i]) MK_DIE("shard %lu of the table has no path", (unsigned long)i);
      shards_.emplace_back(t->shards[i]);
    }
    ple_eos_token_id_ = t->ple_eos_token_id;
    copy(t->ple_layer_multipliers, t->n_ple_layer_multipliers, ple_layer_multipliers_, "layer_multipliers");
    copy(t->ple_head_offsets, t->n_ple_head_offsets, ple_head_offsets_, "head_offsets");
    copy(t->ple_head_vocab_sizes, t->n_ple_head_vocab_sizes, ple_head_vocab_sizes_, "head_vocab_sizes");
    reader_ = t->reader;
    drop_cached_ = t->drop_cached;
    dequant_f32_ = t->dequant_f32;
  }

  size_t n_tensors() const { return tensors_.size(); }
  const GgufTensor &tensor(size_t i) const {
    if (i >= tensors_.size()) MK_DIE("tensor index %zu of %zu", i, tensors_.size());
    return tensors_[i];
  }
  const GgufTensor &tensor(const std::string &name) const {
    const auto it = index_.find(name);
    if (it == index_.end()) MK_DIE("the tensor table has no tensor %s", name.c_str());
    return tensors_[it->second];
  }
  const std::vector<std::string> &shards() const { return shards_; }
  int64_t ple_eos_token_id() const { return ple_eos_token_id_; }
  const std::vector<int64_t> &ple_layer_multipliers() const { return ple_layer_multipliers_; }
  const std::vector<int64_t> &ple_head_offsets() const { return ple_head_offsets_; }
  const std::vector<int64_t> &ple_head_vocab_sizes() const { return ple_head_vocab_sizes_; }
  void drop_cached(const uint8_t *p, uint64_t n) const { drop_cached_(reader_, p, n); }
  void dequant_f32(uint32_t type, const void *src, int64_t n, float *dst) const { dequant_f32_(type, src, n, dst); }

 private:
  static void copy(const int64_t *p, uint64_t n, std::vector<int64_t> &out, const char *what) {
    if (n && !p) MK_DIE("the PLE key %s of the table has %lu values and no pointer", what, (unsigned long)n);
    out.assign(p, p + n);
  }

  std::vector<GgufTensor> tensors_;
  std::unordered_map<std::string, size_t> index_;
  std::vector<std::string> shards_;
  int64_t ple_eos_token_id_ = 0;
  std::vector<int64_t> ple_layer_multipliers_, ple_head_offsets_, ple_head_vocab_sizes_;
  const void *reader_ = nullptr;
  void (*drop_cached_)(const void *, const uint8_t *, uint64_t) = nullptr;
  void (*dequant_f32_)(uint32_t, const void *, int64_t, float *) = nullptr;
};

}  // namespace mk
