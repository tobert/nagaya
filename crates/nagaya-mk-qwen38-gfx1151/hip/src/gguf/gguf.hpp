// SPDX-License-Identifier: MIT
// Our own GGUF v3 reader: header, KV metadata (every value type, including arrays), tensor
// infos, mmap'd across a model's shards, with a bit-identical dequant path (dequant.hpp).
// Original code (not a ggml/llama.cpp port); see types.hpp / tables.hpp / dequant.hpp for the
// ported pieces this builds on.
//
// Crash-over-corruption, throughout: anything the file promises but doesn't deliver (bad
// magic, wrong version, an unknown tensor type, a misaligned or out-of-range tensor offset, a
// split-file mismatch, a duplicate KV key or tensor name) throws std::runtime_error with a
// message that says what and where. There is no silent fallback for a malformed GGUF.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "types.hpp"

namespace mkgguf {

// One parsed KV metadata value. Every integer/bool scalar widens to i64 and every float
// scalar widens to f64 at parse time; a getter narrows back down and range-checks. Arrays
// store one of arr_i/arr_f/arr_s, chosen by elem_type's family (bools go into arr_i as 0/1).
struct KV {
    GgufValueType type = GgufValueType::UINT8;       // the file's own tag: a scalar type, or ARRAY
    GgufValueType elem_type = GgufValueType::UINT8;  // meaningful only when type == ARRAY
    int64_t i = 0;
    double f = 0.0;
    bool b = false;
    std::string s;
    std::vector<int64_t> arr_i;
    std::vector<double> arr_f;
    std::vector<std::string> arr_s;
};

// One tensor's shape and a pointer into its owning shard's mmap. ne[0] is the fastest-varying
// (innermost) dimension, exactly as the GGUF file stores it: a "row" is ne[0] elements, and
// there are n_elements / ne[0] of them.
struct TensorInfo {
    std::string name;
    GgmlType type = GgmlType::F32;
    uint32_t n_dims = 0;
    uint64_t ne[4] = {1, 1, 1, 1};
    uint64_t n_elements = 0;
    uint64_t n_bytes = 0;
    const uint8_t* data = nullptr;  // borrowed; valid for the Model's lifetime, never owned here
};

// One model, opened across every shard of a split GGUF (split.count / split.no). Each shard
// is mmap'd read-only and never copied into anonymous memory: this is how an 88 GB model gets
// read on a box with far less than 88 GB of headroom to spare.
class Model {
public:
    // shard_paths must be given in split order (index i's file must carry split.no == i) and
    // must together cover split.count == shard_paths.size(); anything else is a hard error.
    explicit Model(const std::vector<std::string>& shard_paths);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    size_t n_tensors() const { return tensors_.size(); }
    size_t n_shards() const { return shards_.size(); }
    // Shard i's tensor data offset: everything before it is the header (metadata and tensor infos).
    uint64_t data_offset(size_t shard) const;
    const TensorInfo& tensor(size_t index) const;
    const TensorInfo& tensor(const std::string& name) const;
    bool has_tensor(const std::string& name) const;

    bool has_kv(const std::string& key) const;
    int64_t get_i64(const std::string& key) const;
    double get_f64(const std::string& key) const;
    bool get_bool(const std::string& key) const;
    const std::string& get_str(const std::string& key) const;
    size_t get_arr_len(const std::string& key) const;
    std::vector<int64_t> get_arr_i64(const std::string& key) const;
    std::vector<double> get_arr_f64(const std::string& key) const;
    std::vector<std::string> get_arr_str(const std::string& key) const;

    // Dequantizes rows [row0, row1) of tensor `name` (each row is ne[0] elements) into `out`,
    // which must hold at least (row1 - row0) * ne[0] floats. Crashes on an out-of-range row,
    // an ne[0] that isn't a multiple of the type's block size, or an unknown tensor name.
    void dequant_rows(const std::string& name, uint64_t row0, uint64_t row1, float* out) const;

    // Asks the kernel to drop the page-cache pages lying wholly inside [p, p + n), which must sit
    // within one shard's mapping (anything else throws). Pages shared with bytes outside the range
    // are kept, so a neighbouring tensor (the PLE table, which the host gathers from after load)
    // never loses its edge pages. Our own view of those pages is unmapped first (MADV_DONTNEED),
    // since the kernel won't evict a mapped page, then the file is advised (POSIX_FADV_DONTNEED).
    // Best effort: the kernel may keep the pages (another process maps them, say). The mapping
    // stays valid; a later read re-faults from disk.
    void drop_cached(const uint8_t* p, uint64_t n) const;

private:
    struct Shard {
        int fd = -1;
        void* map = nullptr;
        size_t size = 0;
        size_t data_offset = 0;  // where this shard's tensor data starts (after the header, aligned)
        std::string path;
    };

    std::vector<Shard> shards_;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, size_t> tensor_index_;
    std::unordered_map<std::string, KV> kv_;

    void parse_shard(size_t shard_idx, const std::vector<std::string>& shard_paths);
    void merge_kv(const std::string& key, KV&& value, const std::string& shard_path);
    const KV& require_kv(const std::string& key) const;
};

}  // namespace mkgguf
