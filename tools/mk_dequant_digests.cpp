// SPDX-License-Identifier: MIT
// Write the dequant digests of mk's C++ GGUF reader, the oracle of the Rust port (AGENTS.md milestone 15, step 3).
//
// The tool links mk's own gguf.cpp and dequant.cpp, with the flags of build.rs for those units. It writes two
// fixtures of the crate nagaya-mk-qwen38-gfx1151:
//
// - `blocks`: the SHA-256 of the dequant of fixed pseudo-random blocks of each type, and of every f16 and bf16 value.
//   It needs no model.
// - `shards`: for each tensor of the three shards, its table row and the SHA-256 of its dequant (every row, and a
//   sample of three rows). It also writes the PLE keys that the engine reads.
//
// Build and run it from the repository root, with the C++ copy still in the crate (commit of this file):
//
//   G=crates/nagaya-mk-qwen38-gfx1151/hip/src/gguf
//   /opt/rocm/llvm/bin/clang++ -I$G -x c++ -O2 -g -DNDEBUG -std=gnu++20 -fPIC -Wall -Wextra -Wno-unused-parameter \
//     -ffp-contract=off -fno-fast-math -pthread tools/mk_dequant_digests.cpp $G/gguf.cpp $G/dequant.cpp -lcrypto \
//     -o target/mk_dequant_digests
//   target/mk_dequant_digests blocks > crates/nagaya-mk-qwen38-gfx1151/tests/fixtures/dequant/blocks.txt
//   target/mk_dequant_digests shards S1 S2 S3 > crates/nagaya-mk-qwen38-gfx1151/tests/fixtures/dequant/shards.txt
//
// The pseudo-random bytes come from splitmix64, so the Rust test makes the same bytes.
#include <openssl/evp.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "dequant.hpp"
// The tool needs each tensor's shard and offset, which Model keeps private. This is a test tool, so it opens the class.
#define private public
#include "gguf.hpp"
#undef private

namespace {

using mkgguf::GgmlType;

struct Sha {
  EVP_MD_CTX *c = EVP_MD_CTX_new();
  Sha() { EVP_DigestInit_ex(c, EVP_sha256(), nullptr); }
  ~Sha() { EVP_MD_CTX_free(c); }
  void update(const void *p, size_t n) { EVP_DigestUpdate(c, p, n); }
  std::string hex() {
    unsigned char d[32];
    unsigned int n = 0;
    EVP_DigestFinal_ex(c, d, &n);
    char s[65];
    for (int i = 0; i < 32; ++i) snprintf(s + 2 * i, 3, "%02x", d[i]);
    return s;
  }
};

uint64_t splitmix64(uint64_t &state) {
  uint64_t z = (state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// `n` pseudo-random bytes from the seed: each splitmix64 output gives 8 bytes, little-endian.
std::vector<uint8_t> random_bytes(uint64_t seed, size_t n) {
  std::vector<uint8_t> out(n);
  uint64_t state = seed;
  for (size_t i = 0; i < n; i += 8) {
    const uint64_t v = splitmix64(state);
    for (size_t b = 0; b < 8 && i + b < n; ++b) out[i + b] = uint8_t(v >> (8 * b));
  }
  return out;
}

constexpr GgmlType kTypes[] = {GgmlType::F32, GgmlType::BF16, GgmlType::Q8_0, GgmlType::Q6_K,
                               GgmlType::IQ3_S, GgmlType::IQ4_NL, GgmlType::IQ4_XS};
constexpr size_t kBlocks = 64;

int blocks() {
  printf("# The dequant of mk's C++ copy (dequant.cpp) on fixed inputs. tools/mk_dequant_digests.cpp wrote it.\n");
  printf("# type, blocks, the SHA-256 of the f32 output. The input is %zu blocks of splitmix64 bytes, seed = type id.\n", kBlocks);
  for (GgmlType t : kTypes) {
    const auto tr = mkgguf::ggml_type_traits(t);
    const auto in = random_bytes(uint64_t(t), kBlocks * tr.type_size);
    std::vector<float> out(kBlocks * tr.block_size);
    mkgguf::dequant_f32(t, in.data(), int64_t(out.size()), out.data());
    Sha h;
    h.update(out.data(), out.size() * sizeof(float));
    printf("%s %zu %s\n", mkgguf::ggml_type_name(t), kBlocks, h.hex().c_str());
  }
  Sha h16, hb16;
  for (uint32_t v = 0; v < 65536; ++v) {
    const float a = mkgguf::fp16_to_fp32(uint16_t(v)), b = mkgguf::bf16_to_fp32(uint16_t(v));
    h16.update(&a, sizeof a);
    hb16.update(&b, sizeof b);
  }
  printf("f16-all 65536 %s\n", h16.hex().c_str());
  printf("bf16-all 65536 %s\n", hb16.hex().c_str());
  return 0;
}

struct Row {
  std::string line;
};

int shards(const std::vector<std::string> &paths) {
  const mkgguf::Model m(paths);
  const size_t n = m.n_tensors();
  std::vector<std::string> lines(n);
  std::atomic<size_t> next{0};
  std::mutex say;
  auto work = [&] {
    std::vector<float> buf;
    for (size_t i; (i = next.fetch_add(1)) < n;) {
      const auto &t = m.tensor(i);
      size_t shard = 0;
      for (; shard < m.shards_.size(); ++shard) {
        const auto *base = static_cast<const uint8_t *>(m.shards_[shard].map);
        if (t.data >= base && t.data < base + m.shards_[shard].size) break;
      }
      const uint64_t offset = uint64_t(t.data - static_cast<const uint8_t *>(m.shards_[shard].map));
      const uint64_t ne0 = t.ne[0], rows = t.n_elements / ne0;
      // A chunk of at most 16 Mi floats.
      const uint64_t per = std::max<uint64_t>(1, (16ull << 20) / ne0);
      buf.resize(per * ne0);
      Sha full;
      for (uint64_t r = 0; r < rows; r += per) {
        const uint64_t r1 = std::min(rows, r + per);
        m.dequant_rows(t.name, r, r1, buf.data());
        full.update(buf.data(), (r1 - r) * ne0 * sizeof(float));
      }
      std::vector<uint64_t> pick = {0, rows / 2, rows - 1};
      pick.erase(std::unique(pick.begin(), pick.end()), pick.end());
      Sha sample;
      for (uint64_t r : pick) {
        m.dequant_rows(t.name, r, r + 1, buf.data());
        sample.update(buf.data(), ne0 * sizeof(float));
      }
      char head[512];
      snprintf(head, sizeof head, "%zu %s %s %u %lu %lu %lu %lu %lu %zu %lu ", i, t.name.c_str(),
               mkgguf::ggml_type_name(t.type), t.n_dims, (unsigned long)t.ne[0], (unsigned long)t.ne[1],
               (unsigned long)t.ne[2], (unsigned long)t.ne[3], (unsigned long)t.n_bytes, shard, (unsigned long)offset);
      lines[i] = std::string(head) + full.hex() + " " + sample.hex();
      std::lock_guard<std::mutex> g(say);
      fprintf(stderr, "%zu/%zu %s\n", i + 1, n, t.name.c_str());
    }
  };
  std::vector<std::thread> th;
  for (unsigned k = 0; k < std::max(1u, std::thread::hardware_concurrency()); ++k) th.emplace_back(work);
  for (auto &x : th) x.join();

  printf("# The tensor table and dequant of mk's C++ copy (gguf.cpp, dequant.cpp). tools/mk_dequant_digests.cpp\n");
  printf("# wrote it. A tensor: index, name, type, n_dims, ne0-ne3, bytes, shard, offset in the shard, the SHA-256\n");
  printf("# of the f32 dequant of every row, and of rows 0, rows/2 and rows-1 (each once).\n");
  for (size_t s = 0; s < m.n_shards(); ++s) {
    const char *slash = strrchr(paths[s].c_str(), '/');
    printf("shard %zu %s %zu %lu\n", s, slash ? slash + 1 : paths[s].c_str(), m.shards_[s].size,
           (unsigned long)m.data_offset(s));
  }
  printf("key qwen4exp.ple.eos_token_id %ld\n", (long)m.get_i64("qwen4exp.ple.eos_token_id"));
  for (const char *k : {"qwen4exp.ple.layer_multipliers", "qwen4exp.ple.head_offsets", "qwen4exp.ple.head_vocab_sizes"}) {
    printf("key %s", k);
    for (int64_t v : m.get_arr_i64(k)) printf(" %ld", (long)v);
    printf("\n");
  }
  printf("tensors %zu\n", n);
  for (const auto &l : lines) printf("tensor %s\n", l.c_str());
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc == 2 && !strcmp(argv[1], "blocks")) return blocks();
  if (argc == 5 && !strcmp(argv[1], "shards")) return shards({argv[2], argv[3], argv[4]});
  fprintf(stderr, "usage: %s blocks | %s shards S1 S2 S3\n", argv[0], argv[0]);
  return 2;
}
