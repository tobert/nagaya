// SPDX-License-Identifier: MIT
// The engine's runtime layer: the launches of a pass's stages under the control block and the watchdog, the weight
// relayout, and the bf16 rounding of a context. The Rust host builds the stages and decides every rule (milestone 15
// step 6: src/pass.rs, src/worker.rs).
#pragma once
#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>

#include "model.h"
#include "rt.h"
#include "stage.h"

namespace qwen38 {

// A context: everything a pass continues from. The Rust host makes, reads, writes, builds and advances it
// (src/context.rs, which mirrors this layout as Qwen38Context; extern.hip states it). round_gdn_bf16 reads it here.
struct Context {
  uint32_t n = 0;       // tokens so far
  uint32_t max = 0;     // K/V capacity
  float *gdn_state = nullptr;   // [36][48][128][128]
  float *gdn_conv = nullptr;    // [36][3][10240]  (per layer: the last 3 raw qkv inputs, oldest first, each a row of channels)
  float *ple_conv = nullptr;    // [9][10240]
  float *kv = nullptr;          // [12][max][2 (k,v)][2 heads][256]
  float *idx_keys = nullptr;    // [12][max][128]  raw index keys
  // The pooled block-key cache (goal 08 I2, docs/indexer.md): [12][max/4][128], block b's pooled, normed, roped key from the
  // raw keys of cells 4b..4b+3, in the score's lane layout. Derived state, never stored: blocks [0, pooled_n) are valid,
  // and pooled_n == n / 4 after any pass or restore_n (a pass dies at build if it finds pooled_n past n / 4).
  float *idx_pooled = nullptr;
  uint32_t pooled_n = 0;
  int32_t ple_hist[2] = {int32_t(model::kPleEos), int32_t(model::kPleEos)};  // last 2 token ids
  // A pass extending it stopped on a spin-timeout (Engine::set_recover_spin_timeouts): its recurrent state and its
  // cells >= n are half-written (n itself never advanced). Any later pass of it stops the program; free it.
  bool poisoned = false;
};

// The megakernel's entry point (mega.hip; its own translation unit, see engine.hip).
const void *mega_kernel();
// Re-lay out an IQ3_S matrix (rows x k, GGUF row-major at src) into read-once tiles at dst (tiles.h); synchronizes.
void relayout_iq3s_tiles(const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k);
// The same for a Q8_0 matrix (the dense projections); the tiles take the same bytes.
void relayout_q8_tiles(const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k);
// The same for a Q6_K matrix (the output head).
void relayout_q6_tiles(const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k);
// The same for an IQ4_NL matrix (the expert down matrices).
void relayout_iq4_tiles(const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k);
const void *stage_kernel();
// The control op of a cut pass (control.hip): add, ablate or clamp the vector v of one layer in each of `rows` rows of
// h, with the sums in the order of src/control.rs. sum_squares is v . v from the host, above 0. op is 0 add, 1 ablate,
// 2 clamp. It waits for the device.
void control_apply(float *h, uint32_t rows, const float *v, float sum_squares, uint32_t op, float strength);

// How a launch sequence ended: the control block's status (rt.h Status), the site and the workgroup that recorded it,
// and 1 when the watchdog asked the kernel to stop (its deadline or the memory floor).
struct LaunchResult {
  uint32_t status, site, wg, watchdog_stopped;
};

class Engine {
 public:
  // The watchdog's deadline for each launch sequence, in seconds (the Rust host sets it).
  explicit Engine(double watchdog_deadline_s);
  ~Engine();
  // One cooperative launch of the megakernel over the n stages at ds (host memory the device reads), with grid
  // workgroups and the skip word of mega.hip.
  LaunchResult launch_fused(const Stage *ds, uint32_t n, int grid, uint32_t skip);
  // One launch of k_stage for each of the n stages at ds.
  LaunchResult launch_each(const Stage *ds, uint32_t n);
  // The rounding rule gdn-bf16-v1 (docs/park.md; round.hip): c's GDN recurrent state rounded in place to bf16 (round to
  // nearest, ties to even, on the f32 bit pattern; kept as f32 with the low 16 bits zero). Only the recurrent state moves.
  // A non-finite element, or a finite one that rounds to infinity, stops the program naming the first one (layer, head,
  // index, value); so does a poisoned context.
  void round_gdn_bf16(Context *c);

 private:
  void make_event();
  LaunchResult result();
  ControlBlock cb_;
  // One watchdog thread for the engine's life, begun and finished around each launch sequence (goal 03 C). Measured
  // passes (256 tokens, empty or 3584-token context, 2026-09-30, quiet GPU) take 0.48-0.74 s, so the 10 s default is
  // >10x the longest and still far under the gfx ring timeout.
  Watchdog wd_;
  hipEvent_t launched_ev_ = nullptr;  // recorded after each launch (the watchdog's "kernel still runs" query)
};

}  // namespace qwen38
