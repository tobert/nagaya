// SPDX-License-Identifier: MIT
// Device traps that say which one fired (PLAN.md Problems 2026-09-29 00:11: "HSA_STATUS_ERROR_EXCEPTION ... which trap
// unknown"). A trap site calls HIP_TRAP(status, file): the lane first records a distinct Status and a site id (the file's
// TrapFile and the source line) in the process's trap record, then traps as before (never skipped, never recoverable).
// The trap kills the queue and the HIP runtime aborts the process; the SIGABRT handler installed by trap_sink_install()
// (rt.cpp) reads the record and writes "device trap: <status> at site <file>:<line> (wg <n>)" to stderr first.
//
// How the ops reach the record: one `static __device__` pointer per translation unit (g_trap_ctl below), pointing at a
// Control in coherent host memory (a Control like the engine's, with its own lifetime: never freed, so no Engine or
// op-test library can leave a dangling pointer behind). One per shared object, not per process: mkrt is a static
// library linked into the engine and into each op library, so each carries its own record, binders and SIGABRT handler
// (chained), and each must call trap_sink_install() itself (the Engine constructor; tests/ops/mkops.py lib()). Every
// TU that includes this header registers a binder at static-init time (host pass only); trap_sink_install() runs the
// binders, which copy the record's address into that TU's g_trap_ctl (hipMemcpyToSymbol). Without an installed sink
// g_trap_ctl is null and a trap just traps. Kernel arguments are untouched, so k_mega's budget is (nearly) unchanged:
// the record is loaded only on the trap path.
#pragma once
#include "rt.h"

namespace mk {

// Which file a trap site is in (the high half of the site id; the low half is the line).
enum TrapFile : uint32_t {
  kTrapFileNone = 0,
  kTrapFileQmmh = 1,
  kTrapFileQmm,
  kTrapFileQmmOracle,
  kTrapFileSmall,
  kTrapFileMoe,
  kTrapFileMoeHip,
  kTrapFileAttn,
  kTrapFileStage,
  kTrapFileCount,
};
const char *trap_file_name(uint32_t file);  // async-signal-safe

// visibility("default"): a static device variable is otherwise a local symbol of its code object, and the runtime can't
// find it by name for hipMemcpyToSymbol ("Cannot create GlobalVar Obj for symbol").
static __device__ __attribute__((visibility("default"))) Control *g_trap_ctl = nullptr;

// Record the first failure of the process in the trap record (if a sink is installed). Every lane may call it; the
// first wins, as in set_status (device.h). The release fence makes the stores reach coherent memory before the trap
// takes the wave down.
[[maybe_unused]] static __device__ __attribute__((always_inline)) void trap_record(uint32_t status, uint32_t site) {
  Control *c = g_trap_ctl;
  if (c == nullptr) return;
  uint32_t expected = 0;
  if (__hip_atomic_compare_exchange_strong(&c->status, &expected, status, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_SYSTEM)) {
    __hip_atomic_store(&c->status_wg, blockIdx.x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(&c->status_site, site, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
}

}  // namespace mk

// A trap that names itself. `status` is a mk::Status (rt.h), `file` a mk::TrapFile.
#define HIP_TRAP(status, file)                                                                          \
  do {                                                                                                 \
    ::mk::trap_record((status), (uint32_t(file) << 16) | uint32_t(__LINE__));                          \
    __builtin_trap();                                                                                  \
  } while (0)

namespace mk {
// Host: the binder of this translation unit. Declared for both the host and the device pass of the compiler: the device
// pass has to see the host's use of g_trap_ctl (HIP_SYMBOL), or it keeps the variable internal, folds it to null (nothing
// ever stores to it in device code) and drops every recording. With the use in view, clang externalizes it.
// rt.cpp keeps the list of binders (trap_register_binder) and runs them from trap_sink_install().
void trap_register_binder(void (*bind)(Control *));
[[maybe_unused]] static __host__ void trap_bind_this_tu(Control *c) {
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_trap_ctl), &c, sizeof c));
}
}  // namespace mk

#if !defined(__HIP_DEVICE_COMPILE__) && !defined(HIP_TRAP_NO_BINDER)
namespace mk {
namespace {
struct TrapBinderRegistration {
  TrapBinderRegistration() { trap_register_binder(&trap_bind_this_tu); }
};
[[maybe_unused]] const TrapBinderRegistration g_trap_binder_registration;
}  // namespace
}  // namespace mk
#endif
