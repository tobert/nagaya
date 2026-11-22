// The MoE ops of one Inkling 2 run that experts.cuh and quant.cuh do not hold: OP_ROUTE.
//
// This file belongs to wave B lane M (MoE). Each op below is a stub that traps: the op is reserved and not
// built yet. The lane replaces the body of its own stubs and changes no other file of the interpreter. `run_op` in
// megakernel.cu calls each op.
//
// This header needs `mk_protocol.h`, `MkTaskDescriptor` and the worker functions of megakernel.cu.

#pragma once

// Reserved for wave B lane M (MoE).
__device__ __forceinline__ void op_route(const MkTaskDescriptor& task, uint32_t item) {
    (void)task;
    (void)item;
    __trap();
}
