// The attention op of one Inkling 2 run: OP_ATTN.
//
// This file belongs to wave B lane A (attention). Each op below is a stub that traps: the op is reserved and not
// built yet. The lane replaces the body of its own stubs and changes no other file of the interpreter. `run_op` in
// megakernel.cu calls each op.
//
// This header needs `mk_protocol.h`, `MkTaskDescriptor`, `MkTaskTable` and the worker functions of megakernel.cu.

#pragma once

// Reserved for wave B lane A (attention).
__device__ __forceinline__ void op_attn(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    (void)task;
    (void)table;
    (void)item;
    __trap();
}
