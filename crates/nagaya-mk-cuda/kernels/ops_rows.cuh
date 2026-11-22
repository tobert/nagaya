// The row ops of one Inkling 2 run: embedding, mixes and message gates (OP_EMBED, OP_MIX, OP_GATE_DOWN, OP_GATE_UP, OP_MIX_OUT).
//
// This file belongs to wave B lane R (rows). Each op below is a stub that traps: the op is reserved and not
// built yet. The lane replaces the body of its own stubs and changes no other file of the interpreter. `run_op` in
// megakernel.cu calls each op.
//
// This header needs `mk_protocol.h`, `MkTaskDescriptor` and the worker functions of megakernel.cu.

#pragma once

// Reserved for wave B lane R (rows).
__device__ __forceinline__ void op_embed(const MkTaskDescriptor& task, uint32_t item) {
    (void)task;
    (void)item;
    __trap();
}

// Reserved for wave B lane R (rows).
__device__ __forceinline__ void op_mix(const MkTaskDescriptor& task, uint32_t item) {
    (void)task;
    (void)item;
    __trap();
}

// Reserved for wave B lane R (rows).
__device__ __forceinline__ void op_gate_down(const MkTaskDescriptor& task, uint32_t item) {
    (void)task;
    (void)item;
    __trap();
}

// Reserved for wave B lane R (rows).
__device__ __forceinline__ void op_gate_up(const MkTaskDescriptor& task, uint32_t item) {
    (void)task;
    (void)item;
    __trap();
}

// Reserved for wave B lane R (rows).
__device__ __forceinline__ void op_mix_out(const MkTaskDescriptor& task, uint32_t item) {
    (void)task;
    (void)item;
    __trap();
}
