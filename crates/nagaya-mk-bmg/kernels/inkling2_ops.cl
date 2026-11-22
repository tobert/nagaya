// The ops of the Inkling 2 binary (MK_INKLING2). Each op is one `else if (op == ...)` branch of the task loop of
// megakernel.cl, which includes this file, so each op sees the locals of the loop. The opcodes must match
// src/protocol.rs. The host form of each op is in crates/nagaya-mk-inkling2-bmg/src/host.
#include "inkling2_layer_ops.cl"
        else if (op == OP_EXPAND_NVFP4) {
            // c[row, col] = the bf16 bits of the NVFP4 planes a (codes) and b (scales) of m rows and k columns, rows
            // in the order of the planes. The factor of the expert is the f32 of p0. A check of the expansion: the
            // expert ops read the planes the same way. One subgroup for each (tile, step).
            __global const uint*  A = (__global const uint*)task.a;
            __global const uchar* B = (__global const uchar*)task.b;
            __global ushort*      C = (__global ushort*)task.c;
            uint steps = task.k / 16u, tiles = task.m / 16u;
            float factor = as_float(task.p0);
            for (uint item = gsg; item < tiles * steps; item += n_gsg) {
                uint tile = item / steps, step = item % steps;
                int8 v = inkling2_nvfp4_step(A, B, tile, steps, step, lane, factor);
                __global uint* row = (__global uint*)(C + ((ulong)tile * 16u + lane) * task.k + step * 16u);
                for (uint i = 0; i < 8; i++) row[i] = (uint)v[i];
            }
        }
        else if (op == OP_EXPAND_MXFP8) {
            // The same for the MXFP8 planes: c[row, col] = the bf16 bits of a (values) and b (scales).
            __global const uint*  A = (__global const uint*)task.a;
            __global const uchar* B = (__global const uchar*)task.b;
            __global ushort*      C = (__global ushort*)task.c;
            uint steps = task.k / 16u, tiles = task.m / 16u;
            for (uint item = gsg; item < tiles * steps; item += n_gsg) {
                uint tile = item / steps, step = item % steps;
                int8 v = inkling2_mxfp8_step(A, B, tile, steps, step, lane);
                __global uint* row = (__global uint*)(C + ((ulong)tile * 16u + lane) * task.k + step * 16u);
                for (uint i = 0; i < 8; i++) row[i] = (uint)v[i];
            }
        }
        else if (op == OP_EXPERT_UP) {
            // c[q, i] = swiglu(a[q] . gate[e, i], a[q] . up[e, i]) for each entry q of plan x, e its expert, and i < m;
            // b = the slab of the layer (inkling2_expert_step), k the columns, FLAG_MXFP8 its format. a holds
            // OP_MOE_GATHER's rows in plan order. A block of at most INKLING2_ROW_TOKENS tokens takes the row path,
            // and a larger block the XMX path. On each path, a row's bits do not depend on the other rows of the
            // block. The tiles 2t and 2t + 1 of an expert hold the gate and the up rows of i = 16t + lane.
            __global const float*   X = (__global const float*)task.a;
            __global const uchar*   slab = (__global const uchar*)task.b;
            __global float*         Y = (__global float*)task.c;
            __global const MoEPlan* P = (__global const MoEPlan*)task.x;
            bool mx = (flags & FLAG_MXFP8) != 0;
            uint n_active = P->n_active;
            load_plan(P, slm_active, slm_offset, lid, lsize);
            uint M = task.m, K = task.k, rows = 2u * M;
            uint n = slm_offset[NUM_EXPERTS];
            if (n <= INKLING2_ROW_TOKENS * TOP_K) {
                // The row path. A round gives each work-group num_sgs / parts tiles of 16 rows, and the parts
                // subgroups of a tile each take a fixed range of steps. A lane takes one row, for each entry of the
                // expert. The tiles 2t and 2t + 1 are in one work-group, and the first part of the gate tile joins
                // the parts of both in order, then the SwiGLU.
                uint parts = INKLING2_UP_PARTS, part = sg_id % parts, slot = sg_id / parts, per_wg = num_sgs / parts;
                uint steps = K / 16u, s_lo = part * steps / parts, s_hi = (part + 1u) * steps / parts;
                uint tiles = 2u * (M / 16u), all = n_active * tiles;
                for (uint base = 0; base < all; base += per_wg * num_wgs) {
                    uint tile = base + wg_id * per_wg + slot;
                    uint e = 0, t = 0, q0 = 0, qn = 0;
                    if (tile < all) {
                        e = slm_active[tile / tiles];
                        t = tile % tiles;
                        q0 = slm_offset[e];
                        qn = slm_offset[e + 1] - q0;
                    }
                    // The 256 floats of local memory of the subgroup: the table of a step, then the part of each entry.
                    __local float* own = (__local float*)slm_part + sg_id * 256u;
                    inkling2_part_dots(slab, e, rows, K, t, s_lo, s_hi, X + (ulong)q0 * K, qn, lane, mx, own);
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if (part == 0 && (t & 1u) == 0) {
                        for (uint j = 0; j < qn; j++) {
                            float g = 0.0f, u = 0.0f;
                            for (uint p = 0; p < parts; p++) {
                                g += ((__local float*)slm_part)[(sg_id + p) * 256u + 128u + j * 16u + lane];
                                u += ((__local float*)slm_part)[(sg_id + parts + p) * 256u + 128u + j * 16u + lane];
                            }
                            Y[(ulong)(q0 + j) * M + 16u * (t / 2u) + lane] = inkling2_swiglu(g, u);
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
            else {
                // The XMX path, for a block of more tokens.
                plan_chunks(slm_active, slm_offset, n_active, XMX_TOKENS2, slm_chunk, lid);
                for (uint item = gsg; item < slm_chunk[n_active] * (M / 16u); item += n_gsg) {
                    uint c = item / (M / 16u), t = item % (M / 16u);
                    uint i = chunk_expert(slm_chunk, n_active, c), e = slm_active[i];
                    uint t0 = slm_offset[e] + (c - slm_chunk[i]) * XMX_TOKENS2;
                    uint tn = min((uint)XMX_TOKENS2, slm_offset[e + 1] - t0);
                    float8 g[2], u[2];
                    inkling2_xmx_gate_up(slab, e, rows, K, 2u * t, 2u * t + 1u, X, t0, n, lane, mx, g, u);
                    for (uint mt = 0; mt < 2; mt++) {
                        for (uint r = 0; r < 8; r++) {
                            uint j = 8 * mt + r;
                            if (j < tn) Y[(ulong)(t0 + j) * M + 16u * t + lane] = inkling2_swiglu(g[mt][r], u[mt][r]);
                        }
                    }
                }
            }
        }
        else if (op == OP_EXPERT_DOWN) {
            // c[slot, r] = weight[q] * (a[q] . down[e, r]) for each entry q of plan x, slot its (token, pick), and r <
            // m; b = the slab of the layer, k the columns. The weight multiplies after the sum, as in the reference.
            // The paths are those of OP_EXPERT_UP.
            __global const float*   X = (__global const float*)task.a;
            __global const uchar*   slab = (__global const uchar*)task.b;
            __global float*         Y = (__global float*)task.c;
            __global const MoEPlan* P = (__global const MoEPlan*)task.x;
            bool mx = (flags & FLAG_MXFP8) != 0;
            uint n_active = P->n_active;
            load_plan(P, slm_active, slm_offset, lid, lsize);
            uint M = task.m, K = task.k, rows = M;
            uint n = slm_offset[NUM_EXPERTS];
            if (n <= INKLING2_ROW_TOKENS * TOP_K) {
                // The row path, as in OP_EXPERT_UP: the first part of each tile joins the parts in order.
                uint parts = INKLING2_DOWN_PARTS, part = sg_id % parts, slot = sg_id / parts, per_wg = num_sgs / parts;
                uint steps = K / 16u, s_lo = part * steps / parts, s_hi = (part + 1u) * steps / parts;
                uint tiles = M / 16u, all = n_active * tiles;
                for (uint base = 0; base < all; base += per_wg * num_wgs) {
                    uint tile = base + wg_id * per_wg + slot;
                    uint e = 0, t = 0, q0 = 0, qn = 0;
                    if (tile < all) {
                        e = slm_active[tile / tiles];
                        t = tile % tiles;
                        q0 = slm_offset[e];
                        qn = slm_offset[e + 1] - q0;
                    }
                    // The 256 floats of local memory of the subgroup: the table of a step, then the part of each entry.
                    __local float* own = (__local float*)slm_part + sg_id * 256u;
                    inkling2_part_dots(slab, e, rows, K, t, s_lo, s_hi, X + (ulong)q0 * K, qn, lane, mx, own);
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if (part == 0) {
                        for (uint j = 0; j < qn; j++) {
                            float s = 0.0f;
                            for (uint p = 0; p < parts; p++) s += ((__local float*)slm_part)[(sg_id + p) * 256u + 128u + j * 16u + lane];
                            uint q = q0 + j;
                            Y[(ulong)P->entry_slot[q] * M + 16u * t + lane] = P->entry_weight[q] * s;
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
            else {
                // The XMX path, for a block of more tokens.
                plan_chunks(slm_active, slm_offset, n_active, XMX_TOKENS, slm_chunk, lid);
                for (uint item = gsg; item < slm_chunk[n_active] * (M / 16u); item += n_gsg) {
                    uint c = item / (M / 16u), t = item % (M / 16u);
                    uint i = chunk_expert(slm_chunk, n_active, c), e = slm_active[i];
                    uint t0 = slm_offset[e] + (c - slm_chunk[i]) * XMX_TOKENS;
                    uint tn = min((uint)XMX_TOKENS, slm_offset[e + 1] - t0);
                    float8 acc[4];
                    inkling2_xmx_down(slab, e, rows, K, t, X, t0, n, lane, mx, acc);
                    for (uint mt = 0; mt < 4; mt++) {
                        for (uint r = 0; r < 8; r++) {
                            uint j = 8 * mt + r;
                            if (j < tn) {
                                uint q = t0 + j;
                                Y[(ulong)P->entry_slot[q] * M + 16u * t + lane] = P->entry_weight[q] * acc[mt][r];
                            }
                        }
                    }
                }
            }
        }
        else if (op == OP_EXPERT_SUM) {
            // c[t, r] = the sum of a[8 t + pick, r] over the 8 picks in order, for n tokens of m. The router lists
            // the picks of a token by expert index, so this is the order of the reference.
            __global const float* A = (__global const float*)task.a;
            __global float*       C = (__global float*)task.c;
            uint M = task.m;
            for (ulong i = (ulong)gsg * 16u + lane; i < (ulong)task.n * M; i += (ulong)n_gsg * 16u) {
                ulong t = i / M, r = i % M;
                float s = 0.0f;
                for (uint pick = 0; pick < TOP_K; pick++) s += A[(t * TOP_K + pick) * M + r];
                C[i] = s;
            }
        }
#include "inkling2_attn_ops.cl"
#include "inkling2_vision_ops.cl"
#include "inkling2_lab_ops.cl"
