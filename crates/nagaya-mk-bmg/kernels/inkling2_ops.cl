// The ops of the Inkling 2 binary (MK_INKLING2). Each op is one `else if (op == ...)` branch of the task loop of
// megakernel.cl, which includes this file, so each op sees the locals of the loop. The opcodes must match
// src/protocol.rs. The host form of each op is in crates/nagaya-mk-inkling2-bmg/src/host.
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
