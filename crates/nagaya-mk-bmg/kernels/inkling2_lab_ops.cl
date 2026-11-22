// The lab ops of the Inkling 2 binary (milestone 35 step 8): opcodes 190 to 199. inkling2_ops.cl includes this file
// inside the task loop of megakernel.cl, so each op sees the locals of the loop. The host form of each op is in
// crates/nagaya-mk-inkling2-bmg/src/host/ops.rs.

        else if (op == OP_INKLING2_CONTROL) {
            // One control item on the residual (engine-abi "Control vectors", "The lab of Inkling 2"), in place. It is
            // OP_CONTROL of the LFM3 binary under the opcode of this binary. a = n rows h of k values; b = the vector
            // v [k] of the layer; p0 = 1 add (h + s * v), 2 ablate (h - s * (h.u) * u), or 3 clamp (h + (s - h.u) *
            // u), with u = v / |v|; m = the bits of s. A work-group takes a row, and wg_sum adds the dot products, so
            // the bits of a row do not depend on the count of rows. A vector of zeros keeps the row.
            __global float* H = (__global float*)task.a;
            __global const float* V = (__global const float*)task.b;
            uint K = task.k, how = task.p0;
            float s = as_float(task.m);
            for (uint r = wg_id; r < task.n; r += num_wgs) {
                __global float* h = H + (ulong)r * K;
                float hv = 0.0f, vv = 0.0f;
                for (uint i = lid; i < K; i += lsize) {
                    hv = fma(h[i], V[i], hv);
                    vv = fma(V[i], V[i], vv);
                }
                hv = wg_sum(hv, slm_reduce, lane, sg_id, num_sgs);
                vv = wg_sum(vv, slm_reduce, lane, sg_id, num_sgs);
                if (vv == 0.0f) continue;
                float norm = sqrt(vv), along = hv / norm;
                float coef = how == 1 ? s : how == 2 ? -s * along / norm : (s - along) / norm;
                for (uint i = lid; i < K; i += lsize) h[i] = fma(coef, V[i], h[i]);
            }
        }
