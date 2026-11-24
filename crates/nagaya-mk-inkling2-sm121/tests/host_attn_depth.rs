//! The host model of the merge order of `OP_ATTN` at depth (`long_support/merge_model.rs`): the tree and the
//! double-float merges stay under the bound at every depth. The serial f32 merge is printed and not asserted: it drifts.
#![cfg(feature = "cuda")]

mod long_support;

use long_support::merge_model::*;

use long_support::*;
use nagaya_mk_cuda::attn::{self, GROUP, HEAD_DIM, HEADS, KV_HEADS, REL_EXTENT};
use nagaya_mk_inkling2_bmg::host::ops;

#[test]
fn host_model_of_the_order() {
    let Some(layer) = layer(5) else { return };
    let pool = make_pool(layer, attn::CacheFormat::F32);
    let width = HEADS * HEAD_DIM;
    // leak the pool so closures can hold 'static slices
    let pool: &'static Pool = Box::leak(Box::new(pool));
    let depths: Vec<usize> = std::env::var("DEPTHS")
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![4096, 65_536, 131_072, 409_600, 524_288]);
    for positions in depths {
        let last = positions - 1;
        let input = make_rows(layer, last, 1, 53);
        let host = host_row(layer, pool, &input, 0, last, 0);
        let exact = exact_row(layer, pool, &input, 0, last, 0);
        let rel = &input.rel[..HEADS * attn::REL_STATES];
        let mut profiles = vec![0f32; HEADS * REL_EXTENT];
        ops::relative_logits(rel, &layer.proj, &mut profiles);
        let scaling = (HEAD_DIM as f64).powf(-0.5);
        const NV: usize = 8;
        let per_head = parallel(HEADS, |h| {
            let kvh = h / GROUP;
            let profile = &profiles[h * REL_EXTENT..(h + 1) * REL_EXTENT];
            let q = &input.q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
            let sink = input.sinks[h];
            let dots: Vec<f32> = (0..POOL)
                .map(|jm| {
                    let k = &pool.keys[jm * KV_HEADS + kvh];
                    let d: f64 = q.iter().zip(k).map(|(a, b)| *a as f64 * *b as f64).sum();
                    (d * scaling) as f32
                })
                .collect();
            let scores: Vec<f32> = (0..=last)
                .map(|j| dots[j % POOL] + ops::relative_bias(profile, (last - j) as i64))
                .collect();
            let vals = |j: usize| -> &'static [f32] { &pool.values[(j % POOL) * KV_HEADS + kvh][..] };
            let nchunks = last / CHUNK + 1;
            let mut p32 = Vec::new();
            let mut p64 = Vec::new();
            for c in 0..nchunks {
                let lo = c * CHUNK;
                let hi = (lo + CHUNK - 1).min(last);
                p32.push(chunk_f32(&scores[lo..=hi], &vals, lo));
                p64.push(chunk_f64(&scores[lo..=hi], &vals, lo));
            }
            let sink_st = St {
                m: sink,
                l: 1.0,
                o: vec![0.0; HEAD_DIM],
            };
            let serial = |ps: &Vec<St>| {
                let mut st = sink_st.clone();
                for p in ps {
                    st = combine(&st, p);
                }
                st
            };
            let wide = |ps: &Vec<St>| {
                let mut st = (sink as f64, 1f64, vec![0f64; HEAD_DIM]);
                for p in ps {
                    st = combine64(&st, p);
                }
                St {
                    m: st.0 as f32,
                    l: st.1 as f32,
                    o: st.2.iter().map(|v| *v as f32).collect(),
                }
            };
            let tr = |ps: &Vec<St>| {
                let mut leaves = vec![sink_st.clone()];
                leaves.extend(ps.iter().cloned());
                tree(leaves)
            };
            let outs: [St; NV] = [
                wide(&p64),                // 0: chunk f64, merge f64 (rounded partials)
                wide(&p32),                // 1: chunk f32, merge f64
                serial(&p64),              // 2: chunk f64, merge f32 serial
                serial(&p32),              // 3: kernel model
                tr(&p64),                  // 4: chunk f64, tree
                tr(&p32),                  // 5: chunk f32, tree
                serial_df(&sink_st, &p64), // 6
                serial_df(&sink_st, &p32), // 7
            ];
            let g = &input.gate[h * HEAD_DIM..];
            outs.map(|s| {
                s.o.iter()
                    .zip(g)
                    .map(|(o, g)| (o / s.l) * ops::sigmoid(*g))
                    .collect::<Vec<f32>>()
            })
        });
        let names = [
            "c64+m64",
            "c32+m64",
            "c64+serial32",
            "c32+serial32 (kernel)",
            "c64+tree32",
            "c32+tree32",
            "c64+serial df",
            "c32+serial df",
        ];
        eprintln!(
            "model depth {positions}: host-vs-f64 {:.2e}",
            worst_error(&host, &exact).0
        );
        for v in 0..NV {
            let mut out = Vec::with_capacity(width);
            for head in &per_head {
                out.extend_from_slice(&head[v]);
            }
            let vs_exact = worst_error(&out, &exact).0;
            eprintln!(
                "model depth {positions}: {:24} vs f64 {vs_exact:.2e}, vs host {:.2e}",
                names[v],
                worst_error(&out, &host).0
            );
            // The merges that the kernel can use stay under the bound at every depth.
            if v >= 4 {
                assert!(
                    vs_exact <= ACCURACY_BOUND,
                    "{} is {vs_exact:e} from the f64 form",
                    names[v]
                );
            }
        }
    }
}
