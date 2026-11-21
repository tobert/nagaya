//! The weights of the text model on the card, and their load into an upload sink.
//!
//! The matrices stay `bf16` as the checkpoint holds them, and the queries of the residuals and the router bias stay
//! `f32`. Each expert matrix is one slab of `repack::expert_slab`. Some tensors share one allocation, so that one op
//! reads them (engine-abi "The first Inkling 2 binding", "The ring builder"). A count sink gives the bytes with no
//! card, and the open compares them with the plan (`spec::weights_bytes_of`).

use nagaya_mk_bmg::safetensors::SafeTensorsView;

use crate::formats;
use crate::host::ops::{EXPERTS, HIDDEN, INTERMEDIATE};
use crate::host::weights::{CYCLE_END, CYCLE_START, LAYERS, LOGITS, NVFP4_LAYERS, is_full};
use crate::repack::expert_slab;
use crate::spec::{self, HEAD_ROWS, SINK_ROWS, rounded};

/// A sink of the weights: device memory of the card, or a count of its bytes for the plan.
pub trait Upload {
    /// Device memory of `bytes` bytes, filled with zeros.
    fn alloc(&mut self, bytes: usize) -> u64;
    /// Copy `data` to the device at `at`.
    fn write(&mut self, at: u64, data: &[u8]);
    /// True when `write` copies. A count copies nothing, so the caller need not read or repack the data.
    fn writes(&self) -> bool;
}

/// The count of the device bytes of an upload, with no card.
#[derive(Default)]
pub struct Count {
    pub bytes: usize,
}

impl Upload for Count {
    fn alloc(&mut self, bytes: usize) -> u64 {
        self.bytes += rounded(bytes);
        0
    }

    fn write(&mut self, _at: u64, _data: &[u8]) {}

    fn writes(&self) -> bool {
        false
    }
}

/// The weights of one layer on the card. A field that the layer does not have is 0.
#[derive(Clone, Copy, Debug, Default)]
pub struct Layer {
    /// The `f32` queries of the attention input: 1 row, or 3 on a full layer. Layer 0 has none.
    pub attn_queries: u64,
    /// The `f32` query of the expert input.
    pub mlp_query: u64,
    /// The norms of the query, key and value inputs. A sliding layer names its one norm three times.
    pub norms: [u64; 3],
    pub mlp_norm: u64,
    /// `[3072][2304]`.
    pub wq: u64,
    /// The relative states: `[384][2304]`.
    pub wr: u64,
    /// The sinks: `[32][2304]`, the last 8 rows zero.
    pub sink: u64,
    /// The output gate: `[3072][2304]`.
    pub gate_attn: u64,
    /// The key and value projections: `[1024][2304]`, then on a full layer the 32 rows of the taps.
    pub wk: u64,
    pub wv: u64,
    /// The output projection: `[2304][3072]`.
    pub wo: u64,
    /// The scales of the clips of the query and of the relative states, `bf16`.
    pub q_scale: u64,
    pub rel_scale: u64,
    /// The 16 profiles of the distance, `bf16 [16][1024]`.
    pub rel_proj: u64,
    /// The key and value convolutions: `bf16 [1024][4]` on a sliding layer, `[4][1024]` by lag on a full layer.
    pub k_conv: u64,
    pub v_conv: u64,
    /// The convolutions after the attention and after the experts, `bf16 [2304][4]`.
    pub attn_conv: u64,
    pub mlp_conv: u64,
    /// The router `bf16 [128][2304]` and its `f32` selection bias.
    pub router: u64,
    pub router_bias: u64,
    /// The message gates of a cycle layer: query, key, value, expert input. Each is down, forget and gain.
    pub gates: [u64; 4],
    /// The slabs of the gate and up rows and of the down rows.
    pub w13: u64,
    pub w2: u64,
    /// True for the MXFP8 experts of layers 20 to 23.
    pub mxfp8: bool,
}

/// The weights of the text model on the card.
#[derive(Clone, Debug)]
pub struct Weights {
    /// The embedding table: `bf16 [200058][2304]`.
    pub embed: u64,
    pub embed_norm: u64,
    /// The head: `bf16 [200064][2304]`, the last 6 rows zero.
    pub head: u64,
    pub norm: u64,
    /// The `f32` query of the output residual.
    pub output_query: u64,
    /// The `f32` message queries: `[12][4][2304]`, query, key, value, expert input.
    pub message_queries: u64,
    pub layers: Vec<Layer>,
}

fn llm(name: &str) -> String {
    format!("model.llm.{name}")
}

/// The tensors and the sink of one load.
struct Loader<'s, 'v> {
    sink: &'s mut dyn Upload,
    view: &'s SafeTensorsView<'v>,
}

impl<'v> Loader<'_, 'v> {
    fn bytes(&self, name: &str) -> &'v [u8] {
        self.view
            .tensor_bytes(name)
            .unwrap_or_else(|| panic!("the plan found the tensor {name}, and the view has none"))
    }

    /// One allocation of at least `total` bytes that holds the first `used[i]` bytes of each tensor `names[i]`, one
    /// after the other, then zeros. `None` takes the whole tensor.
    fn put(&mut self, names: &[(String, Option<usize>)], total: usize) -> u64 {
        let parts: Vec<&[u8]> = names
            .iter()
            .map(|(n, used)| {
                let b = self.bytes(n);
                &b[..used.unwrap_or(b.len())]
            })
            .collect();
        let used: usize = parts.iter().map(|p| p.len()).sum();
        assert!(
            used <= total,
            "an allocation of {total} bytes for {used} bytes of tensors"
        );
        let at = self.sink.alloc(total);
        if self.sink.writes() {
            let mut joined = Vec::with_capacity(used);
            for part in parts {
                joined.extend_from_slice(part);
            }
            self.sink.write(at, &joined);
        }
        at
    }

    /// One tensor in its own allocation.
    fn one(&mut self, name: &str) -> u64 {
        let len = self.bytes(name).len();
        self.put(&[(name.to_owned(), None)], len)
    }

    /// The tensors `names` in one allocation.
    fn joined(&mut self, names: &[String]) -> u64 {
        let list: Vec<(String, Option<usize>)> = names.iter().map(|n| (n.clone(), None)).collect();
        let total = list.iter().map(|(n, _)| self.bytes(n).len()).sum();
        self.put(&list, total)
    }

    /// The `f32` values of a tensor of `f32` bytes.
    fn f32s(&self, name: &str) -> Vec<f32> {
        self.bytes(name)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    /// The slab of one expert matrix of layer `l`: `w13` with `gate_up`, else `w2`.
    fn slab(&mut self, l: usize, gate_up: bool) -> Result<u64, String> {
        let w = if gate_up { "w13" } else { "w2" };
        let p = |name: &str| llm(&format!("layers.{l}.mlp.experts.{w}_{name}"));
        let (rows, cols) = if gate_up {
            (2 * INTERMEDIATE, HIDDEN)
        } else {
            (HIDDEN, INTERMEDIATE)
        };
        let mx = l >= NVFP4_LAYERS;
        let at = self.sink.alloc(spec::expert_slab_bytes(EXPERTS, rows, cols, mx));
        if self.sink.writes() {
            let factors = (!mx).then(|| {
                let den = self.f32s(&p("weight_outer_scale_denominator"))[0];
                self.f32s(&p("weight_amax"))
                    .iter()
                    .map(|a| formats::reference_expert_factor(*a, den))
                    .collect::<Vec<f32>>()
            });
            let slab = expert_slab(
                self.bytes(&p("weight")),
                self.bytes(&p("weight_scale")),
                factors.as_deref(),
                EXPERTS,
                rows,
                cols,
                mx,
                gate_up,
            )
            .map_err(|e| format!("layer {l}: the experts {w} do not repack: {e:?}"))?;
            self.sink.write(at, &slab);
        }
        Ok(at)
    }

    fn layer(&mut self, l: usize) -> Result<Layer, String> {
        let p = |name: &str| llm(&format!("layers.{l}.{name}"));
        let full = is_full(l);
        let norms = if full {
            [
                self.one(&p("q_input_norm.weight")),
                self.one(&p("k_input_norm.weight")),
                self.one(&p("v_input_norm.weight")),
            ]
        } else {
            let at = self.one(&p("attn_norm.weight"));
            [at; 3]
        };
        let (wk, wv, k_conv, v_conv) = if full {
            (
                self.joined(&[p("attn.wk_dv.weight"), p("attn.k_sconv.weight_proj")]),
                self.joined(&[p("attn.wv_dv.weight"), p("attn.v_sconv.weight_proj")]),
                self.one(&p("attn.k_sconv.static_w")),
                self.one(&p("attn.v_sconv.static_w")),
            )
        } else {
            (
                self.one(&p("attn.wk_dv.weight")),
                self.one(&p("attn.wv_dv.weight")),
                self.one(&p("attn.k_sconv.weight")),
                self.one(&p("attn.v_sconv.weight")),
            )
        };
        let gates = if (CYCLE_START..CYCLE_END).contains(&l) {
            ["attn", "k", "v", "mlp"].map(|g| {
                self.joined(&[
                    p(&format!("controllers.{g}.down_weight")),
                    p(&format!("controllers.{g}.forget_weight")),
                    p(&format!("controllers.{g}.gain_weight")),
                ])
            })
        } else {
            [0; 4]
        };
        Ok(Layer {
            attn_queries: if l == 0 {
                0
            } else {
                self.one(&llm(&format!("attn_res.attn_queries.{}", l - 1)))
            },
            mlp_query: self.one(&llm(&format!("attn_res.mlp_queries.{l}"))),
            norms,
            mlp_norm: self.one(&p("mlp_norm.weight")),
            wq: self.one(&p("attn.wq_du.weight")),
            wr: self.one(&p("attn.wr_du.weight")),
            sink: self.put(&[(p("attn.attn_sink_proj.weight"), None)], SINK_ROWS * HIDDEN * 2),
            gate_attn: self.one(&p("attn.gate_attn.weight")),
            wk,
            wv,
            wo: self.one(&p("attn.wo_ud.weight")),
            q_scale: self.one(&p("attn.q_l2_clip.scale")),
            rel_scale: self.one(&p("attn.rel_l2_clip.scale")),
            rel_proj: self.one(&p("attn.rel_logits_proj.proj")),
            k_conv,
            v_conv,
            attn_conv: self.one(&p("attn_sconv.weight")),
            mlp_conv: self.one(&p("mlp_sconv.weight")),
            router: self.one(&p("mlp.gate.weight")),
            router_bias: self.one(&p("mlp.gate.bias")),
            gates,
            w13: self.slab(l, true)?,
            w2: self.slab(l, false)?,
            mxfp8: l >= NVFP4_LAYERS,
        })
    }
}

/// Load the text model of `view` into `sink`. A checkpoint that the plan refuses uploads nothing.
pub fn load(sink: &mut dyn Upload, view: &SafeTensorsView<'_>) -> Result<Weights, String> {
    spec::weights_bytes_of(&view.tensors)?;
    let mut ld = Loader { sink, view };
    let rows = LOGITS * HIDDEN * 2;
    let embed = ld.put(&[(llm("embed.weight"), Some(rows))], rows);
    let head = ld.put(&[(llm("unembed.weight"), Some(rows))], HEAD_ROWS * HIDDEN * 2);
    let embed_norm = ld.one(&llm("embed_norm.weight"));
    let norm = ld.one(&llm("norm.weight"));
    let output_query = ld.one(&llm("attn_res.output_query"));
    // The message queries in the order of `host::weights::merge_message_queries`.
    let message_queries = {
        let q = ld.bytes(&llm("middle_cycle_cross_pass_queries"));
        let kv = ld.bytes(&llm("middle_cycle_cross_pass_kv_queries"));
        let row = HIDDEN * 4;
        let at = ld.sink.alloc(q.len() + kv.len());
        if ld.sink.writes() {
            let mut merged = Vec::with_capacity(q.len() + kv.len());
            for l in 0..CYCLE_END - CYCLE_START {
                for (t, r) in [(q, 2 * l), (kv, 2 * l), (kv, 2 * l + 1), (q, 2 * l + 1)] {
                    merged.extend_from_slice(&t[r * row..(r + 1) * row]);
                }
            }
            ld.sink.write(at, &merged);
        }
        at
    };
    let layers = (0..LAYERS).map(|l| ld.layer(l)).collect::<Result<Vec<_>, _>>()?;
    Ok(Weights {
        embed,
        embed_norm,
        head,
        norm,
        output_query,
        message_queries,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_count_rounds_each_allocation_to_64_bytes() {
        let mut c = Count::default();
        assert_eq!((c.alloc(1), c.alloc(64), c.alloc(65)), (0, 0, 0));
        assert_eq!(c.bytes, 64 + 64 + 128);
        assert!(!c.writes());
    }
}
