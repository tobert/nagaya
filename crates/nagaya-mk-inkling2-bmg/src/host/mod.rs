//! The `f32` host form of the Inkling 2 text model: one op for each layer part, and a forward that composes them
//! over the real checkpoint.
//!
//! It fixes the meaning of each op and each checkpoint tensor before the device code exists. Each device op is gated
//! against the host form of its op. The reference is transformers `inkling2_loop` at `046bdb8cc2`
//! (`modeling_inkling2_loop.py`); each op cites its lines. The oracle sums in `f32` with the experts expanded to
//! `bf16` (`tools/inkling2_reference.py`), and so does this form.
//!
//! # The names of the checkpoint
//!
//! The engine reads the names of the checkpoint (`model.llm.*`), not the names of transformers. The map, from
//! `conversion_mapping.py` (`inkling2_text` and `inkling2_loop_text`) and `core_model_loading.py`:
//!
//! | Checkpoint, under `model.llm.` | transformers | Use |
//! |---|---|---|
//! | `embed.weight`, `embed_norm.weight` | `embed_tokens`, `embed_tokens.embed_norm` | the embedding and its RMS norm |
//! | `unembed.weight` [201024, 2304] | `lm_head.weight` | the head; only the first 200,058 rows (`Inkling2UnpadVocab`) |
//! | `norm.weight` | `norm` | the final norm |
//! | `attn_res.output_query` | `final_input.query` | the query of the output residual |
//! | `attn_res.attn_queries.N` | `layers.{N+1}.attention_input.query` | the attention input of layer N + 1; [3, 2304] on a full layer |
//! | `attn_res.mlp_queries.N` | `layers.N.mlp_input.query` | the expert input of layer N |
//! | `middle_cycle_cross_pass_queries`, `..._kv_queries` [24, 2304] | `message_input.query` [12, 4, 2304] | the message queries (`merge_message_queries`) |
//! | `layers.L.attn_norm` | `input_layernorm` | the norm of the attention input of a sliding layer |
//! | `layers.L.{q,k,v}_input_norm` | `{q,k,v}_input_layernorm` | the three norms of a full layer (5, 11, 17, 23) |
//! | `layers.L.mlp_norm` | `post_attention_layernorm` | the norm of the expert input |
//! | `layers.L.attn.wq_du`, `wk_dv`, `wv_dv` | `q_proj.linear`, `k_proj.linear`, `v_proj.linear` | the projections |
//! | `layers.L.attn.q_l2_clip.scale` | `q_proj.norm.scale` | the scale of the query before its clip |
//! | `layers.L.attn.k_sconv.weight`, `v_sconv.weight` [1024, 1, 4] | `k_proj.sconv.conv1d.weight` | the fixed key and value convolutions, sliding layers |
//! | `layers.L.attn.k_sconv.static_w` [4, 1024], `weight_proj` [32, 2304] | `k_proj.sconv.static_weight`, `weight_proj` | the dynamic convolutions, full layers |
//! | `layers.L.attn.wr_du` [384, 2304] | `relative_position_bias.linear` | the relative states |
//! | `layers.L.attn.rel_l2_clip.scale` [24, 16] | `relative_position_bias.norm.scale` | the scale of the relative states |
//! | `layers.L.attn.rel_logits_proj.proj` [16, 1024] | `relative_position_bias.proj` | the 16 profiles of the distance |
//! | `layers.L.attn.attn_sink_proj` | `sink_proj` | the sink of each head |
//! | `layers.L.attn.gate_attn` | `o_proj.gate_proj` | the output gate |
//! | `layers.L.attn.wo_ud` | `o_proj.linear` | the output projection |
//! | `layers.L.attn_sconv.weight`, `mlp_sconv.weight` | `post_attention_convolution`, `post_mlp_convolution` | the convolutions after the attention and after the experts |
//! | `layers.L.mlp.gate.weight`, `mlp.gate.bias` | `mlp.gate.weight`, `e_score_correction_bias` | the router and its selection bias |
//! | `layers.L.mlp.experts.w13_*`, `w2_*` | `gate_up_proj`, `down_proj` | the experts (`crate::formats`); row 2i of `w13` is gate row i, row 2i + 1 up row i |
//! | `layers.L.controllers.attn` | `q_message_gate` | the message gate of the query, layers 6 to 17 |
//! | `layers.L.controllers.{k,v,mlp}` | `{k,v,mlp}_message_gate` | the other message gates |
//! | `controllers.*.down_weight` [64, 4608], `forget_weight`, `gain_weight` [2304, 64] | `down_proj`, `forget_gate_proj`, `message_gate_proj` | the parts of a gate |
//!
//! # The state of a context
//!
//! Each run holds its keys and values after the clip of the key and the norm of the value, and the 3 last inputs of
//! its four convolutions (engine-abi "Inkling 2", "The parts"). A sliding run holds the last 1,024 positions.

pub mod forward;
pub mod ops;
pub mod weights;
