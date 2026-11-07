use super::*;
use crate::spec::tests::{OMNI_CONFIG, TEXT_CONFIG};
use nagaya_mk_bmg::protocol::{
    FLAG_ACCUMULATE, OP_ATTN_SCORES, OP_BIAS, OP_COPY, OP_LAYERNORM, OP_MATMUL, OP_SHORT_CONV, OP_SUBSAMPLE,
};

pub fn omni() -> Encoder {
    Encoder::parse(OMNI_CONFIG)
        .expect("the config of LFM3-Omni-2B parses")
        .expect("Omni has an audio encoder")
}

fn edited(edit: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut value: serde_json::Value = serde_json::from_str(OMNI_CONFIG).unwrap();
    edit(&mut value);
    value.to_string()
}

/// An encoder of 2 layers, a conv layer and an attention layer, with small widths, and its config text.
fn small_config() -> String {
    edited(|c| {
        let a = &mut c["audio_encoder_config"];
        a["num_mel_bins"] = 16.into();
        a["conv_channels"] = 16.into();
        a["hidden_size"] = 32.into();
        let t = &mut a["text_config"];
        t["hidden_size"] = 32.into();
        t["num_attention_heads"] = 1.into();
        t["num_key_value_heads"] = 1.into();
        t["intermediate_size"] = 24.into();
        t["block_multiple_of"] = 16.into();
        t["layer_types"] = serde_json::json!(["conv", "full_attention"]);
        t["num_hidden_layers"] = 2.into();
        c["text_config"]["hidden_size"] = 48.into();
    })
}

fn small() -> Encoder {
    Encoder::parse(&small_config()).unwrap().unwrap()
}

#[test]
fn the_config_of_omni_gives_the_numbers_of_its_audio_encoder() {
    let e = omni();
    assert_eq!((e.mel, e.channels, e.hidden, e.out), (128, 256, 768, 2048));
    assert_eq!((e.heads, e.kv_heads, e.head_dim), (8, 4, 96));
    // Lfm2MLP: int(2 * 2560 / 3) = 1706, up to a multiple of 256.
    assert_eq!(e.intermediate, 1792);
    assert_eq!((e.taps, e.window, e.eps, e.projector_eps), (3, 32, 1e-5, 1e-5));
    assert_eq!(e.layers.len(), 17);
    let attention: Vec<usize> = (0..17).filter(|l| e.layers[*l] == AudioLayer::Attention).collect();
    assert_eq!(attention, vec![2, 5, 8, 11, 14, 15, 16]);
    assert_eq!(e.mel_out(), 16);
}

#[test]
fn the_text_model_has_no_audio_encoder() {
    assert_eq!(Encoder::parse(TEXT_CONFIG).unwrap(), None);
}

#[test]
fn a_config_that_the_tasks_cannot_run_is_refused_with_its_key() {
    type Edit = Box<dyn Fn(&mut serde_json::Value)>;
    let cases: [(&str, Edit); 6] = [
        (
            "audio_encoder_config.num_mel_bins",
            Box::new(|c| c["audio_encoder_config"]["num_mel_bins"] = 100.into()),
        ),
        (
            "audio_encoder_config.text_config.num_attention_heads",
            Box::new(|c| c["audio_encoder_config"]["text_config"]["num_attention_heads"] = 16.into()),
        ),
        (
            "audio_encoder_config.text_config.conv_bias",
            Box::new(|c| c["audio_encoder_config"]["text_config"]["conv_bias"] = true.into()),
        ),
        (
            "audio_encoder_config.text_config.rope_parameters",
            Box::new(|c| c["audio_encoder_config"]["text_config"]["rope_parameters"]["rope_theta"] = 1e4.into()),
        ),
        (
            "audio_encoder_config.text_config.layer_types",
            Box::new(|c| c["audio_encoder_config"]["text_config"]["layer_types"][0] = "sliding_attention".into()),
        ),
        (
            "audio_encoder_config.text_config.hidden_size",
            Box::new(|c| c["audio_encoder_config"]["text_config"]["hidden_size"] = 1024.into()),
        ),
    ];
    for (key, edit) in cases {
        let error = Encoder::parse(&edited(|c| edit(c))).unwrap_err();
        assert!(error.contains(key), "{key}: {error}");
    }
}

/// The checkpoint header of LFM3-Omni-2B at `21d75123` has 176 tensors of the audio encoder and the projector, with
/// 115,278,144 parameters (read on 2026-11-06).
#[test]
fn the_tensors_are_those_of_the_checkpoint() {
    let all = tensors(&omni());
    assert_eq!(all.len(), 176);
    let params: usize = all.iter().map(|(_, s)| s.iter().product::<usize>()).sum();
    assert_eq!(params, 115_278_144);
}

/// A checkpoint of the tensors of `e`, each value a different BF16 number, after `edit` changed the list.
fn checkpoint(e: &Encoder, edit: impl FnOnce(&mut Vec<(String, Vec<usize>)>)) -> Vec<u8> {
    let mut list = tensors(e);
    edit(&mut list);
    let data: Vec<Vec<u8>> = list
        .iter()
        .enumerate()
        .map(|(t, (_, shape))| {
            (0..shape.iter().product::<usize>())
                .flat_map(|i| {
                    let v = (t * 1000 + i) as f32 * 0.25;
                    ((v.to_bits() >> 16) as u16).to_le_bytes()
                })
                .collect()
        })
        .collect();
    let entries: Vec<(&str, &str, &[usize], &[u8])> = list
        .iter()
        .zip(&data)
        .map(|((name, shape), bytes)| (name.as_str(), "BF16", shape.as_slice(), bytes.as_slice()))
        .collect();
    SafeTensorsView::build_synthetic_safetensors(&entries)
}

/// Host memory that records the bytes of each upload.
#[derive(Default)]
struct Mem {
    bytes: Vec<u8>,
}

impl Upload for Mem {
    fn alloc(&mut self, bytes: usize) -> u64 {
        let at = self.bytes.len() as u64;
        self.bytes.resize(self.bytes.len() + rounded(bytes), 0);
        at
    }

    fn write(&mut self, at: u64, data: &[u8]) {
        self.bytes[at as usize..at as usize + data.len()].copy_from_slice(data);
    }

    fn writes(&self) -> bool {
        true
    }
}

#[test]
fn a_checkpoint_with_a_lost_or_other_tensor_is_refused() {
    let e = small();
    let whole = checkpoint(&e, |_| {});
    check(&e, &SafeTensorsView::parse(&whole).unwrap()).unwrap();
    let lost = checkpoint(&e, |list| list.retain(|(n, _)| !n.ends_with("q_layernorm.weight")));
    let error = check(&e, &SafeTensorsView::parse(&lost).unwrap()).unwrap_err();
    assert!(
        error.contains("no tensor model.audio_encoder.lfm.layers.1.self_attn.q_layernorm.weight"),
        "{error}"
    );
    let other = checkpoint(&e, |list| {
        list.push(("model.audio_projector.linear_3.weight".to_owned(), vec![16]))
    });
    let error = check(&e, &SafeTensorsView::parse(&other).unwrap()).unwrap_err();
    assert!(error.contains("model.audio_projector.linear_3.weight"), "{error}");
    let shape = checkpoint(&e, |list| {
        let w = list.iter_mut().find(|(n, _)| n.ends_with("lin_out.weight")).unwrap();
        w.1 = vec![32, 16, 1];
    });
    let error = check(&e, &SafeTensorsView::parse(&shape).unwrap()).unwrap_err();
    assert!(
        error.contains("lin_out.weight") && error.contains("[32, 32, 1]"),
        "{error}"
    );
    let mut mem = Mem::default();
    assert!(load(&mut mem, &e, &SafeTensorsView::parse(&shape).unwrap()).is_err());
    assert!(mem.bytes.is_empty(), "a refused checkpoint uploads nothing");
}

#[test]
fn the_load_turns_lin_out_to_the_plane_widens_the_layernorm_and_joins_the_mlp() {
    let e = small();
    let file = checkpoint(&e, |_| {});
    let view = SafeTensorsView::parse(&file).unwrap();
    let mut mem = Mem::default();
    let w = load(&mut mem, &e, &view).unwrap();
    let bf16_at = |at: u64, i: usize| -> u16 {
        let o = at as usize + 2 * i;
        u16::from_le_bytes([mem.bytes[o], mem.bytes[o + 1]])
    };
    let source = |name: &str, i: usize| -> u16 {
        let b = view.tensor_bytes(name).unwrap();
        u16::from_le_bytes([b[2 * i], b[2 * i + 1]])
    };
    // lin_out: [32][16 channels * 2 mel rows]. Column c * 2 + h of the reference is column h * 16 + c here.
    let lin_out = "model.audio_encoder.downsample.lin_out.weight";
    for r in [0, 31] {
        for c in [0, 5, 15] {
            for h in [0, 1] {
                assert_eq!(
                    bf16_at(w.lin_out.0, r * 32 + h * 16 + c),
                    source(lin_out, r * 32 + c * 2 + h),
                    "row {r}, channel {c}, mel row {h}"
                );
            }
        }
    }
    let ln = "model.audio_projector.layer_norm.bias";
    for i in [0, 31] {
        let o = w.projector_norm.1 as usize + 4 * i;
        let got = f32::from_le_bytes(mem.bytes[o..o + 4].try_into().unwrap());
        assert_eq!(got.to_bits(), (source(ln, i) as u32) << 16, "layer_norm.bias {i}");
    }
    let (w1, w3) = (
        "model.audio_encoder.lfm.layers.0.feed_forward.w1.weight",
        "model.audio_encoder.lfm.layers.0.feed_forward.w3.weight",
    );
    let rows = e.intermediate * e.hidden;
    assert_eq!(bf16_at(w.layers[0].w13, 7), source(w1, 7));
    assert_eq!(bf16_at(w.layers[0].w13, rows + 7), source(w3, 7));
    assert_eq!(w.layers[0].q, 0, "a conv layer has no attention weights");
    assert_ne!(w.layers[1].k_norm, 0);
}

fn weights(e: &Encoder) -> AudioWeights {
    let mut next = 0x1000u64;
    let mut a = || {
        next += 0x100;
        next
    };
    AudioWeights {
        conv: [(a(), a()), (a(), a()), (a(), a())],
        lin: [(a(), a()), (a(), a())],
        lin_out: (a(), a()),
        layers: e
            .layers
            .iter()
            .map(|_| LayerWeights {
                operator_norm: a(),
                ffn_norm: a(),
                w13: a(),
                w2: a(),
                in_proj: a(),
                conv: a(),
                out_proj: a(),
                q: a(),
                k: a(),
                v: a(),
                o: a(),
                q_norm: a(),
                k_norm: a(),
            })
            .collect(),
        norm: a(),
        projector_norm: (a(), a()),
        linear_1: (a(), a()),
        linear_2: (a(), a()),
    }
}

fn bufs(e: &Encoder, positions: usize) -> AudioBufs {
    let mut next = 1u64 << 40;
    AudioBufs::new(
        &mut |bytes| {
            let at = next;
            next += rounded(bytes) as u64;
            at
        },
        e,
        positions,
    )
}

fn count(t: &[MkTaskDescriptor], op: u16) -> usize {
    t.iter().filter(|task| task.opcode == op).count()
}

const OUT: u64 = 1 << 50;

#[test]
fn the_tasks_of_the_oracle_clip_run_the_downsampler_the_layers_and_the_projector() {
    let e = omni();
    let (w, b) = (weights(&e), bufs(&e, MAX_CLIP_POSITIONS));
    let clip = Clip {
        mel: b.mel,
        frames: 149,
        out: OUT,
    };
    let t = encode(&e, &w, &b, clip, Stage::Rows);
    assert_eq!(plane_shapes(&e, 149), [(149, 128), (75, 64), (38, 32), (19, 16)]);
    let sub: Vec<&MkTaskDescriptor> = t.iter().filter(|x| x.opcode == OP_SUBSAMPLE).collect();
    assert_eq!(sub.len(), 3);
    assert_eq!(
        (sub[0].a, sub[0].n, sub[0].k, sub[0].p0, sub[0].layer),
        (b.mel, 149, 128, 1, 1)
    );
    assert_eq!((sub[1].n, sub[1].k, sub[1].p0, sub[1].layer), (75, 64, 256, 0));
    assert_eq!((sub[2].n, sub[2].k, sub[2].p0, sub[2].layer), (38, 32, 256, 0));
    assert_eq!((count(&t, OP_SHORT_CONV), count(&t, OP_COPY)), (10, 10));
    assert_eq!(count(&t, OP_ATTN_SCORES), 7);
    // Each attention layer reads the window of 32 keys, and its value plane follows the most positions.
    for task in t.iter().filter(|x| x.opcode == OP_ATTN_SCORES) {
        assert_eq!((task.n, task.k, task.x), (19, 96, (512u64 << 32) | 32));
        assert_eq!(task.m, 8 | 4 << 16);
    }
    // lin_mid, lin_end, lin_out; 10 conv layers of 3; 7 attention layers of 5; 2 in the projector.
    assert_eq!(count(&t, OP_MATMUL), 3 + 10 * 3 + 7 * 5 + 2);
    let lin_out = t.iter().find(|x| x.opcode == OP_MATMUL && x.k == 4096).unwrap();
    assert_eq!((lin_out.n, lin_out.m, lin_out.c), (19, 768, b.x));
    assert_eq!(count(&t, OP_LAYERNORM), 1);
    let last = t.last().unwrap();
    assert_eq!((last.opcode, last.c, last.n, last.k), (OP_BIAS, OUT, 19, 2048));
    // The residual adds each out_proj and each w2: 17 + 17.
    let added = t
        .iter()
        .filter(|x| x.opcode == OP_MATMUL && x.flags & FLAG_ACCUMULATE != 0)
        .count();
    assert_eq!(added, 34);
}

#[test]
fn each_stage_is_a_prefix_of_the_tasks_of_the_clip() {
    let e = omni();
    let (w, b) = (weights(&e), bufs(&e, MAX_CLIP_POSITIONS));
    let clip = Clip {
        mel: b.mel,
        frames: 149,
        out: OUT,
    };
    let rows = encode(&e, &w, &b, clip, Stage::Rows);
    let down = encode(&e, &w, &b, clip, Stage::Downsample);
    let enc = encode(&e, &w, &b, clip, Stage::Encoder);
    assert!(down.len() < enc.len() && enc.len() < rows.len());
    for (part, name) in [(&down, "downsample"), (&enc, "encoder")] {
        for (i, (a, x)) in part.iter().zip(&rows).enumerate() {
            assert_eq!((a.opcode, a.a, a.c, a.n), (x.opcode, x.a, x.c, x.n), "{name}: task {i}");
        }
    }
    assert_eq!(down.last().unwrap().c, b.x);
    assert_eq!(enc.last().unwrap().c, b.h);
}

#[test]
fn a_clip_of_one_frame_has_one_position_and_the_longest_clip_fits_the_buffers() {
    let e = omni();
    let (w, b) = (weights(&e), bufs(&e, MAX_CLIP_POSITIONS));
    for (frames, positions) in [(1, 1), (8, 1), (9, 2), (MAX_CLIP_FRAMES, MAX_CLIP_POSITIONS)] {
        let t = encode(
            &e,
            &w,
            &b,
            Clip {
                mel: b.mel,
                frames,
                out: OUT,
            },
            Stage::Rows,
        );
        assert_eq!(t.last().unwrap().n as usize, positions, "{frames} frames");
        assert!(t.len() < MAX_TASKS_PER_CARD);
    }
}

#[test]
#[should_panic(expected = "a clip of 4097 frames takes 513 positions, and the buffers hold 1 to 512 positions")]
fn a_clip_past_the_buffers_is_refused() {
    let e = omni();
    let (w, b) = (weights(&e), bufs(&e, MAX_CLIP_POSITIONS));
    encode(
        &e,
        &w,
        &b,
        Clip {
            mel: b.mel,
            frames: MAX_CLIP_FRAMES + 1,
            out: OUT,
        },
        Stage::Rows,
    );
}

#[test]
#[should_panic(expected = "a clip of 0 frames")]
fn a_clip_of_no_frames_is_refused() {
    let e = omni();
    let (w, b) = (weights(&e), bufs(&e, MAX_CLIP_POSITIONS));
    encode(
        &e,
        &w,
        &b,
        Clip {
            mel: b.mel,
            frames: 0,
            out: OUT,
        },
        Stage::Rows,
    );
}

/// Each buffer holds the longest clip: the plane after conv_in is the largest, 2,048 by 64 by 256 values.
#[test]
fn the_buffers_hold_the_longest_clip() {
    let e = omni();
    let b = bufs(&e, MAX_CLIP_POSITIONS);
    assert_eq!(b.plane_b - b.plane_a, (2048 * 64 * 256 * 4) as u64);
    assert_eq!(b.x - b.plane_b, (1024 * 32 * 256 * 4) as u64);
    assert_eq!(b.plane_a - b.mel, (4096 * 128 * 4) as u64);
    let total = bufs_bytes(&e, MAX_CLIP_POSITIONS);
    assert_eq!(b.rows - b.zeros, rounded((e.taps - 1) * e.hidden * 4) as u64);
    // The rows of the clip for the backbone come last: 512 positions of 2,048 values.
    assert_eq!(
        total as u64,
        b.rows + rounded(MAX_CLIP_POSITIONS * e.out * 4) as u64 - b.mel
    );
    assert_eq!(e.out, 2048);
}
