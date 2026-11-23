//! GPU gate of the weights, the slot and the arena of the pass (milestone 37 step 10, commit 4). Each repacked weight
//! goes back through the inverse of its repack and must equal the bytes of the checkpoint, so the test shares no code
//! with the loader but the layout functions that have their own host tests.
//!
//! `INKLING2_MODEL_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-inkling2-sm121 --features cuda
//!  --test cuda_pass_load -- --test-threads=1`

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::experts::{ExpertFormat, unpack_expert, up_source_row};
use nagaya_mk_cuda::proj::{ProjLayout, SINKS, Segment, split};
use nagaya_mk_cuda::repack::{MatmulShape, unpack_bf16};
use nagaya_mk_cuda::rows::{gate_down_shape, gate_up_word_offset};
use nagaya_mk_cuda::{INKLING2_CUBIN, test_cards};
use nagaya_mk_inkling2_sm121::pass::arena::Arena;
use nagaya_mk_inkling2_sm121::pass::dev::Checkpoint;
use nagaya_mk_inkling2_sm121::pass::layout::{Buf, LOGIT_COLUMNS, StepLayout};
use nagaya_mk_inkling2_sm121::pass::slot::{CONV_BYTES, Slot, conv_offset};
use nagaya_mk_inkling2_sm121::pass::weights::DeviceWeights;
use nagaya_mk_inkling2_sm121::spec::{self, HIDDEN, INTERMEDIATE, KV_ROW, LAYERS};

fn model_dir() -> Option<PathBuf> {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    match std::env::var_os("INKLING2_MODEL_DIR") {
        Some(dir) if PathBuf::from(&dir).join(spec::WEIGHTS_FILE).exists() => Some(dir.into()),
        _ => {
            assert!(
                !require,
                "INKLING2_MODEL_DIR has no checkpoint and NAGAYA_REQUIRE_MODELS is set"
            );
            eprintln!("cuda_pass_load: skipped, INKLING2_MODEL_DIR has no checkpoint");
            None
        }
    }
}

#[test]
fn the_slot_and_the_arena_have_the_sizes_of_the_plan() {
    let dev = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU");
    let mut slot = Slot::new(&dev, 4096, 24, CacheFormat::F32).unwrap();
    assert_eq!(slot.runs.len(), 48);
    assert_eq!(slot.bytes() as u64, spec::slot_bytes(4096, 24, CacheFormat::F32));
    // A full run holds each position, and a sliding run holds the ring.
    assert_eq!(slot.runs[5].plane_rows, 4096);
    assert_eq!(slot.runs[1].plane_rows, spec::RING_ROWS);
    assert_eq!((slot.runs[5].window, slot.runs[1].window), (0, 1024));
    assert_eq!(slot.runs[30].layer, spec::state_layer(30));
    let run = &slot.runs[2];
    assert_eq!(run.conv.bytes(), CONV_BYTES);
    assert_eq!(run.conv_ptr(3) - run.conv.ptr(), conv_offset(3) as u64);
    // `zero` clears the bytes, the position and the poison.
    run.conv.write(0, &[0xAB; 64]).unwrap();
    slot.position = 17;
    slot.poisoned = true;
    slot.zero().unwrap();
    assert!(slot.runs[2].conv.read(0, 64).unwrap().iter().all(|b| *b == 0));
    assert_eq!((slot.position, slot.poisoned), (0, false));
    let arena = Arena::new(&dev, 16, 16).unwrap();
    let layout = StepLayout::new(16, 16);
    assert_eq!(arena.buffer.bytes(), layout.total_bytes());
    assert_eq!(arena.addr(Buf::Xw) - arena.buffer.ptr(), layout.offset(Buf::Xw) as u64);
    assert_eq!(arena.rf(Buf::Y) - arena.buffer.ptr(), layout.rf(Buf::Y) as u64);
    assert!(arena.buffer.read(0, 256).unwrap().iter().all(|b| *b == 0));
    assert_eq!(layout.place(Buf::Logits).bytes, 16 * LOGIT_COLUMNS * 4);
}

#[test]
fn every_uploaded_weight_goes_back_to_the_bytes_of_the_checkpoint() {
    let Some(dir) = model_dir() else { return };
    let dev = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU");
    let started = Instant::now();
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "cuda_pass_load: {} MB of weights in {:.1} s",
        weights.bytes() / 1_000_000,
        started.elapsed().as_secs_f64()
    );
    assert_eq!(weights.layers.len(), LAYERS);
    let mut ck = Checkpoint::open(&dir).unwrap();
    let llm = |name: &str| format!("model.llm.{name}");
    let l = |layer: usize, name: &str| format!("model.llm.layers.{layer}.{name}");

    // The embedding goes up as it is, and the head is the first rows of the unembedding, repacked.
    let embed_row = ck
        .range(
            &llm("embed.weight"),
            "BF16",
            &[201_024, HIDDEN],
            12_345 * HIDDEN * 2,
            HIDDEN * 2,
        )
        .unwrap();
    assert_eq!(weights.embed.read(12_345 * HIDDEN * 2, HIDDEN * 2).unwrap(), embed_row);
    let head_shape = MatmulShape::new(LOGIT_COLUMNS, HIDDEN).unwrap();
    let head = unpack_bf16(&weights.head.read(0, weights.head.bytes()).unwrap(), &head_shape).unwrap();
    for row in [0usize, 77, 200_057, 200_063] {
        let want = ck
            .range(
                &llm("unembed.weight"),
                "BF16",
                &[201_024, HIDDEN],
                row * HIDDEN * 2,
                HIDDEN * 2,
            )
            .unwrap();
        assert_eq!(
            head[row * HIDDEN * 2..(row + 1) * HIDDEN * 2],
            want[..],
            "head row {row}"
        );
    }

    for layer in [0usize, 5, 7, 11, 21] {
        let w = &weights.layers[layer];
        let full = spec::is_full(layer);
        assert_eq!(w.full, full);
        // J: each part comes back as the checkpoint tensor.
        let layout = ProjLayout { full };
        let joined = unpack_bf16(&w.j.read(0, w.j.bytes()).unwrap(), &layout.shape()).unwrap();
        for (segment, bytes) in split(&layout, &joined).unwrap() {
            let (name, shape): (String, Vec<usize>) = match segment {
                Segment::Q => ("attn.wq_du.weight".into(), vec![3072, HIDDEN]),
                Segment::Rel => ("attn.wr_du.weight".into(), vec![384, HIDDEN]),
                Segment::Sink => ("attn.attn_sink_proj.weight".into(), vec![SINKS, HIDDEN]),
                Segment::Gate => ("attn.gate_attn.weight".into(), vec![3072, HIDDEN]),
                Segment::K => ("attn.wk_dv.weight".into(), vec![KV_ROW, HIDDEN]),
                Segment::V => ("attn.wv_dv.weight".into(), vec![KV_ROW, HIDDEN]),
                Segment::TapsK => ("attn.k_sconv.weight_proj".into(), vec![32, HIDDEN]),
                Segment::TapsV => ("attn.v_sconv.weight_proj".into(), vec![32, HIDDEN]),
            };
            assert_eq!(
                bytes,
                ck.bytes(&l(layer, &name), "BF16", &shape).unwrap(),
                "layer {layer} {segment:?}"
            );
        }
        // The output matrix and the router.
        let wo = unpack_bf16(
            &w.wo.read(0, w.wo.bytes()).unwrap(),
            &MatmulShape::new(HIDDEN, 3072).unwrap(),
        )
        .unwrap();
        assert_eq!(
            wo,
            ck.bytes(&l(layer, "attn.wo_ud.weight"), "BF16", &[HIDDEN, 3072])
                .unwrap(),
            "layer {layer} wo"
        );
        let router = unpack_bf16(
            &w.router.read(0, w.router.bytes()).unwrap(),
            &MatmulShape::new(128, HIDDEN).unwrap(),
        )
        .unwrap();
        assert_eq!(
            router,
            ck.bytes(&l(layer, "mlp.gate.weight"), "BF16", &[128, HIDDEN]).unwrap(),
            "layer {layer} router"
        );
        // The small tensors.
        let kconv = if full {
            ck.bytes(&l(layer, "attn.k_sconv.static_w"), "BF16", &[4, KV_ROW])
                .unwrap()
        } else {
            ck.bytes(&l(layer, "attn.k_sconv.weight"), "BF16", &[KV_ROW, 1, 4])
                .unwrap()
        };
        assert_eq!(
            w.k_conv.read(0, w.k_conv.bytes()).unwrap(),
            kconv,
            "layer {layer} k conv"
        );
        assert_eq!(w.input_norms.len(), if full { 3 } else { 1 });
        let mlp_norm = ck.f32s(&l(layer, "mlp_norm.weight"), &[HIDDEN]).unwrap();
        let got: Vec<f32> = w
            .mlp_norm
            .read(0, HIDDEN * 4)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        assert_eq!(got, mlp_norm, "layer {layer} mlp norm");
        assert_eq!(w.attn_queries.is_some(), layer > 0);
        assert_eq!(w.gate_down.is_some(), (6..18).contains(&layer));
    }

    // The gates of a cycle layer: the 4 down matrices one after the other, and the words of forget and gain.
    let w = &weights.layers[8];
    let down = w.gate_down.as_ref().unwrap();
    let shape = gate_down_shape();
    for (g, name) in ["attn", "k", "v", "mlp"].iter().enumerate() {
        let one = unpack_bf16(
            &down.read(g * shape.packed_bytes(), shape.packed_bytes()).unwrap(),
            &shape,
        )
        .unwrap();
        assert_eq!(
            one,
            ck.bytes(
                &l(8, &format!("controllers.{name}.down_weight")),
                "BF16",
                &[64, 2 * HIDDEN]
            )
            .unwrap()
        );
        let forget = ck
            .bytes(
                &l(8, &format!("controllers.{name}.forget_weight")),
                "BF16",
                &[HIDDEN, 64],
            )
            .unwrap();
        let gain = ck
            .bytes(&l(8, &format!("controllers.{name}.gain_weight")), "BF16", &[HIDDEN, 64])
            .unwrap();
        let up = w.gate_up.as_ref().unwrap();
        let per_gate = 36 * 16384;
        for (channel, kk) in [(0usize, 0usize), (1000, 22), (2303, 62)] {
            let src = (channel * 64 + kk) * 2;
            let f = up
                .read(g * per_gate + gate_up_word_offset(channel, kk, false), 4)
                .unwrap();
            let n = up
                .read(g * per_gate + gate_up_word_offset(channel, kk, true), 4)
                .unwrap();
            assert_eq!(f, forget[src..src + 4], "gate {name} forget {channel},{kk}");
            assert_eq!(n, gain[src..src + 4], "gate {name} gain {channel},{kk}");
        }
    }

    // The experts: an NVFP4 layer and an MXFP8 layer. Up rows are in the order of `up_source_row`.
    for (layer, format) in [(3usize, ExpertFormat::Nvfp4), (21, ExpertFormat::Mxfp8)] {
        let bank = &weights.layers[layer].experts;
        assert_eq!(bank.format, format);
        assert_eq!(bank.up_amax.is_some(), format == ExpertFormat::Nvfp4);
        let (code_dtype, scale_dtype) = match format {
            ExpertFormat::Nvfp4 => ("U8", "F8_E4M3"),
            ExpertFormat::Mxfp8 => ("F8_E4M3", "U8"),
        };
        for expert in [0usize, 64, 127] {
            for (which, n, k, up) in [
                ("w13", 2 * INTERMEDIATE, HIDDEN, true),
                ("w2", HIDDEN, INTERMEDIATE, false),
            ] {
                let (code_row, scale_row) = (format.code_bytes(k), format.scale_bytes(k));
                let item = format.item_bytes(k) * (n / 64);
                let buffer = if up { &bank.up } else { &bank.down };
                let packed = buffer.read(expert * item, item).unwrap();
                let (codes, scales) = unpack_expert(format, n, k, &packed).unwrap();
                let want_codes = ck
                    .range(
                        &l(layer, &format!("mlp.experts.{which}_weight")),
                        code_dtype,
                        &[128, n, code_row],
                        expert * n * code_row,
                        n * code_row,
                    )
                    .unwrap();
                let want_scales = ck
                    .range(
                        &l(layer, &format!("mlp.experts.{which}_weight_scale")),
                        scale_dtype,
                        &[128, n, scale_row],
                        expert * n * scale_row,
                        n * scale_row,
                    )
                    .unwrap();
                for p in 0..n {
                    let from = if up { up_source_row(p) } else { p };
                    assert_eq!(
                        codes[p * code_row..(p + 1) * code_row],
                        want_codes[from * code_row..(from + 1) * code_row],
                        "layer {layer} {which} expert {expert} codes row {p}"
                    );
                    assert_eq!(
                        scales[p * scale_row..(p + 1) * scale_row],
                        want_scales[from * scale_row..(from + 1) * scale_row],
                        "layer {layer} {which} expert {expert} scales row {p}"
                    );
                }
            }
        }
    }
    let nvfp4 = &weights.layers[3].experts;
    assert!(
        nvfp4.up_denominator != 1.0 && nvfp4.down_denominator > 0.0,
        "{} {}",
        nvfp4.up_denominator,
        nvfp4.down_denominator
    );
    assert_eq!(weights.layers[21].experts.up_denominator, 1.0);
    assert_eq!(weights.message_queries.len(), 12);
}
