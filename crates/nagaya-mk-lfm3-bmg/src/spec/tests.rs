use super::*;
use nagaya_abi::OutputsWanted;

/// `config.json` of `LFM3-1.7B` at `0dc4268b`, and of `LFM3-Omni-2B` at `21d75123`.
pub const TEXT_CONFIG: &str = include_str!("../../tests/fixtures/lfm3.config.json");
pub const OMNI_CONFIG: &str = include_str!("../../tests/fixtures/lfm3-omni.config.json");

/// The backbone of `LFM3-1.7B`.
pub fn backbone() -> Backbone {
    Backbone::parse(TEXT_CONFIG).expect("the config of LFM3-1.7B parses")
}

fn edited(text: &str, edit: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut value: serde_json::Value = serde_json::from_str(text).unwrap();
    edit(&mut value);
    value.to_string()
}

#[test]
fn the_config_of_lfm3_gives_the_backbone_of_the_milestone() {
    let b = backbone();
    assert_eq!(b.model, Model::Text);
    assert_eq!((b.hidden, b.heads, b.kv_heads, b.head_dim), (2048, 16, 4, 128));
    assert_eq!(
        (b.intermediate, b.vocab_rows, b.window, b.taps),
        (7680, 128_000, 256, 7)
    );
    assert_eq!(b.eps, 1e-5);
    assert_eq!(b.layers.len(), 24);
    assert_eq!(b.layers_of(LayerKind::Conv).count(), 13);
    assert_eq!(
        b.layers_of(LayerKind::Full).collect::<Vec<_>>(),
        vec![2, 6, 10, 14, 18, 22]
    );
    assert_eq!(
        b.layers_of(LayerKind::Sliding).collect::<Vec<_>>(),
        vec![4, 8, 12, 16, 20]
    );
    // Full attention turns half of each head (partial_rotary_factor 0.5); sliding attention turns all of it.
    assert_eq!(b.full_rope, Rope { theta: 1e6, pairs: 32 });
    assert_eq!(b.sliding_rope, Rope { theta: 1e4, pairs: 64 });
    assert_eq!((b.kv_row(), b.conv_state()), (512, 6 * 2048));
}

#[test]
fn omni_reads_its_text_config_and_has_the_same_backbone() {
    let omni = Backbone::parse(OMNI_CONFIG).unwrap();
    assert_eq!(omni.model, Model::Omni);
    assert_eq!(omni.model.prefix(), "model.language_model.");
    assert_eq!(omni.model.name(), "lfm3-omni");
    assert!(omni.audio.is_some() && backbone().audio.is_none());
    assert_eq!(
        Backbone {
            model: Model::Text,
            audio: None,
            ..omni
        },
        backbone()
    );
    assert_eq!(backbone().model.name(), "lfm3");
}

/// The engine of Omni holds the buffers of the audio encoder for the longest clip, in the term `work`.
#[test]
fn the_work_of_omni_holds_the_buffers_of_the_audio_encoder() {
    let omni = Backbone::parse(OMNI_CONFIG).unwrap();
    let c = config(1);
    let work = |b: &Backbone| {
        let l = limits(b, &c, 1, None);
        l.memory.mandatory.iter().find(|t| t.name == "work").unwrap().bytes
    };
    let e = omni.audio.clone().unwrap();
    let without = Backbone {
        audio: None,
        ..omni.clone()
    };
    assert_eq!(
        work(&omni) - work(&without),
        crate::audio::bufs_bytes(&e, crate::audio::MAX_CLIP_POSITIONS) as u64
    );
}

#[test]
fn a_value_that_the_kernels_cannot_run_is_refused_with_its_key() {
    type Edit = Box<dyn Fn(&mut serde_json::Value)>;
    let cases: Vec<(&str, Edit)> = vec![
        ("head_dim", Box::new(|c| c["head_dim"] = 64.into())),
        ("num_attention_heads", Box::new(|c| c["num_attention_heads"] = 8.into())),
        ("conv_L_cache", Box::new(|c| c["conv_L_cache"] = 9.into())),
        ("conv_bias", Box::new(|c| c["conv_bias"] = true.into())),
        ("use_sandwich_norm", Box::new(|c| c["use_sandwich_norm"] = false.into())),
        (
            "tie_word_embeddings",
            Box::new(|c| c["tie_word_embeddings"] = false.into()),
        ),
        ("sliding_window", Box::new(|c| c["sliding_window"] = 4096.into())),
        (
            "rope_parameters.full_attention",
            Box::new(|c| c["rope_parameters"]["full_attention"]["rope_theta"] = 5e5.into()),
        ),
        (
            "rope_parameters.sliding_attention",
            Box::new(|c| c["rope_parameters"]["sliding_attention"]["rope_type"] = "yarn".into()),
        ),
        ("layer_types", Box::new(|c| c["layer_types"][3] = "mamba".into())),
        ("num_hidden_layers", Box::new(|c| c["num_hidden_layers"] = 23.into())),
        (
            "architectures",
            Box::new(|c| c["architectures"][0] = "LlamaForCausalLM".into()),
        ),
        ("depth_config", Box::new(|c| c["depth_config"] = serde_json::json!({}))),
    ];
    for (key, edit) in cases {
        let error = Backbone::parse(&edited(TEXT_CONFIG, edit)).unwrap_err();
        assert!(error.contains(key), "{key}: {error}");
    }
    let error = Backbone::parse(&edited(OMNI_CONFIG, |c| c["text_config"]["head_dim"] = 64.into())).unwrap_err();
    assert!(error.contains("text_config.head_dim"), "{error}");
}

#[test]
fn the_parts_follow_the_layer_types() {
    let p = parts(&backbone());
    assert_eq!(p.len(), 13 + 2 * 11);
    let find = |name: &str| {
        p.iter()
            .find(|part| part.name == name)
            .unwrap_or_else(|| panic!("{name}"))
    };
    assert_eq!(
        (find("conv.00").kind, find("conv.00").shape.clone()),
        (PartKind::Fixed, vec![6, 2048])
    );
    assert_eq!(
        (find("k.02").kind, find("k.02").shape.clone()),
        (PartKind::Cells, vec![4, 128])
    );
    assert_eq!(find("v.22").kind, PartKind::Cells);
    assert_eq!(find("k.04").kind, PartKind::Ring);
    assert_eq!(find("v.20").kind, PartKind::Ring);
    assert!(p.iter().all(|part| part.dtype == Dtype::F32 && part.stage == 0));
    assert_eq!(p[0].name, "conv.00");
    assert_eq!((p[2].name.as_str(), p[3].name.as_str()), ("k.02", "v.02"));
}

#[test]
fn a_slot_counts_its_planes_and_its_conv_states() {
    let b = backbone();
    // A slot of 2,048 positions keeps every position in each attention layer.
    assert_eq!(slot_bytes(&b, 2048), 11 * 2 * 2048 * 2048 + 13 * 49_152);
    // A longer slot keeps rings of 2,048 rows and a spill of 1,024 in the 5 sliding layers.
    assert_eq!(slot_bytes(&b, 32_768), 805_306_368 + 62_914_560 + 638_976);
    assert_eq!(slot_bytes(&b, 32_768), 868_859_904);
    assert_eq!((ring_of(2048), ring_of(2049)), (0, 2048));
    assert_eq!(plane_rows(4096, LayerKind::Full), 4096);
    assert_eq!(plane_rows(4096, LayerKind::Sliding), 3072);
}

fn config(slots: usize) -> Config {
    Config {
        model_dir: PathBuf::from("/lfm3"),
        slots,
        max_block: 512,
    }
}

#[test]
fn the_limits_keep_the_feed_rule_and_the_terms() {
    let b = backbone();
    let c = config(3);
    let l = limits(&b, &c, 3_456_000_000, None);
    l.check().unwrap();
    assert_eq!(l.feed, Some(FeedRule { block: 512 }));
    // Decode segments share a pass by D11 rule 6, so `segments_exact` stays false (engine-abi "Batched passes").
    assert!(!l.segments_exact && l.checkpoint_grain.is_none());
    assert_eq!(l.determinism, Determinism::Always);
    assert_eq!((l.max_segments, l.max_pass_inputs, l.max_logit_rows), (8, 512, 16));
    assert_eq!(l.inputs, Inputs::Tokens);
    assert_eq!(l.devices, vec![0]);
    assert!(l.stages.is_empty());
    assert_eq!(
        l.slots,
        vec![SlotLimit {
            bytes: 868_859_904,
            count: 3,
            tokens: 32_768
        }]
    );
    let term = |name: &str| l.memory.mandatory.iter().find(|t| t.name == name).unwrap().clone();
    assert_eq!(term("weights").bytes, 3_456_000_000);
    assert_eq!(term("kv").bytes, 3 * 868_859_904);
    assert_eq!(term("kv").tier, Tier::Device(0));
    assert_eq!(term("pinned").tier, Tier::Host);
    // The scores of a block of 512 over 32,768 keys are most of the work: 16 × 512 × 32,784 floats.
    assert!(term("work").bytes > 16 * 512 * 32_784 * 4, "{}", term("work").bytes);
}

#[test]
fn a_config_outside_the_sizes_is_refused() {
    assert!(config(1).check().is_ok());
    assert_eq!(config(2).seqs(), vec![32_768, 32_768]);
    assert!(config(0).check().unwrap_err().contains("slots"));
    let mut c = config(1);
    c.max_block = 513;
    assert!(c.check().unwrap_err().contains("max_block"));
    c.max_block = 0;
    assert!(c.check().is_err());
}

fn segment(slot: u32, from: u64, tokens: &[u32], logits: &[u64]) -> Segment {
    Segment {
        control: Vec::new(),
        state: State::Slot(slot),
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (from..from + tokens.len() as u64).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted {
            logit_rows: logits.to_vec(),
        },
    }
}

const VOCAB: usize = 125_033;

#[test]
fn a_pass_runs_on_from_the_token_count_and_never_goes_back() {
    let b = backbone();
    let mut held = Held::new(&b, &[1024, 4096]);
    let block = held.check_pass(&[segment(0, 0, &[1, 2, 3], &[2])], 512, VOCAB).unwrap();
    assert_eq!((block.slot, block.pos0, block.logit_rows.clone()), (0, 0, vec![2]));
    held.ran(&block);
    assert_eq!(held.n[0], 3);
    assert!(held.check_pass(&[segment(0, 3, &[4], &[3])], 512, VOCAB).is_ok());
    let back = held.check_pass(&[segment(0, 2, &[4], &[])], 512, VOCAB).unwrap_err();
    assert!(back.contains("does not go back"), "{back}");
    let gap = held.check_pass(&[segment(0, 4, &[4], &[])], 512, VOCAB).unwrap_err();
    assert!(gap.contains("run from 3"), "{gap}");
    for (segments, why) in [
        (vec![], "0 segments"),
        (vec![segment(0, 3, &[1], &[]), segment(1, 0, &[1], &[])], "2 segments"),
        (vec![segment(2, 0, &[1], &[])], "slots"),
        (vec![segment(0, 3, &[], &[])], "0 tokens"),
        (vec![segment(0, 3, &[1; 513], &[])], "513 tokens"),
        (vec![segment(0, 3, &[125_033], &[])], "125033"),
        (vec![segment(0, 3, &[1], &[4])], "logit row at 4"),
        (
            vec![segment(0, 3, &[1; 17], &(3..20).collect::<Vec<_>>())],
            "17 logit rows",
        ),
        (vec![segment(1, 0, &[1; 512], &[])], ""),
    ] {
        let got = held.check_pass(&segments, 512, VOCAB);
        if why.is_empty() {
            assert!(got.is_ok(), "{got:?}");
        } else {
            let error = got.unwrap_err();
            assert!(error.contains(why), "{why}: {error}");
        }
    }
    let mut full = Held::new(&b, &[8]);
    full.n[0] = 6;
    assert!(
        full.check_pass(&[segment(0, 6, &[1, 2, 3], &[])], 512, VOCAB)
            .unwrap_err()
            .contains("do not fit")
    );
    let mut meta = segment(0, 3, &[1], &[]);
    meta.meta = vec![0; 16];
    assert!(held.check_pass(&[meta], 512, VOCAB).unwrap_err().contains("meta"));
}

/// The part ids of `conv.00`, `k.02` (cells) and `k.04` (a ring).
const CONV: PartId = 0;
const CELLS: PartId = 2;
const RING: PartId = 5;

#[test]
fn a_ring_holds_its_last_rows_and_a_short_slot_holds_every_position() {
    let b = backbone();
    assert_eq!(part_list(&b)[RING], Part::K(4));
    let mut held = Held::new(&b, &[2048, 4096]);
    held.n = vec![1000, 3000];
    assert_eq!(held.base(0, 4), 0);
    assert_eq!(held.base(1, 4), 3000 - 2048);
    assert_eq!(held.base(1, 2), 0);
    let row = vec![0u8; 512 * 4];
    let span = held.coverage(State::Slot(1), RING, &row.repeat(2048));
    assert_eq!((span.start, span.end), (952, 3000));
    assert_eq!(held.coverage(State::Slot(1), CONV, &[0; 64]).end, 0);
    let fit = |slot, part, start, end| held.fits(slot, &CoverageSpan { part, start, end });
    assert!(fit(1, RING, 2744, 3000).is_ok());
    assert!(fit(1, RING, 0, 3000).is_ok());
    assert!(fit(1, RING, 2745, 3000).unwrap_err().contains("window"));
    assert!(fit(0, RING, 1, 1000).unwrap_err().contains("from 0"));
    assert!(fit(1, CELLS, 2744, 3000).unwrap_err().contains("from 0"));
    assert!(fit(0, CELLS, 0, 2049).unwrap_err().contains("do not fit"));
    assert!(fit(0, CONV, 0, 0).unwrap_err().contains("conv state"));
}

#[test]
fn a_restore_writes_every_part_and_set_n_takes_the_count_only_then() {
    let b = backbone();
    let mut held = Held::new(&b, &[4096]);
    held.n[0] = 50;
    let state = State::Slot(0);
    let row = 512 * 4;
    // With no write, a count stays or goes to 0, and never goes back to another count.
    assert!(held.check_set_n(state, 50).is_ok());
    assert!(held.check_set_n(state, 0).is_ok());
    assert!(held.check_set_n(state, 49).unwrap_err().contains("does not go back"));
    // A cells part starts at 0; a ring part of a long slot keeps the last 2,048 rows of a write.
    assert!(
        held.check_cells_write(state, CELLS, 1, 1, row)
            .unwrap_err()
            .contains("from 0")
    );
    let (_, _, kept) = held.check_cells_write(state, RING, 0, 3000, 3000 * row).unwrap();
    assert_eq!(kept, 952..3000);
    assert!(held.check_cells_write(state, RING, 0, 2, row).is_err());
    assert!(
        held.check_cells_write(state, CONV, 0, 1, row)
            .unwrap_err()
            .contains("fixed part")
    );
    assert!(
        held.check_part_write(state, CELLS, 49_152)
            .unwrap_err()
            .contains("row part")
    );
    assert!(held.check_part_write(state, CONV, 49_151).is_err());
    let list = part_list(&b);
    for (id, part) in list.iter().enumerate() {
        if matches!(part, Part::Conv(_)) {
            held.check_part_write(state, id, 49_152).unwrap();
            held.wrote(0, id, None);
        } else if id != RING {
            let start = if b.layers[part.layer()] == LayerKind::Sliding {
                300 - 256
            } else {
                0
            };
            held.wrote(0, id, Some(start..300));
        }
    }
    assert_eq!(held.n[0], 0, "a write empties the slot until set_n");
    assert!(
        held.check_set_n(state, 300)
            .unwrap_err()
            .contains(&format!("part {RING} got no bytes"))
    );
    held.wrote(0, RING, Some(45..300));
    assert!(held.check_set_n(state, 300).unwrap_err().contains("window"));
    held.wrote(0, RING, Some(0..299));
    assert!(held.check_set_n(state, 300).unwrap_err().contains("up to 299"));
    held.wrote(0, RING, Some(44..300));
    assert_eq!(held.check_set_n(state, 300), Ok(0));
    held.set_n(0, 300);
    assert_eq!(held.n[0], 300);
    assert!(held.check_set_n(state, 299).is_err(), "set_n took the writes");
    // The rings hold no position before the restored rows, and a later pass moves the base on.
    assert_eq!((held.base(0, 4), held.base(0, 2)), (44, 0));
    held.n[0] = 3000;
    assert_eq!(held.base(0, 4), 952);
    held.clear(0);
    held.n[0] = 100;
    assert_eq!(held.base(0, 4), 0, "a clear forgets the restored start");
}

#[test]
fn a_read_stays_inside_the_rows_that_a_part_holds() {
    let b = backbone();
    let mut held = Held::new(&b, &[4096]);
    held.n[0] = 3000;
    let state = State::Slot(0);
    let row = 512 * 4;
    assert!(held.check_read(state, RING, 2744, 256, 256 * row).is_ok());
    assert!(held.check_read(state, RING, 952, 2048, 2048 * row).is_ok());
    assert!(held.check_read(state, RING, 951, 1, row).unwrap_err().contains("952"));
    assert!(held.check_read(state, CELLS, 0, 3001, 3001 * row).is_err());
    assert!(held.check_read(state, CELLS, 0, 2, row).is_err());
    assert_eq!(held.check_part_read(state, RING, 256 * row).unwrap().2, 952);
    assert!(held.check_part_read(state, CONV, 49_152).is_ok());
    assert!(held.check_part_read(state, CONV, row).is_err());
    assert!(held.check_part_read(State::Slot(1), CONV, 49_152).is_err());
}

/// The held state of Omni, with the placeholders of its config.
fn omni_held(seqs: &[usize]) -> Held {
    let mut held = Held::new(&Backbone::parse(OMNI_CONFIG).unwrap(), seqs);
    held.media = crate::vision::media_rules(OMNI_CONFIG).unwrap();
    held
}

/// A segment of `tokens` and the rows of `spans`, with their meta.
fn media_segment(from: u64, tokens: &[u32], spans: &[crate::media::Span]) -> Segment {
    let bytes = spans.iter().map(|s| s.bytes()).sum();
    Segment {
        control: Vec::new(),
        inputs: SegmentInputs::TokensAndRows {
            tokens: tokens.to_vec(),
            rows: vec![0; bytes],
        },
        meta: crate::media::encode(spans),
        ..segment(0, from, tokens, &[from + tokens.len() as u64 - 1])
    }
}

#[test]
fn omni_takes_the_rows_of_an_image_and_names_them_in_its_block() {
    use crate::media::{Span, SpanKind};
    let rules = crate::vision::media_rules(OMNI_CONFIG).unwrap();
    assert_eq!((rules.image_token, rules.audio_token), (Some(125_030), Some(125_031)));
    assert!(!rules.audio_encoder);
    let held = omni_held(&[4096]);
    let image = Span {
        kind: SpanKind::Image { grid: [1, 14, 22] },
        first: 5,
        count: 77,
    };
    let mut tokens = vec![1u32; 90];
    tokens[5..82].fill(125_030);
    let block = held
        .check_pass(&[media_segment(0, &tokens, &[image])], 512, VOCAB)
        .unwrap();
    assert_eq!(block.spans.len(), 1);
    assert_eq!((block.spans[0].take.clone(), block.spans[0].block_row), (0..77, 5));
    // A placeholder with no rows, the rows with no meta, and the meta with no rows are refused.
    let plain = held.check_pass(&[segment(0, 0, &tokens, &[])], 512, VOCAB).unwrap_err();
    assert!(
        plain.contains("position 5 holds the media placeholder 125030"),
        "{plain}"
    );
    let mut no_meta = media_segment(0, &tokens, &[image]);
    no_meta.meta.clear();
    let e = held.check_pass(&[no_meta], 512, VOCAB).unwrap_err();
    assert!(e.contains("rows and no meta"), "{e}");
    let mut no_rows = segment(0, 0, &tokens, &[]);
    no_rows.meta = crate::media::encode(&[image]);
    let e = held.check_pass(&[no_rows], 512, VOCAB).unwrap_err();
    assert!(e.contains("meta and no rows"), "{e}");
    let rows_only = Segment {
        inputs: SegmentInputs::Rows(vec![0; 64]),
        ..segment(0, 0, &tokens, &[])
    };
    let e = held.check_pass(&[rows_only], 512, VOCAB).unwrap_err();
    assert!(e.contains("rows and no tokens"), "{e}");
    // A clip is refused until the audio encoder exists.
    let clip = Span {
        kind: SpanKind::Audio { frames: 16 },
        first: 0,
        count: 2,
    };
    let e = held
        .check_pass(&[media_segment(0, &[125_031, 125_031, 1], &[clip])], 512, VOCAB)
        .unwrap_err();
    assert!(e.contains("no audio encoder yet"), "{e}");
}

#[test]
fn the_text_model_refuses_an_image_and_omni_plans_its_tower() {
    use crate::media::{Span, SpanKind};
    let image = Span {
        kind: SpanKind::Image { grid: [1, 2, 2] },
        first: 0,
        count: 1,
    };
    let held = Held::new(&backbone(), &[4096]);
    let e = held
        .check_pass(&[media_segment(0, &[125_030, 1], &[image])], 512, VOCAB)
        .unwrap_err();
    assert!(e.contains("no vision tower"), "{e}");
    let (b, c) = (backbone(), config(1));
    let text = limits(&b, &c, 1000, None);
    assert_eq!(text.inputs, Inputs::Tokens);
    assert!(text.memory.mandatory.iter().all(|t| t.name != "vision"));
    let omni = limits(
        &b,
        &c,
        1000,
        Some(VisionBytes {
            weights: 64_000,
            work: 7_000,
        }),
    );
    omni.check().unwrap();
    assert_eq!(omni.inputs, Inputs::TokensAndRows);
    let term = |l: &Limits, name: &str| l.memory.mandatory.iter().find(|t| t.name == name).unwrap().clone();
    assert_eq!(term(&omni, "vision").bytes, 64_000);
    assert_eq!(term(&omni, "vision").tier, Tier::Device(0));
    assert_eq!(term(&omni, "work").bytes, term(&text, "work").bytes + 7_000);
}

#[test]
fn both_models_declare_a_control_table_of_8_vectors_and_the_residual_tap() {
    let (b, c) = (backbone(), config(1));
    let l = limits(&b, &c, 1000, None);
    l.check().unwrap();
    assert!(l.capabilities.contains(&Capability::Control));
    assert_eq!(
        l.control,
        Some(ControlLimit {
            tables: 8,
            layers: 24,
            width: 2048
        })
    );
    let term = l.memory.mandatory.iter().find(|t| t.name == CONTROL_TERM).unwrap();
    // 8 vectors of 24 layers of 2,048 values: 1.5 MiB (engine-abi "Control vectors").
    assert_eq!(
        (term.bytes, term.tier, term.lifetime),
        (1_572_864, Tier::Device(0), Lifetime::Engine)
    );
    // The tap lens of spec 13.1 reads one logit row.
    assert_eq!(
        l.taps,
        vec![
            TapLimit {
                name: RESIDUAL_TAP.to_owned(),
                conditions: Vec::new()
            },
            TapLimit {
                name: "lens".to_owned(),
                conditions: vec!["one logit row".to_owned()]
            }
        ]
    );
    // Omni has the same table since milestone 25 step 9. Its vector acts on its text passes: a frame of its speech is
    // a `frame` call, which takes no control list (spec 11.12).
    let omni = Backbone::parse(OMNI_CONFIG).unwrap();
    let o = limits(&omni, &c, 1000, None);
    o.check().unwrap();
    assert!(o.capabilities.contains(&Capability::Control));
    assert_eq!(o.control, l.control);
    let omni_term = o.memory.mandatory.iter().find(|t| t.name == CONTROL_TERM).unwrap();
    assert_eq!(omni_term.bytes, term.bytes);
    assert_eq!(o.taps, l.taps);
}

/// A batch has one row for each segment, and the buffers hold `max_block` rows, so a small block lowers the segment
/// limit (engine-abi "LFM3", "Batched passes").
#[test]
fn the_segment_limit_is_at_most_the_block() {
    let b = backbone();
    for (block, want) in [(512, 8), (8, 8), (4, 4), (1, 1)] {
        let c = Config {
            max_block: block,
            ..config(3)
        };
        assert_eq!(
            limits(&b, &c, 3_456_000_000, None).max_segments,
            want,
            "a block of {block}"
        );
    }
}
