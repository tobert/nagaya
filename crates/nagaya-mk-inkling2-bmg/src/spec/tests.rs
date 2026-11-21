use super::*;
use nagaya_abi::{ControlItem, ControlOp, LayerLoop, OutputsWanted};

/// `config.json` of `thinkingmachines/Inkling-2-Mini` at `ed40e7da`. The research model at `657b53ba` differs only in
/// `is_varloop` and in the key `dtype`, which the engine does not read.
const MINI_CONFIG: &str = include_str!("../../tests/fixtures/inkling2.config.json");

fn edited(text: &str, edit: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut value: serde_json::Value = serde_json::from_str(text).unwrap();
    edit(&mut value);
    value.to_string()
}

/// The value at a key path, to change it.
fn at_mut<'a>(root: &'a mut serde_json::Value, path: &str) -> &'a mut serde_json::Value {
    path.split('.').fold(root, |v, key| &mut v[key])
}

#[test]
fn the_config_of_mini_and_of_the_research_model_parse() {
    assert_eq!(
        (Model::Mini.name(), Model::Research.name()),
        ("inkling2-mini", "inkling2-varloop")
    );
    assert_eq!(Model::parse(MINI_CONFIG), Ok(Model::Mini));
    let research = edited(MINI_CONFIG, |c| {
        c["text_config"]["loop_config"]["is_varloop"] = true.into();
        c.as_object_mut().unwrap().remove("dtype");
    });
    assert_eq!(Model::parse(&research), Ok(Model::Research));
}

/// Each key that the ops read is refused with another value, and the message names the key and the value.
#[test]
fn a_value_that_the_ops_do_not_run_is_refused_with_its_key() {
    let keys = wanted();
    assert_eq!(keys.len(), 41);
    for (key, want) in keys {
        let wrong: serde_json::Value = match want {
            Want::Text(_) => "llama".into(),
            Want::Count(n) => (n + 1).into(),
            Want::Number(x) => (f64::from(x) * 2.0 + 1.0).into(),
            Want::Flag(b) => (!b).into(),
            Want::Counts(mut list) => {
                list[0] = 5;
                list.into()
            }
        };
        let shown = wrong.to_string();
        let error = Model::parse(&edited(MINI_CONFIG, |c| *at_mut(c, key) = wrong)).unwrap_err();
        assert!(error.contains(&format!("{key} {shown}")), "{key}: {error}");
        let gone = Model::parse(&edited(MINI_CONFIG, |c| *at_mut(c, key) = serde_json::Value::Null)).unwrap_err();
        assert!(gone.contains(&format!("{key} null")), "{key}: {gone}");
    }
    let error = Model::parse(&edited(MINI_CONFIG, |c| {
        c["architectures"][0] = "LlamaForCausalLM".into()
    }))
    .unwrap_err();
    assert!(error.contains("architectures"), "{error}");
    let error = Model::parse(&edited(MINI_CONFIG, |c| {
        c["text_config"]["loop_config"]["is_varloop"] = 1.into()
    }))
    .unwrap_err();
    assert!(error.contains("is_varloop 1, not true or false"), "{error}");
    assert!(Model::parse("{").unwrap_err().contains("not JSON"));
}

#[test]
fn a_pass_of_3_cycles_has_48_runs_in_order() {
    let r = runs(DEFAULT_CYCLES);
    assert_eq!(r.len(), 48);
    let layers: Vec<usize> = r.iter().map(|run| run.layer).collect();
    let want: Vec<usize> = (0..6).chain((0..3).flat_map(|_| 6..18)).chain(18..24).collect();
    assert_eq!(layers, want);
    assert_eq!(r[5].cycle, None);
    assert_eq!(
        (r[6].cycle, r[17].cycle, r[18].cycle, r[41].cycle),
        (Some(1), Some(1), Some(2), Some(3))
    );
    assert_eq!(r[42].cycle, None);
    // The layers hold the runs outside the cycles and of cycle 1. The reserve holds cycles 2 and 3 in run order.
    let states: Vec<usize> = r.iter().map(|run| run.state).collect();
    let want: Vec<usize> = (0..18).chain(24..48).chain(18..24).collect();
    assert_eq!(states, want);
    assert!(r.iter().all(|run| state_layer(run.state) == run.layer));
    // 5 cycles use the reserve of 48.
    let five = runs(5);
    assert_eq!(five.len(), 72);
    assert_eq!(
        five.iter().map(|run| run.state).max(),
        Some(LAYERS + MAX_EXTRA_RUNS - 1)
    );
    assert_eq!(runs(1).len(), 24);
    assert!(runs(1).iter().all(|run| run.state == run.layer));
}

#[test]
fn the_parts_follow_the_states_of_the_runs() {
    let p = parts(DEFAULT_EXTRA_RUNS);
    assert_eq!(p.len(), 3 * 48);
    let names: Vec<&str> = p.iter().map(|part| part.name.as_str()).collect();
    assert_eq!(&names[..6], ["k.00", "v.00", "conv.00", "k.01", "v.01", "conv.01"]);
    assert_eq!(
        &names[71..77],
        ["conv.23", "k.x00", "v.x00", "conv.x00", "k.x01", "v.x01"]
    );
    assert_eq!(names[143], "conv.x23");
    let find = |name: &str| {
        p.iter()
            .find(|part| part.name == name)
            .unwrap_or_else(|| panic!("{name}"))
    };
    for name in ["k.05", "v.11", "k.17", "v.23", "k.x05", "v.x11", "k.x17", "v.x23"] {
        assert_eq!(find(name).kind, PartKind::Cells, "{name}");
    }
    for name in ["k.00", "v.04", "k.06", "v.22", "k.x00", "v.x04", "k.x12", "v.x22"] {
        assert_eq!(find(name).kind, PartKind::Ring, "{name}");
    }
    assert_eq!(find("k.05").shape, vec![8, 128]);
    assert_eq!(
        (find("conv.x07").kind, find("conv.x07").shape.clone()),
        (PartKind::Fixed, vec![3, 6656])
    );
    assert!(
        p.iter()
            .all(|part| part.dtype == Dtype::F32 && part.stage == 0 && part.block_format.is_none())
    );
    assert_eq!(p.iter().filter(|part| part.kind == PartKind::Cells).count(), 2 * 8);
    // A reserve of 48 adds the runs of cycles 4 and 5 after those of cycles 2 and 3.
    let five = parts(MAX_EXTRA_RUNS);
    assert_eq!(&five[..p.len()], &p[..]);
    assert_eq!(five.len(), 3 * 72);
    assert_eq!(five.last().unwrap().name, "conv.x47");
    assert_eq!(five.iter().filter(|part| part.kind == PartKind::Cells).count(), 2 * 12);
}

#[test]
fn a_slot_counts_its_planes_and_its_conv_parts() {
    // 8 runs of cells hold 65,536 bytes of each position, 40 sliding runs hold rings of 3,072 rows, and 48 conv parts
    // hold 79,872 bytes each (engine-abi "Inkling 2", "The cost of a slot").
    let cells = 8 * 2 * 4096;
    let rings = 40 * 2 * 3072 * 4096;
    let convs = 48 * 79_872;
    assert_eq!(cells, 65_536);
    assert_eq!(slot_bytes(32_768, 24), 32_768 * cells + rings + convs);
    assert_eq!(slot_bytes(32_768, 24), 3_157_950_464);
    assert_eq!(slot_bytes(262_144, 24), 18_190_336_000);
    // 5 cycles: 12 runs of cells, 60 rings and 72 conv parts.
    assert_eq!(
        slot_bytes(32_768, 48),
        32_768 * 12 * 2 * 4096 + 60 * 2 * 3072 * 4096 + 72 * 79_872
    );
    assert_eq!(slot_bytes(32_768, 48), 4_736_925_696);
    // A slot of 2,048 positions or fewer keeps every position, but its sliding runs still hold rings on the card,
    // because the attention applies the window only to a ring (engine-abi "The host spec", "Slot sizes").
    assert_eq!(slot_bytes(2048, 24), 8 * 2 * 2048 * 4096 + rings + convs);
    assert_eq!(slot_bytes(2048, 24), 1_144_684_544);
    assert_eq!((ring_of(2048), ring_of(2049)), (0, 2048));
    assert_eq!(
        (plane_rows(2048, 5), plane_rows(2048, 4), plane_rows(8, 4)),
        (2048, 3072, 3072)
    );
    assert_eq!((plane_rows(4096, 5), plane_rows(4096, 4)), (4096, 3072));
    assert_eq!((plane_rows(4096, 24 + 5), plane_rows(4096, 24 + 4)), (4096, 3072));
    assert_eq!((CONV_BYTES, ROW_BYTES), (79_872, 4096));
}

fn config(slots: Vec<usize>) -> Config {
    Config {
        model_dir: PathBuf::from("/inkling2"),
        slots,
        max_block: DEFAULT_MAX_BLOCK,
        extra_runs: DEFAULT_EXTRA_RUNS,
    }
}

#[test]
fn a_config_outside_the_sizes_is_refused() {
    assert!(config(vec![SLOT_TOKENS; 3]).check().is_ok());
    assert!(config(vec![1, 32_768, MAX_SLOT_TOKENS]).check().is_ok());
    let none = config(Vec::new()).check().unwrap_err();
    assert!(none.contains("slots"), "{none}");
    let zero = config(vec![0]).check().unwrap_err();
    assert!(zero.contains("1 to 1048576"), "{zero}");
    let long = config(vec![MAX_SLOT_TOKENS + 1]).check().unwrap_err();
    assert!(long.contains("1048577"), "{long}");
    let down = config(vec![65_536, 32_768]).check().unwrap_err();
    assert!(down.contains("short to long"), "{down}");
    for block in [0, 513] {
        let c = Config {
            max_block: block,
            ..config(vec![SLOT_TOKENS])
        };
        assert!(c.check().unwrap_err().contains("max_block"), "{block}");
    }
    for extra in [24, 36, 48] {
        let c = Config {
            extra_runs: extra,
            ..config(vec![SLOT_TOKENS])
        };
        assert!(c.check().is_ok(), "{extra}");
    }
    for extra in [0, 12, 30, 60] {
        let c = Config {
            extra_runs: extra,
            ..config(vec![SLOT_TOKENS])
        };
        let e = c.check().unwrap_err();
        assert!(e.contains(&format!("extra_runs is {extra}; give 24, 36 or 48")), "{e}");
    }
}

#[test]
fn the_limits_hold_the_terms_the_taps_and_the_control_table_and_declare_no_loop() {
    let c = config(vec![32_768, 32_768, 262_144]);
    let l = limits(&c, 18_880_455_680);
    l.check().unwrap();
    assert_eq!(
        l.capabilities,
        vec![
            Capability::Parts,
            Capability::Slots,
            Capability::Coverage,
            Capability::Close,
            Capability::Cycles,
            Capability::Control
        ]
    );
    // A vector holds 24 layers of 2,304 values (engine-abi "The lab of Inkling 2").
    let table = ControlLimit {
        tables: 8,
        layers: 24,
        width: 2304,
    };
    assert_eq!((l.layer_loops.clone(), l.control), (None, Some(table)));
    assert_eq!(table.table_bytes(), 1_769_472);
    // The reserve of 24 runs holds cycles 2 and 3. Layers 6 to 17 make a cycle (engine-abi "Inkling 2").
    assert_eq!(
        l.cycles,
        Some(CyclesLimit {
            min: 1,
            max: 3,
            default: 3,
            start: 6,
            end: 18
        })
    );
    let names: Vec<&str> = l.taps.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["residual", "fold", "blocks", "messages", "gates", "routes"]);
    assert!(l.taps.iter().all(|t| t.conditions.is_empty()));
    // Both models hold the tower, so both take the rows of images (engine-abi "The choices of the pass").
    assert_eq!(l.inputs, Inputs::TokensAndRows);
    assert_eq!((l.max_segments, l.max_pass_inputs, l.max_logit_rows), (1, 512, 16));
    assert_eq!(l.feed, Some(FeedRule { block: 512 }));
    assert!(!l.segments_exact && l.checkpoint_grain.is_none());
    assert_eq!(l.determinism, Determinism::Always);
    assert_eq!(l.parts, parts(24));
    assert_eq!(
        l.slots,
        vec![
            SlotLimit {
                bytes: 3_157_950_464,
                count: 2,
                tokens: 32_768
            },
            SlotLimit {
                bytes: 18_190_336_000,
                count: 1,
                tokens: 262_144
            }
        ]
    );
    let term = |l: &Limits, name: &str| l.memory.mandatory.iter().find(|t| t.name == name).unwrap().clone();
    assert_eq!(l.memory.mandatory.len(), 6);
    assert_eq!(term(&l, "control").bytes, 1_769_472);
    assert_eq!(term(&l, "weights").bytes, 18_880_455_680);
    assert_eq!(term(&l, "vision").bytes, 636_825_472);
    assert_eq!(term(&l, "kv").bytes, 2 * 3_157_950_464 + 18_190_336_000);
    assert_eq!(term(&l, "pinned").bytes, pinned_bytes());
    assert_eq!(term(&l, "pinned").tier, Tier::Host);
    for name in ["weights", "vision", "kv", "work", "control"] {
        assert_eq!(term(&l, name).tier, Tier::Device(0), "{name}");
    }
    // The work is the buffers of a block, with 16 logit rows of the 200,064 rows of the head, and the ring. Only the
    // scratch of the ring reads the slot size. It is the scratch of the attention of Inkling 2.
    let short = limits(&config(vec![32_768]), 1);
    assert_eq!(
        term(&l, "work").bytes - term(&short, "work").bytes,
        4 * (scratch_floats(262_144) - scratch_floats(32_768)) as u64
    );
    assert!(bufs_bytes(512) > (16 * HEAD_ROWS * 4) as u64);
    assert_eq!(
        scratch_floats(32_768),
        nagaya_mk_bmg::tasks::inkling2_attn_scratch_floats(512, 32_768)
    );
    let logits = block_buffers(1)
        .into_iter()
        .find(|(name, _)| *name == "logits")
        .unwrap();
    assert_eq!(logits.1, 16 * 200_064);
    let small = limits(
        &Config {
            max_block: 1,
            ..config(vec![32_768])
        },
        1,
    );
    // The images of a pass add the tower of the largest image and the media rows of a block.
    assert_eq!(
        term(&short, "work").bytes - term(&small, "work").bytes,
        bufs_bytes(512) - bufs_bytes(1) + crate::vision::PassBufs::bytes(512) - crate::vision::PassBufs::bytes(1)
    );
    assert_eq!(
        term(&short, "work").bytes,
        bufs_bytes(512) + crate::vision::PassBufs::bytes(512) + ring_device_bytes(32_768) + tap_bytes(24)
    );
    // The buffers of the taps hold 16 logit rows at the most cycles of the reserve (engine-abi "The lab of Inkling
    // 2", "The buffers").
    assert_eq!((tap_bytes(24), tap_bytes(48)), (36_372_480, 68_567_040));
    // A reserve of 48 adds its parts and its bytes.
    let five = limits(
        &Config {
            extra_runs: 48,
            ..config(vec![32_768])
        },
        1,
    );
    five.check().unwrap();
    assert_eq!(five.cycles.map(|c| (c.min, c.max, c.default)), Some((1, 5, 3)));
    assert_eq!(cycles_limit(36).max, 4);
    assert_eq!(five.parts.len(), 3 * 72);
    assert_eq!(five.slots[0].bytes, 4_736_925_696);
    assert_eq!(
        term(&five, "work").bytes - term(&short, "work").bytes,
        tap_bytes(48) - tap_bytes(24)
    );
}

/// A pass names each tap of the limits, in any order, and the outputs follow the order of `TAPS`. Another name is
/// refused, and the message names the taps.
#[test]
fn a_pass_names_the_taps_of_the_limits() {
    let tap = |name: &str| nagaya_abi::Tap { name: name.to_owned() };
    assert_eq!(Taps::of(&[]).unwrap(), Taps::default());
    assert!(!Taps::default().any());
    let all: Vec<nagaya_abi::Tap> = TAPS.iter().rev().map(|n| tap(n)).collect();
    let named = Taps::of(&all).unwrap();
    assert!(named.residual && named.fold && named.blocks && named.messages && named.gates && named.routes);
    let one = Taps::of(&[tap("messages")]).unwrap();
    assert_eq!(
        one,
        Taps {
            messages: true,
            ..Taps::default()
        }
    );
    assert!(one.any());
    let e = Taps::of(&[tap("lens")]).unwrap_err();
    assert!(e.contains("no tap `lens`") && e.contains("residual, fold"), "{e}");
}

/// The plan counts a slab of each expert matrix with the bytes that `repack::expert_slab` gives.
#[test]
fn the_slab_bytes_are_the_bytes_of_the_repack() {
    for (rows, cols) in [(32, 64), (16, 128)] {
        let experts = 3;
        let nv = crate::repack::expert_slab(
            &vec![0; experts * rows * cols / 2],
            &vec![0; experts * rows * cols / 16],
            Some(&vec![1.0; experts]),
            experts,
            rows,
            cols,
            false,
            false,
        )
        .unwrap();
        assert_eq!(
            expert_slab_bytes(experts, rows, cols, false),
            nv.len(),
            "NVFP4 {rows} x {cols}"
        );
        let mx = crate::repack::expert_slab(
            &vec![0; experts * rows * cols],
            &vec![0; experts * rows * cols / 32],
            None,
            experts,
            rows,
            cols,
            true,
            false,
        )
        .unwrap();
        assert_eq!(
            expert_slab_bytes(experts, rows, cols, true),
            mx.len(),
            "MXFP8 {rows} x {cols}"
        );
    }
    assert_eq!((rounded(1), rounded(64), rounded(65)), (64, 64, 128));
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
        layer_loop: None,
        cycles: None,
    }
}

#[test]
fn a_pass_runs_on_from_the_token_count_and_never_goes_back() {
    let mut held = Held::new(&[1024, 4096], DEFAULT_EXTRA_RUNS);
    let block = held.check_pass(&[segment(0, 0, &[1, 2, 3], &[2])], 512).unwrap();
    assert_eq!((block.slot, block.pos0, block.logit_rows.clone()), (0, 0, vec![2]));
    assert!(
        block.zero_conv(),
        "a pass from position 0 starts from conv states of zeros"
    );
    held.ran(&block);
    assert_eq!(held.n[0], 3);
    let next = held.check_pass(&[segment(0, 3, &[4], &[3])], 512).unwrap();
    assert!(!next.zero_conv());
    let back = held.check_pass(&[segment(0, 2, &[4], &[])], 512).unwrap_err();
    assert!(back.contains("does not go back"), "{back}");
    let gap = held.check_pass(&[segment(0, 4, &[4], &[])], 512).unwrap_err();
    assert!(gap.contains("run from 3"), "{gap}");
    let mut control = segment(0, 3, &[1], &[]);
    control.control = vec![ControlItem {
        index: 0,
        op: ControlOp::Add,
        strength: 1.0,
        layers: None,
    }];
    let mut outside = segment(0, 3, &[1], &[]);
    outside.control = vec![ControlItem {
        index: 8,
        op: ControlOp::Add,
        strength: 1.0,
        layers: None,
    }];
    let mut endless = segment(0, 3, &[1], &[]);
    endless.control = vec![ControlItem {
        index: 0,
        op: ControlOp::Clamp,
        strength: f32::INFINITY,
        layers: None,
    }];
    let mut looped = segment(0, 3, &[1], &[]);
    looped.layer_loop = Some(LayerLoop {
        start: 6,
        end: 18,
        count: 2,
    });
    let mut cycled = segment(0, 3, &[1], &[]);
    cycled.cycles = Some(3);
    let mut many = segment(0, 3, &[1], &[]);
    many.cycles = Some(4);
    let mut none = segment(0, 3, &[1], &[]);
    none.cycles = Some(0);
    let mut meta = segment(0, 3, &[1], &[]);
    meta.meta = vec![0; 16];
    let rows = Segment {
        inputs: SegmentInputs::TokensAndRows {
            tokens: vec![1],
            rows: vec![0; 16],
        },
        ..segment(0, 3, &[1], &[])
    };
    let mut positions = segment(0, 3, &[1, 2], &[]);
    positions.positions.pop();
    for (segments, why) in [
        (vec![], "0 segments"),
        (vec![segment(0, 3, &[1], &[]), segment(1, 0, &[1], &[])], "2 segments"),
        (vec![segment(2, 0, &[1], &[])], "slots"),
        (vec![segment(0, 3, &[], &[])], "0 tokens"),
        (vec![segment(0, 3, &[1; 513], &[])], "513 tokens"),
        (vec![segment(0, 3, &[200_058], &[])], "200058"),
        (vec![segment(0, 3, &[200_057], &[])], ""),
        (vec![segment(0, 3, &[1], &[4])], "logit row at 4"),
        (
            vec![segment(0, 3, &[1; 17], &(3..20).collect::<Vec<_>>())],
            "17 logit rows",
        ),
        (vec![segment(0, 3, &[1; 16], &(3..19).collect::<Vec<_>>())], ""),
        (vec![control], ""),
        (vec![outside], "names table entry 8; the table has 8 entries"),
        (vec![endless], "strength inf"),
        (vec![looped], "layer_loops"),
        (vec![cycled], ""),
        (vec![many], "runs 4 cycles; the engine runs 1 to 3"),
        (vec![none], "runs 0 cycles"),
        (vec![meta], "meta"),
        (vec![rows], "rows"),
        (vec![positions], "2 tokens and 1 positions"),
        (vec![segment(1, 0, &[1; 512], &[])], ""),
    ] {
        let got = held.check_pass(&segments, 512);
        if why.is_empty() {
            assert!(got.is_ok(), "{got:?}");
        } else {
            let error = got.unwrap_err();
            assert!(error.contains(why), "{why}: {error}");
        }
    }
    let mut full = Held::new(&[8], DEFAULT_EXTRA_RUNS);
    full.n[0] = 6;
    let e = full.check_pass(&[segment(0, 6, &[1, 2, 3], &[])], 512).unwrap_err();
    assert!(e.contains("do not fit"), "{e}");
}

/// The part ids of `k.04` (a ring), `k.05` (cells), `conv.00`, and `k.x00`, the ring of layer 6 in cycle 2.
const RING: PartId = 12;
const CELLS: PartId = 15;
const CONV: PartId = 2;
const RESERVE_RING: PartId = 72;

#[test]
fn a_ring_holds_its_last_rows_and_a_short_slot_holds_every_position() {
    let list = part_list(DEFAULT_EXTRA_RUNS);
    assert_eq!(
        (list[RING], list[CELLS], list[CONV], list[RESERVE_RING]),
        (Part::K(4), Part::K(5), Part::Conv(0), Part::K(24))
    );
    let mut held = Held::new(&[2048, 4096], DEFAULT_EXTRA_RUNS);
    held.n = vec![1000, 3000];
    assert_eq!(held.base(0, 4), 0);
    assert_eq!(held.base(1, 4), 3000 - 2048);
    assert_eq!(held.base(1, 24), 3000 - 2048, "the reserve entry of layer 6 is a ring");
    assert_eq!(held.base(1, 5), 0);
    assert_eq!(held.base(1, 24 + 5), 0, "the reserve entry of layer 11 holds cells");
    let row = vec![0u8; 4096];
    let span = held.coverage(State::Slot(1), RING, &row.repeat(2048));
    assert_eq!((span.start, span.end), (952, 3000));
    assert_eq!(held.coverage(State::Slot(1), CONV, &[0; 64]).end, 0);
    let fit = |slot, part, start, end| held.fits(slot, &CoverageSpan { part, start, end });
    // A ring needs the window of 1,024 positions before its end.
    assert!(fit(1, RING, 1976, 3000).is_ok());
    assert!(fit(1, RESERVE_RING, 1976, 3000).is_ok());
    assert!(fit(1, RING, 0, 3000).is_ok());
    let e = fit(1, RING, 1977, 3000).unwrap_err();
    assert!(e.contains("window of 1024"), "{e}");
    assert!(fit(0, RING, 1, 1000).unwrap_err().contains("from 0"));
    assert!(fit(1, CELLS, 1976, 3000).unwrap_err().contains("from 0"));
    assert!(fit(0, CELLS, 0, 2049).unwrap_err().contains("do not fit"));
    assert!(fit(0, CONV, 0, 0).unwrap_err().contains("conv state"));
}

#[test]
fn a_restore_writes_every_part_and_set_n_takes_the_count_only_then() {
    let mut held = Held::new(&[4096], DEFAULT_EXTRA_RUNS);
    held.n[0] = 50;
    let state = State::Slot(0);
    let row = 4096;
    // With no write, a count stays or goes to 0, and never goes back to another count.
    assert!(held.check_set_n(state, 50).is_ok());
    assert!(held.check_set_n(state, 0).is_ok());
    assert!(held.check_set_n(state, 49).unwrap_err().contains("does not go back"));
    assert!(held.check_set_n(state, 4097).unwrap_err().contains("4096 positions"));
    // A cells part starts at 0; a ring part of a long slot keeps the last 2,048 rows of a write.
    assert!(
        held.check_cells_write(state, CELLS, 1, 1, row)
            .unwrap_err()
            .contains("from 0")
    );
    let (_, _, kept) = held.check_cells_write(state, RING, 0, 3000, 3000 * row).unwrap();
    assert_eq!(kept, 952..3000);
    assert!(held.check_cells_write(state, RING, 0, 2, row).is_err());
    assert!(held.check_cells_write(state, RING, 4000, 97, 97 * row).is_err());
    assert!(
        held.check_cells_write(state, CONV, 0, 1, row)
            .unwrap_err()
            .contains("fixed part")
    );
    assert!(
        held.check_part_write(state, CELLS, 79_872)
            .unwrap_err()
            .contains("row part")
    );
    assert!(held.check_part_write(state, CONV, 79_871).is_err());
    let list = part_list(DEFAULT_EXTRA_RUNS);
    for (id, part) in list.iter().enumerate() {
        if matches!(part, Part::Conv(_)) {
            held.check_part_write(state, id, 79_872).unwrap();
            held.wrote(0, id, None);
        } else if id != RING {
            let start = if is_full(state_layer(part.state())) {
                0
            } else {
                1500 - 1024
            };
            held.wrote(0, id, Some(start..1500));
        }
    }
    assert_eq!(held.n[0], 0, "a write empties the slot until set_n");
    let e = held.check_set_n(state, 1500).unwrap_err();
    assert!(e.contains(&format!("part {RING} got no bytes")), "{e}");
    held.wrote(0, RING, Some(477..1500));
    assert!(held.check_set_n(state, 1500).unwrap_err().contains("window"));
    held.wrote(0, RING, Some(0..1499));
    assert!(held.check_set_n(state, 1500).unwrap_err().contains("up to 1499"));
    held.wrote(0, RING, None);
    assert!(held.check_set_n(state, 1500).unwrap_err().contains("got no rows"));
    held.wrote(0, RING, Some(476..1500));
    assert_eq!(held.check_set_n(state, 1500), Ok(0));
    held.set_n(0, 1500);
    assert_eq!(held.n[0], 1500);
    assert!(held.check_set_n(state, 1499).is_err(), "set_n took the writes");
    // The rings hold no position before the restored rows, and a later pass moves the base on.
    assert_eq!((held.base(0, 4), held.base(0, 5), held.base(0, 30)), (476, 0, 476));
    held.n[0] = 3000;
    assert_eq!(held.base(0, 4), 952);
    held.clear(0);
    held.n[0] = 100;
    assert_eq!(held.base(0, 4), 0, "a clear forgets the restored start");
}

#[test]
fn a_read_stays_inside_the_rows_that_a_part_holds() {
    let mut held = Held::new(&[4096], DEFAULT_EXTRA_RUNS);
    held.n[0] = 3000;
    let state = State::Slot(0);
    let row = 4096;
    assert!(held.check_read(state, RING, 1976, 1024, 1024 * row).is_ok());
    assert!(held.check_read(state, RING, 952, 2048, 2048 * row).is_ok());
    assert!(held.check_read(state, RING, 951, 1, row).unwrap_err().contains("952"));
    assert!(held.check_read(state, CELLS, 0, 3001, 3001 * row).is_err());
    assert!(held.check_read(state, CELLS, 0, 2, row).is_err());
    assert!(
        held.check_read(state, CONV, 0, 1, row)
            .unwrap_err()
            .contains("fixed part")
    );
    assert_eq!(held.check_part_read(state, RING, 1024 * row).unwrap().2, 952);
    assert!(
        held.check_part_read(state, RING, 1000)
            .unwrap_err()
            .contains("whole rows")
    );
    assert!(held.check_part_read(state, CONV, 79_872).is_ok());
    assert!(held.check_part_read(state, CONV, row).is_err());
    assert!(held.check_part_read(State::Slot(1), CONV, 79_872).is_err());
    assert!(
        held.check_part_read(state, 3 * 48, row)
            .unwrap_err()
            .contains("no part 144")
    );
}

/// A pass runs the count of its segment, or the default. A slot keeps the count that fed its positions, because each
/// run of a cycle reads the keys of its own run at the earlier positions (engine-abi "Inkling 2").
#[test]
fn a_slot_runs_the_count_of_cycles_that_fed_it() {
    let mut held = Held::new(&[1024, 4096], MAX_EXTRA_RUNS);
    let counted = |slot, from, n: usize, cycles| Segment {
        cycles,
        ..segment(slot, from, &vec![1; n], &[])
    };
    let block = held.check_pass(&[counted(0, 0, 3, None)], 512).unwrap();
    assert_eq!(block.cycles, DEFAULT_CYCLES);
    let block = held.check_pass(&[counted(0, 0, 3, Some(5))], 512).unwrap();
    assert_eq!(block.cycles, 5);
    held.ran(&block);
    assert_eq!(held.cycles[0], Some(5));
    let e = held.check_pass(&[counted(0, 3, 1, None)], 512).unwrap_err();
    assert!(
        e.contains("slot 0 holds 3 positions of 5 cycles, and the pass runs 3"),
        "{e}"
    );
    let e = held.check_pass(&[counted(0, 3, 1, Some(2))], 512).unwrap_err();
    assert!(e.contains("feed the slot at 5 cycles"), "{e}");
    assert_eq!(held.check_pass(&[counted(0, 3, 1, Some(5))], 512).unwrap().cycles, 5);
    let e = held.check_pass(&[counted(0, 3, 1, Some(6))], 512).unwrap_err();
    assert!(e.contains("runs 6 cycles; the engine runs 1 to 5"), "{e}");
    // The other slot is empty, so it takes any count.
    assert_eq!(held.check_pass(&[counted(1, 0, 2, Some(1))], 512).unwrap().cycles, 1);
    // A clear empties the slot, so the next pass takes another count.
    held.clear(0);
    assert_eq!(held.cycles[0], None);
    let block = held.check_pass(&[counted(0, 0, 2, Some(2))], 512).unwrap();
    held.ran(&block);
    assert_eq!(held.cycles[0], Some(2));
    // A restore writes parts and a count. The bytes do not say their count, so the next pass sets it; the council
    // keys a state by its count (spec 11.17).
    held.wrote(0, CONV, None);
    assert_eq!(held.cycles[0], None);
    held.set_n(0, 0);
    assert_eq!(held.cycles[0], None);
}

/// An image segment of `tokens` from `from` with the record and the rows of `spans` (engine-abi "The choices of the
/// pass").
fn image_segment(from: u64, tokens: &[u32], spans: &[crate::media::Span]) -> Segment {
    let mut meta = Vec::new();
    let mut rows = 0;
    for span in spans {
        meta.extend(
            nagaya_text_inkling2::media::ImageRows {
                grid: span.grid().unwrap(),
                values: Vec::new(),
            }
            .record(span.first as usize),
        );
        rows += span.bytes();
    }
    Segment {
        inputs: SegmentInputs::TokensAndRows {
            tokens: tokens.to_vec(),
            rows: vec![0; rows],
        },
        meta,
        ..segment(0, from, tokens, &[])
    }
}

#[test]
fn a_pass_takes_the_rows_and_the_records_of_its_images() {
    use crate::media::{IMAGE_TOKEN as P, Span};
    use nagaya_text_inkling2::media::Grid;
    let mut held = Held::new(&[4096], DEFAULT_EXTRA_RUNS);
    // An image of 2 by 3 tokens at positions 2 to 8, in blocks of 4 and 6.
    let image = Span::image(Grid { rows: 2, columns: 3 }, 2);
    let block = held
        .check_pass(&[image_segment(0, &[7, 7, P, P], &[image])], 512)
        .unwrap();
    assert_eq!(block.spans.len(), 1);
    assert_eq!((block.spans[0].take.clone(), block.spans[0].block_row), (0..2, 2));
    held.ran(&block);
    let block = held
        .check_pass(&[image_segment(4, &[P, P, P, P, 7, 7], &[image])], 512)
        .unwrap();
    assert_eq!((block.spans[0].take.clone(), block.spans[0].block_row), (2..6, 0));
    held.ran(&block);
    // A text block has no spans.
    let text = held.check_pass(&[segment(0, 10, &[7], &[])], 512).unwrap();
    assert!(text.spans.is_empty());
    for (segments, why) in [
        (
            vec![segment(0, 10, &[7, P], &[])],
            "position 11 holds the image placeholder",
        ),
        (vec![image_segment(10, &[7, P], &[])], "has rows and no meta"),
        (
            vec![Segment {
                inputs: SegmentInputs::Rows(vec![0; 4]),
                ..segment(0, 10, &[7], &[])
            }],
            "rows and no tokens",
        ),
        (
            vec![image_segment(
                10,
                &[7, P],
                &[Span::image(Grid { rows: 1, columns: 1 }, 10)],
            )],
            "position 10 is in media record 0",
        ),
    ] {
        let error = held.check_pass(&segments, 512).unwrap_err();
        assert!(error.contains(why), "{why}: {error}");
    }
    let mut bad = image_segment(10, &[7, P], &[Span::image(Grid { rows: 1, columns: 1 }, 11)]);
    bad.meta[0] = 2;
    let error = held.check_pass(&[bad], 512).unwrap_err();
    assert!(error.contains("the kind 2"), "{error}");
}

/// A vector fits one entry of the table: 24 rows of 2,304 finite `f32` values.
#[test]
fn the_table_takes_a_vector_of_its_size_into_an_entry_that_it_has() {
    let limit = control_limit();
    let good = vec![0u8; 24 * 2304 * 4];
    assert_eq!(check_control_write(&limit, 7, &good), Ok(()));
    let e = check_control_write(&limit, 8, &good).unwrap_err();
    assert!(e.contains("8 entries, not entry 8"), "{e}");
    let e = check_control_write(&limit, 0, &good[4..]).unwrap_err();
    assert!(e.contains("221184 bytes, not 221180"), "{e}");
    let mut nan = good.clone();
    nan[40..44].copy_from_slice(&f32::NAN.to_le_bytes());
    let e = check_control_write(&limit, 0, &nan).unwrap_err();
    assert!(e.contains("value 10 of the vector is not finite"), "{e}");
}

/// The placeholder of a latent position is `pad_token_id` of `generation_config.json` of both models.
#[test]
fn the_latent_placeholder_is_the_pad_token_of_the_generation_config() {
    let text = r#"{"do_sample": true, "temperature": 1.0, "eos_token_id": 200006, "pad_token_id": 199999}"#;
    assert_eq!(check_generation(text), Ok(()));
    let e = check_generation(&text.replace("199999", "200000")).unwrap_err();
    assert!(e.contains("pad_token_id 200000") && e.contains("puts 199999"), "{e}");
    assert!(check_generation("{}").unwrap_err().contains("pad_token_id null"));
}
