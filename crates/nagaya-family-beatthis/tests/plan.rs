//! The plan: a task list of the model with no arithmetic in it. These checks need no interpreter: the shapes, the
//! bounds of each task, and the use of each weight of the checkpoint.

mod common;

use std::collections::BTreeMap;

use nagaya_family_beatthis::Error;
use nagaya_family_beatthis::plan::{self, Config, Prep, Task};
use safetensors::SafeTensors;

const FINAL0: Config = Config {
    stem_dim: 32,
    transformer_dim: 512,
    n_layers: 6,
    head_dim: 32,
    ff_mult: 4,
};
const SMALL0: Config = Config {
    stem_dim: 32,
    transformer_dim: 128,
    n_layers: 6,
    head_dim: 32,
    ff_mult: 4,
};
const LENGTHS: [usize; 8] = [1, 2, 13, 26, 38, 500, 1499, 1500];

fn hp(extra: &str) -> String {
    format!(
        r#"{{"spect_dim": 128, "stem_dim": 32, "transformer_dim": 512, "n_layers": 6, "head_dim": 32, "ff_mult": 4,
            "sum_head": true, "partial_transformers": true{extra}}}"#
    )
}

#[test]
fn the_config_reads_the_hyper_parameters() {
    assert_eq!(Config::from_json(&hp("")).unwrap(), FINAL0);
}

#[test]
fn the_config_refuses_a_model_this_family_does_not_build() {
    for bad in [
        r#"{"spect_dim": 128}"#.to_owned(),
        hp(r#", "sum_head": false"#),
        hp(r#", "partial_transformers": false"#),
        hp("").replace(r#""transformer_dim": 512"#, r#""transformer_dim": 500"#),
        hp("").replace(r#""spect_dim": 128"#, r#""spect_dim": 64"#),
        // 1 divides each dim, but RoPE turns pairs, so head_dim must be even.
        hp("").replace(r#""head_dim": 32"#, r#""head_dim": 1"#),
        "not json".to_owned(),
    ] {
        assert!(
            matches!(Config::from_json(&bad), Err(Error::Config(_))),
            "accepted {bad}"
        );
    }
}

#[test]
fn each_task_stays_in_its_buffers_at_each_length() {
    for config in [FINAL0, SMALL0] {
        for t in LENGTHS {
            let p = plan::build(&config, t);
            p.check().unwrap_or_else(|e| panic!("{config:?} at {t} frames: {e}"));
        }
    }
}

#[test]
fn the_check_finds_a_buffer_that_is_too_short() {
    let p = plan::build(&FINAL0, 26);
    for i in 0..p.buffers.len() {
        let mut q = p.clone();
        q.buffers[i] -= 1;
        assert!(q.check().is_err(), "buffer {i} one value short passed the check");
    }
}

/// The head reads one frame less than the final norm wrote: no buffer is too short, but a value goes unread.
#[test]
fn the_check_finds_a_read_that_does_not_match_its_write() {
    let mut p = plan::build(&FINAL0, 26);
    let Some(Task::Linear { rows, .. }) = p.tasks.last_mut() else {
        panic!("the last task is not the head")
    };
    *rows -= 1;
    let e = p.check().unwrap_err();
    assert!(e.contains("were written"), "{e}");
}

#[test]
fn the_check_finds_a_task_that_writes_what_it_reads() {
    let mut p = plan::build(&FINAL0, 26);
    // A square linear (to_out), so that its write fits its input buffer and only the alias rule can refuse it.
    let i = p
        .tasks
        .iter()
        .position(|t| matches!(t, Task::Linear { din, dout, .. } if din == dout))
        .unwrap();
    let Task::Linear { x, y, .. } = &mut p.tasks[i] else {
        unreachable!()
    };
    *y = *x;
    let e = p.check().unwrap_err();
    assert!(e.contains("reads and writes one buffer"), "{e}");
}

#[test]
fn the_plan_marks_the_rows_of_the_oracle() {
    let p = plan::build(&FINAL0, 1500);
    let names: Vec<&str> = p.marks.iter().map(|m| m.name.as_str()).collect();
    let mut want = vec!["stem", "block0", "block1", "block2", "linear"];
    let layers: Vec<String> = (0..6)
        .flat_map(|i| [format!("layer{i}.attn"), format!("layer{i}.ff")])
        .collect();
    want.extend(layers.iter().map(String::as_str));
    want.push("final_norm");
    assert_eq!(names, want);
}

#[test]
fn the_plan_is_the_same_each_time() {
    assert_eq!(plan::build(&FINAL0, 777), plan::build(&FINAL0, 777));
}

/// Each tensor of the checkpoint is read by the plan, with the shape that the plan expects, and the plan reads no
/// other tensor. A layer that the plan forgets fails here.
fn check_weights(model: &str, config: &Config) {
    let Some(bytes) = common::weights(model) else { return };
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let file: BTreeMap<String, Vec<usize>> = st.tensors().into_iter().map(|(n, t)| (n, t.shape().to_vec())).collect();
    let p = plan::build(config, 1500);
    let mut used: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for w in &p.weights {
        for (name, shape) in w.prep.sources() {
            if let Some(old) = used.insert(name.clone(), shape.clone()) {
                assert_eq!(old, shape, "{name} is read with two shapes");
            }
        }
    }
    assert_eq!(used, file, "{model}: the plan and the file differ");
    let (_, meta) = SafeTensors::read_metadata(&bytes).unwrap();
    let hp = meta
        .metadata()
        .as_ref()
        .and_then(|m| m.get("hyper_parameters"))
        .expect("no hyper_parameters");
    assert_eq!(
        &Config::from_json(hp).unwrap(),
        config,
        "{model}: the config of its metadata"
    );
}

#[test]
fn the_plan_reads_each_weight_of_final0() {
    check_weights("final0", &FINAL0);
}

#[test]
fn the_plan_reads_each_weight_of_small0() {
    check_weights("small0", &SMALL0);
}

#[test]
fn a_column_permutation_names_its_shape() {
    let p = plan::build(&FINAL0, 10);
    let n = p
        .weights
        .iter()
        .filter(|w| matches!(w.prep, Prep::PermuteColumns { .. }))
        .count();
    assert_eq!(n, 1, "only the projection after the frontend permutes its columns");
}
