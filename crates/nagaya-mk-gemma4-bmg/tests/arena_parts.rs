//! The arena of the Gemma 4 engine, with no card: a slab for each layer on its card, and a slab for the head on the
//! second card (spec 10.2, "The names of the fds").

use nagaya_abi::{ArenaPart, Capability};
use nagaya_mk_gemma4_bmg::spec::{self, Config, HEAD_PART, arena_parts, layer_part};

#[test]
fn the_arena_has_a_slab_for_each_layer_and_one_for_the_head() {
    let parts = arena_parts();
    assert_eq!(parts.len(), 31);
    let mut expected: Vec<ArenaPart> = (0..15)
        .map(|l| ArenaPart {
            device: 0,
            part: format!("w{l:02}"),
        })
        .collect();
    expected.extend((15..30).map(|l| ArenaPart {
        device: 1,
        part: format!("w{l:02}"),
    }));
    expected.push(ArenaPart {
        device: 1,
        part: HEAD_PART.to_owned(),
    });
    assert_eq!(parts, expected);
    assert_eq!(layer_part(7), "w07");
    assert_eq!(HEAD_PART, "head");
}

#[test]
fn the_limits_name_the_arena_capability() {
    let config = Config {
        model_dir: "/nonexistent".into(),
        slots: vec![4096],
        max_block: spec::DEFAULT_MAX_BLOCK,
    };
    let limits = spec::limits(&config, [1, 1]);
    assert!(
        limits.capabilities.contains(&Capability::Arena),
        "{:?}",
        limits.capabilities
    );
}
