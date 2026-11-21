//! The recipe of a layer loop (spec 11.16): its bytes follow the bytes of the control list in the lab state of the
//! root recipe, and a context with no loop keeps its root.

use nagaya_abi::{ControlOp, LayerLoop};
use nagaya_council::build::{loop_recipe_bytes, root_recipe, root_recipe_with_control, root_recipe_with_lab};
use nagaya_council::control::{ControlEntry, recipe_bytes};
use nagaya_store::{ContentId, recipe_root};

fn entry() -> ControlEntry {
    ControlEntry {
        vector: ContentId::from_hex(&"ab".repeat(32)).unwrap(),
        op: ControlOp::Add,
        strength: 4.0,
        layers: None,
    }
}

fn span(start: u32, end: u32, count: u32) -> Option<LayerLoop> {
    Some(LayerLoop { start, end, count })
}

#[test]
fn no_loop_keeps_the_root_of_a_plain_context_and_of_a_control_list() {
    let digest = [7u8; 32];
    assert_eq!(
        root_recipe_with_lab(&digest, b"g", &[], None, None),
        root_recipe(&digest, b"g")
    );
    assert_eq!(
        root_recipe_with_lab(&digest, b"", &[entry()], None, None),
        root_recipe_with_control(&digest, b"", &[entry()])
    );
    assert!(loop_recipe_bytes(None).is_empty());
}

#[test]
fn the_bytes_of_a_loop_are_the_marker_then_start_end_and_count_after_the_list() {
    let mut want = u64::MAX.to_le_bytes().to_vec();
    want.extend(10u32.to_le_bytes());
    want.extend(14u32.to_le_bytes());
    want.extend(2u32.to_le_bytes());
    assert_eq!(loop_recipe_bytes(span(10, 14, 2)), want);
    let digest = [3u8; 32];
    let mut lab = recipe_bytes(&[entry()]);
    lab.extend(&want);
    assert_eq!(
        root_recipe_with_lab(&digest, b"", &[entry()], span(10, 14, 2), None),
        recipe_root(&digest, b"", &[], &lab)
    );
}

#[test]
fn each_loop_has_its_own_root_and_count_1_is_not_plain() {
    let digest = [3u8; 32];
    let root = |lp| root_recipe_with_lab(&digest, b"", &[], lp, None);
    let base = root(span(10, 14, 2));
    assert_eq!(base, root(span(10, 14, 2)));
    for other in [
        root(span(6, 14, 2)),
        root(span(10, 18, 2)),
        root(span(10, 14, 3)),
        root(None),
    ] {
        assert_ne!(other, base);
    }
    assert_ne!(
        root(span(10, 14, 1)),
        root_recipe(&digest, b""),
        "count 1 has its own recipe (gogi)"
    );
    assert_ne!(
        root_recipe_with_lab(&digest, b"", &[entry()], span(10, 14, 2), None),
        base,
        "the list still counts"
    );
}
