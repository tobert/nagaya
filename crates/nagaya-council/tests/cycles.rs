//! The recipe of a count of cycles (spec 11.17): a count other than the default follows the bytes of the control list
//! in the lab state of the root recipe, and the default count keeps the root of a context with no count.

use nagaya_abi::{ControlOp, CyclesLimit, LayerLoop};
use nagaya_council::build::{
    cycles_in_recipe, cycles_recipe_bytes, loop_recipe_bytes, root_recipe, root_recipe_with_lab,
};
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

fn limit() -> CyclesLimit {
    CyclesLimit {
        min: 1,
        max: 5,
        default: 3,
        start: 6,
        end: 18,
    }
}

#[test]
fn the_default_count_and_no_count_put_no_count_in_the_recipe() {
    assert_eq!(cycles_in_recipe(Some(&limit()), None), None);
    assert_eq!(cycles_in_recipe(Some(&limit()), Some(3)), None, "the default count");
    assert_eq!(cycles_in_recipe(Some(&limit()), Some(5)), Some(5));
    assert_eq!(cycles_in_recipe(Some(&limit()), Some(1)), Some(1));
    assert_eq!(cycles_in_recipe(None, None), None, "an engine with no cycles");
    assert!(cycles_recipe_bytes(None).is_empty());
    let digest = [7u8; 32];
    assert_eq!(
        root_recipe_with_lab(&digest, b"g", &[], None, None),
        root_recipe(&digest, b"g")
    );
}

#[test]
fn the_bytes_of_a_count_are_the_marker_then_the_count_after_the_list() {
    let mut want = (u64::MAX - 1).to_le_bytes().to_vec();
    want.extend(5u32.to_le_bytes());
    assert_eq!(cycles_recipe_bytes(Some(5)), want);
    assert_ne!(
        cycles_recipe_bytes(Some(5))[..8],
        loop_recipe_bytes(Some(LayerLoop {
            start: 0,
            end: 1,
            count: 5
        }))[..8],
        "the marker of a count is not the marker of a loop"
    );
    let digest = [3u8; 32];
    let mut lab = recipe_bytes(&[entry()]);
    lab.extend(&want);
    assert_eq!(
        root_recipe_with_lab(&digest, b"", &[entry()], None, Some(5)),
        recipe_root(&digest, b"", &[], &lab)
    );
}

#[test]
fn each_count_has_its_own_root() {
    let digest = [3u8; 32];
    let root = |n| root_recipe_with_lab(&digest, b"", &[], None, n);
    let roots: Vec<_> = [None, Some(1), Some(2), Some(4), Some(5)]
        .into_iter()
        .map(root)
        .collect();
    for (i, a) in roots.iter().enumerate() {
        assert!(
            roots[i + 1..].iter().all(|b| b != a),
            "count {i} of the list has its own root"
        );
    }
    assert_eq!(root(Some(5)), root(Some(5)));
    assert_ne!(
        root_recipe_with_lab(&digest, b"", &[entry()], None, Some(5)),
        root(Some(5)),
        "the list still counts"
    );
}
