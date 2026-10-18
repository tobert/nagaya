//! Generate the canonical fixture and print the golden ids.

use nagaya_abi::{Dtype, PartKind};
use nagaya_store::{Snapshot, Tensor, recipe_boundary, recipe_root};

fn main() {
    let snapshot = Snapshot {
        tensors: vec![
            Tensor {
                name: "state".to_owned(),
                kind: PartKind::Fixed,
                dtype: Dtype::U8,
                shape: vec![16],
                data: (0u8..16).collect(),
                coverage: None,
                block_format: None,
            },
            Tensor {
                name: "cells".to_owned(),
                kind: PartKind::Cells,
                dtype: Dtype::U8,
                shape: vec![2, 8],
                data: (0u8..16).collect(),
                coverage: Some((0, 2)),
                block_format: None,
            },
        ],
        parent: None,
    };
    let bytes = snapshot.encode().expect("encode");
    std::fs::write("nagaya-store/tests/fixtures/snapshot.safetensors", &bytes).expect("write fixture");
    println!("fixture {}", snapshot.content_id().expect("id").hex());

    let root = recipe_root(&[1u8; 32], b"geometry", &[b"approx".to_vec()], b"lab");
    println!("root {}", root.hex());
    let first = recipe_boundary(&root, &[1, 2, 3], 3);
    println!("boundary {}", first.hex());
}
