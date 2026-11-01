//! The file of the plan table (spec 12.1, "The plan table of a HIP engine"). These tests need no GPU.

use nagaya_mk_hip::plan::{DataType, GemmShape, PlanError, PlanTable};

const LINEAGE: &str = "hipblaslt 1.2.2; hip 7.2; gfx1151; 1002:1586";

fn shape(outputs: u32) -> GemmShape {
    GemmShape::linear(64, 4096, outputs, DataType::Bf16)
}

#[test]
fn a_pick_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let mut table = PlanTable::open(dir.path(), LINEAGE).unwrap();
    assert!(table.is_empty());
    assert!(
        !PlanTable::path_in(dir.path(), LINEAGE).exists(),
        "an empty table writes no file"
    );
    table.record(shape(4096), 17).unwrap();
    table.record(shape(12288), 3).unwrap();
    let again = PlanTable::open(dir.path(), LINEAGE).unwrap();
    assert_eq!(again.pick(&shape(4096)), Some(17));
    assert_eq!(again.pick(&shape(12288)), Some(3));
    assert_eq!(again.pick(&shape(64)), None);
    assert_eq!(again.len(), 2);
}

#[test]
fn a_shape_keeps_its_pick() {
    let dir = tempfile::tempdir().unwrap();
    let mut table = PlanTable::open(dir.path(), LINEAGE).unwrap();
    table.record(shape(4096), 17).unwrap();
    table.record(shape(4096), 17).unwrap();
    match table.record(shape(4096), 18) {
        Err(PlanError::Repick { held: 17, new: 18, .. }) => {}
        other => panic!("a second pick gave {other:?}"),
    }
    let again = PlanTable::open(dir.path(), LINEAGE).unwrap();
    assert_eq!(again.pick(&shape(4096)), Some(17));
    assert_eq!(again.len(), 1, "the same pick again writes no line");
}

#[test]
fn each_lineage_has_its_own_file() {
    let dir = tempfile::tempdir().unwrap();
    let other = "hipblaslt 1.2.3; hip 7.2; gfx1151; 1002:1586";
    let mut table = PlanTable::open(dir.path(), LINEAGE).unwrap();
    table.record(shape(4096), 17).unwrap();
    let new = PlanTable::open(dir.path(), other).unwrap();
    assert!(new.is_empty(), "a new lineage starts with no pick");
    assert_ne!(
        PlanTable::path_in(dir.path(), LINEAGE),
        PlanTable::path_in(dir.path(), other)
    );
}

#[test]
fn a_file_of_another_lineage_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut table = PlanTable::open(dir.path(), LINEAGE).unwrap();
    table.record(shape(4096), 17).unwrap();
    let path = PlanTable::path_in(dir.path(), LINEAGE);
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, text.replace("1.2.2", "9.9.9")).unwrap();
    match PlanTable::open(dir.path(), LINEAGE) {
        Err(PlanError::Lineage { file, .. }) => assert!(file.contains("9.9.9")),
        other => panic!("another lineage gave {other:?}"),
    }
}

#[test]
fn a_torn_or_a_malformed_line_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let mut table = PlanTable::open(dir.path(), LINEAGE).unwrap();
    table.record(shape(4096), 17).unwrap();
    let path = PlanTable::path_in(dir.path(), LINEAGE);
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, &text[..text.len() - 3]).unwrap();
    assert!(
        matches!(PlanTable::open(dir.path(), LINEAGE), Err(PlanError::Malformed { .. })),
        "torn"
    );
    std::fs::write(&path, format!("{text}not a pick\n")).unwrap();
    match PlanTable::open(dir.path(), LINEAGE) {
        Err(PlanError::Malformed { line, .. }) => assert_eq!(line, 4),
        other => panic!("a bad line gave {other:?}"),
    }
    std::fs::write(&path, format!("{text}{}", text.lines().nth(2).unwrap()) + "\n").unwrap();
    assert!(
        matches!(PlanTable::open(dir.path(), LINEAGE), Err(PlanError::Malformed { .. })),
        "a repeated shape"
    );
    std::fs::write(&path, text.replace("nagaya plan table v1", "a plan table")).unwrap();
    assert!(
        matches!(
            PlanTable::open(dir.path(), LINEAGE),
            Err(PlanError::Malformed { line: 1, .. })
        ),
        "magic"
    );
}
