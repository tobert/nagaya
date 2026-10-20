//! The character classes agree with the `regex` module of Python for every code point.

mod common;

use common::{JsonExt, fixture};
use nagaya_text_qwen38::chars;
use nagaya_text_qwen38::json::Json;

fn in_ranges(ranges: &Json, cp: u32) -> bool {
    ranges.items().iter().any(|r| {
        let r = r.items();
        (r[0].int() as u32..=r[1].int() as u32).contains(&cp)
    })
}

#[test]
fn every_code_point_has_the_class_that_python_gives() {
    let classes = fixture("classes.json");
    let tables: [(&str, fn(char) -> bool); 4] = [
        ("L", chars::is_letter),
        ("M", chars::is_mark),
        ("N", chars::is_number),
        ("s", chars::is_space),
    ];
    for (name, predicate) in tables {
        let ranges = classes.get(name);
        // Walk the code points once with a cursor over the sorted ranges, so that the test stays fast.
        let list: Vec<(u32, u32)> = ranges
            .items()
            .iter()
            .map(|r| (r.items()[0].int() as u32, r.items()[1].int() as u32))
            .collect();
        let mut wrong = Vec::new();
        let mut next = 0;
        for cp in 0..=0x10FFFFu32 {
            let Some(c) = char::from_u32(cp) else { continue };
            while next < list.len() && list[next].1 < cp {
                next += 1;
            }
            let expected = next < list.len() && list[next].0 <= cp;
            if predicate(c) != expected {
                wrong.push(format!("U+{cp:04X}"));
            }
        }
        assert!(
            wrong.is_empty(),
            "class {name} differs from Python at {} code points, first: {:?}",
            wrong.len(),
            &wrong[..wrong.len().min(10)]
        );
    }
}

#[test]
fn the_fixture_ranges_are_sorted_and_apart() {
    let classes = fixture("classes.json");
    for name in ["L", "M", "N", "s"] {
        let mut last_end: i128 = -2;
        for r in classes.get(name).items() {
            let (a, b) = (r.items()[0].int(), r.items()[1].int());
            assert!(
                a <= b && a > last_end + 1,
                "class {name}: range {a}..{b} after {last_end}"
            );
            last_end = b;
        }
    }
    assert!(in_ranges(classes.get("L"), 'a' as u32) && !in_ranges(classes.get("L"), '1' as u32));
}
