//! The character classes agree with the `regex` module of Python for every code point.

mod common;

use common::{JsonExt, fixture};
use nagaya_text_qwen38::chars;
use nagaya_text_qwen38::json::Json;

type Class = fn(char) -> bool;

fn in_ranges(ranges: &Json, cp: u32) -> bool {
    ranges.items().iter().any(|r| {
        let r = r.items();
        (r[0].int() as u32..=r[1].int() as u32).contains(&cp)
    })
}

#[test]
fn every_code_point_has_the_class_that_python_gives() {
    let classes = fixture("classes.json");
    let tables: [(&str, Class); 5] = [
        ("L", chars::is_letter),
        ("M", chars::is_mark),
        ("N", chars::is_number),
        ("s", chars::is_space),
        ("isspace", chars::is_python_space),
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
    for name in ["L", "M", "N", "s", "isspace"] {
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

/// Code points with answers that come from the Unicode Character Database, not from Python or from the tables.
/// Each row is a code point, the answer, and the property that gives the answer. The fixture test above is the
/// full check. These rows stay true if the generator and the tables share one mistake.
struct Known {
    class: &'static str,
    predicate: Class,
    // (code point, property)
    members: &'static [(u32, &'static str)],
    others: &'static [(u32, &'static str)],
}

const KNOWN: [Known; 5] = [
    Known {
        class: "L (General_Category Lu, Ll, Lt, Lm, Lo)",
        predicate: chars::is_letter,
        members: &[
            (0x41, "Lu LATIN CAPITAL LETTER A"),
            (0x7A, "Ll LATIN SMALL LETTER Z"),
            (0xAA, "Lo FEMININE ORDINAL INDICATOR"),
            (0xB5, "Ll MICRO SIGN"),
            (0xC0, "Lu LATIN CAPITAL LETTER A WITH GRAVE"),
            (0xDF, "Ll LATIN SMALL LETTER SHARP S"),
            (0x1C5, "Lt LATIN CAPITAL LETTER D WITH SMALL LETTER Z WITH CARON"),
            (0x2B0, "Lm MODIFIER LETTER SMALL H"),
            (0x3A9, "Lu GREEK CAPITAL LETTER OMEGA"),
            (0x416, "Lu CYRILLIC CAPITAL LETTER ZHE"),
            (0x5D0, "Lo HEBREW LETTER ALEF"),
            (0x627, "Lo ARABIC LETTER ALEF"),
            (0x905, "Lo DEVANAGARI LETTER A"),
            (0xE01, "Lo THAI CHARACTER KO KAI"),
            (0x3042, "Lo HIRAGANA LETTER A"),
            (0x4E16, "Lo CJK UNIFIED IDEOGRAPH-4E16"),
            (0xAC00, "Lo HANGUL SYLLABLE GA"),
            (0xFF21, "Lu FULLWIDTH LATIN CAPITAL LETTER A"),
            (0x1D400, "Lu MATHEMATICAL BOLD CAPITAL A"),
            (0x20000, "Lo CJK UNIFIED IDEOGRAPH-20000"),
        ],
        others: &[
            (0x30, "Nd DIGIT ZERO"),
            (0x5F, "Pc LOW LINE"),
            (0xD7, "Sm MULTIPLICATION SIGN"),
            (0x301, "Mn COMBINING ACUTE ACCENT"),
            (0xB2, "No SUPERSCRIPT TWO"),
            (0x2160, "Nl ROMAN NUMERAL ONE"),
            (0x1F600, "So GRINNING FACE"),
            (0xE000, "Co PRIVATE USE AREA"),
            (0xD800, "Cs HIGH SURROGATE (no scalar value)"),
            (0x378, "Cn unassigned since Unicode 1.1"),
        ],
    },
    Known {
        class: "M (General_Category Mn, Mc, Me)",
        predicate: chars::is_mark,
        members: &[
            (0x300, "Mn COMBINING GRAVE ACCENT"),
            (0x36F, "Mn COMBINING LATIN SMALL LETTER X"),
            (0x483, "Mn COMBINING CYRILLIC TITLO"),
            (0x488, "Me COMBINING CYRILLIC HUNDRED THOUSANDS SIGN"),
            (0x591, "Mn HEBREW ACCENT ETNAHTA"),
            (0x64B, "Mn ARABIC FATHATAN"),
            (0x903, "Mc DEVANAGARI SIGN VISARGA"),
            (0x93E, "Mc DEVANAGARI VOWEL SIGN AA"),
            (0xA83, "Mc GUJARATI SIGN VISARGA"),
            (0xE31, "Mn THAI CHARACTER MAI HAN-AKAT"),
            (0xE47, "Mn THAI CHARACTER MAITAIKHU"),
            (0x1AB0, "Mn COMBINING DOUBLED CIRCUMFLEX ACCENT"),
            (0x20D0, "Mn COMBINING LEFT HARPOON ABOVE"),
            (0x20DD, "Me COMBINING ENCLOSING CIRCLE"),
            (0x302A, "Mn IDEOGRAPHIC LEVEL TONE MARK"),
            (0x3099, "Mn COMBINING KATAKANA-HIRAGANA VOICED SOUND MARK"),
            (0xFE00, "Mn VARIATION SELECTOR-1"),
            (0xFE0F, "Mn VARIATION SELECTOR-16"),
            (0x1D165, "Mc MUSICAL SYMBOL COMBINING STEM"),
            (0xE0100, "Mn VARIATION SELECTOR-17"),
        ],
        others: &[
            (0x61, "Ll LATIN SMALL LETTER A"),
            (0xB4, "Sk ACUTE ACCENT"),
            (0xE32, "Lo THAI CHARACTER SARA AA"),
            (0x200D, "Cf ZERO WIDTH JOINER"),
            (0x2028, "Zl LINE SEPARATOR"),
            (0x1F3FB, "Sk EMOJI MODIFIER FITZPATRICK TYPE-1-2"),
            (0xFE10, "Po PRESENTATION FORM FOR VERTICAL COMMA"),
            (0x3000, "Zs IDEOGRAPHIC SPACE"),
        ],
    },
    Known {
        class: "N (General_Category Nd, Nl, No)",
        predicate: chars::is_number,
        members: &[
            (0x30, "Nd DIGIT ZERO"),
            (0x39, "Nd DIGIT NINE"),
            (0xB2, "No SUPERSCRIPT TWO"),
            (0xB9, "No SUPERSCRIPT ONE"),
            (0xBC, "No VULGAR FRACTION ONE QUARTER"),
            (0x660, "Nd ARABIC-INDIC DIGIT ZERO"),
            (0x6F0, "Nd EXTENDED ARABIC-INDIC DIGIT ZERO"),
            (0x966, "Nd DEVANAGARI DIGIT ZERO"),
            (0xE50, "Nd THAI DIGIT ZERO"),
            (0x2070, "No SUPERSCRIPT ZERO"),
            (0x2080, "No SUBSCRIPT ZERO"),
            (0x2155, "No VULGAR FRACTION ONE FIFTH"),
            (0x2160, "Nl ROMAN NUMERAL ONE"),
            (0x2460, "No CIRCLED DIGIT ONE"),
            (0x3007, "Nl IDEOGRAPHIC NUMBER ZERO"),
            (0x3021, "Nl HANGZHOU NUMERAL ONE"),
            (0xFF10, "Nd FULLWIDTH DIGIT ZERO"),
            (0x1D7CE, "Nd MATHEMATICAL BOLD DIGIT ZERO"),
            (0x1F100, "No DIGIT ZERO FULL STOP"),
            (0x10107, "No AEGEAN NUMBER ONE"),
        ],
        others: &[
            (0x61, "Ll LATIN SMALL LETTER A"),
            (0x4E00, "Lo CJK UNIFIED IDEOGRAPH-4E00 (a letter, though it means one)"),
            (
                0x4E09,
                "Lo CJK UNIFIED IDEOGRAPH-4E09 (a letter, though it means three)",
            ),
            (0x23, "Po NUMBER SIGN"),
            (0x221E, "Sm INFINITY"),
            (0x2212, "Sm MINUS SIGN"),
            (0x1F600, "So GRINNING FACE"),
        ],
    },
    Known {
        class: "s (the White_Space property, which is the \\s of the pattern)",
        predicate: chars::is_space,
        members: &[
            (0x09, "White_Space CHARACTER TABULATION"),
            (0x0A, "White_Space LINE FEED"),
            (0x0B, "White_Space LINE TABULATION"),
            (0x0C, "White_Space FORM FEED"),
            (0x0D, "White_Space CARRIAGE RETURN"),
            (0x20, "White_Space SPACE"),
            (0x85, "White_Space NEXT LINE"),
            (0xA0, "White_Space NO-BREAK SPACE"),
            (0x1680, "White_Space OGHAM SPACE MARK"),
            (0x2000, "White_Space EN QUAD"),
            (0x2005, "White_Space FOUR-PER-EM SPACE"),
            (0x2007, "White_Space FIGURE SPACE"),
            (0x200A, "White_Space HAIR SPACE"),
            (0x2028, "White_Space LINE SEPARATOR"),
            (0x2029, "White_Space PARAGRAPH SEPARATOR"),
            (0x202F, "White_Space NARROW NO-BREAK SPACE"),
            (0x205F, "White_Space MEDIUM MATHEMATICAL SPACE"),
            (0x3000, "White_Space IDEOGRAPHIC SPACE"),
        ],
        others: &[
            (0x1C, "not White_Space: INFORMATION SEPARATOR FOUR, bidi class B"),
            (0x1F, "not White_Space: INFORMATION SEPARATOR ONE, bidi class S"),
            (0x08, "not White_Space: BACKSPACE"),
            (0x0E, "not White_Space: SHIFT OUT"),
            (0x21, "not White_Space: EXCLAMATION MARK"),
            (
                0x180E,
                "not White_Space since Unicode 6.3: MONGOLIAN VOWEL SEPARATOR, Cf",
            ),
            (0x200B, "not White_Space: ZERO WIDTH SPACE, Cf"),
            (0x2060, "not White_Space: WORD JOINER, Cf"),
            (0xFEFF, "not White_Space: ZERO WIDTH NO-BREAK SPACE, Cf"),
            (0xAD, "not White_Space: SOFT HYPHEN, Cf"),
        ],
    },
    Known {
        class: "isspace (bidirectional class WS, B or S, or General_Category Zs)",
        predicate: chars::is_python_space,
        members: &[
            (0x09, "bidi S CHARACTER TABULATION"),
            (0x0A, "bidi B LINE FEED"),
            (0x0B, "bidi S LINE TABULATION"),
            (0x0C, "bidi WS FORM FEED"),
            (0x0D, "bidi B CARRIAGE RETURN"),
            (0x1C, "bidi B INFORMATION SEPARATOR FOUR"),
            (0x1D, "bidi B INFORMATION SEPARATOR THREE"),
            (0x1E, "bidi B INFORMATION SEPARATOR TWO"),
            (0x1F, "bidi S INFORMATION SEPARATOR ONE"),
            (0x20, "Zs SPACE"),
            (0x85, "bidi B NEXT LINE"),
            (0xA0, "Zs NO-BREAK SPACE"),
            (0x1680, "Zs OGHAM SPACE MARK"),
            (0x2003, "Zs EM SPACE"),
            (0x2007, "Zs FIGURE SPACE"),
            (0x2028, "bidi WS LINE SEPARATOR"),
            (0x2029, "bidi B PARAGRAPH SEPARATOR"),
            (0x202F, "Zs NARROW NO-BREAK SPACE"),
            (0x205F, "Zs MEDIUM MATHEMATICAL SPACE"),
            (0x3000, "Zs IDEOGRAPHIC SPACE"),
        ],
        others: &[
            (0x00, "Cc NULL, bidi BN"),
            (0x08, "Cc BACKSPACE, bidi BN"),
            (0x0E, "Cc SHIFT OUT, bidi BN"),
            (0x21, "Po EXCLAMATION MARK"),
            (0x7F, "Cc DELETE, bidi BN"),
            (0x180E, "Cf MONGOLIAN VOWEL SEPARATOR, bidi BN since Unicode 6.3"),
            (0x200B, "Cf ZERO WIDTH SPACE, bidi BN"),
            (0x2060, "Cf WORD JOINER, bidi BN"),
            (0xFEFF, "Cf ZERO WIDTH NO-BREAK SPACE, bidi BN"),
            (0xAD, "Cf SOFT HYPHEN, bidi BN"),
        ],
    },
];

#[test]
fn hand_picked_code_points_have_the_class_that_the_unicode_database_gives() {
    for known in &KNOWN {
        for (set, expected) in [(known.members, true), (known.others, false)] {
            for &(cp, property) in set {
                // A surrogate is not a char. The classes hold no surrogate, so the row checks only that.
                let Some(c) = char::from_u32(cp) else {
                    assert!(!expected, "U+{cp:04X} is not a char");
                    continue;
                };
                assert_eq!(
                    (known.predicate)(c),
                    expected,
                    "class {}: U+{cp:04X} ({property})",
                    known.class
                );
            }
        }
    }
}
