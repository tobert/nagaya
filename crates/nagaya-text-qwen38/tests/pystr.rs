//! The `str` methods of Python, in `pystr`, against Python 3.14.8.

mod common;

use common::{JsonExt, fixture};
use nagaya_text_qwen38::pystr;
use sha2::{Digest, Sha256};

fn put(h: &mut Sha256, text: &str) {
    h.update((text.len() as u32).to_le_bytes());
    h.update(text.as_bytes());
}

#[test]
fn every_code_point_gives_the_result_of_python_alone_and_in_context() {
    let want = fixture("jinja.json");
    let want = want.get("sweep");
    let mut h = Sha256::new();
    let mut swept = 0;
    for cp in 0..=0x10FFFFu32 {
        let Some(c) = char::from_u32(cp) else { continue };
        let s = c.to_string();
        put(&mut h, &pystr::lower(&s));
        put(&mut h, &pystr::upper(&s));
        put(&mut h, &pystr::title(&s));
        put(&mut h, &pystr::capitalize(&s));
        put(&mut h, &pystr::repr(&s));
        h.update([
            pystr::is_alpha(&s) as u8,
            pystr::is_digit(&s) as u8,
            pystr::is_space(&s) as u8,
            pystr::is_printable(&s) as u8,
        ]);
        for text in [
            format!("a{c}b"),
            format!("A\u{3a3}{c}"),
            format!("\u{3a3}{c}A"),
            format!("A{c}\u{3a3}"),
        ] {
            put(&mut h, &pystr::lower(&text));
            put(&mut h, &pystr::title(&text));
            put(&mut h, &pystr::capitalize(&text));
        }
        swept += 1;
    }
    assert_eq!(swept as i128, want.get("code_points").int());
    let digest: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(digest, want.get("sha256").str());
}

/// Results that the Python documentation and the Unicode standard give, not results of the generator.
#[test]
fn known_results_of_the_case_methods() {
    // SpecialCasing.txt: sharp s and the ligature ff go to two letters in upper case, and to a letter and a small
    // letter in title case.
    assert_eq!(pystr::upper("stra\u{df}e"), "STRASSE");
    assert_eq!(pystr::title("\u{df}"), "Ss");
    assert_eq!(pystr::upper("\u{fb00}"), "FF");
    assert_eq!(pystr::title("\u{fb00}"), "Ff");
    // SpecialCasing.txt: the dotted capital I goes to i and a combining dot above.
    assert_eq!(pystr::lower("\u{130}"), "i\u{307}");
    // Final_Sigma: a sigma at the end of a word is a final small sigma.
    assert_eq!(pystr::lower("\u{391}\u{3a3}"), "\u{3b1}\u{3c2}");
    assert_eq!(pystr::lower("\u{391}\u{3a3}\u{392}"), "\u{3b1}\u{3c3}\u{3b2}");
    assert_eq!(pystr::lower("\u{3a3}"), "\u{3c3}");
    assert_eq!(pystr::lower("A.\u{3a3}"), "a.\u{3c2}");
    // The digraph dz has its own title case form.
    assert_eq!(pystr::title("\u{1c6}"), "\u{1c5}");
    assert_eq!(pystr::upper("\u{1c6}"), "\u{1c4}");
    // The Python documentation: title() starts a word after any character that is not a letter.
    assert_eq!(
        pystr::title("they're bill's friends from the UK"),
        "They'Re Bill'S Friends From The Uk"
    );
    assert_eq!(pystr::capitalize("hELLO wORLD"), "Hello world");
    assert_eq!(pystr::capitalize("\u{1c6}ungla"), "\u{1c5}ungla");
    assert_eq!(pystr::capitalize(""), "");
}

#[test]
fn known_results_of_the_class_tests() {
    assert!(pystr::is_alpha("abc\u{e9}\u{4e16}") && !pystr::is_alpha("") && !pystr::is_alpha("ab1"));
    // isdigit accepts a superscript two and an Arabic-Indic digit. It does not accept a fraction or a CJK numeral.
    assert!(pystr::is_digit("123\u{b2}\u{663}") && !pystr::is_digit("") && !pystr::is_digit("\u{bd}"));
    assert!(!pystr::is_digit("\u{4e00}") && !pystr::is_digit("1.5"));
    assert!(pystr::is_space(" \t\n\u{1c}\u{3000}") && !pystr::is_space("") && !pystr::is_space(" a"));
}

#[test]
fn repr_follows_python() {
    assert_eq!(pystr::repr("abc"), "'abc'");
    assert_eq!(pystr::repr("it's"), "\"it's\"");
    assert_eq!(pystr::repr("say \"hi\" it's"), "'say \"hi\" it\\'s'");
    assert_eq!(pystr::repr("a\nb\tc\\d\re"), "'a\\nb\\tc\\\\d\\re'");
    assert_eq!(pystr::repr("\u{0}\u{7f}\u{80}\u{ad}"), "'\\x00\\x7f\\x80\\xad'");
    assert_eq!(pystr::repr("\u{e9}\u{4e16}\u{1f600}"), "'\u{e9}\u{4e16}\u{1f600}'");
    assert_eq!(pystr::repr("\u{200b}\u{2028}\u{fffe}"), "'\\u200b\\u2028\\ufffe'");
    assert_eq!(pystr::repr("\u{e0001}\u{10ffff}"), "'\\U000e0001\\U0010ffff'");
}
