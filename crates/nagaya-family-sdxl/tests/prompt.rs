//! The A1111 prompt syntax (spec 11.9a, "The family of SDXL"): weights, escapes and `BREAK`. These cases follow the
//! syntax as A1111 documents it; the parser is not a port of A1111's code.

use nagaya_family_sdxl::{Piece, parse};

/// The runs of a prompt, with each weight rounded to 4 places.
fn runs(prompt: &str) -> Vec<(String, f32)> {
    parse(prompt)
        .into_iter()
        .map(|piece| match piece {
            Piece::Text { text, weight } => (text, (weight * 10_000.0).round() / 10_000.0),
            Piece::Break => ("BREAK".to_owned(), 0.0),
        })
        .collect()
}

fn run(text: &str, weight: f32) -> (String, f32) {
    (text.to_owned(), weight)
}

#[test]
fn plain_text_has_weight_one() {
    assert_eq!(runs("a red fox"), [run("a red fox", 1.0)]);
    assert_eq!(runs(""), Vec::<(String, f32)>::new());
}

#[test]
fn round_brackets_multiply_by_1_1_and_square_brackets_divide() {
    assert_eq!(runs("(fox)"), [run("fox", 1.1)]);
    assert_eq!(runs("((fox))"), [run("fox", 1.21)]);
    assert_eq!(runs("[fox]"), [run("fox", 0.9091)]);
    assert_eq!(runs("[[fox]]"), [run("fox", 0.8264)]);
}

#[test]
fn a_number_after_a_colon_gives_the_weight_of_its_group() {
    assert_eq!(runs("(fox:1.5)"), [run("fox", 1.5)]);
    assert_eq!(runs("((fox:1.5))"), [run("fox", 1.65)]);
    assert_eq!(runs("(fox: 0.5 )"), [run("fox", 0.5)]);
    assert_eq!(runs("(a:0.5) (b:1.5)"), [run("a", 0.5), run(" ", 1.0), run("b", 1.5)]);
}

#[test]
fn groups_nest_and_the_text_between_them_keeps_its_weight() {
    assert_eq!(
        runs("a (b [c] d) e"),
        [
            run("a ", 1.0),
            run("b ", 1.1),
            run("c", 1.0),
            run(" d", 1.1),
            run(" e", 1.0)
        ]
    );
}

#[test]
fn escaped_brackets_are_text() {
    assert_eq!(runs(r"\(fox\) \[cat\] a\\b"), [run(r"(fox) [cat] a\b", 1.0)]);
    assert_eq!(runs(r"a\nb"), [run(r"a\nb", 1.0)]);
}

#[test]
fn an_open_group_with_no_end_weights_the_rest_of_the_prompt() {
    assert_eq!(runs("a (fox"), [run("a ", 1.0), run("fox", 1.1)]);
    assert_eq!(runs("a [fox"), [run("a ", 1.0), run("fox", 0.9091)]);
}

#[test]
fn a_closing_bracket_with_no_group_is_text() {
    assert_eq!(runs("fox) cat]"), [run("fox) cat]", 1.0)]);
    assert_eq!(runs("[fox)"), [run("fox)", 0.9091)]);
    assert_eq!(runs("fox:1.2)"), [run("fox:1.2)", 1.0)]);
}

#[test]
fn a_colon_with_no_number_is_text() {
    assert_eq!(runs("(fox:big)"), [run("fox:big", 1.1)]);
}

#[test]
fn break_alone_as_a_word_ends_a_chunk() {
    assert_eq!(
        runs("a fox BREAK a cat"),
        [run("a fox", 1.0), run("BREAK", 0.0), run("a cat", 1.0)]
    );
    assert_eq!(runs("BREAKFAST break"), [run("BREAKFAST break", 1.0)]);
}

#[test]
fn equal_weights_next_to_each_other_join() {
    assert_eq!(runs("(a)(b)"), [run("ab", 1.1)]);
}
