//! The tokenizer against the reference `CLIPTokenizer` (`tests/fixtures/ids.json`, from `tools/sdxl_family_fixture.py`).

mod common;

use common::{absent, family, fixture, ids};

#[test]
fn ids_match_the_first_reference_tokenizer() {
    if absent() {
        return;
    }
    let fixture = fixture("ids.json");
    let strings = fixture["strings"].as_array().expect("strings");
    assert!(strings.len() >= 25, "the fixture has too few strings");
    let mut wrong = Vec::new();
    for entry in strings {
        let text = entry["text"].as_str().expect("text");
        let want = ids(&entry["ids1"]);
        let got = family().tokenize(text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
        if got != want {
            wrong.push(format!("{text:?}: got {got:?}, the reference gives {want:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} strings differ:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The second reference tokenizer reads `!` as its pad id 0, and gives the ids of the first one for all other text.
/// The family gives both encoders the ids of the first one (spec 11.9a, "The family of SDXL").
#[test]
fn the_second_reference_tokenizer_differs_only_at_an_exclamation_mark() {
    let fixture = fixture("ids.json");
    let mut differ = 0;
    for entry in fixture["strings"].as_array().expect("strings") {
        let text = entry["text"].as_str().expect("text");
        let (first, second) = (ids(&entry["ids1"]), ids(&entry["ids2"]));
        if first != second {
            differ += 1;
            assert!(text.contains('!'), "{text:?}: {first:?} and {second:?}");
            assert!(second.contains(&0), "{text:?}: {second:?}");
        }
    }
    assert!(differ >= 2, "the fixture keeps strings with `!`");
}

#[test]
fn the_chunk_of_a_short_prompt_is_padded_as_the_reference_pads_it() {
    if absent() {
        return;
    }
    let fixture = fixture("ids.json");
    let padded = &fixture["padded"];
    let prompt = family()
        .prompt(padded["text"].as_str().expect("text"))
        .expect("a prompt");
    assert_eq!(prompt.chunks.len(), 1);
    assert_eq!(prompt.chunks[0].ids1.to_vec(), ids(&padded["ids1"]));
    assert_eq!(prompt.chunks[0].ids2.to_vec(), ids(&padded["ids2"]));
    assert!(prompt.chunks[0].weights.iter().all(|w| *w == 1.0));
}
