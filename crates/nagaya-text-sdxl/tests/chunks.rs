//! The chunks of a prompt (spec 11.9a, "The family of SDXL"): 75 tokens with a start and an end token, at most 3.

use nagaya_text_sdxl::{BOS, CHUNK_TOKENS, EOS, Error, MAX_CHUNKS, PAD2, Piece, chunks};

/// A tokenizer for the tests: each character is one token, with the id of its code point.
fn chars(text: &str) -> Result<Vec<u32>, Error> {
    Ok(text.chars().map(u32::from).collect())
}

fn text(text: &str, weight: f32) -> Piece {
    Piece::Text {
        text: text.to_owned(),
        weight,
    }
}

#[test]
fn a_short_prompt_is_one_padded_chunk() {
    let got = chunks(&[text("ab", 1.5)], chars).unwrap();
    assert_eq!(got.len(), 1);
    let chunk = &got[0];
    assert_eq!(&chunk.ids1[..4], &[BOS, 97, 98, EOS]);
    assert!(
        chunk.ids1[4..].iter().all(|id| *id == EOS),
        "the first encoder pads with the end id"
    );
    assert_eq!(&chunk.ids2[..4], &[BOS, 97, 98, EOS]);
    assert!(
        chunk.ids2[4..].iter().all(|id| *id == PAD2),
        "the second encoder pads with 0"
    );
    assert_eq!(&chunk.weights[..4], &[1.0, 1.5, 1.5, 1.0]);
    assert!(chunk.weights[4..].iter().all(|w| *w == 1.0));
}

#[test]
fn an_empty_prompt_is_one_chunk_of_a_start_and_an_end() {
    let got = chunks(&[], chars).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(&got[0].ids1[..2], &[BOS, EOS]);
}

#[test]
fn a_long_prompt_fills_chunks_of_75_tokens() {
    let long = "x".repeat(CHUNK_TOKENS + 5);
    let got = chunks(&[text(&long, 1.0)], chars).unwrap();
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].ids1[0], BOS);
    assert!(got[0].ids1[1..=CHUNK_TOKENS].iter().all(|id| *id == 120));
    assert_eq!(got[0].ids1[CHUNK_TOKENS + 1], EOS);
    assert_eq!(&got[1].ids1[..7], &[BOS, 120, 120, 120, 120, 120, EOS]);
}

#[test]
fn break_starts_a_new_chunk_and_a_break_at_a_chunk_start_does_nothing() {
    let got = chunks(&[text("a", 1.0), Piece::Break, text("b", 1.0)], chars).unwrap();
    assert_eq!(got.len(), 2);
    assert_eq!(&got[1].ids1[..3], &[BOS, 98, EOS]);
    let got = chunks(&[Piece::Break, text("a", 1.0), Piece::Break, Piece::Break], chars).unwrap();
    assert_eq!(got.len(), 1);
}

#[test]
fn a_prompt_past_three_chunks_is_refused() {
    let long = "x".repeat(CHUNK_TOKENS * MAX_CHUNKS + 1);
    match chunks(&[text(&long, 1.0)], chars) {
        Err(Error::TooLong { tokens, .. }) => assert_eq!(tokens, CHUNK_TOKENS * MAX_CHUNKS + 1),
        other => panic!("expected a refusal, got {other:?}"),
    }
    let breaks = [
        text("a", 1.0),
        Piece::Break,
        text("b", 1.0),
        Piece::Break,
        text("c", 1.0),
        Piece::Break,
        text("d", 1.0),
    ];
    assert!(matches!(chunks(&breaks, chars), Err(Error::TooLong { .. })));
}
