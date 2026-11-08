//! Continuations (gogi "Continuations"): where a reply after an open message starts, and the build plan of an open
//! context.

use gogi_proto::context::Context;
use nagaya_council::continuation::open_plan;
use nagaya_council::family::{Boundary, FamilyError, MediaSpan, Rendering, ReplyStart};
use serde_json::json;

fn context(thinking: Option<bool>, last: serde_json::Value) -> Context {
    let mut value = json!({"thinking": true, "messages": [{"role": "user", "content": "u"}, last]});
    if let Some(thinking) = thinking {
        value["thinking"] = json!(thinking);
    }
    serde_json::from_value(value).unwrap()
}

#[test]
fn a_reply_starts_in_the_phase_that_its_open_message_sets() {
    let closed = context(Some(true), json!({"role": "user", "content": "v"}));
    assert_eq!(
        ReplyStart::of(&closed),
        ReplyStart {
            thinking: true,
            open: false
        }
    );
    let closed = context(Some(false), json!({"role": "user", "content": "v"}));
    assert_eq!(
        ReplyStart::of(&closed),
        ReplyStart {
            thinking: false,
            open: false
        }
    );
    let thought = context(
        Some(true),
        json!({"role": "assistant", "reasoning": "plan", "continue": true}),
    );
    assert_eq!(
        ReplyStart::of(&thought),
        ReplyStart {
            thinking: true,
            open: true
        }
    );
    let empty = json!({"role": "assistant", "reasoning": "plan", "content": "", "continue": true});
    assert_eq!(
        ReplyStart::of(&context(Some(true), empty)),
        ReplyStart {
            thinking: false,
            open: true
        },
        "a content member closes the thinking, even when it is empty"
    );
    let content = json!({"role": "assistant", "content": "Plan:", "continue": true});
    assert_eq!(
        ReplyStart::of(&context(None, content.clone())),
        ReplyStart {
            thinking: false,
            open: true
        }
    );
    assert_eq!(
        ReplyStart::of(&context(Some(false), content)),
        ReplyStart {
            thinking: false,
            open: true
        }
    );
}

fn rendering(len: u32, end: usize, boundaries: &[(usize, usize)]) -> Rendering {
    Rendering {
        tokens: (1..=len).collect(),
        end,
        boundaries: boundaries
            .iter()
            .map(|&(message, tokens)| Boundary { message, tokens })
            .collect(),
        control_text: Vec::new(),
        media: Vec::new(),
    }
}

/// The open context renders 10 tokens: a marked message to token 4, the message before the open one to token 7, and
/// the open message to token 10.
#[test]
fn an_open_context_builds_the_context_before_its_open_message() {
    let open = rendering(10, 10, &[(0, 4), (2, 10)]);
    let before = rendering(7, 7, &[(0, 4), (1, 7)]);
    let plan = open_plan(&open, &before).unwrap();
    assert_eq!(plan.tokens, (1..=7).collect::<Vec<u32>>());
    assert_eq!(plan.end, 7);
    assert_eq!(
        plan.boundaries, before.boundaries,
        "the end of the context before the open message is a boundary"
    );
}

#[test]
fn a_context_before_the_open_message_that_is_not_a_prefix_keeps_only_the_boundaries_of_the_open_rendering() {
    let open = rendering(10, 10, &[(0, 4), (2, 10)]);
    let mut before = rendering(7, 7, &[(0, 4), (1, 7)]);
    before.tokens[6] = 99;
    let plan = open_plan(&open, &before).unwrap();
    assert_eq!((plan.tokens.len(), plan.end), (4, 4));
    assert_eq!(plan.boundaries, [Boundary { message: 0, tokens: 4 }]);
    let alone = rendering(10, 10, &[(1, 10)]);
    let plan = open_plan(&alone, &before).unwrap();
    assert_eq!(
        (plan.tokens.len(), plan.end, plan.boundaries.len()),
        (0, 0, 0),
        "nothing to build: feed it all"
    );
}

#[test]
fn an_open_rendering_with_a_generation_prompt_is_a_family_defect() {
    let before = rendering(7, 7, &[(1, 7)]);
    for bad in [
        rendering(10, 8, &[(2, 8)]),
        rendering(10, 10, &[(2, 9)]),
        rendering(10, 10, &[(0, 10), (2, 10)]),
    ] {
        assert!(
            matches!(open_plan(&bad, &before), Err(FamilyError::Internal(_))),
            "{bad:?}"
        );
    }
}

/// An image of 2 positions at `first`.
fn image(first: usize) -> MediaSpan {
    MediaSpan {
        first,
        count: 2,
        mime: "image/png".to_owned(),
        id: format!("blake3:{}", "c".repeat(64)),
    }
}

/// The plan keeps the media spans of the tokens that it builds, so its recipes name them (spec 11.11).
#[test]
fn the_plan_of_an_open_context_keeps_the_media_spans_that_it_builds() {
    let mut open = rendering(10, 10, &[(0, 4), (2, 10)]);
    open.media = vec![image(1), image(5)];
    let mut before = rendering(7, 7, &[(0, 4), (1, 7)]);
    before.media = vec![image(1), image(5)];
    assert_eq!(open_plan(&open, &before).unwrap().media, before.media);
    // The plan that is not the context before the open message ends at token 4, so it holds only the first span.
    before.tokens[6] = 99;
    assert_eq!(open_plan(&open, &before).unwrap().media, [image(1)]);
}
