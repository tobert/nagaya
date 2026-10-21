//! Continuations (gogi "Continuations"): where a reply after an open message starts, and the build plan of an open
//! context.

use gogi_proto::context::Context;
use nagaya_council::continuation::seam_rendering;
use nagaya_council::family::{Boundary, FamilyError, Rendering, ReplyStart};
use serde_json::json;

fn context(thinking: Option<bool>, last: serde_json::Value) -> Context {
    let mut value = json!({"messages": [{"role": "user", "content": "u"}, last]});
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
    }
}

#[test]
fn an_open_context_builds_all_but_its_last_token_and_feeds_that_token_for_the_first_logit_row() {
    let seam = seam_rendering(&rendering(10, 10, &[(0, 4), (2, 10)])).unwrap();
    assert_eq!(seam.tokens, (1..=9).collect::<Vec<u32>>());
    assert_eq!(seam.end, 9);
    assert_eq!(
        seam.boundaries,
        [Boundary { message: 0, tokens: 4 }, Boundary { message: 2, tokens: 9 }],
        "the seam boundary takes the place of the end, with the index of the open message"
    );
    for bad in [
        rendering(10, 8, &[(0, 4), (2, 8)]),
        rendering(10, 10, &[(0, 9), (2, 10)]),
        rendering(1, 1, &[(0, 1)]),
    ] {
        assert!(
            matches!(seam_rendering(&bad), Err(FamilyError::Internal(_))),
            "{bad:?}: a family defect, not a client error"
        );
    }
}
