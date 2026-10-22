//! Tests that need the test hook, so they live in the crate. They read the GGUF file as the other tests do.
//! A machine with no copy of the file skips them (see `tests/common/model.rs`).

use super::*;

#[path = "../../tests/common/model.rs"]
mod model;

/// The metadata of the model file. A test that calls it has called `model::absent` before.
fn metadata() -> &'static Metadata {
    static M: std::sync::OnceLock<Metadata> = std::sync::OnceLock::new();
    M.get_or_init(|| Metadata::open(&model::gguf_path()).unwrap_or_else(|e| panic!("cannot read the metadata: {e}")))
}

fn base() -> String {
    Qwen38Family::from_metadata(metadata())
        .unwrap()
        .template_id()
        .to_owned()
}

fn unchanged(_: Site, _: &mut [u32]) {}

/// Change one token of a rendering that ends with a generation prompt. A read plan checks that its context's
/// rendering is a prefix of the read, so the hook leaves the other renderings alone.
fn change_context(site: Site, tokens: &mut [u32]) {
    if site == Site::Context {
        tokens[0] += 1;
    }
}

fn change_prefix(site: Site, tokens: &mut [u32]) {
    if site == Site::Prefix {
        tokens[0] += 1;
    }
}

fn change_feed(site: Site, tokens: &mut [u32]) {
    if site == Site::Feed {
        tokens[0] += 1;
    }
}

fn change_raw(site: Site, tokens: &mut [u32]) {
    if site == Site::Raw && !tokens.is_empty() {
        tokens[0] += 1;
    }
}

fn halves(id: &str) -> (&str, &str) {
    let (council, chat) = id.split_once('+').expect("a template id has two parts");
    (council, chat)
}

#[test]
fn the_template_id_has_two_parts_and_is_stable() {
    if model::absent() {
        return;
    }
    let id = base();
    let (council, chat) = halves(&id);
    assert!(council.starts_with("nagaya-council:"), "{id}");
    assert!(chat.starts_with("qwen38-chat:"), "{id}");
    assert_eq!(council.len(), "nagaya-council:".len() + 16);
    assert_eq!(chat.len(), "qwen38-chat:".len() + 16);
    // The second hash is the digest of the chat template text of the GGUF.
    let Some(Value::Str(source)) = metadata().get("tokenizer.chat_template") else {
        panic!("the file has a chat template")
    };
    assert_eq!(chat, format!("qwen38-chat:{}", hash16(source.as_bytes())));
}

/// The template id of the test GGUF, pinned (gogi rule 6). Clients fit thresholds to an id, so a change to this value
/// is a deliberate id change. Change it only when the rendered tokens of the test set change on purpose, and say so
/// in the commit message. A change to a comment or to the code structure must not touch it.
const GOLDEN_TEMPLATE_ID: &str = "nagaya-council:b66d022efb4d4e48+qwen38-chat:12827f24b742ea4e";

#[test]
fn the_template_id_of_the_test_gguf_is_the_golden_id() {
    if model::absent() {
        return;
    }
    assert_eq!(base(), GOLDEN_TEMPLATE_ID);
}

#[test]
fn a_hook_that_changes_nothing_keeps_the_template_id() {
    if model::absent() {
        return;
    }
    assert_eq!(
        Qwen38Family::with_hook(metadata(), unchanged).unwrap().template_id(),
        base()
    );
}

#[test]
fn a_change_to_one_rendered_token_changes_the_council_hash_and_not_the_chat_hash() {
    if model::absent() {
        return;
    }
    let base = base();
    for (name, hook) in [
        ("context", change_context as HookFn),
        ("raw", change_raw),
        ("prefix", change_prefix),
        ("feed", change_feed),
    ] {
        let changed = Qwen38Family::with_hook(metadata(), hook).unwrap();
        assert_ne!(changed.template_id(), base, "a changed {name} token");
        assert_ne!(halves(changed.template_id()).0, halves(&base).0, "{name}");
        assert_eq!(halves(changed.template_id()).1, halves(&base).1, "{name}");
    }
}

/// Each hook changes the renders of the set that it should, and no others. The set is: eight renders of four
/// contexts (the odd ones have a generation prompt), three decisions, and the letters and two suffixes of a raw read.
#[test]
fn each_site_of_the_hook_changes_the_renders_that_use_it() {
    if model::absent() {
        return;
    }
    let base = test_set::renders(&Qwen38Family::from_metadata(metadata()).unwrap());
    assert_eq!(base.len(), 14);
    for (name, hook, want) in [
        ("context", change_context as HookFn, vec![1, 3, 5, 7]),
        ("prefix", change_prefix, vec![8, 9, 10]),
        ("feed", change_feed, vec![8, 9, 10]),
        ("raw", change_raw, vec![11, 12, 13]),
    ] {
        let changed = test_set::renders(&Qwen38Family::with_hook(metadata(), hook).unwrap());
        let got: Vec<usize> = (0..base.len()).filter(|&i| base[i] != changed[i]).collect();
        assert_eq!(got, want, "the renders that the {name} hook changes");
    }
}

/// The test set must keep its meaning, or a change to the code can hide a change to the id. The set holds a plain
/// read, a read after a context, a queue read and a raw read.
#[test]
fn the_test_set_holds_each_kind_of_read() {
    if model::absent() {
        return;
    }
    let family = Qwen38Family::from_metadata(metadata()).unwrap();
    let set = test_set::renders(&family);
    let text = |i: usize| family.tokenizer().decode(&set[i]).unwrap();
    assert!(text(8).starts_with("<|im_start|>user\ni\n\nAnswer each question"));
    assert!(text(8).contains("Answer to question 1 (What.):"));
    assert!(text(9).starts_with("<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nu<|im_end|>\n<|im_start|>user\ni"));
    assert!(text(10).contains("<tool_response>\na case\n</tool_response>"));
    assert!(text(10).contains("<parameter=answers>\n{\"undo\": \"AB"));
    assert!(text(13).contains("<tool_call>"));
}
