//! Tests that need the test hook, so they live in the crate. They read the GGUF file as the other tests do.

use super::*;

const GGUF_VARIABLE: &str = "NAGAYA_QWEN38_GGUF";
const GGUF_DEFAULT: &str =
    "/tank/ml/models/llama.cpp/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";

/// The metadata of the model file. A missing file fails the test: the tests never skip.
fn metadata() -> &'static Metadata {
    static M: std::sync::OnceLock<Metadata> = std::sync::OnceLock::new();
    M.get_or_init(|| {
        let path = std::path::PathBuf::from(std::env::var(GGUF_VARIABLE).unwrap_or_else(|_| GGUF_DEFAULT.to_string()));
        assert!(
            path.is_file(),
            "the Qwen3.8 GGUF file {} does not exist. Put the file there, or set {GGUF_VARIABLE} to its path.",
            path.display()
        );
        Metadata::open(&path).unwrap_or_else(|e| panic!("cannot read the metadata: {e}"))
    })
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
    let id = base();
    assert_eq!(id, base());
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

#[test]
fn a_hook_that_changes_nothing_keeps_the_template_id() {
    assert_eq!(
        Qwen38Family::with_hook(metadata(), unchanged).unwrap().template_id(),
        base()
    );
}

#[test]
fn a_change_to_one_rendered_token_changes_the_council_hash_and_not_the_chat_hash() {
    let base = base();
    for (name, hook) in [("context", change_context as HookFn), ("raw", change_raw)] {
        let changed = Qwen38Family::with_hook(metadata(), hook).unwrap();
        assert_ne!(changed.template_id(), base, "a changed {name} token");
        assert_ne!(halves(changed.template_id()).0, halves(&base).0, "{name}");
        assert_eq!(halves(changed.template_id()).1, halves(&base).1, "{name}");
    }
}
