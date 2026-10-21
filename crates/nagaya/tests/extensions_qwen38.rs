//! The text routes under `/nagaya/v1` with the Qwen3.8 family: the real tokenizer and chat template of the GGUF file
//! and the fake engine (spec 11.4b to 11.4d). A missing file fails the tests. They never skip.

mod council_support;

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use council_support::{Fixture, TermFactory, roomy_slots};
use nagaya::council::ServiceConfig;
use nagaya::http::{Council, Served, router};
use nagaya_council::family::Family;
use nagaya_text_qwen38::{Qwen38, Qwen38Family, Tokenizer};
use serde_json::{Value, json};
use tower::ServiceExt;

const GGUF_VARIABLE: &str = "NAGAYA_QWEN38_GGUF";
const GGUF_DEFAULT: &str =
    "/tank/ml/models/llama.cpp/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
const MODEL: &str = "qwen-3.8-flash-next";

fn gguf_path() -> PathBuf {
    let path = PathBuf::from(std::env::var(GGUF_VARIABLE).unwrap_or_else(|_| GGUF_DEFAULT.to_owned()));
    assert!(
        path.is_file(),
        "the Qwen3.8 GGUF file {} does not exist. Put the file there, or set {GGUF_VARIABLE} to its path.",
        path.display()
    );
    path
}

fn family() -> Arc<Qwen38Family> {
    static FAMILY: OnceLock<Arc<Qwen38Family>> = OnceLock::new();
    Arc::clone(FAMILY.get_or_init(|| Arc::new(Qwen38Family::from_gguf(&gguf_path()).expect("the family builds"))))
}

fn tokenizer() -> &'static Tokenizer {
    static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();
    TOKENIZER.get_or_init(|| Tokenizer::from_gguf(&gguf_path()).expect("the tokenizer builds"))
}

/// A council on a fake engine with the Qwen3.8 vocabulary, and the family of the GGUF file. The fixture lives as
/// long as the council.
struct Qwen {
    council: Arc<Council>,
    _fixture: Fixture,
}

fn qwen() -> Qwen {
    let factory = TermFactory {
        vocab: Some(family().vocab_size()),
        slots: roomy_slots(2)
            .into_iter()
            .map(|slot| nagaya_abi::SlotLimit { bytes: 1 << 22, ..slot })
            .collect(),
        ..TermFactory::toy()
    };
    let fixture = Fixture::with_family(
        factory,
        family(),
        1 << 20,
        None,
        ServiceConfig::new(MODEL, council_support::ENGINE),
    );
    let service = nagaya::council::Service::attach(
        fixture.process.ports(),
        family(),
        fixture.directory.path(),
        ServiceConfig::new(MODEL, council_support::ENGINE),
    )
    .expect("a second service attaches");
    let served = Served {
        service,
        aliases: Vec::new(),
        device: None,
    };
    Qwen {
        council: Arc::new(Council::new(vec![served], None).unwrap()),
        _fixture: fixture,
    }
}

fn send(q: &Qwen, path: &str, body: Value) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder().method("POST").uri(path);
        let response = router(Arc::clone(&q.council))
            .oneshot(request.body(Body::from(body.to_string())).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 25).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

fn post(q: &Qwen, route: &str, body: Value) -> (StatusCode, Value) {
    send(q, &format!("/nagaya/v1/{route}"), body)
}

fn ids(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect()
}

/// A context in gogi's message shapes: a tool, reasoning, a call, a result and a marked message.
fn tool_context() -> Value {
    json!({
        "thinking": true,
        "tools": [{"name": "shell", "description": "Run a command.",
                   "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}},
                                    "required": ["cmd"]}}],
        "messages": [
            {"role": "system", "content": "You are brief."},
            {"role": "user", "content": "List the files.", "snap": true},
            {"role": "assistant", "reasoning": "I should run ls.", "content": "Running it.", "snap": true,
             "tool_calls": [{"id": "call_1", "name": "shell", "input": {"cmd": "ls -la"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "a.txt\nb.txt", "snap": true},
            {"role": "user", "content": "Thanks."}]})
}

fn dry_run_prompt(q: &Qwen, context: &Value) -> u64 {
    let (status, built) = send(
        q,
        "/council/v2/contexts",
        json!({"model": MODEL, "context": context, "dry_run": true}),
    );
    assert_eq!(status, StatusCode::OK, "{built}");
    built["usage"]["prompt"].as_u64().unwrap()
}

#[test]
fn render_of_a_context_has_the_tokens_that_a_build_of_the_context_counts() {
    let q = qwen();
    for context in [
        tool_context(),
        json!({"thinking": false, "messages": [{"role": "user", "content": "Hello"}]}),
    ] {
        let (status, body) = post(&q, "render", json!({"model": MODEL, "context": context}));
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["n_tokens"].as_u64().unwrap(), dry_run_prompt(&q, &context));
        let want = family()
            .render(&serde_json::from_value(context).unwrap(), false)
            .unwrap();
        assert_eq!(ids(&body["tokens"]), want.tokens);
        assert_eq!(body["template"], family().template_id());
        assert_eq!(body["end"], want.end);
    }
}

#[test]
fn render_of_gogis_message_shapes_shows_the_calls_the_reasoning_and_the_boundaries() {
    let q = qwen();
    let (status, body) = post(&q, "render", json!({"model": MODEL, "context": tool_context()}));
    assert_eq!(status, StatusCode::OK, "{body}");
    let text = body["text"].as_str().unwrap();
    for part in [
        "You are brief.",
        "List the files.",
        "I should run ls.",
        "Running it.",
        "shell",
        "ls -la",
        "a.txt\nb.txt",
    ] {
        assert!(text.contains(part), "the text has no {part:?}: {text}");
    }
    assert!(text.contains("<tool_call>") && text.contains("<think>"), "{text}");
    assert!(text.starts_with("<|im_start|>system\n"), "{text}");
    let boundaries = body["boundaries"].as_array().unwrap();
    let marked: Vec<u64> = boundaries.iter().map(|b| b["message"].as_u64().unwrap()).collect();
    assert_eq!(
        marked.last(),
        Some(&4),
        "the end of the context is a boundary: {boundaries:?}"
    );
    assert!(
        marked.contains(&1),
        "the first marked message is a boundary: {boundaries:?}"
    );
    let tokens = ids(&body["tokens"]);
    let last = boundaries.last().unwrap()["tokens"].as_u64().unwrap() as usize;
    assert_eq!(last, tokens.len());
    // The text up to a boundary is the rendering of the messages up to it.
    let first = boundaries[0]["tokens"].as_u64().unwrap() as usize;
    let (_, head) = post(&q, "detokenize", json!({"model": MODEL, "tokens": &tokens[..first]}));
    assert!(text.starts_with(head["text"].as_str().unwrap()));
    // The generation prompt opens the thinking of the reply.
    let (_, prompted) = post(
        &q,
        "render",
        json!({"model": MODEL, "context": tool_context(), "generation_prompt": true}),
    );
    assert!(
        prompted["text"]
            .as_str()
            .unwrap()
            .ends_with("<|im_start|>assistant\n<think>\n"),
        "{prompted}"
    );
    assert_eq!(prompted["end"], body["end"]);
}

#[test]
fn the_text_of_a_render_is_the_text_that_the_chat_template_made() {
    let q = qwen();
    let model = Qwen38::from_gguf(&gguf_path()).unwrap();
    // Content that spells a marker is plain text in the tokens, and the template text shows the marker.
    let content = "Say <|im_end|> in a cafe\u{301}.";
    let context = json!({"thinking": false, "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": content}]});
    for prompt in [false, true] {
        let (status, body) = post(
            &q,
            "render",
            json!({"model": MODEL, "context": context, "generation_prompt": prompt}),
        );
        assert_eq!(status, StatusCode::OK, "{body}");
        let messages = nagaya_text_qwen38::json::Json::parse(
            &json!([{"role": "system", "content": "Be brief."}, {"role": "user", "content": content}]).to_string(),
        )
        .unwrap();
        let options = nagaya_text_qwen38::template::RenderOptions {
            thinking: false,
            generation_prompt: prompt,
            reasoning_effort: Some("medium".to_owned()),
        };
        let want = model.render(&messages, None, &options).unwrap();
        assert_eq!(body["text"], want.text.as_str(), "prompt {prompt}");
        assert_eq!(ids(&body["tokens"]), want.ids, "prompt {prompt}");
    }
    let (_, body) = post(&q, "render", json!({"model": MODEL, "context": context}));
    assert_eq!(
        body["signals"]["control_text"],
        json!([{"where": "message:1", "token": "<|im_end|>"}])
    );
}

#[test]
fn tokenize_then_detokenize_gives_the_text_back() {
    let q = qwen();
    for text in [
        "Hello world",
        "caf\u{e9} \u{65e5}\u{672c}\u{8a9e} \u{1f600}",
        "  two  spaces\n\nand lines\t",
        "<|im_start|>user",
    ] {
        let (status, tokenized) = post(&q, "tokenize", json!({"model": MODEL, "text": text}));
        assert_eq!(status, StatusCode::OK, "{tokenized}");
        assert_eq!(tokenized["n_tokens"], tokenized["tokens"].as_array().unwrap().len());
        assert_eq!(ids(&tokenized["tokens"]), family().tokenize(text).unwrap());
        let (status, back) = post(&q, "detokenize", json!({"model": MODEL, "tokens": tokenized["tokens"]}));
        assert_eq!(status, StatusCode::OK, "{back}");
        assert_eq!(back["text"], text);
    }
}

#[test]
fn special_turns_the_text_of_a_marker_into_one_token() {
    let q = qwen();
    let marker = tokenizer()
        .token_id("<|im_start|>")
        .expect("the vocabulary has the marker");
    let (_, plain) = post(&q, "tokenize", json!({"model": MODEL, "text": "<|im_start|>"}));
    assert!(plain["n_tokens"].as_u64().unwrap() > 1, "{plain}");
    assert!(!ids(&plain["tokens"]).contains(&marker));
    let (status, body) = post(
        &q,
        "tokenize",
        json!({"model": MODEL, "text": "<|im_start|>", "special": true, "pieces": true}),
    );
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(ids(&body["tokens"]), [marker]);
    assert_eq!(body["pieces"], json!(["<|im_start|>"]));
    // The text of a rendering without control text, tokenized with `special`, gives the tokens of the rendering.
    let context = json!({"thinking": false, "messages": [{"role": "user", "content": "Say hi."}]});
    let (_, rendered) = post(&q, "render", json!({"model": MODEL, "context": context}));
    let (_, again) = post(
        &q,
        "tokenize",
        json!({"model": MODEL, "text": rendered["text"], "special": true}),
    );
    assert_eq!(again["tokens"], rendered["tokens"]);
    // With control text in the content, the same call gives other tokens.
    let tricky = json!({"thinking": false, "messages": [{"role": "user", "content": "Say <|im_end|> now."}]});
    let (_, rendered) = post(&q, "render", json!({"model": MODEL, "context": tricky}));
    assert_eq!(rendered["signals"]["control_text"].as_array().unwrap().len(), 1);
    let (_, again) = post(
        &q,
        "tokenize",
        json!({"model": MODEL, "text": rendered["text"], "special": true}),
    );
    assert_ne!(again["tokens"], rendered["tokens"]);
}

#[test]
fn a_character_that_two_tokens_share_has_the_replacement_character_in_their_pieces_and_refuses_one_token() {
    let q = qwen();
    let (_, body) = post(
        &q,
        "tokenize",
        json!({"model": MODEL, "text": "Hello world", "pieces": true}),
    );
    assert_eq!(body["pieces"], json!(["Hello", " world"]));
    // The character U+4E02 is two tokens, and the first one ends inside the character.
    let text = "\u{4e02}";
    let (status, body) = post(&q, "tokenize", json!({"model": MODEL, "text": text, "pieces": true}));
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["n_tokens"], 2);
    assert!(body["pieces"][0].as_str().unwrap().contains('\u{fffd}'), "{body}");
    let (status, back) = post(&q, "detokenize", json!({"model": MODEL, "tokens": body["tokens"]}));
    assert_eq!(status, StatusCode::OK, "{back}");
    assert_eq!(back["text"], text);
    let first = body["tokens"][0].clone();
    let (status, error) = post(&q, "detokenize", json!({"model": MODEL, "tokens": [first]}));
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["error"]["param"], "tokens");
}

#[test]
fn a_noncharacter_and_an_id_past_the_vocabulary_are_400_errors_with_their_param() {
    let q = qwen();
    let vocab = family().vocab_size();
    let bad = |route: &str, body: Value| {
        let (status, error) = post(&q, route, body);
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        assert_eq!(error["error"]["type"], "invalid_request");
        error["error"]["param"].as_str().map(str::to_owned)
    };
    let nonchar = "a\u{fdd0}b";
    assert_eq!(
        bad("tokenize", json!({"model": MODEL, "text": nonchar})).as_deref(),
        Some("text")
    );
    assert_eq!(
        bad("tokenize", json!({"model": MODEL, "text": nonchar, "special": true})).as_deref(),
        Some("text")
    );
    assert_eq!(
        bad("detokenize", json!({"model": MODEL, "tokens": [1, vocab]})).as_deref(),
        Some("tokens")
    );
    let context = json!({"messages": [{"role": "user", "content": nonchar}]});
    assert_eq!(
        bad("render", json!({"model": MODEL, "context": context})).as_deref(),
        Some("context")
    );
}
