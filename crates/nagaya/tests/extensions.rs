//! The text routes under `/nagaya/v1` in the process, with the fake engine and the toy family (spec 11.4b to 11.4f).

mod council_support;

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use council_support::Fixture;
use gogi_proto::identity::ServerIdentity;
use nagaya::http::{Council, Served, router};
use nagaya_council::family::Family;
use nagaya_family_toy::ToyFamily;
use serde_json::{Value, json};
use tower::ServiceExt;

/// A council on a fixture's process, with the model `toy` and the alias `tiny`.
fn council(f: &Fixture) -> Arc<Council> {
    Arc::new(
        Council::new(
            vec![Served {
                service: f.attach_again(),
                aliases: vec!["tiny".to_owned()],
                device: None,
            }],
            None,
        )
        .unwrap(),
    )
}

/// Send one request through the router. Give the status, the request-id header and the JSON body.
fn send(council: &Arc<Council>, method: &str, path: &str, body: Option<Value>) -> (StatusCode, String, Value) {
    send_bytes(council, method, path, body.map(|b| b.to_string()))
}

fn send_bytes(council: &Arc<Council>, method: &str, path: &str, body: Option<String>) -> (StatusCode, String, Value) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder().method(method).uri(path);
        let body = body.map_or_else(Body::empty, Body::from);
        let response = router(Arc::clone(council))
            .oneshot(request.body(body).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let id = response.headers()["request-id"].to_str().unwrap().to_owned();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 25).await.unwrap();
        (status, id, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

fn post(council: &Arc<Council>, route: &str, body: Value) -> (StatusCode, String, Value) {
    send(council, "POST", &format!("/nagaya/v1/{route}"), Some(body))
}

/// A context with a marked first message, a reply and a question.
fn context() -> Value {
    json!({"thinking": true, "messages": [
        {"role": "user", "content": "Say hi.", "snap": true},
        {"role": "assistant", "content": "Hi.", "snap": true},
        {"role": "user", "content": "Again."}]})
}

fn tokens_of(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|t| u32::try_from(t.as_u64().unwrap()).unwrap())
        .collect()
}

/// The error type and param of a refused request, after the checks that every error body has.
fn refused(result: (StatusCode, String, Value), status: StatusCode) -> (String, Option<String>) {
    let (code, id, body) = result;
    assert_eq!(code, status, "{body}");
    assert_eq!(body["request_id"], id, "the body repeats the request-id header");
    let error = &body["error"];
    (
        error["type"].as_str().unwrap().to_owned(),
        error["param"].as_str().map(str::to_owned),
    )
}

// ---------------------------------------------------------------- render

#[test]
fn render_gives_the_rendering_of_the_family_for_a_context() {
    let f = Fixture::new();
    let council = council(&f);
    let family = ToyFamily::new();
    let want = family
        .render(&serde_json::from_value(context()).unwrap(), false)
        .unwrap();
    let (status, id, body) = post(&council, "render", json!({"model": "toy", "context": context()}));
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(gogi_proto::ids::is_request_id(&id), "{id}");
    assert_eq!(body["model"], "toy");
    assert_eq!(body["template"], family.template_id());
    assert_eq!(tokens_of(&body["tokens"]), want.tokens);
    assert_eq!(body["n_tokens"], want.tokens.len());
    assert_eq!(body["end"], want.end);
    assert_eq!(body["text"], family.detokenize(&want.tokens).unwrap());
    let boundaries: Vec<(u64, u64)> = want
        .boundaries
        .iter()
        .map(|b| (b.message as u64, b.tokens as u64))
        .collect();
    let got: Vec<(u64, u64)> = body["boundaries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| (b["message"].as_u64().unwrap(), b["tokens"].as_u64().unwrap()))
        .collect();
    assert_eq!(got, boundaries);
    assert!(got.len() > 1, "the context has boundaries before its end");
    assert_eq!(body["signals"], json!({"control_text": []}));
}

#[test]
fn render_names_the_served_model_by_its_model_and_never_by_an_alias() {
    let f = Fixture::new();
    let council = council(&f);
    let (status, _, body) = post(&council, "render", json!({"model": "tiny", "context": context()}));
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "toy");
}

#[test]
fn render_adds_the_generation_prompt_only_when_asked() {
    let f = Fixture::new();
    let council = council(&f);
    let without = post(&council, "render", json!({"model": "toy", "context": context()})).2;
    let off = post(
        &council,
        "render",
        json!({"model": "toy", "context": context(), "generation_prompt": false}),
    )
    .2;
    let with = post(
        &council,
        "render",
        json!({"model": "toy", "context": context(), "generation_prompt": true}),
    )
    .2;
    assert_eq!(without, off, "false is the default");
    assert_eq!(without["end"], without["n_tokens"]);
    assert!(with["n_tokens"].as_u64() > without["n_tokens"].as_u64());
    assert_eq!(
        with["end"], without["end"],
        "`end` is the count before the generation prompt"
    );
    assert!(
        with["text"]
            .as_str()
            .unwrap()
            .starts_with(without["text"].as_str().unwrap())
    );
}

#[test]
fn render_counts_the_tokens_that_a_dry_run_of_a_build_counts() {
    let f = Fixture::new();
    let council = council(&f);
    let rendered = post(&council, "render", json!({"model": "toy", "context": context()})).2;
    let (status, _, built) = send(
        &council,
        "POST",
        "/council/v2/contexts",
        Some(json!({"model": "toy", "context": context(), "dry_run": true})),
    );
    assert_eq!(status, StatusCode::OK, "{built}");
    assert_eq!(built["usage"]["prompt"], rendered["n_tokens"]);
}

#[test]
fn render_reports_the_control_text_of_a_message_by_its_place() {
    let f = Fixture::new();
    let council = council(&f);
    let context = json!({"thinking": true, "messages": [
        {"role": "user", "content": "a <end> b"},
        {"role": "assistant", "content": "<asst>", "reasoning": "<think>"}]});
    let (status, _, body) = post(&council, "render", json!({"model": "toy", "context": context}));
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["signals"]["control_text"],
        json!([
            {"where": "message:0", "token": "<end>"},
            {"where": "message:1", "token": "<asst>"},
            {"where": "message:1", "token": "<think>"}])
    );
}

#[test]
fn render_does_not_check_the_context_limit() {
    let f = Fixture::new();
    let council = council(&f);
    let long = "x".repeat(5000);
    let context = json!({"thinking": true, "messages": [{"role": "user", "content": long}]});
    let (status, _, body) = post(&council, "render", json!({"model": "toy", "context": context}));
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["n_tokens"].as_u64().unwrap() > 4096);
}

#[test]
fn render_refuses_a_bad_request_with_gogis_error_body() {
    let f = Fixture::new();
    let council = council(&f);
    let bad = |body: Value| refused(post(&council, "render", body), StatusCode::BAD_REQUEST);
    let invalid = |param: Option<&str>| ("invalid_request".to_owned(), param.map(str::to_owned));
    assert_eq!(bad(json!({"context": context()})), invalid(Some("model")));
    assert_eq!(
        bad(json!({"model": "gpt-2", "context": context()})),
        invalid(Some("model"))
    );
    assert_eq!(bad(json!({"model": "toy"})), invalid(Some("context")));
    // The routes do not schedule, and the context has no reasoning effort.
    for member in ["class", "priority", "reasoning_effort", "messages"] {
        let mut body = json!({"model": "toy", "context": context()});
        body[member] = json!("x");
        assert_eq!(bad(body), invalid(Some(member)), "{member}");
    }
    assert_eq!(
        bad(json!({"model": "toy", "context": {"thinking": true, "messages": []}})),
        invalid(Some("context")),
        "a family error with no param names the context"
    );
    assert_eq!(
        bad(
            json!({"model": "toy", "context": {"thinking": true, "messages": [{"role": "user", "content": "x"}], "effort": 1}})
        ),
        invalid(Some("effort"))
    );
}

#[test]
fn a_model_that_is_not_ready_is_a_503_model_unavailable_on_every_text_route() {
    let f = Fixture::new();
    let council = council(&f);
    council.drain();
    let bodies = [
        ("render", json!({"model": "toy", "context": context()})),
        ("tokenize", json!({"model": "toy", "text": "x"})),
        ("detokenize", json!({"model": "toy", "tokens": [1]})),
    ];
    for (route, body) in bodies {
        let (kind, _) = refused(post(&council, route, body), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(kind, "model_unavailable", "{route}");
    }
}

#[test]
fn a_text_route_takes_only_post_and_a_route_under_the_prefix_that_is_not_built_is_a_404() {
    let f = Fixture::new();
    let council = council(&f);
    for route in ["render", "tokenize", "detokenize"] {
        let (kind, _) = refused(
            send(&council, "GET", &format!("/nagaya/v1/{route}"), None),
            StatusCode::NOT_FOUND,
        );
        assert_eq!(kind, "not_found", "{route}");
    }
    let (kind, _) = refused(post(&council, "embeddings", json!({})), StatusCode::NOT_FOUND);
    assert_eq!(kind, "not_found");
}

#[test]
fn a_body_past_16_mib_is_a_413_too_large_that_names_the_body() {
    let f = Fixture::new();
    let council = council(&f);
    let text = "x".repeat(17 << 20);
    let body = json!({"model": "toy", "text": text}).to_string();
    let result = send_bytes(&council, "POST", "/nagaya/v1/tokenize", Some(body));
    assert_eq!(
        refused(result, StatusCode::PAYLOAD_TOO_LARGE),
        ("too_large".to_owned(), Some("body".to_owned()))
    );
}

// ---------------------------------------------------------------- tokenize and detokenize

#[test]
fn tokenize_gives_the_tokens_and_their_count_and_pieces_only_when_asked() {
    let f = Fixture::new();
    let council = council(&f);
    let (status, id, body) = post(&council, "tokenize", json!({"model": "tiny", "text": "Hi <end>"}));
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(gogi_proto::ids::is_request_id(&id));
    let want = ToyFamily::new().tokenize("Hi <end>").unwrap();
    assert_eq!(tokens_of(&body["tokens"]), want);
    assert_eq!(body["n_tokens"], want.len());
    assert_eq!(body["model"], "toy");
    assert!(body.get("pieces").is_none(), "{body}");
    assert!(body.get("text").is_none(), "{body}");
    // The marker text is bytes without `special`.
    assert_eq!(want.len(), 8);
    let (_, _, body) = post(
        &council,
        "tokenize",
        json!({"model": "toy", "text": "Hi", "pieces": true}),
    );
    assert_eq!(body["pieces"], json!(["H", "i"]));
}

#[test]
fn tokenize_with_special_turns_the_text_of_a_marker_into_its_token() {
    let f = Fixture::new();
    let council = council(&f);
    let (status, _, body) = post(
        &council,
        "tokenize",
        json!({"model": "toy", "text": "a<end>", "special": true, "pieces": true}),
    );
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["n_tokens"], 2);
    assert_eq!(body["pieces"], json!(["a", "<end>"]));
    let want = ToyFamily::new().tokenize_special("a<end>").unwrap();
    assert_eq!(tokens_of(&body["tokens"]), want);
    let (_, _, plain) = post(
        &council,
        "tokenize",
        json!({"model": "toy", "text": "a<end>", "special": false}),
    );
    assert_eq!(plain["n_tokens"], 6);
}

#[test]
fn an_empty_text_gives_no_tokens_and_no_tokens_give_an_empty_text() {
    let f = Fixture::new();
    let council = council(&f);
    let (status, _, body) = post(
        &council,
        "tokenize",
        json!({"model": "toy", "text": "", "pieces": true}),
    );
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["tokens"], json!([]));
    assert_eq!(body["n_tokens"], 0);
    assert_eq!(body["pieces"], json!([]));
    let (status, _, body) = post(&council, "detokenize", json!({"model": "toy", "tokens": []}));
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, json!({"model": "toy", "text": ""}));
}

#[test]
fn detokenize_undoes_tokenize_and_render() {
    let f = Fixture::new();
    let council = council(&f);
    let text = "Say hi, caf\u{e9} \u{65e5}\u{672c}";
    let (_, _, tokenized) = post(&council, "tokenize", json!({"model": "toy", "text": text}));
    let (status, _, body) = post(
        &council,
        "detokenize",
        json!({"model": "tiny", "tokens": tokenized["tokens"]}),
    );
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, json!({"model": "toy", "text": text}));
    // The text of a rendering is the detokenized tokens of the rendering.
    let (_, _, rendered) = post(&council, "render", json!({"model": "toy", "context": context()}));
    let (_, _, body) = post(
        &council,
        "detokenize",
        json!({"model": "toy", "tokens": rendered["tokens"]}),
    );
    assert_eq!(body["text"], rendered["text"]);
}

#[test]
fn detokenize_refuses_bad_tokens_with_the_param_tokens() {
    let f = Fixture::new();
    let council = council(&f);
    let vocab = ToyFamily::new().vocab_size();
    let want = ("invalid_request".to_owned(), Some("tokens".to_owned()));
    let bad = |tokens: Value| {
        refused(
            post(&council, "detokenize", json!({"model": "toy", "tokens": tokens})),
            StatusCode::BAD_REQUEST,
        )
    };
    assert_eq!(bad(json!([65, vocab])), want, "an id past the vocabulary");
    assert_eq!(bad(json!([0xc3])), want, "the first byte of a character");
    assert_eq!(bad(json!([-1])), want, "a negative id");
    assert_eq!(bad(json!([4_294_967_296_u64])), want, "an id past 32 bits");
    assert_eq!(bad(json!(["a"])), want, "an id that is text");
    assert_eq!(bad(json!([1.5])), want, "an id that is not whole");
}

#[test]
fn tokenize_and_detokenize_refuse_a_bad_request_with_gogis_error_body() {
    let f = Fixture::new();
    let council = council(&f);
    let bad = |route: &str, body: Value| refused(post(&council, route, body), StatusCode::BAD_REQUEST);
    let invalid = |param: &str| ("invalid_request".to_owned(), Some(param.to_owned()));
    assert_eq!(bad("tokenize", json!({"text": "x"})), invalid("model"));
    assert_eq!(bad("tokenize", json!({"model": "toy"})), invalid("text"));
    assert_eq!(bad("tokenize", json!({"model": "nope", "text": "x"})), invalid("model"));
    assert_eq!(bad("detokenize", json!({"tokens": []})), invalid("model"));
    assert_eq!(
        bad("detokenize", json!({"model": "nope", "tokens": []})),
        invalid("model")
    );
    assert_eq!(bad("detokenize", json!({"model": "toy"})), invalid("tokens"));
    // `special` and `pieces` belong to `tokenize` only, and the routes do not schedule.
    assert_eq!(
        bad("detokenize", json!({"model": "toy", "tokens": [], "special": true})),
        invalid("special")
    );
    assert_eq!(
        bad("detokenize", json!({"model": "toy", "tokens": [], "pieces": true})),
        invalid("pieces")
    );
    assert_eq!(
        bad("tokenize", json!({"model": "toy", "text": "x", "tokens": []})),
        invalid("tokens")
    );
    assert_eq!(
        bad("tokenize", json!({"model": "toy", "text": "x", "class": "bulk"})),
        invalid("class")
    );
}

#[test]
fn a_token_that_ends_inside_a_character_has_the_replacement_character_as_its_piece() {
    let f = Fixture::new();
    let council = council(&f);
    let (_, _, body) = post(
        &council,
        "tokenize",
        json!({"model": "toy", "text": "\u{e9}", "pieces": true}),
    );
    assert_eq!(body["pieces"], json!(["\u{fffd}", "\u{fffd}"]));
}

// ---------------------------------------------------------------- identity

#[test]
fn identity_lists_the_prefix_if_and_only_if_the_text_routes_answer() {
    let f = Fixture::new();
    let council = council(&f);
    let (_, _, body) = send(&council, "GET", "/council/v2/identity", None);
    let identity: ServerIdentity = serde_json::from_value(body).unwrap();
    assert_eq!(identity.extensions, ["/nagaya/v1"]);
    let (status, _, _) = post(&council, "tokenize", json!({"model": "toy", "text": "x"}));
    assert_eq!(status, StatusCode::OK);
    // A process with no served model holds no family, so it lists nothing.
    let empty = Arc::new(Council::new(Vec::new(), None).unwrap());
    let (_, _, body) = send(&empty, "GET", "/council/v2/identity", None);
    let identity: ServerIdentity = serde_json::from_value(body).unwrap();
    assert!(identity.extensions.is_empty(), "{:?}", identity.extensions);
}

/// The toy family reads text only, so the render route refuses an object part by name before the family sees it
/// (gogi "Contexts"). A family that got the part would panic, and a panic stops the process.
#[test]
fn render_refuses_an_object_part_by_name_and_the_process_answers_after() {
    let f = Fixture::new();
    let council = council(&f);
    let id = format!("blake3:{}", "a".repeat(64));
    let with_object = json!({"thinking": true, "messages": [
        {"role": "user", "content": "Say hi."},
        {"role": "user", "content": [{"type": "text", "text": "Look:"}, {"type": "object", "id": id, "mime": "image/png"}]}]});
    let (kind, param) = refused(
        post(&council, "render", json!({"model": "toy", "context": with_object})),
        StatusCode::BAD_REQUEST,
    );
    assert_eq!(
        (kind.as_str(), param.as_deref()),
        ("invalid_request", Some("messages[1].content[1]"))
    );
    let (status, _, _) = post(&council, "render", json!({"model": "toy", "context": context()}));
    assert_eq!(status, StatusCode::OK);
}
