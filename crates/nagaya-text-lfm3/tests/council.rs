//! The family interface of both LFM3 models (spec 11.3, 11.3a, 11.11): renders, thinking, boundaries, reads, media
//! positions, modes and the template id.

mod common;

use common::ids;
use common::model::{OMNI_VARIABLE, absent_any, absent_omni, dir, family, fixture, golden};
use gogi_proto::context::Context;
use gogi_proto::spec::{SpecRules, parse_spec};
use nagaya_council::family::{Advance, Asked, Continues, DecisionInput, Family};
use nagaya_text_lfm3::{Files, Kind, Lfm3Family, MediaKind, Mode};
use serde_json::{Value, json};

fn context(v: Value) -> Context {
    serde_json::from_value(v).expect("the test context is valid")
}

const CRAB: &str = "blake3:0000000000000000000000000000000000000000000000000000000000000001";

/// The sample of the chat fixture as a gogi context.
fn sample(thinking: bool) -> Context {
    context(
        json!({"thinking": thinking, "tools": [{"name": "shell", "description": "Run a shell command.",
        "input_schema": {"type": "object", "properties": {"command": {"type": "string", "description": "the command"},
                         "timeout": {"type": "integer"}}, "required": ["command"]}}],
        "messages": [
        {"role": "system", "content": "You are a careful coding agent."},
        {"role": "user", "content": "List the files, then say done.", "snap": true},
        {"role": "assistant", "content": "", "reasoning": "I should list them.",
         "tool_calls": [{"id": "c0", "name": "shell", "input": {"command": "ls -la", "timeout": 5}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "a.txt\nb.txt", "snap": true},
        {"role": "assistant", "content": "Two files: a.txt and b.txt.", "reasoning": "Both are listed.", "snap": true},
        {"role": "user", "content": "Thanks."}]}),
    )
}

fn chat_case(model: &str, name: &str) -> Vec<u32> {
    let f = fixture("chat.json");
    let case = f["cases"][model]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("no case {name}"))
        .clone();
    ids(&case["ids"])
}

#[test]
fn a_context_renders_as_the_template_renders_its_messages() {
    if absent_any() {
        return;
    }
    for model in ["omni", "text"] {
        let r = family(model).render(&sample(false), true).unwrap();
        assert_eq!(r.tokens, chat_case(model, "sample, thinking off"), "{model}");
        let closed = family(model).render(&sample(false), false).unwrap();
        assert_eq!(closed.tokens, chat_case(model, "sample, no prompt"), "{model}");
        assert_eq!(r.end, closed.tokens.len());
        assert!(r.tokens.starts_with(&closed.tokens));
    }
    let r = family("text").render(&sample(true), true).unwrap();
    assert_eq!(r.tokens, chat_case("text", "sample, thinking on"));
}

#[test]
fn the_boundaries_are_prefixes_of_the_rendering() {
    if absent_any() {
        return;
    }
    for model in ["omni", "text"] {
        let r = family(model).render(&sample(false), false).unwrap();
        let messages: Vec<usize> = r.boundaries.iter().map(|b| b.message).collect();
        assert_eq!(messages, [1, 3, 4, 5], "{model}");
        for b in &r.boundaries {
            let mut part = sample(false);
            part.messages.truncate(b.message + 1);
            assert_eq!(
                family(model).render(&part, false).unwrap().tokens,
                r.tokens[..b.tokens],
                "{model} {b:?}"
            );
        }
    }
}

/// Omni does not reason, so it renders a context with thinking on as with thinking off: one head for both (gogi
/// "thinking"). gogi's suite sends `thinking: true` to each served model.
#[test]
fn omni_renders_thinking_on_as_off_and_takes_an_absent_thinking_as_off() {
    if absent_any() {
        return;
    }
    for prompt in [false, true] {
        let on = family("omni")
            .render(&sample(true), prompt)
            .expect("Omni takes thinking: true");
        let off = family("omni")
            .render(&sample(false), prompt)
            .expect("Omni takes thinking: false");
        assert_eq!(on.tokens, off.tokens, "prompt {prompt}");
    }
    let plain = context(json!({"messages": [{"role": "user", "content": "Name three colors."}]}));
    let think = |model: &str| -> String {
        let r = family(model).render(&plain, true).unwrap();
        family(model).detokenize(&r.tokens).unwrap()
    };
    assert!(think("omni").ends_with("<|im_start|>assistant\n<think></think>"));
    assert!(think("text").ends_with("<|im_start|>assistant\n<think>"));
}

#[test]
fn the_chat_prompts_of_the_oracle_render_from_a_context() {
    if absent_any() {
        return;
    }
    let plain = context(json!({"messages": [{"role": "user", "content": "Name three colors."}]}));
    for (model, file) in [("omni", "lfm3-omni.oracle.json"), ("text", "lfm3.oracle.json")] {
        let Some(oracle) = golden(file) else {
            return;
        };
        let fam = family(model);
        for prompt in oracle["prompts"].as_array().unwrap() {
            let text = prompt["text"].as_str().unwrap();
            let want = ids(&prompt["ids"]);
            if prompt["name"] == "chat" {
                assert_eq!(fam.render(&plain, true).unwrap().tokens, want, "{model}");
            } else {
                // A prompt of plain text starts with <|startoftext|>, as the reference adds it.
                let bos = fam.tokenizer().token_id("<|startoftext|>").unwrap();
                assert_eq!(want[0], bos, "{model}");
                assert_eq!(fam.tokenize(text).unwrap(), want[1..], "{model} {}", prompt["name"]);
            }
        }
    }
}

#[test]
fn an_open_message_goes_on_from_the_generation_prompt() {
    if absent_any() {
        return;
    }
    for (model, thinking) in [("omni", false), ("text", false), ("text", true)] {
        let fam = family(model);
        let mut c = sample(thinking);
        let reply = json!({"role": "assistant", "content": "You are", "continue": true});
        let mut reply: gogi_proto::message::Message = serde_json::from_value(reply).unwrap();
        if thinking {
            reply.reasoning = Some("polite".to_string());
        }
        c.messages.push(reply.clone());
        let open = fam.render(&c, true).unwrap();
        assert_eq!(open.end, open.tokens.len(), "{model}");
        let mut before = c.clone();
        before.messages.pop();
        let prompt = fam.render(&before, true).unwrap().tokens;
        assert!(open.tokens.starts_with(&prompt), "{model} {thinking}");
        let text = fam.detokenize(&open.tokens[prompt.len()..]).unwrap();
        let want = if thinking { "polite</think>You are" } else { "You are" };
        assert_eq!(text, want, "{model}");
        // The closed message renders the same text up to its end.
        let mut closed = c.clone();
        closed.messages.last_mut().unwrap().open = false;
        let closed = fam.render(&closed, false).unwrap().tokens;
        assert!(closed.starts_with(&open.tokens), "{model} {thinking}");
    }
}

fn spec() -> gogi_proto::spec::Spec {
    parse_spec(
        &json!({"name": "g", "instructions": "i", "input_label": "x", "questions": [
            {"id": "verdict", "type": "choice", "instructions": "What. Pick one.", "criteria": [
                {"option": "allow", "means": "a"}, {"option": "ask", "means": "b"}, {"option": "no", "means": "c"}]},
            {"id": "undo", "type": "score", "instructions": "How", "criteria": ["easy", "hard"]}]}),
        &SpecRules::default(),
    )
    .unwrap()
}

#[test]
fn a_plain_read_closes_the_thinking_as_the_template_does() {
    if absent_any() {
        return;
    }
    let spec = spec();
    let asked = [Asked {
        question: 0,
        options: vec![0, 1, 2],
    }];
    for model in ["omni", "text"] {
        let fam = family(model);
        let c = context(json!({"thinking": false, "messages": [{"role": "user", "content": "hi"}]}));
        let rendering = fam.render(&c, false).unwrap();
        let input = DecisionInput {
            context: Some((&c, &rendering)),
            spec: &spec,
            case: "a case",
            asked: &asked,
        };
        let mut plan = fam.decision_read(&input).unwrap();
        let prefix = fam.detokenize(plan.prefix()).unwrap();
        assert!(
            prefix.ends_with("a case<|im_end|>\n<|im_start|>assistant\n<think></think>Answer to question"),
            "{model}: {prefix:?}"
        );
        let Advance::Step(step) = plan.advance(None).unwrap() else {
            panic!("a step first")
        };
        assert_eq!(step.continues, Continues::Prefix);
        assert_eq!(fam.detokenize(&step.feed).unwrap(), " 1 (What.):");
        assert_eq!(
            step.letters,
            fam.tokenize(" A")
                .unwrap()
                .iter()
                .chain(&fam.tokenize(" B").unwrap())
                .chain(&fam.tokenize(" C").unwrap())
                .copied()
                .collect::<Vec<_>>()
        );
        assert!(matches!(
            plan.advance(Some(&[-1.0, -2.0, -3.0])).unwrap(),
            Advance::Done(_)
        ));
        // A read with no context starts with <|startoftext|>.
        let none = DecisionInput { context: None, ..input };
        let plan = fam.decision_read(&none).unwrap();
        assert!(
            fam.detokenize(plan.prefix())
                .unwrap()
                .starts_with("<|startoftext|><|im_start|>user\n")
        );
    }
}

#[test]
fn a_queue_read_writes_the_call_in_the_xml_form() {
    if absent_any() {
        return;
    }
    let spec = spec();
    let asked = [
        Asked {
            question: 0,
            options: vec![0, 1, 2],
        },
        Asked {
            question: 1,
            options: vec![0, 1],
        },
    ];
    let tool = json!({"name": "next", "input_schema": {"type": "object", "properties": {"answers": {"type": "object",
        "properties": {"verdict": {"type": "string", "enum": ["A", "B", "C"]},
                       "undo": {"type": "string", "enum": ["A", "B"]}}}}}});
    let c = context(
        json!({"thinking": false, "tools": [tool], "queue": "next", "messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [{"id": "c0", "name": "next", "input": {}}]}]}),
    );
    for model in ["omni", "text"] {
        let fam = family(model);
        let rendering = fam.render(&c, false).unwrap();
        let input = DecisionInput {
            context: Some((&c, &rendering)),
            spec: &spec,
            case: "the case",
            asked: &asked,
        };
        let mut plan = fam.decision_read(&input).unwrap();
        let mut tokens = plan.prefix().to_vec();
        let mut last: Option<Vec<f64>> = None;
        loop {
            match plan.advance(last.as_deref()).unwrap() {
                Advance::Done(_) => break,
                Advance::Step(step) => {
                    assert_eq!(step.continues, Continues::PreviousStep);
                    tokens.extend(&step.feed);
                    last = Some((0..step.letters.len()).map(|i| -(i as f64) - 1.0).collect());
                }
            }
        }
        let text = fam.detokenize(&tokens).unwrap();
        assert!(
            text.ends_with(
                "<tool_response>\nthe case\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think></think>\
                 <tool_call>\n<function=next>\n<parameter=answers>\n{\"verdict\": \"A\", \"undo\": \""
            ),
            "{model}: {text:?}"
        );
    }
}

#[test]
fn a_raw_read_needs_one_token_for_each_letter() {
    if absent_any() {
        return;
    }
    let fam = family("text");
    let own = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let raw = fam.raw_read(&own(&["A", " B"]), &own(&["<think></think>x"])).unwrap();
    assert_eq!(raw.letters.len(), 2);
    let think = fam.tokenizer().token_id("<think>").unwrap();
    assert_eq!(raw.suffixes[0][0], think, "a suffix spells its markers");
    assert_eq!(raw.control_text.len(), 2);
    let e = fam
        .raw_read(&own(&["AB not one"]), &[])
        .expect_err("a letter of many tokens");
    assert!(e.is_client());
    let e = fam.raw_read(&own(&["A", "A"]), &[]).expect_err("two equal letters");
    assert!(e.is_client());
}

/// An Omni context with an image and a clip in the user message, and speech in an assistant message.
fn media_context() -> Context {
    context(json!({"messages": [
        {"role": "system", "content": "Respond with text."},
        {"role": "user", "content": [{"type": "object", "id": CRAB, "mime": "image/png"},
                                     {"type": "text", "text": "and"},
                                     {"type": "object", "id": CRAB, "mime": "audio/wav"}], "snap": true},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello"},
                                          {"type": "object", "id": CRAB, "mime": "audio/wav"}], "snap": true},
        {"role": "user", "content": "more"}]}))
}

#[test]
fn media_parts_take_the_positions_that_the_caller_gives() {
    if absent_any() {
        return;
    }
    let fam = family("omni");
    let c = media_context();
    let parts = fam.media_parts(&c).unwrap();
    let kinds: Vec<MediaKind> = parts.iter().map(|p| p.kind).collect();
    assert_eq!(kinds, [MediaKind::Image, MediaKind::Audio, MediaKind::AudioOut]);
    let small = fam.render_media(&c, true, &[1, 1, 1]).unwrap();
    let full = fam.render_media(&c, true, &[77, 5, 13]).unwrap();
    assert_eq!(full.rendering.tokens.len(), small.rendering.tokens.len() + 76 + 4 + 12);
    for (span, n) in full.spans.iter().zip([77, 5, 13]) {
        assert_eq!(span.count, n);
        assert!(
            full.rendering.tokens[span.first..span.first + n]
                .iter()
                .all(|&t| t == span.token)
        );
    }
    let vs = fam.tokenizer().token_id("<|vision_start|>").unwrap();
    let ve = fam.tokenizer().token_id("<|vision_end|>").unwrap();
    let image = &full.spans[0];
    assert_eq!(full.rendering.tokens[image.first - 1], vs);
    assert_eq!(full.rendering.tokens[image.first + image.count], ve);
    // The boundaries move with the positions.
    let b: Vec<(usize, usize)> = small
        .rendering
        .boundaries
        .iter()
        .map(|b| (b.message, b.tokens))
        .collect();
    let bf: Vec<(usize, usize)> = full
        .rendering
        .boundaries
        .iter()
        .map(|b| (b.message, b.tokens))
        .collect();
    assert_eq!(b.len(), bf.len());
    assert_eq!(bf[0], (b[0].0, b[0].1 + 76 + 4));
    assert_eq!(bf[1], (b[1].0, b[1].1 + 76 + 4 + 12));
    assert_eq!(full.rendering.end, small.rendering.end + 92);
    // A wrong count list is the caller's defect.
    assert!(!fam.render_media(&c, true, &[1, 1]).unwrap_err().is_client());
    assert!(!fam.render_media(&c, true, &[1, 0, 1]).unwrap_err().is_client());
}

#[test]
fn render_without_objects_and_the_text_model_refuse_media() {
    if absent_any() {
        return;
    }
    let c = media_context();
    for model in ["omni", "text"] {
        let e = family(model)
            .render(&c, true)
            .expect_err("media needs an object reader");
        match e {
            nagaya_council::family::FamilyError::Client(i) => {
                assert_eq!(i.param.as_deref(), Some("messages[1].content[0]"), "{model}")
            }
            other => panic!("{other:?}"),
        }
    }
    let gif = context(json!({"messages": [{"role": "user", "content": [
        {"type": "object", "id": CRAB, "mime": "image/gif"}]}]}));
    let e = family("omni").media_parts(&gif).expect_err("a gif is not read");
    assert!(e.to_string().contains("image/gif"), "{e}");
}

#[test]
fn the_media_prompts_of_the_oracle_have_the_positions_of_the_reference() {
    if absent_any() {
        return;
    }
    let Some(media) = golden("lfm3-omni.media.json") else {
        return;
    };
    let fam = family("omni");
    for prompt in media["prompts"].as_array().unwrap() {
        let want = ids(&prompt["ids"]);
        let name = prompt["name"].as_str().unwrap();
        let (mime, question) = if name.starts_with("image") {
            ("image/png", "Describe the image.")
        } else {
            ("audio/wav", "What do you hear?")
        };
        let kind = if mime == "audio/wav" {
            MediaKind::Audio
        } else {
            MediaKind::Image
        };
        let token = fam.tokenizer().token_id(kind.placeholder()).unwrap();
        let count = want.iter().filter(|&&t| t == token).count();
        let c = context(json!({"messages": [{"role": "user", "content": [
            {"type": "object", "id": CRAB, "mime": mime}, {"type": "text", "text": question}]}]}));
        let got = fam.render_media(&c, true, &[count]).unwrap();
        assert_eq!(got.rendering.tokens, want, "{name}");
    }
}

#[test]
fn the_modes_have_their_markers() {
    if absent_any() {
        return;
    }
    let omni = family("omni");
    let id = |t: &str| omni.tokenizer().token_id(t).unwrap();
    assert_eq!(omni.mode_marker(Mode::Text).unwrap(), None);
    assert_eq!(omni.mode_marker(Mode::Speech).unwrap(), Some(id("<|audio_start|>")));
    assert_eq!(
        omni.mode_marker(Mode::Interleaved).unwrap(),
        Some(id("<|mixed_start|>"))
    );
    assert_eq!(omni.text_suppressed(), [id("<|audio_start|>"), id("<|mixed_start|>")]);
    let e = family("text")
        .mode_marker(Mode::Speech)
        .expect_err("lfm3 does not speak");
    assert!(e.is_client());
    assert_eq!(family("omni").kind(), Kind::Omni);
    assert_eq!(family("text").kind(), Kind::Text);
    assert_eq!(family("omni").vocab_size(), 125_033);
}

#[test]
fn the_template_id_follows_the_tokens_of_the_test_renders() {
    if absent_omni() {
        return;
    }
    let files = Files::read(&dir(OMNI_VARIABLE)).unwrap();
    let id = family("omni").template_id().to_string();
    let (council, chat) = id.split_once("+lfm3-chat:").expect("the form of a template id");
    assert!(
        council.starts_with("nagaya-council:") && council.len() == 31 && chat.len() == 16,
        "{id}"
    );
    // A comment of the template changes the hash of the text, and no rendered token.
    let mut commented = files.clone();
    commented.template = format!("{{#- a comment -#}}{}", files.template);
    let other = Lfm3Family::new(&commented).unwrap();
    let (council2, chat2) = other.template_id().split_once("+lfm3-chat:").unwrap();
    assert_eq!(council2, council);
    assert_ne!(chat2, chat);
    // A change to a rendered token changes the council hash.
    let mut changed = files.clone();
    changed.template = files
        .template
        .replace("<|im_start|>assistant\\n", "<|im_start|>assistant \\n");
    assert_ne!(changed.template, files.template);
    let other = Lfm3Family::new(&changed).unwrap();
    assert_ne!(other.template_id().split_once("+lfm3-chat:").unwrap().0, council);
}
