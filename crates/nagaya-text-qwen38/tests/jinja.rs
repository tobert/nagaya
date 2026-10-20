//! The behavior of Python's jinja2 that the chat template can reach: the methods of strings, the tests and the
//! undefined value. The fixtures are the outputs of mk's `service/template.py`.

mod common;

use common::{JsonExt, fixture};
use nagaya_text_qwen38::json::Json;
use nagaya_text_qwen38::{ChatTemplate, Error, RenderOptions};
use std::sync::OnceLock;

fn jinja() -> &'static Json {
    static F: OnceLock<Json> = OnceLock::new();
    F.get_or_init(|| fixture("jinja.json"))
}

/// The source and the messages of a case. A call case has the receiver in `s` and the arguments in `a`.
fn case_template(case: &Json) -> (String, Json) {
    if case.has("src") {
        let messages = if case.has("msgs") {
            case.get("msgs").clone()
        } else {
            Json::Array(vec![])
        };
        return (case.get("src").str().to_string(), messages);
    }
    let args = case.get("a").items();
    let list: Vec<String> = (0..args.len()).map(|i| format!("messages[0].a{i}")).collect();
    let source = format!(
        "{{{{ messages[0].s.{}({})|tojson }}}}",
        case.get("m").str(),
        list.join(", ")
    );
    let mut object = vec![("s".to_string(), case.get("s").clone())];
    object.extend(args.iter().enumerate().map(|(i, a)| (format!("a{i}"), a.clone())));
    (source, Json::Array(vec![Json::Object(object)]))
}

/// Run each case and collect the differences, so that one run shows all of them.
fn check(cases: &[Json]) -> Vec<String> {
    let mut wrong = Vec::new();
    for case in cases {
        let (source, messages) = case_template(case);
        let result =
            ChatTemplate::from_source(&source).and_then(|t| t.render(&messages, None, &RenderOptions::default()));
        let shown = format!("{source} with {messages:?}");
        match (result, case.has("out")) {
            (Ok(got), true) if got == case.get("out").str() => {}
            (Ok(got), true) => wrong.push(format!("{shown}: got {got:?}, mk gives {:?}", case.get("out").str())),
            (Ok(got), false) => wrong.push(format!(
                "{shown}: got {got:?}, mk refuses with {}: {}",
                case.get("x").str(),
                case.get("error").str()
            )),
            (Err(e), true) => wrong.push(format!("{shown}: refused ({e}), mk gives {:?}", case.get("out").str())),
            (Err(Error::Template(m)), false) => {
                let method = case.get("m").str();
                if !["direct", "join"].contains(&method) && !m.contains(method) {
                    wrong.push(format!("{shown}: the refusal does not name the method {method}: {m}"));
                }
            }
            (Err(other), false) => wrong.push(format!("{shown}: wrong kind of error {other:?}")),
        }
    }
    wrong
}

fn assert_none_wrong(wrong: Vec<String>, total: usize) {
    assert!(
        wrong.is_empty(),
        "{} of {total} cases differ from mk. The first {}:\n{}",
        wrong.len(),
        wrong.len().min(25),
        wrong[..wrong.len().min(25)].join("\n")
    );
}

#[test]
fn string_methods_give_the_results_of_python() {
    let cases = jinja().get("methods").items();
    assert!(cases.len() >= 2000, "{} cases", cases.len());
    assert_none_wrong(check(cases), cases.len());
}

fn render(source: &str) -> Result<String, Error> {
    ChatTemplate::from_source(source)?.render(&Json::Array(vec![]), None, &RenderOptions::default())
}

#[test]
fn a_method_outside_the_list_is_an_error_that_names_it() {
    // Python renders each of these. The library refuses them, and says which method failed.
    for (source, method) in [
        ("{{ 'abc'.rfind('b') }}", "rfind"),
        ("{{ 'abc'.index('b') }}", "index"),
        ("{{ 'abc'.format() }}", "format"),
        ("{{ 'abc'.center(9) }}", "center"),
        ("{{ 'abc'.zfill(9) }}", "zfill"),
        ("{{ 'abc'.ljust(9) }}", "ljust"),
        ("{{ 'abc'.swapcase() }}", "swapcase"),
        ("{{ 'abc'.casefold() }}", "casefold"),
        ("{{ 'abc'.encode() }}", "encode"),
        ("{{ 'abc'.partition('b') }}", "partition"),
        ("{{ 'a\\nb'.splitlines() }}", "splitlines"),
        ("{{ 'abc'.isalnum() }}", "isalnum"),
        ("{{ 'abc'.islower() }}", "islower"),
        ("{{ 'abc'.rsplit('b') }}", "rsplit"),
        ("{{ 'abc'.nonsense() }}", "nonsense"),
    ] {
        match render(source) {
            Err(Error::Template(m)) => {
                assert!(m.contains(&format!("{method:?}")), "{source}: {m}");
                assert!(
                    m.contains("startswith"),
                    "{source}: the message should list the methods: {m}"
                );
            }
            other => panic!("{source}: {other:?}"),
        }
    }
}

#[test]
fn a_method_of_another_kind_of_value_is_an_error_that_names_it() {
    for (source, method) in [
        ("{{ [1, 2].count(1) }}", "count"),
        ("{{ [1, 2].append(3) }}", "append"),
        ("{{ {'a': 1}.keys() }}", "keys"),
        ("{{ {'a': 1}.get('a') }}", "get"),
        ("{{ (5).bit_length() }}", "bit_length"),
        ("{{ messages.pop() }}", "pop"),
    ] {
        match render(source) {
            Err(Error::Template(m)) => assert!(m.contains(&format!("{method:?}")), "{source}: {m}"),
            other => panic!("{source}: {other:?}"),
        }
    }
}

#[test]
fn a_list_stands_for_a_tuple_in_startswith_and_endswith() {
    // The engine has no tuple type. Python refuses a list here, and this library takes it as a tuple. The difference
    // is in the note of spec 11.3. The test pins it, so that a change is a choice.
    assert_eq!(render("{{ 'abc'.startswith(['x', 'a']) }}").unwrap(), "True");
    assert_eq!(render("{{ 'abc'.endswith(['x', 'c']) }}").unwrap(), "True");
}
