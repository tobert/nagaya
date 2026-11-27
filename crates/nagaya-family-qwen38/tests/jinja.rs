//! The behavior of Python's jinja2 that the chat template can reach: the methods of strings, the tests and the
//! undefined value. The fixtures are the outputs of the reference's `service/template.py`.

mod common;

use common::{JsonExt, fixture};
use nagaya_family_qwen38::json::Json;
use nagaya_family_qwen38::{ChatTemplate, Error, RenderOptions};
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
            (Ok(got), true) => wrong.push(format!(
                "{shown}: got {got:?}, the reference gives {:?}",
                case.get("out").str()
            )),
            (Ok(got), false) => wrong.push(format!(
                "{shown}: got {got:?}, the reference refuses with {}: {}",
                case.get("x").str(),
                case.get("error").str()
            )),
            (Err(e), true) => wrong.push(format!(
                "{shown}: refused ({e}), the reference gives {:?}",
                case.get("out").str()
            )),
            (Err(Error::Template(m)), false) => {
                let method = case.get("m").str();
                if !["direct", "join", "format", "test", "undefined"].contains(&method) && !m.contains(method) {
                    wrong.push(format!("{shown}: the refusal does not name the method {method}: {m}"));
                }
                // A message of raise_exception reaches the error as Python writes it.
                let said = case.get("error").str();
                if case.get("x").str() == "TemplateError" && !m.contains(said) {
                    wrong.push(format!("{shown}: the refusal should say {said:?}: {m}"));
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
        "{} of {total} cases differ from the reference. The first {}:\n{}",
        wrong.len(),
        wrong.len().min(400),
        wrong[..wrong.len().min(400)].join("\n")
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

#[test]
fn values_are_written_as_python_writes_them() {
    let cases = jinja().get("formats").items();
    assert!(cases.len() >= 290, "{} cases", cases.len());
    assert_none_wrong(check(cases), cases.len());
}

#[test]
fn the_concatenation_operator_writes_a_container_in_the_form_of_the_engine() {
    // Python writes ['x', 1.5]. The operator ~ of the engine writes its own form, and a template cannot change that.
    // The chat template joins strings and integers only. Spec 11.3 states the difference, and this test pins it.
    let messages = Json::parse(r#"[{"v": ["x", 1.5], "f": 1e16}]"#).unwrap();
    let render = |source: &str| {
        ChatTemplate::from_source(source)
            .unwrap()
            .render(&messages, None, &RenderOptions::default())
            .unwrap()
    };
    assert_eq!(render("{{ 'a' ~ messages[0].v }}"), "a[\"x\", 1.5]");
    assert_eq!(render("{{ 'a' ~ messages[0].f }}"), "a10000000000000000.0");
    // The output of a value, the string filter and raise_exception use the form of Python.
    assert_eq!(render("{{ messages[0].v }}"), "['x', 1.5]");
    assert_eq!(render("{{ messages[0].f|string }}"), "1e+16");
}

#[test]
fn a_tuple_of_the_template_is_a_list() {
    // The engine has no tuple type. Python writes (1, 2) for the tuple. JSON gives no tuple, so only a literal in the
    // template can differ.
    assert_eq!(render("{{ (1, 2) }}").unwrap(), "[1, 2]");
}

#[test]
fn tests_of_jinja2_give_its_answers() {
    let cases = jinja().get("tests").items();
    assert!(cases.len() >= 400, "{} cases", cases.len());
    assert_none_wrong(check(cases), cases.len());
}

#[test]
fn the_undefined_value_acts_as_it_does_in_jinja2() {
    let cases = jinja().get("undefined").items();
    assert!(cases.len() >= 90, "{} cases", cases.len());
    assert_none_wrong(check(cases), cases.len());
}

#[test]
fn known_differences_in_the_undefined_value_and_in_names_of_dict_methods() {
    // Python refuses these, or answers with a method. The engine answers in its own way. The chat template does
    // none of this, and spec 11.3 states the differences. The test pins them, so that a change is a choice.
    let messages = Json::parse(r#"[{"v": 1}]"#).unwrap();
    let render = |source: &str| {
        ChatTemplate::from_source(source).and_then(|t| t.render(&messages, None, &RenderOptions::default()))
    };
    // Python: UndefinedError. Here: zero and True.
    assert_eq!(
        render("{{ nothing|int }}|{{ nothing|float }}|{{ nothing < 1 }}").unwrap(),
        "0|0.0|True"
    );
    // Python: the first item of an empty value is an undefined value, and it writes nothing. Here: an error.
    assert!(render("{{ nothing|first }}").is_err());
    // Python: messages[0].items is the method items of the dict. Here: the key, or an undefined value.
    assert_eq!(render("[{{ messages[0].items }}]").unwrap(), "[]");
}
