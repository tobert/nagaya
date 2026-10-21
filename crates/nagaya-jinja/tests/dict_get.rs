//! The method `get` of a dict, which Google's Gemma 4 template calls (spec 11.3). The expected text is the output of
//! transformers 5.20.2 with jinja2 3.1.6, run on 2026-10-21.

use nagaya_jinja::{Json, Template};

fn render(source: &str) -> Result<String, String> {
    let d = Json::Object(vec![
        ("a".to_string(), Json::Int(1)),
        ("n".to_string(), Json::Null),
        ("s".to_string(), Json::Str("t".to_string())),
    ]);
    let template = Template::from_source(source).map_err(|e| e.to_string())?;
    template.render(&[("d", d)]).map_err(|e| e.to_string())
}

#[test]
fn get_gives_what_python_gives() {
    let cases = [
        (r#"{{ d.get("a") }}"#, "1"),
        (r#"{{ d.get("b") }}"#, "None"),
        (r#"{{ d.get("b", "x") }}"#, "x"),
        (r#"{{ d.get("b") is none }}"#, "True"),
        (r#"{{ d.get("n", 5) }}"#, "None"),
        (r#"{{ d.get("s") ~ "!" }}"#, "t!"),
        (r#"{{ d.get("b") or "dflt" }}"#, "dflt"),
        (r#"{{ d.get("b") | default("z") }}"#, "None"),
    ];
    for (source, want) in cases {
        assert_eq!(render(source).as_deref(), Ok(want), "{source}");
    }
}

#[test]
fn get_refuses_what_python_refuses() {
    for source in [
        r#"{{ d.get() }}"#,
        r#"{{ d.get("a", 1, 2) }}"#,
        r#"{{ d.get(key="a") }}"#,
    ] {
        let e = render(source).expect_err(source);
        assert!(e.contains("dict.get()"), "{source}: {e}");
    }
}

#[test]
fn another_method_of_a_dict_is_refused_and_named() {
    let e = render("{{ d.keys() }}").expect_err("keys is not implemented");
    assert!(e.contains("\"keys\"") && e.contains("get"), "{e}");
}
