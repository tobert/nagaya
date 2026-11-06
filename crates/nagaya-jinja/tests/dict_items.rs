//! The method `items` of a dict, which the LFM3 chat templates call (spec 11.11). The expected text is the output of
//! transformers 5.20.2 with jinja2 3.1.6, run on 2026-11-06.

use nagaya_jinja::{Json, Template};

fn render(source: &str) -> Result<String, String> {
    let d = Json::Object(vec![
        ("a".to_string(), Json::Int(1)),
        ("n".to_string(), Json::Null),
        ("s".to_string(), Json::Str("t".to_string())),
    ]);
    let template = Template::from_source(source).map_err(|e| e.to_string())?;
    template
        .render(&[("d", d), ("e", Json::Object(Vec::new()))])
        .map_err(|e| e.to_string())
}

#[test]
fn items_gives_the_pairs_in_the_order_of_the_dict() {
    let cases = [
        (
            "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}",
            "a=1;n=None;s=t;",
        ),
        ("{% for k, v in e.items() %}x{% else %}empty{% endfor %}", "empty"),
        ("{{ d.items() | length }}", "3"),
        ("{% for p in d.items() %}{{ p[0] }}{% endfor %}", "ans"),
    ];
    for (source, want) in cases {
        assert_eq!(render(source).as_deref(), Ok(want), "{source}");
    }
}

#[test]
fn items_refuses_an_argument_as_python_does() {
    for source in ["{{ d.items(1) }}", "{{ d.items(k=1) }}"] {
        let e = render(source).expect_err(source);
        assert!(e.contains("dict.items()"), "{source}: {e}");
    }
}
