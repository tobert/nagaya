//! The `sort_keys` and `separators` options of `tojson`, which Inkling 2's template calls (spec 11.3).

use nagaya_jinja::{Json, Template};

fn render(source: &str) -> Result<String, String> {
    let data = Json::Object(vec![
        ("b".to_string(), Json::Int(1)),
        (
            "a".to_string(),
            Json::Object(vec![
                ("z".to_string(), Json::Int(2)),
                ("m".to_string(), Json::Array(vec![Json::Int(3), Json::Int(4)])),
            ]),
        ),
    ]);
    let template = Template::from_source(source).map_err(|e| e.to_string())?;
    template.render(&[("d", data)]).map_err(|e| e.to_string())
}

#[test]
fn tojson_sort_keys_and_separators_match_python() {
    let cases = [
        (r#"{{ d | tojson }}"#, r#"{"b": 1, "a": {"z": 2, "m": [3, 4]}}"#),
        (
            r#"{{ d | tojson(sort_keys=true) }}"#,
            r#"{"a": {"m": [3, 4], "z": 2}, "b": 1}"#,
        ),
        (
            r#"{{ d | tojson(separators=(",", ":")) }}"#,
            r#"{"b":1,"a":{"z":2,"m":[3,4]}}"#,
        ),
        (
            r#"{{ d | tojson(sort_keys=true, separators=(",", ":")) }}"#,
            r#"{"a":{"m":[3,4],"z":2},"b":1}"#,
        ),
        (
            r#"{{ d | tojson(sort_keys=false, separators=none) }}"#,
            r#"{"b": 1, "a": {"z": 2, "m": [3, 4]}}"#,
        ),
        (r#"{{ ([1] + [2]) | tojson }}"#, "[1, 2]"),
    ];
    for (source, want) in cases {
        assert_eq!(render(source).as_deref(), Ok(want), "{source}");
    }
}

#[test]
fn tojson_refuses_bad_options() {
    for source in [
        r#"{{ d | tojson(unknown=true) }}"#,
        r#"{{ d | tojson(separators=(",",)) }}"#,
        r#"{{ d | tojson(separators=(1, ":")) }}"#,
        r#"{{ d | tojson(1) }}"#,
    ] {
        render(source).expect_err(source);
    }
}
