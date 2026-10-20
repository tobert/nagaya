//! The JSON reader: values, order of keys, and the inputs that it refuses.

mod common;

use common::fixture;
use nagaya_text_qwen38::json::Json;

fn p(text: &str) -> Json {
    Json::parse(text).unwrap_or_else(|e| panic!("{text:?} should parse: {e}"))
}

fn refused(text: &str) {
    assert!(Json::parse(text).is_err(), "{text:?} should be refused");
}

#[test]
fn reads_scalars() {
    assert_eq!(p("null"), Json::Null);
    assert_eq!(p(" true "), Json::Bool(true));
    assert_eq!(p("false"), Json::Bool(false));
    assert_eq!(p("0"), Json::Int(0));
    assert_eq!(p("-0"), Json::Int(0));
    assert_eq!(p("-17"), Json::Int(-17));
    assert_eq!(p("1.0"), Json::Float(1.0));
    assert_eq!(p("-0.0").clone(), Json::Float(-0.0));
    assert_eq!(p("1e3"), Json::Float(1000.0));
    assert_eq!(p("2.5E-2"), Json::Float(0.025));
    assert_eq!(p("\"\""), Json::Str(String::new()));
}

#[test]
fn keeps_integers_to_128_bits() {
    assert_eq!(p("18446744073709551615"), Json::Int(u64::MAX as i128));
    assert_eq!(p("-9223372036854775808"), Json::Int(i64::MIN as i128));
    assert_eq!(p("170141183460469231731687303715884105727"), Json::Int(i128::MAX));
    assert_eq!(p("-170141183460469231731687303715884105728"), Json::Int(i128::MIN));
    refused("170141183460469231731687303715884105728");
    refused("-170141183460469231731687303715884105729");
}

#[test]
fn reads_string_escapes() {
    assert_eq!(
        p(r#""a\"b\\c\/d\b\f\n\r\t""#),
        Json::Str("a\"b\\c/d\u{8}\u{c}\n\r\t".to_string())
    );
    assert_eq!(p(r#""\u00e9\u4e16""#), Json::Str("\u{e9}\u{4e16}".to_string()));
    assert_eq!(p(r#""\ud83d\ude00""#), Json::Str("\u{1f600}".to_string()));
    assert_eq!(
        p("\"\u{1f600} raw \u{7f}\""),
        Json::Str("\u{1f600} raw \u{7f}".to_string())
    );
    assert_eq!(p(r#""\u0000""#), Json::Str("\0".to_string()));
}

#[test]
fn keeps_the_order_of_keys() {
    let j = p(r#"{"z": 1, "a": {"y": [3, 2], "b": null}, "m": true}"#);
    let Json::Object(items) = &j else { panic!("object") };
    let keys: Vec<&str> = items.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["z", "a", "m"]);
    let Json::Object(inner) = &items[1].1 else {
        panic!("object")
    };
    assert_eq!(inner[0].0, "y");
}

#[test]
fn a_repeated_key_keeps_its_first_place_and_its_last_value() {
    let j = p(r#"{"a": 1, "b": 2, "a": 3}"#);
    assert_eq!(
        j,
        Json::Object(vec![("a".into(), Json::Int(3)), ("b".into(), Json::Int(2))])
    );
}

#[test]
fn reads_containers() {
    assert_eq!(p("[]"), Json::Array(vec![]));
    assert_eq!(p("{}"), Json::Object(vec![]));
    assert_eq!(p("[[], {}, [1, [2]]]").clone(), p("[ [ ] ,{ } ,[1,[2]] ]"));
}

#[test]
fn refuses_what_it_does_not_read() {
    for text in [
        "",
        " ",
        "nul",
        "tru",
        "01",
        "1.",
        ".5",
        "+1",
        "1e",
        "--1",
        "0x10",
        "NaN",
        "Infinity",
        "-Infinity",
        "1e999",
        "\"abc",
        "\"\\x41\"",
        "\"\\u12\"",
        "\"\\ud83d\"",
        "\"\\ude00\"",
        "\"\\ud83dx\"",
        "\"a\nb\"",
        "\"a\tb\"",
        "[1,]",
        "[1 2]",
        "{\"a\" 1}",
        "{\"a\":1,}",
        "{1:2}",
        "{'a':1}",
        "[1] x",
        "1 2",
        "[",
        "{",
        "{\"a\"",
        "'a'",
    ] {
        refused(text);
    }
}

#[test]
fn refuses_nesting_that_would_overflow_the_stack() {
    let deep = "[".repeat(100_000) + &"]".repeat(100_000);
    refused(&deep);
    let ok = "[".repeat(200) + &"]".repeat(200);
    p(&ok);
}

fn to_serde(j: &Json) -> serde_json::Value {
    use serde_json::Value;
    match j {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Int(i) => {
            if let Ok(v) = i64::try_from(*i) {
                Value::from(v)
            } else {
                Value::from(u64::try_from(*i).expect("fixture integers fit in 64 bits"))
            }
        }
        Json::Float(f) => Value::from(*f),
        Json::Str(s) => Value::String(s.clone()),
        Json::Array(a) => Value::Array(a.iter().map(to_serde).collect()),
        Json::Object(o) => Value::Object(o.iter().map(|(k, v)| (k.clone(), to_serde(v))).collect()),
    }
}

#[test]
fn agrees_with_serde_json_on_the_fixtures() {
    for name in ["tokenizer.json", "chat.json", "utf8.json", "classes.json"] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name);
        let text = std::fs::read_to_string(&path).unwrap();
        let theirs: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(to_serde(&fixture(name)), theirs, "{name}");
    }
}

#[test]
fn nesting_stops_at_512_levels_and_the_error_says_so() {
    let nest = |n: usize| "[".repeat(n) + &"]".repeat(n);
    p(&nest(512));
    let objects = |n: usize| "{\"k\":".repeat(n) + "1" + &"}".repeat(n);
    p(&objects(512));
    for text in [nest(513), objects(513)] {
        match Json::parse(&text) {
            Err(nagaya_text_qwen38::Error::Json(m)) => assert!(m.contains("512"), "{m}"),
            Ok(_) => panic!("513 levels should be refused"),
            Err(other) => panic!("513 levels should be a Json error, got {other:?}"),
        }
    }
}

#[test]
fn an_object_with_many_keys_is_read_in_linear_time() {
    let n = 100_000;
    let body: Vec<String> = (0..n).map(|i| format!("\"key.{i:07}\":{i}")).collect();
    let text = format!("{{{}}}", body.join(","));
    let start = std::time::Instant::now();
    let Json::Object(pairs) = p(&text) else {
        panic!("not an object")
    };
    let seconds = start.elapsed().as_secs_f64();
    assert_eq!(pairs.len(), n);
    assert!(seconds < 5.0, "{n} keys took {seconds} seconds");
}

#[test]
fn a_repeated_key_among_many_keeps_its_first_place_and_its_last_value() {
    let mut body: Vec<String> = (0..1000).map(|i| format!("\"k{i}\":{i}")).collect();
    body.push("\"k7\":\"last\"".to_string());
    let Json::Object(pairs) = p(&format!("{{{}}}", body.join(","))) else {
        panic!("not an object")
    };
    assert_eq!(pairs.len(), 1000);
    assert_eq!(pairs[7], ("k7".to_string(), Json::Str("last".to_string())));
}
