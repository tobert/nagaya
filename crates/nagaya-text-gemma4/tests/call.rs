//! The body of a Gemma 4 tool call (mkviolet `tool_calls_parse_strings_numbers_nesting_and_fail_loudly`).

use nagaya_text_gemma4::call::parse_tool_call;
use serde_json::{Value, json};

#[test]
fn a_call_parses_strings_numbers_and_nesting_in_order() {
    let (name, args) =
        parse_tool_call(r#"call:shell{zeta:<|"|>ls -la, now<|"|>,n:3,opts:{deep:true,tags:[<|"|>a<|"|>,2.5,null]}}"#)
            .unwrap();
    assert_eq!(name, "shell");
    assert_eq!(
        Value::Object(args.clone()),
        json!({"zeta": "ls -la, now", "n": 3, "opts": {"deep": true, "tags": ["a", 2.5, null]}})
    );
    assert_eq!(
        args.keys().collect::<Vec<_>>(),
        ["zeta", "n", "opts"],
        "the order that the model wrote"
    );
    assert!(parse_tool_call("call:empty{}").unwrap().1.is_empty());
    let (_, quoted) = parse_tool_call(r#"call:f{<|"|>a key<|"|>:1}"#).unwrap();
    assert_eq!(Value::Object(quoted), json!({"a key": 1}));
}

#[test]
fn a_bad_call_fails_and_names_the_call_when_it_can() {
    assert!(parse_tool_call(r#"call:shell{command:<|"|>open"#).is_err());
    assert_eq!(parse_tool_call("shell{}").unwrap_err().0, None);
    assert_eq!(parse_tool_call("call:{}").unwrap_err().0, None);
    assert_eq!(
        parse_tool_call("call:shell{x:oops}").unwrap_err().0.as_deref(),
        Some("shell")
    );
    assert_eq!(
        parse_tool_call("call:shell{x:1}tail").unwrap_err().0.as_deref(),
        Some("shell")
    );
    assert!(
        parse_tool_call("call:shell{x:\"str\"}").is_err(),
        "a JSON string is not a Gemma string"
    );
}
