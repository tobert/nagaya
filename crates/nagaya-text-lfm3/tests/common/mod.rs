#![allow(dead_code)]

pub mod model;

use nagaya_jinja::Json;
use serde_json::Value;

/// A `serde_json` value in the interpreter's JSON type, with the order of the members kept.
pub fn json_of(value: &Value) -> Json {
    match value {
        Value::Null => Json::Null,
        Value::Bool(b) => Json::Bool(*b),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => Json::Int(i128::from(i)),
            (None, Some(u)) => Json::Int(i128::from(u)),
            (None, None) => Json::Float(n.as_f64().unwrap()),
        },
        Value::String(s) => Json::Str(s.clone()),
        Value::Array(items) => Json::Array(items.iter().map(json_of).collect()),
        Value::Object(map) => Json::Object(map.iter().map(|(k, v)| (k.clone(), json_of(v))).collect()),
    }
}

/// The ids of a fixture.
pub fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .expect("the ids are a list")
        .iter()
        .map(|x| u32::try_from(x.as_u64().expect("an id is a number")).unwrap())
        .collect()
}
