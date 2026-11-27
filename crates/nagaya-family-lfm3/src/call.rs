//! The tool calls of an LFM3 reply: the XML form of the chat template, read as the `response_template` of
//! `LFM3-1.7B` reads it with transformers' `parse_response` (spec 11.11, "The choices of the LFM3 family").
//!
//! A call is `<tool_call>`, then `<function=NAME>`, then a `<parameter=NAME>` block for each argument, then
//! `</function>` and `</tool_call>`. The value of a parameter is the text between its tags, without one new line at
//! each end. The type of the parameter in the tool schema decides how the value reads (`_coerce` of transformers).

use std::collections::BTreeMap;

use gogi_proto::context::Tool;
use nagaya_jinja::Json;
use nagaya_jinja::chars::is_space;
use nagaya_jinja::pystr::{lower, strip};
use serde_json::{Map, Number, Value};

const FUNCTION_OPEN: &str = "<function=";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";
const FUNCTION_CLOSE: &str = "</function>";

/// The parameters of each tool, as the reference keeps them: the `properties` of its schema.
#[derive(Debug, Default)]
pub struct ToolParams(BTreeMap<String, Map<String, Value>>);

impl ToolParams {
    /// The properties of each tool. A tool whose schema has no `properties` object has none.
    pub fn new(tools: &[Tool]) -> ToolParams {
        ToolParams(
            tools
                .iter()
                .map(|t| {
                    let props = match t.input_schema.get("properties") {
                        Some(Value::Object(p)) => p.clone(),
                        _ => Map::new(),
                    };
                    (t.name.clone(), props)
                })
                .collect(),
        )
    }
}

/// Skip the characters of the pattern class `\s` from byte `at`.
fn skip_space(text: &str, at: usize) -> usize {
    at + text[at..].len() - text[at..].trim_start_matches(is_space).len()
}

/// Read the text of a call after `<tool_call>` and before `</tool_call>`. Returns the name and the input, or why the
/// text is not a call.
pub fn parse_call(body: &str, tools: &ToolParams) -> Result<(String, Map<String, Value>), String> {
    let at = skip_space(body, 0);
    let rest = body[at..]
        .strip_prefix(FUNCTION_OPEN)
        .ok_or("expected <function=NAME> after <tool_call>")?;
    let end = rest.find(['>', '\n']).ok_or("the tag <function=NAME> does not close")?;
    if end == 0 || !rest[end..].starts_with('>') {
        return Err("expected <function=NAME> after <tool_call>".to_string());
    }
    let name = &rest[..end];
    let params = parameters(&rest[end + 1..]);
    let props = tools.0.get(name).filter(|p| !p.is_empty());
    let mut input = Map::new();
    for (key, raw) in params {
        let types = props.and_then(|p| p.get(&key)).map(schema_types).unwrap_or_default();
        let value = if types.is_empty() {
            Value::String(raw)
        } else {
            coerce(&raw, &types)?
        };
        input.insert(key, value);
    }
    Ok((name.to_string(), input))
}

/// The parameters of the text after `<function=NAME>`, as the reference's `tag_pattern` finds them:
/// `<parameter=KEY>\n?VALUE\n?</parameter>` with the shortest value, then space and another parameter, the end of
/// the function, or the end. A key that comes again takes the place of its value, as in a Python dict.
fn parameters(body: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut from = 0;
    while let Some(found) = body[from..].find(PARAMETER_OPEN) {
        let start = from + found;
        match parameter_at(body, start) {
            Some((key, value, end)) => {
                match out.iter_mut().find(|(k, _)| *k == key) {
                    Some(slot) => slot.1 = value,
                    None => out.push((key, value)),
                }
                from = end;
            }
            None => from = start + 1,
        }
    }
    out
}

/// One parameter that starts at `start`: its key, its value and the end of its close tag.
fn parameter_at(body: &str, start: usize) -> Option<(String, String, usize)> {
    let key_start = start + PARAMETER_OPEN.len();
    let key_len = body[key_start..].find(['>', '\n'])?;
    if key_len == 0 || !body[key_start + key_len..].starts_with('>') {
        return None;
    }
    let key = &body[key_start..key_start + key_len];
    let mut value_start = key_start + key_len + 1;
    if body[value_start..].starts_with('\n') {
        value_start += 1;
    }
    // The shortest value that a close tag and the look-ahead accept. A new line before the close tag is not part of
    // the value.
    let mut end = value_start;
    loop {
        let close = if body[end..].starts_with("\n</parameter>") {
            Some(end + 1 + PARAMETER_CLOSE.len())
        } else if body[end..].starts_with(PARAMETER_CLOSE) {
            Some(end + PARAMETER_CLOSE.len())
        } else {
            None
        };
        if let Some(close) = close
            && follows(body, close)
        {
            return Some((key.to_string(), body[value_start..end].to_string(), close));
        }
        let next = body[end..].chars().next()?;
        end += next.len_utf8();
    }
}

/// The look-ahead of the reference: space, then another parameter, or the end of the function and then only space,
/// or only space to the end.
fn follows(body: &str, at: usize) -> bool {
    let at = skip_space(body, at);
    let rest = &body[at..];
    if rest.is_empty() || rest.starts_with(PARAMETER_OPEN) {
        return true;
    }
    match rest.strip_prefix(FUNCTION_CLOSE) {
        Some(after) => after.trim_start_matches(is_space).is_empty(),
        None => false,
    }
}

/// The type names of a schema, in the order of `_schema_types` of transformers: `type`, then `anyOf` and `oneOf`, then
/// `null` for `nullable`.
fn schema_types(schema: &Value) -> Vec<String> {
    let Value::Object(schema) = schema else {
        return Vec::new();
    };
    let mut types: Vec<String> = Vec::new();
    match schema.get("type") {
        Some(Value::String(t)) => types.push(t.clone()),
        Some(Value::Array(list)) => types.extend(list.iter().filter_map(|t| t.as_str().map(str::to_string))),
        _ => {}
    }
    for union in ["anyOf", "oneOf"] {
        if let Some(Value::Array(choices)) = schema.get(union) {
            for choice in choices {
                types.extend(schema_types(choice));
            }
        }
    }
    if schema.get("nullable").is_some_and(truthy) && !types.iter().any(|t| t == "null") {
        types.push("null".to_string());
    }
    types
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `_coerce` of transformers: the first type that reads the value wins. A value that no type reads stays text. A
/// number that a JSON number of the API cannot hold is an error, so no value changes on the way.
fn coerce(raw: &str, types: &[String]) -> Result<Value, String> {
    for t in types {
        match t.as_str() {
            "integer" => {
                if let Some(i) = python_int(raw) {
                    return integer(i);
                }
                if let Some(f) = python_float(raw)
                    && f.is_finite()
                    && f.fract() == 0.0
                {
                    return whole(f);
                }
            }
            "number" => {
                if let Some(f) = python_float(raw)
                    && f.is_finite()
                {
                    if f.fract() == 0.0 && !raw.contains('.') {
                        return whole(f);
                    }
                    return Number::from_f64(f)
                        .map(Value::Number)
                        .ok_or_else(|| format!("the number {f} is not a finite JSON number"));
                }
            }
            "boolean" => match lower(&strip(raw, None, true, true)).as_str() {
                "true" | "1" => return Ok(Value::Bool(true)),
                "false" | "0" => return Ok(Value::Bool(false)),
                _ => {}
            },
            "null" => {
                if matches!(strip(raw, None, true, true).as_str(), "null" | "None") {
                    return Ok(Value::Null);
                }
            }
            "object" | "array" => {
                if let Ok(json) = Json::parse(raw) {
                    let fits = matches!(
                        (t.as_str(), &json),
                        ("object", Json::Object(_)) | ("array", Json::Array(_))
                    );
                    if fits {
                        return value_of(&json);
                    }
                }
            }
            _ => {}
        }
    }
    Ok(Value::String(raw.to_string()))
}

fn integer(i: i128) -> Result<Value, String> {
    match (i64::try_from(i), u64::try_from(i)) {
        (Ok(i), _) => Ok(Value::Number(i.into())),
        (_, Ok(u)) => Ok(Value::Number(u.into())),
        _ => Err(format!(
            "the integer {i} does not fit in 64 bits. The council API carries no larger integer."
        )),
    }
}

/// A float with no fraction as an integer, as Python's `int(float)` gives it.
fn whole(f: f64) -> Result<Value, String> {
    if f.abs() >= 1.8446744073709552e19 {
        return Err(format!(
            "the integer {f:e} does not fit in 64 bits. The council API carries no larger integer."
        ));
    }
    integer(f as i128)
}

/// Digits with one `_` at most between two digits, as Python reads them. Returns the digits without the marks.
fn python_digits(s: &str) -> Option<String> {
    if s.is_empty() || s.starts_with('_') || s.ends_with('_') || s.contains("__") {
        return None;
    }
    let digits: String = s.chars().filter(|&c| c != '_').collect();
    digits.chars().all(|c| c.is_ascii_digit()).then_some(digits)
}

/// Python's `int(text)` for ASCII digits: space at the ends, a sign, and digits.
fn python_int(raw: &str) -> Option<i128> {
    let s = strip(raw, None, true, true);
    let (negative, body) = match s.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, s.strip_prefix('+').unwrap_or(&s)),
    };
    let digits = python_digits(body)?;
    let n: i128 = digits.parse().ok()?;
    Some(if negative { -n } else { n })
}

/// Python's `float(text)`: space at the ends, a sign, and a decimal number, `inf`, `infinity` or `nan`.
fn python_float(raw: &str) -> Option<f64> {
    let s = strip(raw, None, true, true);
    let (sign, body) = match s.strip_prefix('-') {
        Some(b) => (-1.0, b),
        None => (1.0, s.strip_prefix('+').unwrap_or(&s)),
    };
    match lower(body).as_str() {
        "inf" | "infinity" => return Some(sign * f64::INFINITY),
        "nan" => return Some(f64::NAN),
        _ => {}
    }
    let (mantissa, exponent) = match body.find(['e', 'E']) {
        Some(at) => (&body[..at], Some(&body[at + 1..])),
        None => (body, None),
    };
    let (whole, fraction) = match mantissa.split_once('.') {
        Some((w, f)) => (w, Some(f)),
        None => (mantissa, None),
    };
    let whole = if whole.is_empty() {
        String::new()
    } else {
        python_digits(whole)?
    };
    let fraction = match fraction {
        Some("") | None => String::new(),
        Some(f) => python_digits(f)?,
    };
    if whole.is_empty() && fraction.is_empty() {
        return None;
    }
    let mut text = format!("{whole}.{fraction}");
    if let Some(e) = exponent {
        let (esign, edigits) = match e.strip_prefix('-') {
            Some(d) => ("-", d),
            None => ("", e.strip_prefix('+').unwrap_or(e)),
        };
        text.push('e');
        text.push_str(esign);
        text.push_str(&python_digits(edigits)?);
    }
    text.parse::<f64>().ok().map(|f| sign * f)
}

/// The JSON of the interpreter as a `serde_json` value. A number that the API cannot hold is an error.
fn value_of(json: &Json) -> Result<Value, String> {
    Ok(match json {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Int(i) => integer(*i)?,
        Json::Float(f) => {
            Value::Number(Number::from_f64(*f).ok_or_else(|| format!("the number {f} is not a finite JSON number"))?)
        }
        Json::Str(s) => Value::String(s.clone()),
        Json::Array(items) => Value::Array(items.iter().map(value_of).collect::<Result<_, _>>()?),
        Json::Object(pairs) => {
            let mut map = Map::new();
            for (k, v) in pairs {
                map.insert(k.clone(), value_of(v)?);
            }
            Value::Object(map)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{python_float, python_int};

    #[test]
    fn numbers_read_as_python_reads_them() {
        assert_eq!(python_int(" 1_000 "), Some(1000));
        assert_eq!(python_int("+7"), Some(7));
        assert_eq!(python_int("1__0"), None);
        assert_eq!(python_int("3.0"), None);
        assert_eq!(python_float("3.0"), Some(3.0));
        assert_eq!(python_float("1e3"), Some(1000.0));
        assert_eq!(python_float(".5"), Some(0.5));
        assert_eq!(python_float("5."), Some(5.0));
        assert_eq!(python_float("1_0.2_5"), Some(10.25));
        assert_eq!(python_float("."), None);
        assert_eq!(python_float("e5"), None);
        assert_eq!(python_float("-Infinity"), Some(f64::NEG_INFINITY));
        assert!(python_float("nan").is_some_and(f64::is_nan));
    }
}
