//! The chat template: the Jinja source in the GGUF, run with the rules of mk.
//!
//! mk renders with Python's jinja2, set up as Hugging Face transformers sets it up (`service/template.py`). This module
//! runs the same source with minijinja and makes the same three choices: `trim_blocks` and `lstrip_blocks` are on,
//! `tojson` writes what `json.dumps` writes with `ensure_ascii` off, and `trim` strips what `str.strip` strips.

use crate::Error;
use crate::gguf::{Metadata, Value as MetaValue};
use crate::json::Json;
use crate::pystr;
use minijinja::value::{Kwargs, Rest, Value, ValueKind};
use minijinja::{AutoEscape, Environment, ErrorKind};

/// What a render takes besides the messages and the tools.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// The template variable `enable_thinking`.
    pub thinking: bool,
    /// The template variable `add_generation_prompt`.
    pub generation_prompt: bool,
    /// The template variable `reasoning_effort`. `None` leaves the variable out.
    pub reasoning_effort: Option<String>,
}

impl Default for RenderOptions {
    fn default() -> Self {
        RenderOptions {
            thinking: true,
            generation_prompt: true,
            reasoning_effort: None,
        }
    }
}

/// The name that the template has inside the engine. Error messages show it.
const NAME: &str = "chat_template";

/// A chat template, ready to render.
pub struct ChatTemplate {
    env: Environment<'static>,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ChatTemplate")
    }
}

fn template_error(e: minijinja::Error) -> Error {
    Error::Template(e.to_string())
}

impl ChatTemplate {
    /// Compile the template source. A syntax error is an error here, not at the first render.
    pub fn from_source(source: &str) -> Result<ChatTemplate, Error> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.add_filter("tojson", tojson);
        env.add_filter("trim", trim);
        env.add_filter("string", |value: Value| py_str(&value));
        env.add_filter("length", length);
        env.add_filter("count", length);
        env.add_filter("items", items);
        env.set_formatter(|out, _state, value| {
            out.write_str(&py_str(value))
                .map_err(|_| minijinja::Error::from(ErrorKind::WriteFailure))
        });
        add_tests_of_jinja2(&mut env);
        env.add_function("raise_exception", raise_exception);
        env.set_unknown_method_callback(crate::methods::string_method);
        env.add_template_owned(NAME, source.to_string())
            .map_err(template_error)?;
        Ok(ChatTemplate { env })
    }

    /// Compile the template in the key `tokenizer.chat_template`.
    pub fn from_metadata(m: &Metadata) -> Result<ChatTemplate, Error> {
        match m.get("tokenizer.chat_template") {
            Some(MetaValue::Str(source)) => ChatTemplate::from_source(source),
            Some(_) => Err(Error::Metadata(
                "tokenizer.chat_template is not a string. Use a GGUF made for this family.".into(),
            )),
            None => Err(Error::Metadata(
                "the key tokenizer.chat_template is missing. Use a GGUF made for this family.".into(),
            )),
        }
    }

    /// The text for `messages`, a JSON array. `tools` that are empty count as no tools, as in mk.
    pub fn render(&self, messages: &Json, tools: Option<&Json>, options: &RenderOptions) -> Result<String, Error> {
        if !matches!(messages, Json::Array(_)) {
            return Err(Error::Template("the messages must be an array".to_string()));
        }
        let mut variables: Vec<(&str, Value)> = vec![
            ("messages", to_value(messages)),
            ("add_generation_prompt", Value::from(options.generation_prompt)),
            ("enable_thinking", Value::from(options.thinking)),
        ];
        if let Some(tools) = tools.filter(|t| truthy(t)) {
            variables.push(("tools", to_value(tools)));
        }
        if let Some(effort) = &options.reasoning_effort {
            variables.push(("reasoning_effort", Value::from(effort.as_str())));
        }
        let template = self.env.get_template(NAME).map_err(template_error)?;
        template.render(Value::from_iter(variables)).map_err(template_error)
    }
}

/// Python's truth of a JSON value.
pub(crate) fn truthy(j: &Json) -> bool {
    match j {
        Json::Null => false,
        Json::Bool(b) => *b,
        Json::Int(i) => *i != 0,
        Json::Float(f) => *f != 0.0,
        Json::Str(s) => !s.is_empty(),
        Json::Array(a) => !a.is_empty(),
        Json::Object(o) => !o.is_empty(),
    }
}

fn to_value(j: &Json) -> Value {
    match j {
        Json::Null => Value::from(()),
        Json::Bool(b) => Value::from(*b),
        Json::Int(i) => Value::from(*i),
        Json::Float(f) => Value::from(*f),
        Json::Str(s) => Value::from(s.as_str()),
        Json::Array(a) => Value::from(a.iter().map(to_value).collect::<Vec<_>>()),
        Json::Object(o) => Value::from_iter(o.iter().map(|(k, v)| (k.as_str(), to_value(v)))),
    }
}

/// The tests of jinja2 on which the engine differs. The engine says that a string is not a sequence and that a bool is
/// not a number, and it has no test `callable`. Python answers the other way, and the undefined value of jinja2 is
/// a sequence, an iterable and callable. (jinja2 `tests.py`.)
fn add_tests_of_jinja2(env: &mut Environment<'_>) {
    use ValueKind::{Bool, Iterable, Map, Number, Plain, Seq, String as Str, Undefined};
    env.add_test("sequence", |v: Value| matches!(v.kind(), Undefined | Str | Seq | Map));
    env.add_test("number", |v: Value| matches!(v.kind(), Number | Bool));
    env.add_test("string", |v: Value| v.kind() == Str);
    env.add_test("mapping", |v: Value| v.kind() == Map);
    env.add_test("iterable", |v: Value| {
        matches!(v.kind(), Undefined | Str | Seq | Map | Iterable)
    });
    env.add_test("defined", |v: Value| v.kind() != Undefined);
    env.add_test("undefined", |v: Value| v.kind() == Undefined);
    env.add_test("none", |v: Value| v.kind() == ValueKind::None);
    env.add_test("true", |v: Value| v.kind() == Bool && v.is_true());
    env.add_test("false", |v: Value| v.kind() == Bool && !v.is_true());
    env.add_test("boolean", |v: Value| v.kind() == Bool);
    env.add_test("integer", |v: Value| v.kind() == Number && v.is_integer());
    env.add_test("float", |v: Value| v.kind() == Number && !v.is_integer());
    env.add_test("callable", |v: Value| matches!(v.kind(), Undefined | Plain));
}

/// `raise_exception(message)` of mk's template: Python's `str` of any value is the message. The function takes one
/// argument, by position or as `msg=`.
fn raise_exception(args: Rest<Value>) -> Result<Value, minijinja::Error> {
    let refuse = |m: &str| minijinja::Error::new(ErrorKind::InvalidOperation, m.to_string());
    let message = match args.split_last() {
        Some((last, [])) if !last.is_kwargs() => last.clone(),
        Some((last, [])) => {
            let keywords = Kwargs::try_from(last.clone())?;
            let message: Value = keywords.get("msg")?;
            keywords.assert_all_used()?;
            message
        }
        _ => return Err(refuse("raise_exception takes one argument, the message")),
    };
    Err(refuse(&py_str(&message)))
}

/// Python's `str(value)`. The default way to write a value in a template, and the result of the `string` filter.
/// A string holds its own text, and a string inside a list or a map is in quotes, as `repr` writes it.
pub(crate) fn py_str(value: &Value) -> String {
    match value.kind() {
        ValueKind::Undefined => String::new(),
        ValueKind::String => value.as_str().expect("a string value has text").to_string(),
        _ => py_repr(value),
    }
}

fn py_repr(value: &Value) -> String {
    match value.kind() {
        ValueKind::Undefined => "Undefined".to_string(),
        ValueKind::None => "None".to_string(),
        ValueKind::Bool => if value.is_true() { "True" } else { "False" }.to_string(),
        ValueKind::Number if value.is_integer() => match i128::try_from(value.clone()) {
            Ok(i) => i.to_string(),
            Err(_) => u128::try_from(value.clone()).map_or_else(|_| value.to_string(), |u| u.to_string()),
        },
        ValueKind::Number => {
            let f = f64::try_from(value.clone()).unwrap_or(f64::NAN);
            if f.is_nan() {
                "nan".to_string()
            } else if f.is_infinite() {
                if f < 0.0 { "-inf" } else { "inf" }.to_string()
            } else {
                python_float(f).expect("a finite float")
            }
        }
        ValueKind::String => pystr::repr(value.as_str().expect("a string value has text")),
        ValueKind::Seq => {
            let items: Vec<String> = value.try_iter().into_iter().flatten().map(|v| py_repr(&v)).collect();
            format!("[{}]", items.join(", "))
        }
        ValueKind::Map => {
            let pairs: Vec<String> = value
                .try_iter()
                .into_iter()
                .flatten()
                .map(|key| {
                    let item = value.get_item(&key).unwrap_or(Value::UNDEFINED);
                    format!("{}: {}", py_repr(&key), py_repr(&item))
                })
                .collect();
            format!("{{{}}}", pairs.join(", "))
        }
        _ => value.to_string(),
    }
}

/// The `length` filter of jinja2: `len(value)`. The undefined value has length 0.
fn length(value: Value) -> Result<usize, minijinja::Error> {
    if value.kind() == ValueKind::Undefined {
        return Ok(0);
    }
    value.len().ok_or_else(|| {
        minijinja::Error::new(
            ErrorKind::InvalidOperation,
            format!("a value of type {} has no length", crate::methods::type_name(&value)),
        )
    })
}

/// The `items` filter of jinja2: the pairs of a map. The undefined value has no pairs.
fn items(value: Value) -> Result<Value, minijinja::Error> {
    match value.kind() {
        ValueKind::Undefined => Ok(Value::from(Vec::<Value>::new())),
        ValueKind::Map => {
            let pairs: Vec<Value> = value
                .try_iter()?
                .map(|key| {
                    let item = value.get_item(&key).unwrap_or(Value::UNDEFINED);
                    Value::from(vec![key, item])
                })
                .collect();
            Ok(Value::from(pairs))
        }
        _ => Err(minijinja::Error::new(
            ErrorKind::InvalidOperation,
            format!("a value of type {} has no items", crate::methods::type_name(&value)),
        )),
    }
}

/// The `trim` filter of jinja2: `str(value).strip(chars)`. The characters may be left out or be `none`.
fn trim(value: Value, rest: Rest<Value>) -> Result<String, minijinja::Error> {
    if rest.len() > 1 {
        return Err(minijinja::Error::new(
            ErrorKind::TooManyArguments,
            "trim takes at most one argument, the characters",
        ));
    }
    let set = match rest.first() {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => Some(v.as_str().ok_or_else(|| {
            minijinja::Error::new(
                ErrorKind::InvalidOperation,
                "the characters of trim must be a string or none",
            )
        })?),
    };
    Ok(pystr::strip(&py_str(&value), set, true, true))
}

/// `json.dumps(value, ensure_ascii=False)`: a comma and a space between items, a colon and a space after a key.
fn tojson(value: Value) -> Result<String, minijinja::Error> {
    let mut out = String::new();
    write_json(&value, &mut out).map_err(|m| minijinja::Error::new(ErrorKind::InvalidOperation, m))?;
    Ok(out)
}

fn write_json(value: &Value, out: &mut String) -> Result<(), String> {
    match value.kind() {
        ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        // A template literal from 2^127 to 2^128-1 is a u128, so this fallback is reachable. A test covers it.
        ValueKind::Number if value.is_integer() => match i128::try_from(value.clone()) {
            Ok(i) => out.push_str(&i.to_string()),
            Err(_) => out.push_str(&u128::try_from(value.clone()).map_err(|e| e.to_string())?.to_string()),
        },
        ValueKind::Number => out.push_str(&python_float(f64::try_from(value.clone()).map_err(|e| e.to_string())?)?),
        ValueKind::String => write_json_string(value.as_str().ok_or("a string value has no text")?, out),
        ValueKind::Seq => {
            out.push('[');
            for (n, item) in value.try_iter().map_err(|e| e.to_string())?.enumerate() {
                if n > 0 {
                    out.push_str(", ");
                }
                write_json(&item, out)?;
            }
            out.push(']');
        }
        ValueKind::Map => {
            out.push('{');
            for (n, key) in value.try_iter().map_err(|e| e.to_string())?.enumerate() {
                if n > 0 {
                    out.push_str(", ");
                }
                write_json_string(&json_key(&key)?, out);
                out.push_str(": ");
                write_json(&value.get_item(&key).map_err(|e| e.to_string())?, out)?;
            }
            out.push('}');
        }
        other => return Err(format!("tojson cannot write a value of kind {other:?}")),
    }
    Ok(())
}

/// The text of a map key, as `json.dumps` converts it: a string stays, a number or a constant becomes its JSON text.
/// Python refuses any other key.
fn json_key(key: &Value) -> Result<String, String> {
    match key.kind() {
        ValueKind::String => Ok(key.as_str().ok_or("a string key has no text")?.to_string()),
        ValueKind::None | ValueKind::Bool | ValueKind::Number => {
            let mut text = String::new();
            write_json(key, &mut text)?;
            Ok(text)
        }
        other => Err(format!(
            "a key in tojson must be a string, a number, a bool or none, not {other:?}"
        )),
    }
}

fn write_json_string(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `repr(float)` of Python: the shortest digits that read back the same, in fixed form for a decimal exponent from
/// -4 to 15, and in scientific form otherwise.
fn python_float(f: f64) -> Result<String, String> {
    if !f.is_finite() {
        return Err("tojson cannot write NaN or infinity".to_string());
    }
    let sign = if f.is_sign_negative() { "-" } else { "" };
    if f == 0.0 {
        return Ok(format!("{sign}0.0"));
    }
    let scientific = format!("{:e}", f.abs());
    let (mantissa, exponent) = scientific.split_once('e').expect("LowerExp always has an exponent");
    let exponent: i32 = exponent.parse().expect("the exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let point = exponent + 1;
    let body = if (-3..=16).contains(&point) {
        if point <= 0 {
            format!("0.{}{digits}", "0".repeat((-point) as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
        }
    } else {
        let rest = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        format!(
            "{}{rest}e{}{:02}",
            &digits[..1],
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        )
    };
    Ok(format!("{sign}{body}"))
}
