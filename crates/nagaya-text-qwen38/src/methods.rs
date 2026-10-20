//! The methods of Python strings that the chat template may call.
//!
//! The list below is the whole list. A template that calls another method fails when it renders, and the error names
//! the method. The behavior of each method is the one of Python 3.14 (`pystr.rs`).

use crate::pystr;
use minijinja::value::{Kwargs, Value, ValueKind};
use minijinja::{Error, ErrorKind, State};

/// The methods of Python strings that this library implements.
pub const METHODS: [&str; 19] = [
    "capitalize",
    "count",
    "endswith",
    "find",
    "isalpha",
    "isdigit",
    "isspace",
    "join",
    "lower",
    "lstrip",
    "removeprefix",
    "removesuffix",
    "replace",
    "rstrip",
    "split",
    "startswith",
    "strip",
    "title",
    "upper",
];

fn refuse(method: &str, what: impl std::fmt::Display) -> Error {
    Error::new(ErrorKind::InvalidOperation, format!("str.{method}(): {what}"))
}

/// The name of a kind of value, as Python names the type.
fn type_name(v: &Value) -> &'static str {
    match v.kind() {
        ValueKind::Undefined => "Undefined",
        ValueKind::None => "NoneType",
        ValueKind::Bool => "bool",
        ValueKind::Number if v.is_integer() => "int",
        ValueKind::Number => "float",
        ValueKind::String => "str",
        ValueKind::Seq => "list",
        ValueKind::Map => "dict",
        _ => "object",
    }
}

/// An integer argument. A bool counts as an integer, as in Python. A number beyond 64 bits moves to the nearest end.
fn integer(v: &Value) -> Option<i64> {
    match v.kind() {
        ValueKind::Bool => Some(i64::from(v.is_true())),
        ValueKind::Number if v.is_integer() => Some(match i128::try_from(v.clone()) {
            Ok(i) => i.clamp(i64::MIN as i128 / 4, i64::MAX as i128 / 4) as i64,
            Err(_) => i64::MAX / 4,
        }),
        _ => None,
    }
}

/// The arguments of one call: the positional values, and the keyword values if there are any.
struct Call<'a> {
    method: &'a str,
    positional: &'a [Value],
    keywords: Option<Kwargs>,
}

impl<'a> Call<'a> {
    fn new(method: &'a str, args: &'a [Value]) -> Result<Call<'a>, Error> {
        match args.split_last() {
            Some((last, positional)) if last.is_kwargs() => Ok(Call {
                method,
                positional,
                keywords: Some(Kwargs::try_from(last.clone()).map_err(|e| refuse(method, e))?),
            }),
            _ => Ok(Call {
                method,
                positional: args,
                keywords: None,
            }),
        }
    }

    /// The call takes between `least` and `most` positional values, and no keyword values.
    fn arity(&self, least: usize, most: usize) -> Result<(), Error> {
        if self.keywords.is_some() {
            return Err(refuse(self.method, "this method takes no keyword arguments"));
        }
        let n = self.positional.len();
        if n < least || n > most {
            return Err(refuse(
                self.method,
                format!("takes from {least} to {most} arguments, and {n} are given"),
            ));
        }
        Ok(())
    }

    fn text(&self, at: usize, what: &str) -> Result<&'a str, Error> {
        let v = &self.positional[at];
        v.as_str()
            .ok_or_else(|| refuse(self.method, format!("{what} must be a string, not {}", type_name(v))))
    }

    /// An index argument that may be left out or be `none`.
    fn index(&self, at: usize) -> Result<Option<i64>, Error> {
        match self.positional.get(at) {
            None => Ok(None),
            Some(v) if v.is_none() => Ok(None),
            Some(v) => integer(v).map(Some).ok_or_else(|| {
                refuse(
                    self.method,
                    format!("an index must be an integer or none, not {}", type_name(v)),
                )
            }),
        }
    }

    /// The `start` and `end` arguments that come after the first.
    fn range(&self) -> Result<(Option<i64>, Option<i64>), Error> {
        Ok((self.index(1)?, self.index(2)?))
    }
}

/// An argument that Python names, and that a template may give by name. `position` is its place among the positional
/// values. The result is `None` when the call leaves it out.
fn named(call: &Call, keywords: &Kwargs, name: &str, position: usize) -> Result<Option<Value>, Error> {
    let by_name: Option<Value> = keywords.get(name).map_err(|e| refuse(call.method, e))?;
    match (call.positional.get(position), by_name) {
        (Some(_), Some(_)) => Err(refuse(
            call.method,
            format!("the argument {name} is given by name and by position"),
        )),
        (Some(v), None) => Ok(Some(v.clone())),
        (None, by_name) => Ok(by_name),
    }
}

fn strings(parts: Vec<String>) -> Value {
    Value::from(parts.into_iter().map(Value::from).collect::<Vec<_>>())
}

/// Call the method `method` of a string. The engine calls this for every method that it does not have itself.
pub fn string_method(_state: &State, value: &Value, method: &str, args: &[Value]) -> Result<Value, Error> {
    let Some(text) = value.as_str() else {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            format!(
                "the method {method:?} is not implemented for a value of type {}. This library implements \
                 methods of strings only, and only these: {}. Change the template.",
                type_name(value),
                METHODS.join(", ")
            ),
        ));
    };
    if !METHODS.contains(&method) {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            format!(
                "the string method {method:?} is not implemented in this library. It implements only these: {}. \
                 Change the template.",
                METHODS.join(", ")
            ),
        ));
    }
    let call = Call::new(method, args)?;
    match method {
        "split" => split(text, &call),
        "replace" => replace(text, &call),
        "startswith" | "endswith" => {
            call.arity(1, 3)?;
            let (start, end) = call.range()?;
            let test = if method == "startswith" {
                pystr::starts_with
            } else {
                pystr::ends_with
            };
            let first = &call.positional[0];
            // Python takes a string, or a tuple of strings. A tuple of the template is a list here.
            let found = match first.kind() {
                ValueKind::String => test(text, first.as_str().expect("a string"), start, end),
                ValueKind::Seq => {
                    // Python tries the items in order, and stops at the first match. An item that is not a string
                    // is an error only when the search reaches it.
                    let mut found = false;
                    for item in first.try_iter()? {
                        let Some(one) = item.as_str() else {
                            return Err(refuse(
                                method,
                                format!("a tuple must hold strings only, not {}", type_name(&item)),
                            ));
                        };
                        if test(text, one, start, end) {
                            found = true;
                            break;
                        }
                    }
                    found
                }
                _ => {
                    return Err(refuse(
                        method,
                        format!(
                            "the first argument must be a string or a tuple of strings, not {}",
                            type_name(first)
                        ),
                    ));
                }
            };
            Ok(Value::from(found))
        }
        "strip" | "lstrip" | "rstrip" => {
            call.arity(0, 1)?;
            let chars = match call.positional.first() {
                None => None,
                Some(v) if v.is_none() => None,
                Some(_) => Some(call.text(0, "the characters")?),
            };
            Ok(Value::from(pystr::strip(
                text,
                chars,
                method != "rstrip",
                method != "lstrip",
            )))
        }
        "lower" | "upper" | "title" | "capitalize" => {
            call.arity(0, 0)?;
            Ok(Value::from(match method {
                "lower" => pystr::lower(text),
                "upper" => pystr::upper(text),
                "title" => pystr::title(text),
                _ => pystr::capitalize(text),
            }))
        }
        "isalpha" | "isdigit" | "isspace" => {
            call.arity(0, 0)?;
            Ok(Value::from(match method {
                "isalpha" => pystr::is_alpha(text),
                "isdigit" => pystr::is_digit(text),
                _ => pystr::is_space(text),
            }))
        }
        "find" | "count" => {
            call.arity(1, 3)?;
            let sub = call.text(0, "the substring")?;
            let (start, end) = call.range()?;
            Ok(Value::from(if method == "find" {
                pystr::find(text, sub, start, end)
            } else {
                pystr::count(text, sub, start, end)
            }))
        }
        "removeprefix" | "removesuffix" => {
            call.arity(1, 1)?;
            let affix = call.text(0, "the argument")?;
            let stripped = if method == "removeprefix" {
                text.strip_prefix(affix)
            } else {
                text.strip_suffix(affix)
            };
            Ok(Value::from(stripped.unwrap_or(text)))
        }
        "join" => {
            call.arity(1, 1)?;
            let argument = &call.positional[0];
            // Python joins any iterable. The undefined value of jinja2 is one, and it is empty.
            let iterable = matches!(
                argument.kind(),
                ValueKind::Undefined | ValueKind::Seq | ValueKind::Map | ValueKind::String | ValueKind::Iterable
            );
            if !iterable {
                return Err(refuse(
                    method,
                    format!(
                        "the argument must be an iterable of strings, not {}",
                        type_name(argument)
                    ),
                ));
            }
            let items = argument.try_iter().map_err(|e| refuse(method, e))?;
            let mut parts = Vec::new();
            for (n, item) in items.enumerate() {
                let Some(part) = item.as_str() else {
                    return Err(refuse(
                        method,
                        format!("item {n} must be a string, not {}", type_name(&item)),
                    ));
                };
                parts.push(part.to_string());
            }
            Ok(Value::from(parts.join(text)))
        }
        other => unreachable!("the method {other} is in the list but has no code"),
    }
}

fn split(text: &str, call: &Call) -> Result<Value, Error> {
    let (sep, maxsplit) = match &call.keywords {
        None => {
            call.arity(0, 2)?;
            (call.positional.first().cloned(), call.positional.get(1).cloned())
        }
        Some(keywords) => {
            if call.positional.len() > 2 {
                return Err(refuse(call.method, "takes at most 2 arguments"));
            }
            let found = (named(call, keywords, "sep", 0)?, named(call, keywords, "maxsplit", 1)?);
            keywords.assert_all_used().map_err(|e| refuse(call.method, e))?;
            found
        }
    };
    let sep = match &sep {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => Some(v.as_str().ok_or_else(|| {
            refuse(
                call.method,
                format!("the separator must be a string or none, not {}", type_name(v)),
            )
        })?),
    };
    let maxsplit = match &maxsplit {
        None => -1,
        Some(v) => integer(v).ok_or_else(|| {
            refuse(
                call.method,
                format!("maxsplit must be an integer, not {}", type_name(v)),
            )
        })?,
    };
    pystr::split(text, sep, maxsplit)
        .map(strings)
        .map_err(|m| refuse(call.method, m))
}

fn replace(text: &str, call: &Call) -> Result<Value, Error> {
    let count = match &call.keywords {
        None => {
            call.arity(2, 3)?;
            call.positional.get(2).cloned()
        }
        Some(keywords) => {
            if call.positional.len() < 2 || call.positional.len() > 3 {
                return Err(refuse(call.method, "takes two or three positional arguments"));
            }
            let found = named(call, keywords, "count", 2)?;
            keywords.assert_all_used().map_err(|e| refuse(call.method, e))?;
            found
        }
    };
    let old = call.text(0, "the first argument")?;
    let new = call.text(1, "the second argument")?;
    let count = match &count {
        None => -1,
        Some(v) => {
            integer(v).ok_or_else(|| refuse(call.method, format!("count must be an integer, not {}", type_name(v))))?
        }
    };
    Ok(Value::from(pystr::replace(text, old, new, count)))
}
