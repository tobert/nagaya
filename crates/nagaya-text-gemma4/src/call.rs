//! The body of a Gemma 4 tool call, the text between `<|tool_call>` and `<tool_call|>`.
//!
//! A body is `call:NAME{key:VALUE,...}`. A string is `<|"|>...<|"|>`. A key is a bare word, or a quoted string.

use serde_json::{Map, Value};

/// The quote of a string in a call.
const QUOTE: &str = "<|\"|>";

/// Why a body did not parse: the name of the call when the parser got that far, and the reason.
pub type CallError = (Option<String>, String);

/// Parse a call body into its name and its arguments, in the order that the model wrote them.
pub fn parse_tool_call(body: &str) -> Result<(String, Map<String, Value>), CallError> {
    let rest = body
        .strip_prefix("call:")
        .ok_or((None, "a call starts with `call:`".to_string()))?;
    let brace = rest.find('{').ok_or((None, "no `{` after the call name".to_string()))?;
    let name = rest[..brace].to_string();
    if name.is_empty() {
        return Err((None, "the call has no name".into()));
    }
    let mut p = ArgParser {
        s: &rest[brace..],
        i: 0,
    };
    let v = p.object().map_err(|e| (Some(name.clone()), e))?;
    if p.i != p.s.len() {
        return Err((Some(name), format!("text after the arguments: {:?}", &p.s[p.i..])));
    }
    match v {
        Value::Object(m) => Ok((name, m)),
        _ => unreachable!("object() gives an object or an error"),
    }
}

struct ArgParser<'a> {
    s: &'a str,
    i: usize,
}

impl<'a> ArgParser<'a> {
    fn rest(&self) -> &'a str {
        &self.s[self.i..]
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.rest().starts_with(lit) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, lit: &str) -> Result<(), String> {
        if self.eat(lit) {
            Ok(())
        } else {
            Err(format!("expected {:?} at {:?}", lit, self.rest()))
        }
    }

    /// `{key:value,...}`. A key is a bare word or a quoted string.
    fn object(&mut self) -> Result<Value, String> {
        self.expect("{")?;
        let mut m = Map::new();
        if self.eat("}") {
            return Ok(Value::Object(m));
        }
        loop {
            let key = if self.rest().starts_with(QUOTE) {
                self.string()?
            } else {
                let end = self.rest().find(':').ok_or("an argument key has no `:`")?;
                let k = self.rest()[..end].to_string();
                if k.is_empty() || k.contains(['{', '}', ',', '[', ']']) {
                    return Err(format!("bad argument key {k:?}"));
                }
                self.i += end;
                k
            };
            self.expect(":")?;
            let v = self.value()?;
            m.insert(key, v);
            if self.eat(",") {
                continue;
            }
            self.expect("}")?;
            return Ok(Value::Object(m));
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(QUOTE)?;
        let end = self.rest().find(QUOTE).ok_or("an unterminated string")?;
        let s = self.rest()[..end].to_string();
        self.i += end + QUOTE.len();
        Ok(s)
    }

    fn value(&mut self) -> Result<Value, String> {
        let r = self.rest();
        if r.starts_with(QUOTE) {
            return Ok(Value::String(self.string()?));
        }
        if r.starts_with('{') {
            return self.object();
        }
        if r.starts_with('[') {
            self.expect("[")?;
            let mut a = Vec::new();
            if self.eat("]") {
                return Ok(Value::Array(a));
            }
            loop {
                a.push(self.value()?);
                if self.eat(",") {
                    continue;
                }
                self.expect("]")?;
                return Ok(Value::Array(a));
            }
        }
        for (lit, v) in [
            ("null", Value::Null),
            ("true", Value::Bool(true)),
            ("false", Value::Bool(false)),
        ] {
            if self.eat(lit) {
                return Ok(v);
            }
        }
        let end = r.find([',', '}', ']']).unwrap_or(r.len());
        let num = &r[..end];
        let v: Value = serde_json::from_str(num).map_err(|_| format!("not a value: {num:?}"))?;
        if !v.is_number() {
            return Err(format!("not a value: {num:?}"));
        }
        self.i += end;
        Ok(v)
    }
}
