//! The blocks that transformers adds to jinja2, written as blocks that the engine has.
//!
//! transformers adds the block `{% generation %}` (`AssistantTracker` in `chat_template_utils.py`). It marks the text
//! of an assistant turn and writes its body as it is. The LFM3 templates use it (spec 11.11). The engine has no such
//! block, so this module writes each `generation` tag as `if true` and each `endgeneration` tag as `endif`. The marks
//! of white space control stay, so the text is the text of jinja2.

/// The source with each `generation` block written as an `if true` block. The word stays as it is in text, in a
/// comment, in a string, in a `raw` block, and in any place that is not the first word of a tag.
pub(crate) fn rewrite(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(at) = rest.find('{') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        if rest.starts_with("{#") {
            let end = rest.find("#}").map_or(rest.len(), |e| e + 2);
            out.push_str(&rest[..end]);
            rest = &rest[end..];
        } else if rest.starts_with("{{") {
            let end = code_end(rest, "}}");
            out.push_str(&rest[..end]);
            rest = &rest[end..];
        } else if rest.starts_with("{%") {
            let end = code_end(rest, "%}");
            let tag = &rest[..end];
            let (head, word, tail) = split_tag(tag);
            match word {
                "generation" => out.push_str(&format!("{head}if true{tail}")),
                "endgeneration" => out.push_str(&format!("{head}endif{tail}")),
                "raw" => {
                    out.push_str(tag);
                    rest = &rest[end..];
                    let close = raw_end(rest);
                    out.push_str(&rest[..close]);
                    rest = &rest[close..];
                    continue;
                }
                _ => out.push_str(tag),
            }
            rest = &rest[end..];
        } else {
            out.push('{');
            rest = &rest[1..];
        }
    }
    out.push_str(rest);
    out
}

/// The end of a tag or an expression that starts at the start of `text`, after `close`. A quoted string can hold the
/// close text. A tag that does not close runs to the end, and the engine reports the error.
fn code_end(text: &str, close: &str) -> usize {
    let bytes = text.as_bytes();
    let mut i = 2;
    let mut quote: Option<u8> = None;
    while i < bytes.len() {
        match quote {
            Some(q) => {
                if bytes[i] == b'\\' {
                    i += 1;
                } else if bytes[i] == q {
                    quote = None;
                }
            }
            None if bytes[i] == b'\'' || bytes[i] == b'"' => quote = Some(bytes[i]),
            None if text[i..].starts_with(close) => return i + close.len(),
            None => {}
        }
        i += 1;
    }
    text.len()
}

/// A tag as three parts: the opening marks and space, the first word, and the rest.
fn split_tag(tag: &str) -> (&str, &str, &str) {
    let inner = &tag[2..];
    let marks = usize::from(inner.starts_with(['-', '+']));
    let after = &inner[marks..];
    let space = after.len() - after.trim_start().len();
    let start = 2 + marks + space;
    let word_len = tag[start..]
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(tag.len() - start);
    (&tag[..start], &tag[start..start + word_len], &tag[start + word_len..])
}

/// The end of a `raw` block: the start of its `endraw` tag, or the end of the text.
fn raw_end(text: &str) -> usize {
    let mut from = 0;
    while let Some(at) = text[from..].find("{%") {
        let start = from + at;
        let end = code_end(&text[start..], "%}");
        if split_tag(&text[start..start + end]).1 == "endraw" {
            return start;
        }
        from = start + 2;
    }
    text.len()
}
