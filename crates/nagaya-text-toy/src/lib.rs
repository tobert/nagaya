//! The toy family library: nagaya's family interface on gogi's `gogi-toy` (spec 11.3b).
//!
//! The tokens, the template, the boundaries, the rendering of an open message and the council read come from
//! `gogi_toy`, so they equal the tokens of `gogi-fake`. The fake engine gives logits over `gogi_toy::VOCAB` tokens (engine-abi, "Logit outputs").

mod plan;
mod reply;
mod test_set;

use gogi_proto::context::{Context, Tool};
use gogi_toy as toy;
use nagaya_council::family::{
    Boundary, ControlHit, DecisionInput, Family, FamilyError, RawRead, ReadPlan, Rendering, ReplyParser, ReplyStart,
};
use nagaya_council::template_id::{hash16, template_id};

/// The name of the family in the template id.
const FAMILY: &str = "toy";

/// Where in the library the test hook changes tokens. The hook exists in test builds only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Site {
    /// The tokens of a rendered context.
    Context,
    /// The prefix of a read plan.
    Prefix,
    /// The tokens that a step of a read plan feeds.
    Feed,
}

/// The test hook: it changes tokens that the library renders. Test builds only.
#[cfg(test)]
pub(crate) type Hook = Option<fn(Site, &mut [u32])>;

/// Apply a hook to tokens.
#[cfg(test)]
pub(crate) fn apply(hook: Hook, site: Site, tokens: &mut [u32]) {
    if let Some(hook) = hook {
        hook(site, tokens);
    }
}

/// The toy family.
pub struct ToyFamily {
    template_id: String,
    #[cfg(test)]
    pub(crate) hook: Hook,
}

impl ToyFamily {
    /// Make the family, and compute its template id from its test renders.
    pub fn new() -> Self {
        let mut family = ToyFamily {
            template_id: String::new(),
            #[cfg(test)]
            hook: None,
        };
        family.template_id = family.compute_template_id();
        family
    }

    /// Make the family with a hook that changes the tokens it renders. Tests use this to see the id change.
    #[cfg(test)]
    pub(crate) fn with_hook(hook: fn(Site, &mut [u32])) -> Self {
        let mut family = ToyFamily {
            template_id: String::new(),
            hook: Some(hook),
        };
        family.template_id = family.compute_template_id();
        family
    }

    fn compute_template_id(&self) -> String {
        let council = nagaya_council::template_id::council_hash(&test_set::renders(self));
        template_id(&council, FAMILY, &hash16(chat_template_text().as_bytes()))
    }

    #[cfg_attr(not(test), allow(unused_variables))]
    pub(crate) fn tweak(&self, site: Site, tokens: &mut [u32]) {
        #[cfg(test)]
        apply(self.hook, site, tokens);
    }
}

impl Default for ToyFamily {
    fn default() -> Self {
        Self::new()
    }
}

/// A logit bias that makes a fake engine speak the toy language, for tests (spec 11.3b).
///
/// The fake engine's logits look random, so a greedy reply holds bytes of 0x80 and up and markers. The template does
/// not render those back as the same tokens: a byte of 0x80 and up becomes U+FFFD, and a marker in a part adds no
/// text. The bias keeps the ASCII bytes and `<end>` at their values, and moves each other token down by 40. Thus a
/// reply is ASCII text that the template renders back, and a generation can hold its head (gogi "Generations").
/// `<end>` gets 1 more, so that a greedy reply ends after about ten tokens. gogi-fake picks from the same set.
pub fn speaking_bias() -> Vec<f32> {
    (0..toy::VOCAB as u32)
        .map(|token| match token {
            0..0x80 => 0.0,
            toy::END => 1.0,
            _ => -40.0,
        })
        .collect()
}

/// The text of the toy chat template: its name and its marker table. A change to a marker changes the digest.
fn chat_template_text() -> String {
    let mut text = String::from("gogi-toy chat template\n");
    for (marker, id) in toy::MARKERS {
        text.push_str(&format!("{marker}={id}\n"));
    }
    text
}

fn control_hit(place: impl Into<String>, token: &str) -> ControlHit {
    ControlHit {
        place: place.into(),
        token: token.to_owned(),
    }
}

/// The control text in `text`, found at `place`.
pub(crate) fn hits(place: &str, text: &str) -> Vec<ControlHit> {
    toy::control_hits(text)
        .into_iter()
        .map(|token| control_hit(place, token))
        .collect()
}

impl Family for ToyFamily {
    fn render(&self, context: &Context, generation_prompt: bool) -> Result<Rendering, FamilyError> {
        context.check()?;
        // An open message renders without its end (gogi-toy). Each open field must tokenize on its own. A toy token
        // is a byte, so the check never refuses, but it is the check that every family makes.
        toy::check_open(&context.tools, context.thinking, &context.messages)?;
        let r = toy::render(&context.tools, context.thinking, &context.messages, generation_prompt);
        let mut tokens = r.tokens;
        self.tweak(Site::Context, &mut tokens);
        let control_text = context
            .messages
            .iter()
            .enumerate()
            .flat_map(|(i, m)| {
                let place = format!("message:{i}");
                let content = hits(&place, &nagaya_council::family::content_text(m).unwrap_or_default());
                let reasoning = hits(&place, m.reasoning.as_deref().unwrap_or(""));
                content.into_iter().chain(reasoning).collect::<Vec<_>>()
            })
            .collect();
        Ok(Rendering {
            tokens,
            end: r.end,
            boundaries: r
                .cuts
                .iter()
                .map(|c| Boundary {
                    message: c.message,
                    tokens: c.tokens,
                })
                .collect(),
            control_text,
            media: Vec::new(),
        })
    }

    fn decision_read(&self, input: &DecisionInput<'_>) -> Result<Box<dyn ReadPlan>, FamilyError> {
        plan::decision_read(self, input)
    }

    fn raw_read(&self, letters: &[String], suffixes: &[String]) -> Result<RawRead, FamilyError> {
        let letters = toy::read::letter_tokens(letters).map_err(|e| FamilyError::client_param("letters", e))?;
        let control_text = suffixes
            .iter()
            .enumerate()
            .flat_map(|(i, s)| hits(&format!("suffix:{i}"), s))
            .collect();
        Ok(RawRead {
            letters,
            suffixes: suffixes.iter().map(|s| toy::tokenize_suffix(s)).collect(),
            control_text,
        })
    }

    fn reply_parser(&self, start: ReplyStart, _tools: &[Tool]) -> Result<Box<dyn ReplyParser>, FamilyError> {
        // The toy call format has no schema, so the tools do not change the parse. The toy parser trims nothing, so
        // a continuation needs no other parse.
        Ok(Box::new(reply::ToyReplyParser::new(start.thinking)))
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        let mut tokens = Vec::new();
        toy::push_text(&mut tokens, text);
        Ok(tokens)
    }

    fn tokenize_special(&self, text: &str) -> Result<Vec<u32>, FamilyError> {
        Ok(toy::tokenize_suffix(text))
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, FamilyError> {
        let mut bytes = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            check_token(i, token)?;
            bytes.extend(token_bytes(token));
        }
        String::from_utf8(bytes).map_err(|e| {
            FamilyError::client_param(
                "tokens",
                format!("the tokens are not UTF-8 text: {e}. Send whole characters."),
            )
        })
    }

    fn pieces(&self, tokens: &[u32]) -> Result<Vec<String>, FamilyError> {
        tokens
            .iter()
            .enumerate()
            .map(|(i, &token)| {
                check_token(i, token)?;
                Ok(String::from_utf8_lossy(&token_bytes(token)).into_owned())
            })
            .collect()
    }

    fn template_id(&self) -> &str {
        &self.template_id
    }

    fn vocab_size(&self) -> usize {
        toy::VOCAB
    }

    /// The model thinks before its reply, so its served model declares `thinking`.
    fn thinks(&self) -> bool {
        true
    }

    /// The fake takes latent rows at positions of the byte 0. Its record names the first position, the count and
    /// the width, as the LFM3 record does.
    fn latent_record(&self, first: usize, count: usize, width: usize) -> Option<(u32, Vec<u8>)> {
        let mut record = Vec::with_capacity(32);
        record.extend(5u32.to_le_bytes());
        record.extend((count as u32).to_le_bytes());
        record.extend((first as u64).to_le_bytes());
        for value in [width as u32, 0, 0, 0] {
            record.extend(value.to_le_bytes());
        }
        Some((LATENT_TOKEN, record))
    }
}

/// The placeholder of a latent position of the toy family: the byte 0.
pub const LATENT_TOKEN: u32 = 0;

/// A token past the vocabulary is a client error that names `tokens`. `i` is its place in the request.
fn check_token(i: usize, token: u32) -> Result<(), FamilyError> {
    if token as usize >= toy::VOCAB {
        return Err(FamilyError::client_param(
            "tokens",
            format!(
                "token {i} is {token}, which is not a token: ids run 0 to {}",
                toy::VOCAB - 1
            ),
        ));
    }
    Ok(())
}

/// The bytes of one token: a byte token is its byte, and a marker is its text.
fn token_bytes(token: u32) -> Vec<u8> {
    if token < 256 {
        vec![token as u8]
    } else {
        toy::piece(token).into_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unchanged(_: Site, _: &mut [u32]) {}

    /// Change one token of a rendering that ends with a generation prompt. A read plan checks that its context's
    /// rendering is a prefix of the read, so the hook leaves the other renderings alone.
    fn change_context(site: Site, tokens: &mut [u32]) {
        if site == Site::Context && matches!(tokens.last(), Some(&toy::ASST | &toy::THINK)) {
            tokens[0] += 1;
        }
    }

    fn change_prefix(site: Site, tokens: &mut [u32]) {
        if site == Site::Prefix {
            tokens[0] += 1;
        }
    }

    fn change_feed(site: Site, tokens: &mut [u32]) {
        if site == Site::Feed {
            tokens[0] += 1;
        }
    }

    #[test]
    fn a_hook_that_changes_nothing_keeps_the_template_id() {
        assert_eq!(
            ToyFamily::with_hook(unchanged).template_id(),
            ToyFamily::new().template_id()
        );
    }

    #[test]
    fn a_change_to_one_rendered_token_changes_the_template_id() {
        let base = ToyFamily::new().template_id().to_owned();
        for (name, hook) in [
            ("context", change_context as fn(Site, &mut [u32])),
            ("prefix", change_prefix),
            ("feed", change_feed),
        ] {
            let changed = ToyFamily::with_hook(hook);
            assert_ne!(changed.template_id(), base, "a changed {name} token");
            // Only the first hash changes. The chat template text is the same.
            assert_eq!(
                changed.template_id().split_once('+').unwrap().1,
                base.split_once('+').unwrap().1,
                "{name}"
            );
        }
    }
}
