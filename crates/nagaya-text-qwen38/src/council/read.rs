//! The council reads of the Qwen3.8 family.

use nagaya_council::family::{ControlHit, DecisionInput, FamilyError, RawRead, ReadPlan};

use super::{Qwen38Family, Site, refusal};

pub(crate) fn decision_read(_: &Qwen38Family, _: &DecisionInput<'_>) -> Result<Box<dyn ReadPlan>, FamilyError> {
    Err(FamilyError::internal("the Qwen3.8 decision read is not built yet"))
}

/// The tokens of a raw read. A letter is client text, so it is content and must be one token. A suffix is the
/// client's own structure: its marker text becomes the marker token (gogi "Raw reads").
pub(crate) fn raw_read(family: &Qwen38Family, letters: &[String], suffixes: &[String]) -> Result<RawRead, FamilyError> {
    let mut ids: Vec<u32> = Vec::new();
    for letter in letters {
        let tokens = family.encode(&family.content(letter)?)?;
        let [id] = tokens[..] else {
            return Err(FamilyError::client_param(
                "letters",
                format!(
                    "letter {letter:?} is {} tokens. Each letter must be exactly one token.",
                    tokens.len()
                ),
            ));
        };
        if ids.contains(&id) {
            return Err(FamilyError::client_param(
                "letters",
                format!("letter {letter:?} is the same token as an earlier letter. Send distinct letters."),
            ));
        }
        ids.push(id);
    }
    family.tweak(Site::Raw, &mut ids);
    let mut tokens = Vec::with_capacity(suffixes.len());
    let mut control_text: Vec<ControlHit> = Vec::new();
    for (i, suffix) in suffixes.iter().enumerate() {
        let place = format!("suffix:{i}");
        let mut t = family.tokenizer().encode(suffix).map_err(|e| match refusal(e) {
            FamilyError::Client(invalid) => FamilyError::client_param("suffixes", invalid.message),
            other => other,
        })?;
        family.tweak(Site::Raw, &mut t);
        tokens.push(t);
        control_text.extend(family.hits(&place, suffix));
    }
    Ok(RawRead {
        letters: ids,
        suffixes: tokens,
        control_text,
    })
}
