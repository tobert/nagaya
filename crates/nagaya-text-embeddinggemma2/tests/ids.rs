//! The ids of an input against sentence-transformers 6.1.0 (spec 11.10, "The family of EmbeddingGemma 2"; 8.3).

mod common;

use nagaya_text_embeddinggemma2::{EmbeddingGemma2Family, Error, TASKS};

/// The gogi task of a name in the model's prompts.
fn task_of(prompt_name: &str) -> &'static str {
    TASKS
        .iter()
        .find(|(_, name)| *name == prompt_name)
        .map(|(task, _)| *task)
        .unwrap_or_else(|| panic!("no task has the prompt name {prompt_name}"))
}

#[test]
fn the_ids_of_the_fixture_match_sentence_transformers() {
    if common::absent() {
        return;
    }
    let family = common::family();
    let fixture = common::fixture("ids.json");
    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() >= 18, "the fixture has {} cases", cases.len());
    for case in cases {
        let task = task_of(case["task"].as_str().unwrap());
        let text = case["text"].as_str().unwrap();
        assert_eq!(family.prompt(task).unwrap(), case["prompt"].as_str().unwrap(), "{task}");
        let got = family.ids(task, text).unwrap();
        assert_eq!(got, common::ids(&case["ids"]), "{} {task}", case["case"]);
        assert_eq!(family.tokens(task, text).unwrap(), got.len(), "{} {task}", case["case"]);
    }
}

/// The paragraph of the long cases of `tools/embeddinggemma2_reference.py`.
const PARAGRAPH: &str = "The nagaya is one process per machine that hosts our engines. It admits each call before it \
allocates, and it keeps its store of content ids, recipes and entries on the disk. A ledger holds the budgets of the \
devices. ";

/// The eight texts of the oracle, as `cases()` of `tools/embeddinggemma2_reference.py` makes them. The short texts
/// come from the fixture. `near_limit` is the most words of 64 paragraphs that give at most 2040 tokens with the
/// prompt of `document`, as the reference searches for it.
fn oracle_texts(family: &EmbeddingGemma2Family) -> Vec<(String, String)> {
    let mut texts: Vec<(String, String)> = Vec::new();
    for case in common::fixture("ids.json")["cases"].as_array().unwrap() {
        let name = case["case"].as_str().unwrap().to_owned();
        if !texts.iter().any(|(n, _)| *n == name) {
            texts.push((name, case["text"].as_str().unwrap().to_owned()));
        }
    }
    texts.push(("long".to_owned(), PARAGRAPH.repeat(16)));
    let words: Vec<String> = PARAGRAPH.repeat(64).split(' ').map(str::to_owned).collect();
    let (mut lo, mut hi) = (0, words.len());
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if family.tokens("document", &words[..mid].join(" ")).unwrap() <= 2040 {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    texts.push(("near_limit".to_owned(), words[..lo].join(" ")));
    texts
}

#[test]
fn the_ids_of_every_oracle_case_match_sentence_transformers() {
    if common::absent() {
        return;
    }
    let (Some(cases), Some(tensors)) = (common::golden("cases.json"), common::golden("golden.safetensors")) else {
        return;
    };
    let family = common::family();
    let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(&cases).unwrap()).unwrap();
    let names: Vec<&str> = cases["tasks"].as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(names.len(), 8);
    let texts = oracle_texts(family);
    assert_eq!(texts.len(), 8, "{:?}", texts.iter().map(|(n, _)| n).collect::<Vec<_>>());
    let mut checked = 0;
    for (case, text) in &texts {
        for name in &names {
            let key = format!("{case}.{name}.ids");
            let got = family.ids(task_of(name), text).unwrap();
            assert_eq!(got, common::safetensors_ids(&tensors, &key), "{key}");
            checked += 1;
        }
    }
    // The token counts that cases.json records agree too.
    for case in cases["cases"].as_array().unwrap() {
        let (name, task) = (case["case"].as_str().unwrap(), case["task"].as_str().unwrap());
        let text = &texts.iter().find(|(n, _)| n == name).unwrap().1;
        let tokens = case["tokens"].as_u64().unwrap() as usize;
        assert_eq!(family.tokens(task_of(task), text).unwrap(), tokens, "{name} {task}");
    }
    assert_eq!(checked, 64);
}

#[test]
fn the_ids_have_bos_first_and_eos_last() {
    if common::absent() {
        return;
    }
    let ids = common::family().ids("query", "a fox").unwrap();
    assert_eq!(ids.first(), Some(&2));
    assert_eq!(ids.last(), Some(&1));
}

#[test]
fn a_special_token_that_the_text_spells_stays_text() {
    if common::absent() {
        return;
    }
    let family = common::family();
    let ids = family.ids("document", "one <eos> two <bos> three").unwrap();
    assert_eq!(ids.iter().filter(|&&id| id == 1).count(), 1, "{ids:?}");
    assert_eq!(ids.iter().filter(|&&id| id == 2).count(), 1, "{ids:?}");
}

#[test]
fn an_unknown_task_is_an_error_that_lists_the_tasks() {
    if common::absent() {
        return;
    }
    let error = common::family().ids("search", "a fox").unwrap_err();
    assert!(matches!(error, Error::Task { .. }), "{error:?}");
    let message = error.to_string();
    assert!(message.contains("\"search\""), "{message}");
    for (task, _) in TASKS {
        assert!(message.contains(task), "{message}");
    }
}

#[test]
fn a_config_without_the_prompt_of_a_task_fails_at_the_open() {
    if common::absent() {
        return;
    }
    let dir = common::model_dir();
    let tokenizer = std::fs::read(dir.join("tokenizer.json")).unwrap();
    let config = std::fs::read(dir.join("config_sentence_transformers.json")).unwrap();
    let mut config: serde_json::Value = serde_json::from_slice(&config).unwrap();
    config["prompts"].as_object_mut().unwrap().remove("FactChecking");
    let config = serde_json::to_vec(&config).unwrap();
    let error = EmbeddingGemma2Family::from_bytes(&tokenizer, &config).expect_err("the open must fail");
    assert!(matches!(error, Error::Prompt { .. }), "{error:?}");
    let message = error.to_string();
    assert!(
        message.contains("fact_checking") && message.contains("FactChecking"),
        "{message}"
    );
}

#[test]
fn the_tokenizer_hash_is_the_sha256_of_the_file() {
    if common::absent() {
        return;
    }
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(common::model_dir().join("tokenizer.json")).unwrap();
    let hex: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(common::family().tokenizer_hash(), format!("sha256:{hex}"));
}
