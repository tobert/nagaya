//! The retrieval gate of the cache formats: can the engine find a fact from early in a context
//! of about 409,600 positions, in each cache format? A format passes when its retrievals and its greedy text match
//! those of the f32 cache.
//!
//! One context, many questions. The context is one user message that holds books of PG-19 with several needles in it
//! (facts with random-looking answers, from a fixed seed), then an assistant turn. Each question is a later user turn.
//! The context tokens are the start of each full render, and the test fails when they are not.
//!
//! The prefill is the cost, so the test pays it once for each format. It reads the slot through the part calls of the
//! `Engine` trait and writes the parts to a snapshot file, keyed by the engine identity, the hash of the context tokens
//! and the format. Each question restores the snapshot, feeds its turn and decodes greedily. A restore is checked: the
//! BLAKE3 of the file, and the bits of the first logit rows of a pass after the restore against the pass after the
//! prefill. The reply is parsed by the family's reply parser. A retrieval is right when the content of the reply holds
//! the code word and the access number of the needle.
//!
//! The test measures. It asserts the soundness of the harness (the prefix, the snapshot, the bits of a restore), and it
//! prints the table and the verdict of the gate. A format that fails a depth is a result, not a failure of the test.
//!
//! The variables (none has a default except where it says so):
//! - `NAGAYA_SNAPSHOT_DIR`: the folder of the snapshots and the results. Unset: the test skips; with
//!   `NAGAYA_REQUIRE_MODELS` it fails. `NAGAYA_HAYSTACK_DIR`: the folder of the PG-19 `NNN.txt` files, the same way.
//!   `INKLING2_MODEL_DIR`, and the card.
//! - `NAGAYA_RET_SEED`: the seed of the needles and of the first book. The default is the constant `SEED` of the first
//!   context, which starts at book 0. Another seed gives other needles and other books, so the context is a new text. The
//!   seed goes into the key of each snapshot.
//! - `NAGAYA_RET_MODE`: `prefill` (make the snapshots that are missing), `ask` (ask the questions of the snapshots),
//!   `both` (default), or `report` (print the table of the saved results).
//! - `NAGAYA_RET_FORMATS`: a list of `f32`, `bf16`, `e4m3` (default all three, in this order).
//! - `NAGAYA_RET_CONTEXT`: the target count of context tokens (default 409,600). `NAGAYA_RET_DEPTHS`: the needles, in
//!   percent of the context (default `1,5,10,25,50,75,90`). `NAGAYA_RET_ANSWER_TOKENS`: the most tokens of a reply
//!   (default 512).
//! - `NAGAYA_RET_PREFILL_BUDGET_S` (default 4800) and `NAGAYA_RET_ASK_BUDGET_S` (default 3600): the seconds that a format
//!   may take. The test stops with an error at the next block or question after a budget. A watchdog ends the process
//!   with code 2 after the budgets and 600 s.
//! - `NAGAYA_RET_MUTATE`: a run that falsifies the harness. `no_needles` leaves the needles out of the context, and
//!   every retrieval must then be wrong. `zero_conv` and `flip_row` damage the restore, and the check of the bits must
//!   then fail.
//!
//! `flock <lock> cargo test -p nagaya-mk-inkling2-sm121 --features cuda --test
//!  cuda_retrieval -- --ignored --test-threads=1 --nocapture`
//!
//! The first run is a measurement of hours, so the test runs only with `--ignored`.

#![cfg(feature = "cuda")]

mod engine_support;
mod retrieval_support;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use nagaya_abi::Engine;
use nagaya_council::family::{Family, ReplyStart};
use nagaya_jinja::Json;
use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_text_inkling2::{Chat, Effort, Inkling2Family, Options, Tokenizer};

use engine_support::{config, open, watchdog};
use retrieval_support::gpu::{self, Mutation};
use retrieval_support::haystack::{assemble, check_prefix};
use retrieval_support::needles::{self, Needle};
use retrieval_support::snapshot::{Key, Reader, Writer, hex, token_hash};

/// The seed of the needles.
const SEED: u64 = 20_261_124;
const ACK: &str = "Understood. I have read the text.";

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |v| {
        v.parse().unwrap_or_else(|_| panic!("{name} is not a number"))
    })
}

fn require() -> bool {
    std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some()
}

/// A folder from a variable, or `None` with the one line of a skip. With `NAGAYA_REQUIRE_MODELS` an unset variable
/// fails.
fn folder(name: &str) -> Option<PathBuf> {
    match std::env::var_os(name) {
        Some(dir) => Some(PathBuf::from(dir)),
        None => {
            assert!(!require(), "{name} is unset and NAGAYA_REQUIRE_MODELS is set");
            eprintln!("cuda_retrieval: skipped, {name} is unset");
            None
        }
    }
}

fn message(role: &str, content: &str) -> Json {
    Json::Object(vec![
        ("role".to_owned(), Json::Str(role.to_owned())),
        ("content".to_owned(), Json::Str(content.to_owned())),
    ])
}

fn format_of(name: &str) -> CacheFormat {
    match name {
        "f32" => CacheFormat::F32,
        "bf16" => CacheFormat::Bf16,
        "e4m3" => CacheFormat::E4m3,
        other => panic!("the format {other} is not f32, bf16 or e4m3"),
    }
}

/// The seed of a run: `NAGAYA_RET_SEED`, or the constant `SEED` of the first context.
fn seed_of_run() -> u64 {
    std::env::var("NAGAYA_RET_SEED").map_or(SEED, |v| v.parse().expect("NAGAYA_RET_SEED is not a number"))
}

/// The context and the questions of one run.
struct Plan {
    /// `None` for the first context, and the seed for another one. It goes into the key of each snapshot.
    seed: Option<u64>,
    needles: Vec<Needle>,
    /// The context tokens, and their hash.
    context: Vec<u32>,
    hash: [u8; 32],
    /// The tokens after the context for each needle: the question turn and the generation prompt.
    tails: Vec<Vec<u32>>,
    /// Where each needle is, in percent of the context tokens.
    realized: Vec<f64>,
    max_tail: usize,
}

fn make_plan(model: &Path, haystack: &Path, mutate: &str, seed: u64) -> Plan {
    let tok = Arc::new(Tokenizer::from_file(&model.join("tokenizer.json")).unwrap_or_else(|e| panic!("{e}")));
    let template = std::fs::read_to_string(model.join("chat_template.jinja")).expect("the template");
    let chat = Chat::new(&template, Arc::clone(&tok)).unwrap_or_else(|e| panic!("{e}"));
    let options = Options {
        effort: Some(Effort::None),
    };
    let percents: Vec<u32> = std::env::var("NAGAYA_RET_DEPTHS")
        .unwrap_or_else(|_| "1,5,10,25,50,75,90".to_owned())
        .split(',')
        .map(|d| d.trim().parse().expect("a depth in percent"))
        .collect();
    let needles = needles::make(seed, &percents);
    let first = needles::first_book(seed, SEED);
    let put: &[Needle] = if mutate == "no_needles" { &[] } else { &needles };
    let target = env_usize("NAGAYA_RET_CONTEXT", 409_600);
    let started = Instant::now();
    let hay = assemble(
        |n| {
            let path = haystack.join(format!("{:03}.txt", (first + n) % 100));
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        },
        |text| tok.encode_content(text).unwrap_or_else(|e| panic!("{e}")).len(),
        target.saturating_sub(64 + 48 * needles.len()),
        put,
    );
    let render = |messages: &[Json]| {
        chat.render_with(messages, &[], options)
            .unwrap_or_else(|e| panic!("{e}"))
    };
    let context = render(&[message("user", &hay.text), message("assistant", ACK)]).tokens;
    // The header of the user message: the tokens before its content.
    let header = render(&[message("user", "")]).tokens.len() - 1;
    let reply_mark = tok
        .token_id("<|message_model|>")
        .expect("the vocabulary has <|message_model|>");
    let mut tails = Vec::new();
    for needle in &needles {
        let mut full = render(&[
            message("user", &hay.text),
            message("assistant", ACK),
            message("user", &needle.question()),
        ])
        .tokens;
        full.push(reply_mark);
        check_prefix(&context, &full, "the full render of a question").unwrap_or_else(|e| panic!("{e}"));
        tails.push(full[context.len()..].to_vec());
    }
    let realized = if mutate == "no_needles" {
        vec![f64::NAN; needles.len()]
    } else {
        hay.at
            .iter()
            .map(|at| {
                let before = tok
                    .encode_content(&hay.text[..*at])
                    .unwrap_or_else(|e| panic!("{e}"))
                    .len();
                (header + before) as f64 / context.len() as f64 * 100.0
            })
            .collect()
    };
    let max_tail = tails.iter().map(Vec::len).max().unwrap_or(0);
    eprintln!(
        "RET plan: {} context tokens ({} chars), needles at {}, questions of {} tokens at most; built in {:.1?}",
        context.len(),
        hay.text.len(),
        realized
            .iter()
            .map(|r| format!("{r:.2}%"))
            .collect::<Vec<_>>()
            .join(", "),
        max_tail,
        started.elapsed()
    );
    Plan {
        seed: (seed != SEED).then_some(seed),
        needles,
        hash: token_hash(&context),
        context,
        tails,
        realized,
        max_tail,
    }
}

/// What one question gave.
struct Answer {
    tokens: Vec<u32>,
    content: String,
    reasoning: String,
    ended: bool,
    right: bool,
    word: bool,
    number: bool,
    /// The answer is in the reasoning, whatever the content holds.
    in_reasoning: bool,
    hashes: Vec<String>,
    ms_per_token: f64,
}

/// Feed the question tail after a context of `context_len` tokens, and decode up to `max` tokens greedily.
fn answer(
    engine: &mut nagaya_mk_inkling2_sm121::engine::Inkling2Sm121Engine,
    family: &Inkling2Family,
    needle: &Needle,
    context_len: usize,
    tail: &[u32],
    max: usize,
) -> Answer {
    let logits = gpu::feed(engine, 0, context_len as u64, tail, true).expect("the tail gives logits");
    let mut parser = family
        .reply_parser(ReplyStart::prompt(false), &[])
        .unwrap_or_else(|e| panic!("{e:?}"));
    let (tokens, steps, hashes) = gpu::greedy(engine, 0, (context_len + tail.len()) as u64, logits, max, |token| {
        parser.push(token).unwrap_or_else(|e| panic!("{e:?}"));
        parser.ended()
    });
    let ended = parser.ended();
    let parsed = parser.finish(!ended);
    let (word, number) = needle.check(&parsed.content);
    Answer {
        right: word && number,
        in_reasoning: needle.right(&parsed.reasoning),
        word,
        number,
        content: parsed.content,
        reasoning: parsed.reasoning,
        ended,
        tokens,
        hashes,
        ms_per_token: if steps.is_empty() {
            0.0
        } else {
            steps.iter().map(Duration::as_secs_f64).sum::<f64>() / steps.len() as f64 * 1000.0
        },
    }
}

fn restore_from(
    engine: &mut nagaya_mk_inkling2_sm121::engine::Inkling2Sm121Engine,
    dir: &Path,
    key: &Key,
    mutation: Mutation,
) -> Result<Duration, String> {
    let reader = Reader::open(dir, key)?.ok_or_else(|| {
        format!(
            "no snapshot {} in {}; run the test with NAGAYA_RET_MODE=prefill first",
            key.file_name(),
            dir.display()
        )
    })?;
    gpu::restore(engine, 0, reader, mutation)
}

fn results_path(dir: &Path, key: &Key) -> PathBuf {
    dir.join(format!(
        "results-{}",
        key.file_name().trim_start_matches("snap-").replace(".bin", ".json")
    ))
}

fn mutation_of(mutate: &str) -> Mutation {
    match mutate {
        "zero_conv" => Mutation::ZeroConv,
        "flip_row" => Mutation::FlipRow,
        _ => Mutation::None,
    }
}

/// Make the snapshot of one format when it is missing, and check that a restore gives the bits of the prefill.
fn prefill_format(model: &Path, dir: &Path, name: &str, plan: &Plan, family: &Inkling2Family, mutate: &str) {
    let format = format_of(name);
    let answer_tokens = env_usize("NAGAYA_RET_ANSWER_TOKENS", 512);
    let budget = Duration::from_secs(env_usize("NAGAYA_RET_PREFILL_BUDGET_S", 4800) as u64);
    let slot_len = plan.context.len() + plan.max_tail + answer_tokens + 16;
    let started = Instant::now();
    let mut engine = open(config(model, &[slot_len], format));
    let key = Key {
        identity: gpu::identity_digest(&engine.identity()),
        tokens: plan.hash,
        format: name.to_owned(),
        seed: plan.seed,
    };
    eprintln!(
        "RET {name}: open {:.1?}; identity {}, context {}, slot of {slot_len} positions",
        started.elapsed(),
        &hex(&key.identity)[..16],
        &hex(&key.tokens)[..16]
    );
    if Reader::open(dir, &key).unwrap_or_else(|e| panic!("{e}")).is_some() {
        eprintln!("RET {name}: the snapshot {} exists; no prefill", key.file_name());
        return;
    }
    let _guard = watchdog(budget + Duration::from_secs(1800), "the prefill of a format");
    let taken = gpu::prefill(&mut engine, 0, &plan.context, name, budget);
    eprintln!(
        "RET {name}: PREFILL {} positions in {:.1}s ({:.0} positions/s)",
        plan.context.len(),
        taken.as_secs_f64(),
        plan.context.len() as f64 / taken.as_secs_f64()
    );
    let mut writer = Writer::create(dir, &key, plan.context.len() as u64).unwrap_or_else(|e| panic!("{e}"));
    let read = Instant::now();
    let bytes = gpu::snapshot(&mut engine, 0, &mut writer).unwrap_or_else(|e| panic!("{e}"));
    let read = read.elapsed();
    // The pass after the prefill: its first logit rows are the bits that a restore must give back.
    let first = answer(
        &mut engine,
        family,
        &plan.needles[0],
        plan.context.len(),
        &plan.tails[0],
        gpu::KEEP_ROWS,
    );
    writer.note = serde_json::json!({"needle": 0, "rows": first.hashes});
    let write = Instant::now();
    let size = writer.finish().unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "RET {name}: SNAPSHOT {bytes} bytes of parts read in {read:.1?}, file of {size} bytes written in {:.1?}",
        write.elapsed()
    );
    let restored = restore_from(&mut engine, dir, &key, mutation_of(mutate)).unwrap_or_else(|e| panic!("{e}"));
    let again = answer(
        &mut engine,
        family,
        &plan.needles[0],
        plan.context.len(),
        &plan.tails[0],
        gpu::KEEP_ROWS,
    );
    eprintln!(
        "RET {name}: RESTORE in {restored:.1?}; the bits of the first rows after the prefill and after the restore"
    );
    for (i, (a, b)) in first.hashes.iter().zip(&again.hashes).enumerate() {
        eprintln!(
            "RET {name}:   row {i}: {} {} {}",
            &a[..16],
            &b[..16],
            if a == b { "equal" } else { "DIFFERENT" }
        );
    }
    assert_eq!(
        first.hashes, again.hashes,
        "{name}: a restore and a pass must give the bits of the slot after its prefill"
    );
    assert_eq!(first.tokens, again.tokens);
}

fn json_of(a: &Answer) -> serde_json::Value {
    serde_json::json!({
        "tokens": a.tokens, "content": a.content, "reasoning": a.reasoning, "ended": a.ended, "right": a.right,
        "word": a.word, "number": a.number, "in_reasoning": a.in_reasoning, "rows": a.hashes,
        "ms_per_token": a.ms_per_token,
    })
}

fn ask_format(model: &Path, dir: &Path, name: &str, plan: &Plan, family: &Inkling2Family, mutate: &str) {
    let format = format_of(name);
    let answer_tokens = env_usize("NAGAYA_RET_ANSWER_TOKENS", 512);
    let budget = Duration::from_secs(env_usize("NAGAYA_RET_ASK_BUDGET_S", 3600) as u64);
    let slot_len = plan.context.len() + plan.max_tail + answer_tokens + 16;
    let _guard = watchdog(budget + Duration::from_secs(600), "the questions of a format");
    let started = Instant::now();
    let mut engine = open(config(model, &[slot_len], format));
    let key = Key {
        identity: gpu::identity_digest(&engine.identity()),
        tokens: plan.hash,
        format: name.to_owned(),
        seed: plan.seed,
    };
    let note = Reader::open(dir, &key)
        .unwrap_or_else(|e| panic!("{e}"))
        .unwrap_or_else(|| {
            panic!(
                "no snapshot {} in {}; run the test with NAGAYA_RET_MODE=prefill first",
                key.file_name(),
                dir.display()
            )
        })
        .note;
    let mut rows = Vec::new();
    for (i, needle) in plan.needles.iter().enumerate() {
        assert!(
            started.elapsed() < budget,
            "{name}: the questions passed their budget of {budget:?} before question {i}"
        );
        let restored = restore_from(&mut engine, dir, &key, Mutation::None).unwrap_or_else(|e| panic!("{e}"));
        let asked = Instant::now();
        let a = answer(
            &mut engine,
            family,
            needle,
            plan.context.len(),
            &plan.tails[i],
            answer_tokens,
        );
        if i == 0 && note["rows"].is_array() {
            let want: Vec<String> = serde_json::from_value(note["rows"].clone()).unwrap();
            assert_eq!(
                a.hashes[..want.len().min(a.hashes.len())],
                want[..],
                "{name}: the first rows after a restore in this process differ from the rows after the prefill"
            );
        }
        eprintln!(
            "RET {name}: needle {} at {:.2}% ({}): {} | word {} number {} | in reasoning {} | {} tokens, ended {} | restore {restored:.1?}, question {:.1?}, {:.0} ms/token",
            i,
            plan.realized[i],
            needle.label,
            if a.right { "RIGHT" } else { "WRONG" },
            a.word,
            a.number,
            a.in_reasoning,
            a.tokens.len(),
            a.ended,
            asked.elapsed(),
            a.ms_per_token
        );
        eprintln!(
            "RET {name}:   content: {:?}",
            a.content.chars().take(300).collect::<String>()
        );
        rows.push(serde_json::json!({
            "label": needle.label, "percent": needle.percent, "realized_percent": plan.realized[i],
            "restore_s": restored.as_secs_f64(), "answer": json_of(&a),
        }));
    }
    let results = serde_json::json!({
        "format": name, "identity": hex(&key.identity), "context": hex(&key.tokens), "context_tokens": plan.context.len(),
        "mutate": mutate, "needles": rows,
    });
    let path = results_path(dir, &key);
    std::fs::write(&path, serde_json::to_vec_pretty(&results).unwrap())
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    eprintln!("RET {name}: results in {}", path.display());
    let right = results["needles"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|n| n["answer"]["right"] == true)
        .count();
    if mutate == "no_needles" {
        assert_eq!(right, 0, "{name}: a context with no needle must give no right answer");
    }
}

/// The table of the saved results of each format, and the verdict of the gate against f32.
fn report(dir: &Path, plan: &Plan, formats: &[&str]) {
    let mut found: Vec<(String, serde_json::Value)> = Vec::new();
    for name in formats {
        let prefix = format!("results-{name}-{}-", &hex(&plan.hash)[..16]);
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map(|d| d.flatten().map(|e| e.path()).collect())
            .unwrap_or_default();
        files.retain(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(&prefix)));
        files.sort();
        if let Some(path) = files.last() {
            let json = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            found.push(((*name).to_owned(), json));
        }
    }
    let Some((_, reference)) = found.iter().find(|(n, _)| n == "f32") else {
        eprintln!("RET report: no f32 results yet");
        return;
    };
    eprintln!("RET report: context of {} tokens", plan.context.len());
    eprintln!(
        "RET report: {:<7} {:<8} {:<6} retrieval | text equals f32 (first token that differs)",
        "needle", "depth %", "format"
    );
    let count = reference["needles"].as_array().unwrap().len();
    for i in 0..count {
        for (name, results) in &found {
            let n = &results["needles"][i];
            let a = &n["answer"];
            let r = &reference["needles"][i]["answer"];
            let (ta, tr) = (a["tokens"].as_array().unwrap(), r["tokens"].as_array().unwrap());
            let first = ta
                .iter()
                .zip(tr)
                .position(|(x, y)| x != y)
                .or((ta.len() != tr.len()).then_some(ta.len().min(tr.len())));
            eprintln!(
                "RET report: {:<7} {:<8.2} {:<6} {} (word {}, number {}) | {}",
                n["label"].as_str().unwrap(),
                n["realized_percent"].as_f64().unwrap_or(f64::NAN),
                name,
                if a["right"] == true { "RIGHT" } else { "WRONG" },
                a["word"],
                a["number"],
                first.map_or("identical".to_owned(), |t| format!("differs at token {t}"))
            );
        }
    }
    for (name, results) in found.iter().filter(|(n, _)| n != "f32") {
        let right = |r: &serde_json::Value, i: usize| r["needles"][i]["answer"]["right"] == true;
        let matches = (0..count)
            .filter(|i| right(results, *i) == right(reference, *i))
            .count();
        let kept = (0..count)
            .filter(|i| right(reference, *i) && right(results, *i))
            .count();
        let f32_right = (0..count).filter(|i| right(reference, *i)).count();
        let same = (0..count)
            .filter(|i| results["needles"][*i]["answer"]["tokens"] == reference["needles"][*i]["answer"]["tokens"])
            .count();
        eprintln!(
            "RET GATE {name}: retrievals match f32 in {matches} of {count}; right where f32 is right in {kept} of {f32_right}; greedy text equals f32 in {same} of {count}; verdict {}",
            if matches == count && kept == f32_right && same == count {
                "PASS"
            } else if matches == count && kept == f32_right {
                "PASS on the retrievals, the greedy text differs"
            } else {
                "FAIL"
            }
        );
    }
}

#[test]
#[ignore = "a measurement of hours; run it with --ignored, NAGAYA_SNAPSHOT_DIR and NAGAYA_HAYSTACK_DIR"]
fn each_cache_format_finds_a_fact_from_early_in_a_long_context() {
    let Some(model) = engine_support::model_dir("cuda_retrieval") else {
        return;
    };
    let Some(haystack) = folder("NAGAYA_HAYSTACK_DIR") else {
        return;
    };
    let Some(snapshots) = folder("NAGAYA_SNAPSHOT_DIR") else {
        return;
    };
    let mode = std::env::var("NAGAYA_RET_MODE").unwrap_or_else(|_| "both".to_owned());
    assert!(
        ["prefill", "ask", "both", "report"].contains(&mode.as_str()),
        "NAGAYA_RET_MODE is {mode}"
    );
    let mutate = std::env::var("NAGAYA_RET_MUTATE").unwrap_or_default();
    assert!(
        ["", "no_needles", "zero_conv", "flip_row"].contains(&mutate.as_str()),
        "NAGAYA_RET_MUTATE is {mutate}"
    );
    let list = std::env::var("NAGAYA_RET_FORMATS").unwrap_or_else(|_| "f32,bf16,e4m3".to_owned());
    let formats: Vec<&str> = list.split(',').map(str::trim).collect();
    let family = Inkling2Family::from_dir(&model).unwrap_or_else(|e| panic!("{e}"));
    let plan = make_plan(&model, &haystack, &mutate, seed_of_run());
    let total = Instant::now();
    for name in &formats {
        if mode != "ask" && mode != "report" {
            prefill_format(&model, &snapshots, name, &plan, &family, &mutate);
        }
        if mode != "prefill" && mode != "report" {
            ask_format(&model, &snapshots, name, &plan, &family, &mutate);
        }
    }
    report(&snapshots, &plan, &["f32", "bf16", "e4m3"]);
    eprintln!("RET total {:.1?}", total.elapsed());
}
