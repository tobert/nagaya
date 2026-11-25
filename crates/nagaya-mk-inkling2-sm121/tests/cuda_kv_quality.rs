//! The quality of the cache formats at depth. The f32 cache is the reference: a CPU
//! golden is not feasible at 100K positions. The test measures; it asserts only loose bounds of sanity.
//!
//! Part 1, teacher forcing. The test tokenizes books of PG-19 and prefills one slot in blocks of 512 rows. At each depth
//! it runs a window of rows with the logits of all of them, and it compares the bf16 and the e4m3 logits of each row
//! with the f32 logits (top token, KL over the vocabulary, gap of the top 64 logits, rank of the f32 top token). It
//! also gives the accuracy of each format against the real next token of the text.
//!
//! Part 2, greedy divergence. The f32 pass decodes 256 tokens greedily from each prompt. A format then decodes with
//! the same tokens as input (teacher forcing). The first row where its top token differs from the f32 token is the
//! first divergence of a free run, because the two runs agree up to that row.
//!
//! The variables (none has a default except where it says so):
//! - `NAGAYA_HAYSTACK_DIR`: the folder of the PG-19 `NNN.txt` files. Unset: the test skips; with
//!   `NAGAYA_REQUIRE_MODELS` it fails.
//! - `INKLING2_MODEL_DIR`, `INKLING2_GOLDEN_DIR` (oracle prompts of part 2; absent: only the text prompts), the card.
//! - `NAGAYA_KVQ_DEPTHS`: the depths, comma separated (default `4096,32768,131072`). `NAGAYA_KVQ_WINDOW`: rows of a
//!   window (default 16, at most 16). `NAGAYA_KVQ_STEPS`: greedy tokens (default 256).
//! - `NAGAYA_KVQ_DEPTH_BUDGET_S`: the seconds of prefill after which a format stops before the next depth (default 1800).
//! - `NAGAYA_KVQ_PARTS`: `windows`, `greedy` or `both` (default). `NAGAYA_KVQ_FORMATS`: a list of `bf16`, `e4m3`.
//! - `NAGAYA_KVQ_MUTATE`: `self` compares f32 with f32 (the test then asserts exact agreement and KL 0); `shift`
//!   compares each row with the next row (the loose bounds must then fail).
//!
//! `flock <lock> env INKLING2_MODEL_DIR=<folder> NAGAYA_HAYSTACK_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo
//!  test -p nagaya-mk-inkling2-sm121 --features cuda --test cuda_kv_quality -- --ignored --test-threads=1 --nocapture`
//!
//! The test is a measurement of about 27 minutes, not a gate of each commit, so it runs only with `--ignored`.

#![cfg(feature = "cuda")]

mod common;
mod kv_quality_support;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use kv_quality_support::{accuracy, argmax, measure_window};
use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_mk_inkling2_sm121::pass::step::{Pass, PassConfig};
use nagaya_mk_inkling2_sm121::pass::weights::DeviceWeights;
use nagaya_mk_inkling2_sm121::spec::{self, LOGITS};
use nagaya_text_inkling2::Tokenizer;

const MAX_ROWS: usize = 512;
const CYCLES: usize = 3;
/// The bounds of sanity. They catch a wrong row or a broken cache, not a loss of quality.
const MIN_AGREEMENT: f64 = 0.5;
const MAX_KL_MEAN: f64 = 2.0;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |v| {
        v.parse().unwrap_or_else(|_| panic!("{name} is not a number"))
    })
}

fn with_watchdog<T>(seconds: u64, what: &str, body: impl FnOnce() -> T) -> T {
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for _ in 0..seconds * 10 {
                std::thread::sleep(Duration::from_millis(100));
                if done.load(std::sync::atomic::Ordering::SeqCst) {
                    return;
                }
            }
            eprintln!("cuda_kv_quality: {what} did not finish within {seconds} s; it hangs");
            std::process::exit(2);
        });
        struct Done<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Done<'_> {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let _guard = Done(&done);
        body()
    })
}

fn require() -> bool {
    std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some()
}

/// The ids of the books, joined, until `want` ids are there. `first` is the number of the first book.
fn book_ids(tok: &Tokenizer, haystack: &std::path::Path, first: usize, want: usize) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut n = first;
    while ids.len() < want {
        let path = haystack.join(format!("{n:03}.txt"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        ids.extend(
            tok.encode_content(&text)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display())),
        );
        n += 1;
    }
    ids
}

fn oracle_prompts() -> Vec<(String, Vec<u32>)> {
    let dir = std::env::var_os("INKLING2_GOLDEN_DIR").map(PathBuf::from);
    let Some(path) = dir
        .map(|d| d.join("inkling2-mini.quant.oracle.json"))
        .filter(|p| p.exists())
    else {
        eprintln!("cuda_kv_quality: no oracle prompts, INKLING2_GOLDEN_DIR has no oracle file");
        return Vec::new();
    };
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    json["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|p| ["plain", "code", "chat", "long"].contains(&p["name"].as_str().unwrap()))
        .map(|p| {
            (
                p["name"].as_str().unwrap().to_owned(),
                p["ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect(),
            )
        })
        .collect()
}

struct Reference {
    /// The logits of each window, row by row.
    windows: Vec<Vec<f32>>,
    /// For each prompt: the greedy tokens and the f32 margin (top 1 minus top 2) before each of them.
    greedy: Vec<(Vec<u32>, Vec<f32>)>,
}

struct Plan {
    depths: Vec<usize>,
    window: usize,
    steps: usize,
    windows_on: bool,
    greedy_on: bool,
    text: Vec<u32>,
    prompts: Vec<(String, Vec<u32>)>,
}

fn margin(logits: &[f32]) -> f32 {
    let top = argmax(logits);
    let second = (0..logits.len())
        .filter(|i| *i != top)
        .map(|i| logits[i])
        .fold(f32::MIN, f32::max);
    logits[top] - second
}

/// Run both parts for one format. For f32 (`reference` is `None`) it returns the reference; for another format it
/// compares and prints, and returns the loose failures.
fn run_format(
    label: &str,
    format: CacheFormat,
    plan: &Plan,
    reference: Option<&Reference>,
    shift: usize,
    exact: bool,
) -> (Reference, Vec<String>) {
    let dir = PathBuf::from(std::env::var_os("INKLING2_MODEL_DIR").expect("INKLING2_MODEL_DIR"));
    let dev = common::gpu::open_cache(format);
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let longest_prompt = plan.prompts.iter().map(|p| p.1.len()).max().unwrap_or(1) + plan.steps;
    let seq = if plan.windows_on {
        plan.depths.iter().max().unwrap() + plan.window
    } else {
        1
    }
    .max(longest_prompt);
    let mut pass = Pass::open(
        &dev,
        &weights,
        &PassConfig {
            seqs: vec![seq],
            max_rows: MAX_ROWS,
            max_cycles: CYCLES,
            profile: false,
            format,
        },
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let mut failures = Vec::new();
    let mut out = Reference {
        windows: Vec::new(),
        greedy: Vec::new(),
    };

    if plan.windows_on {
        pass.slots[0].zero().unwrap();
        let started = Instant::now();
        let mut pos = 0usize;
        let budget = Duration::from_secs(env_usize("NAGAYA_KVQ_DEPTH_BUDGET_S", 1800) as u64);
        let depths: &[usize] = match reference {
            None => &plan.depths,
            Some(r) => &plan.depths[..r.windows.len()],
        };
        'depths: for (k, depth) in depths.iter().enumerate() {
            while pos < *depth {
                if started.elapsed() > budget {
                    eprintln!(
                        "KVQ {label}: the prefill passed {budget:?} before depth {depth}; this depth and the deeper ones are not measured"
                    );
                    break 'depths;
                }
                let n = (depth - pos).min(MAX_ROWS);
                with_watchdog(1800, "a prefill block", || {
                    pass.run_slot(0, CYCLES, &plan.text[pos..pos + n], &[])
                        .unwrap_or_else(|e| panic!("{e}"))
                });
                pos += n;
            }
            let before = started.elapsed();
            let rows: Vec<usize> = (0..plan.window).collect();
            let logits = with_watchdog(1800, "a window", || {
                pass.run_slot(0, CYCLES, &plan.text[pos..pos + plan.window], &rows)
                    .unwrap_or_else(|e| panic!("{e}"))
            });
            pos += plan.window;
            assert_eq!(logits.len(), plan.window * LOGITS);
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "{label}: a logit is not finite at depth {depth}"
            );
            let next = &plan.text[depth + 1..depth + 1 + plan.window];
            let acc = accuracy(&logits, LOGITS, next);
            let time = format!(
                "prefill to depth {before:.1?}, window {:.2?}",
                started.elapsed() - before
            );
            match reference {
                None => eprintln!("KVQ window {label} depth {depth}: accuracy vs text {acc:.3} | {time}"),
                Some(r) => {
                    let m = measure_window(&r.windows[k], &logits, LOGITS, shift);
                    eprintln!(
                        "KVQ window {label} depth {depth}: agreement {:.3} | KL mean {:.3e} max {:.3e} | max gap top64 {:.3} | rank of f32 top: max {} ({} rows not first) | accuracy vs text {acc:.3} | {time}",
                        m.agreement, m.kl_mean, m.kl_max, m.gap_max, m.rank_max, m.rank_nonzero
                    );
                    if exact {
                        assert_eq!(
                            (m.agreement, m.kl_mean),
                            (1.0, 0.0),
                            "{label}: f32 against f32 must be exact"
                        );
                    }
                    if m.agreement < MIN_AGREEMENT || m.kl_mean > MAX_KL_MEAN {
                        failures.push(format!(
                            "{label} depth {depth}: agreement {}, mean KL {}",
                            m.agreement, m.kl_mean
                        ));
                    }
                }
            }
            out.windows.push(logits);
        }
    }

    if plan.greedy_on {
        for (i, (name, ids)) in plan.prompts.iter().enumerate() {
            pass.slots[0].zero().unwrap();
            let mut logits = Vec::new();
            for block in ids.chunks(MAX_ROWS) {
                let want: Vec<usize> = vec![block.len() - 1];
                logits = pass.run_slot(0, CYCLES, block, &want).unwrap_or_else(|e| panic!("{e}"));
            }
            let mut tokens = Vec::new();
            let mut margins = Vec::new();
            let mut differs: Vec<usize> = Vec::new();
            for step in 0..plan.steps {
                let (token, m) = match reference {
                    None => (argmax(&logits) as u32, margin(&logits)),
                    Some(r) => {
                        let want = r.greedy[i].0[step];
                        if argmax(&logits) as u32 != want {
                            differs.push(step);
                        }
                        (want, r.greedy[i].1[step])
                    }
                };
                tokens.push(token);
                margins.push(m);
                if step + 1 < plan.steps {
                    logits = with_watchdog(600, "a decode step", || {
                        pass.run_slot(0, CYCLES, &[token], &[0])
                            .unwrap_or_else(|e| panic!("{e}"))
                    });
                }
            }
            match reference {
                None => eprintln!(
                    "KVQ greedy {label} prompt {name} ({} ids): {} tokens",
                    ids.len(),
                    tokens.len()
                ),
                Some(_) => {
                    let first = differs.first().copied();
                    eprintln!(
                        "KVQ greedy {label} prompt {name} ({} ids): first divergence {} | f32 margin there {} | rows differing in {} steps: {}",
                        ids.len(),
                        first.map_or("none".to_owned(), |s| s.to_string()),
                        first.map_or("-".to_owned(), |s| format!("{:.3}", margins[s])),
                        plan.steps,
                        differs.len()
                    );
                    if exact {
                        assert!(differs.is_empty(), "{label}: f32 against f32 must not diverge");
                    }
                }
            }
            out.greedy.push((tokens, margins));
        }
    }
    (out, failures)
}

#[test]
#[ignore = "a measurement of about 27 minutes; run it with --ignored and NAGAYA_HAYSTACK_DIR"]
fn the_cache_formats_keep_the_f32_logits_at_depth() {
    let Some(haystack) = std::env::var_os("NAGAYA_HAYSTACK_DIR").map(PathBuf::from) else {
        assert!(
            !require(),
            "NAGAYA_HAYSTACK_DIR is unset and NAGAYA_REQUIRE_MODELS is set"
        );
        eprintln!("cuda_kv_quality: skipped, NAGAYA_HAYSTACK_DIR is unset");
        return;
    };
    let model = match std::env::var_os("INKLING2_MODEL_DIR") {
        Some(d) if PathBuf::from(&d).join(spec::WEIGHTS_FILE).exists() => PathBuf::from(d),
        _ => {
            assert!(
                !require(),
                "INKLING2_MODEL_DIR has no checkpoint and NAGAYA_REQUIRE_MODELS is set"
            );
            eprintln!("cuda_kv_quality: skipped, INKLING2_MODEL_DIR has no checkpoint");
            return;
        }
    };
    let tok = Tokenizer::from_file(&model.join("tokenizer.json")).unwrap_or_else(|e| panic!("{e}"));
    let depths: Vec<usize> = std::env::var("NAGAYA_KVQ_DEPTHS")
        .unwrap_or_else(|_| "4096,32768,131072".to_owned())
        .split(',')
        .map(|d| d.trim().parse().expect("a depth"))
        .collect();
    assert!(
        depths.windows(2).all(|w| w[0] + 16 < w[1]),
        "the depths must rise by more than a window"
    );
    let window = env_usize("NAGAYA_KVQ_WINDOW", 16);
    assert!((1..=16).contains(&window));
    let steps = env_usize("NAGAYA_KVQ_STEPS", 256);
    let parts = std::env::var("NAGAYA_KVQ_PARTS").unwrap_or_else(|_| "both".to_owned());
    let (windows_on, greedy_on) = (parts != "greedy", parts != "windows");
    let mutate = std::env::var("NAGAYA_KVQ_MUTATE").unwrap_or_default();
    let formats: Vec<(&str, CacheFormat)> = std::env::var("NAGAYA_KVQ_FORMATS")
        .unwrap_or_else(|_| "bf16,e4m3".to_owned())
        .split(',')
        .map(|f| match f.trim() {
            "bf16" => ("bf16", CacheFormat::Bf16),
            "e4m3" => ("e4m3", CacheFormat::E4m3),
            other => panic!("the format {other} is not bf16 or e4m3"),
        })
        .collect();

    let text = if windows_on {
        let ids = book_ids(&tok, &haystack, 0, depths.last().unwrap() + window + 1);
        eprintln!("KVQ text: {} ids from the books of {}", ids.len(), haystack.display());
        ids
    } else {
        Vec::new()
    };
    let mut prompts = Vec::new();
    if greedy_on {
        prompts.extend(oracle_prompts());
        for (n, book) in [(1500usize, 20usize), (3000, 41)] {
            let ids = book_ids(&tok, &haystack, book, n + 1000);
            prompts.push((format!("pg19-{book:03}"), ids[1000..1000 + n].to_vec()));
        }
    }
    let plan = Plan {
        depths,
        window,
        steps,
        windows_on,
        greedy_on,
        text,
        prompts,
    };
    let total = Instant::now();
    let (reference, _) = run_format("f32", CacheFormat::F32, &plan, None, 0, false);
    let mut failures = Vec::new();
    for (label, format) in &formats {
        let (shift, run_as, exact) = match mutate.as_str() {
            "self" => (0, CacheFormat::F32, true),
            "shift" => (1, *format, false),
            "" => (0, *format, false),
            other => panic!("NAGAYA_KVQ_MUTATE is {other}; use self or shift"),
        };
        let (_, f) = run_format(label, run_as, &plan, Some(&reference), shift, exact);
        failures.extend(f);
    }
    eprintln!("KVQ total {:.1?}", total.elapsed());
    assert!(failures.is_empty(), "{}", failures.join("; "));
}
