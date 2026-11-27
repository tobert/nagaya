//! The helpers of the tests of the engine (`cuda` feature): the model folder, the config, the open and the segments.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use nagaya_abi::{OutputsWanted, PassOut, Segment, SegmentInputs, State};
use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_mk_cuda::test_cards;
use nagaya_mk_inkling2_sm121::engine::Inkling2Sm121Engine;
use nagaya_mk_inkling2_sm121::spec::{self, Config};

/// The model folder of `INKLING2_MODEL_DIR`, or `None` when the checkpoint is absent. `NAGAYA_REQUIRE_MODELS` makes an
/// absent checkpoint fail the test.
pub fn model_dir(who: &str) -> Option<PathBuf> {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    match std::env::var_os("INKLING2_MODEL_DIR") {
        Some(dir) if PathBuf::from(&dir).join(spec::WEIGHTS_FILE).exists() => Some(dir.into()),
        _ => {
            assert!(
                !require,
                "INKLING2_MODEL_DIR has no checkpoint and NAGAYA_REQUIRE_MODELS is set"
            );
            eprintln!("{who}: skipped, INKLING2_MODEL_DIR has no checkpoint");
            None
        }
    }
}

/// The config of an engine with slots of `slots` positions and the cache `cache`.
pub fn config(dir: &Path, slots: &[usize], cache: CacheFormat) -> Config {
    Config {
        model_dir: dir.to_owned(),
        slots: slots.to_vec(),
        max_block: spec::DEFAULT_MAX_BLOCK,
        extra_runs: spec::DEFAULT_EXTRA_RUNS,
        taps: false,
        cache,
    }
}

/// Open the engine on the GPU that `NAGAYA_TEST_CARDS` names.
pub fn open(config: Config) -> Inkling2Sm121Engine {
    Inkling2Sm121Engine::open(config, &test_cards::card()).unwrap_or_else(|e| panic!("{e}"))
}

/// The ids of the prompt `chat` of the golden file.
pub fn chat_ids() -> Vec<u32> {
    let dir = std::env::var_os("INKLING2_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/inkling2-golden")));
    let path = dir.join("inkling2-mini.quant.oracle.json");
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap();
    json["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "chat")
        .expect("the golden has the prompt chat")["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

/// `rows` ids that depend on `seed`, so that two slots hold other rows.
pub fn ids_of(chat: &[u32], seed: usize, at: usize, rows: usize) -> Vec<u32> {
    (at..at + rows)
        .map(|i| chat[(i * 7 + 3 + seed * 5) % chat.len()])
        .collect()
}

/// A segment of `ids` at positions from `from` in `slot`, with logits for the rows `logit_rows` (positions).
pub fn segment(slot: u32, from: u64, ids: &[u32], logit_rows: Vec<u64>, cycles: Option<u8>) -> Segment {
    Segment {
        control: Vec::new(),
        state: State::Slot(slot),
        inputs: SegmentInputs::Tokens(ids.to_vec()),
        positions: (from..from + ids.len() as u64).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted { logit_rows },
        layer_loop: None,
        cycles,
    }
}

/// The logits of a pass as bits, one vector for each row.
pub fn logit_bits(out: &PassOut) -> Vec<Vec<u32>> {
    out.outputs
        .iter()
        .map(|output| {
            output
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect()
        })
        .collect()
}

/// A watchdog: the process exits with code 2 when the guard lives longer than `limit`. A pass that hangs the GPU would
/// otherwise block the test and the lock of the GPU.
pub struct Watchdog(#[allow(dead_code)] std::sync::mpsc::Sender<()>);

pub fn watchdog(limit: std::time::Duration, what: &'static str) -> Watchdog {
    let (sender, receiver) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        // The guard drops its sender when the test ends, and then the receiver gives `Disconnected` at once.
        if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) = receiver.recv_timeout(limit) {
            eprintln!("{what}: no end within {limit:?}; the GPU may hang. The process exits");
            std::process::exit(2);
        }
    });
    Watchdog(sender)
}
