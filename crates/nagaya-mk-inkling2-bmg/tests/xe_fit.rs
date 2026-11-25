//! The gate of the memory check of an open on a B70 card (engine-abi "Inkling 2", "The cost of a
//! slot"). The driver puts a buffer that does not fit in system memory, and then each pass runs 5 to 6 times slower.
//! Thus the open of a slot past the card is refused, and it frees what it took.
//!
//! The card must hold no other engine. A test skips when the model is absent. Release the card first, and run alone:
//! `NAGAYA_TEST_CARDS=<pci> cargo test --release -p nagaya-mk-inkling2-bmg --features xe --test xe_fit --
//! --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use common::card;
use nagaya_mk_inkling2_bmg::Inkling2Engine;
use nagaya_mk_inkling2_bmg::spec::{Config, DEFAULT_EXTRA_RUNS};

fn config(slots: Vec<usize>) -> Config {
    Config {
        model_dir: card::model_dir(Which::Mini),
        slots,
        max_block: 512,
        extra_runs: DEFAULT_EXTRA_RUNS,
    }
}

/// The memory of this process on the card of the test.
fn memory() -> nagaya_mk_bmg::vram::CardMemory {
    nagaya_mk_bmg::vram::card_memory(&nagaya_mk_bmg::test_cards::card()).unwrap()
}

/// A slot of 262,144 positions plans about 40.6 GB on a card of 34.4 GB. Its open is refused, the error names the
/// card, and the memory of the card is free again: an open of a small slot passes after it. The largest slot of the
/// card run of 2026-11-21, 164,352 positions, opens.
#[test]
fn an_open_past_the_memory_of_the_card_is_refused_and_frees_what_it_took() {
    if common::model_absent(Which::Mini) {
        return;
    }
    let pci = nagaya_mk_bmg::test_cards::card();
    let start = memory();
    let Err(e) = Inkling2Engine::open(config(vec![262_144]), &pci) else {
        panic!("a slot of 262,144 positions opened on one card");
    };
    eprintln!("refused: {e}");
    assert!(e.contains(&format!("the card at {pci} did not hold the open")), "{e}");
    let after = memory();
    eprintln!("before {start:?}, after the refusal {after:?}");
    assert!(
        after.vram < start.vram + (1 << 30),
        "a refused open keeps no weights on the card"
    );
    for slots in [vec![4096], vec![164_352]] {
        let opened = Inkling2Engine::open(config(slots.clone()), &pci);
        eprintln!("{slots:?}: {:?} with the engine", memory());
        if let Err(e) = opened {
            panic!("an open of the slots {slots:?} after a refusal: {e}");
        }
    }
}
