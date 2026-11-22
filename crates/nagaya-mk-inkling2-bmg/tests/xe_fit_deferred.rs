//! The memory check of an open when the driver defers the memory of a buffer (engine-abi "Inkling 2", "The cost of a
//! slot"). Under memory pressure, Level Zero creates a buffer with `DRM_XE_GEM_CREATE_FLAG_DEFER_BACKING`. Then the
//! buffer has no memory until its first use, and fdinfo counts it in `drm-total-system`, not in `drm-total-vram0`.
//! On 2026-11-21 the service refused the open of Inkling 2 Mini for this reason: the ring was not yet in use.
//!
//! The test sets the debug keys of Level Zero that defer each buffer, so it does not need the pressure of the service.
//! Thus it is a test binary of its own. The card must hold no other engine. A test skips when the model is absent.
//! Release the card first, and run alone: `NAGAYA_TEST_CARDS=<pci> cargo test --release -p nagaya-mk-inkling2-bmg
//! --features xe --test xe_fit_deferred -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use common::card;
use nagaya_mk_bmg::level_zero::L0Cluster;
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::{INKLING2_KERNEL, KERNEL_ENTRY};
use nagaya_mk_inkling2_bmg::Inkling2Engine;
use nagaya_mk_inkling2_bmg::spec::Config;

/// The slots of the profile of the service on 2026-11-21: two slots of 32,768 positions, with 48 extra runs.
fn config(slots: Vec<usize>) -> Config {
    Config {
        model_dir: card::model_dir(Which::Mini),
        slots,
        max_block: 512,
        extra_runs: 48,
    }
}

/// The open of the profile of the service passes when the driver defers each buffer, and a slot past the card is still
/// refused. Before the open, the process holds the memory of the driver, as the service does after the unload of
/// another engine.
#[test]
fn an_open_that_fits_passes_when_the_driver_defers_the_memory_of_its_buffers() {
    if common::model_absent(Which::Mini) {
        return;
    }
    // SAFETY: this binary has one test, and no other thread of it reads the environment. The driver reads the keys at
    // its first init, which comes after this.
    unsafe {
        std::env::set_var("NEOReadDebugKeys", "1");
        std::env::set_var("EnableDeferBacking", "1");
        std::env::set_var("DeferBackingMemoryPressurePercent", "0");
    }
    let pci = nagaya_mk_bmg::test_cards::card();
    // A launch makes the driver take its own memory of the process, as the engine before the swap did.
    let cluster = L0Cluster::init_at(INKLING2_KERNEL, KERNEL_ENTRY, std::slice::from_ref(&pci)).expect("the card");
    let mut ring = Ring::new(&cluster.cards[0], 1);
    ring.set(&[]);
    // SAFETY: the ring holds only its halt.
    unsafe { ring.run(&cluster.cards[0]) };
    ring.free(&cluster.cards[0]);
    drop(cluster);
    let start = nagaya_mk_bmg::vram::card_memory(&pci).unwrap();
    eprintln!("the driver holds {start:?}");
    let opened = Inkling2Engine::open(config(vec![32_768, 32_768]), &pci);
    eprintln!("with the engine: {:?}", nagaya_mk_bmg::vram::card_memory(&pci).unwrap());
    match opened {
        Ok(engine) => drop(engine),
        Err(e) => panic!("the open of the profile of the service with deferred memory: {e}"),
    }
    let Err(e) = Inkling2Engine::open(config(vec![262_144]), &pci) else {
        panic!("a slot of 262,144 positions opened on one card with deferred memory");
    };
    eprintln!("refused: {e}");
    assert!(e.contains(&format!("the card at {pci} did not hold the open")), "{e}");
}
