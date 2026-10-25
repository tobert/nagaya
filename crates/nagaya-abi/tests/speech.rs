//! The speech capability (engine-abi "Speech"): an engine with no speech gives none.

#![cfg(feature = "fake")]

use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Capability, Engine};

#[test]
fn an_engine_with_no_speech_has_no_speech_capability() {
    let mut engine = FakeAlways::new();
    assert!(!engine.limits().capabilities.contains(&Capability::Speech));
    assert!(engine.speech().is_none());
}
