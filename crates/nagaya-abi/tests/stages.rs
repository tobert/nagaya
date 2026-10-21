//! The rules of the stages and the feed rule in `Limits::check` (engine-abi D1, D11), and the residual buffer.

use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Capability, Engine, FeedRule, Lifetime, Limits, MandatoryTerm, RESIDUAL_TERM, Residual, Stage, Tier};

/// The limits of an engine with two stages on two devices, a feed rule of 4 tokens, and a residual row of 16 bytes.
fn staged() -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.capabilities = vec![
        Capability::Parts,
        Capability::Slots,
        Capability::Coverage,
        Capability::Stages,
    ];
    limits.devices = vec![0, 1];
    limits.stages = vec![
        Stage {
            device: 0,
            layers: 0..1,
            head: false,
        },
        Stage {
            device: 1,
            layers: 1..2,
            head: true,
        },
    ];
    limits.parts[2].stage = 1;
    limits.segments_exact = false;
    limits.max_pass_inputs = 8;
    limits.feed = Some(FeedRule { block: 4 });
    limits.residual_row_bytes = 16;
    limits.memory.mandatory.push(MandatoryTerm {
        name: RESIDUAL_TERM.to_owned(),
        tier: Tier::Host,
        bytes: 2 * 4 * 16,
        lifetime: Lifetime::Engine,
        kinds: Vec::new(),
    });
    limits
}

fn refused(limits: &Limits, words: &str) {
    match limits.check() {
        Ok(()) => panic!("the limits must break a rule about {words}"),
        Err(message) => assert!(message.contains(words), "the message `{message}` must name {words}"),
    }
}

#[test]
fn the_fake_and_a_staged_engine_keep_the_rules() {
    FakeAlways::new().limits().check().unwrap();
    staged().check().unwrap();
    assert_eq!(staged().residual_rows(), 4);
    assert_eq!(staged().residual_bytes(), 128);
}

#[test]
fn a_feed_rule_needs_a_block_that_a_pass_holds_and_no_exact_segments() {
    let mut limits = FakeAlways::new().limits();
    limits.segments_exact = false;
    limits.feed = Some(FeedRule { block: 0 });
    refused(&limits, "feed block");
    limits.feed = Some(FeedRule {
        block: u64::from(limits.max_pass_inputs) + 1,
    });
    refused(&limits, "feed block");
    limits.feed = Some(FeedRule { block: 4 });
    limits.check().unwrap();
    limits.segments_exact = true;
    refused(&limits, "segments_exact");
    limits.segments_exact = false;
    limits.checkpoint_grain = Some(256);
    refused(&limits, "checkpoint grain");
}

#[test]
fn a_part_names_a_stage_that_the_engine_has() {
    let mut limits = FakeAlways::new().limits();
    limits.parts[1].stage = 1;
    refused(&limits, "names stage 1");
    let mut limits = staged();
    limits.parts[1].stage = 2;
    refused(&limits, "names stage 2");
}

#[test]
fn each_stage_has_its_own_device_and_no_seal_or_grain() {
    let mut limits = staged();
    limits.stages[1].device = 0;
    refused(&limits, "device 0");
    let mut limits = staged();
    limits.stages.truncate(1);
    limits.parts[2].stage = 0;
    refused(&limits, "at least two");
    let mut limits = staged();
    limits.capabilities.push(Capability::Seal);
    refused(&limits, "no seal");
    let mut limits = staged();
    limits.feed = None;
    limits.checkpoint_grain = Some(4);
    limits.segments_exact = true;
    refused(&limits, "no seal and no checkpoint grain");
}

#[test]
fn the_residual_term_holds_two_buffers_at_each_boundary() {
    let mut limits = staged();
    limits.residual_row_bytes = 0;
    refused(&limits, "residual_row_bytes");
    let mut limits = staged();
    limits.memory.mandatory.retain(|term| term.name != RESIDUAL_TERM);
    refused(&limits, "the mandatory term `residual`");
    let mut limits = staged();
    limits.memory.mandatory.last_mut().unwrap().bytes = 64;
    refused(&limits, "it must be 128 host bytes");
    let mut limits = staged();
    limits.memory.mandatory.last_mut().unwrap().tier = Tier::Device(0);
    refused(&limits, "it must be 128 host bytes");
    // With no feed rule, a buffer holds the most inputs of a pass.
    let mut limits = staged();
    limits.feed = None;
    assert_eq!(limits.residual_bytes(), 2 * 8 * 16);
}

#[test]
fn a_residual_refuses_more_rows_than_it_holds_and_keeps_its_rows() {
    let mut residual = Residual::new(4, 2);
    residual.write(2).unwrap().copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(residual.rows(), 2);
    let error = residual.write(3).unwrap_err();
    assert!(error.contains("holds 2"), "{error}");
    assert_eq!(residual.bytes(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    residual.write(1).unwrap();
    assert_eq!(residual.bytes(), &[1, 2, 3, 4]);
}
