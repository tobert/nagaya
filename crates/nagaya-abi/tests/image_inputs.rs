//! The inputs of an image job (engine-abi "Image inputs", spec 11.9b).

use std::sync::Arc;

use nagaya_abi::{ImageInit, steps_that_run};

/// `n = floor(steps * strength)` in f64, as Python's `int(num_inference_steps * strength)` (spec 11.9b).
#[test]
fn the_steps_that_run_are_the_floor_in_f64() {
    // The example of the spec: 90 * 0.7 is 62.99999999999999 in f64.
    assert_eq!(steps_that_run(90, 0.7), 62);
    // 10 * 0.7 is 7.000000000000001 in f64. In f32, 0.7 is 0.699999988, so f32 would give 6.
    assert_eq!(steps_that_run(10, 0.7), 7);
    assert_eq!(steps_that_run(30, 0.75), 22);
    assert_eq!(steps_that_run(30, 0.99), 29);
    assert_eq!(steps_that_run(30, 1.0), 30);
    assert_eq!(steps_that_run(30, 0.01), 0);
    assert_eq!(steps_that_run(1, 0.75), 0);
}

/// A debug print of the inputs leaves out the pixels, which are megabytes.
#[test]
fn a_debug_print_of_the_inputs_gives_their_sizes() {
    let init = ImageInit {
        pixels: Arc::new(vec![7; 12]),
        strength: 0.5,
        mask: Some(Arc::new(vec![1; 4])),
    };
    let text = format!("{init:?}");
    assert!(text.contains("pixels: 12") && text.contains("mask: Some(4)"), "{text}");
    assert!(!text.contains("7, 7"), "{text}");
}
