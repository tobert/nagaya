//! The sampler refuses bad input. It never gives a token after an error (spec 8.5).

use nagaya_council::sampler::{SamplerError, Settings, pick, pick_with, survivors};

fn settings() -> Settings {
    Settings {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 20,
        min_p: 0.0,
        seed: 1,
    }
}

#[test]
fn a_non_finite_logit_is_an_error_whatever_the_settings() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let logits = [1.0, 2.0, bad, 0.0];
        for temperature in [0.0, 0.7] {
            let s = Settings {
                temperature,
                ..settings()
            };
            let err = pick_with(&logits, &s, 0.5).unwrap_err();
            assert!(matches!(err, SamplerError::NonFiniteLogit { id: 2, .. }), "{err:?}");
            assert!(survivors(&logits, &s).is_err());
            assert!(pick(&logits, &s, 3).is_err());
        }
    }
}

#[test]
fn empty_logits_are_an_error() {
    assert_eq!(pick_with(&[], &settings(), 0.5), Err(SamplerError::EmptyLogits));
}

#[test]
fn settings_out_of_range_are_errors() {
    let logits = [1.0, 2.0, 3.0];
    let bad = [
        Settings {
            temperature: -0.1,
            ..settings()
        },
        Settings {
            temperature: f64::NAN,
            ..settings()
        },
        Settings {
            temperature: f64::INFINITY,
            ..settings()
        },
        Settings {
            top_p: 0.0,
            ..settings()
        },
        Settings {
            top_p: 1.5,
            ..settings()
        },
        Settings {
            top_p: f64::NAN,
            ..settings()
        },
        Settings {
            min_p: -0.5,
            ..settings()
        },
        Settings {
            min_p: 1.0,
            ..settings()
        },
        Settings {
            min_p: f64::NAN,
            ..settings()
        },
    ];
    for s in bad {
        assert!(
            matches!(pick_with(&logits, &s, 0.5), Err(SamplerError::BadSetting(_))),
            "{s:?}"
        );
    }
}

#[test]
fn edge_settings_in_range_are_accepted() {
    let logits = [1.0, 2.0, 3.0];
    for s in [
        Settings {
            temperature: 0.0,
            ..settings()
        },
        Settings {
            top_p: 1.0,
            ..settings()
        },
        Settings {
            top_p: f64::MIN_POSITIVE,
            ..settings()
        },
        Settings {
            min_p: 0.999_999,
            ..settings()
        },
        Settings { top_k: 0, ..settings() },
        Settings {
            top_k: u64::MAX,
            ..settings()
        },
    ] {
        pick_with(&logits, &s, 0.5).unwrap_or_else(|e| panic!("{s:?}: {e}"));
    }
}

#[test]
fn a_uniform_value_outside_the_unit_interval_is_an_error() {
    for u in [-0.1, 1.0, 2.0, f64::NAN] {
        assert!(
            matches!(pick_with(&[1.0, 2.0], &settings(), u), Err(SamplerError::BadUniform(_))),
            "{u}"
        );
    }
}

#[test]
fn a_temperature_that_overflows_a_logit_is_an_error() {
    let s = Settings {
        temperature: 1e-320,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: 1,
    };
    assert_eq!(pick_with(&[1.0, 2.0], &s, 0.5), Err(SamplerError::TemperatureOverflow));
}

#[test]
fn greedy_takes_the_lowest_id_of_a_tie_and_draws_nothing() {
    let s = Settings {
        temperature: 0.0,
        ..settings()
    };
    let logits = [1.0, 5.0, 5.0, 2.0];
    let a = pick_with(&logits, &s, 0.0).unwrap();
    let b = pick_with(&logits, &s, 0.999).unwrap();
    assert_eq!((a.token, b.token), (1, 1));
}

#[test]
fn the_pick_uses_the_draw_at_the_position() {
    // With a flat row the token follows the uniform value, so different positions reach different tokens.
    let s = Settings {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: 99,
    };
    let logits = vec![0.0f32; 64];
    let tokens: std::collections::BTreeSet<u32> = (0..64).map(|pos| pick(&logits, &s, pos).unwrap().token).collect();
    assert!(tokens.len() > 20, "{} tokens", tokens.len());
    let u = nagaya_council::sampler::draw_uniform(99, 7, nagaya_council::sampler::Purpose::Sample);
    assert_eq!(pick(&logits, &s, 7).unwrap(), pick_with(&logits, &s, u).unwrap());
}

#[test]
fn a_uniform_value_on_a_cdf_edge_goes_to_the_next_token() {
    // The probabilities are 0.5 and 0.5, so the CDF is [0.5, 1.0]. The value 0.5 sits on the first edge.
    // The reference searches to the right, so the edge belongs to the token after it.
    let s = Settings {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: 1,
    };
    assert_eq!(pick_with(&[0.0, 0.0], &s, 0.5).unwrap().token, 1);
    assert_eq!(pick_with(&[0.0, 0.0], &s, 0.499_999).unwrap().token, 0);
}
