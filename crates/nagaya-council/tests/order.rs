//! The survivors keep the order of spec 11.2: logit down, then id up. The sampler's fast sort must give the order of
//! a plain comparison, with ties, minus zero and plus zero, and each filter.

use nagaya_council::sampler::{Settings, survivors};

/// The order of a plain comparison of the logits as numbers, then of the ids.
fn reference(logits: &[f32]) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    ids.sort_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap()
            .then(a.cmp(&b))
    });
    ids
}

fn row(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Few distinct values, so many ties; and both zeros.
            match (x >> 33) % 9 {
                0 => -0.0,
                1 => 0.0,
                v => (v as f32 - 5.0) * 0.75,
            }
        })
        .collect()
}

fn settings(top_k: u64, top_p: f64, min_p: f64) -> Settings {
    Settings {
        temperature: 1.0,
        top_p,
        top_k,
        min_p,
        seed: 1,
    }
}

#[test]
fn the_survivors_keep_the_order_of_a_plain_comparison_with_ties_and_both_zeros() {
    for seed in 0..20 {
        let logits = row(997, seed);
        let want = reference(&logits);
        let all = survivors(&logits, &settings(0, 1.0, 0.0)).unwrap();
        assert_eq!(all.ids, want, "seed {seed}, no filter");
        for k in [1, 5, 300] {
            let got = survivors(&logits, &settings(k, 1.0, 0.0)).unwrap();
            assert_eq!(got.ids, want[..k as usize], "seed {seed}, top_k {k}");
        }
        let got = survivors(&logits, &settings(0, 0.5, 0.0)).unwrap();
        assert_eq!(got.ids, want[..got.ids.len()], "seed {seed}, top_p");
    }
}

#[test]
fn minus_zero_and_plus_zero_tie_and_go_by_id() {
    let logits = [0.0, -0.0, -1.0, 0.0, -0.0];
    let got = survivors(&logits, &settings(0, 1.0, 0.0)).unwrap();
    assert_eq!(got.ids, [0, 1, 3, 4, 2]);
}
