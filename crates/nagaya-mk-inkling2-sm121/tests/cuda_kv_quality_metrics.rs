//! Host tests of the measures of the cache quality harness. They use no GPU. They are the first half of the
//! falsification of the harness: the measures must say "equal" for equal rows and "different" for shifted rows.

mod kv_quality_support;

use kv_quality_support::{accuracy, measure_row, measure_window};

/// A deterministic row with a clear top token and a spread of other values.
fn row(seed: u32, width: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
    (0..width)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x % 10_000) as f32 / 1000.0 - 5.0
        })
        .collect()
}

fn window(rows: usize, width: usize) -> Vec<f32> {
    (0..rows).flat_map(|i| row(i as u32 + 1, width)).collect()
}

#[test]
fn a_row_against_itself_agrees_with_zero_kl() {
    let a = row(7, 4096);
    let m = measure_row(&a, &a);
    assert!(m.agrees);
    assert_eq!(m.kl, 0.0);
    assert_eq!(m.gap, 0.0);
    assert_eq!(m.rank, 0);
}

#[test]
fn a_window_against_itself_agrees_with_zero_kl() {
    let w = window(16, 4096);
    let m = measure_window(&w, &w, 4096, 0);
    assert_eq!((m.agreement, m.kl_mean, m.kl_max, m.rank_max), (1.0, 0.0, 0.0, 0));
}

#[test]
fn a_shifted_window_does_not_agree() {
    let w = window(16, 4096);
    let m = measure_window(&w, &w, 4096, 1);
    assert!(m.agreement < 0.5, "agreement {}", m.agreement);
    assert!(m.kl_mean > 0.1, "kl {}", m.kl_mean);
}

#[test]
fn a_small_noise_gives_a_small_positive_kl_and_a_known_gap() {
    let a = row(3, 4096);
    let mut b = a.clone();
    let top = kv_quality_support::argmax(&a);
    b[top] -= 0.25;
    let m = measure_row(&a, &b);
    assert!(m.kl > 0.0 && m.kl < 0.1, "kl {}", m.kl);
    assert!((m.gap - 0.25).abs() < 1e-6, "gap {}", m.gap);
}

#[test]
fn a_demoted_top_token_has_a_rank() {
    let a = row(5, 4096);
    let mut b = a.clone();
    let top = kv_quality_support::argmax(&a);
    let second = (0..a.len())
        .filter(|i| *i != top)
        .max_by(|x, y| a[*x].total_cmp(&a[*y]))
        .unwrap();
    b[second] = a[top] + 1.0;
    let m = measure_row(&a, &b);
    assert!(!m.agrees);
    assert_eq!(m.rank, 1);
}

#[test]
fn the_accuracy_counts_the_rows_with_the_real_next_token() {
    let w = window(4, 64);
    let next: Vec<u32> = (0..4)
        .map(|i| kv_quality_support::argmax(&w[i * 64..(i + 1) * 64]) as u32)
        .collect();
    assert_eq!(accuracy(&w, 64, &next), 1.0);
    let mut wrong = next.clone();
    wrong[0] = (wrong[0] + 1) % 64;
    assert_eq!(accuracy(&w, 64, &wrong), 0.75);
}
