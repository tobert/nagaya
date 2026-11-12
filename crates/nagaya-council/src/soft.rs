//! Soft inputs (docs/lab.md, "Soft inputs in a request"): the next input of a reply mixes rows of the embedding table,
//! in place of the row of one token. No I/O.
//!
//! `moi` is Mixture of Inputs (Zhuang et al., arXiv 2505.14827). `gumbel` is Stochastic Soft Thinking with the
//! Gumbel-softmax (Wu et al., arXiv 2508.03440).

use crate::sampler::Survivors;

/// The most survivors that one mix reads.
pub const MAX_TOP: usize = 64;

/// The survivors that one mix reads when the request names no `top`.
pub const DEFAULT_TOP: usize = 16;

/// How the weights of a mix come from the distribution of the sampler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SoftMap {
    /// The sampled token, mixed with the distribution by a weight that grows with the entropy.
    Moi { beta: f64 },
    /// The Gumbel-softmax of the distribution. Its largest weight names the token.
    Gumbel { tau: f64 },
}

/// The tokens of a reply that take a soft input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Span {
    /// Each token of the reply.
    All,
    /// Only a token of the reasoning, before the token that ends it.
    Think,
}

/// The soft inputs of one request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftPlan {
    pub map: SoftMap,
    /// The most probable survivors that the mix reads, 1 to [`MAX_TOP`].
    pub top: usize,
    pub span: Span,
}

/// The weights of the rows of one soft input. The first entry is the token, and the weights sum to 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Mix {
    pub weights: Vec<(u32, f64)>,
}

impl Mix {
    /// The token that goes into the reply.
    pub fn token(&self) -> u32 {
        self.weights[0].0
    }

    /// True when the mix is one row, so the pass feeds the token itself.
    pub fn is_token(&self) -> bool {
        self.weights.len() == 1
    }

    /// The share of the rows other than the token: 1 minus the weight of the token.
    pub fn share(&self) -> f64 {
        if self.is_token() { 0.0 } else { 1.0 - self.weights[0].1 }
    }
}

/// The `top` most probable survivors, with their probabilities scaled to a sum of 1. The survivors come in the order
/// of the sampler: the most probable first.
fn cut(survivors: &Survivors, top: usize) -> Vec<(u32, f64)> {
    let n = top.min(survivors.ids.len());
    let sum: f64 = survivors.probs[..n].iter().sum();
    survivors.ids[..n]
        .iter()
        .zip(&survivors.probs[..n])
        .map(|(&id, &p)| (id, p / sum))
        .collect()
}

/// The entropy of the survivors, divided by the log of the vocabulary size. It is in [0, 1].
pub fn normed_entropy(survivors: &Survivors, vocab: usize) -> f64 {
    if vocab < 2 {
        return 0.0;
    }
    let h: f64 = survivors.probs.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    h / (vocab as f64).ln()
}

/// The mix of Mixture of Inputs: `(1 - a) onehot(token) + a p`, with `a = H / (beta + 1)`. `p` is the top of the
/// survivors, and `H` is the normed entropy of all survivors.
pub fn moi(survivors: &Survivors, token: u32, vocab: usize, top: usize, beta: f64) -> Mix {
    let a = normed_entropy(survivors, vocab) / (beta + 1.0);
    if a == 0.0 {
        return Mix {
            weights: vec![(token, 1.0)],
        };
    }
    let p = cut(survivors, top);
    let own = p.iter().find(|(id, _)| *id == token).map_or(0.0, |(_, p)| *p);
    let mut weights = vec![(token, (1.0 - a) + a * own)];
    weights.extend(p.into_iter().filter(|(id, _)| *id != token).map(|(id, p)| (id, a * p)));
    Mix { weights }
}

/// The uniform value in (0, 1) of the Gumbel noise of survivor `id` at a position. It depends on the seed, the position
/// and the id only.
pub fn gumbel_uniform(seed: u64, position: u64, id: u32) -> f64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"nagaya soft v1");
    hasher.update(&seed.to_le_bytes());
    hasher.update(&position.to_le_bytes());
    hasher.update(&id.to_le_bytes());
    let hash = hasher.finalize();
    let first = u64::from_le_bytes(hash.as_bytes()[..8].try_into().expect("a hash has 32 bytes"));
    // The middle of each of 2^53 cells, so neither 0 nor 1.
    ((first >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
}

/// The mix of the Gumbel-softmax: `softmax((ln p_i + g_i) / tau)` over the top of the survivors, with
/// `g_i = -ln(-ln u_i)`. The token is the survivor of the largest weight, the first one in a tie.
pub fn gumbel(survivors: &Survivors, top: usize, tau: f64, seed: u64, position: u64) -> Mix {
    let p = cut(survivors, top);
    let z: Vec<f64> = p
        .iter()
        .map(|&(id, p)| (p.ln() - (-gumbel_uniform(seed, position, id).ln()).ln()) / tau)
        .collect();
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    let first = z.iter().position(|&x| x == max).expect("a mix has a survivor");
    let mut weights = vec![(p[first].0, e[first] / sum)];
    weights.extend(
        p.iter()
            .zip(&e)
            .enumerate()
            .filter(|(i, _)| *i != first)
            .map(|(_, (&(id, _), &e))| (id, e / sum)),
    );
    if weights.len() == 1 {
        weights[0].1 = 1.0;
    }
    Mix { weights }
}

/// The input row of a mix: the sum of each weight times the row of its token, in `f64`, then `f32`.
pub fn row(mix: &Mix, width: usize, table: impl Fn(u32) -> Vec<f32>) -> Vec<f32> {
    let mut sum = vec![0.0f64; width];
    for &(id, w) in &mix.weights {
        let row = table(id);
        assert_eq!(
            row.len(),
            width,
            "the row of token {id} has another width than the table"
        );
        for (s, &x) in sum.iter_mut().zip(&row) {
            *s += w * f64::from(x);
        }
    }
    sum.into_iter().map(|x| x as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn survivors(ids: &[u32], probs: &[f64]) -> Survivors {
        Survivors {
            ids: ids.to_vec(),
            probs: probs.to_vec(),
        }
    }

    fn sum(mix: &Mix) -> f64 {
        mix.weights.iter().map(|(_, w)| w).sum()
    }

    #[test]
    fn one_survivor_gives_the_token_alone() {
        let one = survivors(&[7], &[1.0]);
        let mix = moi(&one, 7, 100, 16, 1.0);
        assert_eq!(mix.weights, vec![(7, 1.0)]);
        assert!(mix.is_token());
        assert_eq!(mix.share(), 0.0);
        let mix = gumbel(&one, 16, 0.5, 3, 9);
        assert_eq!(mix.weights, vec![(7, 1.0)]);
    }

    #[test]
    fn moi_mixes_the_token_with_the_distribution_by_the_normed_entropy() {
        let s = survivors(&[5, 2, 9], &[0.5, 0.3, 0.2]);
        let h = -(0.5f64 * 0.5f64.ln() + 0.3 * 0.3f64.ln() + 0.2 * 0.2f64.ln()) / 8f64.ln();
        assert_eq!(normed_entropy(&s, 8), h);
        let a = h / 2.0;
        let mix = moi(&s, 2, 8, 16, 1.0);
        assert_eq!(mix.weights, vec![(2, (1.0 - a) + a * 0.3), (5, a * 0.5), (9, a * 0.2)]);
        assert!((sum(&mix) - 1.0).abs() < 1e-12);
        assert!((mix.share() - a * 0.7).abs() < 1e-12);
        // A larger beta gives a smaller share.
        assert!(moi(&s, 2, 8, 16, 8.0).share() < mix.share());
    }

    #[test]
    fn the_top_cut_scales_the_mix_but_the_entropy_reads_every_survivor() {
        let s = survivors(&[5, 2, 9], &[0.5, 0.3, 0.2]);
        let a = normed_entropy(&s, 8) / 2.0;
        let mix = moi(&s, 9, 8, 2, 1.0);
        // The token 9 is out of the top 2, so it has no share of the distribution.
        assert_eq!(
            mix.weights,
            vec![(9, 1.0 - a), (5, a * (0.5 / 0.8)), (2, a * (0.3 / 0.8))]
        );
        assert!((sum(&mix) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gumbel_is_fixed_by_the_seed_and_the_position() {
        let s = survivors(&[5, 2, 9], &[0.5, 0.3, 0.2]);
        assert_eq!(gumbel(&s, 16, 0.5, 11, 40), gumbel(&s, 16, 0.5, 11, 40));
        let seeds: Vec<Mix> = (0..8).map(|seed| gumbel(&s, 16, 0.5, seed, 40)).collect();
        assert!(seeds.iter().any(|mix| mix != &seeds[0]));
        let mix = &seeds[0];
        assert!((sum(mix) - 1.0).abs() < 1e-12);
        assert!(mix.weights[1..].iter().all(|(_, w)| *w <= mix.weights[0].1));
    }

    #[test]
    fn the_token_of_gumbel_is_a_draw_from_the_distribution() {
        // Three survivors: with two, the difference of the noises is symmetric, so noise of the wrong sign passes too.
        let s = survivors(&[5, 2, 9], &[0.5, 0.3, 0.2]);
        let n = 20_000;
        for (id, p) in [(5, 0.5), (2, 0.3), (9, 0.2)] {
            let count = (0..n).filter(|&pos| gumbel(&s, 16, 0.5, 1, pos).token() == id).count();
            let rate = count as f64 / n as f64;
            // The standard error is 0.0036 at most, so 0.02 is more than 5 of them.
            assert!((rate - p).abs() < 0.02, "token {id}: {rate}, not {p}");
        }
    }

    #[test]
    fn a_small_tau_puts_the_weight_on_the_token() {
        let s = survivors(&[5, 2, 9], &[0.5, 0.3, 0.2]);
        let sharp = gumbel(&s, 16, 0.01, 4, 2);
        let soft = gumbel(&s, 16, 2.0, 4, 2);
        assert!(sharp.share() < 1e-6, "{}", sharp.share());
        assert!(soft.share() > sharp.share());
        assert_eq!(sharp.token(), soft.token());
    }

    #[test]
    fn the_row_is_the_weighted_sum_of_the_table_rows() {
        let mix = Mix {
            weights: vec![(1, 0.75), (0, 0.25)],
        };
        let table = |id: u32| if id == 0 { vec![4.0, 0.0] } else { vec![0.0, -2.0] };
        assert_eq!(row(&mix, 2, table), vec![1.0, -1.5]);
    }
}
