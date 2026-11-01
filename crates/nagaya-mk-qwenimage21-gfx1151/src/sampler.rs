//! The FlowMatch Euler schedule of Qwen-Image 2.1 (spec 11.9c, "The schedule"), on the host, bit for bit as
//! `pipeline.py:733-746` and `scheduler.py` compute it.

use nagaya_mk_hip::bf16;

/// The `mu` of the dynamic shift for `seq` latent tokens: a line through (256, 0.5) and (8192, 0.9), in f64.
pub fn mu(seq: usize) -> f64 {
    let m = (0.9 - 0.5) / (8192.0 - 256.0);
    let b = 0.5 - 256.0 * m;
    seq as f64 * m + b
}

/// The sigmas of `steps` steps for `seq` latent tokens, then a final 0. Each value is f32.
pub fn sigmas(steps: usize, seq: usize) -> Vec<f32> {
    assert!(steps > 0, "a schedule has at least one step");
    // linspace(1, 1/n, n) in f64, rounded to f32 (pipeline.py:735).
    let last = 1.0 / steps as f64;
    let linear: Vec<f32> = (0..steps)
        .map(|i| {
            if steps == 1 {
                1.0
            } else {
                (1.0 + (last - 1.0) * i as f64 / (steps - 1) as f64) as f32
            }
        })
        .collect();
    // The exponential time shift in f32: e^mu rounds to f32 first, as numpy 2 does (scheduler.py:655-656).
    let e = mu(seq).exp() as f32;
    let shifted: Vec<f32> = linear.iter().map(|&t| e / (e + (1.0f32 / t - 1.0f32))).collect();
    // The stretch to the terminal 0.02 (scheduler.py:262-281).
    let z_last = 1.0f32 - shifted[steps - 1];
    let scale = z_last / 0.98f32;
    let mut out: Vec<f32> = shifted.iter().map(|&s| 1.0f32 - (1.0f32 - s) / scale).collect();
    out.push(0.0);
    out
}

/// The timestep that the transformer reads at a sigma: `1000 sigma` in f32, rounded to bf16, divided by 1000 in
/// bf16 (pipeline.py:782, :786).
pub fn timestep(sigma: f32) -> f32 {
    bf16::round(bf16::round(1000.0f32 * sigma) / 1000.0)
}

/// One Euler step in place: `x + (next - sigma) v` in f32 from bf16 values, rounded to bf16 (scheduler.py:493-526).
pub fn euler(x: &mut [u16], v: &[u16], sigma: f32, next: f32) {
    assert_eq!(x.len(), v.len(), "a latent and its velocity have the same size");
    let dt = next - sigma;
    for (x, &v) in x.iter_mut().zip(v) {
        *x = bf16::from_f32(bf16::to_f32(*x) + dt * bf16::to_f32(v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_schedule_of_the_oracle() {
        // 256 by 256 is 256 latent tokens, so mu is 0.5 (the oracle metadata). The values are the oracle's `sigmas`
        // in f32, from steps.safetensors of tools/qwenimage21_golden.sh; the terminal sigma is 0.02 in f32 math.
        assert_eq!(mu(256), 0.5);
        let want = [1.0f32, 0.744_611_4, 0.426_673_47, 0.019_999_98, 0.0];
        assert_eq!(sigmas(4, 256), want);
    }

    #[test]
    fn the_timestep_rounds_twice_in_bf16() {
        // 1000 * 0.7446 = 744.6 rounds to 744 in bf16; 744 / 1000 = 0.744 rounds to 0.74414.
        let t = timestep(0.7446);
        assert_eq!(t, bf16::round(bf16::round(744.6) / 1000.0));
        assert_ne!(t, 0.7446);
    }

    #[test]
    fn an_euler_step_moves_by_the_sigma_difference() {
        let mut x = vec![bf16::from_f32(1.0), bf16::from_f32(-2.0)];
        let v = vec![bf16::from_f32(4.0), bf16::from_f32(0.5)];
        euler(&mut x, &v, 0.75, 0.5);
        assert_eq!(x, vec![bf16::from_f32(0.0), bf16::from_f32(-2.125)]);
    }
}
