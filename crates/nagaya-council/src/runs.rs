//! The rows of each layer from the rows of each run (spec 11.17, "Rows of layers"). An engine that cycles gives the
//! row after each run in its tap `residual`, and a control vector and a trace read the row of each layer after its
//! last run. No I/O.

pub use nagaya_abi::TAP_EACH_RUN as EACH_RUN;

/// The layer of each run of a pass of `count` cycles over `layers` layers, when layers `start` to `end - 1` run once
/// in each cycle: layers 0 to `start - 1`, `count` times layers `start` to `end - 1`, then the other layers.
pub fn run_layers(layers: usize, start: usize, end: usize, count: usize) -> Vec<usize> {
    let mut out: Vec<usize> = (0..start).collect();
    for _ in 0..count {
        out.extend(start..end);
    }
    out.extend(end..layers);
    out
}

/// The rows of each layer after its last run, from `bytes`: for each logit row, the rows of `runs` of `width`
/// little-endian `f32` values, in the order of the runs. The result is for each logit row the rows of each layer, in
/// layer order. Another count of bytes is an error.
pub fn layer_rows(bytes: &[u8], runs: &[usize], layers: usize, width: usize) -> Result<Vec<u8>, String> {
    let row = width * 4;
    let per = runs.len() * row;
    if per == 0 || !bytes.len().is_multiple_of(per) {
        return Err(format!(
            "the tap residual has {} bytes, not whole sets of {} runs of {width} f32 values",
            bytes.len(),
            runs.len()
        ));
    }
    let mut last = vec![None; layers];
    for (r, layer) in runs.iter().enumerate() {
        let slot = last
            .get_mut(*layer)
            .ok_or_else(|| format!("run {r} is of layer {layer}, and the model has {layers} layers"))?;
        *slot = Some(r);
    }
    let last: Vec<usize> = last
        .into_iter()
        .enumerate()
        .map(|(l, r)| r.ok_or_else(|| format!("no run is of layer {l}")))
        .collect::<Result<_, _>>()?;
    let mut out = Vec::with_capacity(bytes.len() / runs.len() * layers);
    for set in bytes.chunks(per) {
        for r in &last {
            out.extend_from_slice(&set[r * row..(r + 1) * row]);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each layer takes the row of its last run, for each logit row. Falsified by the row of its first run.
    #[test]
    fn each_layer_takes_the_row_of_its_last_run() {
        let runs = run_layers(4, 1, 3, 3);
        assert_eq!(runs, [0, 1, 2, 1, 2, 1, 2, 3]);
        // Two logit rows of 8 runs of width 1: the value is 10 * logit row + run.
        let bytes: Vec<u8> = (0..2)
            .flat_map(|i| (0..8).map(move |r| (10 * i + r) as f32))
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let out = layer_rows(&bytes, &runs, 4, 1).unwrap();
        let values: Vec<f32> = out.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        assert_eq!(values, [0.0, 5.0, 6.0, 7.0, 10.0, 15.0, 16.0, 17.0]);
        // One cycle: each layer has one run, and the rows do not change.
        let one = run_layers(4, 1, 3, 1);
        let bytes: Vec<u8> = (0..4).flat_map(|v| (v as f32).to_le_bytes()).collect();
        assert_eq!(layer_rows(&bytes, &one, 4, 1).unwrap(), bytes);
        assert!(layer_rows(&bytes[..12], &one, 4, 1).unwrap_err().contains("12 bytes"));
        assert!(
            layer_rows(&bytes, &[0, 1, 1, 3], 4, 1)
                .unwrap_err()
                .contains("no run is of layer 2")
        );
    }
}
