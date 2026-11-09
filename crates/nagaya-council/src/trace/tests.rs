use super::*;
use safetensors::SafeTensors;

/// A plan of a model of 3 layers of 4 values and a vocabulary of 6, with the end token 5.
fn plan(layers: Vec<u32>, vectors: Vec<Direction>, lens: bool, rows: bool) -> Plan {
    Plan {
        model: "toy".to_owned(),
        weight_hash: "fake:0".to_owned(),
        layers,
        model_layers: 3,
        width: 4,
        vocab: 6,
        top_k: 3,
        ends: vec![5],
        vectors,
        lens: if lens { Lens::Logit } else { Lens::None },
        rows,
    }
}

/// A readout of a fake lens: the logit row of a row of layer `l` is the row times `l + 1`, then the zeros, so the top
/// ids follow the row.
struct Scaled;

impl Readout for Scaled {
    fn logits(&self, rows: &[(u32, &[f32])]) -> Vec<Vec<f32>> {
        rows.iter()
            .map(|(l, h)| {
                let mut row: Vec<f32> = h.iter().map(|x| x * (*l + 1) as f32).collect();
                row.resize(6, 0.0);
                row
            })
            .collect()
    }
}

fn cpu_lens() -> Lens {
    Lens::Cpu {
        name: "jlens".to_owned(),
        id: "blake3:00ff".to_owned(),
        readout: std::sync::Arc::new(Scaled),
    }
}

fn residual(step: u32) -> Vec<f32> {
    (0..12).map(|i| (i as f32 - 5.0) * 0.5 + step as f32).collect()
}

fn lens_rows(step: u32) -> Vec<f32> {
    (0..18).map(|i| ((i * 7 + step) % 11) as f32 * 0.25).collect()
}

const LOGITS: [f32; 6] = [1.0, 3.0, -2.0, 3.0, 0.5, 2.0];

fn f32_of(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn u32_of(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}

#[test]
fn the_distribution_is_the_untempered_softmax_in_f64() {
    let (lse, entropy) = distribution(&LOGITS);
    let sum: f64 = LOGITS.iter().map(|x| (*x as f64).exp()).sum();
    assert!((lse - sum.ln()).abs() < 1e-12, "{lse} {}", sum.ln());
    let h: f64 = -LOGITS
        .iter()
        .map(|x| {
            let p = (*x as f64).exp() / sum;
            p * p.ln()
        })
        .sum::<f64>();
    assert!((entropy - h).abs() < 1e-12, "{entropy} {h}");
    // The bits are those of the sampler's untempered stats.
    let pick = crate::sampler::pick_with(
        &LOGITS,
        &crate::sampler::Settings {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: 0,
        },
        0.0,
    )
    .unwrap();
    assert_eq!(entropy.to_bits(), pick.h.to_bits());
}

#[test]
fn the_top_entries_come_largest_first_and_a_tie_goes_to_the_smaller_id() {
    assert_eq!(top(&LOGITS, 3), [1, 3, 5]);
    assert_eq!(top(&LOGITS, 10), [1, 3, 5, 0, 4, 2]);
}

#[test]
fn a_direction_is_the_unit_of_each_held_layer_and_none_elsewhere() {
    let vector: BTreeMap<u32, Vec<f32>> = [(0, vec![3.0, 0.0, 4.0, 0.0]), (2, vec![0.0; 4])].into();
    let d = Direction::new("blake3:v", &vector, 3, 4).unwrap();
    assert_eq!(d.units[0], Some(vec![0.6, 0.0, 0.8, 0.0]));
    assert_eq!(d.units[1], None, "a layer that the vector does not hold");
    assert_eq!(d.units[2], None, "a layer of zeros has no direction");
    let wide: BTreeMap<u32, Vec<f32>> = [(0, vec![1.0; 5])].into();
    assert!(Direction::new("blake3:v", &wide, 3, 4).is_err());
    let past: BTreeMap<u32, Vec<f32>> = [(3, vec![1.0; 4])].into();
    assert!(Direction::new("blake3:v", &past, 3, 4).is_err());
}

#[test]
fn a_step_records_each_tensor_of_the_traced_layers() {
    let vector: BTreeMap<u32, Vec<f32>> = [(0, vec![3.0, 0.0, 4.0, 0.0]), (2, vec![0.0, 1.0, 0.0, 0.0])].into();
    let d = Direction::new("blake3:v", &vector, 3, 4).unwrap();
    let mut t = Trace::new(plan(vec![2, 0], vec![d], true, true));
    t.record(3, &LOGITS, &residual(0), Some(&lens_rows(0))).unwrap();
    t.record(5, &LOGITS, &residual(1), Some(&lens_rows(1))).unwrap();
    let bytes = t.to_bytes();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let get = |name: &str| st.tensor(name).unwrap();
    let names: Vec<String> = {
        let mut n: Vec<String> = st.names().into_iter().map(str::to_owned).collect();
        n.sort();
        n
    };
    assert_eq!(
        names,
        [
            "along",
            "end_ids",
            "end_logprob",
            "entropy",
            "layers",
            "lens_ids",
            "lens_logprobs",
            "norms",
            "rows",
            "tokens",
            "top_ids",
            "top_logprobs"
        ]
    );
    assert_eq!(u32_of(get("tokens").data()), [3, 5]);
    assert_eq!(get("tokens").shape(), [2]);
    assert_eq!(u32_of(get("layers").data()), [2, 0]);
    assert_eq!(u32_of(get("end_ids").data()), [5]);
    let (lse, h) = distribution(&LOGITS);
    assert_eq!(f32_of(get("entropy").data()), [h as f32, h as f32]);
    assert_eq!(get("top_ids").shape(), [2, 3]);
    assert_eq!(u32_of(get("top_ids").data()), [1, 3, 5, 1, 3, 5]);
    assert_eq!(f32_of(get("top_logprobs").data())[0], (3.0 - lse) as f32);
    assert_eq!(get("end_logprob").shape(), [2, 1]);
    assert_eq!(f32_of(get("end_logprob").data()), [(2.0 - lse) as f32; 2]);
    // The norms of layer 2, then layer 0, at each step.
    let norm_of = |row: &[f32]| row.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt() as f32;
    let (r0, r1) = (residual(0), residual(1));
    assert_eq!(get("norms").shape(), [2, 2]);
    assert_eq!(
        f32_of(get("norms").data()),
        [
            norm_of(&r0[8..12]),
            norm_of(&r0[0..4]),
            norm_of(&r1[8..12]),
            norm_of(&r1[0..4])
        ]
    );
    // Along: layer 2 reads its unit (0, 1, 0, 0), layer 0 reads (0.6, 0, 0.8, 0).
    assert_eq!(get("along").shape(), [2, 2, 1]);
    let along = f32_of(get("along").data());
    assert_eq!(along[0], r0[9]);
    assert_eq!(along[1], (r0[0] as f64 * 0.6 + r0[2] as f64 * 0.8) as f32);
    assert_eq!(get("rows").shape(), [2, 2, 4]);
    let rows = f32_of(get("rows").data());
    assert_eq!(&rows[..4], &r0[8..12]);
    assert_eq!(&rows[4..8], &r0[0..4]);
    assert_eq!(get("lens_ids").shape(), [2, 2, 3]);
    let lens = lens_rows(0);
    let (want_ids, want_lp) = top_logprobs(&lens[12..18], 3);
    assert_eq!(&u32_of(get("lens_ids").data())[..3], &want_ids[..]);
    assert_eq!(&f32_of(get("lens_logprobs").data())[..3], &want_lp[..]);
    let (want_ids, _) = top_logprobs(&lens[0..6], 3);
    assert_eq!(&u32_of(get("lens_ids").data())[3..6], &want_ids[..]);
    let metadata = st_metadata(&bytes);
    assert_eq!(
        metadata,
        [
            ("lens", "logit"),
            ("model", "toy"),
            ("vectors", "blake3:v"),
            ("weight_hash", "fake:0")
        ]
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .into()
    );
}

fn st_metadata(bytes: &[u8]) -> BTreeMap<String, String> {
    let (_, meta) = SafeTensors::read_metadata(bytes).unwrap();
    meta.metadata().clone().unwrap().into_iter().collect()
}

#[test]
fn a_layer_with_no_direction_gives_nan_along() {
    let vector: BTreeMap<u32, Vec<f32>> = [(1, vec![1.0, 0.0, 0.0, 0.0])].into();
    let d = Direction::new("blake3:v", &vector, 3, 4).unwrap();
    let mut t = Trace::new(plan(vec![0, 1], vec![d], false, false));
    t.record(3, &LOGITS, &residual(0), None).unwrap();
    let bytes = t.to_bytes();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let along = f32_of(st.tensor("along").unwrap().data());
    assert!(along[0].is_nan());
    assert_eq!(along[1], residual(0)[4]);
    assert!(st.tensor("rows").is_err() && st.tensor("lens_ids").is_err());
}

#[test]
fn a_trace_with_no_vector_has_no_along() {
    let mut t = Trace::new(plan(vec![0], Vec::new(), false, false));
    t.record(3, &LOGITS, &residual(0), None).unwrap();
    let bytes = t.to_bytes();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    assert!(st.tensor("along").is_err());
    assert_eq!(st_metadata(&bytes)["vectors"], "");
}

#[test]
fn the_object_is_canonical() {
    let mut t = Trace::new(plan(vec![0, 1, 2], Vec::new(), true, false));
    for step in 0..3 {
        t.record(step, &LOGITS, &residual(step), Some(&lens_rows(step)))
            .unwrap();
    }
    let bytes = t.to_bytes();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    assert_eq!((8 + header_len) % 4096, 0, "the data starts on a 4 KiB boundary");
    let header = std::str::from_utf8(&bytes[8..8 + header_len]).unwrap();
    let json = header.trim_end_matches(' ');
    assert!(
        !json.contains(' ') && !json.contains('\n'),
        "no whitespace but the padding"
    );
    let value: serde_json::Value = serde_json::from_str(json).unwrap();
    let keys: Vec<&String> = value.as_object().unwrap().keys().collect();
    let mut sorted = keys.clone();
    sorted.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    assert_eq!(keys, sorted, "the keys are in bytewise order");
    // The data follows the names in order, with no gap.
    let mut at = 0;
    for key in keys.iter().filter(|k| k.as_str() != "__metadata__") {
        let offsets = &value[key.as_str()]["data_offsets"];
        assert_eq!(offsets[0].as_u64(), Some(at), "{key}");
        at = offsets[1].as_u64().unwrap();
    }
    assert_eq!(8 + header_len + at as usize, bytes.len());
    assert_eq!(Trace::object_bytes(t.plan(), 3), bytes.len() as u64);
    assert!(Trace::object_bytes(t.plan(), 4) > bytes.len() as u64);
}

#[test]
fn the_same_records_give_the_same_bytes_with_any_count_of_threads() {
    let rows: Vec<Vec<f32>> = (0..5).map(|s| lens_rows(s)[..6].to_vec()).collect();
    let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
    let one = tops_of(&refs, 3, 1);
    for threads in [2, 3, 8] {
        assert_eq!(tops_of(&refs, 3, threads), one, "{threads} threads");
    }
    let make = || {
        let mut t = Trace::new(plan(vec![1, 2], Vec::new(), true, true));
        t.record(1, &LOGITS, &residual(0), Some(&lens_rows(0))).unwrap();
        t.to_bytes()
    };
    assert_eq!(make(), make());
}

#[test]
fn a_record_of_another_size_is_an_error_and_changes_nothing() {
    let mut t = Trace::new(plan(vec![0], Vec::new(), true, false));
    assert!(t.record(1, &LOGITS[..5], &residual(0), Some(&lens_rows(0))).is_err());
    assert!(t.record(1, &LOGITS, &residual(0)[..11], Some(&lens_rows(0))).is_err());
    assert!(t.record(1, &LOGITS, &residual(0), Some(&lens_rows(0)[..17])).is_err());
    assert!(t.record(1, &LOGITS, &residual(0), None).is_err());
    assert_eq!(t.steps(), 0);
    let mut plain = Trace::new(plan(vec![0], Vec::new(), false, false));
    assert!(plain.record(1, &LOGITS, &residual(0), Some(&lens_rows(0))).is_err());
}

#[test]
fn a_cpu_lens_reads_each_traced_layer_through_its_readout_and_the_object_names_the_lens() {
    let mut p = plan(vec![2, 0], vec![], false, false);
    p.lens = cpu_lens();
    let mut trace = Trace::new(p.clone());
    trace.record(1, &LOGITS, &residual(0), None).unwrap();
    let bytes = trace.to_bytes();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let ids = u32_of(st.tensor("lens_ids").unwrap().data());
    let logprobs = f32_of(st.tensor("lens_logprobs").unwrap().data());
    let mut want_ids = Vec::new();
    let mut want_logprobs = Vec::new();
    for l in [2u32, 0] {
        let h = &residual(0)[l as usize * 4..(l as usize + 1) * 4];
        let row = &Scaled.logits(&[(l, h)])[0];
        let (lse, _) = distribution(row);
        let top_ids = top(row, 3);
        want_logprobs.extend(top_ids.iter().map(|id| (row[*id as usize] as f64 - lse) as f32));
        want_ids.extend(top_ids);
    }
    assert_eq!(ids, want_ids);
    assert_eq!(logprobs, want_logprobs);
    let meta = st_metadata(&bytes);
    assert_eq!(meta["lens"], "jlens");
    assert_eq!(meta["lens_id"], "blake3:00ff");
    assert_eq!(Trace::object_bytes(&p, 1), bytes.len() as u64);
}

#[test]
fn a_cpu_lens_refuses_a_step_with_the_tap_lens_and_a_logit_lens_names_no_lens_id() {
    let mut p = plan(vec![0], vec![], false, false);
    p.lens = cpu_lens();
    let mut trace = Trace::new(p);
    let error = trace.record(1, &LOGITS, &residual(0), Some(&lens_rows(0))).unwrap_err();
    assert!(error.contains("tap lens"), "{error}");
    assert_eq!(trace.steps(), 0);
    let mut logit = Trace::new(plan(vec![0], vec![], true, false));
    logit.record(1, &LOGITS, &residual(0), Some(&lens_rows(0))).unwrap();
    let meta = st_metadata(&logit.to_bytes());
    assert_eq!(meta["lens"], "logit");
    assert!(!meta.contains_key("lens_id"), "{meta:?}");
}

/// A head with padding rows past the vocabulary, as LFM3's of 128,000 rows for 125,033 ids. The readout gives a row of
/// each head row, and the trace keeps the first `vocab`, as the engine does with its logits. A shorter row is an error.
struct Padded(usize);

impl Readout for Padded {
    fn logits(&self, rows: &[(u32, &[f32])]) -> Vec<Vec<f32>> {
        Scaled
            .logits(rows)
            .into_iter()
            .map(|mut row| {
                row.resize(self.0, 99.0);
                row
            })
            .collect()
    }
}

#[test]
fn a_cpu_lens_keeps_the_vocabulary_of_a_padded_head_and_refuses_a_short_row() {
    let with = |readout: Padded| {
        let mut p = plan(vec![2, 0], vec![], false, false);
        p.lens = Lens::Cpu {
            name: "jlens".to_owned(),
            id: "blake3:00ff".to_owned(),
            readout: std::sync::Arc::new(readout),
        };
        let mut trace = Trace::new(p);
        trace.record(1, &LOGITS, &residual(0), None).map(|()| trace.to_bytes())
    };
    let padded = with(Padded(9)).expect("a head of 9 rows for a vocabulary of 6");
    let plain = with(Padded(6)).unwrap();
    // The padding rows score 99, so a trace that read them would put ids 6 to 8 on top.
    assert_eq!(padded, plain);
    let error = with(Padded(5)).unwrap_err();
    assert!(error.contains("5 values"), "{error}");
}
