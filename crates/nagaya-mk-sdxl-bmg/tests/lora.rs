//! The LoRA files of an image request (spec 11.9a): the keys, the scale, the shapes and the delta. No card is needed.
//!
//! The files are made here, in safetensors form, so no test reads a model.

use nagaya_mk_sdxl_bmg::lora::{Module, Namespace, Target, delta, parse};

const UNET_TO_Q: &str = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q";
const TE2_Q: &str = "text_model.encoder.layers.0.self_attn.q_proj";

fn targets() -> Vec<Target> {
    vec![
        Target {
            namespace: Namespace::Unet,
            name: UNET_TO_Q.to_owned(),
            cin: 4,
            cout: 3,
            kernel: 1,
        },
        Target {
            namespace: Namespace::TextEncoder2,
            name: TE2_Q.to_owned(),
            cin: 8,
            cout: 8,
            kernel: 1,
        },
    ]
}

/// One tensor of a safetensors file: its name, its dtype and shape, and its bytes.
struct Tensor {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn f32s(name: &str, shape: &[usize], values: &[f32]) -> Tensor {
    Tensor {
        name: name.to_owned(),
        dtype: "F32",
        shape: shape.to_vec(),
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// The bytes of a safetensors file of `tensors`, in the order given.
fn file(tensors: &[Tensor]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    let mut data = Vec::new();
    for t in tensors {
        let end = offset + t.data.len();
        header.insert(
            t.name.clone(),
            serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [offset, end]}),
        );
        data.extend_from_slice(&t.data);
        offset = end;
    }
    let text = serde_json::Value::Object(header).to_string().into_bytes();
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend(text);
    out.extend(data);
    out
}

fn kohya_unet(module: &str, rank: usize, cin: usize, cout: usize) -> Vec<Tensor> {
    let base = format!("lora_unet_{}", module.replace('.', "_"));
    let down: Vec<f32> = (0..rank * cin).map(|i| i as f32 * 0.5).collect();
    let up: Vec<f32> = (0..cout * rank).map(|i| 1.0 - i as f32 * 0.25).collect();
    vec![
        f32s(&format!("{base}.lora_down.weight"), &[rank, cin], &down),
        f32s(&format!("{base}.lora_up.weight"), &[cout, rank], &up),
    ]
}

#[test]
fn a_kohya_key_maps_to_its_unet_module_with_scale_weight_times_alpha_over_rank() {
    let mut tensors = kohya_unet(UNET_TO_Q, 2, 4, 3);
    tensors.push(f32s(
        &format!("lora_unet_{}.alpha", UNET_TO_Q.replace('.', "_")),
        &[],
        &[1.0],
    ));
    let lora = parse(&file(&tensors), 0.5, &targets()).expect("a valid file");
    assert_eq!(lora.modules.len(), 1);
    let m = &lora.modules[0];
    assert_eq!(m.target.name, UNET_TO_Q);
    assert_eq!(m.rank, 2);
    // 0.5 * 1.0 / 2
    assert!((m.scale - 0.25).abs() < 1e-7, "scale {}", m.scale);
}

#[test]
fn diffusers_names_on_the_second_text_encoder_scale_by_the_weight_when_alpha_is_absent() {
    let tensors = vec![
        f32s(&format!("text_encoder_2.{TE2_Q}.lora_A.weight"), &[1, 8], &[0.1; 8]),
        f32s(&format!("text_encoder_2.{TE2_Q}.lora_B.weight"), &[8, 1], &[0.2; 8]),
    ];
    let lora = parse(&file(&tensors), 0.75, &targets()).expect("a valid file");
    assert_eq!(lora.modules.len(), 1);
    assert_eq!(lora.modules[0].target.namespace, Namespace::TextEncoder2);
    assert!((lora.modules[0].scale - 0.75).abs() < 1e-7);
}

#[test]
fn the_older_diffusers_linear_layer_names_are_read_as_down_and_up() {
    let tensors = vec![
        f32s(
            &format!("unet.{UNET_TO_Q}.lora_linear_layer.down.weight"),
            &[1, 4],
            &[1.0, 0.0, 0.0, 0.0],
        ),
        f32s(
            &format!("unet.{UNET_TO_Q}.lora_linear_layer.up.weight"),
            &[3, 1],
            &[1.0, 2.0, 3.0],
        ),
    ];
    let lora = parse(&file(&tensors), 1.0, &targets()).expect("a valid file");
    assert_eq!(lora.modules[0].down, vec![1.0, 0.0, 0.0, 0.0]);
    assert_eq!(lora.modules[0].up, vec![1.0, 2.0, 3.0]);
}

#[test]
fn a_key_that_names_no_module_of_sdxl_is_an_error_that_names_the_key() {
    let tensors = kohya_unet("down_blocks.9.no_such_module", 1, 4, 3);
    let err = parse(&file(&tensors), 1.0, &targets()).expect_err("the module is unknown");
    assert!(err.contains("lora_unet_down_blocks_9_no_such_module"), "{err}");
}

#[test]
fn a_key_with_no_namespace_is_an_error_that_names_the_key() {
    let tensors = vec![f32s("something.else.lora_down.weight", &[1, 4], &[0.0; 4])];
    let err = parse(&file(&tensors), 1.0, &targets()).expect_err("no namespace");
    assert!(err.contains("something.else.lora_down.weight"), "{err}");
}

#[test]
fn a_down_matrix_that_does_not_fit_the_module_is_an_error_that_names_the_key() {
    // The module takes 4 inputs. This down matrix takes 5.
    let tensors = kohya_unet(UNET_TO_Q, 2, 5, 3);
    let err = parse(&file(&tensors), 1.0, &targets()).expect_err("the shape does not fit");
    assert!(
        err.contains(&format!("lora_unet_{}", UNET_TO_Q.replace('.', "_"))),
        "{err}"
    );
}

#[test]
fn a_module_with_no_up_matrix_is_an_error() {
    let base = format!("lora_unet_{}", UNET_TO_Q.replace('.', "_"));
    let tensors = vec![f32s(&format!("{base}.lora_down.weight"), &[2, 4], &[0.0; 8])];
    let err = parse(&file(&tensors), 1.0, &targets()).expect_err("no up matrix");
    assert!(err.contains("lora_up") || err.contains("up"), "{err}");
}

#[test]
fn half_and_bfloat_matrices_are_read_as_f32() {
    let base = format!("lora_unet_{}", UNET_TO_Q.replace('.', "_"));
    let down = Tensor {
        name: format!("{base}.lora_down.weight"),
        dtype: "F16",
        shape: vec![1, 4],
        // F16 1.0 is 0x3C00, and F16 0.5 is 0x3800.
        data: [0x3C00u16, 0x3800, 0x0000, 0x0000]
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect(),
    };
    let up = Tensor {
        name: format!("{base}.lora_up.weight"),
        dtype: "BF16",
        shape: vec![3, 1],
        // BF16 2.0 is 0x4000, and BF16 -1.0 is 0xBF80.
        data: [0x4000u16, 0xBF80, 0x0000]
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect(),
    };
    let lora = parse(&file(&[down, up]), 1.0, &targets()).expect("a valid file");
    assert_eq!(lora.modules[0].down, vec![1.0, 0.5, 0.0, 0.0]);
    assert_eq!(lora.modules[0].up, vec![2.0, -1.0, 0.0]);
}

#[test]
fn the_delta_is_scale_times_up_of_down_of_x() {
    // rank 2, 3 inputs, 2 outputs. down is [[1, 0, 0], [0, 1, 0]], so down x = [1, 2] for x = [1, 2, 3].
    // up is [[1, 2], [3, 4]], so up of [1, 2] = [1 + 4, 3 + 8] = [5, 11]. Times the scale 0.5 gives [2.5, 5.5].
    let module = Module {
        target: Target {
            namespace: Namespace::Unet,
            name: "m".to_owned(),
            cin: 3,
            cout: 2,
            kernel: 1,
        },
        rank: 2,
        scale: 0.5,
        down: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        up: vec![1.0, 2.0, 3.0, 4.0],
    };
    assert_eq!(delta(&module, &[1.0, 2.0, 3.0], 1), vec![2.5, 5.5]);
}

#[test]
fn the_delta_gives_one_row_for_each_row_of_x() {
    let module = Module {
        target: Target {
            namespace: Namespace::Unet,
            name: "m".to_owned(),
            cin: 2,
            cout: 1,
            kernel: 1,
        },
        rank: 1,
        scale: 1.0,
        down: vec![1.0, 1.0],
        up: vec![3.0],
    };
    // Row 0 is [1, 0] and row 1 is [0, 2]: down gives 1 and 2, and up gives 3 and 6.
    assert_eq!(delta(&module, &[1.0, 0.0, 0.0, 2.0], 2), vec![3.0, 6.0]);
}
