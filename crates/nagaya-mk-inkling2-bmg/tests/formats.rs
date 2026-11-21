//! The weight formats against the reference, and by hand.

use nagaya_mk_inkling2_bmg::formats::*;
use serde_json::Value;

fn hex_bytes(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn hex_u16(v: &Value) -> Vec<u16> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| u16::from_str_radix(s.as_str().unwrap(), 16).unwrap())
        .collect()
}

fn hex_f32(s: &str) -> f32 {
    f32::from_bits(u32::from_str_radix(s, 16).unwrap())
}

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/formats.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// The width of a row of the output, from the format and the kind of the matrix.
fn columns(case: &Value) -> usize {
    match case["kind"].as_str().unwrap() {
        "w13" => 2304,
        "w2" => 1152,
        _ => 32,
    }
}

/// Expand every row of a case with the host op. The rows are in the order of the checkpoint.
fn expand_case(case: &Value, factor: f32) -> Vec<u16> {
    let weight = hex_bytes(case["weight"].as_str().unwrap());
    let scale = hex_bytes(case["scale"].as_str().unwrap());
    let cols = columns(case);
    let nvfp4 = case["format"] == "nvfp4";
    let (wbytes, sbytes) = if nvfp4 {
        (cols / 2, cols / NVFP4_BLOCK)
    } else {
        (cols, cols / MXFP8_BLOCK)
    };
    let rows = weight.len() / wbytes;
    let mut out = vec![0u16; rows * cols];
    for r in 0..rows {
        let w = &weight[r * wbytes..(r + 1) * wbytes];
        let s = &scale[r * sbytes..(r + 1) * sbytes];
        let o = &mut out[r * cols..(r + 1) * cols];
        if nvfp4 {
            expand_nvfp4_row(w, s, factor, o).unwrap();
        } else {
            expand_mxfp8_row(w, s, o).unwrap();
        }
    }
    out
}

fn factor_of(case: &Value, loader_cast: bool) -> f32 {
    if case["format"] != "nvfp4" {
        return 1.0;
    }
    if loader_cast {
        let bf16 =
            |k: &str| f32::from_bits(u32::from(u16::from_str_radix(case[k].as_str().unwrap(), 16).unwrap()) << 16);
        expert_factor(bf16("amax_bf16_bits"), bf16("denominator_bf16_bits"))
    } else {
        expert_factor(
            hex_f32(case["amax_bits"].as_str().unwrap()),
            hex_f32(case["denominator_bits"].as_str().unwrap()),
        )
    }
}

#[test]
fn the_expansion_equals_the_reference_bits() {
    let doc = fixture();
    let cases = doc["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 20, "the fixture holds 8 real and 12 synthetic cases");
    for case in cases {
        let got = expand_case(case, factor_of(case, false));
        let want = hex_u16(&case["expected"]);
        assert_eq!(got.len(), want.len());
        let bad = got.iter().zip(&want).position(|(g, w)| g != w);
        assert!(
            bad.is_none(),
            "{} {} {}: first bad index {:?}",
            case["format"],
            case["kind"],
            case["source"],
            bad
        );
    }
}

/// The loader casts `amax` and its denominator to `bf16` first. This result is the one of a model loaded in `bf16`.
#[test]
fn the_expansion_equals_the_reference_bits_when_the_loader_casts_the_factor() {
    let doc = fixture();
    let mut real = 0;
    for case in doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["format"] == "nvfp4" && c["kind"] != "synthetic")
    {
        real += 1;
        assert_eq!(
            expand_case(case, factor_of(case, true)),
            hex_u16(&case["expected_loader_cast"])
        );
    }
    assert_eq!(real, 4);
}

#[test]
fn the_rows_of_w13_deinterleave_as_in_the_reference() {
    let doc = fixture();
    let mut checked = 0;
    for case in doc["cases"].as_array().unwrap().iter().filter(|c| c["kind"] == "w13") {
        let flat = expand_case(case, factor_of(case, false));
        let cols = columns(case);
        let rows = flat.len() / cols;
        assert_eq!(rows, 4);
        let want = hex_u16(&case["expected_interleaved"]);
        let mut got = Vec::new();
        for row in 0..rows {
            let src = gate_up_source_row(row, rows);
            got.extend_from_slice(&flat[src * cols..(src + 1) * cols]);
        }
        assert_eq!(got, want);
        checked += 1;
    }
    assert_eq!(checked, 4);
}

#[test]
fn the_source_row_of_a_gate_up_row() {
    // Rows 0 to 5 of a matrix of 6 rows: the gates come from the even rows, the ups from the odd rows.
    let rows: Vec<usize> = (0..6).map(|r| gate_up_source_row(r, 6)).collect();
    assert_eq!(rows, [0, 2, 4, 1, 3, 5]);
}

#[test]
#[should_panic(expected = "the count must be even")]
fn an_odd_count_of_gate_up_rows_panics() {
    gate_up_source_row(0, 5);
}

#[test]
fn each_e2m1_code_has_its_value() {
    let want = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for (code, w) in want.iter().enumerate() {
        assert_eq!(E2M1[code], *w);
        assert_eq!(E2M1[code + 8], -*w);
        assert!(E2M1[code + 8].is_sign_negative());
    }
    // A row of the codes 0 to 15 with scale 1.0 (E4M3 0x38) and factor 1 gives the values in bf16.
    let codes: Vec<u8> = (0..8).map(|i| (2 * i) | ((2 * i + 1) << 4)).collect();
    let mut out = [0u16; 16];
    expand_nvfp4_row(&codes, &[0x38], 1.0, &mut out).unwrap();
    let bits: Vec<u16> = E2M1.iter().map(|v| f32_to_bf16(*v)).collect();
    assert_eq!(out.to_vec(), bits);
    assert_eq!(out[8], 0x8000, "code 8 is negative zero");
    assert_eq!(out[7], 0x40C0, "code 7 is 6.0");
}

#[test]
fn the_even_element_is_the_low_nibble() {
    let mut out = [0u16; 16];
    // Byte 0x71: low nibble 1 (0.5) is element 0, high nibble 7 (6.0) is element 1.
    expand_nvfp4_row(&[0x71, 0, 0, 0, 0, 0, 0, 0], &[0x38], 1.0, &mut out).unwrap();
    assert_eq!((out[0], out[1]), (f32_to_bf16(0.5), f32_to_bf16(6.0)));
}

#[test]
fn e4m3_subnormals_normals_and_the_maximum() {
    assert_eq!(decode_e4m3(0x00).unwrap(), 0.0);
    assert_eq!(decode_e4m3(0x01).unwrap(), 2f32.powi(-9), "the smallest subnormal");
    assert_eq!(decode_e4m3(0x07).unwrap(), 7.0 * 2f32.powi(-9), "the largest subnormal");
    assert_eq!(decode_e4m3(0x08).unwrap(), 2f32.powi(-6), "the smallest normal");
    assert_eq!(decode_e4m3(0x38).unwrap(), 1.0);
    assert_eq!(decode_e4m3(0x7E).unwrap(), 448.0, "the maximum");
    assert_eq!(decode_e4m3(0xB8).unwrap(), -1.0);
    assert!(decode_e4m3(0x80).unwrap().is_sign_negative(), "negative zero");
    assert_eq!(decode_e4m3(0x7F), Err(FormatError::E4m3Nan(0x7F)));
    assert_eq!(decode_e4m3(0xFF), Err(FormatError::E4m3Nan(0xFF)));
}

#[test]
fn ue8m0_edges() {
    assert_eq!(decode_ue8m0(127).unwrap(), 1.0);
    assert_eq!(decode_ue8m0(128).unwrap(), 2.0);
    assert_eq!(decode_ue8m0(1).unwrap(), 2f32.powi(-126), "the smallest normal");
    assert_eq!(
        decode_ue8m0(0).unwrap().to_bits(),
        1 << 22,
        "2^-127 is subnormal in f32"
    );
    assert_eq!(decode_ue8m0(254).unwrap(), 2f32.powi(127));
    assert_eq!(decode_ue8m0(255), Err(FormatError::Ue8m0Nan));
}

#[test]
fn a_nan_code_is_an_error_in_a_row() {
    let mut out = [0u16; 16];
    let e = expand_nvfp4_row(&[0; 8], &[0x7F], 1.0, &mut out).unwrap_err();
    assert_eq!(e, FormatError::E4m3Nan(0x7F));
    let mut out = [0u16; 32];
    assert_eq!(expand_mxfp8_row(&[0; 32], &[255], &mut out), Err(FormatError::Ue8m0Nan));
    let mut values = [0x38u8; 32];
    values[31] = 0xFF;
    assert_eq!(
        expand_mxfp8_row(&values, &[127], &mut out),
        Err(FormatError::E4m3Nan(0xFF))
    );
    assert_eq!(
        expand_nvfp4_row(&[0; 8], &[0x38], f32::NAN, &mut [0u16; 16]),
        Err(FormatError::ResultNan)
    );
    assert!(e.to_string().contains("Download it again"));
}

#[test]
fn a_wrong_length_is_an_error() {
    let e = expand_nvfp4_row(&[0; 8], &[0x38, 0x38], 1.0, &mut [0u16; 16]).unwrap_err();
    assert!(matches!(e, FormatError::Length { got: 2, want: 1, .. }), "{e:?}");
    assert!(expand_mxfp8_row(&[0; 32], &[127], &mut [0u16; 31]).is_err());
}

#[test]
fn bf16_rounds_to_nearest_even() {
    assert_eq!(f32_to_bf16(1.0), 0x3F80);
    // 1 + 2^-8 is halfway between 1.0 (even) and 1 + 2^-7: it rounds down to even.
    assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
    // The next halfway case, with an odd lower neighbour, rounds up.
    assert_eq!(f32_to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
    // Above the halfway case rounds up.
    assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8001)), 0x3F81);
    // A value above the largest bf16 rounds to infinity.
    assert_eq!(f32_to_bf16(f32::MAX), 0x7F80);
}

#[test]
fn the_fixture_holds_the_synthetic_edge_cases() {
    // The synthetic cases reach the scales 0 and 254 of MXFP8, every finite E4M3 code and E4M3 scales of both signs.
    let doc = fixture();
    let n = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "synthetic")
        .count();
    assert_eq!(n, 12);
}

#[test]
fn the_fixtures_stay_small() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let total: u64 = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();
    assert!(total < 1_000_000, "the fixtures take {total} bytes, which is over 1 MB");
}
