//! The prepared weights of a plan, from the tensors of a converted checkpoint (`tools/beatthis_golden.sh`).

use safetensors::{Dtype, SafeTensors};

use crate::Error;
use crate::plan::{Prep, Weight};

/// The eps of `nn.BatchNorm1d` and `nn.BatchNorm2d`, which the model leaves at its default.
const BN_EPS: f64 = 1e-5;

/// The tensors of a converted checkpoint, parsed from its bytes.
pub struct Tensors<'a> {
    st: SafeTensors<'a>,
}

impl<'a> Tensors<'a> {
    /// Parses the bytes of a safetensors file.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let st = SafeTensors::deserialize(bytes).map_err(|e| Error::Tensor {
            name: "(the file)".into(),
            why: format!("it is not a safetensors file: {e}"),
        })?;
        Ok(Tensors { st })
    }

    /// The `hyper_parameters` JSON of the checkpoint's metadata.
    pub fn hyper_parameters(bytes: &[u8]) -> Result<String, Error> {
        let (_, meta) = SafeTensors::read_metadata(bytes)
            .map_err(|e| Error::Config(format!("the file is not a safetensors file: {e}")))?;
        meta.metadata()
            .as_ref()
            .and_then(|m| m.get("hyper_parameters"))
            .cloned()
            .ok_or_else(|| Error::Config("the metadata has no hyper_parameters".into()))
    }

    /// A tensor as f32 values, checked against the shape that the plan reads.
    fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, Error> {
        let err = |why: String| Error::Tensor {
            name: name.to_owned(),
            why,
        };
        let t = self
            .st
            .tensor(name)
            .map_err(|_| err("the file has no such tensor".into()))?;
        if t.dtype() != Dtype::F32 {
            return Err(err(format!("it is {:?}, not F32", t.dtype())));
        }
        if t.shape() != shape {
            return Err(err(format!(
                "its shape is {:?}, and the plan reads {shape:?}",
                t.shape()
            )));
        }
        let (values, _) = t.data().as_chunks::<4>();
        Ok(values.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }
}

/// `weight / sqrt(running_var + eps)` and `bias - running_mean * that`, in f64, each rounded once to f32.
fn batch_norm(t: &Tensors, prefix: &str, channels: usize) -> Result<(Vec<f32>, Vec<f32>), Error> {
    let get = |p: &str| t.f32(&format!("{prefix}.{p}"), &[channels]);
    let (w, b, mean, var) = (get("weight")?, get("bias")?, get("running_mean")?, get("running_var")?);
    Ok((0..channels)
        .map(|c| {
            let s = f64::from(w[c]) / (f64::from(var[c]) + BN_EPS).sqrt();
            (s as f32, (f64::from(b[c]) - f64::from(mean[c]) * s) as f32)
        })
        .unzip())
}

/// The values of each weight of a plan, in its order.
pub fn prepare(weights: &[Weight], tensors: &Tensors) -> Result<Vec<Vec<f32>>, Error> {
    weights
        .iter()
        .map(|w| match &w.prep {
            Prep::Plain { name, shape } => tensors.f32(name, shape),
            Prep::BnScale { prefix, channels } => Ok(batch_norm(tensors, prefix, *channels)?.0),
            Prep::BnShift { prefix, channels } => Ok(batch_norm(tensors, prefix, *channels)?.1),
            &Prep::ConvKernel {
                ref name,
                cout,
                cin,
                kf,
                kt,
            } => {
                let k = tensors.f32(name, &[cout, cin, kf, kt])?;
                let mut out = vec![0.0f32; k.len()];
                for co in 0..cout {
                    for ci in 0..cin {
                        for i in 0..kf {
                            for j in 0..kt {
                                out[((co * kf + i) * kt + j) * cin + ci] = k[((co * cin + ci) * kf + i) * kt + j];
                            }
                        }
                    }
                }
                Ok(out)
            }
            &Prep::PermuteColumns {
                ref name,
                rows,
                channels,
                freqs,
            } => {
                let m = tensors.f32(name, &[rows, channels * freqs])?;
                let cols = channels * freqs;
                let mut out = vec![0.0f32; m.len()];
                for r in 0..rows {
                    for c in 0..channels {
                        for f in 0..freqs {
                            out[r * cols + f * channels + c] = m[r * cols + c * freqs + f];
                        }
                    }
                }
                Ok(out)
            }
        })
        .collect()
}
