//! The transformer of Qwen-Image 2.1 on gfx1151, with the LoRA of the profile folded into its weights (spec 11.9c,
//! "The LoRA fold" and "The steps"; milestone 22 step 6).
//!
//! `begin` runs the conditioning rows through the 32 blocks with the modulation of timestep 0, and keeps the keys and
//! values of each block. A condition row never reads the latent, so these are the same at each step. `step` runs
//! the latent tokens of one image over those keys and values, and gives the velocity.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt;
use std::path::Path;

use nagaya_mk_hip::attention::{Attention, AttentionShape, Mask};
use nagaya_mk_hip::bf16;
use nagaya_mk_hip::gemm::{GemmError, Gemms};
use nagaya_mk_hip::hip::{DeviceBuffer, HipError, Stream, copy_2d};
use nagaya_mk_hip::ops::{self, RopePairs};
use nagaya_mk_hip::plan::{DataType, GemmShape};
use nagaya_mk_hip::rope;

use crate::safetensors::{Dtype, Safetensors, SafetensorsError};

pub const BLOCKS: usize = 32;
pub const DIM: usize = 4096;
pub const HEADS: usize = 32;
pub const HEAD_DIM: usize = 128;
pub const MLP: usize = 12288;
/// The channels of a latent token.
pub const LATENT: usize = 64;
const TIME: usize = 256;
const EPS: f32 = 1e-6;
const THETA: f32 = 10000.0;
const AXES: [usize; 3] = [16, 56, 56];
/// The width of a row of the work buffer: q, k and v.
const QKV: usize = 3 * DIM;
/// The width of the keys and values that a block keeps for each condition row.
const KV: usize = 2 * DIM;

/// The modules of a block that a LoRA may change, with their place in the fused weights: the fused weight, the
/// first row, and the shape `[out, in]`.
const LORA_MODULES: [(&str, Fused, usize, usize, usize); 7] = [
    ("attn.to_q", Fused::Qkv, 0, DIM, DIM),
    ("attn.to_k", Fused::Qkv, DIM, DIM, DIM),
    ("attn.to_v", Fused::Qkv, 2 * DIM, DIM, DIM),
    ("attn.to_out.0", Fused::Out, 0, DIM, DIM),
    ("img_mlp.gate_layer", Fused::GateProj, 0, MLP, DIM),
    ("img_mlp.proj", Fused::GateProj, MLP, MLP, DIM),
    ("img_mlp.out", Fused::MlpOut, 0, DIM, MLP),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fused {
    Qkv,
    Out,
    GateProj,
    MlpOut,
}

/// An error of the transformer.
#[derive(Debug)]
pub enum TransformerError {
    File(SafetensorsError),
    Load(String),
    Hip(HipError),
    Gemm(GemmError),
}

impl fmt::Display for TransformerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransformerError::File(e) => write!(f, "{e}"),
            TransformerError::Load(e) => write!(f, "the transformer: {e}"),
            TransformerError::Hip(e) => write!(f, "{e}"),
            TransformerError::Gemm(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TransformerError {}

impl From<SafetensorsError> for TransformerError {
    fn from(e: SafetensorsError) -> Self {
        TransformerError::File(e)
    }
}

impl From<HipError> for TransformerError {
    fn from(e: HipError) -> Self {
        TransformerError::Hip(e)
    }
}

impl From<GemmError> for TransformerError {
    fn from(e: GemmError) -> Self {
        TransformerError::Gemm(e)
    }
}

/// The weights of one block. q, k and v are one matrix, and so are the gate and the proj of the MLP.
struct Block {
    qkv: DeviceBuffer,
    norm_q: DeviceBuffer,
    norm_k: DeviceBuffer,
    out: DeviceBuffer,
    gate_proj: DeviceBuffer,
    mlp_out: DeviceBuffer,
}

impl Block {
    fn fused(&self, which: Fused) -> &DeviceBuffer {
        match which {
            Fused::Qkv => &self.qkv,
            Fused::Out => &self.out,
            Fused::GateProj => &self.gate_proj,
            Fused::MlpOut => &self.mlp_out,
        }
    }
}

/// The files of the transformer folder: each tensor name and the shard that holds it.
struct Shards {
    files: Vec<Safetensors>,
    map: BTreeMap<String, usize>,
}

impl Shards {
    fn open(dir: &Path) -> Result<Shards, TransformerError> {
        let index_path = dir.join("diffusion_pytorch_model.safetensors.index.json");
        let text = std::fs::read_to_string(&index_path)
            .map_err(|e| TransformerError::Load(format!("cannot read {}: {e}", index_path.display())))?;
        let index: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| TransformerError::Load(format!("{} is not JSON: {e}", index_path.display())))?;
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| TransformerError::Load(format!("{} has no weight_map", index_path.display())))?;
        let mut names: Vec<&str> = weight_map.values().filter_map(|v| v.as_str()).collect();
        names.sort();
        names.dedup();
        let mut files = Vec::new();
        for name in &names {
            files.push(Safetensors::open(&dir.join(name))?);
        }
        let mut map = BTreeMap::new();
        for (tensor, file) in weight_map {
            let file = file.as_str().unwrap_or_default();
            let shard = names
                .iter()
                .position(|n| *n == file)
                .ok_or_else(|| TransformerError::Load(format!("the weight_map names no file for `{tensor}`")))?;
            if !files[shard].tensors().contains_key(tensor) {
                return Err(TransformerError::Load(format!(
                    "`{file}` lacks `{tensor}` of the weight_map"
                )));
            }
            map.insert(tensor.clone(), shard);
        }
        Ok(Shards { files, map })
    }

    fn read(&self, name: &str, shape: &[usize]) -> Result<Vec<u8>, TransformerError> {
        let shard = self
            .map
            .get(name)
            .ok_or_else(|| TransformerError::Load(format!("the folder lacks the tensor `{name}`")))?;
        Ok(self.files[*shard].read(name, Dtype::Bf16, shape)?)
    }
}

/// The transformer in device memory.
pub struct Transformer {
    /// True when attention runs in the fused flash kernel, false for the GEMMs and the softmax kernel.
    pub flash: bool,
    img_in: DeviceBuffer,
    txt_norm: DeviceBuffer,
    txt_in: DeviceBuffer,
    txt_out: DeviceBuffer,
    time_1: DeviceBuffer,
    time_2: DeviceBuffer,
    modulation: DeviceBuffer,
    norm_out: DeviceBuffer,
    proj_out: DeviceBuffer,
    blocks: Vec<Block>,
}

/// The keys and values of the conditioning rows in each block, and their count.
pub struct Conditioned {
    rows: usize,
    kv: Vec<DeviceBuffer>,
}

impl Conditioned {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn bytes(&self) -> usize {
        self.kv.iter().map(DeviceBuffer::bytes).sum()
    }
}

fn at(buffer: &DeviceBuffer, elements: usize) -> *mut c_void {
    buffer.ptr().cast::<u16>().wrapping_add(elements).cast()
}

fn linear(rows: usize, inputs: usize, outputs: usize) -> GemmShape {
    GemmShape::linear(rows as u32, inputs as u32, outputs as u32, DataType::Bf16)
}

/// The shift, scale and gate vectors of one timestep: the four parts of the shared modulation, and the scale of the
/// final norm.
struct Modulation {
    all: DeviceBuffer,
    final_scale: DeviceBuffer,
}

impl Modulation {
    fn scale1(&self) -> *mut c_void {
        self.all.ptr()
    }
    fn gate1(&self) -> *mut c_void {
        at(&self.all, DIM)
    }
    fn scale2(&self) -> *mut c_void {
        at(&self.all, 2 * DIM)
    }
    fn gate2(&self) -> *mut c_void {
        at(&self.all, 3 * DIM)
    }
}

/// The work buffers of a pass of `rows` rows.
struct Work {
    hidden: DeviceBuffer,
    normed: DeviceBuffer,
    mixed: DeviceBuffer,
    y: DeviceBuffer,
    gate_proj: DeviceBuffer,
    act: DeviceBuffer,
}

impl Work {
    fn new(rows: usize) -> Result<Work, HipError> {
        Ok(Work {
            hidden: DeviceBuffer::new(rows * DIM * 2)?,
            normed: DeviceBuffer::new(rows * DIM * 2)?,
            mixed: DeviceBuffer::new(rows * DIM * 2)?,
            y: DeviceBuffer::new(rows * DIM * 2)?,
            gate_proj: DeviceBuffer::new(rows * 2 * MLP * 2)?,
            act: DeviceBuffer::new(rows * MLP * 2)?,
        })
    }
}

impl Transformer {
    /// Load the transformer folder, and fold the LoRA `lora` at `strength` into its weights (spec 11.9c). The fold
    /// runs on the device: `W + strength * scale * B A`, summed in f32 and rounded once to bf16.
    pub fn load(
        dir: &Path,
        lora: Option<(&Safetensors, f32)>,
        gemms: &mut Gemms,
        stream: &Stream,
    ) -> Result<Transformer, TransformerError> {
        let shards = Shards::open(dir)?;
        let mut wanted: Vec<String> = [
            "img_in.weight",
            "txt_in.text_norm.weight",
            "txt_in.in_layer.weight",
            "txt_in.out_layer.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "time_text_embed.timestep_embedder.linear_2.weight",
            "modulation.1.weight",
            "norm_out.linear.weight",
            "proj_out.weight",
        ]
        .map(String::from)
        .to_vec();
        for b in 0..BLOCKS {
            for part in [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",
                "attn.norm_q",
                "attn.norm_k",
                "img_mlp.gate_layer",
                "img_mlp.proj",
                "img_mlp.out",
            ] {
                wanted.push(format!("transformer_blocks.{b}.{part}.weight"));
            }
        }
        for name in shards.map.keys() {
            if !wanted.contains(name) {
                return Err(TransformerError::Load(format!(
                    "the tensor `{name}` is in no list of the load"
                )));
            }
        }
        let upload = |name: &str, shape: &[usize]| -> Result<DeviceBuffer, TransformerError> {
            Ok(DeviceBuffer::from_slice(&shards.read(name, shape)?)?)
        };
        let stack = |parts: &[(String, &[usize])]| -> Result<DeviceBuffer, TransformerError> {
            let mut bytes = Vec::new();
            for (name, shape) in parts {
                bytes.extend_from_slice(&shards.read(name, shape)?);
            }
            Ok(DeviceBuffer::from_slice(&bytes)?)
        };
        let mut blocks = Vec::with_capacity(BLOCKS);
        for b in 0..BLOCKS {
            let n = |part: &str| format!("transformer_blocks.{b}.{part}.weight");
            blocks.push(Block {
                qkv: stack(&[
                    (n("attn.to_q"), &[DIM, DIM]),
                    (n("attn.to_k"), &[DIM, DIM]),
                    (n("attn.to_v"), &[DIM, DIM]),
                ])?,
                norm_q: upload(&n("attn.norm_q"), &[HEAD_DIM])?,
                norm_k: upload(&n("attn.norm_k"), &[HEAD_DIM])?,
                out: upload(&n("attn.to_out.0"), &[DIM, DIM])?,
                gate_proj: stack(&[(n("img_mlp.gate_layer"), &[MLP, DIM]), (n("img_mlp.proj"), &[MLP, DIM])])?,
                mlp_out: upload(&n("img_mlp.out"), &[DIM, MLP])?,
            });
        }
        let transformer = Transformer {
            flash: false,
            img_in: upload("img_in.weight", &[DIM, LATENT])?,
            txt_norm: upload("txt_in.text_norm.weight", &[DIM])?,
            txt_in: upload("txt_in.in_layer.weight", &[DIM, DIM])?,
            txt_out: upload("txt_in.out_layer.weight", &[DIM, DIM])?,
            time_1: upload("time_text_embed.timestep_embedder.linear_1.weight", &[DIM, TIME])?,
            time_2: upload("time_text_embed.timestep_embedder.linear_2.weight", &[DIM, DIM])?,
            modulation: upload("modulation.1.weight", &[4 * DIM, DIM])?,
            norm_out: upload("norm_out.linear.weight", &[DIM, DIM])?,
            proj_out: upload("proj_out.weight", &[LATENT, DIM])?,
            blocks,
        };
        if let Some((file, strength)) = lora {
            transformer.fold(file, strength, gemms, stream)?;
        }
        Ok(transformer)
    }

    /// Fold a LoRA into the weights. A key that names no module of the transformer, or a shape that does not match,
    /// refuses the load (spec 11.9c).
    fn fold(
        &self,
        file: &Safetensors,
        strength: f32,
        gemms: &mut Gemms,
        stream: &Stream,
    ) -> Result<(), TransformerError> {
        let mut seen = 0;
        let rank_of = |name: &str| -> Result<usize, TransformerError> {
            let info = file.info(name)?;
            info.shape
                .first()
                .copied()
                .ok_or_else(|| TransformerError::Load(format!("the LoRA tensor `{name}` has no shape")))
        };
        for (b, block) in self.blocks.iter().enumerate() {
            for (module, fused, row, out, inp) in LORA_MODULES {
                let base = format!("transformer.transformer_blocks.{b}.{module}");
                let (a_name, b_name) = (format!("{base}.lora_A.weight"), format!("{base}.lora_B.weight"));
                if !file.tensors().contains_key(&a_name) && !file.tensors().contains_key(&b_name) {
                    continue;
                }
                let rank = rank_of(&a_name)?;
                let a = DeviceBuffer::from_slice(&file.read(&a_name, Dtype::Bf16, &[rank, inp])?)?;
                let bm = DeviceBuffer::from_slice(&file.read(&b_name, Dtype::Bf16, &[out, rank])?)?;
                let alpha_name = format!("{base}.alpha");
                let scale = if file.tensors().contains_key(&alpha_name) {
                    let bytes = file.read(&alpha_name, Dtype::F32, &[])?;
                    seen += 1;
                    f32::from_le_bytes(bytes[..4].try_into().expect("an f32")) / rank as f32
                } else {
                    1.0
                };
                seen += 2;
                // Column-major: W^T (in by out) += A^T (in by rank) B^T (rank by out), with the rows of W in order.
                let shape = GemmShape {
                    m: inp as u32,
                    n: out as u32,
                    k: rank as u32,
                    trans_a: false,
                    trans_b: false,
                    lda: inp as u64,
                    ldb: rank as u64,
                    ldc: inp as u64,
                    batch: 1,
                    stride_a: 0,
                    stride_b: 0,
                    stride_c: 0,
                    type_ab: DataType::Bf16,
                    type_cd: DataType::Bf16,
                    bias: false,
                };
                let w = at(block.fused(fused), row * inp);
                // SAFETY: the LoRA matrices hold their shapes, and `w` is the [out, in] rows of the module.
                unsafe { gemms.run(&shape, strength * scale, a.ptr(), bm.ptr(), 1.0, w, w, stream)? };
                stream.synchronize()?;
            }
        }
        if seen != file.tensors().len() {
            let known: Vec<String> = (0..BLOCKS)
                .flat_map(|b| {
                    LORA_MODULES.iter().flat_map(move |(m, ..)| {
                        let base = format!("transformer.transformer_blocks.{b}.{m}");
                        [
                            format!("{base}.lora_A.weight"),
                            format!("{base}.lora_B.weight"),
                            format!("{base}.alpha"),
                        ]
                    })
                })
                .collect();
            let unknown = file
                .tensors()
                .keys()
                .find(|k| !known.contains(k))
                .cloned()
                .unwrap_or_default();
            return Err(TransformerError::Load(format!(
                "the LoRA {} has the tensor `{unknown}`, which names no module of the transformer",
                file.path().display()
            )));
        }
        Ok(())
    }

    /// The bytes of the weights in device memory.
    pub fn bytes(&self) -> usize {
        [
            &self.img_in,
            &self.txt_norm,
            &self.txt_in,
            &self.txt_out,
            &self.time_1,
            &self.time_2,
            &self.modulation,
            &self.norm_out,
            &self.proj_out,
        ]
        .iter()
        .map(|b| b.bytes())
        .sum::<usize>()
            + self
                .blocks
                .iter()
                .map(|b| {
                    b.qkv.bytes()
                        + b.norm_q.bytes()
                        + b.norm_k.bytes()
                        + b.out.bytes()
                        + b.gate_proj.bytes()
                        + b.mlp_out.bytes()
                })
                .sum::<usize>()
    }

    /// The modulation of the timestep `t` as the transformer reads it (`sampler::timestep`).
    fn modulation(&self, gemms: &mut Gemms, t: f32, stream: &Stream) -> Result<Modulation, TransformerError> {
        let sinusoid: Vec<u16> = rope::timestep_sinusoid(t, TIME)
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        let input = DeviceBuffer::from_slice(&sinusoid)?;
        let temb = DeviceBuffer::new(DIM * 2)?;
        let hidden = DeviceBuffer::new(DIM * 2)?;
        let all = DeviceBuffer::new(4 * DIM * 2)?;
        let final_scale = DeviceBuffer::new(DIM * 2)?;
        // SAFETY: each buffer holds one row of its width.
        unsafe {
            gemms.run(
                &linear(1, TIME, DIM),
                1.0,
                self.time_1.ptr(),
                input.ptr(),
                0.0,
                hidden.ptr(),
                hidden.ptr(),
                stream,
            )?;
            ops::silu(hidden.ptr(), DIM, stream)?;
            gemms.run(
                &linear(1, DIM, DIM),
                1.0,
                self.time_2.ptr(),
                hidden.ptr(),
                0.0,
                temb.ptr(),
                temb.ptr(),
                stream,
            )?;
            // Both the shared modulation and the final norm read silu(temb).
            ops::silu(temb.ptr(), DIM, stream)?;
            gemms.run(
                &linear(1, DIM, 4 * DIM),
                1.0,
                self.modulation.ptr(),
                temb.ptr(),
                0.0,
                all.ptr(),
                all.ptr(),
                stream,
            )?;
            gemms.run(
                &linear(1, DIM, DIM),
                1.0,
                self.norm_out.ptr(),
                temb.ptr(),
                0.0,
                final_scale.ptr(),
                final_scale.ptr(),
                stream,
            )?;
        }
        Ok(Modulation { all, final_scale })
    }

    /// One block over `rows` rows of `work.hidden`, whose q, k and v go to `qkv` from row `first`. The attention runs
    /// over `keys` key rows of `qkv` from row 0, with `mask`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn block(
        &self,
        block: &Block,
        gemms: &mut Gemms,
        attention: &Attention,
        modulation: &Modulation,
        work: &Work,
        qkv: &DeviceBuffer,
        first: usize,
        rows: usize,
        keys: usize,
        mask: Mask,
        table: &DeviceBuffer,
        stream: &Stream,
    ) -> Result<(), TransformerError> {
        let mine = at(qkv, first * QKV);
        // SAFETY: the caller gives buffers of `rows` rows, and `qkv` of `keys` rows.
        unsafe {
            ops::layernorm_modulate(
                work.hidden.ptr(),
                DIM,
                modulation.scale1(),
                work.normed.ptr(),
                DIM,
                rows,
                DIM,
                EPS,
                stream,
            )?;
            gemms.run(
                &linear(rows, DIM, QKV),
                1.0,
                block.qkv.ptr(),
                work.normed.ptr(),
                0.0,
                mine,
                mine,
                stream,
            )?;
            let rope = |x: *mut c_void, w: &DeviceBuffer| {
                ops::head_norm_rope(
                    x,
                    QKV,
                    rows,
                    HEADS,
                    HEAD_DIM,
                    w.ptr(),
                    EPS,
                    table.ptr(),
                    RopePairs::Adjacent,
                    stream,
                )
            };
            rope(mine, &block.norm_q)?;
            rope(mine.cast::<u16>().wrapping_add(DIM).cast(), &block.norm_k)?;
            let shape = AttentionShape {
                lq: rows,
                lk: keys,
                heads: HEADS,
                kv_heads: HEADS,
                head_dim: HEAD_DIM,
                q_stride: QKV,
                k_stride: QKV,
                v_stride: QKV,
                out_stride: DIM,
                mask,
            };
            if self.flash {
                nagaya_mk_hip::attention::flash(
                    &shape,
                    mine,
                    at(qkv, DIM),
                    at(qkv, 2 * DIM),
                    work.mixed.ptr(),
                    stream,
                )?;
            } else {
                attention.run(
                    gemms,
                    &shape,
                    mine,
                    at(qkv, DIM),
                    at(qkv, 2 * DIM),
                    work.mixed.ptr(),
                    stream,
                )?;
            }
            gemms.run(
                &linear(rows, DIM, DIM),
                1.0,
                block.out.ptr(),
                work.mixed.ptr(),
                0.0,
                work.y.ptr(),
                work.y.ptr(),
                stream,
            )?;
            ops::gate_add(
                work.hidden.ptr(),
                DIM,
                work.y.ptr(),
                DIM,
                modulation.gate1(),
                rows,
                DIM,
                stream,
            )?;
            ops::layernorm_modulate(
                work.hidden.ptr(),
                DIM,
                modulation.scale2(),
                work.normed.ptr(),
                DIM,
                rows,
                DIM,
                EPS,
                stream,
            )?;
            gemms.run(
                &linear(rows, DIM, 2 * MLP),
                1.0,
                block.gate_proj.ptr(),
                work.normed.ptr(),
                0.0,
                work.gate_proj.ptr(),
                work.gate_proj.ptr(),
                stream,
            )?;
            ops::silu_mul(work.gate_proj.ptr(), 2 * MLP, MLP, work.act.ptr(), MLP, rows, stream)?;
            gemms.run(
                &linear(rows, MLP, DIM),
                1.0,
                block.mlp_out.ptr(),
                work.act.ptr(),
                0.0,
                work.y.ptr(),
                work.y.ptr(),
                stream,
            )?;
            ops::gate_add(
                work.hidden.ptr(),
                DIM,
                work.y.ptr(),
                DIM,
                modulation.gate2(),
                rows,
                DIM,
                stream,
            )?;
        }
        Ok(())
    }

    /// The scores that the attention of a step needs: `latent` queries over `rows + latent` keys, one head at a time
    /// at least.
    pub fn attention_capacity(rows: usize, latent: usize) -> usize {
        latent * (rows + latent)
    }

    /// Run the conditioning rows `first..first + rows` of `conditioning` (`[_, 4096]` in bf16: the encoder output, cut
    /// at `first`) through the blocks with the modulation of timestep 0, and keep each block's keys and values (spec
    /// 11.9c, "The steps").
    pub fn begin(
        &self,
        gemms: &mut Gemms,
        attention: &Attention,
        conditioning: &DeviceBuffer,
        first: usize,
        rows: usize,
        stream: &Stream,
    ) -> Result<Conditioned, TransformerError> {
        assert!(rows > 0, "a conditioning has at least one row");
        assert!(
            conditioning.bytes() >= (first + rows) * DIM * 2,
            "the conditioning holds {} rows",
            first + rows
        );
        let conditioning = at(conditioning, first * DIM).cast_const();
        let work = Work::new(rows)?;
        let qkv = DeviceBuffer::new(rows * QKV * 2)?;
        let modulation = self.modulation(gemms, 0.0, stream)?;
        let positions = rope::step_positions(rows, 0, 0);
        let table = DeviceBuffer::from_slice(&rope::axes_table(&positions, AXES, THETA))?;
        let mut kv = Vec::with_capacity(BLOCKS);
        // SAFETY: the conditioning holds `rows` rows; the buffers above hold the rows of the pass.
        unsafe {
            // The text input: the zero-centred RMSNorm, a linear layer, GELU and a linear layer.
            ops::rmsnorm(
                conditioning,
                DIM,
                self.txt_norm.ptr(),
                1.0,
                work.normed.ptr(),
                DIM,
                rows,
                DIM,
                EPS,
                stream,
            )?;
            gemms.run(
                &linear(rows, DIM, DIM),
                1.0,
                self.txt_in.ptr(),
                work.normed.ptr(),
                0.0,
                work.y.ptr(),
                work.y.ptr(),
                stream,
            )?;
            ops::gelu_tanh(work.y.ptr(), DIM, rows, DIM, stream)?;
            gemms.run(
                &linear(rows, DIM, DIM),
                1.0,
                self.txt_out.ptr(),
                work.y.ptr(),
                0.0,
                work.hidden.ptr(),
                work.hidden.ptr(),
                stream,
            )?;
            for block in &self.blocks {
                self.block(
                    block,
                    gemms,
                    attention,
                    &modulation,
                    &work,
                    &qkv,
                    0,
                    rows,
                    rows,
                    Mask::Causal { offset: 0 },
                    &table,
                    stream,
                )?;
                let keep = DeviceBuffer::new(rows * KV * 2)?;
                copy_2d(keep.ptr(), KV * 2, at(&qkv, DIM), QKV * 2, KV * 2, rows, stream)?;
                kv.push(keep);
            }
        }
        stream.synchronize()?;
        Ok(Conditioned { rows, kv })
    }

    /// One step of one image: the velocity `[h * w, 64]` of the latent `[h * w, 64]` in bf16 at the sigma whose
    /// timestep is `t` (`sampler::timestep`).
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        gemms: &mut Gemms,
        attention: &Attention,
        conditioned: &Conditioned,
        latent: &DeviceBuffer,
        h: usize,
        w: usize,
        t: f32,
        stream: &Stream,
    ) -> Result<DeviceBuffer, TransformerError> {
        let n = h * w;
        assert!(latent.bytes() >= n * LATENT * 2, "the latent holds {n} tokens");
        let latent = latent.ptr().cast_const();
        let rows = conditioned.rows;
        let work = Work::new(n)?;
        let qkv = DeviceBuffer::new((rows + n) * QKV * 2)?;
        let modulation = self.modulation(gemms, t, stream)?;
        let positions = rope::step_positions(rows, h, w);
        let table = DeviceBuffer::from_slice(&rope::axes_table(&positions[rows..], AXES, THETA))?;
        let velocity = DeviceBuffer::new(n * LATENT * 2)?;
        // SAFETY: the latent holds `n` tokens; the buffers above hold the rows of the pass.
        unsafe {
            gemms.run(
                &linear(n, LATENT, DIM),
                1.0,
                self.img_in.ptr(),
                latent,
                0.0,
                work.hidden.ptr(),
                work.hidden.ptr(),
                stream,
            )?;
            for (block, kv) in self.blocks.iter().zip(&conditioned.kv) {
                copy_2d(at(&qkv, DIM), QKV * 2, kv.ptr(), KV * 2, KV * 2, rows, stream)?;
                self.block(
                    block,
                    gemms,
                    attention,
                    &modulation,
                    &work,
                    &qkv,
                    rows,
                    n,
                    rows + n,
                    Mask::None,
                    &table,
                    stream,
                )?;
            }
            ops::layernorm_modulate(
                work.hidden.ptr(),
                DIM,
                modulation.final_scale.ptr(),
                work.normed.ptr(),
                DIM,
                n,
                DIM,
                EPS,
                stream,
            )?;
            gemms.run(
                &linear(n, DIM, LATENT),
                1.0,
                self.proj_out.ptr(),
                work.normed.ptr(),
                0.0,
                velocity.ptr(),
                velocity.ptr(),
                stream,
            )?;
        }
        stream.synchronize()?;
        Ok(velocity)
    }
}
