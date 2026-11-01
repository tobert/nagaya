//! The plan table of a HIP engine (spec 12.1, "The plan table of a HIP engine"): the hipBLASLt solution index of each
//! GEMM shape, for one lineage.
//!
//! The table only grows, and a shape keeps its pick. The file is in the store directory, one for each lineage. A pick
//! goes to the file before a job uses it, so a restart keeps each pick. A line that the table cannot read is an
//! error, not a skipped line (spec 8.5).

use std::collections::BTreeMap;
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The first line of a plan table file.
const MAGIC: &str = "nagaya plan table v1";

/// The data type of a matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataType {
    Bf16,
    F32,
}

impl DataType {
    /// The code of the hipBLASLt shim: 0 for bf16, 1 for f32.
    pub fn code(self) -> i32 {
        match self {
            DataType::Bf16 => 0,
            DataType::F32 => 1,
        }
    }

    /// The bytes of one value.
    pub fn bytes(self) -> usize {
        match self {
            DataType::Bf16 => 2,
            DataType::F32 => 4,
        }
    }

    fn name(self) -> &'static str {
        match self {
            DataType::Bf16 => "bf16",
            DataType::F32 => "f32",
        }
    }

    fn parse(text: &str) -> Option<DataType> {
        match text {
            "bf16" => Some(DataType::Bf16),
            "f32" => Some(DataType::F32),
            _ => None,
        }
    }
}

/// The shape of one GEMM, `D = alpha op(A) op(B) + beta C`, in the column-major order of hipBLASLt. `op(A)` is `m`
/// by `k`, `op(B)` is `k` by `n`, and `C` and `D` are `m` by `n`. A batch has `batch` matrices with the given strides.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GemmShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub trans_a: bool,
    pub trans_b: bool,
    pub lda: u64,
    pub ldb: u64,
    pub ldc: u64,
    pub batch: u32,
    pub stride_a: u64,
    pub stride_b: u64,
    pub stride_c: u64,
    pub type_ab: DataType,
    pub type_cd: DataType,
    /// True when the epilogue adds a bf16 bias of `m` values to each column of D.
    pub bias: bool,
}

impl GemmShape {
    /// The shape of a linear layer on row-major tensors: `y[rows, outputs] = x[rows, inputs] W[outputs, inputs]^T`.
    ///
    /// In column-major order this is `y^T = W x^T`: `A` is the weight, read transposed, and `B` is the input.
    pub fn linear(rows: u32, inputs: u32, outputs: u32, type_cd: DataType) -> GemmShape {
        GemmShape {
            m: outputs,
            n: rows,
            k: inputs,
            trans_a: true,
            trans_b: false,
            lda: u64::from(inputs),
            ldb: u64::from(inputs),
            ldc: u64::from(outputs),
            batch: 1,
            stride_a: 0,
            stride_b: 0,
            stride_c: 0,
            type_ab: DataType::Bf16,
            type_cd,
            bias: false,
        }
    }

    /// The same shape with a bias in its epilogue.
    pub fn with_bias(mut self) -> GemmShape {
        self.bias = true;
        self
    }

    /// The elements of A, B and C (also D) that one call reads or writes, over the whole batch.
    pub fn elements(&self) -> (usize, usize, usize) {
        let (a_cols, b_cols) = (
            if self.trans_a { self.m } else { self.k },
            if self.trans_b { self.k } else { self.n },
        );
        let span = |ld: u64, cols: u32, stride: u64| {
            let one = ld as usize * cols as usize;
            one + stride as usize * (self.batch as usize - 1)
        };
        (
            span(self.lda, a_cols, self.stride_a),
            span(self.ldb, b_cols, self.stride_b),
            span(self.ldc, self.n, self.stride_c),
        )
    }

    /// The index of each element of D that one call writes, over the whole batch.
    pub fn output_elements(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.batch as usize).flat_map(move |b| {
            (0..self.n as usize).flat_map(move |j| {
                (0..self.m as usize).map(move |i| b * self.stride_c as usize + j * self.ldc as usize + i)
            })
        })
    }

    fn line(&self, index: i32) -> String {
        format!(
            "m={} n={} k={} ta={} tb={} lda={} ldb={} ldc={} batch={} sa={} sb={} sc={} ab={} cd={} bias={} index={}",
            self.m,
            self.n,
            self.k,
            u8::from(self.trans_a),
            u8::from(self.trans_b),
            self.lda,
            self.ldb,
            self.ldc,
            self.batch,
            self.stride_a,
            self.stride_b,
            self.stride_c,
            self.type_ab.name(),
            self.type_cd.name(),
            u8::from(self.bias),
            index
        )
    }

    fn parse(line: &str) -> Option<(GemmShape, i32)> {
        let mut fields = BTreeMap::new();
        for field in line.split(' ') {
            let (key, value) = field.split_once('=')?;
            if fields.insert(key, value).is_some() {
                return None;
            }
        }
        if fields.len() != 16 {
            return None;
        }
        let int = |key: &str| fields.get(key)?.parse::<u64>().ok();
        let flag = |key: &str| match *fields.get(key)? {
            "0" => Some(false),
            "1" => Some(true),
            _ => None,
        };
        let shape = GemmShape {
            m: u32::try_from(int("m")?).ok()?,
            n: u32::try_from(int("n")?).ok()?,
            k: u32::try_from(int("k")?).ok()?,
            trans_a: flag("ta")?,
            trans_b: flag("tb")?,
            lda: int("lda")?,
            ldb: int("ldb")?,
            ldc: int("ldc")?,
            batch: u32::try_from(int("batch")?).ok()?,
            stride_a: int("sa")?,
            stride_b: int("sb")?,
            stride_c: int("sc")?,
            type_ab: DataType::parse(fields.get("ab")?)?,
            type_cd: DataType::parse(fields.get("cd")?)?,
            bias: flag("bias")?,
        };
        let index = fields.get("index")?.parse::<i32>().ok()?;
        Some((shape, index))
    }
}

/// An error of the plan table. Each one stops the engine's open or its job (spec 8.5).
#[derive(Debug)]
pub enum PlanError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// A line of the file is not a pick, or the file is not a plan table.
    Malformed {
        path: PathBuf,
        line: usize,
        text: String,
    },
    /// The file names another lineage.
    Lineage {
        path: PathBuf,
        file: String,
        want: String,
    },
    /// A shape has a pick, and a different pick came for it.
    Repick {
        shape: GemmShape,
        held: i32,
        new: i32,
    },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanError::Io { path, source } => write!(f, "the plan table {} failed: {source}", path.display()),
            PlanError::Malformed { path, line, text } => write!(
                f,
                "line {line} of the plan table {} is not a pick: `{text}`. Move the file away to start a new table",
                path.display()
            ),
            PlanError::Lineage { path, file, want } => write!(
                f,
                "the plan table {} is for the lineage `{file}`, not `{want}`. Move the file away to start a new table",
                path.display()
            ),
            PlanError::Repick { shape, held, new } => write!(
                f,
                "the shape {shape:?} has the pick {held}, and the table only grows: the new pick {new} is refused"
            ),
        }
    }
}

impl std::error::Error for PlanError {}

/// The FNV-1a hash of `text`, for the file name of a lineage.
fn fnv1a(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}

/// The picks of one lineage, and the file that keeps them.
#[derive(Debug)]
pub struct PlanTable {
    lineage: String,
    path: Option<PathBuf>,
    picks: BTreeMap<GemmShape, i32>,
}

impl PlanTable {
    /// A table with no file. Its picks end with the process. Tests use it.
    pub fn in_memory(lineage: &str) -> PlanTable {
        PlanTable {
            lineage: lineage.to_owned(),
            path: None,
            picks: BTreeMap::new(),
        }
    }

    /// The path of the table of `lineage` in the directory `dir`.
    pub fn path_in(dir: &Path, lineage: &str) -> PathBuf {
        dir.join(format!("plan-table-{:016x}.txt", fnv1a(lineage)))
    }

    /// Open the table of `lineage` in `dir`. An absent file is an empty table, and the first pick makes the file.
    pub fn open(dir: &Path, lineage: &str) -> Result<PlanTable, PlanError> {
        let path = PlanTable::path_in(dir, lineage);
        let mut table = PlanTable {
            lineage: lineage.to_owned(),
            path: Some(path.clone()),
            picks: BTreeMap::new(),
        };
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(table),
            Err(source) => return Err(PlanError::Io { path, source }),
        };
        let malformed = |line: usize, text: &str| PlanError::Malformed {
            path: path.clone(),
            line,
            text: text.to_owned(),
        };
        if !text.ends_with('\n') {
            return Err(malformed(text.lines().count(), text.lines().last().unwrap_or("")));
        }
        let mut lines = text.lines().enumerate();
        match lines.next() {
            Some((_, MAGIC)) => {}
            Some((index, other)) => return Err(malformed(index + 1, other)),
            None => return Err(malformed(1, "")),
        }
        match lines.next() {
            Some((_, line)) if line.strip_prefix("lineage ") == Some(lineage) => {}
            Some((_, line)) if line.starts_with("lineage ") => {
                return Err(PlanError::Lineage {
                    path,
                    file: line["lineage ".len()..].to_owned(),
                    want: lineage.to_owned(),
                });
            }
            Some((index, other)) => return Err(malformed(index + 1, other)),
            None => return Err(malformed(2, "")),
        }
        for (index, line) in lines {
            let (shape, pick) = GemmShape::parse(line).ok_or_else(|| malformed(index + 1, line))?;
            if table.picks.insert(shape, pick).is_some() {
                return Err(malformed(index + 1, line));
            }
        }
        Ok(table)
    }

    /// The lineage of the table.
    pub fn lineage(&self) -> &str {
        &self.lineage
    }

    /// The pick of `shape`, if it has one.
    pub fn pick(&self, shape: &GemmShape) -> Option<i32> {
        self.picks.get(shape).copied()
    }

    /// The count of picks.
    pub fn len(&self) -> usize {
        self.picks.len()
    }

    /// True when the table holds no pick.
    pub fn is_empty(&self) -> bool {
        self.picks.is_empty()
    }

    /// Keep the pick `index` for `shape`. The file gets the line, and its data reaches the disk, before this returns.
    /// The same pick again changes nothing. A different pick is refused.
    pub fn record(&mut self, shape: GemmShape, index: i32) -> Result<(), PlanError> {
        if let Some(held) = self.pick(&shape) {
            return if held == index {
                Ok(())
            } else {
                Err(PlanError::Repick {
                    shape,
                    held,
                    new: index,
                })
            };
        }
        if let Some(path) = &self.path {
            let io = |source| PlanError::Io {
                path: path.clone(),
                source,
            };
            let new = !path.exists();
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(io)?;
            let mut text = String::new();
            if new {
                text.push_str(&format!("{MAGIC}\nlineage {}\n", self.lineage));
            }
            text.push_str(&shape.line(index));
            text.push('\n');
            file.write_all(text.as_bytes()).map_err(io)?;
            file.sync_data().map_err(io)?;
        }
        self.picks.insert(shape, index);
        Ok(())
    }
}

/// Why the plan table refuses a hipBLASLt kernel by its name, or `None` (spec 12.1). A kernel that adds the parts of a
/// sum with atomics can give other bits on each run.
///
/// Tensile names the split of the sum over K in each kernel name. `GSUAMB` writes each part to its own buffer and
/// sums the buffers in order, with no atomics. `GSUASB` adds the parts into one buffer with atomics. `SK<n>` with `n`
/// other than 0 is stream-K, whose tiles can also meet in atomics. hipBLASLt 1.2.2 on gfx1151 gave only `GSUAMB` and
/// `SK0` for the shapes of Qwen-Image (`examples/gemm_candidates.rs`, 2026-11-01).
pub fn refused(kernel: &str) -> Option<&'static str> {
    let tokens = || kernel.split('_');
    if tokens().any(|token| token == "GSUASB") {
        return Some("GSUASB adds the split sums with atomics");
    }
    let stream_k = tokens().any(|token| {
        token
            .strip_prefix("SK")
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) && n.bytes().any(|b| b != b'0'))
    });
    if stream_k {
        return Some("stream-K can add tiles with atomics");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(m: u32) -> GemmShape {
        GemmShape::linear(16, 64, m, DataType::Bf16)
    }

    #[test]
    fn a_line_reads_back_as_its_shape_and_pick() {
        let mut s = shape(32);
        s.batch = 3;
        s.stride_a = 7;
        s.stride_b = 9;
        s.stride_c = 11;
        s.type_cd = DataType::F32;
        assert_eq!(GemmShape::parse(&s.line(-5)), Some((s, -5)));
        let b = s.with_bias();
        assert_eq!(GemmShape::parse(&b.line(1)), Some((b, 1)));
        assert_ne!(b.line(1), s.line(1));
        assert_eq!(GemmShape::parse(&s.line(12)), Some((s, 12)));
    }

    #[test]
    fn a_line_with_a_missing_a_repeated_or_a_bad_field_is_refused() {
        let line = shape(32).line(4);
        assert!(GemmShape::parse(&line.replace(" index=4", "")).is_none());
        assert!(GemmShape::parse(&format!("{line} m=1")).is_none());
        assert!(GemmShape::parse(&line.replace("ta=1", "ta=2")).is_none());
        assert!(GemmShape::parse(&line.replace("ab=bf16", "ab=f16")).is_none());
        assert!(GemmShape::parse(&line.replace("m=32", "m=x")).is_none());
    }

    #[test]
    fn the_shape_of_a_linear_layer_reads_the_weight_transposed() {
        let s = GemmShape::linear(10, 64, 32, DataType::Bf16);
        assert_eq!((s.m, s.n, s.k), (32, 10, 64));
        assert!(s.trans_a && !s.trans_b);
        assert_eq!((s.lda, s.ldb, s.ldc), (64, 64, 32));
        assert_eq!(s.elements(), (64 * 32, 64 * 10, 32 * 10));
    }

    #[test]
    fn the_output_elements_skip_the_padding_of_each_column() {
        let mut s = GemmShape::linear(2, 8, 3, DataType::Bf16);
        s.ldc = 5;
        s.batch = 2;
        s.stride_c = 3;
        let got: Vec<usize> = s.output_elements().collect();
        assert_eq!(got, [0, 1, 2, 5, 6, 7, 3, 4, 5, 8, 9, 10]);
    }

    #[test]
    fn atomic_split_sums_and_stream_k_are_refused_by_name() {
        let ok = "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT96x96x32_GSUAMB_GLS0_SK0_SKFTR0_SKXCCM0_WG32_4_1";
        assert_eq!(refused(ok), None);
        assert!(refused(&ok.replace("GSUAMB", "GSUASB")).is_some());
        assert!(refused(&ok.replace("_SK0_", "_SK3_")).is_some());
        assert!(refused(&ok.replace("_SK0_", "_SK10_")).is_some());
        assert_eq!(
            refused(&ok.replace("_SKFTR0_", "_SKFTR1_")),
            None,
            "SKFTR is not stream-K itself"
        );
    }
}
