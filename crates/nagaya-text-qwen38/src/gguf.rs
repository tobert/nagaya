//! A reader for the metadata of a GGUF file (version 3). It reads the header and the key-value section only.

use crate::Error;
use std::io::Read;
use std::path::Path;

/// One metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// The key-value section of a GGUF file, in file order.
#[derive(Debug, Default)]
pub struct Metadata {
    entries: Vec<(String, Value)>,
}

/// How deep an array may nest. The files that we read nest at most one level.
const MAX_DEPTH: usize = 16;
/// The magic bytes at the start of a GGUF file.
const MAGIC: &[u8; 4] = b"GGUF";
/// The one version of the format that this reader knows.
const VERSION: u32 = 3;

impl Metadata {
    /// Read the header and the key-value section from `reader`, which holds `len` bytes in all.
    ///
    /// The reader stops after the last entry. It never reads the tensor data. `len` bounds each length field, so a
    /// damaged file cannot make the reader allocate more than the file holds.
    pub fn read(reader: impl Read, len: u64) -> Result<Metadata, Error> {
        let mut r = Reader {
            inner: reader,
            pos: 0,
            len,
        };
        let magic = r.take(4)?;
        if magic != MAGIC {
            return Err(Error::Gguf(format!(
                "the magic bytes are {magic:?}, not \"GGUF\". Give a GGUF file."
            )));
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(Error::Gguf(format!(
                "the file is version {version}. This reader reads version {VERSION}."
            )));
        }
        let _tensor_count = r.u64()?;
        let kv_count = r.u64()?;
        let mut entries: Vec<(String, Value)> = Vec::new();
        for _ in 0..kv_count {
            let key = r.string()?;
            if entries.iter().any(|(k, _)| *k == key) {
                return Err(Error::Gguf(format!(
                    "the key {key:?} is in the file twice. Repair the file."
                )));
            }
            let value_type = r.u32()?;
            let value = r.value(value_type, 0)?;
            entries.push((key, value));
        }
        Ok(Metadata { entries })
    }

    /// Read the metadata of the GGUF file at `path`.
    pub fn open(path: &Path) -> Result<Metadata, Error> {
        let io = |source| Error::Io {
            path: path.display().to_string(),
            source,
        };
        let file = std::fs::File::open(path).map_err(io)?;
        let len = file.metadata().map_err(io)?.len();
        Metadata::read(std::io::BufReader::with_capacity(1 << 20, file), len)
    }

    /// The value of `key`, or `None` when the file has no such key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

struct Reader<R> {
    inner: R,
    pos: u64,
    len: u64,
}

impl<R: Read> Reader<R> {
    fn left(&self) -> u64 {
        self.len.saturating_sub(self.pos)
    }

    fn ends_early(&self, wanted: u64) -> Error {
        Error::Gguf(format!(
            "the file ends at byte {} but the metadata needs {wanted} more bytes. The file is cut short.",
            self.len
        ))
    }

    fn take(&mut self, n: u64) -> Result<Vec<u8>, Error> {
        if n > self.left() {
            return Err(self.ends_early(n));
        }
        let mut buf = vec![0u8; n as usize];
        self.inner
            .read_exact(&mut buf)
            .map_err(|e| Error::Gguf(format!("a read failed at byte {}: {e}", self.pos)))?;
        self.pos += n;
        Ok(buf)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let bytes = self.take(N as u64)?;
        Ok(bytes.try_into().expect("take gives N bytes"))
    }

    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn string(&mut self) -> Result<String, Error> {
        let n = self.u64()?;
        if n > self.left() {
            return Err(Error::Gguf(format!(
                "a string of {n} bytes is longer than the {} bytes that are left in the file",
                self.left()
            )));
        }
        String::from_utf8(self.take(n)?)
            .map_err(|e| Error::Gguf(format!("a string is not UTF-8 ({e}). Repair the file.")))
    }

    fn value(&mut self, value_type: u32, depth: usize) -> Result<Value, Error> {
        Ok(match value_type {
            0 => Value::U8(self.array::<1>()?[0]),
            1 => Value::I8(self.array::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.array()?)),
            3 => Value::I16(i16::from_le_bytes(self.array()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.array()?)),
            6 => Value::F32(f32::from_le_bytes(self.array()?)),
            7 => match self.array::<1>()?[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                other => {
                    return Err(Error::Gguf(format!(
                        "a bool holds {other}, not 0 or 1. Repair the file."
                    )));
                }
            },
            8 => Value::Str(self.string()?),
            9 => {
                if depth >= MAX_DEPTH {
                    return Err(Error::Gguf(format!("an array nests deeper than {MAX_DEPTH} levels")));
                }
                let element_type = self.u32()?;
                let n = self.u64()?;
                let least = least_size(element_type)?;
                if n.checked_mul(least).is_none_or(|bytes| bytes > self.left()) {
                    return Err(Error::Gguf(format!(
                        "an array of {n} values is longer than the {} bytes that are left in the file",
                        self.left()
                    )));
                }
                let mut items = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    items.push(self.value(element_type, depth + 1)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.array()?)),
            12 => Value::F64(f64::from_le_bytes(self.array()?)),
            other => return Err(Error::Gguf(format!("the value type {other} is not a GGUF type"))),
        })
    }
}

/// The fewest bytes that one value of a type takes in the file.
fn least_size(value_type: u32) -> Result<u64, Error> {
    Ok(match value_type {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        8 | 10..=12 => 8,
        9 => 12,
        other => return Err(Error::Gguf(format!("the value type {other} is not a GGUF type"))),
    })
}
