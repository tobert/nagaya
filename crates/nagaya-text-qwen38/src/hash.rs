//! The tokenizer hash: the identity of the tokenizer that a GGUF file holds.
//!
//! The engine identity carries this hash in `tokenizer_hash` (engine ABI "Identity"). The binding of the Qwen3.8 engine and this
//! library call the one function below, so they cannot compute two hashes for one file. The function lives in this
//! crate because this crate owns the GGUF reader and reads the same keys. The crate of the Qwen3.8 engine depends on this crate. This
//! crate does not depend on any engine crate, so the dependency has no cycle.

use sha2::{Digest, Sha256};

use crate::Error;
use crate::gguf::{Metadata, Value};

/// A label that starts every hash. A change to the encoding raises the version in the label.
const LABEL: &[u8] = b"nagaya-tokenizer-hash-v1\0";

/// The keys that the hash covers and the file must hold, in the order that the hash reads them.
///
/// These are the keys that `Tokenizer::from_metadata` reads, besides the chat template. The chat template is the
/// identity's `template`, so it is not here.
pub const REQUIRED_KEYS: [&str; 5] = [
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
];

/// The special-token ids that the hash covers when the file holds them, in the order that the hash reads them. The
/// hash records an absent key, so a file without a key and a file with it have different hashes.
pub const OPTIONAL_KEYS: [&str; 5] = [
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.unknown_token_id",
];

/// The hash of the tokenizer keys of a GGUF file: `sha256:` and 64 lowercase hex digits.
///
/// The hash covers `REQUIRED_KEYS`, then `OPTIONAL_KEYS`, in that fixed order and not in the order of the file.
/// For each key it feeds SHA-256 with the key as a length and its bytes, then `0` for an absent key or `1` and the
/// value. A value is its GGUF type code as one byte and its payload. A number is little-endian, a float is its bit
/// pattern, a string is a length and its bytes, and an array is a count and its values. Each length and count is a
/// `u64` in little-endian. The hash does not depend on the other keys.
pub fn tokenizer_hash(metadata: &Metadata) -> Result<String, Error> {
    let mut hasher = Sha256::new();
    hasher.update(LABEL);
    for key in REQUIRED_KEYS.iter().chain(OPTIONAL_KEYS.iter()) {
        feed_string(&mut hasher, key);
        match metadata.get(key) {
            Some(value) => {
                hasher.update([1u8]);
                feed_value(&mut hasher, value);
            }
            None if REQUIRED_KEYS.contains(key) => {
                return Err(Error::Metadata(format!(
                    "the key {key} is missing, so the tokenizer has no hash. Use a GGUF made for this family."
                )));
            }
            None => hasher.update([0u8]),
        }
    }
    let digest = hasher.finalize();
    let mut text = String::from("sha256:");
    for byte in digest {
        text.push_str(&format!("{byte:02x}"));
    }
    Ok(text)
}

fn feed_string(hasher: &mut Sha256, text: &str) {
    hasher.update((text.len() as u64).to_le_bytes());
    hasher.update(text.as_bytes());
}

fn feed_value(hasher: &mut Sha256, value: &Value) {
    // The tag is the GGUF type code of the value.
    match value {
        Value::U8(v) => {
            hasher.update([0u8, *v]);
        }
        Value::I8(v) => {
            hasher.update([1u8]);
            hasher.update(v.to_le_bytes());
        }
        Value::U16(v) => {
            hasher.update([2u8]);
            hasher.update(v.to_le_bytes());
        }
        Value::I16(v) => {
            hasher.update([3u8]);
            hasher.update(v.to_le_bytes());
        }
        Value::U32(v) => {
            hasher.update([4u8]);
            hasher.update(v.to_le_bytes());
        }
        Value::I32(v) => {
            hasher.update([5u8]);
            hasher.update(v.to_le_bytes());
        }
        Value::F32(v) => {
            hasher.update([6u8]);
            hasher.update(v.to_bits().to_le_bytes());
        }
        Value::Bool(v) => {
            hasher.update([7u8, u8::from(*v)]);
        }
        Value::Str(v) => {
            hasher.update([8u8]);
            feed_string(hasher, v);
        }
        Value::Array(items) => {
            hasher.update([9u8]);
            hasher.update((items.len() as u64).to_le_bytes());
            for item in items {
                feed_value(hasher, item);
            }
        }
        Value::U64(v) => {
            hasher.update([10u8]);
            hasher.update(v.to_le_bytes());
        }
        Value::I64(v) => {
            hasher.update([11u8]);
            hasher.update(v.to_le_bytes());
        }
        Value::F64(v) => {
            hasher.update([12u8]);
            hasher.update(v.to_bits().to_le_bytes());
        }
    }
}
