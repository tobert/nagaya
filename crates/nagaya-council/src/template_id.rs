//! The template id of a served model (spec 11.3a): `nagaya-council:<hash>+<family>-chat:<hash>`.
//!
//! The first hash is over the tokens of the test renders that a family library keeps. The second is the
//! digest of the chat template text. Each hash is the first 16 hex digits of a SHA-256.

use sha2::{Digest, Sha256};

/// The number of hex digits in a hash of a template id.
const DIGITS: usize = 16;

/// The first 16 hex digits of the SHA-256 of `bytes`.
pub fn hash16(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(DIGITS);
    for byte in &digest[..DIGITS / 2] {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// The hash of the test renders of a family. A render is a list of token ids.
///
/// Each render counts as its length (8 bytes, little-endian) and then its tokens (4 bytes each, little-endian).
/// Thus the borders between renders are part of the hash.
pub fn council_hash(renders: &[Vec<u32>]) -> String {
    let mut bytes = Vec::new();
    for render in renders {
        bytes.extend((render.len() as u64).to_le_bytes());
        bytes.extend(render.iter().flat_map(|token| token.to_le_bytes()));
    }
    hash16(&bytes)
}

/// The template id from its two hashes. A hash of another form is a bug in the caller, so it panics.
pub fn template_id(council_hash: &str, family: &str, chat_hash: &str) -> String {
    for hash in [council_hash, chat_hash] {
        assert!(
            hash.len() == DIGITS && hash.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "a hash of a template id is 16 hexadecimal digits in lowercase; got {hash:?}"
        );
    }
    format!("nagaya-council:{council_hash}+{family}-chat:{chat_hash}")
}
