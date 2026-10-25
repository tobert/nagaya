//! The codes of each output that a speech service wrote, beside the object, under its content id (spec 11.8a "The
//! codes of an output").
//!
//! The model state of a later line needs the codes of the audio before it, and the engine has no codec encoder. A
//! file holds the 16 codes of each frame, frame after frame, as little-endian u32.

use std::io::Write;
use std::path::{Path, PathBuf};

use nagaya_store::ContentId;

use super::Frame;

/// The directory of the code files in the store directory.
pub const CODES_DIR: &str = "speech-codes";

/// The bytes of one frame in a code file.
const FRAME_BYTES: usize = 16 * 4;

/// The code files of a store directory.
#[derive(Debug)]
pub struct CodeBook {
    dir: PathBuf,
}

impl CodeBook {
    /// Open the code files under `store_dir`, and make their directory when it is absent.
    pub fn open(store_dir: &Path) -> Result<Self, String> {
        let dir = store_dir.join(CODES_DIR);
        std::fs::create_dir_all(&dir).map_err(|error| format!("cannot make `{}`: {error}", dir.display()))?;
        Ok(Self { dir })
    }

    fn path(&self, id: &ContentId) -> PathBuf {
        self.dir.join(id.hex())
    }

    /// Keep the codes of the output `id`. The write goes through a temporary file and a rename, so a reader never
    /// sees a part of a file.
    pub fn put(&self, id: &ContentId, frames: &[Frame]) -> Result<(), String> {
        let path = self.path(id);
        let temporary = path.with_extension("tmp");
        let bytes: Vec<u8> = frames.iter().flatten().flat_map(|code| code.to_le_bytes()).collect();
        let written = std::fs::File::create(&temporary)
            .and_then(|mut file| file.write_all(&bytes).and_then(|()| file.sync_all()))
            .and_then(|()| std::fs::rename(&temporary, &path));
        written.map_err(|error| format!("cannot write the codes of {}: {error}", id.hex()))
    }

    /// Drop the code file of each output that `held` says the object store does not hold, and each temporary file.
    /// It gives the files that it dropped.
    pub fn prune(&self, held: &dyn Fn(&ContentId) -> bool) -> Result<usize, String> {
        let entries =
            std::fs::read_dir(&self.dir).map_err(|error| format!("cannot list `{}`: {error}", self.dir.display()))?;
        let mut dropped = 0;
        for entry in entries {
            let path = entry
                .map_err(|error| format!("cannot list `{}`: {error}", self.dir.display()))?
                .path();
            let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
            let keep = ContentId::from_hex(name).is_ok_and(|id| held(&id));
            if !keep {
                std::fs::remove_file(&path).map_err(|error| format!("cannot remove `{}`: {error}", path.display()))?;
                dropped += 1;
            }
        }
        Ok(dropped)
    }

    /// The codes of the output `id`, or `None` when this book holds none. A file whose size is not a whole count of
    /// frames is an error: the book does not guess at its frames.
    pub fn get(&self, id: &ContentId) -> Result<Option<Vec<Frame>>, String> {
        let path = self.path(id);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("cannot read `{}`: {error}", path.display())),
        };
        if bytes.len() % FRAME_BYTES != 0 {
            return Err(format!(
                "`{}` has {} bytes, which is not a whole count of frames of {FRAME_BYTES} bytes; remove the file, \
                 and speak its line again",
                path.display(),
                bytes.len()
            ));
        }
        let frames = bytes
            .as_chunks::<FRAME_BYTES>()
            .0
            .iter()
            .map(|frame| std::array::from_fn(|g| u32::from_le_bytes(frame[g * 4..g * 4 + 4].try_into().unwrap())))
            .collect();
        Ok(Some(frames))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_read_back_and_an_absent_id_holds_none() {
        let dir = tempfile::tempdir().unwrap();
        let book = CodeBook::open(dir.path()).unwrap();
        let (id, other) = (
            ContentId::from_hex(&blake3::hash(b"a").to_hex()).unwrap(),
            ContentId::from_hex(&blake3::hash(b"b").to_hex()).unwrap(),
        );
        let frames: Vec<Frame> = vec![
            std::array::from_fn(|g| g as u32),
            std::array::from_fn(|g| 100 + g as u32),
        ];
        book.put(&id, &frames).unwrap();
        assert_eq!(book.get(&id).unwrap(), Some(frames));
        assert_eq!(book.get(&other).unwrap(), None);
    }

    #[test]
    fn a_prune_drops_the_codes_of_outputs_that_are_not_held() {
        let dir = tempfile::tempdir().unwrap();
        let book = CodeBook::open(dir.path()).unwrap();
        let id = |text: &[u8]| ContentId::from_hex(&blake3::hash(text).to_hex()).unwrap();
        let (kept, left) = (id(b"kept"), id(b"left"));
        book.put(&kept, &[[1; 16]]).unwrap();
        book.put(&left, &[[2; 16]]).unwrap();
        std::fs::write(dir.path().join(CODES_DIR).join("x.tmp"), b"part").unwrap();
        assert_eq!(book.prune(&|held| *held == kept).unwrap(), 2);
        assert_eq!(book.get(&kept).unwrap(), Some(vec![[1; 16]]));
        assert_eq!(book.get(&left).unwrap(), None);
    }

    #[test]
    fn a_file_of_part_of_a_frame_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let book = CodeBook::open(dir.path()).unwrap();
        let id = ContentId::from_hex(&blake3::hash(b"a").to_hex()).unwrap();
        std::fs::write(dir.path().join(CODES_DIR).join(id.hex()), [0u8; 70]).unwrap();
        assert!(book.get(&id).unwrap_err().contains("70 bytes"));
    }
}
