//! The pending ids of the outputs that a reply writes (gogi "Outputs").
//!
//! A pending id names an output while the server makes it. A `GET` of the id streams its bytes as they come, and ends
//! when the output is sealed. After the seal, the id is a `308` to the content id for at least 60 s. A cancelled output
//! leaves no id, so its `GET` is a `404`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::watch;

/// How long a sealed pending id stays a `308`. gogi asks for at least 60 s.
pub const SEALED_FOR: Duration = Duration::from_secs(600);

/// The bytes of one output so far, and how it ended.
#[derive(Debug, Default)]
struct Bytes {
    data: Vec<u8>,
    /// The content id, once the output is sealed.
    sealed: Option<String>,
    /// True when the output will not be sealed: the reply failed or its client went away.
    cancelled: bool,
}

/// One output that the server makes. The writer appends bytes, and readers wait for them.
#[derive(Debug)]
pub struct PendingOutput {
    /// The media type of the output.
    pub mime: String,
    bytes: Mutex<Bytes>,
    /// Counts each change, so a reader wakes when bytes come or the output ends.
    changes: watch::Sender<u64>,
}

impl PendingOutput {
    fn change(&self) {
        self.changes.send_modify(|count| *count += 1);
    }

    /// Append bytes of the output.
    pub fn push(&self, data: &[u8]) {
        self.bytes
            .lock()
            .expect("the pending lock is poisoned")
            .data
            .extend_from_slice(data);
        self.change();
    }

    /// The bytes from `from` on, and the end of the output when it has one: its content id, or `None` when it was
    /// cancelled. A reader that gets no bytes and no end waits with [`PendingOutput::changed`].
    pub fn read(&self, from: usize) -> (Vec<u8>, Option<Option<String>>) {
        let bytes = self.bytes.lock().expect("the pending lock is poisoned");
        let end = if bytes.cancelled {
            Some(None)
        } else {
            bytes.sealed.clone().map(Some)
        };
        (bytes.data[from.min(bytes.data.len())..].to_vec(), end)
    }

    /// A receiver that wakes at the next change.
    pub fn changed(&self) -> watch::Receiver<u64> {
        self.changes.subscribe()
    }
}

/// What a pending id names now.
#[derive(Debug)]
pub enum Lookup {
    /// The output is being made.
    Making(Arc<PendingOutput>),
    /// The output is sealed under this content id.
    Sealed(String),
    /// The server never gave the id, or the output was cancelled, or the seal is older than [`SEALED_FOR`].
    Unknown,
}

#[derive(Debug)]
enum Entry {
    Making(Arc<PendingOutput>),
    Sealed { id: String, at: Instant },
}

/// The pending ids of a process.
#[derive(Debug, Default)]
pub struct PendingIds {
    entries: Mutex<BTreeMap<String, Entry>>,
}

impl PendingIds {
    /// A new pending id and its output. The id is `pending:` and 64 hex digits of a BLAKE3 hash of a count and the
    /// time, so it differs across outputs and restarts.
    pub fn start(&self, mime: &str) -> (String, Arc<PendingOutput>) {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let mut hasher = blake3::Hasher::new_derive_key("nagaya pending output v1");
        hasher.update(&NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst).to_le_bytes());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos());
        hasher.update(&now.to_le_bytes());
        hasher.update(&std::process::id().to_le_bytes());
        let id = format!("pending:{}", hasher.finalize().to_hex());
        let output = Arc::new(PendingOutput {
            mime: mime.to_owned(),
            bytes: Mutex::new(Bytes::default()),
            changes: watch::channel(0).0,
        });
        let mut entries = self.entries.lock().expect("the pending lock is poisoned");
        // Drop the seals that are old enough, so the map holds only the outputs of the last minutes.
        entries.retain(|_, entry| !matches!(entry, Entry::Sealed { at, .. } if at.elapsed() > SEALED_FOR));
        let old = entries.insert(id.clone(), Entry::Making(Arc::clone(&output)));
        assert!(old.is_none(), "the pending id {id} came twice");
        (id, output)
    }

    /// Seal the output of `pending` under the content id `id`.
    pub fn seal(&self, pending: &str, id: &str) {
        let mut entries = self.entries.lock().expect("the pending lock is poisoned");
        let Some(Entry::Making(output)) = entries.remove(pending) else {
            panic!("the output {pending} is sealed but it is not being made");
        };
        output.bytes.lock().expect("the pending lock is poisoned").sealed = Some(id.to_owned());
        output.change();
        entries.insert(
            pending.to_owned(),
            Entry::Sealed {
                id: id.to_owned(),
                at: Instant::now(),
            },
        );
    }

    /// Drop the output of `pending`: it will not be sealed. A reader of its bytes sees the end with no seal.
    pub fn cancel(&self, pending: &str) {
        let mut entries = self.entries.lock().expect("the pending lock is poisoned");
        if let Some(Entry::Making(output)) = entries.remove(pending) {
            output.bytes.lock().expect("the pending lock is poisoned").cancelled = true;
            output.change();
        }
    }

    /// What the pending id names now.
    pub fn lookup(&self, pending: &str) -> Lookup {
        let entries = self.entries.lock().expect("the pending lock is poisoned");
        match entries.get(pending) {
            Some(Entry::Making(output)) => Lookup::Making(Arc::clone(output)),
            Some(Entry::Sealed { id, at }) if at.elapsed() <= SEALED_FOR => Lookup::Sealed(id.clone()),
            _ => Lookup::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_output_reads_as_it_grows_and_seals_to_its_id() {
        let ids = PendingIds::default();
        let (pending, output) = ids.start("audio/wav");
        assert!(gogi_proto::ids::pending_id_problem(&pending).is_none(), "{pending}");
        output.push(b"RIFF");
        output.push(b"data");
        assert_eq!(output.read(2), (b"FFdata".to_vec(), None));
        assert!(matches!(ids.lookup(&pending), Lookup::Making(_)));
        ids.seal(&pending, "blake3:00");
        assert_eq!(output.read(8), (Vec::new(), Some(Some("blake3:00".to_owned()))));
        assert!(matches!(ids.lookup(&pending), Lookup::Sealed(id) if id == "blake3:00"));
    }

    #[test]
    fn a_cancelled_output_ends_with_no_seal_and_its_id_is_unknown() {
        let ids = PendingIds::default();
        let (pending, output) = ids.start("audio/wav");
        output.push(b"RIFF");
        ids.cancel(&pending);
        assert_eq!(output.read(0), (b"RIFF".to_vec(), Some(None)));
        assert!(matches!(ids.lookup(&pending), Lookup::Unknown));
        assert!(matches!(
            ids.lookup(&format!("pending:{}", "0".repeat(64))),
            Lookup::Unknown
        ));
    }

    #[test]
    fn two_outputs_have_two_ids() {
        let ids = PendingIds::default();
        assert_ne!(ids.start("audio/wav").0, ids.start("audio/wav").0);
    }
}
