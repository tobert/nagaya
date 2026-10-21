//! The park of an engine state to the store (spec 10.3 step 3, ABI D7).

use std::path::Path;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{SystemTime, UNIX_EPOCH};

use nagaya_abi::{BitClass, CoverageSpan, Engine, Identity, PartKind, PartLimits, State, Tier as MemoryTier};
use nagaya_ledger::Admission;

use crate::profile::StoreSpec;
use nagaya_store::{
    ContentId, EngineFacts, EntryState, Put, Reason, RecipeId, Snapshot, Store, StoreError, Tensor, Tier, recipe_root,
};

use crate::process::ParkReport;

/// The owner of a park entry: the process itself, not a client.
pub const PARK_OWNER: u64 = 0;

/// The store's budgets for the park (spec 4.1 note).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreBudget {
    /// The budget of the store's host-memory tier.
    pub host: u64,
    /// The budget of the store's NVMe tier.
    pub nvme: u64,
    /// The most interval copies that the store keeps (spec 6.13).
    pub interval_copies: u32,
}

impl StoreBudget {
    /// The store's budget from the profile's `[store]` table and the admission (spec 4.1a, 2.4).
    pub fn from_spec(spec: &StoreSpec, admission: &Admission) -> Self {
        StoreBudget {
            host: spec.host.unwrap_or_else(|| Self::host_from_admission(admission)),
            nvme: spec.nvme,
            interval_copies: spec.interval_copies,
        }
    }

    /// The store's host budget from a fitting profile: the unassigned host-only bytes (spec 4.4 step 4).
    pub fn host_from_admission(admission: &Admission) -> u64 {
        admission
            .unassigned
            .iter()
            .filter(|(tiers, _)| tiers.as_slice() == [MemoryTier::Host])
            .map(|(_, bytes)| *bytes)
            .sum()
    }
}

/// A request to the store thread.
pub(crate) enum StoreMessage {
    /// Store one parked state.
    Put {
        /// The canonical snapshot.
        snapshot: Snapshot,
        /// The content id of the snapshot.
        content: ContentId,
        /// The recipe of the state.
        recipe: RecipeId,
        /// The engine name, its class and its chain limits.
        engine: Box<EngineFacts>,
        /// The digest of the engine identity.
        identity: [u8; 32],
        /// The owner of the entry.
        owner: u64,
        /// The time of the write.
        now_ms: u64,
        /// The answer of the store.
        reply: Sender<Result<ContentId, StoreError>>,
    },
    /// Stop the store thread.
    Stop,
}

/// Open the machine's store with the park's budgets.
pub fn open_store(dir: &Path, budget: StoreBudget) -> Result<Store, StoreError> {
    Store::open_with_limits(dir, budget.host, budget.nvme, u64::MAX, budget.interval_copies)
}

/// Store each state of one engine that holds tokens, and add its counts or its failure to `report`.
pub(crate) fn park_engine(
    name: &str,
    engine: &mut dyn Engine,
    store: Option<&Sender<StoreMessage>>,
    report: &mut ParkReport,
) {
    let limits = engine.limits();
    let stored: Vec<&PartLimits> = limits.parts.iter().filter(|part| part.is_stored()).collect();
    if stored.is_empty() {
        return;
    }
    let names = part_names(&stored);
    let has_row = stored.iter().any(|part| part.is_row());
    let slots: u32 = limits.slots.iter().map(|slot| slot.count).sum();
    let identity = identity_digest(&engine.identity());
    for id in 0..slots {
        let state = State::Slot(id);
        // A state with no token has nothing to keep (spec 10.3). Without the slots capability, `park_snapshot`
        // reports that a row part has no row count.
        if has_row && engine.slots().is_some_and(|slots| slots.slot_high(id) == 0) {
            continue;
        }
        let snapshot = match park_snapshot(engine, state) {
            Ok(snapshot) => snapshot,
            Err(message) => {
                report.failures.push(format!("engine `{name}` state {id} {message}"));
                continue;
            }
        };
        if snapshot.tensors.is_empty() {
            continue;
        }
        let Some(store) = store else {
            report.failures.push(format!(
                "engine `{name}` state {id} parts {names} were not stored: the process has no store"
            ));
            continue;
        };
        let content = match snapshot.content_id() {
            Ok(content) => content,
            Err(error) => {
                report
                    .failures
                    .push(format!("engine `{name}` state {id} parts {names} {error}"));
                continue;
            }
        };
        let parts = snapshot.tensors.len();
        let bytes: u64 = snapshot.tensors.iter().map(|tensor| tensor.data.len() as u64).sum();
        let (reply, answer) = std::sync::mpsc::channel();
        let message = StoreMessage::Put {
            snapshot,
            content,
            recipe: park_recipe(&identity, &content),
            engine: Box::new(EngineFacts::of(name, &limits)),
            identity,
            owner: PARK_OWNER,
            now_ms: now_ms(),
            reply,
        };
        if store.send(message).is_err() {
            report.failures.push(format!(
                "engine `{name}` state {id} parts {names} were not stored: the store thread stopped"
            ));
            continue;
        }
        match answer.recv() {
            Ok(Ok(_content_id)) => {
                report.states += 1;
                report.parts += parts;
                report.bytes += bytes;
                report.entries += 1;
            }
            Ok(Err(error)) => report.failures.push(format!(
                "engine `{name}` state {id} parts {names} were not stored: {error}"
            )),
            Err(_) => report.failures.push(format!(
                "engine `{name}` state {id} parts {names} were not stored: the store thread stopped"
            )),
        }
    }
}

/// Store one parked state (spec 10.3 step 3). `content` is the content id of the put's snapshot.
///
/// The store can hold those bytes already under another recipe, for example the head that a build put (11.3c). Then
/// the park does not put. It parks that entry to NVMe and keeps its recipe. An interval copy never goes to NVMe
/// (6.13), so the park removes the copy and puts the state under the park recipe.
pub fn store_park(store: &mut Store, put: Put<'_>, content: ContentId) -> Result<ContentId, StoreError> {
    let held = store
        .entries()
        .get(&content)
        .filter(|entry| entry.state == EntryState::Live && entry.recipe != put.recipe)
        .map(|entry| entry.reason);
    match held {
        None => store.put(put),
        Some(Reason::Interval) => {
            store.remove(&content, put.now_ms)?;
            store.put(put)
        }
        Some(_) => store.park(&content, put.now_ms).map(|()| content),
    }
}

/// Build the canonical snapshot of one state (ABI D10).
pub(crate) fn park_snapshot(engine: &mut dyn Engine, state: State) -> Result<Snapshot, String> {
    let State::Slot(slot) = state;
    let limits = engine.limits();
    let stored: Vec<(usize, &PartLimits)> = limits
        .parts
        .iter()
        .enumerate()
        .filter(|(_, part)| part.is_stored())
        .collect();
    if stored.is_empty() {
        return Ok(Snapshot {
            tensors: Vec::new(),
            parent: None,
        });
    }
    let has_row = stored.iter().any(|(_, part)| part.is_row());
    let high = if has_row {
        let Some(slots) = engine.slots() else {
            let part = stored.iter().find(|(_, part)| part.is_row()).expect("a row part");
            return Err(format!(
                "part `{}` has no row count: the engine has no slots capability",
                part.1.name
            ));
        };
        slots.slot_high(slot)
    } else {
        0
    };

    let mut tensors = Vec::with_capacity(stored.len());
    for (id, part) in stored {
        match part.kind {
            PartKind::Derived => continue,
            PartKind::Fixed => {
                let mut data = vec![0u8; shape_bytes(part) as usize];
                engine
                    .parts()
                    .ok_or_else(|| format!("part `{}` has no reader: the engine has no parts capability", part.name))?
                    .part_read(state, id, &mut data)
                    .map_err(|error| format!("part `{}` read failed: {error:?}", part.name))?;
                tensors.push(Tensor {
                    name: part.name.clone(),
                    kind: part.kind,
                    dtype: part.dtype,
                    shape: part.shape.clone(),
                    data,
                    coverage: None,
                    block_format: None,
                });
            }
            PartKind::Cells | PartKind::Ring => {
                let start = if part.kind == PartKind::Ring {
                    engine
                        .coverage()
                        .ok_or_else(|| {
                            format!(
                                "part `{}` has no base: the engine has no coverage capability",
                                part.name
                            )
                        })?
                        .coverage(state, id, &[])
                        .start
                } else {
                    0
                };
                let count = high.saturating_sub(start);
                let row_bytes = shape_bytes(part) as usize;
                let length = (count as usize)
                    .checked_mul(row_bytes)
                    .ok_or_else(|| format!("part `{}` is too large", part.name))?;
                let mut data = vec![0u8; length];
                engine
                    .parts()
                    .ok_or_else(|| format!("part `{}` has no reader: the engine has no parts capability", part.name))?
                    .cells_read(state, id, start, count, &mut data)
                    .map_err(|error| format!("part `{}` read failed: {error:?}", part.name))?;
                let span = match engine.coverage() {
                    Some(coverage) => {
                        let span = coverage.coverage(state, id, &data);
                        if span.start > span.end {
                            return Err(format!(
                                "part `{}` has coverage {}..{}",
                                part.name, span.start, span.end
                            ));
                        }
                        // The store refuses an entry that cannot restore, as `snapshot_from_engine` does.
                        coverage.fits(&span, slot).map_err(|why| {
                            format!(
                                "part `{}` has coverage {}..{} that does not fit slot {slot}: {why}",
                                part.name, span.start, span.end
                            )
                        })?;
                        span
                    }
                    None => CoverageSpan {
                        part: id,
                        start,
                        end: start + count,
                    },
                };
                let mut shape = vec![span.end - span.start];
                shape.extend_from_slice(&part.shape);
                tensors.push(Tensor {
                    name: part.name.clone(),
                    kind: part.kind,
                    dtype: part.dtype,
                    shape,
                    data,
                    coverage: Some((span.start, span.end)),
                    block_format: None,
                });
            }
        }
    }
    Ok(Snapshot { tensors, parent: None })
}

/// The geometry key of a park recipe (tag 2 of the root recipe).
///
/// A token recipe takes its geometry key from the engine. The marker keeps a park recipe apart from it.
pub const PARK_GEOMETRY_KEY: &[u8] = b"nagaya park v1";

/// The recipe of a parked state (spec 6.2).
///
/// Jobs do not track the inputs of a context yet (milestone 12), so a park does not know them. Thus the
/// recipe names the bytes. It is a root recipe over the identity digest. Tag 2 holds `PARK_GEOMETRY_KEY`,
/// tag 3 holds no approximations, and tag 4 holds the 32 bytes of the snapshot's content id.
/// Equal bytes give one recipe and one entry. Different bytes give different recipes, so the tripwire fires
/// only for a real duplicate. A token recipe cannot equal it, because its tag 2 holds the engine's own key.
pub fn park_recipe(identity: &[u8; 32], content: &ContentId) -> RecipeId {
    // The store gives no byte accessor for a content id, so the 32 bytes come from its hex form.
    let hex = content.hex();
    let bytes: Vec<u8> = (0..32)
        .map(|index| u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).expect("a content id is 64 hex digits"))
        .collect();
    recipe_root(identity, PARK_GEOMETRY_KEY, &[], &bytes)
}

/// The digest of an engine identity (spec 6.2, 12.1).
pub fn identity_digest(identity: &Identity) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    add(&mut hasher, 1, identity.model.as_bytes());
    add(&mut hasher, 2, identity.checkpoint_digest.as_bytes());
    add(&mut hasher, 3, identity.tokenizer_hash.as_bytes());
    add(&mut hasher, 4, identity.template.as_bytes());
    let mut numbers = Vec::new();
    numbers.extend_from_slice(&(identity.numbers.len() as u64).to_le_bytes());
    for (name, value) in &identity.numbers {
        add_to(&mut numbers, name.as_bytes());
        add_to(&mut numbers, value.as_bytes());
    }
    add(&mut hasher, 5, &numbers);
    add(&mut hasher, 6, identity.plan_table.as_bytes());
    let mut settings = Vec::new();
    settings.extend_from_slice(&(identity.bit_settings.len() as u64).to_le_bytes());
    for setting in &identity.bit_settings {
        add_to(&mut settings, setting.name.as_bytes());
        add_to(&mut settings, setting.value.as_bytes());
        settings.push(match setting.class {
            BitClass::ReferenceDefining => 1,
            BitClass::Approx => 2,
            BitClass::Policy => 3,
        });
    }
    add(&mut hasher, 7, &settings);
    add(&mut hasher, 8, identity.canary_hash.as_bytes());
    *hasher.finalize().as_bytes()
}

fn add(hasher: &mut blake3::Hasher, tag: u16, bytes: &[u8]) {
    hasher.update(&tag.to_le_bytes());
    add_to(hasher, bytes);
}

fn add_to(writer: &mut impl std::io::Write, bytes: &[u8]) {
    writer
        .write_all(&(bytes.len() as u64).to_le_bytes())
        .expect("a vector write cannot fail");
    writer.write_all(bytes).expect("a vector write cannot fail");
}

fn part_names(parts: &[&PartLimits]) -> String {
    parts
        .iter()
        .map(|part| format!("`{}`", part.name))
        .collect::<Vec<String>>()
        .join(", ")
}

fn shape_bytes(part: &PartLimits) -> u64 {
    part.shape.iter().product::<u64>() * part.dtype.element_bytes()
}

fn now_ms() -> u64 {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("the clock is before the epoch");
    elapsed.as_millis() as u64
}

/// The store thread: it writes the parks of the group threads (spec 6.1). The lock is shared with the group threads,
/// because a build runs `lookup` and `restore` on its group thread (spec 11.3c).
pub(crate) fn store_thread(store: &std::sync::Mutex<Store>, messages: Receiver<StoreMessage>) {
    while let Ok(message) = messages.recv() {
        match message {
            StoreMessage::Put {
                snapshot,
                content,
                recipe,
                engine,
                identity,
                owner,
                now_ms,
                reply,
            } => {
                let mut store = store.lock().expect("the store lock is poisoned");
                let put = Put {
                    snapshot: &snapshot,
                    recipe,
                    previous_recipe: None,
                    chains_from: None,
                    engine: *engine,
                    identity,
                    owner,
                    tier: Tier::Nvme,
                    reason: Reason::Shutdown,
                    sealed_recipe: None,
                    provenance: None,
                    now_ms,
                };
                let result = store_park(&mut store, put, content);
                let _ = reply.send(result);
            }
            StoreMessage::Stop => break,
        }
    }
}
