//! Control vectors on the job path (spec 11.12, engine-abi "Control vectors"): the control list of a context after the
//! checks of gogi "Control vectors", and the worker that keeps the vectors in the table of the engine.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use gogi_proto::Invalid;
use gogi_proto::control::{ControlVector, MEDIA_TYPE};
use nagaya_abi::{ControlItem, ControlLimit, ControlOp};
use nagaya_council::control::ControlEntry;
use nagaya_store::ContentId;

use super::error::ServiceError;

/// What the checks need to know of the served model: its weight hash and the control table of its engine.
#[derive(Clone, Debug)]
pub struct Fit<'a> {
    /// The `weight_hash` of the served model.
    pub weight_hash: &'a str,
    /// The control table of the engine.
    pub limit: ControlLimit,
}

/// The bytes and the media type of a held object, or `None` when the store does not hold it.
pub type Lookup<'a> = dyn FnMut(&ContentId) -> Result<Option<(String, Vec<u8>)>, ServiceError> + 'a;

/// A control list after the checks: the entries of its recipe, its items, and the bytes of each vector that it names.
#[derive(Clone, Debug, PartialEq)]
pub struct Checked {
    /// The entries of the root recipe, with the `f64` strengths that the client sent (spec 11.12).
    pub entries: Vec<ControlEntry>,
    /// The items, in order, with the strength as `f32` for the engine.
    pub items: Vec<(ContentId, ControlOp, f32)>,
    /// Each vector that the list names once, in the order of its first item, as `layers` times `width` `f32` values in
    /// layer order (engine-abi "Control vectors"). A layer that the vector does not hold is all zeros.
    pub vectors: Vec<(ContentId, Arc<[u8]>)>,
}

fn op_of(op: gogi_proto::control::ControlOp) -> ControlOp {
    match op {
        gogi_proto::control::ControlOp::Add => ControlOp::Add,
        gogi_proto::control::ControlOp::Ablate => ControlOp::Ablate,
        gogi_proto::control::ControlOp::Clamp => ControlOp::Clamp,
    }
}

/// The store id of a gogi content id. The member rules checked the text, so a failure is a defect.
fn store_id(id: &gogi_proto::ids::ContentId) -> Result<ContentId, ServiceError> {
    ContentId::from_hex(id.as_str().trim_start_matches("blake3:")).map_err(|error| {
        ServiceError::internal(format!("a checked content id {} does not parse: {error}", id.as_str()))
    })
}

/// Check a control list in the order of gogi "Control vectors" (the member rules and the capability are checked
/// before): the media type of a held object, a held object, the bytes of a control vector, and the fit with the model.
/// A strength that is not finite as `f32` is refused too. `at` names the context, such as `contexts[0].`, or is empty.
/// A list that names more vectors than the table holds is a 400 that names the list.
pub fn check(
    list: &[gogi_proto::control::Control],
    at: &str,
    fit: &Fit,
    lookup: &mut Lookup,
) -> Result<Checked, ServiceError> {
    let mut checked = Checked {
        entries: Vec::with_capacity(list.len()),
        items: Vec::with_capacity(list.len()),
        vectors: Vec::new(),
    };
    for (i, item) in list.iter().enumerate() {
        let strength = item.strength as f32;
        if !strength.is_finite() {
            let param = format!("{at}control[{i}].strength");
            return Err(Invalid::param(
                param.clone(),
                format!(
                    "nothing ran: {param} is {}, which is not finite as a 32-bit float; send a strength of at most \
                     {:e} in size",
                    item.strength,
                    f32::MAX
                ),
            )
            .into());
        }
        let param = format!("{at}control[{i}].vector");
        let id = store_id(&item.vector)?;
        if !checked.vectors.iter().any(|(held, _)| *held == id) {
            let bytes = held_vector(&param, &id, item, fit, lookup)?;
            checked.vectors.push((id, bytes));
        }
        let op = op_of(item.op);
        checked.entries.push(ControlEntry {
            vector: id,
            op,
            strength: item.strength,
        });
        checked.items.push((id, op, strength));
    }
    if checked.vectors.len() > fit.limit.tables as usize {
        let param = format!("{at}control");
        return Err(Invalid::param(
            param.clone(),
            format!(
                "nothing ran: {param} names {} control vectors, and the engine holds {} at once; name fewer vectors",
                checked.vectors.len(),
                fit.limit.tables
            ),
        )
        .into());
    }
    Ok(checked)
}

/// The bytes of a held control vector for the table, after the checks of its object.
fn held_vector(
    param: &str,
    id: &ContentId,
    item: &gogi_proto::control::Control,
    fit: &Fit,
    lookup: &mut Lookup,
) -> Result<Arc<[u8]>, ServiceError> {
    let width = fit.limit.width as usize;
    let vector = held(
        param,
        item.vector.as_str(),
        id,
        fit.weight_hash,
        fit.limit.layers,
        width,
        lookup,
    )?;
    let mut values = vec![0f32; fit.limit.layers as usize * width];
    for (layer, row) in &vector.layers {
        let start = *layer as usize * width;
        values[start..start + width].copy_from_slice(row);
    }
    Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>().into())
}

/// A held control vector after the checks of its object, in the order of gogi "Control vectors": a held object, its
/// media type, its bytes, and the fit with a model of `weight_hash`, `layers` layers and the residual width `width`.
/// `param` names the member, and `quoted` is the id as the client sent it. An object that the store does not hold is a
/// 404; each other fault is a 400.
pub fn held(
    param: &str,
    quoted: &str,
    id: &ContentId,
    weight_hash: &str,
    layers: u32,
    width: usize,
    lookup: &mut Lookup,
) -> Result<ControlVector, ServiceError> {
    let Some((mime, bytes)) = lookup(id)? else {
        return Err(ServiceError::NotHeld {
            param: param.to_owned(),
            message: format!(
                "nothing ran: the server does not hold the control vector {quoted}; put it, then send again"
            ),
        });
    };
    if mime != MEDIA_TYPE {
        return Err(Invalid::param(
            param,
            format!("nothing ran: {param} names an object of media type {mime}; a control vector is {MEDIA_TYPE}"),
        )
        .into());
    }
    let vector = ControlVector::parse(&bytes).map_err(|why| {
        Invalid::param(
            param,
            format!("nothing ran: {param} ({quoted}) is not a control vector: {why}"),
        )
    })?;
    vector
        .check_fits(weight_hash, layers, width)
        .map_err(|why| Invalid::param(param, format!("nothing ran: {param} ({quoted}) does not fit: {why}")))?;
    Ok(vector)
}

/// The write of a vector into an entry of the table (`Control::control_write`).
pub type Write<'a> = dyn FnMut(u32, &[u8]) -> Result<(), String> + 'a;

/// Why the worker did not bind the vectors of a job.
#[derive(Clone, Debug, PartialEq)]
pub enum BindError {
    /// Each entry that the job can take holds a vector that another queued or running job needs.
    Busy(String),
    /// The engine refused the write of a checked vector. It is a defect.
    Write(String),
}

/// The worker of the control table of one engine (engine-abi "Control vectors"). It keeps the content id that each
/// entry holds, and the count of the queued and running jobs that need each vector. It runs on the thread that owns
/// the engine, before the first pass of a job.
#[derive(Debug)]
pub struct ControlTable {
    limit: ControlLimit,
    /// The vector of each entry, and the time of its last use.
    entries: Vec<Option<(ContentId, u64)>>,
    /// The open of the engine that the entries belong to. A new open has an empty table.
    opens: u64,
    tick: u64,
    /// The count of the claims of the queued and running jobs that need each vector.
    needs: BTreeMap<ContentId, usize>,
}

/// The claim of a queued or running job on its vectors. The worker does not replace an entry that holds a claimed
/// vector. The drop ends the claim.
#[derive(Debug)]
pub struct Claim {
    table: Arc<Mutex<ControlTable>>,
    vectors: Vec<ContentId>,
}

impl Drop for Claim {
    fn drop(&mut self) {
        let mut table = self.table.lock().expect("the control table lock is poisoned");
        for vector in &self.vectors {
            let count = table
                .needs
                .get_mut(vector)
                .expect("a claimed vector has a count of needs");
            *count -= 1;
            if *count == 0 {
                table.needs.remove(vector);
            }
        }
    }
}

impl ControlTable {
    /// An empty table of an engine with this limit.
    pub fn new(limit: ControlLimit) -> Self {
        Self {
            limit,
            entries: vec![None; limit.tables as usize],
            opens: 0,
            tick: 0,
            needs: BTreeMap::new(),
        }
    }

    /// Claim the vectors of a job before it goes into the queue.
    pub fn claim(table: &Arc<Mutex<Self>>, vectors: &[ContentId]) -> Claim {
        let mut held = table.lock().expect("the control table lock is poisoned");
        for vector in vectors {
            *held.needs.entry(*vector).or_default() += 1;
        }
        Claim {
            table: Arc::clone(table),
            vectors: vectors.to_vec(),
        }
    }

    /// The vector of each entry, as the worker knows it.
    pub fn held(&self) -> Vec<Option<ContentId>> {
        self.entries
            .iter()
            .map(|entry| entry.map(|(vector, _)| vector))
            .collect()
    }

    /// Give the entry of each vector of a job, at the open `opens` of the engine. A vector that no entry holds goes
    /// into an empty entry, else into the entry of least recent use whose vector no claim needs. `write` writes the
    /// bytes into an entry (`Control::control_write`).
    pub fn bind(
        &mut self,
        opens: u64,
        vectors: &[(ContentId, Arc<[u8]>)],
        write: &mut Write,
    ) -> Result<Vec<u32>, BindError> {
        if opens != self.opens {
            // A new open of the engine allocated a new table, which holds no vector.
            self.entries = vec![None; self.limit.tables as usize];
            self.opens = opens;
        }
        let mut chosen: Vec<u32> = Vec::with_capacity(vectors.len());
        for (vector, bytes) in vectors {
            self.tick += 1;
            if let Some(index) = self
                .entries
                .iter()
                .position(|entry| entry.is_some_and(|(v, _)| v == *vector))
            {
                self.entries[index] = Some((*vector, self.tick));
                chosen.push(index as u32);
                continue;
            }
            let free = |index: usize| !chosen.contains(&(index as u32));
            let empty = (0..self.entries.len()).find(|index| free(*index) && self.entries[*index].is_none());
            let unneeded = || {
                (0..self.entries.len())
                    .filter(|index| free(*index))
                    .filter_map(|index| self.entries[index].map(|(v, used)| (index, v, used)))
                    .filter(|(_, v, _)| !self.needs.contains_key(v))
                    .min_by_key(|(index, _, used)| (*used, *index))
                    .map(|(index, ..)| index)
            };
            let Some(index) = empty.or_else(unneeded) else {
                return Err(BindError::Busy(format!(
                    "each of the {} entries of the control table holds a vector that another job needs; try again",
                    self.limit.tables
                )));
            };
            write(index as u32, bytes).map_err(BindError::Write)?;
            self.entries[index] = Some((*vector, self.tick));
            chosen.push(index as u32);
        }
        Ok(chosen)
    }
}

/// The items of a segment: the entry of each item's vector, its operation and its strength.
pub fn items(
    checked: &[(ContentId, ControlOp, f32)],
    vectors: &[(ContentId, Arc<[u8]>)],
    indices: &[u32],
) -> Vec<ControlItem> {
    checked
        .iter()
        .map(|(vector, op, strength)| {
            let at = vectors
                .iter()
                .position(|(v, _)| v == vector)
                .expect("each item names a vector of the list");
            ControlItem {
                index: indices[at],
                op: *op,
                strength: *strength,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use gogi_proto::control::{Control, ControlOp as Op};

    use super::*;

    const LIMIT: ControlLimit = ControlLimit {
        tables: 2,
        layers: 3,
        width: 2,
    };

    fn fit() -> Fit<'static> {
        Fit {
            weight_hash: "0xW",
            limit: LIMIT,
        }
    }

    fn vector(weight_hash: &str, layers: &[(u32, [f32; 2])]) -> Vec<u8> {
        ControlVector {
            model: "m".to_owned(),
            weight_hash: weight_hash.to_owned(),
            layers: layers.iter().map(|(l, v)| (*l, v.to_vec())).collect(),
        }
        .to_bytes()
    }

    /// The objects of a test, by content id.
    struct Objects(BTreeMap<ContentId, (String, Vec<u8>)>);

    impl Objects {
        fn put(&mut self, mime: &str, bytes: Vec<u8>) -> gogi_proto::ids::ContentId {
            let text = gogi_proto::content_id(&bytes);
            let id = ContentId::from_hex(text.trim_start_matches("blake3:")).unwrap();
            self.0.insert(id, (mime.to_owned(), bytes));
            gogi_proto::ids::ContentId::parse(text).unwrap()
        }

        fn check(&self, list: &[Control], at: &str) -> Result<Checked, ServiceError> {
            let mut lookup = |id: &ContentId| Ok(self.0.get(id).cloned());
            check(list, at, &fit(), &mut lookup)
        }
    }

    fn item(vector: &gogi_proto::ids::ContentId, op: Op, strength: f64) -> Control {
        Control {
            vector: vector.clone(),
            op,
            strength,
        }
    }

    fn param(error: ServiceError) -> String {
        match error {
            ServiceError::Invalid(invalid) => format!("400 {}", invalid.param.unwrap_or_default()),
            ServiceError::NotHeld { param, .. } => format!("404 {param}"),
            other => panic!("not a refusal: {other:?}"),
        }
    }

    fn values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    #[test]
    fn a_list_gives_its_entries_its_items_and_each_vector_once_with_zeros_for_a_layer_it_lacks() {
        let mut objects = Objects(BTreeMap::new());
        let a = objects.put(MEDIA_TYPE, vector("0xW", &[(0, [1.0, 2.0]), (2, [3.0, -4.0])]));
        let b = objects.put(MEDIA_TYPE, vector("0xW", &[(1, [0.5, 0.25])]));
        let list = [
            item(&a, Op::Add, 0.5),
            item(&b, Op::Clamp, 2.0),
            item(&a, Op::Ablate, 1e-3),
        ];
        let checked = objects.check(&list, "").unwrap();
        assert_eq!(checked.entries.len(), 3);
        assert_eq!(
            checked.entries[2].strength, 1e-3,
            "the recipe keeps the f64 that the client sent"
        );
        assert_eq!(checked.items[2].2, 1e-3f32);
        assert_eq!(checked.vectors.len(), 2, "a vector named twice is written once");
        assert_eq!(values(&checked.vectors[0].1), [1.0, 2.0, 0.0, 0.0, 3.0, -4.0]);
        assert_eq!(values(&checked.vectors[1].1), [0.0, 0.0, 0.5, 0.25, 0.0, 0.0]);
    }

    #[test]
    fn each_refusal_names_its_member_in_the_order_of_the_checks() {
        let mut objects = Objects(BTreeMap::new());
        let good = objects.put(MEDIA_TYPE, vector("0xW", &[(0, [1.0, 0.0])]));
        let text = objects.put("text/plain", b"a text".to_vec());
        let junk = objects.put(MEDIA_TYPE, b"not a vector".to_vec());
        let theirs = objects.put(MEDIA_TYPE, vector("0xOTHER", &[(0, [1.0, 0.0])]));
        let past = objects.put(MEDIA_TYPE, vector("0xW", &[(3, [1.0, 0.0])]));
        let wide = ControlVector {
            model: "m".to_owned(),
            weight_hash: "0xW".to_owned(),
            layers: [(0, vec![1.0, 2.0, 3.0])].into_iter().collect(),
        };
        let wide = objects.put(MEDIA_TYPE, wide.to_bytes());
        let unheld = gogi_proto::ids::ContentId::parse(gogi_proto::content_id(b"never put")).unwrap();
        let third = objects.put(MEDIA_TYPE, vector("0xW", &[(1, [1.0, 0.0])]));
        let other = objects.put(MEDIA_TYPE, vector("0xW", &[(2, [1.0, 0.0])]));
        let three = vec![
            item(&good, Op::Add, 1.0),
            item(&third, Op::Add, 1.0),
            item(&other, Op::Add, 1.0),
        ];
        for (what, list, want) in [
            (
                "not held",
                vec![item(&good, Op::Add, 1.0), item(&unheld, Op::Add, 1.0)],
                "404 control[1].vector",
            ),
            (
                "another media type",
                vec![item(&text, Op::Add, 1.0)],
                "400 control[0].vector",
            ),
            ("not a vector", vec![item(&junk, Op::Add, 1.0)], "400 control[0].vector"),
            (
                "other weights",
                vec![item(&theirs, Op::Add, 1.0)],
                "400 control[0].vector",
            ),
            (
                "a layer past the model",
                vec![item(&past, Op::Add, 1.0)],
                "400 control[0].vector",
            ),
            (
                "another width",
                vec![item(&wide, Op::Add, 1.0)],
                "400 control[0].vector",
            ),
            (
                "a strength past f32",
                vec![item(&good, Op::Add, 1e300)],
                "400 control[0].strength",
            ),
            ("three vectors for two entries", three, "400 control"),
        ] {
            assert_eq!(param(objects.check(&list, "").unwrap_err()), want, "{what}");
        }
        let error = objects
            .check(&[item(&unheld, Op::Add, 1.0)], "contexts[2].")
            .unwrap_err();
        assert_eq!(param(error), "404 contexts[2].control[0].vector");
        match objects.check(&[item(&unheld, Op::Add, 1.0)], "").unwrap_err() {
            ServiceError::NotHeld { message, .. } => assert!(message.contains(unheld.as_str()), "{message}"),
            other => panic!("{other:?}"),
        }
        match objects.check(&[item(&text, Op::Add, 1.0)], "").unwrap_err() {
            ServiceError::Invalid(invalid) => {
                assert!(invalid.message.contains("text/plain") && invalid.message.contains(MEDIA_TYPE))
            }
            other => panic!("{other:?}"),
        }
        // A strength past f32 is refused before the store is read, so an unheld vector with it gives the 400.
        let error = objects.check(&[item(&unheld, Op::Add, -1e39)], "").unwrap_err();
        assert_eq!(param(error), "400 control[0].strength");
        // Two vectors fit the table of two entries, also with an item that names a vector again.
        let two = [
            item(&good, Op::Add, 1.0),
            item(&third, Op::Add, 1.0),
            item(&good, Op::Clamp, 0.0),
        ];
        objects.check(&two, "").unwrap();
    }

    fn id(n: u8) -> ContentId {
        ContentId::from_hex(&format!("{n:02x}").repeat(32)).unwrap()
    }

    fn bytes(n: u8) -> Arc<[u8]> {
        vec![n; 4].into()
    }

    type Bound = (Result<Vec<u32>, BindError>, Vec<(u32, u8)>);

    /// Bind `vectors`, and give the entries with each write that the bind made.
    fn bind(table: &mut ControlTable, opens: u64, vectors: &[u8]) -> Bound {
        let list: Vec<(ContentId, Arc<[u8]>)> = vectors.iter().map(|n| (id(*n), bytes(*n))).collect();
        let mut writes = Vec::new();
        let result = table.bind(opens, &list, &mut |index, bytes| {
            writes.push((index, bytes[0]));
            Ok(())
        });
        (result, writes)
    }

    #[test]
    fn the_worker_writes_a_vector_once_and_replaces_the_entry_of_least_recent_use_that_no_job_needs() {
        let table = Arc::new(Mutex::new(ControlTable::new(LIMIT)));
        let mut held = table.lock().unwrap();
        assert_eq!(bind(&mut held, 1, &[1, 2]), (Ok(vec![0, 1]), vec![(0, 1), (1, 2)]));
        assert_eq!(
            bind(&mut held, 1, &[2, 1]),
            (Ok(vec![1, 0]), vec![]),
            "a held vector is not written again"
        );
        // The last bind used 2, then 1, so 2 has the least recent use.
        assert_eq!(bind(&mut held, 1, &[3]), (Ok(vec![1]), vec![(1, 3)]));
        assert_eq!(held.held(), [Some(id(1)), Some(id(3))]);
        drop(held);
        // A claim protects its vector: 1 has the least recent use, but a queued job needs it, so 3 goes.
        let claim = ControlTable::claim(&table, &[id(1)]);
        let mut held = table.lock().unwrap();
        assert_eq!(bind(&mut held, 1, &[4]), (Ok(vec![1]), vec![(1, 4)]));
        drop(held);
        let other = ControlTable::claim(&table, &[id(4)]);
        let mut held = table.lock().unwrap();
        match bind(&mut held, 1, &[5]) {
            (Err(BindError::Busy(message)), writes) => {
                assert!(message.contains("another job needs"), "{message}");
                assert!(writes.is_empty());
            }
            other => panic!("{other:?}"),
        }
        drop(held);
        drop(claim);
        let mut held = table.lock().unwrap();
        assert_eq!(
            bind(&mut held, 1, &[5]),
            (Ok(vec![0]), vec![(0, 5)]),
            "the end of a claim frees its entry"
        );
        // A new open of the engine has an empty table, so each vector is written again.
        assert_eq!(bind(&mut held, 2, &[4]), (Ok(vec![0]), vec![(0, 4)]));
        assert_eq!(held.held(), [Some(id(4)), None]);
        drop(held);
        drop(other);
        assert!(table.lock().unwrap().needs.is_empty(), "each claim ended");
    }

    #[test]
    fn a_job_does_not_replace_its_own_vectors_and_a_refused_write_is_a_write_error() {
        let mut table = ControlTable::new(LIMIT);
        assert_eq!(bind(&mut table, 1, &[1, 2]), (Ok(vec![0, 1]), vec![(0, 1), (1, 2)]));
        // Vector 1 has the least recent use after this bind takes it, so only the job itself protects entry 0.
        assert_eq!(bind(&mut table, 1, &[1, 3]), (Ok(vec![0, 1]), vec![(1, 3)]));
        let list = vec![(id(9), bytes(9))];
        let error = table.bind(1, &list, &mut |_, _| Err("the engine refused".to_owned()));
        assert_eq!(error, Err(BindError::Write("the engine refused".to_owned())));
        assert_eq!(
            table.held(),
            [Some(id(1)), Some(id(3))],
            "a refused write changes no entry"
        );
    }

    #[test]
    fn the_items_name_the_entry_of_their_vector() {
        let vectors = vec![(id(1), bytes(1)), (id(2), bytes(2))];
        let checked = [
            (id(2), ControlOp::Add, 1.0),
            (id(1), ControlOp::Clamp, 0.0),
            (id(2), ControlOp::Ablate, 0.5),
        ];
        let items = items(&checked, &vectors, &[7, 3]);
        let got: Vec<(u32, ControlOp, f32)> = items.iter().map(|i| (i.index, i.op, i.strength)).collect();
        assert_eq!(
            got,
            [
                (3, ControlOp::Add, 1.0),
                (7, ControlOp::Clamp, 0.0),
                (3, ControlOp::Ablate, 0.5)
            ]
        );
    }
}
