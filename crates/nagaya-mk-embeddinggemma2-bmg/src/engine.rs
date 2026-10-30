//! The EmbeddingGemma 2 engine on its card: the `Embedding` capability of engine-abi "Embedding" over the ring of
//! `model`.

use std::collections::BTreeMap;

use nagaya_abi::{Embedding, Engine, Identity, Limits, PassError, PassErrorKind, PassOut, Segment, Tap};
use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::safetensors::MappedSafeTensors;
use nagaya_mk_bmg::{EMBEDDINGGEMMA2_KERNEL, IGC_REVISION, KERNEL_ENTRY, OCLOC_VERSION};

use crate::device::{Device, OnCard};
use crate::model::{self, Bufs};
use crate::plan::limits;
use crate::spec::*;
use crate::weights::{self, Weights, f32_values};

/// The EmbeddingGemma 2 engine.
pub struct EmbeddingGemma2Engine {
    limits: Limits,
    identity: Identity,
    last_error: Option<String>,
    cluster: L0Cluster,
    ring: Ring,
    device: Device,
    weights: Weights,
    bufs: Bufs,
}

// The engine holds USM pointers of its card, and one thread at a time calls it.
unsafe impl Send for EmbeddingGemma2Engine {}

impl Drop for EmbeddingGemma2Engine {
    /// Free the weights, the buffers and the ring (engine-abi "`close` in Rust"). The card destroys its context after
    /// this call, when the cluster drops.
    fn drop(&mut self) {
        let card = &self.cluster.cards[0];
        self.ring.free(card);
        self.device.free_all(card);
    }
}

fn refused(message: String) -> PassError {
    PassError {
        kind: PassErrorKind::Refused,
        message,
    }
}

/// The BLAKE3 digest of the weights file, hashed in chunks of 256 MiB on threads.
fn checkpoint_digest(map: &MappedSafeTensors) -> String {
    let chunks: Vec<&[u8]> = map.as_bytes().chunks(256 << 20).collect();
    let hashes: Vec<blake3::Hash> = std::thread::scope(|scope| {
        let threads: Vec<_> = chunks
            .iter()
            .map(|chunk| scope.spawn(move || blake3::hash(chunk)))
            .collect();
        threads
            .into_iter()
            .map(|t| t.join().expect("a hash thread panicked"))
            .collect()
    });
    let mut all = blake3::Hasher::new();
    for hash in hashes {
        all.update(hash.as_bytes());
    }
    format!("blake3:{}", all.finalize().to_hex())
}

impl EmbeddingGemma2Engine {
    /// Open the card at the PCI address `pci`, and upload the text tower of the checkpoint of `config`.
    pub fn open(config: Config, pci: &str) -> Result<Self, String> {
        check_config(&config.model_dir)?;
        let path = config.model_dir.join("tokenizer.json");
        let tokenizer = std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        // The family gives the same hash (spec 11.10, "The family of EmbeddingGemma 2").
        let tokenizer_hash = format!("sha256:{}", sha256_hex(&tokenizer));
        let map = MappedSafeTensors::open(&config.model_dir.join(WEIGHTS_FILE))?;
        let checkpoint_digest = checkpoint_digest(&map);

        // Only the engine's card gets a context, so the open does not stall the work of another card.
        let cluster = L0Cluster::init_at(EMBEDDINGGEMMA2_KERNEL, KERNEL_ENTRY, &[pci.to_owned()])?;
        let card = &cluster.cards[0];
        let driver = cluster.driver_version();
        let mut device = Device::new(card);
        let weights = weights::load(
            &mut OnCard {
                device: &mut device,
                card,
            },
            &map.view()?,
        )?;
        let weights_bytes = device.bytes as u64;
        let bufs = Bufs::new(&mut |bytes| device.zeros(card, bytes), MAX_TOKENS);
        let ring = Ring::new(card, 1);

        let mut numbers = BTreeMap::new();
        numbers.insert(
            "kernel.megakernel_embeddinggemma2_bmg.bin".to_owned(),
            format!("blake3:{}", blake3::hash(EMBEDDINGGEMMA2_KERNEL).to_hex()),
        );
        numbers.insert("ocloc".to_owned(), OCLOC_VERSION.to_owned());
        numbers.insert("igc".to_owned(), IGC_REVISION.to_owned());
        numbers.insert("driver".to_owned(), driver.to_string());
        numbers.insert("canary".to_owned(), "not run".to_owned());
        let identity = Identity {
            model: MODEL.to_owned(),
            checkpoint_digest,
            family: "embeddinggemma2".to_owned(),
            tokenizer_hash,
            template: String::new(),
            numbers,
            plan_table: format!(
                "ocloc {OCLOC_VERSION} igc {IGC_REVISION} bmg-g31; level-zero driver {driver}; devices {:04x}:{:04x}",
                card.vendor_id, card.device_id
            ),
            bit_settings: Vec::new(),
            canary_hash: String::new(),
        };
        Ok(Self {
            limits: limits(weights_bytes),
            identity,
            last_error: None,
            cluster,
            ring,
            device,
            weights,
            bufs,
        })
    }

    fn card(&self) -> &B70Card {
        &self.cluster.cards[0]
    }

    fn run(&mut self, tasks: &[nagaya_mk_bmg::protocol::MkTaskDescriptor]) {
        self.ring.set(tasks);
        // SAFETY: each task points at the weights or at the buffers of this engine, sized for MAX_TOKENS tokens.
        unsafe { self.ring.run(&self.cluster.cards[0]) };
    }

    fn read_f32(&self, at: u64, n: usize) -> Vec<f32> {
        f32_values(&self.device.read(self.card(), at, n * 4))
    }

    /// Run the ring of `ids` up to the end of `layers` layers, with the tail when `tail`, and leave the buffers on the
    /// card. The caller checked the ids.
    fn forward(&mut self, ids: &[u32], layers: usize, tail: bool) {
        let bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.device.write(self.card(), self.bufs.ids, &bytes);
        let tasks = model::pass_until(&self.weights, &self.bufs, ids.len(), layers, tail);
        self.run(&tasks);
    }

    /// The probes of the oracle (8.3) for `ids`: the scaled embeddings, the per-layer inputs `[n][24][512]`, and the
    /// residual stream after each layer. Each probe runs the ring again from the start.
    pub fn probes(&mut self, ids: &[u32]) -> Result<Probes, String> {
        if let Some(problem) = model::ids_problem(ids) {
            return Err(problem);
        }
        let n = ids.len();
        self.forward(ids, 0, false);
        let embed = self.read_f32(self.bufs.x, n * HIDDEN);
        let ple = self.read_f32(self.bufs.ple, n * LAYERS * PER_LAYER);
        let mut layers = Vec::with_capacity(LAYERS);
        for l in 1..=LAYERS {
            self.forward(ids, l, false);
            layers.push(self.read_f32(self.bufs.x, n * HIDDEN));
        }
        self.forward(ids, LAYERS, true);
        let out = self.read_f32(self.bufs.out, n * DIMENSION);
        Ok(Probes {
            embed,
            ple,
            layers,
            out,
        })
    }

    fn fail(&mut self, error: PassError) -> PassError {
        self.last_error = Some(error.message.clone());
        error
    }
}

/// The hidden states of one input on the card, for the oracle.
pub struct Probes {
    /// The scaled embeddings: `[n][512]`.
    pub embed: Vec<f32>,
    /// The normed per-layer inputs: `[n][24][512]`.
    pub ple: Vec<f32>,
    /// The residual stream after each layer: 24 of `[n][512]`.
    pub layers: Vec<Vec<f32>>,
    /// The last output: `[n][768]`.
    pub out: Vec<f32>,
}

/// The lower-case hex of the SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

impl Embedding for EmbeddingGemma2Engine {
    fn embedding_max_tokens(&self) -> u32 {
        MAX_TOKENS as u32
    }

    fn embedding_dimension(&self) -> u32 {
        DIMENSION as u32
    }

    fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>, PassError> {
        if let Some(problem) = model::ids_problem(ids) {
            return Err(self.fail(refused(format!("nothing ran: {problem}"))));
        }
        self.forward(ids, LAYERS, true);
        let out = self.read_f32(self.bufs.out, ids.len() * DIMENSION);
        Ok(model::mean(&out, ids.len()))
    }
}

impl Engine for EmbeddingGemma2Engine {
    fn identity(&self) -> Identity {
        self.identity.clone()
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    /// An embedding engine takes no pass (engine-abi "Embedding").
    fn pass(&mut self, _segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
        Err(self.fail(refused("an embedding engine takes no pass; use embed".to_owned())))
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.clone()
    }

    fn embedding(&mut self) -> Option<&mut dyn Embedding> {
        Some(self)
    }
}
