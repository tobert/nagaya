//! The council v2 routes over HTTP (spec 11, gogi `docs/council-api.md`), with `axum` on the control runtime.
//!
//! A [`Council`] holds the served models of the process. Each route parses its body, finds its served model by
//! `model`, and runs the blocking call of the [`Service`] on a call thread. The control runtime never runs a service
//! call (spec 3.2). Every response carries a `request-id` header, and every error body carries the same id (11.5).

mod cluster;
mod compose;
mod devices;
pub mod embeddings;
mod errors;
mod extensions;
mod forward;
pub mod image;
mod lab;
mod loads;
mod objects;
mod peers;
pub mod pending;
mod profile;
pub mod replies;
mod routes;
pub mod speech_out;
mod stream;
pub mod voice;

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::extract::{MatchedPath, Request};
use axum::http::HeaderValue;
use axum::middleware::Next;
use axum::response::Response;
use gogi_proto::identity::{Capability, ModelLimits, ModelStatus, ServedModel, ServerIdentity, ServerLimits};
use gogi_proto::ids::{REQUEST_ID_HEADER, TRACEPARENT_HEADER};
use tokio::sync::Semaphore;
use tracing::Instrument;

use crate::council::Service;
use crate::council::error::ServiceError;

pub use errors::Failure;
pub use routes::keep_gogi_places;

/// The most service calls that run at once. A request past this is a 429 `busy`.
pub const MAX_CALLS: usize = 64;

/// The most contexts on each side of a control-vector request (gogi `limits.control_vector_contexts`, spec 11.12,
/// Proposed). The job keeps the residual row of each context in host memory outside the ledger: for `LFM3-1.7B`, 24
/// layers of 2,048 `f32` values, 192 KiB. Two sides of 64 hold 24 MiB.
pub const CONTROL_VECTOR_CONTEXTS: u64 = 64;

/// The largest request body, in bytes. A larger body is a 413 `too_large`.
pub const MAX_BODY_BYTES: usize = 16 << 20;

/// The `http.route` of a request that matched no route.
pub const UNMATCHED: &str = "unmatched";

/// The tracestate request header of W3C Trace Context.
const TRACESTATE_HEADER: &str = "tracestate";

/// One served model: its service and the names that a request may use for it.
pub struct Served {
    /// The service of the served model.
    pub service: Service,
    /// Other names of the served model. A response names it by its `model`.
    pub aliases: Vec<String>,
    /// The device that the identity names, such as `fake-0`.
    pub device: Option<String>,
}

/// The id of one request (gogi "Request ids and tracing").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestId(pub String);

/// A new request id: the start time of the process and a count. Thus ids differ across requests and restarts.
fn request_id() -> RequestId {
    static START: OnceLock<u64> = OnceLock::new();
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let start = *START.get_or_init(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos() as u64)
    });
    RequestId(format!("req_{start:016x}_{}", NEXT.fetch_add(1, Ordering::SeqCst)))
}

/// The status of a served model whose engine has this state (spec 3.3, 11.1). A fatal engine takes no new request,
/// and the process stops (spec 8.5).
pub(crate) fn model_status(state: nagaya_telemetry::EngineState) -> ModelStatus {
    use nagaya_telemetry::EngineState;
    match state {
        EngineState::Loading => ModelStatus::Loading,
        EngineState::Ready => ModelStatus::Ready,
        EngineState::Draining | EngineState::Fatal => ModelStatus::Draining,
        EngineState::Unloaded => ModelStatus::Unloaded,
    }
}

/// The status of the process: it takes requests, or it drains (spec 10.3 step 1).
const READY: u8 = 0;
const DRAINING: u8 = 1;

/// The served models of a process, and the state that the routes share.
pub struct Council {
    models: Vec<Arc<Served>>,
    status: AtomicU8,
    calls: Arc<Semaphore>,
    /// The CPUs of a call thread: the pool set (spec 3.2). `None` pins nothing.
    call_cpus: Option<Vec<u32>>,
    /// The first internal error. The process stops when it sees one (spec 8.5, 11.5).
    failure: Mutex<Option<String>>,
    /// The speech services (spec 11.8).
    speech: Vec<Arc<crate::speech::SpeechService>>,
    /// The image services (spec 11.9).
    images: Vec<Arc<crate::image::ImageService>>,
    /// The embedding services (spec 11.10).
    embeddings: Vec<Arc<crate::embedding::EmbeddingService>>,
    /// The objects and keeps, when the profile names an `objects` budget (spec 6.15).
    objects: Option<Arc<Mutex<nagaya_store::objects::ObjectStore>>>,
    /// `object_bytes` and `keep_bytes` of the object store. They are fixed at its open, so identity takes no lock.
    object_limits: Option<(u64, u64)>,
    /// The pending ids of the outputs that replies make now (gogi "Outputs").
    pending: pending::PendingIds,
    /// The ways into the process for the load routes (spec 3.3b). A council with none has no load routes.
    loads: Option<crate::process::Ports>,
    /// The calls to the main loop for the profile routes (spec 3.3). A council with none has no profile routes.
    profile: Option<std::sync::mpsc::Sender<crate::reprofile::ProfileCall>>,
    /// The cluster from gossip (spec 11.1a). A council with none serves its own models only.
    cluster: Option<Arc<crate::cluster::Cluster>>,
    /// The model folder of each engine whose weights the lab routes read, by engine name (`docs/lab.md`).
    lab: std::collections::BTreeMap<String, std::path::PathBuf>,
    /// The J-Lens of each engine whose profile entry names one, by engine name (`docs/lab.md`, "The rules of the
    /// J-Lens"). The lab reads a lens at its first use.
    lenses: std::collections::BTreeMap<String, lab::LensSlot>,
}

impl Council {
    /// Make the council of a process. It can serve several text models. Their services share one handle to the specs
    /// and the pin book, so two services never write one file (spec 11.7, "Several text models in one process").
    pub fn new(models: Vec<Served>, call_cpus: Option<Vec<u32>>) -> Result<Self, String> {
        let mut names: Vec<&str> = models.iter().map(|served| served.service.model()).collect();
        names.sort_unstable();
        if let Some(pair) = names.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(format!(
                "two engines of the profile serve the model `{}`; give each served model its own name",
                pair[0]
            ));
        }
        Ok(Self {
            models: models.into_iter().map(Arc::new).collect(),
            status: AtomicU8::new(READY),
            calls: Arc::new(Semaphore::new(MAX_CALLS)),
            call_cpus,
            failure: Mutex::new(None),
            speech: Vec::new(),
            images: Vec::new(),
            embeddings: Vec::new(),
            objects: None,
            object_limits: None,
            pending: pending::PendingIds::default(),
            loads: None,
            profile: None,
            cluster: None,
            lab: std::collections::BTreeMap::new(),
            lenses: std::collections::BTreeMap::new(),
        })
    }

    /// Serve the lab routes for the engines of `folders`: the model folder of each one, by engine name. Only an
    /// engine of the kind `mk-lfm3-bmg` has one now (`docs/lab.md`, "The rules of `project`").
    pub fn with_lab(mut self, folders: std::collections::BTreeMap<String, std::path::PathBuf>) -> Self {
        self.lab = folders;
        self
    }

    /// Read the J-Lens of each engine of `files` from its path, by engine name, at its first use.
    pub fn with_lenses(mut self, files: std::collections::BTreeMap<String, std::path::PathBuf>) -> Self {
        self.lenses = files
            .into_iter()
            .map(|(engine, path)| (engine, lab::LensSlot::new(path)))
            .collect();
        self
    }

    /// Serve the cluster that gossip shows (spec 11.1a).
    pub fn with_cluster(mut self, cluster: Arc<crate::cluster::Cluster>) -> Self {
        self.cluster = Some(cluster);
        self
    }

    /// Serve these speech services as council served models (spec 11.8a).
    pub fn with_speech(mut self, services: Vec<crate::speech::SpeechService>) -> Self {
        self.speech = services.into_iter().map(Arc::new).collect();
        self
    }

    /// Serve these image services as council served models (spec 11.9a).
    pub fn with_images(mut self, services: Vec<crate::image::ImageService>) -> Self {
        self.images = services.into_iter().map(Arc::new).collect();
        self
    }

    /// Serve these embedding services as council served models (spec 11.10).
    pub fn with_embeddings(mut self, services: Vec<crate::embedding::EmbeddingService>) -> Self {
        self.embeddings = services.into_iter().map(Arc::new).collect();
        self
    }

    /// Serve objects and keeps from this store (spec 6.15, gogi "Objects and keeps").
    pub fn with_objects(self, store: nagaya_store::objects::ObjectStore) -> Self {
        self.with_object_store(Arc::new(Mutex::new(store)))
    }

    /// Serve objects and keeps from a store that the text services also read (spec 6.15, 11.11, 11.12).
    pub fn with_object_store(mut self, store: Arc<Mutex<nagaya_store::objects::ObjectStore>>) -> Self {
        {
            let held = store.lock().expect("the object store lock is poisoned");
            self.object_limits = Some((held.object_bytes(), held.keep_bytes()));
        }
        // A text service finds the control vectors and the media parts of its contexts in the objects.
        for served in &self.models {
            served.service.serve_objects(Arc::clone(&store));
        }
        self.objects = Some(store);
        self
    }

    /// Serve the load routes of the engines of this process (spec 3.3b).
    pub fn with_loads(mut self, ports: crate::process::Ports) -> Self {
        self.loads = Some(ports);
        self
    }

    /// Serve the profile routes through the main loop that receives these calls (spec 3.3).
    pub fn with_profile(mut self, calls: std::sync::mpsc::Sender<crate::reprofile::ProfileCall>) -> Self {
        self.profile = Some(calls);
        self
    }

    /// The cluster, when the council has one.
    pub(crate) fn cluster(&self) -> Option<&Arc<crate::cluster::Cluster>> {
        self.cluster.as_ref()
    }

    /// Where a request for the served model `name` runs (spec 11.1a, "The forward"). A model of this node runs here,
    /// unless a peer keeps its name (spec 2.3). A text route for a model of a peer runs here when a local family has
    /// the tokenizer hash and the template of the model.
    pub(crate) fn place(&self, name: &str, text: bool) -> forward::Place {
        let Some(cluster) = &self.cluster else {
            return forward::Place::Local;
        };
        let state = cluster.state();
        let local = self
            .served_by_engine()
            .into_values()
            .flatten()
            .find(|model| model.is_named(name));
        if let Some(model) = &local
            && state.keeper(&model.model).is_none()
        {
            return forward::Place::Local;
        }
        match state.holder(name) {
            Some(peer) if text && self.local_family(&peer.entry).is_some() => forward::Place::Local,
            Some(peer) => forward::Place::Peer(Box::new(peer.clone())),
            None if local.is_some() => forward::Place::Local,
            None => forward::Place::Unknown,
        }
    }

    /// The local text model whose family has the tokenizer hash and the template of `entry` (spec 11.1a).
    fn local_family(&self, entry: &ServedModel) -> Option<Arc<Served>> {
        self.models
            .iter()
            .find(|served| {
                let local = self.served_model(served);
                local.tokenizer_hash == entry.tokenizer_hash && local.template == entry.template
            })
            .cloned()
    }

    /// The served model of a text route, and the `model` that its response names (spec 11.4b, 11.1a). A model of a
    /// peer with a local family runs on that family, and a model of a peer that is not `ready` is a 503.
    pub(crate) fn route_text(&self, name: &str) -> Result<(Arc<Served>, String), Failure> {
        let local = match self.route(name) {
            Ok(served) => {
                let model = served.service.model().to_owned();
                return Ok((served, model));
            }
            Err(failure) => failure,
        };
        let Some(cluster) = &self.cluster else {
            return Err(local);
        };
        let state = cluster.state();
        let Some(peer) = state.holder(name) else {
            return Err(local);
        };
        let Some(served) = self.local_family(&peer.entry) else {
            return Err(local);
        };
        if peer.entry.status != ModelStatus::Ready {
            return Err(Failure::unavailable(&peer.entry));
        }
        Ok((served, peer.entry.model.clone()))
    }

    /// Run the job of a generation of a speech model or an image model, with its records (spec 11.13). The record
    /// of the context is held before the job, and the record with the reply after it, which `done.context` names.
    /// The records are files of the home, so a node with no home holds none and names none (spec 6.17).
    pub(crate) fn recorded(
        &self,
        context: &gogi_proto::Context,
        job: impl FnOnce() -> Result<gogi_proto::generation::GenerationDone, ServiceError>,
    ) -> Result<gogi_proto::generation::GenerationDone, ServiceError> {
        let Some(home) = self.models.first() else {
            return job();
        };
        home.service.hold_record(context)?;
        let mut done = job()?;
        home.service.name_reply(context, &mut done)?;
        Ok(done)
    }

    /// The calls of the profile routes, when the council has them.
    fn profile_calls(&self) -> Option<std::sync::mpsc::Sender<crate::reprofile::ProfileCall>> {
        self.profile.clone()
    }

    /// The ports of the load routes, when the council has them.
    fn loads(&self) -> Option<crate::process::Ports> {
        self.loads.clone()
    }

    /// The object store, when the process has one.
    pub fn objects(&self) -> Option<Arc<Mutex<nagaya_store::objects::ObjectStore>>> {
        self.objects.clone()
    }

    /// The pending ids of the outputs that replies make now.
    pub fn pending(&self) -> &pending::PendingIds {
        &self.pending
    }

    /// The status of a served model whose engine has this state. A drain makes every model `draining` (spec 3.3,
    /// 10.3 step 1).
    pub(crate) fn model_status(&self, state: nagaya_telemetry::EngineState) -> ModelStatus {
        match self.status.load(Ordering::SeqCst) {
            READY => model_status(state),
            _ => ModelStatus::Draining,
        }
    }

    /// The served text model of this process with the name `model`, not an alias.
    pub fn text_model(&self, model: &str) -> Option<Arc<Served>> {
        self.models
            .iter()
            .find(|served| served.service.model() == model)
            .cloned()
    }

    /// The speech service of a served speech model.
    pub fn speech_service(&self, model: &str) -> Option<Arc<crate::speech::SpeechService>> {
        self.speech.iter().find(|service| service.model() == model).cloned()
    }

    /// The image service of a served image model.
    pub fn image_service(&self, model: &str) -> Option<Arc<crate::image::ImageService>> {
        self.images.iter().find(|service| service.model() == model).cloned()
    }

    /// The embedding service of a served embedding model.
    pub fn embedding_service(&self, model: &str) -> Option<Arc<crate::embedding::EmbeddingService>> {
        self.embeddings.iter().find(|service| service.model() == model).cloned()
    }

    /// The served speech models.
    pub fn speech_models(&self) -> Vec<String> {
        self.speech.iter().map(|service| service.model().to_owned()).collect()
    }

    /// True when the server takes no new request (spec 10.3 step 1).
    pub fn draining(&self) -> bool {
        self.status.load(Ordering::SeqCst) != READY
    }

    /// Take no new request: every served model reports `draining` (spec 10.3 step 1).
    pub fn drain(&self) {
        self.status.store(DRAINING, Ordering::SeqCst);
    }

    /// The slots that no job of a served model uses, over all served models.
    pub fn free_slots(&self) -> usize {
        self.models.iter().map(|served| served.service.free_slots()).sum()
    }

    /// The first internal error of a route or of a served service, when one happened. The process then stops with
    /// exit 1 (spec 8.5). A service fails when a release of a ledger charge fails (spec 4.2).
    pub fn failure(&self) -> Option<String> {
        let own = self.failure.lock().expect("the failure lock is poisoned").clone();
        own.or_else(|| self.models.iter().find_map(|served| served.service.failure()))
    }

    fn fail(&self, message: &str) {
        let mut failure = self.failure.lock().expect("the failure lock is poisoned");
        if failure.is_none() {
            tracing::error!("a route failed: {message}; the process stops (spec 8.5)");
            *failure = Some(message.to_owned());
        }
    }

    /// The identity of the server: this node and each peer that gossip shows (gogi "Identity, limits, capabilities",
    /// spec 11.1a "The identity of the cluster").
    ///
    /// The models are those of this node, less each model whose name a peer keeps, and those of the peers, in byte
    /// order of `model`. The limits and capabilities are those of this node, except `pins`: it is declared when any
    /// live node declares it, with the smallest `pin_bytes` of those nodes.
    pub fn identity(&self) -> ServerIdentity {
        let mut identity = self.local_identity();
        if let Some(cluster) = &self.cluster {
            let state = cluster.state();
            identity.models.retain(|model| state.keeper(&model.model).is_none());
            identity.models.extend(state.models().map(|model| model.entry.clone()));
            let peer_pins = state
                .servers()
                .iter()
                .filter(|server| server.capabilities.contains(&Capability::Pins))
                .filter_map(|server| server.limits.pin_bytes);
            if let Some(smallest) = identity.limits.pin_bytes.into_iter().chain(peer_pins).min() {
                identity.limits.pin_bytes = Some(smallest);
                if !identity.capabilities.contains(&Capability::Pins) {
                    identity.capabilities.insert(0, Capability::Pins);
                }
            }
            // The text routes forward when no local family has the model (spec 11.4f, 11.1a).
            identity.extensions = if identity.models.is_empty() {
                Vec::new()
            } else {
                vec![extensions::PREFIX.to_owned()]
            };
        }
        identity.models.sort_by(|left, right| left.model.cmp(&right.model));
        identity
    }

    /// The identity of this node alone. Its limits and capabilities go to gossip in `server` (spec 2.3).
    pub fn local_identity(&self) -> ServerIdentity {
        // A speech service with a code book serves speech contexts, so it is a served model too (spec 11.8a).
        let models = self
            .models
            .iter()
            .map(|served| self.served_model(served))
            .chain(
                self.speech
                    .iter()
                    .filter(|service| service.serves_contexts())
                    .map(|service| voice::served_model(service, self.model_status(service.engine_state()))),
            )
            .chain(
                self.images
                    .iter()
                    .map(|service| image::served_model(service, self.model_status(service.engine_state()))),
            )
            .chain(
                self.embeddings
                    .iter()
                    .map(|service| embeddings::served_model(service, self.model_status(service.engine_state()))),
            )
            .collect();
        let defaults = crate::council::ServiceConfig::new("", "");
        let config = self.models.first().map_or(&defaults, |served| served.service.config());
        let pins = !self.models.is_empty();
        let objects = self.object_limits;
        ServerIdentity {
            models,
            limits: ServerLimits {
                case_bytes: config.case_bytes as u64,
                choice_options: config.spec_rules.choice_options as u64,
                questions_per_spec: Some(config.spec_rules.questions_per_spec as u64),
                pin_bytes: pins.then_some(config.pin_bytes),
                context_records: pins.then_some(config.context_records as u64),
                object_bytes: objects.map(|(object, _)| object),
                keep_bytes: objects.map(|(_, keep)| keep),
            },
            capabilities: [
                pins.then_some(Capability::Pins),
                Some(Capability::Classes),
                objects.map(|_| Capability::Objects),
            ]
            .into_iter()
            .flatten()
            .collect(),
            // The text routes need a family, so a process that serves no model lists no prefix (spec 11.4f).
            extensions: if pins {
                vec![extensions::PREFIX.to_owned()]
            } else {
                Vec::new()
            },
        }
    }

    /// The served models of each engine of this process, by engine name (spec 2.3, "What a node publishes for the
    /// cluster"). An engine that serves no model has no entry.
    pub fn served_by_engine(&self) -> std::collections::BTreeMap<String, Vec<ServedModel>> {
        let mut by_engine: std::collections::BTreeMap<String, Vec<ServedModel>> = std::collections::BTreeMap::new();
        let mut add = |engine: &str, model: ServedModel| by_engine.entry(engine.to_owned()).or_default().push(model);
        for served in &self.models {
            add(&served.service.config().engine, self.served_model(served));
        }
        for service in self.speech.iter().filter(|service| service.serves_contexts()) {
            let model = voice::served_model(service, self.model_status(service.engine_state()));
            add(&service.config().engine, model);
        }
        for service in &self.images {
            let model = image::served_model(service, self.model_status(service.engine_state()));
            add(&service.config().engine, model);
        }
        for service in &self.embeddings {
            let model = embeddings::served_model(service, self.model_status(service.engine_state()));
            add(&service.config().engine, model);
        }
        by_engine
    }

    fn served_model(&self, served: &Served) -> ServedModel {
        let service = &served.service;
        let identity = service.identity();
        let config = service.config();
        ServedModel {
            model: identity.model,
            aliases: served.aliases.clone(),
            status: self.model_status(service.engine_state()),
            weight_hash: identity.weight_hash,
            tokenizer_hash: identity.tokenizer_hash,
            template: identity.template,
            engine: identity.engine,
            device: served.device.clone(),
            // spec 11.11, "Text questions of the LFM3 family": a stop token of the family ends a written answer.
            text_stop: service.family().describes().then(|| "stop_token".to_owned()),
            // A family that reads media lists its types, and a model that speaks writes `audio/wav` beside its text
            // (spec 11.11). Its replies hold text.
            media: {
                let inputs = service.family().media_inputs();
                let outputs: Vec<String> = config.speech.iter().map(|_| speech_out::WAV.to_owned()).collect();
                (!inputs.is_empty() || !outputs.is_empty()).then_some(gogi_proto::identity::Media {
                    inputs,
                    outputs,
                    text_replies: true,
                })
            },
            limits: ModelLimits {
                control_vector_contexts: service.controls().then_some(CONTROL_VECTOR_CONTEXTS),
                context_tokens: config.context_tokens as u64,
                decision_context_tokens: config.decision_context_tokens as u64,
                snapshot_bytes: service.snapshot_bytes(),
                contexts_per_decision: config.contexts_per_decision as u64,
                default_timeout_ms: config.default_timeout_ms,
                text_tokens: service
                    .family()
                    .describes()
                    .then_some(config.spec_rules.text_tokens as u64),
                replies_per_generation: Some(replies::REPLIES_PER_GENERATION),
            },
            // Each family library renders an open message and checks its seams, so each served model continues a
            // reply (gogi "Continuations"). Riding decisions are not built, so no model declares `ride`.
            embeddings: None,
            layer_loops: None,
            capabilities: [
                Capability::Reads,
                Capability::Generate,
                Capability::Continuation,
                Capability::DryRun,
                Capability::LeaveOneOut,
                // The replies of a text served model hold text, so one generation writes several (spec 11.15).
                Capability::Replies,
            ]
            .into_iter()
            .chain(service.family().thinks().then_some(Capability::Thinking))
            .chain(service.family().describes().then_some(Capability::Describe))
            .chain(service.controls().then_some(Capability::ControlVectors))
            .collect(),
        }
    }

    /// Refuse what a context asks of a served model that does not declare it (gogi "Contexts"): `thinking: true`
    /// without `thinking`, a control list without `control_vectors` (spec 11.12), and a loop that the served model's
    /// `layer_loops` does not allow (gogi "Layer loops"). The 400 names the member, such as `thinking` or
    /// `loop.count`, under `contexts[i].` in a request with several contexts.
    fn check_thinking(model: &ServedModel, contexts: &[&gogi_proto::Context], several: bool) -> Result<(), Failure> {
        for (i, context) in contexts.iter().enumerate() {
            for (member, check) in [
                ("thinking", context.check_thinking(model)),
                ("control", context.check_control(model)),
                ("loop", context.check_loop(model)),
            ] {
                check.map_err(|e| {
                    let invalid = match several {
                        true => {
                            let param = e.param.as_deref().unwrap_or(member);
                            gogi_proto::Invalid::param(format!("contexts[{i}].{param}"), e.message)
                        }
                        false => e,
                    };
                    Failure::from(crate::council::error::ServiceError::Invalid(invalid))
                })?;
            }
        }
        Ok(())
    }

    /// The route of a request that reads contexts: the served model, after the check of `thinking`.
    fn route_contexts(
        &self,
        name: &str,
        contexts: &[&gogi_proto::Context],
        several: bool,
    ) -> Result<Arc<Served>, Failure> {
        let served = self.route(name)?;
        Self::check_thinking(&self.served_model(&served), contexts, several)?;
        Ok(served)
    }

    /// The served model that a request names, by its `model` or an alias. A name that no served model has is a 400
    /// that names `model`. A served model that is not `ready` is a 503 `model_unavailable`, also a speech or an image
    /// model.
    fn route(&self, name: &str) -> Result<Arc<Served>, Failure> {
        // A speech model takes no text route. When it is not ready, the 503 comes first, as for any served model.
        if let Some(service) = self.speech_service(name).filter(|service| service.serves_contexts()) {
            let model = voice::served_model(&service, self.model_status(service.engine_state()));
            if model.status != ModelStatus::Ready {
                return Err(Failure::unavailable(&model));
            }
        }
        if let Some((_, model)) = image::route(self, name)
            && model.status != ModelStatus::Ready
        {
            return Err(Failure::unavailable(&model));
        }
        // A model that only embeds takes no text route: a 503 when it is not ready, and else the 400 below.
        if let Some((_, model)) = embeddings::route(self, name)
            && model.status != ModelStatus::Ready
        {
            return Err(Failure::unavailable(&model));
        }
        let served = self
            .models
            .iter()
            .find(|served| served.service.model() == name || served.aliases.iter().any(|alias| alias == name))
            .ok_or_else(|| {
                Failure::from(ServiceError::Invalid(gogi_proto::Invalid::param(
                    "model",
                    format!("nothing ran: model {name} is not served here; identity lists the served models"),
                )))
            })?;
        let model = self.served_model(served);
        if model.status != ModelStatus::Ready {
            return Err(Failure::unavailable(&model));
        }
        Ok(Arc::clone(served))
    }

    /// The home: the first text service, which serves the routes of the specs, the pins and the snapshots. They belong
    /// to the server, not to a served model (gogi "Served models"). Each text service shares the same files, and the
    /// home reads the snapshots of each text engine (spec 11.7, "Several text models in one process").
    fn home(&self) -> Result<Arc<Served>, Failure> {
        self.models.first().cloned().ok_or_else(|| {
            Failure::from(ServiceError::Stopping)
                .with_message("nothing ran: this process serves no model, so it holds no spec, pin or snapshot")
        })
    }

    /// The failure of a service error. An internal error stops the process (spec 8.5).
    ///
    /// A job whose engine was not ready names only its engine. The 503 names the served model of that engine and its
    /// status (gogi "Errors").
    fn failed(&self, error: ServiceError) -> Failure {
        if let ServiceError::Internal(message) = &error {
            self.fail(message);
        }
        if let ServiceError::Unavailable { engine, state } = &error {
            let text = self
                .models
                .iter()
                .filter(|served| &served.service.config().engine == engine)
                .map(|served| self.served_model(served))
                .next();
            let model = text.or_else(|| {
                self.speech
                    .iter()
                    .find(|service| &service.config().engine == engine)
                    .map(|service| voice::served_model(service, self.model_status(*state)))
            });
            let model = model.or_else(|| {
                self.images
                    .iter()
                    .find(|service| &service.config().engine == engine)
                    .map(|service| image::served_model(service, self.model_status(*state)))
            });
            let model = model.or_else(|| {
                self.embeddings
                    .iter()
                    .find(|service| &service.config().engine == engine)
                    .map(|service| embeddings::served_model(service, self.model_status(*state)))
            });
            if let Some(mut model) = model {
                model.status = self.model_status(*state);
                return Failure::unavailable(&model);
            }
        }
        Failure::from(error)
    }

    /// Run a blocking call of a service on a call thread, and wait for its result. A panic in the call is an
    /// internal error.
    async fn call<T: Send + 'static>(
        &self,
        work: impl FnOnce() -> Result<T, ServiceError> + Send + 'static,
    ) -> Result<T, Failure> {
        let permit = self.permit()?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.spawn(move || {
            let _ = tx.send(guarded(work));
            drop(permit);
        })?;
        match rx.await {
            Ok(result) => result.map_err(|error| self.failed(error)),
            Err(_) => Err(self.failed(ServiceError::internal(
                "a call thread ended with no result; the log has its panic",
            ))),
        }
    }

    /// A place among the calls that run at once. With none free, the request is a 429 `busy`.
    fn permit(&self) -> Result<tokio::sync::OwnedSemaphorePermit, Failure> {
        Arc::clone(&self.calls).try_acquire_owned().map_err(|_| {
            Failure::from(ServiceError::Busy(format!(
                "nothing ran: {MAX_CALLS} requests run now, the most that this server runs; try again later"
            )))
        })
    }

    /// Start a call thread on the CPUs of the pool set.
    fn spawn(&self, work: impl FnOnce() + Send + 'static) -> Result<(), Failure> {
        let cpus = self.call_cpus.clone();
        std::thread::Builder::new()
            .name("nagaya-call".to_owned())
            .spawn(move || {
                if let Some(cpus) = cpus
                    && let Err(error) = crate::affinity::pin_current_thread(&cpus)
                {
                    // The thread ends before it runs the call. The route sees a call with no result, which is an
                    // internal error, and the process stops with exit 1. Only a control-runtime thread aborts
                    // (spec 3.2a).
                    panic!("cannot pin a call thread to CPUs {cpus:?}: {error}");
                }
                work();
            })
            .map(|_| ())
            .map_err(|error| self.failed(ServiceError::internal(format!("cannot start a call thread: {error}"))))
    }
}

/// Run `work`, and turn a panic into an internal error (spec 8.5).
fn guarded<T>(work: impl FnOnce() -> Result<T, ServiceError>) -> Result<T, ServiceError> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) {
        Ok(result) => result,
        Err(payload) => Err(ServiceError::internal(format!(
            "a service call panicked: {}",
            crate::exit::panic_text(payload.as_ref())
        ))),
    }
}

/// The router of the council routes.
pub fn router(council: Arc<Council>) -> Router {
    routes::routes()
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&council),
            forward::forward,
        ))
        .layer(axum::middleware::from_fn(observe))
        .with_state(council)
}

/// The router of a server: the council routes, and a layer that cuts every open request when `cut` turns true.
fn served_router(council: Arc<Council>, cut: tokio::sync::watch::Receiver<bool>) -> Router {
    routes::routes()
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&council),
            forward::forward,
        ))
        .layer(axum::middleware::from_fn_with_state(cut, cut_open_requests))
        .layer(axum::middleware::from_fn(observe))
        .with_state(council)
}

/// Wait until the server cuts its open requests. A sender that is gone also cuts: nothing holds the requests open.
async fn cut_signal(mut cut: tokio::sync::watch::Receiver<bool>) {
    let _ = cut.wait_for(|cut| *cut).await;
}

/// End a request when the server cuts. Aborting the server task does not end a connection, because axum runs each
/// connection as a task of its own. A handler that still runs gives a 503 `unavailable`. A response body that still
/// streams ends at once, so the generation behind it sees a closed stream (spec 11.7).
async fn cut_open_requests(
    axum::extract::State(cut): axum::extract::State<tokio::sync::watch::Receiver<bool>>,
    request: Request,
    next: Next,
) -> Response {
    use futures_util::StreamExt;

    let id = request
        .extensions()
        .get::<RequestId>()
        .cloned()
        .expect("the observe layer sets the request id");
    let response = tokio::select! {
        response = next.run(request) => response,
        () = cut_signal(cut.clone()) => {
            return Failure::from(ServiceError::Stopping)
                .with_message("nothing ran: the server is stopping and cut this request")
                .respond(&id);
        }
    };
    let (parts, body) = response.into_parts();
    let body = axum::body::Body::from_stream(body.into_data_stream().take_until(cut_signal(cut)));
    Response::from_parts(parts, body)
}

/// Give each request its id, its span and its duration (spec 9.2, 9.4, 11.5).
///
/// The span is `<METHOD> <route template>`, and a child of the caller's span when the request has a `traceparent`.
/// The route template is a bounded set, so it is the `http.route` of the metric.
async fn observe(mut request: Request, next: Next) -> Response {
    let started = Instant::now();
    // A forward names the request id of its coordinator, so both nodes name one request (spec 11.1a).
    let id = forwarded_id(request.headers()).unwrap_or_else(request_id);
    let method = request.method().clone();
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map(|path| path.as_str().to_owned());
    let name = match &route {
        Some(route) => format!("{method} {route}"),
        None => method.to_string(),
    };
    let route = route.unwrap_or_else(|| UNMATCHED.to_owned());
    let span = tracing::info_span!(
        "http.server",
        otel.name = %name,
        otel.kind = "server",
        http.request.method = %method,
        http.route = %route,
        http.response.status_code = tracing::field::Empty,
        request_id = %id.0,
        otel.status_code = tracing::field::Empty,
    );
    let traceparent = header_text(request.headers(), TRACEPARENT_HEADER);
    let tracestate = header_text(request.headers(), TRACESTATE_HEADER);
    nagaya_telemetry::adopt_parent(&span, traceparent.as_deref(), tracestate.as_deref());
    request.extensions_mut().insert(id.clone());
    let mut response = next.run(request).instrument(span.clone()).await;
    let status = response.status().as_u16();
    span.record("http.response.status_code", status);
    if status >= 500 {
        span.record("otel.status_code", "ERROR");
    }
    nagaya_telemetry::instruments().http_request(&route, status, started.elapsed().as_secs_f64());
    response.headers_mut().insert(
        REQUEST_ID_HEADER,
        HeaderValue::from_str(&id.0).expect("a request id is printable ASCII"),
    );
    response
}

/// The request id that a forward carries, when the request has `nagaya-forwarded` and a valid `nagaya-request-id`.
/// A valid id is printable ASCII without spaces, 1 to 128 bytes (gogi "Request ids and tracing").
fn forwarded_id(headers: &axum::http::HeaderMap) -> Option<RequestId> {
    use crate::cluster::call::{FORWARDED_HEADER, REQUEST_ID_HEADER};
    if !headers.contains_key(FORWARDED_HEADER) {
        return None;
    }
    let text = headers.get(REQUEST_ID_HEADER)?.to_str().ok()?;
    let valid = (1..=128).contains(&text.len()) && text.bytes().all(|byte| byte.is_ascii_graphic());
    valid.then(|| RequestId(text.to_owned()))
}

/// The text of a header. A header that is not text is an invalid trace context, which the span records (spec 9.2).
fn header_text(headers: &axum::http::HeaderMap, name: &str) -> Option<String> {
    let value = headers.get(name)?;
    Some(value.to_str().unwrap_or("").to_owned())
}

/// The routes of a server: empty while the process starts, and the council routes after the attach (spec 10.1).
type Attached = Arc<OnceLock<Router>>;

/// Serve a request with the council routes, or give a 503 `unavailable` while the process starts (spec 10.1).
async fn dispatch(
    axum::extract::State((attached, starting)): axum::extract::State<(Attached, Router)>,
    request: Request,
) -> Response {
    use tower::ServiceExt;

    let router = attached.get().cloned().unwrap_or(starting);
    match router.oneshot(request).await {
        Ok(response) => response,
        Err(never) => match never {},
    }
}

/// The answer to each request before the council routes attach.
async fn starting(request: Request) -> Response {
    let id = request
        .extensions()
        .get::<RequestId>()
        .cloned()
        .expect("the observe layer sets the request id");
    Failure::new(
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        gogi_proto::error::ErrorType::Unavailable,
        "nothing ran: the process is starting; try again when it is ready",
    )
    .respond(&id)
}

/// A running HTTP server.
pub struct Server {
    addr: SocketAddr,
    /// The council routes, after [`Server::attach`].
    attached: Attached,
    /// The receiver of `cut`, for the routes that attach later.
    cut_rx: tokio::sync::watch::Receiver<bool>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<std::io::Result<()>>,
    runtime: tokio::runtime::Handle,
    /// Turns true when the server cuts the requests that are still open.
    cut: tokio::sync::watch::Sender<bool>,
}

impl Server {
    /// Bind `bind` and serve the council routes on `runtime`. A bind that fails is an error.
    pub fn start(runtime: &tokio::runtime::Handle, bind: SocketAddr, council: Arc<Council>) -> std::io::Result<Self> {
        let server = Self::bind(runtime, bind)?;
        server.attach(council);
        Ok(server)
    }

    /// Bind `bind` and serve on `runtime` before the council routes exist. Until [`Server::attach`], each request
    /// gets a 503 `unavailable` (spec 10.1). A bind that fails is an error.
    pub fn bind(runtime: &tokio::runtime::Handle, bind: SocketAddr) -> std::io::Result<Self> {
        let listener = std::net::TcpListener::bind(bind)?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let (cut, cut_rx) = tokio::sync::watch::channel(false);
        let attached: Attached = Arc::new(OnceLock::new());
        let starting_router = Router::new()
            .fallback(starting)
            .layer(axum::middleware::from_fn(observe));
        let app = Router::new()
            .fallback(dispatch)
            .with_state((Arc::clone(&attached), starting_router));
        let task = runtime.spawn(async move {
            let listener = tokio::net::TcpListener::from_std(listener)?;
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
        });
        Ok(Self {
            addr,
            attached,
            cut_rx,
            stop: Some(stop),
            task,
            runtime: runtime.clone(),
            cut,
        })
    }

    /// Serve the council routes from now on. A second attach is a defect, so it panics (spec 8.5).
    pub fn attach(&self, council: Arc<Council>) {
        let routes = served_router(council, self.cut_rx.clone());
        if self.attached.set(routes).is_err() {
            panic!("the council routes attached twice to one HTTP server");
        }
    }

    /// The address that the server listens on.
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Stop the server: accept no connection, and wait at most `wait` for the open requests to end. After `wait`, abort
    /// the server task and cut every request that is still open. No connection is served after the budget.
    pub fn stop(mut self, wait: std::time::Duration) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        let mut task = self.task;
        let ended = self
            .runtime
            .block_on(async { tokio::time::timeout(wait, &mut task).await });
        match ended {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(error))) => tracing::error!("the HTTP server stopped with an error: {error}"),
            Ok(Err(error)) => tracing::error!("the HTTP server task failed: {error}"),
            Err(_) => {
                // The task owns the accept loop. Each connection is a task of its own, so the cut ends them.
                task.abort();
                self.cut.send_replace(true);
                // Wait for the abort, so that the listener is closed when this returns.
                let _ = self.runtime.block_on(task);
                tracing::warn!("the HTTP server had open requests after {wait:?}; they are cut");
            }
        }
    }
}
