//! The council v2 routes over HTTP (spec 11, gogi `docs/council-api.md`), with `axum` on the control runtime.
//!
//! A [`Council`] holds the served models of the process. Each route parses its body, finds its served model by
//! `model`, and runs the blocking call of the [`Service`] on a call thread. The control runtime never runs a service
//! call (spec 3.2). Every response carries a `request-id` header, and every error body carries the same id (11.5).

mod errors;
mod extensions;
mod routes;
mod stream;

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

/// The status of the served models of the process. Every served model has the status of the process.
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
}

impl Council {
    /// Make the council of a process. The process serves at most one model in milestone 12, because the specs and
    /// the pins of a service live in the store directory, and two services there would write one file.
    pub fn new(models: Vec<Served>, call_cpus: Option<Vec<u32>>) -> Result<Self, String> {
        if models.len() > 1 {
            let names: Vec<&str> = models.iter().map(|served| served.service.model()).collect();
            return Err(format!(
                "the profile has {} engines with a family library ({names:?}); the process serves at most one",
                models.len()
            ));
        }
        Ok(Self {
            models: models.into_iter().map(Arc::new).collect(),
            status: AtomicU8::new(READY),
            calls: Arc::new(Semaphore::new(MAX_CALLS)),
            call_cpus,
            failure: Mutex::new(None),
        })
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

    fn status(&self) -> ModelStatus {
        match self.status.load(Ordering::SeqCst) {
            READY => ModelStatus::Ready,
            _ => ModelStatus::Draining,
        }
    }

    /// The identity of the server (gogi "Identity, limits, capabilities").
    pub fn identity(&self) -> ServerIdentity {
        let models = self.models.iter().map(|served| self.served_model(served)).collect();
        let defaults = crate::council::ServiceConfig::new("", "");
        let config = self.models.first().map_or(&defaults, |served| served.service.config());
        let pins = !self.models.is_empty();
        ServerIdentity {
            models,
            limits: ServerLimits {
                case_bytes: config.case_bytes as u64,
                choice_options: config.spec_rules.choice_options as u64,
                questions_per_spec: Some(config.spec_rules.questions_per_spec as u64),
                pin_bytes: pins.then_some(config.pin_bytes),
            },
            capabilities: if pins {
                vec![Capability::Pins, Capability::Classes]
            } else {
                vec![Capability::Classes]
            },
            // The text routes need a family, so a process that serves no model lists no prefix (spec 11.4f).
            extensions: if pins {
                vec![extensions::PREFIX.to_owned()]
            } else {
                Vec::new()
            },
        }
    }

    fn served_model(&self, served: &Served) -> ServedModel {
        let service = &served.service;
        let identity = service.identity();
        let config = service.config();
        ServedModel {
            model: identity.model,
            aliases: served.aliases.clone(),
            status: self.status(),
            weight_hash: identity.weight_hash,
            tokenizer_hash: identity.tokenizer_hash,
            template: identity.template,
            engine: identity.engine,
            device: served.device.clone(),
            text_stop: None,
            limits: ModelLimits {
                context_tokens: config.context_tokens as u64,
                decision_context_tokens: config.decision_context_tokens as u64,
                snapshot_bytes: service.snapshot_bytes(),
                contexts_per_decision: config.contexts_per_decision as u64,
                default_timeout_ms: config.default_timeout_ms,
                text_tokens: None,
            },
            // Each family library renders an open message and checks its seams, so each served model continues a
            // reply (gogi "Continuations"). Riding decisions are not built, so no model declares `ride`.
            capabilities: vec![
                Capability::Reads,
                Capability::Generate,
                Capability::Continuation,
                Capability::DryRun,
                Capability::LeaveOneOut,
            ],
        }
    }

    /// The served model that a request names, by its `model` or an alias. A name that no served model has is a 400
    /// that names `model`. A served model that is not `ready` is a 503 `model_unavailable`.
    fn route(&self, name: &str) -> Result<Arc<Served>, Failure> {
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

    /// The service that holds the specs, the pins and the snapshots of the server. They belong to the server, not to
    /// a served model (gogi "Served models").
    fn home(&self) -> Result<Arc<Served>, Failure> {
        self.models.first().cloned().ok_or_else(|| {
            Failure::from(ServiceError::Stopping)
                .with_message("nothing ran: this process serves no model, so it holds no spec, pin or snapshot")
        })
    }

    /// The failure of a service error. An internal error stops the process (spec 8.5).
    fn failed(&self, error: ServiceError) -> Failure {
        if let ServiceError::Internal(message) = &error {
            self.fail(message);
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
        .layer(axum::middleware::from_fn(observe))
        .with_state(council)
}

/// The router of a server: the council routes, and a layer that cuts every open request when `cut` turns true.
fn served_router(council: Arc<Council>, cut: tokio::sync::watch::Receiver<bool>) -> Router {
    routes::routes()
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
    let id = request_id();
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

/// The text of a header. A header that is not text is an invalid trace context, which the span records (spec 9.2).
fn header_text(headers: &axum::http::HeaderMap, name: &str) -> Option<String> {
    let value = headers.get(name)?;
    Some(value.to_str().unwrap_or("").to_owned())
}

/// A running HTTP server.
pub struct Server {
    addr: SocketAddr,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<std::io::Result<()>>,
    runtime: tokio::runtime::Handle,
    /// Turns true when the server cuts the requests that are still open.
    cut: tokio::sync::watch::Sender<bool>,
}

impl Server {
    /// Bind `bind` and serve the council routes on `runtime`. A bind that fails is an error.
    pub fn start(runtime: &tokio::runtime::Handle, bind: SocketAddr, council: Arc<Council>) -> std::io::Result<Self> {
        let listener = std::net::TcpListener::bind(bind)?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let (cut, cut_rx) = tokio::sync::watch::channel(false);
        let task = runtime.spawn(async move {
            let listener = tokio::net::TcpListener::from_std(listener)?;
            axum::serve(listener, served_router(council, cut_rx))
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
        });
        Ok(Self {
            addr,
            stop: Some(stop),
            task,
            runtime: runtime.clone(),
            cut,
        })
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
