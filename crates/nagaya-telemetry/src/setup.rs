//! The setup of the process: stderr logs always, OTLP traces and metrics on request (spec 9.1).

use std::sync::Arc;
use std::time::Duration;

use opentelemetry::KeyValue;
use opentelemetry::metrics::MeterProvider;
use opentelemetry::trace::TracerProvider;
use opentelemetry_otlp::{MetricExporter, SpanExporter, WithExportConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::OTelSdkError;
use opentelemetry_sdk::metrics::exporter::PushMetricExporter;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::{SdkTracer, SdkTracerProvider};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

use crate::format::StderrFormat;
use crate::instruments::{self, Instruments, METER_NAME};
use crate::sampler::NagayaSampler;

/// The environment variable that names the collector (spec 9.1).
pub const ENDPOINT_ENV: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";

/// The environment variable that stops all export (spec 9.1).
pub const DISABLED_ENV: &str = "OTEL_SDK_DISABLED";

/// The environment variables of file export. The nagaya does not build file export yet (spec 9.1).
pub const FILE_DIR_ENVS: [&str; 2] = ["NAGAYA_OTEL_FILE_DIR", "KAIJUTSU_OTEL_FILE_DIR"];

/// The value of a resource attribute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResourceValue {
    /// A text.
    Text(String),
    /// A count or a size.
    Count(i64),
}

/// What the setup reads. The process builds it from the environment and the inventory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// The `service.name`. The nagaya uses `nagaya` (spec 9.1).
    pub service_name: String,
    /// True when `OTEL_SDK_DISABLED` is `true`. Then nothing exports.
    pub disabled: bool,
    /// The OTLP gRPC endpoint, when `OTEL_EXPORTER_OTLP_ENDPOINT` is set and not empty.
    pub endpoint: Option<String>,
    /// The file-export directory variables that are set. The setup warns about them.
    pub file_dirs: Vec<String>,
    /// The attributes of the resource: `host.name` and the hardware from the inventory (spec 9.1).
    pub resource: Vec<(String, ResourceValue)>,
    /// How long one export waits for the collector. `None` keeps the exporter's own rule: `OTEL_EXPORTER_OTLP_TIMEOUT`,
    /// then 10 s. A test sets a short value.
    pub export_timeout: Option<Duration>,
}

impl Config {
    /// Read the configuration through `get`. A test passes a map; the process passes the environment.
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>, resource: Vec<(String, ResourceValue)>) -> Self {
        let disabled = get(DISABLED_ENV).is_some_and(|value| value.eq_ignore_ascii_case("true"));
        let endpoint = get(ENDPOINT_ENV).filter(|value| !value.is_empty());
        let file_dirs = FILE_DIR_ENVS
            .iter()
            .filter(|name| get(name).is_some_and(|value| !value.is_empty()))
            .map(|name| (*name).to_owned())
            .collect();
        Self {
            service_name: "nagaya".to_owned(),
            disabled,
            endpoint,
            file_dirs,
            resource,
            export_timeout: None,
        }
    }

    /// Read the configuration from the environment of the process.
    pub fn from_env(resource: Vec<(String, ResourceValue)>) -> Self {
        Self::from_lookup(|name| std::env::var(name).ok(), resource)
    }

    /// The endpoint to export to, or `None` when export is off.
    pub fn export_endpoint(&self) -> Option<&str> {
        if self.disabled { None } else { self.endpoint.as_deref() }
    }

    fn sdk_resource(&self) -> Resource {
        let attributes = self.resource.iter().map(|(key, value)| match value {
            ResourceValue::Text(text) => KeyValue::new(key.clone(), text.clone()),
            ResourceValue::Count(count) => KeyValue::new(key.clone(), *count),
        });
        // `builder_empty` skips the detectors, so `OTEL_SERVICE_NAME` cannot change the name (spec 9.1).
        Resource::builder_empty()
            .with_service_name(self.service_name.clone())
            .with_attributes(attributes)
            .build()
    }
}

/// A setup failure.
#[derive(Debug)]
pub enum SetupError {
    /// The OTLP exporter cannot start. `endpoint` is the address that the process tried.
    Exporter {
        /// The collector address.
        endpoint: String,
        /// What failed.
        message: String,
    },
    /// The Tokio runtime for the exporters cannot start.
    Runtime(String),
    /// `RUST_LOG` is not a filter.
    Filter(String),
    /// The process installed its telemetry twice.
    Installed(String),
}

impl std::fmt::Display for SetupError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SetupError::Exporter { endpoint, message } => write!(
                formatter,
                "cannot start the OTLP exporter for `{endpoint}`: {message}; check {ENDPOINT_ENV}"
            ),
            SetupError::Runtime(message) => write!(formatter, "cannot start the exporter runtime: {message}"),
            SetupError::Filter(message) => write!(formatter, "cannot use the log filter in RUST_LOG: {message}"),
            SetupError::Installed(message) => {
                write!(
                    formatter,
                    "the telemetry is installed already: {message}; install it once per process"
                )
            }
        }
    }
}

impl std::error::Error for SetupError {}

/// The providers of one process, and what keeps them alive.
///
/// Drop it to flush the batches. A process that ends with no destructor run (`_exit`, `exit`) must drop it first.
pub struct Telemetry {
    instruments: Arc<Instruments>,
    tracer_provider: Option<SdkTracerProvider>,
    meter_provider: Option<SdkMeterProvider>,
    config: Config,
    // The runtime drops last, so the exporters can finish their calls first.
    _runtime: Option<tokio::runtime::Runtime>,
}

/// How long a drop waits for the last export.
pub const FLUSH_TIMEOUT: Duration = Duration::from_secs(2);

impl Telemetry {
    /// Build the providers for `config`. This installs nothing. With export off it builds no provider.
    pub fn build(config: &Config) -> Result<Self, SetupError> {
        let Some(endpoint) = config.export_endpoint() else {
            return Ok(Self {
                instruments: Arc::new(Instruments::noop()),
                tracer_provider: None,
                meter_provider: None,
                config: config.clone(),
                _runtime: None,
            });
        };
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("nagaya-otlp")
            .enable_all()
            .build()
            .map_err(|error| SetupError::Runtime(error.to_string()))?;
        // The tonic channel needs the runtime for its set-up and for its connection task.
        let (spans, metrics) = {
            let _enter = runtime.enter();
            let failed = |error: opentelemetry_otlp::ExporterBuildError| SetupError::Exporter {
                endpoint: endpoint.to_owned(),
                message: error.to_string(),
            };
            let mut spans = SpanExporter::builder().with_tonic().with_endpoint(endpoint);
            let mut metrics = MetricExporter::builder().with_tonic().with_endpoint(endpoint);
            if let Some(timeout) = config.export_timeout {
                spans = spans.with_timeout(timeout);
                metrics = metrics.with_timeout(timeout);
            }
            let spans = spans.build().map_err(failed)?;
            let metrics = metrics.build().map_err(failed)?;
            (spans, metrics)
        };
        let _enter = runtime.enter();
        let mut built = Self::with_exporters(config, spans, metrics, true);
        built._runtime = Some(runtime);
        Ok(built)
    }

    /// Build the providers on these exporters. A test passes in-memory exporters.
    ///
    /// `batch` chooses the batch processor for spans. A test sets it off to read spans at once.
    pub fn with_exporters<S, M>(config: &Config, spans: S, metrics: M, batch: bool) -> Self
    where
        S: opentelemetry_sdk::trace::SpanExporter + 'static,
        M: PushMetricExporter,
    {
        let resource = config.sdk_resource();
        let tracer_builder = SdkTracerProvider::builder()
            .with_sampler(NagayaSampler)
            .with_resource(resource.clone());
        let tracer_provider = if batch {
            tracer_builder.with_batch_exporter(spans).build()
        } else {
            tracer_builder.with_simple_exporter(spans).build()
        };
        let reader = PeriodicReader::builder(metrics).build();
        let meter_provider = SdkMeterProvider::builder()
            .with_resource(resource)
            .with_reader(reader)
            .build();
        let instruments = Arc::new(Instruments::new(&meter_provider.meter(METER_NAME)));
        Self {
            instruments,
            tracer_provider: Some(tracer_provider),
            meter_provider: Some(meter_provider),
            config: config.clone(),
            _runtime: None,
        }
    }

    /// The instruments of these providers.
    pub fn instruments(&self) -> Arc<Instruments> {
        Arc::clone(&self.instruments)
    }

    /// The `tracing` layer that exports spans, when export is on.
    pub fn trace_layer<S>(&self) -> Option<tracing_opentelemetry::OpenTelemetryLayer<S, SdkTracer>>
    where
        S: tracing::Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
    {
        let provider = self.tracer_provider.as_ref()?;
        Some(tracing_opentelemetry::layer().with_tracer(provider.tracer(self.config.service_name.clone())))
    }

    /// Send the batches now. A failure of the spans does not stop the metrics, and the first failure is the result.
    pub fn flush(&self) -> Result<(), OTelSdkError> {
        let spans = self
            .tracer_provider
            .as_ref()
            .map_or(Ok(()), |provider| provider.force_flush());
        let metrics = self
            .meter_provider
            .as_ref()
            .map_or(Ok(()), |provider| provider.force_flush());
        spans.and(metrics)
    }

    /// True when the process exports to a collector.
    pub fn exports(&self) -> bool {
        self.tracer_provider.is_some()
    }
}

impl Drop for Telemetry {
    fn drop(&mut self) {
        if let Some(provider) = self.meter_provider.take()
            && let Err(error) = provider.shutdown_with_timeout(FLUSH_TIMEOUT)
        {
            eprintln!("the metrics did not flush: {error}");
        }
        if let Some(provider) = self.tracer_provider.take()
            && let Err(error) = provider.shutdown_with_timeout(FLUSH_TIMEOUT)
        {
            eprintln!("the spans did not flush: {error}");
        }
    }
}

/// Start the telemetry of the process: stderr logs always, and OTLP export when the configuration asks for it.
///
/// Call it once, before any work. The result must live as long as the process (spec 9.1).
pub fn init(config: &Config) -> Result<Telemetry, SetupError> {
    let filter = match std::env::var("RUST_LOG") {
        Ok(text) if !text.is_empty() => {
            EnvFilter::try_new(&text).map_err(|error| SetupError::Filter(error.to_string()))?
        }
        _ => EnvFilter::new("info"),
    };
    let telemetry = Telemetry::build(config)?;
    let stderr = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .event_format(StderrFormat)
        .with_filter(filter);
    tracing_subscriber::registry()
        .with(stderr)
        .with(telemetry.trace_layer())
        .try_init()
        .map_err(|error| SetupError::Installed(error.to_string()))?;
    instruments::install(telemetry.instruments()).map_err(|error| SetupError::Installed(error.to_string()))?;
    for name in &config.file_dirs {
        tracing::warn!(
            variable = name.as_str(),
            "{name} is set, and the nagaya has no file export yet; set {ENDPOINT_ENV} to export"
        );
    }
    if config.disabled && config.endpoint.is_some() {
        tracing::info!("{DISABLED_ENV} is true: the nagaya exports nothing");
    }
    Ok(telemetry)
}
