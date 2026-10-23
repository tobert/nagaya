//! In-memory telemetry for tests (spec 9.5).
//!
//! A [`Capture`] makes this thread record into memory. A test runs the real call site, then reads the spans and
//! the metric points that the call site recorded. Other threads, and other tests, record nowhere.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use opentelemetry_sdk::metrics::InMemoryMetricExporter;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use opentelemetry_sdk::trace::InMemorySpanExporter;
pub use opentelemetry_sdk::trace::SpanData;
use tracing_subscriber::layer::SubscriberExt;

use crate::instruments::{ScopeGuard, scope};
use crate::setup::{Config, ResourceValue, Telemetry};

/// One data point of a metric.
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    /// The metric name.
    pub name: String,
    /// The unit.
    pub unit: String,
    /// The attributes, as text.
    pub attributes: BTreeMap<String, String>,
    /// A counter's sum, a gauge's last value, or a histogram's sum.
    pub value: f64,
    /// A histogram's count of records. It is 0 for a counter and a gauge.
    pub count: u64,
}

/// The dispatches of all captures in this process. A capture's dispatch must outlive its guard.
///
/// `tracing` rebuilds the callsite interest while it holds a lock, and it holds a temporary copy of each dispatch.
/// When the guard of a capture drops first, the rebuild drops the last copy. The drop of the tracer provider logs,
/// and the log asks for the same lock, so the thread waits for itself. A held copy stops this.
static RETAINED: Mutex<Vec<tracing::Dispatch>> = Mutex::new(Vec::new());

/// How many dispatches the captures of this process keep (see `RETAINED`).
pub fn retained_dispatches() -> usize {
    RETAINED.lock().unwrap().len()
}

/// The telemetry that one test reads.
pub struct Capture {
    telemetry: Telemetry,
    spans: InMemorySpanExporter,
    metrics: InMemoryMetricExporter,
    _dispatch: tracing::subscriber::DefaultGuard,
    _scope: ScopeGuard,
}

impl Capture {
    /// Start a capture on this thread, with the production sampler and a resource of `resource`.
    pub fn start(resource: Vec<(String, ResourceValue)>) -> Self {
        let config = Config::from_lookup(|_| None, resource);
        let spans = InMemorySpanExporter::default();
        let metrics = InMemoryMetricExporter::default();
        let telemetry = Telemetry::with_exporters(&config, spans.clone(), metrics.clone(), false);
        let subscriber = tracing_subscriber::registry().with(telemetry.trace_layer());
        let dispatch = tracing::Dispatch::new(subscriber);
        RETAINED.lock().unwrap().push(dispatch.clone());
        let dispatch = tracing::dispatcher::set_default(&dispatch);
        let scope = scope(telemetry.instruments());
        Self {
            telemetry,
            spans,
            metrics,
            _dispatch: dispatch,
            _scope: scope,
        }
    }

    /// The spans that ended so far.
    pub fn spans(&self) -> Vec<SpanData> {
        self.spans
            .get_finished_spans()
            .expect("the in-memory span exporter failed")
    }

    /// The instruments of this capture.
    pub fn instruments(&self) -> Arc<crate::Instruments> {
        self.telemetry.instruments()
    }

    /// The metric points as of now. The last export of each metric wins.
    pub fn points(&self) -> Vec<Point> {
        self.metrics.reset();
        self.telemetry.flush().expect("the flush failed");
        let finished = self
            .metrics
            .get_finished_metrics()
            .expect("the in-memory metric exporter failed");
        let mut points = Vec::new();
        let Some(last) = finished.last() else {
            return points;
        };
        for scope in last.scope_metrics() {
            for metric in scope.metrics() {
                collect(metric.name(), metric.unit(), metric.data(), &mut points);
            }
        }
        points
    }

    /// The points of one metric.
    pub fn metric(&self, name: &str) -> Vec<Point> {
        self.points().into_iter().filter(|point| point.name == name).collect()
    }

    /// The resource attributes of the exported metrics, as text.
    pub fn resource(&self) -> BTreeMap<String, String> {
        self.metrics.reset();
        self.telemetry.flush().expect("the flush failed");
        let finished = self
            .metrics
            .get_finished_metrics()
            .expect("the in-memory metric exporter failed");
        let mut attributes = BTreeMap::new();
        if let Some(last) = finished.last() {
            for (key, value) in last.resource().iter() {
                attributes.insert(key.to_string(), value.to_string());
            }
        }
        attributes
    }
}

fn attributes<'a>(items: impl Iterator<Item = &'a opentelemetry::KeyValue>) -> BTreeMap<String, String> {
    items
        .map(|item| (item.key.to_string(), item.value.to_string()))
        .collect()
}

fn collect(name: &str, unit: &str, data: &AggregatedMetrics, points: &mut Vec<Point>) {
    macro_rules! walk {
        ($data:expr) => {
            match $data {
                MetricData::Gauge(gauge) => {
                    for point in gauge.data_points() {
                        points.push(Point {
                            name: name.to_owned(),
                            unit: unit.to_owned(),
                            attributes: attributes(point.attributes()),
                            value: point.value() as f64,
                            count: 0,
                        });
                    }
                }
                MetricData::Sum(sum) => {
                    for point in sum.data_points() {
                        points.push(Point {
                            name: name.to_owned(),
                            unit: unit.to_owned(),
                            attributes: attributes(point.attributes()),
                            value: point.value() as f64,
                            count: 0,
                        });
                    }
                }
                MetricData::Histogram(histogram) => {
                    for point in histogram.data_points() {
                        points.push(Point {
                            name: name.to_owned(),
                            unit: unit.to_owned(),
                            attributes: attributes(point.attributes()),
                            value: point.sum() as f64,
                            count: point.count(),
                        });
                    }
                }
                MetricData::ExponentialHistogram(_) => panic!("the nagaya uses no exponential histogram"),
            }
        };
    }
    match data {
        AggregatedMetrics::F64(data) => walk!(data),
        AggregatedMetrics::U64(data) => walk!(data),
        AggregatedMetrics::I64(data) => walk!(data),
    }
}

#[derive(Clone, Default)]
struct Sink(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for Sink {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Sink {
    type Writer = Sink;

    fn make_writer(&'a self) -> Sink {
        self.clone()
    }
}

/// Run `run` with the stderr format of the nagaya on this thread, and give the lines that it wrote.
pub fn log_lines(run: impl FnOnce()) -> Vec<String> {
    let sink = Sink::default();
    let layer = tracing_subscriber::fmt::layer()
        .with_writer(sink.clone())
        .event_format(crate::StderrFormat);
    tracing::subscriber::with_default(tracing_subscriber::registry().with(layer), run);
    let text = String::from_utf8(sink.0.lock().unwrap().clone()).expect("the log is text");
    text.lines().map(str::to_owned).collect()
}
