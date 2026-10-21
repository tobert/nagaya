//! The stderr line format (spec 9.1).

use std::io::Write;
use std::sync::{Arc, Mutex};

use nagaya_telemetry::StderrFormat;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;

#[derive(Clone, Default)]
struct Sink(Arc<Mutex<Vec<u8>>>);

impl Write for Sink {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for Sink {
    type Writer = Sink;

    fn make_writer(&'a self) -> Sink {
        self.clone()
    }
}

fn lines(emit: impl FnOnce()) -> Vec<String> {
    let sink = Sink::default();
    let layer = tracing_subscriber::fmt::layer()
        .with_writer(sink.clone())
        .event_format(StderrFormat);
    tracing::subscriber::with_default(tracing_subscriber::registry().with(layer), emit);
    let text = String::from_utf8(sink.0.lock().unwrap().clone()).unwrap();
    text.lines().map(str::to_owned).collect()
}

#[test]
fn a_line_starts_with_its_message() {
    let lines = lines(|| tracing::info!("gossip listening on 127.0.0.1:7000"));
    assert_eq!(lines, ["gossip listening on 127.0.0.1:7000"]);
}

#[test]
fn fields_follow_the_message() {
    let lines = lines(|| tracing::info!(engines = 2, name = "qwen", "ready"));
    assert_eq!(lines, ["ready engines=2 name=qwen"]);
}

#[test]
fn a_warning_and_an_error_end_with_their_level() {
    let lines = lines(|| {
        tracing::warn!("slow disk");
        tracing::error!(code = 1, "the park failed");
    });
    assert_eq!(lines, ["slow disk level=warn", "the park failed code=1 level=error"]);
}

#[test]
fn a_span_adds_nothing_to_the_line() {
    let lines = lines(|| {
        tracing::info_span!("nagaya.start", engines = 1).in_scope(|| tracing::info!("opening"));
    });
    assert_eq!(lines, ["opening"]);
}
