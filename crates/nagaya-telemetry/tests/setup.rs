//! The activation rules of spec 9.1.

use std::collections::HashMap;
use std::io::Read;
use std::net::TcpListener;
use std::time::Duration;

use nagaya_telemetry::testing::Capture;
use nagaya_telemetry::{Config, ResourceValue, Telemetry, instruments};
use tracing_subscriber::layer::SubscriberExt;

fn config(vars: &[(&str, &str)]) -> Config {
    let map: HashMap<String, String> = vars
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect();
    Config::from_lookup(|name| map.get(name).cloned(), Vec::new())
}

#[test]
fn nothing_set_exports_nothing() {
    let config = config(&[]);
    assert_eq!(config.export_endpoint(), None);
    let telemetry = Telemetry::build(&config).expect("an empty setup builds");
    assert!(!telemetry.exports());
}

#[test]
fn an_endpoint_turns_export_on() {
    let config = config(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")]);
    assert_eq!(config.export_endpoint(), Some("http://localhost:4317"));
}

#[test]
fn an_empty_endpoint_does_not_turn_export_on() {
    let config = config(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "")]);
    assert_eq!(config.export_endpoint(), None);
}

#[test]
fn sdk_disabled_stops_export_even_with_an_endpoint() {
    for value in ["true", "TRUE", "True"] {
        let config = config(&[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            ("OTEL_SDK_DISABLED", value),
        ]);
        assert_eq!(config.export_endpoint(), None, "OTEL_SDK_DISABLED={value}");
    }
    let config = config(&[
        ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
        ("OTEL_SDK_DISABLED", "false"),
    ]);
    assert!(config.export_endpoint().is_some());
}

#[test]
fn the_service_name_is_nagaya_whatever_the_environment_says() {
    let config = config(&[("OTEL_SERVICE_NAME", "other")]);
    assert_eq!(config.service_name, "nagaya");
}

#[test]
fn a_file_directory_variable_is_listed_so_that_the_process_can_warn() {
    let config = config(&[("NAGAYA_OTEL_FILE_DIR", "/tmp/otel"), ("KAIJUTSU_OTEL_FILE_DIR", "")]);
    assert_eq!(config.file_dirs, ["NAGAYA_OTEL_FILE_DIR"]);
}

#[test]
fn the_resource_carries_the_service_name_the_host_and_the_hardware() {
    let resource = vec![
        ("host.name".to_owned(), ResourceValue::Text("gfx1151host".to_owned())),
        ("nagaya.device.count".to_owned(), ResourceValue::Count(1)),
    ];
    let capture = Capture::start(resource);
    instruments().store_lookup("qwen", nagaya_telemetry::LookupOutcome::Miss);
    let resource = capture.resource();
    assert_eq!(resource.get("service.name").map(String::as_str), Some("nagaya"));
    assert_eq!(resource.get("host.name").map(String::as_str), Some("gfx1151host"));
    assert_eq!(resource.get("nagaya.device.count").map(String::as_str), Some("1"));
}

/// Accept connections on `listener` on a new thread. Send the first bytes of each connection to the channel.
fn accept_all(listener: TcpListener) -> std::sync::mpsc::Receiver<Vec<u8>> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let sender = sender.clone();
            std::thread::spawn(move || {
                let mut buffer = vec![0u8; 24];
                if stream.read_exact(&mut buffer).is_ok() {
                    let _ = sender.send(buffer);
                }
                // Keep the connection open and silent until the exporter gives up.
                let mut rest = Vec::new();
                let _ = stream.read_to_end(&mut rest);
            });
        }
    });
    receiver
}

#[test]
fn an_endpoint_makes_the_span_and_metric_exporters_connect_to_it() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let mut config = config(&[("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint.as_str())]);
    config.export_timeout = Some(Duration::from_millis(500));
    let telemetry = Telemetry::build(&config).expect("the exporters build");
    assert!(telemetry.exports());

    let subscriber = tracing_subscriber::registry().with(telemetry.trace_layer());
    tracing::subscriber::with_default(subscriber, || {
        tracing::info_span!("nagaya.test").in_scope(|| {});
    });
    let _scope = nagaya_telemetry::scope(telemetry.instruments());
    instruments().store_lookup("qwen", nagaya_telemetry::LookupOutcome::Miss);

    // The exporters send the HTTP/2 preface of gRPC to the endpoint, one connection for each signal. The flush waits for
    // a reply that never comes, so it runs on its own thread.
    let connections = accept_all(listener);
    let flusher = std::thread::spawn(move || {
        let _ = telemetry.flush();
        telemetry
    });
    for index in 0..2 {
        let bytes = connections
            .recv_timeout(Duration::from_secs(20))
            .unwrap_or_else(|_| panic!("exporter {index} did not connect in 20 s"));
        assert_eq!(bytes, b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n");
    }
    drop(flusher.join().expect("the flush thread ends"));
}
