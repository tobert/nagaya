//! The routes of objects on a council with an object store and no served model (spec 6.15, gogi "Objects and keeps").

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use nagaya::http::{Council, router};
use nagaya_store::objects::ObjectStore;
use tower::ServiceExt;

/// Send raw bytes with headers. Give the status and the body bytes.
fn send(
    council: &Arc<Council>,
    method: &str,
    path: &str,
    body: &[u8],
    headers: &[(&str, &str)],
) -> (StatusCode, Vec<u8>) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut request = Request::builder().method(method).uri(path);
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        let response = router(Arc::clone(council))
            .oneshot(request.body(Body::from(body.to_vec())).unwrap())
            .await
            .unwrap();
        let status = response.status();
        (
            status,
            axum::body::to_bytes(response.into_body(), 1 << 24)
                .await
                .unwrap()
                .to_vec(),
        )
    })
}

/// A put can evict an object after a read looked it up. The read then finds no file: a 404, not an internal error,
/// because an internal error stops the process (spec 8.5).
#[test]
fn a_read_of_an_object_whose_file_went_is_a_404_and_no_failure() {
    let directory = tempfile::tempdir().unwrap();
    let store = ObjectStore::open(&directory.path().join("objects"), 1 << 20).unwrap();
    let council = Arc::new(Council::new(Vec::new(), None).unwrap().with_objects(store));
    let bytes = b"a short sound".to_vec();
    let hex = blake3::hash(&bytes).to_hex().to_string();
    let path = format!("/council/v2/objects/blake3:{hex}");
    let (status, _) = send(&council, "PUT", &path, &bytes, &[("content-type", "audio/wav")]);
    assert_eq!(status, StatusCode::CREATED);
    std::fs::remove_file(directory.path().join("objects").join(&hex[..2]).join(&hex[2..])).unwrap();
    let (status, body) = send(&council, "GET", &path, &[], &[]);
    assert_eq!(status, StatusCode::NOT_FOUND, "{}", String::from_utf8_lossy(&body));
    assert!(council.failure().is_none(), "{:?}", council.failure());
}

fn council_with_store(directory: &std::path::Path) -> Arc<Council> {
    let store = ObjectStore::open(&directory.join("objects"), 1 << 20).unwrap();
    Arc::new(Council::new(Vec::new(), None).unwrap().with_objects(store))
}

#[test]
fn head_gives_the_headers_of_an_object_and_no_body() {
    let directory = tempfile::tempdir().unwrap();
    let council = council_with_store(directory.path());
    let bytes = b"seven b".to_vec();
    let path = format!("/council/v2/objects/blake3:{}", blake3::hash(&bytes).to_hex());
    assert_eq!(
        send(&council, "PUT", &path, &bytes, &[("content-type", "image/png")]).0,
        StatusCode::CREATED
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (status, length, body) = runtime.block_on(async {
        let request = Request::builder()
            .method("HEAD")
            .uri(&path)
            .body(Body::empty())
            .unwrap();
        let response = router(Arc::clone(&council)).oneshot(request).await.unwrap();
        let length = response.headers()["content-length"].to_str().unwrap().to_owned();
        (
            response.status(),
            length,
            axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap(),
        )
    });
    assert_eq!((status, length.as_str()), (StatusCode::OK, "7"));
    assert!(body.is_empty(), "{body:?}");
}

#[test]
fn bytes_that_hash_to_another_id_are_a_400_and_nothing_is_held() {
    let directory = tempfile::tempdir().unwrap();
    let council = council_with_store(directory.path());
    let other = format!("blake3:{}", blake3::hash(b"other").to_hex());
    let (status, body) = send(
        &council,
        "PUT",
        &format!("/council/v2/objects/{other}"),
        b"these",
        &[("content-type", "image/png")],
    );
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        (status, value["error"]["param"].as_str()),
        (StatusCode::BAD_REQUEST, Some("id")),
        "{value}"
    );
    assert_eq!(
        send(&council, "GET", &format!("/council/v2/objects/{other}"), &[], &[]).0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        std::fs::read_dir(directory.path().join("objects/staging"))
            .unwrap()
            .count(),
        0,
        "no staged bytes stay"
    );
}

#[test]
fn a_put_while_the_process_stops_is_a_503() {
    let directory = tempfile::tempdir().unwrap();
    let council = council_with_store(directory.path());
    council.drain();
    let bytes = b"late".to_vec();
    let path = format!("/council/v2/objects/blake3:{}", blake3::hash(&bytes).to_hex());
    assert_eq!(
        send(&council, "PUT", &path, &bytes, &[("content-type", "image/png")]).0,
        StatusCode::SERVICE_UNAVAILABLE
    );
}
