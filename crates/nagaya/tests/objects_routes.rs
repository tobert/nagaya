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

/// Read one response with a `content-length` from a stream. Give its head and its body, or `None` at an early end.
fn read_response(stream: &mut std::net::TcpStream) -> Option<(String, Vec<u8>)> {
    use std::io::Read;
    let mut data = Vec::new();
    let mut byte = [0u8; 1];
    while !data.ends_with(b"\r\n\r\n") {
        match stream.read(&mut byte) {
            Ok(1) => data.push(byte[0]),
            _ => return None,
        }
    }
    let head = String::from_utf8(data).unwrap().to_ascii_lowercase();
    let length: usize = head
        .lines()
        .find_map(|line| line.strip_prefix("content-length:"))
        .expect("the response has a content-length")
        .trim()
        .parse()
        .unwrap();
    let mut body = vec![0u8; length];
    stream.read_exact(&mut body).ok()?;
    Some((head, body))
}

/// A client such as ureq sends the head of a put, then its body, and keeps the connection for the next request. A
/// refused put must not leave a connection that the server closes in silence: the next request then sees an early
/// end. So after each refusal, the server either serves the next request or says `connection: close`.
#[test]
fn a_put_refused_before_its_body_leaves_a_connection_that_serves_or_says_close() {
    use std::io::Write;
    let directory = tempfile::tempdir().unwrap();
    let council = council_with_store(directory.path());
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let served = Arc::clone(&council);
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            axum::serve(listener, router(served)).await.unwrap();
        });
    });
    let body = vec![7u8; 300];
    let hex = blake3::hash(&body).to_hex().to_string();
    let refusals = [
        (format!("sha256:{hex}"), "content-type: application/octet-stream\r\n"),
        (
            format!("blake3:{}", &hex[..16]),
            "content-type: application/octet-stream\r\n",
        ),
        (format!("blake3:{hex}"), ""),
        (format!("blake3:{hex}"), "content-type: Text/Plain\r\n"),
    ];
    for round in 0..5 {
        for (id, content_type) in &refusals {
            let mut stream = std::net::TcpStream::connect(address).unwrap();
            stream.set_nodelay(true).unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(10)))
                .unwrap();
            let head = format!(
                "PUT /council/v2/objects/{id} HTTP/1.1\r\nhost: test\r\n{content_type}content-length: {}\r\n\r\n",
                body.len()
            );
            stream.write_all(head.as_bytes()).unwrap();
            // The body comes after the server answers, as it can from a client that writes the head first.
            std::thread::sleep(std::time::Duration::from_millis(30));
            let _ = stream.write_all(&body);
            let (head, _) = read_response(&mut stream).expect("the refused put gets a response");
            assert!(head.starts_with("http/1.1 400"), "round {round}, {id}: {head}");
            if head.contains("connection: close") {
                continue;
            }
            let next = format!("GET /council/v2/objects/blake3:{hex} HTTP/1.1\r\nhost: test\r\n\r\n");
            let sent = stream.write_all(next.as_bytes());
            let answer = sent.ok().and_then(|()| read_response(&mut stream));
            assert!(
                answer.is_some(),
                "round {round}, {id}: the server closed the connection after a 400 with no `connection: close`, so \
                 the next request on it sees an early end"
            );
        }
    }
}
