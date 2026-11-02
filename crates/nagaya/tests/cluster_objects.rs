//! Objects in the cluster (spec 6.15, "Objects in the cluster"): a read on a miss asks each live peer, and the answer of
//! the peer that holds the object comes back as it is. The coordinator keeps no copy.

mod peer_support;

use peer_support::{Has, Node, link, object};

const BYTES: &[u8] = b"the bytes of an image that only one node holds";

/// Three nodes: `host-a` takes the requests, `host-b` holds nothing, and `host-c` holds the object.
fn three() -> (Node, Node, Node, String, String) {
    let a = Node::start("host-a", Has::Objects);
    let b = Node::start("host-b", Has::Objects);
    let c = Node::start("host-c", Has::Objects);
    link(&[&a, &b, &c]);
    let (path, id) = object(BYTES);
    let put = c.send("PUT", &path, BYTES, &[("content-type", "image/png")]);
    assert_eq!(put.status, 201, "{}", put.text());
    (a, b, c, path, id)
}

#[test]
fn a_get_of_an_object_that_a_peer_holds_gives_its_answer_and_keeps_no_copy() {
    let (a, b, _c, path, _) = three();
    let reply = a.send("GET", &path, &[], &[]);
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(reply.body, BYTES);
    assert_eq!(reply.header("content-type"), Some("image/png"));
    assert_eq!(reply.header("accept-ranges"), Some("bytes"));
    // A request that a peer sent is answered from the node alone, so it shows that the coordinator kept no copy.
    for node in [&a, &b] {
        let own = node.send("GET", &path, &[], &[("nagaya-forwarded", "test")]);
        assert_eq!(own.status, 404, "{} holds a copy: {}", node.name, own.text());
    }
}

#[test]
fn a_range_and_a_head_go_to_the_holder() {
    let (a, _b, _c, path, _) = three();
    let part = a.send("GET", &path, &[], &[("range", "bytes=4-8")]);
    assert_eq!(part.status, 206, "{}", part.text());
    assert_eq!(part.body, &BYTES[4..=8]);
    assert_eq!(
        part.header("content-range"),
        Some(format!("bytes 4-8/{}", BYTES.len()).as_str())
    );
    let past = a.send("GET", &path, &[], &[("range", "bytes=999-")]);
    assert_eq!(past.status, 416, "{}", past.text());
    assert_eq!(
        past.header("content-range"),
        Some(format!("bytes */{}", BYTES.len()).as_str())
    );
    let head = a.send("HEAD", &path, &[], &[]);
    assert_eq!(head.status, 200);
    assert_eq!(head.header("content-length"), Some(BYTES.len().to_string().as_str()));
    assert!(head.body.is_empty());
}

#[test]
fn an_object_that_no_node_holds_is_a_404_of_the_coordinator() {
    let (a, _b, _c, _, _) = three();
    let (path, id) = object(b"bytes that nobody put");
    let reply = a.send("GET", &path, &[], &[]);
    assert_eq!(reply.status, 404, "{}", reply.text());
    assert!(reply.text().contains(&id), "{}", reply.text());
}

#[test]
fn a_node_with_no_store_asks_its_peers_too() {
    let a = Node::start("host-a", Has::Model);
    let c = Node::start("host-c", Has::Objects);
    link(&[&a, &c]);
    let (path, _) = object(BYTES);
    assert_eq!(
        c.send("PUT", &path, BYTES, &[("content-type", "image/png")]).status,
        201
    );
    let reply = a.send("GET", &path, &[], &[]);
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(reply.body, BYTES);
}

#[test]
fn a_pending_id_of_a_peer_streams_and_its_seal_names_a_path_with_no_host() {
    let (a, _b, c, _, id) = three();
    let (pending, output) = c.council.pending().start("image/png");
    output.push(b"part one ");
    let path = format!("/council/v2/objects/{pending}");
    // The output is sealed while the coordinator reads it, so the read ends.
    let sealer = std::thread::spawn({
        let council = std::sync::Arc::clone(&c.council);
        let pending = pending.clone();
        let id = id.clone();
        move || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            let (_, output) = match council.pending().lookup(&pending) {
                nagaya::http::pending::Lookup::Making(output) => ((), output),
                other => panic!("the output is not pending: {other:?}"),
            };
            output.push(b"part two");
            council.pending().seal(&pending, &id);
        }
    });
    let reply = a.send("GET", &path, &[], &[]);
    sealer.join().unwrap();
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(reply.body, b"part one part two");
    let sealed = a.send("GET", &path, &[], &[]);
    assert_eq!(sealed.status, 308, "{}", sealed.text());
    let location = sealed.header("location").unwrap().to_owned();
    assert_eq!(location, format!("/council/v2/objects/{id}"));
    // The client follows the path on the coordinator, which reads the content id from its peer.
    let followed = a.send("GET", &location, &[], &[]);
    assert_eq!(followed.status, 200, "{}", followed.text());
}
