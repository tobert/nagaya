//! A generation as server-sent events (gogi "Streams").
//!
//! A call thread runs the generation, and a forward thread moves each token event to the response. The route waits
//! for the first event before it sends the status. Thus a request that fails before its first token gets a status
//! response, and a failure after the `200` is an `error` event. A closed stream drops the receiver of the response,
//! so the forward thread ends and the generation sees a closed stream at its next token (spec 11.3c).

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::mpsc::{SyncSender, sync_channel};

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use gogi_proto::generation::{GenerationDone, GenerationRequest, StreamEvent, TokenEvent};
use tokio::sync::mpsc;

use super::{Council, Failure, RequestId, Served, guarded};
use crate::council::error::ServiceError;

/// The token events that wait for a slow client. A full queue holds the generation back. It does not drop events.
/// The forward thread and the response channel hold two more, which are in flight.
const QUEUE: usize = 256;

/// What the generation gives the response.
enum Item {
    Token(TokenEvent),
    End(Box<Result<GenerationDone, ServiceError>>),
}

/// Start the forward thread. The call thread sends each token event to the returned sender. The forward thread moves
/// it to `out`, and ends when the client goes away or when the generation drops the sender.
fn relay(out: mpsc::Sender<Item>) -> std::io::Result<(SyncSender<TokenEvent>, std::thread::JoinHandle<()>)> {
    let (tokens, events) = sync_channel::<TokenEvent>(QUEUE);
    let forwarder = std::thread::Builder::new()
        .name("nagaya-stream".to_owned())
        .spawn(move || {
            for event in events {
                if out.blocking_send(Item::Token(event)).is_err() {
                    // The client went away. Dropping `events` closes the stream of the generation.
                    return;
                }
            }
        })?;
    Ok((tokens, forwarder))
}

/// Stream a generation.
pub async fn generation(
    council: Arc<Council>,
    served: Arc<Served>,
    request: GenerationRequest,
    id: RequestId,
) -> Response {
    let permit = match council.permit() {
        Ok(permit) => permit,
        Err(failure) => return failure.respond(&id),
    };
    let (out, mut items) = mpsc::channel::<Item>(1);
    let started = council.spawn(move || {
        let result = match relay(out.clone()) {
            Ok((tokens, forwarder)) => {
                let result = guarded(|| served.service.generate(&request, Some(tokens)));
                // The sender of the token events dropped with the generation, so the forward thread ends.
                let _ = forwarder.join();
                result
            }
            Err(error) => Err(ServiceError::internal(format!("cannot start a stream thread: {error}"))),
        };
        let _ = out.blocking_send(Item::End(Box::new(result)));
        drop(permit);
    });
    if let Err(failure) = started {
        return failure.respond(&id);
    }
    let first = match items.recv().await {
        Some(Item::End(result)) if result.is_err() => {
            let error = result.err().expect("the result is an error");
            return council.failed(error).respond(&id);
        }
        Some(first) => first,
        None => {
            return council
                .failed(ServiceError::internal("a generation ended with no result"))
                .respond(&id);
        }
    };
    let state = (Some(first), items, council, id, false);
    let stream = futures_util::stream::unfold(state, |(first, mut items, council, id, ended)| async move {
        if ended {
            return None;
        }
        let item = match first {
            Some(item) => Some(item),
            None => items.recv().await,
        };
        let (event, ended) = match item {
            Some(Item::Token(token)) => (StreamEvent::Token(token), false),
            Some(Item::End(result)) => match *result {
                Ok(done) => (StreamEvent::Done(Box::new(done)), true),
                Err(error) => (error_event(&council, error, &id), true),
            },
            None => (
                error_event(
                    &council,
                    ServiceError::internal("a generation ended with no result"),
                    &id,
                ),
                true,
            ),
        };
        let sse = Event::default().event(event.name()).data(event.data());
        Some((Ok::<Event, Infallible>(sse), (None, items, council, id, ended)))
    });
    Sse::new(stream).into_response()
}

/// The `error` event of a failure after the `200`: an error body with the request id (gogi "Streams").
fn error_event(council: &Council, error: ServiceError, id: &RequestId) -> StreamEvent {
    let failure: Failure = council.failed(error);
    StreamEvent::Error((*failure.body).with_request_id(id.0.clone()))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use gogi_proto::generation::Phase;

    use super::*;

    fn event(n: u64) -> TokenEvent {
        TokenEvent {
            id: n as u32,
            piece: format!("t{n}"),
            phase: Phase::Content,
            p: 1.0,
            h: 0.0,
            n,
            tps: None,
        }
    }

    /// A client that reads nothing holds the generation back, and the generation drops no event (spec 11.7).
    #[test]
    fn a_reader_that_stops_holds_the_generation_back_and_loses_no_event() {
        let total = QUEUE * 4;
        // The response channel is the one that the route makes.
        let (out, mut items) = mpsc::channel::<Item>(1);
        let (tokens, forwarder) = relay(out).unwrap();
        let sent = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&sent);
        let generation = std::thread::spawn(move || {
            for n in 0..total as u64 {
                tokens.send(event(n)).unwrap();
                counter.fetch_add(1, Ordering::SeqCst);
            }
        });
        // The client reads nothing. The generation must stop within the queue and the two events in flight.
        std::thread::sleep(Duration::from_millis(300));
        let waiting = sent.load(Ordering::SeqCst);
        assert!(
            !generation.is_finished(),
            "the generation ran on past a client that reads nothing"
        );
        assert!(
            waiting <= QUEUE + 2,
            "{waiting} events wait for a client that reads nothing"
        );
        // The client reads again. Each event arrives, in order.
        for n in 0..total as u64 {
            match items.blocking_recv() {
                Some(Item::Token(token)) => assert_eq!(token.n, n),
                _ => panic!("the stream lost the event {n}"),
            }
        }
        generation.join().unwrap();
        forwarder.join().unwrap();
        assert!(
            items.blocking_recv().is_none(),
            "the stream has an event after the last one"
        );
    }
}
