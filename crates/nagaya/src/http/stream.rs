//! A generation as server-sent events (gogi "Streams").
//!
//! A call thread runs the generation, and a forward thread moves each token event to the response. The route waits
//! for the first event before it sends the status. Thus a request that fails before its first token gets a status
//! response, and a failure after the `200` is an `error` event. A closed stream drops the receiver of the response,
//! so the forward thread ends and the generation sees a closed stream at its next token (spec 11.3c).

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use gogi_proto::generation::{GenerationDone, GenerationRequest, StreamEvent, TokenEvent};
use tokio::sync::mpsc;

use super::{Council, Failure, RequestId, Served, guarded};
use crate::council::error::ServiceError;

/// The token events that wait for a slow client. A full queue holds the generation back. It does not drop events.
const QUEUE: usize = 256;

/// What the generation gives the response.
enum Item {
    Token(TokenEvent),
    End(Box<Result<GenerationDone, ServiceError>>),
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
    let (out, mut items) = mpsc::channel::<Item>(QUEUE);
    let started = council.spawn(move || {
        let (tokens, events) = std::sync::mpsc::channel::<TokenEvent>();
        let forward = out.clone();
        let forwarder = std::thread::Builder::new()
            .name("nagaya-stream".to_owned())
            .spawn(move || {
                for event in events {
                    if forward.blocking_send(Item::Token(event)).is_err() {
                        // The client went away. Dropping `events` closes the stream of the generation.
                        return;
                    }
                }
            });
        let result = match forwarder {
            Ok(forwarder) => {
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
