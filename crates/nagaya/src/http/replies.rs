//! Several replies from one generation (gogi "Several replies", spec 11.15).
//!
//! `gogi_proto::replies::split` gives one plain generation body for each reply. The route runs each body as a plain
//! generation, all at the same time, and `gogi_proto::replies::join` makes the response. Thus reply `i` is the reply
//! that the same request gives alone, and no rule of a generation has a second form here.

use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::Response;
use gogi_proto::error::ErrorType;
use gogi_proto::generation::{GenerationDone, GenerationRequest};
use gogi_proto::identity::Capability;
use gogi_proto::replies::{Split, join, param_in, split};
use serde_json::Value;

use super::errors::{Failure, parse};
use super::{Council, RequestId};
use crate::council::error::ServiceError;

/// The most replies of one generation: `limits.replies_per_generation` of each served model that declares `replies`.
pub const REPLIES_PER_GENERATION: u64 = 8;

/// The members that ask for several replies. A request with `forks` or `contexts` has `replies` too.
const MEMBERS: [&str; 3] = ["forks", "contexts", "replies"];

/// The requests of the replies of a generation body, or `None` for a plain generation. The contexts of `body` are
/// whole.
///
/// A body that is not JSON, or that names no model, is a plain generation: its decode gives the 400. A served model
/// that does not declare `replies` refuses the member with a 400 that names it.
pub(super) fn several(council: &Council, body: &[u8]) -> Result<Option<Split>, Failure> {
    let Ok(value) = serde_json::from_slice::<Value>(body) else {
        return Ok(None);
    };
    let Some(member) = MEMBERS.into_iter().find(|member| value.get(member).is_some()) else {
        return Ok(None);
    };
    let Some(name) = value.get("model").and_then(Value::as_str) else {
        return Ok(None);
    };
    let identity = council.identity();
    let model = identity.route(name).map_err(ServiceError::Invalid)?;
    if !model.has(Capability::Replies) {
        return Err(ServiceError::Invalid(gogi_proto::Invalid::param(
            member,
            format!(
                "nothing ran: model {} does not declare the capability replies, so a generation takes no {member}; \
                 send one generation for each reply",
                model.model
            ),
        ))
        .into());
    }
    let limit = model.limits.replies_per_generation.ok_or_else(|| {
        ServiceError::internal(format!(
            "model {} declares replies with no limit replies_per_generation",
            model.model
        ))
    })?;
    // A request with no seed gets the seed of reply 0 as a plain generation gets its seed.
    split(&value, limit, crate::council::generations::fresh_seed)
        .map_err(|invalid| ServiceError::Invalid(invalid).into())
}

/// Run each reply of a request as a plain generation, all at the same time, and join them.
///
/// Each reply is a call with its own call thread and its own job. A reply that finds no free slot is `busy` (spec
/// 11.3c). It waits for the replies of this request that run: when one of them ended, its slot is free, and the
/// route runs the waiting replies again (spec 11.15). When no reply of the request ended, the request has nothing
/// to wait for, and the `busy` stands. The error of the failed reply with the lowest index is the error of the
/// request.
pub(super) async fn generation(council: Arc<Council>, split: Split, id: RequestId) -> Response {
    let mut results: Vec<Option<Result<GenerationDone, Failure>>> = split.requests.iter().map(|_| None).collect();
    let mut waiting: Vec<usize> = (0..split.requests.len()).collect();
    while !waiting.is_empty() {
        let runs = waiting.iter().map(|&i| one(&council, &split.requests[i]));
        let round = futures_util::future::join_all(runs).await;
        let mut ended = false;
        let mut busy = Vec::new();
        for (&i, result) in waiting.iter().zip(round) {
            match &result {
                Ok(_) => ended = true,
                Err(failure) if failure.body.error.r#type == ErrorType::Busy => busy.push(i),
                Err(_) => {}
            }
            results[i] = Some(result);
        }
        if !ended {
            break;
        }
        waiting = busy;
    }
    let mut dones = Vec::with_capacity(results.len());
    for (i, result) in results.into_iter().enumerate() {
        match result.expect("each reply ran in the first round") {
            Ok(done) => dones.push(done),
            Err(mut failure) => {
                // An error of a context of `contexts` names its place in the list.
                if !split.forks {
                    let param = failure.body.error.param.take();
                    failure.body.error.param = param.map(|param| param_in(&param, i));
                }
                return failure.respond(&id);
            }
        }
    }
    count_the_feed(&split, &mut dones);
    super::routes::json(StatusCode::OK, &join(&split, &dones))
}

/// One reply: a plain generation of a text served model, with no stream and no output.
async fn one(council: &Council, body: &Value) -> Result<GenerationDone, Failure> {
    let bytes = serde_json::to_vec(body).expect("a JSON value is JSON");
    let request: GenerationRequest = parse(&bytes, "generation")?;
    let served = council.route_contexts(&request.model, &[&request.context], false)?;
    council.call(move || served.service.generate(&request, None)).await
}

/// Give reply 0 of `forks` the `kept` and the `fed` of the fork that fed the most tokens.
///
/// `join` counts the context of `forks` once, from reply 0. The forks start together, so the fork that feeds the
/// prompt is not always reply 0. The other forks kept what that fork fed. `join` reads no other count of the
/// context, and a reply of `forks` reports none.
fn count_the_feed(split: &Split, dones: &mut [GenerationDone]) {
    if !split.forks {
        return;
    }
    let Some(most) = dones.iter().map(|done| done.usage).max_by_key(|usage| usage.fed) else {
        return;
    };
    dones[0].usage.kept = most.kept;
    dones[0].usage.fed = most.fed;
}

#[cfg(test)]
mod tests {
    use gogi_proto::context::Usage;
    use gogi_proto::generation::Finish;
    use gogi_proto::replies::Wanted;

    use super::*;

    fn done(fed: u64) -> GenerationDone {
        GenerationDone {
            model: "m".into(),
            content: "text".into(),
            reasoning: String::new(),
            tool_calls: vec![],
            finish: Finish::EndTurn,
            usage: Usage {
                prompt: 10,
                kept: 10 - fed,
                fed,
                completion: 2,
                reasoning: 0,
                media: None,
            },
            head: None,
            context: None,
            snapshots: vec![],
            stats: None,
            decisions: vec![],
            seed: Some(7),
            phase: None,
            outputs: vec![],
        }
    }

    /// Falsified by a `count_the_feed` that does nothing: the request reports a feed of 0 tokens.
    #[test]
    fn the_usage_of_forks_counts_the_feed_of_the_fork_that_fed_the_prompt() {
        let forks = Split {
            requests: vec![],
            wanted: Wanted::Full,
            forks: true,
        };
        let mut dones = [done(0), done(0), done(10)];
        count_the_feed(&forks, &mut dones);
        let joined = join(&forks, &dones);
        assert_eq!(
            joined["usage"],
            serde_json::json!({"prompt": 10, "kept": 0, "fed": 10, "completion": 6, "reasoning": 0})
        );
    }

    /// Each reply of `contexts` has its own counts, so the counts stay as they are.
    #[test]
    fn the_usage_of_contexts_stays_as_each_reply_counted_it() {
        let contexts = Split {
            requests: vec![],
            wanted: Wanted::Full,
            forks: false,
        };
        let mut dones = [done(0), done(10)];
        count_the_feed(&contexts, &mut dones);
        assert_eq!((dones[0].usage.fed, dones[0].usage.kept), (0, 10));
    }
}
