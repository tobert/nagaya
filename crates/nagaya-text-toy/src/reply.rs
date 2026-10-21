//! The reply parser of the toy family, on `gogi_toy::reply`.

use gogi_toy::VOCAB;
use gogi_toy::reply::{Call, ReplyParser as Inner};
use nagaya_council::family::{FamilyError, ParsedReply, ReplyCall, ReplyParser, ReplyStep};

pub(crate) struct ToyReplyParser {
    inner: Inner,
    ended: bool,
}

impl ToyReplyParser {
    pub(crate) fn new(thinking: bool) -> Self {
        ToyReplyParser {
            inner: Inner::new(thinking),
            ended: false,
        }
    }
}

impl ReplyParser for ToyReplyParser {
    fn push(&mut self, token: u32) -> Result<ReplyStep, FamilyError> {
        // The inner parser panics on these two. They are bugs in the caller, so they are errors here.
        if self.ended {
            return Err(FamilyError::internal(format!(
                "token {token} came after <end>: the reply is over"
            )));
        }
        if token as usize >= VOCAB {
            return Err(FamilyError::internal(format!(
                "{token} is not a token: ids run 0 to {}",
                VOCAB - 1
            )));
        }
        if token == gogi_toy::END {
            self.ended = true;
        }
        let step = self.inner.push(token);
        Ok(ReplyStep {
            phase: step.phase,
            piece: step.piece,
        })
    }

    fn ended(&self) -> bool {
        self.ended
    }

    fn finish(self: Box<Self>) -> ParsedReply {
        let reply = self.inner.finish();
        ParsedReply {
            content: reply.content,
            reasoning: reply.reasoning,
            calls: reply
                .calls
                .into_iter()
                .map(|call| match call {
                    Call::Parsed { name, input } => ReplyCall::Parsed { name, input },
                    Call::Malformed { text, error } => ReplyCall::Malformed { text, error },
                })
                .collect(),
            ended: reply.ended,
        }
    }
}
