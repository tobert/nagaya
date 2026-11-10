//! Council v2 logic with no I/O (spec 11): the sampler, the family interface, the template id, the plans of a build,
//! the build plan of a continuation, and the numbers of reads and decisions.

pub mod build;
pub mod continuation;
pub mod control;
pub mod decision;
pub mod ensemble;
pub mod family;
pub mod read;
pub mod sampler;
pub mod template_id;
pub mod trace;
