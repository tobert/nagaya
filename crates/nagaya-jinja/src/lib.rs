//! The nagaya's Jinja interpreter for chat templates (spec 11.3).
//!
//! Each family library runs its chat template here. The interpreter gives the results of Python's jinja2 as Hugging
//! Face transformers sets it up, with the `str` methods and the Unicode tables of Python 3.14.

pub mod chars;
mod error;
pub mod json;
mod methods;
pub mod pystr;
pub mod template;
#[rustfmt::skip]
mod unicode_tables;

pub use error::Error;
pub use json::Json;
pub use template::{Template, truthy};
