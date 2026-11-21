//! The `f32` host form of the Inkling 2 text model: one op for each layer part.
//!
//! It fixes the meaning of each op before the device code exists. Each device op is gated against the host form of
//! its op. The reference is transformers `inkling2_loop` at `046bdb8cc2` (`modeling_inkling2_loop.py`); each op
//! cites its lines. The oracle sums in `f32` with the experts expanded to `bf16` (`tools/inkling2_reference.py`), and
//! so does this form.

pub mod ops;
