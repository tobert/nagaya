// The ops of the Inkling 2 binary (MK_INKLING2). Each op is one `else if (op == ...)` branch of the task loop of
// megakernel.cl, which includes this file, so each op sees the locals of the loop. The opcodes must match
// src/protocol.rs. The host form of each op is in crates/nagaya-mk-inkling2-bmg/src/host.
