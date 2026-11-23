//! The descriptors and the task tables of the ops of a run that have no public builder in `nagaya-mk-cuda` (milestone
//! 37 step 10). Each function is a pure function of device addresses and shapes, so a host test checks it against the
//! conditions of the kernel. A function returns an error that names the wrong value, where the kernel would trap.
//!
//! The ops with a builder in `nagaya-mk-cuda` are `OP_PROJ` (`ProjLayout::descriptor`), `OP_ATTN`
//! (`AttnShape::descriptor`, `attn::table`) and the matmul of the head (`MatmulShape::descriptor`, used by [`head`]).

use nagaya_mk_cuda::experts::{
    EXPERT_TABLE_QACT, ExpertFormat, ROUTE_TABLE_BIAS, ROUTE_TABLE_NORM_WEIGHT, ROUTE_TABLE_ROW_FACTOR, route_items,
};
use nagaya_mk_cuda::proj::ProjLayout;
use nagaya_mk_cuda::protocol::{
    EXPERT_TABLE_AMAX, FLAG_DOWN, FLAG_NORM, FLAG_PAIRS, FLAG_SOFTCAP_30, FLAG_UP, MATMUL_TABLE_NORM_WEIGHT,
    MkTaskDescriptor, MkTaskTable, OP_EMBED, OP_EXPERT_MXFP8, OP_EXPERT_NVFP4, OP_GATE_DOWN, OP_GATE_UP, OP_MIX,
    OP_MIX_OUT, OP_QUANT_MXFP8, OP_QUANT_NVFP4, OP_ROUTE, PROJ_TABLE_K_CONV, PROJ_TABLE_K_OUT, PROJ_TABLE_K_STATE,
    PROJ_TABLE_NORM_K, PROJ_TABLE_NORM_Q, PROJ_TABLE_NORM_V, PROJ_TABLE_Q_SCALE, PROJ_TABLE_REL_SCALE, PROJ_TABLE_RF_K,
    PROJ_TABLE_RF_V, PROJ_TABLE_V_CONV, PROJ_TABLE_V_OUT, PROJ_TABLE_V_STATE, PROJ_TABLE_X_K, PROJ_TABLE_X_V,
    expert_pair_bound,
};
use nagaya_mk_cuda::repack::MatmulShape;
use nagaya_mk_cuda::rows::{GATE_DOWN_ITEMS, GATE_UP_ITEM_BYTES, GATE_UP_TILES, gate_down_shape};

use crate::spec::{HIDDEN, INTERMEDIATE, VOCAB};

// The slots of the task tables of the row ops (`kernels/ops_rows.cuh`). A host test reads them from the kernel source.
pub const EMBED_TABLE_NORM: usize = 0;
pub const EMBED_TABLE_PLANE0: usize = 1;
pub const EMBED_TABLE_PLANE0_RF: usize = 2;
pub const MIX_TABLE_STACK: usize = 0;
pub const MIX_TABLE_STACK_RF: usize = 5;
pub const MIX_TABLE_MSG: usize = 10;
pub const MIX_TABLE_MSG_RF: usize = 12;
pub const MIX_TABLE_QUERIES: usize = 14;
pub const MIX_TABLE_MSG_QUERIES: usize = 15;
pub const GATE_DOWN_TABLE_A: usize = 0;
pub const GATE_DOWN_TABLE_M: usize = 3;
pub const GATE_DOWN_TABLE_A_RF: usize = 6;
pub const GATE_DOWN_TABLE_M_RF: usize = 9;
pub const GATE_UP_TABLE_A: usize = 0;
pub const GATE_UP_TABLE_M: usize = 3;
pub const GATE_UP_TABLE_U: usize = 6;
pub const GATE_UP_TABLE_U_RF: usize = 9;
pub const GATE_UP_TABLE_NORM: usize = 12;

/// The most rows in the stack of a mix.
pub const MIX_MAX_STACK: usize = 5;

/// The tiles of an item of the up op of the experts for `tokens` tokens. A pair has 36 tiles, so 9 tiles make 4 items of
/// a pair: a decode step of 8 pairs has 32 items, and the claim of the later items balances their tails. The pick comes
/// from the sweep of lane E (milestone 37 step 10). It is a host rule and not part of the identity: the bits of the op
/// do not depend on it.
pub fn up_tiles_per_item(_tokens: usize) -> usize {
    9
}

/// The tiles of an item of the down op of the experts for `tokens` tokens: 3 tiles, so a decode step has 96 items..
pub fn down_tiles_per_item(_tokens: usize) -> usize {
    3
}

/// The width of the up and the down matrix of an expert in rows: gate and up of 1,152 columns, and the 2,304 outputs.
const EXPERT_ROWS: usize = HIDDEN;

fn aligned(name: &str, address: u64, align: u64) -> Result<(), String> {
    if address.is_multiple_of(align) {
        Ok(())
    } else {
        Err(format!(
            "the address of {name} is {address:#x}, which is not a multiple of {align}. Align the buffer"
        ))
    }
}

fn nonzero(name: &str, address: u64) -> Result<(), String> {
    if address == 0 {
        Err(format!("the address of {name} is 0. Give the buffer"))
    } else {
        Ok(())
    }
}

/// An address that an NVFP4 layer gives and an MXFP8 layer leaves out: `None` is 0.
fn nonzero_if(name: &str, address: Option<u64>) -> Result<u64, String> {
    match address {
        Some(at) => nonzero(name, at).map(|()| at),
        None => Ok(0),
    }
}

fn rows_in(name: &str, rows: usize, most: usize) -> Result<u32, String> {
    if rows == 0 || rows > most {
        return Err(format!(
            "the op {name} takes from 1 to {most} rows, and this one has {rows}. Split the rows over several tasks"
        ));
    }
    Ok(rows as u32)
}

fn count(what: &str, value: usize) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("the {what}, {value}, does not fit in 32 bits"))
}

/// The scale of a mix: `2304^-0.5` as the host form computes it.
pub fn mix_scale() -> f32 {
    (HIDDEN as f32).powf(-0.5)
}

/// `OP_EMBED` for `rows` ids: `ids` (u32), `embed` (bf16, `VOCAB` rows of 2,304), `x` (f32 rows) and `rf` (f32). The
/// table gives the norm weight (f32) and the plane that is a copy of `x` with its row factors.
pub fn embed(rows: usize, ids: u64, embed: u64, x: u64, rf: u64) -> Result<MkTaskDescriptor, String> {
    let rows = rows_in("embed", rows, 512)?;
    aligned("the ids", ids, 4)?;
    aligned("the embedding", embed, 2)?;
    aligned("x", x, 16)?;
    aligned("the row factors", rf, 4)?;
    Ok(MkTaskDescriptor {
        opcode: OP_EMBED,
        items: rows,
        rows,
        a: ids,
        b: embed,
        c: x,
        e: rf,
        n: VOCAB as u32,
        ..Default::default()
    })
}

/// The table of `OP_EMBED`.
pub fn embed_table(norm: u64, plane0: u64, plane0_rf: u64) -> Result<MkTaskTable, String> {
    nonzero("the embedding norm", norm)?;
    nonzero("the plane B0", plane0)?;
    nonzero("the row factors of B0", plane0_rf)?;
    let mut table = MkTaskTable::default();
    table.ptr[EMBED_TABLE_NORM] = norm;
    table.ptr[EMBED_TABLE_PLANE0] = plane0;
    table.ptr[EMBED_TABLE_PLANE0_RF] = plane0_rf;
    Ok(table)
}

/// The queries of a `OP_MIX` task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixShape {
    pub rows: usize,
    /// The queries of the attention mix: 1 on a sliding layer, 3 on a full layer, 0 for none.
    pub queries: usize,
    /// The depth of the stack of the attention mix.
    pub depth: usize,
    /// The queries of the message mix: 4 in a run with messages, or 0.
    pub message_queries: usize,
}

/// `OP_MIX`: `a` receives the attention mix (`queries` x `rows` rows) and `b` its row factors, `c` receives the message
/// mix and `e` its row factors.
pub fn mix(shape: &MixShape, a: u64, b: u64, c: u64, e: u64, table: u64) -> Result<MkTaskDescriptor, String> {
    let rows = rows_in("mix", shape.rows, 512)?;
    if shape.queries + shape.message_queries == 0 {
        return Err("a mix with no query and no message query has no output. Give a query".to_owned());
    }
    if shape.queries != 0 && !(1..=MIX_MAX_STACK).contains(&shape.depth) {
        return Err(format!(
            "the stack of a mix has from 1 to {MIX_MAX_STACK} rows, and this one has {}",
            shape.depth
        ));
    }
    if shape.queries != 0 {
        aligned("the mix output a", a, 16)?;
    }
    if shape.message_queries != 0 {
        aligned("the message mix output c", c, 16)?;
    }
    aligned("the table", table, 128)?;
    Ok(MkTaskDescriptor {
        opcode: OP_MIX,
        items: count("count of items", shape.rows * (shape.queries + shape.message_queries))?,
        rows,
        a,
        b,
        c,
        d: table,
        e,
        n: shape.queries as u32,
        k: shape.depth as u32,
        kt: shape.message_queries as u32,
        group_items: if shape.message_queries > 0 { 2 } else { 0 },
        p0: mix_scale(),
        ..Default::default()
    })
}

/// The addresses of the table of a mix.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MixTable {
    /// The rows of the stack and their row factors, bottom first.
    pub stack: Vec<(u64, u64)>,
    /// The rows of the message mix (`P` and `y`) and their row factors.
    pub message: Vec<(u64, u64)>,
    pub queries: u64,
    pub message_queries: u64,
}

impl MixTable {
    pub fn build(&self) -> Result<MkTaskTable, String> {
        if self.stack.len() > MIX_MAX_STACK || self.message.len() > 2 {
            return Err(format!(
                "a mix has at most {MIX_MAX_STACK} stack rows and 2 message rows, and this one has {} and {}",
                self.stack.len(),
                self.message.len()
            ));
        }
        let mut table = MkTaskTable::default();
        for (i, (plane, rf)) in self.stack.iter().enumerate() {
            nonzero("a plane of the stack", *plane)?;
            nonzero("the row factors of a plane", *rf)?;
            table.ptr[MIX_TABLE_STACK + i] = *plane;
            table.ptr[MIX_TABLE_STACK_RF + i] = *rf;
        }
        for (i, (plane, rf)) in self.message.iter().enumerate() {
            nonzero("a row of the message mix", *plane)?;
            nonzero("the row factors of a message row", *rf)?;
            table.ptr[MIX_TABLE_MSG + i] = *plane;
            table.ptr[MIX_TABLE_MSG_RF + i] = *rf;
        }
        table.ptr[MIX_TABLE_QUERIES] = self.queries;
        table.ptr[MIX_TABLE_MSG_QUERIES] = self.message_queries;
        Ok(table)
    }
}

/// `OP_MIX_OUT`: one query over a stack of `depth` rows. `a` receives the mixed rows and `b` their row factors.
pub fn mix_out(rows: usize, depth: usize, a: u64, b: u64, table: u64) -> Result<MkTaskDescriptor, String> {
    let shape = MixShape {
        rows,
        queries: 1,
        depth,
        message_queries: 0,
    };
    Ok(MkTaskDescriptor {
        opcode: OP_MIX_OUT,
        ..mix(&shape, a, b, 0, 0, table)?
    })
}

/// `OP_GATE_DOWN` for `gates` gates (1 to 3). `weights` holds the repacked down matrices of the gates, one after the
/// other (`gate_down_weights`), `g` receives `gates` x `rows` x 64 floats, and `scratch` holds the counters and the
/// partials.
pub fn gate_down(
    rows: usize,
    gates: usize,
    weights: u64,
    g: u64,
    scratch: u64,
    table: u64,
) -> Result<MkTaskDescriptor, String> {
    let rows = rows_in("gate down", rows, 512)?;
    if !(1..=3).contains(&gates) {
        return Err(format!("a gate down task has 1 to 3 gates, and this one has {gates}"));
    }
    aligned("the gate weights", weights, 16)?;
    aligned("g", g, 4)?;
    aligned("the scratch", scratch, 4)?;
    aligned("the table", table, 128)?;
    let shape = gate_down_shape();
    Ok(MkTaskDescriptor {
        opcode: OP_GATE_DOWN,
        items: count("count of items", gates * GATE_DOWN_ITEMS)?,
        rows,
        b: weights,
        c: g,
        d: table,
        f: scratch,
        n: gates as u32,
        k: shape.k as u32,
        kt: shape.kt as u32,
        group_items: shape.slices() as u32,
        item_bytes: shape.item_bytes() as u32,
        ..Default::default()
    })
}

/// The table of `OP_GATE_DOWN`: for each gate, the row `a`, the message `m`, and their row factors.
pub fn gate_down_table(gates: &[GateInputs]) -> Result<MkTaskTable, String> {
    if gates.len() > 3 {
        return Err(format!(
            "a gate down task has at most 3 gates, and this one has {}",
            gates.len()
        ));
    }
    let mut table = MkTaskTable::default();
    for (i, g) in gates.iter().enumerate() {
        for (name, at) in [("a", g.a), ("m", g.m), ("a_rf", g.a_rf), ("m_rf", g.m_rf)] {
            nonzero(&format!("the gate input {name}"), at)?;
        }
        aligned("a gate row a", g.a, 8)?;
        aligned("a gate message m", g.m, 8)?;
        table.ptr[GATE_DOWN_TABLE_A + i] = g.a;
        table.ptr[GATE_DOWN_TABLE_M + i] = g.m;
        table.ptr[GATE_DOWN_TABLE_A_RF + i] = g.a_rf;
        table.ptr[GATE_DOWN_TABLE_M_RF + i] = g.m_rf;
    }
    Ok(table)
}

/// The inputs of one message gate: the row, the message, and their row factors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GateInputs {
    pub a: u64,
    pub m: u64,
    pub a_rf: u64,
    pub m_rf: u64,
}

/// `OP_GATE_UP` for `gates` gates. `g` is the output of the gate down task, `weights` the repacked forget and gain
/// matrices (`repack_gate_up`), `scratch` the counters and the partials. With `norm_out`, one gate writes `h_e` there.
pub fn gate_up(
    rows: usize,
    gates: usize,
    g: u64,
    weights: u64,
    norm_out: Option<u64>,
    scratch: u64,
    table: u64,
) -> Result<MkTaskDescriptor, String> {
    let rows = rows_in("gate up", rows, 512)?;
    if !(1..=3).contains(&gates) {
        return Err(format!("a gate up task has 1 to 3 gates, and this one has {gates}"));
    }
    if norm_out.is_some() && gates != 1 {
        return Err(format!(
            "only a task of one gate writes h_e, and this one has {gates} gates. Give the norm to a task of one gate"
        ));
    }
    aligned("g", g, 8)?;
    aligned("the gate weights", weights, 16)?;
    aligned("the scratch", scratch, 4)?;
    aligned("the table", table, 128)?;
    if let Some(out) = norm_out {
        aligned("h_e", out, 4)?;
    }
    Ok(MkTaskDescriptor {
        opcode: OP_GATE_UP,
        flags: if norm_out.is_some() { FLAG_NORM } else { 0 },
        items: count("count of items", gates * GATE_UP_TILES)?,
        rows,
        a: g,
        b: weights,
        c: norm_out.unwrap_or(0),
        d: table,
        f: scratch,
        n: gates as u32,
        k: 64,
        item_bytes: GATE_UP_ITEM_BYTES as u32,
        ..Default::default()
    })
}

/// The table of `OP_GATE_UP`: the rows, the messages, the outputs with their row factors, and the norm weight (f32).
pub fn gate_up_table(
    gates: &[GateInputs],
    outputs: &[(u64, u64)],
    norm_weight: Option<u64>,
) -> Result<MkTaskTable, String> {
    if gates.len() != outputs.len() || gates.len() > 3 {
        return Err(format!(
            "a gate up task has 1 to 3 gates with one output each, and this one has {} gates and {} outputs",
            gates.len(),
            outputs.len()
        ));
    }
    let mut table = MkTaskTable::default();
    for (i, (g, (u, u_rf))) in gates.iter().zip(outputs).enumerate() {
        nonzero("a gate row a", g.a)?;
        nonzero("a gate message m", g.m)?;
        nonzero("a gate output u", *u)?;
        nonzero("the row factors of a gate output", *u_rf)?;
        table.ptr[GATE_UP_TABLE_A + i] = g.a;
        table.ptr[GATE_UP_TABLE_M + i] = g.m;
        table.ptr[GATE_UP_TABLE_U + i] = *u;
        table.ptr[GATE_UP_TABLE_U_RF + i] = *u_rf;
    }
    table.ptr[GATE_UP_TABLE_NORM] = norm_weight.unwrap_or(0);
    Ok(table)
}

/// `OP_QUANT_NVFP4` or `OP_QUANT_MXFP8`: quantize `rows` rows of `k` floats at `x` into the layout of `QuantRowsLayout`
/// at `out`.
pub fn quant(format: ExpertFormat, rows: usize, k: usize, x: u64, out: u64) -> Result<MkTaskDescriptor, String> {
    let rows = rows_in("quant", rows, 4096)?;
    if k == 0 || !k.is_multiple_of(128) {
        return Err(format!(
            "a quantized row of {k} values is not a whole number of groups of 128 values"
        ));
    }
    aligned("the rows to quantize", x, 16)?;
    aligned("the quantized rows", out, 16)?;
    Ok(MkTaskDescriptor {
        opcode: match format {
            ExpertFormat::Nvfp4 => OP_QUANT_NVFP4,
            ExpertFormat::Mxfp8 => OP_QUANT_MXFP8,
        },
        items: rows,
        rows,
        k: k as u32,
        a: x,
        c: out,
        ..Default::default()
    })
}

/// The addresses of `OP_ROUTE`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RouteAddrs {
    pub h: u64,
    pub router: u64,
    pub picks: u64,
    pub pairs: u64,
    pub scratch: u64,
    pub table: u64,
}

/// `OP_ROUTE` for `rows` rows. With `norm`, the op applies the norm of `h` itself (the table holds the weight and the
/// row factors).
pub fn route(rows: usize, a: &RouteAddrs, norm: bool) -> Result<MkTaskDescriptor, String> {
    let rows_u32 = rows_in("route", rows, 1024)?;
    aligned("h", a.h, 8)?;
    for (name, at) in [
        ("the router", a.router),
        ("the picks", a.picks),
        ("the pair list", a.pairs),
        ("the scratch", a.scratch),
    ] {
        aligned(name, at, 16)?;
    }
    aligned("the table", a.table, 128)?;
    Ok(MkTaskDescriptor {
        opcode: OP_ROUTE,
        flags: FLAG_PAIRS | if norm { FLAG_NORM } else { 0 },
        items: count("count of items", route_items(rows))?,
        rows: rows_u32,
        a: a.h,
        b: a.router,
        c: a.picks,
        d: a.table,
        e: a.pairs,
        f: a.scratch,
        n: 128,
        k: HIDDEN as u32,
        kt: 768,
        group_items: 3,
        // The producer prefetches the item of its block before the wait, for up to 16 rows.
        item_bytes: if rows <= 16 { 24_576 } else { 0 },
        p0: ROUTE_SCALE,
        ..Default::default()
    })
}

/// The factor of the routing weights (`route_scale`, 8). A host test compares it with `ops::ROUTE_SCALE`.
pub const ROUTE_SCALE: f32 = 8.0;

/// The table of `OP_ROUTE`: the bias (f32, 128), and with the norm, its weight and the row factors.
pub fn route_table(bias: u64, norm: Option<(u64, u64)>) -> Result<MkTaskTable, String> {
    nonzero("the router bias", bias)?;
    let mut table = MkTaskTable::default();
    table.ptr[ROUTE_TABLE_BIAS] = bias;
    if let Some((weight, rf)) = norm {
        nonzero("the norm weight of the router", weight)?;
        nonzero("the row factors of the router", rf)?;
        table.ptr[ROUTE_TABLE_NORM_WEIGHT] = weight;
        table.ptr[ROUTE_TABLE_ROW_FACTOR] = rf;
    }
    Ok(table)
}

fn expert_opcode(format: ExpertFormat) -> u16 {
    match format {
        ExpertFormat::Nvfp4 => OP_EXPERT_NVFP4,
        ExpertFormat::Mxfp8 => OP_EXPERT_MXFP8,
    }
}

/// The addresses of an expert op.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExpertAddrs {
    /// The quantized rows that the op reads.
    pub input: u64,
    /// The bank of the repacked weights.
    pub bank: u64,
    /// The activations (`UP`) or the slot rows (`DOWN`) that the op writes.
    pub out: u64,
    pub table: u64,
    pub pairs: u64,
    pub counters: u64,
}

/// The up op of the experts for `tokens` tokens: `FLAG_PAIRS | FLAG_UP`. `x_rows` is the capacity of the quantized
/// input, and `denominator` is the outer scale denominator of the weight (`p1`, NVFP4 only). An item covers
/// `tiles_per_item` of the 36 tiles of a pair, and the count must divide 36.
pub fn up(
    format: ExpertFormat,
    tokens: usize,
    x_rows: usize,
    tiles_per_item: usize,
    a: &ExpertAddrs,
    denominator: f32,
) -> Result<MkTaskDescriptor, String> {
    rows_in("up", tokens, 512)?;
    let tiles = EXPERT_ROWS / 64;
    if tiles_per_item == 0 || !tiles.is_multiple_of(tiles_per_item) {
        return Err(format!(
            "{tiles_per_item} tiles for an item do not divide the {tiles} tiles of a pair. Use a divisor of {tiles}"
        ));
    }
    aligned("the pair list", a.pairs, 16)?;
    aligned("the activations", a.out, 16)?;
    aligned("the counters", a.counters, 4)?;
    aligned("the table", a.table, 128)?;
    Ok(MkTaskDescriptor {
        opcode: expert_opcode(format),
        flags: FLAG_PAIRS | FLAG_UP,
        items: count("count of items", expert_pair_bound(tokens) * tiles / tiles_per_item)?,
        rows: x_rows as u32,
        a: a.input,
        b: a.bank,
        c: a.out,
        d: a.table,
        e: a.pairs,
        f: a.counters,
        n: EXPERT_ROWS as u32,
        k: HIDDEN as u32,
        kt: tokens as u32,
        group_items: (tiles / tiles_per_item) as u32,
        pos0: tiles_per_item as u32,
        p0: 1.0,
        p1: denominator,
        ..Default::default()
    })
}

/// The table of the up op: the amax of the weights and the buffer of the quantized activations.
pub fn up_table(amax: Option<u64>, qact: u64) -> Result<MkTaskTable, String> {
    let amax = nonzero_if("the amax of the weights", amax)?;
    aligned("the quantized activations", qact, 16)?;
    nonzero("the quantized activations", qact)?;
    let mut table = MkTaskTable::default();
    table.ptr[EXPERT_TABLE_AMAX] = amax;
    table.ptr[EXPERT_TABLE_QACT] = qact;
    Ok(table)
}

/// The down op of the experts for `tokens` tokens: `FLAG_PAIRS | FLAG_DOWN`. The input is `8 * tokens` quantized rows.
/// An item covers `tiles_per_item` of the 36 tiles of a pair, and the count must divide 36.
pub fn down(
    format: ExpertFormat,
    tokens: usize,
    tiles_per_item: usize,
    a: &ExpertAddrs,
    denominator: f32,
) -> Result<MkTaskDescriptor, String> {
    rows_in("down", tokens, 512)?;
    let tiles = EXPERT_ROWS / 64;
    if tiles_per_item == 0 || !tiles.is_multiple_of(tiles_per_item) {
        return Err(format!(
            "{tiles_per_item} tiles for an item do not divide the {tiles} tiles of a pair. Use a divisor of {tiles}"
        ));
    }
    aligned("the pair list", a.pairs, 16)?;
    aligned("the slot rows", a.out, 4)?;
    aligned("the counters of the down op", a.counters, 8)?;
    aligned("the table", a.table, 128)?;
    Ok(MkTaskDescriptor {
        opcode: expert_opcode(format),
        flags: FLAG_PAIRS | FLAG_DOWN,
        items: count("count of items", expert_pair_bound(tokens) * tiles / tiles_per_item)?,
        rows: (8 * tokens) as u32,
        a: a.input,
        b: a.bank,
        c: a.out,
        d: a.table,
        e: a.pairs,
        f: a.counters,
        n: EXPERT_ROWS as u32,
        k: INTERMEDIATE as u32,
        kt: tokens as u32,
        pos0: tiles_per_item as u32,
        p0: 1.0,
        p1: denominator,
        ..Default::default()
    })
}

/// The table of the down op: the amax, the residual `r`, the weight and the state of the conv, and the output.
pub fn down_table(
    amax: Option<u64>,
    residual: u64,
    conv_weight: u64,
    conv_state: u64,
    out: u64,
    out_rf: u64,
) -> Result<MkTaskTable, String> {
    use nagaya_mk_cuda::experts::{
        EXPERT_TABLE_CONV_STATE, EXPERT_TABLE_CONV_WEIGHT, EXPERT_TABLE_OUT, EXPERT_TABLE_OUT_RF, EXPERT_TABLE_RESIDUAL,
    };
    let amax = nonzero_if("the amax of the weights", amax)?;
    for (name, at) in [
        ("the residual", residual),
        ("the conv weight", conv_weight),
        ("the conv state", conv_state),
        ("the output", out),
        ("the row factors of the output", out_rf),
    ] {
        nonzero(name, at)?;
    }
    let mut table = MkTaskTable::default();
    table.ptr[EXPERT_TABLE_AMAX] = amax;
    table.ptr[EXPERT_TABLE_RESIDUAL] = residual;
    table.ptr[EXPERT_TABLE_CONV_WEIGHT] = conv_weight;
    table.ptr[EXPERT_TABLE_CONV_STATE] = conv_state;
    table.ptr[EXPERT_TABLE_OUT] = out;
    aligned("the row factors of the output", out_rf, 4)?;
    table.ptr[EXPERT_TABLE_OUT_RF] = out_rf;
    Ok(table)
}

/// The table of `OP_PROJ`. A sliding layer gives the same pointer for q, k and v.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProjTable {
    pub norms: [u64; 3],
    /// The input rows of the k rows and of the v rows, and their row factors.
    pub x: [u64; 2],
    pub rf: [u64; 2],
    pub q_scale: u64,
    pub rel_scale: u64,
    pub conv: [u64; 2],
    pub state: [u64; 2],
    pub planes: [u64; 2],
}

impl ProjTable {
    pub fn build(&self) -> Result<MkTaskTable, String> {
        let mut table = MkTaskTable::default();
        let slots = [
            (PROJ_TABLE_NORM_Q, self.norms[0], "the q norm"),
            (PROJ_TABLE_NORM_K, self.norms[1], "the k norm"),
            (PROJ_TABLE_NORM_V, self.norms[2], "the v norm"),
            (PROJ_TABLE_X_K, self.x[0], "the input of k"),
            (PROJ_TABLE_X_V, self.x[1], "the input of v"),
            (PROJ_TABLE_RF_K, self.rf[0], "the row factors of k"),
            (PROJ_TABLE_RF_V, self.rf[1], "the row factors of v"),
            (PROJ_TABLE_Q_SCALE, self.q_scale, "the q scale"),
            (PROJ_TABLE_REL_SCALE, self.rel_scale, "the rel scale"),
            (PROJ_TABLE_K_CONV, self.conv[0], "the k conv weight"),
            (PROJ_TABLE_V_CONV, self.conv[1], "the v conv weight"),
            (PROJ_TABLE_K_STATE, self.state[0], "the k conv state"),
            (PROJ_TABLE_V_STATE, self.state[1], "the v conv state"),
            (PROJ_TABLE_K_OUT, self.planes[0], "the k plane"),
            (PROJ_TABLE_V_OUT, self.planes[1], "the v plane"),
        ];
        for (slot, at, name) in slots {
            nonzero(name, at)?;
            table.ptr[slot] = at;
        }
        Ok(table)
    }
}

/// The matmul of the head: `rows` rows of the output residual from `f` (with row factors `rf`), softcap 30. The table
/// holds the norm weight (bf16).
pub fn head(rows: usize, f: u64, rf: u64, weights: u64, logits: u64, scratch: u64) -> Result<MkTaskDescriptor, String> {
    let shape = MatmulShape::new(crate::pass::layout::LOGIT_COLUMNS, HIDDEN)?;
    let mut desc = shape.descriptor(rows, f, weights, logits, scratch)?;
    desc.flags = FLAG_NORM | FLAG_SOFTCAP_30;
    desc.e = rf;
    Ok(desc)
}

/// The table of the head: the norm weight (bf16).
pub fn head_table(norm_bf16: u64) -> Result<MkTaskTable, String> {
    nonzero("the norm weight of the head", norm_bf16)?;
    let mut table = MkTaskTable::default();
    table.ptr[MATMUL_TABLE_NORM_WEIGHT] = norm_bf16;
    Ok(table)
}

/// The layout of `J` for a layer, as `ProjLayout` of `nagaya-mk-cuda`.
pub fn proj_layout(full: bool) -> ProjLayout {
    ProjLayout { full }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nagaya_mk_cuda::protocol::{EXPERT_MAX_ROWS, TABLE_POINTERS};

    const ROWS_CUH: &str = include_str!("../../../nagaya-mk-cuda/kernels/ops_rows.cuh");

    /// The value of `constexpr uint32_t NAME = N;` in a kernel file.
    fn kernel_const(src: &str, name: &str) -> usize {
        let key = format!("constexpr uint32_t {name} = ");
        let at = src
            .find(&key)
            .unwrap_or_else(|| panic!("the kernel has no constant {name}"))
            + key.len();
        let end = src[at..].find(';').unwrap();
        src[at..at + end]
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("{name} is not a number"))
    }

    #[test]
    fn the_slots_are_those_of_the_kernel_source() {
        for (rust, kernel) in [
            (EMBED_TABLE_NORM, "RW_EMBED_NORM"),
            (EMBED_TABLE_PLANE0, "RW_EMBED_PLANE0"),
            (EMBED_TABLE_PLANE0_RF, "RW_EMBED_PLANE0_RF"),
            (MIX_TABLE_STACK, "RW_MIX_STACK"),
            (MIX_TABLE_STACK_RF, "RW_MIX_STACK_RF"),
            (MIX_TABLE_MSG, "RW_MIX_MSG"),
            (MIX_TABLE_MSG_RF, "RW_MIX_MSG_RF"),
            (MIX_TABLE_QUERIES, "RW_MIX_QUERIES"),
            (MIX_TABLE_MSG_QUERIES, "RW_MIX_MSG_QUERIES"),
            (GATE_DOWN_TABLE_A, "RW_GD_A"),
            (GATE_DOWN_TABLE_M, "RW_GD_M"),
            (GATE_DOWN_TABLE_A_RF, "RW_GD_A_RF"),
            (GATE_DOWN_TABLE_M_RF, "RW_GD_M_RF"),
            (GATE_UP_TABLE_A, "RW_GU_A"),
            (GATE_UP_TABLE_M, "RW_GU_M"),
            (GATE_UP_TABLE_U, "RW_GU_U"),
            (GATE_UP_TABLE_U_RF, "RW_GU_U_RF"),
            (GATE_UP_TABLE_NORM, "RW_GU_NORM"),
            (MIX_MAX_STACK, "RW_MAX_STACK"),
        ] {
            assert_eq!(rust, kernel_const(ROWS_CUH, kernel), "{kernel}");
        }
        const { assert!(MIX_TABLE_MSG_QUERIES < TABLE_POINTERS) };
    }

    #[test]
    fn the_route_scale_is_the_scale_of_the_host_form() {
        assert_eq!(ROUTE_SCALE, nagaya_mk_inkling2_bmg::host::ops::ROUTE_SCALE);
    }

    #[test]
    fn embed_has_the_conditions_of_the_kernel() {
        let d = embed(5, 0x1000, 0x2000, 0x3000, 0x4000).unwrap();
        // `items == rows`, `n` is the vocabulary, and the alignment of each address.
        assert_eq!((d.opcode, d.items, d.rows, d.n), (OP_EMBED, 5, 5, 201_024));
        assert_eq!((d.a, d.b, d.c, d.e), (0x1000, 0x2000, 0x3000, 0x4000));
        assert!(
            embed(5, 0x1002, 0x2000, 0x3000, 0x4000)
                .unwrap_err()
                .contains("the ids")
        );
        assert!(embed(5, 0x1000, 0x2000, 0x3008, 0x4000).unwrap_err().contains("x"));
        assert!(
            embed(0, 0x1000, 0x2000, 0x3000, 0x4000)
                .unwrap_err()
                .contains("1 to 512")
        );
        let t = embed_table(8, 16, 24).unwrap();
        assert_eq!(&t.ptr[..3], &[8, 16, 24]);
        assert!(embed_table(8, 0, 24).unwrap_err().contains("B0"));
    }

    #[test]
    fn mix_has_the_item_count_the_depth_and_the_groups_of_the_kernel() {
        // The kernel needs items == rows * (queries + message queries), and the depth from 1 to 5.
        for (rows, q, depth, mq) in [(1, 1, 2, 0), (5, 3, 4, 4), (16, 1, 5, 4), (7, 0, 0, 4)] {
            let shape = MixShape {
                rows,
                queries: q,
                depth,
                message_queries: mq,
            };
            let d = mix(&shape, 0x1000, 0x2000, 0x3000, 0x4000, 0x5080).unwrap();
            assert_eq!(d.items as usize, rows * (q + mq));
            assert_eq!((d.n, d.k, d.kt), (q as u32, depth as u32, mq as u32));
            assert_eq!(d.group_items, if mq > 0 { 2 } else { 0 });
            assert_eq!(d.p0, (2304f32).powf(-0.5));
        }
        let bad = MixShape {
            rows: 1,
            queries: 1,
            depth: 6,
            message_queries: 0,
        };
        assert!(mix(&bad, 16, 16, 0, 0, 128).unwrap_err().contains("from 1 to 5"));
        let none = MixShape {
            rows: 1,
            queries: 0,
            depth: 0,
            message_queries: 0,
        };
        assert!(mix(&none, 16, 16, 0, 0, 128).unwrap_err().contains("no query"));
        let out = mix_out(3, 5, 16, 32, 128).unwrap();
        assert_eq!((out.opcode, out.n, out.kt, out.k, out.items), (OP_MIX_OUT, 1, 0, 5, 3));
    }

    #[test]
    fn a_mix_table_puts_the_rows_in_the_slots_and_names_an_empty_one() {
        let table = MixTable {
            stack: vec![(16, 160), (32, 176)],
            message: vec![(48, 192), (64, 208)],
            queries: 80,
            message_queries: 96,
        }
        .build()
        .unwrap();
        assert_eq!((table.ptr[0], table.ptr[1], table.ptr[2]), (16, 32, 0));
        assert_eq!((table.ptr[5], table.ptr[6]), (160, 176));
        assert_eq!(
            (table.ptr[10], table.ptr[11], table.ptr[12], table.ptr[13]),
            (48, 64, 192, 208)
        );
        assert_eq!((table.ptr[14], table.ptr[15]), (80, 96));
        let error = MixTable {
            stack: vec![(16, 0)],
            ..Default::default()
        }
        .build()
        .unwrap_err();
        assert!(error.contains("row factors"), "{error}");
    }

    #[test]
    fn the_gates_have_the_shapes_of_the_kernel() {
        for gates in 1..=3usize {
            let d = gate_down(16, gates, 16, 4, 4, 128).unwrap();
            // items = gates * 24, k = 4608, kt = 768, 6 slices, item_bytes = 16 * 768 * 2.
            assert_eq!(
                (d.items as usize, d.n, d.k, d.kt, d.group_items),
                (gates * 24, gates as u32, 4608, 768, 6)
            );
            assert_eq!(d.item_bytes, 16 * 768 * 2);
            let u = gate_up(16, gates, 8, 16, None, 4, 128).unwrap();
            assert_eq!(
                (u.items as usize, u.k, u.item_bytes, u.flags),
                (gates * 36, 64, 8 * 4 * 512, 0)
            );
        }
        assert!(gate_down(513, 3, 16, 4, 4, 128).unwrap_err().contains("1 to 512"));
        assert!(gate_down(512, 3, 16, 4, 4, 128).is_ok());
        assert!(gate_down(1, 4, 16, 4, 4, 128).unwrap_err().contains("1 to 3"));
        assert!(gate_down(1, 1, 8, 4, 4, 128).unwrap_err().contains("gate weights"));
        let norm = gate_up(1, 1, 8, 16, Some(4), 4, 128).unwrap();
        assert_eq!((norm.flags, norm.c), (FLAG_NORM, 4));
        assert!(gate_up(1, 2, 8, 16, Some(4), 4, 128).unwrap_err().contains("one gate"));
        let g = GateInputs {
            a: 8,
            m: 16,
            a_rf: 24,
            m_rf: 32,
        };
        let t = gate_down_table(&[g, g, g]).unwrap();
        assert_eq!((t.ptr[0], t.ptr[3], t.ptr[6], t.ptr[9], t.ptr[11]), (8, 16, 24, 32, 32));
        let t = gate_up_table(&[g], &[(40, 48)], Some(56)).unwrap();
        assert_eq!((t.ptr[0], t.ptr[3], t.ptr[6], t.ptr[9], t.ptr[12]), (8, 16, 40, 48, 56));
        assert!(gate_up_table(&[g], &[], None).unwrap_err().contains("0 outputs"));
    }

    #[test]
    fn quant_and_route_have_the_conditions_of_the_kernel() {
        let q = quant(ExpertFormat::Nvfp4, 5, 2304, 16, 32).unwrap();
        assert_eq!((q.opcode, q.items, q.rows, q.k), (OP_QUANT_NVFP4, 5, 5, 2304));
        assert_eq!(
            quant(ExpertFormat::Mxfp8, 40, 1152, 16, 32).unwrap().opcode,
            OP_QUANT_MXFP8
        );
        assert!(quant(ExpertFormat::Nvfp4, 5, 100, 16, 32).unwrap_err().contains("128"));
        let a = RouteAddrs {
            h: 8,
            router: 16,
            picks: 32,
            pairs: 48,
            scratch: 64,
            table: 128,
        };
        for rows in [1usize, 16, 17, 512] {
            let d = route(rows, &a, false).unwrap();
            // items = row tiles * 24, and item_bytes only up to 16 rows.
            assert_eq!(d.items as usize, rows.div_ceil(16) * 24);
            assert_eq!(d.item_bytes, if rows <= 16 { 24_576 } else { 0 });
            assert_eq!((d.n, d.k, d.kt, d.group_items, d.p0), (128, 2304, 768, 3, 8.0));
            assert_eq!(d.flags, FLAG_PAIRS);
        }
        assert_eq!(route(1, &a, true).unwrap().flags, FLAG_PAIRS | FLAG_NORM);
        assert!(route(1025, &a, false).unwrap_err().contains("1 to 1024"));
        let t = route_table(8, Some((16, 24))).unwrap();
        assert_eq!(&t.ptr[..3], &[8, 16, 24]);
    }

    #[test]
    fn the_expert_ops_have_the_bound_the_groups_and_the_modes_of_the_kernel() {
        let a = ExpertAddrs {
            input: 16,
            bank: 16,
            out: 16,
            table: 128,
            pairs: 16,
            counters: 8,
        };
        for tokens in [1usize, 7, 16, 512] {
            let bound = expert_pair_bound(tokens);
            let (tu, td) = (up_tiles_per_item(tokens), down_tiles_per_item(tokens));
            let u = up(ExpertFormat::Nvfp4, tokens, tokens, tu, &a, 1.5).unwrap();
            // items is the bound times the items of a pair, and the group is the items of a pair.
            assert_eq!(u.items as usize, bound * 36 / tu);
            assert_eq!(
                (u.group_items as usize, u.pos0 as usize, u.n, u.k, u.kt as usize),
                (36 / tu, tu, 2304, 2304, tokens)
            );
            assert_eq!(u.flags, FLAG_PAIRS | FLAG_UP);
            let d = down(ExpertFormat::Mxfp8, tokens, td, &a, 1.25).unwrap();
            assert_eq!((u.p0, u.p1, d.p0, d.p1), (1.0, 1.5, 1.0, 1.25));
            assert_eq!(d.items as usize, bound * 36 / td);
            assert_eq!(d.pos0 as usize, td);
            assert_eq!((d.rows as usize, d.k, d.n, d.item_bytes), (8 * tokens, 1152, 2304, 0));
            assert_eq!(d.opcode, OP_EXPERT_MXFP8);
            assert_eq!(d.flags & (FLAG_UP | FLAG_DOWN), FLAG_DOWN);
        }
        // A token has 8 pairs at most, so a decode step has 8 pairs of 4 up items or 12 down items: 32 and 96.
        assert_eq!(up(ExpertFormat::Nvfp4, 1, 1, 9, &a, 1.0).unwrap().items, 32);
        assert_eq!(down(ExpertFormat::Nvfp4, 1, 3, &a, 1.0).unwrap().items, 96);
        // A count of tiles that does not divide the 36 tiles of a pair is an error.
        for bad in [0usize, 5, 7, 8, 72] {
            assert!(
                up(ExpertFormat::Nvfp4, 1, 1, bad, &a, 1.0)
                    .unwrap_err()
                    .contains("do not divide")
            );
            assert!(
                down(ExpertFormat::Nvfp4, 1, bad, &a, 1.0)
                    .unwrap_err()
                    .contains("do not divide")
            );
        }
        assert_eq!(EXPERT_MAX_ROWS, 16);
        assert!(up_table(Some(0), 16).unwrap_err().contains("amax"));
        assert_eq!(up_table(None, 16).unwrap().ptr[0], 0);
        let t = down_table(Some(8), 16, 24, 32, 40, 48).unwrap();
        assert_eq!(&t.ptr[..7], &[8, 0, 16, 24, 32, 40, 48]);
        assert!(
            down_table(Some(8), 16, 24, 32, 40, 50)
                .unwrap_err()
                .contains("row factors")
        );
        assert!(down_table(Some(8), 16, 24, 0, 40, 48).unwrap_err().contains("state"));
    }

    #[test]
    fn the_proj_table_names_each_empty_slot_and_the_head_has_its_flags() {
        let full = ProjTable {
            norms: [8; 3],
            x: [16; 2],
            rf: [24; 2],
            q_scale: 32,
            rel_scale: 40,
            conv: [48; 2],
            state: [56; 2],
            planes: [64; 2],
        };
        let t = full.build().unwrap();
        assert!(t.ptr[..15].iter().all(|p| *p != 0) && t.ptr[15..].iter().all(|p| *p == 0));
        let error = ProjTable {
            planes: [64, 0],
            ..full
        }
        .build()
        .unwrap_err();
        assert!(error.contains("v plane"), "{error}");
        let h = head(16, 16, 32, 64, 128, 256).unwrap();
        assert_eq!(
            (h.n, h.k, h.items, h.flags, h.e),
            (200_064, 2304, 37_512, FLAG_NORM | FLAG_SOFTCAP_30, 32)
        );
        assert!(head(17, 16, 32, 64, 128, 256).is_ok());
        assert_eq!(head_table(8).unwrap().ptr[0], 8);
        assert!(proj_layout(true).rows() > proj_layout(false).rows());
    }
}
