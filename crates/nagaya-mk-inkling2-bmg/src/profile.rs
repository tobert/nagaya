//! The profile of a pass: the time of each task from the clocks of the profile binary, and the sums for each op kind
//! (milestone 36 item 0). None of this needs a card. A profiled open of the engine (`Inkling2Engine::open_profiled`)
//! gives a [`Launch`] for each launch of its rings.
//!
//! Each work-group writes three clocks for each task: at its start, after its wait, and at its end, before its
//! signal (`nagaya_mk_bmg::protocol::PROFILE_WORDS`). A work-group passes the wait of a task only after each
//! work-group signalled the task before. Thus the time of task `i` is the time from the end of task `i - 1` in the
//! last work-group to the end of task `i` in the last work-group. That time has two parts: the boundary, until the
//! first work-group passes the wait, and the work, until the last work-group ends. The times of the tasks of a launch
//! add up to the time from the first start to the last end.
//!
//! The clock counts at the clock of the card, and the Xe cores of a B70 card agree to about 600 counts. Thus a
//! boundary can read up to [`SKEW`] counts below zero, and it then counts as zero.

use std::collections::BTreeMap;

use nagaya_mk_bmg::protocol::{self as p, MkTaskDescriptor, PROFILE_WORDS};

/// The clocks of one launch of a profiled ring.
#[derive(Clone, Debug)]
pub struct Launch {
    /// The tasks of the launch, in ring order, with no halt.
    pub tasks: Vec<MkTaskDescriptor>,
    /// `PROFILE_WORDS` clocks of each task of each work-group, at `(task * groups + group) * PROFILE_WORDS`.
    pub clocks: Vec<u64>,
    /// The work-groups of the launch.
    pub groups: usize,
    /// The counts of the clock in each nanosecond: the clock of the card in MHz, divided by 1,000.
    pub ticks_per_ns: f64,
    /// The host time of the launch call in nanoseconds: the fill of the scoreboard, the copy of the ring, the kernel
    /// and the wait for its end.
    pub host_ns: u64,
}

/// The most counts by which a work-group can read the wait of a task as passed before the last work-group ended the
/// task before. The Xe cores agree to about 600 counts (a probe of 2026-11-22 on card 0); this is 1 us at 2,800 MHz.
pub const SKEW: u64 = 2_800;

/// The time of one task, in counts of the clock.
#[derive(Clone, Copy, Debug)]
pub struct TaskTime {
    pub task: MkTaskDescriptor,
    /// From the end of the task before in the last work-group (the first start, for the first task) to the first
    /// work-group that passed the wait of this task.
    pub boundary: u64,
    /// From the first work-group that passed the wait to the last work-group that ended.
    pub work: u64,
    /// From the first work-group that ended to the last: the imbalance of the work-groups.
    pub spread: u64,
}

/// The time of each task of `launch`. It fails when a clock runs back in a work-group, or when a work-group passes a
/// wait more than [`SKEW`] counts before the last work-group ended the task before. Each of these says that the
/// clocks are not one clock of the card.
pub fn task_times(launch: &Launch) -> Result<Vec<TaskTime>, String> {
    let groups = launch.groups;
    let tasks = launch.tasks.len();
    if groups == 0 || launch.clocks.len() != tasks * groups * PROFILE_WORDS {
        return Err(format!(
            "a launch of {tasks} tasks on {groups} work-groups needs {} clocks, and it holds {}",
            tasks * groups * PROFILE_WORDS,
            launch.clocks.len()
        ));
    }
    let clock = |task: usize, group: usize, word: usize| launch.clocks[(task * groups + group) * PROFILE_WORDS + word];
    for group in 0..groups {
        let mut last = 0;
        for task in 0..tasks {
            for word in 0..PROFILE_WORDS {
                let now = clock(task, group, word);
                if now < last {
                    return Err(format!(
                        "clock {word} of task {task} in work-group {group} is {now}, before the clock before it \
                         ({last}): the clocks are not one clock that runs forward"
                    ));
                }
                last = now;
            }
        }
    }
    let mut out = Vec::with_capacity(tasks);
    let mut end_before = (0..groups).map(|g| clock(0, g, 0)).min().unwrap_or(0);
    for (task, descriptor) in launch.tasks.iter().enumerate() {
        let first_run = (0..groups).map(|g| clock(task, g, 1)).min().unwrap();
        let first_end = (0..groups).map(|g| clock(task, g, 2)).min().unwrap();
        let last_end = (0..groups).map(|g| clock(task, g, 2)).max().unwrap();
        if first_run + SKEW < end_before {
            return Err(format!(
                "a work-group passed the wait of task {task} at {first_run}, before the last work-group ended the task \
                 before at {end_before}: the clocks of the work-groups do not agree"
            ));
        }
        out.push(TaskTime {
            task: *descriptor,
            boundary: first_run.saturating_sub(end_before),
            work: last_end.saturating_sub(first_run.max(end_before)),
            spread: last_end - first_end,
        });
        end_before = last_end;
    }
    Ok(out)
}

/// The counts of the clock in each nanosecond, for launches of one card.
pub fn ticks_per_ns(launches: &[Launch]) -> f64 {
    let rate = launches
        .first()
        .expect("a profile of no launch has no clock")
        .ticks_per_ns;
    assert!(
        launches.iter().all(|l| l.ticks_per_ns == rate),
        "the launches of a profile come from cards of other clocks"
    );
    rate
}

/// The name of an opcode of the B70 interpreter.
pub fn op_name(op: u16) -> String {
    let name = match op {
        p::OP_RMSNORM => "RMSNORM",
        p::OP_MATMUL => "MATMUL",
        p::OP_COPY => "COPY",
        p::OP_MOE_PLAN => "MOE_PLAN",
        p::OP_MOE_GATHER => "MOE_GATHER",
        p::OP_EMBED => "EMBED",
        p::OP_ATTN_RESIDUAL => "ATTN_RESIDUAL",
        p::OP_MESSAGE_GATE => "MESSAGE_GATE",
        p::OP_HEAD_CLIP => "HEAD_CLIP",
        p::OP_OUTPUT_GATE => "OUTPUT_GATE",
        p::OP_RESIDUAL_CONV => "RESIDUAL_CONV",
        p::OP_SIGMOID_ROUTER => "SIGMOID_ROUTER",
        p::OP_MESSAGE_GATE_DOWN => "MESSAGE_GATE_DOWN",
        p::OP_INKLING2_ATTN => "INKLING2_ATTN",
        p::OP_INKLING2_ATTN_SUM => "INKLING2_ATTN_SUM",
        p::OP_EXPAND_NVFP4 => "EXPAND_NVFP4",
        p::OP_EXPAND_MXFP8 => "EXPAND_MXFP8",
        p::OP_EXPERT_UP => "EXPERT_UP",
        p::OP_EXPERT_DOWN => "EXPERT_DOWN",
        p::OP_EXPERT_SUM => "EXPERT_SUM",
        p::OP_INKLING2_VISION_EMBED => "INKLING2_VISION_EMBED",
        p::OP_INKLING2_VISION_ROPE => "INKLING2_VISION_ROPE",
        p::OP_INKLING2_VISION_ATTN => "INKLING2_VISION_ATTN",
        p::OP_INKLING2_CONTROL => "INKLING2_CONTROL",
        _ => return format!("op {op}"),
    };
    name.to_owned()
}

/// The key of a task by its op kind alone.
pub fn by_op(task: &MkTaskDescriptor) -> String {
    op_name(task.opcode)
}

/// The key of a task by its op kind, its flags and its shape (`m`, `n`, `k`).
pub fn by_variant(task: &MkTaskDescriptor) -> String {
    format!(
        "{} flags {:#06x} m {} n {} k {}",
        op_name(task.opcode),
        task.flags,
        task.m,
        task.n,
        task.k
    )
}

/// The sums of the tasks of one key.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Row {
    pub key: String,
    pub count: u64,
    pub boundary: u64,
    pub work: u64,
    pub spread: u64,
}

impl Row {
    /// The ticks of the tasks: their boundaries and their work.
    pub fn total(&self) -> u64 {
        self.boundary + self.work
    }
}

/// The sums of `times` for each key, the largest total first.
pub fn rows(times: &[TaskTime], key: fn(&MkTaskDescriptor) -> String) -> Vec<Row> {
    let mut sums: BTreeMap<String, Row> = BTreeMap::new();
    for time in times {
        let k = key(&time.task);
        let row = sums.entry(k.clone()).or_insert_with(|| Row {
            key: k,
            ..Row::default()
        });
        row.count += 1;
        row.boundary += time.boundary;
        row.work += time.work;
        row.spread += time.spread;
    }
    let mut out: Vec<Row> = sums.into_values().collect();
    out.sort_by(|a, b| b.total().cmp(&a.total()).then_with(|| a.key.cmp(&b.key)));
    out
}

/// A table of `rows` for each of `passes` passes, in milliseconds and microseconds at `ticks_per_ns`: the tasks of a
/// pass, the work, the boundary, the mean work of a task, the imbalance, and the share of the total.
pub fn table(rows: &[Row], passes: usize, ticks_per_ns: f64) -> String {
    let ms = |ticks: u64| ticks as f64 / ticks_per_ns / 1e6 / passes as f64;
    let total: u64 = rows.iter().map(Row::total).sum();
    let mut out = String::from(
        "| op | tasks | work ms | boundary ms | mean work us | spread ms | share |\n|---|---:|---:|---:|---:|---:|---:|\n",
    );
    for row in rows {
        out += &format!(
            "| {} | {} | {:.2} | {:.2} | {:.1} | {:.2} | {:.1}% |\n",
            row.key,
            row.count as f64 / passes as f64,
            ms(row.work),
            ms(row.boundary),
            row.work as f64 / ticks_per_ns / 1e3 / row.count as f64,
            ms(row.spread),
            100.0 * row.total() as f64 / total as f64
        );
    }
    let work: u64 = rows.iter().map(|r| r.work).sum();
    let boundary: u64 = rows.iter().map(|r| r.boundary).sum();
    let count: u64 = rows.iter().map(|r| r.count).sum();
    out += &format!(
        "| sum | {} | {:.2} | {:.2} | | | 100% |\n",
        count as f64 / passes as f64,
        ms(work),
        ms(boundary)
    );
    out
}

#[cfg(test)]
mod tests;
