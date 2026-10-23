//! The cases of mk's `tests/engine/test_sched.hip` and `test_sched.py` (mk `555a2a3`), in the same order.
//!
//! The stages lie in two fake arenas. The schedule compares addresses only, so nothing reads through them. mk ran the
//! five cases that stop the program in child processes. Here the schedule gives an error, and the test reads it.

use super::*;

/// The fake arenas `A` and `B` of mk's test: 1 << 22 floats each.
const ARENA_A: usize = 0x1000_0000;
const ARENA_B: usize = ARENA_A + (1 << 22) * 4;

fn a(offset: usize) -> *mut f32 {
    std::ptr::without_provenance_mut(ARENA_A + 4 * offset)
}

fn b(offset: usize) -> *mut f32 {
    std::ptr::without_provenance_mut(ARENA_B + 4 * offset)
}

fn stg(op: u32, layer: u32) -> Stage {
    Stage::new(op, layer)
}

fn silu(x: *mut f32, y: *mut f32, n: u32) -> Stage {
    let mut s = stg(ST_SILU_SCALE, 0);
    s.args.silu = SiluScaleArgs { x, y, n, scale: 1.0 };
    s
}

/// An f16 qmm stage: `y[t][n] = W x[t][k]`. With `ks > 1` it writes the partials `[ks][t][n]` to `part`.
fn qmmh(x: *mut f32, y: *mut f32, n: u32, k: u32, t: u32, ks: u32, part: *mut f32) -> Stage {
    let mut s = stg(ST_QMM_H, 0);
    // SAFETY: a test stage. The write fills the member that the op names.
    let q = unsafe { &mut s.args.qmm };
    q.r#type = 8; // Q8_0
    q.n = n;
    q.k = k;
    q.t = t;
    q.x = x;
    q.ldx = k;
    q.y = y;
    q.ldy = n;
    q.ksplit = ks;
    q.part = part;
    s
}

fn ksum_of(split: &Stage) -> Stage {
    let mut s = *split;
    s.op = ST_QMM_KSUM;
    s
}

/// A segment's GDN core at rows `row0..` of the fake arena A.
fn gcore(layer: u32, sub: u32, row0: usize, t: u32, state: *mut f32) -> Stage {
    let mut s = stg(ST_GDN_CORE, layer);
    s.sub = sub;
    let (conv, val, heads) = (GDN_CONV_DIM as usize, GDN_VAL_DIM as usize, GDN_V_HEADS as usize);
    s.args.gcore = GdnCoreArgs {
        qkv: a(row0 * conv),
        z: a(200_000 + row0 * val),
        a: a(300_000 + row0 * heads),
        b: a(301_000 + row0 * heads),
        ssm_a: std::ptr::null(),
        dt_bias: std::ptr::null(),
        gamma: std::ptr::null(),
        state_in: state,
        y: a(310_000 + row0 * val),
        state_out: state,
        t,
    };
    s
}

const MAX_NB: u32 = 16;
const MAX_SEL: u32 = IDX_BUDGET + IDX_RATIO - 1;
const TOPK: u32 = IDX_BUDGET / IDX_RATIO;

/// One segment's attention stages (prep, select, attend).
#[derive(Clone, Copy)]
struct AttnSet {
    prep: Stage,
    sel: Stage,
    att: Stage,
}

/// The attention stages of a segment at rows `r..` and cells `p0..` of the K/V at `kv` and the index keys at `ik`. Its
/// block scores are at `soff`.
#[allow(clippy::too_many_arguments)]
fn attn(layer: u32, sub: u32, r: usize, t: u32, p0: u32, kv: *mut f32, ik: *mut f32, soff: usize) -> AttnSet {
    let mut set = AttnSet {
        prep: stg(ST_ATTN_PREP, layer),
        sel: stg(ST_ATTN_SELECT, layer),
        att: stg(ST_ATTN_CORE, layer),
    };
    set.prep.sub = sub;
    set.sel.sub = sub;
    set.att.sub = sub;
    let sel = b(480_000 + r * MAX_SEL as usize).cast::<i32>();
    let seln = b(520_000 + r).cast::<u32>();
    let null = std::ptr::null();
    set.prep.args.aprep = AttnPrepArgs {
        q_raw: b(r * 12_288),
        k_raw: b(200_000 + r * 512),
        v_raw: b(210_000 + r * 512),
        iq_raw: b(220_000 + r * 512),
        ik_raw: b(230_000 + r * 128),
        q_gamma: null,
        k_gamma: null,
        idxq_gamma: null,
        inv_freq: null,
        t,
        p0,
        q_out: b(240_000 + r * 6144),
        gate_out: b(340_000 + r * 6144),
        iq_out: b(440_000 + r * 512),
        kv_cache: kv,
        ik_cache: ik,
    };
    set.sel.args.asel = AttnSelectArgs {
        iq_out: b(440_000 + r * 512),
        ik_cache: ik,
        idxk_gamma: null,
        inv_freq: null,
        t,
        p0,
        budget: IDX_BUDGET,
        ratio: IDX_RATIO,
        block_scores: b(450_000 + soff),
        block_scores_out: b(460_000 + soff),
        picked: b(470_000 + r * TOPK as usize).cast::<i32>(),
        sel,
        sel_count: seln,
        max_nb: MAX_NB,
        max_sel: MAX_SEL,
        block_topk_max: TOPK,
        pooled: std::ptr::null_mut(),
        pool_b0: 0,
    };
    set.att.args.aatt = AttnAttendArgs {
        q_out: b(240_000 + r * 6144),
        gate_out: b(340_000 + r * 6144),
        kv_cache: kv,
        sel,
        sel_count: seln,
        t,
        max_sel: MAX_SEL,
        attn_out: b(530_000 + r * 6144),
    };
    set
}

/// The pool stage of a segment's attention set: blocks `[b0, (p0 + t) / 4)` of the cache at `pk`.
fn pool_of(set: &AttnSet, pk: *mut f32, b0: u32) -> Stage {
    let mut s = set.sel;
    s.op = ST_ATTN_POOL;
    // SAFETY: a test stage. The write fills the member that the op names.
    let args = unsafe { &mut s.args.asel };
    args.block_scores_out = std::ptr::null_mut();
    args.pooled = pk;
    args.pool_b0 = b0;
    s
}

fn score_of(set: &AttnSet, pooled: *mut f32) -> Stage {
    let mut s = set.sel;
    s.op = ST_ATTN_SCORE;
    // SAFETY: a test stage. The write fills the member that the op names.
    let args = unsafe { &mut s.args.asel };
    args.block_scores_out = std::ptr::null_mut();
    args.pooled = pooled;
    s
}

fn emit_of(set: &AttnSet) -> Stage {
    let mut s = set.sel;
    s.op = ST_ATTN_EMIT;
    s.args.asel.block_scores_out = std::ptr::null_mut();
    s
}

fn nobar(s: &Stage) -> bool {
    s.sched & SCHED_NO_BARRIER != 0
}

fn rot(s: &Stage) -> u32 {
    s.sched & SCHED_ROT_MASK
}

fn tailed(s: &Stage) -> bool {
    s.sched & SCHED_TAIL != 0
}

fn footprint(s: &Stage) -> Footprint {
    stage_footprint(s, 0).expect("a footprint")
}

fn conflict(x: &Stage, y: &Stage) -> bool {
    footprints_conflict(&footprint(x), &footprint(y)).expect("no sibling error")
}

/// Schedule `stages` in place. Stop the test on an error.
fn schedule(stages: &mut Vec<Stage>, windows: &[u32], tail: bool) -> u32 {
    schedule_stages(stages, windows, tail).expect("a schedule")
}

fn silu_x(s: &Stage) -> *const f32 {
    s.silu().x
}

fn silu_y(s: &Stage) -> *mut f32 {
    s.silu().y
}

// ---- the hazard check ----

#[test]
fn the_three_hazards_and_their_edges_conflict() {
    assert!(conflict(&silu(a(0), a(1000), 100), &silu(a(1000), a(2000), 100)), "RAW");
    assert!(conflict(&silu(a(0), a(1000), 100), &silu(a(5000), a(0), 100)), "WAR");
    assert!(conflict(&silu(a(0), a(1000), 100), &silu(a(5000), a(1000), 100)), "WAW");
    assert!(
        conflict(&silu(a(0), a(1000), 100), &silu(a(1099), a(3000), 10)),
        "RAW on the last float of a's output"
    );
    assert!(
        !conflict(&silu(a(0), a(1000), 100), &silu(a(1100), a(3000), 10)),
        "adjacent ranges are independent"
    );
    assert!(
        !conflict(&silu(a(0), a(1000), 100), &silu(a(0), a(2000), 100)),
        "read-read"
    );
}

#[test]
fn split_k_partials_and_fused_consumers_have_their_footprints() {
    let part = a(100_000);
    let s1 = qmmh(a(0), a(10_000), 64, 2560, 1, 4, part);
    let s2 = qmmh(a(0), a(20_000), 64, 2560, 1, 4, part.wrapping_add(256));
    let s2o = qmmh(a(0), a(20_000), 64, 2560, 1, 4, part.wrapping_add(255));
    assert!(!conflict(&s1, &s2), "splits with disjoint partial ranges");
    assert!(conflict(&s1, &s2o), "splits with overlapping partial ranges (WAW)");
    assert!(conflict(&s1, &ksum_of(&s1)), "a sum reads its split's partials");
    assert!(
        conflict(&ksum_of(&s1), &s2o),
        "a split overwriting partials a sum still reads (WAR)"
    );
    assert!(!conflict(&ksum_of(&s1), &s2), "a sum and a split on other partials");
    assert!(
        !conflict(&s1, &silu(a(10_000), a(30_000), 64)),
        "a split doesn't write y"
    );
    assert!(
        conflict(&ksum_of(&s1), &silu(a(10_000), a(30_000), 64)),
        "the sum writes y"
    );
    // A fused consumer (xin) reads the producer's partials, not x.
    let mut c = qmmh(std::ptr::null_mut(), a(40_000), 2560, 64, 1, 1, std::ptr::null_mut());
    c.args.qmm.xin = XIN_SILU_SUM;
    c.args.qmm.xks = 4;
    c.args.qmm.xp = part;
    assert!(conflict(&s1, &c), "a fused consumer reads its producer's partials");
    assert!(
        conflict(&c, &s2o),
        "a split overwriting a fused consumer's partials (WAR)"
    );
    c.args.qmm.xin = XIN_SWIGLU_SUM;
    c.args.qmm.xp2 = part.wrapping_add(1000);
    assert!(
        conflict(&qmmh(a(0), a(20_000), 64, 2560, 1, 4, part.wrapping_add(1000)), &c),
        "the swiglu consumer reads xp2 too"
    );
}

#[test]
fn an_rms_reads_through_the_last_row_of_its_groups() {
    let mut r = stg(ST_RMS, 0);
    r.args.rms = RmsArgs {
        x: a(0),
        ldx: 10_240,
        y: a(50_000),
        ldy: 10_240,
        gamma: a(90_000),
        t: 2,
        n_groups: 4,
        d: 2560,
        gamma_per_group: 1,
        eps: 1e-6,
    };
    assert!(
        conflict(&r, &silu(a(3_000_000), a(20_479), 1)),
        "rms reads row 1's last value (WAR)"
    );
    assert!(
        !conflict(&r, &silu(a(3_000_000), a(20_480), 1)),
        "rms reads nothing past its rows"
    );
}

#[test]
fn hc_mix_writes_inject_only_with_z_and_hc_combine_writes_h() {
    let mut m = stg(ST_HC_MIX, 0);
    m.args.mix = HcMixArgs {
        xn: a(0),
        up: a(20_000),
        mixed: a(40_000),
        z: std::ptr::null(),
        inject: a(60_000),
        t: 1,
        e: 2560,
    };
    assert!(
        !conflict(&m, &silu(a(60_000), a(3_000_000), 4)),
        "the final mixer writes no inject"
    );
    m.args.mix.z = a(70_000);
    assert!(
        conflict(&m, &silu(a(60_000), a(3_000_000), 4)),
        "a mixer with z writes inject"
    );
    let mut cb = stg(ST_HC_COMBINE, 0);
    cb.args.comb = HcCombineArgs {
        h: a(100_000),
        out: a(200_000),
        inject: a(60_000),
        t: 1,
        e: 2560,
    };
    assert!(conflict(&cb, &silu(a(100_000), a(3_000_000), 1)), "hc_combine writes h");
    assert!(
        conflict(&silu(a(3_000_000), a(200_000), 1), &cb),
        "hc_combine reads out"
    );
}

fn combine_rms(t: u32) -> Stage {
    let mut fr = stg(ST_HC_COMBINE_RMS, 0);
    fr.args.combrms = HcCombineRmsArgs {
        c: HcCombineArgs {
            h: a(100_000),
            out: a(200_000),
            inject: a(60_000),
            t,
            e: 2560,
        },
        r: RmsArgs {
            x: a(100_000),
            ldx: 10_240,
            y: a(300_000),
            ldy: 10_240,
            gamma: a(400_000),
            t,
            n_groups: 4,
            d: 2560,
            gamma_per_group: 1,
            eps: 1e-6,
        },
    };
    fr
}

#[test]
fn the_fused_combine_and_rms_has_the_union_of_the_two_footprints() {
    let fr = combine_rms(2);
    let mut cb = stg(ST_HC_COMBINE, 0);
    cb.args.comb = fr.combrms().c;
    let mut r = stg(ST_RMS, 0);
    r.args.rms = fr.combrms().r;
    let fp = footprint(&fr);
    // Every access of `x` is inside an access of the same kind in `y`.
    let cover = |x: &Footprint, y: &Footprint| {
        x.acc
            .iter()
            .all(|p| y.acc.iter().any(|q| p.write == q.write && q.lo <= p.lo && p.hi <= q.hi))
    };
    let mut both = footprint(&cb);
    both.acc.extend(footprint(&r).acc);
    assert!(
        cover(&fp, &both) && cover(&both, &fp),
        "the fused footprint is the two stages' union"
    );
    assert!(
        conflict(&fr, &silu(a(300_000), a(3_000_000), 1)),
        "reads xn (RAW on the fused output)"
    );
    assert!(
        conflict(&silu(a(3_000_000), a(300_000 + 20_479), 1), &fr),
        "writes xn's last value (WAW)"
    );
    assert!(conflict(&fr, &silu(a(100_000), a(3_000_000), 1)), "writes h");
    assert!(
        conflict(&silu(a(3_000_000), a(200_000 + 5119), 1), &fr),
        "reads out's last value (WAR)"
    );
    assert!(
        conflict(&silu(a(3_000_000), a(60_000 + 7), 1), &fr),
        "reads inject's last value (WAR)"
    );
    assert!(!conflict(&fr, &silu(a(3_000_000), a(3_100_000), 1)), "and nothing else");
}

#[test]
fn a_stage_with_no_derived_footprint_is_a_fence() {
    let g = stg(ST_GDN_CORE, 0);
    assert!(
        conflict(&g, &silu(a(3_000_000), a(3_100_000), 1)),
        "a fence conflicts with anything"
    );
    assert!(
        conflict(&silu(a(3_000_000), a(3_100_000), 1), &stg(ST_ATTN_PREP, 0)),
        "a fence conflicts with anything"
    );
}

// ---- the scheduler ----

#[test]
fn independent_stages_share_a_segment_with_a_rotation() {
    let mut s = vec![silu(a(0), a(1000), 512), silu(a(2000), a(3000), 256)];
    let bars = schedule(&mut s, &[], false);
    assert_eq!(bars, 1, "one barrier (after the segment)");
    assert!(
        nobar(&s[0]) && !nobar(&s[1]),
        "no barrier inside the segment, one after it"
    );
    assert!(
        rot(&s[0]) == 0 && rot(&s[1]) == stage_items(&s[0]),
        "rotation {}",
        rot(&s[1])
    );
}

#[test]
fn each_hazard_keeps_its_barrier_and_its_order() {
    for kind in 0..3 {
        let second = match kind {
            0 => silu(a(1000), a(2000), 100),
            1 => silu(a(5000), a(0), 100),
            _ => silu(a(5000), a(1000), 100),
        };
        let mut s = vec![silu(a(0), a(1000), 100), second];
        let before = s.clone();
        let bars = schedule(&mut s, &[], false);
        assert!(
            bars == 2 && !nobar(&s[0]) && !nobar(&s[1]),
            "hazard kind {kind}: barriers {bars}"
        );
        assert!(
            silu_x(&s[0]) == silu_x(&before[0]) && silu_x(&s[1]) == silu_x(&before[1]),
            "hazard kind {kind}: order kept"
        );
    }
}

#[test]
fn an_independent_stage_moves_up_a_level() {
    // A (x -> y), B (y -> z, needs A), C (x -> w, independent) becomes A C | B.
    let mut s = vec![
        silu(a(0), a(1000), 100),
        silu(a(1000), a(2000), 100),
        silu(a(0), a(3000), 100),
    ];
    assert_eq!(schedule(&mut s, &[], false), 2, "A C | B: 2 barriers");
    assert!(
        silu_y(&s[0]) == a(1000) && silu_y(&s[1]) == a(3000) && silu_y(&s[2]) == a(2000),
        "order A C B"
    );
    assert!(
        nobar(&s[0]) && !nobar(&s[1]) && !nobar(&s[2]),
        "barrier after C and B only"
    );
}

#[test]
fn a_war_hazard_across_levels_keeps_the_writer_after_the_reader() {
    // A (x -> y), B (y -> z), C (q -> x). C overwrites A's input, so it stays after A: level 1 with B.
    let mut s = vec![
        silu(a(0), a(1000), 100),
        silu(a(1000), a(2000), 100),
        silu(a(9000), a(0), 100),
    ];
    schedule(&mut s, &[], false);
    assert!(silu_y(&s[0]) == a(1000), "A first");
    assert!(!nobar(&s[0]), "C must not share A's segment (it overwrites A's input)");
}

#[test]
fn a_waw_hazard_across_levels_puts_the_second_writer_last() {
    let mut s = vec![
        silu(a(0), a(1000), 100),
        silu(a(1000), a(2000), 100),
        silu(a(9000), a(2000), 100),
    ];
    let bars = schedule(&mut s, &[], false);
    assert!(
        bars == 3 && silu_x(&s[2]) == a(9000),
        "WAW: C last, 3 barriers (got {bars})"
    );
}

#[test]
fn a_fence_splits_the_stages_around_it() {
    let mut s = vec![
        silu(a(0), a(1000), 100),
        stg(ST_GDN_CORE, 0),
        silu(a(5000), a(6000), 100),
    ];
    let bars = schedule(&mut s, &[], false);
    assert!(
        bars == 3 && s[1].op == ST_GDN_CORE && !nobar(&s[0]) && !nobar(&s[1]),
        "fence: barriers {bars}"
    );
}

#[test]
fn stages_across_a_window_start_are_not_merged_or_reordered() {
    let mut s = vec![
        silu(a(0), a(1000), 100),
        silu(a(2000), a(3000), 100),
        silu(a(4000), a(5000), 100),
    ];
    let bars = schedule(&mut s, &[1], false);
    assert!(
        bars == 2 && !nobar(&s[0]) && nobar(&s[1]),
        "window at 1: barriers {bars}"
    );
}

#[test]
fn split_k_pairs_share_levels_only_with_their_own_partial_ranges() {
    let part = a(100_000);
    let s1 = qmmh(a(0), a(10_000), 64, 2560, 1, 4, part);
    let mut s2 = qmmh(a(0), a(20_000), 64, 2560, 1, 4, part);
    let mut s = vec![s1, ksum_of(&s1), s2, ksum_of(&s2)];
    assert_eq!(schedule(&mut s, &[], false), 4, "one partial buffer: no overlap");
    s2.args.qmm.part = part.wrapping_add(256);
    let mut s = vec![s1, ksum_of(&s1), s2, ksum_of(&s2)];
    assert_eq!(schedule(&mut s, &[], false), 2, "own partial ranges: 2 barriers");
    assert!(
        s[0].op == ST_QMM_H && s[1].op == ST_QMM_H && s[2].op == ST_QMM_KSUM && s[3].op == ST_QMM_KSUM,
        "s1 s2 | k1 k2"
    );
}

// ---- segments' state stages: siblings ----

#[test]
fn sibling_state_stages_share_a_level() {
    // Two segments' GDN cores (same layer, subs 1 and 2, their own rows and states): one level, dealt in turn.
    let mut s = vec![gcore(0, 1, 0, 3, a(500_000)), gcore(0, 2, 3, 5, a(1_300_000))];
    assert!(!conflict(&s[0], &s[1]), "sibling cores on their own rows and states");
    assert_eq!(schedule(&mut s, &[], false), 1, "sibling cores: one level");
    assert!(
        nobar(&s[0]) && !nobar(&s[1]) && rot(&s[1]) == 48,
        "sibling cores: rotation {}",
        rot(&s[1])
    );
    // Untagged (a one-segment pass), they stay fences.
    let mut s = vec![gcore(0, 0, 0, 3, a(500_000)), gcore(0, 0, 3, 5, a(1_300_000))];
    assert_eq!(schedule(&mut s, &[], false), 2, "untagged cores: fences");
    // Other layers, the same segment, or another op: not siblings, so still fences.
    assert!(
        conflict(&gcore(0, 1, 0, 3, a(500_000)), &gcore(4, 2, 3, 5, a(1_300_000))),
        "cores of two layers"
    );
    assert!(
        conflict(&gcore(0, 1, 0, 3, a(500_000)), &gcore(0, 1, 3, 5, a(1_300_000))),
        "one segment's two cores"
    );
    let x = attn(3, 1, 0, 2, 10, b(700_000), b(800_000), 0);
    let y = attn(3, 2, 2, 3, 20, b(900_000), b(1_000_000), 2 * MAX_NB as usize);
    assert!(
        conflict(&gcore(3, 2, 3, 5, a(1_300_000)), &x.prep),
        "a sibling against another op's fence"
    );
    assert!(
        conflict(&x.prep, &y.sel) && conflict(&x.sel, &y.att) && conflict(&x.prep, &x.sel),
        "prep, select, attend stay in order"
    );
    assert!(
        conflict(&silu(a(3_000_000), a(3_100_000), 1), &y.att),
        "a fence against a plain stage"
    );
    // Two segments' attention: prep prep | select select | attend attend.
    let mut s = vec![x.prep, y.prep, x.sel, y.sel, x.att, y.att];
    assert_eq!(schedule(&mut s, &[], false), 3, "sibling attention: three levels");
    let ops: Vec<u32> = s.iter().map(|stage| stage.op).collect();
    assert_eq!(
        ops,
        [
            ST_ATTN_PREP,
            ST_ATTN_PREP,
            ST_ATTN_SELECT,
            ST_ATTN_SELECT,
            ST_ATTN_CORE,
            ST_ATTN_CORE
        ],
        "sibling attention: order kept"
    );
    let bars: Vec<bool> = s.iter().map(nobar).collect();
    assert_eq!(
        bars,
        [true, false, true, false, true, false],
        "a barrier after each pair"
    );
    assert!(
        s[0].sub == 1 && rot(&s[1]) == (2u32 * 33).div_ceil(8),
        "prep rotation {}",
        rot(&s[1])
    );
    assert!(
        rot(&s[3]) == 2 && rot(&s[5]) == 2 * 6,
        "select rotation {}, attend {}",
        rot(&s[3]),
        rot(&s[5])
    );
    // A one-token attend has one item per head.
    let one = attn(3, 1, 0, 1, 10, b(700_000), b(800_000), 0);
    assert!(
        stage_items(&one.att) == HEADS && stage_items(&one.sel) == 1,
        "one-token attend {} items",
        stage_items(&one.att)
    );
    // A fork and a copy of one context at the same n: other buffers, so siblings (the K/V cells don't overlap).
    let c = attn(3, 2, 2, 2, 10, b(900_000), b(1_000_000), 2 * MAX_NB as usize);
    assert!(!conflict(&x.prep, &c.prep), "preps on two K/V buffers");
    let mut pc = stg(ST_PLE_CONV, 1);
    pc.args.pconv = PleConvArgs {
        x: a(0),
        state: a(500_000),
        w: std::ptr::null(),
        y: a(100_000),
        t: 3,
        c: 10_240,
    };
    assert!(
        stage_items(&pc) == 40 && stage_items(&gcore(0, 1, 0, 3, a(500_000))) == 48,
        "ple conv / core items"
    );
}

/// Not in mk's suite: the K/V reach of a segment's attend is the cells `[0, p0 + t)` of its prep. The mutation
/// `cells = p0` survived mk's cases.
#[test]
fn a_segments_attend_reads_the_kv_cells_of_its_prep() {
    let kv = b(700_000);
    let x = attn(3, 1, 0, 2, 10, kv, b(800_000), 0);
    let base = kv as u64;
    let cell = 4 * 2 * u64::from(KV_HEADS) * u64::from(HEAD_DIM);
    let reach = stage_footprint(&x.att, 12).unwrap();
    assert!(
        reach
            .acc
            .iter()
            .any(|access| !access.write && access.lo == base && access.hi == base + 12 * cell),
        "the attend reads cells [0, 12) of its K/V"
    );
    assert!(!reach.unbounded);
    // A second segment's attend that writes its output over cell p0 + t - 1 = 11 of the first one's K/V shares state.
    let mut y = attn(3, 2, 2, 2, 20, b(900_000), b(1_000_000), 2 * MAX_NB as usize);
    y.att.args.aatt.attn_out = kv.wrapping_add(11 * 1024);
    let mut s = vec![x.prep, y.prep, x.att, y.att];
    let error = schedule_stages(&mut s, &[], false).expect_err("the attends share bytes");
    assert!(error.to_string().contains("share state"), "{error}");
    // Over cell 12, past the reach, it shares nothing.
    y.att.args.aatt.attn_out = kv.wrapping_add(12 * 1024);
    let mut s = vec![x.prep, y.prep, x.att, y.att];
    assert_eq!(schedule(&mut s, &[], false), 2, "prep prep | attend attend");
}

// ---- I1: the indexer's select split into score and emit ----

#[test]
fn the_score_and_emit_stages_keep_their_order_and_pair_up_across_segments() {
    let x = attn(3, 1, 0, 2, 10, b(700_000), b(800_000), 0);
    let y = attn(3, 2, 2, 3, 20, b(900_000), b(1_000_000), 2 * MAX_NB as usize);
    let null = std::ptr::null_mut();
    // Items: T * ceil(nb_pass / 32) score items (nb_pass = (p0 + T) / 4), T emit items.
    let mut big = attn(3, 0, 0, 3, 4000, b(700_000), b(800_000), 0);
    assert!(
        stage_items(&score_of(&big, null)) == 3 * 32 && stage_items(&emit_of(&big)) == 3,
        "score {} emit {} items",
        stage_items(&score_of(&big, null)),
        stage_items(&emit_of(&big))
    );
    big.sel.args.asel.p0 = 4000 + 1;
    big.sel.args.asel.t = 131;
    assert_eq!(
        stage_items(&score_of(&big, null)),
        131 * 33,
        "score items at 4132 cells"
    );
    // Score writes the block scores that emit reads: ordered; the scores feed no attend directly.
    assert!(
        conflict(&x.prep, &score_of(&x, null))
            && conflict(&score_of(&x, null), &emit_of(&x))
            && conflict(&emit_of(&x), &x.att),
        "prep, score, emit, attend stay in order"
    );
    assert!(
        !conflict(&score_of(&x, null), &score_of(&y, null)) && !conflict(&emit_of(&x), &emit_of(&y)),
        "two segments' scores and emits are siblings"
    );
    // Two segments: prep prep | score score | emit emit | attend attend.
    let mut s = vec![
        x.prep,
        y.prep,
        score_of(&x, null),
        score_of(&y, null),
        emit_of(&x),
        emit_of(&y),
        x.att,
        y.att,
    ];
    assert_eq!(
        schedule(&mut s, &[], false),
        4,
        "sibling attention with a score stage: four levels"
    );
    assert!(
        s[2].op == ST_ATTN_SCORE && s[3].op == ST_ATTN_SCORE && s[4].op == ST_ATTN_EMIT && s[5].op == ST_ATTN_EMIT,
        "order kept"
    );
    assert!(
        nobar(&s[2]) && !nobar(&s[3]) && nobar(&s[4]) && !nobar(&s[5]),
        "a barrier after each pair"
    );
    // A dense segment (no score stage) beside a sparse one: its emit needs no score level.
    let mut s = vec![
        x.prep,
        y.prep,
        score_of(&y, null),
        emit_of(&x),
        emit_of(&y),
        x.att,
        y.att,
    ];
    assert_eq!(schedule(&mut s, &[], false), 4, "dense + sparse segments");
    // A score of a one-segment pass (sub 0) is a fence, like the stages around it.
    let u = attn(3, 0, 0, 2, 10, b(700_000), b(800_000), 0);
    let mut s = vec![u.prep, score_of(&u, null), emit_of(&u), u.att];
    assert_eq!(schedule(&mut s, &[], false), 4, "untagged: four levels");
}

// ---- I2: the pooled block-key cache's writer (pool) and reader (score) ----

#[test]
fn the_pool_stage_writes_exactly_its_blocks_and_orders_before_the_score() {
    let (pk, pk2) = (b(1_100_000), b(1_900_000));
    let idx = IDX_DIM as usize;
    // p0 = 1000, 10 tokens: the pass completes blocks up to (1000 + 10) / 4 = 252; the cache holds [0, 250).
    let x = attn(3, 1, 0, 10, 1000, b(700_000), b(800_000), 0);
    let pl = pool_of(&x, pk, 250);
    // Footprint, exactly: reads the raw keys of the cells of blocks [250, 252), writes those blocks of the cache.
    let f = footprint(&pl);
    let ik = x.sel.asel().ik_cache as usize;
    let at = |base: usize, floats: usize| (base + 4 * floats) as u64;
    assert_eq!(f.acc.len(), 2, "pool footprint has {} accesses", f.acc.len());
    assert!(
        !f.acc[0].write && f.acc[0].lo == at(ik, 250 * 4 * idx) && f.acc[0].hi == at(ik, 252 * 4 * idx),
        "pool reads the raw keys of the cells of blocks [250, 252)"
    );
    assert!(
        f.acc[1].write && f.acc[1].lo == at(pk as usize, 250 * idx) && f.acc[1].hi == at(pk as usize, 252 * idx),
        "pool writes exactly blocks [250, 252) of the cache"
    );
    // Score now reads the cache for every block of the pass, and not the raw keys.
    let fs = footprint(&score_of(&x, pk));
    let rd_cache = fs
        .acc
        .iter()
        .any(|access| !access.write && access.lo == at(pk as usize, 0) && access.hi == at(pk as usize, 252 * idx));
    let rd_raw = fs.acc.iter().any(|access| !access.write && access.lo == ik as u64);
    assert!(
        rd_cache && !rd_raw,
        "a cached score reads blocks [0, 252) of the cache and not the raw keys"
    );
    // Items: one per 8 blocks (a workgroup's waves); and in the schedule it is a fence between prep and score.
    assert_eq!(stage_items(&pl), 1, "pool items for 2 blocks");
    assert!(
        conflict(&x.prep, &pl) && conflict(&pl, &score_of(&x, pk)) && conflict(&pl, &emit_of(&x)),
        "prep, pool, score stay in order"
    );
    let y = attn(3, 2, 10, 3, 2004, b(900_000), b(1_000_000), 10 * MAX_NB as usize);
    let pb = pool_of(&y, pk2, 500);
    assert!(
        !conflict(&pl, &pb),
        "two segments' pool stages on their own caches are siblings"
    );
    // One segment: prep | pool | score | emit | attend, 5 levels; two segments: the same five, a pair each.
    let u = attn(3, 0, 0, 10, 1000, b(700_000), b(800_000), 0);
    let mut s = vec![u.prep, pool_of(&u, pk, 250), score_of(&u, pk), emit_of(&u), u.att];
    assert_eq!(schedule(&mut s, &[], false), 5, "untagged: five levels");
    let mut s = vec![
        x.prep,
        y.prep,
        pl,
        pb,
        score_of(&x, pk),
        score_of(&y, pk2),
        emit_of(&x),
        emit_of(&y),
        x.att,
        y.att,
    ];
    assert_eq!(schedule(&mut s, &[], false), 5, "two segments: five levels");
    assert!(
        s[2].op == ST_ATTN_POOL && s[3].op == ST_ATTN_POOL && s[4].op == ST_ATTN_SCORE && s[5].op == ST_ATTN_SCORE,
        "order kept"
    );
    assert!(
        nobar(&s[2]) && !nobar(&s[3]) && nobar(&s[4]) && !nobar(&s[5]),
        "a barrier after each pair"
    );
    // A dense segment that completes a block beside a sparse one that does not.
    let mut s = vec![
        x.prep,
        y.prep,
        pb,
        score_of(&x, pk),
        emit_of(&x),
        emit_of(&y),
        x.att,
        y.att,
    ];
    assert_eq!(schedule(&mut s, &[], false), 5, "pool for one segment only");
}

// ---- N3: the barrier tail ----

// The sched bit layout.
const _: () = assert!(SCHED_TAIL == 1 << 30 && SCHED_ROT_MASK == SCHED_TAIL - 1 && SCHED_NO_BARRIER & SCHED_TAIL == 0);

#[test]
fn a_one_token_combine_and_rms_alone_in_its_level_is_a_tail() {
    // T = 1, preceded by its producer: the bit; the producer keeps its barrier and its place.
    let mut s = vec![silu(a(0), a(200_000), 2560), combine_rms(1)];
    assert_eq!(
        schedule(&mut s, &[], true),
        2,
        "tail: the barriers are still counted by the scheduler"
    );
    assert!(
        s[1].op == ST_HC_COMBINE_RMS && tailed(&s[1]) && !tailed(&s[0]) && !nobar(&s[0]) && !nobar(&s[1]),
        "T = 1 alone in its level: tail"
    );
    // Off: no bit anywhere.
    let mut s = vec![silu(a(0), a(200_000), 2560), combine_rms(1)];
    schedule(&mut s, &[], false);
    assert!(!tailed(&s[0]) && !tailed(&s[1]), "enable_tail off: no bit");
    // T = 2 (8 items): no bit.
    let mut s = vec![silu(a(0), a(200_000), 5120), combine_rms(2)];
    schedule(&mut s, &[], true);
    assert!(!tailed(&s[1]), "T = 2: no bit");
    // Stage 0 of the pass: no bit (nothing is before it to be the last arriver's barrier).
    let mut s = vec![combine_rms(1)];
    schedule(&mut s, &[], true);
    assert!(!tailed(&s[0]), "stage 0: no bit");
    // Stage 0 of a later window: the barrier before it is the previous window's last.
    let mut s = vec![silu(a(0), a(200_000), 2560), combine_rms(1)];
    schedule(&mut s, &[1], true);
    assert!(
        tailed(&s[1]),
        "first stage of a later window, alone in its level, after a barrier: tail"
    );
    // Sharing its level with an independent stage: no bit. The reader of out shares level 1.
    let mut s = vec![
        silu(a(0), a(200_000), 2560),
        combine_rms(1),
        silu(a(200_000), a(900_000), 2560),
    ];
    assert_eq!(schedule(&mut s, &[], true), 2, "shared level: two levels");
    assert!(
        s[1].op == ST_HC_COMBINE_RMS && !tailed(&s[1]) && !tailed(&s[2]) && nobar(&s[1]),
        "shares a level: no bit"
    );
    // An op off the allowlist alone in its level: no bit.
    let mut s = vec![silu(a(0), a(200_000), 2560), silu(a(200_000), a(900_000), 2560)];
    schedule(&mut s, &[], true);
    assert!(!tailed(&s[0]) && !tailed(&s[1]), "not on the allowlist: no bit");
    // The rotation and barrier bits are what they were without the tail.
    let mut x = vec![
        silu(a(0), a(200_000), 2560),
        silu(a(50_000), a(70_000), 2560),
        combine_rms(1),
    ];
    let mut y = x.clone();
    schedule(&mut x, &[], false);
    schedule(&mut y, &[], true);
    for (index, (p, q)) in x.iter().zip(&y).enumerate() {
        assert_eq!(q.sched & !SCHED_TAIL, p.sched, "stage {index}");
    }
    assert!(
        tailed(&y[2]) && !tailed(&y[0]) && !tailed(&y[1]),
        "bit only on the tail stage"
    );
}

// ---- N3b: moe_group as a second tail client ----

fn group_of(t: u32) -> Stage {
    let mut g = stg(ST_MOE_GROUP, 0);
    g.args.group = MoeGroupArgs {
        topk_idx: a(500_000).cast::<i32>(),
        t,
        count: a(600_000).cast::<u32>(),
        offset: a(610_000).cast::<u32>(),
        assign: a(620_000).cast::<u32>(),
        token: a(630_000).cast::<u32>(),
        item_gu: a(640_000).cast::<u32>(),
        item_dn: a(650_000).cast::<u32>(),
        fast: 0,
    };
    g
}

#[test]
fn a_one_token_moe_group_alone_in_its_level_is_a_tail() {
    assert_eq!(
        stage_items(&group_of(1)),
        1,
        "moe_group is one item (one workgroup does it all)"
    );
    let mut s = vec![silu(a(0), a(500_000), 10), group_of(1)];
    assert_eq!(schedule(&mut s, &[], true), 2, "group tail: barriers counted");
    assert!(
        s[1].op == ST_MOE_GROUP && tailed(&s[1]) && !tailed(&s[0]) && !nobar(&s[0]) && !nobar(&s[1]),
        "moe_group alone in its level: tail"
    );
    // Sharing its level with an independent stage (the shared expert's down projection): no bit.
    let mut s = vec![
        silu(a(0), a(500_000), 10),
        group_of(1),
        silu(a(500_000), a(800_000), 10),
    ];
    assert_eq!(
        schedule(&mut s, &[], true),
        2,
        "group beside an independent stage: two levels"
    );
    assert!(
        s[1].op == ST_MOE_GROUP && !tailed(&s[1]) && !tailed(&s[2]) && nobar(&s[1]),
        "moe_group shares a level: no bit"
    );
    // T = 2: no bit (the allowlist bounds moe_group to T = 1).
    let mut s = vec![silu(a(0), a(500_000), 20), group_of(2)];
    schedule(&mut s, &[], true);
    assert!(!tailed(&s[1]), "moe_group at T = 2: no bit");
    // Stage 0: no bit. Tail off: no bit.
    let mut s = vec![group_of(1)];
    schedule(&mut s, &[], true);
    assert!(!tailed(&s[0]), "moe_group as stage 0: no bit");
    let mut s = vec![silu(a(0), a(500_000), 10), group_of(1)];
    schedule(&mut s, &[], false);
    assert!(!tailed(&s[1]), "enable_tail off: no bit on moe_group");
}

// ---- test_sched.py: the siblings that touch the same bytes stop the schedule ----

/// The error of one case of mk's `die_case`. A schedule that succeeds fails the test.
fn die_case(stages: Vec<Stage>) -> String {
    let mut stages = stages;
    match schedule_stages(&mut stages, &[], false) {
        Ok(_) => panic!("the schedule took siblings that share state"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn siblings_that_touch_the_same_bytes_stop_the_schedule() {
    let max_nb = 2 * MAX_NB as usize;
    // Two forks of one context: the same cells [p0, p0 + t) of one K/V buffer.
    let x = attn(3, 1, 0, 2, 10, b(700_000), b(800_000), 0);
    let y = attn(3, 2, 2, 2, 10, b(700_000), b(800_000), max_nb);
    let kv = die_case(vec![x.prep, y.prep]);
    // Two segments' cores on overlapping rows, and on one recurrent state.
    let rows = die_case(vec![gcore(0, 1, 0, 3, a(500_000)), gcore(0, 2, 2, 3, a(1_300_000))]);
    let state = die_case(vec![gcore(0, 1, 0, 3, a(500_000)), gcore(0, 2, 3, 3, a(500_000))]);
    // A segment's attend without its prep in the window: its K/V reach is unknown.
    let x = attn(3, 1, 0, 2, 10, b(700_000), b(800_000), 0);
    let y = attn(3, 2, 2, 2, 20, b(900_000), b(1_000_000), max_nb);
    let attend = die_case(vec![x.prep, x.att, y.att]);
    // Two segments' pool stages writing blocks of one pooled-key buffer: blocks [2, 3) and [2, 5).
    let pool = die_case(vec![pool_of(&x, b(1_100_000), 2), pool_of(&y, b(1_100_000), 2)]);
    for (case, error, word) in [
        ("die-kv", kv, "share"),
        ("die-rows", rows, "share"),
        ("die-state", state, "share"),
        ("die-attend", attend, "prep"),
        ("die-pool", pool, "share"),
    ] {
        assert!(
            error.contains(word),
            "{case}: the error does not name the reason: {error}"
        );
    }
}

// ---- the calls of the engine (call.rs) ----

#[test]
fn with_no_schedule_each_stage_runs_in_build_order_with_a_barrier_after_it() {
    // MK_ELIDE=0: a reorder and a tail stay as they were built. A (x -> y), B (y -> out), C (x -> w), E (reads out).
    let stages = vec![
        silu(a(0), a(1000), 100),
        silu(a(1000), a(200_000), 2560),
        silu(a(0), a(10_000), 100),
        combine_rms(1),
    ];
    let off = call::pass_plan(&stages, &[], false, true);
    assert_eq!(off.order, [0, 1, 2, 3]);
    assert_eq!(
        off.sched, [0; 4],
        "a grid barrier after each stage, no rotation and no tail"
    );
    assert_eq!(off.barriers, 4);
    let on = call::pass_plan(&stages, &[], true, true);
    assert_eq!(on.order, [0, 2, 1, 3], "A C | B | E");
    assert_eq!(on.sched[3], SCHED_TAIL, "E is a tail");
    assert_eq!(on.barriers, 3);
}
