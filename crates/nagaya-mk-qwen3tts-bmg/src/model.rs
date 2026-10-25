//! The passes of the engine on its card: the prefill of a prompt, one frame, and the codec decode of the frames of a
//! quantum. Each pass is one ring of tasks (`nagaya_mk_bmg::ring::Ring`), so the 16 samples of a frame stay on the card
//! (engine-abi "Speech").
//!
//! The reference is `Qwen3TTSForConditionalGeneration.generate` with `non_streaming_mode=True` and
//! `Qwen3TTSTokenizerV2Decoder.forward`. The codec decodes in quanta and keeps the state of each causal layer, so the
//! audio of the quanta equals the audio of one decode of every frame.

use nagaya_abi::PromptPair;
use nagaya_mk_bmg::protocol::{MkTaskDescriptor, NO_EVENT};
use nagaya_mk_bmg::tasks as t;

use crate::spec::*;
use crate::weights::{Layer, Weights};

/// The work buffers that every pass of the engine shares. One pass runs at a time.
pub struct Work {
    pub ids_text: u64,
    pub ids_codec: u64,
    pub x: u64,
    pub nrm: u64,
    pub q: u64,
    pub att: u64,
    pub mlp: u64,
    pub tmp: u64,
    pub scores: u64,
    pub logits: u64,
    /// The logits of each codebook of the code predictor, `[15][2048]`; a forced run reads them.
    pub cp_logits: u64,
    pub cpx: u64,
    pub cp_nrm: u64,
    pub cp_q: u64,
    pub cp_att: u64,
    pub cp_mlp: u64,
    pub cp_tmp: u64,
    pub cp_scores: u64,
    /// The KV of the code predictor: 5 layers of 16 positions, fresh for each frame.
    pub cp_kv: u64,
    pub pad_row: u64,
    pub zero_id: u64,
    pub pad_id: u64,
    pub shift: u64,
    pub dx: u64,
    pub dn: u64,
    pub dq: u64,
    pub datt: u64,
    pub dg: u64,
    pub du: u64,
    pub dtmp: u64,
    pub dscores: u64,
    pub dh: u64,
    pub ud: u64,
    pub ubig: u64,
    pub ures: u64,
    pub t1: u64,
    pub wav: u64,
    pub pcm: u64,
}

/// The state of one request on the card.
pub struct Slot {
    /// The talker's KV: for each layer, the K plane of `max_positions` rows, then the V plane.
    pub kv: u64,
    /// The codes of each frame, `[max_frames + 1][16]` u32. The first codebook of a frame comes from the talker step
    /// before it.
    pub codes: u64,
    /// The sampler state of codebook 0: seed, draws, history (`OP_SAMPLE`).
    pub state: u64,
    /// The sampler state of the code predictor, with no history.
    pub cp_state: u64,
    /// The talker's last hidden state after its norm, which the code predictor takes.
    pub hidden: u64,
    /// The codec's KV: for each of 8 layers, the K plane of `max_frames` rows, then the V plane.
    pub dec_kv: u64,
    /// The inputs of the causal layers of the codec, each with the state rows of the quantum before at its start.
    pub pc_in: u64,
    pub up_in: [u64; 2],
    pub dw_in: [u64; 2],
    pub dec_in: u64,
    pub ct_in: [u64; 4],
    pub ru_in: [[u64; 3]; 4],
    pub fs: u64,
}

/// Bytes of `n` f32 values.
fn f(n: usize) -> usize {
    n * 4
}

/// The input steps of decoder block `b` for `frames` frames.
fn block_steps(b: usize, frames: usize) -> usize {
    upsampled(frames) * UPSAMPLE_RATES[..b].iter().product::<usize>()
}

fn steps_final(frames: usize) -> usize {
    frames * SAMPLES_PER_FRAME
}

/// The bytes that an allocation of `bytes` takes on the card: at least 64, a multiple of 64.
pub fn rounded(bytes: usize) -> usize {
    bytes.max(64).next_multiple_of(64)
}

/// The device bytes that `alloc` would get for the buffers of `build`, with no card: the plan (spec 2.4).
pub fn bytes_of<T>(build: impl FnOnce(&mut dyn FnMut(usize) -> u64) -> T) -> usize {
    let mut total = 0usize;
    build(&mut |bytes| {
        total += rounded(bytes);
        0
    });
    total
}

impl Work {
    /// The work buffers, each from `z`, which gives zeroed device memory of a byte count.
    pub fn new(z: &mut dyn FnMut(usize) -> u64, config: &Config) -> Self {
        let p = config.max_prompt as usize;
        let q = config.frames_per_quantum as usize;
        let steps = steps_final(q);
        let ures = (0..UPSAMPLE_RATES.len())
            .map(|b| block_steps(b + 1, q) * block_in(b + 1))
            .max()
            .unwrap_or(0);
        Self {
            ids_text: z(p * 4),
            ids_codec: z(p * 4),
            x: z(f(p * HIDDEN)),
            nrm: z(f(p * HIDDEN)),
            q: z(f(p * HEADS * HEAD_DIM)),
            att: z(f(p * HEADS * HEAD_DIM)),
            mlp: z(f(p * INTER)),
            tmp: z(f(p * HIDDEN)),
            scores: z(f(t::score_floats(HEADS as u32, config.max_prompt, 0))),
            logits: z(f(CODEC_VOCAB)),
            cp_logits: z(f((CODEBOOKS - 1) * CP_VOCAB)),
            cpx: z(f(CODEBOOKS * CP_HIDDEN)),
            cp_nrm: z(f(CODEBOOKS * CP_HIDDEN)),
            cp_q: z(f(CODEBOOKS * HEADS * HEAD_DIM)),
            cp_att: z(f(CODEBOOKS * HEADS * HEAD_DIM)),
            cp_mlp: z(f(CODEBOOKS * CP_INTER)),
            cp_tmp: z(f(HIDDEN)),
            cp_scores: z(f(t::score_floats(HEADS as u32, CODEBOOKS as u32, 0))),
            cp_kv: z(f(CP_LAYERS * 2 * CODEBOOKS * KV_HEADS * HEAD_DIM)),
            pad_row: z(f(HIDDEN)),
            zero_id: z(4),
            pad_id: z(4),
            shift: z(f(64 * 1024)),
            dx: z(f(q * DEC_HIDDEN)),
            dn: z(f(q * DEC_HIDDEN)),
            dq: z(f(q * DEC_HEADS * DEC_HEAD_DIM)),
            datt: z(f(q * DEC_HEADS * DEC_HEAD_DIM)),
            dg: z(f(q * DEC_INTER)),
            du: z(f(q * DEC_INTER)),
            dtmp: z(f(q * DEC_HIDDEN)),
            dscores: z(f(t::score_floats(
                DEC_HEADS as u32,
                config.frames_per_quantum,
                config.max_frames,
            ))),
            dh: z(f(q * LATENT)),
            ud: z(f(upsampled(q) * LATENT)),
            ubig: z(f(upsampled(q) * 4 * LATENT)),
            ures: z(f(ures)),
            t1: z(f(ures)),
            wav: z(f(steps)),
            pcm: z(steps * 2),
        }
    }
}

impl Slot {
    /// The buffers of one slot, each from `z`.
    pub fn new(z: &mut dyn FnMut(usize) -> u64, config: &Config) -> Self {
        let q = config.frames_per_quantum as usize;
        let frames = config.max_frames as usize;
        let kv = z(f(LAYERS * 2 * config.max_positions() as usize * KV_HEADS * HEAD_DIM));
        let codes = z((frames + 1) * CODEBOOKS * 4);
        let state = z((4 + frames + 1) * 4);
        let cp_state = z(4 * 4);
        let hidden = z(f(HIDDEN));
        let dec_kv = z(f(DEC_LAYERS * 2 * frames * DEC_HEADS * DEC_HEAD_DIM));
        let pc_in = z(f((2 + q) * CODEBOOK_DIM));
        let up_in = [z(f((1 + q) * LATENT)), z(f((1 + 2 * q) * LATENT))];
        let dw_in = [z(f((6 + 2 * q) * LATENT)), z(f((6 + 4 * q) * LATENT))];
        let dec_in = z(f((6 + upsampled(q)) * LATENT));
        let mut ct_in = [0u64; 4];
        let mut ru_in = [[0u64; 3]; 4];
        for b in 0..UPSAMPLE_RATES.len() {
            ct_in[b] = z(f((1 + block_steps(b, q)) * block_in(b)));
            for (u, d) in [1usize, 3, 9].into_iter().enumerate() {
                ru_in[b][u] = z(f((6 * d + block_steps(b + 1, q)) * block_in(b + 1)));
            }
        }
        let fs = z(f((6 + steps_final(q)) * block_in(UPSAMPLE_RATES.len())));
        Self {
            kv,
            codes,
            state,
            cp_state,
            hidden,
            dec_kv,
            pc_in,
            up_in,
            dw_in,
            dec_in,
            ct_in,
            ru_in,
            fs,
        }
    }

    /// The address of code `g` of frame `frame`.
    pub fn code(&self, frame: usize, g: usize) -> u64 {
        self.codes + ((frame * CODEBOOKS + g) * 4) as u64
    }

    /// The codec's state rows, which `clear_codec` zeroes for a new request: each buffer and its bytes.
    pub fn codec_state(&self, config: &Config) -> Vec<(u64, usize)> {
        let q = config.frames_per_quantum as usize;
        let mut out = vec![
            (self.pc_in, f((2 + q) * CODEBOOK_DIM)),
            (self.up_in[0], f((1 + q) * LATENT)),
            (self.up_in[1], f((1 + 2 * q) * LATENT)),
            (self.dw_in[0], f((6 + 2 * q) * LATENT)),
            (self.dw_in[1], f((6 + 4 * q) * LATENT)),
            (self.dec_in, f((6 + upsampled(q)) * LATENT)),
        ];
        for b in 0..UPSAMPLE_RATES.len() {
            out.push((self.ct_in[b], f((1 + block_steps(b, q)) * block_in(b))));
            for (u, d) in [1usize, 3, 9].into_iter().enumerate() {
                out.push((self.ru_in[b][u], f((6 * d + block_steps(b + 1, q)) * block_in(b + 1))));
            }
        }
        out.push((self.fs, f((6 + steps_final(q)) * block_in(UPSAMPLE_RATES.len()))));
        out
    }
}

/// The offset of row `r` of rows of `width` f32 values.
fn row(base: u64, r: usize, width: usize) -> u64 {
    base + (r * width * 4) as u64
}

/// The tasks of one pass.
#[derive(Default)]
pub struct Tasks(pub Vec<MkTaskDescriptor>);

impl Tasks {
    fn push(&mut self, task: MkTaskDescriptor) {
        self.0.push(task);
    }

    fn extend<const N: usize>(&mut self, tasks: [MkTaskDescriptor; N]) {
        self.0.extend(tasks);
    }

    /// Move the last `state` rows of the `state + new` rows of `buf` to its start, through `tmp`, so the next quantum
    /// finds them as its state.
    fn shift(&mut self, buf: u64, tmp: u64, state: usize, new: usize, width: usize) {
        let words = (state * width) as u32;
        self.push(t::copy(row(buf, new, width), tmp, words));
        self.push(t::copy(tmp, buf, words));
    }
}

/// The buffers of one decoder layer of the talker or the code predictor.
struct Bufs {
    x: u64,
    nrm: u64,
    q: u64,
    att: u64,
    mlp: u64,
    scores: u64,
}

/// One layer of Qwen3 over `n` tokens at positions `p0..`, with its KV at `kv` (`max_seq` rows a plane).
#[allow(clippy::too_many_arguments)]
fn qwen3_layer(
    out: &mut Tasks,
    w: &Layer,
    b: &Bufs,
    kv: u64,
    max_seq: usize,
    n: usize,
    p0: usize,
    hidden: usize,
    inter: usize,
) {
    let (n32, h32) = (n as u32, hidden as u32);
    let qd = (HEADS * HEAD_DIM) as u32;
    let kvd = KV_HEADS * HEAD_DIM;
    let kplane = row(kv, p0, kvd);
    let vplane = row(kv + (max_seq * kvd * 4) as u64, p0, kvd);
    out.push(t::rmsnorm(b.x, w.ln1, b.nrm, n32, h32, EPS, false));
    out.push(t::matmul(b.nrm, w.q, b.q, n32, qd, h32, false));
    out.push(t::matmul(b.nrm, w.k, kplane, n32, kvd as u32, h32, false));
    out.push(t::matmul(b.nrm, w.v, vplane, n32, kvd as u32, h32, false));
    let scale = (HEAD_DIM as f32).powf(-0.5);
    out.push(t::qk_rope(
        b.q,
        w.q_norm,
        n32 * HEADS as u32,
        HEADS as u32,
        HEAD_DIM as u32,
        p0 as u32,
        true,
        true,
        EPS,
        scale,
    ));
    out.push(t::qk_rope(
        kplane,
        w.k_norm,
        n32 * KV_HEADS as u32,
        KV_HEADS as u32,
        HEAD_DIM as u32,
        p0 as u32,
        true,
        true,
        EPS,
        1.0,
    ));
    if n == 1 {
        out.extend(t::attend_one(
            b.q,
            kv,
            b.att,
            HEADS as u32,
            KV_HEADS as u32,
            HEAD_DIM as u32,
            p0 as u32,
            max_seq as u32,
            0,
        ));
    } else {
        out.extend(t::attend_block(
            b.q,
            kv,
            b.scores,
            b.att,
            n32,
            HEADS as u32,
            KV_HEADS as u32,
            HEAD_DIM as u32,
            p0 as u32,
            max_seq as u32,
            0,
        ));
    }
    out.push(t::matmul(b.att, w.o, b.x, n32, h32, qd, true));
    out.push(t::rmsnorm(b.x, w.ln2, b.nrm, n32, h32, EPS, false));
    out.push(t::swiglu(b.nrm, w.gate_up, b.mlp, n32, inter as u32, h32));
    out.push(t::matmul(b.mlp, w.down, b.x, n32, h32, inter as u32, true));
}

/// The sampler settings of a request.
#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: u32,
    pub penalty: f32,
}

/// The ring that projects the text id `tts_pad` into `pad_row`: the trailing text of each frame.
pub fn pad_row(wt: &Weights, work: &Work) -> Tasks {
    let mut out = Tasks::default();
    let h = HIDDEN as u32;
    out.push(t::embed(work.pad_id, wt.text_embed, work.tmp, 1, h, 1, false, false));
    out.push(t::matmul(work.tmp, wt.fc1_w, work.nrm, 1, h, h, false));
    out.push(t::bias(wt.fc1_b, work.nrm, 1, h, false));
    out.push(t::silu_mul(work.nrm, 0, work.nrm, 1, h));
    out.push(t::matmul(work.nrm, wt.fc2_w, work.pad_row, 1, h, h, false));
    out.push(t::bias(wt.fc2_b, work.pad_row, 1, h, false));
    out
}

/// The ids of a prompt for the card: absent ids are `NO_EVENT`.
pub fn prompt_ids(prompt: &[PromptPair]) -> (Vec<u8>, Vec<u8>) {
    let text = prompt
        .iter()
        .flat_map(|pair| pair.text.unwrap_or(NO_EVENT).to_le_bytes())
        .collect();
    let codec = prompt
        .iter()
        .flat_map(|pair| pair.codec.unwrap_or(NO_EVENT).to_le_bytes())
        .collect();
    (text, codec)
}

/// The rows of `n` prompt positions into `work.x`: text_projection(text_embedding(text)) + codec_embedding(codec),
/// each part where its id is present (`generate`, non-streaming). An absent text adds nothing.
pub fn prompt_rows(wt: &Weights, work: &Work, n: usize) -> Tasks {
    let mut out = Tasks::default();
    let (n32, h) = (n as u32, HIDDEN as u32);
    out.push(t::embed(
        work.ids_text,
        wt.text_embed,
        work.tmp,
        n32,
        h,
        1,
        false,
        false,
    ));
    out.push(t::matmul(work.tmp, wt.fc1_w, work.nrm, n32, h, h, false));
    out.push(t::bias(wt.fc1_b, work.nrm, n32, h, false));
    out.push(t::silu_mul(work.nrm, 0, work.nrm, n32, h));
    out.push(t::matmul(work.nrm, wt.fc2_w, work.x, n32, h, h, false));
    out.push(t::bias(wt.fc2_b, work.x, n32, h, false));
    out.push(t::mask_rows(work.ids_text, work.x, n32, h));
    out.push(t::embed(work.ids_codec, wt.codec_embed, work.x, n32, h, 1, false, true));
    out
}

/// The prefill of `n` prompt positions (whose ids are in `work.ids_text` and `work.ids_codec`), then the talker's
/// logits of the first frame's codebook 0, and its sample unless the run is forced.
pub fn prefill(wt: &Weights, work: &Work, slot: &Slot, config: &Config, n: usize, sampling: Option<Sampling>) -> Tasks {
    let mut out = prompt_rows(wt, work, n);
    let b = Bufs {
        x: work.x,
        nrm: work.nrm,
        q: work.q,
        att: work.att,
        mlp: work.mlp,
        scores: work.scores,
    };
    let max_seq = config.max_positions() as usize;
    let plane = 2 * max_seq * KV_HEADS * HEAD_DIM;
    for (l, w) in wt.layers.iter().enumerate() {
        qwen3_layer(
            &mut out,
            w,
            &b,
            slot.kv + (l * plane * 4) as u64,
            max_seq,
            n,
            0,
            HIDDEN,
            INTER,
        );
    }
    talker_head(&mut out, wt, work, slot, row(work.x, n - 1, HIDDEN), 0, sampling);
    out
}

/// The final norm of the talker's last row into the slot's hidden state, the codec head, and the sample of
/// codebook 0 of frame `frame`.
fn talker_head(
    out: &mut Tasks,
    wt: &Weights,
    work: &Work,
    slot: &Slot,
    last: u64,
    frame: usize,
    sampling: Option<Sampling>,
) {
    let h = HIDDEN as u32;
    out.push(t::rmsnorm(last, wt.norm, slot.hidden, 1, h, EPS, false));
    out.push(t::matmul(
        slot.hidden,
        wt.head,
        work.logits,
        1,
        CODEC_VOCAB as u32,
        h,
        false,
    ));
    if let Some(s) = sampling {
        out.push(t::sample(
            work.logits,
            slot.state,
            slot.code(frame, 0),
            CODEC_VOCAB as u32,
            s.top_k,
            s.temperature,
            s.penalty,
            SUPPRESS_FROM,
            CODEC_EOS,
            MIN_NEW,
            true,
        ));
    }
}

/// One frame: the code predictor samples codebooks 1 to 15 of frame `frame` from the talker's hidden state and its
/// codebook 0, then the talker takes the sum of the 16 code embeddings and the trailing text at position
/// `prompt + frame`, and samples codebook 0 of the next frame. A forced run samples nothing: the codes are in place.
pub fn frame(
    wt: &Weights,
    work: &Work,
    slot: &Slot,
    config: &Config,
    prompt: usize,
    frame: usize,
    sampling: Option<Sampling>,
) -> Tasks {
    let mut out = Tasks::default();
    let (h, ch) = (HIDDEN as u32, CP_HIDDEN as u32);
    let cp = Bufs {
        x: work.cpx,
        nrm: work.cp_nrm,
        q: work.cp_q,
        att: work.cp_att,
        mlp: work.cp_mlp,
        scores: work.cp_scores,
    };
    let cp_plane = 2 * CODEBOOKS * KV_HEADS * HEAD_DIM;
    // Positions 0 and 1 of the code predictor: the talker's hidden state and the embedding of codebook 0.
    out.push(t::matmul(slot.hidden, wt.cp_proj_w, work.cpx, 1, ch, h, false));
    out.push(t::bias(wt.cp_proj_b, work.cpx, 1, ch, false));
    out.push(t::embed(
        slot.code(frame, 0),
        wt.codec_embed,
        work.cp_tmp,
        1,
        h,
        1,
        false,
        false,
    ));
    out.push(t::matmul(
        work.cp_tmp,
        wt.cp_proj_w,
        row(work.cpx, 1, CP_HIDDEN),
        1,
        ch,
        h,
        false,
    ));
    out.push(t::bias(wt.cp_proj_b, row(work.cpx, 1, CP_HIDDEN), 1, ch, false));
    for (l, w) in wt.cp_layers.iter().enumerate() {
        qwen3_layer(
            &mut out,
            w,
            &cp,
            work.cp_kv + (l * cp_plane * 4) as u64,
            CODEBOOKS,
            2,
            0,
            CP_HIDDEN,
            CP_INTER,
        );
    }
    cp_head(
        &mut out,
        wt,
        work,
        slot,
        frame,
        1,
        row(work.cpx, 1, CP_HIDDEN),
        sampling,
    );
    // Codes 1 to 14 go back in, each with its own embedding, at positions 2 to 15. Code 15 does not.
    for g in 1..CODEBOOKS - 1 {
        let x = row(work.cpx, g + 1, CP_HIDDEN);
        out.push(t::embed(
            slot.code(frame, g),
            wt.cp_embeds[g - 1],
            work.cp_tmp,
            1,
            h,
            1,
            false,
            false,
        ));
        out.push(t::matmul(work.cp_tmp, wt.cp_proj_w, x, 1, ch, h, false));
        out.push(t::bias(wt.cp_proj_b, x, 1, ch, false));
        let step = Bufs { x, ..cp };
        for (l, w) in wt.cp_layers.iter().enumerate() {
            qwen3_layer(
                &mut out,
                w,
                &step,
                work.cp_kv + (l * cp_plane * 4) as u64,
                CODEBOOKS,
                1,
                g + 1,
                CP_HIDDEN,
                CP_INTER,
            );
        }
        cp_head(&mut out, wt, work, slot, frame, g + 1, x, sampling);
    }
    // The talker's input: the 16 code embeddings and the trailing text, `tts_pad` in non-streaming mode.
    out.push(t::embed(
        slot.code(frame, 0),
        wt.codec_embed,
        work.x,
        1,
        h,
        1,
        false,
        false,
    ));
    for g in 1..CODEBOOKS {
        out.push(t::embed(
            slot.code(frame, g),
            wt.cp_embeds[g - 1],
            work.x,
            1,
            h,
            1,
            false,
            true,
        ));
    }
    out.push(t::embed(work.zero_id, work.pad_row, work.x, 1, h, 1, true, true));
    let b = Bufs {
        x: work.x,
        nrm: work.nrm,
        q: work.q,
        att: work.att,
        mlp: work.mlp,
        scores: work.scores,
    };
    let max_seq = config.max_positions() as usize;
    let plane = 2 * max_seq * KV_HEADS * HEAD_DIM;
    for (l, w) in wt.layers.iter().enumerate() {
        qwen3_layer(
            &mut out,
            w,
            &b,
            slot.kv + (l * plane * 4) as u64,
            max_seq,
            1,
            prompt + frame,
            HIDDEN,
            INTER,
        );
    }
    talker_head(&mut out, wt, work, slot, work.x, frame + 1, sampling);
    out
}

/// The norm, the head of codebook `g` and its sample, from the row `x` of the code predictor.
#[allow(clippy::too_many_arguments)]
fn cp_head(
    out: &mut Tasks,
    wt: &Weights,
    work: &Work,
    slot: &Slot,
    frame: usize,
    g: usize,
    x: u64,
    sampling: Option<Sampling>,
) {
    let logits = row(work.cp_logits, g - 1, CP_VOCAB);
    out.push(t::rmsnorm(x, wt.cp_norm, work.cp_nrm, 1, CP_HIDDEN as u32, EPS, false));
    out.push(t::matmul(
        work.cp_nrm,
        wt.cp_heads[g - 1],
        logits,
        1,
        CP_VOCAB as u32,
        CP_HIDDEN as u32,
        false,
    ));
    if let Some(s) = sampling {
        // The code predictor's processors are temperature and top-k only (the reference's `subtalker_*`).
        out.push(t::sample(
            logits,
            slot.cp_state,
            slot.code(frame, g),
            CP_VOCAB as u32,
            s.top_k,
            s.temperature,
            1.0,
            CP_VOCAB as u32,
            0,
            0,
            false,
        ));
    }
}

/// The codec decode of `n` frames from frame `first`, whose codes are in the slot; `done` frames went before them.
/// The audio goes to `work.pcm`: `n * 1920` samples.
pub fn codec(wt: &Weights, work: &Work, slot: &Slot, config: &Config, first: usize, n: usize, done: usize) -> Tasks {
    let c = &wt.codec;
    let mut out = Tasks::default();
    let n32 = n as u32;
    // The quantizer: the sum of the 16 projected codebook rows.
    for q in 0..CODEBOOKS {
        let table = c.tables + (q * CODEBOOK_SIZE * CODEBOOK_DIM * 4) as u64;
        out.push(t::embed(
            slot.code(first, q),
            table,
            row(slot.pc_in, 2, CODEBOOK_DIM),
            n32,
            CODEBOOK_DIM as u32,
            CODEBOOKS as u32,
            true,
            q > 0,
        ));
    }
    out.push(t::conv1d(
        slot.pc_in,
        c.pre_conv_w,
        c.pre_conv_b,
        work.dh,
        2 + n32,
        CODEBOOK_DIM as u32,
        LATENT as u32,
        3,
        1,
        1,
        2,
        false,
    ));
    out.shift(slot.pc_in, work.shift, 2, n, CODEBOOK_DIM);
    // The transformer over the frames, with its KV at the absolute frame positions and a window of 72.
    let (dh, lat) = (DEC_HIDDEN as u32, LATENT as u32);
    let qd = DEC_HEADS * DEC_HEAD_DIM;
    out.push(t::conv1d(
        work.dh,
        c.in_proj_w,
        c.in_proj_b,
        work.dx,
        n32,
        lat,
        dh,
        1,
        1,
        1,
        0,
        false,
    ));
    let max_frames = config.max_frames as usize;
    for (l, w) in c.layers.iter().enumerate() {
        let kv = slot.dec_kv + (l * 2 * max_frames * qd * 4) as u64;
        let kplane = row(kv, done, qd);
        let vplane = row(kv + (max_frames * qd * 4) as u64, done, qd);
        out.push(t::rmsnorm(work.dx, w.ln1, work.dn, n32, dh, DEC_EPS, true));
        out.push(t::conv1d(
            work.dn, w.q, 0, work.dq, n32, dh, qd as u32, 1, 1, 1, 0, false,
        ));
        out.push(t::conv1d(
            work.dn, w.k, 0, kplane, n32, dh, qd as u32, 1, 1, 1, 0, false,
        ));
        out.push(t::conv1d(
            work.dn, w.v, 0, vplane, n32, dh, qd as u32, 1, 1, 1, 0, false,
        ));
        let rows = n32 * DEC_HEADS as u32;
        let scale = (DEC_HEAD_DIM as f32).powf(-0.5);
        out.push(t::qk_rope(
            work.dq,
            0,
            rows,
            DEC_HEADS as u32,
            DEC_HEAD_DIM as u32,
            done as u32,
            false,
            false,
            DEC_EPS,
            scale,
        ));
        out.push(t::qk_rope(
            kplane,
            0,
            rows,
            DEC_HEADS as u32,
            DEC_HEAD_DIM as u32,
            done as u32,
            false,
            false,
            DEC_EPS,
            1.0,
        ));
        let (heads, hd, p0, ms, win) = (
            DEC_HEADS as u32,
            DEC_HEAD_DIM as u32,
            done as u32,
            max_frames as u32,
            DEC_WINDOW as u32,
        );
        if n == 1 {
            out.extend(t::attend_one(work.dq, kv, work.datt, heads, heads, hd, p0, ms, win));
        } else {
            out.extend(t::attend_block(
                work.dq,
                kv,
                work.dscores,
                work.datt,
                n32,
                heads,
                heads,
                hd,
                p0,
                ms,
                win,
            ));
        }
        out.push(t::conv1d(
            work.datt, w.o, 0, work.dtmp, n32, qd as u32, dh, 1, 1, 1, 0, false,
        ));
        out.push(t::scale_add(work.dtmp, w.attn_scale, work.dx, n32, dh));
        out.push(t::rmsnorm(work.dx, w.ln2, work.dn, n32, dh, DEC_EPS, true));
        out.push(t::conv1d(
            work.dn,
            w.gate,
            0,
            work.dg,
            n32,
            dh,
            DEC_INTER as u32,
            1,
            1,
            1,
            0,
            false,
        ));
        out.push(t::conv1d(
            work.dn,
            w.up,
            0,
            work.du,
            n32,
            dh,
            DEC_INTER as u32,
            1,
            1,
            1,
            0,
            false,
        ));
        out.push(t::silu_mul(work.dg, work.du, work.dg, n32, DEC_INTER as u32));
        out.push(t::conv1d(
            work.dg,
            w.down,
            0,
            work.dtmp,
            n32,
            DEC_INTER as u32,
            dh,
            1,
            1,
            1,
            0,
            false,
        ));
        out.push(t::scale_add(work.dtmp, w.mlp_scale, work.dx, n32, dh));
    }
    out.push(t::rmsnorm(work.dx, c.norm, work.dn, n32, dh, DEC_EPS, true));
    out.push(t::conv1d(
        work.dn,
        c.out_proj_w,
        c.out_proj_b,
        row(slot.up_in[0], 1, LATENT),
        n32,
        dh,
        lat,
        1,
        1,
        1,
        0,
        false,
    ));
    // Two upsampling stages: a transposed convolution, then a ConvNeXt block.
    for (i, (u, ratio)) in c.ups.iter().zip(UPSAMPLING_RATIOS).enumerate() {
        let rows = n << i;
        let out_rows = (rows * ratio) as u32;
        out.push(t::convt1d(
            slot.up_in[i],
            u.ct_w,
            u.ct_b,
            row(slot.dw_in[i], 6, LATENT),
            1 + rows as u32,
            lat,
            lat,
            ratio as u32,
            ratio as u32,
            1,
        ));
        out.shift(slot.up_in[i], work.shift, 1, rows, LATENT);
        out.push(t::conv1d(
            slot.dw_in[i],
            u.dw_w,
            u.dw_b,
            work.ud,
            6 + out_rows,
            lat,
            lat,
            7,
            1,
            lat,
            6,
            false,
        ));
        out.shift(slot.dw_in[i], work.shift, 6, out_rows as usize, LATENT);
        out.push(t::layernorm(work.ud, u.ln_w, u.ln_b, work.ud, out_rows, lat, 1e-6));
        out.push(t::conv1d(
            work.ud,
            u.pw1_w,
            u.pw1_b,
            work.ubig,
            out_rows,
            lat,
            4 * lat,
            1,
            1,
            1,
            0,
            false,
        ));
        out.push(t::gelu(work.ubig, work.ubig, out_rows, 4 * lat));
        out.push(t::conv1d(
            work.ubig,
            u.pw2_w,
            u.pw2_b,
            work.ud,
            out_rows,
            4 * lat,
            lat,
            1,
            1,
            1,
            0,
            false,
        ));
        // The residual is the output of the transposed convolution, which the state rows of `dw_in` follow.
        let next = if i + 1 < c.ups.len() {
            row(slot.up_in[i + 1], 1, LATENT)
        } else {
            row(slot.dec_in, 6, LATENT)
        };
        out.push(t::copy(row(slot.dw_in[i], 6, LATENT), next, out_rows * lat));
        out.push(t::scale_add(work.ud, u.gamma, next, out_rows, lat));
    }
    // The decoder: a convolution, then four blocks of a transposed convolution and three residual units.
    let steps0 = upsampled(n);
    let dec = DECODER_DIM as u32;
    out.push(t::conv1d(
        slot.dec_in,
        c.dec0_w,
        c.dec0_b,
        work.ures,
        6 + steps0 as u32,
        lat,
        dec,
        7,
        1,
        1,
        6,
        false,
    ));
    out.shift(slot.dec_in, work.shift, 6, steps0, LATENT);
    out.push(t::snake(
        work.ures,
        c.blocks[0].alpha,
        c.blocks[0].beta,
        row(slot.ct_in[0], 1, DECODER_DIM),
        steps0 as u32,
        dec,
    ));
    for (b, (blk, rate)) in c.blocks.iter().zip(UPSAMPLE_RATES).enumerate() {
        let (cin, cout) = (block_in(b), block_in(b + 1));
        let steps = block_steps(b, n);
        let out_steps = steps * rate;
        out.push(t::convt1d(
            slot.ct_in[b],
            blk.ct_w,
            blk.ct_b,
            work.ures,
            1 + steps as u32,
            cin as u32,
            cout as u32,
            2 * rate as u32,
            rate as u32,
            1,
        ));
        out.shift(slot.ct_in[b], work.shift, 1, steps, cin);
        for (u, unit) in blk.units.iter().enumerate() {
            let state = 6 * unit.dilation;
            let ru = slot.ru_in[b][u];
            out.push(t::snake(
                work.ures,
                unit.a1,
                unit.b1,
                row(ru, state, cout),
                out_steps as u32,
                cout as u32,
            ));
            out.push(t::conv1d(
                ru,
                unit.c1_w,
                unit.c1_b,
                work.t1,
                (state + out_steps) as u32,
                cout as u32,
                cout as u32,
                7,
                unit.dilation as u32,
                1,
                state as u32,
                false,
            ));
            out.shift(ru, work.shift, state, out_steps, cout);
            out.push(t::snake(
                work.t1,
                unit.a2,
                unit.b2,
                work.t1,
                out_steps as u32,
                cout as u32,
            ));
            out.push(t::conv1d(
                work.t1,
                unit.c2_w,
                unit.c2_b,
                work.ures,
                out_steps as u32,
                cout as u32,
                cout as u32,
                1,
                1,
                1,
                0,
                true,
            ));
        }
        if b + 1 < c.blocks.len() {
            let next = &c.blocks[b + 1];
            out.push(t::snake(
                work.ures,
                next.alpha,
                next.beta,
                row(slot.ct_in[b + 1], 1, cout),
                out_steps as u32,
                cout as u32,
            ));
        } else {
            out.push(t::snake(
                work.ures,
                c.final_alpha,
                c.final_beta,
                row(slot.fs, 6, cout),
                out_steps as u32,
                cout as u32,
            ));
        }
    }
    let samples = steps_final(n);
    let last = block_in(UPSAMPLE_RATES.len());
    out.push(t::conv1d(
        slot.fs,
        c.final_w,
        c.final_b,
        work.wav,
        (6 + samples) as u32,
        last as u32,
        1,
        7,
        1,
        1,
        6,
        false,
    ));
    out.shift(slot.fs, work.shift, 6, samples, last);
    out.push(t::pcm(work.wav, work.pcm, samples as u32));
    out
}

/// The sampler state of a slot for a new request: the seed, no draws, an empty history.
pub fn sampler_state(seed: u64) -> Vec<u8> {
    let words = [seed as u32, (seed >> 32) as u32, 0, 0];
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}
