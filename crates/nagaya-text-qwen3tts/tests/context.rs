//! The rendering of a speech context (spec 11.8a "The rendering of a speech context"): the voice design, the lines,
//! the frames of their audio, and the codec end before a line that follows audio.

mod common;

use common::SMALL_CONFIG;
use nagaya_text_qwen3tts::{Error, PromptPosition, SpeechIds, SpeechItem, SpeechPosition};

const IM_START: u32 = 151644;
const IM_END: u32 = 151645;
const ASSISTANT: u32 = 77091;
const NEWLINE: u32 = 198;
const TTS_PAD: u32 = 151671;
const CODEC_EOS: u32 = 2150;

fn ids() -> SpeechIds {
    SpeechIds::from_config(SMALL_CONFIG).expect("the small config")
}

/// The ids of `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n` for text tokens `text`.
fn assistant(text: &[u32]) -> Vec<u32> {
    let mut out = vec![IM_START, ASSISTANT, NEWLINE];
    out.extend(text);
    out.extend([IM_END, NEWLINE, IM_START, ASSISTANT, NEWLINE]);
    out
}

fn pairs(positions: Vec<PromptPosition>) -> Vec<SpeechPosition> {
    positions.into_iter().map(SpeechPosition::Pair).collect()
}

fn frame(base: u32) -> [u32; 16] {
    std::array::from_fn(|g| base + g as u32)
}

#[test]
fn a_voice_and_a_line_render_as_one_request() {
    let ids = ids();
    let voice = [IM_START, 872, NEWLINE, 500, 501, IM_END, NEWLINE];
    let line = assistant(&[1000, 1001]);
    let r = ids
        .context(Some(&voice), &[SpeechItem::Line(&line)], "english")
        .expect("render");
    assert_eq!(
        r.positions,
        pairs(ids.positions(Some(&voice), &line, "english").unwrap())
    );
    assert_eq!(
        r.ends,
        [voice.len(), r.positions.len()],
        "a boundary after the voice and after the line"
    );
}

#[test]
fn audio_renders_as_frames_and_the_next_line_starts_with_the_codec_end() {
    let ids = ids();
    let (one, two) = (assistant(&[1000]), assistant(&[1001, 1002]));
    let audio = [frame(10), frame(40)];
    let items = [
        SpeechItem::Line(&one),
        SpeechItem::Audio(&audio),
        SpeechItem::Line(&two),
    ];
    let r = ids.context(None, &items, "auto").expect("render");
    let first = pairs(ids.positions(None, &one, "auto").unwrap());
    let mut want = first.clone();
    want.extend([SpeechPosition::Frame(frame(10)), SpeechPosition::Frame(frame(40))]);
    want.push(SpeechPosition::Pair(PromptPosition {
        text: Some(TTS_PAD),
        codec: Some(CODEC_EOS),
    }));
    want.extend(pairs(ids.positions(None, &two, "auto").unwrap()));
    assert_eq!(r.positions, want);
    let after_audio = first.len() + 2;
    assert_eq!(
        r.ends,
        [first.len(), after_audio, want.len()],
        "no boundary for an absent voice"
    );
}

#[test]
fn audio_with_no_line_before_it_is_refused() {
    let ids = ids();
    let line = assistant(&[1000]);
    let audio = [frame(10)];
    for items in [
        vec![SpeechItem::Audio(&audio[..])],
        vec![
            SpeechItem::Line(&line),
            SpeechItem::Audio(&audio[..]),
            SpeechItem::Audio(&audio[..]),
        ],
    ] {
        let error = ids.context(None, &items, "auto").unwrap_err();
        assert!(
            matches!(&error, Error::Context { item, .. } if *item == items.len() - 1),
            "{error:?}"
        );
    }
}
