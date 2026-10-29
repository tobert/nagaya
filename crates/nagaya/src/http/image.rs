//! Generations of a served image model (spec 11.9a): a prompt in, one PNG object for each image out, with the A1111
//! `parameters` chunk. A stream sends a `preview` event after each step when the request asks for previews.

use std::sync::Arc;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use gogi_proto::Invalid;
use gogi_proto::context::{MediaCount, MediaUsage, Usage};
use gogi_proto::generation::{
    Finish, GenerationDone, GenerationRequest, OutputEvent, PreviewEvent, ProgressEvent, SealedOutput, StreamEvent,
};
use gogi_proto::identity::ServedModel;
use gogi_proto::message::{Content, Part, Role};
use gogi_proto::objects::{Named, check_reads, objects_of};
use nagaya_abi::{ImageJob, ImageSampler, ImageSpacing, LoraInput, RgbImage};
use serde_json::Value;
use tokio::sync::mpsc;

use super::{Council, RequestId, guarded};
use crate::council::error::ServiceError;
use crate::image::{ImageError, ImageService, Made};
use nagaya_store::ContentId;
use nagaya_store::objects::ObjectStore;
use std::sync::Mutex;

/// The media type of every output of an image model.
pub const PNG: &str = "image/png";

/// The pixels of the largest image (spec 11.9a).
const MAX_PIXELS: u64 = 1_572_864;

/// The most images of one request.
const MAX_COUNT: u32 = 4;

/// The most LoRAs of one request (spec 11.9a).
const MAX_LORAS: usize = 8;

/// The sampler names of a request, as A1111 writes them in its chunk (spec 11.9a).
const SAMPLERS: [(&str, ImageSampler, &str); 4] = [
    ("euler", ImageSampler::Euler, "Euler"),
    ("euler a", ImageSampler::EulerAncestral, "Euler a"),
    ("dpm++ 2m karras", ImageSampler::DpmPp2mKarras, "DPM++ 2M Karras"),
    (
        "dpm++ 2m sde karras",
        ImageSampler::DpmPp2mSdeKarras,
        "DPM++ 2M SDE Karras",
    ),
];

fn invalid(param: impl Into<String>, message: String) -> ServiceError {
    ServiceError::Invalid(Invalid::param(param, message))
}

/// The service error of an image error. A failed engine is internal, so the process stops (spec 8.5).
pub fn service_error(error: ImageError) -> ServiceError {
    match error {
        ImageError::Busy(message) => ServiceError::Busy(message),
        ImageError::Timeout(_) => ServiceError::Timeout,
        ImageError::Stopping(_) => ServiceError::Stopping,
        ImageError::Unavailable { engine, state } => ServiceError::Unavailable { engine, state },
        ImageError::Failed(message) => ServiceError::internal(message),
        ImageError::Refused { param, message } => invalid(param, message),
    }
}

/// An image generation after its checks.
#[derive(Clone, Debug, PartialEq)]
pub struct Checked {
    pub job: ImageJob,
    /// The texts of the prompts, for the A1111 chunk.
    pub prompt: String,
    pub negative: String,
    /// The tokens of the prompt that the family took.
    pub tokens: u64,
    pub schedule: crate::council::Schedule,
}

/// The text of a message: its string, or its text parts joined.
fn text_of(content: &Option<Content>) -> String {
    content
        .as_ref()
        .map(|c| {
            c.parts()
                .into_iter()
                .filter_map(|part| match part {
                    Part::Text { text } => Some(text),
                    Part::Object { .. } => None,
                })
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

/// A member of `output.options` that must be an integer in `range`.
fn integer(key: &str, value: &Value, range: std::ops::RangeInclusive<u64>) -> Result<u64, ServiceError> {
    value.as_u64().filter(|v| range.contains(v)).ok_or_else(|| {
        invalid(
            format!("output.options.{key}"),
            format!(
                "nothing ran: output.options.{key} is {value}; give an integer from {} to {}",
                range.start(),
                range.end()
            ),
        )
    })
}

/// Check an image generation, and make its job (spec 11.9a).
pub fn check(
    service: &ImageService,
    model: &ServedModel,
    request: &GenerationRequest,
    store: Option<&Mutex<ObjectStore>>,
) -> Result<Checked, ServiceError> {
    request.validate().map_err(ServiceError::Invalid)?;
    let Some(output) = &request.output else {
        return Err(invalid(
            "output",
            format!(
                "nothing ran: {} writes images only; give an `output` of `{PNG}`",
                model.model
            ),
        ));
    };
    output.check(model).map_err(ServiceError::Invalid)?;
    if !request.decisions.is_empty() {
        return Err(invalid(
            "decisions",
            format!("nothing ran: {} takes no riding decision", model.model),
        ));
    }
    let objects: Vec<Named> = objects_of(&request.context.messages);
    for named in &objects {
        named.object.check(&named.at).map_err(ServiceError::Invalid)?;
    }
    check_reads(model, &objects).map_err(ServiceError::Invalid)?;
    let mut prompt = None;
    for message in &request.context.messages {
        if message.open {
            return Err(invalid(
                "continue",
                format!("nothing ran: {} continues no message", model.model),
            ));
        }
        match message.role {
            Role::User => prompt = Some(text_of(&message.content)),
            Role::System | Role::Assistant | Role::Tool => {}
        }
    }
    let prompt = prompt.ok_or_else(|| {
        invalid(
            "messages",
            format!(
                "nothing ran: the context of {} has no `user` message; give the prompt in one",
                model.model
            ),
        )
    })?;

    let config = service.config();
    let mut negative = String::new();
    let (mut width, mut height, mut steps, mut count) = (1024u64, 1024u64, 30u64, 1u64);
    let mut cfg = 7.0f64;
    let mut sampler = ImageSampler::DpmPp2mKarras;
    let mut spacing = ImageSpacing::Leading;
    let mut clip_skip = u64::from(config.clip_skip);
    let mut lora_refs = Vec::new();
    // A request asks for previews in the member `previews`, which needs `stream` (gogi "Steps and previews").
    let previews = request.previews == Some(true);
    for (key, value) in &output.options {
        let at = format!("output.options.{key}");
        match key.as_str() {
            "negative" => {
                negative = value
                    .as_str()
                    .ok_or_else(|| invalid(&at, format!("nothing ran: {at} is {value}, not a string")))?
                    .to_owned();
            }
            "width" | "height" => {
                let size = value
                    .as_u64()
                    .filter(|v| (512..=2048).contains(v) && v % 64 == 0)
                    .ok_or_else(|| {
                        invalid(
                            &at,
                            format!("nothing ran: {at} is {value}; give a multiple of 64 from 512 to 2048"),
                        )
                    })?;
                if key == "width" {
                    width = size;
                } else {
                    height = size;
                }
            }
            "steps" => steps = integer(key, value, 1..=150)?,
            "count" => count = integer(key, value, 1..=u64::from(MAX_COUNT))?,
            "clip_skip" => clip_skip = integer(key, value, 2..=3)?,
            "cfg" => {
                cfg = value.as_f64().filter(|v| (1.0..=30.0).contains(v)).ok_or_else(|| {
                    invalid(
                        &at,
                        format!("nothing ran: {at} is {value}; give a number from 1.0 to 30.0"),
                    )
                })?;
            }
            "loras" => lora_refs = lora_list(value)?,
            "spacing" => {
                spacing = match value.as_str() {
                    Some("leading") => ImageSpacing::Leading,
                    Some("trailing") => ImageSpacing::Trailing,
                    _ => {
                        return Err(invalid(
                            &at,
                            format!("nothing ran: {at} is {value}; give `leading` or `trailing`"),
                        ));
                    }
                };
            }
            "sampler" => {
                sampler = value
                    .as_str()
                    .and_then(|name| SAMPLERS.iter().find(|(given, _, _)| *given == name))
                    .map(|(_, sampler, _)| *sampler)
                    .ok_or_else(|| {
                        invalid(
                            &at,
                            format!(
                                "nothing ran: {at} is {value}; give `euler`, `euler a`, `dpm++ 2m karras` or \
                                 `dpm++ 2m sde karras`"
                            ),
                        )
                    })?;
            }
            _ => {
                return Err(invalid(
                    at,
                    format!(
                        "nothing ran: {} takes the output options `negative`, `width`, `height`, `steps`, `cfg`, \
                         `sampler`, `spacing`, `count` and `clip_skip`, not `{key}`; ask for previews with the request's \
                         `previews`",
                        model.model
                    ),
                ));
            }
        }
    }
    let loras = held_loras(store, &lora_refs)?;
    if spacing == ImageSpacing::Trailing && !matches!(sampler, ImageSampler::Euler | ImageSampler::EulerAncestral) {
        return Err(invalid(
            "output.options.spacing",
            "nothing ran: output.options.spacing is trailing, and the DPM++ samplers take their timesteps from their \
             sigmas; give `euler` or `euler a`"
                .to_owned(),
        ));
    }
    if width * height > MAX_PIXELS {
        return Err(invalid(
            "output.options.width",
            format!("nothing ran: {width} by {height} is more than {MAX_PIXELS} pixels; give a smaller size"),
        ));
    }
    let mut seed = None;
    if let Some(sample) = &request.sample {
        let others = [
            ("top_p", sample.top_p.is_some()),
            ("top_k", sample.top_k.is_some()),
            ("min_p", sample.min_p.is_some()),
        ];
        if let Some((key, _)) = others.iter().find(|(_, given)| *given) {
            return Err(invalid(
                format!("sample.{key}"),
                format!(
                    "nothing ran: {} takes `sample.seed` only, not `sample.{key}`",
                    model.model
                ),
            ));
        }
        seed = sample.seed;
    }
    let family = service.family();
    let (prompt_chunks, tokens) = family
        .prompt(&prompt)
        .map_err(|message| invalid("messages", format!("nothing ran: the prompt {message}")))?;
    let negative_chunks = if negative.is_empty() {
        Vec::new()
    } else {
        family
            .prompt(&negative)
            .map_err(|message| {
                invalid(
                    "output.options.negative",
                    format!("nothing ran: the negative prompt {message}"),
                )
            })?
            .0
    };
    Ok(Checked {
        job: ImageJob {
            prompt: prompt_chunks,
            negative: negative_chunks,
            clip_skip: clip_skip as u32,
            width: width as u32,
            height: height as u32,
            steps: steps as u32,
            cfg: cfg as f32,
            sampler,
            spacing,
            count: count as u32,
            seed: seed.unwrap_or_else(chosen_seed),
            previews,
            loras,
        },
        prompt,
        negative,
        tokens,
        schedule: crate::council::Schedule::of(request.class, request.priority),
    })
}

/// A LoRA of a request as the client names it: the id and the weight of its object.
struct LoraRef {
    id: ContentId,
    text: String,
    weight: f32,
}

/// The list of `output.options.loras`: at most 8 objects, each with an `id` and a `weight` from -2.0 to 2.0 (spec 11.9a).
fn lora_list(value: &Value) -> Result<Vec<LoraRef>, ServiceError> {
    let items = value
        .as_array()
        .filter(|items| items.len() <= MAX_LORAS)
        .ok_or_else(|| {
            invalid(
                "output.options.loras",
                format!("nothing ran: output.options.loras is {value}; give a list of at most {MAX_LORAS} LoRAs"),
            )
        })?;
    items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let at = format!("output.options.loras[{i}]");
            let text = item
                .get("id")
                .and_then(Value::as_str)
                .filter(|t| t.starts_with("blake3:"));
            let id = text.and_then(|t| ContentId::from_hex(t.trim_start_matches("blake3:")).ok());
            let (Some(text), Some(id)) = (text, id) else {
                return Err(invalid(
                    format!("{at}.id"),
                    format!(
                        "nothing ran: {at}.id is {}; give a `blake3:` id of a LoRA object",
                        item["id"]
                    ),
                ));
            };
            let weight = item
                .get("weight")
                .and_then(Value::as_f64)
                .filter(|w| (-2.0..=2.0).contains(w))
                .ok_or_else(|| {
                    invalid(
                        format!("{at}.weight"),
                        format!(
                            "nothing ran: {at}.weight is {}; give a number from -2.0 to 2.0",
                            item["weight"]
                        ),
                    )
                })?;
            Ok(LoraRef {
                id,
                text: text.to_owned(),
                weight: weight as f32,
            })
        })
        .collect()
}

/// The LoRA files of a request, read from the object store. A LoRA that the store does not hold is a 404 that names
/// its part, as for any object (spec 11.9a, 6.15).
fn held_loras(objects: Option<&Mutex<ObjectStore>>, refs: &[LoraRef]) -> Result<Vec<LoraInput>, ServiceError> {
    refs.iter()
        .enumerate()
        .map(|(i, lora)| {
            let at = format!("output.options.loras[{i}]");
            let found = match objects {
                Some(store) => store
                    .lock()
                    .expect("the object store lock is poisoned")
                    .get(&lora.id, super::objects::now_ms())
                    .map_err(|e| ServiceError::internal(e.to_string()))?,
                None => None,
            };
            let Some(found) = found else {
                return Err(ServiceError::NotHeld {
                    message: format!(
                        "nothing ran: {at} is {}, which the server does not hold; PUT it again",
                        lora.text
                    ),
                    param: at,
                });
            };
            let file = std::fs::read(&found.path).map_err(|e| ServiceError::internal(e.to_string()))?;
            Ok(LoraInput {
                id: lora.text.clone(),
                weight: lora.weight,
                file: Arc::new(file),
            })
        })
        .collect()
}

/// A seed for a request that names none: from the clock, so two requests differ.
fn chosen_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos() as u64)
}

/// The A1111 `parameters` text of image `index` of a job (spec 11.9a). A1111 leaves out the negative line when the
/// negative prompt is empty, and so does this.
pub fn parameters(checked: &Checked, model: &str, index: u32) -> String {
    let job = &checked.job;
    let sampler = SAMPLERS
        .iter()
        .find(|(_, sampler, _)| *sampler == job.sampler)
        .map(|(_, _, name)| *name)
        .expect("each sampler has a name");
    let mut text = checked.prompt.clone();
    if !checked.negative.is_empty() {
        text.push_str(&format!("\nNegative prompt: {}", checked.negative));
    }
    text.push_str(&format!(
        "\nSteps: {}, Sampler: {sampler}, CFG scale: {}, Seed: {}, Size: {}x{}, Model: {model}, Clip skip: {}",
        job.steps,
        job.cfg,
        job.seed.wrapping_add(u64::from(index)),
        job.width,
        job.height,
        job.clip_skip
    ));
    // The LoRAs follow the settings, each as its id and its weight, in the order of the request (spec 11.9a).
    if !job.loras.is_empty() {
        let ids: Vec<String> = job
            .loras
            .iter()
            .map(|lora| format!("{} {}", lora.id, lora.weight))
            .collect();
        text.push_str(&format!(", Lora ids: \"{}\"", ids.join(", ")));
    }
    text
}

/// The bytes of a PNG of 8-bit RGB, with a chunk `parameters` when `text` is given: `tEXt` for Latin-1 text, else
/// `iTXt`.
pub fn png(image: &RgbImage, text: Option<&str>) -> Result<Vec<u8>, ServiceError> {
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, image.width, image.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    if let Some(text) = text {
        // A1111 writes tEXt for Latin-1 text and iTXt for other text, which tEXt cannot hold.
        let added = if text.chars().all(|c| u32::from(c) <= 0xff) {
            encoder.add_text_chunk("parameters".to_owned(), text.to_owned())
        } else {
            encoder.add_itxt_chunk("parameters".to_owned(), text.to_owned())
        };
        added.map_err(|error| ServiceError::internal(format!("cannot add the parameters chunk of a PNG: {error}")))?;
    }
    let mut writer = encoder
        .write_header()
        .map_err(|error| ServiceError::internal(format!("cannot write the header of a PNG: {error}")))?;
    writer
        .write_image_data(&image.rgb)
        .map_err(|error| ServiceError::internal(format!("cannot write the pixels of a PNG: {error}")))?;
    writer
        .finish()
        .map_err(|error| ServiceError::internal(format!("cannot end a PNG: {error}")))?;
    Ok(bytes)
}

/// What a generation tells its route as it runs.
enum Item {
    /// A step of each image ended: a `progress` event for each image, each with its `preview` when the request
    /// asked for previews (gogi "Steps and previews").
    Step(Vec<(ProgressEvent, Option<PreviewEvent>)>),
    /// The outputs begin: one pending id for each image.
    Outputs(Vec<OutputEvent>),
    End(Box<Result<GenerationDone, ServiceError>>),
}

/// Run a checked generation to its end. `tell` hears the previews and the pending outputs, and gives false when the
/// client went away. The pending ids are sealed or cancelled before it returns.
fn make(
    council: &Council,
    service: &ImageService,
    model: &str,
    checked: Checked,
    stream: bool,
    tell: &dyn Fn(Item) -> bool,
) -> Result<GenerationDone, ServiceError> {
    let made = service
        .generate(checked.job.clone(), checked.schedule)
        .map_err(service_error)?;
    let mut pending: Vec<(String, Arc<super::pending::PendingOutput>)> = Vec::new();
    let result = (|| {
        for item in made.iter() {
            match item.map_err(service_error)? {
                // A reply with no stream sends no event. The check refused `previews` with no stream.
                Made::Previews { .. } | Made::Step { .. } if !stream => {}
                Made::Step { step } => {
                    let events = (0..checked.job.count).map(|index| (progress(&checked, index, step), None));
                    if !tell(Item::Step(events.collect())) {
                        return Err(ServiceError::Cancelled);
                    }
                }
                Made::Previews { step, previews } => {
                    let mut events = Vec::with_capacity(previews.len());
                    for (index, preview) in previews.iter().enumerate() {
                        let bytes = png(preview, None)?;
                        let id = format!("blake3:{}", super::voice::seal(council, &bytes, PNG)?.hex());
                        let progress = progress(&checked, index as u32, step);
                        let preview = PreviewEvent {
                            index: progress.index,
                            step: progress.step,
                            of: progress.of,
                            id: Some(gogi_proto::ids::ContentId::parse(id).expect("a content id parses")),
                            mime: Some(PNG.to_owned()),
                            text: None,
                        };
                        events.push((progress, Some(preview)));
                    }
                    if !tell(Item::Step(events)) {
                        return Err(ServiceError::Cancelled);
                    }
                }
                Made::Decoding => {
                    if stream {
                        pending = (0..checked.job.count).map(|_| council.pending().start(PNG)).collect();
                        let events = pending
                            .iter()
                            .enumerate()
                            .map(|(index, (id, _))| OutputEvent {
                                index: index as u64,
                                mime: PNG.to_owned(),
                                pending: id.clone(),
                            })
                            .collect();
                        if !tell(Item::Outputs(events)) {
                            return Err(ServiceError::Cancelled);
                        }
                    }
                }
                Made::Images(images) => return done(council, model, &checked, &images, &pending),
            }
        }
        // The job ended with no images and no error: a stop dropped it (spec 11.7).
        Err(ServiceError::Cancelled)
    })();
    if result.is_err() {
        for (id, _) in &pending {
            council.pending().cancel(id);
        }
    }
    result
}

/// The `progress` event of image `index` after step `step` of the job.
fn progress(checked: &Checked, index: u32, step: u32) -> ProgressEvent {
    ProgressEvent {
        index: Some(u64::from(index)),
        step: u64::from(step),
        of: u64::from(checked.job.steps),
    }
}

/// Seal the PNG of each image, and give the reply.
fn done(
    council: &Council,
    model: &str,
    checked: &Checked,
    images: &[RgbImage],
    pending: &[(String, Arc<super::pending::PendingOutput>)],
) -> Result<GenerationDone, ServiceError> {
    let mut outputs = Vec::with_capacity(images.len());
    let mut bytes_written = 0;
    for (index, image) in images.iter().enumerate() {
        let bytes = png(image, Some(&parameters(checked, model, index as u32)))?;
        let id = format!("blake3:{}", super::voice::seal(council, &bytes, PNG)?.hex());
        if let Some((pending_id, output)) = pending.get(index) {
            output.push(&bytes);
            council.pending().seal(pending_id, &id);
        }
        bytes_written += bytes.len() as u64;
        outputs.push(SealedOutput {
            index: index as u64,
            id: gogi_proto::ids::ContentId::parse(id).expect("a content id parses"),
            mime: PNG.to_owned(),
            bytes: bytes.len() as u64,
        });
    }
    let job = &checked.job;
    Ok(GenerationDone {
        model: model.to_owned(),
        content: String::new(),
        reasoning: String::new(),
        tool_calls: Vec::new(),
        finish: Finish::EndTurn,
        usage: Usage {
            prompt: checked.tokens,
            kept: 0,
            fed: checked.tokens,
            completion: 0,
            reasoning: 0,
            media: Some(MediaUsage {
                read: MediaCount::default(),
                written: MediaCount {
                    objects: images.len() as u64,
                    bytes: bytes_written,
                    pixels: Some(images.len() as u64 * u64::from(job.width) * u64::from(job.height)),
                    steps: Some(u64::from(job.steps)),
                    ..MediaCount::default()
                },
            }),
        },
        head: None,
        snapshots: Vec::new(),
        stats: None,
        decisions: Vec::new(),
        seed: Some(job.seed),
        phase: None,
        outputs,
    })
}

/// The served model of an image service ("The served image model"). Its hashes are the engine's after its first
/// open since the start, and empty before.
pub fn served_model(service: &ImageService, status: gogi_proto::identity::ModelStatus) -> ServedModel {
    let identity = service.identity();
    let config = service.config();
    ServedModel {
        model: config.model.clone(),
        aliases: Vec::new(),
        status,
        weight_hash: identity
            .as_ref()
            .map(|i| i.checkpoint_digest.clone())
            .unwrap_or_default(),
        tokenizer_hash: identity.as_ref().map(|i| i.tokenizer_hash.clone()).unwrap_or_default(),
        template: String::new(),
        engine: config.engine.clone(),
        device: None,
        text_stop: None,
        limits: gogi_proto::identity::ModelLimits {
            context_tokens: config.context_tokens,
            decision_context_tokens: 0,
            snapshot_bytes: 0,
            contexts_per_decision: 0,
            default_timeout_ms: config.timeout.as_millis() as u64,
            text_tokens: None,
        },
        capabilities: vec![gogi_proto::identity::Capability::Generate],
        media: Some(gogi_proto::identity::Media {
            inputs: Vec::new(),
            outputs: vec![PNG.to_owned()],
            // A reply holds images and no text, so the model takes no build, decision or raw read (gogi c3acffe).
            text_replies: false,
        }),
    }
}

/// The served model and the service of an image model that a request names.
pub fn route(council: &Council, name: &str) -> Option<(Arc<ImageService>, ServedModel)> {
    let service = council.image_service(name)?;
    let model = served_model(&service, council.model_status(service.engine_state()));
    Some((service, model))
}

/// Serve a generation of an image model: a reply with no stream, or a stream of `progress`, `preview`, `output` and
/// `done`.
pub async fn generation(
    council: Arc<Council>,
    service: Arc<ImageService>,
    model: ServedModel,
    request: GenerationRequest,
    id: RequestId,
) -> Response {
    let objects = council.objects();
    if !request.stream.unwrap_or(false) {
        let inner = Arc::clone(&council);
        let result = council
            .call(move || {
                let checked = check(&service, &model, &request, objects.as_deref())?;
                make(&inner, &service, &model.model, checked, false, &|_| true)
            })
            .await;
        return super::routes::answer(&id, result);
    }
    let permit = match council.permit() {
        Ok(permit) => permit,
        Err(failure) => return failure.respond(&id),
    };
    let (out, mut items) = mpsc::channel::<Item>(8);
    let inner = Arc::clone(&council);
    let started = council.spawn(move || {
        let result = guarded(|| {
            let checked = check(&service, &model, &request, objects.as_deref())?;
            make(&inner, &service, &model.model, checked, true, &|item| {
                out.blocking_send(item).is_ok()
            })
        });
        let _ = out.blocking_send(Item::End(Box::new(result)));
        drop(permit);
    });
    if let Err(failure) = started {
        return failure.respond(&id);
    }
    // A failure before the first event gets a status; a failure after the `200` is an `error` event.
    let first = match items.recv().await {
        Some(Item::End(result)) if result.is_err() => {
            let error = result.err().expect("the result is an error");
            return council.failed(error).respond(&id);
        }
        Some(first) => first,
        None => {
            return council
                .failed(ServiceError::internal("an image generation ended with no result"))
                .respond(&id);
        }
    };
    let state = (Some(first), items, council, id, false);
    let stream = futures_util::stream::unfold(state, |(first, mut items, council, id, ended)| async move {
        if ended {
            return None;
        }
        let item = match first {
            Some(item) => Some(item),
            None => items.recv().await,
        };
        let (events, ended) = match item {
            Some(Item::Step(steps)) => (
                steps
                    .into_iter()
                    .flat_map(|(progress, preview)| {
                        std::iter::once(StreamEvent::Progress(progress)).chain(preview.map(StreamEvent::Preview))
                    })
                    .map(|event| Event::default().event(event.name()).data(event.data()))
                    .collect(),
                false,
            ),
            Some(Item::Outputs(outputs)) => (
                outputs
                    .into_iter()
                    .map(|output| {
                        let event = StreamEvent::Output(output);
                        Event::default().event(event.name()).data(event.data())
                    })
                    .collect(),
                false,
            ),
            Some(Item::End(result)) => {
                let event = match *result {
                    Ok(done) => StreamEvent::Done(Box::new(done)),
                    Err(error) => super::stream::error_event(&council, error, &id),
                };
                (vec![Event::default().event(event.name()).data(event.data())], true)
            }
            None => {
                let error = ServiceError::internal("an image generation ended with no result");
                let event = super::stream::error_event(&council, error, &id);
                (vec![Event::default().event(event.name()).data(event.data())], true)
            }
        };
        let events = futures_util::stream::iter(events.into_iter().map(Ok::<Event, std::convert::Infallible>));
        Some((events, (None, items, council, id, ended)))
    });
    Sse::new(futures_util::StreamExt::flatten(stream)).into_response()
}
