//! Which opened engines the process serves as council models, and with which family library (spec 11.1, 11.3).
//!
//! The kind of an engine in the profile names the loader of its family. The loader then checks the family against
//! the identity of the engine, so the identity selects the library (spec 11.3). An engine whose kind has no family
//! library is not a served model.

use std::path::Path;
use std::sync::Arc;

use nagaya_council::family::Family;

use crate::council::{Service, ServiceConfig};
use crate::http::{Council, Served};
use crate::process::{OpenedEngine, Ports};
use crate::profile::PlacementProfile;

/// The family library and the config of one served model.
struct Plan {
    family: Arc<dyn Family>,
    config: ServiceConfig,
}

/// The family and the config for an engine of `kind`, or `None` when the kind has no family library.
fn plan(kind: &str, engine: &OpenedEngine) -> Result<Option<Plan>, String> {
    // The served model is the `model` of the engine identity, as gossip publishes it (spec 2.3).
    let config = ServiceConfig::new(&engine.identity.model, &engine.name);
    #[cfg(feature = "fake")]
    if kind == crate::admission::FAKE_TOY_KIND {
        return Ok(Some(Plan {
            family: Arc::new(nagaya_text_toy::ToyFamily::new()),
            config,
        }));
    }
    #[cfg(feature = "hip")]
    if kind == crate::mk::KIND {
        return mk_plan(engine, config).map(Some);
    }
    let _ = (kind, config);
    Ok(None)
}

/// The Qwen3.8 family from the first shard of `MK_SHARDS`. Its tokenizer hash must be the one in the identity.
#[cfg(feature = "hip")]
fn mk_plan(engine: &OpenedEngine, mut config: ServiceConfig) -> Result<Plan, String> {
    let paths = nagaya_mk_qwen38_gfx1151::MkPaths::from_env().map_err(|error| error.message().to_owned())?;
    let shard = paths
        .shards
        .first()
        .ok_or_else(|| "MK_SHARDS names no shard; give the three GGUF shards of the model".to_owned())?;
    let metadata = nagaya_text_qwen38::gguf::Metadata::open(shard)
        .map_err(|error| format!("cannot read the GGUF metadata of `{}`: {error}", shard.display()))?;
    let hash = nagaya_text_qwen38::tokenizer_hash(&metadata).map_err(|error| error.to_string())?;
    if hash != engine.identity.tokenizer_hash {
        return Err(format!(
            "the tokenizer of `{}` has the hash {hash}, and the engine `{}` has {}; give the shards of the engine",
            shard.display(),
            engine.name,
            engine.identity.tokenizer_hash
        ));
    }
    let family = nagaya_text_qwen38::Qwen38Family::from_metadata(&metadata)
        .map_err(|error| format!("cannot build the Qwen3.8 family from `{}`: {error}", shard.display()))?;
    config.context_tokens = nagaya_mk_qwen38_gfx1151::MAX_CONTEXT as usize;
    Ok(Plan {
        family: Arc::new(family),
        config,
    })
}

/// The council of a started process: a service for each opened engine whose kind has a family library.
///
/// `call_cpus` are the CPUs of the call threads. A family that does not load, or a service that does not attach,
/// is an error. The process refuses to start then (spec 10.4).
pub fn council(
    profile: &PlacementProfile,
    opened: &[OpenedEngine],
    ports: &Ports,
    store_dir: &Path,
    call_cpus: Option<Vec<u32>>,
) -> Result<Council, String> {
    let mut models = Vec::new();
    for engine in opened {
        let kind = profile
            .engines
            .iter()
            .find(|placement| placement.name == engine.name)
            .map(|placement| placement.kind.as_str())
            .ok_or_else(|| format!("the opened engine `{}` is not in the profile", engine.name))?;
        let Some(plan) = plan(kind, engine)? else {
            tracing::info!(
                "engine `{}` of kind `{kind}` has no family library, so it is not a served model",
                engine.name
            );
            continue;
        };
        let service = Service::attach(ports.clone(), plan.family, store_dir, plan.config)
            .map_err(|error| format!("cannot serve engine `{}`: {error}", engine.name))?;
        tracing::info!("serving model `{}` on engine `{}`", service.model(), engine.name);
        models.push(Served {
            service,
            aliases: Vec::new(),
            device: None,
        });
    }
    Council::new(models, call_cpus)
}
