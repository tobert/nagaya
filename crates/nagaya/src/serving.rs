//! Which opened engines the process serves as council models, and with which family library (spec 11.1, 11.3).
//!
//! The identity of an engine names its family (`Identity::family`). The registry of family libraries opens the
//! library of that name from the identity and from the table `[engines.family]` of the profile. An engine whose
//! family has no library is not a served model.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use nagaya_abi::Identity;
use nagaya_council::family::Family;

use crate::council::{Service, ServiceConfig};
use crate::http::{Council, Served};
use crate::process::{OpenedEngine, Ports};
use crate::profile::PlacementProfile;

/// A family library that an opener made.
pub struct OpenedFamily {
    /// The library.
    pub family: Arc<dyn Family>,
    /// The hash of the tokenizer that the library loaded, or `None` for a family with no tokenizer file. It must
    /// equal the `tokenizer_hash` of the engine identity (spec 11.3).
    pub tokenizer_hash: Option<String>,
}

/// An opener makes a family library from the identity of an engine and the table `[engines.family]` of its entry.
type Opener = Box<dyn Fn(&Identity, &toml::Table) -> Result<OpenedFamily, String> + Send + Sync>;

/// The family library and the config of one served model.
pub struct Plan {
    /// The library.
    pub family: Arc<dyn Family>,
    /// The config of the service.
    pub config: ServiceConfig,
}

/// The openers of the family libraries, by family name (spec 11.3).
#[derive(Default)]
pub struct FamilyRegistry {
    openers: BTreeMap<String, Opener>,
}

impl FamilyRegistry {
    /// Make an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Make a registry with every family library that this build has: `toy` with the feature `fake`, `qwen38`
    /// with the feature `hip`, and `gemma4` with the feature `xe`.
    pub fn compiled_in() -> Self {
        #[allow(unused_mut)]
        let mut registry = Self::new();
        #[cfg(feature = "fake")]
        registry.register("toy", |_, _| {
            Ok(OpenedFamily {
                family: Arc::new(nagaya_text_toy::ToyFamily::new()),
                tokenizer_hash: None,
            })
        });
        #[cfg(feature = "hip")]
        registry.register("qwen38", open_qwen38);
        #[cfg(feature = "xe")]
        registry.register(nagaya_mk_gemma4_bmg::spec::FAMILY, crate::gemma4::open_family);
        registry
    }

    /// Add the opener of a family. A second opener for one name is a defect, and it panics.
    pub fn register(
        &mut self,
        family: &str,
        opener: impl Fn(&Identity, &toml::Table) -> Result<OpenedFamily, String> + Send + Sync + 'static,
    ) {
        let old = self.openers.insert(family.to_owned(), Box::new(opener));
        assert!(old.is_none(), "the family `{family}` has two openers");
    }

    /// The library and the config for an engine, or `None` when its family has no library.
    ///
    /// `table` is the `[engines.family]` table of the engine's entry in the profile. The context limit is the
    /// largest slot of the engine, in tokens.
    pub fn plan(&self, engine: &OpenedEngine, table: &toml::Table) -> Result<Option<Plan>, String> {
        let identity = &engine.identity;
        let Some(opener) = self.openers.get(&identity.family) else {
            return Ok(None);
        };
        let opened = opener(identity, table).map_err(|error| {
            format!(
                "cannot open the family `{}` for engine `{}`: {error}",
                identity.family, engine.name
            )
        })?;
        if let Some(hash) = &opened.tokenizer_hash
            && *hash != identity.tokenizer_hash
        {
            return Err(format!(
                "the tokenizer of the family `{}` has the hash {hash}, and the engine `{}` has {}; name the files of the engine",
                identity.family, engine.name, identity.tokenizer_hash
            ));
        }
        let tokens = engine
            .limits
            .slots
            .iter()
            .map(|class| class.tokens)
            .max()
            .ok_or_else(|| {
                format!(
                    "the engine `{}` declares no slot, so it has no context limit; declare a slot in its limits",
                    engine.name
                )
            })?;
        // The served model is the `model` of the engine identity, as gossip publishes it (spec 2.3).
        let mut config = ServiceConfig::new(&identity.model, &engine.name);
        config.context_tokens =
            usize::try_from(tokens).map_err(|_| format!("a slot of {tokens} tokens is too large"))?;
        Ok(Some(Plan {
            family: opened.family,
            config,
        }))
    }
}

/// The Qwen3.8 family from the GGUF in the key `gguf` of the table, or from the first shard of `MK_SHARDS`.
#[cfg(feature = "hip")]
fn open_qwen38(identity: &Identity, table: &toml::Table) -> Result<OpenedFamily, String> {
    let shard = match table.get("gguf") {
        Some(value) => std::path::PathBuf::from(
            value
                .as_str()
                .ok_or_else(|| "the key `gguf` of `[engines.family]` is not a string; write a path".to_owned())?,
        ),
        None => nagaya_mk_qwen38_gfx1151::MkPaths::from_env()
            .map_err(|error| error.message().to_owned())?
            .shards
            .into_iter()
            .next()
            .ok_or_else(|| "MK_SHARDS names no shard; give the three GGUF shards of the model".to_owned())?,
    };
    let _ = identity;
    let metadata = nagaya_text_qwen38::gguf::Metadata::open(&shard)
        .map_err(|error| format!("cannot read the GGUF metadata of `{}`: {error}", shard.display()))?;
    let hash = nagaya_text_qwen38::tokenizer_hash(&metadata).map_err(|error| error.to_string())?;
    let family = nagaya_text_qwen38::Qwen38Family::from_metadata(&metadata)
        .map_err(|error| format!("cannot build the Qwen3.8 family from `{}`: {error}", shard.display()))?;
    Ok(OpenedFamily {
        family: Arc::new(family),
        tokenizer_hash: Some(hash),
    })
}

/// The council of a started process: a service for each opened engine whose kind has a family library.
///
/// `peers` are the other engines whose snapshots the council reads and pins, with their served models, such as the
/// speech engines (spec 11.8a). `call_cpus` are the CPUs of the call threads. A family that does not load, or a service that does not attach,
/// is an error. The process refuses to start then (spec 10.4).
pub fn council(
    families: &FamilyRegistry,
    profile: &PlacementProfile,
    opened: &[OpenedEngine],
    ports: &Ports,
    store_dir: &Path,
    peers: &[(String, String)],
    call_cpus: Option<Vec<u32>>,
) -> Result<Council, String> {
    let mut models = Vec::new();
    for engine in opened {
        let placement = profile
            .engines
            .iter()
            .find(|placement| placement.name == engine.name)
            .ok_or_else(|| format!("the opened engine `{}` is not in the profile", engine.name))?;
        let Some(mut plan) = families.plan(engine, &placement.family)? else {
            tracing::info!(
                "engine `{}` has the family `{}`, which has no library, so it is not a served model",
                engine.name,
                engine.identity.family
            );
            continue;
        };
        plan.config.peers = peers.iter().filter(|(peer, _)| *peer != engine.name).cloned().collect();
        let service = Service::attach(ports.clone(), plan.family, store_dir, plan.config)
            .map_err(|error| format!("cannot serve engine `{}`: {error}", engine.name))?;
        tracing::info!("serving model `{}` on engine `{}`", service.model(), engine.name);
        models.push(Served {
            service,
            aliases: Vec::new(),
            device: None,
        });
    }
    let council = Council::new(models, call_cpus)?;
    match profile.store.and_then(|store| store.objects) {
        None => Ok(council),
        Some(budget) => {
            let store = nagaya_store::objects::ObjectStore::open(&store_dir.join("objects"), budget)
                .map_err(|error| format!("cannot open the objects in the store directory: {error}"))?;
            tracing::info!("objects: {} of a budget of {budget} bytes held", store.used());
            Ok(council.with_objects(store))
        }
    }
}
