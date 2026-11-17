//! Which opened engines the process serves as council models, and with which family library (spec 11.1, 11.3).
//!
//! The identity of an engine names its family (`Identity::family`). The registry of family libraries opens the
//! library of that name from the identity and from the table `[engines.family]` of the profile. An engine whose
//! family has no library is not a served model.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use nagaya_abi::{Identity, Limits};
use nagaya_council::family::Family;

use crate::admission::Registry;
use crate::council::{ServerFiles, Service, ServiceConfig};
use crate::http::{Council, Served};
use crate::process::{OpenedEngine, Ports};
use crate::profile::{PlacementProfile, Residency};

/// A family library that an opener made.
pub struct OpenedFamily {
    /// The library.
    pub family: Arc<dyn Family>,
    /// The hash of the tokenizer that the library loaded, or `None` for a family with no tokenizer file. It must
    /// equal the `tokenizer_hash` of the engine identity (spec 11.3).
    pub tokenizer_hash: Option<String>,
    /// The speech path of a family that speaks (spec 11.11). The served model then writes `audio/wav` too.
    pub speech: Option<crate::council::speech_out::SpeechOut>,
    /// The source of the served model name of an engine of this family.
    pub served: ServedName,
}

/// The source of the served model name of an engine (spec 11.1, 11.11).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServedName {
    /// The `model` of the engine identity, as gossip publishes it (spec 2.3, 11.1).
    Identity,
    /// The engine name of the profile. Thus two engines of one checkpoint serve two names (spec 11.11).
    Engine,
}

/// The objects of the process, which a family reads for the media parts of a context (spec 6.15, 11.11).
pub type ObjectReader = Arc<std::sync::Mutex<nagaya_store::objects::ObjectStore>>;

/// An opener makes a family library from the identity of an engine, the table `[engines.family]` of its entry, and
/// the objects of the process when the profile has an object store.
type Opener = Box<dyn Fn(&Identity, &toml::Table, Option<&ObjectReader>) -> Result<OpenedFamily, String> + Send + Sync>;

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
    /// with the feature `hip`, and `gemma4` and `lfm3` with the feature `xe`.
    pub fn compiled_in() -> Self {
        #[allow(unused_mut)]
        let mut registry = Self::new();
        #[cfg(feature = "fake")]
        registry.register("toy", |_, _| {
            Ok(OpenedFamily {
                family: Arc::new(nagaya_text_toy::ToyFamily::new()),
                tokenizer_hash: None,
                speech: None,
                served: ServedName::Identity,
            })
        });
        #[cfg(feature = "hip")]
        registry.register("qwen38", open_qwen38);
        #[cfg(feature = "xe")]
        registry.register(nagaya_mk_gemma4_bmg::spec::FAMILY, crate::gemma4::open_family);
        #[cfg(feature = "xe")]
        registry.register_with_objects(nagaya_mk_lfm3_bmg::spec::FAMILY, crate::lfm3::open_family_with);
        registry
    }

    /// Add the opener of a family. A second opener for one name is a defect, and it panics.
    pub fn register(
        &mut self,
        family: &str,
        opener: impl Fn(&Identity, &toml::Table) -> Result<OpenedFamily, String> + Send + Sync + 'static,
    ) {
        self.register_with_objects(family, move |identity, table, _| opener(identity, table));
    }

    /// Add the opener of a family that reads the objects of the process, such as the LFM3 family of Omni (spec
    /// 11.11). A second opener for one name is a defect, and it panics.
    pub fn register_with_objects(
        &mut self,
        family: &str,
        opener: impl Fn(&Identity, &toml::Table, Option<&ObjectReader>) -> Result<OpenedFamily, String>
        + Send
        + Sync
        + 'static,
    ) {
        let old = self.openers.insert(family.to_owned(), Box::new(opener));
        assert!(old.is_none(), "the family `{family}` has two openers");
    }

    /// The library and the config for an engine, or `None` when its family has no library.
    ///
    /// `table` is the `[engines.family]` table of the engine's entry in the profile. The context limit is the
    /// largest slot of the engine, in tokens.
    pub fn plan(&self, engine: &OpenedEngine, table: &toml::Table) -> Result<Option<Plan>, String> {
        self.plan_with(engine, table, None)
    }

    /// Like [`FamilyRegistry::plan`], with the objects of the process for a family that reads media.
    pub fn plan_with(
        &self,
        engine: &OpenedEngine,
        table: &toml::Table,
        objects: Option<&ObjectReader>,
    ) -> Result<Option<Plan>, String> {
        let identity = &engine.identity;
        let Some(opener) = self.openers.get(&identity.family) else {
            return Ok(None);
        };
        let opened = opener(identity, table, objects).map_err(|error| {
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
        let model = match opened.served {
            ServedName::Identity => &identity.model,
            ServedName::Engine => &engine.name,
        };
        let mut config = ServiceConfig::new(model, &engine.name);
        config.context_tokens =
            usize::try_from(tokens).map_err(|_| format!("a slot of {tokens} tokens is too large"))?;
        // A decision reads its context in the same slots (spec 11.3c).
        config.decision_context_tokens = config.context_tokens;
        config.speech = opened.speech;
        Ok(Some(Plan {
            family: opened.family,
            config,
        }))
    }
}

/// The Qwen3.8 family from the GGUF in the key `gguf` of the table, or from the first shard of `QWEN38_SHARDS`.
#[cfg(feature = "hip")]
fn open_qwen38(identity: &Identity, table: &toml::Table) -> Result<OpenedFamily, String> {
    let shard = match table.get("gguf") {
        Some(value) => std::path::PathBuf::from(
            value
                .as_str()
                .ok_or_else(|| "the key `gguf` of `[engines.family]` is not a string; write a path".to_owned())?,
        ),
        None => nagaya_mk_qwen38_gfx1151::Qwen38Paths::from_env()
            .map_err(|error| error.message().to_owned())?
            .shards
            .into_iter()
            .next()
            .ok_or_else(|| "QWEN38_SHARDS names no shard; give the three GGUF shards of the model".to_owned())?,
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
        speech: None,
        served: ServedName::Identity,
    })
}

/// The most tokens of a written answer to a `text` question (gogi `text_tokens`).
const TEXT_TOKENS: usize = 256;

/// An engine on demand that its factory describes: the facts that its open will report (`Factory::describe`).
#[derive(Clone, Debug)]
pub struct Described {
    /// The engine name of the profile.
    pub name: String,
    /// The identity that the open will report.
    pub identity: Identity,
    /// The limits that the open will report.
    pub limits: Limits,
}

/// The engines on demand of `profile` that their factories describe. They are not loaded at the start, and the
/// council serves each one whose family has a library (spec 11.1). A description that fails is an error, so the
/// process refuses to start; a kind with no description gives none.
pub fn describe_on_demand(
    profile: &PlacementProfile,
    registry: &Registry,
    inventory: &crate::inventory::Inventory,
) -> Result<Vec<Described>, String> {
    let mut described = Vec::new();
    for placement in profile
        .engines
        .iter()
        .filter(|placement| placement.residency == Residency::OnDemand)
    {
        let Some(factory) = registry.factory(&placement.kind) else {
            continue;
        };
        let devices = crate::admission::group_devices(placement, inventory)
            .map_err(|error| format!("cannot describe engine `{}`: {error}", placement.name))?;
        let facts = factory
            .describe(&placement.config, &devices)
            .map_err(|error| format!("cannot describe engine `{}`: {}", placement.name, error.message))?;
        if let Some((identity, limits)) = facts {
            described.push(Described {
                name: placement.name.clone(),
                identity,
                limits,
            });
        }
    }
    Ok(described)
}

/// The model folders and the lens files of the lab routes, by engine name (`docs/lab.md`): the folder of each LFM3
/// engine, and the J-Lens that its `[engines.lab]` table names. A lab table on another kind, a key other than `jlens`,
/// or a path that is not a string is an error that names the engine. The start does not read the lens.
#[allow(clippy::type_complexity)]
pub fn lab_files(
    profile: &PlacementProfile,
) -> Result<
    (
        BTreeMap<String, std::path::PathBuf>,
        BTreeMap<String, std::path::PathBuf>,
    ),
    String,
> {
    let (mut folders, mut lenses) = (BTreeMap::new(), BTreeMap::new());
    for placement in &profile.engines {
        let name = &placement.name;
        if placement.kind != crate::lfm3::KIND {
            if !placement.lab.is_empty() {
                return Err(format!(
                    "engine `{name}` of the kind `{}` has a table [engines.lab]; the lab reads only the kind `{}`, so \
                     remove the table",
                    placement.kind,
                    crate::lfm3::KIND
                ));
            }
            continue;
        }
        let what = format!("the config of engine `{name}`");
        folders.insert(name.clone(), crate::lfm3::model_dir(&placement.config, &what)?);
        for (key, value) in &placement.lab {
            if key != "jlens" {
                return Err(format!(
                    "[engines.lab] of engine `{name}` has the key `{key}`; its one key is `jlens`, the path of a J-Lens"
                ));
            }
            let path = value.as_str().ok_or_else(|| {
                format!("[engines.lab] jlens of engine `{name}` is {value}; write the path of the lens file")
            })?;
            lenses.insert(name.clone(), std::path::PathBuf::from(path));
        }
    }
    Ok((folders, lenses))
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
    council_with(families, profile, opened, &[], ports, store_dir, peers, call_cpus)
}

/// Like [`council`], and also a service for each described engine on demand, which is not loaded yet (spec 11.1).
#[allow(clippy::too_many_arguments)]
pub fn council_with(
    families: &FamilyRegistry,
    profile: &PlacementProfile,
    opened: &[OpenedEngine],
    described: &[Described],
    ports: &Ports,
    store_dir: &Path,
    peers: &[(String, String)],
    call_cpus: Option<Vec<u32>>,
) -> Result<Council, String> {
    let mut models = Vec::new();
    // The object store opens first: a family and a text service read the object parts of a request from it.
    let objects: Option<ObjectReader> = match profile.store.and_then(|store| store.objects) {
        None => None,
        Some(budget) => {
            let store = nagaya_store::objects::ObjectStore::open(&store_dir.join("objects"), budget)
                .map_err(|error| format!("cannot open the objects in the store directory: {error}"))?;
            tracing::info!("objects: {} of a budget of {budget} bytes held", store.used());
            Some(Arc::new(std::sync::Mutex::new(store)))
        }
    };
    // A described engine goes through the same family plan as an opened one, from the facts of its description.
    let as_opened: Vec<(OpenedEngine, bool)> = opened
        .iter()
        .cloned()
        .map(|engine| (engine, false))
        .chain(described.iter().map(|engine| {
            (
                OpenedEngine {
                    name: engine.name.clone(),
                    thread: std::thread::current().id(),
                    identity: engine.identity.clone(),
                    limits: engine.limits.clone(),
                    handoff: None,
                },
                true,
            )
        }))
        .collect();
    let mut plans = Vec::new();
    for (engine, is_described) in &as_opened {
        let placement = profile
            .engines
            .iter()
            .find(|placement| placement.name == engine.name)
            .ok_or_else(|| format!("the opened engine `{}` is not in the profile", engine.name))?;
        let Some(plan) = families.plan_with(engine, &placement.family, objects.as_ref())? else {
            tracing::info!(
                "engine `{}` has the family `{}`, which has no library, so it is not a served model",
                engine.name,
                engine.identity.family
            );
            continue;
        };
        plans.push((engine, *is_described, plan));
    }
    // The server holds a spec with a `text` question when one of its served models writes such a question (spec
    // 11.11, "Text questions of the LFM3 family"). The specs of the server have one set of rules.
    if plans.iter().any(|(_, _, plan)| plan.family.describes()) {
        for (_, _, plan) in &mut plans {
            plan.config.spec_rules.describe = true;
            plan.config.spec_rules.text_tokens = TEXT_TOKENS;
        }
    }
    // The home reads and pins the snapshots of each text engine and each speech engine (spec 11.7, Proposed).
    let texts: Vec<(String, String)> = plans
        .iter()
        .map(|(engine, _, plan)| (engine.name.clone(), plan.config.model.clone()))
        .collect();
    // The specs and the pin book of the server open once, with the spec rules of the home.
    let files = plans
        .first()
        .map(|(_, _, plan)| ServerFiles::open(store_dir, plan.config.spec_rules, plan.config.context_records))
        .transpose()
        .map_err(|error| format!("cannot open the specs and the pins of the server: {error}"))?;
    for (engine, is_described, mut plan) in plans {
        plan.config.peers = texts
            .iter()
            .chain(peers)
            .filter(|(peer, _)| *peer != engine.name)
            .cloned()
            .collect();
        let files = files.clone().expect("a plan opened the files of the server");
        let described = is_described.then(|| (engine.identity.clone(), engine.limits.clone()));
        let mut service = Service::attach_to(ports.clone(), plan.family, files, plan.config, described)
            .map_err(|error| format!("cannot serve engine `{}`: {error}", engine.name))?;
        if let Some(objects) = &objects {
            service = service.with_objects(Arc::clone(objects));
        }
        if is_described {
            tracing::info!(
                "serving model `{}` on engine `{}`, which loads on demand",
                service.model(),
                engine.name
            );
        } else {
            tracing::info!("serving model `{}` on engine `{}`", service.model(), engine.name);
        }
        models.push(Served {
            service,
            aliases: Vec::new(),
            device: None,
        });
    }
    let (folders, lenses) = lab_files(profile)?;
    let council = Council::new(models, call_cpus)?.with_lab(folders).with_lenses(lenses);
    Ok(match objects {
        None => council,
        Some(objects) => council.with_object_store(objects),
    })
}
