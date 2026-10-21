//! The start steps of an engine with an arena (spec 10.2): attach a stored arena that matches, else drop the stored
//! fds and load the engine cold, and store the arena of a cold load.

use std::os::fd::OwnedFd;

use nagaya_abi::{ArenaFd, Engine};

use super::{GroupJob, Handoff};
use crate::admission::{AttachError, OpenError, OpenErrorKind};
use crate::arena::{FdName, FdStore, Orphan};

/// Drop the stored fds that no engine of the profile can use. The caller calls this after admission passes.
pub(super) fn drop_orphans(orphans: Vec<Orphan>, fd_store: Option<&FdStore>) {
    for orphan in orphans {
        tracing::warn!(
            "the stored fd `{}` is not used: {}; the process drops it",
            orphan.name,
            orphan.why
        );
        remove(fd_store, &orphan.name);
        drop(orphan.fd);
    }
}

/// Ask systemd to drop the stored fds under `name`. A failure costs only a cold load at the next start.
fn remove(fd_store: Option<&FdStore>, name: &str) {
    if let Some(store) = fd_store
        && let Err(error) = store.remove(name)
    {
        tracing::warn!("cannot remove the stored fd `{name}` from the fd store: {error}");
    }
}

fn other(message: String) -> OpenError {
    OpenError {
        kind: OpenErrorKind::Other,
        message,
    }
}

/// The fd names of the admitted arena parts of a job, in name order.
fn expected_names(job: &GroupJob) -> Result<Vec<FdName>, OpenError> {
    let mut names = job
        .arena_parts
        .iter()
        .map(|part| {
            let device = job
                .devices
                .get(part.device)
                .unwrap_or_else(|| panic!("admission passed the arena part `{}` on no device", part.part));
            FdName::new(&job.placement.name, device.id, &part.part)
                .unwrap_or_else(|why| panic!("admission passed an arena part with a bad name: {why}"))
        })
        .collect::<Vec<_>>();
    names.sort();
    if names.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(other(format!(
            "the engine `{}` gives two arena parts with one name; fix `arena_parts`",
            job.placement.name
        )));
    }
    Ok(names)
}

/// Give `attach` a duplicate of each stored fd (spec 10.2 step 3). The group index replaces the machine device id.
fn duplicates(job: &GroupJob) -> Result<Vec<ArenaFd>, String> {
    job.stored
        .iter()
        .map(|(name, fd)| {
            let device = job
                .devices
                .iter()
                .position(|device| device.id == name.device)
                .ok_or_else(|| format!("the stored fd `{name}` is on a device that the engine does not use"))?;
            let fd = fd
                .try_clone()
                .map_err(|error| format!("cannot duplicate the stored fd `{name}`: {error}"))?;
            Ok(ArenaFd {
                device,
                part: name.part.clone(),
                fd,
            })
        })
        .collect()
}

/// Open the engine of `job` by the start steps of spec 10.2. The stored fds of the job are used or dropped.
pub(super) fn open(
    job: &mut GroupJob,
    fd_store: Option<&FdStore>,
) -> Result<(Box<dyn Engine>, Option<Handoff>), OpenError> {
    let expected = expected_names(job)?;
    let name = job.placement.name.clone();
    if !job.stored.is_empty() {
        let stored: Vec<&FdName> = job.stored.iter().map(|(name, _)| name).collect();
        let why = if stored.iter().copied().eq(expected.iter()) {
            match duplicates(job) {
                Ok(arenas) => match job.factory.attach(&job.placement.config, &job.devices, arenas) {
                    Ok(attached) => {
                        tracing::info!(
                            "engine `{name}` attached its stored arena: {} fds, check {}",
                            job.stored.len(),
                            attached.check.as_str()
                        );
                        // The engine holds its own imports, and systemd holds its copies.
                        job.stored.clear();
                        return Ok((attached.engine, Some(Handoff::Hot(attached.check))));
                    }
                    Err(AttachError::Mismatch(why)) => why,
                    // The arena matched, so the process keeps the stored fds and stops (spec 10.4).
                    Err(AttachError::Open(error)) => return Err(error),
                },
                Err(why) => why,
            }
        } else {
            let names: Vec<String> = stored.iter().map(ToString::to_string).collect();
            let parts: Vec<String> = expected.iter().map(ToString::to_string).collect();
            format!(
                "the stored fds are [{}], and the arena parts are [{}]",
                names.join(", "),
                parts.join(", ")
            )
        };
        tracing::warn!(
            "engine `{name}` does not use its stored arena: {why}; the process drops {} fds and loads it cold",
            job.stored.len()
        );
        for (stored_name, fd) in std::mem::take(&mut job.stored) {
            remove(fd_store, &stored_name.to_string());
            drop::<OwnedFd>(fd);
        }
    }
    let mut engine = job.factory.open(&job.placement.config, &job.devices)?;
    if expected.is_empty() {
        return Ok((engine, None));
    }
    if let Some(store) = fd_store {
        store_arena(&name, engine.as_mut(), job, &expected, store)?;
    }
    Ok((engine, Some(Handoff::Cold)))
}

/// Export the arena of a cold load and send each fd to the fd store under its name.
fn store_arena(
    name: &str,
    engine: &mut dyn Engine,
    job: &GroupJob,
    expected: &[FdName],
    store: &FdStore,
) -> Result<(), OpenError> {
    let arena = engine.arena().ok_or_else(|| {
        other(format!(
            "the engine `{name}` gives arena parts, and it has no arena capability; fix the engine"
        ))
    })?;
    let fds = arena
        .arena_export()
        .map_err(|why| other(format!("the engine `{name}` cannot export its arena: {why}")))?;
    let mut named = Vec::with_capacity(fds.len());
    for fd in fds {
        let device = job.devices.get(fd.device).ok_or_else(|| {
            other(format!(
                "the engine `{name}` exports the part `{}` on device {} of a group of {}",
                fd.part,
                fd.device,
                job.devices.len()
            ))
        })?;
        let fd_name = FdName::new(name, device.id, &fd.part).map_err(other)?;
        named.push((fd_name, fd.fd));
    }
    named.sort_by(|a, b| a.0.cmp(&b.0));
    if !named.iter().map(|(name, _)| name).eq(expected.iter()) {
        let got: Vec<String> = named.iter().map(|(name, _)| name.to_string()).collect();
        let parts: Vec<String> = expected.iter().map(ToString::to_string).collect();
        return Err(other(format!(
            "the engine `{name}` exports [{}], and its arena parts are [{}]; fix the engine",
            got.join(", "),
            parts.join(", ")
        )));
    }
    let mut stored = 0;
    for (fd_name, fd) in &named {
        match store.store(fd_name, fd) {
            Ok(()) => stored += 1,
            Err(error) => tracing::warn!(
                "cannot send the arena fd `{fd_name}` to the fd store: {error}; the next start loads `{name}` cold"
            ),
        }
    }
    tracing::info!("engine `{name}` stored {stored} arena fds");
    Ok(())
}
