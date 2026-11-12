//! The evidence on failure (spec 9.6).
//!
//! Before and after a quantum that fails, the nagaya reads the counters of the machine. It logs the change with the
//! failure. The counters come from `service/telemetry.py` of the reference: the KFD `evicted_ms` of this process, `MemAvailable`,
//! the GTT in use, and four counters of `/proc/vmstat`.
//!
//! The code that runs a quantum calls [`Machine::watch`] before the quantum. When the quantum fails, it calls
//! [`Watch::failed`] with the failure. Each `watch` reads four files, so a caller may skip it for quanta that run
//! many times in one second, such as decode.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

const GIB: f64 = (1u64 << 30) as f64;
const MIB: u64 = 1u64 << 20;
const KIB: u64 = 1u64 << 10;

/// Why the counters cannot be read or compared.
#[derive(Debug)]
pub enum EvidenceError {
    /// A file cannot be read.
    Io {
        /// The file.
        path: PathBuf,
        /// What the operating system said.
        source: io::Error,
    },
    /// A file has no value for something the nagaya needs.
    Missing {
        /// The file.
        path: PathBuf,
        /// The counters that the file lacks.
        what: String,
    },
    /// A file holds text that is not a number.
    Parse {
        /// The file.
        path: PathBuf,
        /// The text.
        text: String,
    },
    /// A counter that only grows is smaller than before. It is another process's counter, or the kernel reset it.
    WentBack {
        /// The counter.
        counter: String,
        /// The value before.
        before: u64,
        /// The value after.
        after: u64,
    },
}

impl fmt::Display for EvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvidenceError::Io { path, source } => write!(formatter, "cannot read {}: {source}", path.display()),
            EvidenceError::Missing { path, what } => write!(formatter, "{} has no {what}", path.display()),
            EvidenceError::Parse { path, text } => {
                write!(
                    formatter,
                    "{} holds `{}`, which is not a number",
                    path.display(),
                    text.trim()
                )
            }
            EvidenceError::WentBack { counter, before, after } => write!(
                formatter,
                "{counter} went back from {before} to {after}; it grows only, so it is another counter or the kernel reset it"
            ),
        }
    }
}

impl std::error::Error for EvidenceError {}

/// The counters of the machine at one moment (spec 9.6).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Counters {
    /// The milliseconds that KFD kept the GPU queues of this process evicted. `None` while the process has no queues.
    pub kfd_evicted_ms: Option<u64>,
    /// `MemAvailable` in bytes.
    pub mem_available_bytes: u64,
    /// The GTT in use, in bytes, summed over the GPUs. `None` on a machine with no amdgpu device.
    pub gtt_used_bytes: Option<u64>,
    /// `compact_stall` of `/proc/vmstat`.
    pub compact_stall: u64,
    /// `pgmigrate_success` of `/proc/vmstat`.
    pub pgmigrate_success: u64,
    /// `pswpout` of `/proc/vmstat`.
    pub pswpout: u64,
    /// `pgscan_direct` of `/proc/vmstat`.
    pub pgscan_direct: u64,
    /// The `drm-total-*` and `drm-resident-*` bytes of the `xe` DRM clients of this process (spec 9.6).
    pub xe_memory_bytes: BTreeMap<String, u64>,
    /// The throttle status and reasons of each Xe GT (`<card>/<tile>/<gt>` -> `(status, reasons)`).
    pub xe_throttle: BTreeMap<String, (u64, String)>,
}

/// What the counters did between two reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Change {
    kfd_evicted_ms: Option<u64>,
    mem_available: (u64, u64),
    gtt: Option<(u64, u64)>,
    vmstat: [(&'static str, u64); 4],
    xe_memory: BTreeMap<String, (u64, u64)>,
    xe_throttle: BTreeMap<String, (u64, u64, String)>,
}

impl fmt::Display for Change {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kfd_evicted_ms {
            Some(ms) => write!(formatter, "KFD evicted {ms} ms")?,
            None => write!(formatter, "KFD evicted n/a")?,
        }
        write!(
            formatter,
            "; MemAvailable {:.1} -> {:.1} GiB",
            self.mem_available.0 as f64 / GIB,
            self.mem_available.1 as f64 / GIB
        )?;
        if let Some((before, after)) = self.gtt {
            write!(
                formatter,
                "; GTT {:.1} -> {:.1} GiB",
                before as f64 / GIB,
                after as f64 / GIB
            )?;
        }
        for (name, (before, after)) in &self.xe_memory {
            write!(
                formatter,
                "; Xe {name} {:.1} -> {:.1} GiB",
                *before as f64 / GIB,
                *after as f64 / GIB
            )?;
        }
        for (gt, (before, after, reasons)) in &self.xe_throttle {
            write!(formatter, "; Xe {gt} throttle {before} -> {after} ({reasons})")?;
        }
        for (name, moved) in self.vmstat {
            write!(formatter, "; {name} +{moved}")?;
        }
        Ok(())
    }
}

fn grown(counter: &str, before: u64, after: u64) -> Result<u64, EvidenceError> {
    after.checked_sub(before).ok_or_else(|| EvidenceError::WentBack {
        counter: counter.to_owned(),
        before,
        after,
    })
}

impl Counters {
    /// What changed since `before`. A counter that only grows and went back is an error.
    pub fn change_since(&self, before: &Counters) -> Result<Change, EvidenceError> {
        let kfd_evicted_ms = match (before.kfd_evicted_ms, self.kfd_evicted_ms) {
            (None, None) => None,
            (None, Some(after)) => Some(after),
            (Some(before), Some(after)) => Some(grown("KFD evicted_ms", before, after)?),
            (Some(before), None) => return Err(grown("KFD evicted_ms", before, 0).unwrap_err()),
        };
        let gtt = match (before.gtt_used_bytes, self.gtt_used_bytes) {
            (Some(before), Some(after)) => Some((before, after)),
            _ => None,
        };
        let mut xe_memory = BTreeMap::new();
        for (name, after) in &self.xe_memory_bytes {
            let before_bytes = before.xe_memory_bytes.get(name).copied().unwrap_or(0);
            xe_memory.insert(name.clone(), (before_bytes, *after));
        }
        let mut xe_throttle = BTreeMap::new();
        for (gt, (after_status, after_reasons)) in &self.xe_throttle {
            let before_status = before.xe_throttle.get(gt).map_or(0, |(status, _)| *status);
            xe_throttle.insert(gt.clone(), (before_status, *after_status, after_reasons.clone()));
        }
        Ok(Change {
            kfd_evicted_ms,
            mem_available: (before.mem_available_bytes, self.mem_available_bytes),
            gtt,
            vmstat: [
                (
                    "compact_stall",
                    grown("compact_stall", before.compact_stall, self.compact_stall)?,
                ),
                (
                    "pgmigrate_success",
                    grown("pgmigrate_success", before.pgmigrate_success, self.pgmigrate_success)?,
                ),
                ("pswpout", grown("pswpout", before.pswpout, self.pswpout)?),
                (
                    "pgscan_direct",
                    grown("pgscan_direct", before.pgscan_direct, self.pgscan_direct)?,
                ),
            ],
            xe_memory,
            xe_throttle,
        })
    }
}

/// The files of one machine, under a root. The real machine has the root `/`. A test has a fixed tree.
#[derive(Clone, Debug)]
pub struct Machine {
    root: PathBuf,
    pid: u32,
}

fn read_text(path: &Path) -> Result<String, EvidenceError> {
    std::fs::read_to_string(path).map_err(|source| EvidenceError::Io {
        path: path.to_owned(),
        source,
    })
}

fn read_number(path: &Path) -> Result<u64, EvidenceError> {
    let text = read_text(path)?;
    text.trim().parse().map_err(|_| EvidenceError::Parse {
        path: path.to_owned(),
        text,
    })
}

fn parse_drm_bytes(path: &Path, text: &str) -> Result<u64, EvidenceError> {
    let mut parts = text.split_whitespace();
    let Some(number) = parts.next().and_then(|part| part.parse::<u64>().ok()) else {
        return Err(EvidenceError::Parse {
            path: path.to_owned(),
            text: text.to_owned(),
        });
    };
    let scale = match parts.next() {
        None | Some("B") => 1,
        Some("KiB") | Some("kB") => KIB,
        Some("MiB") => MIB,
        Some("GiB") => 1u64 << 30,
        Some(_) => {
            return Err(EvidenceError::Parse {
                path: path.to_owned(),
                text: text.to_owned(),
            });
        }
    };
    Ok(number.saturating_mul(scale))
}

impl Machine {
    /// The machine that this process runs on.
    pub fn real() -> Self {
        Self::at("/", std::process::id())
    }

    /// A machine whose files are under `root`, for the process `pid`.
    pub fn at(root: impl Into<PathBuf>, pid: u32) -> Self {
        Self { root: root.into(), pid }
    }

    /// Read the counters.
    pub fn read(&self) -> Result<Counters, EvidenceError> {
        let vmstat = self.root.join("proc/vmstat");
        let text = read_text(&vmstat)?;
        let find = |name: &str| -> Option<u64> {
            text.lines().find_map(|line| {
                let (key, value) = line.split_once(' ')?;
                (key == name).then(|| value.trim().parse().ok()).flatten()
            })
        };
        let names = ["compact_stall", "pgmigrate_success", "pswpout", "pgscan_direct"];
        let missing: Vec<&str> = names.iter().copied().filter(|name| find(name).is_none()).collect();
        if !missing.is_empty() {
            return Err(EvidenceError::Missing {
                path: vmstat,
                what: missing.join(", "),
            });
        }
        Ok(Counters {
            kfd_evicted_ms: self.kfd_evicted_ms()?,
            mem_available_bytes: self.mem_available()?,
            gtt_used_bytes: self.gtt_used()?,
            compact_stall: find("compact_stall").expect("checked above"),
            pgmigrate_success: find("pgmigrate_success").expect("checked above"),
            pswpout: find("pswpout").expect("checked above"),
            pgscan_direct: find("pgscan_direct").expect("checked above"),
            xe_memory_bytes: self.xe_memory()?,
            xe_throttle: self.xe_throttle()?,
        })
    }

    fn mem_available(&self) -> Result<u64, EvidenceError> {
        let path = self.root.join("proc/meminfo");
        let text = read_text(&path)?;
        let line = text.lines().find(|line| line.starts_with("MemAvailable:"));
        let Some(line) = line else {
            return Err(EvidenceError::Missing {
                path,
                what: "MemAvailable".to_owned(),
            });
        };
        let kib = line
            .split_whitespace()
            .nth(1)
            .and_then(|field| field.parse::<u64>().ok());
        match kib {
            Some(kib) => Ok(kib * 1024),
            None => Err(EvidenceError::Parse {
                path,
                text: line.to_owned(),
            }),
        }
    }

    /// The `evicted_ms` of every GPU of this process, summed. `None` while the process has no GPU queues.
    fn kfd_evicted_ms(&self) -> Result<Option<u64>, EvidenceError> {
        let directory = self.root.join(format!("sys/class/kfd/kfd/proc/{}", self.pid));
        let entries = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(source) => {
                return Err(EvidenceError::Io {
                    path: directory,
                    source,
                });
            }
        };
        let mut total = None;
        for entry in entries {
            let entry = entry.map_err(|source| EvidenceError::Io {
                path: directory.clone(),
                source,
            })?;
            if !entry.file_name().to_string_lossy().starts_with("stats_") {
                continue;
            }
            let value = read_number(&entry.path().join("evicted_ms"))?;
            total = Some(total.unwrap_or(0) + value);
        }
        Ok(total)
    }

    /// The GTT in use on every amdgpu device, summed. `None` when no device has the file.
    fn gtt_used(&self) -> Result<Option<u64>, EvidenceError> {
        let directory = self.root.join("sys/class/drm");
        let entries = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(source) => {
                return Err(EvidenceError::Io {
                    path: directory,
                    source,
                });
            }
        };
        let mut total = None;
        for entry in entries {
            let entry = entry.map_err(|source| EvidenceError::Io {
                path: directory.clone(),
                source,
            })?;
            let file = entry.path().join("device/mem_info_gtt_used");
            if !file.is_file() {
                continue;
            }
            total = Some(total.unwrap_or(0) + read_number(&file)?);
        }
        Ok(total)
    }

    /// The `drm-total-*` and `drm-resident-*` bytes of each `xe` DRM client in `/proc/<pid>/fdinfo`, summed once per
    /// `drm-client-id` (spec 9.6).
    fn xe_memory(&self) -> Result<BTreeMap<String, u64>, EvidenceError> {
        let directory = self.root.join(format!("proc/{}/fdinfo", self.pid));
        let entries = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
            Err(source) => {
                return Err(EvidenceError::Io {
                    path: directory,
                    source,
                });
            }
        };
        let mut seen_clients = BTreeSet::new();
        let mut totals = BTreeMap::new();
        for entry in entries {
            let entry = entry.map_err(|source| EvidenceError::Io {
                path: directory.clone(),
                source,
            })?;
            let path = entry.path();
            let text = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                // An fd can close while the read walks `/proc/<pid>/fdinfo`.
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(source) => return Err(EvidenceError::Io { path, source }),
            };
            let mut is_xe = false;
            let mut client_id = None;
            let mut pairs = Vec::new();
            for line in text.lines() {
                let Some((key, raw)) = line.split_once(':') else {
                    continue;
                };
                let key = key.trim();
                let value = raw.trim();
                if key == "drm-driver" && value == "xe" {
                    is_xe = true;
                } else if key == "drm-client-id" {
                    client_id = Some(value.to_owned());
                } else if key.starts_with("drm-total-") || key.starts_with("drm-resident-") {
                    pairs.push((key.to_owned(), value.to_owned()));
                }
            }
            if !is_xe {
                continue;
            }
            if let Some(id) = client_id
                && !seen_clients.insert(id)
            {
                continue;
            }
            for (key, value) in pairs {
                let bytes = parse_drm_bytes(&path, &value)?;
                *totals.entry(key).or_insert(0) += bytes;
            }
        }
        Ok(totals)
    }

    /// The throttle status and reasons of each Xe GT under `/sys/class/drm/card*/device/tile*/gt*/freq0/throttle`.
    fn xe_throttle(&self) -> Result<BTreeMap<String, (u64, String)>, EvidenceError> {
        let directory = self.root.join("sys/class/drm");
        let cards = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
            Err(source) => {
                return Err(EvidenceError::Io {
                    path: directory,
                    source,
                });
            }
        };
        let mut out = BTreeMap::new();
        for card in cards {
            let card = card.map_err(|source| EvidenceError::Io {
                path: directory.clone(),
                source,
            })?;
            let card_name = card.file_name().to_string_lossy().into_owned();
            if !card_name.starts_with("card") || card_name.contains('-') {
                continue;
            }
            let device_dir = card.path().join("device");
            let Ok(tiles) = std::fs::read_dir(&device_dir) else {
                continue;
            };
            for tile in tiles.flatten() {
                let tile_name = tile.file_name().to_string_lossy().into_owned();
                if !tile_name.starts_with("tile") {
                    continue;
                }
                let Ok(gts) = std::fs::read_dir(tile.path()) else {
                    continue;
                };
                for gt in gts.flatten() {
                    let gt_name = gt.file_name().to_string_lossy().into_owned();
                    if !gt_name.starts_with("gt") {
                        continue;
                    }
                    let throttle = gt.path().join("freq0/throttle");
                    let status_path = throttle.join("status");
                    if !status_path.is_file() {
                        continue;
                    }
                    let status = read_number(&status_path)?;
                    let reasons_path = throttle.join("reasons");
                    let reasons = if reasons_path.is_file() {
                        read_text(&reasons_path)?.trim().to_owned()
                    } else {
                        String::new()
                    };
                    out.insert(format!("{card_name}/{tile_name}/{gt_name}"), (status, reasons));
                }
            }
        }
        Ok(out)
    }

    /// Read the counters before a quantum. A read that fails is kept, and [`Watch::failed`] names it.
    pub fn watch(&self) -> Watch {
        Watch {
            machine: self.clone(),
            before: self.read(),
        }
    }
}

/// Log a failed quantum that had no watch, and give the line.
///
/// A decode quantum has none (see the module). The line says so, so that no failure is logged without a reason for the
/// missing evidence.
pub fn failed_unwatched(failure: &str) -> String {
    let line = format!("quantum failed: {failure}; evidence unavailable (not read before a decode quantum)");
    tracing::error!("{line}");
    line
}

/// The counters from before a quantum.
pub struct Watch {
    machine: Machine,
    before: Result<Counters, EvidenceError>,
}

impl Watch {
    /// The quantum failed. Read the counters again, log the failure with the change, and give the line.
    ///
    /// Evidence never hides the failure that it belongs to. A counter that cannot be read or compared is named in
    /// the line as `evidence unavailable (<why>)`, and the failure is logged all the same.
    pub fn failed(self, failure: &str) -> String {
        let after = self.machine.read();
        let evidence = match (&self.before, &after) {
            (Ok(before), Ok(after)) => match after.change_since(before) {
                Ok(change) => change.to_string(),
                Err(error) => format!("evidence unavailable ({error})"),
            },
            (Err(error), _) | (_, Err(error)) => format!("evidence unavailable ({error})"),
        };
        let line = format!("quantum failed: {failure}; {evidence}");
        tracing::error!("{line}");
        line
    }
}
