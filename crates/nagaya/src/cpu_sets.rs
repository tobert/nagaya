//! The CPU sets of spec 3.2a: the housekeeping set and the pool set.
//!
//! The split is a pure function. It takes the online CPUs and gives two sets of CPU numbers. Thread start applies the
//! sets (`affinity`), and a test checks them.

use std::fmt;

use crate::exit::ExitCode;
use crate::inventory::ProbeError;
use crate::probe::FileProbe;

/// The number of online CPUs in the housekeeping set (spec 3.2a).
pub const HOUSEKEEPING_CPUS: usize = 2;

/// The fewest online CPUs that can make two sets. The pool set needs one CPU.
pub const MIN_ONLINE_CPUS: usize = HOUSEKEEPING_CPUS + 1;

/// The two sets of CPU numbers. The sets do not overlap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuSets {
    /// The lowest-numbered online CPUs. Control threads run here.
    pub housekeeping: Vec<u32>,
    /// Every other online CPU. The CPU worker pool runs here.
    pub pool: Vec<u32>,
}

impl CpuSets {
    /// The number of threads in the CPU worker pool. It is one for each CPU in the pool set.
    pub fn pool_size(&self) -> usize {
        self.pool.len()
    }
}

/// A reason that the online CPUs cannot make two sets.
#[derive(Debug, PartialEq, Eq)]
pub enum CpuSetError {
    /// The machine has too few online CPUs for both sets.
    TooFewCpus { online: usize, needed: usize },
    /// The list is not sorted, or a CPU number repeats. `index` is the first position that breaks the order.
    NotIncreasing { index: usize },
}

impl fmt::Display for CpuSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewCpus { online, needed } => write!(
                f,
                "Cannot split the CPUs: {online} online, and {needed} are needed. Run the process on a machine with more \
                 online CPUs."
            ),
            Self::NotIncreasing { index } => write!(
                f,
                "The online CPU list breaks the order at index {index}. Pass each CPU number once, in increasing order."
            ),
        }
    }
}

impl std::error::Error for CpuSetError {}

/// Split the online CPUs into the housekeeping set and the pool set (spec 3.2a).
///
/// `online` must hold each online CPU once, in increasing order. Offline CPUs are not in the list, so the numbers can
/// have holes.
pub fn split(online: &[u32]) -> Result<CpuSets, CpuSetError> {
    if let Some(position) = online.windows(2).position(|pair| pair[0] >= pair[1]) {
        return Err(CpuSetError::NotIncreasing { index: position + 1 });
    }
    if online.len() < MIN_ONLINE_CPUS {
        return Err(CpuSetError::TooFewCpus {
            online: online.len(),
            needed: MIN_ONLINE_CPUS,
        });
    }
    let (housekeeping, pool) = online.split_at(HOUSEKEEPING_CPUS);
    Ok(CpuSets {
        housekeeping: housekeeping.to_vec(),
        pool: pool.to_vec(),
    })
}

/// Why the process cannot make its CPU sets at start.
#[derive(Debug, PartialEq, Eq)]
pub enum CpuSetupError {
    /// The probe could not read the online CPU list.
    Probe(ProbeError),
    /// The online CPUs cannot make two sets.
    Split(CpuSetError),
}

impl CpuSetupError {
    /// The exit code of the failure (spec 3.2a, 10.4). A restart cannot change the machine, so the start is refused.
    pub fn exit_code(&self) -> ExitCode {
        ExitCode::RefusedStart
    }
}

impl fmt::Display for CpuSetupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Probe(error) => write!(f, "cannot read the online CPUs: {error}"),
            Self::Split(error) => write!(f, "cannot make the CPU sets: {error}"),
        }
    }
}

impl std::error::Error for CpuSetupError {}

/// Make the CPU sets of this machine from the online CPU list of the file probe (spec 3.2a, 2.1).
pub fn for_machine(probe: &FileProbe) -> Result<CpuSets, CpuSetupError> {
    let online = probe.online_cpus().map_err(CpuSetupError::Probe)?;
    split(&online).map_err(CpuSetupError::Split)
}
