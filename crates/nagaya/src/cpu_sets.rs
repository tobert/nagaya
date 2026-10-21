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
    /// The set of each machine device, by device id (spec 3.2b). Empty when no device rule was built.
    pub devices: Vec<Vec<u32>>,
}

impl CpuSets {
    /// The set of the stage thread of a machine device (spec 3.2b).
    ///
    /// The plain split (`split`) builds no device rule, and then each device has the pool set. A set that has a device
    /// rule and no entry for `id` is a defect, and this call panics.
    pub fn device(&self, id: u32) -> &[u32] {
        if self.devices.is_empty() {
            return &self.pool;
        }
        self.devices
            .get(id as usize)
            .unwrap_or_else(|| panic!("the CPU sets have no entry for device {id}"))
    }

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
    /// The CPUs near a device hold no pool CPU. The device cannot get a set (spec 3.2b).
    NoPoolCpu { device: usize },
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
            Self::NoPoolCpu { device } => write!(
                f,
                "Device {device} has no pool CPU near it. Check the CPU lists of the device and of the L3 caches."
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
        devices: Vec::new(),
    })
}

/// Make the set of each device (spec 3.2b). `local` holds the `local_cpulist` of each device, by device id.
///
/// - A machine with fewer than two devices has nothing to separate, so the device keeps the pool set.
/// - A device whose local list holds every online CPU has no locality. It takes one L3 domain, in the order of the
///   device ids. With more such devices than domains, the devices take the domains in turn.
/// - Another device takes the CPUs of its local list.
///
/// Each set holds only pool CPUs. A set with none is an error.
pub fn device_sets(
    pool: &[u32],
    online: &[u32],
    domains: &[Vec<u32>],
    local: &[Vec<u32>],
) -> Result<Vec<Vec<u32>>, CpuSetError> {
    if local.len() < 2 {
        return Ok(local.iter().map(|_| pool.to_vec()).collect());
    }
    let mut without_locality = 0;
    let mut sets = Vec::with_capacity(local.len());
    for (device, cpus) in local.iter().enumerate() {
        let near: &[u32] = if online.iter().all(|cpu| cpus.contains(cpu)) && !domains.is_empty() {
            let domain = &domains[without_locality % domains.len()];
            without_locality += 1;
            domain
        } else {
            cpus
        };
        let set: Vec<u32> = pool.iter().copied().filter(|cpu| near.contains(cpu)).collect();
        if set.is_empty() {
            return Err(CpuSetError::NoPoolCpu { device });
        }
        sets.push(set);
    }
    Ok(sets)
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

/// Make the CPU sets, and the set of each device, from the file probe (spec 3.2a, 3.2b).
///
/// The probe lists the devices and reads their CPU lists only when the machine has two or more. A machine with one
/// device needs no separation, so its files are not read.
pub fn for_machine_with_devices(probe: &FileProbe) -> Result<CpuSets, CpuSetupError> {
    let mut sets = for_machine(probe)?;
    let machine = probe.machine().map_err(CpuSetupError::Probe)?;
    let count = machine.devices.len();
    if count < 2 {
        sets.devices = vec![sets.pool.clone(); count];
        return Ok(sets);
    }
    let local = machine
        .devices
        .iter()
        .map(|facts| probe.local_cpus(&facts.device.pci_address))
        .collect::<Result<Vec<_>, _>>()
        .map_err(CpuSetupError::Probe)?;
    let domains = probe.l3_domains(&machine.cpus).map_err(CpuSetupError::Probe)?;
    sets.devices = device_sets(&sets.pool, &machine.cpus, &domains, &local).map_err(CpuSetupError::Split)?;
    Ok(sets)
}
