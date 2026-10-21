//! Device groups and the device check (spec 3.2).

use nagaya_abi::DeviceId;

use crate::admission::AdmittedEngine;
use crate::inventory::Inventory;
use crate::profile::PlacementProfile;

/// A set of devices that one or more engines use. One thread owns the group (spec 3.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceGroup {
    /// The devices in the group, in the order of the first engine that names them.
    pub devices: Vec<DeviceId>,
    /// The engines in the group, by their index in the profile.
    pub engines: Vec<usize>,
}

/// Group the profile's engines by their devices. Two engines that use one device share a group (spec 3.2).
///
/// The overlap is transitive: an engine that uses a device of two groups joins them into one. The groups keep the
/// order of their first engine. The devices of a group are the union, in the order of first appearance.
pub fn device_groups(profile: &PlacementProfile) -> Vec<DeviceGroup> {
    let mut groups: Vec<DeviceGroup> = Vec::new();
    for (index, engine) in profile.engines.iter().enumerate() {
        let mut merged = DeviceGroup {
            devices: Vec::new(),
            engines: Vec::new(),
        };
        let mut rest = Vec::with_capacity(groups.len());
        let mut slot = None;
        for group in groups {
            if group.devices.iter().any(|device| engine.devices.contains(device)) {
                slot.get_or_insert(rest.len());
                merge_into(&mut merged, group);
            } else {
                rest.push(group);
            }
        }
        merge_into(
            &mut merged,
            DeviceGroup {
                devices: engine.devices.clone(),
                engines: vec![index],
            },
        );
        rest.insert(slot.unwrap_or(rest.len()), merged);
        groups = rest;
    }
    groups
}

fn merge_into(target: &mut DeviceGroup, other: DeviceGroup) {
    for device in other.devices {
        if !target.devices.contains(&device) {
            target.devices.push(device);
        }
    }
    target.engines.extend(other.engines);
    target.engines.sort_unstable();
}

/// Why an engine's devices cannot open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceError {
    /// The inventory has no device with this id.
    Unknown {
        /// The engine name.
        engine: String,
        /// The device id.
        device: DeviceId,
    },
    /// The profile names a device that `ignore_devices` ignores (spec 2.1).
    Ignored {
        /// The engine name.
        engine: String,
        /// The device id.
        device: DeviceId,
        /// The PCI address of the device.
        pci_address: String,
    },
    /// The profile names a device that the plan does not.
    NotPlanned {
        /// The engine name.
        engine: String,
        /// The device id.
        device: DeviceId,
    },
    /// The plan names a device that the profile does not.
    NotInProfile {
        /// The engine name.
        engine: String,
        /// The device id.
        device: DeviceId,
    },
    /// The profile and the plan name the same device at different positions.
    Order {
        /// The engine name.
        engine: String,
        /// The device id.
        device: DeviceId,
    },
}

impl std::fmt::Display for DeviceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceError::Unknown { engine, device } => {
                write!(
                    formatter,
                    "engine `{engine}` names device {device}, but the inventory does not have it"
                )
            }
            DeviceError::Ignored {
                engine,
                device,
                pci_address,
            } => {
                write!(
                    formatter,
                    "engine `{engine}` names device {device} at PCI address {pci_address}, but the bootstrap config \
                     ignores that device; name another device in the profile, or remove the address from ignore_devices"
                )
            }
            DeviceError::NotPlanned { engine, device } => {
                write!(
                    formatter,
                    "engine `{engine}` names device {device}, but its plan does not"
                )
            }
            DeviceError::NotInProfile { engine, device } => {
                write!(
                    formatter,
                    "engine `{engine}` plans device {device}, but the profile does not name it"
                )
            }
            DeviceError::Order { engine, device } => {
                write!(
                    formatter,
                    "engine `{engine}` names device {device} at another position than its plan does"
                )
            }
        }
    }
}

impl std::error::Error for DeviceError {}

/// Refuse a profile that names an ignored device (spec 2.1). Call it before `check_devices`.
pub fn check_ignored(profile: &PlacementProfile, inventory: &Inventory) -> Result<(), DeviceError> {
    for engine in &profile.engines {
        for id in &engine.devices {
            if let Some(device) = inventory.device(*id).filter(|device| device.ignored) {
                return Err(DeviceError::Ignored {
                    engine: engine.name.clone(),
                    device: *id,
                    pci_address: device.pci_address.clone(),
                });
            }
        }
    }
    Ok(())
}

/// Check each engine's devices against the inventory and against its planned limits.
pub fn check_devices(
    profile: &PlacementProfile,
    planned: &[AdmittedEngine],
    inventory: &Inventory,
) -> Result<(), DeviceError> {
    assert_eq!(
        profile.engines.len(),
        planned.len(),
        "the profile has {} engines but the plan has {}",
        profile.engines.len(),
        planned.len()
    );
    for (engine, planned) in profile.engines.iter().zip(planned) {
        assert_eq!(
            engine.name, planned.name,
            "the profile and the plan name different engines at one position"
        );
        for device in &engine.devices {
            if inventory.device(*device).is_none() {
                return Err(DeviceError::Unknown {
                    engine: engine.name.clone(),
                    device: *device,
                });
            }
        }
        for device in &engine.devices {
            if !planned.limits.devices.contains(device) {
                return Err(DeviceError::NotPlanned {
                    engine: engine.name.clone(),
                    device: *device,
                });
            }
        }
        for device in &planned.limits.devices {
            if !engine.devices.contains(device) {
                return Err(DeviceError::NotInProfile {
                    engine: engine.name.clone(),
                    device: *device,
                });
            }
        }
        for (index, device) in engine.devices.iter().enumerate() {
            if planned.limits.devices.get(index) != Some(device) {
                return Err(DeviceError::Order {
                    engine: engine.name.clone(),
                    device: *device,
                });
            }
        }
    }
    Ok(())
}
