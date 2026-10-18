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

/// Group the profile's engines by their device set. The groups keep the order of the profile.
pub fn device_groups(profile: &PlacementProfile) -> Vec<DeviceGroup> {
    let mut groups: Vec<(Vec<DeviceId>, DeviceGroup)> = Vec::new();
    for (index, engine) in profile.engines.iter().enumerate() {
        let key = device_key(&engine.devices);
        match groups.iter_mut().find(|(other, _)| *other == key) {
            Some((_, group)) => group.engines.push(index),
            None => {
                let group = DeviceGroup {
                    devices: engine.devices.clone(),
                    engines: vec![index],
                };
                groups.push((key, group));
            }
        }
    }
    groups.into_iter().map(|(_, group)| group).collect()
}

fn device_key(devices: &[DeviceId]) -> Vec<DeviceId> {
    let mut key = devices.to_vec();
    key.sort_unstable();
    key.dedup();
    key
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
