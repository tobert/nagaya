//! The threads of this process, by name, from `/proc/self/task`.

#![allow(dead_code)]

use std::fs;
use std::io::ErrorKind;
use std::path::Path;

/// The name of the thread in this task directory, or `None` when the thread ended after the listing.
///
/// A thread that ended before the open gives `ENOENT`, and one that ended after the open gives `ESRCH`. Each other
/// error stops the test.
pub fn thread_name(task: &Path) -> Option<String> {
    match fs::read_to_string(task.join("comm")) {
        Ok(name) => Some(name.trim().to_owned()),
        Err(error) if error.kind() == ErrorKind::NotFound || error.raw_os_error() == Some(libc::ESRCH) => None,
        Err(error) => panic!("cannot read the name of the thread in {}: {error}", task.display()),
    }
}

/// The ids of the threads of this process that have the name, in the order of the listing.
pub fn threads_named(name: &str) -> Vec<i32> {
    let mut found = Vec::new();
    for entry in fs::read_dir("/proc/self/task").unwrap() {
        let entry = entry.unwrap();
        if thread_name(&entry.path()).as_deref() == Some(name) {
            found.push(entry.file_name().to_str().unwrap().parse().unwrap());
        }
    }
    found
}

/// The names of all threads of this process that still run.
pub fn thread_names() -> Vec<String> {
    fs::read_dir("/proc/self/task")
        .unwrap()
        .filter_map(|entry| thread_name(&entry.unwrap().path()))
        .collect()
}
