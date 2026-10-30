//! The thread names that the CPU tests read from `/proc/self/task`.

mod task_support;

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use task_support::thread_name;

#[test]
fn a_thread_that_ended_has_no_name() {
    let tid = std::thread::spawn(|| {
        // SAFETY: `gettid` has no inputs and cannot fail.
        unsafe { libc::gettid() }
    })
    .join()
    .unwrap();
    let task = format!("/proc/self/task/{tid}");
    let start = Instant::now();
    while Path::new(&task).exists() {
        assert!(start.elapsed() < Duration::from_secs(5), "the thread {tid} did not end");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(thread_name(Path::new(&task)), None);
}

#[test]
fn a_running_thread_has_its_name() {
    // SAFETY: `gettid` has no inputs and cannot fail.
    let tid = unsafe { libc::gettid() };
    let task = format!("/proc/self/task/{tid}");
    let expected = fs::read_to_string(format!("{task}/comm")).unwrap();
    assert_eq!(thread_name(Path::new(&task)).as_deref(), Some(expected.trim()));
}

#[test]
#[should_panic(expected = "cannot read the name of the thread")]
fn another_read_error_fails_the_test() {
    // A `comm` that is a directory gives `EISDIR`, which does not mean that the thread ended.
    let directory = tempfile::tempdir().unwrap();
    fs::create_dir(directory.path().join("comm")).unwrap();
    thread_name(directory.path());
}
