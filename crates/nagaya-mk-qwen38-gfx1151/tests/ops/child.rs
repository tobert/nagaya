//! Run a test in a child process. A device trap and a refused call abort the process that makes them, so a test of
//! such a call runs it in a child and reads the stderr of the child.
//!
//! A child test is an ignored test. The parent runs the test binary again with the exact name of the child test.

#![allow(dead_code)]

use std::process::{Command, Output};

/// Run the ignored test `name` (the path inside this test binary) in a child process, with the extra environment.
pub fn run(name: &str, env: &[(&str, &str)]) -> Output {
    let exe = std::env::current_exe().expect("the path of the test binary");
    let mut command = Command::new(exe);
    command.args([name, "--exact", "--ignored", "--nocapture", "--test-threads=1"]);
    command.envs(env.iter().copied());
    command
        .output()
        .unwrap_or_else(|error| panic!("run the child test {name}: {error}"))
}

/// The environment that hides every GPU from a child, so that a refusal on the host shows before the first HIP call.
pub const NO_GPU: &[(&str, &str)] = &[("HIP_VISIBLE_DEVICES", "-1"), ("ROCR_VISIBLE_DEVICES", "-1")];

pub fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

pub fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// The signal that ended the child, if any (6 is SIGABRT).
pub fn signal(output: &Output) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    output.status.signal()
}
