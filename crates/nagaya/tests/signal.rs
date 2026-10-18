use nagaya::signal::{SIGTERM, SignalFlag, install_terminate};

unsafe extern "C" {
    fn raise(signal: i32) -> i32;
}

#[test]
fn the_flag_counts_every_signal() {
    let flag = SignalFlag::new();
    assert_eq!(flag.count(), 0);
    flag.raise();
    flag.raise();
    assert_eq!(flag.count(), 2);
}

#[test]
fn the_handler_raises_the_flag() {
    static FLAG: SignalFlag = SignalFlag::new();
    install_terminate(&FLAG).unwrap();
    // SAFETY: the handler for SIGTERM is installed, and it only increments the flag.
    assert_eq!(unsafe { raise(SIGTERM) }, 0);
    assert_eq!(FLAG.count(), 1);
}
