//! The library pin: a pin that matches the library's SHA-256 lets open reach the library.

#[path = "fixtures/pin.rs"]
mod fixture;

use fixture::FakeInstall;

#[test]
fn a_matching_pin_lets_open_reach_the_library() {
    let install = FakeInstall::new();
    let pin = install.digest();
    // SAFETY: this binary holds one test, so no other test reads the environment.
    unsafe { install.set_env(Some(&pin)) };
    let error = match nagaya_mk_qwen38_gfx1151::open() {
        // The fake library is not an ELF file, so a load of it cannot succeed. Success is not the point here.
        Ok(_) => panic!("the fake library loaded"),
        Err(error) => error,
    };
    let message = error.message();
    assert!(
        !message.contains("MK_ENGINE_LIB_SHA256"),
        "the pin refused its own digest: {message}"
    );
    assert!(message.contains("load "), "open stopped before the library: {message}");
}
