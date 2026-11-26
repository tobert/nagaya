//! The worker threads of the kernels (spec 3.2c): `threads` persistent workers `nagaya-cpu-<index>` for each thread
//! that calls the kernels. A thread starts its workers at `start` or at its first call of a kernel with `threads`
//! above 1, and keeps them until it ends or calls with another count. A new thread gets the CPU mask of the thread
//! that starts it, so the workers keep the mask of the caller (the group thread in the server).
//!
//! The caller gives a job, `f(index)` for each worker index, and waits until each worker ends it. A worker spins for
//! a short time after a job, so the next task of the plan starts with no wake from the kernel, and then sleeps.

use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// The count of spin loops of a waiting thread before it sleeps: some tens of microseconds.
const SPINS: u32 = 1 << 14;

/// A job with its lifetime erased. `Pool::run` waits until each worker ends the job, so the job outlives each use.
#[derive(Clone, Copy)]
struct Job(*const (dyn Fn(usize) + Sync));

// SAFETY: the job is `Sync`, and `Pool::run` keeps it alive while a worker can read it.
unsafe impl Send for Job {}

struct Shared {
    /// The job of the current generation, and `exit`.
    state: Mutex<(Option<Job>, bool)>,
    wake: Condvar,
    done: Condvar,
    generation: AtomicU64,
    /// The workers that have not ended the job of the current generation.
    remaining: AtomicUsize,
    /// A worker panicked in the job of the current generation.
    panicked: AtomicBool,
}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    fn new(threads: usize) -> Pool {
        let shared = Arc::new(Shared {
            state: Mutex::new((None, false)),
            wake: Condvar::new(),
            done: Condvar::new(),
            generation: AtomicU64::new(0),
            remaining: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
        });
        let workers = (0..threads)
            .map(|i| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("nagaya-cpu-{i}"))
                    .spawn(move || work(&shared, i))
                    .unwrap_or_else(|e| panic!("cannot start the worker thread nagaya-cpu-{i}: {e}"))
            })
            .collect();
        Pool { shared, workers }
    }

    /// Runs `f(i)` on worker `i` for each worker, and returns when each has ended. A panic of a worker panics here.
    fn run(&self, f: &(dyn Fn(usize) + Sync)) {
        let s = &*self.shared;
        // SAFETY: only the lifetime changes. This function returns only after each worker ended the job (it waits
        // for `remaining` to be 0), and no worker reads the job after it ends it.
        let job = Job(unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f) });
        {
            let mut st = s.state.lock().unwrap();
            st.0 = Some(job);
            s.panicked.store(false, Ordering::Relaxed);
            s.remaining.store(self.workers.len(), Ordering::Relaxed);
            s.generation.fetch_add(1, Ordering::Release);
            s.wake.notify_all();
        }
        let mut spins = 0;
        while s.remaining.load(Ordering::Acquire) != 0 {
            if spins < SPINS {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            let mut st = s.state.lock().unwrap();
            while s.remaining.load(Ordering::Acquire) != 0 {
                st = s.done.wait(st).unwrap();
            }
        }
        s.state.lock().unwrap().0 = None;
        if s.panicked.load(Ordering::Acquire) {
            panic!("a kernel worker thread panicked; its message is above");
        }
    }
}

fn work(s: &Shared, index: usize) {
    let mut seen = 0u64;
    loop {
        let mut spins = 0;
        while s.generation.load(Ordering::Acquire) == seen && spins < SPINS {
            spins += 1;
            std::hint::spin_loop();
        }
        let job = {
            let mut st = s.state.lock().unwrap();
            while s.generation.load(Ordering::Acquire) == seen && !st.1 {
                st = s.wake.wait(st).unwrap();
            }
            if st.1 {
                return;
            }
            seen = s.generation.load(Ordering::Acquire);
            st.0.expect("a new generation has a job")
        };
        // SAFETY: the job lives until `remaining` is 0 (Pool::run), and this worker has not yet ended it.
        let f = unsafe { &*job.0 };
        if catch_unwind(AssertUnwindSafe(|| f(index))).is_err() {
            s.panicked.store(true, Ordering::Release);
        }
        if s.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _st = s.state.lock().unwrap();
            s.done.notify_all();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().1 = true;
        self.shared.wake.notify_all();
        for w in self.workers.drain(..) {
            // A worker catches the panics of its jobs, so it ends with no panic.
            let _ = w.join();
        }
    }
}

thread_local! {
    static POOL: RefCell<Option<Pool>> = const { RefCell::new(None) };
}

/// Runs `f(i)` for each `i` in `0..threads` on the workers of this thread, and returns when each has ended.
pub fn run(threads: usize, f: &(dyn Fn(usize) + Sync)) {
    with(threads, |p| p.run(f));
}

/// Starts the `threads` workers of this thread now, so that the first job does not wait for them.
pub fn start(threads: usize) {
    with(threads, |_| {});
}

fn with(threads: usize, f: impl FnOnce(&Pool)) {
    POOL.with(|p| {
        let mut p = p
            .try_borrow_mut()
            .expect("a kernel job started a kernel: the workers of a thread run one job at a time");
        if p.as_ref().is_none_or(|p| p.workers.len() != threads) {
            *p = None;
            *p = Some(Pool::new(threads));
        }
        f(p.as_ref().unwrap());
    });
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    #[test]
    fn each_worker_runs_once_by_its_name_and_the_workers_stay() {
        let names = Mutex::new(Vec::new());
        let ids = Mutex::new(Vec::new());
        for _ in 0..3 {
            super::run(4, &|i| {
                let t = std::thread::current();
                names.lock().unwrap().push((i, t.name().unwrap_or("").to_owned()));
                ids.lock().unwrap().push(t.id());
            });
        }
        let mut names = names.into_inner().unwrap();
        names.sort();
        names.dedup();
        let want: Vec<(usize, String)> = (0..4).map(|i| (i, format!("nagaya-cpu-{i}"))).collect();
        assert_eq!(names, want);
        let mut ids = ids.into_inner().unwrap();
        assert_eq!(ids.len(), 12, "each worker ran each job once");
        ids.sort_by_key(|id| format!("{id:?}"));
        ids.dedup();
        assert_eq!(ids.len(), 4, "the three jobs ran on the same four threads");
    }

    #[test]
    fn a_panic_of_a_worker_panics_the_caller_and_the_workers_go_on() {
        let r = std::panic::catch_unwind(|| super::run(3, &|i| assert_ne!(i, 2, "worker 2 fails")));
        assert!(r.is_err(), "the panic of worker 2 reached the caller");
        let count = std::sync::atomic::AtomicUsize::new(0);
        super::run(3, &|_| {
            count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        assert_eq!(count.into_inner(), 3);
    }
}
