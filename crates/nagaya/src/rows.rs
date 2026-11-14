//! The row threads of spec 3.2: the host work of the rows of one pass, at the same time.
//!
//! After a shared pass, each row has host work that uses no engine: the read of its logit row and the draw of its
//! token. The thread that ran the pass gives that work to its row threads and waits for them. Then the card does not
//! wait for one row after the other.

use std::cell::OnceCell;

use rayon::prelude::*;

/// The count of row threads of one calling thread. It is the largest count of segments that an engine of the lab
/// joins in a pass. A pass with more rows gives its rows to the threads in turn.
pub const ROW_THREADS: usize = 8;

thread_local! {
    /// The row threads of this thread. They start at the first shared pass, and they end with this thread.
    static THREADS: OnceCell<rayon::ThreadPool> = const { OnceCell::new() };
}

/// Start the row threads of the calling thread. Linux gives each one the CPU mask of the thread that starts it, so
/// the row threads of a device thread stay on the CPUs of its card (spec 3.2b).
fn start() -> rayon::ThreadPool {
    let owner = std::thread::current().name().unwrap_or("nagaya").to_owned();
    rayon::ThreadPoolBuilder::new()
        .num_threads(ROW_THREADS)
        .thread_name(|index| format!("nagaya-row-{index}"))
        .build()
        .unwrap_or_else(|error| panic!("cannot start the row threads of the thread `{owner}`: {error}"))
}

/// Run `work` on each item, each on a row thread at the same time, and return when all ended. One item runs on the
/// calling thread, because a thread hand-off costs more than it gains. A panic of `work` comes back to the caller.
pub fn each<T: Send>(items: &mut [T], work: impl Fn(&mut T) + Sync) {
    if items.len() < 2 {
        items.iter_mut().for_each(work);
        return;
    }
    THREADS.with(|threads| {
        threads
            .get_or_init(start)
            .install(|| items.par_iter_mut().with_max_len(1).for_each(&work));
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::sync::Mutex;
    use std::thread::ThreadId;

    /// Falsified by an `each` that runs the items in turn on the calling thread: the set holds the caller.
    #[test]
    fn several_items_run_on_row_threads_and_one_item_runs_on_the_caller() {
        let caller = std::thread::current().id();
        let ran: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
        let ids: Mutex<Vec<ThreadId>> = Mutex::new(Vec::new());
        let mut items = [1u32, 2, 3, 4];
        each(&mut items, |item| {
            *item *= 10;
            ids.lock().unwrap().push(std::thread::current().id());
            ran.lock()
                .unwrap()
                .insert(std::thread::current().name().unwrap_or("").to_owned());
        });
        assert_eq!(items, [10, 20, 30, 40]);
        assert!(!ids.lock().unwrap().contains(&caller), "an item ran on the caller");
        assert!(ran.lock().unwrap().iter().all(|name| name.starts_with("nagaya-row-")));

        let mut one = [5u32];
        let on = Mutex::new(None);
        each(&mut one, |item| {
            *item += 1;
            *on.lock().unwrap() = Some(std::thread::current().id());
        });
        assert_eq!((one, on.into_inner().unwrap()), ([6], Some(caller)));
    }
}
