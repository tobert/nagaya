//! The boundaries that the jobs in flight will hold (gogi "Contexts", "Requests in flight share a prefix").
//!
//! Two requests can be in flight together with contexts that begin with the same tokens, while the store holds no
//! snapshot of them. Without this book, each job looks up its boundaries before the other holds one, and each feeds
//! the shared tokens. With it, the later request waits on its call thread until the earlier job holds the boundary.
//! Then its job starts from that snapshot, and it feeds only its own tokens.

use std::collections::HashMap;
use std::sync::{Condvar, Mutex};

use nagaya_store::RecipeId;

/// The book of one engine. A recipe is in the book while a job in flight will hold a snapshot under it.
#[derive(Debug, Default)]
pub struct InFlight {
    /// Each claimed recipe, with the number of the claim.
    claimed: Mutex<Claimed>,
    /// It wakes each waiter when a recipe leaves the book.
    left: Condvar,
}

#[derive(Debug, Default)]
struct Claimed {
    recipes: HashMap<RecipeId, u64>,
    next: u64,
}

/// The recipes of one request in the book. The drop removes those that are left, so a job that fails releases its
/// waiters.
#[derive(Debug)]
pub struct Claim<'a> {
    book: &'a InFlight,
    number: u64,
}

impl InFlight {
    /// Wait until no other request in flight will hold a boundary of `recipes`. Then claim each recipe that the
    /// store does not hold, as `held` says.
    ///
    /// A request that waits holds no claim, so two requests never wait for each other.
    pub fn enter(&self, recipes: &[RecipeId], held: impl Fn(&RecipeId) -> bool) -> Claim<'_> {
        let mut claimed = self.claimed.lock().expect("the lock of the book is poisoned");
        while recipes.iter().any(|recipe| claimed.recipes.contains_key(recipe)) {
            claimed = self.left.wait(claimed).expect("the lock of the book is poisoned");
        }
        let number = claimed.next;
        claimed.next += 1;
        for recipe in recipes {
            if !held(recipe) {
                claimed.recipes.insert(*recipe, number);
            }
        }
        Claim { book: self, number }
    }

    /// A job held the boundary of `recipe`, or it found that it cannot hold it. The requests that wait for it go on.
    pub fn done(&self, recipe: &RecipeId) {
        let mut claimed = self.claimed.lock().expect("the lock of the book is poisoned");
        if claimed.recipes.remove(recipe).is_some() {
            self.left.notify_all();
        }
    }

    /// The count of the claimed recipes.
    pub fn len(&self) -> usize {
        self.claimed
            .lock()
            .expect("the lock of the book is poisoned")
            .recipes
            .len()
    }

    /// True when the book has no claimed recipe.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        let mut claimed = self.book.claimed.lock().expect("the lock of the book is poisoned");
        let before = claimed.recipes.len();
        claimed.recipes.retain(|_, number| *number != self.number);
        if claimed.recipes.len() != before {
            self.book.left.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;

    /// Recipe `n` of the test: the recipe of a boundary after `n` tokens. Its bytes do not matter.
    fn recipe(n: u8) -> RecipeId {
        let root: RecipeId = serde_json::from_value(serde_json::json!(vec![0u8; 32])).expect("a recipe id is 32 bytes");
        nagaya_store::recipe_boundary(&root, &[u32::from(n)], u64::from(n))
    }

    /// Enter on another thread, and say when the enter returns. The claim lives until `release` gets a message.
    fn enter_elsewhere(book: &'static InFlight, recipes: Vec<RecipeId>) -> (mpsc::Receiver<()>, mpsc::Sender<()>) {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        std::thread::spawn(move || {
            let _claim = book.enter(&recipes, |_| false);
            entered_tx.send(()).unwrap();
            let _ = release_rx.recv();
        });
        (entered_rx, release_tx)
    }

    fn book() -> &'static InFlight {
        Box::leak(Box::default())
    }

    const SOON: Duration = Duration::from_millis(100);

    #[test]
    fn a_request_waits_for_a_boundary_that_another_will_hold_and_goes_on_when_it_is_held() {
        let book = book();
        let first = book.enter(&[recipe(1), recipe(2)], |_| false);
        let (entered, _release) = enter_elsewhere(book, vec![recipe(1), recipe(3)]);
        assert!(
            entered.recv_timeout(SOON).is_err(),
            "the second request waits for recipe 1"
        );
        // Recipe 2 is not a recipe of the second request, so it changes nothing.
        book.done(&recipe(2));
        assert!(entered.recv_timeout(SOON).is_err());
        book.done(&recipe(1));
        entered
            .recv_timeout(Duration::from_secs(5))
            .expect("the second request goes on");
        drop(first);
    }

    #[test]
    fn the_end_of_a_request_releases_the_boundaries_that_it_did_not_hold() {
        let book = book();
        let first = book.enter(&[recipe(1)], |_| false);
        let (entered, _release) = enter_elsewhere(book, vec![recipe(1)]);
        assert!(entered.recv_timeout(SOON).is_err());
        drop(first);
        entered
            .recv_timeout(Duration::from_secs(5))
            .expect("the end of the first request wakes the second");
    }

    #[test]
    fn a_boundary_that_the_store_holds_is_not_claimed_and_requests_with_other_boundaries_do_not_wait() {
        let book = book();
        let first = book.enter(&[recipe(1), recipe(2)], |held| *held == recipe(1));
        assert_eq!(book.len(), 1);
        // Recipe 1 is held, so a request that shares only recipe 1 goes on.
        let (entered, _release) = enter_elsewhere(book, vec![recipe(1), recipe(3)]);
        entered
            .recv_timeout(Duration::from_secs(5))
            .expect("no request claims recipe 1");
        drop(first);
    }

    #[test]
    fn the_drop_of_one_claim_keeps_the_claims_of_another() {
        let book = book();
        let first = book.enter(&[recipe(1)], |_| false);
        let second = book.enter(&[recipe(2)], |_| false);
        drop(first);
        assert_eq!(book.len(), 1);
        drop(second);
        assert!(book.is_empty());
    }
}
