//! `PersistenceStore` + `MockDb` with controllable fault injection.
//!
//! [`MockDb`] is an in-memory [`crate::domain::PersistenceStore`] that doubles
//! as a **fault-injection harness**: it can be told to be slow, to be
//! unavailable, to fail the next *N* batches, or to fail every *N*-th batch.
//! That makes it the workhorse for exercising the flush worker's
//! retry/requeue/back-pressure paths in tests, the demo, and the crash harness,
//! without ever standing up a real database.
//!
//! ## Why a `Mutex` here does NOT violate main-loop isolation
//!
//! The crate's headline guarantee is that [`crate::api::FluxEngine::submit`] —
//! the hot path called from the latency-sensitive main loop — never blocks: no
//! `Mutex`, no syscall, no unbounded wait. That guarantee is about the *submit
//! path*, which only ever touches the lock-free ring.
//!
//! [`MockDb`] lives at the *opposite* end of the pipeline, on the **DB side**.
//! Its only caller is the single background flush [`crate::worker`] thread (via
//! [`PersistenceStore::persist_batch`]) plus test/observer threads flipping
//! fault knobs. The main loop never calls into the store. The contract
//! explicitly designates the worker / DB side as the place where blocking and
//! locking are allowed and expected:
//!
//! ```text
//!   submit() ─▶ Ring ─▶ Cache ─▶ WAL ─▶ Store(MockDb)   ◀── lock lives here
//!   ^^^^^^^^                                              and that is FINE:
//!   lock-free, never blocks            it is off the hot path entirely.
//! ```
//!
//! So a `Mutex` around the in-memory table is sound and idiomatic: contention on
//! it can only ever slow the *worker*, which is designed to absorb such latency
//! (it is the same place we deliberately inject `set_latency` slowness). It can
//! never propagate back to a `submit()` caller, because the ring decouples the
//! two — a slow/locked store causes the cache's dirty set to grow and, if the
//! ring fills, `submit()` sheds load (returns `Dropped`) rather than blocking.
//!
//! ## Cheap sharing & interior mutability
//!
//! Every piece of mutable state lives behind a `Mutex` (the table) or an atomic
//! (the fault knobs), so **all methods take `&self`** — there is no `&mut self`
//! anywhere. That lets a single `Arc<MockDb>` be simultaneously:
//!   * handed to the engine as an `Arc<dyn PersistenceStore>`, and
//!   * cloned and retained by a test/operator thread to flip knobs
//!     ([`MockDb::set_healthy`], [`MockDb::fail_next_batches`],
//!     [`MockDb::set_latency`], [`MockDb::set_fail_every`]) concurrently.
//!
//! This file is **safe code only**.

use crate::domain::{PersistenceStore, PlayerId, PlayerState};
use crate::error::StoreError;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// In-memory [`PersistenceStore`] with configurable fault injection.
///
/// Construct with [`MockDb::new`] (healthy, empty, no faults) and tune failure
/// behaviour with the setters. All state is behind interior mutability so a
/// single `Arc<MockDb>` can be shared across the engine and test threads; see
/// the [module docs](self) for the locking-isolation rationale.
pub struct MockDb {
    /// The simulated "table": player id → latest persisted state.
    ///
    /// Guarded by a `Mutex` because it is mutated by the worker thread on every
    /// `persist_batch` and read by observers. The lock is *intentional* and
    /// safe here — see the module-level note on why DB-side locking does not
    /// affect main-loop isolation.
    table: Mutex<HashMap<PlayerId, PlayerState>>,

    /// Fault knob: when `false`, every `persist_batch` returns a retryable
    /// [`StoreError::Unavailable`] (simulates a downed backend).
    healthy: AtomicBool,

    /// Fault knob: number of upcoming `persist_batch` calls to fail with a
    /// retryable [`StoreError::Transient`], decremented once per failed call.
    fail_next: AtomicU64,

    /// Fault knob: fail every `fail_every`-th *attempt* deterministically
    /// (`0` disables). Compared against [`Self::attempts`] via modulo.
    fail_every: AtomicU64,

    /// Fault knob: artificial latency (in nanoseconds) slept at the *start* of
    /// every `persist_batch` to simulate a slow backend and drive back-pressure.
    ///
    /// Stored as nanoseconds in an atomic so it is cheaply settable from other
    /// threads via interior mutability without locking.
    latency_nanos: AtomicU64,

    /// Monotonic count of `persist_batch` *attempts* that reached the
    /// periodic-failure / upsert decision point (i.e. survived the latency
    /// sleep, the health check and the `fail_next` countdown). Drives the
    /// deterministic `fail_every` modulo. Starts at 0 and is pre-incremented to
    /// 1 on the first such attempt, so `fail_every == 1` fails every attempt and
    /// `fail_every == N` fails attempts N, 2N, 3N, …
    attempts: AtomicU64,

    /// Monotonic count of batches that were *successfully* upserted. Exposed via
    /// [`MockDb::persisted_batches`] for test assertions.
    persisted: AtomicU64,
}

impl MockDb {
    /// Creates a healthy, empty mock store with no injected faults.
    pub fn new() -> Self {
        MockDb {
            table: Mutex::new(HashMap::new()),
            healthy: AtomicBool::new(true),
            fail_next: AtomicU64::new(0),
            fail_every: AtomicU64::new(0),
            latency_nanos: AtomicU64::new(0),
            attempts: AtomicU64::new(0),
            persisted: AtomicU64::new(0),
        }
    }

    /// Toggles global availability. When `false`, [`persist_batch`] fails with a
    /// retryable [`StoreError::Unavailable`] until set healthy again.
    ///
    /// Callable through a shared `Arc<MockDb>` from any thread (interior
    /// mutability via an atomic).
    ///
    /// [`persist_batch`]: PersistenceStore::persist_batch
    pub fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::SeqCst);
    }

    /// Arms the store to fail the next `n` [`persist_batch`] calls with a
    /// retryable [`StoreError::Transient`], then resume normally. Each failed
    /// call decrements the remaining count by one.
    ///
    /// Overwrites any previously-armed count rather than adding to it.
    ///
    /// [`persist_batch`]: PersistenceStore::persist_batch
    pub fn fail_next_batches(&self, n: u64) {
        self.fail_next.store(n, Ordering::SeqCst);
    }

    /// Injects artificial per-batch latency to simulate a slow backend and
    /// exercise the pipeline's back-pressure / load-shedding behaviour.
    ///
    /// The sleep happens at the very start of every [`persist_batch`] call,
    /// before any fault check. Pass [`Duration::ZERO`] to disable.
    ///
    /// [`persist_batch`]: PersistenceStore::persist_batch
    pub fn set_latency(&self, latency: Duration) {
        // Saturate to u64 nanoseconds; anything beyond ~584 years is clamped,
        // which is irrelevant for a test knob.
        let nanos = u64::try_from(latency.as_nanos()).unwrap_or(u64::MAX);
        self.latency_nanos.store(nanos, Ordering::SeqCst);
    }

    /// Fails every `n`-th [`persist_batch`] *attempt* deterministically with a
    /// retryable [`StoreError::Transient`] (`0` disables periodic failure).
    ///
    /// "Attempt" here means a call that has already survived the latency sleep,
    /// the health check and the `fail_next` countdown. With `n == 1` every such
    /// attempt fails; with `n == 3` attempts 3, 6, 9, … fail.
    ///
    /// [`persist_batch`]: PersistenceStore::persist_batch
    pub fn set_fail_every(&self, n: u64) {
        self.fail_every.store(n, Ordering::SeqCst);
    }

    /// Returns the number of batches that have been *successfully* upserted into
    /// the table so far. Useful for test assertions about how many flushes
    /// actually reached the store.
    pub fn persisted_batches(&self) -> u64 {
        self.persisted.load(Ordering::SeqCst)
    }

    /// Returns a point-in-time copy of every persisted [`PlayerState`] in the
    /// table, in arbitrary order. Intended for test assertions about final
    /// persisted state.
    pub fn snapshot(&self) -> Vec<PlayerState> {
        // `expect` is acceptable in this test-harness type: a poisoned lock
        // means a prior panic already corrupted invariants and there is nothing
        // sensible to recover to.
        let table = self.table.lock().expect("MockDb table mutex poisoned");
        table.values().cloned().collect()
    }
}

impl Default for MockDb {
    fn default() -> Self {
        MockDb::new()
    }
}

impl PersistenceStore for MockDb {
    /// Persists a batch, honouring the fault knobs **in this exact order**:
    ///
    /// 1. **Latency** — sleep the configured [`set_latency`](MockDb::set_latency)
    ///    duration (simulate a slow DB). Always happens first, even for calls
    ///    that will subsequently fail, so latency models a slow *round trip*.
    /// 2. **Health** — if not healthy, return [`StoreError::Unavailable`].
    /// 3. **`fail_next`** — if armed (`> 0`), decrement and return a retryable
    ///    [`StoreError::Transient("injected")`](StoreError::Transient).
    /// 4. **`fail_every`** — increment the attempt counter; if `fail_every > 0`
    ///    and `attempt % fail_every == 0`, return
    ///    [`StoreError::Transient("periodic")`](StoreError::Transient).
    /// 5. **Upsert** — otherwise apply every state under **last-write-wins by
    ///    version**: an incoming state overwrites the existing one only when
    ///    `incoming.version >= existing.version` (so replays / equal-version
    ///    re-writes are idempotent and never regress a newer value), then
    ///    increment the persisted-batch counter and return `Ok(())`.
    fn persist_batch(&self, states: &[PlayerState]) -> Result<(), StoreError> {
        // (1) Optional configured latency: simulate a slow DB round-trip. We do
        //     this BEFORE any fault check so even an injected failure pays the
        //     modelled cost, matching a real backend that is slow *and* failing.
        let latency = self.latency_nanos.load(Ordering::SeqCst);
        if latency > 0 {
            std::thread::sleep(Duration::from_nanos(latency));
        }

        // (2) Health gate: a downed backend is retryably Unavailable.
        if !self.healthy.load(Ordering::SeqCst) {
            return Err(StoreError::Unavailable);
        }

        // (3) Countdown failures: fail exactly the next N calls. Decrement only
        //     when we actually consume one, and only fail when the prior value
        //     was non-zero (the `fetch_update` returns the *previous* value).
        let consumed = self
            .fail_next
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |cur| {
                if cur > 0 {
                    Some(cur - 1)
                } else {
                    // Leave at 0; signals "nothing to consume".
                    None
                }
            });
        if consumed.is_ok() {
            // `Ok` means we successfully decremented a previously-positive
            // counter, i.e. this call is an injected transient failure.
            return Err(StoreError::Transient("injected".to_string()));
        }

        // (4) Periodic failures: count this attempt, then fail on multiples of
        //     `fail_every`. Pre-increment so the first attempt is #1; with
        //     `fail_every == N` we fail attempts N, 2N, 3N, …
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
        let fail_every = self.fail_every.load(Ordering::SeqCst);
        if fail_every > 0 && attempt.is_multiple_of(fail_every) {
            return Err(StoreError::Transient("periodic".to_string()));
        }

        // (5) Success path: upsert under last-write-wins by version.
        {
            let mut table = self.table.lock().expect("MockDb table mutex poisoned");
            for incoming in states {
                match table.get(&incoming.player_id) {
                    // Only overwrite when the incoming version is at least as new
                    // as what we already hold. `>=` (not `>`) makes equal-version
                    // re-persists / WAL replays idempotent and lossless.
                    Some(existing) if incoming.version < existing.version => {
                        // Stale write — keep the newer persisted value.
                    }
                    _ => {
                        table.insert(incoming.player_id, incoming.clone());
                    }
                }
            }
        }

        // Count a successfully-persisted batch (incl. an empty batch, which is a
        // valid no-op flush). Done after the upsert so observers that see the
        // bumped counter also see the data.
        self.persisted.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    /// Loads the current persisted state for `id`, if any. Reads the table
    /// under the lock.
    fn load(&self, id: PlayerId) -> Option<PlayerState> {
        let table = self.table.lock().expect("MockDb table mutex poisoned");
        table.get(&id).cloned()
    }

    /// Number of distinct entities currently persisted. Reads the table under
    /// the lock.
    fn count(&self) -> usize {
        let table = self.table.lock().expect("MockDb table mutex poisoned");
        table.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Instant;

    /// Builds a `PlayerState` tersely for tests.
    fn st(player_id: PlayerId, version: u64, blob: &[u8]) -> PlayerState {
        PlayerState {
            player_id,
            version,
            blob: blob.to_vec(),
        }
    }

    #[test]
    fn empty_store_starts_clean() {
        let db = MockDb::new();
        assert_eq!(db.count(), 0);
        assert_eq!(db.persisted_batches(), 0);
        assert!(db.load(1).is_none());
        assert!(db.snapshot().is_empty());
    }

    #[test]
    fn happy_path_upserts_and_counts() {
        let db = MockDb::new();
        db.persist_batch(&[st(1, 1, b"a"), st(2, 1, b"b")]).unwrap();

        assert_eq!(db.count(), 2);
        assert_eq!(db.persisted_batches(), 1);
        assert_eq!(db.load(1), Some(st(1, 1, b"a")));
        assert_eq!(db.load(2), Some(st(2, 1, b"b")));

        let mut snap = db.snapshot();
        snap.sort_by_key(|s| s.player_id);
        assert_eq!(snap, vec![st(1, 1, b"a"), st(2, 1, b"b")]);
    }

    #[test]
    fn empty_batch_is_a_valid_noop_flush() {
        let db = MockDb::new();
        db.persist_batch(&[]).unwrap();
        // An empty flush still counts as a persisted batch but adds no rows.
        assert_eq!(db.persisted_batches(), 1);
        assert_eq!(db.count(), 0);
    }

    // ---- Last-write-wins by version --------------------------------------

    #[test]
    fn last_write_wins_newer_version_overwrites() {
        let db = MockDb::new();
        db.persist_batch(&[st(1, 1, b"old")]).unwrap();
        db.persist_batch(&[st(1, 5, b"new")]).unwrap();
        assert_eq!(db.load(1), Some(st(1, 5, b"new")));
        assert_eq!(db.count(), 1);
    }

    #[test]
    fn last_write_wins_stale_version_ignored() {
        let db = MockDb::new();
        db.persist_batch(&[st(1, 5, b"new")]).unwrap();
        // A lower version must NOT clobber the newer persisted value.
        db.persist_batch(&[st(1, 2, b"stale")]).unwrap();
        assert_eq!(db.load(1), Some(st(1, 5, b"new")));
    }

    #[test]
    fn last_write_wins_equal_version_overwrites_idempotently() {
        let db = MockDb::new();
        db.persist_batch(&[st(1, 3, b"first")]).unwrap();
        // `>=` semantics: an equal-version re-persist (e.g. a WAL replay) is
        // applied and is lossless / idempotent on the value.
        db.persist_batch(&[st(1, 3, b"second")]).unwrap();
        assert_eq!(db.load(1), Some(st(1, 3, b"second")));
    }

    #[test]
    fn last_write_wins_within_single_batch() {
        // Ordering within one batch is honoured too: later entries with a
        // newer-or-equal version win over earlier ones for the same id.
        let db = MockDb::new();
        db.persist_batch(&[st(1, 1, b"a"), st(1, 4, b"b"), st(1, 2, b"c")])
            .unwrap();
        // After v1 → v4 → (v2 stale, ignored), the persisted value is v4.
        assert_eq!(db.load(1), Some(st(1, 4, b"b")));
        assert_eq!(db.count(), 1);
    }

    // ---- Fault mode: unhealthy -------------------------------------------

    #[test]
    fn unhealthy_returns_retryable_unavailable() {
        let db = MockDb::new();
        db.set_healthy(false);

        let err = db.persist_batch(&[st(1, 1, b"x")]).unwrap_err();
        assert!(matches!(err, StoreError::Unavailable));
        assert!(err.is_retryable());
        // Nothing persisted while unhealthy.
        assert_eq!(db.count(), 0);
        assert_eq!(db.persisted_batches(), 0);

        // Recovery: once healthy again the write goes through.
        db.set_healthy(true);
        db.persist_batch(&[st(1, 1, b"x")]).unwrap();
        assert_eq!(db.load(1), Some(st(1, 1, b"x")));
        assert_eq!(db.persisted_batches(), 1);
    }

    // ---- Fault mode: fail_next_batches -----------------------------------

    #[test]
    fn fail_next_batches_fails_then_recovers() {
        let db = MockDb::new();
        db.fail_next_batches(2);

        for _ in 0..2 {
            let err = db.persist_batch(&[st(1, 1, b"x")]).unwrap_err();
            match err {
                StoreError::Transient(ref m) => assert_eq!(m, "injected"),
                other => panic!("expected Transient(\"injected\"), got {other:?}"),
            }
            assert!(err.is_retryable());
        }
        // Still nothing persisted: both attempts were injected failures.
        assert_eq!(db.persisted_batches(), 0);
        assert_eq!(db.count(), 0);

        // Third attempt succeeds.
        db.persist_batch(&[st(1, 1, b"x")]).unwrap();
        assert_eq!(db.persisted_batches(), 1);
        assert_eq!(db.load(1), Some(st(1, 1, b"x")));
    }

    #[test]
    fn fail_next_batches_overwrites_not_accumulates() {
        let db = MockDb::new();
        db.fail_next_batches(5);
        db.fail_next_batches(1); // overwrite, not add → only 1 failure left
        assert!(db.persist_batch(&[st(1, 1, b"x")]).is_err());
        assert!(db.persist_batch(&[st(1, 1, b"x")]).is_ok());
    }

    // ---- Fault mode: fail_every ------------------------------------------

    #[test]
    fn fail_every_third_attempt_is_periodic() {
        let db = MockDb::new();
        db.set_fail_every(3);

        // Attempts 1, 2 succeed; attempt 3 fails periodically; 4, 5 succeed;
        // attempt 6 fails again.
        let mut outcomes = Vec::new();
        for _ in 0..6 {
            outcomes.push(db.persist_batch(&[st(1, 1, b"x")]).is_ok());
        }
        assert_eq!(outcomes, vec![true, true, false, true, true, false]);

        // Four successful upserts of the same id → count 1, persisted 4.
        assert_eq!(db.persisted_batches(), 4);
        assert_eq!(db.count(), 1);

        // The two failures were the retryable "periodic" transient.
        db.set_fail_every(1); // fail every attempt now
        let err = db.persist_batch(&[st(2, 1, b"y")]).unwrap_err();
        match err {
            StoreError::Transient(ref m) => assert_eq!(m, "periodic"),
            other => panic!("expected Transient(\"periodic\"), got {other:?}"),
        }
        assert!(err.is_retryable());
    }

    #[test]
    fn fail_every_zero_disables_periodic_failure() {
        let db = MockDb::new();
        db.set_fail_every(0);
        for _ in 0..10 {
            db.persist_batch(&[st(1, 1, b"x")]).unwrap();
        }
        assert_eq!(db.persisted_batches(), 10);
    }

    // ---- Fault knob ordering ---------------------------------------------

    #[test]
    fn health_check_precedes_fail_next() {
        // Unhealthy must short-circuit BEFORE the fail_next countdown is
        // consumed, so the armed countdown survives the unhealthy window.
        let db = MockDb::new();
        db.fail_next_batches(1);
        db.set_healthy(false);

        assert!(matches!(
            db.persist_batch(&[st(1, 1, b"x")]).unwrap_err(),
            StoreError::Unavailable
        ));

        db.set_healthy(true);
        // The fail_next countdown was NOT consumed while unhealthy.
        match db.persist_batch(&[st(1, 1, b"x")]).unwrap_err() {
            StoreError::Transient(m) => assert_eq!(m, "injected"),
            other => panic!("expected injected transient, got {other:?}"),
        }
        // Now it is clear.
        assert!(db.persist_batch(&[st(1, 1, b"x")]).is_ok());
    }

    // ---- Fault mode: latency ---------------------------------------------

    #[test]
    fn latency_sleeps_before_returning() {
        let db = MockDb::new();
        db.set_latency(Duration::from_millis(20));

        let start = Instant::now();
        db.persist_batch(&[st(1, 1, b"x")]).unwrap();
        let elapsed = start.elapsed();

        // Allow generous slack for scheduler jitter but prove the sleep happened.
        assert!(
            elapsed >= Duration::from_millis(15),
            "expected the injected latency to delay the call, got {elapsed:?}"
        );
        // The write still landed despite the slowness.
        assert_eq!(db.load(1), Some(st(1, 1, b"x")));
    }

    #[test]
    fn latency_applies_even_to_failing_calls() {
        // Latency is the first knob: even an unhealthy (failing) call pays it.
        let db = MockDb::new();
        db.set_latency(Duration::from_millis(15));
        db.set_healthy(false);

        let start = Instant::now();
        assert!(db.persist_batch(&[st(1, 1, b"x")]).is_err());
        assert!(start.elapsed() >= Duration::from_millis(10));
    }

    // ---- Shareability: Arc<MockDb> as Arc<dyn PersistenceStore> + knobs ---

    #[test]
    fn arc_shared_as_trait_object_and_knob_handle() {
        // One Arc handed to the "engine" as a trait object; a clone retained to
        // flip knobs from another thread — proves all state is &self / interior
        // mutability and the type is cheaply shareable.
        let db = Arc::new(MockDb::new());
        let knobs = Arc::clone(&db);
        let store: Arc<dyn PersistenceStore> = db.clone();

        // Engine side persists.
        store.persist_batch(&[st(1, 1, b"a")]).unwrap();
        assert_eq!(store.count(), 1);

        // Operator side injects a fault from a separate thread.
        let handle = std::thread::spawn(move || {
            knobs.set_healthy(false);
        });
        handle.join().unwrap();

        assert!(matches!(
            store.persist_batch(&[st(2, 1, b"b")]).unwrap_err(),
            StoreError::Unavailable
        ));
        // Read knob counter through the original handle.
        assert_eq!(db.persisted_batches(), 1);
    }

    #[test]
    fn concurrent_persists_are_lossless() {
        // Hammer the store from many threads to prove the Mutex keeps the table
        // consistent (DB-side locking is fine and correct).
        let db = Arc::new(MockDb::new());
        let mut handles = Vec::new();
        for t in 0..8u64 {
            let db = Arc::clone(&db);
            handles.push(std::thread::spawn(move || {
                for i in 0..100u64 {
                    let id = t * 100 + i;
                    db.persist_batch(&[st(id, 1, b"v")]).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(db.count(), 800);
        assert_eq!(db.persisted_batches(), 800);
    }
}
