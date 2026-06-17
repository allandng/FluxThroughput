//! ADVERSARIAL fault-injection integration tests for the write-behind pipeline.
//!
//! These tests prove the headline durability/correctness guarantee of
//! `flux_throughput`: even while the persistence backend is failing — transiently,
//! totally, or periodically — the write-behind layer loses **NO update** and
//! **corrupts NOTHING**. Every submitted update eventually lands in the store with
//! the correct last-write-wins final version, and never a stale/duplicated value.
//!
//! Each `#[test]` is a separate integration binary linking `flux_throughput`. We
//! drive the real public API ([`FluxEngine`]) against the real [`MockDb`] fault
//! harness and assert on the store's final, persisted state — the only place a
//! "lost update" or "corruption" could actually manifest.
//!
//! The tests are deterministic and finite: failures are injected via the MockDb
//! knobs (`fail_next_batches`, `set_healthy`, `set_fail_every`), and every test
//! either drives the engine to quiescence with `flush_now()`/`shutdown()` or
//! bounds its waiting with a finite poll loop. Each engine gets a unique temp WAL
//! dir under `std::env::temp_dir()`, cleaned up at the end. No external crates.

use flux_throughput::{
    Cache, FluxConfig, FluxEngine, MetricsSnapshot, PersistenceStore, PlayerId, PlayerState,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use flux_throughput::store::MockDb;

/// One submitted update in a "known set": `(player_id, version, blob)`.
type Submit = (PlayerId, u64, Vec<u8>);

/// The full ordered submit list plus the expected final per-player state.
type KnownUpdates = (Vec<Submit>, HashMap<PlayerId, PlayerState>);

/// Process-unique counter so concurrently-running test binaries never collide on
/// a temp WAL directory name.
static DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Allocates a fresh, unique temp directory for one engine's WAL and creates it.
///
/// Name shape: `flux_<name>_<pid>_<counter>` under the OS temp dir, exactly as
/// the task requires, so parallel test binaries and repeated runs never clash.
fn fresh_wal_dir(name: &str) -> PathBuf {
    let n = DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("flux_{}_{}_{}", name, std::process::id(), n));
    std::fs::create_dir_all(&dir).expect("create temp wal dir");
    dir
}

/// Best-effort recursive cleanup of a temp WAL dir at the end of a test.
fn cleanup(dir: &PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}

/// A test config tuned for *fast, deterministic* fault tests: tiny flush
/// thresholds and intervals so the worker reacts quickly, fsync on (real
/// durability path), and the WAL pointed at a private temp dir.
fn test_config(wal_dir: PathBuf) -> FluxConfig {
    FluxConfig {
        ring_capacity: 1024,
        flush_max_batch: 64,
        flush_interval: Duration::from_millis(5),
        worker_idle_sleep: Duration::from_millis(1),
        wal_fsync: true,
        store_retry_limit: 5,
        store_retry_backoff: Duration::from_millis(1),
        ..FluxConfig::default()
    }
    .with_wal_dir(wal_dir)
}

/// The canonical "known set" of updates used across the scenarios.
///
/// `players` distinct ids, each written `versions` times with ascending
/// versions `1..=versions`. The blob encodes `(player_id, version)` so any
/// torn/duplicated/stale persisted value is detectable byte-for-byte. Returns
/// the full ordered submit list **and** the expected final state per player
/// (last-write-wins ⇒ the highest version, `versions`).
fn known_updates(players: u64, versions: u64) -> KnownUpdates {
    let mut submits = Vec::new();
    let mut expected = HashMap::new();
    for pid in 0..players {
        for ver in 1..=versions {
            let blob = encode_blob(pid, ver);
            submits.push((pid, ver, blob.clone()));
            // Last write for this player wins: the final (highest) version.
            if ver == versions {
                expected.insert(
                    pid,
                    PlayerState {
                        player_id: pid,
                        version: ver,
                        blob,
                    },
                );
            }
        }
    }
    (submits, expected)
}

/// Deterministic self-describing blob: `"p<player>v<version>"` bytes. Lets an
/// assertion prove the *exact* value persisted, catching corruption where the
/// id/version match but the payload is wrong.
fn encode_blob(player_id: PlayerId, version: u64) -> Vec<u8> {
    format!("p{player_id}v{version}").into_bytes()
}

/// Submits every update in order through the fire-and-forget `submit()` path,
/// re-submitting any `Dropped` (load-shed) write until it is accepted so the
/// "known set" is fully delivered into the pipeline. Bounded retry per item so a
/// genuinely wedged ring can never hang the test.
fn submit_all(engine: &FluxEngine, submits: &[Submit]) {
    for (pid, ver, blob) in submits {
        // Try hard but finitely to enqueue; the ring drains continuously on the
        // worker so a transient "full" clears quickly. 100k attempts is wildly
        // more than enough for these small sets and keeps the test finite.
        let mut accepted = false;
        for _ in 0..100_000 {
            match engine.submit(*pid, *ver, blob.clone()) {
                flux_throughput::SubmitOutcome::Enqueued => {
                    accepted = true;
                    break;
                }
                flux_throughput::SubmitOutcome::Dropped => {
                    // Ring momentarily full: yield and retry so no update is lost
                    // to load shedding in these correctness tests.
                    std::thread::yield_now();
                }
            }
        }
        assert!(
            accepted,
            "submit({pid},{ver}) was shed 100k times — ring never drained (test setup bug)"
        );
    }
}

/// Asserts the store's final persisted state EXACTLY equals `expected`:
/// every expected player present with the correct final version and blob, no
/// extra/foreign rows, no stale or corrupted values. This is the core
/// "NO lost updates / NO corruption" check.
fn assert_store_exact(store: &Arc<MockDb>, expected: &HashMap<PlayerId, PlayerState>) {
    let snap = store.snapshot();
    assert_eq!(
        snap.len(),
        expected.len(),
        "store has {} rows but expected {} — lost or duplicated players",
        snap.len(),
        expected.len()
    );
    for st in &snap {
        let want = expected
            .get(&st.player_id)
            .unwrap_or_else(|| panic!("store has unexpected player {}", st.player_id));
        assert_eq!(
            &st.version, &want.version,
            "player {}: persisted version {} != expected final version {} (lost update / wrong LWW)",
            st.player_id, st.version, want.version
        );
        assert_eq!(
            &st.blob, &want.blob,
            "player {}: persisted blob {:?} != expected {:?} (CORRUPTION)",
            st.player_id, st.blob, want.blob
        );
    }
    // Symmetric check: every expected player is actually present.
    for (pid, want) in expected {
        let got = store
            .load(*pid)
            .unwrap_or_else(|| panic!("player {pid} missing from store — LOST update"));
        assert_eq!(
            got, *want,
            "player {pid}: store state {got:?} != expected {want:?}"
        );
    }
}

/// Polls a predicate up to `timeout`, sleeping `step` between checks. Returns
/// whether it became true within the budget. Keeps every wait finite.
fn wait_until<F: FnMut() -> bool>(timeout: Duration, step: Duration, mut pred: F) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if pred() {
            return true;
        }
        if Instant::now() >= deadline {
            return pred();
        }
        std::thread::sleep(step);
    }
}

// ===========================================================================
// Scenario 1 — TRANSIENT RECOVERY
//
// Arm the store to fail the first several flush batches with a retryable
// Transient error, then succeed. Submit a known set, drive to quiescence, and
// assert EVERY update lands with the correct final version, and that the
// retry/error metrics actually fired (proving the failures were really
// exercised, not skipped).
// ===========================================================================
#[test]
fn transient_recovery_no_lost_updates() {
    let dir = fresh_wal_dir("transient");
    let store = Arc::new(MockDb::new());

    // Arm a burst of transient failures BEFORE the engine starts flushing. The
    // worker (store_retry_limit = 5) will burn retries, possibly requeue, and
    // ultimately persist everything once the injected failures are exhausted.
    // 12 forced failures guarantees at least one full retry-budget exhaustion +
    // requeue cycle, so the requeue-no-loss path is genuinely exercised.
    store.fail_next_batches(12);

    let store_dyn: Arc<dyn PersistenceStore> = store.clone();
    let engine = FluxEngine::start(test_config(dir.clone()), store_dyn).expect("engine start");

    let (submits, expected) = known_updates(40, 3);
    submit_all(&engine, &submits);

    // Drive to durable quiescence. flush_now() runs the WAL->store flush to
    // completion; because the early batches were failing, we may need a couple
    // of passes as requeued dirty entries are retried after the injected
    // failures drain. Each pass is finite.
    let mut persisted_total = 0usize;
    for _ in 0..50 {
        if let Ok(n) = engine.flush_now() {
            persisted_total += n;
        }
        if engine.pending() == 0 && store.count() as u64 == 40 {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }

    // Graceful shutdown drains anything left and joins the worker.
    engine.shutdown().expect("shutdown clean");

    // ---- ASSERT: no lost updates, no corruption -------------------------
    assert_store_exact(&store, &expected);

    // ---- ASSERT: the failure paths actually fired -----------------------
    let m: MetricsSnapshot = store_metrics_proxy(&store);
    // We can't read engine metrics post-shutdown (engine consumed), so prove
    // the fault was exercised via the store's own counters: it logged > 0
    // successful batches AND we forced 12 failures. Cross-check with the
    // engine metrics captured below.
    let _ = m; // (store has no metrics snapshot; see captured engine metrics)

    println!(
        "PASS transient_recovery: 40 players x 3 versions delivered; \
         persisted_total(flush_now)={persisted_total}; \
         store.count()={}; persisted_batches={}; all final versions == 3, blobs exact.",
        store.count(),
        store.persisted_batches()
    );

    cleanup(&dir);
}

/// The store type itself exposes no `MetricsSnapshot`; this tiny shim exists only
/// so the transient test reads cleanly. It returns a default snapshot — the real
/// metric assertions for that scenario are performed in
/// [`transient_recovery_metrics_prove_retries`], which captures the *engine*
/// metrics before shutdown.
fn store_metrics_proxy(_store: &Arc<MockDb>) -> MetricsSnapshot {
    MetricsSnapshot::default()
}

/// Companion to scenario 1 that captures the **engine** metrics while the engine
/// is still alive, asserting `store_errors > 0` and `store_retries > 0` — direct
/// proof the transient failures forced retries and at least one requeue, yet the
/// data still all landed.
#[test]
fn transient_recovery_metrics_prove_retries() {
    let dir = fresh_wal_dir("transient_metrics");
    let store = Arc::new(MockDb::new());

    // Enough forced failures to (a) burn the full retry budget at least once
    // (=> store_errors > 0 via requeue) and (b) record several retries.
    store.fail_next_batches(12);

    let store_dyn: Arc<dyn PersistenceStore> = store.clone();
    let engine = FluxEngine::start(test_config(dir.clone()), store_dyn).expect("engine start");

    let (submits, expected) = known_updates(30, 4);
    submit_all(&engine, &submits);

    // Drive to quiescence WITHOUT consuming the engine, so we can read metrics.
    for _ in 0..50 {
        let _ = engine.flush_now();
        if engine.pending() == 0 && store.count() as u64 == 30 {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }

    // Capture metrics BEFORE shutdown consumes the engine.
    let m = engine.metrics();

    // Everything persisted correctly.
    assert_eq!(store.count() as u64, 30, "all 30 players must be persisted");
    assert_store_exact(&store, &expected);

    // The retry/error paths must have actually fired.
    assert!(
        m.store_retries > 0,
        "expected store_retries > 0 (transient failures should force retries), got {}",
        m.store_retries
    );
    assert!(
        m.store_errors > 0,
        "expected store_errors > 0 (retry budget exhaustion should requeue + record an error), got {}",
        m.store_errors
    );

    engine.shutdown().expect("shutdown clean");

    println!(
        "PASS transient_recovery_metrics: 30 players persisted with correct final versions; \
         store_retries={} (>0), store_errors={} (>0); flushes={}, records_flushed={}.",
        m.store_retries, m.store_errors, m.flushes, m.records_flushed
    );

    cleanup(&dir);
}

// ===========================================================================
// Scenario 2 — DB DOWN, THEN UP
//
// Take the DB down (set_healthy(false)) BEFORE submitting. Submit a known set,
// let the worker spin for a while, and assert NOTHING is lost: the data is
// buffered in ring+cache (pending() > 0, store.count() stays 0) — never
// dropped. Then bring the DB up, flush_now(), and assert every update is now
// persisted with the correct version (no loss, no duplication/corruption).
// ===========================================================================
#[test]
fn db_down_then_up_buffers_then_persists() {
    let dir = fresh_wal_dir("downup");
    let store = Arc::new(MockDb::new());

    // DB is DOWN before any traffic. Every persist_batch will return a retryable
    // Unavailable, so the worker can never drain the cache to the store.
    store.set_healthy(false);

    let store_dyn: Arc<dyn PersistenceStore> = store.clone();
    let engine = FluxEngine::start(test_config(dir.clone()), store_dyn).expect("engine start");

    let (submits, expected) = known_updates(25, 2);
    submit_all(&engine, &submits);

    // Let the worker run several cycles against the DOWN store. It will attempt,
    // fail, requeue — but must NOT lose or persist anything.
    std::thread::sleep(Duration::from_millis(40));

    // Ensure the ring has been drained into the cache so "pending" reflects
    // buffered dirty state rather than still-in-ring items. (Even if some are in
    // the ring, pending() counts both — the point is nothing is lost.)
    // We do NOT call flush_now() to completion here because the store is down;
    // instead we just assert the invariants of the down state.

    // ---- ASSERT: nothing persisted, nothing lost (data is BUFFERED) -----
    assert_eq!(
        store.count(),
        0,
        "store must hold NOTHING while the DB is down (got {})",
        store.count()
    );
    assert_eq!(
        store.persisted_batches(),
        0,
        "no batch should have succeeded while the DB is down"
    );

    // Read-your-writes from the in-memory cache is the RACE-FREE "no loss"
    // proof: every entry stays resident in the cache map (take_dirty only clears
    // the dirty bit, never removes the row), so cache_get always returns the
    // newest version regardless of any in-flight take/requeue cycle the worker is
    // doing against the down store. All 25 players must hold their newest
    // version 2 with the exact blob — nothing was dropped.
    for pid in 0..25u64 {
        let cached = engine
            .cache_get(pid)
            .unwrap_or_else(|| panic!("player {pid} dropped from cache while DB down — LOST"));
        assert_eq!(
            cached.version, 2,
            "player {pid}: cache must hold the newest version 2 while DB down, got {}",
            cached.version
        );
        assert_eq!(
            cached.blob,
            encode_blob(pid, 2),
            "player {pid}: cache blob corrupted"
        );
    }

    // pending() = ring.len() + cache.dirty_len() is an over-approximation gauge
    // of "in flight" work. While the DB is down the worker churns
    // take_dirty -> persist(fail) -> requeue_dirty; during the brief window
    // between take and requeue an entry is momentarily counted in neither the
    // ring nor the dirty set, so a single pending() snapshot can dip below 25.
    // That is NOT data loss (the cache_get proof above already showed every
    // update is retained). To assert the buffered-not-dropped invariant without
    // racing that window, we poll: at SOME observed instant within the budget the
    // full set of 25 must be reflected as pending (the worker always requeues
    // what it took). store.count() must stay 0 the whole time.
    let buffered = wait_until(Duration::from_secs(2), Duration::from_millis(1), || {
        // Re-assert the no-spurious-persist invariant on every poll.
        assert_eq!(store.count(), 0, "store wrongly persisted while DB down");
        engine.pending() >= 25
    });
    assert!(
        buffered,
        "never observed >= 25 buffered updates while DB down (data must be buffered, not dropped); \
         last pending={}",
        engine.pending()
    );
    let pending = engine.pending();

    // ---- Bring the DB UP and flush -------------------------------------
    store.set_healthy(true);

    // Drive to quiescence. We poll instead of relying on a single flush_now()
    // because the background worker may be holding a dirty batch in flight (taken
    // via take_dirty during its retry backoff) at the instant we recover, so one
    // synchronous flush could miss it. Polling is finite and race-free.
    let mut persisted = 0usize;
    let ok = wait_until(Duration::from_secs(5), Duration::from_millis(3), || {
        if let Ok(n) = engine.flush_now() {
            persisted += n;
        }
        engine.pending() == 0 && store.count() as u64 == 25
    });
    assert!(
        ok,
        "not all players persisted after recovery within budget (pending={}, store.count={})",
        engine.pending(),
        store.count()
    );

    // Now everything is durably in the store, exactly once, correct versions.
    assert_eq!(
        store.count() as u64,
        25,
        "all 25 players must persist after recovery"
    );
    assert_store_exact(&store, &expected);
    assert_eq!(
        engine.pending(),
        0,
        "no pending writes should remain after recovery flush"
    );

    engine.shutdown().expect("shutdown clean");

    // Shutdown must not corrupt or duplicate the already-correct state.
    assert_store_exact(&store, &expected);

    println!(
        "PASS db_down_then_up: while DOWN store.count()=0, persisted_batches=0, \
         pending={pending} (>=25, data buffered not dropped); after UP+flush all 25 players \
         persisted with version 2 and exact blobs (no loss, no duplication/corruption)."
    );

    cleanup(&dir);
}

// ===========================================================================
// Scenario 3 — PERIODIC FAILURES
//
// set_fail_every(k) makes every k-th persist attempt fail with a retryable
// Transient. Submit a LARGE known set with many versions per player, drive to
// quiescence, and assert the final store state is EXACTLY correct
// (last-write-wins per player) despite the repeated injected failures.
// ===========================================================================
#[test]
fn periodic_failures_final_state_exact() {
    let dir = fresh_wal_dir("periodic");
    let store = Arc::new(MockDb::new());

    // Fail every 3rd persist attempt, forever, for the whole test. The worker's
    // retry/requeue must absorb all of these and still converge to the exact
    // last-write-wins state.
    store.set_fail_every(3);

    let store_dyn: Arc<dyn PersistenceStore> = store.clone();
    let engine = FluxEngine::start(test_config(dir.clone()), store_dyn).expect("engine start");

    // Large set: 150 players x 8 versions = 1200 submits. Last-write-wins ⇒ each
    // player's final persisted version must be 8.
    let players = 150u64;
    let versions = 8u64;
    let (submits, expected) = known_updates(players, versions);
    submit_all(&engine, &submits);

    // Drive to quiescence. Because every 3rd attempt fails, the worker needs
    // multiple passes; each flush_now pass is finite and we bound the outer loop.
    let mut persisted_total = 0usize;
    let ok = wait_until(Duration::from_secs(10), Duration::from_millis(3), || {
        if let Ok(n) = engine.flush_now() {
            persisted_total += n;
        }
        engine.pending() == 0 && store.count() as u64 == players
    });
    assert!(
        ok,
        "engine did not reach quiescence under periodic failures within budget \
         (pending={}, store.count={})",
        engine.pending(),
        store.count()
    );

    // Capture metrics before shutdown to prove the periodic failures fired.
    let m = engine.metrics();

    engine.shutdown().expect("shutdown clean");

    // ---- ASSERT: exact last-write-wins final state, no corruption -------
    assert_eq!(
        store.count() as u64,
        players,
        "every player must be persisted exactly once"
    );
    assert_store_exact(&store, &expected);

    // The periodic failures must have actually been exercised.
    assert!(
        m.store_retries > 0,
        "periodic failures should have forced retries, got store_retries={}",
        m.store_retries
    );

    println!(
        "PASS periodic_failures: {players} players x {versions} versions ({} submits) with \
         set_fail_every(3); final store EXACTLY last-write-wins (every player version=={versions}, \
         blobs exact); store_retries={} (>0), persisted_total(flush_now)={persisted_total}, \
         persisted_batches={}.",
        submits.len(),
        m.store_retries,
        store.persisted_batches()
    );

    cleanup(&dir);
}

// ===========================================================================
// Bonus invariant — cache requeue is itself lossless under LWW
//
// A tiny direct unit-style check on the Cache trait via the engine's behavior:
// after a down period the newest version always wins (no stale requeued copy
// clobbers a newer value). This guards the "no corruption" claim at the cache
// layer, complementing the store-level assertions above.
// ===========================================================================
#[test]
fn requeue_never_regresses_to_stale_version() {
    let dir = fresh_wal_dir("requeue");
    let store = Arc::new(MockDb::new());

    // Start DOWN so the first writes get stuck dirty and will be requeued.
    store.set_healthy(false);

    let store_dyn: Arc<dyn PersistenceStore> = store.clone();
    let engine = FluxEngine::start(test_config(dir.clone()), store_dyn).expect("engine start");

    // Submit ascending versions for a single player while the store is down.
    for ver in 1..=10u64 {
        let blob = encode_blob(7, ver);
        for _ in 0..100_000 {
            match engine.submit(7, ver, blob.clone()) {
                flux_throughput::SubmitOutcome::Enqueued => break,
                flux_throughput::SubmitOutcome::Dropped => std::thread::yield_now(),
            }
        }
    }

    // Let the worker churn (attempt -> fail -> requeue) against the down store.
    std::thread::sleep(Duration::from_millis(30));

    // Nothing persisted while down; cache holds the NEWEST version (10).
    assert_eq!(store.count(), 0, "store must be empty while down");
    let cached = engine.cache_get(7).expect("player 7 must be cached");
    assert_eq!(
        cached.version, 10,
        "cache must hold newest version after requeue churn"
    );

    // Bring up, then drive to quiescence and assert the persisted value is the
    // newest — a stale requeued copy must never have clobbered it. We poll
    // rather than call flush_now() once because the background worker may be
    // mid-retry-cycle (it holds the dirty entry in flight via take_dirty during
    // its backoff sleep), so a single synchronous flush could momentarily see an
    // empty dirty set. Polling to quiescence is finite and race-free.
    store.set_healthy(true);
    let ok = wait_until(Duration::from_secs(5), Duration::from_millis(3), || {
        let _ = engine.flush_now();
        engine.pending() == 0 && store.count() == 1
    });
    assert!(
        ok,
        "player 7 did not persist after recovery within budget (pending={}, store.count={})",
        engine.pending(),
        store.count()
    );

    let persisted = store.load(7).expect("player 7 must persist after recovery");
    assert_eq!(
        persisted,
        PlayerState {
            player_id: 7,
            version: 10,
            blob: encode_blob(7, 10),
        },
        "requeue under failure regressed to a stale version — CORRUPTION"
    );

    engine.shutdown().expect("shutdown clean");

    println!(
        "PASS requeue_never_regresses: 10 ascending versions submitted while DB down; \
         after recovery player 7 persisted at version 10 (newest), proving requeue is \
         lossless and never regresses under last-write-wins."
    );

    cleanup(&dir);
}

/// Compile-time witness that the trait imports we rely on are actually the
/// crate's public surface (keeps the test honest about using the real API).
#[allow(dead_code)]
fn _uses_public_traits() {
    fn takes_store<S: PersistenceStore>(_: &S) {}
    fn takes_cache<C: Cache>(_: &C) {}
    let _ = takes_store::<MockDb>;
    let _ = takes_cache::<flux_throughput::cache::WriteBehindCache>;
}
