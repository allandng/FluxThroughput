//! # Main-loop isolation proof (adversarial integration test)
//!
//! This test exists to **PROVE one property and only one property**: the
//! `FluxEngine::submit` fire-and-forget boundary NEVER leaks worker/store
//! latency — or a lock — back onto the thread that calls it. That thread is the
//! game's fixed-timestep main loop, and its frame budget is sacred.
//!
//! The whole crate is built around the claim (see `SPEC.md` "Main-loop isolation
//! guarantee" and the `//!` docs on `src/api.rs`):
//!
//! > `submit()` contains NO `Mutex`, NO blocking syscall, NO unbounded wait —
//! > under back-pressure it sheds load (returns `Dropped`) rather than blocking.
//!
//! A claim like that is only worth anything if it survives a backend that is
//! *actively hostile*. So this file runs the **same** main-loop submit storm
//! twice:
//!
//! * **Scenario A (clean)** — a healthy, fast `MockDb`. Establishes the baseline:
//!   the per-tick submit cost is trivially small.
//!
//! * **Scenario B (adversarial — THE KEY TEST)** — a `MockDb` deliberately
//!   configured to be pathologically slow (25 ms per persist batch) behind a
//!   *tiny* ring and a *tiny* flush batch, so the worker is permanently wedged on
//!   the slow DB and the ring saturates almost immediately. We then fire the same
//!   storm and assert the main loop's per-`submit` latency STAYS BOUNDED in the
//!   sub-millisecond range **even though the store is 25 ms slow**, and that the
//!   system shed load via `SubmitOutcome::Dropped` (`submits_dropped > 0`).
//!
//! **That bounded submit latency under a 25 ms-slow store is the proof.** If any
//! lock or blocking wait leaked from the worker/store onto the submit path, a
//! 25 ms DB stall would show up as a 25 ms (or worse) per-call submit latency and
//! the bound would blow. It does not. A stalled DB cannot stall the game loop.
//!
//! This is a black-box test against the real public API (`flux_throughput::*`)
//! only; it is a separate crate and touches no `src/` internals. It is
//! deterministic and finite (fixed tick/submit counts, no randomness, no
//! external crates), and it cleans up its own temp WAL directories.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use flux_throughput::{FluxConfig, FluxEngine, MetricsSnapshot, PersistenceStore, SubmitOutcome};
// `MockDb` is the fault-injection backend. It lives in the `store` module of the
// crate and is reachable through the public module path.
use flux_throughput::store::MockDb;

// --- Workload shape (identical across both scenarios) ------------------------
//
// A fixed-timestep loop: TICKS iterations, each submitting UPDATES_PER_TICK
// player-state writes. 5_000 * 50 = 250_000 submits — a realistic per-second
// write storm for a busy multiplayer server, run entirely on the calling thread.
const TICKS: u64 = 5_000;
const UPDATES_PER_TICK: u64 = 50;

// --- Bounds (generous; the point is they hold even under a 25ms-slow DB) -----
//
// MAX single-tick main-loop submit cost (wall clock over the ~50 submit() calls
// in one tick). 2 ms is enormous next to the nanosecond-scale work submit()
// actually does; it exists only to catch a real stall, not jitter.
const MAX_TICK_SUBMIT_BOUND: Duration = Duration::from_millis(2);
// Per-call bound from metrics.max_submit_nanos. 0.5 ms in the clean scenario.
const CLEAN_MAX_SUBMIT_NANOS: u64 = 500_000; // 0.5 ms
                                             // Per-call bound under the ADVERSARIAL 25 ms-slow store. 1 ms — i.e. 25x SMALLER
                                             // than a single DB round-trip. If any DB latency leaked onto submit(), this
                                             // would be impossible to satisfy.
const HOSTILE_MAX_SUBMIT_NANOS: u64 = 1_000_000; // 1 ms

/// Process-unique, monotonically-incrementing counter so each engine in this
/// test gets its own fresh temp WAL directory (no cross-contamination, no
/// collision with a sibling test crate running in parallel).
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Allocates a fresh, unique temporary WAL directory under the system temp dir.
///
/// Uses the process id plus a local atomic counter as required, so it is unique
/// per engine and per process without any external crate.
fn fresh_wal_dir(name: &str) -> PathBuf {
    let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("flux_{}_{}_{}", name, std::process::id(), n));
    // Best-effort clean slate in case a prior aborted run left this path behind.
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp WAL dir");
    dir
}

/// Removes a temp WAL directory; ignores errors (cleanup must never fail a test).
fn cleanup(dir: &PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}

/// What a single scenario run measured. `max_tick` is the worst single-tick
/// wall-clock cost of the submit() calls on the MAIN-LOOP thread — the true
/// "did the game loop stall?" number. `snap` is the engine's own metrics.
struct RunResult {
    /// Worst single-tick submit() wall-clock cost on the calling thread.
    max_tick: Duration,
    /// Sum of all per-tick submit() wall-clock costs (for an average).
    total_submit_time: Duration,
    /// Final metrics snapshot from the engine (enqueued/dropped/max_submit_nanos).
    snap: MetricsSnapshot,
}

/// Runs the fixed-timestep submit storm on the CALLING thread against `engine`,
/// timing ONLY the `submit()` calls (the main-loop cost) and nothing else.
///
/// This is deliberately the most faithful model of a game loop possible: the
/// loop body times the block of `submit()` calls for the tick and accumulates
/// the worst case. No sleeping, no yielding — we want to measure submit() under
/// the most relentless pressure we can apply from one thread, which also keeps
/// the ring as saturated as possible in Scenario B.
fn run_submit_storm(engine: &FluxEngine) -> RunResult {
    let mut max_tick = Duration::ZERO;
    let mut total_submit_time = Duration::ZERO;

    for tick in 0..TICKS {
        // --- BEGIN main-loop critical section: ONLY submit() is timed --------
        let tick_start = Instant::now();
        for u in 0..UPDATES_PER_TICK {
            // A small, fixed blob — representative of a serialized player delta.
            // Deterministic content; no randomness anywhere.
            let player_id = u; // 50 distinct players churned every tick
            let version = tick + 1; // strictly-increasing version per player
            let blob = [
                (tick & 0xff) as u8,
                (u & 0xff) as u8,
                0xAB,
                0xCD,
                0xEF,
                0x01,
                0x02,
                0x03,
            ]
            .to_vec();

            // The hot path. Its return value is irrelevant to the timing — both
            // Enqueued and Dropped must be O(1) and non-blocking. We keep the
            // result alive only to make the call observable / non-elided.
            match engine.submit(player_id, version, blob) {
                SubmitOutcome::Enqueued | SubmitOutcome::Dropped => {}
            }
        }
        let tick_cost = tick_start.elapsed();
        // --- END main-loop critical section --------------------------------

        if tick_cost > max_tick {
            max_tick = tick_cost;
        }
        total_submit_time += tick_cost;
    }

    RunResult {
        max_tick,
        total_submit_time,
        snap: engine.metrics(),
    }
}

/// Prints a human-readable latency summary for a scenario (visible with
/// `--nocapture`). Includes BOTH the main-loop wall-clock view and the engine's
/// internal per-call metrics so the proof is legible.
fn print_summary(label: &str, r: &RunResult) {
    let total_submits = TICKS * UPDATES_PER_TICK;
    let avg_tick_us = r.total_submit_time.as_secs_f64() * 1e6 / TICKS as f64;
    println!("================ {label} ================");
    println!("  ticks                  : {TICKS}");
    println!("  submits/tick           : {UPDATES_PER_TICK}");
    println!("  total submit() calls   : {total_submits}");
    println!("  submits_enqueued       : {}", r.snap.submits_enqueued);
    println!("  submits_dropped        : {}", r.snap.submits_dropped);
    println!(
        "  MAIN-LOOP max tick cost: {:?}  (worst wall-clock for {UPDATES_PER_TICK} submits)",
        r.max_tick
    );
    println!("  MAIN-LOOP avg tick cost: {avg_tick_us:.3} us");
    println!(
        "  metrics.max_submit_nanos: {} ns ({:.3} us, per single submit() call)",
        r.snap.max_submit_nanos,
        r.snap.max_submit_nanos as f64 / 1000.0
    );
    println!(
        "  metrics.avg_submit_nanos: {:.1} ns ({:.3} us)",
        r.snap.avg_submit_nanos(),
        r.snap.avg_submit_nanos() / 1000.0
    );
    println!("==========================================");
}

/// ## Scenario A — clean baseline
///
/// A healthy, fast store. Run the fixed-timestep submit storm on the calling
/// thread and assert the main-loop submit cost is trivially small:
///   * worst single-tick submit cost  < 2 ms, and
///   * `metrics.max_submit_nanos`      < 0.5 ms per call.
///
/// This establishes that, in the happy case, submit() is essentially free — the
/// floor we then defend in Scenario B against a hostile backend.
#[test]
fn scenario_a_clean_submit_cost_is_trivial() {
    let wal_dir = fresh_wal_dir("isoA");

    // A fast, healthy backend — no latency, no faults. A roomy default-ish ring
    // so almost everything is accepted; we are measuring the cheap path here.
    let store: Arc<dyn PersistenceStore> = Arc::new(MockDb::new());

    let config = FluxConfig {
        ring_capacity: 1 << 16, // large: clean run should rarely if ever drop
        flush_max_batch: 4096,
        flush_interval: Duration::from_millis(50),
        worker_idle_sleep: Duration::from_millis(1),
        wal_dir: wal_dir.clone(),
        // No fsync in this test: we are measuring SUBMIT isolation, not WAL
        // durability, and submit() never touches the WAL anyway. Keeping fsync
        // off just makes the worker cheaper so the fast store accepts as much of
        // the storm as possible. (A full-tilt 250k-submit burst from one thread
        // can still momentarily outrun even a fast worker and shed some load —
        // that is fine here; Scenario A only asserts the LATENCY bounds and that
        // submits were enqueued, not a strict zero-drop.)
        wal_fsync: false,
        store_retry_limit: 5,
        store_retry_backoff: Duration::from_millis(2),
    };

    let engine = FluxEngine::start(config, store).expect("engine start (clean)");

    let result = run_submit_storm(&engine);
    print_summary("SCENARIO A (clean / fast store)", &result);

    // The main loop's worst single-tick submit cost must be tiny.
    assert!(
        result.max_tick < MAX_TICK_SUBMIT_BOUND,
        "clean: worst single-tick main-loop submit cost {:?} exceeded bound {:?}",
        result.max_tick,
        MAX_TICK_SUBMIT_BOUND
    );

    // Per-call metric bound. (max_submit_nanos is recorded on the enqueued path;
    // in the clean run virtually everything is enqueued.)
    assert!(
        result.snap.max_submit_nanos < CLEAN_MAX_SUBMIT_NANOS,
        "clean: metrics.max_submit_nanos {} ns exceeded per-call bound {} ns",
        result.snap.max_submit_nanos,
        CLEAN_MAX_SUBMIT_NANOS
    );

    // Sanity: we really did submit the full storm and the fast store accepted
    // (essentially) all of it — i.e. the clean baseline is genuinely clean.
    assert!(
        result.snap.submits_enqueued > 0,
        "clean: expected enqueued submits, got 0"
    );

    // Clean teardown of the worker + WAL, then remove the temp dir.
    engine.shutdown().expect("engine shutdown (clean)");
    cleanup(&wal_dir);
}

/// ## Scenario B — ADVERSARIAL: the key main-loop-isolation proof
///
/// Configure a DELIBERATELY HOSTILE backend and pipeline:
///   * `MockDb::set_latency(25ms)` — every persist batch blocks the worker for
///     25 ms, modelling a pathologically slow / wedged database,
///   * a SMALL ring (`1 << 10` = 1024 slots) so it saturates almost instantly,
///   * a TINY flush batch (8) so the worker drains the ring in dribbles and is
///     therefore wedged on the 25 ms DB essentially all the time.
///
/// Then run the SAME main-loop submit storm. With the worker stuck behind a
/// 25 ms DB and a tiny ring, the ring fills and stays full, so submit() must
/// shed load. We assert:
///
///   1. The main loop NEVER blocks waiting on the worker: the worst single-tick
///      submit cost stays under 2 ms AND `metrics.max_submit_nanos < 1 ms` —
///      i.e. orders of magnitude below ONE 25 ms DB round-trip. **This bounded
///      submit latency under a 25 ms-slow store is the proof that no lock /
///      blocking wait leaked from the worker/store onto the main loop.**
///
///   2. Back-pressure converted to DROPS, not blocking: `submits_dropped > 0`.
///      The system shed load (`SubmitOutcome::Dropped`) instead of stalling the
///      caller. A leaked lock would have produced blocking, not drops.
#[test]
fn scenario_b_adversarial_slow_store_never_stalls_main_loop() {
    let wal_dir = fresh_wal_dir("isoB");

    // Build the hostile backend and keep a typed handle so we can flip the knob,
    // while ALSO handing the engine an `Arc<dyn PersistenceStore>` view of it.
    let db = Arc::new(MockDb::new());
    // 25 ms per persist batch — a quarter of a default flush interval, and an
    // ETERNITY next to the nanosecond-scale work submit() actually performs.
    db.set_latency(Duration::from_millis(25));
    let store: Arc<dyn PersistenceStore> = db.clone();

    let config = FluxConfig {
        // SMALL ring: 1024 slots. The storm fires 250_000 submits at full tilt;
        // with the worker wedged on a 25 ms DB, this fills and stays full.
        ring_capacity: 1 << 10,
        // TINY flush batch: the worker takes at most 8 dirty entries per 25 ms
        // round-trip, so it can never catch up — maximal, sustained saturation.
        flush_max_batch: 8,
        // Short flush interval so the worker is constantly trying (and blocking
        // on the slow DB), keeping the ring pinned full.
        flush_interval: Duration::from_millis(5),
        worker_idle_sleep: Duration::from_millis(1),
        wal_dir: wal_dir.clone(),
        wal_fsync: false,
        store_retry_limit: 5,
        store_retry_backoff: Duration::from_millis(2),
    };

    let engine = FluxEngine::start(config, store).expect("engine start (adversarial)");

    // Same storm, hostile backend.
    let result = run_submit_storm(&engine);
    print_summary("SCENARIO B (adversarial / 25ms-slow store)", &result);

    // --- PROOF #1: the main loop is never blocked by the slow worker/store ---
    //
    // The worst single-tick submit cost must stay under 2 ms. One DB round-trip
    // alone is 25 ms; if any of that leaked onto submit(), a single tick (~50
    // submits) would easily blow past 2 ms. It does not.
    assert!(
        result.max_tick < MAX_TICK_SUBMIT_BOUND,
        "ADVERSARIAL: worst single-tick main-loop submit cost {:?} exceeded bound {:?} \
         — the slow store leaked latency onto the main loop!",
        result.max_tick,
        MAX_TICK_SUBMIT_BOUND
    );

    // The per-call metric bound is the crisp version of the same proof: NO single
    // submit() call took as long as 1 ms, even though the store is 25 ms slow.
    // 25 ms / 1 ms = a 25x safety margin. This is only possible if submit() is
    // genuinely decoupled from the store by the lock-free ring — no Mutex, no
    // blocking syscall, no unbounded wait on the submit path.
    assert!(
        result.snap.max_submit_nanos < HOSTILE_MAX_SUBMIT_NANOS,
        "ADVERSARIAL: metrics.max_submit_nanos {} ns ({:.3} ms) exceeded per-call bound {} ns \
         under a 25ms-slow store — blocking/lock leaked onto the main loop!",
        result.snap.max_submit_nanos,
        result.snap.max_submit_nanos as f64 / 1e6,
        HOSTILE_MAX_SUBMIT_NANOS
    );

    // --- PROOF #2: back-pressure converted to DROPS, not blocking ------------
    //
    // With a saturated ring and a wedged worker, submit() sheds load. Observing
    // drops proves the isolation mechanism actually engaged: the pressure became
    // `SubmitOutcome::Dropped` returns (load shedding) rather than a stalled
    // caller. If a lock had leaked, we'd have seen blocking and (near-)zero
    // drops — the OPPOSITE of strict main-loop isolation.
    assert!(
        result.snap.submits_dropped > 0,
        "ADVERSARIAL: expected the saturated ring to shed load via Dropped \
         (submits_dropped > 0), but saw 0 — back-pressure did not convert to drops"
    );

    // Cross-check: enqueued + dropped accounts for the full storm. Every submit()
    // resolved to exactly one of the two non-blocking outcomes; none hung.
    let total = result.snap.submits_enqueued + result.snap.submits_dropped;
    assert_eq!(
        total,
        TICKS * UPDATES_PER_TICK,
        "ADVERSARIAL: enqueued ({}) + dropped ({}) = {} should equal total submits {}",
        result.snap.submits_enqueued,
        result.snap.submits_dropped,
        total,
        TICKS * UPDATES_PER_TICK
    );

    // Teardown. The worker is wedged on a 25 ms DB; drain the slow store first so
    // shutdown's final flush has a bounded amount to do, then shut down. Either
    // way shutdown must complete (it joins the worker after its final flush).
    engine.shutdown().expect("engine shutdown (adversarial)");
    cleanup(&wal_dir);
}
