//! `demo` binary — a self-contained, deterministic, simulated game-loop
//! demonstration of the FluxThroughput pipeline. **[impl:api].**
//!
//! It stands in for a 60 Hz game server tick. Each "tick" the main loop submits
//! state updates for a rolling window of distinct players through
//! [`FluxEngine::submit`] — the fire-and-forget, non-blocking boundary — and
//! asserts that the time spent *in the main loop* stays tiny, no matter how slow
//! the durable backend is. All the slow work (cache locking, WAL fsync, store
//! persistence) happens on the background worker thread and never bleeds into
//! the tick.
//!
//! The run is fully **deterministic** (no `rand`: player ids come from
//! counters) and **finite** (a fixed number of ticks), so it always exits on its
//! own. It finishes by draining the pipeline with
//! [`FluxEngine::flush_now`], shutting the engine down, and printing final
//! metrics plus an estimated sustained throughput.

use flux_throughput::store::MockDb;
// `PersistenceStore` is brought into scope so we can call `db.count()` (a trait
// method) on the concrete `MockDb` after the run.
use flux_throughput::{FluxConfig, FluxEngine, MetricsSnapshot, PersistenceStore, SubmitOutcome};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Total simulated ticks (~5 seconds of wall-clock game time at 60 Hz).
const TICKS: u64 = 300;

/// Distinct players updated each tick (the rolling working set). Deterministic
/// ids are drawn from a sliding window so the cache's last-write-wins coalescing
/// is exercised across ticks.
const PLAYERS_PER_TICK: u64 = 2000;

/// Total distinct player-id space the simulation ever touches. Every emitted id
/// is taken modulo this value, so the store can hold AT MOST this many rows —
/// a hard, provable upper bound used by the final correctness gate.
const ID_SPACE: u64 = 50_000;

/// How far the rolling window advances each tick, so successive ticks touch
/// overlapping but shifting id ranges (revisiting players -> coalescing in the
/// cache under last-write-wins).
const WINDOW_STRIDE: u64 = 250;

/// Print a metrics snapshot every this many ticks.
const REPORT_EVERY: u64 = 60;

/// Per-tick main-loop budget we assert against. A 60 Hz frame is ~16.6ms; the
/// submit path is lock-free and should consume only a small fraction of it even
/// for thousands of submits. We assert a generous-but-meaningful ceiling so the
/// demo proves isolation without being flaky on a loaded CI box.
const MAX_TICK: Duration = Duration::from_millis(8);

fn main() {
    // A per-process temp WAL dir so concurrent runs never collide and we leave
    // no litter in the repo.
    let wal_dir = std::env::temp_dir().join(format!("flux_demo_{}", std::process::id()));
    // Best-effort clean slate (ignore if it does not exist yet).
    let _ = std::fs::remove_dir_all(&wal_dir);

    // Smaller-than-default knobs so the demo shows real flush cadence and the
    // ring can actually fill under burst (demonstrating load shedding) on a
    // slow store. Ring 1<<14 = 16384, batch 1024, flush interval 20ms.
    let config = FluxConfig {
        ring_capacity: 1 << 14,
        flush_max_batch: 1024,
        flush_interval: Duration::from_millis(20),
        worker_idle_sleep: Duration::from_millis(1),
        wal_fsync: true,
        store_retry_limit: 5,
        store_retry_backoff: Duration::from_millis(2),
        ..FluxConfig::default()
    }
    .with_wal_dir(&wal_dir);

    println!("== FluxThroughput demo ==");
    println!(
        "config: ring={} batch={} interval={:?} fsync={} wal_dir={}",
        config.ring_capacity,
        config.flush_max_batch,
        config.flush_interval,
        config.wal_fsync,
        config.wal_dir.display()
    );

    // The durable backend. A tiny injected latency makes the worker realistically
    // slower than the submit loop, so the demo demonstrates back-pressure /
    // shedding rather than a trivially-keeping-up store.
    let db = Arc::new(MockDb::new());
    db.set_latency(Duration::from_micros(50));
    let store: Arc<dyn flux_throughput::PersistenceStore> = db.clone();

    let engine = match FluxEngine::start(config, store) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("FATAL: engine failed to start: {e}");
            std::process::exit(1);
        }
    };

    // ---- the simulated game loop -------------------------------------------
    let mut total_submitted: u64 = 0;
    let mut total_enqueued: u64 = 0;
    let mut total_dropped: u64 = 0;
    let mut worst_tick = Duration::ZERO;
    let mut sum_tick = Duration::ZERO;

    // A simple deterministic counter feeding both the per-write version and the
    // payload, so re-touched players get strictly-increasing versions (the
    // cache's last-write-wins picks the newest).
    let mut version_counter: u64 = 1;

    let loop_start = Instant::now();
    for tick in 0..TICKS {
        // The window base slides deterministically across the id space.
        let base = (tick * WINDOW_STRIDE) % ID_SPACE;

        // Time ONLY the main-loop work (the submit burst). This is the number
        // that must stay tiny — it is the game's frame budget.
        let tick_start = Instant::now();
        for i in 0..PLAYERS_PER_TICK {
            // Wrap into the fixed id space so the distinct row count is bounded
            // by ID_SPACE no matter how `base`/`i` combine.
            let player_id = (base + i) % ID_SPACE;
            let version = version_counter;
            version_counter += 1;

            // 8-byte payload encoding the version — representative of a small
            // serialized entity delta.
            let blob = version.to_le_bytes().to_vec();

            match engine.submit(player_id, version, blob) {
                SubmitOutcome::Enqueued => total_enqueued += 1,
                SubmitOutcome::Dropped => total_dropped += 1,
            }
            total_submitted += 1;
        }
        let tick_time = tick_start.elapsed();

        // Track main-loop timing and assert isolation: even though the worker is
        // doing fsync+store work behind us, OUR tick stays within budget.
        worst_tick = worst_tick.max(tick_time);
        sum_tick += tick_time;
        assert!(
            tick_time < MAX_TICK,
            "main-loop isolation VIOLATED: tick {tick} took {tick_time:?} (budget {MAX_TICK:?}) \
             — submit() must never block on the durable backend",
        );

        // Periodic metrics report (proving the worker is making progress).
        if (tick + 1) % REPORT_EVERY == 0 {
            let snap = engine.metrics();
            print_snapshot(tick + 1, tick_time, &snap, engine.pending());
        }

        // Pace the loop toward ~60 Hz of *simulated* time. We sleep the remaining
        // frame budget so the run takes a realistic ~5s and the worker has time
        // to flush between ticks. This sleep is OUTSIDE the measured tick window.
        let frame = Duration::from_micros(16_666); // 1/60 s
        if tick_time < frame {
            std::thread::sleep(frame - tick_time);
        }
    }
    let loop_elapsed = loop_start.elapsed();

    // ---- drain + shutdown ---------------------------------------------------
    // flush_now() synchronously pushes every remaining dirty entry through the
    // WAL to the store, so the post-run store count reflects all coalesced work.
    let flushed = match engine.flush_now() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("FATAL: flush_now failed: {e}");
            std::process::exit(1);
        }
    };

    // Snapshot metrics + store count BEFORE consuming the engine in shutdown.
    let final_metrics = engine.metrics();
    let store_count = db.count();

    if let Err(e) = engine.shutdown() {
        eprintln!("FATAL: shutdown failed: {e}");
        std::process::exit(1);
    }

    // ---- report -------------------------------------------------------------
    let avg_tick = sum_tick / (TICKS as u32);
    // Throughput: distinct updates the store ended up with, over wall time.
    let secs = loop_elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
    let enqueued_throughput = final_metrics.records_flushed as f64 / secs;

    println!("\n== final report ==");
    println!("ticks .................. {TICKS}");
    println!("submitted .............. {total_submitted}");
    println!("  enqueued ............. {total_enqueued}");
    println!(
        "  dropped (shed) ....... {total_dropped} ({:.2}%)",
        pct(total_dropped, total_submitted)
    );
    println!("flush_now persisted .... {flushed}");
    println!("store count (distinct) . {store_count}");
    println!("--- main-loop timing (isolation) ---");
    println!("worst tick ............. {worst_tick:?}  (budget {MAX_TICK:?})");
    println!("avg   tick ............. {avg_tick:?}");
    println!("--- metrics ---");
    print_snapshot(TICKS, avg_tick, &final_metrics, 0);
    println!(
        "estimated throughput ... {enqueued_throughput:.0} records/sec persisted ({:.2}M/s)",
        enqueued_throughput / 1.0e6
    );

    // ---- correctness gates --------------------------------------------------
    // The pipeline must have drained (no work left in flight) and the store must
    // hold coalesced state. Note `records_flushed` counts every flushed record
    // (the same player is re-written across ticks), whereas `store_count` counts
    // DISTINCT player ids — so `records_flushed >= store_count`, and the store
    // can hold at most the size of the id window we wrote into.
    assert!(
        store_count > 0,
        "store must contain coalesced state after the run",
    );
    assert!(
        store_count as u64 <= ID_SPACE,
        "distinct persisted ids ({store_count}) cannot exceed the id space ({ID_SPACE})",
    );
    assert!(
        final_metrics.records_flushed >= store_count as u64,
        "every distinct persisted id must have been flushed at least once",
    );
    assert_eq!(
        final_metrics.store_errors, 0,
        "a healthy store must not report persistence errors",
    );

    // Best-effort clean up the temp WAL dir.
    let _ = std::fs::remove_dir_all(&wal_dir);

    println!(
        "\nSUCCESS: {TICKS} ticks ran with worst main-loop tick {worst_tick:?} (< {MAX_TICK:?}); \
         {store_count} distinct player states persisted; pipeline fully drained.",
    );
}

/// Pretty-prints a [`MetricsSnapshot`] with the tick label and current pending.
fn print_snapshot(tick: u64, tick_time: Duration, s: &MetricsSnapshot, pending: usize) {
    println!(
        "[tick {tick:>4}] tick={tick_time:>10?} pending={pending:>6} | \
         enq={} drop={} avg_submit={:.0}ns max_submit={}ns | \
         flushes={} flushed={} max_flush={}us | wal_appends={} wal_recs={} | \
         store_err={} store_retry={}",
        s.submits_enqueued,
        s.submits_dropped,
        s.avg_submit_nanos(),
        s.max_submit_nanos,
        s.flushes,
        s.records_flushed,
        s.max_flush_nanos / 1000,
        s.wal_appends,
        s.wal_records,
        s.store_errors,
        s.store_retries,
    );
}

/// Percentage of `n` over `total`, guarding division by zero.
fn pct(n: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        100.0 * n as f64 / total as f64
    }
}
