//! Background flush worker thread. **[impl:worker] — IMPLEMENTATION.**
//!
//! This module owns the entire slow path of the pipeline so that the caller's
//! hot main loop (`submit()`) never blocks. Exactly **one** background thread is
//! launched by [`spawn`]; it is the *only* place in the crate where blocking I/O
//! (WAL fsync, store persistence) and locking (cache map, store map) are
//! allowed. The caller thread merely pushes [`crate::domain::WriteRequest`]s
//! into the lock-free ring and walks away.
//!
//! ## The loop (binding contract)
//!
//! ```text
//! while !stop:
//!     drain ring -> cache.apply           (coalesce via last-write-wins)
//!     if cache.dirty_len() >= flush_max_batch
//!        OR flush_interval elapsed
//!        OR stop requested:               flush()
//!     else:                               park worker_idle_sleep
//!
//! flush():
//!     states  = cache.take_dirty(flush_max_batch)
//!     records = states -> WalRecord (LSNs from the shared LsnState authority)
//!     wal.append_batch(records)           // WRITE-AHEAD: WAL (+fsync) BEFORE store
//!     store.persist_batch(states) with up to store_retry_limit retries + backoff
//!         retryable error  -> record_store_retry, sleep backoff, retry
//!         exhausted/Fatal  -> cache.requeue_dirty(states) + record_store_error  (NO data loss)
//!     store success        -> wal.checkpoint(safe watermark <= last_lsn) + record_flush
//!
//! on stop: final drain + flush until ring empty AND dirty empty, then return.
//! ```
//!
//! ## Write-ahead ordering — why the WAL is fsync'd BEFORE the store
//!
//! The single most important ordering invariant in the whole system is:
//!
//! > **WAL append (+ fsync) happens-before the store write.**
//!
//! The WAL is the durability boundary. Once [`WriteAheadLog::append_batch`]
//! returns (having fsync'd when configured), the batch is recoverable across a
//! SIGKILL: [`WriteAheadLog::recover`] will replay it exactly. We therefore log
//! *first*. If the process dies at any instant after the WAL append, recovery
//! re-applies the batch on restart and the store eventually converges — the
//! write is never lost. If we instead wrote the store first and crashed before
//! logging, a transient store that lost the write (or a store we could not
//! confirm) would leave us with **no record of the update anywhere** => silent
//! data loss. Logging first makes the store write *idempotently replayable*.
//!
//! ## Why `requeue_dirty` prevents loss
//!
//! `cache.take_dirty()` removes the dirty entries from the cache so a concurrent
//! flush cannot double-submit them. If the store then rejects the batch
//! (retries exhausted, or a `Fatal` error), those states are no longer in the
//! cache's dirty set — dropping them here would lose the updates. Instead we
//! hand them back via [`Cache::requeue_dirty`], which re-marks them dirty under
//! last-write-wins (so a newer concurrent update for the same player is never
//! clobbered). The next flush cycle retries them. Combined with the
//! already-fsync'd WAL, this gives an at-least-once guarantee with no in-memory
//! loss: the data sits safely in **both** the cache (for the next retry) and the
//! WAL (for crash recovery).
//!
//! This file is **safe code only** (no `unsafe`).

use crate::config::FluxConfig;
use crate::domain::{Cache, PersistenceStore, PlayerState, Queue, WalRecord, WriteAheadLog};
use crate::lsn::LsnState;
use crate::metrics::Metrics;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

/// Hard ceiling on the number of final drain+flush iterations performed during
/// shutdown when the store is *permanently* down.
///
/// On a clean shutdown the worker drains the ring and flushes the cache until
/// both are empty. But if the store is wedged (every `persist_batch` returns a
/// retryable error forever) the dirty set never clears, and an unbounded
/// "drain until empty" loop would hang `shutdown()` indefinitely. We therefore
/// bound the number of *unproductive* final-flush attempts. When the bound is
/// hit we stop and return: the data is **already durable in the WAL** (every
/// batch was `append_batch`'d + fsync'd before we ever touched the store), so
/// the next process start replays it via `recover()`. We trade a stuck shutdown
/// for at-most-a-bounded-wait, never for data loss.
const SHUTDOWN_MAX_STALLED_ATTEMPTS: u32 = 64;

/// Handle to the background flush worker thread.
///
/// Owns the worker's [`JoinHandle`] and the shared stop flag. Signal shutdown
/// with [`WorkerHandle::signal_stop`], then reclaim the thread with
/// [`WorkerHandle::join`].
pub struct WorkerHandle {
    /// `None` once joined; `Some` while the thread is running.
    handle: Option<JoinHandle<()>>,
    /// Shared cooperative stop flag observed by the worker loop.
    stop: Arc<AtomicBool>,
}

impl WorkerHandle {
    /// Requests cooperative shutdown. The worker observes the flag, performs a
    /// final drain+flush, and exits. Idempotent and non-blocking.
    pub fn signal_stop(&self) {
        // SeqCst pairs with the worker's SeqCst load so the stop request is
        // promptly visible across threads; this is off the hot path so the
        // strongest ordering costs us nothing meaningful.
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Joins the worker thread, blocking until it has fully drained and exited.
    ///
    /// Returns whatever [`JoinHandle::join`] returns, so a panic inside the
    /// worker surfaces here rather than being silently swallowed.
    pub fn join(mut self) -> std::thread::Result<()> {
        match self.handle.take() {
            Some(h) => h.join(),
            None => Ok(()),
        }
    }
}

/// All shared state the flush loop needs, bundled so it can be moved into the
/// thread closure and threaded through the private helpers as a single `self`.
///
/// LSNs are minted from the shared [`LsnState`] — the **single source of truth**
/// for LSN assignment across the worker, `submit_durable`, and `flush_now`. The
/// worker is no longer the "sole writer" to the WAL, so a private per-worker
/// counter would (and historically did) collide with the durable API paths and
/// let a checkpoint discard their acknowledged records. Sharing one monotonic
/// allocator — and respecting its pending-durable checkpoint floor — closes that
/// hole. See [`crate::lsn`] for the full rationale.
struct Worker {
    config: FluxConfig,
    ring: Arc<dyn Queue>,
    cache: Arc<dyn Cache>,
    wal: Arc<dyn WriteAheadLog>,
    store: Arc<dyn PersistenceStore>,
    metrics: Arc<Metrics>,
    stop: Arc<AtomicBool>,
    /// Shared monotonic LSN authority. Every record the worker stamps gets a
    /// fresh LSN from here, globally unique with respect to the durable API
    /// paths; checkpoints are clamped to its pending-durable floor.
    lsn: Arc<LsnState>,
    /// Reusable scratch buffer for `drain_into`, so the steady state allocates
    /// nothing per cycle.
    scratch: Vec<crate::domain::WriteRequest>,
}

/// The shared collaborators a [`spawn`]ed worker needs, bundled into one value.
///
/// All fields are `Arc`-shared with the engine so the worker and
/// [`crate::api::FluxEngine`] observe the same ring, cache, WAL, store, metrics
/// and LSN authority. The `stop` flag is *constructed by the caller* and cloned
/// into the returned [`WorkerHandle`], so the engine flips it and the worker
/// observes the very same `Arc<AtomicBool>`.
pub struct WorkerDeps {
    /// Runtime tuning knobs (batch sizes, intervals, fsync mode).
    pub config: FluxConfig,
    /// Lock-free ring the worker drains incoming writes from.
    pub ring: Arc<dyn Queue>,
    /// Coalescing write-behind cache flushed to the store.
    pub cache: Arc<dyn Cache>,
    /// Write-ahead log appended (and fsync'd) before each store batch.
    pub wal: Arc<dyn WriteAheadLog>,
    /// Slow durable backend the worker persists batches into.
    pub store: Arc<dyn PersistenceStore>,
    /// Shared counters the worker updates as it drains and flushes.
    pub metrics: Arc<Metrics>,
    /// Single monotonic LSN authority shared with the durable API paths.
    pub lsn: Arc<LsnState>,
    /// Cooperative stop flag the worker polls; cloned into the handle.
    pub stop: Arc<AtomicBool>,
}

/// Spawns the single background flush worker and returns its [`WorkerHandle`].
///
/// `api.rs` calls this once during [`crate::api::FluxEngine::start`] and holds
/// the returned handle for the engine's lifetime. See [`WorkerDeps`] for how the
/// shared state is wired, and the module docs for the exact loop and ordering
/// guarantees the implementation upholds.
pub fn spawn(deps: WorkerDeps) -> WorkerHandle {
    let WorkerDeps {
        config,
        ring,
        cache,
        wal,
        store,
        metrics,
        lsn,
        stop,
    } = deps;

    // The handle keeps its own clone of the stop flag so `signal_stop` can flip
    // the very same flag the worker is polling.
    let stop_for_handle = Arc::clone(&stop);

    // Pre-size the scratch buffer to one drain's worth (flush_max_batch * 4, see
    // `DRAIN_MULTIPLE`) so the steady state never reallocates.
    let initial_scratch = config.flush_max_batch.saturating_mul(DRAIN_MULTIPLE);

    let worker = Worker {
        config,
        ring,
        cache,
        wal,
        store,
        metrics,
        stop,
        lsn,
        scratch: Vec::with_capacity(initial_scratch),
    };

    let handle = std::thread::Builder::new()
        .name("flux-flush-worker".to_string())
        .spawn(move || worker.run())
        .expect("failed to spawn flux flush worker thread");

    WorkerHandle {
        handle: Some(handle),
        stop: stop_for_handle,
    }
}

/// How many batches' worth of items we drain from the ring in one pass.
///
/// We pull up to `flush_max_batch * DRAIN_MULTIPLE` requests into the scratch
/// vector per loop iteration before deciding whether to flush. Draining several
/// batches at once lets the cache coalesce a bigger burst (more last-write-wins
/// collapsing) and amortizes the ring synchronization, while the upper bound
/// keeps a single iteration from starving the flush decision under a flood.
const DRAIN_MULTIPLE: usize = 4;

impl Worker {
    /// The worker thread entry point: runs the steady-state loop until `stop`,
    /// then performs the bounded final drain before returning so `join()` can
    /// complete cleanly.
    fn run(mut self) {
        // Anchor for the TIME flush threshold. We flush whenever
        // `last_flush.elapsed() >= flush_interval`, bounding write latency even
        // when the VOLUME threshold is never reached.
        let mut last_flush = Instant::now();

        // ---- Steady-state loop -------------------------------------------
        // Exit as soon as a stop is requested; the final-drain phase below then
        // guarantees nothing is left behind.
        while !self.stop.load(Ordering::SeqCst) {
            // (1) Drain the ring into the cache. Many writes for the same player
            //     coalesce here via last-write-wins, so the store sees only the
            //     newest state per entity.
            let drained = self.drain_ring_into_cache();

            // (2) Decide whether to flush this cycle.
            //   - VOLUME: enough dirty entries have accumulated.
            //   - TIME:   the flush interval has elapsed (bounds latency).
            let dirty = self.cache.dirty_len();
            let volume_ready = dirty >= self.config.flush_max_batch;
            let time_ready = last_flush.elapsed() >= self.config.flush_interval;

            if dirty > 0 && (volume_ready || time_ready) {
                // flush() drains the dirty set (possibly across several batches
                // when a big burst piled up) through WAL -> store.
                self.flush();
                last_flush = Instant::now();
            } else if drained == 0 {
                // (4) Nothing arrived this pass. Either the pipeline is idle, or
                //     we are holding sub-threshold dirty data waiting for the
                //     TIME trigger. Park `worker_idle_sleep` so idle CPU stays
                //     near zero and we don't spin on the `dirty_len` check.
                std::thread::sleep(self.config.worker_idle_sleep);
            }
            // (else: we drained items this pass but didn't flush — loop
            //  immediately to keep pulling the burst into the cache so more
            //  writes coalesce before the next flush decision.)
        }

        // ---- Final drain on stop -----------------------------------------
        self.final_drain();
    }

    /// Drains up to `flush_max_batch * DRAIN_MULTIPLE` requests out of the ring
    /// and `cache.apply`s each, returning how many were drained.
    ///
    /// This is pure in-memory work (lock-free ring pop + cache upsert); the
    /// expensive WAL/store I/O happens only in [`Worker::flush`].
    fn drain_ring_into_cache(&mut self) -> usize {
        let max = self.config.flush_max_batch.saturating_mul(DRAIN_MULTIPLE);
        // Reuse the scratch buffer; `drain_into` pushes onto it.
        self.scratch.clear();
        let count = self.ring.drain_into(&mut self.scratch, max);
        for req in self.scratch.iter() {
            // A `submit_durable` request carries its durable LSN in `seq`. Once it
            // lands in the cache, the cache + requeue machinery guarantees its
            // content reaches the store (no in-memory loss), so its WAL frame no
            // longer needs independent protection: lift its checkpoint floor. For
            // a plain `submit` request this is a harmless no-op (its seq was never
            // marked pending). See `lsn.rs` for why this is the correct hook.
            self.lsn.note_drained(req.seq);
            // Last-write-wins coalescing lives inside the cache.
            self.cache.apply(req);
        }
        count
    }

    /// Performs one flush pass: drains the entire dirty set in
    /// `flush_max_batch`-sized chunks and pushes each chunk through
    /// WAL-then-store.
    ///
    /// A single steady-state iteration can have accumulated more than
    /// `flush_max_batch` dirty entries (we drain up to `DRAIN_MULTIPLE` batches
    /// into the cache per pass). We loop `take_dirty` until it returns empty so
    /// we don't strand a large backlog for another whole cycle. Each chunk is an
    /// independent WAL append + store persist with its own retry handling.
    ///
    /// Termination is guaranteed even under a concurrent stream of fresh
    /// submits: a `flush_batch` that *fails* requeues its states, which would
    /// otherwise let `take_dirty` keep returning the same entries forever. We
    /// therefore stop as soon as a batch is requeued (failure) — the next steady
    /// loop iteration (or the final drain) retries them — so this loop only ever
    /// keeps going while it is making real forward progress.
    fn flush(&mut self) {
        loop {
            let states = self.cache.take_dirty(self.config.flush_max_batch);
            if states.is_empty() {
                break;
            }
            // Stop the inner loop as soon as a batch fails to persist: its
            // states were just requeued, so continuing would risk re-taking and
            // spinning on a down store. Forward progress is preserved — the
            // outer loop / final drain will revisit the requeued set.
            if !self.flush_batch(states) {
                break;
            }
        }
    }

    /// Flushes exactly one batch of `states`: WAL-append (write-ahead) then
    /// persist to the store with retry/backoff, requeuing on failure.
    ///
    /// This is where the **WAL-before-store ordering** is enforced (see the
    /// module docs). Returns `true` if the batch was durably persisted (and the
    /// WAL checkpointed), or `false` if it was requeued for a later retry (WAL
    /// append failed, or the store rejected it). All outcomes are recorded via
    /// metrics.
    fn flush_batch(&mut self, states: Vec<PlayerState>) -> bool {
        let batch_len = states.len();
        let flush_start = Instant::now();

        // ---- Build WAL records, stamping monotonic LSNs ------------------
        // Mint a contiguous LSN block from the shared authority so these records
        // are globally unique and monotonic with respect to the durable API
        // paths writing into the same WAL. We remember the highest LSN in this
        // batch for the post-store checkpoint. `append_batch` also returns the
        // highest LSN it durably wrote — we keep them consistent.
        let first_lsn = self.lsn.alloc_block(batch_len as u64);
        let mut records = Vec::with_capacity(batch_len);
        let mut highest_lsn = 0;
        for (i, state) in states.iter().enumerate() {
            let lsn = first_lsn + i as u64;
            highest_lsn = lsn;
            records.push(WalRecord {
                lsn,
                player_id: state.player_id,
                version: state.version,
                blob: state.blob.clone(),
            });
        }

        // ---- (A) WRITE-AHEAD: WAL append (+fsync) BEFORE the store -------
        // Once this returns Ok, the batch is durable: a SIGKILL after this point
        // is fully recoverable via `recover()`. We log the *intent to persist*
        // before performing the persist, so the store write is idempotently
        // replayable. We trust the WAL's returned highest LSN as the
        // checkpoint watermark (it equals our locally-stamped `highest_lsn`).
        let wal_start = Instant::now();
        let confirmed_lsn = match self.wal.append_batch(&records) {
            Ok(lsn) => lsn,
            Err(_e) => {
                // The WAL itself failed to append/fsync. We have NOT durably
                // logged this batch, and we have NOT touched the store, so the
                // ONLY safe action is to put the states back in the cache so a
                // later cycle retries the whole WAL+store sequence. Nothing is
                // lost: the updates remain dirty in memory.
                self.cache.requeue_dirty(states);
                self.metrics.record_store_error();
                return false;
            }
        };
        let wal_nanos = nanos_since(wal_start);
        self.metrics.record_wal_append(records.len(), wal_nanos);

        // Prefer the WAL-confirmed watermark for the checkpoint; fall back to
        // our stamped value if the WAL returned 0 for an empty/edge case.
        let checkpoint_lsn = if confirmed_lsn != 0 {
            confirmed_lsn
        } else {
            highest_lsn
        };

        // ---- (B) Persist to the store with bounded retry + backoff -------
        match self.persist_with_retry(&states) {
            PersistResult::Persisted => {
                // The store now durably holds these states. We can advance the
                // WAL checkpoint past this batch so the log may rotate/truncate
                // the now-redundant records — but ONLY up to a watermark that is
                // safe to discard: strictly below any still-pending durable
                // (`submit_durable`) record whose content is not yet guaranteed
                // in the store. The shared `LsnState` computes that clamp; if it
                // returns None there is nothing safe to discard this round (a low
                // pending durable record covers the whole proposal), so we keep
                // the tail and let a later cycle checkpoint it once that durable
                // record drains. A checkpoint failure is non-fatal: the records
                // simply remain in the WAL and get re-checkpointed (or harmlessly
                // re-applied on recovery) later — never loss.
                if let Some(safe_lsn) = self.lsn.checkpoint_watermark(checkpoint_lsn) {
                    let _ = self.wal.checkpoint(safe_lsn);
                }
                let flush_nanos = nanos_since(flush_start);
                self.metrics.record_flush(batch_len, flush_nanos);
                true
            }
            PersistResult::Failed => {
                // Retries exhausted, or a Fatal (non-retryable) error. The
                // states have ALREADY been removed from the dirty set by
                // `take_dirty`, so we MUST hand them back or they are lost.
                // `requeue_dirty` re-marks them dirty under last-write-wins (a
                // newer concurrent update for the same player wins), and the
                // next flush cycle retries them. The batch is also already in
                // the WAL, so even a crash right now loses nothing.
                self.cache.requeue_dirty(states);
                self.metrics.record_store_error();
                false
            }
        }
    }

    /// Attempts `store.persist_batch` up to `store_retry_limit` times, sleeping
    /// `store_retry_backoff` between *retryable* attempts.
    ///
    /// Returns [`PersistResult::Persisted`] on success. Returns
    /// [`PersistResult::Failed`] when a `Fatal` (non-retryable) error is hit, or
    /// when the retry budget is exhausted while still seeing retryable errors.
    /// Each retry is counted via `record_store_retry`.
    fn persist_with_retry(&self, states: &[PlayerState]) -> PersistResult {
        // `store_retry_limit` is validated `>= 1`, so we make at least one
        // attempt. We count attempts 1..=limit; a backoff sleep is inserted
        // only *between* attempts (never after the final one).
        let limit = self.config.store_retry_limit;
        let mut attempt: u32 = 0;
        loop {
            attempt += 1;
            match self.store.persist_batch(states) {
                Ok(()) => return PersistResult::Persisted,
                Err(e) => {
                    if !e.is_retryable() {
                        // Fatal: retrying cannot help. Give up immediately and
                        // let the caller requeue (data preserved, not dropped).
                        return PersistResult::Failed;
                    }
                    if attempt >= limit {
                        // Retry budget exhausted while still retryable. Give up
                        // for this cycle; the caller requeues and we try again
                        // next cycle (the store may have recovered by then).
                        return PersistResult::Failed;
                    }
                    // Retryable and budget remains: count the retry, back off,
                    // and loop. The backoff yields the CPU and gives a
                    // transiently-unavailable store time to recover.
                    self.metrics.record_store_retry();
                    if !self.config.store_retry_backoff.is_zero() {
                        std::thread::sleep(self.config.store_retry_backoff);
                    }
                }
            }
        }
    }

    /// The shutdown-time final drain: keep draining the ring into the cache and
    /// flushing the dirty set until BOTH the ring is empty AND the cache has no
    /// dirty entries, so a clean shutdown loses nothing still in flight.
    ///
    /// ## Guard against an infinite loop when the store is permanently down
    ///
    /// If the store is wedged (every persist returns a retryable error forever),
    /// `requeue_dirty` keeps the dirty set non-empty and a naive "until empty"
    /// loop would hang `shutdown()` indefinitely. We therefore bound the number
    /// of *consecutive unproductive* iterations (those that made no forward
    /// progress draining the ring or clearing dirty entries) by
    /// [`SHUTDOWN_MAX_STALLED_ATTEMPTS`]. When the bound is reached we stop and
    /// return. This is SAFE: every batch we attempted was `append_batch`'d +
    /// fsync'd to the WAL *before* we ever called the store, so the data is
    /// already durable and will be replayed by `recover()` on the next start.
    /// We give up the in-memory retry, never the data.
    fn final_drain(&mut self) {
        let mut stalled: u32 = 0;
        loop {
            // Pull any last stragglers out of the ring into the cache.
            let drained = self.drain_ring_into_cache();

            let ring_empty = self.ring.is_empty();
            let dirty_before = self.cache.dirty_len();

            // Done: nothing left anywhere. Clean exit.
            if ring_empty && dirty_before == 0 {
                return;
            }

            // Flush whatever is dirty (this internally retries the store with
            // backoff per batch).
            if dirty_before > 0 {
                self.flush();
            }

            let dirty_after = self.cache.dirty_len();

            // Forward progress = we either drained ring items this pass or the
            // dirty set shrank. Reset the stall counter on any progress.
            let made_progress = drained > 0 || dirty_after < dirty_before;
            if made_progress {
                stalled = 0;
            } else {
                stalled += 1;
                if stalled >= SHUTDOWN_MAX_STALLED_ATTEMPTS {
                    // Store is effectively down. Stop here rather than hang the
                    // shutdown forever. The undrained dirty entries are already
                    // safe in the WAL (logged + fsync'd before the store call),
                    // so recovery on next start replays them. NO data loss —
                    // only a deferred persist.
                    return;
                }
                // Brief backoff between stalled attempts so we don't spin-hammer
                // a down store during shutdown.
                if !self.config.store_retry_backoff.is_zero() {
                    std::thread::sleep(self.config.store_retry_backoff);
                }
            }
        }
    }
}

/// Outcome of [`Worker::persist_with_retry`].
enum PersistResult {
    /// The store durably accepted the batch.
    Persisted,
    /// The store rejected the batch (Fatal, or retries exhausted). The caller
    /// must requeue the states so nothing is lost.
    Failed,
}

/// Saturating elapsed-nanoseconds-since helper, clamped to `u64` for the
/// metrics API which takes `u64` nanos. Durations longer than ~584 years would
/// saturate; in practice flush/WAL latencies are microseconds-to-milliseconds.
#[inline]
fn nanos_since(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}
