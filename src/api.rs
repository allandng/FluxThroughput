//! The [`FluxEngine`] facade — the **non-blocking boundary** between the hot
//! main loop (e.g. a game tick) and the durable write-behind pipeline.
//! **[impl:api] — full implementation.**
//!
//! `FluxEngine` is the single object the application's main loop talks to. It
//! owns shared (`Arc`) handles to the four pipeline stages — the lock-free
//! [`RingBuffer`], the coalescing [`WriteBehindCache`], the [`Wal`] (durability
//! boundary) and the [`PersistenceStore`] backend — plus the [`Metrics`] and the
//! background flush [`worker`] thread.
//!
//! ## The submit isolation guarantee (this is the whole point)
//!
//! [`FluxEngine::submit`] is **fire-and-forget** and is engineered so that the
//! caller's thread can NEVER be stalled by the slow durable backend:
//!
//! * It performs exactly two synchronizing operations: a single `Relaxed`
//!   `fetch_add` to mint a sequence number, and a single lock-free
//!   [`Queue::try_enqueue`] (a bounded CAS retry on the ring's tail). There is
//!   **NO `Mutex`, NO blocking syscall, NO I/O, NO allocation-on-the-critical
//!   path that can wait, and NO unbounded loop** anywhere on this path.
//! * If the ring is full (the worker / store has fallen behind), `submit`
//!   **sheds load**: it records a drop and returns [`SubmitOutcome::Dropped`]
//!   immediately rather than blocking. The main loop keeps its frame budget.
//! * All locking and all blocking I/O live strictly to the *right* of the ring,
//!   on the [`worker`] thread (cache map lock, WAL fsync, store calls). They are
//!   physically unreachable from `submit`.
//!
//! ## start / recovery ordering
//!
//! [`FluxEngine::start`] (1) validates the config, (2) constructs the ring /
//! cache / WAL / metrics, (3) **recovers the WAL and replays surviving records
//! into the cache as dirty state BEFORE any traffic is accepted**, then (4)
//! spawns the worker. Replaying into the cache (rather than straight into the
//! store) means recovered records re-flow through the normal WAL→store path and
//! are re-persisted with the same retry/requeue guarantees as live writes — see
//! the comment in [`FluxEngine::start`] for why that choice was made.
//!
//! ## Durability boundary (stated precisely — no overclaiming)
//!
//! The WAL is the durability boundary. Writes that have been `append_batch`'d
//! **and fsync'd** survive `SIGKILL` and are replayed exactly by `recover()`.
//! [`FluxEngine::submit`] does NOT fsync — items still in the ring are the
//! explicit at-risk window on a hard kill. Callers needing per-write durability
//! use [`FluxEngine::submit_durable`] (WAL + fsync synchronously) or
//! [`FluxEngine::flush_now`].

use crate::cache::WriteBehindCache;
use crate::config::FluxConfig;
use crate::domain::{
    Cache, Lsn, PersistenceStore, PlayerId, PlayerState, Queue, WalRecord, WriteAheadLog,
    WriteRequest,
};
use crate::error::{FluxError, StoreError};
use crate::lsn::LsnState;
use crate::metrics::{Metrics, MetricsSnapshot};
use crate::ring::RingBuffer;
use crate::wal::Wal;
use crate::worker::{self, WorkerHandle};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Outcome of a fire-and-forget [`FluxEngine::submit`].
pub enum SubmitOutcome {
    /// The write was accepted into the ring and will be flushed by the worker.
    Enqueued,
    /// The ring was full; the write was shed to protect the main loop from
    /// blocking. The caller may retry next tick or rely on a later write
    /// superseding it (last-write-wins coalescing in the cache).
    Dropped,
}

/// The public facade over the whole throughput pipeline.
///
/// Holds shared (`Arc`) handles to the ring, cache, WAL, store and metrics, plus
/// the background [`WorkerHandle`], the cooperative stop flag, and the submit
/// sequence counter. Construct with [`FluxEngine::start`]; tear down with
/// [`FluxEngine::shutdown`].
pub struct FluxEngine {
    /// Lock-free ring shared with the worker (the consumer). The ONLY pipeline
    /// component `submit` ever touches.
    ring: Arc<dyn Queue>,
    /// Write-behind cache shared with the worker. Locking lives here, off the
    /// submit path.
    cache: Arc<dyn Cache>,
    /// Write-ahead log (durability boundary) shared with the worker.
    wal: Arc<dyn WriteAheadLog>,
    /// Durable backend shared with the worker. Held so the store outlives the
    /// worker and is reachable from `flush_now` / `shutdown`.
    store: Arc<dyn PersistenceStore>,
    /// Wait-free metrics shared across the submit path and the worker.
    metrics: Arc<Metrics>,
    /// Handle to the single flush worker thread.
    worker: WorkerHandle,
    /// Cooperative stop flag shared with the worker. Set on `shutdown`.
    stop: Arc<AtomicBool>,
    /// The **single source of truth** for sequence/LSN assignment, shared with
    /// the worker. `submit` mints its diagnostic `seq` here (one Relaxed
    /// `fetch_add` on the hot path — identical cost to a bare counter), and the
    /// synchronous durable paths (`submit_durable` / `flush_now`) mint their WAL
    /// LSNs from the same allocator so every record in the one shared WAL has a
    /// globally unique, monotonic LSN. It also carries the checkpoint floor that
    /// keeps acknowledged durable records from being checkpointed away before the
    /// store has their content. See [`crate::lsn`].
    lsn: Arc<LsnState>,
    /// Immutable engine configuration.
    config: FluxConfig,
}

impl FluxEngine {
    /// Builds and starts the engine against `store`.
    ///
    /// Performs, strictly in order:
    /// 1. **Validate** the config (rejects zero ring/batch/interval/retry).
    /// 2. **Construct** the lock-free ring, write-behind cache, WAL (opening the
    ///    `wal_dir`, honoring `wal_fsync`) and metrics.
    /// 3. **Recover** the WAL *before accepting any traffic*: replay every
    ///    surviving record back into the cache as a fresh dirty
    ///    [`PlayerState`].
    /// 4. **Spawn** the background flush worker, sharing all the `Arc`s.
    ///
    /// Returns a fully-ready engine.
    pub fn start(
        config: FluxConfig,
        store: Arc<dyn PersistenceStore>,
    ) -> std::io::Result<FluxEngine> {
        // ----- (1) validate ---------------------------------------------------
        // A bad config is a programmer error surfaced eagerly as InvalidInput so
        // it never silently corrupts the running pipeline.
        if let Err(msg) = config.validate() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, msg));
        }

        // ----- (2) construct --------------------------------------------------
        // The ring rounds capacity up to a power of two internally. All four
        // stages are behind `Arc<dyn _>` so the worker thread and this facade
        // observe the exact same instances.
        let ring: Arc<dyn Queue> = Arc::new(RingBuffer::new(config.ring_capacity));
        let cache: Arc<dyn Cache> = Arc::new(WriteBehindCache::new());
        let wal: Arc<dyn WriteAheadLog> = Arc::new(Wal::open(&config.wal_dir, config.wal_fsync)?);
        let metrics = Arc::new(Metrics::new());

        // ----- (3) RECOVER before accepting traffic --------------------------
        // recover() returns the records that survived the last (possibly
        // unclean) shutdown, in log order, with a torn/corrupt tail already
        // truncated away. We replay each into the CACHE as a dirty PlayerState.
        //
        // CHOICE: replay-into-cache (NOT a direct store.persist_batch).
        //   * It funnels recovered data back through the *identical* WAL->store
        //     flush path as live traffic, so it inherits last-write-wins
        //     coalescing and the worker's retry/requeue-on-failure guarantee
        //     (no recovered update is lost even if the store is unhealthy at
        //     startup).
        //   * It keeps `start` non-blocking on the slow store: we do not want to
        //     do a synchronous store round-trip (which may be down) before the
        //     engine is even usable.
        //   * Marking them dirty guarantees they are re-persisted exactly once
        //     the worker drains them, and the WAL is checkpointed afterward.
        let recovered = wal.recover()?;
        let mut highest_lsn: Lsn = 0;
        for (i, rec) in recovered.iter().enumerate() {
            highest_lsn = highest_lsn.max(rec.lsn);
            // Re-apply as a normal write. `seq` here is purely diagnostic for the
            // recovered request; last-write-wins in the cache uses `version`. The
            // worker re-stamps a fresh LSN when it later flushes this entry, so
            // this provisional value never reaches the WAL.
            let req = WriteRequest {
                player_id: rec.player_id,
                version: rec.version,
                blob: rec.blob.clone(),
                seq: i as u64,
            };
            cache.apply(&req);
        }

        // The single LSN authority, seeded ONE PAST the highest recovered LSN so
        // every freshly minted seq/LSN is globally monotonic with respect to
        // everything already on disk. This is the one counter the hot-path
        // `submit`, the durable API paths, and the worker all draw from — so a
        // checkpoint can never confuse one appender's LSNs for another's.
        let lsn = Arc::new(LsnState::new(highest_lsn.saturating_add(1)));

        // ----- (4) spawn the worker ------------------------------------------
        // The stop flag is shared: the WorkerHandle keeps its own clone (so
        // `signal_stop` works) and we keep one too, matching the engine contract.
        let stop = Arc::new(AtomicBool::new(false));
        let worker = worker::spawn(worker::WorkerDeps {
            config: config.clone(),
            ring: Arc::clone(&ring),
            cache: Arc::clone(&cache),
            wal: Arc::clone(&wal),
            store: Arc::clone(&store),
            metrics: Arc::clone(&metrics),
            lsn: Arc::clone(&lsn),
            stop: Arc::clone(&stop),
        });

        Ok(FluxEngine {
            ring,
            cache,
            wal,
            store,
            metrics,
            worker,
            stop,
            lsn,
            config,
        })
    }

    /// Fire-and-forget submit. **Lock-free, non-blocking, O(1).**
    ///
    /// This is the hot path the main loop calls every tick. It:
    /// 1. mints a sequence number with a `Relaxed` `fetch_add`, and
    /// 2. attempts a single lock-free [`Queue::try_enqueue`].
    ///
    /// On success it records the submit latency and returns
    /// [`SubmitOutcome::Enqueued`]; on a full ring it records a drop and returns
    /// [`SubmitOutcome::Dropped`] — **it never blocks the caller**.
    ///
    /// ## Isolation guarantee (enforced here, by construction)
    ///
    /// There is deliberately NO `Mutex`, NO blocking syscall, NO file I/O, and NO
    /// unbounded wait below. The only synchronization is one relaxed atomic
    /// increment and one bounded lock-free CAS inside the ring. Every slow
    /// thing — acquiring the cache lock, the WAL fsync, the store round-trip —
    /// happens on the worker thread, on the far side of the ring, and is
    /// physically unreachable from this function. Under back-pressure we shed
    /// load instead of stalling the main loop's frame budget. This is the
    /// single property the whole crate exists to provide.
    pub fn submit(&self, player_id: PlayerId, version: u64, blob: Vec<u8>) -> SubmitOutcome {
        // Measure the full hot-path cost (atomic + enqueue) so the metrics can
        // prove the submit latency stays in the nanosecond/sub-microsecond range.
        let start = Instant::now();

        // (1) One Relaxed `fetch_add` from the shared LSN authority — identical
        // hot-path cost to a bare atomic counter. The seq is a per-write
        // identity/order tag (not a synchronization edge — the ring's own atomics
        // order visibility) AND is globally unique across `submit` /
        // `submit_durable`, which is what lets the worker safely use it to clear
        // a drained durable record's checkpoint floor.
        let seq = self.lsn.alloc();

        let req = WriteRequest {
            player_id,
            version,
            blob,
            seq,
        };

        // (2) one lock-free enqueue attempt; full ring hands the item back.
        match self.ring.try_enqueue(req) {
            Ok(()) => {
                let nanos = start.elapsed().as_nanos() as u64;
                self.metrics.record_submit_enqueued(nanos);
                SubmitOutcome::Enqueued
            }
            Err(_shed) => {
                // Load shedding: drop the (returned) request and report it. No
                // blocking, no retry loop — the caller's tick is protected.
                self.metrics.record_submit_dropped();
                SubmitOutcome::Dropped
            }
        }
    }

    /// Opt-in **blocking** durability for a single write. NOT for the main loop.
    ///
    /// Synchronously builds a [`WalRecord`], appends it to the WAL and (when
    /// `wal_fsync` is configured) fsyncs it to stable storage *before
    /// returning* — so once this call returns `Ok`, the write survives a hard
    /// kill and will be replayed by `recover()`. It then best-effort enqueues
    /// the same write onto the ring so it also flows to the store via the normal
    /// path. The enqueue failing (full ring) is non-fatal: the record is already
    /// durable in the WAL and recovery will re-drive it to the store.
    pub fn submit_durable(
        &self,
        player_id: PlayerId,
        version: u64,
        blob: Vec<u8>,
    ) -> std::io::Result<()> {
        // Mint an LSN from the single shared authority — globally unique and
        // monotonic with respect to every other appender into the one WAL.
        let lsn = self.lsn.alloc();

        // Register this LSN as a PENDING durable record BEFORE we append it. Its
        // WAL frame is about to become its only durable copy; marking it pending
        // pins the worker's checkpoint floor strictly below it, so no concurrent
        // worker flush can checkpoint this acknowledged record out of the WAL
        // before its content is guaranteed to reach the store. The floor is
        // lifted when the worker drains the enqueued copy below (`note_drained`),
        // i.e. once the cache + requeue machinery guarantees store delivery.
        self.lsn.note_pending(lsn);

        let rec = WalRecord {
            lsn,
            player_id,
            version,
            blob: blob.clone(),
        };

        // WRITE-AHEAD + fsync (when configured) happens here and blocks until
        // durable. This is exactly why this method is off the hot path.
        if let Err(e) = self.wal.append_batch(&[rec]) {
            // The append failed: nothing is durable for this LSN, so it can never
            // be a record the store must absorb. Clear its pending floor so it
            // does not pin checkpoints forever, then surface the error.
            self.lsn.note_drained(lsn);
            return Err(e);
        }

        // Best-effort enqueue so the durable write also reaches the store via the
        // worker. We tag the request with `lsn` in `seq` so that when the worker
        // drains it into the cache it clears exactly this record's checkpoint
        // floor. A full ring is acceptable — the WAL already guarantees the write
        // is not lost; its pending floor keeps its frame in the WAL, and recovery
        // re-flows it on the next start.
        let req = WriteRequest {
            player_id,
            version,
            blob,
            seq: lsn,
        };
        let _ = self.ring.try_enqueue(req);

        Ok(())
    }

    /// Returns a point-in-time snapshot of the pipeline metrics.
    pub fn metrics(&self) -> MetricsSnapshot {
        self.metrics.snapshot()
    }

    /// Reads the currently-cached state for `id`, if any.
    ///
    /// Reflects the newest write seen by the cache (including not-yet-persisted
    /// dirty state), so a read-your-writes check can observe a recent `submit`
    /// once the worker has drained it from the ring.
    pub fn cache_get(&self, id: PlayerId) -> Option<PlayerState> {
        self.cache.get(id)
    }

    /// Synchronously drains the ring into the cache, then flushes every dirty
    /// entry through the WAL to the store, returning the number of records
    /// persisted.
    ///
    /// This is a caller-driven, blocking flush (used at shutdown or when an
    /// embedder wants a durability checkpoint). It mirrors the worker's flush
    /// shape but runs to completion: it loops until the cache has no dirty
    /// entries left. Like the worker, it is WRITE-AHEAD (WAL + fsync before the
    /// store) and re-queues the dirty set on store failure so no update is lost.
    pub fn flush_now(&self) -> Result<usize, FluxError> {
        // 1) Drain everything currently in the ring into the cache so it is
        //    eligible for flushing (coalesced under last-write-wins).
        self.drain_ring_into_cache();

        // 2) Flush dirty entries in batches until none remain.
        let mut total_persisted = 0usize;
        while self.cache.dirty_len() > 0 {
            // Pull more requests the worker may have raced in, then take a batch.
            self.drain_ring_into_cache();

            let states = self.cache.take_dirty(self.config.flush_max_batch);
            if states.is_empty() {
                break;
            }

            let persisted = self.flush_states(states)?;
            total_persisted += persisted;
        }

        Ok(total_persisted)
    }

    /// Number of writes not yet persisted: `ring.len() + cache.dirty_len()`.
    ///
    /// An over-approximation under concurrency (a write may be counted in both
    /// as it moves), which is the safe direction for a "still in flight" gauge.
    pub fn pending(&self) -> usize {
        self.ring.len() + self.cache.dirty_len()
    }

    /// Gracefully stops the engine: signal the worker to stop, let it perform
    /// its final drain+flush, join the thread, then fsync the WAL.
    ///
    /// Consumes `self`. After this returns `Ok(())` the worker thread is
    /// reclaimed and all writes the worker managed to flush are durable.
    pub fn shutdown(self) -> Result<(), FluxError> {
        // Signal cooperative shutdown via BOTH the shared flag and the handle
        // (the handle's `signal_stop` sets the same flag; doing both is harmless
        // and makes the intent explicit). The worker observes the flag, performs
        // a final drain of the ring + flush of the cache until both are empty,
        // checkpoints the WAL, and returns.
        self.stop.store(true, Ordering::SeqCst);
        self.worker.signal_stop();

        // Join the worker thread; a panic in the worker surfaces as an I/O error.
        self.worker
            .join()
            .map_err(|_| FluxError::Io(std::io::Error::other("flush worker thread panicked")))?;

        // Final fsync so anything the worker buffered hits stable storage.
        self.wal.flush().map_err(FluxError::Io)?;
        Ok(())
    }

    /// Borrows the immutable engine configuration.
    pub fn config(&self) -> &FluxConfig {
        &self.config
    }

    // ------------------------------------------------------------------------
    // Private helpers — shared by `flush_now`. These duplicate the *shape* of
    // the worker's flush (intentionally; see the SPEC) but are driven by the
    // caller's thread for the synchronous flush path.
    // ------------------------------------------------------------------------

    /// Drains all currently-queued requests from the ring into the cache,
    /// coalescing them under last-write-wins. Bounded per call by the configured
    /// batch size, looped until the ring is empty.
    fn drain_ring_into_cache(&self) {
        let mut buf: Vec<WriteRequest> = Vec::with_capacity(self.config.flush_max_batch);
        loop {
            buf.clear();
            let moved = self.ring.drain_into(&mut buf, self.config.flush_max_batch);
            if moved == 0 {
                break;
            }
            for req in &buf {
                // Mirror the worker: a drained `submit_durable` request's content
                // is now guaranteed to reach the store (cache + requeue), so lift
                // its checkpoint floor. A plain `submit` seq is a harmless no-op.
                self.lsn.note_drained(req.seq);
                self.cache.apply(req);
            }
            // If we pulled a partial batch, the ring is (momentarily) empty.
            if moved < self.config.flush_max_batch {
                break;
            }
        }
    }

    /// Write-ahead + persist a single batch of dirty states with the configured
    /// retry/backoff policy, then checkpoint the WAL on success. On exhausted
    /// retries or a fatal store error the states are re-queued as dirty so no
    /// update is lost, and the error is surfaced to the caller.
    ///
    /// Returns the number of states persisted (the batch length on success).
    fn flush_states(&self, states: Vec<PlayerState>) -> Result<usize, FluxError> {
        let flush_start = Instant::now();
        let batch_len = states.len();

        // ----- WRITE-AHEAD: WAL (+fsync) BEFORE the store --------------------
        // Assign each state a monotonic LSN from the single shared authority, so
        // these caller-driven records are globally unique with respect to the
        // worker and `submit_durable` writing into the same WAL.
        let first_lsn = self.lsn.alloc_block(batch_len as u64);
        let mut records = Vec::with_capacity(batch_len);
        for (i, st) in states.iter().enumerate() {
            records.push(WalRecord {
                lsn: first_lsn + i as u64,
                player_id: st.player_id,
                version: st.version,
                blob: st.blob.clone(),
            });
        }
        let wal_start = Instant::now();
        let last_lsn = match self.wal.append_batch(&records) {
            Ok(lsn) => lsn,
            Err(e) => {
                // WAL failed: nothing is durable, so put the states back as dirty
                // (no data loss) and surface the error.
                self.cache.requeue_dirty(states);
                return Err(FluxError::Io(e));
            }
        };
        self.metrics
            .record_wal_append(records.len(), wal_start.elapsed().as_nanos() as u64);

        // ----- persist to the store with retry/backoff -----------------------
        let mut attempt: u32 = 0;
        loop {
            match self.store.persist_batch(&states) {
                Ok(()) => {
                    // Store durable: checkpoint the WAL up to this batch's LSN,
                    // but clamp to a watermark that is safe to discard — strictly
                    // below any still-pending `submit_durable` record whose
                    // content is not yet guaranteed in the store (its WAL frame is
                    // still its only durable copy). If nothing is safe to discard
                    // this round we simply skip the checkpoint; the records stay
                    // in the WAL and are harmlessly re-applied on recovery or
                    // checkpointed by a later flush — never lost.
                    if let Some(safe_lsn) = self.lsn.checkpoint_watermark(last_lsn) {
                        self.wal.checkpoint(safe_lsn).map_err(FluxError::Io)?;
                    }
                    self.metrics
                        .record_flush(batch_len, flush_start.elapsed().as_nanos() as u64);
                    return Ok(batch_len);
                }
                Err(e) => {
                    attempt += 1;
                    let retryable = e.is_retryable();
                    // Stop when the error is fatal OR we've used our retry budget.
                    if !retryable || attempt >= self.config.store_retry_limit {
                        // Exhausted/fatal: re-queue dirty (NO data loss) + record.
                        self.cache.requeue_dirty(states);
                        self.metrics.record_store_error();
                        return Err(FluxError::Store(e));
                    }
                    // Retryable and budget remains: back off and try again.
                    self.metrics.record_store_retry();
                    self.backoff_sleep(attempt);
                }
            }
        }
    }

    /// Sleeps the configured retry backoff before another store attempt.
    ///
    /// Kept in one place so the retry policy is identical wherever it is used.
    /// `attempt` is accepted for a possible future scaled/linear backoff; the
    /// current policy is the flat configured backoff to match the worker.
    fn backoff_sleep(&self, _attempt: u32) {
        if !self.config.store_retry_backoff.is_zero() {
            std::thread::sleep(self.config.store_retry_backoff);
        }
    }
}

#[allow(dead_code)]
/// Compile-time witness that the engine is `Send + Sync` (so it can be shared
/// across threads / behind an `Arc` by embedders). Purely a static assertion.
fn _assert_engine_thread_safe() {
    fn is_send_sync<T: Send + Sync>() {}
    is_send_sync::<FluxEngine>();
    // Also assert the store error stays usable through the FluxError boundary.
    fn _uses(_e: StoreError) {}
}
