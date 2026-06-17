# FluxThroughput — Authoritative Contract (SPEC)

This is the **single canonical reference** for every Phase-2 implementation
agent. The public signatures below are **binding**: implement the bodies, but do
not change any signature, trait bound, or module path. The architect-owned files
(`domain.rs`, `config.rs`, `error.rs`, `metrics.rs`, `lib.rs`, `Cargo.toml`) are
already fully implemented — treat them as fixed.

- Crate: `flux_throughput`, Rust **edition 2021**, **stable**, **std-only, ZERO
  external dependencies** (no `crossbeam`, `tokio`, `rand`, or `crc` crate).
- `unsafe` is permitted **only in `src/ring.rs`**, and every `unsafe` block must
  be prefaced with a `// SAFETY:` comment. All other files are safe code.
- Heavy docs: `//!` at module top, `///` on every public item. 4-space indent,
  rustfmt-clean, no avoidable warnings.

---

## File ownership

| File | Owner | Status |
|------|-------|--------|
| `Cargo.toml` | architect | done |
| `src/lib.rs` | architect | done (module decls + re-exports) |
| `src/domain.rs` | architect | done (types, traits, WAL frame, CRC32) |
| `src/config.rs` | architect | done (`FluxConfig`) |
| `src/error.rs` | architect | done (error enums) |
| `src/metrics.rs` | architect | done (lock-free `Metrics`) |
| `src/lsn.rs` | impl:worker/api shared | done (LSN authority + checkpoint floor) |
| `src/ring.rs` | impl:ring | STUB → implement |
| `src/wal.rs` | impl:wal | STUB → implement |
| `src/store.rs` | impl:store | STUB → implement |
| `src/cache.rs` | impl:cache | STUB → implement |
| `src/worker.rs` | impl:worker | STUB → implement |
| `src/api.rs` | impl:api | STUB (wired) → implement |
| `src/bin/demo.rs` | impl:api | STUB → implement |
| `src/bin/crash_child.rs` | impl:api | STUB → implement |

Phase-2 agents **overwrite** their stub file with the full implementation,
keeping the public signatures identical.

---

## `src/domain.rs` (architect — DONE)

```rust
pub type PlayerId = u64;
pub type Lsn = u64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlayerState { pub player_id: PlayerId, pub version: u64, pub blob: Vec<u8> }

#[derive(Clone, Debug)]
pub struct WriteRequest { pub player_id: PlayerId, pub version: u64, pub blob: Vec<u8>, pub seq: u64 }
impl WriteRequest { pub fn to_state(&self) -> PlayerState; }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WalRecord { pub lsn: Lsn, pub player_id: PlayerId, pub version: u64, pub blob: Vec<u8> }
impl WalRecord {
    pub fn encode(&self) -> Vec<u8>;
    pub fn decode(buf: &[u8]) -> Result<(WalRecord, usize), crate::error::DecodeError>;
}

// Shared, single-source-of-truth checksum used by domain.rs AND wal.rs.
pub(crate) fn crc32(bytes: &[u8]) -> u32;

pub trait Queue: Send + Sync {
    fn try_enqueue(&self, req: WriteRequest) -> Result<(), WriteRequest>;
    fn try_dequeue(&self) -> Option<WriteRequest>;
    fn drain_into(&self, out: &mut Vec<WriteRequest>, max: usize) -> usize;
    fn len(&self) -> usize;
    fn capacity(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
}

pub trait WriteAheadLog: Send + Sync {
    fn append_batch(&self, records: &[WalRecord]) -> std::io::Result<Lsn>;
    fn checkpoint(&self, up_to_lsn: Lsn) -> std::io::Result<()>;
    fn recover(&self) -> std::io::Result<Vec<WalRecord>>;
    fn flush(&self) -> std::io::Result<()>;
}

pub trait PersistenceStore: Send + Sync {
    fn persist_batch(&self, states: &[PlayerState]) -> Result<(), crate::error::StoreError>;
    fn load(&self, id: PlayerId) -> Option<PlayerState>;
    fn count(&self) -> usize;
}

pub trait Cache: Send + Sync {
    fn apply(&self, req: &WriteRequest);
    fn take_dirty(&self, max: usize) -> Vec<PlayerState>;
    fn requeue_dirty(&self, states: Vec<PlayerState>);
    fn get(&self, id: PlayerId) -> Option<PlayerState>;
    fn dirty_len(&self) -> usize;
    fn len(&self) -> usize;
}
```

### WAL frame wire format (all integers little-endian)

```text
offset  size       field
------  ----       -----------------------------------------------------------
  0      4         magic       = 0x464C5852  ("FLXR")
  4      4         frame_len   = total frame length incl. crc field
  8      8         lsn
 16      8         player_id
 24      8         version
 32      4         blob_len
 36   blob_len     blob bytes
  ..     4         crc32       = IEEE CRC32 of all bytes preceding this field
```

- Fixed header = 36 bytes; `frame_len == 36 + blob_len + 4`.
- CRC covers `frame_len - 4` bytes (everything before the CRC field).
- `decode` errors: `Truncated` (buffer shorter than a complete/consistent
  frame, including inconsistent length fields), `BadMagic` (magic mismatch),
  `BadCrc` (checksum mismatch).
- **CRC**: IEEE 802.3 / ISO-HDLC, reflected polynomial `0xEDB88320`, init/final
  XOR `0xFFFFFFFF`. Verified against the standard check value
  `crc32("123456789") == 0xCBF43926` (matches zlib/gzip/PNG). `wal.rs` MUST use
  `crate::domain::crc32` — do not reimplement.

---

## `src/config.rs` (architect — DONE)

```rust
#[derive(Clone, Debug)]
pub struct FluxConfig {
    pub ring_capacity: usize,                  // rounds UP to next power of two
    pub flush_max_batch: usize,                // VOLUME threshold
    pub flush_interval: std::time::Duration,   // TIME threshold
    pub worker_idle_sleep: std::time::Duration,
    pub wal_dir: std::path::PathBuf,
    pub wal_fsync: bool,
    pub store_retry_limit: u32,
    pub store_retry_backoff: std::time::Duration,
}
impl Default for FluxConfig; // ring 1<<16, batch 4096, interval 50ms, idle 1ms,
                             // fsync true, retry 5, backoff 2ms, wal_dir ./flux-wal
impl FluxConfig {
    pub fn with_wal_dir(self, dir: impl Into<std::path::PathBuf>) -> Self;
    pub fn validate(&self) -> Result<(), String>;
}
```

`validate` rejects: `ring_capacity == 0`, `flush_max_batch == 0`,
`flush_interval == 0`, `store_retry_limit == 0`.

---

## `src/error.rs` (architect — DONE)

```rust
#[derive(Debug, Clone)]
pub enum StoreError { Unavailable, Transient(String), Fatal(String) } // + Display + Error
impl StoreError { pub fn is_retryable(&self) -> bool; } // Unavailable & Transient = true, Fatal = false

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError { Truncated, BadCrc, BadMagic } // + Display + Error

#[derive(Debug)]
pub enum FluxError { Io(std::io::Error), Store(StoreError) }
// + Display + Error + From<std::io::Error> + From<StoreError>
```

---

## `src/metrics.rs` (architect — DONE)

All counters `AtomicU64`, wait-free, **no `Mutex`**. Fetch-max via CAS loop.

```rust
pub struct Metrics { /* private atomics */ }
impl Metrics {
    pub fn new() -> Self;
    pub fn record_submit_enqueued(&self, submit_nanos: u64);
    pub fn record_submit_dropped(&self);
    pub fn record_flush(&self, batch_len: usize, flush_nanos: u64);
    pub fn record_wal_append(&self, records: usize, nanos: u64);
    pub fn record_store_error(&self);
    pub fn record_store_retry(&self);
    pub fn snapshot(&self) -> MetricsSnapshot;
}
impl Default for Metrics;

#[derive(Clone, Debug, Default)]
pub struct MetricsSnapshot {
    pub submits_enqueued: u64, pub submits_dropped: u64,
    pub max_submit_nanos: u64, pub sum_submit_nanos: u64,
    pub flushes: u64, pub records_flushed: u64, pub max_flush_nanos: u64,
    pub wal_appends: u64, pub wal_records: u64,
    pub store_errors: u64, pub store_retries: u64,
}
impl MetricsSnapshot { pub fn avg_submit_nanos(&self) -> f64; } // sum/count, 0.0 when count==0
```

---

## `src/lsn.rs` (impl:worker/api shared) — the LSN authority

The **single source of truth** for WAL log-sequence-number assignment. One
`LsnState`, held behind an `Arc`, is shared by *every* appender into the one WAL:
`submit` (diagnostic seq), `submit_durable`, `flush_now`, AND the background
worker. This guarantees every record in the single WAL carries a globally-unique,
monotonically-increasing LSN regardless of which path wrote it.

```rust
pub struct LsnState { /* AtomicU64 next + Mutex<BTreeSet<Lsn>> pending */ }
impl LsnState {
    pub fn new(start: Lsn) -> Self;                       // first LSN handed out is `start`
    pub fn alloc(&self) -> Lsn;                           // reserve one LSN (wait-free fetch_add)
    pub fn alloc_block(&self, count: u64) -> Lsn;         // reserve [first, first+count); returns first
    pub fn note_pending(&self, lsn: Lsn);                 // pin a durable LSN: no checkpoint may discard it yet
    pub fn note_drained(&self, lsn: Lsn);                 // clear it once its content is guaranteed to reach the store
    pub fn checkpoint_watermark(&self, proposed: Lsn) -> Option<Lsn>;
        // clamp a proposed checkpoint STRICTLY BELOW the lowest pending durable LSN;
        // Some(safe<=proposed) when a checkpoint is safe, None when nothing is safe to discard this round.
}
```

- `new(start)` is seeded by `FluxEngine::start` to **one past the highest LSN
  recovered from disk**, so freshly minted LSNs are globally monotonic with
  respect to everything already in the WAL.
- `alloc` / `alloc_block` are the only LSN sources: `submit`/`submit_durable`
  call `alloc`; the worker flush and `flush_now` call `alloc_block`.
- **Pending-durable checkpoint floor.** A durable append calls `note_pending(lsn)`
  *before* the WAL append; the worker / `flush_now` call `note_drained(seq)` the
  instant a request lands in the cache (where the requeue machinery guarantees it
  reaches the store). `checkpoint_watermark(proposed)` then clamps any proposed
  checkpoint strictly **below** the lowest still-pending durable LSN, so a
  checkpoint can never discard a record whose content the store has not yet
  absorbed.
- Safe code only (no `unsafe`).

### Durability case study (solved dual-LSN bug)

The original design ran **two independent LSN counters** into the **one shared
WAL**: the background worker minted LSNs from its own private `next_lsn` (starting
at `1`), while `submit_durable()` / `flush_now()` minted from the engine's `seq`
(starting at `0`). Because `Wal::checkpoint(up_to_lsn)` discards records by a raw
`lsn <= up_to_lsn` comparison across the **whole file** and cannot tell the two
counters apart, after the worker checkpointed its own batch it could physically
delete a low-LSN, already-fsync'd `submit_durable` frame that the store had
**never** absorbed. The durable record's best-effort ring copy can be load-shed as
`Dropped`, leaving its WAL frame as its *only* durable copy — so this was a silent
loss of an fsync-acknowledged write on `SIGKILL`.

**Fix:** a single `Arc`-shared `LsnState` allocator, seeded one past the highest
recovered LSN, used by `submit` / `submit_durable` / `flush_now` AND the worker,
so every record in the single WAL carries a globally-unique, monotonic LSN. PLUS
the pending-durable checkpoint floor: `note_pending(lsn)` before a durable append,
`note_drained(seq)` when the request lands in the cache, and
`checkpoint_watermark(proposed)` clamping every checkpoint strictly below the
lowest pending durable LSN. Wired at `worker.rs` (`flush_batch` checkpoint path)
and `api.rs` (`flush_now` / `submit_durable`). Guarded by **5 `lsn::tests`** unit
tests plus **`wal::tests::checkpoint_discards_acked_keeps_live`**.

---

## `src/ring.rs` (impl:ring)

```rust
pub struct RingBuffer { /* ... */ }
impl RingBuffer { pub fn new(capacity: usize) -> Self; } // capacity rounded UP to power of two, min 1
impl crate::domain::Queue for RingBuffer { /* all methods */ }
```

- Lock-free **bounded MPMC** ring (Vyukov sequence-stamped or equivalent).
- `try_enqueue` is wait-free; full ring returns `Err(req)` (hands item back).
- `unsafe` allowed here only, each block prefaced with `// SAFETY:`.

---

## `src/wal.rs` (impl:wal)

```rust
pub struct Wal { /* ... */ }
impl Wal { pub fn open(dir: &std::path::Path, fsync: bool) -> std::io::Result<Self>; }
impl crate::domain::WriteAheadLog for Wal { /* all methods */ }
```

- Append-only frames via `WalRecord::encode`; use `crate::domain::crc32`.
- `append_batch` fsyncs before returning when `fsync == true`; returns highest LSN.
- `recover` replays, tolerates a torn/corrupt **tail** (`Truncated`/`BadCrc`):
  truncate file to last good offset, return records read so far. Zero partial
  corruption. Safe code only.
- `checkpoint(up_to_lsn)` rotates/truncates records `<= up_to_lsn` crash-safely.

---

## `src/store.rs` (impl:store)

```rust
pub struct MockDb { /* ... */ }
impl MockDb {
    pub fn new() -> Self;                                   // healthy, empty, no injected faults
    // ---- fault-injection knobs (all &self, interior mutability) ----
    pub fn set_healthy(&self, healthy: bool);              // false => every persist returns retryable Unavailable
    pub fn fail_next_batches(&self, n: u64);               // fail the next n persists with retryable Transient (overwrites, not adds)
    pub fn set_latency(&self, latency: std::time::Duration); // sleep at the start of every persist (ZERO disables)
    pub fn set_fail_every(&self, n: u64);                  // deterministically fail every n-th attempt (0 disables)
    // ---- test observers ----
    pub fn persisted_batches(&self) -> u64;               // count of batches successfully upserted
    pub fn snapshot(&self) -> Vec<PlayerState>;           // point-in-time copy of every persisted state (arbitrary order)
}
impl Default for MockDb;
impl crate::domain::PersistenceStore for MockDb { /* all methods */ }
```

- In-memory map with fault injection. Unhealthy / injected failures return
  retryable `StoreError::Unavailable` (or `Transient`). All locking lives here,
  off the submit path. Safe code only.

---

## `src/cache.rs` (impl:cache)

```rust
pub struct WriteBehindCache { /* ... */ }
impl WriteBehindCache { pub fn new() -> Self; }
impl Default for WriteBehindCache;
impl crate::domain::Cache for WriteBehindCache { /* all methods */ }
```

- `apply`: upsert under **last-write-wins by `version`** (stale/equal ignored),
  set dirty flag.
- `take_dirty(max)`: atomically remove up to `max` dirty entries.
- `requeue_dirty`: re-mark dirty after failed persist, merge under
  last-write-wins so a newer concurrent update is never clobbered. No data loss.
- All locking here, off the submit path. Safe code only.

---

## `src/worker.rs` (impl:worker)

```rust
use std::sync::{Arc, atomic::AtomicBool};
pub struct WorkerHandle { /* JoinHandle<()> + Arc<AtomicBool> stop */ }
impl WorkerHandle {
    pub fn signal_stop(&self);
    pub fn join(self) -> std::thread::Result<()>;
}
/// All collaborators are bundled into one struct (keeps `spawn` to a single
/// argument and the wiring self-documenting). Constructed by `FluxEngine::start`.
pub struct WorkerDeps {
    pub config: crate::config::FluxConfig,
    pub ring: Arc<dyn crate::domain::Queue>,
    pub cache: Arc<dyn crate::domain::Cache>,
    pub wal: Arc<dyn crate::domain::WriteAheadLog>,
    pub store: Arc<dyn crate::domain::PersistenceStore>,
    pub metrics: Arc<crate::metrics::Metrics>,
    pub lsn: Arc<crate::lsn::LsnState>,    // shared LSN authority (see src/lsn.rs)
    pub stop: Arc<AtomicBool>,             // cooperative stop flag, cloned into the handle
}
pub fn spawn(deps: WorkerDeps) -> WorkerHandle;
```

### Worker flush algorithm (binding)

```text
while !stop:
    drain ring -> cache.apply            (coalesce via last-write-wins)
    if cache.dirty_len() >= flush_max_batch OR flush_interval elapsed:
        flush()
    else:
        park worker_idle_sleep

flush():
    states  = cache.take_dirty(flush_max_batch)
    records = states -> WalRecord (LSNs from the shared LsnState authority via alloc_block)
    wal.append_batch(records)            // WRITE-AHEAD: WAL (+fsync) BEFORE store
    persist with up to store_retry_limit retries + store_retry_backoff:
        retryable error  -> record_store_retry, sleep backoff, retry
        exhausted/Fatal  -> cache.requeue_dirty(states) + record_store_error  (NO data loss)
    store success        -> wal.checkpoint(lsn.checkpoint_watermark(last_lsn)) when Some
                            (clamped strictly below the lowest pending durable LSN; None => skip this round)
    record_flush(states.len(), elapsed)

on stop: final drain + flush until ring empty AND dirty empty, then return.
```

The worker thread is the ONLY place blocking I/O and locking are allowed. It
never touches the caller thread. Safe code only.

---

## `src/api.rs` (impl:api)

```rust
pub enum SubmitOutcome { Enqueued, Dropped }

pub struct FluxEngine { /* Arc<dyn Queue>, Arc<dyn Cache>, Arc<dyn WriteAheadLog>,
                          Arc<dyn PersistenceStore>, Arc<Metrics>, WorkerHandle,
                          Arc<AtomicBool> stop, Arc<crate::lsn::LsnState> lsn,
                          FluxConfig */ }
impl FluxEngine {
    // Validate config -> construct -> recover+replay WAL -> spawn worker. Ready engine.
    pub fn start(config: crate::config::FluxConfig,
                 store: std::sync::Arc<dyn crate::domain::PersistenceStore>)
                 -> std::io::Result<FluxEngine>;
    // Hot path: fire-and-forget, lock-free, non-blocking. Sheds load (Dropped) under back-pressure.
    pub fn submit(&self, player_id: crate::domain::PlayerId, version: u64, blob: Vec<u8>) -> SubmitOutcome;
    // Opt-in BLOCKING per-write durability: WAL append (+fsync) before returning Ok; survives SIGKILL. NOT for the main loop.
    pub fn submit_durable(&self, player_id: crate::domain::PlayerId, version: u64, blob: Vec<u8>) -> std::io::Result<()>;
    // Point-in-time snapshot of pipeline metrics.
    pub fn metrics(&self) -> crate::metrics::MetricsSnapshot;
    // Read the newest cached state for `id` (incl. not-yet-persisted dirty state), if any.
    pub fn cache_get(&self, id: crate::domain::PlayerId) -> Option<crate::domain::PlayerState>;
    // Caller-driven blocking flush: drain ring -> cache, then WAL+store every dirty entry until none remain; returns count persisted.
    pub fn flush_now(&self) -> Result<usize, crate::error::FluxError>;
    pub fn pending(&self) -> usize;                                   // ring.len() + cache.dirty_len()
    // Graceful teardown: signal stop, final drain+flush, join worker, fsync WAL. Consumes self.
    pub fn shutdown(self) -> Result<(), crate::error::FluxError>;
    // Borrow the immutable engine configuration.
    pub fn config(&self) -> &crate::config::FluxConfig;
}
```

The submit `seq` is now minted from the shared `LsnState` (still one `Relaxed`
`fetch_add` — identical hot-path cost to a bare counter), so it is globally
unique across `submit` / `submit_durable` and the worker can use it to clear a
drained durable record's checkpoint floor (`note_drained`). `submit_durable`
calls `note_pending(lsn)` before its WAL append; `flush_now`'s checkpoint is
clamped via `lsn.checkpoint_watermark(last_lsn)`.

`start` order: (1) validate config; (2) construct ring/cache/wal/metrics;
(3) **recover WAL and replay survivors into cache BEFORE accepting traffic**,
seeding `LsnState` one past the highest recovered LSN; (4) spawn worker (passing
the shared `lsn`). `submit` is fire-and-forget: `Relaxed` `fetch_add` seq (from
`LsnState`), build `WriteRequest`, single lock-free `try_enqueue`; full ring ->
record drop, return `Dropped`. NO `Mutex` / blocking / unbounded wait on this
path.

### Binaries (impl:api)

- `src/bin/demo.rs` — simulated game-loop demonstration (tight non-blocking
  submit loop over a `MockDb`, show load shedding, print metrics, shut down).
- `src/bin/crash_child.rs` — SIGKILL victim: open engine on a WAL dir from argv,
  perform durable writes, print ready marker, wait to be killed. Parent reopens
  WAL and asserts `recover()` replays exactly the durable records.

---

## Durability boundary (state precisely — do NOT overclaim)

- **The WAL is the durability boundary.** Records that have been
  `append_batch`'d **and fsync'd** survive SIGKILL and are replayed exactly by
  `recover()`, with CRC32 validation. A torn/partially-written **tail** record
  is detected (`BadCrc`/`Truncated`) and cleanly discarded with the file
  truncated to the last good offset ⇒ ZERO partial-state corruption.
- **`submit()` is fire-and-forget.** Items still in the ring (not yet WAL'd) are
  the **explicit at-risk window** on a hard kill. Callers needing per-write
  durability use `submit_durable()` or `flush_now()`.
- **Checkpoint floor (no acknowledged-write loss).** `Wal::checkpoint(up_to_lsn)`
  discards records by raw `lsn <= up_to_lsn` across the whole file. The shared
  `LsnState` pins every still-pending `submit_durable` LSN (`note_pending` before
  append, `note_drained` once the request lands in the cache) and
  `checkpoint_watermark(proposed)` clamps every checkpoint **strictly below** the
  lowest pending durable LSN — so a checkpoint can never delete an fsync'd record
  whose content the store has not yet absorbed. See `src/lsn.rs`.
- **`fsync` granularity caveat (state honestly).** The WAL uses
  `File::sync_data()` (fdatasync-equivalent). This guarantees survival of a
  **process kill (SIGKILL)** — the scenario `tests/crash_resilience.rs`
  exercises — but does **NOT** guarantee **power-loss / drive-cache** survival on
  macOS, which requires `fcntl(F_FULLFSYNC)`. `F_FULLFSYNC` is **not** wired up;
  the swap point is `Wal::append_batch` in `src/wal.rs`. Do **not** claim
  power-loss durability.

## Main-loop isolation guarantee

- `submit()` contains NO `Mutex`, NO blocking syscall, NO unbounded wait: a
  `Relaxed` atomic seq `fetch_add` plus a lock-free ring CAS enqueue. Under
  back-pressure it sheds load (returns `Dropped`) rather than blocking.
- All locking (cache map, store map) lives OFF the submit path, on the
  worker/DB side.

---

## Re-exports (from `lib.rs`)

The crate root re-exports the canonical embedder surface:

`FluxEngine`, `SubmitOutcome`, `FluxConfig`, `Metrics`, `MetricsSnapshot`,
`Lsn`, `PlayerId`, `PlayerState`, `WalRecord`, `WriteRequest`,
`Cache`, `PersistenceStore`, `Queue`, `WriteAheadLog`,
`DecodeError`, `FluxError`, `StoreError`, **`MockDb`** (the built-in
fault-injection store, re-exported at root so test/fault embedders need not reach
into the `store` submodule).

`lsn::LsnState` remains `pub` (reachable as `flux_throughput::lsn::LsnState`) but
is **intentionally NOT** re-exported at the root: it is an internal durability
mechanism the engine wires automatically.

### `prelude`

A single glob — `use flux_throughput::prelude::*;` — covering all three embedder
personas in one import:

- **(a) game-loop integrator:** `FluxEngine`, `SubmitOutcome`, `FluxConfig`,
  `Metrics`, `MetricsSnapshot`.
- **(b) custom `PersistenceStore` implementor:** `PlayerId`, `PlayerState`,
  `PersistenceStore`, `StoreError`, plus the other domain value types/traits
  (`Lsn`, `WalRecord`, `WriteRequest`, `Cache`, `Queue`, `WriteAheadLog`) and
  `DecodeError`, `FluxError`.
- **(c) test / fault-injection embedder:** `MockDb`.

`LsnState` is deliberately omitted from the prelude (internal mechanism).
```
