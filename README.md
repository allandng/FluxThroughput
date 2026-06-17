# FluxThroughput

> A lock-free, crash-safe, write-behind persistence pipeline that isolates a
> latency-sensitive hot loop (e.g. a game tick) from a slow durable backend.
>
> **std-only. Zero external dependencies.** Stable Rust, edition 2021.

The authoritative API contract lives in [`SPEC.md`](./SPEC.md); this README is
the operator-facing overview.

---

## The one guarantee

`submit()` never blocks the main loop. It is a relaxed atomic `fetch_add` plus a
single lock-free ring enqueue — no mutex, no syscall, no I/O, no unbounded wait.
Under back-pressure it sheds load (returns `Dropped`) instead of stalling the
caller. Everything slow — cache locking, WAL fsync, the store round-trip — lives
strictly to the right of the ring, on a single background worker thread, and is
physically unreachable from `submit()`.

## Architecture

```text
 main loop                              background worker thread
 ─────────                              ────────────────────────
 submit(id, ver, blob)
   │ Relaxed fetch_add seq
   │ build WriteRequest
   ▼
 ┌────────────┐  drain   ┌──────────────┐  take_dirty  ┌──────────┐  persist  ┌────────────┐
 │ RingBuffer │ ───────▶ │ WriteBehind  │ ───────────▶ │   Wal    │ ────────▶ │ Persistence│
 │ lock-free  │          │   Cache      │   records    │ durability│           │   Store    │
 │ MPMC ring  │          │ coalesce LWW │              │ boundary │  (after   │ slow backend│
 └────────────┘          └──────────────┘   append +   └────┬─────┘   WAL)     └────────────┘
   ▲  full → Dropped                         fsync FIRST     │ then checkpoint
   │  (load shedding)                                        ▼
 caller never blocks
```

Producers only ever touch the lock-free **RingBuffer**. Everything to the right
of it runs on the single background **worker** thread — the only place blocking
I/O and locking are allowed. Bursts are coalesced in the **cache** under
last-write-wins, written ahead to the **WAL**, and finally persisted to the
**store** with retry/requeue so no in-memory update is ever lost.

| Stage | File | Role |
|-------|------|------|
| `RingBuffer` | `src/ring.rs` | Lock-free bounded MPMC queue (Vyukov sequence-stamped). The only component the hot loop touches. `unsafe` is confined here, each block justified with a `// SAFETY:` note. |
| `WriteBehindCache` | `src/cache.rs` | Coalesces bursts by last-write-wins per `player_id`, tracks a dirty side-index. A burst of N writes to one player collapses to one store row. |
| `Wal` | `src/wal.rs` | Append-only, CRC32-protected, crash-recoverable write-ahead log. **The durability boundary.** Tolerates a torn tail; checkpoint rotates via temp-file + atomic rename. |
| `MockDb` (`PersistenceStore`) | `src/store.rs` | In-memory durable backend with fault injection (unavailable / fail-next-N / fail-every-N / latency). All DB-side locking lives here, off the submit path. |
| `worker` | `src/worker.rs` | Single background thread: drain → coalesce → WAL → store → checkpoint, with bounded retry + lossless requeue. |
| `FluxEngine` | `src/api.rs` | The public facade the application's main loop talks to. |

The architect-owned core — `domain.rs` (types, traits, WAL wire format, CRC32),
`config.rs`, `error.rs`, `metrics.rs`, `lib.rs` — is shared by every stage.
`lsn.rs` holds the single shared LSN authority + checkpoint floor (see the case
study below) that the durable API paths and the worker both draw from.

## Write-behind flush algorithm (time / volume thresholds)

The worker loops:

```text
while !stop:
    drain ring -> cache.apply                 (coalesce via last-write-wins)
    if cache.dirty_len() >= flush_max_batch    (VOLUME threshold) OR
       flush_interval elapsed                  (TIME threshold):
        flush()
    else if nothing was drained this pass:
        park worker_idle_sleep                 (near-zero idle CPU)

flush():
    states  = cache.take_dirty(flush_max_batch)
    records = states -> WalRecord (worker stamps monotonic LSNs)
    wal.append_batch(records)                  // WRITE-AHEAD: WAL (+fsync) BEFORE store
    persist with up to store_retry_limit retries + store_retry_backoff:
        retryable error  -> record_store_retry, sleep backoff, retry
        exhausted/Fatal  -> cache.requeue_dirty(states) + record_store_error  (NO data loss)
    store success        -> wal.checkpoint(last_lsn) + record_flush

on stop: final drain + flush until ring empty AND dirty empty, then return
         (bounded so a permanently-down store cannot hang shutdown — the data is
          already durable in the WAL and replays on the next start).
```

- **VOLUME threshold** (`flush_max_batch`): flush as soon as enough dirty
  entries accumulate, regardless of elapsed time — keeps batches large and the
  store efficient under load.
- **TIME threshold** (`flush_interval`): flush at least this often even if the
  volume threshold is never reached — bounds write latency when traffic is light.

Coalescing is the throughput multiplier: many `apply` calls for the same
`player_id` collapse into a single dirty entry (highest `version` wins), so a
player hammered between two flushes produces exactly one store row, not many.

## Durability boundary (stated precisely — no overclaiming)

- **The WAL is the durability boundary.** Records that have been
  `append_batch`'d **and fsync'd** survive a `SIGKILL` (process kill) and are
  replayed *exactly* by `recover()`, with CRC32 validation. A torn or
  partially-written **tail** frame is detected (`BadCrc` / `Truncated`) and
  cleanly discarded, with the file truncated to the last good offset ⇒ **zero
  partial-state corruption**. Recovery stops at the first bad frame (an
  append-only log cannot be trusted past a torn write) and is deterministic on
  a second run.
  - **Caveat — read this before trusting it against power loss (honest scope):**
    durability uses `File::sync_data()` (an `fdatasync`-equivalent). That
    guarantees survival of a **process kill (`SIGKILL`)** — the *exact* scenario
    `tests/crash_resilience.rs` exercises and proves. It does **not** guarantee
    **power-loss / drive-cache** survival on macOS, which requires
    `fcntl(F_FULLFSYNC)`. That call is **not wired up here**; the swap point is
    the `sync_data()` call on the append path in **`src/wal.rs`**. Do **not**
    claim power-loss durability from this build as shipped.
- **`submit()` is fire-and-forget.** Items still sitting in the ring (not yet
  WAL'd) are the **explicit at-risk window** on a hard kill. They are *not*
  durable until the worker has logged them.
- **Per-write durability is opt-in.** Callers that need it use
  `submit_durable()` (WAL-append + fsync synchronously *before* returning) or
  `flush_now()` (drain + WAL + store for everything currently pending). These
  block — by design — and are **not** for the hot loop.
- **No in-memory loss on store failure.** When the store rejects a batch (retries
  exhausted, or a `Fatal` error), the worker calls `requeue_dirty`, which
  re-marks the states dirty under last-write-wins so a newer concurrent update is
  never clobbered. The data sits safely in **both** the cache (for the next
  retry) and the WAL (for crash recovery).
- **One LSN authority + a checkpoint floor.** A single, `Arc`-shared
  `lsn::LsnState` (seeded one past the highest recovered LSN) mints every WAL
  LSN — for `submit`, `submit_durable`, `flush_now`, *and* the worker — so all
  records in the one WAL are globally monotonic. A pending-durable floor clamps
  every checkpoint strictly below the lowest still-pending `submit_durable` LSN,
  so a checkpoint can never discard an fsync'd record whose content the store
  has not yet absorbed. See the case study above for the bug this closes.

## Main-loop isolation guarantee

`submit()` contains **NO `Mutex`, NO blocking syscall, NO I/O, NO unbounded
wait**: one `Relaxed` atomic `fetch_add` to mint a sequence number, and one
bounded lock-free CAS enqueue on the ring. Under back-pressure it returns
`Dropped` rather than blocking. All locking (the cache map, the store map) and
all blocking I/O (WAL fsync, store calls) live off the submit path, on the
worker / DB side, behind the lock-free ring.

This is verified adversarially in `tests/mainloop_isolation.rs`: the same submit
storm runs against a healthy store and against a store deliberately made **25 ms
slow** behind a tiny saturated ring. Even with the worker permanently wedged on
the 25 ms backend, the worst observed per-call submit latency stays in the
low-microsecond range (~2 µs, average ~0.1 µs) — about 4 orders of magnitude
below one DB round-trip — and back-pressure converts to drops, never to a
stalled caller.

## Case study: the LSN checkpoint race (found & fixed)

An adversarial durability critic, reviewing the WAL/checkpoint seam, found a way
to **silently lose an fsync-acknowledged `submit_durable()` write on a hard
kill**. It is fixed; this section documents it as a solved case, because it is
the clearest evidence of the value adversarial review adds to a durability
system.

### The bug

Two **independent** LSN counters were writing records into the **one** shared
`Wal`:

- the background **worker** minted LSNs from its own private `next_lsn` (started
  at `1`), while
- `api.rs` `submit_durable()` / `flush_now()` minted from the engine's `seq`
  (started at `0`).

`Wal::checkpoint(up_to_lsn)` discards records by **raw value** — `lsn <=
up_to_lsn` across the whole file — and has no way to tell the two counters
apart. So after the worker checkpointed *its own* batch, it could physically
delete a **low-LSN `submit_durable` frame the store had never absorbed**. That
frame is dangerous precisely because the durable record's best-effort ring copy
can be load-shed as `Dropped`, leaving the **WAL frame as its only durable
copy**. Delete it before the store has the content and a `SIGKILL` at that
instant loses an **acknowledged** write — the one thing `submit_durable()`
promises never to do.

### The fix (`src/lsn.rs`)

`LsnState` — a single, `Arc`-shared, monotonic allocator (`alloc` / `alloc_block`)
seeded **one past the highest LSN recovered from disk** and used by
`submit` / `submit_durable` / `flush_now` **and** the worker. Every record in
the single WAL now carries a globally-unique, strictly-increasing LSN, so a
checkpoint can never confuse one appender's numbers for another's.

On top of that, a **pending-durable checkpoint floor**:

- durable appends call `note_pending(lsn)` **before** the append;
- the worker / `flush_now` call `note_drained(seq)` the instant a request lands
  in the cache — the point past which the requeue machinery guarantees it
  reaches the store;
- `checkpoint_watermark(proposed)` clamps any proposed checkpoint LSN to
  **strictly below the lowest still-pending durable LSN**, so a checkpoint can
  never discard a record whose content the store has not yet absorbed.

The floor is wired at `src/worker.rs` (the checkpoint path) and `src/api.rs`
(`flush_now`). A durable write whose ring copy was shed is never drained, so its
LSN stays pending forever and its WAL frame survives every checkpoint until
recovery replays it on the next start — exactly the records the durability
contract protects.

### How it's guarded

5 `lsn::tests` unit tests (monotonic allocation; watermark equals the proposal
when nothing is pending; watermark clamps strictly below the lowest pending LSN;
watermark returns `None` when a pending record sits at `0`; draining lifts the
floor) plus the integration-flavored
`wal::tests::checkpoint_discards_acked_keeps_live`, which proves a checkpoint
discards acked-and-absorbed records while keeping a still-live durable frame.

## Build / test / run

```sh
# Build the library, the demo, and the crash_child victim binary.
cargo build
cargo build --release

# Unit tests (ring / wal / store / cache / domain / metrics).
cargo test --lib

# Adversarial integration tests:
#   crash_resilience  — SIGKILL mid-write + torn-tail recovery (durability)
#   fault_injection   — transient / down / periodic store failures (no lost updates)
#   mainloop_isolation — submit() never blocks, even behind a 25 ms-slow store
cargo test --test crash_resilience --test fault_injection --test mainloop_isolation -- --nocapture

# High-volume concurrent stress / throughput (run in release for a real number).
cargo test --release --test stress_concurrency -- --nocapture

# Run the simulated game-loop demo (prints metrics + SUCCESS).
cargo run --release --bin demo
# or: ./target/release/demo

# Real-backend wiring examples (both std-only, both run offline).
cargo run --example disk_kv_store   # real append-then-index disk store
cargo run --example redis_store     # Redis reference wiring (in-memory FakeRedis)
```

The crate is `std`-only, so a plain `cargo build` / `cargo test` works with no
network access and no external crates.

## Using the crate

One glob import (`flux_throughput::prelude::*`) brings in everything the three
embedder personas need — the engine + outcome, the config/metrics, the domain
types and the `PersistenceStore` trait for a custom backend, and `MockDb` for
tests. Here it wires a `FluxEngine` against a **custom** store:

```rust
use std::sync::Arc;
use flux_throughput::prelude::*;

// Your backend: anything implementing `PersistenceStore`.
// (See `examples/disk_kv_store.rs` for a real std-only one;
//  `MockDb` below stands in for it.)
let store: Arc<dyn PersistenceStore> = Arc::new(MockDb::new());

let engine = FluxEngine::start(FluxConfig::default(), store)?;

// Fire-and-forget, non-blocking. Under back-pressure this returns Dropped.
match engine.submit(/* player_id */ 1, /* version */ 1, b"state".to_vec()) {
    SubmitOutcome::Enqueued => {}
    SubmitOutcome::Dropped => { /* load shed — retry next tick or rely on LWW */ }
}

// Opt-in per-write durability (blocks until fsync'd — NOT for the hot loop):
engine.submit_durable(1, 2, b"durable state".to_vec())?;

let snap = engine.metrics();   // MetricsSnapshot: submits, flushes, drops, errors
let _ = snap;

engine.shutdown()?; // final drain + flush + join the worker
# Ok::<(), Box<dyn std::error::Error>>(())
```

`FluxEngine::start` takes a `FluxConfig` and an `Arc<dyn PersistenceStore>`; on
startup it `recover()`s the WAL, replays the surviving records into the cache,
and seeds the single LSN allocator one past the highest recovered LSN.

## Binaries

- `cargo run --release --bin demo` — a deterministic, finite simulated game-loop
  (300 ticks at ~60 Hz over a `MockDb` with injected latency). It demonstrates
  the fire-and-forget submit path, prints periodic metrics, drains via
  `flush_now()`, shuts down, and prints `SUCCESS`.
- `crash_child` — the `SIGKILL` victim subprocess driven by
  `tests/crash_resilience.rs`: it performs fsync'd `submit_durable` writes,
  prints an `ACK <i>` after each, and is killed mid-stream so the parent can
  prove `recover()` replays exactly the acked tail.

## Wiring a real persistence backend

The crate persists through one seam — the `PersistenceStore` trait — and ships
two **runnable, std-only** examples showing how a real deployment fills it.
`MockDb` (`src/store.rs`) is **not** that backend: it stays in place as the
in-memory **fault-injection** store the whole test suite drives (unavailable /
fail-next-N / fail-every-N / latency knobs). The examples are the production
shape.

- **`examples/disk_kv_store.rs` — a real, working disk backend (start here).**
  A fast append-then-index on-disk KV store, the same shape a real embedded
  engine (RocksDB / LevelDB / sled) uses at its core, written against **nothing
  but `std`**: framed, CRC-protected records appended to a data file +
  `sync_data()`, an in-memory last-write-wins index, and a torn-tail-tolerant
  replay on `open()`. It actually survives a process restart. Run it:

  ```sh
  cargo run --example disk_kv_store
  ```

  Same honesty as the WAL: `sync_data()` covers `SIGKILL`, not power loss
  (`F_FULLFSYNC` is not exposed by `std`).

- **`examples/redis_store.rs` — a Redis reference wiring.** Shows *precisely* how
  to back `PersistenceStore` with Redis, while still compiling and running
  **offline with zero dependencies**. The trick: a tiny local `RedisConn` trait
  captures the only two operations a real Redis backend needs (a pipelined
  multi-SET and a GET), with an in-memory `FakeRedis` for offline builds.
  `RedisStore<C>` is generic over `C: RedisConn`, so swapping in the real client
  is a **one-line type change** (`RedisStore<FakeRedis>` →
  `RedisStore<RealRedisConn>`) — see the in-file PRODUCTION WIRING block. Run it:

  ```sh
  cargo run --example redis_store        # uses the in-memory FakeRedis
  ```

  **Why `redis` is not a default dependency:** the default build is deliberately
  zero-dependency and fully offline. Adding the real `redis` (or `rocksdb`)
  crate to `Cargo.toml` would force a network index fetch and break offline
  builds. To go live you add `redis` to *your* `Cargo.toml`, implement
  `RedisConn` over `redis::Connection` (a few lines), and hand a
  `RedisStore<RealRedisConn>` to `FluxEngine::start`.

## Benchmarks / verified metrics

All numbers below were observed on the development machine (Apple Silicon,
macOS) in **release** builds (`opt-level = 3`, `lto = true`). Your hardware will
differ, but the *shapes* — zero loss, bounded submit latency, seven-figure
submit throughput — are the contract.

- **Stress / throughput** (`cargo test --release --test stress_concurrency`):
  **8** producer threads submitting **320,000** updates across **50,000**
  distinct players. Sustained submit throughput lands around **~0.9–1.0M
  updates/sec** (observed range ~0.90M–1.06M across runs), with
  **`submits_dropped = 0`** and **exact last-write-wins**: every player's
  persisted version equals the MAX version it was submitted (zero lost updates,
  zero corruption).
- **Demo** (`cargo run --release --bin demo`): **600,000** updates over **300**
  ticks at a simulated **60 Hz**; worst single main-loop tick **1.3 ms** (well
  under the **8 ms** budget), average `submit()` **~69 ns**, **50,000** distinct
  player states persisted, **0** store errors, pipeline fully drained,
  `SUCCESS`.
- **Main-loop isolation** (`cargo test --test mainloop_isolation`): behind a
  deliberately **25 ms-slow** store, worst single `submit()` **~2 µs** (average
  **~0.1 µs**); the system sheds load via `SubmitOutcome::Dropped` rather than
  ever blocking the caller — the slow backend never stalls the hot loop.
- **Crash resilience** (`cargo test --test crash_resilience`): after a hard
  `SIGKILL` taken mid-write, `recover()` returns a gap-free, CRC-clean run of
  the freshest acked records reaching the last ack observed — **zero data loss,
  zero corruption** in the WAL's at-risk tail. A torn tail of raw garbage
  appended to the segment is detected and truncated away, leaving exactly the
  valid records.
- **Suite health:** `cargo build` / `--bins` / `--tests` compile with **zero
  warnings**; **50** library unit tests + **10** integration tests all pass.

## License

MIT OR Apache-2.0.
