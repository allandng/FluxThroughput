//! # FluxThroughput
//!
//! A lock-free, crash-safe, **write-behind persistence pipeline** that isolates a
//! latency-sensitive "hot" main loop (think a game tick) from a slow durable
//! backend. The design goal is a single, non-negotiable property:
//!
//! > **`submit()` never blocks the main loop.**
//!
//! ## Architecture
//!
//! ```text
//!  main loop                          background worker thread
//!  ─────────                          ────────────────────────
//!  submit(id, ver, blob)
//!     │  Relaxed fetch_add seq
//!     │  build WriteRequest
//!     ▼
//!  ┌──────────────┐  drain   ┌──────────────┐  take_dirty  ┌──────────────┐
//!  │  RingBuffer  │ ───────▶ │  WriteBehind │ ───────────▶ │     Wal      │
//!  │  (lock-free  │          │    Cache     │   records    │ (durability  │
//!  │   MPMC ring) │          │ (coalesce by │              │   boundary)  │
//!  └──────────────┘          │  last-write- │              └──────┬───────┘
//!     ▲   full → Dropped     │   wins)      │   append+fsync      │ then
//!     │   (load shedding)    └──────────────┘   BEFORE store      ▼
//!     │                                                    ┌──────────────┐
//!  caller is never blocked                                 │ Persistence  │
//!                                                          │    Store     │
//!                                                          └──────────────┘
//! ```
//!
//! Producers only ever touch the lock-free [`ring`]. Everything to the right of
//! it runs on the single background [`worker`] thread, the only place blocking
//! I/O and locking are allowed. Bursts are coalesced in the [`cache`] under
//! last-write-wins, written ahead to the [`wal`], and finally persisted to the
//! [`store`] with retry/requeue so **no update is ever lost**.
//!
//! ## Durability boundary (stated precisely — no overclaiming)
//!
//! * **The WAL is the durability boundary.** Records that have been
//!   `append_batch`'d **and fsync'd** survive `SIGKILL` and are replayed exactly
//!   by [`domain::WriteAheadLog::recover`], with CRC32 validation. A torn /
//!   partially-written **tail** frame is detected
//!   ([`error::DecodeError::BadCrc`] / [`error::DecodeError::Truncated`]) and
//!   cleanly discarded, with the file truncated to the last good offset — ZERO
//!   partial-state corruption.
//! * **[`api::FluxEngine::submit`] is fire-and-forget.** Items still sitting in
//!   the ring (not yet WAL'd) are the **explicit at-risk window** on a hard
//!   kill. Callers that need per-write durability must use
//!   [`api::FluxEngine::submit_durable`] or [`api::FluxEngine::flush_now`].
//! * **fsync granularity (stated honestly).** The WAL uses `File::sync_data()`
//!   (fdatasync-equivalent), which guarantees survival of a process kill
//!   (`SIGKILL`). It does NOT guarantee power-loss / drive-cache survival on
//!   macOS — that needs `fcntl(F_FULLFSYNC)`, which is not wired up (swap point:
//!   `Wal::append_batch` in [`wal`]). Do not rely on this for power-loss
//!   durability.
//! * **Checkpoint floor.** A single [`lsn::LsnState`] (shared by the durable API
//!   paths and the worker) pins every still-pending `submit_durable` LSN and
//!   clamps checkpoints strictly below it, so a checkpoint can never discard an
//!   fsync'd record whose content the store has not yet absorbed.
//!
//! ## Main-loop isolation guarantee
//!
//! [`api::FluxEngine::submit`] contains NO `Mutex`, NO blocking syscall, and NO
//! unbounded wait: it is a `Relaxed` atomic `fetch_add` plus a lock-free ring
//! CAS enqueue. Under back-pressure it sheds load (returns
//! [`api::SubmitOutcome::Dropped`]) rather than blocking. All locking lives off
//! the submit path, on the worker / DB side.
//!
//! ## Dependencies
//!
//! **None.** This crate is `std`-only — no `crossbeam`, `tokio`, `rand`, or
//! `crc` crate. The CRC32 used by the WAL is a pure-Rust IEEE implementation in
//! [`domain`].

// ----------------------------------------------------------------------------
// Module declarations — every module of the crate.
// ----------------------------------------------------------------------------

pub mod api;
pub mod cache;
pub mod config;
pub mod domain;
pub mod error;
pub mod lsn;
pub mod metrics;
pub mod ring;
pub mod store;
pub mod wal;
pub mod worker;

// ----------------------------------------------------------------------------
// Public surface re-exports — the canonical names embedders use.
// ----------------------------------------------------------------------------

// --- Engine facade + outcome (game-loop integrator) ---
pub use api::{FluxEngine, SubmitOutcome};

// --- Configuration ---
pub use config::FluxConfig;

// --- Metrics ---
pub use metrics::{Metrics, MetricsSnapshot};

// --- Core domain value types ---
pub use domain::{Lsn, PlayerId, PlayerState, WalRecord, WriteRequest};

// --- Core domain traits (custom PersistenceStore / Cache / Queue / WAL implementors) ---
pub use domain::{Cache, PersistenceStore, Queue, WriteAheadLog};

// --- Error types ---
pub use error::{DecodeError, FluxError, StoreError};

// --- Built-in fault-injection store ---
// Re-exported at the crate root so test / fault-injection embedders can wire it
// against the engine (`Arc<dyn PersistenceStore>`) without reaching into the
// `store` submodule.
pub use store::MockDb;

// NOTE: `lsn::LsnState` is intentionally NOT re-exported at the crate root. It
// stays reachable as `flux_throughput::lsn::LsnState` for advanced users, but it
// is an internal durability mechanism (the engine wires it automatically), so it
// is kept out of the root surface and out of the [`prelude`].

// ----------------------------------------------------------------------------
// Prelude — one glob covering all three embedder personas.
// ----------------------------------------------------------------------------

/// Everything a typical embedder needs in a single glob import:
///
/// ```
/// use flux_throughput::prelude::*;
/// ```
///
/// This pulls in the three personas the crate serves:
///
/// * **(a) game-loop integrator** — [`FluxEngine`], [`SubmitOutcome`],
///   [`FluxConfig`], [`Metrics`], [`MetricsSnapshot`]: enough to start the
///   engine, submit from the hot loop, and read metrics.
/// * **(b) custom store implementor** — [`PlayerId`], [`PlayerState`],
///   [`PersistenceStore`], [`StoreError`] (plus the other domain value types and
///   traits commonly needed): enough to back the pipeline with your own database.
/// * **(c) test / fault-injection embedder** — [`MockDb`]: the built-in
///   in-memory store with fault knobs for exercising retry / requeue / load-shed
///   paths.
///
/// [`crate::lsn::LsnState`] is deliberately omitted — it is an internal
/// durability mechanism the engine wires automatically and embedders never need.
pub mod prelude {
    // (a) game-loop integrator
    pub use crate::api::{FluxEngine, SubmitOutcome};
    pub use crate::config::FluxConfig;
    pub use crate::metrics::{Metrics, MetricsSnapshot};
    // (b) custom PersistenceStore implementor
    pub use crate::domain::{PersistenceStore, PlayerId, PlayerState};
    pub use crate::error::StoreError;
    // also expose the other domain value types + traits commonly needed
    pub use crate::domain::{Cache, Lsn, Queue, WalRecord, WriteAheadLog, WriteRequest};
    pub use crate::error::{DecodeError, FluxError};
    // (c) test / fault-injection embedder
    pub use crate::store::MockDb;
    // LsnState deliberately omitted from the prelude (internal mechanism).
}
