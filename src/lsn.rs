//! Single source of truth for WAL log-sequence-number (LSN) assignment.
//! **[impl:worker/api shared] — the LSN authority.**
//!
//! ## Why this module exists (the bug it closes)
//!
//! Every record written into the one shared [`crate::wal::Wal`] must carry a
//! **globally unique, monotonically increasing** LSN, no matter which code path
//! wrote it. There are three appenders into that single log:
//!
//! * the background [`crate::worker`] flush loop (steady-state + final drain),
//! * [`crate::api::FluxEngine::submit_durable`] (synchronous WAL + fsync), and
//! * [`crate::api::FluxEngine::flush_now`] (caller-driven synchronous flush).
//!
//! Historically the worker minted LSNs from its own private counter (which
//! started at one) while the durable API paths minted from the engine's submit
//! `seq` (which started at zero). Two independent counters stamping records into
//! one log produced
//! **colliding, non-monotonic LSNs**. Because [`crate::wal::Wal::checkpoint`]
//! deletes records by raw LSN value (`lsn <= up_to_lsn`) across the *whole* file,
//! it could not tell the two counters apart and would physically discard an
//! acknowledged, fsync'd `submit_durable` record that the store had **never**
//! absorbed — silent loss of an acknowledged durable write.
//!
//! [`LsnState`] fixes this at the root by being the **one** allocator all three
//! paths mint from, and by tracking the records whose only durable copy is still
//! the WAL frame so a checkpoint can never delete them prematurely.
//!
//! ## The checkpoint floor (why `submit_durable` records are never lost)
//!
//! A `submit_durable` record is durable the instant its WAL frame is fsync'd, but
//! it only reaches the slow store later, when the worker drains the best-effort
//! ring copy into the cache and flushes it. Until that record's *content* is
//! guaranteed to reach the store, its WAL frame is its **only** durable copy and
//! must survive every checkpoint.
//!
//! We track each durable append's LSN in [`LsnState::pending`]. The crucial
//! observation: once a `submit_durable` request lands in the cache, the
//! cache + requeue machinery guarantees its content *will* reach the store (no
//! in-memory loss — see `worker.rs`). So we clear a pending LSN exactly when the
//! worker drains that request into the cache (`note_drained`). A durable write
//! whose ring enqueue was shed (`Dropped`) is never drained, so its LSN stays
//! pending forever and its WAL frame is preserved until recovery replays it on
//! the next start — precisely the records the durability contract protects.
//!
//! [`checkpoint_watermark`](LsnState::checkpoint_watermark) clamps any proposed
//! checkpoint LSN to **strictly below the lowest pending durable LSN**, so a
//! checkpoint can only ever discard records whose content the store has (or is
//! guaranteed to have) absorbed.
//!
//! This module is **safe code only** (no `unsafe`).

use crate::domain::Lsn;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Shared LSN authority: a single monotonic allocator plus the set of durable
/// records whose content is not yet guaranteed to be in the store.
///
/// Held behind an [`std::sync::Arc`] and shared by the engine facade and the
/// background worker so both mint LSNs from — and respect the checkpoint floor
/// of — the *same* instance.
pub struct LsnState {
    /// The single monotonic counter. The next LSN handed out is the current
    /// value; allocation is a `fetch_add`, so LSNs are globally unique and
    /// strictly increasing across every appender.
    next: AtomicU64,
    /// LSNs of durable (`submit_durable` / `flush_now`) appends whose content is
    /// not yet guaranteed to have reached the store. A checkpoint must never
    /// discard a record at or above the minimum of this set. Guarded by a mutex
    /// that is touched only off the hot submit path (durable appends, worker
    /// drains, and checkpoints — all already on the slow side of the ring).
    pending: Mutex<BTreeSet<Lsn>>,
}

impl LsnState {
    /// Creates an allocator whose first handed-out LSN is `start`.
    ///
    /// `start` is seeded by [`crate::api::FluxEngine::start`] to one past the
    /// highest LSN recovered from the WAL, so freshly minted LSNs are globally
    /// monotonic with respect to everything already on disk.
    pub fn new(start: Lsn) -> Self {
        LsnState {
            next: AtomicU64::new(start),
            pending: Mutex::new(BTreeSet::new()),
        }
    }

    /// Reserves a single LSN and returns it. Wait-free.
    pub fn alloc(&self) -> Lsn {
        self.next.fetch_add(1, Ordering::Relaxed)
    }

    /// Reserves a contiguous block of `count` LSNs and returns the first.
    ///
    /// The reserved range is `[first, first + count)`. Used by batch appenders
    /// (the worker flush and `flush_now`) so a whole batch is stamped from one
    /// atomic operation while staying globally monotonic.
    pub fn alloc_block(&self, count: u64) -> Lsn {
        self.next.fetch_add(count, Ordering::Relaxed)
    }

    /// Registers `lsn` as a durable append whose content is not yet guaranteed
    /// to be in the store, so no checkpoint may discard its WAL frame yet.
    pub fn note_pending(&self, lsn: Lsn) {
        self.pending
            .lock()
            .expect("LSN pending mutex poisoned")
            .insert(lsn);
    }

    /// Clears a previously-[`note_pending`](Self::note_pending)'d durable LSN
    /// because its content is now guaranteed to reach the store (its request was
    /// drained into the cache, or it was synchronously persisted). A no-op if the
    /// LSN was never pending.
    pub fn note_drained(&self, lsn: Lsn) {
        self.pending
            .lock()
            .expect("LSN pending mutex poisoned")
            .remove(&lsn);
    }

    /// Clamps a proposed checkpoint LSN to a value it is **safe** to discard up
    /// to: strictly below the lowest still-pending durable LSN.
    ///
    /// Returns `Some(safe)` when a checkpoint at `safe` (`<= proposed`) is safe,
    /// or `None` when there is nothing safe to checkpoint (the lowest pending
    /// durable record sits at or below everything the caller wanted to discard,
    /// so the caller must keep the whole tail this round and try again later).
    pub fn checkpoint_watermark(&self, proposed: Lsn) -> Option<Lsn> {
        let pending = self.pending.lock().expect("LSN pending mutex poisoned");
        match pending.iter().next().copied() {
            // No pending durable records: the proposal is fully safe.
            None => Some(proposed),
            // There is a pending durable record at `min`. We may discard only
            // records strictly below it. If `min == 0` nothing is safe; if the
            // proposal is already below `min`, it is unaffected.
            Some(min) => {
                if min == 0 {
                    None
                } else {
                    Some(proposed.min(min - 1))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_is_monotonic() {
        let s = LsnState::new(5);
        assert_eq!(s.alloc(), 5);
        assert_eq!(s.alloc(), 6);
        assert_eq!(s.alloc_block(3), 7); // reserves 7,8,9
        assert_eq!(s.alloc(), 10);
    }

    #[test]
    fn watermark_is_proposal_when_no_pending() {
        let s = LsnState::new(0);
        assert_eq!(s.checkpoint_watermark(42), Some(42));
    }

    #[test]
    fn watermark_clamps_below_min_pending() {
        let s = LsnState::new(0);
        s.note_pending(10);
        s.note_pending(20);
        // A proposal above the lowest pending is clamped to just below it.
        assert_eq!(s.checkpoint_watermark(50), Some(9));
        // A proposal already below the lowest pending is unaffected.
        assert_eq!(s.checkpoint_watermark(5), Some(5));
    }

    #[test]
    fn watermark_none_when_pending_is_zero() {
        let s = LsnState::new(0);
        s.note_pending(0);
        assert_eq!(s.checkpoint_watermark(7), None);
    }

    #[test]
    fn draining_clears_the_floor() {
        let s = LsnState::new(0);
        s.note_pending(10);
        assert_eq!(s.checkpoint_watermark(50), Some(9));
        s.note_drained(10);
        // Floor lifted: the full proposal is safe again.
        assert_eq!(s.checkpoint_watermark(50), Some(50));
    }
}
