//! Write-behind cache with per-entry dirty flags + last-write-wins coalescing
//! (`impl Cache`).
//!
//! ## Role in the pipeline
//!
//! ```text
//!   submit() ──▶ Queue (ring) ──▶ Cache (THIS FILE) ──▶ WriteAheadLog ──▶ Store
//!   (hot loop)   lock-free        worker thread          durability       slow backend
//! ```
//!
//! [`WriteBehindCache`] is the **batching / dirty-flag layer**. The background
//! worker drains the lock-free ring and feeds every dequeued [`WriteRequest`]
//! into [`apply`](Cache::apply). The cache then does two jobs that turn a flood
//! of tiny writes into a trickle of large, ordered store batches:
//!
//! 1. **Coalescing via last-write-wins (LWW).** Many writes for the *same*
//!    `player_id` collapse into a single map entry: only the newest `version`
//!    is retained, and the entry is flagged `dirty`. A player hammered with 1000
//!    updates between two flushes produces exactly **one** row in the next store
//!    batch instead of 1000. This is the core mechanism that converts write
//!    *amplification* into write *coalescing* and slashes DB load.
//!
//! 2. **Dirty staging for the WAL+store.** [`take_dirty`](Cache::take_dirty)
//!    atomically snapshots the set of entries that have changed since the last
//!    flush, clears their dirty flags, and hands *copies* to the worker. The
//!    entries themselves stay in the map so concurrent
//!    [`cache_get`](crate::api::FluxEngine::cache_get) reads keep being served
//!    the latest value (read-your-writes within the process).
//!
//! ## Crash / failure safety
//!
//! If the store rejects a staged batch (after the worker exhausts its retries,
//! or on a `Fatal` error), the worker calls
//! [`requeue_dirty`](Cache::requeue_dirty) to re-flag those states dirty so they
//! are retried on the next flush. The requeue is itself LWW-merged against the
//! live map, so a *newer* update that landed concurrently is never clobbered by
//! a stale requeued copy. Net effect: **a failed store flush loses NO update.**
//!
//! ## Locking & isolation
//!
//! The single [`Mutex`] guarding the map lives **here, on the worker side** —
//! `submit()` never touches this file, so the main-loop isolation guarantee
//! holds. This module is safe code only (no `unsafe`).

use std::collections::hash_map::Entry as MapEntry;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use crate::domain::{Cache, PlayerId, PlayerState, WriteRequest};

/// One coalesced cache slot for a single player.
///
/// We store the full [`PlayerState`] plus a `dirty` bit. `dirty == true` means
/// "this state has been modified since the last successful stage and must still
/// be flushed to the store". `dirty == false` means "the store already has (or
/// is in the process of receiving) this exact version" — it is kept purely so
/// reads can still be served from memory.
struct Entry {
    /// The latest known state for this player (highest `version` seen).
    state: PlayerState,
    /// Whether this entry still needs to be flushed to the durable store.
    dirty: bool,
}

/// Write-behind cache with per-entry dirty flags and last-write-wins semantics.
///
/// Construct with [`WriteBehindCache::new`]. Internally a single
/// `Mutex<Inner>` protects the player map and a side index of dirty ids. The
/// side index makes [`take_dirty`](Cache::take_dirty) and
/// [`dirty_len`](Cache::dirty_len) O(dirty) / O(1) instead of requiring a full
/// scan of the (potentially huge) map on every flush tick.
pub struct WriteBehindCache {
    inner: Mutex<Inner>,
}

/// All mutable cache state, guarded by the single [`Mutex`] in
/// [`WriteBehindCache`]. Splitting it into one struct keeps the lock scope
/// obvious: every method locks `inner` exactly once and operates on both the
/// map and the dirty index together, so the two can never drift out of sync.
struct Inner {
    /// `player_id -> Entry`. The authoritative, coalesced view of every player
    /// that has passed through the cache (dirty or clean).
    map: HashMap<PlayerId, Entry>,
    /// Side index of the ids whose `Entry.dirty == true`. This is a pure
    /// performance optimization: it is *always* kept in lockstep with the
    /// `dirty` bits in `map` (an id is in `dirty` iff its entry is dirty), so
    /// correctness never depends on it — it only avoids scanning the whole map.
    dirty: HashSet<PlayerId>,
}

impl WriteBehindCache {
    /// Creates an empty write-behind cache.
    pub fn new() -> Self {
        WriteBehindCache {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                dirty: HashSet::new(),
            }),
        }
    }
}

impl Default for WriteBehindCache {
    fn default() -> Self {
        WriteBehindCache::new()
    }
}

impl Cache for WriteBehindCache {
    /// Upserts `req` into the cache under **last-write-wins by `version`** and
    /// flags the entry dirty.
    ///
    /// ## Coalescing semantics
    ///
    /// * **No existing entry** → insert the new state, mark dirty. (First write
    ///   for this player.)
    /// * **`req.version >= entry.version`** → the incoming write is newer-or-equal,
    ///   so it *wins*: replace the stored state and mark dirty. Using `>=` (rather
    ///   than `>`) means a re-applied equal version still re-arms the dirty flag,
    ///   which is the safe choice — it can only cause a redundant flush, never a
    ///   lost update.
    /// * **`req.version < entry.version`** → the incoming write is **stale /
    ///   out-of-order** (the ring is MPMC, so requests can be reordered relative
    ///   to their `version`). We keep the newer state we already hold and discard
    ///   the stale one. Crucially we do **NOT** clear `dirty`: whatever pending
    ///   flush state the entry was in is preserved. A stale apply must never make
    ///   a dirty entry look clean.
    ///
    /// Because repeated applies for the same player overwrite the same slot, a
    /// burst of N writes to one player between flushes coalesces into a single
    /// dirty entry — the store sees one row, not N.
    fn apply(&self, req: &WriteRequest) {
        // Lock lives on the worker side; submit() never reaches this code path.
        let mut inner = self.inner.lock().unwrap();
        match inner.map.entry(req.player_id) {
            MapEntry::Occupied(mut occ) => {
                let entry = occ.get_mut();
                if req.version >= entry.version() {
                    // Newer-or-equal write wins: overwrite state, re-arm dirty.
                    entry.state = req.to_state();
                    entry.dirty = true;
                    inner.dirty.insert(req.player_id);
                }
                // else: stale/out-of-order write — keep the newer state and DO
                // NOT touch `dirty`. (Falls through, lock released on drop.)
            }
            MapEntry::Vacant(vac) => {
                // First time we have seen this player: insert dirty.
                vac.insert(Entry {
                    state: req.to_state(),
                    dirty: true,
                });
                inner.dirty.insert(req.player_id);
            }
        }
    }

    /// Atomically collects up to `max` dirty entries, clears their dirty flags,
    /// and returns **copies** of their [`PlayerState`] for the worker to flush.
    ///
    /// This is the staging step handed to the WAL + store. It is atomic in the
    /// sense that the whole selection + flag-clearing happens under a single
    /// lock acquisition, so two concurrent flushes can never both claim the same
    /// dirty entry, and an `apply` either lands entirely before or entirely
    /// after this snapshot.
    ///
    /// The entries are intentionally **kept in the map** (only their `dirty` bit
    /// is cleared) so that [`get`](Cache::get) continues to serve the latest
    /// value to in-process readers while the flush is in flight. If the store
    /// later rejects the batch, [`requeue_dirty`](Cache::requeue_dirty) re-arms
    /// the flags from these same copies.
    fn take_dirty(&self, max: usize) -> Vec<PlayerState> {
        let mut inner = self.inner.lock().unwrap();
        if max == 0 || inner.dirty.is_empty() {
            return Vec::new();
        }

        // Pull up to `max` ids out of the dirty index. We drain into a local
        // vec first (can't mutate `inner.map` while borrowing `inner.dirty`'s
        // iterator), then clear each entry's flag.
        let take_n = max.min(inner.dirty.len());
        let chosen: Vec<PlayerId> = inner.dirty.iter().copied().take(take_n).collect();

        let mut out = Vec::with_capacity(chosen.len());
        for id in chosen {
            // Remove from the dirty index (the side index and the bit stay in
            // lockstep) ...
            inner.dirty.remove(&id);
            if let Some(entry) = inner.map.get_mut(&id) {
                // ... clear the per-entry dirty bit ...
                entry.dirty = false;
                // ... and hand the worker a COPY; the entry remains in the map
                // so cache_get keeps serving reads during the flush.
                out.push(entry.state.clone());
            }
        }
        out
    }

    /// Re-marks the given states dirty after a failed store persist, merging
    /// under last-write-wins so **no update is ever lost** and no newer
    /// concurrent write is clobbered.
    ///
    /// ## Why this is the no-data-loss linchpin
    ///
    /// [`take_dirty`](Cache::take_dirty) already cleared these entries' dirty
    /// bits (optimistically assuming the flush would succeed). If the store then
    /// fails, those updates would be silently dropped on the next flush — unless
    /// we put the dirty bits back. That is exactly what this method does.
    ///
    /// ## The concurrency hazard it guards against
    ///
    /// Between `take_dirty` (flush start) and `requeue_dirty` (flush failure),
    /// the worker may have applied a *newer* write for the same player. We must
    /// not resurrect the stale snapshot over that newer state. So for each
    /// requeued state we compare versions against the live entry:
    ///
    /// * **Live version `<=` requeued version** → the live entry is the same-or-
    ///   older as what we tried to flush, so the requeued state is still
    ///   authoritative (or equal). Re-arm dirty. If the live state is strictly
    ///   older we also restore the requeued (newer) state — this happens only if
    ///   a stale apply slipped in, and LWW keeps the newer value.
    /// * **Live version `>` requeued version** → a newer write already replaced
    ///   this entry and is itself dirty (every apply sets dirty). The newer write
    ///   subsumes the failed one, so we leave it untouched: re-flagging would be
    ///   redundant and overwriting would be data loss. We simply drop the stale
    ///   requeued copy.
    /// * **Entry vanished** (e.g. never re-applied) → reinsert the requeued state
    ///   as dirty so the failed update is not lost.
    fn requeue_dirty(&self, states: Vec<PlayerState>) {
        let mut inner = self.inner.lock().unwrap();
        for state in states {
            let id = state.player_id;
            match inner.map.entry(id) {
                MapEntry::Occupied(mut occ) => {
                    let entry = occ.get_mut();
                    if entry.version() > state.version {
                        // A strictly newer write won concurrently. It is already
                        // dirty (apply always sets dirty), so the failed update
                        // is subsumed — do nothing, do NOT clobber the newer one.
                        continue;
                    }
                    // Live entry is same-or-older than the requeued state: the
                    // requeued update is (still) authoritative. Restore its state
                    // under LWW and re-arm dirty so it gets retried next flush.
                    entry.state = state;
                    entry.dirty = true;
                    inner.dirty.insert(id);
                }
                MapEntry::Vacant(vac) => {
                    // Entry was evicted/never re-applied: reinsert it dirty so
                    // the failed store update survives to the next flush.
                    vac.insert(Entry { state, dirty: true });
                    inner.dirty.insert(id);
                }
            }
        }
    }

    /// Reads the currently-cached state for `id`, returning a clone.
    ///
    /// Serves both dirty (pending-flush) and clean (already-flushed) entries, so
    /// in-process readers always see the latest value regardless of flush state.
    fn get(&self, id: PlayerId) -> Option<PlayerState> {
        let inner = self.inner.lock().unwrap();
        inner.map.get(&id).map(|e| e.state.clone())
    }

    /// Number of dirty (un-flushed) entries currently pending a store write.
    ///
    /// O(1) via the side index; used by the worker as the VOLUME flush trigger
    /// (`dirty_len() >= flush_max_batch`).
    fn dirty_len(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.dirty.len()
    }

    /// Total number of cached entries (dirty or clean).
    fn len(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.map.len()
    }
}

impl Entry {
    /// Convenience accessor for the entry's current version.
    #[inline]
    fn version(&self) -> u64 {
        self.state.version
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Cache, PlayerState, WriteRequest};

    /// Builds a `WriteRequest` with a deterministic blob derived from its inputs
    /// so tests can assert *which* version's payload survived coalescing.
    fn req(player_id: PlayerId, version: u64, seq: u64) -> WriteRequest {
        WriteRequest {
            player_id,
            version,
            blob: vec![version as u8],
            seq,
        }
    }

    /// Two applies for the same player coalesce into ONE dirty entry, and the
    /// newer version wins (last-write-wins coalescing).
    #[test]
    fn coalescing_two_applies_one_dirty() {
        let cache = WriteBehindCache::new();
        cache.apply(&req(1, 1, 0));
        cache.apply(&req(1, 2, 1));

        // One map entry, one dirty entry — the burst collapsed.
        assert_eq!(cache.len(), 1, "two writes to one player => one entry");
        assert_eq!(cache.dirty_len(), 1, "coalesced into a single dirty flag");

        // The newer version's state is what is retained.
        let got = cache.get(1).expect("entry present");
        assert_eq!(got.version, 2);
        assert_eq!(got.blob, vec![2u8]);
    }

    /// `take_dirty` returns the coalesced states and clears the dirty flags,
    /// while the entries remain readable via `get`.
    #[test]
    fn take_dirty_clears_flags_but_keeps_entries() {
        let cache = WriteBehindCache::new();
        cache.apply(&req(1, 5, 0));
        cache.apply(&req(2, 9, 1));
        assert_eq!(cache.dirty_len(), 2);

        let mut taken = cache.take_dirty(100);
        taken.sort_by_key(|s| s.player_id);
        assert_eq!(taken.len(), 2);
        assert_eq!(taken[0].player_id, 1);
        assert_eq!(taken[0].version, 5);
        assert_eq!(taken[1].player_id, 2);
        assert_eq!(taken[1].version, 9);

        // Flags cleared ...
        assert_eq!(cache.dirty_len(), 0, "take_dirty clears the dirty flags");
        // ... but entries stay so reads are still served.
        assert_eq!(cache.len(), 2, "entries remain in the map for reads");
        assert_eq!(cache.get(1).unwrap().version, 5);

        // A second take with nothing dirty yields nothing.
        assert!(cache.take_dirty(100).is_empty());
    }

    /// `take_dirty(max)` respects the cap and a follow-up call drains the rest.
    #[test]
    fn take_dirty_respects_max() {
        let cache = WriteBehindCache::new();
        for id in 0..5 {
            cache.apply(&req(id, 1, id));
        }
        assert_eq!(cache.dirty_len(), 5);

        let first = cache.take_dirty(2);
        assert_eq!(first.len(), 2, "honors the max cap");
        assert_eq!(cache.dirty_len(), 3, "remaining still dirty");

        let second = cache.take_dirty(100);
        assert_eq!(second.len(), 3, "drains the remainder");
        assert_eq!(cache.dirty_len(), 0);
    }

    /// After a simulated store failure, `requeue_dirty` restores the dirty flag
    /// so the update is retried (no data loss).
    #[test]
    fn requeue_restores_dirty() {
        let cache = WriteBehindCache::new();
        cache.apply(&req(1, 3, 0));

        // Worker stages the batch (clears dirty), then the store "fails".
        let staged = cache.take_dirty(10);
        assert_eq!(cache.dirty_len(), 0);

        // Requeue puts the dirty flag back so the next flush retries it.
        cache.requeue_dirty(staged);
        assert_eq!(cache.dirty_len(), 1, "requeue re-arms the dirty flag");

        let again = cache.take_dirty(10);
        assert_eq!(again.len(), 1);
        assert_eq!(
            again[0].version, 3,
            "the same update survives to be retried"
        );
    }

    /// `requeue_dirty` must NOT clobber a newer concurrent write: if a higher
    /// version landed while the flush was in flight, the newer state is kept and
    /// the stale requeued copy is discarded.
    #[test]
    fn requeue_does_not_clobber_newer_write() {
        let cache = WriteBehindCache::new();
        cache.apply(&req(1, 3, 0));
        let staged = cache.take_dirty(10); // snapshot of version 3

        // A newer write (version 7) lands concurrently before the failed flush
        // is requeued.
        cache.apply(&req(1, 7, 1));

        cache.requeue_dirty(staged); // tries to restore stale version 3

        // The newer write wins; the stale requeue is dropped.
        let got = cache.get(1).unwrap();
        assert_eq!(got.version, 7, "newer concurrent write preserved");
        assert_eq!(got.blob, vec![7u8]);
        // Still exactly one dirty entry (the newer write), not a resurrected stale one.
        assert_eq!(cache.dirty_len(), 1);
        let drained = cache.take_dirty(10);
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].version, 7);
    }

    /// A stale / out-of-order apply (lower version) is rejected: the newer state
    /// is kept and the dirty flag is NOT disturbed.
    #[test]
    fn stale_version_rejected() {
        let cache = WriteBehindCache::new();
        cache.apply(&req(1, 10, 0));

        // Drain so the entry is clean, to prove a stale apply doesn't re-dirty
        // or downgrade it.
        let _ = cache.take_dirty(10);
        assert_eq!(cache.dirty_len(), 0);

        // Out-of-order older write arrives.
        cache.apply(&req(1, 4, 1));

        // State unchanged (still version 10) and entry stays clean.
        let got = cache.get(1).unwrap();
        assert_eq!(got.version, 10, "stale write rejected, newer state kept");
        assert_eq!(got.blob, vec![10u8]);
        assert_eq!(
            cache.dirty_len(),
            0,
            "stale apply must not re-dirty a clean entry"
        );
    }

    /// An equal-version apply re-arms the dirty flag (`>=` semantics): safe
    /// because it can only cause a redundant flush, never a lost update.
    #[test]
    fn equal_version_rearms_dirty() {
        let cache = WriteBehindCache::new();
        cache.apply(&req(1, 5, 0));
        let _ = cache.take_dirty(10);
        assert_eq!(cache.dirty_len(), 0);

        // Same version applied again.
        cache.apply(&req(1, 5, 1));
        assert_eq!(
            cache.dirty_len(),
            1,
            "equal version re-arms dirty under >= semantics"
        );
    }

    /// `requeue_dirty` reinserts an entry that was evicted/absent, so a failed
    /// update is never lost even if the slot disappeared.
    #[test]
    fn requeue_reinserts_absent_entry() {
        let cache = WriteBehindCache::new();
        // Construct a state that the cache has never seen.
        let orphan = PlayerState {
            player_id: 42,
            version: 1,
            blob: vec![0xAB],
        };
        assert!(cache.get(42).is_none());

        cache.requeue_dirty(vec![orphan.clone()]);
        assert_eq!(cache.dirty_len(), 1, "absent entry reinserted as dirty");
        assert_eq!(cache.get(42).unwrap(), orphan);
    }
}
