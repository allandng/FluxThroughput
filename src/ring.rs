//! Lock-free bounded MPMC ring buffer (`impl Queue`). **[impl:ring]**
//!
//! # Why this guarantees main-loop isolation
//!
//! This is THE component the hot game main loop touches on every
//! [`crate::api::FluxEngine::submit`]. The entire raison d'être of the crate —
//! "`submit()` never blocks the main loop" — rests on this file being genuinely
//! lock-free, so it is worth stating *precisely* what that buys us:
//!
//! * **No mutex, ever.** A [`std::sync::Mutex`] would let a slow consumer (the
//!   flush worker, ultimately blocked on disk fsync or a stalled database) hold
//!   a lock that the main loop then has to wait on — priority inversion that
//!   stalls the game tick. This ring uses only plain atomic loads/stores and a
//!   single bounded compare-and-swap (CAS). A producer is *never* parked,
//!   *never* descheduled waiting on another thread, and *never* makes a syscall.
//! * **CAS-only publish, wait-free in the common case.** Enqueue is a
//!   `compare_exchange_weak` on a single `enqueue_pos` counter. The *only* reason
//!   it ever loops is genuine contention with another producer that won the same
//!   slot — and each such loss corresponds to *another* producer making
//!   guaranteed forward progress (the textbook definition of lock-free). There
//!   is no spin against a held lock, so an OS-descheduled or even crashed
//!   producer cannot wedge the others.
//! * **O(1), bounded work.** Enqueue/dequeue touch exactly one cell and one
//!   shared counter. There is no scanning, no growth, no allocation on the hot
//!   path (the `Box<[Cell]>` is allocated once at construction). The latency of
//!   `submit()` is therefore a small constant regardless of how full the ring
//!   is or how far behind the worker has fallen.
//! * **Bounded ⇒ load-shedding, not blocking.** Because capacity is fixed,
//!   `try_enqueue` on a full ring returns `Err(req)` *immediately* instead of
//!   waiting for space. The caller (`submit`) turns that into
//!   [`crate::api::SubmitOutcome::Dropped`] and moves on. Under sustained
//!   back-pressure the system *sheds load* — the game tick keeps its deadline —
//!   rather than letting an unbounded queue eat all memory or letting a blocking
//!   `send` freeze the frame.
//!
//! # Algorithm: Dmitry Vyukov's bounded MPMC queue
//!
//! Each slot carries a `sequence` stamp (an `AtomicUsize`) used to coordinate
//! producers and consumers *without* a lock and without ABA hazards on the
//! monotonically-increasing `enqueue_pos` / `dequeue_pos` counters:
//!
//! * Cell `i` is initialised with `sequence == i`.
//! * A producer at logical position `pos` may write into cell `pos & mask` only
//!   when that cell's `sequence == pos` (it is empty and "expecting" this lap).
//!   After writing the payload it stores `sequence = pos + 1`, which is exactly
//!   the value a consumer at `pos` is waiting for — a one-way `Release` handoff.
//! * A consumer at logical position `pos` may read cell `pos & mask` only when
//!   its `sequence == pos + 1` (a producer has published). After reading it
//!   stores `sequence = pos + mask + 1` — the stamp the *next* producer to reach
//!   this slot (one full lap later, at logical position `pos + capacity`) is
//!   waiting for.
//!
//! The signed difference `seq - pos` (resp. `seq - (pos + 1)`) is the whole
//! decision: `0` means "this slot is mine, try to claim it", `< 0` means
//! "full" / "empty", `> 0` means "another thread already advanced past me,
//! reload and retry". This is what makes the queue correct under arbitrary
//! interleavings of many producers and many consumers.
//!
//! # `unsafe` usage
//!
//! Per the crate coding standard, `ring.rs` is the **only** file permitted to
//! use `unsafe`, and every `unsafe` block below is prefaced with a `// SAFETY:`
//! comment. The unsafety is confined to the slot payload: a
//! `UnsafeCell<MaybeUninit<WriteRequest>>` whose initialisation state is *proven*
//! by the sequence-stamp protocol above, so that exactly one thread ever touches
//! a given slot's payload at a given time, and a slot is only read after it was
//! written. No data race is possible; the `// SAFETY:` notes spell out the
//! invariant each block relies on.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::domain::{Queue, WriteRequest};

/// A cache-line-aligned wrapper around an [`AtomicUsize`].
///
/// `enqueue_pos` and `dequeue_pos` are written by *disjoint* sets of threads
/// (producers vs. consumers). If they shared a cache line, every producer CAS
/// would invalidate the consumers' cached copy of the *other* counter and vice
/// versa — "false sharing" that can cost an order of magnitude in throughput.
/// `#[repr(align(64))]` forces each onto its own 64-byte line (the common cache
/// line size on x86-64 and aarch64) so the two hot counters never ping-pong.
#[repr(align(64))]
struct CachePadded(AtomicUsize);

/// One ring slot: a published-state stamp plus an uninitialised payload cell.
///
/// `sequence` is the lock-free coordination variable (see the module docs); it
/// is the *only* field that is ever touched atomically. `slot` holds the actual
/// [`WriteRequest`] payload inside an [`UnsafeCell`] so it can be mutated through
/// a shared `&Cell`, wrapped in [`MaybeUninit`] because a slot is logically
/// empty (uninitialised) until a producer publishes into it.
struct Cell {
    /// Publication stamp coordinating producers and consumers for this slot.
    sequence: AtomicUsize,
    /// The payload. Only ever accessed by the single thread that currently
    /// "owns" this slot per the sequence protocol; see the `// SAFETY:` notes.
    slot: UnsafeCell<MaybeUninit<WriteRequest>>,
}

/// Lock-free bounded MPMC ring buffer of [`WriteRequest`]s.
///
/// Implements [`Queue`] using Vyukov's sequence-stamped bounded MPMC algorithm.
/// Construct with [`RingBuffer::new`]; the requested capacity is rounded **up to
/// the next power of two** so the logical position can be reduced to a slot
/// index with a cheap bit-mask (`pos & mask`) instead of a `%` divide.
///
/// See the module-level documentation for the correctness argument and the
/// main-loop-isolation rationale.
pub struct RingBuffer {
    /// The fixed-size slot array, allocated once at construction.
    buffer: Box<[Cell]>,
    /// `capacity - 1`; ANDing a logical position with this yields its slot
    /// index. Valid precisely because `capacity` is a power of two.
    mask: usize,
    /// Power-of-two capacity (number of slots).
    capacity: usize,
    /// Next logical position a producer will claim. Advanced via CAS by
    /// producers; cache-line isolated from `dequeue_pos`.
    enqueue_pos: CachePadded,
    /// Next logical position a consumer will claim. Advanced via CAS by
    /// consumers; cache-line isolated from `enqueue_pos`.
    dequeue_pos: CachePadded,
}

impl RingBuffer {
    /// Creates a ring whose capacity is `capacity` rounded **up to the next
    /// power of two** (minimum 1).
    ///
    /// The slot array is allocated here, once; the hot path performs no further
    /// allocation. Each cell `i` is stamped with `sequence == i` to bootstrap
    /// the Vyukov protocol (every slot starts empty and "expecting" lap 0).
    pub fn new(capacity: usize) -> Self {
        // Round up to a power of two so `pos & mask` indexes a slot. `max(1)`
        // guards the degenerate request of 0; `next_power_of_two()` of 1 is 1.
        let cap = capacity.max(1).next_power_of_two();

        // Build the slot array with cell[i].sequence = i. Using a Vec we
        // `collect` into a boxed slice gives us a single heap allocation with no
        // need for `WriteRequest: Default` (the payload stays uninitialised).
        let buffer: Box<[Cell]> = (0..cap)
            .map(|i| Cell {
                sequence: AtomicUsize::new(i),
                slot: UnsafeCell::new(MaybeUninit::uninit()),
            })
            .collect();

        RingBuffer {
            buffer,
            mask: cap - 1,
            capacity: cap,
            enqueue_pos: CachePadded(AtomicUsize::new(0)),
            dequeue_pos: CachePadded(AtomicUsize::new(0)),
        }
    }
}

impl Queue for RingBuffer {
    /// Attempts to enqueue `req` without blocking.
    ///
    /// Lock-free: loops only when it loses a CAS race to another producer (each
    /// such loss implies that producer made progress). On a full ring returns
    /// `Err(req)`, handing the item straight back so the caller can shed load.
    fn try_enqueue(&self, req: WriteRequest) -> Result<(), WriteRequest> {
        // Snapshot the producer position; we'll try to claim slot `pos & mask`.
        let mut pos = self.enqueue_pos.0.load(Ordering::Relaxed);

        loop {
            let cell = &self.buffer[pos & self.mask];
            // Acquire: synchronises with a consumer's Release store of the
            // "slot freed" stamp, so that once we observe an empty slot we also
            // observe that the previous occupant's read has completed.
            let seq = cell.sequence.load(Ordering::Acquire);
            // Signed compare: how far the slot's stamp is from our position.
            let diff = seq as isize - pos as isize;

            if diff == 0 {
                // Slot is empty and expecting exactly this position. Try to
                // claim `pos` by advancing the shared producer counter. Relaxed
                // is sufficient: the actual data publication is the Release
                // store on `cell.sequence` below; the counter only arbitrates
                // *which* producer owns *which* position.
                match self.enqueue_pos.0.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // We exclusively own this slot now: no other producer
                        // can have claimed this `pos`, and no consumer will
                        // touch the payload until we publish the stamp below.
                        // SAFETY: The CAS above succeeded, so this thread is the
                        // unique owner of logical position `pos`, and the
                        // `diff == 0` test proved the cell is in the empty state
                        // (its previous occupant, if any, was fully read and the
                        // slot freed). No other thread reads or writes this
                        // `MaybeUninit` until we store the published stamp, so
                        // writing through the `UnsafeCell` here is race-free.
                        unsafe {
                            (*cell.slot.get()).write(req);
                        }
                        // Release: publishes the payload write above to the
                        // consumer that will load this stamp with Acquire. The
                        // value `pos + 1` is exactly what a consumer at `pos`
                        // waits for, transferring ownership of the slot to it.
                        cell.sequence.store(pos.wrapping_add(1), Ordering::Release);
                        return Ok(());
                    }
                    // Lost the race; another producer took `pos`. Reload the
                    // observed counter value and retry from the top.
                    Err(actual) => pos = actual,
                }
            } else if diff < 0 {
                // The slot's stamp is *behind* our position: the slot is still
                // occupied by an item from the previous lap that no consumer has
                // drained yet. That means the ring is full. Hand the item back.
                return Err(req);
            } else {
                // diff > 0: another producer has already advanced past `pos`
                // (the global counter moved on). Reload and retry against the
                // new front. This is the benign "someone else made progress"
                // branch that keeps the algorithm lock-free rather than
                // wait-free.
                pos = self.enqueue_pos.0.load(Ordering::Relaxed);
            }
        }
    }

    /// Pops a single item if one is available, without blocking.
    ///
    /// Symmetric to [`try_enqueue`](Self::try_enqueue) against `dequeue_pos`.
    /// Returns `None` on an empty ring.
    fn try_dequeue(&self) -> Option<WriteRequest> {
        // Snapshot the consumer position; we'll try to claim slot `pos & mask`.
        let mut pos = self.dequeue_pos.0.load(Ordering::Relaxed);

        loop {
            let cell = &self.buffer[pos & self.mask];
            // Acquire: synchronises with the producer's Release publish, so once
            // we see the "published" stamp we also see the payload it wrote.
            let seq = cell.sequence.load(Ordering::Acquire);
            // For consumers the target stamp is `pos + 1` (the value a producer
            // stores after publishing), so compare against that.
            let diff = seq as isize - (pos.wrapping_add(1)) as isize;

            if diff == 0 {
                // Slot holds a published item destined for this position. Claim
                // `pos` by advancing the shared consumer counter (Relaxed: data
                // visibility is handled by the Acquire load above and the
                // Release store below).
                match self.dequeue_pos.0.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // We exclusively own this slot's payload now.
                        // SAFETY: The CAS succeeded, so this thread uniquely owns
                        // logical position `pos`, and the `diff == 0` test (stamp
                        // == pos + 1) proved a producer has fully published this
                        // slot with a Release store that our Acquire load above
                        // synchronised with — therefore the `MaybeUninit` is
                        // initialised. No other consumer can observe `pos` again,
                        // and no producer will reuse the slot until we free it
                        // with the Release store below, so reading the value out
                        // by `assume_init_read` (a move) is race-free and does
                        // not double-read.
                        let req = unsafe { (*cell.slot.get()).assume_init_read() };
                        // Release: frees the slot for the producer one full lap
                        // ahead (logical position `pos + capacity`), which waits
                        // for exactly this stamp `pos + mask + 1`.
                        cell.sequence.store(
                            pos.wrapping_add(self.mask).wrapping_add(1),
                            Ordering::Release,
                        );
                        return Some(req);
                    }
                    // Lost the race to another consumer; reload and retry.
                    Err(actual) => pos = actual,
                }
            } else if diff < 0 {
                // The slot's stamp has not yet reached `pos + 1`: no producer has
                // published for this position. The ring is empty (from this
                // consumer's vantage point). Nothing to pop.
                return None;
            } else {
                // diff > 0: another consumer already advanced past `pos`. Reload
                // the counter and retry against the new front.
                pos = self.dequeue_pos.0.load(Ordering::Relaxed);
            }
        }
    }

    /// Bulk-pops up to `max` items into `out`, returning the number moved.
    ///
    /// Used by the worker to amortise the per-item synchronisation across a
    /// batch. Stops early when the ring drains empty. Each pop is an independent
    /// lock-free [`try_dequeue`](Self::try_dequeue); a concurrent producer may
    /// add more after we observe empty, which is fine — the worker loops again.
    fn drain_into(&self, out: &mut Vec<WriteRequest>, max: usize) -> usize {
        let mut count = 0;
        while count < max {
            match self.try_dequeue() {
                Some(req) => {
                    out.push(req);
                    count += 1;
                }
                None => break,
            }
        }
        count
    }

    /// Approximate number of queued items.
    ///
    /// Computed as `enqueue_pos - dequeue_pos`. Because the two counters are read
    /// independently (no global lock), the result is a *snapshot estimate* under
    /// concurrency: it can momentarily appear negative (a dequeue advanced
    /// between our two loads) — which we saturate to `0` — or exceed capacity,
    /// which we clamp to `capacity`. It is exact when the ring is quiescent.
    fn len(&self) -> usize {
        // Load dequeue first, then enqueue. With this order, concurrent activity
        // tends to make `enqueue >= dequeue` (we read the laggard first), but we
        // still guard both extremes defensively.
        let deq = self.dequeue_pos.0.load(Ordering::Relaxed);
        let enq = self.enqueue_pos.0.load(Ordering::Relaxed);
        // `wrapping_sub` then clamp: if a race made `deq > enq`, the wrap yields
        // a huge value which the `min(capacity)` clamp pins to `capacity`; the
        // separate `enq < deq` check maps that case to a saturating `0` so we
        // never report a bogus "full" for what is really "empty".
        if enq <= deq {
            0
        } else {
            (enq - deq).min(self.capacity)
        }
    }

    /// Fixed capacity (the requested value rounded up to a power of two).
    fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Drop for RingBuffer {
    /// Drains and drops every still-initialised slot so the [`WriteRequest`]
    /// payloads (each owning a `Vec<u8>` blob) are not leaked.
    ///
    /// [`MaybeUninit`] does **not** drop its contents automatically; a slot only
    /// holds a live value while it sits between a producer's publish and a
    /// consumer's read. We replay exactly that window — the logical range
    /// `[dequeue_pos, enqueue_pos)` — and drop each occupied slot once.
    fn drop(&mut self) {
        // By the time `drop` runs we hold `&mut self`: there are no other
        // producers or consumers, so plain (non-atomic) reasoning is sound. We
        // still read through the atomics for their value.
        let deq = self.dequeue_pos.0.load(Ordering::Relaxed);
        let enq = self.enqueue_pos.0.load(Ordering::Relaxed);

        let mut pos = deq;
        while pos != enq {
            let cell = &self.buffer[pos & self.mask];
            // SAFETY: We have exclusive (`&mut self`) access, so no concurrent
            // thread touches these slots. Every logical position in
            // `[dequeue_pos, enqueue_pos)` corresponds to a slot a producer
            // published but no consumer has read, so its `MaybeUninit` is
            // initialised exactly once. We visit each such position exactly once
            // (the half-open range has no repeats within one lap, and the ring
            // can hold at most `capacity` live items so the range length is
            // `<= capacity`), so dropping the value here neither double-drops nor
            // drops an uninitialised slot.
            unsafe {
                let slot = &mut *cell.slot.get();
                slot.assume_init_drop();
            }
            pos = pos.wrapping_add(1);
        }
    }
}

// SAFETY: `RingBuffer` is safe to send across threads and to share by reference
// across threads. The `UnsafeCell<MaybeUninit<WriteRequest>>` payloads opt the
// type out of the auto `Send`/`Sync` impls, so we assert them manually:
//
//  * `Send`: moving the whole ring to another thread moves its owned slot
//    payloads with it. Each payload is a `WriteRequest`, which is `Send`, so the
//    aggregate is safe to send. (We require `WriteRequest: Send`, which holds —
//    it is `u64`s plus a `Vec<u8>`.)
//  * `Sync`: sharing `&RingBuffer` across threads only ever exposes the
//    lock-free `Queue` methods. The Vyukov sequence protocol guarantees that at
//    most one thread accesses any given slot's payload at any time, and only
//    after the Acquire/Release handshake on `sequence` has transferred
//    ownership — so concurrent `&self` access never races on the `UnsafeCell`
//    contents. All counter access is via atomics. Hence `&RingBuffer` is safe to
//    share, which is exactly `Sync`.
//
// Both rely on `WriteRequest: Send`; the queue never hands a `&WriteRequest`
// across threads, so `WriteRequest: Sync` is not required.
unsafe impl Send for RingBuffer {}
// SAFETY: see the block comment above `unsafe impl Send`.
unsafe impl Sync for RingBuffer {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::thread;

    /// Builds a `WriteRequest` whose `seq` doubles as a unique identity token in
    /// the multi-threaded test below.
    fn req(seq: u64) -> WriteRequest {
        WriteRequest {
            player_id: seq,
            version: 1,
            blob: vec![(seq & 0xFF) as u8],
            seq,
        }
    }

    #[test]
    fn single_thread_enqueue_dequeue_full_empty() {
        // Request 3 → rounds up to capacity 4.
        let ring = RingBuffer::new(3);
        assert_eq!(ring.capacity(), 4);
        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);

        // Empty ring yields nothing.
        assert!(ring.try_dequeue().is_none());

        // Fill it exactly to capacity.
        for i in 0..4 {
            assert!(ring.try_enqueue(req(i)).is_ok(), "enqueue {i} should fit");
        }
        assert_eq!(ring.len(), 4);

        // One more must be rejected (full) and hand the item back unchanged.
        let overflow = ring.try_enqueue(req(99));
        match overflow {
            Err(returned) => assert_eq!(returned.seq, 99),
            Ok(()) => panic!("ring should have been full"),
        }

        // FIFO drain returns the items in submission order.
        for i in 0..4 {
            let got = ring.try_dequeue().expect("item present");
            assert_eq!(got.seq, i, "FIFO order");
        }

        // Drained back to empty.
        assert!(ring.is_empty());
        assert!(ring.try_dequeue().is_none());

        // Reuse after a full lap still works (exercises slot recycling).
        assert!(ring.try_enqueue(req(123)).is_ok());
        assert_eq!(ring.try_dequeue().unwrap().seq, 123);

        // drain_into bulk-pops and reports the count.
        for i in 0..4 {
            ring.try_enqueue(req(i)).unwrap();
        }
        let mut out = Vec::new();
        let n = ring.drain_into(&mut out, 10);
        assert_eq!(n, 4);
        assert_eq!(out.len(), 4);
        assert!(ring.is_empty());
    }

    #[test]
    fn mpmc_four_producers_one_consumer_no_loss_no_dup() {
        const PRODUCERS: u64 = 4;
        const PER_PRODUCER: u64 = 50_000;
        const TOTAL: u64 = PRODUCERS * PER_PRODUCER;

        // Deliberately small ring so producers genuinely contend AND hit the
        // "full" path, forcing them to retry — this stresses the CAS loop and
        // the back-pressure branch rather than letting everything fit.
        let ring = Arc::new(RingBuffer::new(64));

        // Producer p emits seqs [p*PER_PRODUCER .. (p+1)*PER_PRODUCER), so every
        // emitted seq across all threads is globally unique.
        let mut producers = Vec::new();
        for p in 0..PRODUCERS {
            let ring = Arc::clone(&ring);
            producers.push(thread::spawn(move || {
                let base = p * PER_PRODUCER;
                for i in 0..PER_PRODUCER {
                    let mut item = req(base + i);
                    // Spin-retry on a full ring: the consumer will catch up.
                    // This is the test harness choosing to NOT shed load, so we
                    // can assert exact no-loss semantics.
                    loop {
                        match ring.try_enqueue(item) {
                            Ok(()) => break,
                            Err(returned) => {
                                item = returned;
                                std::thread::yield_now();
                            }
                        }
                    }
                }
            }));
        }

        // Single consumer collects until it has seen all TOTAL items.
        let consumer = {
            let ring = Arc::clone(&ring);
            thread::spawn(move || {
                let mut seen: HashSet<u64> = HashSet::with_capacity(TOTAL as usize);
                while (seen.len() as u64) < TOTAL {
                    if let Some(item) = ring.try_dequeue() {
                        // Insert returns false if the seq was already seen → a
                        // duplicate delivery, which must never happen.
                        assert!(
                            seen.insert(item.seq),
                            "duplicate delivery of seq {}",
                            item.seq
                        );
                    } else {
                        std::thread::yield_now();
                    }
                }
                seen
            })
        };

        for h in producers {
            h.join().expect("producer thread panicked");
        }
        let seen = consumer.join().expect("consumer thread panicked");

        // No loss: exactly TOTAL distinct items, covering the full seq range.
        assert_eq!(seen.len() as u64, TOTAL, "lost or duplicated items");
        for s in 0..TOTAL {
            assert!(seen.contains(&s), "missing seq {s}");
        }

        // Ring must be fully drained at the end.
        assert!(ring.is_empty(), "ring not empty after draining all items");
    }
}
