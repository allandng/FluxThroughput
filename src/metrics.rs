//! Wait-free pipeline metrics.
//!
//! [`Metrics`] is shared (behind an `Arc`) between the hot submit path and the
//! background worker. Every counter is an [`AtomicU64`] updated with `Relaxed`
//! ordering — there is **no `Mutex` anywhere** in this module, so recording a
//! metric can never block the main loop or the worker. "Max" gauges (peak submit
//! latency, peak flush latency) are maintained with a compare-and-swap
//! fetch-max loop because `AtomicU64` has no native `fetch_max` on all targets
//! and we want a portable, lock-free implementation.
//!
//! [`Metrics::snapshot`] takes a consistent-enough point-in-time copy (each
//! counter read independently with `Relaxed`) into a plain
//! [`MetricsSnapshot`] for reporting; minor skew between counters is acceptable
//! for observability and is the price of staying lock-free.

use std::sync::atomic::{AtomicU64, Ordering};

/// Lock-free, wait-free collection of pipeline counters and gauges.
///
/// Cheap to clone the `Arc` around; every method is a handful of relaxed atomic
/// operations. See the module docs for the no-`Mutex` guarantee.
pub struct Metrics {
    submits_enqueued: AtomicU64,
    submits_dropped: AtomicU64,
    max_submit_nanos: AtomicU64,
    sum_submit_nanos: AtomicU64,
    flushes: AtomicU64,
    records_flushed: AtomicU64,
    max_flush_nanos: AtomicU64,
    wal_appends: AtomicU64,
    wal_records: AtomicU64,
    store_errors: AtomicU64,
    store_retries: AtomicU64,
}

impl Metrics {
    /// Creates a fresh, all-zero metrics collector.
    pub fn new() -> Self {
        Metrics {
            submits_enqueued: AtomicU64::new(0),
            submits_dropped: AtomicU64::new(0),
            max_submit_nanos: AtomicU64::new(0),
            sum_submit_nanos: AtomicU64::new(0),
            flushes: AtomicU64::new(0),
            records_flushed: AtomicU64::new(0),
            max_flush_nanos: AtomicU64::new(0),
            wal_appends: AtomicU64::new(0),
            wal_records: AtomicU64::new(0),
            store_errors: AtomicU64::new(0),
            store_retries: AtomicU64::new(0),
        }
    }

    /// Records a successful enqueue from the submit path along with the time, in
    /// nanoseconds, the submit call took. Updates count, running sum, and the
    /// peak-latency gauge.
    pub fn record_submit_enqueued(&self, submit_nanos: u64) {
        self.submits_enqueued.fetch_add(1, Ordering::Relaxed);
        self.sum_submit_nanos
            .fetch_add(submit_nanos, Ordering::Relaxed);
        fetch_max(&self.max_submit_nanos, submit_nanos);
    }

    /// Records a load-shed drop (ring full) on the submit path.
    pub fn record_submit_dropped(&self) {
        self.submits_dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a completed flush of `batch_len` records taking `flush_nanos`.
    pub fn record_flush(&self, batch_len: usize, flush_nanos: u64) {
        self.flushes.fetch_add(1, Ordering::Relaxed);
        self.records_flushed
            .fetch_add(batch_len as u64, Ordering::Relaxed);
        fetch_max(&self.max_flush_nanos, flush_nanos);
    }

    /// Records a WAL append of `records` frames taking `nanos`.
    pub fn record_wal_append(&self, records: usize, _nanos: u64) {
        self.wal_appends.fetch_add(1, Ordering::Relaxed);
        self.wal_records
            .fetch_add(records as u64, Ordering::Relaxed);
    }

    /// Records a store error (after retries were exhausted or a fatal error).
    pub fn record_store_error(&self) {
        self.store_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a single store retry attempt.
    pub fn record_store_retry(&self) {
        self.store_retries.fetch_add(1, Ordering::Relaxed);
    }

    /// Takes a point-in-time copy of all counters into a [`MetricsSnapshot`].
    ///
    /// Each field is read independently with `Relaxed`, so the snapshot may
    /// reflect tiny skew between counters under concurrent updates — acceptable
    /// for observability and required to stay lock-free.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            submits_enqueued: self.submits_enqueued.load(Ordering::Relaxed),
            submits_dropped: self.submits_dropped.load(Ordering::Relaxed),
            max_submit_nanos: self.max_submit_nanos.load(Ordering::Relaxed),
            sum_submit_nanos: self.sum_submit_nanos.load(Ordering::Relaxed),
            flushes: self.flushes.load(Ordering::Relaxed),
            records_flushed: self.records_flushed.load(Ordering::Relaxed),
            max_flush_nanos: self.max_flush_nanos.load(Ordering::Relaxed),
            wal_appends: self.wal_appends.load(Ordering::Relaxed),
            wal_records: self.wal_records.load(Ordering::Relaxed),
            store_errors: self.store_errors.load(Ordering::Relaxed),
            store_retries: self.store_retries.load(Ordering::Relaxed),
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics::new()
    }
}

/// Atomically updates `target` to `max(target, value)` using a lock-free
/// compare-and-swap loop.
///
/// The loop reloads on contention (`Err`) and exits early once the existing
/// value already dominates `value`, so it performs at most one successful CAS
/// per genuinely-new maximum.
fn fetch_max(target: &AtomicU64, value: u64) {
    let mut current = target.load(Ordering::Relaxed);
    while value > current {
        match target.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
}

/// An immutable, point-in-time copy of the pipeline metrics.
#[derive(Clone, Debug, Default)]
pub struct MetricsSnapshot {
    /// Total writes successfully enqueued by `submit`.
    pub submits_enqueued: u64,
    /// Total writes dropped (ring full) by `submit`.
    pub submits_dropped: u64,
    /// Peak observed `submit` latency, in nanoseconds.
    pub max_submit_nanos: u64,
    /// Sum of all observed `submit` latencies, in nanoseconds.
    pub sum_submit_nanos: u64,
    /// Number of flush cycles completed by the worker.
    pub flushes: u64,
    /// Total records persisted across all flushes.
    pub records_flushed: u64,
    /// Peak observed flush duration, in nanoseconds.
    pub max_flush_nanos: u64,
    /// Number of WAL append calls.
    pub wal_appends: u64,
    /// Total records written to the WAL.
    pub wal_records: u64,
    /// Number of store errors (post-retry / fatal).
    pub store_errors: u64,
    /// Number of individual store retry attempts.
    pub store_retries: u64,
}

impl MetricsSnapshot {
    /// Average `submit` latency in nanoseconds, guarding against division by
    /// zero (returns `0.0` when no submits have been enqueued).
    pub fn avg_submit_nanos(&self) -> f64 {
        if self.submits_enqueued == 0 {
            0.0
        } else {
            self.sum_submit_nanos as f64 / self.submits_enqueued as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetch_max_keeps_peak() {
        let m = Metrics::new();
        m.record_submit_enqueued(10);
        m.record_submit_enqueued(50);
        m.record_submit_enqueued(30);
        let s = m.snapshot();
        assert_eq!(s.submits_enqueued, 3);
        assert_eq!(s.max_submit_nanos, 50);
        assert_eq!(s.sum_submit_nanos, 90);
        assert!((s.avg_submit_nanos() - 30.0).abs() < 1e-9);
    }

    #[test]
    fn avg_guards_zero() {
        let s = MetricsSnapshot::default();
        assert_eq!(s.avg_submit_nanos(), 0.0);
    }
}
