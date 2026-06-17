//! Tunable configuration for a [`crate::api::FluxEngine`] instance.
//!
//! [`FluxConfig`] groups every knob that shapes the throughput/latency/durability
//! trade-offs of the pipeline: the size of the lock-free ring, the **volume**
//! and **time** thresholds that trigger a flush, how the worker parks when idle,
//! and the WAL/store durability and retry policy. All fields are public so tests
//! and embedders can build a config literally; [`FluxConfig::validate`] enforces
//! the invariants the rest of the crate relies on.

use std::path::PathBuf;
use std::time::Duration;

/// Configuration controlling ring sizing, flush cadence, WAL durability and
/// store retry behaviour. Construct via [`FluxConfig::default`] and adjust, or
/// build a literal and call [`FluxConfig::validate`].
#[derive(Clone, Debug)]
pub struct FluxConfig {
    /// Desired ring capacity. The ring rounds this **up to the next power of
    /// two** so it can mask instead of modulo. Must be `>= 1`.
    pub ring_capacity: usize,

    /// VOLUME flush threshold: the worker flushes once the cache accumulates at
    /// least this many dirty entries, regardless of elapsed time.
    pub flush_max_batch: usize,

    /// TIME flush threshold: the worker flushes at least this often even if the
    /// volume threshold has not been reached, bounding write latency.
    pub flush_interval: Duration,

    /// How long the worker parks when there is nothing to do, trading a little
    /// latency for far lower idle CPU usage.
    pub worker_idle_sleep: Duration,

    /// Directory holding the write-ahead log files.
    pub wal_dir: PathBuf,

    /// When `true`, [`crate::domain::WriteAheadLog::append_batch`] fsyncs before
    /// returning — the strong durability mode. When `false`, appends are only
    /// buffered (faster, weaker guarantee).
    pub wal_fsync: bool,

    /// Maximum number of retry attempts the worker makes against the store for
    /// a retryable error before requeuing the dirty set.
    pub store_retry_limit: u32,

    /// Base backoff slept between store retry attempts.
    pub store_retry_backoff: Duration,
}

impl Default for FluxConfig {
    /// Production-leaning defaults:
    /// ring `1 << 16` (65 536), batch `4096`, interval `50ms`, idle sleep `1ms`,
    /// fsync `true`, retry limit `5`, backoff `2ms`, WAL dir `./flux-wal`.
    fn default() -> Self {
        FluxConfig {
            ring_capacity: 1 << 16,
            flush_max_batch: 4096,
            flush_interval: Duration::from_millis(50),
            worker_idle_sleep: Duration::from_millis(1),
            wal_dir: PathBuf::from("./flux-wal"),
            wal_fsync: true,
            store_retry_limit: 5,
            store_retry_backoff: Duration::from_millis(2),
        }
    }
}

impl FluxConfig {
    /// Returns a copy of `self` with the WAL directory replaced.
    ///
    /// Convenience for tests that point each engine at a fresh temp dir.
    pub fn with_wal_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.wal_dir = dir.into();
        self
    }

    /// Validates the configuration invariants the rest of the crate assumes.
    ///
    /// # Errors
    /// Returns a human-readable message if any of the following hold:
    /// * `ring_capacity == 0`
    /// * `flush_max_batch == 0`
    /// * `flush_interval == 0`
    /// * `store_retry_limit == 0`
    pub fn validate(&self) -> Result<(), String> {
        if self.ring_capacity == 0 {
            return Err("ring_capacity must be >= 1".to_string());
        }
        if self.flush_max_batch == 0 {
            return Err("flush_max_batch must be >= 1".to_string());
        }
        if self.flush_interval.is_zero() {
            return Err("flush_interval must be > 0".to_string());
        }
        if self.store_retry_limit == 0 {
            return Err("store_retry_limit must be >= 1".to_string());
        }
        Ok(())
    }
}
