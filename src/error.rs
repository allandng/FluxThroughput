//! Error taxonomy for the FluxThroughput pipeline.
//!
//! The crate distinguishes three orthogonal failure domains and keeps them in
//! separate types so the *worker* can make precise retry/requeue decisions
//! without leaking I/O concerns onto the hot submit path:
//!
//! * [`StoreError`] — failures originating in the durable backend
//!   ([`crate::domain::PersistenceStore`]). These carry a *retryability*
//!   classification that the flush worker uses to decide between retrying with
//!   backoff and requeuing the dirty set so that **no update is ever lost**.
//! * [`DecodeError`] — failures decoding a write-ahead-log frame
//!   ([`crate::domain::WalRecord::decode`]). A torn or corrupt tail frame is the
//!   expected outcome of a hard kill mid-write; recovery treats these as a
//!   clean truncation point rather than a fatal error.
//! * [`FluxError`] — the top-level error surfaced by the [`crate::api::FluxEngine`]
//!   facade for the synchronous operations (`flush_now`, `shutdown`). It unifies
//!   `std::io::Error` and [`StoreError`].
//!
//! Every error type here implements [`std::fmt::Display`] and
//! [`std::error::Error`], and all conversions are `From`-based so callers can
//! use `?` freely.

use std::error::Error;
use std::fmt;

/// A failure reported by a [`crate::domain::PersistenceStore`] implementation.
///
/// The variants encode the *kind* of failure so the flush worker can classify
/// it as retryable or not via [`StoreError::is_retryable`]:
///
/// * [`StoreError::Unavailable`] — the backend is temporarily unreachable
///   (e.g. connection refused, leader election in progress). **Retryable.**
/// * [`StoreError::Transient`] — a request-scoped error that is expected to
///   succeed on a later attempt (e.g. timeout, throttling). **Retryable.**
/// * [`StoreError::Fatal`] — a permanent error that retrying cannot fix
///   (e.g. schema violation, serialization bug). **Not retryable** — the worker
///   requeues the dirty set so the data survives in memory/WAL for operator
///   intervention rather than being silently dropped.
#[derive(Debug, Clone)]
pub enum StoreError {
    /// Backend temporarily unreachable. Retryable.
    Unavailable,
    /// Transient, request-scoped failure with a human-readable cause. Retryable.
    Transient(String),
    /// Permanent failure with a human-readable cause. NOT retryable.
    Fatal(String),
}

impl StoreError {
    /// Returns `true` if the flush worker should retry the operation.
    ///
    /// [`StoreError::Unavailable`] and [`StoreError::Transient`] are retryable;
    /// [`StoreError::Fatal`] is not.
    pub fn is_retryable(&self) -> bool {
        matches!(self, StoreError::Unavailable | StoreError::Transient(_))
    }
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::Unavailable => write!(f, "store unavailable (retryable)"),
            StoreError::Transient(msg) => write!(f, "transient store error (retryable): {msg}"),
            StoreError::Fatal(msg) => write!(f, "fatal store error (not retryable): {msg}"),
        }
    }
}

impl Error for StoreError {}

/// A failure decoding a WAL frame produced by [`crate::domain::WalRecord::encode`].
///
/// During [`crate::domain::WriteAheadLog::recover`] a `Truncated` or `BadCrc`
/// error at the *tail* of the log is the normal signature of a process killed
/// mid-append; recovery stops at that frame and truncates the file to the last
/// fully-valid offset. A `BadMagic` mid-stream indicates genuine corruption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// The buffer ended before a complete frame could be read.
    Truncated,
    /// The trailing CRC32 did not match the recomputed checksum.
    BadCrc,
    /// The leading magic number did not match the expected frame magic.
    BadMagic,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::Truncated => write!(f, "truncated WAL frame"),
            DecodeError::BadCrc => write!(f, "WAL frame CRC mismatch"),
            DecodeError::BadMagic => write!(f, "WAL frame magic mismatch"),
        }
    }
}

impl Error for DecodeError {}

/// The top-level error surfaced by synchronous [`crate::api::FluxEngine`] calls.
///
/// Unifies the two failure domains that the engine's blocking operations can
/// hit: filesystem/WAL I/O ([`std::io::Error`]) and the durable backend
/// ([`StoreError`]).
#[derive(Debug)]
pub enum FluxError {
    /// An underlying filesystem or WAL I/O error.
    Io(std::io::Error),
    /// A durable-store error.
    Store(StoreError),
}

impl fmt::Display for FluxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FluxError::Io(e) => write!(f, "io error: {e}"),
            FluxError::Store(e) => write!(f, "store error: {e}"),
        }
    }
}

impl Error for FluxError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            FluxError::Io(e) => Some(e),
            FluxError::Store(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for FluxError {
    fn from(e: std::io::Error) -> Self {
        FluxError::Io(e)
    }
}

impl From<StoreError> for FluxError {
    fn from(e: StoreError) -> Self {
        FluxError::Store(e)
    }
}
