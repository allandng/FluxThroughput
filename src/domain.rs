//! Core domain model: value types, the WAL wire format, and the four traits
//! that decouple the pipeline stages.
//!
//! This module is the shared vocabulary of the crate. It is deliberately free
//! of any I/O, threading, or locking so that every other module can depend on
//! it without taking on those concerns. The four traits — [`Queue`],
//! [`WriteAheadLog`], [`PersistenceStore`] and [`Cache`] — are the seams along
//! which the system is decomposed:
//!
//! ```text
//!   submit() ──▶ Queue (ring) ──▶ Cache (coalesce) ──▶ WriteAheadLog ──▶ PersistenceStore
//!   (hot loop)   lock-free        worker thread        durability         slow backend
//! ```
//!
//! The hot main loop only ever touches [`Queue::try_enqueue`]. Everything to
//! the right of the ring runs on the background worker thread, which is the
//! only place blocking I/O and locking are permitted.
//!
//! ## WAL wire format
//!
//! [`WalRecord::encode`] / [`WalRecord::decode`] define a **self-describing,
//! CRC-protected frame** so that recovery after an unclean shutdown can locate
//! record boundaries and detect a torn tail. See [`WalRecord`] for the exact
//! byte layout. The CRC is computed with the pure-Rust IEEE [`crc32`] helper
//! living in this module — a single source of truth shared by `domain.rs` and
//! `wal.rs` so the two can never disagree about the checksum polynomial.

use crate::error::DecodeError;

/// Stable identity of a player / entity whose state is being persisted.
pub type PlayerId = u64;

/// Log Sequence Number — a monotonically increasing position in the WAL.
pub type Lsn = u64;

/// The canonical, persisted form of a single entity's state.
///
/// `version` is the application-level optimistic-concurrency version used by the
/// cache to implement **last-write-wins**: an [`apply`](Cache::apply) with a
/// lower-or-equal version than what is already cached is ignored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlayerState {
    /// Identity of the entity.
    pub player_id: PlayerId,
    /// Application-level version; higher wins.
    pub version: u64,
    /// Opaque serialized payload.
    pub blob: Vec<u8>,
}

/// A single fire-and-forget write as it travels through the ring.
///
/// `seq` is assigned by [`crate::api::FluxEngine::submit`] via a relaxed atomic
/// fetch-add and provides a global submission order independent of `version`.
/// It is used for diagnostics / tie-breaking and is *not* part of the persisted
/// state (see [`WriteRequest::to_state`]).
#[derive(Clone, Debug)]
pub struct WriteRequest {
    /// Identity of the entity being written.
    pub player_id: PlayerId,
    /// Application-level version; higher wins in the cache.
    pub version: u64,
    /// Opaque serialized payload.
    pub blob: Vec<u8>,
    /// Global submission sequence number assigned at submit time.
    pub seq: u64,
}

impl WriteRequest {
    /// Projects this request to its persisted [`PlayerState`], dropping `seq`.
    pub fn to_state(&self) -> PlayerState {
        PlayerState {
            player_id: self.player_id,
            version: self.version,
            blob: self.blob.clone(),
        }
    }
}

/// One durably-logged write, as it appears in the write-ahead log.
///
/// ## Frame layout (all integers little-endian)
///
/// ```text
/// offset  size  field
/// ------  ----  -----------------------------------------------------------
///   0      4    magic        = 0x464C5852  ("FLXR")
///   4      4    frame_len    = total frame length in bytes (incl. crc field)
///   8      8    lsn
///  16      8    player_id
///  24      8    version
///  32      4    blob_len
///  36   blob_len blob bytes
///   ..     4    crc32        = IEEE CRC32 of every byte preceding this field
/// ```
///
/// The fixed header is therefore 36 bytes, the payload is `blob_len` bytes, and
/// the trailing CRC is 4 bytes, so `frame_len == 36 + blob_len + 4`.
///
/// The CRC covers the entire frame *up to but not including* the CRC field,
/// i.e. `frame_len - 4` bytes. Because `frame_len` is itself inside the CRC'd
/// region, a torn write that corrupts the length is caught either as a
/// [`DecodeError::Truncated`] (not enough bytes for the claimed length) or a
/// [`DecodeError::BadCrc`] (length intact but body damaged).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WalRecord {
    /// Log sequence number assigned to this record.
    pub lsn: Lsn,
    /// Identity of the entity.
    pub player_id: PlayerId,
    /// Application-level version.
    pub version: u64,
    /// Opaque serialized payload.
    pub blob: Vec<u8>,
}

/// Magic number prefixing every WAL frame: ASCII "FLXR" in little-endian.
const WAL_MAGIC: u32 = 0x464C_5852;

/// Size of the fixed frame header preceding the blob:
/// magic(4) + frame_len(4) + lsn(8) + player_id(8) + version(8) + blob_len(4).
const WAL_HEADER_LEN: usize = 4 + 4 + 8 + 8 + 8 + 4;

/// Size of the trailing CRC field.
const WAL_CRC_LEN: usize = 4;

impl WalRecord {
    /// Serializes this record into a single self-describing, CRC-protected frame.
    ///
    /// The returned buffer is exactly `36 + blob.len() + 4` bytes (see the
    /// [type-level layout](WalRecord)) and is the unit that
    /// [`WriteAheadLog::append_batch`] writes to disk.
    pub fn encode(&self) -> Vec<u8> {
        let blob_len = self.blob.len();
        let frame_len = WAL_HEADER_LEN + blob_len + WAL_CRC_LEN;
        let mut buf = Vec::with_capacity(frame_len);

        buf.extend_from_slice(&WAL_MAGIC.to_le_bytes());
        buf.extend_from_slice(&(frame_len as u32).to_le_bytes());
        buf.extend_from_slice(&self.lsn.to_le_bytes());
        buf.extend_from_slice(&self.player_id.to_le_bytes());
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&(blob_len as u32).to_le_bytes());
        buf.extend_from_slice(&self.blob);

        // CRC over everything written so far (the whole frame minus the CRC).
        let crc = crc32(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());

        debug_assert_eq!(buf.len(), frame_len);
        buf
    }

    /// Decodes a single frame from the front of `buf`.
    ///
    /// Returns the decoded [`WalRecord`] and the number of bytes consumed so the
    /// caller can advance through a concatenated stream of frames.
    ///
    /// # Errors
    /// * [`DecodeError::Truncated`] if `buf` does not contain a complete frame.
    /// * [`DecodeError::BadMagic`] if the leading magic is wrong.
    /// * [`DecodeError::BadCrc`] if the trailing CRC does not match.
    pub fn decode(buf: &[u8]) -> Result<(WalRecord, usize), DecodeError> {
        // Need at least the fixed header to learn the frame length.
        if buf.len() < WAL_HEADER_LEN {
            return Err(DecodeError::Truncated);
        }

        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        if magic != WAL_MAGIC {
            return Err(DecodeError::BadMagic);
        }

        let frame_len = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;

        // A frame must be at least header + crc; a smaller claim is corruption
        // we treat as truncation (the length itself is untrustworthy).
        if frame_len < WAL_HEADER_LEN + WAL_CRC_LEN {
            return Err(DecodeError::Truncated);
        }
        if buf.len() < frame_len {
            return Err(DecodeError::Truncated);
        }

        let blob_len = u32::from_le_bytes(buf[32..36].try_into().unwrap()) as usize;
        // The declared blob length must be consistent with the declared frame
        // length. If not, the length fields are corrupt: treat as a torn tail.
        if frame_len != WAL_HEADER_LEN + blob_len + WAL_CRC_LEN {
            return Err(DecodeError::Truncated);
        }

        // Validate the CRC over the frame body (everything before the CRC field).
        let body_end = frame_len - WAL_CRC_LEN;
        let stored_crc = u32::from_le_bytes(buf[body_end..frame_len].try_into().unwrap());
        let actual_crc = crc32(&buf[..body_end]);
        if stored_crc != actual_crc {
            return Err(DecodeError::BadCrc);
        }

        let lsn = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let player_id = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        let version = u64::from_le_bytes(buf[24..32].try_into().unwrap());
        let blob = buf[WAL_HEADER_LEN..WAL_HEADER_LEN + blob_len].to_vec();

        Ok((
            WalRecord {
                lsn,
                player_id,
                version,
                blob,
            },
            frame_len,
        ))
    }
}

/// Pure-Rust IEEE 802.3 (reflected polynomial `0xEDB88320`) CRC-32.
///
/// This is the **single source of truth** for the checksum used by both the
/// WAL frame format here and the `wal.rs` implementation. It computes the CRC
/// from scratch on each call using a lazily-initialized lookup table; the
/// initial/final XOR with `0xFFFF_FFFF` matches the standard "CRC-32/ISO-HDLC"
/// used by `zlib`, `gzip`, PNG, etc.
///
/// No external `crc` crate is used — this keeps the crate std-only.
pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    let table = crc32_table();
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in bytes {
        let idx = ((crc ^ b as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ table[idx];
    }
    crc ^ 0xFFFF_FFFF
}

/// Returns a reference to the lazily-built CRC-32 lookup table.
///
/// The 256-entry table is built once on first use and cached for the life of
/// the process via [`std::sync::OnceLock`], so repeated [`crc32`] calls do no
/// redundant table construction and remain wait-free after warm-up.
fn crc32_table() -> &'static [u32; 256] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        let mut i = 0usize;
        while i < 256 {
            let mut crc = i as u32;
            let mut j = 0;
            while j < 8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB8_8320;
                } else {
                    crc >>= 1;
                }
                j += 1;
            }
            table[i] = crc;
            i += 1;
        }
        table
    })
}

/// A lock-free, bounded, multi-producer/multi-consumer queue of pending writes.
///
/// This is the boundary between the hot main loop (producers calling
/// [`try_enqueue`](Queue::try_enqueue)) and the background flush worker
/// (the consumer calling [`try_dequeue`](Queue::try_dequeue) /
/// [`drain_into`](Queue::drain_into)). Implementations MUST be wait-free on the
/// producer path so `submit()` never blocks the caller.
pub trait Queue: Send + Sync {
    /// Attempts to enqueue `req`. On a full ring returns `Err(req)`, handing the
    /// item back to the caller so it can be dropped (load shedding) without
    /// allocation churn. Never blocks.
    fn try_enqueue(&self, req: WriteRequest) -> Result<(), WriteRequest>;

    /// Pops a single item if one is available. Never blocks.
    fn try_dequeue(&self) -> Option<WriteRequest>;

    /// Bulk-pops up to `max` items into `out`, returning the number moved.
    /// Used by the worker to amortize synchronization across a batch.
    fn drain_into(&self, out: &mut Vec<WriteRequest>, max: usize) -> usize;

    /// Approximate number of queued items.
    fn len(&self) -> usize;

    /// Fixed capacity (rounded up to a power of two by the implementation).
    fn capacity(&self) -> usize;

    /// Returns `true` when the queue is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The durability boundary: an append-only, CRC-protected, recoverable log.
///
/// Records that have been [`append_batch`](WriteAheadLog::append_batch)'d and
/// fsync'd (when configured) survive a hard kill and are replayed exactly by
/// [`recover`](WriteAheadLog::recover). This is the *only* component that
/// provides crash durability; see the crate-level docs for the precise
/// guarantee and the at-risk window.
pub trait WriteAheadLog: Send + Sync {
    /// Appends each record as a frame and (if fsync is configured) flushes to
    /// stable storage. Returns the highest [`Lsn`] now durable.
    fn append_batch(&self, records: &[WalRecord]) -> std::io::Result<Lsn>;

    /// Marks all records with `lsn <= up_to_lsn` as safely in the store, so the
    /// log may rotate/truncate them in a crash-safe manner.
    fn checkpoint(&self, up_to_lsn: Lsn) -> std::io::Result<()>;

    /// Replays the log, tolerating a torn/corrupt **tail** by truncating to the
    /// last fully-valid record. Returns the surviving records in log order.
    fn recover(&self) -> std::io::Result<Vec<WalRecord>>;

    /// Flushes buffered writes to stable storage (fsync).
    fn flush(&self) -> std::io::Result<()>;
}

/// The slow, durable backend that ultimately owns persisted state.
///
/// All access to this trait happens on the worker thread, never on the submit
/// path. Implementations may block and may fail with a classified
/// [`crate::error::StoreError`].
pub trait PersistenceStore: Send + Sync {
    /// Durably persists a batch of states atomically (per the backend's
    /// semantics). On failure returns a classified [`crate::error::StoreError`].
    fn persist_batch(&self, states: &[PlayerState]) -> Result<(), crate::error::StoreError>;

    /// Loads the current persisted state for `id`, if any.
    fn load(&self, id: PlayerId) -> Option<PlayerState>;

    /// Number of distinct entities currently persisted.
    fn count(&self) -> usize;
}

/// The write-behind, coalescing cache sitting between the ring and the store.
///
/// The cache absorbs bursts: many [`apply`](Cache::apply) calls for the same
/// `player_id` collapse to a single dirty entry via **last-write-wins by
/// version**, so the store only ever sees the newest state. Dirty entries are
/// drained in bulk by the worker via [`take_dirty`](Cache::take_dirty); if the
/// store rejects them they are returned via
/// [`requeue_dirty`](Cache::requeue_dirty) so **no update is ever lost**.
pub trait Cache: Send + Sync {
    /// Upserts the state from `req`, applying last-write-wins by `version` and
    /// marking the entry dirty so it will be flushed.
    fn apply(&self, req: &WriteRequest);

    /// Atomically removes and returns up to `max` dirty entries for flushing.
    fn take_dirty(&self, max: usize) -> Vec<PlayerState>;

    /// Re-marks the given states dirty after a failed persist, merging under
    /// last-write-wins so a newer concurrent update is never clobbered.
    fn requeue_dirty(&self, states: Vec<PlayerState>);

    /// Reads the currently-cached state for `id`, if any.
    fn get(&self, id: PlayerId) -> Option<PlayerState>;

    /// Number of dirty (un-flushed) entries.
    fn dirty_len(&self) -> usize;

    /// Total number of cached entries (dirty or clean).
    fn len(&self) -> usize;

    /// Whether the cache holds no entries (dirty or clean).
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vector() {
        // Standard CRC-32/ISO-HDLC check value for the ASCII string "123456789".
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn crc32_empty() {
        assert_eq!(crc32(b""), 0x0000_0000);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let rec = WalRecord {
            lsn: 42,
            player_id: 7,
            version: 3,
            blob: vec![1, 2, 3, 4, 5],
        };
        let buf = rec.encode();
        let (decoded, consumed) = WalRecord::decode(&buf).unwrap();
        assert_eq!(decoded, rec);
        assert_eq!(consumed, buf.len());
    }

    #[test]
    fn decode_truncated() {
        let rec = WalRecord {
            lsn: 1,
            player_id: 1,
            version: 1,
            blob: vec![9; 10],
        };
        let buf = rec.encode();
        assert_eq!(
            WalRecord::decode(&buf[..buf.len() - 1]),
            Err(DecodeError::Truncated)
        );
        assert_eq!(WalRecord::decode(&buf[..4]), Err(DecodeError::Truncated));
    }

    #[test]
    fn decode_bad_magic() {
        let rec = WalRecord {
            lsn: 1,
            player_id: 1,
            version: 1,
            blob: vec![],
        };
        let mut buf = rec.encode();
        buf[0] ^= 0xFF;
        assert_eq!(WalRecord::decode(&buf), Err(DecodeError::BadMagic));
    }

    #[test]
    fn decode_bad_crc() {
        let rec = WalRecord {
            lsn: 1,
            player_id: 1,
            version: 1,
            blob: vec![5, 6, 7],
        };
        let mut buf = rec.encode();
        // Flip a payload byte; magic/length still valid, CRC must catch it.
        let mid = WAL_HEADER_LEN + 1;
        buf[mid] ^= 0xFF;
        assert_eq!(WalRecord::decode(&buf), Err(DecodeError::BadCrc));
    }

    #[test]
    fn decode_stream_of_two() {
        let a = WalRecord {
            lsn: 1,
            player_id: 10,
            version: 1,
            blob: vec![1],
        };
        let b = WalRecord {
            lsn: 2,
            player_id: 11,
            version: 2,
            blob: vec![2, 3],
        };
        let mut buf = a.encode();
        buf.extend_from_slice(&b.encode());

        let (da, na) = WalRecord::decode(&buf).unwrap();
        let (db, nb) = WalRecord::decode(&buf[na..]).unwrap();
        assert_eq!(da, a);
        assert_eq!(db, b);
        assert_eq!(na + nb, buf.len());
    }
}
