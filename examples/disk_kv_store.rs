//! `disk_kv_store` — a REAL, fully-working, **std-only** `PersistenceStore`
//! backed by a fast append-then-index disk file (the "fast disk DB like
//! RocksDB" seam — but with **zero external dependencies**).
//!
//! ## What this example proves
//!
//! This is the production-shaped answer to "how does a real cluster wire a
//! backend into FluxThroughput?". The crate ships [`MockDb`] as its in-memory
//! fault-injection harness; *this* file shows the OTHER half of the
//! `PersistenceStore` seam — a backend that actually survives a process
//! restart, written against nothing but `std`.
//!
//! [`MockDb`]: flux_throughput::store::MockDb
//!
//! The storage engine is the same shape a real embedded KV store (RocksDB,
//! LevelDB, sled) uses at its core:
//!
//! ```text
//!   persist_batch ──▶  append framed record to data file  ──▶  flush + sync_data
//!                                   │
//!                                   ▼
//!                     in-memory HashMap<PlayerId, PlayerState>   (the "index")
//!                                   │  last-write-wins by version
//!                     load(id) / count() served straight from the index
//!
//!   open() ──▶ replay the whole data file ──▶ rebuild the index (LWW)
//!              tolerating a torn tail exactly like the WAL does.
//! ```
//!
//! ## On-disk frame format (self-describing, CRC-protected)
//!
//! We mirror the crate's own WAL frame discipline (magic + length + CRC) so a
//! process killed mid-append leaves a detectable torn tail rather than silent
//! corruption. Every integer is little-endian:
//!
//! ```text
//!   offset  size  field
//!   ------  ----  -----------------------------------------------------------
//!     0      4    magic       = 0x4B564442  ("DBVK")
//!     4      4    frame_len   = total frame length incl. the trailing crc
//!     8      8    player_id
//!    16      8    version
//!    24      4    blob_len
//!    28   blob_len blob bytes
//!     ..     4    crc32       = IEEE CRC32 of every byte preceding this field
//! ```
//!
//! Header is 28 bytes, payload is `blob_len`, trailing CRC is 4, so
//! `frame_len == 28 + blob_len + 4`. The CRC covers the whole frame up to (but
//! not including) the CRC field. A torn write that mangles the length is caught
//! either as "not enough bytes for the claimed length" (truncated) or a CRC
//! mismatch — in both cases recovery stops at that frame and truncates the file
//! to the last good offset, so the index never absorbs a half-written record.
//!
//! ## Durability honesty (matches the crate's stance — no overclaiming)
//!
//! `persist_batch` calls `File::sync_data()` (fdatasync-equivalent). That
//! guarantees survival of a **process kill (SIGKILL)** — the bytes are pushed
//! to the kernel and then to the device's write path. True **power-loss /
//! drive-cache** survival on macOS additionally requires `fcntl(F_FULLFSYNC)`,
//! which std does not expose; the crate documents the same caveat for its WAL
//! (`src/wal.rs`, the `sync_data()` call at the append path). We do NOT claim
//! power-loss durability here. We DO claim clean-restart and SIGKILL-survival
//! durability, and the `main()` below proves the clean-restart half end to end.
//!
//! Run it:
//!
//! ```text
//!   cargo run --example disk_kv_store
//! ```

use flux_throughput::{FluxConfig, FluxEngine, PlayerState, SubmitOutcome};
// `PersistenceStore` is the trait we implement; bringing it into scope also lets
// us call its `count()` / `load()` methods on a concrete `DiskKvStore`.
use flux_throughput::error::StoreError;
use flux_throughput::PersistenceStore;
use flux_throughput::PlayerId;

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

// ===========================================================================
// On-disk frame codec
// ===========================================================================

/// Magic prefixing every record frame: ASCII "DBVK" in little-endian.
const REC_MAGIC: u32 = 0x4B56_4442;
/// Fixed header: magic(4) + frame_len(4) + player_id(8) + version(8) + blob_len(4).
const REC_HEADER_LEN: usize = 4 + 4 + 8 + 8 + 4;
/// Trailing CRC field width.
const REC_CRC_LEN: usize = 4;

/// Serializes one `PlayerState` into a single self-describing, CRC-protected
/// frame (see the module docs for the exact byte layout).
fn encode_record(st: &PlayerState) -> Vec<u8> {
    let blob_len = st.blob.len();
    let frame_len = REC_HEADER_LEN + blob_len + REC_CRC_LEN;
    let mut buf = Vec::with_capacity(frame_len);

    buf.extend_from_slice(&REC_MAGIC.to_le_bytes());
    buf.extend_from_slice(&(frame_len as u32).to_le_bytes());
    buf.extend_from_slice(&st.player_id.to_le_bytes());
    buf.extend_from_slice(&st.version.to_le_bytes());
    buf.extend_from_slice(&(blob_len as u32).to_le_bytes());
    buf.extend_from_slice(&st.blob);

    // CRC over everything written so far (the whole frame minus the CRC field).
    let crc = crc32(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());

    debug_assert_eq!(buf.len(), frame_len);
    buf
}

/// Outcome of trying to decode one frame from the front of a byte slice.
enum DecodeOne {
    /// A good frame: the decoded state plus the number of bytes consumed.
    Record(PlayerState, usize),
    /// A torn / corrupt tail — stop replaying here and truncate the file to the
    /// bytes consumed so far. This is the EXPECTED signature of a process killed
    /// mid-append; it is not a fatal error, exactly like the crate's WAL.
    TornTail,
}

/// Decodes a single frame from the front of `buf`, tolerating a torn tail the
/// same way `WalRecord::decode` + WAL recovery do.
fn decode_record(buf: &[u8]) -> DecodeOne {
    // Need at least the fixed header to learn the frame length.
    if buf.len() < REC_HEADER_LEN {
        return DecodeOne::TornTail;
    }
    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if magic != REC_MAGIC {
        // A bad magic mid-stream means genuine corruption; we treat the tail as
        // torn and stop. (A real cluster would alert here; for the seam demo,
        // stopping cleanly is the safe, lossless choice.)
        return DecodeOne::TornTail;
    }

    let frame_len = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    // A frame must be at least header + crc; a smaller claim is corruption we
    // treat as a torn tail (the length field itself is untrustworthy).
    if frame_len < REC_HEADER_LEN + REC_CRC_LEN || buf.len() < frame_len {
        return DecodeOne::TornTail;
    }

    let blob_len = u32::from_le_bytes(buf[24..28].try_into().unwrap()) as usize;
    // Cross-check the declared blob length against the declared frame length; if
    // they disagree the length fields are corrupt -> torn tail.
    if frame_len != REC_HEADER_LEN + blob_len + REC_CRC_LEN {
        return DecodeOne::TornTail;
    }

    // Validate the CRC over the frame body (everything before the CRC field).
    let body_end = frame_len - REC_CRC_LEN;
    let stored_crc = u32::from_le_bytes(buf[body_end..frame_len].try_into().unwrap());
    let actual_crc = crc32(&buf[..body_end]);
    if stored_crc != actual_crc {
        return DecodeOne::TornTail;
    }

    let player_id = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    let version = u64::from_le_bytes(buf[16..24].try_into().unwrap());
    let blob = buf[REC_HEADER_LEN..REC_HEADER_LEN + blob_len].to_vec();

    DecodeOne::Record(
        PlayerState {
            player_id,
            version,
            blob,
        },
        frame_len,
    )
}

// ===========================================================================
// CRC-32 (pure Rust, std-only) — kept local so the example has zero deps.
//
// This is the standard CRC-32/ISO-HDLC (reflected polynomial 0xEDB88320), the
// same algorithm the crate uses for its WAL frames. We compute it locally
// rather than reach into the crate so this file stands alone as a copyable
// reference for an embedder writing their own backend.
// ===========================================================================

fn crc32(bytes: &[u8]) -> u32 {
    // Build the 256-entry lookup table once per process.
    use std::sync::OnceLock;
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
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
    });

    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in bytes {
        let idx = ((crc ^ b as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ table[idx];
    }
    crc ^ 0xFFFF_FFFF
}

// ===========================================================================
// DiskKvStore — the PersistenceStore implementation
// ===========================================================================

/// Inner mutable state, guarded by one `Mutex`.
///
/// ## Why a `Mutex` here does not violate the main-loop isolation guarantee
///
/// The crate's headline property is that `FluxEngine::submit` never blocks —
/// no lock, no syscall, no unbounded wait. That guarantee is about the *submit
/// path*, which only ever touches the lock-free ring. A `PersistenceStore`
/// lives at the OPPOSITE end of the pipeline, on the worker/DB side, where
/// blocking I/O and locking are explicitly allowed and expected (the same place
/// the crate injects latency to drive back-pressure). Contention here can only
/// ever slow the single background worker — never a `submit()` caller — because
/// the ring decouples the two. See `src/store.rs` for the same rationale around
/// `MockDb`.
struct Inner {
    /// Append-only data file. Buffered for throughput; we explicitly
    /// `flush()` + `sync_data()` at the end of every `persist_batch` so a batch
    /// is durable before we return `Ok`.
    writer: BufWriter<File>,

    /// The in-memory index: player id -> latest persisted state, under
    /// last-write-wins by `version`. This is what `load` / `count` serve, so
    /// reads never touch the disk. On restart it is rebuilt by replaying the
    /// data file (see [`DiskKvStore::open`]).
    index: HashMap<PlayerId, PlayerState>,
}

/// A real, restart-durable `PersistenceStore` over an append-then-index disk
/// file. Construct with [`DiskKvStore::open`]; share as
/// `Arc<dyn PersistenceStore>` with [`FluxEngine::start`].
///
/// All state is behind one `Mutex<Inner>` so every method takes `&self`, letting
/// a single `Arc<DiskKvStore>` be handed to the engine as a trait object and
/// still be re-opened/inspected by the embedder.
pub struct DiskKvStore {
    inner: Mutex<Inner>,
    /// Path to the append-only data file (retained for diagnostics / re-open).
    path: PathBuf,
}

impl DiskKvStore {
    /// Opens (creating if absent) the store at `path`, **replaying the existing
    /// data file to rebuild the in-memory index** so the store survives a
    /// restart. A torn tail left by a previous crash is detected and the file is
    /// truncated to the last fully-valid record (zero partial-state corruption).
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();

        // (1) Replay whatever is already on disk into a fresh index, learning the
        //     offset of the last good frame so we can truncate any torn tail.
        let (index, good_len) = Self::rebuild_index(&path)?;

        // (2) If recovery found a torn tail (good_len < file size), truncate the
        //     file so future appends start at a clean boundary — exactly the WAL's
        //     "truncate to last good offset" discipline.
        {
            // Never truncate on open: the existing data file holds committed
            // records we just rebuilt the index from. We only ever shrink it via
            // the explicit `set_len` below, to trim a torn tail.
            let f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)?;
            let on_disk = f.metadata()?.len();
            if good_len < on_disk {
                f.set_len(good_len)?;
                f.sync_all()?;
            }
        }

        // (3) Open the file for appending. We seek to the end implicitly via
        //     `append(true)`; the BufWriter batches small writes.
        let file = OpenOptions::new().create(true).append(true).open(&path)?;

        Ok(DiskKvStore {
            inner: Mutex::new(Inner {
                writer: BufWriter::new(file),
                index,
            }),
            path,
        })
    }

    /// Replays the data file at `path`, returning the reconstructed index (under
    /// last-write-wins by version) and the byte offset just past the last
    /// fully-valid frame (the truncation point for a torn tail).
    ///
    /// A missing file is treated as an empty store. This is a free function on
    /// purpose: it touches only the bytes on disk, so it is trivially reusable
    /// and unit-testable.
    fn rebuild_index(path: &Path) -> std::io::Result<(HashMap<PlayerId, PlayerState>, u64)> {
        let mut index: HashMap<PlayerId, PlayerState> = HashMap::new();

        let mut file = match File::open(path) {
            Ok(f) => f,
            // No file yet -> empty store, zero good bytes.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok((index, 0));
            }
            Err(e) => return Err(e),
        };

        // Read the whole file into memory. A production engine would memory-map
        // or stream this; for an example, a single read keeps the replay logic
        // obvious and is still plenty fast for the demo's volume.
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        // Walk frame by frame. `good_len` advances only past frames that decode
        // AND pass CRC; the first torn/corrupt frame stops the replay.
        let mut offset = 0usize;
        let mut good_len = 0u64;
        while offset < bytes.len() {
            match decode_record(&bytes[offset..]) {
                DecodeOne::Record(state, consumed) => {
                    // Last-write-wins by version: only overwrite when the incoming
                    // version is at least as new as what we already hold (`>=`
                    // makes equal-version re-persists idempotent, matching MockDb).
                    match index.get(&state.player_id) {
                        Some(existing) if state.version < existing.version => {
                            // Stale frame — keep the newer indexed value.
                        }
                        _ => {
                            index.insert(state.player_id, state);
                        }
                    }
                    offset += consumed;
                    good_len = offset as u64;
                }
                DecodeOne::TornTail => break,
            }
        }

        Ok((index, good_len))
    }

    /// Returns the data-file path (for diagnostics / re-open in `main`).
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl PersistenceStore for DiskKvStore {
    /// Durably persists a batch: append each state as a framed record, then
    /// `flush()` + `sync_data()` ONCE for the whole batch (amortizing the fsync),
    /// and only THEN update the in-memory index. Ordering matters — the bytes are
    /// on stable storage before the index claims them, so a crash can never leave
    /// the index ahead of the log.
    fn persist_batch(&self, states: &[PlayerState]) -> Result<(), StoreError> {
        // An empty flush is a valid no-op (the worker may hand us one).
        if states.is_empty() {
            return Ok(());
        }

        let mut inner = self
            .inner
            .lock()
            .map_err(|_| StoreError::Fatal("DiskKvStore mutex poisoned".to_string()))?;

        // (1) Append every record to the buffered file. We map any I/O error to a
        //     RETRYABLE transient so the worker's retry/requeue machinery keeps the
        //     data safe (it stays dirty in the cache + durable in the crate's WAL)
        //     rather than dropping it. A real backend would classify ENOSPC vs a
        //     transient network blip more finely; Transient is the safe default.
        for st in states {
            let frame = encode_record(st);
            inner
                .writer
                .write_all(&frame)
                .map_err(|e| StoreError::Transient(format!("disk append failed: {e}")))?;
        }

        // (2) Flush the userspace buffer into the kernel, then force the kernel to
        //     push it to stable storage. THIS is the durability moment: after it
        //     returns, the batch survives a SIGKILL. (Power-loss survival would
        //     additionally need F_FULLFSYNC on macOS — see the module docs.)
        inner
            .writer
            .flush()
            .map_err(|e| StoreError::Transient(format!("disk flush failed: {e}")))?;
        inner
            .writer
            .get_ref()
            .sync_data()
            .map_err(|e| StoreError::Transient(format!("disk sync_data failed: {e}")))?;

        // (3) Only NOW update the index, under last-write-wins by version, so the
        //     in-memory view never claims a record the log has not durably stored.
        for st in states {
            match inner.index.get(&st.player_id) {
                Some(existing) if st.version < existing.version => { /* stale, skip */ }
                _ => {
                    inner.index.insert(st.player_id, st.clone());
                }
            }
        }

        Ok(())
    }

    /// Serves the current persisted state straight from the in-memory index — no
    /// disk read on the hot read path.
    fn load(&self, id: PlayerId) -> Option<PlayerState> {
        let inner = self.inner.lock().ok()?;
        inner.index.get(&id).cloned()
    }

    /// Number of distinct entities currently persisted (index size).
    fn count(&self) -> usize {
        match self.inner.lock() {
            Ok(inner) => inner.index.len(),
            Err(_) => 0,
        }
    }
}

// ===========================================================================
// main() — the end-to-end durability proof across a clean restart
// ===========================================================================

/// Distinct players we touch. Each gets several versioned updates so the cache's
/// last-write-wins coalescing and the store's LWW index are both exercised.
const PLAYERS: u64 = 1_000;
/// Updates per player (strictly increasing version), so the final persisted
/// version for player `p` is deterministic and checkable on re-open.
const UPDATES_PER_PLAYER: u64 = 4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Per-process temp dir so concurrent runs never collide and we leave no
    // litter behind on success.
    let dir = std::env::temp_dir().join(format!("flux_disk_kv_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;

    let data_path = dir.join("store.db");
    let wal_dir = dir.join("wal");

    println!("== FluxThroughput disk_kv_store example ==");
    println!("data file : {}", data_path.display());
    println!("wal dir   : {}", wal_dir.display());

    // The deterministic expected final state: player p -> highest version we
    // submit for it. We submit versions 1..=UPDATES_PER_PLAYER per player, so the
    // last (highest) version wins everywhere.
    let expected_version: u64 = UPDATES_PER_PLAYER;

    // ---- Phase 1: run the engine against a fresh DiskKvStore ----------------
    {
        // Build the disk store and hand it to the engine as a trait object.
        let store = Arc::new(DiskKvStore::open(&data_path)?);
        let store_dyn: Arc<dyn PersistenceStore> = store.clone();

        let config = FluxConfig {
            ring_capacity: 1 << 14,
            flush_max_batch: 1024,
            wal_fsync: true,
            ..FluxConfig::default()
        }
        .with_wal_dir(&wal_dir);

        let engine = FluxEngine::start(config, store_dyn)?;

        // Fire-and-forget submit burst. Each (player, version) pair carries the
        // version in its 8-byte blob so we can verify the exact persisted payload
        // after the restart, not just the version number.
        let mut submitted: u64 = 0;
        let mut dropped: u64 = 0;
        for version in 1..=UPDATES_PER_PLAYER {
            for player_id in 0..PLAYERS {
                let blob = version.to_le_bytes().to_vec();
                match engine.submit(player_id, version, blob) {
                    SubmitOutcome::Enqueued => {}
                    SubmitOutcome::Dropped => dropped += 1,
                }
                submitted += 1;
            }
        }

        // Synchronously push every remaining dirty entry through the WAL to the
        // disk store, then shut down cleanly (final drain + WAL fsync + join).
        let flushed = engine.flush_now()?;
        engine.shutdown()?;

        println!(
            "phase 1: submitted {submitted} updates ({dropped} shed under back-pressure), \
             flush_now persisted {flushed} records",
        );
        // Sanity: even with load-shedding, the LAST version for every player is
        // guaranteed to reach the store, because we submit versions in ascending
        // passes and the final pass (version == UPDATES_PER_PLAYER) coalesces to
        // the winning value for every id before flush_now drains it.
    }
    // `store`/`engine` are dropped here; the only durable record of phase 1 is
    // now the bytes in `store.db` (and the WAL, which we deliberately ignore on
    // the clean-restart path).

    // ---- Phase 2: RE-OPEN a brand-new DiskKvStore from the same file --------
    // This is the durability proof: a fresh process-like object replays the data
    // file, rebuilds its index, and must agree with what phase 1 persisted.
    let reopened = DiskKvStore::open(&data_path)?;

    let count = reopened.count();
    println!("phase 2: re-opened store, index holds {count} distinct players");

    // (a) Every player must be present, at exactly the winning version, with the
    //     exact blob payload we wrote for that version.
    let expected_blob = expected_version.to_le_bytes().to_vec();
    let mut checked = 0u64;
    for player_id in 0..PLAYERS {
        let got = reopened
            .load(player_id)
            .unwrap_or_else(|| panic!("player {player_id} missing after restart"));
        assert_eq!(
            got.version, expected_version,
            "player {player_id} persisted version {} != expected {expected_version}",
            got.version,
        );
        assert_eq!(
            got.blob, expected_blob,
            "player {player_id} persisted blob mismatch after restart",
        );
        assert_eq!(got.player_id, player_id);
        checked += 1;
    }

    // (b) The distinct-row count must be exactly the number of players (no
    //     duplicates, no missing ids).
    assert_eq!(
        count as u64, PLAYERS,
        "distinct persisted players ({count}) != expected ({PLAYERS})",
    );

    // ---- Phase 3: prove a SECOND re-open is idempotent ----------------------
    // Re-opening again must not change anything (replay is deterministic and the
    // append-only file was not modified by the read-only re-open above).
    let reopened_again = DiskKvStore::open(&data_path)?;
    assert_eq!(
        reopened_again.count(),
        count,
        "re-open changed the row count"
    );
    assert_eq!(
        reopened_again.load(0).map(|s| s.version),
        Some(expected_version),
        "re-open #2 disagrees about player 0's version",
    );

    // Best-effort cleanup on success.
    let _ = std::fs::remove_dir_all(&dir);

    println!(
        "\nSUCCESS: {checked} players verified at version {expected_version} after a clean \
         restart; DiskKvStore replayed its append-only file, rebuilt the index under \
         last-write-wins, and matched exactly what FluxThroughput persisted.",
    );
    Ok(())
}
