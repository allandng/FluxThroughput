//! Write-ahead log (`impl WriteAheadLog`). **[impl:wal] — the durability boundary.**
//!
//! [`Wal`] is an append-only, CRC-protected, crash-recoverable log of
//! [`crate::domain::WalRecord`] frames. It is the *only* component in the crate
//! that provides durability across a hard kill (SIGKILL / power loss): a record
//! that has been [`append_batch`](WriteAheadLog::append_batch)'d **and fsync'd**
//! is guaranteed to be replayed exactly by [`recover`](WriteAheadLog::recover).
//!
//! ## On-disk layout
//!
//! A single *active segment* file (`wal-<n>.log`) lives under the configured
//! directory. Its byte stream is:
//!
//! ```text
//!   [ 8-byte file magic b"FLUXWAL1" ] [ frame ] [ frame ] [ frame ] ...
//!   ^------ HEADER_MAGIC ------------^^------ WalRecord::encode() bytes ------^
//! ```
//!
//! Each frame is produced by [`crate::domain::WalRecord::encode`] — a
//! self-describing `[magic][frame_len][lsn][player_id][version][blob_len][blob]
//! [crc32]` record. We deliberately reuse that single framing (and the shared
//! `crate::domain::crc32` it embeds) so there is exactly **one** wire format in
//! the crate; this file never invents a second one.
//!
//! ## Write-ahead ordering (why "ahead")
//!
//! The pipeline rule (see SPEC / worker.rs) is: **the WAL append + fsync happens
//! BEFORE the slow store is told about a batch.** That is the whole point of a
//! *write-ahead* log — the intention to persist is made durable first, so that
//! if we crash after the WAL append but before (or during) the store write, the
//! record is still on disk and `recover()` replays it into the store on the next
//! start. The ordering inside [`append_batch`] enforces the on-disk half of this
//! contract:
//!
//! 1. write all frame bytes into the buffered writer,
//! 2. `flush()` the userspace buffer down into the kernel,
//! 3. `sync_data()` so the kernel pushes them to stable storage,
//! 4. only THEN return the highest LSN to the caller.
//!
//! Until step 3 returns, we make **no claim** that the records are durable. The
//! caller (worker) only proceeds to the store after `append_batch` returns Ok,
//! so the "WAL before store" ordering is preserved.
//!
//! ## Torn-tail recovery (why a partial write is safe)
//!
//! A crash can interrupt a write at any byte. The last frame on disk may
//! therefore be *torn*: too short to decode ([`DecodeError::Truncated`]), or
//! present but with a damaged body that fails the CRC ([`DecodeError::BadCrc`]),
//! or its leading magic clobbered ([`DecodeError::BadMagic`]). Because each
//! frame carries its own length and a CRC computed over everything before the
//! CRC field, a torn write can never masquerade as a good record:
//!
//! * If too few bytes survived, `decode` reports `Truncated`.
//! * If the length survived but the payload/CRC did not, `decode` reports
//!   `BadCrc` (or `Truncated` if the length fields are themselves inconsistent).
//!
//! [`recover`](WriteAheadLog::recover) decodes frames sequentially from just
//! past the file magic, advancing by `bytes_consumed`. The first `decode` error
//! is treated as **end of the valid log**: we keep every record decoded so far,
//! physically **truncate the file** to the byte offset of that last good frame
//! (so the garbage can never be misread on a subsequent recovery), fsync the
//! truncation, and return. This yields ZERO partial-state corruption — a half
//! written record is simply discarded, never half-applied.
//!
//! We do not try to "scan past" a corrupt frame to find later good frames: in an
//! append-only log a corrupt frame in the middle means the writer was killed
//! there, so nothing after it can be trusted either. Stopping at the first error
//! (the conservative choice) and truncating is both correct and simple.
//!
//! ## Checkpoint / log rotation (bounded size, crash-safe)
//!
//! Once the store has durably absorbed every record up to some LSN, those
//! records are dead weight in the WAL. [`checkpoint`](WriteAheadLog::checkpoint)
//! garbage-collects them by writing a **fresh** segment containing only the
//! still-live records (`lsn > up_to_lsn`), fsyncing it, then atomically
//! `rename`-ing it over the active segment and fsyncing the directory. Because
//! `rename` within a filesystem is atomic, a crash at any instant leaves either
//! the complete old segment or the complete new one in place — never a torn mix.
//!
//! This module is **safe code only** (no `unsafe`).

use crate::domain::{Lsn, WalRecord, WriteAheadLog};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// 8-byte magic written at the very start of every WAL segment file.
///
/// This is the *file* magic and is distinct from the per-frame magic embedded
/// by [`WalRecord::encode`]. It lets `open`/`recover` immediately reject a file
/// that is not one of ours (or a truncated-to-empty file), and it reserves a
/// fixed 8-byte prefix that recovery skips before decoding frames.
const HEADER_MAGIC: &[u8; 8] = b"FLUXWAL1";

/// Length of the file header (the magic). Frame decoding starts at this offset.
const HEADER_LEN: u64 = 8;

/// Fixed name of the active segment within the WAL directory.
///
/// A single active segment is sufficient for this design: [`checkpoint`] keeps
/// it bounded by rewriting it in place (via a temp file + atomic rename), so the
/// log never grows without bound and we never need to manage a chain of
/// segments at read time. The `-0` suffix anticipates multi-segment growth but
/// is intentionally stable here.
const ACTIVE_SEGMENT: &str = "wal-0.log";

/// Name of the temporary segment that [`checkpoint`] writes before atomically
/// renaming it over [`ACTIVE_SEGMENT`].
const TEMP_SEGMENT: &str = "wal-0.log.tmp";

/// Append-only, CRC-protected, crash-recoverable write-ahead log.
///
/// Open with [`Wal::open`]. All mutable state lives behind a single [`Mutex`]
/// so the type is `Send + Sync` and append/checkpoint/flush are serialized with
/// respect to each other (they all mutate the same file handle and offsets).
/// The mutex is **never** taken on the hot submit path — only the background
/// worker (and explicit `submit_durable`/`flush_now`) call into the WAL.
pub struct Wal {
    /// WAL directory; created on [`open`] if missing. Used to resolve the active
    /// and temp segment paths and to fsync the directory after a rename.
    dir: PathBuf,
    /// Whether appends fsync (`sync_data`) before returning — strong durability.
    fsync: bool,
    /// All write-side mutable state, serialized by one lock.
    inner: Mutex<Inner>,
}

/// The mutable interior of a [`Wal`], guarded by [`Wal::inner`].
struct Inner {
    /// Buffered writer over the active segment, positioned at the append point.
    /// `BufWriter` coalesces many small frame writes into fewer syscalls; we
    /// always `flush()` it (and optionally `sync_data`) before returning from an
    /// append so nothing is silently stranded in userspace.
    writer: BufWriter<File>,
    /// Current write offset (absolute byte position of the next append). Tracked
    /// explicitly so callers never need a `seek` to learn the size, and so a
    /// checkpoint can reset it precisely after rotation.
    write_offset: u64,
    /// Highest LSN appended so far (0 if nothing has been appended). Returned by
    /// [`append_batch`] and used to satisfy its "returns highest lsn" contract.
    last_lsn: Lsn,
}

impl Wal {
    /// Opens (creating if necessary) the write-ahead log rooted at `dir`.
    ///
    /// Creates `dir` if missing, opens the active segment for read+append,
    /// writes-and-verifies the 8-byte file magic, and positions the write offset
    /// at the end of the existing data. `fsync` selects the strong-durability
    /// mode in which each [`append_batch`] flushes to stable storage before
    /// returning.
    ///
    /// Note: this does **not** replay the log — call
    /// [`recover`](WriteAheadLog::recover) for that. `open` followed by
    /// `recover` is the normal startup sequence.
    pub fn open(dir: &Path, fsync: bool) -> std::io::Result<Self> {
        // Ensure the directory exists. `create_dir_all` is a no-op if present.
        std::fs::create_dir_all(dir)?;

        let seg_path = dir.join(ACTIVE_SEGMENT);

        // Open read+write, creating the file if it does not yet exist. We do NOT
        // use `append(true)` because we want explicit control over the cursor
        // (so we can seek to the end after writing/verifying the header, and so
        // recovery's `set_len` truncation behaves predictably).
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            // Never truncate: an existing segment holds un-recovered frames that
            // `recover()` must replay. Truncating here would silently destroy the
            // WAL on every startup and defeat crash durability.
            .truncate(false)
            .open(&seg_path)?;

        // Establish / verify the file header.
        let existing_len = file.metadata()?.len();
        let write_offset = if existing_len < HEADER_LEN {
            // Brand-new (or pathologically short / previously-torn-in-header)
            // file: (re)write the magic from offset 0 and fsync it so the header
            // itself is durable before any frame can follow it.
            file.seek(SeekFrom::Start(0))?;
            file.write_all(HEADER_MAGIC)?;
            file.sync_data()?;
            HEADER_LEN
        } else {
            // Existing file: verify the magic so we never append our frames into
            // a foreign / corrupt file.
            let mut magic = [0u8; HEADER_LEN as usize];
            file.seek(SeekFrom::Start(0))?;
            file.read_exact(&mut magic)?;
            if &magic != HEADER_MAGIC {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "WAL segment header magic mismatch (not a FLUXWAL1 file)",
                ));
            }
            existing_len
        };

        // Position the file cursor at the append point and wrap it in a buffer.
        // The `BufWriter`'s internal cursor mirrors the OS cursor we just set.
        file.seek(SeekFrom::Start(write_offset))?;
        let writer = BufWriter::new(file);

        Ok(Wal {
            dir: dir.to_path_buf(),
            fsync,
            inner: Mutex::new(Inner {
                writer,
                write_offset,
                // last_lsn starts at 0; `recover` (or the first append) updates
                // the in-memory high-water mark. The on-disk LSNs are the source
                // of truth, so a stale 0 here is harmless until first append.
                last_lsn: 0,
            }),
        })
    }

    /// Absolute path of the active segment.
    fn active_path(&self) -> PathBuf {
        self.dir.join(ACTIVE_SEGMENT)
    }

    /// Absolute path of the checkpoint temp segment.
    fn temp_path(&self) -> PathBuf {
        self.dir.join(TEMP_SEGMENT)
    }

    /// fsyncs the WAL directory so a preceding `rename`/`create` is itself
    /// durable (directory entries are metadata and need their own sync).
    ///
    /// On some platforms opening a directory for sync is unsupported; we treat
    /// such an error as benign (best-effort) rather than failing the operation,
    /// because the data file's own `sync_data` already covers the bytes.
    fn sync_dir(&self) -> std::io::Result<()> {
        match File::open(&self.dir) {
            Ok(d) => match d.sync_all() {
                Ok(()) => Ok(()),
                // Directory fsync is a no-op / unsupported on some filesystems.
                Err(e) if e.kind() == std::io::ErrorKind::InvalidInput => Ok(()),
                Err(e) => Err(e),
            },
            Err(e) if e.kind() == std::io::ErrorKind::InvalidInput => Ok(()),
            Err(e) => Err(e),
        }
    }
}

impl WriteAheadLog for Wal {
    /// Appends each record as a [`WalRecord::encode`] frame, then makes the batch
    /// durable, and returns the highest LSN written.
    ///
    /// ## Ordering (the load-bearing part)
    ///
    /// 1. Encode + write every frame into the `BufWriter`. We write the whole
    ///    batch before any flush so a single fsync amortizes over the batch.
    /// 2. `flush()` the `BufWriter` — pushes userspace bytes into the kernel.
    /// 3. If `fsync`, `sync_data()` — pushes kernel bytes to stable storage.
    /// 4. Advance the in-memory write offset / last-LSN and return the high LSN.
    ///
    /// Only after this returns Ok does the caller (worker) talk to the store, so
    /// the WAL is always written *ahead* of the store. If we are killed between
    /// steps 1 and 3, the unsynced tail is exactly the torn tail that `recover`
    /// discards — no half-applied state.
    fn append_batch(&self, records: &[WalRecord]) -> std::io::Result<Lsn> {
        let mut inner = self.inner.lock().expect("WAL mutex poisoned");

        // An empty batch is a well-defined no-op: nothing to encode, nothing to
        // sync, and the highest durable LSN is unchanged.
        if records.is_empty() {
            return Ok(inner.last_lsn);
        }

        // Step 1: serialize every frame into the buffered writer. We accumulate
        // the bytes-written and the max LSN as we go so the commit (step 4) is a
        // single, cheap update once durability is confirmed.
        let mut bytes_written: u64 = 0;
        let mut high_lsn = inner.last_lsn;
        for rec in records {
            let frame = rec.encode();
            inner.writer.write_all(&frame)?;
            bytes_written += frame.len() as u64;
            if rec.lsn > high_lsn {
                high_lsn = rec.lsn;
            }
        }

        // Step 2: drain the userspace buffer into the kernel. Without this, a
        // crash could lose buffered bytes even though we "wrote" them.
        inner.writer.flush()?;

        // Step 3: force the kernel to push the data to stable storage. We use
        // `sync_data` (fdatasync) rather than `sync_all` (fsync) because we only
        // need the file *contents* durable, not its mtime/atime metadata — the
        // file length is implied by the data we just wrote. This is the moment
        // the batch becomes crash-durable.
        if self.fsync {
            inner.writer.get_ref().sync_data()?;
        }

        // Step 4: commit the in-memory bookkeeping only after durability above.
        inner.write_offset += bytes_written;
        inner.last_lsn = high_lsn;
        Ok(high_lsn)
    }

    /// Discards records with `lsn <= up_to_lsn` (now safely in the store) by
    /// rewriting the segment to contain only the still-live tail, crash-safely.
    ///
    /// ## Crash-safe rotation
    ///
    /// 1. Read & decode the current segment, collecting records with
    ///    `lsn > up_to_lsn` (the survivors that the store has NOT yet absorbed).
    /// 2. Write a fresh temp segment: header magic + re-encoded survivor frames.
    /// 3. fsync the temp file's data so the new segment is durable on disk.
    /// 4. Atomically `rename` the temp over the active segment. `rename` is
    ///    atomic within a filesystem, so an observer (or a crash) sees either the
    ///    full old segment or the full new one — never a partial overlap.
    /// 5. fsync the directory so the rename itself is durable.
    /// 6. Reopen the (now rotated) active segment for appending and reset the
    ///    in-memory offset / last-LSN to match.
    ///
    /// If we are killed before step 4 completes, the original active segment is
    /// untouched and fully valid — recovery simply replays the un-checkpointed
    /// records again, which is safe because the store applies them idempotently
    /// (last-write-wins by version). No record is ever lost by a mid-checkpoint
    /// crash.
    fn checkpoint(&self, up_to_lsn: Lsn) -> std::io::Result<()> {
        let mut inner = self.inner.lock().expect("WAL mutex poisoned");

        // Step 0: make sure everything appended so far is on disk before we read
        // the segment back, so survivors aren't missed because they were still
        // buffered in userspace.
        inner.writer.flush()?;

        // Step 1: read the whole active segment and decode it, keeping only the
        // records the store has not yet durably absorbed. We reuse the same
        // tolerant decode loop as recovery: a torn tail here is simply dropped
        // (those records were never acknowledged downstream, so it is safe).
        let active = self.active_path();
        let survivors: Vec<WalRecord> = {
            let bytes = read_all(&active)?;
            let (records, _good_end) = decode_segment(&bytes);
            records.into_iter().filter(|r| r.lsn > up_to_lsn).collect()
        };

        // Step 2: write the survivors into a fresh temp segment.
        let temp = self.temp_path();
        let new_offset = {
            // Truncate-create the temp file so a stale temp from a prior crashed
            // checkpoint can't contaminate the new one.
            let mut tmp = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&temp)?;

            tmp.write_all(HEADER_MAGIC)?;
            let mut offset = HEADER_LEN;
            for rec in &survivors {
                let frame = rec.encode();
                tmp.write_all(&frame)?;
                offset += frame.len() as u64;
            }

            // Step 3: fsync the temp file's contents so the new segment is fully
            // durable BEFORE we make it the active one. If we crash after this
            // but before the rename, recovery still uses the old (valid) active
            // segment and the orphan temp is overwritten next checkpoint.
            tmp.sync_data()?;
            offset
        };

        // Step 4: atomically replace the active segment with the new one.
        std::fs::rename(&temp, &active)?;

        // Step 5: make the rename itself durable (directory entry is metadata).
        self.sync_dir()?;

        // Step 6: reopen the freshly-rotated active segment for appending and
        // reset the in-memory cursor / high-water mark. We reopen rather than
        // reuse the old handle because that handle still refers to the old
        // (now-unlinked) inode; new appends must land in the new file.
        let mut file = OpenOptions::new().read(true).write(true).open(&active)?;
        file.seek(SeekFrom::Start(new_offset))?;
        inner.writer = BufWriter::new(file);
        inner.write_offset = new_offset;
        // The highest surviving LSN (if any) is the new high-water mark; if no
        // survivors remain, retain `up_to_lsn` so subsequently-assigned LSNs
        // stay monotonic relative to what the store already has.
        let max_survivor = survivors.iter().map(|r| r.lsn).max();
        inner.last_lsn = max_survivor.unwrap_or(up_to_lsn).max(up_to_lsn);

        Ok(())
    }

    /// Replays the active segment, tolerating a torn/corrupt **tail**.
    ///
    /// Skips the 8-byte file header, then repeatedly [`WalRecord::decode`]s the
    /// remaining bytes, advancing by `bytes_consumed`. On the first decode error
    /// (`Truncated` / `BadCrc` / `BadMagic`) it stops, treats that offset as the
    /// end of the valid log, **truncates the file** to the last good offset
    /// (`set_len` + `sync_data` + directory fsync) so the garbage can never be
    /// re-read, and returns the records collected so far. Never panics on
    /// corrupt input.
    fn recover(&self) -> std::io::Result<Vec<WalRecord>> {
        let mut inner = self.inner.lock().expect("WAL mutex poisoned");

        // Flush any buffered (not-yet-written) bytes so the on-disk image we are
        // about to read is complete and consistent with our writer.
        inner.writer.flush()?;

        let active = self.active_path();
        let bytes = read_all(&active)?;

        // Decode every fully-valid frame after the header. `good_end` is the
        // absolute byte offset just past the last good frame — i.e. the length
        // the file should have if the tail is torn.
        let (records, good_end) = decode_segment(&bytes);

        // If there is trailing garbage (a torn / partial tail, or mid-file
        // corruption we conservatively treat as end-of-log), physically truncate
        // it away so a future recovery is deterministic and so subsequent
        // appends land immediately after the last good record.
        if good_end < bytes.len() as u64 {
            // Reopen the file for truncation independently of the buffered
            // writer to avoid any cursor confusion, then re-point the writer at
            // the truncated end.
            let file = OpenOptions::new().read(true).write(true).open(&active)?;
            file.set_len(good_end)?;
            file.sync_data()?;
            // Make the size change durable at the directory level too.
            self.sync_dir()?;

            // Re-establish the buffered writer at the new end-of-file.
            let mut wfile = OpenOptions::new().read(true).write(true).open(&active)?;
            wfile.seek(SeekFrom::Start(good_end))?;
            inner.writer = BufWriter::new(wfile);
        }

        // Update in-memory bookkeeping to reflect the recovered, possibly
        // truncated, on-disk state.
        inner.write_offset = good_end;
        inner.last_lsn = records.iter().map(|r| r.lsn).max().unwrap_or(0);

        Ok(records)
    }

    /// Flushes buffered writes and fsyncs the active segment to stable storage.
    fn flush(&self) -> std::io::Result<()> {
        let mut inner = self.inner.lock().expect("WAL mutex poisoned");
        inner.writer.flush()?;
        inner.writer.get_ref().sync_data()?;
        Ok(())
    }
}

/// Reads an entire file into a byte vector.
///
/// Centralized so `recover` and `checkpoint` share one read path. A missing file
/// is impossible here (the segment is created on `open`), but any I/O error is
/// propagated rather than swallowed.
fn read_all(path: &Path) -> std::io::Result<Vec<u8>> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Decodes every fully-valid frame in a segment image, stopping at the first
/// torn/corrupt frame.
///
/// `bytes` is the raw segment (header + frames). Returns `(records, good_end)`
/// where `good_end` is the absolute byte offset (including the header) just past
/// the last successfully-decoded frame — i.e. the length to truncate the file
/// to if anything trails it.
///
/// This is the shared core of the torn-tail logic. It never panics: every decode
/// outcome (`Ok`, `Truncated`, `BadCrc`, `BadMagic`) is handled, and any error
/// terminates the scan (conservative "first error = end of valid log").
fn decode_segment(bytes: &[u8]) -> (Vec<WalRecord>, u64) {
    // A segment too short to even contain the header is treated as empty; the
    // good end is whatever bytes exist (so `open` can rewrite the header path).
    if (bytes.len() as u64) < HEADER_LEN {
        return (Vec::new(), bytes.len() as u64);
    }

    // If the header magic is wrong the whole file is foreign/corrupt — yield no
    // records and a good_end of HEADER_LEN is meaningless, so report the raw
    // length to avoid truncating a file we don't understand here (open() already
    // verified our own files, so in practice this branch never fires for us).
    if &bytes[..HEADER_LEN as usize] != HEADER_MAGIC {
        return (Vec::new(), bytes.len() as u64);
    }

    let mut records = Vec::new();
    // `pos` walks forward through the frame region; `good_end` trails it at the
    // offset just past the last frame we fully trust.
    let mut pos = HEADER_LEN as usize;
    let mut good_end = HEADER_LEN;

    while pos < bytes.len() {
        match WalRecord::decode(&bytes[pos..]) {
            Ok((rec, consumed)) => {
                records.push(rec);
                pos += consumed;
                good_end = pos as u64;
            }
            // Any decode error means the writer was interrupted here (torn tail)
            // or the data is corrupt. Either way everything from this offset on
            // is untrustworthy in an append-only log, so we stop. `good_end`
            // already points just past the last good frame — that is our
            // truncation point. We do NOT panic and we do NOT try to resync.
            Err(_) => break,
        }
    }

    (records, good_end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Returns a fresh, unique temp directory path for an isolated WAL.
    fn temp_dir(tag: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let mut dir = std::env::temp_dir();
        dir.push(format!("fluxwal-test-{tag}-{pid}-{n}-{nanos}"));
        dir
    }

    /// Cleans up a temp dir, ignoring errors (best-effort).
    fn cleanup(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    fn rec(lsn: u64, pid: u64, ver: u64, blob: &[u8]) -> WalRecord {
        WalRecord {
            lsn,
            player_id: pid,
            version: ver,
            blob: blob.to_vec(),
        }
    }

    #[test]
    fn open_creates_dir_and_header() {
        let dir = temp_dir("open");
        let wal = Wal::open(&dir, true).unwrap();
        // The directory and the active segment file with the magic must exist.
        let seg = dir.join(ACTIVE_SEGMENT);
        assert!(seg.exists(), "segment file should be created");
        let bytes = std::fs::read(&seg).unwrap();
        assert_eq!(&bytes[..HEADER_LEN as usize], HEADER_MAGIC);
        drop(wal);
        cleanup(&dir);
    }

    #[test]
    fn append_returns_highest_lsn() {
        let dir = temp_dir("highlsn");
        let wal = Wal::open(&dir, true).unwrap();
        let hi = wal
            .append_batch(&[rec(1, 1, 1, b"a"), rec(5, 2, 1, b"b"), rec(3, 3, 1, b"c")])
            .unwrap();
        assert_eq!(hi, 5, "append_batch returns the highest LSN in the batch");
        cleanup(&dir);
    }

    #[test]
    fn empty_append_is_noop() {
        let dir = temp_dir("empty");
        let wal = Wal::open(&dir, true).unwrap();
        assert_eq!(wal.append_batch(&[]).unwrap(), 0);
        // A subsequent real append still works and reports the right LSN.
        assert_eq!(wal.append_batch(&[rec(7, 1, 1, b"x")]).unwrap(), 7);
        assert_eq!(wal.append_batch(&[]).unwrap(), 7);
        cleanup(&dir);
    }

    /// Round-trip: append several batches, drop the WAL, reopen, recover, and
    /// assert the exact records come back in order.
    #[test]
    fn roundtrip_append_then_recover() {
        let dir = temp_dir("roundtrip");
        let expected = vec![
            rec(1, 100, 1, b"alpha"),
            rec(2, 101, 1, b""),
            rec(3, 100, 2, &[0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            rec(4, 200, 7, b"the quick brown fox"),
        ];

        {
            let wal = Wal::open(&dir, true).unwrap();
            // Two separate append batches to exercise multi-append accumulation.
            wal.append_batch(&expected[..2]).unwrap();
            wal.append_batch(&expected[2..]).unwrap();
            wal.flush().unwrap();
        } // drop: simulate a clean process restart.

        let wal2 = Wal::open(&dir, true).unwrap();
        let recovered = wal2.recover().unwrap();
        assert_eq!(
            recovered, expected,
            "recover replays exactly the records appended"
        );

        // After recovery, appending continues correctly past the recovered tail.
        let hi = wal2.append_batch(&[rec(5, 300, 1, b"more")]).unwrap();
        assert_eq!(hi, 5);
        let recovered2 = wal2.recover().unwrap();
        let mut expected2 = expected.clone();
        expected2.push(rec(5, 300, 1, b"more"));
        assert_eq!(recovered2, expected2);

        cleanup(&dir);
    }

    /// Torn-tail test: append valid frames, then append GARBAGE bytes directly to
    /// the file (simulating a partial/interrupted write). `recover()` must return
    /// exactly the valid frames AND physically truncate the garbage away.
    #[test]
    fn torn_tail_is_discarded_and_truncated() {
        let dir = temp_dir("torn");
        let good = vec![
            rec(1, 1, 1, b"first"),
            rec(2, 2, 1, b"second"),
            rec(3, 3, 1, b"third"),
        ];

        // Write the good frames through the WAL, then close it so the file
        // handle is released before we append raw garbage out-of-band.
        let good_len_on_disk;
        {
            let wal = Wal::open(&dir, true).unwrap();
            wal.append_batch(&good).unwrap();
            wal.flush().unwrap();
            good_len_on_disk = std::fs::metadata(dir.join(ACTIVE_SEGMENT)).unwrap().len();
        }

        // Simulate a torn partial write: append junk bytes that do not form a
        // valid frame (e.g. a half-written magic / random trailing bytes).
        {
            let mut f = OpenOptions::new()
                .append(true)
                .open(dir.join(ACTIVE_SEGMENT))
                .unwrap();
            // Some bytes that look like the start of a frame but are incomplete,
            // followed by pure noise — neither decodes to a valid WalRecord.
            f.write_all(&[
                0x52, 0x58, 0x4C, 0x46, 0xFF, 0xFF, 0x00, 0x13, 0xDE, 0xAD, 0xBE, 0xEF,
            ])
            .unwrap();
            f.sync_data().unwrap();
        }

        // The file is now larger than the valid prefix.
        let torn_len = std::fs::metadata(dir.join(ACTIVE_SEGMENT)).unwrap().len();
        assert!(
            torn_len > good_len_on_disk,
            "garbage should have grown the file"
        );

        // Recover: must return exactly the good frames.
        let wal2 = Wal::open(&dir, true).unwrap();
        let recovered = wal2.recover().unwrap();
        assert_eq!(recovered, good, "recover returns exactly the valid frames");

        // And must have truncated the garbage: the file is back to the valid
        // prefix length, so a SECOND recovery is identical (deterministic).
        let after_len = std::fs::metadata(dir.join(ACTIVE_SEGMENT)).unwrap().len();
        assert_eq!(
            after_len, good_len_on_disk,
            "recover truncates the torn tail to the last good offset"
        );

        // Appending after a torn-tail recovery lands right after the good frames.
        wal2.append_batch(&[rec(4, 4, 1, b"fourth")]).unwrap();
        let recovered2 = wal2.recover().unwrap();
        let mut expected2 = good.clone();
        expected2.push(rec(4, 4, 1, b"fourth"));
        assert_eq!(recovered2, expected2);

        cleanup(&dir);
    }

    /// Mid-file corruption (a flipped byte inside the first frame's body) is
    /// treated conservatively as end-of-valid-log: only frames before it survive.
    #[test]
    fn mid_file_corruption_stops_at_first_bad_frame() {
        let dir = temp_dir("midcorrupt");
        let recs = vec![
            rec(1, 1, 1, b"keepme"),
            rec(2, 2, 1, b"loseme"),
            rec(3, 3, 1, b"alsolost"),
        ];
        {
            let wal = Wal::open(&dir, true).unwrap();
            wal.append_batch(&recs).unwrap();
            wal.flush().unwrap();
        }

        // Corrupt a byte inside the SECOND frame's body. The first frame stays
        // valid; the corrupt CRC on the second stops the scan there.
        let seg = dir.join(ACTIVE_SEGMENT);
        let mut bytes = std::fs::read(&seg).unwrap();
        // Offset of the first frame end = header + first frame length.
        let first_frame_len = recs[0].encode().len();
        let corrupt_at = HEADER_LEN as usize + first_frame_len + 12; // inside frame 2 body
        bytes[corrupt_at] ^= 0xFF;
        std::fs::write(&seg, &bytes).unwrap();

        let wal2 = Wal::open(&dir, true).unwrap();
        let recovered = wal2.recover().unwrap();
        assert_eq!(
            recovered,
            vec![rec(1, 1, 1, b"keepme")],
            "only the pre-corruption frame survives"
        );

        cleanup(&dir);
    }

    /// Checkpoint drops acknowledged records (`lsn <= up_to`) and keeps the rest,
    /// crash-safely rewriting the segment; survivors replay on the next recover.
    #[test]
    fn checkpoint_discards_acked_keeps_live() {
        let dir = temp_dir("checkpoint");
        let wal = Wal::open(&dir, true).unwrap();
        wal.append_batch(&[
            rec(1, 1, 1, b"a"),
            rec(2, 2, 1, b"b"),
            rec(3, 3, 1, b"c"),
            rec(4, 4, 1, b"d"),
        ])
        .unwrap();

        // Store has durably absorbed up to LSN 2; discard those from the WAL.
        wal.checkpoint(2).unwrap();

        let recovered = wal.recover().unwrap();
        assert_eq!(
            recovered,
            vec![rec(3, 3, 1, b"c"), rec(4, 4, 1, b"d")],
            "only records with lsn > up_to_lsn survive checkpoint"
        );

        // The high-water mark is preserved so further appends stay monotonic.
        let hi = wal.append_batch(&[rec(5, 5, 1, b"e")]).unwrap();
        assert_eq!(hi, 5);

        // Reopening from disk shows the same survivors plus the new append.
        drop(wal);
        let wal2 = Wal::open(&dir, true).unwrap();
        let recovered2 = wal2.recover().unwrap();
        assert_eq!(
            recovered2,
            vec![rec(3, 3, 1, b"c"), rec(4, 4, 1, b"d"), rec(5, 5, 1, b"e")]
        );

        cleanup(&dir);
    }

    /// Checkpointing past every record empties the live set but keeps a valid,
    /// appendable segment.
    #[test]
    fn checkpoint_all_leaves_empty_valid_segment() {
        let dir = temp_dir("checkpoint-all");
        let wal = Wal::open(&dir, true).unwrap();
        wal.append_batch(&[rec(1, 1, 1, b"a"), rec(2, 2, 1, b"b")])
            .unwrap();
        wal.checkpoint(10).unwrap();

        assert!(wal.recover().unwrap().is_empty(), "all records discarded");

        // Segment header is intact and we can still append.
        let seg = std::fs::read(dir.join(ACTIVE_SEGMENT)).unwrap();
        assert_eq!(&seg[..HEADER_LEN as usize], HEADER_MAGIC);
        wal.append_batch(&[rec(11, 1, 1, b"after")]).unwrap();
        assert_eq!(wal.recover().unwrap(), vec![rec(11, 1, 1, b"after")]);

        cleanup(&dir);
    }

    /// fsync=false mode still round-trips after an explicit flush.
    #[test]
    fn nofsync_mode_roundtrips() {
        let dir = temp_dir("nofsync");
        let wal = Wal::open(&dir, false).unwrap();
        wal.append_batch(&[rec(1, 1, 1, b"x"), rec(2, 2, 1, b"y")])
            .unwrap();
        wal.flush().unwrap();
        assert_eq!(
            wal.recover().unwrap(),
            vec![rec(1, 1, 1, b"x"), rec(2, 2, 1, b"y")]
        );
        cleanup(&dir);
    }
}
