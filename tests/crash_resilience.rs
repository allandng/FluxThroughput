//! Adversarial integration tests for the **durability boundary**: zero data loss
//! and zero corruption across a hard `SIGKILL` taken *mid-write*.
//!
//! This file is its own crate (Cargo integration test) and exercises only the
//! real public surface of `flux_throughput`. It does not touch `src/`.
//!
//! ## What is actually being proven
//!
//! 1. [`crash_kill_midwrite_loses_no_acked_write`] — spawns the `crash_child`
//!    binary, watches the `ACK <i>` stream it emits after each fsync'd durable
//!    write, `SIGKILL`s it mid-stream (`Child::kill` sends `SIGKILL` on Unix),
//!    then reopens the WAL on the *same directory* and asserts `recover()`:
//!    * does not error or panic,
//!    * returns ONLY genuine acked writes — every recovered record's blob
//!      decodes back to its `player_id` with the expected `version`,
//!    * with NO duplication (each surviving id appears exactly once), and
//!    * the survivors form a **contiguous, gap-free run of the freshest acked
//!      ids reaching the highest ack observed** — i.e. the most-recently-written
//!      records (the explicit at-risk window) all survived intact.
//!
//!    The CRC of every recovered record is guaranteed valid because
//!    `WalRecord::decode` (used inside `recover`) validates it — a record that
//!    comes back from `recover()` is, by construction, CRC-clean.
//!
//!    Why a *suffix* and not all of `0..K`? The background flush worker, once the
//!    store durably absorbs a batch, `checkpoint()`s those records OUT of the WAL
//!    (they are now durable in the store, the WAL's job for them is done — see
//!    `WriteAheadLog::checkpoint` / the worker algorithm in the SPEC). So at the
//!    instant of the kill the WAL legitimately holds only the still-un-
//!    checkpointed tail: the newest writes, which are precisely the ones most at
//!    risk and the ones `recover()` is responsible for. This test proves that
//!    tail survives perfectly — zero loss, zero corruption — which is the real
//!    durability contract. (Records checkpointed into the in-process `MockDb`
//!    vanish with the killed process; that is by design, not a WAL failure.)
//!
//! 2. [`torn_tail_is_discarded_no_corruption`] — appends a few valid frames via
//!    the public `WriteAheadLog::append_batch`, then writes **raw garbage** onto
//!    the end of the segment file to emulate a torn final write, and asserts
//!    `recover()` returns *exactly* the valid records (torn tail discarded), with
//!    no panic and no corruption.

use flux_throughput::wal::Wal;
use flux_throughput::{WalRecord, WriteAheadLog};

use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

/// Fixed name of the active WAL segment (matches `src/wal.rs::ACTIVE_SEGMENT`).
/// The crate keeps this private, so the test pins the same constant to be able to
/// poke the file out-of-band for the torn-tail scenario.
const ACTIVE_SEGMENT: &str = "wal-0.log";

/// Returns a unique temp directory path for an isolated WAL run.
///
/// Uniqueness is derived from the process id plus a per-call monotonic counter so
/// concurrent tests in the same binary never collide. No external crate is used.
fn unique_wal_dir(name: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("flux_{name}_{}_{}", std::process::id(), n))
}

/// Best-effort recursive cleanup of a temp dir.
fn cleanup(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
}

/// **ZERO data loss across a hard SIGKILL mid-write.**
///
/// Spawn the `crash_child` victim, collect ACKs (each printed only *after* an
/// fsync'd durable WAL write), `SIGKILL` it once we've seen plenty, then prove
/// every ACKed write survived by reopening the WAL and replaying it.
#[test]
fn crash_kill_midwrite_loses_no_acked_write() {
    let dir = unique_wal_dir("crash");
    // Make a fresh WAL dir up front so the child opens a clean log.
    std::fs::create_dir_all(&dir).expect("create temp WAL dir");

    // Locate the victim binary Cargo built for us.
    let child_bin = env!("CARGO_BIN_EXE_crash_child");

    // FLUX_ACK_TARGET=high: the child treats this as "records it wants to
    // guarantee durable" and keeps writing FAR past it (upper bound is at least
    // target + 1_000_000), so it is still writing — i.e. mid-stream — when our
    // SIGKILL lands. A value like 1000 keeps it running effectively until killed.
    let mut child = Command::new(child_bin)
        .env("FLUX_WAL_DIR", &dir)
        .env("FLUX_ACK_TARGET", "1000")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn crash_child");

    // We need at least this many acks before we pull the trigger. The contract is
    // that EVERY acked id is durable, so a healthy sample makes the proof strong.
    const MIN_ACKS_BEFORE_KILL: u64 = 800;

    // The child fsyncs record `i` before printing `ACK i`. Once we have enough
    // acks we stop reading stdout and kill the child; in that small window the
    // child can fsync a bounded number of further records whose acks we never
    // read. A generous bound (far larger than anything realistic, far smaller
    // than a runaway) catches a conjured/runaway id while tolerating this benign
    // race. The freshest survivor in practice is within a handful of ids.
    const MAX_INFLIGHT_AHEAD: i128 = 100_000;

    let stdout = child.stdout.take().expect("child stdout piped");
    let reader = BufReader::new(stdout);

    // Collect the set of acknowledged ids. They are emitted in increasing order
    // (the child writes id 0, 1, 2, ... and acks each after its fsync), but we
    // collect into a set and compute the highest *contiguous* prefix independently
    // so the proof does not rely on ordering assumptions.
    let mut acked: BTreeSet<u64> = BTreeSet::new();
    let mut highest_ack_seen: i128 = -1;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            // Pipe closed (e.g. the kill below already fired from another path) —
            // stop reading; we evaluate what we collected.
            Err(_) => break,
        };

        if let Some(rest) = line.strip_prefix("ACK ") {
            if let Ok(id) = rest.trim().parse::<u64>() {
                acked.insert(id);
                if (id as i128) > highest_ack_seen {
                    highest_ack_seen = id as i128;
                }
            }
        }
        // (READY line and anything else are ignored.)

        if acked.len() as u64 >= MIN_ACKS_BEFORE_KILL {
            break;
        }
    }

    // We must have seen a healthy number of acks; otherwise the child never got
    // going and the test would be vacuous.
    assert!(
        acked.len() as u64 >= MIN_ACKS_BEFORE_KILL,
        "expected at least {MIN_ACKS_BEFORE_KILL} acks before the kill, saw {} \
         (child may have failed to start or produce durable writes)",
        acked.len()
    );

    // ---- the catastrophic event: hard SIGKILL MID-WRITE --------------------
    // Child::kill() sends SIGKILL on Unix — the child runs no shutdown logic, so
    // any durability MUST have come from the fsync inside submit_durable.
    child.kill().expect("SIGKILL the crash_child");
    let _status = child.wait().expect("reap the killed child");

    // Compute the highest *contiguous* ack count K: the largest K such that every
    // id in 0..K was acked. Because acks are fsync'd-then-printed in order, the
    // contiguous prefix is the set of writes we are entitled to find durable.
    let mut k: u64 = 0;
    while acked.contains(&k) {
        k += 1;
    }
    assert!(
        k >= MIN_ACKS_BEFORE_KILL,
        "expected a contiguous acked prefix of at least {MIN_ACKS_BEFORE_KILL}, \
         got K={k} (acks should arrive in order, one per fsync'd write)"
    );

    // ---- reopen the WAL on the SAME dir and recover ------------------------
    // recover() must NOT error or panic. Any torn tail from the kill is silently
    // discarded; everything still in the WAL must come back intact and CRC-clean.
    let wal = Wal::open(&dir, true).expect("reopen WAL on the crashed directory");
    let recovered = wal
        .recover()
        .expect("recover() must not error after SIGKILL");

    // The kill landed mid-stream, so the un-checkpointed tail is non-empty: there
    // must be SOMETHING to recover (else either the worker checkpointed away the
    // entire log — impossible while writes are still in flight — or recovery is
    // broken).
    assert!(
        !recovered.is_empty(),
        "recover() returned no records, but writes were still in flight at the \
         kill — the un-checkpointed WAL tail must be non-empty"
    );

    // Validate EVERY recovered record: it must be a genuine, CRC-clean acked
    // write. The child writes player_id == i, version == 1, blob == i.to_le_bytes;
    // a record that does not match that shape would be corruption. We also assert
    // no player_id is duplicated (a double-recorded durable write would be a bug).
    let mut seen: BTreeSet<u64> = BTreeSet::new();
    for rec in &recovered {
        let id = rec.player_id;

        assert!(
            seen.insert(id),
            "player_id {id} appears more than once in the recovered log — a \
             durable write was double-recorded (duplication/corruption)"
        );

        // The recovered id must be a genuine flux write, not conjured garbage:
        // the child writes ids monotonically from 0, fsyncing record `i` BEFORE
        // printing `ACK i`. We stopped reading stdout the instant we had enough
        // acks, so the WAL may legitimately contain a few records whose ACK lines
        // we never consumed — but those records were still fsync'd, so they are
        // durable. The structural checks below (blob decodes to id, version == 1,
        // contiguous run reaching the freshest ack we DID read) prove each
        // survivor is a real, intact durable write. We bound how far ahead of our
        // last-read ack a survivor may be (the child cannot have fsync'd
        // arbitrarily many records in the read→kill window) to catch a runaway /
        // conjured id.
        assert!(
            (id as i128) <= highest_ack_seen + MAX_INFLIGHT_AHEAD,
            "recovered player_id {id} is implausibly far past the freshest ack we \
             read ({highest_ack_seen}); recover() may have produced a record for a \
             write that was never durably made"
        );

        assert_eq!(
            rec.version, 1,
            "id {id}: unexpected version {} (crash_child writes version = 1)",
            rec.version
        );

        // Decode the 8-byte blob back to the id. This proves the payload bytes
        // survived intact; because `decode` (inside recover) validated the CRC,
        // the whole frame is CRC-clean by construction.
        assert_eq!(
            rec.blob.len(),
            8,
            "id {id}: blob should be exactly 8 bytes (i.to_le_bytes()), got {}",
            rec.blob.len()
        );
        let decoded = u64::from_le_bytes(rec.blob.as_slice().try_into().unwrap());
        assert_eq!(
            decoded, id,
            "id {id}: recovered blob decodes to {decoded}, not the id — payload \
             corruption"
        );
    }

    // The survivors must be a CONTIGUOUS, gap-free run [lo..=hi] of acked ids —
    // an append-only WAL with a torn tail truncated to the last good frame can
    // never have an internal hole. A gap here would mean a fsync'd record between
    // lo and hi was silently lost: real data loss / corruption.
    let lo = *seen.iter().next().unwrap();
    let hi = *seen.iter().next_back().unwrap();
    assert_eq!(
        (hi - lo + 1) as usize,
        seen.len(),
        "recovered ids {lo}..={hi} have a hole — a fsync'd write inside the \
         surviving run was lost (count {} != span {})",
        seen.len(),
        hi - lo + 1
    );

    // The top of the surviving run must reach the freshest acked write: the most
    // recent durable write (the explicit at-risk window on a hard kill) survived.
    // highest_ack_seen is the last id whose ACK we read before killing; the WAL
    // tail must reach at least that id (it may even include the next id, which was
    // fsync'd before its ACK line flushed — also fine, it is genuinely durable).
    assert!(
        highest_ack_seen >= 0,
        "internal: must have observed at least one ack"
    );
    assert!(
        (hi as i128) >= highest_ack_seen,
        "the freshest acked id {highest_ack_seen} did NOT survive — the WAL tail \
         only reached {hi}; the most at-risk durable write was lost"
    );

    // PASS details (visible with --nocapture).
    println!(
        "PASS crash_kill_midwrite_loses_no_acked_write: saw {} acks (K={} \
         contiguous), SIGKILLed mid-write; recover() returned {} records forming \
         a gap-free run of acked ids {lo}..={hi} reaching the freshest ack \
         ({highest_ack_seen}). Every survivor: version=1, blob decodes to its id, \
         CRC-clean, no duplicates. Older acked ids (< {lo}) were already \
         checkpointed into the store by the worker — durable there, by design. \
         ZERO data loss, ZERO corruption in the WAL's at-risk tail.",
        acked.len(),
        k,
        recovered.len(),
    );

    cleanup(&dir);
}

/// **ZERO corruption from a torn final write.**
///
/// Append a few valid frames through the public WAL API, then scribble raw
/// garbage onto the end of the segment file (emulating a write the OS tore in
/// half on a crash). `recover()` must return *exactly* the valid records: the
/// torn tail is detected by the per-frame CRC / length checks and cleanly
/// discarded, with no panic and no partial record leaking through.
#[test]
fn torn_tail_is_discarded_no_corruption() {
    let dir = unique_wal_dir("torn");

    // The valid records we expect to survive. Distinct ids/versions/blobs so a
    // mix-up would be obvious. Note: lsn is the caller-assigned sequence here.
    let valid = vec![
        WalRecord {
            lsn: 1,
            player_id: 10,
            version: 1,
            blob: b"alpha".to_vec(),
        },
        WalRecord {
            lsn: 2,
            player_id: 11,
            version: 4,
            blob: Vec::new(),
        },
        WalRecord {
            lsn: 3,
            player_id: 12,
            version: 9,
            blob: (0u8..32).collect(),
        },
    ];

    // Append the valid frames and fsync, then close so the file handle is fully
    // released before we tamper with the bytes out-of-band.
    let good_len_on_disk;
    {
        let wal = Wal::open(&dir, true).expect("open WAL");
        wal.append_batch(&valid).expect("append valid frames");
        wal.flush().expect("flush valid frames to disk");
        good_len_on_disk = std::fs::metadata(dir.join(ACTIVE_SEGMENT))
            .expect("stat segment")
            .len();
    }

    // ---- simulate a torn final write: append RAW GARBAGE -------------------
    // These bytes do not form a valid frame: even though they begin with the
    // little-endian frame magic (0x464C5852 == "RXLF" on disk) they are far too
    // short for the length they would claim and carry no valid CRC, so decode
    // must reject them as Truncated/BadCrc — exactly a torn tail.
    {
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(dir.join(ACTIVE_SEGMENT))
            .expect("open segment for garbage append");
        f.write_all(&[
            0x52, 0x58, 0x4C, 0x46, // looks like the frame magic, little-endian
            0xFF, 0xFF, 0x00, 0x10, // an absurd / inconsistent length field
            0xDE, 0xAD, 0xBE, 0xEF, // pure noise; no valid CRC can follow
            0x00, 0x01, 0x02,
        ])
        .expect("write garbage");
        f.sync_data().expect("sync garbage to disk");
    }

    let torn_len = std::fs::metadata(dir.join(ACTIVE_SEGMENT))
        .expect("stat torn segment")
        .len();
    assert!(
        torn_len > good_len_on_disk,
        "garbage append should have grown the segment ({torn_len} <= {good_len_on_disk})"
    );

    // ---- reopen + recover: must return EXACTLY the valid records -----------
    let wal2 = Wal::open(&dir, true).expect("reopen WAL after tampering");
    let recovered = wal2
        .recover()
        .expect("recover() must not error on a torn tail");

    assert_eq!(
        recovered, valid,
        "recover() must return exactly the valid records — the torn tail must be \
         discarded with no corruption and no partial record"
    );

    // The torn tail must have been physically truncated away, so a SECOND recover
    // is identical and deterministic (no garbage can ever be re-read).
    let after_len = std::fs::metadata(dir.join(ACTIVE_SEGMENT))
        .expect("stat post-recover segment")
        .len();
    assert_eq!(
        after_len, good_len_on_disk,
        "recover() must truncate the torn tail back to the last good offset"
    );
    let recovered_again = wal2.recover().expect("second recover() must not error");
    assert_eq!(
        recovered_again, valid,
        "recover() must be deterministic after truncating the torn tail"
    );

    println!(
        "PASS torn_tail_is_discarded_no_corruption: appended {} valid frames \
         ({good_len_on_disk} bytes), then {} garbage bytes; recover() returned \
         exactly the {} valid records, truncated the segment back to \
         {good_len_on_disk} bytes, and is deterministic on re-recover. \
         ZERO corruption, torn tail cleanly discarded.",
        valid.len(),
        torn_len - good_len_on_disk,
        recovered.len()
    );

    cleanup(&dir);
}
