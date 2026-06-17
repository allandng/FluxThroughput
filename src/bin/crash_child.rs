//! `crash_child` binary — the `SIGKILL` victim subprocess for the crash test.
//! **[impl:api].**
//!
//! This process exists to *prove the durability boundary*. The parent test:
//!
//! 1. spawns this binary with `FLUX_WAL_DIR` (a fresh WAL directory) and
//!    `FLUX_ACK_TARGET` (a number of records the parent wants durably written);
//! 2. reads the `ACK <i>` lines this process flushes to stdout after each
//!    durable write;
//! 3. `SIGKILL`s this process **mid-stream**, well before it would finish; then
//! 4. reopens the WAL itself and asserts `recover()` replays *exactly* the
//!    records that were ACKed before the kill — no torn tail, no lost ack'd
//!    record.
//!
//! Because the kill is a hard `SIGKILL`, this process can run no shutdown logic.
//! That is the whole point: durability must come from the WAL fsync inside
//! [`FluxEngine::submit_durable`], NOT from any clean-exit path. Accordingly,
//! this binary **never exits cleanly on its own** — it intentionally keeps
//! writing past the parent's target so the kill always lands mid-stream.
//!
//! ## The contract this binary upholds
//!
//! For every `i`, the sequence is strictly: `submit_durable(i)` returns `Ok`
//! (the record is WAL'd **and fsync'd** to stable storage) **before** the
//! matching `ACK i` is printed and flushed. Therefore any `ACK i` the parent
//! observed corresponds to a record guaranteed to be on disk and recoverable.

use flux_throughput::store::MockDb;
use flux_throughput::{FluxConfig, FluxEngine};
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

fn main() {
    // ---- read the harness parameters from the environment ------------------
    let wal_dir = match std::env::var("FLUX_WAL_DIR") {
        Ok(d) if !d.is_empty() => d,
        _ => {
            eprintln!("crash_child: FLUX_WAL_DIR must be set to a writable directory");
            std::process::exit(2);
        }
    };

    // The number of records the parent wants to *guarantee* are durable. We will
    // keep writing well beyond this so the parent can SIGKILL us mid-stream after
    // it has seen at least this many ACKs. A missing/garbage value defaults to a
    // large target so we simply run until killed.
    let ack_target: u64 = std::env::var("FLUX_ACK_TARGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    // ---- open an engine with STRONG durability (fsync = true) --------------
    // fsync=true is mandatory here: it is the property under test. The store is a
    // throwaway MockDb — the crash test only cares about the WAL, which is the
    // durability boundary; the store side is irrelevant to recover().
    let config = FluxConfig {
        // Strong durability is the whole point of this victim.
        wal_fsync: true,
        // A modest ring/batch; this process never relies on the worker for
        // durability (submit_durable fsyncs synchronously), so sizing is minor.
        ring_capacity: 1 << 12,
        flush_max_batch: 256,
        flush_interval: Duration::from_millis(50),
        ..FluxConfig::default()
    }
    .with_wal_dir(&wal_dir);

    let store: Arc<dyn flux_throughput::PersistenceStore> = Arc::new(MockDb::new());

    let engine = match FluxEngine::start(config, store) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("crash_child: engine failed to start on {wal_dir}: {e}");
            std::process::exit(2);
        }
    };

    // Lock stdout once and keep an explicit handle so we can FLUSH after every
    // ACK — the parent must see each ack in real time to know which records are
    // guaranteed durable at the instant it decides to kill us.
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    // A "ready" marker before the first durable write, so the parent can sync up
    // on a known line if it wants to.
    let _ = writeln!(out, "READY {ack_target}");
    let _ = out.flush();

    // ---- the durable-write loop --------------------------------------------
    // We run to a deliberately large upper bound (far beyond `ack_target`) so the
    // process is still writing when the SIGKILL arrives. It NEVER exits cleanly
    // under normal test conditions — reaching the upper bound at all would mean
    // the parent never killed us, which is itself a test failure the parent
    // detects (we exit non-zero in that case).
    let upper_bound = ack_target.saturating_mul(1000).max(ack_target + 1_000_000);

    let mut i: u64 = 0;
    while i < upper_bound {
        // 8-byte payload encoding `i` so the parent can verify exact contents on
        // recover(): blob == i.to_le_bytes().
        let blob = i.to_le_bytes().to_vec();

        // Durable write: WAL-append + fsync happen synchronously inside this call
        // and MUST complete before we print the ACK. version = 1 (each id written
        // once); player_id = i so recovered records map 1:1 to acks.
        match engine.submit_durable(i, 1, blob) {
            Ok(()) => {
                // The record is now on stable storage. Announce it and flush so
                // the parent observes the ack immediately.
                if writeln!(out, "ACK {i}").is_err() {
                    // Parent closed the pipe (likely about to kill us). Stop
                    // announcing but keep the process alive so the kill lands.
                    break;
                }
                if out.flush().is_err() {
                    break;
                }
            }
            Err(e) => {
                // A durable write failed: do NOT ack it (it is not guaranteed on
                // disk). Report to stderr and keep going — the parent only trusts
                // ACKed records.
                let _ = writeln!(std::io::stderr(), "ERR {i} {e}");
                let _ = std::io::stderr().flush();
            }
        }

        i += 1;
    }

    // ---- we should never reach here under the crash test -------------------
    // If we did, the parent failed to SIGKILL us in time. Park briefly to give a
    // slow parent a last chance to kill us, then exit NON-ZERO so the test fails
    // loudly rather than passing on a vacuous run.
    let _ = writeln!(
        std::io::stderr(),
        "crash_child: reached upper bound {upper_bound} without being killed — \
         the parent was supposed to SIGKILL this process mid-stream",
    );
    let _ = std::io::stderr().flush();
    // Hold here so a late SIGKILL still works; if it never comes, exit non-zero.
    std::thread::sleep(Duration::from_secs(5));
    std::process::exit(3);
}
