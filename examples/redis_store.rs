//! `redis_store` — a clean REFERENCE implementation showing **precisely how to
//! back `PersistenceStore` with Redis** in a real cluster, that STILL compiles
//! and runs **offline with zero dependencies**.
//!
//! ## The trick: abstract the connection behind a tiny local trait
//!
//! FluxThroughput's default build is intentionally zero-dependency and fully
//! offline (adding the real `redis` crate would force a network index fetch and
//! break offline builds). So this example does NOT depend on `redis`. Instead it
//! defines a minimal local [`RedisConn`] trait that captures exactly the two
//! operations a real Redis backend needs — a pipelined multi-SET and a GET — and
//! ships an in-memory fake ([`FakeRedis`]) implementing it. The
//! [`RedisStore<C>`] is generic over `C: RedisConn` and implements
//! `PersistenceStore`, so **swapping the real Redis client in is a one-line
//! type change** (`RedisStore<FakeRedis>` -> `RedisStore<RealRedisConn>`) with
//! no edits to the store logic at all.
//!
//! ```text
//!   PersistenceStore (the seam)
//!        ▲
//!        │  implemented generically by
//!   RedisStore<C: RedisConn>
//!        │  delegates to
//!        ▼
//!   trait RedisConn { pipeline_set(...); get(...) }
//!        ├── FakeRedis        (in-memory; this example builds/runs offline)
//!        └── RealRedisConn    (wraps redis::Connection; see PRODUCTION WIRING)
//! ```
//!
//! ## Key scheme
//!
//! Each player is stored under `"player:{id}"`. The value is a small
//! self-describing blob: an 8-byte little-endian `version` followed by the
//! opaque payload. Storing the version *inside the value* lets the store enforce
//! **last-write-wins by version** (read-modify guard) and lets `load` reconstruct
//! a full [`PlayerState`]. A production deployment that wants server-side LWW can
//! instead use a Lua `EVAL` (compare-and-set on version) or a per-player Redis
//! Hash with `HSETNX`/Lua; this example keeps it to plain SET/GET for clarity.
//!
//! Run it:
//!
//! ```text
//!   cargo run --example redis_store        # uses the in-memory FakeRedis
//! ```

use flux_throughput::error::StoreError;
use flux_throughput::PersistenceStore;
use flux_throughput::{FluxConfig, FluxEngine, PlayerId, PlayerState, SubmitOutcome};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ===========================================================================
// The connection seam: a tiny trait the real `redis` client satisfies trivially.
// ===========================================================================

/// The minimal Redis surface [`RedisStore`] needs. Deliberately tiny so that a
/// real `redis::Connection` wrapper can implement it in a few lines (see the
/// PRODUCTION WIRING block below), and so the example can ship a pure in-memory
/// fake for offline builds.
///
/// `&mut self`: a Redis connection is not thread-safe to share, so each method
/// takes `&mut self`. [`RedisStore`] holds the connection behind a `Mutex`,
/// which is correct and idiomatic on the DB side of the pipeline (it can only
/// ever slow the single background worker, never a `submit()` caller — see the
/// isolation note on [`RedisStore`]).
pub trait RedisConn: Send {
    /// Pipelined multi-SET: write every `(key, value)` pair in one round trip.
    /// Maps to `redis::pipe().set(k, v)....query(conn)` against a real server.
    /// Returns `Err(msg)` on any transport/protocol failure (retryable upstream).
    fn pipeline_set(&mut self, kv: &[(String, Vec<u8>)]) -> Result<(), String>;

    /// Single GET. Returns `Ok(None)` for a missing key. Maps to
    /// `redis::cmd("GET").arg(key).query(conn)` returning `Option<Vec<u8>>`.
    fn get(&mut self, key: &str) -> Result<Option<Vec<u8>>, String>;
}

// ===========================================================================
// In-memory fake so the example builds & runs offline with zero deps.
// ===========================================================================

/// A pure in-memory stand-in for a Redis server. Behaves like a `String`->`bytes`
/// keyspace with pipelined SET and GET. Used ONLY so this example compiles and
/// runs offline; it is NOT part of the production path.
#[derive(Default)]
pub struct FakeRedis {
    map: HashMap<String, Vec<u8>>,
}

impl FakeRedis {
    pub fn new() -> Self {
        FakeRedis {
            map: HashMap::new(),
        }
    }
}

impl RedisConn for FakeRedis {
    fn pipeline_set(&mut self, kv: &[(String, Vec<u8>)]) -> Result<(), String> {
        // A real pipeline is atomic-ish per connection; for the fake we just
        // apply each SET in order. Last write to a key wins within the batch,
        // matching real Redis pipeline semantics.
        for (k, v) in kv {
            self.map.insert(k.clone(), v.clone());
        }
        Ok(())
    }

    fn get(&mut self, key: &str) -> Result<Option<Vec<u8>>, String> {
        Ok(self.map.get(key).cloned())
    }
}

// ===========================================================================
// Value codec: [u64 version][blob bytes]
// ===========================================================================

/// Redis key for a player. The `"player:{id}"` scheme keeps the keyspace flat
/// and greppable, and lets ops tooling scan `player:*` to size the dataset.
fn player_key(id: PlayerId) -> String {
    format!("player:{id}")
}

/// Encodes a `PlayerState`'s value: 8-byte little-endian version, then the blob.
/// The id is in the key, so it is not repeated in the value.
fn encode_value(st: &PlayerState) -> Vec<u8> {
    let mut v = Vec::with_capacity(8 + st.blob.len());
    v.extend_from_slice(&st.version.to_le_bytes());
    v.extend_from_slice(&st.blob);
    v
}

/// Decodes a stored value back into `(version, blob)`. Returns `None` if the
/// value is too short to contain the version header (treated as absent/corrupt).
fn decode_value(bytes: &[u8]) -> Option<(u64, Vec<u8>)> {
    if bytes.len() < 8 {
        return None;
    }
    let version = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let blob = bytes[8..].to_vec();
    Some((version, blob))
}

// ===========================================================================
// RedisStore<C> — generic over the connection, implements PersistenceStore.
// ===========================================================================

/// A `PersistenceStore` backed by anything implementing [`RedisConn`].
///
/// ## Why the `Mutex<C>` is fine here
///
/// A Redis connection is single-threaded, so we serialize access through a
/// `Mutex`. This lives on the **store side** of the pipeline, whose only caller
/// is the single background flush worker (plus synchronous `flush_now`). It can
/// never block a `submit()` caller: the lock-free ring decouples the hot main
/// loop from everything to its right. This is exactly the contract the crate's
/// `MockDb` relies on (`src/store.rs`) — DB-side locking is allowed and expected.
///
/// A real high-throughput deployment would hold a *connection pool* (e.g.
/// `r2d2_redis`) and check a connection out per batch instead of one shared
/// connection; the `RedisConn` seam abstracts that away — `pipeline_set` /
/// `get` would simply borrow from the pool internally.
pub struct RedisStore<C: RedisConn> {
    conn: Mutex<C>,
}

impl<C: RedisConn> RedisStore<C> {
    /// Wraps an already-connected `RedisConn`.
    pub fn new(conn: C) -> Self {
        RedisStore {
            conn: Mutex::new(conn),
        }
    }
}

impl<C: RedisConn> PersistenceStore for RedisStore<C> {
    /// Persists a batch with a single pipelined multi-SET round trip.
    ///
    /// We do NOT do a server round trip per row: every state in the batch is
    /// encoded to `player:{id} -> [version|blob]` and written in one
    /// `pipeline_set`, which is what makes Redis-backed persistence cheap enough
    /// to keep up with the worker. Any transport error is mapped to a RETRYABLE
    /// [`StoreError::Transient`] so the worker's retry/requeue machinery keeps the
    /// data safe (still dirty in the cache + durable in the crate's WAL) rather
    /// than dropping it.
    ///
    /// Last-write-wins note: within one batch the cache has already coalesced to
    /// one entry per id under LWW, so a plain SET is correct. Across batches, the
    /// worker only ever flushes monotonically-newer versions for a given id, so
    /// an unconditional SET preserves LWW too. (If you need to defend against
    /// out-of-order writers, swap the SET for a Lua compare-and-set on the stored
    /// version — see the PRODUCTION WIRING block.)
    fn persist_batch(&self, states: &[PlayerState]) -> Result<(), StoreError> {
        if states.is_empty() {
            return Ok(());
        }

        // Build all key/value pairs up front, then ship them in one pipeline.
        let kv: Vec<(String, Vec<u8>)> = states
            .iter()
            .map(|st| (player_key(st.player_id), encode_value(st)))
            .collect();

        let mut conn = self
            .conn
            .lock()
            .map_err(|_| StoreError::Fatal("RedisStore mutex poisoned".to_string()))?;

        conn.pipeline_set(&kv)
            .map_err(|e| StoreError::Transient(format!("redis pipeline SET failed: {e}")))?;

        Ok(())
    }

    /// Loads a single player's persisted state via GET, reconstructing the
    /// `PlayerState` from the `[version|blob]` value. A missing key -> `None`.
    fn load(&self, id: PlayerId) -> Option<PlayerState> {
        let mut conn = self.conn.lock().ok()?;
        let raw = conn.get(&player_key(id)).ok()??;
        let (version, blob) = decode_value(&raw)?;
        Some(PlayerState {
            player_id: id,
            version,
            blob,
        })
    }

    /// Number of distinct persisted players.
    ///
    /// IMPORTANT (production note): there is deliberately no cheap, exact "count
    /// my player:* keys" in Redis. `KEYS player:*` is O(N) and blocks the server;
    /// the right tool is `SCAN` with a `MATCH player:*` cursor, or maintaining a
    /// counter / a `SET` of known ids alongside the data. To keep the `RedisConn`
    /// seam minimal (and because the engine never calls `count()` on the hot
    /// path), this reference returns 0. Implement it with `SCAN` if your ops
    /// tooling needs it. The disk_kv_store example shows an exact `count()`.
    fn count(&self) -> usize {
        0
    }
}

// ===========================================================================
// main() — demo the seam end-to-end against the in-memory FakeRedis.
// ===========================================================================

const PLAYERS: u64 = 500;
const UPDATES_PER_PLAYER: u64 = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let wal_dir = std::env::temp_dir().join(format!("flux_redis_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&wal_dir);

    println!("== FluxThroughput redis_store example (offline, in-memory FakeRedis) ==");
    println!("wal dir : {}", wal_dir.display());

    // Construct RedisStore over the in-memory fake. To run against a REAL Redis
    // cluster, this is the ONE line that changes — see PRODUCTION WIRING below:
    //
    //     let store = Arc::new(RedisStore::new(RealRedisConn::connect("redis://...")?));
    //
    // Everything else (engine wiring, submit loop, flush, verification) is identical.
    let store = Arc::new(RedisStore::new(FakeRedis::new()));
    let store_dyn: Arc<dyn PersistenceStore> = store.clone();

    let config = FluxConfig {
        ring_capacity: 1 << 14,
        flush_max_batch: 1024,
        wal_fsync: true,
        ..FluxConfig::default()
    }
    .with_wal_dir(&wal_dir);

    let engine = FluxEngine::start(config, store_dyn)?;

    // Fire-and-forget submit burst: versions 1..=UPDATES_PER_PLAYER per player,
    // ascending, so the highest version wins for every id.
    let mut submitted = 0u64;
    let mut dropped = 0u64;
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

    let flushed = engine.flush_now()?;
    engine.shutdown()?;

    println!(
        "submitted {submitted} updates ({dropped} shed), flush_now persisted {flushed} records",
    );

    // Verify the persisted state via the store's own GET path. The store outlives
    // the engine (we kept an Arc), so we can read it directly.
    let expected_version = UPDATES_PER_PLAYER;
    let expected_blob = expected_version.to_le_bytes().to_vec();
    let mut checked = 0u64;
    for player_id in 0..PLAYERS {
        let got = store
            .load(player_id)
            .unwrap_or_else(|| panic!("player {player_id} missing from redis store"));
        assert_eq!(
            got.version, expected_version,
            "player {player_id} version {} != expected {expected_version}",
            got.version,
        );
        assert_eq!(got.blob, expected_blob, "player {player_id} blob mismatch");
        checked += 1;
    }

    let _ = std::fs::remove_dir_all(&wal_dir);

    println!(
        "\nSUCCESS: {checked} players verified at version {expected_version} via the RedisStore \
         GET path; the same RedisStore<C> works against a real cluster by swapping FakeRedis for \
         a redis::Connection wrapper (see PRODUCTION WIRING in the source).",
    );
    Ok(())
}

// ===========================================================================
// ========================  PRODUCTION WIRING  ==============================
// ===========================================================================
//
// Everything below is a REFERENCE for backing this store with a REAL Redis
// cluster. It is left as a commented block (NOT behind a `cfg`, NOT compiled) so
// the default build stays zero-dependency and offline. To go live you (1) add
// the dependency, (2) paste the `RealRedisConn` impl, and (3) change one line in
// `main` to construct `RedisStore::new(RealRedisConn::connect(url)?)`.
//
// ---------------------------------------------------------------------------
// (1) Cargo.toml — add the dependency (this is the ONLY change to Cargo.toml,
//     and it is intentionally NOT done by default):
//
//         [dependencies]
//         redis = "0.27"
//
//     NOTE ON WHY THIS IS NOT A DEFAULT DEPENDENCY: FluxThroughput's headline
//     property is a std-only, zero-dependency, fully-offline build (the entire
//     test suite, demo, and these examples compile and run with no network).
//     Adding `redis` pulls a transitive dependency tree that requires a network
//     crates.io index fetch on first build, which would break offline / air-gapped
//     builds and CI. Enabling Redis is therefore a deliberate, opt-in step the
//     embedder takes for their own deployment — it is not forced on every user of
//     the crate. The `RedisConn` seam exists precisely so the store logic is
//     fully written, tested, and demonstrated WITHOUT that dependency.
//
// ---------------------------------------------------------------------------
// (2) The real connection adapter. Paste this in (uncommented) once `redis` is a
//     dependency. It implements the SAME `RedisConn` trait the example already
//     uses, so `RedisStore<RealRedisConn>` works with zero changes to the store:
//
//     use redis::Commands; // brings get/set onto Connection
//
//     /// Wraps a live, single-threaded `redis::Connection`.
//     pub struct RealRedisConn {
//         conn: redis::Connection,
//     }
//
//     impl RealRedisConn {
//         /// Connects to a Redis server, e.g.
//         /// `RealRedisConn::connect("redis://127.0.0.1:6379/")`.
//         pub fn connect(url: &str) -> Result<Self, String> {
//             let client = redis::Client::open(url).map_err(|e| e.to_string())?;
//             let conn = client.get_connection().map_err(|e| e.to_string())?;
//             Ok(RealRedisConn { conn })
//         }
//     }
//
//     impl RedisConn for RealRedisConn {
//         fn pipeline_set(&mut self, kv: &[(String, Vec<u8>)]) -> Result<(), String> {
//             // One pipeline = one round trip for the whole batch.
//             let mut pipe = redis::pipe();
//             for (k, v) in kv {
//                 pipe.set(k, v.as_slice());      // SET player:{id} <value>
//                 // To bound memory, set a TTL instead: pipe.set_ex(k, v, ttl_secs);
//             }
//             pipe.query::<()>(&mut self.conn).map_err(|e| e.to_string())
//         }
//
//         fn get(&mut self, key: &str) -> Result<Option<Vec<u8>>, String> {
//             // GET player:{id} -> Option<Vec<u8>> (None when the key is absent).
//             let val: Option<Vec<u8>> = self.conn.get(key).map_err(|e| e.to_string())?;
//             Ok(val)
//         }
//     }
//
//     // Bulk reads (e.g. a warm-cache prefill) would add an MGET method to the
//     // trait and call it like:
//     //
//     //     let vals: Vec<Option<Vec<u8>>> =
//     //         redis::cmd("MGET").arg(&keys).query(&mut self.conn)?;
//     //
//     // and `count()` would scan the keyspace without blocking the server:
//     //
//     //     let mut cursor = 0u64;
//     //     let mut total = 0usize;
//     //     loop {
//     //         let (next, batch): (u64, Vec<String>) = redis::cmd("SCAN")
//     //             .arg(cursor).arg("MATCH").arg("player:*").arg("COUNT").arg(1000)
//     //             .query(&mut self.conn)?;
//     //         total += batch.len();
//     //         if next == 0 { break; }
//     //         cursor = next;
//     //     }
//
// ---------------------------------------------------------------------------
// (3) main() — change exactly one line:
//
//         // before (offline demo):
//         let store = Arc::new(RedisStore::new(FakeRedis::new()));
//         // after (real cluster):
//         let store = Arc::new(RedisStore::new(
//             RealRedisConn::connect("redis://127.0.0.1:6379/")?,
//         ));
//
//     Nothing else changes. `RedisStore<C>` is generic, so the engine wiring,
//     the submit loop, `flush_now`, `shutdown`, and the retry/requeue guarantees
//     are all identical between the fake and the real backend.
//
// ---------------------------------------------------------------------------
// SERVER-SIDE LAST-WRITE-WINS (optional hardening): if writers can race and you
// must reject an out-of-order older version at the server, replace the plain SET
// with a Lua compare-and-set keyed on the stored version header:
//
//     local cur = redis.call('GET', KEYS[1])
//     if cur == false or struct_unpack_version(cur) <= tonumber(ARGV[1]) then
//         redis.call('SET', KEYS[1], ARGV[2])
//     end
//
// evaluated via `redis::Script::new(LUA).key(key).arg(version).arg(value)`.
// FluxThroughput's worker already flushes monotonically per id, so this is only
// needed if multiple independent writers target the same Redis without going
// through one engine.
// ===========================================================================
