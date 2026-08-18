//! Load-time spike: measure the cost of GGUF open + mmap-touch with
//! optional parallel-pread warmer, plus optional MADV_WILLNEED, from a
//! honest cold-cache starting state — without touching global cache.
//!
//! Cold-start protocol (per Codex 019f8ad0):
//!
//!   * `--invalidate` uses `msync(MS_SYNC | MS_INVALIDATE)` on a
//!     disposable `MAP_SHARED` mapping to evict this file's pages from
//!     the unified buffer cache. Only affects the target GGUF, not
//!     unrelated in-flight work. Runs BEFORE the arm timer starts.
//!   * `mincore(2)` residency is reported before and after
//!     invalidation. If residency isn't near zero after invalidation,
//!     the arm should be discarded (contention or a live mapping).
//!   * Per-process `proc_pid_rusage(RUSAGE_INFO_V2)` snapshots taken
//!     around each phase; deltas report actual physical disk reads and
//!     page-ins (unlike `pread`-returned bytes, which count cache hits
//!     as reads).
//!
//! Usage:
//!
//!   cargo run --release -p qwen-llm --example load_spike -- \
//!     /path/to/model.gguf [--invalidate] [--prefetch] \
//!     [--madvise-willneed] [--touch-all-pages] \
//!     [--workers 4] [--chunk-mib 16]
//!
//! Falsifier arms:
//!
//!   D  baseline               = --invalidate --touch-all-pages
//!   C  madvise(WILLNEED)      = --invalidate --madvise-willneed --touch-all-pages
//!   A  parallel-pread warmup  = --invalidate --prefetch --touch-all-pages
//!   AC both                   = --invalidate --prefetch --madvise-willneed --touch-all-pages

use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use qwen_llm::cache_probe::{invalidate_file_cache, probe_file_residency};
use qwen_llm::gguf::GgufFile;
use qwen_llm::pid_metrics::{PidDelta, PidSnapshot};
use qwen_llm::prefetch;

#[path = "support/diag_subscriber.rs"]
mod diag_subscriber;
use diag_subscriber::install_example_diag_subscriber;

#[derive(Debug, Default)]
struct Args {
    model: Option<PathBuf>,
    invalidate: bool,
    prefetch: bool,
    madvise_willneed: bool,
    touch_all_pages: bool,
    workers: Option<usize>,
    chunk_mib: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--invalidate" => args.invalidate = true,
            "--prefetch" => args.prefetch = true,
            "--madvise-willneed" => args.madvise_willneed = true,
            "--touch-all-pages" => args.touch_all_pages = true,
            "--workers" => {
                let v = it.next().ok_or("--workers requires a value")?;
                args.workers = Some(v.parse().map_err(|e| format!("--workers: {e}"))?);
            }
            "--chunk-mib" => {
                let v = it.next().ok_or("--chunk-mib requires a value")?;
                args.chunk_mib = Some(v.parse().map_err(|e| format!("--chunk-mib: {e}"))?);
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown flag: {other}"));
            }
            other => {
                if args.model.is_some() {
                    return Err(format!("unexpected positional argument: {other}"));
                }
                args.model = Some(PathBuf::from(other));
            }
        }
    }
    if args.model.is_none() {
        return Err("missing model path".into());
    }
    Ok(args)
}

fn print_usage(bin: &str) {
    eprintln!(
        "usage: {bin} <model.gguf> [--invalidate] [--prefetch] \
         [--madvise-willneed] [--touch-all-pages] [--workers N] [--chunk-mib N]"
    );
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn fmt_wall(d: Duration) -> String {
    format!("{:>7.3} s", d.as_secs_f64())
}

fn print_delta(label: &str, wall: Duration, delta: PidDelta) {
    println!(
        "{label:<14} {}  pageins={:>7}  diskR={:>7.2} GiB  ΔRSS={:>+7.2} GiB  Δfootprint={:>+7.2} GiB",
        fmt_wall(wall),
        delta.pageins,
        gib(delta.diskio_bytesread),
        delta.resident_size_delta as f64 / (1u64 << 30) as f64,
        delta.phys_footprint_delta as f64 / (1u64 << 30) as f64,
    );
}

fn arm_label(args: &Args) -> &'static str {
    match (args.prefetch, args.madvise_willneed) {
        (true, true) => "AC (prefetch+madvise)",
        (true, false) => "A  (prefetch)",
        (false, true) => "C  (madvise)",
        (false, false) => "D  (baseline)",
    }
}

fn main() {
    install_example_diag_subscriber();
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            print_usage(
                &std::env::args()
                    .next()
                    .unwrap_or_else(|| "load_spike".into()),
            );
            std::process::exit(2);
        }
    };

    let model = args.model.clone().expect("verified above");
    let workers = args.workers.unwrap_or(prefetch::DEFAULT_WORKERS);
    let chunk = args
        .chunk_mib
        .map(|m| m * 1024 * 1024)
        .unwrap_or(prefetch::DEFAULT_CHUNK_BYTES);

    let file_size = match std::fs::metadata(&model) {
        Ok(m) => m.len(),
        Err(e) => {
            eprintln!("error: metadata({}): {e}", model.display());
            std::process::exit(1);
        }
    };

    println!("model:     {}", model.display());
    println!("size:      {:.2} GiB ({} bytes)", gib(file_size), file_size);
    println!("arm:       {}", arm_label(&args));
    println!("touch:     {}", args.touch_all_pages);
    if args.prefetch {
        println!("workers:   {workers}");
        println!("chunk:     {} MiB", chunk / (1024 * 1024));
    }
    println!();
    let _ = std::io::stdout().flush();

    // ---- Untimed reset: verify starting residency, optionally invalidate ----
    let start_resident = match probe_file_residency(&model) {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("warn: probe_file_residency failed: {e}");
            None
        }
    };
    if let Some(r) = start_resident {
        println!(
            "residency (pre-arm): {} / {} pages ({:.1}%)",
            r.resident_pages,
            r.total_pages,
            r.resident_fraction() * 100.0,
        );
    }
    if args.invalidate {
        match invalidate_file_cache(&model) {
            Ok(rep) => {
                println!(
                    "invalidate:   pre={:>6}/{:>6}  post={:>6}/{:>6}",
                    rep.before.resident_pages,
                    rep.before.total_pages,
                    rep.after.resident_pages,
                    rep.after.total_pages,
                );
                if rep.after.resident_fraction() > 0.05 {
                    eprintln!(
                        "warn: post-invalidate residency is {:.1}% \
                         (>5%); another process may hold a mapping. \
                         Consider this sample warm-contaminated.",
                        rep.after.resident_fraction() * 100.0,
                    );
                }
            }
            Err(e) => eprintln!("warn: invalidate_file_cache failed: {e}"),
        }
    }
    println!();

    let t_total = Instant::now();
    let pid_zero = PidSnapshot::now().expect("proc_pid_rusage snapshot 0");

    // ---- Phase 1: optional parallel-pread warmup ----
    if args.prefetch {
        let pid_a = PidSnapshot::now().expect("pid snapshot pre-prefetch");
        let t = Instant::now();
        let report = match prefetch::prefetch_file(&model, workers, chunk) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: prefetch: {e}");
                std::process::exit(1);
            }
        };
        let wall = t.elapsed();
        let pid_b = PidSnapshot::now().expect("pid snapshot post-prefetch");
        let delta = PidDelta::between(pid_a, pid_b);
        println!(
            "prefetch:    {}  bytes-returned={:.2} GiB @ {:.2} GB/s ({} workers x {} MiB)",
            fmt_wall(wall),
            gib(report.bytes),
            report.effective_bytes_per_sec() / 1e9,
            report.workers,
            report.chunk_bytes / (1024 * 1024),
        );
        print_delta("  (rusage)", Duration::ZERO, delta);
    }

    // ---- Phase 2: GgufFile::open (mmaps + validates + parses) ----
    let pid_a = PidSnapshot::now().expect("pid snapshot pre-open");
    let t = Instant::now();
    let gguf = match GgufFile::open(&model) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: GgufFile::open: {e}");
            std::process::exit(1);
        }
    };
    let open_wall = t.elapsed();
    let pid_b = PidSnapshot::now().expect("pid snapshot post-open");
    let mapped = gguf.total_mapped_len() as u64;
    println!(
        "gguf open:   {}  {} shards, {:.2} GiB mapped",
        fmt_wall(open_wall),
        gguf.shard_count(),
        gib(mapped),
    );
    print_delta(
        "  (rusage)",
        Duration::ZERO,
        PidDelta::between(pid_a, pid_b),
    );

    // ---- Phase 3: optional MADV_WILLNEED after open ----
    if args.madvise_willneed {
        let pid_a = PidSnapshot::now().expect("pid snapshot pre-madvise");
        let t = Instant::now();
        for shard in &gguf.shards {
            if let Err(e) = shard.advise(memmap2::Advice::WillNeed) {
                eprintln!("warn: madvise(WILLNEED) on shard failed: {e}");
            }
        }
        let wall = t.elapsed();
        let pid_b = PidSnapshot::now().expect("pid snapshot post-madvise");
        println!("madvise:     {}", fmt_wall(wall));
        print_delta(
            "  (rusage)",
            Duration::ZERO,
            PidDelta::between(pid_a, pid_b),
        );
    }

    // ---- Phase 4: touch every page in each shard ----
    if args.touch_all_pages {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let page = if page == 0 { 4096 } else { page };
        let pid_a = PidSnapshot::now().expect("pid snapshot pre-touch");
        let t = Instant::now();
        let mut checksum: u64 = 0;
        let mut touched_pages: u64 = 0;
        let mut touched_bytes: u64 = 0;
        for shard in &gguf.shards {
            let bytes: &[u8] = shard.mmap_bytes();
            let mut off = 0;
            while off < bytes.len() {
                let byte = unsafe { std::ptr::read_volatile(bytes.as_ptr().add(off)) };
                checksum = checksum.rotate_left(5) ^ u64::from(byte);
                touched_pages += 1;
                touched_bytes += page.min(bytes.len() - off) as u64;
                off += page;
            }
        }
        let touch_wall = t.elapsed();
        let pid_b = PidSnapshot::now().expect("pid snapshot post-touch");
        println!(
            "touch pages: {}  {:.2} GiB, {} pages @ {} B, {:.2} GB/s, chk=0x{:016x}",
            fmt_wall(touch_wall),
            gib(touched_bytes),
            touched_pages,
            page,
            touched_bytes as f64 / touch_wall.as_secs_f64() / 1e9,
            checksum,
        );
        print_delta(
            "  (rusage)",
            Duration::ZERO,
            PidDelta::between(pid_a, pid_b),
        );
    }

    let total_wall = t_total.elapsed();
    let pid_end = PidSnapshot::now().expect("pid snapshot end");
    let end_resident = probe_file_residency(&model).ok();
    println!();
    println!("TOTAL:       {}", fmt_wall(total_wall));
    print_delta(
        "  (rusage)",
        Duration::ZERO,
        PidDelta::between(pid_zero, pid_end),
    );
    if let (Some(pre), Some(post)) = (start_resident, end_resident) {
        println!(
            "residency: pre={}/{}  post={}/{} ({:.1}% -> {:.1}%)",
            pre.resident_pages,
            pre.total_pages,
            post.resident_pages,
            post.total_pages,
            pre.resident_fraction() * 100.0,
            post.resident_fraction() * 100.0,
        );
    }
}
