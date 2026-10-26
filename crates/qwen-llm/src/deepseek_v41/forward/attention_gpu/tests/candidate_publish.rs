use super::*;
use metal::evaluate_metal_memory_admission_with_cpu_bytes;
use objc2::rc::{Weak, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use serde_json::json;
use sha2::{Digest, Sha256};

mod real_pipeline;
mod valid_ids;

const DOMAINS: [usize; 9] = [0, 1, 255, 256, 257, 2048, 2049, 16384, 131072];
const COUNTS: [usize; 4] = [0, 1, 256, 2048];
const PATTERNS: [&str; 4] = ["mixed", "boundary", "duplicates", "invalid"];
const GUARD: u32 = 0x91ab_cdef;
const POISON: u32 = 0xa5a5_5a5a;
const TAIL: usize = 17;
const THREADS: usize = 256;
const BATCH: usize = 4;
const ROUNDS: usize = 3;
const CPU_RESERVE: u64 = 1 << 20;
const LIFETIME_BOOKKEEPING_BYTES: u64 = 4096;
const LIMIT: u64 = 64 << 20;
const PROTOCOL: &str = "v41-candidate-publish-v1";

// A sorted-set lookup is independent of both the incumbent's exhaustive scan
// and the candidate's clear/scatter construction.
fn reference(domain: usize, input: &[u32]) -> Vec<u32> {
    let mut set: Vec<u32> = input
        .iter()
        .copied()
        .filter(|&id| (id as usize) < domain)
        .collect();
    set.sort_unstable();
    set.dedup();
    (0..domain)
        .map(|word| u32::from(set.binary_search(&(word as u32)).is_ok()))
        .collect()
}

fn input_ids(domain: usize, count: usize, pattern: &str) -> Vec<u32> {
    let boundary = [
        0,
        domain.saturating_sub(1) as u32,
        domain as u32,
        domain as u32 + 1,
        u32::MAX,
        255,
        256,
        257,
    ];
    (0..count)
        .map(|i| match pattern {
            "mixed" if i % 3 == 0 => boundary[(i / 3) % boundary.len()],
            "mixed" => ((i * 7919 + 17) % domain.max(1)) as u32,
            "boundary" => boundary[i % boundary.len()],
            "duplicates" => domain.saturating_sub(1) as u32,
            "invalid" if i % 2 == 0 => u32::MAX,
            "invalid" => domain as u32,
            _ => unreachable!("fixture pattern"),
        })
        .collect()
}

fn matches(actual: &[u32], expected: &[u32]) -> bool {
    actual == expected && actual.iter().all(|&word| word <= 1)
}

// Three shared buffers: IDs at 128 bytes, mask at 64, observer at 192.
// Even empty logical ranges have a sentinel word and both redzones.
fn buffer_words(domain: usize, count: usize) -> [usize; 3] {
    [
        32 + count.max(1) + TAIL,
        16 + domain.max(1) + TAIL,
        48 + domain.max(1) + TAIL,
    ]
}

fn cpu_price(domain: usize, count: usize) -> u64 {
    // Eight full inventory copies cover upload staging, ID permutations,
    // sorted oracle/set, expected mask, incumbent snapshot, and guard reads.
    8 * buffer_words(domain, count).iter().sum::<usize>() as u64 * 4
        + CPU_RESERVE
        + LIFETIME_BOOKKEEPING_BYTES
}

#[test]
fn candidate_publish_cpu_reference_contract() {
    assert!(reference(0, &[0, u32::MAX]).is_empty());
    assert_eq!(reference(1, &[0, 0, 1, u32::MAX]), [1]);
    assert_eq!(reference(5, &[4, 1, 1, 5, u32::MAX]), [0, 1, 0, 0, 1]);
    assert_eq!(reference(3, &[]), [0, 0, 0]);
    let config = Config::default();
    assert_eq!((config.candidate_block, config.candidate_topk), (8, 2048));
    let domain = 1_048_576usize.div_ceil(config.candidate_block);
    assert_eq!(domain, 131072);
    assert_eq!(domain * config.candidate_topk, 268435456);
    assert_eq!(config.candidate_block * config.candidate_topk, 16384);
    let first_pruned_keys = config.candidate_block * config.candidate_topk + 1;
    assert_eq!(first_pruned_keys, 16385);
    let first_pruned_domain = first_pruned_keys.div_ceil(config.candidate_block);
    assert_eq!(first_pruned_domain, 2049);
    assert!(DOMAINS.contains(&first_pruned_domain));
    assert_eq!(first_pruned_domain * config.candidate_topk, 4196352);
}

#[test]
fn candidate_publish_cpu_grid_and_controls() {
    let mut cases = 0;
    for domain in DOMAINS {
        for count in COUNTS {
            for pattern in PATTERNS {
                let input = input_ids(domain, count, pattern);
                assert_eq!(input.len(), count);
                let expected = reference(domain, &input);
                assert!(matches(&expected, &expected));
                if count == 0 || pattern == "invalid" {
                    assert!(expected.iter().all(|&word| word == 0));
                }
                if pattern == "duplicates" && domain > 0 && count > 0 {
                    assert_eq!(expected.iter().sum::<u32>(), 1);
                    assert_eq!(expected[domain - 1], 1);
                }
                let mut permuted = input.clone();
                permuted.reverse();
                permuted.rotate_left(count / 3);
                assert_eq!(reference(domain, &permuted), expected);
                if domain > 0 {
                    assert!(!matches(&vec![POISON; domain], &expected), "no-op");
                    let mut wrong = expected.clone();
                    wrong[domain - 1] ^= 1;
                    assert!(!matches(&wrong, &expected), "wrong-word");
                    wrong[domain - 1] = 2;
                    assert!(!matches(&wrong, &expected), "not a 0/1 mask");
                }
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 144);
    // A semantically wrong ID permutation must not be confused with merely
    // reordering the ID list (which is explicitly legal).
    assert_ne!(reference(257, &[0, 255]), reference(257, &[1, 256]));
}

#[test]
fn candidate_publish_cpu_budget_contract() {
    for domain in DOMAINS {
        for count in COUNTS {
            let raw = buffer_words(domain, count).iter().sum::<usize>() as u64 * 4;
            // CPU-only bound allows 64 KiB rounding overhead per buffer;
            // actual Metal pricing is queried and admitted on the worker.
            assert!(raw + 3 * 65536 + cpu_price(domain, count) < LIMIT);
        }
    }
    assert_eq!(buffer_words(0, 0), [50, 34, 66]);
    assert_eq!(buffer_words(131072, 2048), [2097, 131105, 131137]);
    assert_eq!(cpu_price(131072, 2048), 9511520);
    assert!(std::mem::size_of::<BufferProbes>() as u64 <= LIFETIME_BOOKKEEPING_BYTES);
}

type BufferProbes = [Weak<ProtocolObject<dyn MTLBuffer>>; 3];

fn drained_case_pool<G, P>(
    transaction: G,
    work: impl FnOnce(&G) -> P,
    verify_release: impl FnOnce(P),
) {
    let probes = autoreleasepool(|_| work(&transaction));
    verify_release(probes);
    // Admission stays serialized through Rust destruction AND pool drain.
    drop(transaction);
}

#[test]
fn candidate_publish_cpu_case_pool_lifetime_controls() {
    use objc2::rc::Retained;
    use objc2::runtime::NSObject;
    use std::cell::Cell;

    struct Guard<'a>(&'a Cell<bool>);
    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            self.0.set(false);
        }
    }

    let held = Cell::new(true);
    drained_case_pool(
        Guard(&held),
        |_| {
            assert!(held.get());
            let object = NSObject::new();
            let weak = Weak::from_retained(&object);
            let _ = Retained::autorelease_ptr(object);
            assert!(weak.load().is_some(), "no-drain control must still be live");
            weak
        },
        |weak| {
            assert!(held.get(), "guard must outlive the drain/release check");
            assert!(weak.load().is_none(), "case autoreleases must be drained");
        },
    );
    assert!(!held.get());

    drained_case_pool(
        (),
        |_| {
            let object = NSObject::new();
            let weak = Weak::from_retained(&object);
            let _ = Retained::autorelease_ptr(object.clone());
            (weak, object)
        },
        |(weak, retained)| {
            assert!(
                weak.load().is_some(),
                "escaped-retain control must be detected"
            );
            drop(retained);
            assert!(weak.load().is_none());
        },
    );
}

#[test]
fn candidate_publish_cpu_nested_sample_pool_contract() {
    use objc2::rc::Retained;
    use objc2::runtime::NSObject;

    for _ in 0..2 {
        drained_case_pool(
            (),
            |_| {
                let fixture = NSObject::new();
                let fixture_weak = Weak::from_retained(&fixture);
                let _ = Retained::autorelease_ptr(fixture.clone());
                for _ in 0..ROUNDS * 8 {
                    let sample_weak = autoreleasepool(|_| {
                        let command = NSObject::new();
                        let weak = Weak::from_retained(&command);
                        let _ = Retained::autorelease_ptr(command);
                        assert!(weak.load().is_some());
                        assert!(fixture_weak.load().is_some());
                        weak
                    });
                    assert!(sample_weak.load().is_none(), "no sample accumulation");
                    assert!(
                        fixture_weak.load().is_some(),
                        "one fixture spans all samples"
                    );
                }
                drop(fixture);
                assert!(
                    fixture_weak.load().is_some(),
                    "case pool has not drained yet"
                );
                fixture_weak
            },
            |weak| assert!(weak.load().is_none(), "no case accumulation"),
        );
    }
}

fn forbidden_native_variable(name: &str) -> bool {
    name.starts_with("MTL_DEBUG")
        || name.starts_with("MTL_SHADER_VALIDATION")
        || matches!(
            name,
            "METAL_DEVICE_WRAPPER_TYPE"
                | "MTL_DEVICE_WRAPPER_TYPE"
                | "MTL_CAPTURE_ENABLED"
                | "MTL_HUD_ENABLED"
                | "DYLD_INSERT_LIBRARIES"
                | "DYLD_LIBRARY_PATH"
                | "QWEN_REQUIRE_METAL_TESTS"
        )
}

#[test]
fn candidate_publish_cpu_native_launch_policy() {
    for name in [
        "MTL_DEBUG_LAYER",
        "MTL_DEBUG_LAYER_VALIDATE_UNRETAINED_RESOURCES",
        "MTL_SHADER_VALIDATION",
        "MTL_SHADER_VALIDATION_ENABLE_PIPELINES",
        "METAL_DEVICE_WRAPPER_TYPE",
        "MTL_CAPTURE_ENABLED",
        "DYLD_INSERT_LIBRARIES",
        "QWEN_REQUIRE_METAL_TESTS",
    ] {
        assert!(forbidden_native_variable(name));
    }
    assert!(!forbidden_native_variable("QWEN_METAL_LEASE_WAIT"));
    assert!(!forbidden_native_variable("PATH"));
}

#[derive(Clone, Copy, Debug)]
enum Arm {
    Incumbent,
    ClearScatter,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Incumbent => "incumbent",
            Self::ClearScatter => "clear_scatter",
        }
    }

    fn dispatches(self) -> usize {
        match self {
            Self::Incumbent => 1,
            Self::ClearScatter => 2,
        }
    }
}

struct Pipelines {
    keep: metal::Pipeline,
    clear: metal::Pipeline,
    scatter: metal::Pipeline,
    observe: metal::Pipeline,
}

impl Pipelines {
    fn new(ctx: &MetalContext) -> Self {
        let get = |suffix| {
            let p = ctx
                .pipeline(&format!("kernel_deepseek_v41_attention_{suffix}"))
                .unwrap();
            assert!(p.maxTotalThreadsPerThreadgroup() >= THREADS);
            p
        };
        Self {
            keep: get("keep"),
            clear: get("keep_clear"),
            scatter: get("keep_scatter"),
            observe: get("keep_observe"),
        }
    }
}

struct Guarded {
    backing: MetalTensor,
    view: MetalTensor,
    prefix: usize,
    logical: usize,
    sentinel: u32,
}

impl Guarded {
    fn new(ctx: &MetalContext, prefix: usize, logical: usize, sentinel: u32) -> Self {
        let words = prefix + logical.max(1) + TAIL;
        let data = vec![sentinel; words];
        let backing = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&data),
            vec![words as u64],
            GgmlType::I32,
        )
        .unwrap();
        let view = backing.view_subrange(prefix as u64, vec![logical.max(1) as u64]);
        assert_eq!(view.offset, (prefix * 4) as u64);
        Self {
            backing,
            view,
            prefix,
            logical,
            sentinel,
        }
    }

    fn write(&self, words: &[u32]) {
        assert_eq!(words.len(), self.logical);
        // All callers have waited for the prior command; no CPU/GPU overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                words.as_ptr(),
                self.view
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(self.view.offset as usize)
                    .cast::<u32>(),
                words.len(),
            );
        }
    }

    fn read(&self) -> Vec<u32> {
        ids(&self.view, self.logical)
            .into_iter()
            .map(|x| x as u32)
            .collect()
    }

    fn check_guards(&self) {
        let words = ids(&self.backing, self.backing.n_elements() as usize);
        assert!(
            words[..self.prefix]
                .iter()
                .all(|&v| v == self.sentinel as usize)
        );
        assert!(
            words[self.prefix + self.logical..]
                .iter()
                .all(|&v| v == self.sentinel as usize)
        );
    }
}

struct Fixture {
    domain: usize,
    count: usize,
    input: Vec<u32>,
    selected: Guarded,
    mask: Guarded,
    observed: Guarded,
}

impl Fixture {
    fn new(
        ctx: &MetalContext,
        domain: usize,
        count: usize,
        pattern: &str,
        _transaction: &metal::MetalAllocationTransactionGuard,
    ) -> Self {
        assert!(DOMAINS.contains(&domain) && COUNTS.contains(&count));
        let sizes = buffer_words(domain, count).map(|n| n as u64 * 4);
        let priced: u64 = sizes
            .iter()
            .map(|&n| ctx.price_shared_buffer_upper(n).unwrap().priced_upper_bytes)
            .sum();
        let cpu = cpu_price(domain, count);
        assert!(priced + cpu < LIMIT);
        let admission = evaluate_metal_memory_admission_with_cpu_bytes(
            priced,
            cpu,
            0,
            ctx.memory_signals(),
            true,
        );
        assert!(admission.is_admitted(), "fixture refused: {admission:?}");
        eprintln!(
            "{}",
            json!({
                "protocol": PROTOCOL, "event": "admission", "domain_words": domain,
                "kept_id_count": count, "pattern": pattern, "buffer_bytes": sizes,
                "priced_metal_bytes": priced, "cpu_upper_bytes": cpu, "limit_bytes": LIMIT,
            })
        );
        let input = input_ids(domain, count, pattern);
        // Valid zero redzones detect a too-large ID count, not just writes.
        let selected = Guarded::new(ctx, 32, count, 0);
        selected.write(&input);
        Self {
            domain,
            count,
            input,
            selected,
            mask: Guarded::new(ctx, 16, domain, GUARD),
            observed: Guarded::new(ctx, 48, domain, GUARD),
        }
    }

    fn reset(&self) {
        self.mask.write(&vec![POISON; self.domain]);
        self.observed.write(&vec![POISON; self.domain]);
    }

    fn launch(enc: &KernelEncoder, pipeline: &metal::Pipeline, n: usize) {
        enc.set_pipeline(pipeline);
        // Always launch, even for empty work, so the fixture compares exactly
        // one vs two dispatches and exercises padded bounds checks.
        enc.dispatch(
            MTLSize {
                width: n.max(1).div_ceil(THREADS),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode(&self, enc: &KernelEncoder, p: &Pipelines, arm: Arm, observe: bool) {
        assert!(
            !enc.is_concurrent(),
            "clear/scatter/consumer requires serial encoder"
        );
        enc.set_bytes(0, &[self.domain as u32, self.count as u32]);
        enc.set_tensor(1, &self.selected.view);
        enc.set_tensor(2, &self.mask.view);
        match arm {
            Arm::Incumbent => Self::launch(enc, &p.keep, self.domain),
            Arm::ClearScatter => {
                Self::launch(enc, &p.clear, self.domain);
                Self::launch(enc, &p.scatter, self.count);
            }
        }
        if observe {
            self.encode_observer(enc, p);
        }
    }

    fn encode_observer(&self, enc: &KernelEncoder, p: &Pipelines) {
        assert!(!enc.is_concurrent());
        enc.set_bytes(0, &[self.domain as u32, self.count as u32]);
        enc.set_tensor(2, &self.mask.view);
        enc.set_tensor(3, &self.observed.view);
        Self::launch(enc, &p.observe, self.domain);
    }

    fn check(&self, expected: &[u32]) -> Vec<u32> {
        let got = self.mask.read();
        assert!(
            matches(&got, expected),
            "mask domain={} count={}",
            self.domain,
            self.count
        );
        assert!(
            matches(&self.observed.read(), expected),
            "next-dispatch visibility"
        );
        assert_eq!(self.selected.read(), self.input);
        self.selected.check_guards();
        self.mask.check_guards();
        self.observed.check_guards();
        got
    }

    fn correctness(&self, ctx: &MetalContext, p: &Pipelines) {
        let expected = reference(self.domain, &self.input);
        if self.domain > 0 {
            self.reset();
            execute(ctx, |enc| self.encode_observer(enc, p));
            assert!(!matches(&self.mask.read(), &expected), "no-op control");
            assert!(
                !matches(&self.observed.read(), &expected),
                "no-op consumer control"
            );
            let mut wrong = expected.clone();
            wrong[self.domain - 1] ^= 1;
            self.mask.write(&wrong);
            execute(ctx, |enc| self.encode_observer(enc, p));
            assert!(!matches(&self.mask.read(), &expected), "wrong-word control");
            assert!(
                !matches(&self.observed.read(), &expected),
                "wrong-word consumer control"
            );
        }
        self.reset();
        execute(ctx, |enc| self.encode(enc, p, Arm::Incumbent, true));
        let incumbent = self.check(&expected);
        self.reset();
        execute(ctx, |enc| self.encode(enc, p, Arm::ClearScatter, true));
        assert_eq!(self.check(&expected), incumbent);
        let mut permuted = self.input.clone();
        permuted.reverse();
        permuted.rotate_left(self.count / 3);
        self.selected.write(&permuted);
        self.reset();
        execute(ctx, |enc| self.encode(enc, p, Arm::ClearScatter, true));
        assert_eq!(self.mask.read(), expected);
        assert_eq!(self.observed.read(), expected);
        assert_eq!(self.selected.read(), permuted);
        self.selected.check_guards();
        self.mask.check_guards();
        self.observed.check_guards();
        self.selected.write(&self.input);
    }
}

fn with_fixture(
    ctx: &MetalContext,
    domain: usize,
    count: usize,
    pattern: &str,
    phase: &str,
    work: impl FnOnce(&Fixture),
) {
    drained_case_pool(
        ctx.begin_allocation_transaction(),
        |transaction| {
            let f = Fixture::new(ctx, domain, count, pattern, transaction);
            let probes: BufferProbes = [
                Weak::from_retained(&f.selected.backing.buffer),
                Weak::from_retained(&f.mask.backing.buffer),
                Weak::from_retained(&f.observed.backing.buffer),
            ];
            assert!(probes.iter().all(|weak| weak.load().is_some()));
            work(&f);
            drop(f);
            probes
        },
        |probes| {
            for (name, weak) in ["ids", "mask", "observer"].into_iter().zip(probes) {
                assert!(
                    weak.load().is_none(),
                    "{phase} {name} buffer survived case drain"
                );
            }
            eprintln!(
                "{}",
                json!({
                    "protocol": PROTOCOL, "event": "case_release", "phase": phase,
                    "domain_words": domain, "kept_id_count": count, "pattern": pattern,
                    "buffers": ["ids", "mask", "observer"], "all_released": true,
                })
            );
        },
    );
}

fn context() -> MetalContext {
    let ctx = MetalContext::new().expect("explicit fixture requires Metal, never skip");
    assert!(
        ctx.device.name().to_string().contains("M5"),
        "parent-owned M5 fixture only"
    );
    assert!(!metal::dispatch_census_is_active());
    eprintln!(
        "{}",
        json!({
            "protocol": PROTOCOL, "event": "context", "device": ctx.describe(),
            "msl_sha256": format!("{:x}", Sha256::digest(include_bytes!("../../../../../../../kernels/deepseek_v41_attention.metal"))),
            "fixture_sha256": format!("{:x}", Sha256::digest(include_bytes!("candidate_publish.rs"))),
            "candidate_block": Config::default().candidate_block,
            "candidate_topk": Config::default().candidate_topk,
            "domains": DOMAINS, "counts": COUNTS, "patterns": PATTERNS,
            "dense_mask_retained": true, "production_dispatch_changed": false,
        })
    );
    ctx
}

fn correctness_grid(ctx: &MetalContext, p: &Pipelines) {
    let mut cases = 0;
    for domain in DOMAINS {
        for count in COUNTS {
            for pattern in PATTERNS {
                with_fixture(ctx, domain, count, pattern, "correctness", |f| {
                    f.correctness(ctx, p);
                });
                cases += 1;
                eprintln!(
                    "{}",
                    json!({
                        "protocol": PROTOCOL, "event": "correctness_case", "domain_words": domain,
                        "kept_id_count": count, "pattern": pattern, "permuted_candidate": true,
                        "independent_cpu": true, "incumbent": true, "next_dispatch_consumer": true,
                        "negative_controls": if domain > 0 { "no-op and wrong-word through GPU observer" } else { "empty range guards only" },
                    })
                );
            }
        }
    }
    assert_eq!(cases, 144);
    eprintln!(
        "{}",
        json!({"protocol": PROTOCOL, "event": "correctness_complete", "cases": cases})
    );
}

#[test]
#[ignore = "parent-only self-leased M5 correctness; launch with MTL_DEBUG_LAYER=1; <64 MiB"]
fn candidate_publish_m5_correctness() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("production lease and wired guard");
    let ctx = context();
    let p = Pipelines::new(&ctx);
    correctness_grid(&ctx, &p);
}

#[test]
#[ignore = "parent-only self-leased M5 native fixture timing; validation flags absent at launch"]
fn candidate_publish_m5_native_timing() {
    for (name, _) in std::env::vars_os() {
        assert!(
            !forbidden_native_variable(&name.to_string_lossy()),
            "native launch must omit {name:?}"
        );
    }
    let _lease = metal::acquire_metal_benchmark_lease().expect("production lease and wired guard");
    let ctx = context();
    let p = Pipelines::new(&ctx);
    // Correctness is a distinct, completed phase before the first timed sample.
    correctness_grid(&ctx, &p);
    eprintln!(
        "{}",
        json!({
            "protocol": PROTOCOL, "event": "timing_begin", "cases": DOMAINS.len() * COUNTS.len(),
            "rounds": ROUNDS, "orders": ["ABBA", "BAAB"], "publications_per_sample": BATCH,
            "samples_per_case": ROUNDS * 8, "total_samples": DOMAINS.len() * COUNTS.len() * ROUNDS * 8,
            "scope": "resident synthetic mask publication only; NOT full model or real route",
            "includes": "command creation + serial encoding + all publication dispatches + commit + wait",
            "excludes": "allocation, compilation, CPU oracle/readback, observer, model, selection",
            "warmup_samples": 0, "retries": 0, "spread_or_gain_gate": false,
            "known_validation_flags_absent": true,
            "sample_pool_setup_and_drain_in_cpu_wall": false,
        })
    );
    use Arm::{ClearScatter as B, Incumbent as A};
    let mut sample = 0;
    for domain in DOMAINS {
        for count in COUNTS {
            with_fixture(&ctx, domain, count, "mixed", "timing", |f| {
                let expected = reference(domain, &f.input);
                for round in 0..ROUNDS {
                    for (order, arms) in [("ABBA", [A, B, B, A]), ("BAAB", [B, A, A, B])] {
                        for (slot, arm) in arms.into_iter().enumerate() {
                            autoreleasepool(|_| {
                                f.reset();
                                let wall = std::time::Instant::now();
                                let command = ctx.queue.commandBuffer().unwrap();
                                let enc = KernelEncoder::begin(&command);
                                for _ in 0..BATCH {
                                    f.encode(&enc, &p, arm, false);
                                }
                                enc.end();
                                metal::commit_and_wait(&command).unwrap();
                                let wall_ns = wall.elapsed().as_nanos() as u64;
                                let start = command.GPUStartTime();
                                let end = command.GPUEndTime();
                                let valid = start.is_finite()
                                    && end.is_finite()
                                    && start > 0.0
                                    && end > start;
                                // Emit even an invalid measurement before refusing; no retry/filter.
                                eprintln!(
                                    "{}",
                                    json!({
                                        "protocol": PROTOCOL, "event": "sample", "sample": sample,
                                        "domain_words": domain, "kept_id_count": count, "pattern": "mixed",
                                        "logical_incumbent_comparisons_per_publication": domain * count,
                                        "round": round, "order": order, "slot": slot, "arm": arm.name(),
                                        "publications": BATCH, "dispatches": BATCH * arm.dispatches(),
                                        "cpu_wall_ns": wall_ns, "gpu_start_s": start, "gpu_end_s": end,
                                        "gpu_timestamps_valid": valid,
                                        "gpu_batch_ns": if valid { Some((end - start) * 1e9) } else { None },
                                        "gpu_ns_per_publication": if valid { Some((end - start) * 1e9 / BATCH as f64) } else { None },
                                        "cpu_ns_per_publication": wall_ns as f64 / BATCH as f64,
                                    })
                                );
                                assert!(
                                    valid,
                                    "GPU timestamps unavailable; sample retained, no retry"
                                );
                                assert_eq!(f.mask.read(), expected);
                                assert_eq!(f.selected.read(), f.input);
                                f.selected.check_guards();
                                f.mask.check_guards();
                                f.observed.check_guards();
                                sample += 1;
                            });
                        }
                    }
                }
            });
        }
    }
    assert_eq!(sample, 864);
    eprintln!(
        "{}",
        json!({"protocol": PROTOCOL, "event": "timing_complete", "samples": sample})
    );
}
