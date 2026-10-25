use super::*;

const VALID_PROTOCOL: &str = "v41-candidate-publish-valid-unique-v1";
const VALID_DOMAINS: [usize; 3] = [2049, 16384, 131072];
const RETAINED: usize = 2048;
const PERMUTATION_STRIDE: usize = 1597;
const NAN_POISON: u32 = 0x7fc0_0001;

#[derive(Clone, Copy)]
enum IdOrder {
    Ascending,
    ScorePermuted,
}

const ID_ORDERS: [IdOrder; 2] = [IdOrder::Ascending, IdOrder::ScorePermuted];

impl IdOrder {
    fn name(self) -> &'static str {
        match self {
            Self::Ascending => "ascending",
            Self::ScorePermuted => "score_permuted",
        }
    }
}

struct IdOrders {
    ascending: Vec<u32>,
    score_permuted: Vec<u32>,
}

impl IdOrders {
    fn new(domain: usize) -> Self {
        assert!(VALID_DOMAINS.contains(&domain));
        let ascending: Vec<u32> = (0..RETAINED)
            .map(|i| (i * (domain - 1) / (RETAINED - 1)) as u32)
            .collect();
        // Odd stride permutes 2048 ranks; the forced newest block ranks first.
        // This is a deterministic score-order proxy, not captured model scores.
        let score_permuted = (0..RETAINED)
            .map(|rank| ascending[(RETAINED - 1 + rank * PERMUTATION_STRIDE) % RETAINED])
            .collect();
        Self {
            ascending,
            score_permuted,
        }
    }

    fn get(&self, order: IdOrder) -> &[u32] {
        match order {
            IdOrder::Ascending => &self.ascending,
            IdOrder::ScorePermuted => &self.score_permuted,
        }
    }
}

fn valid_unique_newest(domain: usize, input: &[u32]) -> bool {
    if input.len() != RETAINED || input.iter().any(|&id| id as usize >= domain) {
        return false;
    }
    let mut sorted = input.to_vec();
    sorted.sort_unstable();
    sorted.windows(2).all(|w| w[0] < w[1]) && sorted.last() == Some(&((domain - 1) as u32))
}

fn id_digest(input: &[u32]) -> String {
    let mut hash = Sha256::new();
    for id in input {
        hash.update(id.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[derive(serde::Serialize)]
struct CpuInventory {
    both_id_orders_bytes: u64,
    active_input_and_oracle_ids_bytes: u64,
    six_mask_vectors_bytes: u64,
    two_full_usize_readbacks_bytes: u64,
    upload_staging_bytes: u64,
    runtime_reserve_bytes: u64,
    lifetime_bookkeeping_bytes: u64,
}

impl CpuInventory {
    fn new(domain: usize) -> Self {
        assert!(VALID_DOMAINS.contains(&domain));
        let words = buffer_words(domain, RETAINED).iter().sum::<usize>() as u64;
        Self {
            both_id_orders_bytes: 2 * RETAINED as u64 * 4,
            active_input_and_oracle_ids_bytes: 4 * RETAINED as u64 * 4,
            six_mask_vectors_bytes: 6 * domain as u64 * 4,
            two_full_usize_readbacks_bytes: 2 * words * std::mem::size_of::<usize>() as u64,
            upload_staging_bytes: words * 4,
            runtime_reserve_bytes: 1 << 20,
            lifetime_bookkeeping_bytes: 4096,
        }
    }

    fn total(&self) -> u64 {
        self.both_id_orders_bytes
            + self.active_input_and_oracle_ids_bytes
            + self.six_mask_vectors_bytes
            + self.two_full_usize_readbacks_bytes
            + self.upload_staging_bytes
            + self.runtime_reserve_bytes
            + self.lifetime_bookkeeping_bytes
    }
}

#[test]
fn candidate_publish_valid_ids_cpu_orders_contract() {
    let c = Config::default();
    assert_eq!((c.candidate_block, c.candidate_topk), (8, RETAINED));
    assert_eq!(
        (c.candidate_block * RETAINED + 1).div_ceil(c.candidate_block),
        VALID_DOMAINS[0]
    );
    for domain in VALID_DOMAINS {
        let input = IdOrders::new(domain);
        assert_eq!(input.ascending.len(), RETAINED);
        assert!(input.ascending.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(input.ascending.first(), Some(&0));
        assert_eq!(input.ascending.last(), Some(&((domain - 1) as u32)));
        assert_eq!(input.score_permuted.first(), input.ascending.last());
        assert_ne!(input.ascending, input.score_permuted);
        let mut sorted = input.score_permuted.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, input.ascending);
        let expected = reference(domain, &input.ascending);
        assert_eq!(expected.iter().sum::<u32>() as usize, RETAINED);
        assert_eq!(expected[domain - 1], 1);
        for order in ID_ORDERS {
            let ids = input.get(order);
            assert!(valid_unique_newest(domain, ids));
            assert_eq!(reference(domain, ids), expected);
            assert_eq!(ids, IdOrders::new(domain).get(order));
        }
        assert_ne!(
            id_digest(&input.ascending),
            id_digest(&input.score_permuted)
        );
    }
}

#[test]
fn candidate_publish_valid_ids_cpu_wrong_noop_count_controls() {
    assert!(f32::from_bits(NAN_POISON).is_nan());
    for domain in VALID_DOMAINS {
        let input = IdOrders::new(domain);
        let expected = reference(domain, &input.ascending);
        assert!(!matches(&vec![NAN_POISON; domain], &expected));
        let mut wrong = expected.clone();
        wrong[domain - 1] = 0;
        assert!(!matches(&wrong, &expected));
        wrong[domain - 1] = 2;
        assert!(!matches(&wrong, &expected));
        let short = &input.ascending[..RETAINED - 1];
        assert!(!valid_unique_newest(domain, short));
        assert_ne!(reference(domain, short), expected);
        let mut duplicate = input.ascending.clone();
        duplicate[1] = duplicate[0];
        assert!(!valid_unique_newest(domain, &duplicate));
        let mut invalid = input.ascending.clone();
        invalid[0] = domain as u32;
        assert!(!valid_unique_newest(domain, &invalid));
    }
}

#[test]
fn candidate_publish_valid_ids_cpu_budget_contract() {
    assert_eq!(std::mem::size_of::<usize>(), 8);
    for domain in VALID_DOMAINS {
        let shared: u64 = buffer_words(domain, RETAINED)
            .iter()
            .map(|&n| n as u64 * 4)
            .sum();
        let cpu = CpuInventory::new(domain);
        assert_eq!(cpu.both_id_orders_bytes, 16384);
        assert_eq!(cpu.active_input_and_oracle_ids_bytes, 32768);
        assert!(std::mem::size_of::<BufferProbes>() as u64 <= cpu.lifetime_bookkeeping_bytes);
        assert!(shared + 3 * 65536 + cpu.total() < LIMIT);
    }
    assert_eq!(CpuInventory::new(131072).upload_staging_bytes, 1057356);
    assert_eq!(
        CpuInventory::new(131072).two_full_usize_readbacks_bytes,
        4229424
    );
    assert_eq!(CpuInventory::new(131072).total(), 9534332);
}

#[test]
fn candidate_publish_valid_ids_cpu_protocol_grid_contract() {
    assert_ne!(VALID_PROTOCOL, super::PROTOCOL);
    assert_eq!((BATCH, ROUNDS), (4, 3));
    let mut cases = 0;
    let mut samples = [0; 2];
    for domain in VALID_DOMAINS {
        assert!(domain > RETAINED);
        for _ in ID_ORDERS {
            cases += 1;
            for _ in 0..ROUNDS {
                for order in [[0, 1, 1, 0], [1, 0, 0, 1]] {
                    for arm in order {
                        samples[arm] += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 6);
    assert_eq!(samples, [72, 72]);
}

fn reset(f: &Fixture) {
    f.mask.write(&vec![NAN_POISON; f.domain]);
    f.observed.write(&vec![NAN_POISON; f.domain]);
}

fn with_valid_fixture(
    ctx: &MetalContext,
    domain: usize,
    order: IdOrder,
    phase: &str,
    work: impl FnOnce(&Fixture, &IdOrders),
) {
    drained_case_pool(
        ctx.begin_allocation_transaction(),
        |_transaction| {
            let sizes = buffer_words(domain, RETAINED).map(|n| n as u64 * 4);
            let priced: u64 = sizes
                .iter()
                .map(|&n| ctx.price_shared_buffer_upper(n).unwrap().priced_upper_bytes)
                .sum();
            let cpu = CpuInventory::new(domain);
            assert!(priced + cpu.total() < LIMIT);
            let admission = evaluate_metal_memory_admission_with_cpu_bytes(
                priced,
                cpu.total(),
                0,
                ctx.memory_signals(),
                true,
            );
            assert!(
                admission.is_admitted(),
                "valid-ID fixture refused: {admission:?}"
            );
            eprintln!(
                "{}",
                json!({
                    "protocol": VALID_PROTOCOL, "event": "admission", "phase": phase,
                    "domain_words": domain, "kept_id_count": RETAINED, "id_order": order.name(),
                    "buffer_bytes": sizes, "cpu_inventory": cpu, "cpu_upper_bytes": cpu.total(),
                    "priced_metal_bytes": priced, "limit_bytes": LIMIT,
                })
            );
            let orders = IdOrders::new(domain);
            let input = orders.get(order).to_vec();
            assert!(valid_unique_newest(domain, &input));
            let digest = id_digest(&input);
            let selected = Guarded::new(ctx, 32, RETAINED, 0);
            selected.write(&input);
            let f = Fixture {
                domain,
                count: RETAINED,
                input,
                selected,
                mask: Guarded::new(ctx, 16, domain, GUARD),
                observed: Guarded::new(ctx, 48, domain, GUARD),
            };
            let probes: BufferProbes = [
                Weak::from_retained(&f.selected.backing.buffer),
                Weak::from_retained(&f.mask.backing.buffer),
                Weak::from_retained(&f.observed.backing.buffer),
            ];
            assert!(probes.iter().all(|weak| weak.load().is_some()));
            work(&f, &orders);
            drop(f);
            drop(orders);
            (probes, digest)
        },
        |(probes, digest)| {
            for (name, weak) in ["ids", "mask", "observer"].into_iter().zip(probes) {
                assert!(
                    weak.load().is_none(),
                    "valid-ID {phase} {name} survived case drain"
                );
            }
            eprintln!(
                "{}",
                json!({
                    "protocol": VALID_PROTOCOL, "event": "case_release", "phase": phase,
                    "domain_words": domain, "kept_id_count": RETAINED, "id_order": order.name(),
                    "ids_sha256_le": digest, "buffers": ["ids", "mask", "observer"], "all_released": true,
                })
            );
        },
    );
}

fn context(kind: &str) -> MetalContext {
    let ctx = MetalContext::new().expect("explicit valid-ID fixture requires Metal");
    assert!(
        ctx.device.name().to_string().contains("M5"),
        "parent-only M5 fixture"
    );
    assert!(!metal::dispatch_census_is_active());
    eprintln!(
        "{}",
        json!({
            "protocol": VALID_PROTOCOL, "event": "context", "run_kind": kind, "device": ctx.describe(),
            "fixture_sha256": format!("{:x}", Sha256::digest(include_bytes!("valid_ids.rs"))),
            "shared_fixture_sha256": format!("{:x}", Sha256::digest(include_bytes!("../candidate_publish.rs"))),
            "msl_sha256": format!("{:x}", Sha256::digest(include_bytes!("../../../../../../../../kernels/deepseek_v41_attention.metal"))),
            "domains": VALID_DOMAINS, "kept_id_count": RETAINED, "id_orders": ["ascending", "score_permuted"],
            "set_formula": "i*(domain-1)/2047 for i=0..2048 exclusive",
            "rank_formula": "ascending[(2047 + rank*1597)%2048]",
            "score_order_is_synthetic_proxy": true, "candidate_block": Config::default().candidate_block,
            "candidate_topk": Config::default().candidate_topk, "dense_mask_retained": true,
            "production_dispatch_changed": false, "mask_poison_u32": NAN_POISON,
        })
    );
    ctx
}

fn correctness(ctx: &MetalContext, p: &Pipelines, f: &Fixture, orders: &IdOrders) {
    let expected = reference(f.domain, &orders.ascending);
    assert_eq!(reference(f.domain, &orders.score_permuted), expected);
    assert_eq!(expected.iter().sum::<u32>() as usize, RETAINED);
    assert_eq!(expected[f.domain - 1], 1);
    reset(f);
    execute(ctx, |enc| f.encode_observer(enc, p));
    assert!(!matches(&f.mask.read(), &expected), "no-op control");
    assert!(
        !matches(&f.observed.read(), &expected),
        "no-op observer control"
    );
    let mut wrong = expected.clone();
    wrong[f.domain - 1] = 0;
    f.mask.write(&wrong);
    execute(ctx, |enc| f.encode_observer(enc, p));
    assert!(
        !matches(&f.mask.read(), &expected),
        "wrong newest-word control"
    );
    assert!(
        !matches(&f.observed.read(), &expected),
        "wrong-word observer control"
    );
    drop(wrong);
    reset(f);
    execute(ctx, |enc| f.encode(enc, p, Arm::Incumbent, true));
    let incumbent = f.check(&expected);
    reset(f);
    execute(ctx, |enc| f.encode(enc, p, Arm::ClearScatter, true));
    assert_eq!(f.check(&expected), incumbent);
}

fn correctness_grid(ctx: &MetalContext, p: &Pipelines) {
    let mut case = 0;
    for domain in VALID_DOMAINS {
        for order in ID_ORDERS {
            with_valid_fixture(ctx, domain, order, "correctness", |f, orders| {
                correctness(ctx, p, f, orders);
            });
            eprintln!(
                "{}",
                json!({
                    "protocol": VALID_PROTOCOL, "event": "correctness_case", "case": case,
                    "domain_words": domain, "kept_id_count": RETAINED, "id_order": order.name(),
                    "independent_cpu": true, "incumbent": true, "candidate": true,
                    "next_dispatch_consumer": true, "cpu_permutation_invariance": true,
                    "no_op_and_wrong_word_rejected": true, "valid_unique_ids": RETAINED,
                    "invalid_ids": 0, "duplicate_ids": 0, "newest_block_present": true,
                })
            );
            case += 1;
        }
    }
    assert_eq!(case, 6);
    eprintln!(
        "{}",
        json!({"protocol": VALID_PROTOCOL, "event": "correctness_complete", "cases": case})
    );
}

#[test]
#[ignore = "parent-only self-leased M5 valid-unique correctness; launch MTL_DEBUG_LAYER=1; <64 MiB"]
fn candidate_publish_valid_ids_m5_correctness() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("production lease and wired guard");
    let ctx = context("correctness");
    let p = Pipelines::new(&ctx);
    correctness_grid(&ctx, &p);
}

#[test]
#[ignore = "parent-only self-leased M5 valid-unique native timing; validation flags absent at launch"]
fn candidate_publish_valid_ids_m5_native_timing() {
    for (name, _) in std::env::vars_os() {
        assert!(
            !forbidden_native_variable(&name.to_string_lossy()),
            "native launch must omit {name:?}"
        );
    }
    let _lease = metal::acquire_metal_benchmark_lease().expect("production lease and wired guard");
    let ctx = context("native_timing");
    let p = Pipelines::new(&ctx);
    correctness_grid(&ctx, &p);
    eprintln!(
        "{}",
        json!({
            "protocol": VALID_PROTOCOL, "event": "timing_begin", "cases": 6, "rounds": ROUNDS,
            "orders": ["ABBA", "BAAB"], "publications_per_sample": BATCH,
            "samples_per_case": 24, "total_samples": 144, "warmup_samples": 0, "retries": 0,
            "spread_or_gain_gate": false, "known_validation_flags_absent": true,
            "scope": "synthetic valid-unique publication confirmation only; NOT producer/model/route timing",
            "includes": "command creation + serial encoding + all publication dispatches + commit + wait",
            "excludes": "allocation, compilation, reset, oracle, readback, observer, model, selection",
            "sample_pool_setup_and_drain_in_cpu_wall": false,
            "post_sample_observer_dispatches": 1, "post_sample_observer_timed": false,
        })
    );
    use Arm::{ClearScatter as B, Incumbent as A};
    let mut sample = 0;
    let mut case = 0;
    for domain in VALID_DOMAINS {
        for id_order in ID_ORDERS {
            with_valid_fixture(&ctx, domain, id_order, "timing", |f, _| {
                let expected = reference(domain, &f.input);
                let ids_hash = id_digest(&f.input);
                for round in 0..ROUNDS {
                    for (order, arms) in [("ABBA", [A, B, B, A]), ("BAAB", [B, A, A, B])] {
                        for (slot, arm) in arms.into_iter().enumerate() {
                            autoreleasepool(|_| {
                                reset(f);
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
                                eprintln!(
                                    "{}",
                                    json!({
                                        "protocol": VALID_PROTOCOL, "event": "sample", "sample": sample, "case": case,
                                        "domain_words": domain, "kept_id_count": RETAINED, "id_order": id_order.name(),
                                        "ids_sha256_le": ids_hash, "round": round, "order": order, "slot": slot, "arm": arm.name(),
                                        "publications": BATCH, "dispatches": BATCH * arm.dispatches(),
                                        "logical_incumbent_comparisons_per_publication": domain * RETAINED,
                                        "cpu_wall_ns": wall_ns, "gpu_start_s": start, "gpu_end_s": end, "gpu_timestamps_valid": valid,
                                        "gpu_batch_ns": if valid { Some((end - start) * 1e9) } else { None },
                                        "gpu_ns_per_publication": if valid { Some((end - start) * 1e9 / BATCH as f64) } else { None },
                                        "cpu_ns_per_publication": wall_ns as f64 / BATCH as f64,
                                    })
                                );
                                assert!(
                                    valid,
                                    "GPU timestamps unavailable; sample retained, no retry"
                                );
                                // The next GPU dispatch observes the final publication, outside both timers.
                                execute(&ctx, |enc| f.encode_observer(enc, &p));
                                drop(f.check(&expected));
                                eprintln!(
                                    "{}",
                                    json!({
                                        "protocol": VALID_PROTOCOL, "event": "sample_check", "sample": sample, "case": case,
                                        "domain_words": domain, "kept_id_count": RETAINED, "id_order": id_order.name(),
                                        "next_dispatch_consumer": true, "independent_cpu": true, "every_word_and_guards": true,
                                    })
                                );
                                sample += 1;
                            });
                        }
                    }
                }
            });
            case += 1;
        }
    }
    assert_eq!((case, sample), (6, 144));
    eprintln!(
        "{}",
        json!({"protocol": VALID_PROTOCOL, "event": "timing_complete", "cases": case, "samples": sample})
    );
}
