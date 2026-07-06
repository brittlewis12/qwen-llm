// B0 topology probe kernels (bench-only; never referenced by production
// PSO tables). Design: docs/bench/2026-07-05-b0-topology-probe/README.md
// (cx-signed, session 019f347b-c).
//
// MEMORY MODEL (build-time finding recorded in the design doc): this
// toolchain accepts ONLY memory_order_relaxed on device atomics; there is
// no atomic_thread_fence in any signature. Consequently:
//   - arm S slot rings are SELF-VALIDATING: the payload stored in a slot is
//     the epoch value itself, so no cross-object ordering is assumed;
//   - tp_reorder measures the cross-object reordering rate directly (its
//     payload hash is odd and therefore invertible mod 2^32, so staleness
//     is exact, not windowed);
//   - arm D-global orders each TG's device writes with
//     threadgroup_barrier(mem_flags::mem_device) around a sense-reversing
//     relaxed-atomic barrier, and the host verifies stage checksums.
// Every wait in this file is BOUNDED and records timeouts. Abort paths
// propagate through threadgroup memory so ALL threads of a TG exit through
// the SAME barrier sites (no divergent barriers). No kernel can deadlock;
// every kernel terminates within its spin budgets.

#include <metal_stdlib>
using namespace metal;

struct TpParams {
    uint spin_iters;     // dwell/work loop length (calibrated by tp_calibrate)
    uint grid_tgs;       // TGs in this grid (for barriers)
    uint stages;         // K for chain kernels
    uint epochs;         // producer epoch count (arm S)
    uint cadence_iters;  // producer inter-epoch spin (arm S)
    uint spin_budget;    // bounded wait/poll budget (arms S and D-global)
    uint traffic_mask;   // traffic index mask (power-of-two elems - 1)
    uint sample_every;   // arm S latency sampling stride (>= 1)
    uint n_elems;        // chain element count (may be < launch width for
                         // the global-wide persistent-host variant)
};

// Dependent-FMA spin. Seeded from memory and returned to the caller so the
// compiler can neither fold nor elide it.
static inline float tp_spin(float seed, uint iters) {
    float x = seed;
    for (uint i = 0; i < iters; i++) {
        x = fma(x, 1.0000001f, 1e-7f);
    }
    return x;
}

// ---------------------------------------------------------------------------
// Calibration.
// tp_calibrate: host times spin_iters dependent FMAs -> ns per spin iter.
// tp_poll_calibrate: host times spin_budget relaxed polls of an atomic that
// never changes -> ns per poll (arm S latency unit).
// ---------------------------------------------------------------------------
kernel void tp_calibrate(constant TpParams& p [[buffer(0)]],
                         device const float* seed [[buffer(1)]],
                         device float* out [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    out[gid] = tp_spin(seed[gid & 1023u], p.spin_iters);
}

kernel void tp_poll_calibrate(constant TpParams& p [[buffer(0)]],
                              device atomic_uint* flag [[buffer(1)]],
                              device uint* out [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    uint seen = 0;
    for (uint i = 0; i < p.spin_budget; i++) {
        seen += atomic_load_explicit(&flag[0], memory_order_relaxed);
    }
    out[gid] = seen; // flag holds 0; consumed so the loop cannot be elided
}

// ---------------------------------------------------------------------------
// Arm R: residency census.
// Entry: thread 0 registers into g_alive/g_max_alive and records the alive
// value seen at entry. All threads dwell; hi-pressure variants carry NACC
// live accumulators seeded from device memory and reduced into out (host-
// checked) so the pressure cannot be folded away. Exit: thread 0 deregisters.
//   buffer 2: census {g_alive, g_max_alive} (atomic_uint[2])
//   buffer 3: entry_alive[tg_id]
//   buffer 4: out[tg_id] (accumulator reduction, host-checked)
//   buffer 5: traffic buffer (TRAFFIC variants; bound to a 1-elem dummy
//             otherwise)
// ---------------------------------------------------------------------------
#define TP_RESIDENCY(NAME, NACC, TRAFFIC)                                      \
kernel void NAME(constant TpParams& p [[buffer(0)]],                           \
                 device const float* seed [[buffer(1)]],                        \
                 device atomic_uint* census [[buffer(2)]],                      \
                 device uint* entry_alive [[buffer(3)]],                        \
                 device float* out [[buffer(4)]],                               \
                 device const float* traffic [[buffer(5)]],                     \
                 uint tg_id [[threadgroup_position_in_grid]],                   \
                 uint lid [[thread_position_in_threadgroup]],                   \
                 uint tg_w [[threads_per_threadgroup]]) {                       \
    if (lid == 0) {                                                             \
        uint alive = atomic_fetch_add_explicit(&census[0], 1u,                  \
                                               memory_order_relaxed) + 1u;      \
        atomic_fetch_max_explicit(&census[1], alive, memory_order_relaxed);     \
        entry_alive[tg_id] = alive;                                             \
    }                                                                           \
    threadgroup_barrier(mem_flags::mem_none);                                   \
    float acc[NACC];                                                            \
    for (uint a = 0; a < NACC; a++) {                                           \
        acc[a] = seed[(tg_id * tg_w + lid + a) & 1023u];                        \
    }                                                                           \
    uint t_idx = (tg_id * tg_w + lid) * 97u;                                    \
    float t_sum = 0.0f;                                                         \
    for (uint i = 0; i < p.spin_iters; i++) {                                   \
        for (uint a = 0; a < NACC; a++) {                                       \
            acc[a] = fma(acc[a], 1.0000001f + (float)a * 1e-9f, 1e-7f);         \
        }                                                                       \
        if (TRAFFIC) {                                                          \
            t_sum += traffic[t_idx & p.traffic_mask];                           \
            t_idx += 16411u; /* prime stride, defeats cache reuse */            \
        }                                                                       \
    }                                                                           \
    float r = t_sum;                                                            \
    for (uint a = 0; a < NACC; a++) { r += acc[a]; }                            \
    threadgroup_barrier(mem_flags::mem_none);                                   \
    if (lid == 0) {                                                             \
        atomic_fetch_sub_explicit(&census[0], 1u, memory_order_relaxed);        \
        out[tg_id] = r;                                                         \
    }                                                                           \
}

TP_RESIDENCY(tp_residency_lo,      1,  false)
TP_RESIDENCY(tp_residency_hi8,     8,  false)
TP_RESIDENCY(tp_residency_hi32,    32, false)
TP_RESIDENCY(tp_residency_hi64,    64, false)
TP_RESIDENCY(tp_residency_lo_tr,   1,  true)
TP_RESIDENCY(tp_residency_hi8_tr,  8,  true)
TP_RESIDENCY(tp_residency_hi32_tr, 32, true)
TP_RESIDENCY(tp_residency_hi64_tr, 64, true)

// ---------------------------------------------------------------------------
// Arm S: bounded one-way signaling through a self-validating epoch ring.
// TG 0 is the producer; TGs >= 1 are consumers (thread 0 polls).
//   ring slot value IS the epoch number: slot[e % TP_RING] == e once
//   published, so observing v proves delivery of all epochs <= v and no
//   cross-object ordering is assumed.
//   buffer 2: ring (atomic_uint[TP_RING])
//   buffer 3: per-consumer stats [n_consumers x 4]:
//             {last_epoch_seen, corrupt_reads, timed_out, budget_left}
//   buffer 4: latency samples [n_consumers x TP_LAT_SAMPLES] in POLL counts
//             (0xFFFFFFFF = slot unused); only clean single-epoch advances
//             are sampled so the metric stays interpretable.
// Producer never waits on consumers. Consumers give up when the total poll
// budget expires. Cannot deadlock by construction.
// ---------------------------------------------------------------------------
#define TP_RING 64u
#define TP_LAT_SAMPLES 64u

kernel void tp_signal(constant TpParams& p [[buffer(0)]],
                      device const float* seed [[buffer(1)]],
                      device atomic_uint* ring [[buffer(2)]],
                      device uint* stats [[buffer(3)]],
                      device uint* lat [[buffer(4)]],
                      device float* sink [[buffer(5)]],
                      uint tg_id [[threadgroup_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]]) {
    if (tg_id == 0) {
        if (lid == 0) {
            float x = seed[0];
            for (uint e = 1; e <= p.epochs; e++) {
                x = tp_spin(x, p.cadence_iters);
                atomic_store_explicit(&ring[e % TP_RING], e,
                                      memory_order_relaxed);
            }
            sink[0] = x;
        }
        return;
    }
    if (lid != 0) {
        return;
    }
    uint c = tg_id - 1;
    uint last = 0;
    uint corrupt = 0;
    uint budget = p.spin_budget;
    uint waited = 0;
    while (last < p.epochs && budget > 0) {
        uint next = last + 1;
        uint v = atomic_load_explicit(&ring[next % TP_RING],
                                      memory_order_relaxed);
        budget--;
        waited++;
        if (v >= next && v <= p.epochs) {
            if ((v - next) % TP_RING != 0u) {
                // impossible for a published epoch in this slot: corrupt
                corrupt++;
                continue;
            }
            if (v == next && (v % p.sample_every) == 0u) {
                uint s = (v / p.sample_every) % TP_LAT_SAMPLES;
                lat[c * TP_LAT_SAMPLES + s] = waited;
            }
            last = v;
            waited = 0;
        }
    }
    stats[c * 4 + 0] = last;
    stats[c * 4 + 1] = corrupt;
    stats[c * 4 + 2] = (last < p.epochs) ? 1u : 0u;
    stats[c * 4 + 3] = budget;
}

// Cross-object reordering probe: producer writes payload A then flag B
// (both relaxed, separate cache lines); consumers poll B, then read A.
// Payload hash TP_H is odd => invertible mod 2^32 (Newton), so the epoch
// carried by A is decoded exactly and staleness (A older than B) cannot be
// confused with the producer racing ahead.
//   buffer 2: ab atomics; A at [0], B at [TP_B_OFF]
//   buffer 3: per-consumer stats [n x 4]:
//             {fresh_observations, stale_after_fresh, corrupt, timed_out}
// Host runs this only at cadence >= 25 us rows (design doc).
#define TP_B_OFF 32u
#define TP_H 2654435761u

static inline uint tp_h_inv() {
    uint x = TP_H; // Newton: x_{n+1} = x*(2 - h*x); 5 steps suffice mod 2^32
    for (uint i = 0; i < 5; i++) {
        x = x * (2u - TP_H * x);
    }
    return x;
}

kernel void tp_reorder(constant TpParams& p [[buffer(0)]],
                       device const float* seed [[buffer(1)]],
                       device atomic_uint* ab [[buffer(2)]],
                       device uint* stats [[buffer(3)]],
                       device float* sink [[buffer(4)]],
                       uint tg_id [[threadgroup_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]]) {
    if (tg_id == 0) {
        if (lid == 0) {
            float x = seed[0];
            for (uint e = 1; e <= p.epochs; e++) {
                atomic_store_explicit(&ab[0], e * TP_H, memory_order_relaxed);
                atomic_store_explicit(&ab[TP_B_OFF], e, memory_order_relaxed);
                x = tp_spin(x, p.cadence_iters);
            }
            sink[0] = x;
        }
        return;
    }
    if (lid != 0) {
        return;
    }
    const uint hinv = tp_h_inv();
    uint c = tg_id - 1;
    uint last = 0;
    uint fresh = 0;
    uint stale = 0;
    uint corrupt = 0;
    uint budget = p.spin_budget;
    while (last < p.epochs && budget > 0) {
        uint b = atomic_load_explicit(&ab[TP_B_OFF], memory_order_relaxed);
        budget--;
        if (b > last) {
            uint a = atomic_load_explicit(&ab[0], memory_order_relaxed);
            uint ea = a * hinv; // exact epoch decoded from payload
            fresh++;
            if (ea == 0u || ea > p.epochs) {
                corrupt++; // not a value the producer ever wrote
            } else if (ea < b) {
                stale++;   // B's write visible before A's: cross-object reorder
            }
            last = b;
        }
    }
    stats[c * 4 + 0] = fresh;
    stats[c * 4 + 1] = stale;
    stats[c * 4 + 2] = corrupt;
    stats[c * 4 + 3] = (last < p.epochs) ? 1u : 0u;
}

// ---------------------------------------------------------------------------
// Arm D: boundary drain vs persistence.
// Elementwise dependent chain: out[i] = spin(in[i], work), K stages deep.
// The dataflow is slice-parallel BY CONSTRUCTION (matches the B1a
// glue-ladder scoping). Final output depends only on (in, K x spin_iters),
// so ladder and all persistent variants must agree bit-exactly.
// ---------------------------------------------------------------------------

// Ladder stage: one dispatch per stage on a serial encoder (production
// dependency semantics: full barrier between consecutive dispatches).
// Two identical entry points (a/b) so the host can ALTERNATE PSOs between
// consecutive stages, matching production's per-dispatch state changes.
kernel void tp_chain_stage(constant TpParams& p [[buffer(0)]],
                           device const float* in [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid < p.n_elems) {
        out[gid] = tp_spin(in[gid], p.spin_iters);
    }
}

kernel void tp_chain_stage_b(constant TpParams& p [[buffer(0)]],
                             device const float* in [[buffer(1)]],
                             device float* out [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid < p.n_elems) {
        out[gid] = tp_spin(in[gid], p.spin_iters);
    }
}

// D-local (mem): one dispatch; each thread carries its element through all
// K stages with a device-memory round-trip per stage (same traffic as the
// ladder) and NO cross-TG or cross-thread wait: each thread re-reads only
// the element it wrote itself, so no barrier is required. Isolates dispatch-
// boundary cost from memory round-trips when compared against the ladder.
kernel void tp_chain_persistent_local_mem(constant TpParams& p [[buffer(0)]],
                                          device const float* in [[buffer(1)]],
                                          device float* out [[buffer(2)]],
                                          device float* scratch [[buffer(3)]],
                                          uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n_elems) {
        return;
    }
    float x = tp_spin(in[gid], p.spin_iters);
    for (uint s = 1; s < p.stages; s++) {
        // round-trip through device memory, alternating buffers
        device float* d = (s & 1u) ? scratch : out;
        d[gid] = x;
        x = tp_spin(d[gid], p.spin_iters);
    }
    out[gid] = x;
}

// D-local (reg): as above but intermediates stay in registers - the upper
// bound of what slice-local fusion can recover (B1a upside bound).
kernel void tp_chain_persistent_local_reg(constant TpParams& p [[buffer(0)]],
                                          device const float* in [[buffer(1)]],
                                          device float* out [[buffer(2)]],
                                          uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n_elems) {
        return;
    }
    float x = in[gid];
    for (uint s = 0; s < p.stages; s++) {
        x = tp_spin(x, p.spin_iters);
    }
    out[gid] = x;
}

// D-global: persistent grid-stride kernel; ALL TGs advance through stages in
// lockstep via a sense-reversing barrier on relaxed atomics. Grid MUST be
// sized to co-residency (arm R low-water) by the host.
//   buffer 3: bar {arrive, generation} (atomic_uint[2])
//   buffer 4: abort_flag[0] (uint): nonzero => some TG timed out at stage
//             (value-1); host discards the run's timing and records the
//             abort (a forward-progress data point)
//   buffer 5: scratch ping-pong buffer (n elems)
// Abort propagates via threadgroup memory so all threads of the TG exit
// through the SAME barrier sites (no divergent barriers).
kernel void tp_chain_persistent_global(constant TpParams& p [[buffer(0)]],
                                       device const float* in [[buffer(1)]],
                                       device float* out [[buffer(2)]],
                                       device atomic_uint* bar [[buffer(3)]],
                                       device uint* abort_flag [[buffer(4)]],
                                       device float* scratch [[buffer(5)]],
                                       uint tg_id [[threadgroup_position_in_grid]],
                                       uint lid [[thread_position_in_threadgroup]],
                                       uint tg_w [[threads_per_threadgroup]],
                                       uint grid_w [[threads_per_grid]]) {
    // n may be < grid_w: the global-wide variant hosts narrow stages on a
    // full persistent grid; TGs with no elements still pay every barrier
    // (that IS the persistent-host cost being measured).
    const uint n = p.n_elems;
    threadgroup uint tg_abort;
    if (lid == 0) {
        tg_abort = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint my_gen = 0;
    for (uint s = 0; s < p.stages; s++) {
        device const float* src = (s == 0) ? in : ((s & 1u) ? scratch : out);
        device float* dst = (s & 1u) ? out : scratch;
        if (s + 1 == p.stages) {
            dst = out; // final stage always lands in out (in-place is safe:
                       // elementwise, one owner thread per index per stage)
        }
        for (uint i = tg_id * tg_w + lid; i < n; i += grid_w) {
            dst[i] = tp_spin(src[i], p.spin_iters);
        }
        // make this TG's stage writes visible before arriving
        threadgroup_barrier(mem_flags::mem_device);
        if (lid == 0) {
            uint arrived = atomic_fetch_add_explicit(&bar[0], 1u,
                                                     memory_order_relaxed) + 1u;
            if (arrived == p.grid_tgs) {
                atomic_store_explicit(&bar[0], 0u, memory_order_relaxed);
                atomic_fetch_add_explicit(&bar[1], 1u, memory_order_relaxed);
            } else {
                uint budget = p.spin_budget;
                while (budget > 0) {
                    uint gen = atomic_load_explicit(&bar[1],
                                                    memory_order_relaxed);
                    if (gen > my_gen) {
                        break;
                    }
                    budget--;
                }
                if (budget == 0) {
                    abort_flag[0] = 1u + s;
                    tg_abort = 1u;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        if (tg_abort != 0u) {
            return; // uniform across the TG: same barrier sites for all
        }
        my_gen++;
    }
}
