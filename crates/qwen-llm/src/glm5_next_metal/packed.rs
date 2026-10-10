//! Packed (multi-row) prefill for the GLM-5.3 session: one command buffer per
//! chunk of up to `rows` prompt tokens, every block encoded over all rows.
//!
//! Same graph as single-token decode, with batched encoders: mHC rows,
//! row-wise norms, mat-mat projections, packed KDA, causal packed latent
//! attention, row-wise indexer appends, and expert-major routed experts.
//! Latent absorption runs per row through the decode grouped GEMV (exact
//! decode lineage). In [`PackedLineage::Fast`] (default) projections and
//! grouped experts stage activations in half, as llama.cpp's batched prefill
//! does, so packed logits match serial decode closely but not bitwise.
//! [`PackedLineage::Exact`] runs every projection and expert per row through
//! the decode kernels and reproduces serial decode; it exists as the
//! equivalence reference, not as a fast path.

use super::*;
use crate::metal::{
    GroupedDownPolicy, GroupedExperts, encode_grouped_routed_experts_f32x,
    encode_grouped_routed_experts_with_down_policy, encode_indexer_append_rows, encode_kda_prefill,
    encode_mhc4_post_rows, encode_mhc4_repeat_rows, encode_rms_norm_mul_rows_f32,
    encode_route_learned_rows,
};
use objc2_metal::MTLComputePipelineState;

// Diagnostic overrides and timing are absent from product builds.
#[cfg(test)]
#[path = "tests/router_prefill.rs"]
mod router_prefill;

#[cfg(test)]
#[path = "tests/expert_down.rs"]
mod expert_down;

#[cfg(test)]
#[path = "tests/mla_tail.rs"]
mod mla_tail;

#[cfg(test)]
thread_local! {
    static ABSORB_TAIL: std::cell::Cell<(Option<bool>, usize)> = const { std::cell::Cell::new((None, 0)) };
}

#[cfg(test)]
pub(super) fn with_absorb_tail_variant<R>(
    variant: Option<bool>,
    f: impl FnOnce() -> R,
) -> (R, usize) {
    struct Restore((Option<bool>, usize));
    impl Drop for Restore {
        fn drop(&mut self) {
            ABSORB_TAIL.with(|state| state.set(self.0));
        }
    }
    let _restore = Restore(ABSORB_TAIL.with(|state| state.replace((variant, 0))));
    let result = f();
    (result, ABSORB_TAIL.with(|state| state.get().1))
}

#[cfg(test)]
fn absorb_tail_rows(lineage: PackedLineage, rows: usize) -> usize {
    if lineage == PackedLineage::Fast && ABSORB_TAIL.with(|state| state.get().0) == Some(true) {
        rows % 128 / 8 * 8
    } else {
        0
    }
}

/// Packed stage families that have a Fast and an Exact form (map #12
/// attribution: one family at a time can run in its Exact form in tests).
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum Stage {
    DenseFfn,
    Router,
    RoutedExperts,
    SharedExpert,
    KdaProjection,
    KdaExpand,
    MlaProjection,
    MlaAbsorb,
    IndexerProjection,
}

impl Stage {
    #[cfg(test)]
    pub(super) const ALL: [Stage; 9] = [
        Stage::DenseFfn,
        Stage::Router,
        Stage::RoutedExperts,
        Stage::SharedExpert,
        Stage::KdaProjection,
        Stage::KdaExpand,
        Stage::MlaProjection,
        Stage::MlaAbsorb,
        Stage::IndexerProjection,
    ];

    #[cfg(test)]
    fn bit(self) -> u16 {
        1 << self as u16
    }
}

#[cfg(test)]
thread_local! {
    static EXACT_STAGES: std::cell::Cell<u16> = const { std::cell::Cell::new(0) };
}

/// A stage's lineage: the session's, except that tests may run chosen Fast
/// stages in their Exact form ([`ExactStages`]). Product builds always use
/// the session's lineage.
fn stage_lineage(lineage: PackedLineage, stage: Stage) -> PackedLineage {
    #[cfg(test)]
    if lineage == PackedLineage::Fast && EXACT_STAGES.with(|s| s.get()) & stage.bit() != 0 {
        return PackedLineage::Exact;
    }
    let _ = stage;
    lineage
}

/// How a stage computes its quantized projections in this session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum StageMode {
    /// The decode kernels, row by row (bitwise equal to serial decode).
    Exact,
    /// Batched mat-mat tiles with half-staged operands.
    Fast,
    /// Batched mat-mat with F32 operands where an F32-operand tile exists
    /// for the weight type and geometry (Q8_0, Q6_K); otherwise as
    /// [`StageMode::Fast`] ([`F32Census`] counts both outcomes in tests).
    FastF32,
}

#[cfg(test)]
thread_local! {
    static F32_STAGES: std::cell::Cell<u16> = const { std::cell::Cell::new(0) };
    static F32_CENSUS: std::cell::RefCell<Option<CensusCounts>> =
        const { std::cell::RefCell::new(None) };
}

/// How a Fast-lineage matrix call ran, for [`F32Census`].
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum CensusPath {
    /// An F32-operand tile.
    F32Tile,
    /// A narrow F32-operand kernel for a short span (the tile's arithmetic).
    F32Narrow,
    /// The batched kernel of an F32 weight (F32 operands already).
    F32Native,
    /// A half-staged batched kernel.
    Half,
}

/// Calls per (stage family, weight kind, path); the kind is the weight's
/// dtype, or "experts" for the routed experts.
#[cfg(test)]
pub(super) type CensusCounts = std::collections::BTreeMap<(Stage, String, CensusPath), usize>;

/// Counts one Fast-lineage call while an [`F32Census`] is active.
#[cfg(test)]
fn record_census(stage: Stage, kind: String, path: CensusPath) {
    F32_CENSUS.with(|census| {
        if let Some(census) = census.borrow_mut().as_mut() {
            *census.entry((stage, kind, path)).or_default() += 1;
        }
    });
}

/// Test-only census of every Fast-lineage matrix call of the stage families
/// on this thread (Fast and [`StageMode::FastF32`] alike): per stage family
/// and weight kind, how many ran on an F32-operand tile, on an F32 weight's
/// own kernel, or half-staged. Comparing a half-staged run's census with a
/// selection's shows both coverage (the same calls) and the path each took.
/// Recording stops on drop.
#[cfg(test)]
pub(super) struct F32Census(());

#[cfg(test)]
impl F32Census {
    pub(super) fn begin() -> Self {
        F32_CENSUS.with(|census| *census.borrow_mut() = Some(Default::default()));
        Self(())
    }

    /// The counts so far, restarting the census.
    pub(super) fn take(&self) -> CensusCounts {
        F32_CENSUS.with(|census| {
            census
                .borrow_mut()
                .as_mut()
                .map(std::mem::take)
                .unwrap_or_default()
        })
    }
}

#[cfg(test)]
impl Drop for F32Census {
    fn drop(&mut self) {
        F32_CENSUS.with(|census| *census.borrow_mut() = None);
    }
}

/// A stage's projection mode: its lineage ([`stage_lineage`]); Fast stages
/// the session's [`FastPrecision`] selects run in their F32-operand form.
/// Tests may substitute their own selection ([`F32Stages`]).
fn stage_mode(arith: Arith, stage: Stage) -> StageMode {
    match stage_lineage(arith.lineage, stage) {
        PackedLineage::Exact => StageMode::Exact,
        PackedLineage::Fast => {
            #[cfg(test)]
            if let Some(stages) = test_f32_stages() {
                return if stages & stage.bit() != 0 {
                    StageMode::FastF32
                } else {
                    StageMode::Fast
                };
            }
            if arith.precision.f32_stage(stage) {
                StageMode::FastF32
            } else {
                StageMode::Fast
            }
        }
    }
}

/// The F32-operand tiles a [`StageMode::FastF32`] projection may take.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum F32Tile {
    /// `encode_mat_mat_q8_0_f32_r2c4k64` (`n_in % 64`, `n_out % 16`).
    Q8_0,
    /// `encode_mat_mat_q6_k_f32_mm64x32` (`n_in % 256`).
    Q6K,
    /// `encode_grouped_routed_experts_f32x` (gate/up IQ2_S or IQ3_S, down
    /// IQ3_S or IQ4_XS).
    RoutedExperts,
}

impl F32Tile {
    #[cfg(test)]
    pub(super) const ALL: [F32Tile; 3] = [F32Tile::Q8_0, F32Tile::Q6K, F32Tile::RoutedExperts];

    #[cfg(test)]
    fn bit(self) -> u8 {
        1 << self as u8
    }
}

#[cfg(test)]
thread_local! {
    static F32_TILES: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// Spans of at most this many rows take the narrow F32-operand kernels
/// (`encode_mat_mat_q6_k_f32_r8c8`, `encode_mat_mat_q8_0_f32_r2c1k64`),
/// bitwise equal per token to the wide tiles; tests may lower it
/// ([`F32NarrowRows`]) to compare the two.
const F32_NARROW_MAX_ROWS: usize = 8;

#[cfg(test)]
thread_local! {
    static F32_NARROW_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(F32_NARROW_MAX_ROWS) };
}

#[cfg(test)]
fn f32_narrow_max_rows() -> usize {
    F32_NARROW_ROWS.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn f32_narrow_max_rows() -> usize {
    F32_NARROW_MAX_ROWS
}

/// Test-only scope in which FastF32 spans of at most `rows` rows take the
/// narrow kernels (0: none); the previous limit is restored on drop.
#[cfg(test)]
pub(super) struct F32NarrowRows(usize);

#[cfg(test)]
impl F32NarrowRows {
    pub(super) fn set(rows: usize) -> Self {
        Self(F32_NARROW_ROWS.with(|r| r.replace(rows)))
    }
}

#[cfg(test)]
impl Drop for F32NarrowRows {
    fn drop(&mut self) {
        F32_NARROW_ROWS.with(|r| r.set(self.0));
    }
}

/// The stage families a test's [`F32Stages`] scope selects, if one is active
/// (it then replaces the session's precision).
#[cfg(test)]
fn test_f32_stages() -> Option<u16> {
    Some(F32_STAGES.with(|s| s.get())).filter(|&stages| stages != 0)
}

/// Whether a FastF32 projection may take `tile`: every tile, unless a test's
/// [`F32Stages`] scope names a subset.
#[cfg(test)]
fn f32_tile_enabled(tile: F32Tile) -> bool {
    test_f32_stages().is_none() || F32_TILES.with(|t| t.get()) & tile.bit() != 0
}

#[cfg(not(test))]
fn f32_tile_enabled(_: F32Tile) -> bool {
    true
}

/// Test-only scope in which `stages` of Fast sessions on this thread use
/// F32-operand tiles where one exists; the previous selection is restored on
/// drop.
#[cfg(test)]
pub(super) struct F32Stages(u16, u8);

#[cfg(test)]
impl F32Stages {
    /// `stages` may take only `tiles`; other calls stay half-staged.
    pub(super) fn with_tiles(stages: &[Stage], tiles: &[F32Tile]) -> Self {
        let bits = stages.iter().fold(0, |bits, stage| bits | stage.bit());
        let tile_bits = tiles.iter().fold(0, |bits, tile| bits | tile.bit());
        Self(
            F32_STAGES.with(|s| s.replace(bits)),
            F32_TILES.with(|t| t.replace(tile_bits)),
        )
    }
}

#[cfg(test)]
impl Drop for F32Stages {
    fn drop(&mut self) {
        F32_STAGES.with(|s| s.set(self.0));
        F32_TILES.with(|t| t.set(self.1));
    }
}

#[cfg(test)]
thread_local! {
    static ROUND_EXACT_ACTIVATIONS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static ROUND_SCRATCH: std::cell::RefCell<Option<MetalTensor>> = const { std::cell::RefCell::new(None) };
}

/// Test-only precision probe (map #12): in this scope, Exact-lineage matrix
/// inputs with quantized weights (projections, KDA expansions, routed
/// experts' gate/up and down inputs) are rounded through half precision
/// first, on a scratch copy, as half-staged batched kernels round theirs.
/// Weights stay F32-dequantized; accumulation is the decode kernels'.
#[cfg(test)]
pub(super) struct RoundExactActivations(bool);

#[cfg(test)]
impl RoundExactActivations {
    pub(super) fn set() -> Self {
        Self(ROUND_EXACT_ACTIVATIONS.with(|s| s.replace(true)))
    }
}

#[cfg(test)]
impl Drop for RoundExactActivations {
    fn drop(&mut self) {
        ROUND_EXACT_ACTIVATIONS.with(|s| s.set(self.0));
    }
}

/// Under [`RoundExactActivations`], a rounded scratch copy of `x[0..n]`
/// (the packed encoder is serial, so one scratch is reused).
#[cfg(test)]
fn rounded_input(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    n: usize,
) -> Result<Option<MetalTensor>> {
    if !ROUND_EXACT_ACTIVATIONS.with(std::cell::Cell::get) {
        return Ok(None);
    }
    ROUND_SCRATCH.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.as_ref().is_none_or(|t| (t.n_elements() as usize) < n) {
            *slot = Some(MetalTensor::zeros_f32(
                ctx,
                vec![n.max(512 * 16384) as u64],
            )?);
        }
        let scratch = slot
            .as_ref()
            .expect("allocated")
            .view_subrange(0, vec![n as u64]);
        crate::metal::encode_copy_offset_f32(ctx, enc, x, 0, &scratch, n)?;
        crate::metal::encode_round_trip_f16_f32(ctx, enc, &scratch, n)?;
        Ok(Some(scratch))
    })
}

/// Test-only scope in which `stages` run in their Exact form inside Fast
/// sessions on this thread; the previous set is restored on drop.
#[cfg(test)]
pub(super) struct ExactStages(u16);

#[cfg(test)]
impl ExactStages {
    pub(super) fn set(stages: &[Stage]) -> Self {
        let bits = stages.iter().fold(0, |bits, stage| bits | stage.bit());
        Self(EXACT_STAGES.with(|s| s.replace(bits)))
    }
}

#[cfg(test)]
impl Drop for ExactStages {
    fn drop(&mut self) {
        EXACT_STAGES.with(|s| s.set(self.0));
    }
}

/// Operand precision of Fast packed prefill's quantized matrices (map #12
/// accuracy lane). F32 operands bring Fast closer to Exact at some prefill
/// cost (`docs/bench/2026-10-09-glm53-fast-f32-operands/`): on the six
/// frozen natural cases the mean KL against Exact falls 30% (`DenseF32`,
/// +5% on a fresh 2,048-token prompt) or 50% (`F32`, +13-17%). Every
/// setting keeps Fast's bitwise properties (chunk identities, snapshot
/// restore); sessions of different precisions never share state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FastPrecision {
    /// Half-staged mat-mat tiles (the original Fast).
    #[default]
    Half,
    /// Every dense projection (KDA, MLA, indexer, shared expert, dense FFN)
    /// on F32-operand tiles; routed experts half-staged.
    DenseF32,
    /// Every quantized matrix operand in F32: dense projections and routed
    /// experts.
    F32,
}

impl FastPrecision {
    /// Every precision, in order of increasing closeness to Exact.
    pub const ALL: [FastPrecision; 3] = [Self::Half, Self::DenseF32, Self::F32];

    /// The setting's name (`half`, `dense_f32`, `f32`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Half => "half",
            Self::DenseF32 => "dense_f32",
            Self::F32 => "f32",
        }
    }

    /// The precision named `name` ([`FastPrecision::name`]).
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == name)
    }

    /// Whether `stage` runs on F32-operand tiles in a Fast session.
    fn f32_stage(self, stage: Stage) -> bool {
        let dense = matches!(
            stage,
            Stage::KdaExpand
                | Stage::KdaProjection
                | Stage::MlaProjection
                | Stage::IndexerProjection
                | Stage::SharedExpert
                | Stage::DenseFfn
        );
        match self {
            Self::Half => false,
            Self::DenseF32 => dense,
            Self::F32 => dense || stage == Stage::RoutedExperts,
        }
    }
}

/// A packed chunk's arithmetic: lineage and, for Fast, operand precision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Arith {
    pub(super) lineage: PackedLineage,
    pub(super) precision: FastPrecision,
}

/// Arithmetic lineage of packed prefill.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PackedLineage {
    /// Batched mat-mat projections and expert-major grouped experts.
    #[default]
    Fast,
    /// Decode kernels per row: identical to serial decode, slower.
    Exact,
}

fn grouped_down_policy(
    lineage: PackedLineage,
    dtype: GgmlType,
    ffn: usize,
    hidden: usize,
    experts: usize,
    rows: usize,
) -> GroupedDownPolicy {
    if lineage == PackedLineage::Fast
        && dtype == GgmlType::IQ3_S
        && (ffn, hidden, experts) == (2048, 4096, 288)
        && (32..=512).contains(&rows)
    {
        GroupedDownPolicy::Iq3SSmallCounts
    } else {
        GroupedDownPolicy::Incumbent
    }
}

// Measured release geometry and widths; this is not a broad size threshold.
const ROUTER_E8P32_ROWS: [usize; 2] = [128, 512];

fn router_e8p32_selected(
    lineage: PackedLineage,
    hidden: usize,
    experts: usize,
    dtype: GgmlType,
    rows: usize,
    supported: impl FnOnce() -> bool,
) -> bool {
    let requested = ROUTER_E8P32_ROWS.contains(&rows);
    // An active A arm must remain generic after production promotion. B may
    // probe other widths, but cannot bypass lineage, geometry or capability.
    #[cfg(test)]
    let requested = router_prefill::requested_override().unwrap_or(requested);
    lineage == PackedLineage::Fast
        && hidden == 4096
        && experts == 288
        && dtype == GgmlType::F32
        && rows > 0
        && requested
        && supported()
}

fn router_e8p32_capacity_supported(
    execution_width: usize,
    max_threads: usize,
    static_bytes: usize,
    device_bytes: usize,
) -> bool {
    // Same requirements as Flash's strict router: 32 threads, no dynamic TGM.
    execution_width == 32 && max_threads >= 32 && static_bytes <= device_bytes
}

fn router_e8p32_supported(ctx: &MetalContext) -> bool {
    match ctx.pipeline("kernel_mat_mat_f32_f32_router_e8p32_strict") {
        Ok(p) => router_e8p32_capacity_supported(
            p.threadExecutionWidth(),
            p.maxTotalThreadsPerThreadgroup(),
            p.staticThreadgroupMemoryLength(),
            ctx.device.maxThreadgroupMemoryLength(),
        ),
        Err(error) => {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::warn!(
                    "glm5_next: strict E8P32 router unavailable, using the generic path: {error}"
                );
            });
            false
        }
    }
}

/// Activations for up to `rows` packed tokens, shared by every block.
pub(super) struct PackedScratch {
    rows: usize,
    pub(super) lineage: PackedLineage,
    pub(super) precision: FastPrecision,
    token: MetalTensor,
    embedding: MetalTensor,
    residual: [MetalTensor; 2],
    hc_partial_dots: MetalTensor,
    hc_partial_sumsq: MetalTensor,
    mixes: MetalTensor,
    pre: MetalTensor,
    post: MetalTensor,
    comb: MetalTensor,
    collapsed: MetalTensor,
    normed: MetalTensor,
    block_out: MetalTensor,
    q: MetalTensor,
    k: MetalTensor,
    v: MetalTensor,
    rank_a: MetalTensor,
    raw_gate: MetalTensor,
    raw_beta: MetalTensor,
    rank_b: MetalTensor,
    output_gate: MetalTensor,
    kda_out: MetalTensor,
    query_a: MetalTensor,
    query_r: MetalTensor,
    query: MetalTensor,
    latent_raw: MetalTensor,
    latent: MetalTensor,
    query_latent: MetalTensor,
    output_latent: MetalTensor,
    heads_out: MetalTensor,
    index_key: MetalTensor,
    index_gate: MetalTensor,
    dense_gate: MetalTensor,
    dense_up: MetalTensor,
    router: MetalTensor,
    counts: MetalTensor,
    slots: MetalTensor,
    inner: MetalTensor,
    slot_out: MetalTensor,
    routed: MetalTensor,
    shared_gate: MetalTensor,
    shared_up: MetalTensor,
    shared: MetalTensor,
    /// Per MoE block: ids and weights `[top_k, rows]`, status `[rows]`.
    routes: Vec<Option<RouteRecord>>,
    /// Present when the session reaches the sparse frontier.
    sparse: Option<PackedSparseScratch>,
}

/// Packed sparse-selection scratch: chunk-wide indexer queries, weights and
/// per-row visibility; microbatch scores, pools and expanded rows (reused
/// microbatch after microbatch and block after block on the serial
/// encoder); one sticky status per (MLA block, chunk row).
pub(super) struct PackedSparseScratch {
    index_query: MetalTensor,
    index_query_f16: MetalTensor,
    index_weights: MetalTensor,
    visible_pools: MetalTensor,
    visible_rows: MetalTensor,
    scores: MetalTensor,
    pool_ids: MetalTensor,
    pool_counts: MetalTensor,
    row_ids: MetalTensor,
    row_counts: MetalTensor,
    select_status: MetalTensor,
    attention_partials: MetalTensor,
    attention_partial_stats: MetalTensor,
    rows: usize,
    microbatch: usize,
}

impl PackedSparseScratch {
    fn new(ctx: &MetalContext, c: &Glm5NextConfig, capacity: u64, rows: u64) -> Result<Self> {
        let specs = memory::packed_sparse_specs(c, capacity, rows);
        let mut b = SpecBuffers::allocate(ctx, "packed_sparse", &specs)?;
        let s = Self {
            index_query: b.take("index_query")?,
            index_query_f16: b.take("index_query_f16")?,
            index_weights: b.take("index_weights")?,
            visible_pools: b.take("visible_pools")?,
            visible_rows: b.take("visible_rows")?,
            scores: b.take("scores")?,
            pool_ids: b.take("pool_ids")?,
            pool_counts: b.take("pool_counts")?,
            row_ids: b.take("row_ids")?,
            row_counts: b.take("row_counts")?,
            select_status: b.take("select_status")?,
            attention_partials: b.take("attention_partials")?,
            attention_partial_stats: b.take("attention_partial_stats")?,
            rows: rows as usize,
            microbatch: rows.min(memory::PACKED_SPARSE_QUERIES) as usize,
        };
        b.finish()?;
        Ok(s)
    }
}

/// An activation buffer allocated with padded rows
/// ([`memory::packed_activation_rows`]), exposed with `rows` rows.
fn take_rows(b: &mut SpecBuffers, name: &str, rows: usize) -> Result<MetalTensor> {
    Ok(rows_view(&b.take(name)?, rows))
}

impl PackedScratch {
    /// This scratch's chunk arithmetic.
    pub(super) fn arith(&self) -> Arith {
        Arith {
            lineage: self.lineage,
            precision: self.precision,
        }
    }
}

#[cfg(test)]
impl PackedScratch {
    /// Each MoE block's routed expert ids (`[top_k, rows]` I32, row-major by
    /// row) as the last packed chunk left them, with the block index.
    pub(super) fn route_ids(&self) -> Vec<(usize, &MetalTensor)> {
        self.routes
            .iter()
            .enumerate()
            .filter_map(|(block, route)| route.as_ref().map(|route| (block, &route.ids)))
            .collect()
    }
}

impl PackedScratch {
    pub(super) fn new(
        ctx: &MetalContext,
        c: &Glm5NextConfig,
        rows: usize,
        capacity: usize,
    ) -> Result<Self> {
        let r = rows as u64;
        let specs = memory::packed_scratch_specs(c, r);
        let mut b = SpecBuffers::allocate(ctx, "packed_scratch", &specs)?;
        let scratch = Self {
            rows,
            lineage: PackedLineage::Fast,
            precision: FastPrecision::Half,
            token: b.take("token")?,
            embedding: take_rows(&mut b, "embedding", rows)?,
            residual: [
                take_rows(&mut b, "residual_a", rows)?,
                take_rows(&mut b, "residual_b", rows)?,
            ],
            hc_partial_dots: take_rows(&mut b, "hc_partial_dots", rows)?,
            hc_partial_sumsq: take_rows(&mut b, "hc_partial_sumsq", rows)?,
            mixes: take_rows(&mut b, "mixes", rows)?,
            pre: take_rows(&mut b, "pre", rows)?,
            post: take_rows(&mut b, "post", rows)?,
            comb: take_rows(&mut b, "comb", rows)?,
            collapsed: take_rows(&mut b, "collapsed", rows)?,
            normed: take_rows(&mut b, "normed", rows)?,
            block_out: take_rows(&mut b, "block_out", rows)?,
            q: take_rows(&mut b, "q", rows)?,
            k: take_rows(&mut b, "k", rows)?,
            v: take_rows(&mut b, "v", rows)?,
            rank_a: take_rows(&mut b, "rank_a", rows)?,
            raw_gate: take_rows(&mut b, "raw_gate", rows)?,
            raw_beta: take_rows(&mut b, "raw_beta", rows)?,
            rank_b: take_rows(&mut b, "rank_b", rows)?,
            output_gate: take_rows(&mut b, "output_gate", rows)?,
            kda_out: take_rows(&mut b, "kda_out", rows)?,
            query_a: take_rows(&mut b, "query_a", rows)?,
            query_r: take_rows(&mut b, "query_r", rows)?,
            query: take_rows(&mut b, "query", rows)?,
            latent_raw: take_rows(&mut b, "latent_raw", rows)?,
            latent: take_rows(&mut b, "latent", rows)?,
            query_latent: take_rows(&mut b, "query_latent", rows)?,
            output_latent: take_rows(&mut b, "output_latent", rows)?,
            heads_out: take_rows(&mut b, "heads_out", rows)?,
            index_key: take_rows(&mut b, "index_key", rows)?,
            index_gate: take_rows(&mut b, "index_gate", rows)?,
            dense_gate: take_rows(&mut b, "dense_gate", rows)?,
            dense_up: take_rows(&mut b, "dense_up", rows)?,
            router: take_rows(&mut b, "router", rows)?,
            counts: b.take("counts")?,
            slots: b.take("slots")?,
            inner: b.take("inner")?,
            slot_out: b.take("slot_out")?,
            routed: take_rows(&mut b, "routed", rows)?,
            shared_gate: take_rows(&mut b, "shared_gate", rows)?,
            shared_up: take_rows(&mut b, "shared_up", rows)?,
            shared: take_rows(&mut b, "shared", rows)?,
            routes: RouteRecord::for_blocks(ctx, c, Some(r))?,
            sparse: (capacity >= c.sparse_frontier() as usize)
                .then(|| PackedSparseScratch::new(ctx, c, capacity as u64, r))
                .transpose()?,
        };
        b.finish()?;
        Ok(scratch)
    }
}

/// Prefix view with the leading dimensions of `t` and `rows` in the last.
fn rows_view(t: &MetalTensor, rows: usize) -> MetalTensor {
    let mut shape = t.shape.clone();
    let last = shape.len() - 1;
    shape[last] = rows as u64;
    t.view_subrange(0, shape)
}

/// One-dimensional prefix view of `n` elements.
fn flat(t: &MetalTensor, n: usize) -> MetalTensor {
    t.view_subrange(0, vec![n as u64])
}

#[allow(clippy::too_many_arguments)]
fn matmat(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    arith: Arith,
    stage: Stage,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    rows: usize,
) -> Result<()> {
    let mode = stage_mode(arith, stage);
    if mode == StageMode::FastF32 {
        // F32-operand tiles at every row count (one row included), so a
        // token's outputs do not depend on its chunking.
        // Short spans take the narrow kernels, whose per-token outputs are
        // bitwise the wide tiles'.
        let narrow = rows <= f32_narrow_max_rows();
        let f32_tile = match weight.dtype {
            GgmlType::Q8_0
                if f32_tile_enabled(F32Tile::Q8_0)
                    && n_in.is_multiple_of(64)
                    && n_out.is_multiple_of(16) =>
            {
                // Activation buffers are allocated with rows padded to 32
                // (`memory::packed_activation_rows`), the backing these
                // tiles read.
                if narrow {
                    crate::metal::encode_mat_mat_q8_0_f32_r2c1k64(
                        ctx, enc, weight, x, y, n_in, n_out, rows,
                    )?;
                } else {
                    crate::metal::encode_mat_mat_q8_0_f32_r2c4k64(
                        ctx, enc, weight, x, y, n_in, n_out, rows,
                    )?;
                }
                true
            }
            GgmlType::Q6_K if f32_tile_enabled(F32Tile::Q6K) && n_in.is_multiple_of(256) => {
                if narrow {
                    crate::metal::encode_mat_mat_q6_k_f32_r8c8(
                        ctx, enc, weight, x, y, n_in, n_out, rows,
                    )?;
                } else {
                    crate::metal::encode_mat_mat_q6_k_f32_mm64x32(
                        ctx, enc, weight, x, y, n_in, n_out, rows,
                    )?;
                }
                true
            }
            _ => false,
        };
        if f32_tile {
            #[cfg(test)]
            record_census(
                stage,
                format!("{:?}", weight.dtype),
                if narrow {
                    CensusPath::F32Narrow
                } else {
                    CensusPath::F32Tile
                },
            );
            return Ok(());
        }
    }
    if mode == StageMode::Exact {
        #[cfg(test)]
        let rounded = if weight.dtype != GgmlType::F32 {
            rounded_input(ctx, enc, x, rows * n_in)?
        } else {
            None
        };
        #[cfg(test)]
        let x = rounded.as_ref().unwrap_or(x);
        for row in 0..rows {
            let xr = x.view_subrange((row * n_in) as u64, vec![n_in as u64]);
            let yr = y.view_subrange((row * n_out) as u64, vec![n_out as u64]);
            crate::metal_forward::encode_mat_vec_dispatch(ctx, enc, weight, &xr, &yr, n_in, n_out)?;
        }
        return Ok(());
    }
    #[cfg(test)]
    record_census(
        stage,
        format!("{:?}", weight.dtype),
        if weight.dtype == GgmlType::F32 {
            CensusPath::F32Native
        } else {
            CensusPath::Half
        },
    );
    crate::metal_forward::encode_mat_mat_dispatch_with_policy(
        ctx, enc, weight, x, y, n_in, n_out, rows, true,
    )?;
    Ok(())
}

/// KDA low-rank expansion over `rows` rows: the decode kernel
/// ([`super::low_rank_expand`]) for every row in exact lineage, batched
/// mat-mat in fast lineage.
#[allow(clippy::too_many_arguments)]
fn expand_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    arith: Arith,
    stage: Stage,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    rows: usize,
) -> Result<()> {
    match stage_mode(arith, stage) {
        StageMode::Exact => {
            #[cfg(test)]
            let rounded = rounded_input(ctx, enc, x, rows * n_in)?;
            #[cfg(test)]
            let x = rounded.as_ref().unwrap_or(x);
            super::low_rank_expand(ctx, enc, weight, x, y, n_in, n_out, rows)
        }
        StageMode::Fast | StageMode::FastF32 => {
            matmat(ctx, enc, arith, stage, weight, x, y, n_in, n_out, rows)
        }
    }
}

/// Per-head latent absorption or expansion over `rows` token-major rows:
/// grouped Q8_0 mat-mat (F32 accumulation) over whole 128-row blocks in fast
/// lineage, and the decode grouped GEMV per remaining row (every row in exact
/// lineage).
#[allow(clippy::too_many_arguments)]
fn absorb_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    lineage: PackedLineage,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    groups: usize,
    rows: usize,
) -> Result<()> {
    let (in_width, out_width) = (n_in * groups, n_out * groups);
    let blocked = match lineage {
        PackedLineage::Fast => rows / 128 * 128,
        PackedLineage::Exact => 0,
    };
    if blocked > 0 {
        crate::metal::encode_mat_mat_q8_0_grouped_f32(
            ctx,
            enc,
            weight,
            &input.view_subrange(0, vec![in_width as u64, blocked as u64]),
            &output.view_subrange(0, vec![out_width as u64, blocked as u64]),
            n_in,
            n_out,
            groups,
            blocked,
        )?;
    }
    let tail_end = blocked;
    #[cfg(test)]
    let tail_end = {
        let tail_rows = absorb_tail_rows(lineage, rows);
        if tail_rows > 0 {
            crate::metal::encode_mat_mat_q8_0_grouped_tail_f32(
                ctx,
                enc,
                weight,
                &input.view_subrange(
                    (blocked * in_width) as u64,
                    vec![in_width as u64, tail_rows as u64],
                ),
                &output.view_subrange(
                    (blocked * out_width) as u64,
                    vec![out_width as u64, tail_rows as u64],
                ),
                n_in,
                n_out,
                groups,
                tail_rows,
            )?;
        }
        tail_end + tail_rows
    };
    for row in tail_end..rows {
        let x = input.view_subrange((row * in_width) as u64, vec![in_width as u64]);
        let y = output.view_subrange((row * out_width) as u64, vec![out_width as u64]);
        encode_mat_vec_q8_0_grouped_f32(ctx, enc, weight, &x, &y, n_in, n_out, groups)?;
    }
    #[cfg(test)]
    if tail_end > blocked {
        ABSORB_TAIL.with(|state| state.set((state.get().0, state.get().1 + 1)));
    }
    Ok(())
}

impl Glm5NextSession<'_> {
    /// Packed prefill of `tokens` in chunks of the session's prefill rows;
    /// returns the last token's logits. Falls back to serial decode when the
    /// session has no packed scratch. Rows below the sparse frontier attend
    /// densely; rows at or past it run sparse selection in microbatches.
    pub fn prefill_packed(&mut self, ctx: &MetalContext, tokens: &[u32]) -> Result<Vec<f32>> {
        self.prefill_packed_with_checkpoint(ctx, tokens, &mut || Ok(()))
    }

    /// [`Self::prefill_packed`] that calls `checkpoint` before every chunk
    /// (every token without packed scratch). When it returns an error, the
    /// prefill stops with [`Glm5NextMetalError::Cancelled`] at a chunk
    /// boundary: the completed chunks stay committed, the session is not
    /// poisoned, and its position is the end of the last completed chunk.
    pub fn prefill_packed_with_checkpoint(
        &mut self,
        ctx: &MetalContext,
        tokens: &[u32],
        checkpoint: &mut dyn FnMut() -> std::result::Result<(), String>,
    ) -> Result<Vec<f32>> {
        self.validate_request(tokens)?;
        let mut check = || checkpoint().map_err(Glm5NextMetalError::Cancelled);
        let Some(rows) = self.packed.as_ref().map(|p| p.rows) else {
            let (&last, head) = tokens.split_last().expect("validated nonempty");
            for &token in head {
                check()?;
                self.advance(ctx, token)?;
            }
            check()?;
            return self.forward(ctx, last);
        };
        let chunks: Vec<&[u32]> = tokens.chunks(rows).collect();
        let last = chunks.len() - 1;
        let mut logits = None;
        for (index, chunk) in chunks.into_iter().enumerate() {
            check()?;
            logits = self.step_packed(ctx, chunk, index == last)?;
        }
        logits.ok_or_else(|| Glm5NextMetalError::Invalid("missing prefill logits".into()))
    }

    /// Packed sparse selector statuses `[MLA block][chunk row]` (tests).
    #[cfg(test)]
    pub(super) fn packed_sparse_statuses(&self) -> Result<(Vec<i32>, usize)> {
        let sp = self
            .packed
            .as_ref()
            .and_then(|p| p.sparse.as_ref())
            .ok_or_else(|| Glm5NextMetalError::Invalid("no packed sparse scratch".into()))?;
        Ok((read_i32(&sp.select_status)?, sp.rows))
    }

    /// Positions left in the dense range (visible length below the sparse
    /// frontier) from the current position.
    fn dense_rows_remaining(&self) -> usize {
        (self.weights.config.sparse_frontier() as usize - 1).saturating_sub(self.position)
    }

    fn step_packed(
        &mut self,
        ctx: &MetalContext,
        tokens: &[u32],
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        if self.poisoned {
            return Err(Glm5NextMetalError::Poisoned);
        }
        let c = &self.weights.config;
        if let Some(&bad) = tokens.iter().find(|&&t| t >= c.vocab_size) {
            return invalid(format!("token {bad} outside vocabulary"));
        }
        if self.position + tokens.len() > self.capacity {
            return invalid(format!(
                "positions {}..{} exceed capacity {}",
                self.position,
                self.position + tokens.len(),
                self.capacity
            ));
        }
        // Rows from `dense` on attend over the indexer's selection.
        let dense = self.dense_rows_remaining().min(tokens.len());
        let pool = c.indexer_pool as usize;
        let mla = c.block_count(MixerKind::Mla);
        if dense < tokens.len() {
            let packed = self.packed.as_ref().expect("packed scratch");
            let Some(sp) = packed.sparse.as_ref() else {
                return invalid("packed rows past the sparse frontier need packed sparse scratch");
            };
            let visible: Vec<i32> = (0..sp.rows)
                .map(|r| (self.position + r + 1).min(self.capacity) as i32)
                .collect();
            #[allow(unused_mut)]
            let mut pools: Vec<i32> = visible.iter().map(|&l| l / pool as i32).collect();
            #[cfg(test)]
            if let Some(row) = self.corrupt_sparse_row.take() {
                pools[row] = 0;
            }
            write_i32(&sp.visible_rows, &visible)?;
            write_i32(&sp.visible_pools, &pools)?;
            // Unwritten slots fail the check below.
            write_i32(&sp.select_status, &vec![-1; mla * sp.rows])?;
        }
        self.poisoned = true;
        self.encode_chunk(ctx, tokens, want_logits)?;
        let packed = self.packed.as_ref().expect("packed scratch");
        for (layer, route) in packed.routes.iter().enumerate() {
            if let Some(route) = route {
                let status = read_i32(&rows_view(&route.status, tokens.len()))?;
                if let Some((row, &bad)) = status
                    .iter()
                    .enumerate()
                    .find(|(_, s)| **s != ROUTE_STATUS_READY)
                {
                    return Err(Glm5NextMetalError::KernelValidation {
                        stage: "packed route",
                        block: layer,
                        row: Some(row),
                        status: bad,
                    });
                }
            }
        }
        if dense < tokens.len() {
            let sp = packed.sparse.as_ref().expect("checked before encoding");
            let status = read_i32(&sp.select_status)?;
            for block in 0..mla {
                for row in dense..tokens.len() {
                    let code = status[block * sp.rows + row];
                    if code != crate::metal::SELECT_STATUS_OK {
                        return Err(Glm5NextMetalError::KernelValidation {
                            stage: "packed sparse selection",
                            block: super::mla_block(&self.weights.config, block),
                            row: Some(row),
                            status: code,
                        });
                    }
                }
            }
        }
        let logits = want_logits.then(|| read_f32(&self.s.logits)).transpose()?;
        self.position += tokens.len();
        self.poisoned = false;
        Ok(logits)
    }

    fn encode_chunk(&self, ctx: &MetalContext, tokens: &[u32], want_logits: bool) -> Result<()> {
        let w = self.weights;
        let c = &w.config;
        let p = self.packed.as_ref().expect("packed scratch");
        let rows = tokens.len();
        let h = c.hidden_size as usize;
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        // SAFETY: shared-storage I32 [max rows]; no command is in flight.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ids.as_ptr(),
                p.token
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(p.token.offset as usize)
                    .cast::<i32>(),
                rows,
            );
        }
        let v = |t: &MetalTensor| rows_view(t, rows);
        let command = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| Glm5NextMetalError::Invalid("no command buffer".into()))?;
        let enc = KernelEncoder::begin(&command);
        encode_get_rows_f32(
            ctx,
            &enc,
            &w.embedding,
            &v(&p.token),
            &v(&p.embedding),
            rows,
            h,
        )?;
        encode_mhc4_repeat_rows(ctx, &enc, h, rows, &v(&p.embedding), &v(&p.residual[0]))?;
        let mut mla_index = 0;
        for (index, block) in w.blocks.iter().enumerate() {
            let (a, b) = (v(&p.residual[0]), v(&p.residual[1]));
            self.encode_hc_pre_rows(
                ctx,
                &enc,
                rows,
                &a,
                &block.attention_hc,
                &block.attention_norm,
            )?;
            match (&block.mixer, &self.layers[index]) {
                (MixerTensors::Kda(kda), LayerState::Kda { conv, state }) => {
                    self.encode_kda_rows(ctx, &enc, rows, kda, conv, state)?
                }
                (
                    MixerTensors::Mla(mla),
                    LayerState::Mla {
                        latent,
                        pending,
                        pooled,
                    },
                ) => {
                    self.encode_mla_rows(ctx, &enc, rows, mla, latent, pending, pooled, mla_index)?;
                    mla_index += 1;
                }
                _ => return invalid(format!("block {index} state does not match its mixer")),
            }
            encode_mhc4_post_rows(
                ctx,
                &enc,
                h,
                rows,
                &v(&p.block_out),
                &a,
                &v(&p.post),
                &v(&p.comb),
                &b,
            )?;
            self.encode_hc_pre_rows(ctx, &enc, rows, &b, &block.ffn_hc, &block.ffn_norm)?;
            match &block.ffn {
                FfnTensors::Dense(dense) => {
                    let f = c.dense_ffn_size as usize;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::DenseFfn,
                        &dense.gate,
                        &v(&p.normed),
                        &v(&p.dense_gate),
                        h,
                        f,
                        rows,
                    )?;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::DenseFfn,
                        &dense.up,
                        &v(&p.normed),
                        &v(&p.dense_up),
                        h,
                        f,
                        rows,
                    )?;
                    let (g, u) = (flat(&p.dense_gate, f * rows), flat(&p.dense_up, f * rows));
                    encode_clamped_swiglu(ctx, &enc, &g, &u, &g, c.swiglu_clamp)?;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::DenseFfn,
                        &dense.down,
                        &v(&p.dense_gate),
                        &v(&p.block_out),
                        f,
                        h,
                        rows,
                    )?;
                }
                FfnTensors::Moe(moe) => {
                    let route = p.routes[index].as_ref().ok_or_else(|| {
                        Glm5NextMetalError::Invalid(format!(
                            "block {index} has no packed route record"
                        ))
                    })?;
                    let (e, f, k) = (
                        c.expert_count as usize,
                        c.expert_ffn_size as usize,
                        c.expert_used_count as usize,
                    );
                    if router_e8p32_selected(
                        stage_lineage(p.lineage, Stage::Router),
                        h,
                        e,
                        moe.router.dtype,
                        rows,
                        || router_e8p32_supported(ctx),
                    ) {
                        crate::metal::encode_mat_mat_f32_router_e8p32_strict(
                            ctx,
                            &enc,
                            &moe.router,
                            &v(&p.normed),
                            &v(&p.router),
                            h,
                            e,
                            rows,
                        )?;
                        #[cfg(test)]
                        router_prefill::record_substitution();
                    } else {
                        matmat(
                            ctx,
                            &enc,
                            p.arith(),
                            Stage::Router,
                            &moe.router,
                            &v(&p.normed),
                            &v(&p.router),
                            h,
                            e,
                            rows,
                        )?;
                    }
                    let spec = LearnedRoute {
                        experts: e,
                        top_k: k,
                        score: RouteScore::Sigmoid,
                        routed_scale: c.expert_weights_scale,
                    };
                    let (route_ids, route_weights) = (v(&route.ids), v(&route.weights));
                    encode_route_learned_rows(
                        ctx,
                        &enc,
                        &spec,
                        rows,
                        &v(&p.router),
                        &moe.selection_bias,
                        &route_ids,
                        &route_weights,
                        &v(&route.status),
                    )?;
                    if stage_lineage(p.lineage, Stage::RoutedExperts) == PackedLineage::Exact {
                        let s = &self.s;
                        #[cfg(test)]
                        let rounded = rounded_input(ctx, &enc, &v(&p.normed), rows * h)?;
                        #[cfg(test)]
                        let normed = rounded.as_ref().unwrap_or(&p.normed);
                        #[cfg(not(test))]
                        let normed = &p.normed;
                        for row in 0..rows {
                            let x = normed.view_subrange((row * h) as u64, vec![h as u64]);
                            let ids_r = route.ids.view_subrange((row * k) as u64, vec![k as u64]);
                            let w_r = route
                                .weights
                                .view_subrange((row * k) as u64, vec![k as u64]);
                            let st = route.status.view_subrange(row as u64, vec![1]);
                            let out = p.routed.view_subrange((row * h) as u64, vec![h as u64]);
                            crate::metal::encode_all_slots_gate_up_swiglu(
                                ctx,
                                &enc,
                                &moe.gate_experts,
                                &moe.up_experts,
                                &x,
                                &ids_r,
                                &st,
                                &s.expert_inner,
                                h,
                                f,
                                e,
                                k,
                                c.swiglu_clamp,
                            )?;
                            #[cfg(test)]
                            if ROUND_EXACT_ACTIVATIONS.with(std::cell::Cell::get) {
                                crate::metal::encode_round_trip_f16_f32(
                                    ctx,
                                    &enc,
                                    &s.expert_inner,
                                    f * k,
                                )?;
                            }
                            crate::metal::encode_all_slots_down(
                                ctx,
                                &enc,
                                &moe.down_experts,
                                &s.expert_inner,
                                &ids_r,
                                &st,
                                &s.expert_out,
                                f,
                                h,
                                e,
                                k,
                            )?;
                            encode_moe_weighted_sum_f32(
                                ctx,
                                &enc,
                                &s.expert_out,
                                &w_r,
                                &out,
                                h,
                                k,
                            )?;
                        }
                    } else {
                        let grouped = GroupedExperts {
                            gate_bank: &moe.gate_experts,
                            up_bank: &moe.up_experts,
                            down_bank: &moe.down_experts,
                            input: &v(&p.normed),
                            ids: &route_ids,
                            weights: &route_weights,
                            counts: &p.counts,
                            slots: &flat(&p.slots, e * rows),
                            inner: &p.inner.view_subrange(0, vec![f as u64, (k * rows) as u64]),
                            slot_out: &p
                                .slot_out
                                .view_subrange(0, vec![h as u64, (k * rows) as u64]),
                            output: &v(&p.routed),
                        };
                        let f32_mode =
                            stage_mode(p.arith(), Stage::RoutedExperts) == StageMode::FastF32;
                        let f32_taken = f32_mode
                            && f32_tile_enabled(F32Tile::RoutedExperts)
                            && moe.gate_experts.dtype == moe.up_experts.dtype
                            && matches!(moe.gate_experts.dtype, GgmlType::IQ2_S | GgmlType::IQ3_S)
                            && matches!(moe.down_experts.dtype, GgmlType::IQ3_S | GgmlType::IQ4_XS);
                        #[cfg(test)]
                        record_census(
                            Stage::RoutedExperts,
                            "experts".into(),
                            if f32_taken {
                                CensusPath::F32Tile
                            } else {
                                CensusPath::Half
                            },
                        );
                        if f32_taken {
                            encode_grouped_routed_experts_f32x(
                                ctx,
                                &enc,
                                &grouped,
                                h,
                                f,
                                e,
                                k,
                                rows,
                                c.swiglu_clamp,
                            )?;
                        } else {
                            encode_grouped_routed_experts_with_down_policy(
                                ctx,
                                &enc,
                                &grouped,
                                h,
                                f,
                                e,
                                k,
                                rows,
                                c.swiglu_clamp,
                                grouped_down_policy(
                                    p.lineage,
                                    moe.down_experts.dtype,
                                    f,
                                    h,
                                    e,
                                    rows,
                                ),
                            )?;
                            #[cfg(test)]
                            expert_down::after_grouped(
                                ctx,
                                &enc,
                                index,
                                rows,
                                moe.down_experts.dtype,
                                p,
                            )?;
                        }
                    }
                    let sf = c.shared_expert_ffn_size as usize;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::SharedExpert,
                        &moe.shared.gate,
                        &v(&p.normed),
                        &v(&p.shared_gate),
                        h,
                        sf,
                        rows,
                    )?;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::SharedExpert,
                        &moe.shared.up,
                        &v(&p.normed),
                        &v(&p.shared_up),
                        h,
                        sf,
                        rows,
                    )?;
                    let (g, u) = (
                        flat(&p.shared_gate, sf * rows),
                        flat(&p.shared_up, sf * rows),
                    );
                    encode_clamped_swiglu(ctx, &enc, &g, &u, &g, c.swiglu_clamp)?;
                    matmat(
                        ctx,
                        &enc,
                        p.arith(),
                        Stage::SharedExpert,
                        &moe.shared.down,
                        &v(&p.shared_gate),
                        &v(&p.shared),
                        sf,
                        h,
                        rows,
                    )?;
                    encode_add_f32(
                        ctx,
                        &enc,
                        &flat(&p.routed, h * rows),
                        &flat(&p.shared, h * rows),
                        &flat(&p.block_out, h * rows),
                    )?;
                }
            }
            encode_mhc4_post_rows(
                ctx,
                &enc,
                h,
                rows,
                &v(&p.block_out),
                &b,
                &v(&p.post),
                &v(&p.comb),
                &a,
            )?;
        }
        if want_logits {
            // Head on the last row only.
            let s = &self.s;
            let last = p.residual[0].view_subrange(((rows - 1) * h * 4) as u64, vec![h as u64, 4]);
            encode_mhc4_collapse(ctx, &enc, h, &last, &s.quarter, &s.final_hidden)?;
            encode_rms_norm_mul_f32(
                ctx,
                &enc,
                &s.final_hidden,
                &w.output_norm,
                &s.final_normed,
                c.rms_epsilon,
            )?;
            super::matvec(
                ctx,
                &enc,
                &w.output,
                &s.final_normed,
                &s.logits,
                h,
                c.vocab_size as usize,
            )?;
        }
        enc.end();
        command.commit();
        wait_completed(&command)?;
        #[cfg(test)]
        router_prefill::record_completed_command_gpu_time(|| {
            (command.GPUStartTime(), command.GPUEndTime())
        });
        #[cfg(test)]
        expert_down::record_completed_command(|| (command.GPUStartTime(), command.GPUEndTime()));
        #[cfg(test)]
        mla_tail::record_completed_command(|| (command.GPUStartTime(), command.GPUEndTime()));
        Ok(())
    }

    /// [`Glm5NextSession::encode_hc_pre`] for `rows` packed tokens: the same
    /// fused kernels with a rows axis (each row independent), so packed
    /// prefill keeps decode's mHC lineage in both lineages.
    fn encode_hc_pre_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        rows: usize,
        residual: &MetalTensor,
        hc: &crate::glm5_next::HyperConnectionTensors<MetalTensor>,
        norm_weight: &MetalTensor,
    ) -> Result<()> {
        let c = &self.weights.config;
        let p = self.packed.as_ref().expect("packed scratch");
        let v = |t: &MetalTensor| rows_view(t, rows);
        crate::metal::encode_mhc4_pre_q8_0(
            ctx,
            enc,
            c.hidden_size as usize,
            rows,
            super::hc_pre_eps(c),
            &crate::metal::Mhc4PreInputs {
                residual,
                mix: &hc.mix,
                scale: &hc.scale,
                base: &hc.base,
                norm_weight,
            },
            &crate::metal::Mhc4PrePartials {
                dots: &v(&p.hc_partial_dots),
                sumsq: &v(&p.hc_partial_sumsq),
            },
            &crate::metal::Mhc4PreOutputs {
                mixes: &v(&p.mixes),
                pre: &v(&p.pre),
                post: &v(&p.post),
                comb: &v(&p.comb),
                collapsed: &v(&p.collapsed),
                normed: &v(&p.normed),
            },
        )?;
        Ok(())
    }

    fn encode_kda_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        rows: usize,
        kda: &crate::glm5_next::KdaTensors<MetalTensor>,
        conv: &MetalTensor,
        state: &MetalTensor,
    ) -> Result<()> {
        let c = &self.weights.config;
        let p = self.packed.as_ref().expect("packed scratch");
        let v = |t: &MetalTensor| rows_view(t, rows);
        let (h, width, rank) = (
            c.hidden_size as usize,
            c.kda_width() as usize,
            c.kda_head_dim as usize,
        );
        let x = v(&p.normed);
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::KdaProjection,
            &kda.query,
            &x,
            &v(&p.q),
            h,
            width,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::KdaProjection,
            &kda.key,
            &x,
            &v(&p.k),
            h,
            width,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::KdaProjection,
            &kda.value,
            &x,
            &v(&p.v),
            h,
            width,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::KdaProjection,
            &kda.decay_a,
            &x,
            &v(&p.rank_a),
            h,
            rank,
            rows,
        )?;
        expand_rows(
            ctx,
            enc,
            p.arith(),
            Stage::KdaExpand,
            &kda.decay_b,
            &v(&p.rank_a),
            &v(&p.raw_gate),
            rank,
            width,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::KdaProjection,
            &kda.beta,
            &x,
            &v(&p.raw_beta),
            h,
            c.head_count as usize,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::KdaProjection,
            &kda.gate_a,
            &x,
            &v(&p.rank_b),
            h,
            rank,
            rows,
        )?;
        expand_rows(
            ctx,
            enc,
            p.arith(),
            Stage::KdaExpand,
            &kda.gate_b,
            &v(&p.rank_b),
            &v(&p.output_gate),
            rank,
            width,
            rows,
        )?;
        encode_kda_prefill(
            ctx,
            enc,
            c.head_count as usize,
            rows,
            &KdaDecode {
                q: &v(&p.q),
                k: &v(&p.k),
                v: &v(&p.v),
                raw_gate: &v(&p.raw_gate),
                raw_beta: &v(&p.raw_beta),
                output_gate: &v(&p.output_gate),
                q_conv: &kda.query_conv,
                k_conv: &kda.key_conv,
                v_conv: &kda.value_conv,
                neg_exp_a_log: &kda.neg_exp_a_log,
                dt_bias: &kda.decay_bias,
                output_norm: &kda.output_norm,
                conv_state: conv,
                state,
                out: &v(&p.kda_out),
            },
            c.kda_gate_lower_bound,
            c.rms_epsilon,
        )?;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::KdaProjection,
            &kda.output,
            &v(&p.kda_out),
            &v(&p.block_out),
            width,
            h,
            rows,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_mla_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        rows: usize,
        mla: &crate::glm5_next::MlaTensors<MetalTensor>,
        latent: &MetalTensor,
        pending: &MetalTensor,
        pooled: &MetalTensor,
        mla_index: usize,
    ) -> Result<()> {
        let c = &self.weights.config;
        let p = self.packed.as_ref().expect("packed scratch");
        let v = |t: &MetalTensor| rows_view(t, rows);
        let h = c.hidden_size as usize;
        let (q_rank, kv) = (c.q_lora_rank as usize, c.kv_lora_rank as usize);
        let heads = c.head_count as usize;
        let head_dim = c.mla_key_head_dim as usize;
        let x = v(&p.normed);
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::MlaProjection,
            &mla.query_a,
            &x,
            &v(&p.query_a),
            h,
            q_rank,
            rows,
        )?;
        encode_rms_norm_mul_rows_f32(
            ctx,
            enc,
            &v(&p.query_a),
            &mla.query_a_norm,
            &v(&p.query_r),
            rows,
            q_rank,
            c.rms_epsilon,
        )?;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::MlaProjection,
            &mla.query_b,
            &v(&p.query_r),
            &v(&p.query),
            q_rank,
            heads * head_dim,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::MlaProjection,
            &mla.latent,
            &x,
            &v(&p.latent_raw),
            h,
            kv,
            rows,
        )?;
        encode_rms_norm_mul_rows_f32(
            ctx,
            enc,
            &v(&p.latent_raw),
            &mla.latent_norm,
            &v(&p.latent),
            rows,
            kv,
            c.rms_epsilon,
        )?;
        encode_scatter_offset_f32_to_f16(
            ctx,
            enc,
            &flat(&p.latent, kv * rows),
            latent,
            self.position * kv,
            kv * rows,
        )?;
        absorb_rows(
            ctx,
            enc,
            stage_lineage(p.lineage, Stage::MlaAbsorb),
            &mla.key_absorb,
            &p.query,
            &p.query_latent,
            head_dim,
            kv,
            heads,
            rows,
        )?;
        // Indexer cache maintenance first: every pool the chunk completes is
        // published before attention, and each row scores only its own
        // visible prefix (the first L / 4 pools).
        let index_dim = c.indexer_head_dim as usize;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::IndexerProjection,
            &mla.indexer.key,
            &x,
            &v(&p.index_key),
            h,
            index_dim,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::IndexerProjection,
            &mla.indexer.pool_gate,
            &x,
            &v(&p.index_gate),
            h,
            index_dim,
            rows,
        )?;
        encode_indexer_append_rows(
            ctx,
            enc,
            &v(&p.index_key),
            &v(&p.index_gate),
            &mla.indexer.key_norm,
            &mla.indexer.key_norm_bias,
            &mla.indexer.pool_position,
            pending,
            pooled,
            self.position,
            rows,
            c.layer_norm_epsilon,
        )?;
        // Rows before the frontier attend densely; the rest attend over the
        // indexer's selection.
        let scale = 1.0 / (head_dim as f32).sqrt();
        let dense = self.dense_rows_remaining().min(rows);
        if dense > 0 {
            encode_latent_attention(
                ctx,
                enc,
                &rows_view(&p.query_latent, dense),
                latent,
                &self.s.no_sink,
                &rows_view(&p.output_latent, dense),
                self.position,
                dense,
                scale,
            )?;
        }
        if dense < rows {
            self.encode_sparse_rows(ctx, enc, mla, latent, pooled, mla_index, dense, rows, scale)?;
        }
        absorb_rows(
            ctx,
            enc,
            stage_lineage(p.lineage, Stage::MlaAbsorb),
            &mla.value_expand,
            &p.output_latent,
            &p.heads_out,
            kv,
            head_dim,
            heads,
            rows,
        )?;
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::MlaProjection,
            &mla.output,
            &v(&p.heads_out),
            &v(&p.block_out),
            heads * head_dim,
            h,
            rows,
        )?;
        Ok(())
    }

    /// Sparse attention for chunk rows `first..rows` (all at or past the
    /// frontier): chunk-wide indexer queries (F16-rounded) and head weights
    /// scaled by 1 / sqrt(heads * dim), then per microbatch of up to
    /// [`memory::PACKED_SPARSE_QUERIES`] rows: scores over each row's visible
    /// pools, exact top-512 into the (block, row) status slots, expansion
    /// and online attention over exactly the selected latent rows. The same
    /// kernels as decode, so per-row results match serial decode under the
    /// exact lineage.
    #[allow(clippy::too_many_arguments)]
    fn encode_sparse_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        mla: &crate::glm5_next::MlaTensors<MetalTensor>,
        latent: &MetalTensor,
        pooled: &MetalTensor,
        mla_index: usize,
        first: usize,
        rows: usize,
        scale: f32,
    ) -> Result<()> {
        let c = &self.weights.config;
        let p = self.packed.as_ref().expect("packed scratch");
        let sp = p.sparse.as_ref().ok_or_else(|| {
            Glm5NextMetalError::Invalid("packed sparse rows without packed sparse scratch".into())
        })?;
        let (h, q_rank) = (c.hidden_size as usize, c.q_lora_rank as usize);
        let (ih, id) = (c.indexer_head_count as usize, c.indexer_head_dim as usize);
        let (heads, kv) = (c.head_count as usize, c.kv_lora_rank as usize);
        let query_width = ih * id;
        let count = rows - first;
        // Projections over the sparse rows (row-wise in the exact lineage).
        let sub = |t: &MetalTensor, width: usize| {
            t.view_subrange((first * width) as u64, vec![width as u64, count as u64])
        };
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::IndexerProjection,
            &mla.indexer.query,
            &sub(&p.query_r, q_rank),
            &sub(&sp.index_query, query_width),
            q_rank,
            query_width,
            count,
        )?;
        encode_scatter_offset_f32_to_f16(
            ctx,
            enc,
            &sub(&sp.index_query, query_width).view_subrange(0, vec![(query_width * count) as u64]),
            &sp.index_query_f16,
            first * query_width,
            query_width * count,
        )?;
        let weights = sub(&sp.index_weights, ih);
        matmat(
            ctx,
            enc,
            p.arith(),
            Stage::IndexerProjection,
            &mla.indexer.head_weights,
            &sub(&p.normed, h),
            &weights,
            h,
            ih,
            count,
        )?;
        crate::metal::encode_scale_f32_in_place(
            ctx,
            enc,
            &weights,
            1.0 / (query_width as f32).sqrt(),
        )?;
        let pool_capacity = pooled.shape[1] as usize;
        let top_pools = c.selected_pool_count() as usize;
        let row_slots = c.selection_width() as usize;
        let pool = c.indexer_pool as usize;
        let flat = |t: &MetalTensor| t.view_subrange(0, vec![(kv * heads) as u64, rows as u64]);
        let mut start = first;
        while start < rows {
            let n = sp.microbatch.min(rows - start);
            let i32_rows = |t: &MetalTensor| t.view_subrange(start as u64, vec![n as u64]);
            let cols =
                |t: &MetalTensor, width: usize| t.view_subrange(0, vec![width as u64, n as u64]);
            // Visibility grows with the row: the last row bounds the batch.
            let max_pools = (self.position + start + n) / pool;
            let queries_f16 = sp.index_query_f16.view_subrange(
                (start * query_width) as u64,
                vec![id as u64, ih as u64, n as u64],
            );
            let head_weights = sp
                .index_weights
                .view_subrange((start * ih) as u64, vec![ih as u64, n as u64]);
            let (visible_pools, visible_rows) =
                (i32_rows(&sp.visible_pools), i32_rows(&sp.visible_rows));
            let scores = cols(&sp.scores, pool_capacity);
            let (pool_ids, pool_counts) = (
                cols(&sp.pool_ids, top_pools),
                sp.pool_counts.view_subrange(0, vec![n as u64]),
            );
            let (row_ids, row_counts) = (
                cols(&sp.row_ids, row_slots),
                sp.row_counts.view_subrange(0, vec![n as u64]),
            );
            let status = sp
                .select_status
                .view_subrange((mla_index * sp.rows + start) as u64, vec![n as u64]);
            crate::metal::encode_lightning_scores_f16_matrix(
                ctx,
                enc,
                &crate::metal::LightningScores {
                    queries: &queries_f16,
                    head_weights: &head_weights,
                    keys: pooled,
                    visible_counts: &visible_pools,
                    scores: &scores,
                },
                ih,
                id,
                pool_capacity,
                max_pools,
                n,
            )?;
            crate::metal::encode_select_top_k_ids(
                ctx,
                enc,
                &crate::metal::TopKSelection {
                    scores: &scores,
                    visible_counts: &visible_pools,
                    ids: &pool_ids,
                    counts: &pool_counts,
                    status: &status,
                },
                pool_capacity,
                max_pools,
                top_pools,
                n,
            )?;
            crate::metal::encode_indexer_expand_selection(
                ctx,
                enc,
                &crate::metal::IndexerSelection {
                    pool_ids: &pool_ids,
                    pool_counts: &pool_counts,
                    visible_rows: &visible_rows,
                    row_ids: &row_ids,
                    row_counts: &row_counts,
                },
                top_pools,
                row_slots,
                n,
            )?;
            // Split selected attention in query sub-batches: the partition is
            // per query, so the sub-batch size never changes a result and
            // packed rows equal serial decode's (Exact lineage) bitwise.
            let split_queries = memory::PACKED_SPLIT_QUERIES as usize;
            let mut sub = 0;
            while sub < n {
                let m = split_queries.min(n - sub);
                let units = memory::split_attention_units(c, m as u64);
                let sub_rows = |t: &MetalTensor| t.view_subrange(sub as u64, vec![m as u64]);
                crate::metal::encode_online_selected_attention_split_f16(
                    ctx,
                    enc,
                    &crate::metal::SelectedAttention {
                        queries: &flat(&p.query_latent),
                        raw_cache: latent,
                        raw_cache_before_chunk: latent,
                        compressed_cache: latent,
                        selected_ids: &row_ids.view_subrange(
                            (sub * row_slots) as u64,
                            vec![row_slots as u64, m as u64],
                        ),
                        selected_counts: &sub_rows(&row_counts),
                        visible_counts: &sub_rows(&visible_rows),
                        sinks: &self.s.no_sink,
                        output: &flat(&p.output_latent),
                    },
                    crate::metal::SelectedAttentionShape {
                        head_count: heads,
                        query_count: m,
                        query_token_offset: start + sub,
                        token_count: rows,
                        chunk_start_position: self.position,
                        window: 0,
                        raw_cache_is_chunk: false,
                        selected_slots: row_slots,
                        compressed_capacity: latent.shape[1] as usize,
                        scale,
                        direct: true,
                    },
                    &crate::metal::SelectedAttentionPartials {
                        values: &sp
                            .attention_partials
                            .view_subrange(0, vec![kv as u64, units]),
                        stats: &sp.attention_partial_stats.view_subrange(0, vec![2, units]),
                    },
                )?;
                sub += m;
            }
            start += n;
        }
        Ok(())
    }
}

#[cfg(test)]
mod down_policy_tests {
    use super::*;

    #[test]
    fn iq3_s_down_policy_uses_actual_command_width_and_fast_lineage() {
        for rows in 0..=1024 {
            let expected = if (32..=512).contains(&rows) {
                GroupedDownPolicy::Iq3SSmallCounts
            } else {
                GroupedDownPolicy::Incumbent
            };
            assert_eq!(
                grouped_down_policy(PackedLineage::Fast, GgmlType::IQ3_S, 2048, 4096, 288, rows),
                expected,
                "rows={rows}"
            );
            assert_eq!(
                grouped_down_policy(PackedLineage::Exact, GgmlType::IQ3_S, 2048, 4096, 288, rows),
                GroupedDownPolicy::Incumbent,
                "Exact rows={rows}"
            );
        }
        for rows in [4096, usize::MAX] {
            assert_eq!(
                grouped_down_policy(PackedLineage::Fast, GgmlType::IQ3_S, 2048, 4096, 288, rows),
                GroupedDownPolicy::Incumbent
            );
        }
    }

    #[test]
    fn iq3_s_down_policy_preserves_other_geometries_and_dtypes() {
        for rows in [32, 64, 127, 128, 129, 256, 512] {
            for (ffn, hidden, experts) in [
                (2047, 4096, 288),
                (2049, 4096, 288),
                (2048, 4095, 288),
                (2048, 4097, 288),
                (2048, 4096, 287),
                (2048, 4096, 289),
            ] {
                assert_eq!(
                    grouped_down_policy(
                        PackedLineage::Fast,
                        GgmlType::IQ3_S,
                        ffn,
                        hidden,
                        experts,
                        rows
                    ),
                    GroupedDownPolicy::Incumbent
                );
            }
            for dtype in [
                GgmlType::F32,
                GgmlType::F16,
                GgmlType::IQ2_S,
                GgmlType::IQ3_XXS,
                GgmlType::IQ4_XS,
                GgmlType::Q2_0,
                GgmlType::Q4_K,
                GgmlType::Q8_0,
            ] {
                assert_eq!(
                    grouped_down_policy(PackedLineage::Fast, dtype, 2048, 4096, 288, rows),
                    GroupedDownPolicy::Incumbent
                );
            }
        }
    }
}

#[cfg(test)]
mod absorb_tail_policy_tests {
    use super::*;

    #[test]
    fn absorb_tail_selection_is_off_by_default_and_exact_and_keeps_full_prefix() {
        for variant in [None, Some(false), Some(true)] {
            let ((), count) = with_absorb_tail_variant(variant, || {
                for rows in 0..=1024 {
                    let tail = absorb_tail_rows(PackedLineage::Fast, rows);
                    assert_eq!(
                        tail,
                        if variant == Some(true) {
                            rows % 128 / 8 * 8
                        } else {
                            0
                        }
                    );
                    assert_eq!(absorb_tail_rows(PackedLineage::Exact, rows), 0);
                    if variant == Some(true) {
                        assert!(tail <= 120 && tail.is_multiple_of(8));
                        assert!((rows - rows / 128 * 128 - tail) < 8);
                    }
                }
            });
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn absorb_tail_dispatch_split_and_counter_preserve_prefix_and_residual() {
        use crate::metal::{dispatch_census_begin, dispatch_census_take};
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let offset_tensor = |ctx: &MetalContext,
                             prefix: usize,
                             data: &[u8],
                             suffix: usize,
                             shape: Vec<u64>,
                             dtype: GgmlType| {
            let mut bytes = vec![0xA5; prefix];
            bytes.extend_from_slice(data);
            bytes.resize(bytes.len() + suffix, 0x5A);
            MetalTensor {
                buffer: ctx.buffer_from(&bytes).unwrap(),
                offset: prefix as u64,
                shape,
                dtype,
                provenance: crate::metal::MetalTensorProvenance::OwnedWritable,
            }
        };
        let (k, m, groups) = (64usize, 16usize, 2usize);
        let mut bytes = vec![0u8; k / 32 * 34 * m * groups];
        for (i, block) in bytes.chunks_exact_mut(34).enumerate() {
            block[..2].copy_from_slice(&half::f16::from_f32(0.007).to_bits().to_le_bytes());
            for (j, q) in block[2..].iter_mut().enumerate() {
                *q = (((i * 13 + j * 7) % 63) as i8 - 31) as u8;
            }
        }
        let weight = offset_tensor(
            &ctx,
            16,
            &bytes,
            16,
            vec![k as u64, m as u64, groups as u64],
            GgmlType::Q8_0,
        );
        for rows in [7, 8, 15, 120, 127, 128, 135, 136, 143, 248, 255, 256] {
            let x: Vec<f32> = (0..k * groups * rows)
                .map(|i| (i % 37) as f32 * 0.013 - 0.17)
                .collect();
            let input = offset_tensor(
                &ctx,
                16,
                bytemuck::cast_slice(&x),
                16,
                vec![(k * groups) as u64, rows as u64],
                GgmlType::F32,
            );
            for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
                let mut baseline: Option<Vec<f32>> = None;
                for variant in [Some(false), None, Some(true)] {
                    let output = offset_tensor(
                        &ctx,
                        16,
                        bytemuck::cast_slice(&vec![-777.0f32; m * groups * rows]),
                        16,
                        vec![(m * groups) as u64, rows as u64],
                        GgmlType::F32,
                    );
                    let command = ctx.queue.commandBuffer().unwrap();
                    let enc = KernelEncoder::begin(&command);
                    dispatch_census_begin();
                    let (result, substitutions) = with_absorb_tail_variant(variant, || {
                        absorb_rows(
                            &ctx, &enc, lineage, &weight, &input, &output, k, m, groups, rows,
                        )
                    });
                    let census = dispatch_census_take();
                    enc.end();
                    result.unwrap();
                    command.commit();
                    wait_completed(&command).unwrap();
                    let blocked = if lineage == PackedLineage::Fast {
                        rows / 128 * 128
                    } else {
                        0
                    };
                    let tail = if lineage == PackedLineage::Fast && variant == Some(true) {
                        rows % 128 / 8 * 8
                    } else {
                        0
                    };
                    assert_eq!(substitutions, usize::from(tail > 0));
                    let full: Vec<_> = census
                        .iter()
                        .filter(|r| r.kernel == "kernel_mat_mat_q8_0_f32_r2c16k64_grouped")
                        .collect();
                    assert_eq!(full.len(), usize::from(blocked > 0));
                    if blocked > 0 {
                        assert_eq!(full[0].grid_width, (blocked / 128) as u64);
                    }
                    assert_eq!(
                        census
                            .iter()
                            .filter(|r| r.kernel == "kernel_mat_mat_q8_0_f32_r2c16k64_grouped_tail")
                            .count(),
                        usize::from(tail > 0)
                    );
                    assert_eq!(
                        census
                            .iter()
                            .filter(|r| r.kernel == "kernel_mat_vec_q8_0_f32_lcpp_grouped")
                            .count(),
                        rows - blocked - tail
                    );
                    assert_eq!(
                        census.len(),
                        usize::from(blocked > 0) + usize::from(tail > 0) + rows - blocked - tail
                    );
                    let actual = unsafe {
                        std::slice::from_raw_parts(
                            output
                                .buffer
                                .contents()
                                .as_ptr()
                                .cast::<u8>()
                                .add(output.offset as usize)
                                .cast::<f32>(),
                            m * groups * rows,
                        )
                        .to_vec()
                    };
                    assert!(actual.iter().all(|v| v.is_finite()));
                    if let Some(ref baseline) = baseline {
                        let peak = baseline.iter().fold(1.0f32, |peak, v| peak.max(v.abs()));
                        let worst = actual
                            .iter()
                            .zip(baseline)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0f32, f32::max);
                        assert!(
                            worst <= 1e-5 * peak,
                            "rows={rows}, {lineage:?}, {variant:?}: {worst}"
                        );
                    } else {
                        baseline = Some(actual);
                    }
                    let backing = unsafe {
                        std::slice::from_raw_parts(
                            output.buffer.contents().as_ptr().cast::<u8>(),
                            output.buffer.length(),
                        )
                    };
                    assert!(backing[..16].iter().all(|&b| b == 0xA5));
                    assert!(backing[backing.len() - 16..].iter().all(|&b| b == 0x5A));
                }
            }
        }
    }

    #[test]
    fn absorb_tail_scope_restores_nested_unwound_and_thread_state() {
        assert_eq!(ABSORB_TAIL.with(|s| s.get()), (None, 0));
        let ((), count) = with_absorb_tail_variant(Some(true), || {
            ABSORB_TAIL.with(|s| s.set((Some(true), 3)));
            for variant in [None, Some(false), Some(true)] {
                assert_eq!(
                    with_absorb_tail_variant(variant, || ABSORB_TAIL.with(|s| s.get())),
                    ((variant, 0), 0)
                );
            }
            let _ = std::panic::catch_unwind(|| {
                with_absorb_tail_variant(None, || panic!("scope restore probe"))
            });
            assert_eq!(ABSORB_TAIL.with(|s| s.get()), (Some(true), 3));
            assert_eq!(
                std::thread::spawn(|| ABSORB_TAIL.with(|s| s.get()))
                    .join()
                    .unwrap(),
                (None, 0)
            );
        });
        assert_eq!(count, 3);
        assert_eq!(ABSORB_TAIL.with(|s| s.get()), (None, 0));
    }
}
