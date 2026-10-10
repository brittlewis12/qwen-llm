//! Bounded, identity-qualified layer-zero attention HC diagnostic.
use super::*;
use std::cell::RefCell;

pub(crate) const COPY_TAG: &str = "frontier.hc.copy.";
pub(crate) const PROJECTION_TAG: &str = "frontier.hc.projection.";
pub(crate) const HYPER: usize = 10240;
pub(crate) const LOW: usize = 320;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Policy {
    Production,
    F32DownUp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    Down,
    Up,
}

impl Role {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Down => "down",
            Self::Up => "up",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Identity {
    buffer: usize,
    offset: u64,
    shape: [u64; 2],
    dtype: GgmlType,
}

fn identity(t: &MetalTensor) -> Option<Identity> {
    Some(Identity {
        buffer: Retained::as_ptr(&t.buffer) as *const () as usize,
        offset: t.offset,
        shape: t.shape.as_slice().try_into().ok()?,
        dtype: t.dtype,
    })
}

fn eligible(pair: &[Identity; 2]) -> bool {
    pair[0].dtype == GgmlType::BF16
        && pair[1].dtype == GgmlType::BF16
        && pair[0].shape == [HYPER as u64, LOW as u64]
        && pair[1].shape == [LOW as u64, HYPER as u64]
}

fn window(
    candidate: bool,
    command: usize,
    range: (usize, usize),
) -> Result<(usize, usize), Qwen4ExpMetalError> {
    match (candidate, command, range) {
        (false, 0, (2048, 3)) => Ok((0, 3)),
        (false, 1, (2051, 2045)) => Ok((3, 5)),
        (true, 0, (2048, 2048)) => Ok((0, 8)),
        _ => Err(invalid(format!(
            "unexpected frontier HC command {command}: {range:?}"
        ))),
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Record {
    pub(crate) start: usize,
    pub(crate) tokens: usize,
    pub(crate) role: Role,
    pub(crate) policy: Policy,
}

pub(crate) struct Probe {
    pair: [Identity; 2],
    pub(crate) buffers: Vec<(&'static str, MetalTensor)>,
    pub(crate) records: Vec<Record>,
    policy: Policy,
    candidate: bool,
    capture: bool,
}

impl Probe {
    pub(crate) fn specs() -> Vec<(&'static str, Vec<u64>)> {
        [
            ("hc.hyper", HYPER),
            ("hc.normalized", HYPER),
            ("hc.down", LOW),
            ("hc.low", LOW),
            ("hc.up", HYPER),
        ]
        .into_iter()
        .map(|(n, w)| (n, vec![w as u64, 8]))
        .collect()
    }

    /// All specs and CPU oracle retention must be admitted before this call.
    pub(crate) fn new(
        ctx: &MetalContext,
        weights: GatedResidualMetalReadWeights<'_>,
    ) -> Result<Self, Qwen4ExpMetalError> {
        let pair = [identity(weights.down), identity(weights.up)];
        let [Some(down), Some(up)] = pair else {
            return Err(invalid("HC probe requires rank2 weights"));
        };
        if !eligible(&[down, up]) {
            return Err(invalid("HC probe requires BF16 [10240,320]/[320,10240]"));
        }
        let buffers = Self::specs()
            .into_iter()
            .map(|(n, s)| Ok((n, MetalTensor::zeros_f32(ctx, s)?)))
            .collect::<Result<_, MetalError>>()?;
        Ok(Self {
            pair: [down, up],
            buffers,
            records: Vec::new(),
            policy: Policy::Production,
            candidate: false,
            capture: false,
        })
    }

    fn buffer(&self, name: &str) -> &MetalTensor {
        &self.buffers.iter().find(|(n, _)| *n == name).unwrap().1
    }
}

thread_local! { static ACTIVE: RefCell<Option<Probe>> = const { RefCell::new(None) }; }

struct Restore(Option<Probe>);
impl Drop for Restore {
    fn drop(&mut self) {
        ACTIVE.with(|p| {
            p.replace(self.0.take());
        });
    }
}

pub(crate) fn with_frontier_hc_probe<R>(
    mut probe: Probe,
    policy: Policy,
    candidate: bool,
    capture: bool,
    work: impl FnOnce() -> R,
) -> (R, Probe) {
    assert!(
        !qwen4exp_hc_packed_projection_override_active(),
        "overlapping broad HC override"
    );
    probe.policy = policy;
    probe.candidate = candidate;
    probe.capture = capture;
    probe.records.clear();
    let _restore = Restore(ACTIVE.with(|p| p.replace(Some(probe))));
    let result = work();
    (result, ACTIVE.with(|p| p.borrow_mut().take().unwrap()))
}

pub(super) struct Visit {
    start: usize,
    tokens: usize,
    row: usize,
    count: usize,
    policy: Policy,
}

fn copy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    probe: &Probe,
    name: &str,
    source: &MetalTensor,
    row: usize,
    count: usize,
) -> Result<(), Qwen4ExpMetalError> {
    if !probe.capture {
        return Ok(());
    }
    let target = probe.buffer(name);
    let width = target.shape[0] as usize;
    let dest = target.view_subrange((row * width) as u64, vec![(count * width) as u64]);
    let _tag = crate::metal::dispatch_census_tag_scope(|| format!("{COPY_TAG}{name}"));
    crate::metal::encode_copy_offset_f32(ctx, enc, source, 0, &dest, count * width)?;
    Ok(())
}

pub(super) fn before(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weights: GatedResidualMetalReadWeights<'_>,
    input: &MetalTensor,
    tokens: usize,
) -> Result<Option<Visit>, Qwen4ExpMetalError> {
    ACTIVE.with(|p| {
        let p = p.borrow();
        let Some(probe) = p.as_ref() else {
            return Ok(None);
        };
        if qwen4exp_hc_packed_projection_override_active() {
            return Err(invalid("overlapping broad HC override"));
        }
        if [identity(weights.down), identity(weights.up)] != probe.pair.map(Some) {
            return Ok(None);
        }
        if probe.records.len() % 2 != 0 {
            return Err(invalid("unfinished HC projection pair"));
        }
        let range = crate::qwen4exp_composition_trace::qwen4exp_diagnostic_execution_range()
            .ok_or_else(|| invalid("missing HC execution range"))?;
        if range.1 != tokens {
            return Err(invalid("HC token count differs from execution range"));
        }
        let (row, count) = window(probe.candidate, probe.records.len() / 2, range)?;
        copy(ctx, enc, probe, "hc.hyper", input, row, count)?;
        Ok(Some(Visit {
            start: range.0,
            tokens,
            row,
            count,
            policy: probe.policy,
        }))
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn project(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    visit: Option<&Visit>,
    role: Role,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    k: usize,
    m: usize,
    tokens: usize,
) -> Result<bool, Qwen4ExpMetalError> {
    let Some(v) = visit else {
        return Ok(false);
    };
    ACTIVE.with(|p| {
        let mut p = p.borrow_mut();
        let probe = p.as_mut().unwrap();
        let index = match role {
            Role::Down => 0,
            Role::Up => 1,
        };
        if probe.records.len() % 2 != index
            || identity(weight) != Some(probe.pair[index])
            || [k as u64, m as u64] != probe.pair[index].shape
            || tokens != v.tokens
        {
            return Err(invalid("HC projection binding/order changed"));
        }
        let (before, after) = match role {
            Role::Down => ("hc.normalized", "hc.down"),
            Role::Up => ("hc.low", "hc.up"),
        };
        copy(ctx, enc, probe, before, input, v.row, v.count)?;
        {
            let _tag = crate::metal::dispatch_census_tag_scope(|| {
                format!(
                    "{PROJECTION_TAG}{}.absolute{}.N{}",
                    role.label(),
                    v.start,
                    tokens
                )
            });
            let dispatch = || {
                encode_mat_mat_dispatch_without_fewrow(
                    ctx, enc, weight, input, output, k, m, tokens,
                )
            };
            match v.policy {
                Policy::Production => dispatch()?,
                Policy::F32DownUp => {
                    crate::metal_forward::with_matmat_bf16_bfloat_act_override(false, dispatch)?
                }
            }
        }
        copy(ctx, enc, probe, after, output, v.row, v.count)?;
        probe.records.push(Record {
            start: v.start,
            tokens,
            role,
            policy: v.policy,
        });
        Ok(true)
    })
}

#[test]
fn frontier_hc_ranges_and_sizes() {
    assert_eq!(window(false, 0, (2048, 3)).unwrap(), (0, 3));
    assert_eq!(window(false, 1, (2051, 2045)).unwrap(), (3, 5));
    assert_eq!(window(true, 0, (2048, 2048)).unwrap(), (0, 8));
    for (candidate, command, range) in [
        (false, 0, (0, 2048)),
        (true, 0, (2048, 8)),
        (false, 2, (2051, 2045)),
        (false, 0, (2048, 2048)),
    ] {
        assert!(window(candidate, command, range).is_err());
    }
    let bytes: u64 = Probe::specs()
        .iter()
        .map(|(_, s)| 4 * s.iter().product::<u64>())
        .sum();
    assert_eq!(bytes, 1_003_520);
}

#[test]
fn frontier_hc_identity_requires_both_offsets_shapes_and_bf16() {
    let down = Identity {
        buffer: 1,
        offset: 64,
        shape: [10240, 320],
        dtype: GgmlType::BF16,
    };
    let up = Identity {
        buffer: 1,
        offset: 6_553_664,
        shape: [320, 10240],
        dtype: GgmlType::BF16,
    };
    let pair = [down, up];
    assert!(eligible(&pair));
    let mut changed = pair;
    changed[1].offset += 2;
    assert_ne!(changed, pair);
    changed = pair;
    changed[0].buffer += 1;
    assert_ne!(changed, pair);
    changed = pair;
    changed[1].shape.swap(0, 1);
    assert!(!eligible(&changed));
    changed = pair;
    changed[0].dtype = GgmlType::F16;
    assert!(!eligible(&changed));
}

#[test]
fn frontier_hc_scope_restores_nested_unwind_thread_and_rejects_broad_override() {
    fn empty() -> Probe {
        let pair = [
            Identity {
                buffer: 1,
                offset: 0,
                shape: [10240, 320],
                dtype: GgmlType::BF16,
            },
            Identity {
                buffer: 2,
                offset: 0,
                shape: [320, 10240],
                dtype: GgmlType::BF16,
            },
        ];
        Probe {
            pair,
            buffers: Vec::new(),
            records: Vec::new(),
            policy: Policy::Production,
            candidate: false,
            capture: false,
        }
    }
    let state = || {
        ACTIVE.with(|p| {
            p.borrow()
                .as_ref()
                .map(|p| (p.policy, p.candidate, p.capture, p.records.len()))
        })
    };
    assert!(state().is_none());
    with_frontier_hc_probe(empty(), Policy::F32DownUp, false, true, || {
        let outer = state();
        with_frontier_hc_probe(empty(), Policy::Production, true, false, || {
            assert_eq!(state(), Some((Policy::Production, true, false, 0)));
        });
        assert_eq!(state(), outer);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                with_frontier_hc_probe(empty(), Policy::Production, true, false, || {
                    panic!("HC scope probe")
                })
            }))
            .is_err()
        );
        assert_eq!(state(), outer);
        assert!(
            std::thread::spawn(|| ACTIVE.with(|p| p.borrow().is_none()))
                .join()
                .unwrap()
        );
    });
    assert!(state().is_none());
    with_qwen4exp_hc_packed_projection_override(
        Qwen4ExpHcPackedProjectionArm::WideF32Down,
        2048,
        2048,
        || {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    with_frontier_hc_probe(empty(), Policy::Production, true, false, || ())
                }))
                .is_err()
            );
        },
    );
    assert!(state().is_none());
}
