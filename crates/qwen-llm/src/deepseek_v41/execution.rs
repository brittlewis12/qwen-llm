//! CPU-only NativeF32 execution contract, separate from descriptive GGUF validity
//! and request/device memory admission. Optimization widths are not model limits.
use super::forward::Config;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum UnsupportedExecution {
    #[error("unsupported V4.1 execution: Sinkhorn iterations {0}; backend requires 20")]
    SinkhornIterations(usize),
    #[error("unsupported V4.1 execution: routing top-k {0}; backend requires 1..=16")]
    RoutingTopK(usize),
    #[error("unsupported V4.1 execution: {0}")]
    Geometry(&'static str),
}

impl UnsupportedExecution {
    pub fn code(&self) -> &'static str {
        match self {
            Self::SinkhornIterations(_) => "unsupported_v41_sinkhorn_iterations",
            Self::RoutingTopK(_) => "unsupported_v41_routing_topk",
            Self::Geometry(_) => "unsupported_v41_execution_geometry",
        }
    }
}

type Result = std::result::Result<(), UnsupportedExecution>;

pub(crate) fn validate_ffn(c: &Config) -> Result {
    if c.hc_iterations != 20 {
        return Err(UnsupportedExecution::SinkhornIterations(c.hc_iterations));
    }
    if !(1..=crate::metal::ROUTE_MAX_TOP_K).contains(&c.topk) {
        return Err(UnsupportedExecution::RoutingTopK(c.topk));
    }
    if c.topk > c.experts {
        return Err(UnsupportedExecution::Geometry(
            "routing top-k exceeds expert count",
        ));
    }
    if [c.hidden, c.intermediate, c.experts]
        .into_iter()
        .any(|n| n == 0 || n > u32::MAX as usize / 64)
    {
        return Err(UnsupportedExecution::Geometry(
            "GPU FFN extent overflow or zero dimension",
        ));
    }
    if ![c.norm_eps, c.hc_eps, c.clamp, c.gate_temperature]
        .into_iter()
        .all(|n| n.is_finite() && n > 0.0)
        || !c.route_scale.is_finite()
    {
        return Err(UnsupportedExecution::Geometry("invalid forward scalar"));
    }
    Ok(())
}

/// Check intrinsic backend constraints, without a Metal context, model import,
/// request chunk selection, or a local-capacity decision.
pub fn validate(c: &Config) -> Result {
    validate_ffn(c)?;
    if [
        c.vocab,
        c.heads,
        c.head_dim,
        c.query_rank,
        c.output_rank,
        c.groups,
        c.index_heads,
        c.index_dim,
        c.index_topk,
        c.window,
        c.candidate_block,
        c.candidate_topk,
    ]
    .into_iter()
    .any(|n| n == 0 || n > u32::MAX as usize)
    {
        return Err(UnsupportedExecution::Geometry("invalid forward dimension"));
    }
    if !c.heads.is_multiple_of(c.groups)
        || c.rope_dim > c.head_dim
        || c.rope_dim > c.index_dim
        || !c.rope_dim.is_multiple_of(2)
    {
        return Err(UnsupportedExecution::Geometry("invalid attention geometry"));
    }
    for width in [
        c.heads.checked_mul(c.head_dim),
        c.groups.checked_mul(c.output_rank),
        c.index_heads.checked_mul(c.index_dim),
        c.hidden.checked_mul(5),
    ] {
        if width.is_none_or(|n| n > u32::MAX as usize) {
            return Err(UnsupportedExecution::Geometry("projection width overflow"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_math_and_nonfastpath_shapes_are_executable_without_fit_gate() {
        let released = Config::default();
        assert_eq!(
            (released.hc_iterations, released.topk, released.groups),
            (20, 6, 8)
        );
        validate(&released).unwrap();
        for experts in [17, 272, 320, 384, 513] {
            for groups in [1, 8, 16, 17, 32] {
                let c = Config {
                    experts,
                    groups,
                    heads: groups * 2,
                    hidden: 97,
                    intermediate: 103,
                    output_rank: 7,
                    ..released.clone()
                };
                validate(&c).unwrap();
            }
        }
    }

    #[test]
    fn sinkhorn_backend_limit_is_explicit() {
        for iterations in [0, 1, 3, 19, 21, usize::MAX] {
            let c = Config {
                hc_iterations: iterations,
                ..Config::default()
            };
            assert_eq!(
                validate(&c),
                Err(UnsupportedExecution::SinkhornIterations(iterations))
            );
            assert_eq!(validate(&c), validate_ffn(&c));
        }
    }

    #[test]
    fn routing_backend_limit_is_explicit() {
        for topk in [1, 6, 16] {
            validate(&Config {
                topk,
                ..Config::default()
            })
            .unwrap();
        }
        for topk in [0, 17, 32, usize::MAX] {
            let c = Config {
                topk,
                ..Config::default()
            };
            assert_eq!(validate(&c), Err(UnsupportedExecution::RoutingTopK(topk)));
            assert_eq!(validate(&c), validate_ffn(&c));
        }
    }

    #[test]
    fn malformed_config_dimensions_and_controls_are_rejected_without_panics() {
        let base = Config::default();
        for c in [
            Config {
                groups: 0,
                ..base.clone()
            },
            Config {
                groups: 3,
                ..base.clone()
            },
            Config {
                experts: 1,
                ..base.clone()
            },
            Config {
                hidden: u32::MAX as usize / 64 + 1,
                ..base.clone()
            },
            Config {
                output_rank: usize::MAX,
                ..base.clone()
            },
            Config {
                heads: u32::MAX as usize,
                groups: 1,
                ..base.clone()
            },
            Config {
                rope_dim: 3,
                ..base.clone()
            },
            Config {
                norm_eps: f32::NAN,
                ..base.clone()
            },
            Config {
                hc_eps: 0.0,
                ..base.clone()
            },
            Config {
                clamp: -1.0,
                ..base.clone()
            },
            Config {
                gate_temperature: f32::INFINITY,
                ..base.clone()
            },
            Config {
                route_scale: f32::NAN,
                ..base
            },
        ] {
            assert!(matches!(
                validate(&c),
                Err(UnsupportedExecution::Geometry(_))
            ));
        }
    }
}
