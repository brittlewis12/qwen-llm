use super::*;

pub(in super::super) const DECODER_BATCH_LAYERS: usize = 2;
pub(in super::super) const MAX_GROUP_UNITS: usize = 16;
const ENCODING_ALLOWANCE_PER_UNIT: u64 = 1024 * 1024;

pub(in super::super) fn status_guard_words() -> usize {
    if cfg!(test) { 4 } else { 0 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) struct Selection {
    pub layers: usize,
    pub reason: Option<&'static str>,
}

pub(in super::super) fn select(plan: &CedPlan, requested: usize) -> Result<Selection> {
    if requested == 0 {
        return Ok(Selection {
            layers: 0,
            reason: Some("scoped_incumbent_reference"),
        });
    }
    if ![1, 2, 4].contains(&requested) {
        return Err(CedPlanError::InvalidGeometry("decoder batch size"));
    }
    for group in plan.decoder.chunks(requested) {
        if unit_count(plan, group[0].layer..group.last().unwrap().layer + 1)? > MAX_GROUP_UNITS {
            return Ok(Selection {
                layers: 0,
                reason: Some("physical_unit_budget: use bounded incumbent CED commands"),
            });
        }
    }
    Ok(Selection {
        layers: requested,
        reason: None,
    })
}

fn checked_layers(plan: &CedPlan, layers: &Range<usize>) -> Result<Range<usize>> {
    if layers.start < plan.publication_layer
        || layers.start >= layers.end
        || layers.end > add(plan.publication_layer, plan.decoder.len(), "decoder end")?
        || layers.len() > 4
    {
        return Err(CedPlanError::InvalidGeometry("decoder batch layers"));
    }
    Ok(layers.start - plan.publication_layer..layers.end - plan.publication_layer)
}

pub(in super::super) fn unit_count(plan: &CedPlan, layers: Range<usize>) -> Result<usize> {
    let index = checked_layers(plan, &layers)?;
    let tail = super::super::prefill::original_scalar_tail(plan.prompt_rows);
    plan.decoder[index].iter().try_fold(0, |sum, layer| {
        let end = tail.unwrap_or(layer.output_rows.end);
        let outputs = end
            .checked_sub(layer.output_rows.start)
            .ok_or(CedPlanError::InvalidRowView)?;
        add(
            sum,
            add(
                layer.seed_rows.len().div_ceil(DEFAULT_CHUNK_ROWS),
                add(
                    outputs.div_ceil(DEFAULT_CHUNK_ROWS),
                    usize::from(tail.is_some()),
                    "batch unit count",
                )?,
                "batch unit count",
            )?,
            "batch unit count",
        )
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum Kind {
    Seed,
    Rows,
    Scalar,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in super::super) struct Unit {
    pub layer: usize,
    pub rows: Range<usize>,
    pub kind: Kind,
}

pub(in super::super) fn units(plan: &CedPlan, layers: Range<usize>) -> Result<Vec<Unit>> {
    let count = unit_count(plan, layers.clone())?;
    if count > MAX_GROUP_UNITS {
        return Err(CedPlanError::InvalidGeometry(
            "decoder group physical unit budget",
        ));
    }
    let mut units = Vec::with_capacity(count);
    let tail = super::super::prefill::original_scalar_tail(plan.prompt_rows);
    for layer in
        &plan.decoder[layers.start - plan.publication_layer..layers.end - plan.publication_layer]
    {
        for chunk in chunk_views(layer.seed_rows.clone(), DEFAULT_CHUNK_ROWS)? {
            units.push(Unit {
                layer: layer.layer,
                rows: chunk.absolute_rows,
                kind: Kind::Seed,
            });
        }
        for chunk in chunk_views(
            layer.output_rows.start..tail.unwrap_or(layer.output_rows.end),
            DEFAULT_CHUNK_ROWS,
        )? {
            units.push(Unit {
                layer: layer.layer,
                rows: chunk.absolute_rows,
                kind: Kind::Rows,
            });
        }
        if let Some(position) = tail {
            units.push(Unit {
                layer: layer.layer,
                rows: position..position + 1,
                kind: Kind::Scalar,
            });
        }
    }
    Ok(units)
}

pub(in super::super) fn status_words(units: &[Unit]) -> Result<usize> {
    units.iter().try_fold(0usize, |sum, unit| {
        add(
            sum,
            1 + if unit.kind == Kind::Rows {
                unit.rows.len()
            } else {
                0
            } + 2 * status_guard_words() * if unit.kind == Kind::Rows { 2 } else { 1 },
            "batch status words",
        )
    })
}

pub(in super::super) fn maximum_status_words(plan: &CedPlan, layers: usize) -> Result<usize> {
    if ![1, 2, 4].contains(&layers) {
        return Err(CedPlanError::InvalidGeometry("decoder batch size"));
    }
    let mut maximum = 0;
    for group in plan.decoder.chunks(layers) {
        maximum = maximum.max(status_words(&units(
            plan,
            group[0].layer..group.last().unwrap().layer + 1,
        )?)?);
    }
    Ok(maximum)
}

pub(in super::super) fn encoding_reserve_bytes(plan: &CedPlan, layers: usize) -> Result<u64> {
    if ![1, 2, 4].contains(&layers) {
        return Err(CedPlanError::InvalidGeometry("decoder batch size"));
    }
    let mut maximum = 0;
    for group in plan.decoder.chunks(layers) {
        let units = units(plan, group[0].layer..group.last().unwrap().layer + 1)?;
        // Additional CPU/private and device allowance, not a measured or formal
        // driver upper bound. Structural eligibility caps it at 16 MiB/domain.
        let bytes = (units.len() as u64)
            .checked_mul(ENCODING_ALLOWANCE_PER_UNIT)
            .ok_or(CedPlanError::Overflow("batch encoding reserve"))?;
        maximum = maximum.max(bytes);
    }
    Ok(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_groups_preserve_every_absolute_unit_and_work_at_chunk_edges() {
        for window in [1, 3, 128, 1025, 2050] {
            let c = Config {
                window,
                ..Config::default()
            };
            for n in [1, 127, 128, 129, 2047, 2048, 2049, 4095, 4096, 4097, 8192] {
                let p = CedPlan::shrinking(
                    &c,
                    &Schedule::default(),
                    &HashConfig::default(),
                    &CedRequest::cold(n, n),
                )
                .unwrap();
                let mut reference = Vec::new();
                for l in &p.decoder {
                    reference.extend(units(&p, l.layer..l.layer + 1).unwrap());
                }
                assert_eq!(
                    reference
                        .iter()
                        .filter(|u| u.kind == Kind::Seed)
                        .map(|u| u.rows.len())
                        .sum::<usize>(),
                    p.work.decoder_seed_rows
                );
                assert_eq!(
                    reference
                        .iter()
                        .filter(|u| u.kind != Kind::Seed)
                        .map(|u| u.rows.len())
                        .sum::<usize>(),
                    p.work.decoder_output_rows
                );
                for depth in [1, 2, 4] {
                    let selection = select(&p, depth).unwrap();
                    if selection.layers == 0 {
                        assert!(p.decoder.chunks(depth).any(|g| {
                            unit_count(&p, g[0].layer..g.last().unwrap().layer + 1).unwrap()
                                > MAX_GROUP_UNITS
                        }));
                        assert!(maximum_status_words(&p, depth).is_err());
                        continue;
                    }
                    let mut actual = Vec::new();
                    let mut maximum = 0;
                    for g in p.decoder.chunks(depth) {
                        let u = units(&p, g[0].layer..g.last().unwrap().layer + 1).unwrap();
                        assert!(u.len() <= MAX_GROUP_UNITS);
                        assert!(
                            u.iter().map(|u| u.rows.len()).sum::<usize>()
                                <= MAX_GROUP_UNITS * DEFAULT_CHUNK_ROWS
                        );
                        maximum = maximum.max(status_words(&u).unwrap());
                        actual.extend(u);
                    }
                    assert_eq!(actual, reference, "W={window} N={n} depth={depth}");
                    assert_eq!(maximum_status_words(&p, depth).unwrap(), maximum);
                }
                for unit in &reference {
                    assert!(unit.rows.len() <= DEFAULT_CHUNK_ROWS);
                    assert_eq!(
                        unit.kind == Kind::Scalar,
                        super::super::super::prefill::original_scalar_tail(n)
                            .is_some_and(|tail| unit.rows.start == tail)
                    );
                }
            }
        }
    }

    #[test]
    fn large_windows_choose_bounded_incumbent_before_allocating_or_pricing_groups() {
        for n in [32768, 1_048_576] {
            for window in [1, 128, n] {
                let c = Config {
                    window,
                    ..crate::deepseek_v41::forward::toy::config()
                };
                let p = CedPlan::shrinking(
                    &c,
                    &Schedule::default(),
                    &HashConfig::default(),
                    &CedRequest::cold(n, n),
                )
                .unwrap();
                for depth in [1, 2, 4] {
                    let selection = select(&p, depth).unwrap();
                    if window == n {
                        assert_eq!(selection.layers, 0);
                        assert!(selection.reason.unwrap().contains("physical_unit_budget"));
                        assert!(
                            units(&p, p.publication_layer..p.publication_layer + depth).is_err()
                        );
                    } else {
                        assert_eq!(selection.layers, depth);
                        assert!(
                            encoding_reserve_bytes(&p, depth).unwrap()
                                <= MAX_GROUP_UNITS as u64 * ENCODING_ALLOWANCE_PER_UNIT
                        );
                    }
                }
                let memory = crate::deepseek_v41::forward::ced::estimate_from_parts(
                    &c,
                    40,
                    &[],
                    0,
                    n,
                    &p,
                    false,
                    &|n| Ok(n),
                )
                .unwrap();
                assert!(memory.batch_encoding_reserve_bytes <= 16 * 1024 * 1024);
                if window == n {
                    assert_eq!(memory.decoder_batch_layers, 0);
                    assert_eq!(memory.command_reserve_bytes, 16 * 1024 * 1024);
                    assert_eq!(memory.batch_encoding_reserve_bytes, 0);
                    assert_eq!(memory.priced_batch_status_bytes, 0);
                    assert_eq!(memory.batch_cpu_bytes, 0);
                    assert_eq!(memory.priced_batch_diagnostic_bytes, 0);
                    assert!(
                        memory
                            .decoder_schedule_reason
                            .unwrap()
                            .contains("physical_unit_budget")
                    );
                }
            }
        }
    }

    #[test]
    fn production_command_plan_is_derived_not_fixed_to_twenty_decoder_layers() {
        for n in [128, 2048, 8192] {
            let p = CedPlan::shrinking(
                &Config::default(),
                &Schedule::default(),
                &HashConfig::default(),
                &CedRequest::cold(n, n),
            )
            .unwrap();
            for depth in [1, 2, 4] {
                let expected = n.div_ceil(DEFAULT_CHUNK_ROWS) + p.decoder.len().div_ceil(depth);
                assert_eq!(
                    n.div_ceil(DEFAULT_CHUNK_ROWS) + p.decoder.chunks(depth).count(),
                    expected
                );
                if DEFAULT_CHUNK_ROWS == 2048 {
                    let pinned = match (n, depth) {
                        (8192, 1) => 24,
                        (8192, 2) => 14,
                        (8192, 4) => 9,
                        (_, 1) => 21,
                        (_, 2) => 11,
                        (_, 4) => 6,
                        _ => unreachable!(),
                    };
                    assert_eq!(expected, pinned);
                }
            }
            let memory = crate::deepseek_v41::forward::ced::estimate_from_parts(
                &Config::default(),
                40,
                &[],
                0,
                n,
                &p,
                true,
                &|bytes| Ok(if bytes == 0 { 0 } else { bytes + 16384 }),
            )
            .unwrap();
            eprintln!(
                "production_batch_plan N={n} status_words={} {}",
                maximum_status_words(&p, 2).unwrap(),
                serde_json::to_string(&memory).unwrap()
            );
            assert!(maximum_status_words(&p, 0).is_err());
            assert!(units(&p, 20..25).is_err());
        }
    }
}
