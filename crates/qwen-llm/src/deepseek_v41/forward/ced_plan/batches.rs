use super::*;

pub(in super::super) const DECODER_BATCH_LAYERS: usize = 2;

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
    if layers.start < plan.publication_layer
        || layers.start >= layers.end
        || layers.end > plan.publication_layer + plan.decoder.len()
        || layers.len() > 4
    {
        return Err(CedPlanError::InvalidGeometry("decoder batch layers"));
    }
    let mut units = Vec::new();
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
            },
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
        // Explicit driver/encoder allowance: 4 KiB per encoded operation-row
        // plus 1 KiB per unit, in addition to the incumbent command reserve.
        // This is a conservative admission allowance, not a measured peak.
        let bytes = units.iter().try_fold(0usize, |sum, u| {
            add(
                sum,
                add(
                    mul(u.rows.len(), 4096, "batch encoding reserve")?,
                    1024,
                    "batch encoding reserve",
                )?,
                "batch encoding reserve",
            )
        })?;
        maximum = maximum.max(bytes);
    }
    u64::try_from(maximum).map_err(|_| CedPlanError::Overflow("batch encoding reserve"))
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
                    let mut actual = Vec::new();
                    let mut maximum = 0;
                    for g in p.decoder.chunks(depth) {
                        let u = units(&p, g[0].layer..g.last().unwrap().layer + 1).unwrap();
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
    fn production_command_plan_is_derived_not_fixed_to_twenty_decoder_layers() {
        for n in [128, 2048, 8192] {
            let p = CedPlan::shrinking(
                &Config::default(),
                &Schedule::default(),
                &HashConfig::default(),
                &CedRequest::cold(n, n),
            )
            .unwrap();
            for (depth, expected) in [
                (1, if n == 8192 { 24 } else { 21 }),
                (2, if n == 8192 { 14 } else { 11 }),
                (4, if n == 8192 { 9 } else { 6 }),
            ] {
                assert_eq!(
                    n.div_ceil(DEFAULT_CHUNK_ROWS) + p.decoder.chunks(depth).count(),
                    expected
                );
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
