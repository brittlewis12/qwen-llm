use super::super::super::ced_plan::CedRequest;
use super::*;

mod sink_lineage;

fn fixture(n: usize) -> (Config, Schedule, CedPlan) {
    let c = Config {
        window: 3,
        ..Config::default()
    };
    let schedule = Schedule {
        layers: 4,
        kv_source_layers: vec![2],
        index_source_layers: vec![2, 3],
        compression_ratios: vec![0, 0, 1, 1],
        candidate_source_layer: Some(2),
    };
    let engram = engram::HashConfig {
        layers: Vec::new(),
        ..engram::HashConfig::default()
    };
    let plan = CedPlan::shrinking(&c, &schedule, &engram, &CedRequest::cold(n, n + 1)).unwrap();
    (c, schedule, plan)
}

fn encoded(cursor: &mut Cursor, publication: bool) {
    let p = cursor.pending.as_mut().unwrap();
    if let Some(b) = &mut p.batch {
        b.layer = p.end_layer;
        b.row = 0;
    }
    p.encoded = publication || !matches!(p.phase, Phase::Encoder);
    p.finished.set(true);
}

#[test]
fn pending_checked_ranges_match_rows_and_refuse_after_end_atomically() {
    let make = || Pending {
        phase: Phase::Decoder(2),
        range: 4..7,
        encoded_seed_prefix: None,
        batch: Some(BatchCursor {
            start: 4,
            rows: 3,
            layer: 2,
            row: 0,
        }),
        end_layer: 3,
        encoded: false,
        finished: std::cell::Cell::new(false),
        controls: None,
    };
    let mut range = make();
    let mut scalar = make();
    for (position, count) in [(4, 1), (5, 2)] {
        range.advance_attention(2, position, count).unwrap();
        for p in position..position + count {
            scalar.advance_attention(2, p, 1).unwrap();
        }
        assert_eq!(range.batch, scalar.batch);
        assert_eq!(range.encoded, scalar.encoded);
        assert!(!range.finished.get() && range.controls.is_none());
    }
    assert!(range.encoded);
    let done = range.batch;
    for (id, position, count) in [
        (3, 4, 1),
        (3, 4, 3),
        (2, 4, 1),
        (3, 4, 0),
        (3, 4, usize::MAX),
    ] {
        assert!(range.advance_attention(id, position, count).is_err());
        assert_eq!(range.batch, done);
        assert!(range.encoded && !range.finished.get());
    }
    let mut encoder = make();
    encoder.phase = Phase::Encoder;
    encoder.end_layer = 4;
    for id in 2..4 {
        encoder.advance_attention(id, 4, 3).unwrap();
        assert!(!encoder.encoded && !encoder.finished.get());
    }
}

#[test]
fn pending_ranges_encode_without_finishing_coverage_completion_or_group_readiness() {
    let (c, schedule, plan) = fixture(9);
    let mut cursor = Cursor::new(&plan, &c, &schedule, 10).unwrap();
    cursor.begin_encoder(0, 9).unwrap();
    for id in 0..plan.publication_layer {
        cursor
            .pending
            .as_mut()
            .unwrap()
            .advance_attention(id, 0, 9)
            .unwrap();
    }
    assert!(!cursor.pending.as_ref().unwrap().encoded);
    assert!(cursor.complete(Phase::Encoder, 0, 9).is_err());
    assert_eq!(
        (cursor.encoder_completed, cursor.publication_completed),
        (0, 0)
    );
    encoded(&mut cursor, true);
    cursor.complete(Phase::Encoder, 0, 9).unwrap();
    let ticket = cursor.begin_group(2..4).unwrap();
    let coverage = Arc::new(Mutex::new(Coverage {
        origin: plan.candidate_rows.start,
        selected: vec![None; plan.candidate_rows.len()],
        candidates: vec![None; plan.candidate_rows.len()],
    }));
    cursor.group.as_mut().unwrap().coverage = Some(coverage.clone());
    let d = &plan.decoder[0];
    cursor
        .begin_seed(d.layer, d.seed_rows.start, d.seed_rows.len())
        .unwrap();
    encoded(&mut cursor, false);
    cursor
        .stage_group_unit(ticket, d.layer, d.seed_rows.start, d.seed_rows.len(), true)
        .unwrap();
    cursor
        .begin_decoder(d.layer, d.output_rows.start, d.output_rows.len())
        .unwrap();
    let before = (
        cursor.decoder_next,
        cursor.output_completed.clone(),
        cursor.seed_completed.clone(),
    );
    let p = cursor.pending.as_mut().unwrap();
    let original = p.batch;
    for (id, start, count) in [
        (d.layer + 1, d.output_rows.start, 1),
        (d.layer, d.output_rows.start + 1, 1),
        (d.layer, d.output_rows.start, 0),
        (d.layer, d.output_rows.start, usize::MAX),
    ] {
        assert!(p.advance_attention(id, start, count).is_err());
        assert_eq!(p.batch, original);
        assert!(!p.encoded && !p.finished.get());
    }
    p.advance_attention(d.layer, d.output_rows.start, 1)
        .unwrap();
    assert!(!p.encoded);
    p.advance_attention(d.layer, d.output_rows.start + 1, d.output_rows.len() - 1)
        .unwrap();
    assert!(p.encoded && !p.finished.get() && p.controls.is_none());
    let end = p.batch;
    assert!(
        p.advance_attention(d.layer + 1, d.output_rows.start, 1)
            .is_err()
    );
    assert_eq!(p.batch, end);
    assert!(p.encoded && !p.finished.get());
    assert!(
        cursor
            .complete(
                Phase::Decoder(d.layer),
                d.output_rows.start,
                d.output_rows.len()
            )
            .is_err()
    );
    assert!(
        cursor
            .stage_group_unit(
                ticket,
                d.layer,
                d.output_rows.start,
                d.output_rows.len(),
                false
            )
            .is_err()
    );
    assert!(cursor.validate_group(ticket, &(2..4)).is_err());
    assert_eq!(
        (
            cursor.decoder_next,
            cursor.output_completed.clone(),
            cursor.seed_completed.clone()
        ),
        before
    );
    assert_eq!(cursor.group.as_ref().unwrap().units.len(), 1);
    assert!(Arc::ptr_eq(
        cursor.group.as_ref().unwrap().coverage.as_ref().unwrap(),
        &coverage
    ));
    let coverage = coverage.lock().unwrap();
    assert!(
        coverage
            .selected
            .iter()
            .chain(&coverage.candidates)
            .all(Option::is_none)
    );
    assert!(!cursor.ready(9));
}

#[test]
fn decoder_group_keeps_public_frontiers_atomic_and_rejects_bad_tickets_before_mutation() {
    let (c, schedule, plan) = fixture(9);
    let mut cursor = Cursor::new(&plan, &c, &schedule, 10).unwrap();
    assert!(cursor.begin_group(2..4).is_err());
    cursor.begin_encoder(0, 9).unwrap();
    encoded(&mut cursor, true);
    cursor.complete(Phase::Encoder, 0, 9).unwrap();
    for layers in [1..3, 2..5, 3..4, 2..2] {
        assert!(cursor.begin_group(layers).is_err());
    }
    let initial = (
        cursor.seed_completed.clone(),
        cursor.output_completed.clone(),
        cursor.decoder_next,
    );
    let ticket = cursor.begin_group(2..4).unwrap();
    assert!(cursor.begin_group(2..4).is_err());
    assert!(cursor.begin_encoder(0, 1).is_err());
    for layer in &plan.decoder {
        cursor
            .begin_seed(layer.layer, layer.seed_rows.start, layer.seed_rows.len())
            .unwrap();
        assert!(
            cursor
                .stage_group_unit(
                    ticket,
                    layer.layer,
                    layer.seed_rows.start,
                    layer.seed_rows.len(),
                    true
                )
                .is_err()
        );
        encoded(&mut cursor, false);
        for (generation, id, start, rows) in [
            (
                ticket + 1,
                layer.layer,
                layer.seed_rows.start,
                layer.seed_rows.len(),
            ),
            (
                ticket,
                layer.layer + 1,
                layer.seed_rows.start,
                layer.seed_rows.len(),
            ),
            (
                ticket,
                layer.layer,
                layer.seed_rows.start + 1,
                layer.seed_rows.len(),
            ),
        ] {
            assert!(
                cursor
                    .stage_group_unit(generation, id, start, rows, true)
                    .is_err()
            );
            assert!(cursor.pending.is_some());
        }
        assert!(
            cursor
                .complete(
                    Phase::Seed(layer.layer),
                    layer.seed_rows.start,
                    layer.seed_rows.len()
                )
                .is_err()
        );
        cursor
            .stage_group_unit(
                ticket,
                layer.layer,
                layer.seed_rows.start,
                layer.seed_rows.len(),
                true,
            )
            .unwrap();
        cursor
            .begin_decoder(
                layer.layer,
                layer.output_rows.start,
                layer.output_rows.len(),
            )
            .unwrap();
        encoded(&mut cursor, false);
        cursor
            .stage_group_unit(
                ticket,
                layer.layer,
                layer.output_rows.start,
                layer.output_rows.len(),
                false,
            )
            .unwrap();
        assert_eq!(
            (
                cursor.seed_completed.clone(),
                cursor.output_completed.clone(),
                cursor.decoder_next
            ),
            initial
        );
        assert!(!cursor.ready(9));
    }
    for (generation, layers) in [(ticket + 1, 2..4), (ticket, 2..3), (ticket, 3..4)] {
        assert!(cursor.validate_group(generation, &layers).is_err());
        assert_eq!(
            (
                cursor.seed_completed.clone(),
                cursor.output_completed.clone(),
                cursor.decoder_next
            ),
            initial
        );
    }
    cursor.validate_group(ticket, &(2..4)).unwrap();
    cursor.publish_group();
    assert!(cursor.ready(9));
    assert!(cursor.validate_group(ticket, &(2..4)).is_err());
}

#[test]
fn frontiers_advance_only_at_completion_and_commit_requires_every_layer() {
    let (c, schedule, plan) = fixture(9);
    let mut cursor = Cursor::new(&plan, &c, &schedule, 10).unwrap();
    assert!(!cursor.ready(9));
    assert!(cursor.begin_decoder(2, 6, 3).is_err());
    cursor.begin_encoder(0, 4).unwrap();
    assert!(cursor.complete(Phase::Encoder, 0, 4).is_err());
    encoded(&mut cursor, false);
    assert!(cursor.complete(Phase::Encoder, 0, 4).is_err());
    encoded(&mut cursor, true);
    assert_eq!(
        (cursor.encoder_completed, cursor.publication_completed),
        (0, 0)
    );
    assert!(cursor.begin_encoder(4, 5).is_err());
    assert!(cursor.begin_decoder(2, 6, 3).is_err());
    assert!(cursor.complete(Phase::Encoder, 0, 3).is_err());
    cursor.complete(Phase::Encoder, 0, 4).unwrap();
    assert_eq!(
        (cursor.encoder_completed, cursor.publication_completed),
        (4, 4)
    );
    cursor.begin_encoder(4, 5).unwrap();
    encoded(&mut cursor, true);
    cursor.complete(Phase::Encoder, 4, 5).unwrap();
    assert!(cursor.begin_decoder(3, 8, 1).is_err());
    assert!(cursor.begin_decoder(2, 6, 3).is_err());
    for layer in plan.decoder.clone() {
        let j = cursor.decoder_next;
        cursor
            .begin_seed(layer.layer, layer.seed_rows.start, layer.seed_rows.len())
            .unwrap();
        assert!(
            cursor
                .complete(
                    Phase::Seed(layer.layer),
                    layer.seed_rows.start,
                    layer.seed_rows.len()
                )
                .is_err()
        );
        encoded(&mut cursor, false);
        assert_eq!(cursor.seed_completed[j], layer.seed_rows.start);
        let seed = layer.seed_rows;
        cursor
            .complete(Phase::Seed(layer.layer), seed.start, seed.len())
            .unwrap();
        let outputs = layer.output_rows;
        cursor
            .begin_decoder(layer.layer, outputs.start, outputs.len())
            .unwrap();
        encoded(&mut cursor, false);
        cursor.pending.as_ref().unwrap().finished.set(false);
        assert!(
            cursor
                .complete(Phase::Decoder(layer.layer), outputs.start, outputs.len())
                .is_err()
        );
        cursor.pending.as_ref().unwrap().finished.set(true);
        assert_eq!(cursor.output_completed[j], outputs.start);
        assert!(!cursor.ready(9));
        cursor
            .complete(Phase::Decoder(layer.layer), outputs.start, outputs.len())
            .unwrap();
    }
    assert!(cursor.ready(9));
    assert!(!cursor.ready(8));
}

#[test]
fn speculative_encoder_keeps_one_inflight_intent_and_completes_in_fifo_order() {
    let (c, schedule, plan) = fixture(9);
    let mut cursor = Cursor::new(&plan, &c, &schedule, 10).unwrap();
    cursor.begin_encoder(0, 4).unwrap();
    encoded(&mut cursor, true);
    // The normal path still refuses a second encoder while one is pending.
    assert!(cursor.begin_encoder(4, 5).is_err());
    cursor.begin_encoder_speculative(4, 5).unwrap();
    assert_eq!(cursor.inflight.len(), 1);
    assert_eq!(cursor.pending.as_ref().unwrap().range, 4..9);
    // One lookahead only, and decoder work cannot begin before both finish.
    assert!(cursor.begin_encoder_speculative(9, 0).is_err());
    assert!(cursor.begin_decoder(2, 6, 3).is_err());
    assert!(cursor.begin_seed(2, 0, 1).is_err());
    // Completing the older command advances the frontier but keeps the newest.
    cursor.complete(Phase::Encoder, 0, 4).unwrap();
    assert!(cursor.inflight.is_empty());
    assert!(cursor.pending.is_some());
    assert_eq!(
        (cursor.encoder_completed, cursor.publication_completed),
        (4, 4)
    );
    // Out-of-order completion is refused.
    encoded(&mut cursor, true);
    assert!(cursor.complete(Phase::Encoder, 0, 4).is_err());
    cursor.complete(Phase::Encoder, 4, 5).unwrap();
    assert!(cursor.pending.is_none());
    assert_eq!(cursor.encoder_completed, 9);
}

#[test]
fn speculative_encoder_retires_three_contiguous_chunks_in_order() {
    let (c, schedule, plan) = fixture(9);
    let mut cursor = Cursor::new(&plan, &c, &schedule, 10).unwrap();
    cursor.begin_encoder(0, 3).unwrap();
    encoded(&mut cursor, true);
    cursor.begin_encoder_speculative(3, 3).unwrap();
    // Retire the front; the newest stays pending.
    cursor.complete(Phase::Encoder, 0, 3).unwrap();
    assert_eq!(cursor.encoder_completed, 3);
    // A speculative begin needs the newest intent fully encoded.
    assert!(cursor.begin_encoder_speculative(6, 3).is_err());
    encoded(&mut cursor, true);
    cursor.begin_encoder_speculative(6, 3).unwrap();
    cursor.complete(Phase::Encoder, 3, 3).unwrap();
    assert_eq!(cursor.encoder_completed, 6);
    encoded(&mut cursor, true);
    cursor.complete(Phase::Encoder, 6, 3).unwrap();
    assert_eq!(cursor.encoder_completed, 9);
    assert!(cursor.inflight.is_empty() && cursor.pending.is_none());
}

#[test]
fn checked_absolute_ranges_history_and_control_prices() {
    assert_eq!(checked_range(&(100..120), 107, 3).unwrap(), 107..110);
    for (start, rows) in [(99, 1), (120, 1), (107, 0), (100, 2049), (usize::MAX, 2)] {
        assert!(checked_range(&(100..120), start, rows).is_err());
    }
    let history = HistoryRange {
        layer: 20,
        start: 100,
        floor: 98,
        extent: 9,
    };
    history.validate(20, 100, 7, 3).unwrap();
    assert!(history.validate(20, 100, 8, 3).is_err());
    assert!(history.validate(21, 100, 7, 3).is_err());
    assert!(history.validate(20, 101, 6, 3).is_err());
    assert!(
        HistoryRange {
            floor: 99,
            ..history
        }
        .validate(20, 100, 7, 3)
        .is_err()
    );
    let c = Config::default();
    assert_eq!(
        QueryControls::cpu_bytes(10_000..12_414).unwrap(),
        (2414 * std::mem::size_of::<Option<usize>>() * 2 + 1024) as u64
    );
    assert!(QueryControls::cpu_bytes(10..10).is_err());
    assert!(QueryControls::cpu_bytes(0..usize::MAX).is_err());
    for capacity in [16_384usize, 16_385, 1_000_000] {
        let shapes = QueryControls::shapes(&c, capacity, 10_000..12_414).unwrap();
        assert_eq!(
            shapes,
            [
                2414 * c.index_topk,
                2414 * capacity.div_ceil(c.candidate_block)
            ]
        );
        assert_eq!(
            QueryControls::logical_bytes(&c, capacity, 10_000..12_414).unwrap(),
            shapes.iter().sum::<usize>() as u64 * 4
        );
    }
    assert_eq!(PublicationRows::shapes(&c, 7).unwrap(), [7 * c.head_dim]);
    assert_eq!(SeedRows::shapes(&c, 7).unwrap(), [7 * c.head_dim]);
    assert_eq!(
        Rows::widths_with_controls(&c, 16_385).len() + 2,
        Rows::widths(&c, 16_385).len()
    );
}

#[test]
fn compound_seed_encoding_never_publishes_completion_and_retires_both_ranges() {
    let (c, schedule, plan) = fixture(9);
    let mut cursor = Cursor::new(&plan, &c, &schedule, 10).unwrap();
    cursor.begin_encoder(0, 9).unwrap();
    encoded(&mut cursor, true);
    cursor.complete(Phase::Encoder, 0, 9).unwrap();
    cursor.begin_seed(2, 4, 2).unwrap();
    assert!(cursor.chain_seed_to_decoder(2, 6, 2).is_err());
    encoded(&mut cursor, false);
    for (layer, start, rows) in [(3, 6, 2), (2, 7, 2), (2, 6, 0), (2, 6, 4)] {
        assert!(cursor.chain_seed_to_decoder(layer, start, rows).is_err());
    }
    cursor.chain_seed_to_decoder(2, 6, 2).unwrap();
    assert_eq!(cursor.seed_completed, [4, 6]);
    assert_eq!(cursor.output_completed, [6, 8]);
    assert_eq!(cursor.decoder_next, 0);
    let p = cursor.pending.as_ref().unwrap();
    assert_eq!(p.encoded_seed_prefix, Some(4..6));
    assert!(!p.encoded && !p.finished.get() && p.controls.is_none());
    assert!(cursor.chain_seed_to_decoder(2, 6, 2).is_err());
    assert!(cursor.complete(Phase::Seed(2), 4, 2).is_err());
    assert!(cursor.complete(Phase::Decoder(2), 6, 2).is_err());
    assert!(cursor.begin_decoder(2, 8, 1).is_err());
    assert!(cursor.begin_seed(3, 6, 2).is_err());
    encoded(&mut cursor, false);
    cursor.pending.as_ref().unwrap().finished.set(false);
    assert!(cursor.complete(Phase::Decoder(2), 6, 2).is_err());
    cursor.pending.as_ref().unwrap().finished.set(true);
    cursor.pending.as_mut().unwrap().encoded_seed_prefix = Some(3..6);
    assert!(cursor.complete(Phase::Decoder(2), 6, 2).is_err());
    assert_eq!(cursor.seed_completed[0], 4);
    assert_eq!(cursor.output_completed[0], 6);
    cursor.pending.as_mut().unwrap().encoded_seed_prefix = Some(4..6);
    cursor.complete(Phase::Decoder(2), 6, 2).unwrap();
    assert_eq!(cursor.seed_completed[0], 6);
    assert_eq!(cursor.output_completed[0], 8);
    assert_eq!(cursor.decoder_next, 0);
    cursor.begin_decoder(2, 8, 1).unwrap();
    encoded(&mut cursor, false);
    cursor.complete(Phase::Decoder(2), 8, 1).unwrap();
    assert_eq!(cursor.decoder_next, 1);
    cursor.begin_seed(3, 6, 2).unwrap();
    encoded(&mut cursor, false);
    cursor.chain_seed_to_decoder(3, 8, 1).unwrap();
    encoded(&mut cursor, false);
    assert!(!cursor.ready(9));
    cursor.complete(Phase::Decoder(3), 8, 1).unwrap();
    assert!(cursor.ready(9));
}

#[test]
fn compound_rejects_partial_multichunk_seed_and_genuine_scalar_tail() {
    let (mut c, schedule, _) = fixture(9);
    c.window = 2050;
    let engram = engram::HashConfig {
        layers: Vec::new(),
        ..engram::HashConfig::default()
    };
    let plan = CedPlan::shrinking(&c, &schedule, &engram, &CedRequest::cold(8192, 8192)).unwrap();
    let mut cursor = Cursor::new(&plan, &c, &schedule, 8192).unwrap();
    cursor.encoder_completed = 8192;
    cursor.publication_completed = 8192;
    let seed = plan.decoder[0].seed_rows.clone();
    assert_eq!(seed.len(), 2049);
    cursor.begin_seed(2, seed.start, 2048).unwrap();
    encoded(&mut cursor, false);
    assert!(cursor.chain_seed_to_decoder(2, seed.end, 2048).is_err());
    cursor.complete(Phase::Seed(2), seed.start, 2048).unwrap();
    cursor.begin_seed(2, seed.end - 1, 1).unwrap();
    encoded(&mut cursor, false);
    assert!(cursor.chain_seed_to_decoder(2, seed.end, 2048).is_err());
    cursor.complete(Phase::Seed(2), seed.end - 1, 1).unwrap();
    cursor.begin_decoder(2, seed.end, 2048).unwrap();

    let (c, schedule, plan) = fixture(2049);
    let mut cursor = Cursor::new(&plan, &c, &schedule, 2050).unwrap();
    cursor.encoder_completed = 2049;
    cursor.publication_completed = 2049;
    let seed = plan.decoder[0].seed_rows.clone();
    cursor.begin_seed(2, seed.start, seed.len()).unwrap();
    encoded(&mut cursor, false);
    assert!(cursor.chain_seed_to_decoder(2, seed.end, 3).is_err());
    cursor.chain_seed_to_decoder(2, seed.end, 2).unwrap();
    encoded(&mut cursor, false);
    cursor.complete(Phase::Decoder(2), seed.end, 2).unwrap();
    cursor.begin_decoder(2, 2048, 1).unwrap();
    encoded(&mut cursor, false);
    cursor.complete(Phase::Decoder(2), 2048, 1).unwrap();
    let seed = plan.decoder[1].seed_rows.clone();
    cursor.begin_seed(3, seed.start, seed.len()).unwrap();
    encoded(&mut cursor, false);
    assert!(cursor.chain_seed_to_decoder(3, 2048, 1).is_err());
    cursor
        .complete(Phase::Seed(3), seed.start, seed.len())
        .unwrap();
    cursor.begin_decoder(3, 2048, 1).unwrap();
}

#[test]
fn control_producers_keep_owner_masks_and_reindex_only_the_narrowed_subset() {
    let schedule = Schedule::default();
    let mut coverage = Coverage {
        origin: 16_380,
        selected: vec![None; 12],
        candidates: vec![None; 12],
    };
    assert!(coverage.validate(&(16_380..16_392), &schedule, 21).is_err());
    coverage.validate(&(16_380..16_392), &schedule, 20).unwrap();
    coverage.complete(16_380..16_392, &schedule, 20);
    coverage.validate(&(16_380..16_392), &schedule, 21).unwrap();
    coverage.validate(&(16_384..16_392), &schedule, 24).unwrap();
    coverage.complete(16_384..16_392, &schedule, 24);
    coverage.validate(&(16_384..16_392), &schedule, 25).unwrap();
    assert!(coverage.validate(&(16_383..16_392), &schedule, 25).is_err());
    coverage.complete(16_388..16_392, &schedule, 28);
    coverage.validate(&(16_388..16_392), &schedule, 29).unwrap();
    assert!(coverage.validate(&(16_387..16_392), &schedule, 29).is_err());
    assert!(
        coverage
            .candidates
            .iter()
            .all(|&producer| producer == Some(20))
    );
    assert_eq!(&coverage.selected[..4], &[Some(20); 4]);
    assert!(coverage.validate(&(16_379..16_381), &schedule, 20).is_err());
}

#[test]
fn short_and_window_one_plans_need_no_fabricated_seed() {
    for n in [1, 2, 3, 4, 5, 9, 2049] {
        let (mut c, schedule, plan) = fixture(n);
        Cursor::new(&plan, &c, &schedule, n + 1).unwrap();
        let mut invalid = plan.clone();
        invalid.decoder[0].seed_rows.start += 1;
        assert!(Cursor::new(&invalid, &c, &schedule, n + 1).is_err());
        c.window = 1;
        let engram = engram::HashConfig {
            layers: Vec::new(),
            ..engram::HashConfig::default()
        };
        let plan = CedPlan::shrinking(&c, &schedule, &engram, &CedRequest::cold(n, n + 1)).unwrap();
        let mut cursor = Cursor::new(&plan, &c, &schedule, n + 1).unwrap();
        cursor.encoder_completed = n;
        cursor.publication_completed = n;
        assert!(plan.decoder[0].seed_rows.is_empty());
        cursor.begin_decoder(2, n - 1, 1).unwrap();
    }
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn retained_controls_prune_at_16385_and_reindex_uses_owner_query_masks() {
    use super::super::super::toy;
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("CED GPU lease");
    let ctx = MetalContext::new().unwrap();
    let toy = toy::Weights::new(&ctx);
    let model = toy.model(&ctx);
    let mut c = model.config.clone();
    c.index_topk = 512;
    c.rope_dim = 0;
    c.candidate_block = 8;
    c.candidate_topk = 2048;
    let mut weights: Vec<_> = model.layers.iter().map(|l| l.attention.clone()).collect();
    let channel = |w: &mut Linear<'_>, output: usize| {
        let mut values = vec![0.0; w.input * w.output];
        values[output * w.input] = 1.0;
        w.tensor = upload(&ctx, &values, w.tensor.shape.clone()).unwrap();
    };
    for (id, w) in weights.iter_mut().enumerate() {
        w.frequencies.clear();
        w.candidate_block_size = 8;
        w.candidate_topk_blocks = 2048;
        if let Some(index) = &mut w.index {
            index.topk = 512;
            channel(&mut index.query, usize::from(id == 24));
            channel(&mut index.weights, 0);
        }
        channel(&mut w.query_a, 0);
    }
    let (n, capacity, start, rows) = (16_385usize, 16_386usize, 16_383usize, 2usize);
    let engram = engram::HashConfig {
        layers: Vec::new(),
        ..engram::HashConfig::default()
    };
    let plan = CedPlan::shrinking(
        &c,
        &Schedule::default(),
        &engram,
        &CedRequest::cold(n, capacity),
    )
    .unwrap();
    let mut w = Workspace::new(&ctx, &c, capacity, &weights.iter().collect::<Vec<_>>()).unwrap();
    let owner = w.schedule.owner_slot(20).unwrap();
    let mut keys = vec![0.0; capacity * c.index_dim];
    for i in 0..n {
        keys[i * c.index_dim] = if i < 8 { 0.0 } else { 1.0 };
        keys[i * c.index_dim + 1] = (capacity - i) as f32;
    }
    w.owners[owner].keys = upload(&ctx, &keys, w.owners[owner].keys.shape.clone()).unwrap();
    let immutable = f32_slice(&w.owners[owner].keys).unwrap().to_vec();
    let controls = QueryControls::new(&ctx, &c, capacity, plan.candidate_rows.clone()).unwrap();
    let mut x = vec![0.0; c.hidden * rows];
    for p in 0..rows {
        x[p * c.hidden] = 1.0;
    }
    let input = upload(&ctx, &x, vec![c.hidden as u64, rows as u64]).unwrap();
    let output = prefill::allocate(&ctx, c.hidden, rows, GgmlType::F32).unwrap();
    w.begin_ced(&plan).unwrap();
    // Selector-only fixture: inject completed frontiers and published synthetic
    // keys rather than running 16K encoder rows. This is not a full CED witness.
    {
        let cursor = w.ced.as_mut().unwrap();
        cursor.encoder_completed = n;
        cursor.publication_completed = n;
    }
    let read_ids = |t: &MetalTensor, count: usize| unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<i32>()
                .add((t.offset / 4) as usize),
            count,
        )
        .to_vec()
    };
    for id in [20, 24] {
        {
            let cursor = w.ced.as_mut().unwrap();
            let j = id - 20;
            cursor.decoder_next = j;
            cursor.seed_completed[j] = plan.decoder[j].seed_rows.end;
            cursor.output_completed[j] = start;
        }
        let view = controls.checked_tile_view(start, rows).unwrap();
        let scratch = Rows::with_controls(&ctx, &c, capacity, &view).unwrap();
        w.begin_ced_decoder_chunk(id, start, rows).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        w.encode_rows(&ctx, &enc, id, start, &weights[id], &input, &scratch)
            .unwrap();
        w.finish_rows(&ctx, &enc, &weights[id], &scratch, &output)
            .unwrap();
        enc.end();
        metal::commit_and_wait(&command).unwrap();
        w.complete_ced_decoder_chunk(id, start, rows).unwrap();
        let stride = capacity.div_ceil(c.candidate_block);
        let first = read_ids(&prefill::row(view.candidates(), stride, 0), 2048);
        assert!(first.iter().all(|&keep| keep == 1));
        let last = read_ids(&prefill::row(view.candidates(), stride, 1), 2049);
        assert_eq!(last[0], 0);
        assert!(last[1..].iter().all(|&keep| keep == 1));
        let first = read_ids(
            &prefill::row(view.selected(), c.index_topk, 0),
            c.index_topk,
        );
        let last = read_ids(
            &prefill::row(view.selected(), c.index_topk, 1),
            c.index_topk,
        );
        assert_eq!(
            first,
            if id == 24 {
                (0..512).collect::<Vec<_>>()
            } else {
                (8..520).collect()
            }
        );
        assert_eq!(last, (8..520).collect::<Vec<_>>());
        assert_eq!(f32_slice(&w.owners[owner].keys).unwrap(), immutable);
    }

    let seed_plan =
        CedPlan::shrinking(&c, &Schedule::default(), &engram, &CedRequest::cold(47, 48)).unwrap();
    let mut bad_seed = Workspace::new(&ctx, &c, 48, &weights.iter().collect::<Vec<_>>()).unwrap();
    bad_seed.begin_ced(&seed_plan).unwrap();
    let cursor = bad_seed.ced.as_mut().unwrap();
    cursor.encoder_completed = 47;
    cursor.publication_completed = 47;
    let start = seed_plan.decoder[0].seed_rows.start;
    bad_seed.begin_ced_seed(20, start, 1).unwrap();
    let nan = upload(&ctx, &vec![f32::NAN; c.hidden], vec![c.hidden as u64]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    bad_seed
        .seed_ced_scalar(&ctx, &enc, 20, start, &weights[20], &nan)
        .unwrap();
    enc.end();
    metal::commit_and_wait(&command).unwrap();
    assert!(bad_seed.complete_ced_seed(20, start, 1).is_err());
    assert_eq!(bad_seed.ced.as_ref().unwrap().seed_completed[0], start);
}
