//! Workspace capture, composition, reduction, and row/readout fitting.

use super::*;

pub(super) fn copy_workspace_token_capture(
    destination: &mut [f32],
    token_capture: &[f32],
    token: usize,
    n_tokens: usize,
    n_layers: usize,
    hidden_size: usize,
) -> Result<(), WorkspaceLensError> {
    let token_elements = checked_product(n_layers, hidden_size)?;
    let layer_elements = checked_product(n_tokens, hidden_size)?;
    let bank_elements = checked_product(n_layers, layer_elements)?;
    if token_capture.len() != token_elements {
        return Err(WorkspaceLensError::ActivationSize {
            name: "workspace token capture",
            got: token_capture.len(),
            expected: token_elements,
        });
    }
    if destination.len() != bank_elements {
        return Err(WorkspaceLensError::ActivationSize {
            name: "workspace layer-major destination",
            got: destination.len(),
            expected: bank_elements,
        });
    }
    if token >= n_tokens {
        return Err(WorkspaceLensError::ActivationSize {
            name: "workspace token index",
            got: token,
            expected: n_tokens,
        });
    }
    for layer in 0..n_layers {
        let source = checked_product(layer, hidden_size)?;
        let source_end = source
            .checked_add(hidden_size)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        let destination_row = checked_product(layer, n_tokens)?
            .checked_add(token)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        let destination_start = checked_product(destination_row, hidden_size)?;
        let destination_end = destination_start
            .checked_add(hidden_size)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        destination[destination_start..destination_end]
            .copy_from_slice(&token_capture[source..source_end]);
    }
    Ok(())
}

pub(super) fn compose_workspace_vjp(
    target_layer: u32,
    source_layers: &[u32],
    hidden_elements: usize,
    target_cotangent: &[f32],
    mut reverse_block: impl FnMut(
        u32,
        &[f32],
    ) -> Result<
        (Vec<f32>, WorkspaceLensReplayDiagnostic),
        WorkspaceLensError,
    >,
) -> Result<(Vec<f32>, Vec<WorkspaceLensReplayDiagnostic>), WorkspaceLensError> {
    validate_workspace_source_layers(target_layer, source_layers)?;
    if target_cotangent.len() != hidden_elements {
        return Err(WorkspaceLensError::ActivationSize {
            name: "workspace target cotangent",
            got: target_cotangent.len(),
            expected: hidden_elements,
        });
    }
    let output_elements = checked_product(source_layers.len(), hidden_elements)?;
    let mut values = try_zeroed_f32(output_elements, "workspace VJP result")?;
    let mut diagnostics = Vec::new();
    let earliest_source = source_layers
        .iter()
        .copied()
        .min()
        .ok_or(WorkspaceLensError::EmptyWorkspaceSourceLayers)?;
    let first_block = earliest_source
        .checked_add(1)
        .ok_or(WorkspaceLensError::SizeOverflow)?;
    let mut gradient = try_clone_slice(target_cotangent, "workspace VJP gradient")?;
    for layer in (first_block..=target_layer).rev() {
        let (next_gradient, diagnostic) = reverse_block(layer, &gradient)?;
        if next_gradient.len() != hidden_elements {
            return Err(WorkspaceLensError::ActivationSize {
                name: "workspace reversed block cotangent",
                got: next_gradient.len(),
                expected: hidden_elements,
            });
        }
        gradient = next_gradient;
        diagnostics.push(diagnostic);
        let crossed_source = layer - 1;
        for (slot, &source) in source_layers.iter().enumerate() {
            if source == crossed_source {
                let start = checked_product(slot, hidden_elements)?;
                let end = start
                    .checked_add(hidden_elements)
                    .ok_or(WorkspaceLensError::SizeOverflow)?;
                values[start..end].copy_from_slice(&gradient);
            }
        }
    }
    Ok((values, diagnostics))
}

pub(super) fn validate_workspace_vjp_finite(
    values: &[f32],
    diagnostics: &[WorkspaceLensReplayDiagnostic],
) -> Result<(), WorkspaceLensError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(WorkspaceLensError::NonFiniteWorkspaceVjpTrajectory { index });
    }
    if let Some(diagnostic) = diagnostics
        .iter()
        .find(|diagnostic| !diagnostic.residual_replay_max_abs_error.is_finite())
    {
        return Err(WorkspaceLensError::NonFiniteWorkspaceReplayDiagnostic {
            layer: diagnostic.layer,
        });
    }
    Ok(())
}

pub(super) fn build_workspace_target_bank(
    covectors: &[f32],
    n_query: usize,
    n_tokens: usize,
    hidden_size: usize,
    valid_positions: std::ops::Range<usize>,
) -> Result<Vec<f32>, WorkspaceLensError> {
    let covector_elements = checked_product(n_query, hidden_size)?;
    if covectors.len() != covector_elements {
        return Err(WorkspaceLensError::WorkspaceTargetCovectorSize {
            got: covectors.len(),
            hidden_size,
        });
    }
    if n_query == 0 {
        return Err(WorkspaceLensError::EmptyWorkspaceTargetCovectors);
    }
    if let Some(index) = covectors.iter().position(|value| !value.is_finite()) {
        return Err(WorkspaceLensError::NonFiniteWorkspaceTargetCovector { index });
    }
    if valid_positions.is_empty() || valid_positions.end > n_tokens {
        return Err(WorkspaceLensError::WorkspaceNoValidPositions {
            n_tokens,
            skip_first: valid_positions.start,
        });
    }
    let trajectory_elements = checked_product(n_tokens, hidden_size)?;
    let mut bank = try_zeroed_f32(
        checked_product(n_query, trajectory_elements)?,
        "workspace target cotangent bank",
    )?;
    for query in 0..n_query {
        let covector_start = checked_product(query, hidden_size)?;
        let covector_end = covector_start
            .checked_add(hidden_size)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        for position in valid_positions.clone() {
            let destination_start = checked_product(query, trajectory_elements)?
                .checked_add(checked_product(position, hidden_size)?)
                .ok_or(WorkspaceLensError::SizeOverflow)?;
            let destination_end = destination_start
                .checked_add(hidden_size)
                .ok_or(WorkspaceLensError::SizeOverflow)?;
            bank[destination_start..destination_end]
                .copy_from_slice(&covectors[covector_start..covector_end]);
        }
    }
    Ok(bank)
}

pub(super) fn reduce_workspace_source_positions(
    source: &[f32],
    n_tokens: usize,
    hidden_size: usize,
    valid_positions: std::ops::Range<usize>,
    destination: &mut [f32],
) -> Result<(), WorkspaceLensError> {
    let expected = checked_product(n_tokens, hidden_size)?;
    if source.len() != expected {
        return Err(WorkspaceLensError::ActivationSize {
            name: "workspace source trajectory reduction",
            got: source.len(),
            expected,
        });
    }
    if destination.len() != hidden_size {
        return Err(WorkspaceLensError::ActivationSize {
            name: "workspace fitted row destination",
            got: destination.len(),
            expected: hidden_size,
        });
    }
    let count = valid_positions.len();
    if count == 0 || valid_positions.end > n_tokens {
        return Err(WorkspaceLensError::WorkspaceNoValidPositions {
            n_tokens,
            skip_first: valid_positions.start,
        });
    }
    destination.fill(0.0);
    for position in valid_positions {
        let start = checked_product(position, hidden_size)?;
        let end = start
            .checked_add(hidden_size)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        for (column, (sum, &value)) in destination.iter_mut().zip(&source[start..end]).enumerate() {
            let source_index = start
                .checked_add(column)
                .ok_or(WorkspaceLensError::SizeOverflow)?;
            if !value.is_finite() {
                return Err(WorkspaceLensError::NonFiniteWorkspaceVjpTrajectory {
                    index: source_index,
                });
            }
            *sum += value;
            if !sum.is_finite() {
                return Err(WorkspaceLensError::NonFiniteWorkspaceReduction {
                    stage: "sum",
                    index: column,
                });
            }
        }
    }
    let scale = (count as f32).recip();
    for (index, value) in destination.iter_mut().enumerate() {
        *value *= scale;
        if !value.is_finite() {
            return Err(WorkspaceLensError::NonFiniteWorkspaceReduction {
                stage: "scaled output",
                index,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_capture_layers(
    n_layers: u32,
    capture_layers: &[u32],
) -> Result<(), WorkspaceLensError> {
    if let Some(&layer) = capture_layers.iter().find(|&&layer| layer >= n_layers) {
        return Err(WorkspaceLensError::InvalidLayer { layer, n_layers });
    }
    Ok(())
}

impl<'model, 'sequence> WorkspaceLensSession<'model, 'sequence> {
    /// Advance one fresh bounded prompt once while capturing every block's
    /// post-mixer and post-block residuals. The final RMSNorm and LM head are
    /// skipped because reference J/R-lens fitting targets a block residual. A
    /// failure after a successful token poisons the partially consumed
    /// sequence rather than exposing stitchable state.
    pub fn forward_prompt_with_workspace_capture(
        &mut self,
        token_ids: &[i32],
    ) -> Result<WorkspaceLensPromptForward, WorkspaceLensError> {
        if token_ids.is_empty() {
            return Err(WorkspaceLensError::EmptyWorkspacePrompt);
        }
        if token_ids.len() > MAX_WORKSPACE_LENS_TOKENS {
            return Err(WorkspaceLensError::WorkspacePromptTooLong {
                got: token_ids.len(),
                max: MAX_WORKSPACE_LENS_TOKENS,
            });
        }
        if self.sequence.position() != 0 {
            return Err(WorkspaceLensError::WorkspaceCaptureRequiresFreshSequence(
                self.sequence.position(),
            ));
        }
        let arch = self.arch();
        for &token_id in token_ids {
            if token_id < 0 || token_id as u32 >= arch.vocab_size {
                return Err(MfError::BadToken(token_id, arch.vocab_size).into());
            }
        }
        self.sequence.ensure_can_append(token_ids.len())?;
        let last_position = token_ids.len() - 1;
        u32::try_from(last_position)
            .map_err(|_| WorkspaceLensError::PositionOverflow(last_position))?;
        self.validate_workspace_weights()?;

        let n_layers =
            usize::try_from(arch.n_layer).map_err(|_| WorkspaceLensError::SizeOverflow)?;
        let hidden_size = arch.hidden_size as usize;
        let layer_elements = checked_product(token_ids.len(), hidden_size)?;
        let bank_elements = checked_product(n_layers, layer_elements)?;
        let capture_layers: Vec<u32> = (0..arch.n_layer).collect();
        let mut post_mixer_residuals = vec![0.0f32; bank_elements];
        let mut post_block_residuals = vec![0.0f32; bank_elements];
        for (token, &token_id) in token_ids.iter().enumerate() {
            let capture = match self
                .forward_token_with_dense_ffn_capture_no_tail(token_id, &capture_layers)
            {
                Ok(capture) => capture,
                Err(error) => {
                    let state = unsafe { self.sequence.metal_session_mut() };
                    state.poison("workspace prompt capture forward failed");
                    return Err(error);
                }
            };
            copy_workspace_token_capture(
                &mut post_mixer_residuals,
                &capture.pre_ffn_residuals,
                token,
                token_ids.len(),
                n_layers,
                hidden_size,
            )?;
            copy_workspace_token_capture(
                &mut post_block_residuals,
                &capture.post_block_residuals,
                token,
                token_ids.len(),
                n_layers,
                hidden_size,
            )?;
        }
        Ok(WorkspaceLensPromptForward {
            identity: self.identity(),
            owner_token_id: self.model.owner_token_id(),
            token_ids: token_ids.to_vec(),
            n_layers: arch.n_layer,
            hidden_size,
            post_mixer_residuals,
            post_block_residuals,
        })
    }

    /// Apply one target-layer cotangent trajectory to every requested source
    /// layer using the reference current-and-future-position VJP semantics.
    /// The caller performs any source-position reduction (for example, the
    /// paper's mean over valid positions) on the returned `[K,T,H]` values.
    pub fn workspace_vjp(
        &self,
        forward: &WorkspaceLensPromptForward,
        target_layer: u32,
        source_layers: &[u32],
        target_cotangent: &[f32],
        rule: WorkspaceLensRule,
    ) -> Result<WorkspaceLensVjp, WorkspaceLensError> {
        let (arch, n_tokens, hidden_elements) =
            self.validate_workspace_vjp_forward(forward, target_layer)?;
        if target_cotangent.len() != hidden_elements {
            return Err(WorkspaceLensError::ActivationSize {
                name: "workspace target cotangent",
                got: target_cotangent.len(),
                expected: hidden_elements,
            });
        }
        let (values, diagnostics) = compose_workspace_vjp(
            target_layer,
            source_layers,
            hidden_elements,
            target_cotangent,
            |layer, grad_output| {
                let input = forward.input_residuals(layer).ok_or(
                    WorkspaceLensError::WorkspaceSourceNotBeforeTarget {
                        source_layer: layer,
                        target_layer,
                    },
                )?;
                let post_mixer = forward.post_mixer_residuals(layer).ok_or(
                    WorkspaceLensError::InvalidLayer {
                        layer,
                        n_layers: arch.n_layer,
                    },
                )?;
                self.workspace_block_vjp(layer, input, post_mixer, grad_output, n_tokens, rule)
            },
        )?;
        validate_workspace_vjp_finite(&values, &diagnostics)?;
        Ok(WorkspaceLensVjp {
            target_layer,
            source_layers: try_clone_slice(source_layers, "workspace VJP source layers")?,
            n_tokens,
            hidden_size: forward.hidden_size,
            values,
            diagnostics,
        })
    }

    /// Apply a query-major bank of target-layer cotangent trajectories while
    /// sharing each block's dense-FFN and mixer primal replay. Returned values
    /// use `[K,Q,T,H]` order.
    pub fn workspace_vjp_batch(
        &self,
        forward: &WorkspaceLensPromptForward,
        target_layer: u32,
        source_layers: &[u32],
        target_cotangents: &[f32],
        n_query: usize,
        rule: WorkspaceLensRule,
    ) -> Result<WorkspaceLensVjpBatch, WorkspaceLensError> {
        if n_query == 0 {
            return Err(WorkspaceLensError::EmptyQueryBatch);
        }
        if n_query > MAX_WORKSPACE_LENS_DIM_BATCH {
            return Err(WorkspaceLensError::WorkspaceQueryBatchTooLarge {
                got: n_query,
                max: MAX_WORKSPACE_LENS_DIM_BATCH,
            });
        }
        let (_arch, n_tokens, hidden_elements) =
            self.validate_workspace_vjp_forward(forward, target_layer)?;
        let query_elements = checked_product(n_query, hidden_elements)?;
        if target_cotangents.len() != query_elements {
            return Err(WorkspaceLensError::ActivationSize {
                name: "workspace target cotangent query bank",
                got: target_cotangents.len(),
                expected: query_elements,
            });
        }
        let (values, diagnostics) = compose_workspace_vjp(
            target_layer,
            source_layers,
            query_elements,
            target_cotangents,
            |layer, grad_outputs| {
                let input = forward.input_residuals(layer).ok_or(
                    WorkspaceLensError::WorkspaceSourceNotBeforeTarget {
                        source_layer: layer,
                        target_layer,
                    },
                )?;
                let post_mixer = forward.post_mixer_residuals(layer).ok_or(
                    WorkspaceLensError::InvalidLayer {
                        layer,
                        n_layers: forward.n_layers,
                    },
                )?;
                self.workspace_block_vjp_batch(
                    layer,
                    input,
                    post_mixer,
                    grad_outputs,
                    n_tokens,
                    n_query,
                    rule,
                )
            },
        )?;
        validate_workspace_vjp_finite(&values, &diagnostics)?;
        Ok(WorkspaceLensVjpBatch {
            target_layer,
            source_layers: try_clone_slice(source_layers, "workspace VJP batch source layers")?,
            n_query,
            n_tokens,
            hidden_size: forward.hidden_size,
            values,
            diagnostics,
        })
    }

    /// Fit selected rows of the reference current-and-future-position
    /// transport estimator for one captured prompt.
    ///
    /// For each output coordinate, the cotangent is one at every valid target
    /// position `skip_first..T-1` (the final prompt position is excluded). The
    /// resulting source trajectories are averaged over those same positions,
    /// with no second normalization over target positions.
    pub fn workspace_fit_rows(
        &self,
        forward: &WorkspaceLensPromptForward,
        target_layer: u32,
        source_layers: &[u32],
        output_rows: &[u32],
        skip_first: usize,
        rule: WorkspaceLensRule,
    ) -> Result<WorkspaceLensRows, WorkspaceLensError> {
        if output_rows.is_empty() {
            return Err(WorkspaceLensError::EmptyWorkspaceOutputRows);
        }
        if output_rows.windows(2).any(|rows| rows[0] >= rows[1]) {
            return Err(WorkspaceLensError::WorkspaceOutputRowsNotStrict);
        }
        let hidden_size = forward.hidden_size();
        if let Some(&row) = output_rows.iter().find(|&&row| row as usize >= hidden_size) {
            return Err(WorkspaceLensError::WorkspaceOutputRowOutOfRange { row, hidden_size });
        }
        let valid_positions = workspace_valid_position_range(forward.n_tokens(), skip_first)?;
        let n_valid_positions = valid_positions.len();
        let hidden_elements = checked_product(forward.n_tokens(), hidden_size)?;
        let source_row_elements = checked_product(output_rows.len(), hidden_size)?;
        let output_elements = checked_product(source_layers.len(), source_row_elements)?;
        let mut values = vec![0.0f32; output_elements];
        let mut diagnostics = Vec::new();
        let mut target_cotangent = vec![0.0f32; hidden_elements];
        for (row_slot, &row) in output_rows.iter().enumerate() {
            target_cotangent.fill(0.0);
            for position in valid_positions.clone() {
                let offset = checked_product(position, hidden_size)?
                    .checked_add(row as usize)
                    .ok_or(WorkspaceLensError::SizeOverflow)?;
                target_cotangent[offset] = 1.0;
            }
            let vjp = self.workspace_vjp(
                forward,
                target_layer,
                source_layers,
                &target_cotangent,
                rule,
            )?;
            merge_workspace_diagnostics(&mut diagnostics, &vjp.diagnostics)?;
            for source_slot in 0..source_layers.len() {
                let source =
                    vjp.source_values(source_slot)
                        .ok_or(WorkspaceLensError::ActivationSize {
                            name: "workspace fitted source trajectory",
                            got: vjp.values.len(),
                            expected: checked_product(source_layers.len(), hidden_elements)?,
                        })?;
                let destination_row = checked_product(source_slot, output_rows.len())?
                    .checked_add(row_slot)
                    .ok_or(WorkspaceLensError::SizeOverflow)?;
                let destination_start = checked_product(destination_row, hidden_size)?;
                let destination_end = destination_start
                    .checked_add(hidden_size)
                    .ok_or(WorkspaceLensError::SizeOverflow)?;
                reduce_workspace_source_positions(
                    source,
                    forward.n_tokens(),
                    hidden_size,
                    valid_positions.clone(),
                    &mut values[destination_start..destination_end],
                )?;
            }
        }
        Ok(WorkspaceLensRows {
            target_layer,
            source_layers: source_layers.to_vec(),
            output_rows: output_rows.to_vec(),
            n_tokens: forward.n_tokens(),
            n_valid_positions,
            hidden_size,
            values,
            diagnostics,
        })
    }

    /// Fit rows in query-major execution batches while preserving the public
    /// row-shard orientation `[K,R,H]` and the reference estimator exactly.
    pub fn workspace_fit_rows_batched(
        &self,
        forward: &WorkspaceLensPromptForward,
        target_layer: u32,
        source_layers: &[u32],
        output_rows: &[u32],
        skip_first: usize,
        dim_batch: usize,
        rule: WorkspaceLensRule,
    ) -> Result<WorkspaceLensRows, WorkspaceLensError> {
        if dim_batch == 0 {
            return Err(WorkspaceLensError::EmptyQueryBatch);
        }
        if dim_batch > MAX_WORKSPACE_LENS_DIM_BATCH {
            return Err(WorkspaceLensError::WorkspaceQueryBatchTooLarge {
                got: dim_batch,
                max: MAX_WORKSPACE_LENS_DIM_BATCH,
            });
        }
        if output_rows.is_empty() {
            return Err(WorkspaceLensError::EmptyWorkspaceOutputRows);
        }
        if output_rows.windows(2).any(|rows| rows[0] >= rows[1]) {
            return Err(WorkspaceLensError::WorkspaceOutputRowsNotStrict);
        }
        let hidden_size = forward.hidden_size();
        if let Some(&row) = output_rows.iter().find(|&&row| row as usize >= hidden_size) {
            return Err(WorkspaceLensError::WorkspaceOutputRowOutOfRange { row, hidden_size });
        }
        let valid_positions = workspace_valid_position_range(forward.n_tokens(), skip_first)?;
        let n_valid_positions = valid_positions.len();
        let hidden_elements = checked_product(forward.n_tokens(), hidden_size)?;
        let source_row_elements = checked_product(output_rows.len(), hidden_size)?;
        let output_elements = checked_product(source_layers.len(), source_row_elements)?;
        let mut values = vec![0.0f32; output_elements];
        let mut diagnostics = Vec::new();
        let mut first_row_slot = 0usize;
        for rows in output_rows.chunks(dim_batch) {
            let n_query = rows.len();
            let mut target_cotangents = vec![0.0f32; checked_product(n_query, hidden_elements)?];
            for (query_slot, &row) in rows.iter().enumerate() {
                let query_start = checked_product(query_slot, hidden_elements)?;
                for position in valid_positions.clone() {
                    let offset = query_start
                        .checked_add(checked_product(position, hidden_size)?)
                        .and_then(|offset| offset.checked_add(row as usize))
                        .ok_or(WorkspaceLensError::SizeOverflow)?;
                    target_cotangents[offset] = 1.0;
                }
            }
            let vjp = self.workspace_vjp_batch(
                forward,
                target_layer,
                source_layers,
                &target_cotangents,
                n_query,
                rule,
            )?;
            merge_workspace_diagnostics(&mut diagnostics, &vjp.diagnostics)?;
            for source_slot in 0..source_layers.len() {
                for query_slot in 0..n_query {
                    let source = vjp.source_query_values(source_slot, query_slot).ok_or(
                        WorkspaceLensError::ActivationSize {
                            name: "workspace fitted source trajectory query",
                            got: vjp.values.len(),
                            expected: checked_product(
                                source_layers.len(),
                                checked_product(n_query, hidden_elements)?,
                            )?,
                        },
                    )?;
                    let row_slot = first_row_slot
                        .checked_add(query_slot)
                        .ok_or(WorkspaceLensError::SizeOverflow)?;
                    let destination_row = checked_product(source_slot, output_rows.len())?
                        .checked_add(row_slot)
                        .ok_or(WorkspaceLensError::SizeOverflow)?;
                    let destination_start = checked_product(destination_row, hidden_size)?;
                    let destination_end = destination_start
                        .checked_add(hidden_size)
                        .ok_or(WorkspaceLensError::SizeOverflow)?;
                    reduce_workspace_source_positions(
                        source,
                        forward.n_tokens(),
                        hidden_size,
                        valid_positions.clone(),
                        &mut values[destination_start..destination_end],
                    )?;
                }
            }
            first_row_slot = first_row_slot
                .checked_add(n_query)
                .ok_or(WorkspaceLensError::SizeOverflow)?;
        }
        Ok(WorkspaceLensRows {
            target_layer,
            source_layers: source_layers.to_vec(),
            output_rows: output_rows.to_vec(),
            n_tokens: forward.n_tokens(),
            n_valid_positions,
            hidden_size,
            values,
            diagnostics,
        })
    }

    pub(super) fn validate_workspace_vjp_forward(
        &self,
        forward: &WorkspaceLensPromptForward,
        target_layer: u32,
    ) -> Result<(Arch, usize, usize), WorkspaceLensError> {
        if forward.identity != self.identity() {
            return Err(WorkspaceLensError::WorkspaceCaptureModelMismatch);
        }
        if forward.owner_token_id != self.model.owner_token_id() {
            return Err(WorkspaceLensError::WorkspaceCaptureOwnerMismatch);
        }
        let arch = self.arch();
        if target_layer >= arch.n_layer {
            return Err(WorkspaceLensError::InvalidLayer {
                layer: target_layer,
                n_layers: arch.n_layer,
            });
        }
        let n_tokens = forward.n_tokens();
        if n_tokens == 0 || n_tokens > MAX_WORKSPACE_LENS_TOKENS {
            return Err(WorkspaceLensError::WorkspacePromptTooLong {
                got: n_tokens,
                max: MAX_WORKSPACE_LENS_TOKENS,
            });
        }
        if forward.n_layers != arch.n_layer {
            return Err(WorkspaceLensError::ActivationSize {
                name: "workspace capture layer count",
                got: forward.n_layers as usize,
                expected: arch.n_layer as usize,
            });
        }
        if forward.hidden_size != arch.hidden_size as usize {
            return Err(WorkspaceLensError::ActivationSize {
                name: "workspace capture hidden size",
                got: forward.hidden_size,
                expected: arch.hidden_size as usize,
            });
        }
        let hidden_elements = checked_product(n_tokens, forward.hidden_size)?;
        let bank_elements = checked_product(arch.n_layer as usize, hidden_elements)?;
        for (name, values) in [
            (
                "workspace post-mixer residual bank",
                forward.post_mixer_residuals.as_slice(),
            ),
            (
                "workspace post-block residual bank",
                forward.post_block_residuals.as_slice(),
            ),
        ] {
            if values.len() != bank_elements {
                return Err(WorkspaceLensError::ActivationSize {
                    name,
                    got: values.len(),
                    expected: bank_elements,
                });
            }
        }
        Ok((arch, n_tokens, hidden_elements))
    }
}
