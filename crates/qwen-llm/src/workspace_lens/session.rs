//! Session construction, identity, and linear inventory.

use super::*;

impl<'model, 'sequence> WorkspaceLensSession<'model, 'sequence> {
    pub fn arch(&self) -> Arch {
        self.model.arch()
    }

    pub fn identity(&self) -> WorkspaceLensModelIdentity {
        self.model.workspace_lens_identity()
    }

    pub fn linear_info(
        &self,
        id: WorkspaceLensLinear,
    ) -> Result<WorkspaceLensLinearInfo, WorkspaceLensError> {
        let tensor = self.resolve_linear(id)?;
        let shape = linear_shape(id, tensor)?;
        Ok(WorkspaceLensLinearInfo {
            id,
            dtype: tensor.dtype,
            shape,
        })
    }

    pub fn linears(&self) -> Result<Vec<WorkspaceLensLinearInfo>, WorkspaceLensError> {
        let mut ids = vec![WorkspaceLensLinear::LmHead];
        for (index, block) in self.model.metal_model().blocks.iter().enumerate() {
            let index = u32::try_from(index).map_err(|_| WorkspaceLensError::SizeOverflow)?;
            for role in [LinearRole::FfnGate, LinearRole::FfnUp, LinearRole::FfnDown] {
                ids.push(WorkspaceLensLinear::Layer { index, role });
            }
            match block {
                MetalBlock::Gdn(_) => {
                    for role in [
                        LinearRole::GdnQkv,
                        LinearRole::GdnZ,
                        LinearRole::GdnBeta,
                        LinearRole::GdnAlpha,
                        LinearRole::GdnOut,
                    ] {
                        ids.push(WorkspaceLensLinear::Layer { index, role });
                    }
                }
                MetalBlock::Attn(_) => {
                    for role in [
                        LinearRole::AttentionQAndGate,
                        LinearRole::AttentionK,
                        LinearRole::AttentionV,
                        LinearRole::AttentionOut,
                    ] {
                        ids.push(WorkspaceLensLinear::Layer { index, role });
                    }
                }
            }
        }
        ids.into_iter().map(|id| self.linear_info(id)).collect()
    }

    /// Consume a fresh prompt without running the final norm or LM head and
    /// capture post-block residuals for its final token in caller layer order.
    pub fn forward_prompt_last_post_block_residuals(
        &mut self,
        token_ids: &[i32],
        capture_layers: &[u32],
    ) -> Result<WorkspaceLensPromptLastCapture, WorkspaceLensError> {
        if token_ids.is_empty() {
            return Err(WorkspaceLensError::EmptyFullReadoutPrompt);
        }
        if self.sequence.position() != 0 {
            return Err(WorkspaceLensError::FullReadoutRequiresFreshSequence(
                self.sequence.position(),
            ));
        }
        if capture_layers.is_empty() {
            return Err(WorkspaceLensError::EmptyWorkspaceSourceLayers);
        }
        let arch = self.arch();
        validate_capture_layers(arch.n_layer, capture_layers)?;
        for &token_id in token_ids {
            if token_id < 0 || token_id as u32 >= arch.vocab_size {
                return Err(MfError::BadToken(token_id, arch.vocab_size).into());
            }
        }
        self.sequence.ensure_can_append(token_ids.len())?;
        let hidden_size = arch.hidden_size as usize;
        let capture_len = checked_product(capture_layers.len(), hidden_size)?;
        let capture = MetalTensor::zeros_f32(
            self.model.context(),
            vec![u64::try_from(capture_len).map_err(|_| WorkspaceLensError::SizeOverflow)?],
        )?;
        let forward = self.model.forward();
        for (position, &token_id) in token_ids.iter().enumerate() {
            let position_u32 = u32::try_from(position)
                .map_err(|_| WorkspaceLensError::PositionOverflow(position))?;
            let state = unsafe { self.sequence.metal_session_mut() };
            state.ensure_usable()?;
            let result = if position + 1 == token_ids.len() {
                forward.single_token_with_multi_hidden_no_tail(
                    token_id,
                    position_u32,
                    state,
                    capture_layers,
                    &capture,
                )
            } else {
                forward.single_token_no_tail(token_id, position_u32, state)
            };
            if let Err(error) = result {
                state.poison("full-vocabulary prompt capture forward failed");
                return Err(error.into());
            }
            self.sequence.advance_by(1)?;
        }
        let position = token_ids.len() - 1;
        Ok(WorkspaceLensPromptLastCapture {
            position,
            token_id: token_ids[position],
            capture: ActivationCapture {
                layer_ids: try_clone_slice(capture_layers, "full readout capture layers")?,
                hidden_size,
                values: read_f32_fallible(
                    &capture,
                    capture_len,
                    "full readout post-block residuals",
                )?,
            },
        })
    }

    /// Advance one token and return post-block residuals in caller layer order.
    pub fn forward_token(
        &mut self,
        token_id: i32,
        capture_layers: &[u32],
    ) -> Result<WorkspaceLensForward, WorkspaceLensError> {
        self.sequence.ensure_can_append(1)?;
        validate_capture_layers(self.arch().n_layer, capture_layers)?;
        let position = self.sequence.position();
        let position_u32 =
            u32::try_from(position).map_err(|_| WorkspaceLensError::PositionOverflow(position))?;
        let hidden_size = self.arch().hidden_size as usize;
        let capture_len = capture_layers
            .len()
            .checked_mul(hidden_size)
            .ok_or(WorkspaceLensError::SizeOverflow)?;

        let forward = self.model.forward();
        let state = unsafe { self.sequence.metal_session_mut() };
        state.ensure_usable()?;
        let (logits, values) = if capture_layers.is_empty() {
            (
                forward.single_token(token_id, position_u32, state)?,
                Vec::new(),
            )
        } else {
            let capture = MetalTensor::zeros_f32(self.model.context(), vec![capture_len as u64])?;
            let logits = forward.single_token_with_multi_hidden(
                token_id,
                position_u32,
                state,
                capture_layers,
                &capture,
            )?;
            (logits, read_f32(&capture, capture_len))
        };
        self.sequence.advance_by(1)?;

        Ok(WorkspaceLensForward {
            position,
            token_id,
            logits,
            capture: ActivationCapture {
                layer_ids: capture_layers.to_vec(),
                hidden_size,
                values,
            },
        })
    }

    /// Apply the exact activation VJP of a supported frozen resident linear map.
    pub fn frozen_linear_vjp(
        &mut self,
        id: WorkspaceLensLinear,
        grad_output: &[f32],
        n_query: usize,
    ) -> Result<Vec<f32>, WorkspaceLensError> {
        if n_query == 0 {
            return Err(WorkspaceLensError::EmptyQueryBatch);
        }
        let weight = self.resolve_linear(id)?;
        let [n_in, n_out] = linear_shape(id, weight)?;
        if !matches!(
            weight.dtype,
            GgmlType::Q8_0 | GgmlType::BF16 | GgmlType::F16 | GgmlType::F32
        ) {
            return Err(WorkspaceLensError::UnsupportedLinearDtype {
                id,
                dtype: weight.dtype,
            });
        }
        let expected = n_query
            .checked_mul(n_out)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        if grad_output.len() != expected {
            return Err(WorkspaceLensError::CotangentSize {
                got: grad_output.len(),
                expected,
                n_query,
                n_out,
            });
        }
        let grad_output = MetalTensor::from_bytes(
            self.model.context(),
            bytemuck::cast_slice(grad_output),
            vec![n_out as u64, n_query as u64],
            GgmlType::F32,
        )?;
        let grad_input =
            MetalTensor::zeros_f32(self.model.context(), vec![n_in as u64, n_query as u64])?;
        let command = self
            .model
            .context()
            .queue
            .commandBuffer()
            .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
        let encoder = KernelEncoder::begin(&command);
        let encode_result = encode_frozen_linear_vjp_f32(
            self.model.context(),
            &encoder,
            weight,
            &grad_output,
            &grad_input,
            n_in,
            n_out,
            n_query,
        );
        encoder.end();
        encode_result?;
        command.commit();
        command.waitUntilCompleted();
        let status = command.status();
        let error = command.error();
        if status != MTLCommandBufferStatus::Completed || error.is_some() {
            return Err(WorkspaceLensError::CommandBuffer {
                status: format!("{status:?}"),
                error: format!("{error:?}"),
            });
        }
        let output_len = n_query
            .checked_mul(n_in)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        Ok(read_f32(&grad_input, output_len))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn workspace_block_vjp(
        &self,
        layer: u32,
        input_residuals: &[f32],
        post_mixer_residuals: &[f32],
        grad_block_output: &[f32],
        n_tokens: usize,
        rule: WorkspaceLensRule,
    ) -> Result<(Vec<f32>, WorkspaceLensReplayDiagnostic), WorkspaceLensError> {
        let arch = self.arch();
        let hidden_size = arch.hidden_size as usize;
        let hidden_elements = checked_product(n_tokens, hidden_size)?;
        for (name, values) in [
            ("workspace block inputs", input_residuals),
            ("workspace post-mixer residuals", post_mixer_residuals),
            ("workspace block cotangent", grad_block_output),
        ] {
            if values.len() != hidden_elements {
                return Err(WorkspaceLensError::ActivationSize {
                    name,
                    got: values.len(),
                    expected: hidden_elements,
                });
            }
        }
        let block = self.model.metal_model().blocks.get(layer as usize).ok_or(
            WorkspaceLensError::InvalidLayer {
                layer,
                n_layers: arch.n_layer,
            },
        )?;
        let (post_norm, gate, up, down) = match block {
            MetalBlock::Gdn(block) => (
                &block.post_attn_norm,
                &block.ffn_gate,
                &block.ffn_up,
                &block.ffn_down,
            ),
            MetalBlock::Attn(block) => (
                &block.post_attn_norm,
                &block.ffn_gate,
                &block.ffn_up,
                &block.ffn_down,
            ),
        };
        let grad_post_mixer = dense_ffn_vjp_rows_readback(
            self.model.context(),
            layer,
            hidden_size,
            arch.intermediate_size as usize,
            post_mixer_residuals,
            post_norm,
            gate,
            up,
            down,
            grad_block_output,
            n_tokens,
            match rule {
                WorkspaceLensRule::Jacobian => DenseFfnVjpRule::Jacobian,
                WorkspaceLensRule::Relp => DenseFfnVjpRule::Relp,
            },
        )?;
        let (mixer_outputs, grad_mixer_input, kind) = match block {
            MetalBlock::Gdn(block) => {
                let geometry = GdnGeometry::new(layer, arch)?;
                validate_gdn_weights(layer, block, geometry)?;
                let initial_conv_state = vec![0.0f32; geometry.conv_state_elements];
                let initial_recurrence_state = vec![0.0f32; geometry.state_elements];
                let replay = gdn_mixer_replay_vjp_readback(
                    self.model.context(),
                    geometry,
                    GdnMixerWeights::from(block),
                    input_residuals,
                    &initial_conv_state,
                    &initial_recurrence_state,
                    &grad_post_mixer,
                    n_tokens,
                    match rule {
                        WorkspaceLensRule::Jacobian => GdnMixerVjpRule::Jacobian,
                        WorkspaceLensRule::Relp => GdnMixerVjpRule::Relp,
                    },
                    false,
                )?;
                (
                    replay.mixer_outputs,
                    replay.grad_input,
                    WorkspaceLensBlockKind::Gdn,
                )
            }
            MetalBlock::Attn(block) => {
                let geometry = AttnGeometry::new(arch)?;
                validate_attn_weights(layer, block, geometry)?;
                let replay = attn_mixer_replay_vjp_readback(
                    self.model.context(),
                    geometry,
                    AttnMixerWeights::from(block),
                    input_residuals,
                    &grad_post_mixer,
                    n_tokens,
                    match rule {
                        WorkspaceLensRule::Jacobian => AttnBlockVjpRule::Jacobian,
                        WorkspaceLensRule::Relp => AttnBlockVjpRule::Relp,
                    },
                )?;
                (
                    replay.mixer_outputs,
                    replay.grad_input,
                    WorkspaceLensBlockKind::Attention,
                )
            }
        };
        let residual_replay_max_abs_error = input_residuals
            .iter()
            .zip(&mixer_outputs)
            .zip(post_mixer_residuals)
            .map(|((&input, &mixer), &observed)| finite_abs_difference(input + mixer, observed))
            .fold(0.0f32, f32::max);
        if grad_mixer_input.len() != grad_post_mixer.len() {
            return Err(WorkspaceLensError::ActivationSize {
                name: "workspace mixer branch cotangent",
                got: grad_mixer_input.len(),
                expected: grad_post_mixer.len(),
            });
        }
        let values = grad_post_mixer
            .iter()
            .zip(grad_mixer_input)
            .map(|(&identity, branch)| identity + branch)
            .collect();
        Ok((
            values,
            WorkspaceLensReplayDiagnostic {
                layer,
                kind,
                residual_replay_max_abs_error,
            },
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn workspace_block_vjp_batch(
        &self,
        layer: u32,
        input_residuals: &[f32],
        post_mixer_residuals: &[f32],
        grad_block_outputs: &[f32],
        n_tokens: usize,
        n_query: usize,
        rule: WorkspaceLensRule,
    ) -> Result<(Vec<f32>, WorkspaceLensReplayDiagnostic), WorkspaceLensError> {
        if n_query == 0 {
            return Err(WorkspaceLensError::EmptyQueryBatch);
        }
        let arch = self.arch();
        let hidden_size = arch.hidden_size as usize;
        let hidden_elements = checked_product(n_tokens, hidden_size)?;
        let query_elements = checked_product(n_query, hidden_elements)?;
        for (name, values, expected) in [
            ("workspace block inputs", input_residuals, hidden_elements),
            (
                "workspace post-mixer residuals",
                post_mixer_residuals,
                hidden_elements,
            ),
            (
                "workspace block cotangent query bank",
                grad_block_outputs,
                query_elements,
            ),
        ] {
            if values.len() != expected {
                return Err(WorkspaceLensError::ActivationSize {
                    name,
                    got: values.len(),
                    expected,
                });
            }
        }
        let block = self.model.metal_model().blocks.get(layer as usize).ok_or(
            WorkspaceLensError::InvalidLayer {
                layer,
                n_layers: arch.n_layer,
            },
        )?;
        let (post_norm, gate, up, down) = match block {
            MetalBlock::Gdn(block) => (
                &block.post_attn_norm,
                &block.ffn_gate,
                &block.ffn_up,
                &block.ffn_down,
            ),
            MetalBlock::Attn(block) => (
                &block.post_attn_norm,
                &block.ffn_gate,
                &block.ffn_up,
                &block.ffn_down,
            ),
        };
        let grad_post_mixer = dense_ffn_vjp_query_rows_readback(
            self.model.context(),
            layer,
            hidden_size,
            arch.intermediate_size as usize,
            post_mixer_residuals,
            post_norm,
            gate,
            up,
            down,
            grad_block_outputs,
            n_tokens,
            n_query,
            match rule {
                WorkspaceLensRule::Jacobian => DenseFfnVjpRule::Jacobian,
                WorkspaceLensRule::Relp => DenseFfnVjpRule::Relp,
            },
        )?;

        let (grad_mixer_input, residual_replay_max_abs_error, kind) = match block {
            MetalBlock::Gdn(block) => {
                let geometry = GdnGeometry::new(layer, arch)?;
                validate_gdn_weights(layer, block, geometry)?;
                let initial_conv_state = vec![0.0f32; geometry.conv_state_elements];
                let initial_recurrence_state = vec![0.0f32; geometry.state_elements];
                let replay = gdn_mixer_replay_vjp_batch_readback(
                    self.model.context(),
                    geometry,
                    GdnMixerWeights::from(block),
                    input_residuals,
                    &initial_conv_state,
                    &initial_recurrence_state,
                    &grad_post_mixer,
                    n_tokens,
                    n_query,
                    match rule {
                        WorkspaceLensRule::Jacobian => GdnMixerVjpRule::Jacobian,
                        WorkspaceLensRule::Relp => GdnMixerVjpRule::Relp,
                    },
                    false,
                )?;
                let residual_replay_max_abs_error = input_residuals
                    .iter()
                    .zip(&replay.mixer_outputs)
                    .zip(post_mixer_residuals)
                    .map(|((&input, &mixer), &observed)| {
                        finite_abs_difference(input + mixer, observed)
                    })
                    .fold(0.0f32, f32::max);
                if replay.grad_input.len() != query_elements {
                    return Err(WorkspaceLensError::ActivationSize {
                        name: "workspace GDN mixer branch cotangent query bank",
                        got: replay.grad_input.len(),
                        expected: query_elements,
                    });
                }
                (
                    replay.grad_input,
                    residual_replay_max_abs_error,
                    WorkspaceLensBlockKind::Gdn,
                )
            }
            MetalBlock::Attn(block) => {
                let geometry = AttnGeometry::new(arch)?;
                validate_attn_weights(layer, block, geometry)?;
                let replay = attn_mixer_replay_vjp_batch_readback(
                    self.model.context(),
                    geometry,
                    AttnMixerWeights::from(block),
                    input_residuals,
                    &grad_post_mixer,
                    n_tokens,
                    n_query,
                    match rule {
                        WorkspaceLensRule::Jacobian => AttnBlockVjpRule::Jacobian,
                        WorkspaceLensRule::Relp => AttnBlockVjpRule::Relp,
                    },
                )?;
                let residual_replay_max_abs_error = input_residuals
                    .iter()
                    .zip(&replay.mixer_outputs)
                    .zip(post_mixer_residuals)
                    .map(|((&input, &mixer), &observed)| {
                        finite_abs_difference(input + mixer, observed)
                    })
                    .fold(0.0f32, f32::max);
                if replay.grad_input.len() != query_elements {
                    return Err(WorkspaceLensError::ActivationSize {
                        name: "workspace attention mixer branch cotangent query bank",
                        got: replay.grad_input.len(),
                        expected: query_elements,
                    });
                }
                (
                    replay.grad_input,
                    residual_replay_max_abs_error,
                    WorkspaceLensBlockKind::Attention,
                )
            }
        };
        if grad_mixer_input.len() != query_elements {
            return Err(WorkspaceLensError::ActivationSize {
                name: "workspace mixer branch cotangent query bank",
                got: grad_mixer_input.len(),
                expected: query_elements,
            });
        }
        let values = grad_post_mixer
            .into_iter()
            .zip(grad_mixer_input)
            .map(|(identity, branch)| identity + branch)
            .collect();
        Ok((
            values,
            WorkspaceLensReplayDiagnostic {
                layer,
                kind,
                residual_replay_max_abs_error,
            },
        ))
    }

    pub(super) fn resolve_linear(
        &self,
        id: WorkspaceLensLinear,
    ) -> Result<&MetalTensor, WorkspaceLensError> {
        let WorkspaceLensLinear::Layer { index, role } = id else {
            return Ok(&self.model.metal_model().lm_head);
        };
        let block = self.model.metal_model().blocks.get(index as usize).ok_or(
            WorkspaceLensError::InvalidLayer {
                layer: index,
                n_layers: self.arch().n_layer,
            },
        )?;
        let tensor = match (block, role) {
            (MetalBlock::Gdn(block), LinearRole::FfnGate) => &block.ffn_gate,
            (MetalBlock::Gdn(block), LinearRole::FfnUp) => &block.ffn_up,
            (MetalBlock::Gdn(block), LinearRole::FfnDown) => &block.ffn_down,
            (MetalBlock::Gdn(block), LinearRole::GdnQkv) => &block.in_proj_qkv,
            (MetalBlock::Gdn(block), LinearRole::GdnZ) => &block.in_proj_z,
            (MetalBlock::Gdn(block), LinearRole::GdnBeta) => &block.beta_proj,
            (MetalBlock::Gdn(block), LinearRole::GdnAlpha) => &block.alpha_proj,
            (MetalBlock::Gdn(block), LinearRole::GdnOut) => &block.out_proj,
            (MetalBlock::Attn(block), LinearRole::FfnGate) => &block.ffn_gate,
            (MetalBlock::Attn(block), LinearRole::FfnUp) => &block.ffn_up,
            (MetalBlock::Attn(block), LinearRole::FfnDown) => &block.ffn_down,
            (MetalBlock::Attn(block), LinearRole::AttentionQAndGate) => &block.q,
            (MetalBlock::Attn(block), LinearRole::AttentionK) => &block.k,
            (MetalBlock::Attn(block), LinearRole::AttentionV) => &block.v,
            (MetalBlock::Attn(block), LinearRole::AttentionOut) => &block.o,
            _ => return Err(WorkspaceLensError::InvalidLinearRole { layer: index, role }),
        };
        Ok(tensor)
    }
}
