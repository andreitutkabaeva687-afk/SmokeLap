# -*- coding: utf-8 -*-

"""
Bounded Gated SmokeLap.

This model keeps the complete gated/reliable SmokeLap pipeline
and introduces a bounded memory residual before Smoke-guided Fusion.

Fusion:

    smoke_gate =
        clip(
            (s - tau_s) / (1 - tau_s),
            0,
            1
        )

    reliability =
        clip(
            (sim - tau_r) / (1 - tau_r),
            0,
            1
        )

    beta = sigmoid(fusion_logit)

    alpha =
        beta
        * smoke_gate
        * reliability

    delta =
        F_m - F_c

    bounded_delta =
        residual_scale * delta

    F_f =
        F_c
        + alpha * bounded_delta

The residual magnitude is bounded according to the RMS magnitude
of the current semantic feature.

Therefore:

    smoke <= tau_s
        -> alpha = 0
        -> exactly preserve current-frame feature

    smoke > tau_s
        -> memory correction is introduced progressively

    if memory correction is too large
        -> automatically suppress its magnitude
"""


import math

import torch
import torch.nn.functional as F

from .smokelap_reliable import ReliableSmokeLap
from .convlstm import ConvLSTMSegmenter


class SmokeLap(ReliableSmokeLap):
    """
    Reliability-aware SmokeLap model with:

    1. Smoke activation gate
    2. Retrieval reliability
    3. Cross-Attention memory enhancement
    4. Bounded memory residual
    5. Smoke-guided fusion


    def __init__(
        self,
        in_channels=3,
        num_classes=9,
        base_channels=32,
        memory_size=20,
        smoke_threshold=0.50,
        use_cross_attention=True,
        use_smoke_guided_fusion=True,
        memory_write_strategy="smoke_gated",
        smoke_hidden_dim=128,
        handcrafted_dim=5,
        handcrafted_feature_mode="all",
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        attention_heads=4,
        retrieval_threshold=0.50,
        fusion_init_strength=0.02,
        fusion_smoke_threshold=0.50,
        retrieval_topk=1,
        retrieval_recency_weight=0.0,
        retrieval_temperature=1.0,
        zero_init_attention_output=False,
        attention_grad_multiplier=1.0,
        use_spatial_safety_gate=False,
        safety_gate_init=0.50,
        confidence_fallback=False,
        confidence_margin=0.0,
        confidence_fallback_during_training=True,
        return_base_logits=False,
        use_learned_retrieval_keys=False,
        retrieval_key_dim=64,
        fusion_max_strength=1.0,
        use_convlstm_expert=False,
        output_fusion_hidden=16,
        output_memory_bias=-3.0,
        output_fusion_mode='pixel',
        disable_smoke_estimation=False,
    ):

        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            memory_size=memory_size,
            smoke_threshold=smoke_threshold,
            use_cross_attention=use_cross_attention,
            use_smoke_guided_fusion=use_smoke_guided_fusion,
            memory_write_strategy=memory_write_strategy,
            smoke_hidden_dim=smoke_hidden_dim,
            handcrafted_dim=handcrafted_dim,
            handcrafted_feature_mode=handcrafted_feature_mode,
            img_mean=img_mean,
            img_std=img_std,
            attention_heads=attention_heads,
            retrieval_threshold=retrieval_threshold,
            fusion_init_strength=fusion_init_strength,
        )

        self.fusion_smoke_threshold = float(
            fusion_smoke_threshold
        )

        # Retrieval V2 options.  Their defaults exactly preserve the
        # original Top-1 cosine-retrieval behavior.
        self.retrieval_topk = int(retrieval_topk)
        self.retrieval_recency_weight = float(
            retrieval_recency_weight
        )
        self.retrieval_temperature = float(
            retrieval_temperature
        )
        self.zero_init_attention_output = bool(
            zero_init_attention_output
        )
        self.attention_grad_multiplier = float(
            attention_grad_multiplier
        )
        self.use_spatial_safety_gate = bool(
            use_spatial_safety_gate
        )
        self.safety_gate_init = float(
            safety_gate_init
        )
        self.confidence_fallback = bool(
            confidence_fallback
        )
        self.confidence_margin = float(
            confidence_margin
        )
        self.confidence_fallback_during_training = bool(
            confidence_fallback_during_training
        )
        self.return_base_logits = bool(
            return_base_logits
        )
        self.use_learned_retrieval_keys = bool(
            use_learned_retrieval_keys
        )
        self.retrieval_key_dim = int(
            retrieval_key_dim
        )
        self.fusion_max_strength = float(
            fusion_max_strength
        )
        self.use_convlstm_expert = bool(
            use_convlstm_expert
        )
        self.output_fusion_mode = str(
            output_fusion_mode
        ).strip().lower()
        # Controlled ablation switch: remove smoke prediction from all
        # downstream memory/fusion decisions without changing any other
        # architecture or initialization choice.
        self.disable_smoke_estimation = bool(
            disable_smoke_estimation
        )

        if self.retrieval_topk <= 0:
            raise ValueError(
                "retrieval_topk must be positive."
            )

        if self.retrieval_recency_weight < 0.0:
            raise ValueError(
                "retrieval_recency_weight must be >= 0."
            )

        if self.retrieval_temperature <= 0.0:
            raise ValueError(
                "retrieval_temperature must be > 0."
            )

        if self.attention_grad_multiplier < 1.0:
            raise ValueError(
                "attention_grad_multiplier must be >= 1."
            )

        if not 0.0 < self.safety_gate_init < 1.0:
            raise ValueError(
                "safety_gate_init must be in (0,1)."
            )

        if self.confidence_margin < 0.0:
            raise ValueError(
                "confidence_margin must be >= 0."
            )

        if self.retrieval_key_dim <= 0:
            raise ValueError(
                "retrieval_key_dim must be positive."
            )

        if not 0.0 < self.fusion_max_strength <= 1.0:
            raise ValueError(
                "fusion_max_strength must be in (0,1]."
            )

        if fusion_init_strength > self.fusion_max_strength:
            raise ValueError(
                "fusion_init_strength cannot exceed "
                "fusion_max_strength."
            )

        # Interpret fusion_init_strength as the actual beta even when beta is
        # capped below one.  This keeps all old configurations identical and
        # lets the joint-gain version learn beta safely inside [0, cap].
        normalized_beta = float(
            fusion_init_strength
        ) / self.fusion_max_strength
        normalized_beta = min(
            max(normalized_beta, 1e-4),
            1.0 - 1e-4,
        )
        torch.nn.init.constant_(
            self.fusion_logit,
            math.log(
                normalized_beta
                / (1.0 - normalized_beta)
            ),
        )

        # A zero output projection makes Cross-Attention an exact identity
        # at initialization.  The initialized M20 model therefore matches
        # M0 exactly, while gradients can first learn a safe residual through
        # out_proj and then propagate into Q/K/V.
        if self.zero_init_attention_output:
            torch.nn.init.zeros_(
                self.cross_attn.out_proj.weight
            )

        # Optional spatial gate for V3.  It predicts where a memory
        # correction is safe instead of applying one scalar alpha to the
        # complete bottleneck map.  Zero weights make its initial output a
        # constant, while the zero-initialized attention output still
        # guarantees exact M0-equivalent logits.
        if self.use_spatial_safety_gate:
            channels = self.cross_attn.channels
            self.safety_gate = torch.nn.Conv2d(
                channels * 2,
                1,
                kernel_size=1,
                bias=True,
            )
            torch.nn.init.zeros_(
                self.safety_gate.weight
            )
            initial_logit = math.log(
                self.safety_gate_init
                / (1.0 - self.safety_gate_init)
            )
            torch.nn.init.constant_(
                self.safety_gate.bias,
                initial_logit,
            )
        else:
            self.safety_gate = None

        if self.use_learned_retrieval_keys:
            self.retrieval_key_proj = torch.nn.Conv2d(
                self.cross_attn.channels,
                self.retrieval_key_dim,
                kernel_size=1,
                bias=False,
            )
            torch.nn.init.kaiming_normal_(
                self.retrieval_key_proj.weight,
                mode='fan_out',
                nonlinearity='linear',
            )
        else:
            self.retrieval_key_proj = None

        # Optional safety-first output ensemble.  The U-Net++ current-frame
        # prediction and a frozen ConvLSTM expert form the strong reference;
        # the memory branch is a third expert whose per-pixel contribution is
        # learned from all class probabilities plus predicted smoke density.
        if self.use_convlstm_expert:
            self.convlstm_expert = ConvLSTMSegmenter(
                in_channels=in_channels,
                num_classes=num_classes,
                base_channels=base_channels,
            )
            if self.output_fusion_mode == 'pixel':
                self.output_fusion = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        num_classes * 3 + 1,
                        int(output_fusion_hidden),
                        kernel_size=1,
                    ),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(
                        int(output_fusion_hidden),
                        3,
                        kernel_size=1,
                    ),
                )
                torch.nn.init.zeros_(self.output_fusion[-1].weight)
                with torch.no_grad():
                    self.output_fusion[-1].bias.copy_(
                        torch.tensor(
                            [0.0, 0.0, float(output_memory_bias)],
                            dtype=self.output_fusion[-1].bias.dtype,
                        )
                    )
            elif self.output_fusion_mode == 'smoke_classwise':
                self.output_fusion = torch.nn.Conv2d(
                    1,
                    num_classes * 3,
                    kernel_size=1,
                )
                torch.nn.init.zeros_(self.output_fusion.weight)
                initial_bias = torch.tensor(
                    [0.0, 0.0, float(output_memory_bias)],
                    dtype=self.output_fusion.bias.dtype,
                ).view(3, 1).expand(3, num_classes).reshape(-1)
                with torch.no_grad():
                    self.output_fusion.bias.copy_(initial_bias)
            else:
                raise ValueError(
                    'output_fusion_mode must be pixel or '
                    'smoke_classwise.'
                )
        else:
            self.convlstm_expert = None
            self.output_fusion = None

        self.last_memory_diagnostics = {}
        self._diagnostic_buffer = None
        self._aux_residual_ratios = None

        if not (
            0.0
            <= self.fusion_smoke_threshold
            < 1.0
        ):

            raise ValueError(
                "fusion_smoke_threshold must satisfy "
                "0 <= threshold < 1."
            )

    def fusion_strength(self):
        """Return the actual (optionally capped) fusion coefficient."""

        return (
            self.fusion_max_strength
            * torch.sigmoid(
                self.fusion_logit
            )
        )

    # =========================================================
    # Top-k similarity + recency retrieval
    # =========================================================

    def _retrieve(
        self,
        current,
        items,
    ):
        """Retrieve a soft Top-k aggregate from the per-video FIFO bank.

        Similarity remains the primary signal.  A small recency bonus breaks
        near-ties in favour of spatially/temporally closer frames.  Returning
        the weighted *cosine* similarity keeps the existing reliability gate
        semantically compatible with the original implementation.
        """

        if (
            self.memory_size <= 0
            or len(items) == 0
        ):
            if self._diagnostic_buffer is not None:
                self._diagnostic_buffer[
                    'empty_queries'
                ] += 1
            return None, None

        memory = torch.cat(
            [
                x.to(
                    device=current.device,
                    dtype=current.dtype,
                    non_blocking=True,
                )
                for x in items
            ],
            dim=0,
        )

        if self.retrieval_key_proj is not None:
            current_key = self.retrieval_key_proj(
                current
            )
            memory_key = self.retrieval_key_proj(
                memory
            )
        else:
            current_key = current
            memory_key = memory

        q = F.normalize(
            self._vec(current_key),
            dim=1,
        )
        k = F.normalize(
            self._vec(memory_key),
            dim=1,
        )

        cosine = (q @ k.t())[0]
        count = int(cosine.numel())

        # Oldest=0, newest=1.  The bonus is deliberately small and is used
        # only for ranking/weighting; reliability still uses cosine itself.
        recency = torch.linspace(
            0.0,
            1.0,
            steps=count,
            device=cosine.device,
            dtype=cosine.dtype,
        )
        combined = (
            cosine
            + self.retrieval_recency_weight
            * recency
        )

        topk = min(
            self.retrieval_topk,
            count,
        )
        selected_score, selected_idx = torch.topk(
            combined,
            k=topk,
            largest=True,
            sorted=True,
        )

        weights = torch.softmax(
            selected_score
            / self.retrieval_temperature,
            dim=0,
        )
        selected_memory = memory[
            selected_idx
        ]

        # Preserve individual historical maps for multi-memory attention.
        # Top-1 keeps the legacy [B,C,H,W] shape; Top-k uses
        # [B,M,C,H,W], which CrossAttention2D tokenizes without averaging
        # spatially misaligned frames.
        if topk == 1:
            prior = selected_memory
        else:
            prior = selected_memory.unsqueeze(0)

        selected_cosine = cosine[
            selected_idx
        ]
        weighted_similarity = torch.sum(
            weights
            * selected_cosine
        )

        if self._diagnostic_buffer is not None:
            newest_distance = (
                count
                - 1
                - selected_idx.float()
            )
            self._diagnostic_buffer[
                'retrieved_queries'
            ] += 1
            self._diagnostic_buffer[
                'selected_counts'
            ].append(float(topk))
            self._diagnostic_buffer[
                'similarities'
            ].append(
                float(
                    weighted_similarity
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
            )
            self._diagnostic_buffer[
                'memory_ages'
            ].append(
                float(
                    torch.sum(
                        weights.detach()
                        * newest_distance
                    )
                    .float()
                    .cpu()
                    .item()
                )
            )

        return prior, weighted_similarity

    # =========================================================
    # Per-forward diagnostics
    # =========================================================

    @staticmethod
    def _mean(values):
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def forward(
        self,
        images,
        smoke_scores_gt=None,
        video_ids=None,
    ):
        self._diagnostic_buffer = {
            'empty_queries': 0,
            'retrieved_queries': 0,
            'selected_counts': [],
            'similarities': [],
            'memory_ages': [],
            'reliabilities': [],
            'smoke_gates': [],
            'alphas': [],
            'residual_ratios': [],
            'safety_gates': [],
            'fallback_accept': [],
        }
        self._aux_residual_ratios = []

        if images.dim() == 4:
            images = images.unsqueeze(1)

        if images.dim() != 5:
            raise ValueError(
                "Expected [B,T,C,H,W], "
                f"got {tuple(images.shape)}"
            )

        (
            all_encoder,
            bottlenecks,
            smoke_preds,
        ) = self._encode_sequence(
            images
        )

        if self.disable_smoke_estimation:
            smoke_preds = torch.zeros_like(smoke_preds)

        out = self._forward_memory(
            all_encoder,
            bottlenecks,
            smoke_preds,
            smoke_scores_gt,
            video_ids,
        )
        out['fusion_strength'] = self.fusion_strength()

        raw_fused_logits = out['logits']
        out['raw_fused_logits'] = raw_fused_logits

        need_base_logits = (
            self.return_base_logits
            or self.confidence_fallback
        )

        if need_base_logits:
            # U-Net++ is frozen in the bounded-memory training stage.  The
            # no-grad baseline branch is therefore an exact M0 reference and
            # does not duplicate any optimizer state.
            base_trainable = (
                torch.is_grad_enabled()
                and any(
                    parameter.requires_grad
                    for parameter in self.unetpp.parameters()
                )
            )
            with torch.set_grad_enabled(base_trainable):
                base_logits = self.unetpp.decode(
                    all_encoder[-1],
                    bottleneck=bottlenecks[-1],
                )
            out['base_logits'] = base_logits

            apply_confidence_fallback = (
                self.confidence_fallback
                and (
                    not self.training
                    or self.confidence_fallback_during_training
                )
            )

            if apply_confidence_fallback:
                with torch.no_grad():
                    base_confidence = torch.softmax(
                        base_logits,
                        dim=1,
                    ).amax(
                        dim=1,
                        keepdim=True,
                    )
                    fused_confidence = torch.softmax(
                        raw_fused_logits,
                        dim=1,
                    ).amax(
                        dim=1,
                        keepdim=True,
                    )
                    accept_fused = (
                        fused_confidence
                        >= (
                            base_confidence
                            + self.confidence_margin
                        )
                    )

                out['logits'] = torch.where(
                    accept_fused,
                    raw_fused_logits,
                    base_logits,
                )
                self._diagnostic_buffer[
                    'fallback_accept'
                ].append(
                    float(
                        accept_fused
                        .float()
                        .mean()
                        .cpu()
                        .item()
                    )
                )

        if self.use_convlstm_expert:
            if 'base_logits' not in out:
                raise RuntimeError(
                    'use_convlstm_expert requires return_base_logits: true.'
                )

            convlstm_trainable = (
                torch.is_grad_enabled()
                and any(
                    parameter.requires_grad
                    for parameter in self.convlstm_expert.parameters()
                )
            )
            with torch.set_grad_enabled(convlstm_trainable):
                convlstm_logits = self.convlstm_expert(images)

            base_probability = torch.softmax(
                out['base_logits'], dim=1
            )
            convlstm_probability = torch.softmax(
                convlstm_logits, dim=1
            )
            memory_probability = torch.softmax(
                raw_fused_logits, dim=1
            )
            smoke_map = smoke_preds[:, -1].view(
                smoke_preds.shape[0], 1, 1, 1
            ).expand(
                -1,
                -1,
                base_probability.shape[-2],
                base_probability.shape[-1],
            )
            if self.output_fusion_mode == 'pixel':
                gate_input = torch.cat(
                    [
                        base_probability,
                        convlstm_probability,
                        memory_probability,
                        smoke_map,
                    ],
                    dim=1,
                )
                expert_weights = torch.softmax(
                    self.output_fusion(gate_input),
                    dim=1,
                )
                ensemble_probability = (
                    expert_weights[:, 0:1] * base_probability
                    + expert_weights[:, 1:2] * convlstm_probability
                    + expert_weights[:, 2:3] * memory_probability
                )
            else:
                classwise_logits = self.output_fusion(
                    smoke_preds[:, -1].view(-1, 1, 1, 1)
                ).view(
                    smoke_preds.shape[0],
                    3,
                    base_probability.shape[1],
                    1,
                    1,
                )
                expert_weights = torch.softmax(
                    classwise_logits,
                    dim=1,
                )
                expert_probabilities = torch.stack(
                    [
                        base_probability,
                        convlstm_probability,
                        memory_probability,
                    ],
                    dim=1,
                )
                ensemble_probability = torch.sum(
                    expert_weights * expert_probabilities,
                    dim=1,
                )
            out['logits'] = torch.log(
                ensemble_probability.clamp_min(1e-7)
            )
            out['convlstm_logits'] = convlstm_logits
            out['expert_weights'] = expert_weights
            out['reference_logits'] = torch.log(
                (
                    0.5 * base_probability
                    + 0.5 * convlstm_probability
                ).clamp_min(1e-7)
            )

        if self._aux_residual_ratios:
            out['raw_residual_ratio'] = torch.stack(
                self._aux_residual_ratios
            ).mean()
        else:
            out['raw_residual_ratio'] = (
                raw_fused_logits.new_zeros(())
            )

        buf = self._diagnostic_buffer
        self.last_memory_diagnostics = {
            'empty_queries': int(
                buf['empty_queries']
            ),
            'retrieved_queries': int(
                buf['retrieved_queries']
            ),
            'topk_mean': self._mean(
                buf['selected_counts']
            ),
            'similarity_mean': self._mean(
                buf['similarities']
            ),
            'memory_age_mean': self._mean(
                buf['memory_ages']
            ),
            'reliability_mean': self._mean(
                buf['reliabilities']
            ),
            'smoke_gate_mean': self._mean(
                buf['smoke_gates']
            ),
            'alpha_mean': self._mean(
                buf['alphas']
            ),
            'alpha_max': (
                max(buf['alphas'])
                if buf['alphas']
                else 0.0
            ),
            'residual_ratio_mean': self._mean(
                buf['residual_ratios']
            ),
            'safety_gate_mean': self._mean(
                buf['safety_gates']
            ),
            'fallback_accept_mean': self._mean(
                buf['fallback_accept']
            ),
        }
        out['memory_diagnostics'] = dict(
            self.last_memory_diagnostics
        )
        self._diagnostic_buffer = None
        self._aux_residual_ratios = None
        return out

    # =========================================================
    # Smoke-guided bounded feature fusion
    # =========================================================

    def _fuse(
        self,
        current,
        prior,
        similarity,
        smoke_pred,
    ):
        """
        Threshold-gated reliability-aware bounded fusion.

        smoke_gate =
            clip(
                (smoke - fusion_smoke_threshold)
                / (1 - fusion_smoke_threshold),
                0,
                1
            )

        reliability =
            clip(
                (similarity - retrieval_threshold)
                / (1 - retrieval_threshold),
                0,
                1
            )

        beta =
            sigmoid(fusion_logit)

        alpha =
            beta
            * smoke_gate
            * reliability

        delta =
            enhanced
            - current

        bounded_delta =
            residual_scale
            * delta

        fused =
            current
            + alpha
            * bounded_delta
        """

        # -----------------------------------------------------
        # No historical memory
        # -----------------------------------------------------

        if prior is None:

            return current

        # -----------------------------------------------------
        # Cross-Attention candidate
        # -----------------------------------------------------

        if self.use_cross_attention:

            enhanced = self.cross_attn(
                current,
                prior,
            )

        else:

            if prior.dim() == 5:
                prior = prior.mean(
                    dim=1
                )

            if (
                prior.shape[-2:]
                != current.shape[-2:]
            ):

                prior = F.interpolate(
                    prior,
                    size=current.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            enhanced = (
                current
                + prior
            )

        # -----------------------------------------------------
        # Ablation:
        # directly use enhanced feature
        # -----------------------------------------------------

        if not self.use_smoke_guided_fusion:

            return enhanced

        # -----------------------------------------------------
        # Retrieval reliability
        # -----------------------------------------------------

        reliability = (
            self._memory_reliability(
                similarity
            )
        )

        if reliability is None:

            return current

        # -----------------------------------------------------
        # Predicted smoke density
        #
        # Detach:
        # segmentation loss cannot manipulate smoke prediction
        # simply to modify fusion strength.
        # -----------------------------------------------------

        smoke = (
            smoke_pred
            .reshape(-1)[0]
            .detach()
            .clamp(
                min=0.0,
                max=1.0,
            )
        )

        # -----------------------------------------------------
        # Smoke activation gate
        #
        # smoke <= threshold:
        #     smoke_gate = 0
        #
        # smoke > threshold:
        #     progressively increase memory contribution
        # -----------------------------------------------------

        denominator = max(
            1.0
            - self.fusion_smoke_threshold,
            1e-6,
        )

        smoke_gate = (
            smoke
            - self.fusion_smoke_threshold
        ) / denominator

        smoke_gate = (
            smoke_gate.clamp(
                min=0.0,
                max=1.0,
            )
        )

        # -----------------------------------------------------
        # Learnable global fusion strength
        # -----------------------------------------------------

        beta = self.fusion_strength()

        # -----------------------------------------------------
        # Final memory contribution coefficient
        # -----------------------------------------------------

        alpha = (
            beta
            * smoke_gate
            * reliability
        )

        alpha = alpha.reshape(
            1,
            1,
            1,
            1,
        )

        # =====================================================
        # Bounded Memory Residual
        # =====================================================
        #
        # Original:
        #
        #     delta = enhanced - current
        #
        # Problem:
        # Cross-Attention may generate a correction whose
        # magnitude is much larger than the current semantic
        # feature.
        #
        # Therefore we compare:
        #
        #     RMS(delta)
        #     RMS(current)
        #
        # If delta is too large:
        #     scale < 1
        #
        # If delta is already small:
        #     scale = 1
        #
        # =====================================================

        delta = (
            enhanced
            - current
        )

        # -----------------------------------------------------
        # RMS magnitude of memory correction
        #
        # Shape:
        # [B, 1, 1, 1]
        # -----------------------------------------------------

        delta_rms = torch.sqrt(
            torch.mean(
                delta.pow(2),
                dim=(1, 2, 3),
                keepdim=True,
            )
            + 1e-6
        )

        # -----------------------------------------------------
        # RMS magnitude of current semantic feature
        #
        # Shape:
        # [B, 1, 1, 1]
        # -----------------------------------------------------

        current_rms = torch.sqrt(
            torch.mean(
                current.pow(2),
                dim=(1, 2, 3),
                keepdim=True,
            )
            + 1e-6
        )

        # -----------------------------------------------------
        # Residual magnitude limiter
        #
        # delta_rms <= current_rms:
        #
        #     current_rms / delta_rms >= 1
        #     clamp(max=1)
        #     residual_scale = 1
        #
        # delta_rms > current_rms:
        #
        #     residual_scale < 1
        #
        # -----------------------------------------------------

        residual_scale = torch.clamp(
            current_rms
            / (
                delta_rms
                + 1e-6
            ),
            max=1.0,
        )

        # -----------------------------------------------------
        # Bounded Cross-Attention correction
        # -----------------------------------------------------

        bounded_delta = (
            delta
            * residual_scale
        )

        raw_residual_ratio = (
            delta_rms
            / (current_rms + 1e-6)
        ).mean()

        if self._aux_residual_ratios is not None:
            self._aux_residual_ratios.append(
                raw_residual_ratio
            )

        # Keep the forward correction unchanged while increasing only its
        # backward signal.  With beta=0.02, the original Cross-Attention
        # gradients were attenuated by beta * smoke_gate * reliability and
        # were often effectively near zero.
        if (
            self.training
            and self.attention_grad_multiplier > 1.0
        ):
            bounded_delta = (
                bounded_delta.detach()
                + self.attention_grad_multiplier
                * (
                    bounded_delta
                    - bounded_delta.detach()
                )
            )

        if self.safety_gate is not None:
            safety_gate = torch.sigmoid(
                self.safety_gate(
                    torch.cat(
                        [current, bounded_delta],
                        dim=1,
                    )
                )
            )
        else:
            safety_gate = torch.ones_like(
                bounded_delta[:, :1]
            )

        if self._diagnostic_buffer is not None:
            reliability_value = float(
                reliability.detach().float().cpu().item()
            )
            smoke_gate_value = float(
                smoke_gate.detach().float().cpu().item()
            )
            alpha_value = float(
                alpha.detach().float().cpu().item()
            )
            residual_ratio = float(
                (
                    delta_rms
                    / (current_rms + 1e-6)
                )
                .mean()
                .detach()
                .float()
                .cpu()
                .item()
            )
            self._diagnostic_buffer[
                'reliabilities'
            ].append(reliability_value)
            self._diagnostic_buffer[
                'smoke_gates'
            ].append(smoke_gate_value)
            self._diagnostic_buffer[
                'alphas'
            ].append(alpha_value)
            self._diagnostic_buffer[
                'residual_ratios'
            ].append(residual_ratio)
            self._diagnostic_buffer[
                'safety_gates'
            ].append(
                float(
                    safety_gate
                    .mean()
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
            )

        # -----------------------------------------------------
        # Final bounded Smoke-guided Fusion
        #
        # alpha = 0:
        #     fused == current
        #
        # alpha > 0:
        #     inject bounded historical correction
        # -----------------------------------------------------

        fused = (
            current
            + alpha
            * safety_gate
            * bounded_delta
        )

        return fused
