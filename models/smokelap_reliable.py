# -*- coding: utf-8 -*-

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unetpp import UNetPlusPlus
from .fog_estimator import FogEstimatorHead
from .attention import CrossAttention2D


class ReliableSmokeLap(nn.Module):
    """
    Reliability-aware SmokeLap model.

    Core components
    ---------------------------------------------------------
    1. Smoke Density Estimation
    2. Smoke-Aware persistent per-video FIFO Memory Bank
    3. Top-1 cosine-similarity retrieval
    4. Cross-Attention
    5. Reliability-aware Smoke-guided Fusion

    Fusion:

        reliability =
            clip(
                (sim - tau_r) / (1 - tau_r),
                0,
                1
            )

        beta = sigmoid(fusion_logit)

        alpha =
            beta
            * smoke_density
            * reliability

        F_f =
            F_c
            +
            alpha * (F_m - F_c)

    where:

        F_c:
            current-frame semantic feature

        F_m:
            Cross-Attention enhanced feature

        smoke_density:
            predicted current-frame smoke density

        reliability:
            confidence of retrieved memory feature

        beta:
            learnable global fusion strength
    """

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

        # New parameters
        retrieval_threshold=0.50,
        fusion_init_strength=0.10,
    ):
        super().__init__()

        # =====================================================
        # U-Net++ backbone
        # =====================================================

        self.unetpp = UNetPlusPlus(
            in_channels,
            num_classes,
            base_channels
        )

        c = self.unetpp.filters[-1]

        # =====================================================
        # Smoke Density Estimation
        # =====================================================

        self.fog_head = FogEstimatorHead(
            c,
            handcrafted_dim,
            smoke_hidden_dim,
            img_mean,
            img_std,
            handcrafted_feature_mode
        )

        # =====================================================
        # Cross-Attention
        # =====================================================

        self.cross_attn = CrossAttention2D(
            c,
            attention_heads
        )

        # =====================================================
        # Memory configuration
        # =====================================================

        self.memory_size = int(memory_size)

        self.smoke_threshold = float(
            smoke_threshold
        )

        self.use_cross_attention = bool(
            use_cross_attention
        )

        self.use_smoke_guided_fusion = bool(
            use_smoke_guided_fusion
        )

        self.memory_write_strategy = (
            memory_write_strategy
        )

        # =====================================================
        # Reliability-aware fusion
        # =====================================================

        self.retrieval_threshold = float(
            retrieval_threshold
        )

        # Initial beta.
        #
        # fusion_init_strength = 0.10
        # sigmoid(logit) ~= 0.10
        #
        p = float(fusion_init_strength)

        p = min(
            max(p, 1e-4),
            1.0 - 1e-4
        )

        init_logit = math.log(
            p / (1.0 - p)
        )

        self.fusion_logit = nn.Parameter(
            torch.tensor(
                init_logit,
                dtype=torch.float32
            )
        )

        # =====================================================
        # Persistent per-video FIFO memory
        # =====================================================

        self._memory = defaultdict(list)

    # =========================================================
    # Reset memory
    # =========================================================

    def reset_memory(self):

        self._memory = defaultdict(list)

    # =========================================================
    # Global semantic vector
    # =========================================================

    @staticmethod
    def _vec(x):

        return (
            F.adaptive_avg_pool2d(
                x,
                output_size=1
            )
            .flatten(1)
        )

    # =========================================================
    # Memory write condition
    # =========================================================

    def _should_write(
        self,
        score
    ):

        if (
            self.memory_size <= 0
            or self.memory_write_strategy == "none"
        ):
            return False

        if self.memory_write_strategy == "all_frame":
            return True

        if torch.is_tensor(score):

            score = float(
                score
                .detach()
                .float()
                .reshape(-1)[0]
                .item()
            )

        else:

            score = float(score)

        if self.memory_write_strategy == "smoke_gated":

            return (
                score
                < self.smoke_threshold
            )

        raise ValueError(
            "Unknown memory_write_strategy: "
            f"{self.memory_write_strategy}"
        )

    # =========================================================
    # Memory retrieval
    # =========================================================

    def _retrieve(
        self,
        current,
        items
    ):
        """
        Returns
        -------------------------------------------------------
        prior:
            retrieved historical feature

        best_similarity:
            cosine similarity of Top-1 memory item
        """

        if (
            self.memory_size <= 0
            or len(items) == 0
        ):

            return None, None

        # -----------------------------------------------------
        # Move stored CPU features to current device
        # -----------------------------------------------------

        memory = torch.cat(
            [
                x.to(
                    device=current.device,
                    dtype=current.dtype,
                    non_blocking=True
                )
                for x in items
            ],
            dim=0
        )

        # -----------------------------------------------------
        # Current query
        # -----------------------------------------------------

        q = F.normalize(
            self._vec(current),
            dim=1
        )

        # -----------------------------------------------------
        # Memory keys
        # -----------------------------------------------------

        k = F.normalize(
            self._vec(memory),
            dim=1
        )

        # -----------------------------------------------------
        # Cosine similarity
        #
        # [1,C] @ [C,M]
        # -> [1,M]
        # -----------------------------------------------------

        sim = q @ k.t()

        # -----------------------------------------------------
        # Top-1
        # -----------------------------------------------------

        idx = int(
            torch.argmax(
                sim[0]
            ).item()
        )

        prior = memory[
            idx:
            idx + 1
        ]

        best_similarity = sim[
            0,
            idx
        ]

        return (
            prior,
            best_similarity
        )

    # =========================================================
    # FIFO writing
    # =========================================================

    def _write_memory(
        self,
        video_id,
        current,
        gate_score
    ):

        if not self._should_write(
            gate_score
        ):
            return

        vid = str(video_id)

        bank = self._memory[vid]

        bank.append(
            current
            .detach()
            .cpu()
        )

        if len(bank) > self.memory_size:

            del bank[
                0:
                len(bank) - self.memory_size
            ]

    # =========================================================
    # Reliability
    # =========================================================

    def _memory_reliability(
        self,
        similarity
    ):
        """
        Convert Top-1 cosine similarity to [0,1].

        similarity <= threshold:
            reliability = 0

        similarity -> 1:
            reliability -> 1
        """

        if similarity is None:

            return None

        denominator = max(
            1.0 - self.retrieval_threshold,
            1e-6
        )

        reliability = (
            similarity
            - self.retrieval_threshold
        ) / denominator

        reliability = reliability.clamp(
            min=0.0,
            max=1.0
        )

        return reliability

    # =========================================================
    # Smoke-guided feature fusion
    # =========================================================

    def _fuse(
        self,
        current,
        prior,
        similarity,
        smoke_pred
    ):
        """
        Reliability-aware Smoke-guided Fusion.

        F_f =
            F_c
            +
            alpha * (F_m - F_c)

        alpha =
            beta
            * smoke
            * reliability
        """

        # -----------------------------------------------------
        # No memory
        # -----------------------------------------------------

        if prior is None:

            return current

        # -----------------------------------------------------
        # Cross-Attention candidate
        # -----------------------------------------------------

        if self.use_cross_attention:

            enhanced = self.cross_attn(
                current,
                prior
            )

        else:

            if (
                prior.shape[-2:]
                != current.shape[-2:]
            ):

                prior = F.interpolate(
                    prior,
                    size=current.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            # Keep same residual-style ablation behavior.
            enhanced = (
                current
                + prior
            )

        # -----------------------------------------------------
        # Smoke-guided fusion disabled:
        # directly use Cross-Attention result.
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
        # Important:
        # detach so segmentation loss cannot artificially alter
        # smoke density merely to manipulate fusion strength.
        # Smoke head is still trained by L_smoke.
        # -----------------------------------------------------

        smoke = (
            smoke_pred
            .reshape(-1)[0]
            .detach()
            .clamp(
                min=0.0,
                max=1.0
            )
        )

        # -----------------------------------------------------
        # Learnable global fusion strength
        # -----------------------------------------------------

        beta = torch.sigmoid(
            self.fusion_logit
        )

        # -----------------------------------------------------
        # Final memory contribution
        # -----------------------------------------------------

        alpha = (
            beta
            * smoke
            * reliability
        )

        alpha = alpha.reshape(
            1,
            1,
            1,
            1
        )

        # -----------------------------------------------------
        # Safe residual interpolation
        #
        # alpha = 0:
        #     exactly current feature
        #
        # alpha = 1:
        #     Cross-Attention enhanced feature
        # -----------------------------------------------------

        fused = (
            current
            +
            alpha
            * (
                enhanced
                - current
            )
        )

        return fused

    # =========================================================
    # Encode T-frame sequence
    # =========================================================

    def _encode_sequence(
        self,
        images
    ):

        all_encoder = []
        bottlenecks = []
        smoke_preds = []

        for t in range(
            images.shape[1]
        ):

            frame = images[
                :,
                t
            ]

            feats = self.unetpp.encode(
                frame
            )

            bottleneck = feats[-1]

            smoke = self.fog_head(
                bottleneck,
                frame
            )

            all_encoder.append(
                feats
            )

            bottlenecks.append(
                bottleneck
            )

            smoke_preds.append(
                smoke
            )

        smoke_preds = torch.stack(
            smoke_preds,
            dim=1
        )

        return (
            all_encoder,
            bottlenecks,
            smoke_preds
        )

    # =========================================================
    # Memory write gate
    # =========================================================

    def _get_gate_score(
        self,
        smoke_preds,
        smoke_scores_gt,
        bi
    ):

        # -----------------------------------------------------
        # Training:
        # GT smoke density controls memory writing
        # -----------------------------------------------------

        if (
            self.training
            and smoke_scores_gt is not None
        ):

            if smoke_scores_gt.dim() == 1:

                return smoke_scores_gt[
                    bi
                ]

            return smoke_scores_gt[
                bi,
                -1
            ]

        # -----------------------------------------------------
        # Validation / Test:
        # predicted smoke density
        # -----------------------------------------------------

        return (
            smoke_preds[
                bi,
                -1
            ]
            .detach()
        )

    # =========================================================
    # Persistent-memory forward
    # =========================================================

    def _forward_memory(
        self,
        all_encoder,
        bottlenecks,
        smoke_preds,
        smoke_scores_gt,
        video_ids
    ):

        batch_size = (
            smoke_preds.shape[0]
        )

        if video_ids is None:

            raise ValueError(
                "ReliableSmokeLap persistent memory "
                "requires video_ids."
            )

        fused_list = []

        for bi in range(
            batch_size
        ):

            vid = str(
                video_ids[bi]
            )

            # -------------------------------------------------
            # Current TARGET-frame bottleneck
            # -------------------------------------------------

            cur = (
                bottlenecks[-1][
                    bi:
                    bi + 1
                ]
            )

            # =================================================
            # 1. Retrieve BEFORE write
            # =================================================

            (
                prior,
                similarity
            ) = self._retrieve(
                cur,
                self._memory[vid]
            )

            # =================================================
            # 2. Predicted target-frame smoke density
            #
            # Used for feature fusion.
            # =================================================

            smoke_for_fusion = (
                smoke_preds[
                    bi,
                    -1
                ]
            )

            # =================================================
            # 3. Reliability-aware Smoke-guided Fusion
            # =================================================

            fused_cur = self._fuse(
                cur,
                prior,
                similarity,
                smoke_for_fusion
            )

            fused_list.append(
                fused_cur
            )

            # =================================================
            # 4. Memory write gate
            # =================================================

            gate_score = (
                self._get_gate_score(
                    smoke_preds,
                    smoke_scores_gt,
                    bi
                )
            )

            # =================================================
            # 5. Write current TARGET feature
            # =================================================

            self._write_memory(
                vid,
                cur,
                gate_score
            )

        # -----------------------------------------------------
        # Concatenate batch
        # -----------------------------------------------------

        fused = torch.cat(
            fused_list,
            dim=0
        )

        # -----------------------------------------------------
        # Decode target frame
        # -----------------------------------------------------

        logits = self.unetpp.decode(
            all_encoder[-1],
            bottleneck=fused
        )

        return {
            "logits": logits,
            "smoke_preds": smoke_preds,
            "fused_feature": fused,

            # Useful for logging/debugging.
            "fusion_strength":
                torch.sigmoid(
                    self.fusion_logit
                )
        }

    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,
        images,
        smoke_scores_gt=None,
        video_ids=None
    ):

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
            smoke_preds
        ) = self._encode_sequence(
            images
        )

        return self._forward_memory(
            all_encoder,
            bottlenecks,
            smoke_preds,
            smoke_scores_gt,
            video_ids
        )
