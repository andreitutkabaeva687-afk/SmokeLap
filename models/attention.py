# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention2D(nn.Module):
    """
    2D Cross-Attention.

    Query:
        current target-frame feature F_T^c

    Key / Value:
        retrieved memory feature F_j^c

    Output:
        residual cross-attention feature

        F_T^f = F_T^c + CrossAttn(F_T^c, F_j^c)

    This residual connection preserves the semantic information
    of the current frame while introducing complementary
    information from the retrieved historical feature.
    """

    def __init__(
        self,
        channels,
        num_heads=4
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                "channels must be divisible by num_heads"
            )

        self.channels = int(channels)
        self.num_heads = int(num_heads)

        self.head_dim = (
            self.channels
            // self.num_heads
        )

        # -------------------------------------------------
        # Q from current feature
        # K / V from retrieved memory feature
        # -------------------------------------------------
        self.q_proj = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
            bias=False
        )

        self.k_proj = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
            bias=False
        )

        self.v_proj = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
            bias=False
        )

        self.out_proj = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
            bias=False
        )

    # =====================================================
    # Reshape feature map for multi-head attention
    # =====================================================
    def _shape(
        self,
        x
    ):
        """
        Input:
            x: [B, C, H, W]

        Output:
            [B, heads, H*W, head_dim]
        """

        b, c, h, w = x.shape

        x = (
            x.flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

        x = x.view(
            b,
            h * w,
            self.num_heads,
            self.head_dim
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
        )

        return x

    def _shape_memory(
        self,
        x,
    ):
        """Convert one or more memory maps into attention tokens.

        Input:
            [B, M, C, H, W]

        Output:
            [B, heads, M*H*W, head_dim]

        Keeping the memory-frame dimension until tokenization avoids the
        spatial blurring caused by averaging several moving video frames
        before Cross-Attention.
        """

        b, m, c, h, w = x.shape

        x = (
            x.permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(
                b,
                m * h * w,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
            .contiguous()
        )

        return x

    # =====================================================
    # Forward
    # =====================================================
    def forward(
        self,
        current,
        prior
    ):
        """
        current:
            [B, C, H, W]

        prior:
            [B, C, H, W]
            or [B, M, C, H, W]

        Q = current
        K = prior
        V = prior
        """

        # -------------------------------------------------
        # Spatial alignment
        # -------------------------------------------------
        if prior.dim() not in (4, 5):
            raise ValueError(
                "prior must be [B,C,H,W] or "
                f"[B,M,C,H,W], got {tuple(prior.shape)}"
            )

        multi_memory = (
            prior.dim() == 5
        )

        if multi_memory:
            b, m, c, h, w = prior.shape

            if b != current.shape[0]:
                raise ValueError(
                    "current/prior batch mismatch: "
                    f"{current.shape[0]} vs {b}"
                )

            prior_flat = prior.reshape(
                b * m,
                c,
                h,
                w,
            )

            if (
                prior_flat.shape[-2:]
                != current.shape[-2:]
            ):
                prior_flat = F.interpolate(
                    prior_flat,
                    size=current.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            prior_h, prior_w = (
                prior_flat.shape[-2:]
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

        # -------------------------------------------------
        # Q / K / V
        # -------------------------------------------------
        q = self._shape(
            self.q_proj(
                current
            )
        )

        if multi_memory:
            k_projected = self.k_proj(
                prior_flat
            ).reshape(
                b,
                m,
                self.channels,
                prior_h,
                prior_w,
            )
            v_projected = self.v_proj(
                prior_flat
            ).reshape(
                b,
                m,
                self.channels,
                prior_h,
                prior_w,
            )
            k = self._shape_memory(
                k_projected
            )
            v = self._shape_memory(
                v_projected
            )
        else:
            k = self._shape(
                self.k_proj(
                    prior
                )
            )

            v = self._shape(
                self.v_proj(
                    prior
                )
            )

        # -------------------------------------------------
        # Cross-Attention
        # -------------------------------------------------
        if hasattr(
            F,
            "scaled_dot_product_attention"
        ):

            out = (
                F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    is_causal=False
                )
            )

        else:

            attn = torch.matmul(
                q,
                k.transpose(
                    -2,
                    -1
                )
            )

            attn = (
                attn
                / math.sqrt(
                    self.head_dim
                )
            )

            attn = torch.softmax(
                attn,
                dim=-1
            )

            out = torch.matmul(
                attn,
                v
            )

        # -------------------------------------------------
        # Restore [B,C,H,W]
        # -------------------------------------------------
        b, _, n, _ = out.shape

        h, w = current.shape[-2:]

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(
                b,
                n,
                self.channels
            )
            .transpose(1, 2)
            .contiguous()
            .view(
                b,
                self.channels,
                h,
                w
            )
        )

        out = self.out_proj(
            out
        )

        # -------------------------------------------------
        # Residual Cross-Attention
        #
        # Preserve current-frame semantic information.
        #
        # F_T^f =
        # F_T^c + CrossAttn(F_T^c, F_j^c)
        # -------------------------------------------------
        fused = (
            current
            + out
        )

        return fused
