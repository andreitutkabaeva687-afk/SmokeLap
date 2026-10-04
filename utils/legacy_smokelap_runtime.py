# -*- coding: utf-8 -*-


import types

import torch


def install_legacy_smoke_fusion(model):
    """Install the historical smoke-guided fusion rule on one model instance."""

    if not hasattr(model, "_forward_memory") or not hasattr(model, "_fuse"):
        raise TypeError("Expected a compatible SmokeLap instance")

    original_forward_memory = model._forward_memory
    original_fuse = model._fuse

    def legacy_forward_memory(
        self, all_encoder, bottlenecks, smoke_preds, smoke_scores_gt, video_ids
    ):
        self._legacy_fusion_scores = [
            self._get_gate_score(smoke_preds, smoke_scores_gt, bi)
            for bi in range(int(smoke_preds.shape[0]))
        ]
        self._legacy_fusion_index = 0
        try:
            return original_forward_memory(
                all_encoder,
                bottlenecks,
                smoke_preds,
                smoke_scores_gt,
                video_ids,
            )
        finally:
            self._legacy_fusion_scores = None
            self._legacy_fusion_index = 0

    def legacy_fuse(self, current, prior):
        index = int(self._legacy_fusion_index)
        score = self._legacy_fusion_scores[index]
        self._legacy_fusion_index = index + 1

        if prior is None:
            return current

        if self.use_cross_attention:
            memory_feature = self.cross_attn(current, prior)
        else:
            memory_feature = prior

        if not self.use_smoke_guided_fusion:
            return memory_feature

        if not torch.is_tensor(score):
            score = torch.as_tensor(score, device=current.device, dtype=current.dtype)
        else:
            score = score.to(device=current.device, dtype=current.dtype)
        score = torch.clamp(score, 0.0, 1.0).reshape(1, 1, 1, 1)
        return (1.0 - score) * current + score * memory_feature

    model._forward_memory = types.MethodType(legacy_forward_memory, model)
    model._fuse = types.MethodType(legacy_fuse, model)
    model._legacy_runtime_enabled = True
    return model
