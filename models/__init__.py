from .smokelap import SmokeLap


def build_model(cfg):
    model_type = cfg.get("model_type", "smokelap")
    if model_type != "smokelap":
        raise ValueError(
            "This selected public release supports only model_type='smokelap'."
        )

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    common = {
        "in_channels": model_cfg.get("in_channels", 3),
        "num_classes": model_cfg.get("num_classes", data_cfg.get("num_classes", 9)),
        "base_channels": model_cfg.get("base_channels", 32),
    }
    return SmokeLap(
        **common,
        memory_size=model_cfg.get("memory_size", 20),
        smoke_threshold=model_cfg.get("smoke_threshold", 0.50),
        memory_write_strategy=model_cfg.get("memory_write_strategy", "smoke_gated"),
        use_cross_attention=model_cfg.get("use_cross_attention", True),
        attention_heads=model_cfg.get("attention_heads", 4),
        use_smoke_guided_fusion=model_cfg.get("use_smoke_guided_fusion", True),
        retrieval_threshold=model_cfg.get("retrieval_threshold", 0.50),
        fusion_init_strength=model_cfg.get("fusion_init_strength", 0.02),
        fusion_smoke_threshold=model_cfg.get("fusion_smoke_threshold", 0.50),
        retrieval_topk=model_cfg.get("retrieval_topk", 1),
        retrieval_recency_weight=model_cfg.get("retrieval_recency_weight", 0.0),
        retrieval_temperature=model_cfg.get("retrieval_temperature", 1.0),
        zero_init_attention_output=model_cfg.get("zero_init_attention_output", False),
        attention_grad_multiplier=model_cfg.get("attention_grad_multiplier", 1.0),
        use_spatial_safety_gate=model_cfg.get("use_spatial_safety_gate", False),
        safety_gate_init=model_cfg.get("safety_gate_init", 0.50),
        confidence_fallback=model_cfg.get("confidence_fallback", False),
        confidence_margin=model_cfg.get("confidence_margin", 0.0),
        confidence_fallback_during_training=model_cfg.get(
            "confidence_fallback_during_training", True
        ),
        return_base_logits=model_cfg.get("return_base_logits", False),
        use_learned_retrieval_keys=model_cfg.get("use_learned_retrieval_keys", False),
        retrieval_key_dim=model_cfg.get("retrieval_key_dim", 64),
        fusion_max_strength=model_cfg.get("fusion_max_strength", 1.0),
        use_convlstm_expert=model_cfg.get("use_convlstm_expert", False),
        output_fusion_hidden=model_cfg.get("output_fusion_hidden", 16),
        output_memory_bias=model_cfg.get("output_memory_bias", -3.0),
        output_fusion_mode=model_cfg.get("output_fusion_mode", "pixel"),
        disable_smoke_estimation=model_cfg.get("disable_smoke_estimation", False),
        smoke_hidden_dim=model_cfg.get("smoke_hidden_dim", 128),
        handcrafted_dim=model_cfg.get("handcrafted_dim", 5),
        handcrafted_feature_mode=model_cfg.get("handcrafted_feature_mode", "all"),
        img_mean=data_cfg.get("img_mean", [0.485, 0.456, 0.406]),
        img_std=data_cfg.get("img_std", [0.229, 0.224, 0.225]),
    )
