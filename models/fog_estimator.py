import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_handcrafted_smoke_features(
    img,
    mean=None,
    std=None,
    eps=1e-6
):
    """
    Compute 5 handcrafted smoke-related features.

    Feature order:
        0: contrast
        1: edge_density
        2: brightness
        3: saturation
        4: dark_channel
    """

    # =========================================================
    # 1. De-normalize image
    # =========================================================
    if mean is not None and std is not None:

        mean = torch.as_tensor(
            mean,
            device=img.device,
            dtype=img.dtype
        ).view(1, 3, 1, 1)

        std = torch.as_tensor(
            std,
            device=img.device,
            dtype=img.dtype
        ).view(1, 3, 1, 1)

        img = img * std + mean

    img = img.clamp(0.0, 1.0)


    # =========================================================
    # 2. RGB -> Gray
    # =========================================================
    r = img[:, 0:1]
    g = img[:, 1:2]
    b = img[:, 2:3]

    gray = (
        0.299 * r
        + 0.587 * g
        + 0.114 * b
    )


    # =========================================================
    # 3. Contrast
    # =========================================================
    contrast = (
        gray
        .flatten(1)
        .std(
            dim=1,
            unbiased=False
        )
        * 2.0
    ).clamp(
        0.0,
        1.0
    )


    # =========================================================
    # 4. Brightness
    # =========================================================
    brightness = (
        gray
        .flatten(1)
        .mean(dim=1)
    ).clamp(
        0.0,
        1.0
    )


    # =========================================================
    # 5. Saturation
    # =========================================================
    max_rgb, _ = img.max(
        dim=1,
        keepdim=True
    )

    min_rgb, _ = img.min(
        dim=1,
        keepdim=True
    )

    saturation = (
        (max_rgb - min_rgb)
        / (max_rgb + eps)
    )

    saturation = (
        saturation
        .flatten(1)
        .mean(dim=1)
        .clamp(
            0.0,
            1.0
        )
    )


    # =========================================================
    # 6. Edge Density
    # =========================================================
    sobel_x = torch.tensor(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ],
        device=img.device,
        dtype=img.dtype
    ).view(
        1,
        1,
        3,
        3
    )

    sobel_y = torch.tensor(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],
        device=img.device,
        dtype=img.dtype
    ).view(
        1,
        1,
        3,
        3
    )

    gx = F.conv2d(
        gray,
        sobel_x,
        padding=1
    )

    gy = F.conv2d(
        gray,
        sobel_y,
        padding=1
    )

    edge_map = torch.sqrt(
        gx ** 2
        + gy ** 2
        + eps
    )

    edge_density = (
        edge_map
        .flatten(1)
        .mean(dim=1)
        / 4.0
    ).clamp(
        0.0,
        1.0
    )


    # =========================================================
    # 7. Dark Channel
    # =========================================================
    min_channel, _ = img.min(
        dim=1,
        keepdim=True
    )

    dark_map = -F.max_pool2d(
        -min_channel,
        kernel_size=15,
        stride=1,
        padding=7
    )

    dark_channel = (
        dark_map
        .flatten(1)
        .mean(dim=1)
        .clamp(
            0.0,
            1.0
        )
    )


    # =========================================================
    # 8. Feature Vector
    #
    # Order:
    # 0 contrast
    # 1 edge_density
    # 2 brightness
    # 3 saturation
    # 4 dark_channel
    # =========================================================
    feat = torch.stack(
        [
            contrast,
            edge_density,
            brightness,
            saturation,
            dark_channel
        ],
        dim=1
    )

    return feat.detach()


def apply_handcrafted_ablation(
    feat,
    mode='all'
):
    """
    Apply handcrafted feature ablation.

    Feature order:
        0: contrast
        1: edge_density
        2: brightness
        3: saturation
        4: dark_channel


    Main leave-one-group-out modes
    --------------------------------

    all:
        Keep all five handcrafted features.

    no_handcrafted / none:
        Remove all handcrafted features.

    wo_structure:
        Remove:
            contrast
            edge density

        Keep:
            brightness
            saturation
            dark channel

    wo_light_color:
        Remove:
            brightness
            saturation

        Keep:
            contrast
            edge density
            dark channel

    wo_dark_channel:
        Remove:
            dark channel

        Keep:
            contrast
            edge density
            brightness
            saturation


    Legacy modes
    --------------------------------

    structure:
        Keep only:
            contrast
            edge density

    light_color:
        Keep only:
            brightness
            saturation

    dark_channel:
        Keep only:
            dark channel
    """

    # =========================================================
    # All features
    # =========================================================
    if mode == 'all':
        return feat


    # =========================================================
    # Remove all handcrafted features
    # =========================================================
    if mode in (
        'no_handcrafted',
        'none'
    ):
        return torch.zeros_like(
            feat
        )


    # =========================================================
    # Leave-one-group-out ablations
    # =========================================================
    out = feat.clone()


    # ---------------------------------------------------------
    # w/o Structure
    #
    # Remove:
    #   0 contrast
    #   1 edge density
    # ---------------------------------------------------------
    if mode == 'wo_structure':

        out[:, 0] = 0
        out[:, 1] = 0

        return out


    # ---------------------------------------------------------
    # w/o Light / Color
    #
    # Remove:
    #   2 brightness
    #   3 saturation
    # ---------------------------------------------------------
    if mode == 'wo_light_color':

        out[:, 2] = 0
        out[:, 3] = 0

        return out


    # ---------------------------------------------------------
    # w/o Dark Channel
    #
    # Remove:
    #   4 dark channel
    # ---------------------------------------------------------
    if mode == 'wo_dark_channel':

        out[:, 4] = 0

        return out


    # =========================================================
    # Legacy modes
    #
    # These modes mean:
    # "keep only this feature group"
    # =========================================================
    legacy_out = torch.zeros_like(
        feat
    )


    # ---------------------------------------------------------
    # Structure only
    # ---------------------------------------------------------
    if mode == 'structure':

        legacy_out[:, 0] = feat[:, 0]
        legacy_out[:, 1] = feat[:, 1]

        return legacy_out


    # ---------------------------------------------------------
    # Light / Color only
    # ---------------------------------------------------------
    if mode == 'light_color':

        legacy_out[:, 2] = feat[:, 2]
        legacy_out[:, 3] = feat[:, 3]

        return legacy_out


    # ---------------------------------------------------------
    # Dark Channel only
    # ---------------------------------------------------------
    if mode == 'dark_channel':

        legacy_out[:, 4] = feat[:, 4]

        return legacy_out


    # =========================================================
    # Unknown mode
    # =========================================================
    raise ValueError(
        f'Unknown handcrafted_feature_mode: {mode}'
    )


class FogEstimatorHead(nn.Module):
    """
    Smoke density estimation head.

    Semantic feature:
        GAP(F)

    Handcrafted features:
        contrast
        edge density
        brightness
        saturation
        dark channel

    Fusion:
        [GAP(F), handcrafted]
        -> MLP
        -> Sigmoid
    """

    def __init__(
        self,
        in_channels,
        handcrafted_dim=5,
        hidden_dim=128,
        img_mean=None,
        img_std=None,
        handcrafted_feature_mode='all'
    ):
        super().__init__()

        self.img_mean = img_mean
        self.img_std = img_std

        self.handcrafted_dim = int(
            handcrafted_dim
        )

        self.handcrafted_feature_mode = (
            handcrafted_feature_mode
        )

        # =====================================================
        # Semantic feature pooling
        # =====================================================
        self.pool = nn.AdaptiveAvgPool2d(
            1
        )


        # =====================================================
        # Smoke density estimation MLP
        # =====================================================
        self.net = nn.Sequential(

            nn.Linear(
                in_channels
                + self.handcrafted_dim,
                hidden_dim
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                hidden_dim,
                1
            ),

            nn.Sigmoid(),
        )


    def forward(
        self,
        feat,
        img
    ):

        # =====================================================
        # 1. Semantic feature
        # =====================================================
        deep = self.pool(
            feat
        ).flatten(
            1
        )


        # =====================================================
        # 2. Handcrafted features
        # =====================================================

        # -----------------------------------------------------
        # w/o all handcrafted features
        #
        # Keep dimensionality unchanged,
        # but provide zero handcrafted information.
        # -----------------------------------------------------
        if self.handcrafted_feature_mode in (
            'no_handcrafted',
            'none'
        ):

            hand = deep.new_zeros(
                (
                    deep.size(0),
                    self.handcrafted_dim
                )
            )

        else:

            # -------------------------------------------------
            # Compute all five handcrafted features
            # -------------------------------------------------
            hand = compute_handcrafted_smoke_features(
                img,
                self.img_mean,
                self.img_std
            )


            # -------------------------------------------------
            # Apply feature ablation
            # -------------------------------------------------
            hand = apply_handcrafted_ablation(
                hand,
                self.handcrafted_feature_mode
            )


        # =====================================================
        # 3. Safety check
        # =====================================================
        if hand.size(1) != self.handcrafted_dim:

            raise ValueError(
                'Handcrafted feature dimension mismatch: '
                f'got {hand.size(1)}, '
                f'expected {self.handcrafted_dim}'
            )


        # =====================================================
        # 4. Semantic + Handcrafted
        # =====================================================
        x = torch.cat(
            [
                deep,
                hand
            ],
            dim=1
        )


        # =====================================================
        # 5. Smoke density prediction
        # =====================================================
        smoke_score = self.net(
            x
        ).squeeze(
            1
        )

        return smoke_score