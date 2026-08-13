"""Policy observation encoder using an external visual-focus model."""

import os
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from seeker.model.encoder.focus_view_transform import (
    BOX_FILM_MODES,
    FocusViewTransform,
    VISUAL_FOCUS_RECORD_MODES,
)
from seeker.model.common.normalizer import MultiRobotLinearNormalizer
from seeker.model.common.resnet import build_resnet18_backbone
from seeker.focus_toolbox.prediction import VisualFocusPrediction, VisualFocusRecord
from seeker.model.encoder.obs_input import EncoderInputs, ObsInputProcessor
from seeker.util.visualization import visualize
from seeker.focus_toolbox.geometry import normalize_box


class FocusConditionedObsEncoder(ObsInputProcessor):
    """Encode observations with focus-guided crops and ResNet backbones."""

    def __init__(
        self,
        *,
        feat_dim: int = 64,
        n_hidden: int = 128,
        log_freq: int = 1000,
        resnet_pretrained_imagenet: bool = True,
        focus_view_transform: Optional[dict] = None,
    ):
        super().__init__()

        self.log_freq = int(log_freq)
        if focus_view_transform is None:
            raise ValueError(
                "FocusConditionedObsEncoder requires a focus_view_transform config."
            )

        self.focus_view_transform = FocusViewTransform(
            dict(focus_view_transform), verbose=False
        )
        self.policy_normalizer = MultiRobotLinearNormalizer()

        self.enable_eih = self.focus_view_transform.enable_eih

        # Per-view visual encoders
        out_res = self.focus_view_transform.out_res
        cnn_input_shape = (3, out_res, out_res)
        self.agentview_cnn = ResNet18Encoder(
            input_shape=cnn_input_shape,
            out_dim=feat_dim,
            pretrained_imagenet=resnet_pretrained_imagenet,
        )
        out_feature_dim = feat_dim + 11
        self.eih_cnn: Optional[nn.Module] = None
        if self.enable_eih:
            self.eih_cnn = ResNet18Encoder(
                input_shape=cnn_input_shape,
                out_dim=feat_dim,
                pretrained_imagenet=resnet_pretrained_imagenet,
            )
            out_feature_dim += feat_dim

        self.out_proj = nn.Linear(out_feature_dim, n_hidden)

        # Box FiLM conditioning using normalized (cx, cy, scale)
        self.box_film = nn.Linear(3, feat_dim * 2)
        self.count = 0
        self.last_video_boxes = []
        self.last_visual_focus_records: list[VisualFocusRecord] = []

        # Keep preprocessing aligned with the focus transform/Seeker Input Resn.
        self.input_res = self.focus_view_transform.vit_in
        self.viz_dir = None

    def initialize_internal(
        self, dataset_size: int, device: torch.device, viz_dir: Optional[str] = None
    ):
        """Initialize focus cache and optional visualization directory."""
        self.focus_view_transform.initialize_buffer(dataset_size, device)
        self.viz_dir = viz_dir

    def forward(self, obs, obs_index=None, oracle_info=None):
        """Run focus-conditioned observation encoding and ResNet encoding."""
        enc_in = self.process_obs(obs)

        views = {"agentview": enc_in.agentview}
        if self.enable_eih:
            views["eye_in_hand"] = enc_in.eye_in_hand

        processed_views = self.focus_view_transform(
            views,
            composer_in=enc_in.composer_in,
            obs_index=obs_index,
            oracle_info=oracle_info,
        )
        self._record_video_boxes(processed_views, enc_in.T)

        agent_ret = processed_views["agentview"]
        agent_image = agent_ret["image"]
        agent_box_px = agent_ret["box_px"]

        eih_image, eih_box_px = None, None
        if self.enable_eih:
            eih_ret = processed_views["eye_in_hand"]
            eih_image, eih_box_px = eih_ret["image"], eih_ret["box_px"]

        feats = [enc_in.proprio]
        feat = self.encode_agentview(
            agent_image,
            agent_box_px,
        )
        feats.append(feat)
        if self.enable_eih:
            feat = self.encode_eih(eih_image, eih_box_px)
            feats.append(feat)
        feat = self.out_proj(torch.cat(feats, dim=-1))

        views = [[agent_image, None, None, None]]
        if self.enable_eih:
            views.append([eih_image, None, None, None])
        self.visualize(views, enc_in.T)
        self.count += 1
        return feat

    def _record_video_boxes(self, processed_views, T: int) -> None:
        """Store latest-step focus records for rollout video overlays."""
        B = next(iter(processed_views.values()))["box_px"].shape[0] // int(T)
        out = []
        records: list[VisualFocusRecord] = []
        for view, ret in processed_views.items():
            box_px = ret.get("box_px")
            if box_px is None:
                continue

            mode = self.focus_view_transform.view_modes.get(view, "disabled")
            if mode in ("pass_through", "random_overlay"):
                continue
            if mode in VISUAL_FOCUS_RECORD_MODES:
                source = ret["visual_focus"].source if ret.get("visual_focus") else "seeker"
            else:
                source = mode

            last_box = box_px.view(B, int(T), 4)[:, -1].detach()
            out.append(
                {
                    "source": source,
                    "view": view,
                    "source_size": self.focus_view_transform.vit_in,
                    "box_px": last_box,
                }
            )
            prediction = VisualFocusPrediction(box_px=last_box, source=source)
            records.append(
                VisualFocusRecord(
                    source=source,
                    view=view,
                    timestep=int(T) - 1,
                    prediction=prediction,
                    image_size=(
                        int(self.focus_view_transform.vit_in),
                        int(self.focus_view_transform.vit_in),
                    ),
                )
            )
        self.last_video_boxes = out
        self.last_visual_focus_records = records

    def set_normalizer(self, normalizer) -> None:
        """Set local policy normalizer while preserving Seeker focus normalization."""
        self.policy_normalizer.load_state_dict(normalizer.state_dict())
        self.focus_view_transform.set_normalizer(normalizer)

    def encode_agentview(self, image, box_px):
        """Encode agentview image and optionally FiLM-condition on Seeker box."""
        mode = self.focus_view_transform.view_modes["agentview"]
        feat = self.agentview_cnn(image)
        if mode in BOX_FILM_MODES:
            feat = self.apply_box_film(feat, box_px)
        return feat

    def apply_box_film(self, feat, box_px):
        """Condition visual features on normalized focus box geometry."""
        normed_box = normalize_box(box_px, self.focus_view_transform.vit_in)
        gamma, beta = self.box_film(normed_box).chunk(2, dim=-1)
        return feat * (1 + gamma) + beta

    def encode_eih(self, image, box_px):
        """Encode eye-in-hand image when enabled."""
        if image is None:
            return None
        feat = self.eih_cnn(image)
        mode = self.focus_view_transform.view_modes["eye_in_hand"]
        if mode in BOX_FILM_MODES:
            feat = self.apply_box_film(feat, box_px)
        return feat

    def process_obs(self, obs: Dict[str, torch.Tensor]) -> EncoderInputs:
        """Normalize and format observations for the focus view transform."""
        enc_in = self.obs_to_input(
            obs,
            self.policy_normalizer,
            resize=True,
            composer_normalizer=self.focus_view_transform.normalizer,
        )
        if self.focus_view_transform.view_modes["agentview"] == "lowres_crop_only":
            enc_in.agentview = self.degrade_obs_resolution(enc_in.agentview)
        if (
            self.enable_eih
            and self.focus_view_transform.view_modes["eye_in_hand"]
            == "lowres_crop_only"
        ):
            enc_in.eye_in_hand = self.degrade_obs_resolution(enc_in.eye_in_hand)
        return enc_in

    def degrade_obs_resolution(self, image: torch.Tensor) -> torch.Tensor:
        """Remove image detail by downsampling to low_res, then restoring vit_in."""
        low_res = self.focus_view_transform.low_res
        vit_in = self.focus_view_transform.vit_in
        image = F.interpolate(
            image,
            size=(low_res, low_res),
            mode="bilinear",
            align_corners=False,
        )
        return F.interpolate(
            image,
            size=(vit_in, vit_in),
            mode="bilinear",
            align_corners=False,
        )

    def visualize(self, views, T):
        """Optionally save visualization grids during training."""
        if not self.training:
            return
        if self.log_freq <= 0:
            return

        viz_dir = self.viz_dir
        if viz_dir is None:
            viz_dir = "./visualization_temp"
        viz_dir = os.path.join(viz_dir, "train" if self.training else "val")
        freq = self.log_freq if self.training else 1
        if self.count % freq == 0:
            visualize(
                views,
                temporal_dim=T,
                num_viz=16,
                padding=2,
                save_dir=viz_dir,
                step=self.count,
            )

    def get_config(self) -> dict:
        """Expose focus model config for logging/debugging."""
        return self.focus_view_transform.get_config()


class ResNet18Encoder(nn.Module):
    """Lightweight ResNet-18 feature encoder."""

    def __init__(
        self,
        *,
        input_shape,
        out_dim: int,
        pretrained_imagenet: bool = True,
    ):
        super().__init__()

        backbone = build_resnet18_backbone(
            pretrained_imagenet=pretrained_imagenet,
        )

        # Keep convolutional trunk only; pooling/head are defined below.
        self.trunk = nn.Sequential(*list(backbone.children())[:-2])

        with torch.no_grad():
            feat = self.trunk(torch.zeros(1, *input_shape))
            c = feat.shape[1]
            assert c == 512

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(c, out_dim)

    def forward(self, x):
        x = self.trunk(x)
        x = self.pool(x)
        return self.head(x.view(x.shape[0], -1))
