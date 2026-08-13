"""Per-view focus transforms for the v1.0 Seeker backend."""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from seeker.focus_toolbox.geometry import crop_with_box, grid_mask_to_pixel_box
from seeker.focus_toolbox.oracle_baseline import (
    replace_invalid_boxes,
    target_patch_tight_boxes,
)
from seeker.focus_toolbox.prediction import VisualFocusPrediction
from seeker.model.common.augmentation import BackgroundOverlay, CropRandomizer
from seeker.model.common.normalizer import MultiRobotLinearNormalizer
from seeker.model.rvt2_heatmap import RVT2Heatmap
from seeker.model.seeker import Seeker
from seeker.util.formatting import pretty_print_nested


MODE_OPERATION = {
    "pass_through": "pass_through",
    "focus_condition": "focus_condition",
    "focus_crop": "focus_crop",
    "lowres_crop_only": "focus_crop",
    "focus_mask": "focus_mask",
    "focus_mask_crop": "focus_mask_crop",
    "random_overlay": "random_overlay",
    "disabled": "disabled",
}
VALID_FOCUS_MODES = set(MODE_OPERATION)
FOCUS_BACKENDS = {"seeker", "rvt2_heatmap", "oracle"}
NO_FOCUS_MODES = {"pass_through", "random_overlay"}
FOCUS_OPERATION_MODES = {
    mode
    for mode, operation in MODE_OPERATION.items()
    if operation in {"focus_condition", "focus_crop", "focus_mask", "focus_mask_crop"}
}
FOCUS_CROP_MODES = {
    mode for mode, operation in MODE_OPERATION.items() if operation == "focus_crop"
}
FOCUS_MASK_CROP_MODES = {
    mode for mode, operation in MODE_OPERATION.items() if operation == "focus_mask_crop"
}
MASK_CACHE_MODES = {"focus_mask", "focus_mask_crop"}
GUIDED_OVERLAY_MODES = {"focus_mask", "focus_mask_crop"}
BOX_FILM_MODES = {
    mode
    for mode, operation in MODE_OPERATION.items()
    if operation in {"focus_condition", "focus_crop", "focus_mask_crop"}
}
VISUAL_FOCUS_RECORD_MODES = FOCUS_OPERATION_MODES


class FocusViewTransform(nn.Module):
    """Apply v1.0 Seeker crop/mask pipelines for each enabled camera view."""

    def __init__(self, config: Dict[str, Any], *, verbose: bool = False):
        super().__init__()

        config = dict(config)
        self.cfg = config
        self.verbose = bool(verbose)

        self.source_cfg = dict(config["source"])
        self.focus_source = str(self.source_cfg["name"]).strip().lower()
        if self.focus_source in {"", "null", "none"}:
            self.focus_source = "none"
        if self.focus_source == "rvt2":
            self.focus_source = "rvt2_heatmap"
        self.seeker_weights = self.source_cfg.get("weights")
        self.seeker_strict_weights = bool(self.source_cfg.get("strict_weights", True))
        self.vit_in = int(config.get("vit_in"))
        self.low_res = int(config.get("low_res"))
        self.out_res = int(config.get("out_res"))

        overlay_cfg = dict(config.get("overlay", {}))
        guided_overlay_cfg = dict(overlay_cfg.get("guided", {}))
        random_overlay_cfg = dict(overlay_cfg.get("random", {}))

        self.guided_overlay_prob = float(guided_overlay_cfg.get("prob", 0.0))
        self.guided_overlay_noise_std = float(guided_overlay_cfg.get("noise_std", 0.0))
        self.guided_overlay_alpha_min = float(guided_overlay_cfg.get("alpha_min", 0.3))
        self.guided_overlay_alpha_max = float(guided_overlay_cfg.get("alpha_max", 0.8))
        self.guided_overlay_warmup_steps = int(guided_overlay_cfg.get("warmup_steps", 0))
        self.guided_overlay_background_path = guided_overlay_cfg.get("background_path")

        self.random_overlay_prob = float(random_overlay_cfg.get("prob", 0.0))
        self.random_overlay_alpha_min = float(random_overlay_cfg.get("alpha_min", 0.75))
        self.random_overlay_alpha_max = float(random_overlay_cfg.get("alpha_max", 0.75))
        self.random_overlay_warmup_steps = int(random_overlay_cfg.get("warmup_steps", 0))
        self.random_overlay_background_path = random_overlay_cfg.get("background_path")

        self.box_jitter = 0.05
        self.box_margin_px = 8
        self.oracle_camera_by_view = {
            str(k): str(v)
            for k, v in dict(
                self.source_cfg.get("camera_by_view", {"agentview": "agentview"})
            ).items()
        }

        self.views = []
        self.view_modes = {}
        for view, view_cfg in dict(config["views"]).items():
            if view_cfg is None:
                mode = "pass_through"
            elif isinstance(view_cfg, str):
                mode = view_cfg
            else:
                mode = str(view_cfg.get("mode", "pass_through"))
            if mode not in VALID_FOCUS_MODES:
                raise ValueError(f"FocusViewTransform: invalid mode {mode!r} for {view!r}")
            if mode == "disabled":
                continue
            self.views.append(view)
            self.view_modes[view] = mode

        if not self.views:
            raise ValueError("FocusViewTransform: no enabled views")
        if "agentview" not in self.views:
            raise ValueError("FocusViewTransform requires 'agentview'")

        self.enable_eih = "eye_in_hand" in self.views
        self.focus_views = [
            view
            for view, mode in self.view_modes.items()
            if mode in FOCUS_OPERATION_MODES
        ]
        if self.focus_views and self.focus_source not in FOCUS_BACKENDS:
            valid = ", ".join(sorted(FOCUS_BACKENDS))
            raise ValueError(
                f"focus_view_transform.source={self.focus_source!r} cannot provide "
                f"focus boxes; expected one of {valid}"
            )

        self.uses_seeker = bool(self.focus_views) and self.focus_source == "seeker"
        self.uses_rvt2_heatmap = (
            bool(self.focus_views) and self.focus_source == "rvt2_heatmap"
        )
        self.uses_oracle = bool(self.focus_views) and self.focus_source == "oracle"

        if self.uses_rvt2_heatmap:
            for view in self.focus_views:
                if view != "agentview":
                    raise ValueError("RVT2Heatmap focus currently supports agentview only")
        if self.uses_oracle:
            for view in self.focus_views:
                if view != "agentview":
                    raise ValueError("Oracle focus currently supports agentview only")
        self.uses_random_overlay = any(
            mode == "random_overlay" for mode in self.view_modes.values()
        )
        self.uses_masked_overlay = any(
            mode in GUIDED_OVERLAY_MODES for mode in self.view_modes.values()
        )
        self.uses_guided_overlay = (
            self.guided_overlay_prob > 0.0 and self.uses_masked_overlay
        )
        self.uses_random_bg_overlay = (
            self.random_overlay_prob > 0.0 and self.uses_random_overlay
        )
        self.uses_overlay = self.uses_guided_overlay or self.uses_random_bg_overlay

        self.seeker: Optional[Seeker] = None
        if self.uses_seeker:
            if not self.seeker_weights:
                raise ValueError(
                    "FocusViewTransform requires v1.0 Seeker weights. "
                    "Set focus_view_transform.source.weights."
                )
            if not os.path.isfile(self.seeker_weights):
                raise FileNotFoundError(
                    f"Seeker checkpoint not found: {self.seeker_weights}"
                )
            seeker_views = list(self.source_cfg.get("checkpoint_views", self.views))
            self.seeker = Seeker(
                weights=self.seeker_weights,
                views=seeker_views,
                verbose=False,
                strict_weights=self.seeker_strict_weights,
            )
            self.seeker.eval()

        self.normalizer = MultiRobotLinearNormalizer()
        if self.seeker is not None:
            self.normalizer.load_state_dict(self.seeker.normalizer.state_dict())

        self.patch_size = int(getattr(self.seeker, "patch_size", 16))
        self.grid_res = self.vit_in // self.patch_size

        self.rvt2_heatmap: Optional[RVT2Heatmap] = None
        if self.uses_rvt2_heatmap:
            self.rvt2_heatmap = RVT2Heatmap(
                checkpoint=self.source_cfg.get("checkpoint"),
                vit_in=self.vit_in,
            )
            self.patch_size = int(self.rvt2_heatmap.patch_size)
            self.grid_res = int(self.rvt2_heatmap.grid_res)

        self.cached_views = [
            view for view in self.views if self.view_modes[view] not in NO_FOCUS_MODES
        ]
        self.mask_cached_views = [
            view for view in self.cached_views if self.view_modes[view] in MASK_CACHE_MODES
        ]
        self.view_to_box_idx = {view: i for i, view in enumerate(self.cached_views)}
        self.view_to_mask_idx = {
            view: i for i, view in enumerate(self.mask_cached_views)
        }

        self.guided_background_overlay = None
        if self.uses_guided_overlay:
            if not self.guided_overlay_background_path:
                raise ValueError(
                    "focus_view_transform.overlay.guided.background_path is required "
                    "when guided overlay probability is positive."
                )
            alpha_mid = (
                self.guided_overlay_alpha_min + self.guided_overlay_alpha_max
            ) / 2.0
            self.guided_background_overlay = BackgroundOverlay(
                {
                    "prob": self.guided_overlay_prob,
                    "alpha": [alpha_mid, alpha_mid],
                    "background_path": self.guided_overlay_background_path,
                },
                cache_res=self.vit_in,
            )

        self.random_background_overlay = None
        if self.uses_random_bg_overlay:
            if not self.random_overlay_background_path:
                raise ValueError(
                    "focus_view_transform.overlay.random.background_path is required "
                    "when random overlay probability is positive."
                )
            self.random_background_overlay = BackgroundOverlay(
                {
                    "prob": self.random_overlay_prob,
                    "alpha": [
                        self.random_overlay_alpha_min,
                        self.random_overlay_alpha_max,
                    ],
                    "background_path": self.random_overlay_background_path,
                },
                cache_res=self.vit_in,
            )

        self.crop_randomizer = CropRandomizer(
            input_shape=(3, self.low_res, self.low_res),
            crop_size=self.out_res,
        )
        self.buffer_valid: Optional[torch.Tensor] = None
        self.box_buffer: Optional[torch.Tensor] = None
        self.mask_buffer: Optional[torch.Tensor] = None
        self.counter = 0

        if verbose:
            pretty_print_nested(self.get_config(), title="FocusViewTransform")

    def initialize_buffer(self, buffer_size: int, device: torch.device):
        self.buffer_valid = torch.zeros((buffer_size,), dtype=torch.uint8, device=device)
        self.box_buffer = torch.zeros(
            (buffer_size, len(self.cached_views), 4), dtype=torch.uint8, device=device
        )
        if self.mask_cached_views:
            self.mask_buffer = torch.zeros(
                (buffer_size, len(self.mask_cached_views), self.grid_res * self.grid_res),
                dtype=torch.uint8,
                device=device,
            )
        else:
            self.mask_buffer = None

    def set_normalizer(self, normalizer: MultiRobotLinearNormalizer) -> None:
        if self.seeker is None:
            self.normalizer.load_state_dict(normalizer.state_dict())

    def retrieve_from_buffer(
        self, obs_index: torch.Tensor
    ) -> Optional[Dict[str, VisualFocusPrediction]]:
        if self.buffer_valid is None or self.box_buffer is None:
            return None
        obs_index = obs_index.to(torch.int64)
        if (self.buffer_valid[obs_index] == 0).any():
            return None

        out = {}
        for view in self.cached_views:
            box_u8 = self.box_buffer[obs_index, self.view_to_box_idx[view]]
            box_px = (box_u8.float() / 255.0) * float(self.vit_in - 1)
            mask_grid = None
            if view in self.view_to_mask_idx:
                if self.mask_buffer is None:
                    return None
                mask_u8 = self.mask_buffer[obs_index, self.view_to_mask_idx[view]]
                mask_grid = (mask_u8.float() / 255.0).view(
                    -1, 1, self.grid_res, self.grid_res
                )
            out[view] = VisualFocusPrediction(
                box_px=box_px,
                mask_grid=mask_grid,
                source=self._source_for_view(view),
            )
        return out

    def fill_buffer(self, *, obs_index: torch.Tensor, payloads: Dict[str, VisualFocusPrediction]):
        if self.buffer_valid is None or self.box_buffer is None:
            return
        obs_index = obs_index.to(torch.int64)
        for view, value in payloads.items():
            box01 = (value.box_px / float(self.vit_in - 1)).clamp(0.0, 1.0)
            self.box_buffer[obs_index, self.view_to_box_idx[view]] = (
                box01 * 255.0
            ).round().to(torch.uint8)
            if view in self.view_to_mask_idx:
                assert value.mask_grid is not None
                assert self.mask_buffer is not None
                mask_u8 = (value.mask_grid.clamp(0.0, 1.0) * 255.0).round()
                self.mask_buffer[obs_index, self.view_to_mask_idx[view]] = (
                    mask_u8.to(torch.uint8).squeeze(1).flatten(1)
                )
        self.buffer_valid[obs_index] = 1

    @torch.no_grad()
    def infer_all_visual_focus(
        self,
        *,
        images_vit_by_view: Dict[str, torch.Tensor],
        composer_in: dict,
        obs_index: Optional[torch.Tensor],
        oracle_info: Optional[dict] = None,
    ) -> Dict[str, Optional[VisualFocusPrediction]]:
        if self.training and obs_index is not None:
            cached = self.retrieve_from_buffer(obs_index)
            if cached is not None:
                return {
                    view: None if self.view_modes[view] in NO_FOCUS_MODES else cached[view]
                    for view in self.views
                }

        payloads = {}
        out = {}
        for view in self.views:
            mode = self.view_modes[view]
            if mode in NO_FOCUS_MODES:
                out[view] = None
                continue
            if mode in FOCUS_OPERATION_MODES:
                prediction = self.predict_visual_focus(
                    view=view,
                    image=images_vit_by_view[view],
                    composer_in=composer_in,
                    oracle_info=oracle_info,
                )
                payloads[view] = prediction
                out[view] = prediction
                continue

        if self.training and obs_index is not None and self.buffer_valid is not None:
            self.fill_buffer(obs_index=obs_index, payloads=payloads)
        return out

    def predict_visual_focus(
        self,
        *,
        view: str,
        image: torch.Tensor,
        composer_in: dict,
        oracle_info: Optional[dict],
    ) -> VisualFocusPrediction:
        if self.focus_source == "seeker":
            if self.seeker is None:
                raise RuntimeError("Seeker backend is not initialized")
            default_box = torch.tensor(
                [[0.0, 0.0, float(self.vit_in - 1), float(self.vit_in - 1)]],
                device=image.device,
            ).expand(image.shape[0], -1)
            mask_grid = self.seeker(
                image=image,
                view=view,
                composer_in=composer_in,
            ).final.mask
            box_px = grid_mask_to_pixel_box(mask_grid.squeeze(1), default_box)
            mask_grid = mask_grid / (
                mask_grid.amax(dim=(-2, -1), keepdim=True) + 1e-6
            )
            return VisualFocusPrediction(
                box_px=box_px,
                mask_grid=mask_grid.clamp(0.0, 1.0),
                source="seeker",
            )

        if self.focus_source == "rvt2_heatmap":
            if self.rvt2_heatmap is None:
                raise RuntimeError("RVT2Heatmap backend is not initialized")
            return self.rvt2_heatmap.predict_visual_focus(
                image=image,
                composer_in=composer_in,
                view_name=view,
            )

        if self.focus_source == "oracle":
            return self.oracle_visual_focus(
                view=view,
                oracle_info=oracle_info,
                batch_size=image.shape[0],
                device=image.device,
            )

        raise RuntimeError(f"Unhandled focus source: {self.focus_source}")

    def process_view(
        self,
        *,
        view: str,
        image_vit: torch.Tensor,
        visual_focus: Optional[VisualFocusPrediction],
    ) -> Dict[str, Optional[torch.Tensor]]:
        mode = self.view_modes[view]
        default_box_px = torch.tensor(
            [[0.0, 0.0, float(self.vit_in - 1), float(self.vit_in - 1)]],
            device=image_vit.device,
        ).expand(image_vit.shape[0], -1)

        if mode == "pass_through":
            return {
                "image": self.lowres_crop(image_vit),
                "box_px": default_box_px,
                "visual_focus": None,
            }
        if mode == "random_overlay":
            image_aug = self.overlay(image_vit, None)
            return {
                "image": self.lowres_crop(image_aug),
                "box_px": default_box_px,
                "visual_focus": None,
            }

        assert visual_focus is not None
        operation = MODE_OPERATION[mode]
        if operation == "focus_condition":
            return self.focus_condition(image_vit, visual_focus)
        if operation == "focus_crop":
            return self.focus_crop(image_vit, visual_focus)
        if operation == "focus_mask_crop":
            return self.focus_mask_crop(image_vit, visual_focus)
        if operation == "focus_mask":
            return self.focus_mask(image_vit, visual_focus, default_box_px)

        raise RuntimeError(f"Unhandled focus operation {operation!r} for mode {mode!r}")

    def focus_condition(
        self,
        image_vit: torch.Tensor,
        visual_focus: VisualFocusPrediction,
    ) -> Dict[str, Optional[torch.Tensor]]:
        return {
            "image": self.lowres_crop(image_vit),
            "box_px": visual_focus.box_px,
            "visual_focus": visual_focus,
        }

    def focus_crop(
        self,
        image_vit: torch.Tensor,
        visual_focus: VisualFocusPrediction,
    ) -> Dict[str, Optional[torch.Tensor]]:
        crop, _, box_px = self.box_crop(image_vit, None, visual_focus.box_px)
        return {"image": crop, "box_px": box_px, "visual_focus": visual_focus}

    def focus_mask(
        self,
        image_vit: torch.Tensor,
        visual_focus: VisualFocusPrediction,
        default_box_px: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        mask_px = self.upsample_mask(visual_focus.mask_grid)
        image_aug = self.overlay(image_vit, mask_px)
        return {
            "image": self.lowres_crop(image_aug),
            "box_px": default_box_px,
            "visual_focus": visual_focus,
        }

    def focus_mask_crop(
        self,
        image_vit: torch.Tensor,
        visual_focus: VisualFocusPrediction,
    ) -> Dict[str, Optional[torch.Tensor]]:
        mask_px = self.upsample_mask(visual_focus.mask_grid)
        crop, mask_px, box_px = self.box_crop(image_vit, mask_px, visual_focus.box_px)
        crop = self.overlay(crop, mask_px)
        return {"image": crop, "box_px": box_px, "visual_focus": visual_focus}

    def lowres_crop(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] != (self.low_res, self.low_res):
            image = F.interpolate(image, size=(self.low_res, self.low_res), mode="bilinear")
        return self.crop_randomizer(image)

    def upsample_mask(self, mask_grid: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            mask_grid,
            size=(self.vit_in, self.vit_in),
            mode="bilinear",
            align_corners=False,
        )

    def box_crop(
        self,
        image: torch.Tensor,
        mask_px: Optional[torch.Tensor],
        box_px: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        jitter = self.box_jitter if self.training else 0.0
        return crop_with_box(
            image=image,
            box=box_px,
            mask=mask_px,
            output_size=(self.out_res, self.out_res),
            box_jitter=jitter,
            margin=self.box_margin_px,
        )

    def _source_for_view(self, view: str) -> str:
        _ = view
        return self.focus_source

    def _oracle_camera_for_view(self, view: str) -> str:
        return self.oracle_camera_by_view.get(str(view), str(view))

    @staticmethod
    def _oracle_array(oracle_info: dict, names: list[str]) -> Optional[torch.Tensor]:
        for name in names:
            value = oracle_info.get(name)
            if value is not None:
                return value if torch.is_tensor(value) else torch.as_tensor(value)
        return None

    def oracle_visual_focus(
        self,
        *,
        view: str,
        oracle_info: Optional[dict],
        batch_size: int,
        device: torch.device,
    ) -> VisualFocusPrediction:
        if oracle_info is None:
            raise RuntimeError(
                f"{self.view_modes[view]} requires oracle_info. "
                "For training set dataset.include_oracle_info=true; for rollouts "
                "set task.env_runner.enable_oracle_focus_info=true."
            )

        camera = self._oracle_camera_for_view(view)
        mask = self._oracle_array(
            oracle_info,
            [
                f"target_patch_mask_{camera}",
                f"oracle_target_patch_mask_{camera}",
            ],
        )
        if mask is None:
            available = ", ".join(sorted(str(k) for k in oracle_info.keys()))
            raise RuntimeError(
                f"Missing oracle target patch mask for camera {camera!r}. "
                f"Available oracle keys: {available}"
            )

        mask = mask.to(device=device)
        if mask.dim() == 4:
            mask = mask.reshape(-1, *mask.shape[-2:])
        if mask.dim() != 3:
            raise RuntimeError(
                "Expected oracle target patch mask [N,S,S] or [B,T,S,S], "
                f"got {tuple(mask.shape)}"
            )
        if int(mask.shape[0]) != int(batch_size):
            raise RuntimeError(
                f"Oracle target patch mask frame count {mask.shape[0]} does not "
                f"match flattened image batch {batch_size}"
            )

        box_px = target_patch_tight_boxes(mask, image_size=self.vit_in)
        box_px = replace_invalid_boxes(box_px, image_size=self.vit_in)
        return VisualFocusPrediction(
            box_px=box_px.to(device=device),
            mask_grid=mask.float().unsqueeze(1),
            source=self._source_for_view(view),
            metadata={
                "camera": camera,
                "mode": self.view_modes[view],
            },
        )

    def overlay(self, image: torch.Tensor, mask_px: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.training:
            return image
        if mask_px is None:
            prob = self.random_overlay_prob
            if self.random_overlay_warmup_steps > 0:
                prob *= min(
                    1.0,
                    float(self.counter) / float(self.random_overlay_warmup_steps),
                )
            if self.random_background_overlay is None or prob <= 0.0:
                return image
            return self.random_background_overlay(image, prob=prob)

        prob = self.guided_overlay_prob
        if self.guided_overlay_warmup_steps > 0:
            prob *= min(
                1.0,
                float(self.counter) / float(self.guided_overlay_warmup_steps),
            )
        if self.guided_background_overlay is None or prob <= 0.0:
            return image

        mask = mask_px
        if mask.shape[-2:] != image.shape[-2:]:
            mask = F.interpolate(mask, size=image.shape[-2:], mode="bilinear")
        if self.guided_overlay_noise_std > 0:
            mask = (mask + torch.randn_like(mask) * self.guided_overlay_noise_std).clamp(
                0.0, 1.0
            )
        mask = mask.clamp(self.guided_overlay_alpha_min, self.guided_overlay_alpha_max)
        return self.guided_background_overlay(
            image, alpha=mask, prob=prob, replacement=True
        )

    def forward(
        self,
        view_imgs: Dict[str, torch.Tensor],
        composer_in: dict,
        obs_index: Optional[torch.Tensor] = None,
        oracle_info: Optional[dict] = None,
    ) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        assert set(view_imgs.keys()) == set(self.views), (
            f"Expected views {self.views}, got {list(view_imgs.keys())}"
        )
        images_vit_by_view = {}
        for view in self.views:
            image = view_imgs[view]
            if image.shape[-2:] != (self.vit_in, self.vit_in):
                image = F.interpolate(
                    image,
                    size=(self.vit_in, self.vit_in),
                    mode="bilinear",
                    align_corners=False,
                )
            images_vit_by_view[view] = image

        focus_by_view = self.infer_all_visual_focus(
            images_vit_by_view=images_vit_by_view,
            composer_in=composer_in,
            obs_index=obs_index,
            oracle_info=oracle_info,
        )
        out = {
            view: self.process_view(
                view=view,
                image_vit=images_vit_by_view[view],
                visual_focus=focus_by_view[view],
            )
            for view in self.views
        }
        if self.training:
            self.counter += 1
        return out

    def get_config(self) -> dict:
        view_names = {"agentview": "Agent View", "eye_in_hand": "Eye-in-Hand"}
        mode_names = {
            "pass_through": "Pass Through",
            "focus_condition": "Focus Condition",
            "focus_crop": "Focus Crop",
            "lowres_crop_only": "Low-Res Crop",
            "focus_mask": "Focus Mask",
            "focus_mask_crop": "Focus Mask Crop",
            "random_overlay": "Random Overlay",
        }
        focus_cfg = {
            view_names.get(view, view.replace("_", " ").title()): mode_names.get(
                self.view_modes[view],
                self.view_modes[view].replace("_", " ").title(),
            )
            for view in self.views
        }

        overlay_lines = []
        if self.uses_guided_overlay:
            overlay_lines.append(
                f"guided {self.guided_overlay_prob:.2f} prob, "
                f"warmup {self.guided_overlay_warmup_steps}, "
                f"alpha "
                f"{self.guided_overlay_alpha_min:.2f}-"
                f"{self.guided_overlay_alpha_max:.2f}"
            )
        if self.uses_random_bg_overlay:
            overlay_lines.append(
                f"random {self.random_overlay_prob:.2f} prob, "
                f"warmup {self.random_overlay_warmup_steps}, "
                f"alpha "
                f"{self.random_overlay_alpha_min:.2f}-"
                f"{self.random_overlay_alpha_max:.2f}"
            )
        focus_cfg["Overlay"] = (
            "; ".join(overlay_lines) if overlay_lines else "Disabled"
        )

        crop_modes = FOCUS_CROP_MODES | FOCUS_MASK_CROP_MODES
        if any(mode in crop_modes for mode in self.view_modes.values()):
            crop = f"box jitter {100.0 * self.box_jitter:.1f}%"
            if self.box_margin_px:
                crop += f", margin {self.box_margin_px}px"
            focus_cfg["Crop"] = crop
        else:
            focus_cfg["Crop"] = "Disabled"

        focus_cfg["Source"] = self.focus_source.replace("_", " ").title()
        config = {"Focus Transform": focus_cfg}

        if self.uses_rvt2_heatmap:
            config["RVT2 Heatmap"] = {
                "Status": "Enabled",
                "Checkpoint": self.source_cfg.get("checkpoint"),
                "Zoom": (
                    None
                    if self.rvt2_heatmap is None
                    else f"{self.rvt2_heatmap.zoom:.1f}x"
                ),
            }
        elif self.uses_oracle:
            config["Oracle Focus"] = {
                "Status": "Enabled",
                "Camera By View": dict(self.oracle_camera_by_view),
            }
        elif self.uses_seeker:
            seeker_cfg = {
                "Status": "Enabled",
                "Weights": self.seeker_weights,
                "Strict Weights": self.seeker_strict_weights,
            }
            if self.seeker is not None:
                seeker_cfg.update(self.seeker.get_config())
            config["Seeker"] = seeker_cfg
        else:
            config["Visual Focus"] = "Disabled"
        return config
