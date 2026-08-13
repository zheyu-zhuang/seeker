"""Explicit oracle affordance target helpers for rerendered MimicGen caches."""

from __future__ import annotations

from typing import Optional

import numpy as np


COFFEE_POD_HOLDER_GEOMS = tuple(
    f"coffee_machine_pod_holder_cup_body_hc_{i}" for i in range(64)
)
THREADING_RING_GEOMS = tuple(f"tripod_obj_ring_{i}" for i in range(20))

GEOM_TYPE_IDS = {
    "plane": 0,
    "hfield": 1,
    "sphere": 2,
    "capsule": 3,
    "ellipsoid": 4,
    "cylinder": 5,
    "box": 6,
    "mesh": 7,
}


AFFORDANCE_REGISTRY = {
    "square": {
        "square_nut": {"object_name": "SquareNut"},
        "round_nut": {"object_name": "RoundNut"},
        "square_peg": {"body": "peg1"},
        "round_peg": {"body": "peg2"},
    },
    "coffee_preparation": {
        "drawer": {"body": "CabinetObject_drawer_link"},
        "mug": {"env_attr": "mug"},
        "coffee_pod": {"env_attr": "coffee_pod"},
        "coffee_machine": {
            "geoms": ("coffee_machine_base_g0",),
            "subtasks": {
                4: {
                    "body": "coffee_machine_pod_holder_root",
                    "geoms": COFFEE_POD_HOLDER_GEOMS,
                },
            },
        },
    },
    "coffee": {
        "drawer": {
            "geoms": (
                "CabinetObject_drawer_handle_1",
                "CabinetObject_drawer_handle_2",
                "CabinetObject_drawer_handle_3",
            ),
        },
        "coffee_machine": {
            "bodies": (
                "coffee_machine_pod_holder_root",
                "coffee_machine_lid_main",
            ),
            "geoms": (
                "coffee_machine_base_g0",
                "coffee_machine_lid_g0",
            ),
            "subtasks": {
                4: {
                    "body": "coffee_machine_pod_holder_root",
                    "geoms": COFFEE_POD_HOLDER_GEOMS,
                },
            },
        },
    },
    "stack_three": {
        "cubeA": {"env_attr": "cubeA"},
        "cubeB": {"env_attr": "cubeB"},
        "cubeC": {"env_attr": "cubeC"},
    },
    "stack": {
        "cubeA": {"env_attr": "cubeA"},
        "cubeB": {"env_attr": "cubeB"},
    },
    "threading": {
        "needle": {"env_attr": "needle"},
        "tripod": {"geoms": THREADING_RING_GEOMS},
    },
    "mug_cleanup": {
        "drawer": {
            "body": "DrawerObject_drawer_link",
            "geom_types": ("capsule",),
        },
        "object": {"env_attr": "cleanup_object"},
    },
    "pick_place": {
        "milk": {"object_name": "Milk"},
        "cereal": {"object_name": "Cereal"},
        "bread": {"object_name": "Bread"},
        "can": {"object_name": "Can"},
    },
    "three_piece_assembly": {
        "base": {"env_attr": "base"},
        "piece_1": {"env_attr": "piece_1"},
        "piece_2": {"env_attr": "piece_2"},
    },
}


class OracleAffordanceResolver:
    """Resolve explicit task targets and visible segmentation boxes."""

    def __init__(
        self,
        *,
        env,
        env_meta: dict,
        camera_name: str,
        resolution: int,
        patch_size: int,
    ) -> None:
        self.env = env
        self.env_meta = env_meta
        self.camera_name = str(camera_name)
        self.resolution = int(resolution)
        self.patch_size = int(patch_size)

    def affordance_spec(self, ref: str, subtask_idx: int) -> Optional[dict]:
        """Return the explicit target spec for this task/ref/subtask."""
        task_rules = self._task_rules()
        if task_rules is None:
            return None

        ref_spec = task_rules.get(str(ref))
        if ref_spec is None:
            return None

        subtask_rules = ref_spec.get("subtasks", {})
        matching = [
            int(start_idx)
            for start_idx in subtask_rules
            if int(subtask_idx) >= int(start_idx)
        ]
        if matching:
            return subtask_rules[max(matching)]
        return ref_spec

    def affordance_points(
        self,
        *,
        ref: str,
        subtask_idx: int,
        object_xyz: np.ndarray,
        spec: Optional[dict] = None,
    ) -> np.ndarray:
        """Return explicit target points, or object pose when no spec exists."""
        if spec is None:
            spec = self.affordance_spec(ref=ref, subtask_idx=int(subtask_idx))
        points = self._points_from_spec(spec) if spec is not None else None
        if points is not None:
            return points
        return np.asarray(object_xyz, dtype=np.float32).reshape(1, 3)

    def segmentation_box(
        self,
        *,
        ref: str,
        spec: Optional[dict],
        min_patch_area_fraction: float = 0.05,
        min_mask_pixels: int = 0,
    ) -> tuple[Optional[np.ndarray], float, Optional[np.ndarray]]:
        """Return bbox, visible pixel area, and filtered patch mask for a target."""
        _ = ref
        if spec is None:
            return None, 0.0, None

        geom_ids = self._geom_ids_from_spec(spec)
        if not geom_ids:
            return None, 0.0, None

        try:
            seg = self.env.env.sim.render(
                camera_name=self.camera_name,
                width=self.resolution,
                height=self.resolution,
                depth=False,
                segmentation=True,
            )
        except Exception:
            return None, 0.0, None

        geom_seg = np.asarray(seg[::-1, :, 1], dtype=np.int64)
        mask = np.isin(geom_seg, np.asarray(sorted(geom_ids), dtype=np.int64))
        mask_area = int(mask.sum())
        if mask_area <= 0:
            return None, 0.0, None
        if mask_area < max(0, int(min_mask_pixels)):
            return None, float(mask_area), None

        patch_mask = self._patch_mask_from_mask(
            mask,
            patch_size=self.patch_size,
            min_patch_area_fraction=min_patch_area_fraction,
        )
        if not np.any(patch_mask):
            return None, float(mask_area), patch_mask

        box = self._patch_square_box_from_patch_mask(
            patch_mask,
            image_height=mask.shape[0],
            image_width=mask.shape[1],
            patch_size=self.patch_size,
        )
        return box, float(mask_area), patch_mask

    def _task_rules(self) -> Optional[dict]:
        env_name = str(self.env_meta.get("env_name", "")).lower()
        env_name_norm = self._normalize_name(env_name)
        matches = [
            key
            for key in AFFORDANCE_REGISTRY
            if key in env_name or self._normalize_name(key) in env_name_norm
        ]
        if not matches:
            return None
        return AFFORDANCE_REGISTRY[
            max(matches, key=lambda key: len(self._normalize_name(key)))
        ]

    def _points_from_spec(self, spec: dict) -> Optional[np.ndarray]:
        points = []
        for geom_id in sorted(self._geom_ids_from_spec(spec)):
            points.append(self.env.env.sim.data.geom_xpos[int(geom_id)])
        if points:
            return np.asarray(points, dtype=np.float32).reshape(-1, 3)

        for body_name in self._body_names_from_spec(spec):
            body_xyz = self._body_xyz(body_name)
            if body_xyz is not None:
                points.append(body_xyz)
        if points:
            return np.asarray(points, dtype=np.float32).reshape(-1, 3)
        return None

    def _geom_ids_from_spec(self, spec: dict) -> set[int]:
        geom_ids: set[int] = set()
        geom_ids.update(self._geom_ids_from_names(self._geom_names_from_spec(spec)))

        for body_name in self._body_names_from_spec(spec):
            geom_ids.update(
                self._geom_ids_from_body(
                    body_name,
                    geom_types=spec.get("geom_types"),
                )
            )

        env_attr = spec.get("env_attr")
        if env_attr:
            obj = getattr(getattr(self.env, "env", None), str(env_attr), None)
            geom_ids.update(self._geom_ids_from_mujoco_object(obj))

        object_name = spec.get("object_name")
        if object_name:
            geom_ids.update(
                self._geom_ids_from_mujoco_object(
                    self._mujoco_object_by_name(str(object_name))
                )
            )
        return geom_ids

    @staticmethod
    def _geom_names_from_spec(spec: dict) -> tuple[str, ...]:
        names = []
        if "geom" in spec:
            names.append(spec["geom"])
        if "geoms" in spec:
            names.extend(spec["geoms"])
        return tuple(str(name) for name in names if name is not None)

    @staticmethod
    def _body_names_from_spec(spec: dict) -> tuple[str, ...]:
        names = []
        if "body" in spec:
            names.append(spec["body"])
        if "bodies" in spec:
            names.extend(spec["bodies"])
        return tuple(str(name) for name in names if name is not None)

    def _geom_ids_from_names(self, geom_names) -> set[int]:
        model = self.env.env.sim.model
        available = {model.geom_id2name(i): i for i in range(model.ngeom)}
        geom_ids = set()
        for name in geom_names:
            candidates = (
                str(name),
                f"{name}_visual",
                f"{name}_vis",
            )
            geom_ids.update(
                int(available[candidate])
                for candidate in candidates
                if candidate in available
            )
        return geom_ids

    def _geom_ids_from_body(self, body_name: str, *, geom_types=None) -> set[int]:
        model = self.env.env.sim.model
        try:
            body_id = int(model.body_name2id(str(body_name)))
        except Exception:
            return set()

        geom_bodyid = np.asarray(getattr(model, "geom_bodyid", []), dtype=np.int64)
        geom_ids = {int(i) for i in np.where(geom_bodyid == body_id)[0].tolist()}
        if not geom_types:
            return geom_ids

        allowed_types = {
            GEOM_TYPE_IDS[str(geom_type).lower()]
            for geom_type in geom_types
            if str(geom_type).lower() in GEOM_TYPE_IDS
        }
        if not allowed_types:
            return set()

        geom_type = np.asarray(getattr(model, "geom_type", []), dtype=np.int64)
        return {
            geom_id
            for geom_id in geom_ids
            if geom_id < geom_type.shape[0] and int(geom_type[geom_id]) in allowed_types
        }

    def _geom_ids_from_mujoco_object(self, obj) -> set[int]:
        if obj is None:
            return set()

        geom_names = []
        geom_names.extend(list(getattr(obj, "visual_geoms", []) or []))
        geom_names.extend(list(getattr(obj, "contact_geoms", []) or []))
        geom_ids = self._geom_ids_from_names(geom_names)
        if geom_ids:
            return geom_ids

        root_body = getattr(obj, "root_body", None)
        if root_body is not None:
            return self._geom_ids_from_body(root_body)
        return set()

    def _mujoco_object_by_name(self, name: str):
        env = getattr(self.env, "env", None)
        for attr_name in ("objects", "visual_objects", "nuts"):
            for obj in list(getattr(env, attr_name, []) or []):
                if str(getattr(obj, "name", "")) == str(name):
                    return obj
        return None

    def _body_xyz(self, body_name: str) -> Optional[np.ndarray]:
        sim = self.env.env.sim
        model = sim.model
        try:
            body_id = int(model.body_name2id(str(body_name)))
        except Exception:
            return None
        return np.asarray(sim.data.body_xpos[body_id], dtype=np.float32)

    @staticmethod
    def _patch_mask_from_mask(
        mask: np.ndarray,
        patch_size: int,
        min_patch_area_fraction: float,
    ) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        height, width = mask.shape
        patch_size = max(1, int(patch_size))
        mask_area = int(mask.sum())
        min_patch_pixels = max(
            1, int(np.ceil(float(mask_area) * float(min_patch_area_fraction)))
        )
        grid_h = max(1, int(np.ceil(height / patch_size)))
        grid_w = max(1, int(np.ceil(width / patch_size)))
        patch_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)

        for gy in range(grid_h):
            y1 = gy * patch_size
            y2 = min((gy + 1) * patch_size, height)
            if y1 >= y2:
                continue
            for gx in range(grid_w):
                x1 = gx * patch_size
                x2 = min((gx + 1) * patch_size, width)
                if x1 >= x2:
                    continue
                if int(mask[y1:y2, x1:x2].sum()) >= min_patch_pixels:
                    patch_mask[gy, gx] = 1
        return patch_mask

    @staticmethod
    def _patch_square_box_from_patch_mask(
        patch_mask: np.ndarray,
        *,
        image_height: int,
        image_width: int,
        patch_size: int,
    ) -> np.ndarray:
        patch_mask = np.asarray(patch_mask, dtype=bool)
        ys, xs = np.nonzero(patch_mask)
        if ys.size == 0:
            return np.full((4,), np.nan, dtype=np.float32)

        side_patches = max(1, int(max(patch_mask.shape)))
        patch_size = max(1, int(patch_size))
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
        box_side = max(x_max - x_min + 1, y_max - y_min + 1)

        cx = (x_min + x_max + 1) * 0.5
        cy = (y_min + y_max + 1) * 0.5
        gx1 = int(np.floor(cx - box_side * 0.5))
        gy1 = int(np.floor(cy - box_side * 0.5))
        gx1 = int(np.clip(gx1, 0, side_patches - box_side))
        gy1 = int(np.clip(gy1, 0, side_patches - box_side))
        gx2 = gx1 + box_side
        gy2 = gy1 + box_side

        return np.asarray(
            [
                float(gx1 * patch_size),
                float(gy1 * patch_size),
                float(min(gx2 * patch_size, image_width)),
                float(min(gy2 * patch_size, image_height)),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        return "".join(ch for ch in str(name).lower() if ch.isalnum())
