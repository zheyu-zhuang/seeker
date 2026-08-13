from typing import Union, Dict, Callable, Optional

import numpy as np
import torch
import torch.nn as nn


def dict_apply(
    x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


class DictOfTensorMixin(nn.Module):
    def __init__(self, params_dict=None):
        super().__init__()
        if params_dict is None:
            params_dict = nn.ParameterDict()
        self.params_dict = params_dict

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        def dfs_add(dest, keys, value: torch.Tensor):
            if len(keys) == 1:
                dest[keys[0]] = value
                return

            if keys[0] not in dest:
                dest[keys[0]] = nn.ParameterDict()
            dfs_add(dest[keys[0]], keys[1:], value)

        def load_dict(state_dict, prefix):
            out_dict = nn.ParameterDict()
            for key, value in state_dict.items():
                value: torch.Tensor
                if key.startswith(prefix):
                    param_keys = key[len(prefix) :].split(".")[1:]
                    dfs_add(out_dict, param_keys, value.clone())
            return out_dict

        self.params_dict = load_dict(state_dict, prefix + "params_dict")
        self.params_dict.requires_grad_(False)
        return


class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[Dict, torch.Tensor, np.ndarray],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] = _fit(
                    value,
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset,
                )
        else:
            self.params_dict["_default"] = _fit(
                data,
                last_n_dims=last_n_dims,
                dtype=dtype,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset,
            )

    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)

    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str, value: "SingleFieldLinearNormalizer"):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if "_default" not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict["_default"]
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and "_default" in self.params_dict:
            return self.params_dict["_default"]["input_stats"]

        result = dict()
        for key, value in self.params_dict.items():
            if key != "_default":
                result[key] = value["input_stats"]
        return result

    def get_output_stats(self, key="_default"):
        input_stats = self.get_input_stats()
        if "min" in input_stats:
            # no dict
            return dict_apply(input_stats, self.normalize)

        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key: value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[torch.Tensor, np.ndarray],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        self.params_dict = _fit(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
        )

    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj

    @classmethod
    def create_manual(
        cls,
        scale: Union[torch.Tensor, np.ndarray],
        offset: Union[torch.Tensor, np.ndarray],
        input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    ):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x

        # check
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype

        params_dict = nn.ParameterDict(
            {
                "scale": to_tensor(scale),
                "offset": to_tensor(offset),
                "input_stats": nn.ParameterDict(
                    dict_apply(input_stats_dict, to_tensor)
                ),
            }
        )
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            "min": torch.tensor([-1], dtype=dtype),
            "max": torch.tensor([1], dtype=dtype),
            "mean": torch.tensor([0], dtype=dtype),
            "std": torch.tensor([1], dtype=dtype),
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict["input_stats"]

    def get_output_stats(self):
        return dict_apply(self.params_dict["input_stats"], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)


def get_range_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    """Create an affine normalizer mapping stats range to [output_min, output_max]."""
    input_max = stat["max"]
    input_min = stat["min"]
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_image_range_normalizer():
    """Create identity image normalizer for inputs already in [0, 1]."""
    scale = np.array([1], dtype=np.float32)
    offset = np.array([0], dtype=np.float32)
    stat = {
        "min": np.array([0], dtype=np.float32),
        "max": np.array([1], dtype=np.float32),
        "mean": np.array([0.5], dtype=np.float32),
        "std": np.array([np.sqrt(1 / 12)], dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def get_identity_normalizer_from_stat(stat):
    """Create identity normalizer with provided stats payload."""
    scale = np.ones_like(stat["min"])
    offset = np.zeros_like(stat["min"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def array_to_stats(arr: np.ndarray):
    """Compute min/max/mean/std over axis 0."""
    return {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
    }


def _fit(
    data: Union[torch.Tensor, np.ndarray],
    last_n_dims=1,
    dtype=torch.float32,
    mode="limits",
    output_max=1.0,
    output_min=-1.0,
    range_eps=1e-4,
    fit_offset=True,
):
    assert mode in ["limits", "gaussian"]
    assert last_n_dims >= 0
    assert output_max > output_min

    # convert data to torch and type
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    elif not isinstance(data, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor or np.ndarray, got {type(data).__name__}"
        )
    if dtype is not None:
        data = data.type(dtype)

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1, dim)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    # compute scale and offset
    if mode == "limits":
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == "gaussian":
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = -input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)

    # save
    this_params = nn.ParameterDict(
        {
            "scale": scale,
            "offset": offset,
            "input_stats": nn.ParameterDict(
                {
                    "min": input_min,
                    "max": input_max,
                    "mean": input_mean,
                    "std": input_std,
                }
            ),
        }
    )
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params, forward=True):
    assert "scale" in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params["scale"]
    offset = params["offset"]
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x


class MultiRobotLinearNormalizer(nn.Module):
    """
    Holds one LinearNormalizer per robot id.
    Routes normalize/unnormalize by robot_ids.
    """

    def __init__(self, normalizers: Optional[Dict[int, "LinearNormalizer"]] = None):
        super().__init__()
        self.by_robot = nn.ModuleDict()  # str(robot_id) -> LinearNormalizer
        if normalizers is not None:
            for rid, norm in normalizers.items():
                self.add(int(rid), norm)

    def add(self, robot_id: int, normalizer: "LinearNormalizer"):
        n = LinearNormalizer()
        n.load_state_dict(normalizer.state_dict(), strict=True)
        self.by_robot[str(int(robot_id))] = n

    def get(self, robot_id: Union[int, torch.Tensor]) -> "LinearNormalizer":
        rid = int(robot_id.item()) if torch.is_tensor(robot_id) else int(robot_id)
        return self.by_robot[str(rid)]

    def _require_robot(self, rid: int) -> "LinearNormalizer":
        k = str(rid)
        if k not in self.by_robot:
            raise KeyError(
                f"Missing normalizer for robot {rid}. Have {sorted(self.by_robot.keys())}"
            )
        return self.by_robot[k]

    def _require_field(self, norm: "LinearNormalizer", rid: int, field: str):
        if field not in norm.params_dict:
            raise KeyError(
                f"Robot {rid} normalizer missing params_dict['{field}']. "
                f"Available: {sorted(list(norm.params_dict.keys()))}"
            )

    @staticmethod
    def _flat_ids(robot_ids: torch.Tensor) -> torch.Tensor:
        return robot_ids.long().view(-1)

    @torch.no_grad()
    def _apply_field(
        self, x: torch.Tensor, robot_ids: torch.Tensor, field: str, forward: bool
    ) -> torch.Tensor:
        robot_ids = self._flat_ids(robot_ids)
        if x.shape[0] != robot_ids.shape[0]:
            raise ValueError(
                f"x.shape[0]={x.shape[0]} must match len(robot_ids)={robot_ids.shape[0]}"
            )

        out = torch.empty_like(x)
        for rid in torch.unique(robot_ids).tolist():
            rid = int(rid)
            idx = (robot_ids == rid).nonzero(as_tuple=False).squeeze(1)
            norm = self._require_robot(rid)
            self._require_field(norm, rid, field)

            fn = norm[field]  # SingleFieldLinearNormalizer
            chunk = x.index_select(0, idx)
            y = fn.normalize(chunk) if forward else fn.unnormalize(chunk)
            out.index_copy_(0, idx, y)
        return out

    @torch.no_grad()
    def normalize_action(
        self, action: torch.Tensor, robot_ids: torch.Tensor
    ) -> torch.Tensor:
        return self._apply_field(action, robot_ids, field="action", forward=True)

    @torch.no_grad()
    def unnormalize_action(
        self, action: torch.Tensor, robot_ids: torch.Tensor
    ) -> torch.Tensor:
        return self._apply_field(action, robot_ids, field="action", forward=False)

    @torch.no_grad()
    def _route(self, x, robot_ids: torch.Tensor, forward: bool):
        robot_ids = self._flat_ids(robot_ids)

        if isinstance(x, dict):
            out = {k: torch.empty_like(v) for k, v in x.items()}
        else:
            out = torch.empty_like(x)

        for rid in torch.unique(robot_ids).tolist():
            rid = int(rid)
            idx = (robot_ids == rid).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue

            norm = self._require_robot(rid)

            if isinstance(x, dict):
                x_r = {k: v.index_select(0, idx) for k, v in x.items()}
                y_r = norm.normalize(x_r) if forward else norm.unnormalize(x_r)
                for k in out.keys():
                    out[k].index_copy_(0, idx, y_r[k])
            else:
                x_r = x.index_select(0, idx)
                y_r = norm.normalize(x_r) if forward else norm.unnormalize(x_r)
                out.index_copy_(0, idx, y_r)

        return out

    @torch.no_grad()
    def normalize(self, x, robot_ids: torch.Tensor):
        return self._route(x, robot_ids, forward=True)

    @torch.no_grad()
    def unnormalize(self, x, robot_ids: torch.Tensor):
        return self._route(x, robot_ids, forward=False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Materialize by_robot[rid] modules that appear in the checkpoint.
        base = prefix + "by_robot."
        robot_keys = set()
        for k in state_dict.keys():
            if k.startswith(base):
                rest = k[len(base) :]  # "<rid>...."
                rid = rest.split(".", 1)[0]  # "<rid>"
                robot_keys.add(rid)

        for rk in robot_keys:
            if rk not in self.by_robot:
                self.by_robot[rk] = LinearNormalizer()

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __repr__(self) -> str:
        rids = sorted(self.by_robot.keys(), key=lambda x: int(x) if x.isdigit() else x)
        lines = [f"{self.__class__.__name__}("]
        lines.append(f"  robots={rids}")

        for rid in rids:
            norm = self.by_robot[rid]
            fields = (
                sorted(list(norm.params_dict.keys()))
                if hasattr(norm, "params_dict")
                else []
            )
            lines.append(f"  rid={rid}: fields={fields}")

        lines.append(")")
        return "\n".join(lines)
