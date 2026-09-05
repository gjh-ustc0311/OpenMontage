"""Configuration and exact-time helpers for replication preprocessing."""

from __future__ import annotations

import json
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

from .storage import sha256_json


TOOL_IMPLEMENTATION_REVISION = "replication-preprocess-001"


DEFAULT_CONFIG: dict[str, Any] = {
    "config_revision": "replication-preprocess-v1",
    "profile": {
        "name": "default-v1",
        "min_duration_s": "3.2",
        "max_duration_s": "15.0",
        "max_exclusive": True,
        "forced_split_target_max_s": "14.8",
    },
    "scene_detection": {
        "backend": "pyav",
        "initial_threshold": 50,
        "threshold_step": 5,
        "minimum_threshold": 20,
        "min_scene_len_frames": 12,
        "separator_min_duration_s": "0.08",
    },
    "keyframe": {
        "analysis_width_px": 320,
        "search_window_s": "1.0",
        "search_window_ratio": "0.30",
        "laplacian_min": 80,
        "luma_min": 16,
        "luma_max": 239,
        "black_ratio_max": "0.95",
        "white_ratio_max": "0.95",
        "stability_delta_max": "0.12",
    },
    "review": {
        "protocol_version": "boundary-visual-review-v1",
        "max_agent_rounds": 2,
        "context_offset_s": "0.5",
        "expanded_context_offset_s": "1.5",
    },
    "regroup": {
        "max_atomic_segments": 3,
        "allow_merge_two_normal_segments": False,
        "allow_cross_hard_boundary": False,
        "allow_drop_under_1s": False,
        "drop_threshold_s": "1.0",
    },
    "export": {
        "video_codec": "libx264",
        "crf": 18,
        "preset": "medium",
        "pixel_format": "yuv420p",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | None = None) -> dict[str, Any]:
    if not path:
        config = deepcopy(DEFAULT_CONFIG)
    else:
        config_path = Path(path)
        raw = config_path.read_text(encoding="utf-8")
        loaded = json.loads(raw) if config_path.suffix.lower() == ".json" else yaml.safe_load(raw)
        if not isinstance(loaded, dict):
            raise ValueError("Replication config must be a JSON/YAML object")
        config = _deep_merge(DEFAULT_CONFIG, loaded)
    validate_config(config)
    config["tool_fingerprint"] = sha256_json({
        "implementation_revision": TOOL_IMPLEMENTATION_REVISION,
    })
    config["config_fingerprint"] = sha256_json(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    profile = config["profile"]
    minimum = Decimal(str(profile["min_duration_s"]))
    maximum = Decimal(str(profile["max_duration_s"]))
    target = Decimal(str(profile["forced_split_target_max_s"]))
    if (
        minimum <= 0
        or maximum <= minimum
        or target < minimum
        or target >= maximum
        or profile.get("max_exclusive") is not True
    ):
        raise ValueError("Invalid generation duration profile")
    detection = config["scene_detection"]
    if detection.get("backend") != "pyav":
        raise ValueError("Replication preprocessing requires the PyAV SceneDetect backend")
    initial = int(detection["initial_threshold"])
    floor = int(detection["minimum_threshold"])
    step = int(detection["threshold_step"])
    if not (0 < floor <= initial) or step <= 0:
        raise ValueError("Invalid SceneDetect threshold ladder")
    if int(detection["min_scene_len_frames"]) < 1:
        raise ValueError("min_scene_len_frames must be positive")
    if Decimal(str(detection["separator_min_duration_s"])) <= 0:
        raise ValueError("separator_min_duration_s must be positive")
    keyframe = config["keyframe"]
    if int(keyframe["analysis_width_px"]) < 32:
        raise ValueError("analysis_width_px must be at least 32")
    if not Decimal("0") < Decimal(str(keyframe["search_window_ratio"])) <= Decimal("1"):
        raise ValueError("search_window_ratio must be in (0, 1]")
    review = config["review"]
    if int(review["max_agent_rounds"]) != 2:
        raise ValueError("max_agent_rounds must be 2 for the v1 review protocol")
    if (
        Decimal(str(review["context_offset_s"])) <= 0
        or Decimal(str(review["expanded_context_offset_s"]))
        < Decimal(str(review["context_offset_s"]))
    ):
        raise ValueError("Invalid review context offsets")
    regroup = config["regroup"]
    if not 1 <= int(regroup["max_atomic_segments"]) <= 3:
        raise ValueError("max_atomic_segments must be between 1 and 3")
    if regroup.get("allow_merge_two_normal_segments") is not False:
        raise ValueError("Merging two normal segments is unsupported")
    if regroup.get("allow_cross_hard_boundary") is not False:
        raise ValueError("Crossing hard boundaries is unsupported")
    drop_threshold = Decimal(str(regroup["drop_threshold_s"]))
    if drop_threshold <= 0 or drop_threshold > 1:
        raise ValueError("drop_threshold_s must be in (0, 1]")


def fraction_from_time_base(value: dict[str, Any] | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(int(value["num"]), int(value["den"]))


def time_base_json(value: Fraction) -> dict[str, int]:
    return {"num": value.numerator, "den": value.denominator}


def seconds_fraction(pts: int, time_base: dict[str, Any] | Fraction) -> Fraction:
    return int(pts) * fraction_from_time_base(time_base)


def decimal_string(value: Fraction | Decimal, places: int = 9) -> str:
    if isinstance(value, Fraction):
        value = Decimal(value.numerator) / Decimal(value.denominator)
    quantized = value.quantize(Decimal(1).scaleb(-places))
    text = format(quantized, "f").rstrip("0").rstrip(".")
    return text or "0"


def time_point(pts: int, time_base: Fraction) -> dict[str, Any]:
    return {
        "pts": int(pts),
        "time_base": time_base_json(time_base),
        "seconds": decimal_string(seconds_fraction(pts, time_base)),
    }


def interval_duration(start_pts: int, end_pts: int, time_base: Fraction) -> Fraction:
    return (int(end_pts) - int(start_pts)) * time_base


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    return f"{prefix}_{sha256_json(list(parts))[:length]}"
