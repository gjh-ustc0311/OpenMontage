"""Strict, versioned configuration for TikTok replication preprocessing."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .storage import sha256_file, sha256_json


TOOL_IMPLEMENTATION_REVISION = "replication-preprocess-002"
# Keep the existing analysis/export revision stable: selection does not change cuts.
SELECTION_IMPLEMENTATION_REVISION = "representative-selection-001"
DEFAULT_PROFILE = "default-v1"
_PROFILE_RESOURCE = f"profiles/{DEFAULT_PROFILE}.yaml"
_DERIVED_KEYS = {
    "tool_fingerprint",
    "config_fingerprint",
    "config_fingerprints",
    "config_sources",
    "plan_fingerprint",
    "export_fingerprint",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def _decimal(value: str, field_name: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"{field_name} must be a decimal string") from exc
    if not parsed.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _ratio(value: str, field_name: str, *, allow_zero: bool = True) -> Decimal:
    parsed = _decimal(value, field_name)
    lower_ok = parsed >= 0 if allow_zero else parsed > 0
    if not lower_ok or parsed > 1:
        bracket = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{field_name} must be in {bracket}")
    return parsed


class ProfileConfig(_StrictModel):
    name: Literal["default-v1"] = DEFAULT_PROFILE
    min_duration_s: str
    max_duration_s: str
    max_exclusive: Literal[True] = True
    forced_split_target_max_s: str
    forced_split_search_start_s: str
    forced_split_search_end_s: str


class SceneDetectionConfig(_StrictModel):
    backend: Literal["pyav"] = "pyav"
    initial_threshold: int = Field(ge=1, le=255)
    threshold_step: int = Field(ge=1, le=255)
    minimum_threshold: int = Field(ge=1, le=255)
    min_scene_len_frames: int = Field(ge=1)
    separator_min_duration_s: str
    separator_context_s: str
    separator_solid_ratio: str


class QualityScoreWeights(_StrictModel):
    sharpness: str
    exposure: str
    stability: str
    proximity: str


class KeyframeConfig(_StrictModel):
    analysis_width_px: int = Field(ge=32)
    search_window_s: str
    search_window_ratio: str
    laplacian_min: int = Field(ge=0)
    luma_min: int = Field(ge=0, le=255)
    luma_max: int = Field(ge=0, le=255)
    black_pixel_luma_max: int = Field(ge=0, le=255)
    white_pixel_luma_min: int = Field(ge=0, le=255)
    black_ratio_max: str
    white_ratio_max: str
    stability_delta_max: str
    sharpness_normalizer_multiplier: str
    quality_score_weights: QualityScoreWeights


class ReviewConfig(_StrictModel):
    protocol_version: Literal["boundary-visual-review-v1"] = "boundary-visual-review-v1"
    max_agent_rounds: Literal[2] = 2
    context_offset_s: str
    expanded_context_offset_s: str
    background_mask_width_ratio: str
    background_mask_top_ratio: str
    background_mask_bottom_ratio: str
    keyframe_jpeg_quality: int = Field(ge=1, le=100)
    evidence_jpeg_quality: int = Field(ge=1, le=100)
    contact_sheet_jpeg_quality: int = Field(ge=1, le=100)


class RepresentativeSelectionConfig(_StrictModel):
    initial_candidates: int = Field(default=24, ge=4, le=72)
    page_size: int = Field(default=12, ge=1, le=12)
    max_observation_rounds: int = Field(default=3, ge=0, le=10)
    observations_per_round: int = Field(default=16, ge=1, le=72)
    max_candidates: int = Field(default=72, ge=4, le=256)
    preview_long_edge_px: int = Field(default=640, ge=160, le=1280)
    observation_radius_s: str = "0.25"
    laplacian_min: int = Field(default=80, ge=0)
    luma_min: int = Field(default=16, ge=0, le=255)
    luma_max: int = Field(default=239, ge=0, le=255)
    black_ratio_max: str = "0.95"
    white_ratio_max: str = "0.95"
    novelty_min: str = "0.025"

    @model_validator(mode="after")
    def check_limits(self) -> "RepresentativeSelectionConfig":
        if self.page_size > self.initial_candidates:
            raise ValueError("selection page_size must not exceed initial_candidates")
        if self.initial_candidates + self.max_observation_rounds * self.observations_per_round > self.max_candidates:
            raise ValueError("selection candidate budget is smaller than initial plus observation limits")
        if self.luma_min > self.luma_max:
            raise ValueError("selection luma_min must not exceed luma_max")
        if _decimal(self.observation_radius_s, "observation_radius_s") < 0:
            raise ValueError("observation_radius_s must be non-negative")
        for name in ("black_ratio_max", "white_ratio_max", "novelty_min"):
            _ratio(getattr(self, name), name)
        return self


class RegroupConfig(_StrictModel):
    max_atomic_segments: int = Field(ge=1, le=3)
    allow_merge_two_normal_segments: Literal[False] = False
    allow_cross_hard_boundary: Literal[False] = False
    allow_drop_under_1s: bool
    drop_threshold_s: str


class ExportConfig(_StrictModel):
    video_codec: Literal["libx264"] = "libx264"
    crf: int = Field(ge=0, le=51)
    preset: Literal[
        "ultrafast", "superfast", "veryfast", "faster", "fast",
        "medium", "slow", "slower", "veryslow",
    ]
    pixel_format: Literal["yuv420p"] = "yuv420p"
    audio_codec: Literal["aac"] = "aac"
    audio_bitrate: str


class ReplicationPreprocessConfig(_StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    config_revision: Literal["replication-preprocess-v2"] = "replication-preprocess-v2"
    profile: ProfileConfig
    scene_detection: SceneDetectionConfig
    keyframe: KeyframeConfig
    review: ReviewConfig
    regroup: RegroupConfig
    export: ExportConfig
    representative_selection: RepresentativeSelectionConfig = Field(
        default_factory=RepresentativeSelectionConfig
    )

    @model_validator(mode="after")
    def validate_cross_field_contract(self) -> "ReplicationPreprocessConfig":
        minimum = _decimal(self.profile.min_duration_s, "profile.min_duration_s")
        maximum = _decimal(self.profile.max_duration_s, "profile.max_duration_s")
        target = _decimal(
            self.profile.forced_split_target_max_s,
            "profile.forced_split_target_max_s",
        )
        search_start = _decimal(
            self.profile.forced_split_search_start_s,
            "profile.forced_split_search_start_s",
        )
        search_end = _decimal(
            self.profile.forced_split_search_end_s,
            "profile.forced_split_search_end_s",
        )
        if not (0 < minimum < maximum):
            raise ValueError("profile durations must satisfy 0 < min_duration_s < max_duration_s")
        if not (0 < search_start <= search_end <= target < maximum):
            raise ValueError(
                "forced split values must satisfy 0 < search_start <= search_end <= target < max"
            )
        if search_start > maximum - minimum:
            raise ValueError("forced_split_search_start_s leaves no legal trailing segment")

        detection = self.scene_detection
        if detection.minimum_threshold > detection.initial_threshold:
            raise ValueError("minimum_threshold cannot exceed initial_threshold")
        for field_name in ("separator_min_duration_s", "separator_context_s"):
            if _decimal(getattr(detection, field_name), f"scene_detection.{field_name}") <= 0:
                raise ValueError(f"scene_detection.{field_name} must be positive")
        _ratio(
            detection.separator_solid_ratio,
            "scene_detection.separator_solid_ratio",
            allow_zero=False,
        )

        keyframe = self.keyframe
        if keyframe.luma_min > keyframe.luma_max:
            raise ValueError("keyframe.luma_min cannot exceed keyframe.luma_max")
        if keyframe.black_pixel_luma_max >= keyframe.white_pixel_luma_min:
            raise ValueError("black pixel cutoff must be below white pixel cutoff")
        if _decimal(keyframe.search_window_s, "keyframe.search_window_s") <= 0:
            raise ValueError("keyframe.search_window_s must be positive")
        _ratio(keyframe.search_window_ratio, "keyframe.search_window_ratio", allow_zero=False)
        _ratio(keyframe.black_ratio_max, "keyframe.black_ratio_max")
        _ratio(keyframe.white_ratio_max, "keyframe.white_ratio_max")
        _ratio(keyframe.stability_delta_max, "keyframe.stability_delta_max", allow_zero=False)
        if _decimal(
            keyframe.sharpness_normalizer_multiplier,
            "keyframe.sharpness_normalizer_multiplier",
        ) <= 0:
            raise ValueError("keyframe.sharpness_normalizer_multiplier must be positive")
        weights = keyframe.quality_score_weights
        weight_values = [
            _decimal(getattr(weights, name), f"keyframe.quality_score_weights.{name}")
            for name in ("sharpness", "exposure", "stability", "proximity")
        ]
        if any(value < 0 for value in weight_values) or sum(weight_values) != Decimal("1"):
            raise ValueError("keyframe quality score weights must be non-negative and sum to 1")

        review = self.review
        initial_context = _decimal(review.context_offset_s, "review.context_offset_s")
        expanded_context = _decimal(
            review.expanded_context_offset_s,
            "review.expanded_context_offset_s",
        )
        if initial_context <= 0 or expanded_context < initial_context:
            raise ValueError("review context offsets are invalid")
        width = _ratio(
            review.background_mask_width_ratio,
            "review.background_mask_width_ratio",
            allow_zero=False,
        )
        top = _ratio(review.background_mask_top_ratio, "review.background_mask_top_ratio")
        bottom = _ratio(review.background_mask_bottom_ratio, "review.background_mask_bottom_ratio")
        if width <= 0 or top >= bottom:
            raise ValueError("review background mask ratios are invalid")

        drop_threshold = _decimal(self.regroup.drop_threshold_s, "regroup.drop_threshold_s")
        if drop_threshold <= 0 or drop_threshold > 1:
            raise ValueError("regroup.drop_threshold_s must be in (0, 1]")
        if not re.fullmatch(r"[1-9][0-9]*k", self.export.audio_bitrate):
            raise ValueError("export.audio_bitrate must use a positive integer k bitrate")
        return self


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_mapping(raw: str, *, suffix: str, source: str) -> dict[str, Any]:
    try:
        loaded = json.loads(raw) if suffix.lower() == ".json" else yaml.safe_load(raw)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid replication configuration syntax in {source}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Replication configuration must be an object: {source}")
    return loaded


def _profile_payload(profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if profile != DEFAULT_PROFILE:
        raise ValueError(f"Unknown replication profile: {profile}")
    resource = files("lib.replication_preprocess").joinpath(_PROFILE_RESOURCE)
    raw_bytes = resource.read_bytes()
    raw = raw_bytes.decode("utf-8")
    payload = _load_mapping(raw, suffix=".yaml", source=_PROFILE_RESOURCE)
    return payload, {
        "kind": "package_profile",
        "path": f"lib/replication_preprocess/{_PROFILE_RESOURCE}",
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def _fingerprints(effective: dict[str, Any]) -> dict[str, str]:
    analysis = sha256_json({
        "tool_implementation_revision": TOOL_IMPLEMENTATION_REVISION,
        "schema_version": effective["schema_version"],
        "config_revision": effective["config_revision"],
        "profile": effective["profile"],
        "scene_detection": effective["scene_detection"],
        "keyframe": effective["keyframe"],
    })
    review = sha256_json({
        "analysis_fingerprint": analysis,
        "review": effective["review"],
    })
    planning = sha256_json({
        "analysis_fingerprint": analysis,
        "regroup": effective["regroup"],
    })
    export = sha256_json({
        "tool_implementation_revision": TOOL_IMPLEMENTATION_REVISION,
        "export": effective["export"],
    })
    selection = sha256_json({
        "analysis_fingerprint": analysis,
        "implementation_revision": SELECTION_IMPLEMENTATION_REVISION,
        "representative_selection": effective["representative_selection"],
    })
    return {"analysis": analysis, "review": review, "planning": planning, "export": export, "selection": selection}


def load_config(path: str | None = None, *, profile: str = DEFAULT_PROFILE) -> dict[str, Any]:
    """Load the package profile, overlay one explicit project file, and validate."""

    base, base_source = _profile_payload(profile)
    sources = [base_source]
    payload = base
    if path:
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise ValueError(f"Replication config file not found: {config_path}")
        raw = config_path.read_text(encoding="utf-8")
        override = _load_mapping(raw, suffix=config_path.suffix, source=str(config_path))
        payload = _deep_merge(base, override)
        sources.append({
            "kind": "project_override",
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        })
    model = ReplicationPreprocessConfig.model_validate(payload)
    effective = model.model_dump(mode="python")
    fingerprints = _fingerprints(effective)
    result = deepcopy(effective)
    result["tool_fingerprint"] = sha256_json({
        "implementation_revision": TOOL_IMPLEMENTATION_REVISION,
    })
    result["config_fingerprint"] = sha256_json(effective)
    result["config_fingerprints"] = fingerprints
    result["config_sources"] = sources
    return result


def validate_config(config: dict[str, Any]) -> None:
    """Validate an effective or source config mapping without mutating it."""

    if not isinstance(config, dict):
        raise ValueError("Replication configuration must be an object")
    payload = {key: deepcopy(value) for key, value in config.items() if key not in _DERIVED_KEYS}
    if payload.get("scene_detection", {}).get("backend") != "pyav":
        raise ValueError("Replication preprocessing requires the PyAV SceneDetect backend")
    if payload.get("regroup", {}).get("allow_cross_hard_boundary") is not False:
        raise ValueError("Crossing hard boundaries is unsupported")
    ReplicationPreprocessConfig.model_validate(payload)


def compute_plan_fingerprint(config: dict[str, Any], review_digest: str) -> str:
    return sha256_json({
        "planning_config_fingerprint": config["config_fingerprints"]["planning"],
        "review_digest": review_digest,
    })


def compute_export_fingerprint(config: dict[str, Any], plan_fingerprint: str) -> str:
    return sha256_json({
        "plan_fingerprint": plan_fingerprint,
        "export_config_fingerprint": config["config_fingerprints"]["export"],
    })
