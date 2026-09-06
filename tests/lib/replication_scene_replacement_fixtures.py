from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from pathlib import Path

from PIL import Image

from lib.replication_preprocess.config import (
    compute_export_fingerprint,
    compute_plan_fingerprint,
    load_config,
)
from lib.replication_preprocess.delivery import build_manifest
from lib.replication_preprocess.media_assets import media_fingerprint, media_signature
from lib.replication_preprocess.models import time_point
from lib.replication_scene_replacement.storage import atomic_write_json, sha256_file


def _write_json(project: Path, relative: str, value: dict) -> dict[str, str]:
    path = project / relative
    atomic_write_json(path, value)
    return {"path": relative, "sha256": sha256_file(path)}


def _phase1(
    project: Path,
    source: Path,
    revision: str = "r0001",
    *,
    purpose: str = "Main functional view",
    config_digit: str = "1",
    supplementary: bool = False,
    segment_id: str = "segment_1",
) -> Path:
    """Write the smallest immutable, validated Phase 1 package accepted by the importer."""
    time_base = Fraction(1, 15360)
    segment_start = time_point(61440, time_base)
    segment_end = time_point(76800, time_base)
    source_time = time_point(66560, time_base)  # canonical projection is 4.333333333
    offset = time_point(5120, time_base)

    image_path = project / "assets/images/replication/delivery/shared/A01.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        Image.new("RGB", (8, 8), (180, 120, 40)).save(image_path, format="PNG")
    image_ref = {
        "path": image_path.relative_to(project).as_posix(),
        "sha256": sha256_file(image_path),
    }
    clip_path = project / "assets/video/replication/delivery/shared/S01.mp4"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    if not clip_path.exists():
        clip_path.write_bytes(b"immutable exported clip")
    clip_ref = {
        "path": clip_path.relative_to(project).as_posix(),
        "sha256": sha256_file(clip_path),
    }
    selection_ref = _write_json(
        project,
        f"artifacts/replication/revisions/{revision}/keyframe_selection.json",
        {"schema_version": "1.0", "anchor_status": "ready", "segments": []},
    )
    anchor = {
        "anchor_id": "anchor_1",
        "atomic_segment_id": segment_id,
        "role": "primary_representative",
        "purpose": purpose,
        "source_time": deepcopy(source_time),
        "atomic_segment_offset": deepcopy(offset),
        "generation_clip_offset": deepcopy(offset),
        "path": image_ref["path"],
        "sha256": image_ref["sha256"],
        "format": "png",
        "width": 8,
        "height": 8,
        "orientation": "display_normalized",
        "pixel_transform": "decoder_rgb24_to_png",
        "color_conversion": "bt709_to_srgb",
    }
    keyframes = [anchor]
    if supplementary:
        supplementary_path = project / "assets/images/replication/delivery/shared/A02.png"
        Image.new("RGB", (8, 8), (170, 110, 35)).save(supplementary_path, format="PNG")
        supplementary_time = time_point(67072, time_base)
        supplementary_offset = time_point(5632, time_base)
        keyframes.append({
            **deepcopy(anchor),
            "anchor_id": "anchor_2",
            "role": "supplementary_anchor",
            "purpose": "Difficult occlusion view",
            "source_time": supplementary_time,
            "atomic_segment_offset": supplementary_offset,
            "generation_clip_offset": supplementary_offset,
            "path": supplementary_path.relative_to(project).as_posix(),
            "sha256": sha256_file(supplementary_path),
        })
    config = load_config()
    config["config_fingerprint"] = config_digit * 64
    plan_fingerprint = compute_plan_fingerprint(config, "none")
    export_fingerprint = compute_export_fingerprint(config, plan_fingerprint)
    state = {
        "schema_version": "3.0",
        "project_id": project.name,
        "plan_revision": revision,
        "plan_status": "ready",
        "anchor_status": "ready",
        "source": {
            "path": str(source),
            "file_sha256": sha256_file(source),
            "start": time_point(0, time_base),
            "end": time_point(153600, time_base),
            "time_base": {"num": 1, "den": 15360},
            "video_stream_index": 0,
            "audio_stream_index": None,
            "audio_present": False,
            "width": 8,
            "height": 8,
            "nominal_frame_duration_s": "0.033333333",
            "rotation": 0,
            "color": {
                "pix_fmt": "yuv420p",
                "color_range": "tv",
                "color_space": "bt709",
                "color_transfer": "bt709",
                "color_primaries": "bt709",
            },
        },
        "config": config,
        "review_digest": "none",
        "plan_fingerprint": plan_fingerprint,
        "runtime_versions": {"ffmpeg": "fixture-ffmpeg"},
        "atomic_segments": [{
            "segment_id": segment_id,
            "start": deepcopy(segment_start),
            "end": deepcopy(segment_end),
        }],
        "generation_clips": [{
            "clip_id": "clip_1",
            "start": deepcopy(segment_start),
            "end": deepcopy(segment_end),
            "duration_s": "1",
            "atomic_segment_ids": [segment_id],
            "keyframes": keyframes,
        }],
        "dropped_intervals": [],
        "manifest_index": {"keyframe_selection.json": selection_ref},
    }
    state_ref = _write_json(
        project,
        f"artifacts/replication/revisions/{revision}/plan_state.json",
        state,
    )
    planned_clip = state["generation_clips"][0]
    report = {
        "schema_version": "2.0",
        "plan_revision": revision,
        "plan_fingerprint": plan_fingerprint,
        "export_fingerprint": export_fingerprint,
        "export_config_fingerprint": config["config_fingerprints"]["export"],
        "config_sources": deepcopy(config["config_sources"]),
        "status": "validated",
        "encoding": deepcopy(config["export"]),
        "reuse_issues": [],
        "clips": [{
            "media_fingerprint": media_fingerprint(state, planned_clip, config["export"]),
            "media_signature": media_signature(state, planned_clip, config["export"]),
            "anchor_ids": [item["anchor_id"] for item in keyframes],
            "reused": False,
            "clip_id": "clip_1",
            "path": clip_ref["path"],
            "sha256": clip_ref["sha256"],
            "duration_seconds": "1",
            "video_streams": 1,
            "audio_streams": 0,
            "full_decode": "passed",
            "status": "validated",
        }],
    }
    report_ref = _write_json(
        project,
        f"artifacts/replication/exports/{revision}/export_report.json",
        report,
    )
    export_ref = {
        "status": "validated",
        "export_fingerprint": export_fingerprint,
        "export_config_fingerprint": config["config_fingerprints"]["export"],
        "config_sources": deepcopy(config["config_sources"]),
        "report_path": report_ref["path"],
        "report_sha256": report_ref["sha256"],
        "plan_state_ref": state_ref,
    }
    selected_index = {
        "manifest_index": {"plan_state.json": state_ref},
        "export": export_ref,
    }
    delivery = build_manifest(state, selected_index, report)
    for delivered_clip in delivery["clips"]:
        delivered_path = project / delivered_clip["path"]
        delivered_path.parent.mkdir(parents=True, exist_ok=True)
        delivered_path.write_bytes(clip_path.read_bytes())
        for delivered_frame in delivered_clip["keyframes"]:
            source_image = project / delivered_frame["source_path"]
            delivered_image = project / delivered_frame["path"]
            delivered_image.parent.mkdir(parents=True, exist_ok=True)
            delivered_image.write_bytes(source_image.read_bytes())
    delivery_variant = delivery["delivery_fingerprint"][:16]
    delivery_ref = _write_json(
        project,
        f"artifacts/replication/delivery/{revision}/{delivery_variant}/manifest.json",
        delivery,
    )
    index = {
        "schema_version": "3.0",
        "tool": "replication_preprocess",
        "project_id": project.name,
        "plan_revision": revision,
        "plan_status": "ready",
        "anchor_status": "ready",
        "manifest_index": {"plan_state.json": state_ref, "keyframe_selection.json": selection_ref},
        "export": export_ref,
        "delivery": {
            "delivery_fingerprint": delivery["delivery_fingerprint"],
            "manifest_path": delivery_ref["path"],
            "manifest_sha256": delivery_ref["sha256"],
        },
        "delivery_history": None,
        "config_fingerprint": config_digit * 64,
        "config_fingerprints": deepcopy(config["config_fingerprints"]),
    }
    return Path(_write_json(project, "artifacts/replication/index.json", index)["path"])

