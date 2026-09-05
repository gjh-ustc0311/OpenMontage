"""PTS-backed media analysis and visual evidence generation.

Heavy media dependencies are imported inside call sites so registry discovery
stays lightweight when the optional replication feature is not installed.
"""

from __future__ import annotations

import bisect
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .models import (
    TOOL_IMPLEMENTATION_REVISION,
    decimal_string,
    interval_duration,
    stable_id,
    time_base_json,
    time_point,
)
from .storage import sha256_file


class MediaAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameRecord:
    ordinal: int
    pts: int


def collect_runtime_versions() -> dict[str, str]:
    """Return the media runtime versions that make a plan reproducible."""

    try:
        import av
        import cv2
        import scenedetect
    except ImportError as exc:
        raise MediaAnalysisError(
            "PyAV, OpenCV-headless, and PySceneDetect are required"
        ) from exc
    try:
        completed = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise MediaAnalysisError("ffmpeg is required") from exc
    if completed.returncode != 0 or not completed.stdout.strip():
        raise MediaAnalysisError("ffmpeg is required")
    return {
        "tool_implementation": TOOL_IMPLEMENTATION_REVISION,
        "pyav": str(av.__version__),
        "pyscenedetect": str(scenedetect.__version__),
        "opencv": str(cv2.__version__),
        "ffmpeg": completed.stdout.splitlines()[0].strip(),
    }


def _rescale_pts(pts: int, source_base: Fraction, target_base: Fraction) -> int:
    exact = Fraction(int(pts)) * source_base / target_base
    if exact.denominator != 1:
        raise MediaAnalysisError("Frame PTS cannot be represented in the video stream time base")
    return exact.numerator


def _run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise MediaAnalysisError(completed.stderr.strip() or f"Command failed: {command[0]}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MediaAnalysisError(f"Invalid JSON returned by {command[0]}") from exc


def probe_source(path: Path) -> tuple[dict[str, Any], list[FrameRecord], Fraction]:
    """Probe streams and build a presentation-order frame PTS ledger."""

    if not path.is_file():
        raise MediaAnalysisError(f"Source video not found: {path}")
    if not shutil.which("ffprobe"):
        raise MediaAnalysisError("ffprobe is required")
    probe = _run_json([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ])
    video_streams = [item for item in probe.get("streams", []) if item.get("codec_type") == "video"]
    audio_streams = [item for item in probe.get("streams", []) if item.get("codec_type") == "audio"]
    if len(video_streams) != 1:
        raise MediaAnalysisError("Exactly one video stream is required")
    stream_probe = video_streams[0]

    try:
        import av
    except ImportError as exc:
        raise MediaAnalysisError("PyAV is required for PTS-backed analysis") from exc
    ledger: list[FrameRecord] = []
    with av.open(str(path)) as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise MediaAnalysisError("Exactly one decodable video stream is required")
        stream = streams[0]
        if stream.time_base is None:
            raise MediaAnalysisError("Video stream has no time base")
        time_base = Fraction(stream.time_base)
        for ordinal, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise MediaAnalysisError("Decoded frame is missing PTS")
            frame_base = Fraction(frame.time_base or time_base)
            pts = _rescale_pts(frame.pts, frame_base, time_base)
            if ledger and pts <= ledger[-1].pts:
                raise MediaAnalysisError("Video frame PTS values are not strictly increasing")
            ledger.append(FrameRecord(ordinal=ordinal, pts=pts))
        average_rate = Fraction(stream.average_rate) if stream.average_rate else None
    if not ledger:
        raise MediaAnalysisError("Video contains no decodable frames")

    deltas = [right.pts - left.pts for left, right in zip(ledger, ledger[1:])]
    if deltas:
        final_delta = max(1, int(median(deltas)))
    elif average_rate:
        duration_pts = Fraction(1, 1) / average_rate / time_base
        final_delta = max(1, math.ceil(duration_pts))
    else:
        final_delta = 1
    start_pts = ledger[0].pts
    end_pts = ledger[-1].pts + final_delta
    rotation = 0
    tags = stream_probe.get("tags") or {}
    if str(tags.get("rotate", "")).lstrip("-").isdigit():
        rotation = int(tags["rotate"]) % 360
    for side_data in stream_probe.get("side_data_list") or []:
        if side_data.get("rotation") is not None:
            rotation = int(side_data["rotation"]) % 360
    audio_probe = next(
        (
            item for item in audio_streams
            if int((item.get("disposition") or {}).get("default") or 0) == 1
        ),
        audio_streams[0] if audio_streams else None,
    )
    source = {
        "schema_version": "1.0",
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "video_stream_index": int(stream_probe.get("index", 0)),
        "audio_stream_index": int(audio_probe["index"]) if audio_probe else None,
        "audio_present": audio_probe is not None,
        "audio_codec": audio_probe.get("codec_name") if audio_probe else None,
        "audio_channels": int(audio_probe.get("channels") or 0) if audio_probe else None,
        "audio_sample_rate": int(audio_probe.get("sample_rate") or 0) if audio_probe else None,
        "codec": stream_probe.get("codec_name"),
        "width": int(stream_probe.get("width") or 0),
        "height": int(stream_probe.get("height") or 0),
        "rotation": rotation,
        "average_frame_rate": stream_probe.get("avg_frame_rate"),
        "time_base": time_base_json(time_base),
        "start": time_point(start_pts, time_base),
        "end": time_point(end_pts, time_base),
        "duration_s": decimal_string(interval_duration(start_pts, end_pts, time_base)),
        "nominal_frame_duration_s": decimal_string(Fraction(final_delta) * time_base),
        "frame_count": len(ledger),
        "vfr": len(set(deltas)) > 1 if deltas else False,
    }
    return source, ledger, time_base


def detect_scene_candidates(
    path: Path,
    ledger: list[FrameRecord],
    time_base: Fraction,
    minimum_threshold: float,
) -> list[dict[str, Any]]:
    """Run ContentDetector once at the floor and retain its component metrics."""

    try:
        from scenedetect import SceneManager, StatsManager
        from scenedetect.backends.pyav import VideoStreamAv
        from scenedetect.detectors import ContentDetector
    except ImportError as exc:
        raise MediaAnalysisError("PySceneDetect with the PyAV backend is required") from exc
    video = VideoStreamAv(str(path))
    stats = StatsManager(video.base_timecode)
    manager = SceneManager(stats_manager=stats)
    detector = ContentDetector(threshold=float(minimum_threshold), min_scene_len=1)
    manager.add_detector(detector)
    manager.detect_scenes(video)
    pts_to_ordinal = {record.pts: record.ordinal for record in ledger}
    candidates: list[dict[str, Any]] = []
    for cut in manager.get_cut_list(show_warning=False):
        if cut.pts is None or cut.time_base is None:
            raise MediaAnalysisError("SceneDetect returned a cut without PTS")
        pts = _rescale_pts(cut.pts, Fraction(cut.time_base), time_base)
        if pts not in pts_to_ordinal:
            raise MediaAnalysisError(f"SceneDetect cut PTS {pts} is absent from the PyAV ledger")
        values = stats.get_metrics(cut, detector.METRIC_KEYS)
        metrics = dict(zip(detector.METRIC_KEYS, values))
        candidates.append({
            "pts": pts,
            "ordinal": pts_to_ordinal[pts],
            "content_val": float(metrics.get("content_val") or 0.0),
            "delta_hue": float(metrics.get("delta_hue") or 0.0),
            "delta_sat": float(metrics.get("delta_sat") or 0.0),
            "delta_lum": float(metrics.get("delta_lum") or 0.0),
            "delta_edges": float(metrics.get("delta_edges") or 0.0),
        })
    return sorted(candidates, key=lambda item: (item["pts"], -item["content_val"]))


def analyze_frame_quality(
    path: Path,
    time_base: Fraction,
    analysis_width: int,
    config: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Calculate small, deterministic quality metrics for every decoded frame."""

    try:
        import av
        import cv2
        import numpy as np
    except ImportError as exc:
        raise MediaAnalysisError("PyAV, NumPy, and OpenCV-headless are required") from exc
    result: dict[int, dict[str, Any]] = {}
    keyframe = config["keyframe"]
    black_cutoff = int(keyframe["black_pixel_luma_max"])
    white_cutoff = int(keyframe["white_pixel_luma_min"])
    previous = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream_base = Fraction(stream.time_base)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = _rescale_pts(frame.pts, Fraction(frame.time_base or stream_base), time_base)
            rgb = frame.to_ndarray(format="rgb24")
            height, width = rgb.shape[:2]
            target_height = max(1, round(height * analysis_width / max(1, width)))
            rgb = cv2.resize(rgb, (analysis_width, target_height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            stability = 0.0
            if previous is not None and previous.shape == gray.shape:
                stability = float(np.mean(np.abs(gray.astype(np.float32) - previous)) / 255.0)
            result[pts] = {
                "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "luma_mean": float(gray.mean()),
                "black_ratio": float(np.mean(gray <= black_cutoff)),
                "white_ratio": float(np.mean(gray >= white_cutoff)),
                "stability_delta": stability,
            }
            previous = gray
    return result


def _quality_passes(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    keyframe = config["keyframe"]
    return (
        metrics["sharpness"] >= float(keyframe["laplacian_min"])
        and float(keyframe["luma_min"]) <= metrics["luma_mean"] <= float(keyframe["luma_max"])
        and metrics["black_ratio"] < float(keyframe["black_ratio_max"])
        and metrics["white_ratio"] < float(keyframe["white_ratio_max"])
        and metrics["stability_delta"] <= float(keyframe["stability_delta_max"])
    )


def _quality_score(metrics: dict[str, Any], config: dict[str, Any], proximity: float) -> float:
    keyframe = config["keyframe"]
    sharpness_multiplier = float(keyframe["sharpness_normalizer_multiplier"])
    sharp = min(
        metrics["sharpness"]
        / max(float(keyframe["laplacian_min"]) * sharpness_multiplier, 1.0),
        1.0,
    )
    exposure = 1.0 - abs(metrics["luma_mean"] - 127.5) / 127.5
    stability = 1.0 - min(
        metrics["stability_delta"] / max(float(keyframe["stability_delta_max"]), 1e-9), 1.0
    )
    weights = keyframe["quality_score_weights"]
    return (
        float(weights["sharpness"]) * sharp
        + float(weights["exposure"]) * exposure
        + float(weights["stability"]) * stability
        + float(weights["proximity"]) * proximity
    )


def _threshold_ladder(config: dict[str, Any]) -> list[int]:
    detection = config["scene_detection"]
    initial = int(detection["initial_threshold"])
    floor = int(detection["minimum_threshold"])
    step = int(detection["threshold_step"])
    values = list(range(initial, floor - 1, -step))
    if values[-1] != floor:
        values.append(floor)
    return values


def _dedupe_close_candidates(candidates: list[dict[str, Any]], min_frames: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["ordinal"]):
        if not selected or candidate["ordinal"] - selected[-1]["ordinal"] >= min_frames:
            selected.append(candidate)
        elif candidate["content_val"] > selected[-1]["content_val"]:
            selected[-1] = candidate
    return selected


def select_boundaries(
    *,
    source: dict[str, Any],
    ledger: list[FrameRecord],
    candidates: list[dict[str, Any]],
    qualities: dict[int, dict[str, Any]],
    time_base: Fraction,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the threshold ladder and deterministic forced-split policy."""

    start_pts = source["start"]["pts"]
    end_pts = source["end"]["pts"]
    profile = config["profile"]
    min_duration = Fraction(str(profile["min_duration_s"]))
    max_duration = Fraction(str(profile["max_duration_s"]))
    min_frames = int(config["scene_detection"]["min_scene_len_frames"])
    initial_threshold = _threshold_ladder(config)[0]
    natural = _dedupe_close_candidates(
        [
            item for item in candidates
            if item["content_val"] >= initial_threshold
            and item["ordinal"] >= min_frames
            and len(ledger) - item["ordinal"] >= min_frames
        ],
        min_frames,
    )
    selected: dict[int, dict[str, Any]] = {item["pts"]: dict(item) for item in natural}
    for item in selected.values():
        item["accepted_threshold"] = initial_threshold
        item["boundary_type"] = "scenedetect"

    ordinal_by_pts = {record.pts: record.ordinal for record in ledger}
    pts_list = [record.pts for record in ledger]

    def long_intervals() -> list[tuple[int, int]]:
        points = [start_pts, *sorted(selected), end_pts]
        return [
            (left, right)
            for left, right in zip(points, points[1:])
            if interval_duration(left, right, time_base) >= max_duration
        ]

    while True:
        intervals = long_intervals()
        if not intervals:
            break
        left, right = intervals[0]
        left_ordinal = ordinal_by_pts.get(left, 0)
        right_ordinal = bisect.bisect_left(pts_list, right)
        chosen: dict[str, Any] | None = None
        accepted_threshold: int | None = None
        for threshold in _threshold_ladder(config)[1:]:
            eligible = [
                item for item in candidates
                if left < item["pts"] < right
                and item["pts"] not in selected
                and item["content_val"] >= threshold
                and item["ordinal"] - left_ordinal >= min_frames
                and right_ordinal - item["ordinal"] >= min_frames
            ]
            if not eligible:
                continue
            def rank(item: dict[str, Any]) -> tuple[Any, ...]:
                left_duration = interval_duration(left, item["pts"], time_base)
                right_duration = interval_duration(item["pts"], right, time_base)
                endpoint_legal = left_duration >= min_duration and right_duration >= min_duration
                imbalance = abs(left_duration - right_duration)
                return (not endpoint_legal, -item["content_val"], imbalance, item["pts"])
            chosen = min(eligible, key=rank)
            accepted_threshold = threshold
            break
        if chosen is not None:
            accepted = dict(chosen)
            accepted["accepted_threshold"] = accepted_threshold
            accepted["boundary_type"] = "scenedetect"
            selected[accepted["pts"]] = accepted
            continue

        window_start = left + math.ceil(
            Fraction(str(profile["forced_split_search_start_s"])) / time_base
        )
        window_end = min(
            right,
            left + math.floor(
                Fraction(str(profile["forced_split_search_end_s"])) / time_base
            ),
            left + math.floor(Fraction(str(profile["forced_split_target_max_s"])) / time_base),
        )
        legal_rightmost = right - math.ceil(min_duration / time_base)
        forced_frames = [
            record
            for record in ledger
            if window_start <= record.pts <= min(window_end, legal_rightmost)
        ]
        if not forced_frames:
            raise MediaAnalysisError("No decodable frame is available for a required forced split")
        span = max(1, window_end - window_start)
        def forced_rank(record: FrameRecord) -> tuple[Any, ...]:
            metrics = qualities.get(record.pts)
            if metrics is None:
                return (1, 0.0, record.pts)
            proximity = 1.0 - (record.pts - window_start) / span
            return (not _quality_passes(metrics, config), -_quality_score(metrics, config, proximity), record.pts)
        record = min(forced_frames, key=forced_rank)
        record_metrics = qualities.get(record.pts)
        selected[record.pts] = {
            "pts": record.pts,
            "ordinal": record.ordinal,
            "content_val": 0.0,
            "delta_hue": 0.0,
            "delta_sat": 0.0,
            "delta_lum": 0.0,
            "delta_edges": 0.0,
            "accepted_threshold": None,
            "boundary_type": "forced",
            "forced_quality_status": (
                "passed"
                if record_metrics is not None and _quality_passes(record_metrics, config)
                else "forced_low_quality"
            ),
        }

    boundaries: list[dict[str, Any]] = []
    for item in sorted(selected.values(), key=lambda value: value["pts"]):
        boundary_id = stable_id(
            "bnd",
            source["file_sha256"],
            item["pts"],
            config["config_fingerprints"]["analysis"],
        )
        boundary_type = item["boundary_type"]
        boundaries.append({
            "schema_version": "1.0",
            "boundary_id": boundary_id,
            "time": time_point(item["pts"], time_base),
            "frame_ordinal": item["ordinal"],
            "boundary_type": boundary_type,
            "forced_quality_status": item.get("forced_quality_status"),
            "accepted_threshold": item["accepted_threshold"],
            "content_val": round(item["content_val"], 6),
            "components": {
                "delta_hue": round(item["delta_hue"], 6),
                "delta_sat": round(item["delta_sat"], 6),
                "delta_lum": round(item["delta_lum"], 6),
                "delta_edges": round(item["delta_edges"], 6),
            },
            "relationship": "forced_same_scene" if boundary_type == "forced" else "pending_agent",
            "hard": False if boundary_type == "forced" else None,
            "hard_reason": None,
            "diagnostic_status": "accepted",
        })
    selected_pts = set(selected)
    suppressed = [
        {
            **item,
            "diagnostic_status": "suppressed",
            "reason": "not_selected_by_threshold_ladder_or_min_scene_length",
        }
        for item in candidates
        if item["pts"] not in selected_pts
    ]
    return boundaries, suppressed


def build_atomic_segments(
    source: dict[str, Any],
    boundaries: list[dict[str, Any]],
    time_base: Fraction,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    points = [source["start"]["pts"], *[item["time"]["pts"] for item in boundaries], source["end"]["pts"]]
    segments: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(points, points[1:])):
        segment_id = stable_id(
            "seg",
            source["file_sha256"],
            left,
            right,
            config["config_fingerprints"]["analysis"],
        )
        segments.append({
            "schema_version": "1.0",
            "segment_id": segment_id,
            "sequence": index,
            "start": time_point(left, time_base),
            "end": time_point(right, time_base),
            "duration_s": decimal_string(interval_duration(left, right, time_base)),
            "left_boundary_id": boundaries[index - 1]["boundary_id"] if index else None,
            "right_boundary_id": boundaries[index]["boundary_id"] if index < len(boundaries) else None,
        })
    for index, boundary in enumerate(boundaries):
        boundary["left_segment_id"] = segments[index]["segment_id"]
        boundary["right_segment_id"] = segments[index + 1]["segment_id"]
    return segments


def apply_separator_rules(
    boundaries: list[dict[str, Any]],
    ledger: list[FrameRecord],
    qualities: dict[int, dict[str, Any]],
    time_base: Fraction,
    config: dict[str, Any],
) -> None:
    """Mark sustained black/white separators hard; ignore one-frame flashes."""

    if len(ledger) < 3:
        return
    pts_values = [item.pts for item in ledger]
    deltas = [right - left for left, right in zip(pts_values, pts_values[1:])]
    nominal_delta = max(1, int(median(deltas))) if deltas else 1
    minimum = Fraction(str(config["scene_detection"]["separator_min_duration_s"]))
    context_delta = math.ceil(
        Fraction(str(config["scene_detection"]["separator_context_s"])) / time_base
    )
    solid_ratio = float(config["scene_detection"]["separator_solid_ratio"])

    def solid(pts: int) -> bool:
        metrics = qualities.get(pts)
        return bool(metrics) and (
            metrics["black_ratio"] >= solid_ratio
            or metrics["white_ratio"] >= solid_ratio
        )

    for boundary in boundaries:
        if boundary["boundary_type"] != "scenedetect":
            continue
        cut = boundary["time"]["pts"]
        lo = bisect.bisect_left(pts_values, cut - context_delta)
        hi = bisect.bisect_right(pts_values, cut + context_delta)
        window = pts_values[lo:hi]
        runs: list[list[int]] = []
        for pts in window:
            if solid(pts):
                if not runs or pts - runs[-1][-1] > nominal_delta * 2:
                    runs.append([pts])
                else:
                    runs[-1].append(pts)
        confirmed = False
        for run in runs:
            run_duration = (run[-1] - run[0] + nominal_delta) * time_base
            normal_before = any(not solid(pts) for pts in window if pts < run[0])
            normal_after = any(not solid(pts) for pts in window if pts > run[-1])
            near_cut = run[0] - nominal_delta <= cut <= run[-1] + nominal_delta
            if (
                len(run) >= 2
                and run_duration >= minimum
                and normal_before
                and normal_after
                and near_cut
            ):
                confirmed = True
                break
        if confirmed:
            boundary["relationship"] = "auto_separator"
            boundary["hard"] = True
            boundary["hard_reason"] = "sustained_black_or_white_separator"


def choose_keyframes(
    segments: list[dict[str, Any]],
    ledger: list[FrameRecord],
    qualities: dict[int, dict[str, Any]],
    time_base: Fraction,
    config: dict[str, Any],
) -> None:
    pts_values = [item.pts for item in ledger]
    window_cap = Fraction(str(config["keyframe"]["search_window_s"]))
    window_ratio = Fraction(str(config["keyframe"]["search_window_ratio"]))
    for segment in segments:
        start = segment["start"]["pts"]
        end = segment["end"]["pts"]
        duration = interval_duration(start, end, time_base)
        window = min(window_cap, duration * window_ratio)
        window_end = start + math.floor(window / time_base)
        lo = bisect.bisect_left(pts_values, start)
        hi = bisect.bisect_right(pts_values, min(end - 1, window_end))
        records = ledger[lo:hi]
        if not records:
            segment["keyframe"] = {"quality_status": "failed", "reason": "no_decodable_frame"}
            continue
        passed = [record for record in records if record.pts in qualities and _quality_passes(qualities[record.pts], config)]
        if passed:
            chosen = passed[0]
            status = "passed"
        else:
            span = max(1, records[-1].pts - records[0].pts)
            chosen = min(
                records,
                key=lambda record: (
                    -_quality_score(
                        qualities.get(record.pts, {
                            "sharpness": 0.0, "luma_mean": 0.0, "black_ratio": 1.0,
                            "white_ratio": 0.0, "stability_delta": 1.0,
                        }),
                        config,
                        1.0 - (record.pts - records[0].pts) / span,
                    ),
                    record.pts,
                    record.ordinal,
                ),
            )
            status = "low_quality_fallback"
        metrics = qualities.get(chosen.pts, {})
        segment["keyframe"] = {
            "pts": chosen.pts,
            "source_time": time_point(chosen.pts, time_base),
            "segment_offset_s": decimal_string(interval_duration(start, chosen.pts, time_base)),
            "quality_status": status,
            "metrics": {key: round(float(value), 6) for key, value in metrics.items()},
        }


def extract_keyframe_images(
    *,
    path: Path,
    time_base: Fraction,
    rotation: int,
    segments: list[dict[str, Any]],
    output_dir: Path,
    project_dir: Path,
    jpeg_quality: int = 94,
) -> list[str]:
    """Write one canonical keyframe image for every segment with a chosen PTS."""

    try:
        import av
    except ImportError as exc:
        raise MediaAnalysisError("PyAV is required to extract keyframes") from exc
    wanted = {
        int(segment["keyframe"]["pts"]): segment
        for segment in segments
        if segment.get("keyframe", {}).get("pts") is not None
    }
    decoded: dict[int, Image.Image] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream_base = Fraction(stream.time_base)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = _rescale_pts(frame.pts, Fraction(frame.time_base or stream_base), time_base)
            if pts not in wanted:
                continue
            image = frame.to_image().convert("RGB")
            if rotation in {90, 180, 270}:
                image = image.rotate(-rotation, expand=True)
            decoded[pts] = image.copy()
            image.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []
    for pts, segment in wanted.items():
        if pts not in decoded:
            segment["keyframe"] = {"quality_status": "failed", "reason": "selected_frame_decode_failed"}
            continue
        destination = output_dir / f"{segment['segment_id']}.jpg"
        decoded[pts].save(
            destination, format="JPEG", quality=int(jpeg_quality), optimize=True
        )
        relative = destination.resolve().relative_to(project_dir.resolve()).as_posix()
        segment["keyframe"]["path"] = relative
        segment["keyframe"]["sha256"] = sha256_file(destination)
        artifacts.append(str(destination))
    for image in decoded.values():
        image.close()
    return artifacts


def build_keyframe_contact_sheet(
    segments: list[dict[str, Any]],
    destination: Path,
    project_dir: Path,
    jpeg_quality: int = 90,
) -> str | None:
    """Render a compact, labeled overview for every selected atomic keyframe."""

    available = [
        segment for segment in segments
        if segment.get("keyframe", {}).get("path")
    ]
    if not available:
        return None
    columns = 3
    cell_width, cell_height = 340, 250
    rows = math.ceil(len(available) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, segment in enumerate(available):
        keyframe = segment["keyframe"]
        image_path = project_dir / keyframe["path"]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        image.thumbnail((cell_width - 16, cell_height - 58), Image.Resampling.LANCZOS)
        column, row = index % columns, index // columns
        x = column * cell_width + (cell_width - image.width) // 2
        y = row * cell_height + 48
        sheet.paste(image, (x, y))
        image.close()
        label = (
            f"{segment['segment_id']}  pts={keyframe['pts']}  "
            f"t={keyframe['source_time']['seconds']}s  {keyframe['quality_status']}"
        )
        draw.text((column * cell_width + 8, row * cell_height + 12), label, fill="white", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="JPEG", quality=int(jpeg_quality))
    sheet.close()
    return str(destination)


def _nearest_pts(ledger_pts: list[int], target: int, lower: int, upper: int) -> int:
    start = bisect.bisect_left(ledger_pts, lower)
    stop = bisect.bisect_left(ledger_pts, upper)
    candidates = ledger_pts[start:stop]
    if not candidates:
        raise MediaAnalysisError("No evidence frame available inside segment")
    return min(candidates, key=lambda value: (abs(value - target), value))


def evidence_targets(
    boundary: dict[str, Any],
    segment_by_id: dict[str, dict[str, Any]],
    ledger: list[FrameRecord],
    time_base: Fraction,
    context_seconds: Fraction,
) -> dict[str, int]:
    pts_values = [item.pts for item in ledger]
    left = segment_by_id[boundary["left_segment_id"]]
    right = segment_by_id[boundary["right_segment_id"]]
    cut = boundary["time"]["pts"]
    context_delta = math.floor(context_seconds / time_base)
    left_start, right_end = left["start"]["pts"], right["end"]["pts"]
    cut_index = bisect.bisect_left(pts_values, cut)
    left_edge = pts_values[max(0, cut_index - 1)]
    right_edge = pts_values[min(cut_index, len(pts_values) - 1)]
    return {
        "left_keyframe": int(left["keyframe"]["pts"]),
        "left_context": _nearest_pts(pts_values, cut - context_delta, left_start, cut),
        "left_edge": left_edge,
        "right_edge": right_edge,
        "right_context": _nearest_pts(pts_values, cut + context_delta, cut, right_end),
        "right_keyframe": int(right["keyframe"]["pts"]),
    }


def extract_evidence_images(
    *,
    path: Path,
    time_base: Fraction,
    rotation: int,
    boundary_targets: dict[str, dict[str, int]],
    boundary_metadata: dict[str, dict[str, Any]],
    output_dir: Path,
    project_dir: Path,
    review_config: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    """Decode requested PTS once, write role images and per-boundary boards."""

    try:
        import av
    except ImportError as exc:
        raise MediaAnalysisError("PyAV is required to extract review evidence") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    review_config = review_config or {}
    evidence_quality = int(review_config.get("evidence_jpeg_quality", 92))
    mask_width = float(review_config.get("background_mask_width_ratio", "0.50"))
    mask_top = float(review_config.get("background_mask_top_ratio", "0.15"))
    mask_bottom = float(review_config.get("background_mask_bottom_ratio", "0.95"))
    wanted = {pts for roles in boundary_targets.values() for pts in roles.values()}
    decoded: dict[int, Image.Image] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream_base = Fraction(stream.time_base)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = _rescale_pts(frame.pts, Fraction(frame.time_base or stream_base), time_base)
            if pts not in wanted:
                continue
            image = frame.to_image().convert("RGB")
            if rotation in {90, 180, 270}:
                image = image.rotate(-rotation, expand=True)
            decoded[pts] = image.copy()
            image.close()
    missing = sorted(wanted - set(decoded))
    if missing:
        raise MediaAnalysisError(f"Could not decode review evidence PTS: {missing[:5]}")

    manifests: dict[str, dict[str, dict[str, Any]]] = {}
    board_paths: list[str] = []
    for boundary_id, roles in boundary_targets.items():
        boundary_dir = output_dir / boundary_id
        boundary_dir.mkdir(parents=True, exist_ok=True)
        role_manifest: dict[str, dict[str, Any]] = {}
        role_images: list[tuple[str, Image.Image]] = []
        for role, pts in roles.items():
            destination = boundary_dir / f"{role}.jpg"
            decoded[pts].save(
                destination, format="JPEG", quality=evidence_quality, optimize=True
            )
            relative = destination.resolve().relative_to(project_dir.resolve()).as_posix()
            role_manifest[role] = {
                "path": relative,
                "sha256": sha256_file(destination),
                "pts": pts,
                "time": time_point(pts, time_base),
            }
            role_images.append((role, decoded[pts]))

        # Background-emphasis views conceal the central foreground without a model.
        derived_images: list[Image.Image] = []
        for name, source_role in (("left_background", "left_context"), ("right_background", "right_context")):
            image = decoded[roles[source_role]].copy()
            draw = ImageDraw.Draw(image)
            width, height = image.size
            mask_left = (1.0 - mask_width) / 2.0
            mask_right = mask_left + mask_width
            draw.rectangle(
                (
                    width * mask_left,
                    height * mask_top,
                    width * mask_right,
                    height * mask_bottom,
                ),
                fill=(96, 96, 96),
            )
            destination = boundary_dir / f"{name}.jpg"
            image.save(
                destination, format="JPEG", quality=evidence_quality, optimize=True
            )
            relative = destination.resolve().relative_to(project_dir.resolve()).as_posix()
            role_manifest[name] = {
                "path": relative,
                "sha256": sha256_file(destination),
                "pts": roles[source_role],
                "time": time_point(roles[source_role], time_base),
            }
            role_images.append((name, image))
            derived_images.append(image)

        board_width = 960
        cell_width = board_width // 3
        cell_height = 230
        header_height = 48
        board = Image.new("RGB", (board_width, header_height + cell_height * 3), "white")
        draw = ImageDraw.Draw(board)
        font = ImageFont.load_default()
        metadata = boundary_metadata[boundary_id]
        header = (
            f"{boundary_id}  pts={metadata['time']['pts']}  "
            f"threshold={metadata['accepted_threshold']}  "
            f"content_val={metadata['content_val']}"
        )
        draw.text((8, 16), header, fill="black", font=font)
        for index, (role, image) in enumerate(role_images):
            thumb = image.copy()
            thumb.thumbnail((cell_width - 12, cell_height - 36), Image.Resampling.LANCZOS)
            x = (index % 3) * cell_width + (cell_width - thumb.width) // 2
            y = header_height + (index // 3) * cell_height + 24
            board.paste(thumb, (x, y))
            draw.text(
                ((index % 3) * cell_width + 8, header_height + (index // 3) * cell_height + 6),
                role,
                fill="black",
                font=font,
            )
            thumb.close()
        board_path = boundary_dir / "evidence-board.jpg"
        board.save(board_path, format="JPEG", quality=evidence_quality)
        board.close()
        for image in derived_images:
            image.close()
        role_manifest["evidence_board"] = {
            "path": board_path.resolve().relative_to(project_dir.resolve()).as_posix(),
            "sha256": sha256_file(board_path),
        }
        manifests[boundary_id] = role_manifest
        board_paths.append(str(board_path))
    for image in decoded.values():
        image.close()
    return manifests, board_paths


def build_overview_contact_sheet(
    board_paths: list[str], destination: Path, jpeg_quality: int = 90
) -> str | None:
    if not board_paths:
        return None
    boards: list[Image.Image] = []
    for path in board_paths:
        with Image.open(path) as source_image:
            boards.append(source_image.convert("RGB"))
    width = 640
    rendered: list[Image.Image] = []
    for board in boards:
        copy = board.copy()
        copy.thumbnail((width, 460), Image.Resampling.LANCZOS)
        rendered.append(copy)
    total_height = sum(item.height for item in rendered) + 12 * (len(rendered) - 1)
    sheet = Image.new("RGB", (width, total_height), (32, 32, 32))
    y = 0
    for item in rendered:
        sheet.paste(item, ((width - item.width) // 2, y))
        y += item.height + 12
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="JPEG", quality=int(jpeg_quality))
    sheet.close()
    for image in boards + rendered:
        image.close()
    return str(destination)
