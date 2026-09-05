from __future__ import annotations

from fractions import Fraction
from copy import deepcopy

import pytest

from lib.replication_preprocess.analysis import (
    FrameRecord,
    apply_separator_rules,
    choose_keyframes,
    select_boundaries,
)
from lib.replication_preprocess.models import load_config, stable_id, time_point, validate_config
from lib.replication_preprocess.planner import build_generation_plan, build_scene_groups
from lib.replication_preprocess.review import (
    ReviewValidationError,
    apply_review_submission,
    build_review_request,
)


TIME_BASE = Fraction(1, 10)
SOURCE_HASH = "a" * 64


def _segment(sequence: int, start: int, end: int) -> dict:
    return {
        "schema_version": "1.0",
        "segment_id": f"seg_{sequence}",
        "sequence": sequence,
        "start": time_point(start, TIME_BASE),
        "end": time_point(end, TIME_BASE),
        "duration_s": str(Fraction(end - start, 10)),
        "keyframe": {"pts": start, "quality_status": "passed"},
    }


def _boundary(left: dict, right: dict, relationship: str, *, hard: bool | None = None) -> dict:
    return {
        "schema_version": "1.0",
        "boundary_id": f"bnd_{left['sequence']}_{right['sequence']}",
        "left_segment_id": left["segment_id"],
        "right_segment_id": right["segment_id"],
        "relationship": relationship,
        "hard": hard,
        "content_val": 31.25,
    }


def _plan(segments: list[dict], boundaries: list[dict], *, drops: set[str] | None = None):
    config = load_config()
    return build_generation_plan(
        segments,
        boundaries,
        config,
        source_hash=SOURCE_HASH,
        review_digest="review-digest",
        approved_drop_ids=drops,
    )


def test_planner_keeps_two_legal_segments_as_separate_generation_clips() -> None:
    segments = [_segment(0, 0, 33), _segment(1, 33, 74)]
    boundary = _boundary(segments[0], segments[1], "same_scene", hard=False)

    clips, dropped, unresolved = _plan(segments, [boundary])

    assert [clip["duration_s"] for clip in clips] == ["3.3", "4.1"]
    assert dropped == []
    assert unresolved == []


def test_planner_merges_short_segments_only_across_confirmed_soft_boundary() -> None:
    segments = [_segment(0, 0, 14), _segment(1, 14, 34)]
    soft = _boundary(segments[0], segments[1], "same_scene", hard=False)

    clips, _, unresolved = _plan(segments, [soft])

    assert len(clips) == 1
    assert clips[0]["atomic_segment_ids"] == ["seg_0", "seg_1"]
    assert clips[0]["duration_s"] == "3.4"
    assert [item["generation_clip_offset_s"] for item in clips[0]["keyframes"]] == [
        "0", "1.4"
    ]
    assert [item["role"] for item in clips[0]["keyframes"]] == [
        "primary", "internal_anchor"
    ]
    assert unresolved == []

    hard = _boundary(segments[0], segments[1], "different_scene", hard=True)
    clips, _, unresolved = _plan(segments, [hard])
    assert clips == []
    assert unresolved


def test_planner_partitions_four_short_segments_without_crossing_more_than_three() -> None:
    segments = [_segment(index, index * 20, (index + 1) * 20) for index in range(4)]
    boundaries = [
        _boundary(left, right, "same_scene", hard=False)
        for left, right in zip(segments, segments[1:])
    ]

    clips, _, unresolved = _plan(segments, boundaries)

    assert [len(clip["atomic_segment_ids"]) for clip in clips] == [2, 2]
    assert unresolved == []


def test_scene_groups_stop_at_hard_or_unresolved_boundaries() -> None:
    segments = [_segment(index, index * 20, (index + 1) * 20) for index in range(3)]
    boundaries = [
        _boundary(segments[0], segments[1], "same_scene", hard=False),
        _boundary(segments[1], segments[2], "uncertain", hard=None),
    ]

    groups = build_scene_groups(segments, boundaries, SOURCE_HASH)

    assert [group["segment_ids"] for group in groups] == [["seg_0", "seg_1"], ["seg_2"]]


def test_explicit_drop_is_the_only_way_to_remove_a_subsecond_segment() -> None:
    segments = [_segment(0, 0, 7), _segment(1, 7, 40)]
    boundary = _boundary(segments[0], segments[1], "different_scene", hard=True)
    clips, dropped, unresolved = _plan(segments, [boundary])
    assert clips == []
    assert dropped == []
    assert unresolved

    config = load_config()
    config["regroup"]["allow_drop_under_1s"] = True
    clips, dropped, unresolved = build_generation_plan(
        segments,
        [boundary],
        config,
        source_hash=SOURCE_HASH,
        review_digest="review-digest",
        approved_drop_ids={"seg_0"},
    )
    assert [item["segment_id"] for item in dropped] == ["seg_0"]
    assert dropped[0]["audio_start"] == segments[0]["start"]
    assert [clip["atomic_segment_ids"] for clip in clips] == [["seg_1"]]
    assert unresolved == []


def _quality(*, passed: bool = True, solid: str | None = None) -> dict:
    return {
        "sharpness": 120.0 if passed else 0.0,
        "luma_mean": 120.0 if solid is None else (0.0 if solid == "black" else 255.0),
        "black_ratio": 1.0 if solid == "black" else 0.0,
        "white_ratio": 1.0 if solid == "white" else 0.0,
        "stability_delta": 0.01,
    }


def test_long_interval_forces_repeatable_low_risk_splits() -> None:
    config = load_config()
    ledger = [FrameRecord(index, index) for index in range(310)]
    qualities = {record.pts: _quality() for record in ledger}
    source = {
        "file_sha256": SOURCE_HASH,
        "start": time_point(0, TIME_BASE),
        "end": time_point(310, TIME_BASE),
    }

    boundaries, suppressed = select_boundaries(
        source=source,
        ledger=ledger,
        candidates=[],
        qualities=qualities,
        time_base=TIME_BASE,
        config=config,
    )

    assert [item["time"]["pts"] for item in boundaries] == [80, 160, 240]
    assert all(item["relationship"] == "forced_same_scene" for item in boundaries)
    assert all(item["forced_quality_status"] == "passed" for item in boundaries)
    assert suppressed == []


def test_sustained_separator_is_hard_but_single_flash_is_not() -> None:
    config = load_config()
    ledger = [FrameRecord(index, index) for index in range(20)]
    base_boundary = {
        "boundary_id": "bnd_separator",
        "boundary_type": "scenedetect",
        "time": time_point(10, TIME_BASE),
        "relationship": "pending_agent",
        "hard": None,
        "hard_reason": None,
    }
    sustained = {record.pts: _quality() for record in ledger}
    sustained[10] = _quality(passed=False, solid="black")
    sustained[11] = _quality(passed=False, solid="black")
    boundary = deepcopy(base_boundary)
    apply_separator_rules([boundary], ledger, sustained, TIME_BASE, config)
    assert boundary["relationship"] == "auto_separator"
    assert boundary["hard"] is True

    flash = {record.pts: _quality() for record in ledger}
    flash[10] = _quality(passed=False, solid="white")
    boundary = deepcopy(base_boundary)
    apply_separator_rules([boundary], ledger, flash, TIME_BASE, config)
    assert boundary["relationship"] == "pending_agent"
    assert boundary["hard"] is None


def test_keyframe_selector_skips_a_black_start_and_records_true_offset() -> None:
    config = load_config()
    ledger = [FrameRecord(index, index) for index in range(10)]
    qualities = {record.pts: _quality() for record in ledger}
    qualities[0] = _quality(passed=False, solid="black")
    qualities[1] = _quality(passed=False, solid="black")
    segment = _segment(0, 0, 10)

    choose_keyframes([segment], ledger, qualities, TIME_BASE, config)

    assert segment["keyframe"]["pts"] == 2
    assert segment["keyframe"]["segment_offset_s"] == "0.2"
    assert segment["keyframe"]["quality_status"] == "passed"


def _review_fixture() -> tuple[list[dict], dict]:
    segments = [_segment(0, 0, 20), _segment(1, 20, 40)]
    boundary = _boundary(segments[0], segments[1], "pending_agent", hard=None)
    boundary["boundary_id"] = "bnd_" + "b" * 16
    item = {
        "review_item_id": "bri_" + "c" * 16,
        "boundary_id": boundary["boundary_id"],
        "left_segment_id": segments[0]["segment_id"],
        "right_segment_id": segments[1]["segment_id"],
        "time": time_point(20, TIME_BASE),
        "content_val": 42.0,
        "accepted_threshold": 20,
        "components": {"delta_lum": 42.0},
        "evidence": {
            "left_edge": {"path": "left.jpg", "sha256": "d" * 64},
            "right_edge": {"path": "right.jpg", "sha256": "e" * 64},
        },
    }
    request = build_review_request(
        source_sha256=SOURCE_HASH,
        parent_plan_revision="r0001",
        config_fingerprint="f" * 64,
        protocol_version="boundary-visual-review-v1",
        review_round=1,
        items=[item],
    )
    return [boundary], request


def _submission(request: dict, relationship: str = "same_scene") -> dict:
    item = request["items"][0]
    return {
        "schema_version": "1.0",
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "parent_plan_revision": request["parent_plan_revision"],
        "review_protocol_version": request["review_protocol_version"],
        "reviewer": {"kind": "ai_coding_assistant", "name": "test-agent"},
        "decisions": [{
            "review_item_id": item["review_item_id"],
            "boundary_id": item["boundary_id"],
            "relationship": relationship,
            "confidence": "low" if relationship == "uncertain" else "high",
            "evidence_refs": ["left_edge", "right_edge"],
            "rationale": "The product and background remain continuous across the cut.",
        }],
    }


def test_review_submission_is_complete_strict_and_replayable() -> None:
    boundaries, request = _review_fixture()
    submission = _submission(request)

    reviewed, digest, unresolved = apply_review_submission(boundaries, request, submission)

    assert reviewed[0]["relationship"] == "same_scene"
    assert reviewed[0]["hard"] is False
    assert len(digest) == 64
    assert unresolved == []
    assert boundaries[0]["relationship"] == "pending_agent"


def test_review_submission_rejects_stale_or_incomplete_decisions() -> None:
    boundaries, request = _review_fixture()
    stale = _submission(request)
    stale["request_sha256"] = "0" * 64
    with pytest.raises(ReviewValidationError, match="does not match request"):
        apply_review_submission(boundaries, request, stale)

    incomplete = _submission(request)
    incomplete["decisions"] = []
    with pytest.raises(ReviewValidationError, match="incomplete"):
        apply_review_submission(boundaries, request, incomplete)


def test_stable_ids_and_requests_do_not_depend_on_wall_clock() -> None:
    boundaries, request_a = _review_fixture()
    _, request_b = _review_fixture()
    assert request_a == request_b
    assert stable_id("seg", SOURCE_HASH, 0, 33) == stable_id("seg", SOURCE_HASH, 0, 33)
    assert boundaries[0]["boundary_id"].startswith("bnd_")


def test_config_rejects_backends_or_policies_the_engine_does_not_implement() -> None:
    config = load_config()
    config["scene_detection"]["backend"] = "opencv"
    with pytest.raises(ValueError, match="PyAV"):
        validate_config(config)

    config = load_config()
    config["regroup"]["allow_cross_hard_boundary"] = True
    with pytest.raises(ValueError, match="hard boundaries"):
        validate_config(config)
