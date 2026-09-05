"""Scene grouping and globally constrained generation-clip planning."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
from typing import Any, Iterable

from .config import compute_plan_fingerprint
from .models import decimal_string, fraction_from_time_base, stable_id


@dataclass(frozen=True)
class _Solution:
    cost: tuple[Any, ...]
    clips: tuple[dict[str, Any], ...]
    dropped: tuple[dict[str, Any], ...]


def _duration(segment: dict[str, Any]) -> Fraction:
    return Fraction(segment["duration_s"])


def _strength_cost(boundaries: Iterable[dict[str, Any]]) -> int:
    return sum(int(Decimal(str(item.get("content_val", 0))) * 1000) for item in boundaries)


def _is_crossable(boundary: dict[str, Any]) -> bool:
    if boundary.get("hard") is True:
        return False
    return boundary.get("relationship") in {"same_scene", "forced_same_scene", "human_soft"}


def build_scene_groups(
    segments: list[dict[str, Any]], boundaries: list[dict[str, Any]], source_hash: str
) -> list[dict[str, Any]]:
    """Chain adjacent segments across confirmed soft boundaries."""

    if not segments:
        return []
    by_right_segment = {
        boundary.get("right_segment_id"): boundary for boundary in boundaries
    }
    groups: list[list[dict[str, Any]]] = [[segments[0]]]
    for segment in segments[1:]:
        boundary = by_right_segment.get(segment["segment_id"])
        if boundary and _is_crossable(boundary):
            groups[-1].append(segment)
        else:
            groups.append([segment])
    result: list[dict[str, Any]] = []
    for members in groups:
        group_id = stable_id(
            "sg", source_hash, members[0]["segment_id"], members[-1]["segment_id"]
        )
        result.append({
            "schema_version": "1.0",
            "scene_group_id": group_id,
            "segment_ids": [item["segment_id"] for item in members],
        })
        for item in members:
            item["scene_group_id"] = group_id
    return result


def build_generation_plan(
    segments: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    source_hash: str,
    review_digest: str,
    approved_drop_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return generation clips, dropped intervals, and unresolved review items.

    The dynamic program is deterministic.  It may cross only boundaries already
    classified as soft by the agent, a human override, or a forced split rule.
    """

    approved_drop_ids = approved_drop_ids or set()
    if not segments:
        return [], [], [{"reason": "empty_timeline", "candidate_actions": []}]
    profile = config["profile"]
    regroup = config["regroup"]
    minimum = Fraction(str(profile["min_duration_s"]))
    maximum = Fraction(str(profile["max_duration_s"]))
    drop_max = Fraction(str(regroup["drop_threshold_s"]))
    max_members = int(regroup["max_atomic_segments"])
    plan_fingerprint = config.get("plan_fingerprint") or compute_plan_fingerprint(
        config, review_digest
    )
    boundary_by_pair = {
        (item.get("left_segment_id"), item.get("right_segment_id")): item
        for item in boundaries
    }
    n = len(segments)
    best: list[_Solution | None] = [None] * (n + 1)
    best[n] = _Solution((Fraction(0), 0, 0, 0, ()), (), ())

    for index in range(n - 1, -1, -1):
        candidates: list[_Solution] = []
        current = segments[index]
        if (
            current["segment_id"] in approved_drop_ids
            and bool(regroup["allow_drop_under_1s"])
            and _duration(current) < drop_max
            and best[index + 1] is not None
        ):
            tail = best[index + 1]
            assert tail is not None
            dropped = {
                "schema_version": "1.0",
                "segment_id": current["segment_id"],
                "start": current["start"],
                "end": current["end"],
                "duration_s": current["duration_s"],
                "audio_start": current["start"],
                "audio_end": current["end"],
                "reason": "human_approved_drop_under_1s",
            }
            candidates.append(_Solution(
                (
                    tail.cost[0] + _duration(current),
                    tail.cost[1] + 1,
                    tail.cost[2],
                    tail.cost[3],
                    ("drop", current["segment_id"], tail.cost[4]),
                ),
                tail.clips,
                (dropped,) + tail.dropped,
            ))

        running = Fraction(0)
        for end_index in range(index, min(n, index + max_members)):
            member = segments[end_index]
            running += _duration(member)
            members = segments[index : end_index + 1]
            if running >= maximum:
                break
            normal_count = sum(_duration(item) >= minimum for item in members)
            if normal_count > 1:
                continue
            if running < minimum:
                continue
            internal: list[dict[str, Any]] = []
            crossable = True
            for left, right in zip(members, members[1:]):
                boundary = boundary_by_pair.get((left["segment_id"], right["segment_id"]))
                if boundary is None or not _is_crossable(boundary):
                    crossable = False
                    break
                internal.append(boundary)
            if not crossable or best[end_index + 1] is None:
                continue
            tail = best[end_index + 1]
            assert tail is not None
            clip_id = stable_id(
                "gc",
                source_hash,
                members[0]["segment_id"],
                members[-1]["segment_id"],
                plan_fingerprint,
            )
            clip_time_base = fraction_from_time_base(members[0]["start"]["time_base"])
            clip_keyframes: list[dict[str, Any]] = []
            for member_index, item in enumerate(members):
                keyframe = deepcopy(item.get("keyframe"))
                if keyframe is not None and keyframe.get("pts") is not None:
                    keyframe["generation_clip_offset_s"] = decimal_string(
                        (int(keyframe["pts"]) - int(members[0]["start"]["pts"]))
                        * clip_time_base
                    )
                    keyframe["role"] = "primary" if member_index == 0 else "internal_anchor"
                    keyframe["atomic_segment_id"] = item["segment_id"]
                clip_keyframes.append(keyframe)
            clip = {
                "schema_version": "1.0",
                "clip_id": clip_id,
                "start": members[0]["start"],
                "end": members[-1]["end"],
                "duration_s": decimal_string(running),
                "atomic_segment_ids": [item["segment_id"] for item in members],
                "keyframes": clip_keyframes,
                "internal_boundaries": [item["boundary_id"] for item in internal],
                "status": "planned",
            }
            candidates.append(_Solution(
                (
                    tail.cost[0],
                    tail.cost[1],
                    tail.cost[2] + _strength_cost(internal),
                    tail.cost[3] + max(0, len(members) - 1),
                    (clip_id, tail.cost[4]),
                ),
                (clip,) + tail.clips,
                tail.dropped,
            ))
        if candidates:
            best[index] = min(candidates, key=lambda item: item.cost)

    if best[0] is None:
        unresolved = [
            {
                "schema_version": "1.0",
                "review_item_id": stable_id("ri", source_hash, item["segment_id"], "no_legal_group"),
                "reason": "no_legal_generation_clip",
                "affected_segment_ids": [item["segment_id"]],
                "candidate_actions": ["review_adjacent_boundary", "insert_boundary", "drop_if_under_1s"],
            }
            for item in segments
            if _duration(item) < minimum
        ]
        return [], [], unresolved or [{
            "schema_version": "1.0",
            "review_item_id": stable_id("ri", source_hash, "timeline", "no_path"),
            "reason": "no_legal_timeline_path",
            "affected_segment_ids": [item["segment_id"] for item in segments],
            "candidate_actions": ["review_boundaries"],
        }]
    return list(best[0].clips), list(best[0].dropped), []
