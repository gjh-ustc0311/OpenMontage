"""Bounded candidate evidence and explicit, replayable representative selection.

No semantic decisions are made here. The coding agent submits those decisions.
"""

from __future__ import annotations

import bisect
import json
from copy import deepcopy
from fractions import Fraction
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator

from .config import SELECTION_IMPLEMENTATION_REVISION
from .models import fraction_from_time_base, stable_id, time_point
from .review import ReviewValidationError
from .storage import sha256_json

PROTOCOL = "representative-selection-v1"
SELECTION_ACTIONS = ("keyframe_submission", "keyframe_review_segment_ids", "keyframe_observation_requests")
STATUSES = ("pending_agent", "needs_more_evidence", "needs_human", "ready", "legacy_unreviewed")


def validate_document(name: str, value: Any) -> None:
    schema = json.loads(files("schemas.replication").joinpath(name + ".schema.json").read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: str(e.path))
    if errors:
        error = errors[0]
        raise ReviewValidationError(f"{name} {list(error.path)}: {error.message}")


def technical(metrics: dict[str, Any], config: dict[str, Any]) -> tuple[list[str], Fraction]:
    issues = []
    if metrics["sharpness"] < config["laplacian_min"]:
        issues.append("blur")
    if not config["luma_min"] <= metrics["luma_mean"] <= config["luma_max"]:
        issues.append("exposure")
    for key in ("black_ratio", "white_ratio"):
        if Fraction(str(metrics[key])) >= Fraction(config[key + "_max"]):
            issues.append(key)
    sharp = min(Fraction(str(metrics["sharpness"])) / 160, 1)
    luma = Fraction(str(metrics["luma_mean"]))
    exposure = 1 - abs(luma - Fraction(255, 2)) / Fraction(255, 2)
    return issues, (8 * sharp + 5 * exposure) / 13


def distance(left: dict, right: dict) -> Fraction:
    return Fraction((left["dhash"] ^ right["dhash"]).bit_count(), 128) + Fraction(
        sum(abs(a - b) for a, b in zip(left["rgb_grid"], right["rgb_grid"])), 24480
    )


def nearest(values: list[int], target: Fraction | int) -> int:
    index = bisect.bisect_left(values, target)
    return min(values[max(0, index - 1):index + 1], key=lambda pts: (abs(pts - target), pts))


def candidate_pts(values: list[int], qualities: dict[int, dict], config: dict,
                  limit: int, excluded: set[int] | None = None) -> dict[int, list[str]]:
    """Half temporal coverage, one quarter per-bin quality, one quarter novelty."""
    if not values or limit <= 0:
        return {}
    excluded = excluded or set()
    selected: dict[int, list[str]] = {}
    scores = {pts: technical(qualities[pts], config) for pts in values}
    def rank(pts: int) -> tuple:
        issues, score = scores[pts]
        return bool(issues), -score, pts
    def add(pts: int, reason: str) -> None:
        if pts not in excluded:
            selected.setdefault(pts, []).append(reason)
    temporal = max(1, limit // 2)
    for index in range(temporal):
        target = values[0] + Fraction(index, max(1, temporal - 1)) * (values[-1] - values[0])
        add(nearest(values, target), "temporal_coverage")
    bins = max(1, limit // 4) if limit > 1 else 0
    for index in range(bins):
        lower = values[0] + Fraction(index, bins) * (values[-1] - values[0] + 1)
        upper = values[0] + Fraction(index + 1, bins) * (values[-1] - values[0] + 1)
        bucket = [p for p in values if lower <= p < upper and p not in excluded]
        if bucket:
            add(min(bucket, key=rank), "bucket_quality")
    remaining = [p for p in values if p not in selected and p not in excluded]
    novelty_count = max(0, limit - temporal - bins)
    descriptors = {p: qualities[p]["descriptor"] for p in values}
    changes = {}
    for i, pts in enumerate(values):
        neighbors = values[max(0, i - 1):i] + values[i + 1:i + 2]
        changes[pts] = max((distance(descriptors[pts], descriptors[n]) for n in neighbors), default=Fraction(0))
    for _ in range(novelty_count):
        if not remaining:
            break
        evidence = [p for p in values if p in selected or p in excluded]
        novelty = {p: min((distance(descriptors[p], descriptors[s]) for s in evidence), default=Fraction(1)) for p in remaining}
        eligible = [p for p in remaining if novelty[p] > Fraction(config["novelty_min"])]
        if not eligible:
            break
        pts = min(eligible, key=lambda p: (rank(p)[0], -novelty[p], -changes[p], rank(p)[1], p))
        add(pts, "visual_change")
        remaining.remove(pts)
    return dict(sorted(selected.items()))


def make_candidate(source_hash: str, segment: dict, pts: int, metrics: dict,
                   reasons: list[str], config: dict) -> dict:
    base = fraction_from_time_base(segment["start"]["time_base"])
    issues, score = technical(metrics, config)
    return {
        "candidate_id": stable_id("kfc", source_hash, segment["segment_id"], pts),
        "pts": pts, "source_time": time_point(pts, base),
        "atomic_segment_offset": time_point(pts - segment["start"]["pts"], base),
        "atomic_segment_id": segment["segment_id"],
        "technical_status": "failed" if issues else "passed",
        "quality_issues": issues,
        "technical_score": float(score),
        "metrics": {k: v for k, v in metrics.items() if k != "descriptor"},
        "reasons": reasons, "role": "observation",
    }


def request_for(state: dict, segment_ids: list[str]) -> dict:
    body = {
        "schema_version": "1.0", "protocol_version": PROTOCOL,
        "implementation_revision": SELECTION_IMPLEMENTATION_REVISION,
        "source_sha256": state["source"]["file_sha256"],
        "parent_plan_revision": state["plan_revision"],
        "selection_fingerprint": state["config"]["config_fingerprints"]["selection"],
        "limits": state["config"]["representative_selection"],
        "items": [],
    }
    for segment in state["atomic_segments"]:
        if segment["segment_id"] not in segment_ids:
            continue
        selection = segment["anchor_selection"]
        body["items"].append({
            "segment_id": segment["segment_id"], "start": segment["start"], "end": segment["end"],
            "selection_task_id": selection["task_id"],
            "observation_rounds": selection["observation_rounds"],
            "candidates": deepcopy(selection["candidates"]),
            "contact_sheets": deepcopy(selection["contact_sheets"]),
        })
    request = {"request_id": stable_id("kfr", body), **body}
    request["request_sha256"] = sha256_json(request)
    return request


def check_submission(state: dict, request: dict, submission: dict) -> dict[str, dict]:
    validate_document("keyframe_selection_submission", submission)
    if not submission["reviewer"]["name"].strip():
        raise ReviewValidationError("Reviewer name must not be blank")
    for field in ("request_id", "request_sha256", "parent_plan_revision", "protocol_version"):
        if submission[field] != request[field]:
            raise ReviewValidationError(f"Keyframe submission {field} does not match request")
    if request["source_sha256"] != state["source"]["file_sha256"] or request["selection_fingerprint"] != state["config"]["config_fingerprints"]["selection"]:
        raise ReviewValidationError("Keyframe request dependencies changed")
    expected = {item["segment_id"]: item for item in request["items"]}
    current = {s["segment_id"]: s for s in state["atomic_segments"]}
    decisions = {}
    for decision in submission["decisions"]:
        sid = decision["segment_id"]
        if sid not in expected or sid in decisions:
            raise ReviewValidationError("Unknown or duplicate keyframe segment")
        item = expected[sid]
        if item["start"] != current[sid]["start"] or item["end"] != current[sid]["end"] or item["selection_task_id"] != current[sid]["anchor_selection"]["task_id"]:
            raise ReviewValidationError("Keyframe segment dependencies changed")
        candidates = {c["candidate_id"]: c for c in item["candidates"]}
        if any(ref not in candidates for ref in decision["evidence_refs"]):
            raise ReviewValidationError("Unknown keyframe evidence reference")
        selected = decision.get("primary_candidate_id")
        supplements = decision.get("supplementary_anchors", [])
        if any(not a["purpose"].strip() for a in supplements):
            raise ReviewValidationError("Supplementary anchors require an observable purpose")
        if decision["status"] == "ready":
            ids = [selected, *[s["candidate_id"] for s in supplements]]
            if not selected or any(c not in candidates for c in ids) or len(ids) != len(set(ids)):
                raise ReviewValidationError("Ready selection requires one primary and unique known anchors")
            if any(c not in decision["evidence_refs"] for c in ids):
                raise ReviewValidationError("Selected anchors must be included in viewed evidence")
            if any(candidates[c]["technical_status"] != "passed" for c in ids):
                if submission["reviewer"]["kind"] != "human" or not decision.get("quality_override_reason", "").strip():
                    raise ReviewValidationError("Technical quality failure requires an explicit human override reason")
        elif selected is not None or supplements:
            raise ReviewValidationError("Unresolved selection cannot expose adopted anchors")
        if not decision["rationale"].strip():
            raise ReviewValidationError("Keyframe rationale must not be blank")
        decisions[sid] = deepcopy(decision)
    if set(decisions) != set(expected):
        raise ReviewValidationError("Keyframe submission must address every requested segment")
    return decisions


def bind_anchors(state: dict) -> None:
    """Update image associations only; do not repartition the timeline."""
    by_id = {s["segment_id"]: s for s in state["atomic_segments"]}
    for clip in state.get("generation_clips", []):
        clip["schema_version"] = "3.0"
        anchors = []
        statuses = []
        for sid in clip["atomic_segment_ids"]:
            selection = by_id[sid].get("anchor_selection", {})
            statuses.append(selection.get("status", "legacy_unreviewed"))
            if selection.get("status") != "ready":
                continue
            for anchor in selection["anchors"]:
                value = deepcopy(anchor)
                base = fraction_from_time_base(value["source_time"]["time_base"])
                offset = time_point(value["pts"] - clip["start"]["pts"], base)
                value.update(generation_clip_id=clip["clip_id"], generation_clip_offset=offset,
                             generation_clip_offset_s=offset["seconds"])
                anchors.append(value)
        clip["keyframes"] = anchors
        clip["anchor_status"] = aggregate_status(statuses)
    dropped = {s["segment_id"] for s in state.get("dropped_intervals", [])}
    state["anchor_status"] = aggregate_status([
        s.get("anchor_selection", {}).get("status", "legacy_unreviewed")
        for s in state["atomic_segments"] if s["segment_id"] not in dropped
    ])


def aggregate_status(statuses: list[str]) -> str:
    if all(s == "ready" for s in statuses):
        return "ready"
    # Do not let one human blocker hide independent agent work.
    for status in ("pending_agent", "needs_more_evidence", "needs_human", "legacy_unreviewed"):
        if status in statuses:
            return status
    return "pending_agent"
