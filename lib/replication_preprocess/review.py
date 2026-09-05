"""Strict, replayable contract for AI-assisted boundary review."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .storage import sha256_json


class ReviewValidationError(ValueError):
    pass


_RELATIONSHIPS = {"same_scene", "different_scene", "uncertain"}
_CONFIDENCE = {"high", "medium", "low"}


def build_review_request(
    *,
    source_sha256: str,
    parent_plan_revision: str,
    config_fingerprint: str,
    protocol_version: str,
    review_round: int,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    body = {
        "schema_version": "1.0",
        "source_sha256": source_sha256,
        "parent_plan_revision": parent_plan_revision,
        "config_fingerprint": config_fingerprint,
        "review_protocol_version": protocol_version,
        "review_round": review_round,
        "allowed_relationships": sorted(_RELATIONSHIPS),
        "items": items,
    }
    request_id = f"brq_{sha256_json(body)[:16]}"
    request = {"request_id": request_id, **body}
    request["request_sha256"] = sha256_json(request)
    return request


def apply_review_submission(
    boundaries: list[dict[str, Any]],
    request: dict[str, Any],
    submission: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Validate a complete submission and apply it to copied boundaries."""

    for field in (
        "request_id",
        "request_sha256",
        "parent_plan_revision",
        "review_protocol_version",
        "reviewer",
        "decisions",
    ):
        if field not in submission:
            raise ReviewValidationError(f"Review submission missing {field}")
    for field in ("request_id", "request_sha256", "parent_plan_revision", "review_protocol_version"):
        expected = request[field]
        if submission[field] != expected:
            raise ReviewValidationError(f"Review submission {field} does not match request")
    reviewer = submission["reviewer"]
    if not isinstance(reviewer, dict) or reviewer.get("kind") not in {
        "ai_coding_assistant", "human"
    }:
        raise ReviewValidationError("reviewer.kind must be ai_coding_assistant or human")
    if not isinstance(submission["decisions"], list):
        raise ReviewValidationError("Review submission decisions must be an array")

    expected_items = {item["review_item_id"]: item for item in request["items"]}
    seen: set[str] = set()
    decision_by_boundary: dict[str, dict[str, Any]] = {}
    for decision in submission["decisions"]:
        if not isinstance(decision, dict):
            raise ReviewValidationError("Each review decision must be an object")
        item_id = decision.get("review_item_id")
        if item_id in seen:
            raise ReviewValidationError(f"Duplicate review item: {item_id}")
        if item_id not in expected_items:
            raise ReviewValidationError(f"Unknown review item: {item_id}")
        seen.add(item_id)
        expected = expected_items[item_id]
        if decision.get("boundary_id") != expected["boundary_id"]:
            raise ReviewValidationError(f"Boundary mismatch for {item_id}")
        relationship = decision.get("relationship")
        confidence = decision.get("confidence")
        if relationship not in _RELATIONSHIPS:
            raise ReviewValidationError(f"Invalid relationship for {item_id}")
        if confidence not in _CONFIDENCE:
            raise ReviewValidationError(f"Invalid confidence for {item_id}")
        if relationship == "uncertain" and confidence != "low":
            raise ReviewValidationError("uncertain decisions must use low confidence")
        if relationship != "uncertain" and confidence == "low":
            raise ReviewValidationError("low-confidence decisions must be uncertain")
        evidence_refs = decision.get("evidence_refs")
        if not isinstance(evidence_refs, list) or not evidence_refs:
            raise ReviewValidationError(f"Evidence references required for {item_id}")
        allowed_refs = set(expected.get("evidence", {}))
        if any(ref not in allowed_refs for ref in evidence_refs):
            raise ReviewValidationError(f"Unknown evidence reference for {item_id}")
        rationale = decision.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ReviewValidationError(f"Rationale required for {item_id}")
        decision_by_boundary[expected["boundary_id"]] = deepcopy(decision)
    if seen != set(expected_items):
        missing = sorted(set(expected_items) - seen)
        raise ReviewValidationError(f"Review submission incomplete: {missing}")

    result = deepcopy(boundaries)
    unresolved: list[str] = []
    for boundary in result:
        decision = decision_by_boundary.get(boundary["boundary_id"])
        if not decision:
            continue
        relationship = decision["relationship"]
        boundary["relationship"] = relationship
        boundary["review"] = {
            "reviewer": reviewer,
            "confidence": decision["confidence"],
            "evidence_refs": decision["evidence_refs"],
            "rationale": decision["rationale"],
        }
        if relationship == "same_scene":
            boundary["hard"] = False
            boundary["hard_reason"] = None
        elif relationship == "different_scene":
            boundary["hard"] = True
            boundary["hard_reason"] = "agent_review_different_scene"
        else:
            boundary["hard"] = None
            boundary["hard_reason"] = "agent_review_uncertain"
            unresolved.append(boundary["boundary_id"])
    return result, sha256_json(submission), unresolved
