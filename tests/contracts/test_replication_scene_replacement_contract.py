from __future__ import annotations

from copy import deepcopy

import pytest

from schemas.artifacts import validate_artifact
from schemas.replication import (
    PHASE2_SCHEMA_PATHS,
    load_phase2_schema,
    validate_phase2_document,
)
from tests.lib.test_replication_scene_replacement_v2 import _plan


HASH = "a" * 64
REFERENCE = {"path": "artifacts/evidence.json", "sha256": HASH}
HUMAN = {"kind": "human", "actor": "reviewer", "reason": "Approved"}


def _rules(rule_ids: list[str]) -> list[dict]:
    return [
        {
            "rule_id": rule_id,
            "status": "pass",
            "reason": "Passed",
            "evidence_refs": [REFERENCE],
        }
        for rule_id in rule_ids
    ]


def test_registry_exposes_only_direct_scene_replacement_contracts() -> None:
    names = {path.name for path in PHASE2_SCHEMA_PATHS}
    assert "replication_scene_replacement_v2.schema.json" in names
    assert "scene_replacement_v2_state.schema.json" in names
    assert "replication_scene_replacement.schema.json" not in names
    assert "scene_replacement_protection_submission.schema.json" not in names
    with pytest.raises(FileNotFoundError):
        load_phase2_schema("scene_replacement_protection_submission")


def test_scene_plan_requires_exactly_three_unselected_directions() -> None:
    plan = _plan(HASH)
    validate_phase2_document("scene_replacement_v2_plan", plan)

    two = deepcopy(plan)
    two["physical_scenes"][0]["directions"].pop()
    with pytest.raises(Exception):
        validate_phase2_document("scene_replacement_v2_plan", two)


def test_public_tool_records_selection_against_exact_presented_options() -> None:
    submission = {
        "operation": "record_direction_selection",
        "project_id": "demo",
        "output_path": (
            "/projects/demo/artifacts/replication/scene-replacement-v2/index.json"
        ),
        "parent_replacement_revision": "r0002",
        "submission": {
            "scene_plan_id": "plan_1",
            "scene_id": "scene_1",
            "presented_direction_ids": ["direction_a", "direction_b", "direction_c"],
            "directions_sha256": HASH,
            "selected_direction_id": "direction_a",
            "human_confirmation": HUMAN,
        },
    }
    validate_phase2_document("replication_scene_replacement_v2", submission)
    invalid = deepcopy(submission)
    invalid["submission"]["presented_direction_ids"].pop()
    with pytest.raises(Exception):
        validate_phase2_document("replication_scene_replacement_v2", invalid)


def test_prepare_edit_contract_prevents_caller_prompt_or_reference_override() -> None:
    valid = {
        "operation": "prepare_edit",
        "project_id": "demo",
        "output_path": (
            "/projects/demo/artifacts/replication/scene-replacement-v2/index.json"
        ),
        "parent_replacement_revision": "r0003",
        "submission": {
            "anchor_id": "anchor_1",
            "generation_mode": "initial",
            "creative_instructions": "Keep the environment casually lived in",
        },
    }
    validate_phase2_document("replication_scene_replacement_v2", valid)
    for forbidden in ("prompt", "approved_target", "ordered_references"):
        invalid = deepcopy(valid)
        invalid["submission"][forbidden] = "not allowed"
        with pytest.raises(Exception):
            validate_phase2_document("replication_scene_replacement_v2", invalid)


def test_global_review_requires_candidate_digest_order_and_human_confirmation() -> None:
    review = {
        "stage": "global",
        "scope_id": "candidate_1",
        "candidate_id": "candidate_1",
        "candidate_fingerprint": HASH,
        "comparison_sha256": HASH,
        "anchor_ids": ["anchor_1"],
        "status": "pass",
        "reviewer": {"kind": "human", "name": "reviewer"},
        "rules": _rules(["all_anchors_present"]),
        "evidence_refs": [REFERENCE],
        "issues": [],
        "human_confirmation": HUMAN,
    }
    validate_phase2_document("scene_replacement_v2_review_submission", review)
    invalid = deepcopy(review)
    invalid.pop("comparison_sha256")
    with pytest.raises(Exception):
        validate_phase2_document("scene_replacement_v2_review_submission", invalid)


def test_package_checkpoint_exposes_global_approval_gate() -> None:
    package = {
        "version": "2.1",
        "project_id": "demo",
        "replacement_revision": "r0010",
        "status": "awaiting_global_approval",
        "current_index_ref": REFERENCE,
        "source_snapshot_ref": REFERENCE,
        "counts": {
            "scenes_total": 1,
            "scenes_approved": 1,
            "anchors_total": 1,
            "anchors_adopted": 1,
        },
        "scenes": [
            {
                "scene_id": "scene_1",
                "status": "ready",
                "anchor_ids": ["anchor_1"],
                "group_human_approved": True,
            }
        ],
        "blockers": [],
        "global_comparison_ref": REFERENCE,
        "global_review_id": None,
        "global_human_approved": False,
    }
    validate_artifact("scene_replacement_package", package)
