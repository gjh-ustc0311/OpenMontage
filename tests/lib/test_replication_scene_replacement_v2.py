from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from lib.replication_scene_replacement.engine_v2 import (
    AUDIENCE_PROFILE,
    GENERATED_RESULT_RULES,
    GLOBAL_RULES,
    GROUP_RULES,
    PRE_GENERATION_RULES,
    SAMPLE_RULES,
    SceneReplacementV2Engine,
)
from lib.replication_scene_replacement.errors import SceneReplacementError
from lib.replication_scene_replacement.storage import sha256_file, sha256_json
from lib.replication_scene_replacement.upstream import presentation_map_from_delivery
from schemas.replication import validate_phase2_document
from tests.lib.replication_scene_replacement_fixtures import _phase1


def _ref(value: dict) -> dict[str, str]:
    return {"path": value["path"], "sha256": value["sha256"]}


def _direction(direction_id: str, setting: str) -> dict:
    return {
        "direction_id": direction_id,
        "description": f"Realistic {setting}",
        "visual_changes": [f"Replace the original environment with a {setting}"],
        "function_preservation": (
            "Keep product pose, clearance, support, and interaction unchanged"
        ),
        "cross_view_strategy": (
            "Reuse the same worktop, wall, light direction, and ordinary storage"
        ),
        "consumer_context": {
            "setting_type": setting,
            "ownership_or_access_rationale": (
                "An ordinary North American household can realistically use this place"
            ),
            "usage_habit_rationale": (
                "Adults aged 30-50 commonly prepare practical family meals here"
            ),
            "authenticity_cues": [
                "normal household lighting",
                "believable everyday storage",
            ],
            "anti_studio_constraints": [
                "no seamless backdrop or hero-light product staging"
            ],
        },
    }


def _plan(snapshot_hash: str) -> dict:
    return {
        "schema_version": "2.1",
        "source_snapshot_fingerprint": snapshot_hash,
        "audience_profile": deepcopy(AUDIENCE_PROFILE),
        "physical_scenes": [
            {
                "scene_id": "scene_1",
                "anchor_ids": ["anchor_1"],
                "shared_constraints": [
                    "Keep a credible consumer food-prep work area"
                ],
                "per_anchor_constraints": {
                    "anchor_1": [
                        "Keep exact product appearance, pose, position, and contact"
                    ]
                },
                "directions": [
                    _direction("direction_a", "lived-in suburban kitchen"),
                    _direction("direction_b", "finished basement snack counter"),
                    _direction("direction_c", "covered backyard patio prep table"),
                ],
                "selected_direction_id": None,
                "selection_confirmation": None,
                "main_anchor_id": "anchor_1",
                "difficult_view": {
                    "status": "not_required",
                    "reason": (
                        "The only anchor exposes the complete product and support plane"
                    ),
                },
            }
        ],
    }


@pytest.fixture
def v2_workflow(tmp_path: Path):
    project = tmp_path / "projects" / "demo"
    project.mkdir(parents=True)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"phase one source bytes")
    _phase1(project, source)
    engine = SceneReplacementV2Engine(
        project,
        project / "artifacts/replication/scene-replacement-v2/index.json",
    )
    initialized = engine.initialize(
        {
            "project_id": "demo",
            "upstream_index_path": str(
                project / "artifacts/replication/index.json"
            ),
        }
    )
    proposed = engine.apply_scene_plan(
        {
            "project_id": "demo",
            "parent_replacement_revision": initialized["replacement_revision"],
            "submission": _plan(
                initialized["source_snapshot"]["fingerprints"]["snapshot"]
            ),
        }
    )
    assert proposed["package_status"] == "awaiting_human"
    scene = proposed["scene_plan"]["physical_scenes"][0]
    selected = engine.record_direction_selection(
        {
            "project_id": "demo",
            "parent_replacement_revision": proposed["replacement_revision"],
            "submission": {
                "scene_plan_id": proposed["scene_plan"]["scene_plan_id"],
                "scene_id": "scene_1",
                "presented_direction_ids": [
                    item["direction_id"] for item in scene["directions"]
                ],
                "directions_sha256": sha256_json(scene["directions"]),
                "selected_direction_id": "direction_a",
                "human_confirmation": {
                    "kind": "human",
                    "actor": "reviewer",
                    "reason": (
                        "The ordinary home-kitchen direction best matches the buyer"
                    ),
                },
            },
        }
    )
    return project, engine, selected


def _evidence(project: Path, state: dict) -> dict[str, str]:
    source = state["source_snapshot"]["anchors"][0]["image"]
    return {"path": source["path"], "sha256": sha256_file(project / source["path"])}


def _rules(
    rule_ids: set[str], evidence: dict[str, str], *, failed: str | None = None
) -> list[dict]:
    return [
        {
            "rule_id": rule_id,
            "status": "fail" if rule_id == failed else "pass",
            "reason": f"Evaluated {rule_id}",
            "evidence_refs": [deepcopy(evidence)],
        }
        for rule_id in sorted(rule_ids)
    ]


def _prepare_initial(
    project: Path, engine: SceneReplacementV2Engine, state: dict
) -> dict:
    prepared = engine.prepare_edit(
        {
            "project_id": "demo",
            "parent_replacement_revision": state["replacement_revision"],
            "submission": {
                "anchor_id": "anchor_1",
                "generation_mode": "initial",
                "creative_instructions": "Keep normal signs of everyday household use",
                "inspection_regions": [
                    {
                        "name": "product",
                        "bbox": [0, 0, 8, 8],
                        "reason": "Check all product details",
                    }
                ],
            },
        }
    )
    packet = prepared["generation_packet"]
    assert "720x1280" in packet["prompt"]
    assert "9:16" in packet["prompt"]
    assert "Do not use masks, compositing" in packet["prompt"]
    assert packet["prompt_sha256"] == hashlib.sha256(
        packet["prompt"].encode("utf-8")
    ).hexdigest()
    assert [item["role"] for item in packet["ordered_image_references"]] == [
        "source_frame"
    ]
    assert [item["role"] for item in packet["context_refs"]] == ["scene_design"]
    assert packet["output_requirement"] == {
        "width": 720,
        "height": 1280,
        "aspect_ratio": "9:16",
        "max_aspect_ratio_difference_percent": 5,
    }
    return prepared


def _dispatch_and_import(
    project: Path, engine: SceneReplacementV2Engine, prepared: dict
) -> tuple[dict, dict]:
    request_id = prepared["generation_packet"]["request_id"]
    evidence = _evidence(project, prepared)
    reviewed = engine.apply_review(
        {
            "project_id": "demo",
            "parent_replacement_revision": prepared["replacement_revision"],
            "submission": {
                "stage": "pre_generation",
                "scope_id": request_id,
                "anchor_id": "anchor_1",
                "status": "pass",
                "reviewer": {
                    "kind": "ai_coding_assistant",
                    "name": "test-agent",
                },
                "rules": _rules(PRE_GENERATION_RULES, evidence),
                "evidence_refs": [evidence],
                "issues": [],
            },
        }
    )
    dispatched = engine.mark_dispatched(
        {
            "project_id": "demo",
            "parent_replacement_revision": reviewed["replacement_revision"],
            "submission": {"request_id": request_id},
        }
    )
    generated_path = project / f"{request_id}.png"
    Image.new("RGB", (9, 16), (60, 90, 110)).save(generated_path, format="PNG")
    imported = engine.import_result(
        {
            "project_id": "demo",
            "parent_replacement_revision": dispatched["replacement_revision"],
            "submission": {
                "request_id": request_id,
                "success": True,
                "result_path": str(generated_path),
                "execution": {
                    "requested_capability": "imagegen2",
                    "actual_tool": "image_gen",
                    "model": "fixture",
                    "seed": None,
                    "cost_usd": None,
                },
            },
        }
    )
    task = next(iter(imported["logical_tasks"].values()))
    result = imported["results"][task["latest_result_id"]]
    assert result["normalization"]["method"] == (
        "lanczos_resize_to_target_within_5pct_aspect_ratio"
    )
    assert result["normalization"]["image"]["width"] == 720
    assert result["normalization"]["image"]["height"] == 1280
    return imported, result


def _failure_review(
    engine: SceneReplacementV2Engine,
    imported: dict,
    result: dict,
    *,
    failed_rule: str,
    severity: str,
    strategy: str,
) -> dict:
    evidence = _ref(result["comparison"])
    return engine.apply_review(
        {
            "project_id": "demo",
            "parent_replacement_revision": imported["replacement_revision"],
            "submission": {
                "stage": "generated_result",
                "scope_id": result["result_id"],
                "anchor_id": "anchor_1",
                "result_id": result["result_id"],
                "result_sha256": result["normalization"]["image"]["sha256"],
                "status": "fail",
                "reviewer": {
                    "kind": "ai_coding_assistant",
                    "name": "test-agent",
                },
                "rules": _rules(
                    GENERATED_RESULT_RULES, evidence, failed=failed_rule
                ),
                "evidence_refs": [evidence],
                "issues": [
                    {
                        "rule_id": failed_rule,
                        "anchor_id": "anchor_1",
                        "region": "upper-right background",
                        "severity": severity,
                        "evidence_refs": [evidence],
                        "observed_failure": "An implausible floating utensil remains",
                        "possible_cause": "Localized generation artifact",
                        "rework_action": "Remove only the floating utensil",
                        "affected_scope": "localized background region",
                        "expected_improvement": "Natural background; product unchanged",
                    }
                ],
                "rework_decision": {
                    "highest_severity": severity,
                    "strategy": strategy,
                    "reason": "Route by severity and hard-rule status",
                },
            },
        }
    )


def _adopt_and_approve_sample(
    engine: SceneReplacementV2Engine, imported: dict, result: dict
) -> dict:
    evidence = _ref(result["comparison"])
    adopted = engine.apply_review(
        {
            "project_id": "demo",
            "parent_replacement_revision": imported["replacement_revision"],
            "submission": {
                "stage": "generated_result",
                "scope_id": result["result_id"],
                "anchor_id": "anchor_1",
                "result_id": result["result_id"],
                "result_sha256": result["normalization"]["image"]["sha256"],
                "adopt_result_id": result["result_id"],
                "status": "pass",
                "reviewer": {
                    "kind": "ai_coding_assistant",
                    "name": "test-agent",
                },
                "rules": _rules(GENERATED_RESULT_RULES, evidence),
                "evidence_refs": [evidence],
                "issues": [],
            },
        }
    )
    return engine.apply_review(
        {
            "project_id": "demo",
            "parent_replacement_revision": adopted["replacement_revision"],
            "submission": {
                "stage": "sample",
                "scope_id": result["result_id"],
                "scene_id": "scene_1",
                "anchor_id": "anchor_1",
                "result_id": result["result_id"],
                "result_sha256": result["normalization"]["image"]["sha256"],
                "status": "pass",
                "reviewer": {"kind": "human", "name": "reviewer"},
                "rules": _rules(SAMPLE_RULES, evidence),
                "evidence_refs": [evidence],
                "issues": [],
                "human_confirmation": {
                    "kind": "human",
                    "actor": "reviewer",
                    "reason": "Approved the main direct-generation sample",
                },
            },
        }
    )


def test_contract_has_three_direction_gate_and_no_legacy_operations(
    v2_workflow,
) -> None:
    _, _, selected = v2_workflow
    from lib.replication_scene_replacement.engine_v2 import OPERATIONS

    assert "record_direction_selection" in OPERATIONS
    assert "prepare_global_review" in OPERATIONS
    assert "apply_protection" not in OPERATIONS
    assert "composite" not in OPERATIONS
    assert selected["scene_plan"]["status"] == "ready"
    assert selected["scene_plan"]["physical_scenes"][0]["selection_receipt_id"]


def test_plan_schema_rejects_two_directions() -> None:
    plan = _plan("a" * 64)
    plan["physical_scenes"][0]["directions"].pop()
    with pytest.raises(Exception):
        validate_phase2_document("scene_replacement_v2_plan", plan)


def test_presentation_map_orders_clips_and_preserves_keyframe_ownership() -> None:
    delivery = {
        "naming_version": "sequence-v1",
        "clips": [
            {
                "clip_id": "clip_b",
                "sequence": 2,
                "display_id": "S02",
                "keyframes": [
                    {
                        "anchor_id": "anchor_3",
                        "clip_id": "clip_b",
                        "sequence": 1,
                        "display_id": "S02_K01",
                    }
                ],
            },
            {
                "clip_id": "clip_a",
                "sequence": 1,
                "display_id": "S01",
                "keyframes": [
                    {
                        "anchor_id": "anchor_2",
                        "clip_id": "clip_a",
                        "sequence": 2,
                        "display_id": "S01_K02",
                    },
                    {
                        "anchor_id": "anchor_1",
                        "clip_id": "clip_a",
                        "sequence": 1,
                        "display_id": "S01_K01",
                    },
                ],
            },
        ],
    }
    mapping = presentation_map_from_delivery(delivery)
    assert [item["display_id"] for item in mapping] == [
        "S01_K01",
        "S01_K02",
        "S02_K01",
    ]
    assert [item["timeline_sequence"] for item in mapping] == [1, 2, 3]


def test_prepare_derives_prompt_target_and_references(v2_workflow) -> None:
    project, engine, selected = v2_workflow
    prepared = _prepare_initial(project, engine, selected)
    request = prepared["edit_requests"][prepared["generation_packet"]["request_id"]]
    scene = prepared["scene_plan"]["physical_scenes"][0]
    assert request["selection_receipt_id"] == scene["selection_receipt_id"]
    assert request["approved_target"]["id"] == "direction_a"
    with pytest.raises(SceneReplacementError, match="derives"):
        engine.prepare_edit(
            {
                "project_id": "demo",
                "parent_replacement_revision": prepared["replacement_revision"],
                "submission": {
                    "anchor_id": "anchor_1",
                    "generation_mode": "initial",
                    "prompt": "caller-controlled prompt",
                },
            }
        )


def test_direct_v2_state_upgrade_preserves_history_and_reopens_direction_gate(
    v2_workflow,
) -> None:
    _, engine, selected = v2_workflow
    legacy = deepcopy(selected["state"])
    legacy["schema_version"] = "2.0"
    legacy.pop("presentation_map")
    legacy.pop("scene_review_candidates")
    legacy.pop("global_review_candidate")
    legacy.pop("global_review_id")
    legacy["source_snapshot"]["schema_version"] = "1.0"
    legacy["source_snapshot"].pop("presentation_map")
    legacy["source_snapshot"]["fingerprints"].pop("presentation")

    engine._upgrade_state(legacy)

    assert legacy["schema_version"] == "2.1"
    assert legacy["presentation_map"][0]["display_id"] == "S01_K01"
    assert legacy["scene_plan"]["status"] == "awaiting_human"
    assert "selection_receipt_id" not in legacy["scene_plan"]["physical_scenes"][0]


def test_minor_non_hard_failure_uses_generated_image_local_adjustment(
    v2_workflow,
) -> None:
    project, engine, selected = v2_workflow
    prepared = _prepare_initial(project, engine, selected)
    imported, result = _dispatch_and_import(project, engine, prepared)
    failed = _failure_review(
        engine,
        imported,
        result,
        failed_rule="artifact_free",
        severity="minor",
        strategy="local_adjustment",
    )
    rework = engine.prepare_edit(
        {
            "project_id": "demo",
            "parent_replacement_revision": failed["replacement_revision"],
            "submission": {
                "anchor_id": "anchor_1",
                "generation_mode": "local_adjustment",
                "trigger_review_id": failed["reviews"][-1]["review_id"],
                "base_result_id": result["result_id"],
            },
        }
    )
    references = rework["generation_packet"]["ordered_image_references"]
    assert [(item["role"], item["usage"]) for item in references[:2]] == [
        ("generated_result", "edit_target"),
        ("source_frame", "invariant_reference"),
    ]


def test_hard_failure_regenerates_from_source_without_failed_image(
    v2_workflow,
) -> None:
    project, engine, selected = v2_workflow
    prepared = _prepare_initial(project, engine, selected)
    imported, result = _dispatch_and_import(project, engine, prepared)
    failed = _failure_review(
        engine,
        imported,
        result,
        failed_rule="subject_preservation",
        severity="minor",
        strategy="source_regeneration",
    )
    rework = engine.prepare_edit(
        {
            "project_id": "demo",
            "parent_replacement_revision": failed["replacement_revision"],
            "submission": {
                "anchor_id": "anchor_1",
                "generation_mode": "source_regeneration",
                "trigger_review_id": failed["reviews"][-1]["review_id"],
            },
        }
    )
    references = rework["generation_packet"]["ordered_image_references"]
    assert references[0]["role"] == "source_frame"
    assert all(item["role"] != "generated_result" for item in references)


def test_confirmed_generation_failure_retries_from_source_and_keeps_budget(
    v2_workflow,
) -> None:
    project, engine, selected = v2_workflow
    prepared = _prepare_initial(project, engine, selected)
    request_id = prepared["generation_packet"]["request_id"]
    evidence = _evidence(project, prepared)
    reviewed = engine.apply_review(
        {
            "project_id": "demo",
            "parent_replacement_revision": prepared["replacement_revision"],
            "submission": {
                "stage": "pre_generation",
                "scope_id": request_id,
                "anchor_id": "anchor_1",
                "status": "pass",
                "reviewer": {
                    "kind": "ai_coding_assistant",
                    "name": "test-agent",
                },
                "rules": _rules(PRE_GENERATION_RULES, evidence),
                "evidence_refs": [evidence],
                "issues": [],
            },
        }
    )
    dispatched = engine.mark_dispatched(
        {
            "project_id": "demo",
            "parent_replacement_revision": reviewed["replacement_revision"],
            "submission": {"request_id": request_id},
        }
    )
    failed = engine.import_result(
        {
            "project_id": "demo",
            "parent_replacement_revision": dispatched["replacement_revision"],
            "submission": {
                "request_id": request_id,
                "success": False,
                "error": "provider returned no image",
                "execution": {
                    "requested_capability": "imagegen2",
                    "actual_tool": "image_gen",
                },
            },
        }
    )
    task = next(iter(failed["logical_tasks"].values()))
    assert task["counted_dispatches"] == 1
    assert failed["edit_requests"][request_id]["execution_status"] == (
        "confirmed_failed"
    )
    retry = engine.prepare_edit(
        {
            "project_id": "demo",
            "parent_replacement_revision": failed["replacement_revision"],
            "submission": {
                "anchor_id": "anchor_1",
                "generation_mode": "source_regeneration",
                "trigger_execution_request_id": request_id,
            },
        }
    )
    assert retry["generation_packet"]["attempt_number"] == 2
    assert [
        item["role"] for item in retry["generation_packet"]["ordered_image_references"]
    ] == ["source_frame"]


def test_publish_requires_and_records_final_multiframe_human_approval(
    v2_workflow,
) -> None:
    project, engine, selected = v2_workflow
    prepared = _prepare_initial(project, engine, selected)
    imported, result = _dispatch_and_import(project, engine, prepared)
    sample = _adopt_and_approve_sample(engine, imported, result)

    with pytest.raises(SceneReplacementError, match="global"):
        engine.publish(
            {
                "project_id": "demo",
                "parent_replacement_revision": sample["replacement_revision"],
            }
        )

    scene_evidence = engine.prepare_scene_review(
        {
            "project_id": "demo",
            "parent_replacement_revision": sample["replacement_revision"],
            "submission": {"scene_id": "scene_1"},
        }
    )
    scene_candidate = scene_evidence["scene_review_candidates"]["scene_1"]
    scene_refs = [
        _ref(scene_candidate["comparison"]),
        deepcopy(scene_candidate["binding_manifest"]),
    ]
    grouped = engine.apply_review(
        {
            "project_id": "demo",
            "parent_replacement_revision": scene_evidence["replacement_revision"],
            "submission": {
                "stage": "group",
                "scope_id": scene_candidate["candidate_id"],
                "candidate_id": scene_candidate["candidate_id"],
                "candidate_fingerprint": scene_candidate[
                    "dependency_fingerprint"
                ],
                "scene_id": "scene_1",
                "anchor_ids": ["anchor_1"],
                "status": "pass",
                "reviewer": {"kind": "human", "name": "reviewer"},
                "rules": _rules(GROUP_RULES, scene_refs[0]),
                "evidence_refs": scene_refs,
                "issues": [],
                "human_confirmation": {
                    "kind": "human",
                    "actor": "reviewer",
                    "reason": "Approved the complete paired scene comparison",
                },
            },
        }
    )
    assert grouped["package_status"] == "in_progress"

    global_evidence = engine.prepare_global_review(
        {
            "project_id": "demo",
            "parent_replacement_revision": grouped["replacement_revision"],
            "submission": {},
        }
    )
    candidate = global_evidence["global_review_candidate"]
    assert candidate["item_count"] == 1
    assert candidate["ordered_display_ids"] == ["S01_K01"]
    assert (project / candidate["global_comparison"]["path"]).is_file()
    global_refs = [
        _ref(candidate["global_comparison"]),
        deepcopy(candidate["binding_manifest"]),
        deepcopy(candidate["accepted_assets_manifest"]),
    ]
    globally_approved = engine.apply_review(
        {
            "project_id": "demo",
            "parent_replacement_revision": global_evidence[
                "replacement_revision"
            ],
            "submission": {
                "stage": "global",
                "scope_id": candidate["candidate_id"],
                "candidate_id": candidate["candidate_id"],
                "candidate_fingerprint": candidate["dependency_fingerprint"],
                "comparison_sha256": candidate["global_comparison"]["sha256"],
                "anchor_ids": ["anchor_1"],
                "status": "pass",
                "reviewer": {"kind": "human", "name": "reviewer"},
                "rules": _rules(GLOBAL_RULES, global_refs[0]),
                "evidence_refs": global_refs,
                "issues": [],
                "human_confirmation": {
                    "kind": "human",
                    "actor": "reviewer",
                    "reason": "Approved the final total multi-frame comparison",
                },
            },
        }
    )
    assert globally_approved["package_status"] == "ready"
    published = engine.publish(
        {
            "project_id": "demo",
            "parent_replacement_revision": globally_approved[
                "replacement_revision"
            ],
        }
    )
    validated = engine.validate()
    assert published["package_status"] == "published"
    assert validated["validation_status"] == "passed"
    delivery = validated["delivery"]
    accepted = delivery["accepted_results"][0]
    assert accepted["display_id"] == "S01_K01"
    assert Path(accepted["image"]["path"]).name == "S01_K01.png"
    assert accepted["prompt"] == prepared["generation_packet"]["prompt"]
    assert accepted["attempt_history"][0]["prompt"] == accepted["prompt"]
    assert delivery["global_review_id"] == globally_approved["global_review_id"]
    assert delivery["global_comparison_ref"] == _ref(
        candidate["global_comparison"]
    )
