from __future__ import annotations

from pathlib import Path

from lib.checkpoint import _stage_requires_approval, get_pipeline_stages
from lib.pipeline_loader import get_required_tools, get_stage_order, get_stage_sub_stages, load_pipeline
from lib.replication_scene_replacement.engine_v2 import (
    GENERATED_RESULT_RULES,
    GLOBAL_RULES,
    GROUP_RULES,
    HARD_RESULT_RULES,
    PRE_GENERATION_RULES,
    SAMPLE_RULES,
)


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = "tiktok-short-video-replication"


def test_phase2_pipeline_stage_and_gate_contract() -> None:
    manifest = load_pipeline(PIPELINE)
    assert get_stage_order(manifest) == ["source_lock", "scene_replacement", "delivery"]
    assert get_pipeline_stages(PIPELINE) == ["source_lock", "scene_replacement", "delivery"]
    assert {
        stage: _stage_requires_approval(PIPELINE, stage)
        for stage in get_stage_order(manifest)
    } == {"source_lock": False, "scene_replacement": True, "delivery": False}

    units = get_stage_sub_stages(manifest, "scene_replacement")
    assert [unit["name"] for unit in units] == [
        "scene_analysis", "direction_selection", "main_reference",
        "difficult_view", "group_expansion", "group_approval", "global_approval",
    ]
    gated = {unit["name"] for unit in units if unit["human_approval_default"]}
    assert gated == {
        "direction_selection", "main_reference", "group_approval", "global_approval"
    }


def test_pipeline_declares_session_imagegen_without_fallback() -> None:
    manifest = load_pipeline(PIPELINE)
    source_lock = manifest["stages"][0]
    assert source_lock["required_tools"] == ["replication_scene_replacement"]
    assert source_lock["optional_tools"] == ["replication_preprocess"]
    assert "replication_preprocess" not in get_required_tools(manifest)
    capability = manifest["metadata"]["external_agent_capabilities"][0]
    assert capability == {
        "requested_name": "imagegen2",
        "session_tool": "image_gen",
        "required": True,
        "fallback_allowed": False,
        "note": "The session tool is called by the Agent, never by the Python tool.",
    }
    assert "image_gen" not in {
        tool
        for stage in manifest["stages"]
        for tool in stage.get("tools_available", [])
    }


def test_pipeline_skill_references_exist() -> None:
    manifest = load_pipeline(PIPELINE)
    references = set(manifest["required_skills"])
    references.add(manifest["orchestration"]["skill"])
    references.update(stage["skill"] for stage in manifest["stages"])
    for reference in references:
        assert (ROOT / "skills" / f"{reference}.md").is_file(), reference


def test_phase2_pipeline_and_skills_are_in_distribution_metadata() -> None:
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    manifest_text = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    required = (
        "pipeline_defs/tiktok-short-video-replication.yaml",
        "skills/meta/replication-scene-replacement-review.md",
        "skills/pipelines/tiktok-short-video-replication/delivery-director.md",
        "skills/pipelines/tiktok-short-video-replication/executive-producer.md",
        "skills/pipelines/tiktok-short-video-replication/scene-replacement-director.md",
        "skills/pipelines/tiktok-short-video-replication/source-lock-director.md",
    )
    for relative_path in required:
        assert relative_path in setup_text
    assert "include pipeline_defs/tiktok-short-video-replication.yaml" in manifest_text
    assert "include skills/meta/replication-scene-replacement-review.md" in manifest_text
    assert (
        "recursive-include skills/pipelines/tiktok-short-video-replication *.md"
        in manifest_text
    )


def test_phase2_director_keeps_agent_python_boundary_explicit() -> None:
    text = (ROOT / "skills/pipelines/tiktok-short-video-replication/scene-replacement-director.md").read_text(
        encoding="utf-8"
    )
    assert "imagegen2" in text
    assert "image_gen" in text
    assert "Do not invoke" in " ".join(text.split())
    assert "initial call plus two" in text


def test_phase2_contract_requires_720x1280_with_five_percent_ratio_tolerance() -> None:
    paths = (
        "pipeline_defs/tiktok-short-video-replication.yaml",
        "skills/pipelines/tiktok-short-video-replication/executive-producer.md",
        "skills/pipelines/tiktok-short-video-replication/scene-replacement-director.md",
        "skills/meta/replication-scene-replacement-review.md",
        "docs-dev/plans/tiktok-short-video-replication-phase2-plan.md",
        "docs-dev/prds/tiktok-short-video-replication-phase2-prd.md",
    )
    for relative_path in paths:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        compact = text.replace(" ", "")
        assert "720x1280" in compact
        assert "5%" in compact


def test_phase2_docs_require_pre_generation_review_before_dispatch() -> None:
    paths = (
        "skills/pipelines/tiktok-short-video-replication/scene-replacement-director.md",
        "docs-dev/plans/tiktok-short-video-replication-phase2-plan.md",
        "docs-dev/prds/tiktok-short-video-replication-phase2-prd.md",
    )
    ordered_steps = (
        "prepare_edit",
        "apply_review(pre_generation)",
        "mark_dispatched",
        "image_gen",
    )

    for relative_path in paths:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        cursor = -1
        for step in ordered_steps:
            cursor = text.find(step, cursor + 1)
            assert cursor >= 0, f"{relative_path} must order {ordered_steps}"


def test_phase2_directors_checkpoint_engine_wrappers_without_reconstruction() -> None:
    expected = {
        "source-lock-director.md": "replication_source_snapshot",
        "scene-replacement-director.md": "scene_replacement_package",
        "delivery-director.md": "scene_replacement_delivery",
    }
    director_root = ROOT / "skills/pipelines/tiktok-short-video-replication"

    for filename, artifact_name in expected.items():
        text = (director_root / filename).read_text(encoding="utf-8")
        assert 'checkpoint_artifacts = data["checkpoint_artifacts"]' in text
        assert f'checkpoint_artifacts["{artifact_name}"]' in text
        assert "Do not hand-build" in text


def test_supplementary_anchor_exception_is_narrow_and_human_confirmed() -> None:
    text = (ROOT / "skills/meta/replication-keyframe-selection.md").read_text(encoding="utf-8")
    assert "Phase 2 supplementary-edit-anchor exception" in text
    assert "Explicit human confirmation" in text
    assert "minimum" in text.lower()


def test_review_skill_names_every_engine_required_rule_id() -> None:
    text = (ROOT / "skills/meta/replication-scene-replacement-review.md").read_text(
        encoding="utf-8"
    )
    required = (
        PRE_GENERATION_RULES | GENERATED_RESULT_RULES |
        HARD_RESULT_RULES | SAMPLE_RULES | GROUP_RULES | GLOBAL_RULES
    )
    missing = {rule_id for rule_id in required if f"`{rule_id}`" not in text}
    assert not missing
