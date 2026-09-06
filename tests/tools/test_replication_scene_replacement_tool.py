from __future__ import annotations

from pathlib import Path

from tools.enhancement import replication_scene_replacement as tool_module
from tools.enhancement.replication_scene_replacement import (
    OPERATIONS,
    ReplicationSceneReplacement,
)
from tools.tool_registry import ToolRegistry


def test_contract_is_local_deterministic_and_has_no_generation_fallback() -> None:
    tool = ReplicationSceneReplacement()

    assert tool.resource_profile.vram_mb == 0
    assert tool.resource_profile.network_required is False
    assert tool.fallback_tools == []
    assert tool.agent_skills == []
    assert tool.supports["requested_capability"] == "imagegen2"
    assert tool.supports["session_tool"] == "image_gen"
    assert tool.input_schema["properties"]["operation"]["enum"] == list(OPERATIONS)
    assert "apply_protection" not in OPERATIONS
    assert "composite" not in OPERATIONS
    assert "supplementary_anchor_confirmations" in tool.idempotency_key_fields


def test_supplementary_confirmations_do_not_collide_in_idempotency_key() -> None:
    tool = ReplicationSceneReplacement()
    base = {
        "operation": "initialize",
        "project_id": "demo",
        "output_path": "/projects/demo/artifacts/replication/scene-replacement-v2/index.json",
        "upstream_index_path": "/projects/demo/artifacts/replication/index.json",
        "upstream_plan_revision": "r0002",
    }
    first = {
        **base,
        "supplementary_anchor_confirmations": [
            {
                "anchor_id": "anchor_2",
                "purpose": "rear contact edge",
                "gap": "primary view is occluded",
                "upstream_plan_revision": "r0002",
                "human_confirmation": {
                    "kind": "human",
                    "actor": "reviewer-a",
                    "reason": "approved difficult view",
                },
            }
        ],
    }
    second = {
        **base,
        "supplementary_anchor_confirmations": [
            {
                **first["supplementary_anchor_confirmations"][0],
                "human_confirmation": {
                    "kind": "human",
                    "actor": "reviewer-b",
                    "reason": "approved after reselection",
                },
            }
        ],
    }

    assert tool.idempotency_key(first) == tool.idempotency_key(first)
    assert tool.idempotency_key(first) != tool.idempotency_key(second)


def test_registry_discovers_scene_replacement_tool() -> None:
    registry = ToolRegistry()
    registry.discover("tools.enhancement")

    assert registry.get("replication_scene_replacement") is not None


def test_tool_requires_caller_created_workspace(tmp_path: Path, monkeypatch) -> None:
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr(tool_module, "PROJECTS_DIR", projects)

    result = ReplicationSceneReplacement().execute(
        {
            "operation": "initialize",
            "project_id": "missing-project",
            "output_path": str(
                projects
                / "missing-project"
                / "artifacts"
                / "replication"
                / "scene-replacement-v2"
                / "index.json"
            ),
            "upstream_index_path": str(tmp_path / "upstream.json"),
        }
    )

    assert result.success is False
    assert "must be created" in (result.error or "")


def test_tool_rejects_project_leaf_symlink_without_external_writes(
    tmp_path: Path, monkeypatch
) -> None:
    projects = tmp_path / "projects"
    external = tmp_path / "external-project"
    projects.mkdir()
    external.mkdir()
    (projects / "demo").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(tool_module, "PROJECTS_DIR", projects)

    result = ReplicationSceneReplacement().execute(
        {
            "operation": "initialize",
            "project_id": "demo",
            "output_path": str(
                projects
                / "demo"
                / "artifacts"
                / "replication"
                / "scene-replacement-v2"
                / "index.json"
            ),
            "upstream_index_path": str(
                projects / "demo" / "artifacts" / "replication" / "index.json"
            ),
        }
    )

    assert result.success is False
    assert "leaf must not be a symlink" in (result.error or "")
    assert list(external.iterdir()) == []


def test_tool_rejects_output_escape_without_external_writes(
    tmp_path: Path, monkeypatch
) -> None:
    projects = tmp_path / "projects"
    project = projects / "demo"
    external = tmp_path / "external-output"
    project.mkdir(parents=True)
    external.mkdir()
    monkeypatch.setattr(tool_module, "PROJECTS_DIR", projects)

    result = ReplicationSceneReplacement().execute(
        {
            "operation": "initialize",
            "project_id": "demo",
            "output_path": str(external / "index.json"),
            "upstream_index_path": str(
                project / "artifacts" / "replication" / "index.json"
            ),
        }
    )

    assert result.success is False
    assert "output_path must stay inside" in (result.error or "")
    assert list(external.iterdir()) == []


def test_tool_forwards_to_engine_and_preserves_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    projects = tmp_path / "projects"
    project = projects / "demo"
    project.mkdir(parents=True)
    output = project / "artifacts" / "replication" / "scene-replacement-v2" / "index.json"
    captured: dict[str, object] = {}

    class FakeEngine:
        def __init__(self, project_dir: Path, output_path: Path):
            captured["project_dir"] = project_dir
            captured["output_path"] = output_path

        @staticmethod
        def validate_project_id(project_id: str) -> None:
            assert project_id == "demo"

        def relative(self, path: Path) -> str:
            return path.relative_to(project).as_posix()

        def execute(self, operation: str, inputs: dict) -> dict:
            captured["operation"] = operation
            captured["inputs"] = inputs
            return {
                "replacement_revision": "r0001",
                "package_status": "source_locked",
                "validation_status": "passed",
                "artifacts": [str(output), str(project / "snapshot.json")],
            }

    monkeypatch.setattr(tool_module, "PROJECTS_DIR", projects)
    monkeypatch.setattr(tool_module, "SceneReplacementEngine", FakeEngine)
    inputs = {
        "operation": "initialize",
        "project_id": "demo",
        "output_path": str(output),
        "upstream_index_path": str(project / "upstream.json"),
    }

    result = ReplicationSceneReplacement().execute(inputs)

    assert result.success is True
    assert result.data["index_path"].endswith("scene-replacement-v2/index.json")
    assert result.data["replacement_revision"] == "r0001"
    assert result.artifacts == [str(output), str(project / "snapshot.json")]
    assert captured["operation"] == "initialize"
    assert captured["inputs"] == inputs


def test_tool_returns_structured_domain_error(tmp_path: Path, monkeypatch) -> None:
    projects = tmp_path / "projects"
    project = projects / "demo"
    project.mkdir(parents=True)

    class FailingEngine:
        def __init__(self, project_dir: Path, output_path: Path):
            pass

        @staticmethod
        def validate_project_id(project_id: str) -> None:
            pass

        def execute(self, operation: str, inputs: dict) -> dict:
            raise tool_module.SceneReplacementError("upstream delivery is not ready")

    monkeypatch.setattr(tool_module, "PROJECTS_DIR", projects)
    monkeypatch.setattr(tool_module, "SceneReplacementEngine", FailingEngine)

    result = ReplicationSceneReplacement().execute(
        {
            "operation": "validate",
            "project_id": "demo",
            "output_path": str(project / "artifacts" / "replacement.json"),
        }
    )

    assert result.success is False
    assert result.data["package_status"] == "blocked"
    assert result.data["validation_status"] == "failed"
    assert result.data["issues"] == ["upstream delivery is not ready"]
