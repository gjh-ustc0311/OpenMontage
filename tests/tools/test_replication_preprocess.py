from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.analysis import replication_preprocess as tool_module
from tools.analysis.replication_preprocess import ReplicationPreprocess


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)


def _make_fixture(path: Path, *, with_audio: bool = False) -> None:
    command = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=160x120:r=10:d=3.5",
        "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=10:d=3.5",
    ]
    if with_audio:
        command += [
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=7",
        ]
    command += [
        "-filter_complex", "[0:v:0][1:v:0]concat=n=2:v=1:a=0[v]",
        "-map", "[v]",
    ]
    if with_audio:
        command += ["-map", "2:a:0", "-c:a", "aac", "-b:a", "96k"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps({
            "scene_detection": {
                "initial_threshold": 20,
                "minimum_threshold": 20,
                "threshold_step": 5,
            },
            "export": {"preset": "ultrafast"},
        }),
        encoding="utf-8",
    )


def _submission(request: dict) -> dict:
    return {
        "schema_version": "1.0",
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "parent_plan_revision": request["parent_plan_revision"],
        "review_protocol_version": request["review_protocol_version"],
        "reviewer": {"kind": "ai_coding_assistant", "name": "pytest-reviewer"},
        "decisions": [
            {
                "review_item_id": item["review_item_id"],
                "boundary_id": item["boundary_id"],
                "relationship": "different_scene",
                "confidence": "high",
                "evidence_refs": ["left_edge", "right_edge", "evidence_board"],
                "rationale": "The full-frame color and visual context change at the cut.",
            }
            for item in request["items"]
        ],
    }


def test_tool_requires_caller_created_workspace(tmp_path: Path, monkeypatch) -> None:
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr(tool_module, "PROJECTS_DIR", projects)

    result = ReplicationPreprocess().execute({
        "operation": "plan",
        "project_id": "missing-project",
        "source_path": str(tmp_path / "missing.mp4"),
        "output_path": str(projects / "missing-project" / "replication.json"),
    })

    assert result.success is False
    assert "must be created" in (result.error or "")


@pytest.mark.parametrize("with_audio", [False, True])
def test_plan_review_export_validate_round_trip(
    tmp_path: Path, monkeypatch, with_audio: bool
) -> None:
    pytest.importorskip("av")
    pytest.importorskip("scenedetect")
    pytest.importorskip("cv2")

    projects = tmp_path / "projects"
    project = projects / "demo"
    project.mkdir(parents=True)
    source = tmp_path / "source.mp4"
    config = tmp_path / "replication.json"
    index = project / "artifacts" / "replication" / "index.json"
    _make_fixture(source, with_audio=with_audio)
    _write_config(config)
    monkeypatch.setattr(tool_module, "PROJECTS_DIR", projects)
    tool = ReplicationPreprocess()
    base = {
        "project_id": "demo",
        "source_path": str(source),
        "config_path": str(config),
        "output_path": str(index),
    }

    first = tool.execute({"operation": "plan", **base})
    assert first.success is True, first.error
    assert first.data["plan_status"] == "needs_review"
    assert first.data["review_status"] == "pending_agent"
    assert first.data["plan_revision"] == "r0001"

    repeated = tool.execute({"operation": "plan", **base})
    assert repeated.success is True, repeated.error
    assert repeated.data["plan_revision"] == "r0001"
    assert len(list((project / "artifacts" / "replication" / "revisions").iterdir())) == 1

    request_path = project / first.data["index"]["next_action"]["request_path"]
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["items"]
    for item in request["items"]:
        assert set(item["evidence"]) >= {
            "left_observation", "left_context", "left_edge", "right_edge",
            "right_context", "right_observation", "left_background",
            "right_background", "evidence_board",
        }
        for evidence in item["evidence"].values():
            assert (project / evidence["path"]).is_file()

    if not with_audio:
        evidence_path = project / request["items"][0]["evidence"]["left_edge"]["path"]
        original_evidence = evidence_path.read_bytes()
        evidence_path.write_bytes(original_evidence + b"tampered")
        rejected = tool.execute({
            "operation": "plan",
            **base,
            "parent_plan_revision": "r0001",
            "review_submission": _submission(request),
        })
        assert rejected.success is False
        assert "evidence hash mismatch" in (rejected.error or "").lower()
        evidence_path.write_bytes(original_evidence)

    reviewed = tool.execute({
        "operation": "plan",
        **base,
        "parent_plan_revision": "r0001",
        "review_submission": _submission(request),
    })
    assert reviewed.success is True, reviewed.error
    assert reviewed.data["plan_status"] == "ready"
    assert reviewed.data["review_status"] == "accepted"
    assert reviewed.data["plan_revision"] == "r0002"
    assert len(reviewed.data["generation_clips"]) == 2

    exported = tool.execute({
        "operation": "export",
        "project_id": "demo",
        "output_path": str(index),
    })
    assert exported.success is True, exported.error
    assert exported.data["export_status"] == "validated"
    assert len(exported.data["clip_paths"]) == 2
    assert all((project / path).is_file() for path in exported.data["clip_paths"])
    report = json.loads(
        (project / exported.data["export_report_path"]).read_text(encoding="utf-8")
    )
    assert all(item["audio_streams"] == int(with_audio) for item in report["clips"])

    repeated_export = tool.execute({
        "operation": "export",
        "project_id": "demo",
        "output_path": str(index),
    })
    assert repeated_export.success is True, repeated_export.error
    assert repeated_export.data["clip_paths"] == exported.data["clip_paths"]

    validated = tool.execute({
        "operation": "validate",
        "project_id": "demo",
        "output_path": str(index),
    })
    assert validated.success is True, validated.error
    assert validated.data["validation_status"] == "passed"
    assert validated.data["issues"] == []

    if not with_audio:
        state_path = project / "artifacts" / "replication" / "revisions" / "r0002" / "plan_state.json"
        original_state = state_path.read_bytes()
        state_path.write_bytes(original_state + b" ")
        tampered = tool.execute({
            "operation": "validate",
            "project_id": "demo",
            "output_path": str(index),
        })
        assert tampered.success is False
        assert "manifest_hash_mismatch:plan_state.json" in tampered.data["issues"]


def test_two_uncertain_rounds_escalate_then_manual_override_resolves(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("av")
    pytest.importorskip("scenedetect")
    pytest.importorskip("cv2")
    projects = tmp_path / "projects"
    project = projects / "uncertain"
    project.mkdir(parents=True)
    source = tmp_path / "source.mp4"
    config = tmp_path / "replication.json"
    index = project / "artifacts" / "replication" / "index.json"
    _make_fixture(source)
    _write_config(config)
    monkeypatch.setattr(tool_module, "PROJECTS_DIR", projects)
    tool = ReplicationPreprocess()
    base = {
        "project_id": "uncertain",
        "source_path": str(source),
        "config_path": str(config),
        "output_path": str(index),
    }

    first = tool.execute({"operation": "plan", **base})
    request_path = project / first.data["index"]["next_action"]["request_path"]
    request = json.loads(request_path.read_text(encoding="utf-8"))

    def uncertain_submission(value: dict) -> dict:
        submission = _submission(value)
        for decision in submission["decisions"]:
            decision.update({
                "relationship": "uncertain",
                "confidence": "low",
                "rationale": "The available views do not establish spatial continuity.",
            })
        return submission

    second = tool.execute({
        "operation": "plan",
        **base,
        "parent_plan_revision": "r0001",
        "review_submission": uncertain_submission(request),
    })
    assert second.success is True, second.error
    assert second.data["review_status"] == "needs_more_evidence"
    request2_path = project / second.data["index"]["next_action"]["request_path"]
    request2 = json.loads(request2_path.read_text(encoding="utf-8"))
    assert request2["review_round"] == 2

    third = tool.execute({
        "operation": "plan",
        **base,
        "parent_plan_revision": "r0002",
        "review_submission": uncertain_submission(request2),
    })
    assert third.success is True, third.error
    assert third.data["plan_status"] == "needs_review"
    assert third.data["review_status"] == "needs_human"
    assert third.data["generation_clips"] == []

    boundary_id = third.data["boundaries"][0]["boundary_id"]
    resolved = tool.execute({
        "operation": "plan",
        **base,
        "parent_plan_revision": "r0003",
        "manual_overrides": [{
            "action": "force_hard",
            "boundary_id": boundary_id,
            "actor": "human-reviewer",
            "reason": "The locations are visibly different on expanded evidence.",
            "parent_plan_revision": "r0003",
            "config_fingerprint": third.data["config"]["config_fingerprint"],
        }],
    })
    assert resolved.success is True, resolved.error
    assert resolved.data["plan_status"] == "ready"
    assert resolved.data["review_status"] == "accepted"


def test_output_path_cannot_escape_project_workspace(tmp_path: Path, monkeypatch) -> None:
    projects = tmp_path / "projects"
    project = projects / "safe"
    project.mkdir(parents=True)
    monkeypatch.setattr(tool_module, "PROJECTS_DIR", projects)

    result = ReplicationPreprocess().execute({
        "operation": "plan",
        "project_id": "safe",
        "source_path": str(tmp_path / "source.mp4"),
        "output_path": str(tmp_path / "outside.json"),
    })

    assert result.success is False
    assert result.data["plan_status"] == "blocked"
    assert "under project workspace" in (result.error or "")


def test_invalid_config_is_rejected_before_media_decode(tmp_path: Path, monkeypatch) -> None:
    projects = tmp_path / "projects"
    project = projects / "invalid-config"
    project.mkdir(parents=True)
    source = tmp_path / "not-a-video.mp4"
    source.write_bytes(b"not a decodable video")
    config = tmp_path / "invalid.yaml"
    config.write_text("scene_detection:\n  unknown_option: 1\n", encoding="utf-8")
    monkeypatch.setattr(tool_module, "PROJECTS_DIR", projects)

    result = ReplicationPreprocess().execute({
        "operation": "run",
        "project_id": "invalid-config",
        "source_path": str(source),
        "config_path": str(config),
        "output_path": str(project / "artifacts" / "replication" / "index.json"),
    })

    assert result.success is False
    assert "unknown_option" in (result.error or "")
    assert "ffprobe" not in (result.error or "").lower()


def test_run_reuses_stages_and_keeps_export_variants(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("av")
    pytest.importorskip("scenedetect")
    pytest.importorskip("cv2")
    projects = tmp_path / "projects"
    project = projects / "stage-reuse"
    project.mkdir(parents=True)
    source = tmp_path / "source.mp4"
    config_path = tmp_path / "replication.json"
    index = project / "artifacts" / "replication" / "index.json"
    _make_fixture(source)
    _write_config(config_path)
    monkeypatch.setattr(tool_module, "PROJECTS_DIR", projects)
    tool = ReplicationPreprocess()
    base = {
        "project_id": "stage-reuse",
        "source_path": str(source),
        "config_path": str(config_path),
        "output_path": str(index),
    }

    first = tool.execute({"operation": "run", **base})
    request = json.loads(
        (project / first.data["index"]["next_action"]["request_path"]).read_text(
            encoding="utf-8"
        )
    )
    ready = tool.execute({
        "operation": "run",
        **base,
        "parent_plan_revision": first.data["plan_revision"],
        "review_submission": _submission(request),
    })
    assert ready.success is True, ready.error
    assert ready.data["plan_revision"] == "r0002"
    assert set(ready.data["executed_stages"]) >= {"review", "planning", "export"}
    first_export = ready.data["export_report_path"]

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload["export"]["crf"] = 22
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    export_only = tool.execute({"operation": "run", **base})

    assert export_only.success is True, export_only.error
    assert export_only.data["plan_revision"] == "r0002"
    assert export_only.data["executed_stages"] == ["export"]
    assert set(export_only.data["reused_stages"]) >= {"analysis", "review", "planning"}
    assert export_only.data["invalidated_stages"] == ["export"]
    assert export_only.data["export_report_path"] != first_export
    assert (project / first_export).is_file()
    assert (project / export_only.data["export_report_path"]).is_file()

    config_payload["regroup"] = {"max_atomic_segments": 2}
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    regrouped = tool.execute({"operation": "run", **base})

    assert regrouped.success is True, regrouped.error
    assert regrouped.data["plan_revision"] == "r0003"
    assert "planning" in regrouped.data["executed_stages"]
    assert set(regrouped.data["reused_stages"]) >= {"analysis", "review", "export"}
    assert regrouped.data["clip_paths"] == export_only.data["clip_paths"]
    assert regrouped.data["invalidated_stages"] == ["planning", "export"]

    old_state = json.loads(
        (
            project / "artifacts" / "replication" / "revisions" / "r0003" / "plan_state.json"
        ).read_text(encoding="utf-8")
    )
    config_payload["review"] = {"evidence_jpeg_quality": 91}
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    review_changed = tool.execute({"operation": "run", **base})

    assert review_changed.success is True, review_changed.error
    assert review_changed.data["plan_revision"] == "r0004"
    assert review_changed.data["plan_status"] == "needs_review"
    assert review_changed.data["executed_stages"] == ["review"]
    assert review_changed.data["reused_stages"] == ["analysis"]
    assert review_changed.data["invalidated_stages"] == ["review", "planning", "export"]
    new_state = json.loads(
        (
            project / "artifacts" / "replication" / "revisions" / "r0004" / "plan_state.json"
        ).read_text(encoding="utf-8")
    )
    assert [item["boundary_id"] for item in new_state["boundaries"]] == [
        item["boundary_id"] for item in old_state["boundaries"]
    ]

    config_payload["scene_detection"]["initial_threshold"] = 25
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    analysis_changed = tool.execute({"operation": "run", **base})

    assert analysis_changed.success is True, analysis_changed.error
    assert analysis_changed.data["plan_revision"] == "r0005"
    assert "analysis" in analysis_changed.data["executed_stages"]
    assert analysis_changed.data["reused_stages"] == []
    assert analysis_changed.data["invalidated_stages"] == [
        "analysis", "review", "planning", "export"
    ]
