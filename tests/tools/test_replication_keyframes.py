from __future__ import annotations

import json
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest

from lib.replication_preprocess import engine as engine_module
from lib.replication_preprocess.delivery import build_manifest
from lib.replication_preprocess.models import time_point
from lib.replication_preprocess.storage import sha256_file
from tools.analysis import replication_preprocess as tool_module
from tools.analysis.replication_preprocess import ReplicationPreprocess
from tests.tools.test_replication_preprocess import _make_fixture, _write_config, _submission


def document(project, result, filename):
    entry = result.data["index"]["manifest_index"][filename]
    return json.loads((project / entry["path"]).read_text())


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    pytest.importorskip("av")
    pytest.importorskip("cv2")
    pytest.importorskip("scenedetect")
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    projects = tmp_path / "projects"
    project = projects / "selection"
    project.mkdir(parents=True)
    source = tmp_path / "source.mp4"
    config = tmp_path / "config.json"
    _make_fixture(source, with_audio=True)
    _write_config(config)
    monkeypatch.setattr(tool_module, "PROJECTS_DIR", projects)
    base = dict(project_id="selection", source_path=str(source), config_path=str(config),
                output_path=str(project / "artifacts/replication/index.json"))
    tool = ReplicationPreprocess()
    first = tool.execute({"operation": "plan", **base})
    assert first.success, first.error
    return project, tool, base, first


def selection_submission(request, *, position=-1, supplementary=False, status="ready"):
    decisions = []
    for item in request["items"]:
        candidates = item["candidates"]
        primary = candidates[position]["candidate_id"]
        decision = {
            "segment_id": item["segment_id"], "status": status,
            "rationale": "Fixture: later source state is visible; inspect its actual time.",
            "evidence_refs": [c["candidate_id"] for c in candidates],
            "information_gaps": [] if status == "ready" else ["Need more detail"],
        }
        if status == "ready":
            decision.update(primary_candidate_id=primary, quality_override_reason="Human explicitly accepts the synthetic solid-color fixture.")
            if supplementary:
                other = next(c for c in candidates if c["candidate_id"] != primary)
                decision["supplementary_anchors"] = [{"candidate_id": other["candidate_id"], "purpose": "Earlier product state"}]
        decisions.append(decision)
    return {"schema_version": "1.0", "protocol_version": request["protocol_version"],
            "request_id": request["request_id"], "request_sha256": request["request_sha256"],
            "parent_plan_revision": request["parent_plan_revision"],
            "reviewer": {"kind": "human", "name": "fixture-reviewer"}, "decisions": decisions}


def test_delivery_names_sort_and_preserve_clip_keyframe_ownership():
    base = {"num": 1, "den": 1000}
    clips = []
    report_clips = []
    for index in reversed(range(100)):
        start = index * 1000
        clip_id = f"g-{index:03d}"
        anchor_id = f"a-{index:03d}"
        clips.append({
            "clip_id": clip_id,
            "start": {"pts": start, "time_base": base, "seconds": str(index)},
            "end": {"pts": start + 1000, "time_base": base, "seconds": str(index + 1)},
            "duration_s": "1",
            "atomic_segment_ids": [f"s-{index:03d}"],
            "keyframes": [{
                "anchor_id": anchor_id,
                "atomic_segment_id": f"s-{index:03d}",
                "role": "primary_representative",
                "source_time": {"pts": start + 500, "time_base": base, "seconds": f"{index}.5"},
                "atomic_segment_offset": {"pts": 500, "time_base": base, "seconds": "0.5"},
                "generation_clip_offset": {"pts": 500, "time_base": base, "seconds": "0.5"},
                "path": f"assets/images/{anchor_id}.png",
                "sha256": "a" * 64,
            }],
        })
        report_clips.append({
            "clip_id": clip_id, "path": f"assets/video/{clip_id}.mp4",
            "sha256": "b" * 64, "status": "validated", "full_decode": "passed",
        })
    state = {
        "project_id": "fixture", "plan_revision": "r0001", "plan_status": "ready",
        "anchor_status": "ready", "source": {"file_sha256": "c" * 64},
        "generation_clips": clips,
    }
    index = {
        "manifest_index": {"plan_state.json": {"path": "artifacts/replication/revisions/r0001/plan_state.json", "sha256": "d" * 64}},
        "export": {"status": "validated", "report_path": "artifacts/replication/exports/report.json", "report_sha256": "e" * 64},
    }
    manifest = build_manifest(state, index, {"status": "validated", "clips": report_clips})
    assert [manifest["clips"][i]["display_id"] for i in (0, 8, 9, 99)] == ["S001", "S009", "S010", "S100"]
    assert manifest["clips"][0]["keyframes"][0]["display_id"] == "S001_K01"
    assert manifest["clips"][0]["keyframes"][0]["clip_id"] == manifest["clips"][0]["clip_id"]


@pytest.mark.parametrize("selection_first", [False, True])
def test_reviews_are_independent_and_anchors_keep_real_time(workflow, selection_first):
    project, tool, base, first = workflow
    boundary_request = document(project, first, "boundary_review_request.json")
    keyframe_request = document(project, first, "keyframe_selection_request.json")
    assert first.data["anchor_status"] == "pending_agent"
    assert len(first.data["next_actions"]) == 2
    actions = [("keyframe_submission", selection_submission(keyframe_request, supplementary=True)),
               ("review_submission", _submission(boundary_request))]
    if not selection_first:
        actions.reverse()
    current = first
    for field, submission in actions:
        current = tool.execute({"operation": "plan", **base, "parent_plan_revision": current.data["plan_revision"], field: submission})
        assert current.success, current.error
    assert current.data["plan_status"] == "ready"
    assert current.data["anchor_status"] == "ready"
    for clip in current.data["generation_clips"]:
        assert len(clip["keyframes"]) == 2
        primary = next(a for a in clip["keyframes"] if a["role"] == "primary_representative")
        assert Fraction(primary["generation_clip_offset_s"]) > 1
        assert primary["generation_clip_offset"]["pts"] == primary["pts"] - clip["start"]["pts"]
        assert primary["path"].endswith(".png")
    assert first.data["atomic_segments"][0]["start"] == current.data["atomic_segments"][0]["start"]
    assert current.data["next_actions"] == []


def test_pending_anchors_export_and_reselection_reuses_video(workflow, monkeypatch):
    project, tool, base, first = workflow
    boundary = document(project, first, "boundary_review_request.json")
    exported = tool.execute({"operation": "run", **base, "parent_plan_revision": first.data["plan_revision"], "review_submission": _submission(boundary)})
    assert exported.success, exported.error
    assert exported.data["anchor_status"] == "pending_agent"
    paths = exported.data["clip_paths"]
    calls = []
    real_run = engine_module.subprocess.run
    def spy(command, *args, **kwargs):
        if "-c:v" in command:
            calls.append(command)
        return real_run(command, *args, **kwargs)
    monkeypatch.setattr(engine_module.subprocess, "run", spy)
    request = document(project, exported, "keyframe_selection_request.json")
    selected = tool.execute({"operation": "run", **base, "parent_plan_revision": exported.data["plan_revision"], "keyframe_submission": selection_submission(request)})
    assert selected.success, selected.error
    assert selected.data["clip_paths"] == paths
    assert selected.data["anchor_status"] == "ready"
    assert selected.data["delivery_status"] == "ready"
    selected_delivery_paths = selected.data["delivery_keyframe_paths"]
    assert not calls
    sid = request["items"][0]["segment_id"]
    reselect = tool.execute({"operation": "run", **base, "parent_plan_revision": selected.data["plan_revision"], "keyframe_review_segment_ids": [sid]})
    assert reselect.success, reselect.error
    assert reselect.data["delivery_status"] == "pending"
    request2 = document(project, reselect, "keyframe_selection_request.json")
    assert len(request2["items"]) == 1
    resubmitted = tool.execute({"operation": "run", **base, "parent_plan_revision": reselect.data["plan_revision"], "keyframe_submission": selection_submission(request2, position=0)})
    assert resubmitted.success, resubmitted.error
    assert resubmitted.data["clip_paths"] == paths
    assert resubmitted.data["delivery_status"] == "ready"
    assert resubmitted.data["delivery_keyframe_paths"] != selected_delivery_paths
    assert not calls
    state = document(project, resubmitted, "plan_state.json")
    assert state["anchor_changes"][0]["previous_revision"] == selected.data["plan_revision"]


def test_run_publishes_and_repairs_readable_delivery(workflow):
    project, tool, base, first = workflow
    boundary = document(project, first, "boundary_review_request.json")
    boundary_ready = tool.execute({
        "operation": "plan", **base,
        "parent_plan_revision": first.data["plan_revision"],
        "review_submission": _submission(boundary),
    })
    assert boundary_ready.success, boundary_ready.error
    keyframes = document(project, boundary_ready, "keyframe_selection_request.json")
    ready = tool.execute({
        "operation": "run", **base,
        "parent_plan_revision": boundary_ready.data["plan_revision"],
        "keyframe_submission": selection_submission(keyframes),
    })
    assert ready.success, ready.error
    assert ready.data["delivery_status"] == "ready"
    assert [Path(path).name for path in ready.data["delivery_clip_paths"]] == ["S01.mp4", "S02.mp4"]
    assert [Path(path).name for path in ready.data["delivery_keyframe_paths"]] == ["S01_K01.png", "S02_K01.png"]
    manifest_path = project / ready.data["delivery"]["manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    export_report = json.loads((project / ready.data["export_report_path"]).read_text())
    canonical_by_id = {item["clip_id"]: item for item in export_report["clips"]}
    for clip in manifest["clips"]:
        delivered = project / clip["path"]
        assert delivered.is_file() and not delivered.is_symlink()
        assert sha256_file(delivered) == canonical_by_id[clip["clip_id"]]["sha256"] == clip["sha256"]
        for frame in clip["keyframes"]:
            delivered_frame = project / frame["path"]
            assert delivered_frame.is_file() and not delivered_frame.is_symlink()
            assert sha256_file(delivered_frame) == frame["sha256"]

    repeated = tool.execute({"operation": "export", **base})
    assert repeated.success, repeated.error
    assert repeated.data["delivery"] == ready.data["delivery"]

    damaged = project / ready.data["delivery_clip_paths"][0]
    damaged.write_bytes(b"damaged delivery copy")
    invalid = tool.execute({"operation": "validate", **base})
    assert not invalid.success
    assert invalid.data["delivery_status"] == "failed"
    repaired = tool.execute({"operation": "export", **base})
    assert repaired.success, repaired.error
    assert repaired.data["delivery_status"] == "ready"
    assert "-repair-1/" in repaired.data["delivery"]["manifest_path"]
    assert repaired.data["delivery_clip_paths"] != ready.data["delivery_clip_paths"]
    valid = tool.execute({"operation": "validate", **base})
    assert valid.success, valid.error
    assert valid.data["delivery_status"] == "ready"


def test_quality_failures_cannot_be_accepted_by_agent(workflow):
    project, tool, base, first = workflow
    request = document(project, first, "keyframe_selection_request.json")
    submission = selection_submission(request)
    submission["reviewer"]["kind"] = "ai_coding_assistant"
    result = tool.execute({"operation": "plan", **base, "parent_plan_revision": first.data["plan_revision"], "keyframe_submission": submission})
    assert not result.success
    assert "quality failure" in result.error
    unresolved = tool.execute({"operation": "plan", **base, "parent_plan_revision": first.data["plan_revision"], "keyframe_submission": selection_submission(request, status="needs_human")})
    assert unresolved.success, unresolved.error
    assert unresolved.data["anchor_status"] == "needs_human"
    assert all(not s["anchor_selection"]["anchors"] for s in unresolved.data["atomic_segments"])


def test_observation_cap_idempotency_and_no_automatic_anchor_promotion(workflow):
    project, tool, base, current = workflow
    for round_index in range(4):
        request = document(project, current, "keyframe_selection_request.json")
        item = request["items"][0]
        center = (item["start"]["pts"] + item["end"]["pts"]) // 2
        inputs = {"operation": "plan", **base, "parent_plan_revision": current.data["plan_revision"],
                  "keyframe_observation_requests": [{"segment_id": item["segment_id"], "request_sha256": request["request_sha256"], "target_pts": center, "radius_s": "1.5", "reason": "Inspect transient operation"}]}
        current = tool.execute(inputs)
        assert current.success, current.error
        replay = tool.execute(inputs)
        assert replay.success, replay.error
        assert replay.data["plan_revision"] == current.data["plan_revision"]
        selection = current.data["atomic_segments"][0]["anchor_selection"]
        assert selection["observation_rounds"] == min(round_index + 1, 3)
        assert len(selection["candidates"]) <= 72
        assert selection["anchors"] == []
    assert selection["status"] == "needs_human"
    assert "observation_budget_exhausted" in selection["information_gaps"]


def test_tampered_candidate_or_stale_request_is_rejected(workflow):
    project, tool, base, first = workflow
    request = document(project, first, "keyframe_selection_request.json")
    preview = project / request["items"][0]["candidates"][0]["preview"]["path"]
    original = preview.read_bytes()
    preview.write_bytes(original + b"tampered")
    bad = tool.execute({"operation": "plan", **base, "parent_plan_revision": first.data["plan_revision"], "keyframe_submission": selection_submission(request)})
    assert not bad.success and "hash mismatch" in bad.error
    preview.write_bytes(original)
    changed = deepcopy(request)
    changed["request_sha256"] = "0" * 64
    bad = tool.execute({"operation": "plan", **base, "parent_plan_revision": first.data["plan_revision"], "keyframe_submission": selection_submission(changed)})
    assert not bad.success


def test_selection_policy_change_keeps_atomic_ids_and_boundary_request(workflow):
    project, tool, base, first = workflow
    before = document(project, first, "boundary_review_request.json")
    config = json.loads(Path(base["config_path"]).read_text())
    config["representative_selection"] = {"laplacian_min": 40}
    Path(base["config_path"]).write_text(json.dumps(config))
    changed = tool.execute({"operation": "run", **base})
    assert changed.success, changed.error
    after = document(project, changed, "boundary_review_request.json")
    assert before == after
    assert [s["segment_id"] for s in first.data["atomic_segments"]] == [s["segment_id"] for s in changed.data["atomic_segments"]]
    assert first.data["boundaries"] == changed.data["boundaries"]


def test_source_change_is_not_hidden_by_export_cache(workflow):
    project, tool, base, first = workflow
    boundary = document(project, first, "boundary_review_request.json")
    exported = tool.execute({"operation": "run", **base, "parent_plan_revision": first.data["plan_revision"], "review_submission": _submission(boundary)})
    assert exported.success, exported.error
    source = Path(base["source_path"])
    source.write_bytes(source.read_bytes() + b"changed")
    repeated = tool.execute({"operation": "export", **base})
    assert not repeated.success and "Source file changed" in repeated.error


def test_bind_multiple_members_uses_member_roles_and_exact_offsets():
    from lib.replication_preprocess.selection import bind_anchors
    base = Fraction(1, 1000)
    segments = []
    for i, pts in enumerate([1600, 4100]):
        anchor = {"pts": pts, "source_time": time_point(pts, base), "role": "primary_representative"}
        segments.append({"segment_id": f"s{i}", "anchor_selection": {"status": "ready", "anchors": [anchor]}})
    segments[1]["anchor_selection"]["anchors"].append({"pts": 4700, "source_time": time_point(4700, base), "role": "supplementary_anchor"})
    state = {"atomic_segments": segments, "generation_clips": [{"clip_id": "g", "start": time_point(1200, base), "atomic_segment_ids": ["s0", "s1"], "internal_boundaries": ["b"]}]}
    bind_anchors(state)
    clip = state["generation_clips"][0]
    assert [a["generation_clip_offset_s"] for a in clip["keyframes"]] == ["0.4", "2.9", "3.5"]
    assert [a["role"] for a in clip["keyframes"]].count("primary_representative") == 2
    assert clip["internal_boundaries"] == ["b"]


def make_legacy_package(project, exported, *, keyframe_boundary_evidence=False):
    """Materialize the v2 contract, including legacy reports without media signatures."""
    from lib.replication_preprocess.storage import sha256_file
    state = document(project, exported, "plan_state.json")
    index = deepcopy(exported.data["index"])
    state["schema_version"] = index["schema_version"] = "2.0"
    for key in ("anchor_status", "next_actions", "selection_strategy", "keyframe_request", "keyframe_submission"):
        state.pop(key, None)
        index.pop(key, None)
    index.pop("delivery", None)
    index.pop("delivery_history", None)
    state["config"].pop("representative_selection")
    state["config"]["config_fingerprints"].pop("selection")
    for segment in state["atomic_segments"]:
        selection = segment.pop("anchor_selection")
        candidate = selection["candidates"][0]
        segment["keyframe"] = {"pts": candidate["pts"], "source_time": candidate["source_time"],
                               "path": candidate["preview"]["path"], "sha256": candidate["preview"]["sha256"],
                               "quality_status": "low_quality_fallback"}
    by_id = {s["segment_id"]: s for s in state["atomic_segments"]}
    for clip in state["generation_clips"]:
        clip["schema_version"] = "1.0"
        clip.pop("anchor_status", None)
        clip["keyframes"] = [{**by_id[sid]["keyframe"], "role": "primary" if i == 0 else "internal_anchor"} for i, sid in enumerate(clip["atomic_segment_ids"])]
    if keyframe_boundary_evidence:
        state["boundaries"][0]["review"]["evidence_refs"] = ["left_keyframe", "right_keyframe"]
    for filename in list(index["manifest_index"]):
        if filename.startswith("keyframe_selection"):
            index["manifest_index"].pop(filename)
            state["manifest_index"].pop(filename, None)
    replacements = {"atomic_segments.json": {"schema_version": "1.0", "atomic_segments": state["atomic_segments"]},
                    "generation_clips.json": {"schema_version": "1.0", "generation_clips": state["generation_clips"]},
                    "boundaries.json": {"schema_version": "1.0", "boundaries": state["boundaries"]}}
    for name, content in replacements.items():
        path = project / index["manifest_index"][name]["path"]
        path.write_text(json.dumps(content))
        index["manifest_index"][name]["sha256"] = sha256_file(path)
        state["manifest_index"][name] = deepcopy(index["manifest_index"][name])
    state_path = project / index["manifest_index"]["plan_state.json"]["path"]
    state_path.write_text(json.dumps(state))
    index["manifest_index"]["plan_state.json"]["sha256"] = sha256_file(state_path)
    report_path = project / index["export"]["report_path"]
    report = json.loads(report_path.read_text())
    for asset in report["clips"]:
        asset.pop("media_signature", None)
        asset.pop("media_fingerprint", None)
    report_path.write_text(json.dumps(report))
    index["export"]["report_sha256"] = sha256_file(report_path)
    index["export"].pop("plan_state_ref", None)
    (project / "artifacts/replication/index.json").write_text(json.dumps(index))
    return state


@pytest.mark.parametrize("keyframe_boundary_evidence", [False, True])
def test_legacy_auto_upgrade_freezes_timeline_and_preserves_media(workflow, monkeypatch, keyframe_boundary_evidence):
    project, tool, base, first = workflow
    boundary = document(project, first, "boundary_review_request.json")
    exported = tool.execute({"operation": "run", **base, "parent_plan_revision": first.data["plan_revision"], "review_submission": _submission(boundary)})
    assert exported.success, exported.error
    legacy = make_legacy_package(project, exported, keyframe_boundary_evidence=keyframe_boundary_evidence)
    calls = []
    original = engine_module.subprocess.run
    def spy(command, *args, **kwargs):
        if "-c:v" in command:
            calls.append(command)
        return original(command, *args, **kwargs)
    monkeypatch.setattr(engine_module.subprocess, "run", spy)
    upgraded = tool.execute({"operation": "run", **base})
    assert upgraded.success, upgraded.error
    assert upgraded.data["index"]["schema_version"] == "3.0"
    assert upgraded.data["anchor_status"] == "pending_agent"
    assert upgraded.data["clip_paths"] == exported.data["clip_paths"]
    assert not calls
    state = document(project, upgraded, "plan_state.json")
    assert [s["segment_id"] for s in state["atomic_segments"]] == [s["segment_id"] for s in legacy["atomic_segments"]]
    assert state["timeline_map"] == legacy["timeline_map"]
    request = document(project, upgraded, "keyframe_selection_request.json")
    selected = tool.execute({"operation": "run", **base, "parent_plan_revision": upgraded.data["plan_revision"], "keyframe_submission": selection_submission(request)})
    assert selected.success, selected.error
    assert not calls
    if keyframe_boundary_evidence:
        assert selected.data["plan_status"] == "needs_review"
        assert selected.data["previous_timeline"]["timeline_map"] == legacy["timeline_map"]
        assert selected.data["review_status"] == "pending_agent"
        new_request = document(project, selected, "boundary_review_request.json")
        assert new_request["review_protocol_version"] == "boundary-visual-review-v2"
    else:
        assert selected.data["clip_paths"] == exported.data["clip_paths"]


def test_corrupted_asset_is_regenerated_individually_on_reselection(workflow, monkeypatch):
    project, tool, base, first = workflow
    request = document(project, first, "boundary_review_request.json")
    exported = tool.execute({"operation": "run", **base, "parent_plan_revision": first.data["plan_revision"], "review_submission": _submission(request)})
    original_paths = exported.data["clip_paths"]
    (project / original_paths[0]).write_bytes(b"broken video")
    calls = []
    original_run = engine_module.subprocess.run
    def spy(command, *args, **kwargs):
        if "-c:v" in command:
            calls.append(command)
        return original_run(command, *args, **kwargs)
    monkeypatch.setattr(engine_module.subprocess, "run", spy)
    keyframes = document(project, exported, "keyframe_selection_request.json")
    repaired = tool.execute({"operation": "run", **base, "parent_plan_revision": exported.data["plan_revision"], "keyframe_submission": selection_submission(keyframes)})
    assert repaired.success, repaired.error
    assert len(calls) == 1
    assert repaired.data["clip_paths"][0] != original_paths[0]
    assert repaired.data["clip_paths"][1] == original_paths[1]


def test_corrupted_current_export_can_be_repaired_without_new_plan(workflow, monkeypatch):
    project, tool, base, first = workflow
    request = document(project, first, "boundary_review_request.json")
    exported = tool.execute({"operation": "run", **base, "parent_plan_revision": first.data["plan_revision"], "review_submission": _submission(request)})
    assert exported.success, exported.error
    paths = exported.data["clip_paths"]
    (project / paths[0]).unlink()
    repaired = tool.execute({"operation": "export", **base})
    assert repaired.success, repaired.error
    assert repaired.data["plan_revision"] == exported.data["plan_revision"]
    assert repaired.data["clip_paths"][1] == paths[1]
    assert repaired.data["clip_paths"][0] != paths[0]
    assert (project / exported.data["export_report_path"]).is_file()
    repeated = tool.execute({"operation": "export", **base})
    assert repeated.success, repeated.error
    assert repeated.data["clip_paths"] == repaired.data["clip_paths"]


def test_full_resolution_observation_is_evidence_until_submitted(workflow):
    project, tool, base, first = workflow
    request = document(project, first, "keyframe_selection_request.json")
    item = request["items"][0]
    candidate = item["candidates"][0]
    observed = tool.execute({"operation": "plan", **base, "parent_plan_revision": first.data["plan_revision"],
        "keyframe_observation_requests": [{"segment_id": item["segment_id"], "request_sha256": request["request_sha256"],
            "target_pts": candidate["pts"], "radius_s": "0", "full_resolution": True, "reason": "Inspect the product detail at source resolution"}]})
    assert observed.success, observed.error
    selection = observed.data["atomic_segments"][0]["anchor_selection"]
    detail = next(c for c in selection["candidates"] if c["pts"] == candidate["pts"])
    assert detail["working_image"]["format"] == "png"
    assert detail["working_image"]["width"] == 160
    assert detail["role"] == "observation" and selection["anchors"] == []


def test_reselection_keeps_previously_approved_drop(workflow):
    import subprocess
    project, tool, base, _ = workflow
    source = Path(base["source_path"])
    subprocess.run(["ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=160x120:r=10:d=0.5",
        "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=10:d=4",
        "-filter_complex", "[0:v:0][1:v:0]concat=n=2:v=1:a=0[v]", "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)], check=True)
    config = json.loads(Path(base["config_path"]).read_text())
    config["regroup"] = {"allow_drop_under_1s": True}
    config["scene_detection"]["min_scene_len_frames"] = 1
    Path(base["config_path"]).write_text(json.dumps(config))
    first = tool.execute({"operation": "plan", **base})
    assert first.success, first.error
    request = document(project, first, "boundary_review_request.json")
    short_id = first.data["atomic_segments"][0]["segment_id"]
    ready = tool.execute({"operation": "run", **base, "parent_plan_revision": first.data["plan_revision"],
        "review_submission": _submission(request), "manual_overrides": [{"action": "drop", "segment_id": short_id,
            "actor": "human", "reason": "Explicitly discard the half-second lead-in", "parent_plan_revision": first.data["plan_revision"],
            "analysis_fingerprint": first.data["config_fingerprints"]["analysis"]}]})
    assert ready.success, ready.error
    state = document(project, ready, "plan_state.json")
    assert len(state["dropped_intervals"]) == 1
    remaining_id = state["generation_clips"][0]["atomic_segment_ids"][0]
    changed = tool.execute({"operation": "run", **base, "parent_plan_revision": ready.data["plan_revision"], "keyframe_review_segment_ids": [remaining_id]})
    assert changed.success, changed.error
    new_state = document(project, changed, "plan_state.json")
    assert new_state["dropped_intervals"] == state["dropped_intervals"]
    assert new_state["timeline_map"] == state["timeline_map"]
    assert changed.data["clip_paths"] == ready.data["clip_paths"]
