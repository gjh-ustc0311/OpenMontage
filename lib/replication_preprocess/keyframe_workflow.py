"""Persistence operations for agent-authored keyframe decisions."""

from __future__ import annotations

import time
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

from .analysis import analyze_frame_quality, collect_runtime_versions, probe_source
from .frame_images import contact_pages, save_frames
from .models import fraction_from_time_base, stable_id
from .review import ReviewValidationError
from .selection import (
    PROTOCOL, candidate_pts, check_submission, make_candidate, request_for,
    validate_document,
)
from .storage import next_revision, resolve_under, sha256_file, sha256_json


class KeyframeWorkflow:
    """Concrete preparation/submission operations; no LLM or semantic policy."""

    def _prepare_keyframes(self, state, segment_ids=None, *, ledger=None, qualities=None):
        started = time.monotonic()
        if segment_ids is not None and (not isinstance(segment_ids, list) or not segment_ids
                or any(not isinstance(s, str) for s in segment_ids) or len(segment_ids) != len(set(segment_ids))):
            raise ReviewValidationError("keyframe_review_segment_ids must be a nonempty unique string array")
        source_path = Path(state["source"]["path"])
        if ledger is None:
            source, ledger, base = probe_source(source_path)
            if source["file_sha256"] != state["source"]["file_sha256"]:
                raise ReviewValidationError("Source changed during keyframe preparation")
            state["source"] = source
        else:
            base = fraction_from_time_base(state["source"]["time_base"])
        config = state["config"]
        policy = config["representative_selection"]
        if qualities is None:
            qualities = analyze_frame_quality(source_path, base, config["keyframe"]["analysis_width_px"], config, include_descriptors=True)
        targets = set(segment_ids if segment_ids is not None else [s["segment_id"] for s in state["atomic_segments"]])
        if not targets or targets - {s["segment_id"] for s in state["atomic_segments"]}:
            raise ReviewValidationError("Keyframe review requires existing segment IDs")
        root = self.image_root / "revisions" / state["plan_revision"] / "selection"
        wanted = set()
        for segment in state["atomic_segments"]:
            if segment["segment_id"] not in targets:
                continue
            old = segment.get("anchor_selection", {})
            legacy = segment.pop("keyframe", None)
            if legacy:
                segment["legacy_keyframe"] = legacy
            previous_primary = next((a["pts"] for a in old.get("anchors", []) if a["role"] == "primary_representative"), None)
            if previous_primary is None:
                previous_primary = old.get("previous_primary_pts", segment.get("legacy_keyframe", {}).get("pts"))
            values = [f.pts for f in ledger if segment["start"]["pts"] <= f.pts < segment["end"]["pts"]]
            chosen = candidate_pts(values, qualities, policy, policy["initial_candidates"])
            candidates = [make_candidate(state["source"]["file_sha256"], segment, pts, qualities[pts], reasons, policy) for pts, reasons in chosen.items()]
            segment["anchor_selection"] = {
                "strategy": PROTOCOL, "task_id": stable_id("kft", segment["segment_id"], state["plan_revision"], config["config_fingerprints"]["selection"]),
                "status": "pending_agent" if candidates else "needs_human",
                "candidates": candidates, "contact_sheets": [], "anchors": [],
                "observation_rounds": 0, "observation_history": [],
                "selection_fingerprint": config["config_fingerprints"]["selection"],
                "previous_revision": state.get("parent_plan_revision"),
                "previous_primary_pts": previous_primary,
                "information_gaps": [] if candidates else ["no_decodable_frame"],
            }
            wanted.update(chosen)
        images = save_frames(path=source_path, time_base=base, pts_set=wanted, source=state["source"],
                             output_dir=root / "previews", project_dir=self.project_dir,
                             preview_size=policy["preview_long_edge_px"])
        for segment in state["atomic_segments"]:
            if segment["segment_id"] not in targets:
                continue
            selection = segment["anchor_selection"]
            for candidate in selection["candidates"]:
                candidate["preview"] = images[candidate["pts"]]
            selection["contact_sheets"] = contact_pages(selection["candidates"], root / segment["segment_id"], self.project_dir, policy["page_size"])
        self._refresh_selection_request(state)
        state.setdefault("selection_timings", []).append({"stage": "candidate_preparation", "duration_seconds": round(time.monotonic() - started, 3)})

    def _refresh_selection_request(self, state):
        pending = [s["segment_id"] for s in state["atomic_segments"]
                   if s.get("anchor_selection", {}).get("status") in {"pending_agent", "needs_more_evidence", "needs_human"}]
        state["keyframe_request"] = request_for(state, pending) if pending else None

    def _verify_keyframe_request(self, state):
        request = state.get("keyframe_request")
        if not request:
            raise ReviewValidationError("No pending keyframe selection request")
        body = deepcopy(request)
        claimed = body.pop("request_sha256")
        if sha256_json(body) != claimed:
            raise ReviewValidationError("Keyframe request hash mismatch")
        if request["source_sha256"] != state["source"]["file_sha256"] or request["selection_fingerprint"] != state["config"]["config_fingerprints"]["selection"]:
            raise ReviewValidationError("Keyframe request dependencies changed")
        for item in request["items"]:
            for image in [*[c["preview"] for c in item["candidates"]],
                          *[c["working_image"] for c in item["candidates"] if c.get("working_image")],
                          *item["contact_sheets"]]:
                path = resolve_under(self.project_dir / image["path"], self.project_dir, must_exist=True)
                if sha256_file(path) != image["sha256"]:
                    raise ReviewValidationError("Keyframe evidence hash mismatch")
        return request

    def _observe_keyframes(self, state, observations):
        request = self._verify_keyframe_request(state)
        validate_document("keyframe_observation_requests", observations)
        policy = state["config"]["representative_selection"]
        by_id = {s["segment_id"]: s for s in state["atomic_segments"]}
        seen = set()
        for observation in observations:
            sid = observation["segment_id"]
            if sid not in by_id or sid in seen:
                raise ReviewValidationError("Observation requires unique existing segment IDs")
            seen.add(sid)
            if observation["request_sha256"] != request["request_sha256"]:
                raise ReviewValidationError("Observation request hash is stale")
            segment = by_id[sid]
            if not segment["start"]["pts"] <= observation["target_pts"] < segment["end"]["pts"]:
                raise ReviewValidationError("Observation target PTS must be inside its segment")
            if segment["anchor_selection"]["status"] == "ready":
                raise ReviewValidationError("Start targeted keyframe review before changing a ready selection")
        source, ledger, base = probe_source(Path(state["source"]["path"]))
        qualities = analyze_frame_quality(Path(source["path"]), base, state["config"]["keyframe"]["analysis_width_px"], state["config"], include_descriptors=True)
        root = self.image_root / "revisions" / state["plan_revision"] / "observations"
        for observation in observations:
            segment = by_id[observation["segment_id"]]
            selection = segment["anchor_selection"]
            if selection["observation_rounds"] >= policy["max_observation_rounds"] or len(selection["candidates"]) >= policy["max_candidates"]:
                selection["status"] = "needs_human"
                selection["information_gaps"] = ["observation_budget_exhausted", observation["reason"]]
                continue
            radius = Fraction(observation.get("radius_s", policy["observation_radius_s"]))
            lower = max(Fraction(segment["start"]["pts"]), observation["target_pts"] - radius / base)
            upper = min(Fraction(segment["end"]["pts"]), observation["target_pts"] + radius / base)
            values = [f.pts for f in ledger if lower <= f.pts <= upper and f.pts < segment["end"]["pts"]]
            if not values:
                from .selection import nearest
                available = [f.pts for f in ledger if segment["start"]["pts"] <= f.pts < segment["end"]["pts"]]
                values = [nearest(available, observation["target_pts"])] if available else []
            old_pts = {c["pts"] for c in selection["candidates"]}
            limit = min(policy["observations_per_round"], policy["max_candidates"] - len(old_pts))
            chosen = candidate_pts(values, qualities, policy, limit, old_pts)
            detail_pts = None
            if observation.get("full_resolution") and values:
                from .selection import nearest
                detail_pts = nearest(values, observation["target_pts"])
                if detail_pts not in old_pts and detail_pts not in chosen:
                    if len(chosen) >= limit:
                        chosen.pop(next(reversed(chosen)))
                    chosen[detail_pts] = ["explicit_detail_target"]
            images = save_frames(path=Path(source["path"]), time_base=base, pts_set=set(chosen), source=source,
                                 output_dir=root / segment["segment_id"] / "previews", project_dir=self.project_dir,
                                 preview_size=policy["preview_long_edge_px"])
            for pts, reasons in chosen.items():
                candidate = make_candidate(source["file_sha256"], segment, pts, qualities[pts], ["targeted_observation", *reasons], policy)
                candidate["preview"] = images[pts]
                selection["candidates"].append(candidate)
            if detail_pts is not None:
                details = save_frames(path=Path(source["path"]), time_base=base, pts_set={detail_pts}, source=source,
                                      output_dir=root / segment["segment_id"] / "details", project_dir=self.project_dir)
                next(c for c in selection["candidates"] if c["pts"] == detail_pts)["working_image"] = details[detail_pts]
            selection["candidates"].sort(key=lambda c: c["pts"])
            selection["observation_rounds"] += 1
            selection["observation_history"].append({"request": observation, "actual_pts": list(chosen), "detail_pts": detail_pts, "round": selection["observation_rounds"]})
            selection["status"] = "pending_agent" if chosen or detail_pts is not None else "needs_human"
            selection["information_gaps"] = [observation["reason"]]
            selection["contact_sheets"] = contact_pages(selection["candidates"], root / segment["segment_id"], self.project_dir, policy["page_size"])
        self._refresh_selection_request(state)

    def _accept_keyframes(self, state, submission):
        request = self._verify_keyframe_request(state)
        decisions = check_submission(state, request, submission)
        selected = {}
        for sid, decision in decisions.items():
            ids = [] if decision["status"] != "ready" else [decision["primary_candidate_id"], *[a["candidate_id"] for a in decision.get("supplementary_anchors", [])]]
            item = next(i for i in request["items"] if i["segment_id"] == sid)
            selected[sid] = [c for cid in ids for c in item["candidates"] if c["candidate_id"] == cid]
        base = fraction_from_time_base(state["source"]["time_base"])
        images = save_frames(path=Path(state["source"]["path"]), time_base=base,
                             pts_set={c["pts"] for candidates in selected.values() for c in candidates},
                             source=state["source"], output_dir=self.image_root / "revisions" / state["plan_revision"] / "anchors",
                             project_dir=self.project_dir)
        changed_primary = set()
        changes = []
        for segment in state["atomic_segments"]:
            sid = segment["segment_id"]
            if sid not in decisions:
                continue
            selection = segment["anchor_selection"]
            decision = decisions[sid]
            selection.update(status=decision["status"], decision=decision, reviewer=submission["reviewer"],
                             request_sha256=request["request_sha256"], submission_sha256=sha256_json(submission), anchors=[],
                             information_gaps=decision["information_gaps"])
            if selection["status"] == "needs_more_evidence" and selection["observation_rounds"] >= state["config"]["representative_selection"]["max_observation_rounds"]:
                selection["status"] = "needs_human"
            supplements = {a["candidate_id"]: a["purpose"] for a in decision.get("supplementary_anchors", [])}
            for index, candidate in enumerate(selected[sid]):
                role = "primary_representative" if index == 0 else "supplementary_anchor"
                anchor = {**deepcopy(candidate), **images[candidate["pts"]], "role": role,
                          "anchor_id": stable_id("kfa", candidate["candidate_id"], role),
                          "purpose": decision["rationale"] if index == 0 else supplements[candidate["candidate_id"]],
                          "quality_override_reason": decision.get("quality_override_reason"),
                          "runtime_versions": state["runtime_versions"]}
                selection["anchors"].append(anchor)
                if index == 0 and selection.get("previous_primary_pts") is not None and candidate["pts"] != selection["previous_primary_pts"]:
                    changed_primary.add(sid)
            changes.append({"segment_id": sid, "previous_revision": selection.get("previous_revision"),
                            "anchor_ids": [a["anchor_id"] for a in selection["anchors"]],
                            "dependency_status": "requires_downstream_check"})
        state["keyframe_submission"] = deepcopy(submission)
        state["anchor_changes"] = changes
        # Keep unresolved request provenance intact; successful items need no new review.
        unresolved = [s["segment_id"] for s in state["atomic_segments"] if s["anchor_selection"]["status"] != "ready"]
        state["keyframe_request"] = request_for(state, unresolved) if unresolved else None
        affected = []
        for boundary in state["boundaries"]:
            refs = boundary.get("review", {}).get("evidence_refs", [])
            if boundary.get("manual_override"):
                continue
            if ((boundary["left_segment_id"] in changed_primary and "left_keyframe" in refs) or
                (boundary["right_segment_id"] in changed_primary and "right_keyframe" in refs)):
                boundary["previous_review"] = deepcopy(boundary.get("review"))
                boundary.update(relationship="pending_agent", hard=None, hard_reason="representative_evidence_changed")
                affected.append(boundary)
        if affected:
            state["previous_timeline"] = {key: deepcopy(state.get(key, [])) for key in ("generation_clips", "timeline_map", "dropped_intervals")}
            source, ledger, base = probe_source(Path(state["source"]["path"]))
            pending = [b for b in state["boundaries"] if b["relationship"] in {"pending_agent", "uncertain"}]
            request, _, sheet = self._build_review_artifacts(source_path=Path(source["path"]), source=source, ledger=ledger,
                time_base=base, segments=state["atomic_segments"], pending=pending, config=state["config"], revision=state["plan_revision"], review_round=1)
            state.update(review_request=request, review_status="pending_agent", contact_sheet_path=sheet)
            self._finish_state(state, set(state.get("approved_drop_ids", [])))

    def _selection_revision(self, inputs, config, *, current=None):
        started = time.monotonic()
        index = self._load_json(self.index_path)
        self._assert_index_integrity(index)
        parent = current or self._load_state(index["plan_revision"])
        source_path = Path(inputs["source_path"]).expanduser().resolve()
        if sha256_file(source_path) != parent["source"]["file_sha256"]:
            raise ReviewValidationError("Keyframe source does not match parent")
        if parent["runtime_versions"] != collect_runtime_versions():
            raise ReviewValidationError("Media runtime changed; run before submitting keyframes")
        supplied_parent = inputs.get("parent_plan_revision")
        fingerprint = sha256_json({"selection_inputs": inputs, "config": config["config_fingerprint"]})
        existing = self._find_idempotent(fingerprint)
        if existing:
            existing["index"] = index
            return existing
        if supplied_parent and supplied_parent != index["plan_revision"]:
            raise ReviewValidationError("Keyframe parent revision is stale")
        if not current and not supplied_parent:
            raise ReviewValidationError("Keyframe changes require parent_plan_revision")
        for stage in ("analysis", "review", "planning"):
            if parent["config"]["config_fingerprints"][stage] != config["config_fingerprints"][stage]:
                raise ReviewValidationError(f"{stage} configuration changed; run without a selection action first")
        state = deepcopy(parent)
        state["source"]["path"] = str(source_path)
        state.update(schema_version="3.0", plan_revision=next_revision(self.revisions_dir),
                     parent_plan_revision=parent["plan_revision"], config=config, input_fingerprint=fingerprint,
                     executed_stages=["selection"], reused_stages=["analysis", "review", "planning"],
                     invalidated_stages=["selection"], next_action=None, keyframe_submission=None)
        if inputs.get("keyframe_submission") is not None:
            self._accept_keyframes(state, inputs["keyframe_submission"])
        elif inputs.get("keyframe_observation_requests") is not None:
            self._observe_keyframes(state, inputs["keyframe_observation_requests"])
        else:
            self._prepare_keyframes(state, inputs.get("keyframe_review_segment_ids"))
        state.setdefault("selection_timings", []).append({"stage": "selection_operation", "duration_seconds": round(time.monotonic() - started, 3)})
        return self._write_revision(state, [])
