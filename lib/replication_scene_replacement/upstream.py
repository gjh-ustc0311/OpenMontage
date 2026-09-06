"""Strict import of an immutable, ready Phase 1 replication package."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import re
from typing import Any

from lib.replication_preprocess.models import decimal_string, stable_id
from lib.replication_preprocess.delivery import DeliveryError, validate_selected_delivery

from .errors import SceneReplacementError
from .storage import REVISION_RE, load_json, relative_ref, resolve_under, sha256_file, sha256_json, verify_ref


PATH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _human_confirmation(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("kind") != "human":
        return False
    return bool(str(value.get("actor") or value.get("name") or "").strip())


def _path_component(value: Any, label: str) -> str:
    if not isinstance(value, str) or not PATH_COMPONENT_RE.fullmatch(value):
        raise SceneReplacementError(f"{label} must be a safe single path component")
    return value


def _time_value(point: dict[str, Any], label: str) -> Fraction:
    try:
        pts = point["pts"]
        num = point["time_base"]["num"]
        den = point["time_base"]["den"]
        seconds_text = str(point["seconds"])
        if not all(isinstance(value, int) for value in (pts, num, den)):
            raise TypeError
        exact = Fraction(pts * num, den)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise SceneReplacementError(f"{label} is not an exact PTS time point") from exc
    if seconds_text != decimal_string(exact):
        raise SceneReplacementError(f"{label} seconds do not match the canonical PTS projection")
    return exact


def _same_time(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    if left != right or _time_value(left, label) != _time_value(right, label):
        raise SceneReplacementError(f"{label} differs between Phase 1 artifacts")


def _external_source(source: dict[str, Any]) -> dict[str, Any]:
    path_value, digest = source.get("path"), source.get("file_sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise SceneReplacementError("Phase 1 source reference is incomplete")
    unresolved = Path(path_value).expanduser()
    if unresolved.is_symlink():
        raise SceneReplacementError("Phase 1 source video must not be a symlink")
    path = unresolved.resolve()
    if not path.is_file() or sha256_file(path) != digest:
        raise SceneReplacementError("Phase 1 source video is missing or has changed")
    return {
        "external_path": str(path),
        "sha256": digest,
        "start": deepcopy(source.get("start")),
        "end": deepcopy(source.get("end")),
        "time_base": deepcopy(source.get("time_base")),
        "rotation": int(source.get("rotation") or 0),
        "color": deepcopy(source.get("color", {})),
    }


def _verify_entries(entries: Any, project_dir: Path, label: str) -> None:
    if not isinstance(entries, dict) or not entries:
        raise SceneReplacementError(f"{label} manifest index is incomplete")
    for name, reference in entries.items():
        verify_ref(reference, project_dir, label=f"{label} manifest {name}")


def _delivery_for_revision(index: dict[str, Any], project_dir: Path, revision: str):
    for reference in [index.get("delivery"), *(index.get("delivery_history") or [])]:
        if not isinstance(reference, dict):
            continue
        try:
            path = verify_ref(
                {"path": reference.get("manifest_path"), "sha256": reference.get("manifest_sha256")},
                project_dir,
                label="Phase 1 delivery manifest",
            )
            delivery = load_json(path, project_dir)
        except SceneReplacementError:
            continue
        if delivery.get("plan_revision") == revision:
            return {
                "delivery_fingerprint": reference.get("delivery_fingerprint"),
                "manifest_path": reference.get("manifest_path"),
                "manifest_sha256": reference.get("manifest_sha256"),
            }, delivery
    raise SceneReplacementError(f"No retained validated Phase 1 delivery for revision {revision}")


def _selection_refs(state: dict[str, Any], project_dir: Path) -> list[dict[str, str]]:
    entries = state.get("manifest_index", {})
    wanted = ("keyframe_selection.json", "keyframe_selection_request.json", "keyframe_selection_submission.json")
    refs = [deepcopy(entries[name]) for name in wanted if name in entries]
    if not refs:
        raise SceneReplacementError("Phase 1 state has no representative-selection artifact")
    for position, reference in enumerate(refs, 1):
        verify_ref(reference, project_dir, label=f"representative-selection evidence {position}")
    return refs


def anchor_fingerprint(anchor: dict[str, Any]) -> str:
    return sha256_json(anchor)


def presentation_map_from_delivery(delivery: dict[str, Any]) -> list[dict[str, Any]]:
    """Freeze Phase 1 sequence aliases without deriving identity from filenames."""

    if delivery.get("naming_version") != "sequence-v1":
        raise SceneReplacementError("Phase 1 delivery must use sequence-v1 readable naming")
    rows: list[dict[str, Any]] = []
    seen_anchors: set[str] = set()
    seen_display_ids: set[str] = set()
    for clip in delivery.get("clips", []):
        clip_id = _path_component(clip.get("clip_id"), "Phase 1 clip_id")
        clip_sequence = clip.get("sequence")
        clip_display_id = _path_component(
            clip.get("display_id"), "Phase 1 clip display_id"
        )
        if not isinstance(clip_sequence, int) or isinstance(clip_sequence, bool) or clip_sequence < 1:
            raise SceneReplacementError("Phase 1 clip sequence must be a positive integer")
        for frame in clip.get("keyframes", []):
            anchor_id = _path_component(frame.get("anchor_id"), "Phase 1 anchor_id")
            keyframe_sequence = frame.get("sequence")
            display_id = _path_component(
                frame.get("display_id"), "Phase 1 keyframe display_id"
            )
            if (
                not isinstance(keyframe_sequence, int)
                or isinstance(keyframe_sequence, bool)
                or keyframe_sequence < 1
            ):
                raise SceneReplacementError(
                    "Phase 1 keyframe sequence must be a positive integer"
                )
            if frame.get("clip_id") != clip_id or not display_id.startswith(
                f"{clip_display_id}_K"
            ):
                raise SceneReplacementError(
                    "Phase 1 readable keyframe name does not match its clip ownership"
                )
            if anchor_id in seen_anchors or display_id in seen_display_ids:
                raise SceneReplacementError(
                    "Phase 1 presentation map contains duplicate anchors or display IDs"
                )
            seen_anchors.add(anchor_id)
            seen_display_ids.add(display_id)
            rows.append(
                {
                    "clip_id": clip_id,
                    "clip_sequence": clip_sequence,
                    "clip_display_id": clip_display_id,
                    "anchor_id": anchor_id,
                    "keyframe_sequence": keyframe_sequence,
                    "display_id": display_id,
                }
            )
    rows.sort(
        key=lambda item: (
            item["clip_sequence"],
            item["keyframe_sequence"],
            item["anchor_id"],
        )
    )
    for timeline_sequence, row in enumerate(rows, 1):
        row["timeline_sequence"] = timeline_sequence
    if not rows:
        raise SceneReplacementError("Phase 1 presentation map is empty")
    return rows


def _flatten_anchors(state: dict[str, Any], delivery: dict[str, Any], project_dir: Path):
    delivered: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for delivered_clip in delivery.get("clips", []):
        for frame in delivered_clip.get("keyframes", []):
            anchor_id = _path_component(frame.get("anchor_id"), "Phase 1 anchor_id")
            if anchor_id in delivered:
                raise SceneReplacementError("Phase 1 delivery contains duplicate or invalid anchor IDs")
            delivered[anchor_id] = delivered_clip, frame
    selection_refs = _selection_refs(state, project_dir)
    anchors, seen = [], set()
    for clip in state.get("generation_clips", []):
        for anchor in clip.get("keyframes", []):
            anchor_id = _path_component(anchor.get("anchor_id"), "Phase 1 anchor_id")
            if anchor_id in seen or anchor_id not in delivered:
                raise SceneReplacementError("Phase 1 state and delivery anchor sets are inconsistent")
            seen.add(anchor_id)
            delivered_clip, delivered_frame = delivered[anchor_id]
            if delivered_clip.get("clip_id") != clip.get("clip_id"):
                raise SceneReplacementError(f"Anchor clip ownership differs in delivery: {anchor_id}")
            for field in ("role", "atomic_segment_id", "sha256"):
                if delivered_frame.get(field) != anchor.get(field):
                    raise SceneReplacementError(f"Anchor {field} differs in delivery: {anchor_id}")
            for field in ("source_time", "atomic_segment_offset", "generation_clip_offset"):
                _same_time(anchor.get(field), delivered_frame.get(field), f"anchor {anchor_id} {field}")
            image_path = verify_ref(
                {"path": delivered_frame.get("path"), "sha256": delivered_frame.get("sha256")},
                project_dir,
                label=f"anchor image {anchor_id}",
            )
            if image_path.suffix.lower() != ".png" or anchor.get("format") != "png":
                raise SceneReplacementError(f"Anchor must use its lossless PNG working image: {anchor_id}")
            try:
                width, height = int(anchor["width"]), int(anchor["height"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SceneReplacementError(f"Anchor image metadata is incomplete: {anchor_id}") from exc
            purpose = str(anchor.get("purpose") or "").strip()
            if width <= 0 or height <= 0 or not purpose:
                raise SceneReplacementError(f"Anchor dimensions or purpose are invalid: {anchor_id}")
            transform = "; ".join(filter(None, (anchor.get("pixel_transform"), anchor.get("color_conversion")))) or "decoder_rgb24"
            value = {
                "anchor_id": anchor_id,
                "role": anchor["role"],
                "purpose": purpose,
                "atomic_segment_id": anchor["atomic_segment_id"],
                "edit_slot_id": stable_id(
                    "sres", anchor["atomic_segment_id"], anchor["role"]
                ),
                "clip_id": clip["clip_id"],
                "source_time": deepcopy(anchor["source_time"]),
                "atomic_segment_offset": deepcopy(anchor["atomic_segment_offset"]),
                "generation_clip_offset": deepcopy(anchor["generation_clip_offset"]),
                "image": {
                    "path": delivered_frame["path"],
                    "sha256": delivered_frame["sha256"],
                    "width": width,
                    "height": height,
                    "format": "png",
                    "orientation": anchor.get("orientation") or "display_normalized",
                    "color": {
                        "mode": "RGB",
                        "icc_profile_sha256": None,
                        "source_to_working_transform": transform,
                    },
                },
                "selection_refs": deepcopy(selection_refs),
            }
            anchors.append(value)
    if set(delivered) != seen or not anchors:
        raise SceneReplacementError("Phase 1 state and delivery anchor sets differ or are empty")
    roles_by_segment: dict[str, list[str]] = {}
    for anchor in anchors:
        roles_by_segment.setdefault(anchor["atomic_segment_id"], []).append(anchor["role"])
    for segment_id, roles in roles_by_segment.items():
        if roles.count("primary_representative") != 1:
            raise SceneReplacementError(
                f"Atomic segment {segment_id} must have exactly one primary representative"
            )
        if roles.count("supplementary_anchor") > 1:
            raise SceneReplacementError(
                f"Atomic segment {segment_id} may have at most one supplementary anchor"
            )
    anchors.sort(key=lambda item: (_time_value(item["source_time"], "anchor source_time"), item["anchor_id"]))
    presentation_map = presentation_map_from_delivery(delivery)
    if {item["anchor_id"] for item in presentation_map} != {item["anchor_id"] for item in anchors}:
        raise SceneReplacementError("Phase 1 presentation map and anchor set differ")
    return anchors, presentation_map


def build_source_snapshot(
    *, project_dir: Path, upstream_index_path: Path, replacement_revision: str,
    expected_project_id: str, upstream_plan_revision: str | None = None,
    supplementary_anchor_confirmations: list[dict[str, Any]] | None = None,
    inherited_supplementary_anchor_confirmations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify and freeze a current or retained historic Phase 1 revision."""
    try:
        index_path = resolve_under(upstream_index_path, project_dir, must_exist=True)
    except (OSError, ValueError) as exc:
        raise SceneReplacementError("Phase 1 index must be inside the project workspace") from exc
    expected_index = (project_dir / "artifacts" / "replication" / "index.json").resolve()
    if index_path != expected_index:
        raise SceneReplacementError(
            "Phase 1 source lock requires canonical artifacts/replication/index.json"
        )
    index = load_json(index_path, project_dir)
    if index.get("schema_version") != "3.0" or index.get("tool") != "replication_preprocess":
        raise SceneReplacementError("Phase 2 requires a Phase 1 schema 3.0 replication_preprocess index")
    if index.get("project_id") != expected_project_id:
        raise SceneReplacementError("Phase 1 project_id does not match the Phase 2 project")
    revision = upstream_plan_revision or index.get("plan_revision")
    if not REVISION_RE.fullmatch(str(revision)):
        raise SceneReplacementError("Requested Phase 1 revision is invalid")
    delivery_ref, delivery = _delivery_for_revision(index, project_dir, revision)
    state_ref = delivery.get("plan_state_ref")
    state_path = verify_ref(state_ref, project_dir, label="Phase 1 plan state")
    state = load_json(state_path, project_dir)
    if any((
        state.get("schema_version") != "3.0",
        state.get("project_id") != expected_project_id,
        state.get("plan_revision") != revision,
        state.get("plan_status") != "ready",
        state.get("anchor_status") != "ready",
    )):
        raise SceneReplacementError("Selected Phase 1 plan state is not ready")
    _verify_entries(state.get("manifest_index"), project_dir, "selected Phase 1")
    if state.get("source", {}).get("file_sha256") != delivery.get("source_sha256"):
        raise SceneReplacementError("Phase 1 source hash differs between delivery and plan state")
    export_ref = delivery.get("export_ref")
    if not isinstance(export_ref, dict) or export_ref.get("status") != "validated":
        raise SceneReplacementError("Selected Phase 1 export is not validated")
    report_path = verify_ref(
        {"path": export_ref.get("report_path"), "sha256": export_ref.get("report_sha256")},
        project_dir,
        label="Phase 1 export report",
    )
    report = load_json(report_path, project_dir)
    if report.get("status") != "validated" or export_ref.get("plan_state_ref") not in (None, state_ref):
        raise SceneReplacementError("Selected Phase 1 export report is invalid or stale")
    if delivery.get("schema_version") != "1.0" or delivery.get("delivery_fingerprint") != delivery_ref["delivery_fingerprint"]:
        raise SceneReplacementError("Phase 1 delivery is not bound to the selected plan")
    try:
        validated_delivery = validate_selected_delivery(
            project_dir,
            state,
            state_ref,
            export_ref,
            {
                "delivery_fingerprint": delivery_ref["delivery_fingerprint"],
                "manifest_path": delivery_ref["manifest_path"],
                "manifest_sha256": delivery_ref["manifest_sha256"],
            },
        )
    except (DeliveryError, OSError, ValueError, KeyError, TypeError) as exc:
        raise SceneReplacementError(f"Selected Phase 1 delivery is invalid: {exc}") from exc
    if validated_delivery != delivery:
        raise SceneReplacementError("Selected Phase 1 delivery changed during validation")
    for clip in delivery.get("clips", []):
        verify_ref({"path": clip.get("path"), "sha256": clip.get("sha256")}, project_dir, label=f"delivered clip {clip.get('clip_id')}")

    anchors, presentation_map = _flatten_anchors(state, delivery, project_dir)
    supplied = supplementary_anchor_confirmations or []
    inherited = inherited_supplementary_anchor_confirmations or []
    if not isinstance(supplied, list) or not isinstance(inherited, list):
        raise SceneReplacementError("supplementary_anchor_confirmations must be an array")
    supplementary = {
        anchor["anchor_id"]: anchor
        for anchor in anchors
        if anchor["role"] == "supplementary_anchor"
    }
    supplied_by_id: dict[str, dict[str, Any]] = {}
    for item in supplied:
        if not isinstance(item, dict):
            raise SceneReplacementError("Each supplementary anchor confirmation must be an object")
        anchor_id = _path_component(item.get("anchor_id"), "Supplementary anchor confirmation ID")
        if anchor_id in supplied_by_id:
            raise SceneReplacementError("Supplementary anchor confirmations must be unique")
        if anchor_id not in supplementary:
            raise SceneReplacementError("Confirmation may identify only a current supplementary anchor")
        supplied_by_id[anchor_id] = item
    inherited_by_id = {
        item.get("anchor_id"): item
        for item in inherited
        if isinstance(item, dict) and item.get("anchor_id") in supplementary
    }
    confirmations = []
    for anchor_id, anchor in sorted(supplementary.items()):
        prior = inherited_by_id.get(anchor_id)
        if (
            isinstance(prior, dict)
            and prior.get("purpose") == anchor["purpose"]
            and _human_confirmation(prior.get("human_confirmation"))
            and str(prior.get("gap") or "").strip()
        ):
            confirmation = deepcopy(prior)
        else:
            confirmation = deepcopy(supplied_by_id.get(anchor_id))
            if not isinstance(confirmation, dict):
                raise SceneReplacementError(
                    f"Supplementary anchor {anchor_id} requires explicit human confirmation"
                )
            if confirmation.get("upstream_plan_revision") != revision:
                raise SceneReplacementError(
                    "New supplementary anchor confirmation must bind the selected Phase 1 revision"
                )
        if (
            confirmation.get("purpose") != anchor["purpose"]
            or not str(confirmation.get("gap") or "").strip()
            or not _human_confirmation(confirmation.get("human_confirmation"))
        ):
            raise SceneReplacementError(
                f"Supplementary anchor confirmation is incomplete or stale: {anchor_id}"
            )
        confirmations.append(confirmation)
    timeline = {
        "atomic_segments": [{key: deepcopy(item[key]) for key in ("segment_id", "start", "end")} for item in state.get("atomic_segments", [])],
        "generation_clips": [{key: deepcopy(item[key]) for key in ("clip_id", "start", "end", "atomic_segment_ids")} for item in state.get("generation_clips", [])],
        "dropped_segments": [
            {
                "segment_id": item["segment_id"],
                "start": deepcopy(item["start"]),
                "end": deepcopy(item["end"]),
                **({"reason": item["reason"]} if item.get("reason") else {}),
            }
            for item in state.get("dropped_intervals", [])
        ],
    }
    for collection in ("atomic_segments", "generation_clips", "dropped_segments"):
        for item in timeline[collection]:
            _time_value(item["start"], f"{collection} {item.get('segment_id') or item.get('clip_id')} start")
            _time_value(item["end"], f"{collection} {item.get('segment_id') or item.get('clip_id')} end")
            if _time_value(item["start"], "timeline start") >= _time_value(item["end"], "timeline end"):
                raise SceneReplacementError(f"{collection} interval must have positive duration")
    body = {
        "schema_version": "1.1",
        "tool": "replication_scene_replacement",
        "project_id": expected_project_id,
        "replacement_revision": replacement_revision,
        "upstream": {
            "schema_version": "3.0",
            "tool": "replication_preprocess",
            "plan_revision": revision,
            "index_observed_ref": relative_ref(index_path, project_dir),
            "delivery_ref": {"path": delivery_ref["manifest_path"], "sha256": delivery_ref["manifest_sha256"]},
            "plan_state_ref": deepcopy(state_ref),
            "source_video_ref": _external_source(state["source"]),
            "config_fingerprint": state["config"]["config_fingerprint"],
            "selection_fingerprint": state["config"]["config_fingerprints"]["selection"],
            "delivery_fingerprint": delivery_ref["delivery_fingerprint"],
        },
        "timeline": timeline,
        "anchors": anchors,
        "presentation_map": presentation_map,
        "supplementary_anchor_confirmations": confirmations,
        "fingerprints": {
            "timeline": sha256_json(timeline),
            "anchor_set": sha256_json([{"anchor_id": a["anchor_id"], "fingerprint": anchor_fingerprint(a)} for a in anchors]),
            "presentation": sha256_json(presentation_map),
        },
    }
    body["fingerprints"]["snapshot"] = sha256_json(body)
    body["snapshot_id"] = stable_id("srs", body)
    return body


def verify_source_snapshot(snapshot: dict[str, Any], project_dir: Path) -> None:
    """Recheck active immutable dependencies, excluding the mutable discovery index."""
    upstream = snapshot.get("upstream", {})
    delivery_path = verify_ref(upstream.get("delivery_ref"), project_dir, label="Phase 1 delivery")
    delivery = load_json(delivery_path, project_dir)
    if snapshot.get("schema_version") == "1.1":
        expected_presentation = presentation_map_from_delivery(delivery)
        if snapshot.get("presentation_map") != expected_presentation:
            raise SceneReplacementError("Frozen presentation map differs from Phase 1 delivery")
        if sha256_json(expected_presentation) != snapshot.get("fingerprints", {}).get(
            "presentation"
        ):
            raise SceneReplacementError("Frozen presentation-map fingerprint mismatch")
    verify_ref(upstream.get("plan_state_ref"), project_dir, label="Phase 1 plan state")
    export_ref = delivery.get("export_ref", {})
    verify_ref({"path": export_ref.get("report_path"), "sha256": export_ref.get("report_sha256")}, project_dir, label="Phase 1 export report")
    source = upstream.get("source_video_ref", {})
    source_path = Path(str(source.get("external_path", ""))).expanduser()
    if source_path.is_symlink() or not source_path.resolve().is_file() or sha256_file(source_path.resolve()) != source.get("sha256"):
        raise SceneReplacementError("Phase 1 source video is missing or has changed")
    for anchor in snapshot.get("anchors", []):
        verify_ref(anchor.get("image"), project_dir, label=f"anchor image {anchor.get('anchor_id')}")
        for position, reference in enumerate(anchor.get("selection_refs", []), 1):
            verify_ref(reference, project_dir, label=f"selection evidence {position}")
    fingerprints = snapshot.get("fingerprints", {})
    rebuilt = deepcopy(snapshot)
    rebuilt.pop("snapshot_id", None)
    rebuilt.get("fingerprints", {}).pop("snapshot", None)
    if sha256_json(rebuilt) != fingerprints.get("snapshot"):
        raise SceneReplacementError("Frozen source snapshot fingerprint mismatch")


__all__ = [
    "anchor_fingerprint",
    "build_source_snapshot",
    "presentation_map_from_delivery",
    "verify_source_snapshot",
]
