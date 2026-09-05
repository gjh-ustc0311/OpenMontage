"""Verified media identity independent of keyframe and manifest revisions."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

from .storage import sha256_json


def media_signature(state: dict, clip: dict, encoding: dict) -> dict:
    source = state["source"]
    return {
        "source_sha256": source["file_sha256"],
        "start": {k: clip["start"][k] for k in ("pts", "time_base")},
        "end": {k: clip["end"][k] for k in ("pts", "time_base")},
        "video_stream_index": source["video_stream_index"],
        "audio_stream_index": source.get("audio_stream_index"),
        "audio_present": source["audio_present"],
        "rotation": source.get("rotation", 0),
        "width": source["width"], "height": source["height"],
        "encoding": deepcopy(encoding),
        "filter_revision": "copyts-pts-trim-even-dimensions-v1",
        "ffmpeg": state["runtime_versions"]["ffmpeg"],
    }


def media_fingerprint(state: dict, clip: dict, encoding: dict) -> str:
    return sha256_json(media_signature(state, clip, encoding))


def legacy_trim_equivalent(state: dict, clip: dict) -> bool:
    """Old seconds-based exports are provably equivalent for exact zero-origin cuts."""
    if state["source"]["start"]["pts"] != 0:
        return False
    for field in ("start", "end"):
        point = clip[field]
        if Fraction(point["seconds"]) != point["pts"] * Fraction(point["time_base"]["num"], point["time_base"]["den"]):
            return False
    return True
