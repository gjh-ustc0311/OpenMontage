from __future__ import annotations

from fractions import Fraction

import pytest

from lib.replication_preprocess.analysis import FrameRecord, evidence_targets
from lib.replication_preprocess.config import RepresentativeSelectionConfig, load_config
from lib.replication_preprocess.models import time_point
from lib.replication_preprocess.selection import candidate_pts, make_candidate, nearest, technical


def metrics(sharpness=120, luma=128, *, transient=False):
    return {"sharpness": sharpness, "luma_mean": luma, "black_ratio": 0,
            "white_ratio": 0, "stability_delta": 1.0,
            "descriptor": {"dhash": 255 if transient else 0,
                           "rgb_grid": [240 if transient else 64] * 48}}


def test_full_interval_preserves_brief_visual_state_and_temporal_coverage():
    policy = RepresentativeSelectionConfig().model_dump()
    values = list(range(200))
    qualities = {pts: metrics() for pts in values}
    qualities[151] = metrics(transient=True)
    chosen = candidate_pts(values, qualities, policy, 24)
    assert 0 in chosen and 199 in chosen and 151 in chosen
    assert "visual_change" in chosen[151]
    assert len(chosen) <= 24
    assert chosen == candidate_pts(values, qualities, policy, 24)


def test_moving_clear_frames_pass_but_all_blurry_candidates_stay_flagged():
    policy = RepresentativeSelectionConfig().model_dump()
    assert not technical(metrics(), policy)[0]
    assert "blur" in technical(metrics(0), policy)[0]
    segment = {"segment_id": "s", "start": time_point(100, Fraction(1, 1000))}
    candidate = make_candidate("abc", segment, 950, metrics(0), ["temporal_coverage"], policy)
    assert candidate["technical_status"] == "failed"
    assert candidate["role"] == "observation"
    assert candidate["atomic_segment_offset"] == time_point(850, Fraction(1, 1000))


def test_nearest_uses_real_vfr_pts_with_earlier_tie():
    assert nearest([1100, 1140, 1200, 1290], 1170) == 1140
    assert nearest([1100, 1140, 1200, 1290], 1201) == 1200


def test_selection_overlay_is_independent_of_cut_and_export_fingerprints(tmp_path):
    config = tmp_path / "selection.json"
    config.write_text('{"representative_selection":{"laplacian_min":30}}')
    old, changed = load_config(), load_config(str(config))
    for stage in ["analysis", "review", "planning", "export"]:
        assert old["config_fingerprints"][stage] == changed["config_fingerprints"][stage]
    assert old["config_fingerprints"]["selection"] != changed["config_fingerprints"]["selection"]
    assert old["keyframe"] == changed["keyframe"]


@pytest.mark.parametrize("patch", [{"initial_candidates": 73}, {"page_size": 13}, {"max_candidates": 30}, {"novelty_min": "NaN"}, {"observation_radius_s": "-0.1"}])
def test_invalid_resource_or_quality_limits_are_rejected(patch):
    with pytest.raises(ValueError):
        RepresentativeSelectionConfig(**patch)


def test_boundary_evidence_does_not_depend_on_representative_pts():
    base = Fraction(1, 100)
    segments = {
        "left": {"start": time_point(100, base), "end": time_point(400, base)},
        "right": {"start": time_point(400, base), "end": time_point(900, base)},
    }
    ledger = [FrameRecord(i, pts) for i, pts in enumerate(range(100, 900, 10))]
    boundary = {"left_segment_id": "left", "right_segment_id": "right", "time": time_point(400, base)}
    before = evidence_targets(boundary, segments, ledger, base, Fraction(1, 2))
    segments["left"]["keyframe"] = {"pts": 390}
    segments["right"]["keyframe"] = {"pts": 850}
    assert before == evidence_targets(boundary, segments, ledger, base, Fraction(1, 2))
    assert before["left_edge"] == 390 and before["right_edge"] == 400
    assert before["left_observation"] == 250 and before["right_observation"] == 650
