from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageCms

from lib.replication_scene_replacement.imaging import (
    ImageAssetError,
    copy_image_asset,
    import_image_asset,
    normalize_direct_result,
)
from lib.replication_scene_replacement.quality import (
    generate_direct_comparison,
    generate_direct_review_crops,
    generate_multiframe_comparison,
)
from lib.replication_scene_replacement.storage import sha256_file


def _save_rgb(
    path: Path, color: tuple[int, int, int], size: tuple[int, int] = (4, 4)
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    try:
        image.save(path, "PNG")
    finally:
        image.close()


def test_image_import_records_exif_and_icc_then_normalizes_to_project_png(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (2, 3), (40, 90, 140))
    exif = Image.Exif()
    exif[274] = 6
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    image.save(external, "JPEG", exif=exif, icc_profile=profile, quality=95)
    image.close()

    imported = import_image_asset(
        external,
        project,
        "assets/generated.png",
        "direct_generated_result",
        raw_destination_path="assets/raw-response.jpg",
    )
    assert imported["source"]["exif_orientation"] == 6
    assert imported["source"]["size"] == [2, 3]
    assert [imported["width"], imported["height"]] == [3, 2]
    assert imported["source"]["icc_profile_present"]
    assert imported["color"]["source_to_working_transform"] == (
        "embedded_icc_to_srgb"
    )
    assert sha256_file(project / imported["raw"]["path"]) == sha256_file(external)

    with pytest.raises(ImageAssetError, match="stay under"):
        import_image_asset(
            external, project, tmp_path / "escape.png", "direct_generated_result"
        )


def test_direct_result_resizes_only_within_five_percent_ratio(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    near_target = project / "near-target.png"
    at_limit = project / "at-limit.png"
    over_limit = project / "over-limit.png"
    _save_rgb(near_target, (10, 20, 30), size=(95, 166))
    _save_rgb(at_limit, (10, 20, 30), size=(189, 320))
    _save_rgb(over_limit, (10, 20, 30), size=(100, 160))

    normalized = normalize_direct_result(
        near_target, project, "assets/direct.png"
    )
    assert normalized["method"] == (
        "lanczos_resize_to_target_within_5pct_aspect_ratio"
    )
    assert normalized["candidate_size"] == [95, 166]
    assert normalized["target_size"] == [720, 1280]
    assert normalized["aspect_ratio_difference_percent"] < 5
    with Image.open(project / normalized["image"]["path"]) as image:
        assert image.size == (720, 1280)

    boundary = normalize_direct_result(at_limit, project, "assets/boundary.png")
    assert boundary["aspect_ratio_difference_percent"] == 5
    with pytest.raises(ImageAssetError, match="exceeding the allowed 5%"):
        normalize_direct_result(over_limit, project, "assets/blocked.png")


def test_readable_delivery_copy_is_byte_identical_and_sequence_named(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "accepted.png"
    _save_rgb(source, (5, 10, 15), size=(720, 1280))
    copied = copy_image_asset(
        source,
        project,
        "assets/images/replication/scene-replacement-v2/delivery/r0001/hash/S01_K01.png",
    )
    assert copied["path"].endswith("/S01_K01.png")
    assert copied["sha256"] == sha256_file(source)
    assert (copied["width"], copied["height"]) == (720, 1280)


def test_direct_review_evidence_and_total_multiframe_board(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "source.png"
    result = project / "S01_K01.png"
    _save_rgb(source, (90, 100, 110), size=(720, 1280))
    _save_rgb(result, (20, 30, 40), size=(720, 1280))

    comparison = generate_direct_comparison(
        source, result, "assets/evidence/source-vs-result.png", project
    )
    assert comparison["panels"] == ["source", "direct_result"]
    crops = generate_direct_review_crops(
        source,
        result,
        [{"name": "product", "bbox": [10, 20, 100, 220]}],
        "assets/evidence/crops",
        project,
    )
    assert len(crops) == 1

    board = generate_multiframe_comparison(
        [
            {
                "display_id": "S01_K01",
                "scene_id": "scene_1",
                "direction_id": "direction_a",
                "source_path": "source.png",
                "result_path": "S01_K01.png",
            }
        ],
        "assets/evidence/all-anchors-comparison.png",
        project,
        title="FINAL MULTI-FRAME COMPARISON",
    )
    assert board["renderer_version"] == "paired-grid-v1"
    assert board["item_count"] == 1
    assert board["bindings"][0]["display_id"] == "S01_K01"
    assert (project / board["path"]).is_file()
