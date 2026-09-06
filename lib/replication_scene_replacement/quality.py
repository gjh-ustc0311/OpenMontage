"""Deterministic evidence for direct-generation review and final approval."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from lib.replication_preprocess.storage import sha256_file

from .errors import SceneReplacementError
from .imaging import (
    ImageAssetError,
    _atomic_save_png,
    _has_nonopaque_alpha,
    _orientation_value,
    _project_input,
    _project_output,
    _relative,
)


MULTIFRAME_RENDERER_VERSION = "paired-grid-v1"
MULTIFRAME_MAX_ITEMS = 200
MULTIFRAME_MAX_PIXELS = 100_000_000


class QualityEvidenceError(SceneReplacementError, ValueError):
    """Raised when deterministic review evidence cannot be constructed."""


def _open_rgb(path: Path, *, expected_size: tuple[int, int] | None = None) -> Image.Image:
    try:
        image = Image.open(path)
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise QualityEvidenceError(f"Evidence input cannot be decoded: {path}") from exc
    try:
        if _orientation_value(image) != 1:
            raise QualityEvidenceError("Evidence inputs must already be display-normalized")
        if _has_nonopaque_alpha(image):
            raise QualityEvidenceError("Evidence inputs must be fully opaque")
        if expected_size is not None and image.size != expected_size:
            raise QualityEvidenceError(
                f"Evidence input dimensions {image.size} do not match {expected_size}"
            )
        return image.convert("RGB")
    finally:
        image.close()


def _asset(path: Path, root: Path, **extra: Any) -> dict[str, Any]:
    value = {"path": _relative(path, root), "sha256": sha256_file(path)}
    value.update(extra)
    return value


def _sanitize(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-") or "region"


def _expanded_bbox(
    bbox: Sequence[int], size: tuple[int, int], margin: int = 12
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    return (
        max(0, left - margin),
        max(0, top - margin),
        min(size[0], right + margin),
        min(size[1], bottom + margin),
    )


def _map_bbox_between_canvases(
    bbox: Sequence[int],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    source_width, source_height = source_size
    target_width, target_height = target_size
    return (
        max(0, min(target_width - 1, left * target_width // source_width)),
        max(0, min(target_height - 1, top * target_height // source_height)),
        max(1, min(target_width, math.ceil(right * target_width / source_width))),
        max(1, min(target_height, math.ceil(bottom * target_height / source_height))),
    )


def _align_source_to_direct_result(
    source: Image.Image, result_size: tuple[int, int]
) -> tuple[Image.Image, str]:
    if source.size == result_size:
        return source.copy(), "identity"
    return (
        source.resize(result_size, Image.Resampling.LANCZOS),
        "lanczos_review_alignment_to_direct_result_canvas",
    )


def generate_direct_comparison(
    source_path: str | Path,
    result_path: str | Path,
    output_path: str | Path,
    project_dir: str | Path,
) -> dict[str, Any]:
    """Generate a labeled source/direct-result pair without implying compositing."""

    try:
        root, source_file = _project_input(project_dir, source_path)
        _, result_file = _project_input(root, result_path)
        _, output_file = _project_output(root, output_path)
    except ImageAssetError as exc:
        raise QualityEvidenceError(str(exc)) from exc
    if output_file.suffix.lower() != ".png":
        raise QualityEvidenceError("Direct comparison evidence must use .png")
    source_native = _open_rgb(source_file)
    result = _open_rgb(result_file)
    source, alignment_method = _align_source_to_direct_result(source_native, result.size)
    try:
        header = 28
        sheet = Image.new("RGB", (source.width * 2, source.height + header), "#202020")
        try:
            draw = ImageDraw.Draw(sheet)
            for index, (label, panel) in enumerate(
                (("SOURCE", source), ("DIRECT RESULT", result))
            ):
                x = index * source.width
                draw.text((x + 8, 8), label, fill="white")
                sheet.paste(panel, (x, header))
            _atomic_save_png(sheet, output_file)
        finally:
            sheet.close()
        return _asset(
            output_file,
            root,
            panels=["source", "direct_result"],
            source_sha256=sha256_file(source_file),
            result_sha256=sha256_file(result_file),
            source_native_size=list(source_native.size),
            result_size=list(result.size),
            source_review_alignment=alignment_method,
        )
    finally:
        result.close()
        source.close()
        source_native.close()


def generate_direct_review_crops(
    source_path: str | Path,
    result_path: str | Path,
    review_regions: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    project_dir: str | Path,
) -> list[dict[str, Any]]:
    """Create paired source/direct-result crops for declared QA regions."""

    try:
        root, source_file = _project_input(project_dir, source_path)
        _, result_file = _project_input(root, result_path)
    except ImageAssetError as exc:
        raise QualityEvidenceError(str(exc)) from exc
    source_native = _open_rgb(source_file)
    result = _open_rgb(result_file)
    source, _ = _align_source_to_direct_result(source_native, result.size)
    crops: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(review_regions):
            if not isinstance(item, Mapping):
                raise QualityEvidenceError(f"review_regions[{index}] must be an object")
            name, bbox = item.get("name"), item.get("bbox")
            if not isinstance(name, str) or not name.strip():
                raise QualityEvidenceError(
                    f"review_regions[{index}].name must be non-empty"
                )
            if (
                isinstance(bbox, (str, bytes))
                or not isinstance(bbox, Sequence)
                or len(bbox) != 4
                or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox)
            ):
                raise QualityEvidenceError(
                    f"review_regions[{index}].bbox must contain four integers"
                )
            left, top, right, bottom = bbox
            if not (
                0 <= left < right <= source_native.width
                and 0 <= top < bottom <= source_native.height
            ):
                raise QualityEvidenceError(
                    f"review_regions[{index}].bbox must be inside the canvas"
                )
            source_bounded = _expanded_bbox(
                (left, top, right, bottom), source_native.size
            )
            bounded = _map_bbox_between_canvases(
                source_bounded, source_native.size, result.size
            )
            source_crop = source.crop(bounded)
            result_crop = result.crop(bounded)
            try:
                header = 24
                comparison = Image.new(
                    "RGB",
                    (source_crop.width * 2, source_crop.height + header),
                    "#202020",
                )
                try:
                    draw = ImageDraw.Draw(comparison)
                    draw.text((6, 6), "SOURCE", fill="white")
                    draw.text((source_crop.width + 6, 6), "DIRECT RESULT", fill="white")
                    comparison.paste(source_crop, (0, header))
                    comparison.paste(result_crop, (source_crop.width, header))
                    _, crop_path = _project_output(
                        root,
                        Path(output_dir)
                        / f"crop-{index + 1:02d}-{_sanitize(name)}.png",
                    )
                    _atomic_save_png(comparison, crop_path)
                    crops.append(_asset(crop_path, root))
                finally:
                    comparison.close()
            finally:
                result_crop.close()
                source_crop.close()
        return crops
    finally:
        result.close()
        source.close()
        source_native.close()


def generate_contact_sheet(
    items: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    project_dir: str | Path,
    *,
    columns: int = 3,
    thumbnail_size: tuple[int, int] = (320, 240),
) -> dict[str, Any]:
    """Generate a small result-only sheet for non-approval diagnostics."""

    if isinstance(columns, bool) or not isinstance(columns, int) or not 1 <= columns <= 8:
        raise QualityEvidenceError("columns must be an integer between 1 and 8")
    thumb_width, thumb_height = thumbnail_size
    if thumb_width <= 0 or thumb_height <= 0:
        raise QualityEvidenceError("thumbnail_size must be positive")
    try:
        root = Path(project_dir).resolve(strict=True)
        _, output_file = _project_output(root, output_path)
    except (FileNotFoundError, ImageAssetError) as exc:
        raise QualityEvidenceError(str(exc)) from exc
    normalized: list[tuple[str, str | None, Path]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise QualityEvidenceError(f"items[{index}] must be an object")
        label, status, path = item.get("label"), item.get("status"), item.get("path")
        if not isinstance(label, str) or not label or not isinstance(path, (str, Path)):
            raise QualityEvidenceError(f"items[{index}] requires label and path")
        _, local_path = _project_input(root, path)
        normalized.append((label, status if isinstance(status, str) else None, local_path))
    if not normalized:
        raise QualityEvidenceError("Contact sheet requires at least one image")
    cell_width, cell_height = thumb_width + 20, thumb_height + 54
    rows = math.ceil(len(normalized) / columns)
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), "#202020")
    try:
        draw = ImageDraw.Draw(sheet)
        for index, (label, status, path) in enumerate(normalized):
            image = _open_rgb(path)
            try:
                thumbnail = ImageOps.contain(
                    image, (thumb_width, thumb_height), Image.Resampling.LANCZOS
                )
                try:
                    column, row = index % columns, index // columns
                    x, y = column * cell_width, row * cell_height
                    sheet.paste(
                        thumbnail,
                        (
                            x + 10 + (thumb_width - thumbnail.width) // 2,
                            y + 42 + (thumb_height - thumbnail.height) // 2,
                        ),
                    )
                    draw.text((x + 10, y + 8), label, fill="white")
                    if status:
                        draw.text((x + 10, y + 24), status, fill="#b8c4d8")
                finally:
                    thumbnail.close()
            finally:
                image.close()
        _atomic_save_png(sheet, output_file)
    finally:
        sheet.close()
    return _asset(
        output_file,
        root,
        item_count=len(normalized),
        columns=columns,
        rows=rows,
    )


def _fitted_panel(path: Path, size: tuple[int, int]) -> Image.Image:
    image = _open_rgb(path)
    try:
        thumbnail = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
        panel = Image.new("RGB", size, "#171717")
        panel.paste(
            thumbnail,
            ((size[0] - thumbnail.width) // 2, (size[1] - thumbnail.height) // 2),
        )
        thumbnail.close()
        return panel
    finally:
        image.close()


def generate_multiframe_comparison(
    items: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    project_dir: str | Path,
    *,
    title: str,
) -> dict[str, Any]:
    """Render one chronological source/result board plus cell binding metadata."""

    rows_in = list(items)
    if not rows_in:
        raise QualityEvidenceError("Multi-frame comparison requires at least one pair")
    if len(rows_in) > MULTIFRAME_MAX_ITEMS:
        raise QualityEvidenceError(
            f"Multi-frame comparison is limited to {MULTIFRAME_MAX_ITEMS} anchors"
        )
    columns = 4 if len(rows_in) <= 24 else 6 if len(rows_in) <= 96 else 8
    panel_size = (360, 640)
    pair_width, pair_height, header_height = panel_size[0] * 2, panel_size[1], 80
    row_count = math.ceil(len(rows_in) / columns)
    canvas_size = (pair_width * columns, header_height + pair_height * row_count)
    if canvas_size[0] * canvas_size[1] > MULTIFRAME_MAX_PIXELS:
        raise QualityEvidenceError(
            "Multi-frame comparison would exceed the 100MP safety limit"
        )
    try:
        root = Path(project_dir).resolve(strict=True)
        _, output_file = _project_output(root, output_path)
    except (FileNotFoundError, ImageAssetError) as exc:
        raise QualityEvidenceError(str(exc)) from exc
    if output_file.suffix.lower() != ".png":
        raise QualityEvidenceError("Multi-frame comparison output must use .png")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(rows_in):
        display_id = item.get("display_id")
        source_path, result_path = item.get("source_path"), item.get("result_path")
        if (
            not isinstance(display_id, str)
            or not display_id
            or display_id in seen
            or not isinstance(source_path, (str, Path))
            or not isinstance(result_path, (str, Path))
        ):
            raise QualityEvidenceError(
                f"items[{index}] requires a unique display_id, source_path, and result_path"
            )
        seen.add(display_id)
        _, source_file = _project_input(root, source_path)
        _, result_file = _project_input(root, result_path)
        normalized.append(
            {
                "display_id": display_id,
                "scene_id": str(item.get("scene_id") or ""),
                "direction_id": str(item.get("direction_id") or ""),
                "source": source_file,
                "result": result_file,
            }
        )

    canvas = Image.new("RGB", canvas_size, "#111111")
    bindings: list[dict[str, Any]] = []
    try:
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.text((32, 22), title, fill="white")
        draw.text(
            (32, 46),
            f"SOURCE | GENERATED  -  {len(normalized)} ANCHORS",
            fill="#B9C0C9",
        )
        for index, item in enumerate(normalized):
            column, row = index % columns, index // columns
            pair_x = column * pair_width
            pair_y = header_height + row * pair_height
            for side, (role, path) in enumerate(
                (("SOURCE", item["source"]), ("GENERATED", item["result"]))
            ):
                panel = _fitted_panel(path, panel_size)
                try:
                    x = pair_x + side * panel_size[0]
                    canvas.paste(panel, (x, pair_y))
                    draw.rectangle((x, pair_y, x + panel_size[0], pair_y + 50), fill=(0, 0, 0, 175))
                    draw.text((x + 12, pair_y + 10), f"{item['display_id']}  {role}", fill="white")
                    if item["scene_id"] or item["direction_id"]:
                        draw.text(
                            (x + 12, pair_y + 28),
                            f"{item['scene_id']}  {item['direction_id']}".strip(),
                            fill="#B9C0C9",
                        )
                finally:
                    panel.close()
            bindings.append(
                {
                    "display_id": item["display_id"],
                    "scene_id": item["scene_id"],
                    "direction_id": item["direction_id"],
                    "bbox": [pair_x, pair_y, pair_x + pair_width, pair_y + pair_height],
                    "source_sha256": sha256_file(item["source"]),
                    "result_sha256": sha256_file(item["result"]),
                }
            )
        _atomic_save_png(canvas, output_file)
    finally:
        canvas.close()
    return {
        **_asset(
            output_file,
            root,
            item_count=len(normalized),
            columns=columns,
            rows=row_count,
            width=canvas_size[0],
            height=canvas_size[1],
            renderer_version=MULTIFRAME_RENDERER_VERSION,
        ),
        "bindings": bindings,
    }


__all__ = [
    "MULTIFRAME_MAX_ITEMS",
    "MULTIFRAME_MAX_PIXELS",
    "MULTIFRAME_RENDERER_VERSION",
    "QualityEvidenceError",
    "generate_contact_sheet",
    "generate_direct_comparison",
    "generate_direct_review_crops",
    "generate_multiframe_comparison",
]
