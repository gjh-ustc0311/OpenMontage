"""Stream exact decoded frames to disk; PNG is the RGB working baseline."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

from PIL import Image, ImageDraw

from .storage import sha256_file


def iter_images(path: Path, time_base: Fraction, pts_set: set[int], rotation: int = 0) -> Iterator[tuple[int, Image.Image]]:
    """Yield one owned RGB image at a time. Consumers must close it promptly."""
    import av
    from .analysis import MediaAnalysisError, _rescale_pts

    if not pts_set:
        return
    if rotation not in {0, 90, 180, 270}:
        raise MediaAnalysisError("Only right-angle display rotations are supported")
    seen = set()
    last_pts = max(pts_set)
    transpose = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = _rescale_pts(frame.pts, Fraction(frame.time_base or stream.time_base), time_base)
            if pts > last_pts:
                break
            if pts not in pts_set:
                continue
            picture = Image.fromarray(frame.to_ndarray(format="rgb24"))
            if rotation:
                rotated = picture.transpose(transpose[rotation])
                picture.close()
                picture = rotated
            seen.add(pts)
            try:
                yield pts, picture
            finally:
                picture.close()
    if seen != pts_set:
        raise MediaAnalysisError(f"Could not decode requested PTS: {sorted(pts_set - seen)[:8]}")


def save_frames(*, path: Path, time_base: Fraction, pts_set: set[int], source: dict[str, Any],
                output_dir: Path, project_dir: Path, preview_size: int | None = None,
                jpeg_quality: int = 92) -> dict[int, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for pts, picture in iter_images(path, time_base, pts_set, int(source.get("rotation") or 0)):
        original_size = picture.size
        if preview_size:
            picture.thumbnail((preview_size, preview_size), Image.Resampling.LANCZOS)
        extension = "jpg" if preview_size else "png"
        destination = output_dir / f"pts-{pts}.{extension}"
        if preview_size:
            picture.save(destination, "JPEG", quality=jpeg_quality)
        else:
            picture.save(destination, "PNG")
        result[pts] = {
            "path": destination.resolve().relative_to(project_dir.resolve()).as_posix(),
            "sha256": sha256_file(destination),
            "width": picture.width, "height": picture.height,
            "working_width": original_size[0], "working_height": original_size[1],
            "format": extension, "source_rotation": int(source.get("rotation") or 0),
            "orientation": "display_normalized", "source_color": source.get("color", {}),
            "pixel_transform": "decoder_rgb24_then_right_angle_transpose",
            "color_conversion": "decoder_default; no transfer-function or gamut normalization",
        }
    return result


def contact_pages(candidates: list[dict[str, Any]], output_dir: Path, project_dir: Path,
                  page_size: int = 12) -> list[dict[str, str]]:
    """Bound both page height and decoded preview residency."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for offset in range(0, len(candidates), page_size):
        items = candidates[offset:offset + page_size]
        sheet = Image.new("RGB", (3 * 340, ((len(items) + 2) // 3) * 260), "#202020")
        draw = ImageDraw.Draw(sheet)
        for i, candidate in enumerate(items):
            x, y = (i % 3) * 340, (i // 3) * 260
            with Image.open(project_dir / candidate["preview"]["path"]) as picture:
                picture.thumbnail((324, 210))
                sheet.paste(picture, (x + (340 - picture.width) // 2, y + 45))
            draw.text((x + 8, y + 5), f"{candidate['candidate_id']}\nPTS {candidate['pts']}  {candidate['technical_status']}", fill="white")
        destination = output_dir / f"page-{offset // page_size + 1:03}.jpg"
        sheet.save(destination, "JPEG", quality=90)
        sheet.close()
        pages.append({"path": destination.relative_to(project_dir).as_posix(), "sha256": sha256_file(destination)})
    return pages
