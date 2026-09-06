"""Safe image import and whole-canvas normalization for direct generation."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from lib.replication_preprocess.storage import resolve_under, sha256_file

from .errors import SceneReplacementError


DIRECT_RESULT_TARGET_SIZE = (720, 1280)
DIRECT_RESULT_MAX_ASPECT_RATIO_DIFFERENCE_PERCENT = 5


class ImageAssetError(SceneReplacementError, ValueError):
    """Raised when a direct-generation image cannot be safely handled."""


def _source_file(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ImageAssetError(f"Image source must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ImageAssetError(f"Image source does not exist: {candidate}") from exc
    if not resolved.is_file():
        raise ImageAssetError(f"Image source is not a regular file: {candidate}")
    return resolved


def _project_root(project_dir: str | Path) -> Path:
    try:
        root = Path(project_dir).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ImageAssetError(f"Project directory does not exist: {project_dir}") from exc
    if not root.is_dir():
        raise ImageAssetError(f"Project directory is not a directory: {project_dir}")
    return root


def _project_input(project_dir: str | Path, path: str | Path) -> tuple[Path, Path]:
    root = _project_root(project_dir)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.is_symlink():
        raise ImageAssetError(f"Project image input must not be a symlink: {path}")
    try:
        resolved = resolve_under(candidate, root, must_exist=True)
    except (FileNotFoundError, ValueError) as exc:
        raise ImageAssetError(
            f"Image input must be an existing project-local file: {path}"
        ) from exc
    if not resolved.is_file():
        raise ImageAssetError(f"Image input is not a regular file: {path}")
    return root, resolved


def _project_output(project_dir: str | Path, path: str | Path) -> tuple[Path, Path]:
    root = _project_root(project_dir)
    destination = Path(path)
    if not destination.is_absolute():
        destination = root / destination
    try:
        resolved = resolve_under(destination, root)
    except ValueError as exc:
        raise ImageAssetError(f"Image output must stay under the project directory: {path}") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved = resolve_under(resolved, root)
    except ValueError as exc:
        raise ImageAssetError(f"Image output must stay under the project directory: {path}") from exc
    if resolved.exists() and (resolved.is_symlink() or not resolved.is_file()):
        raise ImageAssetError(f"Image output is not a regular independent file: {path}")
    return root, resolved


def _relative(path: Path, root: Path) -> str:
    return path.resolve(strict=True).relative_to(root).as_posix()


def _publish_immutable(temporary_name: str, destination: Path) -> None:
    temporary = Path(temporary_name)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ImageAssetError(
                f"Image destination is not a regular independent file: {destination}"
            )
        if sha256_file(temporary) != sha256_file(destination):
            raise ImageAssetError(f"Image destination collision: {destination}")
        temporary.unlink()
        return
    os.replace(temporary, destination)


def _atomic_copy(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(source_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        _publish_immutable(temporary_name, destination)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_save_png(image: Image.Image, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        image.save(temporary_name, format="PNG", optimize=False, compress_level=9)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        _publish_immutable(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _verify_decodable(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageAssetError(f"Image cannot be decoded: {path}") from exc


def _orientation_value(image: Image.Image) -> int:
    try:
        return int(image.getexif().get(274, 1) or 1)
    except (AttributeError, TypeError, ValueError):
        return 1


def _has_nonopaque_alpha(image: Image.Image) -> bool:
    if "A" not in image.getbands() and "transparency" not in image.info:
        return False
    alpha = image.convert("RGBA").getchannel("A")
    try:
        low, _ = alpha.getextrema()
        return low != 255
    finally:
        alpha.close()


def _normalize_rgb(image: Image.Image, icc_profile: bytes | None) -> tuple[Image.Image, str]:
    if _has_nonopaque_alpha(image):
        raise ImageAssetError(
            "Transparent image results are unsupported; provide a fully opaque canvas"
        )
    if icc_profile:
        try:
            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
            destination_profile = ImageCms.createProfile("sRGB")
            return (
                ImageCms.profileToProfile(
                    image, source_profile, destination_profile, outputMode="RGB"
                ),
                "embedded_icc_to_srgb",
            )
        except Exception as exc:
            raise ImageAssetError("Embedded ICC profile could not be converted to sRGB") from exc
    return image.convert("RGB"), "decoded_channels_assumed_srgb"


def inspect_image(path: str | Path) -> dict[str, Any]:
    source = _source_file(path)
    _verify_decodable(source)
    try:
        with Image.open(source) as image:
            icc = image.info.get("icc_profile")
            orientation = _orientation_value(image)
            normalized = ImageOps.exif_transpose(image)
            try:
                normalized_size = [normalized.width, normalized.height]
            finally:
                if normalized is not image:
                    normalized.close()
            return {
                "source_path": str(source),
                "sha256": sha256_file(source),
                "format": image.format,
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
                "exif_orientation": orientation,
                "normalized_size": normalized_size,
                "has_alpha": "A" in image.getbands() or "transparency" in image.info,
                "icc_profile": {
                    "present": bool(icc),
                    "sha256": hashlib.sha256(icc).hexdigest() if icc else None,
                    "byte_length": len(icc) if icc else 0,
                },
            }
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageAssetError(f"Image cannot be inspected: {source}") from exc


def import_image_asset(
    source_path: str | Path,
    project_dir: str | Path,
    destination_path: str | Path,
    role: str,
    *,
    raw_destination_path: str | Path | None = None,
) -> dict[str, Any]:
    """Import an orientation-normalized, fully opaque sRGB PNG."""

    if not isinstance(role, str) or not role.strip():
        raise ImageAssetError("Image role must be a non-empty string")
    source = _source_file(source_path)
    _verify_decodable(source)
    root, destination = _project_output(project_dir, destination_path)
    if destination.suffix.lower() != ".png":
        raise ImageAssetError("Normalized image destination must use a .png suffix")
    raw_destination: Path | None = None
    if raw_destination_path is not None:
        _, raw_destination = _project_output(root, raw_destination_path)
        if raw_destination == destination:
            raise ImageAssetError("Raw and normalized image destinations must differ")
    try:
        with Image.open(source) as image:
            original_format = image.format
            original_mode = image.mode
            original_size = [image.width, image.height]
            orientation = _orientation_value(image)
            icc_profile = image.info.get("icc_profile")
            transposed = ImageOps.exif_transpose(image)
            try:
                normalized, color_conversion = _normalize_rgb(transposed, icc_profile)
                try:
                    _atomic_save_png(normalized, destination)
                    normalized_size = [normalized.width, normalized.height]
                finally:
                    normalized.close()
            finally:
                if transposed is not image:
                    transposed.close()
    except ImageAssetError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageAssetError(f"Image import failed: {source}") from exc
    if raw_destination is not None:
        _atomic_copy(source, raw_destination)
    result: dict[str, Any] = {
        "role": role,
        "path": _relative(destination, root),
        "sha256": sha256_file(destination),
        "width": normalized_size[0],
        "height": normalized_size[1],
        "format": "png",
        "mode": "RGB",
        "orientation": "exif_transposed_to_display_normalized",
        "color": {
            "mode": "RGB",
            "icc_profile_sha256": None,
            "source_to_working_transform": color_conversion,
            "working_space": "sRGB",
        },
        "source": {
            "sha256": sha256_file(source),
            "format": original_format,
            "mode": original_mode,
            "size": original_size,
            "exif_orientation": orientation,
            "icc_profile_present": bool(icc_profile),
            "icc_profile_sha256": hashlib.sha256(icc_profile).hexdigest()
            if icc_profile
            else None,
        },
    }
    if raw_destination is not None:
        result["raw"] = {
            "path": _relative(raw_destination, root),
            "sha256": sha256_file(raw_destination),
        }
    return result


def normalize_direct_result(
    candidate_path: str | Path,
    project_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Resize a near-9:16 full canvas to 720x1280 without crop or padding."""

    root, candidate = _project_input(project_dir, candidate_path)
    _, output = _project_output(root, output_path)
    if output.suffix.lower() != ".png":
        raise ImageAssetError("Direct-result output must use a .png suffix")
    target_width, target_height = DIRECT_RESULT_TARGET_SIZE
    _verify_decodable(candidate)
    try:
        with Image.open(candidate) as image:
            if _orientation_value(image) != 1:
                raise ImageAssetError(
                    "Generated result must be display-normalized before resize"
                )
            if _has_nonopaque_alpha(image):
                raise ImageAssetError("Generated result must be a fully opaque canvas")
            rgb = image.convert("RGB")
            try:
                source_size = rgb.size
                source_width, source_height = source_size
                ratio_delta = abs(
                    source_width * target_height - source_height * target_width
                )
                ratio_baseline = source_height * target_width
                ratio_difference_percent = ratio_delta * 100 / ratio_baseline
                if source_size == DIRECT_RESULT_TARGET_SIZE:
                    method = "identity"
                    normalized = rgb.copy()
                else:
                    if (
                        ratio_delta * 100
                        > ratio_baseline
                        * DIRECT_RESULT_MAX_ASPECT_RATIO_DIFFERENCE_PERCENT
                    ):
                        raise ImageAssetError(
                            "Generated result aspect ratio differs from the 720x1280 target by "
                            f"{ratio_difference_percent:.3f}%, exceeding the allowed 5%"
                        )
                    method = "lanczos_resize_to_target_within_5pct_aspect_ratio"
                    normalized = rgb.resize(
                        DIRECT_RESULT_TARGET_SIZE, Image.Resampling.LANCZOS
                    )
                try:
                    _atomic_save_png(normalized, output)
                finally:
                    normalized.close()
            finally:
                rgb.close()
    except ImageAssetError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageAssetError(f"Direct-result normalization failed: {candidate}") from exc
    return {
        "status": "ready",
        "method": method,
        "candidate_size": list(source_size),
        "target_size": list(DIRECT_RESULT_TARGET_SIZE),
        "aspect_ratio_difference_percent": round(ratio_difference_percent, 6),
        "image": {
            "role": "direct_generated_result",
            "path": _relative(output, root),
            "sha256": sha256_file(output),
            "width": target_width,
            "height": target_height,
            "format": "png",
            "mode": "RGB",
            "orientation": "display_normalized",
            "color": {
                "mode": "RGB",
                "icc_profile_sha256": None,
                "source_to_working_transform": "none_after_import"
                if method == "identity"
                else "lanczos_resize_to_720x1280_within_5pct_aspect_ratio",
                "working_space": "sRGB",
            },
        },
    }


def copy_image_asset(
    source_path: str | Path,
    project_dir: str | Path,
    destination_path: str | Path,
) -> dict[str, Any]:
    """Publish an immutable byte-identical PNG copy at a readable delivery path."""

    root, source = _project_input(project_dir, source_path)
    _, destination = _project_output(root, destination_path)
    if source.suffix.lower() != ".png" or destination.suffix.lower() != ".png":
        raise ImageAssetError("Readable delivery images must use .png")
    _atomic_copy(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise ImageAssetError("Readable delivery copy hash differs from its accepted result")
    with Image.open(destination) as image:
        width, height = image.size
    return {
        "path": _relative(destination, root),
        "sha256": sha256_file(destination),
        "width": width,
        "height": height,
        "format": "png",
    }


__all__ = [
    "DIRECT_RESULT_MAX_ASPECT_RATIO_DIFFERENCE_PERCENT",
    "DIRECT_RESULT_TARGET_SIZE",
    "ImageAssetError",
    "copy_image_asset",
    "import_image_asset",
    "inspect_image",
    "normalize_direct_result",
]
