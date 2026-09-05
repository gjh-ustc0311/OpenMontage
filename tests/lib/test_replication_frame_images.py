from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction

import pytest
from PIL import Image

from lib.replication_preprocess.analysis import MediaAnalysisError, probe_source
from lib.replication_preprocess.frame_images import iter_images, save_frames

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")


def make_vfr(path):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    pts = [1100]
    for i in range(119):
        pts.append(pts[-1] + (40 if i % 2 else 70))
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=20)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "bgr0"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for i, timestamp in enumerate(pts):
            pixels = np.zeros((64, 96, 3), dtype=np.uint8)
            pixels[:, :48] = (20 + i, 60, 160)
            pixels[:32, 48:] = (220, 40, 70)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_vfr_nonzero_pts_and_lossless_png_working_baseline(tmp_path):
    np = pytest.importorskip("numpy")
    path = tmp_path / "vfr.mkv"
    make_vfr(path)
    source, ledger, base = probe_source(path)
    assert source["vfr"] and source["start"]["pts"] > 0
    wanted = {ledger[1].pts, ledger[-1].pts}
    images = save_frames(path=path, time_base=base, pts_set=wanted, source=source, output_dir=tmp_path / "images", project_dir=tmp_path)
    for pts, decoded in iter_images(path, base, wanted):
        with Image.open(tmp_path / images[pts]["path"]) as saved:
            assert np.array_equal(np.asarray(decoded), np.asarray(saved))
        assert images[pts]["source_color"] == source["color"]
    with pytest.raises(MediaAnalysisError, match="Could not decode"):
        save_frames(path=path, time_base=base, pts_set={source["end"]["pts"]}, source=source,
                    output_dir=tmp_path / "bad", project_dir=tmp_path)


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_rotation_matches_ffmpeg_display_direction(tmp_path, angle):
    pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    pixels = np.zeros((64, 96, 3), dtype=np.uint8)
    pixels[:32, :48] = (250, 20, 20)
    pixels[:32, 48:] = (20, 240, 20)
    pixels[32:, :48] = (20, 20, 230)
    pixels[32:, 48:] = (200, 180, 20)
    seed = tmp_path / "pattern.png"
    Image.fromarray(pixels).save(seed)
    original, rotated = tmp_path / "original.mp4", tmp_path / "rotated.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-loop", "1", "-i", str(seed), "-t", "0.2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(original)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(original), "-c", "copy", "-metadata:s:v:0", f"rotate={angle}", str(rotated)], check=True)
    source, ledger, base = probe_source(rotated)
    images = save_frames(path=rotated, time_base=base, pts_set={ledger[0].pts}, source=source,
                         output_dir=tmp_path / "frames", project_dir=tmp_path)
    info = images[ledger[0].pts]
    reference = subprocess.run(["ffmpeg", "-v", "error", "-i", str(rotated), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], check=True, capture_output=True).stdout
    expected = np.frombuffer(reference, dtype=np.uint8).reshape(info["height"], info["width"], 3)
    with Image.open(tmp_path / info["path"]) as image:
        actual = np.asarray(image)
    # ffmpeg and bundled libswscale may differ by a rounding unit.
    assert np.abs(actual.astype(int) - expected.astype(int)).mean() < 2


def test_requested_original_images_do_not_accumulate(tmp_path, monkeypatch):
    path = tmp_path / "vfr.mkv"
    make_vfr(path)
    source, ledger, base = probe_source(path)
    real_fromarray, real_close = Image.fromarray, Image.Image.close
    active = set()
    maximum = 0
    def created(*args, **kwargs):
        nonlocal maximum
        image = real_fromarray(*args, **kwargs)
        active.add(id(image))
        maximum = max(maximum, len(active))
        return image
    def closed(image):
        active.discard(id(image))
        return real_close(image)
    monkeypatch.setattr(Image, "fromarray", created)
    monkeypatch.setattr(Image.Image, "close", closed)
    save_frames(path=path, time_base=base, pts_set={f.pts for f in ledger}, source=source,
                output_dir=tmp_path / "frames", project_dir=tmp_path)
    assert maximum == 1 and not active


def test_nonzero_vfr_source_exports_the_planned_interval(tmp_path, monkeypatch):
    pytest.importorskip("scenedetect")
    pytest.importorskip("cv2")
    from tools.analysis import replication_preprocess as module
    project = tmp_path / "projects" / "vfr"
    project.mkdir(parents=True)
    source = tmp_path / "source.mkv"
    make_vfr(source)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"scene_detection": {"initial_threshold": 255, "minimum_threshold": 255}, "export": {"preset": "ultrafast"}}))
    monkeypatch.setattr(module, "PROJECTS_DIR", project.parent)
    result = module.ReplicationPreprocess().execute({"operation": "run", "project_id": "vfr", "source_path": str(source),
        "config_path": str(config), "output_path": str(project / "artifacts/replication/index.json")})
    assert result.success, result.error
    assert result.data["validation_status"] == "passed"
    assert len(result.data["clip_paths"]) == 1
    exported = project / result.data["clip_paths"][0]
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(exported)], capture_output=True, text=True, check=True).stdout)
    assert Fraction(probe["streams"][0]["start_time"]) == 0


@pytest.mark.parametrize("transfer", ["smpte2084", "arib-std-b67"])
def test_explicit_hdr_is_rejected_before_decode(tmp_path, monkeypatch, transfer):
    from lib.replication_preprocess import analysis
    path = tmp_path / "tagged.mp4"
    path.write_bytes(b"probe fixture")
    monkeypatch.setattr(analysis, "_run_json", lambda _: {"streams": [{"codec_type": "video", "color_transfer": transfer}]})
    with pytest.raises(MediaAnalysisError, match="HDR PQ/HLG"):
        probe_source(path)
