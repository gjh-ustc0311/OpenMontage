"""Exercise interpreter selection without installing dependencies or running demos."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path):
    for name in ("Makefile", "render-demo.sh"):
        shutil.copy2(ROOT / name, tmp_path / name)
    (tmp_path / "render_demo.py").write_text(
        "import json, sys\nprint('demo-args=' + json.dumps(sys.argv[1:]))\n"
    )
    return tmp_path


def interpreter(directory: Path, version=(3, 12, 12)) -> Path:
    """Run the real version checks with a controlled interpreter version."""
    executable = directory / "bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, runpy, sys\n"
        "from pathlib import Path\n"
        "with Path(__file__).with_suffix('.calls').open('a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        f"sys.version_info = {version!r}\n"
        "if sys.argv[1:2] == ['-c']:\n"
        "    exec(sys.argv[2], {'__name__': '__main__'})\n"
        "elif sys.argv[1:3] == ['-m', 'pip']:\n"
        "    print('pip test stub')\n"
        "else:\n"
        "    sys.argv = sys.argv[1:]\n"
        "    runpy.run_path(sys.argv[0], run_name='__main__')\n"
    )
    executable.chmod(0o755)
    return executable


def run(workspace, *command, **overrides):
    env = dict(os.environ)
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX", "VENV_DIR", "PYTHON_VERSION", "MAKEFLAGS", "MFLAGS", "OS"):
        env.pop(key, None)
    env.update(overrides)
    return subprocess.run(
        command, cwd=workspace, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
    )


@pytest.mark.skipif(shutil.which("make") is None, reason="make is unavailable")
def test_make_prefers_project_over_unrelated_active_environment(workspace):
    project = interpreter(workspace / ".venv")
    active = interpreter(workspace / "external", (3, 10, 20))
    result = run(workspace, "make", "venv", VIRTUAL_ENV=str(active.parent.parent))
    assert result.returncode == 0, result.stdout
    assert project.with_suffix(".calls").exists()
    assert not active.with_suffix(".calls").exists()


@pytest.mark.skipif(shutil.which("make") is None, reason="make is unavailable")
@pytest.mark.parametrize("active_only", [False, True])
def test_make_rejects_old_environment_without_replacing_it(workspace, active_only):
    old = interpreter(workspace / ("external" if active_only else ".venv"), (3, 10, 20))
    original = old.read_bytes()
    env = {"VIRTUAL_ENV": str(old.parent.parent)} if active_only else {}
    result = run(workspace, "make", "venv", **env)
    assert result.returncode != 0
    assert "requires Python 3.12+" in result.stdout
    assert "docs/python-upgrade.md" in result.stdout
    assert old.read_bytes() == original
    calls = [json.loads(line) for line in old.with_suffix(".calls").read_text().splitlines()]
    assert all(args[:1] == ["-c"] for args in calls)


@pytest.mark.skipif(shutil.which("make") is None, reason="make is unavailable")
@pytest.mark.parametrize("requested,version,success", [
    ("3.13", (3, 12, 12), False),
    ("3.13", (3, 13, 0), True),
    ("3.10", (3, 10, 20), False),
])
def test_make_checks_requested_version_and_project_minimum(workspace, requested, version, success):
    interpreter(workspace / ".venv", version)
    result = run(workspace, "make", "venv", f"PYTHON_VERSION={requested}")
    assert (result.returncode == 0) is success, result.stdout


@pytest.mark.skipif(shutil.which("make") is None, reason="make is unavailable")
@pytest.mark.parametrize("target", ["venv", "demo-list"])
def test_make_accepts_custom_environment_with_spaces(workspace, target):
    custom = interpreter(workspace / "custom environment")
    result = run(workspace, "make", target, f"VENV_DIR={custom.parent.parent}")
    assert result.returncode == 0, result.stdout
    if target == "demo-list":
        assert 'demo-args=["--list"]' in result.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_demo_uses_project_environment_and_forwards_arguments(workspace):
    project = interpreter(workspace / ".venv")
    active = interpreter(workspace / "external", (3, 10, 20))
    result = run(workspace, "bash", "render-demo.sh", "--list", "two words",
                 VIRTUAL_ENV=str(active.parent.parent))
    assert result.returncode == 0, result.stdout
    assert 'demo-args=["--list", "two words"]' in result.stdout
    assert project.with_suffix(".calls").exists()
    assert not active.with_suffix(".calls").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_demo_refuses_old_project_environment(workspace):
    interpreter(workspace / ".venv", (3, 11, 0))
    result = run(workspace, "bash", "render-demo.sh", "--list")
    assert result.returncode != 0
    assert "requires Python 3.12+" in result.stdout
    assert "demo-args=" not in result.stdout
