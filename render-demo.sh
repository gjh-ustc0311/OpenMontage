#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_with_python() {
  local interpreter="$1"
  shift
  if ! "$interpreter" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
    echo "ERROR: OpenMontage requires Python 3.12+: $interpreter" >&2
    echo "See docs/python-upgrade.md to migrate the environment." >&2
    exit 1
  fi
  exec "$interpreter" "$SCRIPT_DIR/render_demo.py" "$@"
}

for environment in "${VENV_DIR:-$SCRIPT_DIR/.venv}" "${VIRTUAL_ENV:-}" "${CONDA_PREFIX:-}"; do
  [ -n "$environment" ] || continue
  for executable in "$environment/bin/python" "$environment/Scripts/python.exe"; do
    if [ -x "$executable" ]; then
      run_with_python "$executable" "$@"
    fi
  done
done

for executable in python3.12 python3 python; do
  if command -v "$executable" >/dev/null 2>&1; then
    run_with_python "$executable" "$@"
  fi
done

echo "ERROR: Python 3.12+ was not found. Run make venv first." >&2
exit 1
