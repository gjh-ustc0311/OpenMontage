# Python 3.12 migration

OpenMontage requires Python **3.12 or newer**. New project environments default
to 3.12, and CI tests both 3.12 and 3.13. Optional tools can have additional
Python-version restrictions; they are not installed by this migration.

`make` and `render-demo.sh` use the project environment (`.venv`, or
`VENV_DIR`) first. When it is absent, an explicitly activated virtualenv or
conda environment is used. An incompatible selected environment produces an
error; it is never silently deleted or upgraded. To use an alternative
environment when `.venv` exists, set `VENV_DIR` to its path.

## Migrate an existing environment

Run these commands from the repository root. Deactivate an old virtualenv or
conda environment first. Use a different backup directory if
`scratch/python-upgrade` already exists; keep the backup until validation passes.

```bash
mkdir -p scratch
mkdir scratch/python-upgrade
.venv/bin/python -m pip freeze --exclude-editable > scratch/python-upgrade/constraints.txt
mv .venv scratch/python-upgrade/venv
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-dev.txt -r requirements-replication.txt -c scratch/python-upgrade/constraints.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip check
make lint
make test
```

The constraints retain the old dependency versions. If the resolver identifies
an incompatible pin, update that pin in the migration constraints and retry;
avoid an unrelated upgrade of every dependency. The replication profile keeps
its pinned PyAV and headless PySceneDetect versions.

Create the replacement environment at its final `.venv` path. Moving a newly
built virtualenv into place later leaves absolute paths in activation and
entry-point scripts pointing to the wrong location.

For a fresh checkout, `make install-dev install-replication` creates the default
environment. Use `make PYTHON_VERSION=3.13 install-dev install-replication` to
create a 3.13 environment. An existing environment is checked against both the
3.12 project minimum and the requested version.

On Windows, use `py -3.12 -m venv .venv` and `.venv\Scripts\python.exe` instead.
Back up and restore the environment directory using PowerShell `Move-Item`.

## Roll back

If the replacement environment fails validation, keep it for diagnosis and
restore the original directory to its original path:

```bash
mv .venv scratch/python-upgrade/venv-failed
mv scratch/python-upgrade/venv .venv
```

Also revert the Python-version upgrade changes before using `make` with the old
interpreter. Preserve unrelated working-tree changes. Re-activate the restored
environment in the terminal. Project media, configuration and credentials do
not need to be moved or recreated.
