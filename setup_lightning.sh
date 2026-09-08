#!/usr/bin/env bash
# Install project deps into the *current* Python (Lightning AI / Studio / single-env cloud).
# Assumes PyTorch is already provided by the image. Does not create conda envs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
PIP="${PIP:-pip}"
REQ_FILE="${REQ_FILE:-requirements-lightning.txt}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"

echo "==> Using interpreter: $($PYTHON -c 'import sys; print(sys.executable)')"
echo "==> Python: $($PYTHON -c 'import sys; print(sys.version.split()[0])')"

if ! $PYTHON -c 'import torch' 2>/dev/null; then
  echo "ERROR: torch is not importable. On Lightning AI the image should provide it." >&2
  echo "       Refusing to install torch from requirements-lightning.txt." >&2
  exit 1
fi

$PYTHON - <<'PY'
import torch
print(f"==> Found torch {torch.__version__} (cuda={torch.cuda.is_available()})")
try:
    import torchvision
    print(f"==> Found torchvision {torchvision.__version__}")
except Exception:
    print("==> WARNING: torchvision missing; install it only if hardpicks/models need it")
PY

echo "==> Upgrading pip/setuptools/wheel"
$PYTHON -m pip install --upgrade pip setuptools wheel

echo "==> Installing deps from ${REQ_FILE} (no torch reinstall)"
$PYTHON -m pip install -r "$REQ_FILE"

if [[ "$SKIP_VERIFY" == "1" ]]; then
  echo "==> SKIP_VERIFY=1 — done (no import checks)."
  exit 0
fi

echo "==> Verifying imports"
$PYTHON - <<'PY'
from pathlib import Path
import torch
import hardpicks

print("torch:", torch.__version__)
print("hardpicks:", hardpicks.__file__)
print("TOP_DIR:", hardpicks.TOP_DIR)
assert (Path(hardpicks.TOP_DIR) / "config").is_dir(), (
    "hardpicks is not editable — config/ missing under TOP_DIR"
)
import hardpicks.data.fbp.gather_parser  # noqa: F401
import h5py, numpy, matplotlib, tqdm  # noqa: F401
print("verify OK")
PY

echo "==> Done."
