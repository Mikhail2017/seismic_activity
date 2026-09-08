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
except Exception as exc:
    print(
        "ERROR: torchvision missing. hardpicks.data.transforms and models.resnet require it.\n"
        "       Install a build matching the image torch, e.g.:\n"
        "         pip install torchvision --index-url https://download.pytorch.org/whl/cu121\n"
        f"       ({type(exc).__name__}: {exc})",
        flush=True,
    )
    raise SystemExit(1)
PY

echo "==> Installing/pinning pip tooling (setuptools must stay <82 for pkg_resources)"
$PYTHON -m pip install --upgrade "pip" "wheel>=0.40" "packaging>=23" "setuptools>=68,<82"

echo "==> Installing deps from ${REQ_FILE} (no torch reinstall)"
$PYTHON -m pip install -r "$REQ_FILE"

if [[ "$SKIP_VERIFY" == "1" ]]; then
  echo "==> SKIP_VERIFY=1 — done (no import checks)."
  exit 0
fi

echo "==> Verifying imports"
$PYTHON - <<'PY'
from pathlib import Path
import importlib
import sys

sys.path.insert(0, str(Path(".").resolve()))

required = [
    "torch",
    "pytorch_lightning",
    "hardpicks",
    "h5py",
    "numpy",
    "scipy",
    "pandas",
    "PIL",
    "cv2",
    "yaml",
    "einops",
    "mlflow",
    "mock",
    "orion",
    "deepdiff",
    "segmentation_models_pytorch",
    "timm",
    "pretrainedmodels",
    "fairscale",
    "tqdm",
]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__}: {exc}")

if missing:
    print("MISSING/FAILED:")
    for m in missing:
        print(" ", m)
    raise SystemExit(1)

import hardpicks
import torch
import torchvision
from seismic_utils.hardpicks_pl_compat import ensure_hardpicks_lightning_compat

print("PL compat:", ensure_hardpicks_lightning_compat())
import hardpicks.data.fbp.data_module  # noqa: F401
import hardpicks.models.fbp.unet  # noqa: F401

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("hardpicks:", hardpicks.__file__)
print("TOP_DIR:", hardpicks.TOP_DIR)
assert (Path(hardpicks.TOP_DIR) / "config").is_dir(), (
    "hardpicks is not editable — config/ missing under TOP_DIR"
)
print("verify OK")
PY

echo "==> Done."
