#!/usr/bin/env bash
# Bootstrap seismic_activity on a fresh machine (viewer + hardpicks training deps).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required" >&2
  exit 1
fi

ENV_NAME="${ENV_NAME:-seismic_activity}"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating existing env: $ENV_NAME"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  pip install -r requirements.txt
else
  echo "Creating env: $ENV_NAME"
  conda env create -f environment.yml
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
fi

python - <<'PY'
import hardpicks
from pathlib import Path
print("hardpicks OK:", hardpicks.__file__)
print("TOP_DIR:", hardpicks.TOP_DIR)
assert (Path(hardpicks.TOP_DIR) / "config").is_dir(), "editable hardpicks install required (config/ missing)"
import hardpicks.data.fbp.gather_parser as gp  # noqa: F401
print("gather_parser OK")
PY

echo "Done. Activate with: conda activate $ENV_NAME"
