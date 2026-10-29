#!/usr/bin/env bash
# Best-effort LIBERO control-arm setup (the third-party stack drifts; pin versions).
# Usage: bash scripts/setup_libero.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-agent-libero}"
PY_VER="${PY_VER:-3.10}"

echo "==> creating conda env ${ENV_NAME} (python ${PY_VER})"
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda first, or use your own venv and skip this script." >&2
  exit 1
fi
conda create -y -n "${ENV_NAME}" "python=${PY_VER}"

# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "==> installing deps (pin these to whatever your base paper used)"
pip install -e ".[api]"
pip install robosuite==1.4.0 bddl
pip install libero

echo "==> fetching LIBERO datasets (~ tens of GB; pick the suites you need)"
python - <<'PY'
from libero.libero import get_libero_path  # noqa: F401
print("libero import ok")
PY

cat <<'NOTE'

Next:
  1. implement EnvBinding in src/agent_smartsim/envs/bindings/libero_binding.py
  2. point configs/experiment_libero.json at it (copy experiment_smartsim.example.json)
  3. start with LIBERO-Spatial / -Object pick-and-place, 3 tasks x 20 rollouts,
     then repeat on LIBERO-PRO perturbations for the robustness number.

Reminder: report the standard-LIBERO and LIBERO-PRO numbers SEPARATELY.
Standard LIBERO is saturated (>90% for strong VLAs); LIBERO-PRO is where
agents and policies actually separate.
NOTE
