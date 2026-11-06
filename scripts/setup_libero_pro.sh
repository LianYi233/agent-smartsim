#!/usr/bin/env bash
# Install LIBERO + the LIBERO-PRO fork for the control arm (Linux + NVIDIA GPU).
#
#   bash scripts/setup_libero_pro.sh [conda-env-name]
#
# Notes:
#   - datasets/demos are NOT needed for agent evaluation; tasks + init states
#     ship inside the LIBERO-PRO repo itself.
#   - If github.com is unreachable from your network, set GITHUB_MIRROR, e.g.
#       GITHUB_MIRROR=https://mirror.ghproxy.com bash scripts/setup_libero_pro.sh
set -euo pipefail

ENV_NAME="${1:-agent-libero}"
LIBERO_PRO_DIR="${LIBERO_PRO_DIR:-$HOME/LIBERO-PRO}"
MIRROR="${GITHUB_MIRROR:-}"

have() { command -v "$1" >/dev/null 2>&1; }

# 1) python environment
if have conda; then
  conda env list | grep -q "^${ENV_NAME} " || conda create -n "$ENV_NAME" python=3.10 -y
  PY="$(conda info --base)/envs/$ENV_NAME/bin/python"
else
  PY="$(command -v python3 || true)"
  [ -n "$PY" ] || { echo "error: need conda or python3" >&2; exit 1; }
  echo "conda not found; using $PY"
fi
"$PY" -m pip install -q -U pip

# 2) simulator stack (versions LIBERO pins)
"$PY" -m pip install -q "robosuite==1.4.0" bddl

# 3) the LIBERO-PRO fork: standard suites + perturbation suites, one package
if [ ! -d "$LIBERO_PRO_DIR" ]; then
  URL="https://github.com/Zxy-MLlab/LIBERO-PRO.git"
  [ -n "$MIRROR" ] && URL="${MIRROR%/}/https://github.com/Zxy-MLlab/LIBERO-PRO.git"
  echo "==> cloning LIBERO-PRO into $LIBERO_PRO_DIR"
  git clone "$URL" "$LIBERO_PRO_DIR"
fi
"$PY" -m pip install -q -e "$LIBERO_PRO_DIR"

# 4) sanity: the benchmark registry must answer before anything else matters
"$PY" - <<'PY'
from libero.libero import benchmark
d = benchmark.get_benchmark_dict()
perturbed = sorted(n for n in d if "_with_" in n or "_trigger" in n)
print(f"{len(d)} suites registered; {len(perturbed)} are LIBERO-PRO perturbation suites")
print("  standard:", ", ".join(sorted(n for n in d if n in
      ("libero_spatial", "libero_object", "libero_goal", "libero_10"))))
print("  perturbed (first 6):", ", ".join(perturbed[:6]), "...")
assert "libero_spatial" in d and "libero_object" in d
PY

cat <<EOF

install OK. next steps:
  conda activate $ENV_NAME
  export MUJOCO_GL=egl                       # headless rendering on the 4090 box
  agent-smartsim probe --suite libero_object --list
  agent-smartsim probe --suite libero_object --camera-size 128 --gpu 0
EOF
