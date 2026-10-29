#!/usr/bin/env bash
# Dataset downloader for this project, targeting an NFS/local volume (default /data1/datasets).
#
# Uses the huggingface_hub PYTHON API, not the CLI: huggingface-cli was removed in
# huggingface_hub 1.0 (replaced by `hf`), so any script that shells out to it breaks
# on a fresh install. snapshot_download() is the stable surface.
#
# Sizes measured 2026-10-29 via the HF API:
#   TeleEmbodied/SMART-Data              2,028.76 GB total  -> NEVER download whole; cherry-pick zips
#   physical-intelligence/libero            34.94 GB        -> original LIBERO hdf5 demos
#   yifengzhu-hf/LIBERO-datasets           100.44 GB        -> LIBERO full bundle (incl. videos)
#   lerobot/libero + lerobot/libero-assets   1.94 GB + few  -> LeRobot format + env assets
#   lerobot/libero_{spatial,object,goal,10}_image          -> 6.6 / 9.2 / 6.3 / 12.8 GB
#
# huggingface.co is unreachable from some networks; the mirror is the default.
# Usage:
#   scripts/download_datasets.sh libero-hdf5
#   scripts/download_datasets.sh libero-lerobot
#   scripts/download_datasets.sh smart-tasks Franka_single/atomic/left_open_laptop_lid.zip
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/download_datasets.sh <target> [args]
  libero-hdf5                     original LIBERO hdf5 demos (~35 GB)
  libero-lerobot                  LeRobot demos + env assets (~40 GB)
  smart-tasks [task.zip ...]      cherry-pick SMART-Data zips (repo total 2 TB!)
env:
  DATASETS_DIR=/data1/datasets    default target
  HF_ENDPOINT=...                 default https://hf-mirror.com
  HF_HOME=...                     default $DATASETS_DIR/.hf_home (keeps cache off $HOME)
  PYTHON=...                      force a specific interpreter (must import huggingface_hub)

NOTE: SMART-Sim itself (simulator + scene/object assets) is NOT released yet;
the upstream assets/ dir only holds preview images and a
"dataset-links-coming-soon" placeholder. Datasets give you demonstrations, not
a closed-loop environment.
EOF
}

case "${1:-help}" in
  help|-h|--help|"") usage; exit 0 ;;
esac

DATASETS_DIR="${DATASETS_DIR:-/data1/datasets}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$DATASETS_DIR/.hf_home}"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

find_python() { # a python that can already import huggingface_hub
  local c
  for c in ${PYTHON:-} python3 python; do
    [ -n "$c" ] || continue
    command -v "$c" >/dev/null 2>&1 || continue
    if "$c" -c "import huggingface_hub" >/dev/null 2>&1; then
      printf '%s\n' "$c"; return 0
    fi
  done
  return 1
}

HF_PY="$(find_python || true)"
if [ -z "$HF_PY" ]; then
  BASE_PY="${PYTHON:-$(command -v python3 || command -v python || true)}"
  [ -n "$BASE_PY" ] || die "no python found; install python3 first"
  echo "==> huggingface_hub missing in '$BASE_PY'; installing"
  "$BASE_PY" -m pip install -U huggingface_hub hf_transfer \
    || die "pip install failed; if this is a managed env, run: pip install --user huggingface_hub hf_transfer"
  HF_PY="$(find_python || true)"
  [ -n "$HF_PY" ] || die "installed but still not importable. Retry with PYTHON=/path/to/python $0 $*"
fi
echo "python: $HF_PY  (huggingface_hub $("$HF_PY" -c 'import huggingface_hub as h; print(h.__version__)'))"

# hf_transfer only when actually importable, otherwise hub raises at download time.
if "$HF_PY" -c "import hf_transfer" >/dev/null 2>&1; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
else
  unset HF_HUB_ENABLE_HF_TRANSFER || true
fi

echo "target: $DATASETS_DIR   endpoint: $HF_ENDPOINT   cache: $HF_HOME"
if [ -d "$DATASETS_DIR" ]; then
  df -h "$DATASETS_DIR" | tail -1
else
  echo "creating $DATASETS_DIR"
  mkdir -p "$DATASETS_DIR" || die "cannot create $DATASETS_DIR (permissions? try: sudo mkdir -p $DATASETS_DIR && sudo chown \$USER $DATASETS_DIR)"
fi
mkdir -p "$HF_HOME"

download() { # <repo> <local-dir> [allow_pattern ...]
  local repo="$1" dir="$2"; shift 2
  local patterns=("$@")

  if [ "$repo" = "TeleEmbodied/SMART-Data" ] && [ "${#patterns[@]}" -eq 0 ]; then
    die "refusing to download TeleEmbodied/SMART-Data in full (2.03 TB).
       pass task zips instead, e.g.:
         $0 smart-tasks Franka_single/atomic/left_open_laptop_lid.zip"
  fi

  if [ "${#patterns[@]}" -gt 0 ]; then
    echo "==> $repo -> $dir"
    printf '    pattern: %s\n' "${patterns[@]}"
  else
    echo "==> $repo -> $dir  (full repo)"
  fi

  mkdir -p "$dir"
  "$HF_PY" - "$repo" "$dir" ${patterns[@]+"${patterns[@]}"} <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, local_dir, *patterns = sys.argv[1:]
path = snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=patterns or None,
    max_workers=8,
)
print("    done:", path)
PY
}

case "$1" in
  libero-hdf5)
    # Original LIBERO demonstration hdf5s (the copy most VLA repos use).
    download physical-intelligence/libero "$DATASETS_DIR/libero/hdf5"
    ;;
  libero-lerobot)
    # LeRobot-format demos per suite. NOTE: assets are required to RUN the envs.
    download lerobot/libero-assets        "$DATASETS_DIR/libero/assets"
    download lerobot/libero_spatial_image "$DATASETS_DIR/libero/lerobot/spatial"
    download lerobot/libero_object_image  "$DATASETS_DIR/libero/lerobot/object"
    download lerobot/libero_goal_image    "$DATASETS_DIR/libero/lerobot/goal"
    download lerobot/libero_10_image     "$DATASETS_DIR/libero/lerobot/libero10"
    ;;
  smart-tasks)
    # Cherry-pick SMART-Data task zips (1-12 GB each). The repo is 2 TB in total.
    shift
    if [ "$#" -eq 0 ]; then
      set -- "Franka_single/atomic/left_open_laptop_lid.zip"   # hinge,  ~1.1 GB
      # one per kinematic family, add as needed:
      #   Franka_single/atomic/left_press_down_dispenser.zip     (press,  ~1.9 GB)
      #   Franka_single/atomic/left_turn_faucet_handle_left.zip  (rotate, ~2.2 GB)
    fi
    download TeleEmbodied/SMART-Data "$DATASETS_DIR/SMART-Data" "$@"
    ;;
  *)
    usage; exit 2 ;;
esac
