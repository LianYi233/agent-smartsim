#!/usr/bin/env bash
# Dataset downloader for this project, targeting an NFS/local volume (default /data1/datasets).
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

NOTE: SMART-Sim itself (simulator + scene/object assets) is NOT on the Hub yet;
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

if ! command -v huggingface-cli >/dev/null 2>&1; then
  if command -v pip >/dev/null 2>&1; then
    echo "==> installing huggingface_hub[cli] + hf_transfer"
    pip install -U "huggingface_hub[cli]" hf_transfer
  else
    echo "huggingface-cli not found and pip is unavailable; install it first:" >&2
    echo "  pip install -U 'huggingface_hub[cli]' hf_transfer" >&2
    exit 1
  fi
fi
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

echo "target: $DATASETS_DIR   endpoint: $HF_ENDPOINT"
if [ -d "$DATASETS_DIR" ]; then
  df -h "$DATASETS_DIR" | tail -1
else
  echo "creating $DATASETS_DIR"
  mkdir -p "$DATASETS_DIR" || { echo "cannot create $DATASETS_DIR (permissions?)" >&2; exit 1; }
fi

download() { # <repo> <local-dir> [extra hf args...]
  local repo="$1" dir="$2"; shift 2
  echo "==> $repo -> $dir"
  huggingface-cli download "$repo" --repo-type dataset --local-dir "$dir" "$@"
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
    download lerobot/libero_10_image      "$DATASETS_DIR/libero/lerobot/libero10"
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
    args=()
    for t in "$@"; do args+=(--include "$t"); done
    download TeleEmbodied/SMART-Data "$DATASETS_DIR/SMART-Data" "${args[@]}"
    ;;
  *)
    usage; exit 2 ;;
esac
