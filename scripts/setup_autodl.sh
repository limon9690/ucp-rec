#!/usr/bin/env bash
# One-time setup on an AutoDL instance.
# Image: PyTorch 1.10.0 / Python 3.8 (ubuntu20.04) / CUDA 11.3
#
# Layout:
#   /root/ucp-rec                  code (system disk; git pull here)
#   /root/autodl-tmp/ucp-rec-data  raw CSVs + preprocessed files (data disk)
#   /root/autodl-tmp/ucp-rec-res   checkpoints and logs (data disk)
# The repo's data/<group> folders and res/ are symlinks into the data disk,
# so nothing large lives on the small system disk.
set -euo pipefail
cd "$(dirname "$0")/.."

# AutoDL academic network acceleration (speeds up GitHub); harmless if absent
[[ -f /etc/network_turbo ]] && source /etc/network_turbo || true

pip install -r requirements.txt

STORE=/root/autodl-tmp
mkdir -p "$STORE/ucp-rec-data"/{amazon,douban,epinions} "$STORE/ucp-rec-res"
for g in amazon douban epinions; do
  [[ -L data/$g ]] || { [[ -e data/$g ]] && { echo "data/$g exists and is not a symlink; move it first"; exit 1; }; ln -s "$STORE/ucp-rec-data/$g" data/$g; }
done
[[ -L res ]] || { [[ -e res ]] && { echo "res exists and is not a symlink; move it first"; exit 1; }; ln -s "$STORE/ucp-rec-res" res; }

python3 scripts/check_env.py
echo
echo "Next: upload the raw CSVs into $STORE/ucp-rec-data/<group>/, then"
echo "  bash scripts/prepare_data.sh"
