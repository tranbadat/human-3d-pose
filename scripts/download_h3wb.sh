#!/usr/bin/env bash
# Fetch H3WB pre-processed Human3.6M data used by mini_train_h3wb.py.
#
# Source: https://github.com/wholebody3d/wholebody3d
# File:   train_data.npy  (~234 MB, S1/S5/S6/S7, 4 cameras, COCO-WholeBody 133 kp)
# Drive:  1LZh4Jsg3_ZKBF0iEPiexzoGHE4srLgfC
set -euo pipefail

DEST="${DEST:-./dataset}"
FILE_ID="${H3WB_FILE_ID:-1LZh4Jsg3_ZKBF0iEPiexzoGHE4srLgfC}"
OUT="${DEST}/train_data.npy"

mkdir -p "${DEST}"

if [[ -f "${OUT}" ]]; then
    echo "[h3wb] already present: ${OUT} ($(du -h "${OUT}" | cut -f1))"
    exit 0
fi

command -v gdown >/dev/null || pip install -q gdown

echo "[h3wb] downloading train_data.npy from Drive id=${FILE_ID} ..."
gdown "${FILE_ID}" -O "${OUT}"

echo "[h3wb] done:"
ls -lh "${OUT}"
