#!/usr/bin/env bash
# Mini-train sanity: 5-10 epochs on a small subset to verify loss converges
# before paying for RTX 6000 time.
set -euo pipefail

ROOT_PATH="${ROOT_PATH:-./dataset/}"
IMAGE_ROOT="${IMAGE_ROOT:-./images/}"

python main_img.py \
  --root_path "${ROOT_PATH}" \
  --image_root_path "${IMAGE_ROOT}" \
  --model dsvtformer \
  --num_view 2 \
  --view_indices 0,1 \
  --frames 27 \
  --depth 2 \
  --embed_dim_ratio 32 \
  --img_embed_dim_ratio 16 \
  --batch_size 16 \
  --workers 2 \
  --nepoch 10 \
  --lr 0.0002 \
  --subset 0.02 \
  --lambda_vel 0.5 \
  --lambda_acc 0.1 \
  --gpu 0
