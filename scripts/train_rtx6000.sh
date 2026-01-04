#!/usr/bin/env bash
# Optimized training run for RTX 6000 Ada (96GB VRAM)
# Cost: ~40k VND/hr — make every GPU-hour count.
set -euo pipefail

ROOT_PATH="${ROOT_PATH:-/data/dataset/}"
IMAGE_ROOT="${IMAGE_ROOT:-/data/Human3.6M/images/}"

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
  --batch_size 384 \
  --workers 16 \
  --nepoch 60 \
  --lr 0.0004 \
  --lr_decay 0.98 \
  --large_decay_epoch 5 \
  --lambda_vel 0.5 \
  --lambda_acc 0.1 \
  --gpu 0
