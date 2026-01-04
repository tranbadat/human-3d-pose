#!/usr/bin/env bash
# Build-only smoke test for the production Dockerfile.
#
# On Apple Silicon / Mac this runs under x86_64 emulation (slow, ~10-15 min)
# but proves that:
#   - apt packages resolve
#   - PyTorch 1.8.0+cu111 + timm 0.4.12 wheels still install
#   - COPY layout is correct
#   - HEALTHCHECK + CMD parse without error
#
# It does NOT run training (no NVIDIA GPU on Mac). For runtime verification,
# build & run on the actual RTX 6000 host.
set -euo pipefail

IMAGE="${IMAGE:-dsvtformer:test}"
PLATFORM="${PLATFORM:-linux/amd64}"

if ! docker info >/dev/null 2>&1; then
    echo "[docker] daemon not running — open Docker Desktop first." >&2
    exit 1
fi

echo "[docker] building ${IMAGE} for ${PLATFORM} (emulated on Apple Silicon)..."
time docker build --platform "${PLATFORM}" -t "${IMAGE}" .

echo
echo "[docker] image layers:"
docker image inspect "${IMAGE}" --format '{{range .RootFS.Layers}}{{println .}}{{end}}' | head -20

echo
echo "[docker] image size:"
docker images "${IMAGE}" --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}'

echo
echo "[docker] entry-point smoke test (CMD=bash, no GPU expected):"
docker run --rm --platform "${PLATFORM}" "${IMAGE}" \
    python -c "import torch, timm, fvcore; print('torch', torch.__version__); print('timm', timm.__version__); print('fvcore OK')"

echo
echo "[docker] BUILD-ONLY TEST PASSED. Push this Dockerfile to the RTX 6000 host"
echo "         and run: docker run --gpus all ${IMAGE} bash scripts/train_rtx6000.sh"
