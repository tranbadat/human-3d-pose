#!/usr/bin/env bash
# Fetch Human3.6M pre-processed labels (VideoPose3D distribution).
#
# DSVTformer's data loader (common/Mydataset_img.py) consumes:
#   - data_2d_h36m_cpn_ft_h36m_dbb.npz   CPN 2D keypoints (default)
#   - data_3d_h36m.npz                   3D ground-truth
#
# Both files originate from facebookresearch/VideoPose3D. Per their
# DATASETS.md, the 3D file must be reconstructed from the official
# Human3.6M CDF release (EULA-gated at vision.imar.ro/human3.6m), and
# the 2D CPN file is hosted on Google Drive but the link rotates, so
# we cannot hard-code an ID here.
#
# This script:
#   1) prints the canonical source URLs,
#   2) attempts to download via $H36M_2D_URL / $H36M_3D_URL if you
#      export them yourself (signed Drive link, S3 mirror, etc.),
#   3) verifies the files end up under ./dataset/.
set -euo pipefail

DEST="${DEST:-./dataset}"
mkdir -p "${DEST}"

cat <<'EOF'
================================================================
  Human3.6M (pre-processed) — manual step required
================================================================
1. data_2d_h36m_cpn_ft_h36m_dbb.npz
     Source: https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md
     ("Human3.6M" -> "Setup from preprocessed dataset")

2. data_3d_h36m.npz
     Build with VideoPose3D's data/prepare_data_h36m.py from the
     official Human3.6M CDF files (EULA at vision.imar.ro/human3.6m).
     Or use a community mirror that you trust.

To auto-download here, export the URLs first, then re-run:
    export H36M_2D_URL="https://drive.google.com/uc?id=<your_id>"
    export H36M_3D_URL="https://<mirror>/data_3d_h36m.npz"
    bash scripts/download_h36m.sh

Otherwise, drop the two .npz files into ./dataset/ manually.
================================================================
EOF

if [[ -n "${H36M_2D_URL:-}" ]]; then
    echo "[h36m] downloading 2D CPN keypoints..."
    if [[ "${H36M_2D_URL}" == *"drive.google.com"* ]]; then
        command -v gdown >/dev/null || pip install -q gdown
        gdown "${H36M_2D_URL}" -O "${DEST}/data_2d_h36m_cpn_ft_h36m_dbb.npz"
    else
        curl -L "${H36M_2D_URL}" -o "${DEST}/data_2d_h36m_cpn_ft_h36m_dbb.npz"
    fi
fi

if [[ -n "${H36M_3D_URL:-}" ]]; then
    echo "[h36m] downloading 3D ground-truth..."
    if [[ "${H36M_3D_URL}" == *"drive.google.com"* ]]; then
        command -v gdown >/dev/null || pip install -q gdown
        gdown "${H36M_3D_URL}" -O "${DEST}/data_3d_h36m.npz"
    else
        curl -L "${H36M_3D_URL}" -o "${DEST}/data_3d_h36m.npz"
    fi
fi

echo
echo "[h36m] files now in ${DEST}:"
ls -lh "${DEST}"/*.npz 2>/dev/null || echo "  (none yet — see instructions above)"
