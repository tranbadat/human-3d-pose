# DSVTformer — RTX 6000 (96GB) training image
# CUDA 11.1 + PyTorch 1.8.0 per upstream README
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3-pip \
        git wget ca-certificates ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && ln -sf /usr/bin/python3.8 /usr/bin/python3

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
        torch==1.8.0+cu111 torchvision==0.9.0+cu111 \
        -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install \
        numpy==1.23.5 \
        einops \
        timm==0.4.12 \
        tqdm \
        fvcore \
        pytz \
        opencv-python \
        tensorboard \
        matplotlib \
        pillow

WORKDIR /workspace/DSVTformer
COPY . /workspace/DSVTformer

ENV PYTHONPATH=/workspace/DSVTformer

HEALTHCHECK --interval=30s --timeout=10s --retries=2 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to container'" || exit 1

CMD ["bash"]
