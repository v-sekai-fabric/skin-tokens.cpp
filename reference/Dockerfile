FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libsm6 libx11-6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    omegaconf==2.3.0 safetensors==0.5.3 transformers==4.57.1 \
    scipy==1.15.3 lightning==2.5.2 diffusers==0.35.1 \
    einops==0.8.1 trimesh==4.7.4 open3d==0.19.0 \
    python-box==7.3.2 addict==2.4.0 bpy==4.5.4 \
    fast-simplification==0.1.12

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxi6 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
ENTRYPOINT ["python", "/work/reference/extract_safe.py"]
