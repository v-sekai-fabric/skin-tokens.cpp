FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN python -m pip install --no-cache-dir \
    omegaconf==2.3.0 safetensors==0.5.3 transformers==4.57.1 \
    scipy==1.15.3 lightning==2.5.2 diffusers==0.35.1 \
    einops==0.8.1 trimesh==4.7.4 python-box==7.3.2 addict==2.4.0

WORKDIR /work
ENTRYPOINT ["python", "/work/reference/extract_safe.py"]
