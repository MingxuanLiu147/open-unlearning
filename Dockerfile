FROM nvcr.io/nvidia/pytorch:24.09-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/open-unlearning

COPY requirements.txt .
ENV DS_BUILD_OPS=0
RUN pip install --no-cache-dir \
    huggingface-hub==0.29.1 \
    transformers==4.45.1 \
    hydra-core==1.3 \
    hydra_colorlog==1.2.0 \
    datasets==3.0.1 \
    accelerate==0.34.2 \
    bitsandbytes==0.44.1 \
    rouge-score==0.1.2 \
    tensorboard==2.18.0 \
    deepspeed==0.15.4 \
    "peft>=0.10.0" \
    pyreft==0.0.7 \
    lm-eval==0.4.8

COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

ENV PYTHONPATH=/workspace/open-unlearning:$PYTHONPATH

RUN mkdir -p data saves

CMD ["bash"]
