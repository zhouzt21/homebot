FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04 
# 11.3.0
ENV NVIDIA_DRIVER_CAPABILITIES all
ARG PYTHON_VERSION=3.8

# Install os-level packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    rsync \
    tmux \
    unzip \
    vim \
    vulkan-utils \
    wget \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install (mini) conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda install -y python="$PYTHON_VERSION" && \
    /opt/conda/bin/conda clean -ya
# https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# https://github.com/haosulab/ManiSkill/issues/9
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

# RUN pip install mani-skill2==0.5.0 && pip cache purge
# Install the engine instead
RUN pip install "numpy<1.24" \
    scipy \
    gymnasium>=0.28.1 \
    sapien==2.2.2 \
    h5py \
    pyyaml \
    tqdm \
    GitPython \
    tabulate \
    gdown>=4.6.0 \
    # vision
    transforms3d \
    opencv-python \
    imageio \
    imageio[ffmpeg] \
    trimesh \
    rtree
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install matplotlib pre-commit wandb
# mvp and learning stuff
RUN pip install huggingface-hub==0.17.3 \
    timm==0.9.7 \
    tokenizers==0.14.1 \
    transformers==4.34.0
#RUN pip install git+https://github.com/ir413/mvp

#COPY mae_pretrain_egosoup_vit_base.pth /tmp/mvp-download-cache/mae_pretrain_egosoup_vit_base.pth
# diffusion policy
RUN pip install diffusers==0.11.1
