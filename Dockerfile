FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglu1-mesa-dev \
    cmake \
    curl \
    git

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

ENV XLA_PYTHON_CLIENT_ALLOCATOR platform
ENV BOP_DATA_DIR /bop_data

RUN git clone https://github.com/NVlabs/nvdiffrast.git /root/nvdiffrast
RUN cp /root/nvdiffrast/docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
RUN rm -rf /root/nvdiffrast

ADD https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh /Miniconda3-latest-Linux-x86_64.sh
RUN bash /Miniconda3-latest-Linux-x86_64.sh -b && rm -f /Miniconda3-latest-Linux-x86_64.sh 
RUN /root/miniconda3/bin/conda install mamba -n base -c conda-forge
ADD environment.yml /environment.yml
RUN /root/miniconda3/bin/mamba env create -f /environment.yml
RUN /root/miniconda3/bin/conda init bash

SHELL ["/root/miniconda3/bin/conda", "run", "-n", "threednel", "/bin/bash", "-c"]
RUN pip install --upgrade pip
RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install ninja
COPY . /root/threednel/
RUN pip install -e /root/threednel
RUN git clone https://github.com/nishadgothoskar/pararender.git /root/pararender
RUN pip install -e /root/pararender
ENTRYPOINT ["/root/threednel/entrypoint.sh"]
