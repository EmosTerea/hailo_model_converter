############################
#  Hailo Dataflow-Compiler #
#    + optional Node       #
############################
FROM nvidia/cuda:12.5.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# ---- 1. system prerequisites ---------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    build-essential git curl graphviz libgraphviz-dev python3-tk \
 && rm -rf /var/lib/apt/lists/*

# The CUDA images already configure the CUDA APT repo, so we can apt install:
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 libcudnn9-dev-cuda-12 \
 && rm -rf /var/lib/apt/lists/*

# ---- 2. non-root user (safer) --------------------------------
ARG USERNAME=hailo
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} \
 && useradd  -u ${UID} -g ${GID} -m -s /bin/bash ${USERNAME}
USER ${USERNAME}
WORKDIR /workspace

# ---- 3. Python virtual env -----------------------------------
RUN python3.10 -m venv /home/${USERNAME}/hailo_venv
ENV PATH="/home/${USERNAME}/hailo_venv/bin:${PATH}"

# Upgrade pip/setuptools in the venv (often avoids wheel issues)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ---- 3.5. Install Hailo wheel ----------------------------
COPY --chown=${USERNAME}:${USERNAME} hailo*.whl /home/${USERNAME}/wheels/
RUN pip install --no-cache-dir /home/${USERNAME}/wheels/hailo*.whl \
&& rm -rf /home/${USERNAME}/wheels

# ---- 4. entrypoint -------------------------------------------
CMD ["bash"]
