############################
#  Hailo Dataflow-Compiler  #
#    + Codex build env      #
############################
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---- 1. system prerequisites ---------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        build-essential git curl graphviz graphviz-dev libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- 1b. Node 22 + Codex CLI (runs as root) -------------------
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash -     \
 && apt-get install -y nodejs                                      \
 && npm install -g @openai/codex                                   \
 && npm cache clean --force
# (NodeSource is the official way to install current Node on Ubuntu)  :contentReference[oaicite:2]{index=2}

# ---- 2. non-root user (safer) ---------------------------------
ARG USERNAME=hailo
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} \
 && useradd  -u ${UID} -g ${GID} -m -s /bin/bash ${USERNAME}
USER ${USERNAME}
WORKDIR /workspace

# ---- 3. Python virtual-env with DFC & Model Zoo ---------------
RUN python3.10 -m venv ~/hailo_venv
ENV PATH="/home/${USERNAME}/hailo_venv/bin:${PATH}"

RUN pip install --no-cache-dir hailo_dataflow_compiler-3.32.0-py3-none-linux_x86_64


RUN pip install --no-cache-dir -e .

# ---- 4. entrypoint -------------------------------------------
CMD ["bash"]
