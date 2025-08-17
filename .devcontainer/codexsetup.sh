# .devcontainer/codexsetup.sh
#!/usr/bin/env bash

###############################################################################
# Helper: run apt only once and keep the image lean
###############################################################################
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    binutils sudo build-essential bzr curl \
    default-libmysqlclient-dev dnsutils gettext \
    git git-lfs gnupg2 inotify-tools iputils-ping jq \
    libbz2-dev libc6 libc6-dev libcurl4-openssl-dev libdb-dev libedit2 \
    libffi-dev libgcc-13-dev libgcc1 libgdbm-compat-dev libgdbm-dev \
    libgdiplus libgssapi-krb5-2 liblzma-dev libncurses-dev \
    libncursesw5-dev libnss3-dev libpq-dev libpsl-dev libpython3-dev \
    libreadline-dev libsqlite3-dev libssl-dev libstdc++-13-dev libunwind8 \
    libuuid1 libxml2-dev libz3-dev make moreutils netcat-openbsd \
    openssh-client pkg-config protobuf-compiler python3-pip ripgrep rsync \
    software-properties-common sqlite3 swig3.0 tk-dev tzdata unixodbc-dev \
    unzip uuid-dev xz-utils zip zlib1g zlib1g-dev ripgrep
sudo rm -rf /var/lib/apt/lists/*

###############################################################################
# Node (you already get Node 22 from the devcontainer *feature*)
###############################################################################
# Install the Codex CLI and anything else you need
npm i -g @openai/codex @google/gemini-cli


echo "✔ Codex system layer installed."
