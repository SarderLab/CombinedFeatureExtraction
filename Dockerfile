FROM rocker/r-ubuntu:20.04

LABEL maintainer="Anish Tatke - Sarder Lab. <anish.tatke@ufl.edu>"

ENV DEBIAN_FRONTEND=noninteractive

# System deps for building Python + your usual deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl wget git unzip \
      build-essential make \
      libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
      libffi-dev liblzma-dev tk-dev \
      libncursesw5-dev xz-utils \
      libcurl4-openssl-dev libexpat1-dev libhdf5-dev \
      libxml2-dev libxslt1-dev \
      ffmpeg libsm6 libxext6 \
      libtool pkg-config autoconf automake cmake \
      libmemcached-dev memcached \
    && rm -rf /var/lib/apt/lists/*

# --- Install Python 3.12 from source ---
ENV PYTHON_VERSION=3.12.8
RUN curl -fsSLO https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd / && rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

WORKDIR /

# --- Make python/python3 point to Python 3.12  ---
RUN ln -sf /usr/local/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/local/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.12 /usr/bin/pip3 && \
    ln -sf /usr/local/bin/pip3.12 /usr/bin/pip

# --- Install pip in a stable way (no get-pip.py needed) ---
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

ENV build_path=/build
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

ENV fe_path=/opt/FExtract
RUN mkdir -p $fe_path

RUN apt-get update && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . $fe_path/
WORKDIR $fe_path

# Upgrade setuptools, as the version in Conda won't upgrade cleanly unless it is ignored.
RUN python -m pip install --no-cache-dir --no-build-isolation . && \
    python -m pip freeze > /tmp/requirements.txt && \
    rm -rf /root/.cache/pip/*

# Show what was installed
RUN which python && python --version && python -m pip --version

# Define entrypoint through which all CLIs can be run
LABEL entry_path=$fe_path/fextract/cli
WORKDIR $fe_path/fextract/cli

# Test our entrypoint.  If we have incompatible versions of numpy and
# Openslide, one of these will fail
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli && \
    python -m slicer_cli_web.cli_list_entrypoint PathomicsFE --help

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
