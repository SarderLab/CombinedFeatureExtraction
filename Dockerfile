# This Dockerfile is used to generate the docker image dsarchive/histomicstk
# This docker image includes the HistomicsTK python package along with its
# dependencies.
#
# All plugins of HistomicsTK should derive from this docker image

FROM python:3.11

LABEL maintainer="Sumanth Devarasetty and Sam Border - Computational Microscopy Imaging Lab. <sumanth.devarasetty@medicine.ufl.edu> <samuel.border@medicine.ufl.edu>"

RUN apt-get update && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    libexpat1-dev \
    unzip \
    libhdf5-dev \
    software-properties-common \
    libssl-dev \
    # Standard build tools \
    build-essential \
    cmake \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libmemcached-dev && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update ##[edited]

RUN apt-get install libxml2-dev libxslt1-dev -y

# Required for opencv-python (cv2)
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /
# Make Python3 the default and install pip.  Whichever is done last determines
# the default python version for pip.

##Make a specific version of python the default and install pip
#RUN rm -f /usr/bin/python && \
#    rm -f /usr/bin/python3 && \
#    ln `which python3.11` /usr/bin/python && \
#    ln `which python3.11` /usr/bin/python3 && \
#    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
#    python get-pip.py && \
#    rm get-pip.py && \
#    ln `which pip3` /usr/bin/pip
#
RUN which  python && \
    python --version

ENV build_path=$PWD/build
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Copying location of feature extraction scripts
ENV ftx_sc_path=$PWD/feature_ext_sc
RUN mkdir -p $ftx_sc_path

RUN apt-get update && \
    apt-get install -y --no-install-recommends memcached && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . $ftx_sc_path/
WORKDIR $ftx_sc_path

#   Upgrade setuptools, as the version in Conda won't upgrade cleanly unless it
# is ignored.

# Installing packages in setup.py
RUN pip install --no-cache-dir --upgrade --ignore-installed pip setuptools && \
    pip install --no-cache-dir 'large-image[memcached]' && \
    pip install --no-cache-dir .  --find-links https://girder.github.io/large_image_wheels && \
    rm -rf /root/.cache/pip/*

# Show what was installed
RUN python --version && pip --version && pip freeze

# pregenerate font cache
#RUN python -c "from matplotlib import pylab"

# define entrypoint through which all CLIs can be run
WORKDIR $ftx_sc_path/feature_ext_sc/cli

# Test our entrypoint.  If we have incompatible versions of numpy and
# openslide, one of these will fail
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint Ftx_sc --help


ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]