FROM rocker/r-ubuntu:22.04



LABEL maintainer="Sayat Mimar - Sarder Lab. <sayat.mimar@ufl.edu>"

RUN apt-get update && \
    apt-get install --yes --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get --yes --no-install-recommends -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" dist-upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    libexpat1-dev \
    unzip \
    libhdf5-dev \
    libpython3-dev \
    python3.11-dev \
    python3.11-distutils \
    python3-tk \
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

# Refresh OS CA certs so TLS verification against servers signed by newer
# CAs (e.g. InCommon RSA OV SSL CA 3) doesn't fail with an outdated bundle.
RUN apt-get update && \
    apt-get install --reinstall -y ca-certificates && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

RUN apt-get install libxml2-dev libxslt1-dev -y

WORKDIR /
# Make Python3 the default and install pip.  Whichever is done last determines
# the default python version for pip.

#Make a specific version of python the default and install pip
RUN rm -f /usr/bin/python && \
    rm -f /usr/bin/python3 && \
    ln -s $(which python3.11) /usr/bin/python && \
    ln -s $(which python3.11) /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    ln -s $(which pip3) /usr/bin/pip

RUN which  python && \
    python --version

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

ENV fe_path=/FExtract
RUN mkdir -p $fe_path

RUN apt-get update && \
    apt-get install -y --no-install-recommends memcached && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . $fe_path/
WORKDIR $fe_path

# Upgrade setuptools, as the version in Conda won't upgrade cleanly unless it
# is ignored.

RUN pip install --no-cache-dir --upgrade --ignore-installed pip setuptools && \
    pip install --no-cache-dir .  && \
    rm -rf /root/.cache/pip/*

# Show what was installed
RUN python --version && pip --version && pip freeze

# Define entrypoint through which all CLIs can be run
WORKDIR $fe_path/fextract/cli
LABEL entry_path=$fe_path/fextract/cli
# Test our entrypoint.  If we have incompatible versions of numpy and
# Openslide, one of these will fail
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint ClassicalFeatures --help
RUN python -m slicer_cli_web.cli_list_entrypoint ExpandedGranularFeatures --help

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
