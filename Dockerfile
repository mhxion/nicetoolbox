FROM docker.io/nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

# install make, git, ffmpeg + some essential stuff
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt update && \
    apt install -y build-essential software-properties-common \
    unzip git-all ffmpeg wget curl

# install python from deadsnakes and pip from get-pip
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.10 python3.10-venv python3.10-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# conda from miniforge
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
RUN wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh

# copy the repo + submodules
COPY . /nicetoolbox
WORKDIR /nicetoolbox

# do all toolbox installation steps
ARG NICETOOLBOX_DEV
RUN make create_machine_specifics && \
    make download_assets && \
    make download_dataset && \
    make install DEV=${NICETOOLBOX_DEV}

# for version tracking we need to get git repository metadata 
ARG NICETOOLBOX_GIT_HASH
ARG NICETOOLBOX_GIT_SUMMARY

# we set git hash and summary into env variables
# they will be resolved in runtime by toolbox 
ENV NICETOOLBOX_GIT_HASH=${NICETOOLBOX_GIT_HASH}
ENV NICETOOLBOX_GIT_SUMMARY=${NICETOOLBOX_GIT_SUMMARY}