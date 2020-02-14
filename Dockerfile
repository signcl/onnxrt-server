FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
SHELL ["/bin/bash", "-xc"]

ENV DEBIAN_FRONTEND=noninteractive

RUN true && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y locales python3 python3-dev python3-distutils curl libev-dev && \
    locale-gen zh_CN.UTF-8 && \
    python3 <(curl https://bootstrap.pypa.io/get-pip.py) && \
    pip3 install -U 'poetry>=1.0.0' 'keyrings.alt>=3.4.0' && \
    poetry config virtualenvs.create false && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    true

ENV TERM xterm
ENV LANG zh_CN.UTF-8
ENV LANGUAGE zh_CN:en
ENV LC_ALL zh_CN.UTF-8
ENV NVIDIA_VISIBLE_DEVICES void

ADD tini /usr/bin/tini
ADD src/poetry.lock /onnxrt/poetry.lock
ADD src/pyproject.toml /onnxrt/pyproject.toml
WORKDIR /onnxrt

RUN rm -rf .venv && \
    poetry install --no-dev && \
    mkdir /wheel && cd /wheel && \
    pip download 'onnxruntime>=1.1.1' && \
    pip download 'onnxruntime-gpu>=1.1.1'

ADD onnxrt-server /onnxrt-server
ADD src /onnxrt
