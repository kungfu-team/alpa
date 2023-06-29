#!/usr/bin/env -S sh -c 'docker build --rm -t kungfu.azurecr.io/mw-alpa-local-repo:latest -f $0 .'

FROM kungfu.azurecr.io/mw-alpa:latest

RUN pip install \
    numpy==1.22 \
    wrapt==1.11.0 \
    clu \
    tensorflow \
    tensorflow-datasets \
    orbax-checkpoint

WORKDIR /workspace

ADD . /workspace/alpa

WORKDIR /workspace/alpa
