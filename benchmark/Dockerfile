FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get update && \
    apt-get install -y \
        python3 python3-pip \
        && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt giotto-deep/

RUN cd giotto-deep && \
    pip3 install --no-cache-dir --disable-pip-version-check -r requirements.txt

COPY ./benchmark/requirements.txt giotto-deep/requirements.txt

RUN cd giotto-deep && \
    pip3 install --no-cache-dir --disable-pip-version-check -r requirements.txt

COPY ./setup.py giotto-deep/
COPY ./setup.cfg giotto-deep/
COPY ./README.md giotto-deep/
COPY ./gdeep giotto-deep/gdeep/
COPY ./examples giotto-deep/examples/
COPY ./benchmark giotto-deep/benchmark/

RUN cd giotto-deep && pip3 install --no-cache-dir --disable-pip-version-check -e .

ENTRYPOINT [ "python3", "/giotto-deep/benchmark/benchmark.py" ]
