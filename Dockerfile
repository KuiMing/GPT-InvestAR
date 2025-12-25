FROM apache/airflow:2.6.3-python3.9


USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends xvfb \
 && apt-get install -y build-essential gcc g++ make \
 && apt-get install -y cmake \
 && apt-get install -y python3.9-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

 RUN apt-get update && apt-get install -y --no-install-recommends \
 build-essential \
 cmake \
 ninja-build \
 python3-dev \
 git \
 pkg-config \
 libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp-dev \
    libpthread-stubs0-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/airflow/requirements.txt

ENV CMAKE_ARGS="-DCMAKE_CXX_FLAGS=-pthread -DCMAKE_EXE_LINKER_FLAGS=-lpthread"
ENV FORCE_CMAKE=1
ENV LDFLAGS="-lpthread"

RUN python3.9 -m pip install -U pip setuptools wheel
RUN pip3.9 install -r /opt/airflow/requirements.txt
RUN python3.9 -m pip install "llama-cpp-python" --prefer-binary || \
    (export CMAKE_ARGS="-DCMAKE_CXX_FLAGS=-pthread -DCMAKE_EXE_LINKER_FLAGS=-lpthread" && \
     export LDFLAGS="-lpthread" && \
     python3.9 -m pip install "llama-cpp-python" --no-cache-dir)

USER airflow
