FROM python:3.13-slim AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

ENV CMAKE_BUILD_PARALLEL_LEVEL=2 \
    MAKEFLAGS="-j2"

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p generated

ENV OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
