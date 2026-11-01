FROM python:3.11.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-api.txt /tmp/requirements-api.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --location --retry 5 --retry-all-errors --retry-delay 2 --continue-at - \
       --output /tmp/torch-2.2.2+cpu-cp311-cp311-linux_x86_64.whl \
       'https://download.pytorch.org/whl/cpu/torch-2.2.2%2Bcpu-cp311-cp311-linux_x86_64.whl' \
    && echo '90089cae572672fb449c8ff1dc1b29daaffa117bf97ede7463dcd2fd1b991e4c  /tmp/torch-2.2.2+cpu-cp311-cp311-linux_x86_64.whl' | sha256sum --check - \
    && python -m pip install --no-cache-dir /tmp/torch-2.2.2+cpu-cp311-cp311-linux_x86_64.whl \
    && rm /tmp/torch-2.2.2+cpu-cp311-cp311-linux_x86_64.whl \
    && python -m pip install --no-cache-dir -r /tmp/requirements-api.txt

# Keep the package layout and the two legacy import modules used by inference.
COPY src/ ./src/
COPY test_attention.py ./test_attention.py
COPY runtime-artifacts/ /opt/runtime-artifacts/

ENV TETOUAN_RUN_NAME=tetouan_t4_20261101_v1 \
    TETOUAN_ARTIFACT_DIR=/opt/runtime-artifacts/data/processed/tetouan \
    TETOUAN_EXPERIMENT_DIR=/opt/runtime-artifacts/results/tetouan/tetouan_t4_20261101_v1 \
    TETOUAN_CHECKPOINT_DIR=/opt/runtime-artifacts/checkpoints/tetouan \
    TETOUAN_LATEST_EXPERIMENT_ID=extended_training_v2 \
    TETOUAN_LATEST_EXPERIMENT_DIR=/opt/runtime-artifacts/results/tetouan/extended_training_v2 \
    TETOUAN_LATEST_CHECKPOINT_DIR=/opt/runtime-artifacts/checkpoints/tetouan/extended_training_v2 \
    TETOUAN_LATEST_EVALUATION_DIR=/opt/runtime-artifacts/results/tetouan/extended_training_v2/evaluation_test_v1 \
    TETOUAN_DEVICE=cpu

RUN groupadd --system app && useradd --system --gid app --home-dir /app app \
    && chown -R app:app /app
USER app

EXPOSE 10000
CMD ["sh", "-c", "exec python -m uvicorn src.api.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1"]
