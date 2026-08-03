# syntax=docker/dockerfile:1.7
ARG PYTORCH_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM ${PYTORCH_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/src
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git libgl1 libglib2.0-0 ninja-build \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder
WORKDIR /build
COPY pyproject.toml requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip wheel --wheel-dir=/wheelhouse -r requirements.lock \
    && python -m pip wheel --wheel-dir=/wheelhouse usd-core==24.8

FROM base AS runtime
COPY --from=builder /wheelhouse /wheelhouse
COPY requirements.lock /tmp/requirements.lock
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links=/wheelhouse -r /tmp/requirements.lock \
    && rm -rf /wheelhouse /tmp/requirements.lock
WORKDIR /workspace
COPY docker/entrypoint.sh /usr/local/bin/tomato-entrypoint
RUN chmod +x /usr/local/bin/tomato-entrypoint
ENTRYPOINT ["tomato-entrypoint"]
CMD ["python", "-c", "import tomato_recon; print(tomato_recon.__version__)"]

FROM runtime AS dev
RUN --mount=type=cache,target=/root/.cache/pip pip install pytest==8.3.2 pytest-cov==5.0.0

FROM runtime AS release
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps -e .

FROM runtime AS usd-export
COPY --from=builder /wheelhouse /tmp/wheelhouse
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links=/tmp/wheelhouse usd-core==24.8 \
    && rm -rf /tmp/wheelhouse
