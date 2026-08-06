# fasterRag API and worker image.
#
# One image serves both roles: `fasterrag serve` and `fasterrag worker` are the same package
# with different entry points, and shipping two images that must stay in lockstep is a
# version-skew bug waiting to be filed.
#
# Built in two stages so the runtime layer carries no compiler, no build backend, and no
# package index cache. The `.[all]` extras are deliberately NOT installed: they pull a
# multi-gigabyte deep-learning runtime that a hosted-provider deployment never loads. Build
# with `--build-arg EXTRAS=huggingface` (or `all`) when local models are wanted.

ARG PYTHON_VERSION=3.12

# Pinned to a minor version, never `latest`, matching the rule in docs/deployment.md §1.
# A digest would be stronger still — a tag is a moving target, so the same Dockerfile can
# produce a different base months apart — but a digest has to be refreshed deliberately, and
# that belongs with the supply-chain work in TASK-0158 rather than hard-coded here.
FROM python:${PYTHON_VERSION}-slim AS builder

ARG EXTRAS=""

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Copied before the source so a source-only change does not re-resolve dependencies.
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY config.yaml ./

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && if [ -n "$EXTRAS" ]; then \
         /opt/venv/bin/pip install ".[${EXTRAS}]"; \
       else \
         /opt/venv/bin/pip install .; \
       fi


FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTERRAG_CONFIG=/app/config.yaml

# CRITICAL: non-root. This is an image fasterRag builds itself, so the rule in
# docs/deployment.md §1 applies — third-party images we merely provision keep their upstream
# user because forcing one onto an image that owns its storage volume can make it unstartable,
# but that reasoning does not extend to our own.
RUN useradd --create-home --uid 10001 fasterrag

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=fasterrag:fasterrag config.yaml /app/config.yaml

# Journal, traces, and caches live here. Declared as a volume so a container restart does not
# discard the ingestion journal that crash-resume depends on.
RUN mkdir -p /app/.fasterrag && chown -R fasterrag:fasterrag /app
VOLUME ["/app/.fasterrag"]

USER fasterrag

EXPOSE 8000

# Readiness rather than liveness: `/healthz` answers as soon as the process is up, while
# `/readyz` answers only once the vector database is reachable. A container reporting healthy
# before it can serve a query is a container a load balancer sends traffic to too early.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4).status == 200 else 1)"

ENTRYPOINT ["fasterrag"]
CMD ["serve", "--host", "0.0.0.0"]
