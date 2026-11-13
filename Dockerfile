# fasterRag API and worker image.
#
# One image serves both roles: `fasterrag serve` and `fasterrag worker` are the same package
# with different entry points, and shipping two images that must stay in lockstep is a
# version-skew bug waiting to be filed.
#
# Built in stages so the runtime layer carries no compiler, no build backend, no installer,
# and no package index cache. The image always carries BASE_EXTRAS, what a served deployment
# of the default config.yaml needs: `server` (FastAPI, uvicorn), `qdrant`, and the `pdf`,
# `docx` and `html` parsers, which were the core install until TASK-0444 made them extras. The
# `.[all]` extras are deliberately NOT installed: they pull a multi-gigabyte deep-learning
# runtime that a hosted-provider deployment never loads. Build with
# `--build-arg EXTRAS=huggingface` (or `all`) when local models are wanted.

# Both base images are pinned by digest, with the tag kept beside each for a human. A tag is a
# moving target: `python:3.12-slim` is rebuilt for every Debian and CPython patch, so the same
# Dockerfile produced a different base months apart. The digest is the multi-platform index,
# so the pin holds on arm64 as well as amd64. Dependabot (.github/dependabot.yml) reads these
# two lines and opens a pull request when the tag moves, which is what keeps a pin from
# becoming a way of never taking a security fix.
#
# python:3.12-slim — 3.12.16-slim-trixie, resolved 2026-11-12
FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928 AS base

# ghcr.io/astral-sh/uv:0.12.3 — resolved 2026-11-12. A stage of its own rather than a
# `COPY --from=<image>` so the pin sits on a FROM line, the only place Dependabot looks.
FROM ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 AS uv


FROM base AS builder

ARG EXTRAS=""
ARG BASE_EXTRAS="server,qdrant,pdf,docx,html"

# CRITICAL: UV_PYTHON_DOWNLOADS=never. Without it uv may fetch its own interpreter into the
# builder, and the virtual environment copied into the runtime stage would point at a Python
# that stage does not have.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /build

# The dependency layer: exactly what uv.lock names, hash-checked, nothing resolved here.
# `--locked` fails the build when uv.lock no longer matches pyproject.toml instead of quietly
# relocking, so the image cannot carry a dependency set no gate has seen. Until 2026-11-12
# this was `pip install .`, which resolved pyproject.toml afresh on every build: the same
# commit built on two days could ship two different sets of packages.
#
# Copied before the source so a source-only change reuses this layer. EXTRAS is a
# comma-separated list of extras (`huggingface`, `huggingface,redis`, `all`), installed on top
# of BASE_EXTRAS.
COPY pyproject.toml uv.lock ./
RUN extras=""; \
    for extra in $(echo "${BASE_EXTRAS},${EXTRAS}" | tr ',' ' '); do extras="${extras} --extra ${extra}"; done; \
    uv sync --locked --no-dev --no-editable --no-install-project ${extras}

# CRITICAL: .env.example is a build input, not documentation. The wheel target force-includes
# it (pyproject.toml [tool.hatch.build.targets.wheel.force-include]), so omitting it here
# fails the build with "Forced include not found".
#
# The project itself is built here by its build backend (hatchling), which uv.lock does not
# pin: a lockfile records what is installed, not what builds it. The backend never reaches
# the runtime stage; only the wheel it produced does.
COPY README.md .env.example config.yaml ./
COPY src/ ./src/
RUN extras=""; \
    for extra in $(echo "${BASE_EXTRAS},${EXTRAS}" | tr ',' ' '); do extras="${extras} --extra ${extra}"; done; \
    uv sync --locked --no-dev --no-editable ${extras}


FROM base AS runtime

LABEL org.opencontainers.image.title="fasterrag" \
      org.opencontainers.image.source="https://github.com/balrajsingh165/FasterRag" \
      org.opencontainers.image.licenses="Apache-2.0"

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
