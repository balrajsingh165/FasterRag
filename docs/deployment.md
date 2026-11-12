# deployment.md — Self-Hosting, Docker, Sizing, Rollback

fasterRag is self-hosted software in beta (no managed cloud). Three ways to run it:

1. **Library mode** — embed in your own process ([python-api.md](python-api.md)). Install from a source checkout (`pip install -e ".[all]"`) — **`pip install fasterrag` does not work yet**, because nothing has been published to PyPI (TASK-0087, blocked on the license decision TASK-0164 and the version stamp TASK-0020).
2. **Service mode** — `fasterrag serve` (API). One `serve` process is the whole service: ingestion runs inside it, in the CPU and embedding pools that `workers.*` sizes. A separate worker process that takes ingestion load off the API is not built (TASK-0130). `serve` binds `127.0.0.1` unless `app.host` or `--host` says otherwise, and **refuses** to bind anything else with `security.auth` off unless `security.allow_unauthenticated: true` is set (§3). The dashboard has its own bind, `observability.dashboard_host`, loopback by default.
3. **Docker mode** — compose profiles for the full stack.

## 1. Compose profiles

| Profile | Containers | Enabled by |
|---|---|---|
| `core` | API (ingestion runs inside it) | always |
| `qdrant` | Qdrant (system-managed) | `vector_db.mode: docker` |
| `dashboard` | fasterRag read-only dashboard | `observability.dashboard: true` |
| *(langfuse)* | Langfuse v3 stack (web, worker, Postgres, ClickHouse, Redis, MinIO) | `observability.langfuse: true` — **provisioned, not in this file** |
| *(grafana)* | Grafana with provisioning-as-code mounts | `observability.grafana: true` — **provisioned, not in this file** |

`docker-compose.yml` ships the first three profiles. Langfuse and Grafana are deliberately **not** in it: both are provisioned by their config toggle (`observability.langfuse: true` writes and runs its own compose file under `.fasterrag/langfuse/`; Grafana starts on a user-defined network), and restating them here would create a second source of truth that drifts the first time a pinned version changes on one side. Turn the toggle on and run `fasterrag doctor`.

```console
$ docker compose --profile core --profile qdrant up -d
$ docker compose -f docker-compose.yml -f deploy/compose.medium.yml --profile core up -d
```

Sizing presets live in `deploy/compose.{small,medium,large}.yml` and map to the table in §4. They set resource limits and pass pipeline settings through `FASTERRAG_SET`, the environment-variable form of `--set` — a container has no command line to add flags to, and overriding `command:` in a layer file means restating it and keeping it in step by hand.

`config.yaml` ships `vector_db.host: localhost`, which is right for a process on the host and wrong inside a container. The base compose file therefore sets `FASTERRAG_SET: vector_db.host=qdrant` on every fasterRag service. Compose merges `environment` **per key**, so a layer file that sets `FASTERRAG_SET` replaces that value rather than adding to it — every preset repeats the host for that reason, and so must any override file you write. Point it at your own address when the database is external.

**Published ports bind to `127.0.0.1`, not `0.0.0.0`.** The API serves unauthenticated until `security.auth: true`, and the dashboard renders prompts and responses verbatim, so a stack that published on every interface would expose the corpus the moment it started. Put a TLS-terminating proxy in front and widen the binding deliberately.

> **`docker compose config` prints your secrets.** It renders `env_file` contents inline, so inspecting the stack dumps everything in `.env` to stdout — terminal scrollback, CI logs, pasted bug reports. Use `docker compose config --no-interpolate`, which leaves `${VAR}` unexpanded, whenever the output is going anywhere it will be kept.

The image fasterRag builds (`Dockerfile`) runs as uid 10001, carries no compiler or build backend in its runtime layer, and healthchecks against `/readyz` rather than `/healthz` — the latter answers as soon as the process is up, so a container reporting healthy on it gets traffic before it can serve a query. Containers whose images **fasterRag builds itself** run non-root; third-party images fasterRag provisions (Qdrant, the Langfuse stack, Grafana) run as their upstream images ship — forcing a `--user` onto an image that expects to own its storage volume risks an unstartable container (resolved AUDIT-0007, 2026-08-02). Images are pinned by tag (no `latest`); provisioning is idempotent, with doctor-gating shipped for Qdrant and pending for Langfuse/Grafana (TASK-0147/0149 in [todo.md](todo.md)).

It is built from the lockfile: `uv sync --locked` installs exactly what `uv.lock` names, and the build fails rather than re-resolving when `uv.lock` no longer matches `pyproject.toml`, so the same commit builds the same dependency set on any day. Both base images (`python:3.12-slim` and the image the `uv` binary is copied from) are pinned by digest, with the tag beside each; Dependabot proposes the new digest when a tag moves. `--build-arg EXTRAS=` takes a comma-separated list of extras (`huggingface`, `huggingface,redis`, `all`) and installs their locked versions too; an empty value (the default) installs none. Measured 2026-11-12, Docker 29.7.2 on Windows 11: the default build succeeds, `docker run --rm <image> --help` exits 0, `id -u` is 10001, and the 49 distributions in the image are exactly the versions `uv.lock` names; an `EXTRAS=redis,openai` build imports both.

### Running the image: the unauthenticated-bind refusal (TASK-0460, 2026-11-12)

The image runs `fasterrag serve --host 0.0.0.0` — inside a container a loopback bind is unreachable through any published port — and the `config.yaml` it carries has `security.auth: false`. Since TASK-0460 `serve` **refuses** a non-loopback bind with auth off unless `security.allow_unauthenticated: true` is set, so what a default container does now depends on how it is started:

| Started as | What happens | What an operator must change |
|---|---|---|
| `docker compose … up` with the shipped files | Starts. Every `FASTERRAG_SET` in `docker-compose.yml` and `deploy/compose.*.yml` carries `security.allow_unauthenticated=true`, because every port they publish is on the host's `127.0.0.1` | Nothing — but **an override file that sets `FASTERRAG_SET` must repeat the flag** (compose merges `environment` per key), or the container exits at start with `refusing to bind 0.0.0.0` and the healthcheck never passes |
| `docker run fasterrag` (bare) | **Exits with code 2**: `refusing to bind 0.0.0.0 with security.auth off`. It used to start and warn | Either turn auth on — mount a config with `security.auth: true` and pass `-e FASTERRAG_API_KEY=<16+ characters>` — or, for a loopback-only port mapping, `docker run -p 127.0.0.1:8000:8000 -e FASTERRAG_SET=security.allow_unauthenticated=true fasterrag` |
| Behind an authenticating reverse proxy | Same as the bare case | Set `security.allow_unauthenticated: true` deliberately, and keep the container's port off every network the proxy does not front |

The flag is a statement that something other than fasterRag stands between the port and the network. **Never keep it while publishing on a routable address** — remove the `127.0.0.1:` prefix from a `ports:` entry and you must turn auth on instead. The `large` preset drops the published port for a load balancer; put auth on before that load balancer faces anyone. With the flag set `serve` still prints a `WARNING` on every start, and with `security.auth: true` the flag does nothing.

The container `HEALTHCHECK` probes `/readyz` without a key. Its status code keeps its meaning under auth: `/readyz` stays unauthenticated and answers `200` or `503` for every caller; only the per-dependency detail is withheld from a caller without an `admin` key ([api-reference.md](api-reference.md)).

## 2. Vector DB deployment modes (Qdrant reference)

| Mode | Config | Notes |
|---|---|---|
| System-managed Docker | `vector_db.mode: docker` | fasterRag launches and manages the container. Storage on a **named Docker volume** — mandatory on Windows/WSL (bind mounts have known data-loss issues per Qdrant's install docs). |
| External, no Docker | `vector_db.mode: external`, `host: localhost` | User runs Qdrant themselves (binary, service, or their own container). |
| Remote | `vector_db.mode: external`, `host: <remote-ip>` | Qdrant on another machine. **Expose BOTH 6333 (REST) and 6334 (gRPC)** — clients attempting gRPC fail if only 6333 is reachable (Qdrant Discussion #2195). Set `QDRANT__SERVICE__API_KEY` on the server and put the key in fasterRag's `.env`. Use TLS/a private network for cross-host traffic. |

## 3. Remote connections & networking

| Port | Component | Exposure guidance |
|---|---|---|
| 8000 | fasterRag API | Behind a reverse proxy with TLS; auth on (`security.auth: true`). |
| 8080 | Dashboard | Internal network only; it is read-only but still shows prompts/responses. Bound by `observability.dashboard_host` (default `127.0.0.1`), not by `app.host`. |
| 6333 + 6334 | Qdrant REST + gRPC | Private network only; API-key auth. Both ports required. |
| 3000 | Langfuse web | The only publicly-exposed Langfuse component (by Langfuse's design); protect with its auth. |
| 3030, 5432, 6379, 8123/9000, 9090/9091 | Langfuse worker, Postgres, Redis, ClickHouse, MinIO | Never exposed publicly; compose-internal network. |

## 4. Sizing guidance (starting points, not measurements)

| Corpus scale | API tier | Pools (inside each API process) | Vector DB | Notes |
|---|---|---|---|---|
| ≤ 1M chunks (dev/small) | 1× (4 vCPU, 6 GB) | cpu_pool 4 / embed_pool 1 (CPU embeddings OK) | Qdrant 2 vCPU, 4 GB, SSD | Single machine fine. |
| 1M–50M chunks | 1× (12 vCPU, 20 GB) | cpu_pool = cores; embed_pool 2 | Qdrant 8 vCPU, 32 GB, NVMe | Separate DB host recommended (remote mode). |
| 50M–500M chunks | 3× API replicas (20 vCPU, 36 GB each) | cpu_pool = cores; embed_pool 4 | Qdrant cluster: shards ≥ 4, replication ≥ 2 | Use `collection.shard_number` / `replication_factor`. Each replica runs the ingestion jobs it accepts. |

These rows are the limits `deploy/compose.{small,medium,large}.yml` set. Ingestion runs inside the API process, so the API tier carries both loads: dedicated worker hosts, and a queue backend that lets them take ingestion off the API, are not built (TASK-0130). Until 2026-11-12 the presets also started a `worker` container; it checked the vector database and waited, so its limits are now the API's.

Rules of thumb: vector RAM ≈ `vectors × dimensions × 4 bytes` (plus index overhead — budget 1.5×); embedding throughput scales linearly with `embedding_pool_size` until GPU saturation; `queue_depth` × average chunk size bounds pipeline memory. These are planning defaults — real numbers come from the [benchmark ledger](benchmarks.md) once measured (see provable-claims policy).

## 5. Upgrades

- SemVer discipline: patch/minor upgrades are drop-in; majors carry CHANGELOG migration notes.
- Langfuse/Grafana stacks are upgraded by the provisioner only when the pinned versions in fasterRag change; **`SALT`/`ENCRYPTION_KEY`/`NEXTAUTH_SECRET` are never regenerated** (see [observability.md](observability.md)).
- Embedding-model upgrades go through D2 zero-downtime reindexing, never in-place mutation.
- **Stopping the API** (TASK-0436). On a graceful stop the process refuses new ingestion jobs, waits `reliability.shutdown_grace_seconds` (default 20) for running ones, cancels the rest and settles each `failed` / `INTERRUPTED` with its checkpoint kept. Give the process longer than that before it is killed: `docker stop` allows 10 s by default (`stop_grace_period` in Compose, `--time` on the command line) and Kubernetes 30 s (`terminationGracePeriodSeconds`). The compose files in `deploy/` do not set a stop period, so under them the default 20 s wait is cut short at 10 s — raise the period or lower the setting.

## 6. Revert playbook (rollback discipline)

| What went wrong | Revert action |
|---|---|
| Bad code/doc change (any slice) | `git revert <sha>` or revert the slice tag range `git revert v0.x.0-sN..v0.x.0-sN+1`; small single-line commits on a linear `main` keep every increment independently revertible. |
| Bad reindex / embedding-model change | **Alias flip back**: `fasterrag index rollback <collection>` (or `POST /v1/collections/{name}/rollback`) — one alias flip with no rebuild, because the previous collection is retained for `index.reindex.rollback_retention_hours` (D2). The flip's own latency is unmeasured (D2 proof metric). |
| Bad config change | Restore the previous `config.yaml` from git; `fasterrag config validate`; restart. Config is fully versioned because it contains no secrets. |
| Retrieval-quality regression | The D7 regression gate should have blocked it; if it reached prod, alias-rollback (above) and file the gate gap as a bug in [todo.md](todo.md). |
| Provisioned tool broken (Langfuse/Grafana) | `fasterrag provision <tool> --down` then re-provision; data volumes and secrets persist. |
| Data loss / corruption | Restore per [disaster-recovery.md](disaster-recovery.md). |

Risky features (auto-provisioning, autopilot, dashboard) are behind config flags defaulting to `false`, so reverting them is a config flip, not a deploy.
