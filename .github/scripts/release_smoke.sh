#!/usr/bin/env bash
# The smoke test a release must pass, run inside a container that has never seen the
# repository: nothing but a Python base image, this file, and (for the rehearsal) the built
# wheel.
#
# release.yml runs it before anything is published, against the wheel that is about to be
# uploaded, and again after PyPI has accepted the upload, against the version the index serves.
# Each time it runs twice: once installed the way the documents tell a user to install
# (`pip install "fasterrag>=<version>"` for the published package), and once with `--pre`,
# which also takes pre-releases of every dependency. The documented command is the one users
# follow; `--pre` is the one some of them type anyway, and 0.1.0b1 broke under it because an
# uncapped httpx resolved to 1.0.dev6 (TASK-0556). The same script every time, so a rehearsal
# that passes has exercised exactly what the post-publish check will.
#
# What it proves: the distribution installs with pip into an empty environment; the version it
# reports is the one that was released; the import comes from site-packages and not from a
# checkout; every import docs/python-api.md promises resolves; parsing, chunking and fusion
# run with no configuration, no key and no server; and the console script can write a starter
# configuration and run its own preflight. When LOCAL_PROFILE_REQUIREMENT is set it also adds
# the local profile's extras and runs scripts/local_profile_probe.py: `config init --profile
# local`, ingest a note of several chunks, and retrieve the one chunk that answers a question,
# with no service and no key (TASK-0542). That installs CPU-only torch and downloads the
# embedding model, so the `--pre` run leaves it out.
#
#   REQUIREMENT                what pip installs: a wheel path, or `fasterrag>=0.1.0b2`
#   VERSION                    the `__version__` the installed package must report
#   PIP_FLAGS                  extra pip arguments for every install, e.g. `--pre` (default none)
#   ATTEMPTS                   how many times to try the install, 30 s apart (default 1). A new
#                              upload takes a moment to reach every index mirror, so the
#                              post-publish run waits for it.
#   LOCAL_PROFILE_REQUIREMENT  optional: the same package with `[chroma,huggingface]`
#   LOCAL_PROFILE_PROBE        the probe's path inside the container, required with the above
set -euo pipefail

: "${REQUIREMENT:?set REQUIREMENT to a wheel path or a pinned requirement}"
: "${VERSION:?set VERSION to the version the package must report}"
attempts="${ATTEMPTS:-1}"
read -r -a pip_flags <<< "${PIP_FLAGS:-}"

python -m venv /tmp/release-smoke
export PATH="/tmp/release-smoke/bin:${PATH}"
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

attempt=1
until pip install --quiet "${pip_flags[@]}" "${REQUIREMENT}"; do
  if [ "${attempt}" -ge "${attempts}" ]; then
    echo "::error::pip could not install ${REQUIREMENT} after ${attempt} attempt(s)"
    exit 1
  fi
  attempt=$((attempt + 1))
  echo "install failed; attempt ${attempt} of ${attempts} in 30 s"
  sleep 30
done

# An empty directory, so nothing here can be read from a checkout by accident.
cd "$(mktemp -d)"

installed="$(python -c 'import fasterrag; print(fasterrag.__version__)')"
if [ "${installed}" != "${VERSION}" ]; then
  echo "::error::installed fasterrag reports ${installed}, expected ${VERSION}"
  exit 1
fi

python - <<'PY'
import pathlib

import fasterrag
from fasterrag import FasterRag
from fasterrag.chunking import LayoutChunker, RecursiveChunker, SemanticChunker
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.evals import evaluate
from fasterrag.parsing import parse_document
from fasterrag.retrieval import rrf_fuse
from fasterrag.sync import FasterRag as BlockingFasterRag

assert "site-packages" in fasterrag.__file__, fasterrag.__file__

note = pathlib.Path("note.md")
note.write_text("# Title\n\nA paragraph about retrieval.\n\n## Section\n\nOne about fusion.\n")
chunks = RecursiveChunker(chunk_size=768, overlap=64).split(parse_document(str(note)))
assert chunks and "retrieval" in chunks[0].text, chunks

fused = rrf_fuse(["a", "b", "c"], ["b", "c", "d"], k=60)
assert [result.id for result in fused][:2] == ["b", "c"], fused
print(f"fasterrag {fasterrag.__version__}: imports, parsing, chunking and fusion work")
PY

fasterrag --help > /dev/null
fasterrag config init
test -f config.yaml
test -f .env.example

# Placeholder keys so doctor gets past configuration validation and runs its environment
# checks. `vector_db.mode=external` because the shipped default is `docker`, and in a
# container with no Docker executable doctor stops there with a typed PROVISIONING_FAILED and
# prints no report at all. Exit 4 is accepted: nothing is listening where the vector database
# should be, and doctor saying so is doctor working. Any other non-zero exit is a crash, and
# the report must still say the configuration and the interpreter are sound.
FASTERRAG_API_KEY=smoke QDRANT_API_KEY=smoke OPENAI_API_KEY=smoke \
  fasterrag --set vector_db.mode=external doctor --json > doctor.json || [ $? -eq 4 ]
python - <<'PY'
import json

report = json.load(open("doctor.json"))
checks = {check["name"]: check for check in report["checks"]}
for name in ("config", "python"):
    assert checks[name]["passed"], checks[name]
print(f"doctor ran {len(checks)} checks; config and python passed")
PY

if [ -n "${LOCAL_PROFILE_REQUIREMENT:-}" ]; then
  : "${LOCAL_PROFILE_PROBE:?set LOCAL_PROFILE_PROBE to local_profile_probe.py inside the container}"
  # CPU-only torch: PyPI's Linux wheel pulls gigabytes of CUDA the local profile never loads.
  pip install --quiet "${pip_flags[@]}" \
    --extra-index-url https://download.pytorch.org/whl/cpu "${LOCAL_PROFILE_REQUIREMENT}"
  cd "$(mktemp -d)"
  python -I "${LOCAL_PROFILE_PROBE}" "$(command -v fasterrag)"
fi

echo "release smoke test passed for fasterrag ${VERSION} (pip flags: ${PIP_FLAGS:-none})"
