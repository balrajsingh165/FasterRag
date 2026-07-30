"""BM25 sparse encoding for the keyword retrieval leg.

Hybrid retrieval needs BM25 because dense vectors reliably miss exact identifiers, part
numbers, and rare terms — the things users actually paste into a search box
(``docs/adr/ADR-0004``).

The work is split with the backend, deliberately (``docs/adr/ADR-0007``):

* **Here**: tokenization, stemming, and the term-frequency saturation half of BM25. Doing
  it here means every backend sees identical terms, so the sparse leg behaves the same on
  Qdrant as on pgvector.
* **The backend**: inverse document frequency. IDF is a statistic over the whole live
  corpus and changes with every ingest and delete, so only the store that holds the corpus
  can compute it correctly.

Documents and queries encode differently on purpose. A document's terms carry the saturated
term frequency, while a query's terms carry a flat weight — the query has no length to
normalize against, and weighting its terms by their rarity is precisely the IDF the backend
applies.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import snowballstemmer

__all__ = [
    "AVERAGE_DOCUMENT_LENGTH",
    "K1",
    "STOPWORDS",
    "B",
    "SparseVector",
    "encode_document",
    "encode_query",
    "term_index",
    "tokenize",
]

K1: Final = 1.2
B: Final = 0.75
AVERAGE_DOCUMENT_LENGTH: Final = 256

_MINIMUM_TERM_LENGTH: Final = 2
_TERM_MASK: Final = 0xFFFFFFFF

_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")

STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "of",
        "on",
        "or",
        "our",
        "shall",
        "should",
        "so",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

_stemmer = snowballstemmer.stemmer("english")


@dataclass(frozen=True, slots=True)
class SparseVector:
    """Term indices paired with their weights.

    Vendor-neutral: an adapter translates this into whatever sparse representation its
    backend accepts.
    """

    indices: Sequence[int]
    values: Sequence[float]

    def __post_init__(self) -> None:
        """Reject a vector whose indices and values do not line up."""
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"a sparse vector has {len(self.indices)} indices but {len(self.values)} values"
            )

    def __len__(self) -> int:
        """Return the number of terms."""
        return len(self.indices)

    @property
    def empty(self) -> bool:
        """Return whether the vector carries no terms."""
        return not self.indices


@lru_cache(maxsize=100_000)
def term_index(term: str) -> int:
    """Return the stable numeric index for a term.

    Sparse vectors are keyed by integer, so terms are hashed. The hash is stable across
    processes and releases — changing it would silently invalidate every indexed sparse
    vector, exactly like changing the point-id namespace.
    """
    return zlib.crc32(term.encode("utf-8")) & _TERM_MASK


def tokenize(text: str) -> list[str]:
    """Split text into stemmed, stopword-filtered terms.

    Identifiers keep their internal punctuation, so ``bge-small-en`` and ``v1.9.0`` survive
    as single terms rather than fragmenting into meaningless pieces — those exact tokens are
    the main reason a keyword leg exists at all.
    """
    candidates = [
        token
        for token in _TOKEN.findall(text.lower())
        if len(token) >= _MINIMUM_TERM_LENGTH and token not in STOPWORDS
    ]
    if not candidates:
        return []

    stemmed = _stemmer.stemWords(candidates)
    return [term for term in stemmed if term and term not in STOPWORDS]


def encode_document(text: str) -> SparseVector:
    """Encode a passage as saturated term frequencies.

    Saturation is what stops a term repeated fifty times from dominating: the weight grows
    quickly for the first few occurrences and then flattens. Length normalization keeps a
    long document from outscoring a short, precise one purely by having more words.
    """
    terms = tokenize(text)
    if not terms:
        return SparseVector(indices=(), values=())

    length = len(terms)
    normalizer = K1 * (1 - B + B * length / AVERAGE_DOCUMENT_LENGTH)

    counted = Counter(terms)
    indices: list[int] = []
    values: list[float] = []
    for term, frequency in sorted(counted.items()):
        indices.append(term_index(term))
        values.append(frequency * (K1 + 1) / (frequency + normalizer))

    return SparseVector(indices=indices, values=values)


def encode_query(text: str) -> SparseVector:
    """Encode a query as flat term weights, leaving rarity to the backend's IDF."""
    terms = sorted(set(tokenize(text)))
    return SparseVector(
        indices=[term_index(term) for term in terms],
        values=[1.0] * len(terms),
    )
