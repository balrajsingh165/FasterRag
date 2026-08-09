"""BM25 encoding invariants, over generated text rather than chosen sentences.

The sparse leg exists for exact identifiers — part numbers, versions, SKUs — so the
inputs that matter are the ones nobody writes an example for. Two general forms turned out
to be false against them, both fixed under TASK-0228:

* **Term indices are a 32-bit CRC, and CRC32 collides.** Two distinct terms landing on one
  index made ``encode_document`` emit a sparse vector carrying that index *twice*, with two
  different values. A sparse vector with a repeated index has no defined meaning: the
  backend either rejects the upsert or keeps whichever value it saw last, so one term's
  weight vanished. Colliding terms are now merged, their frequencies added, which is how
  the backend has to treat them anyway.
* **``tokenize`` dropped single characters before stemming but not after.** ``ueds`` stems
  to ``u``, so a single-character term reached the index despite the filter written to keep
  it out. The stopword filter was already applied twice; the length filter was not.

Both are the reason this file hard-codes inputs instead of trusting search, and the two sit
at opposite ends of the same problem. A CRC32 collision needs roughly 2**16 distinct terms
before the birthday bound makes one likely — measured here at about 75k identifier-shaped
tokens for the first three, which is a small corpus of part numbers, not a big one, but far
more than any generator will emit inside one example. A stem that trips a filter is rarer
still in a different way: only twelve three- and four-letter words shorten past the minimum.
Generation reaches neither reliably, which is exactly how the missing post-stem length filter
survived a mutation run of the searched property below.

Every test over a hard-coded input re-derives what makes it interesting — the collision, the
stem — so a change to the hash, the tokenizer, or the stemmer makes it fail loudly rather
than pass over inputs that are no longer the case it was written for.
"""

import math
from itertools import pairwise

import snowballstemmer
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from fasterrag.core.retrieval.bm25 import (
    AVERAGE_DOCUMENT_LENGTH,
    STOPWORDS,
    SparseVector,
    encode_document,
    encode_query,
    term_index,
    tokenize,
)

# CRITICAL: distinct terms that collide under term_index. Verified in every test that uses
# them, so they can never silently stop being collisions and leave the assertions vacuous.
COLLIDING_PAIRS = [
    ("joqwmdew2y1", "pqm4"),
    ("0wuuqn", "3yu1pq9"),
    ("m34jrs", "rdoy0f89"),
]

# CRITICAL: tokens whose *stem* trips a filter their unstemmed form passed. Across every
# three- and four-letter word, exactly twelve stem shorter than the minimum and thirty-eight
# stem into a stopword — so generated text reaches them on some seeds and not others, which
# is how the missing post-stem length filter survived a run of the property below. Pinned for
# the same reason the collisions are, and each test re-derives the stem so a stemmer upgrade
# fails loudly rather than leaving the assertions vacuous.
STEMS_BELOW_THE_MINIMUM = ["ueds", "aed", "oed", "aing", "ieds"]
STEMS_INTO_A_STOPWORD = ["doing", "ans", "ifs", "ofs", "ande"]

SENTINEL = "zqxjvk"

# Wide enough to reach identifiers, digits, mixed case and the punctuation the tokenizer
# keeps inside a term. Narrowing this to lowercase letters would put `bge-small-en`,
# `v1.9.0` and every SKU shape out of reach of the generator.
ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"

VOCABULARY = st.sampled_from(
    [
        "termination",
        "Terminating",
        "notice",
        "policies",
        "running",
        "doing",
        "does",
        "the",
        "and",
        "a",
        "ueds",
        "bge-small-en",
        "v1.9.0",
        "sku-4471",
        SENTINEL,
    ]
)
WORD = st.one_of(st.text(alphabet=ALPHABET, min_size=1, max_size=12), VOCABULARY)
SEPARATOR = st.sampled_from([" ", "\n", ", ", "\t", " -- ", "; "])
TEXT = st.builds(lambda words, gap: gap.join(words), st.lists(WORD, max_size=25), SEPARATOR)
WILD_TEXT = st.text(max_size=200)

# CRITICAL: both, always. Unrestricted text reaches the tokenizer's unicode and punctuation
# edges but will never emit `ueds` or `doing`, the tokens whose *stems* break the filters —
# those live in VOCABULARY. Either strategy alone leaves half the tokenizer untested.
ANY_TEXT = st.one_of(TEXT, WILD_TEXT)
K1 = st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False)
B = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def weights(vector: SparseVector) -> dict[int, float]:
    """Return the vector as an index-to-weight mapping."""
    return dict(zip(vector.indices, vector.values, strict=True))


def weight_of(vector: SparseVector, term: str) -> float:
    """Return the weight the vector gives ``term``."""
    return weights(vector)[term_index(term)]


def assert_collides(first: str, second: str) -> None:
    """Fail loudly if a hard-coded pair has stopped being a usable collision."""
    assert first != second
    assert tokenize(first) == [first]
    assert tokenize(second) == [second]
    assert term_index(first) == term_index(second)


def stem_of(token: str) -> str:
    """Return what the stemmer does to ``token``, independently of the tokenizer."""
    return str(snowballstemmer.stemmer("english").stemWords([token])[0])


@given(text=WILD_TEXT, k1=K1, b=B)
def test_encoding_the_same_text_twice_gives_the_same_vector(text: str, k1: float, b: float) -> None:
    """Re-indexing a chunk must not move it, or a re-ingest silently rewrites the index."""
    first = encode_document(text, k1=k1, b=b)
    term_index.cache_clear()
    second = encode_document(text, k1=k1, b=b)

    assert list(first.indices) == list(second.indices)
    assert list(first.values) == list(second.values)
    assert list(encode_query(text).indices) == list(encode_query(text).indices)


@given(words=st.lists(WORD, max_size=20), data=st.data())
def test_word_order_does_not_change_the_encoding(words: list[str], data: st.DataObject) -> None:
    """A bag of words is a bag: the same terms in any order encode identically.

    This is what makes the encoding canonical rather than merely repeatable — two ingests
    of the same passage split differently by an upstream parser must land on one vector.
    """
    shuffled = data.draw(st.permutations(words))
    first = encode_document(" ".join(words))
    second = encode_document(" ".join(shuffled))

    assert list(first.indices) == list(second.indices)
    assert list(first.values) == list(second.values)


@given(text=WILD_TEXT, k1=K1, b=B)
def test_a_sparse_vector_never_carries_an_index_twice(text: str, k1: float, b: float) -> None:
    """One entry per index, and one value per entry, or the backend cannot store it."""
    document = encode_document(text, k1=k1, b=b)
    query = encode_query(text)

    assert len(set(document.indices)) == len(document.indices)
    assert len(set(query.indices)) == len(query.indices)
    assert len(weights(document)) == len(document)


@given(pair=st.sampled_from(COLLIDING_PAIRS), text=TEXT, k1=K1, b=B)
def test_two_terms_sharing_an_index_collapse_to_one_entry(
    pair: tuple[str, str], text: str, k1: float, b: float
) -> None:
    """The shape random search will not find, so it is constructed instead."""
    first, second = pair
    assert_collides(first, second)

    document = encode_document(f"{first} {text} {second}", k1=k1, b=b)
    query = encode_query(f"{first} {second} {text}")

    assert len(set(document.indices)) == len(document.indices)
    assert len(set(query.indices)) == len(query.indices)
    assert term_index(first) in set(document.indices)


@given(pair=st.sampled_from(COLLIDING_PAIRS), k1=K1, b=B)
def test_a_collision_adds_the_frequencies_rather_than_losing_one(
    pair: tuple[str, str], k1: float, b: float
) -> None:
    """Whatever the backend does with the index, both occurrences must be counted once."""
    first, second = pair
    assert_collides(first, second)

    collided = encode_document(f"{first} {second}", k1=k1, b=b)
    twice = encode_document(f"{second} {second}", k1=k1, b=b)

    assert len(collided) == 1
    assert list(collided.values) == list(twice.values)


@given(text=TEXT, k1=K1, b=B, repeats=st.integers(min_value=1, max_value=6))
def test_repeating_a_term_never_lowers_its_weight(
    text: str, k1: float, b: float, repeats: int
) -> None:
    """Saturation must be monotone: an extra mention is never evidence against a term.

    Each repetition also lengthens the document, which pulls the normalizer the other way,
    so the two effects are tested together rather than at an artificially fixed length.
    """
    assert tokenize(SENTINEL) == [SENTINEL]
    observed = [
        weight_of(encode_document(f"{text} {(SENTINEL + ' ') * count}", k1=k1, b=b), SENTINEL)
        for count in range(1, repeats + 1)
    ]

    for earlier, later in pairwise(observed):
        assert later >= earlier - 1e-12


@given(text=TEXT, k1=st.floats(min_value=0.1, max_value=3.0), b=B)
@settings(max_examples=60)
def test_each_extra_mention_is_worth_less_than_the_last(text: str, k1: float, b: float) -> None:
    """The half of saturation that stops a keyword-stuffed page from winning."""
    assert tokenize(SENTINEL) == [SENTINEL]
    observed = [
        weight_of(encode_document(f"{text} {(SENTINEL + ' ') * count}", k1=k1, b=b), SENTINEL)
        for count in range(1, 8)
    ]
    steps = [later - earlier for earlier, later in pairwise(observed)]

    for earlier, later in pairwise(steps):
        assert later <= earlier + 1e-12


@given(
    filler=st.integers(min_value=1, max_value=600),
    k1=st.floats(min_value=0.1, max_value=3.0),
)
@settings(max_examples=60)
def test_length_normalisation_moves_monotonically_with_b(filler: int, k1: float) -> None:
    """``b`` only ever trades short documents against long ones, around the average.

    Below the average length, raising ``b`` rewards a document for being short; above it,
    ``b`` penalises the extra length. Generating counts on both sides of
    ``AVERAGE_DOCUMENT_LENGTH`` is the point — a strategy capped below it would assert only
    half the behaviour and never notice the sign flip.
    """
    text = " ".join([SENTINEL, *(f"w{index}" for index in range(filler))])
    length = len(tokenize(text))
    assume(length != AVERAGE_DOCUMENT_LENGTH)

    off = weight_of(encode_document(text, k1=k1, b=0.0), SENTINEL)
    full = weight_of(encode_document(text, k1=k1, b=1.0), SENTINEL)

    if length < AVERAGE_DOCUMENT_LENGTH:
        assert full > off
    else:
        assert full < off


@given(text=WILD_TEXT, k1=K1, b=B)
def test_weights_stay_positive_and_bounded_by_the_saturation_ceiling(
    text: str, k1: float, b: float
) -> None:
    """``f(k1+1)/(f+n)`` cannot exceed ``k1+1``; a weight that does means ``n`` went wrong."""
    for value in encode_document(text, k1=k1, b=b).values:
        assert value > 0.0
        assert value <= k1 + 1.0 + 1e-12
        assert math.isfinite(value)


@given(words=st.lists(st.sampled_from(sorted(STOPWORDS)), min_size=1, max_size=20))
def test_text_made_only_of_stopwords_encodes_to_nothing(words: list[str]) -> None:
    """Otherwise every document shares a handful of terms and the leg stops discriminating."""
    text = " ".join(words)

    assert tokenize(text) == []
    assert encode_document(text).empty
    assert encode_query(text).empty
    assert len(encode_document(text)) == 0


@given(text=ANY_TEXT)
def test_no_emitted_term_is_a_stopword_or_shorter_than_the_minimum(text: str) -> None:
    """Both filters have to survive stemming, which can shorten a token or reshape it.

    Searched rather than constructed, so it covers shapes nobody thought to pin — but it is
    the two tests below, not this one, that reliably fail when a filter stops running after
    stemming. The tokens that expose it are too rare for generation to be depended on.
    """
    for term in tokenize(text):
        assert len(term) >= 2
        assert term not in STOPWORDS
        assert term == term.lower()


@given(token=st.sampled_from(STEMS_BELOW_THE_MINIMUM), text=TEXT)
def test_a_token_stemming_below_the_minimum_never_reaches_the_index(token: str, text: str) -> None:
    """The length filter has to run again after stemming, not only before it."""
    stem = stem_of(token)
    assert len(token) >= 2
    assert token not in STOPWORDS
    assert len(stem) < 2

    assert tokenize(token) == []
    assert stem not in tokenize(f"{text} {token}")


@given(token=st.sampled_from(STEMS_INTO_A_STOPWORD), text=TEXT)
def test_a_token_stemming_into_a_stopword_never_reaches_the_index(token: str, text: str) -> None:
    """And so does the stopword filter: ``doing`` passes it, ``do`` must not."""
    stem = stem_of(token)
    assert token not in STOPWORDS
    assert stem in STOPWORDS

    assert tokenize(token) == []
    assert stem not in tokenize(f"{text} {token}")


@given(text=WILD_TEXT)
def test_a_text_with_no_terms_encodes_to_an_empty_vector(text: str) -> None:
    """The empty case is a real one: a query of punctuation must not index anything."""
    assume(not tokenize(text))

    assert encode_document(text).empty
    assert encode_query(text).empty


@given(term=st.text(min_size=0, max_size=30))
def test_a_term_index_is_stable_and_fits_the_sparse_key_space(term: str) -> None:
    """A moving index would invalidate every vector already written to the store."""
    first = term_index(term)
    term_index.cache_clear()

    assert term_index(term) == first
    assert 0 <= first <= 0xFFFFFFFF


@given(text=WILD_TEXT)
def test_a_query_weights_every_term_flat_and_lists_it_once(text: str) -> None:
    """Rarity is the backend's IDF to apply, so the client must not pre-weight anything."""
    query = encode_query(text)

    assert set(query.values) <= {1.0}
    assert len(query.values) == len(query.indices)
    assert len(set(query.indices)) == len(query.indices)


@given(text=TEXT, k1=K1, b=B)
def test_every_query_term_a_document_contains_is_in_that_document_s_vector(
    text: str, k1: float, b: float
) -> None:
    """The two encoders must agree on terms, or the sparse leg matches nothing."""
    assume(tokenize(text))

    assert set(encode_query(text).indices) == set(encode_document(text, k1=k1, b=b).indices)
