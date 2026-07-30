import pytest

from fasterrag.core.retrieval.bm25 import (
    AVERAGE_DOCUMENT_LENGTH,
    SparseVector,
    encode_document,
    encode_query,
    term_index,
    tokenize,
)


def weights(vector: SparseVector) -> dict[int, float]:
    return dict(zip(vector.indices, vector.values, strict=True))


def weight_of(vector: SparseVector, term: str) -> float:
    return weights(vector)[term_index(term)]


def test_terms_are_lowercased_and_stemmed() -> None:
    assert tokenize("Termination TERMINATING terminate") == ["termin", "termin", "termin"]


def test_a_query_and_a_document_agree_on_inflections() -> None:
    document = encode_document("The agreement covers terminations and notices.")
    query = encode_query("terminate notice")

    assert set(query.indices) <= set(document.indices)


def test_stopwords_are_dropped() -> None:
    assert tokenize("the and of a to with") == []


def test_identifiers_survive_as_single_terms() -> None:
    terms = tokenize("model bge-small-en version v1.9.0")

    assert "bge-small-en" in terms
    assert "v1.9.0" in terms


def test_single_characters_are_dropped() -> None:
    assert tokenize("a b c termination") == ["termin"]


def test_empty_text_produces_an_empty_vector() -> None:
    vector = encode_document("   ")

    assert vector.empty is True
    assert len(vector) == 0


def test_term_indices_are_stable_and_distinct() -> None:
    assert term_index("termination") == term_index("termination")
    assert term_index("termination") != term_index("notice")
    assert 0 <= term_index("termination") <= 0xFFFFFFFF


def fixed_length_document(occurrences: int, length: int = 100) -> float:
    """Return the weight of ``notice`` in a document of constant length."""
    filler = " ".join(f"term{index}" for index in range(length - occurrences))
    body = ("notice " * occurrences) + filler
    return weight_of(encode_document(body), "notic")


def test_document_weights_saturate_with_repetition() -> None:
    once, twice, thrice = (fixed_length_document(count) for count in (1, 2, 3))

    assert once < twice < thrice
    assert twice - once > thrice - twice


def test_saturation_bounds_the_weight_however_often_a_term_repeats() -> None:
    weight = fixed_length_document(90)

    assert weight < 1 + 1.2


def test_a_longer_document_dilutes_the_same_term() -> None:
    short = weight_of(encode_document("notice period"), "notic")
    padded = "notice period " + ("filler word " * 200)
    long_document = weight_of(encode_document(padded), "notic")

    assert long_document < short


def test_a_document_of_average_length_normalizes_to_one() -> None:
    body = " ".join(f"term{index}" for index in range(AVERAGE_DOCUMENT_LENGTH))
    vector = encode_document(body + " notice")

    assert 0.0 < weight_of(vector, "notic") < 1.5


def test_repeated_query_terms_collapse_to_one_entry() -> None:
    vector = encode_query("notice notice notice")

    assert len(vector) == 1
    assert list(vector.values) == [1.0]


def test_query_weights_are_flat_because_idf_is_the_backend_job() -> None:
    vector = encode_query("termination notice period")

    assert set(vector.values) == {1.0}


def test_indices_are_ordered_so_encodings_are_reproducible() -> None:
    first = encode_document("notice period termination clause")
    second = encode_document("clause termination period notice")

    assert list(first.indices) == list(second.indices)


def test_word_order_does_not_change_a_bag_of_words_encoding() -> None:
    first = encode_document("notice period")
    second = encode_document("period notice")

    assert weights(first) == weights(second)


def test_a_misaligned_sparse_vector_is_refused() -> None:
    with pytest.raises(ValueError, match="2 indices but 1 values"):
        SparseVector(indices=[1, 2], values=[0.5])


def test_a_document_and_its_query_share_no_weights_but_share_terms() -> None:
    document = encode_document("the termination notice period is thirty days")
    query = encode_query("termination notice")

    assert set(query.indices) < set(document.indices)
    assert list(query.values) == [1.0, 1.0]
