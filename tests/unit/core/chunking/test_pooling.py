"""Late-chunking pooling, against a fake model rather than a downloaded one.

The fake returns a distinct constant vector per token id, so a pooled span is the mean of
exactly the tokens inside it — which makes "did the right tokens get pooled" an equality
rather than an approximation.

The model is faked but ``torch`` is not: pooling averages real tensors. Torch ships in the
optional ``huggingface`` extra, so this module skips rather than erroring without it — a
hard import made a fresh core-only install fail to *collect* the whole unit suite, not just
fail these tests.
"""

from typing import Any

import pytest

from fasterrag.core.chunking.pooling import WINDOW_OVERLAP_FRACTION, pool_spans, supports_pooling
from fasterrag.errors import EmbedError

torch = pytest.importorskip("torch", reason="pooling needs the optional 'huggingface' extra")

WIDTH = 4


class FakeTokenizer:
    """Splits on spaces, reporting each word's character offsets."""

    cls_token_id = 101
    sep_token_id = 102

    def __call__(
        self, text: str, add_special_tokens: bool = True, return_offsets_mapping: bool = False
    ) -> dict[str, Any]:
        """Return word ids with their character offsets."""
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for word in text.split(" "):
            if word:
                start = text.index(word, cursor)
                ids.append(len(ids) + 1)
                offsets.append((start, start + len(word)))
                cursor = start + len(word)
        return {"input_ids": ids, "offset_mapping": offsets}


class FakeNetwork:
    """Returns one deterministic row per input token."""

    def __init__(self) -> None:
        self.config = type("c", (), {"hidden_size": WIDTH})()
        self.windows: list[int] = []
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def parameters(self) -> Any:
        return iter([self._parameter])

    def __call__(self, input_ids: Any, attention_mask: Any) -> Any:
        """Return one row per input token, recording the window width."""
        self.windows.append(int(input_ids.shape[1]))
        rows = torch.stack([torch.full((WIDTH,), float(value)) for value in input_ids[0].tolist()])
        return type("o", (), {"last_hidden_state": rows.unsqueeze(0)})()


class FakeModel:
    """A stand-in exposing exactly what pooling needs."""

    def __init__(self, max_seq_length: int = 512) -> None:
        self.tokenizer = FakeTokenizer()
        self.network = FakeNetwork()
        self.max_seq_length = max_seq_length

    def __getitem__(self, index: int) -> Any:
        """Return the transformer module, as sentence-transformers exposes it."""
        return type("m", (), {"auto_model": self.network})()


TEXT = "alpha beta gamma delta epsilon"


def test_a_model_with_token_output_is_usable() -> None:
    assert supports_pooling(FakeModel()) is True


def test_a_model_without_token_output_is_not() -> None:
    """Every hosted provider lands here, which is why the caller must fall back."""
    assert supports_pooling(object()) is False


def test_one_vector_per_span_in_order() -> None:
    vectors = pool_spans(FakeModel(), TEXT, [(0, 5), (6, 10), (11, 29)])

    assert len(vectors) == 3
    assert all(len(vector) == WIDTH for vector in vectors)


def test_a_span_pools_only_its_own_tokens() -> None:
    """The whole feature is which tokens are averaged; getting the range wrong is silent."""
    vectors = pool_spans(FakeModel(), TEXT, [(0, 5)])

    # "alpha" is token id 1, so every component is 1 before normalization, and 0.5 after.
    assert vectors[0] == pytest.approx([0.5] * WIDTH)


def test_every_vector_is_unit_length() -> None:
    vectors = pool_spans(FakeModel(), TEXT, [(0, 10), (11, 29)])

    for vector in vectors:
        assert sum(value * value for value in vector) == pytest.approx(1.0)


def test_no_spans_makes_no_pass() -> None:
    model = FakeModel()

    assert pool_spans(model, TEXT, []) == []
    assert model.network.windows == []


def test_a_short_document_takes_one_pass() -> None:
    """The single pass is the point; windowing a document that fits would waste it."""
    model = FakeModel()

    pool_spans(model, TEXT, [(0, 29)])

    assert len(model.network.windows) == 1


def test_a_long_document_is_windowed_rather_than_truncated() -> None:
    """Truncating would silently drop every token past the model's context."""
    model = FakeModel(max_seq_length=12)
    text = " ".join(f"word{index}" for index in range(60))

    vectors = pool_spans(model, text, [(0, len(text))])

    assert len(model.network.windows) > 1
    assert sum(value * value for value in vectors[0]) == pytest.approx(1.0)


def test_windows_overlap_so_edge_tokens_get_context() -> None:
    model = FakeModel(max_seq_length=12)
    text = " ".join(f"word{index}" for index in range(60))

    pool_spans(model, text, [(0, len(text))])

    body = model.max_seq_length - 2
    assert WINDOW_OVERLAP_FRACTION > 1
    assert all(width <= body + 2 for width in model.network.windows)


def test_a_span_matching_no_token_still_gets_a_vector() -> None:
    """A zero vector would rank against every query; a whitespace span must not do that."""
    vectors = pool_spans(FakeModel(), TEXT, [(5, 6)])

    assert sum(value * value for value in vectors[0]) == pytest.approx(1.0)


def test_a_model_that_cannot_pool_is_refused() -> None:
    with pytest.raises(EmbedError) as caught:
        pool_spans(object(), TEXT, [(0, 5)])

    assert "token-level" in caught.value.detail


def test_a_document_with_no_tokens_is_refused() -> None:
    with pytest.raises(EmbedError):
        pool_spans(FakeModel(), "   ", [(0, 3)])
