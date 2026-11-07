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

from fasterrag.core.chunking.pooling import (
    WINDOW_OVERLAP_FRACTION,
    _windows,
    pool_spans,
    supports_pooling,
)
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
    assert caught.value.retryable is False


def test_a_document_with_no_tokens_is_refused() -> None:
    """Retrying a document the tokenizer returned nothing for would spin on the same answer."""
    with pytest.raises(EmbedError) as caught:
        pool_spans(FakeModel(), "   ", [(0, 3)])

    assert caught.value.retryable is False


ONE_HOT_WIDTH = 8


class OneHotNetwork:
    """Returns a distinct one-hot row per token id, so a pooled set stays readable."""

    def __init__(self) -> None:
        self.config = type("c", (), {"hidden_size": ONE_HOT_WIDTH})()
        self.windows: list[int] = []
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def parameters(self) -> Any:
        return iter([self._parameter])

    def __call__(self, input_ids: Any, attention_mask: Any) -> Any:
        """Return a one-hot row per input token, recording the window width."""
        self.windows.append(int(input_ids.shape[1]))
        rows = torch.stack(
            [
                torch.nn.functional.one_hot(
                    torch.tensor(int(value) % ONE_HOT_WIDTH), ONE_HOT_WIDTH
                ).float()
                for value in input_ids[0].tolist()
            ]
        )
        return type("o", (), {"last_hidden_state": rows.unsqueeze(0)})()


class OneHotModel:
    """A stand-in whose token vectors differ, making "which tokens" an equality.

    ``FakeModel`` gives every token the same direction, so any mean normalizes to the same
    unit vector and a span that pools one token too many is indistinguishable from a correct
    one. Only a model with distinguishable tokens can assert the selection rule.
    """

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.network = OneHotNetwork()
        self.max_seq_length = 512

    def __getitem__(self, index: int) -> Any:
        """Return the transformer module, as sentence-transformers exposes it."""
        return type("m", (), {"auto_model": self.network})()


def one_hot(*indices: int) -> list[float]:
    """Return the unit vector that pooling the given token indices produces."""
    scale = len(indices) ** -0.5
    return [scale if index in indices else 0.0 for index in range(ONE_HOT_WIDTH)]


def test_a_span_pools_exactly_the_tokens_that_lie_inside_it() -> None:
    """Which tokens are averaged is the whole feature, and getting it wrong is silent."""
    vectors = pool_spans(OneHotModel(), TEXT, [(0, 5), (6, 10), (0, 11)])

    assert vectors[0] == pytest.approx(one_hot(1))
    assert vectors[1] == pytest.approx(one_hot(2))
    assert vectors[2] == pytest.approx(one_hot(1, 2))


def test_a_token_starting_where_the_span_ends_is_not_pooled() -> None:
    """One character of slack widens every chunk's vector by a whole neighbouring token."""
    assert pool_spans(OneHotModel(), TEXT, [(0, 6)])[0] == pytest.approx(one_hot(1))


def test_a_token_ending_where_the_span_starts_is_not_pooled() -> None:
    assert pool_spans(OneHotModel(), TEXT, [(5, 10)])[0] == pytest.approx(one_hot(2))


def test_windows_step_forward_by_the_window_less_its_overlap() -> None:
    """The step is what gives an edge token a window in which it is not on an edge."""
    assert _windows(20, 8) == [(0, 8), (6, 14), (12, 20)]
    assert _windows(5, 8) == [(0, 5)]


class ZeroWidthTokenizer(FakeTokenizer):
    """Reports a zero-width token between the words, as tokenizers do for dropped marks."""

    def __call__(
        self, text: str, add_special_tokens: bool = True, return_offsets_mapping: bool = False
    ) -> dict[str, Any]:
        """Return the word tokens plus one token that spans no characters at all."""
        base = super().__call__(text, add_special_tokens, return_offsets_mapping)
        ids: list[int] = list(base["input_ids"])
        offsets: list[tuple[int, int]] = list(base["offset_mapping"])
        gap = offsets[0][1]
        return {
            "input_ids": [ids[0], len(ids) + 1, *ids[1:]],
            "offset_mapping": [offsets[0], (gap, gap), *offsets[1:]],
        }


class ZeroWidthModel(OneHotModel):
    """A one-hot stand-in whose tokenizer emits a token covering no characters."""

    def __init__(self) -> None:
        """Build the model with the zero-width tokenizer in place."""
        super().__init__()
        self.tokenizer: Any = ZeroWidthTokenizer()


def test_a_token_spanning_no_characters_is_never_pooled() -> None:
    """A zero-width token sits inside every span around it, so admitting one dilutes them all."""
    vectors = pool_spans(ZeroWidthModel(), "alpha beta", [(0, 10)])

    assert vectors[0] == pytest.approx(one_hot(1, 2))


POSITION_WIDTH = 4
WINDOWED_TEXT = " ".join(f"w{index:02d}" for index in range(20))


class PositionNetwork:
    """Returns a row naming the call that produced it and the slot it occupied.

    Both other fakes answer by token id alone, so a token pooled from the window where
    it is least contextual produces exactly the same vector as a correct one.
    """

    def __init__(self) -> None:
        self.config = type("c", (), {"hidden_size": POSITION_WIDTH})()
        self.calls = 0
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def parameters(self) -> Any:
        return iter([self._parameter])

    def __call__(self, input_ids: Any, attention_mask: Any) -> Any:
        """Return one row per slot, encoding this call's number and the slot's index."""
        self.calls += 1
        rows = torch.stack(
            [
                torch.tensor([float(self.calls), float(slot), 1.0, 1.0])
                for slot in range(int(input_ids.shape[1]))
            ]
        )
        return type("o", (), {"last_hidden_state": rows.unsqueeze(0)})()


class PositionModel:
    """A stand-in whose vectors say which window a token's representation came from."""

    def __init__(self, max_seq_length: int) -> None:
        self.tokenizer = FakeTokenizer()
        self.network = PositionNetwork()
        self.max_seq_length = max_seq_length

    def __getitem__(self, index: int) -> Any:
        """Return the transformer module, as sentence-transformers exposes it."""
        return type("m", (), {"auto_model": self.network})()


def from_window(call: int, slot: int) -> list[float]:
    """Return the unit vector a token taken from window ``call`` at ``slot`` pools to."""
    row = torch.tensor([float(call), float(slot), 1.0, 1.0])
    return [float(value) for value in torch.nn.functional.normalize(row, dim=0).tolist()]


def test_a_token_takes_its_vector_from_the_window_it_sits_deepest_inside() -> None:
    """Centrality is the whole point of windowing, and choosing wrong is silent.

    Slot 0 of every window holds the CLS sentinel, so a token at window offset ``n``
    reports slot ``n + 1``.
    """
    model = PositionModel(max_seq_length=12)
    spans = [(index * 4, index * 4 + 3) for index in (8, 9, 16, 17)]

    vectors = pool_spans(model, WINDOWED_TEXT, spans)

    assert model.network.calls == 3
    assert vectors[0] == pytest.approx(from_window(1, 9))
    assert vectors[1] == pytest.approx(from_window(2, 2))
    assert vectors[2] == pytest.approx(from_window(2, 9))
    assert vectors[3] == pytest.approx(from_window(3, 2))


TIED_TEXT = " ".join(f"t{index:02d}" for index in range(8))


def test_a_centrality_tie_is_kept_by_the_window_that_already_holds_the_token() -> None:
    """A later window that ties is not more contextual, and the last one is the shortest.

    Windows (0, 4), (3, 7) and (6, 8) tie on edge distance at tokens 3 and 6. Conceding
    the tie hands token 6 to a two-token window instead of the four-token one holding it.
    """
    model = PositionModel(max_seq_length=6)

    vectors = pool_spans(model, TIED_TEXT, [(12, 15), (24, 27)])

    assert model.network.calls == 3
    assert vectors[0] == pytest.approx(from_window(1, 4))
    assert vectors[1] == pytest.approx(from_window(2, 4))


class ContextlessModel:
    """A stand-in that declares no context length, as a bare transformer module does not."""

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.network = FakeNetwork()

    def __getitem__(self, index: int) -> Any:
        """Return the transformer module, as sentence-transformers exposes it."""
        return type("m", (), {"auto_model": self.network})()


def test_a_model_that_declares_no_context_length_is_windowed_at_the_shipped_default() -> None:
    """Guessing high feeds the model more tokens than it can take and the pass fails.

    The default is 512 positions, of which the two sentinels take one each, so the
    widest tensor the model is ever handed is exactly 512 wide.
    """
    model = ContextlessModel()
    text = " ".join(f"w{index:04d}" for index in range(600))

    pool_spans(model, text, [(0, len(text))])

    assert len(model.network.windows) > 1
    assert max(model.network.windows) == 512


class SentinelBoundModel(OneHotModel):
    """A one-hot stand-in whose whole context is spent on the two sentinels."""

    def __init__(self) -> None:
        """Build the model with a context length equal to the sentinel budget."""
        super().__init__()
        self.max_seq_length = 2


def test_a_context_spent_entirely_on_sentinels_still_leaves_one_token_per_window() -> None:
    """Without the floor the window is empty, no state is ever written and the vector is zero."""
    vectors = pool_spans(SentinelBoundModel(), "alpha beta", [(0, 10)])

    assert vectors[0] == pytest.approx(one_hot(1, 2))


class ModuleListModel(OneHotModel):
    """A one-hot stand-in whose modules differ, as a real sentence-transformers model's do."""

    def __getitem__(self, index: int) -> Any:
        """Return the transformer at module 0; everything after it is pooling."""
        if index == 0:
            return type("m", (), {"auto_model": self.network})()
        return type("p", (), {"word_embedding_dimension": ONE_HOT_WIDTH})()


def test_the_transformer_is_read_from_the_first_module_and_not_a_later_one() -> None:
    """Module 0 is the transformer; the modules behind it pool and normalize its output.

    Both other fakes answer every index with the transformer, so reading the wrong
    module looks identical to reading the right one.
    """
    model = ModuleListModel()

    assert supports_pooling(model) is True
    assert pool_spans(model, TEXT, [(0, 5)])[0] == pytest.approx(one_hot(1))


class TokenizerOnlyModel:
    """Carries a tokenizer and no modules at all, as a hosted-provider wrapper does."""

    tokenizer = FakeTokenizer()


def test_a_model_whose_modules_cannot_be_indexed_is_refused() -> None:
    """Answering yes here fails deep inside the pass instead of falling back to embedding."""
    assert supports_pooling(TokenizerOnlyModel()) is False


class StrictTokenizer(FakeTokenizer):
    """Honours both flags: wraps the ids in sentinels, and withholds offsets unless asked."""

    def __call__(
        self, text: str, add_special_tokens: bool = True, return_offsets_mapping: bool = False
    ) -> dict[str, Any]:
        """Return the word tokens, wrapped and annotated only as the flags allow."""
        base = super().__call__(text, False, True)
        ids: list[int] = list(base["input_ids"])
        offsets: list[tuple[int, int]] = list(base["offset_mapping"])
        if add_special_tokens:
            ids = [self.cls_token_id, *ids, self.sep_token_id]
            offsets = [(0, 0), *offsets, (len(text), len(text))]
        payload: dict[str, Any] = {"input_ids": ids}
        if return_offsets_mapping:
            payload["offset_mapping"] = offsets
        return payload


class StrictModel(OneHotModel):
    """A one-hot stand-in whose tokenizer obeys the flags pooling passes it."""

    def __init__(self) -> None:
        """Build the model with the flag-honouring tokenizer and a short context."""
        super().__init__()
        self.tokenizer: Any = StrictTokenizer()
        self.max_seq_length = 10


def test_the_document_is_tokenized_without_the_models_own_sentinels() -> None:
    """Pooling wraps every window itself, so a second pair eats two slots of each one.

    Eight tokens fill one window of a ten-position context exactly; asking the
    tokenizer for sentinels too pushes the document into a second, shorter window.
    """
    model = StrictModel()
    text = " ".join(f"t{index:02d}" for index in range(8))

    pool_spans(model, text, [(0, 3)])

    assert model.network.windows == [10]
