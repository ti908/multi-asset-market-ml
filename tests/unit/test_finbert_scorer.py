"""Tests for the FinBERT sentiment scorer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from market_prediction.sentiment.finbert import (
    FinBertScorer,
)
from market_prediction.sentiment.scorer import (
    SentimentScores,
)


MODEL_NAME = "ProsusAI/finbert"
MODEL_REVISION = (
    "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
)


class FakeBatch(dict):
    """Minimal tokenizer batch supporting device transfer."""

    def __init__(self) -> None:
        super().__init__(
            {
                "input_ids": torch.tensor(
                    [[101, 102]]
                ),
                "attention_mask": torch.tensor(
                    [[1, 1]]
                ),
            }
        )
        self.moved_to: str | None = None

    def to(self, device: str):
        self.moved_to = device

        for key, value in self.items():
            self[key] = value.to(device)

        return self


class FakeTokenizer:
    """Capture tokenizer arguments without network access."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.last_batch: FakeBatch | None = None

    def __call__(
        self,
        text: str,
        **kwargs,
    ) -> FakeBatch:
        self.calls.append(
            {
                "text": text,
                **kwargs,
            }
        )

        batch = FakeBatch()
        self.last_batch = batch
        return batch


class FakeModel:
    """Return deterministic three-class logits."""

    def __init__(
        self,
        *,
        id2label: dict[int, str],
        logits: list[float] | None = None,
    ) -> None:
        self.config = SimpleNamespace(
            id2label=id2label
        )
        self.logits = logits or [
            3.0,
            1.0,
            0.0,
        ]
        self.moved_to: str | None = None
        self.eval_called = False
        self.calls = 0

    def to(self, device: str):
        self.moved_to = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **kwargs):
        self.calls += 1

        device = kwargs["input_ids"].device

        return SimpleNamespace(
            logits=torch.tensor(
                [self.logits],
                dtype=torch.float32,
                device=device,
            )
        )


def make_scorer(
    *,
    id2label: dict[int, str] | None = None,
    logits: list[float] | None = None,
    device: str = "cpu",
    max_tokens: int = 512,
) -> tuple[
    FinBertScorer,
    FakeTokenizer,
    FakeModel,
]:
    """Return a scorer using injected offline test doubles."""

    tokenizer = FakeTokenizer()

    model = FakeModel(
        id2label=(
            id2label
            or {
                0: "positive",
                1: "negative",
                2: "neutral",
            }
        ),
        logits=logits,
    )

    scorer = FinBertScorer(
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        max_tokens=max_tokens,
        device=device,
        tokenizer=tokenizer,
        model=model,
    )

    return scorer, tokenizer, model


def test_finbert_scorer_exposes_model_metadata() -> None:
    scorer, _, _ = make_scorer()

    assert scorer.model_name == MODEL_NAME
    assert scorer.model_revision == MODEL_REVISION
    assert scorer.scoring_method == "finbert"
    assert scorer.max_tokens == 512
    assert scorer.device == "cpu"


def test_finbert_scorer_moves_model_and_sets_eval() -> None:
    _, _, model = make_scorer()

    assert model.moved_to == "cpu"
    assert model.eval_called is True


def test_finbert_tokenizer_uses_configured_truncation() -> None:
    scorer, tokenizer, _ = make_scorer(
        max_tokens=256
    )

    scorer.score(
        "Apple reported record quarterly revenue."
    )

    assert tokenizer.calls == [
        {
            "text": (
                "Apple reported record quarterly revenue."
            ),
            "return_tensors": "pt",
            "truncation": True,
            "max_length": 256,
        }
    ]


def test_finbert_batch_is_moved_to_device() -> None:
    scorer, tokenizer, _ = make_scorer()

    scorer.score("Apple reported revenue.")

    assert tokenizer.last_batch is not None
    assert tokenizer.last_batch.moved_to == "cpu"


def test_finbert_score_returns_sentiment_scores() -> None:
    scorer, _, _ = make_scorer()

    result = scorer.score(
        "Apple reported record quarterly revenue."
    )

    assert isinstance(result, SentimentScores)


def test_finbert_probabilities_sum_to_one() -> None:
    scorer, _, _ = make_scorer()

    result = scorer.score(
        "Apple reported record quarterly revenue."
    )

    total = (
        result.positive_probability
        + result.neutral_probability
        + result.negative_probability
    )

    assert total == pytest.approx(1.0)


def test_finbert_uses_model_id2label_not_fixed_indices() -> None:
    scorer, _, _ = make_scorer(
        id2label={
            0: "neutral",
            1: "positive",
            2: "negative",
        },
        logits=[
            0.0,
            4.0,
            1.0,
        ],
    )

    result = scorer.score(
        "Apple reported record quarterly revenue."
    )

    assert result.positive_probability > 0.90
    assert (
        result.positive_probability
        > result.negative_probability
    )
    assert (
        result.positive_probability
        > result.neutral_probability
    )
    assert result.predicted_label == "positive"


def test_finbert_label_mapping_is_case_insensitive() -> None:
    scorer, _, _ = make_scorer(
        id2label={
            0: "POSITIVE",
            1: "NEGATIVE",
            2: "NEUTRAL",
        }
    )

    result = scorer.score(
        "Apple reported record quarterly revenue."
    )

    assert result.predicted_label == "positive"


@pytest.mark.parametrize(
    "id2label",
    [
        {
            0: "positive",
            1: "negative",
        },
        {
            0: "positive",
            1: "negative",
            2: "other",
        },
        {
            0: "positive",
            1: "positive",
            2: "neutral",
        },
    ],
)
def test_finbert_rejects_invalid_label_mapping(
    id2label: dict[int, str],
) -> None:
    with pytest.raises(
        ValueError,
        match="FinBERT label mapping",
    ):
        make_scorer(id2label=id2label)


@pytest.mark.parametrize(
    "input_text",
    [
        "",
        " ",
        "\n",
        "\t",
    ],
)
def test_finbert_rejects_blank_input_text(
    input_text: str,
) -> None:
    scorer, _, model = make_scorer()

    with pytest.raises(
        ValueError,
        match="input_text must not be blank",
    ):
        scorer.score(input_text)

    assert model.calls == 0


def test_finbert_rejects_non_string_input_text() -> None:
    scorer, _, model = make_scorer()

    with pytest.raises(TypeError):
        scorer.score(123)  # type: ignore[arg-type]

    assert model.calls == 0


@pytest.mark.parametrize(
    "max_tokens",
    [
        0,
        -1,
        513,
    ],
)
def test_finbert_rejects_invalid_max_tokens(
    max_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        make_scorer(max_tokens=max_tokens)


def test_finbert_rejects_boolean_max_tokens() -> None:
    with pytest.raises(TypeError):
        make_scorer(
            max_tokens=True  # type: ignore[arg-type]
        )

def test_from_pretrained_loads_exact_model_and_revision(
    monkeypatch,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(
        id2label={
            0: "positive",
            1: "negative",
            2: "neutral",
        }
    )

    tokenizer_calls = []
    model_calls = []

    def fake_tokenizer_loader(
        model_name,
        **kwargs,
    ):
        tokenizer_calls.append(
            (model_name, kwargs)
        )
        return tokenizer

    def fake_model_loader(
        model_name,
        **kwargs,
    ):
        model_calls.append(
            (model_name, kwargs)
        )
        return model

    monkeypatch.setattr(
        "market_prediction.sentiment.finbert."
        "AutoTokenizer.from_pretrained",
        fake_tokenizer_loader,
    )
    monkeypatch.setattr(
        "market_prediction.sentiment.finbert."
        "AutoModelForSequenceClassification."
        "from_pretrained",
        fake_model_loader,
    )

    scorer = FinBertScorer.from_pretrained(
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        max_tokens=512,
        device="cpu",
    )

    assert tokenizer_calls == [
        (
            MODEL_NAME,
            {
                "revision": MODEL_REVISION,
            },
        )
    ]

    assert model_calls == [
        (
            MODEL_NAME,
            {
                "revision": MODEL_REVISION,
            },
        )
    ]

    assert scorer.model_name == MODEL_NAME
    assert scorer.model_revision == MODEL_REVISION
    assert scorer.device == "cpu"


def test_from_pretrained_returns_working_scorer(
    monkeypatch,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(
        id2label={
            0: "positive",
            1: "negative",
            2: "neutral",
        }
    )

    monkeypatch.setattr(
        "market_prediction.sentiment.finbert."
        "AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "market_prediction.sentiment.finbert."
        "AutoModelForSequenceClassification."
        "from_pretrained",
        lambda *args, **kwargs: model,
    )

    scorer = FinBertScorer.from_pretrained(
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        max_tokens=512,
        device="cpu",
    )

    result = scorer.score(
        "Apple reported record revenue."
    )

    assert isinstance(result, SentimentScores)
    assert result.predicted_label == "positive"


def test_resolve_device_prefers_mps_when_available(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.backends.mps,
        "is_available",
        lambda: True,
    )

    assert (
        FinBertScorer.resolve_device()
        == "mps"
    )


def test_resolve_device_falls_back_to_cpu(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.backends.mps,
        "is_available",
        lambda: False,
    )

    assert (
        FinBertScorer.resolve_device()
        == "cpu"
    )


def test_from_pretrained_uses_resolved_device_when_omitted(
    monkeypatch,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(
        id2label={
            0: "positive",
            1: "negative",
            2: "neutral",
        }
    )

    monkeypatch.setattr(
        "market_prediction.sentiment.finbert."
        "AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "market_prediction.sentiment.finbert."
        "AutoModelForSequenceClassification."
        "from_pretrained",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        FinBertScorer,
        "resolve_device",
        staticmethod(lambda: "cpu"),
    )

    scorer = FinBertScorer.from_pretrained(
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        max_tokens=512,
    )

    assert scorer.device == "cpu"
    assert model.moved_to == "cpu"


def test_explicit_device_overrides_automatic_selection(
    monkeypatch,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(
        id2label={
            0: "positive",
            1: "negative",
            2: "neutral",
        }
    )

    monkeypatch.setattr(
        "market_prediction.sentiment.finbert."
        "AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "market_prediction.sentiment.finbert."
        "AutoModelForSequenceClassification."
        "from_pretrained",
        lambda *args, **kwargs: model,
    )

    def fail_if_called():
        raise AssertionError(
            "resolve_device must not be called "
            "when device is explicit"
        )

    monkeypatch.setattr(
        FinBertScorer,
        "resolve_device",
        staticmethod(fail_if_called),
    )

    scorer = FinBertScorer.from_pretrained(
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        max_tokens=512,
        device="cpu",
    )

    assert scorer.device == "cpu"