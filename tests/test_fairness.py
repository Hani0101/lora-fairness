"""Tests for the fairness module: CDA augmentation and PLL scoring mechanics.

The augmentation tests need no network access. The scoring tests use a tiny
hand-built vocabulary and a fake model rather than a real tokenizer/RoBERTa,
so they run fast and don't depend on HF hub availability -- they check the
scoring *mechanics* (masking, alignment, aggregation), not real bias numbers.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fairness.cda import augment_corpus, swap_gendered_terms  # noqa: E402
from src.fairness.crows_pairs import (  # noqa: E402
    BiasResult,
    CrowsPair,
    _unmodified_positions,
    evaluate_bias,
)


# ---------------------------------------------------------------------------
# CDA augmentation
# ---------------------------------------------------------------------------

def test_swap_basic_pronoun():
    assert swap_gendered_terms("He went to the store.") == "She went to the store."


def test_swap_preserves_case():
    assert swap_gendered_terms("HE and his dog left.") == "SHE and her dog left."
    assert swap_gendered_terms("he and his dog left.") == "she and her dog left."


def test_swap_is_symmetric():
    assert swap_gendered_terms("She is his brother.") == "He is her sister."


def test_swap_no_match_returns_none():
    assert swap_gendered_terms("The weather is nice today.") is None


def test_swap_does_not_touch_word_fragments():
    # "help" contains "he" but must not be partially replaced.
    assert swap_gendered_terms("Please help me.") is None


def test_swap_matches_whole_word_that_is_itself_a_pair():
    # "hero" is itself a listed pair (hero/heroine), so it SHOULD swap --
    # this is a real match, not a fragment false-positive.
    assert swap_gendered_terms("The hero arrived.") == "The heroine arrived."


def test_augment_corpus_duplicates_only_matching_sentences():
    sentences = ["He is a doctor.", "The sky is blue.", "Her brother called."]
    augmented = augment_corpus(sentences)
    # 2 sentences had matches (get originals + counterfactuals), 1 didn't.
    assert len(augmented) == 5
    assert "He is a doctor." in augmented
    assert "She is a doctor." in augmented
    assert "The sky is blue." in augmented
    assert "Her brother called." in augmented
    assert "His sister called." in augmented


# ---------------------------------------------------------------------------
# CrowS-Pairs scoring mechanics
# ---------------------------------------------------------------------------

def test_unmodified_positions_finds_shared_tokens():
    # "the cat sat" vs "the dog sat" -> position 0 ("the") and 2 ("sat") shared.
    ids_a = [10, 11, 12]  # the cat sat
    ids_b = [10, 13, 12]  # the dog sat
    shared = _unmodified_positions(ids_a, ids_b)
    assert 0 in shared
    assert 2 in shared
    assert 1 not in shared


def test_unmodified_positions_identical_sentences():
    ids = [5, 6, 7, 8]
    shared = _unmodified_positions(ids, ids)
    assert shared == [0, 1, 2, 3]


class _FakeTokenizer:
    """Minimal whitespace tokenizer standing in for a real HF tokenizer."""

    mask_token_id = 0
    all_special_ids = {0}

    def __init__(self):
        self.vocab = {"<mask>": 0}

    def _id(self, word: str) -> int:
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab)
        return self.vocab[word]

    def __call__(self, text, return_tensors=None):
        ids = [self._id(w) for w in text.split()]
        return {"input_ids": torch.tensor([ids])}


class _FakeMLM(nn.Module):
    """Returns higher logits for token 1 than any other token, always.

    So this "model" always prefers whichever sentence's unmodified tokens
    happen to be token id 1 more often -- used only to check that scoring
    and aggregation wire together correctly, not to test real bias.
    """

    def __init__(self, vocab_size: int = 20):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        logits[:, :, 1] = 5.0  # token id 1 is always the "preferred" token
        return type("Output", (), {"logits": logits})()


def test_evaluate_bias_runs_and_aggregates():
    tokenizer = _FakeTokenizer()
    model = _FakeMLM()
    pairs = [
        CrowsPair("the cat sat", "the dog sat", "stereo", "gender"),
        CrowsPair("a bird flew", "a fish swam", "stereo", "race-color"),
    ]
    result = evaluate_bias(pairs, model, tokenizer, torch.device("cpu"))

    assert isinstance(result, BiasResult)
    assert result.n_pairs == 2
    assert 0.0 <= result.pct_stereotype <= 100.0
    assert set(result.by_bias_type.keys()) == {"gender", "race-color"}


def test_evaluate_bias_empty_list():
    result = evaluate_bias([], _FakeMLM(), _FakeTokenizer(), torch.device("cpu"))
    assert result.n_pairs == 0
    assert result.pct_stereotype == 0.0
