"""Counterfactual Data Augmentation (CDA) for gender bias mitigation.

Swaps gendered terms in a sentence (he <-> she, actor <-> actress, ...) to
produce a counterfactual duplicate. Fine-tuning on a corpus containing both
the original and swapped versions balances gendered associations that the
pretraining corpus wouldn't have had equally (Zhao et al., 2018; Zmigrod et
al., 2019).

Scope: gender only. CrowS-Pairs also covers race, religion, age, and other
categories that this augmentation does not address -- see README.
"""

from __future__ import annotations

import re

# Curated gendered word-pairs in the spirit of Zhao et al. (2018)'s list.
# Not a verified reproduction of their exact 193-pair list -- a representative
# set covering pronouns, kinship terms, titles, and common gendered nouns.
GENDER_PAIRS: list[tuple[str, str]] = [
    ("he", "she"), ("him", "her"), ("his", "her"),
    ("himself", "herself"), ("man", "woman"), ("men", "women"),
    ("boy", "girl"), ("boys", "girls"), ("father", "mother"),
    ("fathers", "mothers"), ("dad", "mom"), ("daddy", "mommy"),
    ("son", "daughter"), ("sons", "daughters"), ("brother", "sister"),
    ("brothers", "sisters"), ("husband", "wife"), ("husbands", "wives"),
    ("grandfather", "grandmother"), ("grandpa", "grandma"),
    ("uncle", "aunt"), ("uncles", "aunts"), ("nephew", "niece"),
    ("king", "queen"), ("kings", "queens"), ("prince", "princess"),
    ("lord", "lady"), ("gentleman", "lady"), ("gentlemen", "ladies"),
    ("actor", "actress"), ("actors", "actresses"), ("waiter", "waitress"),
    ("host", "hostess"), ("steward", "stewardess"), ("hero", "heroine"),
    ("widower", "widow"), ("groom", "bride"), ("bachelor", "bachelorette"),
    ("mr", "mrs"), ("sir", "madam"), ("guy", "gal"), ("guys", "gals"),
    ("male", "female"), ("males", "females"), ("boyfriend", "girlfriend"),
    ("businessman", "businesswoman"), ("chairman", "chairwoman"),
    ("policeman", "policewoman"), ("spokesman", "spokeswoman"),
    ("congressman", "congresswoman"), ("fireman", "firewoman"),
    ("salesman", "saleswoman"), ("mailman", "mailwoman"),
]

# Build a case-insensitive lookup covering both directions of every pair.
_SWAP_MAP: dict[str, str] = {}
for a, b in GENDER_PAIRS:
    _SWAP_MAP[a.lower()] = b.lower()
    _SWAP_MAP[b.lower()] = a.lower()

_WORD_RE = re.compile(r"\b[a-zA-Z]+\b")


def _match_case(source: str, target: str) -> str:
    """Apply source's capitalization pattern to target, so 'He' -> 'She' not 'she'."""
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def swap_gendered_terms(sentence: str) -> str | None:
    """Return a gender-swapped counterfactual, or None if nothing matched."""
    changed = False

    def _replace(match: re.Match) -> str:
        nonlocal changed
        word = match.group(0)
        swapped = _SWAP_MAP.get(word.lower())
        if swapped is None:
            return word
        changed = True
        return _match_case(word, swapped)

    result = _WORD_RE.sub(_replace, sentence)
    return result if changed else None


def augment_corpus(sentences: list[str]) -> list[str]:
    """Return the original sentences plus a counterfactual for each that matched.

    This is the standard CDA recipe: every sentence appears once as-is, and
    once more, gender-swapped, if it contained a gendered term at all. A
    sentence with no gendered term contributes only its original form.
    """
    augmented: list[str] = []
    for sentence in sentences:
        augmented.append(sentence)
        counterfactual = swap_gendered_terms(sentence)
        if counterfactual is not None:
            augmented.append(counterfactual)
    return augmented
