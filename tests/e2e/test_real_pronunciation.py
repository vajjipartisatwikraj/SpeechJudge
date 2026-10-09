"""Phoneme pronunciation service on real generated voices (US: david, zira; Indian: ravi, heera).

Relative behaviour only: TTS speech is clean, so these checks prove the pipeline discriminates
(correct > truncated/wrong) and that the Indian-English profile avoids false penalties.
"""
from __future__ import annotations

import dataclasses

import pytest

from app.services.pronunciation.phoneme_service import PhonemePronunciationService

CORRECT = "The doctor suggested that the apple is good for your health."
VOICES = ["david", "zira", "ravi", "heera"]
INDIAN = ("ravi", "heera")


@pytest.fixture(scope="module")
def service(real_settings):
    svc = PhonemePronunciationService(real_settings)
    svc.load()
    return svc


@pytest.fixture(scope="module")
def assess(service, load_clip):
    cache = {}

    def _assess(voice, clip, text):
        key = (voice, clip, text)
        if key not in cache:
            cache[key] = service.assess(load_clip(f"{voice}_{clip}"), text)
        return cache[key]
    return _assess


def word_accuracy(result, word):
    return next(w["accuracy"] for w in result.words if w["word"] == word)


@pytest.mark.parametrize("voice", VOICES)
def test_correct_reading_is_not_penalised(assess, voice):
    floor = 0.85 if voice in INDIAN else 0.65   # US voices are scored against an Indian-tolerant reference
    assert assess(voice, "correct", CORRECT).pronunciation >= floor


@pytest.mark.parametrize("voice", VOICES)
def test_different_sentence_scores_low(assess, voice):
    assert assess(voice, "question", CORRECT).pronunciation < 0.3


@pytest.mark.parametrize("voice", VOICES)
def test_truncated_reading_lowers_completeness_and_score(assess, voice):
    full, cut = assess(voice, "correct", CORRECT), assess(voice, "truncated", CORRECT)
    assert cut.completeness < full.completeness and cut.pronunciation < full.pronunciation


@pytest.mark.parametrize("voice", VOICES)
def test_wrong_word_is_flagged_on_that_word(assess, voice):
    good, wrong = assess(voice, "correct", CORRECT), assess(voice, "wrongword", CORRECT)
    assert word_accuracy(wrong, "health") < word_accuracy(good, "health")


@pytest.mark.parametrize("voice", INDIAN)
def test_accent_profile_removes_false_penalties(service, assess, load_clip, voice):
    tolerant = assess(voice, "correct", CORRECT).pronunciation
    service._profile, original = dataclasses.replace(
        service._profile, accepted={}, equivalent_groups=[], near=frozenset(), _group_of={}), service._profile
    try:
        strict = service.assess(load_clip(f"{voice}_correct"), CORRECT).pronunciation
    finally:
        service._profile = original
    assert tolerant > strict + 0.1
