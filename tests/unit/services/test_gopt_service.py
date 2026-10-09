import pytest

from app.core.errors import ServiceLoadError
from app.services.pronunciation.gopt_service import GoptPronunciationService, normalize_gopt_output
from tests.support import make_settings


def test_gopt_normalization_to_unit_range():
    r = normalize_gopt_output({
        "utterance": {"accuracy": 8, "completeness": 10, "fluency": 7, "prosodic": 5, "total": 7},
        "words": [{"word": "hi", "accuracy": 9, "stress": 10, "total": 9}],
        "phones": [{"phone": "HH", "word_index": 0, "accuracy": 1.5}]})
    assert (r.pronunciation, r.fluency, r.prosody, r.completeness) == (0.8, 0.7, 0.5, 1.0)
    assert r.words[0]["accuracy"] == 0.9 and r.phonemes[0]["accuracy"] == 0.75


def test_gopt_service_requires_config():
    with pytest.raises(ServiceLoadError):
        GoptPronunciationService(make_settings()).load()
