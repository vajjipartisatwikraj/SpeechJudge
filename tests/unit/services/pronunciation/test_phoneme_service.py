import numpy as np

from app.core.config import get_settings
from app.models.domain import AudioData
from app.services.pronunciation.phoneme_service import PhonemePronunciationService
from app.services.pronunciation.phonemes import load_profile
from tests.support import SR, make_settings

PROFILE = load_profile(get_settings().accent_profile_path)


# ---- service glue with fakes (no model, no espeak) ----------------------------------------------
class FakePhoneme(PhonemePronunciationService):
    LEX = {"think": ["θ", "ɪ", "ŋ", "k"], "cat": ["k", "æ", "t"]}

    def __init__(self, heard):
        super().__init__(make_settings(), profile=PROFILE)
        self.heard = heard

    def _phonemize_word(self, w):
        return self.LEX[w]

    def _recognize(self, samples):
        return self.heard


def test_service_returns_structured_result():
    svc = FakePhoneme(["t", "ɪ", "ŋ", "k", "k", "æ", "t"])
    res = svc.assess(AudioData(np.zeros(SR, dtype=np.float32), 1.0), "Think cat.")
    assert res.method == "phoneme_ctc" and res.fluency is None and res.prosody is None
    assert res.pronunciation == 1.0 and res.completeness == 1.0
    assert [w["word"] for w in res.words] == ["think", "cat"]
    assert res.details["accent_profile"] == "indian_english"
    bad = FakePhoneme(["t", "ɪ", "ŋ", "k"]).assess(AudioData(np.zeros(SR, dtype=np.float32), 1.0), "Think cat.")
    assert bad.completeness == 0.5 and bad.pronunciation < 1.0
