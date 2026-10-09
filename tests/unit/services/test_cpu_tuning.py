"""CPU thread budget and compact (score-only) Qwen output."""
import sys
import types

import pytest

from app.models.domain import QuestionType
from app.services.language.ollama_service import OllamaLanguageEvaluator
from app.services.language.prompts import build_messages, rubric_for
from app.services.language.qwen_service import QwenLanguageEvaluator, parse_language_output
from app.services.stt.whisper_service import WhisperService
from tests.support import make_settings

QA = rubric_for(QuestionType.QUESTION_ANSWER)
COMPACT = "{" + ",".join(f'"{d}": 0.6' for d in QA) + "}"


# ---- thread budget -----------------------------------------------------------------------
def test_defaults_leave_every_library_on_its_own_defaults():
    s = make_settings()
    assert (s.whisper_cpu_threads, s.phoneme_threads, s.ollama_num_thread) == (0, 0, 0)
    assert s.llm_justifications is True


def test_ollama_sends_num_thread_only_when_configured():
    assert "num_thread" not in OllamaLanguageEvaluator(make_settings())._options()
    opts = OllamaLanguageEvaluator(make_settings(ollama_num_thread=2))._options()
    assert opts["num_thread"] == 2 and opts["temperature"] == 0


def test_ollama_request_carries_the_thread_option():
    svc = OllamaLanguageEvaluator(make_settings(ollama_num_thread=3))
    sent = {}
    svc._post = lambda path, payload, timeout: sent.update(payload) or {"message": {"content": "{}"}}
    svc._generate([{"role": "user", "content": "x"}])
    assert sent["options"]["num_thread"] == 3 and sent["format"] == "json"


@pytest.mark.parametrize("threads,expect", [(0, {}), (3, {"cpu_threads": 3})])
def test_whisper_receives_cpu_threads_only_when_configured(monkeypatch, threads, expect):
    seen = {}

    class FakeModel:
        def __init__(self, name, **kwargs):
            seen.update(kwargs)
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeModel))
    svc = WhisperService(make_settings(whisper_cpu_threads=threads, device="cpu", whisper_compute_type="int8"))
    svc.load()
    assert seen.get("cpu_threads") == expect.get("cpu_threads") and ("cpu_threads" in seen) == bool(expect)


# ---- compact Qwen output -----------------------------------------------------------------
def test_compact_prompt_asks_for_bare_numbers():
    full = build_messages(QuestionType.QUESTION_ANSWER, "Q?", "answer")[1]["content"]
    compact = build_messages(QuestionType.QUESTION_ANSWER, "Q?", "answer", justifications=False)[1]["content"]
    assert "justification" in full and "justification" not in compact
    assert len(compact) < len(full)


def test_parser_accepts_bare_scores_only_in_compact_mode():
    ok = parse_language_output(COMPACT, QA, justifications=False)
    assert ok and ok["relevance"] == {"score": 0.6, "justification": ""}
    assert parse_language_output(COMPACT, QA, justifications=True) is None         # strict mode unchanged
    with_dict = "{" + ",".join(f'"{d}": {{"score": 0.7}}' for d in QA) + "}"
    assert parse_language_output(with_dict, QA, justifications=False) is not None  # dict without text is fine
    assert parse_language_output(COMPACT.replace("0.6", "1.7", 1), QA, justifications=False) is None


def test_language_evaluator_in_compact_mode_end_to_end():
    class FakeQwen(QwenLanguageEvaluator):
        seen_prompt = ""

        def _generate(self, messages):
            FakeQwen.seen_prompt = messages[1]["content"]
            return COMPACT

    res = FakeQwen(make_settings(llm_justifications=False)).evaluate(QuestionType.QUESTION_ANSWER, "Q?", "answer")
    assert res.scores()["grammar"] == 0.6 and "justification" not in FakeQwen.seen_prompt
