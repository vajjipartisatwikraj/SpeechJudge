"""Repeat evaluator: same services as Reading, Repeat-specific scoring weights."""
from __future__ import annotations

from app.models.domain import QuestionType
from app.evaluators.reading import PronunciationEvaluator


class RepeatEvaluator(PronunciationEvaluator):
    question_type = QuestionType.REPEAT
    scoring_key = QuestionType.REPEAT.value
