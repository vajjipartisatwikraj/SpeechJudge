"""Storytelling evaluator: same flow as Q&A with the narrative rubric."""
from __future__ import annotations

from app.models.domain import QuestionType
from app.evaluators.question_answer import OpenEndedEvaluator


class StorytellingEvaluator(OpenEndedEvaluator):
    question_type = QuestionType.STORYTELLING
    scoring_key = QuestionType.STORYTELLING.value
