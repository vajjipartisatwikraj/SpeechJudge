"""Question_Router: exactly one evaluator per Question_Type (Requirement 8)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from app.models.domain import QuestionType
from app.evaluators.base import QuestionEvaluator
from app.evaluators.jumbled import JumbledEvaluator
from app.evaluators.question_answer import QuestionAnswerEvaluator
from app.evaluators.reading import ReadingEvaluator
from app.evaluators.repeat import RepeatEvaluator
from app.evaluators.storytelling import StorytellingEvaluator
from app.scoring.config import ScoringConfig
from app.services.registry import ServiceRegistry


class QuestionRouter:
    def __init__(self, services: ServiceRegistry, scoring: ScoringConfig,
                 executor: ThreadPoolExecutor) -> None:
        # The same shared service instances are handed to every evaluator (Req 7.2).
        self._evaluators: dict[QuestionType, QuestionEvaluator] = {
            QuestionType.READING: ReadingEvaluator(services, scoring, executor),
            QuestionType.REPEAT: RepeatEvaluator(services, scoring, executor),
            QuestionType.JUMBLED: JumbledEvaluator(services, scoring, executor),
            QuestionType.QUESTION_ANSWER: QuestionAnswerEvaluator(services, scoring, executor),
            QuestionType.STORYTELLING: StorytellingEvaluator(services, scoring, executor),
        }

    def route(self, question_type: QuestionType) -> QuestionEvaluator:
        return self._evaluators[question_type]
