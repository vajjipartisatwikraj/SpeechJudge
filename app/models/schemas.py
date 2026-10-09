"""Public API contract (request/response models)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.models.domain import EvaluationStatus, JobStatus, QuestionType


class EvaluationRequest(BaseModel):
    request_id: str
    question_type: QuestionType
    audio: str | None = Field(default=None, description="Base64-encoded audio (WAV/MP3/OGG/WebM/M4A)")
    expected_text: str | None = None
    question: str | None = None
    question_config: dict[str, Any] = Field(default_factory=dict)


class EvaluationDetails(BaseModel):
    transcription: dict[str, Any] | None = None
    gopt: dict[str, Any] | None = None
    acoustic: dict[str, Any] | None = None
    text_comparison: dict[str, Any] | None = None
    language: dict[str, Any] | None = None
    score_components: list[dict[str, Any]] | None = None


class EvaluationResult(BaseModel):
    request_id: str
    status: EvaluationStatus
    question_type: QuestionType
    score: float | None = None
    reason: str | None = None
    flags: list[str] = Field(default_factory=list)
    raw_metrics: dict[str, float] | None = None  # kept separate from the final score (Req 15.14)
    evaluation: EvaluationDetails = Field(default_factory=EvaluationDetails)


class AsyncSubmitResponse(BaseModel):
    request_id: str
    job_id: str
    status: JobStatus


class JobResponse(BaseModel):
    job_id: str
    request_id: str
    status: JobStatus
    result: EvaluationResult | None = None
    reason: str | None = None


# --- Sections: a whole exam section (e.g. 10 Repeat questions) in one request -----------------
_SECTION_ALIASES = {
    "QA": "QUESTION_ANSWER", "Q&A": "QUESTION_ANSWER", "QUESTION_AND_ANSWER": "QUESTION_ANSWER",
    "QUESTIONANSWER": "QUESTION_ANSWER", "STORY": "STORYTELLING", "STORY_TELLING": "STORYTELLING",
}


class SectionQuestion(BaseModel):
    question_id: str
    audio: str | None = Field(default=None, description="Base64-encoded audio")
    expected_text: str | None = None
    question: str | None = None
    question_config: dict[str, Any] = Field(default_factory=dict)


class SectionRequest(BaseModel):
    request_id: str
    student_id: str | None = None
    section_id: QuestionType = Field(description="READING | REPEAT | JUMBLED | QUESTION_ANSWER | STORYTELLING")
    questions: list[SectionQuestion] = Field(min_length=1)
    include_details: bool = Field(default=False, description="Add the full per-question evaluation to the results")

    @field_validator("section_id", mode="before")
    @classmethod
    def _normalise_section(cls, value: Any) -> Any:
        if isinstance(value, str):
            key = value.strip().upper().replace("-", "_").replace(" ", "_")
            return _SECTION_ALIASES.get(key, key)
        return value


class SectionQuestionResult(BaseModel):
    question_id: str
    status: EvaluationStatus
    score: float | None = None
    reason: str | None = None
    flags: list[str] = Field(default_factory=list)
    detail: EvaluationResult | None = None      # only when include_details is true


class SectionResult(BaseModel):
    section_id: QuestionType
    student_id: str | None = None
    results: list[SectionQuestionResult]
    section_score: float | None = None          # mean score of the questions that were scored
    questions_total: int
    questions_scored: int


class SectionTiming(BaseModel):
    queued_s: float | None = None               # waiting in the Redis queue
    processing_s: float | None = None           # worker time (all questions of the section)
    total_s: float | None = None                # submit -> finished


class SectionJobResponse(BaseModel):
    job_id: str
    request_id: str
    status: JobStatus
    section_id: QuestionType | None = None
    student_id: str | None = None
    questions_total: int | None = None
    questions_scored: int | None = None
    section_score: float | None = None
    results: list[SectionQuestionResult] | None = None
    timing: SectionTiming | None = None
    reason: str | None = None


class ErrorResponse(BaseModel):
    detail: str
    errors: list[str] = Field(default_factory=list)
    request_id: str | None = None
