"""Evaluation_Request validation (Requirement 4)."""
from __future__ import annotations

from app.core.config import Settings
from app.core.errors import ApiError
from app.models.domain import QuestionType
from app.evaluators.jumbled import get_submitted_text
from app.models.schemas import EvaluationRequest, SectionRequest

MAX_REQUEST_ID_LEN = 256


def _blank(value: str | None) -> bool:
    return value is None or not value.strip()


def validate_request_id(request_id: str | None) -> None:
    if request_id is None or not request_id.strip():
        raise ApiError(422, "request_id must be a non-empty string", details=["request_id"])
    if len(request_id) > MAX_REQUEST_ID_LEN or any(ord(c) < 32 or ord(c) == 127 for c in request_id):
        raise ApiError(422, "request_id is too long or contains control characters",
                       details=["request_id"])


def validate_request(req: EvaluationRequest) -> None:
    rid = req.request_id
    validate_request_id(rid)
    qt = req.question_type

    if qt in (QuestionType.READING, QuestionType.REPEAT):
        required = {"audio": req.audio, "expected_text": req.expected_text}
    elif qt in (QuestionType.QUESTION_ANSWER, QuestionType.STORYTELLING):
        required = {"audio": req.audio, "question": req.question}
    else:  # JUMBLED
        required = {"expected_text": req.expected_text}

    missing = [name for name, value in required.items() if _blank(value)]
    if missing:
        raise ApiError(422, f"Missing required field(s) for {qt.value}: {', '.join(missing)}",
                       request_id=rid, details=[f"{m} is required" for m in missing])

    if qt == QuestionType.JUMBLED and _blank(req.audio) and get_submitted_text(req.question_config) is None:
        raise ApiError(422, "JUMBLED requires one input mode: audio or question_config.submitted_text",
                       request_id=rid, details=["audio or question_config.submitted_text is required"])


def validate_section(section: SectionRequest, settings: Settings) -> None:
    """A section is accepted or rejected as a whole: every question must be well formed, so a typo
    in question 7 is reported now rather than after the other 9 were scored."""
    rid = section.request_id
    validate_request_id(rid)
    if len(section.questions) > settings.section_max_questions:
        raise ApiError(422, f"A section may contain at most {settings.section_max_questions} questions",
                       request_id=rid, details=[f"questions: {len(section.questions)} given"])

    problems: list[str] = []
    seen: set[str] = set()
    total_audio = 0
    for index, question in enumerate(section.questions, start=1):
        qid = question.question_id
        label = f"question {index}" + (f" ({qid})" if qid and qid.strip() else "")
        if _blank(qid) or len(qid) > MAX_REQUEST_ID_LEN or any(ord(c) < 32 or ord(c) == 127 for c in qid):
            problems.append(f"{label}: question_id must be a non-empty string without control characters")
            continue
        if qid in seen:
            problems.append(f"{label}: duplicate question_id")
            continue
        seen.add(qid)
        total_audio += len(question.audio or "") * 3 // 4
        single = EvaluationRequest(request_id=f"{rid}:{qid}", question_type=section.section_id,
                                   audio=question.audio, expected_text=question.expected_text,
                                   question=question.question, question_config=question.question_config)
        try:
            validate_request(single)
        except ApiError as exc:
            problems.extend(f"{label}: {d}" for d in (exc.details or [exc.message]))
    if problems:
        raise ApiError(422, "Invalid section", request_id=rid, details=problems)
    if total_audio > settings.section_max_total_audio_bytes:
        raise ApiError(413, "Section audio limit exceeded", request_id=rid)
