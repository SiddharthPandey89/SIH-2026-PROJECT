"""
backend/model_router/task_classifier.py

Deterministic, local/offline task classifier for the Sovereign AI Workbench.

Role of this module:
    - Look at an incoming user message (and whether it has an attachment)
      and decide which task type it belongs to, using the task types that
      model_registry.py already knows how to route:
      "chat", "code", "document_qa", "summarization", "spreadsheet", "vision".
    - Return a typed, explainable classification result for router.py to
      consume (score breakdown + matched signals), while still exposing a
      plain string-returning function for backward compatibility with the
      existing call in backend/api/chat.py:

          task_type = await classify_task(message, has_attachment=bool(payload.file_id))

Explicitly OUT of scope for this module (by design):
    - No model inference of any kind. Classification is done with local,
      deterministic keyword/pattern matching -- no network calls, no GPU,
      no external service, so it's instant and always available even before
      any LLM is loaded.

Extensibility:
    - New task types (or better signals for existing ones) can be added by
      calling register_signal_rule(...) at import time elsewhere, without
      editing the matching logic in this file.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Pattern

from pydantic import BaseModel, ConfigDict, Field

from backend.model_router.model_registry import (
    DEFAULT_TASK_TYPE,
    TASK_CHAT,
    TASK_CODE,
    TASK_DOCUMENT_QA,
    TASK_SPREADSHEET,
    TASK_SUMMARIZATION,
    TASK_VISION,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TaskClassification",
    "classify",
    "classify_task",
    "register_signal_rule",
    "get_registered_task_types",
]


# ---------------------------------------------------------------------------
# Typed result
# ---------------------------------------------------------------------------


class TaskClassification(BaseModel):
    """
    Typed classification result. router.py can use `task_type` directly for
    model selection, and `confidence` / `scores` / `matched_signals` for
    logging, UI display, or fallback decisions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_type: str = Field(..., description="Selected task type.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Share of total matched signal weight held by the winning task type."
    )
    scores: Dict[str, float] = Field(
        default_factory=dict, description="Raw signal score for every task type considered."
    )
    matched_signals: List[str] = Field(
        default_factory=list, description="Human-readable labels of the signals that fired for the winning task type."
    )


# ---------------------------------------------------------------------------
# Signal rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalRule:
    label: str
    pattern: Pattern[str]
    weight: float


_TASK_RULES: Dict[str, List[SignalRule]] = {}


def register_signal_rule(
    task_type: str,
    label: str,
    pattern: str,
    weight: float = 1.0,
    flags: int = re.IGNORECASE,
) -> None:
    """
    Register a new detection signal for a task type. This is the extension
    point for adding new task types (e.g. a future "translation" task) or
    strengthening detection of existing ones, without touching classify().
    """
    compiled = re.compile(pattern, flags)
    _TASK_RULES.setdefault(task_type, []).append(SignalRule(label=label, pattern=compiled, weight=weight))


def get_registered_task_types() -> List[str]:
    """All task types that currently have at least one signal rule registered."""
    return sorted(_TASK_RULES.keys())


def _register_default_rules() -> None:
    # --- CODE -----------------------------------------------------------
    register_signal_rule(TASK_CODE, "code_fence", r"```", weight=3.0)
    register_signal_rule(TASK_CODE, "inline_code", r"`[^`\n]+`", weight=1.0)
    register_signal_rule(TASK_CODE, "keyword_code_generic", r"\b(code|script|function|method|class|module|library|package)\b", weight=1.0)
    register_signal_rule(TASK_CODE, "keyword_debug", r"\b(bug|debug|error|exception|traceback|stack trace|crash|fix this)\b", weight=1.5)
    register_signal_rule(TASK_CODE, "keyword_dev_action", r"\b(refactor|implement|write a function|unit test|compile|algorithm|optimi[sz]e this)\b", weight=1.5)
    register_signal_rule(TASK_CODE, "python_syntax", r"\b(def |import |class |print\(|elif |self\.)", weight=2.0)
    register_signal_rule(TASK_CODE, "sql_syntax", r"\b(select .+ from|insert into|create table)\b", weight=2.0)
    register_signal_rule(TASK_CODE, "web_syntax", r"(</?\w+>|function\s*\(|const |let |=>)", weight=1.5)
    register_signal_rule(TASK_CODE, "file_extension", r"\.(py|js|ts|java|cpp|c|go|rs|sql|sh)\b", weight=1.5)

    # --- DOCUMENT_QA ------------------------------------------------------
    register_signal_rule(TASK_DOCUMENT_QA, "reference_phrase", r"\b(according to|as per|based on|refer to|as mentioned in|in the attached|in the uploaded)\b", weight=2.0)
    register_signal_rule(TASK_DOCUMENT_QA, "document_noun", r"\b(document|report|manual|sop|drawing|correspondence|inspection report|approval note)\b", weight=1.0)
    register_signal_rule(TASK_DOCUMENT_QA, "extraction_verb", r"\b(find|extract|pull out|look up|search for|locate)\b", weight=1.0)
    register_signal_rule(TASK_DOCUMENT_QA, "question_word", r"\b(who|what|when|where|why|how|which)\b", weight=0.5)

    # --- SUMMARIZATION ------------------------------------------------------
    register_signal_rule(TASK_SUMMARIZATION, "keyword_summarize", r"\b(summari[sz]e|summary|tl;dr|condense|shorten)\b", weight=2.5)
    register_signal_rule(TASK_SUMMARIZATION, "keyword_key_points", r"\b(key (points|findings)|main points|executive summary|brief overview|in short|highlights)\b", weight=2.0)

    # --- SPREADSHEET ------------------------------------------------------
    register_signal_rule(TASK_SPREADSHEET, "keyword_spreadsheet", r"\b(spreadsheet|excel|worksheet|workbook)\b", weight=2.5)
    register_signal_rule(TASK_SPREADSHEET, "file_extension", r"\.(xlsx|xls|csv)\b", weight=2.0)
    register_signal_rule(TASK_SPREADSHEET, "spreadsheet_terms", r"\b(pivot table|vlookup|formula|cell [a-z]?\d+|column [a-z]|budget|financial model)\b", weight=1.5)
    register_signal_rule(TASK_SPREADSHEET, "calc_phrase", r"\b(calculate the total|sum of|running total|cost breakdown)\b", weight=1.0)

    # --- VISION ------------------------------------------------------
    register_signal_rule(TASK_VISION, "keyword_visual", r"\b(image|photo|picture|photograph|screenshot|scanned)\b", weight=2.0)
    register_signal_rule(TASK_VISION, "keyword_drawing", r"\b(drawing|diagram|p&id|piping and instrument|blueprint|schematic|sketch)\b", weight=2.0)
    register_signal_rule(TASK_VISION, "keyword_visual_task", r"\b(what does this look like|read the handwriting|handwritten|ocr|inspect the drawing)\b", weight=2.0)
    register_signal_rule(TASK_VISION, "file_extension", r"\.(png|jpe?g|tiff?|bmp|heic)\b", weight=2.0)

    # --- CHAT (soft signals; CHAT is also the default fallback) -----------
    register_signal_rule(TASK_CHAT, "greeting", r"\b(hi|hello|hey|good morning|good evening)\b", weight=1.0)
    register_signal_rule(TASK_CHAT, "conversational", r"\b(thanks|thank you|what do you think|can you help|let's discuss)\b", weight=0.5)


_register_default_rules()

# Extra weight added when the caller indicates an attachment is present.
# Attachments most commonly are scanned reports (document_qa) or images/
# drawings (vision) in this industrial context, so both get a bonus rather
# than guessing a single type from has_attachment alone.
_ATTACHMENT_BONUS: Dict[str, float] = {
    TASK_DOCUMENT_QA: 1.5,
    TASK_VISION: 1.0,
}
_ATTACHMENT_SIGNAL_LABEL = "has_attachment"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


async def classify(message: str, has_attachment: bool = False) -> TaskClassification:
    """
    Classify a user message into one of the task types known to
    model_registry.py. Deterministic and local-only: no model inference,
    no network calls.

    Returns a TaskClassification with a full score breakdown so callers
    (e.g. router.py) can make informed decisions beyond just the winning
    label -- for example, treating a low-confidence result differently.
    """
    text = (message or "").strip()

    scores: Dict[str, float] = {task_type: 0.0 for task_type in _TASK_RULES}
    matched: Dict[str, List[str]] = {task_type: [] for task_type in _TASK_RULES}

    if text:
        for task_type, rules in _TASK_RULES.items():
            for rule in rules:
                if rule.pattern.search(text):
                    scores[task_type] += rule.weight
                    matched[task_type].append(rule.label)

    if has_attachment:
        for task_type, bonus in _ATTACHMENT_BONUS.items():
            scores.setdefault(task_type, 0.0)
            scores[task_type] += bonus
            matched.setdefault(task_type, []).append(_ATTACHMENT_SIGNAL_LABEL)

    total_score = sum(scores.values())

    if total_score <= 0.0:
        logger.debug("No classification signals matched; defaulting to task_type='%s'.", DEFAULT_TASK_TYPE)
        return TaskClassification(
            task_type=DEFAULT_TASK_TYPE,
            confidence=0.0,
            scores=scores,
            matched_signals=[],
        )

    winning_task_type = max(scores, key=lambda t: (scores[t], t == DEFAULT_TASK_TYPE))
    confidence = scores[winning_task_type] / total_score

    return TaskClassification(
        task_type=winning_task_type,
        confidence=round(confidence, 4),
        scores=scores,
        matched_signals=matched[winning_task_type],
    )


async def classify_task(message: str, has_attachment: bool = False) -> str:
    """
    Backward-compatible entry point matching the exact signature already
    relied on by backend/api/chat.py. Returns only the winning task_type
    string; use classify() directly for the full typed result.
    """
    result = await classify(message, has_attachment=has_attachment)
    return result.task_type