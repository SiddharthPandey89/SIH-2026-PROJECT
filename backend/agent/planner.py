"""
backend/agent/planner.py

Planner for the Sovereign AI Workbench.

Responsibilities:
- Convert a user's request into a structured executable plan.
- Use the existing local ModelRouter for planning.
- Keep planning separate from tool execution.
- Produce a deterministic JSON-compatible plan contract for Executor.
- Remain local/offline-first.

The planner DOES NOT execute tools.
The planner DOES NOT directly load models.
The planner DOES NOT contain tool implementations.

Expected output:

{
    "goal": "...",
    "task_type": "...",
    "steps": [
        {
            "id": "step_1",
            "tool": "tool_name",
            "args": {},
            "continue_on_error": false
        }
    ]
}
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from backend.model_router.router import ModelRouter, get_model_router
from backend.model_router.task_classifier import classify

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PlannerError(Exception):
    """Base exception for planner failures."""


class InvalidPlanError(PlannerError):
    """Raised when the model returns an invalid plan."""


class PlanGenerationError(PlannerError):
    """Raised when a plan cannot be generated."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PlanStep:
    """One executable step in an agent plan."""

    id: str
    tool: str
    args: Dict[str, Any]
    continue_on_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "args": self.args,
            "continue_on_error": self.continue_on_error,
        }


@dataclass
class Plan:
    """Structured plan returned by Planner."""

    goal: str
    task_type: str
    steps: List[PlanStep]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "task_type": self.task_type,
            "steps": [step.to_dict() for step in self.steps],
        }


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner:
    """
    Converts a user request into an executable structured plan.

    The planner uses the existing ModelRouter rather than talking to
    Ollama/vLLM/llama.cpp directly.

    This keeps model selection inside model_router/ and planning inside
    agent/.
    """

    def __init__(
        self,
        model_router: Optional[ModelRouter] = None,
    ) -> None:
        self._model_router = model_router or get_model_router()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def create_plan(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
        has_attachment: bool = False,
    ) -> Dict[str, Any]:
        """
        Create an executable plan for a user request.

        Args:
            message:
                User's request.

            history:
                Previous conversation turns.

            context_chunks:
                Retrieved knowledge-base context, when available.

            has_attachment:
                Whether the current request has an uploaded attachment.

        Returns:
            JSON-compatible dictionary containing:
                goal
                task_type
                steps
        """

        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string.")

        # ---------------------------------------------------------------
        # Classify the task using the repository's existing classifier.
        # ---------------------------------------------------------------

        classification = await classify(
            message,
            has_attachment=has_attachment,
        )

        task_type = classification.task_type

        logger.debug(
            "Planner classified request as task_type='%s'.",
            task_type,
        )

        # ---------------------------------------------------------------
        # Build planning prompt.
        # ---------------------------------------------------------------

        planning_prompt = self._build_planning_prompt(
            message=message,
            task_type=task_type,
            history=history,
            context_chunks=context_chunks,
        )

        # ---------------------------------------------------------------
        # Ask existing ModelRouter for the plan.
        #
        # Planning itself is a model-generation task, but the selected
        # model remains controlled by model_router/model_registry.
        # ---------------------------------------------------------------

        try:
            generation = await self._model_router.generate(
                message=planning_prompt,
                task_type=task_type,
                history=None,
                context_chunks=None,
            )
        except Exception as exc:
            logger.exception("Planner model generation failed.")
            raise PlanGenerationError(
                f"Unable to generate agent plan: {exc}"
            ) from exc

        raw_plan = generation.get("answer", "")

        if not isinstance(raw_plan, str) or not raw_plan.strip():
            raise PlanGenerationError(
                "Planner model returned an empty response."
            )

        # ---------------------------------------------------------------
        # Parse and validate model-generated JSON.
        # ---------------------------------------------------------------

        parsed_plan = self._parse_plan(
            raw_plan=raw_plan,
            goal=message,
            task_type=task_type,
        )

        return parsed_plan

    # -----------------------------------------------------------------------
    # Prompt construction
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_planning_prompt(
        message: str,
        task_type: str,
        history: Optional[List[Dict[str, str]]],
        context_chunks: Optional[List[Dict[str, Any]]],
    ) -> str:
        """
        Build a strict planning prompt.

        The model is explicitly instructed to return JSON only.
        """

        history_block = ""

        if history:
            history_lines: List[str] = []

            for turn in history[-10:]:
                if not isinstance(turn, Mapping):
                    continue

                role = str(turn.get("role", "user"))
                content = str(turn.get("content", ""))

                if content:
                    history_lines.append(
                        f"{role}: {content}"
                    )

            if history_lines:
                history_block = (
                    "\nPrevious conversation:\n"
                    + "\n".join(history_lines)
                    + "\n"
                )

        context_block = ""

        if context_chunks:
            context_lines: List[str] = []

            for chunk in context_chunks[:10]:
                if not isinstance(chunk, Mapping):
                    continue

                title = str(chunk.get("title", "Untitled"))
                snippet = str(chunk.get("snippet", ""))

                if snippet:
                    context_lines.append(
                        f"[{title}] {snippet}"
                    )

            if context_lines:
                context_block = (
                    "\nRetrieved knowledge-base context:\n"
                    + "\n".join(context_lines)
                    + "\n"
                )

        return f"""
You are the planning component of an offline/on-premise
Sovereign AI Workbench.

Your job is ONLY to convert the user's request into an executable
structured plan.

Do NOT execute any tool.
Do NOT invent tool results.
Do NOT answer the user's request directly.
Do NOT include markdown.
Return JSON only.

Detected task type:
{task_type}

User request:
{message}
{history_block}
{context_block}

The executor understands this exact structure:

{{
  "goal": "short description of the user's goal",
  "steps": [
    {{
      "id": "step_1",
      "tool": "tool_name",
      "args": {{}},
      "continue_on_error": false
    }}
  ]
}}

Rules:

1. Every step must have a unique id.
2. Step IDs must be step_1, step_2, step_3, etc.
3. Each step must contain:
   - id
   - tool
   - args
4. args must always be a JSON object.
5. Do not execute tools.
6. Do not fabricate tool outputs.
7. Keep steps minimal and logically ordered.
8. A simple request may contain one step.
9. A complex request may contain multiple dependent steps.
10. If a later step needs an earlier result, reference it as "$step_N".
11. Do not create imaginary tool names unless the requested operation clearly
    requires a capability that is expected to be implemented as a backend tool.
12. Never include explanations outside the JSON object.
13. Do not use markdown code fences.
14. Do not include comments inside JSON.

Return JSON only.
""".strip()

    # -----------------------------------------------------------------------
    # JSON parsing
    # -----------------------------------------------------------------------

    @classmethod
    def _parse_plan(
        cls,
        raw_plan: str,
        goal: str,
        task_type: str,
    ) -> Dict[str, Any]:
        """
        Parse model output into the executor-compatible plan format.

        Handles:
        - pure JSON
        - accidental markdown code fences
        - surrounding whitespace
        """

        cleaned = raw_plan.strip()

        # Remove accidental markdown fences.
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        cleaned = cleaned.strip()

        # First try direct JSON parsing.
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try extracting the outermost JSON object.
            parsed = cls._extract_json_object(cleaned)

        if not isinstance(parsed, dict):
            raise InvalidPlanError(
                "Planner response must be a JSON object."
            )

        # ---------------------------------------------------------------
        # Normalize goal.
        # ---------------------------------------------------------------

        parsed_goal = parsed.get("goal")

        if not isinstance(parsed_goal, str) or not parsed_goal.strip():
            parsed_goal = goal

        # ---------------------------------------------------------------
        # Normalize task type.
        # ---------------------------------------------------------------

        parsed_task_type = parsed.get("task_type")

        if not isinstance(parsed_task_type, str) or not parsed_task_type.strip():
            parsed_task_type = task_type

        # ---------------------------------------------------------------
        # Validate steps.
        # ---------------------------------------------------------------

        raw_steps = parsed.get("steps")

        if not isinstance(raw_steps, list):
            raise InvalidPlanError(
                "Planner response must contain a 'steps' list."
            )

        steps: List[PlanStep] = []
        seen_ids = set()

        for index, raw_step in enumerate(raw_steps, start=1):

            if not isinstance(raw_step, Mapping):
                raise InvalidPlanError(
                    f"Step {index} must be a JSON object."
                )

            step_id = raw_step.get("id", f"step_{index}")

            if not isinstance(step_id, str) or not step_id.strip():
                raise InvalidPlanError(
                    f"Step {index} has an invalid id."
                )

            step_id = step_id.strip()

            if step_id in seen_ids:
                raise InvalidPlanError(
                    f"Duplicate step id: '{step_id}'."
                )

            seen_ids.add(step_id)

            tool = raw_step.get("tool")

            if not isinstance(tool, str) or not tool.strip():
                raise InvalidPlanError(
                    f"Step '{step_id}' must contain a tool."
                )

            tool = tool.strip()

            args = raw_step.get("args", {})

            if not isinstance(args, Mapping):
                raise InvalidPlanError(
                    f"Step '{step_id}' args must be a JSON object."
                )

            continue_on_error = raw_step.get(
                "continue_on_error",
                False,
            )

            if not isinstance(continue_on_error, bool):
                continue_on_error = bool(continue_on_error)

            steps.append(
                PlanStep(
                    id=step_id,
                    tool=tool,
                    args=dict(args),
                    continue_on_error=continue_on_error,
                )
            )

        return Plan(
            goal=parsed_goal.strip(),
            task_type=parsed_task_type.strip(),
            steps=steps,
        ).to_dict()

    # -----------------------------------------------------------------------
    # JSON extraction fallback
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any]:
        """
        Extract the first balanced JSON object from model output.

        This handles cases where a local model accidentally adds a small
        amount of text before or after the JSON.
        """

        start = text.find("{")

        if start == -1:
            raise InvalidPlanError(
                "No JSON object found in planner response."
            )

        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            char = text[index]

            if escaped:
                escaped = False
                continue

            if char == "\\" and in_string:
                escaped = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1

            elif char == "}":
                depth -= 1

                if depth == 0:
                    candidate = text[start : index + 1]

                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError as exc:
                        raise InvalidPlanError(
                            "Planner returned malformed JSON."
                        ) from exc

                    if not isinstance(parsed, dict):
                        raise InvalidPlanError(
                            "Planner JSON root must be an object."
                        )

                    return parsed

        raise InvalidPlanError(
            "Planner returned an incomplete JSON object."
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


_default_planner: Optional[Planner] = None


def get_planner() -> Planner:
    """
    Return the shared Planner instance.

    Kept zero-argument so it can later be safely used as a FastAPI
    dependency in routes_agent.py.
    """

    global _default_planner

    if _default_planner is None:
        _default_planner = Planner(
            model_router=get_model_router()
        )

    return _default_planner


__all__ = [
    "Planner",
    "Plan",
    "PlanStep",
    "PlannerError",
    "InvalidPlanError",
    "PlanGenerationError",
    "get_planner",
]