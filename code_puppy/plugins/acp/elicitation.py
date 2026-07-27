"""Forward Code Puppy's ``ask_user_question`` tool to native ACP elicitation.

``ask_user_question`` renders an interactive terminal picker and reads stdin --
which, over ACP, *is* the JSON-RPC pipe, so it can't run. The bridge historically
blocked it and steered the model to ask in plain text.

ACP 0.11 added ``connection.create_elicitation(message, mode)``, and Zed turned
it on for ACP agents in **v1.12.0** ("Enabled ACP elicitations by default,
allowing ACP agents to collect structured user input", zed PR #60749). A client
that advertises ``elicitation.form`` can render a structured form and return the
user's choices, so this module is the two-way mapping:

* **out** — Code Puppy's ``questions`` payload -> an ``ElicitationSchema``
  (one property per question: a single-select becomes a string with an
  ``enum``; a multi-select becomes an ``array`` of enum items).
* **back** — the client's accept/decline/cancel response -> the
  ``AskUserQuestionOutput`` shape the tool already returns, so the model sees
  exactly what it would have from the terminal picker.

Everything is best-effort: if the client lacks elicitation, the payload is
malformed, or the call fails, ``ask`` returns ``None`` and the caller falls back
to the plain-text block. We never crash a turn over a question.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from code_puppy.plugins.acp import state

logger = logging.getLogger(__name__)


def supported() -> bool:
    """Whether this connection's client can render form elicitations."""
    return state.client_supports_form_elicitation()


def _as_dicts(questions: Any) -> List[Dict[str, Any]]:
    """Normalise the tool's ``questions`` arg into a list of plain dicts.

    The arg may arrive as raw JSON dicts (tool-call args) or as validated
    pydantic models; both are coerced so schema-building has one shape.
    """
    if questions is None:
        return []
    items = questions
    if isinstance(questions, dict):
        items = questions.get("questions", questions)
    out: List[Dict[str, Any]] = []
    for q in items or []:
        if hasattr(q, "model_dump"):
            out.append(q.model_dump())
        elif isinstance(q, dict):
            out.append(q)
    return out


def _option_labels(question: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    for opt in question.get("options") or []:
        if isinstance(opt, dict):
            label = opt.get("label")
        else:
            label = getattr(opt, "label", None)
        if label:
            labels.append(str(label))
    return labels


def _build_schema(questions: List[Dict[str, Any]]) -> Optional[Any]:
    """Build an ``ElicitationSchema`` from the tool's questions, or ``None``.

    Returns ``None`` when no question carries options, so a degenerate payload
    falls back to the plain-text block rather than a useless empty form.
    """
    from acp.schema import (
        ElicitationMultiSelectPropertySchema,
        ElicitationSchema,
        ElicitationStringPropertySchema,
        EnumOption,
        TitledMultiSelectItems,
    )

    properties: Dict[str, Any] = {}
    required: List[str] = []
    for question in questions:
        header = str(question.get("header") or "").strip()
        labels = _option_labels(question)
        if not header or not labels:
            continue
        title = str(question.get("question") or header)
        if question.get("multi_select"):
            properties[header] = ElicitationMultiSelectPropertySchema(
                type="array",
                title=title,
                items=TitledMultiSelectItems(
                    any_of=[EnumOption(const=label, title=label) for label in labels]
                ),
            )
        else:
            properties[header] = ElicitationStringPropertySchema(
                type="string",
                title=title,
                one_of=[EnumOption(const=label, title=label) for label in labels],
            )
        required.append(header)
    if not properties:
        return None
    return ElicitationSchema(
        type="object",
        title="Code Puppy needs your input",
        properties=properties,
        required=required,
    )


def _content_to_output(content: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map an accepted elicitation's ``content`` to ``AskUserQuestionOutput``.

    Each property value (a label string, or a list of labels for multi-select)
    becomes a ``QuestionAnswer`` keyed by the question header.
    """
    answers: List[Dict[str, Any]] = []
    for header, value in (content or {}).items():
        if isinstance(value, list):
            selected = [str(v) for v in value]
        elif value is None:
            selected = []
        else:
            selected = [str(value)]
        answers.append({"question_header": header, "selected_options": selected})
    return {"answers": answers, "cancelled": False, "error": None, "timed_out": False}


async def ask(questions: Any) -> Optional[Dict[str, Any]]:
    """Ask ``questions`` via client elicitation; return an output dict or ``None``.

    ``None`` means "couldn't elicit" (unsupported client, no connection, empty
    schema, or a failed call) so the caller can fall back to the plain-text
    block. A user *cancel/decline* is a real answer -- it returns an output dict
    with ``cancelled=True``, not ``None``.
    """
    if not supported():
        return None
    connection = state.get_connection()
    session_id = state.get_active_session_id()
    if connection is None or session_id is None:
        return None
    parsed = _as_dicts(questions)
    schema = _build_schema(parsed)
    if schema is None:
        return None

    from acp.schema import ElicitationFormSessionMode

    message = (
        str(parsed[0].get("question")) if parsed else ""
    ) or "Code Puppy has a question for you."
    mode = ElicitationFormSessionMode(
        session_id=session_id,
        tool_call_id=f"tc_{uuid.uuid4().hex[:12]}",
        requested_schema=schema,
    )
    try:
        response = await connection.create_elicitation(message, mode)
    except Exception:  # noqa: BLE001
        logger.debug("ACP: create_elicitation failed", exc_info=True)
        return None
    action = getattr(response, "action", None)
    if action == "accept":
        return _content_to_output(getattr(response, "content", None) or {})
    # decline / cancel -> a genuine "user opted out" outcome.
    return {"answers": [], "cancelled": True, "error": None, "timed_out": False}
