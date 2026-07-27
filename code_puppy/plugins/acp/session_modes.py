"""ACP session modes, mapped onto Code Puppy's agent catalogue.

An ACP *session mode* answers "how should the agent behave for this thread?".
Code Puppy already expresses behaviour profiles as **agents** (the default
``code-puppy``, plus any pack/JSON/clone agents), so the honest mapping is
one-to-one: **a mode is an agent**, and switching a mode rebinds the session to
that agent (its message history + client-injected MCP servers are preserved,
so the switch is invisible to the running conversation).

Two client-facing surfaces are built from this ONE source of truth, because
different clients read different mechanisms:

* **``category="mode"`` config option** — the latest Zed binds its bottom-bar
  mode dropdown to whichever mechanism the agent exposes, with a strict
  precedence: *if the agent sends any config options, Zed renders those and
  ignores the top-level ``SessionModeState``* (see zed
  ``crates/agent_ui/src/conversation_view.rs`` — "Config options take
  precedence over legacy mode/model selectors"). Code Puppy sends config
  options, so this select is what actually renders in Zed.
* **``SessionModeState``** — the canonical ACP field on the session responses.
  We still publish it (from the same list) so a pure-ACP client that sends *no*
  config options still gets a working mode picker.

Everything here is best-effort: any failure to enumerate agents degrades to
"no modes to offer" rather than sinking a session.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# One mode row: (id, display_name, description). ``id`` is the agent's stable
# name -- exactly what ``load_agent`` accepts and what a mode id round-trips as.
Mode = Tuple[str, str, str]


def available_modes() -> List[Mode]:
    """Return the selectable modes, one per available Code Puppy agent.

    Best-effort: any discovery failure yields an empty list so mode support
    silently no-ops instead of raising into a session response.
    """
    try:
        from code_puppy.agents.agent_manager import (
            get_agent_descriptions,
            get_available_agents,
        )

        names = get_available_agents() or {}
        descriptions = get_agent_descriptions() or {}
    except Exception:  # noqa: BLE001
        logger.debug("ACP: could not enumerate agent modes", exc_info=True)
        return []
    return [
        (agent_id, display or agent_id, descriptions.get(agent_id, "") or "")
        for agent_id, display in names.items()
    ]


def is_known_mode(mode_id: str) -> bool:
    """Whether ``mode_id`` names a currently-available agent-mode."""
    return any(m[0] == mode_id for m in available_modes())


def resolve_current(current_mode_id: Optional[str]) -> Optional[str]:
    """Coerce ``current_mode_id`` into a valid mode id, or ``None`` if no modes.

    A caller's remembered mode that no longer exists (agent removed) falls back
    to the first offered mode, so a picker always has a valid selection.
    """
    modes = available_modes()
    if not modes:
        return None
    if current_mode_id and any(m[0] == current_mode_id for m in modes):
        return current_mode_id
    return modes[0][0]


def mode_state(current_mode_id: Optional[str]) -> Optional[Any]:
    """Build the SDK ``SessionModeState`` for a session's current agent.

    Returns ``None`` when there are no agents to offer, so the ``modes`` field
    is simply omitted from the session response.
    """
    from acp.schema import SessionMode, SessionModeState

    modes = available_modes()
    if not modes:
        return None
    current = resolve_current(current_mode_id)
    return SessionModeState(
        current_mode_id=current or modes[0][0],
        available_modes=[
            SessionMode(id=mode_id, name=name, description=description or None)
            for mode_id, name, description in modes
        ],
    )
