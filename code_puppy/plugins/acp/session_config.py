"""Session model selection + config options exposed to the ACP client.

Everything is delivered as ACP **config options** (``session/set_config_option``)
-- clients (Zed, opencode, oh-my-pi) bind each bottom-bar dropdown to a config
option by its ``category``, so this is the shape that actually populates them:

* **Model picker** — a ``select`` option tagged ``category="model"``:
  ``{id: "model", category: "model", type: "select", ...}``. Backed by
  ``model_picker_completion.load_model_names`` + ``config.get_global_model_name``
  / ``set_model_name``. Changing it rebinds the live session's model.
* **Mode picker** — a ``select`` option tagged ``category="mode"``, one entry
  per available Code Puppy **agent** (a mode *is* an agent; see
  ``session_modes``). Selecting one rebinds the live session to that agent.
  The latest Zed binds its mode dropdown to config options in preference to the
  top-level ``SessionModeState`` ("Config options take precedence over legacy
  mode/model selectors"), so this select is what actually renders there; the
  canonical ``SessionModeState`` is published in parallel for pure-ACP clients.
* **Streaming toggle** — an On/Off ``select`` option. (A boolean option would
  be more natural, but Zed 1.7.2 renders only ``select`` options in its bottom
  bar -- a boolean shows as "Unknown" -- so every control here is a select.) We
  deliberately expose only this safe setting; we do **not** expose a
  yolo/approval-bypass toggle (permissions must stay client-driven).

All accessors are best-effort and degrade to "nothing to offer" on error.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from acp.schema import (
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
)

logger = logging.getLogger(__name__)

# Config-option ids. ``MODEL_OPTION_ID`` is public because ``agent.py`` keys
# its "rebind the live model" behaviour off it.
MODEL_OPTION_ID = "model"
MODE_OPTION_ID = "mode"
_STREAMING_ID = "enable_streaming"
_STREAMING_ON = "on"
_STREAMING_OFF = "off"


def _mode_option(
    current_mode_id: Optional[str] = None,
) -> Optional[SessionConfigOptionSelect]:
    """Build the agent picker as a ``category="mode"`` select option.

    One entry per available agent (see ``session_modes``); ``current_mode_id``
    is the session's currently-bound agent, coerced into the offered set.
    Returns ``None`` when no agents can be enumerated, so a discovery hiccup
    simply omits the control instead of showing a blank dropdown.
    """
    from code_puppy.plugins.acp import session_modes

    modes = session_modes.available_modes()
    if not modes:
        return None
    current = session_modes.resolve_current(current_mode_id) or modes[0][0]
    return SessionConfigOptionSelect(
        id=MODE_OPTION_ID,
        name="Mode",
        category="mode",
        type="select",
        current_value=current,
        options=[
            SessionConfigSelectOption(
                value=mode_id, name=name, description=description or None
            )
            for mode_id, name, description in modes
        ],
    )


def _model_choices() -> Tuple[List[str], Optional[str]]:
    """Return ``(available_model_names, current_selection)`` from config.

    ``current`` is coerced into the available list so a picker always has a
    valid selection. Returns ``([], None)`` when there are no models to offer.
    """
    from code_puppy.command_line.model_picker_completion import load_model_names
    from code_puppy.config import get_global_model_name

    names = list(load_model_names() or [])
    if not names:
        return [], None
    current = get_global_model_name()
    return names, (current if current in names else names[0])


def _model_option() -> Optional[SessionConfigOptionSelect]:
    """Build the model picker as a ``category="model"`` select option.

    Returns ``None`` when there are no models to offer.
    """
    names, current = _model_choices()
    if not names or current is None:
        return None
    return SessionConfigOptionSelect(
        id=MODEL_OPTION_ID,
        name="Model",
        category="model",
        type="select",
        current_value=current,
        options=[SessionConfigSelectOption(value=n, name=n) for n in names],
    )


def set_model(model_id: str) -> bool:
    """Switch the active model. Returns ``True`` on success."""
    try:
        from code_puppy.config import set_model_name

        set_model_name(model_id)
        return True
    except Exception:  # noqa: BLE001
        logger.debug("ACP: set_model failed", exc_info=True)
        return False


def config_options(current_mode_id: Optional[str] = None) -> List[Any]:
    """Build the config-option list for a session (model + mode + streaming).

    Order matters for presentation: the model picker leads, then the agent
    (mode) picker. ``current_mode_id`` is the session's currently-bound agent
    so the mode dropdown reflects reality. Any entry is omitted if it can't be
    built, so a config hiccup never sinks the session.
    """
    opts: List[Any] = []
    try:
        model_opt = _model_option()
        if model_opt is not None:
            opts.append(model_opt)
    except Exception:  # noqa: BLE001
        logger.debug("ACP: could not build model option", exc_info=True)
    try:
        mode_opt = _mode_option(current_mode_id)
        if mode_opt is not None:
            opts.append(mode_opt)
    except Exception:  # noqa: BLE001
        logger.debug("ACP: could not build mode option", exc_info=True)
    try:
        from code_puppy.config import get_enable_streaming

        # A select (On/Off), not a boolean: Zed 1.7.2 only renders select-type
        # config options in its bottom bar; a boolean shows as "Unknown".
        opts.append(
            SessionConfigOptionSelect(
                id=_STREAMING_ID,
                name="Streaming responses",
                type="select",
                current_value=_STREAMING_ON
                if get_enable_streaming()
                else _STREAMING_OFF,
                options=[
                    SessionConfigSelectOption(value=_STREAMING_ON, name="On"),
                    SessionConfigSelectOption(value=_STREAMING_OFF, name="Off"),
                ],
                description="Stream model output token-by-token.",
            )
        )
    except Exception:  # noqa: BLE001
        logger.debug("ACP: could not build streaming option", exc_info=True)
    return opts


def apply_config_option(config_id: str, value: Any) -> List[Any]:
    """Apply a config-option change and return the refreshed option list."""
    try:
        if config_id == MODEL_OPTION_ID:
            set_model(str(value))
        elif config_id == _STREAMING_ID:
            from code_puppy.config import set_config_value

            set_config_value("enable_streaming", "true" if _as_bool(value) else "false")
    except Exception:  # noqa: BLE001
        logger.debug("ACP: apply_config_option failed", exc_info=True)
    return config_options()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")
