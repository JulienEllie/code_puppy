"""``CodePuppyAgent`` — Code Puppy as a native Agent Client Protocol agent.

This implements the official SDK's ``Agent`` interface, so Code Puppy runs as
an external agent in any ACP client (Zed, and other editors that speak ACP).
The SDK owns the wire: ``acp.run_agent`` binds stdio, frames JSON-RPC, parses
params into typed models, and hands us a ``Client`` connection (via
``on_connect``) for talking back to the client. We own the behaviour: mapping ACP sessions to Code Puppy agents,
running prompts, and translating events (through ``EventBridge``) and I/O
(through ``permissions`` + ``io_delegation``).

Only capability-gated methods we actually support are implemented; the SDK's
router resolves unimplemented ones to a clean "method not found", so we never
have to stub protocol surface we don't mean.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional

from acp import Agent
from acp.schema import (
    AgentCapabilities,
    AuthenticateResponse,
    AvailableCommand,
    AvailableCommandsUpdate,
    ClientCapabilities,
    CloseSessionResponse,
    CurrentModeUpdate,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    SessionInfo,
    SetSessionModeResponse,
)

from code_puppy.plugins.acp import (
    capabilities,
    io_delegation,
    mcp_config,
    permissions,
    persistence,
    replay,
    session_config,
    session_modes,
    state,
)
from code_puppy.plugins.acp.bridge import EventBridge
from code_puppy.plugins.acp.session import ACPSession

logger = logging.getLogger(__name__)

# The ACP protocol version this plugin targets. Source it from the SDK rather
# than hardcoding a literal, so a library upgrade can't leave us silently
# advertising a stale version. Falls back to v1 (the stable baseline) if an
# older SDK doesn't export the constant.
try:
    from acp import PROTOCOL_VERSION as PROTOCOL_VERSION
except ImportError:  # pragma: no cover - defensive for older SDKs
    PROTOCOL_VERSION = 1


def _code_puppy_version() -> str:
    try:
        return version("code-puppy")
    except PackageNotFoundError:
        return "0.0.0"


def _current_agent_name() -> str:
    """The globally-selected agent name, or a safe default on error."""
    try:
        from code_puppy.agents.agent_manager import get_current_agent_name

        return get_current_agent_name()
    except Exception:  # noqa: BLE001
        return "code-puppy"


class CodePuppyAgent(Agent):
    """One ACP connection's worth of Code Puppy, spread across client threads."""

    def __init__(self) -> None:
        self._sessions: Dict[str, ACPSession] = {}
        self._bridge = EventBridge()
        self._client_caps: Optional[ClientCapabilities] = None

    # ---- Connection lifecycle ---------------------------------------------
    def on_connect(self, conn: Any) -> None:
        """Store the client handle + loop, wire events, install approvals.

        Called synchronously by the SDK from inside ``run_agent`` — already on
        the running event loop — so ``get_running_loop`` returns the loop our
        permission and I/O bridges marshal onto.
        """
        state.set_connection(conn, asyncio.get_running_loop())
        self._bridge.register()
        permissions.install()

    def shutdown(self) -> None:
        """Unwire this agent's event hooks. Call once the connection ends.

        ``permissions`` and ``io_delegation`` are torn down by the plugin entry
        point; this drops the ``EventBridge`` hooks it owns so a closed
        connection leaves the global callback registry clean.
        """
        self._bridge.unregister()

    # ---- Handshake --------------------------------------------------------
    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Optional[ClientCapabilities] = None,
        client_info: Optional[Implementation] = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        """Negotiate capabilities and install matching I/O delegation.

        Once we know what the client can do, reroute Code Puppy's workspace
        file I/O and shell to the client's ``fs/*`` / ``terminal/*`` methods
        (only the edges
        the client advertises). Everything else stays local.
        """
        self._client_caps = client_capabilities
        io_delegation.install(client_capabilities)
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=self._agent_capabilities(),
            agent_info=Implementation(
                name="code-puppy",
                title="Code Puppy",
                version=_code_puppy_version(),
            ),
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse:
        """No auth flow — Code Puppy authenticates via its own model config."""
        return AuthenticateResponse()

    # ---- Session lifecycle ------------------------------------------------
    async def new_session(
        self,
        cwd: str,
        additional_directories: Optional[List[str]] = None,
        mcp_servers: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        """Create a session bound to a fresh agent instance.

        Uses ``load_agent`` (not the cached ``get_current_agent``) so each
        client thread gets its own ``_message_history``. ``cwd`` +
        ``additional_directories`` anchor the session's tools; ``mcp_servers``
        the client injects are attached to the agent.
        """
        session_id = f"sess_{uuid.uuid4().hex[:16]}"
        session = self._make_session(
            session_id, cwd, additional_directories, mcp_servers
        )
        self._announce_commands_soon(session_id)
        return NewSessionResponse(
            session_id=session_id,
            modes=session_modes.mode_state(session.agent_name),
            config_options=session_config.config_options(session.agent_name) or None,
        )

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: Optional[List[str]] = None,
        mcp_servers: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        """Re-open a thread the client remembers, replaying persisted history.

        If we have the session's history on disk (persisted after each turn),
        it is rehydrated into the fresh agent so the model continues the real
        conversation, AND replayed to the client as ``session/update``
        notifications so the client rebuilds the thread UI (without this the
        client shows -- and may discard -- an empty thread). Otherwise the
        thread is re-created empty but functional.
        """
        session = self._make_session(
            session_id, cwd, additional_directories, mcp_servers, rehydrate=True
        )
        await replay.replay_history(session_id, session.agent.get_message_history())
        self._announce_commands_soon(session_id)
        return LoadSessionResponse(
            modes=session_modes.mode_state(session.agent_name),
            config_options=session_config.config_options(session.agent_name) or None,
        )

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: Optional[List[str]] = None,
        mcp_servers: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Resume a session across a restart, rehydrating + replaying history."""
        from acp.schema import ResumeSessionResponse

        session = self._make_session(
            session_id, cwd, additional_directories, mcp_servers, rehydrate=True
        )
        await replay.replay_history(session_id, session.agent.get_message_history())
        self._announce_commands_soon(session_id)
        return ResumeSessionResponse(
            modes=session_modes.mode_state(session.agent_name),
            config_options=session_config.config_options(session.agent_name) or None,
        )

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: Optional[List[str]] = None,
        mcp_servers: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Branch a session: duplicate its history into a new session id.

        The source's history is copied into a fresh agent so the fork continues
        the conversation independently. The source may be a live in-memory
        session OR one persisted by a prior process -- forking must survive a
        restart just like ``load``/``resume`` do, so we fall back to the
        pickled history when the source isn't live.
        """
        from acp.schema import ForkSessionResponse

        source = self._sessions.get(session_id)
        if source is not None:
            source_history = list(source.agent.get_message_history())
            source_cwd = source.cwd
            source_mode = source.agent_name
        else:
            source_history = persistence.load_history(session_id)
            if source_history is None:
                raise ValueError(f"unknown session to fork: {session_id}")
            source_cwd = None
            source_mode = None
        new_id = f"sess_{uuid.uuid4().hex[:16]}"
        session = self._make_session(
            new_id,
            cwd or source_cwd or "",
            additional_directories,
            mcp_servers,
        )
        try:
            session.agent.set_message_history(source_history)
        except Exception:  # noqa: BLE001
            logger.debug("ACP: fork history copy failed", exc_info=True)
        # Preserve the source's mode (agent) so a fork continues in the same
        # agent it branched from, not whatever the global default happens to be.
        if source_mode and source_mode != session.agent_name:
            self._rebind_session_agent(new_id, agent_name=source_mode)
        self._announce_commands_soon(new_id)
        return ForkSessionResponse(
            session_id=new_id,
            modes=session_modes.mode_state(session.agent_name),
            config_options=session_config.config_options(session.agent_name) or None,
        )

    async def set_session_mode(
        self, mode_id: str, session_id: str, **kwargs: Any
    ) -> SetSessionModeResponse:
        """Switch a session's mode by rebinding it to the named agent.

        A Code Puppy *mode* is an *agent* (see ``session_modes``), so
        ``session/set_mode`` rebinds the live session to ``mode_id``'s agent,
        preserving its message history and client-injected MCP servers so the
        switch is invisible to the running conversation. An unknown mode id or a
        no-change request is a safe no-op. On a real change we push a
        ``current_mode_update`` and re-announce commands (a different agent may
        expose different slash commands).
        """
        session = self._sessions.get(session_id)
        if session is None or not session_modes.is_known_mode(mode_id):
            return SetSessionModeResponse()
        if mode_id == session.agent_name:
            return SetSessionModeResponse()
        self._rebind_session_agent(session_id, agent_name=mode_id)
        await self._announce_mode(session_id, session.agent_name or mode_id)
        self._announce_commands_soon(session_id)
        return SetSessionModeResponse()

    def _rebind_session_agent(
        self, session_id: str, agent_name: Optional[str] = None
    ) -> None:
        """Rebuild a live session's agent, keeping its state.

        Used for both a mode switch (``agent_name`` given -> bind that agent)
        and a model change (``agent_name`` omitted -> rebuild the session's
        *current* agent on the freshly-selected model). Message history and any
        client-injected MCP servers are preserved, so the switch is invisible to
        the conversation. Best-effort: a rebind failure leaves the existing
        agent in place.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return
        try:
            history = list(session.agent.get_message_history())
            name = agent_name or session.agent_name
            session.agent = self._new_agent(name)
            session.agent_name = name
            session.agent.set_message_history(history)
            if session.mcp_specs:
                mcp_config.attach(session.agent, session.mcp_specs)
        except Exception:  # noqa: BLE001
            logger.debug("ACP: agent rebind failed", exc_info=True)

    async def _announce_mode(self, session_id: str, mode_id: str) -> None:
        """Push a ``current_mode_update`` so the client reflects the new mode."""
        connection = state.get_connection()
        if connection is None:
            return
        try:
            await connection.session_update(
                session_id,
                CurrentModeUpdate(
                    session_update="current_mode_update", current_mode_id=mode_id
                ),
            )
        except Exception:  # noqa: BLE001
            logger.debug("ACP: current_mode_update failed", exc_info=True)

    async def set_config_option(
        self, config_id: str, session_id: str, value: Any, **kwargs: Any
    ) -> Any:
        """Apply a config-option change and return the refreshed options.

        * ``model`` rebinds the live session's agent to the newly-selected model
          (history + client MCP servers preserved), so the client's model picker
          switches the model mid-thread.
        * ``mode`` rebinds the session to a different agent (see
          ``set_session_mode``) -- the same behaviour, reached via the config
          option the latest Zed renders in preference to ``SessionModeState``.
        """
        from acp.schema import SetSessionConfigOptionResponse

        if config_id == session_config.MODE_OPTION_ID:
            session = self._sessions.get(session_id)
            if (
                session is not None
                and session_modes.is_known_mode(str(value))
                and str(value) != session.agent_name
            ):
                self._rebind_session_agent(session_id, agent_name=str(value))
                self._announce_commands_soon(session_id)
        else:
            session_config.apply_config_option(config_id, value)
            if config_id == session_config.MODEL_OPTION_ID:
                self._rebind_session_agent(session_id)
        session = self._sessions.get(session_id)
        current_mode = session.agent_name if session is not None else None
        return SetSessionConfigOptionResponse(
            config_options=session_config.config_options(current_mode) or None
        )

    async def list_sessions(
        self, cursor: Optional[str] = None, cwd: Optional[str] = None, **kwargs: Any
    ) -> ListSessionsResponse:
        """List every revivable session: live in-memory + persisted on disk.

        Live sessions (this process) merge with sessions persisted to disk in
        prior runs, so a client's picker still shows -- and can
        ``session/load`` / ``session/resume`` -- threads that outlived the
        process that made them. Live wins on id collision (it carries the
        freshest state). Uncursored single page; optional ``cwd`` filter.
        """
        infos: List[SessionInfo] = []
        seen: set[str] = set()
        for s in self._sessions.values():
            if cwd is not None and s.cwd != cwd:
                continue
            seen.add(s.session_id)
            infos.append(
                SessionInfo(
                    session_id=s.session_id,
                    cwd=s.cwd or cwd or "",
                    additional_directories=s.additional_directories or None,
                )
            )
        for record in persistence.list_persisted():
            if record.session_id in seen:
                continue
            if cwd is not None and record.cwd != cwd:
                continue
            seen.add(record.session_id)
            infos.append(
                SessionInfo(
                    session_id=record.session_id,
                    cwd=record.cwd or cwd or "",
                    additional_directories=record.additional_directories or None,
                )
            )
        return ListSessionsResponse(sessions=infos, next_cursor=None)

    async def close_session(
        self, session_id: str, **kwargs: Any
    ) -> CloseSessionResponse:
        """Drop a session, cancelling any in-flight run first.

        A close is a deliberate act, so its persisted history is deleted too --
        otherwise the session would resurrect in the next ``list_sessions``,
        which is exactly the "disappeared session that won't stay gone" bug in
        reverse.
        """
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.cancel()
        persistence.delete(session_id)
        return CloseSessionResponse()

    # ---- Prompt turn ------------------------------------------------------
    async def prompt(
        self, prompt: List[Any], session_id: str, **kwargs: Any
    ) -> PromptResponse:
        """Run the agent on a user prompt, streaming updates via the bridge."""
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"unknown session: {session_id}")
        outcome = await session.prompt(prompt)
        return PromptResponse(stop_reason=outcome.stop_reason, usage=outcome.usage)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """Cancel the in-flight run for a session (notification)."""
        session = self._sessions.get(session_id)
        if session is not None:
            session.cancel()

    # ---- Helpers ----------------------------------------------------------
    def _agent_capabilities(self) -> AgentCapabilities:
        return capabilities.agent_capabilities()

    def _make_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: Optional[List[str]],
        mcp_servers: Optional[List[Any]],
        *,
        rehydrate: bool = False,
    ) -> ACPSession:
        """Build + register an ``ACPSession`` with a fresh agent.

        When ``rehydrate`` is set, any persisted history for ``session_id`` is
        replayed into the agent so the model continues the real conversation.
        Client-injected ``mcp_servers`` are attached best-effort.
        """
        agent = self._new_agent()
        agent_name = getattr(agent, "name", None) or _current_agent_name()
        if rehydrate:
            history = persistence.load_history(session_id)
            if history:
                try:
                    agent.set_message_history(history)
                except Exception:  # noqa: BLE001
                    logger.debug("ACP: history rehydrate failed", exc_info=True)
        if mcp_servers:
            mcp_config.attach(agent, mcp_servers)
        session = ACPSession(
            session_id,
            agent,
            cwd=cwd,
            additional_directories=additional_directories,
            mcp_specs=mcp_servers,
            agent_name=agent_name,
        )
        self._sessions[session_id] = session
        return session

    @staticmethod
    def _new_agent(agent_name: Optional[str] = None) -> Any:
        from code_puppy.agents.agent_manager import load_agent

        return load_agent(agent_name or _current_agent_name())

    def _announce_commands_soon(self, session_id: str) -> None:
        """Schedule an ``available_commands_update`` after the response ships.

        We must not send the update before the ``new_session``/``load_session``
        reply reaches the client (it wouldn't know the session yet).
        Scheduling a
        task defers it until this coroutine yields, i.e. after the reply.
        """
        loop = state.get_loop()
        if loop is not None:
            loop.create_task(self._announce_commands(session_id))

    async def _announce_commands(self, session_id: str) -> None:
        """Tell the client which slash commands Code Puppy exposes.

        Deferred behind a short delay: ``create_task`` only yields one tick,
        which isn't enough to guarantee the ``new_session`` reply has been
        serialized onto the wire. Without this the client can receive the
        notification before it knows the session id and drop it as "unknown
        session". 100ms is imperceptible and comfortably after the reply.
        """
        await asyncio.sleep(0.1)
        connection = state.get_connection()
        if connection is None:
            return
        try:
            from code_puppy.command_line.command_registry import get_unique_commands

            available = [
                AvailableCommand(name=c.name, description=c.description)
                for c in get_unique_commands()
            ]
        except Exception:  # noqa: BLE001
            logger.debug("ACP: could not enumerate slash commands", exc_info=True)
            return
        if not available:
            return
        update = AvailableCommandsUpdate(
            session_update="available_commands_update",
            available_commands=available,
        )
        try:
            await connection.session_update(session_id, update)
        except Exception:  # noqa: BLE001
            logger.debug("ACP: available_commands_update failed", exc_info=True)
