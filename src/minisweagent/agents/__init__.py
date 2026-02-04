"""Agent implementations for mini-SWE-agent."""

import importlib

from typing import Type

_AGENT_CLASS_MAPPING = {
    "default": "minisweagent.agents.default.DefaultAgent",
    "interactive": "minisweagent.agents.interactive.InteractiveAgent",
    "toolcall_only": "minisweagent.agents.tool_call_only.ToolCallOnlyAgent",
}


def resolve_agent_class(agent_class: str | type | None, *, default: type) -> Type:
    """Resolve an agent class from config or return the default."""
    if agent_class is None:
        return default
    if isinstance(agent_class, type):
        return agent_class
    full_path = _AGENT_CLASS_MAPPING.get(agent_class, agent_class)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as exc:
        msg = f"Unknown agent class: {agent_class} (resolved to {full_path})"
        raise ValueError(msg) from exc


__all__ = ["resolve_agent_class"]
