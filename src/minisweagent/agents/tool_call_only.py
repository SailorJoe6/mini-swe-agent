"""Agent that forces tool calls and requires reasoning in bash tool arguments."""

from minisweagent.agents.default import DefaultAgent


class ToolCallOnlyAgent(DefaultAgent):
    """Force tool calling and require reasoning in tool calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configure_tool_calling()

    def _configure_tool_calling(self) -> None:
        config = getattr(self.model, "config", None)
        if config is None:
            return
        if hasattr(config, "tool_choice"):
            config.tool_choice = "required"
        if hasattr(config, "require_reasoning"):
            config.require_reasoning = True
