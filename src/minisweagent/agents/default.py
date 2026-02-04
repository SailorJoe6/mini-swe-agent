"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import json
import logging
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded
from minisweagent.models.context_window import (
    load_context_window_map,
    lookup_context_window,
    normalize_model_name,
    update_context_window_map,
)
from minisweagent.utils.serialize import recursive_merge, to_jsonable


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    output_path: Path | None = None
    """Save the trajectory to this path."""


class DefaultAgent:
    context_window_mode = "auto"

    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self._live_trajectory_path: Path | None = None
        self.context_window_max: int | None = None
        self.context_window_prompt_tokens: int | None = None
        self.context_left_percent: int | None = None

    def set_live_trajectory_path(self, path: Path | None) -> None:
        """Set a live JSONL trajectory path and clear any existing file."""
        self._live_trajectory_path = path
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.unlink(missing_ok=True)
        except Exception as exc:
            self.logger.warning("Failed to initialize live trajectory file %s: %s", path, exc)

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "context_window_max": self.context_window_max,
                "context_window_prompt_tokens": self.context_window_prompt_tokens,
                "context_left_percent": self.context_left_percent,
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        if self._live_trajectory_path:
            try:
                self._live_trajectory_path.parent.mkdir(parents=True, exist_ok=True)
                with self._live_trajectory_path.open("a", encoding="utf-8") as handle:
                    for message in messages:
                        handle.write(json.dumps(to_jsonable(message)) + "\n")
            except Exception as exc:
                self.logger.warning(
                    "Failed to write live trajectory to %s: %s", self._live_trajectory_path, exc
                )
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self._resolve_context_window_max()
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        return self.execute_actions(self.query())

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if self.context_window_max is None:
            self._resolve_context_window_max()
        self.n_calls += 1
        message = self.model.query(self.messages)
        self._update_context_window_stats(message)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        outputs = [self.env.execute(action) for action in message.get("extra", {}).get("actions", [])]
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def _resolve_context_window_max(self) -> None:
        if self.context_window_max is not None:
            return
        model_name = getattr(self.model, "config", None)
        model_name = getattr(model_name, "model_name", None)
        if not model_name or not isinstance(model_name, str):
            return
        model_name = model_name.strip()
        if not model_name:
            return
        context_map = load_context_window_map()
        resolved = lookup_context_window(model_name, context_map)
        normalized_name = normalize_model_name(model_name)
        normalized_keys = {normalize_model_name(key) for key in context_map}
        if resolved is None and self.context_window_mode == "interactive":
            resolved = self._prompt_for_context_window(model_name)
            if resolved is not None:
                update_context_window_map(model_name, resolved)
        if resolved is not None:
            self.context_window_max = int(resolved)
            if normalized_name not in normalized_keys:
                update_context_window_map(model_name, resolved)

    def _prompt_for_context_window(self, model_name: str) -> int | None:
        return None

    def _update_context_window_stats(self, message: dict) -> None:
        prompt_tokens = self._extract_prompt_tokens(message)
        if prompt_tokens is None:
            return
        self.context_window_prompt_tokens = prompt_tokens
        if not self.context_window_max:
            return
        left = int(100 * (1 - (prompt_tokens / self.context_window_max)))
        self.context_left_percent = max(0, min(100, left))
        message["context_left_percent"] = self.context_left_percent

    @staticmethod
    def _extract_prompt_tokens(message: dict) -> int | None:
        usage = None
        extra = message.get("extra") or {}
        response = extra.get("response")
        if response and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None and isinstance(message.get("usage"), dict):
            usage = message.get("usage")
        if usage is None and hasattr(message.get("usage"), "model_dump"):
            usage = message["usage"].model_dump()
        if usage is None and hasattr(message.get("usage"), "__dict__"):
            usage = message["usage"].__dict__
        if not isinstance(usage, dict):
            return None
        prompt_tokens = usage.get("prompt_tokens")
        return prompt_tokens if isinstance(prompt_tokens, int) else None

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(to_jsonable(data), indent=2))
        return data
