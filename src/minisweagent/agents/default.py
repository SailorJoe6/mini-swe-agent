"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import json
import re
import subprocess
import time
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model
from minisweagent.models.context_window import (
    load_context_window_map,
    lookup_context_window,
    normalize_model_name,
    update_context_window_map,
)


class AgentConfig(BaseModel):
    # Check the config files in minisweagent/config for example settings
    system_template: str
    instance_template: str
    timeout_template: str
    format_error_template: str
    action_observation_template: str
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    step_limit: int = 0
    cost_limit: float = 3.0


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self._live_traj_path: Path | None = None
        self.context_window_max: int | None = None
        self.context_window_source: str | None = None
        self.context_window_mode: str = "interactive"

    def set_live_trajectory_path(self, path: Path | str | None):
        """Set the path for live trajectory streaming (JSONL format).

        When set, each message will be appended to this file as it's added,
        allowing real-time monitoring of the agent's thoughts and actions.
        Use: tail -f <path> | jq .
        """
        self._live_traj_path = Path(path) if path else None
        if self._live_traj_path:
            # Ensure parent directory exists and clear any existing file
            self._live_traj_path.parent.mkdir(parents=True, exist_ok=True)
            self._live_traj_path.unlink(missing_ok=True)

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = self.config.model_dump() | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        message = {"role": role, "content": content, "timestamp": time.time(), **kwargs}
        self.messages.append(message)

        # Stream to live trajectory file if configured
        if self._live_traj_path:
            try:
                with open(self._live_traj_path, 'a') as f:
                    f.write(json.dumps(message) + '\n')
            except Exception:
                pass  # Don't let streaming failures break the agent

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        self._resolve_context_window_pre_lifecycle()
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)
        return response

    def _query_model_raw(self, user_prompt: str) -> dict:
        """Query the model without enforcing step/cost limits (pre-lifecycle)."""
        self.add_message("user", user_prompt)
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)
        return response

    def prompt_user(self, prompt: str) -> str:
        """Prompt the user for input. Subclasses can override for richer UIs."""
        return input(prompt)

    def _prompt_user_logged(self, prompt: str) -> str:
        self.add_message("assistant", prompt)
        response = self.prompt_user(prompt)
        self.add_message("user", response)
        return response.strip()

    def _resolve_context_window_pre_lifecycle(self) -> None:
        model_name = getattr(getattr(self.model, "config", None), "model_name", "")
        if not model_name:
            return

        context_map = load_context_window_map()
        resolved = lookup_context_window(model_name, context_map)
        if resolved:
            self.context_window_max = resolved
            return

        if self.context_window_mode == "batch":
            resolved, source = self._resolve_context_window_via_search(model_name)
            if resolved is None:
                self.add_message(
                    "assistant",
                    f"Failed to resolve context window for {model_name} in batch mode. Aborting run.",
                )
                raise RuntimeError(f"Context window lookup failed for {model_name}")
            update_context_window_map(model_name, resolved)
            self.context_window_max = resolved
            self.context_window_source = source
            self.add_message(
                "assistant",
                f"Resolved context window for {model_name}: {resolved} tokens (source: {source}).",
            )
            return

        attempts = 0
        max_attempts = 10
        while attempts < max_attempts:
            attempts += 1
            choice = self._prompt_user_logged(
                "Context window for this model is unknown. "
                "Type 'manual' to enter a max token limit or 'search' to let the agent look it up."
            ).lower()
            if choice in {"manual", "m", "1", "enter"}:
                raw_value = self._prompt_user_logged(
                    "Enter the model's max context window (tokens), e.g. 128000 or 128k."
                )
                resolved = self._interpret_context_window_input(model_name, raw_value)
                if resolved is None:
                    self.add_message(
                        "assistant",
                        "Could not interpret that value. Please provide a number like 128000 or 128k.",
                    )
                    continue
                update_context_window_map(model_name, resolved)
                self.context_window_max = resolved
                self.context_window_source = "manual"
                self.add_message(
                    "assistant",
                    f"Saved context window for {normalize_model_name(model_name)}: {resolved} tokens.",
                )
                return
            if choice in {"search", "s", "2"}:
                resolved, source = self._resolve_context_window_via_search(model_name)
                if resolved is None:
                    self.add_message(
                        "assistant",
                        "Search did not produce a clear context window. Try manual entry or search again.",
                    )
                    continue
                confirmation = self._prompt_user_logged(
                    f"Found {resolved} tokens for {model_name} from {source}. Accept? (y/n)"
                ).lower()
                if confirmation in {"y", "yes"}:
                    update_context_window_map(model_name, resolved)
                    self.context_window_max = resolved
                    self.context_window_source = source
                    self.add_message(
                        "assistant",
                        f"Saved context window for {normalize_model_name(model_name)}: {resolved} tokens.",
                    )
                    return
                self.add_message("assistant", "Okay, not saved. Let's try again.")
                continue
            self.add_message("assistant", "Please respond with 'manual' or 'search'.")

        self.add_message(
            "assistant",
            f"Unable to resolve context window for {model_name} after {max_attempts} attempts. Aborting run.",
        )
        raise RuntimeError(f"Context window lookup failed for {model_name}")

    def _interpret_context_window_input(self, model_name: str, user_input: str) -> int | None:
        prompt = (
            "Interpret the user's input as a context window size in tokens.\n"
            f"Model: {model_name}\n"
            f"User input: {user_input!r}\n"
            "Return ONLY JSON: {\"token_limit\": <int or null>}. "
            "If the input is ambiguous or not a number, return null. "
            "Treat 'k' as thousand (64k => 64000)."
        )
        response = self._query_model_raw(prompt)
        content = response.get("content", "")
        token_limit = _extract_token_limit_from_content(content)
        return token_limit

    def _resolve_context_window_via_search(self, model_name: str) -> tuple[int | None, str | None]:
        prompt = (
            "Find the authoritative documentation source for the max context window (tokens) "
            f"of the model named {model_name}. Return ONLY JSON: "
            "{\"url\": \"...\", \"notes\": \"...\"}. If unsure, return {\"url\": null}."
        )
        response = self._query_model_raw(prompt)
        content = response.get("content", "")
        url = _extract_url_from_content(content)
        if not url:
            return None, None

        self.add_message("assistant", f"Fetching context window source: {url}")
        fetch_output = self._fetch_url_text(url)
        if fetch_output is None:
            return None, url

        extract_prompt = (
            "Extract the max context window (tokens) for the specified model from the text below.\n"
            f"Model: {model_name}\n"
            f"Source: {url}\n"
            "Return ONLY JSON: {\"token_limit\": <int or null>}. "
            "If the value is not present, return null.\n\n"
            f"TEXT:\n{fetch_output}"
        )
        response = self._query_model_raw(extract_prompt)
        token_limit = _extract_token_limit_from_content(response.get("content", ""))
        return token_limit, url

    def _fetch_url_text(self, url: str) -> str | None:
        """Fetch and lightly extract text from a URL via the environment."""
        command = (
            "python - <<'PY'\n"
            "import re\n"
            "import sys\n"
            "import urllib.request\n"
            "url = " + repr(url) + "\n"
            "req = urllib.request.Request(url, headers={'User-Agent': 'mini-swe-agent'})\n"
            "try:\n"
            "    with urllib.request.urlopen(req, timeout=20) as resp:\n"
            "        raw = resp.read(200000)\n"
            "except Exception as exc:\n"
            "    print(f'ERROR: {exc}')\n"
            "    sys.exit(1)\n"
            "text = raw.decode('utf-8', errors='ignore')\n"
            "text = re.sub(r'(?is)<script.*?>.*?</script>', ' ', text)\n"
            "text = re.sub(r'(?is)<style.*?>.*?</style>', ' ', text)\n"
            "text = re.sub(r'(?is)<[^>]+>', ' ', text)\n"
            "lines = [line.strip() for line in text.splitlines() if line.strip()]\n"
            "selected = [line for line in lines if re.search(r'(context|token)', line, re.I)]\n"
            "if not selected:\n"
            "    selected = lines[:200]\n"
            "print('\\n'.join(selected[:80]))\n"
            "PY"
        )
        result = self.env.execute(command)
        output = result.get("output", "")
        if result.get("returncode") != 0:
            self.add_message(
                "assistant",
                f"Failed to fetch {url}. Output: {output[:500]}",
            )
            return None
        snippet = output.strip()
        self.add_message(
            "assistant",
            f"Fetched {len(output)} characters from {url}.",
        )
        return snippet[:4000]

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)

        # Add step limit warnings
        observation = self._add_step_limit_warning(observation)

        self.add_message("user", observation)
        return output

    def _add_step_limit_warning(self, observation: str) -> str:
        """Add warnings when approaching step limit."""
        if self.config.step_limit <= 0:
            return observation

        steps_used = self.model.n_calls
        steps_remaining = self.config.step_limit - steps_used
        usage_percent = steps_used / self.config.step_limit

        if steps_remaining == 1:
            # Final step warning
            warning = """
<URGENT_WARNING>
⚠️ THIS IS YOUR FINAL STEP! You have used {used}/{limit} steps.

You MUST submit your solution NOW. Run:
```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff --cached
```

If you have not staged your changes yet, you will lose all your work.
If you have unstaged changes, run `git add -A` first (this step), then submit on your next (final) step.
</URGENT_WARNING>
""".format(used=steps_used, limit=self.config.step_limit)
            return observation + warning

        elif usage_percent >= 0.9:
            # 90% warning
            warning = """
<WARNING>
⏰ RUNNING LOW ON STEPS: You have used {used}/{limit} steps ({remaining} remaining).

You should wrap up your work soon and submit your solution.
Remember the two-step submission process:
1. Stage changes: `git add -A` (review output for errors)
2. Submit: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff --cached`
</WARNING>
""".format(used=steps_used, limit=self.config.step_limit, remaining=steps_remaining)
            return observation + warning

        return observation

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(self.config.action_regex, response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except (TimeoutError, subprocess.TimeoutExpired) as e:
            output = e.output.decode("utf-8", errors="replace") if getattr(e, "output", None) else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        self.has_finished(output)
        return output | {"action": action["action"]}

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    return stripped


def _extract_token_limit_from_content(content: str) -> int | None:
    text = _strip_code_fences(content)
    try:
        data = json.loads(text)
        token_limit = data.get("token_limit")
        if token_limit is None:
            return None
        return int(token_limit)
    except Exception:
        pass
    match = re.search(r"(\d{2,7})\s*(k|K)?", text)
    if not match:
        return None
    value = int(match.group(1))
    if match.group(2):
        value *= 1000
    return value if value > 0 else None


def _extract_url_from_content(content: str) -> str | None:
    text = _strip_code_fences(content)
    try:
        data = json.loads(text)
        url = data.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    except Exception:
        pass
    match = re.search(r"https?://\S+", text)
    return match.group(0) if match else None
