"""
AutoHarness: Automatically synthesizing a code harness to prevent LLM agents
from taking illegal/invalid actions in an environment.

Based on: arxiv.org/abs/2603.03329
"AutoHarness: improving LLM agents by automatically synthesizing a code harness"

Architecture:
  Environment → [Harness (auto-synthesized code)] → LLM Agent → Action
                        ↑
               Iterative refinement via environment feedback

Two modes:
  1. Harness mode     : Code validates/filters LLM output → only legal actions pass
  2. Policy mode      : Code entirely replaces the LLM at decision time
"""

import re
import traceback
from typing import Callable, Optional
from anthropic import Anthropic

client = Anthropic()

# ---------------------------------------------------------------------------
# Harness Synthesis
# ---------------------------------------------------------------------------

HARNESS_SYSTEM_PROMPT = """You are an expert Python programmer specializing in building robust
validation layers for game-playing AI agents.

Your job is to write a Python function called `get_legal_action` that:
1. Takes the raw action string from an LLM agent and the current game observation.
2. Parses and validates the action against the game rules.
3. Returns the validated (and possibly corrected/re-parsed) action if legal.
4. Raises a ValueError with a clear message if the action is illegal.

The function signature must be exactly:
    def get_legal_action(raw_action: str, observation: str) -> str:

Rules:
- Only output the Python function, nothing else. No markdown fences, no explanation.
- Be defensive: handle malformed strings, extra whitespace, wrong formats, etc.
- Prefer parsing robustly over rejecting valid-but-noisy output.
"""

POLICY_SYSTEM_PROMPT = """You are an expert Python programmer specializing in game-playing AI.

Your job is to write a Python function called `compute_action` that:
1. Takes the current game observation string.
2. Analyses the game state using pure Python logic.
3. Returns the best legal action as a string.

The function signature must be exactly:
    def compute_action(observation: str) -> str:

Rules:
- Only output the Python function, nothing else. No markdown fences, no explanation.
- The function must NEVER return an illegal action.
- Use heuristics, search, or pattern matching — whatever works best.
"""


def _extract_function(text: str, func_name: str) -> str:
    """Strip any accidental markdown fences and return clean Python code."""
    text = re.sub(r"```(?:python)?", "", text).strip("`").strip()
    # Ensure the function is present
    if f"def {func_name}" not in text:
        raise ValueError(f"Synthesized code does not contain `def {func_name}`")
    return text


def _execute_function(code: str, func_name: str, *args):
    """Compile and execute a synthesized function in an isolated namespace."""
    namespace: dict = {}
    exec(compile(code, "<harness>", "exec"), namespace)  # noqa: S102
    if func_name not in namespace:
        raise ValueError(f"Function `{func_name}` not found after exec.")
    return namespace[func_name](*args)


class HarnessGenerator:
    """
    Iteratively synthesizes and refines a validation harness using an LLM.

    Args:
        game_description: Natural-language description of the game rules and
                          the action format expected by the environment.
        max_iterations:   Maximum refinement rounds before giving up.
        mode:             'harness' (validate LLM output) or
                          'policy'  (replace LLM entirely with code).
        model:            Claude model to use for synthesis.
    """

    def __init__(
        self,
        game_description: str,
        max_iterations: int = 10,
        mode: str = "harness",
        model: str = "claude-sonnet-4-20250514",
    ):
        self.game_description = game_description
        self.max_iterations = max_iterations
        self.mode = mode
        self.model = model

        self._func_name = "get_legal_action" if mode == "harness" else "compute_action"
        self._system_prompt = (
            HARNESS_SYSTEM_PROMPT if mode == "harness" else POLICY_SYSTEM_PROMPT
        )
        self._code: Optional[str] = None
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, sample_observations: list[str]) -> str:
        """
        Run the full iterative synthesis loop.

        For each sample observation we:
          1. Ask the synthesizer to (re)write the harness/policy.
          2. Execute the code against the observation.
          3. If it errors, feed the traceback back and repeat.

        Returns the final synthesized code.
        """
        initial_user_msg = (
            f"Game description:\n{self.game_description}\n\n"
            f"Sample observations to validate against:\n"
            + "\n---\n".join(sample_observations)
            + "\n\nWrite the `"
            + self._func_name
            + "` function."
        )
        self._history = [{"role": "user", "content": initial_user_msg}]

        for iteration in range(1, self.max_iterations + 1):
            print(f"[AutoHarness] Synthesis iteration {iteration}/{self.max_iterations}")

            # Ask LLM to (re)write the function
            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self._system_prompt,
                messages=self._history,
            )
            code_text = response.content[0].text
            self._history.append({"role": "assistant", "content": code_text})

            try:
                self._code = _extract_function(code_text, self._func_name)
            except ValueError as e:
                self._history.append(
                    {
                        "role": "user",
                        "content": f"Error extracting function: {e}\nPlease rewrite.",
                    }
                )
                continue

            # Validate against sample observations
            errors = self._validate_on_samples(sample_observations)
            if not errors:
                print(f"[AutoHarness] ✓ Harness synthesized successfully in {iteration} iteration(s).")
                return self._code

            # Feed errors back for refinement
            feedback = (
                "The synthesized code produced errors on the sample observations.\n"
                "Please fix ALL of the following issues and rewrite the complete function:\n\n"
                + "\n\n".join(errors)
            )
            self._history.append({"role": "user", "content": feedback})

        raise RuntimeError(
            f"[AutoHarness] Failed to synthesize a valid harness after "
            f"{self.max_iterations} iterations."
        )

    def get_code(self) -> Optional[str]:
        """Return the currently synthesized code (None if not yet synthesized)."""
        return self._code

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_on_samples(self, sample_observations: list[str]) -> list[str]:
        """Run the synthesized code on sample observations and collect errors."""
        errors = []
        for i, obs in enumerate(sample_observations):
            try:
                if self.mode == "harness":
                    # For harness mode, we just need it not to crash on a dummy action
                    _execute_function(self._code, self._func_name, "dummy_action", obs)
                else:
                    result = _execute_function(self._code, self._func_name, obs)
                    if not isinstance(result, str) or not result.strip():
                        errors.append(
                            f"Sample {i}: compute_action returned empty/non-string: {result!r}"
                        )
            except Exception:
                tb = traceback.format_exc()
                errors.append(f"Sample {i} (observation: {obs[:80]!r}):\n{tb}")
        return errors


# ---------------------------------------------------------------------------
# Harnessed Agent
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """You are an expert, logical, and strategic AI game player.
Analyze the game state and determine the single best move.

First, reason step-by-step. Then enclose your final move in <move></move> tags.
Do not add any text after the closing </move> tag.
"""


class HarnessedAgent:
    """
    An LLM agent wrapped with an auto-synthesized code harness.

    Workflow per step:
      1. LLM proposes a raw action given the observation.
      2. The harness validates the action.
      3. If invalid → retry up to `max_action_retries` times.
      4. Return the first valid action, or raise if all retries fail.
    """

    def __init__(
        self,
        harness_code: str,
        model: str = "claude-sonnet-4-20250514",
        max_action_retries: int = 3,
    ):
        self.harness_code = harness_code
        self.model = model
        self.max_action_retries = max_action_retries

    def act(self, observation: str) -> str:
        """Return a validated action for the given observation."""
        history = [{"role": "user", "content": observation}]

        for attempt in range(1, self.max_action_retries + 1):
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=AGENT_SYSTEM_PROMPT,
                messages=history,
            )
            raw = response.content[0].text

            # Extract action from <move> tags
            match = re.search(r"<move>(.*?)</move>", raw, re.DOTALL)
            raw_action = match.group(1).strip() if match else raw.strip()

            # Validate through harness
            try:
                validated = _execute_function(
                    self.harness_code, "get_legal_action", raw_action, observation
                )
                return validated
            except (ValueError, Exception) as e:
                print(f"  [Agent] Attempt {attempt}: illegal action '{raw_action}' → {e}")
                if attempt < self.max_action_retries:
                    history.append({"role": "assistant", "content": raw})
                    history.append(
                        {
                            "role": "user",
                            "content": (
                                f"That action was illegal: {e}\n"
                                "Please try a different, LEGAL move. "
                                "Remember to use <move></move> tags."
                            ),
                        }
                    )

        raise RuntimeError("All action retries exhausted — could not produce a legal action.")


class PolicyAgent:
    """
    An agent whose entire decision policy is synthesized Python code.
    Zero LLM calls at inference time.
    """

    def __init__(self, policy_code: str):
        self.policy_code = policy_code

    def act(self, observation: str) -> str:
        return _execute_function(self.policy_code, "compute_action", observation)
