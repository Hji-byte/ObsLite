"""Shared, auditable context policies for WebShop training and evaluation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping, Sequence


MASK_OLD_OBSERVATIONS = "mask_old_observations"
FULL_HISTORY = "full_history"
RECENT_ACTION_WINDOW = "recent_action_window"


@dataclass(frozen=True)
class RecentActionWindowPolicy:
    """SPEAR-style history selection with our SFT output format (evaluation only)."""

    keep_recent_history: int = 2

    def __post_init__(self):
        if type(self.keep_recent_history) is not int or self.keep_recent_history < 0:
            raise ValueError("keep_recent_history must be a non-negative integer")

    def to_config(self):
        return {"type": RECENT_ACTION_WINDOW,
                "keep_recent_history": self.keep_recent_history,
                "preserve_error_observations": False,
                "preserve_history_thoughts": False,
                "prompt_version": "sft_action_window_v1"}

    def apply(self, messages, trajectory_start, task_instruction=None):
        from .generate_teacher import parse_action

        if not task_instruction:
            raise ValueError("Action window requires the live environment task instruction")
        if not 0 <= trajectory_start < len(messages):
            raise ValueError("trajectory_start is outside the message sequence")
        prefix = deepcopy(list(messages[:trajectory_start]))
        if any(item.get("role") != "system" for item in prefix):
            raise ValueError("Action window evaluation requires a no-demo system prefix")
        history = messages[trajectory_start:]
        if len(history) % 2 != 1 or any(
                item.get("role") != ("user" if i % 2 == 0 else "assistant")
                for i, item in enumerate(history)):
            raise ValueError("Expected alternating observations/actions ending in observation")
        past_count = len(history) // 2
        start = max(0, past_count - self.keep_recent_history)
        lines = [f"Task instruction:\n{task_instruction}",
                 f"Previously attempted steps: {past_count}."]
        for i in range(start, past_count):
            # Old available-action lists are not part of the retained observation.
            observation = str(history[2 * i]["content"]).split("\n\nAvailable clicks:", 1)[0]
            try:
                action = parse_action(str(history[2 * i + 1]["content"]))
            except ValueError:
                # Do not leak malformed Thought text back into the prompt.
                action = "[Invalid format; no action executed]"
            lines.append(f"Previous step {i + 1}:\n{observation}\nAction: {action}")
        lines.append(f"Current step {past_count + 1}:\n{history[-1]['content']}")
        prefix.append({"role": "user", "content": "\n\n".join(lines)})
        return prefix, {"observation_count": past_count + 1,
                        "retained_history_steps": past_count - start,
                        "dropped_history_steps": start,
                        "history_assistant_messages_removed": past_count}


@dataclass(frozen=True)
class ObservationMaskingPolicy:
    """Mask old trajectory observations while retaining actions and recent state.

    ``keep_recent_history`` counts observations immediately before the current
    observation.  The task's first observation and the current observation are
    always retained in addition to that window.
    """

    keep_recent_history: int = 4
    preserve_error_observations: bool = True

    def __post_init__(self):
        if (not isinstance(self.keep_recent_history, int) or
                isinstance(self.keep_recent_history, bool) or self.keep_recent_history < 0):
            raise ValueError("keep_recent_history must be a non-negative integer")
        if not isinstance(self.preserve_error_observations, bool):
            raise ValueError("preserve_error_observations must be a boolean")

    def to_config(self):
        return {"type": MASK_OLD_OBSERVATIONS,
                "keep_recent_history": self.keep_recent_history,
                "preserve_error_observations": self.preserve_error_observations}

    def apply(self, messages: Sequence[Mapping], trajectory_start: int):
        """Return a prompt copy and masking statistics; never mutate the trajectory."""
        if not 0 <= trajectory_start <= len(messages):
            raise ValueError("trajectory_start is outside the message sequence")
        result = deepcopy(list(messages))
        observations = [index for index in range(trajectory_start, len(result))
                        if result[index].get("role") == "user"]
        if not observations:
            return result, {"masked_observations": 0, "observation_count": 0}

        keep = {observations[0], observations[-1]}
        if self.keep_recent_history:
            keep.update(observations[:-1][-self.keep_recent_history:])
        masked = 0
        for observation_step, message_index in enumerate(observations):
            content = str(result[message_index].get("content", ""))
            is_error = "Invalid action!" in content or "Invalid format." in content
            if message_index in keep or (self.preserve_error_observations and is_error):
                continue
            result[message_index]["content"] = (
                "Observation:\n"
                f"[Earlier observation after step {observation_step} omitted by context policy; "
                "the corresponding assistant action is retained.]"
            )
            masked += 1
        return result, {"masked_observations": masked,
                        "observation_count": len(observations)}


def context_policy_from_config(config):
    """Parse a plain/OmegaConf mapping. Full history deliberately returns None."""
    if config is None:
        return None
    if isinstance(config, str):
        policy_type = config
        config = {}
    else:
        policy_type = config.get("type", FULL_HISTORY)
    if policy_type == FULL_HISTORY:
        return None
    if policy_type == RECENT_ACTION_WINDOW:
        return RecentActionWindowPolicy(config.get("keep_recent_history", 2))
    if policy_type != MASK_OLD_OBSERVATIONS:
        raise ValueError(f"Unknown WebShop context policy: {policy_type}")
    recent = config.get("keep_recent_history", 4)
    if not isinstance(recent, int) or isinstance(recent, bool):
        raise ValueError("keep_recent_history must be a non-negative integer")
    preserve_errors = config.get("preserve_error_observations", True)
    if not isinstance(preserve_errors, bool):
        raise ValueError("preserve_error_observations must be a boolean")
    return ObservationMaskingPolicy(
        keep_recent_history=recent,
        preserve_error_observations=preserve_errors)


def prepare_context(messages, policy, trajectory_start, task_instruction=None):
    """Keep the disabled path byte-for-byte compatible with the existing runner."""
    if policy is None:
        return messages, None
    if isinstance(policy, RecentActionWindowPolicy):
        return policy.apply(messages, trajectory_start, task_instruction)
    return policy.apply(messages, trajectory_start)
