from __future__ import annotations

from typing import Any

from slime.utils.types import Sample

from examples.agent_env.rewards.registry import score


async def group_reward(args: Any, samples: Sample | list[Sample], **_: Any) -> float | list[float]:
    """Slime custom RM entrypoint for agent-env rewards.

    Reward composition lives in examples.agent_env.rewards. The post-process
    hook only adapts already-computed RM rewards to Slime's reward tensor path.
    """

    return await score(args, samples)
