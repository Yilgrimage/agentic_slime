from __future__ import annotations

from typing import Any

import torch

from slime.utils.ppo_utils import get_grpo_returns

from examples.agent_env import credit_assignment


def segment_credit_assignment_advantage(args: Any, rollout_data: dict[str, Any]) -> None:
    """Add judge-derived segment shaping on top of GRPO-style scalar returns.

    This is not a replacement for GRPO's trajectory-level advantage. The base
    term remains Slime's scalar GRPO return; ``process_advantages`` is an
    optional response-aligned shaping tensor produced while the rollout side can
    still see full groups.
    """

    if getattr(args, "advantage_estimator", "grpo") not in {"grpo", "gspo", "cispo"}:
        raise ValueError("segment credit assignment currently supports GRPO/GSPO/CISPO-style scalar rewards only")

    kl: list[torch.Tensor] = rollout_data["kl"]
    rewards = torch.tensor(rollout_data["rewards"], dtype=torch.float32, device=kl[0].device)
    base_returns = get_grpo_returns(rewards, kl)
    beta = credit_assignment.beta(args)
    process_advantages = rollout_data.get("process_advantages")
    if not process_advantages or beta == 0:
        rollout_data["advantages"] = [value for value in base_returns]
        rollout_data["returns"] = base_returns
        return

    advantages: list[torch.Tensor] = []
    for idx, base in enumerate(base_returns):
        process = process_advantages[idx].to(device=base.device, dtype=base.dtype, non_blocking=True)
        if process.shape != base.shape:
            raise ValueError(
                f"process_advantages[{idx}] shape {tuple(process.shape)} does not match "
                f"base return shape {tuple(base.shape)}"
            )
        advantages.append(base + beta * process)
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = advantages
