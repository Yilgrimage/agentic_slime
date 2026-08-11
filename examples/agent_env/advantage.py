from __future__ import annotations

from typing import Any

import torch

from slime.utils.ppo_utils import get_grpo_returns

from examples.agent_env import credit_assignment


def segment_credit_assignment_advantage(args: Any, rollout_data: dict[str, Any]) -> None:
    """Fuse scalar outcome credit with token-aligned process credit.

    All CA modes share the same rollout-side prerequisite:
    ``examples.agent_env.credit_assignment`` extracts assistant-turn spans from
    ``metadata.token_segments``, maps judge step evidence onto those spans, and
    expands the turn values to a response-token-aligned ``process_advantages``
    tensor. This train-side hook only chooses the final composition formula.

    - ``outcome_plus_process`` (A): keep outcome GRPO as the anchor and add a
      residual process term, ``A_token = A_outcome + beta * A_process``.
    - ``outcome_reweight`` (B): preserve the sign of the outcome advantage and
      use process credit only to redistribute magnitude inside a trajectory,
      ``A_token = A_outcome * clamp_positive(1 + beta * tanh(A_process))``.
    - ``segment_reward_group_turn_norm`` (C): use the precomputed turn-level
      normalized segment reward. Rollout post-process builds
      ``R_outcome + beta * process_credit`` for each assistant turn, normalizes
      those values across turns in the same group, then expands the result to
      response tokens. This is REINFORCE++-style centering/scaling, not a
      same-prefix GRPO relative advantage.
    """

    if getattr(args, "advantage_estimator", "grpo") not in {"grpo", "gspo", "cispo"}:
        raise ValueError("segment credit assignment currently supports GRPO/GSPO/CISPO-style scalar rewards only")

    kl: list[torch.Tensor] = rollout_data["kl"]
    rewards = torch.tensor(rollout_data["rewards"], dtype=torch.float32, device=kl[0].device)
    base_returns = get_grpo_returns(rewards, kl)
    mode = credit_assignment.advantage_mode(args)
    beta = credit_assignment.beta(args)
    process_advantages = rollout_data.get("process_advantages")
    if not process_advantages:
        rollout_data["advantages"] = [value for value in base_returns]
        rollout_data["returns"] = base_returns
        return
    if beta == 0 and mode != "segment_reward_group_turn_norm":
        rollout_data["advantages"] = [value for value in base_returns]
        rollout_data["returns"] = base_returns
        return

    process_tensors = _process_tensors(process_advantages, base_returns)
    if mode == "outcome_plus_process":
        advantages = _outcome_plus_process(base_returns, process_tensors, beta)
    elif mode == "outcome_reweight":
        advantages = _outcome_reweight(base_returns, process_tensors, beta)
    elif mode == "segment_reward_group_turn_norm":
        del beta
        advantages = _segment_reward_group_turn_norm(process_tensors)
    else:
        raise ValueError(f"Unsupported credit assignment advantage mode: {mode}")

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = advantages


def _process_tensors(process_advantages: list[Any], base_returns: list[torch.Tensor]) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for idx, base in enumerate(base_returns):
        process = process_advantages[idx].to(device=base.device, dtype=base.dtype, non_blocking=True)
        if process.shape != base.shape:
            raise ValueError(
                f"process_advantages[{idx}] shape {tuple(process.shape)} does not match "
                f"base return shape {tuple(base.shape)}"
            )
        tensors.append(process)
    return tensors


def _outcome_plus_process(
    base_returns: list[torch.Tensor],
    process_tensors: list[torch.Tensor],
    beta: float,
) -> list[torch.Tensor]:
    """Scheme A: additive macro outcome advantage plus micro process credit."""

    return [base + beta * process for base, process in zip(base_returns, process_tensors, strict=True)]


def _outcome_reweight(
    base_returns: list[torch.Tensor],
    process_tensors: list[torch.Tensor],
    beta: float,
) -> list[torch.Tensor]:
    """Scheme B: sign-preserving process reweighting of the outcome advantage."""

    return [
        base * torch.clamp(1.0 + beta * torch.tanh(process), min=1e-6)
        for base, process in zip(base_returns, process_tensors, strict=True)
    ]


def _segment_reward_group_turn_norm(
    process_tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Scheme C: use post-process-computed group-turn segment advantages.

    The actual turn grouping lives in ``examples.agent_env.credit_assignment``
    because only the rollout-side sample metadata contains ``token_segments``.
    Here the actor simply consumes the response-aligned tensor.
    """

    return [tensor for tensor in process_tensors]
