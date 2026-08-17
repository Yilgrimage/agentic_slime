from __future__ import annotations

from typing import Any

import torch

from slime.utils.ppo_utils import get_grpo_returns

from examples.agent_env import advantage_dump, credit_assignment


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
      ``A_token = A_outcome * clamp_positive(1 + beta * sign(A_outcome) * A_process)``.
      The ``sign(A_outcome)`` term makes positive process credit always move the
      final advantage in the encouraging direction: it strengthens positive
      outcome credit and weakens negative outcome credit. Negative process
      credit does the reverse.
    - ``outcome_bangbang_reweight`` (hard-B): preserve the sign of the outcome
      advantage with a discrete gate. Positive process marks force the token
      advantage to a configured positive boundary on positive-outcome trajectories and ``0`` on
      negative-outcome trajectories. Negative process marks force ``0`` on
      positive-outcome trajectories and the negative boundary on negative-outcome trajectories.
      Neutral tokens keep the original outcome advantage. This mode is intended
      to test strong credit assignment without allowing process credit to flip
      the outcome direction.
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
        advantage_dump.maybe_dump_token_advantages(args, rollout_data, base_returns, None, base_returns)
        return
    if beta == 0 and mode != "segment_reward_group_turn_norm":
        rollout_data["advantages"] = [value for value in base_returns]
        rollout_data["returns"] = base_returns
        advantage_dump.maybe_dump_token_advantages(args, rollout_data, base_returns, None, base_returns)
        return

    process_tensors = _process_tensors(process_advantages, base_returns)
    if mode == "outcome_plus_process":
        advantages = _outcome_plus_process(base_returns, process_tensors, beta)
    elif mode == "outcome_reweight":
        advantages = _outcome_reweight(base_returns, process_tensors, beta)
    elif mode == "outcome_bangbang_reweight":
        advantages, bangbang_metrics = _outcome_bangbang_reweight(
            base_returns,
            process_tensors,
            credit_assignment.clip(args),
        )
        rollout_data.update(bangbang_metrics)
    elif mode == "segment_reward_group_turn_norm":
        del beta
        advantages = _segment_reward_group_turn_norm(process_tensors)
    else:
        raise ValueError(f"Unsupported credit assignment advantage mode: {mode}")

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = advantages
    advantage_dump.maybe_dump_token_advantages(args, rollout_data, base_returns, process_tensors, advantages)


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
    """Scheme B: sign-preserving process reweighting of outcome credit.

    Positive process credit should make a sampled action more likely even when
    the trajectory-level outcome advantage is negative. Multiplying by
    ``sign(base)`` flips the process-credit effect for negative-outcome
    trajectories while the positive clamp keeps this mode as a magnitude
    redistribution rather than a sign-flipping additive advantage.
    """

    advantages: list[torch.Tensor] = []
    for base, process in zip(base_returns, process_tensors, strict=True):
        signed_process = torch.sign(base) * process
        multiplier = torch.clamp(1.0 + beta * signed_process, min=1e-6)
        advantages.append(base * multiplier)
    return advantages


def _outcome_bangbang_reweight(
    base_returns: list[torch.Tensor],
    process_tensors: list[torch.Tensor],
    boundary: float,
) -> tuple[list[torch.Tensor], dict[str, list[float]]]:
    """Hard-gated scheme B: process marks redistribute outcome credit by sign.

    This is deliberately not a smooth beta-scaled multiplier. It tests the
    cleanest outcome-anchored hypothesis:

    * positive outcome + good step -> fixed positive boundary credit;
    * positive outcome + bad step -> remove positive credit;
    * negative outcome + good step -> remove negative credit;
    * negative outcome + bad step -> fixed negative boundary credit.

    Process credit never flips the sign implied by the outcome. Unmarked tokens
    keep the original scalar outcome advantage.
    """

    advantages: list[torch.Tensor] = []
    boundary_delta_abs: list[float] = []
    zero_delta_abs: list[float] = []
    boundary_token_count: list[float] = []
    zero_token_count: list[float] = []
    boundary_overshoot_rate: list[float] = []
    boundary = abs(float(boundary))
    for base, process in zip(base_returns, process_tensors, strict=True):
        value = base.clone()
        good = process > 0
        bad = process < 0
        positive_outcome = base > 0
        negative_outcome = base < 0

        push_boundary = (positive_outcome & good) | (negative_outcome & bad)
        push_zero = (positive_outcome & bad) | (negative_outcome & good)
        positive_target = torch.full_like(value, boundary)
        negative_target = torch.full_like(value, -boundary)
        boundary_target = torch.where(positive_outcome, positive_target, negative_target)
        zero_target = torch.zeros_like(value)

        boundary_count = float(push_boundary.sum().item())
        zero_count = float(push_zero.sum().item())
        boundary_token_count.append(boundary_count)
        zero_token_count.append(zero_count)
        if boundary_count:
            boundary_delta = torch.abs(boundary_target[push_boundary] - base[push_boundary])
            boundary_delta_abs.append(float(boundary_delta.mean().item()))
            boundary_overshoot = (
                (positive_outcome & good & (base > boundary))
                | (negative_outcome & bad & (base < -boundary))
            )
            boundary_overshoot_rate.append(float(boundary_overshoot.sum().item() / boundary_count))
        else:
            boundary_delta_abs.append(0.0)
            boundary_overshoot_rate.append(0.0)
        if zero_count:
            zero_delta = torch.abs(zero_target[push_zero] - base[push_zero])
            zero_delta_abs.append(float(zero_delta.mean().item()))
        else:
            zero_delta_abs.append(0.0)

        value = torch.where(positive_outcome & good, positive_target, value)
        value = torch.where(positive_outcome & bad, zero_target, value)
        value = torch.where(negative_outcome & good, zero_target, value)
        value = torch.where(negative_outcome & bad, negative_target, value)
        advantages.append(value)
    return advantages, {
        "ca_bangbang_boundary_delta_abs": boundary_delta_abs,
        "ca_bangbang_zero_delta_abs": zero_delta_abs,
        "ca_bangbang_boundary_token_count": boundary_token_count,
        "ca_bangbang_zero_token_count": zero_token_count,
        "ca_bangbang_boundary_overshoot_rate": boundary_overshoot_rate,
    }


def _segment_reward_group_turn_norm(
    process_tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Scheme C: use post-process-computed group-turn segment advantages.

    The actual turn grouping lives in ``examples.agent_env.credit_assignment``
    because only the rollout-side sample metadata contains ``token_segments``.
    Here the actor simply consumes the response-aligned tensor.
    """

    return [tensor for tensor in process_tensors]
