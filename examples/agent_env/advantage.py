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
      use discrete good/bad marks only to redistribute its magnitude. A mark
      aligned with the outcome direction multiplies the advantage by the
      configured aligned scale; an opposed mark multiplies it by the opposed
      scale. Neutral tokens and zero advantages are unchanged.
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
    - ``teacher_anchored_state_aggregation`` (TASA-GRPO): consume a
      state-conditioned local REINFORCE baseline computed from teacher-anchored
      semantic state predicates. Tokens whose abstract state lacks enough peer
      support keep the original GRPO advantage exactly.
    - ``teacher_anchored_value_gae`` (TASA-GAE): consume a teacher-prior,
      student-outcome calibrated state value estimate. Supported tokens use the
      segment GAE advantage directly; unsupported tokens keep vanilla GRPO.
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
    if beta == 0 and mode in {"outcome_plus_process", "teacher_anchored_state_aggregation"}:
        rollout_data["advantages"] = [value for value in base_returns]
        rollout_data["returns"] = base_returns
        advantage_dump.maybe_dump_token_advantages(args, rollout_data, base_returns, None, base_returns)
        return

    process_tensors = _process_tensors(process_advantages, base_returns)
    if mode == "outcome_plus_process":
        advantages = _outcome_plus_process(base_returns, process_tensors, beta)
    elif mode == "outcome_reweight":
        advantages, reweight_metrics = _outcome_reweight(
            base_returns,
            process_tensors,
            aligned_scale=credit_assignment.reweight_aligned_scale(args),
            opposed_scale=credit_assignment.reweight_opposed_scale(args),
        )
        rollout_data.update(reweight_metrics)
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
    elif mode == "teacher_anchored_state_aggregation":
        support_masks = rollout_data.get("process_advantage_masks")
        if not support_masks:
            raise ValueError("TASA-GRPO requires response-aligned process_advantage_masks")
        support_tensors = _process_tensors(support_masks, base_returns)
        advantages = _teacher_anchored_state_aggregation(base_returns, process_tensors, support_tensors, beta)
    elif mode == "teacher_anchored_value_gae":
        support_masks = rollout_data.get("process_advantage_masks")
        if not support_masks:
            raise ValueError("TASA-GAE requires response-aligned process_advantage_masks")
        support_tensors = _process_tensors(support_masks, base_returns)
        advantages = _teacher_anchored_value_gae(base_returns, process_tensors, support_tensors)
    else:
        raise ValueError(f"Unsupported credit assignment advantage mode: {mode}")

    if mode in {"teacher_anchored_state_aggregation", "teacher_anchored_value_gae"}:
        advantages = _scale_tasa_teacher_advantages(
            args,
            advantages,
            rollout_data.get("off_policy_loss_masks"),
        )

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
    *,
    aligned_scale: float,
    opposed_scale: float,
) -> tuple[list[torch.Tensor], dict[str, list[float]]]:
    """Scheme B: sign-preserving discrete reweighting of outcome credit."""

    advantages: list[torch.Tensor] = []
    aligned_token_count: list[float] = []
    opposed_token_count: list[float] = []
    marked_zero_advantage_token_count: list[float] = []
    aligned_delta_abs: list[float] = []
    opposed_delta_abs: list[float] = []
    sign_flip_rate: list[float] = []
    for base, process in zip(base_returns, process_tensors, strict=True):
        marked = process != 0
        aligned = marked & ((base * process) > 0)
        opposed = marked & ((base * process) < 0)
        marked_zero_advantage = marked & (base == 0)
        value = torch.where(aligned, base * aligned_scale, base)
        value = torch.where(opposed, base * opposed_scale, value)
        advantages.append(value)

        aligned_count = float(aligned.sum().item())
        opposed_count = float(opposed.sum().item())
        marked_count = float(marked.sum().item())
        aligned_token_count.append(aligned_count)
        opposed_token_count.append(opposed_count)
        marked_zero_advantage_token_count.append(float(marked_zero_advantage.sum().item()))
        aligned_delta_abs.append(
            float(torch.abs(value[aligned] - base[aligned]).mean().item()) if aligned_count else 0.0
        )
        opposed_delta_abs.append(
            float(torch.abs(value[opposed] - base[opposed]).mean().item()) if opposed_count else 0.0
        )
        flips = marked & (base != 0) & ((base * value) < 0)
        sign_flip_rate.append(float(flips.sum().item() / marked_count) if marked_count else 0.0)
    return advantages, {
        "ca_reweight_aligned_token_count": aligned_token_count,
        "ca_reweight_opposed_token_count": opposed_token_count,
        "ca_reweight_marked_zero_advantage_token_count": marked_zero_advantage_token_count,
        "ca_reweight_aligned_delta_abs": aligned_delta_abs,
        "ca_reweight_opposed_delta_abs": opposed_delta_abs,
        "ca_reweight_sign_flip_rate": sign_flip_rate,
    }


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


def _teacher_anchored_state_aggregation(
    base_returns: list[torch.Tensor],
    local_tensors: list[torch.Tensor],
    support_tensors: list[torch.Tensor],
    beta: float,
) -> list[torch.Tensor]:
    """TASA-GRPO: mix GRPO with supported state-conditioned local baselines."""

    advantages: list[torch.Tensor] = []
    for base, local, support in zip(base_returns, local_tensors, support_tensors, strict=True):
        supported = support > 0
        mixed = (1.0 - beta) * base + beta * local
        advantages.append(torch.where(supported, mixed, base))
    return advantages


def _teacher_anchored_value_gae(
    base_returns: list[torch.Tensor],
    local_tensors: list[torch.Tensor],
    support_tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    """TASA-GAE: replace supported tokens with the value-layer GAE advantage."""

    advantages: list[torch.Tensor] = []
    for base, local, support in zip(base_returns, local_tensors, support_tensors, strict=True):
        supported = support > 0
        advantages.append(torch.where(supported, local, base))
    return advantages


def _scale_tasa_teacher_advantages(
    args: Any,
    advantages: list[torch.Tensor],
    off_policy_masks: list[torch.Tensor] | None,
) -> list[torch.Tensor]:
    """Optionally scale Luffy/off-policy teacher-token advantages in TASA modes."""

    weight = credit_assignment.tasa_teacher_weight(args)
    if weight == 1.0 or not off_policy_masks:
        return advantages
    scaled: list[torch.Tensor] = []
    for advantage, mask in zip(advantages, off_policy_masks, strict=False):
        if mask is None:
            scaled.append(advantage)
            continue
        teacher_tokens = mask.to(device=advantage.device, dtype=torch.bool, non_blocking=True)
        if teacher_tokens.shape != advantage.shape:
            raise ValueError(
                f"off_policy_loss_mask shape {tuple(teacher_tokens.shape)} does not match "
                f"advantage shape {tuple(advantage.shape)}"
            )
        scaled.append(torch.where(teacher_tokens, advantage * weight, advantage))
    return scaled
