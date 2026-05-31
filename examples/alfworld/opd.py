from __future__ import annotations

from typing import Any

import torch

from slime.utils.types import Sample


def _teacher_log_probs(teacher_response: dict, response_length: int) -> torch.Tensor:
    meta_info = teacher_response.get('meta_info', {})
    entries = meta_info.get('input_token_logprobs') or []
    log_probs = torch.tensor([item[0] for item in entries[1:]], dtype=torch.float32)
    return log_probs[-response_length:]


def _env_rewards(samples: list[Sample]) -> list[float]:
    return [float((sample.metadata or {}).get('alfworld_reward', 0.0)) for sample in samples]


def _normalize_like_slime_default(args: Any, rewards: list[float]) -> list[float]:
    if args.advantage_estimator not in ['grpo', 'gspo', 'reinforce_plus_plus_baseline']:
        return rewards
    if not getattr(args, 'rewards_normalization', False):
        return rewards

    tensor = torch.tensor(rewards, dtype=torch.float32)
    group_size = int(getattr(args, 'n_samples_per_prompt', len(rewards)))
    if group_size > 0 and tensor.numel() % group_size == 0:
        grouped = tensor.reshape(-1, group_size)
    else:
        grouped = tensor.view(1, -1)
    normalized = grouped - grouped.mean(dim=-1, keepdim=True)
    if args.advantage_estimator in ['grpo', 'gspo'] and getattr(args, 'grpo_std_normalization', False):
        normalized = normalized / (grouped.std(dim=-1, keepdim=True) + 1e-6)
    return normalized.flatten().tolist()


def post_process_rewards(args: Any, samples: list[Sample], **kwargs):
    for sample in samples:
        teacher_response = sample.reward
        if not isinstance(teacher_response, dict):
            raise ValueError('ALFWorld OPD post-process expected raw teacher response in sample.reward.')
        sample.teacher_log_probs = _teacher_log_probs(teacher_response, sample.response_length)
        sample.reward = float((sample.metadata or {}).get('alfworld_reward', 0.0))

    raw_rewards = _env_rewards(samples)
    rewards = _normalize_like_slime_default(args, raw_rewards)
    return raw_rewards, rewards
