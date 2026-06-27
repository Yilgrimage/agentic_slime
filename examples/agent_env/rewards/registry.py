from __future__ import annotations

from typing import Any, Awaitable, Callable

from slime.utils.types import Sample

from . import legacy, naive, ropd, valleydance
from .config import reward_cfg_path
from .extractors import record_reward_result, runtime_env
from .types import RewardResult

RewardFn = Callable[[Any, list[Sample]], Awaitable[list[RewardResult]]]


def impl_name(args: Any) -> str:
    name = runtime_env(args, "AGENT_ENV_RM_IMPL", "").strip().lower()
    if not name:
        name = runtime_env(args, "AGENT_ENV_REWARD_IMPL", "").strip().lower()
    if not name:
        name = str(
            reward_cfg_path(args, "impl", "")
            or reward_cfg_path(args, "rm_impl", "")
            or reward_cfg_path(args, "type", "")
        ).strip().lower()
    if not name:
        return "legacy"
    aliases = {
        "none": "legacy",
        "aux": "legacy",
        "legacy_aux": "legacy",
        "generic": "legacy",
        "naive_judge": "naive",
        "valleydance_like": "valleydance",
        "valleydance_v2": "valleydance",
        "ropd_like": "ropd",
    }
    return aliases.get(name, name)


async def score(args: Any, samples: Sample | list[Sample], **_: Any) -> float | list[float]:
    single = isinstance(samples, Sample)
    sample_list = [samples] if single else samples
    if not isinstance(sample_list, list):
        raise TypeError("agent-env reward registry expects a Sample or list[Sample]")

    name = impl_name(args)
    impl = {
        "legacy": legacy.score,
        "naive": naive.score,
        "valleydance": valleydance.score,
        "ropd": ropd.score,
    }.get(name)
    if impl is None:
        raise ValueError("AGENT_ENV_RM_IMPL must be one of: legacy, naive, valleydance, ropd")

    results = await impl(args, sample_list, single=single)
    if len(results) != len(sample_list):
        raise ValueError(f"reward impl {name} returned {len(results)} scores for {len(sample_list)} samples")
    values = []
    for sample, result in zip(sample_list, results, strict=True):
        record_reward_result(sample, name, result)
        values.append(float(result.score))
    return values[0] if single else values
