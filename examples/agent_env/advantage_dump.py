from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from examples.agent_env import dump


def maybe_dump_token_advantages(
    args: Any,
    rollout_data: dict[str, Any],
    base_returns: list[torch.Tensor],
    process_tensors: list[torch.Tensor] | None,
    final_advantages: list[torch.Tensor],
) -> None:
    """Dump the exact token-level advantages consumed by the actor.

    This is a debug artifact, not a metric. It is disabled by default and is
    intended for one-step audits that compare reward-side CA segment dumps with
    the final token advantages used by PPO.
    """

    per_step_limit = _per_step_limit(args)
    if per_step_limit <= 0:
        return
    out_dir = _dump_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"advantage_pid{os.getpid()}.jsonl"
    total_limit = _total_limit(args)
    dump_step = _dump_step_label(rollout_data)
    include_token_ids = _include_token_ids(args)
    response_lengths = rollout_data.get("response_lengths")
    loss_masks = rollout_data.get("loss_masks")
    rewards = rollout_data.get("rewards")
    sample_indices = rollout_data.get("sample_indices")
    rollout_ids = rollout_data.get("rollout_ids")
    source_names = rollout_data.get("source_names")

    with out_path.open("a", encoding="utf-8") as handle:
        for idx, final_tensor in enumerate(final_advantages):
            slot = dump.reserve_dump_slot(
                namespace="advantage",
                stage="token",
                dump_step=dump_step,
                per_step_limit=per_step_limit,
                total_limit=total_limit,
            )
            if slot is None:
                continue
            index_in_step, total_index = slot
            final_values = _to_float_list(final_tensor)
            base_values = _to_float_list(base_returns[idx])
            process_values = (
                [0.0] * len(final_values) if process_tensors is None else _to_float_list(process_tensors[idx])
            )
            loss_mask = _to_int_list(_list_get(loss_masks, idx, []))
            if len(loss_mask) != len(final_values):
                loss_mask = [1] * len(final_values)
            response_length = int(_list_get(response_lengths, idx, len(final_values)) or len(final_values))
            token_ids = _response_token_ids(rollout_data, idx, response_length) if include_token_ids else []
            if len(base_values) != len(final_values) or len(process_values) != len(final_values):
                raise ValueError(
                    "advantage dump shape mismatch: "
                    f"base={len(base_values)} process={len(process_values)} final={len(final_values)}"
                )
            payload = {
                "schema_version": "agent_env.token_advantage_audit.v1",
                "time": time.time(),
                "pid": os.getpid(),
                "dump_step": dump_step,
                "index_in_step": index_in_step,
                "index": total_index,
                "sample_index_in_batch": idx,
                "sample_index": _list_get(sample_indices, idx),
                "rollout_id": _list_get(rollout_ids, idx),
                "source_name": _list_get(source_names, idx),
                "reward": _list_get(rewards, idx),
                "response_length": len(final_values),
                "loss_mask_sum": sum(int(value) for value in loss_mask),
                "stats": {
                    "base_advantage": _stats(base_values, loss_mask),
                    "process_advantage": _stats(process_values, loss_mask),
                    "final_advantage": _stats(final_values, loss_mask),
                    "delta": _stats(
                        [final_value - base_value for final_value, base_value in zip(final_values, base_values, strict=True)],
                        loss_mask,
                    ),
                },
                "token_runs": _token_runs(
                    loss_mask=loss_mask,
                    process=process_values,
                    base=base_values,
                    final=final_values,
                ),
            }
            if include_token_ids:
                payload["response_token_ids"] = token_ids
                payload["tokens"] = _token_records(
                    token_ids=token_ids,
                    loss_mask=loss_mask,
                    process=process_values,
                    base=base_values,
                    final=final_values,
                )
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _per_step_limit(args: Any) -> int:
    return dump.int_runtime_env(args, "AGENT_ENV_ADVANTAGE_DUMP_N", "0")


def _total_limit(args: Any) -> int:
    return dump.int_runtime_env(args, "AGENT_ENV_ADVANTAGE_DUMP_TOTAL_N", "0")


def _include_token_ids(args: Any) -> bool:
    raw = dump.runtime_env(args, "AGENT_ENV_ADVANTAGE_DUMP_TOKEN_IDS", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _dump_dir(args: Any) -> Path:
    raw = dump.runtime_env(args, "AGENT_ENV_ADVANTAGE_DUMP_DIR", "").strip()
    if raw:
        return Path(raw)
    run_root = dump.runtime_env(args, "RUN_ROOT", "").strip()
    if run_root:
        return Path(run_root) / "reward_artifacts" / "advantage"
    return Path(os.getcwd()) / "reward_artifacts" / "advantage"


def _dump_step_label(rollout_data: dict[str, Any]) -> str:
    rollout_ids = rollout_data.get("rollout_ids")
    if isinstance(rollout_ids, (list, tuple)) and rollout_ids:
        return str(rollout_ids[0])
    return "unknown"


def _to_float_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().float().cpu().tolist()]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return []


def _to_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return []


def _list_get(values: Any, idx: int, default: Any = None) -> Any:
    if isinstance(values, (list, tuple)) and 0 <= idx < len(values):
        return values[idx]
    return default


def _stats(values: list[float], loss_mask: list[int] | None = None) -> dict[str, Any]:
    if loss_mask is not None and len(loss_mask) == len(values):
        selected = [value for value, mask in zip(values, loss_mask, strict=True) if int(mask) != 0]
    else:
        selected = list(values)
    if not selected:
        return {"count": 0}
    nonzero = sum(1 for value in selected if abs(value) > 1e-12)
    positive = sum(1 for value in selected if value > 0)
    negative = sum(1 for value in selected if value < 0)
    return {
        "count": len(selected),
        "mean": sum(selected) / len(selected),
        "min": min(selected),
        "max": max(selected),
        "nonzero_rate": nonzero / len(selected),
        "positive_rate": positive / len(selected),
        "negative_rate": negative / len(selected),
    }


def _response_token_ids(rollout_data: dict[str, Any], sample_idx: int, response_length: int) -> list[int]:
    tokens = _to_int_list(_list_get(rollout_data.get("tokens"), sample_idx, []))
    if response_length <= 0 or len(tokens) < response_length:
        return []
    return tokens[-response_length:]


def _token_runs(
    *,
    loss_mask: list[int],
    process: list[float],
    base: list[float],
    final: list[float],
) -> list[dict[str, Any]]:
    length = len(final)
    if length == 0:
        return []
    runs: list[dict[str, Any]] = []
    start = 0
    for idx in range(1, length + 1):
        boundary = idx == length
        if not boundary:
            boundary = (
                int(loss_mask[idx]) != int(loss_mask[start])
                or abs(process[idx] - process[start]) > 1e-12
                or abs((final[idx] - base[idx]) - (final[start] - base[start])) > 1e-12
            )
        if not boundary:
            continue
        span = slice(start, idx)
        span_base = base[span]
        span_final = final[span]
        span_process = process[span]
        span_delta = [value - reference for value, reference in zip(span_final, span_base, strict=True)]
        count = idx - start
        runs.append(
            {
                "start": start,
                "end": idx,
                "token_count": count,
                "loss_mask_sum": sum(int(value) for value in loss_mask[span]),
                "process_mean": sum(span_process) / count,
                "base_advantage_mean": sum(span_base) / count,
                "final_advantage_mean": sum(span_final) / count,
                "delta_mean": sum(span_delta) / count,
                "process_first": span_process[0],
                "base_advantage_first": span_base[0],
                "final_advantage_first": span_final[0],
                "delta_first": span_delta[0],
            }
        )
        start = idx
    return runs


def _token_records(
    *,
    token_ids: list[int],
    loss_mask: list[int],
    process: list[float],
    base: list[float],
    final: list[float],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, final_value in enumerate(final):
        token_id = token_ids[idx] if idx < len(token_ids) else None
        base_value = base[idx]
        records.append(
            {
                "i": idx,
                "token_id": token_id,
                "loss_mask": int(loss_mask[idx]),
                "process_advantage": process[idx],
                "base_advantage": base_value,
                "final_advantage": final_value,
                "delta": final_value - base_value,
            }
        )
    return records
