"""Agent-env Slime training entrypoint.

This entrypoint keeps agent-env runtime arguments on the Slime ``args`` object
instead of relying on Ray actor environment-variable propagation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path

from slime.backends.megatron_utils import arguments as megatron_arguments
from slime.utils.arguments import parse_args


_LOGGER = logging.getLogger(__name__)
_ORIGINAL_HF_VALIDATE_ARGS = megatron_arguments._hf_validate_args
_ITER_CHECKPOINT_RE = re.compile(r"iter_(\d{7})$")


def _save_with_common_state_preprocess(iteration, model, optimizer, opt_param_scheduler) -> None:
    """Save Megatron checkpoints with Megatron's common-state rank scrubber.

    Slime's helper calls ``save_checkpoint`` without the preprocessor used by
    Megatron's native training loop. In multi-rank agent-env runs that leaves
    rank-local fields in ``args`` and can make distributed checkpoint validation
    fail at the final save.
    """

    from megatron.training.global_vars import get_args
    from megatron.training.training import preprocess_common_state_dict
    from slime.backends.megatron_utils.checkpoint import save_checkpoint
    from slime.backends.megatron_utils.model import (
        disable_forward_pre_hook,
        enable_forward_pre_hook,
        should_disable_forward_pre_hook,
    )

    args = get_args()
    disable_pre_hook = should_disable_forward_pre_hook(args)
    if disable_pre_hook:
        disable_forward_pre_hook(model)
    save_checkpoint(
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far=0,
        checkpointing_context=None,
        train_data_iterator=None,
        preprocess_common_state_dict_fn=preprocess_common_state_dict,
    )
    if disable_pre_hook:
        enable_forward_pre_hook(model)


def _train_runtime_env(args, name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value:
        return value
    train_env_vars = getattr(args, "train_env_vars", None) or {}
    value = train_env_vars.get(name)
    if value is None or value == "":
        return default
    return str(value)


def _agent_env_max_checkpoints(args) -> int | None:
    raw = _train_runtime_env(args, "AGENT_ENV_MAX_CHECKPOINTS")
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        _LOGGER.warning("Ignoring invalid AGENT_ENV_MAX_CHECKPOINTS=%r", raw)
        return None
    if value < 1:
        return None
    return value


def _is_global_rank_zero() -> bool:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:  # pragma: no cover - rank checks must not break saves.
        _LOGGER.debug("Could not query torch distributed rank for checkpoint pruning", exc_info=True)
    return os.environ.get("RANK", "0") == "0"


def _iter_checkpoint_dirs(save_root: Path) -> list[tuple[int, Path]]:
    if not save_root.is_dir():
        return []
    checkpoints: list[tuple[int, Path]] = []
    for path in save_root.iterdir():
        match = _ITER_CHECKPOINT_RE.fullmatch(path.name)
        if match and path.is_dir():
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def _prune_old_checkpoints(args) -> None:
    limit = _agent_env_max_checkpoints(args)
    if limit is None or not _is_global_rank_zero():
        return

    save_root_raw = getattr(args, "save", None)
    if not save_root_raw:
        return
    save_root = Path(save_root_raw)
    checkpoints = _iter_checkpoint_dirs(save_root)
    stale = checkpoints[:-limit]
    for _iteration, path in stale:
        try:
            shutil.rmtree(path)
            _LOGGER.info("Pruned old checkpoint: %s", path)
        except FileNotFoundError:
            continue
        except Exception:
            _LOGGER.warning("Failed to prune old checkpoint: %s", path, exc_info=True)


def _install_agent_env_megatron_actor() -> None:
    """Route agent-env training actors through the checkpoint-safe save path."""

    from slime.backends.megatron_utils import actor as actor_module
    from slime.utils.timer import timer

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        if self.args.offload_train:
            self.wake_up()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        _save_with_common_state_preprocess(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
        )

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            actor_module.save_hf_model_to_path(
                self.args,
                Path(self.args.save_hf.format(rollout_id=rollout_id)),
                self.model,
            )

        if not self.args.async_save or force_sync:
            _prune_old_checkpoints(self.args)

        if self.args.offload_train:
            self.sleep()

    actor_module.MegatronTrainRayActor.save_model = save_model


def _raw_text_config_has_moe_keys(hf_config, args) -> bool:
    config_path = Path(getattr(hf_config, "_name_or_path", "") or getattr(args, "hf_checkpoint", "")) / "config.json"
    try:
        raw = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return True

    text_config = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else raw
    return any(
        key in text_config
        for key in (
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "num_experts",
            "n_routed_experts",
            "num_local_experts",
        )
    )


def _is_dense_qwen35_synthetic_moe_error(exc: AssertionError, hf_config, args) -> bool:
    if getattr(args, "num_experts", None) is not None:
        return False
    if _raw_text_config_has_moe_keys(hf_config, args):
        return False

    message = str(exc)
    prefix = "hf_validate_args failed: "
    if not message.startswith(prefix):
        return False

    errors = [part.strip() for part in message[len(prefix) :].split(";") if part.strip()]
    return bool(errors) and all(
        (
            "moe_intermediate_size" in error
            and "moe_ffn_hidden_size None" in error
        )
        or (
            "shared_expert_intermediate_size" in error
            and "moe_shared_expert_intermediate_size None" in error
        )
        for error in errors
    )


def _hf_validate_args(args, hf_config):
    try:
        return _ORIGINAL_HF_VALIDATE_ARGS(args, hf_config)
    except AssertionError as exc:
        if _is_dense_qwen35_synthetic_moe_error(exc, hf_config, args):
            _LOGGER.warning(
                "Ignoring synthesized MoE HF config defaults for dense checkpoint %s: %s",
                getattr(args, "hf_checkpoint", None),
                exc,
            )
            return None
        raise


def add_agent_env_arguments(parser):
    parser.add_argument(
        "--agent-env-train-loop",
        choices=("sync", "async"),
        default="async",
        help="Select the base Slime training loop for agent-env runs.",
    )
    parser.add_argument(
        "--env-server-url",
        type=str,
        default=None,
        help="Agent-env router URL consumed by custom rollout functions.",
    )
    return parser


def main() -> None:
    megatron_arguments._hf_validate_args = _hf_validate_args
    _install_agent_env_megatron_actor()

    args = parse_args(add_agent_env_arguments)
    if not getattr(args, "env_server_url", None):
        raise ValueError("agent-env training requires --env-server-url; do not rely on Ray env propagation.")

    if args.agent_env_train_loop == "sync":
        from train import train
    else:
        from train_async import train

    train(args)


if __name__ == "__main__":
    main()
