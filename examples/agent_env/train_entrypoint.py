"""Agent-env Slime training entrypoint.

This entrypoint keeps agent-env runtime arguments on the Slime ``args`` object
instead of relying on Ray actor environment-variable propagation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from slime.backends.megatron_utils import arguments as megatron_arguments
from slime.utils.arguments import parse_args


_LOGGER = logging.getLogger(__name__)
_ORIGINAL_HF_VALIDATE_ARGS = megatron_arguments._hf_validate_args


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
