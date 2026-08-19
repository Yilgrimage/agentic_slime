from argparse import Namespace
from types import SimpleNamespace
import sys

from examples.agent_env import metrics


def test_define_wandb_step_metrics_binds_exact_nested_and_env_metrics(monkeypatch):
    calls = []
    fake_wandb = SimpleNamespace(
        run=SimpleNamespace(id="run-1"),
        define_metric=lambda name, **kwargs: calls.append((name, kwargs)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    metrics._WANDB_DEFINED_STEP_METRICS.clear()

    args = Namespace(use_wandb=True)
    metrics._define_wandb_step_metrics(
        args,
        {
            "rollout/step": 100,
            "rollout/response_len/median": 42.0,
            "appworld/success_rate": 0.5,
            "reward/credit_assignment/nonzero_token_rate": 0.25,
        },
        step_metric="rollout/step",
    )

    assert calls == [
        ("rollout/step", {}),
        ("appworld/success_rate", {"step_metric": "rollout/step"}),
        ("reward/credit_assignment/nonzero_token_rate", {"step_metric": "rollout/step"}),
        ("rollout/response_len/median", {"step_metric": "rollout/step"}),
    ]

    metrics._define_wandb_step_metrics(
        args,
        {"rollout/step": 101, "appworld/success_rate": 0.6},
        step_metric="rollout/step",
    )
    assert len(calls) == 4


def test_define_wandb_step_metrics_uses_independent_eval_clock_and_run(monkeypatch):
    calls = []
    first_run = SimpleNamespace(id="run-1")
    fake_wandb = SimpleNamespace(
        run=first_run,
        define_metric=lambda name, **kwargs: calls.append((name, kwargs)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    metrics._WANDB_DEFINED_STEP_METRICS.clear()

    args = Namespace(use_wandb=True)
    payload = {"eval/step": 200, "eval/dev/appworld/success_rate": 0.7}
    metrics._define_wandb_step_metrics(args, payload, step_metric="eval/step")
    second_run = SimpleNamespace(id="run-2")
    fake_wandb.run = second_run
    metrics._define_wandb_step_metrics(args, payload, step_metric="eval/step")

    assert calls == [
        ("eval/step", {}),
        ("eval/dev/appworld/success_rate", {"step_metric": "eval/step"}),
        ("eval/step", {}),
        ("eval/dev/appworld/success_rate", {"step_metric": "eval/step"}),
    ]
