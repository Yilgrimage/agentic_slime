# AppWorld Evaluation Handoff

Portable experiment record as of 2026-08-21. This file is committed with the
ops skill so a replacement cluster does not need the original NAS or chat
history to recover accepted AppWorld results.

Exact canonical rows are also stored in `appworld_eval_results.tsv` beside this
file. Use the TSV for calculations and this document for validity decisions.

## Reading Rules

- Treat `success_rate` as strict task completion and `env_score_mean` as
  AppWorld partial-credit score. Do not infer strict success from a nonzero
  partial score.
- Compare methods only under the same split, temperature, prompt, max-turn,
  context, and whole-episode retry protocol.
- `env_reward_mean` was scaled by either `1` or `10` in different reward
  profiles. Never compare that key across these runs; use `success_rate` and
  `env_score_mean` below.
- Formal split sizes are `dev=57`, `test_normal=168`, and
  `test_challenge=417`. The old 57-row result labeled `normal` was a lightweight
  probe, not the official full normal split.
- Temperature `0` caused much longer generations, truncation, retries, and
  occasional discards in this stack. The accepted checkpoint comparison uses
  temperature `1.0`.
- A zero discard count does not make two runs comparable when their training
  ancestry differs. In particular, `warm99` is not a from-scratch result.

## Canonical Eval Protocol

- Native agent-env eval, not a custom wrapper.
- Eval config: `appworld_eval_temperature1.yaml`.
- Sampling temperature: `1.0`.
- Complete official split, with per-task whole-episode retry only.
- Primary metrics: strict `success_rate`, raw `env_score_mean`, response length,
  truncation, discard count, and retry count.
- The fixed comparison bundle used evaluator commit `7cf28c41`.
- Source summary SHA256:
  `e6442216ab22cc4bbd289f416bc4153207ba3d88583640e0736ebe1415039f34`.

## From-Scratch CA-B + LUFFY

This is the clean from-zero run, not the warm-started CA-B series:

`appworld-Qwen3-4B-grpo-fullasync-4x8-success01-ca-b-luffy-gpt54pool-fresh0to300-execbound-nodyn-spacefix-s50-20260820T0907`

Training completed normally with 300 rollouts. Reward profile was
`appworld_ropd_ca_b_luffy_gpt54_pool.yaml`; dynamic sampling was disabled. The
temperature-1 full eval used `native_ckpt_sweep_20260821T0106`.

| Ckpt | Split | N | Strict success | Env score | Mean response tokens | Truncated | Discard | Whole-task retries |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 199 | normal | 168 | 44.05% | 0.6885 | 6,741 | 10.12% | 0 | 0 |
| 199 | challenge | 417 | 27.34% | 0.4995 | 7,544 | 25.66% | 0 | 0 |
| 249 | normal | 168 | 51.19% | 0.7263 | 7,408 | 12.50% | 0 | 0 |
| 249 | challenge | 417 | 28.30% | 0.5316 | 8,536 | 25.66% | 0 | 1 |
| 299 | normal | 168 | 46.43% | 0.7077 | 7,073 | 11.31% | 0 | 0 |
| 299 | challenge | 417 | 28.30% | 0.5362 | 8,220 | 23.50% | 0 | 0 |

Accepted conclusion: the clean CA-B run did not establish an advantage over
the 0-1+LUFFY checkpoint-299 baseline. Its best normal checkpoint was 249, and
the challenge result plateaued at 28.30%.

The dev sweep for this run used the superseded temperature-0 route and produced
49.12% / 57.89% / 47.37% strict success at checkpoints 199/249/299. Keep these
only as evidence of high dev variance; do not mix them into the temperature-1
table.

## Fixed Temperature-1 Historical Comparison

The accepted fixed bundle is:

`qwen3_4b_appworld_ca_b_vs_success01_luffy_temp1_fixed_20260820.tsv`

CA-B ancestry is confounded: checkpoint 199 came from the `warm99to300` run,
and checkpoints 249/299 came from its `resume199to300` continuation. The
0-1+LUFFY series is method-consistent but also spans resumed run segments.

| Method | Ckpt | Dev success | Normal success | Challenge success | Normal env score | Challenge env score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CA-B mask-only + LUFFY, warm99 | 199 | 64.91% | 52.38% | 29.26% | 0.7293 | 0.5128 |
| CA-B mask-only + LUFFY, warm99 | 249 | 59.65% | 55.36% | 32.37% | 0.7716 | 0.5554 |
| CA-B mask-only + LUFFY, warm99 | 299 | 82.46% | 58.33% | 33.57% | 0.7568 | 0.5272 |
| 0-1 success + LUFFY | 199 | 66.67% | not retained | not retained | not retained | not retained |
| 0-1 success + LUFFY | 299 | not in fixed table | 56.55% | 28.78% | 0.7562 | 0.5055 |

At checkpoint 299, warm-started CA-B was `+1.79` percentage points on normal
strict success and `+4.80` points on challenge strict success versus
0-1+LUFFY. This is useful evidence but not a clean causal comparison because
the CA-B series inherited a step-99 ROPD+LUFFY checkpoint. The from-scratch
table above removes that apparent advantage.

All rows in the fixed table had zero discard and zero retry. The recovered
0-1+LUFFY checkpoint-199 dev row also had zero discard/retry; it is preserved
from the formal native log because the curated TSV omitted it.

## Older Checkpoint Results

These results remain useful for historical context but are not the primary
comparison protocol.

| Run or method | Checkpoint | Split | Strict success | Env score | Caveat |
| --- | ---: | --- | ---: | ---: | --- |
| 0-1 success, no LUFFY | 199 | dev | 24.56% | not retained | dev only |
| 0-1 success, no LUFFY | 249 | dev | 24.56% | not retained | dev only |
| 0-1 success, no LUFFY | 299 | dev | 31.58% | not retained | dev only |
| 0-1+LUFFY | 299 | normal | 57.14% | 0.7752 | older native protocol |
| 0-1+LUFFY | 299 | challenge | 31.65% | 0.4997 | older native protocol |
| ROPD rubric-shaping + LUFFY | 299 | normal | 53.57% | 0.7171 | old eval RM/discard keys invalid |
| ROPD rubric-shaping + LUFFY | 299 | challenge | 32.13% | 0.4986 | old eval RM/discard keys invalid |
| CA bang-bang + LUFFY | 299 | dev | 56.14% | not retained | no challenge eval |
| CA bang-bang + LUFFY | 299 | normal | 42.86% | not retained | no challenge eval |
| Answer/process-50 + LUFFY | 299 | dev | 45.61% | not retained | verifier input polluted |
| Answer/process-50 + LUFFY | 299 | normal | 42.86% | not retained | verifier input polluted |
| Answer/process-50 + LUFFY | 299 | challenge | 20.14% | not retained | verifier input polluted |

The old ROPD+LUFFY eval emitted `discard_sample_rate=1.0` because the eval path
did not run the teacher-dependent ROPD RM. That key is not an infra-discard
measurement. Use only strict success and raw environment score from those rows.

## API Baseline Probe

These were custom full-dev API rollouts from 2026-07-06, not native checkpoint
evals. They are useful as model-capability and teacher-collection evidence, not
as direct training comparisons.

| Policy | N | Strict success | Mean AppWorld score | Notes |
| --- | ---: | ---: | ---: | --- |
| DeepSeek V4 Flash | 57 | 56.14% | 0.8709 | all 57 requests completed |
| DeepSeek V4 Pro | 57 | 59.65% | 0.8868 | all 57 requests completed |
| Qwen3.5-4B seed | 57 | 5.26% | 0.1101 | one request failed |
| Seed API partial probe | 14 | 14.29% | 0.3809 | incomplete; do not cite as full dev |

## Invalid Or Diagnostic-Only Evidence

- Early 8-checkpoint AppWorld GRPO sweep: only 57 rows despite being labeled
  `normal`; strict success was zero at every checkpoint while partial score rose
  from roughly `0.04` to `0.35` and mean turns collapsed from about `36` to
  `2.8`. This is reward-hacking evidence, not a formal normal result.
- Temperature-0 checkpoint-299 comparison: 34%-55% truncation, 5-97 retries,
  and some challenge discards. It is superseded by temperature 1. Source
  summary SHA256:
  `443ac2f4d5ca043e03445d97b97a6e0997980b703aae26b74701b63e25f5bc3e`.
- Answer/process-50 ROPD+LUFFY: student verifier input retained AppWorld official
  prompt/few-shot material. Keep only as a failed data-path ablation.
- TASA-GAE v2 had no accepted eval. The latest run stopped after checkpoint 49
  because its training curve was worse; see `docs/agent_env/tasa_handoff.md`.

## Portable Update Contract

When a new AppWorld result is accepted:

1. Verify full official split size, sampling config, checkpoint ancestry,
   discard/retry counts, and raw environment metrics.
2. Add the machine-readable summary to the normal eval index while the NAS is
   available.
3. Update this file with the compact result and explicit caveats before a
   cluster or trail is released.
4. Commit and push the skill update. Chat messages and W&B curves are not the
   durable handoff record.
