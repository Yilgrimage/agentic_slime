# TASA-GAE Handoff

Status date: 2026-08-21

This note captures the reproducible implementation state. It does not claim a
positive training result.

## Implementation Anchors

- `f87007a0`: restore trajectory-global GAE over semantic state boundaries.
- `15688a38`: add strict-LOO coverage-adaptive teacher fill and flatten TASA
  metrics to `reward/tasa/*`.
- `9a2bb33a`: align formal pure, student-MC, and LUFFY group-size profiles.

Agent-env changes remain outside Slime core. Reward and credit semantics are
owned by `configs/agent_env/rewards/*.yaml`; topology and group lifecycle are
owned by run/train profiles.

## Current Estimators

Full TASA-GAE uses teacher-defined milestone states, strict leave-one-out
student outcomes, and coverage-adaptive interpolation. For a student and state:

```text
B = maximum other on-policy students in the group
N = other students that reached the state
V_MC = mean terminal outcome of those N students
V = (1 - N / B) * V_teacher + (N / B) * V_MC
```

The LUFFY teacher sample never enters `N`, `B`, or `V_MC`. Global GAE propagates
terminal evidence across milestone boundaries. Advantage scaling is RMS/std
only and must not mean-center or reverse signs.

The student-MC ablation uses the same teacher-defined semantic states but no
teacher value prior. It retains `tasa_min_peer_support`; an under-supported
state is not a value anchor.

## Canonical Profiles

| Variant | Run profile | On-policy students | Teacher slots | LOO peer budget |
| --- | --- | ---: | ---: | ---: |
| Full TASA | `appworld_qwen3_4b_ropd_tasa_gae_fullasync_4x8_gpt54.env` | 16 | 0 | 15 |
| Student-MC | `appworld_qwen3_4b_ropd_tasa_gae_student_mc_fullasync_4x8_gpt54.env` | 16 | 0 | 15 |
| TASA+LUFFY | `appworld_qwen3_4b_ropd_tasa_gae_luffy_fullasync_4x8_gpt54.env` | 15 | 1 | 14 |

All three use 16 total group slots, rollout batch size 8, and global batch size
128. LUFFY comparisons must use the same 15-student `replace_anchor` topology;
pure/student-MC comparisons must use the same 16-student topology.

## Required Evidence

Formal TASA profiles keep one bounded judge dump and one credit dump per step.
Do not remove these until the estimator is validated. At minimum inspect:

- strict-LOO peer-count histograms, separately excluding ROOT;
- peer coverage and teacher/MC weight by milestone depth;
- semantic state reuse and prerequisite validity;
- non-terminal milestone set/unset precision against raw traces;
- successful terminal positive and failed terminal negative advantage signs;
- token-segment alignment and train-token coverage.

The relevant W&B namespace is `reward/tasa/*`. Primary task outcomes remain
`appworld/success_rate` and `appworld/env_score_mean`; judge reward is not a
substitute for environment evaluation.

## Last Stopped Run

The stopped run is:

```text
/mnt/bn/jixf-nas-lq/yanjingyuan_runs_appworld_tasa_pure/Qwen3-4B_appworld_ropd_grpo/appworld-Qwen3-4B-grpo-fullasync-4x8-ropd-tasa-gae-v2-luffy-gpt54pool-fresh0to300-s50-20260821T064134
```

It exited normally at 2026-08-21 12:05:42 +08:00. Its last complete checkpoint
is iteration 49. The run started before the coverage-adaptive and group-size
commits above were loaded, so its curve is not a validation of the current
estimator. Its logs, checkpoint, judge dumps, and credit dumps were retained.

Offline audit of 54 judge groups from that run found an average of 4.13 schema
milestones. Student hit rates were approximately M1 65.6%, M2 26.0%, M3 16.1%,
with deeper states sparse. Prerequisite projection was mostly valid, but
non-terminal set/unset still needs manual precision checks; only four unset
events appeared and at least one regression was missed. This is evidence for
continued preflight auditing, not for forcing the judge to emit more states.

## Verification And Next Run

The implementation passed 80 agent-env tests on the control host. The omitted
runtime-resilience test imports `sglang_router`, which is unavailable in that
host runtime; rerun the full suite inside the materialized Slime runtime.

Before a long run on new compute:

1. Check out the pushed `agentic-env-backend` branch and record its commit.
2. Materialize Slime, W&B, AppWorld, data, and model artifacts on GPU nodes.
3. Run the full agent-env test suite inside the Slime runtime.
4. Launch a one-step, no-W&B preflight with formal group size and both dumps.
5. Audit LOO peer accounting, milestone events, state values, global GAE, and
   token placement from that real group.
6. Only then start a formal W&B run with explicit checkpoint retention.

Do not reuse the stopped run as a resume source for a clean method comparison.
