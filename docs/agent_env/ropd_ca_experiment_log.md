# ROPD Credit Assignment Experiment Log

This note is an operational record for AppWorld/WebShop ROPD-CA experiments.
Keep it concise: record the evidence, then decide the next run.

## Current Anchor

- Primary task metric: env success rate and raw env score, not ROPD judge score.
- Useful training-curve proxy: last 10 to 20 train steps of `<env>/success_rate`
  and `<env>/env_score_mean`.
- LUFFY is a separate teacher-token loss. It can dominate learning, so CA runs
  must be compared against the same LUFFY setting.
- Dynamic sampling should be off while debugging CA. Zero scalar reward does
  not mean zero process credit.

## CA Schemes

- A, `outcome_plus_process`:
  `A_token = A_outcome + beta * A_process`.
  This is direct and strong, but `A_process` must be meaningful as an
  advantage-like correction.
- B, `outcome_reweight`:
  `A_token = A_outcome * clamp(1 + beta * sign(A_outcome) * A_process)`.
  This preserves the outcome sign and uses process credit to redistribute
  magnitude inside the trajectory.
- C, `segment_reward_group_turn_norm`:
  Build turn rewards from `R_outcome + beta * process_credit`, normalize turns
  within the prompt group, then expand to tokens. This is closer to
  REINFORCE++-style centering than same-prefix GRPO.

All three schemes should share the same upstream trace compression, GPT
rubric/judge output, step mark construction, and token segment expansion.

## Process Mark Semantics

- Milestone hit: strong positive mark.
- Predecessor: weaker positive credit for actions before a milestone.
- Negative hit: explicit bad-action mark.
- Direct good/bad marks outrank predecessor marks. If direct good and direct
  bad hit the same step, their values add.
- Predecessor decay is now explicit. With `predecessor_reward=0.5` and
  `predecessor_decay=0.5`, the predecessor sequence is `0.5, 0.25, 0.125, ...`
  as distance from the milestone increases.

## Runs To Compare

- Plain ROPD+LUFFY:
  `appworld-Qwen3-4B-grpo-fullasync-4x8-ropd-luffy-rubricshaping-gpt54pool-resume199to300-save50-20260809T024056Z`
- 0-1 reward + LUFFY:
  compare under `Qwen3-4B_appworld_grpo` with raw env score/success.
- CA-B beta=0.5:
  `appworld-Qwen3-4B-grpo-fullasync-4x8-ropd-ca-b-luffy-gpt54pool-warm99to300-s50-schedfix-20260812T105442Z`
- CA-B beta=2.0 predecessor decay=0.5:
  `appworld-Qwen3-4B-grpo-fullasync-4x8-ropd-ca-b-beta2-decay05-luffy-gpt54pool-warm99to300-s50-20260812T153651Z`

## Current Hypotheses

- If CA-B remains indistinguishable from plain ROPD+LUFFY, likely causes are:
  process reweighting is still too weak after normalization, LUFFY dominates the
  gradient, or GPT masks are mostly generic positives rather than localized
  credit.
- If CA-B improves train success but not eval, process marks may be rewarding
  scaffold compliance or easy local progress instead of durable task completion.
- If many groups show `reward_error`, inspect ROPD verifier dumps before changing
  CA math. Do not treat reward infra failures as algorithm conclusions.
- For AppWorld 0-1 outcome reward, most early student trajectories receive zero
  scalar reward. In a group with rare successes, those zero-reward trajectories
  get negative outcome advantage. This negative pressure is useful: it suppresses
  common failed behaviors. A strong positive process term on failed trajectories
  can accidentally erase that pressure if GPT marks too many local steps as
  "good enough". Treat large beta values as risky unless masks are very sparse
  and localized.

## Decision Rules

1. At roughly step 150, compare the last 20 train steps of CA-B beta=2.0 against
   plain ROPD+LUFFY and CA-B beta=0.5 on env success and env score.
2. Inspect credit dumps from several kept groups:
   milestone count, predecessor distance distribution, negative coverage, and
   whether marked steps match actual useful or harmful actions.
3. If CA-B beta=2.0 is still flat versus plain ROPD+LUFFY and dumps look
   reasonable, stop this branch and try C.
4. If dumps show noisy or over-broad marks, fix the judge/mask protocol before
   launching another long run.
5. If C also fails to move env metrics after a fair window, shift WebShop to the
   main CA tuning task because it is faster and has higher-quality env reward.

## 2026-08-13 CA-B beta=2.0 Decay=0.5

Run:
`appworld-Qwen3-4B-grpo-fullasync-4x8-ropd-ca-b-beta2-decay05-luffy-gpt54pool-warm99to300-s50-20260812T153651Z`

Initial checks:

- Resumed from `iter_0000099` and reached step 107 by 2026-08-13 00:00.
- Average step time from steps 100 to 107: about 2.28 min/step.
- W&B project: `Qwen3-4B_appworld_ropd_grpo`.
- `ENABLE_DYNAMIC_SAMPLING_FILTER=0`, `SAVE_INTERVAL=50`,
  `AGENT_ENV_MAX_CHECKPOINTS=0`.
- Credit dump records `advantage_mode=outcome_reweight`, `beta=2.0`,
  `predecessor_decay=0.5`.
- First credit dump sample shows expected predecessor values:
  `0.5, 0.25, 0.125, ...`, plus direct `2.0` milestone and `-1.0` negative
  marks.
- Early credit dump summary from 198 rows:
  894 marked steps, 714 positive, 180 negative. Value counts include
  `2.0`, `0.5`, `0.25`, `0.125`, and `-1.0`.
- Watch item: several rollout groups were skipped with
  `reason_counts={'': 8, 'reward_error': 8}`. Before drawing an algorithm
  conclusion, inspect verifier dumps and reward errors.
- If early W&B curves show degradation versus plain ROPD+LUFFY, a likely
  mechanism is not gradient absence but over-protection of failed trajectories:
  CA-B with `beta=2.0` can shrink negative outcome advantages toward zero on
  many positively marked process tokens.

Estimated checkpoints:

- Step 150: around 2026-08-13 02:00.
- Step 200: around 2026-08-13 04:00.
- Step 300: around 2026-08-13 07:30 to 08:00 if speed remains stable.
