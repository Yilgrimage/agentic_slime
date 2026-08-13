# Learning from a black-box teacher in on-policy optimization

## 1.Intro

如何利用已有黑盒教师模型的轨迹去增强我们的agent。

我们首选 AppWorld 和 WebShop 用于 idea 验证；每当方法论有进展后，再迁移到 Codex / OpenClaw 这类更重的 agent 任务，分析 scale 上去后是否依旧有效。

目前观察到的关键点是：0-1 verifier reward + LUFFY 是很强的 baseline；scalar ROPD 没有稳定超过它。下一步应把 teacher / judge 信息落到 step 或 segment level credit assignment，而不是继续堆 trajectory-level scalar reward。

## 2.related work

SFT 通常作为标准训练方法，但它只模仿 teacher 轨迹，不利用 on-policy exploration。

GAD 要训 critic，链路更重，且已有结果里效果并不理想。

ROPD 本质是 rubric-based RL。原始方法主要面向 reasoning 任务；直接搬到 agent 任务时，如果只把 rubric 压成一个 trajectory-level reward，就没有真正解决多轮任务里的信用分配。

LUFFY 提出 teacher 数据作为一条组内数据参与 advantage 计算，并通过独特 loss 项将 teacher 数据纳入训练。我们当前实验里，LUFFY 是最稳定有效的增强项。

## 3. Current Method

### 3.1 Synthetic Teacher Data

我们通过多次采样和失败重试生成 teacher traces。当前主要使用：

- AppWorld teacher：DeepSeek V4 Flash，多轮 rerun，保留严格成功或接近满分的 trace；
- WebShop teacher：已有一版 DeepSeek/GPT 采样数据，但 teacher quality 和 task alignment 仍需复查；
- 未来 Codex/OpenClaw 类任务：计划使用更强 teacher 或人工/强模型筛选。

Teacher 数据采集原则：

1. 原始 trace 尽量完整保存，包括 task id、完整 tool call、tool response、assistant message、环境 score/success。
2. 训练时再通过统一 trace rendering/compaction 决定是否去掉 reasoning、system prompt、tool response 中间部分等。
3. teacher 和 student 进入 judge 的 trace 格式必须一致，避免 teacher 因格式优势或冗余字段被误判。
4. teacher 数据必须与训练 task id 对齐；缺 teacher 的 task 不应静默 fallback 到错误样本。

### 3.2 Offline teacher in on-policy optimization

LUFFY 是这套方法里最核心的训练模块之一，不应该只作为 baseline 名字出现。

我们对每个 task 额外准备一条 teacher trace，并把它作为 group 内的 off-policy teacher sample 接入训练。teacher sample 不来自当前 policy rollout，但会和 student samples 一起参与组内 reward / advantage 计算。

训练时，teacher action tokens 通过 `off_policy_loss_mask` 标记，只在 teacher loss 上生效；student rollout 仍走正常 GRPO loss。

简化写法：

```text
student loss = - A_student * log pi_theta(student_action)
teacher loss = - c * A_teacher * reshape(pi_theta) * log pi_theta(teacher_action)
```

因此 teacher trace 在这里有两重作用：一是参与组内 advantage 标定，二是通过额外 token loss 直接给出策略监督。

### 3.3 Current Scalar ROPD

当前 ROPD 实现流程：

1. 对同一 task 的 teacher trace 和 group 内 student traces 做统一压缩渲染；
2. rubricator 生成 rubric/behavior criteria；
3. judge 对 student traces 打分；
4. reward profile 将 judge score/env success/teacher signal 合成为 sample-level reward；
5. GRPO 将这个 sample reward 转成 group-normalized advantage。

当前问题：

```text
judge process information -> one scalar reward -> broadcast to all action tokens
```

这不是充分的 credit assignment。

## 4. Next direction: credit assignment

当前 scalar ROPD 的主要问题是：judge 看到过程信息，但最后仍压成一个 trajectory-level reward。下一步先调研和设计如何把 teacher / judge 信息转成更局部的 credit assignment，再决定是否写成正式方法。

重点先看三类路线：value-based token/step advantage、LLM judge 标注局部好坏行为、以及 teacher trace 直接参与训练的 LUFFY 类方法。

## 5. Experimental Setup

### 5.1 Compute

当前 A100 集群：4 nodes × 8 GPUs = 32 × A100。

典型配置：

| Setting | Actor | Rollout | Aux/Judge | 用途 |
|---|---:|---:|---:|---|
| 3 rollout + 1 actor | 1 node / 8 GPU | 3 nodes / 24 GPU | external GPT API | 当前 AppWorld/WebShop 主线 |
| 2 rollout + 1 actor + 1 aux | 1 node / 8 GPU | 2 nodes / 16 GPU | 1 node local Qwen 27B | 早期 ROPD/Tau 测试 |
| 4 node fullasync no aux | 1 node actor + 3 node rollout | 24 rollout GPU | none | 0-1 reward / LUFFY |

关键训练配置：

- Base model：Qwen3-4B / Qwen3.5-4B depending on task；
- GRPO full-async；
- `global_batch_size=128`；
- `rollout_batch_size=16`，`n_samples_per_prompt=8`；
- `USE_ROLLOUT_LOGPROBS=1`；
- dynamic sampling filter enabled；
- GLM-style infra padding enabled；
- AppWorld context/response：`32768 / 4096`；
- WebShop context/response：`16384 / 4096`；
- ALFWorld context/response：`16384 / 2048`。

### 5.2 Benchmarks and Metrics

#### AppWorld

AppWorld 是长程工具调用任务，环境提供 0-1 的 strict success 和 0-1 区间的 env score。官方 success 口径是满分才成功。

主要指标：

- `success_rate`：strict success；
- `env_score_mean`：环境连续分数；
- `truncated_ratio`、`format_error_rate`：行为质量与训练健康度；
- 对 ROPD：`reward/ropd/*` 只作为 reward-quality/debug 信号，不直接等同环境成功率。

#### WebShop

WebShop 是商品搜索/选择任务，环境返回连续 reward。很多文献会把非零 reward 当 success，但我们主要汇报 env score/reward，避免不同口径混淆。

#### ALFWorld

ALFWorld 是 text-action household task。当前 Qwen3-4B GRPO 已能达到很高验证分数，是验证 agentic Slime 训练链路可靠性的参考任务。

#### Tau2

Tau2 需要 user simulator。早期使用本地 Qwen 27B 作为 user sim，后续发现 user sim 能力和速度会显著影响结果，因此用 DeepSeek Flash/Pro 做了对照评估。

### 5.3 Baseline settings

选取如下方式作为参考 baseline setting：

1. **直接使用环境提供的 reward 信号进行 GRPO。**

   WebShop 上可用；AppWorld 上容易 hacking，因为部分任务里什么也不做也能拿到非零环境分。

2. **只使用 0-1 二元 success 信号进行 GRPO。**

   只有严格成功 case 给非零奖励。训练信号更稀疏，但 AppWorld 上更稳定。

3. **0-1 二元 success 信号 + LUFFY。**

   当前最强 baseline，需要超过它才能证明新方法有效。

4. **ROPD / ROPD + LUFFY。**

   用 teacher trace 和 judge 生成额外奖励信号；目前 scalar reward 版本没有稳定超过 0-1 + LUFFY，因此下一步转向 step/segment-level credit assignment。

## 6. Existing Results

### 6.1 ALFWorld: GRPO is Strong

Qwen3-4B，full-async 3×8，300 steps，strict prompt，native full eval：

| Split | N | Eval reward / 10 | Approx success |
|---|---:|---:|---:|
| valid_train | 200 | 9.300 | 93.0% |
| valid_seen | 140 | 9.135 | 91.4% |
| valid_unseen | 134 | 8.875 | 88.8% |

训练后期 10 step rollout raw reward：`7.73 / 10`。最终 eval 高于训练曲线，说明训练时动态采样/分布和 full eval split 不完全同口径。

结论：ALFWorld 已不是主要瓶颈，继续做 scalar ROPD 意义有限。

### 6.2 WebShop: GRPO Works, ROPD Data Quality Still Weak

Qwen3-4B WebShop native eval：

| Method | Final ckpt | Eval score / 10 | Truncated ratio | Notes |
|---|---:|---:|---:|---|
| GRPO thinking, no KL | 299 | 6.340 | 0.0215 | 当前较强 WebShop baseline |
| GRPO no-thinking + KL=0.001 | 299 | 5.842 | 0.1146 | 与 thinking/KL 共同变化，非纯 thinking 消融 |
| ROPD from IL/Qwen 27B | 299 | 6.034 | 0.0950 | 未超过 strong GRPO；teacher data/task alignment 质量存疑 |

训练后期 10 step rollout raw reward：

| Method | Last-10 rollout raw reward / 10 |
|---|---:|
| thinking, no KL | 6.07 |
| no-thinking + KL=0.001 | 7.08 |
| ROPD IL/Qwen27 | 暂未从日志稳定抽取 |

注意：WebShop 训练曲线与 native eval 出现过不一致，需要优先相信 native eval。此前 WebShop ROPD 有误判：将 LLM judge 分数和 env actual score 混合比较，导致误以为显著涨点。

### 6.3 AppWorld: Direct Env Score GRPO Can Hack

早期 AppWorld 直接使用环境连续分数训练，normal split checkpoint sweep 显示 strict success 长期为 0。观察认为模型容易利用 partial score / non-destructive behavior，而不是完成任务。

因此 AppWorld 主线改为 strict 0-1 success reward。

### 6.4 AppWorld: 0-1 + LUFFY vs ROPD + LUFFY

Full test 使用 `test_normal=168` 和 `test_challenge=417`，合计 585 条。

| Method | test_normal success | test_challenge success | full success | full env_score_mean |
|---|---:|---:|---:|---:|
| 0-1 + LUFFY | 57.14% | 31.65% | 38.97% | 0.579 |
| ROPD scalar + LUFFY, GPT-5.4 | 53.57% | 32.13% | 38.29% | 0.561 |

训练后期 10 step rollout raw reward：

| Method | Last-10 rollout raw reward | Interpretation |
|---|---:|---|
| 0-1 + LUFFY | 0.786 | strict success reward scale |
| ROPD scalar + LUFFY | 8.441 | ROPD train reward scale，不可和 0-1 直接比较 |

Dev eval ckpt 299：

| Method | Dev success | Dev env_score_mean |
|---|---:|---:|
| 0-1 + LUFFY | 64.91% | 0.748 |
| ROPD scalar + LUFFY | 68.42% | 0.774 |

Interpretation：ROPD 在 dev 上看起来更好，但 full test 没有稳定超过 0-1 + LUFFY。当前结论是：**ROPD scalar shaping 尚未证明有效优于 verifier reward + LUFFY。**

### 6.5 Tau2: User Simulator Matters

Qwen3.5-4B Tau2 eval 100 tasks：

| Policy | User simulator | Score rate | Notes |
|---|---|---:|---|
| Qwen3.5 base | DeepSeek V4 Flash | 0.63 | API usersim |
| ckpt049 | DeepSeek V4 Flash | 0.73 | improves |
| ckpt099 | DeepSeek V4 Flash | 0.830 | best in sweep |
| ckpt149 | DeepSeek V4 Flash | 0.769 | starts dropping |
| ckpt199 | DeepSeek V4 Flash | 0.332 | collapse/overtraining or instability |
| DeepSeek V4 Flash policy | DeepSeek V4 Flash usersim | 0.79 | API baseline |
| DeepSeek V4 Pro policy | DeepSeek V4 Flash usersim | 0.83 | API baseline |
| Qwen ckpt099 | Qwen 27B usersim | 0.508 | local usersim weaker |

结论：Tau2 对 user sim 非常敏感。用 Qwen 27B usersim 训练/评估会显著低估策略能力，也可能训练到偏离真实用户分布的策略。

## 7. What We Have Learned

### 7.1 Infrastructure Lessons

1. Full-async GRPO 必须保留 rollout-time logprobs：`USE_ROLLOUT_LOGPROBS=1`。
2. Dropout 必须关闭，否则 PPO/GRPO KL 与 ratio 异常。
3. rollout 并发、env pool、rbs/gbs 要对齐，避免额外 staleness。
4. Formal eval 必须 full split；smoke 或 100 条 eval 不能作为最终结论。
5. ROPD/eval 指标必须区分：
   - training reward / judge score；
   - raw environment score；
   - strict success。

### 7.2 Algorithmic Lessons

1. Reliable verifier reward 仍然是 AppWorld/WebShop 这类任务最强的 outcome signal。
2. LUFFY-style teacher token loss 明显有效，像一个 on-policy context 下的动态 SFT/teacher-policy injection。
3. Scalar ROPD 的问题是把过程信息压成一个最终分数，没有解决内部 credit assignment。
4. 下一阶段应让 ROPD 输出 behavior-level reward/penalty mask，并映射到 action segment 的 token advantage。

## 8. Next Plan

### 8.1 Main Next Experiment: ROPD-CA + LUFFY

实现 behavior-guided credit assignment：

1. GPT stage 1：从 teacher trace 抽取 positive/negative behaviors。
2. GPT stage 2：把 behaviors 对齐到 student trace 的 `segment_id`。
3. 训练中构造：

```text
A_segment = A_outcome + beta * A_process_segment
```

4. dynamic filter 允许全失败但 process mask 有差异的 group 保留。
5. 对比：
   - 0-1 + LUFFY；
   - scalar ROPD + LUFFY；
   - ROPD-CA + LUFFY。

### 8.2 Expected Outcome

我们不预期 ROPD-CA 在所有 split 上大幅超过 0-1 + LUFFY。更合理的目标是：

- 在 AppWorld challenge split 上提升泛化；
- 在训练早期提高全失败 group 的有效样本比例；
- 降低 teacher_below_student / judge misranking；
- 证明 process signal 只有落到 segment-level credit assignment 后才有价值。

### 8.3 Proposed Ablations

| Ablation | Purpose |
|---|---|
| 0-1 reward | verifier-only baseline |
| 0-1 + LUFFY | current strongest baseline |
| scalar ROPD + LUFFY | current ROPD baseline |
| ROPD-CA + LUFFY | proposed method |
| ROPD-CA without LUFFY | isolate process credit from teacher token loss |
| positive mask only / negative mask only | test whether penalty is more reliable than reward |
| beta sweep 0.05/0.1/0.2 | tune process signal strength |

### 8.4 Paper Story

Possible thesis:

**Black-box teacher traces can improve on-policy LLM agent optimization in two complementary ways: direct off-policy teacher-token learning and teacher-guided segment-level credit assignment. While scalar rubric reward does not reliably outperform verifier reward, behavior-level credit assignment turns teacher comparisons into actionable local advantages.**

## 9. Open Questions

1. GPT behavior alignment 是否足够稳定，能否可靠输出 `segment_id`？
2. `A_process` 应该在 group 内归一，还是按 task/rubric 全局归一？
3. 对失败轨迹中的 positive behavior 应该给多大权重？
4. fatal error 后的 action 是否应全部 mask 掉 positive credit？
5. ROPD-CA 在没有 verifier 的 OpenClaw/Codex 场景中是否更有价值？

## 10. Immediate Action Items

1. 实现 ROPD behavior guide / behavior alignment schema。
2. 在 rollout metadata 中确保每个 assistant action segment 有稳定 `segment_id` 和 token span。
3. 实现 custom advantage function，将 process mask 转成 segment-level advantage。
4. 修改 dynamic sampling filter，保留 process-mask 有方差的全失败 group。
5. 在 AppWorld 上跑 200-300 step ROPD-CA + LUFFY。
6. 若 AppWorld 有增益，再迁移 WebShop；若仍无增益，优先转向 OpenClaw/Codex 这类 verifier 更弱或 answer 更开放的任务。
