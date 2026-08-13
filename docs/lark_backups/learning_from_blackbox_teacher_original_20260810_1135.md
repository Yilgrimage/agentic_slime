# Learning from a black-box teacher in on-policy optimization

## 1.Intro

如何利用已有黑盒教师模型的轨迹去增强我们的agent，

我们首选appworld和webshop用于idea验证，每当方法论有进展后在codex训练上进行尝试，分析scale上去后是否依旧有效



## 2.related work

SFT通常作为标准训练方法，

GAD要训critc且效果不佳；

ROPD如何如何，但是其强调reasoning任务并没有直接作用于agent任务的解法，本质是rubric-based RL；

LUFFY提出teacher数据作为一条组内数据参与adv计算并通过独特的loss项将teacher数据也纳入训练。



## 3.Method

### Synthetic Teacher Data

我们通过多次采样，注入经验反思来采样出在环境中得到满分（或接近满分）teacher数据，teacher目前使用deepseek v4 flash。

### Rubric-based reward

参考ropd，我们将一个group的student和对应的teacher数据交给llm去生成rubric，再将rubric和轨迹送给另一个llm进行盲评

### Offline teacher in on-policy optimization

LUFFY提出将离线的teacher数据也参与训练可以提升训练效果，我们注意到，

（补全公式，不管是teacher的loss还是teacher参与组内优势计算）



## 4.Experiment

### source

实验目前均使用**32卡A100**进行，部分teacher数据在H卡上部署deepseek flash v4来采集（因为A100没有推理框架支持deepseek v4），不干预主线训练探索进展。

早期实验使用qwen 27b作为teacher模型，需要占用一个节点，训练topo通常为2 node rollout + 1 node actor的全异步训练，本意是希望验证这种方式无需极强judge模型，但是拉低训练速度且有偶发judge误判现象，因此近期训练已替换为3 node rollout + 1 node actor，teacher使用gpt-5.4作为替代。

### benchmark & metrics

**Appworld**

涉及xxx类型任务，例如xxx；环境针对每个任务提供一些规则验证来提供一个0到1的环境分数，这个环境官方口径认为**只有分数为满分才被记录为success**

**Webshop**

涉及xxx类型任务，例如xxx；环境针对每个任务提供一些规则验证来提供一个0到1的环境分数，这个环境官方口径认为**只要有非0分数就被记录为success**

目前在集中尝试appworld，因为webshop的teacher数据合成耽误比较久，虽然已有一版ready，但是之前集中在appworld上进行。早期从官方提供的xxx数据进行训练有效的结果属于误报，当时指标对比失误，将llm judge的分数和env实际分数进行比较，误以为有明显涨点，但实际上因为方法尚未ready且teacher数据存在问题，没有超越使用env reward的webshop。

### baseline

选取如下方式作为参考baseline setting：

1. **直接使用环境提供的reward信号进行GRPO；**

这个方式在webshop上是可用的，可以通过grpo很容易的训练出效果，但是对于appworld而言，直接grpo会出现hacking，因为appworld有很多是防止agent破坏性操作的，有时候什么也不做也会拿一些分，直接训练会策略退化



1. **只使用0-1二元success信号进行GRPO；**

只当有严格成功的case时才记录为非0奖励，其他时间均为0 reward，虽然训练速度慢，但是在appworld上能带来稳定上涨。



1. **0-1二元success信号 + luffy**

目前最强基线，需要超过这个方法才能证明方法有效

后续不少尝试性训练没有进行eval，通过观察训练时success rate粗判断效果

### exp1 ropd，生成process rubric和answer的rubric

暂时不使用process rubric生成的分数进行训练，这个环节是为了后续使用process reward进行信用分配做准备留入口，暂时先用answer的rubric对应的reward分数作为训练。能够训练出来，但是效果远弱于0-1 reward GRPO

### exp2 ropd，生成process rubric和answer core和answer support的rubric，但是强制answer core占比50

观察发现大量和核心结果无关的rubric分掉了大量reward配额，真正对任务成功强相关的信号占比太弱。调整配比后有所提升，xxx

### exp3 ropd，生成process rubric和answer core，用process + core来作为训练信号

分析发现，类似codex任务一样去给answer丰富的rubric意义不大，直接利用process reward应该是更好的策略，但是训练曲线没有明显变化，和前述setting并没有太大差异

### exp4 ropd组合env reward，同时接入luffy

不再依赖模型自己判断answer的质量，而是直接接入env reward信号。具体而言，xxx

效果有提升，也是目前的实现方式，但是没有证明超过0-1 reward，目前结论是将process rubric得到的分数堆叠在outcome reward，即使理论上提供了更细致的奖励信号但是对于本身有强验证器的任务而言并不能有效提升效果。



下一步准备变化形式，让llm judge得到精确的step level的奖惩信号，而不是直接堆叠到最后的outcome上去。
