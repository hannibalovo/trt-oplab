# 面向 TensorRT 的算子优化 Workbench：设计提案 v0.1

## 1. 不是“做一个 Skill”，而是建设可验证的实验闭环

**目标**：让不同开发者/AI Agent 在相同实验契约下，从真实模型推理定位热点、提取算子、执行 standalone 优化、接入 TRT Plugin、检查融合/布局/Tactic、副作用和端到端收益；把“某个 kernel 快了”与“完整模型优化成立”严格区分。

**当前已知的具体痛点**：

1. 原生产 Engine 的 Attention 并非直接可调用的 `.cubin` API；捕获单层 Q/K/V/O 需要独立的调试图与 provenance。
2. ONNX Node、TRT Inspector Layer、GPU Kernel launch 不一一对应；重建 Engine 时 Layer 编号会变化，靠编号比较会误导。
3. 改一个 Plugin 节点可能改变前后或跨 Block 的融合、内部布局与 tactic，甚至造成整模型回退。
4. Kernel-only、完整算子调用、Engine、离线真实历史 chunk 和机械臂真实闭环是不同的性能/正确性边界。
5. 结果容易留在对话、日志或临时脚本里；负结果和不可复现的“优化成功”难被长期追踪。

## 2. 推荐的架构

```text
operator-optimization-workbench/
├── SKILL.md              # 八步导航、阶段门槛、禁止错误结论
├── references/            # 每一步的具体契约，不堆到 Skill 入口
├── helpers/               # 确定性脚本；输出机器可读证据
├── templates/             # Capture / Decision / Review 规范
├── examples/              # 真实 Engine Inspector 的结构分析示例
└── AGENTS.md              # agent 在仓库中遵守的工作纪律

项目适配层（另建，当前未虚构）：
    LingBot capture adapter / standalone harness / build scripts / offline replay
                  ↓ 输出/归档
    experiments/<id>/experiment.json + evidence/ + history.jsonl
                  ↓
               评审决策
```

**不建议立即做多级 Skill**：目前核心工作流具有共同的实验状态、模型/Engine 语义和验收逻辑。多个独立 Skill 会让跨阶段状态交接成为新问题。先复用已有 NCU、KernelWiki 等知识模块，等出现独立团队使用、不同触发规则与生命周期时再拆分。

## 3. 五个阶段、五个停机口

| 阶段 | 本阶段回答的问题 | 完成的证据 | 常见停止条件 |
|---|---|---|---|
| Baseline | 真正热点、成本和指标是什么？ | 真实 trace + Inspector 对应 + 工作负载指纹 | 只靠模型直觉选热点 |
| Capture | 采的是否是同一数学边界？ | Q/K/V/O、scale/mask、RoPE、KV 状态、调试与生产路径差异 | 参考结果来源不明 |
| Local | 候选核心及完整调用是否有潜力？ | 数值误差 + kernel-only + full-call 性能 | 数值不对，或布局开销吃掉收益 |
| Integration | 改图是否影响其他融合/实现？ | candidate Engine、Inspector 差异、Nsys kernel 轨迹 | 未解释的 reformat/tactic 变化或 NaN/Inf |
| End-to-end | 对目标工作负载是否值得保留？ | 相同真实历史回放、质量守卫、生成+commit 的 wall 性能 | 只有单 forward 或微基准加速 |

**关键设计原则：研究边界 ≠ 数据捕获边界 ≠ 最终 Plugin 边界。** 可以先单独重放 SDPA，最终根据成本模型决定只替换 MHA、融合 Layout+MHA，或保留 TRT 原生实现。

## 4. 一个容易被忽略的角度：优化的是“实验决策质量”

建议同时评估 workbench 本身的工程收益，而非用 Skill 文档多少来衡量：

- **证据追溯率**：每个性能结论是否能够定位输入、设备、Engine、脚本、源码哈希与原始日志？
- **误报率**：多少次 kernel-only 加速在 full-call/Engine 中回退？能否早期发现而非反复 Build？
- **定位成本**：从发现 regress 到锁定是数值边界、布局、融合、同步还是 Runtime，需要多少轮实验？
- **复用度**：新的 Attention/FFN、Video/Action、其他模型能复用多少测试契约，而不复制代码？
- **失败知识资产**：失败的候选是否保留最小重现和理由，避免不同人重复撞墙？

更深一层：Workbench 可以看成一套**带证据的决策状态机**，AI 负责提出假设/分析日志，人类/测试负责确认接受条件，脚本负责可重复检查，不应由 LLM 自行宣布结论已成立。

## 5. 现成可运行的产物（本包已实现）

- `helpers/layerinfo.py summary`：解析用户提供的 Action JSON（包括该文件特有的多余起始 `{`），输出各 LayerType/MHA 分布、shape/tactic/ONNX 锚点。
- `helpers/layerinfo.py diff`：基于 ONNX SDPA source anchor 及唯一节点归属做保守对照；**不会用不稳定的 Inspector 索引强制配对**；无法识别的 Plugin 命名报告 `UNRESOLVED`，不伪称融合已经变化。
- `helpers/experiment.py`：初始化实验、阶段状态、真实证据文件校验、数值门槛基本检验、追加历史记录。
- 示例数据：当前 Action Engine 中共 643 个 Inspector Layer、60 个 MHA，其中 30 个动态 KV 形状、30 个固定 KV 形状；同类 Tactic 的具体统计见 `examples/action_summary.json`。**这是结构分析，不是算子耗时。**

已在无 GPU 环境进行脚本单元测试；未运行 CUDA/TensorRT benchmark，也未自动改动公司的源码。

## 6. 应如何真正落地，而不是停留在“文档产出”

**第一阶段（选一条真实链路做纵向贯通）**：选 LingBot Action Layer 0 或 Video Layer 14 的一个固定真实输入状态。把同事已有的捕获方法挂接到 Capture Contract；保留生产/调试 Engine 和输入形状/Cache 元信息，再跑 standalone 验证。优先完成“一条真实案例走通”，不要先接所有模型。

**第二阶段（候选对照）**：接入现有 CUDA/Triton/cuDNN 候选，自动产出数值误差、kernel-only、full-call 的同条件对比。只在证据显示有潜力后讨论 Plugin 边界；候选结果写回实验目录而不是聊天上下文。

**第三阶段（集成差异和回归）**：候选 ONNX rewrite/build 后自动生成 LayerInfo diff；结合 Nsys 核实新增转换/融合变化；在已有 offline matched-history replay 上执行相同 episode/shape 的 generate+commit 比较。失败样本归档。

**暂不做**：自研 TensorRT Builder 规则、通用自动抽取所有 Engine 内部算子、全自动选择 Plugin 最优边界、复杂多 Agent 工作流编排。它们成本高，且目前缺少驱动这些能力的重复使用证据。

## 7. 给组长的方案判断题

> 我们想做的是一个“可复用的算子实验与集成框架”，还是一个“自动帮我们写 kernel 的 AI Agent”？

前者核心是契约、数据捕获、性能度量与判断门槛；后者必须建立在前者之上。若没有可重复且可信的算子评估闭环，自动生成更多候选只会更快地产生不可判定的结果。

**建议决定**：先批准前者的最小纵向切片；完成一个真实 Attention 候选从捕获到最终拒绝/接受的全过程后，再决定是否需要拆出独立 Skills 或多 Agent 并行。


## v0.3 补充：验证器落地

组内已有的 `operator_validation` 是一个可运行的 Local 正确性执行器，不应再复制一套。建议在原 Skill 的 `capture/local` 阶段对接该执行器，使用独立参考、冻结数据、误差标准、缓冲区写入和负例测试；在它之后单独建立生产 Shape 的 K/O 性能测量，并明确标识原 TRT kernel 无法同形状重放时的对照缺口。现阶段只有缩小样本的 CPU/Thor CUDA **报告** PASS，没有生产 Shape 加速或 Plugin 回插结果。原始数据始终保存在公司环境，通用 Skill 包只放文档、解析器和模板。
