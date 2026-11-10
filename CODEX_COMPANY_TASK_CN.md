# 给公司侧 Codex 的任务：补齐真实优化工作流证据（优先以现有代码为基础）

请在现有 `lingbot-tensorrt` 仓库工作，不要重建一套验证器，也不要擅自覆盖现有冻结 fixture。你的首要任务是**审计并补齐当前 operator_validation 的 Capture→Local Numerical→Local Performance 证据链**；暂时不要改生产 ONNX、不要回插 Plugin、不要大规模跑实验。明确区分 Action 和 Video Engine；已提供的三个 JSON 是 **Video 层 0 Self-Attention** 的一次缩小样本验证，虽然 Layer 名可能同样叫 `_gemm_mha_v2_myl0_17`，绝不能仅根据名字与 Action Engine 混合。

## 第一阶段（先做，尽量不用 GPU）

1. 阅读已有 `operator_validation/{README.md,contract.json,manifest.json,prepare_example.py,reference.py,candidate_example.py,validate.py,test_validation.py}`、真实捕获 runner、原始实验 `full_depth_20261102` 相关脚本，以及近期 Runner/README。给出具体源码路径和关键行号。不要假定 capture/source JSON 的布尔值已经被你重新验证。
2. 确认 Debug Engine 的 Q/K/V/O 抓取方式及可重复运行的命令；比较原始与 Debug 的 Engine hash、Layer/Tactic 映射、匹配输入下外部输出。列出所有实际受影响/不确定的内部 fusion/layout。给出 Production/Debug Engine 的限制说明，不能仅凭外部输出逐位一致推断内部逐位一致。
3. 审计 `contract.json` 是否完整定义 scale、mask、causal、Q/K/V/O 物理 strides、精度、输出 buffer/stream、历史 KV 及 current KV 的语义。分别说明当前裁剪样本与生产 Shape 的差异。验证其 SHA256/冻结策略。请不要上传真实 safetensors 到 ChatGPT；只返回结构化数值统计及测试命令。
4. 审计并运行现有验证器和它的负例单元测试（前提是环境可用），保留现有输出，生成新的只追加报告：报告 status、candidate hash、device、数值门槛、输入修改/guard/扰动结果、实际运行命令；不能只复制旧的 PASS。

## 第二阶段（允许离峰 GPU，先测一条真实路径）

5. 选择 Video Self-Attention 的**原始完整 Shape**作为第一个性能工作负载，不使用 `[1,32,128]` 裁剪样本冒充原始 TRT baseline。参考已报告的 Full Video `Q[48,512,128]`、`K[48,128,8672]`、`V[48,8672,128]`，在现场核实真实 profile/stride、mask/scale、现有输出和缓存状态。若内存或成本很高，先测一条代表性完整路径，记录限制；保留完整 K/V。
6. 按相同输入/环境测候选 kernel-only K，以及包含必需布局/pack/dispatch 的 full-call O；评估原始 TRT 是否能**同边界、同形状**对照。如不能，从 Nsys/Engine trace 提供可解释的成本分解，但将 `baseline_kernel_only` 标记 unavailable，**不要**把整个 Engine Layer 计时直接等同于一个独立可调用原 TRT Kernel。收集热身、重复、p50/p90、是否 CUDA Graph、同步点、stream、GPU 背景负载、版本/构建 hash；数值通过才进入性能分析。
7. 至少给出一个短/中/长历史 KV 状态的覆盖计划（可先只执行完整长历史点）；涉及另外的 Action 模式时请作为独立 workload 单列，而不是跨 Engine 混算。同事当前演示 PyTorch Math SDPA 不是优化候选，不要据此宣称速度提升。
8. 输出一个符合 `templates/benchmark_report.json` 思想的结构化报告（可扩展）：明确每条 latency 的测量边界、实际采样量、对照的可比性；记录待补的真实 Plugin + Engine + E2E 证据，但**本任务暂不执行回插**。

## 补充决策门：正确性基线与 Local / Plugin 边界

本节用于避免把“单算子能算对”“局部 kernel 更快”和“换进 TensorRT 后整体更好”当成同一结论。请在审计报告中按下列层级标记证据，不要只写一个笼统的 baseline 或 PASS：

| 代号 | 基线/证据 | 能回答什么 | 不能据此声称什么 |
|---|---|---|---|
| M | 独立数学参考（FP32 或适当高精度，语义与舍入写清） | 候选是否符合目标算子公式 | 生产 TensorRT 的内部输出或性能 |
| D | 固定输入下 Debug Engine 抓取的边界输入/输出 | 候选是否接近这次 debug capture 的行为 | Debug 内部值必然等于 Production 内部值 |
| P | 原始 Production Engine 的同输入外部输出及任务状态 | 改动是否影响生产可见行为 | 某个 Inspector Layer 就是一条可单独调用的 TRT kernel |
| K | 同边界、同形状、同输入且可重复调用的原 TRT kernel | Local K 的直接 kernel 对照 | 完整算子调用、集成 Engine 或端到端收益 |

K 若无法真实重放，写 `unavailable` 并说明原因；不得用整层/整 Engine 时间、PyTorch SDPA、Debug Engine 中相似名字的 layer 或猜测的 tactic 代替。D 与 P 的 engine hash、构建配置、profile、layer/tactic 映射和同输入外部输出对照要分别记录；可观测性改图可能改变 fusion、layout、舍入或 tactic，因此即使外部输出逐位相同，也不能推导内部张量逐位相同。结论要注明是“源码确认”“ONNX 图确认”“Engine 结构确认”“同输入实测”还是“推断”。

在评估候选前冻结每个 workload 的语义和数值门槛：算子/融合边界、Q/K/V/O 的 shape、物理 stride、dtype、layout/alignment、scale、mask/causal、RoPE/norm 状态、历史与 current KV 含义、cache 状态、输出写入/别名规则、workspace 和 stream。原 TRT 输出用于行为对照，独立数学参考用于语义诊断；分别报告有限性、逐元素门槛（例如 `abs(candidate-reference) <= atol + rtol*abs(reference)`）、超限元素数、MAE/RMSE/max error，以及输入保护、输出完整写入、重复性和适用的扰动/负例。先测参考相对原 TRT 的误差和最终 dtype 舍入，再冻结阈值；不得在看到候选结果后调阈值，也不得把某个 layer/shape 的门槛自动推广到其他层或任务质量门槛。

### 何时留在 Local，何时评估 TensorRT Plugin

- **留在 Local/拒绝候选：** Local 是独立验证与测量方式，不等于生产执行路径。若候选只在 kernel-only K 上更快，但包含真实 layout/pack/copy/dispatch/sync 的完整调用 O 没有优势，就不要因 K 的结果申请回插；先定位成本，尝试更合适的边界或拒绝该候选。若项目运行时本来就能在 TensorRT 外合法调用该实现，且 O 和目标应用指标有可重复收益，可继续评估 Local/runtime 路径，不必为了“进 TensorRT”强行做 Plugin。
- **提出 Local → Plugin 集成评估：** 只有 Local 数值/行为门槛通过，输入输出契约与目标生产 workload 对齐，且 O 显示有值得保留的空间后，才进入 Plugin 设计。需要 Plugin 的典型原因是目标计算必须在 TRT 图内替换；Local 独立调用造成无法接受的数据物化/格式转换/额外 launch 或同步；需要在 TRT 持有的 layout、buffer 生命周期和 stream 上直接消费/产出；或希望把相邻算子融合以消除已测出的开销。报告必须写清拟替换的最小/完整子图、边界张量、预计消掉的成本、会损失的 TRT 原生 fusion/tactic，以及为何普通 TRT 图无法表达同一优化。
- **不得把 Local PASS 当成 Plugin PASS：** Plugin 改变图切分后会重新选择 fusion、layout、precision 和 tactic。后续集成必须单独用隔离 ONNX + Plugin build/runtime 注册，核对输入输出语义与 shape/type/format 支持，重建匹配 profile 的 Engine；再做固定输入的局部/整 Engine 数值验证、LayerInfo 结构差异、Nsys 实际 launch/copy/sync 检查、带 cache commit/discard 的多步 replay 和任务质量/E2E 延迟。任一层级失败都回退到原 Production Engine，并记录失败边界；不能仅凭单层通过扩大到 30 层或全任务。
- **本任务停止点：** 本轮只审计 Capture、Local Numerical、Local Performance，并将后续 Plugin/E2E 需要的输入、决策理由和验收项列成待办；不得因此改 ONNX、实现/回插 Plugin、覆盖生产 Engine 或宣称集成通过。证据不足时标为 `blocked`/`unavailable`，保留当前生产路径。

## 交付

- 公司仓库内保存：审计 MD、采集/运行命令、非覆盖的新 JSON 报告、Benchmark 报告与必要脚本（如已有脚本能复用，不额外造框架）。
- 发回 ChatGPT：**脱敏的** Shape/stride/dtype/cache/mask/scale、调试抓取对齐方法、结果摘要和性能聚合统计、关键脚本位置、失败与未测项目。原始 safetensors、权重、Engine 二进制、敏感服务配置与输入内容留在公司。
- 若无法运行 GPU、找不到生产等价数据或原始 TRT kernel 不能重放，明确说明 blocker 并停在已有证据，不要伪造 PASS/性能数字。
