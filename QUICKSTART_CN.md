# 中文上手卡片

## 先记住三句话

1. **算子提取** = 在明确数学边界下，保存真实输入和参考输出，脱离整模型重放。
2. **Plugin 集成** = 选择新的计算图边界，把候选实现交给 TensorRT 执行，不等于算子提取本身。
3. **局部速度 ≠ 整体收益**：分别看 kernel-only、含转换的 full-call、Engine、generate+commit 真实回放。

## 层级

```text
SKILL.md       导航/必要性/阶段门槛
  references/  捕获、实验、集成、评审等详细流程
  helpers/     解析/差异/状态/证据校验脚本
  templates/   数据与决策契约
  examples/    已经验证过的实际文件结构输出
```

## 可以立即跑的操作

在包根目录：

```bash
# 你上传的 Action Engine Inspector 文件可直接作为输入
python helpers/layerinfo.py summary /path/to/action_engine_layerinfo.json \
  --out /tmp/action_summary.json --markdown /tmp/action_summary.md

# 将来 Plugin 候选编译后得到另一个 LayerInfo，用相同语义锚点比较
python helpers/layerinfo.py diff /path/to/action_engine_layerinfo.json \
  /path/to/candidate.layerinfo.json --out /tmp/layer_diff.json

# 创建一个有状态的实验；不会自动启动 GPU
python helpers/experiment.py init --out /tmp/exp-001 \
  --hypothesis "减少 History KV 布局准备开销" \
  --target "Action Self-Attention, B2, Sq32"
python helpers/experiment.py status /tmp/exp-001
```

注意 `experiment.json` 里的 `workload_contract` 要由实际运行环境填写；
`stage pass` 需要真实存在的证据文件、说明；Baseline 通过前必须填写目标负载的设备、TRT 版本、形状/KV 状态、精度、源码/Engine 哈希和预先声明的指标；Capture 要填写 `study_boundary`，Integration 要填写 `candidate_integration_boundary`；Local/Integration 还需数值门槛和测速字段。
不要为了“通过脚本”而伪造数值。

## 当前实际实现范围

- 已实现：静态 LayerInfo 解析、Tactic 对照、实验阶段与证据记录、单元测试。
- 未实现：Thor 上真实的 Q/K/V/O 捕获、CUDA/Triton Harness、Plugin build、Nsys 自动关联、离线回放连接器。这些必须适配公司实际项目；目前只提供接入标准，不虚构可用性。

最值得先完成的下一项是 **一个确实可运行的端到端样板案例**：例如同事已完成的 Video Block 14 Capture → Local Replay → 原 Engine/候选 Engine 对照 → Decision Record。之后再推广到 Video 0/29、FFN 等。


## v0.3 补充：真实验证器与性能测量

已有项目 `operator_validation` 是执行器而不是 Skill：优先复用项目里的
`validate.py`，不要复制一个并行验证器。产出的 `example_cpu.json`、
`example_cuda.json`、`source_capture.json` 可由 Skill 自带的纯 Python 工具
审计（只检查报告内部一致性，不重新计算结果）：

```bash
python helpers/evidence_audit.py --cpu /path/to/example_cpu.json \
  --cuda /path/to/example_cuda.json --capture /path/to/source_capture.json \
  --out /tmp/local_report_audit.json
```

真实 safetensors、原始 TRT Engine、权重保留在公司；不需要上传进 Skill。
局部正确性 PASS 与 GPU 速度更快是两个不同的结论；不要用缩小后的
`[1,32,128]` 示例同原始 `[48,512,128]` 的 TRT 实现比较速度。
下一步要求公司侧 Codex 做真实形状的 K/O 性能边界测量，具体提示词在
`CODEX_COMPANY_TASK_CN.md`。
