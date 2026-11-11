# 中文上手卡片

## 工作边界

一次只推进一个可证伪的优化假设。先确认 workload、调用边界和基线；
再复用项目现有捕获/验证工具，最后决定是否集成。算子局部时延、Engine
时延和应用端到端时延是不同结果，不能互相替代。

```text
SKILL.md       流程导航与阶段门槛
references/    详细契约和诊断步骤
helpers/       确定性、可测试的元数据工具
templates/     实验记录格式
examples/      简短脱敏案例及其证据限制
```

## 可立即运行的只读/记录命令

```bash
python helpers/layerinfo.py summary /path/to/layerinfo.json \
  --out /tmp/summary.json --markdown /tmp/summary.md
python helpers/layerinfo.py diff /path/to/base.json \
  /path/to/candidate.json --out /tmp/diff.json
python helpers/experiment.py init --out /tmp/exp-001 \
  --hypothesis "减少已测量的算子开销" --target "声明的工作负载"
python helpers/experiment.py status /tmp/exp-001
```

这些命令不会启动 GPU、捕获张量或自动验证数值。`stage pass` 需要现存
证据、说明和适用的数值/测速字段；报告审计器只检查 JSON 声明的一致性，
不重新计算张量结果。

## 先界定运行时生命周期

若 Engine、ExecutionContext、warmup 或 CUDA Graph 可能延迟初始化，先写明
哪些阶段计入 Ready 前启动时间、首个请求，或 steady state。冷文件缓存需要
独立的缓存驻留证据；新进程不等于冷缓存。详见
[`references/09-engine-lifecycle.md`](references/09-engine-lifecycle.md)。

若改用 paged KV，先定义 Engine 输入 ABI、page table 顺序、长度、staging、
提交/清理/淘汰语义，再开始改图或接 Plugin。详见
[`references/10-paged-kv-contract.md`](references/10-paged-kv-contract.md)。

所有原始模型数据、权重、Engine 文件、凭据和项目私有报告应留在其来源
项目中；公开案例只保留脱敏的结论边界和可复用经验。
