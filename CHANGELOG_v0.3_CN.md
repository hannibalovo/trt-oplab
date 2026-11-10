# v0.3：将真实算子验证证据接入 Skill

保留 v0.1 的五阶段导航、Engine Inspector 分析与实验状态工具，新增：

- `references/07-validation-harness.md`：可复用的真实数据验证契约；优先对接项目现有验证器，不复制实现。
- `references/08-benchmark-protocol.md`：区分缩小后的正确性样本与生产形状，严格标记 K/O/I 性能测量边界。
- `helpers/evidence_audit.py`：纯标准库读取 source_capture、CPU/CUDA 报告、可选 Manifest，验证报告内声明的一致性，生成机器可读审计。
- `tests/test_evidence_audit.py`：正例、伪坏候选、外部匹配 ≠ 内部完全等价三个方向的测试。
- `templates/benchmark_report.json`：性能报告空模板，**没有填入虚构的性能数据**。
- `CODEX_COMPANY_TASK_CN.md`：在公司仓库里执行的可操作数据采集任务，要求只返回可分享的汇总，不上传原始 safetensors。
- 修正 `experiment.py` 的本地/集成数值门槛：必须有显式的 `numeric_validation_pass=true`，不能用 `max_abs_error <= atol` 代替逐元素 `atol+rtol*abs(ref)`。
- 降低默认安装包的数据泄露风险：不再打包以真实公司 Engine 为来源的旧 `examples/action_summary.*`。

## 局限

本版 **没有**实现通用 TensorRT 中间张量抓取、原始 TRT kernel 导出、公司 Plugin 回插、CUDA/NCU 自动 benchmark。三份本地报告中的 PASS 是文件里**报告的结果**，不是我们无张量复算出来的结果。
