# Capture + Local Numerical 源码审计与运行记录

日期：2026-11-09（Thor，Python 3.12.3 / PyTorch 2.10.0a0+b558c986e8.nv25.11 / CUDA 13.0）

## 1. 仓库与实现位置

- `realpath /home/nvidia/sjz_workspace/repos/lingbot-va-opt` 和 `realpath /mnt/8tb_wd/Thor_data_new/sjz_workspace/repos/lingbot-va-opt` 均为 `/mnt/8tb_wd/Thor_data_new/sjz_workspace/repos/lingbot-va-opt`；是同一工作区。
- 当前 Git 根目录：`/mnt/8tb_wd/Thor_data_new/sjz_workspace/repos/lingbot-va-opt`。运行时 `git rev-parse --show-toplevel` 返回该目录。
- 被审计的实现位于同事 checkout：`/home/nvidia/HY_workspace/git_codes/lingbot-va-opt/lingbot-tensorrt/operator_validation`；原始完整捕获产物位于 `/mnt/8tb_wd/Thor_data_new/HY_workspace/lingbot_va_tensorrt/attention_validation/full_depth_20261102`。这两处不属于上述当前 Git 根目录。
- 本次只新增本报告及 `audit_results/capture_local_20261109/validation_cuda.json`；没有修改 `operator_validation` 源码、冻结 fixture、既有 Evidence、ONNX、engine 或生产运行时。

## 2. Q/K/V/O 的来源与调用链

### 真实输入与历史 KV

真实输入捕获入口是同一实验目录下的 `scripts/capture_full_video_input.py`，它复用 `video_sdpa_real_capture_20261029/harness/capture_trt_video_input.py`：

1. `capture_trt_video_input.py` 用 Episode 7 的记录 RGB/action 建立离线评估；默认 commits=15、seed=0，关闭机器人 I/O。配置为 Video/Action 各 5 步、FlashAttention，见 [capture_trt_video_input.py:45-74, 88-115, 128-143](/home/nvidia/HY_workspace/hy_codes/lingbot_c++/experiments/video_sdpa_real_capture_20261029/harness/capture_trt_video_input.py:45)。
2. 它执行 15 次真实 cache commit；每次 512 个 Video token + 32 个 Action token，因此历史长度为 8160。随后拦截下一次 Video generate、在执行 engine 前保存 `noisy_latents/timesteps/grid_id/text_emb` 与 layer 0 的完整 packed `history_k/v`，见 [capture_trt_video_input.py:181-217, 221-253](/home/nvidia/HY_workspace/hy_codes/lingbot_c++/experiments/video_sdpa_real_capture_20261029/harness/capture_trt_video_input.py:221)。manifest 记载该输入为 episode 7、15 commits、8160 history tokens，并保存 Engine/input 哈希。
3. `capture_full_video_input.py` 包装该入口，并在保存时把 30 层各自的历史 K/V 分 shard 写入 safetensors 与 `full_history_manifest.json`，见 [capture_full_video_input.py:58-73, 110-132](/home/nvidia/HY_workspace/hy_codes/lingbot_c++/experiments/video_sdpa_validation_20261030/scripts/capture_full_video_input.py:58)。`validate_full_depth.py` 再逐层读 shard、校验 SHA256，拼成 `[30,2,8160,24,128]` 历史输入，见 [validate_full_depth.py:152-176](/home/nvidia/HY_workspace/hy_codes/lingbot_c++/experiments/video_sdpa_validation_20261030/scripts/validate_full_depth.py:152)。

### 完整 TRT Forward 到 `sdpa_layer00.safetensors`

1. `validate_full_depth.py build` 从原始 30 层 Video ONNX 创建一个启用 `mark_unfused_tensors_as_debug_tensors()` 的 Debug Engine；使用同一 timing cache/profile，但它是重新 build 的 engine，见 [validate_full_depth.py:31-66](/home/nvidia/HY_workspace/hy_codes/lingbot_c++/experiments/video_sdpa_validation_20261030/scripts/validate_full_depth.py:31)。
2. `capture` 顺序运行原始 serialized Production Engine 和 Debug Engine，绑定同一份记录输入，并用 `execute_async_v3` 在独立 CUDA stream 上执行，见 [validate_full_depth.py:69-110, 177-205](/home/nvidia/HY_workspace/hy_codes/lingbot_c++/experiments/video_sdpa_validation_20261030/scripts/validate_full_depth.py:69)。对 Debug Engine，Inspector 的 MHA 输入/输出名映射为 Q/K/V/O；`IDebugListener` 从设备地址异步拷到 pinned host 并同步回调 stream，见 [validate_full_depth.py:127-145](/home/nvidia/HY_workspace/hy_codes/lingbot_c++/experiments/video_sdpa_validation_20261030/scripts/validate_full_depth.py:127) 与 [capture_debug_sdpa.py:146-182](/home/nvidia/HY_workspace/hy_codes/lingbot_c++/experiments/video_sdpa_validation_20261030/scripts/capture_debug_sdpa.py:146)。
3. 收齐选定层 0/14/29 的 Q/K/V/O 后，按角色连续化并写到 `reports/sdpa_layer00.safetensors` 等文件；同一脚本还保存两套 Engine 的外部 `noise_pred/current_k/current_v`，见 [validate_full_depth.py:148-220](/home/nvidia/HY_workspace/hy_codes/lingbot_c++/experiments/video_sdpa_validation_20261030/scripts/validate_full_depth.py:148)。
4. `operator_validation/prepare_example.py` 读取 layer 0 完整 capture：Q/O 取 `[0:1,0:32,:]`，K/V 保留 `[0:1,:,:]`，全部 `.contiguous()` 后写入 safetensors；FP32 输出从既有 local replay 报告取对应 slice，见 [prepare_example.py:42-64](/home/nvidia/HY_workspace/git_codes/lingbot-va-opt/lingbot-tensorrt/operator_validation/prepare_example.py:42)。脚本检测目标 fixture 已存在时会拒绝覆盖。

复现入口（会建立新的大文件 run，不要把 `$RUN` 指向已有 `full_depth_20261102`；本轮没有重新 build/capture）：

```bash
cd /home/nvidia/HY_workspace/hy_codes/lingbot_c++/experiments/video_sdpa_validation_20261030
RUN=/mnt/8tb_wd/Thor_data_new/HY_workspace/lingbot_va_tensorrt/attention_validation/full_depth_20261109_audit
python scripts/capture_full_video_input.py --output "$RUN/input/recorded_chunk15_trt_input.pt" --episode 7 --commits 15 --seed 0
python scripts/validate_full_depth.py build --run-dir "$RUN"
python scripts/validate_full_depth.py capture --run-dir "$RUN" --layers 0 14 29
python scripts/validate_full_depth.py replay --run-dir "$RUN" --layers 0 14 29
```

这些入口需要同事的模型、dataset、ONNX/engine 和其 checkout 中的依赖；build 需要数分钟/大量内存。以上仅是源码级复现命令，本轮未重跑它们；本轮重跑的是下一节的固定 fixture validator。

## 3. 契约审计：哪些已确认、哪些沿用历史记录

| 项目 | 当前 fixture/记录 | 证据级别与限制 |
|---|---|---|
| 算子边界 | Self-attention 的 Q/K/V 到 attention 输出 O；O 在 flatten 与 `to_out[0]` 投影前 | ONNX exporter 源码可确认 Q/K norm、RoPE 后执行 `attn_op(query,key,value)`，然后 flatten/projection，见 [export_lingbot_va_cache_step_bf16_onnx.py:49-71](/mnt/8tb_wd/Thor_data_new/sjz_workspace/repos/lingbot-va-opt/lingbot-tensorrt/scripts/export_lingbot_va_cache_step_bf16_onnx.py:49)。Debug tensor 名称/对应 TRT layer 来自既有 Inspector capture 映射，不是本轮重新抓取。 |
| 完整生产样本 shape | BF16：Q/O `[48,512,128]`，K `[48,128,8672]`，V `[48,8672,128]` | Debug callback 记录与 safetensors 元信息；完整 Q/O stride `[65536,128,1]`，K/V stride `[1110016,8672,1]`、`[1110016,128,1]`。来自既有 capture 元数据。 |
| 当前 fixture shape/stride | BF16 Q/O `[1,32,128]`，stride `[4096,128,1]`；K `[1,128,8672]`，stride `[1110016,8672,1]`；V `[1,8672,128]`，stride `[1110016,128,1]`。FP32 `o_ref_fp32` `[1,32,128]`，stride `[4096,128,1]` | contract、manifest 和 fixture SHA256 本轮由 validator 实际核验。切片后的 Q/O 被复制成连续 tensor；这是裁剪连续样本，不保留原始完整 Q/O 的 stride/batch/head 覆盖。 |
| Scale | `1/sqrt(128) = 0.08838834764831845` | contract 与 local reference 代码确认；不是从 TRT callback 单独读取的标量。 |
| Mask / causal / dropout | `mask=null`，non-causal，dropout=0 | contract 和既有 capture semantics 记录；TensorRT 内部 mask/bias 元数据未被单独抓取。本 fixture 对应的 PyTorch reference/示例按这些语义计算。 |
| KV cache 语义 | 历史 8160 + 当前 Video K/V 512 = attention K/V 长度 8672；K/V 位于同一层自己的 history 与当前投影拼接后 | exporter 源码确认先做 Q/K norm 和 RoPE，再记录 current K/V、与 explicit history 在 token 维 concat，见 [export_lingbot_va_cache_step_bf16_onnx.py:49-68](/mnt/8tb_wd/Thor_data_new/sjz_workspace/repos/lingbot-va-opt/lingbot-tensorrt/scripts/export_lingbot_va_cache_step_bf16_onnx.py:49)。15 次真实 commit/8160 与逐层实际 history bytes 是历史捕获 manifest 记录；本轮没有重放 15 次 episode。 |
| 输出写入 | caller 提供 BF16 `y`；候选必须原位写满，不能替换 `outputs['y']`；输入不可修改 | `validate.py` 对 NaN 初始化输出、前后 canary、输入字节、shape/dtype 和有限性实际检查；这是 validator harness 契约，不是 TensorRT Plugin ABI。 |
| Stream | CUDA 情况传入 `torch.cuda.current_stream(device)`，在该 stream 上调用候选并同步后检查；candidate adapter 使用 PyTorch math backend | [validate.py:31-56](/home/nvidia/HY_workspace/git_codes/lingbot-va-opt/lingbot-tensorrt/operator_validation/validate.py:31) 实码确认。capture 使用的 TRT 执行 stream 与 local validator stream 是两件事。 |
| 数值门槛 | candidate 对 `o_trt` 逐元素满足 `abs(diff) <= 0.02 + 0.02*abs(o_trt)`；另以 `1e-5/1e-5` 检查保存的 FP32 reference 可复算 | contract/validator 实码确认，适用于此 reduced local fixture；不等于生产任务或深层误差门槛。 |

## 4. Production / Debug 对照证据及局限

从既有 `full_depth_20261102/analysis/capture.json` 和 `build.json` 读取（历史记录，不是本轮重跑）：

- Production Engine SHA256：`f8693f268f96845777d9448a48193f0909dc3763274859acdc389ff478edac79`；Debug Engine SHA256：`be1ac0b51b50bbf7bf421bac5efdfa5e7844dc6ae6bd55ed1ecc4613852555ea`。两者不同；Debug 是同 ONNX、启用 unfused debug tensor 后重建。
- 两边 Inspector layer 数都是 643；layer/tactic 序列记录为一致；第 0 层映射 `_gemm_mha_v2_myl0_17`，tactic `_gemm_mha_v2_0xce3ea3e0390335d10d2ce77c980a231f`。
- 对该次固定输入，外部 `noise_pred`、`current_k`、`current_v` 均记录 finite 且 bitwise equal；选中层 0/14/29 的 callback 无 listener 错误，`capture_passed=true`。
- 这是对 Debug capture 可信度的支持证据：生产可见输出、层数和 tactic 对齐。但 engine 哈希不同，加入 debug 可观测性可能改变内部 fusion、layout、舍入或数据流。外部输出逐位相同并不能证明内部 Q/K/V/O 与原 Production Engine 内部值逐位相同；层名相同也不证明内部算子 kernel 等价。

完整生产 capture 与本 fixture 的关键差异：生产 shape 有 48 个 batch-head 和 512 个 query；fixture 仅取一个 batch-head 的前 32 个 query，但保留全部 8672 个 K/V。因为 attention 为 non-causal，query/head 行相互独立，该切片可用于验证对应行的 local 数值；它不覆盖完整 Q/O、其它 head/query、Action Engine、其他 cache 长度，也没有在缩小 shape 上执行原 TRT kernel。

## 5. 本轮实际执行

环境探测：`nvidia-smi` 显示 NVIDIA Thor，`torch.cuda.is_available()` 为 `True`，CUDA 13.0。没有运行性能 benchmark、Nsight、Plugin、ONNX build 或生产 Engine。

实际命令与结果：

```bash
cd /home/nvidia/HY_workspace/git_codes/lingbot-va-opt/lingbot-tensorrt/operator_validation
/usr/bin/python validate.py --device cuda \
  --output /mnt/8tb_wd/Thor_data_new/sjz_workspace/repos/lingbot-va-opt/operator-optimization-workbench-v0.3/audit_results/capture_local_20261109/validation_cuda.json
/usr/bin/python -m unittest -v test_validation
```

- CUDA `validate.py`: **PASS**，candidate 是 `candidate_example.py`（PyTorch Math SDPA，FP32 中间值后写 BF16），candidate SHA256 `c38aed73cadd74e1191395ddb4b2a5d39ba7d8d53e1d8a996134ee75e5a8d036`。
- 对 `o_trt` 的 4096 个元素：outside tolerance=0，MAE `0.0001450609`，RMSE `0.0003703116`，max abs `0.00390625`，cosine `0.9999985831`。
- 保存的 FP32 reference 复算通过（max abs `5.126e-06`，reference 自身检查门槛 `1e-5/1e-5`）。candidate/FP32 诊断 MAE `0.0001605329`、RMSE `0.0003080005`、max abs `0.003656745`。
- 输入未变、caller output 保持、完整写入、canary 未破坏、shape/dtype 正确、重复输出 bitwise 相同；Q/K/V 各自扰动后 reference 与 candidate 输出均变化且扰动 case 通过。
- `test_validation.py`: **6/6 PASS**：有效 adapter、错误输出、部分写入、输入修改、canary 覆盖、缓存答案冒充计算均按预期接受/拒绝。
- 新 JSON：[`validation_cuda.json`](audit_results/capture_local_20261109/validation_cuda.json)。未覆盖 `evidence/example_cuda.json` 或任何冻结 fixture。

## 6. 尚未解决与下一轮 Benchmark 前置条件

1. 本轮验证对象是方法示例 `candidate_example.py`，不是优化后的 CUDA/Triton kernel，也不是导出的 TRT 原始 kernel；因此只证明 fixture、reference 和 validator 当下可运行，不能支持性能结论。
2. Local 数值运行仅覆盖 `[1,32,128]` Q 的切片；要做生产形状性能分析，必须先用完整 `[48,512,128]` Q/O、完整 `[48,128,8672]` K、`[48,8672,128]` V 对实际候选完成数值/行为验证，保留真实 stride/layout 与 cache 语义。
3. 目前没有同边界、同输入、同 shape 的独立原 TRT kernel-only 重放证据。下一轮如无法获得原 TRT kernel 与参数/资源的可重放边界，`baseline_kernel_only` 必须记 `unavailable`；不得把 Inspector layer 或整 Engine 延迟当作原 kernel 时间。
4. 进入性能测量前需固定 candidate 源码/编译 hash、完整 workload hash、设备/版本、精度与数值门槛；分别测 kernel-only K、包含必需布局转换/pack/dispatch 的 full-call O，并记录 warmup、重复次数、p50/p90、stream、同步点、CUDA Graph 与 GPU 背景负载。性能测试另开报告，本报告不包含 benchmark。
5. 若候选声称覆盖 Action、不同历史长度或非连续输入，要为这些 workload 各自补齐契约与正确性样本；当前 Video 满历史、单层、单 slice 结果不能外推。

## 当前结论

Capture 链和 fixture 派生关系已由源码追踪到 safetensors 写入点；Production/Debug 对照证据已从历史产物核对，并标明其内部值可信度边界。Thor 上现有 local validator 实际 PASS，负例测试 6/6 PASS。Capture 源头未重跑，local PASS 仅适用于缩小 fixture 上的 PyTorch Math SDPA 示例。GPU 性能 Benchmark、Plugin 回插和生产 Engine 修改均未进行。
