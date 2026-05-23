# Baseline Metrics

记录每个 Phase 合并前 `bash scripts/baseline.sh --tag <phase-name>` 的关键指标。
所有数值都来自 `scripts/baseline_logs/<tag>_<datetime>.log` 以及 `Experiment/ExcelResults/`。

## 1. 跑分命令

```bash
# 完整运行（每个 case 200 epoch，约 0.5-1 小时）
bash scripts/baseline.sh --tag phase-0

# 快速回归（每个 case 50 epoch，约 10 分钟），仅做 smoke check
bash scripts/baseline.sh --tag phase-0-fast --fast
```

合并任意 Phase 前都需要至少跑一次完整模式，并把以下表格补完一列。

## 2. 指标说明

| 指标 | 含义 | 出处 |
| --- | --- | --- |
| `test_acc` | 测试集 accuracy（mean ± std，task_num=5） | bench.py stdout / Excel |
| `test_f1`  | 测试集 macro F1（mean ± std） | bench.py stdout / Excel |
| `test_auc` | 测试集 AUROC（mean ± std）；二分类才有意义 | bench.py stdout / Excel |
| `wall_time_s` | 单次完整 run 的墙钟时间，秒 | `/usr/bin/env time -p` 输出的 `real` |
| `peak_mem_mb` | 训练阶段 GPU/MPS/CPU 峰值显存 | bench.py / nvidia-smi / `torch.mps.current_allocated_memory()` |

> 缺失项写 `-`，跑失败写 `FAIL` 并在“备注”列写明原因。

## 3. Case 1 — Cora + GraphCL + GPF (NodeTask, shot=5)

| 列 = Phase Tag | test_acc | test_f1 | test_auc | wall_time_s | peak_mem_mb | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| phase-0 (baseline) | 0.1857±0.0590 | 0.0442±0.0112 | 0.5050±0.0100 | 90.63 | - | bench.py fast (50 ep, num_iter=1) |
| phase-1 |  |  |  |  |  | 8 个 bug 修复后 |
| phase-2 |  |  |  |  |  | API / 配置统一后 |
| phase-3 |  |  |  |  |  | 数据/seed 治理后 |
| phase-4 |  |  |  |  |  | 训练循环重构后 |
| phase-5 |  |  |  |  |  | 评估与汇报统一后 |
| phase-6 |  |  |  |  |  | 测试与 CI 接入后 |

## 4. Case 2 — MUTAG + GraphCL + All-in-one (GraphTask, shot=5)

| 列 = Phase Tag | test_acc | test_f1 | test_auc | wall_time_s | peak_mem_mb | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| phase-0 (baseline) | 0.6307±0.0744 | 0.5786±0.1269 | 0.6220±0.1571 | ~4 | - | downstream_task.py fast (50 ep, CPU) |
| phase-1 |  |  |  |  |  |  |
| phase-2 |  |  |  |  |  |  |
| phase-3 |  |  |  |  |  |  |
| phase-4 |  |  |  |  |  |  |
| phase-5 |  |  |  |  |  |  |
| phase-6 |  |  |  |  |  |  |

## 5. Case 3 — PubMed + GraphCL + Gprompt (NodeTask, shot=1)

| 列 = Phase Tag | test_acc | test_f1 | test_auc | wall_time_s | peak_mem_mb | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| phase-0 (baseline) | 0.4169±0.0549 | 0.4082±0.0441 | 0.5996±0.0627 | ~142 | - | downstream_task.py (5 ep, MPS) |
| phase-1 |  |  |  |  |  |  |
| phase-2 |  |  |  |  |  |  |
| phase-3 |  |  |  |  |  |  |
| phase-4 |  |  |  |  |  |  |
| phase-5 |  |  |  |  |  |  |
| phase-6 |  |  |  |  |  |  |

## 6. 验收口径

- **回归判定**：相对前一列 `test_acc` 下降超过 **1.5 个百分点**，或 `wall_time_s` 上涨超过 **30%**，必须在合并前给出解释或回滚。
- **数据漂移**：Phase 3 修复数据泄漏后，`test_acc` 可能会下跌（这是预期），需在备注里写“数据泄漏修复，结果是新真值”。
- **硬件差异**：换机器跑分需在备注里写 `host=<hostname>, device=<cuda:0/mps/cpu>`，否则不要和旧列比较。

## 7. 日志归档

每次 `baseline.sh` 都会写入 `scripts/baseline_logs/<tag>_<datetime>.log`，
合并 PR 时把这份日志一起提交（或在 PR 描述里贴出 commit hash + 日志路径）。

## 8. 当前回填状态（截至 2026-05-22）

> phase-1 ~ phase-6 列**全部仍空**。Phase 1 完成于 commit `e276670`，Phase 2-3 完成于 `87b01c2`，Phase 4 完成于 PR #1-#8，Phase 5 完成于 PR #9-#15，Phase 6 完成于 `e0d9d4c` + PR #19；但合并各 Phase 时未跑 `bash scripts/baseline.sh` 回填本表。
>
> 这是 `Docs/IMPROVEMENTS.md` §7 中 P1 项 `docs-baseline-metrics-empty` 的具体表现。下次跑回归时建议批量补齐：
>
> ```bash
> for tag in phase-1 phase-2 phase-3 phase-4 phase-5 phase-6; do
>   bash scripts/baseline.sh --tag "$tag" --fast
> done
> ```
>
> 然后把每个 case 的 `test_acc` / `test_f1` / `test_auc` / `wall_time_s` 从 `scripts/baseline_logs/<tag>_*.log` 摘出来填进上面 §3 / §4 / §5 的对应行。`peak_mem_mb` 暂无统一采集脚本，可留 `-`。
>
> 若想直接对当前 HEAD 跑一次完整基线、覆盖 phase-0 列，请用 `--tag phase-0-current`，并在备注里注明 host + device，避免和 2026-05-12 的 fast-run 列直接比较。
