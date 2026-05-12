# ProG 文档

ProG 是一个 graph prompt learning 框架。这里是文档目录的索引——按你想做的事挑一个入口：

| 我想… | 看这里 |
|---|---|
| 跑一次 baseline，或者了解三个入口 (`pre_train.py` / `downstream_task.py` / `bench.py`) | [`running.md`](./running.md) |
| 看支持哪些数据集、磁盘根目录约定、踩坑点 | [`datasets.md`](./datasets.md) |
| 理解模块边界、`PromptStrategy` 协议、初始化顺序 | [`architecture.md`](./architecture.md) |
| 看历史 bug、Phase 0-6 改进 roadmap | [`IMPROVEMENTS.md`](./IMPROVEMENTS.md) |
| Diff baseline metric | [`baseline_metrics.md`](./baseline_metrics.md) |
| 给 ProG 提 PR 的完整规范 | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) |
| 看给 AI 协作者 (Claude / Cursor / etc.) 的项目级约定 | [`../CLAUDE.md`](../CLAUDE.md) |

---

## 快速链接

- **数据集表** — Node-level 9 + Graph-level 14；详见 [`datasets.md`](./datasets.md) §1-2
- **预训练范式** — DGI / GraphMAE / GraphCL / SimGRACE / Edgepred_GPPT / Edgepred_Gprompt / MultiGprompt；详见 [`running.md`](./running.md) §4
- **Prompt 类型** — `None` / `GPF` / `GPF-plus` / `Gprompt` / `All-in-one` / `GPPT` / `MultiGprompt`；详见 [`architecture.md`](./architecture.md) §3
- **GNN backbone** — `GCN` / `GAT` / `GIN` / `GraphSAGE` / `GCov` / `GraphTransformer`；通过 `prompt_graph.model.build_gnn` 注册
- **下游任务** — `NodeTask` / `GraphTask`（`LinkTask` 暂未接入 strategy 框架，见 [`IMPROVEMENTS.md`](./IMPROVEMENTS.md) §1.10）

---

## 历史

本目录从 Phase 6.2 起拆分。原来的"单文件 Docs/README.md（生成于 2026/05/01）"被切成 [`architecture.md`](./architecture.md) / [`datasets.md`](./datasets.md) / [`running.md`](./running.md) 三份，本文档收敛为导航 hub。如果你在找 "支持的数据集表"、"使用示例"、"Prompt 类型表" 等具体内容，按上面的索引跳过去。
