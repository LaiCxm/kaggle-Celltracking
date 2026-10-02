# 交接文档 — Biohub Cell Tracking During Development

> 生成时间：2026-08-24 | 当前最佳：EXP003-child000 LB 0.926（divsub：dual-seed + center-confirmed + ported division rule） | EXP004-child000 LB 0.925（Harmonic+divsub 略降） | 详见 `EXP/EXP_SUMMARY.md:1` `docs/OVERVIEW.md:1`

## 1. 项目与工作流
- **Notebook 命名约定：后续所有 notebook 文件名必须以 `CELL_` 为前缀**（例如 `CELL_infer_harmonic_dualseed.ipynb`），已提交的 EXP002 历史文件暂不追溯改名。

- 工作流复刻自 Kaggle CSIRO Image2Biomass 第5名（本地思考 + Kaggle GPU 执行），两级实验 `EXP{NNN}/config/child-exp{NNN}.yaml`，记忆表 `EXP/EXP_SUMMARY.md`。
- 已按新要求改为**本地独显优先**：`AGENTS.md:23` / `CLAUDE.md:23` / `README.md:21` 新增执行策略，`tools/check_gpu.py:1` 一键自检（`nvidia-smi` + `torch.cuda.is_available()` + 显存≥4GB）。有算力时 `python EXP/EXP001/train.py --config ...` 本地跑，仅 `kaggle competitions submit` 走 API。

## 2. 比赛情报（已沉淀）

- 比赛：`biohub-cell-tracking-during-development`，Research，奖金$60K，截止2026-09-29，已报名（`laicxm`）。
- 任务：3D+t 荧光显微视频，输出 node（t/z/y/x）+ edge（相邻帧 t→t+1，允许二分裂）。
- 指标（官方 `royerlab/kaggle-cell-tracking-competition@075fc5f`）：7µm 最优指派匹配 → `score = adj_edge_jaccard + 0.1*division_jaccard`，`adj = jaccard*(1-0.1*(Tpred-Ttrue)/Ttrue)`，按 `w=TP+FP+FN` 加权。`docs/OVERVIEW.md:9` `docs/DATASET.md:1`
- 数据：zarr `T×Z×Y×X uint16`，SCALE `(1.625,0.40625,0.40625)`，`test` 仅4视频（`44b6_0113de3b,44b6_0b24845f,6bba_05b6850b,6bba_05db0fb1`），`train` 17+。提交 10 列 `id/dataset/row_type/node_id/t/z/y/x/source_id/target_id`。
- LB 现状：榜首0.952，公开 clean 链 0.908→0.915。

## 3. 高分方案抓取（2026-08-19）

- Kaggle Discussion + `kaggle kernels list --sort-by voteCount` 前30，GitHub `biohub cell tracking kaggle` 搜28仓库。
- 拉取10份高票 notebook 到 `C:\UserData\AppData\Local\Temp\opencode\ct_nbs` 并用3个 explore agent 拆解，沉淀 `docs/Idea_Research/weekly-2026-08-19.md:1`。
- 核心 support-pack：`pilkwang/biohub-tracking-support-pack-50ep-v1`（349MB，公开，9324次下载）含 `TemporalUNet3D` + `SimpleNodeTransformer` 权重 `edge_predictor_best.pth` 与离线 wheels；双种子第二模型 `pilkwang/biohub-temporal-unet3d-seed314159-v1`；P100 兼容 torch `suioshion/p100-torch-wheels-cu118`。
- 标记 hack：`kaiwalyaatulraut/solution` 注入 hub 节点（t=-1000）刷 division，已禁。

## 4. 核心成果：EXP001 复现（LB 0.908 对齐）

- 来源：`yusuketogashi/clean-approach-lightweight-local-cv-no-hack`（Biohub 132，`ADAPTIVE_SHORT_TRACK_RESCUE` 5节点高置信救援）。
- 路径：`EXP/EXP001/infer_clean_repro.ipynb:1`（20 cells，删封面图，artifact 路径加 `/kaggle/input/<slug>` fallback），`config/child-exp000.yaml:1`，`tools/push_custom_kernel.py:1`（支持 `--data-source` + `machine_shape`）。
- 关键修复：
  - Kaggle 默认 torch 2.10+cu128 不支持 P100(sm_60) → 离线装 cu118 torch（重命名 wheel 为 PEP440 + `--no-index --no-deps --force-reinstall`，nvidia 依赖同目录逐个装）`EXP/EXP001/infer_clean_repro.ipynb:6`
  - `machine_shape` 必须用 SDK 枚举 `NvidiaTeslaT4`（`gpu_t4` 无效会分到 P100）`tools/push_custom_kernel.py:88`
  - 比赛禁 internet，仅允许 T4 x2，已全适配；v11 `NvidiaTeslaT4` 分到 T4 成功
- 验证：4次运行 submission SHA `512d3f5f...` 完全一致，236,203 行覆盖全部4测试视频；CV `0.87893`（固定8胚胎 `CV8`，`reference_124 0.87921`），`outputs/child-exp000/submission.csv:1` 已提交 LB **0.908** 对齐。
- 提交机制：code competition，仅接受 notebook 生成的 `submission.csv`，`kaggle competitions submit` 直传被拒 400，需在 notebook 的 Output 页点 Submit。

## 5. 迭代实验（单点原则）

| 实验 | 改动 | CV | 结论 |
|---|---|---|---|
| child000 | 基线 | 0.87893 | LB 0.908 最佳 |
| child001 | ILP appearance 0.0→0.1 | 0.87880 | ❌ |
| child002 | ILP disappearance 1.575→1.2 | 0.87320 | ❌ |
| child003 | det_threshold 0.96875→0.95 | **0.88167** | CV升LB 0.907 基本平（gap） |
| child004 | safe_div 几何门 4.66/8.5/7.65→13 | 0.87699 | ❌ division仍0 |
| child005 | 双种子 raw 融合 α0.475/β0.15 | 0.87871 | ≈持平（TP↑但FP↑） |
| child006 | 双种子 LMC 融合 | 0.87723 | ❌ |

- 参数空间已饱和；最大弱点 `division_jaccard=0`（ILP 不产生 fork(out≥2)，后处理 293→447 分裂源全被 metric 忽略）。GT 诊断：CV8 仅7个分裂，父→子距离1.28–12.85µm，safe_div 4.66µm 门是瓶颈但放宽仍无效（metric 判定极严，后处理 ROI 极低，需模型级改进）。

## 6. 新环境与清理

- 体积：全项目 **24.98 MB**，`EXP/EXP001/outputs:23.62 MB`（两份 submission 各11MB），其余1.3MB。
- 清理：仅删 `EXP/EXP001/test_autosubmit.ipynb:1`（dummy 机制测试，无价值）；其余 `diag_*.ipynb`/`exp001_*.ipynb` 保留为历史；如需打包瘦身可排除 `EXP/*/outputs/`（可重跑）。
- 迁移后首跑：`python tools/check_gpu.py`，数据 `kaggle competitions download -c biohub-cell-tracking-during-development -p data/`。

## 7. 下一步建议

- 已验证路线（0.908→0.915）：完整 two-seeds + DeepCenter（需挂 `biohub-deepcenter-unet3d-center-prior-v1`），改动较大但依据最强。
- 或接受 0.908 基线，深挖 division 需训练新模型（当前无训练代码）。
- 提交：本项目 code competition，需 `kaggle competitions submit` 或 notebook Output 提交，文件名必须 `submission.csv`。

## 8. 目录速览

```
AGENTS.md / CLAUDE.md       护栏（含本地优先策略）
README.md                   使用手册（含本地/云端双路径）
EXP/EXP001/                 infer_clean_repro.ipynb + 6个child变体 + outputs/
docs/                       OVERVIEW / DATASET / Idea_Research/weekly-2026-08-19.md
HANDOVER.md                 本文档
tools/                      push_training/inference/pull_results/push_custom_kernel/check_gpu
templates/                  内核模板（勿手改）
```
