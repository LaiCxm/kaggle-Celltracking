# AGENTS.md — Kaggle Agent 工作流护栏

本仓库是一套"本地 agent 思考 + Kaggle GPU 执行"的竞赛工作流。
结构复刻自 Kaggle CSIRO Image2Biomass 第 5 名方案(Claude Code + Codex 协作)。

## 项目结构(两级实验)

```
EXP/
  EXP{NNN}/                  # 大实验:改动 train.py 的主干逻辑时新建
    train.py                 # 训练脚本(接口契约见下)
    infer.py                 # 推理脚本
    config/
      child-exp{NNN}.yaml    # 小实验:只改参数/loss/模型设置时,只加新 yaml
    outputs/                 # 结果回填目录(本地,由 pull_results.py 拉取)
  EXP_SUMMARY.md             # 实验记忆表(必须持续维护)
docs/                        # OVERVIEW.md / DATASET.md / Idea_Research/
templates/                   # Kaggle 内核模板(不要手改,由 tools 渲染)
tools/                       # push/pull 工具脚本
output/CV_LB/                # CV-LB 相关性分析输出
```

## 执行策略：本地独显优先（GPU 配额受限时新增，2026-08-20 起生效）

> Kaggle GPU 配额受限（每周 30h）时，优先使用本地独立显卡，仅评分/提交走 Kaggle API。

1. **自动检测**：每次训练/推理前，agent 需执行 `nvidia-smi` 与 `python -c "import torch; print(torch.cuda.is_available())"` 探测；
   若 `torch.cuda.is_available()==True` 且显存 ≥4GB，判定为“本地有算力”。
2. **本地有算力时**：
   - 直接本地执行 `python EXP/EXP001/train.py --config EXP/EXP001/config/child-expXXX.yaml --folds 0,1,2,3,4 --use_wandb False --create_oof True`
     与 `python EXP/EXP001/infer.py --config ... --submission outputs/submission.csv`；
   - 输出仍遵循契约 `outputs/models/best_model_fold{i}.pth` / `oof_predictions.csv` / `results.json`；
   - 数据需预先 `kaggle competitions download -c biohub-cell-tracking-during-development -p data/` 并将 `config` 的 `data_dir` 指向本地 `data/`；
   - 仅在需要官方评分/提交时使用 Kaggle API：`kaggle competitions submit` 或 `kaggle competitions leaderboard`。
3. **本地无算力时**：走原流程 `python tools/push_training.py --exp ... --child ... --competition ...` 推送至 Kaggle GPU，`tools/pull_results.py` 拉回。
4. **工具兼容**：`tools/push_*.py` 保留不变；新增 `tools/check_gpu.py`（`python tools/check_gpu.py` 一键自检）供迁移后首跑验证。

## 硬性规则(违反即为任务未完成)

1. **评测函数必须与官方 metric 完全一致**。写任何实验代码前,先在
   `docs/DATASET.md` 确认 metric 定义,实现后**先写单元测试验证正确性**
   (参考教训:一个 `.mean()` 的评测实现错误曾导致 CV/LB gap 恶化 17%)。
2. **每个实验必须回填 EXP/EXP_SUMMARY.md**(CV、LB、成败、备注),
   失败也要记录。不回填 = 任务未完成。
3. **提新想法前,先 grep EXP_SUMMARY.md**,确认该想法没被试过。
   已失败的想法不许重复提出。
4. **改动默认数据增强/预处理风险极高**,默认配置往往已在测试分布上最优,
   非明确理由不得更改;确需尝试必须开新 child-exp 单独验证。
5. **CV 高 ≠ LB 高**。OOF 优化的 ensemble、过度的后处理往往不迁移到
   公开 LB,这类操作需谨慎并明确记录 gap。
6. **train.py 必须支持以下 CLI 接口**(tools 与内核模板依赖此契约):
   ```
   python train.py --config config/child-exp000.yaml \
                   --folds 0,1,2,3,4 \
                   --use_wandb True --create_oof True
   ```
   输出约定:
   - `outputs/models/best_model_fold{i}.pth`
   - `outputs/oof_predictions.csv`
   - `outputs/results.json`(含 fold 得分、mean CV、OOF 得分)

## 每周例行任务:开源方案抓取(每周至少一次,过期视为失职)

**目的**:竞赛环境每周都在变,公开讨论区/开源仓库里的新方案是最便宜的"情报"。
每过一周(从项目启用起算,每周固定一次)执行一次抓取,流程:

1. **Kaggle 讨论区与本比赛 Notebook**:
   - 访问本比赛 Discussion 页,抓取新增的高赞/精华帖(特别是 EDA、CV 策略、
     标签处理、数据泄露讨论)
   - 用 `kaggle kernels list --competition <slug> --sort-by voteCount` 拉取
     高票 notebook,重点看:指标实现、数据处理、模型结构、训练技巧
2. **GitHub 搜索**:
   - 搜索本比赛 slug(如 `rsna-knee-abnormality-detection`)和相似比赛
     (同类型任务、同数据形态、同 metric)的开源方案仓库
   - 关键词示例:`<比赛名> kaggle solution`、`kaggle 金牌`、`<metric> kaggle`
3. **借鉴参考并沉淀**(必须产出,否则抓取无意义):
   - 有价值的发现写入 `docs/Idea_Research/weekly-YYYY-MM-DD.md`(新文件,
     日期命名),格式:来源链接 + 一句话要点 + 可借鉴程度(高/中/低)+ 建议实验
   - 与当前实验冲突或可直接复用的想法,按护栏规则走:先查 EXP_SUMMARY.md
     是否试过,没试过就转成新 child-exp 提给用户
   - 每周抓取后,在 `docs/Idea_Research/weekly-*.md` 顶部更新一行"最近抓取日期"
4. **提交前检查**:每次准备提交新实验前,若距上次抓取超过 7 天,先执行本轮
   抓取再提交实验(防止重复造轮子)

## 指令模板(给 agent 下任务的标准句式)

- 大实验:在 EXP{NNN} 下写 train.py,实现 XX 方法
- 小实验:在 EXP{NNN}/config 下新建 child-exp{NNN}.yaml,把 loss 改成 XX,
  其余保持不变
- 分析:读 EXP{NNN}/outputs 的 results.json 和 oof_predictions.csv,分析
  CV/LB gap,给出下一轮建议,更新 EXP_SUMMARY.md

## 工具命令(本地执行)

**本地有算力时（优先）：**
```
python tools/check_gpu.py
python EXP/EXP001/train.py --config EXP/EXP001/config/child-exp000.yaml --folds 0,1,2,3,4 --use_wandb False --create_oof True
python EXP/EXP001/infer.py --config EXP/EXP001/config/child-exp000.yaml --submission outputs/submission.csv
kaggle competitions submit -c <slug> -f outputs/submission.csv -m "EXP001 child000"
```

**本地无算力时（回退 Kaggle GPU）：**
```
python tools/push_training.py --exp EXP000 --child child-exp000 --competition <比赛slug> [--no-gpu]
python tools/pull_results.py --exp EXP000 --child child-exp000
python tools/push_inference.py --exp EXP000 --child child-exp000 --competition <比赛slug>
```

- GPU 配额:每周 30 小时,单次内核最长 12 小时。
- 调试/冒烟测试用 `--no-gpu`(CPU 内核不占 GPU 配额)。
- 模型等产物以 Kaggle Dataset 为中转:`{user}/exp-results-{exp}-{child}`。
