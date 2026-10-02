# Kaggle Agent 工作流 — 使用手册

复刻自 Kaggle CSIRO Image2Biomass 第 5 名方案(Claude Code + Codex 协作打竞赛)。
核心设计:**本地 agent 负责思考,Kaggle 官方 GPU 负责执行**。

## 部署到新比赛(每次只做一次)

```powershell
# 1. 把本模板的【内容】复制到比赛项目根目录(不要嵌套一层文件夹!)
#    agent 工具只读取工作目录根部的 AGENTS.md/CLAUDE.md,必须平铺
Copy-Item "D:\kaggle\skills\kaggle-agent\*" "D:\kaggle\<新比赛>\"
# 2. 在该目录开 agent 会话(如 opencode / claude code)
# 3. 让 agent 通读 AGENTS.md、docs/OVERVIEW.md 模板,填好比赛信息
```

> 注意:母版 `D:\kaggle\skills\kaggle-agent` 永不改动,每打一个比赛复制一份。
> 复制后 `EXP/` 里的基线、`docs/` 里的比赛文档都按新比赛重写。

## 架构

```
┌──────────────────────────────────────────────────────────────┐
│  本地(你的机器)                          │  Kaggle 云端       │
│  ├─ Claude Code / Codex / opencode       │  ├─ 训练内核(GPU)  │
│  ├─ EXP/ 两级实验 + EXP_SUMMARY.md 记忆  │  ├─ 推理提交内核    │
│  └─ tools/ push·pull 脚本                │  └─ Dataset 中转    │
└──────────────┬───────────────────────────┴─────┬─────────────┘
               │  kaggle kernels push            │  结果回传
               └────────────►  ────────────────► ┘
```

数据流:本地代码 → push 训练内核 → Kaggle GPU 训练 → 产物上传为
Dataset(`{user}/exp-results-{exp}-{child}`)→ pull 回本地 → agent 分析
→ 更新 EXP_SUMMARY → 下一轮。提交时 push 推理内核 → 拿 LB。

> **本地独显优先（2026-08-20 新增）**：若 `nvidia-smi` 检测到本地独显（≥4GB 且 `torch.cuda.is_available()`），则**优先本地执行** `python EXP/EXP001/train.py ...` / `infer.py ...`，输出仍遵循契约；仅评分/提交走 Kaggle API。详见 `AGENTS.md` 的“执行策略：本地独显优先”。

## 第 0 步:一次性配置

1. Kaggle 账号 → Settings → Create API Token,得到 kaggle.json,
   放到 `~/.kaggle/kaggle.json`(Windows:`C:\Users\你的用户名\.kaggle\kaggle.json`)
2. 验证:`kaggle competitions list` 能列出比赛即成功
3. 报名你要打的比赛(必须,数据才可下载)

## 第 1 步:开新比赛(一次)

1. 让 agent 通读比赛主页,填写 `docs/OVERVIEW.md`(任务、metric 公式、提交格式)
2. 让 agent 拉数据样例,填写 `docs/DATASET.md`(字段、目标、CV 策略)
3. 让 agent 按接口契约写第一个基线 `EXP/EXP000/train.py` + `config/child-exp000.yaml`
   (参考现有骨架;评测函数必须写单测验证)

## 第 2 步:跑训练(每次实验)

**优先本地执行（若 `nvidia-smi` 检测到独显）：**
```powershell
python tools/check_gpu.py
python EXP/EXP000/train.py --config EXP/EXP000/config/child-exp000.yaml --folds 0,1,2,3,4 --use_wandb False --create_oof True
```
若本地有算力，直接本地跑；仅评分/提交走 Kaggle。

**无本地 GPU 时，推送至 Kaggle：**
```powershell
python tools/push_training.py --exp EXP000 --child child-exp000 --competition <slug>
```

- 冒烟测试(不占 GPU 配额):加 `--no-gpu`
- 比赛数据 >20GB 内核挂载会失败:加 `--no-competition`(不挂载比赛数据),
  需要的数据用 `--data-source <user/dataset>` 挂载或在内核内下载
- 查看状态:`kaggle kernels status <你的用户名>/exp000-child-exp000`
- 训练完成后内核会自动把模型+OOF+results.json 上传为 Dataset

## 第 3 步:拉结果 → 分析 → 记忆

```powershell
python tools/pull_results.py --exp EXP000 --child child-exp000
```

让 agent:读 `EXP/EXP000/outputs/child-exp000/results.json` 和
`oof_predictions.csv` → 分析 CV/LB gap → 更新 `EXP/EXP_SUMMARY.md`。

## 第 4 步:提交

**本地有算力时：**
```powershell
python EXP/EXP000/infer.py --config EXP/EXP000/config/child-exp000.yaml --submission outputs/submission.csv
kaggle competitions submit -c <slug> -f outputs/submission.csv -m "EXP000 child000"
```

**无本地 GPU 时：**
```powershell
python tools/push_inference.py --exp EXP000 --child child-exp000 --competition <slug>
```

同样支持 `--no-competition` / `--data-source`(结果 Dataset 会自动挂载)。

内核跑完生成 submission.csv,在 Kaggle 上提交(或内核内 `kaggle competitions submit`),
拿到 LB 后回填 EXP_SUMMARY.md。

## 第 5 步:迭代纪律(照抄金牌选手的教训)

- 新想法先 grep EXP_SUMMARY.md,失败过的不要再试
- 大改动开新 EXP(改 train.py),小改动只加 child-exp yaml
- 默认 augmentation 别乱动;CV 高不代表 LB 高;OOF 优化的 ensemble 常不迁移
- 模型收敛后停止折腾,把时间留给集成或后处理

## 资源限制(必须遵守)

- GPU 每周 30 小时,单内核最长 12 小时,最多 2 个 GPU 会话
- 调试一律 --no-gpu;每个 12h 大训练前先短跑验证
- Kaggle 内核自带你的 kaggle 凭据(无需再配)

## 目录速览

```
AGENTS.md / CLAUDE.md     护栏文件(agent 自动读取)
EXP/EXP{NNN}/             两级实验结构
EXP/EXP_SUMMARY.md        实验记忆表
docs/                     比赛上下文
templates/                内核模板(勿手改,由 tools 渲染)
tools/                    push_training / push_inference / pull_results
output/CV_LB/             CV-LB 分析输出
```
