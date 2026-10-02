# EXP051：公开 0.947+ 方案的官方 CV 审计

## 来源

- Kaggle Notebook：`sjlee101/biohub-lf-dctta`
- 原始版本：`348619543`
- 本地锁定入口：[CELL_public_0947_official_cv.ipynb](CELL_public_0947_official_cv.ipynb)
- 原始复现血缘：[EXP017](../EXP017/CELL_infer_public_0947_dctta.ipynb)
- Kaggle CV 内核：[laicxm/exp051-cell-public-0947-official-cv-ipynb-infer](https://www.kaggle.com/code/laicxm/exp051-cell-public-0947-official-cv-ipynb-infer)

## 实验目的

测量周报抓取的公开 `0.947` 候选在与 EXP049 完全相同的四个完整带标签视频上的官方 patched scorer 分数。该实验只回答“公开方案的官方 CV 是多少”，不做新调参。

## 固定条件

- 完整复用 EXP017 的冻结权重、检测、双向调和关联和全部后处理代码；
- 固定公开运行最终选择的 `MOTION_RELINK_TIGHT_UM=5.5`（tight55）；
- 固定四个验证视频：`44b6_12dfb391`、`44b6_267148e4`、`6bba_062c8d37`、`6bba_07e24132`；
- 四个视频共包含 5 个 GT division；
- 使用 `csv_to_geffs.py`、`evaluate.py/evaluate_pairs` 和 `metrics.summarise`；
- 不使用旧的 notebook proxy 公式；
- 不运行 FOCUS3D，不接入 EXP049 的 ILP 后 division 晋级，不生成 `submission.csv`，不扫描任何参数。

## 对比对象

EXP049 `cos060` 已在相同四个视频和官方 patched scorer 上得到 `0.9614675166`。只有 EXP051 完成后，才能把两个数字作为同口径验证集 CV 比较；二者都不是排行榜 LB。

## 运行修复记录

version 1 已完成四个验证视频的冻结模型推理，但官方评分脚本加载器将 `importlib.util` 的别名错误地再次访问 `.util`，在评分开始前报 `AttributeError`。version 2 已改为直接调用 `spec_from_file_location` 和 `module_from_spec`，相关静态测试通过并重新推送。
