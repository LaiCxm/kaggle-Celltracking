# EXP055 测试集评分分析

## 结果

EXP055 version 3 成功完成隐藏测试四视频推理，并在同一个内核中分别运行联合 Top-2
和 Top-3。Kaggle 最新提交记录为：

- submission ref：`56541122`
- 状态：`COMPLETE`
- 公开 LB：`0.943`

内核同时导出了：

- `submission_exp054_joint_top2.csv`
- `submission_exp054_joint_top3.csv`
- 最终提交 `submission.csv`

三份 CSV 的 SHA256 均为：

```text
ad5b398bbde6a6b6d2f45e387b2a04f04ab89a1ba9059f43f8c5bd9581972f78
```

因此 Top-2 和 Top-3 在这四个测试视频上产生了完全相同的最终提交图，不存在“Top-3
尚未评分、可能更高”的剩余不确定性。

## 与已有结果比较

| 方案 | 公开 LB | 说明 |
|---|---:|---|
| EXP018-A | **0.946** | 当前生产最佳，另一条 FOCUS3D 仲裁血缘 |
| EXP050 / EXP049 `cos060` | 0.943 | 后 ILP 单边晋级 |
| EXP055 joint Top-2 | 0.943 | 后 ILP 联合分裂选择 |
| EXP055 joint Top-3 | 0.943 | 与 Top-2 文件逐字节相同 |

EXP055 没有超过 EXP050，也没有接近 EXP018-A。此前 EXP054 在四个带标签视频上的
official CV 中，联合 Top-2/Top-3 也完全没有提高 division TP；测试集结果进一步说明，
这条后 ILP 联合选择路线没有可迁移收益。

## 最终判断

EXP055 的目标是一次性消耗评分机会比较 Top-2/Top-3，实验已经完成。两条分支的测试图
完全一致且 LB 均为 `0.943`，因此无需再为 Top-3 单独提交。

比赛只剩三天时，当前应停止继续扫描 Top-k、余弦门限或后 ILP 联合门。最终生产回退应
保留已验证的 `EXP018-A`（LB=`0.946`）；EXP049、EXP054、EXP055 作为 division 后处理
诊断记录保存，不再替代生产提交。
