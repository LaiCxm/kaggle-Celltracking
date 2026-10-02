# EXP042：几何优势与边置信度联合门控

## 目的

EXP041 证明相对距离门控可以消除局部重连误报，但没有增加真实 division TP。EXP042 固定 EXP041 最佳的 1 um 距离优势，再加入已有边的模型置信度差：

```text
parent -> daughter1 的保留边置信度
    - q -> daughter2 的被替换边置信度
        >= 0 / 0.05 / 0.10
```

只有同时满足几何优势和边置信度优势，才允许删除 `q -> daughter2` 并新增 `parent -> daughter2`。

## 预注册分支

| 分支 | 距离优势 | 边置信度优势 |
| --- | ---: | ---: |
| `base` | 不重连 | 不适用 |
| `geom1_prob0` | 至少 1 um | 至少 0.00 |
| `geom1_prob05` | 至少 1 um | 至少 0.05 |
| `geom1_prob10` | 至少 1 um | 至少 0.10 |

共同约束：parent/daughter 门限 14 um，下一帧发散增量 2.25 um；每帧最多 1 次、每视频最多 6 次；严格一删一加；不新增节点、不使用 GT。

## 成功标准

主要以 4 个完整视频的官方 patched scorer 总分判断，同时检查 edge/division TP/FP/FN。候选数量和局部排序不作为上线依据。本 Notebook 仅运行官方 CV，不生成 `submission.csv`。

