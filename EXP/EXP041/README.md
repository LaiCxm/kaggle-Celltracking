# EXP041：相对边优势门控的局部重连

## 目的

EXP040 的固定预算重连在官方 CV 中新增 division 误报。EXP041 只保留一种更强的替换证据：候选父节点 `parent` 到被占用的 `daughter2` 必须比当前父节点 `q` 更近至少指定距离。

```text
删除：q -> daughter2
新增：parent -> daughter2
```

## 预注册分支

| 分支 | parent-daughter 门限 | daughter-daughter 门限 | 新父节点相对距离优势 |
| --- | ---: | ---: | ---: |
| `base` | 不重连 | 不重连 | 不适用 |
| `dominance_1_14` | 14 um | 14 um | 至少 1 um |
| `dominance_2_14` | 14 um | 14 um | 至少 2 um |
| `dominance_4_14` | 14 um | 14 um | 至少 4 um |

共同约束：下一帧发散增量至少 2.25 um；每帧最多 1 次、每视频最多 6 次；严格一删一加；不新增节点、不使用 GT。

## 成功标准

主要以 4 个完整视频的官方 patched scorer 总分判断。辅助检查普通 edge 和 division 的 TP/FP/FN，重点防止以普通边小幅增加换取 division FP。候选数和局部排序不作为上线依据。

本 Notebook 仅运行官方 CV，不生成 `submission.csv`。

