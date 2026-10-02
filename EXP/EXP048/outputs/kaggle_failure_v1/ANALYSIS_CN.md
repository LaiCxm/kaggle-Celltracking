# EXP048 version 1 失败分析

## 失败位置

内核在完成依赖安装、双种子权重装载和 CUDA 检查后，于第 5 个代码单元退出，尚未开始四视频验证推理，也没有产生 official CV。

```text
NameError: name '_s' is not defined
```

## 根因

确定性补丁需要修改 `predict_unet_transformer.py`，但在读取该文件之前就调用了 `_s.count(...)`。原静态测试人为向补丁作用域注入了 `_s`，因此掩盖了 Notebook 真实执行顺序中的未定义变量。

## 修复

- 在确定性补丁内部显式执行 `_s = _ps.read_text()`；
- 验证器不再预先注入 `_s`，确保补丁自身完成状态初始化；
- 重新编译 Notebook 的全部代码单元，并检查读取语句位于锚点匹配之前。

该失败属于 Notebook 工程错误，不构成算法或确定性方案的实验结果。
