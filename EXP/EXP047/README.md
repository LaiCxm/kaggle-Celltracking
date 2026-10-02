# EXP047：EXP046 召回关闭对照

EXP046 的 adjusted official score 高于 EXP019 参考，但 division 完全没有增加、普通 edge Jaccard 下降，存在节点数量调整项和运行差异的混杂。EXP047 完全复用 EXP046 的 Notebook、模型、官方 scorer、后处理和数据源，只把第二出边召回阈值设为超出概率范围的值，使召回规则关闭。

目标是获得与 EXP046 同一代码血缘下的干净关闭对照，判断 EXP046 的 `0.940344` 是否由召回规则造成。该实验不提交排行榜。
