
[1] [Relative position embedding](https://zhuanlan.zhihu.com/p/364828960)
transformer原生的PE是基于sin/cos函数的，在模型训练时输入序列长度有限，所以模型只能见到有限长度的Position Embedding，导致测试集里的样本长度远大于训练集中的普遍长度时，得到的位置编码是网络没见过的，因此网络会得到不鲁棒的结果。
相对位置编码：
