# 204-Metric评价指标及损失函数-Error系列之均方根对数误差（Root Mean Square Log Error，RMSLE）

[类似的参考知乎专栏](https://www.zhihu.com/people/shi-jie-shi-wo-gai-bian-de/posts)

[回归算法的4个常用指标](https://zhuanlan.zhihu.com/p/409173995)

今天带来的内容是**Error**系列的指标及loss损失函数，该系列有：

- **均方误差（Mean Square Error，MSE）**
- **平均绝对误差（Mean Absolute Error，MAE）**
- **均方根误差（Root Mean Square Error，RMSE）**
- **均方对数误差（Mean Squared Log Error）**
- **平均相对误差（Mean Relative Error，MAE）**
- **均方根对数误差（Root Mean Squared Log Error）**

这次讲一下**均方根对数误差（Root Mean Squared Log Error）**的原理介绍。

## 一. Root Mean Squared Log Error **介绍**

均方根对数误差指的就是模型预测值 f(x) 与样本真实值 y ，各自取对数，再做他们之间的距离差，再取平方的平均值，取结果后再开方。其公式如下所示：
$$
RMSLE=\sqrt{\frac{\sum{(log{\frac{f(x_i)+1}{y_i+1}})^2}}{m}}
$$
其中，$y_i$ 和 $f(x_i)$ 分别表示第 $i$ 个样本的真实值和预测值，$M$ 为样本个数。

如果想对error的惩罚控制的时候，就用这个指标。它主要用来[低估]error。

例如，假设 y=1。如果我们的模型给出 y-hat = 0，则误差为 [log(1/2)]² = 0.09。但是，如果模型给出 y-hat=2，则误差为 [log(3/2)]² = 0.03。如果我们使用 MSE，那么无论哪种方式，我们的错误都是 1。

此外，该指标考虑了真实值和预测值的相对比例。例如，如果 y=9 和 y-hat=99，则误差为 1。另一方面，如果 y=99 和 y-hat=999，则误差仍然为 1。

## 使用 RMSLE 的优点

1. RMSLE 惩罚**欠预测**大于过预测，适用于某些需要欠预测损失更大的场景，如预测共享单车需求。

   假如真实值为 1000，若预测值为 600，那么 RMSE=400， RMSLE=0.510
   假如真实值为 1000，若预测值为 1400， 那么 RMSE=400， RMSLE=0.336

   可以看出来在 RMSE 相同的情况下，预测值比真实值小这种情况的 RMSLE 比较大，即对于预测值小这种情况惩罚较大。

2. 如果预测的值的范围很大，RMSE 会被一些大的值主导。这样即使你很多小的值预测准了，但是有一个非常大的值预测的不准确，RMSE 就会很大。 相应的，如果另外一个比较差的算法对这一个大的值准确一些，但是很多小的值都有偏差，可能 RMSE 会比前一个小。先取 log 再求 RMSE，可以稍微解决这个问题。RMSE 一般对于**固定的平均分布的预测值**才合理。
   ————————————————
   版权声明：本文为CSDN博主「AI 开发者」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
   原文链接：https://blog.csdn.net/qq_24671941/article/details/95868747