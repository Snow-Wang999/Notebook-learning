# 203-Metric评价指标及损失函数-Error系列之均方根误差（Root Mean Square Error，RMSE）

https://www.zhihu.com/people/shi-jie-shi-wo-gai-bian-de/posts

参考知乎专栏，地址如上。

今天带来的内容是**Error**系列的指标及loss损失函数，该系列有：

- **均方误差（Mean Square Error，MSE）**
- **平均绝对误差（Mean Absolute Error，MAE）**
- **均方根误差（Root Mean Square Error，RMSE）**
- **均方对数误差（Mean Squared Log Error）**
- **平均相对误差（Mean Relative Error，MAE）**

这次讲一下**均方根误差（Root Mean Square Error，RMSE）**的原理介绍及MindSpore的实现代码。

## 一. Root Mean Squared Error **介绍**

均方根误差指的就是模型预测值 f(x) 与样本真实值 y 之间距离平方的平均值，取结果后再开方。

其公式如下所示：
$$
RMSE=\sqrt{\frac{1}{m}\sum_{i=1}^{m}{(y_i-f(x_i))^2}}
$$


其中，$y_i $和 $f(x_i)$ 分别表示第 $i $ 个样本的真实值和预测值，$M$ 为样本个数。

从公式中看出，RMSE的结果是基于MSE的。