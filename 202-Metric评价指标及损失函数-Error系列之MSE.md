# 202-Metric评价指标及损失函数-Error系列之均方误差（Mean Square Error，MSE）

今天带来的内容是**Error**系列的指标及loss损失函数，该系列有：

- **均方误差（Mean Square Error，MSE）**
- **平均绝对误差（Mean Absolute Error，MAE）**
- **均方根误差（Root Mean Square Error，RMSE）**
- **均方对数误差（Mean Squared Log Error）**
- **平均相对误差（Mean Relative Error，MAE）**

今天就先讲一下**Mean Squared Error 均方误差**的原理介绍及MindSpore的实现代码。

https://zhuanlan.zhihu.com/p/353075180

## 一. Mean Squared Error **介绍**

均方误差指的就是模型预测值 f(x) 与样本真实值 y 之间距离平方的平均值。其公式如下所示：
$$
MSE=\frac{1}{m}\sum_{i=1}^{m}{(y_i-f(x_i))^2}
$$
其中，$y_i$ 和 $f(x_i)$ 分别表示第 $i$ 个样本的真实值和预测值，$M$ 为样本个数。

为了简化讨论，忽略下标 $i，m = 1$，以 $y-f(x) $为横坐标，MSE 为纵坐标，绘制其损失函数的图形：

![image-20221106214949905](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106214949905.png)

MSE 曲线的特点是光滑连续、可导，便于使用梯度下降算法，是比较常用的一种损失函数。而且，MSE 随着误差的减小，梯度也在减小，这有利于函数的收敛，即使固定学习因子，函数也能较快取得最小值。

平方误差有个特性，就是当 $y_i $与 $f(xi) $差值大于 1 时，会增大其误差；当$ y_i $与 $f(x_i)$ 的差值小于 1 时，会减小其误差。这是由平方的特性决定的。也就是说， MSE 会对误差较大（>1）的情况给予更大的惩罚，对误差较小（<1）的情况给予更小的惩罚。从训练的角度来看，模型会更加偏向于惩罚较大的点，赋予其更大的权重。

如果样本中存在离群点，MSE 会给离群点赋予更高的权重，但是却是以牺牲其他正常数据点的预测效果为代价，这最终会降低模型的整体性能。我们来看一下使用 MSE 解决含有离群点的回归模型。（我们先用numpy来举一个例子，matplotlib画一下图来说明一下）

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 20, 40)
y = x + [np.random.choice(4) for _ in range(40)]
y[-5:] -= 8
X = np.vstack((np.ones_like(x),x))    # 引入常数项 1
m = X.shape[1]
# 参数初始化
W = np.zeros((1,2))

# 迭代训练
num_iter = 20
lr = 0.01
J = []
for i in range(num_iter):
   y_pred = W.dot(X)
   loss = 1/(2*m) * np.sum((y-y_pred)**2)
   J.append(loss)
   W = W + lr * 1/m * (y-y_pred).dot(X.T)

# 作图
y1 = W[0,0] + W[0,1]*1
y2 = W[0,0] + W[0,1]*20
plt.scatter(x, y)
plt.plot([1,20],[y1,y2])
plt.show()
```

拟合结果如下图所示：

![image-20221106220853650](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106220853650.png)

可见，使用 MSE 损失函数，受离群点的影响较大，虽然样本中只有 5 个离群点，但是拟合的直线还是比较偏向于离群点。这往往是我们不希望看到的。

****

对于以下部分， y 是真实值，y-hat 是预测值，n 是测试实例的数量，i从 1 到 n。此外，所有指标都在一个测试集上进行评估。

像 MAE 一样，当我们对每个计算误差进行平方时，我们正在破坏方向信息。MSE 也总是大于或等于 0。但是，我们现在能够区分上面的两个模型。异常值能够辨别。

![image-20221130234727044](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130234727044.png)

MSE 与偏差-方差权衡有关。可以证明给定测试点的预期测试 MSE 可以写为：

![image-20221130234810758](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130234810758.png)

其中 0 下标是测试数据点的索引，ϵ 是数据中的噪声。

方差是指当我们改变训练集时 y-hat 的变化量。通常，更灵活的方法具有更高的方差。方差还取决于我们拥有多少数据。训练数据集越大，方差越小。因此，我们可以在**海量数据**的限制下，根据模型的偏差和随机噪声来解释给定测试数据点的 MSE。

当我们尝试用更简单的方法估计预测变量和目标之间的复杂关系时，就会出现**偏差**。例如，我们经常假设 x 和 y 具有线性或多项式关系，因为我们知道这些方程的形式，这将问题简化为我们可以估计的几个参数。实际上，x 和 y 可能**没有这种关系**。

最后，MSE 的一个主要缺点是 y 的单位是平方，这意味着很容易误解**（放大）最终结果**。