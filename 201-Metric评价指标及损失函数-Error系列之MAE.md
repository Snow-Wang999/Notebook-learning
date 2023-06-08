# 201-Metric评价指标及损失函数-Error系列之平均绝对误差（Mean Absolute Error，MAE）



![image-20221107164248042](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221107164248042.png)

https://www.zhihu.com/people/shi-jie-shi-wo-gai-bian-de/posts

参考知乎专栏，地址如上。

公式参考：

https://blog.csdn.net/bingxuesiyang/article/details/88949130

今天带来的内容是**Error**系列的指标及loss损失函数，该系列有：

- **均方误差（Mean Square Error，MSE）**
- **平均绝对误差（Mean Absolute Error，MAE）**
- **均方根误差（Root Mean Square Error，RMSE）**
- **均方对数误差（Mean Squared Log Error）**
- **平均相对误差（Mean Relative Error，MAE）**

这次讲一下**平均绝对误差（Mean Absolute Error，MAE）**的原理介绍及MindSpore的实现代码。

https://zhuanlan.zhihu.com/p/353125247

## 一. 平均绝对误差（Mean Absolute Error，MAE）介绍

平均绝对误差指的就是模型预测值 f(x) 与样本真实值 y 之间距离的平均值。即：
$$
\Delta=\frac{|\Delta 1|+|\Delta 2|+...+|\Delta n|}{n}
$$
其公式如下所示：
$$
MAE	= \frac{1}{m}\sum_{i=1}^{m}{|y_i-f(x_i)|}
$$
为了简化讨论，忽略下标 $i，m = 1$，以 $y-f(x)$ 为横坐标，MAE 为纵坐标，绘制其损失函数的图形：

![image-20221106180807274](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106180807274.png)

直观上来看，MAE 的曲线呈 V 字型，连续但在 $y-f(x)=0$ 处不可导，计算机求解导数比较困难。而且 MAE 大部分情况下梯度都是相等的，这意味着即使对于小的损失值，其梯度也是大的。这不利于函数的收敛和模型的学习。



值得一提的是，MAE 相比 MSE 有个优点就是 MAE **对离群点不那么敏感**，更有包容性。因为 MAE 计算的是误差 $y-f(x)$ 的绝对值，无论是 $y-f(x)>1$ 还是 $y-f(x)<1$，没有平方项的作用，惩罚力度都是一样的，所占权重一样。针对 MSE 中的例子，我们来使用 MAE 进行求解，看下拟合直线有什么不同。（我们先用numpy举一下例子。）

```python
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
   loss = 1/m * np.sum(np.abs(y-y_pred))
   J.append(loss)
   mask = (y-y_pred).copy()
   mask[y-y_pred > 0] = 1
   mask[mask <= 0] = -1
   W = W + lr * 1/m * mask.dot(X.T)

# 作图
y1 = W[0,0] + W[0,1]*1
y2 = W[0,0] + W[0,1]*20
plt.scatter(x, y)
plt.plot([1,20],[y1,y2],'r--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('MAE')
plt.show()
```

注意上述代码中对 MAE 计算梯度的部分。

拟合结果如下图所示：

![image-20221106181106404](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106181106404.png)

显然，使用 MAE 损失函数，受离群点的影响较小，拟合直线能够较好地表征正常数据的分布情况。这一点，MAE 要优于 MSE。二者的对比图如下：

![image-20221106181123010](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106181123010.png)

- **选择 MSE 还是 MAE 呢？**

  实际应用中，我们应该选择 MSE 还是 MAE 呢？

  - 求解梯度的复杂度:

    从计算机求解梯度的复杂度来说，MSE 要优于 MAE，而且梯度也是动态变化的，能较快准确达到收敛。

  - 离群点的重要性：

    但是从离群点角度来看，如果离群点是实际数据或重要数据，而且是应该被检测到的异常值，那么我们应该使用MSE。另一方面，离群点仅仅代表数据损坏或者错误采样，无须给予过多关注，那么我们应该选择MAE作为损失。
    
  - 预测值误差的实际情况：
  
    平均绝对误差与平均误差相比，平均绝对误差由于离差被绝对值化，不会出现正负相抵消的情况，因而，平均绝对误差能更好地反映预测值误差的实际情况。

平均绝对误差是一个非常直观的指标。它是预测值和真实值之间的平均距离。为了**避免错误相互抵消**，取计算绝对值。最好的模型通常是 MAE 最低的模型。

尽管解释起来很简单，但 MAE 有一些缺点。它**不会告诉模型是否倾向于高估或低估**，因为取绝对值会破坏任何方向信息。此外，该指标可能**对异常值不敏感**。看看下面的例子。

![image-20221130234637641](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130234637641.png)