# 21-线性回归和随机梯度下降（Linear regression and random gradient descent）

**在 Python 3.6 中测试正确性:**

- numpy 1.15.4
- pandas 0.23.4
- 不一定要一致，能运行即可

您将根据公司在电视、报纸和广播广告方面的投资来预测公司的收入。

你将学习：

- 解决线性回归恢复问题
- 实现随机梯度下降来调整它
- 解析地解决线性回归问题

## 介绍

### 线性回归（Linear regression）

线性回归是研究最充分的机器学习方法之一，它允许您将**定量特征的值预测为其他特征**与参数（模型权重）的线性组合。最佳（在某些误差函数的最小意义上）线性回归参数可以使用**正规方程**进行分析或使用**优化方法**在数值上找到。

线性回归使用一个简单的质量函数——**标准误差**。我们将使用包含 3 个特征的样本。调整模型的参数（权重），解决以下问题：
$$
\Large \frac{1}{\ell}\sum_{i=1}^\ell{{((w_0 + w_1x_{i1} + w_2x_{i2} +  w_3x_{i3}) - y_i)}^2} \rightarrow \min_{w_0, w_1, w_2, w_3},
$$

$$
其中x_{i1}, x_{i2}, x_{i3}是第 i-го个对象的特征值，y_i是第 i-го个对象的目标特征值，\ell是训练集中的对象个数。
$$

### 梯度下降（gradient descent）

$$
参数 w_0, w_1, w_2, w_3 可以使用梯度下降在数值上找到均方根误差最小化的参数。
$$

权重的梯度步骤如下所示：
$$
\Large w_0 \leftarrow w_0 - \frac{2\eta}{\ell} \sum_{i=1}^\ell{{((w_0 + w_1x_{i1} + w_2x_{i2} +  w_3x_{i3}) - y_i)}}
$$

$$
\Large w_j \leftarrow w_j - \frac{2\eta}{\ell} \sum_{i=1}^\ell{{x_{ij}((w_0 + w_1x_{i1} + w_2x_{i2} +  w_3x_{i3}) - y_i)}},\ j \in \{1,2,3\}
$$

这里𝜂是一个参数，是梯度下降步数。

### 随机梯度下降（random gradient descent）

如上所述，梯度下降的问题在于，在大样本上，在每一步计算所有可用数据的梯度可能在计算上非常困难。

在梯度下降的随机变体中，仅考虑训练样本的一个随机对象来计算权重的校正：
$$
\Large w_0 \leftarrow w_0 - \frac{2\eta}{\ell} {((w_0 + w_1x_{k1} + w_2x_{k2} +  w_3x_{k3}) - y_k)}
$$

$$
\Large w_j \leftarrow w_j - \frac{2\eta}{\ell} {x_{kj}((w_0 + w_1x_{k1} + w_2x_{k2} +  w_3x_{k3}) - y_k)},\ j \in \{1,2,3\},
$$

$$
其中 k- 随机索引, k \in \{1, \ldots, \ell\}.
$$

### 正规方程（normal_equation）

找到最佳权重向量 𝑤 也可以通过解析来完成。我们希望找到这样一个权重向量 𝑤 ，以便通过将矩阵 𝑋（由训练样本对象的除目标对象之外的所有特征组成）乘以权重向量 𝑤 来获得逼近目标特征的向量 𝑦。即满足矩阵方程：
$$
\Large y = Xw
$$
在左边乘以 $X^T$ ，
$$
\Large X^Ty = X^TXw
$$
这很好，因为现在矩阵𝑋𝑇𝑋 是正方形的，并且可以找到解（向量𝑤）：
$$
\Large w = {(X^TX)}^{-1}X^Ty
$$
$ (𝑋𝑇𝑋)−1𝑋𝑇 $ - 矩阵X的伪逆. 在 NumPy 中，可以使用函数计算这样的矩阵[numpy.linalg.pinv](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.pinv.html).

然而，在矩阵𝑋（多重共线性问题）的行列式很小的情况下，求伪逆矩阵是一个计算复杂且不稳定的操作。在实践中，最好通过求解矩阵方程来找到权向量𝑤
$$
\Large X^TXw = X^Ty
$$
这可以通过 [numpy.linalg.solve](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.solve.html) 函数来完成。

但实际上，对于大型矩阵𝑋，梯度下降的工作速度更快，尤其是它的随机版本。

## 执行说明

[**task_1**](https://github.com/RBVV23/Coursera/blob/21fa00d145e3ec0e26c1a617091000b7e9548e00/%D0%9E%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BD%D0%B0%20%D1%80%D0%B0%D0%B7%D0%BC%D0%B5%D1%87%D0%B5%D0%BD%D0%BD%D1%8B%D1%85%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85/Week_1/Project_1/task_1.py)

[第 2 课。特征缩放。正则化。随机梯度下降。](https://github.com/mahhets/Data_analysis_algs/blob/21d58a6884af1c7de7f900ca164fd5a5de897686/2_Scalers_L1_L2_StochasticGD/Lesson_2.ipynb)

### 1. 加载数据

将 ads.csv 文件中的数据加载到 pandas DataFrame 对象中。[数据源](http://www-bcf.usc.edu/~gareth/ISL/data.html)。

```python
import pandas as pd
adver_data = pd.read_csv('../input/advertising/advertising.csv')
adver_data.head(5)
```

![image-20221101125653822](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101125653822.png)

```python
adver_data.describe()
```

![image-20221101125707388](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101125707388.png)

#### 创建numpy和pandas数据

从 TV、Radio 和 Newspaper 列创建 NumPy 数组 X，从 Sales 列创建 y。使用 pandas DataFrame 对象的 values 属性。

```python
#X = adver_data[['TV','Radio','Newspaper']].values
#y = adver_data[['Sales']].values
#print (X[0])
#print (y[0])
```

```python
#X = adver_data.drop(columns=['Sales']).values
#y = adver_data["Sales"].values
# X choose first three lists and y choose last list
X_origin,y = adver_data.iloc[:,0:3].values, adver_data.iloc[:,3:].values
print("scale of features:",X_origin.shape)
print("scale of labels:",y.shape)
```

```
scale of features: (200, 3)
scale of labels: (200, 1)
```

#### 计算mean和std

通过从每个值中减去相应列的平均值并将结果除以标准偏差来缩放 X 矩阵的列。

为了具体起见，请使用 NumPy 向量的均值mean和标准差std方法（标准的熊猫实现可能会有所不同）。

请注意，在 numpy 中，调用不带参数的 .mean() 函数会返回数组所有元素的平均值，而不是 pandas 中列的平均值。要按列计算，您必须指定轴参数。

```python
#means, stds = np.mean(X,axis=0),np.std(X,axis=0)
#X =  (X - means)/stds
```

```python
# calculate  means and stds
data_nolabel = adver_data.drop(columns=['Sales'])
means, stds = data_nolabel.mean(axis = 0, skipna = True),data_nolabel.std(axis = 0, skipna = True)
print("means:",means)
print("stds:",stds)
```

```
means: TV           147.0425
Radio         23.2640
Newspaper     30.5540
dtype: float64
stds: TV           85.854236
Radio        14.846809
Newspaper    21.778621
dtype: float64
```

```python
X_origin = (X_origin - means.values)/stds.values
```

another way

```python
# 获取各列的均值和标准差
means = np.mean(X_origin, axis=0)
stds = np.std(X_origin, axis=0)
print("means:",means)
print("stds:",stds)
# axis参数指定按列计算值，而不是整个数组
#（参见源代码部分的文档）
# 从均值中减去每个特征均值并除以标准差
for i in range(X_origin.shape[0]):
    for j in range(X_origin.shape[1]):
        X_origin[i][j] = (X_origin[i][j] - means[j])/stds[j]

```

```
means: [147.0425  23.264   30.554 ]
stds: [85.63933176 14.80964564 21.72410606]
```



#### 添加单位向量列

使用 hstack、ones 和 reshape NumPy 方法向 X 矩阵添加一列。为了不单独处理线性回归的系数𝑤0，需要一个单位向量。

```python
import numpy as np
n,m = X_origin.shape
X_1 = np.ones((n,1))
print(X_1)
X = np.hstack((X_1,X_origin))
print(X)
print("scale of features:",X.shape)
```

![image-20221101130130392](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101130130392.png)

![image-20221101150937478](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101150937478.png)

```
scale of features: (200, 4)
```

### 2.实现函数mserror

[一种更简单的求最小平方均值函数（MSE)的方法 -- 梯度下降法。](https://blog.csdn.net/weixin_42342803/article/details/81366699)

预测的均方根误差。它有两个参数 - 系列对象 y（目标特征值）和 y_pred（预测值）。不要在这个函数中使用循环——那么它的计算效率会很低。

```python
'''
#也可行
def mserror(y, y_pred):
   #return np.sqrt(((y_pred - y) ** 2).mean())
   #return(sum((y - y_pred)**2)[0])/float(y.shape[0])
   return sum((y - y_pred)**2,0)/y.shape[0]
'''
# y.shape[0] 行数
```

```python
#此处不用
def mserror_1(X, w, y_pred):
    y = X.dot(w)
    return (sum((y - y_pred)**2)) / len(y)
```

```python
def mserror_2(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    return np.mean((y - y_pred)**2)
```

如果总是预测原始样本的 Sales 中值，那么预测 Sales 值的标准误是多少？结果，四舍五入到小数点后 3 位，是“1 个任务”的答案。

median as y_pred

```python
'''也可行
eye = np.array([np.median(y)]*y.shape[0]).reshape((y.shape[0], 1))# the median of the y multiply the number of the y
answer1 = mserror(y, eye)
print(np.round(answer1, 3))
#write_answer_to_file(answer1, '1.txt')
'''
```

```python
N = X.shape[0]
med = np.median(np.array(adver_data['Sales']))
y_pred = np.ones((N))*med
y = np.array(adver_data['Sales'])
# print(y_pred, y)
answer1 =  mserror_2(y, y_pred)
print('\tanswer 1 = ', round(answer1, 3))
```

```
answer 1 =  28.346
```

### 3. 实现 normal_equation 函数

给定矩阵（NumPy 数组）X 和 y，根据正态线性回归方程计算权重向量 𝑤。
$$
\large Xθ=Y (3.1)
$$
![image-20221101113822367](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101113822367.png)

![image-20221101113838920](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101113838920.png)

![image-20221101113849556](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101113849556.png)

![image-20221101113858982](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101113858982.png)

```python
#Least Squares
def normal_equation_1(X, y):
    X_t = X.transpose()
    X_obr = np.dot(X_t,X)
    X_obr = np.linalg.inv(X_obr)
    Sol = np.dot(X_obr, X_t)
    return np.dot(Sol, y)
```



```python
'''
#也可行
#Least Squares
def normal_equation(X, y):
    return np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y) 
	#np.dot(X.T, X)= X的转置与X的点积 = X^T*X
    #np.linalg.pinv(np.dot(X.T, X)) = X的转置与X的点积的伪逆矩阵 = (X^T*X)^(-1)
    #np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T) = (X^T*X)^(-1)*X^T
    #np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y) = ((X^T*X)^(-1)*X^T )*y=weight
'''
```

```python
norm_eq_weights_1= normal_equation_1(X, y)
print(norm_eq_weights_1)
```

```
[[14.0225    ]
 [ 3.91925365]
 [ 2.79206274]
 [-0.02253861]]
```

在电视、广播和报纸广告的平均投资情况下，使用正态方程找到权重的线性模型预测的销售额是多少？ （即，缩放的 TV、Radio 和 Newspaper 特征的值为零）。得到的结果，四舍五入到小数点后 3 位，是“2 个任务”的答案。

```python
X_0 = np.array([1, 0, 0, 0])
answer2 = np.dot(X_0,norm_eq_weights_1)
print('\tanswer 2 = ', np.round(answer2, 3))
```

```
answer 2 =  [14.022]
```

```python
'''也可行
answer2 = np.dot(np.mean(X, axis=0), norm_eq_weights)[0]#X的每行平均值与权重进行点积
print(np.round(answer2, 3))
'''
```

```
14.022
```

### 4.linear_prediction 函数

编写一个 linear_prediction 函数，它以矩阵 X 和线性模型的权重向量 w 作为输入，并返回一个预测向量作为矩阵 X 的列与权重 w 的线性组合。

```python
def linear_prediction(X, w):
    return np.dot(X, w)
```

使用正规方程找到权重的线性模型预测销售额的标准误差是多少？结果，四舍五入到小数点后 3 位，是“问题 3”的答案。

```python
y_pred = linear_prediction(X, norm_eq_weights_1)
```

```
answer3 = mserror_2(y,y_pred)
print('\tanswer 3 = ', np.round(answer3, 3))
```

```
answer 3 =  2.784
```

### 5.stochastic_gradient_step 函数

编写一个 stochastic_gradient_step 函数，实现线性回归的随机梯度下降步骤。该函数必须接受一个矩阵 X，向量 y 和 w，数字 train_ind 是训练样本对象的索引（矩阵 X 的行），通过它计算权重的变化，数字 𝜂 (eta) 是梯度下降步骤（默认 eta=0.01）。结果将是一个更新权重的向量。我们的函数实现将针对具有 3 个特征的数据显式编写，但是对于任意数量的特征很容易修改，你可以做到。

```python
def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    res = w[0]*X[train_ind,0]+w[1]*X[train_ind,1]+w[2]*X[train_ind,2]+w[3]*X[train_ind,3]
    grad0 = X[train_ind,0]*(res-y[train_ind])
    grad1 = X[train_ind,1]*(res-y[train_ind])
    grad2 = X[train_ind,2]*(res-y[train_ind])
    grad3 = X[train_ind,3]*(res-y[train_ind])
    return  w - 2*eta * np.array([grad0, grad1, grad2, grad3])
```

或者

```python
def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    res = X[train_ind][0]*w[0] + X[train_ind][1]*w[1] + X[train_ind][2]*w[2] + X[train_ind][3]*w[3]
    grad0 =  ( res - y[train_ind] ) * X[train_ind][0]
    grad1 =  ( res - y[train_ind] ) * X[train_ind][1]
    grad2 =  ( res - y[train_ind] ) * X[train_ind][2]
    grad3 =  ( res - y[train_ind] ) * X[train_ind][3]
    return  w - 2*eta * np.array([grad0, grad1, grad2, grad3])
```

### 6.有参数的线性回归的随机梯度下降

编写一个 stochastic_gradient_descent 函数，实现线性回归的随机梯度下降。该函数将以下参数作为输入：

- X - 对应于训练样本的矩阵 
- y - 目标特征的值向量 
- w_init - 模型初始权重的向量 
- eta - 梯度下降步骤（默认 0.01） 
- max_iter - 梯度下降迭代的最大次数（默认 10000） 
- min_weight_dist - 算法停止运行的相邻梯度下降迭代中权重向量之间的最大欧几里得距离（默认 1e-8） 
- seed - 用于生成伪随机数的可重复性数字（默认 42） 
- verbose - 用于打印信息的标志（例如，用于调试，默认为 False）

在每次迭代中，均方根误差的当前值必须写入向量（列表）。该函数必须返回权重向量 𝑤 以及错误向量（列表）。

1e4=10000

```python
def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42, verbose=False): 
    # 初始化相邻权重向量之间的距离
	# 大量迭代。
    weight_dist = np.inf
    # 初始化权重向量
    w = w_init
    # 在这里，我们将记录每个迭代的错误
    errors = []
    # 迭代计数器
    iter_num = 0
    # 生成伪随机数
	# （要更改权重的对象编号）
	# seed使用此伪随机数序列。
    np.random.seed(seed)
        
    # 主循环
    while weight_dist > min_weight_dist and iter_num < max_iter:
        #制造伪随机
		#学习样本对象索引
        random_ind = np.random.randint(X.shape[0])
        
        # Ваш код здесь
        iter_num += 1
        # 更新权重
        w_new = stochastic_gradient_step(X=X, y=y, w=w, train_ind=random_ind, eta=eta)
        #calculate distance between old weight and new weight
        weight_dist = (sum((w - w_new) ** 2)) ** 0.5
        # 预测值等于x乘以w_new
        y_pred = linear_prediction(X, w_new)
        # 计算均方差，与记录
        error = mserror(y, y_pred)
        errors.append(error)
        w = w_new
        if (iter_num % 100) == 0 and verbose == True:
            print('iter_num = ', iter_num)
            print('\tweight_dist = ', weight_dist)
            print('\terror = ', error)
            print('\trandom_ind = ', random_ind)
    if verbose == True:
        print('w = ', w)
        print('errors[0] = ', errors[0])
        print('errors[-1] = ', errors[-1])
        print('mean.errors[-1] = ', np.mean(errors))

    return w, errors
```

运行 10^5 次随机梯度下降迭代。指定由零组成的初始 w_init 权重向量。将 eta 和 seed 参数保留为默认值（eta=0.01，seed=42 - 这对于检查答案很重要）。

```python
w_init = np.array([0, 0, 0, 0])
%%time
stoch_grad_desc_weights, stoch_errors_by_iter = stochastic_gradient_descent(
    X, y, w_init, eta=0.01,
    max_iter=1e5, 
    min_weight_dist=1e-8,
    seed=42, 
    verbose=False)
```

```
CPU times: user 7.75 s, sys: 30.9 ms, total: 7.78 s
Wall time: 7.78 s
```

让我们看看随机梯度下降的前 50 次迭代的误差是多少。我们看到误差不一定会在每次迭代中减少。

```python
%pylab inline
plot(range(50), stoch_errors_by_iter[:50])
xlabel('Iteration number')
ylabel('MSE')
```

![image-20221101153146409](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101153146409.png)

现在让我们看一下随机梯度下降的 10^5 次迭代的误差对迭代次数的依赖性。我们看到算法收敛了。

```python
%pylab inline
plot(range(1000), stoch_errors_by_iter[:1000])
xlabel('Iteration number')
ylabel('MSE')
```

![image-20221101153247956](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101153247956.png)

```python
%pylab inline
plot(range(len(stoch_errors_by_iter)), stoch_errors_by_iter)
xlabel('Iteration number')
ylabel('MSE')
```

![image-20221101153219463](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101153219463.png)

让我们看一下该方法收敛到的权重向量。

```python
stoch_grad_desc_weights
```

```
array([[13.97836994, 13.97836994, 13.97836994, 13.97836994],
       [ 3.87934503,  3.87934503,  3.87934503,  3.87934503],
       [ 3.14134212,  3.14134212,  3.14134212,  3.14134212],
       [ 0.18323907,  0.18323907,  0.18323907,  0.18323907]])
```

让我们看看最后一次迭代的均方误差。

```python
stoch_errors_by_iter[-1]
```

```
array([3.00045025, 3.00045025, 3.00045025, 3.00045025])
```

```python
print('stoch_grad_desc_weights = ', stoch_grad_desc_weights)
print('stoch_errors_by_iter[-1] = ', stoch_errors_by_iter[-1])
```

```
stoch_grad_desc_weights =  [[13.97836994 13.97836994 13.97836994 13.97836994]
 [ 3.87934503  3.87934503  3.87934503  3.87934503]
 [ 3.14134212  3.14134212  3.14134212  3.14134212]
 [ 0.18323907  0.18323907  0.18323907  0.18323907]]
stoch_errors_by_iter[-1] =  [3.00045025 3.00045025 3.00045025 3.00045025]
```

将 Sales 预测为使用梯度下降找到权重的线性模型的标准误差是多少？得到的结果，四舍五入到小数点后 3 位，是“任务 4”的答案。

```python
'''也可以
answer4 = mserror_2(y, linear_prediction(X, stoch_grad_desc_weights))
print(np.round(answer4, 3))
'''
```

```
y_pred = linear_prediction(X, stoch_grad_desc_weights)
#y = np.array(adver_data['Sales'])
answer4 = mserror_2(y, y_pred)
print('\tanswer 4 = ', round(answer4, 3))
```

```
answer 4 =  3.0
```