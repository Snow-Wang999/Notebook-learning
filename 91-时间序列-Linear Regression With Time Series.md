# 91-时间序列-Linear Regression With Time Series

使用时间序列特有的两个特性：滞后（lags）和时间步长（time step）。

## 欢迎来到时间系列！ 

预测可能是机器学习在现实世界中最常见的应用。企业预测**产品需求**，政府预测**经济和人口增长**，气象学家预测**天气**。对未来事物的理解是科学、政府和工业界的迫切需求（更不用说我们的个人生活了！），这些领域的从业者越来越多地应用机器学习来满足这一需求。 

**时间序列预测**是一个具有悠久历史的广泛领域。本课程侧重于将现代机器学习方法应用于时间序列数据，以产生最准确的预测。本课程中的课程灵感来自过去Kaggle预测比赛中获胜的解决方案，但只要准确预测是优先事项，这些课程将适用。

完成本课程后，您将知道如何： 

- 对主要时间序列**组件**（趋势、季节和周期[ *trends*, *seasons*, and *cycles*]）进行建模的工程特征， 
- 用多种时间序列图**可视化**时间序列， 
- 创建结合互补模型优势的预测**混合**模型，以及 
- 使机器学习方法适应各种预测任务。

作为练习的一部分，您将有机会参加我们的商店销售-时间序列预测入门比赛（[Store Sales - Time Series Forecasting](https://www.kaggle.com/c/29781)）。在这场竞争中，您的任务是预测Corporatción Favorita（一家总部位于厄瓜多尔的大型杂货零售商）近1800种产品类别的销售额。

## 什么是时间序列？ 

预测的基本目标是时间序列，这是一组随时间记录的观测结果。在预测应用中，观测值通常以**常规频率记录**，如每日或每月。

```python
import pandas as pd

df = pd.read_csv(
    "../input/ts-course-data/book_sales.csv",
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)

df.head()
```

![image-20221203141304212](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203141304212.png)

该系列记录了零售店30天内的精装书`hardover`销量。请注意，我们有一列带有时间索引`Date`的观察结果`Hardcover`。

## 时间序列线性回归 

在本课程的第一部分，我们将使用线性回归算法构建预测模型。线性回归在实践中得到了广泛应用，甚至可以很自然地适应复杂的预测任务。 

线性回归算法学习如何根据其输入特征进行加权和。对于两个功能，我们将具有：

```python
target = weight_1 * feature_1 + weight_2 * feature_2 + bias
```

在训练期间，回归算法学习最适合目标的参数`weight_1`、`weight_2`和`bias`的值。（该算法通常被称为普通最小二乘法，因为它选择的值使目标和预测之间的平方误差最小化。）权重也称为回归系数，偏差也称为截距，因为它告诉您该函数的图形与y轴相交的位置。

### 时间步长特征(Time-step features) 

时间序列有两种独特的特征：时间步长特征和滞后特征。 

时间步长特征是我们可以直接从时间索引中导出的特征。最基本的时间步长特征是时间虚拟值(**time dummy**)，它从开始到结束对序列中的时间步长进行计数。

```python
import numpy as np

df['Time'] = np.arange(len(df.index))

df.head()
```

![image-20221203141512088](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203141512088.png)

使用时间虚拟的线性回归产生模型：

```python
target = weight * time + bias
```

然后，时间虚拟模型让我们将曲线拟合到时间图中的时间序列，其中时间形成了x轴。

```python
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
%config InlineBackend.figure_format = 'retina'

fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=df, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales');
```

![image-20221203141633206](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203141633206.png)

时间步长特性可以为时间依赖性(**time dependence**)建模。如果序列的值可以从其发生的时间预测，则序列是时间相关的。在精装销售系列中，我们可以预测本月晚些时候的销售通常高于本月早些时候的销售。

### 滞后特性(Lag features) 

为了形成滞后特征，我们将目标序列的观测值进行移位，以便它们看起来在稍后时间出现。在这里，我们创建了一个1步滞后特性，但也可以进行多步移位。

```python
df['Lag_1'] = df['Hardcover'].shift(1)
#在行方向上移动一个单位
df = df.reindex(columns=['Hardcover', 'Lag_1'])

df.head()
```

![image-20221203141831730](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203141831730.png)

---

[图解Pandas中的移动函数shift](https://blog.csdn.net/qq_25443541/article/details/120067270)

```python
DataFrame.shift(periods=1,freq=None, axis=0,fill_value=<no_default>)
```



- periods：表示移动的幅度，可正可负；默认值是1,1就表示移动一次。注意这里移动的都是数据，而索引是不移动的，移动之后没有对应值的，就赋值为NaN。

- freq：DateOffset, timedelta, or time rule string，可选参数，默认值为None，只适用于时间序列。如果这个参数存在，那么会按照参数值移动时间索引，而数据值没有发生变化。

- axis：表示按照哪个轴移动。axis=0表示index，横轴；axis=1表示columns，纵轴

- fill_value：表示当我们数据发生了移动之后，产生的缺失值用什么数据填充。如果是数值型的缺失值，用np.nan；如果是时间类型的缺失值，用NaT（not a time）

---

具有滞后特性的线性回归生成模型：

```python
target = weight * lag + bias
```

因此，滞后特征允许我们将曲线拟合到滞后图，其中，一系列中的每个观测值都与先前观测值进行比较。

```python
fig, ax = plt.subplots()
ax = sns.regplot(x='Lag_1', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_aspect('equal')
ax.set_title('Lag Plot of Hardcover Sales');
```

![image-20221203141918045](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203141918045.png)

从滞后图中可以看出，**一天的销售额（精装）与前一天的销售（滞后_1）相关。**当你看到这样的关系时，你就会知道滞后特性会很有用。 

更一般地，滞后特性允许您模拟串行依赖 (**serial dependence**)。当可以从先前的观测预测观测时，时间序列具有**序列相关性**。在精装销售中，我们可以预测，一天的高销售额通常意味着第二天的高销量。 

使机器学习算法适应时间序列问题在很大程度上是关于具有时间索引和滞后的特征工程。在大多数情况下，我们使用线性回归是为了简单，但无论您选择哪种算法进行预测，这些特性都会很有用。

## Example - Tunnel Traffic

隧道交通量是一个时间序列，描述了从2003年11月到2005年11月每天通过瑞士巴里格隧道的车辆数量。在本例中，我们将获得一些将线性回归应用于时间步长特征和滞后特征的实践。 

隐藏的单元格设置了一切。

```python
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

simplefilter("ignore")  # ignore warnings to clean up output cells

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
%config InlineBackend.figure_format = 'retina'


# Load Tunnel Traffic dataset
#如果数据包含日期列，还可以在读取时使用 parse_dates 定义日期列。
data_dir = Path("../input/ts-course-data")
tunnel = pd.read_csv(data_dir / "tunnel.csv", parse_dates=["Day"])

# Create a time series in Pandas by setting the index to a date
# column. We parsed "Day" as a date type by using `parse_dates` when
# loading the data.
#通过将索引设置为日期，在Pandas中创建时间序列 
tunnel = tunnel.set_index("Day")

# By default, Pandas creates a `DatetimeIndex` with dtype `Timestamp`
# (equivalent to `np.datetime64`, representing a time series as a
# sequence of measurements taken at single moments. A `PeriodIndex`,
# on the other hand, represents a time series as a sequence of
# quantities accumulated over periods of time. Periods are often
# easier to work with, so that's what we'll use in this course.
# .to_period 函数允许将日期转换为特定的时间间隔。
tunnel = tunnel.to_period()
tunnel.head()
```

![image-20221203142106534](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203142106534.png)

---

[pandas.read_csv() 处理 CSV 文件的 6 个有用参数](https://baijiahao.baidu.com/s?id=1737474727586728032&wfr=spider&for=pc)

**read_csv中的参数parse_dates**

如果数据包含日期列，还可以在读取时使用 parse_dates 定义日期列。 Pandas 将自动从指定的“日期”列推断日期格式。 我们将date传入parse_dates ， pandas 自动会将“date”列推断为日期 dtype。

通过将索引设置为日期，在Pandas中创建时间序列 

`tunnel = tunnel.set_index("Day")`

[pandas数据日期函数之date_range()、resample()与to_period()](https://blog.csdn.net/m0_69435474/article/details/124339573)

---

### 时间步长特性 

如果时间序列没有任何缺失的日期，我们可以通过计算序列的长度来创建一个时间伪。

```python
df = tunnel.copy()

df['Time'] = np.arange(len(tunnel.index))

df.head()
```

![image-20221203142144716](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203142144716.png)

拟合线性回归模型的程序遵循scikit学习的标准步骤。

```python
from sklearn.linear_model import LinearRegression

# Training data
X = df.loc[:, ['Time']]  # features
y = df.loc[:, 'NumVehicles']  # target

# Train the model
model = LinearRegression()
model.fit(X, y)

# Store the fitted values as a time series with the same time index as
# the training data
y_pred = pd.Series(model.predict(X), index=X.index)
```

实际创建的模型（近似值）为：

`Vehicles = 22.5 * Time + 98176`

绘制随时间变化的拟合值，向我们展示了如何将线性回归拟合到时间虚拟值，从而创建由该方程定义的趋势线。

```python
ax = y.plot(**plot_params)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot of Tunnel Traffic');
```

![image-20221203142248436](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203142248436.png)

### 滞后特性 

Pandas为我们提供了一种延迟序列的简单方法，即移位法。

```python
df['Lag_1'] = df['NumVehicles'].shift(1)
df.head()
```

![image-20221203142322944](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203142322944.png)

在创建滞后特征时，我们需要决定如何处理生成的缺失值。填充它们是一个选项，可能使用0.0或使用第一个已知值“回填”。相反，我们只删除缺失的值，确保同时删除相应日期的目标值。

```python
from sklearn.linear_model import LinearRegression

X = df.loc[:, ['Lag_1']]
X.dropna(inplace=True)  # drop missing values in the feature set
y = df.loc[:, 'NumVehicles']  # create the target
y, X = y.align(X, join='inner')  # drop corresponding values in target

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
```

滞后图显示了我们能够很好地拟合一天的车辆数量和前一天的数量之间的关系。

```python
fig, ax = plt.subplots()
ax.plot(X['Lag_1'], y, '.', color='0.25')
ax.plot(X['Lag_1'], y_pred)
ax.set_aspect('equal')
ax.set_ylabel('NumVehicles')
ax.set_xlabel('Lag_1')
ax.set_title('Lag Plot of Tunnel Traffic');
```

![image-20221203142409821](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203142409821.png)

从滞后特征的预测对于我们如何预测时间序列意味着什么？下面的时间图向我们展示了我们的预测现在如何响应最近一段时间的系列行为。

```python
ax = y.plot(**plot_params)
ax = y_pred.plot()
```

![image-20221203142451844](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203142451844.png)

最佳时间序列模型通常包括时间步长特征和滞后特征的组合。在接下来的几节课中，我们将学习如何使用本课中的特征作为起点，设计时间序列中最常见模式的特征建模。

### 轮到你了 

继续练习，您将使用本教程中学习的技巧开始预测门店销售额( [**forecasting Store Sales**](https://www.kaggle.com/kernels/fork/19615998))。

## Exercise: Linear Regression With Time Series

### 介绍 

运行此单元以设置所有内容！

```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.time_series.ex1 import *

# Setup notebook
from pathlib import Path
from learntools.time_series.style import *  # plot style settings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression


data_dir = Path('../input/ts-course-data/')
comp_dir = Path('../input/store-sales-time-series-forecasting')

book_sales = pd.read_csv(
    data_dir / 'book_sales.csv',
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)
book_sales['Time'] = np.arange(len(book_sales.index))
book_sales['Lag_1'] = book_sales['Hardcover'].shift(1)
book_sales = book_sales.reindex(columns=['Hardcover', 'Time', 'Lag_1'])

ar = pd.read_csv(data_dir / 'ar.csv')

dtype = {
    'store_nbr': 'category',
    'family': 'category',
    'sales': 'float32',
    'onpromotion': 'uint64',
}
store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    dtype=dtype,
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales = store_sales.set_index('date').to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family'], append=True)
average_sales = store_sales.groupby('date').mean()['sales']
```

---

线性回归比更复杂的算法有一个优势，那就是它创建的模型是可解释的——很容易解释每个特征对预测的贡献。在模型`target=weight*feature+bias`中，权重通过特征中每个单位的平均变化来告诉您目标的变化程度。 

运行下一个单元格，查看精装销售的线性回归。

```python
fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=book_sales, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=book_sales, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales');
```

![image-20221203180737827](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203180737827.png)

### 1） 用时间伪函数解释线性回归 

线性回归线的方程式为（近似）`Hardcover = 3.33 * Time + 150.5`。在6天内，您预计精装书的销量平均会有多大变化？在你考虑过之后，运行下一个单元格。

时间上的6步变化对应于精装销量的平均变化6*3.33=19.98。

---

解释回归系数可以帮助我们识别时间图中的序列相关性。考虑模型`target=weight*lag_1+error`，其中误差(error)是随机噪声，权重(weight)是介于-1和1之间的数字。在这种情况下，权重告诉您下一个时间步长与上一个时间步骤具有相同符号的可能性：权重接近1意味着目标可能与上一步骤具有相同的符号，而权重接近-1意味着目标很可能具有相反的符号。

### 2） 用滞后特征解释线性回归 

运行以下单元格以查看根据刚才描述的模型生成的两个系列。

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
ax1.plot(ar['ar1'])
ax1.set_title('Series 1')
ax2.plot(ar['ar2'])
ax2.set_title('Series 2');
```

![image-20221203181020796](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203181020796.png)

这些系列中的一个具有方程`target = 0.95 * lag_1 + error`，另一个具有等式`target = -0.95 * lag_1 + error`，仅通过滞后特征上的符号不同。你能说出每个级数对应的方程式吗？

系列1由`target = 0.95 * lag_1 + error`生成，系列2由`target = -0.95 * lag_1 + error`生成

---

现在我们将从商店销售-时间序列预测竞争数据开始。整个数据集包括从2013年到2017年，记录各种产品系列的商店销售额的近1800个系列。在本课中，我们只使用每天平均销售额的单个系列（`average_sales`）。

### 3） 适合时间步长特征 

完成下面的代码以创建一个线性回归模型，该模型具有一系列平均产品销售额的时间步长特征。目标位于名为“sales”的列中。

```python
from sklearn.linear_model import LinearRegression

df = average_sales.to_frame()

# YOUR CODE HERE: Create a time dummy
time = np.arange(len(df.index))

df['time'] = time 

# YOUR CODE HERE: Create training data
X = df.loc[:,['time']]  # features
y = df.loc[:,'sales']  # target

# Train the model
model = LinearRegression()
model.fit(X, y)

# Store the fitted values as a time series with the same time index as
# the training data
y_pred = pd.Series(model.predict(X), index=X.index)

```

如果您想查看结果的绘图，请运行此单元格。

```python
ax = y.plot(**plot_params, alpha=0.5)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot of Total Store Sales');
```

![image-20221203181836857](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203181836857.png)

### 4） 为门店销售设置滞后功能 

完成下面的代码以创建一个线性回归模型，该模型具有一系列平均产品销售的滞后特性。目标位于名为“sales”的df列中。

```python
df = average_sales.to_frame()

# YOUR CODE HERE: Create a lag feature from the target 'sales'
lag_1 = df['sales'].shift(1)

df['lag_1'] = lag_1  # add to dataframe

X = df.loc[:, ['lag_1']].dropna()  # features
y = df.loc[:, 'sales']  # target
y, X = y.align(X, join='inner')  # drop corresponding values in target

# YOUR CODE HERE: Create a LinearRegression instance and fit it to X and y.
model = LinearRegression()
model.fit(X,y)

# YOUR CODE HERE: Create Store the fitted values as a time series with
# the same time index as the training data
y_pred = pd.Series(model.predict(X) ,index=X.index)
```

如果您想查看结果，请运行下一个单元格。

```python
fig, ax = plt.subplots()
ax.plot(X['lag_1'], y, '.', color='0.25')
ax.plot(X['lag_1'], y_pred)
ax.set(aspect='equal', ylabel='sales', xlabel='lag_1', title='Lag Plot of Average Sales');
```

![image-20221203182255712](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203182255712.png)

### Keep going

用移动平均图和时间虚拟图对时间序列中的趋势进行建模。