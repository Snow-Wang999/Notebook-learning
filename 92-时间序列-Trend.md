# 92-时间序列-Trend

用移动平均值和时间虚拟值（time dummy）模拟长期变化。

## 什么是趋势？ 

时间序列的趋势成分代表序列平均值的持续、长期变化。这一趋势是一个系列中移动最慢的部分，代表重要的最大时间尺度。在产品销售的时间序列中，随着越来越多的人逐年了解该产品，市场扩张可能会产生增长趋势。

![image-20221203182701530](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203182701530.png)

四个时间序列中的趋势模式。

在本课程中，我们将关注均值中的趋势。更一般地说，一个序列中的任何持续和缓慢的变化都可能构成一种趋势——例如，时间序列的变化通常具有趋势。

## 移动平均图(Moving Average Plots)

要了解时间序列可能具有什么样的趋势，我们可以使用移动平均线图。为了计算时间序列的移动平均值，我们计算**某个定义宽度的滑动窗口内的值的平均值**。图表上的每个点都表示序列中位于两侧窗口内的所有值的平均值。其目的是消除该系列中的任何短期波动，从而**只留下长期变化**。

![image-20221203182820829](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203182820829.png)

![An animated plot showing an undulating curve slowly increasing with a moving average line developing from left to right within a window of 12 points (in red).](https://i.imgur.com/EZOXiPs.gif)

说明线性趋势的移动平均图。曲线上的每个点（蓝色）是大小为12的窗口内的点（红色）的平均值。

请注意上面的Mauna Loa系列是如何年复一年地重复着上下波动的——一种短期的季节性变化。要使变化成为趋势的一部分，它应该比任何季节性变化发生的时间更长。因此，为了可视化趋势，我们采用比系列中任何季节性周期更长的周期的平均值。对于Mauna Loa系列，我们选择了一个12大小的窗口，以在每年的季节中保持平稳。

## 工程趋势 

一旦我们确定了趋势的形状，我们就可以尝试使用时间步长特征对其进行建模。我们已经看到了如何使用时间虚拟本身来建模线性趋势：

```python
target = a * time + b
```

我们可以通过时间虚拟的变换来拟合许多其他类型的趋势。如果趋势看起来是二次曲线（抛物线），我们只需将时间虚拟的平方添加到特征集，即可得到：

```python
target = a * time ** 2 + b * time + c
```

线性回归将学习系数a、b和c。 

下图中的趋势曲线均使用这些特征和`scikit learn`的`LinearRegression`进行拟合：

![image-20221203183022489](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203183022489.png)

![image-20221203183047851](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203183047851.png)

顶部：具有线性趋势的系列。下图：具有二次趋势的系列。

如果你以前没有看过这个技巧，你可能还没有意识到线性回归可以拟合曲线而不是直线。其想法是，如果您可以提供适当形状的曲线作为特征，那么线性回归可以学习如何以最适合目标的方式组合它们。

## Example - Tunnel Traffic

在本例中，我们将为隧道交通数据集创建一个趋势模型。

```python
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

simplefilter("ignore")  # ignore warnings to clean up output cells

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
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
data_dir = Path("../input/ts-course-data")
tunnel = pd.read_csv(data_dir / "tunnel.csv", parse_dates=["Day"])
tunnel = tunnel.set_index("Day").to_period()
```

让我们做一个移动平均图，看看这个系列有什么样的趋势。由于本系列有每日观察，让我们选择一个365天的窗口来平滑一年内的任何短期变化。 

要创建移动平均值，首先使用滚动`rolling`方法开始窗口计算。按照平均值方法计算窗口上的平均值。正如我们所看到的，隧道交通的趋势似乎是线性的。

```python
moving_average = tunnel.rolling(
    window=365,       # 365-day window
    center=True,      # puts the average at the center of the window
    min_periods=183,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)

ax = tunnel.plot(style=".", color="0.5")
moving_average.plot(
    ax=ax, linewidth=3, title="Tunnel Traffic - 365-Day Moving Average", legend=False,
);
```

![image-20221203183233367](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203183233367.png)

在第1课中，我们直接在Pandas中设计了时间dummy。但是，从现在起，我们将使用`statsmodels`库中的一个名为`DeterministicProcess`的函数。使用此函数将帮助我们避免时间序列和线性回归可能出现的一些棘手的失败案例。阶参数指的是多项式阶：1表示线性，2表示二次，3表示三次，依此类推。

[`statsmodels.tsa.deterministic.DeterministicProcess`](https://www.statsmodels.org/dev/generated/statsmodels.tsa.deterministic.DeterministicProcess.html)

```python
from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(
    index=tunnel.index,  # dates from the training data
    #进程的索引。在预测应用中使用时，通常应为“样本中”索引。
    constant=True,       # dummy feature for the bias (y_intercept)
    #是否包含常量。
    order=1,             # the time dummy (trend)
    #要包含的时间趋势的顺序。例如，2将包括线性项和二次项。0排除时间趋势项。
    drop=True,           # drop terms if necessary to avoid collinearity(避免共线)
    #一种标志，指示检查完全共线并删除任何线性相关项。
)
# `in_sample` creates features for the dates given in the `index` argument
#`in_sample为“index”参数中给定的日期创建功能
X = dp.in_sample()

X.head()
```

![image-20221203183343220](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203183343220.png)

（顺便说一句，确定性过程是一个非随机或完全确定的时间序列的技术术语，就像常量和趋势序列一样。从时间指数得出的特征通常是确定性的。） 我们创建的趋势模型基本上与之前一样，但注意添加了`fit_intercept=False`参数。

```python
from sklearn.linear_model import LinearRegression

y = tunnel["NumVehicles"]  # the target

# The intercept is the same as the `const` feature from
# DeterministicProcess. LinearRegression behaves badly with duplicated
# features, so we need to be sure to exclude it here.
model = LinearRegression(fit_intercept=False)
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
```

线性回归模型发现的趋势与移动平均图几乎相同，这表明在这种情况下，线性趋势是正确的决定。

```python
ax = tunnel.plot(style=".", color="0.5", title="Tunnel Traffic - Linear Trend")
_ = y_pred.plot(ax=ax, linewidth=3, label="Trend")
```

![image-20221203183436059](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203183436059.png)

为了进行预测，我们将模型应用于“样本外”特征。“样本外”是指训练数据观察期之外的时间。以下是我们如何做出30天的预测：

```python
X = dp.out_of_sample(steps=30)

y_fore = pd.Series(model.predict(X), index=X.index)

y_fore.head()
```

```
2005-11-17    114981.801146
2005-11-18    115004.298595
2005-11-19    115026.796045
2005-11-20    115049.293494
2005-11-21    115071.790944
Freq: D, dtype: float64
```

让我们绘制系列的一部分，以查看未来30天的趋势预测：

```python
ax = tunnel["2005-05":].plot(title="Tunnel Traffic - Linear Trend Forecast", **plot_params)
ax = y_pred["2005-05":].plot(ax=ax, linewidth=3, label="Trend")
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color="C3")
_ = ax.legend()
```

![image-20221203183539986](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221203183539986.png)

我们在本课中学习的趋势模型证明是有用的，原因有很多。除了作为更复杂模型的基线或起点外，我们还可以将其用作“混合模型”的一个组件，该模型具有**无法学习趋势的算法（如XGBoost和随机森林）**。我们将在第5课中了解有关此技术的更多信息。

## 轮到你了 

在商店销售中建模趋势，并了解使用高阶多项式进行预测的风险。

## Exercise: Trend

### 介绍

运行以下代码设置一切。

```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.time_series.ex2 import *

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

retail_sales = pd.read_csv(
    data_dir / "us-retail-sales.csv",
    parse_dates=['Month'],
    index_col='Month',
).to_period('D')
food_sales = retail_sales.loc[:, 'FoodAndBeverage']
auto_sales = retail_sales.loc[:, 'Automobiles']

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

### 1） 用移动平均图确定趋势 

美国零售销售数据集包含美国多个零售行业的月度销售数据。运行下一个单元格，查看《餐饮》系列的剧情。

```python
ax = food_sales.plot(**plot_params)
ax.set(title="US Food and Beverage Sales", ylabel="Millions of Dollars");
```

![image-20221204133915745](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204133915745.png)

现在制作一个移动平均图来估计这个系列的趋势。

```python
# YOUR CODE HERE: Add methods to `food_sales` to compute a moving
# average with appropriate parameters for trend estimation.
trend = food_sales.rolling(
    window=12,# 12-month window
    center=True,# puts the average at the center of the window
    min_periods=5,# choose about half the window size
).mean() # compute the mean (could also do median, std, min, max, ...)

# Make a plot
ax = food_sales.plot(**plot_params, alpha=0.5)
ax = trend.plot(ax=ax, linewidth=3)
```

![image-20221204140904263](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204140904263.png)

### 2） 确定趋势 

什么样的顺序多项式趋势可能适用于食品和饮料销售系列？你能想出一条非多项式的曲线吗？ 

趋势的向上弯曲表明2阶（二次）多项式可能是合适的。 

如果你以前研究过经济时间序列，你可能会猜测食品和饮料销售的增长率最好用百分比变化来表示。百分比变化通常可以使用指数曲线进行建模。（如果不熟悉，不要担心！）

在本课中，我们将继续使用平均销售额的时间序列。运行此单元格以查看估计趋势的average_sales的移动平均图。

```python
trend = average_sales.rolling(
    window=365,
    center=True,
    min_periods=183,
).mean()

ax = average_sales.plot(**plot_params, alpha=0.5)
ax = trend.plot(ax=ax, linewidth=3)
```

![image-20221204141237886](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204141237886.png)

### 3） 创建趋势特征 

使用`DeterministicProcess`为立方体趋势模型创建特征集。还可以为90天预报创建功能。

```python
from statsmodels.tsa.deterministic import DeterministicProcess

y = average_sales.copy()  # the target

# YOUR CODE HERE: Instantiate `DeterministicProcess` with arguments
# appropriate for a cubic trend model
dp = DeterministicProcess(
    index=y.index,# dates from the training data
    #进程的索引。在预测应用中使用时，通常应为“样本中”索引。
    order=3, # the time dummy (trend)
    #要包含的时间趋势的顺序。例如，2将包括线性项和二次项。0排除时间趋势项。
    drop=True, # drop terms if necessary to avoid collinearity(避免共线)
)

# YOUR CODE HERE: Create the feature set for the dates given in y.index
X = dp.in_sample()
#`in_sample为“index”参数中给定的日期创建功能

# YOUR CODE HERE: Create features for a 90-day forecast.
X_fore = dp.out_of_sample(steps=90)
```

通过运行下一个单元格，可以看到结果的a图。

```python
model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y.plot(**plot_params, alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, linewidth=3, label="Trend", color='C0')
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color='C3')
ax.legend();
```

![image-20221204143518314](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204143518314.png)

---

拟合更复杂趋势的一种方法是增加所用多项式的阶数。为了更好地适应商店销售中有些复杂的趋势，我们可以尝试使用11阶多项式。

```python
from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(index=y.index, order=11)
X = dp.in_sample()

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

ax = y.plot(**plot_params, alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, linewidth=3, label="Trend", color='C0')
ax.legend();
```

![image-20221204143622162](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204143622162.png)

### 4） 了解高阶多项式预测的风险 

然而，高阶多项式通常不太适合预测。你能猜出原因吗？

一个11阶多项式将包括t**11这样的项。这样的项在训练期外往往会迅速偏离，因此预测非常不可靠。

运行此单元格以使用11阶多项式查看相同的90天预测。这是否证实了你的直觉？

```python
X_fore = dp.out_of_sample(steps=90)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y.plot(**plot_params, alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, linewidth=3, label="Trend", color='C0')
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color='C3')
ax.legend();
```

![image-20221204143715056](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204143715056.png)

### （可选）用样条曲线拟合趋势 (Fit trend with splines)

当您想要拟合趋势时，样条曲线(*Splines*)是多项式的一个很好的替代方案。`pyearth`库中的多元自适应回归样条（MARS-*Multivariate Adaptive Regression Splines*）算法功能强大且易于使用。您可能需要研究很多超参数。

```python
from pyearth import Earth

# Target and features are the same as before
y = average_sales.copy()
dp = DeterministicProcess(index=y.index, order=1)
X = dp.in_sample()

# Fit a MARS model with `Earth`
model = Earth()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

ax = y.plot(**plot_params, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, linewidth=3, label="Trend")
```

![image-20221204143855381](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204143855381.png)

预测像这样复杂的趋势通常很困难（如果不是不可能的话）。然而，对于历史数据，您可以使用样条曲线通过**去趋势**来隔离时间序列中的其他模式。

```python
y_detrended = y - y_pred   # remove the trend from store_sales

y_detrended.plot(**plot_params, title="Detrended Average Sales");
```

![image-20221204143952113](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204143952113.png)

### 继续前进 

模型季节性([**Model seasonality**](https://www.kaggle.com/ryanholbrook/seasonality))，另一种常见的时间依赖性，具有指标和傅里叶特征(indicators and Fourier features)。