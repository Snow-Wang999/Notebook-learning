# 96-时间序列-Forecasting With Machine Learning

（使用机器学习进行预测）

使用这四种策略将ML应用于任何预测任务。

[关于时序数据稳定性的讨论](https://zhuanlan.zhihu.com/p/334732886?ivk_sa=1024320u)

## 介绍 

在第2课和第3课中，我们将预测视为一个简单的回归问题，所有的特征都来自于一个输入，即时间指数。我们只需生成所需的趋势和季节特征，就可以轻松创建未来任何时间的预测。 

然而，当我们在第4课中添加了滞后特性时，问题的性质发生了变化。滞后特征要求在预测时已知滞后目标值。滞后1功能将时间序列向前移动1步，这意味着您可以预测未来的1步，但不能预测2步。 

在第4课中，我们只是假设我们总是可以产生滞后，直到我们想要预测的时间段（换句话说，每一个预测都只是向前一步）。现实世界中的预测通常需要更多，因此在本课中，我们将学习如何对各种情况进行预测。

## 定义预测任务（Defining the Forecasting Task）

在设计预测模型之前，需要确定两件事： 

- 在做出预测时什么信息可用（特征）， 

- 需要预测值（目标）的时间段。 

预测起点( **forecast origin** )是您进行预测的时间。实际上，您可能会将预测原点视为**最后一次获得预测时间**的训练数据。直到原点的一切都可以用来创建特征。 

预测范围(**forecast horizon**)是您进行预测的时间。我们通常通过预测范围内的时间步数( time steps )来描述预测：例如，“一步”预测或“五步”预测。预测范围描述了目标。

![image-20221207214345264](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207214345264.png)

使用四个滞后特征（*lag features*），具有两步提前期（*lead time*）的三步预测范围（*forecast horizon*）。该图表示的是一行训练数据——换句话说，一个简单预测的数据。

原点和地平线之间的时间是预测的提前时间（*lead time*）（有时是延迟时间-*latency*）。预测的提前期由从原点到地平线的步数来描述：例如，“提前一步”或“提前三步”预测。实际上，由于数据采集或处理的延迟，预测可能需要提前多个步骤开始。

## 为预测准备数据（Preparing Data for Forecasting）

为了使用ML算法预测时间序列，我们需要将序列转换为可用于这些算法的数据帧。（当然，除非您只使用趋势和季节性等确定性特征。） 

我们在第4课中看到了这个过程的前半部分，当时我们创建了一个消除滞后的特征集。下半场正在准备目标。我们如何做到这一点取决于预测任务。 

数据帧中的**每一行表示一个预测**。该行的时间索引是预测范围中的第一次，但我们将整个范围的值排列在同一行中。对于**多步骤预测**，这意味着我们需要一个模型来产生多个输出，每个步骤一个。

```python
import numpy as np
import pandas as pd

N = 20
ts = pd.Series(
    np.arange(N),
    index=pd.period_range(start='2010', freq='A', periods=N, name='Year'),
    dtype=pd.Int8Dtype,
)

# Lag features
X = pd.DataFrame({
    'y_lag_2': ts.shift(2),
    'y_lag_3': ts.shift(3),
    'y_lag_4': ts.shift(4),
    'y_lag_5': ts.shift(5),
    'y_lag_6': ts.shift(6),    
})

# Multistep targets
y = pd.DataFrame({
    'y_step_3': ts.shift(-2),
    'y_step_2': ts.shift(-1),
    'y_step_1': ts,
})

data = pd.concat({'Targets': y, 'Features': X}, axis=1)

data.head(10).style.set_properties(['Targets'], **{'background-color': 'LavenderBlush'}) \
                   .set_properties(['Features'], **{'background-color': 'Lavender'})
```

![image-20221207214536146](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207214536146.png)

上面说明了如何准备数据集，类似于定义预测图：一个三步预测任务，使用五个滞后特征，两步提前期。原始时间序列是`y_step_1`。我们可以填写或删除缺失的值。

## 多步骤预测策略（Multistep Forecasting Strategies）

有许多策略可用于生成预测所需的多个目标步骤。我们将概述四种常见策略，每个策略都有优缺点。

### 多输出模型（Multioutput model）

使用自然产生多个输出的模型。线性回归和神经网络都可以产生多个输出。这种策略简单有效，但不可能适用于您可能想要使用的每种算法。例如，XGBoost无法做到这一点。

![image-20221207214852446](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207214852446.png)

### 直接战略（Direct strategy）

为范围上的每一步训练一个单独的模型：一个模型预测一步，另一个预测两步，依此类推。预测一步与预测两步是不同的问题（依此类推），因此可以让不同的模型**对每一步进行预测**。缺点是训练大量模型的计算**成本很高**。

![image-20221207214921663](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207214921663.png)

### 递归策略（Recursive strategy）

训练一个单一的一步模型，并使用其预测来更新下一步的滞后特征。使用递归方法，我们将模型的一步预测反馈到同一模型中，作为下一步预测的滞后特征。我们只需要训练一个模型，但由于**误差会一步一步地传播**，因此**长期预测可能不准确**。

![image-20221207214932543](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207214932543.png)

### DirRec策略（DirRec strategy）

直接和递归策略的组合：**为每个步骤训练一个模型，并将先前步骤的预测作为新的滞后特征**。一步一步地，每个模型都会获得额外的滞后输入。由于每个模型都有一组最新的滞后特征，因此DirRec策略比Direct策略能够更好地捕获串行相关性，但它也会像递归一样受到错误传播的影响。

![image-20221207215016556](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207215016556.png)

## Example - Flu Trends

在本例中，我们将将MultiOutput和Direct策略应用于第4课中的流感趋势数据，这一次将在训练期之后的多周内进行真实预测。 

我们将定义预测任务，以8周为期限，1周为提前期。换句话说，我们将从下周开始预测八周的流感病例。 

隐藏的单元格设置了示例并定义了一个辅助函数plot_multistep。

```python
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

simplefilter("ignore")

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)
%config InlineBackend.figure_format = 'retina'


def plot_multistep(y, every=1, ax=None, palette_kwargs=None):
    palette_kwargs_ = dict(palette='husl', n_colors=16, desat=None)
    if palette_kwargs is not None:
        palette_kwargs_.update(palette_kwargs)
    palette = sns.color_palette(**palette_kwargs_)
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler('color', palette))
    for date, preds in y[::every].iterrows():
        preds.index = pd.period_range(start=date, periods=len(preds))
        preds.plot(ax=ax)
    return ax


data_dir = Path("../input/ts-course-data")
flu_trends = pd.read_csv(data_dir / "flu-trends.csv")
flu_trends.set_index(
    pd.PeriodIndex(flu_trends.Week, freq="W"),
    inplace=True,
)
flu_trends.drop("Week", axis=1, inplace=True)
```

首先，我们将为多步骤预测准备我们的目标系列（每周的流感办公室访问）。一旦完成，训练和预测将非常直接。

```python
def make_lags(ts, lags, lead_time=1):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)


# Four weeks of lag features
y = flu_trends.FluVisits.copy()
X = make_lags(y, lags=4).fillna(0.0)


def make_multistep_target(ts, steps):
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)


# Eight-week forecast
y = make_multistep_target(y, steps=8).dropna()

# Shifting has created indexes that don't match. Only keep times for
# which we have both targets and features.
y, X = y.align(X, join='inner', axis=0)
```

### 多输出模型 （Multioutput model）

我们将使用线性回归作为多输出策略。一旦我们为多个输出准备好数据，训练和预测就一如既往。

```python
# Create splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y.columns)
y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)
```

请记住，多步骤模型将为用作输入的每个实例生成完整的预测。训练集有269周，测试集有90周，我们现在对每一周都进行了8步预测。

```python
train_rmse = mean_squared_error(y_train, y_fit, squared=False)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print((f"Train RMSE: {train_rmse:.2f}\n" f"Test RMSE: {test_rmse:.2f}"))

palette = dict(palette='husl', n_colors=64)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
ax1 = flu_trends.FluVisits[y_fit.index].plot(**plot_params, ax=ax1)
ax1 = plot_multistep(y_fit, ax=ax1, palette_kwargs=palette)
_ = ax1.legend(['FluVisits (train)', 'Forecast'])
ax2 = flu_trends.FluVisits[y_pred.index].plot(**plot_params, ax=ax2)
ax2 = plot_multistep(y_pred, ax=ax2, palette_kwargs=palette)
_ = ax2.legend(['FluVisits (test)', 'Forecast'])
```

```
Train RMSE: 389.12
Test RMSE: 582.33
```

![image-20221207215336903](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207215336903.png)

### 直接战略（Direct strategy）

XGBoost无法为回归任务生成多个输出。但通过应用直接缩减策略，我们仍然可以使用它来生成多步骤预测。这就像用scikit learn的`MultiOutputRegressor`包装一样简单。

```python
from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(XGBRegressor())
model.fit(X_train, y_train)

y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y.columns)
y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)
```

这里的XGBoost显然对训练集过度拟合。但在测试集上，它似乎能够比线性回归模型更好地捕捉流感季节的一些动态。通过一些超参数调整，它可能会做得更好。

```python
train_rmse = mean_squared_error(y_train, y_fit, squared=False)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print((f"Train RMSE: {train_rmse:.2f}\n" f"Test RMSE: {test_rmse:.2f}"))

palette = dict(palette='husl', n_colors=64)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
ax1 = flu_trends.FluVisits[y_fit.index].plot(**plot_params, ax=ax1)
ax1 = plot_multistep(y_fit, ax=ax1, palette_kwargs=palette)
_ = ax1.legend(['FluVisits (train)', 'Forecast'])
ax2 = flu_trends.FluVisits[y_pred.index].plot(**plot_params, ax=ax2)
ax2 = plot_multistep(y_pred, ax=ax2, palette_kwargs=palette)
_ = ax2.legend(['FluVisits (test)', 'Forecast'])
```

```
Train RMSE: 1.22
Test RMSE: 526.45
```

![image-20221207215740102](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207215740102.png)

要使用DirRec策略，只需要用另一个scikit学习包装器RegressorChain替换MultiOutputRegressor。递归策略我们需要自己编码。

## 轮到你了 

为门店销售创建预测数据集并应用DirRec策略。

## Exercise: Forecasting With Machine Learning

### 引言

运行此单元以设置所有内容！

```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.time_series.ex6 import *

# Setup notebook
from pathlib import Path
import ipywidgets as widgets
from learntools.time_series.style import *  # plot style settings
from learntools.time_series.utils import (create_multistep_example,
                                          load_multistep_data,
                                          make_lags,
                                          make_multistep_target,
                                          plot_multistep)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


comp_dir = Path('../input/store-sales-time-series-forecasting')

store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

family_sales = (
    store_sales
    .groupby(['family', 'date'])
    .mean()
    .unstack('family')
    .loc['2017']
)

test = pd.read_csv(
    comp_dir / 'test.csv',
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
test['date'] = test.date.dt.to_period('D')
test = test.set_index(['store_nbr', 'family', 'date']).sort_index()
```

考虑以下三项预测任务： 

- 使用4个滞后特征和2步提前期的3步预测 

- 使用3个滞后特征和1步提前期进行1步预测 

- 使用4个滞后特征和1步提前期的3步预测 

运行下一个单元格以查看三个数据集，每个数据集代表上述任务之一。

数据集分页

```python
datasets = load_multistep_data()

data_tabs = widgets.Tab([widgets.Output() for _ in enumerate(datasets)])
for i, df in enumerate(datasets):
    data_tabs.set_title(i, f'Dataset {i+1}')
    with data_tabs.children[i]:
        display(df)

display(data_tabs)
```

![image-20221210162515410](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210162515410.png)

![image-20221210164629347](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210164629347.png)

![image-20221210164642538](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210164642538.png)

### 1） 将描述与数据集匹配

 您能将每个任务与适当的数据集匹配吗？

```python
# YOUR CODE HERE: Match the task to the dataset. Answer 1, 2, or 3.
task_a = 2 #a是第二个数据集，有三步预测（target的列），四个滞后值，两步lead，所以滞后值从2开始
task_b = 1 #b是第一个数据集
task_c = 3 #c是第三个数据集
```

查看训练和测试集的时间索引。根据这些信息，您能否确定门店销售的预测任务？

```python
print("Training Data", "\n" + "-" * 13 + "\n", store_sales)
print("\n")
print("Test Data", "\n" + "-" * 9 + "\n", test)
```

![image-20221210172214789](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210172214789.png)

![image-20221210172305638](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210172305638.png)

### 2） 确定门店销售竞争的预测任务

尝试确定预测来源和预测范围。预测范围内有多少步骤？预测的提前期是多久？ 

**思考好答案后运行此单元格。**

培训将于2017-08-15结束，这为我们提供了**预测来源**。测试集包括日期2017-08-16至2017-08-31，这为我们提供了**预测范围**。起点和地平线之间只有一步之遥，所以我们有**一天的前置时间**。 换句话说，我们需要一个**16步的预测和一个1步的提前期(a 16-step forecast with a 1-step lead time)**。我们可以使用从滞后1开始的滞后，并使用**2017-08-15的特征**进行整个16步预测。

在本教程中，我们了解了如何为单个时间序列创建多步骤数据集。幸运的是，我们可以对多个系列的数据集使用完全相同的过程。

### 3） 为门店销售

创建多步骤数据集 创建适合门店销售预测任务的目标。使用4天的延迟功能。删除目标和功能中缺失的值。

```python
# YOUR CODE HERE
y = family_sales.loc[:, 'sales']

# YOUR CODE HERE: Make 4 lag features
X = make_lags(y, lags=4).dropna()

# YOUR CODE HERE: Make multistep target
y = make_multistep_target(y, steps=16).dropna()

y, X = y.align(X, join='inner', axis=0)
```

在本教程中，我们了解了如何使用流感趋势系列的多输出和直接策略进行预测。现在，您将把DirRec策略应用于商店销售的多个时间序列。 确保您已成功完成上一个练习，然后运行此单元为XGBoost准备数据。

```python
le = LabelEncoder()
X = (X
    .stack('family')  # wide to long
    .reset_index('family')  # convert index to column
    .assign(family=lambda x: le.fit_transform(x.family))  # label encode
)
y = y.stack('family')  # wide to long

display(y)
```

![image-20221210182242135](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210182242135.png)

### 4） 使用DirRec策略进行预测

建立一个将DirRec策略应用于XGBoost的模型。

```python
from sklearn.multioutput import RegressorChain

# YOUR CODE HERE
model = RegressorChain(XGBRegressor())
# RegressorChain(base_estimator=XGBRegressor())
```

如果你想训练这个模型，就运行这个单元。

```python
model.fit(X, y)

y_pred = pd.DataFrame(
    model.predict(X),
    index=y.index,
    columns=y.columns,
).clip(0.0)
```

并使用此代码查看该模型对训练数据进行的16步预测的样本。

```python
FAMILY = 'BEAUTY'
START = '2017-04-01'
EVERY = 16

y_pred_ = y_pred.xs(FAMILY, level='family', axis=0).loc[START:]
y_ = family_sales.loc[START:, 'sales'].loc[:, FAMILY]

fig, ax = plt.subplots(1, 1, figsize=(11, 4))
ax = y_.plot(**plot_params, ax=ax, alpha=0.5)
ax = plot_multistep(y_pred_, ax=ax, every=EVERY)
_ = ax.legend([FAMILY, FAMILY + ' Forecast'])
```

![image-20221210182518338](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210182518338.png)

### 下一步

祝贺你已经完成了Kaggle的时间序列课程。如果你还没有，加入我们的竞争对手：商店销售-时间序列预测([Store Sales - Time Series Forecasting](https://www.kaggle.com/c/29781) )，并运用你学到的技能。 

要获得灵感，请查看Kaggle以前的预测比赛。研究获胜的竞争解决方案是提升技能的好方法。

- [**Corporación Favorita**](https://www.kaggle.com/c/favorita-grocery-sales-forecasting): the competition *Store Sales* is derived from.
- [**Rossmann Store Sales**](https://www.kaggle.com/c/rossmann-store-sales)
- [**Wikipedia Web Traffic**](https://www.kaggle.com/c/web-traffic-time-series-forecasting/)
- [**Walmart Store Sales**](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
- [**Walmart Sales in Stormy Weather**](https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather)
- [**M5 Forecasting - Accuracy**](https://www.kaggle.com/c/m5-forecasting-accuracy)



工具书类 

这里有一些很好的资源，您可能想了解更多关于时间序列和预测的信息。他们都在塑造这一过程中发挥了作用： 

- 卡斯珀·索尔海姆·博杰尔（Casper Solheim Bojer）和延斯·佩德·梅尔德加德（Jens Peder Meldgaard）撰写的一篇文章《从Kaggle的预测比赛中学习》。 
- Rob J Hyndmann和George Athanasopoulos的著作《预测：原则与实践》。 
- Galit Shmueli和Kenneth C.Lichtendahl Jr.的著作《用R进行实际时间序列预测》。 
- 时间序列分析及其应用，Robert H.Shumway和David S.Stoffer的著作。 
- 时间序列预测的机器学习策略，Gianluca Bontempi、Souhaib Ben Taieb和Yann Aeål Le Borgne的文章。 
- Christoph Bergmeir和JoséM.Benítez撰写的关于时间序列预测值评估中交叉验证的文章。

> - *Learnings from Kaggle's forecasting competitions*, an article by Casper Solheim Bojer and Jens Peder Meldgaard.
> - *Forecasting: Principles and Practice*, a book by Rob J Hyndmann and George Athanasopoulos.
> - *Practical Time Series Forecasting with R*, a book by Galit Shmueli and Kenneth C. Lichtendahl Jr.
> - *Time Series Analysis and Its Applications*, a book by Robert H. Shumway and David S. Stoffer.
> - *Machine learning strategies for time series forecasting*, an article by Gianluca Bontempi, Souhaib Ben Taieb, and Yann-Aël Le Borgne.
> - *On the use of cross-validation for time series predictor evaluation*, an article by Christoph Bergmeir and José M. Benítez.