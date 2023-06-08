# 94-时间序列-Time Series as Features

用滞后嵌入从过去预测未来。

## 什么是序列相依性？（What is Serial Dependence?） 

在前面的课程中，我们研究了时间序列的特性，这些特性最容易被建模为**与时间相关的特性**，也就是说，我们可以直接从时间索引中导出特性。然而，一些时间序列属性只能建模为**序列相依属性（*serially dependent* properties）**，即使用**目标序列的过去值**作为特征。这些时间序列的结构可能从随时间变化的图中看不出来；然而，根据过去的值绘制，结构变得清晰——如下图所示。

![image-20221205171537273](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205171537273.png)

这两个系列具有序列相关性，但不具有时间相关性。右侧的点具有坐标（时间t-1的值，时间t的值）。

通过趋势和季节性，我们训练了模型，使曲线与上图左侧的曲线相吻合——这些模型学习了时间依赖性。本课的目标是训练模型，使曲线与右边的曲线吻合——我们希望他们学习**序列相关性**。

## 周期(循环-cycles)

序列依赖性的一种特别常见的表现方式是**周期（循环-cycles）**。周期是时间序列中的增长和衰减（growth and decay）模式，与序列中某一时间的值如何取决于**先前时间的值**有关，但不一定取决于时间步长本身。循环行为是系统的特征，这些系统会影响自身或其反应会随着时间的推移而持续。经济、流行病、动物种群、火山爆发和类似的自然现象通常表现出周期性行为。

![image-20221205172211512](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205172211512.png)

具有循环行为的四个时间序列。

周期行为与季节性的区别在于，**周期不一定像季节一样依赖于时间。**一个周期中发生的事情，与其说是特定的发生日期，不如说是*最近发生的事情*。与时间的独立性（至少是相对的）意味着**周期性行为可能比季节性更不规则**。

## 滞后系列和滞后图(Lagged Series and Lag Plots)

为了研究时间序列中可能的序列相关性（如周期），我们需要创建序列的“滞后”副本。滞后时间序列意味着将其值向前移动一个或多个时间步长，或者等效地，将其索引中的时间向后移动一个或者多个步长。无论是哪种情况，其影响都是滞后序列中的观测结果将在稍后时间出现。 

这显示了美国的月失业率（y）及其第一和第二个滞后系列（分别为y_lag_1和y_lag_2）。请注意，滞后序列的值是如何在时间上向前移动的。

```python
import pandas as pd

# Federal Reserve dataset: https://www.kaggle.com/federalreserve/interest-rates
reserve = pd.read_csv(
    "../input/ts-course-data/reserve.csv",
    parse_dates={'Date': ['Year', 'Month', 'Day']},
    index_col='Date',
)

y = reserve.loc[:, 'Unemployment Rate'].dropna().to_period('M')
df = pd.DataFrame({
    'y': y,
    'y_lag_1': y.shift(1),
    'y_lag_2': y.shift(2),    
})

df.head()
```

![image-20221205172509892](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205172509892.png)

通过使时间序列滞后，我们可以使其过去的值与我们试图预测的值同时出现（换句话说，在同一行中）。这使得滞后序列作为建模序列相关性的特征非常有用。为了预测美国失业率序列，我们可以使用`y_lag_1`和`y_lag_2`作为特征来预测目标`y`。这将预测未来失业率，作为前两个月失业率的函数。

### 滞后曲线图(Lag plots)

时间序列的滞后图显示了其值与滞后的关系。通过查看滞后图，时间序列中的序列相关性通常会变得明显。我们可以从美国失业率的滞后图中看出，当前失业率与过去的失业率之间存在着明显的线性关系。

![image-20221205172631220](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205172631220.png)

显示了美国失业率与自相关的滞后图。

最常用的序列相关性度量称为自相关(**autocorrelation**)，即时间序列与其滞后之一的相关性。美国失业率在滞后1时具有0.99的自相关，在滞后2时具有0.98的自相关等。

### 选择滞后时间(Choosing lags)

当选择要用作特征的滞后时，通常将每个滞后都包含在较大的自相关中是没有用的。例如，在美国失业率中，滞后2的自相关可能完全来自滞后1的“衰减”信息——这只是前一步中的相关性。如果滞后2不包含任何新内容，那么如果我们已经有滞后1，就没有理由包含它。 

**部分自相关(partial autocorrelation)**告诉你一个滞后的相关性，它解释了所有之前的滞后——可以说，滞后所带来的“新”相关性的数量。绘制部分自相关可以帮助您选择要使用的滞后特征。在下图中，滞后1到滞后6超出了“无相关性”的区间（蓝色），因此我们可以选择滞后1到落后6作为美国失业率的特征。（滞后11很可能是假阳性。）

超出蓝色阴影部分的滞后特征可以选择。

![image-20221205173412420](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205173412420.png)

美国失业率通过滞后12的部分自相关，95%置信区间不相关。

像上面这样的图被称为**相关图（ *correlogram*）**。延迟特征的相关图本质上与傅里叶特征的周期图相同。 

最后，我们需要注意，**自相关和部分自相关**是**线性相关性**的度量。由于真实世界的时间序列通常具有显著的非线性相关性，因此在选择滞后特征时，最好查看滞后图（或使用一些更一般的相关性度量，如相互信息（[mutual information](https://www.kaggle.com/ryanholbrook/mutual-information)））。太阳黑子序列具有非线性相关性，我们可能会忽略自相关。

![image-20221205173431912](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205173431912.png)

太阳黑子系列的滞后情节。

这样的非线性关系可以被转换为线性关系，或者通过适当的算法学习。

## Example - Flu Trends

流感趋势数据集包含2009年至2016年几周内医生的流感就诊记录。我们的目标是预测未来几周的流感病例数。 

我们将采取两种方法。首先，我们将使用**滞后特征**预测医生的就诊。我们的第二种方法是使用**另一组时间序列的滞后时间**来预测医生的就诊情况：谷歌趋势捕捉到的流感相关搜索词。

```python
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import periodogram
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf

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

#滞后图
def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax


def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    import math
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig


data_dir = Path("../input/ts-course-data")
flu_trends = pd.read_csv(data_dir / "flu-trends.csv")
flu_trends.set_index(
    pd.PeriodIndex(flu_trends.Week, freq="W"),
    inplace=True,
)
flu_trends.drop("Week", axis=1, inplace=True)

ax = flu_trends.FluVisits.plot(title='Flu Trends', **plot_params)
_ = ax.set(ylabel="Office Visits")
```

![image-20221205173446987](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205173446987.png)

我们的流感趋势数据显示了**不规则的周期**，而不是有规律的季节性：高峰往往发生在新年前后，但有时更早或更晚，有时更大或更小。利用**滞后特征**对这些周期进行建模将使我们的预报员能够动态地**对变化的条件作出反应**，而不是像季节性特征那样被限制在准确的日期和时间。 

让我们先看看滞后和自相关图：

```python
_ = plot_lags(flu_trends.FluVisits, lags=12, nrows=2)
_ = plot_pacf(flu_trends.FluVisits, lags=12)
```

![image-20221205173459750](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205173459750.png)

![image-20221205173512185](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205173512185.png)

滞后图表明，`FluVisits`与其滞后的关系大多是线性的，而部分自相关表明，可以使用**滞后1、2、3和4**来捕捉相关性。我们可以使用移位法（shift）在Pandas中滞后时间序列。对于这个问题，我们将用0.0填充滞后创建的缺失值。

```python
def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)


X = make_lags(flu_trends.FluVisits, lags=4)
X = X.fillna(0.0)
```

在之前的课程中，我们能够为训练数据之外的任意多个步骤创建预测。然而，当使用滞后特征时，我们仅限于预测滞后值可用的时间步长。使用周一的滞后1特性，我们无法对周三进行预测，因为所需的滞后1值是周二，而这还没有发生。 

我们将在第6课中看到处理此问题的策略。对于本例，我们将只使用测试集中的值。

```python
# Create target series and data splits
y = flu_trends.FluVisits.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=60, shuffle=False)

# Fit and predict
model = LinearRegression()  # `fit_intercept=True` since we didn't use DeterministicProcess
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_train), index=y_train.index)
y_fore = pd.Series(model.predict(X_test), index=y_test.index)
```

```python
ax = y_train.plot(**plot_params)
ax = y_test.plot(**plot_params)
ax = y_pred.plot(ax=ax)
_ = y_fore.plot(ax=ax, color='C3')
```

![image-20221205173528951](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205173528951.png)

仅看预测值，我们就可以看到我们的模型需要一个**时间步长（time step）**来对目标序列中的突然变化做出反应。这是仅使用**目标Series的滞后lags**作为特征的模型的常见限制。

```python
ax = y_test.plot(**plot_params)
_ = y_fore.plot(ax=ax, color='C3')
```

![image-20221205173542463](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205173542463.png)

为了改进预测，我们可以尝试找到**主要指标（ *leading indicators*）**，时间序列可以为流感病例的变化提供“预警（early warning）”。对于我们的第二种方法，我们将在我们的训练数据中添加一些**与流感相关的搜索词的流行程度**，这是通过谷歌趋势来衡量的。 

将搜索短语“`FluCough`”与目标“`FluVisits`”进行对比表明，这样的搜索词可以作为**主要指标（ *leading indicators*）**：与流感相关的搜索往往在**办公室访问前几周**变得更流行。

---

![image-20221205173554600](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205173554600.png)

数据集包含129个这样的术语，但我们只使用几个。

```python
search_terms = ["FluContagious", "FluCough", "FluFever", "InfluenzaA", "TreatFlu", "IHaveTheFlu", "OverTheCounterFlu", "HowLongFlu"]

# Create three lags for each search term
X0 = make_lags(flu_trends[search_terms], lags=3)

# Create four lags for the target, as before
X1 = make_lags(flu_trends['FluVisits'], lags=4)

# Combine to create the training data
X = pd.concat([X0, X1], axis=1).fillna(0.0)
```

我们的预测有点粗略，但我们的模型似乎更能预测流感访问量的突然增加，这表明搜索热度的几个时间序列确实是有效的领先指标。

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=60, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_train), index=y_train.index)
y_fore = pd.Series(model.predict(X_test), index=y_test.index)

ax = y_test.plot(**plot_params)
_ = y_fore.plot(ax=ax, color='C3')
```

![image-20221205173607656](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205173607656.png)

本课中所示的时间序列可能被称为“纯周期性”：它们没有明显的趋势或季节性。但时间序列同时具有**趋势性（trend）、季节性（seasonality）和周期性（cycles）**这三个组成部分并不罕见。您可以通过为每个组件添加适当的特性，使用线性回归对此类系列进行建模。您甚至可以组合经过训练的模型来单独学习组件，我们将在下一课中学习如何预测混合动力（*forecasting hybrids*）。

## 轮到你了 

为商店销售创建滞后特性，并探索其他类型的时间序列特性。

## Exercise: Time Series as Features

### 介绍 

运行此单元以设置所有内容！

```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.time_series.ex4 import *

# Setup notebook
from pathlib import Path
from learntools.time_series.style import *  # plot style settings
from learntools.time_series.utils import plot_lags, make_lags, make_leads
#lags, leading
# plot_lags时序滞后图
# make_lags制作滞后时序列表
# make_leads制作主要指标，比如使用相关搜索词的时序数据


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from statsmodels.graphics.tsaplots import plot_pacf
#时序滞后自相关图
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
#傅里叶时序季节性特征，确定性趋势

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
    .loc['2017', ['sales', 'onpromotion']]
)
```

---

并非每个产品系列的销售都表现出周期性（cyclic）行为，平均销售列也不如此。然而，学校和办公用品的销售显示出增长和衰退的模式，并没有很好地表现出趋势或季节特征。在这个问题和下一个问题中，您将使用滞后特性对学校和办公用品的销售周期进行建模。 

趋势性和季节性都会产生一系列的相关性，这些相关性会在相关图和滞后图中显示出来。为了隔离任何**纯循环**行为，我们将从**列去季节化**开始。使用下一个单元格中的代码取消`Supply Sales`的季节化。我们将结果存储在变量`y_desason`中。

```python
supply_sales = family_sales.loc(axis=1)[:, 'SCHOOL AND OFFICE SUPPLIES']
y = supply_sales.loc[:, 'sales'].squeeze()

fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    constant=True,
    index=y.index,
    order=1,
    seasonal=True,
    drop=True,
    additional_terms=[fourier],
)
X_time = dp.in_sample()
X_time['NewYearsDay'] = (X_time.index.dayofyear == 1)

model = LinearRegression(fit_intercept=False)
model.fit(X_time, y)
y_deseason = y - model.predict(X_time)
y_deseason.name = 'sales_deseasoned'

ax = y_deseason.plot()
ax.set_title("Sales of School and Office Supplies (deseasonalized)");
```

![image-20221207113343843](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207113343843.png)

这个去季节化的系列是否显示出循环模式？为了证实我们的直觉，我们可以尝试使用**移动平均图来隔离循环行为**，就像我们对趋势所做的那样。其想法是选择一个足够长的窗口来平滑短期季节性，但足够短的窗口来保持周期。

### 1） 绘图周期（Plotting cycles）

根据供应销售系列y创建七天移动平均线。使用居中的窗口，但不要设置min_periods参数。

```python
# YOUR CODE HERE
y_ma = y.rolling(7,center=True).mean()
#window为7，参数是以天为单位，所以周期是7天

# Plot
ax = y_ma.plot()
ax.set_title("Seven-Day Moving Average");
```

![image-20221207114049791](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207114049791.png)

你看到移动平均线图与去季节化系列图的相似之处了吗？在这两种情况下，我们都可以看到循环行为。

---

[pandas库之DataFrame滑动窗口（rolling window）(官网介绍）](https://blog.csdn.net/weixin_45526117/article/details/124758135)

[DataFrame](https://so.csdn.net/so/search?q=DataFrame&spm=1001.2101.3001.7020)的滑动窗口

```python
DataFrame.rolling(
    window,
    min_periods=None,
    center=False,
    win_type=None,
    on=None,
    axis=0,
    closed=None,
    method='single')
```

参数：

- **window**：int, offset, or BaseIndexer subclass

  **移动窗口的大小**，如果是整数，代表每个窗口覆盖的固定数量；如果是offset（pandas时间序列），代表每个窗口的时间段，每个窗口的大小将根据时间段中包含的观察值而变化，仅对datetimelike索引有效。

- **min_periods**：int, default None

  窗口计算值要求**至少有min_periods个观测值**。窗口由时间类型指定，则min_periods默认为1，窗口为整数，则min_periods默认为窗口大小

- **center**：bool, default False

  是否将窗口**中间索引**设为窗口计算后的标签

- **win_type**：str, default None

  **观测值的权重分布**。如果为None，则所有点的权重均相等。如果是字符串，要求是 scipy.signal window function函数

- **on**：str, optional

  对于 DataFrame，计算滚动窗口所依照的**列标签或索引级别**，而不是 DataFrame 的索引

- **axis**：int or str, default 0

  如果是0或’index’，**按行滚动**；如果是1或’columns’，按列滚动

- **closed**：str, default None

  ‘right’：窗口中的第一个点将从计算中**排除**；‘left‘：窗口中的最后一个点将从计算中排除；‘both’：窗口中没有点将从计算中排除；‘neither’：窗口中的第一个点和最后一个点将从计算中排除；默认’right’

---

让我们来检查我们的去季节化序列的序列相关性。看看部分自相关相关图和滞后图。

```python
plot_pacf(y_deseason, lags=8);
plot_lags(y_deseason, lags=8, nrows=2);
```

![image-20221207115039304](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207115039304.png)

![image-20221207115057877](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207115057877.png)

### 2） 检查商店销售中的序列相关性

根据相关图，是否有任何滞后显著？滞后图是否显示了相关图中不明显的关系？ 

思考好答案后，运行下一个单元格。

**相关图表明第一个滞后很可能是显著的，还有可能是第八个滞后。滞后图表明，这种影响主要是线性的。**

---

回想一下教程中提到的领先指标是一个系列（series），它在某一时间的值可以用来预测未来的目标——领先指标提供目标变化的“提前通知”。 

比赛数据集包括一个可能用作领先指标的时间序列——促销（`onpromotion`）系列，其中包含当天特别促销的项目数量。由于公司自己决定何时进行促销，因此不必担心“前瞻性泄露”；例如，我们可以使用周二的促销值来预测周一的销售额。 

使用下一个单元格检查与学校和办公用品销售相关的促销领先和滞后值。

```python
onpromotion = supply_sales.loc[:, 'onpromotion'].squeeze().rename('onpromotion')

# Drop days without promotions
plot_lags(x=onpromotion.loc[onpromotion > 1], y=y_deseason.loc[onpromotion > 1], lags=3, leads=3, nrows=1);
```

![image-20221207124315129](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207124315129.png)

### 3） 检查时间序列特征

促销的超前或滞后值是否可以作为一种功能使用？

**滞后图表明，促销的领先值和滞后值都与供应销售相关。这表明这两种值都可以作为功能使用。也可能存在一些非线性效应。**

### 4） 创建时间序列特征

创建问题3的解决方案中所示的功能。如果该系列中没有有用的功能，请使用空的dataframe `pd.dataframe()`作为答案。

```python
# YOUR CODE HERE: Make features from `y_deseason`
X_lags = make_lags(y_deseason,lags=1)

# YOUR CODE HERE: Make features from `onpromotion`
# You may want to use `pd.concat`
X_promo = pd.concat([
    make_lags(onpromotion,lags=1),
    onpromotion,
    make_leads(onpromotion,leads=1),
],axis=1)

X = pd.concat([X_time, X_lags, X_promo], axis=1).dropna()
y, X = y.align(X, join='inner')
```

如果您希望从生成的模型中看到预测，请使用下一个单元格中的代码。

```python
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=30, shuffle=False)

model = LinearRegression(fit_intercept=False).fit(X_train, y_train)
y_fit = pd.Series(model.predict(X_train), index=X_train.index).clip(0.0)
y_pred = pd.Series(model.predict(X_valid), index=X_valid.index).clip(0.0)

rmsle_train = mean_squared_log_error(y_train, y_fit) ** 0.5
rmsle_valid = mean_squared_log_error(y_valid, y_pred) ** 0.5
print(f'Training RMSLE: {rmsle_train:.5f}')
print(f'Validation RMSLE: {rmsle_valid:.5f}')

ax = y.plot(**plot_params, alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_fit.plot(ax=ax, label="Fitted", color='C0')
ax = y_pred.plot(ax=ax, label="Forecast", color='C3')
ax.legend();
```

![image-20221207125747049](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207125747049.png)

---

Kaggle预测比赛的获胜者通常在他们的特征集中包括移动平均值(moving average)和其他滚动统计数据(rolling statistics)。当与XGBoost等**GBDT算法**一起使用时，这些功能似乎特别有用。 

在第2课中，您学习了如何计算移动平均值来估计趋势。计算用作特征的滚动统计数据是类似的，只是我们需要注意避免前瞻性泄漏(lookahead leakage)。首先，结果应该设置在窗口的右端，而不是中心——也就是说，我们应该在滚动方法中使用`center=False`（默认值）。第二，**目标应该滞后一步**。

### 5） 创建统计特征 

编辑下一个单元格中的代码以创建以下功能： 

- 滞后目标的14天滚动中值（中值） 
- 滞后目标的7天滚动标准偏差（std） 
- “促销”项目的7天总和（总和），窗口居中

```python
y_lag = supply_sales.loc[:, 'sales'].shift(1)
onpromo = supply_sales.loc[:, 'onpromotion']

# 7-day mean of lagged target
mean_7 = y_lag.rolling(7).mean()
# YOUR CODE HERE: 14-day median of lagged target
median_14 = y_lag.rolling(14).median()
# YOUR CODE HERE: 7-day rolling standard deviation of lagged target
std_7 = y_lag.rolling(7).std()
# YOUR CODE HERE: 7-day sum of promotions with centered window
promo_7 = onpromo.rolling(7,center=True).sum()
```

查看Pandas Window文档([`Window` documentation](https://pandas.pydata.org/pandas-docs/stable/reference/window.html) )，了解更多可以计算的统计信息。也可以使用`ewm`代替滚动(`rolling`)来尝试“**指数加权**”窗口；指数衰减通常是效果如何随时间传播的更现实的表示。

### 继续前进 

创建混合预测器(hybrid forecasters)，并结合两种机器学习算法的优势。