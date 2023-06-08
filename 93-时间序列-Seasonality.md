# 93-时间序列-Seasonality（季节性）

创建指标和傅里叶特征以捕捉周期性变化。

## 什么是季节性（Seasonality）？ 

我们说，每当时间序列的平均值有规律的周期性变化时，时间序列就表现出季节性。季节性变化通常遵循时钟和日历——在一天、一周或一年内重复是常见的。季节性通常是由自然世界几天几年的周期或围绕日期和时间的社会行为惯例所驱动的。

![image-20221204214934928](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204214934928.png)

四个时间序列中的季节模式。

我们将学习模拟季节性的两种特征。第一种**指标（indicators）**，最好是在**观察很少**的季节，比如每周观察一次的季节。第二种是**傅里叶特征（Fourier features）**，它最适合于有**很多观测**的季节，比如每年的需要每天观测的季节。

## 季节性图和季节性指标（Seasonal Plots and Seasonal Indicators）

就像我们使用移动平均图来发现一个系列的趋势一样，我们可以使用季节图来发现季节性模式。 

季节图显示了时间序列中与**某个公共周期**相对应的部分，该周期是您要观察的“季节”。该图显示了维基百科关于三角测量（Trigonometry）的文章的每日浏览量的季节图：该文章在一个共同的每周时段内的每日浏览。

![image-20221204215206208](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204215206208.png)

在这个系列中有一个明显的每周季节性模式，工作日较高，周末下降。

### 季节性指标（Seasonal indicators）

季节性指标是表示时间序列水平的季节性差异的二进制特征。如果您将季节性时段视为分类特征并应用一个热独编码，则会得到季节性指标。 

通过一周中的一天热独编码，我们可以得到每周的季节性指标。为三角测量系列创建每周指标将为我们提供六个新的“虚拟”功能。（如果你放弃其中一个指标，线性回归效果最好；我们在下面的框架中选择了周一。）

![image-20221204215352989](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204215352989.png)

将季节性指标添加到训练数据中有助于模型区分季节性期间的平均值：

![image-20221204215419534](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204215419534.png)

普通线性回归学习季节中每个时间的平均值。

指示灯（indicators）充当开/关开关。在任何时候，这些指示器中最多有一个的值为1（开）。线性回归学习Mon的基线值2379，然后根据当天的哪个指标的值进行调整；其余为0并消失。

## 傅里叶特征与周期图（Fourier Features and the Periodogram）

我们现在讨论的这种特征更适合长季节，而不是许多指标不切实际的观测结果。傅里叶特征不是为每个日期创建一个特征，而是尝试**用几个特征来捕捉季节曲线的整体形状**。 

让我们来看看三角测量的年度季节图。注意各种频率的重复：一年三次长的上下运动，一年52次短的每周运动，也许还有其他。

![image-20221204215530933](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204215530933.png)

Wiki三角测量系列中的年度季节性。

正是这些频率在一个季节内，我们试图用傅里叶特征捕捉。我们的想法是在我们的训练数据中包含与我们试图建模的季节具有相同频率的周期曲线。我们使用的曲线是三角函数正弦和余弦的曲线。 

**傅里叶特征**是*成对的正弦和余弦曲线*，从最长的季节开始，**每个潜在频率对应一对**。傅里叶对模拟年度季节性的频率为：每年一次，每年两次，每年三次，依此类推。

![image-20221204215633176](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204215633176.png)

年度季节性的前两个傅里叶对。顶部：频率为每年一次。底部：频率为每年两次。

如果我们将一组正弦/余弦曲线添加到训练数据中，线性回归算法将计算出适合目标序列中季节分量的权重。该图说明了线性回归如何使用四个傅里叶对来模拟Wiki三角测量系列中的年度季节性。

![image-20221204215713848](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204215713848.png)

顶部：四个傅里叶对的曲线，一个正弦和余弦与回归系数之和。每条曲线模拟不同的频率。

底部：这些曲线的总和近似于季节模式。

注意，我们只需要八个特征（四个正弦/余弦对）就可以很好地估计年度季节性。将其与季节性指标方法进行比较，该方法需要数百个功能（一年中的每一天一个）。通过使用傅里叶特征仅对季节性的“主要影响”进行建模，通常需要在训练数据中添加少得多的特征，这意味着减少了计算时间，减少了过度拟合的风险。

### 利用周期图选择傅里叶特征（Choosing Fourier features with the Periodogram）

我们应该在特征集中实际包含多少傅里叶对？我们可以用周期图（Periodogram）来回答这个问题。周期图告诉你时间序列中**频率的强度**。具体而言，图的y轴上的值为$（a^2+b^2）/2$，其中a和b是该频率下的正弦和余弦系数（如上面的傅里叶分量图所示）。

![image-20221204215858926](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204215858926.png)

Wiki三角测量系列的周期图。

从左到右，周期图在季度后下降，每年四次。这就是为什么我们选择了四个傅里叶对来模拟每年的季节。我们忽略了每周频率，因为它更好地用指标建模。

### 计算傅里叶特征（可选）（Computing Fourier features (optional)）

了解傅里叶特征是如何计算的，这对使用它们并不重要，但如果看到细节可以澄清问题，下面的单元格隐藏单元格说明了如何从时间序列的索引中导出一组傅里叶特征。（不过，我们将在应用程序中使用statsmodels中的库函数。）

```python
import numpy as np


def fourier_features(index, freq, order):
    time = np.arange(len(index), dtype=np.float32)
    k = 2 * np.pi * (1 / freq) * time
    features = {}
    for i in range(1, order + 1):
        features.update({
            f"sin_{freq}_{i}": np.sin(i * k),
            f"cos_{freq}_{i}": np.cos(i * k),
        })
    return pd.DataFrame(features, index=index)


# Compute Fourier features to the 4th order (8 new features) for a
# series y with daily observations and annual seasonality:
#
# fourier_features(y, freq=365.25, order=4)
```

计算四阶傅里叶特征（8个新特征） 

具有每日观察和年度季节性的系列y： 

fourier_features（y，freq(频率)=365.25，order(阶数)=4）

## Example - Tunnel Traffic

我们将再次继续使用隧道交通数据集。此隐藏单元加载数据并定义两个函数：seasonal_plot和plot_periodogram。

```python
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

simplefilter("ignore")

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
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
    legend=False,
)
%config InlineBackend.figure_format = 'retina'


# annotations: https://stackoverflow.com/a/49238256/5769929
def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


data_dir = Path("../input/ts-course-data")
tunnel = pd.read_csv(data_dir / "tunnel.csv", parse_dates=["Day"])
tunnel = tunnel.set_index("Day").to_period("D")
```

让我们来看看一周和一年的季节图。

```python
X = tunnel.copy()

# days within a week
X["day"] = X.index.dayofweek  # the x-axis (freq)
X["week"] = X.index.week  # the seasonal period (period)

# days within a year
X["dayofyear"] = X.index.dayofyear
X["year"] = X.index.year
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6))
seasonal_plot(X, y="NumVehicles", period="week", freq="day", ax=ax0)
seasonal_plot(X, y="NumVehicles", period="year", freq="dayofyear", ax=ax1);
```

![image-20221204220527577](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204220527577.png)

![image-20221204220544900](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204220544900.png)

现在让我们看看周期图：

```python
plot_periodogram(tunnel.NumVehicles);
```

![image-20221204220635510](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204220635510.png)

周期图与上述季节图一致：一个强劲的周季和一个较弱的年季。我们将用指标对每周的季节进行建模，并用傅里叶特征对每年的季节进行模拟。从右到左，周期图在双月（6）和月（12）之间下降，所以我们使用10个傅里叶对。 

我们将使用`DeterministicProcess`创建季节性特征，这是我们在第2课中用于创建趋势特征的相同工具。要使用两个季节周期（每周和每年），我们需要将其中一个实例化为“附加术语”：

```python
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

fourier = CalendarFourier(freq="A", order=10)  # 10 sin/cos pairs for "A"nnual seasonality

dp = DeterministicProcess(
    index=tunnel.index,
    constant=True,               # dummy feature for bias (y-intercept)
    order=1,                     # trend (order 1 means linear)
    seasonal=True,               # weekly seasonality (indicators)
    additional_terms=[fourier],  # annual seasonality (fourier)
    drop=True,                   # drop terms to avoid collinearity
)

X = dp.in_sample()  # create features for dates in tunnel.index
```

创建了特征集后，我们就可以拟合模型并进行预测了。我们将添加一个90天的预测，以查看我们的模型如何超出训练数据。这里的代码与前面课程中的代码相同。

```python
y = tunnel["NumVehicles"]

model = LinearRegression(fit_intercept=False)
_ = model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=y.index)
X_fore = dp.out_of_sample(steps=90)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y.plot(color='0.25', style='.', title="Tunnel Traffic - Seasonal Forecast")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax = y_fore.plot(ax=ax, label="Seasonal Forecast", color='C3')
_ = ax.legend()
```

![image-20221204220742638](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221204220742638.png)

---

我们还可以做更多的时间序列来改进我们的预测。在下一课中，我们将学习如何使用时间序列本身作为功能。使用时间序列作为预测的输入，我们可以对序列中经常出现的另一个组成部分进行建模：周期（cycles）。

## 轮到你了 

为商店销售创建季节性功能，并将这些技术扩展到捕捉假日效果。

## Exercise: Seasonality

### 介绍 

运行此单元以设置所有内容！

```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.time_series.ex3 import *

# Setup notebook
from pathlib import Path
from learntools.time_series.style import *  # plot style settings
from learntools.time_series.utils import plot_periodogram, seasonal_plot

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess


comp_dir = Path('../input/store-sales-time-series-forecasting')

holidays_events = pd.read_csv(
    comp_dir / "holidays_events.csv",
    dtype={
        'type': 'category',
        'locale': 'category',
        'locale_name': 'category',
        'description': 'category',
        'transferred': 'bool',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
holidays_events = holidays_events.set_index('date').to_period('D')

store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales['date'] = store_sales.date.dt.to_period('D')
# pandas.Series.dt.to_period 以特定频率强制转换为PeriodArray/Index。
# 将DatetimeArray/Index转换为PeriodArray/Index。
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
average_sales = (
    store_sales
    .groupby('date').mean()
    .squeeze()
    .loc['2017']
)
```

----

补充：

1. 数据类型（dtype）：`'category'`。

   [pandas文档Categorical data](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html#differences-to-r-s-factor)

   类别（Categoricals）是与统计中的类别变量相对应的熊猫数据类型。分类变量具有有限且通常固定的可能值（类别；R中的级别）。例如性别、社会阶层、血型、国家归属、观察时间或Likert量表评分。

   与统计分类变量相反，分类数据可能有顺序（例如“强烈一致”vs“一致”或“第一次观察”vs“第二次观察”），但数字运算（加法、除法…）是不可能的。

   分类数据的所有值都在类别(categories)或`np.nan`中。顺序由类别的顺序定义，而不是值的词汇顺序。在内部，数据结构由一个类别数组和一个指向类别数组中实际值的整数代码数组组成。

   分类数据类型在以下情况下有用：

   - 仅由几个不同值组成的字符串变量。将这样的字符串变量转换为类别变量将节省一些内存，请参见此处。 
   - 变量的词汇顺序与逻辑顺序（“一”、“二”、“三”）不同。通过转换为类别并指定类别的顺序，排序和min/max将使用逻辑顺序而不是词汇顺序，请参见此处。 
   - 作为向其他Python库发出的信号，应将此列视为分类变量（例如，使用适当的统计方法或绘图类型）。

---

`from learntools.time_series.utils import plot_periodogram, seasonal_plot` 里面包含现成的季节图和周期图。

```python
print(X.index)
```

![image-20221205122538186](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205122538186.png)

![image-20221205123106775](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205123106775.png)

以时间作为index的可以取实例，比如week，day，year等。（按Tab键，查看黄色实例部分）

![image-20221205122959928](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205122959928.png)

检查以下季节图：

```python
X = average_sales.to_frame()
X["week"] = X.index.week
X["day"] = X.index.dayofweek
seasonal_plot(X, y='sales', period='week', freq='day');
```

![image-20221205113640945](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205113640945.png)

（周期是week，频率是day）

还有周期图：

```python
plot_periodogram(average_sales);
```

![image-20221205123335816](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205123335816.png)

### 1） 确定季节性

你看到了什么样的季节性证据？一旦你想好了，运行下一个单元进行一些讨论。

季节图和周期图都表明**每周季节性较强**。从周期图来看，似乎也有一些每月和每两周的成分。事实上，《商店销售》数据集的注释显示，公共部门的工资每两周发放一次，即每月15日和最后一天——这可能是这些季节的一个来源。

### 2） 创建季节性功能 

使用`DeterministicProcess`和`CalendarFourier`创建： 

- 每周季节和 
- 月度季节的傅里叶特征为4级。

```python
y = average_sales.copy()

# YOUR CODE HERE
fourier = CalendarFourier(freq='M',order=4)
dp = DeterministicProcess(
    index=y.index,
    constant=True,# dummy feature for bias (y-intercept)
    order=1,# trend (order 1 means linear)
    # YOUR CODE HERE
    seasonal=True,#每周季节性指标
    additional_terms=[fourier],#添加额外每月四阶季节性傅里叶指标
    drop=True,# drop terms to avoid collinearity
)
X = dp.in_sample()#create features for dates in tunnel.index
```

现在运行此单元格以适应季节模型。

```python
model = LinearRegression().fit(X, y)
y_pred = pd.Series(
    model.predict(X),
    index=X.index,
    name='Fitted',
)

y_pred = pd.Series(model.predict(X), index=X.index)
ax = y.plot(**plot_params, alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax.legend();
```

![image-20221205124404948](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205124404948.png)

***

从一个系列中删除其趋势或季节称为**取消趋势或取消季节化**。 看看去季节化序列的周期图。

```python
y_deseason = y - y_pred

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))
ax1 = plot_periodogram(y, ax=ax1)
ax1.set_title("Product Sales Frequency Components")
ax2 = plot_periodogram(y_deseason, ax=ax2);
ax2.set_title("Deseasonalized");
```

![image-20221205124511662](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205124511662.png)

### 3） 检查剩余季节性 

根据这些周期图，您的模型如何有效地捕捉了平均销售额的季节性？周期图与去季节化序列的时间图一致吗？

去季节化序列的周期图缺少任何大值。通过将其与原始系列的周期图进行比较，我们可以看到，我们的模型能够捕捉平均销售额的季节变化。

---

商店销售数据集包括厄瓜多尔假日表。

```python
# National and regional holidays in the training set
holidays = (
    holidays_events
    .query("locale in ['National', 'Regional']")
    .loc['2017':'2017-08-15', ['description']]
    .assign(description=lambda x: x.description.cat.remove_unused_categories())
)

display(holidays)
```

![image-20221205124651283](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205124651283.png)

从去季节化的平均销售额图来看，这些假期似乎具有一定的预测能力。

```python
ax = y_deseason.plot(**plot_params)
plt.plot_date(holidays.index, y_deseason[holidays.index], color='C3')
ax.set_title('National and Regional Holidays');
```

![image-20221205124927245](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205124927245.png)

### 4） 创建假日功能

您可以创建什么样的功能来帮助您的模型利用这些信息？在下一个单元格中输入答案。（Scikit-learn和Pandas都有实用程序，可以让这件事变得简单。如果您想了解更多详细信息，请参阅提示。）

```python

# Scikit-learn solution
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)

X_holidays = pd.DataFrame(
    ohe.fit_transform(holidays),
    index=holidays.index,
    columns=holidays.description.unique(),
)


# Pandas solution
X_holidays = pd.get_dummies(holidays)


# Join to training data
X2 = X.join(X_holidays, on='date').fillna(0.0)
```

---

提示1：使用Pandas，可以使用`pd.get_dummies`。使用scikit学习，您可以使用`sklearn.preprrocessing.OneHotEncoder`。使用Pandas可以更容易地将X_holidays连接到X2，因为它返回一个保留每个假日日期的DataFrame。（对于另一个提示，请调用.hint（2）） 

提示2：在Pandas中，您的解决方案如下：

```python
X_holidays = pd.get_dummies(____)

X2 = X.join(X_holidays, on='date').fillna(0.0)
In scikit-learn, your solution would look like:

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)

X_holidays = pd.DataFrame(
    ____,
    index=____,
    columns=holidays.description.unique(),  # optional,  but nice to have
)

X2 = X.join(X_holidays, on='date').fillna(0.0)
```

---

使用此单元格以适合季节性模型，并添加假日功能。拟合值是否有所改善？

```python
model = LinearRegression().fit(X2, y)
y_pred = pd.Series(
    model.predict(X2),
    index=X2.index,
    name='Fitted',
)

y_pred = pd.Series(model.predict(X2), index=X2.index)
ax = y.plot(**plot_params, alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax.legend();
```

![image-20221205132410685](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205132410685.png)

原来没使用holiday特征的图：

![image-20221205124404948](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205124404948.png)

比较发现：极值被考虑到了，效果更好了。

---

### （可选）提交到商店销售竞争 

本部分练习将引导您完成本课程同伴竞赛的第一次提交：商店销售-时间序列预测。完成课程并不需要参加比赛，但这是一种尝试新技能的好方法。

---

```python
#之前的代码
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
store_sales.head()
```

![image-20221205133233651](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205133233651.png)

```python
#现在代码
store_sales.unstack(['store_nbr', 'family']).loc["2017"].head()
```

![image-20221205133318625](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205133318625.png)

用unstack解绑index

---

下一个单元格将为包含所有1800个时间序列的完整商店销售数据集创建一个您在本课中学习过的季节性模型。

```python
y = store_sales.unstack(['store_nbr', 'family']).loc["2017"]

# Create training data
fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
X = dp.in_sample()
X['NewYear'] = (X.index.dayofyear == 1)

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
y_pred = pd.DataFrame(model.predict(X), index=X.index, columns=y.columns)
```

您可以使用此单元格查看其一些预测。

```python
STORE_NBR = '1'  # 1 - 54
FAMILY = 'PRODUCE'
# Uncomment to see a list of product families
# display(store_sales.index.get_level_values('family').unique())

ax = y.loc(axis=1)['sales', STORE_NBR, FAMILY].plot(**plot_params)
ax = y_pred.loc(axis=1)['sales', STORE_NBR, FAMILY].plot(ax=ax)
ax.set_title(f'{FAMILY} Sales at Store {STORE_NBR}');
```

![image-20221205133844119](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205133844119.png)

```python
STORE_NBR = '20'
```

![image-20221205134621717](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221205134621717.png)

最后，此单元加载测试数据，为预测期创建一个功能集，然后创建提交文件submission.csv。

```python
df_test = pd.read_csv(
    comp_dir / 'test.csv',
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
df_test['date'] = df_test.date.dt.to_period('D')
df_test = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()

# Create features for test set
X_test = dp.out_of_sample(steps=16)
X_test.index.name = 'date'
X_test['NewYear'] = (X_test.index.dayofyear == 1)


y_submit = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)
y_submit = y_submit.stack(['store_nbr', 'family'])
y_submit = y_submit.join(df_test.id).reindex(columns=['id', 'sales'])
y_submit.to_csv('submission.csv', index=False)
```

为了测试你的预测，你需要加入竞争（如果你还没有）。因此，单击此链接打开一个新窗口。然后单击“加入竞争”按钮。 接下来，按照以下说明进行操作： 

1. 首先单击窗口右上角的“保存版本”按钮。这将生成一个弹出窗口。 

2. 确保选择了“保存并全部运行”选项，然后单击“保存”按钮。 

3. 这将在笔记本的左下角生成一个窗口。运行完成后，单击“保存版本”按钮右侧的数字。这将在屏幕右侧弹出一个版本列表。单击最新版本右侧的省略号（…），然后选择在查看器中打开。这将使您进入同一页面的查看模式。您需要向下滚动以返回这些说明。 
4. 单击屏幕右侧的输出选项卡。然后，单击要提交的文件，然后单击“提交”按钮将结果提交到排行榜。 

您现在已成功提交比赛！ 

如果您想继续工作以提高性能，请选择屏幕右上角的“编辑”按钮。然后您可以更改代码并重复该过程。还有很多改进的空间，你会在工作中爬上排行榜。

### 继续前进 

使用时间序列作为特征来捕获周期和其他类型的序列相关性。