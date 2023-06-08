# 95-时间序列-Hybrid Models

将两位预报员（forecasters）的优势与这一强大的技术结合起来。

## 介绍 

线性回归（linear regression）擅长推断趋势，但无法学习交互作用。XGBoost擅长学习互动，但无法推断趋势。在本课中，我们将学习如何创建“混合”预测器，将互补的学习算法结合起来，让其中一个的优点弥补另一个的缺点。

- **线性回归**：推断趋势（extrapolate trends）

- **XGBoost**：学习交互（interaction）作用

## 成分和残留物(Components and Residuals)

为了能够设计出有效的混合模型，我们需要更好地理解时间序列是如何构造的。到目前为止，我们已经研究了三种依赖模式：**趋势、季节和周期（trends，seasonality，cycles）**。许多时间序列可以通过以下三个分量加上一些基本上不可预测的**完全随机误差**的加法模型来描述：

```python
series = trend + seasons + cycles + error
```

这个模型中的每一项我们都称为时间序列的一个组成部分。 

模型的**残差**是模型所训练的**目标和**模型所做的**预测之间的差异**，即实际曲线和拟合曲线之间的差异。根据一个特征绘制残差，您可以得到目标的“剩余”部分，或者模型未能从该特征中学习到目标的内容。

![image-20221207131233749](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207131233749.png)

目标序列和预测之间的差异（蓝色）给出了残差序列。

上图左侧是第3课中隧道交通量系列和季节性趋势曲线的一部分。减去拟合曲线后，剩余部分位于右侧。残差包含隧道交通的所有信息，而趋势季节模型没有学习到。 

我们可以想象学习时间序列的组成部分是一个**迭代过程**：

- 首先学习**趋势**并从序列中减去它，
- 然后从去趋势残差中学习**季节性**并减去季节，
- 然后学习**周期**并减去周期，
- 最后只剩下不可预测的**误差**。

![image-20221207131320092](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207131320092.png)

逐步学习Mauna Loa CO2的成分。从其系列中减去拟合曲线（蓝色），以在下一步中获得系列。

把我们学到的所有组件加在一起，我们就得到了完整的模型。这基本上是线性回归所能做的，如果你在一组完整的特征建模趋势、季节和周期上训练它。

![image-20221207131354693](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207131354693.png)

添加学习的组件以获得完整的模型。

## 残差混合预测(Hybrid Forecasting with Residuals)

在前面的课程中，我们使用了一个算法（线性回归）来一次学习所有组件。但也可以对某些组件使用一种算法，对其他组件使用另一种算法。这样，我们可以始终为每个组件选择最佳算法。为此，我们使用一个算法拟合原始序列，然后使用第二个算法拟合残差序列。 

具体来说，流程如下：

```python
# 1. Train and predict with first model
model_1.fit(X_train_1, y_train)
y_pred_1 = model_1.predict(X_train)

# 2. Train and predict with second model on residuals
model_2.fit(X_train_2, y_train - y_pred_1)
y_pred_2 = model_2.predict(X_train_2)

# 3. Add to get overall predictions
y_pred = y_pred_1 + y_pred_2
```

我们通常希望使用不同的特征集（上面的X_train_1和X_train_2），这取决于我们希望每个模型学习的内容。例如，如果我们使用第一个模型来学习趋势，我们通常不需要第二个模型的趋势特征。 

虽然可以使用两个以上的模型，但在实践中，它似乎没有特别大的帮助。事实上，构建混合体最常见的策略是我们刚刚描述的策略：

- 简单（通常是线性）学习算法，（简单模型通常被设计为后续强大算法的“助手”。）

- 然后是复杂的非线性学习器，如GBDT或深度神经网络，

## 设计混合特征(Designing Hybrids) 

除了我们在本课中概述的方法之外，还有许多方法可以组合机器学习模型。然而，成功地组合模型需要我们深入研究这些算法的运行方式。 

### **回归算法**

回归算法通常有两种方式进行预测：要么通过**变换特征**，要么通过**变换目标**。特征变换算法学习一些数学函数，该函数将特征作为输入，然后将它们组合并变换，以生成与训练集中的目标值匹配的输出。**线性回归和神经网络**就是这种。 

**目标变换算法**使用特征对训练集中的目标值进行分组，并通过对组中的值进行平均来进行预测；一组特征仅指示要平均的组。**决策树和最近邻**就是这种。 （<u>*目标变换器的预测将始终限制在训练集的范围内*</u>。）

重要的是：特征变换器通常可以在给定适当的特征作为输入的情况下将目标值外推到训练集之外，但<u>*目标变换器的预测将始终限制在训练集的范围内*</u>。如果时间虚拟值继续计算**时间步长**，则线性回归将继续绘制**趋势线**。给定相同的时间虚拟值，**决策树**将永远预测训练数据的**最后一步**所指示的趋势。**决策树无法推断趋势。**随机森林和梯度增强决策树（如XGBoost）是决策树的集合，因此它们也无法推断趋势。

![image-20221207131703929](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207131703929.png)

决策树将无法推断（*extrapolate*）出超出训练集的趋势。

这一差异正是本课中混合设计的动机：

- 使用线性回归推断趋势，
- 转换目标以消除趋势，并将XGBoost应用于去趋势残差。

为了混合一个神经网络（一个特征变换器），你可以**将另一个模型的预测作为一个特征**，然后神经网络将其作为自己预测的一部分。

**拟合残差的方法**实际上与*梯度增强算法*使用的方法相同，因此我们将这些增强混合算法称为**增强（boosted）混合算法**；使用**预测作为特征**的方法被称为“叠加（stacking）”，因此我们将称这些**叠加（stacked）混合**。

> **Kaggle比赛中获胜的混合算法**
>
> 为了获得灵感，以下是一些来自过去比赛的顶级得分解决方案：
>
> - [STL boosted with exponential smoothing](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8125) - Walmart Recruiting - Store Sales Forecasting
> - [ARIMA and exponential smoothing boosted with GBDT](https://www.kaggle.com/c/rossmann-store-sales/discussion/17896) - Rossmann Store Sales
> - [An ensemble of stacked and boosted hybrids](https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/39395) - Web Traffic Time Series Forecasting
> - [Exponential smoothing stacked with LSTM neural net](https://github.com/Mcompetitions/M4-methods/blob/slaweks_ES-RNN/118 - slaweks17/ES_RNN_SlawekSmyl.pdf) - M4 (non-Kaggle)

## Example - US Retail Sales

美国零售销售数据集（ [*US Retail Sales*](https://www.census.gov/retail/index.html) ）包含美国人口普查局收集的1992年至2019年各零售行业的月度销售数据。我们的目标是预测2016-2019年的销售额，同时考虑到早些年的销售额。除了创建**线性回归+XGBoost混合**，我们还将了解如何设置用于XGBoost的时间序列数据集。

```python
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from xgboost import XGBRegressor


simplefilter("ignore")

# Set Matplotlib defaults
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
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)

data_dir = Path("../input/ts-course-data/")
industries = ["BuildingMaterials", "FoodAndBeverage"]
retail = pd.read_csv(
    data_dir / "us-retail-sales.csv",
    usecols=['Month'] + industries,
    parse_dates=['Month'],
    index_col='Month',
).to_period('D').reindex(columns=industries)
retail = pd.concat({'Sales': retail}, names=[None, 'Industries'], axis=1)

retail.head()
```

![image-20221207132009133](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207132009133.png)

首先，让我们使用线性回归模型来了解每个系列的趋势。为了演示，我们将使用二次（2阶）趋势。（这里的代码与前几节课中的代码基本相同。）虽然不完美，但它足以满足我们的需求。

```python
y = retail.copy()

# Create trend features
dp = DeterministicProcess(
    index=y.index,  # dates from the training data
    constant=True,  # the intercept
    order=2,        # quadratic trend
    drop=True,      # drop terms to avoid collinearity
)
X = dp.in_sample()  # features for the training data

# Test on the years 2016-2019. It will be easier for us later if we
# split the date index instead of the dataframe directly.
idx_train, idx_test = train_test_split(
    y.index, test_size=12 * 4, shuffle=False,
)
X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
y_train, y_test = y.loc[idx_train], y.loc[idx_test]

# Fit trend model
model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)

# Make predictions
y_fit = pd.DataFrame(
    model.predict(X_train),
    index=y_train.index,
    columns=y_train.columns,
)
y_pred = pd.DataFrame(
    model.predict(X_test),
    index=y_test.index,
    columns=y_test.columns,
)

# Plot
axs = y_train.plot(color='0.25', subplots=True, sharex=True)
axs = y_test.plot(color='0.25', subplots=True, sharex=True, ax=axs)
axs = y_fit.plot(color='C0', subplots=True, sharex=True, ax=axs)
axs = y_pred.plot(color='C3', subplots=True, sharex=True, ax=axs)
for ax in axs: ax.legend([])
_ = plt.suptitle("Trends")
```

![image-20221207132056545](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207132056545.png)

虽然线性回归算法能够进行多输出回归（multi-output regression），但XGBoost算法不能。为了使用XGBoost同时预测多个序列，我们将把这些序列从宽（wide）格式（每列一个时间序列）转换为长（long）格式（每行按类别索引序列）。

ps：

- “宽”格式：举例说明，我有5个列，100行。每行行数据为5个数据

- “长“格式：举例说明，我有5个列，100行。我把”宽“格式转换为”长”格式，称为“堆叠（stacking）”，每行行数据为100个数据。如有n个index，则上一级的index下面的每一项都有5行。

```python
# The `stack` method converts column labels to row labels, pivoting from wide format to long
X = retail.stack()  # pivot dataset wide to long
display(X.head())
y = X.pop('Sales')  # grab target series
```

![image-20221207132128149](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207132128149.png)

为了让XGBoost学会区分我们的两个时间序列，我们将把“行业”的行标签转换为带有**标签编码**的分类特征。我们还将通过从时间索引中**提取月份数字**来创建年度季节性特征。

```python
# Turn row labels into categorical feature columns with a label encoding
X = X.reset_index('Industries')
# Label encoding for 'Industries' feature
for colname in X.select_dtypes(["object", "category"]):
    X[colname], _ = X[colname].factorize()

# Label encoding for annual seasonality
X["Month"] = X.index.month  # values are 1, 2, ..., 12

# Create splits
X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
y_train, y_test = y.loc[idx_train], y.loc[idx_test]
```

现在我们将把之前的趋势预测转换为长格式，然后从原始序列中减去它们。这将为我们提供XGBoost可以学习的去趋势（残差）系列。

`squeeze()`：convert DataFrame to Series.

`stack()`： Pivot wide to long

```python
# Pivot wide to long (stack) and convert DataFrame to Series (squeeze)
y_fit = y_fit.stack().squeeze()    # trend from training set
y_pred = y_pred.stack().squeeze()  # trend from test set

# Create residuals (the collection of detrended series) from the training set
y_resid = y_train - y_fit

# Train XGBoost on the residuals
xgb = XGBRegressor()
xgb.fit(X_train, y_resid)

# Add the predicted residuals onto the predicted trends
y_fit_boosted = xgb.predict(X_train) + y_fit
y_pred_boosted = xgb.predict(X_test) + y_pred
```

虽然我们可以看到XGBoost学习的趋势与线性回归学习的趋势一样好，但拟合度似乎很好——特别是XGBoost无法弥补`BuildingMaterials`系列中拟合度较差的趋势。

```python
axs = y_train.unstack(['Industries']).plot(
    color='0.25', figsize=(11, 5), subplots=True, sharex=True,
    title=['BuildingMaterials', 'FoodAndBeverage'],
)
axs = y_test.unstack(['Industries']).plot(
    color='0.25', subplots=True, sharex=True, ax=axs,
)
axs = y_fit_boosted.unstack(['Industries']).plot(
    color='C0', subplots=True, sharex=True, ax=axs,
)
axs = y_pred_boosted.unstack(['Industries']).plot(
    color='C3', subplots=True, sharex=True, ax=axs,
)
for ax in axs: ax.legend([])
```

![image-20221207132242413](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207132242413.png)

## 轮到你了 

使用XGBoost混合预测门店销售额，并尝试其他ML算法组合。

## Exercise: Hybrid Models

### 介绍 

运行此单元以设置所有内容！

```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.time_series.ex5 import *

# Setup notebook
from pathlib import Path
from learntools.time_series.style import *  # plot style settings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.deterministic import DeterministicProcess
from xgboost import XGBRegressor


comp_dir = Path('../input/store-sales-time-series-forecasting')
data_dir = Path("../input/ts-course-data")

store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
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
```

---

在接下来的两个问题中，您将通过实现一个新的Python类为`StoreSales`数据集创建一个增强的混合。运行此单元格以创建初始类定义。您将添加`fit`和`predict`方法，以提供一个类似`scikit`学习的界面。

```python
# You'll add fit and predict methods to this minimal class
class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None  # store column names from fit method
```

### 1） 定义增强混合模型的拟合方法(Define fit method for boosted hybrid) 

完成BoostedHybrid类的拟合定义。如果需要，请参阅本教程中混合预测与残差部分的步骤1和2。

```python
def fit(self, X_1, X_2, y):
    # YOUR CODE HERE: fit self.model_1
    self.model_1.fit(X_1,y)

    y_fit = pd.DataFrame(
        # YOUR CODE HERE: make predictions with self.model_1
        self.model_1.predict(X_1),
        index=X_1.index, columns=y.columns,
    )

    # YOUR CODE HERE: compute residuals
    y_resid = y - y_fit
    y_resid = y_resid.stack().squeeze() # wide to long

    # YOUR CODE HERE: fit self.model_2 on residuals
    self.model_2.fit(X_2, y_resid)

    # Save column names for predict method
    self.y_columns = y.columns
    # Save data for question checking
    self.y_fit = y_fit
    self.y_resid = y_resid


# Add method to class
BoostedHybrid.fit = fit
```

### 2） 定义增强混合模型的预测方法(Define predict method for boosted hybrid) 

现在定义`BoostedHybrid`类的预测方法。如果需要，请参阅本教程中混合预测与残差部分的步骤3。

```python
def predict(self, X_1, X_2):
    y_pred = pd.DataFrame(
        # YOUR CODE HERE: predict with self.model_1
        self.model_1.predict(X_1),
        index=X_1.index, columns=self.y_columns,
    )
    y_pred = y_pred.stack().squeeze()  # wide to long

    # YOUR CODE HERE: add self.model_2 predictions to y_pred
    y_pred += self.model_2.predict(X_2)
    
    return y_pred.unstack()  # long to wide


# Add method to class
BoostedHybrid.predict = predict
```

现在，您可以使用新的`BoostedHybrid`类为`StoreSales`数据创建模型了。运行下一个单元格以设置训练数据。

```python
# Target series
y = family_sales.loc[:, 'sales']


# X_1: Features for Linear Regression
dp = DeterministicProcess(index=y.index, order=1)
X_1 = dp.in_sample()


# X_2: Features for XGBoost
X_2 = family_sales.drop('sales', axis=1).stack()  # onpromotion feature

# Label encoding for 'family'
le = LabelEncoder()  # from sklearn.preprocessing
X_2 = X_2.reset_index('family')
X_2['family'] = le.fit_transform(X_2['family'])

# Label encoding for seasonality
X_2["day"] = X_2.index.day  # values are day of the month
```

### 3） 训练增强混合模型(Train boosted hybrid)

通过使用`LinearRegression()`和`XGBRegressor()`实例初始化`BoostedHybrid`类来创建混合模型。

```python
# YOUR CODE HERE: Create LinearRegression + XGBRegressor hybrid with BoostedHybrid
model = BoostedHybrid(
    model_1=LinearRegression() ,
    model_2=XGBRegressor() ,
)

# YOUR CODE HERE: Fit and predict
model.fit(X_1, X_2, y)
y_pred = model.predict(X_1,X_2)
#clip函数：限制一个array的上下界
y_pred = y_pred.clip(0.0)
```

---

根据您的问题，您可能希望使用其他混合组合，而不是在前面的问题中创建的 线性回归+`XGBoost` 混合。运行下一个单元，尝试`scikit`学习中的其他算法。

```python
# Model 1 (trend)
from pyearth import Earth
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.linear_model import LinearRegression

# Model 2
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
# Boosted Hybrid

# YOUR CODE HERE: Try different combinations of the algorithms above
model = BoostedHybrid(
    model_1=Ridge(),
    model_2=KNeighborsRegressor(),
)
# 以下可选组合
## model_1: LinearRegression(),Earth(),ElasticNet(), Lasso(), Ridge()
## model_2: XGBRegressor(),ExtraTreesRegressor(), RandomForestRegressor(),KNeighborsRegressor(),MLPRegressor()
```

这些只是一些建议。您可能会在scikit学习用户指南([User Guide](https://scikit-learn.org/stable/supervised_learning.html))中发现您喜欢的其他算法。 

使用此单元格中的代码来查看您的混合预测。

```python
y_train, y_valid = y[:"2017-07-01"], y["2017-07-02":]
X1_train, X1_valid = X_1[: "2017-07-01"], X_1["2017-07-02" :]
X2_train, X2_valid = X_2.loc[:"2017-07-01"], X_2.loc["2017-07-02":]

# Some of the algorithms above do best with certain kinds of
# preprocessing on the features (like standardization), but this is
# just a demo.
model.fit(X1_train, X2_train, y_train)
y_fit = model.predict(X1_train, X2_train).clip(0.0)
y_pred = model.predict(X1_valid, X2_valid).clip(0.0)

families = y.columns[0:6]
axs = y.loc(axis=1)[families].plot(
    subplots=True, sharex=True, figsize=(11, 9), **plot_params, alpha=0.5,
)
_ = y_fit.loc(axis=1)[families].plot(subplots=True, sharex=True, color='C0', ax=axs)
_ = y_pred.loc(axis=1)[families].plot(subplots=True, sharex=True, color='C3', ax=axs)
for ax, family in zip(axs, families):
    ax.legend([])
    ax.set_ylabel(family)
```

![image-20221207163858902](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207163858902.png)

### 4） 适合不同的学习算法

尝试使用不同的组合：

- model_1: 
  - LinearRegression(),
  - Earth(),
  - ElasticNet(), 
  - Lasso(), 
  - Ridge()

- model_2: 
  - XGBRegressor(),
  - ExtraTreesRegressor(), 
  - RandomForestRegressor(),
  - KNeighborsRegressor(),
  - MLPRegressor()

|        model_2\model_1        |                      `LinearRegression`                      |                       `....Earth()...`                       |                      `..ElasticNet().`                       |                       `...Lasso()...`                        |                       `...Ridge()...`                        |
| :---------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|     **`XGBRegressor()`**      | ![image-20221207164744676](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207164744676.png) | ![image-20221207165117745](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207165117745.png) | ![image-20221207165955023](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207165955023.png) | ![image-20221207170034827](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207170034827.png) | ![image-20221207170127393](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207170127393.png) |
|  **`ExtraTreesRegressor()`**  | ![image-20221207171318582](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207171318582.png) | ![image-20221207171415709](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207171415709.png) | ![image-20221207171504441](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207171504441.png) | ![image-20221207171606225](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207171606225.png) | ![image-20221207171702435](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207171702435.png) |
| **`RandomForestRegressor()`** | ![image-20221207171805273](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207171805273.png) | ![image-20221207171903417](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207171903417.png) | ![image-20221207171946178](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207171946178.png) | ![image-20221207172040129](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172040129.png) | ![image-20221207172130468](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172130468.png) |
|  **`KNeighborsRegressor()`**  | ![image-20221207172331212](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172331212.png) | ![image-20221207172434502](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172434502.png) | ![image-20221207172513531](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172513531.png) | ![image-20221207172603887](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172603887.png) | ![image-20221207172225386](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172225386.png) |
|     **`MLPRegressor()`**      | ![image-20221207172841890](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172841890.png) | ![image-20221207173005286](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207173005286.png) | ![image-20221207173051593](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207173051593.png) | ![image-20221207172721753](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172721753.png) | ![image-20221207173155566](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207173155566.png) |



```python
model = BoostedHybrid(
    model_1=LinearRegression() ,
    model_2=XGBRegressor() ,
)
```

![image-20221207164744676](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207164744676.png)

```python
model = BoostedHybrid(
    model_1=Earth() ,
    model_2=XGBRegressor() ,
)
```

![image-20221207165117745](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207165117745.png)

```python
model = BoostedHybrid(
    model_1=ElasticNet() ,
    model_2=XGBRegressor() ,
)
```

![image-20221207165955023](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207165955023.png)

```python
model = BoostedHybrid(
    model_1=Lasso() ,
    model_2=XGBRegressor() ,
)
```

![image-20221207170034827](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207170034827.png)

```python
model = BoostedHybrid(
    model_1=Ridge() ,
    model_2=XGBRegressor() ,
)
```

![image-20221207170127393](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207170127393.png)

```python
model = BoostedHybrid(
    model_1=LinearRegression() ,
    model_2=ExtraTreesRegressor() ,
)
```

![image-20221207171331271](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207171331271.png)

```python
model = BoostedHybrid(
    model_1=Earth() ,
    model_2=ExtraTreesRegressor() ,
)
```

![image-20221207171406410](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207171406410.png)

```python
model = BoostedHybrid(
    model_1=ElasticNet() ,
    model_2=ExtraTreesRegressor() ,
)
```

![image-20221207171457663](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207171457663.png)

```python
model = BoostedHybrid(
    model_1=Lasso() ,
    model_2=ExtraTreesRegressor() ,
)
```

![image-20221207171557911](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207171557911.png)

```python
model = BoostedHybrid(
    model_1=Ridge() ,
    model_2=ExtraTreesRegressor() ,
)
```

![image-20221207171655625](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207171655625.png)

```python
model = BoostedHybrid(
    model_1=LinearRegression() ,
    model_2=RandomForestRegressor() ,
)
```

![image-20221207171759457](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207171759457.png)

```python
model = BoostedHybrid(
    model_1=Earth() ,
    model_2=RandomForestRegressor() ,
)
```

![image-20221207171855187](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207171855187.png)

```python
model = BoostedHybrid(
    model_1=ElasticNet() ,
    model_2=RandomForestRegressor() ,
)
```

![image-20221207171946178](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207171946178.png)

```python
model = BoostedHybrid(
    model_1=Lasso() ,
    model_2=RandomForestRegressor() ,
)
```

![image-20221207172048192](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172048192.png)

```python
model = BoostedHybrid(
    model_1=Ridge() ,
    model_2=RandomForestRegressor() ,
)
```

![image-20221207172121122](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172121122.png)

```python
model = BoostedHybrid(
    model_1=LinearRegression() ,
    model_2=KNeighborsRegressor() ,
)
```

![image-20221207172320034](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172320034.png)

```python
model = BoostedHybrid(
    model_1=Earth() ,
    model_2=KNeighborsRegressor() ,
)
```

![image-20221207172426899](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172426899.png)

```python
model = BoostedHybrid(
    model_1=ElasticNet() ,
    model_2=KNeighborsRegressor() ,
)
```

![image-20221207172522825](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172522825.png)

```python
model = BoostedHybrid(
    model_1=Lasso() ,
    model_2=KNeighborsRegressor() ,
)
```

![image-20221207172551384](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172551384.png)

```python
model = BoostedHybrid(
    model_1=Ridge() ,
    model_2=KNeighborsRegressor() ,
)
```

![image-20221207172240961](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172240961.png)

```python
model = BoostedHybrid(
    model_1=LinearRegression() ,
    model_2=MLPRegressor() ,
)
```

![image-20221207172852639](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172852639.png)

```python
model = BoostedHybrid(
    model_1=Earth() ,
    model_2=MLPRegressor() ,
)
```

![image-20221207172948747](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172948747.png)

```python
model = BoostedHybrid(
    model_1=ElasticNet() ,
    model_2=MLPRegressor() ,
)
```

![image-20221207173058971](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207173058971.png)

```python
model = BoostedHybrid(
    model_1=Lasso() ,
    model_2=MLPRegressor() ,
)
```

![image-20221207172703137](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207172703137.png)

```python
model = BoostedHybrid(
    model_1=Ridge() ,
    model_2=MLPRegressor() ,
)
```

![image-20221207173143184](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221207173143184.png)

### 继续前进 

使用四种ML预测策略将任何预测任务转换为机器学习问题。