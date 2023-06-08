# 75-特征工程-Principal Component Analysis（主成分分析）

通过分析变化发现新功能。

## 介绍 

在上一课中，我们研究了第一种基于模型的特征工程方法：聚类。在这节课中，我们看下一节课：**主成分分析（PCA）**。就像聚类是基于接近度（proximity）对数据集的划分（partition）一样，您可以将PCA视为**对数据变化的划分**。PCA是一个很好的工具，可以帮助您发现数据中的重要关系，也可以用于创建更多信息的特征。 

（技术说明：PCA通常应用于标准化（[standardized](https://www.kaggle.com/alexisbcook/scaling-and-normalization)）数据。对于标准化数据，“变异（variation）”表示“相关性（correlation）”。对于非标准化数据“变异（variation）”意味着“协方差（covariance）”。在应用PCA之前，本课程中的所有数据都将被标准化。）

## Principal Component Analysis（主成分分析）

在鲍鱼（[*Abalone*](https://www.kaggle.com/rodolfomendes/abalone-dataset)）数据集中，有数千只塔斯马尼亚鲍鱼的物理测量结果。（鲍鱼是一种很像蛤蜊或牡蛎的海洋生物。）我们现在只看两个特征：它们外壳的“高度（height）”和“直径（diameter）”。 

你可以想象，在这些数据中有“变化轴（axes of variation）”，它们描述了鲍鱼之间的差异。从图片上看，这些轴显示为沿数据的自然维度延伸的垂直线，每个原始特征对应一个轴。

![image-20221201123503607](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201123503607.png)

通常，我们可以为这些变化轴命名。我们可以称之为“Size”组件的长轴：小高度和小直径（左下）与大高度和大直径（右上）形成对比。我们可以称之为“Shape”的较短轴：小高度和大直径（扁平形状）与大高度和小直径（圆形形状）形成对比。 

请注意，我们可以用“大小”和“形状”来描述鲍鱼，而不是用“高度”和“直径”来描述它们。事实上，这就是PCA的全部思想：我们用变化轴来描述数据，而不是用原始特征来描述数据。变化轴成为新的特征。

![image-20221201125919780](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201125919780.png)

通过在特征空间中**旋转数据集**，主成分成为新特征。

新特征PCA构造实际上只是原始特征的线性组合（加权和）：

```python
df["Size"] = 0.707 * X["Height"] + 0.707 * X["Diameter"]
df["Shape"] = 0.707 * X["Height"] - 0.707 * X["Diameter"]
```

这些新特性称为数据的主要组成部分。权重本身称为**载荷（loading）**。原始数据集中的主要组件将与特征数量一样多：如果我们使用了十个特征而不是两个，那么我们将得到十个组件。 

组件的负载通过符号和大小告诉我们它表达了什么变化：

![image-20221201130022178](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201130022178.png)

该载荷表告诉我们，在“尺寸”组件中，“高度”和“直径”在同一方向（相同符号）变化，但在“形状”组件中它们在相反方向（相反符号）变化。在每个组件中，负载都是相同的大小，因此特征在两者中的贡献相等。 

PCA还告诉我们**每个成分的变化量**。我们可以从图中看到，与形状组件相比，大小组件的数据变化更大。PCA通过每个分量的解释方差百分比（**percent of explained variance**）来精确计算。

![image-20221201130101508](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201130101508.png)

在高度和直径之间的差异中，尺寸约占96%，形状约占4%。

“大小”组件捕捉“高度”和“直径”之间的大部分变化。然而，重要的是要记住，一个组件的方差量不一定对应于它作为一个预测值的好坏：这取决于你试图预测的是什么。

## PCA for Feature Engineering（特征工程PCA）

有两种方法可以将PCA用于特征工程。 

第一种方法是将其用作一种**描述性技术**。由于组件告诉你变化，你可以计算组件的MI分数，看看哪种变化最能预测你的目标。这可以为您提供各种要创建的特征的想法——例如，如果“尺寸”很重要，则为“高度”和“直径”的乘积；如果“形状”很重要的话，则为高度和直径的比值。您甚至可以尝试在一个或多个高分组件上进行聚类。

第二种方法是将**组件本身**用作特征。因为组件直接暴露了数据的变化结构，所以它们通常比原始特征更具信息性。以下是一些用例：

- **降维（Dimensionality reduction）**：当你的特征高度冗余（特别是多重共线性）时，PCA会将**冗余**划分为一个或多个接近零的方差分量，然后你可以删除这些分量，因为它们几乎不包含信息。 
- **异常检测（Anomaly detection）**：从原始特征看不明显的异常变化通常会在**低方差分量**中出现。这些组件在异常或异常值检测任务中可能具有**高度的信息量**。 
- **降噪（Noise reduction）**：传感器读数的集合通常会共享一些常见的背景噪声。PCA有时可以将（信息性的）信号收集成较少数量的特征，而不考虑噪声，从而**提高信噪比**。 
- **去相关（Decorrelation）**：一些ML算法难以**处理高度相关的特征**。PCA将相关的特征转换为不相关的成分，这对于您的算法来说更容易处理。

PCA基本上可以让您直接访问数据的相关结构。毫无疑问，你会想出自己的应用程序！

> **PCA最佳实践** 
>
> 在应用PCA时，需要记住以下几点： 
>
> - PCA仅适用于**数字特征**，如连续数量或计数。 
> - PCA**对尺度敏感**。在应用PCA之前对数据进行标准化是很好的做法，除非你知道自己有充分的理由不这样做。 
> - 考虑**删除或限制异常值**，因为它们可能会对结果产生不适当的影响。

## Example - 1985 Automobiles

在本例中，我们将返回汽车数据集并应用PCA，将其作为一种**描述性**技术来发现特征。我们将在练习中查看其他用例。 

此隐藏单元加载数据并定义函数`plot_variance`和`make_mi_scores`。

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.feature_selection import mutual_info_regression


plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

#还需进一步了解细节
def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


df = pd.read_csv("../input/fe-course-data/autos.csv")
```

我们选择了涵盖一系列属性的四个功能。这些功能中的每一个都具有较高的MI分数，目标是价格。我们将对数据进行标准化，因为这些特性自然不在同一尺度上。

```python
features = ["highway_mpg", "engine_size", "horsepower", "curb_weight"]

X = df.copy()
y = X.pop('price')
X = X.loc[:, features]

# Standardize
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
```

现在我们可以拟合scikit-learn的PCA估计器并创建主成分。这里可以看到转换数据集的前几行。

```python
from sklearn.decomposition import PCA

# Create principal components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

X_pca.head()
```

![image-20221201130441695](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201130441695.png)

拟合后，PCA实例在其`components_`属性中包含加载。（不幸的是，PCA的术语不一致。我们遵循惯例，将X_PCA中的转换列称为组件，否则这些列没有名称。）我们将在数据帧中打包加载。

```python
loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=component_names,  # so the columns are the principal components
    index=X.columns,  # and the rows are the original features
)
loadings
```

![image-20221201130520437](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201130520437.png)

回想一下，组件负载的符号和大小（振幅）告诉我们它捕获了什么样的变化。第一个组件（PC1）显示了大型、动力强劲、气体同化能力差的车辆和小型、更经济、气体同化性能好的车辆之间的对比。我们可以称之为“豪华/经济”轴。下一张图显示，我们选择的四个功能主要沿豪华/经济轴变化。

```python
# Look at explained variance
plot_variance(pca);
```

![image-20221201130551720](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201130551720.png)

我们还来看看组件的MI分数。毫不奇怪，PC1具有高度的信息量，尽管其余组件尽管差异很小，但仍与价格有重要关系。检查这些组成部分可能有助于找到奢侈品/经济主轴没有捕捉到的关系。

```python
mi_scores = make_mi_scores(X_pca, y, discrete_features=False)
mi_scores
```

```
PC1    1.013710
PC2    0.378915
PC3    0.306730
PC4    0.203985
Name: MI Scores, dtype: float64
```

第三部分显示了马力和车重之间的对比——跑车和货车（sports cars vs. wagons），看起来。

```python
# Show dataframe sorted by PC3
idx = X_pca["PC3"].sort_values(ascending=False).index
cols = ["make", "body_style", "horsepower", "curb_weight"]
df.loc[idx, cols]
```

![image-20221201130657892](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201130657892.png)

为了表达这种对比，让我们创建一个新的比率特征：

```python
df["sports_or_wagon"] = X.curb_weight / X.horsepower
sns.regplot(x="sports_or_wagon", y='price', data=df, order=2);
```

![image-20221201130730409](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201130730409.png)

## 轮到你了 

通过分解Ames Housing中的变化来改进您的特征集，并使用主要组件来检测异常值。

## Exercise: Principal Component Analysis

### 介绍 

在本练习中，您将完成对[*Ames*](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)数据集的PCA应用。

运行此单元以设置所有内容！

```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.feature_engineering_new.ex5 import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
```

#### 自定义函数apply_pca

[【python】sklearn中PCA的使用方法](https://blog.csdn.net/qq_20135597/article/details/95247381)

[PCA中n_components的设置](https://blog.csdn.net/weixin_41857483/article/details/109604239)

```python
def apply_pca(X, standardize=True):
    # Standardize标准化
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components创建主成分
    pca = PCA()
    X_pca = pca.fit_transform(X)
    #用X来训练PCA模型，同时返回降维后的数据。
    # Convert to dataframe
    # 创建主成分的dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    # 创建载量
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    return pca, X_pca, loadings
```

#### 自定义函数plot_variance

```python
def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    # PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n
    grid = np.arange(1, n + 1)
    #grid是等差数列，[1,2,...,n+1]
    # Explained variance
    evr = pca.explained_variance_ratio_
    #通过.explained_variance_ratio_返回所保留的n个成分的方差百分比
    axs[0].bar(grid, evr)
    #bar是条形图
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    # 累计差异
    cv = np.cumsum(evr)
    # 前i个特征总共在原始数据信息占比
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs
```

#### numpy.arange()

`numpy.arange（[ start，] stop，[ step，] dtype = None ）`
在给定间隔内返回均匀间隔的值。

也就是说，np.arange()函数生成的是一个[等差数列](https://so.csdn.net/so/search?q=等差数列&spm=1001.2101.3001.7020)

#### 自定义函数make_mi_scores

计算互信息的影响

```python
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # factorize函数可以将Series中的标称型数据映射称为一组数字,相同的标称型映射为相同的数字。
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    #mutual_info_regression离散目标变量的互信息
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
```

### 自定义函数score_dataset

计算交叉验证的均方根对数误差

```python
def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
        # factorize函数可以将Series中的标称型数据映射称为一组数字,相同的标称型映射为相同的数字。
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score
```

```python
df = pd.read_csv("../input/fe-course-data/ames.csv")
```

让我们选择一些与我们的目标SalePrice高度相关的功能。

```python
features = [
    "GarageArea",
    "YearRemodAdd",
    "TotalBsmtSF",
    "GrLivArea",
]

print("Correlation with SalePrice:\n")
print(df[features].corrwith(df.SalePrice))
```

![image-20221201163221630](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201163221630.png)

我们将依靠PCA来解开这些特征的相关结构，并提出可以用新特征进行有效建模的关系。 运行此单元以应用PCA并提取负载。

```python
X = df.copy()
y = X.pop("SalePrice")
X = X.loc[:, features]

# `apply_pca`, defined above, reproduces the code from the tutorial
pca, X_pca, loadings = apply_pca(X)
print(loadings)
```

![image-20221201163350412](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201163350412.png)

### 1） 解释组件加载 

看看PC1和PC3组件的负载。你能想象一下每个组件捕获了什么样的对比度吗？在你考虑过之后，运行下一个单元格以获得解决方案。

第一个组件PC1似乎是一种“大小Size”组件，类似于我们在教程中看到的：所有功能都具有相同的符号（正），表明该组件描述了这些功能的大值房屋和小值房屋之间的对比。 

第三组分PC3的解释稍微复杂一些。`GarageArea`和`YearRemovedAdd`的功能都几乎为零，所以我们忽略它们。该组件主要与`TotalBsmtSF`和`GrLivArea`有关。它描述了居住面积很大但地下室很小（或不存在）的房子与相反的房子之间的对比：小房子和大地下室。

***

这个问题的目标是使用PCA的结果来发现一个或多个新的特征，以提高模型的性能。一个选项是创建受加载启发的功能，就像我们在教程中所做的那样。另一种选择是使用组件本身作为特征（即，向X添加一列或多列X_pca）。

### 2） 创建新特征 

向数据集X添加一个或多个新功能。要获得正确的解决方案，请获得低于0.140 RMSLE的验证分数。（如果你被卡住了，请随时使用下面的提示！）

```python
# Solution 1: Inspired by loadings
X = df.copy()
y = X.pop("SalePrice")

X["Feature1"] = X.GrLivArea + X.TotalBsmtSF
X["Feature2"] = X.YearRemodAdd * X.TotalBsmtSF

score = score_dataset(X, y)
print(f"Your score: {score:.5f} RMSLE")


# Solution 2: Uses components
X = df.copy()
y = X.pop("SalePrice")

X = X.join(X_pca)
score = score_dataset(X, y)
print(f"Your score: {score:.5f} RMSLE")
```

```
Your score: 0.13707 RMSLE
```

***

下一个问题探讨了如何**使用PCA检测数据集中的异常值**（即在某种程度上异常极端的数据点）。异常值可能会对模型性能产生不利影响，因此，如果您需要采取纠正措施，最好了解这些异常值。PCA尤其可以向你展示从原始特征中可能看不到的异常变化：无论是小房子还是有大型地下室的房子都是不寻常的，但对于小房子来说，拥有大型地下室是很不寻常的。这是一个主要组件可以向你展示的东西。 

运行下一个单元格以显示上面创建的每个主要组件的分布图。

```python
sns.catplot(
    y="value",
    col="variable",
    data=X_pca.melt(),
    kind='boxen',
    sharey=False,
    col_wrap=2,
);
```

![image-20221201185551887](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201185551887.png)

![image-20221201185853710](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201185853710.png)

正如您所看到的，在每一个组件中，都有几个点位于分布的最末端，即异常值。 

现在运行下一个单元，查看位于某个组件极端的房屋：

```python
# You can change PC1 to PC2, PC3, or PC4
component = "PC1"

idx = X_pca[component].sort_values(ascending=False).index
df.loc[idx, ["SalePrice", "Neighborhood", "SaleCondition"] + features]
```

![image-20221201190326014](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201190326014.png)

## 3） 孤立点检测(Outlier Detection)

你注意到极值中的任何模式吗？异常值似乎来自数据的某个特殊子集吗？ 

思考完答案后，运行下一个单元格，查找解决方案并进行一些讨论。

请注意，在 `Edwards` 社区，有几处住宅被列为部分（`Partial`）销售。部分出售是指当一处房产有多个所有者，其中一个或多个所有者出售其对该房产的“部分”所有权时发生的情况。 

这类销售通常发生在家族财产结算或企业解散期间，不会公开发布广告。如果你试图预测公开市场上房子的价值，你可能有理由从你的数据集中删除这样的销售额——它们确实是异常值。

### 继续前进 

应用目标编码来增强分类特征。