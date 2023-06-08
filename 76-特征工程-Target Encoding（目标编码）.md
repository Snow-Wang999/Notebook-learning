# 76-特征工程-Target Encoding（目标编码）

## 介绍 

我们在本课程中看到的大多数技术都是针对数值特征的。我们将在本课中讨论的技术，*目标编码*，是用于分类特征的。这是一种将类别编码为数字的方法，就像一个热编码或标签编码，区别在于它还使用*目标*来创建编码。这就是我们所说的监督特征工程技术。

```python
import pandas as pd

autos = pd.read_csv("../input/fe-course-data/autos.csv")
```

## 目标编码 

目标编码是用从目标派生的某个数字替换特征类别的任何类型的编码。 

一个简单有效的版本是应用第3课中的组聚合，就像均值一样。使用汽车数据集，计算每辆汽车的平均价格：

```python
autos["make_encoded"] = autos.groupby("make")["price"].transform("mean")

autos[["make", "price", "make_encoded"]].head(10)
```

![image-20221201200702599](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201200702599.png)

这种目标编码有时被称为均值编码（mean encoding）。应用于二进制目标，也称为二进制计数（bin counting）。（您可能遇到的其他名称包括：似然编码（likelihood encoding）、影响编码（impact encoding）和遗漏编码（leave-one-out encoding）。）

## 平滑的 （Smoothing）

然而，这样的编码会带来一些问题。首先是**未知类别**。目标编码会产生<u>*过度拟合*</u>的特殊风险，这意味着它们需要在独立的“编码”分割上进行训练。当您将编码连接到将来的拆分时，Pandas将为编码拆分中不存在的任何类别填充缺失的值。这些缺失的值必须以某种方式计算。 

其次是**稀有类别**。当一个类别在数据集中只出现几次时，对其组计算的任何统计数据都不太可能非常准确。在Automobiles数据集中，商品（`mercurcy`）制造只发生一次。我们计算的“平均”价格只是这辆车的价格，这可能不是我们未来可能看到的任何Mercuries的代表。对罕见类别进行目标编码可能会导致<u>*过度拟合*</u>。 

解决这些问题的方法是添加**平滑**。其理念是**将类别内的平均值与总体平均值相结合**。罕见的类别在其类别平均值上的权重较小，而缺失的类别仅获得总体平均值。 

在伪代码中：

```python
encoding = weight * in_category + (1 - weight) * overall
```

其中权重是从类别频率计算的介于0和1之间的值。 

确定权重值的一种简单方法是计算m估计值：

```python
weight = n / (n + m)
```

其中n是该类别在数据中出现的总次数。参数m决定“平滑因子”。m值越大，对总体估计的权重越大。

![image-20221201201243937](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201201243937.png)

在Automobiles数据集中，有三辆车的品牌为雪佛兰。如果您选择m=2.0，那么雪佛兰类别将以雪佛兰平均价格的60%加上整体平均价格的40%进行编码。

```python
chevrolet = 0.6 * 6000.00 + 0.4 * 13285.03
```

在选择m的值时，请考虑**类别的噪声程度**。每个品牌的汽车**价格是否相差很大**？你需要大量的数据来获得好的估计吗？如果是这样的话，最好为**m选择一个更大的值**；如果每个品牌的平均价格相对稳定，那么较小的价格也可以。

> **目标编码用例** 
>
> 目标编码适用于： 
>
> - **高基数特性（High-cardinality features）**：具有<u>*大量类别的特征*</u>可能很难编码：一次热编码会产生太多的特性，而替代方案（如标签编码）可能不适合该特性。<u>目标编码使用特征最重要的属性（即与目标的关系）导出类别的编号</u>。 
>
> - **领域驱动的特征（Domain-motivated features）**：根据以前的经验，您可能会怀疑分类特征应该很重要，即使它在特征度量中得分很低。目标编码可以帮助揭示**特征的真实信息性**。

## Example - MovieLens1M

MovieLens1M数据集包含MovieLens网站用户对100万部电影的评分，其中包含描述每个用户和电影的功能。此隐藏单元格设置所有内容：

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

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
warnings.filterwarnings('ignore')


df = pd.read_csv("../input/fe-course-data/movielens1m.csv")
df = df.astype(np.uint8, errors='ignore') # reduce memory footprint
print("Number of Unique Zipcodes: {}".format(df["Zipcode"].nunique()))
```

```
Number of Unique Zipcodes: 3439
```

Zipcode功能有3000多个类别，是目标编码的一个很好的候选，而这个数据集的大小（超过一百万行）意味着我们可以腾出一些数据来创建编码。 

我们将首先创建25%的分割来训练目标编码器。

```python
X = df.copy()
y = X.pop('Rating')

X_encode = X.sample(frac=0.25)
y_encode = y[X_encode.index]
X_pretrain = X.drop(X_encode.index)
y_train = y[X_pretrain.index]
```

`scikit-learn-contrib`中的`category_encoders`包实现了一个**m估计编码器**，我们将使用它来编码Zipcode特性。

```python
from category_encoders import MEstimateEncoder

#创建编码器实例。选择m以控制噪声。
# Create the encoder instance. Choose m to control noise.
encoder = MEstimateEncoder(cols=["Zipcode"], m=5.0)

#将编码器安装在编码分割上。
# Fit the encoder on the encoding split.
encoder.fit(X_encode, y_encode)

#对Zipcode列进行编码以创建最终训练数据
# Encode the Zipcode column to create the final training data
X_train = encoder.transform(X_pretrain)
```

让我们将编码值与目标值进行比较，以了解编码的信息量。

```python
plt.figure(dpi=90)
ax = sns.distplot(y, kde=False, norm_hist=True)
ax = sns.kdeplot(X_train.Zipcode, color='r', ax=ax)
ax.set_xlabel("Rating")
ax.legend(labels=['Zipcode', 'Rating']);
```

![image-20221201202123567](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201202123567.png)

编码后的Zipcode特性的分布大致遵循实际收视率的分布，这意味着电影观众在不同的Zipcode之间的收视率差异很大，因此我们的目标编码能够捕捉到有用的信息。

## 轮到你了 

将目标编码应用于Ames中的特征，并研究目标编码可能导致过度拟合的令人惊讶的方式。

## Exercise: Target Encoding

### 介绍 

在本练习中，您将对Ames数据集中的特征应用目标编码。 运行此单元以设置所有内容！

```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.feature_engineering_new.ex6 import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from category_encoders import MEstimateEncoder
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
warnings.filterwarnings('ignore')


def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


df = pd.read_csv("../input/fe-course-data/ames.csv")
```

首先，您需要选择要应用目标编码的功能。具有大量类别的类别功能通常是很好的候选。运行此单元格以查看Ames数据集中每个分类特征有多少类别。

```python
df.select_dtypes(["object"]).nunique()
```

```
MSSubClass       16
MSZoning          7
Street            2
Alley             3
LotShape          4
LandContour       4
Utilities         3
LotConfig         5
LandSlope         3
Neighborhood     28
Condition1        9
Condition2        8
BldgType          5
HouseStyle        8
OverallQual      10
OverallCond       9
RoofStyle         6
RoofMatl          8
Exterior1st      16
Exterior2nd      17
MasVnrType        5
ExterQual         4
ExterCond         5
Foundation        6
BsmtQual          6
BsmtCond          6
BsmtExposure      5
BsmtFinType1      7
BsmtFinType2      7
Heating           6
HeatingQC         5
CentralAir        2
Electrical        6
KitchenQual       5
Functional        8
FireplaceQu       6
GarageType        7
GarageFinish      4
GarageQual        6
GarageCond        6
PavedDrive        3
PoolQC            5
Fence             5
MiscFeature       6
SaleType         10
SaleCondition     6
dtype: int64
```

我们讨论了M估计编码如何使用平滑来改善稀有类别的估计。要查看类别在数据集中出现的次数，可以使用value_counts方法。此单元格显示SaleType的计数，但您可能也需要考虑其他值。

```python
df["SaleType"].value_counts()
```

```
WD       2536
New       239
COD        87
ConLD      26
CWD        12
ConLI       9
ConLw       8
Oth         7
Con         5
VWD         1
Name: SaleType, dtype: int64
```

### 1） 选择编码功能 

您为目标编码确定了哪些特征？思考好答案后，运行下一个单元格进行讨论。

```
MSSubClass       16
Neighborhood     28
Exterior1st      16
Exterior2nd      17
OverallQual      10
SaleType         10
```

`Neighborhood`特征看起来很有前途。它是所有特征中种类最多的，而且很少有几个种类。其他值得考虑的是`SaleType`、`MSSubClass`、`Exterior1st`和`Exterior2nd`。事实上，由于稀有类别的盛行，几乎所有的名义特征都值得尝试。

***

现在，您将对选择的特征应用目标编码。正如我们在教程中所讨论的，为了避免过度拟合，我们需要在训练集中保留的数据上拟合编码器。运行此单元格以创建编码和训练分割：

```python
# Encoding split
X_encode = df.sample(frac=0.20, random_state=0)
y_encode = X_encode.pop("SalePrice")

# Training split
X_pretrain = df.drop(X_encode.index)
y_train = X_pretrain.pop("SalePrice")
```

### 2） 应用M估计编码

将目标编码应用于分类特征的选择。还要为平滑参数m选择一个值（任何值都可以得到正确答案）。

```python
# YOUR CODE HERE: Create the MEstimateEncoder
# Choose a set of features to encode and a value for m
features = ['Neighborhood','MSSubClass','Exterior1st','Exterior2nd','OverallQual','SaleType']
encoder = MEstimateEncoder(cols=features,m=5.0)

# Fit the encoder on the encoding split
encoder.fit(X_encode, y_encode)

# Encode the training split
X_train = encoder.transform(X_pretrain, y_train)
```

如果您想查看编码功能与目标的比较，可以运行以下单元格：

```python
feature = encoder.cols

plt.figure(dpi=90)
ax = sns.distplot(y_train, kde=True, hist=False)
ax = sns.distplot(X_train[feature], color='r', ax=ax, hist=True, kde=False, norm_hist=True)
ax.set_xlabel("SalePrice");
```

![image-20221201224753347](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201224753347.png)

从分布图来看，编码是否具有信息性？ 

此单元格将显示编码集与原始集的比较分数：

```python
X = df.copy()
y = X.pop("SalePrice")
score_base = score_dataset(X, y)
score_new = score_dataset(X_train, y_train)

print(f"Baseline Score: {score_base:.4f} RMSLE")
print(f"Score with Encoding: {score_new:.4f} RMSLE")
```

```
Baseline Score: 0.1428 RMSLE
Score with Encoding: 0.1410 RMSLE
```

在这种情况下，您认为目标编码值得吗？根据您选择的一个或多个功能，您可能会得到比基线更差的分数。在这种情况下，编码获得的额外信息很可能无法弥补用于编码的数据丢失。

----

在这个问题中，您将探讨目标编码过拟合的问题。这将说明在从训练集中获得的数据上训练拟合目标编码器的重要性。 因此，让我们看看当我们将编码器和模型放在同一个数据集上时会发生什么。为了强调过度拟合的戏剧性，我们的意思是编码一个与SalePrice无关的特征，一个计数：0、1、2、3、4、5。。。。

```python
# Try experimenting with the smoothing parameter m
# Try 0, 1, 5, 50
m = 0

X = df.copy()
y = X.pop('SalePrice')

# Create an uninformative feature
X["Count"] = range(len(X))
X["Count"][1] = 0  # actually need one duplicate value to circumvent error-checking in MEstimateEncoder

# fit and transform on the same dataset
encoder = MEstimateEncoder(cols="Count", m=m)
X = encoder.fit_transform(X, y)

# Results
score =  score_dataset(X, y)
print(f"Score: {score:.4f} RMSLE")
```

当m = 0，

```
Score: 0.0293 RMSLE
```

当m = 1，

```
Score: 0.0298 RMSLE
```

当m = 5，

```
Score: 0.0291 RMSLE
```

当m = 50，

```
Score: 0.0303 RMSLE
```

几乎是满分！

```python
plt.figure(dpi=90)
ax = sns.distplot(y, kde=True, hist=False)
ax = sns.distplot(X["Count"], color='r', ax=ax, hist=True, kde=False, norm_hist=True)
ax.set_xlabel("SalePrice");
```

当m = 0，

![image-20221201225943397](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201225943397.png)

当m = 1，

![image-20221201230049956](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201230049956.png)

当m = 5，

![image-20221201230116194](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201230116194.png)

当m = 50，

![image-20221201225826939](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201225826939.png)

分布也几乎完全相同。

### 3） 与目标编码器过度匹配 

基于您对**均值编码**工作原理的理解，您能否解释XGBoost如何在均值编码计数功能后获得近乎完美的匹配？

由于`Count`从不具有任何重复值，因此平均编码的`Count`本质上是目标的精确副本。换句话说，均值编码将一个完全无意义的特征变成了一个完美的特征。 

现在，唯一有效的原因是我们在训练编码器的同一组上训练XGBoost。如果我们使用了一个保持(hold-out)集，那么这些“假”编码都不会转移到训练数据中。 

教训是，当使用目标编码器时，使用**单独的数据集**来训练编码器和训练模型非常重要。否则，结果可能非常令人失望！

### 结束 

这就是特征工程！我们希望您在我们这里过得愉快。 

现在，你准备好尝试你的新技能了吗？现在是加入我们的房价（ [Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)）入门比赛的好时机。我们甚至准备了一节“奖励课（[Bonus Lesson](https://www.kaggle.com/ryanholbrook/feature-engineering-for-house-prices)）”，将我们一起完成的所有工作汇总到入门笔记本中。

### 工具书类 

这里有一些很好的资源，您可能想查阅更多信息。他们都在塑造这一过程中发挥了作用： 

- 《特征工程的艺术》，巴勃罗·杜布的书。

  *The Art of Feature Engineering*, a book by Pablo Duboue.

- 杰夫·希顿（Jeff Heaton）的一篇文章《预测建模的特征工程实证分析》。 

  *An Empirical Analysis of Feature Engineering for Predictive Modeling*, an article by Jeff Heaton.

- Alice Zheng和Amanda Casari的《机器学习的特征工程》一书。集群教程的灵感来自这本优秀的书。

  *Feature Engineering for Machine Learning*, a book by Alice Zheng and Amanda Casari. The tutorial on clustering was inspired by this excellent book.

- Max Kuhn和Kjell Johnson的《特征工程与选择》一书。

  *Feature Engineering and Selection*, a book by Max Kuhn and Kjell Johnson.

 





