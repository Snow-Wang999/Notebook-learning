# 74-特征工程-Clustering With K-means

用簇标签（cluster labels）解开复杂的空间关系。

## 介绍 

本课和下一节课使用了所谓的无监督学习算法。无监督算法不使用目标；相反，它们的目的是学习数据的某些属性，以某种方式**表示特征的结构**。在预测特征工程的背景下，您可以将无监督算法视为“特征发现（figure discovery）”技术。 

**聚类（clustering）**简单地意味着根据点之间的相似程度将数据点分配给组。聚类算法可以说是“物以类聚（birds of a feather flock together）”。 

当用于特征工程时，我们可以尝试发现代表某个细分市场（a market segment）的客户群，例如，或共享类似天气模式的地理区域。添加集群标签（cluster labels）的功能可以帮助机器学习模型解开空间或邻近度（proximity）的复杂关系。

## Cluster Labels as a Feature

应用于单个实值特征，聚类就像传统的“装箱（binning）”或“离散化（[discretization](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization_classification.html) ）”变换。在多个特征上，这就像“多维分仓”（有时称为矢量量化（*vector quantization*））。

![image-20221201104444384](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201104444384.png)

左：对单个特征进行聚类。右：跨越两个特征的聚类。

添加到数据帧中，集群标签的功能可能如下所示：

![image-20221201104518010](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201104518010.png)

重要的是要记住，这个集群（Cluster）特征是分类的。这里，它显示了一个标签编码（label encoding）（即，作为一个整数序列），这是典型的聚类算法所产生的；根据您的模型，单热编码（one-hot encoding）可能更合适。 

添加集群标签的动机是，集群将把功能之间的复杂关系分解为更简单的块。然后，我们的模型可以一个接一个地学习简单的块（simpler chunks），而不是一次学习复杂的整体。这是一种“分而治之（divide and conquer）”的策略。

![image-20221201104546153](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201104546153.png)

对`YearBuilt`特性进行聚类有助于该线性模型了解其与`SalePrice`的关系。

该图显示了聚类如何改进简单的线性模型。`YearBuilt`和`SalePrice`之间的曲线关系对于这类模型来说太复杂了——它太低估了。然而，在较小的块上，关系几乎是线性的，并且模型可以很容易地学习。

## k-Means Clustering

有很多聚类算法。它们的主要区别在于如何衡量“相似性（similarity）”或“接近性（proximity）”，以及它们使用的特征类型。我们将使用的算法k-means是直观的，易于在特征工程环境中应用。根据您的应用程序，另一种算法可能更合适。 

K-means聚类使用普通直线距离（换句话说，**欧几里德距离（Euclidean distance）**）来度量相似性。它通过在要素空间内放置多个点（称为质心（**centroids**））来创建簇。数据集中的每个点都被分配给它最接近的质心的簇。“k-means”中的“k”是它创建的质心（即簇）的数量。你自己定义k。 

您可以想象每个质心通过一系列辐射圆（rediating circles）捕捉点。当来自竞争质心的圆组重叠时，它们形成一条直线。结果就是所谓的Voronoi细分（**Voronoi tessallation**）。细分显示了未来数据将分配给哪些集群；细分本质上是k-means从其训练数据中学习到的。 

上述 [*Ames*](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) 数据集上的聚类是k均值聚类。这是与所示的细分和质心相同的图。

![image-20221201105649330](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201105649330.png)

K-means聚类创建特征空间的Voronoi细分。

让我们回顾一下k-means算法是如何学习聚类的，以及这对特征工程意味着什么。我们将关注scikit-learn实现中的三个参数：

- n_clusters、
- max_iter、
- n_init。 

这是一个简单的两步过程。该算法首先随机初始化一些**预定义数量的质心（n_clusters）**。然后迭代这两个操作：

1. 将点指定给最近的簇质心 
2. 移动每个质心以最小化到其点的距离

它在这两个步骤上迭代，直到质心不再移动，或者直到经过了**最大迭代次数（max_iter）**。 

通常情况下，质心的初始随机位置以较差的聚类结束。为此，**算法重复多次（n_init）**，并返回每个点与其质心之间总距离最小的聚类，即最佳聚类（optimal clustering）。 

下面的动画显示了正在运行的算法。它说明了结果对初始质心的依赖性以及迭代直到收敛的重要性。

![image-20221201105849111](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201105849111.png)

![image-20221201105829110](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201105829110.png)

![img](https://i.imgur.com/tBkCqXJ.gif)

纽约Airbnb租赁的K-means聚类算法。

对于大量集群，您可能需要增加`max_iter`，对于复杂的数据集，可能需要增加`n_init`。通常，您需要自己选择的唯一参数是`n_clusters`（即k）。一组功能的最佳分区取决于您使用的模型和您试图预测的内容，因此最好像任何超参数（hyperparameter）一样对其进行调整（例如，通过交叉验证）。

## Example - California Housing

作为空间特征，加州住房（[*California Housing*](https://www.kaggle.com/camnugent/california-housing-prices)）的“纬度（`Latitude`）”和“经度（`Longitude`）”是k均值聚类的自然候选。在本例中，我们将这些与`MedInc`（中等收入）进行聚类，以在加利福尼亚州的不同地区创建经济部门。

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

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

df = pd.read_csv("../input/fe-course-data/housing.csv")
X = df.loc[:, ["MedInc", "Latitude", "Longitude"]]
X.head()
```

![image-20221201112102065](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201112102065.png)

由于k-means聚类对尺度（scale）敏感，因此可以用极值重新缩放（rescale）或标准化（normalize）数据。我们的特征已经大致在尺度上相同，因此我们将保持原样。

```python
# Create cluster feature
kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")

X.head()
```

`.astype("category")`标签编码只是将列中的每个值转换为数字。

[python astype category_Python知识点整理（持续更新）](https://blog.csdn.net/weixin_39943383/article/details/109981797)

![image-20221201113020455](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201113020455.png)

现在让我们看几个图，看看这有多有效。首先，显示集群地理分布的散点图。该算法似乎为沿海高收入地区创建了单独的细分市场。

```python
sns.relplot(
    x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
);
```

![image-20221201113139664](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201113139664.png)

该数据集中的目标是`MedHouseVal`（房屋中值）。这些方框图显示了每个集群中目标的分布。如果聚类是有信息的，那么这些分布在很大程度上应该在`MedHouseVal`中分开，这确实是我们所看到的。

```python
X["MedHouseVal"] = df["MedHouseVal"]
sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6);
```

![image-20221201113324526](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201113324526.png)

## 轮到你了 

向Ames添加集群标签的特征，并了解集群可以创建的另一种特征。

## Exercise: Clustering With K-Means

### 介绍 

在本练习中，您将探索用于创建特征的第一种无监督学习技术，即k均值聚类。 

运行此单元以设置所有内容！

```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.feature_engineering_new.ex4 import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
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


# Prepare data
df = pd.read_csv("../input/fe-course-data/ames.csv")
```

k均值算法**对尺度敏感**。这意味着我们需要考虑如何以及是否重新调整我们的特征，因为根据我们的选择，我们可能会得到非常不同的结果。根据经验，如果这些特性已经可以直接比较（例如不同时间的测试结果），那么您就不需要重新缩放。另一方面，不在可比尺度上的特征（如高度和重量）通常会从重新缩放中受益。有时，选择并不明确。在这种情况下，您应该尝试使用常识，记住值越大的特性权重越大。

#### 1） 缩放功能 

考虑以下几组功能。对于每一项，决定是否： 

- 它们肯定应该重新缩放， 

- 它们绝对不应该被重新缩放，或者 

- 两者都可能是合理的

特征： 

1. `Latitude` and `Longitude` of cities in California
2. `Lot Area` and `Living Area` of houses in Ames, Iowa
3. `Number of Doors` and `Horsepower` of a 1989 model car

思考完答案后，运行下面的单元格进行讨论。

1. 不需要重新缩放，因为重新缩放会扭曲**纬度和经度**所描述的自然距离。 
2. 任何一种选择都可能是合理的，但因为**住宅的居住面积**往往每平方英尺更有价值，所以重新调整这些特征是有意义的，这样，地块面积在聚类中的权重不会与其对SalePrice的影响不成比例，如果这是你试图预测的话。 
3. 需要重新缩放，因为它们**没有可比的单位**。如果不重新调整比例，一辆车的门数（通常为2或4个）与马力（通常为数百个）相比，重量可以忽略不计。

你应该从中得到的是，**是否以及如何重新缩放特征**的决定很少是自动的——这通常取决于有关数据的一些领域知识以及你试图预测的内容。通过交叉验证比较不同的重新缩放方案也很有帮助。（您可能想查看scikit learn中的预处理模块，了解它提供的一些重新缩放方法。）

#### 2） 创建群集标签的特征 

使用以下参数创建k均值聚类： 

- 特征（features）：`LotArea`、`TotalBsmtSF`、`FirstFlrSF`、`SecondFlrSF`和`GrLivArea` 

- 簇数（number of clusters）：10 

- 迭代次数（iterations）：10 

（这可能需要一段时间才能完成。）

```python
X = df.copy()
y = X.pop("SalePrice")


# YOUR CODE HERE: Define a list of the features to be used for the clustering
features = ['LotArea', 'TotalBsmtSF', 'FirstFlrSF', 'SecondFlrSF','GrLivArea']


# Standardize
X_scaled = X.loc[:, features]
X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)


# YOUR CODE HERE: Fit the KMeans model to X_scaled and create the cluster labels
kmeans = KMeans(n_clusters=10,n_init=10,max_iter=300, random_state=0)
X["Cluster"] = kmeans.fit_predict(X_scaled)
```

如果您愿意，可以运行此单元格以查看集群的结果。

```python
Xy = X.copy()
Xy["Cluster"] = Xy.Cluster.astype("category")
Xy["SalePrice"] = y
sns.relplot(
    x="value", y="SalePrice", hue="Cluster", col="variable",
    height=4, aspect=1, facet_kws={'sharex': False}, col_wrap=3,
    data=Xy.melt(
        value_vars=features, id_vars=["SalePrice", "Cluster"],
    ),
);
```

![image-20221201120646488](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201120646488.png)

与之前一样，`score_dataset`将为`XGBoost`模型打分，并将此新功能添加到训练数据中。

```python
score_dataset(X, y)
```

```
0.14046321396663294
```

---

k均值算法提供了一种创建特征的替代方法。它可以测量从点到所有质心的距离，并将这些距离作为特征返回，而不是用最近的簇质心标记每个特征。

### 3） 群集距离特征 (Cluster-Distance Features)

现在将集群距离特征添加到数据集。您可以通过使用kmeans的fit_transform方法而不是fit_predict来获得这些距离特征。

```python
kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)


# YOUR CODE HERE: Create the cluster-distance features using `fit_transform`
X_cd = kmeans.fit_transform(X_scaled)


# Label features and join to dataset
X_cd = pd.DataFrame(X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])])
X = X.join(X_cd)
```

![image-20221201121211985](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221201121211985.png)

如果您愿意，运行此单元格为这些新功能打分。

```python
score_dataset(X, y)
```

```
0.1414333760054382
```

### 继续前进 

应用主成分分析从数据的变化中创建要素。