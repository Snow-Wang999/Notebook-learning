# 72-特征工程-Mutual Information（交互信息）

找到最有潜力的功能。

## 介绍 

第一次遇到一个新的数据集有时会感到不知所措。您可能会看到数百或数千个功能，甚至没有描述。你从哪里开始？ 

一个很好的第一步是用特征效用度量（**feature utility metric**）构建一个排名，这是一个度量特征和目标之间关联的函数。然后，您可以选择一组较小的最有用的特性进行初始开发，并更有信心充分利用您的时间。 

我们将使用的度量称为“相互信息（mutual information）”。相互信息很像相关性（correlation），因为它测量两个量之间的关系。相互信息的优点是它可以**检测任何类型的关系**，而**相关性只检测线性关系**。

相互信息是一个很好的通用指标，在功能开发开始时，当您可能还不知道要使用什么模型时，它尤其有用。它是： 

- 易于使用和解释， 
- 计算效率高， 
- 理论上是有根据的， 
- 阻止过拟合， 
- 能够检测任何类型的关系

## Mutual Information and What it Measures

交互信息度量了两个变量之间相互依赖的程度。相互信息以**不确定性**描述关系。【两个量之间的交互信息（MI）是对一个知识的信息量的减去另一个信息量的不确定性的程度的度量。】

对于两个随机变量，MI是一个随机变量由于已知另一个随机变量而减少的“信息量”（单位通常为比特）。互信息的概念与随机变量的[熵](https://zh.wikipedia.org/wiki/熵_(信息论))紧密相关，[熵](https://zh.wikipedia.org/wiki/熵_(信息论))是[信息论](https://zh.wikipedia.org/wiki/信息论)中的基本概念，它量化的是随机变量中所包含的“[信息量](https://zh.wikipedia.org/w/index.php?title=信息量&action=edit&redlink=1)”。

MI不仅仅是度量实值随机变量和线性相关性(如相关系数)，它更为通用。MI决定了随机变量${\displaystyle {\displaystyle (X,Y)}}$的联合分布与${\displaystyle X}$和${\displaystyle Y}$的边缘分布的乘积之间的差异。MI是点互信息（Pointwise Mutual Information，PMI）的期望。克劳德·香农在他的论文A Mathematical Theory of Communication中定义并分析了这个度量，但是当时他并没有将其称为“互信息”。这个词后来由罗伯特·法诺[1]创造。互信息也称为信息增益。

### 互信息的定义

设随机变量$(X,Y)$是空间$X\times Y$中的一对随机变量。若他们的联合分布是$p(x,y)$，边缘分布是 $p(x)$ 和 $p(y)$ ，那么，他们之间的互信息可以定义为：
$$
I(X;Y)=D_{KL}(p(x,y)\vert|p(x)\otimes p(y))
$$
其中，$D_{KL}$ 是KL散度（Kullback-Leibler divergence）。注意，根据KL散度的性质，若联合分布 $p(x,y)$ 等于边缘分布是 $p(x)$ 和 $p(y)$ 的乘积，则 $I(X;Y)=0$，即当 $X$ 和 $Y$ 相互独立的时候，观测到Y对于我们预测X没有任何帮助，此时他们的互信息为0。

### 离散变量的互信息

离散随机变量 X 和 Y 的互信息可以计算为：
$$
I(X;Y)=\sum_{y \in Y}{\sum_{x \in X}{log(\frac{p(x,y)}{p(x)p(y)})}}
$$
其中 p(x, y) 是 X 和 Y 的联合概率质量函数，而 $p(x)$ 和 $p(y)$  分别是 X 和 Y 的边缘概率质量函数。

### 连续变量的互信息

在[连续随机变量](https://zh.wikipedia.org/wiki/连续函数)的情形下，求和被替换成了[二重定积分](https://zh.wikipedia.org/wiki/二重积分)：
$$
I(X;Y)=\int_{Y}{\int_{X}{log(\frac{p(x,y)}{p(x)p(y)})dxdy}}
$$
其中 *p*(*x*, *y*) 当前是 *X* 和 *Y* 的联合概率*密度*函数，而 $p(x)$ 和 $p(y)$ 分别是 *X* 和 *Y* 的边缘概率密度函数。

如果对数以 2 为基底，互信息的单位是[bit](https://zh.wikipedia.org/wiki/位元)。

直观上，互信息度量 *X* 和 *Y* 共享的信息：它度量知道这两个变量其中一个，对另一个不确定度减少的程度。例如，如果 *X* 和 *Y* 相互独立，则知道 *X* 不对 *Y* 提供任何信息，反之亦然，所以它们的互信息为零。在另一个极端，如果 *X* 是 *Y* 的一个确定性函数，且 *Y* 也是 *X* 的一个确定性函数，那么传递的所有信息被 *X* 和 *Y* 共享：知道 *X* 决定 *Y* 的值，反之亦然。因此，在此情形互信息与 *Y*（或 *X*）单独包含的不确定度相同，称作 *Y*（或 *X*）的[熵](https://zh.wikipedia.org/wiki/信息熵)。而且，这个互信息与 *X* 的熵和 *Y* 的熵相同。（这种情形的一个非常特殊的情况是当 *X* 和 *Y* 为相同随机变量时。）

互信息是 *X* 和 *Y* 的[联合分布](https://zh.wikipedia.org/wiki/联合分布)相对于假定 *X* 和 *Y* 独立情况下的联合分布之间的内在依赖性。 于是互信息以下面方式度量依赖性：*I*(*X*; *Y*) = 0 [当且仅当](https://zh.wikipedia.org/wiki/当且仅当) *X* 和 *Y* 为独立随机变量。从一个方向很容易看出：当 *X* 和 *Y* 独立时，*p*(*x*,*y*) = *p*(*x*) *p*(*y*)，因此：
$$
log(\frac{p(x,y)}{p(x)p(y)})=log1=0
$$
此外，互信息是非负的（即 ${\displaystyle I(X;Y)\geq 0}$; 见下文），而且是[对称的](https://zh.wikipedia.org/w/index.php?title=对称函数&action=edit&redlink=1)（即 ${\displaystyle I(X;Y)=I(Y;X)}$ )。

***

补充：$\otimes$ 是[克罗内克积](https://baike.baidu.com/item/%E5%85%8B%E7%BD%97%E5%86%85%E5%85%8B%E7%A7%AF/6282573?fr=aladdin)，是张量积的特殊形式

![image-20221122191144909](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122191144909.png)

### 举例

如果你知道某个功能的价值，你会对目标更有信心吗？ 

这是Ames住房数据的一个例子。该图显示了房屋外观质量与售价之间的关系。每个点代表一座房子。

![image-20221122182633631](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122182633631.png)

从图中，我们可以看到，知道`ExterQual`的值应该可以让您更加确定相应的`SalePrice`——`ExterQual`的每个类别都倾向于将`SalePrice`集中在一定范围内。`ExterQual`与`SalePrice`之间的相互信息是，在`ExterQual`的四个值上，`SalePrice`的不确定性平均减少。例如，由于`Fair`发生的频率低于`Typical`，`Fair`在MI评分中的权重较小。

（技术说明：我们所说的不确定性是用信息理论中的一个量“熵”来衡量的。一个变量的熵大致意思是：“平均来说，你需要多少个是或否的问题来描述这个变量的发生。”。“你要问的问题越多，你对变量的不确定性就越大。相互信息就是你希望该特征能回答多少关于目标的问题。”

## 解释相互信息得分

数量之间最不可能的相互信息是0.0。当MI为零时，数量是独立的：两者都不能告诉你关于另一个的任何信息。相反，理论上，MI没有上限。在实践中，尽管值高于2.0左右是不常见的。（相互信息是一个对数的量，因此增长非常缓慢。） 

下一个图将让您了解MI值如何对应于特征与目标的关联类型和程度。

![image-20221122183214476](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122183214476.png)

左：互信息随着特征和目标之间的依赖性变得更紧密而增加。

右：相互信息可以捕捉任何类型的关联（不只是线性的，比如相关性）

在应用相互信息时，请记住以下几点： 

- MI可以帮助您了解一个特征作为目标预测因子的相对潜力，并自行考虑。 
- 当一个特征与其他特征交互时，它可能会提供非常丰富的信息，但不是单独提供信息。MI无法检测特征之间的交互。它是一个单变量（univariate）度量。 
- 功能的实际有用性取决于使用它的模型。一个特性只有在它与目标的关系是模型可以学习的关系时，才有用。仅仅因为一个特性具有高MI分数并不意味着你的模型能够利用这些信息做任何事情。您可能需要首先转换特征以暴露关联。

## Example - 1985 Automobiles

汽车数据集（[*Automobile*](https://www.kaggle.com/toramky/automobile-dataset)）由1985年款的193辆汽车组成。该数据集的目标是根据汽车的23个特征（如品牌`make`、车身风格`body_style`和马力`horsepower`）预测汽车的价格【`price`】（目标）。在本例中，我们将使用互信息对特征进行排序，并通过数据可视化研究结果。 

此隐藏单元格导入一些库并加载数据集。

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-whitegrid")

df = pd.read_csv("../input/fe-course-data/autos.csv")
df.head()
```

![image-20221122183920109](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122183920109.png)

MI的scikit学习算法处理离散特征与连续特征不同。因此，你需要告诉它哪些是哪些。根据经验，任何必须具有浮点数据类型的东西都不是离散的。通过给类别（对象或类别数据类型）一个标签编码，可以将其视为离散的。（您可以在我们的分类变量课程([Categorical Variables](http://www.kaggle.com/alexisbcook/categorical-variables) )中查看标签编码。）

```python
X = df.copy()
y = X.pop("price")

# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()
# pd.Series.factorize()将对象编码为枚举类型或分类变量。

# All discrete features should now have integer dtypes (double-check this before using MI!)
#所有离散特性现在都应该有整数数据类型（在使用MI之前，请仔细检查！）
discrete_features = X.dtypes == int
```

`Scikit-learn`在其特征选择模块中有两个互信息度量：

- 一个用于实值目标（mutual_info_regression），
- 一个用于分类目标（mutual_info_classification）。

我们的目标价格是实值的。下一个单元计算我们的功能的MI分数，并将其封装在一个漂亮的数据帧`dataframe`中。

```python
from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]  # show a few features with their MI scores
```

```
curb_weight          1.486440
highway_mpg          0.950989
length               0.607955
bore                 0.489772
stroke               0.380041
drive_wheels         0.332973
compression_ratio    0.134799
fuel_type            0.048139
Name: MI Scores, dtype: float64
```

现在是一个条形图，以便于比较：

```python
def plot_mi_scores(scores):
    """画条形图
    """
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
```

![image-20221122184755579](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122184755579.png)

数据可视化是效用排名的一个重要后续。让我们仔细看看其中的几个。 

正如我们可能预期的那样，高得分的遏制重量（`curb_weight`）特性与目标价格有着密切的关系。

```python
sns.relplot(x="curb_weight", y="price", data=df);
```

![image-20221122184930116](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122184930116.png)

`fuel_type`特性具有相当低的MI分数，但正如我们从图中看到的，它明显地将马力`horsepower`特性中具有不同趋势的两个价格群体区分开来。这表明`fuel_type`有助于交互作用，而且可能并不重要。在根据MI评分决定某个特性不重要之前，最好调查任何可能的交互影响——领域知识可以在这里提供很多指导。

```python
sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=df);
```

![image-20221122185101046](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122185101046.png)

数据可视化是功能工程工具箱的一大补充。除了相互信息等实用性度量外，这些可视化可以帮助您发现数据中的重要关系。查看我们的数据可视化课程( [Data Visualization](https://www.kaggle.com/learn/data-visualization))以了解更多信息！

## 轮到你了 

对Ames Housing数据集的特征进行排序([**Rank the features**](https://www.kaggle.com/kernels/fork/14393925))，并选择要开始开发的第一组特征。

## Exercise: Mutual Information

### 介绍 

在本练习中，您将使用互信息得分和交互图确定Ames数据集中要开发的一组初始特征。 运行此单元以设置所有内容！

```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.feature_engineering_new.ex2 import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

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


# Load data
df = pd.read_csv("../input/fe-course-data/ames.csv")


# Utility functions from Tutorial
#测量互信息
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
        # pd.Series.factorize()将对象编码为枚举类型或分类变量。
    # All discrete features should now have integer dtypes
    #所有离散特性现在都应该有整数数据类型（在使用MI之前，请仔细检查！）
    #mutual_info_regression用于实值目标的测量
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

#画互信息
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
```

首先，让我们通过查看Ames数据集中的一些特征来回顾相互信息的含义。

```python
features = ["YearBuilt", "MoSold", "ScreenPorch"]
sns.relplot(
    x="value", y="SalePrice", 
    col="variable", 
    data=df.melt(id_vars="SalePrice", value_vars=features), 
    facet_kws=dict(sharex=False),
);
#sns.relplot需要去理解参数含义
```

![image-20221130164917876](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130164917876.png)

### 1） 了解相互信息

根据这些情节，你认为哪个功能与`SalePrice`的相互信息最高？

**`YearBuilt`**相互信息最高。

根据这些图，`YearBuilt`应该具有最高的MI分数，因为知道年份往往会将`SalePrice`限制在较小的可能值范围内。然而，`MoSold`通常并非如此。最后，由于`ScreenPorch`通常只有一个值，平均值为0，所以它不会告诉你多少`SalePrice`（尽管比`MoSold`更多）。

***

Ames数据集有78个功能——一次可以处理很多功能！幸运的是，您可以确定最有潜力的功能。 使用make_mi_scores函数（在教程中介绍）计算Ames功能的互信息得分：

```python
X = df.copy()
y = X.pop('SalePrice')

mi_scores = make_mi_scores(X, y)
```

现在使用此单元格中的函数检查分数。尤其是在高层和底层。

```python
print(mi_scores.head(20))
# print(mi_scores.tail(20))  # uncomment to see bottom 20

plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores.head(20))
# plot_mi_scores(mi_scores.tail(20))  # uncomment to see bottom 20
```

![image-20221130172126788](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130172126788.png)

![image-20221130172140747](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130172140747.png)

### 2） 检查MI分数 

分数看起来合理吗？高分特征是否代表了你认为大多数人在家中会看重的东西？你注意到他们描述的主题了吗？

这些功能中的一些常见主题是： 

- 地点：`Neighborhood`

- 大小：所有`Area`和`SF`功能，包括`FullBath`和`GarageCars` 

- 质量：`Qual`的所有功能 

- 年份：建造年份和拆除年份 (`YearBuilt` and `YearRemodAdd`)

- 类型：功能和样式的描述，如`Foundation`和`GarageType`

这些都是你在房地产列表中常见的特征（比如在Zillow上），我们的共同信息指标对它们的评分很高。另一方面，排名最低的功能在某种程度上似乎大多代表了罕见或特殊的东西，因此与普通购房者无关。

***

在本步骤中，您将研究`BldgType`功能的可能交互效果。该特征描述了五类住宅的总体结构： 

>  建筑类型（标称）：住宅类型 
>
> (Bldg Type (Nominal): Type of dwelling)
>
> - `1Fam`——Single-family Detached(单人家庭分离) 
> - `2FmCon`——Two-family Conversion; originally built as one-family dwelling(两族转换；最初作为一个家庭住宅建造 )
> - `Duplx`——Duplex(双工 )
> - `TwnhsE`——Townhouse End Unit(Townhouse终端单元 )
> - `TwnhsI`——Townhouse Inside Unit(Townhouse内部单元 )

`BldgType`功能没有获得很高的MI分数。一个图表证实了`BldgType`中的类别在区分`SalePrice`中的值方面做得不好（换句话说，分布看起来相当相似）：

```python
sns.catplot(x="BldgType", y="SalePrice", data=df, kind="boxen");
```

![image-20221130172951010](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130172951010.png)

尽管如此，住宅的类型似乎应该是重要的信息。调查`BldgType`是否与以下任何一项产生重大影响：

- `GrLivArea`  # Above ground living area(地上生活区)
- `MoSold`    # Month sold（月销售量）

运行以下单元格两次，第一次使用feature=“GrLivArea”，第二次使用feature=“MoSold”：

```python
feature = "GrLivArea"

sns.lmplot(
    x=feature, y="SalePrice", hue="BldgType", col="BldgType",
    data=df, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,
);
```

![image-20221130173230714](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130173230714.png)

```python
feature = "MoSold"

sns.lmplot(
    x=feature, y="SalePrice", hue="BldgType", col="BldgType",
    data=df, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,
);
```

![image-20221130173405544](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130173405544.png)

从一个类别到下一个类别的趋势线显著不同，表明存在交互作用。

### 3） 发现交互 

从图中看，`BldgType`是否显示出与`GrLivArea`或`MoSold`的交互作用？

**`BldgType`的每个类别中的趋势线明显非常不同**，表明这些特征之间的相互作用。由于了解了`BldgType`，我们可以更多地了解**`GrLivArea`**与`SalePrice`的关系，因此我们应该考虑将`BldgType`包含在我们的功能集中。 

然而，`MoSold`的趋势线几乎相同。对于了解`BldgType`，此功能并没有提供更多信息。

### 第一组发展特点 

让我们花点时间列出我们可能关注的功能。在第3课的练习中，您将开始通过您认为具有高潜力的原始功能的组合来构建一个信息量更大的功能集。 

您发现MI得分最高的十项功能是：

```python
mi_scores.head(10)
```

![image-20221130173850753](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130173850753.png)

你认识这里的主题吗？位置、大小和质量。您不必将开发局限于这些顶级功能，但现在您有了一个好的开始。将这些**顶级特性与其他相关特性相结合**，特别是那些您认为可以**创建交互的特性**，是一个很好的策略，可以为您的模型提供一组信息量很大的特性。

### 继续前进 

开始创建特性，并了解不同模型最可能从中受益的转换类型。