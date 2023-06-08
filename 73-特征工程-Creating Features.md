# 73-特征工程-Creating Features

使用Pandas变换功能以适合您的模型。

介绍 

一旦您确定了一组具有一定潜力的功能，就应该开始开发它们了。在本课中，您将学习完全可以在Pandas中完成的许多常见转换。如果你感到生疏，我们有一个很棒的熊猫课程。 

在本课中，我们将使用四个具有一系列特征类型的数据集：美国交通事故、1985年汽车、具体配方和客户终身价值（[*US Traffic Accidents*](https://www.kaggle.com/sobhanmoosavi/us-accidents), [*1985 Automobiles*](https://www.kaggle.com/toramky/automobile-dataset), [*Concrete Formulations*](https://www.kaggle.com/sinamhd9/concrete-comprehensive-strength), and [*Customer Lifetime Value*](https://www.kaggle.com/pankajjsh06/ibm-watson-marketing-customer-value-data).）。下面的隐藏单元格将加载它们。

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

accidents = pd.read_csv("../input/fe-course-data/accidents.csv")
autos = pd.read_csv("../input/fe-course-data/autos.csv")
concrete = pd.read_csv("../input/fe-course-data/concrete.csv")
customer = pd.read_csv("../input/fe-course-data/customer.csv")
```

> **关于发现新特征的提示** 
>
> - 了解特征。如果可用，请参阅数据集的数据文档。 
>
> - 研究问题领域以获取**领域知识**。如果你的问题是预测房价，那就做一些房地产研究吧。维基百科可能是一个很好的起点，但书籍和期刊文章通常会提供最好的信息。 
>
> - 研究以前的工作。过去Kaggle比赛的解决方案总结是一个很好的资源。 
>
> - 使用数据可视化。可视化可以揭示特征分布中的病理或可以简化的复杂关系。在完成特征工程过程时，确保将数据集可视化。

## 数学变换 

数字特征之间的关系通常通过数学公式来表达，这是您在领域研究中经常遇到的。在Pandas中，您可以将算术运算应用于列，就像它们是普通数字一样。 

在汽车数据集中有描述汽车发动机的特征。研究得出了创建潜在有用新功能的各种公式。例如，“冲程比（stroke ratio）”是衡量发动机效率与性能的指标：(bore缸径，stroke行程)

```python
autos["stroke_ratio"] = autos.stroke / autos.bore

autos[["stroke", "bore", "stroke_ratio"]].head()
```

![image-20221130175246205](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130175246205.png)

组合越复杂，模型学习起来就越困难，比如发动机“排量(displacement)”的公式，即发动机功率的度量：

```python
autos["displacement"] = (
    np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders
)
```

$$
displacement = \pi\times(\frac{1}{2}bore)^2\times stroke \times numOfCylinders
$$

数据可视化可以建议转换，通常是通过幂或对数对特征进行“重塑”。例如，风速在美国事故中的分布非常不平衡。在这种情况下，对数有效地将其归一化：

```python
# If the feature has 0.0 values, use np.log1p (log(1+x)) instead of np.log
accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)

# Plot a comparison
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(accidents.WindSpeed, shade=True, ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed, shade=True, ax=axs[1]);
```

![image-20221130181558234](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130181558234.png)

查看我们在数据清理( [*Data Cleaning*](https://www.kaggle.com/learn/data-cleaning) )中的标准化课程( [lesson on normalization](https://www.kaggle.com/alexisbcook/scaling-and-normalization))，您还将了解Box-Cox变换，这是一种非常通用的标准化器。

## 计数

描述某事物存在或不存在的特征通常是成套的，比如说，一组疾病的危险因素。您可以通过创建计数来聚合这些功能。 这些特性将是二进制的（1表示存在，0表示不存在）或布尔值（True或False）。在Python中，布尔值可以像整数一样相加。 

在交通事故中，有几个特征表明某些道路物体是否在事故附近。这将使用求和方法创建附近道路特征总数的计数：

```python
roadway_features = ["Amenity", "Bump", "Crossing", 
                    "GiveWay","Junction", "NoExit", 
                    "Railway", "Roundabout", "Station", 
                    "Stop", "TrafficCalming", "TrafficSignal"]
accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)

accidents[roadway_features + ["RoadwayFeatures"]].head(10)
```

![image-20221130182056159](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130182056159.png)

您还可以使用数据帧的内置方法来创建布尔值。在混凝土数据集中是混凝土配方中组分的数量。许多配方缺少一种或多种组分（即组分的值为0）。这将计算数据帧内置大于`gt`方法的公式中有多少个组分：

```python
components = [ "Cement", 
               "BlastFurnaceSlag",
               "FlyAsh", 
               "Water",
               "Superplasticizer", 
               "CoarseAggregate",
               "FineAggregate"]
concrete["Components"] = concrete[components].gt(0).sum(axis=1)
#统计大于0（由gt方法获得）的值有几个

concrete[components + ["Components"]].head(10)
```

![image-20221130182712957](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130182712957.png)

## 构建和分解功能

通常，你会有复杂的字符串，这些字符串可以有效地分解成更简单的片段。一些常见示例：

- ID numbers: `'123-45-6789'`
- Phone numbers: `'(999) 555-0123'`
- Street addresses: `'8241 Kaggle Ln., Goose City, NV'`
- Internet addresses: `'http://www.kaggle.com`
- Product codes: `'0 36000 29145 2'`
- Dates and times: `'Mon Sep 30 07:06:05 2013'`

像这样的功能通常具有某种结构，您可以利用这些结构。例如，美国电话号码有一个区号（“（999）”部分），可以告诉你来电者的位置。一如既往，一些研究可以在这里得到回报。 

`str`访问器允许您将拆分等字符串方法直接应用于列。

Customer Lifetime Value数据集包含描述保险公司客户的功能。从“政策”功能中，我们可以将“类型”与覆盖“级别”分开：

```python
customer[["Type", "Level"]] = (  # Create two new features
    customer["Policy"]           # from the Policy feature
    .str                         # through the string accessor
    .split(" ", expand=True)     # by splitting on " "
                                 # and expanding the result into separate columns
)

customer[["Policy", "Type", "Level"]].head(10)
```

![image-20221130183244398](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130183244398.png)

如果您有理由相信组合中存在一些交互，您也可以将简单功能合并为组合功能：

```python
autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]
autos[["make", "body_style", "make_and_style"]].head()
```

![image-20221130183425915](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130183425915.png)

> **Kaggle Learn上的其他地方 **
>
> 还有一些其他类型的数据我们在这里没有讨论过，它们的信息特别丰富。幸运的是，我们为您提供了保障！ 
>
> 有关**日期和时间**，请参阅数据清理课程中的分析日期([Parsing Dates](https://www.kaggle.com/alexisbcook/parsing-dates))。 
>
> 关于**纬度和经度**，请参阅我们的地理空间分析课程( [Geospatial Analysis](https://www.kaggle.com/learn/geospatial-analysis))。

## 组变换（Group Transforms）

最后，我们有Group转换，它聚合按某个类别分组的多个行中的信息。通过Group转换，你可以创建诸如“一个人居住状态的平均收入”或“按类型划分的工作日上映电影的比例”等特征。 

使用聚合函数(aggregation function)，组转换结合了两个特性：

- 一个提供分组的分类特性，
- 另一个要聚合其值的特性。

对于“按州平均收入”，您可以选择`State`作为分组功能，`mean`作为聚合功能，`income`作为聚合特征。为了在Pandas中计算这一点，我们使用groupby和transform方法：

```python
customer["AverageIncome"] = (
    customer.groupby("State")  # for each state
    ["Income"]                 # select the income
    .transform("mean")         # and compute its mean
)

customer[["State", "Income", "AverageIncome"]].head(10)
```

![image-20221130191153525](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130191153525.png)

`mean`函数是一个内置的数据帧方法，这意味着我们可以将其作为字符串传递以进行转换。其他方便的方法包括`max`、`min`、`median`、`var`、`std`和`count`。以下是如何计算数据集中每个状态发生的频率：

```python
customer["StateFreq"] = (
    customer.groupby("State")
    ["State"]
    .transform("count")
    / customer.State.count()
)

customer[["State", "StateFreq"]].head(10)
```

![image-20221130191355710](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130191355710.png)

您可以使用这样的变换为分类特征创建“频率编码”。 

如果要使用训练和验证分割(training and validation splits)，为了保持它们的独立性，最好只使用**训练集创建一个分组特征**，然后将其加入到验证集。在训练集上使用`drop_duplicates`创建一组唯一的值后，我们可以使用验证集的`merge`方法：

```python
# Create splits
df_train = customer.sample(frac=0.5)
df_valid = customer.drop(df_train.index)

# Create the average claim amount by coverage type, on the training set
#在培训集上按覆盖类型创建平均索赔金额
df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")

# Merge the values into the validation set
df_valid = df_valid.merge(
    df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
    on="Coverage",#用于连接的列名
    how="left",#连接方式
)

df_valid[["Coverage", "AverageClaim"]].head(10)
```

![image-20221130191929206](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221130191929206.png)

### [pandas的merge方法详解](https://blog.csdn.net/trayvontang/article/details/103787648)

`left.merge(right,on="",how=left)`

![连接方式](https://img-blog.csdnimg.cn/20191231185001613.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3RyYXl2b250YW5n,size_16,color_FFFFFF,t_70)

> **关于创建要素的提示 **
>
> 在创建功能时，最好记住模型自身的优点和缺点。以下是一些指导原则：
>
> - 线性模型自然地学习和与差，但不能学习更复杂的东西。 
> - 对于大多数模型来说，**比率（Ratio）**似乎很难学习。**比率组合**通常会带来一些简单的性能提升。 
> - 线性模型和神经网络通常在**归一化**特征方面做得更好。
>   - 神经网络尤其需要缩放到离0不太远的值的特征。
>   - 基于树的模型（如随机森林和XGBoost）有时可以从**标准化(归一化)**中受益，但通常要少得多。 
> - 树模型可以学习近似几乎任何特征的组合，但当组合特别重要时，它们仍然可以从显式创建中受益，尤其是当数据有限时。 
> - **计数**对于**树模型**特别有用，因为这些模型没有一种自然的方式来同时聚合多个特性的信息。

### 轮到你了 

结合并转换Ames的功能，提高模型的性能。

## Exercise: Creating Features

### 介绍 

在本练习中，您将开始开发练习2中确定的最有潜力的功能。在完成本练习时，您可能需要花一些时间再次查看数据文档，并考虑我们正在创建的功能是否从真实世界的角度来看有意义，以及是否有任何有用的组合值得您注意。 

运行此单元以设置所有内容！

```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.feature_engineering_new.ex3 import *

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


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
X = df.copy()
y = X.pop("SalePrice")
```

让我们从几个数学组合开始。我们将重点关注描述区域的特性——具有相同的单位（平方英尺）可以方便地以合理的方式组合它们。因为我们使用的是XGBoost（基于树的模型），所以我们将重点关注比率和总和。

### 1） 创建数学变换

创建以下特征：

- `LivLotRatio`: the ratio of `GrLivArea` to `LotArea`
- `Spaciousness`: the sum of `FirstFlrSF` and `SecondFlrSF` divided by `TotRmsAbvGrd`
- `TotalOutsideSF`: the sum of `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `Threeseasonporch`, and `ScreenPorch`

```python
X_1 = pd.DataFrame()  # dataframe to hold new features

X_1["LivLotRatio"] = X.GrLivArea / X.LotArea
X_1["Spaciousness"] = (X.FirstFlrSF+X.SecondFlrSF)/X.TotRmsAbvGrd 
X_1["TotalOutsideSF"] = X.WoodDeckSF+X.OpenPorchSF+X.EnclosedPorch+X.Threeseasonporch+X.ScreenPorch

```

如果您发现了数字特征和分类特征之间的交互作用，您可能希望使用单热编码对其进行显式建模，如下所示：

```python
#一个热编码分类功能，添加列前缀“Cat”
# One-hot encode Categorical feature, adding a column prefix "Cat"
X_new = pd.get_dummies(df.Categorical, prefix="Cat")

# Multiply row-by-row
X_new = X_new.mul(df.Continuous, axis=0)

#将新要素连接到要素集
# Join the new features to the feature set
X = X.join(X_new)
```

### 2） 与类别的交互 

我们在练习2中发现了BldgType和GrLivArea之间的交互。现在创建它们的交互特性。

```python
# One-hot encode BldgType. Use `prefix="Bldg"` in `get_dummies`
X_2 = pd.get_dummies(X.BldgType,prefix='Bldg') 
# Multiply
X_2 = X_2.mul(X.GrLivArea,axis=0)
```



### 3） 计数功能 

让我们尝试创建一个功能，描述一个住宅有多少种户外区域。创建一个功能PorchTypes，计算以下值中有多少大于0.0：

```
WoodDeckSF
OpenPorchSF
EnclosedPorch
Threeseasonporch
ScreenPorch
```

```python
X_3 = pd.DataFrame()
outdoor_area_type=[ 'WoodDeckSF',
                    'OpenPorchSF',
                    'EnclosedPorch',
                    'Threeseasonporch',
                    'ScreenPorch']
# YOUR CODE HERE
X_3["PorchTypes"] = X[outdoor_area_type].gt(0).sum(axis=1)
```



### 4） 分解分类特征

`MSSubClass`描述住宅类型：

```python
df.MSSubClass.unique()
```

```
array(['One_Story_1946_and_Newer_All_Styles', 'Two_Story_1946_and_Newer',
       'One_Story_PUD_1946_and_Newer',
       'One_and_Half_Story_Finished_All_Ages', 'Split_Foyer',
       'Two_Story_PUD_1946_and_Newer', 'Split_or_Multilevel',
       'One_Story_1945_and_Older', 'Duplex_All_Styles_and_Ages',
       'Two_Family_conversion_All_Styles_and_Ages',
       'One_and_Half_Story_Unfinished_All_Ages',
       'Two_Story_1945_and_Older', 'Two_and_Half_Story_All_Ages',
       'One_Story_with_Finished_Attic_All_Ages',
       'PUD_Multilevel_Split_Level_Foyer',
       'One_and_Half_Story_PUD_All_Ages'], dtype=object)
```

您可以看到，每个类别的第一个单词（大致）描述了一个更一般的分类。通过在第一个下划线`_`处拆分`MSSubClass`，创建仅包含这些第一个单词的功能。（提示：在split方法中，使用参数n=1。）

```python
X_4 = pd.DataFrame()

# YOUR CODE HERE
X_4['MSClass']=(
    X['MSSubClass']
    .str
    .split('_',n=1,expand=True)[0]
)
```



### 5） 使用分组变换 

房子的价值通常取决于它与附近典型住宅的比较。创建一个功能`MedNhbdArea`，该功能描述在Neighborhood上分组的`GrLivArea`的中值。

```python
X_5 = pd.DataFrame()

# YOUR CODE HERE
X_5["MedNhbdArea"] = X.groupby('Neighborhood').GrLivArea.transform('median')
```

现在，您已经制作了第一个新功能集！如果您愿意，可以运行下面的单元格，为添加了所有新功能的模型打分：

```python
X_new = X.join([X_1, X_2, X_3, X_4, X_5])
score_dataset(X_new, y)
```

```
0.13847331710099203
```

### 继续前进 

通过向数据集添加群集标签来解除空间关系。