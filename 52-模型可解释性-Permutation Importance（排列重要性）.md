# 52-Permutation Importance（排列重要性）

您的模型认为哪些特征很重要？

## 介绍

我们可能会问模型的最基本问题之一是：哪些特征对预测的影响最大？ 

这个概念称为**特征重要性（feature importance）**。 

有多种方法可以衡量特征重要性。有些方法对上述问题的回答略有不同。其他方法也有缺点。 

在本课中，我们将重点关注**排列重要性（permutation importance）**。与大多数其他方法相比，排列重要性是： 

- 快速计算， 
- 广泛使用和理解，以及 
- 与我们希望特征重要性度量具有的属性一致。

## 这个怎么运作？

排列重要性使用的模型与您目前所见的任何模型都不同，许多人一开始觉得它很混乱。因此，我们将从一个示例开始，以使其更加具体。 

考虑具有以下格式的数据：

![image-20221119133224208](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221119133224208.png)

我们想使用 10 岁时可用的数据来预测一个人 20 岁时的身高。 

我们的数据包括有用的特征（10 岁时的身高）、几乎没有预测能力的特征（拥有的袜子），以及我们在本说明中不会重点关注的其他一些特征。

**排列重要性是在模型拟合后计算的。**所以我们不会改变模型或改变我们对给定的身高值、袜子数量等的预测。 

相反，我们会问以下问题：如果我随机打乱（shuffle）验证数据的单个列，将目标和所有其他列留在原地，这将如何影响现在打乱的数据中预测的准确性？

![image-20221119133419997](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221119133419997.png)

随机重新排序单个列应该会导致不太准确的预测，因为生成的数据不再对应于现实世界中观察到的任何数据。如果我们对模型严重依赖于预测的列进行洗牌，则模型准确性尤其会受到影响。在这种情况下，在 10 岁时改组身高会导致可怕的预测。如果我们改为洗牌袜子，那么由此产生的预测就不会受到那么大的影响。

有了这个洞察，过程如下： 

1. 获得训练有素的模型。 

2. 打乱单列中的值（Shuffle the values in a single column），使用生成的数据集进行预测。使用这些预测和真实的目标值来计算损失函数遭受洗牌的程度。这种性能下降（performance deterioration）衡量了你刚刚洗牌的变量的重要性。 

3. 将数据返回到原始顺序（撤消步骤 2 中的随机打乱）。现在对数据集中的下一列重复步骤 2，直到计算出每一列的重要性。

## Code Example

我们的示例将使用一个模型来预测一支足球/橄榄球队是否会根据球队的统计数据获得“最佳球员”。 “比赛最佳人”奖授予比赛中表现最好的球员。模型构建不是我们当前的重点，因此下面的单元格加载数据并构建基本（简陋的-rudimentary）模型。

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=100,
                                  random_state=0).fit(train_X, train_y)
```

以下是如何使用 eli5 库计算和显示重要性：

```python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```

![image-20221119134154590](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221119134154590.png)

## 解释排列重要性

**顶部的值**是最重要的特征，而底部的值最不重要。 

每行中的**第一个数字**显示随机改组后模型**性能下降了多少**（在这种情况下，使用**“准确性”**作为性能指标）。

与数据科学中的大多数事情一样，对列进行改组的确切性能变化存在一些随机性。我们通过多次洗牌重复该过程来测量排列重要性计算中的随机性。 **± 后面的数字**衡量**性能从一次改组到下一次改组的变化情况**。 

您偶尔会看到排列重要性的**负值**。在这些情况下，对打乱（或嘈杂）数据的预测恰好比真实数据更准确。当**特征无关紧要**（重要性应该接近 0）时会发生这种情况，但随机机会导致对混洗数据的预测更加准确。这在小型数据集（如本例中的数据集）中更为常见，因为运气/机会的空间更大。

在我们的示例中，最重要的特征是**进球数(Goals scored)**。这似乎很明智。足球迷可能对其他变量的排序是否令人惊讶有一些直觉。

## Exercise: Permutation Importance

### 介绍

您将使用出租车票价预测竞赛中的数据样本来思考和计算排列重要性。 

我们暂时不会专注于数据探索或模型构建。您只需运行下面的单元格即可

- 加载数据 
- 将数据分为训练和验证 
- 建立一个预测出租车费用的模型 
- 打印几行供您查看

```python
# Loading data, dividing, modeling and EDA below
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows=50000)

# Remove data with extreme outlier coordinates or negative fares
# 删除具有极端异常坐标或负票价的数据
data = data.query('pickup_latitude > 40.7 and pickup_latitude < 40.8 and ' +
                  'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and ' +
                  'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +
                  'dropoff_longitude > -74 and dropoff_longitude < -73.9 and ' +
                  'fare_amount > 0'
                  )

y = data.fare_amount

base_features = ['pickup_longitude',
                 'pickup_latitude',
                 'dropoff_longitude',
                 'dropoff_latitude',
                 'passenger_count']

X = data[base_features]


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
first_model = RandomForestRegressor(n_estimators=50, random_state=1).fit(train_X, train_y)

# Environment Set-Up for feedback system.
from learntools.core import binder
binder.bind(globals())
from learntools.ml_explainability.ex2 import *
print("Setup Complete")

# show data
print("Data sample:")
data.head()
```

![image-20221119135646926](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221119135646926.png)

以下两个单元格也可能有助于理解训练数据中的值：

```python
train_X.describe()
```

![image-20221119135712125](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221119135712125.png)

```python
train_y.describe()
```

![image-20221119135730372](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221119135730372.png)

### Question 1

第一个模型使用以下特征 

- pickup_longitude
- pickup_latitude
- dropoff_longitude
- dropoff_latitude
- passenger_count

在运行任何代码之前......哪些变量似乎对预测出租车票价可能有用？您是否认为排列重要性必然会将这些特征识别为重要？ 

考虑之后，运行下面的 q_1.solution() 以查看在运行代码之前您可能如何考虑它。

答：

**了解纽约市出租车是否根据乘客数量改变价格会很有帮助**。大多数地方不会根据乘客人数改变票价。如果您假设纽约市是一样的，那么只有列出的前 4 个特征才是重要的。乍一看，所有这些似乎都同等重要。

### Question 2

创建一个名为 `perm` 的 `PermutationImportance` 对象以显示来自 `first_model` 的重要性。用适当的数据拟合它并显示权重。 为方便起见，教程中的代码已复制到此代码单元格的注释中。

```python
import eli5
from eli5.sklearn import PermutationImportance

# Make a small change to the code below to use in this problem. 
perm = PermutationImportance(first_model, random_state=1).fit(val_X, val_y)

# uncomment the following line to visualize your results
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```

![image-20221119140204768](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221119140204768.png)

### Question 3

在看到这些结果之前，我们可能认为 4 个方向特征中的每一个都同等重要。 

但是，平均而言，纬度特征比经度特征更重要。你能对此提出任何假设吗？ 

在您考虑之后，请查看此处以获取一些可能的解释：

答：

1. 旅行的**纬度距离可能比经度距离大**。如果经度值通常靠得更近，那么将它们改组就没有那么重要了。 
2. **城市的不同地区可能有不同的定价规则**（例如每英里的价格），并且定价规则因纬度而不是经度而异。 
3. **向北<->向南（纬度变化）的道路收费可能高于向东<->向西（经度变化）的道路。**因此，纬度会对预测产生更大的影响，因为它反映了通行费的数量。

### Question 4

如果不了解纽约市的详细信息，就很难排除关于为什么纬度特征比经度更重要的大多数假设。 

一个很好的下一步是将位于城市某些地区的影响与总行驶距离的影响区分开来。

下面的代码为经度和纬度距离创建了新特征。然后它会构建一个模型，将这些新功能添加到您已有的功能中。

填写两行代码来计算并显示这组新特征的重要性权重。像往常一样，您可以取消注释下面的行以检查您的代码、查看提示或获得解决方案。

```python
# create new features
data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)
data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)

features_2  = ['pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'abs_lat_change',
               'abs_lon_change']

X = data[features_2]
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)

# Create a PermutationImportance object on second_model and fit it to new_val_X and new_val_y
# Use a random_state of 1 for reproducible results that match the expected solution.
perm2 = PermutationImportance(second_model, random_state=1).fit(new_val_X, new_val_y)

# show the weights for the permutation importance you just calculated
eli5.show_weights(perm2, feature_names = new_val_X.columns.tolist())
```

![image-20221119140911140](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221119140911140.png)

您如何解释这些重要性分数？

答：

**行进的距离似乎比任何位置效应都重要得多。** 

但**位置仍然会影响模型预测，下车位置现在比上车位置更重要**。

你对为什么会这样有任何假设吗？下一课中的技巧将帮助您更深入地了解这一点。

### Question 5

一位同事观察到 `abs_lon_change` 和 `abs_lat_change` 的值非常小（所有值都在 -0.1 和 0.1 之间），而其他变量的值更大。您认为这可以解释为什么这些坐标在这种情况下具有更大的排列重要性值吗？ 

考虑一个替代方案，您创建并使用了一个比这些特征大 100 倍的特征，并将该更大的特征用于训练和重要性计算。这会改变输出的排列重要性值吗？ 

为什么或者为什么不？ 

在你想好你的答案后，要么尝试这个实验，要么在下面的单元格中查找答案。

```python
# create new features
data['abs_lon_change'] = 100*abs(data.dropoff_longitude - data.pickup_longitude)
data['abs_lat_change'] = 100*abs(data.dropoff_latitude - data.pickup_latitude)

features_2  = ['pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'abs_lat_change',
               'abs_lon_change']

X = data[features_2]
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)

# Create a PermutationImportance object on second_model and fit it to new_val_X and new_val_y
# Use a random_state of 1 for reproducible results that match the expected solution.
perm2 = PermutationImportance(second_model, random_state=1).fit(new_val_X, new_val_y)

# show the weights for the permutation importance you just calculated
eli5.show_weights(perm2, feature_names = new_val_X.columns.tolist())
```

![image-20221119141514991](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221119141514991.png)

答：

**特征的规模本身并不影响排列重要性。**重新缩放特征会影响 PI(排列重要性) 的唯一原因是间接的，如果重新缩放有助于或损害我们用来利用该特征的特定学习方法的能力。基于树的模型不会发生这种情况，例如此处使用的随机森林。如果您熟悉岭回归（Ridge Regression），您可能会想到它会受到怎样的影响。也就是说，绝对变化特征非常重要，因为它们捕获总行驶距离，这是出租车费用的主要决定因素......它不是特征量级的产物。

### Question 6

您已经看到纬度距离的特征重要性大于经度距离的重要性。由此，我们能否得出结论，在固定的纬度距离上行驶是否往往比在相同的经度距离上行驶更昂贵？ 为什么或者为什么不？检查下面的答案。

答：

我们无法从排列重要性结果中判断在固定的纬度距离上行驶比在相同的经度距离上行驶更昂贵还是更便宜。纬度特征比经度特征更重要的可能原因： 

1. 数据集中的纬度距离往往更大 
2. 行进固定纬度距离的成本更高 
3. 以上两者都是，如果 `abs_lon_change` 值非常小，经度可能会对模型更不重要，即使沿该方向每英里行驶的成本很高。

### 继续 

排列重要性对于调试、理解模型以及传达模型的高级概览很有用。 接下来，了解部分依赖图（**[partial dependence plots](https://www.kaggle.com/dansbecker/partial-plots)**）以查看每个特征如何影响预测。