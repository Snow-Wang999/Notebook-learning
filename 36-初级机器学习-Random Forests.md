# 35-Random Forests

使用更复杂的机器学习算法。

## 介绍

决策树让您做出艰难的决定。一棵有很多叶子的深树会过度拟合，因为每个预测都来自其叶子上少数房屋的历史数据。但是一棵叶子很少的浅树会表现不佳，因为它无法捕获原始数据中的尽可能多的区别。 

即使是当今最复杂的建模技术也面临着欠拟合和过拟合之间的这种紧张关系。但是，许多模型都有巧妙的想法，可以带来更好的性能。我们将以**随机森林**为例。 

随机森林使用很多树，它通过**平均**每个组件树的**预测**来进行预测。它通常比单个决策树具有更好的预测准确性，并且它适用于默认参数。如果你继续建模，你可以学习更多性能更好的模型，但其中许多模型对获得正确的参数很敏感。

## Example

您已经看过几次加载数据的代码。在数据加载结束时，我们有以下变量：

- train_X
- val_X
- train_y
- val_y

```python
import pandas as pd
    
# Load data
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
# Filter rows with missing values
melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
```

我们构建了一个随机森林模型，类似于我们在 `scikit-learn`中构建决策树的方式——这次使用的是 `RandomForestRegressor` 类而不是 `DecisionTreeRegressor`。

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
```

```
191669.7536453626
```

## 结论 

可能还有进一步改进的空间，但这比最佳决策树错误 250,000 有了很大改进。有一些参数可以让你改变随机森林的性能，就像我们改变单个决策树的最大深度一样。但随机森林模型的最佳特性之一是，即使没有这种调整，它们通常也能正常工作。

## Exercise: Random Forests

### 回顾

```python
# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))


# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex6 import *
print("\nSetup complete")
```

### Exercises

数据科学并不总是那么容易。但是用随机森林替换决策树将是一个轻松的胜利。

#### 第 1 步：使用随机森林

![image-20221115223417630](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221115223417630.png)

```python
from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# fit your model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
```

```
Validation MAE for Random Forest Model: 21857.15912981083
```

到目前为止，您在项目的每个步骤中都遵循了特定的说明。这有助于学习关键思想并构建您的第一个模型，但现在您知道的足够多，可以自己尝试了。 

机器学习竞赛是您尝试自己的想法并在您独立完成机器学习项目时了解更多信息的好方法。