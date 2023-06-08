# 46-XGBoost

最准确的结构化数据建模技术。

在本教程中，您将学习如何使用梯度提升构建和优化模型。这种方法在许多 Kaggle 比赛中占据主导地位，并在各种数据集上取得了最先进的结果。

## 介绍

对于本课程的大部分内容，您已经使用随机森林方法进行了预测，该方法仅通过对许多决策树的预测进行平均来获得比单个决策树更好的性能。 

我们将随机森林方法称为“集成方法”。根据定义，集成方法结合了多个模型的预测（例如，在随机森林的情况下，有几棵树）。 接下来，我们将学习另一种称为梯度提升的集成方法。

## 梯度提升（Gradient Boosting）

梯度提升是一种通过循环将模型迭代添加到集成中的方法。 

它首先使用单个模型初始化集成，其预测可能非常幼稚。 （即使它的预测非常不准确，随后添加到集合中的内容也会解决这些错误。） 

然后，我们开始循环： 

- 首先，我们使用当前集合为数据集中的每个观察结果生成预测。为了进行预测，我们将来自集成中所有模型的预测相加。 

- 这些预测用于计算损失函数（例如均方误差）。 

- 然后，我们使用损失函数来拟合将添加到集成中的新模型。具体来说，我们确定模型参数，以便将这个新模型添加到集成中可以减少损失。 （旁注：“梯度提升”中的“梯度”指的是我们将在损失函数上使用梯度下降来确定这个新模型中的参数。） 

- 最后，我们将新模型添加到集成中，然后...... 

- ... 重复！

![image-20221118175303641](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221118175303641.png)

## Example

我们首先在 X_train、X_valid、y_train 和 y_valid 中加载训练和验证数据。

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
```

在此示例中，您将使用 XGBoost 库。 XGBoost 代表 extreme gradient boosting，它是梯度提升的一种实现，具有几个专注于性能和速度的附加功能。 （Scikit-learn 有另一个版本的 gradient boosting，但 XGBoost 有一些技术优势。） 在下一个代码单元中，我们导入 XGBoost 的 scikit-learn API (xgboost.XGBRegressor)。这使我们能够像在 scikit-learn 中一样构建和拟合模型。正如您将在输出中看到的，XGBRegressor 类有许多可调参数——您很快就会了解这些！

```python
from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)
```

```python
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
             gamma=0, gpu_id=-1, importance_type=None,
             interaction_constraints='', learning_rate=0.300000012,
             max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,
             monotone_constraints='()', n_estimators=100, n_jobs=4,
             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
```

我们还进行预测并评估模型。

```python
from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
```

```
Mean Absolute Error: 239435.01260125183
```

## 参数调整(parameter tuning)

XGBoost 有一些参数可以显着影响准确性和训练速度。您应该了解的第一个参数是：

### `n_estimators`

`n_estimators` 指定经历上述**建模循环的次数**。它等于我们包含在集成中的模型数量。 

- 值太低会导致欠拟合，从而导致对训练数据和测试数据的预测不准确。 

- 太高的值会导致过拟合，这会导致对训练数据的预测准确，但对测试数据的预测不准确（这是我们关心的）。 

典型值范围为 **100-1000**，但这在很大程度上取决于下面讨论的 learning_rate 参数。 

下面是设置集合中模型数量的代码：

```python
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)
```

```
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
             gamma=0, gpu_id=-1, importance_type=None,
             interaction_constraints='', learning_rate=0.300000012,
             max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,
             monotone_constraints='()', n_estimators=500, n_jobs=4,
             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
```

### `early_stopping_rounds`

early_stopping_rounds 提供了一种自动找到 n_estimators 理想值的方法。提前停止会导致模型在验证分数停止提高时停止迭代，即使我们没有处于 n_estimators 的硬停止。为 n_estimators 设置一个高值然后使用 early_stopping_rounds 找到停止迭代的最佳时间是明智的。 

由于随机机会有时会导致验证分数没有提高的单轮，因此您需要**指定一个数字来表示在停止之前允许多少轮直接恶化**。设置 `early_stopping_rounds=5` 是一个合理的选择。在这种情况下，我们在验证分数连续 5 轮恶化后停止。 

当使用 early_stopping_rounds 时，您还需要预留一些数据用于**计算验证分数**——这是通过设置 `eval_set` 参数来完成的。 

我们可以修改上面的示例以包括提前停止：

```python
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)
```

```
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
             gamma=0, gpu_id=-1, importance_type=None,
             interaction_constraints='', learning_rate=0.300000012,
             max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,
             monotone_constraints='()', n_estimators=500, n_jobs=4,
             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
```

如果您稍后想要使用所有数据拟合模型，请将 n_estimators 设置为您在使用提前停止运行时发现的**最佳值**。

### `learning_rate`

我们不是通过简单地将每个组件模型的预测相加来获得预测，而是可以在将它们相加之前将每个模型的预测乘以一个小数字（称为学习率）。 

这意味着我们添加到合奏中的每棵树对我们的帮助都会减少。因此，我们可以为 n_estimators 设置更高的值而不会过度拟合。如果我们使用 early stopping，合适的树数将自动确定。 

一般来说，**小的学习率和大量的估计器**会产生更准确的 XGBoost 模型，尽管它也会**花费更长的时间**来训练模型，因为它在整个循环中进行了更多的迭代。默认情况下，XGBoost 设置 learning_rate=0.1。 

修改上面的示例以更改学习率会产生以下代码：

```python
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
```

```
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
             gamma=0, gpu_id=-1, importance_type=None,
             interaction_constraints='', learning_rate=0.05, max_delta_step=0,
             max_depth=6, min_child_weight=1, missing=nan,
             monotone_constraints='()', n_estimators=1000, n_jobs=4,
             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
```

### `n_jobs`

在考虑运行时的**较大数据集**上，您可以使用**并行性**来更快地构建模型。通常将参数 n_jobs 设置为等于计算机上的**内核数**。在较小的数据集上，这无济于事。 

生成的模型不会更好，因此对拟合时间进行微优化通常只会分散注意力。但是，它在大型数据集中很有用，否则您会在 fit 命令期间等待很长时间。 

这是修改后的示例：

```python
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
```

```
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
             gamma=0, gpu_id=-1, importance_type=None,
             interaction_constraints='', learning_rate=0.05, max_delta_step=0,
             max_depth=6, min_child_weight=1, missing=nan,
             monotone_constraints='()', n_estimators=1000, n_jobs=4,
             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
```

## 结论 

[XGBoost](https://xgboost.readthedocs.io/en/latest/)  是一个领先的软件库，用于处理**标准表格数据**（您存储在 Pandas DataFrames 中的数据类型，而不是图像和视频等更奇特的数据类型）。通过仔细的参数调整，您可以训练出高度准确的模型。

## Exercise: XGBoost

在本练习中，您将使用新知识来训练具有梯度提升的模型。

### 设置

以下问题将为您提供有关工作的反馈。运行以下单元格以设置反馈系统。

```python
# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex6 import *
print("Setup Complete")
```

您将使用上一个练习中的 Kaggle Learn 用户数据集的房价竞争。

在不更改的情况下运行下一个代码单元，以在 X_train、X_valid、y_train 和 y_valid 中加载训练集和验证集。测试集加载到 X_test 中。

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
#对齐X_train, X_valid
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)
```

pandas中`get_dummies`函数相当于one-hot encoding，把字符串数据转为数值矩阵。

align() 对象方法的语法为：

```python
align(self, other, join: 'str' = 'outer',
      axis: 'Axis | None' = None,
      level: 'Level | None' = None,
      copy: 'bool' = True, fill_value=None,
      method: 'str | None' = None, limit=None,
      fill_axis: 'Axis' = 0,
      broadcast_axis: 'Axis | None' = None) -> 'DataFrame'
```

参数：

- other : DataFrame or Series，要对齐的对象
- join : {'outer', 'inner', 'left', 'right'}, 默认 'outer'，对齐时的连接方式
- axis : 另一个对象对齐轴, 默认为 None，行列同时对齐，可选 0 或者 1
- level : int or level 名称, 默认 None，在一个级别上广播，匹配通过多索引级别。
- copy : bool, 默认 True，总是返回新对象。如果copy=False 且不需要重新索引然后返回原始对象。
- fill_value : scalar, 默认 np.NaN，用于填充缺失值的值。默认为 NaN，但可以是任意值
  “兼容”值。
- method : {'backfill', 'bfill', 'pad', 'ffill', None}, 默认 None。用于填充缺失值的方法：
  - pad / ffill: 将上一个有效观察向前传播到下一个有效观察
  - backfill / bfill: 使用下一个有效的观察来填补空白
- limit : int, 默认 None。如果指定了method，则这是连续要向前/向后填充的NaN值。换句话说，如果有如果差距超过了连续的NAN数，那么
  部分填满。如果未指定方法，则这是沿整个轴输入NAN的最大数量填满。如果不是无，则必须大于0
- fill_axis : {0 or 'index', 1 or 'columns'}, 默认 0，填充轴、方法和限制。
- broadcast_axis : {0 or 'index', 1 or 'columns'}, 默认 None。如果将两个对象对齐，则沿该轴广播值不同的维度。

返回：

(left, right) : (DataFrame, type of other)，对齐的对象。

https://www.gairuo.com/p/pandas-align

### 第 1 步：构建模型

#### Part A

在此步骤中，您将使用梯度提升构建和训练您的第一个模型。 

- 首先将 my_model_1 设置为 XGBoost 模型。使用 XGBRegressor 类，并将随机种子设置为 0 (random_state=0)。将所有其他参数保留为默认值。 
- 然后，将模型拟合到 X_train 和 y_train 中的训练数据。

```python
from xgboost import XGBRegressor

# Define the model
my_model_1 = XGBRegressor(random_state=0) 

# Fit the model
my_model_1.fit(X_train,y_train)
```

#### Part B

将 `predictions_1` 设置为模型对验证数据的预测。回想一下，验证功能存储在 `X_valid` 中。

```python
from sklearn.metrics import mean_absolute_error

# Get predictions
predictions_1 = my_model_1.predict(X_valid)
```

#### Part C

最后，使用 `mean_absolute_error()` 函数计算与验证集的预测相对应的平均绝对误差 (MAE)。回想一下，验证数据的标签存储在 `y_valid` 中。

```python
# Calculate MAE
mae_1 = mean_absolute_error(predictions_1, y_valid)

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_1)
```

```
Mean Absolute Error: 17662.736729452055
```

### 第二步：改进模型

现在您已经训练了一个默认模型作为基线，是时候修改参数了，看看您是否可以获得更好的性能！ 

- 首先使用 XGBRegressor 类将 my_model_2 设置为 XGBoost 模型。使用您在上一教程中学到的知识来弄清楚如何更改默认参数（如 n_estimators 和 learning_rate）以获得更好的结果。 
- 然后，将模型拟合到 X_train 和 y_train 中的训练数据。 
- 将 predictions_2 设置为模型对验证数据的预测。回想一下，验证功能存储在 X_valid 中。 
- 最后，使用 mean_absolute_error() 函数计算与验证集上的预测对应的平均绝对误差 (MAE)。回想一下，验证数据的标签存储在 y_valid 中。 

为了将此步骤标记为正确，my_model_2 中的模型必须获得比 my_model_1 中的模型更低的 MAE。

```python
# Define the model
my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.05,random_state=0)  

# Fit the model
my_model_2.fit(X_train,y_train)

# Get predictions
predictions_2 = my_model_2.predict(X_valid)

# Calculate MAE
mae_2 = mean_absolute_error(predictions_2, y_valid)

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_2)
```

```
Mean Absolute Error: 16688.691513270547
```

### 第 3 步：打破模型

在此步骤中，您将创建一个性能比步骤 1 中的原始模型差的模型。这将帮助您培养如何设置参数的直觉。你甚至可能会发现你意外地获得了更好的表现，这最终是一个很好的问题和宝贵的学习经验！ 

- 首先使用 XGBRegressor 类将 my_model_3 设置为 XGBoost 模型。使用您在上一教程中学到的知识来弄清楚如何更改默认参数（如 n_estimators 和 learning_rate）来设计模型以获得高 MAE。 

- 然后，将模型拟合到 X_train 和 y_train 中的训练数据。 

- 将 predictions_3 设置为模型对验证数据的预测。回想一下，验证功能存储在 X_valid 中。 

- 最后，使用 mean_absolute_error() 函数计算与验证集上的预测对应的平均绝对误差 (MAE)。回想一下，验证数据的标签存储在 y_valid 中。 

为了将此步骤标记为正确，my_model_3 中的模型必须获得比 my_model_1 中的模型更高的 MAE。

```python
# Define the model
my_model_3 = XGBRegressor(n_estimators=50,learning_rate=0.5)
# my_model_3 = XGBRegressor(n_estimators=1)

# Fit the model
my_model_3.fit(X_train, y_train) # Your code here

# Get predictions
predictions_3 = my_model_3.predict(X_valid)

# Calculate MAE
mae_3 = mean_absolute_error(predictions_3, y_valid)

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_3)
```

```
Mean Absolute Error: 20948.60493364726
```

```
Mean Absolute Error: 127895.0828807256
```

### 继续 

继续了解数据泄露(**[data leakage](https://www.kaggle.com/alexisbcook/data-leakage)**)。这是数据科学家需要理解的一个重要问题，它有可能以微妙而危险的方式破坏你的模型！