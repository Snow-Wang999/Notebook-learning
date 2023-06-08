# 42-Missing Values

缺失值发生。为真实数据集中的这一常见挑战做好准备。

在本教程中，您将学习三种处理缺失值的方法。然后，您将比较这些方法在真实世界数据集上的有效性。

## 介绍 

数据以缺失值结束的方式有很多种。例如， 

- 两居室的房子不包括第三间卧室的价值。 
- 调查受访者可以选择不分享他的收入。 

如果您尝试使用具有缺失值的数据构建模型，大多数机器学习库（包括 scikit-learn）都会出错。因此，您需要选择以下策略之一。

### 三种方法

1) #### 一个简单的选择：删除有缺失值的列

   最简单的选择是删除具有缺失值的列。

   ![image-20221117115352835](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221117115352835.png)

   除非删除的列中的大多数值丢失，否则模型将无法使用这种方法访问大量（可能有用！）信息。举一个极端的例子，考虑一个有 10,000 行的数据集，其中一个重要的列缺少一个条目。这种方法会完全丢弃该列！

2) #### 更好的选择：插补（Imputation）

   插补用一些数字填充缺失值。例如，我们可以填写每一列的平均值。

   ![image-20221117115524524](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221117115524524.png)

   在大多数情况下，推算值不会完全正确，但与完全删除该列相比，它通常会产生更准确的模型。

3) #### 插补的扩展

   插补是标准方法，通常效果很好。但是，估算值可能系统地高于或低于其实际值（未在数据集中收集）。或者，具有缺失值的行可能以其他方式唯一。在这种情况下，您的模型会通过考虑最初缺失的值来做出更好的预测。

   在这种方法中，我们像以前一样估算缺失值。此外，对于原始数据集中缺少条目的每一列，我们添加一个新列来显示估算条目的位置。 

   在某些情况下，这将显着改善结果。在其他情况下，它根本没有帮助。

## Example

在示例中，我们将使用 Melbourne Housing 数据集。我们的模型将使用房间数量和土地面积等信息来预测房价。 我们不会关注数据加载步骤。相反，您可以想象您已经在 X_train、X_valid、y_train 和 y_valid 中拥有训练和验证数据。

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select target
y = data.Price

# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
```

### 定义函数来衡量每种方法的质量

我们定义了一个函数 score_dataset() 来比较处理缺失值的不同方法。此函数报告随机森林模型的平均绝对误差 (MAE)。

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
```

#### Score from Approach 1 (Drop Columns with Missing Values)

由于我们同时使用训练集和验证集，因此我们要小心地在两个 DataFrame 中删除相同的列。

```python
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
```

```
MAE from Approach 1 (Drop columns with missing values):
183550.22137772635
```

any方法

https://blog.csdn.net/qq_44710204/article/details/107423702

`DataFrame.any(axis=0, bool_only=None, skipna=True, level=None) `

作用：返回是否至少一个元素为真

#### Score from Approach 2 (Imputation)

接下来，我们使用 SimpleImputer 将缺失值替换为每列的平均值。 

虽然很简单，但填充平均值通常表现得很好（但这因数据集而异）。虽然统计学家已经尝试使用更复杂的方法来确定估算值（例如回归插补），但一旦将结果插入复杂的机器学习模型，复杂的策略通常不会带来额外的好处。

```python
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
```

```
MAE from Approach 2 (Imputation):
178166.46269899711
```

我们看到方法 2 的 MAE 低于方法 1，因此方法 2 在此数据集上表现更好。

#### Score from Approach 3 (An Extension to Imputation)

接下来，我们估算缺失值，同时跟踪估算了哪些值。

```python
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
```

```
MAE from Approach 3 (An Extension to Imputation):
178927.503183954
```

正如我们所见，方法 3 的性能略低于方法 2。

那么，为什么插补比删除列表现更好？

训练数据有 10864 行和 12 列，其中三列包含缺失数据。对于每一列，不到一半的条目丢失。因此，删除列会删除很多有用的信息，因此插补会表现得更好是有道理的。

```python
# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
```

```
(10864, 12)
Car               49
BuildingArea    5156
YearBuilt       4307
dtype: int64
```

### 结论 

通常，相对于我们简单地删除具有缺失值的列（在方法 1 中），估算缺失值（在方法 2 和方法 3 中）产生了更好的结果。

## Exercise: Missing Values

现在轮到您测试您对缺失值处理的新知识了。你可能会发现它有很大的不同。

### 设置

这些问题将为您提供有关您工作的反馈。运行以下单元格以设置反馈系统。

```python
# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex2 import *
print("Setup Complete")
```

在本练习中，您将使用 Kaggle Learn 用户的房价竞赛数据。

在不更改的情况下运行下一个代码单元，以在 X_train、X_valid、y_train 和 y_valid 中加载训练集和验证集。测试集加载到 X_test 中。

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
```

使用下一个代码单元打印数据的前五行。

```python
X_train.head()
```

![image-20221117123617059](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221117123617059.png)

您已经可以在前几行中看到一些缺失值。在下一步中，您将更全面地了解数据集中的缺失值。

### 第一步：初步调查

```python
# Shape of training data (num_rows, num_columns)
print(X_train.shape)
​
# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
```

```
(1168, 36)
LotFrontage    212
MasVnrArea       6
GarageYrBlt     58
dtype: int64
```

#### Part A

使用以上输出回答以下问题。

```python
# Fill in the line below: How many rows are in the training data?
num_rows = 1168

# Fill in the line below: How many columns in the training data
# have missing values?
num_cols_with_missing = 3

# Fill in the line below: How many missing entries are contained in 
# all of the training data?
tot_missing = 212+6+58
```

#### Part B

考虑到您上面的回答，您认为处理缺失值的最佳方法可能是什么？

由于数据中缺失的条目相对较少（缺失值百分比最大的列缺失的条目少于其条目的 20%），我们可以预期删除列不太可能产生良好的结果。这是因为我们会丢弃很多有价值的数据，因此插补可能会表现得更好。

要比较处理缺失值的不同方法，您将使用本教程中相同的 score_dataset() 函数。此函数报告随机森林模型的平均绝对误差 (MAE)。

### 第 2 步：删除具有缺失值的列

在此步骤中，您将预处理 X_train 和 X_valid 中的数据以删除具有缺失值的列。将预处理后的 DataFrames 分别设置为 reduced_X_train 和 reduced_X_valid。

```python
# Fill in the line below: get names of columns with missing values
cols_with_missing=[col for col in X_train.columns if X_train[col].isnull().any()] # Your code here

# Fill in the lines below: drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing,axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing,axis=1)
```

```python
print("MAE (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
```

```
MAE (Drop columns with missing values):
17837.82570776256
```

### 第 3 步：插补

#### Part A

使用下一个代码单元格用每列的平均值来估算缺失值。将预处理后的 DataFrame 设置为 imputed_X_train 和 imputed_X_valid。确保列名与 X_train 和 X_valid 中的列名匹配。

```python
from sklearn.impute import SimpleImputer

# Fill in the lines below: imputation
my_imputer= SimpleImputer() # Your code here
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
```

```python
print("MAE (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
```

```
MAE (Imputation):
18062.894611872147
```

#### Part B

比较每种方法的 MAE。结果有什么让你吃惊的吗？为什么您认为一种方法比另一种方法表现更好？

鉴于数据集中的缺失值非常少，我们预计插补比完全删除列表现得更好。但是，我们看到删除列的性能稍微好一些！虽然这可能部分归因于数据集中的噪声，但另一种可能的解释是插补方法与该数据集不太匹配。也就是说，也许不是填写平均值，而是将每个缺失值设置为 0 值、填写最常遇到的值或使用其他一些方法更有意义。例如，考虑 GarageYrBlt 列（表示车库建造的年份）。在某些情况下，缺失值可能表示房屋没有车库。在这种情况下，沿着每列填写中值是否更有意义？或者我们可以通过在每一列中填写最小值来获得更好的结果吗？目前尚不清楚在这种情况下什么是最好的，但也许我们可以立即排除一些选项——例如，将此列中的缺失值设置为 0 可能会产生可怕的结果！

### 第 4 步：生成测试预测

在这最后一步中，您将使用您选择的任何方法来处理缺失值。对训练和验证功能进行预处理后，您将训练和评估随机森林模型。然后，您将在生成可以提交给比赛的预测之前预处理测试数据！

#### Part A

使用下一个代码单元预处理训练和验证数据。将预处理后的 DataFrames 设置为 final_X_train 和 final_X_valid。您可以在这里使用您选择的任何方法！为了将此步骤标记为正确，您只需要确保： 

- 预处理后的 DataFrame 具有相同的列数， 
- 预处理后的 DataFrame 没有缺失值， 
- final_X_train 和 y_train 具有相同的行数，并且 
- final_X_valid 和 y_valid 具有相同的行数。

```python
# Preprocessed training and validation features
# Fill in the lines below: imputation
my_imputer= SimpleImputer(strategy="median") # Your code here
final_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Fill in the lines below: imputation removed column names; put them back
final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns
```



运行下一个代码单元以训练和评估随机森林模型。 （请注意，我们没有使用上面的 score_dataset() 函数，因为我们很快就会使用训练好的模型来生成测试预测！）

```python
# Define and fit model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)

# Get validation predictions and MAE
preds_valid = model.predict(final_X_valid)
print("MAE (Your approach):")
print(mean_absolute_error(y_valid, preds_valid))
```

```
MAE (Your approach):
17791.59899543379
```

#### Part B

使用下一个代码单元预处理您的测试数据。确保您使用的方法与您预处理训练和验证数据的方式一致，并将预处理的测试特征设置为 final_X_test。 

然后，使用预处理后的测试特征和训练好的模型在 preds_test 中生成测试预测。 

为了将此步骤标记为正确，您只需要确保： 

预处理的测试 DataFrame 没有缺失值，并且 final_X_test 的行数与 X_test 相同。

```python
# Fill in the line below: preprocess test data
final_X_test = pd.DataFrame(my_imputer.transform(X_test))
final_X_test.columns = X_test.columns

# Fill in the line below: get test predictions
preds_test = model.predict(final_X_test)
```

```python
# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
```

