# 43-Categorical Variables

那里有很多非数字数据。以下是如何将其用于机器学习。

在本教程中，您将了解什么是分类变量，以及处理此类数据的三种方法。

## 介绍

分类变量仅采用有限数量的值。 

- 考虑一项调查，询问您吃早餐的频率并提供四个选项：“从不”、“很少”、“大部分时间”或“每天”。在这种情况下，数据是分类数据，因为响应属于一组固定的类别。 

- 如果人们对一项关于他们拥有什么品牌的汽车的调查做出回应，他们的回答将分为“本田”、“丰田”和“福特”等类别。在这种情况下，数据也是分类的。 

如果您尝试将这些变量插入 Python 中的大多数机器学习模型而不先对其进行预处理，则会出现错误。在本教程中，我们将比较可用于准备分类数据的三种方法。

## 三种方法

### 1）删除分类变量（Drop Categorical Variables）

处理分类变量最简单的方法是简单地将它们从数据集中删除。这种方法只有在列不包含有用信息的情况下才会有效。

### 2）序数编码（Ordinal Encoding）

序号编码将每个唯一值分配给不同的整数。

![image-20221117133505690](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221117133505690.png)

此方法假定类别的排序：“从不”(0) <“很少”(1) <“大部分时间”(2) <“每天”(3)。 

这个假设在这个例子中是有意义的，因为**类别的排名是无可争辩的**。并非所有分类变量的值都有明确的顺序，但我们将那些有顺序的变量称为顺序变量。对于基于树的模型（如决策树和随机森林），您可以期望序数编码能够很好地处理序数变量。

### 3）One-Hot 编码

单热编码创建新列，指示原始数据中每个可能值的存在（或不存在）。为了理解这一点，我们将通过一个例子来工作。

![image-20221117133702359](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221117133702359.png)

在原始数据集中，“Color”是一个分类变量，具有三个类别：“Red”、“Yellow”和“Green”。相应的 one-hot 编码包含一列对应每个可能的值，一行对应原始数据集中的每一行。无论原始值是“红色”，我们都在“红色”列中输入 1；如果原始值为“黄色”，我们在“黄色”列中输入 1，依此类推。

与序号编码相反，one-hot 编码不假定类别的顺序。因此，如果分类数据中没有明确的排序（例如，“红色”不多于或少于“黄色”），您可以期望这种方法特别有效。我们将**没有内在排名**的分类变量称为**名义变量**（**nominal variables**.）。

如果分类变量具有大量值（即，您通常不会将它用于具有超过 15 个不同值的变量），则**单热编码通常不会很好地执行（分类变量值超过15个）**。

## Example

与之前的教程一样，我们将使用 [Melbourne Housing dataset](https://www.kaggle.com/dansbecker/melbourne-housing-snapshot/home).

我们不会关注数据加载步骤。相反，您可以想象您已经在 X_train、X_valid、y_train 和 y_valid 中拥有训练和验证数据。

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Separate target from predictors
y = data.Price
X = data.drop(['Price'], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# Drop columns with missing values (simplest approach)
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()] 
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

# "Cardinality" means the number of unique values in a column
# “基数”表示列中唯一值的数量
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
# 选择基数相对较低的分类列（方便但随意）
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
```

unique()方法返回的是去重之后的不同值，而nunique()方法则直接返回不同值的个数。

我们使用下面的 head() 方法查看训练数据。

```python
X_train.head()
```

![image-20221117134838523](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221117134838523.png)

接下来，我们获得训练数据中所有分类变量的列表。 

我们通过检查每一列的数据类型（或数据类型）来做到这一点。 object dtype 表示一列有文本（理论上它可能有其他内容，但这对我们的目的来说并不重要）。对于此数据集，带有文本的列表示分类变量。

```python
# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
```

```
Categorical variables:
['Type', 'Method', 'Regionname']
```

### 定义函数来衡量每种方法的质量

我们定义了一个函数 score_dataset() 来比较处理分类变量的三种不同方法。此函数报告随机森林模型的平均绝对误差 (MAE)。一般来说，我们希望 MAE 越低越好！

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
```

### Score from Approach 1 (Drop Categorical Variables)

我们使用 select_dtypes() 方法删除object列。

```python
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
```

```
MAE from Approach 1 (Drop categorical variables):
175703.48185157913
```

### Score from Approach 2 (Ordinal Encoding)

`Scikit-learn` 有一个 `OrdinalEncoder` 类，可用于获取序号编码。我们遍历分类变量并将序数编码器分别应用于每一列。

```python
from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
```

```
MAE from Approach 2 (Ordinal Encoding):
165936.40548390493
```

在上面的代码单元中，对于每一列，我们将每个唯一值随机分配给不同的整数。这是一种比提供自定义标签更简单的常用方法；然而，如果我们为所有序数变量提供更明智的标签，我们可以期待性能的额外提升。

### Score from Approach 3 (One-Hot Encoding)

我们使用 scikit-learn 中的 `OneHotEncoder` 类来获得单热编码。有许多参数可用于自定义其行为。 

- 我们设置`handle_unknown='ignore' `以避免验证数据包含训练数据中未表示的类时出现错误，并且 
- 设置 `sparse=False` 确保编码列作为 numpy 数组（而不是稀疏矩阵）返回。 

要使用编码器，我们只提供我们想要进行单热编码的分类列。例如，为了对训练数据进行编码，我们提供 `X_train[object_cols]`。 （下面代码单元格中的 `object_cols` 是带有分类数据的列名列表，因此 `X_train[object_cols]` 包含训练集中的所有分类数据。）

```python
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
```

```
MAE from Approach 3 (One-Hot Encoding):
166089.4893009678
```

`pd.concat([a,b],axis)`数据按列的方向合并=a的列+b的列

[pandas中concat()的用法](https://blog.csdn.net/xinyihhh/article/details/122141336)

## 哪种方法最好？

在这种情况下，删除分类列（方法 1）表现最差，因为它具有最高的 MAE 分数。至于其他两种方法，由于返回的 MAE 分数的值非常接近，因此似乎没有任何有意义的优势优于另一种。 

一般来说，one-hot 编码（方法 3）通常会表现最好，而删除分类列（方法 1）通常表现最差，但它因情况而异。

## 结论 

世界充满了分类数据。如果您知道如何使用这种常见的数据类型，您将成为一名更有效率的数据科学家！

## Exercise: Categorical Variables

通过对分类变量进行编码，您将获得迄今为止最好的结果！

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
from learntools.ml_intermediate.ex3 import *
print("Setup Complete")
```

在本练习中，您将使用 Kaggle Learn 用户的房价竞赛数据。

在不更改的情况下运行下一个代码单元，以在 X_train、X_valid、y_train 和 y_valid 中加载训练集和验证集。测试集加载到 X_test 中。

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv('../input/train.csv', index_col='Id') 
X_test = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y,train_size=0.8, test_size=0.2,random_state=0)
```

使用下一个代码单元打印数据的前五行

```python
X_train.head()
```

请注意，数据集包含数值变量和分类变量。在训练模型之前，您需要对分类数据进行编码。 要比较不同的模型，您将使用教程中相同的 score_dataset() 函数。此函数报告随机森林模型的平均绝对误差 (MAE)。

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
```

### Step 1: Drop columns with categorical data

您将从最直接的方法开始。使用下面的代码单元对 X_train 和 X_valid 中的数据进行预处理，以删除包含分类数据的列。将预处理后的 DataFrames 分别设置为 drop_X_train 和 drop_X_valid。

```python
# Fill in the lines below: drop columns in training and validation data
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
```

```python
print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
```

```
MAE from Approach 1 (Drop categorical variables):
17837.82570776256
```

在进入序号编码之前，我们将研究数据集。具体来说，我们将查看“Condition2”列。下面的代码单元打印训练和验证集中的唯一条目。

```python
print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())
```

```
Unique values in 'Condition2' column in training data: ['Norm' 'PosA' 'Feedr' 'PosN' 'Artery' 'RRAe']

Unique values in 'Condition2' column in validation data: ['Norm' 'RRAn' 'RRNn' 'Artery' 'Feedr' 'PosN']
```

### Step 2: Ordinal encoding

#### Part A

如果您现在编写代码： 

- 将序号编码器拟合到训练数据，然后 

- 用它来转换训练和验证数据， 

你会得到一个错误。你能看出为什么会这样吗？ （您需要使用上面的输出来回答这个问题。）

将序数编码器拟合到训练数据中的列，为训练数据中出现的每个唯一值创建相应的整数值标签。如果**验证数据包含未出现在训练数据中的值，编码器将抛出错误**，因为这些值没有分配给它们的整数。请注意，验证数据中的“Condition2”列包含值“RRAn”和“RRNn”，但这些值不会出现在训练数据中——因此，如果我们尝试将序号编码器与 scikit-learn 一起使用，代码会抛出错误。

这是您在处理真实世界数据时会遇到的常见问题，有许多方法可以解决此问题。例如，您可以编写自定义序号编码器来处理新类别。然而，最简单的方法是删除有问题的分类列。 运行下面的代码单元将有问题的列保存到 Python 列表 bad_label_cols。同样，可以安全地进行序号编码的列存储在 good_label_cols 中。

```python
# Categorical columns in the training data
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if 
                   set(X_valid[col]).issubset(set(X_train[col]))]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
print('Categorical columns that will be ordinal encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)
```

#### Part B

使用下一个代码单元对 X_train 和 X_valid 中的数据进行顺序编码。将预处理后的 DataFrames 分别设置为 label_X_train 和 label_X_valid。 

- 我们在下面提供了代码，用于从数据集中删除 bad_label_cols 中的分类列。 

- 您应该对 good_label_cols 中的分类列进行序号编码。

```python
from sklearn.preprocessing import OrdinalEncoder

# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply ordinal encoder 
ordinal_encoder = OrdinalEncoder()
label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(X_valid[good_label_cols])
```

运行下一个代码单元以获得此方法的 MAE。

```python
print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
```

```
MAE from Approach 2 (Ordinal Encoding):
17098.01649543379
```

到目前为止，您已经尝试了两种不同的方法来处理分类变量。而且，您已经看到编码分类数据比从数据集中删除列产生更好的结果。 很快，您将尝试单热编码。在此之前，我们需要讨论一个额外的主题。首先运行下一个代码单元而不做任何更改。

```python
# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])
```

[闲聊Pandas map函数](https://zhuanlan.zhihu.com/p/417546385)

```
[('Street', 2),
 ('Utilities', 2),
 ('CentralAir', 2),
 ('LandSlope', 3),
 ('PavedDrive', 3),
 ('LotShape', 4),
 ('LandContour', 4),
 ('ExterQual', 4),
 ('KitchenQual', 4),
 ('MSZoning', 5),
 ('LotConfig', 5),
 ('BldgType', 5),
 ('ExterCond', 5),
 ('HeatingQC', 5),
 ('Condition2', 6),
 ('RoofStyle', 6),
 ('Foundation', 6),
 ('Heating', 6),
 ('Functional', 6),
 ('SaleCondition', 6),
 ('RoofMatl', 7),
 ('HouseStyle', 8),
 ('Condition1', 9),
 ('SaleType', 9),
 ('Exterior1st', 15),
 ('Exterior2nd', 16),
 ('Neighborhood', 25)]
```

### Step 3: Investigating cardinality

#### Part A

上面的输出显示，对于具有分类数据的每一列，该列中唯一值的数量。例如，训练数据中的“Street”列有两个唯一值：“Grvl”和“Pave”，分别对应于碎石路和柏油路。 

我们将分类变量的唯一条目数称为该分类变量的基数。例如，“街道”变量的基数为 2。 使用上面的输出来回答下面的问题。

```python
# Fill in the line below: How many categorical variables in the training data
# have cardinality greater than 10?
high_cardinality_numcols = 3

# Fill in the line below: How many columns are needed to one-hot encode the 
# 'Neighborhood' variable in the training data?
num_cols_neighborhood = 25
```

#### Part B

对于多行的大型数据集，one-hot encoding 可以极大地扩展数据集的大小。出于这个原因，我们通常只会对基数相对较低的列进行单热编码。然后，可以从数据集中删除高基数列，或者我们可以使用序号编码。 

例如，假设一个数据集有 10,000 行，并且包含一个具有 100 个唯一条目的分类列。 

- 如果将此列替换为相应的 one-hot 编码，则将多少条目添加到数据集中？ 

- 如果我们改为用序号编码替换该列，则会添加多少条目？ 

使用您的答案填写下面的行。

```python
# Fill in the line below: How many entries are added to the dataset by 
# replacing the column with a one-hot encoding?
OH_entries_added = 1e4*100-1e4

# Fill in the line below: How many entries are added to the dataset by
# replacing the column with an ordinal encoding?
label_entries_added = 0
```

接下来，您将试验单热编码。但是，您不会对数据集中的所有分类变量进行编码，而只会为基数小于 10 的列创建单热编码。

在不更改的情况下运行下面的代码单元，将 `low_cardinality_cols` 设置为包含将被一次性编码的列的 Python 列表。同样，`high_cardinality_cols` 包含将从数据集中删除的分类列列表。

```python
# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
```

```
Categorical columns that will be one-hot encoded: ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition']

Categorical columns that will be dropped from the dataset: ['Neighborhood', 'Exterior2nd', 'Exterior1st']
```

### Step 4: One-hot encoding

使用下一个代码单元对 X_train 和 X_valid 中的数据进行单热编码。将预处理后的 DataFrames 分别设置为 OH_X_train 和 OH_X_valid。 

数据集中分类列的完整列表可以在 Python 列表 object_cols 中找到。 您应该只对 low_cardinality_cols 中的分类列进行单热编码。应从数据集中删除所有其他分类列。

```python
from sklearn.preprocessing import OneHotEncoder

# Use as many lines of code as you need!
# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train,OH_cols_train],axis=1)
OH_X_valid = pd.concat([num_X_valid,OH_cols_valid],axis=1)
```

运行下一个代码单元以获得此方法的 MAE。

```python
print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
```

```
MAE from Approach 3 (One-Hot Encoding):
17525.345719178084
```

生成测试预测并提交您的结果

完成第 4 步后，如果您想使用所学知识将结果提交到排行榜，则需要在生成预测之前对测试数据进行预处理。

## 继续 

通过缺失值处理和分类编码，您的建模过程变得越来越复杂。当您想要保存模型以供将来使用时，这种复杂性会变得更糟。管理这种复杂性的关键是称为管道的东西。 

学习使用管道预处理包含分类变量、缺失值和您的数据给您带来的任何其他混乱的数据集。
