# 14-Model Validation

测量模型的性能，以便您可以测试和比较备选方案。

你已经建立了一个模型。但它有多好？ 在本课中，您将学习使用模型验证来衡量模型的质量。衡量模型质量是迭代改进模型的关键。

## 什么是模型验证？

您将需要评估您曾经构建的几乎所有模型。在大多数（尽管不是全部）应用中，模型质量的相关衡量标准是预测准确性。换句话说，模型的预测是否会接近实际发生的情况。 

许多人在测量预测准确性时犯了一个巨大的错误。他们使用训练数据进行预测，并将这些预测与训练数据中的目标值进行比较。稍后您将看到这种方法的问题以及如何解决它，但让我们先考虑一下我们将如何做到这一点。 

您首先需要将模型质量总结为一种易于理解的方式。如果您比较 10,000 套房屋的预测和实际房屋价值，您可能会发现预测好坏参半。查看包含 10,000 个预测值和实际值的列表将毫无意义。我们需要将其总结为一个指标。 

总结模型质量的指标有很多，但我们将从一个称为平均绝对误差（也称为 MAE）的指标开始。让我们从最后一个词 error 开始分解这个指标。

每栋房屋的预测误差为：

```
error=actual−predicted
```

因此，如果一栋房子的价格为 150,000 美元，而您预测它的价格为 100,000 美元，则误差为 50,000 美元。 

使用 MAE 度量，我们取每个错误的绝对值。这会将每个错误转换为正数。然后我们取这些绝对误差的平均值。这是我们衡量模型质量的标准。用简单的英语来说，可以说是

平均而言，我们的预测相差大约 X。

要计算 MAE，我们首先需要一个模型。它内置在下面的隐藏单元格中，您可以通过单击代码按钮查看。

```python
# Data Loading Code Hidden Here
import pandas as pd

# Load data
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)#若某行有空值，则删除该行
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(X, y)
```

```
DecisionTreeRegressor()
```

[Python中缺失值删除 pd.dropna()函数](https://blog.csdn.net/liujingwei8610/article/details/123014771)

一旦我们有了一个模型，下面是我们计算平均绝对误差的方法：

```python
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
```

```
434.71594577146544
```

[Metric评价指标及损失函数-Error系列之平均绝对误差（Mean Absolute Error，MAE）](https://zhuanlan.zhihu.com/p/353125247)

## The Problem with "In-Sample" Scores(“样本内”分数的问题)

我们刚刚计算的度量可以称为“样本内”分数。我们使用单个房屋“样本”来构建模型和评估它。这就是为什么这很糟糕。 

想象一下，在大型房地产市场中，门的颜色与房价无关。 

但是，在您用于构建模型的数据样本中，所有带绿色门的房屋都非常昂贵。该模型的工作是找到预测房价的模式，因此它会看到这种模式，并且总是会预测绿门房屋的高价。 

由于此模式源自训练数据，因此模型在训练数据中将显得准确。 

但是，如果在模型看到新数据时这种模式不成立，那么在实践中使用该模型就会非常不准确。 

由于模型的实用价值来自对新数据的预测，因此我们测量了未用于构建模型的数据的性能。最直接的方法是从模型构建过程中排除一些数据，然后使用这些数据来测试模型对以前从未见过的数据的准确性。该数据称为验证数据(**validation data**)。

## 编码它

scikit-learn 库有一个函数 train_test_split 将数据分成两部分。我们将使用其中一些数据作为训练数据来拟合模型，我们将使用其他数据作为验证数据来计算 mean_absolute_error。 这是代码：

```python
from sklearn.model_selection import train_test_split

# 将数据拆分为训练和验证数据，用于特征和目标
# 拆分基于随机数生成器。提供一个数值到
# random_state 参数保证每次我们得到相同的分割
# 运行这个脚本。
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
```

```
258930.03550677857
```

结论

您对样本内数据的平均绝对误差约为 500 美元。样本外超过 250,000 美元。 

这是一个几乎完全正确的模型和一个无法用于大多数实际目的的模型之间的区别。作为参考，验证数据中的平均房屋价值为 110 万美元。所以新数据的误差大约是平均房屋价值的四分之一。 有很多方法可以改进这个模型，例如尝试寻找更好的特征或不同的模型类型。

## Exercise: Model Validation

### 回顾

你已经建立了一个模型。在本练习中，您将测试您的模型有多好。 运行下面的单元格以在上一个练习停止的地方设置您的编码环境。

```python
# Code you have previously used to load data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex4 import *
print("Setup Complete")
```

## Exercises

### Step 1: Split Your Data

使用 train_test_split 函数拆分数据。 给它参数 random_state=1 以便检查函数知道在验证您的代码时会发生什么。 回想一下，您的功能加载在 DataFrame X 中，而您的目标加载在 y 中。

```python
# Import the train_test_split function and uncomment
from sklearn.model_selection import train_test_split

# fill in and uncomment
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=1)
```

[Sklearn : train_test_split()函数的用法](https://blog.csdn.net/DebugYing/article/details/122477435)

参数介绍
① X ：（必需） 待划分的样本集
② y ：（非必需） 样本标签target（如果你只是想把数据简单的分为两部分，不涉及分类算法等需要标注数据标签的情况就无须设置）
③ train_size ： （非必需） int型或float型，整型表示划分后的数据个数；浮点型表示划分数据的比例。
④ test_size ：（非必需） 同上
⑤ random_state ：（非必需） int 类型，默认值为None。先笼统的认为是一个控制分裂过程随机性的一个参数。不用管内部实现过程。
⑥ shuffle ：（非必需） 默认为True。控制拆分数据前，原始数据集是否需要打乱再拆分。
⑦ stratify ：（非必需）

### Step 2: Specify and Fit the Model

创建一个 DecisionTreeRegressor 模型并将其拟合到相关数据。创建模型时再次将 random_state 设置为 1。

```python
# You imported DecisionTreeRegressor in your last exercise
# and that code has been copied to the setup code above. So, no need to
from sklearn.tree import DecisionTreeRegressor

# Specify the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit iowa_model with the training data.
iowa_model.fit(train_X,train_y)
```

```
[186500. 184000. 130000.  92000. 164500. 220000. 335000. 144152. 215000.
 262000.]
[186500. 184000. 130000.  92000. 164500. 220000. 335000. 144152. 215000.
 262000.]
```

### Step 3: Make Predictions with Validation data

```python
# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)
```

从验证数据中检查您的预测和实际值。

```python
# print the top few validation predictions
print(iowa_model.predict(val_X.head()))
# print the top few actual prices from validation data
print(y.head().tolist())
```

```
[186500. 184000. 130000.  92000. 164500.]
[208500, 181500, 223500, 140000, 250000]
```

您注意到哪些与您在样本内预测中看到的不同（打印在本页顶部代码单元格之后）。 

你还记得为什么验证预测与样本内（或训练）预测不同吗？这是上一课的一个重要思想。

### Step 4: Calculate the Mean Absolute Error in Validation Data(计算验证数据中的平均绝对误差)

```python
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y,val_predictions)

# uncomment following line to see the validation_mae
print(val_mae)
```

```
29652.931506849316
```

那个MAE好吗？对于跨应用程序适用的良好值没有一般规则。但是您将在下一步中看到如何使用（和改进）这个数字。