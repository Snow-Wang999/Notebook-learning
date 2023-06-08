# 33-Your First Machine Learning Model

## Selecting Data for Modeling

您的数据集有太多变量，无法将您的头脑包裹起来，甚至无法很好地打印出来。您如何将这些庞大的数据量缩减为您可以理解的内容？ 

我们将从使用我们的直觉选择一些变量开始。以后的课程将向您展示自动确定变量优先级的统计技术。 

## 选择变量/列

我们需要查看数据集中所有列的列表。这是通过 DataFrame 的 `.columns` 属性（下面代码的最后一行）完成的。

```python
import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data.columns
```

```python
Index(['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
       'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
       'Longtitude', 'Regionname', 'Propertycount'],
      dtype='object')
```

## `dropna` 删除缺失值

```python
# 墨尔本数据有一些缺失值（一些房屋的一些变量没有记录。）
# 我们将在后面的教程中学习处理缺失值。
# 您的爱荷华州数据在您使用的列中没有缺失值。
# 所以我们现在将采取最简单的选择，并从我们的数据中删除房屋。
# 现在不用担心这么多，虽然代码是：
# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.  
# Your Iowa data doesn't have missing values in the columns you use. 
# So we will take the simplest option for now, and drop houses from our data. 
# Don't worry about this much for now, though the code is:

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)
```

有很多方法可以选择数据的子集。 Pandas 课程更深入地介绍了这些，但我们现在将重点关注两种方法。 

- 点表示法（dot notation），我们用它来选择“预测目标” 

- 使用列的列表（a column list）进行选择，我们使用它来选择“特征”

## Selecting The Prediction Target(选取目标列)

您可以使用点符号提取变量。这个单列存储在一个系列中，它大致类似于只有一列数据的 DataFrame。 

我们将使用点符号来选择我们想要预测的列，这称为预测目标。按照惯例，预测目标称为 y。所以我们需要保存墨尔本数据中房价的代码是

```python
y = melbourne_data.Price
```

## Choosing "Features"（选择特征列）

输入到我们模型中的列（后来用于进行预测）称为“特征”。在我们的例子中，这些将是用于确定房价的列。有时，您将使用除目标之外的所有列作为特征。其他时候，使用更少的功能会更好。 

现在，我们将构建一个只有几个特征的模型。稍后您将看到如何迭代和比较使用不同功能构建的模型。 

我们通过在括号内提供列名列表来选择多个特征。该列表中的每个项目都应该是一个字符串（带引号）。 

这是一个例子：

```python
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
```

按照惯例，此数据称为 X。

```python
X = melbourne_data[melbourne_features]
```

让我们快速回顾一下我们将使用 describe 方法和 head 方法预测房价的数据，它显示了前几行。

```python
X.describe()
```

|       | Rooms       | Bathroom    | Landsize     | Lattitude   | Longtitude  |
| :---- | :---------- | :---------- | :----------- | :---------- | ----------- |
| count | 6196.000000 | 6196.000000 | 6196.000000  | 6196.000000 | 6196.000000 |
| mean  | 2.931407    | 1.576340    | 471.006940   | -37.807904  | 144.990201  |
| std   | 0.971079    | 0.711362    | 897.449881   | 0.075850    | 0.099165    |
| min   | 1.000000    | 1.000000    | 0.000000     | -38.164920  | 144.542370  |
| 25%   | 2.000000    | 1.000000    | 152.000000   | -37.855438  | 144.926198  |
| 50%   | 3.000000    | 1.000000    | 373.000000   | -37.802250  | 144.995800  |
| 75%   | 4.000000    | 2.000000    | 628.000000   | -37.758200  | 145.052700  |
| max   | 8.000000    | 8.000000    | 37000.000000 | -37.457090  | 145.526350  |

```python
X.head()
```

|      | Rooms | Bathroom | Landsize | Lattitude | Longtitude |
| :--- | :---- | :------- | :------- | :-------- | ---------- |
| 1    | 2     | 1.0      | 156.0    | -37.8079  | 144.9934   |
| 2    | 3     | 2.0      | 134.0    | -37.8093  | 144.9944   |
| 4    | 4     | 1.0      | 120.0    | -37.8072  | 144.9941   |
| 6    | 3     | 2.0      | 245.0    | -37.8024  | 144.9993   |
| 7    | 2     | 1.0      | 256.0    | -37.8060  | 144.9954   |

使用这些命令直观地检查您的数据是数据科学家工作的重要组成部分。您会经常在数据集中发现值得进一步检查的惊喜。

## 建立你的模型

您将使用 scikit-learn 库来创建模型。在编码时，这个库被编写为 sklearn，正如您将在示例代码中看到的那样。 Scikit-learn 很容易成为最流行的库，用于对通常存储在 DataFrame 中的数据类型进行建模。

构建和使用模型的步骤是： 

- 定义（**Define**）：它将是什么类型的模型？决策树？其他类型的模型？还指定了模型类型的一些其他参数。 
- 拟合（**Fit**）：从提供的数据中捕获模式。这是建模的核心。 
- 预测（**Predict**）：听起来像什么 
- 评估（**Evaluate**）：确定模型预测的准确性。

这是一个使用 scikit-learn 定义决策树模型并将其与特征和目标变量拟合的示例。

```python
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)
```

```
DecisionTreeRegressor(random_state=1)
```

许多机器学习模型在模型训练中允许一些随机性。为 random_state 指定一个数字可确保您在每次运行中获得相同的结果。这被认为是一种很好的做法。您使用任何数字，模型质量不会有意义地取决于您选择的值。 

我们现在有一个拟合模型，可以用来进行预测。 

在实践中，您会希望对即将上市的新房屋做出预测，而不是我们已经有价格的房屋。但我们将对训练数据的前几行进行预测，以了解预测函数的工作原理。

```python
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
```

```
Making predictions for the following 5 houses:
   Rooms  Bathroom  Landsize  Lattitude  Longtitude
1      2       1.0     156.0   -37.8079    144.9934
2      3       2.0     134.0   -37.8093    144.9944
4      4       1.0     120.0   -37.8072    144.9941
6      3       2.0     245.0   -37.8024    144.9993
7      2       1.0     256.0   -37.8060    144.9954
The predictions are
[1035000. 1465000. 1600000. 1876000. 1636000.]
```

![image-20221106160315501](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106160315501.png)

## Exercise: Your First Machine Learning Model

回顾(recap)

到目前为止，您已经加载了数据并使用以下代码对其进行了审查。运行此单元格以在上一步停止的位置设置您的编码环境。

```python
# Code you have previously used to load data
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *

print("Setup Complete")
```

## Exercises

### 步骤 1：指定预测目标

选择对应于销售价格的目标变量。将其保存到一个名为 y 的新变量中。您需要打印列列表以查找所需列的名称。

```py
# print the list of columns
home_data.columns
```

```
Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'SalePrice'],
      dtype='object')
```

```python
y = home_data.SalePrice
```

### 第 2 步：创建 X

现在，您将创建一个名为 X 的 DataFrame，其中包含预测特征。 

由于您只需要原始数据中的一些列，因此您将首先创建一个列表，其中包含您想要在 X 中使用的列的名称。 

您将只使用列表中的以下列（您可以复制并粘贴整个列表以节省一些输入，但您仍然需要添加引号）： 

- LotArea
- YearBuilt
- 1stFlrSF
- 2ndFlrSF
- FullBath
- BedroomAbvGr
- TotRmsAbvGrd

创建该功能列表后，使用它来创建将用于拟合模型的 DataFrame。

```python
# Create the list of features below
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Select data corresponding to features in feature_names
X = home_data[feature_names]
```

#### 查看数据

在构建模型之前，快速查看 X 以验证它看起来是否合理

```python
# Review data
# print description or statistics from X
print(X.describe())

# print the top few lines
print(X.head())
```

```
             LotArea    YearBuilt     1stFlrSF     2ndFlrSF     FullBath  \
count    1460.000000  1460.000000  1460.000000  1460.000000  1460.000000   
mean    10516.828082  1971.267808  1162.626712   346.992466     1.565068   
std      9981.264932    30.202904   386.587738   436.528436     0.550916   
min      1300.000000  1872.000000   334.000000     0.000000     0.000000   
25%      7553.500000  1954.000000   882.000000     0.000000     1.000000   
50%      9478.500000  1973.000000  1087.000000     0.000000     2.000000   
75%     11601.500000  2000.000000  1391.250000   728.000000     2.000000   
max    215245.000000  2010.000000  4692.000000  2065.000000     3.000000   

       BedroomAbvGr  TotRmsAbvGrd  
count   1460.000000   1460.000000  
mean       2.866438      6.517808  
std        0.815778      1.625393  
min        0.000000      2.000000  
25%        2.000000      5.000000  
50%        3.000000      6.000000  
75%        3.000000      7.000000  
max        8.000000     14.000000  
   LotArea  YearBuilt  1stFlrSF  2ndFlrSF  FullBath  BedroomAbvGr  \
0     8450       2003       856       854         2             3   
1     9600       1976      1262         0         2             3   
2    11250       2001       920       866         2             3   
3     9550       1915       961       756         1             3   
4    14260       2000      1145      1053         2             4   

   TotRmsAbvGrd  
0             8  
1             6  
2             6  
3             7  
4             9  
```

### 第 3 步：指定和拟合模型

创建一个 DecisionTreeRegressor 并将其保存为 iowa_model。确保您已从 sklearn 完成相关导入以运行此命令。 然后使用您在上面保存的 X 和 y 中的数据拟合您刚刚创建的模型。

```python
from sklearn.tree import DecisionTreeRegressor
#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model
iowa_model.fit(X,y)
```

### 第 4 步：做出预测

使用模型的 predict 命令使用 X 作为数据进行预测。将结果保存到名为预测的变量中。

```python
predictions = iowa_model.predict(X)
print(predictions)
```

### 想想你的结果

使用 head 方法将前几个预测与这些相同房屋的实际房屋价值（以 y 为单位）进行比较。有什么意外吗？

```python
# You can write code in this cell
predictions = iowa_model.predict(X.head())
print(predictions)
print(y.head())
```

![image-20221106163019504](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106163019504.png)

很自然地会问模型的预测有多准确，以及如何改进它。那将是你的下一步。