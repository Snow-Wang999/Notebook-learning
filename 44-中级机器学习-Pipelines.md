# 44-Pipelines

通过预处理部署（甚至测试）复杂模型的一项关键技能。

在本教程中，您将学习如何使用管道清理您的建模代码。

## 介绍

管道(**Pipelines**)是保持数据预处理和建模代码井井有条的简单方法。具体来说，流水线捆绑了预处理和建模步骤，因此您可以像使用单个步骤一样使用整个捆绑包。 许多数据科学家在没有管道的情况下拼凑模型，但管道有一些重要的好处。这些包括：

- **更简洁的代码**：在预处理的每一步计算数据都会变得混乱。使用管道，您无需在每个步骤手动跟踪您的训练和验证数据。 
- **错误更少**：误用步骤或忘记预处理步骤的机会更少。 
- **更易于生产**：将模型从原型转变为可大规模部署的模型可能出奇地困难。我们不会在这里讨论许多相关的问题，但管道可以提供帮助。 
- **模型验证的更多选项**：您将在下一个教程中看到一个示例，其中包含交叉验证。

## Example

与之前的教程一样，我们将使用 Melbourne Housing 数据集。

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

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
```

我们使用下面的 head() 方法查看训练数据。请注意，数据包含分类数据和具有缺失值的列。有了管道，两者都可以轻松应对！

```python
X_train.head()
```

![image-20221118135936457](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221118135936457.png)

我们分三步构建完整的管道。

### 第 1 步：定义预处理步骤

类似于管道将预处理和建模步骤捆绑在一起的方式，我们使用 `ColumnTransformer` 类将不同的预处理步骤捆绑在一起。下面的代码： 

- 估算**数值(numerical)**数据中的缺失值，以及 

- 估算缺失值并将单热编码应用于**分类数据(categorical)**。

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
```

### 第 2 步：定义模型

接下来，我们使用熟悉的 `RandomForestRegressor` 类定义随机森林模型。

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)
```

### 第 3 步：创建和评估管道

最后，我们使用 Pipeline 类来定义一个将预处理和建模步骤捆绑在一起的管道。有几件重要的事情需要注意：

- 通过管道，我们预处理训练数据并在一行代码中拟合模型。 （相比之下，如果没有管道，我们必须在不同的步骤中进行插补、单热编码和模型训练。如果我们必须同时处理数值变量和分类变量，这将变得特别混乱！） 

- 通过管道，我们将 `X_valid` 中未处理的特征提供给 `predict()` 命令，管道会在生成预测之前自动对特征进行预处理。 （但是，如果没有管道，我们必须记住在进行预测之前对验证数据进行预处理。）

```python
from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
```

```
MAE: 160679.18917034855
```

## 结论 

管道对于清理机器学习代码和避免错误很有价值，对于具有复杂数据预处理的工作流尤其有用。

## Exercise: Pipelines

### 设置

以下问题将为您提供有关工作的反馈。运行以下单元格以设置反馈系统

```python
# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex4 import *
print("Setup Complete")
```

您将使用 Kaggle Learn 用户的房价竞赛数据。

在不更改的情况下运行下一个代码单元，以在 X_train、X_valid、y_train 和 y_valid 中加载训练集和验证集。测试集加载到 X_test 中。

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True) # 丢弃‘SalePrice’这列中有缺失值的行
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)#丢弃‘SalePrice’这一整列

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2,random_state=0)

# "Cardinality" means the number of unique values in a column，基数——列中唯一值的数量
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
```

```python
X_train.head()
```

![image-20221118143703484](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221118143703484.png)

下一个代码单元使用教程中的代码来预处理数据和训练模型。无需更改即可运行此代码。

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))
```

```
MAE: 17861.780102739725
```

该代码为平均绝对误差 (MAE) 生成一个大约 17862 的值。在下一步中，您将修改代码以做得更好。

### 第 1 步：提高性能

#### Part A

现在轮到你了！在下面的代码单元中，定义您自己的预处理步骤和随机森林模型。填写以下变量的值： 

- 数值转换器 
- 分类变压器 
- 模型 

要通过这部分练习，您只需定义有效的预处理步骤和随机森林模型。

```python
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median') 

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
]) 

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)
```

#### 补充：

##### 缺失值填充策略：

https://blog.csdn.net/m0_72662900/article/details/127427564

###### 1、集中局势

（1）众数：出现次数最多的变量值（M0）；不易受极端值的影响，一个数据集可能没有众数或者有几个众数，用于定序数据和数值型数据。 

（2）中位数：排序后处于中间位置上的1值用Me表示；不易受极端值的影响；主要用于定序数据也可用于数值型数据但不能用于定类数据。
$$
M_e=\begin{cases}X_{(\frac{N+1}{2})},当N为奇数时\\\frac{1}{2}(X_{(\frac{N}{2})}+X_{(\frac{N}{2}+1)}),当N为偶数时\end{cases}
$$
（3）平均数：一组数相加后除以数据的个数而得到的，也称均值；集中趋势最常用的测度值；易受极端值影响。
$$
\bar X=\frac{x_1+x_2+...+x_n}{n}=\frac{\sum_{i=1}^{n}{x_i}}{n}
$$

###### 2、集中趋势的关系

![image-20221118145635424](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221118145635424.png)

###### 3、缺失值的显示方式

- 方法一：info（）查看

  <img src="C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221118145732438.png" alt="image-20221118145732438" style="zoom:50%;" />

  从图上我们就可以看到有缺失，总共50882条数据，8、9、10分别有39191、30631、30631条数据非空。

- 方法2：缺失值矩阵图：图中白色的地方就是存在缺失值的地方，从图中可以看出Cabin字段存在大量的数据缺失。

  ```python
  import matplotlib.pyplot as plt
  import missingno as mnso
  mnso.matrix(data)
  plt.show()
  ```

  

  ![image-20221118145903866](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221118145903866.png)

  明显可以看出来：

  'Health Indicator', 'Holding_policy_Duration',  'Holding_Policy_Type' 存在缺失值，我们可以再利用缺失值条形图可以直观看到每一列数据的个数和缺失值的个数。

- 方法三：缺失值条形图

  ```python
  mnso.bar(data)
  plt.show()
  ```

  ![image-20221118150327232](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221118150327232.png)

定类数据

[9个value_counts()的小技巧，提高Pandas 数据分析效率](https://cloud.tencent.com/developer/article/1877700?from=15425)

[数据预处理：标称型特征的编码和缺失值处理](https://www.jianshu.com/p/1a4539970473/)

[标称变量（Categorical Features）或者分类变量（Categorical Features）缺失值填补、详解及实战](https://blog.csdn.net/zhongkeyuanchongqing/article/details/116403216)

[每天一点sklearn之SimpleImputer（9.19）](https://zhuanlan.zhihu.com/p/83173703?from_voters_page=true)

#### Part B

运行下面的代码单元而不做任何更改。

要通过这一步，您需要在 A 部分中定义一个流水线，该流水线的 MAE 低于上面的代码。我们鼓励您在这里花点时间尝试许多不同的方法，看看您可以获得多低的 MAE！ （如果您的代码没有通过，请修改A部分的预处理步骤和模型。）

```python
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

```

```
MAE: 17553.371061643833
```

### 第 2 步：生成测试预测

现在，您将使用经过训练的模型通过测试数据生成预测。

```python
# Preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test)
```

在不做任何更改的情况下运行下一个代码单元，将您的结果保存到 CSV 文件中，该文件可以直接提交给比赛。

```python
# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
```

### 继续 

继续学习交叉验证，这是一种可以用来获得更准确的模型性能估计的技术！

## juptyer notebook常用快捷键：

1. 运行当前Cell：Ctrl + Enter  

2. 运行当前Cell并在其下方插入一个新的Cell：Alt + Enter  

3. 运行当前Cell并选中其下方的Cell：Shift + Enter  

4. 蓝色方框状态下，将Cell Type由Code转换成Markdown：M  

5. 蓝色方框状态下，将Cell Type由Markdown转换成code：Y  

6. 在当前Cell下方插入一个新的Cell：B  

7. 在当前Cell上方插入一个新的Cell：A  

8. 剪切当前的Cell：X  

9. 复制当前的Cell：C  

10. 将剪切板内容粘贴到当前Cell的下方：V  

11. 将剪切板内容粘贴到当前Cell的上方：Shift + V  

12. 删除掉当前Cell：Double D  

13. 撤销删除操作：Z

https://www.lmlphp.com/user/62524/article/item/1262456/