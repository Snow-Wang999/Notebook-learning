# 37-Machine Learning Competitions

进入机器学习竞赛的世界，不断改进并查看您的进步。

机器学习竞赛是提高数据科学技能和衡量进步的好方法。 在下一个练习中，您将为 Kaggle Learn 用户创建并提交房价竞赛的预测。

## Exercise: Machine Learning Competitions

### 介绍

在本练习中，您将为 Kaggle 比赛创建并提交预测。然后，您可以改进您的模型（例如通过添加功能）以应用您所学到的知识并在排行榜上上升。 首先运行下面的代码单元以设置代码检查和数据集的文件路径。

```python
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex7 import *

# Set up filepaths
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
```

这是您到目前为止编写的一些代码。再次运行它。

```python
# Import helpful libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
iowa_file_path = '../input/train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

# Create X (After completing the exercise, you can return to modify this line!)
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Select columns corresponding to features, and preview the data
X = home_data[features]
X.head()

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
```

```
Validation MAE for Random Forest Model: 21,857
```

### Train a model for the competition

上面的代码单元在 train_X 和 train_y 上训练随机森林模型。 使用下面的代码单元构建随机森林模型并在所有 X 和 y 上对其进行训练。

```python
# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=1)

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X,y)
```

```
RandomForestRegressor(random_state=1)
```

现在，阅读“测试”数据文件，并应用您的模型进行预测。

```python
# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)
```

在提交之前，请运行检查以确保您的 `test_preds` 具有正确的格式。

### 生成提交

运行下面的代码单元以生成包含您的预测的 CSV 文件，您可以使用该文件提交给比赛。

```python
# Run the code to save predictions in the format used for competition scoring

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
```

### 提交比赛

要测试您的结果，您需要参加比赛（如果您还没有参加的话）。因此，请单击此链接**[this link](https://www.kaggle.com/c/home-data-for-ml-course)**.打开一个新窗口。然后点击加入比赛按钮。

接下来，按照以下说明进行操作： 

- 首先单击窗口右上角的“保存版本”按钮。这将生成一个弹出窗口。
- 确保选中 Save and Run All 选项，然后单击 Save 按钮。
- 这会在笔记本的左下角生成一个窗口。完成运行后，单击“保存版本”按钮右侧的数字。这会在屏幕右侧拉出一个版本列表。单击最新版本右侧的省略号 (...)，然后选择在查看器中打开。这会将您带入同一页面的查看模式。您需要向下滚动才能返回这些说明。 
- 单击屏幕右侧的“输出”选项卡。然后，单击您要提交的文件，然后单击“提交”按钮将您的结果提交到排行榜。

您现在已经成功提交了比赛！

如果您想继续努力提高您的表现，请选择屏幕右上角的“编辑”按钮。然后您可以更改代码并重复该过程。有很大的改进空间，您将在工作时爬上排行榜。

### 继续你的进步

有很多方法可以改进您的模型，而试验是此时学习的好方法。

改进模型的最佳方法是添加特征。要向数据添加更多功能，请重新访问第一个代码单元格，然后更改这行代码以包含更多列名称：

```python
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
```

由于缺少值或非数字数据类型等问题，某些功能会导致错误。以下是您可能想要使用的潜在列的完整列表，并且不会引发错误：

- 'MSSubClass'
- 'LotArea'
- 'OverallQual'
- 'OverallCond'
- 'YearBuilt'
- 'YearRemodAdd'
- '1stFlrSF'
- '2ndFlrSF'
- 'LowQualFinSF'
- 'GrLivArea'
- 'FullBath'
- 'HalfBath'
- 'BedroomAbvGr'
- 'KitchenAbvGr'
- 'TotRmsAbvGrd'
- 'Fireplaces'
- 'WoodDeckSF'
- 'OpenPorchSF'
- 'EnclosedPorch'
- '3SsnPorch'
- 'ScreenPorch'
- 'PoolArea'
- 'MiscVal'
- 'MoSold'
- 'YrSold'

查看列列表并考虑可能影响房价的因素。要了解有关这些功能的更多信息，请查看竞赛页面上的数据描述。 

更新上面定义特征的代码单元后，重新运行所有代码单元以评估模型并生成新的提交文件。

### 下一步是什么？ 

如上所述，如果您尝试使用某些功能来训练您的模型，它们会引发错误。中级机器学习课程将教您如何处理这些类型的功能。您还将学习使用 xgboost，这是一种比随机森林提供更高准确性的技术。 

Pandas 课程将为您提供数据处理技能，以便您在数据科学项目中快速从概念想法转变为实施。 

您还为深度学习课程做好了准备，您将在其中构建在计算机视觉任务中性能优于人类水平的模型。

## 竞赛提高

1. ### 筛选特征列

   #### 第一步：查看数据

   使用pd.read_csv读取的数据可以使用以下的代码获取相关信息：

   - `df.head()`-查看前5列
   - `df.describe()`-查看数值的mean，std，max，min等
   - `df.info()`-查看数据类型
   - `df.apply(lambda s: type(s))`

   ### 第二步：特征查看

   1. 数据转换：

   - df.astype('str')

   2. 选择列（类型）

   - 根据类型选择列-df.select_dtypes(include='int64')

   - 根据类型排除列-df.select_dtypes(exclude='int64')

     数据类型：

   - 一般类型：'bool', 'float64','int64','object'

   - 选择所有**数字**类型的列，用 `np.number` 或 `'number'`

   - 选择**字符串**类型的列，必须用 `object`，注意，这将返回**所有**数据类型为 `object` 的列

   - 选择**日期时间**类型的列，用`np.datetime64`、`'datetime'` 或 `'datetime64'`

   - 选择 **timedelta** 类型的列，用`np.timedelta64`、`'timedelta'` 或 `'timedelta64'`

   - 选择 **category** 类型类别，用 `'category'`

   - 选择 **datetimetz** 类型的列，用`'datetimetz'`或 `'datetime64[ns, tz]'`
   - **object在pandas中是代表不确定的数据类型**

   3. 去掉重复值

   ```python
   df.drop_duplicates()
   ```

   4. 寻找缺失值

   `data_obj.apply(lambda s: s.hasnans)`

   5. 替换缺失值

      `data_nobj=data_nobj.fillna(data_nobj.mean())`

   6. Pandas计算元素的数量和频率（出现的次数）

      - pandas.Series.unique():返回NumPy数组ndarray中唯一元素值的列表
      - pandas.Series.value_counts():返回唯一元素的值及其在出现的次数。
      - pandas.Series.nunique(), pandas.DataFrame.nunique():返回int，pandas.Series中唯一元素的数量。
        

   7. 统计列中元素包含某字符的次数

      https://blog.csdn.net/pdcfighting/article/details/125342097

      

      

   

2. 评估特征与目标的关系

   ```python
   #评估特征与目标之间的关系
   #画图nrows*ncols个图
   fig, axes = plt.subplots(nrows=10, ncols=4, figsize=(15, 10))
   for idx, feature in enumerate(data_nobj.columns[:-1]):
       data_nobj.plot(feature, "SalePrice", subplots=True, kind="scatter", ax=axes[idx // 4, idx % 4])#此处的4与之前的ncols是一致的
       #data_nobj.columns[:-1]是特征数据的index，SalePrice是目标数据
   ```

   

3. 评估特征与特征的关系

   让我们更严格地评估特征和目标变量之间的线性关系水平。衡量两个向量之间线性关系的一个很好的方法是 Pearson 相关性。在 pandas 中，可以使用两种数据框方法进行计算：corr 和 corrwith。 df.corr 方法计算数据帧中所有特征的相关矩阵。 df.corrwith 方法需要再提供一个数据帧作为参数，然后它将计算来自 df 的特征与该数据帧之间的成对相关性。

   ```python
   df[df.columns[:-1]].corrwith(df['cnt'])
   #df.columns[:-1]是特征数据的index，cnt是目标数据
   ```

   ![image-20221116221035308](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221116221035308.png)

   计算真实特征之间的相关性

   ```python
   df[['temp', 'atemp', 'hum', 'windspeed(mph)', 'windspeed(ms)', 'cnt']].corr()
   ```

   ![image-20221116221020504](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221116221020504.png)

4. 选择特征

   ```python
   # create X with features I choose
   X=data_nobj.iloc[:,1:37]
   X.head()
   
   # Split into validation and training data
   train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
   
   # Define a random forest model
   rf_model = RandomForestRegressor(max_leaf_nodes=500,random_state=1)
   rf_model.fit(train_X, train_y)
   rf_val_predictions = rf_model.predict(val_X)
   rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
   
   print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
   
   # To improve accuracy, create a new Random Forest model which you will train on all training data
   rf_model_on_full_data = RandomForestRegressor(random_state=1)
   
   # fit rf_model_on_full_data on all data from the training data
   rf_model_on_full_data.fit(X,y)
   
   # path to file you will use for predictions
   test_data_path = '../input/test.csv'
   
   # read test data file using pandas
   test_data = pd.read_csv(test_data_path)
   
   # create test_X which comes from test_data but includes only the columns you used for prediction.
   # The list of columns is stored in a variable called features
   test_X = test_data[list(X.columns.values)]
   test_X = test_X.fillna(test_X.mean())
   
   # make predictions which we will submit. 
   test_preds = rf_model_on_full_data.predict(test_X)
   ```

   ```python
   # Run the code to save predictions in the format used for competition scoring
   
   output = pd.DataFrame({'Id': test_data.Id,
                          'SalePrice': test_preds})
   output.to_csv('submission.csv', index=False)
   ```

   