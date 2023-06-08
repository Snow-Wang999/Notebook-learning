# 45-Cross-Validation

测试模型的更好方法。

在本教程中，您将学习如何使用交叉验证来更好地衡量模型性能。

## 介绍

机器学习是一个迭代过程。 

您将面临有关使用**哪些预测变量**、使用**哪些类型的模型**、为这些模型提供**哪些参数**等的选择。到目前为止，您已经通过验证测量模型质量以数据驱动的方式做出了这些选择（或坚持）设置。 

但是这种方法有一些缺点。为了解这一点，假设您有一个包含 5000 行的数据集。您通常会保留大约 20% 的数据作为验证数据集，即 1000 行。但这在确定模型分数时留下了一些随机机会。也就是说，一个模型可能在一组 1000 行上表现良好，即使它在不同的 1000 行上不准确。 

在极端情况下，您可以想象验证集中只有一行数据。如果你比较替代模型，哪个模型对单个数据点做出最好的预测将主要取决于运气！ 

一般来说，验证集越大，我们衡量模型质量的**随机性（也称为“噪声”）**就越少，它就越可靠。不幸的是，我们只能通过从训练数据中删除行来获得大型验证集，而较小的训练数据集意味着更差的模型！

## 什么是交叉验证（cross-validation）？

在交叉验证中，我们在不同的数据子集上运行我们的建模过程以获得模型质量的多种度量。 

例如，我们可以先将数据分成 5 份，每份占整个数据集的 20%。在这种情况下，我们说我们已经将数据分成 5 个“**折叠（folds）**”。

![image-20221118160812244](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221118160812244.png)

然后，我们对每一折进行一个实验：

- 在实验 1 中，我们使用第一个折叠作为验证（或保持）集，其他所有作为训练数据。这为我们提供了基于 20% 保留集的模型质量度量。 
- 在实验 2 中，我们保留了第二次折叠的数据（并使用除第二次折叠之外的所有数据来训练模型）。然后使用 holdout 集对模型质量进行第二次估计。 
- 我们重复这个过程，使用每个折叠一次作为 holdout 集。将这些放在一起，100% 的数据在某个时候被用作保留值，我们最终得到一个基于数据集中所有行的模型质量度量（即使我们没有同时使用所有行） .

## 什么时候应该使用交叉验证？

交叉验证可以更准确地衡量模型质量，如果您要做出大量建模决策，这一点尤为重要。但是，它可能需要更长时间才能运行，因为它估计多个模型（每个折叠一个）。 

那么，考虑到这些权衡，您应该何时使用每种方法？ 

- 对于小数据集，额外的计算负担不是什么大问题，您应该运行交叉验证。 

- 对于更大的数据集，单个验证集就足够了。您的代码将运行得更快，并且您可能拥有足够的数据，几乎不需要重新使用其中的一些数据来保持。 

对于什么构成大数据集和小数据集，没有简单的阈值。但是，如果您的模型需要几分钟或更短的时间来运行，那么切换到交叉验证可能是值得的。 

或者，您可以运行交叉验证并查看每个实验的分数是否接近。如果每个实验产生相同的结果，则单个验证集可能就足够了。

## Example

我们将使用与上一教程中相同的数据。我们在 X 中加载输入数据，在 y 中加载输出数据。

```python
import pandas as pd

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price
```

然后，我们定义了一个管道，它使用一个输入器来填充缺失值和一个随机森林模型来进行预测。 

虽然可以在没有管道的情况下进行交叉验证，但这非常困难！使用管道将使代码非常简单。

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50,random_state=0))
                             ])
```

我们使用 scikit-learn 的 cross_val_score() 函数获得交叉验证分数。我们使用 cv 参数设置折叠数。

```python
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
```

```
MAE scores:
 [301628.7893587  303164.4782723  287298.331666   236061.84754543
 260383.45111427]
```

评分参数选择要报告的模型质量度量：在这种情况下，我们选择负平均绝对误差 (MAE)。 scikit-learn 的文档显示了一个选项列表。 

我们指定负 MAE 有点令人惊讶。 Scikit-learn 有一个约定，其中定义了所有指标，因此数字越大越好。在这里使用负值可以让它们与该约定保持一致，尽管负 MAE 在其他地方几乎闻所未闻。 

我们通常需要单一的模型质量度量来比较替代模型。所以我们取实验的平均值。

```python
print("Average MAE score (across experiments):")
print(scores.mean())
```

```
Average MAE score (across experiments):
277707.3795913405
```

## 结论 

使用交叉验证可以更好地衡量模型质量，还有清理代码的额外好处：请注意，我们不再需要跟踪单独的训练集和验证集。所以，特别是对于小数据集，这是一个很好的改进！

## Exercise: Cross-Validation

在本练习中，您将利用所学知识通过交叉验证调整机器学习模型。

## 设置

以下问题将为您提供有关工作的反馈。运行以下单元格以设置反馈系统。

```python
# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex5 import *
print("Setup Complete")
```

您将使用上一个练习中针对 Kaggle Learn 用户的房价竞赛。

运行下一个代码单元而不更改，以在 X 和 X_test 中加载训练和测试数据。为简单起见，我们删除了分类变量。

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
train_data = pd.read_csv('../input/train.csv', index_col='Id')
test_data = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice              
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()
```

使用下一个代码单元打印数据的前几行。

```python
X.head()
```

![image-20221118170121503](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221118170121503.png)

到目前为止，您已经了解了如何使用 scikit-learn 构建管道。例如，在使用 RandomForestRegressor() 训练随机森林模型进行预测之前，下面的管道将使用 SimpleImputer() 替换数据中的缺失值。我们使用 n_estimators 参数设置随机森林模型中的树数，设置 random_state 可确保再现性。

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])
```

您还学习了如何在交叉验证中使用管道。下面的代码使用 cross_val_score() 函数来获取平均绝对误差 (MAE)，对五个不同的折叠进行平均。回想一下，我们使用 cv 参数设置折叠数。

```python
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())
```

```
Average MAE score: 18276.410356164386
```

### 第一步：写一个有用的函数

在本练习中，您将使用交叉验证来为机器学习模型选择参数。 

首先编写一个函数 `get_score()` 来报告机器学习管道的平均（超过三个交叉验证折叠）MAE，该管道使用： 

- X 和 y 中的数据以创建折叠，

- `SimpleImputer()` （所有参数保留为默认值）替换缺失值，以及 

- `RandomForestRegressor()`（random_state=0）以适应随机森林模型。 

提供给 `get_score()` 的 `n_estimators` 参数在设置随机森林模型中的树数时使用。

```python
def get_score(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.
    
    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    # Replace this body with your own code
    my_pipeline=Pipeline(steps=[
        ('preprocessor',SimpleImputer()),
        ('model',RandomForestRegressor(n_estimators, random_state=0))
    ])
    scores = -1*cross_val_score(my_pipeline,X,y,cv=3,scoring='neg_mean_absolute_error')
    return scores.mean()
```

### 第二步：测试不同的参数值

现在，您将使用您在步骤 1 中定义的函数来评估随机森林中树木数量的八个不同值对应的模型性能：50、100、150、...、300、350、400。 将结果存储在 Python 字典 `results` 中，其中 `results[i]` 是 `get_score(i)` 返回的平均 MAE。

```python
results = {}
for i in range(1,9):
    results[50*i]=get_score(50*i)
```

```python
print(results)
```

```
{50: 18353.8393511688, 100: 18395.2151680032, 150: 18288.730020956387, 200: 18248.345889801505, 250: 18255.26922247291, 300: 18275.241922621914, 350: 18270.29183308043, 400: 18270.197974402367}
```

使用下一个单元格可视化步骤 2 的结果。运行代码而不做任何更改。

```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(list(results.keys()), list(results.values()))
plt.show()
```

![image-20221118172838590](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221118172838590.png)

### 第三步：寻找最佳参数值

根据结果，n_estimators 的哪个值似乎最适合随机森林模型？使用您的答案设置 n_estimators_best 的值。

```python
n_estimators_best = min(results,key = results.get)
```

在本练习中，您探索了一种在机器学习模型中选择合适参数的方法。 

如果您想了解有关超参数优化的更多信息，我们鼓励您从网格搜索开始，这是一种为机器学习模型确定最佳参数组合的直接方法。值得庆幸的是，scikit-learn 还包含一个内置函数 [`GridSearchCV()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)，可以使您的网格搜索代码非常高效！

https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

### Hyper-parameter optimizers

| [`model_selection.GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)(estimator, ...) | Exhaustive search over specified parameter values for an estimator. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`model_selection.HalvingGridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html#sklearn.model_selection.HalvingGridSearchCV)(...[, ...]) | Search over specified parameter values with successive halving. |
| [`model_selection.ParameterGrid`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html#sklearn.model_selection.ParameterGrid)(param_grid) | Grid of parameters with a discrete number of values for each. |
| [`model_selection.ParameterSampler`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterSampler.html#sklearn.model_selection.ParameterSampler)(...[, ...]) | Generator on parameters sampled from given distributions.    |
| [`model_selection.RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV)(...[, ...]) | Randomized search on hyper parameters.                       |
| [`model_selection.HalvingRandomSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html#sklearn.model_selection.HalvingRandomSearchCV)(...[, ...]) | Randomized search on hyper parameters.                       |

### 继续 

继续学习梯度提升，这是一种强大的技术，可以在各种数据集上获得最先进的结果。

### 补充参考文献

1. [关于交叉验证的一点事儿](https://zhuanlan.zhihu.com/p/98209649)
2. [Scikit-learn的K-fold交叉验证类ShuffleSplit、GroupShuffleSplit用法介绍](https://blog.csdn.net/hurry0808/article/details/80797969)
3. https://www.baidu.com/s?ie=UTF-8&wd=GroupShuffleSplit
