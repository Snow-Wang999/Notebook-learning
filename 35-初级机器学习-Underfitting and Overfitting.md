# 35-Underfitting and Overfitting

微调您的模型以获得更好的性能。

在这一步结束时，您将了解欠拟合和过拟合的概念，并且您将能够应用这些想法来使您的模型更加准确。

## Experimenting With Different Models-尝试不同的模型

现在您有了一种可靠的方法来衡量模型的准确性，您可以尝试替代模型并查看哪个模型的预测效果最好。但是对于模型你有什么选择呢？

您可以在 scikit-learn 的文档【[documentation](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)】中看到**决策树模型**有很多选项（比您想要或长时间需要的更多）。最重要的选项决定了树的深度。回想一下本课程的第一课【[the first lesson in this course](https://www.kaggle.com/dansbecker/how-models-work)】，树的深度是衡量在进行预测之前进行了多少分裂的度量。这是一棵相对较浅的树。

![image-20221115214017596](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221115214017596.png)

在实践中，一棵树在顶层（所有房屋）和一片叶子之间有 10 个分裂并不少见。随着树越来越深，数据集被分割成房子更少的叶子。如果一棵树只有 1 个拆分，它将数据分成 2 组。如果每组再次拆分，我们将得到 4 组房屋。再次拆分每个将创建 8 个组。如果我们通过在每个级别添加更多拆分来使组数不断增加一倍，那么到第 10 级时，我们将拥有 $2^{10}$ 组房屋。那是1024片叶子。

当我们将房屋划分为许多叶子时，每片叶子中的房屋也会减少。房子很少的叶子会做出非常接近这些房子的实际价值的预测，但他们可能对新数据做出非常不可靠的预测（因为每个预测都只基于少数房子）。

[^叶子中的房屋]: 房屋总量不变，叶子的数量*叶子中的房屋数=总房屋数，所以随着叶子增多时，叶子中的房屋数减少了。

这是一种称为**过度拟合**的现象，其中模型几乎完美匹配训练数据，但在验证和其他新数据方面表现不佳。另一方面，如果我们把树做得很浅，它就不会把房子分成非常不同的组。

在极端情况下，如果一棵树只把房子分成 2 或 4 组，每组仍然有各种各样的房子。对于大多数房屋，即使在训练数据中，结果预测也可能相去甚远（出于同样的原因，在验证中也会很糟糕）。当模型无法捕获数据中的重要区别和模式时，即使在训练数据中也表现不佳，这称为**欠拟合**。

由于我们关心根据**验证数据**估计的新数据的准确性，因此我们希望找到欠拟合和过拟合之间的最佳点。在视觉上，我们想要下图中（红色）验证曲线的低点。

![image-20221115215026453](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221115215026453.png)

## Example

有几种控制树深度的替代方法，并且许多方法允许通过树的某些路径比其他路径具有更大的深度。但是 `max_leaf_nodes` 参数提供了一种非常明智的方法来控制过拟合和欠拟合。我们允许模型制作的叶子越多，我们从上图中的欠拟合区域移动到过拟合区域的次数就越多。

我们可以使用实用函数来帮助比较不同 max_leaf_nodes 值的 MAE 分数：

```python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```

使用您已经看过（并且已经编写）的代码将数据加载到 train_X、val_X、train_y 和 val_y 中。

```python
# Data Loading Code Runs At This Point
import pandas as pd
    
# Load data
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
# Filter rows with missing values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
```

我们可以使用 for 循环来比较使用不同的 max_leaf_nodes 值构建的模型的准确性。

```python
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
```

![image-20221115215620964](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221115215620964.png)

在列出的选项中，500 是最佳叶数。

## 结论 

这是外卖：模型可能会受到以下任一问题的影响： 

- 过度拟合：捕获未来不会重现的虚假模式，导致预测不准确，或者 
- 欠拟合：未能捕获相关模式，再次导致预测不准确。 

我们使用未在模型训练中使用的验证数据来衡量候选模型的准确性。这让我们可以尝试许多候选模型并保留最好的模型。

## Exercise: Underfitting and Overfitting

### 回顾

您已经构建了第一个模型，现在是时候优化树的大小以做出更好的预测了。运行此单元格以在上一步停止的位置设置您的编码环境。

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
print("Validation MAE: {:,.0f}".format(val_mae))

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex5 import *
print("\nSetup complete")
```

### Exercises

您可以自己编写函数 get_mae。现在，我们将提供它。这与您在上一课中读到的功能相同。只需运行下面的单元格。

```python
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```

第 1 步：比较不同树的大小

编写一个循环，从一组可能的值中尝试 max_leaf_nodes 的以下值。 对 max_leaf_nodes 的每个值调用 `get_mae` 函数。以某种方式存储输出，使您可以选择 max_leaf_nodes 的值，从而为您的数据提供最准确的模型。

```python
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size,train_X,val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
print(scores)
# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = min(scores,key=scores.get)
print(best_tree_size)
```

```
{5: 35044.51299744237, 25: 29016.41319191076, 50: 27405.930473214907, 100: 27282.50803885739, 250: 27893.822225701646, 500: 29454.18598068598}
100
```

第 2 步：使用所有数据拟合模型

你知道最好的树大小。如果您打算在实践中部署此模型，您可以通过使用所有数据并保持该树大小来使其更加准确。也就是说，既然您已经做出了所有的建模决策，您就不需要保留验证数据。

```python
# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)
```

您已调整此模型并改进了结果。但我们仍在使用决策树模型，根据现代机器学习标准，这些模型并不是很复杂。在下一步中，您将学习使用随机森林来进一步改进您的模型。