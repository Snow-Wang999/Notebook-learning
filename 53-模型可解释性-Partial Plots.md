# 53-Partial Plots

每个特征如何影响您的预测？

## 偏相关图（Partial Dependence Plots）

虽然特征重要性显示哪些变量对预测影响最大，但部分依赖图显示一个特征如何影响预测。 

这有助于回答以下问题： 

- 控制所有其他房屋特征，经度和纬度对房价有什么影响？重申一下，类似大小的房屋在不同地区的定价如何？ 

- 两组人之间的健康预测差异是由于饮食差异还是其他因素造成的？ 

如果您熟悉**线性或逻辑回归模型**，则可以将偏相关图解释为类似于这些**模型中的系数**。但是，复杂模型的部分依赖图可以捕获比简单模型的系数更复杂的模式。如果您不熟悉线性回归或逻辑回归，请不要担心这种比较。 

我们将展示几个示例，解释这些图的解释，然后查看创建这些图的代码。

## How it Works

与排列重要性一样，部分依赖图是在模型拟合后计算的。该模型适用于未以任何方式人为操纵的**真实数据**。 

在我们的足球示例中，球队可能在很多方面有所不同。他们的传球次数、射门次数、进球数等等。乍一看，似乎很难理清这些特征的影响。 

要了解部分图如何分离出每个特征的影响，我们首先考虑单行数据。例如，该行数据可能代表一支球队有 50% 的控球时间、传球 100 次、射门 10 次并打进 1 球。 

我们将使用拟合模型来预测我们的结果（他们的球员赢得“比赛最佳球员”的概率）。但是我们**反复更改一个变量的值**以进行一系列预测。如果球队只有 40% 的时间控球，我们就可以预测结果。然后我们预测他们有 50% 的时间控球。然后再次预测 60%。等等。当我们从控球率的小值移动到大值（在水平轴上）时，我们追踪预测结果（在垂直轴上）。 

在这个描述中，我们只使用了一行数据。特征之间的相互作用可能导致单行图不典型。因此，我们用原始数据集中的多行重复该心理实验，并在垂直轴上绘制平均预测结果。

## 代码示例 

模型构建不是我们的重点，因此我们不会专注于数据探索或模型构建代码。

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)
```

我们的第一个示例使用决策树，您可以在下面看到。在实践中，您将为实际应用程序使用更复杂的模型。

```python
from sklearn import tree
import graphviz

tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=feature_names)
graphviz.Source(tree_graph)
```

![image-20221119145958944](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221119145958944.png)

作为阅读树的指导： 

- 有孩子的叶子在顶部显示它们的分裂标准 
- 底部的一对值分别显示树的该节点中数据点的目标的假值和真值的计数。

---

补充：

Getting Started with Titanic

https://www.kaggle.com/c/titanic

random forest model

![image-20221121113615690](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121113615690.png)

---

下面是使用 [PDPBox library](https://pdpbox.readthedocs.io/en/latest/) 库创建部分依赖图的代码。

```python
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=feature_names, feature='Goal Scored')

# plot it
pdp.pdp_plot(pdp_goals, 'Goal Scored')
plt.show()
```

![image-20221121144833832](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121144833832.png)

![image-20221121144848651](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121144848651.png)

在您解释此图时，有几项值得指出 

- y 轴被解释为根据基线或最左边值的预测之间的变化。 

- 蓝色阴影区域表示置信度。 

从这个特定的图表中，我们看到进球会大大增加您赢得“比赛最佳球员”的机会。但除此之外的额外目标似乎对预测影响不大。 

这是另一个示例图：

```python
feature_to_plot = 'Distance Covered (Kms)'
pdp_dist = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=feature_names, feature=feature_to_plot)

pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()
```

![image-20221121145310695](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121145310695.png)

这张图似乎太简单了，无法代表现实。但那是因为模型太简单了。您应该能够从上面的决策树中看出，这恰好代表了模型的结构。 

您可以轻松比较不同模型的结构或含义。这是带有随机森林模型的同一图。

```python
# Build Random Forest model
rf_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

pdp_dist = pdp.pdp_isolate(model=rf_model, dataset=val_X, model_features=feature_names, feature=feature_to_plot)

pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()
```

![image-20221121145426718](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121145426718.png)

该模型认为，如果您的球员在比赛过程中总共跑了 100 公里，您更有可能赢得全场最佳。虽然运行更多会导致较低的预测。 

一般来说，这条曲线的平滑形状似乎比决策树模型的阶跃函数更合理。尽管这个数据集足够小，但我们在解释任何模型时都会非常小心。

## 2D Partial Dependence Plots

如果您对特征之间的交互感到好奇，二维部分依赖图也很有用。一个例子可以澄清这一点。 

我们将再次对该图使用决策树模型。它将创建一个极其简单的图，但您应该能够将您在图中看到的内容与树本身相匹配。

```python
# Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot
features_to_plot = ['Goal Scored', 'Distance Covered (Kms)']
inter1  =  pdp.pdp_interact(model=tree_model, dataset=val_X, model_features=feature_names, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()
```

![image-20221121145833078](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121145833078.png)

该图表显示了对进球数和覆盖距离的任意组合的预测。 

例如，当一支球队至少进了 1 个球并且他们跑的总距离接近 100 公里时，我们看到最高的预测。如果他们进了 0 个球，那么距离并不重要。你能通过跟踪目标为 0 的决策树看到这一点吗？ 

但是如果他们进球，距离会影响预测。确保您可以从 2D 部分依赖图中看到这一点。你也能在决策树中看到这种模式吗？

## Exercise: Partial Plots

### 设置

今天，您将创建部分依赖图并练习使用出租车票价预测竞赛中的数据建立洞察力。 

我们再次提供代码来进行基本加载、审查和模型构建。运行下面的单元格以设置所有内容：

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Environment Set-Up for feedback system.
from learntools.core import binder
binder.bind(globals())
from learntools.ml_explainability.ex3 import *
print("Setup Complete")

# Data manipulation code below here
data = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows=50000)

# Remove data with extreme outlier coordinates or negative fares
data = data.query('pickup_latitude > 40.7 and pickup_latitude < 40.8 and ' +
                  'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and ' +
                  'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +
                  'dropoff_longitude > -74 and dropoff_longitude < -73.9 and ' +
                  'fare_amount > 0'
                  )

y = data.fare_amount

base_features = ['pickup_longitude',
                 'pickup_latitude',
                 'dropoff_longitude',
                 'dropoff_latitude']

X = data[base_features]


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
first_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(train_X, train_y)
print("Data sample:")
data.head()
```

![image-20221121160721007](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121160721007.png)

```python
data.describe()
```

![image-20221121160746122](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121160746122.png)

### Question 1

下面是绘制pickup_longitude的部分依赖图的代码。运行以下单元格。

```python
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

feat_name = 'pickup_longitude'
pdp_dist = pdp.pdp_isolate(model=first_model, dataset=val_X, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()
```

![image-20221121160811059](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121160811059.png)

为什么部分依赖图有这种U形？ 

你的解释是否说明了其他特征在部分依赖图中的形状？ 

在下面的for循环中创建所有其他部分绘图（从上面的代码中复制相应的行）。

```python
for feat_name in base_features:
    pdp_dist = pdp.pdp_isolate(model=first_model, dataset=val_X, model_features=base_features, feature=feat_name)
    pdp.pdp_plot(pdp_dist, feat_name)
    plt.show()
```

![image-20221121161012329](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121161012329.png)

![image-20221121161024199](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121161024199.png)

![image-20221121161055439](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121161055439.png)

![image-20221121161116268](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121161116268.png)

这些形状是否符合您对其形状的期望？既然你已经看到了它们，你能解释一下形状吗？ 

从排列重要性结果中我们可以看出，距离是出租车票价最重要的决定因素。 

该模型不包括距离度量（如纬度或经度的绝对变化）作为特征，因此坐标特征（如pickup_longitude）捕捉距离的影响。在经度值的中心附近拾取会降低平均预测票价，因为这意味着（平均而言）更短的行程。 

出于同样的原因，我们在所有的部分依赖图中都看到了一般的U形。

### Question 2

现在，您将运行2D部分依赖图。作为提醒，这里是教程中的代码。

```python
inter1  =  pdp.pdp_interact(model=my_model, dataset=val_X, model_features=feature_names, features=['Goal Scored', 'Distance Covered (Kms)'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['Goal Scored', 'Distance Covered (Kms)'], plot_type='contour')
plt.show()
```

为要素pickup_longitude和dropoff_longitude创建二维绘图。适当地绘制？ 

你希望它看起来像什么？

```python
# Add your code here
fnames = ['pickup_longitude', 'dropoff_longitude']
longitudes_partial_plot  =  pdp.pdp_interact(model=first_model, dataset=val_X,
                                            model_features=base_features, features=fnames)
pdp.pdp_interact_plot(pdp_interact_out=longitudes_partial_plot,
                      feature_names=fnames, plot_type='contour')
plt.show()
```

![image-20221121162612912](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121162612912.png)

您应该期望绘图具有沿对角线延伸的轮廓。我们在某种程度上看到了这一点，尽管有一些有趣的警告。

我们期望对角线等高线，因为这是一对值，其中上下经纬度在附近，表示行程较短（控制其他因素）。

当你离中心对角线越来越远时，我们应该预计价格会随着接送经度之间的距离的增加而增加。 

令人惊讶的是，当你进一步走到这张图的右上方时，价格会上涨，甚至会停留在45度线附近。 

这可能值得进一步研究，尽管与远离45度线相比，移到图的右上方的影响很小。

### Question 3

考虑一次从-73.92经度开始，到-74经度结束的骑行。使用上一个问题中的图表，估计如果骑手在-73.98经度开始骑行，会节省多少钱？

```python
savings_from_shorter_trip = 24-9
```

大约15美元。价格从略高于24美元降至略高于9美元。

提示：首先找到与-74下降经度相对应的垂直高度。然后读取正在切换的水平值。使用白色轮廓线确定自己接近的值。你可以四舍五入到最接近的整数，而不是强调精确到最接近一便士的成本

### Question 4

在迄今为止您所看到的PDP中，位置特征主要用作捕捉行驶距离的代理。在排列重要性课程中，您添加了abs_ln_change和abs_lat_change作为距离的更直接的度量。 

在此再次创建这些功能。你只需要填写前两行。然后运行以下单元格。 

运行它之后，确定这个部分依赖图和没有绝对值特征的图之间最重要的区别。用于生成没有绝对值特征的PDP的代码位于该代码单元的顶部。

```python
# This is the PDP for pickup_longitude without the absolute difference features. Included here to help compare it to the new PDP you create
feat_name = 'pickup_longitude'
pdp_dist_original = pdp.pdp_isolate(model=first_model, dataset=val_X, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist_original, feat_name)
plt.show()
```

![image-20221121164914597](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121164914597.png)

```python
# create new features
data['abs_lon_change'] = abs(data['pickup_longitude']-data['dropoff_longitude'])
data['abs_lat_change'] = abs(data['pickup_latitude']-data['dropoff_latitude'])

features_2  = ['pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'abs_lat_change',
               'abs_lon_change']

X = data[features_2]
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)

feat_name = 'pickup_longitude'
pdp_dist = pdp.pdp_isolate(model=second_model, dataset=new_val_X, model_features=features_2, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()

# Check your answer
q_4.check()
```

![image-20221121164949859](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121164949859.png)

最大的区别是部分依赖图变得更小。在顶部图表中，最低垂直值大约比最高垂直值低15，而在刚刚创建的图表中，这一差异仅为3左右。换句话说，一旦控制了绝对行驶距离，pickup_longitude对预测的影响就非常小。

### Question 5

考虑一个只有两个预测特征的场景，我们将其称为`feat_A`和`feat_B`。这两个特征的最小值均为-1，最大值均为1。`feat_A`的部分依赖图在其整个范围内急剧增加，而特征B的部分依赖曲线在其整个区域内以较慢的速度（不那么陡峭）增加。 

这是否保证`feat_A`将具有比`feat_B`更高的排列重要性。为什么？ 

考虑过之后，取消注释下面的行以获得解决方案。

不，这不能保证`feat_A`更重要。例如，`feat_A`在其变化的情况下可能会产生很大的影响，但99%的时间内可能只有一个值。在这种情况下，置换`feat_A`无关紧要，因为大多数值都是不变的。

### Question 6

下面的代码单元执行以下操作： 

1. 创建具有范围[-2，2]内随机值的两个要素X1和X2。 
2. 创建始终为1的目标变量y。 
3. 训练`RandomForestProgressor`模型以预测给定X1和X2的y。 
4. 创建X1的PDP图和X1与y的散点图。 

你对PDP图的外观有预测吗？运行单元格以查找。 

修改y的初始化，使我们的PDP图在[-1,1]范围内具有正斜率，在其他地方具有负斜率。（注意：您应该只修改y的创建，保持X1、X2和my_model不变。）

```python
import numpy as np
from numpy.random import rand

n_samples = 20000

# Create array holding predictive feature
X1 = 4 * rand(n_samples) - 2
X2 = 4 * rand(n_samples) - 2
# Create y. you should have X1 and X2 in the expression for y
y = -2 * X1 * (X1<-1) + X1 - 2 * X1 * (X1>1) - X2

# create dataframe because pdp_isolate expects a dataFrame as an argument
my_df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
predictors_df = my_df.drop(['y'], axis=1)

my_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(predictors_df, my_df.y)

pdp_dist = pdp.pdp_isolate(model=my_model, dataset=my_df, model_features=['X1', 'X2'], feature='X1')

# visualize your results
pdp.pdp_plot(pdp_dist, 'X1')
plt.show()
```

![image-20221121170626928](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121170626928.png)

### Question 7

创建一个具有2个特征和一个目标的数据集，这样第一个特征的pdp是平坦的，但其排列重要性很高。我们将使用RandomForest作为模型。 

注意：您只需要提供创建变量X1、X2和y的行。提供了构建模型和计算细节的代码。

```python
import eli5
from eli5.sklearn import PermutationImportance

n_samples = 20000

# Create array holding predictive feature
X1 = 4 * rand(n_samples) - 2
X2 = 4 * rand(n_samples) - 2
# Create y. you should have X1 and X2 in the expression for y
y = X1*X2


# create dataframe because pdp_isolate expects a dataFrame as an argument
my_df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
predictors_df = my_df.drop(['y'], axis=1)

my_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(predictors_df, my_df.y)


pdp_dist = pdp.pdp_isolate(model=my_model, dataset=my_df, model_features=['X1', 'X2'], feature='X1')
pdp.pdp_plot(pdp_dist, 'X1')
plt.show()

perm = PermutationImportance(my_model).fit(predictors_df, my_df.y)

# Check your answer
q_7.check()

# show the weights for the permutation importance you just calculated
eli5.show_weights(perm, feature_names = ['X1', 'X2'])
```

![image-20221121171449067](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121171449067.png)

![image-20221121171457721](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221121171457721.png)

### 继续前进 

部分依赖图可能非常有趣。我们有一个讨论线程（[discussion thread](https://www.kaggle.com/learn-forum/65782) ）来讨论你想看到的现实世界的主题或问题，这些主题或问题是通过部分依赖图解决的。 

接下来，学习SHAP值（**[SHAP values](https://www.kaggle.com/dansbecker/shap-values)**）如何帮助您理解每个预测的逻辑。

补充参考资料：

[5.2 部分依赖图 (Partial Dependence Plot, PDP)](https://blog.csdn.net/weixin_43336281/article/details/119413757)