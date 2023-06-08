# 71-特征工程-What Is Feature Engineering

学习创建更好功能的步骤和原则



欢迎来到特征工程！ 

在本课程中，您将学习构建伟大的机器学习模型的最重要步骤之一：特征工程。您将学习如何： 

- 利用相互信息（*mutual information*）确定哪些特征最重要
- decompose a dataset's variation into features with *principal component analysis*

- 在几个现实问题领域（real-world problem domains）中发明新特征

- 用目标编码（*target encoding*）编码高基数类别（high-cardinality categoricals） 

- 使用k-means聚类创建分割特征（create segmentation features with *k-means clustering*）

- 用主成分分析（*principal component analysis*）将数据集的变化分解为特征（ decompose variation into features ）

实践练习形成了一个完整的笔记本，应用了所有这些技巧，向房价入门竞赛提交了材料。完成本课程后，您将有几个想法可用于进一步提高绩效。 

你准备好了吗？走吧！

## The Goal of Feature Engineering

特征工程的目标仅仅是**使您的数据更适合当前的问题**。 

考虑“表观温度（apparent temperature）”指标，如热指数和风寒。这些量试图根据空气温度、湿度和风速来测量人类感知的温度，这些都是我们可以直接测量的。你可以把表观温度看作是一种特征工程的结果，**一种试图使观测到的数据与我们真正关心的东西更加相关**的尝试：外界的真实感受！ 

您可以执行功能工程以： 

- 提高模型的预测性能 

- 减少计算或数据需求 

- 提高结果的可解释性

## A Guiding Principle of Feature Engineering

为了使特征有用，它必须**与模型能够学习的目标有关系**。例如，线性模型只能学习线性关系。因此，当使用线性模型时，您的目标是转换特征，使其与目标的关系呈线性。 

这里的关键思想是，**应用于特征的转换本质上成为模型本身的一部分**。假设你试图从一边的长度预测正方形地块的价格。将线性模型直接拟合到“长度”会产生糟糕的结果：关系不是线性的。

![image-20221122180529757](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122180529757.png)

然而，如果我们对“长度特征”求平方以获得“面积”，我们将创建一个线性关系。将“面积”添加到特征集意味着该线性模型现在可以拟合抛物线（parabola）。换句话说，对一个特征进行平方，使线性模型能够拟合平方特征（squared features）。

![image-20221122180711596](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122180711596.png)

这将向您展示为什么在特征工程中可以获得如此高的时间回报。无论您的模型无法学习到什么关系，您都可以通过转换为自己提供。在开发特征集时，考虑模型可以使用哪些信息来实现最佳性能。

## Example - Concrete Formulations（示例-混凝土配方）

为了说明这些想法，我们将看到向数据集添加一些合成特征可以如何提高随机森林模型的预测性能。 

混凝土数据集包含各种混凝土配方和最终产品的抗压强度，这是一种衡量这种混凝土能够承受多少荷载的指标。该数据集的任务是预测给定配方的混凝土抗压强度。

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv("../input/fe-course-data/concrete.csv")
df.head()
```

![image-20221122180917810](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122180917810.png)

你可以在这里看到各种混凝土中的各种成分。稍后我们将看到，添加从这些特性中派生的一些附加合成特性如何帮助模型了解它们之间的重要关系。 

我们将首先通过在未增强的数据集上训练模型来建立基线。这将帮助我们确定我们的新功能是否实际有用。 

在特征工程过程开始时，建立这样的基线是很好的做法。基准分数可以帮助你决定你的新功能是否值得保留，或者你是否应该放弃它们并尝试其他东西。

```python
X = df.copy()
y = X.pop("CompressiveStrength")

# Train and score baseline model
baseline = RandomForestRegressor(criterion="mae", random_state=0)
baseline_score = cross_val_score(
    baseline, X, y, cv=5, scoring="neg_mean_absolute_error"
)
baseline_score = -1 * baseline_score.mean()

print(f"MAE Baseline Score: {baseline_score:.4}")
```

```
MAE Baseline Score: 8.232
```

如果你曾经在家做饭，你可能知道食谱中的成分比例通常比绝对量更能预测食谱的结果。我们可能会认为，上述特征的比率是压缩强度(`CompressiveStrength`)的一个很好的预测指标。 

下面的单元格为数据集添加了三个新的比率特征。

```python
X = df.copy()
y = X.pop("CompressiveStrength")

# Create synthetic features
X["FCRatio"] = X["FineAggregate"] / X["CoarseAggregate"]
X["AggCmtRatio"] = (X["CoarseAggregate"] + X["FineAggregate"]) / X["Cement"]
X["WtrCmtRatio"] = X["Water"] / X["Cement"]

# Train and score model on dataset with additional ratio features
model = RandomForestRegressor(criterion="mae", random_state=0)
score = cross_val_score(
    model, X, y, cv=5, scoring="neg_mean_absolute_error"
)
score = -1 * score.mean()

print(f"MAE Score with Ratio Features: {score:.4}")
```

```
MAE Score with Ratio Features: 7.948
```

果然，性能提高了！这证明了这些新的比率特征向模型暴露了以前没有检测到的重要信息。

## 持续 

我们已经看到，设计新功能可以提高模型性能。但是，如何识别数据集中可能有助于组合的特征？通过相互信息发现有用的功能([**Discover useful features**](https://www.kaggle.com/ryanholbrook/mutual-information))。