# 55-Advanced Uses of SHAP Values

聚合SHAP值以获得更详细的模型见解

## Recap

我们首先学习了排列重要性和部分依赖图，以概述模型所学的内容。

然后，我们了解了SHAP值，以分解各个预测的组成部分。 

现在我们将扩展SHAP值，看看聚合许多SHAP值如何为排列重要性和部分相关性图提供更详细的替代方案。

## SHAP Values Review

Shap值显示给定特征对我们的预测有多大的改变（与我们在该特征的某个基线值下进行预测相比）。 

例如，考虑一个非常简单的模型：
$$
y=4*x_1+2*x_2
$$
如果`x1`取值2，而不是基线值0，那么`x1`的SHAP值将为8（从4乘以2）。

我们在实践中使用的复杂模型很难计算这些问题。但通过一些算法上的聪明，Shap值允许我们将任何预测分解为每个特征值的效果之和，从而生成如下图：

![image-20221122151843953](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122151843953.png)

除了每个预测都有很好的细分，Shap库( [Shap library](https://github.com/slundberg/shap) )还提供了Shap值组的出色可视化。我们将关注其中两个可视化。这些可视化与排列重要性和部分相关性图在概念上相似。因此，前面练习中的多个线程将在这里汇集在一起。

## 汇总图（Summary Plots）

排列重要性非常重要，因为它创建了简单的数字度量，以确定哪些特征对模型至关重要。这有助于我们轻松地对功能进行比较，您可以向非技术受众展示结果图。 

但它并不能告诉你每一项功能的重要性。如果一个特征具有中等排列重要性，则可能意味着它具有 

- 对少数预测有较大影响，但总体上没有影响，或 
- 所有预测的中等效果。 

SHAP总结图为我们提供了一个关于特征重要性以及驱动因素的鸟瞰图。我们将浏览足球数据的示例图：

![image-20221122152307281](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122152307281.png)

这幅图是由许多点组成的。每个点有三个特征： 

- 垂直位置显示其所描绘的特征 
- 颜色显示数据集中该行的特征是高还是低 
- 水平位置显示该值的影响是导致预测更高还是更低。

例如，左上角的点是一支进球很少的球队，预测值降低了0.25。 

有些事情你应该能够很容易地挑出来： 

- 该模型忽略了`Red` 和`Yellow&Red` 特征。 

- 通常“黄牌”不会影响预测，但有一种极端情况，即高值导致预测值低得多。 

- 进球得分（Goal scored）的高值导致预测较高，低值导致预测较低 

如果你找的时间足够长，这个图表中有很多信息。你将面临一些问题来测试你在练习中如何阅读它们。

## 代码中的摘要图（Summary Plots in Code）

您已经看到了加载足球/足球数据的代码：

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
```

我们使用以下代码获取所有验证数据的SHAP值。它很短，我们在评论中对此进行了解释。

```python
import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(val_X)

# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values[1], val_X)
```

![image-20221122153332750](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122153332750.png)

代码并不太复杂。但也有一些警告。 

- 绘制时，我们调用`shap_values[1]`。对于分类问题，每个可能的结果都有一个单独的SHAP值数组。在这种情况下，我们索引以获得预测“True”的SHAP值。 

- 计算SHAP值可能很慢。这在这里不是问题，因为这个数据集很小。但在运行这些程序以绘制合理大小的数据集时，您需要小心。例外情况是使用`xgboost`模型，SHAP对其进行了一些优化，因此速度更快。

这为模型提供了一个很好的概述，但我们可能需要深入研究一个特性。这就是SHAP依赖贡献图发挥作用的地方。

## SHAP Dependence Contribution Plots

我们以前使用过部分依赖图来显示单个特征如何影响预测。这些都是有洞察力的，并且与许多真实世界的用例相关。此外，只要稍加努力，就可以向非技术观众解释它们。 

但有很多他们没有表现出来。例如，影响的分布是什么？具有某个值的效果是相当恒定的，还是取决于其他特征的值而变化很大。SHAP相关性贡献图提供了与PDP类似的见解，但它们增加了更多细节。

![image-20221122154041006](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122154041006.png)

从关注形状开始，我们将在一分钟内返回到颜色。每个点代表一行数据。水平位置是数据集的实际值，垂直位置显示该值对预测的影响。这个向上倾斜的事实表明，你掌握的球越多，模型对赢得最佳比赛奖的预测就越高。 

该分布表明，其他特征必须与控球（Ball Possession %）交互。例如，这里我们强调了两个控球值相似的点。该值导致一个预测增加，另一个预测减少。

<img src="C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122154422104.png" alt="image-20221122154422104" style="zoom:33%;" />

为了进行比较，简单的线性回归将产生完美的直线图，而没有这种扩散。 

这意味着我们要深入研究交互作用，而图中包括颜色编码来帮助实现这一点。当主要趋势是向上时，您可以直观地检查这是否随点颜色而变化。 

考虑以下非常窄的具体示例。

![image-20221122154535532](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122154535532.png)

这两点在空间上突出，远离上升趋势。他们都是紫色的，表示球队进了一球。你可以这样解释：**总的来说，拥有球会增加球队让球员获得奖项的机会。但如果他们只进了一球，这种趋势就会逆转，如果他们进得太少，颁奖评委可能会惩罚他们因为他们有太多的球。 **

除了这些少数异常值之外，颜色所表示的交互作用在这里并不是很显著。但有时它会跳出来攻击你。

## Dependence Contribution Plots in Code

我们用下面的代码得到了相关性贡献图。与`summary_plot`不同的唯一一行是最后一行。

```python
import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# calculate shap values. This is what we will plot.
shap_values = explainer.shap_values(X)

# make plot.
shap.dependence_plot('Ball Possession %', shap_values[1], X, interaction_index="Goal Scored")
```

从3.3开始，同时传递参数norm和vmin/vmax已被弃用，在两个小版本之后将成为一个错误。创建vmin/vmax时，请将其直接传递给norm。

![image-20221122155017794](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122155017794.png)

如果您没有为`interaction_index`提供参数，Shapley会使用一些逻辑来选择一个可能有趣的参数。 

这不需要编写大量代码。但这些技术的诀窍在于批判性地思考结果，而不是编写代码本身。

## 轮到你了 

用一些问题来测试自己，以提高这些技巧的技能。

## Exercise: Advanced Uses of SHAP Values

### 设置 

我们再次提供了进行基本加载、检查和模型构建的代码。运行下面的单元格以设置所有内容：

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap

# Environment Set-Up for feedback system.
from learntools.core import binder
binder.bind(globals())
from learntools.ml_explainability.ex5 import *
print("Setup Complete")


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/hospital-readmissions/train.csv')
y = data.readmitted
base_features = ['number_inpatient', 'num_medications', 'number_diagnoses', 'num_lab_procedures', 
                 'num_procedures', 'time_in_hospital', 'number_outpatient', 'number_emergency', 
                 'gender_Female', 'payer_code_?', 'medical_specialty_?', 'diag_1_428', 'diag_1_414', 
                 'diabetesMed_Yes', 'A1Cresult_None']

# Some versions of shap package error when mixing bools and numerics
X = data[base_features].astype(float)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# For speed, we will calculate shap values on smaller subset of the validation data
small_val_X = val_X.iloc[:150]
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
```

```python
data.describe()
```

![image-20221122160259055](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122160259055.png)

![image-20221122160328903](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122160328903.png)

前几个问题需要检查每个功能的效果分布，而不仅仅是每个功能的平均效果。运行以下单元格以获取再次入院`shap_value`的总结图。运行大约需要20秒。

```python
explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(small_val_X)

shap.summary_plot(shap_values[1], small_val_X)
```

![image-20221122160408627](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122160408627.png)

### Question 1

以下哪个特征对预测的影响范围更大（即最积极和最消极影响之间的差异更大） 

- diag_1_428或 
- payer_code_？

```python
# set following variable to 'diag_1_428' or 'payer_code_?'
#diag_1_428的范围更广，主要是由于最右边的几个点
feature_with_bigger_range_of_effects = 'diag_1_428'
```

### Question 2

您是否认为效果大小的范围（最小效果和最大效果之间的距离）很好地表明了哪个特征将具有更高的排列重要性？为什么？ 

如果**影响大小的范围**衡量的是与**排列重要性**不同的东西：那么这是对“在讨论人群中的再入院风险时，模型认为这两个特征中的哪一个对我们更重要？”这一问题的更好答案 

答：

不。效果范围的宽度（The width of the effects range）不是排列重要性的合理近似值。就这一点而言，范围的宽度并不能很好地反映出任何直观的“重要性”，因为它可以**由几个异常值来确定**。然而，如果**图上的所有点彼此广泛分散**，这是排列重要性很高的合理指示。因为影响范围对异常值非常敏感，所以特征重要性是一个通常更好的衡量模型重要性的指标。

### Question 3

`diag_1_428`和`payer_code_？`是二进制变量，取值为0或1。 从图表中，您认为哪一项通常会对预测的再入院风险产生更大的影响： 

将`diag_1_428`从0更改为1 

更改`payer_code_？`从0到1 

为了节省您的滚动，我们在下面包含了一个单元格，以再次绘制图形（这一个快速运行）。

```python
small_val_X_1 = small_val_X.copy()
small_val_X_1['diag_1_428'] = small_val_X_1[['diag_1_428']].replace({0.:1.})
small_val_X_1['payer_code_?'] = small_val_X_1[['payer_code_?']].replace({0.:1.})
small_val_X_1.head()
```

![image-20221122164010096](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122164010096.png)

```python
shap.summary_plot(shap_values[1], small_val_X_1)
```

![image-20221122164055112](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122164055112.png)

与原图的对比：

![image-20221122160408627](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122160408627.png)

```python
# Set following var to "diag_1_428" if changing it to 1 has bigger effect.  Else set it to 'payer_code_?'
bigger_effect_when_changed = "diag_1_428"
```

尽管`diag_1_428`的大多数SHAP值都很小，但少数粉红点（变量的高值，对应于该诊断的人）具有较大的SHAP值。换言之，这个变量的粉色点远远不是0，让某人拥有更高的（粉色）值会显著增加他们的再入院风险。在现实世界中，这种诊断是罕见的，但对患有这种疾病的人来说风险更大。相比之下，`payer_code_？`具有许多蓝色和粉色的值，并且两者的SHAP值都与0有意义的不同。

但是更改`payer_code_？`从0（蓝色）到1（粉色）的影响可能小于更改`diag_1_428`。

### Question 4

一些特征（如`number_inpatient`）在蓝色和粉色点之间有相当清晰的分隔。其他变量（如`num_lab_procedures`）的蓝色和粉色圆点混杂在一起，即使SHAP值（或对预测的影响）不都是0。 

你认为你从`num_lab_procedures`的蓝色和粉色圆点混杂在一起这一事实中学到了什么？

答：

混淆表明，有时增加该特征会导致更高的预测，而其他时候则会导致更低的预测。换句话说，特征的高值和低值都会对预测产生积极和消极的影响。对于这种“混杂”效应，最可能的解释是**变量**（在本例中为`num_lab_procedures`）**与其他变量具有交互作用**。例如，可能有一些诊断需要很多实验室程序，而其他诊断则表明风险增加。我们还不知道还有什么其他特性与`num_lab_procedures`交互，尽管我们可以用SHAP贡献依赖图来研究这一点。

### Question 5

考虑以下SHAP贡献依赖图。 

x轴显示`feature_of_interest`，点基于`other_feature`着色。

![image-20221122165025986](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122165025986.png)

`feature_of_interest`和`other_feature`之间是否存在交互作用？如果是，当`other_feature`较高或较低时，`feature_of_interest`是否对预测有更积极的影响？ 

准备好回答后，运行以下代码。

答：

首先，回想一下SHAP值是对给定特征对预测的影响的估计。因此，如果点从左上角到右下角，这意味着`feature_of_interest`的低值会导致更高的预测。

返回此图：

`feature_of_interest`对于`other_feature`的高值向下倾斜。要看到这一点，请将目光集中在粉色点（`other_feature`较高的地方），并想象通过这些粉色点的最佳拟合线。它向下倾斜，表明预测随着`feature_of_interest`的增加而下降。 

现在，把你的目光集中在蓝色的点上，想象一条穿过这些点的最佳直线。它通常非常平坦，甚至可能在图的右侧向上弯曲。

因此，**当`other_feature`较高时，增加`feature_of_interest`对预测有更积极的影响。**

### Question 6

通过运行以下单元格查看再入院数据的汇总图：

```py
shap.summary_plot(shap_values[1], small_val_X)
```

![image-20221122160408627](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122160408627.png)

`num_medicinations`和`num_lab_procedures`共享粉色和蓝色圆点的混合。 

除了`num_medications`具有更大的影响（更积极和更消极）之外，很难看出这两个特征如何影响再入院风险。为每个变量创建SHAP相关性贡献图，并描述您认为这两个变量如何影响预测的不同之处。 

作为提醒，这里是您以前看到的创建此类绘图的代码。 

```python
shap.dependence_plot(feature_of_interest, shap_values[1], val_X)
```

请记住，您的验证数据名为`small_val_X`。



粗略地说，`num_lab_procedures`看起来像是一个没有什么不可识别模式的云。它在任何点上都不会陡峭地向上或向下倾斜。很难说我们从那个情节中学到了很多东西。同时，这些值并非都非常接近0。因此，模型似乎认为这是一个相关的特性。一个潜在的下一步将是通过使用不同的其他功能来搜索交互来探索更多。 

```python
shap.dependence_plot('num_lab_procedures', shap_values[1], small_val_X)
```

![image-20221122170954779](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122170954779.png)

```python
shap.dependence_plot('num_medications', shap_values[1], small_val_X)
```

![image-20221122171012771](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122171012771.png)

另一方面，`num_medications`明显向上倾斜，直到值大约为20，然后又向下倾斜。如果没有更多的医学背景，这似乎是一个令人惊讶的现象……你可以做一些探索，看看这些患者是否对其他特征也有不寻常的价值。但下一步很好的办法是与领域专家（在本例中是医生）讨论这一现象。

```python
shap.dependence_plot('num_medications', shap_values[1], small_val_X, interaction_index="num_lab_procedures")
```

![image-20221122171041177](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122171041177.png)



### 祝贺

就是这样！机器学习模型不应该再像黑匣子，因为你有工具来检查它们，了解它们对世界的了解。 

这是调试模型、建立信任和学习洞察力以做出更好决策的绝佳技能。这些技术彻底改变了我做数据科学的方式，我希望他们也能为你做同样的事情。 

真正的数据科学需要探索。我希望你能找到一个有趣的数据集来尝试这些技术（Kaggle有很多免费的数据集可以尝试）。如果你了解到世界上一些有趣的事情，请在这个论坛上分享你的工作。我很高兴看到你用你的新技能做了什么。