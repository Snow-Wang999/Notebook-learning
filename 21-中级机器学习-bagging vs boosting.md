# 07-bagging vs boosting

[集成学习（Bagging、Boosting、Stacking）](https://blog.csdn.net/Shingle_/article/details/81953564)

[机器学习中的Bagging](https://blog.csdn.net/qq_39197555/article/details/115366849)

[**集成学习算法总结----Boosting和Bagging**](https://blog.51cto.com/sddai/3073302)

[Bagging与方差 ](https://www.cnblogs.com/massquantity/p/9029611.html)

在这篇文章中，我将讨论 Bagging 和 Boosting。这些（Bagging 和 Boosting）是全世界数据科学家常用的术语。但是这些术语究竟是什么意思以及它如何帮助数据科学家。在这个内核中，我们将了解 bagging 和 boosting 以及它们在实践中的使用方式。

![image-20221101154915182](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101154915182.png)

![image-20221101154855260](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101154855260.png)

## 1. Introduction to Ensemble Learning-集成学习

- Bagging 和 boosting 通过构建一组弱估计器的组合来替代一个单一的强估计器。
- **Bagging 有助于减少模型的方差**。
- **boosting有助于减少模型的偏差**。
- 这些方法旨在**提高机器学习算法的稳定性和准确性**。

> - Bagging 和 boosting 都是机器学习中的集成学习方法。
>
> - Bagging 和 boosting 的相似之处在于它们都是集成技术，将一组弱学习器组合起来创建一个比单个学习器获得更好性能的强学习器。
>
> - 集成学习通过**组合多个模型**来帮助提高机器学习模型的性能。与单个模型相比，这种方法可以产生更好的预测性能。
>
> - 集成学习背后的基本思想是**学习一组分类器（专家）并允许他们投票**。机器学习的这种多样化是通过一种称为集成学习的技术实现的。这里的想法是训练多个模型，每个模型的目标是预测或分类一组结果。
>
> - Bagging 和 boosting 是两种类型的集成学习技术。这两个**减少了单个估计的方差**，因为它们结合了来自不同模型的多个估计。所以结果可能是一个稳定性更高的模型。
>
> - 学习中产生错误的主要原因是噪声、偏差和方差。 Ensemble 有助于最大限度地减少这些因素。通过使用集成方法，我们能够提高最终模型的稳定性并减少前面提到的错误。
>
> - **Bagging 有助于减少模型的方差**。
>
> - **boosting有助于减少模型的偏差**。
>
> - 这些方法旨在**提高机器学习算法的稳定性和准确性**。多个分类器的组合减少了方差，特别是在分类器不稳定的情况下，并且可能产生比单个分类器更可靠的分类。
>
> - 要使用 Bagging 或 Boosting，您必须选择基础学习器算法。例如，如果我们选择一个分类树，Bagging 和 Boosting 将由一个我们想要的树池组成，如下图所示：
>
>   ![image-20221101000414499](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101000414499.png)

在了解 bagging 和 boosting 以及这两种算法如何选择不同的分类器之前，我们需要先了解 Bootstrapping。

## 2. Bootstrapping

- Bootstrap 是指有放回的随机抽样。 Bootstrap 使我们能够更好地理解数据集的偏差和方差。

- 因此，Bootstrapping 是一种采样技术，我们从原始数据集中创建带有替换的观察子集。子集的大小与原始集的大小相同。

- Bootstrap 涉及从数据集中随机抽样一小部分数据。这个子集可以被替换。

- 数据集中所有样本的选择概率相等。这种方法可以帮助更好地理解数据集的均值和标准偏差。

- 假设我们有一个包含“n”个值 (x) 的样本，并且我们想要估计样本的平均值。我们可以计算如下：
  $$
  mean(x) = 1/n * sum(x)
  $$

- Bootstrapping可以用图表表示如下：

  ![image-20221101000916276](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101000916276.png)

  现在，我们将注意力转向 bagging 和 boosting。

## 3. Bagging

- Bagging（或 Bootstrap Aggregation）是一种简单且非常强大的集成方法。 Bagging 是将 Bootstrap 过程应用于高方差机器学习算法，通常是决策树。

- bagging 背后的想法是结合多个模型（例如，所有决策树）的结果来获得通用结果。现在，自举进入画面。

- Bagging（或 Bootstrap Aggregating）技术使用这些子集（包）来获得分布（完整集）的公平概念。为 bagging 创建的子集的大小可能小于原始集。

- 它可以表示如下：

  ![image-20221101001603256](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101001603256.png)

Bagging works as follows:

1. 从原始数据集中创建多个子集，选择替换观察。 
2. 在每个子集上创建一个基本模型（弱模型）。 
3. 这些模型并行运行并且彼此独立。 
4. 最终的预测是通过结合所有模型的预测来确定的。

现在，bagging 可以用图表表示如下：

![image-20221101001744827](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101001744827.png)

## **4. Boosting** 

- Boosting 是一个**顺序过程**，每个后续模型都尝试纠正前一个模型的错误。后续模型依赖于先前模型。 

- 在这种技术中，学习者是按顺序学习的，早期学习者将简单的模型拟合到数据中，然后分析数据中的错误。换句话说，我们拟合连续的树（随机样本），并且在每一步，目标都是解决先前树的净误差。 

- 当输入被假设错误分类时，其权重会增加，以便下一个假设更有可能对其进行正确分类。通过在最后组合整个集合将弱学习器转换为性能更好的模型。

- 让我们在以下步骤中了解boosting的工作方式。

  1. 从原始数据集创建一个子集。 

  2. 最初，所有数据点都被赋予相同的权重。 

  3. 在这个子集上创建一个基本模型。 

  4. 该模型用于对整个数据集进行预测。

     ![image-20221101002100451](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101002100451.png)

  1. 使用实际值和预测值计算误差。 

  2. 被错误预测的观测值被赋予更高的权重。 （这里，三个错误分类的蓝加点将被赋予更高的权重） 

  3. 创建另一个模型并对数据集进行预测。 （这个模型试图纠正以前模型的错误）

     ![image-20221101002217152](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101002217152.png)

  1. 类似地，创建了多个模型，每个模型都纠正了前一个模型的错误。 

  2. 最终模型（强学习器）是所有模型（弱学习器）的加权平均值。

     ![image-20221101002314706](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101002314706.png) 

     - 因此，boosting 算法将多个弱学习器组合成一个强学习器。 

     - 单个模型在整个数据集上表现不佳，但它们在数据集的某些部分上运行良好。 

     - 因此，每个模型实际上都提高了集成的性能。

       ![image-20221101002437825](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101002437825.png)

### **boosting的类型：**

- AdaBoost

  [AdaBoost算法——理论与sklearn代码实现](https://zhuanlan.zhihu.com/p/422401906)

- GBDT

- XGBoot

- LightGBM

## **5. Getting N learners for Bagging and Boosting**

- Bagging 和 Boosting 通过在训练阶段生成额外的数据来获得 N 个学习者。 

- N个新的训练数据集是通过随机抽样产生的，并从原始数据集进行替换。 

- 通过替换抽样，可以在每个新的训练数据集中重复一些观察。 

- 在 Bagging 的情况下，任何元素都有相同的概率出现在新的数据集中。 

- 然而，对于 Boosting，观察是加权的，因此其中一些将更频繁地参与新的集合。

-  这些多组用于训练相同的学习器算法，因此产生不同的分类器。

-  这用图表表示如下：

  ![image-20221101002608609](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101002608609.png)

## 6. Weighted data elements-加权数据元素

- 现在，我们知道了这两种方法的主要区别。 

- 虽然 Bagging 的训练阶段是并行的（即每个模型都是独立构建的），但 Boosting 以如下顺序构建新学习器：

  ![image-20221101002741643](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101002741643.png)

- 在 Boosting 算法中，每个分类器都根据数据进行训练，同时考虑到之前分类器的成功。 

- 在每个训练步骤之后，重新分配权重。**错误分类的数据会增加其权重以强调最困难的情况。**

-  这样，后续的学习者在培训过程中就会专注于他们。

## 7. Classification stage in action

- 为了预测新数据的类别，我们只需要将 N 个学习器应用于新的观察。 

- 在 Bagging 中，结果是通过平均 N 个学习者的响应（或多数投票）获得的。 

- 然而，Boosting 为 N 个分类器分配了第二组权重，以便对其估计值进行加权平均。 

- 如下图所示：

  ![image-20221101002958301](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101002958301.png)

- 在 Boosting 训练阶段，算法为每个结果模型分配权重。

- 在训练数据上具有良好分类结果的学习器将被分配比较差的学习器更高的权重。 

- 因此，在评估新学习者时，Boosting 也需要跟踪学习者的错误。 

- 让我们看看过程中的差异：

  ![image-20221101003059646](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101003059646.png)

- 一些 Boosting 技术包括保留或丢弃单个学习器的额外条件。 

- 例如，在最著名的 AdaBoost 中，维持模型需要小于 50% 的误差；否则，重复迭代直到获得比随机猜测更好的学习器。 

- 上图显示了 Boosting 方法的一般过程，但存在几种替代方法，它们具有不同的方法来确定在下一个训练步骤和分类阶段使用的权重。

## 8. Selecting the best technique- Bagging or Boosting-选择最佳技术

+ 现在，我们可能会想到一个问题——对于特定问题是选择 Bagging 还是 Boosting。 
+ 这取决于数据、模拟和环境。 
+ Bagging 和 Boosting 减少了单个估计的方差，因为它们结合了来自不同模型的多个估计。所以结果可能是一个稳定性更高的模型。 
+ 如果问题是单个模型的性能非常低，Bagging 很少会得到更好的偏差。但是，Boosting 可以生成具有较低错误的组合模型，因为它优化了单个模型的优势并减少了缺陷。 
+ 相比之下，如果单个模型的难度是过拟合，那么 Bagging 是最好的选择。就其本身而言，boosting无助于避免过度拟合。 
+ 事实上，这项技术本身就面临着这个问题。出于这个原因，Bagging 比 Boosting 更有效。

## 9. Similarities between Bagging and Boosting

Bagging 和 Boosting 的相似之处如下：

1. 两者都是从 1 个学习者中获取 N 个学习者的集成方法。 
2. 两者都通过随机抽样生成几个训练数据集。 
3. 两者都通过平均 N 个学习者（或取其中的大多数，即多数投票）来做出最终决定。 
4. 两者都擅长减少方差并提供更高的稳定性。

## 10. Differences between Bagging and Boosting

Bagging和Boosting的区别如下：

1. Bagging 是组合属于同一类型的预测的最简单方法，而 Boosting 是一种组合属于不同类型的预测的方法。 
2. Bagging 旨在减少方差，而不是偏差，而 Boosting 旨在减少偏差，而不是方差。
3. 在 Baggiing 中，每个模型都获得相同的权重，而在 Boosting 中，模型根据它们的性能进行加权。 
4. 在 Bagging 中，每个模型都是独立构建的，而在 Boosting 中，新模型受先前构建模型的性能影响。 
5. 在 Bagging 中，从整个训练数据集中随机抽取不同的训练数据子集进行替换。在 Boosting 中，每个新子集都包含以前模型错误分类的元素。 
6. Bagging 试图解决过拟合问题，而 Boosting 试图减少偏差。 
7. 如果分类器不稳定（高方差），那么我们应该应用 Bagging。如果分类器稳定且简单（高偏差），那么我们应该应用 Boosting。 
8. Bagging 扩展到随机森林模型，而 Boosting 扩展到 Gradient Boosting。

## 11. Summary and Conclusion

- 在这篇文章中，我们讨论了两种非常重要的集成学习技术——Bagging 和 Boosting。 
- 我们已经详细讨论了 Bootstrapping、Bagging 和 Boosting。 
- 我们已经讨论了行动中的分类阶段。 
- 然后，我们展示了如何为特定问题选择最佳技术 - Bagging 或 Boosting。 
- 最后，我们讨论了 Bagging 和 Boosting 之间的异同。 
- 我希望这篇文章能让你对 Bagging 和 Boosting 有一个深入的了解。

## 12. References

这些想法、概念和图表取自以下网站：

- https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/
- https://medium.com/swlh/difference-between-bagging-and-boosting-f996253acd22
- https://www.geeksforgeeks.org/comparison-b-w-bagging-and-boosting-data-mining/
- https://hub.packtpub.com/ensemble-methods-optimize-machine-learning-models/
- https://towardsdatascience.com/decision-tree-ensembles-bagging-and-boosting-266a8ba60fd9

---

## Exercise：bagging and boosting

### load library

```python
# Setup plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

# Setup datatable
import datatable as dt1
print(dt1.__version__)
from datatable import (dt, f, by, ifelse, update, sort,
                       count, min, max, mean, sum, rowsum,isna)
# Setup data preprocessing
import numpy as np
import pandas as pd
```

### load data

```python
col_diabetes2=['f1','f2','f3','f4','f5','f6','f7','f8','target_0','target_1']
df_diabetes2 = dt1.fread('../input/diabetes/diabetes2.dt',columns=col_diabetes2)
df_diabetes2
```

![image-20221101022129552](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101022129552.png)

### select data

```python
df_diabetes2[0:7,:]1
```

![image-20221101022214755](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101022214755.png)

```python
df_diabetes2_1=df_diabetes2 .copy()
del df_diabetes2_1[0:7,:]
df_diabetes2_1
```

![image-20221101022243386](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101022243386.png)

```python
df_diabetes2_2 = df_diabetes2_1[:, dt.as_type(f['f1','f2','f3','f4','f5','f6','f7','f8'], dt.Type.float32).extend(f.target_0)]
df_diabetes2_2
```

![image-20221101022309471](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101022309471.png)

```python
df_diabetes2_2.shape
# (768, 9)
```

```python
df_diabetes2_2.ltypes
'''
(ltype.real,
 ltype.real,
 ltype.real,
 ltype.real,
 ltype.real,
 ltype.real,
 ltype.real,
 ltype.real,
 ltype.bool)
'''
```

```
 # count un-None data, check if has None
df_diabetes2_2[:, dt.count(f[:])]
```

![image-20221101022524721](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101022524721.png)

```python
X = df_diabetes2_2[:,0:-1]
X
```

![image-20221101022546943](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101022546943.png)

```python
X[:, dt.mean(f[:])]
X[:, dt.max(f[:])]
X[:, dt.min(f[:])]
```

![image-20221101022644684](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101022644684.png)

![image-20221101022651491](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101022651491.png)

![image-20221101022659816](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101022659816.png)

```python
# 将0/1取值映射为-1/1取值
y = df_diabetes2_2[:,f.target_0 *2-1]
y
```

![image-20221101022725751](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101022725751.png)

### Preprocessing

```python
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector

X = X.to_pandas()
y = y.to_pandas()

preprocessor = make_column_transformer(
    (SimpleImputer(missing_values=0, strategy="mean", copy=False),
    make_column_selector(dtype_include=np.number)),
)

X = preprocessor.fit_transform(X)
X
```

```
array([[0.352941  , 0.83      , 0.606557  , ..., 0.396423  , 0.0964987 ,
        0.75      ],
       [0.352941  , 0.77      , 0.606557  , ..., 0.436662  , 0.324936  ,
        0.3       ],
       [0.117647  , 0.46      , 0.622951  , ..., 0.360656  , 0.691716  ,
        0.116667  ],
       ...,
       [0.529412  , 0.61      , 0.459016  , ..., 0.496274  , 0.442357  ,
        0.2       ],
       [0.0588235 , 0.755     , 0.491803  , ..., 0.388972  , 0.0431255 ,
        0.0166667 ],
       [0.352941  , 0.435     , 0.655738  , ..., 0.345753  , 0.00256191,
        0.183333  ]], dtype=float32)
```

### split the data

```python
from sklearn.model_selection import train_test_split
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

print('Size of training characteristics:',X_train.shape)
print('Size of training tag:',y_train.shape)
print('Size of testing characteristics:',X_test.shape)
print('Size of testing label:',y_test.shape)
```

### bagging

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score  
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV
import time
```

```python
# k-near neighbors classifier as base learner（基学习器）
knc = KNeighborsClassifier()
bagging_knc = BaggingClassifier(base_estimator = knc)
# decision tree as base learner（基学习器）
dtree = DecisionTreeClassifier()
bagging_tr = BaggingClassifier(base_estimator=dtree)
# logistic regression as base learner（基学习器）
lr = LogisticRegression()
bagging_lr = BaggingClassifier(base_estimator=lr)
```

#### 1. k-nearest neighbor algorithm

```python
#from sklearn.model_selection import RandomizedSearchCV
params = {
    'n_estimators':[3,5,7,11,13,15,17,19,21,25,30,40,45,50],#要集成的基估计器的个数。
    #'n_estimators':[2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97], #要集成的基估计器的个数。
    'max_samples':[0.5,1.0], #决定从x_train抽取去训练基估计器的样本数量,
    'max_features':[0.5,1.0], #决定从x_train抽取去训练基估计器的特征数量
    'oob_score':[True, False], #决定是否使用包外估计（out of bag estimate）泛化误差
    'base_estimator__n_neighbors':[3, 5, 7, 9, 11] #knn的近邻个数
    #'base_estimator__n_neighbors':[2,3,5,7,11,13,17,19,23,29,31,37] #knn的近邻个数
}
rs_knc = RandomizedSearchCV(bagging_knc, params, n_iter = 30, verbose = 0,cv = 3, n_jobs = -1)
rs_knc.fit(X_train,y_train)
```

```python
rs_knc.best_params_
```

```
{'oob_score': True,
 'n_estimators': 7,
 'max_samples': 0.5,
 'max_features': 1.0,
 'base_estimator__n_neighbors': 7}
```

```python
knc = KNeighborsClassifier(n_neighbors = 7)
br_knc = BaggingClassifier(base_estimator=knc, oob_score = True, n_estimators=7, max_features=1.0, max_samples=0.5)
br_knc.fit(X_train, y_train)
```



##### define evaluate function

```python
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# evaluate model
def evaluate(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    print("TRAINIG RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

    print("TESTING RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")
```

```python
evaluate(br_knc, X_train, X_test, y_train, y_test)
```

![image-20221101021321720](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101021321720.png)

```python
scores = {
    'Bagging Classifier': {
        'Train': accuracy_score(y_train, br_knc.predict(X_train)),
        'Test': accuracy_score(y_test, br_knc.predict(X_test)),
    },
}
print(scores)
```

```python
{'Bagging Classifier': {'Train': 0.7951582867783985, 'Test': 0.7619047619047619}}
```

```python
#from sklearn.metrics import accuracy_score
y_pred = br_knc.predict(X_test)
accuracy_score(y_test,y_pred,normalize = True)
```

0.7619047619047619

```python
#from sklearn.utils.multiclass import type_of_target
y_test = np.array(y_test.astype(int))
y_pred = np.array(y_pred.astype(int))
```

```python
#from sklearn.metrics import mean_squared_error
#from math import sqrt
rmse = sqrt(mean_squared_error(y_test,y_pred))
rmse
```

```
0.9759000729485332
```

#### 2. decision trees

```python
#from sklearn.model_selection import RandomizedSearchCV
params = {
    'n_estimators':[3,5,7,11,20,30,40,50,60,70,80,90],#要集成的基估计器的个数。
    #'n_estimators':[int(x) for x in np.linspace(start = 200,stop = 2000,num = 10)]
    'max_samples':[0.3,0.5,0.7,1.0], #决定从x_train抽取去训练基估计器的样本数量,
    'max_features':[0.3,0.5,0.7,1.0], #决定从x_train抽取去训练基估计器的特征数量
    'oob_score':[True, False], #决定是否使用包外估计（out of bag estimate）泛化误差
    'base_estimator__criterion':['gini', 'entropy'],
    'base_estimator__min_samples_split':[2,5,10],
    'base_estimator__min_samples_leaf':[1,2,4],
    'base_estimator__max_depth':[5,8,10],
    #'bootstrap':[True,False]
}
#参数修改
rs_tr = RandomizedSearchCV(bagging_tr, params, n_iter = 100, verbose = 2, n_jobs = -1, cv = 3, random_state = 0)
rs_tr.fit(X_train,y_train)
```

```python
rs_tr.best_params_
```

```python
{'oob_score': False,
 'n_estimators': 90,
 'max_samples': 1.0,
 'max_features': 0.7,
 'base_estimator__min_samples_split': 10,
 'base_estimator__min_samples_leaf': 1,
 'base_estimator__max_depth': 8,
 'base_estimator__criterion': 'entropy'}
```

```python
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=10, min_samples_leaf=1)
br_tr = BaggingClassifier(base_estimator=dtree, oob_score = False, n_estimators=90, max_features=0.7, max_samples=1.0)
br_tr.fit(X_train, y_train)
```

```python
evaluate(br_tr, X_train, X_test, y_train, y_test)
```

![image-20221101023103819](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101023103819.png)

```python
scores = {
    'Bagging Classifier': {
        'Train': accuracy_score(y_train, br_tr.predict(X_train)),
        'Test': accuracy_score(y_test, br_tr.predict(X_test)),
    },
}
print(scores)
```

```
{'Bagging Classifier': {'Train': 0.931098696461825, 'Test': 0.7532467532467533}}
```

```python
#from sklearn.metrics import mean_squared_error
#from math import sqrt
rmse = sqrt(mean_squared_error(y_test,y_pred))
rmse
```

```
0.993485272670404
```

#### 3. logistic regression

```python
#from sklearn.model_selection import RandomizedSearchCV
params = {
    'n_estimators':[3,5,7,11,20,30,40,50,60,70,80,90],#要集成的基估计器的个数。
    'max_samples':[0.3,0.5,0.7,1.0], #决定从x_train抽取去训练基估计器的样本数量,
    'max_features':[0.3,0.5,0.7,1.0], #决定从x_train抽取去训练基估计器的特征数量
    'oob_score':[True, False], #决定是否使用包外估计（out of bag estimate）泛化误差
    'base_estimator__penalty':['l1', 'l2', 'elasticnet'],
    'base_estimator__multi_class':['auto', 'ovr', 'multinomial'],
    'base_estimator__solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    #'bootstrap':[True, False] 
}
#参数修改
rs_lr = RandomizedSearchCV(bagging_lr, params, n_iter = 100, verbose = 2, n_jobs = -1, cv = 3, random_state = 0)
rs_lr.fit(X_train,y_train)
```

```python
rs_lr.best_params_
```

```python
{'oob_score': False,
 'n_estimators': 11,
 'max_samples': 0.7,
 'max_features': 1.0,
 'base_estimator__solver': 'sag',
 'base_estimator__penalty': 'l2',
 'base_estimator__multi_class': 'multinomial'}
```

```python
lr = LogisticRegression(penalty='l2', multi_class="multinomial", solver="sag")#solver="newton-cg"
br_lr = BaggingClassifier(base_estimator=lr, oob_score = False, n_estimators=11, max_features=1.0, max_samples=0.7)
br_lr.fit(X_train, y_train)
```

```python
evaluate(br_lr, X_train, X_test, y_train, y_test)
```

![image-20221101023517691](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101023517691.png)

```python
scores = {
    'Bagging Classifier': {
        'Train': accuracy_score(y_train, br_lr.predict(X_train)),
        'Test': accuracy_score(y_test, br_lr.predict(X_test)),
    },
}
print(scores)
```

```python
{'Bagging Classifier': {'Train': 0.7746741154562383, 'Test': 0.7705627705627706}}
```

```python
#from sklearn.metrics import mean_squared_error
#from math import sqrt
rmse = sqrt(mean_squared_error(y_test,y_pred))
rmse
```

```python
0.993485272670404
```

### grid_search

比较慢

### boosting

#### 1. Adaboosting

AdaBoost 分类器是一种元估计器，它首先在原始数据集上拟合分类器，然后在同一数据集上拟合分类器的其他副本，但调整错误分类实例的权重，以便后续分类器更多地关注困难案例。

**AdaBoostClassifier Params**:

- `base_estimator` : The base estimator from which the boosted ensemble is built.

------

- `n_estimators` : The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.

------

- `learning_rate` : Learning rate shrinks the contribution of each classifier by `learning_rate`. There is a trade-off between `learning_rate` and `n_estimators`.

------

- `algorithm` : If 'SAMME.R' then use the SAMME.R real boosting algorithm. `base_estimator` must support calculation of class probabilities. If 'SAMME' then use the SAMME discrete boosting algorithm. The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.

AdaBoostClassifier 参数： 

- base_estimator ：构建增强集成的基本估计器。 
- n_estimators ：终止提升的估计器的最大数量。在完美契合的情况下，学习过程会提前停止。 
- learning_rate ：学习率通过 learning_rate 缩小每个分类器的贡献。 learning_rate 和 n_estimators 之间存在权衡。 
- algorithm : 如果 'SAMME.R' 则使用 SAMME.R 真实增强算法。 base_estimator 必须支持类概率的计算。如果是“SAMME”，则使用 SAMME 离散增强算法。 SAMME.R 算法通常比 SAMME 收敛得更快，以更少的提升迭代实现更低的测试误差。

```python
from sklearn.ensemble import AdaBoostClassifier

ada_boost_clf = AdaBoostClassifier(n_estimators=30)
ada_boost_clf.fit(X_train, y_train)
evaluate(ada_boost_clf, X_train, X_test, y_train, y_test)
```

![image-20221101025931067](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101025931067.png)

```python
scores['AdaBoost'] = {
        'Train': accuracy_score(y_train, ada_boost_clf.predict(X_train)),
        'Test': accuracy_score(y_test, ada_boost_clf.predict(X_test)),
    }
```

#### 2. Stochastic Gradient Boosting

GB 以前向阶段方式构建加法模型；它允许优化任意可微损失函数。在每个阶段，n_classes_ 回归树都适合二项式或多项式偏差损失函数的负梯度。二元分类是一种特殊情况，其中只引入了一个回归树。

**GradientBoostingClassifier Parameters**:

- `loss` : loss function to be optimized. 'deviance' refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss 'exponential' gradient boosting recovers the AdaBoost algorithm.

------

- `learning_rate` : learning rate shrinks the contribution of each tree by `learning_rate`. There is a trade-off between learning_rate and n_estimators.

------

- `n_estimators` : The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.

------

- `subsample` : The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. `subsample` interacts with the parameter `n_estimators`. Choosing `subsample < 1.0` leads to a reduction of variance and an increase in bias.

------

- `criterion` : The function to measure the quality of a split. Supported criteria are "friedman_mse" for the mean squared error with improvement score by Friedman, "mse" for mean squared error, and "mae" for the mean absolute error. The default value of "friedman_mse" is generally the best as it can provide a better approximation in some cases.

------

- `min_samples_split`: The minimum number of samples required to split an internal node.

------

- `min_samples_leaf`: The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least `min_samples_leaf` training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.

------

- `min_weight_fraction_leaf`: The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.

------

- `max_depth`: maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.

------

- `min_impurity_decrease`: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

------

- `min_impurity_split`: Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.

------

- `max_features`: The number of features to consider when looking for the best split.

------

- `max_leaf_nodes`: Grow trees with `max_leaf_nodes` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

------

- `warm_start`: When set to `True`, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just erase the previous solution.

------

- `validation_fraction`: The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if `n_iter_no_change` is set to an integer.

------

- `n_iter_no_change`: used to decide if early stopping will be used to terminate training when validation score is not improving. By default it is set to None to disable early stopping. If set to a number, it will set aside `validation_fraction` size of the training data as validation and terminate training when validation score is not improving in all of the previous `n_iter_no_change` numbers of iterations. The split is stratified.

------

- `tol`: Tolerance for the early stopping. When the loss is not improving by at least tol for `n_iter_no_change` iterations (if set to a number), the training stops.

------

- `ccp_alpha`: Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than `ccp_alpha` will be chosen.

GradientBoostingClassifier 参数： 

- loss：要优化的损失函数。 “偏差”是指用于概率输出分类的偏差（=逻辑回归）。对于损失，“指数”梯度提升恢复了 AdaBoost 算法。 
- learning_rate ：学习率通过 learning_rate 缩小每棵树的贡献。 learning_rate 和 n_estimators 之间存在权衡。 
- n_estimators ：要执行的提升阶段的数量。梯度提升对过度拟合相当稳健，因此较大的数量通常会带来更好的性能。 
- subsample ：用于拟合单个基础学习器的样本分数。如果小于 1.0，则会导致随机梯度提升。 subsample 与参数 n_estimators 交互。选择子样本 < 1.0 会导致方差减少和偏差增加。 
- 标准：衡量分割质量的函数。支持的标准是均方误差的“friedman_mse”，Friedman 的改进分数，均方误差的“mse”和平均绝对误差的“mae”。 “friedman_mse”的默认值通常是最好的，因为它在某些情况下可以提供更好的近似值。 
- min_samples_split：拆分内部节点所需的最小样本数。 
- min_samples_leaf：叶节点所需的最小样本数。只有在左右分支中的每个分支中至少留下 min_samples_leaf 训练样本时，才会考虑任何深度的分割点。这可能具有平滑模型的效果，尤其是在回归中。 
- min_weight_fraction_leaf：需要在叶节点处的（所有输入样本的）权重总和的最小加权分数。当未提供 sample_weight 时，样本具有相同的权重。 
- max_depth：单个回归估计器的最大深度。最大深度限制了树中的节点数。调整此参数以获得最佳性能；最佳值取决于输入变量的相互作用。 
- min_impurity_decrease：如果该分裂导致杂质减少大于或等于该值，则该节点将被分裂。 
- min_impurity_split：树木生长提前停止的阈值。如果一个节点的杂质高于阈值，它就会分裂，否则它就是一个叶子。 
- max_features：寻找最佳分割时要考虑的特征数量。 
- max_leaf_nodes：以最佳优先方式使用 max_leaf_nodes 种植树。最佳节点定义为杂质的相对减少。如果 None 则无限数量的叶节点。 
- warm_start：当设置为 True 时，重用前一个调用的解决方案来拟合并添加更多的估计器到集成中，否则，只是删除以前的解决方案。 
- validation_fraction：为提前停止而留出作为验证集的训练数据的比例。必须介于 0 和 1 之间。仅在 n_iter_no_change 设置为整数时使用。 
- n_iter_no_change：用于决定当验证分数没有提高时是否使用提前停止来终止训练。默认情况下，它设置为 None 以禁用提前停止。如果设置为一个数字，它将保留训练数据的 
- validation_fraction 大小作为验证，并在验证分数在所有先前的 n_iter_no_change 迭代次数中都没有提高时终止训练。分裂是分层的。 
- tol：提前停止的公差。当 n_iter_no_change 迭代（如果设置为数字）的损失至少没有改善 tol 时，训练停止。 
- ccp_alpha：用于最小成本复杂度修剪的复杂度参数。将选择具有最大成本复杂度且小于 ccp_alpha 的子树。

```python
from sklearn.ensemble import GradientBoostingClassifier

grad_boost_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
grad_boost_clf.fit(X_train, y_train)
evaluate(grad_boost_clf, X_train, X_test, y_train, y_test)
```

![image-20221101030015958](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101030015958.png)

```python
scores['Gradient Boosting'] = {
        'Train': accuracy_score(y_train, grad_boost_clf.predict(X_train)),
        'Test': accuracy_score(y_test, grad_boost_clf.predict(X_test)),
    }
```

### Model Comparison

```python
scores_df = pd.DataFrame(scores)

scores_df.plot(kind='barh', figsize=(15, 8))
```

k-nearest neighbor vs decision tree vs logistic regression

![image-20221101031700936](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101031700936.png)

bagging vs adaboosting vs gbc

![image-20221101025537672](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101025537672.png)

