# 6-3-模式识别-svm classification

## 工作目标：

熟悉支持向量机的分类算法。

## 实验室工作任务：

基于三个数据集之一的支持向量机构建分类器。研究分类器内核的选择和其他参数对模型结果的影响。报告已完成的工作。
模型学习数据集：
sklearn.datasets.load_breast_cancer¶
sklearn.datasets.load_iris
sklearn.datasets.load_wine

## 理论部分：

支持向量机（SVM，Support Vector Machine）是一种教师辅助学习算法。支持向量机既可用于分类任务，也可用于回归任务。该算法应用于面部识别、电子邮件、新闻和网页分类、基因分类和手写文本识别等领域。SVM方法特别适用于复杂但小型或中型数据集的分类。

该方法的主要思想是将原始向量转换为更高维度的空间，并搜索该空间中间隙最大的分隔超平面。两个平行的超平面在分隔类的超平面的两侧构造。分离超平面是产生两个平行超平面最大距离的超平面。该算法基于这样一个假设，即这些平行超平面之间的差异或距离越大，分类器的平均误差就越小。

### SVC 算法

```python
class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
```

C-支持向量分类。 

该实现基于libsvm。拟合时间至少与样本数量成二次方比例，超过数万个样本可能不切实际。对于大型数据集，考虑使用[`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)或 [`SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier) ，可能在[`Nystroem`](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html#sklearn.kernel_approximation.Nystroem) 变压器之后。 

多类支持根据一对一方案进行处理。 

有关所提供内核函数的精确数学公式以及gamma、coeff0和degree如何相互影响的详细信息，请参阅叙事文档中的相应部分：[Kernel functions](https://scikit-learn.org/stable/modules/svm.html#svm-kernels)。

---

#### SVC 算法参数：

- **C**：float，默认值=1.0 

  正则化参数。正则化的强度与C成反比。必须严格为正。惩罚是l2的平方惩罚。 

- **kernel** 内核：｛‘linear‘，‘poly‘，‘rbf‘，‘sigmoid‘，‘precomputed‘｝或可调用，默认值=‘rbf‘ 

  指定要在算法中使用的内核类型。如果没有给出，将使用“rbf”。如果给定了一个可调用函数，则它用于从数据矩阵中预计算内核矩阵；该矩阵应该是一个形状数组`(n_samples, n_samples)`。 

- **degree**：int，默认值=3 

  多项式核函数的次数（‘poly’）。必须为非负。被所有其他内核忽略。 

- **gamma**：｛‘scale‘，‘auto‘｝或float，默认值=‘scale’ 

  “rbf”、“poly”和“sigmoid”的核系数。 

  - 如果通过`gamma='scale'`（默认值），则使用`1/(n_features*X.var())`作为gamma的值， 

  - 如果为“auto”，则使用1/n_features 

  - 如果为float，则必须为非负。 

    在版本0.22中更改：gamma的默认值从“auto”更改为“scale”。 

- **coef0**：float，默认值=0.0 

  核函数中的独立项。它只在“poly”和“sigmoid”中有意义。

- **shrinking**：bool，默认值=True 

  是否使用收缩启发式。请参阅《用户指南》。 

- **probability**：bool，默认值=False 

  是否启用概率估计。这必须在调用`fit`之前启用，这会降低该方法的速度，因为它内部使用5倍交叉验证，`predict_proba`可能与`predict`不一致。阅读《用户指南》中的更多信息。 

- **tol**：float，默认值=1e-3 

  停止标准的公差。 

- **cache_size**：float，默认值=200 

  指定内核缓存的大小（以MB为单位）。 

- **class_weight**：dict或“balanced”，默认值=None 

  对于SVC，将类i的参数C设置为`class_weight[i]*C`。如果没有给出，所有的课程都应该有一个权重。“平衡(balanced)”模式使用y的值自动调整与输入数据中的类频率成反比的权重，即`n_samples/(n_classes*np.bincount(y))`。

- **verbose**：bool，默认值=False 

  启用详细输出。注意，此设置利用了libsvm中的每进程运行时设置，如果启用该设置，则在多线程上下文中可能无法正常工作。 

- **max_iter**：int，默认值=-1 

  解算器内迭代的硬限制，或-1表示无限制。 

- **decision_function_shape**：｛'ovo'，'ovr'｝，默认值='ovr' 

  是返回`shape(n_samples，n_classes)`的`one-vs-rest('ovr')`决策函数作为所有其他分类器，还是返回具有`shape(n_samples，n_classes*(n_classes-1)/2)`的libsvm的原始`one-vs-one('ovo')`决策函数。然而，请注意，在内部，`one-vs-one('ovo')` 总是被用作训练模型的多类策略；ovr矩阵仅由ovo矩阵构造。对于二进制分类，忽略该参数。 

  在版本0.19中更改：默认情况下，decision_function_shape为“ovr”。 

  0.17版中的新功能：建议使用decision_function_shape='ovr'。 

  在版本0.17中更改：不推荐的decision_function_shape='ovo'和None。 

- **break_ties**：bool，默认值=False 

  如果为真，`decision_function_shape='ovr'`，且类数>2，则预测( [predict](https://scikit-learn.org/stable/glossary.html#term-predict) )将根据 [decision_function](https://scikit-learn.org/stable/glossary.html#term-decision_function) 的置信值打破联系；否则返回绑定类中的第一个类。请注意，与简单预测相比，打破联系的计算成本相对较高。 

  0.22版中的新功能。 

- **random_state**：int，RandomState实例或None，默认值=None 

  控制伪随机数的生成，以便对概率估计的数据进行混洗。当概率`probability`为False时忽略。在多个函数调用之间传递一个int以获得可复制的输出。请参阅词汇表。

---

### 图解SVC

假设有一组数据分为两类

![image-20221228184046395](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221228184046395.png)

SVM训练的任务是在这些类之间划出边界，以便这些类不会相互混合。在这些点之间可以画无限多条线。以下是两条可能的路线：

![image-20221228184127215](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221228184127215.png)

这两条线将两个类分开，并且不混合线两侧的类。这些线称为超平面。

SVM选择分隔类的线（或超平面）。SVM选择一个超平面，该超平面与两侧的边界点具有最大距离，这意味着在训练SVM时，SVM考虑到超平面的最近点，如下图所示：

![image-20221228184238516](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221228184238516.png)

这个超平面与两侧最近点的最大距离。这两个点（上图中阴影区域中的两个点）是支持最优超平面的点，因此这些点被称为参考向量。参考向量之间的距离称为margin。

因此，SVM算法试图最大化Margin，以确保类之间的最佳分离。

基本超平面（黑线）被称为解的边界，以区别于左右两个其他超平面。

#### 超参数C

通过允许某些错误（或错误分类），可以使用一个称为“C”的参数来控制错误的数量。它可以是任何值，例如0.01，甚至100或更多。这取决于问题的类型和可用的数据。

“C”直接影响超平面。“C”与字段宽度成反比。因此，C越大，Margin就越小，反之亦然。



### HalvingGridSearchCV

[`sklearn.model_selection`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection).HalvingGridSearchCV

```python
class sklearn.model_selection.HalvingGridSearchCV(estimator, param_grid, *, factor=3, resource='n_samples', max_resources='auto', min_resources='exhaust', aggressive_elimination=False, cv=5, scoring=None, refit=True, error_score=nan, return_train_score=True, random_state=None, n_jobs=None, verbose=0)
```

- **estimator**: 估计器对象( estimator object )

  假设这是为了实现scikit学习估计器接口。估计器需要提供评分 `score` 函数，或者必须通过评分`scoring`。 

- **param_grid**: 字典或词典列表( dict or list of dictionaries )

  将参数名称（字符串）作为关键字的字典，将要尝试的参数设置列表作为值，或此类字典的列表，在这种情况下，将探究列表中每个字典所跨越的网格。这允许搜索任何参数设置序列。 

- **factor**: int或float，默认值(default)=3 ( int or float, default=3 )

  “减半”参数，它确定为每个后续迭代选择的候选项的比例。例如，`factor=3` 意味着只有三分之一的候选人被选中。 

- **resource**: `'n_samples'`或str，默认值='n_samples' 

  定义随每次迭代而增加的资源。默认情况下，资源是样本数。也可以将其设置为接受正整数值的基础估计器的任何参数，例如，对于梯度增强估计器，“n_iterations”或“n_estimators”。在这种情况下，`max_resources`不能为“auto”，必须显式设置。 

- **max_resources**: int，默认值="auto"(自动)

  允许任何候选项用于给定迭代的最大资源量。默认情况下，当`resource='n_samples'`（默认值）时，该值设置为`n_samples`，否则会引发错误。 

- **min_resources**:｛'exhaust'，'minimal'｝或int，默认值='exhaust' 

  允许任何候选项用于给定迭代的最小资源量。等效地，这定义了在第一次迭代时为每个候选分配的资源r0的数量。 

  - “minimal”是将r0设置为较小值的启发式方法： 
    - 当回归问题的 `resource='n_samples'` 时，`n_splits * 2` 
    - 当分类问题的 `resource='n_samples'`时，`n_classes * n_splits * 2` 
    -  `resource != 'n_samples'` 时为1
  - “exhaust”将设置r0，以便**最后一次**迭代使用尽可能多的资源。即，最后一次迭代将使用小于max_resources的最大值，即min_resources和factor的倍数。一般来说，使用“exhaust”会导致更准确的估算，但会稍微耗费更多时间。

  注意，每次迭代使用的资源量总是min_resources的倍数。 

- **aggressie_elimition**: bool，默认值=False 

  这只在没有足够的资源在最后一次迭代后将剩余的候选项减少到最多的 `factor` 的情况下才相关。如果为 `True` ，则搜索过程将根据需要“回放(replay)”第一次迭代，直到候选的数量足够少。默认情况下为 `False` ，这意味着最后一次迭代可能会评估多个候选因子 `factor`。有关更多详细信息，请参阅积极淘汰候选人(  [Aggressive elimination of candidates](https://scikit-learn.org/stable/modules/grid_search.html#aggressive-elimination)  )。 

- **cv**: int，交叉验证生成器或可迭代，默认值=5 

  确定交叉验证拆分策略。cv的可能输入包括： 

  - 整数，以指定（分层）KFold( `(Stratified)KFold` )中的折叠数， 
  - CV分离器([CV splitter](https://scikit-learn.org/stable/glossary.html#term-CV-splitter))
  - 可迭代的领域（训练、测试）拆分为索引数组。 

  对于整数/无输入，如果估计器是分类器，并且y是二进制或多类，则使用**StratifiedKFold**。在所有其他情况下，使用**KFold**。这些拆分器是使用`shuffle=False`实例化的，因此在调用之间拆分是相同的。 

  有关此处可使用的各种交叉验证策略，请参阅《用户指南》( [User Guide](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation) )。 

  ```py
  注意：由于实现的细节，cv生成的折叠在多次调用cv.split()时必须相同。对于内置的scikit学习迭代器，这可以通过停用shuffling（shuffle=False）或将cv的random_state参数设置为整数来实现。
  ```

- **scoring**: str，callable或None，默认值=None 

  一个字符串（请参见评分参数：定义模型评估规则( [The scoring parameter: defining model evaluation rules](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter))）或一个可调用字符串（请参阅从度量函数定义评分策略([Defining your scoring strategy from metric functions](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring))）来评估测试集上的预测。如果无，则使用估计器的得分方法。 

- **refit**: bool，默认值=True 

  如果为True，则使用整个数据集上找到的最佳参数重新调整估计器。 重新编译的估计器在best_estimator_属性中可用，并允许在此HalvingGridSearchCV实例上直接使用预测。 

- **error_score**: 'raise'或数字 

  如果估计器拟合中出现错误，则分配给得分的值。如果设置为“raise”，则会引发错误。如果给定数值，则会引发FitFailedWarning。此参数不会影响重新安装步骤，这将始终导致错误。默认值为`np.nan`。 

- **return_train_score**: bool，默认值=False 

  如果为`False`，`cv_results_`属性将不包含训练分数。计算训练分数用于深入了解不同的参数设置如何影响过度拟合/不足拟合权衡。然而，计算训练集上的分数在计算上可能是昂贵的，并且不严格要求选择产生最佳泛化性能的参数。 

- **random_state**: int，RandomState实例或None，默认值=None 

  当资源！='时，用于对数据集进行二次采样的伪随机数生成器状态n_samples’。否则忽略。在多个函数调用之间传递一个int以获得可复制的输出。请参阅词汇表。 

- **n_job**: int或None，默认值=None 

  要并行运行的作业数。除非在**joblib.paralle_backend**上下文中，否则`None`表示`1`，`-1`表示使用所有处理器。有关详细信息，请参阅词汇表([Glossary](https://scikit-learn.org/stable/glossary.html#term-n_jobs))。 

- **verbose**: int 

  控制详细程度：越高，消息越多。

---

```python
sklearn.svm.SVC
(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,random_state=None)

参数：

C：C-SVC的惩罚参数C?默认值是1.0
C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。

kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

– 线性：u'v

– 多项式：(gammau'v + coef0)^degree

– RBF函数：exp(-gamma|u-v|^2)

– sigmoid：tanh(gammau'v + coef0)

degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
probability ：是否采用概率估计？.默认为False
shrinking ：是否采用shrinking heuristic方法，默认为true
tol ：停止训练的误差值大小，默认为1e-3
cache_size ：核函数cache缓存大小，默认为200
class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
verbose ：允许冗余输出？
max_iter ：最大迭代次数。-1为无限制。
decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3
random_state ：数据洗牌时的种子值，int值
主要调节的参数有：C、kernel、degree、gamma、coef0。
```

## 实验

### 加载库

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import seaborn as sns

from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.decomposition import PCA,KernelPCA,FastICA # Dimensionality reduction
from sklearn.cross_decomposition import CCA # Dimensionality reduction
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix 
# explicitly require this experimental feature, for HalvingGridSearchCV,HalvingRandomSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import KFold
%matplotlib inline
```

```
# load library
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import cm
import pandas as pd
import seaborn as sns

# load dataset and preprocessing
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# model
from sklearn.svm import SVC,LinearSVC
from sklearn.decomposition import PCA,KernelPCA # Dimensionality reduction

# evaluation
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix 

# find best parameters
# explicitly require this experimental feature, for HalvingGridSearchCV,HalvingRandomSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV 
```



### 数据查看

```python
# 读取数据
def dataset_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df
```

```python
df_breast_cancer = dataset_to_df(load_breast_cancer())
df_breast_cancer.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 569 entries, 0 to 568
Data columns (total 31 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   mean radius              569 non-null    float64
 1   mean texture             569 non-null    float64
 2   mean perimeter           569 non-null    float64
 3   mean area                569 non-null    float64
 4   mean smoothness          569 non-null    float64
 5   mean compactness         569 non-null    float64
 6   mean concavity           569 non-null    float64
 7   mean concave points      569 non-null    float64
 8   mean symmetry            569 non-null    float64
 9   mean fractal dimension   569 non-null    float64
 10  radius error             569 non-null    float64
 11  texture error            569 non-null    float64
 12  perimeter error          569 non-null    float64
 13  area error               569 non-null    float64
 14  smoothness error         569 non-null    float64
 15  compactness error        569 non-null    float64
 16  concavity error          569 non-null    float64
 17  concave points error     569 non-null    float64
 18  symmetry error           569 non-null    float64
 19  fractal dimension error  569 non-null    float64
 20  worst radius             569 non-null    float64
 21  worst texture            569 non-null    float64
 22  worst perimeter          569 non-null    float64
 23  worst area               569 non-null    float64
 24  worst smoothness         569 non-null    float64
 25  worst compactness        569 non-null    float64
 26  worst concavity          569 non-null    float64
 27  worst concave points     569 non-null    float64
 28  worst symmetry           569 non-null    float64
 29  worst fractal dimension  569 non-null    float64
 30  target                   569 non-null    int64  
dtypes: float64(30), int64(1)
memory usage: 137.9 KB
```

数据集没有缺失值，且都是数值型数据。

```python
df_breast_cancer.describe()
```

![image-20230109153038411](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230109153038411.png)

数据集不是标准化分布，需要规范化数据集的特征。

```python
data = load_breast_cancer()
print(list(data.target_names))
print(list(data.feature_names))
```

```
['malignant', 'benign']
['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']
```

```python
sns.pairplot(df_breast_cancer,hue='target',palette='Dark2')
```

![image-20230109153149619](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230109153149619.png)

![image-20230109153253582](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230109153253582.png)

![image-20230109153340883](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230109153340883.png)

### 数据处理

```python
X,y = load_breast_cancer(return_X_y=True) # numpy arrays
```

```python
# Draw data points in different categories and colors
cdict = {0: 'red', 1: 'green'}
_, ax = plt.subplots(figsize=(10,10))
 
for g in np.unique(y):
    ix = np.where(y == g)
    ax.scatter(X[ix, 0], X[ix, 1], c = cdict[g], label = g, s = 100)

plt.show()
```

#### Standarize features

```python
# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
```

#### split the data

```python
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3,random_state=109) 
```



### SVM Margins

用PCA或KernelPCA把30维数据降到2维，再画SVM Margins

```python
def plot_svc_margin(X, Y, fignum, title, transform, kernel=None, distance=0):
    '''
    draw the margin of the svc model
    params:
    - X: features
    - Y: targets
    - fignum: A unique identifier for the figure.
    - title: figure title
    - transform: dimensionality reduction: PCA or KernelPCA, use "pca" or "kpca"
    - kernel(=None): kernel PCA: 
            kernel{'linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'}
    - distance(=0): change the size of the picture
    '''
    
    # choose Dimensionality reduction pca or cca
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "kpca":
        X = KernelPCA(n_components=2, kernel=kernel).fit_transform(X)
        # kernel{'linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'}
    else:
        raise ValueError
    
    # Standarize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # fit the model
    clf = SVC(kernel="linear")
    clf.fit(X, Y)
    
    # Limit the range of horizontal and vertical axes
    min_x = np.min(X[:, 0])-distance
    max_x = np.max(X[:, 0])+distance
    min_y = np.min(X[:, 1])-distance
    max_y = np.max(X[:, 1])+distance
    
    # get the separating hyperplane 绘制超平面
    # clf = Abbreviation of classifier，classifier的缩写
    w = clf.coef_[0]  # View Weight Matrix 查看权重矩阵
    a = -w[0] / w[1] 
    xx = np.linspace(min_x-5, max_x+5)  # make sure the line is long enough
    # Take out the intercept 取出截距
    bias = clf.intercept_[0]
    b = - bias / w[1]
    yy = a * xx + b
    
    # 绘制通过支持向量的分离超平面的平行线（在垂直于超平面的方向上远离超平面的边缘）。
    # 这是二维垂直方向的sqrt（1+a^2）
    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    margin = 1 / np.sqrt(np.sum(clf.coef_**2))
    yy_down = yy - np.sqrt(1 + a**2) * margin
    yy_up = yy + np.sqrt(1 + a**2) * margin
    
    # plot the line, the points, and the nearest vectors to the plane
    # plt.subplot(2, 2, subplot)
    
    plt.figure(fignum, figsize=(10, 10)) 
    # fignum is the unique identification for the picture
    plt.clf()
    plt.plot(xx, yy, "k-") # hyperplane
    plt.plot(xx, yy_down, "k--") # down margin line
    plt.plot(xx, yy_up, "k--") # up margin line
    # plt.plot(xx, yy, linestyle, label=label)
    
    # Circle the support vector 
    plt.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=80,
        facecolors="none",
        zorder=10,
        edgecolors="k",
        cmap=cm.get_cmap("RdBu"),
    )
    
    # plot the data on the map
    plt.scatter(
        X[:, 0], X[:, 1], c=Y, zorder=10, cmap=cm.get_cmap("RdBu"), edgecolors="k"
    )
    
    plt.axis("tight")
    
    # get the distance from the instance to the hyperplane
    YY, XX = np.meshgrid(yy, xx) 
    # meshgrid: The input is changed from the original array to a matrix
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    # decision_function: the distance from the parameter instance 
    #                    to the hyperplane represented by each class 

    # Put the result into a contour plot
    plt.contourf(XX, YY, Z, cmap=cm.get_cmap("RdBu"), alpha=0.5, linestyles=["-"])

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    
```

```python
# pca
# choose Dimensionality reduction pca or cca
plot_svc_margin(X_std, y, 1,  "With unlabeled samples + PCA", "pca",distance=0.5)
```

![image-20230109152307208](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230109152307208.png)

```python
# kernel pca linear
# choose Dimensionality reduction pca or cca
plot_svc_margin(X_std, y, 6, "With unlabeled samples + Kernel PCA", "kpca",'linear',distance=0.5)
```

![image-20230109152326595](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230109152326595.png)

```python
# kernel pca poly
# choose Dimensionality reduction pca or cca
plot_svc_margin(X_std, y, 2, "With unlabeled samples + Kernel PCA", "kpca",'poly',distance=0.5)
# kernel{'linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'}
```

![image-20230109152355027](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230109152355027.png)

```python
# kernel pca rbf
# choose Dimensionality reduction pca or cca
plot_svc_margin(X_std, y, 3, "With unlabeled samples + Kernel PCA", "kpca",'rbf',distance=0.5)
```

![image-20230109152423303](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230109152423303.png)

```python
# kernel pca sigmoid
# choose Dimensionality reduction pca or cca
plot_svc_margin(X_std, y, 4, "With unlabeled samples + Kernel PCA", "kpca",'sigmoid',distance=0.5)
```

![image-20230109152443703](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230109152443703.png)

```python
# kernel pca cosine
# choose Dimensionality reduction pca or cca
plot_svc_margin(X_std, y, 5, "With unlabeled samples + Kernel PCA", "kpca",'cosine',distance=0.5)
```

![image-20230109152515723](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230109152515723.png)

### svm 核函数的比较

```python
kernels = ['linear','poly','rbf','sigmoid']
for kernel in kernels:
    # Train a SVC model using different kernal
    degree = np.where(kernel=='poly',3,0)
    svclassifier = SVC(kernel=kernel, degree=degree, gamma="auto")
    svclassifier.fit(X_train, y_train)# Make prediction
    y_pred = svclassifier.predict(X_test)# Evaluate our model
    print("Evaluation:", kernel, "kernel")
    print(classification_report(y_test,y_pred))
```

```
Evaluation: linear kernel
              precision    recall  f1-score   support

           0       1.00      0.97      0.98        63
           1       0.98      1.00      0.99       108

    accuracy                           0.99       171
   macro avg       0.99      0.98      0.99       171
weighted avg       0.99      0.99      0.99       171

Evaluation: poly kernel
              precision    recall  f1-score   support

           0       1.00      0.73      0.84        63
           1       0.86      1.00      0.93       108

    accuracy                           0.90       171
   macro avg       0.93      0.87      0.89       171
weighted avg       0.91      0.90      0.90       171

Evaluation: rbf kernel
              precision    recall  f1-score   support

           0       1.00      0.95      0.98        63
           1       0.97      1.00      0.99       108

    accuracy                           0.98       171
   macro avg       0.99      0.98      0.98       171
weighted avg       0.98      0.98      0.98       171

Evaluation: sigmoid kernel
              precision    recall  f1-score   support

           0       0.98      0.95      0.97        63
           1       0.97      0.99      0.98       108

    accuracy                           0.98       171
   macro avg       0.98      0.97      0.97       171
weighted avg       0.98      0.98      0.98       171
```

```python
#from sklearn import svm

#Create a svm Classifier
clf = SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
```

```python
#Import scikit-learn metrics module for accuracy calculation
#from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
```

```python
#X_train, X_test, y_train, y_test = train_test_split(x_second, y_second, test_size=0.3,random_state=109)
clf = SVC(kernel='linear') # Linear Kernel
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
clf.get_params()
```

```
Accuracy: 0.9883040935672515
```

[113]:

```
{'C': 1.0,
 'break_ties': False,
 'cache_size': 200,
 'class_weight': None,
 'coef0': 0.0,
 'decision_function_shape': 'ovr',
 'degree': 3,
 'gamma': 'scale',
 'kernel': 'linear',
 'max_iter': -1,
 'probability': False,
 'random_state': None,
 'shrinking': True,
 'tol': 0.001,
 'verbose': False}
```

## 超参数搜索：

### 参数列表

```python
# Set the parameters by cross-validation
tuned_parameters = [{
    # 解释：kernel(内核)：｛‘linear‘，‘poly‘，‘rbf‘，‘sigmoid‘，‘precomputed‘｝或可调用，默认值=‘rbf‘ 
    #'kernel': ['linear','poly','rbf'],  
    # 解释：degree（‘poly’）：表示选择的多项式核函数的最高次数。必须为非负。
    # 'degree': range(3, 5),
    # 解释：coef0：核函数常数值(y=kx+b中的b值)，只有‘poly’和‘sigmoid’核函数有
    # 'coef0': np.arange(0, 1, 0.1),
    # 解释：C：惩罚系数，用来控制损失函数的惩罚系数，类似于LR中的正则化系数。
    # C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，
    # 这样会出现训练集测试时准确率很高，但泛化能力弱，容易导致过拟合。 
    # C值小，对误分类的惩罚减小，容错能力增强，泛化能力较强，但也可能欠拟合。
    'C': [0.01,0.1,1, 10, 100,1000],
    #'C': np.logspace(-3, 5, 17),
    #'break_ties': [True, False],
    'cache_size': [50,100,200,300,400],# 缓冲大小，用来限制计算量大小，默认是200M。
    # 解释：decision_function_shape: 原始的SVM只适用于二分类问题，如果要将其扩展到多类分类，就要采取一定的融合策略，这里提供了三种选择。
    'decision_function_shape': ['ovr','ovo'],
    # 解释：gamma：核函数系数，该参数是rbf，poly和sigmoid的内核系数；默认是'auto'
    # gamma越大，σ越小，使得高斯分布又高又瘦，造成模型只能作用于支持向量附近，可能导致过拟合；
    # 反之，gamma越小，σ越大，高斯分布会过于平滑，在训练集上分类效果不佳，可能导致欠拟合。
    'gamma': ['scale','auto'],
    #'gamma': np.logspace(-3, 5, 17)
    # 解释：probability：是否使用概率估计，默认是False。必须在fit()方法前使用，该方法的使用会降低运算速度。
    'probability': [True, False],
    # 解释：shrinking：是否进行启发式。
    #'shrinking': [True, False],
    # 解释：tol：残差收敛条件，默认是0.0001，即容忍1000分类里出现一个错误，与LR中的一致；误差项达到指定值时则停止训练。
    'tol': [0.01,0.001,0.0001],
    # 解释：verbose：是否启用详细输出
    #'verbose': [True, False],

    # 解释：class_weight:{dict, ‘balanced’}，字典类型或者'balance'字符串。该参数就是指每个类所占据的权重，
    #'class_weight': None,
    # 解释：max_iter: 最大迭代次数，默认是-1，即没有限制。
    # 这个是硬限制，它的优先级要高于tol参数，不论训练的标准和精度达到要求没有，都要停止训练。
    #'max_iter': [-1],
    # 解释：random_state：在使用SVM训练数据时，要先将训练数据打乱顺序，用来提高分类精度，这里就用到了伪随机序列。
    #'random_state': None,
    }]
```

```python
# Set the parameters by cross-validation
tuned_parameters = [{
    # 解释：kernel(内核)：｛‘linear‘，‘poly‘，‘rbf‘，‘sigmoid‘，‘precomputed‘｝或可调用(or callable, default=’rbf’)，default=‘rbf‘ 
    #'kernel': ['linear','poly','rbf'], 
    # -----------------------------------------------------------------------------------------
    # C: penalty coefficient, which is used to control the penalty coefficient of loss function, 
    # similar to the regularization coefficient in LR.
    # 1) The larger the C is, the greater the penalty for misclassification will be. 
    # The accuracy of training set testing is very high, but the generalization ability is weak, 
    # which is easy to lead to over fitting.
    # 2) The C value is small, the punishment for misclassification is reduced, 
    # the fault tolerance ability is enhanced, and the generalization ability is strong, 
    # but the fitting may also lead to under fitting.
    # 解释：C：惩罚系数，用来控制损失函数的惩罚系数，类似于LR中的正则化系数。
    # C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，
    # 这样会出现训练集测试时准确率很高，但泛化能力弱，容易导致过拟合。 
    # C值小，对误分类的惩罚减小，容错能力增强，泛化能力较强，但也可能欠拟合。
    'C': [0.001,0.01,0.1,1, 10, 100,1000],
    # -----------------------------------------------------------------------------------------
    # cache_size: The buffer size is used to limit the calculation amount. The default value is 200M.
    # 缓冲大小，用来限制计算量大小，默认是200M。
    'cache_size': [50,100,200,300,400],
    # -----------------------------------------------------------------------------------------
    # probability: whether to use probability estimation. The default value is False. 
    # It must be used before the fit () method, which will reduce the operation speed.
    # 解释：probability：是否使用概率估计，默认是False。必须在fit()方法前使用，该方法的使用会降低运算速度。
    'probability': [True, False],
    # -----------------------------------------------------------------------------------------
    # tol: residual convergence condition, which is 0.0001 by default, 
    # that is, an error in 1000 classification is tolerated, which is consistent with LR; 
    # Stop training when the error item reaches the specified value.
    # 解释：tol：残差收敛条件，默认是0.0001，即容忍1000分类里出现一个错误，与LR中的一致；误差项达到指定值时则停止训练。
    'tol': [0.01,0.001,0.0001],
    # -----------------------------------------------------------------------------------------
    # Explanation: gamma: kernel function coefficient. 
    # This parameter is the kernel coefficient of rbf, poly and sigmoid; Default is' auto '
    # 1) The larger the gamma, σ The smaller the size is, 
    # the higher and thinner the Gaussian distribution is, 
    # resulting in the model can only act near the support vector, which may lead to over fitting;
    # 2) On the contrary, the smaller the gamma, σ The larger the size is, 
    # the smoother the Gaussian distribution will be, 
    # and the classification effect on the training set will be poor, which may lead to under fitting.
    # 解释：gamma：核函数系数，该参数是rbf，poly和sigmoid的内核系数；默认是'auto'
    # gamma越大，σ越小，使得高斯分布又高又瘦，造成模型只能作用于支持向量附近，可能导致过拟合；
    # 反之，gamma越小，σ越大，高斯分布会过于平滑，在训练集上分类效果不佳，可能导致欠拟合。
    'gamma': ['scale','auto'],# use for'rbf'
    # -----------------------------------------------------------------------------------------
    # 解释：verbose：是否启用详细输出
    #'verbose': [True, False],
    # -----------------------------------------------------------------------------------------
    # shrinking：Whether to use heuristics.
    # 解释：shrinking：是否进行启发式。
    #'shrinking': [True, False],
    # -----------------------------------------------------------------------------------------
    # class_ Weight: {dict, 'balanced'}, dictionary type or 'balanced' string. 
    # This parameter refers to the weight of each class,
    # 解释：class_weight:{dict, ‘balanced’}，字典类型或者'balance'字符串。该参数就是指每个类所占据的权重，
    #'class_weight': None,
    # -----------------------------------------------------------------------------------------
    # 解释：max_iter: 最大迭代次数，默认是-1，即没有限制。
    # max_ Iter: the maximum number of iterations. The default is - 1, that is, there is no limit.
    # 这个是硬限制，它的优先级要高于tol参数，不论训练的标准和精度达到要求没有，都要停止训练。
    #'max_iter': [-1],
    # -----------------------------------------------------------------------------------------
    # 解释：random_state：在使用SVM训练数据时，要先将训练数据打乱顺序，用来提高分类精度，这里就用到了伪随机序列。
    # random_ State: When using SVM training data, the training data should be disordered to improve the classification accuracy. 
    # Pseudo random sequences are used here.
    #'random_state': None,
    # -----------------------------------------------------------------------------------------
    # kernel is 'poly'
    # degree ('poly '): indicates the highest degree of the selected polynomial kernel function. Must be non negative.
    # 解释：degree（‘poly’）：表示选择的多项式核函数的最高次数。必须为非负。
    'degree': range(3, 5),
    # -----------------------------------------------------------------------------------------
    # kernel is 'poly'
    # coef0: constant value of kernel function (value b in 'y=kx+b'). 
    # Only 'poly' and 'sigmoid' kernel functions have it.
    # 解释：coef0：核函数常数值(y=kx+b中的b值)，只有‘poly’和‘sigmoid’核函数有
    'coef0': np.arange(0, 1, 0.1),
    # -----------------------------------------------------------------------------------------
    # multilabel
    # decision_ function_ Shape: The original SVM is only applicable to the binary classification problem. 
    # If you want to extend it to multi class classification, 
    # you need to adopt a certain fusion strategy. Here are three options.
    # 解释：decision_function_shape: 原始的SVM只适用于二分类问题，如果要将其扩展到多类分类，就要采取一定的融合策略，这里提供了三种选择。
    #'decision_function_shape': ['ovr','ovo'],
    # -----------------------------------------------------------------------------------------
    # multilabel
    # break_ties: 1) if it is true, decision_ function_ Shape='vr ', 
    # and the number of classes>2, then the prediction will break the relationship 
    # according to the confidence value of decision  ufunction;
    # 2) Otherwise, the first class in the binding class is returned. 
    # Note that the cost of breaking the link is relatively high compared to a simple forecast.
    # 解释：break_ties: 如果为真，decision_function_shape='vr'，且类数>2，则预测将根据decision\ufunction的置信值打破联系；
    # 否则返回绑定类中的第一个类。请注意，与简单预测相比，打破联系的计算成本相对较高。
    #'break_ties': [True, False],
    }]
```



### 网格搜索svc模型参数

```python
import time
# Set the parameters by cross-validation
tuned_parameters = [{
    # C: penalty coefficient, which is used to control the penalty coefficient of loss function, 
    # similar to the regularization coefficient in LR.
    # 1) The larger the C is, the greater the penalty for misclassification will be. 
    # The accuracy of training set testing is very high, but the generalization ability is weak, 
    # which is easy to lead to over fitting.
    # 2) The C value is small, the punishment for misclassification is reduced, 
    # the fault tolerance ability is enhanced, and the generalization ability is strong, 
    # but the fitting may also lead to under fitting.
    # 解释：C：惩罚系数，用来控制损失函数的惩罚系数，类似于LR中的正则化系数。
    # C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，
    # 这样会出现训练集测试时准确率很高，但泛化能力弱，容易导致过拟合。 
    # C值小，对误分类的惩罚减小，容错能力增强，泛化能力较强，但也可能欠拟合。
    'C': [0.001,0.01,0.1,1, 10, 100,1000],
    
    # cache_size: The buffer size is used to limit the calculation amount. The default value is 200M.
    # 缓冲大小，用来限制计算量大小，默认是200M。
    'cache_size': [50,100,200,300,400],
    
    # probability: whether to use probability estimation. The default value is False. 
    # It must be used before the fit () method, which will reduce the operation speed.
    # 解释：probability：是否使用概率估计，默认是False。必须在fit()方法前使用，该方法的使用会降低运算速度。
    'probability': [True, False],
    
    # tol: residual convergence condition, which is 0.0001 by default, 
    # that is, an error in 1000 classification is tolerated, which is consistent with LR; 
    # Stop training when the error item reaches the specified value.
    # 解释：tol：残差收敛条件，默认是0.0001，即容忍1000分类里出现一个错误，与LR中的一致；误差项达到指定值时则停止训练。
    'tol': [0.01,0.001,0.0001],
    
    # verbose: whether verbose output is enabled
    # 解释：verbose：是否启用详细输出
    #'verbose': [True, False],
    
    }]
# 解释：kernel(内核)：｛‘linear‘，‘poly‘，‘rbf‘，‘sigmoid‘，‘precomputed‘｝或可调用(or callable, default=’rbf’)，default=‘rbf‘ 
#'kernel': ['linear','poly','rbf'], 
clf = SVC(kernel='linear')

aa = time.time()
# CV in GridSearchCV are Cross Validation
clf = HalvingGridSearchCV(clf, # 估计器接口
                          param_grid=tuned_parameters, # 参数字典
                          n_jobs=-1, # 要并行运行的作业数。’-1‘表示使用所有处理器。
                          min_resources="exhaust",#  “exhaust”将设置r0，以便最后一次迭代使用尽可能多的资源。
                          factor=3 # `factor=3` 意味着只有三分之一的候选人被选中。 
                         )
#clf = GridSearchCV(clf, param_grid = grid_list, n_jobs = 4, cv = 3)
#clf = RandomizedSearchCV(estimator=clf, param_distributions=tuned_parameters, n_iter=200, n_jobs=-1,cv=3)

clf.fit(X_train, y_train)
bb = time.time() 
cc = bb-aa
print('run time:', cc)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)

print("Accuracy:",metrics.accuracy_score(y_test, clf.predict(X_test)))
```

```
run time: 2.0971462726593018
Best parameters set found on development set:

{'C': 0.1, 'cache_size': 400, 'probability': True, 'tol': 0.0001}
Accuracy: 0.9883040935672515
```

```python
import time
# Set the parameters by cross-validation
tuned_parameters = [{
    # C: penalty coefficient, which is used to control the penalty coefficient of loss function, 
    # similar to the regularization coefficient in LR.
    # 1) The larger the C is, the greater the penalty for misclassification will be. 
    # The accuracy of training set testing is very high, but the generalization ability is weak, 
    # which is easy to lead to over fitting.
    # 2) The C value is small, the punishment for misclassification is reduced, 
    # the fault tolerance ability is enhanced, and the generalization ability is strong, 
    # but the fitting may also lead to under fitting.
    # 解释：C：惩罚系数，用来控制损失函数的惩罚系数，类似于LR中的正则化系数。
    # C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，
    # 这样会出现训练集测试时准确率很高，但泛化能力弱，容易导致过拟合。 
    # C值小，对误分类的惩罚减小，容错能力增强，泛化能力较强，但也可能欠拟合。
    'C': [0.001,0.01,0.1,1, 10, 100,1000],
    
    # cache_size: The buffer size is used to limit the calculation amount. The default value is 200M.
    # 缓冲大小，用来限制计算量大小，默认是200M。
    'cache_size': [50,100,200,300,400],
    
    # probability: whether to use probability estimation. The default value is False. 
    # It must be used before the fit () method, which will reduce the operation speed.
    # 解释：probability：是否使用概率估计，默认是False。必须在fit()方法前使用，该方法的使用会降低运算速度。
    'probability': [True, False],
    
    # tol: residual convergence condition, which is 0.0001 by default, 
    # that is, an error in 1000 classification is tolerated, which is consistent with LR; 
    # Stop training when the error item reaches the specified value.
    # 解释：tol：残差收敛条件，默认是0.0001，即容忍1000分类里出现一个错误，与LR中的一致；误差项达到指定值时则停止训练。
    'tol': [0.01,0.001,0.0001],
    
    # Explanation: gamma: kernel function coefficient. 
    # This parameter is the kernel coefficient of rbf, poly and sigmoid; Default is' auto '
    # 1) The larger the gamma, σ The smaller the size is, 
    # the higher and thinner the Gaussian distribution is, 
    # resulting in the model can only act near the support vector, which may lead to over fitting;
    # 2) On the contrary, the smaller the gamma, σ The larger the size is, 
    # the smoother the Gaussian distribution will be, 
    # and the classification effect on the training set will be poor, which may lead to under fitting.
    # 解释：gamma：核函数系数，该参数是rbf，poly和sigmoid的内核系数；默认是'auto'
    # gamma越大，σ越小，使得高斯分布又高又瘦，造成模型只能作用于支持向量附近，可能导致过拟合；
    # 反之，gamma越小，σ越大，高斯分布会过于平滑，在训练集上分类效果不佳，可能导致欠拟合。
    'gamma': ['scale','auto'],# use for'rbf'
    
    # verbose: whether verbose output is enabled
    # 解释：verbose：是否启用详细输出
    #'verbose': [True, False],
    }]
# 解释：kernel(内核)：｛‘linear‘，‘poly‘，‘rbf‘，‘sigmoid‘，‘precomputed‘｝或可调用(or callable, default=’rbf’)，default=‘rbf‘ 
#'kernel': ['linear','poly','rbf'], 
clf = SVC(kernel='rbf')
#clf.estimator.get_params().keys()

aa = time.time()
clf = HalvingGridSearchCV(clf, # 估计器接口
                          param_grid=tuned_parameters, # 参数字典
                          n_jobs=-1, # 要并行运行的作业数。’-1‘表示使用所有处理器。
                          min_resources="exhaust",#  “exhaust”将设置r0，以便最后一次迭代使用尽可能多的资源。
                          factor=3 # `factor=3` 意味着只有三分之一的候选人被选中。 
                         )
#clf = GridSearchCV(clf, param_grid = grid_list, n_jobs = 4, cv = 3)
# GridSearchCV中的CV就是Cross Validation

#clf = RandomizedSearchCV(estimator=clf, param_distributions=tuned_parameters, n_iter=200, n_jobs=-1,cv=3)

clf.fit(X_train, y_train)
bb = time.time() 
cc = bb-aa
print('run time:', cc)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)

print("Accuracy:",metrics.accuracy_score(y_test, clf.predict(X_test)))
```

```
run time: 3.7974188327789307
Best parameters set found on development set:

{'C': 10, 'cache_size': 200, 'gamma': 'scale', 'probability': True, 'tol': 0.01}
Accuracy: 0.9883040935672515
```

```python
import time
# Set the parameters by cross-validation
tuned_parameters = [{
    # C: penalty coefficient, which is used to control the penalty coefficient of loss function, 
    # similar to the regularization coefficient in LR.
    # 1) The larger the C is, the greater the penalty for misclassification will be. 
    # The accuracy of training set testing is very high, but the generalization ability is weak, 
    # which is easy to lead to over fitting.
    # 2) The C value is small, the punishment for misclassification is reduced, 
    # the fault tolerance ability is enhanced, and the generalization ability is strong, 
    # but the fitting may also lead to under fitting.
    # 解释：C：惩罚系数，用来控制损失函数的惩罚系数，类似于LR中的正则化系数。
    # C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，
    # 这样会出现训练集测试时准确率很高，但泛化能力弱，容易导致过拟合。 
    # C值小，对误分类的惩罚减小，容错能力增强，泛化能力较强，但也可能欠拟合。
    'C': [0.001,0.01,0.1,1, 10, 100,1000],
    
    # cache_size: The buffer size is used to limit the calculation amount. The default value is 200M.
    # 缓冲大小，用来限制计算量大小，默认是200M。
    'cache_size': [50,100,200,300,400],
    
    # probability: whether to use probability estimation. The default value is False. 
    # It must be used before the fit () method, which will reduce the operation speed.
    # 解释：probability：是否使用概率估计，默认是False。必须在fit()方法前使用，该方法的使用会降低运算速度。
    'probability': [True, False],
    
    # tol: residual convergence condition, which is 0.0001 by default, 
    # that is, an error in 1000 classification is tolerated, which is consistent with LR; 
    # Stop training when the error item reaches the specified value.
    # 解释：tol：残差收敛条件，默认是0.0001，即容忍1000分类里出现一个错误，与LR中的一致；误差项达到指定值时则停止训练。
    'tol': [0.01,0.001,0.0001],
    
    # Explanation: gamma: kernel function coefficient. 
    # This parameter is the kernel coefficient of rbf, poly and sigmoid; Default is' auto '
    # 1) The larger the gamma, σ The smaller the size is, 
    # the higher and thinner the Gaussian distribution is, 
    # resulting in the model can only act near the support vector, which may lead to over fitting;
    # 2) On the contrary, the smaller the gamma, σ The larger the size is, 
    # the smoother the Gaussian distribution will be, 
    # and the classification effect on the training set will be poor, which may lead to under fitting.
    # 解释：gamma：核函数系数，该参数是rbf，poly和sigmoid的内核系数；默认是'auto'
    # gamma越大，σ越小，使得高斯分布又高又瘦，造成模型只能作用于支持向量附近，可能导致过拟合；
    # 反之，gamma越小，σ越大，高斯分布会过于平滑，在训练集上分类效果不佳，可能导致欠拟合。
    'gamma': ['scale','auto'],# use for'rbf'
    
    # degree ('poly '): indicates the highest degree of the selected polynomial kernel function. Must be non negative.
    # 解释：degree（‘poly’）：表示选择的多项式核函数的最高次数。必须为非负。
    'degree': range(3, 5),
    
    # coef0: constant value of kernel function (value b in 'y=kx+b'). 
    # Only 'poly' and 'sigmoid' kernel functions have it.
    # 解释：coef0：核函数常数值(y=kx+b中的b值)，只有‘poly’和‘sigmoid’核函数有
    'coef0': np.arange(0, 1, 0.1),
    
    # verbose: whether verbose output is enabled
    # 解释：verbose：是否启用详细输出
    #'verbose': [True, False],
    }]
# 解释：kernel(内核)：｛‘linear‘，‘poly‘，‘rbf‘，‘sigmoid‘，‘precomputed‘｝或可调用(or callable, default=’rbf’)，default=‘rbf‘ 
#'kernel': ['linear','poly','rbf'], 
clf = SVC(kernel='poly')
#clf.estimator.get_params().keys()

aa = time.time()
clf = HalvingGridSearchCV(clf, # 估计器接口
                          param_grid=tuned_parameters, # 参数字典
                          n_jobs=-1, # 要并行运行的作业数。’-1‘表示使用所有处理器。
                          min_resources="exhaust",#  “exhaust”将设置r0，以便最后一次迭代使用尽可能多的资源。
                          factor=3 # `factor=3` 意味着只有三分之一的候选人被选中。 
                         )
#clf = GridSearchCV(clf, param_grid = grid_list, n_jobs = 4, cv = 3)
# GridSearchCV中的CV就是Cross Validation

#clf = RandomizedSearchCV(estimator=clf, param_distributions=tuned_parameters, n_iter=200, n_jobs=-1,cv=3)

clf.fit(X_train, y_train)
bb = time.time() 
cc = bb-aa
print('run time:', cc)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)

print("Accuracy:",metrics.accuracy_score(y_test, clf.predict(X_test)))
```

```
run time: 63.88416814804077
Best parameters set found on development set:

{'C': 1, 'cache_size': 400, 'coef0': 0.5, 'degree': 4, 'gamma': 'scale', 'probability': True, 'tol': 0.0001}
Accuracy: 0.9824561403508771
```

```python
clf = SVC(kernel='rbf',tol=0.01, gamma='scale', cache_size=200, C=10, probability=True) # Linear Kernel
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
```

```python
clf = SVC(kernel='poly',tol=0.0001, gamma='scale', cache_size=400,degree=4,coef0=0.5, C=1, probability=True) # Linear Kernel
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
```

```python
clf = SVC(kernel='linear',tol=0.0001, cache_size=400, C=0.1) # Linear Kernel
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
```

```python
# use HalvingGridSearchCV,GridSearchCV,RandomizedSearchCV,
import time
# Set the parameters by cross-validation
tuned_parameters = [{
    # C: penalty coefficient, which is used to control the penalty coefficient of loss function, 
    # similar to the regularization coefficient in LR.
    # 1) The larger the C is, the greater the penalty for misclassification will be. 
    # The accuracy of training set testing is very high, but the generalization ability is weak, 
    # which is easy to lead to over fitting.
    # 2) The C value is small, the punishment for misclassification is reduced, 
    # the fault tolerance ability is enhanced, and the generalization ability is strong, 
    # but the fitting may also lead to under fitting.
    # 解释：C：惩罚系数，用来控制损失函数的惩罚系数，类似于LR中的正则化系数。
    # C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，
    # 这样会出现训练集测试时准确率很高，但泛化能力弱，容易导致过拟合。 
    # C值小，对误分类的惩罚减小，容错能力增强，泛化能力较强，但也可能欠拟合。
    'C': [0.001,0.01,0.1,1, 10, 100,1000],
    
    # cache_size: The buffer size is used to limit the calculation amount. The default value is 200M.
    # 缓冲大小，用来限制计算量大小，默认是200M。
    'cache_size': [50,100,200,300,400],
    
    # probability: whether to use probability estimation. The default value is False. 
    # It must be used before the fit () method, which will reduce the operation speed.
    # 解释：probability：是否使用概率估计，默认是False。必须在fit()方法前使用，该方法的使用会降低运算速度。
    'probability': [True, False],
    
    # tol: residual convergence condition, which is 0.0001 by default, 
    # that is, an error in 1000 classification is tolerated, which is consistent with LR; 
    # Stop training when the error item reaches the specified value.
    # 解释：tol：残差收敛条件，默认是0.0001，即容忍1000分类里出现一个错误，与LR中的一致；误差项达到指定值时则停止训练。
    'tol': [0.01,0.001,0.0001],
    
    # verbose: whether verbose output is enabled
    # 解释：verbose：是否启用详细输出
    #'verbose': [True, False],
    
    }]
# 解释：kernel(内核)：｛‘linear‘，‘poly‘，‘rbf‘，‘sigmoid‘，‘precomputed‘｝或可调用(or callable, default=’rbf’)，default=‘rbf‘ 
#'kernel': ['linear','poly','rbf'], 


# CV in GridSearchCV are Cross Validation
clfs_name=['GridSearchCV','HalvingGridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']
for i in range(4):
    print('search strategy:',clfs_name[i])
    clf = SVC(kernel='linear')
    aa = time.time()
    if i == 0 :
        clf = GridSearchCV(clf, param_grid = tuned_parameters, n_jobs = -1, cv = 3)
    
    elif i == 1:
        clf = HalvingGridSearchCV(clf, # 估计器接口
                          param_grid=tuned_parameters, # 参数字典
                          n_jobs=-1, # 要并行运行的作业数。’-1‘表示使用所有处理器。
                          min_resources="exhaust",#  “exhaust”将设置r0，以便最后一次迭代使用尽可能多的资源。
                          factor=3 # `factor=3` 意味着只有三分之一的候选人被选中。 
                         )
    elif i == 2:
        clf = RandomizedSearchCV(estimator=clf, param_distributions=tuned_parameters, n_iter=200, n_jobs=-1,cv=3)
    elif i == 3:
        clf = HalvingRandomSearchCV(clf, # 估计器接口
                          param_distributions=tuned_parameters, # 参数字典
                          n_jobs=-1, # 要并行运行的作业数。’-1‘表示使用所有处理器。
                          # min_resources="exhaust",#  “exhaust”将设置r0，以便最后一次迭代使用尽可能多的资源。
                          factor=3 # `factor=3` 意味着只有三分之一的候选人被选中。 
                         )
    else:
        raise ValueError('out of cycle')
    clf.fit(X_train, y_train)
    bb = time.time() 
    cc = bb-aa
    
    print('run time:', cc)
    print()
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Accuracy:",metrics.accuracy_score(y_test, clf.predict(X_test)))
    print()
```

```
search strategy: GridSearchCV
run time: 1.48978853225708

Best parameters set found on development set:
{'C': 0.1, 'cache_size': 50, 'probability': True, 'tol': 0.01}
Accuracy: 0.9883040935672515

search strategy: HalvingGridSearchCV
run time: 1.621852159500122

Best parameters set found on development set:
{'C': 0.1, 'cache_size': 200, 'probability': True, 'tol': 0.0001}
Accuracy: 0.9883040935672515

search strategy: RandomizedSearchCV
run time: 1.4543657302856445

Best parameters set found on development set:
{'tol': 0.01, 'probability': True, 'cache_size': 50, 'C': 0.1}
Accuracy: 0.9883040935672515

search strategy: HalvingRandomSearchCV
run time: 0.2953622341156006

Best parameters set found on development set:
{'tol': 0.01, 'probability': True, 'cache_size': 100, 'C': 0.1}
Accuracy: 0.9883040935672515
```

HalvingRandomSearchCV 速度最快，差5倍时间

### Plot different SVM classifiers

```python
def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy
```

```python
def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.
    绘制分类器的决策边界

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
```

```python
# import some data to play with
breast_cancer = datasets.load_breast_cancer()

# Take the first two features. We could avoid this by using a two-dim dataset
# 只选两列数据
X = breast_cancer.data[:, :2]
y = breast_cancer.target

# Standarize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (
    SVC(kernel="linear", C=C),
    LinearSVC(C=C, max_iter=10000),
    SVC(kernel="rbf", gamma=0.7, C=C, ),
    SVC(kernel="poly", degree=3, gamma="auto", C=C),
)
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = (
    "SVC with linear kernel",
    "LinearSVC (linear kernel)",
    "SVC with RBF kernel",
    "SVC with polynomial (degree 3) kernel",
)

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel("Sepal length")
    ax.set_ylabel("Sepal width")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
```

![image-20230109153852442](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230109153852442.png)

## 算法的使用补充资料：

1. 数据：[sklearn.datasets.**load_breast_cancer**](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

2. 特征降维选用：[sklearn.decomposition.**KernelPCA**](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA)

   1. [**PCA算法 | 数据集特征数量太多怎么办？用这个算法对它降维打击！**](https://blog.51cto.com/u_15183480/2747077)

   2. [常用降维方法的总结](https://welts.xyz/2022/03/17/rd/)：

      1. 主成分分析（PCA）【尝试找到数据集中的主成分 (实际上就是一个向量组)，然后用这些主成分重新描述数据。】，

      2. 独立成分分析（ICA）【一开始用于解决盲信号分离问题。】，

      3. 线性判别分析（LDA）【属于有监督的降维，它的目标是降维之后，同类样本尽可能的近，异类样本尽可能远。】，

      4. 非负矩阵分解（NMF）【NLP的词袋模型，N-gram模型，以及延伸出来的TF-IDF，虽然让数据高度稀疏化，但保证数据集矩阵的非负性。因此NMF很好契合了这些模型的特点。对于单词-文档矩阵，使用NMF进行分解，能够同时得到单词向量和文档向量的稀疏表示。】，

      5. 随机投影（一种新的降维技术，主要用于高容量数据集或高维特征空间。），

      6. 随机傅里叶特征（RFF）【主要是针对的是核函数 (移位不变核)的降维。核函数会将数据映射到甚高维，但随机傅里叶特征通过蒙特卡洛抽样将数据从甚高维降到相对的低维。】，

      7. Johnson Lindenstrauss Lemma，

         <img src="C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230108233011644.png" alt="image-20230108233011644" style="zoom: 80%;" />

      8. 自编码器（Autoencoder）【自编码器是一种神经网络，它被训练去试图将其输入复制到其输出，自编码器作为一种降维方法取得了巨大的成功。Autoencoder的雏形是一个单隐层的神经网络，我们想将这个神经网络训练成一个单位映射，也就是输入和输出相同。如果我们将单隐层神经元数目设置成比输入要小，则可以将其隐层的数据作为输入数据的降维表示。】，

         **卷积自编码器**将卷积神经网络思想融入了自编码器。也就是数据输入后，经过卷积，池化后形成了低维向量，这是编码器部分。然后通过反池化和反卷积操作，将降维后的数据进行还原，这是解码器部分。事实上卷积自编码器在处理图像降噪上有不小的贡献。

      9. t-SNE (T-distributed Stochastic Neighbor Embedding）【一种非线性降维方法，特别适用于高维数据集的可视化。该方法将高维的欧几里得距离转换成表征相似度的条件概率。】

   3. [四大降维算法的比较和一些理解（PCA、LDA、LLE、LEP）](https://blog.csdn.net/weixin_43909872/article/details/85415399)

      1. PCA：主成分分析（Principle components analysis） 【它的目标是通过某种线性投影，将高维的数据映射到低维的空间中表示，并期望在所投影的维度上数据的方差最大，以此使用较少的数据维度，同时保留住较多的原数据点的特性。】
      2. LDA：线性判别分析（Linear Discriminant Analysis）【一种有监督的（supervised）线性降维算法。与PCA保持数据信息不同，LDA是为了使得降维后的数据点尽可能地容易被区分！】 
      3. LLE：局部线性嵌入（Locally linear embedding ）【一种非线性降维算法，它能够使降维后的数据较好地保持原有流形结构。LLE可以说是流形学习方法最经典的工作之一。】
      4. LEP：拉普拉斯特征映射（Laplacian Eigenmaps）【从局部的角度去构建数据之间的关系，一种基于图的降维算法，它希望相互间有关系的点（在图中相连的点）在降维后的空间中尽可能的靠近，从而在降维后仍能保持原有的数据结构。】

   4. [【机器学习算法系列之三】简述多种降维算法](【机器学习算法系列之三】简述多种降维算法)

      ![image-20230108233813663](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230108233813663.png)

      1. [3.1 主成分分析PCA](https://chenrudan.github.io/blog/2016/04/01/dimensionalityreduction.html#3.1)
      2. [3.2 多维缩放(MDS)](https://chenrudan.github.io/blog/2016/04/01/dimensionalityreduction.html#3.2)
      3. [3.3 线性判别分析(LDA)](https://chenrudan.github.io/blog/2016/04/01/dimensionalityreduction.html#3.3)
      4. [3.4 等度量映射(Isomap)](https://chenrudan.github.io/blog/2016/04/01/dimensionalityreduction.html#3.4)
      5. [3.5 局部线性嵌入(LLE)](https://chenrudan.github.io/blog/2016/04/01/dimensionalityreduction.html#3.5)
      6. [3.6 t-SNE](https://chenrudan.github.io/blog/2016/04/01/dimensionalityreduction.html#3.6)
      7. [3.7 Deep Autoencoder Networks](https://chenrudan.github.io/blog/2016/04/01/dimensionalityreduction.html#3.7)

   5. [Python sklearn.cross_decomposition.CCA实例讲解](http://www.manongjc.com/detail/31-ozzusprfxebhfkt.html)

3. scikit-learn 库：[scikit-learn的六大模块](https://scikit-learn.org/stable/)

   1. [Classification](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)：识别对象所属的类别。 应用：垃圾邮件检测，图像识别。 算法：支持向量机、最近邻、随机森林等。。。
   2. [Regression](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)：预测与对象关联的连续值属性。 应用：药物反应，股价。 算法：SVR、最近邻、随机森林等。。。
   3. [Clustering](https://scikit-learn.org/stable/modules/clustering.html#clustering)：将相似对象自动分组到集合中。 应用：客户细分、分组实验结果 算法：k-Means、谱聚类、均值偏移等。。。
   4. [Dimensionality reduction](https://scikit-learn.org/stable/modules/decomposition.html#decompositions)：减少要考虑的随机变量的数量。 应用：可视化，提高效率 算法：PCA、特征选择、非负矩阵分解等。。。
   5. [Model selection](https://scikit-learn.org/stable/model_selection.html#model-selection)：比较、验证和选择参数和模型。 应用：通过参数调整提高精度 算法：网格搜索、交叉验证、度量等。。。
   6. [Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing)：特征提取和归一化。 应用：转换输入数据，如文本，用于机器学习算法。 算法：预处理、特征提取等。。。

4. 算法模块：[API Reference](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)

   这是scikit学习的类和函数参考。有关详细信息，请参阅完整的用户指南，因为类和函数的原始规范可能不足以提供有关其使用的完整指南。有关在API中重复的概念的参考，请参阅通用术语表和API元素。

5. [python sklearn 线性回归 报错_sklearn11_函数汇总](https://blog.csdn.net/weixin_32001071/article/details/113639350)

6. [`sklearn.svm`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm): Support Vector Machines

   [`sklearn.svm`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)模块包括支持向量机算法。 

   用户指南：有关详细信息，请参阅支持向量机部分（ [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html#svm) ）。

   *SVC*与*NuSVC*是类似的方法,但是接受稍微不同的参数集合并具有不同的数学公式,并且*NuSVC*可以使用参数来控制支持向量的个数。*SVC*和*NuSVC*方法基本一致,唯一区别就是损失函数的度量方式不同(*NuSVC*中的nu参数和*SVC*中的C参数)。

   LinearSVC是实现线性核函数的支持向量分类，没有kernel参数，也缺少一些方法的属性，如support_等。

   | Estimators：                                                 | Name                                  |
   | ------------------------------------------------------------ | ------------------------------------- |
   | [`svm.LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)([penalty, loss, dual, tol, C, ...]) | Linear Support Vector Classification. |
   | [`svm.LinearSVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR)(*[, epsilon, tol, C, loss, ...]) | Linear Support Vector Regression.     |
   | [`svm.NuSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC)(*[, nu, kernel, degree, gamma, ...]) | Nu-Support Vector Classification.     |
   | [`svm.NuSVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html#sklearn.svm.NuSVR)(*[, nu, C, kernel, degree, gamma, ...]) | Nu Support Vector Regression.         |
   | [`svm.OneClassSVM`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM)(*[, kernel, degree, gamma, ...]) | Unsupervised Outlier Detection.       |
   | [`svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)(*[, C, kernel, degree, gamma, ...]) | C-Support Vector Classification.      |
   | [`svm.SVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR)(*[, kernel, degree, gamma, coef0, ...]) | Epsilon-Support Vector Regression.    |

   | [`svm.l1_min_c`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.l1_min_c.html#sklearn.svm.l1_min_c)(X, y, *[, loss, fit_intercept, ...]) | Return the lowest bound for C. |
   | ------------------------------------------------------------ | ------------------------------ |

7.  画svm超平面：

   1. [SVM Margins Example](https://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html#sphx-glr-auto-examples-svm-plot-svm-margin-py)
   2. [SVM 分类器的分类超平面的绘制](https://blog.csdn.net/ericcchen/article/details/79332781)
   3. [支持向量机+sklearn绘制超平面](https://www.zhihu.com/column/p/237701627)【sklearn画出超平面步骤】

8. SVM理论解释：

   1. [1.4. Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html#svm) 【支持向量机（SVM）是一组用于分类( [classification](https://scikit-learn.org/stable/modules/svm.html#svm-classification))、回归([regression](https://scikit-learn.org/stable/modules/svm.html#svm-regression))和异常值检测( [outliers detection](https://scikit-learn.org/stable/modules/svm.html#svm-outlier-detection))的监督学习方法。】

   2. [SVM简介及sklearn参数](https://www.cnblogs.com/solong1989/p/9620170.html)

      使用SVM作为模型时，通常采用如下流程：

      1. 对样本数据进行归一化
      2. 应用核函数对样本进行映射（最常采用和核函数是RBF和Linear，在样本线性可分时，Linear效果要比RBF好）
      3. 用cross-validation和grid-search对超参数进行优选
      4. 用最优参数训练得到模型
      5. 测试

9. 交叉验证（CV）：GridSearchCV包含CV

   1. [机器学习笔记（十七）：交叉验证](https://www.freesion.com/article/79111078415/)
   
10. 搜索最优模型：

    ### Hyper-parameter optimizers

    | [`model_selection.GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)(estimator, ...) | Exhaustive search over specified parameter values for an estimator. |
    | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | [`model_selection.HalvingGridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html#sklearn.model_selection.HalvingGridSearchCV)(...[, ...]) | Search over specified parameter values with successive halving. |
    | [`model_selection.ParameterGrid`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html#sklearn.model_selection.ParameterGrid)(param_grid) | Grid of parameters with a discrete number of values for each. |
    | [`model_selection.ParameterSampler`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterSampler.html#sklearn.model_selection.ParameterSampler)(...[, ...]) | Generator on parameters sampled from given distributions.    |
    | [`model_selection.RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV)(...[, ...]) | Randomized search on hyper parameters.                       |
    | [`model_selection.HalvingRandomSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html#sklearn.model_selection.HalvingRandomSearchCV)(...[, ...]) | Randomized search on hyper parameters.                       |

    1. [`sklearn.model_selection`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection).HalvingGridSearchCV的[HalvingGridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html#sklearn.model_selection.HalvingGridSearchCV)

    2. [交叉验证（Cross Validation）与网格搜索（Grid Search）-代码收集对比](https://www.jianshu.com/p/705d6ee40903)

    3. [使用Scikit-Learn的HalvingGridSearchCV进行更快的超参数调优](https://baijiahao.baidu.com/s?id=1700962429896647716&wfr=spider&for=pc)

    4. 两个实验性超参数优化器类（在model_selection模块中）：HalvingGridSearchCV和HalvingRandomSearchCV。

       他们的连续二分搜索策略并不是独立搜索超参数集候选项，而是“开始用少量资源评估所有候选项，并使用越来越多的资源迭代地选择最佳候选项。”默认资源是样本的数量，但用户可以将其设置为任何正整数模型参数，如梯度增强轮。因此，减半方法具有在更短的时间内找到好的超参数的潜力。

11. 可视化：seaborn绘图功能概述](https://seaborn.pydata.org/tutorial/function_overview.html)

12. [十分钟掌握Seaborn，进阶Python数据可视化分析](https://zhuanlan.zhihu.com/p/49035741)

13. 补充：[np.where()的使用方法](https://blog.csdn.net/island1995/article/details/90200151)

14. 评分：[【SKLEARN】classification_report函数与confusion_matrix函数](https://blog.csdn.net/m0_58810879/article/details/121808046)

    classification_report函数:

    ![image-20230108235458089](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230108235458089.png)

    confusion_matrix函数:

    |           | 真实值为x | 真实值为y |
    | --------- | --------- | --------- |
    | 预测值为x | data1     | data2     |
    | 预测值为y | data3     | data4     |

15. 