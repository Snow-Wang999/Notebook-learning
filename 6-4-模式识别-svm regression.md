# 6-4-模式识别-svm regression

## 联系老师：

bragin@tpu.ru

https://zoom.us/j/4360817882?pwd=VXB5LzhHaSttcEptcDNMdHVWUWVFQT09#success

https://drive.google.com/drive/folders/1ulKZJ1iI6HnGMHq6ILeWhMwm5vhMaaOJ

还有一个实践作业

## 工作目标：

熟悉支持向量方法在回归问题中的应用。

## 实验室工作任务：

基于给定数据集之一的支持向量机构建回归模型。研究分类器内核的选择和其他参数对模型结果的影响。报告已完成的工作。

### 模型学习数据集：

`sklearn.datasets.load_diabetes`
#`sklearn.datasets.load_boston`

```
class sklearn.svm.SVR(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=- 1)
```

## 理论部分：

假设有一组数据分为两类

![image-20221228200252859](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221228200252859.png)

SVM训练的任务是在这些类之间划出边界，以便这些类不会相互混合。在这些点之间可以画无限多条线。以下是两条可能的路线：

![image-20221228200327459](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221228200327459.png)

这两条线将两个类分开，并且不混合线两侧的类。这些线称为超平面。

绘制线（将类分开）是SVM的整个想法。
SVM算法在这方面与其他分类算法不同。SVM选择分隔类的线（或超平面）。SVM选择一个超平面，该超平面与两侧的边界点具有最大距离，这意味着在训练SVM时，SVM考虑到超平面的最近点，如下图所示：

![image-20221228200402787](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221228200402787.png)

这个超平面与两侧最近点的最大距离。这两个点（上图中阴影区域中的两个点）是支持最优超平面的点，因此这些点被称为参考向量。参考向量之间的距离称为margin。

因此，SVM算法试图最大化Margin，以确保类之间的最佳分离。

基本超平面（黑线）被称为解的边界，以区别于左右两个其他超平面。

### 超参数C

通过允许某些错误（或错误分类），可以使用一个称为“C”的参数来控制错误的数量。它可以是任何值，例如0.01，甚至100或更多。这取决于问题的类型和可用的数据。

“C”直接影响超平面。“C”与字段宽度成反比。因此，C越大，Margin就越小，反之亦然。

支持向量机也可以用作回归方法，保留算法的所有基本特征（最大化边缘化）。目标是在参考向量的线之间放置尽可能多的点，同时限制线外的点。

![image-20221228200526140](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221228200526140.png)

解决：
$$
min\frac{1}{2}\parallel{w}\parallel^2
$$
约束：
$$
y_i-wx_i-b \leq \varepsilon
$$

$$
wx_i+b-y_i \leq \varepsilon
$$

![image-20221228202027187](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221228202027187.png)

最小化：
$$
\frac{1}{2}\parallel{w}\parallel^2+C{\sum^{N}_{i=1}(\xi_i+\xi_i^*)}
$$
约束：
$$
y_i-wx_i-b \leq \varepsilon+\xi_i
$$

$$
wx_i+b-y_i \leq \varepsilon+\xi_i^*
$$

$$
\xi_i,\xi_i^* \geq 0
$$

![image-20221228202525889](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221228202525889.png)

### 支持向量回归模型

[基于SVM的糖尿病数据集回归问题](https://blog.csdn.net/m0_37758063/article/details/124086219)

SVM（Support Vector Machine）又称为支持向量机，最初是一种二分类的模型，后来修改之后也是可以用于多类别问题的分类。支持向量机可以分为线性核非线性两大类。其主要思想为找到空间中的一个更够将所有数据样本划开的超平面，并且使得数据集中所有数据到这个超平面的距离最短。支持向量也可以用于回归，此时叫支持向量回归（Support Vector Regression,简称SVR）。

回归就像是寻找一堆数据的内在的关系。不论这堆数据有几种类别组成，得到一个公式，拟合这些数据，当给个新的坐标值时，能够求得一个新的值。所以对于SVR，就是求得一个面或者一个函数，可以把所有数据拟合了（就是指所有的数据点，不管属于哪一类，数据点到这个面或者函数的距离最近）

统计上的理解就是：使得所有的数据的类内方差最小，把所有的类的数据看作是一个类。

传统的回归方法当且仅当回归f(x)完全等于y时才认为是预测正确，需计算其损失；而支持向量回归(SVR)则认为只要是f(x)与y偏离程度不要太大，既可认为预测正确，不用计算损失。具体的就是设置一个阈值α，只是计算 |f(x) - y| > α 的数据点的loss。支持向量回归表示只要在虚线内部的值都可认为是预测正确，只要计算虚线外部的值的损失即可。

支持向量回归模型（Support Vector Regression，SVR）是使用SVM来拟合曲线，做回归分析。与分类的输出是有限个离散的值不同的是，回归模型的输出在一定范围内是连续的。与SVM是使用一个条带来进行分类一样，SVR也是使用一个条带来拟合数据。这个条带的宽度可以自己设置，利用参数ϵ来控制。有一点和SVM是正好相反的：SVR希望样本点都落在“隔离带”内，而SVM希望样本点都在“隔离带”外。如图：

![image-20230110160859030](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230110160859030.png)

### 加载库

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats # 统计学库
from scipy.stats import norm # 用于拟合正太分布曲线

from sklearn import datasets,ensemble
from sklearn.datasets import load_diabetes

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

# 设置风格尺度
plt.style.use('seaborn-whitegrid') # seaborn 主题
sns.set_style('white')

```



#### 超参数优化包

```python
!pip install Optunity
```

optunity是一个包含用于超参数优化的各种优化器的库。 超参数整定是许多机器学习任务中经常遇到的问题， 既有监督又无监督。优化示例包括优化 正则化或核参数。

从优化的角度来看，调谐问题可以被认为是 如下：目标函数是非凸的、不可微的和 评估通常很昂贵。

这个包提供了几种不同的方法来解决这些问题，包括 一些有用的工具，如交叉验证和大量的分数函数。

optunity库是用python实现的，它允许 其他机器学习环境的集成，包括R和Matlab。

```python
import math
import itertools
import optunity
import optunity.metrics
import sklearn.svm
```



## 加载数据集

```python
#dataset to dataframe
# 数据里不包含object数据
def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df
```

```python
# Load the diabetes dataset
#diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
df_diabetes = sklearn_to_df(load_diabetes())
df_diabetes.info()
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 442 entries, 0 to 441
Data columns (total 11 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   age     442 non-null    float64
 1   sex     442 non-null    float64
 2   bmi     442 non-null    float64
 3   bp      442 non-null    float64
 4   s1      442 non-null    float64
 5   s2      442 non-null    float64
 6   s3      442 non-null    float64
 7   s4      442 non-null    float64
 8   s5      442 non-null    float64
 9   s6      442 non-null    float64
 10  target  442 non-null    float64
dtypes: float64(11)
memory usage: 38.1 KB
```

### 数据集详细介绍

原文链接：https://blog.csdn.net/m0_52896752/article/details/127681957

diabetes 是一个关于糖尿病的数据集， 该数据集包括442个病人的生理数据及一年以后的病情发展情况。

该数据集共442条信息，特征值总共10项, 如下:

- age: 年龄

- sex: 性别

- bmi(body mass index): 身体质量指数，是衡量是否肥胖和标准体重的重要指标，理想BMI(18.5~23.9) = 体重(单位Kg) ÷ 身高的平方 (单位m)

- bp(blood pressure): 血压（平均血压）

- s1,s2,s3,s4,s4,s6:六种血清的化验数据，是血液中各种疾病级数指针的6的属性值。
  - s1——tc，T细胞（一种白细胞）
  - s2——ldl，低密度脂蛋白
  - s3——hdl，高密度脂蛋白
  - s4——tch，促甲状腺激素
  - s5——ltg，拉莫三嗪
  - s6——glu，血糖水平

【注意】：以上的数据是经过特殊处理， 10个数据中的每个都做了均值中心化处理，然后又用标准差乘以个体数量调整了数值范围。验证就会发现任何一列的所有数值平方和为1。

这10个特征变量中的每一个都以平均值为中心，并按标准差乘以“n_samples”(即每列的平方和总计为1)进行缩放。

### 实验要求：

一、加载糖尿病数据集diabetes，观察数据

1.载入糖尿病情数据库diabetes，查看数据。

2.切分数据，组合成DateFrame数据，并输出数据集前几行，观察数据。
二、基于线性回归对数据集进行分析

3.查看数据集信息，从数据集中抽取训练集和测试集。

4.建立线性回归模型，训练数据，评估模型。
三、考察每个特征值与结果之间的关联性，观察得出最相关的特征

5.考察每个特征值与结果之间的关系，分别以散点图展示。

思考：根据散点图结果对比，哪个特征值与结果之间的相关性最高？
四、使用回归分析找出XX特征值与糖尿病的关联性，并预测出相关结果

6.把5中相关性最高的特征值提取，然后进行数据切分。

8.创建线性回归模型，进行线性回归模型训练。

9.对测试集进行预测，求出权重系数。

10.对预测结果进行评价，结果可视化。

### 目标变量转换

”target“是我们需要预测的目标变量，下面对”target“做一些分析。用正太分布去拟合”target“，同时做其正太概率图图可以发现目标变量呈现右偏态分布。

```python
def norm_comparision_plot(data,title,figsize=(12,10),color="#099DD9",
                          ax=None,surround=True,grid=True):
    '''
    function: 传入 DataFrame 指定行，绘制其概率分布曲线与正太分布曲线（比较）
    color：默认为标准天蓝 #F79420: 浅橙 ‘green’：直接绿色（透明度自动匹配）
    ggplot 经典三原色：‘#F77B72’：浅红，‘#7885CB’：浅紫，‘#4CB5AB’：浅绿
    ax=None：默认无需绘制子图的效果
    surround：sns.despine 的经典组合，默认开启，需要显示关闭
    grid：是否添加网格线，默认开启，需显示关闭
    '''
    plt.figure(figsize=figsize) # 设置图片大小
    # distplot 有可能在未来消失
    sns.distplot(data, \
                 fit=norm, \
                 color=color,\
                 kde_kws={"color": color,"lw":3}, \
                 ax=ax)
    # fit=norm: 同等条件下的正太曲线（默认黑色线）
    #lw--line width 线宽
    (mu, sigma) = norm.fit(data)# 求同等条件下正太分布的mu和sigma
     # 添加图例：使用格式化输入
    plt.legend(['Normal dist.($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu,sigma)],loc='best')
    # loc='best':表示自动将图例放到最合适的位置 
    plt.ylabel('Frequency')
    plt.title("Distribution")
    if surround == True:
        # trim=True - 隐藏上面根右边的边框线，left=True - 隐藏左边的边框线
        # offset： 偏移量， x 轴向下偏移，更加美观
        sns.despine(trim=True, left=True, offset=10)
    if grid == True:
        plt.grid(True) # 添加网格线
    plt.savefig(title)
    plt.show()
```

```python
norm_comparision_plot(df_diabetes['target'],'target_distribution_curve(Probability_Positive).png',figsize=(12,10),color="#099DD9",
                      ax=None,surround=True,grid=True)
```

![target_distribution_curve(Probability_Positive)](D:\00研二上\模式识别-2\IRM\SVM_regression\target_distribution_curve(Probability_Positive).png)

由上图可见概率分布曲线与正太分布曲线不一致，目标数据是非正态分布数据。因为大多数机器学习模型不能很好地处理非正态分布数据，应用log(1+x)变换来修正倾斜。

```python
df_diabetes['target']=np.log(df_diabetes['target'])
```

```python
norm_comparision_plot(df_diabetes['target'],'target_distribution_curve(Probability_Positive)_new_log.png',figsize=(12,10),color="#099DD9",
                      ax=None,surround=True,grid=True)
```

![target_distribution_curve(Probability_Positive)_new_log](D:\00研二上\模式识别-2\IRM\SVM_regression\target_distribution_curve(Probability_Positive)_new_log.png)

以下是我自己写的目标规范化处理：

```python
# Series normalization
# mean_norm(df_diabetes['target'])
t_origin = df_diabetes['target']
t_mean = df_diabetes['target'].mean()
t_std = df_diabetes['target'].std()
df_diabetes['target'] = (t_origin-t_mean) / t_std
```

```python
norm_comparision_plot(df_diabetes['target'],'target_distribution_curve(Probability_Positive)_new.png',figsize=(12,10),color="#099DD9",
                      ax=None,surround=True,grid=True)
```

![target_distribution_curve(Probability_Positive)_new](D:\00研二上\模式识别-2\IRM\SVM_regression\target_distribution_curve(Probability_Positive)_new.png)

### 偏度与峰值(Skewness and kurtosis)

原来的数据：

```python
print('Skewness:%f' % df_diabetes['target'].skew())
print('Kurtosis:%f' % df_diabetes['target'].kurt())
```

```
Skewness:0.440563
Kurtosis:-0.883057
```

log变换：

```python
print('Skewness:%f' % df_diabetes['target'].skew())
print('Kurtosis:%f' % df_diabetes['target'].kurt())
```

```
Skewness:-0.332567
Kurtosis:-0.817170
```



### 特征相关性

```python
def correlation_heatmap(df):
    correlations = df.corr()
    
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0,  fmt = '.2f',
                square=True, linewidths = 0.5, annot=True, 
                cbar_kws={"shrink":0.70}
               )
    plt.show()
```

```python
correlation_heatmap(df_diabetes)
```

![features_relation](D:\00研二上\模式识别-2\IRM\SVM_regression\features_relation.png)

从上述热力图可以看出一些明显的特征，如一年后患疾病的定量指标target和身体质量指数bmi正相关系数比较大，说明身体质量指数测试值如果高于正常值（18.5-23.9）的话，一年后患疾病的可能性就越大。同理，平均血压bp与一年后患疾病的定量指标target之间的相关性也比较强。

根据热力图，可以看到有些特征信息与一年后患疾病的的定量指标target相关性比较大，有些特征信息与一年后患疾病的的定量指标target相关性很小，因此可以将不相关特征信息进行剔除，只选取与一年后患疾病的的定量指标target相关性较大的特征信息进行回归预测。

```python
diabetes = load_diabetes()       #导入糖尿病数据集
X = diabetes.data           #影响一年后患疾病定量指标的特征信息数据
y = diabetes.target          #一年后患疾病的定量指标数据
name = diabetes['feature_names']      #特征信息名称
#数据处理
unsF = []      #次要特征下标
for i in range(len(name)):
    if  name[i] == 'age'or name[i] == 'sex' or name[i] == 's4'  or name[i] == 's2' or name[i] == 's3'or name[i] == 's6':
        continue
    unsF.append(i)
X = np.delete(X, unsF, axis=1)       #删除次要特征
```



---

这些相关类型中的每一种都存在于由0到1的值表示的频谱中，其中微弱或高度正相关的特征可以是0.5或0.7。如果存在强而完全的正相关，则用0.9或1的相关分值表示结果。

如果存在很强的负相关关系，则表示为-1。

如果你的数据集具有完全正或负的属性，那么模型的性能很可能会受到一个称为“多重共线性”的问题的影响。**多重共线性**发生在多元回归模型中的一个预测变量可以由其他预测变量线性预测，且预测精度较高。这可能导致歪曲或误导的结果。幸运的是，决策树和提升树算法天生不受多重共线性的影响。当它们决定分裂时，树只会选择一个完全相关的特征。然而，其他算法，如逻辑回归或线性回归，也不能避免这个问题，你应该在训练模型之前修复它。

如上图的热力图，s1和s2存在强而完全的正相关。容易出现多重共线性，使得支持向量机的算法被误导或扭曲。



```python
sns.set()
sns.pairplot(df_diabetes.loc[:,['s1','s2']],palette='Dark2',height=2.5)
plt.savefig('pairplot_diabetes_s1_s2.png')
plt.show()
```

![pairplot_diabetes_s1_s2](D:\00研二上\模式识别-2\IRM\SVM_regression\pairplot_diabetes_s1_s2.png)

从上图中也可验证，s1和s2存在强而完全的正相关。

可以选择删除或者变换它。

### 对特征进行规范化处理

diabetes的数据是经过特殊处理, 10个数据中的每个都做了均值中心化处理,然后又用标准差乘以个体数量调整了数值范围.验证就会发现任何一列的所有数值平方和为1。

[diabetes dataset](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html) 已经进行过规范化处理了，无需再规范化一次。

若是其他数据，可采用下列方法：

1. 直接返回dataframe

```python
# 分离特征和目标
# X,y = load_diabetes(return_X_y=True) # numpy arrays
X = df_diabetes.iloc[:,:10]
y = df_diabetes.iloc[:,9:10]
```

```python
# Standarize features for DataFrame
def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
```

```python
X_df_std = mean_norm(X)
```

2. 不返回dataframe，返回ndarray，再返回dataframe

```python
# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
# ndarray to dataframe
X_std = pd.DataFrame(X_std, columns=X.columns)
```

特征与目标的关系：

算不出来，之后用互信息计算过了

```python
print("Correlation with target:\n")
print(X_std.corrwith(y))
```

```
Correlation with target:

s6     1.0
age    NaN
bmi    NaN
bp     NaN
s1     NaN
s2     NaN
s3     NaN
s4     NaN
s5     NaN
sex    NaN
dtype: float64
```

### split the data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13
)
```

### 特征与目标的关系

特征与目标的相关性：这点比较显见，与目标相关性高的特征，应当优选选择。除方差法外，本文介绍的其他方法均从相关性考虑。

基于以上两点，特征选择 的常用方法有移除低方差的特征，卡方(Chi2)检验，Pearson相关系数，互信息和最大信息系数，距离相关系数，Wrapper，Embedded。
————————————————
版权声明：本文为CSDN博主「weixin_39873208」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_39873208/article/details/111347231

diabetes的数据是*连续变量*—这些变量具有位于某个区间的实际值。

```python
# from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    '''
    每个特征和标签之间的估计相互信息。
    
    mutual_info_regression：估计一个连续目标变量的互信息。
    	两个随机变量之间的互信息（MI）是非负值，用于衡量变量之间的依存关系。
    	当且仅当两个随机变量是独立的，并且等于较高的值意味着较高的依赖性时，它等于零。
		该函数依赖于非参数方法，该方法基于k-邻近邻居距离的熵估计。
    '''
    
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
```

#### mutual_info_regression 参数详解

[特征选择过滤器 - mutual_info_regression（连续目标变量的互信息）](https://blog.csdn.net/weixin_46072771/article/details/106188129)

```python
sklearn.feature_selection.mutual_info_regression(X, y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
```

```python
Parameters
----------
	X：array_like or sparse matrix, shape (n_samples, n_features)
  	   Feature matrix.
       特征矩阵。

	y：array_like, shape (n_samples,)
       Target vector.
       标签向量。

	discrete_features：{'auto', bool, array_like}, default ‘auto’
					   如果为'auto'，则将其分配给False（表示稠密）X，将其分配给True（表示稀疏）X。
			           如果是bool，则确定是考虑所有特征是离散特征还是连续特征。
			           如果是数组，则它应该是具有形状（n_features，）的布尔蒙版或具有离散特征索引的数组。

	n_neighbors: int, default=3
				 用于连续变量的MI估计的邻居数;
				 较高的值会减少估计的方差，但可能会带来偏差。

	copy: bool, default=True
	      是否复制给定的数据。如果设置为False，则初始数据将被覆盖。

	random_state: int, RandomState instance or None, optional, default None
				  确定随机数生成，以将小噪声添加到连续变量中以删除重复值。
				  在多个函数调用之间传递int以获得可重复的结果。
				  
Returns
-------
	mi: ndarray, shape (n_features,)
		每个特征和标签之间的估计相互信息。

```



```python
#所有连续特性现在都应该有浮点数数据类型（在使用MI之前，请仔细检查！）
discrete_features = X.dtypes == float
mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::1]  # show a few features with their MI scores
```

```
s6     3.915261
bmi    0.132010
s1     0.126048
bp     0.096786
s2     0.080600
s5     0.076013
age    0.057469
s4     0.038115
s3     0.024330
sex    0.015510
Name: MI Scores, dtype: float64
```

```python
def plot_mi_scores(scores,title):
    """画条形图
    """
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.figure(dpi=100, figsize=(8, 5))
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.savefig(title)
    plt.show()
```

```python
sns.set_style('white')
plot_mi_scores(mi_scores,"Mutual_Information_Scores.png")
```

![Mutual_Information_Scores](D:\00研二上\模式识别-2\IRM\SVM_regression\Mutual_Information_Scores.png)

```python
plt.figure(dpi=100, figsize=(10, 10))
sns.relplot(x="s6", y="target", data=df_diabetes)
plt.title("Scatter distribution")
plt.xticks(rotation = 45, fontsize = 10)
plt.savefig("Scatter_distribution_s6_and_target.png")
plt.show()
```

<img src="D:\00研二上\模式识别-2\IRM\SVM_regression\Scatter_distribution_s6_and_target.png" alt="Scatter_distribution_s6_and_target" style="zoom:50%;" />

![image-20230110145545218](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230110145545218.png)

### PCA for Decorrelation

```python
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_std)
```

```python
# Convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)
X_pca.head()
```

![image-20230110152341567](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230110152341567.png)

```python
loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=component_names,  # so the columns are the principal components
    index=X.columns,  # and the rows are the original features
)
loadings
```

![image-20230110152323998](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230110152323998.png)

```python
#还需进一步了解细节
def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs
```

```
# Look at explained variance
plot_variance(pca);
```

![image-20230110152301431](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230110152301431.png)

```python
mi_scores = make_mi_scores(X_pca, y, discrete_features=False)
mi_scores
```

![image-20230110152246476](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230110152246476.png)

mi分数为0或几乎为0，代表它与其他特征相互独立。所以”PC2“、”PC3“、”PC4“、”PC5“、”PC7“、”PC8“、”PC9“、”PC10“都是独立特征。所以，”PC1“和”PC6“与目标有一定的关系。

PC1= s2+s5+s4+s6-s3

PC6=s6-bmi-bp-age



### 子图

```
Three integers (nrows, ncols, index).  
The subplot will take the index position on a grid with nrows rows and ncols columns.  index starts at 1 in the upper left corner and increases to the right.  index can also be a two-tuple specifying the (first, last) indices (1-based, and including last) of the subplot,  e.g., fig.add_subplot(3, 1, (1, 2)) makes a subplot that spans the upper 2/3 of the figure. 
```

三个整数（nrows，ncols，index）。 

子图将在具有nrows行和ncols列的网格上占据索引位置。

索引从左上角的1开始，向右增加。 索引也可以是两元组， 例如，图add_subplot（3，1，（1，2））构成了一个横跨图的上2/3的子图。

### 评价

为了对建立好的模型进行性能评估，采用准确率、特异度和敏感度作为评价指标。其中，Acc代表测试集分类准确率（Accuracy）;Sen，代表测试集的灵敏度（Sensitivity）;即测试集分类的准确能力;Spe代表测试集的特异度（Specificity）。

Для оценки эффективности разработанных моделей в качестве оценочных показателей используются точность, специфичность и чувствительность.  Среди них Acc представляет точность классификации тестовых наборов (Accuracy);  Sen - чувствительность тестового набора (Sensitivity);  Точность классификации тестовых наборов;  Spe - это специфичность тестового набора (Specificity).

![image-20230112135526359](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230112135526359.png)

其中：TP（True Po.sitive）、FN（Fal.se Negative）、TN（TmeNegative）及FP（ Fal.se Positive）均针对测试数据集合。TP指将正例样本判断正确的数目;FN指将正例样本判断错误的数目;TN指将负例样本判断正确的数目;FP指将负例样本判断错误的数目。

r2_score()函数可以表示特征模型对特征样本预测的好坏，即确定系数，也称为拟合优度。拟合优度越大，自变量对因变量的解释程度越高，自变量引起的变动占总变动的百分比越高。

此处knn模型可更改为svr模型。

```python
#knn回归模型
knn = KNeighborsRegressor(n_neighbors=13)
knn.fit(X_train,y_train)           #训练数据,学习模型参数
y_predict = knn.predict(x_test)        #进行预测
对于超参数n_neigbors(邻居数)的选取，可以通过K折交叉验证法来获取最合适的n_neigbors值；如下列程序所示：
#使用网格来搜索候选值
from sklearn.model_selection import GridSearchCV    #通过网络方式来获取参数
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split    #导入数据集划分模块

diabetes = load_diabetes()     #导入糖尿病数据集
X = diabetes['data']
y = diabetes['target']

X_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#设置需要搜索的K值，'n_neightbors'是sklearn中KNN的参数
parameter = {
    
    'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]}
knn = KNeighborsRegressor()  #注意：这里不用指定参数

#通过GridSearchCV来搜索最好的K值。这个模块的内部其实就是对每一个K值进行评估
clf = GridSearchCV(knn,parameter,cv=5)      #5折
clf.fit(X_train,y_train)
print(f'评估最合适的K值为：{(clf.best_params_)["n_neighbors"]}',"其准确率为：%.2f"%clf.best_score_)
```



```python
#与验证值作比较
score = r2_score(y_test, y_predict).round(5)         #确定系数
print("Test set R^2:{}".format(score))

#可视化显示
plt.plot(y_test,label='true')
plt.plot(y_predict,label='knn_predict')
plt.legend()
plt.show()
```

## 实验工作

### 加载环境

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import time

from sklearn import datasets,svm,ensemble
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 模型评估，使用R-squared、MSE、MAE指标评估
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
```

### 加载数据集

```python
def load_data_regression():
    '''
    ------------------------------------------------------------------
    加载用于回归问题的数据集
    ------------------------------------------------------------------
    :return: 一个元组，用于回归问题。
    元组元素依次为：训练样本集、测试样本集、训练样本集对应的值、测试样本集对应的值
    ------------------------------------------------------------------
    Load data sets for regression problems
    ------------------------------------------------------------------
    : return: A tuple used for regression problems.
    The tuple elements are: 
        1. training sample set, 
        2. test sample set, 
        3. value corresponding to training sample set, and 
        4. value corresponding to test sample set
    ------------------------------------------------------------------
    '''
    #使用 scikit-learn 自带的一个糖尿病病人的数据集
    diabetes = load_diabetes() 
    X = diabetes.data
    y = diabetes.target
    # 拆分成训练集和测试集，测试集大小为原始数据集大小的 0.05
    return  train_test_split(X, y, test_size=0.05, random_state=33)
```

### 参数设置

svr参数初设

```python
params_svr = {
    "kernel": 'poly', 
    "degree": 3, 
    "gamma": 'scale', 
    "coef0": 0.0, 
    "tol": 0.001, 
    "C": 1.0, 
    "epsilon": 0.1, 
    "shrinking": True, 
    "cache_size": 200, 
    "verbose": False, 
    "max_iter": -1
}
```

梯度下降加速回归参数初设

```python
params_GBR = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}
```

### 主程序运行测试svr和gbr，设立基准点

```python
if __name__=="__main__":
    X_train, X_test, y_train, y_test = load_data_regression() # 生成用于回归问题的数据集
    
    # simple svr
    reg_svr = svm.SVR(**params_svr)
    reg_svr.fit(X_train, y_train)
    predict_svr = reg_svr.predict(X_test)
    mse_svr = mean_squared_error(y_test, predict_svr)
    mae_svr = mean_absolute_error(y_test, predict_svr)
    print("Model: Simple Support Vector Regressor")
    print('Score: %.2f' % reg_svr.score(X_test, y_test))# score函数返回方差
    print('R-squared value of SVR is', r2_score(y_test, predict_svr))# 一般来说,R-Squared 越大,表示模型拟合效果越好。
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse_svr))
    print("The mean absolute error (MAE) on test set: {:.4f}".format(mae_svr))
    print()
    # Gradient Boosting Regressor
    reg_GBR = ensemble.GradientBoostingRegressor(**params_GBR)
    reg_GBR.fit(X_train, y_train)
    predict_GBR = reg_GBR.predict(X_test)
    mse_GBR = mean_squared_error(y_test, predict_GBR)
    mae_GBR = mean_absolute_error(y_test, predict_GBR)
    print("Model: Gradient Boosting Regressor")
    print('Score: %.2f' % reg_GBR.score(X_test, y_test))# score函数返回方差
    print('R-squared value of Gradient Boosting Regressor is', r2_score(y_test, predict_GBR))
    # 一般来说,R-Squared 越大,表示模型拟合效果越好。
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse_GBR))
    print("The mean absolute error (MAE) on test set: {:.4f}".format(mae_GBR))
```

```
Model: Simple Support Vector Regressor
Score: 0.33
R-squared value of SVR is 0.3257356239903695
The mean squared error (MSE) on test set: 3928.7104
The mean absolute error (MAE) on test set: 52.0157

Model: Gradient Boosting Regressor
Score: 0.40
R-squared value of Gradient Boosting Regressor is 0.3980128205848601
The mean squared error (MSE) on test set: 3507.5756
The mean absolute error (MAE) on test set: 51.7565
```

### 寻找最佳参数

```python
if __name__=="__main__":
    X_train, X_test, y_train, y_test = load_data_regression() # 生成用于回归问题的数据集
​
    tuned_parameters=[{
        # 'C': Regularization parameter. The strength of the regularization is inversely proportional to C. 
        # Must be strictly positive. The penalty is a squared l2 penalty.
        'C': [0.001,0.01,0.1,1, 10, 100,1000],
        # 'tol': Tolerance for stopping criterion.
        'tol': [0.01,0.001,0.0001],
        # 'gamma': Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        # 'gamma': ['scale','auto'],
        'gamma': np.arange(0, 100, 10),
        # 'degree': Degree of the polynomial kernel function (‘poly’). Must be non-negative. Ignored by all other kernels.
        'degree': range(2, 5),
        # 'coef0': Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
        'coef0': np.arange(0, 1, 0.1),
        # 'epsilon': Epsilon in the epsilon-SVR model. 
        # It specifies the epsilon-tube within which no penalty is associated in the training loss function 
        # with points predicted within a distance epsilon from the actual value. Must be non-negative.
        'epsilon': np.arange(0, 10, 1)
    }]
​
    # 解释：kernel(内核)：｛‘linear‘，‘poly‘，‘rbf‘，‘sigmoid‘，‘precomputed‘｝或可调用(or callable, default=’rbf’)，default=‘rbf‘ 
    #'kernel': ['linear','poly','rbf'],
    regr = svm.SVR(kernel='poly')
​
    aa = time.time()
    # CV in GridSearchCV are Cross Validation
    regr = RandomizedSearchCV(estimator=regr, # 估计器接口
                              param_distributions=tuned_parameters, # 参数字典
                              n_iter=200, 
                              n_jobs=-1, # 要并行运行的作业数。’-1‘表示使用所有处理器。
                              cv=3)
    '''
    regr = GridSearchCV(regr, # 估计器接口
                        param_grid=tuned_parameters, # 参数字典
                        n_jobs=-1, # 要并行运行的作业数。’-1‘表示使用所有处理器。
                        cv = 3)'''
    regr.fit(X_train, y_train)
    bb = time.time() 
    cc = bb-aa
    mse_svr_poly = mean_squared_error(y_test, regr.predict(X_test))
    mae_svr_poly = mean_absolute_error(y_test, regr.predict(X_test))
    print('run time:', cc)
    print("Best parameters set found on development set:")
    print()
    print(regr.best_params_)
​
    print("Model: SVR Polynomial Kernel")
    print('Score: %.2f' % regr.score(X_test, y_test))# score函数返回方差
    print('R-squared value of Gradient Boosting Regressor is', r2_score(y_test, regr.predict(X_test)))
    # 一般来说,R-Squared 越大,表示模型拟合效果越好。
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse_svr_poly))
    print("The mean absolute error (MAE) on test set: {:.4f}".format(mae_svr_poly))
```

```
run time: 21.973620414733887
Best parameters set found on development set:

{'tol': 0.001, 'gamma': 20, 'epsilon': 4, 'degree': 2, 'coef0': 0.5, 'C': 10}
Model: SVR Polynomial Kernel
Score: 0.50
R-squared value of Gradient Boosting Regressor is 0.49974265417062835
The mean squared error (MSE) on test set: 2914.8303
The mean absolute error (MAE) on test set: 45.4251
```

线性svr的最高预测性能

### 类：测试参数对于svr线性模型的影响

```python
class ParamsTestLinearSVR:
    '''LinearSVR 的参数对它的预测性能的影响
    '''
    def __init__(self, data, losses, epsilons, Cs):
        '''
        初始化 ParamsTestLinearSVR 参数
        
        :param data: 已分割的数据集。可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = data
        self.losses = losses
        self.epsilons = epsilons
        self.Cs = Cs
        self.train_scores_epsilon = []
        self.test_scores_epsilon = []
        self.train_scores_C = []
        self.test_scores_C = []
        self.train_mse_epsilon = []
        self.test_mse_epsilon = []
        self.train_mse_C = []
        self.test_mse_C = []
    
    # 被test_LinearSVR，test_LinearSVR_loss调用的函数
    def fit_LinearSVR_score(self,regr):
        '''
        拟合 LinearSVR 曲线，输出预测的分数与相关系数与残差系数
        
        :param regr: 根据给定的训练数据拟合回归模型
        :return: None
        '''
        regr.fit(self.X_train,self.y_train)
        print('Coefficients:%s, intercept %s'%(regr.coef_,regr.intercept_)) 
        # coef_：分配给特征的权重（原始问题的系数），仅在线性内核的情况下可用。
        # intercept_：决策函数的截距
        print('Score: %.2f' % regr.score(self.X_test, self.y_test))# score函数返回方差
        mse = mean_squared_error(self.y_test, regr.predict(self.X_test))
        print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
        print()
    
    # 被test_LinearSVR_epsilon，test_LinearSVR_C调用的函数
    def fit_LinearSVR_scoreList_mseList(self,regr, train_scores, test_scores,train_mses,test_mses):
        '''
        拟合 LinearSVR 曲线，输出预测的分数的列表数据
        
        :param regr: 拟合曲线使用的回归模型。
        :param train_scores: 用于记录在训练集上训练的模型的预测分数的列表数据变量
        :param test_scores: 用于记录在测试集上训练的模型的预测分数的列表数据变量
        :param train_mses: 用于记录在训练集上训练的模型的mse分数的列表数据变量
        :param test_mses: 用于记录在测试集上训练的模型的mse分数的列表数据变量
        :return: None
        '''
        regr.fit(self.X_train,self.y_train)
        train_scores.append(regr.score(self.X_train, self.y_train))
        test_scores.append(regr.score(self.X_test, self.y_test))
        train_mses.append(mean_squared_error(self.y_test, regr.predict(self.X_test)))
        test_mses.append(mean_squared_error(self.y_test, regr.predict(self.X_test)))
        
    # 被test_LinearSVR_epsilon，test_LinearSVR_C调用的函数    
    def plot_scoreList(self, uniqueNum, fignrows,figncols, fignum, test_paramsList, 
                       train_scores, test_scores, title, xlabel, ylabel):
        '''
        画出回归模型同一个参数不同参数值，运行后预测分数列表的曲线
        
        :param uniqueNum: 画图的唯一标识
        :param fignrows: 画图的子图总行数
        :param figncols: 画图的子图总列数
        :param fignum: 画图的子图序数
        :param test_paramsList: 回归模型需要测试的参数的参数值的列表。
        :param train_scores: 用于记录在训练集上训练的模型的预测分数的列表数据变量
        :param test_scores: 用于记录在测试集上训练的模型的预测分数的列表数据变量
        :param params_name: 回归模型需要测试的参数的名字。
        :param xlabel: 回归曲线预测分数的图的横坐标的标签，即参数名。
        :param ylabel: 回归曲线预测分数的图的纵坐标的标签，即预测分数名。
        :return: None
        '''
        fig=plt.figure(num=uniqueNum,figsize=(20,20))
        ax=fig.add_subplot(fignrows,figncols,fignum)
        ax.plot(test_paramsList, train_scores, label =" Training score ", marker='+' )
        ax.plot(test_paramsList, test_scores, label = " Testing  score ", marker='o' )
        ax.set_title(title)# "epsilon","C" 
        ax.set_xscale("log")
        ax.set_xlabel(xlabel) #r"$\epsilon$", r"C"
        ax.set_ylabel(ylabel)#"score"
        if ylabel == "score":
            ax.set_ylim(-1,1.05)
        ax.legend(loc="best",framealpha=0.5)
        # plt.savefig(filename) #取消”#“，函数plot_scoreList需要添加参数”filename“
        # 设置 X 轴的网格线，风格为 点画线
        plt.grid(axis='x',linestyle='-.')
        plt.show()
        
    
    
    # 外部使用的函数
    def test_LinearSVR(self, ):
        '''
        测试 LinearSVR 的用法
        
        :return: None
        '''
        print("Test LinearSVR Without Parameters:")
        regr = svm.LinearSVR()
        self.fit_LinearSVR_score(regr)
    
    # 外部使用的函数
    def test_LinearSVR_loss(self,):
        '''
        测试 LinearSVR 的预测性能随不同损失函数loss的影响
​
        :return: None
        '''
        print("Test LinearSVR With Loss Parameter:")
        for loss in self.losses:
            print("loss：%s"%loss)
            regr=svm.LinearSVR(loss=loss)
            self.fit_LinearSVR_score(regr)
    
    # 外部使用的函数
    def test_LinearSVR_epsilon(self,loss='squared_epsilon_insensitive'):
        '''
        测试 LinearSVR 的预测性能随 epsilon 参数的影响
        
        :param loss: 测试参数 epsilon 时，固定loss参数，默认值：default='squared_epsilon_insensitive'
        :return: None
        '''
        print("Test LinearSVR With Epsilon Parameter:")
        for epsilon in  self.epsilons:
            regr=svm.LinearSVR(epsilon=epsilon,loss=loss)
            self.fit_LinearSVR_scoreList_mseList(regr, self.train_scores_epsilon, self.test_scores_epsilon,
                                                self.train_mse_epsilon, self.test_mse_epsilon)
        self.plot_scoreList(1,2,2,1,self.epsilons, self.train_scores_epsilon, self.test_scores_epsilon, 
                            "LinearSVR_epsilon", r"$\epsilon$","score")
        self.plot_scoreList(2,2,2,2,self.epsilons, self.train_mse_epsilon, self.test_mse_epsilon, 
                            "LinearSVR_epsilon", r"$\epsilon$","mean squared error (MSE)")
    
    # 外部使用的函数
    def test_LinearSVR_C(self, epsilon=0.1, loss='squared_epsilon_insensitive',max_iter=1000):
        '''
        测试 LinearSVR 的预测性能随 C 参数的影响
​
        :param epsilon: 测试参数 C 时， 固定epsilon参数，默认值： default=0.1
        :param loss: 测试参数 C 时，固定loss参数，默认值：default='squared_epsilon_insensitive'
        :return: None
        '''
        print("Test LinearSVR With Epsilon Parameter:")
        for C in self.Cs:
            regr=svm.LinearSVR(epsilon=epsilon,loss=loss,C=C,max_iter=max_iter)
            self.fit_LinearSVR_scoreList_mseList(regr, self.train_scores_C, self.test_scores_C,
                                                self.train_mse_C, self.test_mse_C)   
        self.plot_scoreList(3,2,2,1,self.Cs, self.train_scores_C, self.test_scores_C, 
                            "LinearSVR_C", r"C","score")
        self.plot_scoreList(4,2,2,2,self.Cs, self.train_mse_C, self.test_mse_C, 
                            "LinearSVR_C", r"C","mean squared error (MSE)")
```

### 主程序：测试参数对于svr线性模型的影响

```python
if __name__=="__main__":
    data = load_data_regression() # 生成用于回归问题的数据集
    # LinearSVR 模型的参数的参数值列表
    losses = ['epsilon_insensitive','squared_epsilon_insensitive']
    # 等比数列
    epsilons = np.logspace(-2,2)
    Cs = np.logspace(-2,2)
    test_1 = ParamsTestLinearSVR(data, losses, epsilons, Cs)
    test_1.test_LinearSVR() # 调用 test_LinearSVR
    test_1.test_LinearSVR_loss() # 调用 test_LinearSVR_loss
    test_1.test_LinearSVR_epsilon(loss='squared_epsilon_insensitive') # 调用 test_LinearSVR_epsilon
    test_1.test_LinearSVR_C(epsilon=0.1, loss='squared_epsilon_insensitive',max_iter=10000) # 调用 test_LinearSVR_C
```

```
Test LinearSVR Without Parameters:
Coefficients:[ 3.06892232  0.22090771  8.48531942  6.31916682  3.48826018  3.42290525
 -6.77298371  6.77682585  8.78441028  5.07020258], intercept [108.22061421]
Score: -0.30
The mean squared error (MSE) on test set: 7579.2342

Test LinearSVR With Loss Parameter:
loss：epsilon_insensitive
Coefficients:[ 3.0969555   0.21980477  8.55893237  6.36317595  3.43404979  3.43410945
 -6.78650531  6.75681273  8.69419193  5.08829057], intercept [108.24532057]
Score: -0.30
The mean squared error (MSE) on test set: 7577.0949

loss：squared_epsilon_insensitive
Coefficients:[  22.95355014 -141.78262896  375.99848962  242.18578298  -19.86319347
  -59.90446517 -182.882332    117.63381422  334.22068586  106.54368606], intercept [152.83864072]
Score: 0.46
The mean squared error (MSE) on test set: 3154.5451

Test LinearSVR With Epsilon Parameter:
```

![img](D:\00研二上\模式识别-2\IRM\SVM_regression\__results___7_1.png)

![img](D:\00研二上\模式识别-2\IRM\SVM_regression\__results___7_2.png)

```
Test LinearSVR With Epsilon Parameter:
```

![img](D:\00研二上\模式识别-2\IRM\SVM_regression\__results___7_4.png)

![img](D:\00研二上\模式识别-2\IRM\SVM_regression\__results___7_5.png)

### 类：测试参数对于svr非线性模型的影响

```python
class ParamsTestSVR:
    '''SVR 的参数对它的预测性能的影响
    '''
    def __init__(self, data, poly_degrees, poly_gammas, poly_rs, rbf_gammas, sigmoid_gammas, sigmoid_rs):
        '''
        初始化 ParamsTestSVR 参数
        
        :param data: 已分割的数据集。可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = data
        self.poly_degrees = poly_degrees
        self.poly_gammas = poly_gammas
        self.poly_rs = poly_rs
        self.rbf_gammas = rbf_gammas
        self.sigmoid_gammas = sigmoid_gammas
        self.sigmoid_rs = sigmoid_rs
        # save scores
        self.train_scores_poly_degree = []
        self.test_scores_poly_degree = []
        self.train_scores_poly_gamma = []
        self.test_scores_poly_gamma = []
        self.train_scores_poly_r = []
        self.test_scores_poly_r = []
        self.train_scores_rbf_gamma = []
        self.test_scores_rbf_gamma = [] 
        self.train_scores_sigmoid_gamma = []
        self.test_scores_sigmoid_gamma = []
        self.train_scores_sigmoid_r = []
        self.test_scores_sigmoid_r = []
        # save mses
        self.train_mses_poly_degree = []
        self.test_mses_poly_degree = []
        self.train_mses_poly_gamma = []
        self.test_mses_poly_gamma = []
        self.train_mses_poly_r = []
        self.test_mses_poly_r = []
        self.train_mses_rbf_gamma = []
        self.test_mses_rbf_gamma = [] 
        self.train_mses_sigmoid_gamma = []
        self.test_mses_sigmoid_gamma = []
        self.train_mses_sigmoid_r = []
        self.test_mses_sigmoid_r = []
        
    # 被调用的函数
    def fit_SVR_linear_score(self,regr):
        '''
        拟合 SVR linear 曲线，输出预测的分数与相关系数与残差系数
        
        :param regr: 根据给定的训练数据拟合回归模型
        :return: None
        '''
        regr.fit(self.X_train,self.y_train)
        print('Coefficients:%s, intercept %s'%(regr.coef_,regr.intercept_)) 
        # coef_：分配给特征的权重（原始问题的系数），仅在线性内核的情况下可用。
        # intercept_：决策函数的截距
        print('Score: %.2f' % regr.score(self.X_test, self.y_test))# score函数返回方差
        mse = mean_squared_error(self.y_test, regr.predict(self.X_test))
        print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
        print()
        
    # 被调用的函数
    def fit_SVR_scoreList_mseList(self,regr, train_scores, test_scores, train_mses, test_mses):
        '''
        拟合 SVR 曲线，输出预测的分数的列表数据
        
        :param regr: 拟合曲线使用的回归模型。
        :param train_scores: 用于记录在训练集上训练的模型的预测分数的列表数据变量
        :param test_scores: 用于记录在测试集上训练的模型的预测分数的列表数据变量
        :param train_mses: 用于记录在训练集上训练的模型的mse分数的列表数据变量
        :param test_mses: 用于记录在测试集上训练的模型的mse分数的列表数据变量
        :return: None
        '''
        regr.fit(self.X_train,self.y_train)
        train_scores.append(regr.score(self.X_train, self.y_train))
        test_scores.append(regr.score(self.X_test, self.y_test))
        train_mses.append(mean_squared_error(self.y_test, regr.predict(self.X_test)))
        test_mses.append(mean_squared_error(self.y_test, regr.predict(self.X_test)))
        
    # 被调用的函数    
    def plot_scoreList(self,uniqueNum, fignrows,figncols, fignum,test_paramsList, 
                       train_scores, test_scores, title, xlabel, ylabel):
        '''
        画出回归模型同一个参数不同参数值，运行后预测分数列表的曲线
        
        :param uniqueNum: 画图的唯一标识
        :param fignrows: 画图的子图总行数
        :param figncols: 画图的子图总列数
        :param fignum: 画图的子图序数
        :param test_paramsList: 回归模型需要测试的参数的参数值的列表。
        :param train_scores: 用于记录在训练集上训练的模型的预测分数的列表数据变量
        :param test_scores: 用于记录在测试集上训练的模型的预测分数的列表数据变量
        :param params_name: 回归模型需要测试的参数的名字。
        :param xlabel: 回归曲线预测分数的图的横坐标的标签，即参数名。
        :param ylabel: 回归曲线预测分数的图的纵坐标的标签，即预测分数名。
        :return: None
        '''
        
        fig=plt.figure(num=uniqueNum,figsize=(30,30))
        ax=fig.add_subplot(fignrows,figncols,fignum)
        #ax = axes[col_num] # 矩阵从0开始
        ax.plot(test_paramsList, train_scores, label =" Training score ", marker='+' )
        ax.plot(test_paramsList, test_scores, label = " Testing  score ", marker='o' )
        ax.set_title(title)# "epsilon","C" 
        ax.set_xscale("log")
        ax.set_xlabel(xlabel) #r"$\epsilon$", r"C"
        ax.set_ylabel(ylabel)
        if ylabel == "score":
            ax.set_ylim(-1,1.05)
        elif ylabel == "mean squared error (MSE)":
            ax.set_ylim(1500,5200)
            #ax.set_ylim(0,1)
        ax.legend(loc="best",framealpha=0.5)
        # plt.savefig(filename) #取消”#“，函数plot_scoreList需要添加参数”filename“
        # 设置 X 轴的网格线，风格为 点画线
        plt.grid(axis='x',linestyle='-.')
        plt.show()
    
    def test_SVR_linear(self,):
        '''
        测试 SVR 的用法。这里使用最简单的线性核

        :return: None
        '''
        regr=svm.SVR(kernel='linear')
        self.fit_SVR_linear_score(regr)
        
    def test_SVR_poly(self,coef0_nr=1,degree_nd=3,gamma_ngd=20):
        '''
        测试 多项式核的 SVR 的预测性能随  degree、gamma、coef0 的影响.
        
        :param coef0_nr: 除了测试r时，不能用，测试gamma和degree都能用，默认值：default=1
        :param degree_nd: 除了测试degree时，不能用，测试gamma和r都能用，默认值：default=3
        :param gamma_ngd: 除了测试gamma和degree时，不能用,测试r时能用，默认值：default=20
        :return: None
        '''
        ### 测试 degree ####
        print("Test poly degree:")
        for degree in self.poly_degrees:
            regr=svm.SVR(kernel='poly',degree=degree,coef0=coef0_nr)
            self.fit_SVR_scoreList_mseList(regr, self.train_scores_poly_degree, self.test_scores_poly_degree,
                                          self.train_mses_poly_degree, self.test_mses_poly_degree)
        self.plot_scoreList(5,3,3,1, self.poly_degrees, 
                            self.train_scores_poly_degree, self.test_scores_poly_degree, 
                            "SVR_poly_degree r=1", "p","score")
        self.plot_scoreList(6,3,3,2, self.poly_degrees, 
                            self.train_mses_poly_degree, self.test_mses_poly_degree, 
                            "SVR_poly_degree r=1", "p", "mean squared error (MSE)")

        ### 测试 gamma，固定 degree为3， coef0 为 1 ####
        print("Test poly gamma:")
        for gamma in self.poly_gammas:
            regr=svm.SVR(kernel='poly',gamma=gamma,degree=degree_nd,coef0=coef0_nr)
            self.fit_SVR_scoreList_mseList(regr, self.train_scores_poly_gamma, self.test_scores_poly_gamma,
                                          self.train_mses_poly_gamma, self.test_mses_poly_gamma)
        self.plot_scoreList(7,3,3,3, self.poly_gammas, 
                            self.train_scores_poly_gamma, self.test_scores_poly_gamma, 
                            "SVR_poly_gamma  r=1", r"$\gamma$","score")
        self.plot_scoreList(8,3,3,4, self.poly_gammas, 
                            self.train_mses_poly_gamma, self.test_mses_poly_gamma, 
                            "SVR_poly_gamma  r=1", r"$\gamma$","mean squared error (MSE)")

        ### 测试 r，固定 gamma 为 20，degree为 3 ######
        print("Test poly coef0:")
        for r in self.poly_rs:
            regr=svm.SVR(kernel='poly',gamma=gamma_ngd,degree=degree_nd,coef0=r)
            self.fit_SVR_scoreList_mseList(regr, self.train_scores_poly_r, self.test_scores_poly_r,
                                          self.train_mses_poly_r, self.test_mses_poly_r)
        self.plot_scoreList(9,3,3,5, self.poly_rs, 
                            self.train_scores_poly_r, self.test_scores_poly_r, 
                            "SVR_poly_r gamma=50 degree=2", r"r","score")
        self.plot_scoreList(10,3,3,6, self.poly_rs, 
                            self.train_mses_poly_r, self.test_mses_poly_r, 
                            "SVR_poly_r gamma=50 degree=2", r"r","mean squared error (MSE)")
        
    def test_SVR_rbf(self,):
        '''
        测试 高斯核的 SVR 的预测性能随 gamma 参数的影响

        :return: None
        '''
        print("Test rbf gamma:")
        for gamma in self.rbf_gammas:
            regr=svm.SVR(kernel='rbf',gamma=gamma) 
            self.fit_SVR_scoreList_mseList(regr, self.train_scores_rbf_gamma, self.test_scores_rbf_gamma,
                                          self.train_mses_rbf_gamma, self.test_mses_rbf_gamma)
        self.plot_scoreList(11,3,3,1, self.rbf_gammas, 
                            self.train_scores_rbf_gamma, self.test_scores_rbf_gamma, 
                            "SVR_rbf_gamma", r"$\gamma$","score")
        self.plot_scoreList(12,3,3,2, self.rbf_gammas, 
                            self.train_mses_rbf_gamma, self.test_mses_rbf_gamma, 
                            "SVR_rbf_gamma", r"$\gamma$","mean squared error (MSE)")
        
    def test_SVR_sigmoid(self,coef0_gamma=0.01,gamma_r=10):
        '''
        测试 sigmoid 核的 SVR 的预测性能随 gamma、coef0 的影响.

        :param coef0_gamma: 测试gamma时能用，默认值：default=0.01
        :param gamma_r: 测试r时能用，默认值：default=10 
        :return: None
        '''
        ### 测试 gamma，固定 coef0 为 0.01 ####
        for gamma in self.sigmoid_gammas:
            regr=svm.SVR(kernel='sigmoid',gamma=gamma,coef0=coef0_gamma)
            self.fit_SVR_scoreList_mseList(regr, self.train_scores_sigmoid_gamma, self.test_scores_sigmoid_gamma,
                                          self.train_mses_sigmoid_gamma, self.test_mses_sigmoid_gamma)
        self.plot_scoreList(13,3,3,1, self.sigmoid_gammas, 
                            self.train_scores_sigmoid_gamma, self.test_scores_sigmoid_gamma, 
                            "SVR_sigmoid_gamma r=0.01", r"$\gamma$","score")
        self.plot_scoreList(14,3,3,2, self.sigmoid_gammas, 
                            self.train_mses_sigmoid_gamma, self.test_mses_sigmoid_gamma, 
                            "SVR_sigmoid_gamma r=0.01", r"$\gamma$","mean squared error (MSE)")
        
        ### 测试 r ，固定 gamma 为 10 ######
        for r in self.sigmoid_rs:
            regr=svm.SVR(kernel='sigmoid',coef0=r,gamma=gamma_r)
            self.fit_SVR_scoreList_mseList(regr, self.train_scores_sigmoid_r, self.test_scores_sigmoid_r,
                                          self.train_mses_sigmoid_r, self.test_mses_sigmoid_r)
        self.plot_scoreList(15,3,3,3, self.sigmoid_rs, 
                            self.train_scores_sigmoid_r, self.test_scores_sigmoid_r, 
                            "SVR_sigmoid_r gamma=90", r"r","score")
        self.plot_scoreList(16,3,3,4, self.sigmoid_rs, 
                            self.train_mses_sigmoid_r, self.test_mses_sigmoid_r, 
                            "SVR_sigmoid_r gamma=90", r"r","mean squared error (MSE)")
```

### 主程序：测试参数对于svr非线性模型的影响

```python
if __name__=="__main__":
    data = load_data_regression() # 生成用于回归问题的数据集
    # SVR 模型的参数的参数值列表
    # 等比数列
    poly_degrees=range(1,20)
    poly_gammas=range(1,100)
    poly_rs=range(-4,20)
    rbf_gammas=range(1,100)
    sigmoid_gammas=np.logspace(-1,3)
    sigmoid_rs=np.logspace(-2,2)
    # 实例化
    test_2 = ParamsTestSVR(data, poly_degrees, poly_gammas, poly_rs, rbf_gammas, sigmoid_gammas, sigmoid_rs)
    test_2.test_SVR_linear() # 调用 test_SVR_linear
    test_2.test_SVR_poly(coef0_nr=1,degree_nd=2,gamma_ngd=50) # 调用 test_SVR_poly
    test_2.test_SVR_rbf() # 调用 test_SVR_rbf
    test_2.test_SVR_sigmoid(coef0_gamma=0.01,gamma_r=90)# 调用 test_SVR_sigmoid
```

```
Coefficients:[[ 2.80068715 -0.19064351  9.16679169  7.04181819  3.01608867  2.52961327
  -6.92516071  6.85807331  9.56967364  5.73267597]], intercept [140.82689225]
Score: 0.01
The mean squared error (MSE) on test set: 5770.5340
```

```
Test poly degree:
```

![img](D:\00研二上\模式识别-2\IRM\SVM_regression\__results___9_1.png)

![img](D:\00研二上\模式识别-2\IRM\SVM_regression\__results___9_2.png)

```
Test poly gamma:
```

![img](D:\00研二上\模式识别-2\IRM\SVM_regression\__results___9_4.png)

![img](D:\00研二上\模式识别-2\IRM\SVM_regression\__results___9_5.png)

```
Test poly coef0:
```

![img](D:\00研二上\模式识别-2\IRM\SVM_regression\__results___9_7.png)

![img](D:\00研二上\模式识别-2\IRM\SVM_regression\__results___9_8.png)

```
Test rbf gamma:
```

![img](D:\00研二上\模式识别-2\IRM\SVM_regression\__results___9_10.png)

![img](D:\00研二上\模式识别-2\IRM\SVM_regression\__results___9_11.png)

![img](D:\00研二上\模式识别-2\IRM\SVM_regression\__results___9_12.png)

![img](D:\00研二上\模式识别-2\IRM\SVM_regression\__results___9_13.png)

![img](D:\00研二上\模式识别-2\IRM\SVM_regression\__results___9_14.png)

![img](D:\00研二上\模式识别-2\IRM\SVM_regression\__results___9_15.png)





## 参考文献：

1. 可视化
   1. [Seaborn | Regression Plots](https://www.geeksforgeeks.org/seaborn-regression-plots/)
   2. [Seaborn详细操作（一）之风格设置](https://www.cnblogs.com/slyu/p/15234272.html)
   
2. 目标的是否非正态分布
   1. [python数据可视化seaborn（二）—— 分布数据可视化](https://zhuanlan.zhihu.com/p/53461307)
   2. [**Python 数据可视化：seaborn displot 正态分布曲线拟合图代码注释超详解(放入自写库，一行代码搞定复杂细节绘图)**](https://blog.51cto.com/u_15441143/4674352)
   
3. 相关性(特征与特征的关系)
   1. [5分钟入门 Seaborn 热力图可视化](https://mp.weixin.qq.com/s?__biz=MzAxMjUyNDQ5OA==&mid=2653561779&idx=2&sn=0f0f5a2692e39f45e7700de4198386e6&chksm=806e0d0eb71984189e9a567b266a0818b8b252a331776bcced7aa3a605e4364fbbb5941702df&scene=27)
   
4. 相关性(特征与目标的关系)
   1. [去相关性（decorrelation）在图像处理中的意义？为什么说DCT变换具有去相关性。](https://www.zhihu.com/question/40982621)
   2. PCA去相关性：[R语言 主成分分析PCA（绘图+原理）](https://zhuanlan.zhihu.com/p/511500205)
   
5. 回归整体方法
   1. [svr预测出来是一条直线_机器学习 | 一个基于机器学习的简单小实践：波斯顿房价预测分析...](https://blog.csdn.net/weixin_39873208/article/details/111347231)
   2. [[统计]_线性回归中因变量一定要正态分布吗？](https://blog.csdn.net/weixin_39366714/article/details/123870680) 不太需要，如果你觉得系数过大或过小可以对变量进行对数化处理
   3. [my target feature is non normal dist can i use any regression models?](https://www.kaggle.com/questions-and-answers/205790)
   4. [**ML之LassoR&RidgeR：基于datasets糖尿病数据集利用LassoR和RidgeR算法(alpha调参)进行(9→1)回归预测**](https://blog.51cto.com/yunyaniu/5089832)
   4. [python特征相关性热力图怎么画_Python实战链家二手房房价预测模型](https://blog.csdn.net/weixin_39843738/article/details/110521362)
   4. [实验1 线性回归 实操项目1——糖尿病情预测](https://blog.csdn.net/m0_52896752/article/details/127681957)
   4. [基于SVM的糖尿病数据集回归问题](https://blog.csdn.net/m0_37758063/article/details/124086219) \# 模型评估，使用R-squared、MSE、MAE指标评估
   4. [基于最邻近算法的糖尿病数据集回归](https://www.codetd.com/article/13752782#35__235)
   
6. svr
   
   1. [API Reference](https://scikit-learn.org/stable/modules/classes.html)
   2. [`sklearn.svm`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm).[SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR)
   3. [`sklearn.svm`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm).[LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR)
   
7. svr的参数寻找

   1. [机器学习之非线性回归SVR](https://blog.csdn.net/mr_muli/article/details/84700414)
   2. [机器学习之线性回归SVR](https://blog.csdn.net/mr_muli/article/details/84699260)
   2. [ Vertica Model.regression_report / report](https://www.vertica.com/python/documentation_last/learn/Model/regression_report/)

8. 画子图
   1. [画子图(add_subplot & subplot)](https://blog.csdn.net/You_are_my_dream/article/details/53439518)
   2. [【DS with Python】Matplotlib入门(一)：架构概述、面向对象编程绘图与函数式绘图基础](https://blog.csdn.net/Mart_inn/article/details/122352935)
   3. [【DS with Python】Matplotlib入门(二)：子图集、绘图布局与常用统计图形](https://blog.csdn.net/Mart_inn/article/details/122461228)

9. svr-sklearn 中文手册
   1. [sklearn.svm.LinearSVR](https://scikit-learn.org.cn/view/777.html)
   2. [sklearn.svm.SVR](https://scikit-learn.org.cn/view/782.html)

10. 类对象的创建
   1. [如何理解Python里的class？](https://www.zhihu.com/question/26692035/answer/2361915301)
   2. 类的继承，不明白

11. 随机数

    1. [python生成随机数组-多种方法](https://www.codenong.com/cs106752111/)

       ```python
       # 导入 random 包
       import random
       
       # 返回一个 1 到 9 之间的数字
       np.random.seed(0)
       X = np.random.randint(10, size=(100,10))
       y = np.random.randint(2, size=(10,))
       ```

       

    2. [Python random randint() 方法](https://www.runoob.com/python3/ref-random-randint.html)

    3. [Python机器学习、深度学习中将已知数据集顺序打乱的方法](https://baijiahao.baidu.com/s?id=1729902631277823237&wfr=spider&for=pc)

12. mse

    1. [请问MSE loss 大小多少才表示模型优化效果好呢？0.01大概是什么水平？](https://www.zhihu.com/question/426007107)

       光看MSE loss 0.01没办法提供任何信息，在于你不知道这个任务本身的难易程度。如果你的模型达到了99% 的precision，但是你的baseline模型用一个简单的linear regression也能达到99%，那证明并不是模型本身效果好，而是任务简单；相反，如果baseline的precision是60%，你的达到了70%，那能证明你的模型在这个任务里优化效果很棒。

    2. 

13. 模型集成

    1. [svr预测出来是一条直线_机器学习 | 一个基于机器学习的简单小实践：波斯顿房价预测分析...](https://blog.csdn.net/weixin_39873208/article/details/111347231)

       **模型融合与评估**

       模型融合和寻找高级特征是提升机器学习性能的两个重要手段。模型融合的方法很多，比如bagging，stacking，boosting，average weight，voting等。本文选择average weight和stacking这两种方法。用于融合的模型有LinearRegression，Ridge，Lasso，Random Forrest，Gradient Boosting Tree，Support Vector Regression，Linear Support Vector Regression，ElasticNet，Stochastic Gradient Descent，BayesianRidge，KernelRidge，ExtraTreesRegressor共12个基础模型。

       **评估函数**

       因为该案例是典型的回归问题，对于回归问题最适合采用基于距离的的评估函数，本文采用均方误差,调用库scikit-learn中cross_val_score函数评估模型效果。cross_val_score函数采用K折交叉验证，将训练样本分割成K份，一份被保留作为验证模型的数据（test set），其他K-1份用来训练（train set）。交叉验证重复K次，每份验证一次，平均K次的结果或者使用其它结合方式，最终得到一个单一估测，这个方法的优势在于，同时重复运用随机产生的子样本进行训练和验证，运用同样的样本可以训练模型制定的次数，在样本量不足的环境下有用，交叉验证用于评估模型的预测性能，尤其是训练好的模型在新数据上的表现，可以在一定程度上减小过拟合，还可以从有限的数据中获取尽可能多的有效信息。应用cross_val_score计算出各模型的得分情况如下

       ![image-20230112141616810](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230112141616810.png)

       

       ```python
       pca_model = PCA(n_components=63)
       Features= pca_model.fit_transform(features)
       
       # 设置交叉验证
       kf = KFold(n_splits=5, random_state=42, shuffle=True)
        
       # 定义错误度量
       def rmsle(y, y_pred):
           return np.sqrt(mean_squared_error(y, y_pred))
        
       def cv_rmse(model, X=Features):
           rmse = np.sqrt(-cross_val_score(model, X, erhouse_labels, scoring="neg_mean_squared_error", cv=kf))
           return (rmse)
       ```

       **超参数调优**

       超参数是在开始学习过程之前设置值的参数，而不是通过训练得到的参数数据。通常情况下，需要对超参数进行优化，给学习机选择一组最优超参数，以提高学习的性能和效果。对于所选择的12个备用模型，很多都有需要自己设置的超参数，一十不知道如何设置。我们采用网格搜索最优参数。搜索前，先给每个参数准备一个参数网，然后调用scikit-learn库中的GridSearchCV搜索最有或者次优参数。以Kernel Ridge（核岭回归）为例，KernelRidge()有四个超参数，alpha，kernel，degree，coef0。根据经验，设置参数网param_grid={'alpha':[0.2,0.3,0.4],'kernel':["polynomial"],'degree':[3],'coef0':[0.8,1.0]}。结果如下

       ![image-20230112141725224](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230112141725224.png)

       由此此网格中的最优参数alpha:0.2,coef:1,degree:3,kernel:polynomial。注意采用网格搜索无法求出全局的最优参数，只能求出指定网格中的最优参数表，因而是次优的。可以依次求出各个模型的最佳超参数如下。

       ![image-20230112141744114](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230112141744114.png)

       **模型集成**

       接下来进行模型融合，先使用加权平均的方法，根据备选模型选择得分最佳的6个模型来进行融合，并且根据得分情况分配他们的权重。模型分别是

       | 模型                                                         | 权重 |
       | ------------------------------------------------------------ | ---- |
       | lasso=Lasso(alpha=0.0005,max_iter=10000)                     | 0.02 |
       | ridge = Ridge(alpha=60)                                      | 0.2  |
       | svr=SVR(gamma=0.0004,kernel='rbf',C=13,epsilon=0.009)        | 0.25 |
       | ker=KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 ,coe8) | 0.3  |
       | ela = ElasticNet(alpha=0.005,l1_ratio=0.08,max_iter=10000)   | 0.03 |
       | bay = BayesianRidge()                                        | 0.2  |

       
       模型融合后的最终得分为0.1077，好于单个模型的得分情况。
       

14. 交叉验证

    1. [使用sklearn的cross_val_score进行交叉验证](https://blog.csdn.net/qq_36523839/article/details/80707678)

       ![image-20230112142313872](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230112142313872.png)

15. 