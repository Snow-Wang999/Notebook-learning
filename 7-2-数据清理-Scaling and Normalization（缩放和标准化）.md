# 7-2-数据清理-Scaling and Normalization（缩放和标准化）

转换数字变量以具有有用的属性。

在本笔记本中，我们将研究如何缩放和标准化数据（以及两者之间的区别！）。 让我们开始吧！

## 设置我们的环境 

我们需要做的第一件事是加载我们要使用的库。

```python
# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# set seed for reproducibility
np.random.seed(0)
```

## 缩放与标准化：有什么区别？(Scaling vs. Normalization: What's the difference?) 

缩放和标准化之间容易混淆的原因之一是，这些术语有时可以互换使用，更令人困惑的是，它们非常相似！在这两种情况下，您都在转换数字变量的值，以便转换后的数据点具有特定的有用属性。

区别在于： 

- 在缩放中，您正在改变数据的范围，而 
- 在标准化中，您正在改变数据分布的形状。 让我们更深入地讨论一下这些选项中的每一个。

### 缩放比例 (Scaling)

这意味着您正在转换数据，使其符合特定的比例，如0-100或0-1。当您使用基于数据点距离度量的方法（如支持向量机（[support vector machines (SVM)](https://en.wikipedia.org/wiki/Support_vector_machine)）或k近邻（[k-nearest neighbors (KNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)））时，您需要缩放数据。使用这些算法，任何数字特征中“1”的变化都具有相同的重要性。 

例如，您可能正在查看某些产品的日元和美元价格。一美元大约值100日元，但如果你不调整价格，SVM或KNN等方法会认为1日元的差价与1美元的差价一样重要！这显然不符合我们对世界的直觉。使用货币，您可以在货币之间进行转换。但如果你在看身高和体重之类的东西呢？多少磅等于一英寸（或者多少公斤等于一米）并不完全清楚。 

通过缩放变量，您可以在平等的基础上比较不同的变量。为了帮助确定缩放的样子，让我们看一个虚构的示例。（别担心，我们将在以下练习( [**the following exercise**](https://www.kaggle.com/kernels/fork/10824404)!)中使用真实数据！）

```python
# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size=1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0])

# plot both together to compare
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(scaled_data, ax=ax[1], kde=True, legend=False)
ax[1].set_title("Scaled data")
plt.show()
```

![image-20230109195923687](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230109195923687.png)

请注意，数据的形状没有改变，但它的范围不是从0到8，而是从0到1。

### 规范化(Normalization)

缩放只是改变数据的范围。标准化是一个更为激进的转变。标准化的目的是改变你的观察结果，使它们可以被描述为正态分布。 

> 正态分布：也称为“钟形曲线(bell curve)”，这是一种特定的统计分布，其中大致相等的观测值落在平均值之上和之下，平均值和中值相同，并且有更多的观测值接近平均值。正态分布也称为高斯分布。 

一般来说，如果您要使用机器学习或统计技术，假设您的数据是正常分布的，那么您将标准化您的数据。其中的一些例子包括线性判别分析（LDA）和高斯朴素贝叶斯。（专业提示：任何名称中带有**“Gaussian”**的方法都可能**假定为正态**。） 我们在这里使用的标准化方法叫做Box-Cox变换。让我们快速了解一下标准化某些数据的情况：

```python
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(normalized_data[0], ax=ax[1], kde=True, legend=False)
ax[1].set_title("Normalized data")
plt.show()
```

![image-20230109200256229](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230109200256229.png)

请注意，数据的形状发生了变化。在正常化之前，它几乎是L形。但是在标准化之后，它看起来更像钟形的轮廓（因此是“钟形曲线”）。





-----

## 补充：dataframe 归一化

数据的标准化或归一化是[特征工程](https://www.zhihu.com/search?q=特征工程&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2275283055})的第一步。列的归一化将涉及到把列的值带到一个共同的尺度，主要是针对范围不同的列进行的。在  中，可以通过多种函数对 Dataframes 的列进行归一化。

1.用 mean 归一化来归纳 Pandas 

“均值 “归一化是对不同范围的 DataFrame 进行归一化的最简单方法之一。归一化是通过减去 DataFrame 所有元素的平均值并除以[标准差](https://www.zhihu.com/search?q=标准差&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2275283055})来完成的。

```python
import pandas as pd
import numpy as np

np.random.seed(0)

df = pd.DataFrame(np.random.randint(-100,100,size=(20, 4)), columns=list('ABCD'))

def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

df_mean_norm = mean_norm(df)
print(df_mean_norm)

```

```
A         B         C         D
0   1.452954 -1.090261  0.278088  1.247208
1  -0.514295  1.585670  0.037765 -1.333223
2  -1.376137 -1.289148 -0.236890 -0.473079
3  -0.120845  0.591236 -0.734701  1.261309
4  -1.038895 -0.367037  1.256545 -0.219266
5  -0.251995  1.043252 -1.301176 -0.374374
6  -0.420617 -1.777325  0.810231  0.161453
7   1.921346 -0.511681  1.273711  1.247208
8  -0.233260 -0.150069  1.308043 -1.051208
9   0.984561  0.717801  0.707236  0.894690
10 -1.170045  1.549509 -1.575831  1.148503
11  0.609847 -1.361470 -1.198181  0.669079
12  1.284333  0.121140  1.411038 -1.065309
13 -1.132573  0.374269  0.466913  0.852388
14 -0.776595  0.464672 -1.078020 -1.220417
15 -0.289467  0.446591  0.072097 -0.867899
16  1.715254 -1.379551  0.329586 -1.446028
17 -0.551766  1.115574 -0.751867 -0.966604
18  0.141455  0.211543 -1.541499  0.993395
19 -0.233260 -0.294714  0.466913  0.542172
```

2.用 最小-最大 [归一化方法](https://www.zhihu.com/search?q=归一化方法&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2275283055})对 Pandas DataFrame 进行归一化

这是广泛使用的归一化方法之一。归一化输出减去 DataFrame 的最小值，然后除以相应列的最高值和最低值之差。

```python
import pandas as pd
import numpy as np

np.random.seed(0)

df = pd.DataFrame(np.random.randint(-100,100,size=(20, 4)), columns=list('ABCD'))

def minmax_norm(df_input):
    return (df - df.min()) / ( df.max() - df.min())

df_minmax_norm = minmax_norm(df)

print(df_minmax_norm)
```

```python
A         B         C         D
0   0.857955  0.204301  0.620690  0.994792
1   0.261364  1.000000  0.540230  0.041667
2   0.000000  0.145161  0.448276  0.359375
3   0.380682  0.704301  0.281609  1.000000
4   0.102273  0.419355  0.948276  0.453125
5   0.340909  0.838710  0.091954  0.395833
6   0.289773  0.000000  0.798851  0.593750
7   1.000000  0.376344  0.954023  0.994792
8   0.346591  0.483871  0.965517  0.145833
9   0.715909  0.741935  0.764368  0.864583
10  0.062500  0.989247  0.000000  0.958333
11  0.602273  0.123656  0.126437  0.781250
12  0.806818  0.564516  1.000000  0.140625
13  0.073864  0.639785  0.683908  0.848958
14  0.181818  0.666667  0.166667  0.083333
15  0.329545  0.661290  0.551724  0.213542
16  0.937500  0.118280  0.637931  0.000000
17  0.250000  0.860215  0.275862  0.177083
18  0.460227  0.591398  0.011494  0.901042
19  0.346591  0.440860  0.683908  0.734375
```

在上面的输出中，我们可以推断出每一列的最小值被转化为 `0`，每一列的最大值被转化为 `1`。



3.使用 [分位数](https://www.zhihu.com/search?q=分位数&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2275283055}) 归一化对 Pandas DataFrame 进行归一化

量子化归一化用于[高维数据分析](https://www.zhihu.com/search?q=高维数据分析&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2275283055})。它观察并假设每一列的[统计分布](https://www.zhihu.com/search?q=统计分布&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2275283055})是相同的。分位数归一化包括以下步骤。

(1)对每列内的数值进行排序（Ranking）； 

(2)每行的平均值，用平均值代替行中每个元素的值。

(3)将数值重新排序到最初的顺序。

```python
import numpy as np
import pandas as pd

np.random.seed(0)

df = pd.DataFrame(np.random.randint(-100,100,size=(20, 4)), columns=list('ABCD'))

def quantile_norm(df_input):
    sorted_df = pd.DataFrame(np.sort(df_input.values,axis=0), index=df_input.index, columns=df_input.columns)
    mean_df = sorted_df.mean(axis=1)
    mean_df.index = np.arange(1, len(mean_df) + 1)
    quantile_df =df_input.rank(method="min").stack().astype(int).map(mean_df).unstack()
    return(quantile_df)

df_quantile_norm = quantile_norm(df)

print(df_quantile_norm)
```

```
A      B      C      D
0   77.00 -58.25   8.25  77.00
1  -36.50  92.00 -10.50 -79.25
2  -90.00 -66.50 -20.00 -20.00
3   24.75  44.00 -36.50  92.00
4  -66.50 -36.50  71.75  -3.00
5   -3.00  71.75 -73.00 -10.50
6  -20.00 -90.00  54.00   8.25
7   92.00 -41.00  77.00  77.00
8    8.25 -10.50  87.00 -58.25
9   54.00  54.00  44.00  44.00
10 -79.25  87.00 -90.00  71.75
11  44.00 -73.00 -66.50  24.75
12  71.75  -3.00  92.00 -66.50
13 -73.00  18.00  24.75  31.75
14 -58.25  31.75 -58.25 -73.00
15 -10.50  24.75  -3.00 -36.50
16  87.00 -79.25  18.00 -90.00
17 -41.00  77.00 -41.00 -41.00
18  31.75   8.25 -79.25  54.00
19   8.25 -20.00  24.75  18.00
```

[在 Pandas DataFrame 中如何按索引删除列? - 知乎 (zhihu.com)](https://www.zhihu.com/question/506256434/answer/2272205798)

[在 Pandas DataFrame 中两列可以直接相减吗？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/506031329/answer/2270739713)

[可以直接比较两个 Pandas DataFrame 对象吗？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/506030489/answer/2270730211)

[怎样拆分 Pandas DataFrame？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/506028641/answer/2270721579)

[Pandas 中如何获取特定列满足给定条件的所有行的索引？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/505784526/answer/2269146260)



作者：特立独行的小象
链接：https://www.zhihu.com/question/506722733/answer/2275283055
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

---

## 参考文献：

1. [在 Pandas DataFrame 中如何归一化某列？](https://www.zhihu.com/question/506722733)

2. [`sklearn.preprocessing`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing).[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

   通过去除平均值并缩放到单位方差来标准化特征。 

   样本x的标准分数计算如下： 

   $z=(x-u)/s $

   - 其中u是训练样本的平均值，如果_mean=False，则为零；
   - s是训练样本标准偏差，如果_std=False，则是一。

3. 