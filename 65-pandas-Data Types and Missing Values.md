# 65-pandas-Data Types and Missing Values

处理最常见的进度受阻问题

## 介绍 

在本教程中，您将学习如何调查`DataFrame`或`Series`中的数据类型。您还将学习如何查找和替换条目。 

要开始本主题的练习，请单击此处。

## 数据类型(Dtypes)

`DataFrame`或`Series`中列的数据类型称为`dtype`。 可以使用`dtype`属性获取特定列的类型。例如，我们可以在`reviews` `DataFrame`中获取`price`列的`dtype`：

```python
import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option('max_rows', 5)
```

---

补充：

[pandas 8 个常用的 set_option 设置方法](https://zhuanlan.zhihu.com/p/382820842)

```python
# 使用方法
import pandas as pd
pd.set_option()
pd.get_option()

'''
以下8个常用的set_option:
- 显示更多行
- 显示更多列
- 改变列宽
- 设置float列的精度
- 数字格式化显示
- 更改绘图方法
- 配置info()的输出
- 打印出当前设置并重置所有选项
'''
pd.set_option('display.max_rows',xxx) # 最大行数
pd.set_option('display.min_rows',xxx) # 最小显示行数
pd.set_option('display.max_columns',xxx) # 最大显示列数
pd.set_option ('display.max_colwidth',xxx) #最大列字符数
pd.set_option( 'display.precision',2) # 浮点型精度
pd.set_option('display.float_format','{:,}'.format) #逗号分隔数字
pd.set_option('display.float_format',  '{:,.2f}'.format) #设置浮点精度
pd.set_option('display.float_format', '{:.2f}%'.format) #百分号格式化
pd.set_option('plotting.backend', 'altair') # 更改后端绘图方式
pd.set_option('display.max_info_columns', 200) # info输出最大列数
pd.set_option('display.max_info_rows', 5) # info计数null时的阈值
pd.describe_option() #展示所有设置和描述
pd.reset_option('all') #重置所有设置选项
```

系列内容，请看「[pandas100个骚操作](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/mp/appmsgalbum%3F__biz%3DMzUzODYwMDAzNA%3D%3D%26action%3Dgetalbum%26album_id%3D1699019347278561282%26scene%3D173%26from_msgid%3D2247515707%26from_itemidx%3D4%26count%3D3%26nolastread%3D1%23wechat_redirect)」

---

```python
reviews.price.dtype
```

```
dtype('float64')
```

或者，dtypes属性返回DataFrame中每一列的dtype：

```python
reviews.dtypes
```

![image-20221126120126507](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126120126507.png)

数据类型告诉我们熊猫如何在内部存储数据。`float64`表示它使用64位浮点数字；`int64`表示大小类似的整数，依此类推。 

要记住的一个特点（这里显示得非常清楚）是，**完全由字符串组成的列没有自己的类型**；相反，它们被赋予对象`object`类型。

使用`astype()`函数可以将一种类型的列转换为另一种类型，只要这种转换是有意义的。例如，我们可以将`points`列从其现有的`int64`数据类型转换为`float64`数据类型：

```python
reviews.points.astype('float64')
```

![image-20221126120738519](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126120738519.png)

`DataFrame`或`Series`索引也有自己的数据类型：

```python
reviews.index.dtype
```

```
dtype('int64')
```

Pandas还支持更奇特的数据类型，如分类数据(categorical data)和时间序列数据(timeseries data)。因为这些数据类型很少使用，所以我们将省略它们，直到本教程的稍后部分。

## Missing data

缺少值的条目被赋予`NaN`值，即“非数字”的缩写。由于技术原因，这些`NaN`值始终为float64数据类型。 

Pandas提供了一些特定于缺失数据的方法。**要选择`NaN`条目**，可以使用`pd.isnull()`（或其伴随的`pd.notnull()`）。这意味着要这样使用：

```python
reviews[pd.isnull(reviews.country)]
```

![image-20221126121259061](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126121259061.png)

**替换缺少的值**是一项常见操作。Pandas为这个问题提供了一个非常方便的方法：`fillna()`。`fillna()`提供了几种不同的策略来减轻这些数据。例如，我们可以简单地将每个`NaN`替换为“未知”：

```python
reviews.region_2.fillna("Unknown")
```

![image-20221126121429956](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126121429956.png)

或者我们可以用数据库中给定记录之后出现的第一个非空值来填充每个缺失的值。这就是所谓的回填策略。

或者，我们可能有一个非空值，希望替换它。例如，假设自该数据集发布以来，评审员Kerin O'Keefe已将她的Twitter账户从`@kerinokeefe`更改为`@kerino`。在数据集中反映这一点的一种方法是使用`replace()`方法：

```python
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")
```

![image-20221126121622285](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126121622285.png)

这里值得一提的是`replace()`方法，因为它可以方便地替换数据集中给定某种哨兵值的缺失数据：如`“Unknown”`、`“Undisclosed”`、`“Invalid”`等。

## Exercise: Data Types and Missing Values

### 介绍 

运行以下单元格以加载数据和一些实用程序函数。

```python
import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.data_types_and_missing_data import *
print("Setup complete.")
```

### Exercises

1. 数据集中`points`列的数据类型是什么？

   ```python
   dtype = reviews.points.dtype
   print(dtype)
   ```

   ```
   int64
   ```

2. 从`points`列中的条目创建`Series`，但将条目转换为字符串。提示：字符串在原生Python中是`str`。

   ```python
   point_strings = reviews.points.astype('str')
   print(point_strings)
   ```

   ![image-20221126122715316](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126122715316.png)

3. 有时价格列为空。数据集中有多少评论缺少价格？

   ```python
   missing_price_reviews = reviews[reviews.price.isnull()]
   n_missing_prices = len(missing_price_reviews)
   # Cute alternative solution: if we sum a boolean series, True is treated as 1 and False as 0
   n_missing_prices = reviews.price.isnull().sum()
   # or equivalently:
   n_missing_prices = pd.isnull(reviews.price).sum()
   ```

   ![image-20221126123151012](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126123151012.png)

4. 最常见的葡萄酒产区是什么？创建一个序列，计算每个值在`region_1`字段中出现的次数。此字段通常缺少数据，因此请将缺少的值替换为“未知”。按降序排序。您的输出应该如下所示：

   ![image-20221126122301678](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126122301678.png)

   ```python
   reviews_per_region = reviews.region_1.fillna('Unknown').value_counts().sort_values(ascending=False)
   ```

   ![image-20221126123645217](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126123645217.png)

### 继续

转到重命名和合并。