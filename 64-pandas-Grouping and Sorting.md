# 64-pandas-Grouping and Sorting

提升你的洞察力水平。数据集越复杂，这就越重要

## 介绍

映射允许我们转换`DataFrame`或`Series`中的数据，一次为整个列转换一个值。然而，通常我们希望对数据进行分组，然后针对数据所在的组执行特定的操作。 正如您将了解到的，我们使用`groupby()`操作来完成此操作。我们还将介绍一些其他主题，例如为`DataFrame`编制索引的更复杂的方法，以及如何对数据进行排序。

**To start the exercise for this topic, please click [here](https://www.kaggle.com/kernels/fork/598715).**

## 分组分析(Groupwise analysis)

到目前为止，我们一直在大量使用的一个函数是`value_counts()`函数。我们可以通过执行以下操作来复制`value_counts()`的作用：

```python
import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
```

```python
reviews.groupby('points').points.count()
```

![image-20221124213008008](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221124213008008.png)

`groupby()`创建了一组评论，为给定的葡萄酒分配了相同的分值。然后，对于这些组中的每一个，我们抓住`points()`列并计算它出现的次数。`value_counts()`只是此`groupby()`操作的快捷方式。 我们可以使用以前使用过的任何汇总函数处理这些数据。例如，要获得每个点值类别中最便宜的葡萄酒，我们可以执行以下操作：

```python
reviews.groupby('points').price.min()
```

![image-20221124213302466](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221124213302466.png)

您可以将我们生成的每个组视为`DataFrame`的一部分，其中只包含值匹配的数据。我们可以使用`apply()`方法直接访问这个`DataFrame`，然后我们可以以我们认为合适的任何方式处理数据。例如，这里有一种方法可以选择数据集中每个酒庄的第一款葡萄酒的名称：

```python
reviews.groupby('winery').apply(lambda df: df.title.iloc[0])
```

```
winery
1+1=3                          1+1=3 NV Rosé Sparkling (Cava)
10 Knots                 10 Knots 2010 Viognier (Paso Robles)
                                  ...                        
àMaurice    àMaurice 2013 Fred Estate Syrah (Walla Walla V...
Štoka                         Štoka 2009 Izbrani Teran (Kras)
Length: 16757, dtype: object
```

对于更细粒度的控件，还可以按多个列进行分组。例如，以下是我们如何按国家和省份挑选最好的葡萄酒：

```python
reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])
```

![image-20221124213700952](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221124213700952.png)

另一个值得一提的`groupby()`方法是`agg()`，它允许您同时在`DataFrame`上运行一系列不同的函数。例如，我们可以生成数据集的简单统计摘要，如下所示：

```python
reviews.groupby(['country']).price.agg([len, min, max])
```

![image-20221124213857506](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221124213857506.png)

有效地使用`groupby()`将允许您使用数据集做许多非常强大的事情。

## Multi-indexes

在迄今为止我们看到的所有示例中，我们一直在使用带有单个标签索引的`DataFrame`或`Series`对象。`groupby()`略有不同，因为根据我们运行的操作，它有时会导致所谓的多索引。 

多索引与常规索引的不同之处在于它具有多个级别。例如：

```python
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
countries_reviewed
```

![image-20221124214057175](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221124214057175.png)

```python
mi = countries_reviewed.index
type(mi)
```

```
pandas.core.indexes.multi.MultiIndex
```

多指数(Multi-indices)有几种处理其分层结构的方法，而单级指数没有这种方法。它们还需要两个级别的标签来检索值。对于熊猫新手来说，处理多索引输出是一个常见的“陷阱”。 

熊猫文档的“多索引/高级选择( [MultiIndex / Advanced Selection](https://pandas.pydata.org/pandas-docs/stable/advanced.html) )”部分详细介绍了多索引的用例以及使用它们的说明。 

然而，通常情况下，您最常用的多索引方法是用于转换回常规索引的方法，即`reset_index()`方法：

```python
countries_reviewed.reset_index()
```

![image-20221124214347618](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221124214347618.png)

## Sorting

再次查看`countries_reviewed`，我们可以看到分组按索引顺序而不是按值顺序返回数据。也就是说，当输出`groupby`的结果时，行的顺序取决于索引中的值，而不是数据中的值。 

为了按照需要的顺序获取数据，我们可以自己对其进行排序。`sort_values()`方法对此很方便。

```python
countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len')
```

![image-20221124214523107](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221124214523107.png)

`sort_values()`默认为升序排序，其中最低值优先。然而，大多数时候我们需要降序排序，其中较高的数字优先。这是如此：

```python
countries_reviewed.sort_values(by='len', ascending=False)
```

![image-20221124214600145](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221124214600145.png)

要按索引值排序，请使用伴随方法`sort_index()`。此方法具有相同的参数和默认顺序：

```python
countries_reviewed.sort_index()
```

![image-20221124214646669](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221124214646669.png)

最后，要知道您可以一次按多个列排序：

```python
countries_reviewed.sort_values(by=['country', 'len'])
```

![image-20221124214722036](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221124214722036.png)

## 轮到你了 

如果你还没有开始练习，你可以在这里开始。

## Exercise: Grouping and Sorting

### 介绍 

在这些练习中，我们将对数据集应用分组分析。

在运行练习之前，运行下面的代码单元以加载数据。

```python
import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
#pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.grouping_and_sorting import *
print("Setup complete.")
```

### Exercises

1. 数据集中最常见的葡萄酒评论者是谁？创建一个系列，其索引为数据集中的`taster_twitter_handle`类别，其值计算每个人写的评论数。

   ```python
   reviews_written = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
   # or
   #reviews_written = reviews.groupby('taster_twitter_handle').size()
   ```

   

2. 我能花一定的钱买到的最好的葡萄酒是什么？创建一个系列，该系列的指数是葡萄酒价格，其值是葡萄酒在评审中给出的最高点数。按价格升序排序（这样，4.0美元位于顶部，3300.0美元位于底部）。

   ```python
   best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()
   ```

   或者

   ```python
   points_by_price = reviews.groupby(['price']).points.agg([max])
   best_rating_per_price = points_by_price['max'].sort_index()
   ```

   

3. 每种葡萄酒的最低和最高价格是多少？创建一个`DataFrame`，其索引是数据集中的品种类别，其值是其最小值和最大值。

   ```python
   price_extremes = reviews.groupby('variety').price.agg([min,max])
   ```

   

4. 最贵的葡萄酒品种是什么？创建一个变量`sorted_variades`，其中包含上一个问题中的数据帧副本，其中根据最低价格，然后根据最高价格，按降序对品种进行排序（以打破联系）。

   ```python
   sorted_varieties = price_extremes.sort_values(by=['min','max'],ascending = False)
   ```

   

5. 创建一个系列，其索引为审阅者，其值为该审阅者给出的平均审阅分数。提示：您需要`taster_name`和`points`列。

   ```python
   reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
   ```

   不同评审员分配的平均分数是否存在显著差异？运行下面的单元格，使用describe（）方法查看值范围的摘要。

   ```python
   reviewer_mean_ratings.describe()
   ```

   ![image-20221124231810192](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221124231810192.png)

6. 最常见的国家和品种组合是什么？创建一个系列，其索引为`{country，variety}`对的多索引。例如，在美国生产的黑比诺应该对应于`{“US”，“pinot noir”}`。根据葡萄酒计数按降序对系列中的值进行排序。

   ```python
   country_variety_counts = reviews.groupby(['country','variety']).variety.count().sort_values(ascending=False)
   ```

   或者

   ```python
   country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)
   ```

   ![image-20221124232815430](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221124232815430.png)

### 继续前进 

转到数据类型和缺少的数据。