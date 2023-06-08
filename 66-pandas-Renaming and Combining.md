# 66-pandas-Renaming and Combining

数据来源很多。帮助所有这些一起变得有意义

## 介绍 

通常，数据会以列名、索引名或其他我们不满意的命名约定出现在我们面前。在这种情况下，您将学习如何使用panda函数将违规条目的名称更改为更好的名称。 

您还将探索如何组合来自多个`DataFrame and/or Series`的数据。 

要开始本主题的练习，请单击此处。

## Renaming

我们将在这里介绍的第一个函数是`rename`，它允许您更改索引名和/或列名。例如，要将数据集中的`points`列更改为`score`，我们将执行以下操作：

```python
import pandas as pd
pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
```

```python
reviews.rename(columns={'points': 'score'})
```

![image-20221126182924080](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126182924080.png)

`rename()`允许您通过分别指定`index`或者`column`关键字参数来重命名索引或列值。它支持多种输入格式，但通常Python字典是最方便的。下面是一个使用它重命名索引中某些元素的示例。

```python
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})
```

![image-20221126183143268](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126183143268.png)

您可能会经常重命名列，但很少重命名索引值。为此，`set_index()`通常更方便。 

行索引和列索引都可以有自己的`name`属性。可以使用补充的`rename_axis()`方法来更改这些名称。例如：

```python
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')
```

![image-20221126183404144](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126183404144.png)

## Combining

在数据集上执行操作时，我们有时需要以非平凡的方式组合不同的`DataFrame`和/或`Series`。

Pandas有三个核心方法来实现组合不同的`DataFrame`和/或`Series`的列（按照增加复杂性的顺序）：

- `concat()`
- `join()`
- `merge()`

`merge()`所能做的大部分工作也可以通过`join()`更简单地完成，因此我们将省略它，并在这里集中讨论前两个函数。 

最简单的组合方法是`concat()`。给定一个元素列表，此函数将沿着一个轴将这些元素糊粘在一起。 

当数据位于不同的`DataFrame`或Series对象中但具有相同的字段（列）时，这非常有用。一个例子是YouTube视频数据集( [YouTube Videos dataset](https://www.kaggle.com/datasnaek/youtube-new))，它根据原产国（例如，本例中的加拿大和英国）将数据进行分割。如果我们想同时研究多个国家，我们可以使用`concat()`将它们放在一起：

```python
canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")

pd.concat([canadian_youtube, british_youtube])
```

![image-20221126183522418](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126183522418.png)

就复杂性而言，最中间的组合器是`join()`。`join()`允许您组合具有共同索引的不同`DataFrame`对象。例如，要删除恰好在加拿大和英国同一天流行的视频，我们可以执行以下操作：

```python
left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK')
```

![image-20221126183553773](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126183553773.png)

`lsuffix`和`rsuffix`参数在这里是必需的，因为数据在英国和加拿大数据集中具有相同的列名。如果这不是真的（因为，比如说，我们已经事先给它们改名了），我们就不需要它们了。

## 轮到你了

如果你还没有开始练习，你可以在这里开始。

## Exercise: Renaming and Combining

### 介绍 

运行以下单元格以加载数据和一些实用程序函数。

```python
import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.renaming_and_combining import *
print("Setup complete.")
```

### Exercises

通过运行以下单元格查看数据的前几行：

```python
reviews.head()
```

![image-20221126185456732](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126185456732.png)

1. `region_1`和`region_2`是数据集中区域设置列的非格式名称。创建`reviews`的副本，将这些列分别重命名为`region`和`locale`。

   ```python
   renamed = reviews.rename(columns={'region_1':'region','region_2':'locale'})
   # 或者
   # renamed = reviews.rename(columns=dict(region_1='region', region_2='locale'))
   ```

   ![image-20221126190317137](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126190317137.png)

2. 将数据集中的索引名称设置为`wines`。

   ```python
   reindexed = reviews.rename_axis('wines',axis='rows')
   ```

   ![image-20221126190242980](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126190242980.png)

3. Reddit上的东西数据集( [Things on Reddit](https://www.kaggle.com/residentmario/things-on-reddit/data) )包括来自Reddit.com上排名最高的论坛（“subreddit”）的产品链接。

   运行下面的单元格以加载/r/gaming子插件中提到的产品的数据帧和r//movies子插件中提及的产品的另一个数据帧。

   ```python
   gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
   gaming_products['subreddit'] = "r/gaming"
   movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
   movie_products['subreddit'] = "r/movies"
   ```

   创建任一子reddit上提到的产品的`DataFrame`。

   ```python
   #combined_products = pd.concat([gaming_products,movie_products],axis=0)
   combined_products = pd.concat([gaming_products, movie_products])
   combined_products.head(-5)
   ```

   ![image-20221126191305978](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126191305978.png)

4. Kaggle上的Powerlifting数据库数据集([Powerlifting Database](https://www.kaggle.com/open-powerlifting/powerlifting-database) )包括一个用于电力提升会议(powerlifting meets)的CSV表和一个单独的用于电力提升竞争对手(powerlifting competitors)的CSV表。运行下面的单元格将这些数据集加载到数据帧中：

   ```python
   powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
   powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
   ```

   两个表都包含对`MeetID`的引用，`MeetID`是数据库中包含的每个会议（比赛）的唯一密钥。使用此选项，生成将两个表合并为一个表的数据集。

   ```python
   left = powerlifting_meets.set_index(['MeetID'])
   right = powerlifting_competitors.set_index(['MeetID'])
   powerlifting_combined = left.join(right)
   powerlifting_combined.head()
   ```

   ![image-20221126192315425](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221126192315425.png)



### 祝贺 

你已经完成了熊猫微课程。许多数据科学家认为，熊猫的效率是他们拥有的最有用和最实用的技能，因为它可以让你在任何项目中快速进步。 

如果您想将新技能应用于检查地理空间数据，我们鼓励您查看我们的地理空间分析(**[Geospatial Analysis](https://www.kaggle.com/learn/geospatial-analysis)**)微课程。 

您还可以通过参加Kaggle竞赛( **[Kaggle Competition](https://www.kaggle.com/competitions)** )或回答您使用Kaggle数据集(**[Kaggle Datasets](https://www.kaggle.com/datasets)**.)感兴趣的问题来利用您的熊猫技能。