# 63-pandas-Summary Functions and Maps

从数据中提取见解。

## 介绍

在上一个教程中，我们学习了如何从DataFrame或Series中选择相关数据。正如我们在练习中所演示的，从数据表示中提取正确的数据对于完成工作至关重要。 

然而，数据并非总是以我们想要的格式从内存中取出。有时，我们必须自己做更多的工作，以便为手头的任务重新格式化。本教程将介绍我们可以应用于数据以获得“恰到好处”的输入的不同操作。

要开始本主题的练习，请单击此处。

我们将使用《葡萄酒杂志》的数据进行演示。

```python
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
```

```python
reviews
```

![image-20221123175300394](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123175300394.png)

## Summary functions

Pandas提供了许多简单的“摘要函数”（非官方名称），这些函数以某种有用的方式重新构造数据。例如，考虑`describe()`方法：

```python
reviews.points.describe()
```

![image-20221123175410042](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123175410042.png)

此方法生成给定列属性的高级摘要。它是类型感知的，这意味着它的输出会根据输入的数据类型而改变。上述输出仅适用于数值数据；对于字符串数据，我们得到的是：

```python
reviews.taster_name.describe()
```

![image-20221123175439188](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123175439188.png)

如果您想获得有关`DataFrame`或`Series`中某列的特定简单摘要统计信息，通常会有一个有用的panda函数来实现。 例如，要查看分配分数的平均值（例如，平均评分的葡萄酒表现如何），我们可以使用`mean()`函数：

```python
reviews.points.mean()
```

```
88.44713820775404
```

要查看唯一值的列表，我们可以使用`unique()`函数：

```python
reviews.taster_name.unique()
```

```
array(['Kerin O’Keefe', 'Roger Voss', 'Paul Gregutt',
       'Alexander Peartree', 'Michael Schachner', 'Anna Lee C. Iijima',
       'Virginie Boone', 'Matt Kettmann', nan, 'Sean P. Sullivan',
       'Jim Gordon', 'Joe Czerwinski', 'Anne Krebiehl\xa0MW',
       'Lauren Buzzeo', 'Mike DeSimone', 'Jeff Jenssen',
       'Susan Kostrzewa', 'Carrie Dykes', 'Fiona Adams',
       'Christina Pickard'], dtype=object)
```

要查看唯一值的列表以及它们在数据集中出现的频率，我们可以使用`value_counts()`方法：

```python
reviews.taster_name.value_counts()
```

![image-20221123175741575](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123175741575.png)

## Maps

映射(map)是一个从数学中借来的术语，用于表示一组值并将它们“映射”到另一组值的函数。在数据科学中，我们经常需要从现有数据中创建新的表示，或者将数据从现在的格式转换为我们希望以后使用的格式。映射(map)是处理这项工作的工具，它对完成工作非常重要！

您将经常使用两种映射方法。

`map()`是第一个，而且稍微简单一些。例如，假设我们希望将葡萄酒的得分重新平均计算为0。我们可以这样做：

```python
review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)
```

![image-20221123180215371](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123180215371.png)

传递给`map()`的函数应该期望`Series`中有一个值（在上面的示例中是一个点值），并返回该值的转换版本。`map()`返回一个新的`Series`，其中所有值都已由函数转换。 

如果我们想通过对每一行调用自定义方法来转换整个`DataFrame`，`apply()`是等效的方法。

```python
review_points_mean = reviews.points.mean()

def remean_points(row):
    row.points = row.points - review_points_mean
    return row

reviews.apply(remean_points, axis='columns')
```

![image-20221123180649521](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123180649521.png)

如果我们调用`reviews.apply`，`axis='index'`，那么我们需要提供一个函数来转换每一行，而不是传递一个函数。 

请注意，`map()`和`apply()`分别返回新的、转换后的`Series`和`DataFrames`。他们**不会修改他们调用的原始数据**。如果我们查看第一行`reviews`，我们可以看到它仍然具有原始的`points`数值。

```python
reviews.head(1)
```

![image-20221123180956938](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123180956938.png)

Pandas提供了许多内置的常见映射操作。例如，这里有一种更快的方法来记住我们的`points`列：

```python
review_points_mean = reviews.points.mean()
reviews.points - review_points_mean
```

![image-20221123181041721](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123181041721.png)

在这段代码中，我们在左侧的许多值（系列中的所有值）和右侧的一个值（平均值）之间执行操作。Pandas查看了这个表达式，发现我们必须从数据集中的每个值中减去平均值。 

Pandas还将了解如果我们在长度相等的系列之间执行这些操作，该怎么办。例如，在数据集中组合国家和地区信息的一种简单方法是：

```python
reviews.country + " - " + reviews.region_1
```

![image-20221123181135476](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123181135476.png)

这些操作符比`map()`或`apply()`更快，因为它们使用panda内置的加速功能。所有标准Python运算符（`>`、`<`、`==`等）都以这种方式工作。 

然而，它们不像`map()`或`apply()`那样灵活，后者可以做更高级的事情，比如应用条件逻辑，而这不能单独用加法和减法来完成。

## 轮到你了

如果你还没有开始练习，你可以在这里开始。

## Exercise: Summary Functions and Maps

介绍

现在，您可以更深入地了解数据了。 

运行以下单元格以加载数据和一些实用程序函数（包括检查答案的代码）。

```python
import pandas as pd
pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.summary_functions_and_maps import *
print("Setup complete.")

reviews.head()
```

![image-20221124203116980](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221124203116980.png)

## Exercises

1. `reviews` `DataFrame`中`points`数列的中值是多少？

   ```python
   median_points = reviews.points.median()
   ```

   

2. 数据集中代表了哪些国家？（您的答案不应包含任何重复项。）

   ```python
   countries = reviews.country.unique()
   ```

   

3. 每个国家出现在数据集中的频率是多少？创建一个系列`reviews_per_country`，将国家映射到该国家的葡萄酒评论数量。

   ```python
   reviews_per_country = reviews.country.value_counts()
   ```

   

4. 创建包含减去平均价格的`price`列版本的变量`centered_price`。 （注意：这种“居中”转换是应用各种机器学习算法之前的常见预处理步骤。）

   ```python
   centered_price = reviews.price - reviews.price.mean()
   ```

   

5. 我是一个经济的葡萄酒买家。哪种酒最便宜？创建一个变量`bargain_line`，其中包含数据集中点数与价格比率最高的葡萄酒的标题。

   ```python
   bargain_index = (reviews.points / reviews.price).idxmax()
   bargain_wine = reviews.loc[bargain_index,'title']
   ```

   

6. 描述一瓶葡萄酒时，你能用的词只有这么多。葡萄酒更可能是“热带”还是“果味”？创建一个Series `descriptor_counts`，计算这两个单词在数据集中的描述列中出现的次数。（为了简单起见，让我们忽略这些单词的大写版本。）

   ```python
   n_trop = reviews.description.map(lambda desc: 'tropical' in desc).sum()
   n_fruity = reviews.description.map(lambda desc: 'fruity' in desc).sum()
   descriptor_counts = pd.Series([n_trop,n_fruity],index=['tropical','fruity'])
   ```

   ```
   tropical    3607
   fruity      9090
   dtype: int64
   ```

7. 我们希望在我们的网站上发布这些葡萄酒评论，但从80到100分的评分系统太难理解了-我们希望将其转化为简单的星级评分。95分或更高的分数为3星，至少85分但低于95分的分数为2星。其他得分为1星。 此外，加拿大葡萄酒商协会在该网站上购买了大量广告，因此任何来自加拿大的葡萄酒都应自动获得3颗星，无论分数如何。 创建一系列star_ratings，其中包含与数据集中每个评论相对应的星星数。

   ```python
   def to_stars(row):
       if row.country == 'Canada':
           return 3
       elif row.points >= 95:
           return 3
       elif row.points >= 85:
           return 2
       else:
           return 1
   
   star_ratings=reviews.apply(to_stars,axis='columns')
   ```

   