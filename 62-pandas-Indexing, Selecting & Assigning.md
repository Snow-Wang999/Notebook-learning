# 62-pandas-Indexing, Selecting & Assigning

专业数据科学家每天要做几十次。你也可以！

## 介绍 

在您将要运行的几乎所有数据操作中，选择pandas `DataFrame`或`Series`的特定值都是一个隐含的步骤，因此在Python中处理数据时首先需要学习的一件事就是如何快速有效地选择与您相关的数据点。

```python
import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option('max_rows', 5)
```

**To start the exercise for this topic, please click [here](https://www.kaggle.com/kernels/fork/587910).**

## Native accessors(本机访问者)

原生Python对象提供了索引数据的好方法。熊猫把所有这些都带了过来，这有助于让它很容易上手。 

考虑此DataFrame：

```python
reviews
```

![image-20221123095646234](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123095646234.png)

129971 rows × 13 columns

在Python中，我们可以通过将对象作为属性访问来访问它的属性。例如，`book`对象可能具有`title`属性，我们可以通过调用`book.title`来访问该属性。panda `DataFrame`中的列的工作方式大致相同。 

因此，为了访问审查的国家属性，我们可以使用：

```python
reviews.country
```

![image-20221123095849003](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123095849003.png)

如果我们有一个Python字典，我们可以使用索引`([])`运算符访问它的值。我们可以对`DataFrame`中的列执行同样的操作：

```python
reviews['country']
```

![image-20221123100003001](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123100003001.png)

这是从`DataFrame`中选择特定系列的两种方法。两者在语法上都不比另一个有效，但索引运算符[]确实有一个优点，即它可以处理其中包含保留字符的列名（例如，如果我们有国家/地区`country providence`列，`reviews.country providence`将不起作用）。 

pandas Series看起来有点像一本花哨的字典吗？实际上，要深入到一个特定的值，我们只需要再次使用索引运算符`[]`，这就不足为奇了：

```python
reviews['country'][0]
```

```
'Italy'
```

## Indexing in pandas

索引运算符(index operator)和属性选择(attribute selection)很好，因为它们的工作方式与Python生态系统的其他部分一样。作为一个新手，这使它们易于学习和使用。然而，panda有自己的访问器操作符`loc`和`iloc`。对于更高级的操作，这些是您应该使用的操作。

### 基于索引的选择(Index-based selection)

Pandas索引采用两种模式之一。第一种是**基于索引的选择(index-based selection**)：根据数据中的数字位置选择数据。`iloc`遵循这种范式(paradigm)。

要选择`DataFrame`中的第一行数据，我们可以使用以下方法：

```python
reviews.iloc[0]
```

```
country                                                    Italy
description    Aromas include tropical fruit, broom, brimston...
                                     ...                        
variety                                              White Blend
winery                                                   Nicosia
Name: 0, Length: 13, dtype: object
```

`loc`和`iloc`都是**行第一，列第二**。这与我们在原生Python中所做的相反，即**列第一，行第二**。

这意味着检索行(retrieve rows)稍微容易一些，检索列稍微困难一些。要使用`iloc`获取列，我们可以执行以下操作：

```python
reviews.iloc[:, 0]
```

![image-20221123100757190](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123100757190.png)

同样来自原生Python的：`:`运算符本身意味着“一切”。然而，当与其他选择器组合时，它可以用于指示值的范围。例如，要仅从第一、第二和第三行中选择"国家/地区"列，我们将执行以下操作：

```python
reviews.iloc[:3, 0]
```

![image-20221123100913314](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123100913314.png)

或者，为了只选择第二个和第三个条目，我们将执行以下操作：

```python
reviews.iloc[1:3, 0]
```

![image-20221123100943801](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123100943801.png)

还可以传递列表：

```python
reviews.iloc[[0, 1, 2], 0]
```

![image-20221123101013109](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123101013109.png)

最后，值得一提的是，负数可以用于选择。这将从值的末尾开始向前计数。例如，这里是数据集的最后五个元素。

```python
reviews.iloc[-5:]
```

![image-20221123101046794](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123101046794.png)

### 基于标签的选择(Label-based selection)

属性选择的第二个范例是`loc`运算符所遵循的范例：基于标签的选择。在这种范式中，重要的是数据索引值，而不是其位置。 例如，为了获得评论中的第一个条目，我们现在将执行以下操作：

```python
reviews.loc[0, 'country']
```

```
'Italy'
```

`iloc`在概念上比`loc`简单，因为它忽略了数据集的索引。当我们使用`iloc`时，我们将数据集视为一个大矩阵（一个列表列表），我们必须根据位置对其进行索引。相比之下，`loc`使用索引中的信息来完成工作。由于数据集通常具有有意义的索引，因此使用`loc`通常更容易。例如，这里有一个使用`loc`更容易的操作：

```python
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
```

![image-20221123101339418](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123101339418.png)



### 在`loc`和`iloc`之间选择

在`loc`和`iloc`之间选择或转换时，有一个“`gotcha`”值得记住，那就是两种方法使用的索引方案略有不同。 

- `iloc`使用Python `stdlib`索引方案，其中包含范围的第一个元素，排除最后一个元素。因此`0:10`将选择条目`0,...,9`. 

- `loc`，同时，包含索引。因此`0:10`将选择条目`0,...,10`。 

为什么要改变？请记住，`loc`可以索引任何`stdlib`类型：例如字符串。如果我们有一个索引值为`Apples,...,Potatoes,...` 的`DataFrame`，并且我们想选择“Apples和Potatoes之间的所有按字母顺序排列的水果选项”，那么索引`df.loc['Apples':'Potatoes']`比索引`df.loc['Apples'，'Potatoet']`（字母表中t后面是s）这样的东西更容易。 

当`DataFrame`索引是一个简单的数字列表时，这尤其令人困惑，例如`0,...,1000`。`df.iloc[0:1000]`将返回1000个条目，而`df.loc[0:1000]`返回1001个！要使用`loc`获得1000个元素，您需要降低一个级别并请求`df.loc[0:999]`。 

否则，使用`loc`的语义与`iloc`的语义相同。

## 操纵索引(Manipulating the index)

基于标签的选择源于索引中的标签。关键的是，我们使用的索引不是一成不变的。我们可以以我们认为合适的任何方式操纵索引。 

可以使用`set_index()`方法来完成该任务。下面是将`set_index()`设置为`title`字段时发生的情况：

```python
reviews.set_index("title")
```

![image-20221123105439740](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123105439740.png)

如果您可以为数据集创建一个比当前索引更好的索引，这将非常有用。

## 条件选择(Conditional selection) 

到目前为止，我们一直在使用`DataFrame`本身的结构属性来索引数据的各个步骤。然而，为了用数据做有趣的事情，我们通常需要根据条件提出问题。 

例如，假设我们特别感兴趣的是意大利出产的比一般葡萄酒更好的葡萄酒。 

我们可以从检查每种葡萄酒是否为意大利葡萄酒开始：

```python
reviews.country == 'Italy'
```

![image-20221123105646334](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123105646334.png)

该操作根据每个记录的国家产生了一系列真/假布尔值。然后可以在`loc`内部使用此结果来选择相关数据：

```python
reviews.loc[reviews.country == 'Italy']
```

![image-20221123105747291](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123105747291.png)

此`DataFrame`有约20000行。最初的葡萄酒大约有130000瓶。这意味着大约15%的葡萄酒来自意大利。 

我们还想知道哪些比平均水平好。葡萄酒的评分标准为80-100分，因此这意味着葡萄酒的评分至少为90分。 

我们可以使用与号（&）将两个问题联系起来：

```python
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]
```

![image-20221123105842136](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123105842136.png)

假设我们会购买意大利产的葡萄酒，或是评级高于平均水平的葡萄酒。为此，我们使用管道`(|)`：

```python
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]
```

![image-20221123105952865](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123105952865.png)

Pandas附带了一些内置的条件选择器，其中两个我们将在这里重点介绍。 

第一个是`isin`。`isin`允许您选择值“in”值列表中的数据。例如，以下是我们如何使用它来选择仅来自意大利或法国的葡萄酒：

```python
reviews.loc[reviews.country.isin(['Italy', 'France'])]
```

![image-20221123110105845](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123110105845.png)

第二个是`isnull`（它的同伴`notnull`）。这些方法允许您高亮显示空（或非空）的值（`NaN`）。例如，要过滤掉数据集中缺少价格标签的葡萄酒，我们可以这样做：

```python
reviews.loc[reviews.price.notnull()]
```

![image-20221123110227396](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123110227396.png)

## 分配数据(Assigning data)

相反，将数据分配给DataFrame很容易。您可以指定一个常量值：

```python
reviews['critic'] = 'everyone'
reviews['critic']
```

![image-20221123110318311](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123110318311.png)

或者使用可迭代的值：

```python
reviews['index_backwards'] = range(len(reviews), 0, -1)
reviews['index_backwards']
```

![image-20221123110402241](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123110402241.png)

## Your turn-轮到你了 

If you haven't started the exercise, you can **[get started here](https://www.kaggle.com/kernels/fork/587910)**.

如果你还没有开始练习，你可以在这里开始。

## Exercise: Indexing, Selecting & Assigning

### 介绍 

在这组练习中，我们将使用葡萄酒评论数据集。

运行以下单元格以加载数据和一些实用程序函数（包括检查答案的代码）。

```python
import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.indexing_selecting_and_assigning import *
print("Setup complete.")
```

```python
reviews.head()
```

![image-20221123111013088](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123111013088.png)

### Exercises

1. 从评论中选择描述列，并将结果分配给变量`desc`。

   ```python
   # Your code here
   desc = reviews.description
   print(desc)
   ```

   ![image-20221123111155956](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123111155956.png)

2. 从`reviews`的描述列中选择第一个值，将其分配给变量first_description。

   ```python
   first_description = reviews.description.iloc[0]
   
   first_description
   ```

   注意，虽然这是获取`DataFrame`中的条目的首选方式，但许多其他选项将返回有效结果，例如`reviews.description.loc[0]`，`reviews.description[0]`，以及更多！

   ```
   "Aromas include tropical fruit, broom, brimstone and dried herb. The palate isn't overly expressive, offering unripened apple, citrus and dried sage alongside brisk acidity."
   ```

3. 从`reviews`中选择第一行数据（第一条记录），并将其分配给变量first_row。

   ```python
   first_row = reviews.iloc[0,:]
   # or
   #first_row = reviews.iloc[0]
   
   first_row
   ```

   ```
   country                                                    Italy
   description    Aromas include tropical fruit, broom, brimston...
                                        ...                        
   variety                                              White Blend
   winery                                                   Nicosia
   Name: 0, Length: 13, dtype: object
   ```

4. 从`reviews`中的`description`列中选择前10个值，将结果分配给变量`first_descriptions`。 提示：将输出格式化为熊猫系列。

   ```python
   first_descriptions = reviews.description[0:10]
   #first_descriptions = reviews.description.iloc[:10]
   #Note that many other options will return a valid result, such as **desc.head(10)** and **reviews.loc[:9, "description"]**.
   
   first_descriptions
   ```

   ![image-20221123112133541](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123112133541.png)

5. 选择索引标签为1、2、3、5和8的记录，将结果分配给变量sample_reviews。 换句话说，生成以下`DataFrame`：

   ![image-20221123112212334](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123112212334.png)

   ```python
   sample_reviews = reviews.iloc[[1,2,3,5,8],:]
   #indices = [1, 2, 3, 5, 8]
   #sample_reviews = reviews.loc[indices]
   
   sample_reviews
   ```

   ![image-20221123112438544](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123112438544.png)

6. 创建一个变量`df`，其中包含索引标签为0、1、10和100的记录的`country`、`province`、`region_1`和`region_2`列。换句话说，生成以下`DataFrame`：

   ![image-20221123112544536](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123112544536.png)

   ```python
   index_1 = [0,1,10,100]
   columns_1 = ['country','province','region_1','region_2']
   df = reviews.loc[index_1,columns_1]
   #.loc[],此处是方括号
   df
   ```

   ![image-20221123113016089](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123113016089.png)

7. 创建一个变量`df`，其中包含前100条记录的国家和地区列。

   提示：您可以使用`loc`或`iloc`。在回答这个问题和接下来的几个问题时，请保持教程中描述的以下“gotcha”：

   > `iloc`使用Python `stdlib`索引方案，其中包含范围的第一个元素，排除最后一个元素。同时，`loc`包含索引。 
   >
   > 当`DataFrame`索引是一个简单的数字列表时，这尤其令人困惑，例如`0，…，1000`。`iloc[0:1000]`将返回1000个条目，而`df.loc[0:1000]`返回1001个！要使用`loc`获得1000个元素，您需要降低一个要求`df.iloc[0:999]`。

   ```python
   cols = ['country','variety']
   df = reviews.loc[0:99,cols]
   #or
   #cols_idx = [0, 11]
   #df = reviews.iloc[:100, cols_idx]
   
   df
   ```

   ![image-20221123113623404](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123113623404.png)

8. 创建包含意大利葡萄酒评论的`DataFrame italian_wines`。提示：评论。国家等于什么？

   ```python
   italian_wines = reviews.loc[reviews.country == 'Italy']
   italian_wines = reviews[reviews.country == 'Italy']
   ```

   

9. 创建一个`DataFrame` `top_oceania_wines`，其中包含澳大利亚或新西兰葡萄酒的所有评论，至少95分（满分100分）。

   ```python
   top_oceania_wines = reviews[
       (reviews.points>=95)
       &(reviews.country.isin(['Australia','New Zealand']))]
   
   top_oceania_wines
   ```

   ![image-20221123114328759](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123114328759.png)

### 继续前进 

继续了解汇总函数和映射(summary functions and maps)。