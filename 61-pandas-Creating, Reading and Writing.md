# 61-pandas-Creating, Reading and Writing

## 介绍 

在本微课程中，您将了解panda的所有知识，panda是最流行的数据分析Python库。 

在此过程中，您将使用真实世界的数据完成几个动手练习。我们建议您在阅读相应教程的同时进行练习。 

要开始第一个练习，请单击此处。 

在本教程中，您将学习如何创建自己的数据，以及如何使用已经存在的数据。 

## 入门 

要使用panda，通常从以下代码行开始。

```python
import pandas as pd
```

## 正在创建数据 

panda中有两个核心对象：`DataFrame`和Series。 

### 数据帧(`DataFrame`) 

`DataFrame`是一个表。它包含一个单独条目的数组，每个条目都有一个特定的值。每个条目对应一行（或记录）和一列。 例如，考虑以下简单的`DataFrame`：

```python
pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
```

![image-20221122171959766](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122171959766.png)

在本例中，“0，No”条目的值为131。“0，Yes”条目的数值为50，依此类推。 

`DataFrame`条目不限于整数。例如，这里有一个`DataFrame`，其值为字符串：

```python
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})
```

![image-20221122172052185](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122172052185.png)

我们使用`pd.DataFrame（）`构造函数来生成这些`DataFrame`对象。声明新字典的语法是一个字典，其键是列名（本例中为Bob和Sue），其值是条目列表。这是构造新`DataFrame`的标准方法，也是您最可能遇到的方法。

字典列表构造函数为列标签赋值，但只对行标签使用从0（0，1，2，3，…）开始的递增计数。有时这是可以的，但通常我们会想自己分配这些标签。 

`DataFrame`中使用的行标签列表称为**索引(index)**。我们可以在构造函数中使用索引参数为其赋值：

```python
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])
```

#### Series 

相反，`Series`是一系列数据值。如果`DataFrame`是表，则`Series`是列表。事实上，您可以创建一个列表：

```python
pd.Series([1, 2, 3, 4, 5])
```

![image-20221122172336505](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122172336505.png)

Series本质上是`DataFrame`的一列。因此，可以使用索引参数以与之前相同的方式将行标签分配给Series。但是，系列没有列名，它只有一个整体名称`name`：

```python
pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
```

```
2015 Sales    30
2016 Sales    35
2017 Sales    40
Name: Product A, dtype: int64
```

`Series`和`DataFrame`密切相关。将`DataFrame`想象成实际上只是一堆“粘在一起”的系列是很有帮助的。我们将在本教程的下一节中看到更多内容。

## 读取数据文件

能够手动创建`DataFrame`或`Series`非常方便。但是，大多数时候，我们实际上不会用手创建自己的数据。相反，我们将处理已经存在的数据。 

数据可以以多种不同的形式和格式存储。到目前为止，最基本的是简单的CSV文件。当您打开CSV文件时，会看到如下内容：

```python
Product A,Product B,Product C,
30,21,9,
35,34,1,
41,11,11
```

因此，CSV文件是一个用逗号分隔的值表。因此得名：“逗号分隔值”或CSV。

现在，让我们抛开我们的玩具数据集，看看当我们将其读入DataFrame时，真实的数据集是什么样子。我们将使用pd.read_csv（）函数将数据读入DataFrame。这是如此：

```python
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")
```

我们可以使用`shape`属性检查生成的DataFrame的大小：

```python
wine_reviews.shape
```

```
(129971, 14)
```

因此，我们的新DataFrame有130000条记录，分布在14个不同的列中。差不多有200万条！ 

我们可以使用head（）命令检查结果DataFrame的内容，该命令获取前五行：

```python
wine_reviews.head()
```

![image-20221122172749877](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122172749877.png)

`pd.read_csv()`函数具有很好的特性，可以指定30多个可选参数。例如，您可以在这个数据集中看到CSV文件有一个内置索引，panda并没有自动获取该索引。为了使panda使用该列作为索引（而不是从头创建一个新列），我们可以指定一个`index_col`。

```python
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
```

![image-20221122172934680](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122172934680.png)

## Exercise: Creating, Reading and Writing

### 介绍 

大多数数据分析项目的第一步是读取数据文件。在本练习中，您将通过手动和读取数据文件来创建Series和DataFrame对象。 

运行下面的代码单元以加载您需要的库（包括检查答案的代码）。

```python
import pandas as pd
pd.set_option('max_rows', 5)
from learntools.core import binder; binder.bind(globals())
from learntools.pandas.creating_reading_and_writing import *
print("Setup complete.")
```

### Exercises

1. 在下面的单元格中，创建如下所示的DataFrame果实：

   ![image-20221123092650634](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123092650634.png)

   ```python
   # Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.
   fruits = pd.DataFrame([[30,21]],columns=['Apples','Bananas'])
   
   fruits
   ```

   ![image-20221123093137386](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123093137386.png)

2. 创建与下图匹配的数据帧`fruit_sales`：

   ![image-20221123093213397](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123093213397.png)

   ```python
   # Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
   fruit_sales = pd.DataFrame(
       [[35,21],[41,34]],
       columns=['Apples', 'Bananas'],
       index=['2017 Sales','2018 Sales'])
   
   fruit_sales
   ```

   ![image-20221123093506866](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123093506866.png)

3. 创建一个系列的可变成分，其外观如下：

   ```
   Flour     4 cups
   Milk       1 cup
   Eggs     2 large
   Spam       1 can
   Name: Dinner, dtype: object
   ```

   ```python
   quantities = ['4 cups', '1 cup', '2 large', '1 can']
   items = ['Flour', 'Milk', 'Eggs', 'Spam']
   recipe = pd.Series(quantities, index=items, name='Dinner')
   recipe
   ```

   ![image-20221123093848987](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123093848987.png)

4. 将以下葡萄酒评论csv数据集读入名为reviews的DataFrame中：

   ![image-20221123093937562](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123093937562.png)

   The filepath to the csv file is `../input/wine-reviews/winemag-data_first150k.csv`. The first few lines look like:

   ```
   ,country,description,designation,points,price,province,region_1,region_2,variety,winery
   0,US,"This tremendous 100% varietal wine[...]",Martha's Vineyard,96,235.0,California,Napa Valley,Napa,Cabernet Sauvignon,Heitz
   1,Spain,"Ripe aromas of fig, blackberry and[...]",Carodorum Selección Especial Reserva,96,110.0,Northern Spain,Toro,,Tinta de Toro,Bodega Carmen Rodríguez
   ```

   ```python
   reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',header='infer',index_col=0)
   
   reviews
   ```

   ![image-20221123094413477](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123094413477.png)

5. 运行下面的单元格以创建并显示名为animals的DataFrame：

   ```python
   animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
   animals
   ```

   ![image-20221123094503500](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221123094503500.png)

   在下面的单元格中，编写代码将此DataFrame保存为csv文件，文件名为cows_and_goats.csv。

   ```python
   animals.to_csv('cows_and_goats.csv')
   ```

### 继续前进 

继续学习索引、选择和分配。