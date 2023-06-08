# 32-Basic Data Exploration

## 使用 Pandas 熟悉您的数据

任何机器学习项目的第一步都是熟悉数据。为此，您将使用 Pandas 库。 Pandas 是数据科学家用于探索和处理数据的主要工具。大多数人在他们的代码中将 pandas 缩写为 pd。我们使用命令执行此操作

```py
import pandas as pd
```

Pandas 库中最重要的部分是 DataFrame。 DataFrame 包含您可能认为是表格的数据类型。这类似于 Excel 中的工作表或 SQL 数据库中的表。 Pandas 具有强大的方法，可以处理您希望使用此类数据执行的大多数操作。 例如，我们将查看有关澳大利亚墨尔本房价的数据。在动手练习中，您将对新数据集应用相同的过程，该数据集包含爱荷华州的房价。 示例（墨尔本）数据位于文件路径 ../input/melbourne-housing-snapshot/melb_data.csv 中。 我们使用以下命令加载和探索数据：

```python
# save filepath to variable for easier access
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
melbourne_data.describe()
```

![image-20221106145516255](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106145516255.png)

![image-20221106150007998](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106150007998.png)

## 解释数据描述(Interpreting Data Description)

![image-20221106150044134](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106150044134.png)

结果显示原始数据集中每列有 8 个数字。

- 计数(count)，显示有多少行具有非缺失值。 出现缺失值的原因有很多。例如，测量一居室房屋时，不会收集第二居室的大小。我们将回到丢失数据的主题。 

- 平均值(mean)，也就是平均值。

- 标准偏差(std)，它衡量数值在数值上的分布程度。
- min, 25%, 50%, 75%, max，  想象一下从最低到最高对每一列进行排序。第一个（最小）值是最小值。如果您遍历列表的四分之一，您会发现一个大于 25% 的值且小于 75% 的值的数字。那是 25% 的值（发音为“25th percentile”）。第 50 个和第 75 个百分位数的定义类似，最大值为最大数。

## Exercise: Explore Your Data

本练习将测试您阅读数据文件和理解有关数据的统计信息的能力。 

在后面的练习中，您将应用技术来过滤数据、构建机器学习模型并迭代地改进您的模型。 课程示例使用墨尔本的数据。

为确保您可以自己应用这些技术，您必须将它们应用到一个新数据集（来自爱荷华州的房价）。 

练习使用“笔记本”编码环境。如果您不熟悉笔记本电脑，我们有一个 90 秒的介绍视频。

### Exercises

```python
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex2 import *
print("Setup Complete")
```

### 第 1 步：加载数据

```python
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)
```

第 2 步：查看数据

使用您学习的命令查看数据的摘要统计信息。然后填写变量回答下列问题

```python
# Print summary statistics in next line
home_data.describe()
```

![image-20221106152717927](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106152717927.png)

```python
# What is the average lot size (rounded to nearest integer)?
avg_lot_size = 10517

# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = 12
```

## 想想你的数据 

您数据中最新的房子并不是那么新。对此有一些可能的解释： 

1. 他们还没有建造收集这些数据的新房子。 
2. 数据是很久以前收集的。数据发布后建造的房屋不会出现。 

如果原因是上面的解释 #1，这是否会影响您对使用此数据构建的模型的信任？如果是原因 #2 呢？ 

您如何深入研究数据以查看哪种解释更合理？ 

查看此讨论主题以了解其他人的想法或添加您的想法。

Check out this **[discussion thread](https://www.kaggle.com/learn-forum/60581)** to see what others think or to add your ideas.