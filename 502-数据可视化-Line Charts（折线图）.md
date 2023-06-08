# 502-数据可视化-Line Charts（折线图）

可视化一段时间内的趋势

现在您已经熟悉了编码环境，是时候学习如何制作自己的图表了！ 

在本教程中，您将学习足够的Python来创建专业外观的折线图。然后，在下面的练习中，您将运用新技能来处理真实世界的数据集。

## 设置笔记本 

我们首先设置编码环境。（此代码是隐藏的，但您可以通过单击此文本右下方的“代码”按钮来取消隐藏。）

```python
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
```

```
Setup Complete
```

## 选择数据集 

本教程的数据集跟踪音乐流服务[Spotify](https://en.wikipedia.org/wiki/Spotify)上的全球每日流。我们专注于2017年和2018年的五首流行歌曲：

1. 《你的形状(Shape of You)》，作者：Ed Sheeran（[link](https://bit.ly/2tmbfXp)） 
2. “慢慢地(Despacito)”，作者Luis Fonzi（[link](https://bit.ly/2vh7Uy6)） 
3. 《就像这样的东西(Something Just Like This)》，由Chainsmokers和酷玩乐队（[link](https://bit.ly/2OfSsKk)） 
4. 《人性(HUMBLE)》肯德里克·拉马尔（[link](https://bit.ly/2YlhPw4)） 
5. 《难忘(Unforgettable)》，法国蒙大拿州（[link](https://bit.ly/2oL7w8b)）

![image-20221210225756173](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210225756173.png)

注意，第一个出现的日期是2017年1月6日，与艾德·希兰（Ed Sheeran）的《你的形状》（the Shape of You）的上映日期相对应。使用该表，您可以看到《你的形状》在发布当天在全球播放了12287078次。请注意，其他歌曲在第一行中缺少值，因为它们直到后来才发布！

## 加载数据 

正如您在上一教程中所了解的，我们使用`pd.read_csv`命令加载数据集。

```python
# Path of the file to read
spotify_filepath = "../input/spotify.csv"

# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)
```

运行上面两行代码的最终结果是，我们现在可以使用`spotify_data`访问数据集。

## 检查数据

我们可以使用您在上一教程中了解的head命令打印数据集的前五行。

```python
# Print the first 5 rows of the data
spotify_data.head()
```

![image-20221210230022344](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210230022344.png)

现在检查前五行是否与上面的数据集图像一致（当我们在Excel中看到它的样子时）。

> 空条目将显示为`NaN`，即“不是数字(Not a Number)”的缩写。

我们还可以通过只做一个小修改（其中.head（）变为.tail（））来查看数据的最后五行：

```python
# Print the last five rows of the data
spotify_data.tail()
```

![image-20221210230205171](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210230205171.png)

值得庆幸的是，每首歌每天都有数百万的全球流，我们可以继续绘制数据！

## 绘制数据 

现在数据集已加载到笔记本中，我们只需要一行代码即可制作折线图！

```python
# Line chart showing daily global streams of each song 
sns.lineplot(data=spotify_data)
```

![image-20221210230340895](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210230340895.png)

如上所述，代码行相对较短，有两个主要部分： 

- `sns.lineplot`告诉笔记本我们要创建一个折线图。 

  - 在本课程中学习的每个命令都将以`sns`开头，这表示该命令来自seaborn包。例如，我们使用`sns.lineplot`可制作折线图。很快，你就会知道我们使用`sns.barplot`和`sns.heatmap`分别制作条形图和热图。 

  - ```python
    # 折线图
    sns.lineplot
    # 条形图
    sns.barplot
    # 热图 
    sns.heatmap
    ```

- `data=spotify_data` 选择将用于创建图表的数据。

请注意，在创建折线图时，您将始终使用相同的格式，而随着新数据集的变化，唯一的变化就是数据集的名称。因此，例如，如果使用名为`financial_data`的不同数据集，代码行将显示如下： 

```python
sns.lineplot(data=financial_data)
```

有时我们还想修改一些额外的细节，比如图的大小和图表的标题。这些选项中的每一个都可以通过一行代码轻松设置。

```python
# Set the width and height of the figure
#设置图形的宽度和高度
plt.figure(figsize=(14,6))

# Add title
#添加标题
plt.title("Daily Global Streams of Popular Songs in 2017-2018")

# Line chart showing daily global streams of each song 
#显示每首歌曲每日全球流的折线图
sns.lineplot(data=spotify_data)
```

![image-20221210231145456](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210231145456.png)

第一行代码将图形的大小设置为14英寸（宽）乘6英寸（高）。要设置任何图形的大小，只需复制显示的同一行代码即可。然后，如果要使用自定义尺寸，请将提供的值14和6更改为所需的宽度和高度。 

第二行代码设置图形的标题。请注意，标题必须始终用引号（“…”）括起来！

## 绘制数据子集 

到目前为止，您已经学习了如何为数据集中的每一列绘制一条线。在本节中，您将学习如何绘制列的子集。 

我们将首先**打印所有列的名称**。这只需一行代码即可完成，只需交换数据集的名称（在本例中为spotify_data），即可适用于任何数据集。

```python
list(spotify_data.columns)
```

```
['Shape of You',
 'Despacito',
 'Something Just Like This',
 'HUMBLE.',
 'Unforgettable']
```

在下一个代码单元中，我们绘制与数据集中前两列相对应的行。

```python
# Set the width and height of the figure
plt.figure(figsize=(14,6))

# Add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")

# Line chart showing daily global streams of 'Shape of You'
sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")

# Line chart showing daily global streams of 'Despacito'
sns.lineplot(data=spotify_data['Despacito'], label="Despacito")

# Add label for horizontal axis
plt.xlabel("Date")
```

![image-20221210231639932](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210231639932.png)

前两行代码设置了图形的标题和大小（看起来应该很熟悉！）。 接下来的两行分别在折线图中添加一行。例如，考虑第一个，它为“你的形状”添加了一行： 

```python
# Line chart showing daily global streams of 'Shape of You'
sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")
```

这一行看起来与我们绘制数据集中每一行时使用的代码非常相似，但它有一些关键区别： 

- 我们没有设置`data=spotify_data`，而是设置`data=spotify_data['Shape of You']`。通常，为了只绘制一列，我们使用这种格式，将列的名称放在单引号中，并用方括号括起来。（为了确保正确指定列的名称，可以使用上面学习的命令打印所有列名称的列表。） 

- 我们还添加`label=“Shape of You”`以使线条显示在图例中并设置相应的标签。 

最后一行代码修改水平轴（或x轴）的标签，其中所需标签放在引号（“…”）中。 

## 接下来是什么？

将您的新技能用于编码练习！

## Exercise: Line Charts

在本练习中，您将使用您的新知识为现实场景提出解决方案。要成功，您需要将数据导入Python，使用数据回答问题，并生成折线图以了解数据中的模式。

### 脚本（Scenario）

你最近被聘来管理洛杉矶市的博物馆。您的第一个项目集中在下图中所示的四个博物馆。

![image-20221211155741697](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221211155741697.png)

您将利用洛杉矶数据门户（ [Data Portal](https://data.lacity.org/)）网站的数据，跟踪每个博物馆的每月访客。

![image-20221211155831985](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221211155831985.png)

### 安装程序 （Setup）

运行下一个单元以导入和配置完成练习所需的Python库。

```python
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
```

```
Setup Complete
```

以下问题将为您的工作提供反馈。运行以下单元格以设置反馈系统。

```python
# Set up code checking
import os
if not os.path.exists("../input/museum_visitors.csv"):
    os.symlink("../input/data-for-datavis/museum_visitors.csv", "../input/museum_visitors.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex2 import *
print("Setup Complete")
```

```
Setup Complete
```

### 步骤1：加载数据 

您的第一项任务是将洛杉矶博物馆访客数据文件读入`museum_data`。注意： 

- 数据集的文件路径存储为`museum_filepath`。请不要更改提供的文件路径值。 

- 用作行标签的列的名称为`Date`。（当文件在Excel中打开时，这可以在单元格A1中看到。）

为了帮助实现这一点，您可能会发现重新访问教程中的一些相关代码非常有用，我们已将其粘贴在下面：

```python
# Path of the file to read
spotify_filepath = "../input/spotify.csv"

# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)
```

您现在需要编写的代码看起来非常相似！

```python
# Path of the file to read
museum_filepath = "../input/museum_visitors.csv"

# Fill in the line below to read the file into a variable museum_data
museum_data = pd.read_csv(museum_filepath, index_col="Date", parse_dates=True)
```

### 步骤2：查看数据 

使用Python命令打印最后5行数据。

```
# Print the last five rows of the data 
museum_data.tail() # Your code here
```

![image-20221211184930256](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221211184930256.png)

最后一行（2018-11-01）跟踪2018年11月每个博物馆的访客数量，倒数第二行（2018-10-01）跟踪2017年10月每个博物馆访客数量，依此类推。 使用最后5行数据回答以下问题。

```python
# Fill in the line below: How many visitors did the Chinese American Museum 
# receive in July 2018?
ca_museum_jul18 = 2620

# Fill in the line below: In October 2018, how many more visitors did Avila 
# Adobe receive than the Firehouse Museum?
avila_oct18 = 19280-4622
```

### 第三步：说服博物馆董事会 

消防博物馆声称，他们在2014年举办了一场活动，吸引了大量游客，他们应该获得额外的预算，再次举办类似的活动。其他博物馆认为这些类型的活动没有那么重要，预算应该纯粹根据平均每天最近的游客来分配。 

为了向博物馆董事会展示该活动与各博物馆的常规交通相比的情况，请创建一个折线图，显示各博物馆的游客数量随时间的变化。你的数字应该有四行（每个博物馆一行）。

> （可选）注意：如果您以前有使用Python绘制图形的经验，您可能会熟悉`plt.show()`命令。如果您决定使用此命令，请将其放在检查答案的代码行之后（在这种情况下，请将它放在下面的`step_3.check()`之后）——否则，检查代码将返回错误！

```python
# Line chart showing the number of visitors to each museum over time
# Your code here
# Set the width and height of the figure
plt.figure(figsize=(14,6))

# Add title
plt.title("The number of visitors to each 4 museum changes over time")

# Line chart showing monthly the number of visitors of museum 'Avila Adobe'
sns.lineplot(data=museum_data['Avila Adobe'],label='Avila Adobe')

# Line chart showing monthly the number of visitors of museum 'Firehouse Museum'
sns.lineplot(data=museum_data['Firehouse Museum'],label='Firehouse Museum')

# Line chart showing monthly the number of visitors of museum 'Chinese American Museum'
sns.lineplot(data=museum_data['Chinese American Museum'],label='Chinese American Museum')

# Line chart showing monthly the number of visitors of museum 'America Tropical Interpretive Center'
sns.lineplot(data=museum_data['America Tropical Interpretive Center'],label='America Tropical Interpretive Center')

# Add label for horizontal axis
plt.xlabel("Month")

# Check your answer
step_3.check()
```

![image-20221211190458463](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221211190458463.png)

### 步骤4：评估季节性 

在与Avila Adobe的员工会面时，你会听到一个主要的痛点是，博物馆游客的数量随季节变化很大，有淡季（员工人手充足、心情愉快），也有旺季（员工人手不足、压力大）。你意识到，如果你能预测这些旺季和淡季，你可以提前计划雇佣一些额外的季节性员工来帮助完成额外的工作。 

#### A部分 

创建一个折线图，显示Avila Adobe的访问者数量随时间的变化。（如果代码返回错误，首先要检查的是列的名称拼写是否正确！列的名称必须与数据集中显示的名称完全相同。）

```python
# Line plot showing the number of visitors to Avila Adobe over time
# Your code here
# Set the width and height of the figure
plt.figure(figsize=(14,6))

# Add title
plt.title("The number of visitors to museum 'Avila Adobe' changes over time")

# Line chart showing monthly the number of visitors of museum 'Avila Adobe'
sns.lineplot(data=museum_data['Avila Adobe'],label='Avila Adobe')

# Add label for horizontal axis
plt.xlabel("Month")

# Check your answer
step_4.a.check()
```

![image-20221211190538330](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221211190538330.png)

#### B部分 

Avila Adobe是否获得更多访客： 

- 9月至2月（洛杉矶，秋季和冬季），或 

- 三月到八月（在洛杉矶，春天和夏天）？ 

利用这些信息，博物馆什么时候应该增加季节性员工？

解决方案：折线图通常在每年的早期（12月和1月）下降到相对较低的值，在年中（尤其是5月和6月）达到最高值。因此，Avila Adobe通常会在3月至8月（或春季和夏季）吸引更多游客。考虑到这一点，Avila Adobe肯定会从雇佣更多的季节性员工来帮助在3月至8月（春季和夏季）的额外工作中受益！

### 继续前进 

继续使用新数据集了解条形图和热图！
