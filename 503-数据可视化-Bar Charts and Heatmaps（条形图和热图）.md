# 503-数据可视化-Bar Charts and Heatmaps（条形图和热图）

使用颜色或长度比较数据集中的类别

现在您可以创建自己的折线图了，是时候了解更多图表类型了！

> 顺便说一下，如果这是您第一次使用Python编写代码，那么您应该为迄今为止所取得的所有成就感到自豪，因为学习一种全新的技能从来都不容易！如果你坚持这门课程，你会发现一切只会变得更容易（而你将建立的图表将变得更令人印象深刻！），因为所有图表的代码都非常相似。与任何技能一样，随着时间的推移，随着重复，编码变得自然。

在本教程中，您将学习条形图和热图。

### 设置笔记本 

与往常一样，我们从设置编码环境开始。（此代码是隐藏的，但您可以通过单击此文本右下方的“代码”按钮来取消隐藏。）

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

在本教程中，我们将使用美国交通部的数据集来跟踪航班延误。 

在Excel中打开此CSV文件会显示每个"月"的一行（其中`1`=1月，`2`=2月等）和每个航空公司代码的一列。

![image-20221211203125205](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221211203125205.png)

每个条目显示不同航空公司和月份的平均到达延迟（以分钟为单位）（均在2015年）。**负条目**表示（平均而言）倾向于**提前到达**的航班。例如，美国航空公司1月份的航班（航空公司代码：**AA**）平均晚点约7分钟，阿拉斯加航空公司（航空公司代码：**AS**）4月份的航班平均晚点约3分钟。

## 加载数据 

与之前一样，我们使用 `pd.read_csv` 命令加载数据集。

```python
# Path of the file to read
flight_filepath = "../input/flight_delays.csv"

# Read the file into a variable flight_data
flight_data = pd.read_csv(flight_filepath, index_col="Month")
```

您可能会注意到，代码比我们在上一教程中使用的代码略短。在这种情况下，由于**行标签（来自“月”列）与日期不对应**，因此我们不会在括号中添加`parse_dates=True`。但是，我们保留了前两段文本，以便同时提供： 

- 数据集的文件路径（在本例中为飞行文件路径），以及 

- 将用于索引行的列的名称（在本例中，index_col=“Month”）。

## 检查数据 

由于数据集很小，我们可以轻松打印其所有内容。这是通过只使用数据集的名称编写一行代码来完成的。

```python
# Print the data
flight_data
```

![image-20221211203202698](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221211203202698.png)

## 条形图(bar chart) 

假设我们想创建一个条形图，显示Spirit Airlines（航空公司代码：NK）航班的平均到达延迟，按月份划分。

```python
# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data['NK'])

# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")
```

![image-20221211203324541](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221211203324541.png)

用于自定义文字（标题和垂直轴标签）和图片尺寸的命令在上一教程中很熟悉。创建条形图的代码是新的：

```python
# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data['NK'])
```

它有三个主要组成部分： 

- `sns.barplot`-这告诉笔记本我们要创建条形图。 请记住，**`sns`**指的是*[seaborn](https://seaborn.pydata.org/)* 包，本课程中用于创建图表的所有命令都将以该前缀开头。 

- `x=flight_data.index`-这决定了在水平轴上使用什么。在本例中，我们选择了**对行进行索引的列**（在本例，是包含月份的列）。 

- `y=flight_data['NK']`-这将设置数据中用于**确定每个条形图高度的列**。在本例中，我们选择“NK”列。

> 重要提示：必须选择带有`flight_data.index`的索引列，并且不可能使用`flight_data['Month']`（这将返回错误）。这是因为当我们加载数据集时，“月”列用于索引行。我们总是必须使用这种特殊的符号来选择索引列。

## 热图(heatmap) 

我们还有一种剧情类型需要学习：热图！ 

在下面的代码单元中，我们创建了一个热图来快速可视化 `flight_data` 中的模式。每个单元格根据其对应值进行颜色编码。

```python
# Set the width and height of the figure
plt.figure(figsize=(14,7))

# Add title
plt.title("Average Arrival Delay for Each Airline, by Month")

# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=flight_data, annot=True)

# Add label for horizontal axis
plt.xlabel("Airline")
```

![image-20221211203653639](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221211203653639.png)

创建热图的相关代码如下：

```python
# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=flight_data, annot=True)
```

该代码有三个主要部分： 

- `sns.heatmap`-这告诉笔记本我们要创建热图。 

- `data=flight_data`-这告诉笔记本使用flight_data中的所有条目来创建热图。 

- `annot=True`-这确保**每个单元格的值显示在图表上**。（省略此项将删除每个单元格中的数字！） 

您可以在表格中检测到哪些模式？例如，如果你仔细观察，对所有航空公司来说，临近年底的几个月（尤其是9-11个月）显得相对黑暗。这表明，航空公司在这几个月里（平均而言）更好地遵守时间表！

## 接下来是什么？ 

通过编码练习创建自己的可视化效果！

## Exercise: Bar Charts and Heatmaps

在本练习中，您将使用您的新知识为现实场景提出解决方案。要成功，您需要将数据导入Python，使用数据回答问题，并生成条形图和热图以了解数据中的模式。

### 脚本(Scenario) 

您最近决定创建自己的视频游戏！作为IGN游戏评论([IGN Game Reviews](https://www.ign.com/reviews/games))的狂热读者，你会听到所有最新的游戏发布，以及他们从专家那里获得的排名，从0（灾难）到10（杰作）。

![image-20221211205556965](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221211205556965.png)

您有兴趣使用IGN评论([IGN reviews](https://www.ign.com/reviews/games))来指导您即将推出的游戏的设计。谢天谢地，有人在一个非常有用的CSV文件中总结了排名，您可以使用它来指导分析。

### 安装程序 

运行下一个单元以导入和配置完成练习所需的Python库。

```python
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
```

以下问题将为您的工作提供反馈。运行以下单元格以设置反馈系统。

```python
# Set up code checking
import os
if not os.path.exists("../input/ign_scores.csv"):
    os.symlink("../input/data-for-datavis/ign_scores.csv", "../input/ign_scores.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex3 import *
print("Setup Complete")
```

### 步骤1：加载数据 

将IGN数据文件读入IGN_data。使用“平台”列标记行。

```python
# Path of the file to read
ign_filepath = "../input/ign_scores.csv"

# Fill in the line below to read the file into a variable ign_data
ign_data = pd.read_csv(ign_filepath, index_col="Platform")

# Run the line below with no changes to check that you've loaded the data correctly
step_1.check()
```

### 步骤2：查看数据 

使用Python命令打印整个数据集。

```python
# Print the data
# Your code here
ign_data
```

![image-20221211210526686](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221211210526686.png)

![image-20221211210554497](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221211210554497.png)

您刚刚打印的数据集显示了平台和类型的平均得分。使用数据回答以下问题。

```python
# Fill in the line below: What is the highest average score received by PC games,
# for any genre?
high_score = 7.759930

# Fill in the line below: On the Playstation Vita platform, which genre has the 
# lowest average score? Please provide the name of the column, and put your answer 
# in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)
worst_genre = 'Simulation'
```

### 第三步：哪个平台最好？ 

你还记得，你最喜欢的电子游戏是2008年为Wii平台发布的赛车游戏《马里奥卡丁车Wii( [**Mario Kart Wii**](https://www.ign.com/games/mario-kart-wii))》。IGN同意你的观点，这是一款很棒的游戏——他们对这款游戏的评分高达8.9！受这款游戏的成功启发，您正在考虑为Wii平台创建自己的赛车游戏。 

#### A部分

创建一个条形图，显示每个平台的赛车游戏平均得分。图表中每个平台应有一个条形图。

```python
# Bar chart showing average score for racing games by platform
# Your code here
# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("Average Score for Racing Games, by Platform")

# Bar chart showing the average score for racing games, for each platform
sns.barplot(x=ign_data['Racing'], y=ign_data.index)

# Add label for horizontal axis
plt.xlabel("")

# Check your answer
step_3.a.check()
```

![image-20221211211717391](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221211211717391.png)

#### B部分

根据条形图，您是否期望Wii平台的赛车游戏获得高评级？如果不是，什么游戏平台似乎是最好的选择？

解决方案：根据数据，我们不应该期望Wii平台的赛车游戏获得高评级。事实上，Wii赛车游戏的平均得分低于任何其他平台。Xbox One似乎是最好的选择，因为它的平均收视率最高。

### 步骤4：所有可能的组合！ 

最终，你决定不为Wii创建赛车游戏，但你仍然致力于创建自己的视频游戏！由于您的游戏兴趣非常广泛（……您通常喜欢大多数电子游戏），因此您决定使用IGN数据来通知您对类型和平台的新选择。 

#### A部分

使用数据创建按类型和平台的平均得分热图。

```python
# Heatmap showing average game score by platform and genre
# Your code here
# Set the width and height of the figure
plt.figure(figsize=(14,7))

# Add title
plt.title("Average Score by Genre and Platform")

# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=ign_data, annot=True)

# Add label for horizontal axis
plt.xlabel("Genre")

# Check your answer
step_4.a.check()
```

![image-20221211212208322](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221211212208322.png)

#### B部分 

哪种类型和平台组合的平均收视率最高？哪个组合的平均排名最低？

```python
# Check your answer (Run this code cell to receive credit!)
step_4.b.solution()
```

解决方案：Playstation 4的模拟游戏获得最高的平均评分（9.2）。Game Boy Color的射击和格斗游戏平均排名最低（4.5）。

### 继续前进 

继续学习散点图！