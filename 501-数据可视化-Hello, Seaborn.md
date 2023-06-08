# 501-数据可视化-Hello, Seaborn

## 欢迎使用数据可视化！ 

在本实践课程中，您将学习如何使用[seaborn](https://seaborn.pydata.org/index.html)（一种功能强大但易于使用的数据可视化工具）将数据可视化提升到一个新的水平。要使用seaborn，您还将学习如何使用流行编程语言Python编写代码。也就是说， 

- 该课程针对的是那些没有编程经验的人，以及 

- 每个图表都使用简短的代码，使seaborn比许多其他数据可视化工具（例如Excel）更快、更容易使用。 

因此，如果你从未写过一行代码，并且你想学习最起码的东西，以便今天开始制作更快、更吸引人的情节，那么你就在正确的位置！要查看您将制作的一些图表，请查看下面的数字。

![image-20221210192326656](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210192326656.png)

## 您的编码环境 

现在花点时间快速上下滚动此页面。你会注意到有很多不同类型的信息，包括： 

1. text-文本（就像你现在正在阅读的文本！）， 

2. code-代码（始终包含在称为代码单元的灰色框内），以及 

3. code output-代码输出（或运行代码的打印结果，始终显示在相应代码的正下方）。 

我们将这些页面称为Jupyter笔记本（或者，通常只是笔记本），我们将在整个迷你课程中使用它们。笔记本的另一个示例可以在下图中找到。

![image-20221210210739049](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210210739049.png)

在您正在阅读的笔记本中，我们已经为您运行了所有代码。很快，你就可以使用一个笔记本来编写和运行自己的代码了！

## 设置笔记本

您需要在每个笔记本的顶部运行几行代码来设置您的编码环境。现在理解这些代码行并不重要，所以我们现在还不深入讨论细节。（请注意，它作为输出返回：Setup Complete。）

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

- `pd.plotting.register_matplotlib_converters()`的解释：

  它可以确保`pandas`像这样的数据类型`pd.Timestamp`可以在`matplotlib`绘图中使用，而不必将其转换为其他类型。

  https://www.nuomiphp.com/eplan/223290.html

- Matplotlib中`%matplotlib inline`如何使用

  `%matplotlib inline`

  是一个魔法函数（Magic Functions）。官方给出的定义是：IPython有一组预先定义好的所谓的魔法函数（Magic Functions），你可以通过**命令行的语法形式**来访问它们。可见“%[matplotlib](https://so.csdn.net/so/search?q=matplotlib&spm=1001.2101.3001.7020) inline”就是模仿命令行来访问magic函数的在IPython中独有的形式。

  magic函数分两种：一种是面向行的，另一种是面向单元型的。

  行magic函数是用前缀“%”标注的，很像我们在系统中使用命令行时的形式，例如在Mac中就是你的用户名后面跟着“$”。“%”后面就是magic函数的参数了，但是它的参数是没有被写在括号或者引号中来传值的。

  单元型magic函数是由两个“%%”做前缀的，它的参数不仅是当前“%%”行后面的内容，也包括了在当前行以下的行。

  **注意**：既然是IPython的内置magic函数，那么在Pycharm中是不会支持的。

  **总结**：%matplotlib inline 可以在Ipython编译器里直接使用，功能是可以内嵌绘图，并且**可以省略掉`plt.show()`**这一步。————————————————
  版权声明：本文为CSDN博主「liming89」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
  原文链接：https://blog.csdn.net/liming89/article/details/109662966

- **Seaborn**（seaborn是python中的一个可视化库，是对matplotlib进行二次封装而成，既然是基于matplotlib，所以seaborn的很多图表接口和参数设置与其很是接近）

  [Seaborn的简述](https://blog.csdn.net/qq_52669357/article/details/122821249)

## 加载数据 

在本笔记本中，我们将使用六个国家的FIFA历史排名数据集：阿根廷（ARG）、巴西（BRA）、西班牙（ESP）、法国（FRA）、德国（GER）和意大利（ITA）。数据集存储为CSV文件（逗号分隔值文件的缩写【 [comma-separated values file](https://bit.ly/2Iu5D4x)】）。在Excel中打开CSV文件会显示每个日期的一行，以及每个国家的一列。

![image-20221210211545679](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210211545679.png)

要将数据加载到笔记本中，我们将使用两个不同的步骤，在下面的代码单元中实现如下： 

- 首先指定可以访问数据集的位置（或文件路径），然后 
- 使用文件路径将数据集的内容加载到笔记本中。

```python
# Path of the file to read
fifa_filepath = "../input/fifa.csv"

# Read the file into a variable fifa_data
fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)
```

![image-20221210211646378](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210211646378.png)

请注意，上面的代码单元有四条不同的线。

## 注释 (comments)

其中两行前面有一个磅符号（#），其中包含的文本显示为褪色和斜体。 

当代码运行时，计算机完全忽略了这两行，它们只出现在这里，以便任何阅读代码的人都能快速理解它。我们将这两行称为注释，最好将它们包括在内，以确保代码易于解释。

## 可执行代码 (Executable code)

另外两行是可执行代码，或由计算机运行的代码（在本例中，用于查找和加载数据集）。 

第一行将 `fifa_filepath` 的值设置为可以访问数据集的位置。在本例中，我们为您提供了文件路径（用引号）。请注意，这行可执行代码上方的注释提供了它的快速描述！ 

第二行设置`fifa_data`的值以包含数据集中的所有信息。这是通过 `pd.read_csv` 完成的。它后面紧跟着三段不同的文本（上图中带下划线），它们用括号括起来，并用逗号分隔。这些用于自定义数据集加载到笔记本中时的行为：

- `fifa_filepath`-始终需要首先提供数据集的**文件路径**。 
- `index_col="Date"`-加载数据集时，我们希望第一列中的**每个条目表示不同的行**。为此，我们将index_col的值设置为第一列的名称（`"Date"`，在Excel中打开时在文件的单元格A1中找到）。 
- `parse_dates=True`-这告诉笔记本将**每行标签理解为日期**（而不是数字或其他具有不同含义的文本）。

当您有机会在动手练习中加载自己的数据集时，这些细节将很快变得更有意义。

> 现在，重要的是要记住，运行这两行代码的最终结果是，我们现在可以使用`fifa_data`从笔记本访问数据集。

顺便说一句，您可能已经注意到这些代码行没有任何输出（而您之前在笔记本中运行的代码行返回设置完成作为输出）。这是预期的行为——并不是所有的代码都会返回输出，这段代码就是一个很好的例子！

## 检查数据 

现在，我们将快速查看`fifa_data`中的数据集，以确保其正确加载。 

我们通过编写一行代码打印数据集的前五行，如下所示： 

- 从包含数据集的变量开始（在本例中为`fifa_data`），然后 

- 用.head（）跟随它。 

您可以在下面的代码行中看到这一点。

```python
# Print the first 5 rows of the data
fifa_data.head()
```

![image-20221210214305249](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210214305249.png)

现在检查前五行是否与上面的数据集图像一致（当我们在Excel中看到它的样子时）。

## 绘制数据 

在本课程中，您将了解许多不同的情节类型。在许多情况下，您只需要一行代码就可以制作图表！ 

要了解您将要学习的内容，请查看下面生成折线图的代码。

```python
# Set the width and height of the figure
plt.figure(figsize=(16,6))

# Line chart showing how FIFA rankings evolved over time 
sns.lineplot(data=fifa_data)
```

![image-20221210214410086](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210214410086.png)

这段代码应该还没有意义，您将在接下来的教程中了解更多。现在，继续您的第一个练习，在那里您将有机会亲身体验编码环境！

## 接下来是什么？

在第一次编码练习中写出第一行代码！

## Exercise: Hello, Seaborn

在本练习中，您将编写第一行代码，并学习如何使用本课程的编码环境！

### 安装程序

首先，您将学习如何运行代码，我们将从下面的代码单元开始。（请记住，笔记本中的代码单元只是一个灰色框，其中包含我们要运行的代码。）

- 首先在代码单元格内单击。 
- 单击代码单元格左侧出现的蓝色三角形（呈“播放按钮”形状）。 
- 如果代码成功运行，您将在单元格下方看到 `Setup Complete` 作为输出。

![image-20221210222118195](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210222118195.png)

下面的代码单元导入并配置完成练习所需的Python库。 单击单元格并运行它。

```python
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Set up code checking
import os
if not os.path.exists("../input/fifa.csv"):
    os.symlink("../input/data-for-datavis/fifa.csv", "../input/fifa.csv")  
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex1 import *
print("Setup Complete")
```

您刚刚运行的代码设置了系统，以便为您的工作提供反馈。您将在下一步中了解有关反馈系统的更多信息。

### 步骤1：探索反馈系统

每个练习都让您使用真实世界的数据集测试新技能。在这一过程中，您将收到有关工作的反馈。你会看到你的答案是否正确，得到定制的提示，并看到官方的解决方案（如果你想看一看！）。 

为了探索反馈系统，我们将从一个编码问题的简单示例开始。按顺序执行以下步骤：

1. 运行下面的代码单元而不进行任何编辑。它将显示以下输出：

   > Check: When you've updated the starter code, `check()` will tell you whether your code is correct. You need to update the code that creates variable `one`
   >
   > 检查：当您更新了起始代码后，Check（）将告诉您代码是否正确。您需要更新创建变量1的代码

   这意味着您需要更改代码以将变量1设置为以下空白以外的其他内容（____）。

2. 将下划线替换为2，使代码行显示为1=2。然后，运行代码单元。这将返回以下输出：

   > Incorrect: Incorrect value for `one`: `2`
   >
   > 不正确：1的值不正确：2

3. 现在，将2更改为1，使代码行显示为one=1。然后，运行代码单元。答案应标记为正确。您现在已完成此问题！

```python
# Fill in the line below
one = 1

# Check your answer
step_1.check()
```

在本练习中，您负责填写设置变量`one`值的代码行。不要编辑检查答案的代码。您需要像提供的那样运行`step_1.check()`和`step_2.check()`等代码行。 

这个问题相对简单，但对于更困难的问题，您可能希望收到提示或查看官方解决方案。现在运行下面的代码单元以接收这两个问题。

```python
step_1.hint()
step_1.solution()
```

![image-20221210224104471](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210224104471.png)

### 步骤2：加载数据 

您已经准备好开始一些数据可视化了！您将从加载上一教程中的数据集开始。 

您需要的代码已在下面的单元格中提供。运行那个单元。如果显示正确的结果，您可以继续前进了！

```python
# Path of the file to read
fifa_filepath = "../input/fifa.csv"

# Read the file into a variable fifa_data
fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)

# Check your answer
step_2.check()
```

接下来，回想一下注释和可执行代码之间的区别： 

- 注释前面有一个磅号（#），并包含褪色和斜体的文本。当代码运行时，它们被计算机完全忽略。 

- 可执行代码是由计算机运行的代码。 

在下面的代码单元中，每一行都是注释：

```python
# Uncomment the line below to receive a hint
#step_2.hint()
#step_2.solution()
```

如果运行下面的代码单元而不做任何更改，它将不会返回任何输出。现在就试试吧！

```python
# Uncomment the line below to receive a hint
#step_2.hint()
# Uncomment the line below to see the solution
#step_2.solution()
```

接下来，删除step_2.int（）之前的磅符号，以便上面的代码单元显示如下：

```python
# Uncomment the line below to receive a hint
step_2.hint()
#step_2.solution()
```

```
Hint: Use pd.read_csv, and follow it with three pieces of text that are enclosed in parentheses and separated by commas. (1) The filepath for the dataset is provided in fifa_filepath. (2) Use the "Date" column to label the rows. (3) Make sure that the row labels are recognized as dates.

提示：使用pd.read_csv，并在后面加上三段用括号括起来并用逗号分隔的文本。
（1） 数据集的文件路径在fifa_filepath中提供。
（2） 使用“日期”列标记行。
（3） 确保行标签被识别为日期。
```

当我们删除代码行之前的磅符号时，我们会取消注释该行。这将注释转换为计算机运行的一行可执行代码。现在运行代码单元，它应该返回Hint作为输出。 最后，取消注释该行以查看解决方案，因此代码单元显示如下：

```python
# Uncomment the line below to receive a hint
step_2.hint()
step_2.solution()
```

```python
#Hint: Use pd.read_csv, and follow it with three pieces of text that are enclosed in parentheses and separated by commas. (1) The filepath for the dataset is provided in fifa_filepath. (2) Use the "Date" column to label the rows. (3) Make sure that the row labels are recognized as dates.

#Solution:

fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)
```

然后，运行代码单元。您应该收到提示和解决方案。 

如果您在任何时候都无法找到问题的正确答案，欢迎在完成单元格之前获得提示或解决方案。（因此，在运行给您提示或解决方案的代码之前，您不需要得到正确的结果。）

### 步骤3：绘制数据 

现在数据已加载到笔记本中，您可以将其可视化了！ 

运行下一个代码单元格而不进行更改，以生成折线图。这段代码可能还没有意义——您将在下一篇教程中了解它的全部内容！

```python
# Set the width and height of the figure
plt.figure(figsize=(16,6))

# Line chart showing how FIFA rankings evolved over time
sns.lineplot(data=fifa_data)
```

![image-20221210224458265](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221210224458265.png)

有些问题不需要你写任何代码。相反，您将解释可视化。 

例如，考虑一个问题：仅考虑数据集中所代表的年份，哪些国家至少连续5年位居第一？ 

要接收提示，请取消注释下面的行，然后运行代码单元。

```python
step_3.b.hint()
```

一旦有了答案，请检查解决方案，以获得完成问题的学分，并确保您的解释正确。

```python
# Check your answer (Run this code cell to receive credit!)
step_3.b.solution()
```

解决方案：唯一符合这一标准的国家是巴西（代码：BRA），因为它在1996-2000年保持着最高排名。其他国家确实花了一些时间位居第一，但巴西是唯一一个至少连续(**consecutive**)五年保持这一排名的国家。

恭喜你——你已经完成了第一次编码练习！

### 继续前进 

继续学习使用新数据集创建自己的折线图(line charts)。