# `DataTable`库

[`Datatable`：Python数据分析提速高手，飞一般的感觉！](https://www.sohu.com/a/379877205_505915)

## 介绍

`Datatable`是一个Python库：

![image-20221031173452619](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031173452619.png)

https://datatable.readthedocs.io/en/latest/?badge=latest

![image-20221031173508821](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031173508821.png)

![image-20221031173523702](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031173523702.png)

**`Datatable`的优点：**

- 高效的多线程算法
- Memory-thrifty
- 内存映射磁盘上的数据集
- 本地C++实现
- 完全开源

## `Datatable`主要语法

在`Datatable`中，所有这些操作的主要工具是方括号表示法，其灵感来自传统的矩阵索引。

![image-20221031173801169](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031173801169.png)

`i`是行选择器，`j`是列选择器。`...`表示附加修饰符。当前可用的修饰符是`by`、`join`和`sort`。这个工具包与pandas非常相似，但更**侧重于速度和大数据支持。**

接下来，我们将使用`Datatable`的`fread`函数读取获取和性能文件。下面的`fread`函数既强大又非常快。它可以自动检测和解析大多数文本文件的参数，从.zip档案或url加载数据，读取Excel文件等等。

### 下载datatable

```python
$ pip install datatable
```

载入datatable库

```python
import datatable as dt
print(dt.__version__)
```

查看datatable库里的函数

```
dir(datatable)
```

```python
输出：
['FExpr',
 'Frame',
 'Namespace',
 'Type',
 '__all__',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 '__version__',
 '_build_info',
 'abs',
 'as_type',
 'bool8',
 'build_info',
 'by',
 'cbind',
 'corr',
 'count',
 'cov',
 'cut',
 'dt',
 'exceptions',
 'exp',
 'expr',
 'f',
 'first',
 'float32',
 'float64',
 'frame',
 'fread',
 'g',
 'ifelse',
 'init_styles',
 'int16',
 'int32',
 'int64',
 'int8',
 'internal',
 'intersect',
 'iread',
 'isna',
 'join',
 'last',
 'lib',
 'log',
 'log10',
 'ltype',
 'math',
 'max',
 'mean',
 'median',
 'min',
 'obj64',
 'options',
 'qcut',
 'rbind',
 're',
 'repeat',
 'rowall',
 'rowany',
 'rowcount',
 'rowfirst',
 'rowlast',
 'rowmax',
 'rowmean',
 'rowmin',
 'rowsd',
 'rowsum',
 'sd',
 'setdiff',
 'shift',
 'sort',
 'split_into_nhot',
 'str',
 'str32',
 'str64',
 'stype',
 'sum',
 'symdiff',
 'time',
 'types',
 'union',
 'unique',
 'update',
 'utils',
 'xls']
```

```python
from datatable import (dt, f, by, ifelse, update, sort, count, min, max, mean, sum, rowsum,isna)
```



### 加载数据文件

可加载普通python list，pandas，numpy array，csv/text/excel文件，二进制`.jay`文件 

```python
DT1 = dt.Frame(A=range(5), B=[1.7, 3.4, 0, None, -math.inf],
               stypes={"A": dt.int64})
DT2 = dt.Frame(pandas_dataframe)
DT3 = dt.Frame(numpy_array)
DT4 = dt.fread("~/Downloads/dataset_01.csv")
DT5 = dt.open("data.jay")
```

```python
cols = ['c0','c1']
df_1= dt.fread('../input/xxx.dt',columns=cols)
```

```
df = dt.Frame({"A": [1, None, 6, 4],
               "B": [2, 4, 5, 6],
               "C": [3, 5, 4, 7],
               "D": [4, -3, 3, 8],
               "E": [5, 1, 2, 9]})
df
```



### 数据操作

将数据加载到 Frame 后，您可能希望对其执行某些操作：提取/删除/修改数据子集、执行计算、重塑、分组、与其他数据集连接等。在 datatable 中，主要工具所有这些操作都是方括号符号，它受传统矩阵索引的启发，但功能过于强大（这种符号是在 R data.table 中首创的，是这两个库之间的主要交叉点）。

简而言之，几乎所有带有 Frame 的操作都可以表示为

```python
DT[i, j, ...]
#dt[:,:]
#dt[0:1,2:3]
```

其中 `i `是行选择器，`j` 是列选择器，`... `表示可能会添加其他修饰符。如果这对您来说很熟悉，那是因为它是。在索引矩阵、C/C++、R、pandas、`numpy` 等时，在数学中使用完全相同的 DT[i, j] 表示法。`datatable` 引入的唯一区别是它允许` i` 是任何东西，它可以想象或被解释为一个行选择器：一个只选择一行的整数、一个切片、一个范围、一个整数列表、一个切片列表、一个表达式、一个布尔值的帧、一个整数值的帧、一个整数 `numpy`数组、生成器等。

`j` 列选择器更加通用。在最简单的情况下，您可以通过索引或名称仅选择单个列。但也接受的是列列表、切片、字符串切片（形式为`“A”：“Z”`）、指示选择哪些列的布尔值列表、表达式、表达式列表和表达式。 （键将用作被选择列的新名称。） `j` 表达式甚至可以是 python 类型（例如` int` 或 `dt.float32`），选择与该类型匹配的所有列。

除了上面显示的选择器表达式，我们还支持更新和删除语句：

```python
DT[i, j] = r
del DT[i, j]
```

第一个表达式：将用来自 r 的值替换帧 DT 的子集 [i, j] 中的值，r 可以是常数，也可以是适当大小的帧，或者是对帧 DT 进行操作的表达式。 

第二个表达式：删除子集 [i, j] 中的值。这解释如下：如果 i 选择所有行，则 j 给出的列从 Frame 中删除；如果 j 选择所有列，则删除 i 给出的行；如果 i 和 j 都不跨越 Frame 的所有行/列，则子集 [i, j] 中的元素将替换为 NA。

### 基本框架属性

```python
DT.shape   # (nrows, ncols)
DT.names   # column names
DT.types   # column types
```





### 数据转换

```python
df_2 = df[:,dt.as_type(f['A','B'],dt.Type.float32)]
#A列与B列变为float32
```

### 数据列添加

```python
df[:,dt.extend(f['F'])]
```

### 数据计算

```python
y = df[:,f.B *2-1]
y
```

#### 计算平均值

```python
X[:, dt.mean(f[:])]
```

![image-20221031204339545](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031204339545.png)

#### 计算最大值或最小值

```python
#max
X[:, dt.max(f[:])]
#min
X[:, dt.min(f[:])]
```



### 删除重复项

```python
dt.unique(df[:, "LoanID"]).head(5)
```



### 缺失值

```py
df = dt.Frame({"A": [1, None, 6, 4],
               "B": [2, 4, 5, 6],
               "C": [3, 5, 4, 7],
               "D": [4, -3, 3, 8],
               "E": [5, 1, 2, 9]})
df
```

![image-20221031191913493](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031191913493.png)



```
df[:, dt.count(f[:])]
```

![image-20221031191949164](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031191949164.png)

```
df[:, dt.isna(f[:])]
```

![image-20221031192015066](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031192015066.png)

### f代表什么

在 datatable 中，f 代表 frame_proxy，它提供一种简单的方式来引用**当前正在操作的帧**。在上面的例子中，dt.f 只代表 dt_df。

[媲美Pandas？告诉你Python的Datatable包到底怎么用！](https://zhuanlan.zhihu.com/p/70831136)

### 拆分数据集

[如何在python中将datatable dataframe拆分为训练和测试数据集](https://www.cnpython.com/qa/1294725)

转换为numpy并在拆分后返回到datatable dataframe：

```python
dt_df = dt.fread(csv_file_path)
# source code before split method

dt_df = dt_df.to_numpy()
#dt_df = dt_df.to_pandas()

X_train, X_test, y_train, y_test = train_test_split(dt_df, classe, test_size=test_size)

X_train = dt.Frame(X_train)

# source code after split method
```

![image-20221031203825578](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031203825578.png)

datatable 文件与各种类型数据的转换

![image-20221101104947754](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221101104947754.png)

```python
.to_arrow() #frame转为arrow table
.to_csv(file) #frame转为csv
.to_dict() #通过列把frame转为dict
.to_jay(file) #把frame数据存储进二进制jay格式
.to_list() #通过列返回作为一系列列表的frame数据
.to_numpy() #frame转为numpy array
.to_pandas() #frame转为pandas dataframe
.to_tuples() #通过行返回作为一列元数据（tuples）的frame数据

```

[关于python 读取csv最快的Datatable的用法,你都学会了吗](https://www.jb51.net/article/225671.htm)

[从dataTable中删除包含空值的行](https://qa.1r1g.com/sf/ask/1222526791/)

```python
public static void RemoveNullColumnFromDataTable(DataTable dt)
{
    for (int i = dt.Rows.Count - 1; i >= 0; i--)
    {
        if (dt.Rows[i][1] == DBNull.Value)
            dt.Rows[i].Delete();
    }
    dt.AcceptChanges();
}
```

[Datatable删除行的Delete和Remove方法](https://www.cnblogs.com/jhxk/articles/2328744.html)

[在python中将datatable框架中的字符串列转换为日期格式](https://cloud.tencent.com/developer/ask/sof/1460472)

***

## 补充：`DataTable`详解

[深入详解DataTable](https://www.pianshen.com/article/8377588335/)

### ADO.NET

在学习`DataTable`知识之前，我们有必要了解下ADO.NET。以下摘自MSDN：

> ADO.NET 对 Microsoft SQL Server 和 XML 等数据源以及通过 OLE DB 和 XML 公开的数据源提供一致的访问。数据共享使用者应用程序可以使用 ADO.NET 来连接到这些数据源，并检索、处理和更新所包含的数据。 

> ADO.NET 通过数据处理将数据访问分解为多个可以单独使用或一前一后使用的不连续组件。ADO.NET 包含用于连接到数据库、执行命令和检索结果的 .NET Framework 数据提供程序。您可以直接处理检索到的结果，或将其放入 ADO.NET `DataSet` 对象，以便与来自多个源的数据或在层之间进行远程处理的数据组合在一起，以特殊方式向用户公开。

> ADO.NET `DataSet` 对象也可以独立于 .NET Framework 数据提供程序使用，以管理应用程序本地的数据或源自 XML 的数据。 

> ADO.NET 类在 `System.Data.dll` 中，并且与 `System.Xml.dll` 中的 XML 类集成。当编译使用 `System.Data` 命名空间的代码时，请引用 `System.Data.dll` 和 `System.Xml.dll`。有关连接到数据库、从数据库中检索数据并在命令提示中显示该数据的 ADO.NET 应用程序示例，请参见 ADO.NET 示例应用程序。 

> ADO.NET 向编写托管代码的开发人员提供了类似于 ActiveX 数据对象 (ADO) 为本机组件对象模块 (COM) 开发人员提供的功能。

ADO.NET中包含的对象及其关系如下图：

![image-20221031171902757](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031171902757.png)

![image-20221031171943092](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031171943092.png)

## 1. `DataTable`简介

### 1.1 `DataTable`的定义

表示内存中数据的一个表。 我们知道数据库中存储的是实体表，实体表中有一系列的数据。而`DataTable`即存储在内存中的表，在持久化到数据库之前，是不会对数据库产生影响的，持久化到数据库可以使用`dataAdapter.Update`的方法（`dataAdapter`是某个实例化的`DataAdapter`对象）。

注意：当访问 `DataTable` 对象时，请注意它们是按条件区分大小写的。例如，如果一个 `DataTable` 被命名为“`mydatatable`”，另一个被命名为“`Mydatatable`”，则用于搜索其中一个表的字符串被认为是区分大小写的。但是，如果“`mydatatable`”存在而“`Mydatatable`”不存在，则认为该搜索字符串不区分大小写。

### 