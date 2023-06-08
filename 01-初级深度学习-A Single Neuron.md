# A Single Neuron

一个单一神经元

## Linear Units in Keras

最简单创建一个神经元模型，就是通过`keras.Sequential`.

创建一个线性的神经元的模型，输出是卡路里，输入是糖、纤维和蛋白质。

```python
from tensorflow import keras
from tensorflow.keras import layers

# Create a network with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
```

`layers.Dense`中`units`是输出的神经元的数量，`input_shape`是输入神经元的形状，模型将接受三个特征作为输入



#### 为什么 input_shape 是 Python 列表？ 

我们将在本课程中使用的数据将是表格数据，就像在 Pandas 数据框中一样。我们将为数据集中的每个特征提供一个输入。特征按列排列，所以我们总是有 `input_shape=[num_columns]`。 Keras 在这里使用列表的原因是允许使用更复杂的数据集。例如，图像数据可能需要三个维度：[高度、宽度、通道]。

## Exercise： A Single Neuron



```python
# Setup plotting
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',titleweight='bold', titlesize=18, titlepad=10)

# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning_intro.ex1 import *
# learntools 文件夹包含一个python包，它为Kaggle Learn课程的用户提供反馈。
```

### 画图`matplotlib.pyplot`

[matplotlib.pyplot的使用总结大全（入门加进阶）](https://zhuanlan.zhihu.com/p/139052035 "matplotlib.pyplot的使用总结大全（入门加进阶）")

ps：其实matplotlib语法和Matlab很像，不懂的地方可以参考Matlab的文档

`matplotlib.pyplot` 是一个在python中可实现的函数，例如创建图形、在图形中创建创建一个绘图区域、在绘图区域中你那个绘制一些线、在图形中添加标签之类的。

#### 我们经常会在画图的代码里看到，有用`plt.`的，有用`ax.`的，两者到底有什么区别呢？

```python
#fig.
fig=plt.figure(num=1,figsize=(4,4))
#plt.subplot(111) 可以省略
plt.plot([1,2,3,4],[1,2,3,4])
plt.show()
```

![图片plt](https://pic1.zhimg.com/80/v2-2a1f051d62be8ea70d772ec4b93dc8e8_720w.webp "plt")

```python
#ax.
fig=plt.figure(num=1,figsize=(4,4))
ax=fig.add_subplot(111)#选定一个子区域
ax.plot([1,2,3,4],[1,2,3,4])
plt.show()
```

![图片alt](https://pic1.zhimg.com/80/v2-2a1f051d62be8ea70d772ec4b93dc8e8_720w.webp "图片title")

**我们看到上面两种画图方式可视化结果并无不同，那区别在哪呢？**

其实呢，第一种方式呢，是先生成了一个画布，然后在这个画布上隐式的生成一个画图区域来进行画图，第二种方式，先生成一个画布，然后，我们在此画布上，选定一个子区域画了一个子图，上一张官方的图，看看你能不能更好的理解。

### `plt`的用法

`plt.style.use`是设置背景样式的函数

用`plt.style.available`查出可用样式列表如下

[【Matplotlib】plt.style.use设置背景样式](https://zhuanlan.zhihu.com/p/483906129 "超链接title")

`plt.rc`

[你真的了解matplotlib吗？---坐标轴和rc参数设置](https://zhuanlan.zhihu.com/p/138468596)

```python
plt.rcParams['font.sans-serif']=['Simhei']  #显示中文
plt.rcParams['axes.unicode_minus']=False    #显示负号   
```

**rc参数修改的是全局默认属性，也就是说，这个参数一旦设置，后续进行的所有操作都会受到rc参数的影响！**

[属性总结（三）：plt.rcParams](https://blog.csdn.net/weixin_39010770/article/details/88200298)

- 线条样式：lines

- 横、纵轴：xtick、ytick

- figure中的子图：axes

- 图像、图片：figure、savefig

[Python Matplotlib.pyplot.rc()用法及代码示例](https://vimsky.com/examples/usage/matplotlib-pyplot-rc-in-python.html)

```python
plt.rc('figure', autolayout=True)
#autolyout使axes自适应整个figure框
```

[Python数据处理笔记——matplotlib篇（一）](https://www.itdaan.com/blog/2017/09/11/2b5758288f8d4c96da5b8fa25a3da15b.html)

关键词：坐标轴范围，图像保存，坐标轴密度，axes自适应figure，matplotlib面向对象，部分理论概念

 `learntools` 文件夹包含一个python包，它为Kaggle Learn课程的用户提供反馈。

### 数据集

```python
import pandas as pd

red_wine = pd.read_csv('../input/dl-course-data/red-wine.csv')
red_wine.head()
```

红酒质量数据集包含来自大约 1600 种葡萄牙红酒的物理化学测量值。还包括盲品测试对每种葡萄酒的质量评级。

```python
#您可以使用 shape 属性获取数据框（或 Numpy 数组）的行数和列数。
red_wine.shape # (rows, columns)
```



#### 设置一个线性模型

```python
input_shape = [n]
#n是输入的神经元的数量
```

```python
#导入keras库
from tensorflow import keras
from tensorflow.keras import layers
```

```python
#Dense全连接层
model = keras.Sequential([
    layers.Dense(units=1,input_shape=[11])
])
```

Keras有两种不同的构建模型的方法：

1. **Sequential models**
2. **Functional API**

本文将要讨论的就是keras中的Sequential模型。

##### 理解Sequential模型

Sequential模型字面上的翻译是顺序模型，是简单的线性模型，但它可以构建非常复杂的神经网络，包括全连接神经网络(FCN)、卷积神经网络(CNN)、循环神经网络(RNN)、等等。

这里的Sequential更准确的应该理解为**堆叠**，通过堆叠许多层，构建出深度神经网络。

Sequential模型的核心操作是添加layers（图层）

##### 拓展：Sequential模型的核心操作

[深度学习(3)-全连接层、激活函数](https://blog.csdn.net/dgvv4/article/details/121725742)

```python
# 按顺序或堆叠构建的模型
from keras.models import Sequential
# Dense全连接层,Activation是激活函数
from keras.layers import Dense,Activation

#构建一个模型
model=Sequential()

#在模型中添加一个全连接层
model.add(Dense(units=1,activation='relu'))

#输入层为11个神经元
model.build(input_shape=[11])

#查看网络结构
model.summary()

#查看网络所有的权重w和偏置b
for p in model.trainable_variables:
    print(p.name,p.shape)
```

[理解keras中的sequential模型](https://blog.csdn.net/mogoweb/article/details/82152174)

```python
#Sequential的卷积网络
from keras.layers import Conv2d,MaxPooling2D，Flatten,Dropout

model=Sequential()
#卷积层
model.add(Conv2D(64,(3,3),activation='relu'))

#最大池化层
model.add(MaxPooling2D(pool_size=(2,2)))

#全连接层，实现的运算是output = activation(dot(input, kernel)+bias)。其中activation是逐元素计算的激活函数，kernel是本层的权值矩阵，bias为偏置向量
model.add(Dense(256,activation='relu'))

#Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合。
model.add(Dropout(0.5))

#flattening layer-展平层，常用在从卷积层到全连接层的过渡。
model.add(Flatten())
```

```python
# 打印权值和偏置值
w, b = model.layers[0].get_weights()
print('W=',w,'b=',b)
print(len(model.layers))
```

```python
#选择优化器和损失函数
model.compile(loss='binary_crossentropy',optimizer='rmsprop')
```

```python
#将训练数据提供给模型，可以指定批次大小（batch size）、迭代次数（epochs）、验证数据集（validation_data）等
model.fit(x_train,y_train,batch_size=32,epochs=10,validation_data=(x_val,y_val))
```

```python
#用测试数据评估模型
model.evaluate(x_test,y_test,batch_size = 32)
```

[理解和使用Keras的sequential模型](https://blog.csdn.net/qq_41082686/article/details/125382813)

```python
#预测数据结果
model.predict(x_data)
```

您是否看到每个输入（和偏差）都有一个权重？请注意，权重的值似乎没有任何模式。在训练模型之前，将权重设置为随机数（并将偏差设置为 0.0）。神经网络通过为其权重找到更好的值来学习。

```python
# 打印权值和偏置值
w, b = model.weights
print("Weights\n{}\n\nBias\n{}".format(w, b))
```

#### 绘制未经训练的线性模型的输出

我们提到在训练模型的权重之前是随机设置的。运行下面的单元几次，以查看随机初始化产生的不同行。 

```python
import tensorflow as tf
import matplotlib.pyplot as plt

model = keras.Sequential([
    layers.Dense(1, input_shape=[1]),
])

#随机生成输入张量
x = tf.linspace(-1.0, 1.0, 100)
y = model.predict(x)

plt.figure(dpi=100)
#dpi是分辨率
plt.plot(x, y, 'k')
#设置刻度范围
plt.xlim(-1, 1)#x轴从-1到1
plt.ylim(-1, 1)#y轴从-1到1
#设置轴标签
plt.xlabel("Input: x")#x轴标签
plt.ylabel("Target y")#y轴标签
#权重和偏置值
w, b = model.weights # you could also use model.get_weights() here
#图片的标题
plt.title("Weight: {:0.2f}\nBias: {:0.2f}".format(w[0][0], b[0]))
#图片的展示
plt.show()
```

**生成常量，序列和随机张量**

```python
tf.linspace(start, end, num)
#这个函数主要的参数就这三个，start代表起始的值，end表示结束的值，num(整数)表示在这个区间里生成数字的个数，生成的数组是等间隔生成的。start和end这两个数字必须是浮点数，不能是整数，如果是整数会出错的，请注意！
#间隔的计算公式为（end - start） /（ num - 1）

np.linspace(start, end, num)
#区别：默认精度不同。NumPy的默认精度是np.float64，而TensorFlow的默认精度是tf.float32。
#start和end这两个数字可以是整数或者浮点数！
```



