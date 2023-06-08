# 401-``TensorFlow``框架

``TensorFlow``基础

## 1. 前言

深度学习算法的成功使人工智能的研究和应用取得了突破性进展，并极大地改变了我们的生活。越来越多的开发人员都在学习深度学习方面的开发技术。

**Google 推出的``TensorFlow`` 是目前最为流行的开源深度学习框架，在图形分类、音频处理、推荐系统和自然语言处理等场景下都有丰富的应用。**

尽管功能强大，该框架学习门槛并不高，只要掌握 Python 安装和使用，并对机器学习和神经网络方面的知识有所了解就可以上手。
本文就带你来一趟 ``TensorFlow`` 的启蒙之旅。

## 2. 初识``TensorFlow``

### 2.1 `TensorFlow` 安装说明

我们先来安装 `TensorFlow` 。`TensorFlow` 对环境不算挑剔，在 Python 2.7 和 Python3 下面均可运行，操作系统 Linux、MAC、Windows 均可（注意新版本刚出来时可能只支持部分操作系统），只要是 64 位。安装`TensorFlow` 主要不同之处是 `TensorFlow` 安装包分**支持 GPU 和不支持 GPU **两种版本，名称分别为 `tensorflow-gpu`和 `tensorflow`。

- **支持GPU**

  实际生产环境最好安装支持 GPU 的版本，以利于 GPU 强大的计算能力， 不过这需要先**安装相应的 `CUDA ToolKit` 和 `CuDNN`**。

- **不支持GPU**

  相比之下，安装不支持 GPU 的`TensorFlow` 包容易些，顺利的话执行一句 `pip install tensorflow` 就 OK。如果读者在安装中遇到问题，可根据错误提示在网上搜索解决办法。

安装后，可在命令行下启动 Python 或打开 `Jupyter Notebook`，执行下面的语句验证 `TensorFlow` 是否安装成功。 

```python
import tensorflow as tf 
#用 tf 引用 TensorFlow 包已成为一种约定。在本文的所有示例代码中，均假定已事 先执行该语句。
```

### 2.2  `TensorFlow`计算模型

创建的变量都是张量（Tensor）而不是数字。

#### **张量（数据类型）：**

- 数学含义是多维数组。

  - 0维数组称为标量：1
  - 1维数组称为向量：[1,2,3]
  - 2维数组称为矩阵：[[1,2], [3,4]]
  - 3维数组称为立方：[[[1,2],[3,4]], [[5,6],[7,8]]]
  - 0维，1维，2维等等都称为张量

#### **张量的举例：**

- 深度学习中的所有数据可看成张量，如神经网络的权重、偏置等
- 一张黑白图片可以用 2 维张量表示，其中的每个元素表示图片上一个像素的灰度值。
- 一张彩色图片则需要用 3 维张量表示，其中两个维度为宽和高，另一个维度为颜色通道。

#### `TensorFlow`的计算

`TensorFlow` 的名字中就含有张量 （Tensor）这个词。另一个词 Flow 的意思是“流”，表示通过张量的流动来表达计算。

`TensorFlow` 是一个通过图（Graph）的形式来表述计算的编程系统，图中每个节点为一种操作（Operation），包括计算、初始化、赋值等。张量则为操作的输入和输出。

最新使用`tf.function`可以将eager 代码（立即执行）一键封装成graph（图）进行计算，可参考《14-1-tensorflow函数》。

```python
a = tf.constant(3)
b = tf.constant(2)
c = a + b
sess = tf.Session()
print(sess.run(c)) 
#output是5
```

如上面的 c=a+b 为张量的加法操作，等效于 `c=tf.add(a,b)`，a 和 b 是加法操作的输入，c 是加法操作的输出。把张量提交给会话对象（Session）执行，就可以得到具体的数值。

在 `TensorFlow` 中计算包含两个阶段，:

- 先以计算图的方式定义计算过程，
- 再提交给会话对象，执行计算并返回计算结果。

这是由于，`TensorFlow` 的核心不是用 Python 语言实现的， **每一步调用都需要函数库与 Python 之间的切换，存在很大开销。**而且 `TensorFlow` 通常 在 GPU 上执行，如果每一步都自动执行的话，则 GPU 把大量资源浪费在多次接收和返回数据上，远不如一次性接收返回数据高效。

我们可以把 `TensorFlow` 的计算过程设想为叫外卖。如果我们到馆子里用餐，可以边吃边上菜。如果叫外卖的话，就得先一次性点好菜谱，再让对方把饭菜做好后打包送来， 让送餐的多次跑路不太合适。

与 `sess.run (c)` 的等效的语句是 `c.eval (session = sess)`。

作为对象和参数，张量和会话刚好调了个位置。如果上下文中只用到一个会话，则
可用 `tf.InteractiveSession() `创建默认的会话对象，后面执行计算时无需再指定。即：

```python
a = tf.constant(3)
b = tf.constant(2)
c = a + b
sess = tf.InteractiveSession()
print(c.eval()) 
#output是5
```

##### 常量：`tf.constant`

另外，在先前的代码中，参数 3 和 2 被固化在代码中。

##### `tf.placeholder`

如果要多次执行加法运算，我们可以用 `tf.placeholder` 代替 `tf.constant`，而在执行时再给参数赋值。

```python
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
c = a + b
sess = tf.InteractiveSession()
# 下面的语句也可写成 print (sess.run (c, {a:3, b:2}))
print(c.eval({a:3, b:2})) 
#output是5
print (c.eval ({a:[1,2,3], b:[4,5,6]})) 
#output是[5 7 9]
```

##### 变量：`tf.Variable`

另一种存储参数的方式是使用变量对象（`tf.Variable`）。`tf.constant` 函数创建的张量不同，变量对象支持参数的更新，不过这也意味着依赖更多的资源，与会话绑定得更紧。变量对象必须在会话对象中明确地被初始化，通常调用 `tf.global_ variables_initializer` 函数**一次性初始化所有变量**。

```python
a = tf.Variable(3)
b = tf.Variable(2)
c = a + b
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
init.run()
print(v.eval())
#output是5
a.load(7)
b.load(8)
print(v.eval())
#output是15
```

在深度学习中，变量对象通常用于表示**待优化的模型参数如权重、偏置等**，其数值在训练过程中自动调整。这在本文后面的例子中可以看到。

## 3. `TensorFlow`机器学习入门

### 3.1 导入数据(入门样例)

MNIST 是一个非常有名的手写体数字识别数据集，常常被用作机器学习的入门样例。 `TensorFlow` 的封装 让使用 MNIST 更加方便。现在我们就以 MINIST 数字识别问题为例探 讨如何使用 `TensorFlow` 进行机器学习。

MNIST 是一个图片集，包含 70000 张手写数字图片。 它也包含每一张图片对应的标签，告诉我们这个是数字几。比如，标签分别是 5，0， 4，1。

```python
#在下面的代码中,input_data.read_data_sets() 函数下载数据并解压。 
from tensorflow.examples.tutorials.mnist import input_data 
# MNIST_data 为随意指定的存储数据的临时目录 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

下载下来的数据集被分成 3 部分：55000 张训练数据（minist.train）；5000 张验 证数据（mnist.validation）；10000 张测试数据（mnist.test）。切分的目的是确保 模型设计时有一个单独的测试数据集不用于训练而是用来评估这个模型的性能，从而更 加容易把设计的模型推广到其他数据集上。 

每一张图片包含个像素点。我们可以用一个数字数组来表示一张图片：

![image-20221109162643202](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221109162643202.png)

数组展开为长度是784的向量，则训练数据集 `mnist.train.images` 是一个形状为 [60000,784] 的张量。在此张量里的每一个元素，都表示某张图片里的某个像素的灰度， 其值介于 0 和 1 之间。 

MNIST 数据集的标签是长度为 10 的 one-hot 向量（因为前面加载数据时指定了 one_hot 为 True）。

**一个 one-hot 向量除了某一位的数字是 1 以外其余各维度数字都是 0。**

比如，标签 3 将表示成 ([0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0])。因此，`mnist.train. labels` 是一个 [55000, 10] 的数字矩阵。

### 3.2 设计模型

现在我们通过训练一个叫做 `Softmax` 的机器学习模型来预测图片里的数字。

回顾一 下，分类和回归（数值预测）是最基本的机器学习问题。

- 线性回归是针对回归问题最基本的机器学习模型，其基本思想是为各个影响因素分配合适的权重，预测的结果是各影响因素的加权和。

- 逻辑（Logistic）回归则常用来处理分类问题，它在线性回归的基础上， 通过 Logistic 函数（也称 Sigmoid 函数）把低于和高于参照值的结果分别转换为接近 0 和 1 的数值。

  Sigmoid 函数，连续非线性函数
  $$
  y = \frac{1}{1+e^{-x}}
  $$
  ![image-20221109184634301](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221109184634301.png)

  [Sigmoid函数解析](https://blog.csdn.net/su_mo/article/details/79281623)

  不过逻辑回归只能处理二分问题。

  `Softmax` 回归则是逻辑回归在多分类问 题上的推广。**`Softmax`把数值转换为概率分布**。

  或者用线性代数公式表示为：

$$
y=softmax(xW+b)
$$

- x 为输入数据的特征向量，向量的长度为图片的像素（28**28=784），向量中的每个元素为图片上各点的灰度值。

- W 为x的权重矩阵 , 其中 784 对应于图片的像素，10 对应于 0-9 这 10 个数字。

- b 为长度为 10 的向量，向量中的每个元素为 0-9 各个数字的偏置，得到各个数字的权重，最后 `softmax` 函数把权重转换为概率分布。
- 通常我们最后只保留概率最高的那个数字，不过有时也关注概率较高的其他数字。

下面是 `TensorFlow` 中实现该公式的代码，核心代码为最后一句，其中 **`tf.matmul` 函数表示 Tensor 中的矩阵乘法**。

注意与公式中略有不同的是，这里把 x 声明为 2 维的张量，其中第 1 维为任意长度，这样我们就可以批量输入图片进行处理。另外，为了简单起见，我们用 0 填充 W 和 b。

```python
x = tf.placeholder (tf.float32, [None, 784]) 
#placeholder函数相当于一个占位符
W = tf.Variable (tf.zeros([784, 10])) 
b = tf.Variable (tf.zeros([10])) 
y = tf.nn.softmax (tf.matmul (x, W) + b)
#tf.matmul(x,y)=x*y
```

#### 损失（成本）函数（loss function）

除了模型外，我们还需要定义一个指标来指示如何优化模型中的参数。我们通常定 义指标来表示一个模型不尽人意的程度，然后尽量最小化这个指标。这个指标称为成本 函数。

成本函数与模型是密切相关的。

- 回归问题一般用均方误差作成本函数，

- 分类问题，常用的成本函数是交叉熵（cross-entropy），公式为：
  $$
  -\sum_{i}{y_i'log(y_i)}
  $$
  其中 y 是我们猜测的概率分布，$y'$ 是实际的分布。交叉熵刻画了***<u>两个概率分布之间的距离</u>***。

  对交叉熵的理解涉及信息论方面的知识，这里我们可以把它看作**反映预测不匹配**的指标，或者说该指标**反映实际情况出乎预料的程度**。注意交叉熵是非对称的。

  交叉熵刻画了***<u>两个概率分布之间的距离</u>***，他是分类问题中使用比较广泛的损失函数。

  在`TensorFlow`中，交叉熵表示为下面的代码：

  ```python
  cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
  #reduce_sum() 用于计算张量tensor沿着某一维度的和，可以在求和后降维。
  #axis是多维数组每个维度的坐标。
  ```

  [tf.reduce_sum()用法介绍](https://blog.csdn.net/sunmingyang1987/article/details/111409678) 有简单示例

  因为交叉熵一般会与 `Softmax` 回归一起使用，所以 `TensorFlow` 对这两个功能进行了统一封装，并提供了 `tf.nn.softmax_cross_entropy_with_logits` 函数。可以直接通 过下面的代码来实现使用了 `Softmax` 回归之后的交叉熵函数。注意与公式中的 y 不同， 代码中的 y 是 `Softmax` 函数调用前的值。最后调用 `tf.reduce_mean` 函数取平均值，因为图片是批量传入的，针对每张图片会计算出一个交叉熵。

  ```python
  y = tf.matmul (x, W) + b
  cross_entropy = tf.reduce_mean (
  tf.nn.softmax_cross_entropy_with_logits (labels = y_, logits = y))
  ```

### 3.3. 设计优化算法

现在我们需要考虑:

- **如何调整参数使成本函数最小**，

这在机器学习中称为优化算法的设计问题。笔者这里对 `TensorFlow` 实现优化的过程作一个简要的介绍，要知道优化算法从某种意义上讲比模型更重要。

`TensorFlow` 是一个基于神经网络的深度学习框架。对于 `Softmax` 这样的模型， 被当作是不含隐藏层的全连接神经网络。通过调整神经网络中的参数对训练数据进行拟合，可以使得模型对未知的样本提供预测的能力，表现为前向传播和反向传播 （Backpropagation）的迭代过程。

在每次迭代的开始，首先需要选取全部或部分训练数据，通过前向传播算法得到神经网络模型的预测结果。因为训练数据都是有正确答案标注的，所以可以计算出当前神经网络模型的预测答案与正确答案之间的差距。最后，基于预测值和真实值之间的差距， 反向传播算法会相应更新神经网络参数的取值，使得在这批数据上神经网络模型的预测 结果和真实答案更加接近。

`TensorFlow` 支持多种不同的优化器，读者可以根据具体的应用选择不同的优化算法。 比较常用的优化方法有三种：

- `tf.train.GradientDescentOptimizer`
- `tf.train. AdamOptimizer` 
- `tf.train.MomentumOptimizer` 

```python
train_step = tf.train.GradientDescentOptimizer (0.01).minimize (cross_ entropy) 
```

在这里，我们要求 `TensorFlow` 用梯度下降算法（Gradient Descent）以 0.01 的学习速率最小化交叉熵。

梯度下降算法是一个简单的学习过程，`TensorFlow` 只需将每个变量一点点地往使损失（成本）不断降低的方向移动。语句返回的 train_step 表示执行优化的操作（Operation）， 可以提交给会话对象运行。

### 3.4. 训练模型

现在我们开始训练模型，迭代 1000 次。注意会话对象执行的不是 W、b 也不是 y， 而是 train_step。

```python
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run (train_step, feed_dict = {x: batch_xs, y_: batch_ys}
```

该循环的每个步骤中，我们都会随机抓取训练数据中的 100 个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符(placeholder)来运行 train_step 操作。 

使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 在这里更确切的说是**随机梯度下降训练**。 

在理想情况下，我们希望用我们所有的数据来进行每一步的训练，因为这能给我们更好的训练结果，但显然这需要很大的计算开销。所以，每一次训练我们可以**使用不同的数据子集**，这样做既可以减少计算开销，又可以最大化地学习到数据集的总体特性。

### 3.5. 评估模型

到验证我们的模型是否有效的时候了。我们可以基于训练好的 W 和 b，用测试图片计算出 y，并取预测的数字与测试图片的实际标签进行对比。 

在 `Numpy` 中有个非常有用的函数 `argmax`，它能给出数组中最大元素所在的索引值。 由于标签向量是由 0,1 组成，因此最大值 1 所在的索引位置就是类别标签。对 y 而言， 最大权重的索引位置就是预测的数字，因为 `softmax` 函数是单调递增的。下面代码比较各个测试图片的预测与实际是否匹配，并通过均值函数计算正确率。

**argmax: 给出数组中最大元素所在的索引值。**

```python
import numpy as np
output = sess.run (y, feed_dict = {x: mnist.test.images})
print (np.mean (np.argmax(output,1) == np.argmax(mnist.test.labels,1)))
```

我们也可以让 `TensorFlow` 来执行比较，这在很多时候更为方便和高效 `TensorFlow` 中也有类似的 argmax 函数。

```python
correct_prediction = tf.equal (tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean (tf.cast(correct_prediction, "float"))
print (sess.run (accuracy, feed_dict = {x: mnist.test.images, y_: mnist.
test.labels}))
```

这个最终结果值应该大约是 91%。完整的代码请参考 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py，有少量修改。

## 4.TensorFlow 深度学习入门

### 4.1. 卷积神经网络介绍

​		前面我们使用了单层神经网络。如果**增加神经网络的层数**，可以进一步提高正确率。 不过，增加层数会使*<u>需要训练的参数增多</u>*，这除了导致计算速度减慢，还很容易引发<u>过拟合问题</u>。所以需要一个更合理的神经网络结构来有效地减少神经网络中参数个数。 对于图像识别这类问题，卷积神经网络（CNN）是目前最为有效的结构。 

#### 卷积神经网络（CNN）

<u>卷积神经网络是一个层级递增的结构，其基本思想是从对像素、边缘的认识开始， 再到局部形状，最后才是整体感知。</u>

**传统方法**：在分类前对图像进行预处理，如平滑、去噪、光照归一化等， 从中提取角点、梯度等特征，

而卷积神经网络把这一过程自动化。

当然，神经网络是一 个黑盒子，没有前面所提到的这些概念，它所提取的都是**抽象意义上的特征**，与人类理解的语意特征无法对应。况且经过多层变换，图片早已面目全非。另外卷积神经网络也可以用于图像识别以外的领域。不过为了浅显易懂，下文中仍然使用像素、颜色之类的日常用语。

卷积神经网络中特征识别的基本手段是卷积（Convolution）。我们可以理解为把图片进行**特效处理**，新图片的每个位置的像素值是原图片对应位置及相邻位置像素值的某种方式的叠加或取反，类似于 Photoshop 中的**滤镜**如模糊、锐化、马赛克什么的， `TensorFlow` 中称为过滤器（Filter）。卷积的计算方式是相邻区域内像素的加权求和， 用公式表示的话，仍是，不过计算限定在很小的矩形区域内。

![动图](https://pic1.zhimg.com/50/v2-c658110eafe027eded16864fb6a28f46_720w.webp?source=1940ef5c)

由于卷积**只针对图片的相邻位置**，可保证训练后能够对于局部的输入特征有最强的响应。另外，不论在图像的什么位置，都使用同一组权重，相当于把过滤器当作手电筒在图片上来回扫描，这使图像内容**在图片中的位置不影响判断结果**。卷积网络的这些特点使它**显着减少参数数量**的同时，又能够更好的利用图像的结构信息，提取出图像从低级到复杂的特征，甚至可以超过人类的表现。

神经网络需要使用激活函数去除线性化，否则即便增加网络的深度也依旧还是线性映射，起不到多层的效果。与 `Softmax` 模型所使用的 Sigmoid 函数不同，卷积神经网络钟爱激活函数的是 `ReLU`，有利于反向传播阶段的计算，也能缓解过拟合。`ReLU` 函数很简单，就是忽略小于 0 的输出，可以理解为像折纸那样对数据进行区分。注意在使用 `ReLU` 函数时，比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为 0 的问题。

- softmax函数的激活函数：sigmoid函数

- 卷积神经网络的激活函数：`ReLU`函数


除了卷积外，卷积神经网络通常还会用到**降采样（`downsampling` 或 `subsampling`）**。我们可以理解为<u>把图片适当缩小</u>，由此在一定程度上<u>控制过拟合并减少图像旋转、扭曲对特征提取的影响</u>，因为降采样过程中**模糊了方向信息**。卷积神经网络正是通过卷积和降采样，成功将数据量庞大的图像识别问题不断降维，最终使其能够被训练。降采样在卷积神经网络中通常被称为**池化（Pooling）**，包括最大池化、平均池化等。其中最常见的是最大池化，它将输入数据分成不重叠的矩形框区域，对于每个矩形框的数值取最大值作为输出。

### 4.2. 构建 LeNet-5 网络

对卷积神经网络有了基本了解后，我们现在开始使用这种网络来处理 MNIST 数字识 别问题。这里参照最 经典的 LeNet-5 模型，介绍如何使用 TensorFlow 进行深度学习。

LeNet-5 的结构如下图所示。可看出，LeNet-5 中包含两次的卷积和降采样，再经过两 次全连接并使用 Softmax 分类作为输出。

<img src="C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221110133812336.png" alt="image-20221110133812336"  />

模型第一层是卷积层。输入是原始图片，尺寸为（28，28，1），颜色用灰度表示，因此数据类型为float32，考虑到批量输入x，数据应有 4 个维度。过滤器尺寸为32，计算 32 个特征，因此权重 W 为（5，5，1，32）?的张量，偏置 b 为长度 32 的向量。另外，为确保输出的图片仍为大小（28，28），在对图片边缘的像素进行卷积时，我们用 0 补齐周边。

`TensorFlow` 中，tf.nn.conv2d 函数实现卷积层前向传播的算法。这个函数的前两个参数分别表示输入数据 x 和权重 W，均为 4 个维度的张量，如前所述。权重在初始化时应该加入少量的噪声来打破对称性以及避免 0 梯度，这里我们用 `tf.truncated_ normal` 函数生成的随机量填充。函数的随后两个参数定义卷积的方式，包括过滤器在图像上滑动时移动的步长及填充方式。步长用长度为 4 的数组表示，对应输入数据的 4 个 维度，实际上只需要调整中间两个数字，这里我们设置为 [1, 1, 1, 1]，表示一个像素 一个像素地移动。填充方式有“SAME”或“VALID”两种选择，其中“SAME”表示添加 全 0 填充，“VALID”表示不添加。

下面的代码实现模型第一层：

```python
x = tf.placeholder(tf.float32,[None, 784])
# 这里使用tf.reshape函数校正张量的维度，-1表示自适应。
x_image = tf.reshape(x,[-1,28,28,1])
W_conv1 = tf.Variable (tf.truncated_normal ([5, 5, 1, 32], stddev = 0.1))
# tf.truncated_normal生成的随机量填充,权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。
# tf.truncated_normal产生正态分布，均值和标准差自己设定。
b_conv1 = tf.Variable (tf.constant (0.1, shape = [32]))
# 执行卷积后使用 ReLU 函数去线性化
h_conv1 = tf.nn.relu (tf.nn.conv2d(
    x_image, W_conv1, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv1)
```

模型第二层为降采样层。采样窗口尺寸为[1, 2, 2, 1]，不重叠，因此步长也是[1, 2, 2, 1]，采用最大池化，采样后图像的尺寸缩小为原来的一半。实现图片最大池化的函数是 `tf.nn.max_pool`。它的参数与 `tf.nn.conv2d` 类似，只不过第二个参数设置的不是权重而是采样窗口的大小，用长度为 4 的数组表示，对应输入数据的 4 个维度。

```python
h_pool1 = tf.nn.max_pool (h_conv1, ksize = [1, 2, 2, 1],
strides = [1, 2, 2, 1], padding = 'SAME')
```

模型第三层为卷积层。输入数据尺寸为[5, 5, 32, 64]，有 32 个特征，过滤器尺寸仍为32，需计算 64 个特征，因此权重 W 的类型为[5, 5, 32, 64]，偏置 b 为长度 64 的向量。

```python
W_conv2 = tf.Variable (tf.truncated_normal ([5, 5, 32, 64], stddev =
0.1))
b_conv2 = tf.Variable (tf.constant(0.1, shape = [64]))
# 执行卷积后使用 ReLU 函数去线性化
h_conv2 = tf.nn.relu (tf.nn.conv2d(
h_pool1, W_conv2, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv2)
```

模型第四层为降采样层，与第二层类似。图像尺寸再次缩小一半。

```python
h_pool2 = tf.nn.max_pool (h_conv2, ksize = [1, 2, 2, 1],
strides = [1, 2, 2, 1], padding = 'SAME')
```

模型第五层为全连接层。输入数据尺寸为[7 * 7 * 64, 1024]，有 64 个特征，输出 1024 个神经元。 由于是全连接，输入数据 x 和权重 W 都应为 2 维的张量。全连接参数较多，这里引入 Dropout 避免过拟合。Dropout 在每次训练时随机禁用部分权重，相当于多个训练实例 上取平均结果，同时也减少了各个权重之间的耦合。`TensorFlow` 中实现 Dropout 的函数 为 `tf.nn.dropout`。该函数第二个参数表示每个权重不被禁用的概率。

```python
W_fc1 = tf.Variable (tf.truncated_normal ([7 * 7 * 64, 1024], stddev =
0.1))
b_fc1 = tf.Variable (tf.constant (0.1, shape = [1024]))
# 把 4 维张量转换为 2 维
h_pool2_flat = tf.reshape (h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu (tf.matmul (h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder (tf.float32) 
h_fc1_drop = tf.nn.dropout (h_fc1, keep_prob)
```

模型最后一层为全连接加上 `Softmax` 输出，类似之前介绍的单层模型。

```python
W_fc2 = tf.Variable (tf.truncated_normal ([1024, 10], stddev = 0.1))
b_fc2 = tf.Variable (tf.constant(0.1, shape = [10]))
y_conv = tf.matmul (h_fc1_drop, W_fc2) + b_fc2
cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits (labels = y_, logits = y_conv))
```



### 4.3 训练和评估模型

为了进行训练和评估，我们使用与之前简单的单层 `Softmax` 模型几乎相同的一套代码，只是我们会用更加复杂的 ADAM 优化器来缩短收敛时间，另外在 feed_dict 中加入额外的参数 keep_prob 来控制 Dropout 比例。然后每 100 次迭代输出一次日志。

```python
train_step = tf.train.AdamOptimizer (1e-4).minimize (cross_entropy)
correct_prediction = tf.equal (tf.argmax (y_conv, 1), tf.argmax (y_, 1))
accuracy = tf.reduce_mean (tf.cast (correct_prediction, tf.float32))
sess = tf.InteractiveSession()
sess.run (tf.global_variables_initializer()) for i in range(20000):
batch_xs, batch_ys = mnist.train.next_batch(50) if i % 100 == 0:
train_accuracy = accuracy.eval (feed_dict = { x: batch_xs, y_: batch_ys,
keep_prob: 1.0})
print('step %d, training accuracy %g' % (i, train_accuracy)) train_step.
run (feed_dict = {x: batch_xs, y_: batch_ys, keep_prob: 0.5})
print ('test accuracy %g' % accuracy.eval (feed_dict = {
x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```

以上代码，在最终测试集上的准确率大概是 99.2%。完整的代码请参考 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py，局部有修改。

## 5. 总结

在本文中，我们介绍了 `TensorFlow` 的基本用法，并以 MNIST 数据为例，基于 `Softmax` 模型和卷积神经网络分别讲解如何使用 `TensorFlow` 进行机器学习和深度学习。`TensorFlow` 对深度学习提供了强大的支持，包含丰富的训练模型，还提供了 `TensorBoard`、`TensorFlow` 游乐场、`TensorFlow Debugger` 等可视化和调试等手段方便。 限于篇幅，这里不一一介绍，详见 `TensorFlow` 的官方文档。深度学习是一名较新的技术， 理论和实践中都有不少坑。不过只要多学多上手，相信能让 `TensorFlow` 成为您手中的利器。

### 参考文献：

1.《TensorFlow：实战 Google 深度学习框架》 才云科技、郑泽宇、顾思宇著
2.《面向机器智能的 TensorFlow 实践》 Sam Abrahams 等著，段菲、陈澎译
3.《你好，TensorFlow》 http://mp.weixin.qq.com/s/0qJmicqIxwS7ChTvIcuJ-g
4.《TensorFlow 白皮书》（译文） http://www.jianshu.com/p/65dc64e4c81f
5.《卷积神经网络》 http://blog.csdn.net/celerychen2009/article/details/8973218
6.《卷积神经网络入门学习》 http://blog.csdn.net/hjimce/article/details/51761865

#### 作者简介（本文是摘录）

王小鉴，重庆大学计算机硕士，IT 老兵，现于重庆一家公司从事技术研发及团队管理。对海量数据存储、分布式计算、数据分析、机器学习有浓厚兴趣，重点关注性能优化、自然语言处理等技术。
