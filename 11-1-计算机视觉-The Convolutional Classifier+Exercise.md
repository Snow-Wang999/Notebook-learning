# 卷积分类器-The Convolutional Classifier

在本课程中，您将： 

- 用Keras与现代深度学习网络创建图像分类器（image classifier）
- 使用可重用的块设计您自己的自定义卷积网络（custom convnet） 
- 了解视觉特征提取（feature extraction）背后的基本思想
- 掌握迁移学习（transfer learning）的艺术来提升你的模型
- 利用数据扩充（data augmentation）来扩展您的数据集

## 介绍

本课程将向您介绍计算机视觉的基本思想。我们的目标是了解**神经网络如何“理解”自然图像**，足以解决人类视觉系统可以解决的同类问题。

最擅长这项任务的神经网络称为卷积神经网络（**convolutional neural networks**）（有时我们说 **convnet**  或 **CNN**。）卷积是一种**数学运算**，它赋予卷积网络的层独特的结构。在以后的课程中，您将了解为什么这种结构在解决计算机视觉问题方面如此有效。

我们将把这些想法应用到**图像分类**问题上：给定一张图片，我们可以训练计算机告诉我们它是什么图片吗？您可能已经看到可以从照片中识别植物种类的应用程序。那是一个图像分类器！在本课程中，您将学习如何构建与专业应用程序中使用的图像分类器一样强大的图像分类器。

虽然我们的重点将放在图像分类上，但您将在本课程中学到的内容与各种计算机视觉问题相关。最后，您将准备好继续使用更高级的应用程序，例如生成对抗网络（generative adversarial networks-GAN）和图像分割（image segmentation）。

## 卷积分类器-The Convolutional Classifier

用于图像分类的卷积网络由两部分组成：

- 卷积基——a convolutional base 
- 密集头——a dense head

![image-20221030144207276](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030144207276.png)

卷积基（a convolutional base）用于**从图像中提取特征**。它主要由执行卷积操作的层组成，但通常也包括其他类型的层。 

密集头（a dense head）用于确定**图像的类别**。它主要由dense层组成，但可能包括其他层，如 dropout。 

我们所说的视觉特征是什么意思？特征可以是线条、颜色、纹理、形状、图案——或一些复杂的组合。 

整个过程是这样的：

![image-20221030144432516](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030144432516.png)

实际提取的特征看起来有点不同，但它给出了这个想法。

## 训练分类器-Training the Classifier

网络在训练期间的目标是学习两件事： 

- 从图像（基础-base）中提取哪些特征【which features to extract from an image (base),】
- 哪个类具有什么特征（头部-head）【which class goes with what features (head)】

如今，很少从头开始训练卷积网络。更常见的是，我们重用预训练模型的基础（**reuse the base of a pretrained model**. ）。然后，我们将一个未经训练的头部连接（**attach an untrained head**）到预训练的基础上。换句话说，我们重用网络中已经学会做的部分 1. 提取特征，并附加一些新的层来学习 2. 分类。

![image-20221030144838780](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030144838780.png)

因为头部通常只包含几个密集（dense）层，所以可以从相对较少的数据中创建非常准确的分类器。

重用预训练模型是一种称为迁移学习的技术。它非常有效，以至于现在几乎每个图像分类器都会使用它。

## Example - Train a Convnet Classifier

在整个课程中，我们将创建试图解决以下问题的分类器：

**这是汽车的图片还是卡车的图片？**

我们的数据集是大约 10,000 张各种汽车的图片，大约一半是汽车，一半是卡车。

### 第一步 - 数据加载

```python
# Imports
import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
#gridspec.Gridspec(5，5)创建网格子图5*5

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
#从自己硬盘加载自己的图像数据集

# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed(31415)

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # to clean up output cells


# Load training and validation sets
ds_train_ = image_dataset_from_directory(
    '../input/car-or-truck/train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)
ds_valid_ = image_dataset_from_directory(
    '../input/car-or-truck/valid',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)

# Data Pipeline
# 数据类型转换为浮点数
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
ds_valid = (
    ds_valid_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
```

```python
import matplotlib.pyplot as plt
```

***

#### 复现结果-随机种子-random seed

为什么数据和代码一样，但是模型结果不一样？

因为在机器学习中，很有处理或算法带有一定的随机性，因此，产生差异。

常见的带有随机性的一些算法及处理步骤： 

1. 神经网络当中的**初始化权重w**； 
2. 聚类算法，例如K-means算法的**初始聚类中心**； 
3. 随机森林算法中牵涉到的**数据或特征抽样**； 
4. 在整体数据集当中，**随机抽取样本组成的训练集及测试集**。

要复现结果，用到随机种子。

计算机生成随机数是利用伪随机数生成器（PRNG）

“伪随机数”可以通过固定初始值，保证随机数的一致性，而这个初始值就是随机种子（seed），就可以复现结果

```py
# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)    
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed(31415)
```

[python中random.seed（）究竟做什么用？](https://www.zhihu.com/question/413925742)

***

#### 数据可视化

[Python数据可视化之使用GridSpec自定义子图](https://blog.csdn.net/FrankieHello/article/details/79626728)

`gridspec.Gridspec(5，5)`创建网格子图5*5

```python
random_nums = np.random.random_integers(1,100,(100,2))
gs = gridspec.Gridspec(5，5)
fig1 = plt.figure()
ax1 = fig1.add_subplot(gs[0:2,0:2])
ax1.plot(range(len(random_nums)), random_nums)

plt.show()
```

***

#### 从自己硬盘加载自己的图像数据集

[Python `tf.keras.utils.image_dataset_from_directory`用法及代码示例](https://vimsky.com/zh-tw/examples/usage/python-tf.keras.utils.image_dataset_from_directory-tf.html)

```python
tf.keras.utils.image_dataset_from_directory(
    directory, 
    #数据所在目录
    labels='inferred', 
    #inferred包含子目录，从目录结构生成标签
    label_mode='int',
    #标签的编码，选项有'categorical'，'binary' 
    class_names=None, 
    #类目的显示列表
    color_mode='rgb', 
    #图像颜色通道，默认为'rgb'，选项为 "grayscale"、 "rgb"、 "rgba" 之一。
    batch_size=32, 
    #数据批次大小，默认为32
    image_size=(256,256), 
    #读取图像后调整的图像大小，默认为 (256, 256)
    shuffle=True,
    #是否打乱数据 ，默认值：真。
    seed=None, 
    #用于洗牌和转换的可选随机种子
    validation_split=None,
    #保留用于验证的数据的一部分，0-1的可选浮点数 
    subset=None,
    #"training" 或 "validation" 之一。仅在设置 validation_split 时使用子集。
    interpolation='bilinear', 
    #字符串，调整图像大小时使用的插值方法，默認為 bilinear 。支持 bilinear , nearest , bicubic , area , lanczos3 , lanczos5 , gaussian , mitchellcubic 。
    follow_links=False,
    #是否访问符号链接指向的子目录。默认为假。
    crop_to_aspect_ratio=False,
    #如果为 True，则调整图像大小而不会出现纵横比失真。当原始纵横比与目标纵横比不同时，将裁剪输出图像以返回与目标纵横比匹配的图像(大小为image_size)中最大的可能窗口。默认情况下(crop_to_aspect_ratio=False)，可能不会保留纵横比。
    **kwargs
)
```

***

#### TF-高效流水线优化pipeline

```python
AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
```

##### 1. 介绍-Tensorflow高效流水线Pipeline

[**Tensorflow高效流水线Pipeline**](https://www.cnblogs.com/huangyc/p/10340766.html)

GPU和TPU可以显著缩短执行单个训练步所需的时间。实现最高性能需要高效的输入流水线，以在当前时间步完成之前为下一步提供数据。

tf.data API可以帮助我们构建灵活高效的输入流水线。

##### 2. Pipeline Structure输入流水线结构

我们可以将典型的 TensorFlow 训练输入流水线视为 ETL 流程： 

- Extract：从永久性存储（可以是 HDD 或 SSD 等本地存储或 GCS 或 HDFS 等远程存储）**读取数据**。

- Transform：使用CPU核心解析数据并对其执行**预处理操作**，例如图像解压缩、数据增强转换（例如随机裁剪、翻转和颜色失真）、重排和批处理、数据类型转换。 

- Load：将转换后的数据**加载到执行机器学习模型的加速器设备**（例如，GPU 或 TPU）上。

这种模式可高效利用 CPU，同时预留加速器来完成对模型进行训练的繁重工作。此外，将输入流水线视为 ETL 流程可提供便于应用性能优化的结构。

###### 2.1 最佳Pipeline步骤

- 使用 <u>prefetch 转换可将提供方和使用方的工作重叠</u>。我们特别建议将 `prefetch(n)`（其中 n 是单步训练使用的元素数/批次数）添加到输入流水线的末尾，以便将在 CPU 上执行的转换与在加速器上执行的训练重叠。 
- <u>通过设置 `num_parallel_calls` 参数并行处理 map 转换</u>。建议您将其值设为可用 CPU 核心的数量。 
- 如果您使用 batch 转换将预处理元素组合到一个批次中，<u>建议您使用 map_and_batch 混合转换；特别是在您使用的批次较大时</u>。 
- 如果您要处理<u>远程存储的数据并/或需要反序列化</u>，建议您使用 parallel_interleave 转换来重叠从不同文件读取（和反序列化）数据的操作。 
- 向量化传递给 map 转换的低开销用户定义函数，以分摊与调度和执行相应函数相关的开销。 
- 如果内存可以容纳您的数据，请使<u>用 cache 转换在第一个周期中将数据缓存在内存中</u>，以便后续周期可以避免与读取、解析和转换该数据相关的开销。 
- 如果<u>预处理操作会增加数据大小</u>，建议您首先应用 interleave、prefetch 和 shuffle（如果可以）以减少内存使用量。
-  建议您在<u>应用 repeat 转换之前先应用 shuffle 转换</u>，最好使用 shuffle_and_repeat 混合转换。

##### 3. 优化性能

由于新型计算设备（例如 GPU 和 TPU）可以不断提高神经网络的训练速度，因此，CPU 处理很容易成为瓶颈。tf.data API 为用户提供构建块来设计可高效利用 CPU 的输入流水线，并优化 ETL 流程的每个步骤。

###### 自动调整数据集性能

```
tf.data.experimental.AutotuneOptions()
```

[**tf.data模块--二 优化pipeline性能**](https://zhuanlan.zhihu.com/p/163656225)

###### 3.1 prefetch预取数据

`dataset.prefetch()`的作用是会在第n个epoch的training的同时预先fetch第n+1个epoch的data,这个操作的实现是在background开辟一个新的线程，将数据读取在cache中，这也大大的缩减了总的训练的时间。

```python
benchmark(
    ArtificialDataset()
    .prefetch(tf.data.experimental.AUTOTUNE)
)
```

###### 3.2 map并行处理数据转换（Parallelizing Data Transformation）

准备批次数据时，可能需要预处理输入元素。

为此，tf.data API 提供了 tf.data.Dataset.map 转换，以将用户定义的函数（例如，正在运行的示例的 parse_fn）应用于输入数据集的每个元素。由于输入元素彼此独立，因此可以跨多个 CPU 核心并行执行预处理。为实现这一点，map 转换提供了 num_parallel_calls 参数来指定并行处理级别。

并行后，由于数据预处理的时间缩短，整体的时间也减少了。如何为 num_parallel_calls 参数选择最佳值取决于硬件、训练数据的特征（例如其大小和形状）、映射函数的成本以及同时在 CPU 上进行的其他处理；一个简单的启发法是设为可用 CPU 核心的数量。例如，如果执行以上示例的机器有 4 个核心，则设置 num_parallel_calls=4 会更高效。另一方面，将 num_parallel_calls 设置为远大于可用 CPU 数量的值可能会导致调度效率低下，进而减慢速度。

###### 3.3 并行处理远程数据提取（**Parallel data extraction**）

在实际设置中，输入数据可能会***远程存储***（例如，GCS 或 HDFS），这是因为输入数据<u>不适合本地存储</u>，或因为训练是<u>分布式训练</u>，因此在每台机器上复制输入数据没有意义。非常适合在本地读取数据的数据集流水线在远程读取数据时可能会遇到 I/O 瓶颈，这是因为*本地存储和远程存储之间存在以下差异*： 

- **首字节时间**：与本地存储相比，从远程存储读取文件的首字节所用时间可能要多出几个数量级。 

- **读取吞吐量**：虽然远程存储通常可提供较大的聚合带宽，但读取单个文件可能只能利用此带宽的一小部分。 

此外，将原始字节读入内存中后，可能还需要对数据进行反序列化或解密（例如，protobuf），这会带来额外的开销。无论数据是在本地还是远程存储，都存在这种开销，但如果未有效预取数据，则在远程存储的情况下可能更糟。 

为了降低各种数据提取开销的影响，tf.data API 提供了 tf.contrib.data.parallel_interleave 转换。使用此转换可以并行执行其他数据集（例如数据文件读取器）并交错这些数据集的内容。可以通过 cycle_length 参数指定要重叠的数据集的数量。

###### 3.4 缓存（cache）

据官网的介绍cache操作可以将数据集缓存在本地或者内存中，可以节省在每一个新的epoch的时间

###### 3.5 建议减小内存的消耗

类似于interleave, prefetch,shuffle都会有内部存储的消耗，map的顺序会影响内存的消耗，通常我们会选择内存消耗较小的顺序，除非特定的顺序能带来较大的性能的提升。

缓存部分的操作

```text
dataset.map(time_consuming_mapping).cache().map(memory_consuming_mapping)
```

##### 4. 训练时间开销图

###### 1) 普通训练时间开销

![image-20221030162810127](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030162810127.png)

![image-20221030165847925](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030165847925.png)

###### 2) prefetch训练时间开销

![image-20221030162956704](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030162956704.png)

![image-20221030165859089](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030165859089.png)

3）map并行处理数据转换

![image-20221030180307444](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030180307444.png)

![image-20221030180332165](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030180332165.png)

例如，下图说明了将 num_parallel_calls=2 设置为 map 转换的效果：

![image-20221030170018851](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030170018851.png)

4）并行处理远程数据提取-parallel_interleave

![image-20221030170425815](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030170425815.png)![image-20221030170530671](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030170530671.png)

![image-20221030170207377](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030170207377.png)

5）cache缓存

![image-20221030180414835](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030180414835.png)

***

### 第 2 步 - 定义预训练基础(Define Pretrained Base)

最常用的预训练数据集是 ImageNet，这是一个包含多种自然图像的大型数据集。 Keras 在其应用程序模块中包含在 ImageNet 上预训练的各种模型。我们将使用的预训练模型称为 VGG16。

```python
pretrained_base = tf.keras.models.load_model(
    '../input/cv-course-models/cv-course-models/vgg16-pretrained-base',
)
pretrained_base.trainable = False
```

***

## 第 3 步 - 连接头部(Attach Head)

接下来，我们附加分类器头。对于这个例子，我们将使用一层隐藏单元（第一个dense层），然后使用一层将输出转换为第 1 类（卡车）的概率分数。 Flatten 层将基础的二维输出转换为头部所需的一维输入。

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    pretrained_base,
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])
```

***

### 第 4 步 - 训练(Train)

最后，让我们训练模型。由于这是一个二分类问题，我们将使用交叉熵(crossentropy)和准确性(accuracy)的二进制版本。Adam优化器通常表现良好，所以我们也会选择它。

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=30,
    verbose=0,
)
```

在训练神经网络时，检查损失(loss)和度量图(metric plots)总是一个好主意。历史对象在字典 history.history 中包含此信息。我们可以使用 Pandas 将此字典转换为数据框并使用内置方法绘制它。

```python
import pandas as pd

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();
```

![image-20221030162203639](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030162203639.png)

![image-20221030162211942](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030162211942.png)

***

### 结论

在本课中，我们了解了卷积网络分类器的结构：在执行特征提取的基础之上充当分类器的头部。 (a **head** to act as a classifier at the top of a **base(预训练的模型)**)

头部本质上是一个普通的分类器，就像你在入门课程中学到的一样。对于特征，它使用基础提取的那些特征。这是卷积分类器背后的基本思想：我们可以将一个**执行特征工程的单元**附加到分类器本身。 

这是深度神经网络相对于传统机器学习模型的一大优势：给定正确的网络结构，深度神经网络可以学习如何设计解决问题所需的特征。 

在接下来的几节课中，我们将看看卷积基如何完成特征提取。然后，您将学习如何应用这些想法并设计一些您自己的分类器。

***

## Exercise: The Convolutional Classifier

### kaggle-CPU转GPU

1. 注册/登录kaggle（www.kaggle.com）-谷歌账户

2. 上传dataset或打开已有数据集

3. 创建notebook

   ![image-20221030184218006](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030184218006.png)

4. 联网前需手机验证

   在右边侧边栏（toggle sidebar visibility）里的setting栏里面点击手机验证

5. accelerator出现前需联网

   ![image-20221030184116709](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030184116709.png)

6. setting栏里accelerator中选gpu型号

   ![image-20221030184344212](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030184344212.png)

7. 免费gpu资源

   - Kaggle 提供对2个NVIDIA T4的GPU的免费访问，并且CPU运算相关的内存升级到了30GB！
   - Kaggle 提供对 NVIDIA TESLA P100 GPU 的免费访问。
   - GPU配额没有变化，两个GPU环境将计入相同的配额。（这带来了令人兴奋的新机会，如训练更大的模型和更快的一些工作负载的训练时间，同时也提供了一种学习如何使用多gpu环境的好方法。）
   - 这些 GPU 对于训练深度学习模型很有用，尽管它们不能加速大多数其他工作流程（即，像 pandas 和 scikit-learn 之类的库不能从使用 GPU 中受益）。
   - 您每周最多可以使用一个配额限制的 GPU。配额每周重置一次，根据需求和资源为 30 小时或有时更高

8. GPU配额使用技巧

   - 仅当您打算使用 GPU 时才打开 GPU。 GPU 仅在您使用利用 GPU 加速库（例如 TensorFlow、PyTorch 等）的代码时才有用。 
   - 主动监控和管理您的 GPU 使用情况 
   - Kaggle 在笔记本编辑器的设置菜单、kaggle.com/notebooks 页面顶部、您的个人资料页面和会话管理窗口中提供了用于监控 GPU 使用情况的工具。 
   - 避免使用批处理会话（提交按钮）来保存或检查您的进度。批处理会话（提交）从上到下运行所有代码。这比简单地从笔记本编辑器下载 .ipynb 文件效率低。 
   - 取消不必要的批处理会话 
   - 如果您在完成第一次提交之前按下提交按钮，则同一笔记本可以有多个并发批处理会话。如果与以前的代码相比，您的最新代码已更新，您最好取消第一次提交并只保留第二次提交运行。 
   - 在关闭窗口之前停止交互式会话。交互式会话保持活动状态，直到达到 60 分钟的空闲超时限制。如果您在关闭窗口之前停止会话，则最多可以节省 60 分钟的计算时间。 
   - 您可以使用屏幕左下角的活动事件窗口来管理活动会话，包括停止未使用的交互式会话。在此处了解有关活动事件的更多信息。 
   - 考虑使用 Kaggle-API 来完全避免交互式会话。使用 Kaggle API，您可以推送新版本的笔记本，而无需在笔记本编辑器中打开交互式会话。

***

### 介绍

在本教程中，我们看到了如何通过将密集层的头部（a head of dense）附加到预训练的基础（pretrained base）上来构建图像分类器。我们使用的base来自一个名为 VGG16 的模型。我们看到 VGG16 架构容易过度拟合这个数据集。在本课程中，您将学习多种可以在初次尝试后进行改进的方法。 

您将看到的第一种方法是使用更适合数据集的基础。该模型来自的基础称为 InceptionV1（也称为 GoogLeNet）。 InceptionV1 是 ImageNet 竞赛的早期获胜者之一。它的继任者之一，InceptionV4，是当今最先进的技术之一。 

```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.computer_vision.ex1 import *

# Imports
import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # to clean up output cells


# Load training and validation sets
ds_train_ = image_dataset_from_directory(
    '../input/car-or-truck/train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)
ds_valid_ = image_dataset_from_directory(
    '../input/car-or-truck/valid',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)

# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
ds_valid = (
    ds_valid_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

```

在 ImageNet 上预训练的 InceptionV1 模型在 TensorFlow Hub 存储库中可用，但我们将从本地副本加载它。运行此单元为您的基础加载 InceptionV1。

```python
import tensorflow_hub as hub

pretrained_base = tf.keras.models.load_model(
    '../input/cv-course-models/cv-course-models/inceptionv1'
)
```



### 1. 定义预训练基础

现在你已经有了一个预训练的基础来进行我们的特征提取，决定这个基础是否应该是可训练的。

```python
pretrained_base.trainable = False
```

在进行迁移学习时，重新训练整个基础通常不是一个好主意——至少不是不小心。原因是头部中的随机权重最初会创建大的梯度更新，这些梯度更新会传播回基础层并破坏大部分预训练。使用称为微调的技术，可以根据新数据进一步训练基础，但这需要一些小心才能做好。

### 2. 连接头（attach head）

现在定义了基础来进行特征提取，创建一个密集层的头部来执行分类，如下图：

![image-20221030190710434](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030190710434.png)

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    pretrained_base,
    layers.Flatten(),
    layers.Dense(6, activation ='relu'),
    layers.Dense(1, activation = 'sigmoid'),
])
```

### 3. 训练

在 Keras 中训练模型之前，您需要指定一个优化器来执行梯度下降、一个要最小化的损失函数以及（可选）任何性能指标。我们将在本课程中使用的优化算法称为“Adam”，无论您要解决什么样的问题，它通常都能很好地执行。 

然而，损失和指标需要与您尝试解决的问题类型相匹配。我们的问题是一个二进制分类问题：汽车编码为 0，卡车编码为 1。为二进制分类选择适当的损失和适当的准确度度量。

```python
# YOUR CODE HERE: what loss function should you use for a binary
# classification problem? (Your answer for each should be a string.)
optimizer = tf.keras.optimizers.Adam(epsilon=0.01)
model.compile(
    optimizer=optimizer,
    loss = 'binary_crossentropy',
    metrics=['binary_accuracy'],
)

```

```python
history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=30,
)
```

画图

```python
import pandas as pd
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();
```

![image-20221030192717702](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030192717702.png)

![image-20221030192729153](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030192729153.png)

### 4. 评估-检查损失和准确性

您是否注意到这些学习曲线与教程中的 VGG16 曲线之间的差异？这个差异告诉你这个模型（InceptionV2）与 VGG16 相比学到了什么？有没有一种方法比另一种更好？更差？ 想好之后，运行下面的单元格来查看答案。

```python
history_frame.loc[15:, ['loss', 'val_loss']].plot()
history_frame.loc[15:, ['binary_accuracy', 'val_binary_accuracy']].plot();
```

![image-20221030192759073](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030192759073.png)

![image-20221030192809162](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030192809162.png)

训练损失和验证损失保持相当接近证明该模型不仅仅是记忆训练数据，而是学习这两个类的一般属性。但是，由于该模型在比 VGG16 模型更大的损失下收敛，因此它可能会欠拟合一些，并且可以从一些额外的容量中受益。

### 5. 结论

在第一课中，您学习了**卷积图像分类器**的基础知识，它们由一个用于从图像中提取特征的**基础(base)**和一个使用这些特征来决定图像类别的**头部(head)**组成。您还看到了如何在预训练的基础上构建带有**迁移学习**的分类器。
