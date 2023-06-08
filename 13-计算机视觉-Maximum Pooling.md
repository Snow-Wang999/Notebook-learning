# 最大池化(Maximum Pooling）

## 介绍

在第 2 课中，我们开始讨论卷积网络中的基础如何执行特征提取。我们了解了此过程中的前两个操作如何在具有 relu 激活的 Conv2D 层中发生。 在本课中，我们将查看此序列中的第三个（也是最后一个）操作：使用最大池化进行压缩，在 Keras 中由 MaxPool2D 层完成。

## Condense with Maximum Pooling

将condense步骤添加到我们之前的模型中，将为我们提供：

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3), # activation is None
    layers.MaxPool2D(pool_size=2),
    # More layers follow
])
```

[【转载】Keras.layers.Conv2D参数详解 搭建图片分类 CNN （卷积神经网络）](https://blog.csdn.net/Checkmate9949/article/details/119609758)

### 导入keras

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Flatten,Dense
```

### Conv2D

构建卷积层。用于从输入的高维数组中提取特征。卷积层的每个过滤器就是一个特征映射，用于提取某一个特征，过滤器的数量决定了卷积层输出特征个数，或者输出深度。因此，图片等高维数据每经过一个卷积层，深度都会增加，并且等于过滤器的数量。

![Keras.layers.Conv2D 搭建图片分类 CNN （卷积神经网络）](https://img-blog.csdnimg.cn/img_convert/b9eacc71b591a3aaf86d2696448be5f6.gif)

Conv2D(filters, kernel_size, strides, padding, activation=‘relu’, input_shape)

> - filters: 过滤器数量，输出的神经元
> - kernel_size: 指定（方形）卷积窗口的高和宽的数字
> - strides: 卷积步长, 默认为 1
> - padding: 卷积如何处理边缘。选项包括 ‘valid’ 和 ‘same’。默认为 ‘valid’
> - activation: 激活函数，通常设为 relu。如果未指定任何值，则不应用任何激活函数。强烈建议你向网络中的每个卷积层添加一个 ReLU 激活函数。
> - input_shape: 指定输入层的高度，宽度和深度的元组。当卷积层作为模型第一层时，必须提供此参数，否则不需要。
>   

```python
# first layer
Conv2D(filters=16, kernel_size=2, strides=2, activation=‘relu’, input_shape=(200,200,1))
# second layer
Conv2D(filters=32, kernel_size=2, padding='same', activation=‘relu’)
# last layer
Conv2D(64,(2,2), activation=‘relu’)
```

### MaxPooling2D

![image-20221105142325509](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105142325509.png)

构建最大池化层。如果说，卷积层通过过滤器从高维数据中提取特征，增加了输出的深度（特征数），那么，最大池化层的作用是降低输出维度（宽高）。在 CNN 架构中，最大池化层通常出现在卷积层后，后面接着下一个卷积层，交替出现，结果是，输入的高维数组，深度逐次增加，而维度逐次降低。最终，高维的空间信息，逐渐转换成 1 维的特征向量，然后连接全联接层或其他分类算法，得到模型输出。

**MaxPooling2D(pool_size, strides, padding)**

> - pool_size: 指定池化窗口高度和宽度的数字
> - strides: 垂直和水平 stride。如果不指定任何值，则 strides 默认为 pool_size
> - padding: 选项包括 `valid` 和 `same`。如果不指定任何值，则 padding 设为 `valid`

```python
MaxPooling2D(pool_size=2, strides=2)
```

MaxPool2D 层很像 Conv2D 层，不同之处在于它使用简单的**最大函数**而不是内核，pool_size 参数类似于 kernel_size。然而，MaxPool2D 层不像卷积层在其内核中那样具有任何可训练的权重。 --**没有权重**

让我们再看一下上一课的提取图。请记住 MaxPool2D 是 Condense 步骤。

![image-20221102170401193](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102170401193.png)

请注意，在应用 ReLU 函数（检测）后，特征图最终会出现很多“死区”，即仅包含 0 的大片区域（图像中的黑色区域）。在整个网络中携带这些0的激活会增加模型的大小，但不会添加太多有用的信息。相反，我们希望压缩特征图以仅保留最有用的部分——特征本身。 

这实际上就是最大池化所做的。最大池化在原始特征图中获取一个激活补丁，并用该补丁中的最大激活替换它们。

![image-20221102170812000](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102170812000.png)

在 ReLU 激活后应用时，它具有“强化”特征的效果。池化步骤将活动像素的比例增加到零像素。



## Example - Apply Maximum Pooling

让我们在第 2 课示例中所做的特征提取中添加“浓缩”步骤。下一个隐藏单元将带我们回到我们离开的地方。

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # to clean up output cells

# Read image
image_path = '../input/computer-vision-resources/car_feature.jpg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image)

# Define kernel
kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
], dtype=tf.float32)

# Reformat for batch compatibility.
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])

# Filter step
image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    # we'll talk about these two in the next lesson!
    strides=1,
    padding='SAME'
)

# Detect step
image_detect = tf.nn.relu(image_filter)

# Show what we have so far
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(tf.squeeze(image), cmap='gray')
plt.axis('off')
plt.title('Input')
plt.subplot(132)
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.title('Filter')
plt.subplot(133)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title('Detect')
plt.show();
```

![image-20221102171005768](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102171005768.png)

我们将使用 tf.nn 中的另一个函数来应用池化步骤 tf.nn.pool。这是一个 Python 函数，它与您在模型构建时使用的 MaxPool2D 层做同样的事情，但是，作为一个简单的函数，更容易直接使用。

```python
import tensorflow as tf

image_condense = tf.nn.pool(
    input=image_detect, # image in the Detect step above
    window_shape=(2, 2),
    pooling_type='MAX',
    # we'll see what these do in the next lesson!
    strides=(2, 2),
    padding='SAME',
)

plt.figure(figsize=(6, 6))
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.show();
```

![image-20221102171051878](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102171051878.png)

很酷！希望您可以看到池化步骤如何通过将图像集中在最活跃的像素周围来增强特征。

### 平移不变性(Translation Invariance)

- 最大池化层会损失一部分位置信息，使得对**位置信息不太依赖的特征识别**有帮助

我们称零像素为“不重要的”。这是否意味着它们根本不携带任何信息？事实上，**零像素携带位置信息**。空白区域仍将特征定位在图像内。

当 MaxPool2D 移除其中一些像素时，它会<u>**移除特征图中的一些位置信息**</u>。这为卷积网络提供了一个称为平移不变性的属性。这意味着具有最大池化的卷积网络往往不会通过图像在图像中的位置来区分特征。 （“平移”是一个数学词，用于在不旋转或改变其形状或大小的情况下改变某物的位置。） 

观察当我们反复将最大池化应用于以下特征图时会发生什么。

![image-20221102171217101](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102171217101.png)

原始图像中的两个点在重复池化后变得无法区分。换句话说，池化破坏了他们的一些位置信息。由于网络无法再在特征图中区分它们，因此它也无法在原始图像中区分它们：它已变得不受位置差异的影响。 

事实上，池化只会在网络中**创建小距离的平移不变性**，就像图像中的两个点一样。开始相距很远的特征在合并后将保持不同；只有**一些位置信息丢失了，但不是全部丢失**。

![image-20221102171242580](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102171242580.png)

这种对特征位置微小差异的不变性对于图像分类器来说是一个很好的属性。仅仅因为透视或取景的不同，<u>**同一种特征可能位于原始图像的不同部分，但我们仍然希望分类器能够识别它们是相同的**</u>。

因为这种不变性是内置在网络中的，所以我们可以使用更少的数据进行训练：我们不再需要教它忽略这种差异。与只有密集层的网络相比，这使卷积网络具有很大的效率优势。 （您将在第 6 课的数据增强中看到另一种免费获得不变性的方法！）

### 结论 

在本课中，我们学习了特征提取的最后一步：使用 MaxPool2D 进行压缩。在第 4 课中，我们将完成对滑动窗口的卷积和池化的讨论。

## Exercise: Maximum Pooling

### 介绍

在这些练习中，您将总结练习 2 中开始的特征提取，探索最大池化如何创建不变性，然后了解另一种池化：平均池化。 运行下面的单元格以设置所有内容。

### 环境设置

```py
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.computer_vision.ex3 import *

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import learntools.computer_vision.visiontools as visiontools

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
```

### predefined kernel

```python
# Read image
image_path = '../input/computer-vision-resources/car_illus.jpg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[400, 400])

# Embossing kernel
kernel = tf.constant([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2],
])

# Reformat for batch compatibility.
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)

image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=1,
    padding='VALID',
)

image_detect = tf.nn.relu(image_filter)

# Show what we have so far
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(tf.squeeze(image), cmap='gray')
plt.axis('off')
plt.title('Input')
plt.subplot(132)
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.title('Filter')
plt.subplot(133)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title('Detect')
plt.show();
```

### 1）添加池化层

对于序列中的最后一步，使用 2×2 池化窗口应用最大池化

```
image_condense = tf.nn.pool(
    input=image_detect,
    window_shape=____,
    pooling_type=____,
    strides=(2, 2),
    padding='SAME',
)
```

```python
image_condense = tf.nn.pool(
    input=image_detect,
    window_shape=(2,2),
    pooling_type="MAX",
    strides=(2,2),
    padding='SAME',
)
```

```python
plt.figure(figsize=(8, 6))
plt.subplot(121)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title("Detect (ReLU)")
plt.subplot(122)
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.title("Condense (MaxPool)")
plt.show();
```

![image-20221105115331501](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105115331501.png)

`maxpooling`增加了图像的明暗对比度

我们了解了 MaxPool2D 层如何赋予卷积网络在小距离上平移不变性的特性。在本练习中，您将有机会观察到这一点。 

下一个代码单元将随机对一个圆圈应用一个小位移，然后使用最大池化数次压缩图像。运行一次单元格并记下最后生成的图像。

```python
REPEATS = 4
SIZE = [64, 64]

# Create a randomly shifted circle
image = visiontools.circle(SIZE, r_shrink=4, val=1)
image = tf.expand_dims(image, axis=-1)
image = visiontools.random_transform(image, jitter=3, fill_method='replicate')
image = tf.squeeze(image)

plt.figure(figsize=(16, 4))
plt.subplot(1, REPEATS+1, 1)
plt.imshow(image, vmin=0, vmax=1)
plt.title("Original\nShape: {}x{}".format(image.shape[0], image.shape[1]))
plt.axis('off')

# Now condense with maximum pooling several times
for i in range(REPEATS):
    ax = plt.subplot(1, REPEATS+1, i+2)
    image = tf.reshape(image, [1, *image.shape, 1])
    image = tf.nn.pool(image, window_shape=(2,2), strides=(2, 2), padding='SAME', pooling_type='MAX')
    image = tf.squeeze(image)
    plt.imshow(image, vmin=0, vmax=1)
    plt.title("MaxPool {}\nShape: {}x{}".format(i+1, image.shape[0], image.shape[1]))
    plt.axis('off')
```

![image-20221105115649122](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105115649122.png)

### 2）探索不变性（Explore Invariance）

假设您在不同的方向上做了一个小的转变——您期望这对结果图像有什么影响？如果愿意，请尝试再运行几次单元格，以获得新的随机班次。

```python
REPEATS = 5
SIZE = [128, 128]

# Create a randomly shifted circle
image = visiontools.circle(SIZE, r_shrink=4, val=1)
image = tf.expand_dims(image, axis=-1)
image = visiontools.random_transform(image, jitter=3, fill_method='replicate')
image = tf.squeeze(image)

plt.figure(figsize=(16, 4))
plt.subplot(1, REPEATS+1, 1)
plt.imshow(image, vmin=0, vmax=1)
plt.title("Original\nShape: {}x{}".format(image.shape[0], image.shape[1]))
plt.axis('off')

# Now condense with maximum pooling several times
for i in range(REPEATS):
    ax = plt.subplot(1, REPEATS+1, i+2)
    image = tf.reshape(image, [1, *image.shape, 1])
    image = tf.nn.pool(image, window_shape=(2,2), strides=(2, 2), padding='SAME', pooling_type='MAX')
    image = tf.squeeze(image)
    plt.imshow(image, vmin=0, vmax=1)
    plt.title("MaxPool {}\nShape: {}x{}".format(i+1, image.shape[0], image.shape[1]))
    plt.axis('off')
```

![image-20221105120407694](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105120407694.png)

在本教程中，我们讨论了最大池化如何在小距离上创建平移不变性。这意味着我们预计在重复最大池化后小的变化会消失。如果您多次运行单元格，您可以看到生成的图像始终相同；池化操作会破坏那些小的翻译。

### 全局平均池化（Global Average Pooling）

我们在之前的练习中提到，平均池化在很大程度上已被卷积基内的最大池化所取代。然而，有一种平均池化仍然广泛用于卷积网络的头部。这是全球平均池化。 GlobalAvgPool2D 层通常用作网络头部的部分或全部隐藏 Dense 层的替代，如下所示：

```
model = keras.Sequential([
    pretrained_base,
    layers.GlobalAvgPool2D(),
    layers.Dense(1, activation='sigmoid'),
])
```

这个层在做什么？请注意，我们不再有通常位于基础之后的 Flatten 层，用于将 2D 特征数据转换为分类器所需的 1D 数据。现在 GlobalAvgPool2D 层正在提供这个功能。但是，它不是“分解（unstacking）”特征（如 Flatten），而是简单地用其**平均值替换整个特征图**。虽然非常具有破坏性，但它通常效果很好，并且具有**减少模型中参数数量的优势**。 让我们看看 GlobalAvgPool2D 在一些随机生成的特征图上做了什么。这将帮助我们理解它如何“扁平化（flatten）”由基础生成的特征图堆栈。

```python
feature_maps = [visiontools.random_map([5, 5], scale=0.1, decay_power=4) for _ in range(8)]

gs = gridspec.GridSpec(1, 8, wspace=0.01, hspace=0.01)
plt.figure(figsize=(18, 2))
for i, feature_map in enumerate(feature_maps):
    plt.subplot(gs[i])
    plt.imshow(feature_map, vmin=0, vmax=1)
    plt.axis('off')
plt.suptitle('Feature Maps', size=18, weight='bold', y=1.1)
plt.show()

# reformat for TensorFlow
feature_maps_tf = [tf.reshape(feature_map, [1, *feature_map.shape, 1])
                   for feature_map in feature_maps]

global_avg_pool = tf.keras.layers.GlobalAvgPool2D()
pooled_maps = [global_avg_pool(feature_map) for feature_map in feature_maps_tf]
img = np.array(pooled_maps)[:,:,0].T

plt.imshow(img, vmin=0, vmax=1)
plt.axis('off')
plt.title('Pooled Feature Maps')
plt.show();
```

![image-20221105121936219](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105121936219.png)

8个特征（5*5）用各自的全局平均值（1）替代，输出一列8个平均值的数列。

由于每个 5×5 特征图都被缩减为一个值，全局池化将表示这些特征所需的参数数量减少了 25 倍——节省了大量资金！

现在我们将继续了解池化特征。 

在我们将特征汇集到一个单一的值之后，头部是否还有足够的信息来确定一个类别？这部分练习将调查这个问题。 让我们通过 VGG16 从我们的 Car 或 Truck 数据集中传递一些图像，并检查池化后产生的特征。首先运行此单元以定义模型并加载数据集。

```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Load VGG16
pretrained_base = tf.keras.models.load_model(
    '../input/cv-course-models/cv-course-models/vgg16-pretrained-base',
)

model = keras.Sequential([
    pretrained_base,
    # Attach a global average pooling layer after the base
    layers.GlobalAvgPool2D(),
])

# Load dataset
ds = image_dataset_from_directory(
    '../input/car-or-truck/train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=1,
    shuffle=True,
)

ds_iter = iter(ds)
```

请注意我们如何在预训练的 VGG16 基础之后附加 GlobalAvgPool2D 层。通常，VGG16 将为每张图像生成 512 个特征图。如果您愿意，GlobalAvgPool2D 层会将这些中的每一个减少为一个值，即“平均像素”。 

下一个单元格将通过 VGG16 运行来自 Car 或 Truck 数据集的图像，并显示由 GlobalAvgPool2D 创建的 512 个平均像素。运行该单元几次，观察汽车产生的像素与卡车产生的像素。

```python
car = next(ds_iter)

car_tf = tf.image.resize(car[0], size=[128, 128])
car_features = model(car_tf)
car_features = tf.reshape(car_features, shape=(16, 32))
label = int(tf.squeeze(car[1]).numpy())

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.imshow(tf.squeeze(car[0]))
plt.axis('off')
plt.title(["Car", "Truck"][label])
plt.subplot(122)
plt.imshow(car_features)
plt.title('Pooled Feature Maps')
plt.axis('off')
plt.show();
```

![image-20221105131149707](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105131149707.png)

![image-20221105131211986](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105131211986.png)

![image-20221105131241829](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105131241829.png)

![image-20221105131315691](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105131315691.png)

### 3) 了解池化特征

你看到了什么？汽车和卡车的汇集特征是否足以区分它们？您将如何解释这些汇总值？这对分类有何帮助？在您考虑之后，运行下一个单元格以获得答案。 （或者先看看提示！）

VGG16 基础生成 512 个特征图。我们可以将每个特征图视为代表原始图像中的一些高级视觉特征——可能是轮子或窗口。池化地图给了我们一个单一的数字，我们可以将其视为该特征的分数：如果特征存在则大，如果不存在则小。汽车往往在一组功能上得分高，而卡车在另一组功能上得分高。现在，头部不必尝试将原始特征映射到类，而只需使用 GlobalAvgPool2D 产生的这些分数，这是一个更容易解决的问题。

提示：VGG16 从图像中创建 512 个特征图，这些特征图可能代表像轮子或窗户之类的东西。 Pooled Feature Maps 中的每个正方形代表一个特征。一个特性的大值意味着什么？

全局平均池通常用于现代卷积网络。一个很大的优势是它极大地减少了模型中的参数数量，同时仍然告诉你图像中是否存在某些特征——这对于分类来说通常是最重要的。如果您正在创建卷积分类器，那么值得一试！

### 结论 

在本课中，我们探讨了特征提取过程中的最终操作：使用最大池化进行压缩。池化是卷积网络的基本特征之一，有助于为它们提供一些特有的优势：视觉数据的效率、与密集网络相比减小的参数大小、平移不变性。我们已经看到，它不仅在特征提取时用在base中，而且在分类时也可以用在head中。理解它对于全面理解卷积网络至关重要。
