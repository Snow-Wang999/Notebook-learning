# Convolution and `ReLU`

## show_kernel function

```python
import numpy as np
from itertools import product

def show_kernel(kernel, label=True, digits=None, text_size=28):
    # Format kernel
    kernel = np.array(kernel)
    if digits is not None:
        kernel = kernel.round(digits)

    # Plot kernel
    cmap = plt.get_cmap('Blues_r')
    plt.imshow(kernel, cmap=cmap)
    rows, cols = kernel.shape
    thresh = (kernel.max()+kernel.min())/2
    # Optionally, add value labels
    if label:
        for i, j in product(range(rows), range(cols)):
            val = kernel[i, j]
            color = cmap(0) if val > thresh else cmap(255)
            plt.text(j, i, val, 
                     color=color, size=text_size,
                     horizontalalignment='center', verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
```

## 笛卡儿积

itertools.product：类似于求多个可迭代对象的笛卡尔积。

```python
import itertools
aa = itertoolls.product(['天使','恶魔','人类'],['受到伤害','医治病患'])
bb = list(aa)
print(bb)
```

```python
output =
[
    ('天使','受到伤害'),
    ('天使','医治病患'),
    ('恶魔','受到伤害'),
    ('恶魔','医治病患'),
    ('人类','受到伤害'),
    ('人类','医治病患')
]
```

3*2

```python
aa = itertoolls.product(['天使','恶魔','人类'],['受到伤害','医治病患'],repeat=3)
# aa = itertoolls.product(output,output,output)
# 和上面等同
bb = list(aa)
print(bb)
```

list的长度是6\*6\*6=216

生成随机坐标

```python
# one way
random_list = list(itertoolls.product(range(1,4),range(1,2))
# another way
n =2
aa = random.sample(random_list, n)   
```

## 介绍

在上一课中，我们看到卷积分类器有两部分：

卷积基（a convolutional **base**）和密集层的头部（a **head** of dense layers）。 

- base：从图像中提取视觉特征

- head：使用这些特征对图像进行分类 

在接下来的几节课中，我们将了解您通常会在卷积图像分类器的基础中找到的两种最重要的层类型。

- **convolutional layer** with **ReLU activation**：具有 ReLU 激活的卷积层
- **maximum pooling layer**：最大池化层。

在第 5 课中，您将学习如何通过将这些层组合成执行特征提取的块来设计自己的卷积网络。 

本课是关于带有 ReLU 激活函数的卷积层。

## 特征提取（Feature Extraction）

在我们深入了解卷积的细节之前，让我们讨论一下这些层在网络中的用途。我们将了解如何使用这三个操作（卷积、ReLU 和最大池化）来实现特征提取过程。

基础执行的特征提取包括三个基本操作： 

- 针对特定特征**过滤（filter）**图像（convolution） 
- condense在过滤后的图像中**检测（detect）**该特征 (`ReLU`) 
- **压缩（condense）**图像以增强特征（maximum pooling）

下图说明了这个过程。您可以看到这三个操作如何能够隔离原始图像的某些特定特征（在本例中为水平线）。

![image-20221102122634581](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102122634581.png)

![image-20221102122655914](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102122655914.png)

通常，网络将在单个图像上并行执行多个提取。在现代卷积网络中，基础的最后一层产生超过 1000 个独特的视觉特征的情况并不少见。

## 卷积过滤

卷积层执行过滤步骤。您可以在 Keras 模型中定义一个卷积层，如下所示：

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
	layers.Conv2D(filters=64, kernel_size=3), # activation is None
    # more layers follow
)
```

我们可以通过查看它们与层的权重和激活的关系来理解这些参数。现在让我们这样做。

### 权重（weights）

卷积网络在训练期间学习的权重主要包含在其卷积层中。这些权重我们称为内核（kernels）。我们可以将它们表示为小数组：

```python
[
    -1,2,-1,
    -1,2,-1,
    -1,2,-1,
]
```

内核通过扫描图像并产生**像素值的加权和**来进行操作。通过这种方式，内核的行为有点像偏光镜片，强调或不强调某些信息模式。

![image-20221102132650183](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102132650183.png)

内核定义了卷积层如何连接到后面的层。上面的内核将输出中的每个神经元连接到输入中的九个神经元。通过使用 kernel_size 设置内核的尺寸，您可以告诉 convnet 如何形成这些连接。大多数情况下，内核将具有奇数维度——如 kernel_size=(3, 3) 或 (5, 5)——因此单个像素位于中心，但这不是必需的。 

卷积层中的内核决定了它创建了什么样的特征。在训练期间，卷积网络试图了解解决分类问题所需的特征。这意味着为其内核找到最佳值。

### 激活(activation)

网络中的激活我们称为特征图(**feature maps**)。它们是我们对图像应用滤镜(**filter**)时的结果；它们包含内核提取的视觉特征。下面是一些带有他们生成的特征图的内核。

![image-20221102133130881](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102133130881.png)

从内核中的数字模式，您可以分辨出它创建的特征图的种类。通常，卷积在其输入中强调的内容将与内核中正数的形状相匹配。上面的左侧和中间内核都将过滤水平形状。 

使用过滤器参数，您可以告诉卷积层您希望它创建多少个特征图作为输出。

### 使用 ReLU 检测(Detect with ReLU)

过滤后，特征图通过激活函数。 rectifier 函数有一个如下图：

![image-20221102133438124](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102133438124.png)

带有整流器的神经元称为整流线性单元。出于这个原因，我们也可以将整流函数称为 ReLU 激活函数，甚至称为 ReLU 函数。 

ReLU 激活可以在其自己的激活层中定义，但大多数情况下，您只需将其作为 Conv2D 的激活函数。

```python
model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3, activation='relu')
    # More layers follow
])
```

您可以将激活函数视为根据某种重要性度量对像素值进行评分(scoring pixel values according to some measure of importance)。 ReLU 激活表示负值不重要，因此将它们设置为 0。（“所有不重要的东西同样不重要。”） 

这是 ReLU 应用了上面的特征图。注意它是如何成功隔离特征的。

![image-20221102133720602](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102133720602.png)

与其他激活函数一样，ReLU 函数是非线性的。从本质上讲，这意味着网络中所有层的总效果与通过将效果相加得到的效果不同——这与仅使用单个层可以实现的效果相同。非线性确保特征在深入网络时以有趣的方式组合。 （我们将在第 5 课中进一步探讨这种“特征复合(**feature compounding**)”。）

## Example - Apply Convolution and ReLU

在这个例子中，我们将自己进行提取，以更好地了解卷积网络在“幕后”做什么。 这是我们将用于此示例的图像：

```python
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')

image_path = '../input/computer-vision-resources/car_feature.jpg'
image = tf.io.read_file(image_path)
# tf.io.read_file 返回值是二进制格式
image = tf.io.decode_jpeg(image)
# tf.io.decode_jpeg 读取二进制文件，然后channels是输出的图片的通道数，3是rpb三个通道，1是灰度图片，ratio是图片缩放比例
#输出的image的type是tensorflow特别的tensor形式（EagerTensor）,是uint8类型，0-255
'''
image = tf.image.resize(image,[256,256]) # 统一图片大小
image = tf.cast(image,tf.float32) # 转换类型
image = image/255 # 归一化，标准化，float32
'''

plt.figure(figsize=(6, 6))
plt.imshow(tf.squeeze(image), cmap='gray')
plt.axis('off')
plt.show();
```

![image-20221102141748097](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102141748097.png)

### TF构建数据集

```python
def read_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3, ratio=1)
    image = tf.image.resize(image, [256, 256])  # 统一图片大小
    image = tf.cast(image, tf.float32)  # 转换类型
    image = image / 255  # 归一化
    return image

images = tf.io.gfile.glob('./*.jpeg')
#glob获取文件list
dataset = tf.data.Dataset.from_tensor_slices(images)
#返回tensorflow的dataset类型，可理解为一个可迭代的list
AUTOTUNE = tf.data.experimental.AUTOTUNE
#根据cpu的情况，自动判断多线程的数量
dataset = dataset.map(read_image,num_parallel_calls=AUTOTUNE)
#.map实现定义好的函数，对处理dataset中每一个元素，在上面代码中是把路径的字符串变成该路径读取的图片张量，对图片的预处理应该也在这部分进行吧
dataset = dataset.shuffle(1).batch(1)
#dataset.shuffle就是乱序，.batch就是组装batch袋
for a in dataset.take(2):
    print(a.shape)
#.take(num)获取dataset中的元素，num是从dataset中取出来的batch的数量
```

### 构建kernel

对于过滤步骤，我们将定义一个内核，然后将其与卷积一起应用。这种情况下的内核是“边缘检测(edge detection)”内核。你可以用 `tf.constant` 定义它，就像你在 `Numpy` 中用 `np.array` 定义一个数组一样。这会创建一个 `TensorFlow` 使用的张量。

```python
import tensorflow as tf

kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
])

plt.figure(figsize=(3, 3))
show_kernel(kernel)
```

![image-20221102140453810](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102140453810.png)



TensorFlow 在其 tf.nn 模块中包含许多由神经网络执行的常见操作。我们将使用的两个是 conv2d 和 relu。这些只是 Keras 层的简单功能版本。 

下一个隐藏单元格会进行一些重新格式化以使内容与 TensorFlow 兼容。对于这个例子，细节并不重要。

### 批处理兼容性

```python
# Reformat for batch compatibility.
# 重新格式化以实现批处理兼容性。
image = tf.image.convert_image_dtype(image, dtype=tf.float32)#转换image类型
image = tf.expand_dims(image, axis=0)
#给函数增加维度，axis=0是在第一个维度上增加一个维度，从[10,10,3]变成[1,10,10,3]
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])#重塑kernel的大小
kernel = tf.cast(kernel, dtype=tf.float32)#转换kernel类型
```

### 应用kernel

```python
image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    # we'll talk about these two in lesson 4!
    strides=1,
    padding='SAME',
)

plt.figure(figsize=(6, 6))
plt.imshow(tf.squeeze(image_filter))
#tf.squeeze()函数用于从张量形状中移除大小为1的维度
#eg:[1,3,2,1,2]变成[3,2,2]
plt.axis('off')
plt.show();
```

![image-20221102141530378](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102141530378.png)

### ReLU检测

```python
image_detect = tf.nn.relu(image_filter)

plt.figure(figsize=(6, 6))
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.show();
```

![image-20221102141631663](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102141631663.png)

现在我们已经创建了一个特征图！像这样的图像是头部用来解决其分类问题的。我们可以想象，某些特征可能更具有 Cars 的特征，而其他特征可能更具有 Trucks 的特征。训练期间卷积网络的任务是创建可以找到这些特征的内核。

### 结论 

我们在本课中看到了 convnet 用于执行特征提取的前两个步骤：使用 Conv2D 层进行过滤和使用 relu 激活进行检测。

***

## Exercise: Convolution and `ReLU`

### 介绍

在本练习中，您将围绕特征提取建立一些直觉。首先，我们将再次介绍我们在教程中所做的示例，但这一次，内核由您自己选择。在本课程中，我们主要使用图像，但我们正在学习的所有操作背后都是数学。因此，我们还将看看如何将这些特征图表示为数字数组，以及使用内核的卷积将对它们产生什么影响。 

```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.computer_vision.ex2 import *

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')

tf.config.run_functions_eagerly(True)
#eager execution（饥饿执行）,立即执行每一步代码，非常的饥渴。就像搞一夜情，认识后就立即“执行
```

### 应用转换

接下来的几个练习将介绍特征提取。

```python
image_path = '../input/computer-vision-resources/car_illus.jpg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[400, 400])

img = tf.squeeze(image).numpy()
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show();
```

![image-20221102144602362](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102144602362.png)

您可以运行此单元来查看图像处理中使用的一些标准内核。

```python
import learntools.computer_vision.visiontools as visiontools
from learntools.computer_vision.visiontools import edge, bottom_sobel, emboss, sharpen

kernels = [edge, bottom_sobel, emboss, sharpen]
names = ["Edge Detect", "Bottom Sobel", "Emboss", "Sharpen"]

plt.figure(figsize=(12, 12))
for i, (kernel, name) in enumerate(zip(kernels, names)):
    plt.subplot(1, 4, i+1)
    visiontools.show_kernel(kernel)
    plt.title(name)
plt.tight_layout()
```

![image-20221102144640994](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102144640994.png)

### 1. 定义内核（Define kernel）

使用下一个代码单元定义内核。您可以选择应用哪种内核。要记住的一件事是内核中数字的总和决定了最终图像的亮度。通常，您应该尝试将数字的总和保持在 0 和 1 之间（尽管这不是正确答案所必需的）。 

一般来说，一个内核可以有任意数量的行和列。对于这个练习，让我们使用一个 3×3的内核，它通常会给出最好的结果。用 tf.constant 定义一个内核。

```python
# YOUR CODE HERE: Define a kernel with 3 rows and 3 columns.
kernel = tf.constant([
    [-1,2,-1],
    [2,8,2],
    [-1,2,-1]
])
# Uncomment to view kernel
# visiontools.show_kernel(kernel)
```

![image-20221102144921701](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102144921701.png)

```python
# Reformat for batch compatibility.
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)
```

### 2. 应用卷积

现在我们将通过卷积将内核应用于图像。 Keras 中执行此操作的层是 layers.Conv2D。 TensorFlow 中执行相同操作的后端函数是什么？

```python
# YOUR CODE HERE: Give the TensorFlow convolution function (without arguments)
conv_fn = tf.nn.conv2d
```

#### 执行卷积

```python
image_filter = conv_fn(
    input=image,
    filters=kernel,
    strides=1, # or (1, 1)
    padding='SAME',
)

plt.imshow(
    # Reformat for plotting
    tf.squeeze(image_filter)
)
plt.axis('off')
plt.show();
```

![image-20221102150115321](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102150115321.png)

### 3. 应用ReLU

现在使用 ReLU 函数检测特征。在 Keras 中，您通常会将其用作 Conv2D 层中的激活函数。 TensorFlow 中做同样事情的后端函数是什么？

```python
# YOUR CODE HERE: Give the TensorFlow ReLU function (without arguments)
relu_fn = tf.nn.relu
```

找到解决方案后，运行此单元以使用 ReLU 检测特征并查看结果！ 

您在下面看到的图像是您选择的内核生成的特征图。如果您愿意，可以尝试上面推荐的其他一些内核，或者尝试发明一个能够提取某种特征的内核。

```python
image_detect = relu_fn(image_filter)

plt.imshow(
    # Reformat for plotting
    tf.squeeze(image_detect)
)
plt.axis('off')
plt.show();
```

![image-20221102150257892](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102150257892.png)

在本教程中，我们对内核和特征图的讨论主要是视觉的。我们通过观察它们如何转换一些示例图像来了解 Conv2D 和 ReLU 的效果。 

但是卷积网络中的操作（就像在所有神经网络中一样）通常是通过数学函数，通过对数字的计算来定义的。在下一个练习中，我们将花一点时间来探索这个观点。 

让我们首先定义一个简单的数组作为图像，以及另一个数组作为内核。运行以下单元格以查看这些数组。

```python
# Sympy is a python library for symbolic mathematics. It has a nice
# pretty printer for matrices, which is all we'll use it for.
import sympy
sympy.init_printing()
from IPython.display import display

image = np.array([
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 0],
])

kernel = np.array([
    [1, -1],
    [1, -1],
])

display(sympy.Matrix(image))
display(sympy.Matrix(kernel))
# Reformat for Tensorflow
image = tf.cast(image, dtype=tf.float32)
image = tf.reshape(image, [1, *image.shape, 1])
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)
```

![image-20221102150400906](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102150400906.png)

### 4. 观察数值矩阵上的卷积

你看到了什么？图像只是左侧的一条长垂直线和右下方的一条短水平线。内核呢？你认为它会对这张图片产生什么影响？在您考虑之后，运行下一个单元格以获得答案。

```python
image_filter = conv_fn(
    input=image,
    filters=kernel,
    strides=1, # or (1, 1)
    padding='SAME',
)
relu_fn = tf.nn.relu
image_detect = relu_fn(image_filter)

plt.imshow(
    # Reformat for plotting
    tf.squeeze(image_detect)
)
plt.axis('off')
plt.show();
```

![image-20221102150944009](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102150944009.png)

在本教程中，我们讨论了正数模式如何告诉您内核将提取的特征类型。这个内核有一个 1 的垂直列，所以我们希望它返回垂直线的特征。

现在让我们试一试。运行下一个单元格以将卷积和 ReLU 应用于图像并显示结果。

```python
image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=1,
    padding='VALID',
)
image_detect = tf.nn.relu(image_filter)

# The first matrix is the image after convolution, and the second is
# the image after ReLU.
display(sympy.Matrix(tf.squeeze(image_filter).numpy()))
display(sympy.Matrix(tf.squeeze(image_detect).numpy()))
```

![image-20221102151032796](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221102151032796.png)

### 结论 

在本课中，您了解了卷积分类器用于特征提取的前两个操作：使用卷积过滤图像并使用校正线性单元( **rectified linear unit**-ReLU)检测特征。