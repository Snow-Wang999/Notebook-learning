# 14-The Sliding Window（滑动窗口）

## 自定义函数

```python
import numpy as np
from itertools import product
from skimage import draw, transform

#创建圆形图
def circle(size, val=None, r_shrink=0):
    circle = np.zeros([size[0]+1, size[1]+1])
    rr, cc = draw.circle_perimeter(
        size[0]//2, size[1]//2,
        radius=size[0]//2 - r_shrink,
        shape=[size[0]+1, size[1]+1],
    )
    if val is None:
        circle[rr, cc] = np.random.uniform(size=circle.shape)[rr, cc]
    else:
        circle[rr, cc] = val
    circle = transform.resize(circle, size, order=0)
    return circle

#展示kernel
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

#展示特征提取
def show_extraction(image,
                    kernel,
                    conv_stride=1,
                    conv_padding='valid',
                    activation='relu',
                    pool_size=2,
                    pool_stride=2,
                    pool_padding='same',
                    figsize=(10, 10),
                    subplot_shape=(2, 2),
                    ops=['Input', 'Filter', 'Detect', 'Condense'],
                    gamma=1.0):
    # Create Layers
    model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(
                        filters=1,
                        kernel_size=kernel.shape,
                        strides=conv_stride,
                        padding=conv_padding,
                        use_bias=False,
                        input_shape=image.shape,
                    ),
                    tf.keras.layers.Activation(activation),
                    tf.keras.layers.MaxPool2D(
                        pool_size=pool_size,
                        strides=pool_stride,
                        padding=pool_padding,
                    ),
                   ])

    layer_filter, layer_detect, layer_condense = model.layers
    kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
    layer_filter.set_weights([kernel])

    # Format for TF
    image = tf.expand_dims(image, axis=0)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) 
    
    # Extract Feature
    image_filter = layer_filter(image)
    image_detect = layer_detect(image_filter)
    image_condense = layer_condense(image_detect)
    
    images = {}
    if 'Input' in ops:
        images.update({'Input': (image, 1.0)})
    if 'Filter' in ops:
        images.update({'Filter': (image_filter, 1.0)})
    if 'Detect' in ops:
        images.update({'Detect': (image_detect, gamma)})
    if 'Condense' in ops:
        images.update({'Condense': (image_condense, gamma)})
    
    # Plot
    plt.figure(figsize=figsize)
    for i, title in enumerate(ops):
        image, gamma = images[title]
        plt.subplot(*subplot_shape, i+1)
        plt.imshow(tf.image.adjust_gamma(tf.squeeze(image), gamma))
        plt.axis('off')
        plt.title(title)
```

## 介绍

在前两节课中，我们了解了从图像中进行特征提取的三个操作： 

1. 带有卷积层(convolution)的过滤器 

2. 使用 ReLU 激活检测 

3. 用最大池化层(maximum pooling)凝聚

卷积和池化操作有一个共同的特点：它们都是在一个滑动窗口上执行的。使用卷积，这个“窗口”由内核的维度、参数 kernel_size 给出。对于池化，它是池化窗口，由 pool_size 给出。

![image-20221105143424948](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105143424948.png)

![A 2D sliding window.](https://i.imgur.com/LueNK6b.gif)

还有两个额外的参数会影响卷积层和池化层——这些是窗口的步幅以及是否在图像边缘使用填充。 strides 参数表示窗口应该在每一步移动多远，而 padding 参数描述我们如何处理输入边缘的像素。 有了这两个参数，定义两个层就变成了：

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  activation='relu'),
    layers.MaxPool2D(pool_size=2,
                     strides=1,
                     padding='same')
    # More layers follow
])
```

## Stride（步长）

窗口在每一步移动的距离称为步幅。我们需要在图像的两个维度上指定步幅：一个用于从左到右移动，一个用于从上到下移动。此动画显示 strides=(2, 2)，每步移动 2 个像素。

![image-20221105143441506](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105143441506.png)

![Sliding window with a stride of (2, 2).](https://i.imgur.com/Tlptsvt.gif)

步幅有什么作用？每当任一方向的步幅大于 1 时，滑动窗口将在每一步跳过输入中的一些像素。 

因为我们希望使用高质量的特征进行分类，所以卷积层通常具有 strides=(1, 1)。增加步幅意味着我们错过了摘要中可能有价值的信息。然而，最大池化层几乎总是具有大于 1 的步幅值，例如 (2, 2) 或 (3, 3)，但不会大于窗口本身。 

最后需要注意的是，当strides的值在两个方向都是相同的数字时，只需要设置那个数字即可；例如，您可以使用 strides=2 代替 strides=(2, 2) 进行参数设置。

## Padding（边缘填充）

在执行滑动窗口计算时，存在一个关于在输入边界处做什么的问题。完全停留在输入图像内意味着窗口永远不会像输入中的每个其他像素那样直接位于这些边界像素上。由于我们没有对所有像素进行完全相同的处理，会不会有问题？ 

卷积对这些边界值的作用取决于其填充参数。在 TensorFlow 中，您有两种选择：

- padding='same' 

  这里的技巧是在输入的边界周围用 0 填充，使用足够的 0 使输出的大小与输入的大小相同。然而，这可能具有稀释边界像素的影响的效果。下面的动画显示了一个带有“相同”填充的滑动窗口。

  ![image-20221105143919946](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105143919946.png)

  ![Illustration of zero (same) padding.](https://i.imgur.com/RvGM2xb.gif)

- padding='valid'

  当我们设置 padding='valid' 时，卷积窗口将完全留在输入中。缺点是输出会缩小（丢失像素），并且对于较大的内核会缩小更多。这将限制网络可以包含的层数，尤其是当输入规模较小时。 

## Example - Exploring Sliding Windows

为了更好地理解滑动窗口参数的影响，它可以帮助观察低分辨率图像上的特征提取，以便我们可以看到各个像素。让我们看一个简单的圆圈。 下一个隐藏单元将为我们创建一个图像和内核。

```python
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')

image = circle([64, 64], val=1.0, r_shrink=3)
image = tf.reshape(image, [*image.shape, 1])
# Bottom sobel
kernel = tf.constant(
    [[-1, -2, -1],
     [0, 0, 0],
     [1, 2, 1]],
)

show_kernel(kernel)
```



![image-20221105144124644](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105144124644.png)

VGG 架构相当简单。它使用步幅为 1 的卷积和具有 2×2 窗口和步幅为 2 的最大池化。我们在 visiontools 实用程序脚本中包含了一个函数，它将向我们展示所有步骤。

````python
show_extraction(
    image, kernel,

    # Window parameters
    conv_stride=1,
    pool_size=2,
    pool_stride=2,

    subplot_shape=(1, 4),
    figsize=(14, 6),
)
````

![image-20221105144316268](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105144316268.png)

这很好用！内核设计用于检测水平线，我们可以看到，在生成的特征图中，输入的更多水平部分最终具有最大的激活。 如果我们将卷积的步长改为 3 会发生什么？

```python
show_extraction(
    image, kernel,

    # Window parameters
    conv_stride=3,
    pool_size=2,
    pool_stride=2,

    subplot_shape=(1, 4),
    figsize=(14, 6),    
)
```

![image-20221105144407725](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105144407725.png)

这似乎降低了提取特征的质量。我们的输入圆圈相当“精细”，只有 1 个像素宽。步长为 3 的卷积太粗糙，无法从中生成良好的特征图。 有时，模型会在其初始层中使用具有较大步幅的卷积。这通常也会与更大的内核相结合。例如，ResNet50 模型在其第一层使用 7×7 内核，步长为 2。这似乎加速了大规模特征的产生，而不会牺牲输入中的太多信息。 

结论

在本课中，我们研究了卷积和池化共有的特征计算：滑动窗口和影响其在这些层中的行为的参数。这种类型的窗口计算贡献了卷积网络的大部分特征，并且是其功能的重要组成部分。

## Exercise: The Sliding Window

介绍

在这些练习中，您将探索一些流行的卷积网络架构用于特征提取的操作，了解卷积网络如何通过堆叠层（stacking layers）捕获大规模视觉特征，最后了解卷积如何用于一维数据，在这种情况下，是一个时间序列。

```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.computer_vision.ex4 import *

import tensorflow as tf
import matplotlib.pyplot as plt
import learntools.computer_vision.visiontools as visiontools


plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
```

### (Optional) Experimenting with Feature Extraction

本练习旨在让您有机会探索滑动窗口计算以及它们的参数如何影响特征提取。没有任何正确或错误的答案——这只是一个尝试的机会！ 我们为您提供了一些您可以使用的图像和内核。运行此单元格以查看它们。

```python
from learntools.computer_vision.visiontools import edge, blur, bottom_sobel, emboss, sharpen, circle

image_dir = '../input/computer-vision-resources/'
circle_64 = tf.expand_dims(circle([64, 64], val=1.0, r_shrink=4), axis=-1)
kaggle_k = visiontools.read_image(image_dir + str('k.jpg'), channels=1)
car = visiontools.read_image(image_dir + str('car_illus.jpg'), channels=1)
car = tf.image.resize(car, size=[200, 200])
images = [(circle_64, "circle_64"), (kaggle_k, "kaggle_k"), (car, "car")]

plt.figure(figsize=(14, 4))
for i, (img, title) in enumerate(images):
    plt.subplot(1, len(images), i+1)
    plt.imshow(tf.squeeze(img))
    plt.axis('off')
    plt.title(title)
plt.show();

kernels = [(edge, "edge"), (blur, "blur"), (bottom_sobel, "bottom_sobel"),
           (emboss, "emboss"), (sharpen, "sharpen")]
plt.figure(figsize=(14, 4))
for i, (krn, title) in enumerate(kernels):
    plt.subplot(1, len(kernels), i+1)
    visiontools.show_kernel(krn, digits=2, text_size=20)
    plt.title(title)
plt.show()
```

![image-20221105160034053](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105160034053.png)

要选择一个进行试验，只需在下面的适当位置输入它的名称。然后，设置窗口计算的参数。尝试一些不同的组合，看看他们做了什么！

```python
# YOUR CODE HERE: choose an image
image = circle_64

# YOUR CODE HERE: choose a kernel
kernel = bottom_sobel

visiontools.show_extraction(
    image, kernel,

    # YOUR CODE HERE: set parameters
    conv_stride=1,
    conv_padding='valid',
    pool_size=2,
    pool_stride=2,
    pool_padding='same',
    
    subplot_shape=(1, 4),
    figsize=(14, 6),
)
```

![image-20221105160751037](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105160751037.png)

kernel = bottom_sobel![image-20221105160139232](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105160139232.png)

kernel = edge

![image-20221105160509024](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105160509024.png)

kernel = blur

![image-20221105160539469](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105160539469.png)

kernel = emboss

![image-20221105160613108](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105160613108.png)

kernel = sharpen

![image-20221105160646270](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105160646270.png)



### The Receptive Field(感受野)

追溯来自某个神经元的所有连接，最终到达输入图像。一个神经元连接到的所有输入像素都是该神经元的感受野。感受野只是告诉你神经元从输入图像的哪些部分接收信息。 

正如我们所见，如果您的第一层是具有 3×3 内核的卷积，那么该层中的每个神经元都会从 3×3 像素块（可能在边界处除外）获取输入。

如果添加另一个具有 3×3 内核的卷积层会发生什么？考虑下一个插图：

![image-20221105161319960](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105161319960.png)

现在追溯顶部神经元的连接，您可以看到它连接到输入（底层）中的 5×5 像素块：中间层 3×3 块中的每个神经元连接到一个 3×3 的输入补丁，但它们在一个 5×5 的补丁中重叠。所以顶部的神经元有一个 5×5 的感受野。

![image-20221105163308156](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105163308156.png)

那么为什么要这样堆叠层呢？三个(3, 3)内核有 27 个参数，而一个 (7, 7) 内核有 49 个参数，尽管它们都创建相同的感受野。这种堆叠层技巧是卷积网络能够在不过多增加参数数量的情况下创建大感受野的方式之一。您将在下一课中看到如何自己执行此操作！

### (Optional) One-Dimensional Convolution（一维卷积）

卷积网络不仅对（二维）图像有用，而且在时间序列（一维）和视频（三维）等方面也很有用。

我们已经看到卷积网络如何学习从（二维）图像中提取特征。事实证明，卷积神经网络还可以学习从时间序列（一维）和视频（三维）等事物中提取特征。 

在这个（可选）练习中，我们将看到卷积在时间序列上的样子。 

我们将使用的时间序列来自 Google 趋势。它衡量了从 2015 年 1 月 25 日到 的时间序列来自 Google 趋势。它衡量了从 2015 年 1 月 25 日到 2020 年 1 月 15 日 的几周内搜索词“机器学习”的流行度。

```python
import pandas as pd

# Load the time series as a Pandas dataframe
machinelearning = pd.read_csv(
    '../input/computer-vision-resources/machinelearning.csv',
    parse_dates=['Week'],
    index_col='Week',
)

machinelearning.plot();
```

![image-20221105171945088](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105171945088.png)

内核呢？图像是二维的，因此我们的内核是二维数组。时间序列是一维的，那么内核应该是什么？一维数组！

以下是一些有时用于时间序列数据的内核：

```python
detrend = tf.constant([-1, 1], dtype=tf.float32)

average = tf.constant([0.2, 0.2, 0.2, 0.2, 0.2], dtype=tf.float32)

spencer = tf.constant([-3, -6, -5, 3, 21, 46, 67, 74, 67, 46, 32, 3, -5, -6, -3], dtype=tf.float32) / 320
```

序列上的卷积就像图像上的卷积一样工作。不同之处在于序列上的滑动窗口只有一个方向——从左到右——而不是图像上的两个方向。就像以前一样，挑选出来的特征取决于内核中数字的模式。 

你能猜出这些内核提取了什么样的特征吗？取消注释下面的内核之一并运行单元格查看！

```python
# UNCOMMENT ONE
kernel = detrend
# kernel = average
# kernel = spencer

# Reformat for TensorFlow
ts_data = machinelearning.to_numpy()
ts_data = tf.expand_dims(ts_data, axis=0)
ts_data = tf.cast(ts_data, dtype=tf.float32)
kern = tf.reshape(kernel, shape=(*kernel.shape, 1, 1))

ts_filter = tf.nn.conv1d(
    input=ts_data,
    filters=kern,
    stride=1,
    padding='VALID',
)

# Format as Pandas Series
machinelearning_filtered = pd.Series(tf.squeeze(ts_filter).numpy())

machinelearning_filtered.plot();
```

kernel = detrend

![image-20221105172402563](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105172402563.png)

kernel = average

![image-20221105172428526](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105172428526.png)

kernel = spencer

![image-20221105172523068](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105172523068.png)

事实上，去趋势核(detrend)过滤了序列中的变化，而平均值(average)和斯宾塞(spencer)都是过滤序列中低频分量的“平滑器”。 

如果您对**预测搜索词的未来流行度(predicting the future popularity of search terms)**感兴趣，您可以在像这样的时间序列上训练一个卷积网络。它将尝试了解这些系列中的哪些特征对预测最有用。 

尽管卷积网络本身并不是解决这类问题的最佳选择，但它们通常因其特征提取能力而被合并到其他模型中。

### 结论 

本课结束我们对特征提取的讨论。希望在完成这些课程后，您已经对流程的工作原理以及为什么通常的实施选择通常是最佳选择有了一些直觉。