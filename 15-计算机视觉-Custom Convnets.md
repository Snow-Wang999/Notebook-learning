# 15-Custom Convnets

设计你自己的卷积网络。

[【转载】Keras.layers.Conv2D参数详解 搭建图片分类 CNN （卷积神经网络）](https://blog.csdn.net/Checkmate9949/article/details/119609758)

[计算机视觉(Computer Vision)入门五----自定义卷积网络](https://zhuanlan.zhihu.com/p/493969021)

![image-20221109114327340](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221109114327340.png)

![image-20221109114354587](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221109114354587.png)

## 介绍

现在您已经看到了卷积网络用来提取特征的层，是时候将它们组合在一起并构建您自己的网络了！

## Simple to Refined

在最后三节课中，我们看到了卷积网络如何通过三个操作进行特征提取：**过滤(filter)**、**检测(detect)**和**压缩(condense)**。单轮特征提取只能从图像中提取相对简单的特征，例如简单的线条或对比度。这些太简单了，无法解决大多数分类问题。相反，convnets 会一遍又一遍地重复这种提取，使得特征在深入网络时变得更加复杂和精细。

![image-20221105173659568](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105173659568.png)

## Convolutional Blocks

它通过将它们通过执行此提取的长链卷积块来做到这一点。

![image-20221105173826129](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105173826129.png)

这些卷积块是 Conv2D 和 MaxPool2D 层的堆栈，我们在上几节课中了解了它们在特征提取中的作用。

![image-20221105173859538](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105173859538.png)

每个块代表一轮提取，通过组合这些块，convnet 可以组合和重组产生的特征，增长它们并塑造它们以更好地适应手头的问题。现代 convnet 的深层结构使得这种复杂的特征工程成为可能，并在很大程度上为其卓越的性能负责。

## Example - Design a Convnet

让我们看看如何定义一个能够设计复杂特征的深度卷积网络。在此示例中，我们将创建一个 Keras 序列模型，然后在我们的 Cars 数据集上对其进行训练。

### 第 1 步 - 加载数据

```python
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

### 第 2 步 - 定义模型

![image-20221105174250075](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105174250075.png)

现在我们将定义模型。看看我们的模型如何由三个 Conv2D 和 MaxPool2D 层（基础）块组成，然后是一个 Dense 层的头部。只需填写适当的参数，我们就可以或多或少地将这个图表直接转换为 Keras Sequential 模型。

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # First Convolutional Block
    layers.Conv2D(
        filters=32,
        kernel_size=3,
    	activation='relu',
        padding='SAME',
        # give the input dimensions in the first layer
        # [height, width, color channels(RGB)]
        input_shape=[128, 128, 3]),
    layers.MaxPool2D(),
    
    # Second Convolutional Block
	layers.Conv2D(
        filters=64,
        kernel_size=3,
    	activation='relu',
        padding='SAME'),
    layers.MaxPool2D(),
    
    # Third Convolutional Block
    layers.Conv2D(
        filters=128,
        kernel_size=3,
    	activation='relu',
        padding='SAME'),
    layers.MaxPool2D(),
    
    # Classifier Head
    layers.Flatten(),
    layers.Dense(units=6, activation="relu"),
    layers.Dense(units=1,activation='sigmoid'),
])
model.summary()#输出模型中各层的一些信息。
```

![image-20221105180151931](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105180151931.png)

请注意，此定义中的过滤器数量如何逐块增加一倍：32、64、128。这是一种常见模式。由于 MaxPool2D 层正在减少特征图的大小，我们可以增加我们创建的数量。

### 第 3 步 - 训练

我们可以像第 1 课中的模型一样训练这个模型：使用优化器以及适合二元分类的损失和度量对其进行编译。

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=40,
    verbose=0,
)
```

```python
import pandas as pd

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();
```

![image-20221105180714134](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105180714134.png)

![image-20221105180727836](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105180727836.png)

这个模型比第 1 课的 VGG16 模型要小得多——只有 3 个卷积层，而 VGG16 有 16 个。尽管如此，它仍然能够很好地拟合这个数据集。我们可能仍然可以通过添加更多卷积层来改进这个简单的模型，希望创建更适合数据集的特征。这是我们将在练习中尝试的。

### 结论 

在本教程中，您了解了如何构建由许多卷积块组成并能够进行复杂特征工程的自定义卷积网络。

## Exercise: Custom Convnets

### 介绍

在这些练习中，您将构建一个性能与第 1 课中的 VGG16 模型相媲美的自定义卷积网络。 

```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.computer_vision.ex5 import *

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

### Design a Convnet

让我们设计一个具有块架构的卷积网络，就像我们在教程中看到的那样。示例中的模型具有三个块，每个块都有一个卷积层。它在“汽车或卡车”问题上的表现还可以，但远非预训练的 VGG16 所能达到的。可能是我们的简单网络缺乏提取足够复杂特征的能力。我们可以尝试通过添加更多块或向我们拥有的块添加卷积来改进模型。 让我们采用第二种方法。我们将保留三块结构，但将第二块中的 Conv2D 层数增加到两个，第三块中的 Conv2D 层数增加到三个。

![image-20221105181433215](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105181433215.png)

#### 1) Define Model

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # Block One
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same',
                  input_shape=[128, 128, 3]),
    layers.MaxPool2D(),

    # Block Two
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Three
    # YOUR CODE HERE
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Head
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid'),
])

# Check your answer
q_1.check()
```

#### 2) Compile

要准备训练，请使用“汽车或卡车”数据集的适当损失和准确度指标来编译模型。

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)
```

最后，我们来测试一下这款新机型的性能。首先运行此单元以使模型适合训练集。

```python
history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=50,
)
```

现在运行下面的单元格来绘制此训练运行的损失和度量曲线。

```python
import pandas as pd
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();
```

![image-20221105182552123](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105182552123.png)

![image-20221105182603310](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105182603310.png)

教程中模型的学习曲线变化很快。这表明它容易过度拟合并且需要一些正则化(regularization)。我们新模型中的附加层将使其更容易过度拟合。但是，使用 Dropout 层添加一些正则化有助于防止这种情况发生。这些变化将模型的验证准确性提高了几个点。

### 结论 

这些练习向您展示了如何设计自定义卷积网络来解决特定的分类问题。尽管如今大多数模型都将建立在预训练的基础之上，但在某些情况下，更小的自定义卷积网络可能仍然更可取——例如使用较小或不寻常的数据集，或者当计算资源非常有限时。正如您在此处看到的，对于某些问题，它们的性能与预训练模型一样好。