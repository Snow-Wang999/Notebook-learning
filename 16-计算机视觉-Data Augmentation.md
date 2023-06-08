# 16-Data Augmentation（数据增强）

通过创建额外的训练数据来提高性能。

[计算机视觉(Computer Vision)入门五----自定义卷积网络](https://zhuanlan.zhihu.com/p/493969021)

## 介绍

现在您已经了解了卷积分类器的基础知识，您可以继续学习更高级的主题了。 在本课中，您将学习一个可以增强图像分类器的技巧：它称为数据增强（**data augmentation**）。

## 假数据的用处

提高机器学习模型性能的最佳方法是在**更多数据**上对其进行训练。模型必须学习的示例越多，它就能更好地识别图像中哪些差异重要，哪些不重要。更多数据有助于模型更好地泛化。 

获取更多数据的一种简单方法是**使用您已有的数据**。如果我们可以以保留类别的方式转换数据集中的图像，我们可以教我们的分类器忽略这些类型的转换（ ignore those kinds of transformations）。例如，汽车在照片中是面向左侧还是面向右侧并不会改变它是汽车而不是卡车的事实。因此，如果我们用翻转的图像来扩充我们的训练数据，我们的分类器将知道“左或右”是它应该忽略的差异。 

这就是数据增强背后的全部想法：添加一些看起来与真实数据相当相似的额外假数据，您的分类器将得到改进。

## Using Data Augmentation

对于图片数据，常见的数据增强方式包括：

- **随机水平翻转：**
- **随机的裁剪；**
- **随机调整明亮程度；**
- 其他方式等。

通常，在扩充数据集时会使用多种转换。这些可能包括**旋转图像、调整颜色或对比度、扭曲图像或许多其他通常组合应用的东西**。以下是单个图像可能被转换的不同方式的示例。

![image-20221105183521356](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105183521356.png)

数据增强通常是在线（online）完成的，这意味着，当图像被输入网络进行训练时。回想一下，训练通常是在小批量数据上进行的。这就是使用数据增强时一批 16 幅图像的样子。

![image-20221105183705768](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105183705768.png)

每次在训练期间使用图像时，都会应用新的随机变换。这样，模型总是看到与以前看到的有些不同的东西。训练数据中的这种额外差异有助于模型处理新数据。 

重要的是要记住，并非每个转换都对给定问题有用。最重要的是，您使用的任何转换都不应混淆类。例如，如果您正在训练数字识别器，旋转图像会混淆 '9' 和 '6'。最后，找到好的增强的最佳方法与大多数 ML 问题相同：试试看！

## Example - Training with Data Augmentation

在 `TensorFlow` 之中进行图像数据增强的方式主要有两种：

- **使用 tf.keras 的预处理层进行图像数据增强**；
- **使用 tf.image 进行数据增强**。

## 使用tf.image进行数据增强

使用 tf.image 是 TensorFlow 最原生的一种增强方式，使用这种方式可以实现**更多、更加个性化的**数据增强。

其中包含的数据增强方式主要包括：

- tf.image.flip_left_right (img)：将图片进行水平翻转；
- tf.image.rgb_to_grayscale (img)：将 RGB 图像转化为灰度图像；
- tf.image.adjust_saturation (image, f)：将 image 图像按照 f 参数进行饱和度的调节；
- tf.image.adjust_brightness (image, f)：将 image 图像按照 f 参数进行亮度的调节；
- tf.image.central_crop (image, central_fraction)：按照 p 的比例进行图片的中心裁剪，比如如果 p 是 0.5 ，那么裁剪后的长、宽就是原来图像的一半；
- tf.image.rot90 (image)：将 image 图像逆时针旋转 90 度。

可以看到，很多的 tf.image 数据增强方式并不提供随机化选项，因此我们需要手动进行随机化。

也正是因为上述特性，tf.image 数据增强主要用在一些自定义的模型之中，从而可以实现数据增强的自定义化。

## 使用 tf.keras 的预处理层进行数据增强的实例

[在 TensorFlow 之中进行数据增强](https://book.itxueyuan.com/p3WR/B22PK)

`Keras` 允许您以两种方式扩充数据。

- 第一种方法是使用 `ImageDataGenerator` 之类的函数将其包含在数据管道中。

- 第二种方法是使用 `Keras` 的预处理层将其包含在模型定义中。

是我们将采取的方法。对我们来说，主要优势是图像转换将在 GPU 而不是 CPU 上计算，这可能会加快训练速度。 

在本练习中，我们将学习如何通过数据增强来改进第 1 课中的分类器。下一个隐藏单元设置数据管道。

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

为了说明增强的效果，我们将向教程 1 中的模型添加几个简单的转换。

```python
from tensorflow import keras
from tensorflow.keras import layers
# these are a new feature in TF 2.2
from tensorflow.keras.layers.experimental import preprocessing


pretrained_base = tf.keras.models.load_model(
    '../input/cv-course-models/cv-course-models/vgg16-pretrained-base',
)
pretrained_base.trainable = False

model = keras.Sequential([
    # Preprocessing
    preprocessing.RandomFlip('horizontal'), # flip(翻转) left-to-right
    preprocessing.RandomContrast(0.5), # contrast change by up to 50% 按p的概率进行随机的图像色相翻转
    # Base
    pretrained_base,
    # Head
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])
```

#### 如何使用 tf.keras 的预处理层进行图像数据增强

```python
tf.keras.layers.experimental.preprocessing
```

在这个包之中，我们最常用的数据增强 API 包括：

- `tf.keras.layers.experimental.preprocessing.RandomFlip(mode)`: **将输入的图片进行随机翻转**，一般我们会取 `mode=“horizontal”` ，因为这代表水平旋转；而 `mode=“vertical”` 则代表随机进行上下翻转；
- `tf.keras.layers.experimental.preprocessing.RandomRotation`: **按照旋转角度（单位为弧度） p 将输入的图片进行随机的旋转**；
- `tf.keras.layers.experimental.preprocessing.RandomContrast`：**按照 P 的概率将输入的图片进行随机的图像色相翻转**；
- `tf.keras.layers.experimental.preprocessing.CenterCrop(height, width)`：**使用 height \* width 的大小的裁剪框，在数据的中心进行裁剪**。

### Step 3 - Train and Evaluate

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

```python
import pandas as pd

history_frame = pd.DataFrame(history.history)

history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();
```

![image-20221105220244005](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105220244005.png)

![image-20221105220256583](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221105220256583.png)

教程 1 中的模型中的训练和验证曲线发散得相当快，这表明它可以从一些正则化中受益。该模型的学习曲线能够更紧密地结合在一起，并且我们在验证损失和准确性方面取得了一些适度的改进。这表明数据集确实从增强中受益。

## Exercise: Data Augmentation

### 介绍

在这些练习中，您将探索各种随机变换对图像的影响，考虑在给定数据集上可能适用的增强类型，然后使用 Car 或 Truck 数据集的数据增强来训练自定义网络。

```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.computer_vision.ex6 import *

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

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



### (Optional) Explore Augmentation

取消注释转换并运行单元格以查看它的作用。如果愿意，您也可以试验参数值。 （因子参数应大于 0，通常小于 1。）如果您想获得新的随机图像，请再次运行单元格。

```python
# all of the "factor" parameters indicate a percent-change
augment = keras.Sequential([
    # preprocessing.RandomContrast(factor=0.5),
    preprocessing.RandomFlip(mode='horizontal'), # meaning, left-to-right
    # preprocessing.RandomFlip(mode='vertical'), # meaning, top-to-bottom
    # preprocessing.RandomWidth(factor=0.15), # horizontal stretch
    # preprocessing.RandomRotation(factor=0.20),
    # preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
])


ex = next(iter(ds_train.unbatch().map(lambda x, y: x).batch(1)))

plt.figure(figsize=(10,10))
for i in range(16):
    image = augment(ex, training=True)
    plt.subplot(4, 4, i+1)
    plt.imshow(tf.squeeze(image))
    plt.axis('off')
plt.show()
```

RandomFlip(mode='horizontal')

左右翻转

![image-20221106123703351](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106123703351.png)

![image-20221106123735047](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106123735047.png)

RandomFlip(mode='vertical')

上下翻转

![image-20221106124229394](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106124229394.png)

RandomContrast(factor=0.5)

按照 P 的概率将输入的图片进行随机的图像色相翻转

![image-20221106124344505](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106124344505.png)

RandomRotation(factor=0.20)

随机旋转

![image-20221106124548819](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106124548819.png)

RandomWidth(factor=0.15)

水平上拉伸与缩减

![image-20221106124724153](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106124724153.png)

RandomTranslation(height_factor=0.1, width_factor=0.1)

平移

![image-20221106124858992](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106124858992.png)

您选择的转换对于 Car 或 Truck 数据集是否合理？

在本练习中，我们将查看一些数据集并考虑哪种增强可能是合适的。您的推理可能与我们在解决方案中讨论的不同。没关系。这些问题的重点只是考虑转换如何与分类问题相互作用——无论好坏。

EuroSAT 数据集由按地理特征分类的地球卫星图像组成。以下是来自该数据集的一些图像。

![image-20221106123905161](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106123905161.png)

![image-20221106123919380](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106123919380.png)

![image-20221106123940304](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106123940304.png)

### 1) EuroSAT 

什么样的转换可能适合这个数据集？

```python
augment = keras.Sequential([
    # preprocessing.RandomContrast(factor=0.5),
     preprocessing.RandomFlip(mode='horizontal'), # meaning, left-to-right
     preprocessing.RandomFlip(mode='vertical'), # meaning, top-to-bottom
    # preprocessing.RandomWidth(factor=0.15), # horizontal stretch
     preprocessing.RandomRotation(factor=0.20),
    # preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
])
```



在这位作者看来，翻转和旋转值得首先尝试，因为直接从头顶拍摄的照片没有方向的概念。然而，这些转换似乎都不会混淆类。

### 2) TensorFlow Flowers

哪些类型的转换可能适合 TensorFlow Flowers 数据集？

```python
augment = keras.Sequential([
    # preprocessing.RandomContrast(factor=0.5),
     preprocessing.RandomFlip(mode='horizontal'), # meaning, left-to-right
    # preprocessing.RandomFlip(mode='vertical'), # meaning, top-to-bottom
    # preprocessing.RandomWidth(factor=0.15), # horizontal stretch
     preprocessing.RandomRotation(factor=0.20),
    # preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
])
```



在这位作者看来，水平翻转和适度旋转值得首先尝试。一些增强库包括色调转换（如红色到蓝色）。由于一朵花的颜色似乎在其类别中与众不同，因此改变色调可能不太成功。另一方面，玫瑰等栽培花卉种类繁多，因此，根据数据集，这毕竟可能是一种改进！

### 3) Add Preprocessing Layers

将这些预处理层添加到给定模型中。 

```python
preprocessing.RandomContrast(factor=0.10),
preprocessing.RandomFlip(mode='horizontal'),
preprocessing.RandomRotation(factor=0.10),
```

hint:提示

```python
model = keras.Sequential([
    layers.InputLayer(input_shape=[128, 128, 3]),

    # Data Augmentation
    preprocessing.____,
    preprocessing.____,
    preprocessing.____,

    # Block One
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # More layers follow...
])
```

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.InputLayer(input_shape=[128, 128, 3]),
    
    # Data Augmentation
    preprocessing.RandomContrast(factor=0.10),
    preprocessing.RandomFlip(mode='horizontal'),
    preprocessing.RandomRotation(factor=0.10),

    # Block One
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Two
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Three
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Head
    layers.BatchNormalization(renorm=True),
    layers.Flatten(),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

```

### 4) Train Model

现在我们将训练模型。运行下一个单元以使用损失和准确度指标对其进行编译，并将其拟合到训练集。

```python
optimizer = tf.keras.optimizers.Adam(epsilon=0.01)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=50,
)

# Plot learning curves
import pandas as pd
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();
```

![image-20221106134002911](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106134002911.png)

![image-20221106134017850](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221106134017850.png)

检查训练曲线。有什么过拟合的迹象？与您在本课程中训练的其他模型相比，此模型的性能如何？

与以前的模型相比，该模型中的学习曲线保持在一起的时间要长得多。这表明增强有助于防止过度拟合，从而使模型能够继续改进。 

注意，该模型达到了课程中所有模型的最高准确度！情况并非总是如此，但它表明精心设计的自定义卷积网络有时可以与更大的预训练模型一样好或更好。根据您的应用程序，拥有更小的模型（需要更少的资源）可能是一个很大的优势。

### 结论 

数据增强是改进模型训练的强大且常用的工具，不仅适用于卷积网络，也适用于许多其他类型的神经网络模型。无论您遇到什么问题，原则都是一样的：您可以通过添加“假”数据来弥补数据的不足。尝试增强是了解数据可以走多远的好方法！

### 结束 

这就是 Kaggle Learn 上的计算机视觉的全部内容！你准备好应用你的知识了吗？查看我们的两个奖励课程！他们将指导您准备比赛的提交，同时您学习如何使用 Kaggle 最先进的加速器 TPU 训练神经网络。最后，您将拥有一个完整的笔记本，可以用您自己的想法进行扩展。

- [Create Your First Submission](https://www.kaggle.com/ryanholbrook/create-your-first-submission) - 为我们的 Petals 准备提交到 Metal Getting Started 比赛。您将训练一个神经网络来识别 100 多种花卉。
- [Cassava Leaf Disease](https://www.kaggle.com/jessemostipak/getting-started-tpus-cassava-leaf-disease) - 宁愿争夺金钱和奖牌？训练神经网络来诊断木薯植物的常见病害，木薯是非洲的主要安全作物。