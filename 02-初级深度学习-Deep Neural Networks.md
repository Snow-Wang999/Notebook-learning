# Deep Neural Networks

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # the hidden ReLU layers
    # 隐藏层
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer
    # 输出层
    layers.Dense(units=1),
])
```

## Exercise: Deep Neural Networks

### 数据集-concrete

在混凝土数据集中，您的任务是预测根据各种配方制造的混凝土的抗压强度。 运行下一个代码单元而不进行更改以加载数据集。

```python
import pandas as pd

concrete = pd.read_csv('../input/dl-course-data/concrete.csv')
concrete.head()
```

### 创建隐藏层

现在创建一个具有三个隐藏层的模型，每个隐藏层有 512 个单元和 ReLU 激活。确保包含一个单元且没有激活的输出层，并将 input_shape 作为第一层的参数。

```python
from tensorflow import keras
from tensorflow.keras import layers

# 模型创建
model = keras.Sequential([
    layers.Dense(512,activation = 'relu',input_shape=[8]),
    layers.Dense(512,activation = 'relu'),
    layers.Dense(512,activation = 'relu'),
    layers.Dense(1),
])
```

#### 模型层的分离

你可能希望在 Dense 层和它的激活函数之间放置一些其他层

```python
#单独使用layers.Activation('relu')
model = keras.Sequential([
    layers.Dense(32, input_shape=[8]),
    layers.Activation('relu'),
    layers.Dense(32),
    layers.Activation('relu'),
    layers.Dense(1),
])
```

**激活函数**

[Exercise: Deep Neural Networks](https://www.kaggle.com/code/wangsnow/exercise-deep-neural-networks/edit)

'relu' 激活有一系列变体——'elu'、'selu' 和 'swish' 等等——所有这些你都可以在 Keras 中使用。

```python
activation_layer = layers.Activation('relu')
#activation_layer = layers.Activation('elu')
#activation_layer = layers.Activation('selu')
#activation_layer = layers.Activation('swish')
```

relu的图片

![image-20221026131920643](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221026131920643.png)

elu的图片

![image-20221026131939441](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221026131939441.png)

selu的图片

![image-20221026131956732](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221026131956732.png)

swish的图片

![swish](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221026131731044.png)

