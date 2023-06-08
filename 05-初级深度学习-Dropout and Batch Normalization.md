# Dropout and Batch Normalization（批量标准化）

本章中，我们学习两种特殊的层，本身不包含任何神经元，但它们会使模型受益。

## Dropout

其中第一个是“dropout layer”，它可以帮助纠正过拟合。

在网络学习中，会陷入虚假的权重的“阴谋”中，由于他们的脆弱，只要移除一个神经元就可以瓦解“阴谋”。这就是**dropout**。

我们在每一步训练中**随机删除了一层输入单元的一部分**，这使得网络更难学习训练数据中的那些虚假模式。相反，它必须搜索广泛的、一般的模式，其权重模式往往更稳健。

![An animation of a network cycling through various random dropout configurations.](https://i.imgur.com/a86utxY.gif)

您也可以将 dropout 视为创建一种网络集合。预测将不再由一个大网络做出，而是由一个由较小网络组成的委员会（子集）做出。委员会中的个人倾向于犯不同类型的错误，但同时也是正确的，这使得委员会作为一个整体比任何个人都好。 （如果您熟悉作为决策树集合的随机森林，那也是同样的想法。）

### Adding Dropout-随机建立子网络

在 Keras 中，dropout rate 参数 rate 定义了要关闭的输入单元的百分比。将 Dropout 图层放在要应用 dropout 的图层之前：

```python
keras.Sequential([
    # ...
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.Dense(16),
    # ...
])
```



## Batch Normalization-批量标准化

特殊层“批量标准化”（或“`batchnorm`”），它可以帮助纠正缓慢或不稳定的训练。

对于神经网络，将所有数据放在一个共同的尺度上通常是一个好主意，也许使用 scikit-learn 的 StandardScaler 或 MinMaxScaler 之类的东西。原因是 SGD 将根据数据产生的激活量大小来调整网络权重。倾向于产生非常不同大小的激活的特征可能会导致不稳定的训练行为。 

现在，如果在数据进入网络之前对其进行规范化是好的，那么**在网络内部进行规范化可能会更好**！事实上，我们有一种特殊的层可以做到这一点，即批量标准化层。批次归一化层会查看每个批次，首先使用其自己的均值和标准差对批次进行归一化，然后使用两个可训练的重新缩放参数将数据置于新的尺度上。实际上，Batchnorm 对其输入进行了一种协调的重新调整。 

大多数情况下，batchnorm 被添加为优化过程的辅助（尽管它有时也可以帮助**预测性能**）。具有 batchnorm 的模型往往需要**更少的 epoch **来完成训练。此外，batchnorm 还可以**修复各种可能导致训练“卡住”的问题**。考虑为您的模型添加批量标准化，尤其是当您在训练期间遇到问题时。

### Adding Batch Normalization

似乎可以在网络中的几乎任何点使用批量标准化。你可以把它放在一层之后...

```python
layers.Dense(16, activation='relu'),
layers.BatchNormalization(),

... or between a layer and its activation function:

layers.Dense(16),
layers.BatchNormalization(),
layers.Activation('relu'),
```

如果您将其添加为网络的第一层，它可以充当一种自适应预处理器，代替 Sci-Kit Learn 的 `StandardScaler` 之类的东西。

## Example - Using Dropout and Batch Normalization

让我们继续开发红酒模型。现在我们将进一步增加容量，但添加 dropout 以控制过度拟合和批量标准化以加速优化。这一次，我们还将停止对数据进行标准化，以展示批量标准化如何稳定训练。

```python
# Setup plotting
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)


import pandas as pd
red_wine = pd.read_csv('../input/dl-course-data/red-wine.csv')

# Create training and validation splits
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']
```

添加 dropout 时，您可能需要增加 Dense 层中的单元数。

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(1024, activation = 'relu', input_shape=[11]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation= 'relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation= 'relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
])
```

```python
model.compile(
	optimizer = 'adam',
    loss = 'mae'
)

history = model.fit(
	X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size = 256,
    epochs = 100,
    verbose = 0,
)

# show the learning curves
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
```

![image-20221029144300698](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221029144300698.png)

如果在将数据用于训练之前对其进行标准化，通常会获得更好的性能。然而，我们完全能够使用原始数据，这表明批量标准化在更困难的数据集上是多么有效。

***

窗口置顶方法：

Topmost_x64.exe双击打开

窗口置顶/取消快捷键：点击窗口，按ctrl+alt+space

***

## Exercise: Dropout and Batch Normalization

### 加载基础环境

```python
# Setup plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning_intro.ex5 import *
```

### 载入数据集-load Spotify dataset

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

spotify = pd.read_csv('../input/dl-course-data/spotify.csv')
#spotify.head()
X = spotify.copy().dropna()
y = X.pop('track_popularity')
artists = X['track_artist']

features_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']
features_cat = ['playlist_genre']

preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat),
)

def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

X_train, X_valid, y_train, y_valid = group_split(X, y, artists)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train = y_train / 100
y_valid = y_valid / 100

input_shape = [X_train.shape[1]]
print("Input shape: {}".format(input_shape))

```

Input shape: [18]

### 1. 将 Dropout 添加到 Spotify 模型

这是练习 4 中的最后一个模型。添加两个 dropout 层，一个在具有 128 个单元的 Dense 层之后，一个在具有 64 个单元的 Dense 层之后。将两者的辍学率设置为 0.3。

```python
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1),
])
```

```python
model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    verbose=0,
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
```

![image-20221029152000691](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221029152000691.png)

### 2. Evaluate Dropout

```python
history_df.loc[10:, ['loss', 'val_loss']].plot()
```

![image-20221029152231243](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221029152231243.png)

从学习曲线中，您可以看到验证损失保持在一个恒定的最小值附近，即使训练损失继续减少。所以我们可以看到这次添加 dropout 确实防止了过度拟合。此外，通过使网络更难拟合虚假模式，dropout 可能会鼓励网络寻找更多真实模式，也可能会在一定程度上改善验证损失）。

### 3. Add Batch Normalization Layers

现在，我们将切换主题来探索批量标准化如何解决训练中的问题。 加载concrete数据集。这次我们不会做任何标准化。这将使批量标准化的效果更加明显。

```python
import pandas as pd

concrete = pd.read_csv('../input/dl-course-data/concrete.csv')
df = concrete.copy()

df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

X_train = df_train.drop('CompressiveStrength', axis=1)
X_valid = df_valid.drop('CompressiveStrength', axis=1)
y_train = df_train['CompressiveStrength']
y_valid = df_valid['CompressiveStrength']

input_shape = [X_train.shape[1]]
```

在未标准化的具体数据上训练网络。

```python
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=input_shape),
    layers.Dense(512, activation='relu'),    
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])
model.compile(
    optimizer='sgd', # SGD is more sensitive to differences of scale
    loss='mae',
    metrics=['mae'],
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=64,
    epochs=100,
    verbose=0,
)

history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print(("Minimum Validation Loss: {:0.4f}").format(history_df['val_loss'].min()))
```

![image-20221029152725006](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221029152725006.png)

没有收敛

批量标准化可以帮助纠正这样的问题。 

添加四个`BatchNormalization` 层，在每个密集层之前添加一个。 （记得将 `input_shape` 参数移动到新的第一层

```python
# YOUR CODE HERE: Add a BatchNormalization layer before each Dense layer
model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(512, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dense(1),
])
```

训练模型

```python
model.compile(
    optimizer='sgd',
    loss='mae',
    metrics=['mae'],
)
EPOCHS = 100
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=64,
    epochs=EPOCHS,
    verbose=0,
)

history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print(("Minimum Validation Loss: {:0.4f}").format(history_df['val_loss'].min()))
```

![image-20221029153339298](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221029153339298.png)

您可以看到，在第一次尝试中添加批量标准化是一个很大的改进！在数据通过网络时自适应地缩放数据，批量标准化可以让您在困难的数据集上训练模型。