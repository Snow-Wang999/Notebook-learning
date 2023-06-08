# Overfitting and Underfitting

通过额外容量或提前停止来提高性能。

## 介绍

回想一下上一课中的示例，Keras 将在训练模型的各个时期**保留训练和验证损失的历史记录**。在本课中，我们将学习如何解释这些学习曲线以及如何使用它们来指导模型开发。特别是，我们将检查学习曲线以**寻找欠拟合和过拟合的证据**，并研究一些纠正它的策略。

## 解释学习曲线 

您可能会认为训练数据中的信息有两种：**信号和噪声**。信号是泛化的部分，可以帮助我们的模型根据新数据进行预测的部分。噪声是仅对训练数据有效的部分；噪声是来自现实世界中的数据的所有随机波动，或者是实际上无法帮助模型做出预测的所有附带的、无信息的模式。噪音是该部分可能看起来有用但实际上不是。

我们通过选择最小化训练集损失的权重或参数来训练模型。但是，您可能知道，要准确评估模型的性能，我们需要在一组新数据（**验证数据**）上对其进行评估。 （您可以在机器学习简介中查看我们关于模型验证的课程以进行复习。）

当我们训练一个模型时，我们一直在逐个地绘制训练集上的损失。为此，我们还将添加验证数据图。这些图我们称为学习曲线。为了有效地训练深度学习模型，我们需要能够解释它们。

![image-20221027163516055](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027163516055.png)

现在，无论是模型学习信号还是学习噪声，训练损失都会下降。但是只有当模型学习到信号时，验证损失才会下降。 （无论模型从训练集中学到的任何噪声都不会推广到新数据。）因此，当模型学习信号时，两条曲线都会下降，但是当它学习噪声时，曲线中会产生间隙。差距的大小告诉你模型学到了多少噪音。 

理想情况下，我们将创建学习所有信号而不学习噪声的模型。这几乎不会发生。相反，我们进行交易。我们可以让模型以学习更多噪声为代价学习更多信号。只要交易对我们有利，验证损失就会继续减少。然而，在某个点之后，交易可能对我们不利，成本超过收益，验证损失开始上升。

![image-20221027163659483](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027163659483.png)

这种权衡表明在训练模型时可能会出现两个问题：**信号不足或噪声过多**。训练集欠拟合是因为模型没有学习到足够的信号，所以损失没有尽可能低。过度拟合训练集是因为模型学习了太多噪声而导致损失没有尽可能低。训练深度学习模型的诀窍是在两者之间找到最佳平衡。 

我们将研究几种从训练数据中获取更多信号同时减少噪声量的方法。

## 容量（Capacity）

> 模型的容量是指它能够学习的模式的大小和复杂性。对于神经网络，这在很大程度上取决于它有多少神经元以及它们如何连接在一起。如果您的网络似乎**欠拟合数据，您应该尝试增加其容量**。 

> 您可以通过使其更宽（**现有层的神经元更多**）或使其更深（**添加更多层**）来增加网络的容量。更广泛的网络更容易学习更多的线性关系，而更深的网络更喜欢更非线性的关系。哪个更好取决于数据集。

解决欠拟合，有两种增加容量的方法：

- （wider）增加现有层的神经元——学习更多的线性关系
- （deeper）添加更多层——学习更多非线性关系

```python
model = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])

wider = keras.Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dense(1),
])

deeper = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])
```

## 提前停止（Early Stopping）

我们提到，当模型过于急切地学习噪声时，**验证损失可能会在训练期间开始增加**。为了防止这种情况，只要验证损失似乎不再减少，我们就可以简单地停止训练。以这种方式中断训练称为提前停止。

![image-20221027164511351](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027164511351.png)

一旦我们检测到验证损失再次开始上升，我们可以**将权重重置回最小值出现的位置**。这确保了模型不会继续学习噪声和过度拟合数据。

提前停止训练也意味着我们在网络完成学习信号之前，使过早停止训练的危险减小。所以除了防止过拟合训练时间过长外，提前停止还可以防止欠拟合训练时间不够长。只需将您的训练时期设置为较大的数字（超出您的需要），然后提前停止即可处理其余的问题。

### 添加提前停止（Adding Early Stopping）

在 Keras 中，我们包括通过回调提前停止训练。回调只是您希望在网络训练时经常运行的函数。提前停止回调将在每个 epoch 之后运行。 （Keras 预定义了各种有用的回调，但您也可以定义自己的。）

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement-最小量的改变算作改进
    patience=20, # how many epochs to wait before stopping-之前要等待多少个 epoch
    restore_best_weights=True,
)
```

这些参数表示：“如果在前 20 个 epoch 中验证损失没有至少提高 0.001，那么停止训练并保留你找到的最佳模型。”有时很难判断验证损失是由于过度拟合还是由于随机批次变化而上升。这些参数允许我们围绕何时停止设置一些余量。 

正如我们将在示例中看到的，我们将此回调与损失和优化器一起传递给 fit 方法。

#### 补充：回调函数

当程序跑起来时，一般情况下，应用程序（application program）会时常通过API调用库里所预先备好的函数。但是有些**库函数（library function）**却要求应用先传给它一个函数，好在合适的时候调用，以完成目标任务。这个被传入的、后又被调用的函数就称为**回调函数（callback function）**。

打个比方，有一家旅馆提供叫醒服务，但是要求旅客自己**决定叫醒的方法**。可以是打客房电话，也可以是派服务员去敲门，睡得死怕耽误事的，还可以要求往自己头上浇盆水。这里，“叫醒”这个行为是旅馆提供的，相当于库函数（library function），但是叫醒的方式是由旅客决定并告诉旅馆的，也就是回调函数（callback function）。而旅客告诉旅馆怎么叫醒自己的动作，也就是把回调函数传入库函数的动作，称为**登记回调函数**（to register a callback function）。

![image-20221027165138569](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027165138569.png)

#### 示例 - 使用提前停止训练模型（Example - Train a Model with Early Stopping）

让我们继续从上一个教程中的示例开发模型。我们将增加该网络的容量，但还会添加一个提前停止回调以防止过度拟合。

##### 对数据进行预处理

```python
import pandas as pd
from IPython.display import display

red_wine = pd.read_csv('../input/dl-course-data/red-wine.csv')

# Create training and validation splits
df_train = red_wine.sample(frac=0.7, random_state=0)
#把70%的数据拿来做训练数据
df_valid = red_wine.drop(df_train.index)
display(df_train.head(4))
#从全面数据中剔除训练数据，剩下作为验证数据

# Scale to [0, 1]
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']
#输出目标是‘quality’
```

![image-20221028160836095](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221028160836095.png)

现在让我们增加网络的容量。我们将选择一个相当大的网络，但是一旦验证损失显示出增加的迹象，就依靠回调来停止训练。

```python
from tensorflow import keras
from tensorflow.keras import layers, callbacks

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement-被视为改进的最小更改量
    patience=20, # how many epochs to wait before stopping-在停止之前要等待多少个 epoch
    restore_best_weights=True,
)

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[11]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])
model.compile(
    optimizer='adam',
    loss='mae',
)
```

定义回调后，将其作为参数添加到 fit 中（您可以有多个，所以将它放在一个列表中）。使用提前停止时选择大量的时期，比你需要的要多。

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping], # put your callbacks in a list
    verbose=0,  # turn off training log-关闭训练日志
)

# show the image
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

```

![image-20221028161306664](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221028161306664.png)

果然，Keras 在完整的 500 个 epoch 之前就停止了训练！

***

## Exercise: Overfitting and Underfitting

### 1. 介绍 

在本练习中，您将学习如何通过包含提前停止回调以防止过度拟合来提高训练结果。 

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
from learntools.deep_learning_intro.ex4 import *
```

详情请查看A single Neuron

首先加载 Spotify 数据集。您的任务是根据各种音频特征（如“节奏”、“舞蹈能力”和“模式”）预测歌曲的流行度。

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

# 读取数据
spotify = pd.read_csv('../input/dl-course-data/spotify.csv')

# 把输入输出的数据分开
X = spotify.copy().dropna()
y = X.pop('track_popularity')
artists = X['track_artist']

#数值特征
features_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']
#文字特征
features_cat = ['playlist_genre']

preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat),
)

# We'll do a "grouped" split to keep all of an artist's songs in one split or the other. This is to help prevent signal leakage.
#我们将进行“分组”拆分，以将艺术家的所有歌曲保留在一个拆分或另一个拆分中。这有助于防止信号泄漏。
def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

X_train, X_valid, y_train, y_valid = group_split(X, y, artists)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train = y_train / 100 # popularity is on a scale 0-100, so this rescales to 0-1.
y_valid = y_valid / 100

input_shape = [X_train.shape[1]]
print("Input shape: {}".format(input_shape))
```

***

#### 补充：[Scikit-learn的K-fold交叉验证类ShuffleSplit、GroupShuffleSplit用法介绍](https://blog.csdn.net/hurry0808/article/details/80797969)

[关于交叉验证的一点事儿](https://zhuanlan.zhihu.com/p/98209649)

当样本数据量比较小时，K-fold交叉验证是训练、评价模型时的常用方法，该方法的作用如下：

- 交叉验证用于评估模型的预测性能，尤其是训练好的模型在新数据上的表现，可以在一定程度上减小过拟合
- 交叉验证可以从有限的数据中获取尽可能多的有效信息

本文介绍`sklearn`的可用于K-fold交叉验证的集合划分类`ShuffleSplit`、`GroupShuffleSplit`的用法。

##### `ShuffleSplit`

`sklearn.model_selection.ShuffleSplit`类用于将样本集合随机“打散”后划分为训练集、测试集(可理解为验证集，下同)，类申明如下：

```python
class sklearn.model_selection.ShuffleSplit(n_splits=10, test_size='default', train_size=None, random_state=None)
```

参数：

- `n_splits:int`, 划分训练集、测试集的次数，默认为10
- `test_size: float, int, None, default=0.1`； 测试集比例或样本数量，该值为[0.0, 1.0]内的浮点数时，表示测试集占总样本的比例；该值为整型值时，表示具体的测试集样本数量；`train_size`不设定具体数值时，该值取默认值0.1，`train_size`设定具体数值时，`test_size`取剩余部分
- `train_size:float, int, None`； 训练集比例或样本数量，该值为[0.0, 1.0]内的浮点数时，表示训练集占总样本的比例；该值为整型值时，表示具体的训练集样本数量；该值为`None`(默认值)时，训练集取总体样本除去测试集的部分
- `random_state: int, RandomState instance or None`；随机种子值，默认为`None`

***

##### `GroupShuffleSplit`

`sklearn.model_selection.GroupShuffleSplit`作用与`ShuffleSplit`相同，不同之处在于`GroupShuffleSplit`先将待划分的样本集分组，再按照分组划分训练集、测试集。

```python
class sklearn.model_selection.GroupShuffleSplit(n_splits=5, test_size='default', train_size=None, random_state=None)
```

参数个数及含义同`ShuffleSplit`，只是默认值有所不同：

- n_splits:int, 划分训练集、测试集的次数，默认为5
- test_size:float, int, None, default=0.1； 测试集比例或样本数量，该值为[0.0, 1.0]内的浮点数时，表示测试集占总样本的比例；该值为整型值时，表示具体的测试集样本数量；train_size不设定具体数值时，该值取默认值0.2，train_size设定具体数值时，test_size取剩余部分
- train_size:float, int, None； 训练集比例或样本数量，该值为[0.0, 1.0]内的浮点数时，表示训练集占总样本的比例；该值为整型值时，表示具体的训练集样本数量；该值为None(默认值)时，训练集取总体样本除去测试集的部分
- random_state:int, RandomState instance or None；随机种子值，默认为None
  

#### from sklearn.compose import make_column_transformer

***

#### 线性模型

让我们从最简单的网络开始，一个线性模型。该型号容量低。 在不做任何更改的情况下运行下一个单元，以在 Spotify 数据集上训练线性模型。

```python
model = keras.Sequential([
    layers.Dense(1, input_shape=input_shape),
])
model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    verbose=0, # suppress output since we'll plot the curves
)
history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
```

![image-20221028171957245](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221028171957245.png)

曲线遵循您在此处看到的“曲棍球棒”模式并不少见。这使得训练的最后部分很难看到，所以让我们从 epoch 10 开始：

```python
# Start the plot at epoch 10
history_df.loc[10:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
```

![image-20221028172100981](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221028172100981.png)

### 2. 评估基线

![image-20221028171957245](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221028171957245.png)

由上图可知，这些**曲线之间的差距非常小**，并且**验证损失永远不会增加**，因此网络更可能是**欠拟合**而不是过拟合。值得尝试更多的容量，看看是否是这种情况。

### 3. 增加容量

现在让我们为我们的网络添加一些容量。我们将添加三个隐藏层，每个隐藏层 128 个单元。运行下一个单元来训练网络并查看学习曲线。

```python
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
```

![image-20221028173109502](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221028173109502.png)

![image-20221028173123936](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221028173123936.png)

![image-20221028173140724](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221028173140724.png)

现在**验证损失很早就开始上升，而训练损失继续减少**。这表明网络已经开始**过拟合**。在这一点上，我们需要尝试一些方法来防止它，或者通过**减少单元的数量，或者通过像提前停止***这样的方法。 （我们将在下一课中看到另一个！）

### 4. 定义提前停止回调

现在定义一个提前停止回调，等待 5 个 epoch（`patience`）以使验证损失发生至少 0.001（min_delta）的变化，并保持权重具有最佳损失（restore_best_weights）。

```python
from tensorflow.keras import callbacks

early_stopping = callbacks.EarlyStopping(min_delta=0.001,patience=5,restore_best_weights=True)
```

现在运行这个单元来训练模型并获得学习曲线。注意 model.fit 中的回调参数。

```python
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    callbacks=[early_stopping]
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
```

![image-20221028174209324](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221028174209324.png)

![image-20221028174223656](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221028174223656.png)

一旦网络开始过度拟合，早期停止回调确实停止了训练。此外，通过包含 restore_best_weights，我们仍然可以将模型保持在验证损失最低的位置。

如果您愿意，请尝试`patience`和 `min_delta`，看看它可能会产生什么不同。

```python
early_stopping = callbacks.EarlyStopping(min_delta=0.0001,patience=5,restore_best_weights=True,verbose=0)
```

![image-20221028174917168](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221028174917168.png)

- verbose：信息展示模式

  - verbose = 0 为不在标准输出流输出日志信息
  - verbose = 1 为输出进度条记录
  - verbose = 2 为每个epoch输出一行记录
  - verbose=2，只在fit中能用，evaluate不能

- patience =20 ，当验证集损失在连续20次训练周期中都没有降低时，停止模型训练，以防止过拟合。即能忍受多少个epoch内都没有improvement。

  [early_stopping的参数解释](https://www.likecs.com/show-203904564.html)

#### early_stopping找出合适数量的epoch

如果epoch数量太少，网络有可能发生欠拟合（即对于定型数据的学习不够充分）；如果epoch数量太多，则有可能发生过拟合（即网络对定型数据中的“噪声”而非信号拟合）。

根本原因就是因为继续训练会导致测试集上的准确率下降。
**那继续训练导致测试准确率下降的原因**可能是

1. 过拟合 
2. 学习率过大导致不收敛 
3. 使用正则项的时候，Loss的减少可能不是因为准确率增加导致的，而是因为权重大小的降低。

##### **early_stopping的使用步骤：**

- 将数据分为训练集和验证集
- 每个epoch结束后（或每N个epoch后)： 在验证集上获取测试结果，随着epoch的增加，如果在验证集上发现测试误差上升，则停止训练；
- 将停止之后的权重作为网络的最终参数。

这种做法很符合直观感受，因为精度都不再提高了，在继续训练也是无益的，只会提高训练的时间。

###### 那么该做法的一个重点便是**怎样才认为验证集精度不再提高**了呢？

并不是说验证集精度一降下来便认为不再提高了，因为可能经过这个Epoch后，精度降低了，但是随后的Epoch又让精度又上去了，所以不能根据一两次的连续降低就判断不再提高。

一般的做法是，在训练的过程中，记录到目前为止最好的验证集精度，<u>**当连续10次Epoch（或者更多次）没达到最佳精度时，则可以认为精度不再提高了。**</u>

![image-20221029134111669](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221029134111669.png)

最优模型是在垂直虚线的时间点保存下来的模型，即处理测试集时准确率最高的模型。

###### 为什么能减小过拟合？

![image-20221029134300955](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221029134300955.png)

当还未在神经网络运行太多迭代过程的时候，w参数接近于0，因为随机初始化w值的时候，它的值是较小的随机值。当你开始迭代过程，w的值会变得越来越大。到后面时，w的值已经变得十分大了。

所以early stopping要做的就是在中间点停止迭代过程。我们将会得到一个中等大小的w参数，会得到与L2正则化相似的结果，选择了w参数较小的神经网络。

总结：随着epoch越大，w的值也会越大，而在中间点停止，会使w处于中等大小，得到与L2正则化相似的结果。

##### Early Stopping的优缺点

**优点：**只运行一次梯度下降，我们就可以找出w的较小值，中间值和较大值。而无需尝试L2正则化超级参数lambda的很多值。

**缺点**：不能独立地处理以上两个问题，使得要考虑的东西变得复杂。

没有采取不同的方式来解决优化损失函数和降低方差这两个问题，而是用一种方法同时解决两个问题 ，结果就是要考虑的东西变得更复杂。之所以不能独立地处理，因为如果你停止了优化代价函数，你可能会发现代价函数的值不够小，同时你又不希望过拟合。

###### 补充意见：

如果不用early stopping降低过拟合，另一种方法就是L2正则化，但需尝试L2正则化超级参数λ的很多值，个人（不是我）更倾向于使用L2正则化，尝试许多不同的λ值。

[深度学习-keras的EarlyStopping使用与技巧](https://blog.csdn.net/zwqjoy/article/details/86677030)

[pytorch-EarlyStopping](https://blog.csdn.net/qq_35054151/article/details/115986287)