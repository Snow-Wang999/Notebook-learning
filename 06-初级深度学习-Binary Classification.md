# Binary Classification-二进制分类

## 介绍

到目前为止，在本课程中，我们已经了解了神经网络如何解决回归问题。现在我们要将神经网络应用于另一个常见的机器学习问题：分类。到目前为止，我们学到的大部分内容仍然适用。主要区别在于我们使用的损失函数以及我们希望最终层产生什么样的输出。

### Binary Classification

**分类为两个类别之一**是一个常见的机器学习问题。您可能想要预测客户是否可能进行购买、信用卡交易是否具有欺诈性、深空信号是否显示新行星的证据或疾病的医学测试证据。这些都是二元分类问题。 

在您的原始数据中，类可能由 “Yes” 和 “No” 或 “Dog” 和 “Cat” 等字符串表示。在使用这些数据之前，我们将分配一个**类别标签**：一个类别为 0，另一个类别为 1。分配数字标签将数据置于神经网络可以使用的形式中。

### Accuracy and Cross-Entropy（准确性和交叉熵）

**准确性**是用于衡量分类问题成功的众多指标之一。 Accuracy 是正确预测与总预测的比率：

`accuracy = number_correct / total`

始终正确预测的模型的准确度得分为 1.0。在其他条件相同的情况下，只要数据集中的类以大致相同的频率出现，准确性就是一个合理的指标。

**准确性**（以及大多数其他分类指标）的问题在于它**不能用作损失函数**。 SGD 需要一个平滑变化的损失函数，但准确性（作为计数比率）会在“跳跃”中发生变化。因此，我们必须选择一个替代品作为损失函数。这个**替代品是交叉熵函数**(loss function)。

现在，回想一下损失函数在训练期间定义了网络的目标。通过回归，我们的目标是最小化预期结果和预测结果之间的距离。我们选择 MAE 来测量这个距离。

对于分类，我们想要的是**概率之间的距离**，这就是交叉熵提供的。交叉熵是一种度量从一个概率分布到另一个概率分布的距离。

![image-20221029160214596](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221029160214596.png)

Cross-entropy penalizes incorrect probability predictions.

交叉熵惩罚不正确的概率预测。

这个想法是我们希望我们的网络以概率 1.0 预测正确的类别。预测概率离 1.0 越远，交叉熵损失越大。

我们使用交叉熵的技术原因有点微妙，但是从本节中要带走的主要内容就是：使用交叉熵进行分类损失；您可能关心的其他指标（如准确性）往往会随之提高。

### 用 Sigmoid 函数做概率

Making Probabilities with the Sigmoid Function

交叉熵和准确度函数都需要概率作为输入，即从 0 到 1 的数字。为了将dense层产生的**实值输出转换为概率**，我们附加了一种新的激活函数，即 sigmoid 激活。

![image-20221029160910895](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221029160910895.png)

为了得到最终的类预测，我们定义了一个阈值（threshold）概率。通常这将是 0.5，因此四舍五入将为我们提供正确的类：低于 0.5 表示标签为 0 的类，而 0.5 或以上表示标签为 1 的类。Keras 默认使用 0.5 阈值及其准确度指标。

#### 补充：[激活函数](https://blog.csdn.net/qq_17614495/article/details/116359149)

![image-20221029163721657](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221029163721657.png)

##### 激活函数可以分为**两大类** ：

- **饱和**激活函数： sigmoid、 tanh
- **非饱和**激活函数: ReLU 、Leaky Relu  、ELU【指数线性单元】、PReLU【**参数化的**ReLU 】、RReLU【随机ReLU】

##### 相对于饱和激活函数，使用“**非饱和激活函数”的优势**在于两点：

1. 首先，“非饱和激活函数”能解决[深度](https://so.csdn.net/so/search?q=深度&spm=1001.2101.3001.7020)神经网络【层数非常多！！】的“**梯度消失”问题**，浅层网络【三五层那种】才用[sigmoid](https://so.csdn.net/so/search?q=sigmoid&spm=1001.2101.3001.7020) 作为激活函数。
2. 其次，它能**加快收敛速度**。

***

## Example - Binary Classification

电离层数据集包含从聚焦于地球大气层电离层的雷达信号中获得的特征。任务是确定信号是否显示某些物体的存在，或者只是空的空气。

```python
import pandas as pd
from IPython.display import display

ion = pd.read_csv('../input/dl-course-data/ion.csv', index_col=0)
display(ion.head())

df = ion.copy()
df['Class'] = df['Class'].map({'good': 0, 'bad': 1})

df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)

df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)
df_train.dropna(axis=1, inplace=True) # drop the empty feature in column 2
df_valid.dropna(axis=1, inplace=True)

X_train = df_train.drop('Class', axis=1)
X_valid = df_valid.drop('Class', axis=1)
y_train = df_train['Class']
y_valid = df_valid['Class']
```

![image-20221029161216146](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221029161216146.png)

我们将像为回归任务所做的那样定义我们的模型，但有一个例外。在最后一层包括一个“sigmoid”激活，以便模型产生类概率。

```python
我们将像为回归任务所做的那样定义我们的模型，但有一个例外。在最后一层包括一个“sigmoid”激活，以便模型产生类概率。
```

使用其编译方法将交叉熵损失和准确度度量添加到模型中。对于两类问题，请务必使用“二进制”版本。 （更多类的问题会略有不同。）Adam 优化器也适用于分类，所以我们会坚持下去。

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)
```

这个特定问题中的模型可能需要相当多的 epoch 才能完成训练，因此为了方便起见，我们将包括一个提前停止回调。

```python
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
)
```

我们将一如既往地查看学习曲线，并检查我们在验证集上获得的损失和准确性的最佳值。 （请记住，提前停止会将权重恢复为获得这些值的权重。）

```python
history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))
```

![image-20221029161527542](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221029161527542.png)

![image-20221029161536678](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221029161536678.png)

## Exercise: Binary Classification

### 加载环境库

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
from learntools.deep_learning_intro.ex6 import *
```

### 读取数据集

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

hotel = pd.read_csv('../input/dl-course-data/hotel.csv')
hotel.head()
```

32列

<img src="C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221029162911121.png" alt="image-20221029162911121"  />

```python
from sklearn.Imputer import 
transformer_cat = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="NA"),
    #OneHotEncoder(handle_unknown='ignore'),
)
# SimpleImputer是用来填充数据里面的缺失值的。
```

`SimpleImputer`，首先解释一下，这个类是用来填充数据里面的缺失值的。

[每天一点sklearn之SimpleImputer（9.19）](https://zhuanlan.zhihu.com/p/83173703?from_voters_page=true)

- strategy:

  也就是你采取什么样的策略对于每一列去填充空值，总共有4种选择。

  - mean-该列则由该列的均值填充
  - median-中位数
  - most_frequent-众数
  - constant，如果是constant,则可以将空值填充为自定义的值，这就要涉及到后面一个参数了，也就是fill_value。如果`strategy='constant'`， 则填充fill_value的值。

### 数据预处理

```python
X = hotel.copy()
y = X.pop('is_canceled')

X['arrival_date_month'] = \
    X['arrival_date_month'].map(
        {'January':1, 'February': 2, 'March':3,
         'April':4, 'May':5, 'June':6, 'July':7,
         'August':8, 'September':9, 'October':10,
         'November':11, 'December':12}
    )

features_num = [
    "lead_time", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "required_car_parking_spaces",
    "total_of_special_requests", "adr",
]
features_cat = [
    "hotel", "arrival_date_month", "meal",
    "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type",
]

transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"), # there are a few missing values
    StandardScaler(),
)
transformer_cat = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown='ignore'),
)

preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat),
)

# stratify - make sure classes are evenlly represented across splits
X_train, X_valid, y_train, y_valid = \
    train_test_split(X, y, stratify=y, train_size=0.75)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

input_shape = [X_train.shape[1]]
```

### 1. 创建模型

我们这次将使用的模型将同时具有批量标准化和 dropout 层。为了便于阅读，我们将图表分成块，但您可以像往常一样逐层定义它。 使用此图给出的架构定义模型：

![image-20221029170341818](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221029170341818.png)

```python
from tensorflow import keras
from tensorflow.keras import layers

# YOUR CODE HERE: define the model given in the diagram
model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(256,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1,activation='sigmoid'),
])

```

### Add Optimizer, Loss, and Metric-添加优化器，损失函数，和测量指标

现在使用 Adam 优化器和交叉熵损失和准确度指标的二进制版本编译模型。

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)
```

最后，运行此单元格来训练模型并查看学习曲线。它可能会运行大约 60 到 70 个 epoch，这可能需要一两分钟。

```python
early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=200,
    callbacks=[early_stopping],
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
```

![image-20221030132610395](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030132610395.png)

![image-20221030132618601](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030132618601.png)

```python
print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))
```

Best Validation Loss: 0.3500
Best Validation Accuracy: 0.8424

### 3. 训练和评估

您如何看待学习曲线？它看起来像模型欠拟合还是过拟合？交叉熵损失是一个很好的替代品吗？

```python
patience=30
```

![image-20221030140553203](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030140553203.png)

![image-20221030140607998](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221030140607998.png)

虽然我们可以看到训练损失继续下降，但提前停止回调防止了任何过度拟合。此外，准确率的上升速度与交叉熵下降的速度相同，因此最小化交叉熵似乎是一个很好的替代方案。总而言之，这次培训看起来很成功！

### 总结

凭借您的新技能，您已准备好采用更高级的应用程序，例如计算机视觉和情感分类。你接下来想做什么？ 

为什么不尝试我们的入门比赛之一？ 

- Classify images with TPUs in [**Petals to the Metal**](https://www.kaggle.com/c/tpu-getting-started)-在 Petals to the Metal 中使用 TPU 对图像进行分类 
- Create art with GANs in [**I'm Something of a Painter Myself**](https://www.kaggle.com/c/gan-getting-started)-在 I'm Something of a Painter Myself 中使用 GAN 创作艺术 
- Classify Tweets in [**Real or Not? NLP with Disaster Tweets**](https://www.kaggle.com/c/nlp-getting-started)-对推文进行分类是否真实？带有灾难推文的 NLP 
- Detect contradiction and entailment in [**Contradictory, My Dear Watson**](https://www.kaggle.com/c/contradictory-my-dear-watson)-在矛盾中发现矛盾和蕴涵，我亲爱的沃森