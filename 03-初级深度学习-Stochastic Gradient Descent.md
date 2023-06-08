# 随机梯度下降（Stochastic Gradient Descent）

在前两节课中，我们学习了如何从密集层的堆栈中构建全连接网络。首次创建时，网络的所有权重都是随机设置的——网络还不“知道”任何东西。在本课中，我们将了解**如何训练神经网络**；我们将看到**神经网络是如何学习的**。 

与所有机器学习任务一样，我们从数据训练开始，数据由特征（输入）和预期目标（输出）组成。训练网络是通过调整权重使其特征转换为预期目标。它的权重代表这些特征与预期目标之间的关系。例如，谷物的数据集中，通过改变每种谷物的“糖”、“纤维”和“蛋白质”的含量，来预测该谷物的“卡路里”。

> **除了训练数据，我们还需要两件事：** 
>
> - 衡量网络预测有多好的“损失函数”。 
> - 一个“优化器”，可以告诉网络如何改变其权重。

---

## 概念学习

### 损失函数（The  Loss Function）

我们已经看到了如何为网络设计架构，但我们还没有看到如何告诉网络要解决什么问题。这是损失函数的工作。

损失函数是**衡量目标的真实值与模型的预测值之间的差异**。

> |`y_true` - `y_pred`|= ？

回归问题的常见损失函数是平均绝对误差 (**MAE**-mean absolute error)。对于每个预测 y_pred，MAE 通过绝对差 abs(y_true - y_pred) 测量与真实目标 y_true 的差异。 

数据集上的总 MAE 损失是所有这些绝对差异的平均值。

![A graph depicting error bars from data points to the fitted line..](https://i.imgur.com/VDcvkZN.png)

除了 MAE，您可能会看到回归问题的其他损失函数是均方误差 (**MSE**-mean-squared error) 或 **Huber** 损失（在 Keras 中都可用）。 

在训练期间，模型将使用损失函数作为找到正确权重值的指南（损失越低越好）。换句话说，损失函数告诉网络它的目标。

***

### 优化器（The Optimizer）

**随机梯度下降（Stochastic Gradient Descent）**

优化器是一种以**最小化损失**调整权重的算法。

实际上，深度学习中使用的所有优化算法都属于***随机梯度下降的家族***。它们是逐步训练网络的迭代算法。

> 训练的一个步骤是这样的：
>
> - 采样一些训练数据并通过网络运行以进行预测。
> - 测量预测值和真实值之间的损失。 
> - 最后，向使损失更小的方向调整权重。

然后一遍又一遍地这样做，直到损失尽可能小（或者直到它不会进一步减少。）

![Fitting a line batch by batch. The loss decreases and the weights approach their true values.](https://i.imgur.com/rFI1tIk.gif)

每次迭代的训练数据样本称为 **minibatch**（或“batch”），而完整的一轮训练数据称为 **epoch**。您训练的 epoch 数是网络将看到的每个训练示例的次数。

> 动画显示了第 1 课中的线性模型正在使用 SGD 进行训练。淡红点描绘了整个训练集，而实心红点是小批量。每次 SGD 看到一个新的小批量时，它都会将权重（w 斜率和 b y 截距）移向该批次上的正确值。一批接一批，这条线最终会收敛到最合适的位置。您可以看到，随着权重越来越接近其真实值，损失变得越来越小。

***

### 学习率和批量大小

请注意，该线仅在每个批次的方向上进行微小的移动（而不是一直移动）。这些变化的大小由学习率决定。

**较小的学习率**意味着网络需要在其权重收敛到最佳值之前看到**更多的小批量**。 

学习率和小批量的大小是对 SGD 训练进行方式影响最大的两个参数。它们的相互作用通常是微妙的，并且这些参数的正确选择并不总是显而易见的。 （我们将在练习中探讨这些影响。） 

幸运的是，对于大多数工作来说，不需要进行广泛的超参数搜索即可获得令人满意的结果。 **Adam** 是一种 SGD 算法，它具有**<u>自适应学习率</u>**，使其适用于大多数问题而无需任何参数调整（从某种意义上说，它是“自我调整”）。 Adam 是一个伟大的通用优化器。

***

### 添加损失和优化器

定义模型后，您可以使用模型的`compile`方法添加损失函数和优化器：

```python
model.compile(
    optimizer="adam",
    loss="mae",
)
```

您可以看到 `Keras` 会在模型训练时让您了解损失的最新信息。

现在我们准备开始训练了！我们告诉 `Keras` 一次向优化器提供 256 行训练数据（batch_size），并在整个数据集（epochs）中执行 10 次。 

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10,
)
```

通常，查看损失的更好方法是绘制它。 fit 方法实际上记录了在 History 对象中训练期间产生的损失。我们将数据转换为 Pandas 数据框，这使得绘图变得容易。

```python
import pandas as pd

# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# use Pandas native plot method
history_df['loss'].plot();
```

![image-20221027124329083](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027124329083.png)

***

## Exercise: Stochastic Gradient Descent

### 1. 介绍 

在本练习中，您将在 Fuel Economy 数据集上训练神经网络，然后探索学习率和批量大小对 SGD 的影响。 

```python
# Setup plotting
import matplotlib.pyplot as plt
from learntools.deep_learning_intro.dltools import animate_sgd
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
#动画
plt.rc('animation', html='html5')

# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning_intro.ex3 import *
```

### 2. 数据集

在燃油经济性数据集中，您的任务是根据发动机类型或制造年份等特征预测汽车的燃油经济性。 

首先通过运行下面的单元格来加载数据集。

```python
import numpy as np
import pandas as pd
#scikit-learn中的数据预处理
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

fuel = pd.read_csv('../input/dl-course-data/fuel.csv')
fuel.head()
```

![image-20221027125004150](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027125004150.png)

```python
X = fuel.copy()
# Remove target
y = X.pop('FE')
y.head()
```

![image-20221027125107528](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027125107528.png)

通过对燃油经济数据的分析，对数据进行标准化和把文本数据转换成数值型，用log函数快速收敛标签数据

### 3. 数据预处理

```python
preprocessor = make_column_transformer(
    (StandardScaler(),
    make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False),
     make_column_selector(dtype_include=object)),
)

X = preprocessor.fit_transform(X)
y = np.log(y) # log transform target instead of standardizing

input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))
```

Input shape: [50]

```python
# Uncomment to see original data
# fuel.head()
# Uncomment to see processed features
pd.DataFrame(X[:10,:]).head()
```

![image-20221027125628033](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027125628033.png)

### 4. 创建模型

创建一个输入层（50个神经元），第一个隐藏层（输出为128个神经元，激活函数为relu），第二个隐藏层（输出为128个神经元，激活函数为relu），第三个隐藏层（输出为64个神经元，激活函数为relu），输出层为一个神经元的模型。

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),    
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])
```

### 5. 损失函数和优化器

使用模型的编译方法，添加 Adam 优化器和 MAE 损失。

```python
model.compile(loss='mae',optimizer='adam')
```

### 6. 训练模型

定义模型并使用损失和优化器对其进行编译后，您就可以进行训练了。将网络训练 200 轮（epochs=200），批量大小为 128（batch=128）。输入数据为 X，目标为 y。

```python
history = model.fit(
    X,y,
    batch_size=128,
    epochs=200
)
```

```python
import pandas as pd

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5. You can change this to get a different view.
history_df.loc[5:, ['loss']].plot();
```

![image-20221027130730713](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027130730713.png)

### 7. 评估训练模型（evaluate model）

```python
history_1 = model.fit(X,y,batch_size=128,epochs=300)
```

```python
import pandas as pd

history_df_1 = pd.DataFrame(history_1.history)
# Start the plot at epoch 5. You can change this to get a different view.
history_df_1.loc[1:, ['loss']].plot();
```

![image-20221027130939653](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027130939653.png)

### 8. 参数调节

通过学习率和批量大小，您可以控制：

- 训练一个模型需要多长时间 

- 学习曲线有多嘈杂 

- 损失有多小 

为了更好地理解这两个参数，我们将看看线性模型，我们最简单的神经网络。只有一个权重和一个偏差，更容易看出参数变化的影响。 

下一个单元格将生成类似于教程中的动画。更改 learning_rate、batch_size 和 num_examples（多少个数据点）的值，然后运行该单元格。 （可能需要一两分钟。）尝试以下组合，或尝试一些您自己的组合：

| `learning_rate` | `batch_size` | `num_examples` |
| --------------- | ------------ | -------------- |
| 0.05            | 32           | 256            |
| 0.05            | 2            | 256            |
| 0.05            | 128          | 256            |
| 0.02            | 32           | 256            |
| 0.2             | 32           | 256            |
| 1.0             | 32           | 256            |
| 0.9             | 4096         | 8192           |
| 0.99            | 4096         | 8192           |

#### Learning Rate and Batch Size

```python
# YOUR CODE HERE: Experiment with different values for the learning rate, batch size, and number of examples
learning_rate = 0.05
batch_size = 32
num_examples = 256

animate_sgd(
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_examples=num_examples,
    # You can also change these, if you like
    steps=50, # total training steps (batches seen)
    true_w=3.0, # the slope of the data
    true_b=2.0, # the bias of the data
)
```

![image-20221027141856625](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027141856625.png)



```python
learning_rate = 0.05
batch_size = 2
num_examples = 256
```

![image-20221027142049934](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027142049934.png)

```python
learning_rate = 0.05
batch_size = 128
num_examples = 256
```

![image-20221027142109998](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027142109998.png)

```python
learning_rate = 0.02
batch_size = 32
num_examples = 256
```

![image-20221027142136372](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027142136372.png)

```python
learning_rate = 0.2
batch_size = 32
num_examples = 256
```

![image-20221027142207870](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027142207870.png)

```python
learning_rate = 1.0
batch_size = 32
num_examples = 256
```

![image-20221027142254314](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027142254314.png)

```python
learning_rate = 0.9
batch_size = 4096
num_examples = 8192
```

![image-20221027142322334](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027142322334.png)

```python
learning_rate = 0.99
batch_size = 4096
num_examples = 8192
```

![image-20221027142351994](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221027142351994.png)

您可能已经看到，较小的批次大小会产生更嘈杂的权重更新和损失曲线。这是因为每个批次都是一个小数据样本，而较小的样本往往会给出更嘈杂的估计。较小的批次可能会产生“平均”效果，但这可能是有益的。 

较小的学习率使更新更小，训练需要更长的时间才能收敛。大的学习率可以加快训练，但也不要“安定”到最低限度。当学习率太大时，训练可能会完全失败。 （尝试将学习率设置为较大的值，例如 0.99 来查看。）

#### 结论：

- 批次大小（`batch_size`）太小，会导致更嘈杂的权重更新和损失曲线；太大会造成训练时间的延长。

- 学习率（`learning_rate`）太小，效率低；学习率太大，无法收敛。

- 数据点（`num_examples`）太多，容易造成训练时间过长，但是曲线更光滑。





***

### 补充：数据预处理

数据预处理的工具有许多，在我看来主要有两种：

- `pandas`数据预处理-[Pandas数据处理与分析](https://blog.csdn.net/qq_40195360/article/details/84570503?spm=1001.2014.3001.5502)

- `scikit-learn`中的`sklearn.preprocessing`数据预处理。-[数据预处理（`sklearn.preprocessing`）](https://blog.csdn.net/qq_40195360/article/details/88378248)

此处，主要介绍`sklearn.preprocessing`。

1. #### 标准化

   1. `StandardScaler`

      将特征数据的分布调整成标准正太分布，也叫高斯分布，也就是使得数据的均值为0，方差为1.

   5. 补充：

      - 大多数机器学习算法中，会选择`StandardScaler`来进行特征缩放，因为<u>**`MinMaxScaler`对异常值非常敏感**</u>。在PCA，聚类，逻辑回归，支持向量机，神经网络这些算法中，`StandardScaler`往往是最好的选择。

   ***

2. #### 非线性变换

   ***

3. #### 归一化（ Normalizer）

   归一化的目的：

   - 加快了梯度下降求最优解的速度
   - 有可能提高精度

   公式：对于整数p>1，
   $$
   Lp norm = \sum{(|vector|^p)(\frac{1}{p})}
   $$
   归一化是缩放单个样本以具有单位范数的过程，这里的”范数”，可以使用L1或L2范数。如果你计划使用二次形式(如点积或任何其他核函数)来量化任何样本间的相似度，则此过程将非常有用。

   这个观点基于 ***向量空间模型(Vector Space Model)***，经常在**文本分类**和**内容聚类**中使用。
   ```python
   sklearn.preprocessing.Normalizer(norm='l2',copy = True)
   #norm : ‘l1’, ‘l2’, or ‘max’, optional (‘l2’ by default)
   ```

   **区别**：正则化（Regularization）

   目的：防止模型出现过拟合，导致模型“泛化”能力太差。

   正则化-规则化-给需要训练的目标函数加上一些规则限制。

   [【直观详解】什么是正则化](https://blog.csdn.net/kdongyi/article/details/83932945)

   [【深度学习概念区分】Normalization vs. Regularization](https://zhuanlan.zhihu.com/p/477747129)

   ****

4. #### 编码分类特征

   **目的：将文字型数据转换为数值型**

   **将文字型数据转换为数值型**。

   1. `OneHotEncoder`

      类别`OrdinalEncoder`可以用来处理有序变量，但对于**名义变量**，我们只有使用哑变量的方式来处理，才能够尽量向算法传达最准确的信息。

      ```python
      import pandas as pd
      from sklearn.preprocessing import OneHotEncoder
      X = pd.DataFrame(['male','female','male','female','female','female'],columns=['sex'])
      X = OneHotEncoder().fit_transform(X).toarray()
      ```


   ***

5. #### 离散化

6. #### 缺失值处理(`Imputer`)

7. #### 生成多项式特征(`PolynomialFeatures`)

8. #### 自定义转换器(`FunctionTransformer`)


对于新的知识个人建议去scikit-learn官网中进行查找阅读。

