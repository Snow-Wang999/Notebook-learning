# 6-1-模式识别-对抗性机器学习（Adversarial machine learning）

工作目的：熟悉神经网络攻击算法。

## 介绍

机器学习已被广泛应用于人类的许多领域。机器学习算法用于识别交通标志、过滤垃圾邮件、面部识别和股票交易决策。在算法做出重要决策的系统中，需要**确保对欺骗的抵抗力**。*Adversarial Machine Learning*正在成为软件行业的一个重要领域。近年来，在其产品中大量使用机器学习算法的大型公司越来越多地面临Adversarial攻击。谷歌、微软和IBM已经开始投资于机器学习系统的安全性。

**对抗性机器学习（Adversarial Machine Learning）**是一种机器学习技术，旨在通过**提供误导性数据来欺骗机器学习模型**。它包括创建和识别Adversarial示例，这些示例是专门为欺骗分类器而创建的输入数据。这种攻击在图像分类和垃圾邮件检测等领域得到了广泛的研究。

**Adversarial攻击**是一种创建Adversarial实例的方法。Adversarial示例是输入算法的向量，其中算法输出不正确。

与老师一起学习分类是机器学习的任务之一。在给定的问题中，使用对象-标记对，模型必须学会预测新对象的值。

如果从几何学的角度来考虑这个问题，则必须对空间进行分割，以便在新对象上预测“正确”类。此外，如果我们有一个总的数据集（例如，对于MNIST手写数字集，所有数字都有各种各样的图像），那么这个超平面可以完美地绘制，前提是类的可分割性。但是，由于通常不存在一个整体，因此我们使用机器学习算法来解决这个问题-使用我们拥有的数据尽可能精确地接近“理想”超平面。

超平面与理想平面的任何偏差都会产生一些间隙，一旦进入，物体就不正确地分类。这就是为什么像熊猫这样的例子被归类为长臂猿的原因。攻击者的任务是更改对象参数向量，使其进入空隙。

## 攻击分类

对机器学习模型的攻击可分为两类：WhiteBox（WB）和BlackBox（BB）。WhiteBox攻击是一种场景，其中攻击者可以完全访问目标模型，包括模型体系结构及其参数。Blackbox攻击是一种情况，攻击者无法访问模型，只能监视目标模型的输出。您还可以选择graybox攻击的变体，即攻击者不知道有关训练模型的信息，但有关于算法类型及其超参数的信息。然而，这种类型并没有被单独分类，因为额外的信息不足以切换到WB，这意味着它只是BB攻击的额外信息集。

有许多不同的攻击可以用来对付机器学习算法。其中许多与深度学习算法和传统模型（如支持向量机（SVM）和线性回归）一起工作。Adversarial攻击主要分为以下几类：

- 中毒袭击（Poisoning Attacks）
- 躲避攻击 （Evasion Attacks）
- 模型提取（Model Extraction）

当实施中毒攻击时，攻击者会攻击培训数据或其标签，以在部署期间破坏模型。因此，中毒本质上是对学习**数据Adversarial污染**。由于机器学习系统可以使用运行时收集的数据进行重新训练，攻击者可能会在运行时输入恶意样本来毒害数据，从而破坏或影响重新学习。

Evasion攻击是最常见和研究最多的攻击类型。攻击者修改对模型的请求（requests）以获得所需的结果。这是通过选择模型查询（query）和监视输出（monitoring output）来完成的。由于这种攻击是针对已经运行的模型进行的，因此它们是最实用和常用的攻击。Adversarial实例的形成方式是为了避免被发现，它们被归类为可接受的。

模型提取（Model Extraction）是Blackbox攻击的一种场景，攻击者检查机器学习系统，以恢复模型或检索训练它的数据。当学习数据或模型本身是保密和保密的时，这一点尤为重要。

让我们看看最流行的Evasion攻击示例。

### 有限内存BFGS（L-BFGS）

Broyden–Fletcher–Goldfarb–Channo有限内存算法（L-BFGS）是一种基于梯度的非线性数值优化算法，可以**最小化添加到图像中的扰动次数**。此攻击旨在将损失函数最小化到攻击者所需的目标类，并将更改最小化。这种类型的攻击在创建对抗性实例时非常有效，但对计算能力要求很高，因为它是一种具有块限制的优化方法。这种方法既费时又不切实际。

算法的主要任务是解决以下优化问题：
$$
minimize\  c\cdot||x-x'||^2_2+J(\theta,x',y'),
$$
![image-20221221113050693](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221113050693.png)

> 四条竖线表示范数。按照你文中的表示，应该指的是向量范数。
>
> - 下面的数字2表示向量的2-范数。即表示向量元素绝对值的平方和再开方。
> - 上面的数字2表示范数的平方.

其中$x$是输入数据，

$x'$ 是相应的Adversarial示例，

$y'$ 是Adversarial示例的目标类标记，

$J$ ：损失函数

### 快速gradient sign（FGSM）

一种简单而快速的梯度方法用于创建Adversarial示例，以最小化向图像的任何像素添加的最大扰动量，从而导致不正确的分类。最著名的攻击类型在算法的第一阶段，根据目标图像的真实类标签计算损失函数及其梯度值，然后根据梯度符号生成输出Adversarial示例。该算法可以用以下表达式表示：
$$
x_{adv}=x+\varepsilon \times sign(\nabla_xJ(\theta,x,y))
$$
其中 $x_{adv}$ ：Adversarial图像，

$x$：输入图像，

$y$：真实图像标记，

$\varepsilon$：最小化原始图像ε变化的系数，

$\theta$：模型参数，

$J$：损失函数。

$sign()$：符号函数。

![image-20221221113803913](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221113803913.png)

![image-20221221101852374](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221101852374.png)

上图FGSM算法原理说明

### 雅可比值图攻击（JSMA）

最初用于可视化机器学习模型的预测过程。权重图评估每个输入特征（例如图像的单个像素）对输入向量与给定类关联模型预测结果的影响。该方法基于计算每个参数的导数，然后绘制梯度图。然后，地图分析每个参数对输出结果的贡献。此方法的优点是对参数的更改最小（因为更改类所需的参数仅更改）。

### Deepfool攻击

这种创建Adversarial样本的非目标方法旨在最小化扰动样本和原始样本之间的欧氏距离。对类之间的决策边界进行评估，并迭代地添加扰动。该算法找到两类边界的最近超平面，并将原始图像x移到与该超平面正交的向量上。90%的MNIST测试图像可以通过仅将图像更改为0.1的∥∞规范来“转换”到另一类。

### Carlini-Wagner攻击（C&W）

Carlini-Wagner攻击是针对L-BFGS和FGSM类型的Adversarial攻击的防御解决方案。建议使用所有有目标类的概率间隙最小化，而不是最小化L-BFGS方法中提出的函数。通过这种优化，您可以找到抵御Adversarial攻击的图像。

### 生成式对抗网络（GAN）

当两个神经网络相互竞争时，GAN网络被用来创建Adversarial攻击。因此，一个作为发生器，另一个作为鉴别器。两个网络玩一个零和游戏，发生器试图生成歧视者错误分类的样本。同时，鉴别器试图区分真实的样本和生成器创建的样本。这种方法的优点是生成不同于学习中使用的示例，但GAN学习需要大量的计算资源，并且可能非常不稳定。

### 零阶优化攻击（Zoo）

Zoo方法允许在不访问分类器的情况下估计分类器的梯度，使其成为Blackbox攻击的理想选择。不需要替代模型培训或分类器信息。缺点是需要对目标分类器进行大量查询。

### Python

有许多用于研究Python机器学习模型攻击的库，包括[Foolbox](https://foolbox.readthedocs.io/en/v3.3.3/modules/attacks.html)(https://pypi.org/project/foolbox/)、CleverHans和Art-IBM([adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox))。在本实验室工作中，将考虑使用Foolbox库的示例。Foolbox是一组Python工具，用于创建欺骗神经网络的Adversarial示例。首先，导入攻击ImageNet上预定义的RESNET50网络所需的模块。

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input,
decode_predictions
import numpy as np
import matplotlib.pyplot as plt
from foolbox import TensorFlowModel, Model
from foolbox.attacks import LinfFastGradientAttack
```

接下来，我们将下载神经网络本身：

```python
model = tf.keras.applications.ResNet50(weights="imagenet")
```

使用一捆香蕉的照片作为攻击的图像。下载图片并检查模型的工作原理：

```python
img_path = 'banana_origin_224.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
plt.figure()
plt.imshow(img)
plt.title('Banana image')
plt.show()
print('Predicted:', decode_predictions(preds, top=3)[0])
```

![image-20221221103116555](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221103116555.png)

```
Predicted: [('n07753592', 'banana', 0.99861276), ('n07753113',
'fig', 0.0004591437), ('n07749582', 'lemon', 0.00022873387)]
```

神经网络成功地识别出图像中的香蕉（网络的信心接近100%）。

因为在本例中，使用RESNET50网络指定给定网络体系结构所需的图像预处理参数，即BGR颜色通道的顺序和[103.939、116.779、123.68]上的规范化。

```python
pre = dict(flip_axis=-1, mean=[103.939, 116.779, 123.68]) # RGB to BGR
fmodel: Model = TensorFlowModel(model, bounds=(0, 255), preprocessing=pre)
```

然后，将FGSM攻击算法应用于神经网络，指定适用于Adversarial图像的允许扰动范围：

```python
# apply the attack
attack = LinfFastGradientAttack()
# так как преобразования к изображению применяются внутри модели foolbox,
# еще раз поместим его в переменную х
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
banana_image, banana_label = tf.convert_to_tensor(x),
tf.convert_to_tensor(tf.cast([954], tf.int64))
#Заданим значения допустимых возмущений
epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
raw_advs, clipped_advs, success = attack(fmodel, banana_image, banana_label,
epsilons=epsilons)
```

然后我们将这些图像和神经网络的分类结果显示出来：

```python
for sample in raw_advs:
	preds = model.predict(sample)
	print('Predicted:', decode_predictions(preds, top=3)[0])
	plt.figure()
	plt.imshow(sample[0]/ 255)
	plt.title('Adversarial banana image')
	plt.show()
```

```
Predicted: [('n07753592', 'banana', 0.44590968), ('n03825788', 'nipple', 0.172575),
('n04522168', 'vase', 0.052172087)]
```

![image-20221221103425019](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221103425019.png)

```
Predicted: [('n07753592', 'banana', 0.44580537), ('n03825788', 'nipple', 0.17252426),
('n04522168', 'vase', 0.05218799)]
```

![image-20221221103459784](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221103459784.png)

```
Predicted: [('n07753592', 'banana', 0.44485044), ('n03825788', 'nipple', 0.17207411),
('n04522168', 'vase', 0.05233555)]
```

![image-20221221103518772](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221103518772.png)

```
Predicted: [('n07753592', 'banana', 0.4424706), ('n03825788', 'nipple', 0.17113739),
('n04522168', 'vase', 0.05267355)]
```

![image-20221221103540586](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221103540586.png)

```
Predicted: [('n07753592', 'banana', 0.43487912), ('n03825788', 'nipple', 0.1660144),
('n04522168', 'vase', 0.05391883)]
```

![image-20221221103640214](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221103640214.png)

```
Predicted: [('n07753592', 'banana', 0.42326275), ('n03825788', 'nipple', 0.14756009),
('n04522168', 'vase', 0.05571003)]
```

![image-20221221103717399](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221103717399.png)

```
Predicted: [('n07753592', 'banana', 0.40108785), ('n03825788', 'nipple', 0.13709487),
('n04522168', 'vase', 0.05890414)]
```

![image-20221221103738281](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221103738281.png)

```
Predicted: [('n07753592', 'banana', 0.35506487), ('n03825788', 'nipple', 0.10025605),
('n04131690', 'saltshaker', 0.073854975)]
```

![image-20221221103747120](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221103747120.png)

```
Predicted: [('n07753592', 'banana', 0.28482595), ('n04131690', 'saltshaker', 0.10566427),
('n07753113', 'fig', 0.10099223)]
```

![image-20221221103802575](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221103802575.png)

```
Predicted: [('n07753592', 'banana', 0.113057286), ('n02808304', 'bath_towel',
0.10506969), ('n03775071', 'mitten', 0.079404004)]
```

![image-20221221103812121](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221103812121.png)

```
Predicted: [('n03775071', 'mitten', 0.37582743), ('n02808304', 'bath_towel', 0.09073115),
('n03825788', 'nipple', 0.07180546)]
```

![image-20221221103820088](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221103820088.png)

正如代码块的输出所示，随着ε系数的增加，神经网络对图像中香蕉的信心也会降低。在这种情况下，对ε的修改很少。当图像中的ε值为10时，仍然可以观察到一捆香蕉，但神经网络已经将其识别为手套。

## 实验室工作任务

使用`foolbox.attacks`模块方法实现多种攻击方法（3-4）。攻击和预定义的模型，简要描述所选的攻击方法，获得的图像示例，比较结果。

## Adversarial ML -01

### 加载库foolbox

Скачать foolbox

```python
#install a foolbox module
!pip install foolbox
```

```
Collecting foolbox
  Downloading foolbox-3.3.3-py3-none-any.whl (1.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 1.8 MB/s eta 0:00:0000:0100:010m
Requirement already satisfied: requests>=2.24.0 in /opt/conda/lib/python3.7/site-packages (from foolbox) (2.28.1)
Requirement already satisfied: numpy in /opt/conda/lib/python3.7/site-packages (from foolbox) (1.21.6)
Requirement already satisfied: scipy in /opt/conda/lib/python3.7/site-packages (from foolbox) (1.7.3)
Requirement already satisfied: GitPython>=3.0.7 in /opt/conda/lib/python3.7/site-packages (from foolbox) (3.1.27)
Requirement already satisfied: setuptools in /opt/conda/lib/python3.7/site-packages (from foolbox) (59.8.0)
Requirement already satisfied: typing-extensions>=3.7.4.1 in /opt/conda/lib/python3.7/site-packages (from foolbox) (4.4.0)
Collecting eagerpy>=0.30.0
  Downloading eagerpy-0.30.0-py3-none-any.whl (31 kB)
Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/conda/lib/python3.7/site-packages (from GitPython>=3.0.7->foolbox) (4.0.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.7/site-packages (from requests>=2.24.0->foolbox) (2022.9.24)
Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.7/site-packages (from requests>=2.24.0->foolbox) (3.3)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /opt/conda/lib/python3.7/site-packages (from requests>=2.24.0->foolbox) (1.26.12)
Requirement already satisfied: charset-normalizer<3,>=2 in /opt/conda/lib/python3.7/site-packages (from requests>=2.24.0->foolbox) (2.1.0)
Requirement already satisfied: smmap<6,>=3.0.1 in /opt/conda/lib/python3.7/site-packages (from gitdb<5,>=4.0.1->GitPython>=3.0.7->foolbox) (3.0.5)
Installing collected packages: eagerpy, foolbox
Successfully installed eagerpy-0.30.0 foolbox-3.3.3
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv class="ansi-yellow-fg">
```

```python
import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image #keras图像预处理
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
from foolbox import TensorFlowModel, Model
from foolbox.attacks import LinfFastGradientAttack, LinfDeepFoolAttack, L2CarliniWagnerAttack, EADAttack, BoundaryAttack
```

### 加载预训练resnet50模型

Загрузка предварительно обученной модели resnet50

```python
model = tf.keras.applications.ResNet50(weights="imagenet")
```

```
2022-12-24 08:53:25.275865: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5
102973440/102967424 [==============================] - 4s 0us/step
102981632/102967424 [==============================] - 4s 0us/step
```

```python
# RGB to BGR
pre = dict(flip_axis=-1, mean=[103.939, 116.779, 123.68])  
# 对tensorflow model 进行预处理
fmodel: Model = TensorFlowModel(model, bounds=(0, 255), preprocessing=pre) 
```

### 加载图片

Загрузить изображение

```python
img_path = '/kaggle/input/banana/banana_origin_224.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
plt.figure()
plt.imshow(img)
plt.title('Banana image')
plt.show()
print('Predicted:', decode_predictions(preds, top=3)[0])
```

```
2022-12-24 08:58:56.126645: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
```

![image-20221221185849087](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221185849087.png)

```
Downloading data from https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
40960/35363 [==================================] - 0s 0us/step
49152/35363 [=========================================] - 0s 0us/step
Predicted: [('n07753592', 'banana', 0.99861276), ('n07753113', 'fig', 0.00045913932), ('n07749582', 'lemon', 0.00022873365)]
```

### 数据转换（图片到numpy到tensor）

Преобразование данных (изображение в numpy в tensor)

```python
# Since the image conversion is applied in the Foolbox model, it is put in the variable x again.
# 由于图像转换是在Foolbox模型中应用的，所以再次将其放在变量x中。
# image.load_img()只是加载了一个文件，没有形成numpy数组。
# 下面的numpy数组是通过image.img_to_array()的函数形成的
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
# np.expand_dims的作用是增加一个维度
x = np.expand_dims(x, axis=0)
# 把numpy数据转为tensor数据，tf.cast转换数据类型
banana_image, banana_label = tf.convert_to_tensor(x), tf.convert_to_tensor(tf.cast([954], tf.int64))
```

#### np.expand_dims(x, axis)

**作用**：增加一个维度。

现在我们假设有一个数组A，数组A是一个两行三列的矩阵。大小我们记成（2,3），是 $2 \times 3$ 的二维矩阵。

- 如果设置 `axis=0`，那A矩阵的大小就变成了 $(1,2,3)$ ，变成了一个 $1 \times 2 \times 3$ 的三维矩阵。
- 如果设置 `axis=1`，那A矩阵的大小就变成了 $(2,1,3)$ ，变成了一个 $2 \times 1 \times 3$ 的三维矩阵。
- 如果设置 `axis=2`，那A矩阵的大小就变成了 $(2,3,1)$ ，变成了一个 $2 \times 3 \times 1$ 的三维矩阵。

#### tf.cast(x, dtype, name=None)

**释义**：数据类型转换

- x，输入张量
- dtype，转换数据类型
- name，名称

### 攻击图片方法设置

https://foolbox.readthedocs.io/en/v3.3.3/modules/attacks.html

#### 快速梯度符号法（FGSM）

https://jiuaidu.com/jianzhan/869434/

理论：

![image-20221224230657031](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224230657031.png)

Goodfellow等人（2014）假设，对抗性示例是由神经网络过于线性导致的，并且容易受到线性失真的影响。为了利用这种漏洞，他们创建了快速梯度标记法（FGSM）攻击。攻击的思想是在输入像素的损失函数梯度的符号方向上扰动图像x。用于计算FGSM的函数是
$$
x′=x+\epsilon \operatorname{sgn}\left(\nabla_{x} \mathcal{L}( x, y_l)\right)
$$
其中$L_f(x，y)$是给定x及其正确标签$y_l$的模型f的损失。定义了x′和x之间的$l∞$距离。与其他方法（如l-BFGS、BIM（见第3.1.5节）和C&W（见第3.1.6节）攻击）相比，这种创建对抗性神经网络的方法非常快速，计算成本低。然而，它也创造了较弱的对抗性例子。

```python
# apply the attack
attack = LinfFastGradientAttack()

#Set allowable disturbance value
#Заданим значения допустимых возмущений
epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]

t1 = time.time()
raw_advs, clipped_advs, success = attack(fmodel, banana_image, banana_label, epsilons=epsilons)
t2 = time.time()
total_time = (t2 - t1)
print("Total Time: ", total_time, "seconds")
```

```
Total Time:  1.819326400756836 seconds
```

```python
for sample in raw_advs:
  preds = model.predict(sample)
  print('Predicted:', decode_predictions(preds, top=3)[0])
  plt.figure()
  plt.imshow(sample[0]/ 255)
  plt.title('Adversarial banana image')
  plt.show()
```

```
Predicted: [('n07753592', 'banana', 0.44591025), ('n03825788', 'nipple', 0.17257455), ('n04522168', 'vase', 0.052171905)]
Predicted: [('n07753592', 'banana', 0.4458071), ('n03825788', 'nipple', 0.17252246), ('n04522168', 'vase', 0.052187994)]
Predicted: [('n07753592', 'banana', 0.44484785), ('n03825788', 'nipple', 0.17207459), ('n04522168', 'vase', 0.052335992)]
Predicted: [('n07753592', 'banana', 0.44246757), ('n03825788', 'nipple', 0.1711364), ('n04522168', 'vase', 0.05267459)]
Predicted: [('n07753592', 'banana', 0.4348748), ('n03825788', 'nipple', 0.16601036), ('n04522168', 'vase', 0.053921584)]
Predicted: [('n07753592', 'banana', 0.42325485), ('n03825788', 'nipple', 0.14754747), ('n04522168', 'vase', 0.055720575)]
Predicted: [('n07753592', 'banana', 0.40104732), ('n03825788', 'nipple', 0.13710597), ('n04522168', 'vase', 0.058917735)]
Predicted: [('n07753592', 'banana', 0.35499972), ('n03825788', 'nipple', 0.100291006), ('n04131690', 'saltshaker', 0.07383629)]
Predicted: [('n07753592', 'banana', 0.28499323), ('n04131690', 'saltshaker', 0.10560722), ('n07753113', 'fig', 0.10102505)]
Predicted: [('n07753592', 'banana', 0.11342495), ('n02808304', 'bath_towel', 0.105103604), ('n03775071', 'mitten', 0.07937481)]
Predicted: [('n03775071', 'mitten', 0.37477252), ('n02808304', 'bath_towel', 0.09061144), ('n03825788', 'nipple', 0.07166909)]
```

![image-20221224223013612](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224223013612.png)

![image-20221224223038467](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224223038467.png)

![image-20221224223109728](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224223109728.png)

![image-20221224223117817](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224223117817.png)

![image-20221224223128434](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224223128434.png)

![image-20221224223152064](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224223152064.png)

![image-20221224223238812](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224223238812.png)

![image-20221224223248130](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224223248130.png)

![image-20221224223258547](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224223258547.png)

![image-20221224223310957](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224223310957.png)

![image-20221224223323188](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224223323188.png)

![image-20221224223334135](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224223334135.png)

#### Attack 1:Deep Fool Attack

理论：https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/108067350

Moosavi Dezfoloi等人（2015）提出了基于L2规范的DeepFool攻击。该攻击使用迭代方法将决策边界近似为多面体。然后，通过选择多面体的最接近原始图像的部分来找到图像的鲁棒性。他们表明，DeepFool能够以比FGSM更低的扰动创建对抗性示例。

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3RnYmh1WjRDdFJqaWMzdUE0RElpYXJ4d3hwOGtEcUdiSEwxSGNsWTJLcWlhWE0yazZEd2tQdUpIU1BTTXN4UjlRd3lpYVNNNzBnS1VkcXcvNjQw?x-oss-process=image/format,png)

```python
# apply the attack
attack = LinfDeepFoolAttack()

#Set allowable disturbance value
#Заданим значения допустимых возмущений
epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]

t1 = time.time()
raw_advs, clipped_advs, success = attack(fmodel, banana_image, banana_label, epsilons=epsilons)
t2 = time.time()
total_time = (t2 - t1)
print("Total Time: ", total_time, "seconds")
```

```
Total Time:  1.9228878021240234 seconds
```

```python
for sample in raw_advs:
  preds = model.predict(sample)
  print('Predicted:', decode_predictions(preds, top=3)[0])
  plt.figure()
  plt.imshow(sample[0]/ 255)
  plt.title('Adversarial banana image')
  plt.show()
```

```
Predicted: [('n03775071', 'mitten', 0.33119386), ('n03476684', 'hair_slide', 0.05406385), ('n04325704', 'stole', 0.051128566)]
Predicted: [('n03775071', 'mitten', 0.33119386), ('n03476684', 'hair_slide', 0.05406385), ('n04325704', 'stole', 0.051128566)]
Predicted: [('n03775071', 'mitten', 0.33119386), ('n03476684', 'hair_slide', 0.05406385), ('n04325704', 'stole', 0.051128566)]
Predicted: [('n03775071', 'mitten', 0.33119386), ('n03476684', 'hair_slide', 0.05406385), ('n04325704', 'stole', 0.051128566)]
Predicted: [('n03775071', 'mitten', 0.33119386), ('n03476684', 'hair_slide', 0.05406385), ('n04325704', 'stole', 0.051128566)]
Predicted: [('n03775071', 'mitten', 0.33119386), ('n03476684', 'hair_slide', 0.05406385), ('n04325704', 'stole', 0.051128566)]
Predicted: [('n03775071', 'mitten', 0.33119386), ('n03476684', 'hair_slide', 0.05406385), ('n04325704', 'stole', 0.051128566)]
Predicted: [('n03775071', 'mitten', 0.33119386), ('n03476684', 'hair_slide', 0.05406385), ('n04325704', 'stole', 0.051128566)]
Predicted: [('n03775071', 'mitten', 0.33119386), ('n03476684', 'hair_slide', 0.05406385), ('n04325704', 'stole', 0.051128566)]
Predicted: [('n03775071', 'mitten', 0.33119386), ('n03476684', 'hair_slide', 0.05406385), ('n04325704', 'stole', 0.051128566)]
Predicted: [('n03775071', 'mitten', 0.33119386), ('n03476684', 'hair_slide', 0.05406385), ('n04325704', 'stole', 0.051128566)]
```

图片是一样的，11张

![image-20221224214654899](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224214654899.png)

#### Attack 2: C&W ( L2 Carlini Wagner Attack)

理论：

Carlini和Wagner（2017）创建了他们的攻击，作为Papernot等人（2015b）的防御蒸馏的后续，这是针对对抗性示例提出的防御。卡林尼·瓦格纳（C&W）攻击类似于L-BFGS攻击。然而，他们有一些不同之处，他们使用逻辑而不是软最大损失，并使用tanh来限制对抗性示例的范围。攻击是在l0、l2和l∞范数上创建的，所有这些范数都会击败防御蒸馏。他们还表明，与FGSM、BIM、Deepwole和JSMA等其他攻击相比，他们的攻击能够以较少的干扰创建对抗性示例。对于FGSM和BIM，他们进行了搜索，以找到能够欺骗网络的最小扰动。

Carlini и Wagner (2017) создали свою атаку в качестве последующего защитного дистилляции Papernot et al. (2015b), защиты, предложенной против антагонистических примеров.  Атака Калинни Вагнера (C & W) похожа на атаку L - BFGS.  Тем не менее, у них есть некоторые различия, они используют логику вместо мягких максимальных потерь и используют tanh, чтобы ограничить диапазон антагонистических примеров.  Атака была создана на нормах L0, L2 и L -, все из которых побеждают защитную дистилляцию.  Они также показали, что их атаки могут создавать примеры конфронтации с меньшим количеством помех по сравнению с другими атаками, такими как FGSM, BIM, Deepwole и JSMA.  Для FGSM и BIM они провели поиск, чтобы найти минимальные возмущения, которые могут обмануть сеть.

```python
# apply the attack
attack = L2CarliniWagnerAttack()

#Set allowable disturbance value
#Заданим значения допустимых возмущений
epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]

t1 = time.time()
raw_advs, clipped_advs, success = attack(fmodel, banana_image, banana_label, epsilons=epsilons)
t2 = time.time()
total_time = (t2 - t1)
print("Total Time: ", total_time, "seconds")
```

```
Total Time:  1403.964637517929 seconds
```

```python
for sample in raw_advs:
  preds = model.predict(sample)
  print('Predicted:', decode_predictions(preds, top=3)[0])
  plt.figure()
  plt.imshow(sample[0]/ 255)
  plt.title('Adversarial banana image')
  plt.show()
```

结果是一样的，11张

```
Predicted: [('n07753592', 'banana', 0.4081335), ('n03825788', 'nipple', 0.1590686), ('n04131690', 'saltshaker', 0.05764023)]
```

![image-20221224215107278](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224215107278.png)

#### Attack 3: EAD Attack

理论：

Chen等人（2017）指出，尽管L1规范在图像处理（如去噪和恢复）领域很流行，但依赖L1规范的对抗性攻击几乎没有发展。为了说明这一点，他们提出了弹性网络攻击（EAD），它基于C&W攻击和弹性网络正则化。他们表明，这种攻击造成的对手L1距离远小于C&W L2攻击

Chen et al (2017) отмечают, что, хотя спецификация L1 популярна в области обработки изображений, таких как шумоподавление и восстановление, антагонистические атаки, основанные на спецификации L1, практически не развиваются.  Чтобы проиллюстрировать это, они предложили гибкие кибератаки (EAD), которые основаны на C & W - атаках и гибкой регуляризации сети.  Они показывают, что эта атака вызывает гораздо меньшее расстояние L1 для противника, чем атака C & W L2

```python
# apply the attack
attack = EADAttack()

#Set allowable disturbance value
#Заданим значения допустимых возмущений
epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]

t1 = time.time()
raw_advs, clipped_advs, success = attack(fmodel, banana_image, banana_label, epsilons=epsilons)
t2 = time.time()
total_time = (t2 - t1)
print("Total Time: ", total_time, "seconds")
```

```
Total Time:  1477.1710419654846 seconds
```

```python
for sample in raw_advs:
  preds = model.predict(sample)
  print('Predicted:', decode_predictions(preds, top=3)[0])
  plt.figure()
  plt.imshow(sample[0]/ 255)
  plt.title('Adversarial banana image')
  plt.show()
```

结果是一样的，11张

```
Predicted: [('n01930112', 'nematode', 0.14914379), ('n03041632', 'cleaver', 0.026409004), ('n03838899', 'oboe', 0.024560813)]
```

![image-20221224215455030](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224215455030.png)

#### Attack 4: Boundary Attack

理论：

Brendel等人（2017）提出了边界攻击（BA），它不需要访问神经网络的任何部分，只需要访问其预测。该算法的思想是使用一个对抗性示例x′，该示例在像素方向远离原始图像，然后向原始图像改变x′。当发现x′不再是对抗性示例的边界时，攻击会随着决策边界改变x′，使得x′仍然是对抗性图像，并且与原始图像的距离减小。他们表明，这种攻击给出的对抗图像与原始图像的距离比C&W略高，使BA略差。然而，它也使用更少的来自神经网络的信息来创建对抗性示例

Brendel et al. (2017) предлагает пограничные атаки (BA), которые не требуют доступа к какой - либо части нейронной сети, а просто к ее прогнозам.  Идея алгоритма заключается в использовании антагонистического примера x ', который удаляется от исходного изображения в направлении пикселей, а затем изменяет x' к исходному изображению.  Когда обнаруживается, что x 'больше не является границей антагонистического примера, атака изменяет x' с границами принятия решений, делая x 'все еще антагонистическим изображением и уменьшая расстояние от исходного изображения.  Они показали, что эта атака дает конфронтационное изображение на несколько большее расстояние от исходного изображения, чем C & W, что делает BA немного хуже.  Тем не менее, он также использует меньше информации из нейронных сетей для создания антагонистических примеров.

```python
# apply the attack
attack = BoundaryAttack()

#Set allowable disturbance value
#Заданим значения допустимых возмущений
epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]

t1 = time.time()
raw_advs, clipped_advs, success = attack(fmodel, banana_image, banana_label, epsilons=epsilons)
t2 = time.time()
total_time = (t2 - t1)
print("Total Time: ", total_time, "seconds")
```

```
Total Time:  1673.8733484745026 seconds
```

```python
for sample in raw_advs:
  preds = model.predict(sample)
  print('Predicted:', decode_predictions(preds, top=3)[0])
  plt.figure()
  plt.imshow(sample[0]/ 255)
  plt.title('Adversarial banana image')
  plt.show()
```

```
Predicted: [('n07753592', 'banana', 0.27647233), ('n03825788', 'nipple', 0.11950041), ('n04522168', 'vase', 0.05646867)]
```

![image-20221224222800308](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224222800308.png)

#### Attack 5: Fast Gradient Sign Method (FGSM)

Простой и быстрый метод градиента используется для создания примеров Adversarial, чтобы минимизировать максимальное количество возмущений, добавленных к любому пикселю изображения, что приводит к неправильной классификации.  Наиболее известные типы атак на первом этапе алгоритма вычисляют функцию потерь и ее градиент на основе тега реального класса целевого изображения, а затем генерируют пример Adversarial на основе градиента.  Алгоритм может быть представлен следующим выражением:
$$
x_{adv}=x+\varepsilon \times sign(\nabla_xJ(\theta,x,y))
$$
где $x_{adv}$ ：Образ противника,

$x$：Введите изображение,

$y$：Знак реального изображения,

$\varepsilon$：Минимизировать исходное изображение  ε  Коэффициент изменения,

$\theta$：Параметры модели,

$J$：Функция потерь.

$sign()$：Символические функции.

![image-20221221113803913](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221113803913.png)

![image-20221221101852374](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221221101852374.png)

Описание принципа алгоритма FGSM

```python
# apply the attack
attack = LinfFastGradientAttack()

#Set allowable disturbance value
#Заданим значения допустимых возмущений
epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]

t1 = time.time()
raw_advs, clipped_advs, success = attack(fmodel, banana_image, banana_label, epsilons=epsilons)
t2 = time.time()
total_time = (t2 - t1)
print("Total Time: ", total_time, "seconds")
```

```
Total Time:  1.819326400756836 seconds
```

```python
for sample in raw_advs:
  preds = model.predict(sample)
  print('Predicted:', decode_predictions(preds, top=3)[0])
  plt.figure()
  plt.imshow(sample[0]/ 255)
  plt.title('Adversarial banana image')
  plt.show()
```

## 参考文献：

1. [Fast Gradient Attack on Network Embedding 论文笔记](https://zhuanlan.zhihu.com/p/510028180)
2. 
