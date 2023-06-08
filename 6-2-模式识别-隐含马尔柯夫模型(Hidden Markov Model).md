# 6-2-模式识别-隐含马尔柯夫模型(Hidden Markov Model)

## 介绍

马尔可夫随机过程是以著名的俄罗斯数学家A.A.Markov（1856-1922）命名的，他首先开始研究随机变量的概率关系，并创建了一个理论，可以称为“概率动力学”。

这一理论的基础后来成为一般随机过程理论以及扩散过程理论、可靠性理论、大规模维护理论等重要应用科学的基础。

1998年，劳伦斯·佩奇、谢尔盖·布林、拉吉夫·莫特瓦尼和特里·维诺格拉德发表了一篇题为《帕格朗克引文排名：为Web带来秩序》的论文。它描述了著名的PageRank算法，它是谷歌的基础。尽管它的算法已经有了很大的发展，PageRank仍然是谷歌排名算法的“象征”。算法的标准解释之一。

马尔可夫链是一系列事件或动作，其中每个新事件只依赖于前一个事件，而不考虑所有其他事件。这种算法不记得以前是什么，只看以前的状态。

马尔可夫链是一种常见且相当简单的模拟随机事件的方法。它用于各种各样的领域，从文本生成到财务建模。一个著名的例子是`Subredditsimulator`。在这种情况下，马尔可夫链用于自动化整个Subreddit的内容创建。

## 基本概念

设一个系统Ω，它有有限的n个可能状态，由1到n个数字编号。

在外部干预的作用下，在特定时间点 $t_0<t_1<t_2<...$ 系统从一种状态过渡到另一种状态。

假设$λ_{ij}$是系统从状态I过渡到状态J的概率。假设$λ_{ij}$在系统从一个状态过渡到另一个状态的任何时刻都有一个常量。

$λ_{ij}$ 的概率可以用以下矩阵表示：
$$
\lambda=\left[
\begin{matrix}
\lambda_{11} & \lambda_{21} & \lambda_{31} \\
\lambda_{12} & \lambda_{22} & \lambda_{32} \\
\lambda_{13} & \lambda_{23} & \lambda_{33} 
\end{matrix}
\right]\tag{1}
$$
它被称为系统从状态过渡到状态的概率矩阵或简单地称为过渡矩阵。在这种情况下，每个$λ_{ij}$的概率必须大于零，每个状态的概率和必须是1。如果马尔可夫链有n个可能的态，则矩阵的形式为$N\times N$。

马尔可夫过程模型是一个图，其中节点表示模拟对象的状态，弧表示从一种状态过渡到另一种状态的概率。

![image-20221224232850543](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221224232850543.png)

$S_k$-模拟对象的状态

$λ_{ij}$-从 $i$ 态过渡到 $j$ 态的概率

马尔可夫链有一个初始状态向量表示为$N\times 1$矩阵。它描述了$N$种可能状态中开始概率的分布。
$$
x^{(0)}=\left[
\begin{matrix}
p^{0}_{1}  \\
p^{0}_{2} \\
p^{0}_{3}
\end{matrix}
\right]\tag{2}
$$
马尔可夫过程可分为两类：

1. 离散马尔可夫链，其中系统在特定的时间周期内改变其状态（P电路）

2. 连续马尔可夫链，其中系统在任意时刻改变其状态（Q电路）

考虑一个基于马尔可夫链的天气预报的简单例子。根据前一天的天气情况，天气条件的概率（每天的状态被视为下雨或阳光）可以表示为过渡矩阵：
$$
P=\left[
\begin{matrix}
0.9 & 0.1 \\
0.5 & 0.5 
\end{matrix}
\right]\tag{3}
$$
矩阵P提供了一个天气模型，在这个模型中，阳光明媚的一天有90%的概率会跟随另一个阳光明媚的一天，而雨天有50%的概率会跟随另一个雨天。列可以标记为“阳光”和“下雨”，行可以按相同的顺序标记。

假设观测第一天的天气是晴朗的。然后初始状态向量如下：
$$
x^{(0)}=\left[
\begin{matrix}
1 & 0 \end{matrix}
\right]\tag{4}
$$
可以使用以下表达式预测第二天的天气：
$$
x^{(1)}=x^{(0)}P=\left[
\begin{matrix}
1 & 0 \end{matrix}
\right]
\left[
\begin{matrix}
0.9 & 0.1 \\
0.5 & 0.5 
\end{matrix}
\right]=[0.9 & 0.1]\tag{5}
$$
也就是说，第二天也是阳光明媚的一天的可能性是90%。

在第三天，天气条件的概率如下：
$$
x^{(2)}=x^{(1)}P=x^{(0)}P^2=\left[
\begin{matrix}
1 & 0 \end{matrix}
\right]
\left[
\begin{matrix}
0.9 & 0.1 \\
0.5 & 0.5 
\end{matrix}
\right]^2=[0.86 & 0.14]\tag{6}
$$
对于第n天，表达式为：
$$
x^{(n)}=x^{(n-1)}P\tag{7}
$$

$$
x^{(n)}=x^{(0)}P^n\tag{8}
$$

## 隐马尔可夫模型

当需要计算观察到的事件序列的概率时，马尔可夫链是有用的。然而，在许多情况下，相关事件是隐藏的，无法直接观察到。

隐藏马尔可夫模型（HMM）允许我们谈论观察到的事件和隐藏的事件，我们认为这些事件是概率模型中的因果因素。

隐马尔可夫模型（CMM）是一种统计模型，模拟一个过程，类似于马尔可夫过程，具有未知参数，其任务是根据观察到的参数解析未知参数。这些参数可以用于进一步的分析，例如图像识别。

鲍姆在20世纪60年代首次发表了关于隐藏马尔可夫模型的笔记，并在70年代首次应用于语音识别。自20世纪80年代中期以来，CMM一直被用于分析生物序列，特别是DNA。

CMM主要应用于语音识别、写作、运动和生物信息学。此外，SMM还用于密码分析、机器翻译。

马尔可夫隐链模型假设该系统具有以下性质：

1. 在每个时间段内，系统可能处于一组有限的状态；

2. 系统意外地从一种状态过渡到另一种状态（可能是同一种状态），过渡的概率仅取决于它所处的状态；

3. 在任何给定的时间，系统都会给出一个观察特性值——一个仅取决于系统当前状态的随机值。

HMM模型可以描述为：𝐻𝑀𝑀=<𝑀，𝑆，𝐼，𝑇，𝐸>

- M——状态数；
- $S=\{S_1,...,S_M\}$ ——有限多个状态；
- $I=(P(q_0=S_i)=\pi_i) $ ——系统在时间0时处于$i$状态的概率向量；
- $T=\begin{Vmatrix} (P(q_t=S_j|q_{t-1}=S_i)=a_{ij})\end{Vmatrix}$对于$\forall t \in [1,T_m]$ ，从状态 $i$ 到状态 $j$ 的概率矩阵；
- $E=(E_1,...,E_M), E_i=f(o_t|q_t=S_i)$——观察随机变量的分布向量，对应于给定为O上定义的密度或分布函数的每个状态（所有状态的观测值总和）。

时间t被假定为离散的，由非负整数给出，其中0对应于初始时间点，$T_m$对应于最大值。

## 隐马尔可夫模型模拟

本节提供了一个模拟简单天气系统的例子，并根据以下信息预测了每天的温度：

1. 寒冷的日子编码为0，炎热的日子编码为1。
2. 序列的第一天很冷的概率为80%。
3. 在寒冷的一天之后，30%的概率可能是炎热的一天。
4. 炎热的一天有20%的机会变冷。
5. 每一天的温度值正常分布，冷日平均值和标准差为0和5，热日平均值和标准差为15和10。也就是说，在炎热的一天，平均温度为15度，范围从5到25度不等。

为了模拟系统，我们使用TensorFlow库和Google Colaboratory开发环境。HiddenMarkovModel类实现了离散隐藏马尔可夫模型，其中初始状态、过渡概率和观察状态由用户提供的分布指定。在第一步中，导入必要的图书馆：

```python
import tensorflow_probability as tfp
import tensorflow as tf
```

接下来，我们将设置系统建模的参数:

```python
tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.2 0.8])#条款2
transition_distribution = tfd.Categorical(probs=[[0.7,0.3],[0.2,0.8]])# 条款3与4
observation_distribution = tfd.Normal(loc=[0.,15.],scale=[5.,10.])# 条款5
#loc-平均值,scale-标准差
```

初始化模型：

```python
model = tfd.HiddenMarkovModel(
	inital_distribution = initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps=7
)
```

numu steps参数是预测的步骤数。在本例中，它使用每周7天的步骤。

最后一步是启动模型并输出预测：

```python
mean = model.mean()

with tf.compat.v1.Session() as sess:
    print(mean.numpy())
```

![image-20221225001455883](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221225001455883.png)

## 实验室工作任务：

基于隐藏马尔可夫模型构建模型以识别其中一组数据。研究模型参数选择对模型结果的影响。

1）语音识别

2）MNIST 

3）自己的数据集

## Python音频信号处理

音频信号是模拟信号，我们需要将其保存为数字信号，才能对语音进行算法操作，WAV是Microsoft开发的一种声音文件格式，通常被用来保存未压缩的声音数据。

语音信号有三个重要的参数：**声道数**、**取样频率**和**量化位数**。

- **声道数**：可以是单声道或者是双声道
- **采样频率**：一秒内对声音信号的采集次数，44100Hz采样频率意味着每秒钟信号被分解成44100份，如果采样率高，那么媒体播放音频时会感觉信号是连续的。
- **量化位数**：用多少bit表达一次采样所采集的数据，通常有8bit、16bit、24bit和32bit等几种

如果你需要自己录制和编辑声音文件，推荐使用[Audacity]([http://audacity.sourceforge.net)它是一款开源的、跨平台、多声道的录音编辑软件。

```python
from scipy.io import wavfile
import numpy as np
import matplotlib.pylab as plt
samplimg_freq, audio = wavfile.read("data/input_freq.wav")
plt.plot(np.arange(audio.shape[0]),audio)
plt.show()
```

音频的时域信号波形：

![image-20221226201854799](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226201854799.png)

语音信号是一个非平稳的时变信号，但语音信号是由声门的激励脉冲通过声道形成的，而声道(人的口腔、鼻腔)的肌肉运动是缓慢的，所以“短时间”(10-30ms)内可以认为语音信号是平稳时不变的。由此构成了语音信号的“短时分析技术”。 在短时分析中，将语音信号分为一段一段的语音帧，每一帧一般取10-30ms，我们的研究就建立在每一帧的语音特征分析上。 提取的不同的语音特征参数对应着不同的语音信号分析方法：时域分析、频域分析、倒谱域分析…由于语音信号最重要的感知特性反映在功率谱上，而相位变化只起到很小的作用，所有语音频域分析更加重要。

### 梅尔倒频谱系数MFCC

MFCCs中文名为“ **梅尔倒频谱系数** ”（Mel Frequency Cepstral Coefficents）是一种在自动语音和说话人识别中广泛使用的特征。它是在1980年由Davis和Mermelstein搞出来的。从那时起。在语音识别领域，MFCCs在人工特征方面可谓是鹤立鸡群，一枝独秀，从未被超越啊（至于说Deep Learning的特征学习那是后话了）。

> 任何自动语音识别系统中的第一步都是提取特征，即识别音频信号中有利于识别语言内容的成分，并丢弃所有携带诸如背景噪声、情绪等信息的其他成分。 

> 理解语音的要点是，人类发出的声音是由包括舌头、牙齿等在内的声道形状过滤的。这种形状决定了发出什么声音。如果我们能够准确地确定形状，这将为我们提供所产生的音素的准确表示。声道的形状表现在短时功率谱的包络中，MFCC的工作是准确地表示该包络。本页将提供有关MFCC的简短教程。 

> Mel频率倒谱系数（MFCC）是一种广泛用于自动语音和说话人识别的特征。它们是由戴维斯和梅梅尔斯坦在20世纪80年代引进的，从那以后一直是最先进的。在引入MFCC之前，线性预测系数（LPC）和线性预测倒谱系数（LPCC）（单击此处了解倒谱和LPCC教程）是自动语音识别（ASR）的主要特征类型，尤其是HMM分类器。本页将介绍MFCC的主要方面，为什么它们是ASR的好功能，以及如何实现它们。
>
> http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

### 步骤一览 

我们将对实现步骤进行高层次的介绍，然后深入了解我们为什么要做这些事情。最后，我们将对如何计算MFCC进行更详细的描述。 

1. 将信号分为短帧(short frames)。 
2. 对于每一帧，计算功率谱的周期图估计([periodogram estimate](http://en.wikipedia.org/wiki/Periodogram))。 
3. 将mel滤波器组应用于功率谱(power spectra)，对每个滤波器中的能量求和。 
4. 取所有滤波器组能量(filterbank energies)的对数。 
5. 取对数滤波器组能量的DCT。 
6. 保持DCT系数2-13，丢弃其余系数。 

还有一些常见的操作，有时将帧能量附加到每个特征向量。[Delta and Delta-Delta](http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#deltas-and-delta-deltas)特征通常也会附加。提升也通常应用于最终特征。

### 我们为什么要做这些事？ 

现在，我们将更缓慢地完成这些步骤，并解释为什么每个步骤都是必要的。 

音频信号是不断变化的，所以为了简化事情，我们假设在短时间尺度上音频信号变化不大（当我们说它没有变化时，我们的意思是统计上的，即统计上的平稳，很明显，样本在甚至短时间尺度下也在不断变化）。这就是为什么我们将信号帧化为**20-40ms帧**。如果帧短得多，我们没有足够的样本来获得可靠的频谱估计，如果帧长得多，信号在整个帧中变化太大。 

下一步是计算每帧的功率谱。这是由人类耳蜗（耳朵中的一个器官）驱动的，它根据传入声音的频率在不同的位置振动。根据耳蜗中振动的位置（振动小毛发），不同的神经会发出信号，告知大脑存在某些频率。我们的**周期图估计**为我们执行类似的工作，**识别帧中存在的频率。** 

周期图谱估计仍然包含许多自动语音识别（ASR）所不需要的信息。特别地，耳蜗不能辨别两个紧密间隔的频率之间的差异。随着频率的增加，这种效应变得更加明显。出于这个原因，我们取一组周期图箱，并将它们相加，以了解在不同频率区域中存在多少能量。这是由我们的梅尔滤波器组执行的：第一个滤波器非常窄，并给出了在0赫兹附近存在多少能量的指示。随着频率越来越高，我们的滤波器也越来越宽，因为我们越来越不关心变化。我们**只对每个点发生的能量大致多少感兴趣**。Mel规模准确地告诉我们如何划分滤波器组的空间，以及它们的宽度。有关如何计算间距，请参见下文。 

一旦我们有了滤波器组能量，我们就取它们的**对数**。这也是人类听觉的驱动因素：我们听不到线性范围内的响度。一般来说，要使声音的感知音量翻倍，我们需要将8倍的能量投入其中。这意味着，如果声音一开始就很响亮，能量的巨大变化听起来可能不会那么不同。这种压缩操作使我们的特征更接近人类实际听到的声音。为什么是对数而不是立方根？对数允许我们使用倒谱平均减法，这是一种信道归一化技术。 

最后一步是计算对数滤波器组能量的DCT。执行此操作的主要原因有两个。因为我们的滤波器组都是重叠的，所以滤波器组能量彼此非常相关。**DCT去相关**能量，这意味着对角协方差矩阵可以用于例如HMM分类器中的特征建模。但请注意，26个DCT系数中**只有12个被保留。**这是因为较高的DCT系数代表了滤波器组能量的快速变化，事实证明，这些快速变化实际上会降低ASR性能，因此我们通过丢弃它们来获得一个小的改进。

### 短时傅里叶变换(STFT)

声音信号本是一维的时域信号，直观上很难看出频率变化规律。如果通过傅里叶变换把它变到频域上，虽然可以看出信号的频率分布，但是丢失了时域信息，无法看出频率分布随时间的变化。为了解决这个问题，很多时频分析手段应运而生。短时傅里叶，小波，Wigner分布等都是常用的时频域分析方法。

 短时傅里叶变换（STFT）是最经典的时频域分析方法。傅里叶变换（FT）想必大家都不陌生，这里不做专门介绍。所谓短时傅里叶变换，顾名思义，是对短时的信号做傅里叶变化。那么短时的信号怎么得到的? 是长时的信号分帧得来的。这么一想，STFT的原理非常简单，**把一段长信号分帧、加窗，再对每一帧做傅里叶变换（FFT）**，最后把每一帧的结果沿另一个维度堆叠起来，得到类似于一幅图的二维信号形式。如果我们原始信号是声音信号，那么通过STFT展开得到的二维信号就是所谓的声谱图。

![image-20221226215642327](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226215642327.png)

### 声谱图(Spectrogram)

一段语音被分为很多帧，每帧语音都对应于一个频谱（通过短时FFT计算），**频谱表示频率与能量的关系**(不同频率的振幅大小不同)。在实际使用中，频谱图有三种，即线性振幅谱、对数振幅谱、自功率谱(对数振幅谱中各谱线的振幅都作了对数计算，所以其纵坐标的单位是dB（分贝）。这个变换的目的是**使那些振幅较低的成分相对高振幅成分得以拉高**，以便观察掩盖在低幅噪声中的周期信号)，下图一纵坐标是振幅，横坐标为频率，我们把它旋转90度后，依然是振幅和频率的关系，但我们还想再插入时间维度，可以选择把振幅用颜色深浅来表示，振幅越大，颜色越深，即**把幅度映射到一个灰度级表示**(0表示白，255表示黑)，幅度值越大，相应的区域越黑。

![image-20221226215729322](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226215729322.png)

这样，我们依次把切分出的每一帧时域信号都进行上述处理并按时间维度进行排列，就得到了随着时间变化的频谱图， 即声谱图。

![image-20221226215756767](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226215756767.png)

下图是一段语音的声谱图，很黑的地方就是频谱图中的峰值(共振峰formants)。

![image-20221226215821374](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226215821374.png)

**那我们为什么要在声谱图中表示语音呢？**

首先，音素（Phones）的属性可以更好的在这里面观察出来。另外，通过观察共振峰和它们的转变可以更好的识别声音。隐马尔科夫模型（Hidden Markov Models）就是隐含地**对声谱图进行建模以达到好的识别性能**。还有一个作用就是它可以直观的**评估TTS系统（text to speech）的好坏**，直接对比合成的语音和自然的语音声谱图的匹配度即可。

### 梅尔频谱

声谱图往往是很大的一张图，为了得到合适大小的声音特征，往往把它通过梅尔标度滤波器组（mel-scale filter banks），变换为梅尔频谱。什么是梅尔滤波器组呢？这里要从梅尔标度（mel scale）说起。

**梅尔标度**，the mel scale，由Stevens，Volkmann和Newman在1937年命名。我们知道，频率的单位是赫兹（Hz），人耳能听到的频率范围是20-20000Hz，但人耳对Hz这种标度单位并不是线性感知关系。例如如果我们适应了1000Hz的音调，如果把音调频率提高到2000Hz，我们的耳朵只能觉察到频率提高了一点点，根本察觉不到频率提高了一倍。如果将普通的频率标度转化为梅尔频率标度，映射关系如下式所示：
$$
mel(f)=2595*log_{10}(1+f/700)
$$
or
$$
mel(f)=1125*ln(1+f/700)
$$
![image-20221226220018211](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226220018211.png)

则人耳对频率的感知度就成了线性关系。也就是说，**在梅尔标度下，如果两段语音的梅尔频率相差两倍，则人耳可以感知到的音调大概也相差两倍**。让我们观察一下从Hz到mel的映射图，由于它们是log的关系，当频率较小时，mel随Hz变化较快；当频率很大时，mel的上升很缓慢，曲线的斜率很小。这说明了**人耳对低频音调的感知较灵敏，在高频时人耳是很迟钝的**，梅尔标度滤波器组启发于此。

![image-20221226220040039](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226220040039.png)

如上图所示，40个三角滤波器组成滤波器组，低频处滤波器密集，门限值大，高频处滤波器稀疏，门限值低。恰好对应了**频率越高人耳越迟钝**这一客观规律。上图所示的滤波器形式叫做**等面积梅尔滤波器**（Mel-filter bank with same bank area），在<u>*人声领域（语音识别，说话人辨认）等领域应用广泛*</u>，但是如果用到<u>*非人声领域*</u>，就会丢掉很多高频信息。这时我们更喜欢的或许是**等高梅尔滤波器**（Mel-filter bank with same bank height）：

![image-20221226220100919](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226220100919.png)

### 倒谱分析(Cepstrum Analysis)

下面是截取一段语音得到的频谱图。峰值就表示语音的主要频率成分，我们把这些峰值称为**共振峰(formants)**，而共振峰就是携带了声音的辨识属性（就是个人身份证一样）。所以它特别重要。用它就可以识别不同的声音。

![image-20221226220133179](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226220133179.png)

既然它那么重要，那我们就是需要把它提取出来！我们要提取的不仅仅是共振峰的位置，还得提取它们转变的过程。所以我们提取的是频谱的包络（Spectral Envelope）。这包络就是一条**连接这些共振峰点的平滑曲线**。

![image-20221226220153691](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226220153691.png)

我们可以这么理解，将原始的频谱由两部分组成：**包络(大的趋势)和频谱的细节(小区域跳动)**。这里用到的是对数频谱，所以单位是dB。那现在我们需要把这两部分分离开，分别得到**包络线和细节**，即下图 $logH[k]$ 和 $logE[k]$ 各点相加后能得到$logX[k]$，即$logX[k] = logH[k] + logE[k]$

![image-20221226220225056](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226220225056.png)

那怎么把它们分开呢，也就是，怎么在给定 $logX[k]$ 的基础上，求得 $logH[k]$ 和 $logE[k]$ 以满足$logX[k] = logH[k] + logE[k]$呢？

为了达到这个目标，我们需要Play a Mathematical Trick，那就是对频谱做FFT。(傅立叶定理告诉我们，任何连续测量的时序或信号，都可以表示为不同频率的正弦波信号的无限叠加而成。时域信号做FFT变换后得到其不同频率组成及各频率的振幅。如下图的**时域信号** $x(t)$，按频率展开，可以取前一个或几个低频且幅度较大的信号叠加在一起记为$h(t)$，认为是**时域信号的络**，大概能反映出信号的走势，然后用$x(t)-h(t)=e(t)$，得到**信号的细节**。即x(t)是由以h(t)描述的大的走势下，然后局部区域由e(t)扰动叠加而成)

![image-20221226220252528](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226220252528.png)

现在我们依然把$logX[k]$看成是时域信号，借用上面方法求包络和细节。由于$logX[k]$是频域信号，那在频谱上做傅里叶变换就相当于**逆傅里叶变换Inverse FFT (IFFT)**。需要注意的一点是，我们是在频谱的**对数域**上面处理的，这也属于Trick的一部分。这时候，在对数频谱上面做IFFT就相当于在一个**伪频率（pseudo-frequency）**坐标轴上面描述信号。

![image-20221226220321793](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226220321793.png)

在实际中咱们已经知道$logX[k]$，所以我们可以得到了$x[k]$。那么由图可以知道，$h[k]$是$x[k]$的低频部分，那么我们**将$x[k]$通过一个低通滤波器就可以得到$h[k]$了，继而通过$x[k]-h[k]=e[k]$**。没错，到这里咱们就可以将它们分离开了，得到了我们想要的$h[k]$，也就是频谱的包络。

**$x[k]$实际上就是倒谱Cepstrum（这个是一个新造出来的词，把频谱的单词spectrum的前面四个字母顺序倒过来就是倒谱的单词了）。而我们所关心的$h[k]$就是倒谱的低频部分。$h[k]$描述了频谱的包络(上面说了，它可以取前一个或几个低频且幅度较大的信号叠加)，它在语音识别中被广泛用于描述特征。**

那现在总结下倒谱分析，它实际上是这样一个过程：

1. 将原语音信号经过傅里叶变换得到频谱：$X[k]=H[k]E[k]$；

   只考虑幅度就是：$|X[k] |=|H[k]||E[k] |$；

2. 我们在两边取对数：$log||X[k] ||= log ||H[k] ||+ log ||E[k] ||$。

3. 再在两边取逆傅里叶变换得到倒谱：$x[k]=h[k]+e[k]$。

这实际上有个专业的名字叫做**同态信号处理**。它的目的是将非线性问题转化为线性问题的处理方法。对应上面，原来的语音信号实际上是一个卷性信号（声道相当于一个线性时不变系统，声音的产生可以理解为一个激励通过这个系统），第一步通过卷积将其变成了乘性信号（时域的卷积相当于频域的乘积）。第二步通过取对数将乘性信号转化为加性信号，第三步进行逆变换，使其恢复为卷性信号。这时候，虽然前后均是时域序列，但它们所处的离散时域显然不同，所以后者称为**倒谱频域**。

**总结下，倒谱（cepstrum）就是一种信号的傅里叶变换经对数运算后再进行傅里叶反变换得到的谱(两次FFT运算)。它的计算过程如下：**

![image-20221226215125819](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226215125819.png)

### Mel频率分析(Mel-Frequency Analysis)

好了，到这里，我们先看看我们刚才做了什么？给我们一段语音，我们可以得到了它的频谱包络（连接所有共振峰值点的平滑曲线）了。但是，对于人类听觉感知的实验表明，人类听觉的感知只聚焦在某些特定的区域，而不是整个频谱包络。

而Mel频率分析就是基于人类听觉感知实验的。实验观测发现**人耳就像一个滤波器组**一样，它只关注某些特定的频率分量(人的听觉对频率是有选择性的)。也就说，它只让某些频率的信号通过，而压根就直接无视它不想感知的某些频率信号。但是这些滤波器在频率坐标轴上却不是统一分布的，在低频区域有很多的滤波器，他们分布比较密集，但在高频区域，滤波器的数目就变得比较少，分布很稀疏。

![image-20221226220534861](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226220534861.png)

人的听觉系统是一个特殊的非线性系统，它响应不同频率信号的灵敏度是不同的。**在语音特征的提取上，人类听觉系统做得非常好，它不仅能提取出语义信息, 而且能提取出说话人的个人特征，这些都是现有的语音识别系统所望尘莫及的**。如果在语音识别系统中能模拟人类听觉感知处理特点，就有可能提高语音的识别率。

**梅尔频率倒谱系数(Mel Frequency Cepstrum Coefficient, MFCC)考虑到了人类的听觉特征，先将线性频谱映射到基于听觉感知的Mel非线性频谱中，然后转换到倒谱上**。

再次提及将普通频率转化到Mel频率的公式是：
$$
mel(f)=2595*log_{10}(1+f/700)
$$
**在Mel频域内，人对音调的感知度为线性关系。**举例来说，如果两段语音的Mel频率相差两倍，则人耳听起来两者的音调也相差两倍。

### 梅尔频率倒谱系数(Mel Frequency Cepstrum Coefficient)

我们将频谱通过一组Mel滤波器就得到Mel频谱。公式表述就是：$log X[k] = log (Mel-Spectrum)$。这时候我们在$log X[k]$上进行倒谱分析：

1）取对数：$log X[k] = log H[k] + log E[k]$。

2）进行逆变换：$x[k] = h[k] + e[k]$。

在Mel频谱上面获得的倒谱系数$h[k]$就称为Mel频率倒谱系数，简称MFCC。

![image-20221226220702326](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226220702326.png)

## 提取MFCC特征的过程

现在咱们来总结下提取MFCC特征的过程:

1. 先对语音进行**预加重、分帧和加窗**；

2. 对每一个短时分析窗，**通过FFT得到对应的频谱**；

3. 将上面的频谱**通过Mel滤波器组得到Mel频谱**；

4. 在Mel频谱上面进行**倒谱分析**（取对数，做逆变换，实际逆变换一般是通过DCT离散余弦变换来实现，**取DCT后的第2个到第13个系数作为MFCC系数）**，获得Mel频率倒谱系数MFCC，这个MFCC就是**这帧语音的特征**；

![image-20221226215337405](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226215337405.png)

https://zhuanlan.zhihu.com/p/350846654

### **预加重**

预增强以帧为单位进行，目的在于加强高频。去除口唇辐射的影响，增加语音的高频分辨率。因为高频端大约在800Hz以上按6dB/oct (倍频程)衰减，频率越高相应的成分越小，为此要在对语音信号进行分析之前对其高频部分加以提升，也可以改善高频信噪比。k是预增强系数，常用0.97。

```python
pre_emphasis = 0.97
emphasized_signal = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
plt.plot(np.arange(emphasized_signal.shape[0]),emphasized_signal)
plt.show()
```

![image-20221226224436541](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226224436541.png)

### **分帧**

分帧是将不定长的音频切分成固定长度的小段。为了避免窗边界对信号的遗漏，因此对帧做偏移时候，帧间要有帧移(帧与帧之间需要重叠一部分)，$帧长(wlen) = 重叠(overlap)+帧移(inc)$。inc为帧移，表示后一帧第前一帧的偏移量，fs表示采样率，fn表示一段语音信号的分帧数。

$\frac{N-overlap}{inc}=\frac{N-wlen+inc}{inc}$

通常的选择是帧长25ms（下图绿色），帧移为10ms（下图黄色）。接下来的操作是对单帧进行的。要分帧是因为语音信号是快速变化的，而傅里叶变换适用于分析平稳的信号。帧和帧之间的时间差常常取为10ms，这样帧与帧之间会有重叠（下图红色），否则，由于帧与帧连接处的信号会因为加窗而被弱化，这部分的信息就丢失了。 

![image-20221226224502944](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226224502944.png)

### **语音信号的短时频域处理**

在语音信号处理中，在语音信号处理中，信号在频域或其他变换域上的分析处理占重要的位置，在频域上研究语音可以使信号在时域上无法表现出来的某些特征变得十分明显，一个音频信号的本质是由其频率内容决定的，将时域信号转换为频域信号一般对语音进行**短时傅里叶变换**。

## **python_speech_features**

python_speech_features的比较好用的地方就是自带预加重参数，只需要设定preemph的值，就可以对语音信号进行预加重，增强高频信号。 python_speech_features模块提供的函数主要包括两个：MFCC和FBank。API定义如下：

>  python_speech_features.base.fbank(signal, samplerate=16000, winlen=0.025, winstep=0.01, nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, winfunc=>)  

从一个音频信号中计算梅尔滤波器能量特征,返回：2个值。第一个是一个包含着特征的大小为nfilt的numpy数组，每一行都有一个特征向量。第二个返回值是每一帧的能量。

>  python_speech_features.base.logfbank(signal, samplerate=16000, winlen=0.025, winstep=0.01, nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97)  

从一个音频信号中计算梅尔滤波器能量特征的对数,返回： 一个包含特征的大小为nfilt的numpy数组，每一行都有一个特征向量

参数 

```python
参数：

signal - 需要用来计算特征的音频信号，应该是一个N*1的数组

samplerate - 我们用来工作的信号的采样率

winlen - 分析窗口的长度，按秒计，默认0.025s(25ms)

winstep - 连续窗口之间的步长，按秒计，默认0.01s（10ms）

numcep - 倒频谱返回的数量，默认13

nfilt - 滤波器组的滤波器数量，默认26

nfft - FFT的大小，默认512

lowfreq - 梅尔滤波器的最低边缘，单位赫兹，默认为0

highfreq - 梅尔滤波器的最高边缘，单位赫兹，默认为采样率/2

preemph - 应用预加重过滤器和预加重过滤器的系数，0表示没有过滤器，默认0.97

ceplifter - 将升降器应用于最终的倒谱系数。 0没有升降机。默认值为22。

appendEnergy - 如果是true，则将第0个倒谱系数替换为总帧能量的对数。

winfunc - 分析窗口应用于每个框架。 默认情况下不应用任何窗口。 你可以在这里使用numpy窗口函数 例如：winfunc=numpy.hamming
```

## **MFCC特征和过滤器特征**

```python
from python_speech_features import mfcc, logfbank

#提取MFCC特征和过滤器特征
mfcc_features = mfcc(audio, samplimg_freq)
filterbank_features = logfbank(audio, samplimg_freq)

#打印参数，查看可生成多少个窗体：
print('\nMFCC:\nNumber of windows =', mfcc_features.shape[0])
print('Length of each feature =', mfcc_features.shape[1])
print('\nFilter bank:\nNumber of windows=', filterbank_features.shape[0])
print('Length of each feature =', filterbank_features.shape[1])

#将MFCC特征可视化。转置矩阵，使得时域是水平的。
mfcc_features = mfcc_features.T
plt.matshow(mfcc_features)
plt.title('MFCC')

#将滤波器组特征可视化。转置矩阵，使得时域是水平的。
filterbank_features = filterbank_features.T
plt.matshow(filterbank_features)
plt.title('Filter bank')
plt.show()
```

输出如下：

```python
MFCC:
Number of windows = 42
Length of each feature = 13

Filter bank:
Number of windows= 42
Length of each feature = 26
```

![image-20221226224644473](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226224644473.png)

![image-20221226224655085](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226224655085.png)

## **触发词检测**

![image-20221226224739175](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226224739175.png)

![image-20221226224759831](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221226224759831.png)

https://cloud.tencent.com/developer/article/2159931

### hmmlearn库的使用

hmmlearn 一共实现了三种HMM模型类，按照数据的观测状态是离散的还是连续的可以划分为两类。GaussianHMM （高斯HMM模型）和GMMHMM（混合高斯模型）是观测状态为连续的模型。 MultinomialHMM（多项式分布HMM模型）是观测状态为离散的模型。这三种算法都可以被用来估计模型的参数。
————————————————
版权声明：本文为CSDN博主「Starry memory」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/doswynkfsw/article/details/124356671

## Exercise：HMM_Speech_Recognition

### 安装python-speech-features

```python
!pip install python_speech_features
```

```
Collecting python_speech_features
  Downloading python_speech_features-0.6.tar.gz (5.6 kB)
  Preparing metadata (setup.py) ... done
Building wheels for collected packages: python_speech_features
  Building wheel for python_speech_features (setup.py) ... done
  Created wheel for python_speech_features: filename=python_speech_features-0.6-py3-none-any.whl size=5888 sha256=bf998b955175c3b0f6d21a21590830db5077835cc84f090c1ffc6963904939d8
  Stored in directory: /root/.cache/pip/wheels/b0/0e/94/28cd6afa3cd5998a63eef99fe31777acd7d758f59cf24839eb
Successfully built python_speech_features
Installing collected packages: python_speech_features
Successfully installed python_speech_features-0.6
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv class="ansi-yellow-fg">
```

### 加载库

```python
#from __future__ import print_function
#使版本python2.x使用python3.x的print的函数
import warnings
import os
from python_speech_features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np
```

```python
warnings.filterwarnings('ignore')#忽略警告消息
```

#### mfcc(python_speech_features)

```python
python_speech_features.base.mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=<function <lambda>>)
```

https://python-speech-features.readthedocs.io/en/latest/

根据音频信号计算MFCC特征。 

参数： 

- `signal`–用于计算特征的音频信号。应为N*1数组 
- `samplerate`–我们正在处理的信号的采样率。 
- `winlen`–分析窗口的长度（秒）。默认值为0.025s（25毫秒） 
- `winstep`–连续窗口之间的步长（秒）。默认值为0.01秒（10毫秒） 
- `numcep`–要返回的倒谱数，默认值为13 
- `nfilt`–滤波器组中的滤波器数量，默认值为26。 
- `nfft`–FFT大小。默认值为512。 
- `lowfreq`–mel滤波器的最低频带边缘。以Hz为单位，默认值为0。 
- `highfreq`–mel滤波器的最高频带边缘。以Hz为单位，默认值为采样率/2 
- `preemph`–应用以preemph为系数的预加重滤波器。0不是筛选器。默认值为0.97。 
- `ceplifter`–将一个提升器应用于最终的倒谱系数。0不是提升器。默认值为22。 
- `appendEnergy`–如果为真，则第0个倒谱系数将替换为总帧能量的对数。 
- `winfunc`–应用于每个帧的分析窗口。默认情况下，不应用任何窗口。您可以在此处使用numpy窗口函数，例如winfunc=numpy.hamming

返回：包含特性的大小（NUMFRAMES by numcep）的numpy数组。每行包含1个特征向量。

### 创建音频数据集

```python
def buildDataSet(dir):
    '''
    建立数据集
    1、筛选出wav数据
    2、提取出label作为数据集的key
    3、用mfcc提取音频特征
    4、把标签和特征结合在一起
    '''
    # Filter out the wav audio files under the dir
    # 筛选出目录下的wav音频文件
    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    for fileName in fileList:
        tmp = fileName.split('.')[0]
        label = tmp.split('_')[1]
        full_path = os.path.join(dir,fileName)
        feature = extract_mfcc(full_path)
        if label not in dataset.keys():
            dataset[label] = []
            dataset[label].append(feature)
        else:
            exist_feature = dataset[label]
            exist_feature.append(feature)
            dataset[label] = exist_feature
    return dataset
```



### 提取音频的mfcc特征

```python
def extract_mfcc(full_audio_path):
    '''
    提取音频的mfcc特征
    参数：
    full_audio_path：音频路径
    解释：
    sample_rate：音频的采样率
    wave：数据
    mfcc_features：梅尔倒频谱系数特征
    '''
    sample_rate, wave =  wavfile.read(full_audio_path)
    mfcc_features = mfcc(wave, samplerate=sample_rate)
    return mfcc_features
```

#### scipy.io.wavfile.read()函数

https://blog.csdn.net/zzjxx_/article/details/123426568

```python
rate, data = scipy.io.wavfile.read(filename, mmap=False)
# rate：采样率
# data：数据
```

1. 函数含义：

   打开一个 WAV 文件。从 LPCM WAV 文件返回采样率(rate)(以样本/秒为单位)和数据(data)。所以需要两个参数

2. 参数：
   - `filename`： 字符串或打开文件句柄，输入 WAV 文件。
   - `mmap`： 布尔型，可选是否读取数据为memory-mapped(默认：False)。与某些位深度不兼容。仅用于真实文件.

3. 返回：
   - rate： **int**。WAV 文件的采样率。
   - data： numpy **数组**。从 WAV 文件中读取的数据。数据类型由文件确定。对于 1 通道 WAV，数据是 1-D，否则是 2-D 形状(Nsamples，Nchannels)。如果传递了没有 C-like 文件说明符(例如 io.BytesIO )的 file-like 输入，则这将不可写。

---

### 补充：初识HMM

隐马尔科夫模型（Hidden Markov Model，简称HMM）是用来描述隐含未知参数的统计模型，HMM已经被成功于语音识别、文本分类、生物信息科学、故障诊断和寿命预测等领域。

HMM可以由三个要素组成： $λ=\{A,B,∏\}$，其中$A$为状态转移概率矩阵，$B$为观测状态概率矩阵，$∏$为隐藏状态初始概率分布。

HMM有两个基本假设，

- 一是齐次马尔可夫性假设，隐马尔可夫链 $t$ 的状态只和 $t-1$ 状态有关；

- 二是观测独立性假设，观测只和当前时刻状态有关。

#### HMM有三个基本问题：

- 预测问题（解码问题）——给定模型参数和观测数据，估计隐藏状态的最优序列。 
- 概率计算问题——给定模型参数和观测数据，计算观测序列出现的概率，即模型似然性（model likelihood）。 
- 学习问题——仅给定观测数据，估计模型参数。

**HMM解决的三个问题**：

- 一是概率计算问题，已知模型和观测序列，**计算观测序列出现的概率**，该问题求解的方法为向前向后法；
- 二是学习问题，已知观测序列，**估计模型的参数**，该问题求解的方法为鲍姆-韦尔奇算法
- 三是预测问题（解码问题），已知模型和观测序列，**求解状态序列**，该问题求解的方法为动态规划的维特比算法。【实例分析】

HMM的实现：python的**hmmlearn**类，按照观测状态是连续状态还是离散状态，可以分为两类。

#### HMM常用的三种模型

1. GaussianHMM 观测状态连续型且符合高斯分布
2. GMMHMM 观测状态连续型且符合混合高斯分布
3. MultinomialHMM 观测状态离散型

语音识别是连续状态，所以此处选择GMMHMM模型，它是混合高斯模型。

---

### 补充：概率基础知识复习

#### 基础知识

**随机标量变量**：一个基于随机实验结果的实数或实数变量； 

**随机向量变量**：彼此相关或独立的随机标量变量的一个集合； 

**域**：随机变量的所有可能取值； 

**连续型随机变量x的基本特性**：它的分布或概率密度函数，通常记为 p(x) ； 

连续型随机变量在 $x=a$ 处的**概率密度函数**定义：
$$
p(a) \approx \lim_{\triangle a\to 0}\frac{P(a-\triangle a<x\leq a)}{\triangle a}\geq 0
$$
其中， $P(·)$ 表示事件的概率。

连续型随机变量 $x$ 在 $x=a$ 处的**累积分布函数**定义：
$$
P(a) \approx P(x\leq a)=\int^{a}_{-\infty}{p(x)dx}
$$
概率密度函数需要满足**归一化**性质
$$
P(x< \infty)=\int^{\infty}_{-\infty}p(x)dx=1
$$
没有满足归一化性质的概率密度函数称为一个不当密度或非归一化分布。

- 对一个连续随机向量 $\vec{x}=(x_1,x_2,...,x_D)^T∈R^D$ ，定义它们的联合概率密度为 $p(x_1,x_2,...,x_D)$ ； 
- 对每一个在随机向量 $\vec{x}$ 中的随机变量 $x_i$，**边缘概率密度**函数定义为

$$
p(x_i)=\iint_{all\ x_j:x_j \neq x_i}{p(x_1,x_2,...,x_D)d{x_{1}...x_{i-1}x_{i+1}...x_{D}}}
$$

它和标量随机变量的概率密度函数具有相同的性质。 

- 极大似然估计：利用已知样本的结果，反推最大概率导致这样结果的参数值。 

- 高斯分布：如果连续型标量随机变量 $x$ 的概率密度函数是

$$
p(x)=\frac{1}{(2\pi)^{1\over 2}\sigma}\exp{[-{1\over2}({x-\mu \over\sigma })^{2}]}=N{(x;\mu,\sigma^{2})},\ (-\infty <x<\infty; \sigma>0)
$$

那么它是服从正态分布或高斯分布的。上式的一个等价标记是 $x∼N(μ,σ^2)$ ，表示随机变量 x 服从均值为 $μ$ 、方差为 $σ^2$ 的正态分布。使用精度参数（精度是方差的倒数）代替方差后，高斯分布的概率密度函数也可以写为
$$
p(x)=\sqrt{r\over 2\pi}\exp{[-{r\over2}{(x-\mu)^2}]}
$$
易证，对一个高斯随机变量 $x$ ，期望和方差分别满足
$$
E(x)=\mu
$$

$$
var(x)=\sigma^{2}=r^{-1}
$$

由下面的联合概率密度函数定义的正态随机变量 $x=(x_1,x_2,...,x_D)^T $ 也称多元或向量值高斯随机变量：
$$
p(x)=\frac{1}{(2\pi)^{D\over 2}{\vert {\Sigma}\vert^{1\over2}}}\exp{[-{1\over2}({x-\mu})^{T}{\Sigma^{-1}{(x-\mu)}}]}=N{(x;\mu,\Sigma)}
$$
与其等价的表示是 $x∼N(μ∈R^D,\Sigma∈R^{D×D})$ 。对于多元高斯随机变量，其均值和协方差矩阵可由 $E(x)=μ $， $E[{(x−\overline x)}{(x−\overline x)^T}]=\Sigma$ 给出。

#### EM算法

EM算法是一种迭代算法，用于含有隐变量的概率模型参数的极大似然估计，或极大后验概率估计。EM算法的每次迭代由两步组成：E步，求期望；M步，求极大。所以这一算法称为期望极大算法，简称EM算法。

##### EM算法的概要

1.EM算法是含有隐变量的概率模型极大似然估计或极大后验概率估计的迭代算法。含有隐变量的概率模型的数据表示为 $P(Y,Z|θ)$ 。这里， $Y$ 是观测变量的数据， $Z$ 是隐变量的数据， $θ$ 是模型参数。EM算法通过迭代求解观测数据的对数似然函数 $L(θ)=logP(Y|θ)$ 的极大化，实现极大似然估计。每次迭代包括两步：E步，求期望，即求 $logP(Y,Z|θ)$ 关于 $P(Z|Y,θ^{(i)})$ 的期望：
$$
Q(\theta,\theta^{(i)})=\sum_{Z}{\log P(Y,Z\ |\ \theta)P(Z\ |\ Y,\theta^{(i)} )}
$$
称为 $Q$ 函数，这里 $θ^{(i)}$ 是参数的现估计值；$M$步，求极大，即极大化 $Q$ 函数得到参数的新估计值：
$$
P(Y\ |\ \theta^{(i+1)})\geq P(Y\ |\ \theta^{(i)})
$$
在构建具体的EM算法时，重要的是定义 $Q$ 函数。每次迭代中， EM算法通过极大化 $Q$ 函数来增大对数似然函数 $L(θ) $。

**2.**EM算法在每次迭代后均提高观测数据的似然函数值，即
$$
\theta^{(i+1)}=argmax_{\theta}Q(\theta,\theta^{(i)})
$$
在一般条件下EM算法是收敛的，但不能保证收敛到全局最优。

#### 混合高斯模型

混合高斯模型是一个概率聚类模型。训练GMM的想法是通过 ’$k$’ 高斯分布/簇的线性组合（也称为GMM的组成部分）来近似一类的概率分布。

最初，它通过K-means算法识别数据中的 $k$ 个簇，并分配相等的权重 $w = \frac{1}{k}$ 每个集群。 然后将$k$个高斯分布拟合到这 $k$ 个聚类。 $\mu$，$\sigma$ 和 $w$ 所有群集中的所有群集都会迭代更新，直到收敛为止。用于此估计的最普遍使用的方法是期望最大化（EM）算法。

混合高斯模型是指具有如下形式的概率分布模型：
$$
P(y\ |\ \theta)=\sum^{K}_{k=1}{\alpha_{k}\phi{(y\ |\ \theta_k)}}
$$
其中， $α_k$ 是系数， $α_k≥0 $， $∑_{k=1}^K{α_k}=1$ ； $ϕ(y\ |\ θ_k)$ 是高斯分布密度， $θ_k=(μ_k,σ_k^2)$
$$
\phi(y\ |\ \theta)={1\over \sqrt{2\pi}\sigma_k}{\exp(-{{(y-y_k)^2} \over 2\sigma^2_k})}
$$
称为第 $k$ 个分模型。一般混合模型可以由任意概率分布密度代替上式中的高斯分布密度。

- 混合高斯分布最明显的性质是它的多模态 ($K>1$) ，不同于高斯分布的单模态性质 $K=1$ 。这使得混合高斯模型足以描述很多显示出多模态性质的物理数据（包括语音数据），而单高斯分布则不适合。数据中的多模态性质可能来自多种潜在因素，每一个因素决定分布中一个特定的混合成分。如果因素被识别出来，那么混合分布就可以被分解成由多个因素独立分布组成的集合。 
- 原始语音数据经过短时傅里叶变换形式或者取倒谱后会成为特征序列，在忽略时序信息的条件下，混合高斯分布就非常适合拟合这样的语音特征。也就是可以以帧为单位，用混合高斯模型 （GMM）对语音特征进行建模

> 用GMM建模声学特征（Acoustic Feature）$O_1,O_2,...,O_n$，可以理解成：
>
> 每一个特征是由一个音素确定的，即不同特征可以按音素来聚类。由于在HMM中音素被表示为隐变量（状态），故等价于：
>
> **每一个特征是由某几个状态确定的，即不同特征可以按状态来聚类。**
>
> 则设$P(O\ |\ S_i)$符合正态分布，则根据GMM的知识，$O_1,O_2,...,O_n$实际上就是一个混合高斯模型下的采样值。

因此，**GMM被整合进HMM中，用来拟合基于状态的输出分布。**但若包含语音顺序信息的话，GMM就不再是一个好模型，因为它不包含任何顺序信息。若当给定HMM的一个状态后，若要对属于该状态的语音特征向量的概率分布进行建模，GMM仍不失为一个好的模型。

#### Reference

[1]《解析深度学习：语音识别实践》，俞栋 ，邓力 著

[2]《统计学习方法》，李航 著

[3] https://www.pianshen.com/article/1625338770/

[4] https://zhuanlan.zhihu.com/p/416632669

---

### 补充：语音识别的声学模型与因素（phoneme）

语音识别就分为三步：第一步，把帧识别成状态（难点）。第二步，把状态组合成音素。第三步，把音素组合成单词。第一步可以当做gmm做的，后面都是hmm做的。

#### 声学模型：

描述一种语言的基本单位被称为音素Phoneme，例如BRYAN这个词就可以看做是由B, R, AY, AX, N五个音素构成的。

单音素（monophone）：英语中貌似有50多个音素，可以用50几个HMM state来表示这些音素，这种表示方法就是context independent模型中的单音素monophone模式。

三因素（Triphone）：然而语音没有图像识别那幺简单，因为我们再说话的时候很多发音都是连在一起的，很难区分，所以一般用左中右三个HMM state来描述一个音素，也就是说BRYAN这个词中的R音素就变成了用B-R, R, R-AY三个HMM state来表示。这样BRYAN这个词根据上下文就需要15个state了，根据所有单词的上下文总共大概需要几千个HMM state，这种方式属于context dependent模型中的三音素triphone模式。

这个HMM state的个数在各家语音识别系统中都不一样，是一个需要调的参数。所以声学模型就是如何设置HMM state，对于信号中的每一frame抽怎样的特征，然后用训练什么分类器。



### GMMHMM参数

```python
class hmmlearn.hmm.GMMHMM(n_components=1, n_mix=1, min_covar=0.001, startprob_prior=1.0, transmat_prior=1.0, weights_prior=1.0, means_prior=0.0, means_weight=0.0, covars_prior=None, covars_weight=None, algorithm='viterbi', covariance_type='diag', random_state=None, n_iter=10, tol=0.01, verbose=False, params='stmcw', init_params='stmcw', implementation='log')
```

具有高斯混合排放的隐马尔可夫模型。

变量：

- monitor（收敛监测——ConvergenceMonitor）–用于**检查EM收敛**的监视器对象。 

- startprob（初始向量——array，shape（n_components，））–初始状态占用分布。 

- transmat（转移矩阵——array，shape（n_components，n_components））–状态之间转换概率的矩阵。 

- weights（权重——array，shape（n_components，n_mix））–每个状态的混合权重。 

- means（均值——array，shape（n_components，n_mix，n_features））–每个状态下每个混合物组分的平均参数。 

- covars（方差——array）– 

  每个状态下每个混合物组分的协方差参数。 

  形状取决于协变类型（covariance_type）： 

  - （n_components，n_mix）如果“球形(spherical)”， 
  - （n_components，n_mix，n_features）如果“diag”， 
  - （n_components，n_mix，n_features，n_feature）如果“full” 
  - （n_components，n_features，n_features）如果“绑定(tied)”。

```python
__init__(
    n_components=1, # 模型中的状态数,比如（3-6）个最长音节，3*3,每个音节分为{起始，中间，结尾}，y,i,e,r,s,a,n,w,u,l,q,b,j,h——14个实际出现的字母，总共26个字母，63个拼音字母，n_components=[3,5,6,9,14,26,63]
    	n_mix=1, # GMM中的状态数
    min_covar=0.001, # 协方差矩阵对角线上的下限，以防止过度拟合
    	startprob_prior=1.0, # 初始概率startprob_的Dirichlet先验分布参数 
    	transmat_prior=1.0, # 每行转移概率transmat_的Dirichlet先验分布参数。
    	weights_prior=1.0, # weights_的Dirichlet先验分布参数。
    	means_prior=0.0, # means_的正态先验分布的均值和精度。
    	means_weight=0.0, # means_的正态先验分布的均值和精度。
    	covars_prior=None, # 协方差矩阵covars_的先验分布参数。
    	covars_weight=None, # 协方差矩阵covars_的先验分布参数。
    algorithm='viterbi', # 解码器算法｛“viterbi”，“map”｝
    	covariance_type='diag', # 协方差参数类型｛“sperical”，“diag”，“full”，”tied“｝
    	random_state=None, # 随机数生成器实例，seed
    n_iter=10, # 要执行的最大迭代次数,n_iter=[10,20,50,100]
    tol=0.01, # 收敛阈值
    	verbose=False, # 是否将每次迭代收敛报告打印到sys.stderr。
    	params='stmcw', # 在（params）训练期间更新或在（init_param）训练之前初始化的参数。可以包含startprob的“s”、transmat的“t”、means的“m”、covars的“c”和GMM混合权重的“w”的任意组合。
    	init_params='stmcw', # 在训练期间（params）更新或在训练之前（init_param）初始化的参数。
    	implementation='log'# {“log”[向后兼容]，“scaling”[向前兼容]})
```

参数 :

- n_components（int）–模型中的状态数。 

- n_mix（int）–GMM中的状态数。 

- covariance_type（｛“sperical”，“diag”，“full”，”tied“｝，可选）

  要使用的协方差参数类型： 

  - “sperical”-每个状态使用适用于所有特征的单个方差值。 
  - “diag”-每个状态使用对角协方差矩阵（默认）。 
  - “full”-每个状态使用完全（即无限制）协方差矩阵。 
  - “tied”-每个状态的所有混合成分使用相同的全协方差矩阵（注意，这与GaussianHMM不同）。 

- min_covar（float，可选）–协方差矩阵对角线上的下限，以防止过度拟合。默认值为1e-3。 

- startprob_prior（array，shape（n_components，），可选）–`startprob_`的Dirichlet先验分布参数。

- transmat_prior（数组，形状（n_components，n_components），可选）–每行转移概率`transmat_`的Dirichlet先验分布参数。 

- weights_prior（array，shape（n_mix，），可选）–`weights_`的Dirichlet先验分布参数。 

- means_prior（数组，形状（n_mix，），可选）–means_的正态先验分布的平均值和精度。 

- means_weight（数组，形状（n_mix，），可选）–means_的正态先验分布的平均值和精度。 

- covars_prior（数组，形状（n_mix，），可选）– 协方差矩阵covars_的先验分布参数。 如果协变类型为“球形”或“diag”，则先验为逆伽马分布，否则为逆威斯哈特分布。 

- covars_weight（数组，形状（n_mix，），可选）– 协方差矩阵covars_的先验分布参数。 如果协变类型为“球形”或“diag”，则先验为逆伽马分布，否则为逆威斯哈特分布。 

- algorithm（｛“viterbi”，“map”｝，可选）–解码器算法。 

- random_state（RandomState或int种子，可选）–随机数生成器实例。 

- n_iter（int，可选）–要执行的最大迭代次数。 

- tol（浮动，可选）–收敛阈值。如果对数似然增益低于此值，EM将停止。 

- verbose（bool，可选）–是否将每次迭代收敛报告打印到sys.stderr。也可以使用monitor_属性诊断收敛。 

- params（字符串，可选）–在（params）训练期间更新或在（init_param）训练之前初始化的参数。可以包含startprob的“s”、transmat的“t”、means的“m”、covars的“c”和GMM混合权重的“w”的任意组合。默认为所有参数。 

- init_params（字符串，可选）–在训练期间（params）更新或在训练之前（init_param）初始化的参数。可以包含startprob的“s”、transmat的“t”、means的“m”、covars的“c”和GMM混合权重的“w”的任意组合。默认为所有参数。 

- implementation（字符串，可选）–确定是使用对数（“log”）还是使用缩放（“scaling”）实现前向后退算法。默认情况下，使用对数进行向后兼容。

语言解码用串接。

---

#### 最大似然估计和最大后验概率的不同

GMMHMM的参数：algorithm（｛“viterbi”，“map”｝

- “viterbi”：针对卷积码的最大似然译码

- “map”：最大后验概率译码

学派的比较

- 频率学派 - Frequentist - Maximum Likelihood Estimation (MLE，最大似然估计)
- 贝叶斯学派 - Bayesian - Maximum A Posteriori (MAP，最大后验估计)

##### 就举抛硬币的例子来说明。

概率函数 vs 似然函数

> 首先要知道P(x|θ)的意思：
> 输入有两个：x表示某一个具体的数据；θ表示模型的参数。
>
> **概率函数(probability function)**：θ是已知确定的，x是变量。它描述对于不同的样本点x，其出现概率是多少。(就是我们已经知道抛硬币只有两面，如果硬币质量均匀，就抛一次，样本空间是{正面，反面}，那么请问出现正面的概率是多少？0.5)
>
> **似然函数(likelihood function)**： x是已知确定的，θ是变量。它描述对于不同的模型参数，出现x这个样本点的概率是多少。（就是我们已经抛完硬币了，抛了十次，其中正面出现7次，那么根据这个结果请问抛硬币出现正面的概率是多少，才最可能得到我们现有的实验结果（十次出现七次正面）？0.7）

最大似然估计 vs 最大后验概率估计

>根据**最大似然估计**的思想：
>
>我们要求出合适的 $θ$ 让 $P(x|θ)$ 尽可能的大，那么 $θ$ 等于多少的时候，十次出现七次正面的情况最可能呢？算来算去，发现 $θ=0.7$。
>
>**但是**我们的主观认知判断与所得出的结果不一样。
>
>我们是人啊，我们是有思想的，一个硬币就正反面，我们的认知都是正面反面出现的概率各一半，结果求出个 $θ=0.7$ ，这不合理啊。
>
>于是**最大后验概率估计**出现了：
>
>我们带着**先验（主观色彩）**去求这个 $θ$，让$P(x_0|θ)P(θ)$最大，其中$P(x_0|θ)$就是我们上面看到的公式，$P(θ)$就是我们的先验（我们认为出现正面的概率是$0.5$）了，我们要求出一个既符合实验结果，又符合我们观念的$θ$，那么，最后求出的$θ$就在$0.5$到$0.7$之间（这也算是种妥协）。


链接：https://www.jianshu.com/p/640b35f328e4

---

### 批量建立数据集

```python
trainDir = '/kaggle/input/speech-recognition/train_audio'
trainDataSet = buildDataSet(trainDir)
print("Finish prepare the training data")

testDir = '/kaggle/input/speech-recognition/test_audio'
testDataSet = buildDataSet(testDir)
print("Finish prepare the test data")
```



### 建立训练混合高斯隐含马尔可夫模型

```python
def train_GMMHMM(dataset,states_num,n_iter):
    '''
    训练混合高斯隐马尔可夫模型
    '''
    GMMHMM_Models = {}

    for label in dataset.keys():
        model = hmm.GMMHMM(
            n_components=states_num, # 模型中的状态数。
            n_iter=n_iter, # 要执行的最大迭代次数。 
            algorithm='viterbi', #｛“viterbi”，“map”｝–解码器算法。
            n_mix = 3 # GMM中的状态数。)
        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
        trainData = np.vstack(trainData)
        model.fit(trainData, lengths=length)  # get optimal parameters
        GMMHMM_Models[label] = model
    return GMMHMM_Models
```

```python
def train_GMMHMM(dataset,states_num,n_iter):
    '''
    Training mixed gaussian hidden markov model
    -------------------------------------------
    Parameters:
    
    dataset: training dataset
    states_num: Number of states in the model
    n_iter: Maximum number of iterations to execute
    '''
    GMMHMM_Models = {}

    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=states_num,
                           n_iter=n_iter,
                           algorithm='viterbi',# Decoder algorithm
                           n_mix = 3)# Number of states in GMM
        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
        trainData = np.vstack(trainData)
        model.fit(trainData, lengths=length)  # get optimal parameters
        GMMHMM_Models[label] = model
    return GMMHMM_Models
```



### 建立模型预测函数

```python
def predict_GMMHMM(hmmModels,testDataSet):
    '''
    predict the model of GMMHMM
    -------------------------------------------
    Parameters:
    
    hmmModels: Trained mixed gaussian hidden markov model
    testDataSet: testing dataset
    '''
    score_cnt = 0
    for label in testDataSet.keys():
        feature = testDataSet[label]
        scoreList = {}
        for model_label in hmmModels.keys():
            model = hmmModels[model_label]
            score = model.score(feature[0])
            scoreList[model_label] = score
        predict = max(scoreList, key=scoreList.get)
        print("Test on true label ", label, ": predict result label is ", predict)
        if predict == label:
            score_cnt+=1
    hmm_accuracy = 100.0*score_cnt/len(testDataSet.keys())
    print("Final recognition rate is %.2f"%(hmm_accuracy), "%")
    return hmm_accuracy
```

### 参数分析

```python
def hmm_params_change(n_components_list,n_iter_list):
    '''
    search the best parameters of hmm model
    params:
    1. eg: n_components_list=[2,3,5,7,...]
    2. eg: n_iter_list=[10,20,50,...]
    '''
    accuracy_hmm=[]
    # change n_components
    for states_num in n_components_list:
        # change n_iter
        for n_iter in n_iter_list:
            time_add = 0
            accuracy_hmm_sum = 0
            initial_iter = 10
            # change initial value for iterations by random
            for i in range(initial_iter):
                aa=time.time()# start time
                # comfirm this iteration is use which parameters 
                print("n_components:",states_num,"n_iter:",n_iter) 
                # train the model
                hmmModels = train_GMMHMM(trainDataSet,states_num,n_iter) 
                # ignore the value error: startprob_ must sum to 1.0(got nan) issue, 
                #                         I can't figure it out right now
                try: 
                    hmm_accuracy=predict_GMMHMM(hmmModels,testDataSet)
                except ValueError:
                    train_GMMHMM(trainDataSet,states_num,n_iter)
                    try:
                        hmm_accuracy=predict_GMMHMM(hmmModels,testDataSet)
                    except ValueError:
                        initial_iter=initial_iter-1
                        hmm_accuracy = 0
                        pass # do nothing!
                    pass  # do nothing!
                bb=time.time()# end time
                cc=bb-aa #run time in seconds
                print("run time:",cc)
                time_add = time_add+cc
                accuracy_hmm_sum = accuracy_hmm_sum + hmm_accuracy
            # create dictionary for recoding data
            my_dict={'states_num':states_num,'n_iter':n_iter,
                     'accuracy(%)':accuracy_hmm_sum/initial_iter,
                     "run time":time_add/initial_iter}
            accuracy_hmm.append(my_dict)
    # create DataFrame
    df_accuracy_hmm=pd.DataFrame(accuracy_hmm)
    return df_accuracy_hmm
```

```python
n_components_list=[2,3,5,7]
n_iter_list=[10,20,50]
df_1 = hmm_params_change(n_components_list,n_iter_list)
```

```python
n_components_list=[2]
n_iter_list=[10,50,100,500,1000,5000]
df_2 = hmm_params_change(n_components_list,n_iter_list)
```



```python
n_components_list=[2,3,5,7]# 状态数
n_iter_list=[10,20,50]# 最大迭代次数
accuracy_hmm=[]# 初始化准确度
for states_num in n_components_list:# 选择不同的状态数
    for n_iter in n_iter_list:# 选择不同的最大迭代次数
        time_add = 0 # 累计时间
        accuracy_hmm_sum = 0 # 累计准确率
        initial_iter = 10 # 因为算法容易得到局部最优解，所以采用多次随机初始化，然后获得平均值
        for i in range(initial_iter): #循环随机初始化
            aa=time.time()# start time
            print("n_components:",states_num,"n_iter:",n_iter)# 确认改变的参数值
            hmmModels = train_GMMHMM(trainDataSet,states_num,n_iter)# 训练模型
            try:
                hmm_accuracy=predict_GMMHMM(hmmModels,testDataSet)# 测试模型
            except ValueError:# 忽视错误，“startprob_ must sum to 1.0(got nan)”，目前无法解决
                train_GMMHMM(trainDataSet,states_num,n_iter)# 重新尝试，随机初始化
                try:
                    hmm_accuracy=predict_GMMHMM(hmmModels,testDataSet)# 重新预测
                except ValueError:
                    initial_iter=initial_iter-1 #有错误，随机初始化次数减一
                    hmm_accuracy = 0 # 准确率为0
                    pass # do nothing!
                pass  # do nothing!
            bb=time.time()# end time
            cc=bb-aa #run time in seconds
            print("run time:",cc) # 输出运行时间
            time_add = time_add+cc #累加
            accuracy_hmm_sum = accuracy_hmm_sum + hmm_accuracy #累加
        my_dict={'states_num':states_num,'n_iter':n_iter,'accuracy(%)':accuracy_hmm_sum/initial_iter,"run time":time_add/initial_iter}# 建立数据字典
        accuracy_hmm.append(my_dict)# 添加数据字典
df_accuracy_hmm=pd.DataFrame(accuracy_hmm)# 建立DataFrame表格
```

结果

```
n_components: 2 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 2.450742721557617
n_components: 2 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 1.7114121913909912
n_components: 2 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 1.7271842956542969
n_components: 2 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 1.7537436485290527
n_components: 2 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 1.732576608657837
n_components: 2 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 1.71976900100708
n_components: 2 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 1.7238919734954834
n_components: 2 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 1.8002102375030518
n_components: 2 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 1.740821361541748
n_components: 2 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 1.71500825881958
n_components: 2 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 3.274686813354492
n_components: 2 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 2.290432929992676
n_components: 2 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 2.528331756591797
n_components: 2 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  9
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 2.3193912506103516
n_components: 2 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  9
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 2.3104538917541504
n_components: 2 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  9
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 2.2340760231018066
n_components: 2 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 2.212191581726074
n_components: 2 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 2.343439817428589
n_components: 2 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  9
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 2.305053234100342
n_components: 2 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  9
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 2.3227856159210205
n_components: 2 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 2.9792680740356445
n_components: 2 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  9
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 2.792550802230835
n_components: 2 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 3.975672960281372
n_components: 2 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 2.9432528018951416
n_components: 2 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 3.1413543224334717
n_components: 2 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 3.238600015640259
n_components: 2 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 2.9967219829559326
n_components: 2 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 3.0533602237701416
n_components: 2 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 3.191815137863159
n_components: 2 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 2.9578800201416016
n_components: 3 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 2.3297858238220215
n_components: 3 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 2.3051438331604004
n_components: 3 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 2.411924123764038
n_components: 3 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 3.125032424926758
n_components: 3 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 2.290712356567383
n_components: 3 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 2.288325071334839
n_components: 3 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 2.410796880722046
n_components: 3 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 2.265122652053833
n_components: 3 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  10
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 90.00 %
run time: 2.273853302001953
n_components: 3 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 2.274024724960327
n_components: 3 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 3.246736764907837
n_components: 3 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 3.440688371658325
n_components: 3 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  10
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  8
Final recognition rate is 80.00 %
run time: 3.1341094970703125
n_components: 3 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 3.260716438293457
n_components: 3 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 3.936065673828125
n_components: 3 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 3.2121517658233643
n_components: 3 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 3.2990505695343018
n_components: 3 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 3.324921131134033
n_components: 3 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 2.9998505115509033
n_components: 3 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  10
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 90.00 %
run time: 3.328256368637085
n_components: 3 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  10
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 90.00 %
run time: 4.140909194946289
n_components: 3 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  8
Final recognition rate is 60.00 %
run time: 4.26991605758667
n_components: 3 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 4.2523486614227295
n_components: 3 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  9
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  10
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 90.00 %
run time: 5.1558897495269775
n_components: 3 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  9
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 4.076594591140747
n_components: 3 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  8
Final recognition rate is 60.00 %
run time: 4.521424293518066
n_components: 3 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 4.310274124145508
n_components: 3 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  8
Final recognition rate is 70.00 %
run time: 4.617772817611694
n_components: 3 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  9
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  8
Final recognition rate is 60.00 %
run time: 4.6359148025512695
n_components: 3 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  9
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 5.028588771820068
n_components: 5 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  5
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  8
Final recognition rate is 70.00 %
run time: 3.442204713821411
n_components: 5 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  9
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 3.4014458656311035
n_components: 5 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  5
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 3.21246337890625
n_components: 5 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  8
Test on true label  9 : predict result label is  8
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  8
Final recognition rate is 60.00 %
run time: 3.390582799911499
n_components: 5 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  5
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 3.3926119804382324
n_components: 5 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  5
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 3.3093512058258057
n_components: 5 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  5
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 3.3076977729797363
n_components: 5 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 3.395550012588501
n_components: 5 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 4.179227352142334
n_components: 5 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 6.548856973648071
n_components: 5 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  5
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  3
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  8
Final recognition rate is 60.00 %
run time: 4.885706424713135
n_components: 5 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 5.107625961303711
n_components: 5 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 4.986398696899414
n_components: 5 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  9
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  5
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 4.974166631698608
n_components: 5 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  5
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 5.7362823486328125
n_components: 5 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  9
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  9
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  8
Final recognition rate is 70.00 %
run time: 5.075243234634399
n_components: 5 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  5
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 5.189988613128662
n_components: 5 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 5.027871370315552
n_components: 5 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  8
Final recognition rate is 70.00 %
run time: 5.168599367141724
n_components: 5 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  5
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  5
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 5.012248516082764
n_components: 5 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  5
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 7.80222487449646
n_components: 5 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 7.133404731750488
n_components: 5 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  9
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 8.422839403152466
n_components: 5 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 7.977307319641113
n_components: 5 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  5
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  8
Final recognition rate is 60.00 %
run time: 8.762477159500122
n_components: 5 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  9
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 7.90978217124939
n_components: 5 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  5
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 8.941989421844482
n_components: 5 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 10.07790756225586
n_components: 5 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  9
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 7.826136589050293
n_components: 5 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  8
Final recognition rate is 60.00 %
run time: 7.893152713775635
n_components: 7 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 4.371081113815308
n_components: 7 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 4.365537166595459
n_components: 7 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  2
Test on true label  9 : predict result label is  8
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 4.3671863079071045
n_components: 7 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  9
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  9
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 5.279240131378174
n_components: 7 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  8
Final recognition rate is 60.00 %
run time: 4.37165093421936
n_components: 7 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  3
Test on true label  9 : predict result label is  5
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 4.419567823410034
n_components: 7 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  5
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 4.474944114685059
n_components: 7 n_iter: 10
run time: 8.715610265731812
n_components: 7 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 4.4199793338775635
n_components: 7 n_iter: 10
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  5
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 5.151064395904541
n_components: 7 n_iter: 20
run time: 13.844327449798584
n_components: 7 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 6.711825132369995
n_components: 7 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  5
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  5
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 7.876823902130127
n_components: 7 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  8
Test on true label  9 : predict result label is  8
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 6.915038347244263
n_components: 7 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  9
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  9
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 6.727265357971191
n_components: 7 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  9
Test on true label  9 : predict result label is  5
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 6.752735137939453
n_components: 7 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  8
Final recognition rate is 60.00 %
run time: 6.695443630218506
n_components: 7 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  9
Test on true label  9 : predict result label is  8
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 7.1593451499938965
n_components: 7 n_iter: 20
run time: 13.1316397190094
n_components: 7 n_iter: 20
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  8
Test on true label  9 : predict result label is  8
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 6.62230110168457
n_components: 7 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  5
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 12.006242513656616
n_components: 7 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 10.590995788574219
n_components: 7 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 9.897172689437866
n_components: 7 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 9.955508470535278
n_components: 7 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  3
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 9.574098825454712
n_components: 7 n_iter: 50
run time: 21.186558723449707
n_components: 7 n_iter: 50
run time: 22.84959602355957
n_components: 7 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
run time: 11.62863564491272
n_components: 7 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  6
Test on true label  9 : predict result label is  7
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 10.268575191497803
n_components: 7 n_iter: 50
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  5
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  8
Test on true label  9 : predict result label is  8
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 70.00 %
run time: 11.84316635131836
```



```python
# DataFrame 按准确率降序排列
sort_df_accuracy=df_accuracy_hmm.sort_values(by="accuracy(%)",ascending=False)
sort_df_accuracy
```

![image-20230104232123054](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230104232123054.png)

|      | states_num | n_iter | accuracy(%) | run time  |
| ---: | ---------: | -----: | ----------: | --------- |
|    0 |          2 |     10 |   80.000000 | 1.807536  |
|    1 |          2 |     20 |   80.000000 | 2.414084  |
|    2 |          2 |     50 |   75.000000 | 3.127048  |
|    4 |          3 |     20 |   75.000000 | 3.318255  |
|   11 |          7 |     50 |   75.000000 | 16.225069 |
|    3 |          3 |     10 |   73.000000 | 2.397472  |
|    6 |          5 |     10 |   72.000000 | 3.757999  |
|    8 |          5 |     50 |   72.000000 | 8.274722  |
|    9 |          7 |     10 |   71.111111 | 5.548429  |
|    5 |          3 |     50 |   71.000000 | 4.500963  |
|   10 |          7 |     20 |   70.000000 | 10.304593 |
|    7 |          5 |     20 |   69.000000 | 5.116413  |



|      | states_num | n_iter | accuracy(%) | run time  |
| ---: | ---------: | -----: | ----------: | --------- |
|    0 |          2 |     10 |   80.000000 | 1.278616  |
|    1 |          2 |     20 |   78.000000 | 1.749310  |
|    6 |          5 |     10 |   77.000000 | 2.493409  |
|    2 |          2 |     50 |   76.000000 | 2.276151  |
|    9 |          7 |     10 |   75.714286 | 5.874691  |
|   11 |          7 |     50 |   75.714286 | 14.447396 |
|    5 |          3 |     50 |   74.000000 | 3.297206  |
|   10 |          7 |     20 |   73.750000 | 7.382898  |
|    4 |          3 |     20 |   73.000000 | 2.479817  |
|    8 |          5 |     50 |   71.000000 | 5.688183  |
|    3 |          3 |     10 |   70.000000 | 1.750696  |
|    7 |          5 |     20 |   70.000000 | 3.717229  |

![image-20230115195825617](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230115195825617.png)

#### 找到最好的参数进行重新训练与测试

```python
aa=time.time()# start time
hmmModels = train_GMMHMM(trainDataSet,2,10)
hmm_accuracy=predict_GMMHMM(hmmModels,testDataSet)
bb=time.time()# end time
cc=bb-aa #run time in seconds
print("Finish testing of the GMM_HMM models for digits 0-9")
print("run time:",cc)
```

```
Test on true label  7 : predict result label is  7
Test on true label  1 : predict result label is  1
Test on true label  8 : predict result label is  6
Test on true label  2 : predict result label is  2
Test on true label  6 : predict result label is  6
Test on true label  5 : predict result label is  5
Test on true label  10 : predict result label is  7
Test on true label  9 : predict result label is  9
Test on true label  3 : predict result label is  3
Test on true label  4 : predict result label is  4
Final recognition rate is 80.00 %
Finish testing of the GMM_HMM models for digits 0-9
run time: 1.7564122676849365
```



#### 输出模型的转移矩阵

```python
print('1:',hmmModels['1'].transmat_)# key is one of ['1','2','3','4','5','6','7','8','9','10']
print('2:',hmmModels['2'].transmat_)
print('3:',hmmModels['3'].transmat_)
print('4:',hmmModels['4'].transmat_)
print('5:',hmmModels['5'].transmat_)
print('6:',hmmModels['6'].transmat_)
print('7:',hmmModels['7'].transmat_)
print('8:',hmmModels['8'].transmat_)
print('9:',hmmModels['9'].transmat_)
print('10:',hmmModels['10'].transmat_)
```

```
1: [[9.56588321e-01 4.34116794e-02]
 [1.14649929e-20 1.00000000e+00]]
2: [[0.99174062 0.00825938]
 [0.05251792 0.94748208]]
3: [[0.93012347 0.06987653]
 [0.05784694 0.94215306]]
4: [[0.88349895 0.11650105]
 [0.00400046 0.99599954]]
5: [[0.95965962 0.04034038]
 [0.00627172 0.99372828]]
6: [[1.00000000e+00 3.19125639e-46]
 [5.93383292e-02 9.40661671e-01]]
7: [[1.00000000e+00 1.17784529e-79]
 [1.04017349e-01 8.95982651e-01]]
8: [[0.98658016 0.01341984]
 [0.06942037 0.93057963]]
9: [[1.00000000e+00 9.54087596e-38]
 [6.09296593e-02 9.39070341e-01]]
10: [[0.94063577 0.05936423]
 [0.04795068 0.95204932]]
```

```python
#隐藏状态的个数
print("隐藏状态的个数", model.n_components)
#均值矩阵：隐藏层状态
print("均值矩阵")
print(model.means_)
#协方差矩阵：对应隐藏层状态。对角线的值为该状态下的方差，方差越大，代表该状态的预测不可信
print("协方差矩阵")
print(model.covars_)
#状态转移矩阵：代表隐藏层状态的转移概率。
print("状态转移矩阵--A")
print(model.transmat_)
```

