# 9-3-智能图像处理-大作业-sift

SIFT提取图像局部特征
SIFT算法是提取特征的一个重要算法，该算法对图像的扭曲，光照变化，视角变化，尺度旋转都具有不变性。SIFT算法提取的图像特征点数不是固定值，维度是统一的128维。

KMeans聚类获得视觉单词，构建视觉单词词典
现在得到的是所有图像的128维特征，每个图像的特征点数目还不一定相同（大多有差异）。现在要做的是构建一个描述图像的特征向量，也就是将每一张图像的特征点转换为特征向量。这儿用到了词袋模型，词袋模型源自文本处理，在这儿用在图像上，本质上是一样的。词袋的本质就是用一个袋子将所有维度的特征装起来，在这儿，词袋模型的维度需要我们手动指定，这个维度也就确定了视觉单词的聚类中心数。

SIFT和BOW算法结合的步骤

第一步：利用SIFT算法从不同类别的图像中提取视觉词汇向量，这些向量代表的是图像中局部不变的特征点；

第二步：将所有特征点向量集合到一块，利用K-Means算法合并词义相近的视觉词汇，构造一个包含K个词汇的单词表；

第三步：统计单词表中每个单词在图像中出现的次数，从而将图像表示成为一个K维数值向量。

## sift算法步骤梳理

简介：SIFT算法是检测和描述局部特征的一种方法，具有尺度不变性，对于光线，噪声等的容忍度相当高。即便少数几个物体也可以产生大量SIFT特征。

SIFT算法实质上是在不同尺度空间上查找关键点，并计算出关键点的方向。

http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

Lowe将SIFT算法分解为如下四步：

1. 尺度空间极值检测：搜索所有尺度上的图像位置。通过高斯微分函数来识别潜在的对于尺度和旋转不变的兴趣点。

2. 关键点定位：在每个候选的位置上，通过一个拟合精细的模型来确定位置和尺度。关键点的选择依据于它们的稳定程度。

3. 方向确定：基于图像局部的梯度方向，分配给每个关键点位置一个或多个方向。所有后面的对图像数据的操作都相对于关键点的方向、尺度和位置进行变换，从而提供对于这些变换的不变性。

4. 关键点描述：在每个关键点周围的邻域内，在选定的尺度上测量图像局部的梯度。这些梯度被变换成一种表示，这种表示允许比较大的局部形状的变形和光照变化。

## 算法步骤：

### 1. 构建尺度空间

SIFT算法是在不同的尺度空间上查找关键点，而尺度空间的获取需要使用高斯模糊来实现，Lindeberg等人已证明高斯卷积核是实现尺度变换的唯一变换核，并且是唯一的线性核。本节先介绍**高斯模糊**算法。

高斯核函数：
$$
G(x,y,\sigma)={{1}\over{\sqrt{2\pi}\sigma}}\exp{(-{{x^2+y^2}\over{2\sigma^2}})}
$$
输入图像通过高斯核函数连续的对尺度进行参数变换，最终得到多尺度空间序列。图像中某一尺度的空间函数由可变尺度的二维高斯函数$G(x,y,\sigma)$和原输入图像$I(x,y)$卷积得出：
$$
L(x,y,\sigma)=I(x,y)*G(x,y,\sigma)
$$

$$
G(x_i,y_i,\sigma)={{1}\over{\sqrt{2\pi}\sigma}}\exp{(-{{(x-x_i)^2+(y-y_i)^2}\over{2\sigma^2}})}
$$

其中 G(x,y,σ) 是尺度可变高斯函数（x，y）是空间坐标，σ是尺度坐标。σ大小决定图像的平滑程度，大尺度对应图像的概貌特征，小尺度对应图像的细节特征。大的σ值对应粗糙尺度(低分辨率)，反之，对应精细尺度(高分辨率)。为了有效的在尺度空间检测到稳定的关键点，提出了高斯差分尺度空间（DOG scale-space）。利用不同尺度的高斯差分核与图像卷积生成。
$$
D(x,y,\sigma)=[G(x,y,k\sigma)-G(x,y,\sigma)]*I(x,y)=L(x,y,k\sigma)-L(x,y,\sigma)
$$
- 高斯差分图像

![image-20230202154421279](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230202154421279.png)

### 图像金字塔的建立

对于一幅图像I,建立其在不同尺度(scale)的图像，也成为子八度（octave），这是为了scale-invariant，也就是在任何尺度都能够有对应的特征点，第一个子八度的scale为原图大小，后面每个octave为上一个octave降采样的结果，即原图的1/4（长宽分别减半），构成下一个子八度（高一层金字塔）。

每一个尺度图像层是先对图像进行高斯平滑，然后对图像做下采样。一般先将图像扩大一倍，在扩大的图像基础上构建高斯金字塔，然后您对该尺寸下图像进行高斯模糊，几幅模糊的图像集合构成了一个八度，然后对该八度下倒数第三张图片进行下采样，长和宽分别缩短一倍，图像面积变为原来的四分之一。以此类推。

- ![image-20230202190542035](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230202190542035.png)

$$
2^{i-1}(\sigma,k\sigma,k^2\sigma,...,k^{n-1}\sigma),k=2^{1\over S}
$$
尺度空间的所有取值，i为octave的塔数（第几个塔），s为每塔层数。

由图片size决定建几个塔，每塔几层图像(S一般为3-5层)。0塔的第0层是原始图像(或你double后的图像)，往上每一层是对其下一层进行Laplacian变换（高斯卷积，其中σ值渐大，例如可以是σ, k\*σ, k\*k\*σ…），直观上看来越往上图片越模糊。塔间的图片是降采样关系，例如1塔的第0层可以由0塔的第3层down sample得到，然后进行与0塔类似的高斯卷积操作。

下图所示不同σ下图像尺度空间：

![image-20230202184708981](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230202184708981.png)

在Lowe的论文中 ，将第0层的初始尺度定为1.6（最模糊），图片的初始尺度定为0.5（最清晰）. 在检测极值点前对原始图像的高斯平滑以致图像丢失高频信息，所以 Lowe 建议在建立尺度空间前首先对原始图像长宽扩展一倍，以保留原始图像信息，增加特征点数量。尺度越大图像越模糊。

### 高斯金字塔

模仿图像的不同尺度
生成步骤：高斯平滑-->对图像做下采样（一般先将图像扩大一倍，在扩大的图像基础上构建高斯金字塔，然后您对该尺寸下图像进行高斯模糊，几幅模糊的图像集合构成了一个八度，然后对该八度下倒数第三张图片进行下采样，长和宽分别缩短一倍，图像面积变为原来的四分之一。以此类推）
为了保持尺度空间的连续性，选倒数第三张进行下采样。根据下图公式可以计算得出第o组第S层的图像尺度，可以发现下一组的第o层图像恰好和上一组倒数第三张图一致，所以每一组的第0张图像只需要用上一层的倒数第三张进行下采样即可。
$$
\sigma(o,s)=\sigma_02^{o+{s\over S}},O \in[0,...,O-1],s \in [0,..,S+2]
$$
尺度空间的所有取值，o为octave的塔数（第几个塔），s为每塔层数

### 2. LoG近似DoG找到关键点<检测DOG尺度空间极值点>

为了寻找尺度空间的极值点，每一个采样点要和它所有的相邻点比较，看其是否比它的图像域和尺度域的相邻点大或者小。如图所示，中间的检测点和它同尺度的8个相邻点和上下相邻尺度对应的9×2个点共26个点比较，以确保在尺度空间和二维图像空间都检测到极值点。 一个点如果在DOG尺度空间本层以及上下两层的26个领域中是最大或最小值时，就认为该点是图像在该尺度下的一个特征点,如图所示。

![image-20230202155109822](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230202155109822.png)

同一组中的相邻尺度（由于k的取值关系，肯定是上下层）之间进行寻找。

![image-20230202155145674](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230202155145674.png)

s=3的情况

在极值比较的过程中，每一组图像的首末两层是无法进行极值比较的，**为了满足尺度****变化的连续性**（下面有详解）**，**我们在每一组图像的顶层继续用高斯模糊生成了 3 幅图像，高斯金字塔有每组S+3层图像。DOG金字塔每组有S+2层图像.

### 3. 除去不好的特征点

一旦找到了潜在的关键点位置，就必须对其进行细化以获得更准确的结果。他们使用尺度空间的泰勒级数展开来获得更准确的极值位置，如果该极值处的强度小于阈值(根据论文的说法为0.03)，则会被拒绝。这个阈值在OpenCV中称为*contrastThreshold*。

DoG对边缘有较高的响应，因此也需要去除边缘。为此，使用了类似于Harris 角点检测器的概念。他们使用2x2 Hessian 矩阵(H)来计算主曲率。我们从 Harris 角点检测器得知，对于边，一个特征值比另一个大。这里他们用了一个简单的函数，如果这个比值大于一个阈值 ( OpenCV 中称为 edgeThreshold )，则该关键点将被丢弃。论文中是10。

这一步本质上要去掉DoG局部曲率非常不对称的像素。

通过拟和三维二次函数以精确确定关键点的位置和尺度（达到亚像素精度），同时去除低对比度的关键点和不稳定的边缘响应点(因为DoG算子会产生较强的边缘响应)，以增强匹配稳定性、提高抗噪声能力，在这里使用近似Harris Corner检测器。

1. 空间尺度函数泰勒展开式如下：
   $$
   D(x)=D+{{\part D^T}\over{\part x}}Δx+{1\over2}Δx^T{{\part^2 D}\over{\part x^2}}Δx
   $$
   ![image-20230202190734178](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230202190734178.png)

   对上式求导,并令其为0,得到精确的位置, 得
   $$
   \hat{x}=-({{\part^2 D}\over{\part x^2}})^{-1}{{\part D}\over{\part x}}
   $$
   
2. 在已经检测到的特征点中,要去掉低对比度的特征点和不稳定的边缘响应点。去除低对比度的点：把公式(2)代入公式(1)，即在DoG Space的极值点处D(x)取值，只取前两项可得：
   $$
   D(\hat{x})=D+{1\over2}{{\part D^T}\over{\part x}}\hat{x}
   $$
   若$9|D(\hat{x})|\geq 0.03$，该特征就保留下来，否则就丢弃。

3. 边缘响应的去除

   一个定义不好的高斯差分算子的极值在横跨边缘的地方有较大的主曲率，而在垂直边缘的方向有较小的主曲率。主曲率通过一个2×2 的Hessian矩阵H求出:
   $$
   H = \begin{bmatrix} D_{xx} & D_{xy} \\D_{xy} & D_{yy} \end{bmatrix}
   $$
   导数由采样点相邻差估计得到。

   D的主曲率和H的特征值成正比，令α为较大特征值，β为较小的特征值，则
   $$
   Tr(H)=D_{xx}+D_{yy}=\alpha+\beta
   $$

   $$
   Det(H)=D_{xx}D_{yy}-(D_{xy})^2=\alpha\beta
   $$

   令$α=rβ$，则
   $$
   {{Tr(H)^2}\over{Det(H)}}={{(\alpha+\beta)^2}\over{\alpha\beta}}={{(r\beta+\beta)^2}\over{r\beta^2}}={{(r+1)^2}\over{r}}=10
   $$
    ${{(r + 1)^2}\over r}$的值在两个特征值相等的时候最小，随着r的增大而增大，因此，为了检测主曲率是否在某域值r下，只需检测
   $$
   {{Tr(H)^2}\over{Det(H)}}<{{(r + 1)^2}\over r}
   $$
   if ${{(\alpha+\beta)^2}\over{\alpha\beta}}> {{(r + 1)^2}\over r}$, throw it out.  在Lowe的文章中，取$r＝10$。

4. 给特征点赋值一个128维方向参数

   上一步中确定了每幅图中的特征点，为每个特征点计算一个方向，依照这个方向做进一步的计算， 利用关键点邻域像素的梯度方向分布特性为每个关键点指定方向参数，使算子具备旋转不变性。

   梯度幅值
   $$
   m(x,y)=\sqrt{(L(x+1,y)-L(x-1,y))^2+(L(x,y+1)-L(x,y-1))^2}
   $$

   $$
   \theta(x,y)=\alpha tan2({{L(x,y+1)-L(x,y-1)}\over{L(x+1,y)-L(x-1,y)}})
   $$

   m(x,y)和theta(x,y)是(x,y)处梯度的模值和方向公式。其中L所用的尺度为每个关键点各自所在的尺度。至此，图像的关键点已经检测完毕，每个关键点有三个信息：位置，所处尺度、方向，由此可以确定一个SIFT特征区域。

梯度直方图的范围是0～360度，其中每10度一个柱，总共36个柱。随着距中心点越远的领域其对直方图的贡献也响应减小.Lowe论文中还提到要使用高斯函数对直方图进行平滑，减少突变的影响。

在实际计算时，我们在以关键点为中心的邻域窗口内采样，并用直方图统计邻域像素的梯度方向。梯度直方图的范围是0～360度，其中每45度一个柱，总共8个柱, 或者每10度一个柱，总共36个柱。Lowe论文中还提到要使用高斯函数对直方图进行平滑，减少突变的影响。直方图的峰值则代表了该关键点处邻域梯度的主方向，即作为该关键点的方向。

![image-20230202185314824](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230202185314824.png)

![image-20230202183524757](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230202183524757.png)

直方图中的峰值就是主方向，其他的达到最大值80%的方向可作为辅助方向。

![image-20230202183555420](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230202183555420.png)

由梯度方向直方图确定主梯度方向

该步中将建立所有scale中特征点的描述子（128维）

![image-20230202183632707](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230202183632707.png)

确定峰值，并为关键点指定方向和大小之和。 用户可以基于关键点分配的大小之和来选择阈值以排除关键点。

![image-20230202183701756](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230202183701756.png)

关键点描述子的生成步骤：

1. 旋转方向：将坐标轴旋转为关键点的方向，以确保旋转不变性
2. 生成描述子：对于一个关键点产生128个数据，即最终形成128维的SIFT特征向量
3. 归一化处理：将特征向量的长度归一化，则可以进一步去除光照变化的影响。

通过对关键点周围图像区域分块，计算块内梯度直方图，生成具有独特性的向量，这个向量是该区域图像信息的一种抽象，具有唯一性。

### 5. 关键点描述子的生成

首先将坐标轴旋转为关键点的方向，以确保旋转不变性。以关键点为中心取8×8的窗口。

![image-20230202184237564](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230202184237564.png)

Figure.16*16的图中其中1/4的特征点梯度方向及scale，右图为其加权到8个主方向后的效果。

图左部分的中央为当前关键点的位置，每个小格代表关键点邻域所在尺度空间的一个像素，利用公式求得每个像素的梯度幅值与梯度方向，箭头方向代表该像素的梯度方向，箭头长度代表梯度模值，然后用高斯窗口对其进行加权运算。
图中蓝色的圈代表高斯加权的范围（越靠近关键点的像素梯度方向信息贡献越大）。然后在每4×4的小块上计算8个方向的梯度方向直方图，绘制每个梯度方向的累加值，即可形成一个种子点，如图右部分示。此图中一个关键点由2×2共4个种子点组成，每个种子点有8个方向向量信息。这种邻域方向性信息联合的思想增强了算法抗噪声的能力，同时对于含有定位误差的特征匹配也提供了较好的容错性。


计算keypoint周围的16*16的window中每一个像素的梯度，而且使用高斯下降函数降低远离中心的权重。

![image-20230202184313921](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230202184313921.png)

![image-20230202190940085](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230202190940085.png)

在每个4*4的1/16象限中，通过加权梯度值加到直方图8个方向区间中的一个，计算出一个梯度方向直方图。

这样就可以对每个feature形成一个4\*4\*8=128维的描述子，每一维都可以表示4\*4个格子中一个的scale/orientation. 将这个向量归一化之后，就进一步去除了光照的影响。
5. 根据SIFT进行Match

生成了A、B两幅图的描述子，（分别是k1\*128维和k2\*128维），就将两图中各个scale（所有scale）的描述子进行匹配，匹配上128维即可表示两个特征点match上了。

> 实际计算过程中，为了增强匹配的稳健性，Lowe建议对每个关键点使用4×4共16个种子点来描述，这样对于一个关键点就可以产生128个数据，即最终形成128维的SIFT特征向量。此时SIFT特征向量已经去除了尺度变化、旋转等几何变形因素的影响，再继续将特征向量的长度归一化，则可以进一步去除光照变化的影响。 当两幅图像的SIFT特征向量生成后，下一步我们采用关键点特征向量的欧式距离来作为两幅图像中关键点的相似性判定度量。取图像1中的某个关键点，并找出其与图像2中欧式距离最近的前两个关键点，在这两个关键点中，如果最近的距离除以次近的距离少于某个比例阈值，则接受这一对匹配点。降低这个比例阈值，SIFT匹配点数目会减少，但更加稳定。为了排除因为图像遮挡和背景混乱而产生的无匹配关系的关键点,Lowe提出了比较最近邻距离与次近邻距离的方法,距离比率ratio小于某个阈值的认为是正确匹配。因为对于错误匹配,由于特征空间的高维性,相似的距离可能有大量其他的错误匹配,从而它的ratio值比较高。Lowe推荐ratio的阈值为0.8。但作者对大量任意存在尺度、旋转和亮度变化的两幅图片进行匹配，结果表明ratio取值在0. 4~0. 6之间最佳，小于0. 4的很少有匹配点，大于0. 6的则存在大量错误匹配点。(如果这个地方你要改进，最好给出一个匹配率和ration之间的关系图，这样才有说服力)作者建议ratio的取值原则如下:

> ratio=0. 4　对于准确度要求高的匹配；
> ratio=0. 6　对于匹配点数目要求比较多的匹配； 
> ratio=0. 5　一般情况下。
> 也可按如下原则:当最近邻距离<200时ratio=0. 6，反之ratio=0. 4。ratio的取值策略能排分错误匹配点。

当两幅图像的SIFT特征向量生成后，下一步我们采用关键点特征向量的欧式距离来作为两幅图像中关键点的相似性判定度量。取图像1中的某个关键点，并找出其与图像2中欧式距离最近的前两个关键点，在这两个关键点中，如果最近的距离除以次近的距离少于某个比例阈值，则接受这一对匹配点。降低这个比例阈值，SIFT匹配点数目会减少，但更加稳定。

快速最近邻近似匹配

两个图像之间的关键点通过识别它们最近的邻居来匹配。但在某些情况下，第二个最接近的匹配可能非常接近第一个。这可能是由于噪音或其他原因造成的。在这种情况下，取最近距离与次近距离的比值。如果大于0.8，则拒绝它们。根据这篇论文，它消除了大约90%的错误匹配，而只丢弃了5%的正确匹配。

## 词袋模型表示图像

Bag-of-words模型简介
Bag-of-words模型是信息检索领域常用的文档表示方法。在信息检索中，BOW模型假定对于一个文档，忽略它的单词顺序和语法、句法等要素，将其仅仅看作是若干个词汇的集合，文档中每个单词的出现都是独立的，不依赖于其它单词是否出现。也就是说，文档中任意一个位置出现的任何单词，都不受该文档语意影响而独立选择的。例如有如下两个文档：

 

     1：Bob likes to play basketball, Jim likes too.
    
     2：Bob also likes to play football games.

 




    基于这两个文本文档，构造一个词典：

 




     Dictionary = {1:”Bob”, 2. “like”, 3. “to”, 4. “play”, 5. “basketball”, 6. “also”, 7. “football”, 8. “games”, 9. “Jim”, 10. “too”}。

 




    这个词典一共包含10个不同的单词，利用词典的索引号，上面两个文档每一个都可以用一个10维向量表示（用整数数字0~n（n为正整数）表示某个单词在文档中出现的次数）：

 




     1：[1, 2, 1, 1, 1, 0, 0, 0, 1, 1]
    
     2：[1, 1, 1, 1 ,0, 1, 1, 1, 0, 0]

 




    向量中每个元素表示词典中相关元素在文档中出现的次数(下文中，将用单词的直方图表示)。不过，在构造文档向量的过程中可以看到，我们并没有表达单词在原来句子中出现的次序（这是本Bag-of-words模型的缺点之一，不过瑕不掩瑜甚至在此处无关紧要）。

 



Bag-of-words模型的应用
Bag-of-words模型的适用场合

现在想象在一个巨大的文档集合D，里面一共有M个文档，而文档里面的所有单词提取出来后，一起构成一个包含N个单词的词典，利用Bag-of-words模型，每个文档都可以被表示成为一个N维向量，计算机非常擅长于处理数值向量。这样，就可以利用计算机来完成海量文档的分类过程。

考虑将Bag-of-words模型应用于图像表示。为了表示一幅图像，我们可以将图像看作文档，即若干个“视觉词汇”的集合，同样的，视觉词汇相互之间没有顺序。


由于图像中的词汇不像文本文档中的那样是现成的，我们需要首先从图像中提取出相互独立的视觉词汇，这通常需要经过三个步骤：（1）特征检测，（2）特征表示，（3）单词本的生成。


通过观察会发现，同一类目标的不同实例之间虽然存在差异，但我们仍然可以找到它们之间的一些共同的地方，比如说人脸，虽然说不同人的脸差别比较大，但眼睛，嘴，鼻子等一些比较细小的部位，却观察不到太大差别，我们可以把这些不同实例之间共同的部位提取出来，作为识别这一类目标的视觉词汇。

而SIFT算法是提取图像中局部不变特征的应用最广泛的算法，因此我们可以用SIFT算法从图像中提取不变特征点，作为视觉词汇，并构造单词表，用单词表中的单词表示一幅图像。

### 创建BOW训练器，指定k-means参数k   

把处理好的特征数据全部合并，利用聚类把特征词分为若干类，此若干类的数目由自己设定，每一类相当于一个视觉词汇

要准备好需要的数据，还需要把图片数据转换为SIFT特征数据的算法。

此外，还需要把SIFT特征数据转换为词袋数据的算法。

开始准备数据

### 分类算法的比较

本文采取的算法分别是支持向量机（SVM）分类模型、随机森林（Random forest）分类模型和集成算法中的投票（Voting）分类模型。

支持向量机是一种二分类模型算法，它的基本模型是定义在特征空间的间隔最大的线性分类器，说白了就是在中间画一条线，然后以 “最好地” 区分这两类点。以至如果以后有了新的点，这条线也能做出很好的分类。SVM 适合中小型数据样本、非线性、高维的分类问题。

随机森林是一种监督学习算法，可用于分类和回归。 但是，它主要用于分类问题。 众所周知，森林由树木组成，更多的树木意味着更坚固的森林。 同样，随机森林算法在数据样本上创建决策树，然后从每个样本中获取预测，最后通过投票选择最佳解决方案。 它是一种集成方法，比单个决策树要好，因为它通过对结果求平均值来减少过度拟合。

在所有集成学习方法中，最直观的是多数投票。因为其目的是输出基础学习者的预测中最受欢迎（或最受欢迎）的预测。多数投票是最简单的集成学习技术，它允许多个基本学习器的预测相结合。与选举的工作方式类似，该算法假定每个基础学习器都是投票者，每个类别都是竞争者。为了选出竞争者为获胜者，该算法会考虑投票。将多种预测与投票结合起来的主要方法有两种：一种是硬投票，另一种是软投票。本文介绍软投票。

软投票考虑了预测类别的可能性。为了结合预测结果，软投票计算每个类别的平均概率，并假设获胜者是具有最高平均概率的类别。如果所有的分类器都能够估计类别的概率(即sklearn中的predict_proba()方法)，那么可以求出类别的概率平均值，投票分类器将具有最高的平均概率的类作为自己的预测。这称为软投票。

如果选择的分类模型在上图detector函数的选择范围之外，本文另创建了一个拟合曲线的功能函数。

其中的Precision-Recall(PR)曲线的函数方法如下：

![image-20230314182705978](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230314182705978.png)

Precision-Recall(PR)曲线中，以Recall为x轴，Precision为y轴。

P-R曲线，被称为precision(查准率)和[recall](https://so.csdn.net/so/search?q=recall&spm=1001.2101.3001.7020)(查全率)曲线，以查准率为纵轴，以查全率为横轴，绘制出二维图像。平均精度AP就是P-R曲线下的面积。

通过改变不同的置信度阈值，可以获得多对Precision和Recall值，Recall值放X轴，Precision值放Y轴，可以画出一个Precision-Recall曲线，简称P-R曲线。

我们当然希望检测的结果P越高越好，R也越高越好，但事实上这两者在某些情况下是矛盾的。比如极端情况下，我们只检测出了一个结果，且是准确的，那么Precision就是100%，但是Recall就很低；而如果我们把所有结果都返回，那么必然Recall必然很大，但是Precision很低。

因此在不同的场合中需要自己判断希望P比较高还是R比较高。如果是做实验研究，可以绘制Precision-Recall曲线来帮助分析。

Precision-Recall曲线可以衡量目标检测模型的好坏，但不便于模型和模型之间比较。在Precision-Recall曲线基础上，通过计算每一个recall值对应的Precision值的平均值，可以获得一个数值形式的评估指标：AP(Average Precision），用于衡量的是训练出来的模型在感兴趣的类别上的检测能力的好坏。我们现在常用的算法是每个Recall的区间内我们只取这个区间内Precision的最大值然后和这个区间的长度做乘积，最后体现出来就是一系列的矩形的面积。

## 主程序

此处，假设contrastThreshold=0.04, clusterCount=32。正式运行主程序，从预处理数据到比较不同分类模型的目标检测的AP(Average Precision）值，得到初始结果。

利用超参数搜索，找到随机森林和支持向量机的最佳AP值。

寻找词袋模型最佳的聚类中心个数（clusterCount）

寻找SIFT算法最佳的对比阙值（contrastThreshold）

最后，用最佳的参数：contrastThreshold=0.04, clusterCount=8。运行主程序，从预处理数据到比较不同分类模型的目标检测的AP(Average Precision）值，得到最终结果。

主程序初始运行的结果。

reference:

1. [SIFT特征提取分析](https://blog.csdn.net/abcjennifer/article/details/7639681)

2. 第九章三续：SIFT算法的应用--目标识别之Bag-of-words模型
   https://blog.51cto.com/u_1875963/3410979
