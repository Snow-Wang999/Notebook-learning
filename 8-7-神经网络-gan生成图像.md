# 8-7-神经网络-gan生成图像

## 上传项目

[Kaggle 新手入门必看，手把手教学](https://blog.csdn.net/qq_46450354/article/details/126835206)

1. 上传压缩包到上传数据集的input

   1. 尽量把文件放到文件夹下面，即原始文件夹下面再建立文件夹

2. copy 文件

   1. 转移.py文件

      ```python
      # import module we'll need to import our custom module
      from shutil import copyfile
       
      # copy our file into the working directory (make sure it has .py suffix)
      copyfile(src = "../input/create-function/my_functions.py", dst = "../working/my_functions.py")
      
      ```

      

   2. 转移整个项目文件夹

      ```python
      import shutil
      shutil.copytree(r'../input/vitcode/vision_transformer', r'./visio_transformer')
      ```

   3. 引入文件，运行py文件中的方法得到输出

      ```python
      from my_functions import *
      aaa()# my_functions中的函数
      ```

      

3. 一个一个文件写入

   1. 将该文件复制到一个 notebook 的 cell 中，然后在该 cell 的顶部添加一行代码：

      ```python
      %%writefile filename.py
      ```

      然后运行这个 cell，这个 cell 中的内容就会被写入到 output 的 `filename.py` 文件。后面就像往常一样了，直接`import` 使用即可。

      ![image-20230118172246243](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230118172246243.png)

   2. 如果想要更改其中内容的话，打开一个空的 cell，输入 `%load filename.py` 然后运行即可载入该文件的内容，改完后用上面的方法重新写入即可。

      ```python
      %load mixmodel.py
      ```

      ![image-20230118172257453](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230118172257453.png)

   3. 重新写入文件

      ```python
      %%writefile filename.py
      #######################
      # 以下是修改后的文件内容
      ```

      

## 修改文件

[kaggle添加、修改自己的模块和文件](https://blog.csdn.net/wxyczhyza/article/details/125488592)

如果要修改或者增减模块的文件，需要首先更新数据集版本，然后再notebook中刷新模块的版本号。具体操作如下：

1. 在kaggle的【datasets】中找到自定义的数据集，并打开数据集。

   ![image-20230118173650294](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230118173650294.png)

2. 打开数据集后，点击页面底部的【new version】，更新数据集版本：

   ![image-20230118173753245](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230118173753245.png)

3. 在弹出的窗口中，将修改过的模块【document】重新上传到数据集中，上传时系统会记录上传的时间作为版本号，见下图【Date Update 2022./06/28】：

   ![image-20230118173846719](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230118173846719.png)

4. 更新完数据集后，回到notebook，将光标移到模块上，会有【more actions选项】，并选择其中的【pin to version】更新模块。

   ![image-20230118174054403](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230118174054403.png)

   ![image-20230118173936502](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230118173936502.png)

5. 在弹出的窗口中选择要更新的版本号来更新。

   ![image-20230118174011732](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230118174011732.png)

6. 更新完模块后，模块会显示重新上传的文件内容。至此，模块的更新修改完毕。

   ![image-20230118174024752](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20230118174024752.png)

## 报错

### 库不存在

```python
import tensorflow.contrib.slim as slim
```

```
No module named 'tensorflow.contrib'
```

[tf.contrib.slim的介绍](https://www.cnblogs.com/japyc180717/p/9419184.html)

解决方案：

```python
!pip install  tf_slim
```

tensorflow中没有的模块，更新

```
tf.compat.v1.ConfigProto
tf.compat.v1.Session()
```



### 解压缩

gzip.open('xxx.gz')没有.gz结尾，因为kaggle路径不显示。代码中添加

```python
gzip.open(filename+'.gz')

traffic_station_df = pd.read_csv('../input/dot_traffic_stations_2015.txt.gz', compression='gzip', 
                                 header=0, sep=',', quotechar='"')
```

[Python压缩解压–gzip](https://blog.csdn.net/juzicode00/article/details/124722897)

#### gz 文件名

[loadlocal_mnist: A function for loading MNIST from the original ubyte files](http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/)

```python
from mlxtend.data import loadlocal_mnist
import platform

if not platform.system() == 'Windows':
    X, y = loadlocal_mnist(
            images_path='train-images-idx3-ubyte', 
            labels_path='train-labels-idx1-ubyte')

else:
    X, y = loadlocal_mnist(
            images_path='train-images.idx3-ubyte', 
            labels_path='train-labels.idx1-ubyte')
```



### 自动参数传递

[argparse.ArgumentParser()用法解析 ](https://www.cnblogs.com/yibeimingyue/p/13800159.html)

notebook中这样用

```python
!python main.py --dataset fashion-mnist --gan_type GAN --epoch 40 --batch_size 64
```

### github 加速下载

[如何提高gitHub下载速度](https://blog.csdn.net/zxyhj/article/details/126509761)

使用chrome插件 github加速

[cailuo](https://github.com/cailuo)/**[kaggle](https://github.com/cailuo/kaggle)**

https://www.kaggle.com/code/canhlu/gans-on-fashion-mnist-dataset

https://github.com/zalandoresearch/fashion-mnist