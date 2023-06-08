# 14-1-tensorflow函数

## tf.function

[一文搞懂tf.function](https://blog.csdn.net/jiangjunshow/article/details/119908750)

在tensorflow1.x的时候，代码默认的执行方式是graph execution（图执行），而从tensorflow2.0开始，改为了eager execution（饥饿执行）。

> eager execution就像搞一夜情，认识后就立即“执行”，而graph execution就像婚恋，认识后先憋着，不会立即“执行”，要经过了长时间的“积累”后，再一次性“执行”。

代码执行方式：

- graph execution（图执行）【在tensorflow1.x的时候】

  - 执行方式：会将所有代码组合成一个graph（图）后再执行。

  - graph是个数据结构

    里面定义了一些操作指令和数据，所以任何地方只要能解释这些操作和数据，那么就能运行这个模型。

    > 而graph 模式下，代码的执行效率要高一些；而且由于graph其实就是一个由操作指令和数据组成的一个数据结构，所以graph可以很方便地被导出并保存起来，甚至之后可以运行在其它非python的环境下（因为graph就是个数据结构，里面定义了一些操作指令和数据，所以任何地方只要能解释这些操作和数据，那么就能运行这个模型）；也正因为graph是个数据结构，所以不同的运行环境可以按照自己的喜好来解释里面的操作和数据，这样一来，解释后生成的代码会更加符合当前运行的环境，这里一来代码的执行效率就更高了。
    >
    > 假设graph里面包含了两个数据x和y，另外还包含了一个操作指令“将x和y相加”。当C++的环境要运行这个graph时，“将x和y相加”这个操作就会被翻译成相应的C++代码，当Java环境下要运行这个graph时，就会被解释成相应的Java代码。graph里面只是一些数据和指令，具体怎么执行命令，要看当前运行的环境。
    >
    > 除了上面所说的，graph还有很多内部机制使代码更加高效运行。总之，graph execution可以让[tensorflow](https://so.csdn.net/so/search?q=tensorflow&spm=1001.2101.3001.7020)模型运行得更快，效率更高，更加并行化，更好地适配不同的运行环境和运行设备。

- eager execution（饥饿执行）【tensorflow2.0开始】

  - 执行方式：立即执行每一步代码，非常的饥渴。

    > 在eager 模式下，代码的编写变得很自然很简单，而且因为代码会被立即执行，所以调试时也变得很方便。

- 两者的对比

  graph 虽然运行很高效，但是代码却没有eager 的简洁。

- 解决方案

  为了兼顾两种模式的优点，所以出现了tf.function。使用tf.function可以将eager 代码一键封装成graph。

  > 既然是封装成graph，那为什么名字里使用function这个单词内，不应该是tf.graph吗？
  >
  > 因为tf.function的作用就是将python function转化成包含了graph的tensorflow function。所以使用function这个单词也说得通。

下面的代码可以帮助大家更好地理解。

```python
import tensorflow as tf
import timeit
from datetime import datetime

#定义一个 Python function.
def a_regular_function(x,y,b):
    x = tf.matmul(x,y)
    x = x + b
    return x

# `a_function_that_uses_a_graph` 是一个 Tensorflow `Function`
a_function_that_uses_a_graph = tf.function(a_regular_function)

#定义一些tensorflow tensors.
x1= tf.constant([1.0,2.0])
y1= tf.constant([[2.0],[3.0]])
b1 = tf.constant(4.0)

orig_value = a_regular_function(x1,y1,b1).numpy()
# 在python中可以直接调用tensorflow Function。就像用python自己的function一样。
tf_funciton_value = a_function_that_uses_a_graph(x1,y1,b1).numpy()
assert(orig_value == tf_function_value)
```

同一个tensorflow function可能会生成不同的graph。因为每一个tf.Graph的input输入类型必须是固定的，所以如果在调用tensorflow function时传入了新的数据类型，那么这次的调用就会生成一个新的graph。输入的类型以及维度被称为signature（签名），tensorflow function就是根据签名来生成graph的，遇到新的签名就会生成新的graph。下面的代码可以帮助你理解。

最后我给出tf.function相关的几点建议：

- 当需要切换eager和graph模式时，应该使用tf.config.run_functions_eagerly来进行明显的标注。

- 应该在python function的外面创建tenforflow的变量（tf.Variables)，在里面修改它们的值。这条建议同样适用于其它那些使用tf.Variables的tenforflow对象（例如keras.layers,keras.Models,tf.optimizers）。

- 避免函数内部依赖外部定义的python变量。

- 应该尽量将更多的计算量代码包含在一个tf.function中而不是包含在多个tf.function里，这样可以将代码执行效率最大化。

- 最好是用tenforflow的数据类型作为function的输入参数。
  

## tf.squeeze()函数

tf.squeeze()函数用于从张量形状中移除大小为1的维度

```python
squeeze(
	input,
    axis=None,
    name=Nome,
    squeeze_dims=None,
)

```

给定张量输入，此操作返回相同类型的张量，并删除所有维度为1的维度。 如果不想删除所有维度1维度，可以通过指定squeeze_dims来删除特定维度1维度。 如果不想删除所有大小是1的维度，可以通过squeeze_dims指定。

参数：   

- input：A Tensor。输入要挤压。   
- axis：一个可选列表ints。默认为[]。如果指定，只能挤压列出的尺寸。维度索引从0开始。压缩非1的维度是错误的。必须在范围内[-- rank(input), rank(input))。   
- name：操作的名称(可选)。   
- squeeze_dims：现在是轴的已弃用的关键字参数。 
  - 函数返回值： 一Tensor。与输入类型相同。 包含与输入相同的数据，但具有一个或多个删除尺寸1的维度。 
  - 可能引发的异常：   ValueError：当两个squeeze_dims和axis指定。

### Example 1

该函数返回一个张量，这个张量是将原始input中所有维度为1的那些维都删掉的结果。

axis可以用来指定要删掉的为1的维度，此处要注意指定的维度必须确保其是1，否则会报错。

默认删除所有维度是1的维度。

```python
import tensorflow as tf
import numpy as np

value = np.floor(10*np.random.random((1,3,2,1,2)))
with tf.Session() as sess:
    print(sess.run(tf.shape(tf.squeeze(value))))
```

[3,2,2]

### Example 2

如果不想删除所有尺寸1尺寸，可以通过指定axis来删除特定维度1的维度。

```python
import tensorflow as tf
import numpy as np

value = np.floor(10*np.random.random((1,3,2,1,2)))
with tf.Session() as sess:
    print(sess.run(tf.shape(tf.squeeze(value,[0]))))
```

[3,2,1,2]