# Dataframe描述信息

[09 DataFrame 描述信息](https://blog.csdn.net/weixin_47326735/article/details/116353134)

## 1. DataFrame的基础描述信息

DataFrame的基础属性：

index、cloumns、values、shape、ndim等

```python
df.shape # 行数 列数
df.dtypes # 列数据类型
df.ndim # 数据维度
df.index # 行索引
df.columns # 列索引
df.values # 对象值，二维ndarray数组
```

![image-20221031171150633](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031171150633.png)

![image-20221031171202012](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031171202012.png)

![image-20221031170510720](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031170510720.png)

![image-20221031170532335](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031170532335.png)

![image-20221031171010372](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031171010372.png)

## 2. DataFrame的整体描述信息

DataFrame的整体情况查询：

info、describe、head、tail

```python
df.head(3) # 显示头部几行，默认5行
df.tail(3) # 显示末尾几行，默认5行
df.info() # 相关信息概览：行数、列数、列索引、列非空值个数、列类型、列类型统计、内存占用
df.describe() # 快速综合统计结果：计数、均值、标准差、最大值、四分位数、最小值

```



![image-20221031171050351](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031171050351.png)![image-20221031171041238](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221031171041238.png)



