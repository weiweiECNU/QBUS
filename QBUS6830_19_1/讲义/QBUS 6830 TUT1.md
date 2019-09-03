# QBUS6830 Lab1

## 读取数据

MATLAB 读取数据既可以通过 `导入数据助手`也可通过代码解决。但本身 `导入数据助手`也是通过调用代码处理数据的。

### 导入数据助手

1. 按菜单当中的 `导入数据`，选择需要导入的文件。
2. 进入导入界面后，修改导入文件的变量名，类型和数据范围

![](/Users/apple/Downloads/2019s1/QBUS6830/week1/pic/WX20190309-163917@2x.png)

* 注意 导入数据的类型不同可能会造成数据的丢失，尤其是数值矩阵中的日期和时间数据

  ![](/Users/apple/Downloads/2019s1/QBUS6830/week1/pic/WX20190309-164534@2x.png)

  * 如果想导入日期和时间数据，应该先导入成 table 然后再用[`table2array`](https://ww2.mathworks.cn/help/matlab/ref/table2array.html)转化成 datetime array，或者用 table 的[点索引](https://ww2.mathworks.cn/help/matlab/matlab_prog/access-data-in-a-table.html)获得 datetime array 

### 代码导入数据

MATLAB 还可以用函数来读取各类数据文件，但是不是很好用

TODO



## Array Matrix Cell 

 在 MATLAB 运算的所有数据类型，都是按照 array 或 matrix 的形式进行存储和计算的。

### Array

#### 构建 Row vector 和 Column vector

用中括号将被空格或者逗号分割的数列赋值给 Row vector

```matlab
%构建 行向量
>> a = [1 2 3 4] % 或者 [1,2,3,4]

a =

     1     2     3     4

%构建 列向量
>> a = [1 ;2; 3; 4]
a =

     1
     2
     3
     4
```

#### 引用向量元素

Array 有和 python 类似的索引引用

```matlab
>> a(3)

ans =

     3
```

* 注意这里的索引用的是 小括号，而不是 python 里的中括号

同样用来代表全部的 冒号

```matlab
>> a(:)

ans =

     1
     2
     3
     4
```

以及引用一个范围的数据

```matlab
>> a(2:4)

ans =

     2
     3
     4
```

`end`检索最后一位元素

```matlab
>> a(end)

ans =

     4
```



#### 向量加减法

在 MATLAB 中当进行两个向量的加法与减法的时候，这两个向量的元素必须有相同的类型和数量。

```matlab
>> b = [5;6;7;8]

b =

     5
     6
     7
     8

>> a + b

ans =

     6
     8
    10
    12
>> b = [5,6,7,8]

b =

     5     6     7     8

>> a + b

ans =

     6     7     8     9
     7     8     9    10
     8     9    10    11
     9    10    11    12
```

#### 标量向量乘法/加法

让一个数字乘以/加一个向量。标量乘法/加法会产生相同类型的新的一个向量，原先的向量的每个元素乘以/加以数量

```matlab
>> m = 5 * a

m =

     5
    10
    15
    20
    
>> 5+m

ans =

    10
    15
    20
    25
```

#### 转置

向量和矩阵 的转置 都是用的 `‘`运算符

```matlab
m'
ans =
5    10    15    20 
```

#### 添加向量

将原向量 m 和要添加的向量n放在同一行：[m n] 或者[m , n]

将原向量 m 和要添加的向量n组合成新矩阵：[m;n]，m和 n 的长度应该一样

```
>> m'

ans =

     5    10    15    20

>> n = [3,6,9,12]

n =

     3     6     9    12

>> [m' n]

ans =

     5    10    15    20     3     6     9    12
```

#### 点积  dot(a , b)

```matlab
>> v1 = [2 3 4];
v2 = [1 2 3];
dp = dot(v1, v2);
>> dp

dp =

    20
```

#### 生成等差元素向量

`a = [s:l:f]`

生成第一个元素是 s，最后一个元素是 f, 公差是 l 的向量

```matlab
>> a = [1:2:19]

a =

     1     3     5     7     9    11    13    15    17    19
```

###Matrix

TODO

###Cell

TODO

## 画图

### Plot 二维线形图

`plot(X,Y,LineSpec)`

[Plot](https://ww2.mathworks.cn/help/matlab/ref/plot.html) 函数生成二维线图

 `X` 和 `Y` 长度必须相同，`Linespec` 字符串设置线的形状、标记和颜色

| Color   | Code  |
| ------- | ----- |
| White   | **w** |
| Black   | **k** |
| Blue    | **b** |
| Red     | **r** |
| Cyan    | **c** |
| Green   | **g** |
| Magenta | **m** |
| Yellow  | **y** |

`hand on`

在同一张图上画线，而不是生成新的图像

`hand off`

停止在同一张图上画线，下一次画图将生成新的图像



## 随机数函数

### Rand

`rand(s1,s2,…)`

[rand](https://ww2.mathworks.cn/help/matlab/ref/rand.html?searchHighlight=rand&s_tid=doc_srchtitle)生成尺寸为(s1 * s2 * … )的满足U(0,1)均匀分布的随机数矩阵

```matlab
> rand(5)

ans =

    0.9058    0.2785    0.9706    0.4218    0.0357
    0.1270    0.5469    0.9572    0.9157    0.8491
    0.9134    0.9575    0.4854    0.7922    0.9340
    0.6324    0.9649    0.8003    0.9595    0.6787
    0.0975    0.1576    0.1419    0.6557    0.7577
```

```matlab
rand(3,2)

ans =

    0.7431    0.1712
    0.3922    0.7060
    0.6555    0.0318
```

### Randn

`randn(s1,s2,…)`

[rand](https://ww2.mathworks.cn/help/matlab/ref/randn.html?searchHighlight=randn&s_tid=doc_srchtitle)生成尺寸为(s1 * s2 * … )的满足N(0,1)正态分布的随机数矩阵

###  normrnd

` normrnd(mu,sigma,sz1,...,szN)`

[normrnd](https://ww2.mathworks.cn/help/stats/normrnd.html?searchHighlight=normrnd&s_tid=doc_srchtitle)产生尺寸为(s1 * s2 * … )的满足N(mu,sigma)正态分布的随机数矩阵

### trnd

`r = trnd(nu,[m,n,…])`

[trnd]() 生成尺寸为(s1 * s2 * … )的满足nu 自由度的t分布的随机数矩阵



### Task 用到的几个函数

### .^2 和 ^2

`.^n` 按元素求幂

`^n`矩阵幂 

其他[运算符和特殊字符](https://ww2.mathworks.cn/help/matlab/matlab_prog/matlab-operators-and-special-characters.html#bvg3oy_-2?s_tid=doc_ta)

```matlab
>> b = [1,2,3;4,5,6;7,8,9]

b =

     1     2     3
     4     5     6
     7     8     9

>> b.^2

ans =

     1     4     9
    16    25    36
    49    64    81

>> b^2

ans =

    30    36    42
    66    81    96
   102   126   150
```

### diff

```matlab
Y = diff(X)
Y = diff(X,n)
Y = diff(X,n,dim)
```

` Y = diff(X)` 计算沿大小不等于 1 的第一个数组维度的 `X` 相邻元素之间的差分：

`Y = diff(X,n)` 通过递归应用 `diff(X)` 运算符 `n` 次来计算第 n 个差分。

`Y = diff(X,n,dim)` 是沿 `dim` 指定的维计算的第 n 个差分。`dim` 输入是一个正整数标量。

以一个二维 p x m 输入数组 `A` 为例：

* `diff(A,1,1)` 会对 `A` 的列中的连续元素进行处理，然后返回 (p-1)xm 的差分矩阵。
* `diff(A,1,2)` 会对 `A` 的行中的连续元素进行处理，然后返回 px(m-1) 的差分矩阵。

### prctile

`Y = prctile(X,p)`

返回数组X中元素的百分位数。

P为分位值。表示被调查群体中有p%的数据小于此数值X



## Lab 提到但目前没用到的函数

| command                                                      | function                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| mean(x)                                                      | 平均数                                                       |
| var(x)                                                       | 方差                                                         |
| [qqplot](https://ww2.mathworks.cn/help/stats/qqplot.html?searchHighlight=qqplot&s_tid=doc_srchtitle)(x) | 分位数 - 分位数图                                            |
| [autocorr](https://ww2.mathworks.cn/help/econ/autocorr.html?searchHighlight=autocorr&s_tid=doc_srchtitle)(x) | 绘制具有置信界限的单变量随机时间序列y的样本自相关函数（ACF） |
| [parcorr](https://ww2.mathworks.cn/help/econ/parcorr.html?searchHighlight=parcorr&s_tid=doc_srchtitle)(x) | 绘制具有置信界限的单变量随机时间序列y的样本部分自相关函数（PACF）。 |
| [price2ret]()(x)                                             |                                                              |
| length(x)                                                    |                                                              |
| log(x)                                                       |                                                              |
| log10(x)                                                     |                                                              |
| diff(x)                                                      |                                                              |
| [normcdf](https://ww2.mathworks.cn/help/stats/normcdf.html?searchHighlight=normcdf&s_tid=doc_srchtitle) | 正态分布函数                                                 |
| [norminv](https://ww2.mathworks.cn/help/stats/norminv.html?searchHighlight=norminv&s_tid=doc_srchtitle) | 正态逆分布函数                                               |
| [Tcdf]()                                                     | t分布函数                                                    |
| [Tinv](https://ww2.mathworks.cn/help/stats/tinv.html?searchHighlight=Tinv&s_tid=doc_srchtitle) | t逆分布函数                                                  |


## 一些有用技巧

1. 注释掉一段程序：%{、%}

2. help 命令名

   调用官方的帮助文件

   doc 命令名

   调用官方的命令文档，比 help 更详细而且有例子

3. clc 清屏

   清除命令窗口中的所有输入和输出信息，不影响命令的历史记录。

   clear 和clear all

   clear 变量名：可以清除workspace中的无用的变量，尤其是一些特别大的矩阵，不用时及时清理，可以减少内存占用。

   clear all 清除所有的变量，使workspace一无所有，当重新开始一次算法验证时，最好执行一次，让workspace中的变量一目了然。。

4. Tab补全

5. 上箭头寻找以前的命令



### 





