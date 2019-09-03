# QBUS6840: Tutorial 6   exponential smoothing(trend) 
![](/Users/apple/Downloads/2019s1/qbus6840/讲义/屏幕快照 2019-04-29 下午2.11.00.png)



## Simple exponential smoothing

###  图示

![](/Users/apple/Downloads/2019s1/qbus6840/讲义/WX20190430-140303@2x.png)
### 迭代公式：
$$
y_{\widehat {t+1|1:t}} = l_{t} = \alpha y_{t} + (1-\alpha)l_{t-1}
$$
当然 
$$
0 \le \alpha  \le 1
$$
展开下：
$$
\begin{aligned}
\begin{equation}


l_{1} =\alpha y_{1} + (1-\alpha) l_{0} \\
l_{2} =\alpha y_{2} + (1-\alpha) l_{1} = \alpha y_{2} + (1-\alpha)\alpha y_{1} + (1-\alpha)^{2} l_{0} \\
l_{3} =\alpha y_{3} + (1-\alpha) l_{2} =  \alpha y_{3} + (1-\alpha)\alpha y_{2}+ (1-\alpha)^{2}\alpha y_{1} + (1-\alpha)^{3}l_{0}


\end{equation}
\end{aligned}
$$
通项公式：
$$
l_{t} = \alpha y_{t} + (1 − \alpha )l_{t−1} = αyt + (1−α)αy_{t−1} +(1−α)^{2}αy_{t−2} +...+(1−α)^{t−1}αy_{1} +(1−α)^{t}l_{0}
$$
可以观察到：

1. 由于 $y_{i}$ 都是能得到的，决定预测好坏的就是 $\alpha$ 和 $l_{0}$ ；
2. $y_{\widehat {t+1|1:t}}$ 或者说  $l_{t}$ 是迭代生成的。
3. $y_{t}$ 的 t 其实是 从1开始的，但 python 里 list 的第一个位置的 index 是 0，即 $y_{t}$ 就是 python 里的 Y[t-1]，比如$y_{1}$ 在 python 里就是 Y[0]  

### 手动实现 Simple exponential smoothing

```python
# 伪代码 
#第一步确定 α 和 l_0, smoothed_manual / level 存的是 l_0 到 l_t, 同时也是 y_1 到 y_t+1 的预测值

alpha = 0.1
#smoothed_manual = [y[0]]
level = [l_0]

# 根据迭代公式生成新的 level 
for i in range( data_length - 1 ):
	level.append( alpha * Y[i] + (1 - alpha) * level[i] )

```

* 为什么 range 的范围是 data_length - 1:

  举例子说明，假如有1000 个数据( $y_{1}$ 到 $y_{1000}$)，data_length 为 1000。我们想根据这组数据做 Simple exponential smoothing, 得到1000个预测值( $\hat {y_{ 1 }}$ 到  $\hat y_{1000}$) ，但是也就是 $l_0 $  到 $l_{999}$ , 迭代生成 $l_{999}$ 的时候要用到 Y[998] 和 $l_{998}$ 。 所以 i 的范围应该是[0, 998], 也就是 range(999), 换句话说 range(data_length -1 )

  * 但这里的 $\hat y_{1}$ 是由 自己设置的$l_{0}$ 得到的， 是不可信的



### Pandas EWM  指数加权滑动 

如果要求滑动平均 EWMA 的话，先 ewm() 再 mean()

`DataFrame.ewm`(*alpha*, *adjust*)

* alpha 和手动的 alpha 一样

* adjust 

  ![](/Users/apple/Downloads/2019s1/qbus6840/讲义/WX20190427-160110@2x.png)

```python
smoothed = y.ewm(alpha=0.05, adjust=False).mean()
```



### 寻找最佳 alpha 

1. 首先实现 SSE 的公式

$$
SSE = \Sigma (y_{i} - \hat{y_{i}} )
$$

```python
def sse(x, y):
	return np.sum(np.power(x - y,2))
```

2. 遍历(0,1) 的所有 alpha 值，计算其对应的 SSE 的值。

```python
SSE_alphas = []
alphas = np.arange(0.01,1,0.01)

for i in alphas:
	smoothed = y.ewm(alpha = i, adjust=False).mean()
	SSE_alphas.append( sse(smoothed[:-1], y.values[1:]) )
```

* smoothed $\hat {y_{2}}$ 到 $\hat{y_{t+1}}$, smoothed[:-1] $\hat {y_{2}}$ 到 $\hat{y_{t}}$
* y.values 是 y 对应的一维 array,   y.values[1:] 指的是 $y_{2}$ 到 $y_{t}$

3. 用 np.argmin 函数找到 SSE 最小的 alpha 值

   np.argmin() 返回 array 最小值的 index

4. ```python
   optimal_alpha_one = alphas[ np.argmin(sse_one) ]
   ```



### 寻找最佳的 $l_{0}$

 Tutorial 的意思是寻找 $l_{0}$ 的过程和 $\alpha$  一样，就是 比较 SSE 的大小。

![](/Users/apple/Downloads/2019s1/qbus6840/讲义/WX20190428-011322@2x.png)



$l_{0}$ 一般有两种方式获得， 

Lec5 P12

![](/Users/apple/Downloads/2019s1/qbus6840/讲义/WX20190428-011153@2x.png)



* 后一种在 lec 6 里有说，用的是线性回归



## Holt’s linear method

### 图示

![](/Users/apple/Downloads/2019s1/qbus6840/讲义/WX20190430-140322@2x.png)

### 递推公式：

![](/Users/apple/Downloads/2019s1/qbus6840/讲义/WX20190428-010041@2x.png)

这里的 h 指的是 h-step-ahead forecast 的 h

可以观察到：

1. 需要调的参数是 $\alpha$ 和 $\beta$ 

2. $l_{0}$ 和 $b_{0}$ 对结果好坏至关重要



### 初始化参数

```python 
alpha = 0.1
beta = 0.1
l = [y[0]]
b = [y[1] - y[0]]

Y = y.tolist()
```

* 设定 $\alpha $ 和 $\beta$ 值，以及 $l_{0}$ 和 $b_{0}$ 

* Y 存的是 $y_{t}$ ， 也就是 $y_{1}$ $y_{2}$ ...

### Smoothing without forecasting

```python
holtsmoothed_manual = [] 
for i in range(len(y)):
    l.append(alpha * Y[i] + (1 - alpha) * (l[i] + b[i]))
    b.append(beta * (l[i+1] - l[i]) + (1 - beta) * b[i])
    holtsmoothed_manual.append(l[i+1])
```

* holtsforecast_manual 存的是 $\hat{y}_{t+0|t}$ = $l_{t}$  , t [1,len(y)]
* Y 存的是 $y_{t}$ ， 也就是 $y_{1}$ $y_{2}$ ...
* len(y) = 312 ,range(len(y)) = [0,311]

  * i = 0  $l_{1} = \alpha y_{0} + (1 - \alpha) l_{0} $
  * i = 1  $l_{2} = \alpha y_{1} + (1 - \alpha) l_{1} $
  * i = 2  $l_{3} = \alpha y_{2} + (1 - \alpha) l_{2} $
  * …...
  * i = 311  $l_{312} = \alpha y_{311} + (1 - \alpha) l_{311} $
* holtsmoothed_manual 存的是 [$l_{1}$ 到 $l_{312}$ ]


### 1-step forecasting(12 months)

```python
holtsforecast_manual = []

for i in range(len(y)+12):
    if i == len(Y):
        Y.append(l[-1] + b[-1])

    l.append(alpha * Y[i] + (1 - alpha) * (l[i] + b[i]))
    b.append(beta * (l[i+1] - l[i]) + (1 - beta) * b[i])

		holtsforecast_manual.append(l[i] + b[i])
```

* holtsforecast_manual 存的是 $\hat{y}_{t+1|t}$ = $l_{t}  + b_{t}$ , t [0,len(y) + 11]
* 第 7， 8， 9 行和之前的 smoothed 过程一样，因为 forecast 是在 smoothed 结果基础上进行的
* 第四行和第五行在Y 原来的312 个数据 smooth 完之后，将 $l_{t} + b_{t}$ 作为新的 $y_{t+1}$ 用来预测。
* range( len(y) + 12) 范围是[0，323]
* Y 存的是 $y_{t}$ ， 也就是 $y_{1}$ $y_{2}$ ...
* i = 0    $l_{1} = \alpha y_{1}  + (1-\alpha) (l_{0} + b_{0}) $   $ b_{1} = \beta (l_{1}-l_{0}) + (1-\beta)b_{0} $   $\hat{y}_{1} = l_{0} + b_{0} $
* i = 1    $l_{2} = \alpha y_{2}  + (1-\alpha) (l_{1} + b_{1}) $   $ b_{2} = \beta (l_{2}-l_{1}) + (1-\beta)b_{1} $   $\hat{y}_{2} = l_{1} + b_{1} $
* i = 2    $l_{3} = \alpha y_{3}  + (1-\alpha) (l_{2} + b_{2}) $   $ b_{3} = \beta (l_{3}-l_{2}) + (1-\beta)b_{2} $   $\hat{y}_{3} = l_{2} + b_{2} $
* ......
* i = 311    $l_{312} = \alpha y_{312}  + (1-\alpha) (l_{311} + b_{311}) $   $ b_{312} = \beta (l_{312}-l_{311}) + (1-\beta)b_{311} $   $\hat{y}_{312} = l_{311} + b_{311} $

**之后 **

**i == len(Y)  = 312 条件成立，Y[312] = $y_{313} = l_{312} + b_{312}$ ** 

* i = 312  $l_{313} = \alpha y_{313}  + (1-\alpha) (l_{312} + b_{312}) $   $ b_{313} = \beta (l_{313}-l_{312}) + (1-\beta)b_{312} $   $\hat{y}_{313} = l_{312} + b_{312} $
* ….
* i = 323  $l_{324} = \alpha y_{324}  + (1-\alpha) (l_{323} + b_{323}) $   $ b_{324} = \beta (l_{323}-l_{323}) + (1-\beta)b_{322} $   $\hat{y}_{324} = l_{323} + b_{323} $
* holtsforecast_manual 存的就是$\hat{y}_{1}$ …. $\hat{y}_{324}$ 



### 2-step forecasting(12 months)

```python
holtsforecast_manual2 = []

for i in range(len(y)+12):
    if i == len(Y):
        Y.append(l[-1] + 2*b[-1])

    l.append(alpha * Y[i] + (1 - alpha) * (l[i] + b[i]))
    b.append(beta * (l[i+1] - l[i]) + (1 - beta) * b[i])

		holtsforecast_manual.append(l[i] + 2*b[i])
```

* 和 1-step 的区别在于 $\hat{y}_{t+2|t} = l_{t} + 2b_{t}$

  

###n-step forecasting (m months)

```python
def holt(n,m):
	holtsforecast_manual2 = []

	for i in range(len(y)+n):
    	if i == len(Y):
        	Y.append(l[-1] + m*b[-1])

    	l.append(alpha * Y[i] + (1 - alpha) * (l[i] + b[i]))
    	b.append(beta * (l[i+1] - l[i]) + (1 - beta) * b[i])

			holtsforecast_manual.append(l[i] + m*b[i])
```

* 和 1-step 的区别在于 $\hat{y}_{t+n|t} = l_{t} + nb_{t}$






