# Qbus 6840 TUT 4  Time Series Decomposition

第四周在第三周的基础上继续分解 Time Series， 分解 seasonality

## Additive Vs Multiplicative

当时间数列图显示的时间数列的季节变动大致相等时，或时间数列图随时间推移等宽推进时，采用**加法模型**

当时间数列图显示的时间数列的季节变动与时间数列的长期趋势大致成正比是，采用**乘法模型**

乘法模型的 TS 图大致呈喇叭状或发射状

![Additive 和 Multiplicative的比较](/Users/apple/Downloads/2019s1/qbus6840/讲义/timeseries-forecasting-profiles-resized-600.jpg)

如果数据中的模式不很明显，并且在加法模型和乘法模型之间进行选择还有困难，则可以尝试两种模型，然后选择准确度度量较小的模型。

Additive model 和 Multiplicative model 都是用来解释 TS 的组成部分

$$
y_{t} = f(T_{t},S_{t},C_{t},e_{t})
$$

* $T_{t}$ : Trend component
* $S_{t}$ : Seasonal component
* C_{t}: Cycle component
* e_{t} : Residual component

Additive model 的分解公式：
$$
y_{t} = T_{t}+S_{t}+C_{t}+e_{t}
$$
Multiplicative model 的分解公式：
$$
y_{t} = T_{t}*S_{t}*C_{t}*e_{t}
$$

## 一些化简处理

1. Tutorial 3 中用 rolling 函数计算的 Moving Average($Y_{t}$) 

>For most cases, we will treat the moving
>average results as the combination of trend and cycle. 

$$
T_{t} + C_{t} = Moving Average(Y_{t})
$$

 Multiplicative model 对应的：
$$
T_{t} * C_{t} = Moving Average(Y_{t})
$$


2. 省略 cycle component

>  In most cases, cycle 𝐶" is hard to model, we will not take this
> component into account.

$$
T_{t} = T_{t} + C_{t}
$$

Multiplicative model 对应的：
$$
T_{t} * C_{t} =T_{t}
$$

## 计算 seasonal index

### 数学推导

由于 Tutorial 3 最后

```python
ts_res = ts_log - Trend
```

得到的 `ts_res`代表的是 Seasonal Component $𝑆_{t} + 𝑒_{t}​$

根据 lecture3 Page 19 的说法，由于 season components 是 stationary 的 (Tutorial 3 最后的Dickey-fuller Test 测试) 

![](/Users/apple/Downloads/2019s1/qbus6840/讲义/2.png)

根据 lecture3 P26 的计算过程

![](/Users/apple/Downloads/2019s1/qbus6840/讲义/3.png)

lecture 里遇到的是 Multiplicative model 的数据，


$$
\bar{s_{t}}= mean(s_{t})\\
c = \frac{1}{mean(\bar{s_{t}})}\\
\bar{s_{m}} = \bar{s_{t}} * c = \frac{\bar{s_{t}}}{mean(\bar{s_{t}})}
$$
类比成 Additive model 的话
$$
\bar{s_{t}}= mean(s_{t}) \\
\bar{s_{m}} = \bar{s_{t}}-mean(\bar{s_{t}})
$$
$s_{t}$ 指的是每个月的数据，t = [1,12]

计算 seasonal index 需要我们先求 每个月 多年数据的平均值，再减去每个月平均值的平均值。

### 代码实现

1. 由于`ts_res` 中存在 NaN 的值，在使用 mean() 函数时，数据里是不能有 NaN 值。我们需要进行数据的填充，一种方式是 Tutorial 中的使用 [nan_to_num函数](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.nan_to_num.html) 用 0 填充NaN 数据。在该情景下，使用平均值用来填充更为合适

```python
# Tutorial  
ts_res_zero = np.nan_to_num(ts_res)
  
# 改进
ts_res_zero = ts_res.fillna(ts_res.mean())
```

2. 用 [reshape函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html)将数据整形成 12 * 12 的方阵，每一行代表每一年12个月每个月的数据($S_{t}​$)

numpy.reshape(*a***,** *newshape*)

* a 被整形的 array 或 matrix
* newshape： 整形后的 size, 如(2，3) 二行三列

```python
monthly_S = np.reshape(ts_res_zero, (12,12))
```

3. 用 [mean函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html) 求每个月平均值

> 因为 第0年和第11年的数据是用 0 填充的 NaN 值，在计算平均值时不计算，所以选择[1:11,:] 第一行到第11行的所有列

numpy.mean(*a***,** *axis=None*)

* a 要被求平均数的数据
* axis : 0 垂直方向求平均值(每列求平均)， 1 水平方向求平均值(每行求平均)

```python
monthly_avg = np.mean(monthly_S[1:11,:], axis=0)
```

4.减去每个月平均值的平均值

```python
mean_allmonth = monthly_avg.mean()
monthly_avg_normalized = monthly_avg - mean_allmonth
print(monthly_avg_normalized.mean())
```

## 计算 seasonal adjusted data

### 数学推导

类比于 Page27 的 Multiplicative model：
$$
\widehat{ {T_{t} \times C_{t}} \times e_{t} } = \frac{Y_{t}}{\hat{S_{t}}} = \frac{Y_{t}}{\bar{S_{m}}}
$$
Additive model:
$$
\widehat{ {T_{t} + C_{t}} + e_{t} } = Y_{t} - \hat{S_{t}} = Y_{t} - \bar{S_{m}}
$$

### 代码实现

1. 用 [`numpy.tile`函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html?highlight=tile#numpy.tile) 复制 $\bar{S_{m}}$ (1年) 成 12 年的 matrix

```python
tiled_avg = np.tile(monthly_avg_normalized, 12)
```

`numpy.tile(A, reps)`

* A  被复制的 array 
* reps: 复制的格式，e.g.  2  - 复制遍， (2，3)复制成 2*3 个 A 组成的矩阵

2. 用 tutorial 3 里被 log 处理过的原数据（$Y_{t}$）减去 这个矩阵

```python
seasonally_adjusted = ts_log - tiled_avg
```

## 更新 trend-cycle

在原数据中减去了 seasonality 的 Seasonally Adjusted Series （$\widehat{ {T_{t} + C_{t}} + e_{t} }$）我们需要再次进行 estimate 

re-estimate 的方式：

* Tutorial4 里使用的 Rolling(也就是求 moving average) 
* Lecture 3/4 中的 linear regression

```python
T_final = seasonally_adjusted.rolling(12, center =True).mean().rolling(2, center = True).mean()
```

## 补充：直接用函数分解 $y_{t}$

在 tutorial 4 最后提供的网页中提供了直接调用库函数分解 y_(t) 的办法[seasonal_decompose函数](https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html)

statsmodels.tsa.seasonal.seasonal_decompose(x, model)

* x 待分解的 TS
* model :  **"additive"** 或者 **"multiplicative"**
* 返回值  seasonal, trend, and resid attributes

```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label = "Oringinal")
plt.legend(loc = 'best')
plt.subplot(412)
plt.plot(ts_log, label = "Trend")
plt.legend(loc = 'best')
plt.subplot(413)
plt.plot(ts_log, label = "Seasonality")
plt.legend(loc = 'best')
plt.subplot(414)
plt.plot(ts_log, label = "Residuals")
plt.legend(loc = 'best')
```

结果：

![](/Users/apple/Downloads/2019s1/qbus6840/讲义/4.png)



从时间序列中删除 Trend 和 Seasonality ，得到 Residuals



```

```





