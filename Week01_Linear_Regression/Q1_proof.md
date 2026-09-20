# 第一题：代数证明

## 题目

证明在线性回归中，残差和满足

$$
\sum_{i=1}^{n}e_i=0
$$

且残差与自变量的乘积和满足

$$
\sum_{i=1}^{n}x_ie_i=0.
$$

## 1. 建立线性回归模型

设简单线性回归模型为：

$$
y_i=\beta_0+\beta_1x_i+e_i
$$

其中，$e_i$ 为第 $i$ 个样本的残差。

最小二乘法（OLS）的目标是使残差平方和（RSS）最小：

$$
RSS=\sum_{i=1}^{n}(y_i-\beta_0-\beta_1x_i)^2
$$

## 2. 证明残差和为 0

对 RSS 关于 $\beta_0$ 求偏导：

$$
\frac{\partial RSS}{\partial\beta_0}
=
-2\sum_{i=1}^{n}(y_i-\beta_0-\beta_1x_i)
$$

在最小二乘估计值 $\hat{\beta}_0,\hat{\beta}_1$ 处，偏导数应为 0：

$$
\sum_{i=1}^{n}
(y_i-\hat{\beta}_0-\hat{\beta}_1x_i)=0
$$

根据残差定义：

$$
e_i=y_i-\hat{\beta}_0-\hat{\beta}_1x_i
$$

因此：

$$
\boxed{\sum_{i=1}^{n}e_i=0}
$$

所以，在包含截距项的 OLS 线性回归中，所有残差之和等于 0。

## 3. 证明残差与自变量乘积和为 0

对 RSS 关于 $\beta_1$ 求偏导：

$$
\frac{\partial RSS}{\partial\beta_1}
=
-2\sum_{i=1}^{n}x_i
(y_i-\beta_0-\beta_1x_i)
$$

在最小二乘估计值处令偏导数等于 0：

$$
\sum_{i=1}^{n}x_i
(y_i-\hat{\beta}_0-\hat{\beta}_1x_i)=0
$$

根据残差定义：

$$
e_i=y_i-\hat{\beta}_0-\hat{\beta}_1x_i
$$

所以：

$$
\boxed{\sum_{i=1}^{n}x_ie_i=0}
$$

## 4. 结论

因此，OLS 回归在包含截距项时满足两个正交条件：

$$
\boxed{\sum_{i=1}^{n}e_i=0}
$$

以及

$$
\boxed{\sum_{i=1}^{n}x_ie_i=0}
$$

这意味着残差与截距项对应的常数向量、以及自变量 $x$ 都正交。
