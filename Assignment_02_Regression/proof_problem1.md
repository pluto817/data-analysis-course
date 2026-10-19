Ridge 回归在普通最小二乘法（OLS）的基础上增加 $L\_2$ 正则化项，其目标函数为：​
$$
L(\\beta)=(y-X\\beta)^T(y-X\\beta)+\\lambda\\beta^T\\beta
$$​
其中，$X$ 为设计矩阵，$y$ 为响应变量，$\\beta$ 为回归系数向量，$\\lambda>0$ 为正则化参数。​
对 $\\beta$ 求梯度，得到：​
$$
\\frac{\\partial L(\\beta)}{\\partial\\beta}
=-2X^Ty+2X^TX\\beta+2\\lambda\\beta
$$​
令梯度等于零：​
$$
-2X^Ty+2X^TX\\beta+2\\lambda\\beta=0
$$​
整理得：​
$$
(X^TX+\\lambda I)\\beta=X^Ty
$$​
因此，Ridge 回归的估计量为：​
$$
\\hat{\\beta}\_{\\mathrm{Ridge}}
*=(X^TX+\\lambda I)^{-1}X^Ty
$$​
在正交设计条件下，假设：​
$$
X^TX=I
$$​
代入上式：​
$$
\\begin{aligned}
\\hat{\\beta}\_{\\mathrm{Ridge}}*
\&=(I+\\lambda I)^{-1}X^Ty  
\&=((1+\\lambda)I)^{-1}X^Ty  
\&=\\frac{1}{1+\\lambda}X^Ty
\\end{aligned}
$$​
而普通最小二乘法的估计量为：​
$$
\\hat{\\beta}*{\\mathrm{OLS}}
=(X^TX)^{-1}X^Ty
$$​
由于 $X^TX=I$，因此：​
$$
\\hat{\\beta}*{\\mathrm{OLS}}=X^Ty
$$​
代入 Ridge 回归的表达式，得到：​
$$
\\hat{\\beta}{\\mathrm{Ridge}}
=\\frac{1}{1+\\lambda}\\hat{\\beta}{\\mathrm{OLS}}
$$​
故得证。​
结论：在正交设计条件 $X^TX=I$ 下，Ridge 回归系数等于 OLS 回归系数乘以 $\\frac{1}{1+\\lambda}$。由于 $\\lambda>0$，因此 $\\frac{1}{1+\\lambda}<1$，说明 Ridge 回归通过正则化对回归系数进行收缩。

