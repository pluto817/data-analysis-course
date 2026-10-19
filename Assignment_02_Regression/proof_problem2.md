对于实对称矩阵 $A$，如果对任意非零向量 $v\\neq 0$，均有：​
$$
v^TAv>0
$$​
则称矩阵 $A$ 为严格正定矩阵。​
令：​
$$
A=X^TX+\\lambda I
$$​
其中，$X$ 为实矩阵，$I$ 为单位矩阵，且 $\\lambda>0$。​
任取非零向量 $v\\neq 0$，有：​
$$
\\begin{aligned}
v^TAv
\&=v^T(X^TX+\\lambda I)v  
\&=v^TX^TXv+\\lambda v^TIv  
\&=(Xv)^T(Xv)+\\lambda v^Tv  
\&=|Xv|\_2^2+\\lambda|v|\_2^2
\\end{aligned}
$$​
由于平方范数始终非负，因此：​
$$
|Xv|\_2^2\\geq 0
$$​
又因为 $v\\neq 0$，所以：​
$$
|v|\_2^2>0
$$​
结合 $\\lambda>0$，可得：​
$$
\\lambda|v|\_2^2>0
$$​
因此：​
$$
v^TAv
=|Xv|\_2^2+\\lambda|v|\_2^2>0
$$​
即对任意非零向量 $v$，均有：​
$$
v^T(X^TX+\\lambda I)v>0
$$​
同时，$X^TX+\\lambda I$ 是实对称矩阵，因为：​
$$
(X^TX+\\lambda I)^T=X^TX+\\lambda I
$$​
因此，根据严格正定矩阵的定义，得到：​
$$
\\boxed{X^TX+\\lambda I\\text{ 严格正定}}
$$​
故得证。​
结论
对于任意实矩阵 $X$ 和任意 $\\lambda>0$，矩阵 $X^TX+\\lambda I$ 都是严格正定矩阵。因此，该矩阵一定可逆，即使 $X^TX$ 本身是奇异矩阵，也不影响这一结论。这也是 Ridge 回归能够获得唯一系数解的重要原因。

