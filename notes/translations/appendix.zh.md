## 附录 A. 概率论复习

我们给出概率论中若干基本概念的简要综述。本节部分取材自 Lipman et al. (2024)。

### A.1 随机向量

考虑 $d$ 维欧氏空间（Euclidean space）中的数据 $x=(x^1,\ldots,x^d)\in\mathbb{R}^d$，其上配备标准欧氏内积（inner product）$\left\langle x,y\right\rangle=\sum_{i=1}^d x^i y^i$ 与范数（norm）$\left\Vert x\right\Vert=\sqrt{\left\langle x,x\right\rangle}$。我们将考虑取值于 $\mathbb{R}^d$ 的随机变量（random variable, RV）$X$，其具有连续的概率密度函数（probability density function, PDF）。该密度定义为一个连续函数 $p_X:\mathbb{R}^d\to\mathbb{R}_{\geq 0}$，它赋予事件（event）$A$ 的概率为

$$
\mathbb{P}(X\in A)=\int_A p_X(x)\,\mathrm{d}x,
\qquad (96)
$$

其中 $\int p_X(x)\,\mathrm{d}x=1$。按照约定，当积分区域为全空间时，我们省略积分区间（$\int\equiv\int_{\mathbb{R}^d}$）。为使记号简洁，我们把随机变量 $X_t$ 的概率密度 $p_{X_t}$ 简记为 $p_t$。我们用记号 $X\sim p$ 或 $X\sim p(X)$ 表示 $X$ 服从分布 $p$。生成建模中一种常见的概率密度是 $d$ 维各向同性高斯（isotropic Gaussian）：

$$
\mathcal{N}(x;\mu,\sigma^2 I)=(2\pi\sigma^2)^{-\frac{d}{2}}\exp\left(-\frac{\left\Vert x-\mu\right\Vert_2^2}{2\sigma^2}\right),
\qquad (97)
$$

其中 $\mu\in\mathbb{R}^d$ 与 $\sigma\in\mathbb{R}_{>0}$ 分别表示该分布的均值（mean）与标准差（standard deviation）。

随机变量的期望（expectation）是在最小二乘（least-squares）意义下最接近 $X$ 的常向量：

$$
\mathbb{E}[X]=\mathrm{arg\,min}_{z\in\mathbb{R}^d}\int\left\Vert x-z\right\Vert^2 p_X(x)\,\mathrm{d}x=\int x\,p_X(x)\,\mathrm{d}x.
\qquad (98)
$$

计算随机变量之函数的期望时，一个有用的工具是不自觉统计员定律（law of the unconscious statistician）：

$$
\mathbb{E}[f(X)]=\int f(x)\,p_X(x)\,\mathrm{d}x.
\qquad (99)
$$

必要时，我们把期望所针对的随机变量标出，写作 $\mathbb{E}_X f(X)$。

### A.2 条件密度与期望

给定两个随机变量 $X,Y\in\mathbb{R}^d$，它们的联合概率密度函数（joint PDF）$p_{X,Y}(x,y)$ 具有边缘分布（marginal）

$$
\int p_{X,Y}(x,y)\,\mathrm{d}y=p_X(x)\quad\quad\int p_{X,Y}(x,y)\,\mathrm{d}x=p_Y(y).
\qquad (100)
$$

见图 21，它示意了两个取值于 $\mathbb{R}$ 的随机变量（$d=1$）的联合概率密度。

**图注.** 联合概率密度 $p_{X,Y}$（以阴影表示）及其边缘分布 $p_X$ 与 $p_Y$（以黑线表示）。图取自 Lipman et al. (2024)。

条件概率密度（conditional PDF）$p_{X|Y}$ 描述的是：在密度满足 $p_Y(y)>0$ 的事件 $Y=y$ 之下，随机变量 $X$ 的概率密度，

$$
p_{X|Y}(x|y):=\frac{p_{X,Y}(x,y)}{p_Y(y)},
\qquad (101)
$$

对条件概率密度 $p_{Y|X}$ 亦然。贝叶斯法则（Bayes' rule）用 $p_{X|Y}$ 表示条件概率密度 $p_{Y|X}$：

$$
p_{Y|X}(y|x)=\frac{p_{X|Y}(x|y)\,p_Y(y)}{p_X(x)},
\qquad (102)
$$

其中 $p_X(x)>0$。

条件期望（conditional expectation）$\mathbb{E}[X|Y]$ 是在最小二乘意义下对 $X$ 的最佳逼近函数 $g_\star(Y)$：

$$
\begin{aligned}
g_\star&:=\mathrm{arg\,min}_{g:\mathbb{R}^d\to\mathbb{R}^d}\mathbb{E}\left[\left\Vert X-g(Y)\right\Vert^2\right] \\
&=\mathrm{arg\,min}_{g:\mathbb{R}^d\to\mathbb{R}^d}\int\left\Vert x-g(y)\right\Vert^2 p_{X,Y}(x,y)\,\mathrm{d}x\,\mathrm{d}y \\
&=\mathrm{arg\,min}_{g:\mathbb{R}^d\to\mathbb{R}^d}\int\left[\int\left\Vert x-g(y)\right\Vert^2 p_{X|Y}(x|y)\,\mathrm{d}x\right]p_Y(y)\,\mathrm{d}y.
\end{aligned}
\qquad (103)
$$

因此，对满足 $p_Y(y)>0$ 的 $y\in\mathbb{R}^d$，条件期望函数为

$$
\mathbb{E}[X|Y=y]:= g_\star(y)=\int x\,p_{X|Y}(x|y)\,\mathrm{d}x,
\qquad (104)
$$

其中第二个等号来自：对 $Y=y$，取式 (103) 内层方括号中的最小值点，其做法与式 (98) 相同。把 $g_\star$ 与随机变量 $Y$ 复合，得到

$$
\mathbb{E}[X|Y]:= g_\star(Y),
\qquad (105)
$$

这是一个取值于 $\mathbb{R}^d$ 的随机变量。颇为令人困惑的是，$\mathbb{E}[X|Y=y]$ 与 $\mathbb{E}[X|Y]$ 都常常被称为条件期望，但它们是不同的对象。具体而言，$\mathbb{E}[X|Y=y]$ 是一个函数 $\mathbb{R}^d\to\mathbb{R}^d$，而 $\mathbb{E}[X|Y]$ 是一个取值于 $\mathbb{R}^d$ 的随机变量。为区分这两个术语，后文讨论将采用此处引入的记号。

塔性质（tower property）是一个有用的性质，它有助于简化涉及两个随机变量 $X$ 与 $Y$ 的条件期望推导：

$$
\mathbb{E}\left[\mathbb{E}[X|Y]\right]=\mathbb{E}[X].
\qquad (106)
$$

因为 $\mathbb{E}[X|Y]$ 本身是随机变量，而且是随机变量 $Y$ 的函数，外层期望计算的就是 $\mathbb{E}[X|Y]$ 的期望。塔性质可以用上面的若干定义加以验证：

$$
\begin{aligned}
\mathbb{E}\left[\mathbb{E}[X|Y]\right]&=\int\left(\int x\,p_{X|Y}(x|y)\,\mathrm{d}x\right)p_Y(y)\,\mathrm{d}y \\
&=\int\int x\,p_{X,Y}(x,y)\,\mathrm{d}x\,\mathrm{d}y \\
&=\int x\,p_X(x)\,\mathrm{d}x=\mathbb{E}[X].
\end{aligned}
$$

最后，考虑一个涉及两个随机变量 $f(X,Y)$ 与 $Y$ 的有用性质，其中 $X$ 与 $Y$ 是任意的两个随机变量。于是，把不自觉统计员定律与式 (104) 合用，得到恒等式

$$
\mathbb{E}[f(X,Y)|Y=y]=\int f(x,y)\,p_{X|Y}(x|y)\,\mathrm{d}x.
\qquad (107)
$$

## 附录 B. 福克–普朗克方程（Fokker–Planck Equation）的一个证明

本节给出福克–普朗克方程的一个自洽证明，连续性方程（continuity equation）是其特例（定理 11）。我们强调：**要理解本文其余部分，并不必须阅读本节**，而且本节在数学上更为深入。若你希望弄清福克–普朗克方程从何而来，那么本节便是为你而写的。

**定理 41（福克–普朗克方程）.** 设 $p_t$ 是一条概率路径（probability path），且 $p_0=p_{\mathrm{init}}$。考虑随机微分方程（stochastic differential equation, SDE）

$$
X_0\sim p_{\mathrm{init}},\quad\mathrm{d}X_t=u_t(X_t)\,\mathrm{d}t+\sigma_t\,\mathrm{d}W_t.
$$

则对一切 $0\leq t\leq 1$，$X_t$ 的分布为 $p_t$，当且仅当如下福克–普朗克方程成立：

$$
\partial_t p_t(x)=-\mathrm{div}(p_t u_t)(x)+\frac{\sigma_t^2}{2}\Delta p_t(x)\quad x\in\mathbb{R}^d,\; 0\leq t\leq 1,
\qquad (108)
$$

我们首先说明福克–普朗克方程是必要条件，也就是说，若 $X_t\sim p_t$，则福克–普朗克方程成立。证明的技巧是使用检验函数（test function）$f$，即函数 $f:\mathbb{R}^d\to\mathbb{R}$，它无穷次可微（“光滑”（smooth）），并且只在某个有界区域上非零（紧支集（compact support））。我们用到如下事实：对任意可积函数 $g_1,g_2:\mathbb{R}^d\to\mathbb{R}$，

$$
g_1(x)=g_2(x)\ \ x\in\mathbb{R}^d\quad\Leftrightarrow\quad\int f(x)\,g_1(x)\,\mathrm{d}x=\int f(x)\,g_2(x)\,\mathrm{d}x\ \ f,
\qquad (109)
$$

换言之，逐点相等可以表达为积分相等。检验函数的用处在于它们是光滑的，因而可以取梯度以及更高阶的导数。特别地，对任意检验函数 $f_1,f_2$，可以使用分部积分（integration by parts）：

$$
\int f_1(x)\frac{\partial}{\partial x_i}f_2(x)\,\mathrm{d}x=-\int f_2(x)\frac{\partial}{\partial x_i}f_1(x)\,\mathrm{d}x,
\qquad (110)
$$

条件是 $f_1$、$f_2$ 以及它们的乘积 $f_1\cdot f_2$ 可积。把上式与散度（divergence）和拉普拉斯算子（Laplacian）的定义（见式 (22)）合用，得到恒等式

$$
\int\nabla f_1^{\mathsf{T}}(x)\,f_2(x)\,\mathrm{d}x=-\int f_1(x)\,\mathrm{div}(f_2)(x)\,\mathrm{d}x\quad(f_1:\mathbb{R}^d\to\mathbb{R},\; f_2:\mathbb{R}^d\to\mathbb{R}^d),
\qquad (111)
$$

$$
\int f_1(x)\,\Delta f_2(x)\,\mathrm{d}x=\int f_2(x)\,\Delta f_1(x)\,\mathrm{d}x\quad(f_1:\mathbb{R}^d\to\mathbb{R},\; f_2:\mathbb{R}^d\to\mathbb{R}),
\qquad (112)
$$

现在进入证明。我们采用式 (6) 中 SDE 轨迹的随机更新：

$$
X_{t+h}=X_t+h u_t(X_t)+\sigma_t(W_{t+h}-W_t)+h R_t(h),
\qquad (113)
$$

$$
\approx X_t+h u_t(X_t)+\sigma_t(W_{t+h}-W_t),
\qquad (114)
$$

眼下为了可读性，我们先忽略误差项 $R_t(h)$，因为反正要取 $h\to 0$。于是可以作如下计算：

$$
\begin{aligned}
&f(X_{t+h})-f(X_t) \\
&=f\big(X_t+h u_t(X_t)+\sigma_t(W_{t+h}-W_t)\big)-f(X_t) \\
&=\nabla f(X_t)^{\mathsf{T}}\big(h u_t(X_t)+\sigma_t(W_{t+h}-W_t)\big) \\
&\quad+\frac{1}{2}\big(h u_t(X_t)+\sigma_t(W_{t+h}-W_t)\big)^{\mathsf{T}}\nabla^2 f(X_t)\big(h u_t(X_t)+\sigma_t(W_{t+h}-W_t)\big) \\
&=h\nabla f(X_t)^{\mathsf{T}}u_t(X_t)+\sigma_t\nabla f(X_t)^{\mathsf{T}}(W_{t+h}-W_t) \\
&\quad+\frac{1}{2}h^2 u_t(X_t)^{\mathsf{T}}\nabla^2 f(X_t)\,u_t(X_t)+h\sigma_t u_t(X_t)^{\mathsf{T}}\nabla^2 f(X_t)(W_{t+h}-W_t) \\
&\quad+\frac{1}{2}\sigma_t^2(W_{t+h}-W_t)^{\mathsf{T}}\nabla^2 f(X_t)(W_{t+h}-W_t),
\end{aligned}
$$

其中（i）对 $f$ 在 $X_t$ 处作了二阶泰勒近似（Taylor approximation），（ii）用到了海森矩阵（Hessian）$\nabla^2 f$ 是对称矩阵这一事实。注意 $\mathbb{E}[W_{t+h}-W_t|X_t]=0$，且 $W_{t+h}-W_t|X_t\sim\mathcal{N}(0,h I_d)$。因此

$$
\begin{aligned}
&\mathbb{E}[f(X_{t+h})-f(X_t)|X_t] \\
&=h\nabla f(X_t)^{\mathsf{T}}u_t(X_t)+\frac{1}{2}h^2 u_t(X_t)^{\mathsf{T}}\nabla^2 f(X_t)\,u_t(X_t)+\frac{h}{2}\sigma_t^2\,\mathbb{E}_{\epsilon_t\sim\mathcal{N}(0,I_d)}\!\left[\epsilon_t^{\mathsf{T}}\nabla^2 f(X_t)\,\epsilon_t\right] \\
&=h\nabla f(X_t)^{\mathsf{T}}u_t(X_t)+\frac{1}{2}h^2 u_t(X_t)^{\mathsf{T}}\nabla^2 f(X_t)\,u_t(X_t)+\frac{h}{2}\sigma_t^2\mathrm{trace}\!\left(\nabla^2 f(X_t)\right) \\
&=h\nabla f(X_t)^{\mathsf{T}}u_t(X_t)+\frac{1}{2}h^2 u_t(X_t)^{\mathsf{T}}\nabla^2 f(X_t)\,u_t(X_t)+\frac{h}{2}\sigma_t^2\Delta f(X_t),
\end{aligned}
$$

其中（i）用到 $\mathbb{E}_{\epsilon_t\sim\mathcal{N}(0,I_d)}[\epsilon_t^{\mathsf{T}} A\epsilon_t]=\mathrm{trace}(A)$，（ii）用到拉普拉斯算子与海森矩阵的定义。由此得到

$$
\begin{aligned}
&\partial_t\mathbb{E}[f(X_t)] \\
&=\lim_{h\to 0}\frac{1}{h}\mathbb{E}[f(X_{t+h})-f(X_t)] \\
&=\lim_{h\to 0}\frac{1}{h}\mathbb{E}\!\left[\mathbb{E}[f(X_{t+h})-f(X_t)|X_t]\right] \\
&=\mathbb{E}\!\left[\lim_{h\to 0}\frac{1}{h}\left(h\nabla f(X_t)^{\mathsf{T}}u_t(X_t)+\frac{1}{2}h^2 u_t(X_t)^{\mathsf{T}}\nabla^2 f(X_t)\,u_t(X_t)+\frac{h}{2}\sigma_t^2\Delta f(X_t)\right)\right] \\
&=\mathbb{E}\!\left[\nabla f(X_t)^{\mathsf{T}}u_t(X_t)+\frac{1}{2}\sigma_t^2\Delta f(X_t)\right] \\
&=\int\nabla f(x)^{\mathsf{T}}u_t(x)\,p_t(x)\,\mathrm{d}x+\int\frac{1}{2}\sigma_t^2\Delta f(x)\,p_t(x)\,\mathrm{d}x \\
&=-\int f(x)\,\mathrm{div}(u_t p_t)(x)\,\mathrm{d}x+\int\frac{1}{2}\sigma_t^2 f(x)\,\Delta p_t(x)\,\mathrm{d}x \\
&=\int f(x)\left(-\mathrm{div}(u_t p_t)(x)+\frac{1}{2}\sigma_t^2\Delta p_t(x)\right)\mathrm{d}x,
\end{aligned}
$$

其中（i）用到 $p_t$ 是 $X_t$ 的分布这一假设，（ii）用到式 (111) 与式 (112)。注意，要用上这些恒等式，需要乘积 $p_t(x)u_t(x)$ 可积，即

$$
\int p_t(x)\,\left\Vert u_t(x)\right\Vert\,\mathrm{d}x<\infty.
$$

注意，这一条件在机器学习中几乎总是成立的（由于数值精度的限制，数据与函数都是有界的）。因此有

$$
\partial_t\mathbb{E}[f(X_t)]=\int f(x)\left(-\mathrm{div}(p_t u_t)(x)+\frac{\sigma_t^2}{2}\Delta p_t(x)\right)\mathrm{d}x
\quad f,\;0\leq t\leq 1,
\qquad (115)
$$

$$
\Leftrightarrow\quad\partial_t\int f(x)\,p_t(x)\,\mathrm{d}x=\int f(x)\left(-\mathrm{div}(p_t u_t)(x)+\frac{\sigma_t^2}{2}\Delta p_t(x)\right)\mathrm{d}x
\quad f,\;0\leq t\leq 1,
\qquad (116)
$$

$$
\Leftrightarrow\quad\int f(x)\,\partial_t p_t(x)\,\mathrm{d}x=\int f(x)\left(-\mathrm{div}(p_t u_t)(x)+\frac{\sigma_t^2}{2}\Delta p_t(x)\right)\mathrm{d}x
\quad f,\;0\leq t\leq 1,
\qquad (117)
$$

$$
\Leftrightarrow\quad\partial_t p_t(x)=-\mathrm{div}(p_t u_t)(x)+\frac{\sigma_t^2}{2}\Delta p_t(x)
\quad x\in\mathbb{R}^d,\;0\leq t\leq 1,
\qquad (118)
$$

其中（i）用到假设 $X_t\sim p_t$，（ii）交换了导数与积分的次序，（iii）用到式 (109)。这就完成了福克–普朗克方程是必要条件的证明。

最后，我们说明它也是充分条件。福克–普朗克方程是一个偏微分方程（partial differential equation, PDE）。更具体地说，它是所谓的抛物型偏微分方程（parabolic partial differential equation）。与定理 3 类似，在给定的初始条件（initial condition）下，这类微分方程具有唯一解（例如见 Evans (2022) 第 7 章）。现在，若式 (108) 对 $p_t$ 成立，我们刚刚已经证明它也必须对 $X_t$ 的真实分布 $q_t$ 成立（即 $X_t\sim q_t$）。换言之，$p_t$ 与 $q_t$ 都是这条抛物型偏微分方程的解。此外，我们知道初始条件相同，即由插值概率路径（interpolating probability path）的构造有 $p_0=q_0=p_{\mathrm{init}}$。因此，由微分方程解的唯一性，对一切 $0\leq t\leq 1$ 有 $p_t=q_t$，也就是 $X_t\sim q_t=p_t$，这正是我们要证明的。
