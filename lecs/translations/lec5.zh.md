# 讲座 5：离散扩散模型与离散流匹配

## 第 1 页

**讲座 05：离散扩散模型与离散流匹配**

MIT IAP 2026｜2025 年 1 月 30 日

Peter Holderrieth 与 Ron Shprints

赞助人：Tommi Jaakkola

## 第 2 页

### 课程概览

- 讲座 1：流模型与扩散模型。
- 讲座 2：流匹配——训练算法。
- 讲座 3：分数匹配与引导——如何以提示为条件。
- 讲座 4：构建图像生成器——潜空间与网络架构。
- 讲座 5：离散扩散模型与离散流匹配。

## 第 3 页

### 新闻中的离散扩散模型

本页介绍离散扩散模型在近期研究中的出现。

图示：Subham Sahoo。

## 第 4 页

### 扩散大语言模型可以按任意顺序生成文本

参考：Austin 等，《Structured Denoising Diffusion Models in Discrete State-Spaces》，NeurIPS 2022。

图示：Subham Sahoo。

## 第 5 页

### 离散扩散模型与离散流匹配

- 面向离散序列数据的模型：语言、蛋白质序列等。
- 注意：离散空间中不存在扩散/SDE，也不存在流/ODE。
- 但是，流匹配和去噪的学习原理可以推广到离散数据。
- 数学模型：连续时间马尔可夫链（continuous-time Markov chain，CTMC）。
- 今天的内容：
  - CTMC 模型。
  - 离散流匹配：离散概率路径、离散边缘化技巧、离散流匹配目标。

## 第 6 页

### 连续时间马尔可夫链

连续时间马尔可夫链（CTMC）是在离散状态空间中描述随机跳转的模型。

图示：Andrew Campbell。

## 第 7 页

### 连续时间马尔可夫链示例

状态空间为

$$
S=\{a,b\},
$$

速率矩阵为

$$
Q=\begin{pmatrix}
-\lambda&\lambda\\
\lambda&-\lambda
\end{pmatrix}.
$$

通过对演化方程求导，可以得到转移概率矩阵

$$
\begin{pmatrix}
\mathbb{P}(X_{t+h}=a\mid X_t=a)&\mathbb{P}(X_{t+h}=a\mid X_t=b)\\
\mathbb{P}(X_{t+h}=b\mid X_t=a)&\mathbb{P}(X_{t+h}=b\mid X_t=b)
\end{pmatrix}
=\frac12\begin{pmatrix}
1+e^{-2\lambda h}&1-e^{-2\lambda h}\\
1-e^{-2\lambda h}&1+e^{-2\lambda h}
\end{pmatrix}.
$$

当 \(h\to\infty\) 时，

$$
P(h)\longrightarrow
\begin{pmatrix}
\frac12&\frac12\\
\frac12&\frac12
\end{pmatrix}.
$$

## 第 8 页

### 邻居示例

对于序列状态，若两个状态只在一个位置上不同，就称它们互为邻居：

- \(x\) 与 \(y\) 是邻居。
- \(y\) 与 \(z\) 是邻居。
- \(z\) 与 \(x\) 不是邻居，因为它们在多个位置上不同。

## 第 9 页

### 一般 CTMC 与因子化 CTMC

一般 CTMC 的状态空间可能非常大。因子化 CTMC 只允许一次改变一个词元，因此只对邻居状态赋予非零跳转速率。

若 \(x,y\in\mathcal{V}^{d}\)，则因子化条件可以写为

$$
Q_t^\theta(y\mid x)=0
\quad\text{当 }y\text{ 与 }x\text{ 在多于一个位置上不同}.
$$

这是一种稀疏性约束。

## 第 10 页

### 从因子化 CTMC 模型采样

因子化模型把输出表示为每个位置、每个词表符号的跳转速率，输出形状为

$$
d\times V,
$$

其中 \(d\) 是序列长度，\(V\) 是词表大小。

在小步长 \(h\) 内，可以对每个词元并行执行 Euler 更新：

$$
\tilde p_{j,t}(v\mid x)=
\begin{cases}
hq_j(v),&v\neq x,\\
1-h\displaystyle\sum_{v'\in\mathcal{V}\setminus\{x\}}q_j(v'),&v=x.
\end{cases}
$$

## 第 11 页

### 采样过程示意

从初始分布抽取一个完全加噪的离散序列，然后根据因子化 CTMC 的跳转速率逐步更新词元。

参考：Austin 等，《Structured Denoising Diffusion Models in Discrete State-Spaces》，NeurIPS 2022。

图示：Subham Sahoo。

## 第 12 页

### 用 CTMC 进行生成建模

数据分布可以是互联网上文本的分布，初始分布可以是均匀分布等简单分布。

目标是用 CTMC 把“噪声”转换为数据：

$$
X_0\sim p_{\mathrm{init}},\qquad
X_t\text{ 为 }Q_t\text{ 的 CTMC}
\quad\Longrightarrow\quad
X_1\sim p_{\mathrm{data}}.
$$

## 第 13 页

### 连续空间中的流匹配

连续流匹配的三列对象是：

| 条件概率路径 | 条件向量场 | 条件流匹配损失 |
|---|---|---|
| \(p_t(x\mid z)\) | \(u_t^z(x)\) | \(\lVert u_t^\theta(x)-u_t^z(x)\rVert^2\) |
| 边缘概率路径 \(p_t(x)\) | 边缘向量场 \(u_t(x)\) | 边缘流匹配损失 |

## 第 14 页

### 离散流匹配矩阵

在离散空间中，把连续空间中的向量场替换为速率矩阵：

| 条件概率路径 | 条件速率矩阵 | 离散流匹配损失 |
|---|---|---|
| \(p_t(x\mid z)\) | \(Q_t^z(y\mid x)\) | 分类损失 |
| 边缘概率路径 \(p_t(x)\) | \(Q_t(y\mid x)\) | 离散流匹配损失 |

## 第 15 页

### 因子化混合路径示例

设调度器满足

$$
0\leq\kappa_t\leq1,\qquad \kappa_0=0,\qquad \kappa_1=1.
$$

每个词元独立地以概率 \(1-\kappa_t\) 被噪声破坏。条件概率路径为

$$
p_t(x\mid z)=\prod_{j=1}^{d}
\left[(1-\kappa_t)p_{\mathrm{init}}^{(j)}(x_j)
+\kappa_t\delta_{z_j}(x_j)\right].
$$

等价的采样过程为

$$
\begin{aligned}
m_j&\sim\mathrm{Bernoulli}(\kappa_t),
&\xi_j&\sim p_{\mathrm{init}}^{(j)},\\
x_j&=m_jz_j+(1-m_j)\xi_j,
&j&=1,\ldots,d,\\
x&=(x_1,\ldots,x_d).
\end{aligned}
$$

## 第 16 页

### 玩具示例

在离散状态空间中，概率质量可以从一个离散状态转移到另一个离散状态。

图示：Yaron Lipman。

## 第 17 页

### 离散概率路径

离散概率路径在初始分布与终点数据分布之间插值。对固定终点 \(z\)，条件路径从 \(p_{\mathrm{init}}\) 逐渐变为集中在 \(z\) 上的 Dirac 分布。

## 第 18 页

### 与连续空间的区别

概率质量在离散空间中是“瞬移”的：它不是沿连续空间中的方向移动，而是降低一个分布的权重、提高另一个分布的权重。

## 第 19 页

### 离散概率路径

边缘概率路径由条件路径对数据分布求平均得到：

$$
p_t(x)=\sum_{z\in S}p_t(x\mid z)p_{\mathrm{data}}(z).
$$

它满足

$$
p_0=p_{\mathrm{init}},\qquad p_1=p_{\mathrm{data}}.
$$

## 第 20 页

### Kolmogorov 前向方程

Kolmogorov 前向方程（KFE）是连续性方程在离散空间中的对应物。

CTMC 的速率矩阵为 \(Q_t\) 时，CTMC 遵循概率路径 \(p_t\) 当且仅当

$$
X_t\sim p_t\quad(0\leq t\leq1)
$$

并且满足

$$
\frac{\mathrm{d}}{\mathrm{d}t}p_t(x)
=\sum_{y\in S}Q_t(x\mid y)p_t(y).
$$

左侧是概率的变化率，右侧是所有状态流入 \(x\) 的净流量。

## 第 21 页

### 因子化混合路径的条件速率矩阵

条件速率矩阵同样是因子化的，只允许一次更新一个词元：

$$
Q_t^z(y\mid x)=\bigl(Q_t^z(v_i,j\mid x_j)\bigr)_{v_i,j},
$$

$$
Q_t^z(v_i,j\mid x_j)
=\frac{\dot\kappa_t}{1-\kappa_t}
\left(\delta_{z_j}(v_i)-\delta_{x_j}(v_i)\right).
$$

等价地，

$$
Q_t^z(v_i,j\mid x_j)=\frac{\dot\kappa_t}{1-\kappa_t}
\begin{cases}
0,&x_j=z_j,\\
1,&v_i=z_j,\ x_j\neq z_j,\\
0,&v_i\neq z_j,\ x_j\neq z_j,\\
-1,&v_i=x_j,\ x_j\neq z_j.
\end{cases}
$$

如果当前位置已经正确，速率为零；如果不正确，就跳到目标词元。

## 第 22 页

（本页无可提取文字。）

## 第 23 页

### 因子化混合路径的条件速率矩阵

条件速率矩阵仍为

$$
Q_t^z(v_i,j\mid x_j)
=\frac{\dot\kappa_t}{1-\kappa_t}
\left(\delta_{z_j}(v_i)-\delta_{x_j}(v_i)\right).
$$

当 \(t\to1\) 时，分母 \(1-\kappa_t\to0\)，因此速率可能发散。这反映了模型在终点附近必须快速完成剩余的词元更新。

## 第 24 页

### 条件概率路径、向量场与分数

在离散情形中，对应关系为：

| 对象 | 记号 | 关键性质 | 因子化混合路径 |
|---|---|---|---|
| 条件概率路径 | \(p_t(x\mid z)\) | 在初始分布与数据点 \(z\) 之间插值 | \(\displaystyle\prod_{j=1}^{d}[(1-\kappa_t)p_{\mathrm{init}}^{(j)}(x_j)+\kappa_t\delta_{z_j}(x_j)]\) |
| 条件速率矩阵 | \(Q_t^z(y\mid x)\) | CTMC 遵循条件路径 | \(\displaystyle Q_t^z(v_i,j\mid x_j)=\frac{\dot\kappa_t}{1-\kappa_t}(\delta_{z_j}(v_i)-\delta_{x_j}(v_i))\) |

## 第 25 页

### 边缘概率路径、速率矩阵与分数

边缘概率路径为

$$
p_t(x)=\sum_{z\in S}p_t(x\mid z)p_{\mathrm{data}}(z),
$$

它在 \(p_{\mathrm{init}}\) 与 \(p_{\mathrm{data}}\) 之间插值。边缘速率矩阵满足

$$
Q_t(y\mid x)
=\sum_{z\in S}Q_t^z(y\mid x)
\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}.
$$

## 第 26 页

### 因子化混合路径的边缘速率矩阵

条件速率矩阵是已知的；未知部分是后验概率

$$
p_{1\mid t}(z_j=v_i\mid x).
$$

边缘速率矩阵为

$$
Q_t(v_i,j\mid x)
=\frac{\dot\kappa_t}{1-\kappa_t}
\left(p_{1\mid t}(z_j=v_i\mid x)-\delta_{x_j}(v_i)\right).
$$

因此，只要学习给定当前噪声序列 \(x\) 时终点词元的条件概率，就可以得到边缘速率矩阵。

## 第 27 页

### 离散流匹配损失

使用后验概率网络预测每个位置的终点词元分布：

$$
p_{1\mid t}^{\theta}(z_j\mid x).
$$

通过分类训练网络：

$$
\mathcal{L}_{\mathrm{DFM}}(\theta)
=\mathbb{E}_{\substack{z\sim p_{\mathrm{data}},\ t\sim\mathrm{Unif}_{[0,1]}\\
x\sim p_t(\cdot\mid z)}}
\left[-\sum_{j=1}^{d}\log p_{1\mid t}^{\theta}(z_j\mid x)\right].
$$

这就是每个序列位置上的交叉熵损失。

## 第 28 页

（本页无可提取文字。）

## 第 29 页

### 掩码扩散语言模型

向词表中引入新词元 '[MASK]'，表示参考词元被掩码。

初始分布可以设为所有位置都为 '[MASK]'：

$$
X_0=([\mathrm{MASK}],\ldots,[\mathrm{MASK}]).
$$

随后模型逐步预测并替换被掩码的词元。

## 第 30 页

（本页无可提取文字。）

## 第 31 页

### 从掩码语言模型采样

从完全掩码的初始分布开始：

> '[MASK]' '[MASK]' '[MASK]' ... '[MASK]'

每一步都根据当前部分已知的序列预测更多词元。

## 第 32 页

### \(t=0.3\)

此时大多数词元仍然是 '[MASK]'，只有少量位置已被模型填充。例如：

> Many years later, as he faced '[MASK]' '[MASK]', '[MASK]' was to remember '[MASK]' when '[MASK]' ...

## 第 33 页

### \(t=0.6\)

更多词元被还原：

> Many years later, as he faced '[MASK]' '[MASK]', Colonel Aureliano Buendía was to remember that distant '[MASK]' when his father took him to discover '[MASK]'. At that time Macondo was a village of twenty adobe houses, built on the bank of a river of clear water ...

## 第 34 页

### \(t=0.8\)

序列已经接近完整：

> Many years later, as he faced the firing squad, Colonel Aureliano Buendía was to remember that distant '[MASK]' when his father took him to discover ice. At that time Macondo was a village of twenty adobe houses, built on the bank of a river of clear water that ran along a bed of polished stones, which were white and enormous, like prehistoric eggs. The world was so recent that many things lacked names, and in order to indicate them it was '[MASK]' to point.

## 第 35 页

### \(t=1.0\)

所有词元都已还原：

> 许多年以后，面对行刑队时，奥雷里亚诺·布恩迪亚上校会想起那个遥远的下午：父亲带他去见识冰块。那时，马孔多还是一座村庄，有二十间土坯房，建在一条清澈河流的岸边；河水沿着一床光滑的石头流过，那些石头洁白而巨大，像史前的蛋。世界还很年轻，许多事物都没有名字；为了指称它们，人们不得不用手指指向它们。

## 第 36 页

### 掩码扩散大语言模型生成文本

掩码扩散语言模型通过逐步去除掩码来生成文本，而不是固定地从左到右生成。

参考：Austin 等，《Structured Denoising Diffusion Models in Discrete State-Spaces》，NeurIPS 2022。

图示：Subham Sahoo。

## 第 37 页

### 离散扩散模型与自回归模型的讨论

**优点：**

- 可以并行生成多个词元，可能带来更高速度。
- 可以按任意顺序生成词元，因此可能支持文本编辑。
- 可以设计新的概率路径，探索更有语义意义的生成过程。

**缺点：**

- 没有 KV cache，速度优势可能受限。
- 必须学习任意顺序生成词元，学习难度可能更高。
- 自回归模型从左到右的顺序具有语义合理性，需要权衡改变顺序是否值得。

## 第 38 页

### 连续空间中的流匹配

再次回顾连续流匹配的结构：

| 条件概率路径 | 条件向量场 | 条件流匹配损失 |
|---|---|---|
| \(p_t(x\mid z)\) | \(u_t^z(x)\) | 回归损失 |
| 边缘概率路径 \(p_t(x)\) | \(u_t(x)\) | 边缘流匹配损失 |

## 第 39 页

### 离散流匹配矩阵

离散版本把向量场替换成速率矩阵，把回归目标替换成分类目标：

| 条件概率路径 | 条件速率矩阵 | 离散流匹配损失 |
|---|---|---|
| \(p_t(x\mid z)\) | \(Q_t^z(y\mid x)\) | 交叉熵 |
| 边缘概率路径 \(p_t(x)\) | \(Q_t(y\mid x)\) | 离散流匹配损失 |

## 第 40 页

### 为什么离散数据和 CTMC 中的流匹配流程如此相似？

流匹配背后的原理更加一般。它可以从一大类马尔可夫过程中推导出来，而不局限于欧氏空间中的 ODE 或 SDE。

## 第 41 页

### 课程回顾

- 讲座 1：流模型与扩散模型。
- 讲座 2：流匹配——训练算法。
- 讲座 3：分数匹配与引导——如何以提示为条件。
- 讲座 4：构建图像生成器——潜空间与网络架构。
- 讲座 5：离散扩散模型与离散流匹配。

## 第 42 页

### 这是我们的最后一节课！

感谢你的参与！

请填写课程评价问卷。
