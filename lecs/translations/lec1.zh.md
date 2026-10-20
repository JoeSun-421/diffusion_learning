# 讲座 1：流模型与扩散模型

## 第 1 页

基于随机微分方程（stochastic differential equations, SDEs）的生成式人工智能（Generative AI）

流模型（flow models）与扩散模型（diffusion models）导论

MIT IAP 2026 | Jan 20, 2025

Peter Holderrieth 与 Ron Shprints

赞助人：Tommi Jaakkola

## 第 2 页

欢迎来到课程 6.S184！

课程讲师：Peter、Ron

赞助人：Tommi

## 第 3 页

生成式人工智能：新一代人工智能系统

- 艺术图像
- 写实视频
- 文本草稿：ChatGPT

这些系统是“有创造力的”：它们生成新的对象。

本课教你用来生成对象的算法。

## 第 4 页

流模型与扩散模型：生成图像、视频与蛋白质的最先进（state-of-the-art, SOTA）模型！

- Stable Diffusion、DALEE
- OpenAI Sora、Meta MovieGen
- AlphaFold3、Boltz

用于图像、视频、蛋白质与机器人的大多数最先进生成式人工智能模型：扩散模型与流模型。

## 第 5 页

本课：流模型与扩散模型的理论与实践

- 理论：常微分方程（ordinary differential equations, ODEs）与随机微分方程
- 实践：如何实现并应用这些机器学习（machine learning）模型

本课的目标是教给你：

1. 从第一性原理（first principles）出发理解流模型与扩散模型。
2. 为第 1 点所需的、最少但必要的数学。
3. 如何实现并应用这些算法。

## 第 6 页

课程概览

- 讲座 1（今天）：
  - 从生成到采样（sampling）：把“生成一张图像等等”形式化。
  - 构造流模型与扩散模型
- 讲座 2 — 流匹配（Flow Matching）：训练算法。
- 讲座 3 — 分数匹配（Score Matching）、引导（Guidance）：如何以提示（prompt）为条件。
- 讲座 4 — 构建图像生成器：网络架构（network architectures）+ 潜空间（latent spaces）
- 讲座 5 — 进阶专题：离散扩散模型（discrete diffusion models）、蒸馏模型（distilled models）

## 第 7 页

第 1 节：从生成到采样

目标：把“生成”某物的含义形式化。

## 第 8 页

我们把图像、视频与蛋白质表示为向量（vectors）

图像：

- 高度 \(H\) 与宽度 \(W\)
- 3 个颜色通道（RBG）

$$
z \in \mathbb{R}^{H \times W \times 3}
$$

视频：

- \(T\) 个时间帧
- 每一帧都是一幅图像

$$
z \in \mathbb{R}^{T \times H \times W \times 3}
$$

分子结构：

- \(N\) 个原子
- 每个原子有 3 个坐标

$$
z \in \mathbb{R}^{N \times 3}
$$

我们把想要生成的对象表示为向量：

$$
z \in \mathbb{R}^{d}
$$

## 第 9 页

成功生成某物意味着什么？

提示：“一张狗的照片”

无用 \(<\) 差 \(<\) 动物不对 \(<\) 很好！

这些是主观判断。我们能把它形式化吗？

## 第 10 页

数据分布（data distribution）：我们有多“可能”在互联网上找到这张图？

提示：“一张狗的照片”

不可能 \(<\) 罕见 \(<\) 不太可能 \(<\) 非常可能

一张图像有多好 \(\approx\) 它在数据分布下有多可能。

## 第 11 页

生成即从数据分布中采样

数据分布：我们想要生成的那些对象的分布：\(p_{\mathrm{data}}\)

概率密度（probability density）：

$$
p_{\mathrm{data}} : \mathbb{R}^{d} \to \mathbb{R}_{\geq 0}, \qquad z \mapsto p_{\mathrm{data}}(z)
$$

注：我们并不知道这个概率密度！

生成意味着对数据分布采样：

$$
z \sim p_{\mathrm{data}}
$$

## 第 12 页

数据集（dataset）由数据分布中的样本组成

训练需要数据集：要训练我们的算法，就需要一个数据集。

例子：

- 图像：互联网上公开可得的图像
- 视频：YouTube
- 蛋白质结构：科学数据（例如 Protein Data Bank）

数据集由数据分布中的有限个样本组成：

$$
z_1, \ldots, z_N \sim p_{\mathrm{data}}
$$

## 第 13 页

条件生成（conditional generation）让我们可以在提示上做条件化

- 数据分布 \(p_{\mathrm{data}}\)：固定提示“狗”
- 条件数据分布（conditional data distribution）\(p_{\mathrm{data}}(\cdot \mid y)\)：条件变量（condition variable）\(y\)
  - \(y =\) “狗”
  - \(y =\) “猫”
  - \(y =\) “风景”

条件生成意味着对条件数据分布采样：

$$
z \sim p_{\mathrm{data}}(\cdot \mid y)
$$

我们会先专注于无条件生成（unconditional generation），然后再学习如何把无条件模型翻译成条件模型。

## 第 14 页

生成模型（generative models）从数据分布中生成样本

初始分布（initial distribution）：\(p_{\mathrm{init}}\)

默认：

$$
p_{\mathrm{init}} = \mathcal{N}(0, I_d)
$$

生成模型把来自初始分布（例如高斯分布（Gaussian））的样本，转换成来自数据分布的样本：

\(x \sim p_{\mathrm{init}}\) \(\rightarrow\) 生成模型 \(\rightarrow\) \(z \sim p_{\mathrm{data}}\)

## 第 15 页

小结

- 要生成的对象：我们关注表示数据对象（例如图像、视频）的向量 \(z\)。
- 数据分布：对我们认为“好”的对象赋予更高概率的分布。
- 生成即采样：生成一个对象 = 从数据分布中采样。
- 数据集：用于训练的、来自数据分布的有限个样本。
- 条件生成：在标签 \(y\) 上做条件化，并从条件数据分布中采样。
- 生成模型：训练一个模型，把来自简单分布（例如高斯分布）的样本变换成数据分布。

## 第 16 页

第 2 节：流模型与扩散模型

目标：理解微分方程，以及如何用它们构建生成模型。

## 第 17 页

常微分方程轨迹（trajectory）示例

来源：https://mariogemoll.com/flow-matching

## 第 18 页

流（flow）示例

## 第 19 页

常微分方程的存在唯一性定理

定理（Picard–Lindelöf 定理（Picard–Lindelöf theorem））：若向量场（vector field）\(u_t(x)\) 连续可微（continuously differentiable）且导数有界，则常微分方程

$$
X_0 = x_0, \qquad \frac{\mathrm{d}}{\mathrm{d}t} X_t = u_t(X_t)
$$

存在唯一解。换言之，流映射（flow map）存在。更一般地，只要向量场是利普希茨（Lipschitz）的，结论就成立。

要点：在机器学习所关心的情形中，常微分方程与流存在唯一解。

数学课：通过 Picard 迭代（Picard iteration）构造解。

## 第 20 页

例子：线性常微分方程

简单向量场：

$$
u_t(x) = -\theta x \qquad (\theta > 0)
$$

断言：流由下式给出

$$
\psi_t(x_0) = \exp(-\theta t)\, x_0
$$

证明：

1. 初始条件：

$$
\psi_0(x_0) = \exp(0)\, x_0 = x_0
$$

2. 常微分方程：

$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d}t}\psi_t(x_0)
&= \frac{\mathrm{d}}{\mathrm{d}t}\bigl(\exp(-\theta t)\, x_0\bigr) \\
&= -\theta \exp(-\theta t)\, x_0 \\
&= -\theta \psi_t(x_0) \\
&= u_t\bigl(\psi_t(x_0)\bigr).
\end{aligned}
$$

横轴：时间（\(t\)）。纵轴：\(X_t\)。

## 第 21 页

常微分方程的数值仿真 — 欧拉法（Euler method）

**算法 1** 用欧拉法仿真常微分方程

**需要：** 向量场 \(u_t\)，初始条件 \(x_0\)，步数 \(n\)

1. 令 \(t = 0\)
2. 令步长（step size）\(h = \frac{1}{n}\)
3. 令 \(X_0 = x_0\)
4. 对 \(i = 1, \ldots, n - 1\) 执行
5. \(X_{t+h} = X_t + h u_t(X_t)\)（沿向量场方向迈出一小步）
6. 更新 \(t \leftarrow t + h\)
7. 结束循环
8. 返回轨迹 \(X_0, X_h, X_{2h}, \ldots, X_1\)

## 第 22 页

欧拉法示例

- 大步长：更高效，但误差更高
- 小步长：误差更低，但效率更低

来源：https://mariogemoll.com/flow-matching

## 第 23 页

玩具示例

图中标注：\(p_{\mathrm{init}}\)，\(X_0\)，\(p_{\mathrm{data}}\)，\(X_1\)。

图注：Yaron Lipman

## 第 24 页

玩具流模型

图注：Yaron Lipman

## 第 25 页

如何用流模型生成对象

**算法 1** 用欧拉法从流模型中采样

**需要：** 神经网络（neural network）向量场 \(u_t^{\theta}\)，步数 \(n\)

1. 令 \(t = 0\)
2. 令步长 \(h = \frac{1}{n}\)
3. 抽取样本 \(X_0 \sim p_{\mathrm{init}}\)（随机初始化！）
4. 对 \(i = 1, \ldots, n - 1\) 执行
5. \(X_{t+h} = X_t + h u_t^{\theta}(X_t)\)
6. 更新 \(t \leftarrow t + h\)
7. 结束循环
8. 返回最终点 \(X_1\)

## 第 26 页

示例：通过流模型采样生成的视频

## 第 27 页

随机过程（stochastic process）：随机轨迹

样本 1、样本 2、样本 3

注：同一个随机过程可以产生许多条轨迹，因为演化本身变成随机的。

来源：https://mariogemoll.com/flow-matching

## 第 28 页

布朗运动（Brownian motion）：随机轨迹

样本 1、样本 2、样本 3

布朗运动的行为像一条连续的随机游走（random walk）。此处是二维的。

来源：https://mariogemoll.com/flow-matching

## 第 29 页

布朗运动

横轴：时间（\(t\)）。纵轴：\(W_t\)。

## 第 30 页

随机微分方程的存在唯一性定理

定理：若向量场 \(u_t(x)\) 连续可微且导数有界，并且扩散系数（diffusion coefficient）连续，则随机微分方程

$$
X_0 = x_0, \qquad \mathrm{d}X_t = u_t(X_t)\, \mathrm{d}t + \sigma_t\, \mathrm{d}W_t
$$

存在在分布意义下（in distribution）的唯一解。更一般地，只要向量场是利普希茨的，结论就成立。

要点：在机器学习所关心的情形中，随机微分方程存在唯一解。

随机微积分课：通过随机积分（stochastic integrals）与 Ito–Riemann 和（Ito–Riemann sums）构造解。

## 第 31 页

随机微分方程的数值仿真（欧拉–丸山方法（Euler–Maruyama method））

**算法 2** 从随机微分方程采样（欧拉–丸山方法）

**需要：** 向量场 \(u_t\)，步数 \(n\)，扩散系数 \(\sigma_t\)

1. 令 \(t = 0\)
2. 令步长 \(h = \frac{1}{n}\)
3. 令 \(X_0 = x_0\)
4. 对 \(i = 1, \ldots, n - 1\) 执行
5. 抽取样本 \(\epsilon \sim \mathcal{N}(0, I_d)\)
6. \(X_{t+h} = X_t + h u_t(X_t) + \sigma_t \sqrt{h}\, \epsilon\)（额外加入方差为 \(h\) 的噪声，并按扩散系数 \(\sigma_t\) 缩放）
7. 更新 \(t \leftarrow t + h\)
8. 结束循环
9. 返回 \(X_0, X_h, X_{2h}, X_{3h}, \ldots, X_1\)

## 第 32 页

Ornstein–Uhlenbeck 过程（Ornstein–Uhlenbeck process）

$$
\mathrm{d}X_t = -\theta X_t\, \mathrm{d}t + \sigma\, \mathrm{d}W_t
$$

扩散系数 \(\sigma\) 逐渐增大。横轴：时间（\(t\)）。纵轴：\(X_t\)。

## 第 33 页

**算法 2** 从扩散模型采样（欧拉–丸山方法）

**需要：** 神经网络 \(u_t^{\theta}\)，步数 \(n\)，扩散系数 \(\sigma_t\)

1. 令 \(t = 0\)
2. 令步长 \(h = \frac{1}{n}\)
3. 抽取样本 \(X_0 \sim p_{\mathrm{init}}\)
4. 对 \(i = 1, \ldots, n - 1\) 执行
5. 抽取样本 \(\epsilon \sim \mathcal{N}(0, I_d)\)
6. \(X_{t+h} = X_t + h u_t^{\theta}(X_t) + \sigma_t \sqrt{h}\, \epsilon\)
7. 更新 \(t \leftarrow t + h\)
8. 结束循环
9. 返回 \(X_1\)

## 第 34 页

常微分方程采样　　随机微分方程采样

## 第 35 页

课程安排

网站：https://diffusion.csail.mit.edu/

如何通过本课：

1. 来上课
2. 做实验（要通过就必须做）！

支持：

1. 使用讲义（自成体系）
2. 来答疑时间

实验 1 今天发布（见网站）！我们建议你尽快开始做。

## 第 36 页

下次课：

周四（后天），上午 11:00–12:30

E25-111（同一间教室）

答疑时间：周三（明天），上午 11:00–12:30，地点 36-144
