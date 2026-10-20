# 流模型与扩散模型课程：讲义 1–5 中文翻译

本文件按讲义顺序合并自同目录下的逐讲翻译；公式、页码标题和原有内容均保留。

---

<!-- 来源文件：lec1.zh.md -->

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

---

<!-- 来源文件：lec2.zh.md -->

# 讲座 2：流匹配

## 第 1 页

讲座 2

流匹配（Flow Matching）

MIT IAP 2026｜2025 年 1 月 22 日

Peter Holderrieth 与 Ron Shprints

赞助人：Tommi Jaakkola

## 第 2 页

回顾：流模型（flow model）与扩散模型（diffusion model）

**流模型**

初始化：\(X_0 \sim p_{\mathrm{init}}\)（例如高斯（Gaussian））。

常微分方程（ODE）：

$$
\mathrm{d}X_t = u_t^{\theta}(X_t)\,\mathrm{d}t
$$

其中 \(u_t^{\theta}\) 是神经网络（neural network）向量场（vector field）。

**扩散模型**

初始化：\(X_0 \sim p_{\mathrm{init}}\)。

随机微分方程（SDE）：

$$
\mathrm{d}X_t = u_t^{\theta}(X_t)\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t
$$

其中 \(\sigma_t\) 是扩散系数（diffusion coefficient）。

要得到样本（sample），从 \(t=0\) 到 \(t=1\) 模拟 ODE 或 SDE，并返回 \(X_1\)。

## 第 3 页

下一步：训练流模型

若不训练，模型产生的是「无意义」的结果，因此我们需要训练 \(u_t^{\theta}\)。

训练就是寻找参数（parameter）\(\theta\)，使得

$$
X_0 \sim p_{\mathrm{init}}, \qquad \mathrm{d}X_t = u_t^{\theta}(X_t)\,\mathrm{d}t
$$

蕴含

$$
X_1 \sim p_{\mathrm{data}}.
$$

- 从初始分布（initial distribution）出发。
- 沿着向量场前进。
- 终点的分布等于数据分布（data distribution）。

本讲（今天）的目标：推导流匹配，也就是流模型的训练算法（training algorithm）。

## 第 4 页

第 2 节：流匹配

目标：推导流模型的训练算法。

## 第 5 页

流匹配矩阵

| 条件（conditional）概率路径（probability path） | 条件向量场 | 条件流匹配损失（flow matching loss） |
| --- | --- | --- |
| 边缘（marginal）概率路径 | 边缘向量场 | 边缘流匹配损失 |

每一行从左到右依次箭头连接。

「条件」等于「针对单个数据点（data point）」。

「边缘」等于「在数据点的分布上整体来看」。

## 第 6 页

概率路径：从噪声（noise）到数据的路径

图中时间轴从 \(t=0\) 指向 \(t=1\)。左端标为噪声，右端标为数据。上方一行是一张猫的图像，从纯噪声逐步变得清晰。下方一行是数字网格，从噪声逐步变为清晰的手写数字。

## 第 7 页

条件概率路径 \(p_t(\cdot \mid z)\)

左侧为 \(p_{\mathrm{init}}\)。横轴从 \(t=0\) 到 \(t=1\)。五个时刻的密度图依次为 \(t=0.00\)、\(t=0.25\)、\(t=0.50\)、\(t=0.75\)、\(t=1.00\)：一团分布逐渐收缩，最后集中到数据点 \(z\)（红星）。纵轴标注为「条件」。

## 第 8 页

在时间上绘制条件概率路径的样本

图中是黑底上的一团绿色样本点。

注：概率路径只规定各时刻的边缘分布（每一张快照）。它完全不说明单个粒子随时间如何演化（没有动力学（dynamics））。

来源：https://mariogemoll.com/flow-matching

## 第 9 页

条件概率路径 \(p_t(\cdot \mid z)\)

上排与第 7 页相同：分布从 \(p_{\mathrm{init}}\) 收缩到单个数据点 \(z\)。

标注 \(z \sim p_{\mathrm{data}}\)，箭头指向一张棋盘格式的样本图，其旁标为 \(p_{\mathrm{data}}\)。

## 第 10 页

上排：条件概率路径 \(p_t(\cdot \mid z)\)，从 \(p_{\mathrm{init}}\) 收缩到单个数据点 \(z\)。

下排：边缘概率路径 \(p_t\)。左侧为 \(p_{\mathrm{init}}\)，右侧为 \(p_{\mathrm{data}}\)。密度从一团高斯云逐渐铺开，最后成为棋盘格分布。时刻同样为 \(t=0.00\)、\(t=0.25\)、\(t=0.50\)、\(t=0.75\)、\(t=1.00\)。

## 第 11 页

条件概率路径

|  | 记号（notation） | 关键性质（key property） | 高斯例子 |
| --- | --- | --- | --- |
| 条件概率路径 | \(p_t(\cdot \mid z)\) | 在 \(p_{\mathrm{init}}\) 与数据点 \(z\) 之间插值（interpolate） | \(\mathcal{N}(\alpha_t z, \beta_t^2 I_d)\) |
| 条件向量场 |  |  |  |

## 第 12 页

边缘概率路径

|  | 记号 | 关键性质 | 公式（formula） |
| --- | --- | --- | --- |
| 边缘概率路径 | \(p_t\) | 在 \(p_{\mathrm{init}}\) 与 \(p_{\mathrm{data}}\) 之间插值 | \(\displaystyle \int p_t(x \mid z)\, p_{\mathrm{data}}(z)\,\mathrm{d}z\) |
| 边缘向量场 |  |  |  |

## 第 13 页

例子：高斯情形的条件向量场

$$
u_t^{\mathrm{target}}(x \mid z) = \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right) z + \frac{\dot{\beta}_t}{\beta_t} x
$$

这里的上标 target 表示目标（target）。

证明提纲：

步骤 1：通过检验 ODE，说明该向量场的流（flow）为

$$
\psi_t^{\mathrm{target}}(x_0 \mid z) = \alpha_t z + \beta_t x_0.
$$

步骤 2：若 \(X_0 = x_0 \sim \mathcal{N}(0, I_d)\) 是随机的，则

$$
X_t = \psi_t(X_0 \mid z) = \alpha_t z + \beta_t X_0 \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d) = p_t(\cdot \mid z).
$$

## 第 14 页

高斯条件概率路径与条件向量场

图中，左侧一团同心等高线沿直线逐渐缩小，汇聚到右侧的一个点。一条带箭头的轨线指向该点。

图注：Yaron Lipman

## 第 15 页

用条件向量场模拟 ODE，以得到条件概率路径

注：这是动画 GIF，在 PDF 中为静态画面。

图中是一张网格。蓝色箭头构成向量场，中央有一团正在收缩的同心等高线。

图注：Yaron Lipman

## 第 16 页

玩具例子

注：这是动画 GIF，在 PDF 中为静态画面。

图中左侧标为 \(p_{\mathrm{init}}\)，是一团同心等高线；一张噪声图 \(X_0\) 指向其中一点。右侧标为 \(p_{\mathrm{data}}\)，是一组双峰轮廓；其中一点连到一张狗的图片 \(X_1\)。

图注：Yaron Lipman

## 第 17 页

用边缘向量场模拟 ODE，以得到高斯概率路径

图中是一张网格，中央一群黑点叠在收缩的同心等高线上。

图注：Yaron Lipman

## 第 18 页

三列标题：真值（ground truth）、ODE 样本、ODE 轨迹（trajectory）。

两行标题：\(p_t(\cdot \mid z)\) 与 \(p_t\)。

上排是条件概率路径，下排是边缘概率路径。图例用颜色区分时刻，星号表示数据点 \(z\)。左列是该路径的真实样本，中列是 ODE 样本，右列是 ODE 轨迹。下排右图中，轨迹从中心散开，通向周围各个模式。

## 第 19 页

连续性方程（continuity equation）

随机初始化的 ODE

给定：

$$
X_0 \sim p_{\mathrm{init}}, \qquad \frac{\mathrm{d}}{\mathrm{d}t} X_t = u_t(X_t).
$$

跟随概率路径：

$$
X_t \sim p_t \quad (0 \le t \le 1).
$$

边缘分布是 \(p_t\)。

上式与下面的陈述等价：连续性方程成立，

$$
\frac{\mathrm{d}}{\mathrm{d}t} p_t(x) = -\operatorname{div}(p_t u_t)(x).
$$

也就是偏微分方程（PDE）成立。

## 第 20 页

连续性方程

$$
\frac{\mathrm{d}}{\mathrm{d}t} p_t(x) = -\operatorname{div}(p_t u_t)(x).
$$

左端是 \(x\) 处概率质量（probability mass）的变化。右端是由 \(u\) 造成的概率质量流出（outflow）减去流入（inflow）。

图中背景为向量场 \(u_t(x)\)。方框标出位置 \(x\)：上方箭头离开方框，标为流出；下方箭头进入方框，标为流入。

## 第 21 页

条件概率路径、向量场与分数（score）

|  | 记号 | 关键性质 | 高斯例子 |
| --- | --- | --- | --- |
| 条件概率路径 | \(p_t(\cdot \mid z)\) | 在 \(p_{\mathrm{init}}\) 与数据点 \(z\) 之间插值 | \(\mathcal{N}(\alpha_t z, \beta_t^2 I_d)\) |
| 条件向量场 | \(u_t^{\mathrm{target}}(x \mid z)\) | ODE 跟随条件路径 | \(\displaystyle\left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right) z + \frac{\dot{\beta}_t}{\beta_t} x\) |

## 第 22 页

边缘概率路径、向量场与分数

|  | 记号 | 关键性质 | 公式 |
| --- | --- | --- | --- |
| 边缘概率路径 | \(p_t\) | 在 \(p_{\mathrm{init}}\) 与 \(p_{\mathrm{data}}\) 之间插值 | \(\displaystyle \int p_t(x \mid z)\, p_{\mathrm{data}}(z)\,\mathrm{d}z\) |
| 边缘向量场 | \(u_t^{\mathrm{target}}(x)\) | ODE 跟随边缘路径 | \(\displaystyle \int u_t^{\mathrm{target}}(x \mid z)\, \frac{p_t(x \mid z)\, p_{\mathrm{data}}(z)}{p_t(x)}\,\mathrm{d}z\) |

## 第 23 页

边缘向量场示例：Meta MovieGen

图为两段视频的画面。左图是一只猴子泡在热气腾腾的水中，右图是一段明亮、虚化的动物镜头。

这些视频是通过用学到的边缘向量场模拟 ODE 生成的。

## 第 24 页

算法 3：流匹配训练流程（一般情形）

**需要：** 数据集，其中样本 \(z \sim p_{\mathrm{data}}\)；神经网络 \(u_t^{\theta}\)。

**1.** 对每个数据小批量（mini-batch）执行

**2.** 从数据集中采样一个数据样本 \(z\)。

**3.** 采样随机时间 \(t \sim \mathrm{Unif}_{[0,1]}\)（\([0,1]\) 上的均匀分布（uniform））。

**4.** 采样 \(x \sim p_t(\cdot \mid z)\)。

**5.** 计算损失

$$
\mathcal{L}(\theta) = \left\lVert u_t^{\theta}(x) - u_t^{\mathrm{target}}(x \mid z) \right\rVert^{2}.
$$

**6.** 通过对 \(\mathcal{L}(\theta)\) 做梯度下降（gradient descent），更新模型参数 \(\theta\)。

**7.** 结束循环

## 第 25 页

高斯概率路径上的条件流匹配（conditional flow matching, CFM）

概率路径：\(\mathcal{N}(\alpha_t z, \beta_t^2 I_d)\)。

条件向量场：

$$
u_t^{\mathrm{target}}(x \mid z) = \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right) z + \frac{\dot{\beta}_t}{\beta_t} x.
$$

噪声采样（noise sampling）：

$$
x \sim p_t(\cdot \mid z) \iff x = \alpha_t z + \beta_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_d).
$$

把噪声采样代入 CFM 损失，得到

$$
\begin{aligned}
L_{\mathrm{CFM}}(\theta)
&= \mathbb{E}_{t \sim \mathrm{Unif},\, z \sim p_{\mathrm{data}},\, x \sim p_t(\cdot \mid z)}
\left[\left\lVert u_t^{\theta}(x) - u_t^{\mathrm{target}}(x \mid z) \right\rVert^{2}\right] \\
&= \mathbb{E}_{t \sim \mathrm{Unif},\, z \sim p_{\mathrm{data}},\, \epsilon \sim \mathcal{N}(0, I_d)}
\left[\left\lVert u_t^{\theta}(\alpha_t z + \beta_t \epsilon) - u_t^{\mathrm{target}}(\alpha_t z + \beta_t \epsilon \mid z) \right\rVert^{2}\right] \\
&= \mathbb{E}_{t \sim \mathrm{Unif},\, z \sim p_{\mathrm{data}},\, \epsilon \sim \mathcal{N}(0, I_d)}
\left[\left\lVert u_t^{\theta}(\alpha_t z + \beta_t \epsilon) - (\dot{\alpha}_t z + \dot{\beta}_t \epsilon) \right\rVert^{2}\right].
\end{aligned}
$$

其中 \(\alpha_t z + \beta_t \epsilon\) 标为噪声 + 数据，\(\dot{\alpha}_t z + \dot{\beta}_t \epsilon\) 标为速度（velocity）。

## 第 26 页

直线调度（straight line schedule）

$$
\begin{aligned}
L_{\mathrm{CFM}}(\theta)
&= \mathbb{E}_{t \sim \mathrm{Unif},\, z \sim p_{\mathrm{data}},\, \epsilon \sim \mathcal{N}(0, I_d)}
\left[\left\lVert u_t^{\theta}(\alpha_t z + \beta_t \epsilon) - (\dot{\alpha}_t z + \dot{\beta}_t \epsilon) \right\rVert^{2}\right] \\
&= \mathbb{E}_{t \sim \mathrm{Unif},\, z \sim p_{\mathrm{data}},\, \epsilon \sim \mathcal{N}(0, I_d)}
\left[\left\lVert u_t^{\theta}\bigl(tz + (1-t)\epsilon\bigr) - (z - \epsilon) \right\rVert^{2}\right].
\end{aligned}
$$

\(tz + (1-t)\epsilon\) 是噪声与数据的线性插值（linear interpolation）。

\(z - \epsilon\) 是噪声与数据之差。

## 第 27 页

图注：Yaron Lipman

图中，左侧噪声图像标为 \(\epsilon\)，右侧一张狗的照片标为 \(z\)。两点之间有一条蓝色直线，中点标为

$$
tz + (1-t)\epsilon.
$$

背景是两端分布的等高线。

## 第 28 页

算法 4：条件最优传输（CondOT）路径的流匹配训练

**需要：** 数据集，其中样本 \(z \sim p_{\mathrm{data}}\)；神经网络 \(u_t^{\theta}\)。

**1.** 对每个数据小批量执行

**2.** 从数据集中采样一个数据样本 \(z\)。

**3.** 采样随机时间 \(t \sim \mathrm{Unif}_{[0,1]}\)。

**4.** 采样噪声 \(\epsilon \sim \mathcal{N}(0, I_d)\)。

**5.** 令 \(x = tz + (1-t)\epsilon\)。

**6.** 计算损失

$$
\mathcal{L}(\theta) = \left\lVert u_t^{\theta}(x) - (z - \epsilon) \right\rVert^{2}.
$$

**7.** 通过对 \(\mathcal{L}(\theta)\) 做梯度下降，更新模型参数 \(\theta\)。

**8.** 结束循环

## 第 29 页

流匹配示例：Meta MovieGen

图与第 23 页相同：左为水中的猴子，右为明亮、虚化的动物镜头。

生成这些视频的神经网络是用上一页的算法训练的。

## 第 30 页

流匹配示例：Stable Diffusion 3

图为三张生成图像：行星地面上的巨大甜甜圈与宇航员；撑粉伞的猪、戴礼帽的鸟与月球表面；树屋、小鸡与几名宇航员。

生成这些图像的神经网络是用刚刚给出的算法训练的。

## 第 31 页

回顾：流模型的采样算法

算法 1：用 Euler 方法（Euler method）从流模型中采样

**需要：** 神经网络向量场 \(u_t^{\theta}\)，步数 \(n\)。

1. 令 \(t = 0\)。
2. 令步长 \(h = \dfrac{1}{n}\)。
3. 抽取样本 \(X_0 \sim p_{\mathrm{init}}\)。随机初始化！
4. **对** \(i = 1, \ldots, n-1\) **执行**
5. \(X_{t+h} = X_t + h u_t^{\theta}(X_t)\)。
6. 更新 \(t \leftarrow t + h\)。
7. **结束循环**
8. **返回** \(X_1\)。返回终点。

## 第 32 页

流匹配矩阵

| 条件概率路径 | 条件向量场 | 条件流匹配损失 |
| --- | --- | --- |
| 边缘概率路径 | 边缘向量场 | 边缘流匹配损失 |
| 定义从噪声到数据的分布 | 定义我们希望学习的训练目标 | 训练过程中希望最小化的损失函数 |

每一行从左到右依次箭头连接。下方三句话分别对应三列。

## 第 33 页

条件概率路径、向量场与流匹配损失

|  | 记号 | 关键性质 | 高斯例子 |
| --- | --- | --- | --- |
| 条件概率路径 | \(p_t(\cdot \mid z)\) | 在 \(p_{\mathrm{init}}\) 与数据点 \(z\) 之间插值 | \(\mathcal{N}(\alpha_t z, \beta_t^2 I_d)\) |
| 条件向量场 | \(u_t^{\mathrm{target}}(x \mid z)\) | ODE 跟随条件路径 | \(\displaystyle\left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right) z + \frac{\dot{\beta}_t}{\beta_t} x\) |
| 条件流匹配损失 | \(L_{\mathrm{CFM}}(\theta)\) | 训练时我们要最小化的损失 | \(\displaystyle \mathbb{E}_{t,z,x}\left[\left\lVert u_t^{\theta}(x) - u_t^{\mathrm{target}}(x \mid z) \right\rVert^{2}\right]\) |

这些对象都是可直接处理的（tractable）。它们只是解析公式！

## 第 34 页

边缘概率路径、向量场与流匹配损失

|  | 记号 | 关键性质 | 公式 |
| --- | --- | --- | --- |
| 边缘概率路径 | \(p_t\) | 在 \(p_{\mathrm{init}}\) 与 \(p_{\mathrm{data}}\) 之间插值 | \(\displaystyle \int p_t(x \mid z)\, p_{\mathrm{data}}(z)\,\mathrm{d}z\) |
| 边缘向量场 | \(u_t^{\mathrm{target}}(x)\) | ODE 跟随边缘路径 | \(\displaystyle \int u_t^{\mathrm{target}}(x \mid z)\, \frac{p_t(x \mid z)\, p_{\mathrm{data}}(z)}{p_t(x)}\,\mathrm{d}z\) |
| 边缘流匹配损失 | \(L_{\mathrm{FM}}(\theta)\) | 通过条件流匹配损失被隐式地最小化 | \(\displaystyle \mathbb{E}_{t,z,x}\left[\left\lVert u_t^{\theta}(x) - u_t^{\mathrm{target}}(x) \right\rVert^{2}\right]\) |

这些对象都不可直接处理。但我们仍然可以学习它们！

## 第 35 页

下一课：

周五（明天），上午 11:00–12:30

分数匹配（score matching）与引导（guidance）！

E25-111（同一教室）

办公时间：明天 15:00–16:30，地点 36-156

---

<!-- 来源文件：lec3.zh.md -->

# 讲座 3：分数匹配与引导

## 第 1 页

讲座 03

分数匹配（Score Matching）与引导（Guidance）

MIT IAP 2026 | Jan 23, 2025

Peter Holderrieth 与 Ron Shprints

赞助人：Tommi Jaakkola

## 第 2 页

回顾：条件概率路径（conditional probability path）与条件向量场（conditional vector field）

| | 记号 | 关键性质 | 高斯（Gaussian）例子 |
| --- | --- | --- | --- |
| 条件概率路径 | \(p_t(\cdot \mid z)\) | 在初始分布（initial distribution）\(p_{\mathrm{init}}\) 与数据点（data point）\(z\) 之间插值 | \(\mathcal{N}(\alpha_t z, \beta_t^2 I_d)\) |
| 条件向量场 | \(u_t^{\mathrm{target}}(x \mid z)\) | 常微分方程（ordinary differential equation, ODE）沿着条件路径前进 | \(\left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right) z + \frac{\dot{\beta}_t}{\beta_t} x\) |

## 第 3 页

回顾：边缘概率路径（marginal probability path）与边缘向量场（marginal vector field）

| | 记号 | 关键性质 | 公式 |
| --- | --- | --- | --- |
| 边缘概率路径 | \(p_t\) | 在 \(p_{\mathrm{init}}\) 与数据分布（data distribution）\(p_{\mathrm{data}}\) 之间插值 | \(\displaystyle\int p_t(x \mid z)\, p_{\mathrm{data}}(z)\, \mathrm{d}z\) |
| 边缘向量场 | \(u_t^{\mathrm{target}}(x)\) | 常微分方程沿着边缘路径前进 | \(\displaystyle\int u_t^{\mathrm{target}}(x \mid z)\, \frac{p_t(x \mid z)\, p_{\mathrm{data}}(z)}{p_t(x)}\, \mathrm{d}z\) |

## 第 4 页

**算法 3** 流匹配（Flow Matching）训练流程（一般情形）

**需要：** 数据集中的样本 \(z \sim p_{\mathrm{data}}\)，神经网络（neural network）\(u_t^{\theta}\)

1. 对每个数据小批量（mini-batch）执行
2. 从数据集中采样一个数据样本 \(z\)。
3. 采样一个随机时间 \(t \sim \mathrm{Unif}_{[0,1]}\)。
4. 采样 \(x \sim p_t(\cdot \mid z)\)。
5. 计算损失（loss）

$$
\mathcal{L}(\theta) = \lVert u_t^{\theta}(x) - u_t^{\mathrm{target}}(x \mid z) \rVert^2
$$

6. 通过在 \(\mathcal{L}(\theta)\) 上做梯度下降（gradient descent）更新模型参数 \(\theta\)。
7. 结束循环

我们可以对许多不同的数据点 \(z\)，通过近似条件向量场来学习边缘向量场。

## 第 5 页

回顾：流模型（flow model）的采样（sampling）算法

**算法 1** 用欧拉法（Euler method）从流模型中采样

**需要：** 神经网络向量场 \(u_t^{\theta}\)，步数 \(n\)

1. 令 \(t = 0\)
2. 令步长（step size）\(h = \frac{1}{n}\)
3. 抽取样本 \(X_0 \sim p_{\mathrm{init}}\)。随机初始化！
4. 对 \(i = 1, \ldots, n - 1\) 执行
5. \(X_{t+h} = X_t + h u_t^{\theta}(X_t)\)
6. 更新 \(t \leftarrow t + h\)
7. 结束循环
8. 返回最终点 \(X_1\)

## 第 6 页

第 4 节：训练算法之二，分数匹配

目标：从新的角度理解流模型与扩散模型（diffusion models），以及随机微分方程（stochastic differential equation, SDE）/ 随机采样（stochastic sampling）。

## 第 7 页

分数函数（score function）= 对数似然（log-likelihood）的梯度

对数似然：\(\log q(x)\)

分数函数：\(\nabla \log q(x)\)

左图：\(\log q(x)\) 的等高线。右图：同一分布的分数函数，画成指向高对数似然方向的向量场。

## 第 8 页

例子：高斯概率路径的分数

$$
\nabla \log p_t(x \mid z) = -\frac{1}{\beta_t^2} x + \frac{\alpha_t}{\beta_t^2} z
$$

证明：

$$
p_t(x \mid z) = \mathcal{N}(x; \alpha_t z, \beta_t^2 I_d) = \frac{1}{(2\pi)^{d/2} \beta_t^{d}} \exp\left(-\frac{1}{2\beta_t^2}\lVert x - \alpha_t z \rVert^2\right)
$$

$$
\log p_t(x \mid z) = \log \mathcal{N}(x; \alpha_t z, \beta_t^2 I_d) = -\frac{d}{2}\log(2\pi) - d\log \beta_t - \frac{1}{2\beta_t^2}\lVert x - \alpha_t z \rVert^2
$$

$$
\nabla \log p_t(x \mid z) = \nabla \log \mathcal{N}(x; \alpha_t z, \beta_t^2 I_d) = -\frac{x - \alpha_t z}{\beta_t^2}
$$

## 第 9 页

条件概率路径、向量场与分数

| | 记号 | 关键性质 | 高斯例子 |
| --- | --- | --- | --- |
| 条件概率路径 | \(p_t(\cdot \mid z)\) | 在 \(p_{\mathrm{init}}\) 与数据点 \(z\) 之间插值 | \(\mathcal{N}(\alpha_t z, \beta_t^2 I_d)\) |
| 条件向量场 | \(u_t^{\mathrm{target}}(x \mid z)\) | 常微分方程沿着条件路径前进 | \(\left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right) z + \frac{\dot{\beta}_t}{\beta_t} x\) |
| 条件分数函数（conditional score function） | \(\nabla \log p_t(x \mid z)\) | 对数似然的梯度 | \(\dfrac{\alpha_t}{\beta_t^2} z - \dfrac{1}{\beta_t^2} x\) |

## 第 10 页

边缘概率路径、向量场与分数

| | 记号 | 关键性质 | 公式 |
| --- | --- | --- | --- |
| 边缘概率路径 | \(p_t\) | 在 \(p_{\mathrm{init}}\) 与 \(p_{\mathrm{data}}\) 之间插值 | \(\displaystyle\int p_t(x \mid z)\, p_{\mathrm{data}}(z)\, \mathrm{d}z\) |
| 边缘向量场 | \(u_t^{\mathrm{target}}(x)\) | 常微分方程沿着边缘路径前进 | \(\displaystyle\int u_t^{\mathrm{target}}(x \mid z)\, \frac{p_t(x \mid z)\, p_{\mathrm{data}}(z)}{p_t(x)}\, \mathrm{d}z\) |
| 边缘分数函数（marginal score function） | \(\nabla \log p_t(x)\) | 可以把常微分方程目标改写成随机微分方程 | \(\displaystyle\int \nabla \log p_t(x \mid z)\, \frac{p_t(x \mid z)\, p_{\mathrm{data}}(z)}{p_t(x)}\, \mathrm{d}z\) |

## 第 11 页

观察：条件向量场和条件分数都是线性函数，只是系数不同。

| | 记号 | 关键性质 | 高斯例子 |
| --- | --- | --- | --- |
| 条件向量场 | \(u_t^{\mathrm{target}}(x \mid z)\) | 常微分方程沿着条件路径前进 | \(\left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right) z + \frac{\dot{\beta}_t}{\beta_t} x\) |
| 条件分数函数 | \(\nabla \log p_t(x \mid z)\) | 对数似然的梯度 | \(\dfrac{\alpha_t}{\beta_t^2} z - \dfrac{1}{\beta_t^2} x\) |

## 第 12 页

重参数化（reparameterization）：速度场（velocity field）\(\to\) 分数函数

$$
a_t = \left(\beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t\right), \qquad b_t = \frac{\dot{\alpha}_t}{\alpha_t}
$$

$$
u_t^{\mathrm{target}}(x \mid z) = a_t \nabla \log p_t(x \mid z) + b_t x
$$

$$
u_t^{\mathrm{target}}(x) = a_t \nabla \log p_t(x) + b_t x
$$

证明：代数运算。代入公式。见讲义。

早期扩散模型学习的是分数函数，然后再把它变换成向量场。二者等价。

## 第 13 页

**算法 6** 分数匹配训练流程（一般情形）

**需要：** 数据集中的样本 \(z \sim p_{\mathrm{data}}\)，分数网络（score network）\(s_t^{\theta}\)

1. 对每个数据小批量执行
2. 从数据集中采样一个数据样本 \(z\)。
3. 采样一个随机时间 \(t \sim \mathrm{Unif}_{[0,1]}\)。
4. 采样 \(x \sim p_t(\cdot \mid z)\)。
5. 计算损失

$$
\mathcal{L}(\theta) = \lVert s_t^{\theta}(x) - \nabla \log p_t(x \mid z) \rVert^2
$$

6. 通过在 \(\mathcal{L}(\theta)\) 上做梯度下降更新模型参数 \(\theta\)。
7. 结束循环

## 第 14 页

高斯概率路径的去噪分数匹配（denoising score matching）

$$
\nabla \log p_t(x \mid z) = -\frac{x - \alpha_t z}{\beta_t^2}
$$

$$
\epsilon \sim \mathcal{N}(0, I_d) \implies x = \alpha_t z + \beta_t \epsilon \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d)
$$

$$
\mathcal{L}_{\mathrm{dsm}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif},\, z \sim p_{\mathrm{data}},\, x \sim p_t(\cdot \mid z)}\left[\left\lVert s_t^{\theta}(x) + \frac{x - \alpha_t z}{\beta_t^2} \right\rVert^2\right]
$$

$$
= \mathbb{E}_{t \sim \mathrm{Unif},\, z \sim p_{\mathrm{data}},\, \epsilon \sim \mathcal{N}(0, I_d)}\left[\left\lVert s_t^{\theta}(\alpha_t z + \beta_t \epsilon) + \frac{\epsilon}{\beta_t} \right\rVert^2\right]
$$

注意网络在做什么：它需要预测用来腐蚀数据点的噪声（noise）。这就是去噪扩散模型。

## 第 15 页

**算法 5** 高斯概率路径的分数匹配训练流程

**需要：** 数据集中的样本 \(z \sim p_{\mathrm{data}}\)，分数网络 \(s_t^{\theta}\) 或噪声预测器（noise predictor）\(\epsilon_t^{\theta}\)

**需要：** 调度（schedulers）\(\alpha_t, \beta_t\)，满足 \(\alpha_0 = \beta_1 = 0\)，\(\alpha_1 = \beta_0 = 1\)

1. 对每个数据小批量执行
2. 从数据集中采样一个数据样本 \(z\)。
3. 采样一个随机时间 \(t \sim \mathrm{Unif}_{[0,1]}\)。
4. 采样噪声 \(\epsilon \sim \mathcal{N}(0, I_d)\)。
5. 令 \(x_t = \alpha_t z + \beta_t \epsilon\)。
6. 计算损失

$$
\mathcal{L}(\theta) = \left\lVert s_t^{\theta}(x_t) + \frac{\epsilon}{\beta_t} \right\rVert^2
$$

箭头指向 \(\beta_t\)：\(\beta\) 较小时数值不稳定！

7. 通过在 \(\mathcal{L}(\theta)\) 上做梯度下降更新模型参数 \(\theta\)。
8. 结束循环

## 第 16 页

福克–普朗克方程（Fokker–Planck equation）

随机初始化的随机微分方程

给定：\(X_0 \sim p_{\mathrm{init}}\)，\(\mathrm{d}X_t = u_t(X_t)\, \mathrm{d}t + \sigma_t\, \mathrm{d}W_t\)

二者等价：

跟随概率路径：\(X_t \sim p_t\)（\(0 \le t \le 1\)）。边缘分布为 \(p_t\)。

福克–普朗克方程成立：

$$
\frac{\mathrm{d}}{\mathrm{d}t} p_t(x) = -\mathrm{div}(p_t u_t)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x)
$$

\(-\mathrm{div}(p_t u_t)(x)\)：连续性方程（continuity equation），其中 \(\mathrm{div}\) 为散度（divergence）。\(\frac{\sigma_t^2}{2}\Delta p_t(x)\)：热方程（heat equation）。

## 第 17 页

福克–普朗克方程

$$
\frac{\mathrm{d}}{\mathrm{d}t} p_t(x) = -\mathrm{div}(p_t u_t)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x)
$$

\(-\mathrm{div}(p_t u_t)(x)\)：\(x\) 处概率质量（probability mass）的变化。

\(\frac{\sigma_t^2}{2}\Delta p_t(x)\)：热弥散（dispersion）。

图中标注：实线箭头为 \(u_t(x)\)；虚线箭头为扩散系数（diffusion coefficient）\(\sigma_t\)，从 \(x\) 向外弥散。

## 第 18 页

扩散模型的随机采样

选定噪声水平（noise level）\(\sigma_t\)。借助随机微分方程扩展技巧（SDE extension trick），我们可以从下面的过程采样：

$$
\mathrm{d}X_t = \left[u_t^{\mathrm{target}}(X_t) + \frac{\sigma_t^2}{2} \nabla \log p_t(X_t)\right] \mathrm{d}t + \sigma_t\, \mathrm{d}W_t
$$

对于高斯概率路径，可以只用分数把它写出来：

$$
\mathrm{d}X_t = \left[\left(a_t + \frac{\sigma_t^2}{2}\right) \nabla \log p_t(X_t) + b_t X_t\right] \mathrm{d}t + \sigma_t\, \mathrm{d}W_t
$$

代入分数网络：

$$
\mathrm{d}X_t = \left[\left(a_t + \frac{\sigma_t^2}{2}\right) s_t^{\theta}(X_t) + b_t X_t\right] \mathrm{d}t + \sigma_t\, \mathrm{d}W_t
$$

## 第 19 页

确定性采样（deterministic sampling）与随机采样

左图：确定性采样的粒子轨迹。右图：随机采样的粒子轨迹。

## 第 20 页

用扩散模型做随机（随机微分方程）采样

通过随机微分方程采样，把噪声转换成蛋白质结构。

幻灯片署名：Jason Yim

## 第 21 页

为什么我们想要随机 / 随机微分方程动力学？

理论上：任何扩散系数都会得到同一个结果（从数据分布中采样）。

实践中：

- 训练误差（training error）：神经网络并没有完美学到边缘向量场 / 分数。
- 模拟误差（simulation error）：我们需要仿真随机微分方程 / 常微分方程，从而带来离散化误差（discretization error）。

下游应用：微调（fine-tuning）、推理时优化（inference-time optimization）等，可能需要随机演化。

好消息：常微分方程采样往往能得到最好的结果。因此，随机微分方程采样是一个选项，而不是必须。

## 第 22 页

旁白：朗之万动力学（Langevin dynamics）——分子动力学（molecular dynamics）仿真的基础

分子动力学仿真的是朗之万动力学。这等于边缘向量场为零、且概率路径为常数时的随机微分方程扩展技巧。

$$
\mathrm{d}X_t = \frac{\sigma_t^2}{2} \nabla \log p_t(X_t)\, \mathrm{d}t + \sigma_t\, \mathrm{d}W_t
$$

$$
p_t(x) = p_{\mathrm{Boltzmann}}(x) = \frac{1}{Z} \exp(-U(x))
$$

我们已经证明：

$$
X_0 \sim p_{\mathrm{Boltzmann}} \implies X_t \sim p_{\mathrm{Boltzmann}}
$$

平衡分布（equilibrium distribution）

图：一团在朗之万动力学下演化的粒子。

## 第 23 页

要点：

- 转换公式：对高斯概率路径而言，学习边缘向量场与学习分数函数是等价的。
- 去噪分数匹配：通过近似条件分数函数来学习边缘分数函数的简单办法。
- 用分数模型采样：加入所需的噪声量，并对向量场做修正。

## 第 24 页

第 6 节：无分类器引导（classifier-free guidance, CFG）

目标：理解如何强制生成结果与提示（prompt）保持一致。

## 第 25 页

图像来源：Scaling Rectified Flow Transformers for High-Resolution Image Synthesis [1]

图注：

- 一幅戴珍珠耳环的沼泽食人魔，Johannes Vermeer 所作
- 一辆用蔬菜做成的汽车。
- 宇宙的热寂，线稿

无引导：“生成一幅图像。”

有引导：“生成一幅一只猫在烤蛋糕的图像。”

## 第 26 页

朴素引导采样（vanilla guided sampling）

**算法 7** 引导采样流程

**需要：** 一个已经训练好的引导向量场 \(u_t^{\theta}(x \mid y)\)。

1. 选择一个提示 \(y \in \mathcal{Y}\)，例如“一只猫在烤蛋糕”。
2. 初始化 \(X_0 \sim p_{\mathrm{init}}\)。
3. 从 \(t = 0\) 到 \(t = 1\) 仿真 \(\mathrm{d}X_t = u_t^{\theta}(X_t \mid y)\, \mathrm{d}t\)。

## 第 27 页

朴素引导会得到不够好的结果

图像来源：Classifier-free diffusion guidance [5]。

提示：“柯基犬”（“Corgi dog”）

这些图像与提示吻合得不好，而且本身有错误。

图为四张与提示吻合较差的柯基图像。

## 第 28 页

直观理解：分类器引导（classifier guidance）

图中标注：无提示的向量为 \(u_t^{\mathrm{target}}(x)\)；与提示有关的分量为 \(a_t \nabla_x \log p_t(y \mid x)\)；二者合成为 \(u_t^{\mathrm{target}}(x \mid y)\)。

## 第 29 页

直观理解：分类器引导

图中标注：把与提示有关的部分放大，从 \(a_t \nabla_x \log p_t(y \mid x)\) 放大为 \(w a_t \nabla_x \log p_t(y \mid x)\)。由此得到的斜向向量是提示强化向量场 \(\tilde{u}_t^{w}(x \mid y)\)。

## 第 30 页

无分类器引导

图中标注：竖直分量为 \(u_t^{\mathrm{target}}(x \mid y) - u_t^{\mathrm{target}}(x)\)，水平底边为 \(u_t^{\mathrm{target}}(x)\)，斜边为 \(u_t^{\mathrm{target}}(x \mid y)\)。

## 第 31 页

无分类器引导

图中标注：把差值放大为 \(w\bigl(u_t^{\mathrm{target}}(x \mid y) - u_t^{\mathrm{target}}(x)\bigr)\)。由此得到的斜向向量是提示强化向量场 \(\tilde{u}_t^{w}(x \mid y)\)。

## 第 32 页

无分类器引导的训练：计入空标记（empty token）\(\varnothing\)

**算法 5** 无分类器引导的训练

**需要：** 成对数据集 \((z, y) \sim p_{\mathrm{data}}\)，神经网络 \(u_t^{\theta}\)

1. 对每个数据小批量执行
2. 从数据集中采样一个数据样本 \((z, y)\)。
3. 采样一个随机时间 \(t \sim \mathrm{Unif}_{[0,1]}\)。
4. 采样噪声 \(\epsilon \sim \mathcal{N}(0, I_d)\)。
5. 令 \(x = \alpha_t z + \beta_t \epsilon\)。
6. 以概率 \(p\) 丢掉标签：\(y \leftarrow \varnothing\)。以某个概率丢掉标签！
7. 计算损失

$$
\mathcal{L}(\theta) = \lVert u_t^{\theta}(x \mid y) - u_t^{\mathrm{target}}(x \mid z) \rVert^2
$$

8. 通过在 \(\mathcal{L}(\theta)\) 上做梯度下降更新模型参数 \(\theta\)。
9. 结束循环

## 第 33 页

用无分类器引导做采样，流程与以前相同，只是改用加权向量场：

$$
u_t^{\theta, w}(x) = (1 - w)\, u_t^{\theta}(x \mid \varnothing) + w\, u_t^{\theta}(x \mid y)
$$

**算法 8** 无分类器引导采样流程

**需要：** 一个已经训练好的引导向量场 \(u_t^{\theta}(x \mid y)\)。

1. 选择一个提示 \(y \in \mathcal{Y}\)，或者取 \(y = \varnothing\) 做无引导采样。
2. 选择引导尺度（guidance scale）\(w > 1\)。
3. 初始化 \(X_0 \sim p_{\mathrm{init}}\)。
4. 从 \(t = 0\) 到 \(t = 1\) 仿真

$$
\mathrm{d}X_t = \bigl[(1 - w)\, u_t^{\theta}(X_t \mid \varnothing) + w\, u_t^{\theta}(X_t \mid y)\bigr] \mathrm{d}t.
$$

## 第 34 页

例子：无分类器引导

图像来源：Classifier-free diffusion guidance [5]。

左：\(w = 1.0\)。右：\(w = 4.0\)。

## 第 35 页

例子：无分类器引导

图像来源：Classifier-free diffusion guidance [5]。

图为若干类别（猫、柯基、火山）的生成样例网格。

## 第 36 页

你看到的几乎所有图像或视频都使用了无分类器引导！

- 无分类器引导是关键：没有它，几乎什么都行不通。

图为三幅生成图像。

例子 — Stable Diffusion 3：无分类器引导尺度 \(w \approx 4.0\)

## 第 37 页

无分类器引导不再建模数据分布！

- 无分类器引导是启发式方法（heuristic）：我们不再建模数据分布。事实上，我们走到了它外面。它主要靠很好的经验结果来辩护。

图中标注：左侧为数据分布；箭头方向为引导强度增大。分布中的若干模式随着引导变强而彼此分开、各自变尖。

图像来源：Classifier-free diffusion guidance [5]。

## 第 38 页

第 5 节：扩散文献导读

目标：理解扩散模型的不同表述。

## 第 39 页

时间约定：

“流时间约定（flow time convention）”：

- 数据：\(t = 1\)
- 噪声：\(t = 0\)

对应：流匹配、整流流（rectified flows）、随机插值（stochastic interpolants）

“扩散时间约定（diffusion time convention）”：

- 数据：\(t = 0\)
- 噪声：\(t \to \infty\)

对应：带随机微分方程的基于分数的扩散模型（score-based diffusion models）

“离散时间（discrete time）”：

- 用离散时间步，而不是连续时间步
- 没有常微分方程或随机微分方程，而是马尔可夫链（Markov chain）
- DDIM \(\approx\) 概率流常微分方程（probability flow ODE）

对应：DDPM、DDIM

## 第 40 页

加噪过程（noising procedure）

概率路径（此处）：

$$
p_t(x \mid z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)
$$

对应：流匹配、整流流

插值函数（interpolant function）：

$$
I_t(\epsilon, z) = \alpha_t \epsilon + \beta_t z
$$

对应：随机插值

“前向”扩散过程（forward diffusion process）：

$$
\mathrm{d}X_t = a_t(X_t)\, \mathrm{d}t + \sigma_t\, \mathrm{d}W_t
$$

对应：去噪扩散模型

注：对于高斯概率路径，上面这些做法是等价的。

## 第 41 页

用前向加噪过程构造加噪流程

前向随机微分方程（数据 \(\to\) 噪声），从 \(x(0)\) 到 \(x(T)\)：

$$
\mathrm{d}x = f(x, t)\, \mathrm{d}t + g(t)\, \mathrm{d}w
$$

反向随机微分方程（噪声 \(\to\) 数据），从 \(x(T)\) 回到 \(x(0)\)：

$$
\mathrm{d}x = \left[f(x, t) - g^2(t)\, \nabla_x \log p_t(x)\right] \mathrm{d}t + g(t)\, \mathrm{d}\bar{w}
$$

其中 \(\nabla_x \log p_t(x)\) 标为分数函数。中间一排图是从清晰的狗逐渐变成噪声。

时间反转的随机微分方程（time-reversed SDE）是我们讨论过的随机微分方程扩展技巧在某一特定噪声水平下的一个特解。经验上，它往往并不是实践中最好的解。

## 第 42 页

此处：流匹配

- 可以说是最简单的流与扩散算法
- 允许你只使用流
- 允许你从任意 \(p_{\mathrm{init}}\) 走到任意 \(p_{\mathrm{data}}\)

注：这里给出的方法可以把任意分布转换成任意分布。

## 第 43 页

连接任意分布：例子

没有音频的视频 \(\to\) 有音频的视频

低分辨率图像 \(\to\) 高分辨率图像

未受扰动的细胞 \(\to\) 受扰动的细胞

等等。

图注：Michael Albergo

图为若干照片，其中一部分区域被噪声块替换。

## 第 44 页

课程概览

- 讲座 1 — 生成即采样。流模型与扩散模型
- 讲座 2 — **流匹配**：训练算法。
- 讲座 3 — **分数匹配、引导**：如何以提示为条件。
- 讲座 4 — **构建图像生成器**：网络架构（network architectures）+ 潜空间（latent spaces）
- 讲座 5 — **进阶专题**：离散扩散模型（discrete diffusion models）+ 蒸馏模型（distilled models）

## 第 45 页

下一次课：

周一，上午 11:00–12:30

神经网络架构 + 潜空间！

E25-111（同一教室）

办公时间：今天 15:00–16:30，地点 36-156

## 第 46 页

参考文献

1. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis, https://arxiv.org/abs/2403.03206
2. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, https://arxiv.org/abs/2010.11929
3. Scalable Diffusion Models with Transformers, https://arxiv.org/abs/2212.09748
4. High Resolution Image Synthesis with Latent Diffusion Models, https://arxiv.org/abs/2112.10752
5. Classifier-Free Diffusion Guidance, https://arxiv.org/abs/2207.12598

---

<!-- 来源文件：lec4.zh.md -->

# 讲座 4：潜空间与神经网络架构

## 第 1 页

**讲座 04：潜空间与神经网络架构**

MIT IAP 2026｜2025 年 1 月 28 日

Peter Holderrieth 与 Ron Shprints

赞助人：Tommi Jaakkola

## 第 2 页

### 课程概览

- 讲座 1：流模型与扩散模型。
- 讲座 2：流匹配——训练算法。
- 讲座 3：分数匹配与引导——如何以提示为条件。
- 讲座 4：构建图像生成器——潜空间与网络架构。
- 讲座 5：进阶专题——离散扩散模型与蒸馏模型。

## 第 3 页

## 第 6 节：潜空间

**目标：** 理解如何把数据转换到更容易建模的潜变量中。

## 第 4 页

（本页无可提取文字。）

## 第 5 页

### 为什么需要潜空间？

- 一幅高分辨率图像有 3 个颜色通道，高度为 600、宽度为 1000，因此维度为

$$
3\times 600\times 1000=1{,}800{,}000.
$$

- 这是一个非常高维的空间！

## 第 6 页

### 为什么需要潜空间？

高分辨率图像的维度为 \(3\times600\times1000=1.8\) 百万。这样会带来几个问题：

- GPU 内存占用急剧增长。
- 学习问题非常困难。
- 存在冗余：相邻像素高度相关。

## 第 7 页

（本页无可提取文字。）

## 第 8 页

### 为什么需要潜空间？

高分辨率图像的维度为 \(3\times600\times1000=1.8\) 百万。

问题包括：

- GPU 内存占用急剧增长。
- 学习问题非常困难。
- 相邻像素高度相关，存在大量冗余。

## 第 9 页

### 为什么这对扩散模型是问题，而对监督学习不是？

因为扩散模型学习的是一个向量场，其输出本身是高维向量，而且在模拟 ODE 时要反复应用这个向量场：

$$
u_t^\theta(x)\in\mathbb{R}^{d},
$$

并且需要多次计算 \(u_t^\theta\)。

## 第 10 页

### 用自编码器构造潜空间

**自编码器 = 编码器 + 解码器**

数据空间中的对象先经过编码器映射到潜空间，再由解码器映射回数据空间：

$$
x\in\mathbb{R}^{d}\xrightarrow{\text{Encoder}}z\in\mathbb{R}^{k}\xrightarrow{\text{Decoder}}\hat{x}\in\mathbb{R}^{d}.
$$

## 第 11 页

### 用自编码器构造潜空间

自编码器的核心作用是**压缩**：

$$
d\gg k,
$$

也就是把高维数据 \(x\) 表示为维度更低的潜变量 \(z\)，再从 \(z\) 重构数据。

## 第 12 页

### 标准自编码器的问题：一个“糟糕”的潜空间

我们的目标是把数据转换为潜变量，然后学习潜变量的分布。

问题是：把数据分布变换到潜空间后，分布会发生什么？

答案是：我们并不知道。重构损失并没有约束潜变量分布的形状。

因此，标准自编码器可能确实完成了压缩，却把潜分布变成了很难学习、很难采样的分布。我们的目标是构造一个能产生“良好”潜分布的自编码器。

## 第 13 页

### KL 散度

对于两个概率密度 \(q,p\)，Kullback–Leibler 散度（KL 散度）定义为

$$
D_{\mathrm{KL}}(q(x)\Vert p(x))
=\int q(x)\log\frac{q(x)}{p(x)}\,\mathrm{d}x
=\mathbb{E}_{X\sim q}\left[\log\frac{q(X)}{p(X)}\right].
$$

KL 散度衡量两个分布有多不同，并满足

$$
D_{\mathrm{KL}}(q\Vert p)\geq 0,
\qquad
D_{\mathrm{KL}}(q\Vert p)=0\iff q=p.
$$

## 第 14 页

### 两个正态分布之间的 KL 散度

设

$$
q(x)=\mathcal{N}(x;\mu_q,\sigma_q^2 I_d),
\qquad
p(x)=\mathcal{N}(x;\mu_p,\sigma_p^2 I_d).
$$

则

$$
D_{\mathrm{KL}}(q\Vert p)
=\frac{1}{2}\left(
 d\,\mathcal{K}\!\left(\frac{\sigma_q^2}{\sigma_p^2}\right)
+\frac{\lVert\mu_q-\mu_p\rVert^2}{\sigma_p^2}
\right),
$$

其中

$$
\mathcal{K}(\alpha)=\alpha-\log\alpha-1.
$$

第二项衡量均值的差异，第一项衡量方差的差异。

## 第 15 页

### 绘图

函数

$$
\mathcal{K}(\alpha)=\alpha-\log\alpha-1
$$

在 \(\alpha=1\) 处取得最小值。这说明当两个分布的方差相同时，方差项对 KL 散度的贡献最小。

## 第 16 页

### 两个正态分布之间的 KL 散度

再次设

$$
q(x)=\mathcal{N}(x;\mu_q,\sigma_q^2 I_d),
\qquad
p(x)=\mathcal{N}(x;\mu_p,\sigma_p^2 I_d).
$$

公式中的两项分别对应：

- 方差的距离：\(d\,\mathcal{K}(\sigma_q^2/\sigma_p^2)\)。
- 均值的距离：\(\lVert\mu_q-\mu_p\rVert^2/\sigma_p^2\)。

## 第 17 页

### VAE 训练算法

变分自编码器（VAE）同时优化重构质量和潜分布的正则化：

$$
\mathcal{L}_{\mathrm{VAE}}
=\mathcal{L}_{\mathrm{Recon}}+\beta\mathcal{L}_{\mathrm{KL}}.
$$

重构项要求解码结果接近原始数据，KL 项要求编码后的分布接近简单的先验分布（通常是 \(\mathcal{N}(0,I)\)）。

## 第 18 页

### 潜扩散模型（LDM）的流程

1. **数据：** 取全部训练数据 \(x_1,\ldots,x_N\)，例如互联网上的图像。
2. **编码：** 把每幅图像编码到潜空间；对于 VAE，通常取编码器的均值预测。
3. **潜数据：** 得到潜变量数据集 \(z_1,\ldots,z_N\)，其规模显著小于高分辨率图像。
4. **潜扩散模型：** 在潜变量数据集上训练扩散模型，也就是让扩散模型生成潜向量。
5. **解码：** 从扩散模型采样后，把生成的潜变量映射回数据空间。

最后返回解码后的图像作为样本。整个流程与之前相同，只是把数据集替换成了变换后的潜数据集。

## 第 19 页

### 几乎所有 AI 生成的图像或视频都在潜空间中生成

潜空间非常关键：如果直接在像素空间中建模，内存消耗会迅速失控。潜空间让模型可以集中建模重要结构。

示例：

- Stable Diffusion：\([3,256,256]\rightarrow[4,32,32]\)。
- FLUX 2.0：\([3,1024,1024]\rightarrow[32,64,64]\)。

## 第 20 页

## 第 7 节：神经网络架构

**目标：** 理解如何为扩散模型构建神经网络。

## 第 21 页

### 用神经网络参数化向量场

神经网络接收三个输入：

- 时间 \(t\)。
- 潜图像或当前状态 \(x\)。
- 提示 \(y\)。

输出带参数的向量场

$$
u_t^\theta(x\mid y).
$$

## 第 22 页

### 编码时间

时间 \(t\) 只有一个维度，而其他变量通常是高维的。为了让时间信息在网络中占据足够作用，通常把它嵌入为高维的正弦特征：

$$
\mathrm{TimeEmb}(t)=\sqrt{\frac{2}{d}}
\begin{bmatrix}
\cos(2\pi w_1t)&\cdots&\cos(2\pi w_{d/2}t)&
\sin(2\pi w_1t)&\cdots&\sin(2\pi w_{d/2}t)
\end{bmatrix}^{\!T}.
$$

频率按几何级数选择：

$$
w_i=w_{\min}\left(\frac{w_{\max}}{w_{\min}}\right)^{\frac{i-1}{d/2-1}},
\qquad i=1,\ldots,d/2.
$$

具体的频率范围并不重要；关键是得到一个 \(d\) 维、归一化的时间嵌入，例如

$$
\left\lVert\mathrm{TimeEmb}(t)\right\rVert=1.
$$

注：幻灯片把前因子写成 \(1/\sqrt d\)，这与课程讲义中的归一化约定存在差异；这里采用课程讲义中与 \(d\) 维嵌入一致的 \(\sqrt{2/d}\) 形式。若要逐字对应幻灯片，可将前因子写成 \(1/\sqrt d\)。

## 第 23 页

### 编码语言提示

示例提示：

> “一只狗在意大利城市的公园草地上、阳光下奔跑。”

大多数模型依赖预训练语言嵌入来编码文本，例如：

- CLIP（Contrastive Language–Image Pre-training，对比语言–图像预训练）嵌入。
- T5 等其他预训练模型的嵌入。
- 大语言模型的嵌入。

文本编码器可以输出一个长度为 \(S\) 的向量序列：

$$
\mathrm{PromptEmb}(y_{\mathrm{raw}})\in\mathbb{R}^{S\times k}.
$$

## 第 24 页

### Patchify：把图像变成向量序列

给定图像

$$
x\in\mathbb{R}^{3\times H\times W},
$$

把图像切成大小为 \(P\times P\) 的图块，再把图块展平，得到

$$
\tilde{x}=\mathrm{Patchify}(x)\in\mathbb{R}^{L\times k},
\qquad
L=\frac{H}{P}\frac{W}{P}.
$$

其中 \(L\) 是序列长度；每个图块被表示为一个向量。图源：<https://gowrishankar.info/blog/transformers-everywhere-patch-encoding-technique-for-vision-transformersvit-explained/>。

## 第 25 页

### 扩散 Transformer（DiT）

DiT 是用于扩散模型的 Transformer。它针对扩散模型所需的输入和条件形式，对标准 Transformer 做了适配。

参考：Peebles 与 Xie，2023，《Scalable Diffusion Models with Transformers》，发表于 IEEE/CVF International Conference on Computer Vision，第 4195–4205 页。

## 第 26 页

### 扩散 Transformer 概览

典型流程为：

1. 输入时间嵌入、文本嵌入和图像图块。
2. 通过 Transformer 注意力层循环处理图块序列。
3. 对输出执行反图块化（unpatchify），恢复为图像形状。

## 第 27 页

### 背景：缩放点积注意力

注意力通过矩阵乘法和 softmax，把一个向量序列映射为另一个向量序列。它有三个输入：键（keys）、值（values）和查询（queries）。

给定

$$
Q\in\mathbb{R}^{N\times d_h},\qquad
K,V\in\mathbb{R}^{M\times d_h},
$$

缩放点积注意力为

$$
\mathrm{Attn}(Q,K,V)
=\mathrm{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_h}}\right)V
\in\mathbb{R}^{N\times d_h},
$$

其中 softmax 按行计算。

## 第 28 页

### DiTBlock 概览

- **图像的自注意力：** 处理图像自身。\(Q=\) 图像，\(K=\) 图像，\(V=\) 图像。
- **文本的交叉注意力：** 让图像关注文本。\(Q=\) 图像，\(K=V=\) 文本嵌入。
- **时间的自适应层归一化：** 归一化的缩放和偏移由时间变量决定。

这些输入通过相加组合起来。理解这类变换的最好方法是亲自实现；详见实验 03 和课程讲义。

## 第 29 页

## 加餐：大规模扩散模型

**目标：** 理解扩散模型如何进行大规模训练。

## 第 30 页

### 案例研究：Stable Diffusion 3

概览：

- 使用“直线”调度器（CondOT 路径）的流匹配模型。
- 无分类器引导，权重为 \(2.0\)–\(5.0\)。
- 在潜空间中进行流匹配，使用预训练 VAE。
- 模型参数量：80 亿。
- 模拟/采样步数：50。
- 数据集：LAION。

图源：*Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* [1]。

## 第 31 页

### Stable Diffusion 3 的神经网络架构

- 通过交叉注意力，以 CLIP 的粗粒度文本嵌入和 T5-XXL 的序列级文本嵌入作为条件。
- MM-DiT 架构把 DiT 从类别条件扩展到文本条件，并在整个网络中通过交叉注意力处理文本与图像。

图源：*Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* [1]。

## 第 32 页

### 案例研究：Meta MovieGen

## 第 33 页

### 案例研究：Meta MovieGen

- 使用“直线”调度器（CondOT 路径）的流匹配模型。
- 无分类器引导。
- 在潜空间中进行流匹配，使用预训练 VAE。对视频来说，由于增加了时间维度，这一点尤其关键。
- 神经网络架构：把 DiT 适配到视频上（具体改变是什么？）。
- 模型参数量：300 亿。
- 使用 6,144 张 H100 GPU。

图源：*Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* [1]。

## 第 34 页

## 加餐：扩散模型文献指南

**目标：** 理解扩散模型的不同解释方式。

## 第 35 页

### 时间约定

**“流时间”约定：**

- 数据：\(t=1\)。
- 噪声：\(t=0\)。
- 常见于流匹配、整流流和随机插值。

**“扩散时间”约定：**

- 数据：\(t=0\)。
- 噪声：\(t\to\infty\)。
- 常见于基于 SDE 分数的扩散模型。

**“离散时间”：**

- 使用离散时间步，而不是连续时间步。
- 不使用 ODE 或 SDE，而使用马尔可夫链。
- DDIM 近似于概率流 ODE。

典型模型包括 DDPM 与 DDIM。

## 第 36 页

### 加噪过程：如何破坏数据？

对于高斯概率路径，以下三种写法是等价的：

**概率路径：**

$$
p_t(x\mid z)=\mathcal{N}(x;\alpha_tz,\beta_t^2I_d).
$$

**插值函数：**

$$
I_t(\epsilon,z)=\alpha_t\epsilon+\beta_tz.
$$

**“前向”扩散过程：**

$$
\mathrm{d}X_t=a_t(X_t)\,\mathrm{d}t+\sigma_t\,\mathrm{d}W_t.
$$

它们分别对应流匹配/整流流、随机插值，以及去噪扩散模型。

## 第 37 页

### 通过前向加噪过程构造加噪过程

前向 SDE 把数据 \(x(0)\) 逐渐变成噪声 \(x(T)\)：

$$
\mathrm{d}x=f(x,t)\,\mathrm{d}t+g(t)\,\mathrm{d}w.
$$

对应的时间反演 SDE 把噪声变回数据：

$$
\mathrm{d}x=
\left[f(x,t)-g^2(t)\nabla_x\log p_t(x)\right]\mathrm{d}t
+g(t)\,\mathrm{d}\bar w.
$$

其中 \(\nabla_x\log p_t(x)\) 是分数函数。时间反演 SDE 是针对特定噪声水平的 SDE 扩展技巧的一种具体解；在实践中，它往往不是最优解。

## 第 38 页

### 这里采用流匹配

- 可以说是最简单的流模型与扩散模型算法之一。
- 可以把自己限制在流模型上。
- 可以把任意初始分布 \(p_{\mathrm{init}}\) 转换为任意数据分布 \(p_{\mathrm{data}}\)。

**注意：** 这里介绍的方法可以把任意分布转换为任意其他分布。

## 第 39 页

### 连接任意分布：例子

流匹配可以连接许多不同的数据分布，例如：

- 没有音频的视频 \(\rightarrow\) 带音频的视频。
- 低分辨率图像 \(\rightarrow\) 高分辨率图像。
- 未受扰动的细胞 \(\rightarrow\) 受扰动的细胞。
- 等等。

图示致谢：Michael Albergo。

## 第 40 页

### 下一节课

周五 12:30–14:00

**离散扩散模型**

E25-111（同一教室）。

---

<!-- 来源文件：lec5.zh.md -->

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
