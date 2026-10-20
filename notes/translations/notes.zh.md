# 流模型与扩散模型课程讲义中文翻译

本文件按课程讲义顺序合并自同目录下的分节翻译：第 1–7 节及附录。

---

<!-- 来源文件：01-02.zh.md -->

## 1. 引言

> 从数据（data）制造噪声（noise）很容易；从噪声制造数据，便是生成式建模（generative modeling）。
>
> ——Song 等人（yangsong\_sde）

### 1.1 概述

近年来，我们都目睹了人工智能（artificial intelligence，AI）的一场巨大变革。图像生成器，例如 *Nano Banana* 或 *Stable Diffusion 3*，能够在多种多样的风格中生成照片级真实（photorealistic）的图像与艺术图像；视频模型，例如 Meta 的 *VEO-3*，能够生成高度逼真的电影片段；大语言模型（large language models），例如 *ChatGPT*，能够针对文本提示生成看似达到人类水平的回答。这场变革的核心，在于人工智能系统的一种新能力：生成（generate）对象的能力。前几代人工智能系统主要用于预测（prediction），而这些新的人工智能系统是有创造性的：它们会做梦，或者根据用户指定的输入构想出新的对象。这样的生成式人工智能（generative AI）系统，正处于这场近期人工智能变革的核心。

本课程的目标，是向你讲授两种使用最为广泛的生成式人工智能算法：去噪扩散模型（denoising diffusion models）（song2020score）与流匹配（flow matching）（lipman2022flow；liu2022flow；albergo2023stochastic；lipman2024flow）。这些模型是目前最好的图像、音频与视频生成模型（例如 *Nano Banana*、*FLUX* 或 *VEO-3*）的骨干，并且最近已成为蛋白质结构等科学应用中的最先进方法（例如 *AlphaFold3* 就是一个扩散模型）。毫无疑问，理解这些模型确实是一项极其有用的技能。

所有这些生成模型都通过迭代地把噪声转化为数据来生成对象。从噪声到数据的这一演化，是通过模拟常微分方程或随机微分方程（ordinary differential equations / stochastic differential equations，ODE/SDE）来实现的。流匹配与去噪扩散模型是一族技术，使我们能够借助深度神经网络（deep neural networks），在大规模上构造、训练并模拟这样的常微分方程与随机微分方程。这些模型实现起来相当简单，但随机微分方程的技术性会使这些模型难以理解。在本课程中，我们针对微分方程所需的数学工具箱给出一份自成体系的介绍，使你能够系统地理解这些模型。随后，我们逐步讲解最先进图像与视频生成器的现代技术栈。除了适用面广泛，我们相信流模型（flow model）与扩散模型背后的理论本身也是优美的。因此，最重要的是，我们希望这门课能给你带来许多乐趣。

**注记（补充资源）.** 虽然本讲义是自成体系的，我们仍鼓励你使用另外两份材料：

1. **课程录像：** 它们以授课的形式引导你学习每一节。
2. **实验：** 它们引导你从零开始实现自己的扩散模型。我们强烈建议你“亲手上手”去写代码。

你可以在我们的课程网站上找到它们：https://diffusion.csail.mit.edu/。

### 1.2 课程结构

我们对本讲义作一个简要概述。

- **第 1 节，生成式建模即采样（sampling）：** 我们把“生成”一幅图像、一段视频、一个蛋白质等等意味着什么加以形式化。我们将把例如“如何生成一幅狗的图像？”这样的问题，转化为从某个概率分布（probability distribution）中采样这一更精确的问题。
- **第 2 节，流模型与扩散模型：** 我们解释生成的机制。正如你可以从本课程的名称猜到的那样，这一机制就是模拟常微分方程与随机微分方程。我们介绍微分方程，并解释如何用它们来构造生成模型。
- **第 3 节，流匹配：** 接下来，我们解释并推导流匹配。这是一个简单且可扩展的算法，位于前面提到的所有大规模生成模型的核心，例如 Stable Diffusion、*Nano Banana* 或 SORA。
- **第 4 节，得分匹配（score matching）：** 我们研究得分函数（score functions），以及如何通过得分匹配来学习它们。这不仅是扩散模型的训练算法，它还使得随机微分方程采样与引导（guidance）成为可能。
- **第 5 节，引导：** 我们学习如何让样本以一个提示为条件（conditioning）（例如「一幅猫的图像」），以及如何通过无分类器引导（classifier-free guidance）来强制模型遵守这样的提示。
- **第 6 节，潜空间（latent space）与神经网络架构：** 我们讨论如何构建大规模的图像与视频生成器，例如 *Nano Banana*。这包括常见的神经网络架构，以及如何在潜空间中进行构建。我们还会综述最先进的模型。
- **第 7 节（可选），离散扩散模型（discrete diffusion models）：** 我们学习如何把扩散模型的原理从欧氏空间（Euclidean space）迁移到语言这类离散数据上。这使得我们可以运用扩散模型的原理来构造大语言模型。

#### 所需背景

由于这一主题具有技术性，我们建议你具备一定基础的数学成熟度，尤其是对概率论（probability theory）有所了解。因此，我们在「概率论回顾」附录中放了一节简短的复习。如果那里的某些概念你还不熟悉，不必担心。

### 1.3 生成式建模即采样

让我们先来思考可能遇到的各种数据类型，或者说数据模态（data modalities），以及我们将如何用数值来表示它们：

1. **图像：** 考虑具有 $H \times W$ 个像素的图像，其中 $H$ 表示图像的高度，$W$ 表示宽度，每个像素有三个颜色通道（RGB）。对每一个像素和每一个颜色通道，我们得到一个属于 $\mathbb{R}$ 的强度值。因此，一幅图像可以用一个元素 $z \in \mathbb{R}^{H \times W \times 3}$ 来表示。
2. **视频：** 视频就是时间上的一连串图像。如果有 $T$ 个时间点，或者说帧（frames），那么一段视频就可以用一个元素 $z \in \mathbb{R}^{T \times H \times W \times 3}$ 来表示。
3. **分子结构：** 一种朴素的做法是用矩阵 $z = (z^{1}, \ldots, z^{N}) \in \mathbb{R}^{3 \times N}$ 来表示分子的结构，其中 $N$ 是分子中的原子数目，每个 $z^{i} \in \mathbb{R}^{3}$ 描述该原子的位置。当然，还有其他更精细的方式来表示这样的分子。

在上面所有例子中，我们想要生成的对象都可以在数学上表示为一个向量（在可能先做展平（flattening）之后）。因此，在整个文档中，我们有：

**关键想法（对象即向量）.** 我们将被生成的对象视为向量 $z \in \mathbb{R}^{d}$。

上述说法的一个显著例外是文本数据。语言模型（例如 *ChatGPT*）通常把它建模为离散对象。连续数据 $z \in \mathbb{R}^{d}$ 是我们的主要焦点，但我们也会在第 7 节中研究文本生成。

#### 生成即采样

让我们定义“生成”某个东西意味着什么。例如，假设我们想生成一幅狗的图像。自然会有许多幅让我们满意的狗的图像。特别地，并不存在单独一幅“最好”的狗的图像。相反，存在一个谱：有些图像更契合，有些更不契合。在机器学习（machine learning）中，通常把这种可能图像的多样性实现为图像空间上的一个概率分布。我们称这样的分布为数据分布（data distribution），并记作 $p_{\mathrm{data}}$。在数学上，可以把 $p_{\mathrm{data}}$ 看成一个概率密度（probability density），即一个函数 $p_{\mathrm{data}} : \mathbb{R}^{d} \to \mathbb{R}_{\geq 0}$，它给每一个可能的对象 $z \in \mathbb{R}^{d}$ 指定一个似然（likelihood）$p_{\mathrm{data}}(z) \geq 0$。在狗的图像这个例子中，该分布因此会给看起来更像狗的图像 $z$ 更高的似然 $p_{\mathrm{data}}(z)$。于是，一幅图像、一段视频或一个分子“拟合得好不好”——这是一个相当主观的说法——被替换成它在数据分布 $p_{\mathrm{data}}$ 下有多“可能”。由此，我们可以把生成任务在数学上表达为从（未知的）分布 $p_{\mathrm{data}}$ 中采样：

**关键想法（生成即采样）.** 生成一个对象 $z$，被建模为从数据分布中采样 $z \sim p_{\mathrm{data}}$。

生成模型（generative model）是一个机器学习模型，它使我们能够从 $p_{\mathrm{data}}$ 生成样本。在机器学习中，我们需要数据来训练模型。在生成式建模中，我们通常假设能够获得有限个从 $p_{\mathrm{data}}$ 独立抽取的例子，它们合在一起充当真实分布的代理。

**关键想法（数据集）.** 数据集（dataset）由有限个样本 $z_{1}, \ldots, z_{N} \sim p_{\mathrm{data}}$ 构成。

对于图像，我们可以通过汇集互联网上公开可得的图像来构造数据集。对于视频，我们同样可以考虑使用 YouTube。对于蛋白质结构，诸如 RCSB 蛋白质数据库（Protein Data Bank，PDB）之类的来源提供了数十万个由实验解析的结构。随着数据集的规模变得非常大，它对底层分布 $p_{\mathrm{data}}$ 的代表也就越来越好。

#### 引导生成与条件生成

在许多情形中，我们希望生成一个以某些数据 $y$ 为条件的对象。例如，我们可能希望生成一幅图像，其条件为 $y =$「一只狗跑下覆盖着雪的小山，背景是群山」。我们可以把这重新表述为从条件分布（conditional distribution）中采样：

**关键想法（引导生成）.** 引导生成就是从 $z \sim p_{\mathrm{data}}(\cdot \mid y)$ 中采样，其中 $y$ 是条件变量。

我们称 $p_{\mathrm{data}}(\cdot \mid y)$ 为引导数据分布（guided data distribution）。引导式的生成建模任务通常涉及学习以任意的、而不是固定的 $y$ 为条件。沿用前面的例子，我们也许转而希望以另一段文本提示为条件，例如 $y =$「一幅猫吹灭生日蜡烛的照片级真实图像」。因此，我们寻求单一的一个模型，它可以以 $y$ 的任何这样一种选择为条件。事实表明，用于无条件（unconditional）生成的技术可以很容易地推广到条件情形。因此，在前 3 节中，我们将几乎完全专注于无条件情形（同时记住，条件生成（conditional generation）才是我们正在走向的目标）。

#### 生成模型

抽象地说，生成模型是一个返回 $z \sim p_{\mathrm{data}}$ 的样本（或至少近似如此）的算法。如果 $p_{\mathrm{data}}$ 是狗的图像的分布，这个算法就会返回随机的狗的图像。在本课程中，我们将专注于使用流模型或扩散模型来构造生成模型这一特定方式，因为它们代表当前的最先进水平。不过，重要的是要记住，人们已经发展出许多其他的生成模型（未来也许还会发现更多）。

**小结（生成即采样）.** 我们总结本节的结论：

1. 在本讲义中，我们主要考虑生成表示为向量 $z \in \mathbb{R}^{d}$ 的对象这一任务，例如图像、视频和分子结构。
2. 生成是这样一项任务：从概率分布 $p_{\mathrm{data}}$ 生成样本，并且在训练期间能够使用样本数据集 $z_{1}, \ldots, z_{N} \sim p_{\mathrm{data}}$。
3. 引导生成假定我们把分布条件于标签 $y$ 上，并且想要从 $p_{\mathrm{data}}(\cdot \mid y)$ 中采样，同时在训练期间能够使用样本对 $(z_{1}, y), \ldots, (z_{N}, y)$ 组成的数据集。
4. 我们的目标是构造一个生成模型，也就是一个在训练之后从 $p_{\mathrm{data}}$ 返回样本的模型。

## 2. 流模型与扩散模型

在上一节中，我们把生成式建模形式化为从数据分布 $p_{\mathrm{data}}$ 中采样。进一步，我们把目标形式化了：构造一个生成模型，也就是一个返回样本 $z \sim p_{\mathrm{data}}$ 的算法。在本节中，我们描述如何把生成模型建成为对一个适当构造的微分方程的模拟。例如，流匹配与扩散模型分别涉及模拟常微分方程与随机微分方程。因此，本节的目标是定义并构造这些生成模型，因为它们将在讲义的其余部分中使用。具体而言，我们首先定义常微分方程与随机微分方程，并讨论对它们的模拟。其次，我们描述如何用深度神经网络来参数化一个常微分方程或随机微分方程。这导向流模型与扩散模型的定义，以及从这类模型中采样的基本算法。在后面的几节中，我们再探讨如何训练这些模型。

### 2.1 流模型

我们从定义常微分方程开始。常微分方程的解由一条轨迹（trajectory）定义，也就是一个形如

$$
X : [0,1] \to \mathbb{R}^{d}, \quad t \mapsto X_{t}
$$

的函数，它把时间 $t$ 映到空间 $\mathbb{R}^{d}$ 中的某个位置。每一个常微分方程都由一个向量场（vector field）$u$ 定义，也就是一个形如

$$
u : \mathbb{R}^{d} \times [0,1] \to \mathbb{R}^{d}, \quad (x, t) \mapsto u_{t}(x)
$$

的函数。也就是说，对每一个时刻 $t$ 和每一个位置 $x$，我们得到一个向量 $u_{t}(x) \in \mathbb{R}^{d}$，它指定空间中的一个速度（见下文的流示意图）。常微分方程对轨迹施加一个条件：我们想要一条从点 $x_{0}$ 出发、“沿着向量场 $u_{t}$ 的那些线往前走”的轨迹 $X$。我们可以把这样一条轨迹形式化为如下方程的解：

$$
\frac{\mathrm{d}}{\mathrm{d} t} X_{t} = u_{t}(X_{t}) \qquad \blacktriangleright\ 
$$

$$
X_{0} = x_{0} \qquad \blacktriangleright\ 
$$

常微分方程要求 $X_{t}$ 的导数由 $u_{t}$ 给出的方向来指定。初值条件（initial conditions）要求我们在时刻 $t = 0$ 从 $x_{0}$ 出发。现在可以问：如果在 $t = 0$ 时从 $X_{0} = x_{0}$ 出发，那么在时刻 $t$ 我们在哪里（$X_{t}$ 是什么）？回答这个问题的函数称为流（flow），它是如下常微分方程的解：

$$
\psi : \mathbb{R}^{d} \times [0,1] \to \mathbb{R}^{d}, \quad (x_{0}, t) \mapsto \psi_{t}(x_{0})
$$

$$
\frac{\mathrm{d}}{\mathrm{d} t} \psi_{t}(x_{0}) = u_{t}(\psi_{t}(x_{0})) \qquad \blacktriangleright\ 
$$

$$
\psi_{0}(x_{0}) = x_{0} \qquad \blacktriangleright\ 
$$

对给定的初值 $X_{0} = x_{0}$，常微分方程的一条轨迹可以通过 $X_{t} = \psi_{t}(X_{0})$ 恢复出来。因此，向量场、常微分方程与流，直观上是同一个对象的三种描述：**向量场定义常微分方程，而其解就是流**。和每一个方程一样，面对常微分方程我们也应当问自己：解是否存在？如果存在，它是否唯一？数学中的一个基本结果对两者的回答都是“是！”，只要我们对 $u_{t}$ 施加较弱的假定：

**定理（流的存在性与唯一性）.** 若 $u : \mathbb{R}^{d} \times [0,1] \to \mathbb{R}^{d}$ 连续可微（continuously differentiable）且导数有界，则流方程有由某个流 $\psi_{t}$ 给出的唯一解。此时，对所有 $t$，$\psi_{t}$ 都是微分同胚（diffeomorphism），即 $\psi_{t}$ 连续可微，并且具有连续可微的逆 $\psi_{t}^{-1}$。

请注意，流的存在性与唯一性所需要的假定在机器学习中几乎总是成立的，因为我们用神经网络来参数化 $u_{t}(x)$，而它们的导数总是有界的。因此，流的存在性与唯一性定理不应当让你担心，反而是一个好消息：**在我们关心的情形中，流存在，并且是常微分方程的唯一解。** 证明可见 perko2013differential 与 coddington1956theory。

**图注.** 流 $\psi_{t} : \mathbb{R}^{d} \to \mathbb{R}^{d}$（红色方格）由速度场（velocity field）$u_{t} : \mathbb{R}^{d} \to \mathbb{R}^{d}$（用蓝色箭头表示）定义，该速度场规定了它在所有位置处的瞬时运动（这里 $d = 2$）。我们展示三个不同的时刻 $t$。可以看到，流是一个微分同胚，它会“扭曲”空间。图来自 lipman2024flow。

**例（线性向量场）.** 让我们考虑向量场 $u_{t}(x)$ 的一个简单例子：它是 $x$ 的简单线性函数，即对 $\theta > 0$ 有 $u_{t}(x) = -\theta x$。于是函数

$$
\psi_{t}(x_{0}) = \exp(-\theta t)\, x_{0}
$$

定义了一个求解流方程的流 $\psi$。你可以自己验证这一点：检查 $\psi_{0}(x_{0}) = x_{0}$，并计算

$$
\frac{\mathrm{d}}{\mathrm{d} t}\psi_{t}(x_{0}) = \frac{\mathrm{d}}{\mathrm{d} t}\bigl(\exp(-\theta t)\, x_{0}\bigr) = -\theta\exp(-\theta t)\, x_{0} = -\theta\psi_{t}(x_{0}) = u_{t}(\psi_{t}(x_{0})),
$$

其中在 (i) 处我们使用了链式法则（chain rule）。在后文的 Ornstein–Uhlenbeck 过程示意图中，我们把这种形式的、指数收敛到 $0$ 的流画了出来。

#### 模拟常微分方程

一般而言，如果 $u_{t}$ 并不像上一个例子那样简单，就不可能显式计算出流 $\psi_{t}$。在这些情形中，人们使用数值方法（numerical methods）来模拟常微分方程。幸运的是，这是数值分析（numerical analysis）中一个经典且研究充分的题目，存在大量强有力的方法（iserles2009first）。最简单、也最直观的方法之一是 Euler 方法（Euler method）。在 Euler 方法中，我们以 $X_{0} = x_{0}$ 初始化，并按下式更新

$$
X_{t+h} = X_{t} + h\, u_{t}(X_{t}) \quad (t = 0, h, 2h, 3h, \ldots, 1 - h)
$$

其中 $h = n^{-1} > 0$ 是步长（step size），$n \in \mathbb{N}$ 是模拟步数。对本课程而言，Euler 方法已经足够好。为了让你体会一种更复杂的方法，我们来考虑 Heun 方法（Heun's method），其更新规则为

$$
\begin{aligned}
X_{t+h}' &= X_{t} + h\, u_{t}(X_{t}) && \blacktriangleright\  \\
X_{t+h} &= X_{t} + \frac{h}{2}\bigl(u_{t}(X_{t}) + u_{t+h}(X_{t+h}')\bigr) && \blacktriangleright\  u 
\end{aligned}
$$

直观地说，Heun 方法是这样的：它先对下一步可能落在何处取一个初始猜测 $X_{t+h}'$，再通过更新后的猜测来修正最初所取的方向。

#### 流模型

现在，我们可以把向量场取成神经网络向量场（neural network vector field）$u_{t}^{\theta}$，从而经由常微分方程构造一个生成模型。目前，我们仅仅是指 $u_{t}^{\theta}$ 是一个带参数 $\theta$ 的参数化函数 $u_{t}^{\theta} : \mathbb{R}^{d} \times [0,1] \to \mathbb{R}^{d}$。稍后，我们将讨论神经网络架构的具体选择。记住，我们的目标是从分布 $p_{\mathrm{data}}$ 生成样本 $z \sim p_{\mathrm{data}}$。特别地，这些样本必须是随机的。不过请注意，常微分方程本身并不是随机的，而是完全确定性（deterministic）的。为了注入一些随机性，我们只需让初值 $X_{0}$ 成为随机的。具体而言，我们选择一个初始分布（initial distribution）$p_{\mathrm{init}}$。在大多数情形中，我们把 $p_{\mathrm{init}} = \mathcal{N}(0, I_{d})$ 取成一个简单的标准高斯（standard Gaussian）。最重要的是，无论你选择哪一个分布，它都必须是我们在推理阶段（inference time）能够轻易从中采样的分布。于是，流模型由如下常微分方程描述

$$
\begin{aligned}
X_{0} &\sim p_{\mathrm{init}} && \blacktriangleright\  \\
\frac{\mathrm{d}}{\mathrm{d} t} X_{t} &= u_{t}^{\theta}(X_{t}) && \blacktriangleright\ 
\end{aligned}
$$

我们的目标是使轨迹的终点 $X_{1}$ 具有分布 $p_{\mathrm{data}}$，即

$$
X_{1} \sim p_{\mathrm{data}} \quad \Leftrightarrow \quad \psi_{1}^{\theta}(X_{0}) \sim p_{\mathrm{data}}
$$

其中 $\psi_{t}^{\theta}$ 描述由 $u_{t}^{\theta}$ 诱导的流。不过请注意：尽管它被称为流模型，**神经网络参数化的是向量场，而不是流**。为了计算出流，我们需要模拟常微分方程。在下面的算法中，我们总结如何从流模型中采样。

**算法（用 Euler 方法从流模型采样）.**

**输入：** 神经网络向量场 $u_{t}^{\theta}$，步数 $n$

1. 令 $t = 0$。
2. 令步长 $h = \frac{1}{n}$。
3. 抽取样本 $X_{0} \sim p_{\mathrm{init}}$。
4. 对 $i = 1, \ldots, n$：
   - 计算 $X_{t+h} = X_{t} + h\, u_{t}^{\theta}(X_{t})$。
   - 更新 $t \leftarrow t + h$。
5. 返回 $X_{1}$。

### 2.2 扩散模型

随机微分方程把常微分方程的确定性轨迹，扩展为带有随机性的轨迹。一条随机轨迹通常称为随机过程（stochastic process）$(X_{t})_{0 \leq t \leq 1}$，并由下面两点给出：

$$
\begin{aligned}
& X_{t}  0 \leq t \leq 1  \\
& X : [0,1] \to \mathbb{R}^{d}, \quad t \mapsto X_{t}  X 
\end{aligned}
$$

特别地，当我们把同一个随机过程模拟两次时，可能会得到不同的结果，因为其动力学被设计成是随机的。

#### 布朗运动

随机微分方程经由布朗运动（Brownian motion）来构造。布朗运动是一种基本的随机过程，源于对物理扩散过程的研究。你可以把布朗运动想成一次连续的随机游走（continuous random walk）。

**图注.** 维数 $d = 1$ 的布朗运动 $W_{t}$ 的样本轨迹，使用后文给出的模拟式得到。

让我们来定义它。布朗运动 $W = (W_{t})_{0 \leq t \leq 1}$ 是一个随机过程，满足 $W_{0} = 0$，轨迹 $t \mapsto W_{t}$ 连续，并且下面两个条件成立：

1. **正态增量（normal increments）：** 对所有 $0 \leq s < t$，$W_{t} - W_{s} \sim \mathcal{N}(0, (t - s) I_{d})$。也就是说，增量服从高斯分布，其方差随时间线性增长（$I_{d}$ 是单位矩阵（identity matrix））。
2. **独立增量（independent increments）：** 对任意 $0 \leq t_{0} < t_{1} < \cdots < t_{n} = 1$，增量 $W_{t_{1}} - W_{t_{0}}, \ldots, W_{t_{n}} - W_{t_{n-1}}$ 是相互独立的随机变量。

布朗运动也称为 Wiener 过程（Wiener process），这也是我们用 $W$ 来记它的原因。Norbert Wiener 是一位曾在麻省理工学院（MIT）任教的著名数学家。在麻省理工学院数学系，现在仍能看到他的肖像。我们可以用步长 $h > 0$ 轻易地近似模拟一条布朗运动：令 $W_{0} = 0$，并按下式更新

$$
W_{t+h} = W_{t} + \sqrt{h}\, \epsilon_{t}, \quad \epsilon_{t} \sim \mathcal{N}(0, I_{d}) \quad (t = 0, h, 2h, \ldots, 1 - h)
$$

在上面的布朗运动轨迹图中，我们画出了若干条示例轨迹。布朗运动之于随机过程的研究，正如高斯分布之于概率分布的研究，都处于中心位置。从金融到统计物理，再到流行病学，对布朗运动的研究有着远远超出机器学习的广泛应用。例如在金融中，布朗运动被用来为复杂金融工具的价格建模。仅仅作为一种数学构造，布朗运动也令人着迷：例如，布朗运动的路径是连续的（因此你画它时笔可以始终不离开纸面），但它们又是无限长的（因此你永远也画不完）。

**图注.** 维数 $d = 1$、$\theta = 0.25$、并取若干不同的 $\sigma$（从左到右递增）时，Ornstein–Uhlenbeck 过程（Ornstein-Uhlenbeck process；见后文相应方程）的示意。当 $\sigma = 0$ 时，我们恢复一个流（光滑的确定性轨迹），它在 $t \to \infty$ 时收敛到原点。当 $\sigma > 0$ 时，路径是随机的，并在 $t \to \infty$ 时收敛到高斯分布 $\mathcal{N}\bigl(0, \frac{\sigma^{2}}{2\theta}\bigr)$。

#### 从常微分方程到随机微分方程

随机微分方程的想法，是在常微分方程的确定性动力学之上，再加上由布朗运动驱动的随机动力学。因为一切都是随机的，我们不能再像常微分方程那样取导数。因此，我们需要找到常微分方程的一种不使用导数的等价表述。为此，让我们把常微分方程的轨迹 $(X_{t})_{0 \leq t \leq 1}$ 改写如下：

$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d} t} X_{t} &= u_{t}(X_{t}) && \blacktriangleright\  \\
\Leftrightarrow\quad \frac{1}{h}\left(X_{t+h} - X_{t}\right) &= u_{t}(X_{t}) + R_{t}(h) && \\
\Leftrightarrow\quad X_{t+h} &= X_{t} + h\, u_{t}(X_{t}) + h\, R_{t}(h) && \blacktriangleright\ 
\end{aligned}
$$

其中 $R_{t}(h)$ 描述一个在 $h$ 很小时可以忽略的函数，即 $\lim_{h \to 0} R_{t}(h) = 0$，而在 (i) 处我们只是使用了导数的定义。上面的推导不过是把我们已经知道的事情重新说一遍：常微分方程的一条轨迹 $(X_{t})_{0 \leq t \leq 1}$ 在每一个时间步上，都沿着方向 $u_{t}(X_{t})$ 走一小步。现在我们可以修改最后一个方程，使它成为随机的：随机微分方程的一条轨迹 $(X_{t})_{0 \leq t \leq 1}$ 在每一个时间步上，都沿着方向 $u_{t}(X_{t})$ 走一小步，再加上来自布朗运动的一份贡献：

$$
X_{t+h} = X_{t} + h\, u_{t}(X_{t}) + \sigma_{t} (W_{t+h} - W_{t}) + h\, R_{t}(h)
$$

其中 $\sigma_{t} \geq 0$ 描述扩散系数（diffusion coefficient），而 $R_{t}(h)$ 描述一个随机误差项，使得标准差 $\mathbb{E}[\|R_{t}(h)\|^{2}]^{1/2} \to 0$ 在 $h \to 0$ 时趋于零。上式描述的就是一条随机微分方程。通常把它写成下面这种符号记号：

$$
\begin{aligned}
\mathrm{d} X_{t} &= u_{t}(X_{t})\, \mathrm{d} t + \sigma_{t}\, \mathrm{d} W_{t} && \blacktriangleright\  \\
X_{0} &= x_{0} && \blacktriangleright\ 
\end{aligned}
$$

不过，始终要记住：上面的 $\mathrm{d} X_{t}$ 记号，只是无穷小更新式的一种纯形式化的写法。遗憾的是，随机微分方程不再具有流映射 $\phi_{t}$。这是因为 $X_{t}$ 的值不再完全由 $X_{0} \sim p_{\mathrm{init}}$ 决定，演化本身就是随机的。尽管如此，和常微分方程一样，我们有：

**定理（随机微分方程解的存在性与唯一性）.** 若 $u : \mathbb{R}^{d} \times [0,1] \to \mathbb{R}^{d}$ 连续可微且导数有界，并且 $\sigma_{t}$ 连续，则该随机微分方程有解，这个解由满足无穷小更新式的唯一随机过程 $(X_{t})_{0 \leq t \leq 1}$ 给出。

如果这是一门随机分析课，我们会花好几讲来证明这个定理，并以完全的数学严格性来构造随机微分方程，也就是从第一性原理构造布朗运动，并通过随机积分（stochastic integration）来构造过程 $X_{t}$。由于本课程聚焦于机器学习，更技术性的处理请参阅 mao2007stochastic。最后请注意，每一个常微分方程也都是随机微分方程——只要扩散系数消失，即 $\sigma_{t} = 0$。因此，在本课程的其余部分，**当我们谈论随机微分方程时，把常微分方程视为其特例**。

**例（Ornstein–Uhlenbeck 过程）.** 让我们考虑常数扩散系数 $\sigma_{t} = \sigma \geq 0$，以及常数线性漂移（drift）$u_{t}(x) = -\theta x$，其中 $\theta > 0$。由此得到随机微分方程

$$
\mathrm{d} X_{t} = -\theta X_{t}\, \mathrm{d} t + \sigma\, \mathrm{d} W_{t}.
$$

上述随机微分方程的解 $(X_{t})_{0 \leq t \leq 1}$ 称为 Ornstein–Uhlenbeck 过程。我们已在前面的示意图中把它画了出来。向量场 $-\theta x$ 把过程推回它的中心 $0$（因为漂移始终指向与当前位置相反的方向），而扩散系数 $\sigma$ 则不断加入更多噪声。如果把这个过程模拟到 $t \to \infty$，它会收敛到高斯分布 $\mathcal{N}(0, \sigma^{2}/(2\theta))$。请注意，当 $\sigma = 0$ 时，我们得到的就是前面在线性向量场的流中已经研究过的、具有线性向量场的流。

#### 模拟随机微分方程

如果到目前为止你觉得随机微分方程的抽象定义难以把握，那也不必担心。思考随机微分方程的一种更直观的方式，是回答这个问题：我们该如何模拟一条随机微分方程？最简单的这类格式称为 Euler–Maruyama 方法（Euler-Maruyama method）。它对随机微分方程的意义，本质上就是 Euler 方法对常微分方程的意义。使用 Euler–Maruyama 方法时，我们初始化 $X_{0} = x_{0}$，并迭代地按下式更新

$$
X_{t+h} = X_{t} + h\, u_{t}(X_{t}) + \sqrt{h}\, \sigma_{t}\, \epsilon_{t}, \quad \epsilon_{t} \sim \mathcal{N}(0, I_{d})
$$

其中 $h = n^{-1} > 0$ 是步长超参数（hyperparameter），$n \in \mathbb{N}$。换句话说，用 Euler–Maruyama 方法做模拟时，我们沿着 $u_{t}(X_{t})$ 的方向走一小步，并加上一份由 $\sqrt{h}\, \sigma_{t}$ 缩放的高斯噪声。在本课程中模拟随机微分方程时（例如在配套的实验中），我们通常就坚持使用 Euler–Maruyama 方法。

**算法（用 Euler–Maruyama 方法从扩散模型采样）.**

**输入：** 神经网络 $u_{t}^{\theta}$，步数 $n$，扩散系数 $\sigma_{t}$

1. 令 $t = 0$。
2. 令步长 $h = \frac{1}{n}$。
3. 抽取样本 $X_{0} \sim p_{\mathrm{init}}$。
4. 对 $i = 1, \ldots, n$：
   - 抽取样本 $\epsilon \sim \mathcal{N}(0, I_{d})$。
   - 计算 $X_{t+h} = X_{t} + h\, u_{t}^{\theta}(X_{t}) + \sigma_{t} \sqrt{h}\, \epsilon$。
   - 更新 $t \leftarrow t + h$。
5. 返回 $X_{1}$。

#### 扩散模型

现在，我们可以用与常微分方程相同的方式，经由随机微分方程来构造一个生成模型。记住，我们的目标是把一个简单分布 $p_{\mathrm{init}}$ 转化成一个复杂分布 $p_{\mathrm{data}}$。和常微分方程一样，以 $X_{0} \sim p_{\mathrm{init}}$ 随机初始化后再模拟一条随机微分方程，是实现这一转化的自然选择。为了参数化这条随机微分方程，我们可以简单地把它的核心成分——向量场 $u_{t}$——用神经网络 $u_{t}^{\theta}$ 来参数化。于是，扩散模型由下式给出

$$
\begin{aligned}
X_{0} &\sim p_{\mathrm{init}} && \blacktriangleright\  \\
\mathrm{d} X_{t} &= u_{t}^{\theta}(X_{t})\, \mathrm{d} t + \sigma_{t}\, \mathrm{d} W_{t} && \blacktriangleright\ 
\end{aligned}
$$

在上面的算法中，我们描述了如何用 Euler–Maruyama 方法从扩散模型中采样。我们将本节的结果总结如下。

**小结（随机微分方程生成模型）.** 在整个文档中，扩散模型由一个带参数 $\theta$、用来参数化向量场的神经网络 $u_{t}^{\theta}$，以及一个固定的扩散系数 $\sigma_{t}$ 组成：

$$
\begin{aligned}
\quad & u^{\theta} : \mathbb{R}^{d} \times [0,1] \to \mathbb{R}^{d},\; (x, t) \mapsto u_{t}^{\theta}(x)  \theta \\
\quad & \sigma_{t} : [0,1] \to [0, \infty),\; t \mapsto \sigma_{t}
\end{aligned}
$$

为了从我们的随机微分方程模型中取得样本（也就是生成对象），步骤如下：

$$
\begin{aligned}
\quad X_{0} &\sim p_{\mathrm{init}} && \blacktriangleright\  \\
\quad \mathrm{d} X_{t} &= u_{t}^{\theta}(X_{t})\, \mathrm{d} t + \sigma_{t}\, \mathrm{d} W_{t} && \blacktriangleright\  0  1 \\
\quad X_{1} &\sim p_{\mathrm{data}} && \blacktriangleright\  X_{1}  p_{\mathrm{data}}
\end{aligned}
$$

$\sigma_{t} = 0$ 的扩散模型就是流模型。

---

<!-- 来源文件：03.zh.md -->

# 3 流匹配（Flow Matching）

在上一节中，我们把流模型（flow model）和扩散模型（diffusion model）构造为生成模型（generative model），其参数是神经网络（neural network）向量场（vector field）$u_t^\theta$。然而，我们尚未讨论如何*训练*它们，也就是如何优化参数 $\theta$，使得生成模型返回合理的结果，例如一张好看的图像，或一段精彩的视频。接下来，我们讨论**流匹配**（Lipman 等，2022；Albergo 等，2023；Liu 等，2022），这是一种训练 $u_t^\theta$ 的算法，它简单、可扩展，并代表当前的最优水平（state-of-the-art）。

在本节中，我们只限于流模型，也就是说，我们有一个神经网络 $u_t^\theta$，并通过模拟常微分方程（ordinary differential equation, ODE）

$$
X_0\sim p_{\mathrm{init}},\quad \mathrm{d} X_t = u_t^\theta(X_t)\,\mathrm{d} t \qquad 
$$

从生成模型取得样本，再把 $t=1$ 时的端点 $X_1$ 用作样本。正如我们已经讨论过的，我们的目标是让 $X_1$ 服从数据分布（data distribution）$p_{\mathrm{data}}$，即 $X_1\sim p_{\mathrm{data}}$。因此，“如何训练”这个神经网络，真正要问的是下面这个问题：**我们应当如何优化 $\theta$，使得模拟本节开头写出的流模型之后，得到的样本来自数据分布 $X_1\sim p_{\mathrm{data}}$？**

**图注.** 一组图像经由高斯条件概率路径（Gaussian conditional probability path），从噪声（noise）逐步插值到数据（data）。注意，每张图像都是维数为 $d=32\times 32$ 的一个数据点，因此我们绘制的是概率路径（probability path）上的单个样本；而在后文那幅条件路径与边缘路径的直方图中，我们把分布画成二维直方图。

## 3.1 条件概率路径与边缘概率路径

流匹配的第一步，是指定一条概率路径。直观地说，概率路径指定了噪声 $p_{\mathrm{init}}$ 与数据 $p_{\mathrm{data}}$ 之间的逐步插值（见上面从噪声到数据的图）。但我们为什么需要它？回想一下，我们想要的常微分方程轨迹（trajectory）在 $t=0$ 时满足 $X_0\sim p_{\mathrm{init}}$，在 $t=1$ 时满足 $X_1\sim p_{\mathrm{data}}$。那么，起点与终点之间的时刻 $0<t<1$ 呢？事实表明，中间应当发生什么，我们有一定的选择自由，而这正是概率路径在数学上所形式化的内容。

下文中，对数据点 $z\in\mathbb{R}^d$，我们用 $\delta_z$ 表示**狄拉克 $\delta$（Dirac delta）**“分布”。这是所能想象的最简单的分布：从 $\delta_z$ 抽样总是返回 $z$（也就是说，它是确定性的）。**条件（插值）概率路径（conditional (interpolating) probability path）**是 $\mathbb{R}^d$ 上的一族分布 $p_t(x|z)$，满足

$$
p_0(\cdot|z)=p_{\mathrm{init}}, \quad p_1(\cdot|z)=\delta_{z}\quad z\in\mathbb{R}^d.
$$

换句话说，条件概率路径把初始分布（initial distribution）$p_{\mathrm{init}}$ 逐步变成*单个*数据点（例如见上面从噪声到数据的图）。你可以把概率路径想成分布空间中的一条轨迹。

每条条件概率路径 $p_t(x|z)$ 都诱导一条**边缘概率路径（marginal probability path）** $p_t(x)$，定义为如下得到的分布：先从数据分布中抽取一个数据点 $z\sim p_{\mathrm{data}}$，再从 $p_t(\cdot|z)$ 中抽样：

$$
\begin{aligned}
z&\sim p_{\mathrm{data}}, \quad x\sim p_t(\cdot|z)\quad \Rightarrow x\sim p_t &&\blacktriangleright\,\,\\
p_t(x) &= \int p_t(x|z)\, p_{\mathrm{data}}(z)\, \mathrm{d} z &&\blacktriangleright\,\,
\end{aligned}
$$

注意，我们知道如何从 $p_t$ 抽样，但不知道密度值 $p_t(x)$，因为这个积分是难解的（也就是说，我们实际上可以计算上面“从边缘路径抽样”的那一式，但不能计算“边缘路径的密度”那一式）。请自行验证：由于条件概率路径的端点条件对 $p_t(\cdot|z)$ 的限制，边缘概率路径 $p_t$ 在 $p_{\mathrm{init}}$ 与 $p_{\mathrm{data}}$ 之间插值：

$$
p_0 = p_{\mathrm{init}}\quad\quad p_1=p_{\mathrm{data}}. \qquad \blacktriangleright\,\,
$$

概率路径中远远最重要的例子是高斯概率路径（Gaussian probability path），因此我们强烈建议把下一个例子读透。

**图注.** 条件（上）与边缘（下）概率路径的示意。这里绘制的是 $\alpha_t=t$、$\beta_t=1-t$ 的高斯概率路径。条件概率路径在高斯分布 $p_{\mathrm{init}}=\mathcal{N}(0,I_d)$ 与单个数据点 $z$ 所对应的 $p_{\mathrm{data}}=\delta_z$ 之间插值。边缘概率路径在一个高斯分布与数据分布 $p_{\mathrm{data}}$ 之间插值（这里 $p_{\mathrm{data}}$ 是维数 $d=2$ 的玩具分布，用棋盘图案表示）。

**例（高斯条件概率路径）.** 一条特别流行的概率路径是高斯概率路径。这是**大多数当前最优模型所使用的概率路径**。设 $\alpha_t,\beta_t$ 为**噪声调度（noise scheduler）**：两个连续可微、单调的函数，满足 $\alpha_0=\beta_1=0$ 且 $\alpha_1=\beta_0=1$。我们于是把条件概率路径定义为

$$
p_t(\cdot|z) = \mathcal{N}(\alpha_t z,\beta_t^2 I_d) \qquad \blacktriangleright \,\, 
$$

由我们对 $\alpha_t$ 与 $\beta_t$ 所加的条件，它满足

$$
\begin{aligned}
p_0(\cdot|z) &= \mathcal{N}(\alpha_0 z,\beta_0^2 I_d) = \mathcal{N}(0,I_d),\\
p_1(\cdot|z) &= \mathcal{N}(\alpha_1 z,\beta_1^2 I_d) = \delta_{z},
\end{aligned}
$$

其中用到了这一事实：均值为 $z$、方差为零的正态分布就是 $\delta_z$。因此，这样选取的 $p_t(x|z)$ 在 $p_{\mathrm{init}}=\mathcal{N}(0,I_d)$ 时满足条件概率路径的端点条件，从而是一条有效的条件插值路径。在从噪声到数据的那幅图中，我们展示了它在图像上的应用。从边缘路径 $p_t$ 抽样可以写成

$$
z\sim p_{\mathrm{data}},\,\epsilon\sim p_{\mathrm{init}} = \mathcal{N}(0,I_d) \,\Rightarrow\, x=\alpha_t z+\beta_t \epsilon\sim p_t \qquad \blacktriangleright \,\,
$$

直观地说，上述过程在较小的 $t$ 上加入更多噪声，直到时刻 $t=0$，那时只剩下噪声。在条件路径与边缘路径的直方图中，我们绘制了这样一条插值路径的例子。

## 3.2 条件向量场与边缘向量场

概率路径 $(p_t)_{0\leq t\leq 1}$ 指定了轨迹上的点 $X_t$ *应当*具有的分布 $X_t\sim p_t$。到这一步，这还只是我们“希望”成立的事情。但怎样才能找到一个向量场，使得轨迹 $X_t$ 跟随这条概率路径？流匹配显式地构造了这样一个向量场——“边缘向量场（marginal vector field）”——我们在本节中解释它。

对每个数据点 $z\in\mathbb{R}^d$，令 $u_t^{\mathrm{target}}(\cdot|z)$ 表示一个**条件向量场（conditional vector field）**。它可以是任何一个向量场，只要对应的常微分方程给出条件概率路径 $p_t(\cdot|z)$，也就是使得

$$
X_0\sim p_{\mathrm{init}},\quad \frac{\mathrm{d}}{\mathrm{d} t}X_t = u_t^{\mathrm{target}}(X_t|z)\quad \Rightarrow \quad X_t\sim p_t(\cdot|z)\quad (0\leq t\leq 1).
$$

我们常常可以亲手做一点代数，解析地求出条件向量场 $u_t^{\mathrm{target}}(\cdot|z)$。下面以贯穿全文的高斯概率路径为例，推导条件向量场 $u_t(x|z)$，见后文“高斯概率路径的目标常微分方程”这一例。

乍看之下，条件向量场似乎没有用，因为常微分方程的全部端点 $X_1$ 都会坍缩到 $X_1=z$，也就是说，我们只是在重新生成已知的数据点 $z$。然而，条件向量场是一块基本构件，用来构造一个能从 $p_{\mathrm{data}}$ 生成真实样本的向量场：

**定理（边缘化技巧（marginalization trick））.** 设 $u_t^{\mathrm{target}}(x|z)$ 是一个条件向量场（即满足上面条件向量场的常微分方程条件）。则按下式定义的边缘向量场 $u_t^{\mathrm{target}}(x)$

$$
u_t^{\mathrm{target}}(x) = \int u_t^{\mathrm{target}}(x|z)\frac{p_t(x|z)\,p_{\mathrm{data}}(z)}{p_t(x)}\,\mathrm{d} z,
$$

跟随边缘概率路径，即

$$
X_0\sim p_{\mathrm{init}},\quad \frac{\mathrm{d}}{\mathrm{d} t}X_t = u_t^{\mathrm{target}}(X_t)\quad \Rightarrow \quad X_t\sim p_t\quad (0\leq t\leq 1).
$$

**特别地，这条常微分方程满足 $X_1\sim p_{\mathrm{data}}$，因此我们可以说“$u_t^{\mathrm{target}}$ 把噪声 $p_{\mathrm{init}}$ 转换成数据 $p_{\mathrm{data}}$”。**

**图注.** 边缘化技巧定理的示意。用常微分方程模拟一条概率路径。数据分布 $p_{\mathrm{data}}$ 为蓝色背景。高斯分布 $p_{\mathrm{init}}$ 为红色背景。上行：条件概率路径。左：来自条件路径 $p_t(\cdot|z)$ 的真值（ground truth）样本。中：随时间变化的常微分方程样本。右：用后文条件高斯向量场公式中的 $u_t^{\mathrm{target}}(x|z)$ 模拟常微分方程所得的轨迹。下行：模拟一条边缘概率路径。左：来自 $p_t$ 的真值样本。中：随时间变化的常微分方程样本。右：用边缘向量场 $u_t^{\mathrm{flow}}(x)$ 模拟常微分方程所得的轨迹。可以看到，条件向量场跟随条件概率路径，边缘向量场跟随边缘概率路径。

**例（高斯概率路径的目标常微分方程）.** 和前面一样，对噪声调度 $\alpha_t,\beta_t$，令 $p_t(\cdot|z)=\mathcal{N}(\alpha_t z,\beta_t^2 I_d)$（见高斯条件概率路径的定义）。令 $\dot{\alpha}_t=\partial_t\alpha_t$、$\dot{\beta}_t=\partial_t\beta_t$ 分别表示 $\alpha_t$ 与 $\beta_t$ 各自的时间导数。这里要说明，由下式给出的**条件高斯向量场（conditional Gaussian vector field）**

$$
u_t^{\mathrm{target}}(x|z) = \left(\dot{\alpha}_t-\frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z+\frac{\dot{\beta}_t}{\beta_t}x
$$

在边缘化技巧定理的意义下是一个有效的条件向量场模型：若 $X_0\sim\mathcal{N}(0,I_d)$，则其常微分方程轨迹 $X_t$ 满足 $X_t\sim p_t(\cdot|z)=\mathcal{N}(\alpha_t z,\beta_t^2 I_d)$。在上面那幅用常微分方程模拟概率路径的图中，我们把条件概率路径的样本（真值）与这条流模拟出的常微分方程轨迹样本加以比较，从视觉上确认了这一点。可以看到，分布是吻合的。现在来证明它。

**证明.** 我们首先构造一个条件流模型（conditional flow model）$\psi_t^{\mathrm{target}}(x|z)$，定义为

$$
\psi_t^{\mathrm{target}}(x|z) = \alpha_t z + \beta_t x.
$$

若 $X_t$ 是 $\psi_t^{\mathrm{target}}(\cdot|z)$ 的常微分方程轨迹，且 $X_0\sim p_{\mathrm{init}}=\mathcal{N}(0,I_d)$，则按定义

$$
X_t = \psi_t^{\mathrm{target}}(X_0|z) = \alpha_t z + \beta_t X_0 \sim \mathcal{N}(\alpha_t z,\beta_t^2 I_d) = p_t(\cdot|z).
$$

我们由此断定，这些轨迹的分布与条件概率路径相同（也就是说，条件向量场的常微分方程条件成立）。剩下的是从 $\psi_t^{\mathrm{target}}(x|z)$ 中提取向量场 $u_t^{\mathrm{target}}(x|z)$。按流（flow）的定义（流所满足的常微分方程），对所有 $x,z\in\mathbb{R}^d$ 成立

$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d} t}\psi_t^{\mathrm{target}}(x|z) &= u_t^{\mathrm{target}}(\psi_t^{\mathrm{target}}(x|z)|z)\\
\Leftrightarrow\quad \dot{\alpha}_t z+\dot{\beta}_t x &= u_t^{\mathrm{target}}(\alpha_t z+\beta_t x|z)\\
\Leftrightarrow\quad \dot{\alpha}_t z+\dot{\beta}_t\left(\frac{x-\alpha_t z}{\beta_t}\right) &= u_t^{\mathrm{target}}(x|z)\\
\Leftrightarrow\quad \left(\dot{\alpha}_t-\frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z+\frac{\dot{\beta}_t}{\beta_t}x &= u_t^{\mathrm{target}}(x|z)
\end{aligned}
$$

其中，(i) 用到了 $\psi_t^{\mathrm{target}}(x|z)$ 的定义（即上面的高斯条件流），(ii) 把 $x$ 重新参数化为 $(x-\alpha_t z)/\beta_t$，(iii) 只是做了一些代数。注意，最后一个方程正是我们在条件高斯向量场公式中定义的条件高斯向量场。这就证明了所述结论。也可以把它代入本节稍后引入的连续性方程（continuity equation），再检验一次。

上面那幅图是边缘化技巧定理的一幅示意。现在来获得对边缘向量场的一些直观认识。统计学中的**贝叶斯法则（Bayes' rule）**指出，下面这一项描述的是一个后验分布（posterior）

$$
\frac{p_t(x|z)\,p_{\mathrm{data}}(z)}{p_t(x)} = xz
$$

其中 $p_{\mathrm{data}}(z)$ 是先验分布（prior）。边缘向量场于是只是一个平均：对每一个*可能的*数据点 $z$，它取速度（velocity）$u_t(x|z)$，也就是会把我们带到 $z$ 的那个方向，再按我们有多相信 $x$ 来自 $z$，给这个速度加权。对所有数据点取平均，就得到边缘向量场。

本节余下部分将把这一直观认识严格化，并证明边缘化技巧定理。作为主要的数学工具，我们将使用连续性方程，它是数学与物理学中的一个基本方程。把**散度（divergence）**算子 $\mathrm{div}$ 定义为

$$
\mathrm{div}(v_t)(x)=\sum_{i=1}^{d}\frac{\partial}{\partial x_i}v_t^i(x)
$$

其中 $v_t^i$ 是 $v_t$ 的第 $i$ 个坐标。

**定理（连续性方程）.** 考虑一个流模型，其向量场为 $u_t^{\mathrm{target}}$，且 $X_0\sim p_{\mathrm{init}}=p_0$。则对所有 $0\leq t\leq 1$ 都有 $X_t\sim p_t$，当且仅当

$$
\partial_t p_t(x)=-\mathrm{div}(p_t u_t^{\mathrm{target}})(x)\quad x\in\mathbb{R}^d,\ 0\leq t\leq 1,
$$

其中 $\partial_t p_t(x)=\frac{\mathrm{d}}{\mathrm{d} t}p_t(x)$ 表示 $p_t(x)$ 的时间导数。这个方程称为连续性方程。

对偏爱数学的读者，我们在福克–普朗克方程（Fokker–Planck equation）的证明一节中给出连续性方程的一个自包含证明。在继续之前，让我们试着直观地理解连续性方程。左端 $\partial_t p_t(x)$ 描述 $x$ 处的概率 $p_t(x)$ 随时间改变了多少。直观地说，这一改变应当对应于概率质量的净流入。对于流模型，粒子 $X_t$ 沿着向量场 $u_t^{\mathrm{target}}$ 运动。你或许还记得物理学中的结论：散度衡量的是向量场的某种净流出。因此，负的散度衡量净流入。再乘上当前位于 $x$ 处的全部概率质量，我们就得到，净量 $-\mathrm{div}(p_t u_t)$ 衡量的是概率质量的总流入。由于概率质量守恒（积分总是为 1），方程的左端与右端应当相同！下面给出边缘化技巧定理的证明。

**证明（边缘化技巧定理）.** 由连续性方程定理，我们必须证明：按边缘向量场积分公式定义的边缘向量场 $u_t^{\mathrm{target}}$ 满足连续性方程。可以直接计算：

$$
\begin{aligned}
\partial_t p_t(x) &= \partial_t\int p_t(x|z)\, p_{\mathrm{data}}(z)\, \mathrm{d} z = \int \partial_t p_t(x|z)\, p_{\mathrm{data}}(z)\, \mathrm{d} z\\
&= \int -\mathrm{div}\big(p_t(\cdot|z)\, u_t^{\mathrm{target}}(\cdot|z)\big)(x)\, p_{\mathrm{data}}(z)\, \mathrm{d} z\\
&= -\mathrm{div}\left(\int p_t(x|z)\, u_t^{\mathrm{target}}(x|z)\, p_{\mathrm{data}}(z)\, \mathrm{d} z\right)\\
&= -\mathrm{div}\left(p_t(x)\int u_t^{\mathrm{target}}(x|z)\frac{p_t(x|z)\,p_{\mathrm{data}}(z)}{p_t(x)}\,\mathrm{d} z\right)(x)\\
&= -\mathrm{div}\big(p_t u_t^{\mathrm{target}}\big)(x),
\end{aligned}
$$

其中，(i) 用到了边缘概率路径中 $p_t(x)$ 的定义，(ii) 用到了条件概率路径 $p_t(\cdot|z)$ 的连续性方程，(iii) 利用散度的定义交换了积分与散度算子，(iv) 乘上并除以 $p_t(x)$，(v) 用到了边缘向量场的积分公式。上面这一串等式的开头与结尾表明，连续性方程对 $u_t^{\mathrm{target}}$ 成立。由连续性方程定理，这足以推出边缘常微分方程跟随边缘概率路径的结论，证明完毕。

## 3.3 学习边缘向量场

现在我们可以描述训练算法了。流匹配的目标是训练神经网络 $u_t^\theta$，使它等于边缘向量场 $u_t^{\mathrm{target}}$。若这一点成立，则由边缘化技巧定理可知，端点 $X_1\sim p_{\mathrm{data}}$ 具有所期望的分布。下文中，我们用 $\mathrm{Unif}=\mathrm{Unif}_{[0,1]}$ 表示区间 $[0,1]$ 上的均匀分布（uniform distribution），用 $\mathbb{E}$ 表示随机变量的期望（expected value）。得到 $u_t^\theta\approx u_t^{\mathrm{target}}$ 的一种直观做法，是使用均方误差（mean-squared error），也就是使用如下定义的**流匹配损失（flow matching loss）**

$$
\begin{aligned}
\mathcal{L}_{\mathrm{FM}}(\theta) &= \mathbb{E}_{t\sim\mathrm{Unif},\, x\sim p_t}\big[\|u_t^\theta(x)-u_t^{\mathrm{target}}(x)\|^2\big]\\
&= \mathbb{E}_{t\sim\mathrm{Unif},\, z\sim p_{\mathrm{data}},\, x\sim p_t(\cdot|z)}\big[\|u_t^\theta(x)-u_t^{\mathrm{target}}(x)\|^2\big],
\end{aligned}
$$

其中 $p_t(x)=\int p_t(x|z)\,p_{\mathrm{data}}(z)\,\mathrm{d} z$ 是边缘概率路径，而 (i) 用到了从边缘路径抽样的步骤。直观地说，这个损失的意思是：第一，抽取一个随机时间 $t\in[0,1]$。第二，从我们的数据集中抽取一个随机点 $z$，从 $p_t(\cdot|z)$ 中抽样（例如加上一些噪声），并计算 $u_t^\theta(x)$。最后，计算神经网络的输出与边缘向量场 $u_t^{\mathrm{target}}(x)$ 之间的均方误差。不幸的是，到这里我们*还没有*做完。虽然由边缘化技巧定理我们知道 $u_t^{\mathrm{target}}$ 的公式，但由于积分难解，无法高效地计算它。我们转而利用另一件事实：**条件**速度场（velocity field）$u_t^{\mathrm{target}}(x|z)$ 是可解的。为此，定义**条件流匹配损失（conditional flow matching loss）**

$$
\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t\sim\mathrm{Unif},\, z\sim p_{\mathrm{data}},\, x\sim p_t(\cdot|z)}\big[\|u_t^\theta(x)-u_t^{\mathrm{target}}(x|z)\|^2\big].
$$

注意它与边缘流匹配损失的差别：我们使用条件向量场 $u_t^{\mathrm{target}}(x|z)$，而不是边缘向量场 $u_t^{\mathrm{target}}(x)$。由于我们有 $u_t^{\mathrm{target}}(x|z)$ 的解析公式，上面的损失很容易最小化。可是，如果我们关心的是边缘向量场，却去对条件向量场做回归（regression），这有什么意义？事实表明，通过*显式地*对可解的条件向量场做回归，我们就是在*隐式地*对难解的边缘向量场做回归。下一个结果把这一直观认识精确化。

**定理.** 边缘流匹配损失与条件流匹配损失只相差一个常数。也就是说，

$$
\mathcal{L}_{\mathrm{FM}}(\theta) = \mathcal{L}_{\mathrm{CFM}}(\theta) + C,
$$

其中 $C$ 与 $\theta$ 无关。因此它们的梯度相同：

$$
\nabla_\theta \mathcal{L}_{\mathrm{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\mathrm{CFM}}(\theta).
$$

因而，用例如随机梯度下降（stochastic gradient descent, SGD）最小化 $\mathcal{L}_{\mathrm{CFM}}(\theta)$，等价于以同样的方式最小化 $\mathcal{L}_{\mathrm{FM}}(\theta)$。特别地，**对 $\mathcal{L}_{\mathrm{CFM}}(\theta)$ 的最小点 $\theta^*$，将有 $u_t^{\theta^*}=u_t^{\mathrm{target}}$，即神经网络将等于边缘向量场**（假定参数化具有无限的表达能力）。

**证明（直接证明）.** 证明的做法是把均方误差展开成三项，并去掉常数：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{FM}}(\theta) &= \mathbb{E}_{t\sim\mathrm{Unif},\, x\sim p_t}\big[\|u_t^\theta(x)-u_t^{\mathrm{target}}(x)\|^2\big]\\
&= \mathbb{E}_{t\sim\mathrm{Unif},\, x\sim p_t}\big[\|u_t^\theta(x)\|^2 - 2u_t^\theta(x)^T u_t^{\mathrm{target}}(x) + \|u_t^{\mathrm{target}}(x)\|^2\big]\\
&= \mathbb{E}_{t\sim\mathrm{Unif},\, x\sim p_t}\big[\|u_t^\theta(x)\|^2\big] - 2\mathbb{E}_{t\sim\mathrm{Unif},\, x\sim p_t}\big[u_t^\theta(x)^T u_t^{\mathrm{target}}(x)\big] + \mathbb{E}_{t\sim\mathrm{Unif}_{[0,1]},\, x\sim p_t}\big[\|u_t^{\mathrm{target}}(x)\|^2\big]\\
&= \mathbb{E}_{t\sim\mathrm{Unif},\, z\sim p_{\mathrm{data}},\, x\sim p_t(\cdot|z)}\big[\|u_t^\theta(x)\|^2\big] - 2\mathbb{E}_{t\sim\mathrm{Unif},\, x\sim p_t}\big[u_t^\theta(x)^T u_t^{\mathrm{target}}(x)\big] + C_1
\end{aligned}
$$

其中 (i) 由定义成立，(ii) 用到公式 $\|a-b\|^2=\|a\|^2-2a^T b+\|b\|^2$，(iii) 中我们定义常数 $C_1$，(iv) 用到从边缘路径抽样的步骤所给出的 $p_t$ 的抽样过程。让我们把第二项重新写出来：

$$
\begin{aligned}
&\mathbb{E}_{t\sim\mathrm{Unif},\, x\sim p_t}\big[u_t^\theta(x)^T u_t^{\mathrm{target}}(x)\big]\\
&= \int_{0}^{1}\int p_t(x)\, u_t^\theta(x)^T u_t^{\mathrm{target}}(x)\, \mathrm{d} x\, \mathrm{d} t\\
&= \int_{0}^{1}\int p_t(x)\, u_t^\theta(x)^T\left[\int u_t^{\mathrm{target}}(x|z)\frac{p_t(x|z)\,p_{\mathrm{data}}(z)}{p_t(x)}\,\mathrm{d} z\right]\mathrm{d} x\, \mathrm{d} t\\
&= \int_{0}^{1}\int\int u_t^\theta(x)^T u_t^{\mathrm{target}}(x|z)\, p_t(x|z)\, p_{\mathrm{data}}(z)\, \mathrm{d} z\, \mathrm{d} x\, \mathrm{d} t\\
&= \mathbb{E}_{t\sim\mathrm{Unif},\, z\sim p_{\mathrm{data}},\, x\sim p_t(\cdot|z)}\big[u_t^\theta(x)^T u_t^{\mathrm{target}}(x|z)\big]
\end{aligned}
$$

其中，(i) 把期望写成积分，(ii) 用到边缘向量场的积分公式，(iii) 用到积分是线性的这一事实，(iv) 把积分写成期望。注意，这正是证明中真正关键的一步：等式的开头使用边缘向量场 $u_t^{\mathrm{target}}(x)$，而结尾使用条件向量场 $u_t^{\mathrm{target}}(x|z)$。我们把它代入 $\mathcal{L}_{\mathrm{FM}}$ 的式子，得到

$$
\begin{aligned}
\mathcal{L}_{\mathrm{FM}}(\theta) &= \mathbb{E}_{t\sim\mathrm{Unif},\, z\sim p_{\mathrm{data}},\, x\sim p_t(\cdot|z)}\big[\|u_t^\theta(x)\|^2\big] - 2\mathbb{E}_{t\sim\mathrm{Unif},\, z\sim p_{\mathrm{data}},\, x\sim p_t(\cdot|z)}\big[u_t^\theta(x)^T u_t^{\mathrm{target}}(x|z)\big] + C_1\\
&= \mathbb{E}_{t\sim\mathrm{Unif},\, z\sim p_{\mathrm{data}},\, x\sim p_t(\cdot|z)}\big[\|u_t^\theta(x)\|^2 - 2u_t^\theta(x)^T u_t^{\mathrm{target}}(x|z) + \|u_t^{\mathrm{target}}(x|z)\|^2 - \|u_t^{\mathrm{target}}(x|z)\|^2\big] + C_1\\
&= \mathbb{E}_{t\sim\mathrm{Unif},\, z\sim p_{\mathrm{data}},\, x\sim p_t(\cdot|z)}\big[\|u_t^\theta(x)-u_t^{\mathrm{target}}(x|z)\|^2\big] + \mathbb{E}_{t\sim\mathrm{Unif},\, z\sim p_{\mathrm{data}},\, x\sim p_t(\cdot|z)}\big[-\|u_t^{\mathrm{target}}(x|z)\|^2\big] + C_1\\
&= \mathcal{L}_{\mathrm{CFM}}(\theta) + C_2+C_1
\end{aligned}
$$

其中，(i) 代入了刚才导出的式子，(ii) 加上并减去同一个值，(iii) 再次用到公式 $\|a-b\|^2=\|a\|^2-2a^T b+\|b\|^2$，(iv) 定义了一个关于 $\theta$ 为常数的量。证明至此结束。

**算法.** 流匹配训练过程（针对高斯 CondOT 路径 $p_t(x|z)=\mathcal{N}(tz,(1-t)^2)$）

**输入.** 数据集中的样本 $z\sim p_{\mathrm{data}}$，神经网络 $u_t^\theta$。

对每一个数据小批量（mini-batch）：

1. 从数据集中采样一个数据样本 $z$。
2. 采样一个随机时间 $t\sim\mathrm{Unif}_{[0,1]}$。
3. 采样噪声 $\epsilon\sim\mathcal{N}(0,I_d)$。
4. 令
$$
x = tz + (1-t)\epsilon \qquad x\sim p_t(\cdot\mid z)
$$
5. 计算损失
$$
\mathcal{L}(\theta) = \|u_t^\theta(x)-(z-\epsilon)\|^2 \qquad  = \|u_t^\theta(x)-u_t^{\mathrm{target}}(x|z)\|^2
$$
6. 更新 $\theta \gets \mathrm{grad\_update}(\mathcal{L}(\theta))$。

因此，**流匹配训练就是最小化条件流匹配损失。** 训练过程概括在上面的流匹配训练过程中，并在后文棋盘分布上的流匹配训练效果图中加以展示。注意，这个算法有几个显著的特点。第一，训练过程中我们实际上从不模拟任何常微分方程。人们把算法的这一特点称为**免模拟（simulation-free）**。这使得训练极其便宜，因为训练时不必把常微分方程的轨迹逐步展开（那需要很多步）。第二，训练是一个简单的回归目标——我们只是在对 $u_t^{\mathrm{target}}(x|z)$ 做回归。所以它说到底与监督学习（supervised learning）差别并不太大。最后，这个算法极其简单——很难再想出一个简单得多的训练目标。这一切使流匹配成为大规模机器学习模型极具吸引力的方法。一旦 $u_t^\theta$ 训练完成，我们就可以模拟流模型

$$
\mathrm{d} X_t = u_t^\theta(X_t)\, \mathrm{d} t,\quad\quad X_0\sim p_{\mathrm{init}}
$$

例如借助流模型的采样算法，以得到样本 $X_1\sim p_{\mathrm{data}}$。整套流程在文献中称为流匹配（Lipman 等，2022；Liu 等，2022；Albergo 等，2023；Lipman 等，2024）。

现在让我们把条件流匹配损失具体写到高斯概率路径上。

**例（高斯条件概率路径上的流匹配）.** 让我们回到高斯概率路径 $p_t(\cdot|z)=\mathcal{N}(\alpha_t z;\beta_t^2 I_d)$ 的例子。我们可以经由下式从条件路径抽样：

$$
\epsilon\sim\mathcal{N}(0,I_d)\quad \Rightarrow\quad x_t = \alpha_t z + \beta_t \epsilon \sim \mathcal{N}(\alpha_t z,\beta_t^2 I_d)=p_t(\cdot|z).
$$

正如我们在条件高斯向量场公式中所推导的，条件向量场 $u_t^{\mathrm{target}}(x|z)$ 由下式给出

$$
u_t^{\mathrm{target}}(x|z) = \left(\dot{\alpha}_t-\frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z+\frac{\dot{\beta}_t}{\beta_t}x,
$$

其中 $\dot{\alpha}_t=\partial_t\alpha_t$、$\dot{\beta}_t=\partial_t\beta_t$ 是各自的时间导数。代入这一公式，条件流匹配损失为

$$
\begin{aligned}
\mathcal{L}_{\mathrm{CFM}}(\theta) &= \mathbb{E}_{t\sim\mathrm{Unif},\, z\sim p_{\mathrm{data}},\, x\sim\mathcal{N}(\alpha_t z,\beta_t^2 I_d)}\Big[\Big\| u_t^\theta(x)-\left(\dot{\alpha}_t-\frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z-\frac{\dot{\beta}_t}{\beta_t}x\Big\|^2\Big]\\
&= \mathbb{E}_{t\sim\mathrm{Unif},\, z\sim p_{\mathrm{data}},\, \epsilon\sim\mathcal{N}(0,I_d)}\big[\|u_t^\theta(\alpha_t z+\beta_t\epsilon)-(\dot{\alpha}_t z+\dot{\beta}_t\epsilon)\|^2\big]
\end{aligned}
$$

其中 (i) 代入了从高斯条件路径抽样的步骤，并把 $x$ 替换为 $\alpha_t z+\beta_t\epsilon$。注意 $\mathcal{L}_{\mathrm{CFM}}$ 有多简单：我们采样一个数据点 $z$，采样一些噪声 $\epsilon$，然后取均方误差。让我们在特殊情形 $\alpha_t=t$、$\beta_t=1-t$ 上把它写得更具体。相应的概率 $p_t(x|z)=\mathcal{N}(tz,(1-t)^2)$ 有时称为（高斯）**CondOT 概率路径（CondOT probability path）**。于是有 $\dot{\alpha}_t=1$、$\dot{\beta}_t=-1$，从而

$$
\mathcal{L}_{\mathrm{cfm}}(\theta) = \mathbb{E}_{t\sim\mathrm{Unif},\, z\sim p_{\mathrm{data}},\, \epsilon\sim\mathcal{N}(0,I_d)}\big[\|u_t^\theta(tz+(1-t)\epsilon)-(z-\epsilon)\|^2\big]
$$

许多著名的当前最优模型都用这个简单而有效的过程训练，例如 *Stable Diffusion 3*、Meta 的 *Movie Gen Video*，以及很可能还有更多专有模型。在后文的棋盘图中，我们用一个简单例子把它可视化，并在上面的流匹配训练过程中概括了训练步骤。

**图注.** 流匹配损失等价定理在高斯 CondOT 概率路径上的示意：模拟一个已经训练好的流匹配模型的常微分方程。数据分布是棋盘图案（右上）。上行：来自真值边缘概率路径 $p_t(x)$ 的直方图。下行：来自流匹配模型的样本直方图。可以看到，训练之后上行与下行是吻合的（相差不超过训练误差）。该模型用上面的流匹配训练过程训练。

让我们总结本节的结果。

**小结（流匹配）.** **流匹配训练就是学习边缘向量场 $u_t^{\mathrm{target}}$。** 为了构造它，我们选取一条条件概率路径 $p_t(x|z)$，满足 $p_0(\cdot|z)=p_{\mathrm{init}}$、$p_1(\cdot|z)=\delta_z$。接下来，我们找到一个条件向量场 $u_t^{\mathrm{target}}(x|z)$，使得它对应的流 $\psi_t^{\mathrm{target}}(x|z)$ 满足

$$
X_0\sim p_{\mathrm{init}} \quad \Rightarrow \quad X_t = \psi_t^{\mathrm{target}}(X_0|z) \sim p_t(\cdot|z),
$$

或者等价地，$u_t^{\mathrm{target}}$ 满足连续性方程。随后，由下式定义的边缘向量场

$$
u_t^{\mathrm{target}}(x) = \int u_t^{\mathrm{target}}(x|z)\frac{p_t(x|z)\,p_{\mathrm{data}}(z)}{p_t(x)}\,\mathrm{d} z,
$$

跟随边缘概率路径，即

$$
X_0\sim p_{\mathrm{init}},\quad \mathrm{d} X_t = u_t^{\mathrm{target}}(X_t)\,\mathrm{d} t \Rightarrow X_t\sim p_t\quad (0\leq t\leq 1).
$$

特别地，这条常微分方程满足 $X_1\sim p_{\mathrm{data}}$，因此 $u_t^{\mathrm{target}}$ “把噪声转换成数据”，正如我们所希望的。为了学习它，我们最小化条件流匹配损失

$$
\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t\sim\mathrm{Unif},\, z\sim p_{\mathrm{data}},\, x\sim p_t(\cdot|z)}\big[\|u_t^\theta(x)-u_t^{\mathrm{target}}(x|z)\|^2\big].
$$

用得最广的例子是高斯概率路径。对这一情形，公式变为

$$
\begin{aligned}
p_t(x|z) &= \mathcal{N}(x;\alpha_t z,\beta_t^2 I_d)\\
u_t^{\mathrm{flow}}(x|z) &= \left(\dot{\alpha}_t-\frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z+\frac{\dot{\beta}_t}{\beta_t}x\\
\mathcal{L}_{\mathrm{CFM}}(\theta) &= \mathbb{E}_{t\sim\mathrm{Unif},\, z\sim p_{\mathrm{data}},\, \epsilon\sim\mathcal{N}(0,I_d)}\big[\|u_t^\theta(\alpha_t z+\beta_t\epsilon)-(\dot{\alpha}_t z+\dot{\beta}_t\epsilon)\|^2\big]
\end{aligned}
$$

其中噪声调度 $\alpha_t,\beta_t\in\mathbb{R}$ 是连续可微、单调的函数，我们把它们选成满足 $\alpha_0=\beta_1=0$、$\alpha_1=\beta_0=1$（例如 $\alpha_t=t$、$\beta_t=1-t$）。

---

<!-- 来源文件：04.zh.md -->

## 4 分数函数（Score Functions）与分数匹配（Score Matching）

上一节我们说明了如何用流匹配（flow matching）训练一个流模型（flow model）。本节讨论扩散模型（diffusion models），并演示如何用分数匹配来训练它们。

### 4.1 条件分数函数（Conditional Score Function）与边缘分数函数（Marginal Score Function）

到目前为止，我们研究的核心对象是向量场（vector field）$u_t(x)$。扩散模型（song2020score, song2021sde）采取另一种视角，关注分数函数。因此，本节将用分数函数的语言重述我们在这里已经学到的内容，从而提供一种新的视角。设 $q(x)$ 为任意概率分布（probability distribution）。则 $q$ 的分数函数定义为 $\nabla\log q(x)$，也就是 $q$ 的对数似然（log-likelihood）关于 $x$ 的梯度（gradient）。分数有直观的含义：$\nabla\log q(x)$ 是关于对数似然的最速上升方向（direction of steepest ascent）。这一点如图 8 所示。

**图注.** 图 8：分数函数 $\nabla\log q(x)$ 的示意图，右图把它画成黑色行，左图是一般概率分布 $q(x)$。

让我们回到流匹配一节中条件概率路径（conditional probability path）$p_t(x|z)$ 与边缘概率路径（marginal probability path）$p_t(x)$ 的设定。于是我们可以等价地把条件分数函数定义为 $\nabla\log p_t(x|z)$，把边缘分数函数定义为 $\nabla\log p_t(x)$。与边缘向量场公式类似，边缘分数可以借助条件分数函数 $\nabla\log p_t(x|z)$ 表示为

$$
\nabla\log p_t(x)
=\int \nabla \log p_t(x|z)\frac{ p_t(x|z)p_{\mathrm{data}}(z)}{p_t(x)}\mathrm{d} z.
\qquad (38)
$$

因此，**条件分数与边缘分数之间的关系，类似于条件向量场（conditional vector field）与边缘向量场（marginal vector field）之间的关系**。注意，我们可以如下证明式 (38)：

$$
\nabla\log p_t(x) = \frac{\nabla p_t(x)}{p_t(x)}
=\frac{\nabla \int p_t(x|z)p_{\mathrm{data}}(z)\mathrm{d} z}{p_t(x)}
=\frac{\int \nabla p_t(x|z)p_{\mathrm{data}}(z)\mathrm{d} z}{p_t(x)}
=\int \nabla \log p_t(x|z)\frac{ p_t(x|z)p_{\mathrm{data}}(z)}{p_t(x)}\mathrm{d} z,
\qquad (39)
$$

其中我们把法则 $\partial_{y} \log y=1/y$ 与链式法则（chain rule）结合使用了两次。

**例 15（高斯概率路径（Gaussian Probability Path）的分数函数）.** 对高斯路径 $p_t(x|z)=\mathcal{N}(x;\alpha_t z,\beta_t^2 I_d)$，我们可以利用高斯概率密度（Gaussian probability density）的形式（见附录中的高斯密度公式）得到

$$
\nabla \log p_t(x|z) = \nabla\log \mathcal{N}(x;\alpha_t z,\beta_t^2 I_d) = -\frac{x-\alpha_t z}{\beta_t^2}.
\qquad (40)
$$

注意，高斯概率路径的分数函数是 $x$ 与 $z$ 的线性（linear）函数。条件向量场 $u_t(x|z)$ 也同样如此（见条件高斯向量场公式）。因此二者可以互相转换，如下一命题所示。

**命题 1（高斯概率路径的转换公式）.** 对高斯概率路径 $p_t(x|z)=\mathcal{N}(\alpha_t z,\beta_t^2 I_d)$，条件（相应地，边缘）向量场与条件（相应地，边缘）分数由下列恒等式相联系：

$$
u^{\text{target}}_t(x|z)=a_t\nabla\log p_t(x|z)+b_tx,\quad a_t=\left(\beta_t^2\frac{\dot{\alpha}_t}{\alpha_t}-\dot{\beta}_t\beta_t\right),\quad b_t=\frac{\dot{\alpha}_t}{\alpha_t}.
\qquad (41)
$$

$$
u^{\text{target}}_t(x)=a_t\nabla\log p_t(x)+b_tx.
\qquad (42)
$$

特别地，我们指出，条件（相应地，边缘）向量场可以从条件（相应地，边缘）分数恢复出来，反之亦然。

**证明.** 对条件向量场与条件分数，我们可以推导：

$$
\begin{aligned}
u^{\text{target}}_t(x|z)&=\left(\dot{\alpha}_t-\frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z+\frac{\dot{\beta}_t}{\beta_t}x \\
&=\left(\beta_t^2\frac{\dot{\alpha}_t}{\alpha_t}-\dot{\beta}_t\beta_t\right)\left(\frac{\alpha_tz-x}{\beta_t^2}\right)+\frac{\dot{\alpha}_t}{\alpha_t}x \\
&=\left(\beta_t^2\frac{\dot{\alpha}_t}{\alpha_t}-\dot{\beta}_t\beta_t\right)\nabla\log p_t(x|z)+\frac{\dot{\alpha}_t}{\alpha_t}x,
\end{aligned}
$$

其中在 (i) 处我们只做了一些代数运算。取积分之后，同一恒等式对边缘流向量场与边缘分数函数成立：

$$
\begin{aligned}
u^{\text{target}}(x) &= \int u^{\text{target}}_t(x|z)\frac{ p_t(x|z)p_{\mathrm{data}}(z)}{p_t(x)}\mathrm{d} z \\
&=\int\left[a_t\nabla\log p_t(x|z)+b_tx\right] \frac{ p_t(x|z)p_{\mathrm{data}}(z)}{p_t(x)}\mathrm{d} z \\
&= a_t\nabla\log p_t(x)+b_tx,
\end{aligned}
$$

其中在 (i) 处我们使用了式 (38)，以及后验密度（posterior density）积分为 $1$ 这一事实。

命题 1 令人瞩目，因为它说明：一旦我们学到了 $u^{\text{target}}_t$，也就学到了分数函数 $\nabla\log p_t(x)$，反之亦然。因此，许多扩散模型改为通过神经网络（neural network）来学习分数函数 $\nabla\log p_t(x)$。我们将在第 4.3 节讨论这一点。

**注 16（分数的重参数化）.** 高斯概率路径在式 (41) 中的重参数化（reparameterization）公式之所以可能，是因为两边（条件向量场与条件分数）都是 $x$ 与 $z$ 的线性函数。一旦我们取边缘（边缘向量场与边缘分数），两边都只是后验均值（posterior mean）$\mathbb{E}_{z|x}\left[z\right]$ 的线性重参数化。由此可知，任何能够恢复 $\mathbb{E}_{z|x}\left[z\right]$ 的量，反过来都可以用来恢复无条件向量场与分数（unconditional vector field and score）。进一步，从数值稳定性或训练稳定性的角度看，这样做甚至可能更可取。一个常见选择是后验均值本身，通常称为去噪器（denoiser）。形式上，我们把**条件去噪器与边缘去噪器（conditional and marginal denoiser）**定义为

$$
D_t(x|z)=z, \quad
D_t(x)
=\int z\,\frac{ p_t(x|z)p_{\mathrm{data}}(z)}{p_t(x)}\mathrm{d} z = \frac{1}{\dot{\alpha}_t\beta_t-\alpha_t\dot{\beta}_t}(\beta_t u^{\text{target}}_t(x_t)-\dot{\beta}_tx_t).
\qquad (43)
$$

这里，(i) 来自与命题 1 等价的推导。去噪器有非常直观的解释：它是给定噪声数据 $x$ 时干净数据 $z$ 的期望值。[^clean] 人们常把这类模型称为**去噪扩散模型（denoising diffusion models）**，因为学习 $D_t$ 与学习 $u^{\text{target}}_t$ 在理论上是等价的。

[^clean]: 供思考：去噪器是否总会输出一个「干净」的数据点？为什么会，或为什么不会，而这又可能取决于什么？

### 4.2 用随机微分方程（SDE）采样

**图注.** 图 9：定理 17 的示意图。用随机微分方程模拟一条概率路径。这是把条件路径与边缘路径的模拟图，改用式 (44) 的随机微分方程采样之后重画的结果。数据分布 $p_{\mathrm{data}}$ 为蓝色背景。高斯分布 $p_{\mathrm{init}}$ 为红色背景。上排：条件路径。下排：边缘概率路径。可以看到，该随机微分方程把来自 $p_{\mathrm{init}}$ 的样本输送成来自 $\delta_{z}$ 的样本（对条件路径），以及输送到 $p_{\mathrm{data}}$（对边缘路径）。

到目前为止，我们已经演示了如何借助边缘向量场 $u^{\text{target}}_t$，构造一条常微分方程（ODE）的轨迹（trajectory）$X_t$，使其跟随所期望的概率路径 $p_t$。但这一做法局限于流模型。扩散模型又如何呢？利用分数函数，现在让我们把这一结果推广到随机微分方程。

**定理 17（随机微分方程扩展技巧（SDE Extension Trick））.** 像前面一样定义条件向量场与边缘向量场 $u^{\text{target}}_t(x|z)$ 与 $u^{\text{target}}_t(x)$。于是，对任意扩散系数（diffusion coefficient）$\sigma_t\geq 0$，我们可以通过把随机动力学（stochastic dynamics）加到原始常微分方程的动力学上，构造如下随机微分方程：

$$
\begin{aligned}
X_0 &\sim p_{\mathrm{init}}, \\
\mathrm{d} X_t
  &= u^{\text{target}}_t(X_t)\mathrm{d} t + \frac{\sigma_t^2}{2}\nabla\log p_t(X_t)\mathrm{d} t
    + \sigma_t\mathrm{d} W_t \\
  &= \Big[u^{\text{target}}_t(X_t)+\frac{\sigma_t^2}{2}\nabla\log p_t(X_t)\Big]\mathrm{d} t
    + \sigma_t\mathrm{d} W_t.
\end{aligned}
\qquad (44)
$$

$$
\Rightarrow\quad X_t \sim p_t\quad (0\le t\le 1).
\qquad (45)
$$

特别地，对这个随机微分方程有 $X_1\sim p_{\mathrm{data}}$。我们指出，随机动力学与**朗之万动力学（Langevin dynamics）**密切相关，可以看作在保持边缘分布（marginal distribution）$p_t$ 的同时注入噪声。我们在注 20 中简要讨论朗之万动力学。

我们用图 9 说明定理 17 所描述的动力学。可以看到，轨迹现在呈锯齿状，体现了随机微分方程演化的随机本质。然而，正如定理 17 所确立的，边缘分布 $p_t$ 保持不变。注意，上述结果令人瞩目之处在于：即使在网络已经训练完成之后，我们仍可以选择*任意*扩散系数 $\sigma_t\geq 0$。理论上，定理 17 对 $\sigma_t$ 的任何选择都成立。然而，*在实践中*，我们同时受到训练误差（training error）（神经网络并不能完美逼近边缘向量场与分数）和仿真误差（simulation error）（例如当 $\sigma_t\gg 0$ 时，在扩散模型的采样算法中将不得不采用小到不可行的步长）的影响。实践中，对一个已经训练好的固定模型，于是存在一个可以凭经验确定的最优 $\sigma_t\geq 0$（karras2022elucidating, albergo2023stochastic, ma2024sit）。[^sigma]

[^sigma]: 我们再次强调，「最佳 $\sigma_t$」的存在，是模型训练得不完美以及计算预算有限所造成的人为产物，而不是关于这些动力学在其连续极限中的理论陈述。

对高斯概率路径，只要学到了边缘向量场，分数函数就可以免费得到。

**例 18（高斯情形的随机微分方程扩展技巧）.** 由命题 1，对高斯概率路径，我们可以纯粹用分数函数来写出定理 17 中的随机微分方程：

$$
X_0\sim p_{\mathrm{init}},\quad \mathrm{d} X_t =\left[\left(a_t+\frac{\sigma_t^2}{2}\right)\nabla\log p_t(X_t)+b_tX_t\right]\mathrm{d} t +\sigma_t\mathrm{d} W_t.
\qquad (46)
$$

$$
\Rightarrow X_t\sim p_t\quad (0\leq t\leq 1),
\qquad (47)
$$

其中 $a_t,b_t$ 的定义如命题 1。

在本节余下的部分，我们将借助**福克–普朗克方程（Fokker–Planck equation）**证明定理 17。该方程把连续性方程（continuity equation）从常微分方程推广到随机微分方程。为此，我们先经由下式，对标量场（scalar field）$w_t:\mathbb{R}^d\to\mathbb{R}$ 定义**拉普拉斯算子（Laplacian）** $\Delta$：

$$
\Delta w_t(x)=\sum_{i=1}^{d}\frac{\partial^2}{\partial x_i^2}w_t(x)=\mathrm{div}(\nabla w_t)(x),
\qquad (48)
$$

**定理 19（福克–普朗克方程）.** 设 $p_t$ 是一条概率路径，并考虑随机微分方程

$$
X_0\sim p_{\mathrm{init}}, \quad \mathrm{d} X_t = u_t(X_t)\mathrm{d} t + \sigma_t\mathrm{d} W_t.
$$

则对所有 $0\leq t\leq 1$，$X_t$ 以 $p_t$ 为分布，当且仅当福克–普朗克方程成立：

$$
\partial_t p_t(x) = -\mathrm{div} (p_t u_t)(x)+\frac{\sigma_t^2}{2}\Delta p_t (x)\quad x\in\mathbb{R}^d,\ 0\leq t\leq 1.
\qquad (49)
$$

福克–普朗克方程的一份自洽证明见附录中的相应证明。注意，当 $\sigma_t=0$ 时，从福克–普朗克方程可以恢复连续性方程定理。额外的拉普拉斯项 $\Delta p_t$ 一开始可能难以理解。熟悉物理的人会注意到，同一项也出现在热方程（heat equation）中（热方程事实上是福克–普朗克方程的一个特例）。热在介质中扩散。我们也加入一个扩散过程（diffusion process）（不是物理过程，而是数学过程），因此要加上这一额外的拉普拉斯项。现在让我们用福克–普朗克方程来帮助证明定理 17。

**定理 17 的证明.** 由定理 19，我们需要证明式 (44) 中定义的随机微分方程对 $p_t$ 满足福克–普朗克方程。我们可以直接计算：

$$
\begin{aligned}
\partial_t p_t(x) &= - \mathrm{div}(p_t u^{\text{target}}_t)(x) \\
&= - \mathrm{div}(p_t u^{\text{target}}_t)(x) -\frac{\sigma_t^2}{2}\Delta p_t(x)+\frac{\sigma_t^2}{2}\Delta p_t(x) \\
&= - \mathrm{div}(p_t u^{\text{target}}_t)(x) -\mathrm{div}\left(\frac{\sigma_t^2}{2}\nabla p_t\right)(x)+\frac{\sigma_t^2}{2}\Delta p_t(x) \\
&= - \mathrm{div}(p_t u^{\text{target}}_t)(x) -\mathrm{div}\left(p_t\left[\frac{\sigma_t^2}{2}\nabla \log p_t\right]\right)(x)+\frac{\sigma_t^2}{2}\Delta p_t(x) \\
&= -\mathrm{div}\left(p_t\left[u^{\text{target}}_t+\frac{\sigma_t^2}{2}\nabla \log p_t\right]\right)(x)+\frac{\sigma_t^2}{2}\Delta p_t(x),
\end{aligned}
$$

其中 (i) 使用了连续性方程定理，(ii) 加上并减去同一项，(iii) 使用了拉普拉斯算子的定义（式 (48)），(iv) 使用了 $\nabla\log p_t=\frac{\nabla p_t}{p_t}$，(v) 使用了散度（divergence）算子的线性性。上述推导表明，式 (44) 中定义的随机微分方程对 $p_t$ 满足福克–普朗克方程。由定理 19，这蕴含对 $0\leq t\leq 1$ 有 $X_t\sim p_t$，即为所求。

**图注.** 图 10：上排：粒子在式 (50) 给出的朗之万动力学下演化，其中 $p(x)$ 取为一个具有 5 个模态的高斯混合（Gaussian mixture）。下排：上排同一样本的核密度估计（kernel density estimate）。可以看到，样本的分布收敛到平衡分布（equilibrium distribution）$p$（蓝色背景）。

**注 20（可选：朗之万动力学）.** 上述构造有一个著名特例：概率路径是常数，即对某个固定分布 $p$ 有 $p_t=p$。此时我们令 $u^{\text{target}}_t=0$，得到随机微分方程

$$
\mathrm{d} X_t = \frac{\sigma_t^2}{2}\nabla\log p(X_t)\mathrm{d} t + \sigma_t \mathrm{d} W_t,
\qquad (50)
$$

它通常称为朗之万动力学。$p_t$ 为常数这一事实意味着 $\partial_t p_t(x)=0$。由定理 17 立即得到：这些动力学对定理 17 中的静态路径（static path）$p_t=p$ 满足福克–普朗克方程。因此我们可以断定，$p$ 是朗之万动力学的平稳分布（stationary distribution）：

$$
X_0 \sim p\quad \Rightarrow \quad X_t \sim p\quad (t\geq 0).
$$

与许多马尔可夫过程（Markov process）一样，这些动力学在相当一般的条件下收敛到平稳分布 $p$。也就是说，若我们改为取 $X_0 \sim p' \neq p$，从而 $X_t \sim p_t'$，则在温和条件下 $p_t \to p$。这一事实使朗之万动力学极其有用，并因此成为例如**分子动力学（molecular dynamics）**模拟，以及贝叶斯统计与自然科学中许多其他马尔可夫链蒙特卡罗（Markov chain Monte Carlo，MCMC）方法的基础。特别地，当 $p$ 为高斯分布时，奥恩斯坦–乌伦贝克过程（Ornstein–Uhlenbeck process）作为朗之万动力学的特例被恢复出来，并构成扩散模型最初表述的基础。

**注 21（可选：GLASS 流，用常微分方程实现随机演化）.** 随机微分方程采样（与常微分方程相比）的显著性质是，演化成为随机的，也就是说初始点 $X_0$ 并不能完全决定 $t>0$ 时的 $X_t$。或许令人惊讶的是，通过一种称为 *GLASS 流（GLASS Flows）* 的简单采样技巧（holderrieth2025glass），也可以纯粹借助常微分方程得到同样的随机转移。这使得我们可以利用随机微分方程的随机性（例如通过搜索算法），同时保持常微分方程的效率。

### 4.3 分数匹配

还需要说明我们如何能够学习边缘分数函数 $\nabla\log p_t(x)$。当然，对高斯概率路径，我们可以简单地按命题 1 变换 $u^{\text{target}}_t(x)$。但一般情况下又如何呢？事实表明，我们也可以直接学习边缘分数函数。为了逼近边缘分数 $\nabla\log p_t$，我们使用一个神经网络，称为**分数网络（score network）** $s_t^\theta:\mathbb{R}^d\times[0,1]\to\mathbb{R}^d$。与前面相同，我们可以设计**分数匹配损失（score matching loss）**与**去噪分数匹配损失（denoising score matching loss）**：

$$
\begin{aligned}
\mathcal{L}_{\text{SM}}(\theta) &=\mathbb{E}_{t\sim\text{Unif},\,z\sim p_{\mathrm{data}},\, x\sim p_t(\cdot|z)}\left[\left\|s_t^\theta(x) - \nabla\log p_t(x)\right\|^2\right] &&  \\
\mathcal{L}_{\text{CSM}}(\theta) &=\mathbb{E}_{t\sim\text{Unif},\,z\sim p_{\mathrm{data}},\, x\sim p_t(\cdot|z)}\left[\left\|s_t^\theta(x) - \nabla\log p_t(x|z)\right\|^2\right] && .
\end{aligned}
$$

其中差别同样在于：用的是边缘分数 $\nabla\log p_t(x)$，还是条件分数 $\nabla\log p_t(x|z)$。和以前一样，理想情况下我们希望最小化分数匹配损失，但做不到，因为我们并不知道 $\nabla\log p_t(x)$。但与以前类似，去噪分数匹配损失是一个可计算的替代方案：

**定理 22.** 分数匹配损失与去噪分数匹配损失相等，只相差一个常数：

$$
\mathcal{L}_{\text{SM}}(\theta) = \mathcal{L}_{\text{CSM}}(\theta) + C,
$$

其中 $C$ 与参数 $\theta$ 无关。因此它们的梯度一致：

$$
\nabla_\theta \mathcal{L}_{\text{SM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CSM}}(\theta).
$$

特别地，对极小化子 $\theta^*$，将有 $s_t^{\theta^*}=\nabla\log p_t$。

**证明.** 注意 $\nabla\log p_t$ 的公式（式 (38)）看起来与 $u^{\text{target}}_t$ 的公式（边缘向量场公式）相同。因此，把流匹配损失定理的证明中的 $u^{\text{target}}_t$ 替换为 $\nabla\log p_t$，证明就完全相同。

**例 23（去噪扩散模型：高斯概率路径的分数匹配）.** 让我们把去噪分数匹配损失具体写到 $p_t(x|z)=\mathcal{N}(\alpha_tz,\beta_t^2 I_d)$ 的情形。正如我们在式 (40) 中所推导的，条件分数 $\nabla\log p_t(x|z)$ 的公式为

$$
\nabla\log p_t(x|z) = -\frac{x-\alpha_t z}{\beta_t^2}.
\qquad (51)
$$

把这个公式代进去，条件分数匹配损失变为：

$$
\begin{aligned}
\mathcal{L}_{\text{CSM}}(\theta) &=\mathbb{E}_{t\sim\text{Unif},\,z\sim p_{\mathrm{data}},\,x\sim p_t(\cdot|z)}\left[\left\|s_t^\theta(x)+\frac{x-\alpha_t z}{\beta_t^2}\right\|^2\right] \\
&=\mathbb{E}_{t\sim\text{Unif},\,z\sim p_{\mathrm{data}},\,\epsilon\sim \mathcal{N}(0,I_d)}\left[\left\|s_t^\theta(\alpha_tz+\beta_t\epsilon)+\frac{\epsilon}{\beta_t}\right\|^2\right] \\
&=\mathbb{E}_{t\sim\text{Unif},\,z\sim p_{\mathrm{data}},\,\epsilon\sim \mathcal{N}(0,I_d)}\left[\frac{1}{\beta_t^2}\left\|\beta_ts_t^\theta(\alpha_tz+\beta_t\epsilon)+\epsilon\right\|^2\right],
\end{aligned}
$$

其中在 (i) 处我们代入了高斯路径的采样公式，并把 $x$ 替换为 $\alpha_tz+\beta_t\epsilon$。注意，网络 $s_t^\theta$ 本质上学习的是预测用来污染数据样本 $z$ 的噪声。这就解释了为什么上面的训练损失被称为去噪分数匹配。人们很快意识到，当 $\beta_t\approx 0$、接近于零时，上述损失在数值上是不稳定的（也就是说，只有当你加入足够多的噪声时，去噪分数匹配才有效）。因此，在关于去噪扩散模型的一些最早工作中（见**去噪扩散概率模型（Denoising Diffusion Probabilistic Models）**，ho2020denoising），有人提出丢掉损失中的常数 $\frac{1}{\beta_t^2}$，并通过下式把 $s_t^\theta$ 重参数化为**噪声预测器（noise predictor）**网络 $\epsilon_t^\theta:\mathbb{R}^d\times[0,1]\to\mathbb{R}^d$：

$$
-\beta_t s_t^\theta(x) = \epsilon_t^\theta(x)\quad \Rightarrow \quad \mathcal{L}_{\text{DDPM}}(\theta) =\mathbb{E}_{t\sim\text{Unif},\,z\sim p_{\mathrm{data}},\, \epsilon\sim \mathcal{N}(0,I_d)}\left[\|\epsilon_t^\theta(\alpha_tz+\beta_t\epsilon)-\epsilon\|^2\right].
$$

和以前一样，网络 $\epsilon_t^\theta$ 本质上学习的是预测用来污染数据样本 $z$ 的噪声。我们在算法 4 中总结这一训练过程。

**算法 4.** 高斯概率路径的分数匹配训练过程

**输入.** 样本数据集 $z\sim p_{\mathrm{data}}$，分数网络 $s_t^\theta$ 或噪声预测器 $\epsilon_t^\theta$。

对每个数据小批量（mini-batch）：

1. 从数据集中抽取一个数据样本 $z$。
2. 抽取随机时间 $t \sim \text{Unif}_{[0,1]}$。
3. 抽取噪声 $\epsilon\sim\mathcal{N}(0,I_d)$。
4. 令 $x_t=\alpha_t z + \beta_t\epsilon$。（一般情形：$x_t\sim p_t(\cdot|z)$。）
5. 计算损失

$$
\begin{aligned}
\mathcal{L}(\theta) &= \left\|s_t^\theta(x_t)+\frac{\epsilon}{\beta_t}\right\|^2 && =\left\|s_t^\theta(x_t)-\nabla\log p_t(x_t|z)\right\|^2 \\
\mathcal{L}(\theta) &= \left\|\epsilon_t^\theta(x_t)-\epsilon\right\|^2.
\end{aligned}
$$

6. 通过对 $\mathcal{L}(\theta)$ 做梯度下降（gradient descent）来更新模型参数 $\theta$。

让我们总结本节的结果：

**小结 24（分数函数、分数匹配与随机采样）.** 设 $p_t(x|z),p_t(x)$ 为条件概率路径与边缘概率路径。条件分数函数由 $\nabla\log p_t(x|z)$ 给出，边缘分数函数由 $\nabla\log p_t(x)$ 给出。对每一个扩散系数 $\sigma_t\geq 0$，下列随机微分方程的轨迹跟随该概率路径：

$$
X_0\sim p_{\mathrm{init}},\quad \mathrm{d} X_t =\left[u^{\text{target}}_t(X_t)+\frac{\sigma_t^2}{2}\nabla\log p_t(X_t)\right]\mathrm{d} t +\sigma_t\mathrm{d} W_t.
\qquad (52)
$$

$$
\Rightarrow X_t\sim p_t\quad (0\leq t\leq 1),
\qquad (53)
$$

其中 $u^{\text{target}}_t(x)$ 仍是如前所述的边缘向量场（见边缘向量场公式）。

**分数匹配.** 为了学习边缘分数函数 $\nabla\log p_t(x)$，我们可以使用分数网络 $s_t^\theta$，并通过去噪分数匹配来训练它：

$$
\mathcal{L}_{\text{CSM}}(\theta) = \mathbb{E}_{z\sim p_{\mathrm{data}}, \,t\sim \text{Unif},\,x\sim p_t(\cdot|z)}[\|s_t^\theta(x) - \nabla\log p_t(x|z)\|^2] \qquad .
\qquad (54)
$$

**高斯概率路径.** 对高斯概率路径这一最重要的情形，即 $p_t(x|z)=\mathcal{N}(x;\alpha_tz,\beta_t^2 I_d)$，不必把 $s_t^\theta$ 与 $u_t^\theta$ 分开训练，因为我们可以用下列公式把它们互相转换：

$$
u_t^\theta(x)=a_t s_t^\theta(x)+b_tx,\quad a_t=\left(\beta_t^2\frac{\dot{\alpha}_t}{\alpha_t}-\dot{\beta}_t\beta_t\right),\quad b_t=\frac{\dot{\alpha}_t}{\alpha_t}.
$$

训练之后，对任意扩散系数 $\sigma_t\geq 0$，我们可以模拟下列随机微分方程

$$
X_0\sim p_{\mathrm{init}},\quad \mathrm{d} X_t =\left[\left(1+\frac{\sigma_{t}^2}{2a_t}\right)u_t^\theta(X_t)-\frac{\sigma_{t}^2 b_t}{2a_t}X_t\right]\mathrm{d} t +\sigma_t\mathrm{d} W_t
\qquad (55)
$$

$$
=\left[\left(a_t+\frac{\sigma_{t}^2}{2}\right)s_t^\theta(X_t)+b_tX_t\right]\mathrm{d} t +\sigma_t\mathrm{d} W_t
\qquad (56)
$$

以得到近似样本 $X_1\sim p_{\mathrm{data}}$。可以凭经验找到最优的 $\sigma_t\geq 0$。

---

<!-- 来源文件：05.zh.md -->

# 5 引导（Guidance）：如何以提示（prompt）为条件

到目前为止，我们考虑过的生成模型都是**无引导**（unguided）的，例如一个图像模型只会生成某一幅图像。从数学上说，这意味着我们的模型返回的是**无条件数据分布**（unconditional data distribution）$p_{\mathrm{data}}(z)$ 的样本。然而在大多数情形中，我们的目标并不仅仅是生成一个任意对象，而是生成一个**以某些额外信息为条件**（conditioned）的对象。换句话说，我们希望**引导**（guide）模型去生成某一类对象。例如，可以设想一个图像生成模型：它接收文本提示（text prompt）$y$，然后生成一幅与该文本提示 $y$ 相符的图像 $x$。正如引言一节所讨论的，这意味着我们希望从 $p_{\mathrm{data}}(z|y)$ 中采样，也就是以 $y$ 为条件的**引导数据分布**（guided data distribution）。我们将在本节讨论这一点。

**注记 25（术语）。** 为避免记号和术语上的冲突——“条件（conditional）”一词被用来指对 $z \sim p_{\mathrm{data}}$ 做条件化（条件概率路径（conditional probability path）/向量场（vector field））——我们将使用术语**引导**，专门指对 $y$（例如文本提示）做条件化。

## 5.1 普通引导（Vanilla Guidance）

首先，我们讨论构建引导生成模型（guided generative model）的“标准”做法。简短的回答如下：我们只需在训练和推理时把输入提示 $y$ 提供给网络，其余一切都按以前的方式进行。下面把这一点形式化。我们把条件变量（conditioning variable）或提示 $y$ 看作取值于某个空间 $\mathcal{Y}$。例如，当 $y$ 对应于一条文本提示时，$\mathcal{Y}$ 就是所有文本构成的空间。当 $y$ 对应于某个离散的类别标签（class label）时，$\mathcal{Y}$ 就是离散的。我们对 $\mathcal{Y}$ 不作任何限制。

我们定义**引导扩散模型**（guided diffusion model）由一个**引导向量场**（guided vector field）$u_t^{\theta}(\cdot | y)$ 与一个含时扩散系数（diffusion coefficient）$\sigma_t$ 组成。前者由某个神经网络（neural network）参数化。二者合在一起由下式给出

$$
\begin{aligned}
&\ u^\theta: \mathbb{R}^d \times \mathcal{Y} \times [0,1] \to \mathbb{R}^d,\ (x,y,t) \mapsto u_t^{\theta}(x|y)\\
&\ \sigma_t: [0,1] \to [0,\infty),\ t \mapsto \sigma_t
\end{aligned}
$$

请注意它与前文随机微分方程生成模型小结的差别：我们还额外地用输入 $y\in \mathcal{Y}$ 来引导 $u_t^\theta$。对任意这样的 $y \in \mathcal{Y}$，可以按如下方式从该模型生成样本：

$$
\begin{aligned}
\quad X_0 &\sim p_{\mathrm{init}} && \blacktriangleright\ \\
\quad \mathrm{d} X_t &= u_t^\theta(X_t|y)\,\mathrm{d} t + \sigma_t\,\mathrm{d} W_t && \blacktriangleright\ t=0t=1\\
\quad X_1 &\sim p_{\mathrm{data}}(\cdot | y) && \blacktriangleright\ X_1p_{\mathrm{data}}(\cdot|y)
\end{aligned}
$$

当 $\sigma_t = 0$ 时，我们称这样的模型为引导流模型（guided flow model）。在下文中，为使叙述更简洁，我们只限于流匹配（flow matching）与流模型（flow model），但一切都同样适用于一般情形。

接下来我们讨论：如何训练引导流模型 $u_t^\theta(x|y)$？一个简单的办法或许是固定我们对 $y$ 的选取，并把数据分布取为 $p_{\mathrm{data}}(x|y)$。这样我们就回到了与以前相同的无引导生成问题，因而可以相应地用条件流匹配目标（conditional flow matching objective）构造生成模型，即

$$
\mathbb{E}_{z \sim p_{\text{data}}(\cdot|y),\ x \sim p_t(\cdot|z)} \lVert u_t^{\theta}(x|y) - u^{\text{target}}_t(x|z)\rVert^2.
\qquad (57)
$$

注意，标签 $y$ 并不影响条件概率路径 $p_t(\cdot|z)$，也不影响条件向量场（conditional vector field）$u^{\text{target}}_t(x|z)$（尽管原则上可以让它们依赖于 $y$）。把期望对所有这些 $y$ 的选取展开，我们就得到**引导条件流匹配目标**（guided conditional flow matching objective）

$$
\mathcal{L}_{\text{CFM}}^{\text{guided}}(\theta) = \mathbb{E}_{(z,y) \sim p_{\text{data}}(z,y),\,t\sim \text{Unif}[0,1],\,x\sim p_t(\cdot|z)} \lVert u_t^{\theta}(x|y) - u^{\text{target}}_t(x|z)\rVert^2.
\qquad (58)
$$

上文这个引导目标与前文无引导的条件流匹配目标有一个主要差别：这里采样的是 $(z,y) \sim p_{\mathrm{data}}$，而不只是 $z \sim p_{\mathrm{data}}$。原因在于，我们的数据分布现在原则上是一个联合分布（joint distribution），例如同时覆盖图像 $z$ 与文本提示 $y$。在实践中，这意味着上述引导条件流匹配目标的 PyTorch 实现会用到一个数据加载器（dataloader），它返回的批次**同时包含 $z$ 和 $y$**。

**图注.** 图 11：以提示/类别 $y=$“柯基犬”生成图像。左：用普通引导生成的样本，图像与提示吻合得不好。右：用分类器引导（classifier guidance）且 $w = 4$ 生成的样本。如图所示，无分类器引导（classifier-free guidance）提高了对提示的遵循程度。图取自文献 [18]。

**图注.** 图 12：分类器引导与无分类器引导的示意图。分类器引导把引导向量场 $u^{\text{target}}_t(x|y)$ 与分类器 $\log p_t(y|x)$ 的梯度分解开来，并用引导尺度（guidance scale）$w>1$ 放大该分类器。无分类器引导放大两个向量场之差，从而达到同样的效果，但不必训练一个单独的分类器模型。

## 5.2 无分类器引导

理论上，普通引导应当给出 $p_{\mathrm{data}}(\cdot|y)$ 的忠实生成过程。然而人们很快从经验上发现，用这一过程得到的图像样本与期望的标签 $y$ 吻合得不够好（见以“柯基犬”为提示的生成图）。这可能有多种原因：模型可能欠拟合（underfit）（也就是说，我们实际上并没有学到真正的边缘向量场（marginal vector field）），或者我们的数据可能并不完美（例如，来自万维网的文本–图像配对含有大量错误）。因此，要想真正生成与提示吻合得更好的样本，我们必须找到一种人为加强提示变量 $y$ 的办法。这样做的主要技术称为**无分类器引导**，它被广泛用于当前最先进的扩散模型（diffusion model），我们接下来就讨论它。

**分类器引导。** 为简单起见，这里我们聚焦于高斯概率路径（Gaussian probability path）的情形。回忆前文的高斯条件概率路径公式：一条高斯条件概率路径由 $p_t(\cdot|z) = \mathcal{N}(\alpha_t z,\beta_t^2 I_d)$ 给出，其中噪声调度（noise scheduler）$\alpha_t$ 与 $\beta_t$ 连续可微、单调，并且满足 $\alpha_0 = \beta_1 = 0$ 以及 $\alpha_1 = \beta_0 = 1$。此外，回忆我们可以用前文关于高斯概率路径转换公式的命题，借助引导分数函数（guided score function）$\nabla\log p_t(x|y)$，把引导向量场 $u^{\text{target}}_t(x|y)$ 改写成如下形式

$$
u^{\text{target}}_t(x|y) = a_t\nabla \log p_t(x|y)+b_tx,
\qquad (59)
$$

接下来注意到，$p_t(x|y)$ 是一个条件密度（conditional density）。因此我们可以用贝叶斯法则（Bayes' rule）把引导分数改写为

$$
p_t(x|y)=\frac{p_t(x)p_t(y|x)}{p_t(y)}
\qquad (60)
$$

$$
\nabla \log p_t(x|y) = \nabla \log \left(\frac{p_t(x)p_t(y|x)}{p_t(y)}\right) = \nabla \log p_t(x) + \nabla \log p_t(y|x),
\qquad (61)
$$

其中我们用到：梯度 $\nabla$ 是对变量 $x$ 取的，因而 $\nabla \log p_t(y) = 0$。于是我们可以改写

$$
u^{\text{target}}_t(x|y) = b_tx + a_t(\nabla \log p_t(x) + \nabla \log p_t(y|x)) = u^{\text{target}}_t(x) + a_t \nabla \log p_t(y|x).
$$

注意上述方程的形状：引导向量场 $u^{\text{target}}_t(x|y)$ 等于无引导向量场 $u^{\text{target}}_t(x)$ **加上**引导变量（guidance variable）$y$ 的似然（likelihood）$p_t(y|x)$ 的一个梯度。当人们观察到自己的图像 $x$ 与提示 $y$ 吻合得不够好时，一个自然的想法就是放大 $\nabla \log p_t(y|x)$ 这一项的贡献，从而得到

$$
\tilde{u}_t(x|y) = u^{\text{target}}_t(x) + w a_t \nabla \log p_t(y|x),\quad ()
\qquad (62)
$$

其中 $w > 1$ 称为**引导尺度**。如何学习 $\log p_t(y|x)$ 这一项？注意，这可以看作一种加噪数据（noised data）的分类器（也就是说，它给出给定 $x$ 时 $y$ 的对数似然（log-likelihood））。因此我们可以简单地用监督学习（supervised learning）来学习它。这就引出了**分类器引导** [11, 43]（示意图见分类器引导与无分类器引导的那幅图）。分类器引导在很大程度上已被无分类器引导取代，因此这里不再进一步讨论它。然而，正如我们下面将看到的，它构成了无分类器引导的基础。最后要注意，这是一种启发式（heuristic）：当 $w \neq 1$ 时，有 $\tilde{u}_t(x|y) \neq u^{\text{target}}_t(x|y)$，因此它并不是“真正的”引导向量场。

**无分类器引导。** 分类器引导原则上是可能的，但它带来一些困难。首先，我们需要在流模型或扩散模型之外再训练一个分类器，于是就有 2 个网络而不是 1 个。其次，如果 $y$ 是高维的，例如一条文本提示而不仅仅是一个类别，那么 $p_t(y|x)$ 可能非常难学，梯度 $\nabla\log p_t(y|x)$ 也很难得到。出于这一原因，人们引入了无分类器引导 [18]。无分类器引导能得到与分类器引导在理论上等价的效果，却不必训练一个单独的分类器。

为此，我们可以再次应用等式

$$
\nabla \log p_t(x|y) = \nabla \log p_t(x) + \nabla \log p_t(y|x)
$$

从而得到

$$
\begin{aligned}
\tilde{u}_t(x|y) &= u^{\text{target}}_t(x) + w a_t \nabla \log p_t(y|x)\\
&= u^{\text{target}}_t(x) + w a_t (\nabla \log p_t(x|y) - \nabla \log p_t(x))\\
&= u^{\text{target}}_t(x) - (w b_tx + w a_t \nabla \log p_t(x)) + (w b_t x + w a_t \nabla \log p_t(x|y))\\
&= (1-w) u^{\text{target}}_t(x) + w u^{\text{target}}_t(x|y).
\end{aligned}
$$

因此，我们可以把经过放大的引导向量场 $\tilde{u}_t(x|y)$ 表示为无引导向量场 $u^{\text{target}}_t(x)$ 与引导向量场 $u^{\text{target}}_t(x|y)$ 的线性组合。于是想法就是：既训练一个无引导的 $u^{\text{target}}_t(x)$（例如使用前文的条件流匹配目标），也训练一个有引导的 $u^{\text{target}}_t(x|y)$（例如使用上文的引导条件流匹配目标），然后在推理时把它们组合起来，得到 $\tilde{u}_t(x|y)$。“且慢！”你或许会问，“那我们不就得训练两个模型了吗？”事实表明，我们可以在同一个模型里把两者都训练出来：我们可以给标签集增加一个新的、额外的 $\varnothing$ 标签，表示**条件缺失**（absence of conditioning）。然后我们可以把 $u^{\text{target}}_t(x)=u^{\text{target}}_t(x|\varnothing)$。这样一来，就不需要再训练一个单独的模型，去加强某个假设中的分类器的效果。这种在一个模型中同时训练条件模型与无条件模型（并随后加强条件作用）的做法，称为**无分类器引导**（CFG）[18]（示意图见分类器引导与无分类器引导的那幅图）。

**注记 26（一般概率路径的推导）。** 注意，构造

$$
\tilde{u}_t(x|y) = (1-w) u^{\text{target}}_t(x) + w u^{\text{target}}_t(x|y),
$$

对概率路径的任意选取都同样成立，并不只限于高斯路径。当 $w=1$ 时，很容易验证 $\tilde{u}_t(x|y)=u^{\text{target}}_t(x|y)$。我们借助高斯路径所做的推导，只是为了说明这一构造背后的直觉，尤其是放大某个假设“分类器”$\nabla \log p_t(y|x)$ 之贡献这一直觉。

**训练与无分类器引导。** 现在我们必须修改上文的引导条件流匹配目标，以便计入 $y = \varnothing$ 的可能性。困难在于，当采样 $(z,y) \sim p_{\mathrm{data}}$ 时，我们永远不会得到 $y = \varnothing$。因此我们必须人为地引入 $y = \varnothing$ 的可能性。为此，我们定义某个超参数（hyperparameter）$\eta$，作为丢弃原始标签 $y$ 并将其替换为 $\varnothing$ 的概率。于是我们得到 **CFG 条件流匹配训练目标**（CFG conditional flow matching training objective）

$$
\mathcal{L}_{\text{CFM}}^{\text{CFG}}(\theta) = \mathbb{E}_{\mathrm{sample}} \lVert u_t^{\theta}(x|y) - u^{\text{target}}_t(x|z)\rVert^2
\qquad (63)
$$

$$
\mathrm{sample} = (z,y) \sim p_{\text{data}}(z,y),\, t \sim \text{Unif}[0,1],\, x \sim p_t(\cdot|z),\ \eta,\;y\gets\varnothing
\qquad (64)
$$

**算法 5.** 针对高斯概率路径 $p_t(x|z)=\mathcal{N}(x;\alpha_t z,\beta_t^2 I_d)$ 的无分类器引导训练

**要求：** 成对数据集 $(z,y)\sim p_{\mathrm{data}}$，神经网络 $u_t^\theta$

1. 对每个数据小批量（mini-batch）循环：
2. 从数据集中采样一个数据样本 $(z,y)$。
3. 采样一个随机时间 $t \sim \text{Unif}_{[0,1]}$。
4. 采样噪声 $\epsilon\sim\mathcal{N}(0,I_d)$。
5. 令 $x=\alpha_t z + \beta_t \epsilon$。
6. 以概率 $p$ 丢弃标签：$y\leftarrow \varnothing$。
7. 计算损失

$$
\mathcal{L}(\theta) = \lVert u_t^\theta(x|y)-(\dot{\alpha}_t z+\dot{\beta}_t\epsilon)\rVert^2
$$

8. 通过对 $\mathcal{L}(\theta)$ 做梯度下降（gradient descent）来更新模型参数 $\theta$。
9. 结束循环。

我们把上述结论总结如下。

**总结 27（流模型的无分类器引导）。** 给定无引导边缘向量场 $u^{\text{target}}_t(x|\varnothing)$、引导边缘向量场 $u^{\text{target}}_t(x|y)$，以及**引导尺度** $w > 1$，我们把**无分类器引导向量场**（classifier-free guided vector field）$\tilde{u}_t(x|y)$ 定义为

$$
\tilde{u}_t(x|y) = (1-w) u^{\text{target}}_t(x|\varnothing) + w u^{\text{target}}_t(x|y).
\qquad (65)
$$

通过用同一个神经网络近似 $u^{\text{target}}_t(x|\varnothing)$ 与 $u^{\text{target}}_t(x|y)$，我们可以使用下面的**无分类器引导条件流匹配**（classifier-free guidance CFM，CFG-CFM）目标

$$
\mathcal{L}_{\text{CFM}}^{\text{CFG}}(\theta) = \mathbb{E}_{\mathrm{sample}} \lVert u_t^{\theta}(x|y) - u^{\text{target}}_t(x|z)\rVert^2
\qquad (66)
$$

$$
\mathrm{sample} = (z,y) \sim p_{\text{data}}(z,y),\, t \sim \text{Unif}[0,1],\, x \sim p_t(\cdot|z),\ \eta,\;y\gets\varnothing
\qquad (67)
$$

用直白的话说，$\mathcal{L}_{\text{CFM}}^{\text{CFG}}$ 可以近似为

$$
\begin{aligned}
(z,y) &\sim p_{\mathrm{data}}(z,y)\\
t &\sim \text{Unif}[0,1)\\
x &\sim p_t(x|z)\\
\eta,\ y &\gets \varnothing\\
\hat{\mathcal{L}_{\text{CFM}}^{\text{CFG}}(\theta)} &= \lVert u_t^{\theta}(x|y) - u^{\text{target}}_t(x|z)\rVert^2
\end{aligned}
$$

在推理时，对固定选取的 $y$，我们可以按如下方式采样

$$
\begin{aligned}
\quad X_0 &\sim p_{\mathrm{init}}(x) && \blacktriangleright\ \\
\quad \mathrm{d} X_t &= \tilde{u}_t^\theta(X_t|y)\,\mathrm{d} t && \blacktriangleright\ t=0t=1\\
\quad X_1 & && \blacktriangleright\ X_1y
\end{aligned}
$$

注意，如果使用权重 $w>1$，$X_1$ 的分布就不再必然与 $X_1 \sim p_{\mathrm{data}}(\cdot | y)$ 一致。然而从经验上看，这样做与条件化对齐得更好。因此，无分类器引导是一种**启发式**，其依据主要是出色的经验结果。事实上，你所看到的几乎任何由人工智能生成的图像或视频，都严重依赖 $w\geq 4$ 的无分类器引导。在以“柯基犬”为提示的生成图中，我们展示了 128×128 ImageNet 上基于类别的无分类器引导，做法与文献 [18] 相同。类似地，在 MNIST 手写数字的引导效果图中，我们展示了把无分类器引导用于从手写数字数据集 MNIST 采样时，不同引导尺度 $w$ 所带来的影响。

**图注.** 图 13：在手写数字数据集 MNIST 上，以不同引导尺度施加无分类器引导的效果。左：引导尺度设为 $w = 1.0$。中：引导尺度设为 $w = 2.0$。右：引导尺度设为 $w = 4.0$。你将在第三次实验中自己生成一幅类似的图像！

**注记 28（扩散模型的引导）。** 把上述讨论从流模型推广到扩散模型是直接的。只需把 $u_t^\theta(x|y)$ 替换为 $\tilde{u}_t^\theta(x|y)$，并像分数函数与分数匹配一节中讨论的那样，用随机微分方程进行采样。

---

<!-- 来源文件：06.zh.md -->

# 6 构建大规模图像或视频生成器

在前面各节中，我们学习了如何训练一个流匹配（flow matching）或扩散模型（diffusion model），以便从分布 $p_{\mathrm{data}}(x|y)$ 中采样。这一做法是通用的，可以应用到多种不同的数据类型与应用上。在本节中，我们深入考察大规模图像与视频生成这一具体情形，并包括若干知名模型，例如 *FLUX 2.0、Stable Diffusion 3、Nano Banana* 以及 *VEO-3 或 Meta Movie Gen Video*。最后，我们将把迄今所学应用到实验中，从零开始构建我们自己的此类模型！本节大致安排如下：

1. **神经网络架构（neural network architecture）：** 我们首先讨论如何把原始条件输入，包括时间 $t$ 以及引导变量（guidance variable）$y_{\mathrm{raw}}$（即离散的类别标签（class label）或原始文本），转换或**嵌入（embedding）**为模型 $u_t^\theta(x|y)$ 本身能够处理的向量值形式。随后我们讨论 $u_t^\theta(x|y)$ 的常见架构选择，包括 **U-Net** 与**扩散 Transformer（diffusion transformer，DiT）**。
2. **潜空间（latent space）：** 我们讨论**变分自编码器（variational autoencoder，VAE）**，它使生成建模得以在更低维的潜空间中进行，从而能够生成超高分辨率图像。
3. **案例研究：** 最后，我们将深入考察上面提到的两个最先进的图像与视频模型——*Stable Diffusion* 与 *Meta MovieGen*——以便让你体会在大规模情形下事情是如何完成的。

## 6.1 神经网络架构

让我们首先把注意力转向面向类图像模态（例如图像与视频）的流模型（flow model）与扩散模型所用的可扩展神经网络架构设计。具体而言，我们将探讨带参数 $\theta$ 的（有引导的）向量场（vector field）$u_t^\theta(x|y)$ 这一任务在实践中如何实现。注意，该神经网络必须有 3 个输入：向量 $x\in\mathbb{R}^d$、条件变量（conditioning variable）$y\in\mathcal{Y}$，以及时间值 $t\in [0,1]$；并有 1 个输出，即向量 $u_t^\theta(x|y)\in\mathbb{R}^d$。对于低维分布（例如前面各节见过的玩具分布），把 $u_t^\theta(x|y)$ 参数化为多层感知机（multi-layer perceptron，MLP），亦即全连接神经网络（fully connected neural network），便已足够。也就是说，在这一简单设定下，$u_t^\theta(x|y)$ 的一次前向传播是把输入 $x$、$y$ 与 $t$ 拼接起来，再送入一个多层感知机。然而，对于复杂的高维分布，例如图像、视频与蛋白质上的分布，多层感知机很可能不够，通常会使用专门的、针对应用的架构。在本小节余下的部分，我们将考虑**图像**（并由此推广到视频）的情形。首先，我们将考虑如何把原始条件信息——时间 $t$ 与条件变量 $y$——嵌入为实际模型能够处理的向量值形式。其次，我们将考虑此类模型的两种常见架构选择：U-Net [38, 17, 22, 11]，以及扩散 Transformer [12, 30, 28]。

### 6.1.1 嵌入条件变量

#### 嵌入时间

对于简单的玩具模型，把 $t$ 的原始值拼接到输入上，就足以训练出一个性能尚可的网络。在实践中，标量时间常常用傅里叶特征（Fourier feature）嵌入到更高维的空间中，使模型能够更忠实地捕捉高频的时间依赖性 [46]。显式地，该特征化由下式给出

$$
\mathrm{TimeEmb}(t) = \sqrt{\frac{2}{d}}\begin{bmatrix} \cos(2\pi w_1 t) & \cdots & \cos(2\pi w_{d/2} t) & \sin(2\pi w_1 t) & \cdots & \sin(2\pi w_{d/2} t) \end{bmatrix}^{T},
\qquad (68)
$$

其中频率 $w_i$ 按如下方式设定

$$
w_i = w_{\min}\left(\frac{w_{\max}}{w_{\min}}\right)^{\frac{i-1}{d/2-1}}, \qquad i=1,\ldots,d/2.
\qquad (69)
$$

这一 $\mathrm{TimeEmb}$ 的选择是标准做法，但这一精确形式并非严格必要。更确切地说，上面只是得到维数为 $d$ 的单位范数嵌入的一种方便方式，即 $\|\mathrm{TimeEmb}(t)\|=1$（因为 $\sin^2+\cos^2=1$）。

#### 嵌入类别标签

当 $y_{\mathrm{raw}} \in \mathcal{Y} \triangleq \{0,\cdots, N\}$ 只是一个类别标签时，通常最简单的做法是为 $y_{\mathrm{raw}}$ 的 $N+1$ 个可能取值各自学习一个单独的嵌入向量，并把 $y$ 设为该嵌入向量。人们会把这些嵌入的参数视为包含在 $u_t^\theta(x|y)$ 的参数之中，因此会在训练期间学习它们。

#### 嵌入文本输入

当 $y_{\mathrm{raw}}$ 是文本提示（text prompt）时，情况更为复杂，各种做法在很大程度上依赖于冻结的预训练模型。这类模型被训练来把离散文本输入嵌入为一个捕捉相关信息的连续向量。其中一个这样的模型称为 CLIP（Contrastive Language-Image Pre-training，对比语言–图像预训练）。CLIP 被训练来为图像与文本提示学习一个共享嵌入空间（shared embedding space），其训练损失旨在鼓励图像嵌入靠近其对应的提示，同时远离其他图像与提示的嵌入 [34]。因此我们可以取 $y = \mathrm{CLIP}(y_{\mathrm{raw}}) \in \mathbb{R}^{d_{\mathrm{CLIP}}}$ 为冻结的预训练 CLIP 模型所产生的嵌入。在某些情形下，把整个序列压缩成单一表示可能并不理想。此时，还可以考虑用一个预训练 Transformer 来嵌入提示，从而得到一列嵌入。在做条件化时，把多个这样的预训练嵌入组合起来也很常见，以便同时获得每个模型的好处 [14, 33]。就我们的目的而言，可以简单地假设在施加这样一个模型之后，提示嵌入的形状为

$$
\mathrm{PromptEmbed}(y_{\mathrm{raw}})\in \mathbb{R}^{S\times k}
$$

**图注.** 图 14。左：扩散 Transformer 架构的概览，取自 [30]。右：对比式 CLIP 损失的示意图，其中学习一个共享的图像–文本嵌入空间，取自 [34]。

### 6.1.2 扩散 Transformer

在深入这些架构的细节之前，让我们回顾导论中的一点：一幅图像只不过是一个向量 $x \in \mathbb{R}^{C_{\mathrm{image}} \times H \times W}$。这里 $C_{\mathrm{image}}$ 表示通道（channel）数（一幅 RGB 图像通常有 $C_{\mathrm{input}} = 3$ 个颜色通道），$H$ 与 $W$ 分别表示图像以像素计的高度（height）与宽度（width）。一类尤其突出的架构是所谓的扩散 Transformer 及其变体，它们使用注意力（attention）机制来构造网络 [49, 30, 28]。扩散 Transformer 有不同的风格。我们在这里解释一种通用设计，不过需要指出，扩散 Transformer 的具体实例可能因模型与应用而有所不同。在本节余下部分，我们用 $d$ 表示隐藏维度（hidden dimension），用 $L$ 表示 Transformer 层数，用 $h$ 表示每层的头数。扩散 Transformer 基于视觉 Transformer（vision transformer，ViT），其主要想法本质上是把一幅图像划分成图块（patch），嵌入这些图块以得到一列词元（token），再经由标准注意力处理所得词元 [13]。在末尾施加一次最终的反图块化（depatchification）运算，以恢复形状正确的图像。初始的图块化（patchification）运算只是对图像张量（tensor）$x\in \mathbb{R}^{C\times H\times W}$ 的重新排列：

$$
\mathrm{Patchify}(x)\in \mathbb{R}^{N\times C'}
$$

其中 $C'=CP^2$，$N=(H/P)\cdot(W/P)$，$P$ 为图块大小。接下来，我们对输出施加一个线性变换，得到最终的图块嵌入

$$
\mathrm{PatchEmb}(x)=\mathrm{Patchify}(x)W \in \mathbb{R}^{N\times d}
$$

其中 $W\in \mathbb{R}^{C'\times d}$ 是可学习的权重矩阵。扩散 Transformer 的输入于是是时间嵌入、提示嵌入，以及图块化后的图像张量，由下式给出（见关于嵌入条件变量的小节）：

$$
\begin{aligned}
\tilde{t} &=\mathrm{TimeEmb}(t)\in \mathbb{R}^{d}\\
\tilde{y}&=\mathrm{PromptEmb}(y)\in\mathbb{R}^{S \times d}\\
\tilde{x}_{0}&=\mathrm{PatchEmb}(x)\in\mathbb{R}^{N\times d}
\end{aligned}
$$

注意，现在所有元素都具有 Transformer 所期望的隐藏维度。随后，扩散 Transformer 对 $i=0,\cdots,L-1$ 通过 **DiT 块（DiTBlock）** 中的 Transformer 层迭代更新 $\tilde{z}_i$（细节见关于 DiT 块的注记）：

$$
\tilde{x}_{i+1}=\mathrm{DiTBlock}(\tilde{x}_i,\tilde{t},\tilde{y})\in\mathbb{R}^{N\times d}\quad (i=0,\cdots,L-1).
\qquad (70)
$$

其中 $N$ 是层数。最后，一步最终运算施加反图块化，把扩散 Transformer 的输出映射回所期望的输出形状：

$$
u=\mathrm{Depatchify}(\tilde{x}_N\tilde{W})\in \mathbb{R}^{C\times H\times W},
$$

其中 $\tilde{W}\in \mathbb{R}^{d\times C'}$。最终张量 $u$ 随后作为模型的输出，以及预测速度（velocity）$u_t^\theta(x|y)$。

**注记 29（DiT 块）.** 为完整起见，我们给出单个 DiT 层的简要数学描述。虽然我们试图纳入足够的细节，以便对扩散 Transformer 模型族有一个总体理解，但我们提醒读者，这些讲义选择强调关键的算法选择，而不是架构细节。现在，令 $x\in\mathbb{R}^{N\times d}$ 表示当前的图块词元序列（这里 $x=\tilde{x}_i$），并令 $y\in\mathbb{R}^{S \times d}$ 表示嵌入后的引导变量（这里 $y=\tilde{y}$）。于是，一个典型的 DiT 块用 (i) 图块上的自注意力（self-attention），(ii) 对提示的交叉注意力（cross-attention），以及 (iii) 经由自适应归一化（adaptive normalization，AdaLN）的时间条件化，来更新 $x$。

**缩放点积注意力（scaled dot-product attention）.** 给定查询（query）$Q\in\mathbb{R}^{N\times d_h}$、键（key）$K\in\mathbb{R}^{M\times d_h}$ 与值（value）$V\in\mathbb{R}^{M\times d_h}$，

$$
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V \in\mathbb{R}^{N\times d_h},
$$

其中 softmax 按行施加。

**多头注意力（multi-head attention）.** 令 $h$ 表示头数，$d_h=\frac{d}{h}$ 为每个头的维度。对每个头 $h\in\{1,\cdots,n_{\mathrm{heads}}\}$，学习投影矩阵 $W_Q^{(h)},W_K^{(h)},W_V^{(h)}\in\mathbb{R}^{k\times d_h}$。定义

$$
\mathrm{head}_h(x,z) = \mathrm{Attn}\!\big(xW_Q^{(h)},\,zW_K^{(h)},\,zW_V^{(h)}\big),
$$

其中源序列（source sequence）$z$ 或者是

$$
z=x \quad ,\qquad z=y \quad .
$$

把各个头拼接起来，并施加一个输出投影 $W_O\in\mathbb{R}^{d\times d}$：

$$
\mathrm{MultiHeadattention}(x,z) = \mathrm{Concat}\big(\mathrm{head}_1(x,z),\cdots,\mathrm{head}_{h}(x,z)\big)\,W_O \in\mathbb{R}^{N\times d}.
$$

**经由自适应归一化的时间条件化.** 令 $\tilde{t}\in\mathbb{R}^d$ 为时间步嵌入。扩散 Transformer 中的一个标准选择是用 $\tilde{t}$ 产生逐通道的缩放与平移参数，用以调制归一化后的激活 [31]。具体地，令 $g:\mathbb{R}^d\to\mathbb{R}^{2d}$ 为一个多层感知机，并令

$$
(\gamma,\beta) = g(\tilde{t}),
$$

其中 $\gamma,\beta\in\mathbb{R}^{d}$（或者，视实现而定，对注意力与多层感知机等不同子层使用各自的 $(\gamma,\beta)$ 对）。给定词元矩阵 $x\in\mathbb{R}^{N\times d}$ 以及一个归一化算子 $\mathrm{Norm}(\cdot)$（例如层归一化（LayerNorm）），定义调制后的归一化

$$
\mathrm{AdaNorm}_{\tilde{t}}(x) = \bigl(1+\gamma\bigr)\odot \mathrm{Norm}(H) + \beta,
$$

其中 $\odot$ 表示沿词元维广播（broadcasting）的逐元素乘法。

**合在一起.** 合并后的运算，从而也就是 DiT 块，由下式给出。

$$
\begin{aligned}
x &\gets x + g_{\mathrm{self}}(\tilde{t})\odot \mathrm{MultiHeadattention}\!\big(\mathrm{AdaNorm}_{\tilde{t}}(x),\,\mathrm{AdaNorm}_{\tilde{t}}(x)\big)\\
x &\gets x + g_{\mathrm{cross}}(\tilde{t})\mathrm{MultiHeadattention}\!\big(\mathrm{AdaNorm}_{\tilde{t}}(x),\,y\big)\\
x &\gets x + g_{\mathrm{MLP}}(\tilde{t})\mathrm{MLP}\!\big(\mathrm{AdaNorm}_{\tilde{t}}(x)\big),
\end{aligned}
$$

其中多层感知机是逐位置的前馈网络（position-wise feed-forward network），而 $g_{\cdots}$ 是可学习的门控参数（gating parameter）。输出 $x\in \mathbb{R}^{N\times d}$ 成为下一层的图块词元序列（按我们的记号，即 $\tilde{x}_{i+1}$）。最后我们指出，类别条件的扩散 Transformer，例如实验中实现的那个，通常更为简单，并且省去交叉注意力层，改为使用基于时间与类别的自适应归一化条件。

### 6.1.3 U-Net

U-Net 架构 [38] 是扩散 Transformer 架构的一种替代，并且是一种特定类型的卷积神经网络（convolutional neural network）。它最初为图像分割（image segmentation）而设计，其关键特征是输入与输出都具有图像的形状（通道数可以不同）。这使它非常适合用来参数化向量场 $x\mapsto u_t^\theta(x|y)$，因为在 $y,t$ 固定时，其输入具有图像的形状，输出亦然。因此，U-Net 在扩散模型的早期文献中被广泛使用 [17, 22, 11]。一个 U-Net 由一系列编码器（encoder）$\mathcal{E}_i$、与之对应的一列解码器（decoder）$\mathcal{D}_i$，以及中间的一个潜变量处理块组成，我们将该块称为中层编码块（midcoder）。[^midcoder] 作为例子，让我们走一遍一幅图像 $x_t \in \mathbb{R}^{3 \times 256 \times 256}$（我们取 $(C_{\mathrm{input}}, H, W) = (3, 256, 256)$）在被 U-Net 处理时所经过的路径：

$$
\begin{aligned}
x^{\mathrm{input}}_t &\in \mathbb{R}^{3 \times 256 \times 256} && \blacktriangleright\ \\
x^{\mathrm{latent}}_t = \mathcal{E}(x^{\mathrm{input}}_t) &\in \mathbb{R}^{512 \times 32 \times 32} && \blacktriangleright\ \\
x^{\mathrm{latent}}_t = \mathcal{M}(x^{\mathrm{latent}}_t) &\in \mathbb{R}^{512 \times 32 \times 32} && \blacktriangleright\ \\
x^{\mathrm{output}}_t = \mathcal{D}(x^{\mathrm{latent}}_t) &\in \mathbb{R}^{3 \times 256 \times 256} && \blacktriangleright\ 
\end{aligned}
$$

注意，当输入经过编码器时，其表示中的通道数增加，而图像的高度与宽度减小。编码器与解码器通常都由一系列卷积层组成（中间夹有激活函数、池化运算等）。上面没有画出两点：第一，输入 $x^{\mathrm{input}}_t\in \mathbb{R}^{3 \times 256 \times 256}$ 常常先送入一个初始的预编码块，以增加通道数，然后再送入第一个编码器块。第二，编码器与解码器之间常常由残差连接（residual connection）相连。完整图景见 U-Net 架构图。

**图注.** 图 15。一个简化的 U-Net 架构（2025 年版本本课程的实验 03 使用了类似这样的架构）。

从高层来看，大多数 U-Net 都包含上述做法的某种变体。然而，上面描述的某些设计选择，在实践中完全可能与各种实现有所不同。具体而言，我们上面选择的是纯卷积架构，而在编码器与解码器各处同时加入注意力层也很常见。U-Net 得名于其编码器与解码器所形成的“U”形（见 U-Net 架构图）。

[^midcoder]: 中层编码块是这里使用的一个完全非标准的术语，用来指 U-Net 堆叠最底部的部分，并与编码器和解码器相类比。

## 6.2 在潜空间中工作：（变分）自编码器

到目前为止，我们一直在数据空间 $\mathbb{R}^d$ 中操作。然而，随着分辨率越来越高，直接在这样一个空间里建模的代价很快就会高得难以承受。例如，一幅 $1024 \times 1024$、带三个 RGB 颜色通道的图像，对应的总维数是 $d=H\cdot W\cdot 3\approx 3*10^{6}$！注意，对视频而言维数还会进一步增加，因为一切都随帧数 $T$ 缩放。可以想象，在这样一个空间上训练很快就变得不可行。与图像分类不同——图像分类的低维输出允许把卷积堆叠逐渐变窄——我们基于流的建模方法要求输出 $u_t^\theta(x)\in\mathbb{R}^d$ 与输入一样大。于是重要的问题变成：**如何在合理的内存与计算预算内对高维图像建模？**

### 6.2.1 标准自编码器

这个问题的一个自然答案在于压缩（compression）：也许图像的实际空间，例如，位于高维图像空间中一个维数低得多的流形（manifold）附近。更具体地，我们可以考虑一个编码器 $\mu_{\phi}: \mathbb{R}^d \to \mathbb{R}^k$，以及某个解码器 $\mu_{\theta}: \mathbb{R}^k \to \mathbb{R}^d$，二者分别把原始图像 $x \in \mathbb{R}^d$ 映射到潜变量 $z\in\mathbb{R}^k$，以及从潜变量映射回来。维数 $k$ 通常选得远小于 $d$。对于图像，例如 $d = 3 \times 1024 \times 1024$ 时，下采样到例如 $k = 3 \times \tfrac{1024}{16} \times \tfrac{1024}{16}$ 并不少见。合在一起，$\mu_{\phi}$ 与 $\mu_{\theta}$ 被称为自编码器（autoencoder）。理想情况下，选择 $\mu_{\phi}$ 与 $\mu_{\theta}$ 是为了达到较高的重构质量，换句话说，使得 $\mu_{\theta}(\mu_{\phi}(x))$ 平均而言像 $x$。因此，自编码器通常用重构损失（reconstruction loss）来训练

$$
\mathcal{L}_{\mathrm{Recon}}(\phi,\theta)=\mathbb{E}_{x\sim p_{\mathrm{data}}}\left[\|\mu_\theta(\mu_\phi(x))-x\|^2\right].
$$

它度量原始数据点 $x$ 与重构点 $\mu_\theta(\mu_\phi(x))$ 之间的平方误差。

#### 对生成建模的适宜性

不幸的是，上面的重构损失不足以训练出一个“好”的自编码器。回忆一下，我们最终的目标是在潜空间中训练一个生成模型，其目标是由 $z = \mu_{\phi}(x)$、$x \sim p_{\mathrm{data}}$ 给出的潜分布（latent distribution）$p_{\mathrm{latent}}(z)$。$p_{\mathrm{data}}(x)$ 的一个生成模型随后这样实现：把潜空间生成模型的输出送过解码器 $\mu_{\theta}$。按我们目前的表述，自编码器有一个微妙的问题：我们对 $p_{\mathrm{latent}}(z)$ 几乎没有控制，因此本质上无法保证 $p_{\mathrm{latent}}(z)$ 甚至表现得足够好，以至于适宜用来训练这样一个生成模型（也就是说，良好、简单、像高斯分布）。虽然在潜空间中变换数据可能已经把它压缩了，但我们也可能把数据分布 $p_{\mathrm{data}}$ 变成了一个非常难学的分布 $p_{\mathrm{latent}}$。因此问题是：如何确保潜分布 $p_{\mathrm{latent}}$ 仍然表现良好且易于学习？为了能更显式地正则化潜分布，我们现在把自编码器的概念放进一个更一般的概率框架中重新表述，从而得到变分自编码器的概念。

### 6.2.2 变分自编码器

变分自编码器由我们（确定性的）标准自编码器表述得到，做法是放宽编码器与解码器必须是确定性函数这一约束。具体地，让我们考虑带参数 $\phi$ 的编码器 $q_\phi(z|x)$，以及带参数 $\theta$ 的解码器 $p_\theta(x|z)$。最常见的选择是取

$$
q_\phi(z|x)=\mathcal{N}(z;\mu_\phi(x),\mathrm{diag}(\sigma_\phi^2(x))),\quad p_\theta(x|z)=\mathcal{N}(x;\mu_\theta(z),\sigma_\theta^2(z)I_d)
\qquad (71)
$$

其中 $\mu_\phi(x)\in \mathbb{R}^k$，$\sigma_\phi^2(x)\in\mathbb{R}_{\geq 0}^k$，$\mu_\theta(z)\in \mathbb{R}^d$，以及 $\sigma_\theta^2(z)\in\mathbb{R}_{\geq 0}$ 被参数化为神经网络，$\mathrm{diag}$ 表示对角矩阵。要编码或解码一个变量，我们采样

$$
\begin{aligned}
z &\sim q_\phi(\cdot|x) &\quad& () \\
x &\sim p_\theta(\cdot|z) &\quad& ()
\end{aligned}
$$

最后我们指出，当恒有 $\sigma_\phi(x) = 0$ 且 $\sigma_\theta(x) = 0$ 时，我们就恢复出一个标准自编码器。让我们考察重构损失是什么样子。一个自然的目标是下面这个：

$$
\mathcal{L}_{\mathrm{VAE\text{-}Recon}}(\phi,\theta)=-\mathbb{E}_{x \sim p_{\mathrm{data}}(x),z\sim q_\phi(\cdot|x)} \left[\log p_\theta(x|z)\right]
\qquad (72)
$$

注意这两处变化：我们不再做确定性编码，而是采样 $z\sim q_\phi(z|x)$。此外，我们现在取 $x$ 在解码下的负对数似然（negative log-likelihood），也就是说，该损失实际上在问：如果我们把原始数据点 $x$ 编码再解码，它会有多大的可能——并且由于事情现在变成了随机的，我们把所有可能的解码与编码都考虑进来。对于高斯情形，这一重构损失变成：

$$
\mathcal{L}_{\mathrm{VAE\text{-}Recon}}(\phi,\theta)=\mathbb{E}_{x \sim p_{\mathrm{data}}(x),z\sim q_\phi(z|x)} \left[\frac{1}{2\sigma^2_\theta(z)}\|x-\mu_\theta(z)\|^2+\frac{d}{2}\log \sigma_{\theta}^2(z)\right]+\mathrm{const}
\qquad (73)
$$

其中我们用了正态分布的密度（见附录中的高斯分布密度）。因此，变分自编码器的重构损失与标准自编码器的重构损失并没有太大不同，我们只是必须把所有可能的编码 $z\sim q_\phi(\cdot|x)$ 都考虑进来。依赖于解码器方差的第二项，控制着重构精度与预测不确定性之间的权衡。许多实现，包括实验中的实现，把 $\sigma_\phi(x)$ 与 $\sigma_\theta(z)$ 固定为学到的标量常数（也就是分别不依赖于 $x$ 与 $z$），从而避免病态行为，以及在学习方差时的数值不稳定。因此，此时的变分自编码器重构损失，在计入编码的随机性与常数之后，基本上就变成了标准自编码器的重构损失：

$$
\mathcal{L}_{\mathrm{VAE\text{-}Recon}}(\phi,\theta)=\mathbb{E}_{x \sim p_{\mathrm{data}}(x),z\sim q_\phi(z|x)} \left[\frac{1}{2\sigma_\theta^2}\|x-\mu_\theta(z)\|^2\right]+\mathrm{const}
\qquad (74)
$$

让我们现在回到目标：我们想为数据分布 $p_{\mathrm{data}}(x)$ 构造一种编码，使得把它映射到潜空间之后，分布变得“良好”或易于学习。为此，让我们现在引入潜变量 $z$ 上的先验分布（prior distribution）$p_{\mathrm{prior}}(z)$。就我们的目的而言，我们取 $p_{\mathrm{prior}}=\mathcal{N}(0,I_k)$ 为各向同性高斯（isotropic Gaussian）。先验分布 $p_{\mathrm{prior}}$ 的这一选择，有效地代表了潜分布应当长成什么样的“理想”情形。正态分布会非常容易学习，因此会满足我们得到一个“可训练”潜分布的目标。于是，大的想法是正则化我们的编码器，以确保编码后的数据分布尽可能接近 $p_{\mathrm{prior}}$，我们通过辅助损失来做到这一点

$$
\mathcal{L}_{\mathrm{VAE\text{-}Prior}}(\phi)=\mathbb{E}_{x \sim p_{\mathrm{data}}(x)} \left[D_{\mathrm{KL}}(q_\phi(\cdot|x)\|p_{\mathrm{prior}})\right],
\qquad (75)
$$

其中 $D_{\mathrm{KL}}$ 是 KL 散度（Kullback–Leibler divergence）。KL 散度是度量两个概率分布有多不同的一种基本方式。详细解释它会超出本讲义的范围，但我们在关于 KL 散度背景的注记中给读者一个简短的回顾。这里定义的损失 $\mathcal{L}_{\mathrm{VAE\text{-}Prior}}$ 现在非常直观：我们希望，对任意数据点 $x$，编码分布看起来都像一个高斯分布。若对所有 $x$ 都这样做，那么自然可以期望我们的潜分布看起来也会是一个高斯分布。

**注记 30（KL 散度的背景）.** 对两个概率密度 $q,p$，KL 散度定义为

$$
D_{\mathrm{KL}}(q(x)\|p(x))=\int q(x) \log \frac{q(x)}{p(x)}=\mathbb{E}_{X\sim q}\left[\log \frac{q(X)}{p(X)}\right].
$$

KL 散度是分布之间不相似程度的一种标准度量。特别地，KL 散度满足下面这些有用的性质：

$$
D_{\mathrm{KL}}(q(x)\|p(x))\geq 0,
\qquad (76)
$$

$$
D_{\mathrm{KL}}(q(x)\|p(x))=0\quad \Leftrightarrow \quad q=p.
\qquad (77)
$$

亦即，它总是非负的，并且当且仅当两个概率分布重合时为零。

要定义变分自编码器的损失函数，我们现在可以把重构损失与先验损失用一个参数权重 $\beta\geq 0$ 组合起来，得到变分自编码器训练目标

$$
\mathcal{L}_{\mathrm{VAE}}(\phi,\theta)=\mathcal{L}_{\mathrm{VAE\text{-}Recon}}(\phi,\theta)+\beta\mathcal{L}_{\mathrm{VAE\text{-}Prior}}(\phi)
\qquad (78)
$$

$$
=-\mathbb{E}_{x \sim p_{\mathrm{data}}(x),z\sim q_\phi(z|x)} \left[\log p_\theta(x\mid z)\right]+\beta\mathbb{E}_{x \sim p_{\mathrm{data}}(x)} \left[D_{\mathrm{KL}}(q_\phi(\cdot|x)\|p_{\mathrm{prior}})\right]
\qquad (79)
$$

其中第一项强制潜变量能够被有效地解码回数据，第二项强制我们的潜分布接近于高斯分布。参数 $\beta$ 控制每一项的强度。为使这一损失更具体，让我们推导高斯情形的 KL 散度：

**例 31（各向同性高斯之间的 KL 散度）.** 令 $q(x)=\mathcal{N}(x;\mu_q,\mathrm{diag}(\sigma_q^2))$ 与 $p(x)=\mathcal{N}(x;\mu_p,\mathrm{diag}(\sigma_p^2))$ 为协方差矩阵对角的高斯分布，其中 $\sigma_q,\sigma_{p}\in\mathbb{R}_{\geq 0}^d$，且 $x \in \mathbb{R}^d$。则

$$
D_{\mathrm{KL}}(q\|p) =\frac12\left( \mathcal{K}\left(\frac{\sigma_{q}^2}{\sigma_{p}^2}\right)+ \frac{\|\mu_q-\mu_p\|^2}{\sigma_p^2} \right),\quad \mathcal{K}(\alpha)=\sum_{i=1}^{d}\alpha_i-\log \alpha_i-1.
\qquad (80)
$$

上面的表达式是直观的：若均值与方差彼此一致，则 $D_{\mathrm{KL}}(q\|p)=0$。此外，它随均值向量之间的平方误差 $\|\mu_{q}-\mu_{p}\|^2$ 增大。最后，函数 $\mathcal{K}(\alpha)$ 在 $\alpha=1$ 处有唯一最小值，因此当 $\sigma_{q}=\sigma_{p}$ 时 $D_{\mathrm{KL}}(q\|p)$ 达到最小。

**证明.** 我们对 $d=1$ 做证明（$d>1$ 时的证明类似，只需把每个维度加起来）。给定正态分布的密度，我们知道（见附录中的高斯分布密度）：

$$
\log q(x)= -\frac {1}{2}\log(2\pi\sigma_q^2)-\frac{1}{2\sigma_q^2}\|x-\mu_q\|^2, \qquad \log p(x)= -\frac 12\log(2\pi\sigma_p^2)-\frac{1}{2\sigma_p^2}\|x-\mu_p\|^2
$$

于是

$$
\begin{aligned}
D_{\mathrm{KL}}(q\|p) &=\mathbb{E}_{x\sim q}\big[\log q(x)-\log p(x)\big]=\frac 12\log\frac{\sigma_p^2}{\sigma_q^2} +\frac{1}{2\sigma_p^2}\mathbb{E}_q\!\left[\|x-\mu_p\|^2\right] -\frac{1}{2\sigma_q^2}\mathbb{E}_q\!\left[\|x-\mu_q\|^2\right].
\end{aligned}
\qquad (81)
$$

对 $x\sim \mathcal{N}(\mu_q,\sigma_q^2 I)$，我们有

$$
\mathbb{E}_q\!\left[\|x-\mu_q\|^2\right]=\mathrm{tr}(\sigma_q^2 I)=\sigma_q^2.
$$

把这一点与 $x-\mu_p=(x-\mu_q)+(\mu_q-\mu_p)$ 以及 $\mathbb{E}_q[x-\mu_q]=0$ 结合起来，我们得到

$$
\mathbb{E}_q\!\left[\|x-\mu_p\|^2\right] =\mathbb{E}_q\!\left[\|x-\mu_q\|^2\right]+\|\mu_q-\mu_p\|^2 =\sigma_q^2+\|\mu_q-\mu_p\|^2.
$$

将这些结果代入上文正态 KL 散度的展开，即得各向同性高斯之间的 KL 散度公式。

现在让我们假设编码器取高斯形式。于是我们得到：

$$
\mathcal{L}_{\mathrm{VAE\text{-}Prior}}(\phi)=\mathbb{E}_{x \sim p_{\mathrm{data}}(x)} \left[D_{\mathrm{KL}}(q_\phi(\cdot|x)\|\mathcal{N}(0,I_k))\right]=\mathbb{E}\left[\frac{1}{2} \mathcal{K}\left(\sigma_{\phi}^2(x)\right)+ \frac{1}{2} \|\mu_\phi(x)\|^2\right]
\qquad (82)
$$

这一损失是直观的：均值 $\mu_\phi(x)$ 因偏离零而受罚，方差因偏离 $1$ 而受罚。作为变分自编码器的总损失，我们得到

$$
\begin{aligned}
&\mathcal{L}_{\mathrm{VAE}}(\phi,\theta)\\
&=\mathcal{L}_{\mathrm{VAE\text{-}Recon}}(\phi,\theta)+\beta\mathcal{L}_{\mathrm{VAE\text{-}Prior}}(\phi)\\
&=\mathbb{E}_{x \sim p_{\mathrm{data}}(x),z\sim q_\phi(z|x)} \left[\frac{1}{2\sigma^2_\theta(z)}\|x-\mu_\theta(z)\|^2+\frac{d}{2}\log \sigma_{\theta}^2(z)+\frac{\beta}{2} \mathcal{K}\left(\sigma_{\phi}^2(x)\right)+ \frac{\beta}{2} \|\mu_\phi(x)\|^2 \right]
\end{aligned}
\qquad (83)
$$

上述损失函数的四项非常直观：第一项就是重构误差。第二项误差描述解码器的不确定性：更小的方差使解码器更为“有信心”，但也会更强地惩罚重构误差。此外，我们希望使潜变量方差为 $1$、均值为 $0$，以强制潜空间中的分布接近高斯分布。

#### 训练变分自编码器

还需要讨论我们如何最小化变分自编码器损失 $\mathcal{L}_{\mathrm{VAE}}(\phi,\theta)$。这一损失的问题在于，到目前为止，我们取其期望的那个分布（$q_\phi(z|x)$）仍然依赖于参数 $\phi$。不过，我们可以用所谓的重参数化技巧（reparameterization trick）把它改写。具体地，对于

$$
q_\phi(z|x)=\mathcal{N}(z;\mu_\phi(x),\sigma_\phi^2(x)I_k)
$$

我们可以通过下面的方式得到样本

$$
\epsilon\sim\mathcal{N}(0,I_k),\quad z=\mu_\phi(x)+\sigma_\phi(x)\epsilon \quad \Rightarrow\quad z\sim q_\phi(\cdot|x)
$$

注意，在这个等式里，噪声或随机性的唯一来源是 $\epsilon$，它的分布与 $\phi$ 无关。因此，我们可以把损失改写为：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{VAE}}(\phi,\theta)=&\mathbb{E}_{x \sim p_{\mathrm{data}}(x),\epsilon\sim\mathcal{N}(0,I_k)} \left[\frac{1}{2\sigma^2_\theta(z)}\|x-\mu_\theta(\mu_\phi(x)+\sigma_\phi(x)\epsilon)\|^2+\frac{d}{2}\log \sigma_{\theta}^2(z) +\frac{\beta}{2} \mathcal{K}\left(\sigma_{\phi}^2(x)\right)+ \frac{\beta}{2} \|\mu_\phi(x)\|^2 \right]
\end{aligned}
$$

重参数化之后，随机性只来自 $\epsilon\sim\mathcal{N}(0,I_k)$，其分布不依赖于 $\phi$。因此，我们可以用深度学习的标准工具来最小化这一损失。为了把事情进一步简化，我们可以再次令 $\sigma_{\theta}^2(z)=\sigma^2$ 处处为常数，并得到：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{VAE}}(\phi,\theta)=&\mathbb{E}_{x \sim p_{\mathrm{data}}(x),\epsilon\sim\mathcal{N}(0,I_k)} \left[\frac{1}{2\sigma^2}\|x-\mu_\theta(\mu_\phi(x)+\sigma_\phi(x)\epsilon)\|^2 +\frac{\beta}{2} \mathcal{K}\left(\sigma_{\phi}^2(x)\right)+ \frac{\beta}{2} \|\mu_\phi(x)\|^2 \right]
\end{aligned}
$$

在下文的 $\beta$-VAE 训练过程中，我们总结变分自编码器的训练步骤。

**算法 6.** $\beta$-VAE 训练过程（方差固定的高斯解码器 $p_\theta(x|z)=\mathcal{N}(x;\mu_\theta(z),\tilde{\sigma}^2 I_d)$）

**要求：** 样本数据集 $x\sim p_{\mathrm{data}}$，编码器网络 $(\mu_\phi(x),\log\sigma_\phi^2(x))$，解码器网络 $\mu_\theta(z)$，潜变量维数 $k$，常数 $\beta\ge 0$，$\sigma^2>0$。

对每个小批量 $\{x_i\}_{i=1}^B$：

1. 编码每个 $x_i$：$\mu_i \gets \mu_\phi(x_i)$，$\log\sigma_i^2 \gets \log\sigma_\phi^2(x_i)$。
2. 采样噪声 $\epsilon_i \sim \mathcal{N}(0,I_k)$。
3. 重参数化：$z_i \gets \mu_i + \sigma_i \odot \epsilon_i$（其中 $\sigma_i=\exp(\tfrac{1}{2} \log\sigma_i^2)$）。
4. 解码均值：$\hat{x}_i \gets \mu_\theta(z_i)$。
5. 重构损失：

$$
\mathcal{L}_{\mathrm{recon}} \gets \frac{1}{B}\sum_{i=1}^B \frac{1}{2\tilde{\sigma}^2}\,\|x_i-\hat{x}_i\|^2
$$

6. 到先验 $p_{\mathrm{prior}}(z)=\mathcal{N}(0,I_k)$ 的 KL 损失：

$$
\mathcal{L}_{\mathrm{KL}} \gets \frac{1}{B}\sum_{i=1}^B \frac12 \sum_{j=1}^k \left(\mu_{i,j}^2 + \sigma_{i,j}^2 - \log \sigma_{i,j}^2 - 1\right)
$$

7. 总损失：$\mathcal{L}\gets \mathcal{L}_{\mathrm{recon}} + \beta\,\mathcal{L}_{\mathrm{KL}}$。
8. 更新 $(\phi,\theta)\gets \mathrm{grad\_update}(\mathcal{L})$。

#### 实践注记

我们在这里发展的构造展示了自编码器设计的原理。当然，在实践中，人们可能会加入更多损失项或其他约束。因此，我们最后补充几条关于自编码器的实践注记：

1. **选择 $\beta$（以及 KL 预热（KL warm-up））。** 大的 $\beta$ 迫使潜变量更接近先验，但可能损害重构，并可能触发后验坍缩（posterior collapse）（编码器忽略 $x$，并输出 $q_\phi(z|x)\approx \mathcal{N}(0,I_k)$）。一种常见的稳定化手段是 KL 预热：从 $\beta=0$ 开始，并在最初若干个轮次中把它逐渐增加到一个目标值。然而，在所有现代自编码器中，$\beta$ 的值都非常小，亦即 $\beta\ll 1$。
2. **解码器方差。** 学习高斯解码器方差 $\sigma_\theta^2$ 在数值上可能很微妙，并且除非加以正则化，否则可能导致退化的解。为了稳定，许多实现把 $p_\theta(x|z)=\mathcal{N}(x;\mu_\theta(z),\sigma^2 I_d)$ 固定为常数 $\sigma^2$，这使得重构项在相差常数的意义下正比于均方误差。
3. **超出像素均方误差的重构损失。** 对于图像，逐像素的高斯似然（均方误差）常常给出过于平滑的重构。在实践中，人们加入感知损失（perceptual loss）（使用预训练网络的特征空间损失）来改善锐度与语义保真度。
4. **对抗目标与混合目标。** 为了进一步改善视觉真实感，可以把变分自编码器目标与对抗损失（adversarial loss）结合起来（VAE-GAN 风格），对解码样本使用一个判别器。这通常会让输出更锐利，但会带来额外的优化不稳定以及额外的超参数。

**注记 32（在潜空间中工作）.** 要训练一个潜空间生成模型，我们只需遵循已有的训练做法，但直接在潜空间中工作。在训练时，我们从 $q_\phi(z|x)$ 中抽取样本，其中 $x \sim p_{\mathrm{data}}$；在推断时，我们从潜扩散或流模型中采样 $z$，然后用 $x = \mu_{\mathrm{mean}}(z)$ 解码（注意我们取的是均值，而不是随机样本，以避免噪声引起的伪影）。直观地说，一个训练良好的自编码器可以被看作滤掉了高频的、或者在语义上没有意义的细节，使生成模型能够“聚焦”于重要的、与感知相关的特征 [36]。在撰写本文时，几乎所有最先进的图像与视频生成方法都遵循所谓的潜扩散（latent diffusion）范式，即在自编码器的潜空间中训练一个流模型或扩散模型 [36, 48]。不过，需要指出的是：在训练扩散模型之前，还需要先训练自编码器。至关重要的是，性能现在也取决于自编码器把图像压缩到潜空间、并恢复出美观图像的好坏。

我们在关于变分自编码器补充视角的附录中提供了进一步的讨论。

## 6.3 案例研究：Stable Diffusion 3 与 Meta Movie Gen

我们通过简要考察两个大规模生成模型来结束本节：用于图像生成的 *Stable Diffusion 3*，以及用于视频生成的 Meta 的 *Movie Gen Video* [14, 33]。你将看到，这些模型使用我们在本讲义中描述过的技术，并加上额外的架构增强，以便既扩大规模，又容纳结构丰富的条件模态，例如基于文本的输入。

### 6.3.1 Stable Diffusion 3

Stable Diffusion 是一系列最先进的图像生成模型。这些模型是最早将大规模潜扩散模型用于图像生成的模型之一。如果你还没有试过，我们强烈建议你在线亲自测试（https://stability.ai/news/stable-diffusion-3）。

Stable Diffusion 3 使用的是我们在本讲义中研究的同一个条件流匹配（conditional flow matching）目标（见高斯路径上条件流匹配的训练算法）。[^sd3noise] 正如他们的论文所述，他们广泛测试了多种流模型与扩散模型的替代方案，并发现流匹配表现最好。在训练时，它使用上文所述的无分类器引导（classifier-free guidance）训练（并丢弃类别标签）。此外，Stable Diffusion 3 遵循神经网络架构一节所概述的做法，在一个预训练自编码器的潜空间中进行训练。训练出一个好的自编码器，是最早几篇 Stable Diffusion 论文的一项重要贡献。

为增强文本条件，Stable Diffusion 3 同时使用 3 种不同类型的文本嵌入（包括 CLIP 嵌入，以及谷歌 T5-XXL 编码器的一个预训练实例所产生的序列输出 [35]，这与 [3, 39] 中采取的做法类似）。CLIP 嵌入提供的是输入文本的一种粗粒度、总览式嵌入，而 T5 嵌入提供更细粒度的上下文，使模型有可能去注意条件文本中的特定要素。为容纳这些序列上下文嵌入，作者进而提议扩展扩散 Transformer，使其不仅注意图像的图块，也注意文本嵌入，从而把条件容量从最初为扩散 Transformer 提出的基于类别的方案，扩展到序列上下文嵌入。这一提议的改版扩散 Transformer 被称为多模态扩散 Transformer（multi-modal DiT，MM-DiT），并画在多模态扩散 Transformer 的架构图中。他们最终的、最大的模型有 **80 亿参数**。采样时，他们使用 $50$ 步（也就是必须把网络求值 $50$ 次），采用欧拉仿真格式（Euler simulation scheme），无分类器引导权重在 $2.0$–$5.0$ 之间。

[^sd3noise]: 在他们的工作中，他们对噪声做条件化时使用了不同的约定。但这只是记号不同，算法是一样的。

**图注.** 图 16。[14] 中提出的多模态扩散 Transformer 的架构。图亦取自 [14]。

### 6.3.2 Meta Movie Gen Video

接下来，我们讨论 Meta 的视频生成器 *Movie Gen Video*（https://ai.meta.com/research/movie-gen/）。由于数据不是图像而是视频，数据 $x$ 位于空间 $\mathbb{R}^{T \times C \times H \times W}$ 中，其中 $T$ 表示新的时间维（temporal dimension）（即帧数）。我们将看到，这一视频设定中的许多设计选择，都可以看作是把图像设定中已有的技术（例如自编码器、扩散 Transformer 等）加以改造，以处理这一额外的时间维。

Movie Gen Video 使用条件流匹配目标，并采用同样的直线调度（straight line scheduler）$\alpha_t=t$，$\sigma_{t}=1-t$。与 Stable Diffusion 3 一样，Movie Gen Video 也在一个冻结的、预训练的自编码器的潜空间中运作。注意，用自编码器来降低内存消耗，对视频比对图像更为重要——这也是为什么目前大多数视频生成器所能生成的视频长度相当有限。具体而言，作者提议引入时间自编码器（temporal autoencoder，TAE）来处理新增的时间维，它把原始视频 $x_t' \in \mathbb{R}^{T' \times 3 \times H \times W}$ 映射到潜变量 $x_t\in\mathbb{R}^{T \times C \times H \times W}$，并且 $\tfrac{T'}{T} = \tfrac{H'}{H} = \tfrac{W'}{W} = 8$ [33]。为容纳长视频，他们提出一种时间分块（temporal tiling）流程：把视频切成若干段，每一段分别编码，再把潜变量拼接起来 [33]。模型本身——也就是 $u_t^\theta(x_t)$——由一个类似扩散 Transformer 的骨干给出，其中 $x_t$ 沿时间维与空间维被图块化。这些图像图块随后送入一个 Transformer，它既在图像图块之间做自注意力，又与语言模型嵌入做交叉注意力，类似于 Stable Diffusion 3 所采用的多模态扩散 Transformer。对于文本条件，Movie Gen Video 采用三种文本嵌入：UL2 嵌入，用于细粒度的、基于文本的推理 [47]；ByT5 嵌入，用于注意字符级细节（例如提示明确要求出现特定文字）[50]；以及 MetaCLIP 嵌入，在一个共享的文本–图像嵌入空间中训练 [24, 33]。他们最终的、最大的模型有 **300 亿参数**。若需要显著更详细、更全面的论述，我们鼓励读者查阅 Movie Gen 技术报告本身 [33]。

---

<!-- 来源文件：07.zh.md -->

## 7 离散扩散模型（Discrete Diffusion Models）：用扩散构建语言模型（Language Models）

在前面各节中，我们把流模型与扩散模型（flow and diffusion models）作为欧氏空间（Euclidean space）$\mathbb{R}^d$ 上的生成模型来讨论，它们使我们能够生成由向量（vector）$z\in\mathbb{R}^d$ 表示的数据点。然而，并非所有数据都自然地建模为欧氏空间 $\mathbb{R}^d$ 中的一个点。许多数据类型，例如文本或 DNA，更自然地被看作离散状态空间（discrete state space）$S$ 中的元素。最重要的是，语言由我们希望建模的一串离散词元（token）组成。我们如何把流模型与扩散模型应用到这类数据上？事实上，前面各节学到的原理同样可以推广到这类数据。所得模型在机器学习文献中称为**离散扩散模型**（Campbell et al., 2022；Gat et al., 2024）。不过必须记住：并不存在数学意义上的扩散过程（随机微分方程（stochastic differential equation, SDE）在离散状态空间中并不存在）。我们使用的不是常微分方程（ordinary differential equation, ODE）或随机微分方程，而是**连续时间马尔可夫链**（continuous-time Markov chain, CTMC）。下面我们将解释连续时间马尔可夫链模型（见第 7.1 节），以及如何学习它们（见第 7.2 节），从而能够运用流模型与扩散模型的原理来构建大语言模型（large language model, LLM）。

### 7.1 连续时间马尔可夫链模型

本节解释连续时间马尔可夫链。可以把连续时间马尔可夫链看作随机微分方程的离散类比，用来构建生成离散状态的神经网络（neural network）模型。此外，我们将引入连续时间马尔可夫链模型，即利用连续时间马尔可夫链生成文本等离散序列的神经网络模型。

**图注.**（图 17）状态空间 $S=\{S_1,S_2,S_3\}$（序列长度 $d=1$）上一条连续时间马尔可夫链轨迹的示意图。图改编自 Campbell et al., 2022。

让我们从刻画**状态空间** $S$ 开始。设 $\mathcal{V}=\{v_1,\cdots,v_V\}$ 为我们的词表（vocabulary）。状态空间由 $S=\mathcal{V}^d$ 给出，其中 $d\in\mathbb{N}$ 是序列长度（sequence length），$V\in\mathbb{N}$ 是词表大小（vocabulary size）。对语言而言，$\{v_1,\cdots,v_V\}$ 可以枚举字母表或一组离散词元，而 $S$ 表示长度为 $d$ 的序列（或句子）的集合。对 DNA 而言，$\{v_1,\cdots,v_V\}$ 可以是全部 4 种 DNA 碱基，而 $S$ 是所有长度为 $d$ 的 DNA 序列。

接下来，设 $X_t$ 是 $S$ 上的随机过程（stochastic process），即 $S$ 中的一条随机轨迹（random trajectory）$X:[0,1]\to S$，$t\mapsto X_t$。我们要求 $X_t$ 是**马尔可夫过程**（Markov process），即没有记忆的过程。具体地说，下列条件成立

$$
p(X_{t+h}|X_t,X_{t_1},\cdots,X_{t_k})=p(X_{t+h}|X_t)\quad (0<h,\ 0\leq t_1<t_2<\cdots<t_k<t).
$$

换言之，未来事件的概率只依赖于现在——过去对未来不再有任何意义。注意，常微分方程与随机微分方程虽然不定义在离散状态空间上，它们也是马尔可夫过程。这里 $X_t$ 位于离散空间上，因此称为马尔可夫链（Markov chain），具体地说是**连续时间马尔可夫链**。量 $p_{t+h|t}(X_{t+h}|X_t)$ 是**转移概率**（transition probabilities），它们与马尔可夫链的初始分布（initial distribution）$X_0\sim p_0$ 一起完全决定该连续时间马尔可夫链。因此，说到连续时间马尔可夫链时，也可以直接把它想成转移概率 $p_{t+h|t}(X_{t+h}|X_t)$。

接下来，我们推导离散设定下与向量场（vector field）对应的对象。由于我们处在离散设定中，只能在状态之间跳跃（或切换），而不能再像指定常微分方程时那样沿某个方向行进。因此，我们定义**速率矩阵**（rate matrix）$Q_t(y|x)$，它有效地概括了从状态 $x\in S$ 跳跃（或切换）到状态 $y\in S$ 的速率。形式地，速率矩阵 $Q_t$ 由一个有界函数（关于时间连续）给出

$$
Q:S\times S\times[0,1]\to\mathbb{R},\quad (x,y,t)\mapsto Q_t(y|x),
\qquad (84)
$$

其中 $Q_t(y|x)$ 描述从 $x$ 切换到 $y$ 的速率，并满足

$$
\quad Q_t(y|x)\geq 0\quad x\neq y,
\qquad (85)
$$

$$
\quad Q_t(x|x)=-\sum_{y\neq x}Q_t(y|x)\quad x.
\qquad (86)
$$

这两个条件是直观的。第一个条件说，从 $x$ 切换到另一个状态 $y\neq x$ 的速率只能是非负的（不切换正好对应 $0$，因此速率小于 $0$ 没有意义）。第二个条件说，停留在 $x$ 的速率 $Q_t(x|x)$ 应当与离开 $x$ 的速率相互抵消——它本质上是一个一致性条件，表示你要么留在 $x$，要么离开（没有第三种选择）。注意，这些条件特别蕴含 $Q_t(x|x)\leq 0$。因此，$Q_t(y|x)$ 是一个矩阵，其对角元全部非正，而所有非对角元非负。

现在可以定义与微分方程对应的对象，即要求连续时间马尔可夫链“跟随”速率矩阵的条件。基本想法是，$X$ 的分布或演化应当跟随速率矩阵 $Q_t$。换言之，我们要求转移概率满足

$$
\frac{\mathrm{d}}{\mathrm{d} h}p_{t+h|t}(X_{t+h}=y|X_t=x)\big|_{h=0}=Q_t(y|x)\quad x,y\in S,\ 0\leq t.
\qquad (87)
$$

左端是从 $x$ 切换到 $y$ 的概率的无穷小变化率。我们施加的条件是，这些概率应当按照速率矩阵所规定的方式变化。让我们简要核对这些条件是否合理：若只是按上文的速率矩阵微分方程（式 (87)）来设定 $Q_t(y|x)$，它是否构成合法的速率矩阵？当 $h=0$ 时，从 $x$ 切换到 $y\neq x$ 的概率为零（因为没有时间流逝），即对所有 $y\neq x$ 有 $p_{t|t}(y|x)=0$。因此，导数必须非负，并且只要 $y\neq x$ 就有 $Q_t(y|x)\geq 0$。这核验了流出速率条件（式 (85)）成立。进一步，我们知道

$$
\begin{aligned}
\sum_{y\neq x}Q_t(y|x)
&=\sum_{y\neq x}\frac{\mathrm{d}}{\mathrm{d} h}p(X_{t+h}=y|X_t=x)\big|_{h=0}\\
&=\frac{\mathrm{d}}{\mathrm{d} h}\sum_{y\neq x}p(X_{t+h}=y|X_t=x)\big|_{h=0}\\
&=\frac{\mathrm{d}}{\mathrm{d} h}\bigl(1-p(X_{t+h}=x|X_t=x)\bigr)\\
&=-Q_t(x|x),
\end{aligned}
$$

其中用到概率之和为 $1$。这表明停留速率条件（式 (86)）成立。由此可知，每一条连续时间马尔可夫链至少有一个满足速率矩阵微分方程（式 (87)）的速率矩阵。但若反过来：如果我们指定 $Q_t$，是否存在对应的连续时间马尔可夫链，并且若存在，它是否唯一？答案确实如此。

**定理 33（连续时间马尔可夫链的存在性与唯一性）.** 对任意速率矩阵 $Q_t$（关于时间 $t$ 有界且连续），存在唯一的马尔可夫链 $X_t$（即唯一的一组转移概率 $p_{t+h|t}(y|x)$），使得速率矩阵微分方程（式 (87)）成立。

对有兴趣的读者，我们在附录 C（连续时间马尔可夫链的存在性与唯一性）中给出一个自洽的证明。该定理的关键结论是：就机器学习的目的而言，我们可以构造一个速率矩阵 $Q_t$（例如借助神经网络），并假定存在与 $Q_t$ 对应的唯一马尔可夫链。

**例 34（具有相等跳跃速率（jump rate）的二状态连续时间马尔可夫链）.** 设 $S=\{a,b\}$，并考虑一条时间齐次（time-homogeneous）的连续时间马尔可夫链 $(X_t)_{t\geq 0}$，它以常数速率 $\lambda>0$ 在两个状态之间切换：

$$
Q=
\begin{array}{c|cc}
 & a & b \\
a & -\lambda & \lambda \\
b & \lambda & -\lambda
\end{array}.
$$

则在时间增量 $h\geq 0$ 上的转移概率也与时间 $t$ 无关，并由下式给出

$$
\begin{pmatrix}
p(X_{t+h}=a|X_t=a) & p(X_{t+h}=a|X_t=b)\\
p(X_{t+h}=b|X_t=a) & p(X_{t+h}=b|X_t=b)
\end{pmatrix}
=
\frac{1}{2}
\begin{pmatrix}
1+e^{-2\lambda h} & 1-e^{-2\lambda h}\\
1-e^{-2\lambda h} & 1+e^{-2\lambda h}
\end{pmatrix}.
$$

可以手算验证速率矩阵微分方程（式 (87)）成立，也就是说，这些转移概率对该速率矩阵而言确实是正确的。事实上，这些速率非常直观：该链以瞬时速率 $\lambda$ 不断翻转。指数项 $e^{-2\lambda h}$ 刻画初始状态的记忆如何衰减。当时间趋于无穷，即 $h\to\infty$ 时，有

$$
P(h)\to
\begin{pmatrix}
\frac{1}{2} & \frac{1}{2}\\
\frac{1}{2} & \frac{1}{2}
\end{pmatrix},
$$

因此该链忘记自己从何处出发，并以概率 $1/2$ 处于 $a$ 或 $b$。切换速率 $\lambda>0$ 越高，这一收敛越快。

**连续时间马尔可夫链的仿真.** 接下来考虑如何仿真一条连续时间马尔可夫链的轨迹。设 $h>0$ 为步长（step size），$p_{\mathrm{init}}$ 为 $S$ 上的初始分布，例如 $p_{\mathrm{init}}=\mathrm{Unif}_S$ 是 $S$ 上的均匀分布（uniform distribution）。然后可以迭代地仿真：令 $X_0\sim p_{\mathrm{init}}$，并令

$$
X_{t+h}\sim p_{t+h|t}(\cdot|X_t).
$$

若我们知道 $p_{t+h|t}(\cdot|X_t)$，这样做是可行的。然而，除了最简单的连续时间马尔可夫链之外，我们通常不知道转移核（transition kernel）的闭式（closed form），只能接触到速率矩阵 $Q_t$。尽管如此，由速率矩阵微分方程（式 (87)）有

$$
\begin{aligned}
p_{t+h|t}(X_{t+h}=y|X_t=x)
&=p_{t|t}(X_t=y|X_t=x)+hQ_t(y|x)+R_t(h)\\
&=1_{y=x}+hQ_t(y|x)+R_t(h),
\end{aligned}
$$

其中 $R_t(h)$ 是在 $h$ 较小时可以忽略的误差项（error term）。因此，对小的 $h$，可以令

$$
p_{t+h|t}(X_{t+h}=y|X_t=x)\approx 1_{y=x}+hQ_t(y|x)=:\tilde{p}_{t+h|t}(y|x).
$$

由我们对速率矩阵施加的条件可以核验，当 $h$ 较小时 $\tilde{p}_{t+h|t}(y|x)$ 确实是合法的概率分布。因此，可以按下式近似抽取下一个点

$$
X_{t+h}\sim \tilde{p}_{t+h|t}(\cdot|x)=(1_{y=x}+hQ_t(y|x))_{y\in S}.
\qquad (88)
$$

上式只是一个离散分布，可以用标准方法轻松抽样。这是仿真连续时间马尔可夫链的一种简单方式。

**连续时间马尔可夫链模型.** 接下来定义如何用神经网络参数化一条连续时间马尔可夫链。一个**连续时间马尔可夫链模型**（或**离散扩散模型**）由 $S$ 上的初始分布 $p_{\mathrm{init}}$ 以及带参数 $\theta$ 的神经网络 $Q_t^\theta$ 给出，使得对每个输入 $x\in S$，模型返回速率矩阵的一列

$$
x\mapsto \{Q_t^\theta(y|x)\}_{y\in S}.
$$

我们希望模型返回整列，因为仿真连续时间马尔可夫链（式 (88)），也就是抽取下一个状态，需要这一列。

上述模型的一个困难是，空间 $S$ 可能*非常*大。特别地，$|S|=V^d$，其中 $V$ 是词表大小，$d$ 是序列长度。这种指数增长使得把速率矩阵的一整列存入内存基本上不可能——$\{Q_t^\theta(y|x)\}_{y\in S}$ 永远无法在计算机中表示。因此必须约束模型。具体地说，几乎所有连续时间马尔可夫链模型都是因子化（factorized）的（见因子化连续时间马尔可夫链示意图，图 18），这实际上是一种稀疏性约束（sparsity constraint）。具体地说，**因子化连续时间马尔可夫链模型**是这样一个连续时间马尔可夫链模型 $Q_t^\theta$：对所有 $y=(y_1,\cdots,y_d)$、$x=(x_1,\cdots,x_d)\in S=\mathcal{V}^d$，有

$$
Q_t^\theta(y|x)=0\quad iy_i\neq x_i.
$$

我们把所有与 $x$ 至多相差一个词元的 $y$ 称为 $x$ 的**邻居**（neighbors）$N(x)$。这样的因子化连续时间马尔可夫链模型可以写成

$$
x\mapsto \{Q_t^\theta(y|x)\}_{y\in N(x)}
=
\begin{pmatrix}
Q_t^\theta(v_1,1|x) & \cdots & Q_t^\theta(v_V,1|x)\\
\cdots & & \\
Q_t^\theta(v_1,d|x) & \cdots & Q_t^\theta(v_V,d|x)
\end{pmatrix},
$$

其中 $Q_t(y|x)=Q_t^\theta(v_i,j|x)$ 现在给出从 $x=(x_1,\cdots,x_d)$ 走到 $x$ 的那个邻居的速率，该邻居由把第 $j$ 个元素换成 $v_i$ 得到，即 $y=(x_1,\cdots,x_{j-1},v_i,x_{j+1},\cdots,x_d)$。每一行对应一个位置 $i=1,\cdots,d$ 上的速率矩阵，即我们要求

$$
Q_t^\theta(v,i|x)\geq 0v\neq x_i,\quad Q_t(x_i,i|x)=-\sum_{v\neq x_i}Q_t^\theta(v,i|x).
$$

这些条件很容易施加在神经网络的输出上，例如可以使用序列长度为 $d$、输出维数为 $V$ 的 Transformer（transformer）模型。还要注意，因子化速率矩阵使输出形状为 $d\times V$——这一规模随维数线性增长（而不是指数增长）。

**图注.**（图 18）因子化连续时间马尔可夫链模型的示意图。仅当起点与终点只在一个维度上不同时，因子化连续时间马尔可夫链才有非零速率（$Q_t(y|x)\neq 0$）（此处 $d=2$）。图取自 Lipman et al., 2024。

**仿真连续时间马尔可夫链模型.** 要从连续时间马尔可夫链模型中抽样，我们抽取 $X_0\sim p_{\mathrm{init}}$，并按连续时间马尔可夫链仿真公式（式 (88)）迭代抽取下一个状态。算法见下方的算法 7。如该算法所示，对因子化连续时间马尔可夫链模型，可以使用逐词元并行的欧拉近似（Euler approximation），在一个小步长 $h>0$ 内每个词元独立更新。该近似与完整的连续时间马尔可夫链欧拉步在 $h$ 的一阶上一致，但允许多个词元同时更新的概率为 $O(h^2)$。

**算法 7　从因子化连续时间马尔可夫链模型采样**

**输入.** 速率网络 $Q_t^\theta$（已因子化），初始分布 $p_{\mathrm{init}}$，步数 $n$。

1. 令 $t\leftarrow 0$，步长 $h\leftarrow \frac{1}{n}$。
2. 抽取样本 $X_0\sim p_{\mathrm{init}}$，其中 $X_0=(X_0^{(1)},\cdots,X_0^{(d)})\in\mathcal{V}^d$。
3. 对 $i=1,\cdots,n$：
   - 计算因子化跳跃速率 $\{q_j(v)\}_{j=1,\ldots,d,\ v\in\mathcal{V}}\leftarrow Q_t^\theta(\cdot\mid X_t)$。
   - 对 $j=1,\cdots,d$（**并行**）：
     - $x\leftarrow X_t^{(j)}$（位置 $j$ 上的当前词元）。
     - 按如下方式定义逐位置的欧拉转移概率 $\tilde{p}_{j,t}(\cdot\mid X_t^{(j)}=x)$：

$$
\tilde{p}_{j,t}(v\mid x)=
\begin{cases}
h\, q_j(v), & v\neq x,\\
1-h\sum_{v'\in\mathcal{V}\setminus\{x\}}q_j(v'), & v=x.
\end{cases}
$$

     - 采样 $X_{t+h}^{(j)}\sim \mathrm{Categorical}\bigl(\{\tilde{p}_{j,t}(v\mid x)\}_{v\in\mathcal{V}}\bigr)$。
   - 令 $t\leftarrow t+h$。
4. 返回 $X_1$。

### 7.2 训练连续时间马尔可夫链模型

接下来讨论如何学习连续时间马尔可夫链模型。原理与流匹配（flow matching）相同：（1）构造一条在噪声与数据之间插值的概率路径（probability path）；（2）推导条件速率矩阵（conditional rate matrix）与边缘速率矩阵（marginal rate matrix）；（3）以无仿真（simulation-free）的方式学习边缘速率矩阵。下面逐步解释这一做法。

在本节中，**数据分布**（data distribution）$p_{\mathrm{data}}$ 是 $S$ 上由概率质量函数（probability mass function）刻画的分布。即 $p_{\mathrm{data}}:S\to\mathbb{R}_{\geq 0}$，$z\mapsto p_{\mathrm{data}}(z)$，且 $\sum_{z\in S}p_{\mathrm{data}}(z)=1$。我们并不知道 $p_{\mathrm{data}}$，但在训练时可以以数据集的形式取得样本 $z\sim p_{\mathrm{data}}$。例如，万维网上的全部文本。我们的目标是学会生成样本 $z\sim p_{\mathrm{data}}$。我们的目标是训练连续时间马尔可夫链模型 $Q_t^\theta$，使得

$$
X_0\sim p_{\mathrm{init}},\quad X_tQ_t^\theta\quad\Rightarrow\quad X_1\sim p_{\mathrm{data}}.
$$

因此你可能会意识到，这与欧氏情形 $\mathbb{R}^d$ 并无不同（见第 2 节“流与扩散模型”与第 3 节“流匹配”），只是我们使用连续时间马尔可夫链模型，而不是流模型或扩散模型。

#### 7.2.1 条件概率路径（Conditional Probability Path）与边缘概率路径（Marginal Probability Path）

我们定义函数 $\delta_z(x)$，使得当 $x\neq z$ 时 $\delta_z(x)=0$，当 $x=z$ 时 $\delta_z(x)=1$。一条（离散）**条件概率路径**由一族分布 $p_t(x|z)$ 给出，其中 $x,z\in S$ 且 $0\leq t\leq 1$，并满足

$$
p_0(\cdot|z)=p_{\mathrm{init}},\quad p_1(\cdot|z)=\delta_z.
$$

因此，与欧氏情形类似，离散条件概率路径在一个与 $z$ 无关的分布和一个全部质量集中于 $z$ 的分布之间插值。一条（离散）**边缘概率路径**则由下式给出

$$
p_t(x)=\sum_{z\in S}p_t(x|z)p_{\mathrm{data}}(z).
$$

容易验证，边缘概率路径在“噪声”与数据之间插值：

$$
p_0=p_{\mathrm{init}},\quad p_1=p_{\mathrm{data}}.
\qquad (89)
$$

**例 35（因子化混合路径（factorized mixture path）：逐词元独立加噪）.** 设 $S=\mathcal{V}^d$，并设 $p_{\mathrm{init}}(x)=\prod_{j=1}^d p_{\mathrm{init}}^{(j)}(x_j)$ 为因子化的初始分布。固定一个**调度器**（scheduler）$0\leq\kappa_t\leq 1$，使得 $\kappa_0=0$、$\kappa_1=1$，且 $\frac{\mathrm{d}}{\mathrm{d} t}\dot{\kappa}_t\geq 0$。把条件路径定义为

$$
p_t(x|z)=\prod_{j=1}^d\Bigl[(1-\kappa_t)\,p_{\mathrm{init}}^{(j)}(x_j)+\kappa_t\,\delta_{z_j}(x_j)\Bigr].
$$

等价地，可以这样抽取 $x\sim p_t(\cdot\mid z)$：抽取独立同分布（independent and identically distributed, i.i.d.）的掩码（mask）$m_j=0,1$ 以及噪声 $\xi_j\sim p_{\mathrm{init}}^{(j)}$，然后令

$$
\begin{aligned}
m_j&\sim\mathrm{Bernoulli}(\kappa_t),\quad \xi_j\sim p_{\mathrm{init}}^{(j)},\\
x_j&=m_j\, z_j+(1-m_j)\,\xi_j,\qquad j=1,\cdots,d,\\
x&=(x_1,\cdots,x_d).
\end{aligned}
$$

我们把上述对象称为**因子化混合路径**。上述过程以概率 $1-\kappa_t$ 对序列中每个位置独立地“破坏”第 $j$ 个词元，即当 $t=0$ 时 $1-\kappa_t=1$，全部信息被破坏；当 $t=1$ 时 $1-\kappa_t=0$，没有信息被破坏。注意，这与高斯概率路径（Gaussian probability path）一例（例 8）相似，信息以由调度器 $\kappa_t$ 决定的速度逐步被破坏。然而，它也与高斯概率路径不同：因子化混合路径并不移动或输运概率质量（我们处在离散空间中，没有方向）——它只是让一个分布淡入、另一个分布淡出。

**图注.**（图 19）$d=2$ 时离散概率路径的示意图。上排：在初始分布与狄拉克分布（Dirac distribution）之间插值的条件概率路径。下排：在初始分布与数据分布之间的插值（此处为棋盘图案）。请注意它与条件及边缘概率路径直方图（图 5）的相似与不同：这里的概率路径是被“瞬移”（teleported）的（我们降低初始分布的权重、提高终点分布的权重）。

#### 7.2.2 条件速率矩阵与边缘速率矩阵

下一步，我们构造离散流匹配（discrete flow matching）的训练目标。首先构造条件速率矩阵，它对应于流匹配中的条件向量场（conditional vector field）。设对每个数据点 $z\in S$，$Q_t^z(y|x)$ 是一个速率矩阵。若

$$
X_0\sim p_{\mathrm{init}},\quad X_tQ_t^z\quad\Rightarrow\quad X_t\sim p_t(\cdot|z),
$$

则称它为**条件速率矩阵**。换言之，条件速率矩阵使得它的连续时间马尔可夫链“跟随”条件概率路径。条件速率矩阵是构造跟随边缘概率路径的边缘速率矩阵的构件。

**定理 36（离散边缘化技巧（Discrete Marginalization Trick））.** 由下式定义的**边缘速率矩阵**

$$
Q_t(y|x)=\sum_{z\in S}Q_t^z(y|x)\frac{p_t(x|z)p_{\mathrm{data}}(z)}{p_t(x)}=\sum_{z\in S}Q_t^z(y|x)p_{1|t}(z|x),
\qquad (90)
$$

其中 $p_{1|t}(z|x):=\frac{p_t(x|z)p_{\mathrm{data}}(z)}{p_t(x)}$，是合法的速率矩阵，并满足下列条件：

$$
X_0\sim p_{\mathrm{init}},\quad X_tQ_t\quad\Rightarrow\quad X_t\sim p_t.
$$

特别地，由离散概率路径的插值关系（式 (89)）有 $X_1\sim p_{\mathrm{data}}$，即边缘速率矩阵的连续时间马尔可夫链把噪声转换成数据。

为证明这一陈述，我们需要连续时间马尔可夫链的一个基本方程，即所谓的*科尔莫戈罗夫前向方程*（Kolmogorov Forward Equation, KFE）。

**命题 2（科尔莫戈罗夫前向方程）.** 设对每个 $0\leq t\leq 1$，$p_t$ 是 $S$ 上的一族分布。又设 $X_t$ 是速率矩阵为 $Q_t$、初始分布为 $p_0$ 的连续时间马尔可夫链。则对所有 $0\leq t\leq 1$ 有 $X_t\sim p_t$，当且仅当**科尔莫戈罗夫前向方程**成立：

$$
\frac{\mathrm{d}}{\mathrm{d} t}p_t(x)=\sum_{y\in S}Q_t(x|y)p_t(y).
$$

**科尔莫戈罗夫前向方程的证明.** 为说明该方程是必要的，假设 $p_t(x)$ 是连续时间马尔可夫链的真实边缘分布，即对每个 $0\leq t\leq 1$ 有 $X_t\sim p_t$。则可以计算

$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d} t}p_t(x)
&=\left.\frac{\mathrm{d}}{\mathrm{d} h}\right|_{h=0}p_{t+h}(x)\\
&=\left.\frac{\mathrm{d}}{\mathrm{d} h}\right|_{h=0}\sum_{y}p_{t+h|t}(x|y)p_t(y)\\
&=\sum_{y}\left.\frac{\mathrm{d}}{\mathrm{d} h}\right|_{h=0}p_{t+h|t}(x|y)p_t(y)\\
&=\sum_{y}Q_t(x|y)p_t(y),
\end{aligned}
$$

其中 (i) 只是作了时间平移，(ii) 用到转移概率的定义，(iii) 交换了求和与求导，(iv) 用到速率矩阵的定义（见速率矩阵微分方程，式 (87)）。

接下来，为说明该方程是充分的，可以把科尔莫戈罗夫前向方程改写成矩阵形式

$$
\frac{\mathrm{d}}{\mathrm{d} t}p_t=Q_t p_t,
$$

在这个方程中，我们把 $p_t=(p_t(x))_{x\in S}$ 视为向量，把 $Q_t=(Q_t(y|x))_{x,y\in S}$ 视为矩阵。注意，上式是向量空间 $\mathbb{R}^S$ 上的线性常微分方程。其初始条件由本命题所述的 $p_0$ 固定。因此，若另有一组边缘分布 $q_t$ 满足该方程，则由常微分方程的唯一性（定理 3）可得 $q_t=p_t$。这表明科尔莫戈罗夫前向方程也是充分的。

**定理 36 的证明.** 利用科尔莫戈罗夫前向方程，余下要证明的是：定理中定义的边缘速率矩阵（见离散边缘速率矩阵，式 (90)）满足科尔莫戈罗夫前向方程：

$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d} t}p_t(x)
&=\frac{\mathrm{d}}{\mathrm{d} t}\sum_{z\in S}p_t(x|z)p_{\mathrm{data}}(z)\\
&=\sum_{z\in S}\frac{\mathrm{d}}{\mathrm{d} t}p_t(x|z)p_{\mathrm{data}}(z)\\
&=\sum_{z\in S}\left[\sum_{y\in S}Q_t^z(x|y)p_t(y|z)\right]p_{\mathrm{data}}(z)\\
&=\sum_{y\in S}p_t(y)\left[\sum_{z\in S}Q_t^z(x|y)\frac{p_t(y|z)p_{\mathrm{data}}(z)}{p_t(y)}\right]\\
&=\sum_{y\in S}p_t(y)Q_t(x|y),
\end{aligned}
$$

其中 (i) 由边缘概率路径的定义得到，(ii) 交换了求和与求导，(iii) 对条件速率矩阵使用科尔莫戈罗夫前向方程，(iv) 乘以并除以 $p_t(y)$，(v) 用到边缘速率矩阵 $Q_t(y|x)$ 的定义。这表明科尔莫戈罗夫前向方程成立。结论由科尔莫戈罗夫前向方程命题（命题 2）得到。

现在为因子化混合路径推导一个具体的条件速率矩阵例子。

**例 37（因子化混合路径的条件速率矩阵）.** 令 $\frac{\mathrm{d}}{\mathrm{d} t}\kappa_t=\dot{\kappa}_t$。因子化混合路径具有因子化的条件速率矩阵

$$
\begin{aligned}
Q_t^z(y|x)&=(Q_t^z(v_i,j|x_j))_{v_i,j},\\
Q_t^z(v_i,j|x_j)&=\frac{\dot{\kappa}_t}{1-\kappa_t}\bigl(\delta_{z_j}(v_i)-\delta_{x_j}(v_i)\bigr)\\
&=\frac{\dot{\kappa}_t}{1-\kappa_t}
\begin{cases}
0 & x_j=z_j,\\
1 & v_i=z_j,\ x_j\neq z_j,\\
0 & v_i\neq z_j,\ x_j\neq z_j,\\
-1 & v_i=x_j,\ x_j\neq z_j.
\end{cases}
\end{aligned}
$$

注意，这是一个非常简单的速率矩阵：它只允许跳到 $z^j$，也就是说，若任一位置 $j$ 上的词元被更新，它必须跳到终点数据点 $z=(z_1,\cdots,z_d)$ 在该位置的词元取值；并且只有当我们尚未到达 $z^j$ 时才会跳向 $z^j$。

**证明.** 注意，因子化混合路径完全分解为独立分量，所建议的条件速率矩阵也同样如此。因此可以不失一般性地假设 $d=1$。于是我们只对每个维度做计算。然后可以推导

$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d} t}p_t(x|z)
&=\frac{\mathrm{d}}{\mathrm{d} t}\Bigl[(1-\kappa_t)p_{\mathrm{init}}(x)+\kappa_t\delta_z(x)\Bigr]\\
&=\dot{\kappa}_t\delta_z(x)-\dot{\kappa}_t p_{\mathrm{init}}(x)\\
&=\frac{\dot{\kappa}_t}{1-\kappa_t}\Bigl(\delta_z(x)-\bigl[(1-\kappa_t)p_{\mathrm{init}}(x)+\kappa_t\delta_z(x)\bigr]\Bigr)\\
&=\frac{\dot{\kappa}_t}{1-\kappa_t}\bigl(\delta_z(x)-p_t(x|z)\bigr)\\
&=\frac{\dot{\kappa}_t}{1-\kappa_t}\delta_z(x)\bigl(1-p_t(x|z)\bigr)+\frac{\dot{\kappa}_t}{1-\kappa_t}\bigl(\delta_z(x)-1\bigr)p_t(x|z)\\
&=\sum_{y\neq x}\frac{\dot{\kappa}_t}{1-\kappa_t}\delta_z(x)p_t(y|z)+\frac{\dot{\kappa}_t}{1-\kappa_t}\bigl(\delta_z(x)-1\bigr)p_t(x|z)\\
&=\sum_{y\neq x}Q_t^z(x|y)p_t(y|z)+Q_t^z(x|x)p_t(x|z)\\
&=\sum_{y\in S}Q_t^z(x|y)p_t(y|z),
\end{aligned}
$$

其中 (i) 用到 $d=1$ 时因子化混合路径的定义，(ii) 由求导并把 $\frac{\mathrm{d}}{\mathrm{d} t}\kappa_t=\dot{\kappa}_t$ 代入得到，(iii) 由简单代数得到，(iv) 由因子化混合路径的定义得到，(v) 由简单代数得到，(vi) 由 $\sum_{y\in S}p_t(y|z)=1$ 这一事实得到，(vii) 由速率矩阵的定义得到，(viii) 由简单代数得到。上式表明科尔莫戈罗夫前向方程成立，因此结论成立。

#### 7.2.3 学习边缘速率矩阵

本节推导训练连续时间马尔可夫链模型的基本算法。由离散边缘化技巧（定理 36），训练连续时间马尔可夫链模型 $Q_t^\theta(y|x)$ 可以通过学习边缘速率矩阵来完成。

本节现在限定在因子化混合路径上（见因子化混合路径一例，例 35），因为这是迄今大多数离散扩散模型与离散流匹配模型所使用的路径。在这种情形下，边缘速率矩阵具有非常直观的形式。

**定理 38（因子化混合路径的边缘化技巧）.** 因子化混合路径的边缘速率矩阵是因子化的，并具有形式

$$
Q_t(v_i,j|x)=\frac{\dot{\kappa}_t}{1-\kappa_t}\bigl(p_{1|t}(z_j=v_i|x)-\delta_{x_j}(v_i)\bigr),
$$

其中 $p_{1|t}(z_j=v_i|x)$ 是在给定整条带噪序列 $x$ 时，第 $j$ 个位置（序列中的第 $j$ 个词元）等于 $v_i$ 的条件概率。

**证明.** 边缘速率矩阵由下式给出

$$
Q_t(y|x)=\sum_{z\in S}Q_t^z(y|x)p_{1|t}(z|x).
\qquad (91)
$$

现在，只要 $y$ 与 $x$ 不是邻居（相差多于一个词元），对每个 $z$ 都有 $Q_t^z(y|x)=0$。因此在这种情形下也有 $Q_t(y|x)=0$。这表明边缘速率矩阵同样因子化。于是有

$$
Q_t(v_i,j|x)=\sum_{z\in S}Q_t^z(v_i,j|x)p_{1|t}(z|x).
\qquad (92)
$$

$$
Q_t(v_i,j|x)=\sum_{z\in S}\frac{\dot{\kappa}_t}{1-\kappa_t}\bigl(\delta_{z_j}(v_i)-\delta_{x_j}(v_i)\bigr)p_{1|t}(z|x).
\qquad (93)
$$

$$
Q_t(v_i,j|x)=\frac{\dot{\kappa}_t}{1-\kappa_t}\left(\sum_{z\in S}\delta_{z_j}(v_i)p_{1|t}(z|x)-\delta_{x_j}(v_i)\right).
\qquad (94)
$$

$$
Q_t(v_i,j|x)=\frac{\dot{\kappa}_t}{1-\kappa_t}\bigl(p_{1|t}(z_j=v_i|x)-\delta_{x_j}(v_i)\bigr).
\qquad (95)
$$

其中 (i) 由条件速率矩阵的公式得到（见因子化混合路径的条件速率矩阵一例，例 37），(ii) 由 $\sum_{z\in S}p_{1|t}(z|x)=1$ 得到，(iii) 由边缘化得到。证明完毕。

前一定理值得注意：边缘速率矩阵实际上是概率 $p_{1|t}(z_j=v_i|x)$ 的一种重新参数化。这实际上无非是对每个词元位置 $j=1,\cdots,d$ 学习一个分类器（classifier）。换言之，我们可以简单地定义一个**去噪概率网络**（denoising probabilities network）

$$
p_{1|t}^\theta:x\mapsto\bigl(p_{1|t}^\theta(z_j=v_i|x)\bigr)_{j=1,\cdots,d,\ v_i\in\mathcal{V}}.
$$

注意，网络输出的形状为 $d\times V$。可以通过简单的 softmax 层得到每个词元位置上的概率。网络本身可以是标准的序列到序列（sequence-to-sequence）网络，例如 Transformer 即可（见第 6.1.2 节，扩散 Transformer）。

由于这只是每个位置 $j$ 上的一个分类器，可以通过每个 $j=1,\cdots,d$ 上的交叉熵损失（cross-entropy loss）来训练这样的网络。这引出由下式给出的**离散流匹配损失**

$$
\mathcal{L}_{\text{DFM}}(\theta)=\mathbb{E}_{z\sim p_{\mathrm{data}},\, t\sim\mathrm{Unif}_{[0,1]},\, x\sim p_t(\cdot|z)}\left[\sum_{j=1}^{d}-\log p_{1|t}^\theta(z_j|x)\right].
$$

这一点值得注意：要训练一个生成模型，我们需要做的全部事情就是对每个位置 $j$ 训练一个分类器模型。正如连续流匹配化归为简单的回归（regression）（见第 3 节），离散流匹配与离散扩散模型化归为简单的分类（classification）训练。训练算法概括于算法 8。训练之后，可以按算法 7 抽样。

**例 39（掩码扩散语言模型（masked diffusion language models, MDLMs））.** 上述方法的一个具体情形是**掩码扩散语言模型**。掩码扩散语言模型的想法是，可以把词元词表 $\mathcal{V}=\{v_1,\cdots,v_V\}$ 扩展一个新词元 $\text{[mask]}$，表示该词元缺失（或曾被掩码）。具体地说，我们令 $\mathcal{V}=\{v_1,\cdots,v_V,\text{[mask]}\}$，而初始点就是 $\text{[mask]}^d$，即全部被掩码的序列。形式地，这意味着在上述框架中令 $p_{\mathrm{init}}=\delta_{\text{[mask]}^d}$。采样过程见掩码扩散语言模型轨迹示意图（图 20）。

**算法 8　训练因子化连续时间马尔可夫链模型（离散扩散）**

**输入.** 序列数据集，其中 $z\sim p_{\mathrm{data}}$，且 $z=(z_1,\cdots,z_d)\in\mathcal{V}^d$；$\mathcal{V}$ 上的初始（噪声）词元边缘分布 $p_{\mathrm{init}}^{(j)}$；调度 $\kappa_t\in[0,1]$；返回 $\mathcal{V}$ 上逐位置对数几率（logits）的后验网络（posterior network）$f_\theta$；优化器（optimizer）$\mathrm{Opt}$。

1. 对每一次训练迭代：
   - 抽取一个数据点 $z\sim p_{\mathrm{data}}$。
   - 抽取时间 $t\sim\mathrm{Unif}[0,1]$，并计算 $\kappa\leftarrow\kappa_t$。
   - 抽取带噪状态 $x\sim p_t(\cdot\mid z)$（因子化混合路径）：
   - 对 $j=1,\cdots,d$（**并行**）：
     - 抽取掩码 $m_j\sim\mathrm{Bernoulli}(\kappa)$。
     - 抽取噪声词元 $\xi_j\sim p_{\mathrm{init}}^{(j)}$。
     - 令 $x_j\leftarrow m_j\, z_j+(1-m_j)\,\xi_j$。
   - $x\leftarrow(x_1,\cdots,x_d)$。
   - 通过网络给出的对数几率预测终点词元的后验：

$$
\ell_j(\cdot)\leftarrow f_\theta(x,t)_j\qquad\Rightarrow\qquad p^\theta_{1|t}(v\mid x)_j=\mathrm{Softmax}\bigl(\ell_j\bigr)(v).
$$

   - 离散流匹配损失（$z$ 的逐词元负对数似然（negative log-likelihood, NLL））：

$$
\mathcal{L}_{\text{DFM}}(\theta)\leftarrow\sum_{j=1}^d\Bigl[-\log p^\theta_{1|t}(z_j\mid x)_j\Bigr].
$$

   - 更新参数：$\theta\leftarrow\mathrm{Opt.step}\bigl(\nabla_\theta\mathcal{L}_{\text{DFM}}(\theta)\bigr)$。

**图注.**（图 20）掩码扩散语言模型轨迹的示意图。

至此，我们完成了训练与采样连续时间马尔可夫链模型的完整流程，从而可以生成文本等离散序列。当前最先进的离散扩散模型（Bie et al., 2025）使用本文所述的做法，神经网络（通常是 Transformer）在网络规模的数据（web-scale data）上训练。

**注 40（生成元匹配（Generator Matching））.** 你或许会问，为什么流模型与扩散模型的原理可以如此顺畅地翻译到离散状态空间。事实上，流匹配的原理并非流甚至连续时间马尔可夫链所独有。它们其实是用*马尔可夫过程*构造生成模型的一般学习原理。这一想法引出**生成元匹配**框架（Holderrieth et al., 2024），该框架把离散与连续的流模型及扩散模型推广并统一为一个框架。生成元（generator）是向量场 $u_t$ 与速率矩阵 $Q_t$ 的推广。马尔可夫过程与生成元可以为任何数据模态和状态空间来构造。例如，你可以为光滑流形（smooth manifold）构造模型（Chen and Lipman, 2023；De Bortoli et al., 2022）（例如几何数据），为混合状态空间（mixed state space）构造模型（例如文本与图像的联合生成）（Campbell et al., 2024），以及为跳跃过程（jump process）等其他马尔可夫过程构造模型（Holderrieth et al., 2024；Campbell et al., 2023）。

---

<!-- 来源文件：appendix.zh.md -->

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
