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
