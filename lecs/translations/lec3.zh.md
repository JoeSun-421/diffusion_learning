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
