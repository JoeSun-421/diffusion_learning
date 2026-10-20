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
