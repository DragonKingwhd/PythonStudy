"""Chapter_9 — 来自 Inside-Deep-Learning 仓库，自动转成可运行 .py 脚本。

原始 notebook 位于 ../Inside-Deep-Learning/。
"""

# ====== 单元 0 (代码) ======
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import pandas as pd

from sklearn.metrics import accuracy_score

import time

from idlmam import train_network, Flatten, weight_reset, View, set_seed

# ====== 单元 1 (代码) ======
# [已剥离] %matplotlib inline
# [已剥离] from IPython.display import set_matplotlib_formats
# [已剥离] set_matplotlib_formats('png', 'pdf')

# ====== 单元 2 (代码) ======
torch.backends.cudnn.deterministic=True
print(set_seed(42))

# ====== 单元 3 (代码) ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch.cuda.is_available() else torch.device("cpu")

# ====== 单元 4 (代码) ======
batch_size = 128
latent_d = 128
neurons = 512
out_shape = (-1, 28, 28) # 也可以用 (-1, 1, 28, 28) 表示 1 通道，但后续 numpy 代码会更繁琐
num_epochs = 10

def fcLayer(in_neurons, out_neurons, leak=0.1): # 我们的辅助函数
    """
    in_neurons: 该层的输入个数
    out_neurons: 该层的输出个数
    leak: LeakyReLU 的泄漏值。
    """
    return nn.Sequential(
        nn.Linear(in_neurons, out_neurons),
        nn.LeakyReLU(leak),
        nn.LayerNorm(out_neurons)
    )

# ====== 单元 5 (代码) ======
def simpleGAN(latent_d, neurons, out_shape, sigmoidG=False, leak=0.2):
    """
    该函数会创建一个简单的 GAN 供我们训练。返回一个元组 (G, D)，
    分别是生成器和判别器网络。

    latent_d: 作为生成器 G 输入的潜变量数量。
    neurons: 每个隐藏层使用的隐藏神经元数量
    out_shape: 判别器 D 的输出形状（也是真实数据的形状）。
    sigmoidG: 若为 True，生成器 G 末尾会接一个 Sigmoid 激活；
        若为 False，则直接返回无界激活值。
    """
    G = nn.Sequential(
        fcLayer(latent_d, neurons, leak),
        fcLayer(neurons, neurons, leak),
        fcLayer(neurons, neurons, leak),
        nn.Linear(neurons, abs(np.prod(out_shape)) ),# np.prod 会把 shape 中所有值相乘，得到所需输出总数；abs 用于消除批维度 "-1" 的影响。
        View(out_shape)# 将输出 reshape 成 D 期望的形状。
    )
    # 有时希望 G 输出 sigmoid 值（即 [0,1]），有时不希望，因此用条件分支包装
    if sigmoidG:
        G = nn.Sequential(G, nn.Sigmoid())
    
    D = nn.Sequential(
        nn.Flatten(),
        fcLayer(abs(np.prod(out_shape)), neurons, leak),
        fcLayer(neurons, neurons, leak),
        fcLayer(neurons, neurons, leak),
        nn.Linear(neurons, 1 ) # 二分类问题，D 输出 1 维
    )
    return G, D

# ====== 单元 6 (代码) ======
G, D = simpleGAN(latent_d, neurons, out_shape, sigmoidG=True)

# ====== 单元 7 (代码) ======
G.to(device)
D.to(device)

# 初始化 BCEWithLogitsLoss 损失函数。BCE 损失用于二分类问题，我们的任务正是判别真假
loss_func = nn.BCEWithLogitsLoss()

# 训练期间真/假标签的约定
real_label = 1
fake_label = 0

# 为 G 和 D 分别设置 Adam 优化器
optimizerD = torch.optim.AdamW(D.parameters(), lr=0.0001, betas=(0.0, 0.9))
optimizerG = torch.optim.AdamW(G.parameters(), lr=0.0001, betas=(0.0, 0.9))

# ====== 单元 8 (代码) ======
train_data = torchvision.datasets.MNIST("./", train=True, transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST("./", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# ====== 单元 9 (代码) ======
G_losses = []
D_losses = []

for epoch in tqdm(range(num_epochs)):
    for data, class_label in tqdm(train_loader, leave=False):
        # 准备 batch 并制作标签
        real_data = data.to(device)
        y_real = torch.full((batch_size,1), real_label, dtype=torch.float32, device=device)
        y_fake = torch.full((batch_size,1), fake_label, dtype=torch.float32, device=device)

        # 步骤 1) $\ell ( D( x_{\mathit{real}}) ,\ y_{\mathit{real}})\$ 和 $\ell ( D(\boldsymbol{x}_{\mathit{fake}}) ,\ y_{\mathit{fake}})$
        D.zero_grad()

        # 在全真 batch 上计算损失
        errD_real = loss_func(D(real_data), y_real)
        # 反向传播为 D 计算梯度
        errD_real.backward()

        ## 用全假 batch 进行训练
        # 采样潜变量 $z \sim \mathcal{N}(\vec{0}, 1)$
        z = torch.randn(batch_size, latent_d, device=device)
        # 用 G 生成假图像 batch
        # 用 D 对全假 batch 分类。保存以便第 2 步复用。
        fake = G(z)
        # 为什么这里要 detach？因为我们不希望梯度影响 G。
        # 当前目标只是更新判别器。
        # 但是，我们还会复用这些假数据来更新判别器，所以要保留
        # 未 detach 的版本！
        # 在全假 batch 上计算 D 的损失
        errD_fake = loss_func(D(fake.detach()), y_fake)
        # 计算本 batch 的梯度
        errD_fake.backward()
        # 累加全真和全假 batch 的梯度
        errD = errD_real + errD_fake
        # 更新 D
        optimizerD.step()

        # 步骤 2) $\ell ( D(\boldsymbol{x}_{\mathit{fake}}) ,\ y_{\mathit{real}})$
        G.zero_grad()
        # 由于刚刚更新了 D，再次对全假 batch 做一次 D 的前向传播
        # 基于此输出计算 G 的损失
        errG = loss_func(D(fake), y_real)
        # 计算 G 的梯度
        errG.backward()
        # 更新 G
        optimizerG.step()

        G_losses.append(errG.item())
        D_losses.append(errD.item())

# ====== 单元 10 (代码) ======
with torch.no_grad():
    noise = torch.randn(batch_size, latent_d, device=device) #$\boldsymbol{z} \sim \mathcal{N}(\vec{0}, \boldsymbol{I})$
    fake_digits = G(noise)
    scores = torch.sigmoid(D(fake_digits))
    
    fake_digits = fake_digits.cpu()
    scores = scores.cpu().numpy().flatten()

# ====== 单元 11 (代码) ======
def plot_gen_imgs(fake_digits, scores=None):
    batch_size = fake_digits.size(0)
    # 此代码假设处理的是黑白图像
    fake_digits = fake_digits.reshape(-1, fake_digits.size(-1), fake_digits.size(-1))
    i_max = int(round(np.sqrt(batch_size)))
    j_max = int(np.floor(batch_size/float(i_max)))
    f, axarr = plt.subplots(i_max,j_max, figsize=(10,10))
    for i in range(i_max):
        for j in range(j_max):
            indx = i*j_max+j
            axarr[i,j].imshow(fake_digits[indx,:].numpy(), cmap='gray', vmin=0, vmax=1)
            axarr[i,j].set_axis_off()
            if scores is not None:
                axarr[i,j].text(0.0, 0.5, str(round(scores[indx],2)), dict(size=20, color='red'))
print(plot_gen_imgs(fake_digits, scores))
plt.show()

# ====== 单元 12 (代码) ======
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ====== 单元 13 (代码) ======
print(set_seed(42))

# ====== 单元 14 (代码) ======
gausGrid = (3, 3) # 网格的大小
samples_per = 10000 # 网格中每个点的样本数

# ====== 单元 15 (代码) ======
X = [] # 在此存储所有数据
for i in range(gausGrid[0]):
    for j in range(gausGrid[1]): # 这两层循环遍历每个均值中心
        z = np.random.normal(0, 0.05, size=(samples_per, 2)) # 采样一批紧密聚集的点
        z[:,0] += i/1.0-(gausGrid[0]-1)/2.0 # 将该随机样本平移到指定 x 轴位置
        z[:,1] += j/1.0-(gausGrid[1]-1)/2.0 # 同样在 y 轴上平移
        X.append(z) # 收集所有样本
X = np.vstack(X) # 将列表转换为形状为 (N, 2) 的大 numpy 张量

# ====== 单元 16 (代码) ======
plt.figure(figsize=(10,10))
sns.kdeplot(x=X[:,0], y=X[:,1], shade=True, fill=True, thresh=-0.001) # 绘制理想的玩具数据
plt.show()

# ====== 单元 17 (代码) ======
toy_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
toy_loader = DataLoader(toy_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
latent_d = 64
G, D = simpleGAN(latent_d, 512, (-1, 2)) # 为玩具问题构建一个仅含 2 个输出特征的新 GAN

# ====== 单元 18 (代码) ======
G.to(device)
D.to(device)

# 为 G 和 D 分别设置 Adam 优化器。
optimizerD = torch.optim.AdamW(D.parameters(), lr=0.0001, betas=(0.0, 0.9))
optimizerG = torch.optim.AdamW(G.parameters(), lr=0.0001, betas=(0.0, 0.9))

# ====== 单元 19 (代码) ======
for epoch in tqdm(range(20)):
    for i, (data,) in enumerate(tqdm(toy_loader, leave=False), 0):
        # 准备 batch 并制作标签
        real_data = data.to(device)
        y_real = torch.full((batch_size,1), real_label, dtype=torch.float32, device=device)
        y_fake = torch.full((batch_size,1), fake_label, dtype=torch.float32, device=device)

        # 步骤 1) $\ell ( D( x_{\mathit{real}}) ,\ y_{\mathit{real}})\$ 和 $\ell ( D(\boldsymbol{x}_{\mathit{fake}}) ,\ y_{\mathit{fake}})$
        D.zero_grad()

        # 在全真 batch 上计算损失
        errD_real = loss_func(D(real_data), y_real)
        # 反向传播为 D 计算梯度
        errD_real.backward()

        ## 用全假 batch 进行训练
        # 采样潜变量 $z \sim \mathcal{N}(\vec{0}, 1)
        z = torch.randn(batch_size, latent_d, device=device)
        # 用 G 生成假图像 batch
        # 用 D 对全假 batch 分类。保存以便第 2 步复用。
        fake = G(z)
        # 为什么这里要 detach？因为我们不希望梯度影响 G。
        # 当前目标只是更新判别器。
        # 但是，我们还会复用这些假数据来更新判别器，所以要保留
        # 未 detach 的版本！
        # 在全假 batch 上计算 D 的损失
        errD_fake = loss_func(D(fake.detach()), y_fake)
        # 计算本 batch 的梯度
        errD_fake.backward()
        # 累加全真和全假 batch 的梯度
        errD = errD_real + errD_fake
        # 更新 D
        optimizerD.step()

        # 步骤 2) $\ell ( D(\boldsymbol{x}_{\mathit{fake}}) ,\ y_{\mathit{real}})$
        G.zero_grad()
        # 由于刚刚更新了 D，再次对全假 batch 做一次 D 的前向传播
        # 基于此输出计算 G 的损失
        errG = loss_func(D(fake), y_real)
        # 计算 G 的梯度
        errG.backward()
        # 更新 G
        optimizerG.step()

        G_losses.append(errG.item())
        D_losses.append(errD.item())

# ====== 单元 20 (代码) ======
with torch.no_grad():
    noise = torch.randn(X.shape[0], latent_d, device=device) # 采样一些随机 $\boldsymbol{z} \sim \mathcal{N}(0,1)$
    fake_samples = G(noise).cpu().numpy() # 生成假数据 $G(\boldsymbol{z})$

# ====== 单元 21 (代码) ======
plt.figure(figsize=(10,10))
sns.kdeplot(x=fake_samples[:,0], y=fake_samples[:,1], shade=True, thresh=-0.001) # 绘制 G 学到的玩具数据分布
plt.xlim(-1.5, 1.5)# 手动将 x 轴设置到数据集原始的范围
plt.ylim(-1.5, 1.5)# y 轴同理
plt.show()

# ====== 单元 22 (代码) ======
set_seed(42)
toy_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
toy_loader = DataLoader(toy_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# ====== 单元 23 (代码) ======
def train_wgan(D, G, loader, latent_d, epochs=20, d_updates=1, device="cpu"):
    G_losses = []
    D_losses = []

    G.to(device)
    D.to(device)

    # 为 G 和 D 分别设置 Adam 优化器
    optimizerD = torch.optim.AdamW(D.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerG = torch.optim.AdamW(G.parameters(), lr=0.0001, betas=(0.0, 0.9))

    for epoch in tqdm(range(epochs)):
        for count, data in enumerate(tqdm(loader, leave=False)):
            if isinstance(data, tuple) or len(data) == 2:
                data, class_label = data
            elif isinstance(data, list) and len(data) == 1:
                data = data[0]
            batch_size = data.size(0)
            real = data.to(device)
            
            D.zero_grad()
            G.zero_grad()

            # 步骤 1) D-score、G-score 和梯度惩罚
            # D 在真实数据上的表现
            D_success = D(real)

            ## 用全假 batch 进行训练
            # 采样一批潜变量
            noise = torch.randn(batch_size, latent_d, device=device)
            # 用 G 生成假图像 batch
            fake = G(noise)
            # 用 D 对全假 batch 分类
            D_failure = D(fake)

            # 现在计算梯度惩罚
            eps_shape = [batch_size]+[1]*(len(data.shape)-1)
            eps = torch.rand(eps_shape, device=device)
            fake = eps*real + (1-eps)*fake
            output = D(fake)

            grad = torch.autograd.grad(outputs=output, inputs=fake,
                                  grad_outputs=torch.ones(output.size(), device=device),
                                  create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]

            D_grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()

            # 在全假 batch 上计算 D 的损失
            errD = (D_failure-D_success).mean() + D_grad_penalty.mean()*10
            errD.backward()
            # 更新 D
            optimizerD.step()

            D_losses.append(errD.item())

            if count % d_updates != d_updates-1:
                continue

            # 步骤 2) -D(G(z))
            D.zero_grad()
            G.zero_grad()
            # 由于刚刚更新了 D，再次对全假 batch 做一次 D 的前向传播

            noise = torch.randn(batch_size, latent_d, device=device)
            output = -D(G(noise))
            # 基于此输出计算 G 的损失
            errG = output.mean()
            # 计算 G 的梯度
            errG.backward()
            # 更新 G
            optimizerG.step()
            
            G_losses.append(errG.item())
            
    return D_losses, G_losses

# ====== 单元 24 (代码) ======
G, D = simpleGAN(latent_d, 512, (-1, 2))
train_wgan(D, G, toy_loader, latent_d, epochs=20, device=device)
G, D = G.eval(), D.eval()

# ====== 单元 25 (代码) ======
with torch.no_grad():
    noise = torch.randn(X.shape[0], latent_d, device=device)
    fake_samples_w = G(noise).cpu().numpy()
plt.figure(figsize=(10,10))
ax = sns.kdeplot(x=fake_samples_w[:,0], y=fake_samples_w[:,1], shade=True, thresh=-0.001)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()

# ====== 单元 26 (代码) ======
latent_d = 128
out_shape = (-1, 1, 28, 28)
G, D = simpleGAN(latent_d, neurons, out_shape, sigmoidG=True)

D_losses, G_losses = train_wgan(D, G, train_loader, latent_d, epochs=40, device=device)

G = G.eval()
D = D.eval()

# ====== 单元 27 (代码) ======
with torch.no_grad():
    noise = torch.randn(batch_size, latent_d, device=device)
    fake_digits = G(noise)
    scores = D(fake_digits)
    
    fake_digits = fake_digits.cpu()
    scores = scores.cpu().numpy().flatten()
print(plot_gen_imgs(fake_digits))

# ====== 单元 28 (代码) ======
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(np.convolve(G_losses, np.ones((100,))/100, mode='valid') ,label="G")
plt.plot(np.convolve(D_losses, np.ones((100,))/100, mode='valid') ,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ====== 单元 29 (代码) ======
print(set_seed(42))

# ====== 单元 30 (代码) ======
start_size = 28//4  # 初始宽和高，便于做两轮转置卷积
latent_channels = 16
latent_d_conv = latent_channels*(start_size**2) # 潜空间所需的值个数
in_shape = (-1, latent_channels, start_size, start_size )

# ====== 单元 31 (代码) ======
n_filters = 32 # 进入潜空间的通道数
k_size= 5 # 卷积 GAN 默认使用的卷积核大小
k_size_t = 4 # 转置卷积默认的卷积核大小
leak = 0.2

# 用于创建一个隐藏卷积层的辅助函数。
def cnnLayer(in_channels, out_channels, filter_size, wh_size, leak=0.2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, filter_size, padding=filter_size//2),
        nn.LeakyReLU(leak),
        nn.LayerNorm([out_channels, wh_size, wh_size]),
    )
# 与 cnnLayer 类似，但使用转置卷积来扩大空间尺寸
def tcnnLayer(in_channels, out_channels, wh_size, leak=0.2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, k_size_t, padding=1, output_padding=0, stride=2), 
        nn.LeakyReLU(leak),
        nn.LayerNorm([out_channels, wh_size, wh_size]),
    )

# ====== 单元 32 (代码) ======
G = nn.Sequential(
    View(in_shape),
    cnnLayer(latent_channels, n_filters, k_size, 28//4, leak),
    cnnLayer(n_filters, n_filters, k_size, 28//4, leak),
    cnnLayer(n_filters, n_filters, k_size, 28//4, leak),
    tcnnLayer(n_filters, n_filters//2, 28//2, leak),
    cnnLayer(n_filters//2, n_filters//2, k_size, 28//2, leak),
    cnnLayer(n_filters//2, n_filters//2, k_size, 28//2, leak),
    tcnnLayer(n_filters//2, n_filters//4, 28, leak),
    cnnLayer(n_filters//4, n_filters//4, k_size, 28, leak),
    cnnLayer(n_filters//4, n_filters//4, k_size, 28, leak),
    nn.Conv2d(n_filters//4, 1, k_size, padding=k_size//2), 
    nn.Sigmoid(),
)

# ====== 单元 33 (代码) ======
D = nn.Sequential(
    cnnLayer(1, n_filters, k_size, 28, leak),
    cnnLayer(n_filters, n_filters, k_size, 28, leak),
    nn.AvgPool2d(2), # 为避免稀疏梯度，这里使用平均池化而非最大池化。
    cnnLayer(n_filters, n_filters, k_size, 28//2, leak),
    cnnLayer(n_filters, n_filters, k_size, 28//2, leak),
    nn.AvgPool2d(2),
    cnnLayer(n_filters, n_filters, 3, 28//4, leak),
    cnnLayer(n_filters, n_filters, 3, 28//4, leak),
    nn.AdaptiveAvgPool2d(4),# 注意这里使用的是自适应池化，所以可以保证此处尺寸为 4x4。既能更激进地池化（对卷积 GAN 通常有利），又能让代码更简单。
    nn.Flatten(),
    nn.Linear(n_filters*4**2,256),
    nn.LeakyReLU(leak),
    nn.Linear(256,1),
)

# ====== 单元 34 (代码) ======
D_losses, G_losses = train_wgan(D, G, train_loader, latent_d_conv, epochs=15, device=device)

G = G.eval()
D = D.eval()

# ====== 单元 35 (代码) ======
with torch.no_grad():
    noise = torch.randn(batch_size, latent_d_conv, device=device)
    fake_digits = G(noise)
    scores = D(fake_digits)
    
    fake_digits = fake_digits.cpu()
    scores = scores.cpu().numpy().flatten()
print(plot_gen_imgs(fake_digits))

# ====== 单元 36 (代码) ======
plt.figure(figsize=(10,5))
plt.title("Conv-WGAN Generator and Discriminator Loss")
plt.plot(np.convolve(G_losses, np.ones((100,))/100, mode='valid') ,label="G")
plt.plot(np.convolve(D_losses, np.ones((100,))/100, mode='valid') ,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ====== 单元 37 (代码) ======
print(set_seed(42))

# ====== 单元 38 (代码) ======
class ConditionalWrapper(nn.Module):
    def __init__(self, input_shape, neurons, classes, main_network, leak=0.2):
        """
        input_shape: 潜变量 $\boldsymbol{z}$ 应有的形状。
        neurons: 隐藏层使用的神经元数
        classes: 标签 $y$ 的类别数
        main_network: 生成器 $G$ 或判别器 $D$
        """
        super().__init__()

        self.input_shape = input_shape
        self.classes = classes
        # 根据潜变量形状计算潜参数的数量
        input_size = abs(np.prod(input_shape))
        # 创建一个 embedding 层，将标签转换为向量
        self.label_embedding = nn.Embedding(classes, input_size)

        # 在 forward 中，我们会把标签与原始数据拼接成一个向量。然后这个 'combiner' 会将这个加大的张量
        # 转换为原始 'input_shape' 大小的新张量。这样就把（来自 label_embedding 的）条件信息融入到了潜向量中。
        self.combiner = nn.Sequential(
            nn.Flatten(),
            fcLayer(input_size*2, input_size, leak=leak),# 一个 FC 层
            nn.Linear(input_size, input_size),# 第二个 FC 层，但先应用线性与激活
            nn.LeakyReLU(leak),
            View(input_shape), # 这样我们可以根据目标输出形状对输出进行 reshape 并归一化。让 Conditional wrapper 同时适用于线性和卷积模型。
            nn.LayerNorm(input_shape[1:]),
        )
        self.net = main_network


    # forward 函数定义了从输入到输出的计算过程。
    def forward(self, x, condition=None):
        if condition is None:# 如果未给出标签，就随机挑一个
            condition = torch.randint(0, self.classes, size=(x.size(0),), device=x.get_device())
        # 对标签做 embedding 并 reshape 成所需形状
        embd = self.label_embedding(condition)
        # 确保 label embd 与数据 x 形状相同，便于拼接
        embd = embd.view(self.input_shape)
        x = x.view(self.input_shape)
        # 将潜变量输入与 embedding 后的标签拼接起来
        x_comb = torch.cat([x, embd], dim=1)
        # 返回网络在合并输入上的结果
        return self.net(self.combiner(x_comb))

# ====== 单元 39 (代码) ======
latent_d = 128
out_shape = (-1, 1, 28, 28)
in_shape = (-1, latent_d)
classes = 10
G, D = simpleGAN(latent_d, neurons, out_shape, sigmoidG=True)

G = ConditionalWrapper(in_shape, neurons, classes, G)
D = ConditionalWrapper(out_shape, neurons, classes, D)

# ====== 单元 40 (代码) ======
def train_c_wgan(D, G, loader, latent_d, epochs=20, device="cpu"):
    G_losses = []
    D_losses = []

    G = G.to(device)
    D = D.to(device)

    # 为 G 和 D 分别设置 Adam 优化器
    optimizerD = torch.optim.AdamW(D.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerG = torch.optim.AdamW(G.parameters(), lr=0.0001, betas=(0.0, 0.9))

    for epoch in tqdm(range(epochs)):
        for data in tqdm(loader, leave=False):
            if isinstance(data, tuple) or len(data) == 2:
                data, class_label = data
            batch_size = data.size(0)
            D.zero_grad()
            G.zero_grad()
            real = data.to(device)
            class_label = class_label.to(device)
            # 步骤 1) D-score、G-score 和梯度惩罚
            # D 在真实数据上的表现
            D_success = D(real, class_label)

            ## 用全假 batch 进行训练
            # 采样一批潜变量
            noise = torch.randn(batch_size, latent_d, device=device)
            # 用 G 生成假图像 batch
            fake = G(noise, class_label)
            # 用 D 对全假 batch 分类
            D_failure = D(fake, class_label)

            # 现在计算梯度惩罚
            eps_shape = [batch_size]+[1]*(len(data.shape)-1)
            eps = torch.rand(eps_shape, device=device)
            fake = eps*real + (1-eps)*fake
            output = D(fake, class_label)

            grad = torch.autograd.grad(outputs=output, inputs=fake,
                                  grad_outputs=torch.ones(output.size(), device=device),
                                  create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]

            D_grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()

            # 在全假 batch 上计算 D 的损失
            errD = (D_failure-D_success).mean() + D_grad_penalty.mean()*10
            errD.backward()
            # 更新 D
            optimizerD.step()

            D_losses.append(errD.item())

            # 步骤 2) -D(G(z))
            D.zero_grad()
            G.zero_grad()
            # 由于刚刚更新了 D，再次对全假 batch 做一次 D 的前向传播

            noise = torch.randn(batch_size, latent_d, device=device)
            output = -D(G(noise, class_label), class_label)
            # 基于此输出计算 G 的损失
            errG = output.mean()
            # 计算 G 的梯度
            errG.backward()
            # 更新 G
            optimizerG.step()
            
            G_losses.append(errG.item())
            
    return D_losses, G_losses

# ====== 单元 41 (代码) ======
D_losses, G_losses = train_c_wgan(D, G, train_loader, latent_d, epochs=20, device=device)

G = G.eval()
D = D.eval()

# ====== 单元 42 (代码) ======
with torch.no_grad():
    noise =  torch.randn(batch_size, latent_d, device=device) # 像往常一样生成随机噪声
    labels = torch.fmod(torch.arange(0, batch_size, device=device), classes) # 但这里让标签从 0 递增到 9 再循环。
    fake_digits = G(noise, labels)# 现在从噪声生成时传入标签，由标签控制生成具体的数字。
    scores = D(fake_digits, labels)
    
    fake_digits = fake_digits.cpu()
    scores = scores.cpu().numpy().flatten()
print(plot_gen_imgs(fake_digits))

# ====== 单元 43 (代码) ======
with torch.no_grad():
    # 生成 10 个潜噪声向量，并重复 10 次。即复用相同的潜在编码
    noise =  torch.randn(10, latent_d, device=device).repeat((1,10)).view(-1, latent_d)
    # 从 0 数到 9，再循环回 0。重复 10 次
    labels = torch.fmod(torch.arange(0, noise.size(0), device=device), classes)
    # 现在使用相同的潜变量噪声生成 10 张图像，但每次改变标签。
    fake_digits = G(noise, labels)
    scores = D(fake_digits, labels)
    
    fake_digits = fake_digits.cpu()
    scores = scores.cpu().numpy().flatten()
print(plot_gen_imgs(fake_digits))

# ====== 单元 44 (代码) ======
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', model_name='celebAHQ-512', pretrained=True, useGPU=False)

# ====== 单元 45 (代码) ======
import torchvision

# ====== 单元 46 (代码) ======
print(set_seed(3))

# ====== 单元 47 (代码) ======
num_images = 2
noise, _ = model.buildNoiseData(num_images)
with torch.no_grad():
    generated_images = model.test(noise)

# ====== 单元 48 (代码) ======
grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.show()

# ====== 单元 49 (代码) ======
print(noise)

# ====== 单元 50 (代码) ======
steps = 8
interpolated_z = [] # 用于保存插值后的图像
for x in torch.arange(0,steps)/float(steps)+0.5/steps:
    # 取第一个潜变量的 step/steps，加上第二个潜变量的 (1-step/steps)，即 "walking"
    z_mix = x*noise[0,:] + (1-x)*noise[1,:]
    interpolated_z.append(z_mix)
# 现在根据插值生成图像
with torch.no_grad():
    mixed_g = model.test(torch.stack(interpolated_z)).clamp(min=-1, max=1)
# 当我们可视化时，会看到生成的输出像是两者的合适混合！
grid = torchvision.utils.make_grid(mixed_g.clamp(min=-1, max=1), scale_each=True, normalize=True)
plt.figure(figsize=(15,10))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.show()

# ====== 单元 51 (代码) ======
set_seed(3)# 获得可复现的结果
# 生成一些随机样本
noise, _ = model.buildNoiseData(8*4)
with torch.no_grad():
    generated_images = model.test(noise)
# 将它们全部可视化。
grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
plt.figure(figsize=(13,6))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.show()

# ====== 单元 52 (代码) ======
# 标记每张图像是否看起来是男性或微笑。这两个列表是手工逐张标注得到的。
male = [0, 1, 0, 0, 1, 0, 0, 0, 
       1, 1, 1, 1, 0, 1, 0, 0, 
       1, 0, 0, 1, 0, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 1, 0]
smile = [1, 1, 0, 0, 1, 0, 1, 1,
        0, 0, 0, 0, 0, 1, 0, 0, 
        1, 0, 1, 0, 1, 1, 1, 1,
        0, 0, 1, 1, 0, 0, 0, 1]
male = np.array(male, dtype=np.bool)
smile = np.array(smile, dtype=np.bool)

# 将形状从 (32) 转换为 (32, 1)
male = torch.tensor(np.expand_dims(male, axis=-1))
smile = torch.tensor(np.expand_dims(smile, axis=-1))

# ====== 单元 53 (代码) ======
def extractVec(labels, noise):
    posVec = torch.sum(noise*labels, axis=0)/torch.sum(labels) # 取类别标签为 0 的所有样本的平均
    negVec = torch.sum(noise*~labels, axis=0)/torch.sum(~labels) # 类别标签为 1 的样本的平均
    return posVec-negVec # 用两者均值之差来近似两种潜在概念之间的 "差异"。

# ====== 单元 54 (代码) ======
# 提取 "性别" 向量
gender_vec = extractVec(male, noise)
with torch.no_grad():
    # 把性别向量加到原始潜变量上，生成新图像
    generated_images = model.test(noise+gender_vec)
# 绘制结果！
grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
plt.figure(figsize=(13,6))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.show()

# ====== 单元 55 (代码) ======
with torch.no_grad():
    generated_images = model.test(noise-gender_vec)
grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
plt.figure(figsize=(13,6))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.show()

# ====== 单元 56 (代码) ======
smile_vec = extractVec(smile, noise)
with torch.no_grad():
    generated_images = model.test(noise+smile_vec)
grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
plt.figure(figsize=(13,6))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.show()

