"""Chapter_14 — 来自 Inside-Deep-Learning 仓库，自动转成可运行 .py 脚本。

原始 notebook 位于 ../Inside-Deep-Learning/。
"""

# ====== 单元 0 (代码) ======
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from idlmam import set_seed

import scipy
import scipy.ndimage

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
# [已剥离] %matplotlib inline
# [已剥离] from IPython.display import set_matplotlib_formats
# [已剥离] set_matplotlib_formats('png', 'pdf')

import pandas as pd

from sklearn.metrics import accuracy_score

import time

from idlmam import LastTimeStep, train_network, Flatten, weight_reset, View, LambdaLayer
from idlmam import AttentionAvg, GeneralScore, DotScore, AdditiveAttentionScore, getMaskByFill

import os

# ====== 单元 1 (代码) ======
# [已剥离] %matplotlib inline
# [已剥离] from IPython.display import set_matplotlib_formats
# [已剥离] set_matplotlib_formats('png', 'pdf')

# ====== 单元 2 (代码) ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch.cuda.is_available() else torch.device("cpu")

# ====== 单元 3 (代码) ======
torch.backends.cudnn.deterministic=True
print(set_seed(42))

# ====== 单元 4 (代码) ======
import requests
from PIL import Image
from io import BytesIO

#这张图片来自 Sajjad Fazel https://commons.wikimedia.org/wiki/User:SajjadF
url = "https://upload.wikimedia.org/wikipedia/commons/9/9c/Zebra_in_Mikumi.JPG"

response = requests.get(url)
img = Image.open(BytesIO(response.content))

# ====== 单元 5 (代码) ======
to_tensor = transforms.ToTensor() # 该 transform 将 PIL 图像转为 PyTorch 张量
resize = torchvision.transforms.Resize(1000) #将最小边缩放为 1000 像素
crop = torchvision.transforms.CenterCrop((1000, 1000)) #裁取中心 1000x1000 像素
img_tensor_big = to_tensor(crop(resize(img))) #组合三种变换以处理图像

# ====== 单元 6 (代码) ======
to_img = transforms.ToPILImage()
print(to_img(img_tensor_big))

# ====== 单元 7 (代码) ======
shrink_factor = 4 # 池化的程度
img_tensor_small = F.max_pool2d(img_tensor_big, (shrink_factor,shrink_factor)) #应用池化
print(to_img(img_tensor_small))

# ====== 单元 8 (代码) ======
B = 128
epochs = 30

train_transform = transforms.Compose( #训练用变换：随机裁剪 -> PyTorch 张量
    [
        transforms.RandomCrop((24,24)),
        transforms.ToTensor(),
    ])
test_transform = transforms.Compose( #测试用变换：中心裁剪 -> PyTorch 张量
    [
        transforms.CenterCrop((24,24)),
        transforms.ToTensor(),
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=B, shuffle=True, num_workers=2)
#保留完整 32x32 图像的测试集副本，方便测试特定裁剪
testset_nocrop = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
#评估时使用的测试 loader 采用确定性的中心裁剪
test_loader = torch.utils.data.DataLoader(testset, batch_size=B, shuffle=False, num_workers=2)
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') #将 CIFAR10 类别索引映射回原始名称

# ====== 单元 9 (代码) ======
f, axarr = plt.subplots(1,4, figsize=(20,10)) #制作一个 1x4 的网格
for i in range(4):
    x, y = trainset[30] #从训练集中取一张特定样本（我比较喜欢飞机）
    axarr[i].imshow(x.numpy().transpose(1,2,0)) #调整为 numpy/matplotlib 图像偏好的 (W, H, C) 顺序
    axarr[i].text(0.0, 0.5, cifar10_classes[y].upper(), dict(size=30, color='black')) #在角落标注类别名
plt.show()

# ====== 单元 10 (代码) ======
C = 3 #输入通道数
h = 16 #隐藏层通道数
filter_size = 3
pooling_rounds = 2

def cnnLayer(in_size, out_size, filter_size): #辅助函数，与之前多次写过的类似
    return nn.Sequential(
        nn.Conv2d(in_size, out_size, filter_size, padding=filter_size//2),
        nn.BatchNorm2d(out_size),
        nn.ReLU())

normal_CNN = nn.Sequential( #普通 CNN，由若干"两层卷积 + 最大池化"块组成
    cnnLayer(C, h, filter_size),
    cnnLayer(h, h, filter_size),
    nn.MaxPool2d(2),
    cnnLayer(h, h, filter_size),
    cnnLayer(h, h, filter_size),
    nn.MaxPool2d(2),
    cnnLayer(h, h, filter_size),
    cnnLayer(h, h, filter_size),
    nn.Flatten(),
    nn.Linear(h*(24//(2**pooling_rounds))**2, len(cifar10_classes)) # $\text{# channels} \cdot \left(\frac{24 \text{pixels}}{2^{\text{rounds of pooling}}}\right)^2 = $ 最终层的输入数量
)

loss = nn.CrossEntropyLoss()
#设置优化器并配合学习率调度器以获得最佳性能
optimizer = torch.optim.AdamW(normal_CNN.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
#按常规方式训练模型
normal_results = train_network(normal_CNN, loss, train_loader, epochs=epochs, device=device,  test_loader=test_loader, optimizer=optimizer, lr_schedule=scheduler, score_funcs={'Accuracy': accuracy_score})

# ====== 单元 11 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=normal_results, label='Regular')
plt.show()

# ====== 单元 12 (代码) ======
test_img_id = 213 # 要选取的测试图像
x, y = testset_nocrop[test_img_id] # 获取原始 32x32 图像
offset_predictions = [] #将每个 24x24 子图的预测结果存到这里
normal_CNN = normal_CNN.eval()
for i in range(8): # 上下平移
    for j in range(8): #左右平移
        x_crop = x[:,i:i+24, j:j+24].to(device) #取裁剪后的图像
        with torch.no_grad():
            prob_y = F.softmax(normal_CNN(x_crop.unsqueeze(0)), dim=-1).cpu().numpy()[0,y] #分类图像并取得正确类别的概率
            offset_predictions.append((x_crop, prob_y)) #保存结果分数

# ====== 单元 13 (代码) ======
f, axarr = plt.subplots(8,8, figsize=(10,10)) # 8x8 图像网格
for i in range(8): #遍历每一行
    pos = 0 #记录当前所在的平移位置
    for x, score in offset_predictions[i*8:][:8]: #取接下来的 8 张图填充列
        axarr[i, pos].imshow(x.cpu().numpy().transpose(1,2,0)) #绘制 24x24 子图
        axarr[i, pos].text(0.0, 0.5, str(round(score,2)), dict(size=20, color='green'))#在左上角打印正确类别的概率
        pos += 1 #移动到下一个图像位置
plt.show()

# ====== 单元 14 (代码) ======
class BlurLayer(nn.Module):
    def __init__(self, kernel_size=5, stride=2, D=2):
        """
        kernel_size: 模糊核的宽度
        stride: 输出的缩小程度
        D: 输入的维数。D=1、D=2、D=3 分别对应形状为 (B, C, W)、(B, C, W, H)、(B, C, W, H, Z) 的张量。
        """
        super(BlurLayer, self).__init__()

        base_1d = scipy.stats.binom.pmf(list(range(kernel_size)), kernel_size, p=0.5)#构造一维二项分布。该计算可得到所有 k 值对应的归一化 filter_i 值。
        #z 是一个 1d 滤波器
        if D <= 0 or D > 3:
            raise Exception() #D 取值非法！
        if D >= 1:
            z = base_1d #保持不变
        if D >= 2:
            z = base_1d[:,None]*z[None,:] #2-d 滤波器由两个 1-d 滤波器相乘得到
        if D >= 3:
            z = base_1d[:,None,None]*z #3-d 滤波器由 2-d 与 1-d 相乘得到
        #应用滤波器即卷积运算，因此将其作为此层的参数保存。requires_grad=False 表示不希望它被更新
        self.weight = nn.Parameter(torch.tensor(z, dtype=torch.float32).unsqueeze(0), requires_grad=False)
        self.stride = stride

    def forward(self, x):
        C = x.size(1) #当前有多少通道？
        ks = self.weight.size(0)#内部滤波器多宽？

        #三种调用本质相同，只需根据维数选择对应的卷积函数
        #由于不像普通卷积层那样有多个滤波器，groups 参数用于将同一个滤波器作用到每个通道
        if len(self.weight.shape)-1 == 1:
            return F.conv1d(x, torch.stack([self.weight]*C), stride=self.stride, groups=C, padding=ks//self.stride)
        elif len(self.weight.shape)-1 == 2:
            return F.conv2d(x, torch.stack([self.weight]*C), stride=self.stride, groups=C, padding=ks//self.stride)
        elif len(self.weight.shape)-1 == 3:
            return F.conv3d(x, torch.stack([self.weight]*C), stride=self.stride, groups=C, padding=ks//self.stride)
        else:
            raise Exception() #这段代码理论上不会被执行，若出现则说明存在 bug

# ====== 单元 15 (代码) ======
tmp = F.max_pool2d(img_tensor_big, (shrink_factor,shrink_factor), stride=1, padding=shrink_factor//2) #使用步幅为 1 的最大池化
img_tensor_small_better = BlurLayer(kernel_size=int(1.5*shrink_factor), stride=shrink_factor)(tmp.unsqueeze(0)) #对最大池化结果进行模糊
print(to_img(img_tensor_small_better.squeeze()))

# ====== 单元 16 (代码) ======
class MaxPool2dAA(nn.Module):
    def __init__(self, kernel_size=2, ratio=1.7):
        """
        kernel_size: 池化大小
        ratio: 模糊滤波器相对池化大小的放大倍数
        """
        super(MaxPool2dAA, self).__init__()

        blur_ks = int(ratio*kernel_size) #构造一个略大一些的模糊滤波器
        self.blur = BlurLayer(kernel_size=blur_ks, stride=kernel_size, D=2) #创建模糊核
        self.kernel_size = kernel_size #保存池化大小

    def forward(self, x):
        ks = self.kernel_size
        tmp = F.max_pool2d(x, ks, stride=1, padding=ks//2) #使用步幅为 1 的池化
        return self.blur(tmp) #对结果进行模糊

# ====== 单元 17 (代码) ======
aaPool_CNN = nn.Sequential( #结构与 normal_CNN 相同，只是把池化换成了抗混叠版本
    cnnLayer(C, h, filter_size),
    cnnLayer(h, h, filter_size),
    MaxPool2dAA(2),
    cnnLayer(h, h, filter_size),
    cnnLayer(h, h, filter_size),
    MaxPool2dAA(2),
    cnnLayer(h, h, filter_size),
    cnnLayer(h, h, filter_size),
    nn.Flatten(),
    nn.Linear((24//(2**pooling_rounds))**2*h, len(cifar10_classes))
)

optimizer = torch.optim.AdamW(aaPool_CNN.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

aaPool_results = train_network(aaPool_CNN, loss, train_loader, epochs=epochs, device=device, test_loader=test_loader, optimizer=optimizer, lr_schedule=scheduler, score_funcs={'Accuracy': accuracy_score})

# ====== 单元 18 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=normal_results, label='Regular')
sns.lineplot(x='epoch', y='test Accuracy', data=aaPool_results, label='Anti-Alias Pooling')
plt.show()

# ====== 单元 19 (代码) ======
x, y = testset_nocrop[test_img_id] # 获取原始 32x32 图像
offset_predictions_aa = [] #将每个 24x24 子图的预测结果存到这里
aaPool_CNN = aaPool_CNN.eval()
for i in range(8): # 上下平移
    for j in range(8): #左右平移
        x_crop = x[:,i:i+24, j:j+24].to(device) #取裁剪后的图像
        with torch.no_grad():
            prob_y = F.softmax(aaPool_CNN(x_crop.unsqueeze(0)), dim=-1).cpu().numpy()[0,y] #分类图像并取得正确类别的概率
            offset_predictions_aa.append((x_crop, prob_y)) #保存结果分数

sns.lineplot(x=list(range(8*8)), y=[val for img,val in offset_predictions], label='Regular')
ax = sns.lineplot(x=list(range(8*8)), y=[val for img,val in offset_predictions_aa], label='Anti-Alias Pooling')
ax.set(xlabel='Pixel shifts', ylabel='Predicted probability of correct class')
plt.show()

# ====== 单元 20 (代码) ======
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, channels, kernel_size=3, stride=1, activation=nn.ReLU(), ReZero=True):
        """
        in_channels: 输入到该残差块的通道数
        channels: 该残差块的输出通道数
        kernel_size: 该残差块中使用的滤波器大小
        stride: 该块中卷积的步幅。步幅越大输出越小。
        activation: 使用的激活函数
        ReZero: 是否使用 ReZero 风格的初始化。
        """
        super().__init__()

        self.activation = activation
        #padding 数值，使 W/H 保持不变
        pad = (kernel_size-1)//2
        filter_size = (kernel_size,kernel_size)

        #网络的复杂分支，包含两轮层
        self.F = nn.Sequential(
            nn.Conv2d(in_channels, channels, filter_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
            activation,
            nn.Conv2d(channels, channels, filter_size, padding=pad, stride=stride, bias=False),
            nn.BatchNorm2d(channels),
        )

        #若不使用 ReZero，alpha 为浮点数；使用时则为可学习的 Parameter
        self.alpha = 1.0
        if ReZero:
            self.alpha = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

        #shortcut 是恒等函数，直接返回输入作为输出
        self.shortcut = nn.Identity()
        #若 F 的输出形状会因通道数或步幅改变而不同，
        #则把 shortcut 改为 1x1 卷积作为"投影"，使形状对齐
        if in_channels != channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, padding=0, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x):
        #按需计算 F(x) 与 x 的结果
        f_x = self.F(x)
        x = self.shortcut(x)

        if isinstance(self.alpha,nn.Parameter):#ReZero
            return x + self.alpha * self.activation(f_x)
        else:#普通残差块
            return self.activation(x + f_x)
#说明：ReZero 风格残差块的实现，并带有可选的 shortcut 连接来缩小某层的尺寸。ReZero 方法将 alpha 作为可学习参数，否则使用普通风格的残差块。

# ====== 单元 21 (代码) ======
resnetReZero_cifar10 = nn.Sequential( #使用 ReZero 方法训练一个新的残差网络
    ResidualBlock(C, h, ReZero=True),
    *[ResidualBlock(h, h, ReZero=True) for _ in range(6)],
    ResidualBlock(h, 2*h, ReZero=True, stride=2), #这里不使用池化，而是带步幅的卷积层。这样可以保留跳跃连接而无需额外代码
    *[ResidualBlock(2*h, 2*h, ReZero=True) for _ in range(6)],
    ResidualBlock(2*h, 4*h, ReZero=True, stride=2),
    *[ResidualBlock(4*h, 4*h, ReZero=True) for _ in range(6)],
    ResidualBlock(4*h, 4*h, ReZero=True, stride=2),
    *[ResidualBlock(4*h, 4*h, ReZero=True) for _ in range(6)],
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(4*h, len(cifar10_classes)), #自适应池化已经降到 1x1，因此最终层的输入数量很容易计算
)

optimizer = torch.optim.AdamW(resnetReZero_cifar10.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
resnetReZero_results = train_network(resnetReZero_cifar10, loss, train_loader, epochs=epochs, device=device, test_loader=test_loader, optimizer=optimizer, lr_schedule=scheduler, score_funcs={'Accuracy': accuracy_score})

# ====== 单元 22 (代码) ======
resnet_cifar10 = nn.Sequential( #先训练一个不带 ReZero 的普通残差网络
    ResidualBlock(C, h, ReZero=False),
    *[ResidualBlock(h, h, ReZero=False) for _ in range(6)],
    ResidualBlock(h, 2*h, ReZero=False, stride=2),
    *[ResidualBlock(2*h, 2*h, ReZero=False) for _ in range(6)],
    ResidualBlock(2*h, 4*h, ReZero=False, stride=2),
    *[ResidualBlock(4*h, 4*h, ReZero=False) for _ in range(6)],
    ResidualBlock(4*h, 4*h, ReZero=False, stride=2),
    *[ResidualBlock(4*h, 4*h, ReZero=False) for _ in range(6)],
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(4*h, len(cifar10_classes)), #自适应池化已经降到 1x1，因此最终层的输入数量很容易计算
)
optimizer = torch.optim.AdamW(resnet_cifar10.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
resnet_results = train_network(resnet_cifar10, loss, train_loader, epochs=epochs, device=device, test_loader=test_loader, optimizer=optimizer, lr_schedule=scheduler, score_funcs={'Accuracy': accuracy_score})

# ====== 单元 23 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=normal_results, label='Regular')
sns.lineplot(x='epoch', y='test Accuracy', data=resnet_results, label='ResNet')
sns.lineplot(x='epoch', y='test Accuracy', data=resnetReZero_results, label='ResNet ReZero')
plt.show()

# ====== 单元 24 (代码) ======
range_01 = np.arange(100)[1:]/100 #沿 x 轴取 100 个点用于绘图
for alpha in [0.1, 0.2, 0.3, 0.4]: #演示四个不同的超参数值
    plt.plot(range_01, scipy.stats.beta(alpha, alpha).pdf(range_01), lw=2, ls='-', alpha=0.5, label=r'$\alpha='+str(alpha)+"$") #为每个取值绘制 Beta 分布
plt.xlabel(r"$\lambda \sim Beta(\alpha, \alpha)$")
plt.ylabel(r"PDF")
plt.legend()
plt.show()

# ====== 单元 25 (代码) ======
class MixupLoss(nn.Module):
    def __init__(self, base_loss=nn.CrossEntropyLoss()):
        """
        base_loss: 作为 Mixup 子组件使用的原始损失函数，也用于测试时评估表现。
        """
        super(MixupLoss, self).__init__()
        self.loss = base_loss

    def forward(self, y_hat, y):
        if isinstance(y, tuple): #说明应该执行 mixup！
            if len(y) != 3:
                raise Exception() #应当是 (y_i, y_j, lambda) 的三元组！
            y_i, y_j, lambda_ = y #拆解元组
            return lambda_ * self.loss(y_hat, y_i) + (1 - lambda_) * self.loss(y_hat, y_j)
        #否则 y 是普通张量和普通标签！按常规方式计算
        return self.loss(y_hat, y)

# ====== 单元 26 (代码) ======
from torch.utils.data.dataloader import default_collate

class MixupCollator(object):
    def __init__(self, alpha=0.25, base_collate=default_collate):
        """
        alpha: 数据混合的激进程度，推荐取 [0.1, 0.4]，也可取 [0, 1]
        base_collate: 将数据点列表合并为一个 batch 的函数。默认与 PyTorch DataLoader 类的默认行为相同。
        """
        self.alpha = alpha
        self.base_collate = base_collate
    def __call__(self, batch):
        #batch 是一个列表，先将其转换为真正的数据 batch
        x, y = self.base_collate(batch)
        #采样要使用的 lambda 值。变量名末尾加 "_"，
        #因为 lambda 是 Python 关键字
        lambda_ = np.random.beta(self.alpha, self.alpha)
        #创建随机打乱顺序 pi
        B = x.size(0)
        shuffled_order = torch.randperm(B)

        #计算输入数据的混合版本
        x_tilde = lambda_ * x + (1 - lambda_) * x[shuffled_order, :]
        #获取标签
        y_i, y_j = y, y[shuffled_order]
        #返回二元组。第一个是输入数据，第二个又是
        #包含 3 项的元组，供 MixupLoss 使用
        return x_tilde, (y_i, y_j, lambda_)

# ====== 单元 27 (代码) ======
#用新的、使用 MixupCollator 的 data loader 替换原 loader
train_loader_mixup = torch.utils.data.DataLoader(trainset, batch_size=B, num_workers=2, shuffle=True, collate_fn=MixupCollator())

resnetReZero_cifar10.apply(weight_reset) #偷懒，直接重置权重
#优化器和调度器保持不变
optimizer = torch.optim.AdamW(resnetReZero_cifar10.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
#由于使用 Mixup 训练，需要用新的 `MixupLoss` 包裹原始 `loss`
resnetReZero_mixup_results = train_network(resnetReZero_cifar10, MixupLoss(loss), train_loader_mixup, epochs=epochs, device=device, test_loader=test_loader, optimizer=optimizer, lr_schedule=scheduler, score_funcs={'Accuracy': accuracy_score})

# ====== 单元 28 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=normal_results, label='Regular')
sns.lineplot(x='epoch', y='test Accuracy', data=resnet_results, label='ResNet')
sns.lineplot(x='epoch', y='test Accuracy', data=resnetReZero_results, label='ResNet ReZero')
sns.lineplot(x='epoch', y='test Accuracy', data=resnetReZero_mixup_results, label='ResNet ReZero + MixUp')
plt.show()

