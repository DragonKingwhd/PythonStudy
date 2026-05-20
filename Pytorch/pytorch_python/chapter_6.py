"""Chapter_6 — 来自 Inside-Deep-Learning 仓库，自动转成可运行 .py 脚本。

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

from idlmam import train_network, Flatten, weight_reset, set_seed
from idlmam import LanguageNameDataset, pad_and_pack, EmbeddingPackable, LastTimeStep, LambdaLayer

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
train_data = torchvision.datasets.FashionMNIST("./", train=True, transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.FashionMNIST("./", train=True, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

# ====== 单元 5 (代码) ======
# 图像的宽度和高度是多少？
W, H = 28, 28 #
# 输入中有多少个值？用它来帮助确定后续层的大小
D = 28*28 # 28 * 28 的图像
# 隐藏层大小
n = 256
# 输入有多少个通道？
C = 1
# 每个卷积层有多少个 filter
n_filters = 32
# 一共有多少类？
classes = 10#

# ====== 单元 6 (代码) ======
fc_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(D,  n), nn.Tanh(), # 第一个隐藏层
    *[nn.Sequential(nn.Linear(n,  n),nn.Tanh()) for _ in range(5)], # 由于剩余每一层的输入/输出尺寸都相同，可以用列表解包一次性创建
    nn.Linear(n, classes),
)

# ====== 单元 7 (代码) ======
cnn_model = nn.Sequential(
    nn.Conv2d(C, n_filters, 3, padding=1),             nn.Tanh(),
    nn.Conv2d(n_filters, n_filters, 3, padding=1),     nn.Tanh(),
    nn.Conv2d(n_filters, n_filters, 3, padding=1),     nn.Tanh(),
    nn.MaxPool2d((2,2)),
    nn.Conv2d(  n_filters, 2*n_filters, 3, padding=1), nn.Tanh(),
    nn.Conv2d(2*n_filters, 2*n_filters, 3, padding=1), nn.Tanh(),
    nn.Conv2d(2*n_filters, 2*n_filters, 3, padding=1), nn.Tanh(),
    nn.MaxPool2d((2,2)),
    nn.Conv2d(2*n_filters, 4*n_filters, 3, padding=1), nn.Tanh(),
    nn.Conv2d(4*n_filters, 4*n_filters, 3, padding=1), nn.Tanh(),
    nn.Flatten(),
    nn.Linear(D*n_filters//4, classes),
)

# ====== 单元 8 (代码) ======
loss_func = nn.CrossEntropyLoss()
fc_results = train_network(fc_model, loss_func, train_loader, test_loader=test_loader, epochs=10, score_funcs={'Accuracy': accuracy_score}, device=device)
cnn_results = train_network(cnn_model, loss_func, train_loader, test_loader=test_loader, epochs=10, score_funcs={'Accuracy': accuracy_score}, device=device)

# ====== 单元 9 (代码) ======
del fc_model
del cnn_model

# ====== 单元 10 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results, label='Fully Connected')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_results, label='CNN')
plt.show()

# ====== 单元 11 (代码) ======
def sigmoid(x):
    return np.exp(activation_input)/(np.exp(activation_input)+1)

activation_input = np.linspace(-5, 5, num=200)
tanh_activation = np.tanh(activation_input)
sigmoid_activation = sigmoid(activation_input)

sns.lineplot(x=activation_input, y=tanh_activation, color='red', label="tanh(x)")
sns.lineplot(x=activation_input, y=sigmoid_activation, color='blue', label="$\sigma(x)$")
plt.show()

# ====== 单元 12 (代码) ======
def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

tanh_deriv = tanh_deriv(activation_input)
sigmoid_deriv = sigmoid_derivative(activation_input)

sns.lineplot(x=activation_input, y=tanh_deriv, color='red', label="tanh'(x)")
sns.lineplot(x=activation_input, y=sigmoid_deriv, color='blue', label="$\sigma'(x)$")
plt.show()

# ====== 单元 13 (代码) ======
activation_input = np.linspace(-5, 5, num=200)
relu_activation = np.maximum(0,activation_input)
leaky_relu_activation = np.maximum(0.3*activation_input,activation_input)

sns.lineplot(x=activation_input, y=tanh_activation, color='red', label="tanh(x)")
sns.lineplot(x=activation_input, y=sigmoid_activation, color='blue', label="$\sigma(x)$")
sns.lineplot(x=activation_input, y=relu_activation, color='green', label="ReLU(x)")
sns.lineplot(x=activation_input, y=leaky_relu_activation, color='purple', label="LeakyReLU(x)")
plt.show()

# ====== 单元 14 (代码) ======
relu_deriv = 1.0*(activation_input > 0)
leaky_deriv = 1.0*(activation_input > 0) + 0.3*(activation_input <= 0)

sns.lineplot(x=activation_input, y=tanh_deriv, color='red', label="tanh'(x)")
sns.lineplot(x=activation_input, y=sigmoid_deriv, color='blue', label="$\sigma'(x)$")
sns.lineplot(x=activation_input, y=relu_deriv, color='green', label="ReLU'(x)")
sns.lineplot(x=activation_input, y=leaky_deriv, color='purple', label="LeakyReLU'(x)")
plt.show()

# ====== 单元 15 (代码) ======
leak_rate = 0.1 # LeakyReLU 的"泄漏"系数。设在 [0.01, 0.3] 区间内都可以。

# ====== 单元 16 (代码) ======
fc_relu_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(D,  n), nn.LeakyReLU(leak_rate),
    *[nn.Sequential(nn.Linear(n,  n), nn.LeakyReLU(leak_rate)) for _ in range(5)], 
    nn.Linear(n, classes),
)

# ====== 单元 17 (代码) ======
def cnnLayer(in_filters, out_filters=None, kernel_size=3):
    """
    in_filters: 进入该层的通道数
    out_filters: 该层要学习/输出的通道数；若为 `None`，表示与输入通道数相同。
    kernel_size: 卷积核大小
    """
    if out_filters is None:
        out_filters = in_filters # 这是一种常见模式，没有指定时自动设为默认
    padding=kernel_size//2 # padding 用于保持尺寸不变
    return nn.Sequential( # 把卷积层和激活函数组合成一个单元
        nn.Conv2d(in_filters, out_filters, kernel_size, padding=padding),
        nn.LeakyReLU(leak_rate)
    )

# ====== 单元 18 (代码) ======
cnn_relu_model = nn.Sequential(
    cnnLayer(C, n_filters), cnnLayer(n_filters), cnnLayer(n_filters),
    nn.MaxPool2d((2,2)),
    cnnLayer(n_filters, 2*n_filters), cnnLayer(2*n_filters), cnnLayer(2*n_filters), 
    nn.MaxPool2d((2,2)),
    cnnLayer(2*n_filters, 4*n_filters), cnnLayer(4*n_filters),
    nn.Flatten(),
    nn.Linear(D*n_filters//4, classes),
)
# 说明：这是我们通用的 CNN 代码块。先不管对象名，通过修改 cnnLayer 函数的定义，这段代码可以被复用于多种不同风格的 CNN 隐藏层。

# ====== 单元 19 (代码) ======
fc_relu_results = train_network(fc_relu_model, loss_func, train_loader, test_loader=test_loader, epochs=10, score_funcs={'Accuracy': accuracy_score}, device=device)
del fc_relu_model
cnn_relu_results = train_network(cnn_relu_model, loss_func, train_loader, test_loader=test_loader, epochs=10, score_funcs={'Accuracy': accuracy_score}, device=device)
del cnn_relu_model

# ====== 单元 20 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results, label='FC')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_relu_results, label='FC-ReLU')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_results, label='CNN')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_relu_results, label='CNN-ReLU')
plt.show()

# ====== 单元 21 (代码) ======
fc_bn_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(D,  n), nn.BatchNorm1d(n), nn.LeakyReLU(leak_rate),
    *[nn.Sequential(nn.Linear(n,  n), nn.BatchNorm1d(n), nn.LeakyReLU(leak_rate)) for _ in range(5)], 
    nn.Linear(n, classes),
)

# ====== 单元 22 (代码) ======
def cnnLayer(in_filters, out_filters=None, kernel_size=3):
    if out_filters is None:
        out_filters = in_filters # 这是一种常见模式，没有指定时自动设为默认
    padding=kernel_size//2 # padding 用于保持尺寸不变
    return nn.Sequential( # 把卷积层和激活函数组合成一个单元
        nn.Conv2d(in_filters, out_filters, kernel_size, padding=padding),
        nn.BatchNorm2d(out_filters), # 唯一变化：在卷积后加入 BatchNorm2d！
        nn.LeakyReLU(leak_rate)
    )

# ====== 单元 23 (代码) ======
cnn_bn_model = nn.Sequential(
    cnnLayer(C, n_filters), cnnLayer(n_filters), cnnLayer(n_filters),
    nn.MaxPool2d((2,2)),
    cnnLayer(n_filters, 2*n_filters), cnnLayer(2*n_filters), cnnLayer(2*n_filters), 
    nn.MaxPool2d((2,2)),
    cnnLayer(2*n_filters, 4*n_filters), cnnLayer(4*n_filters),
    nn.Flatten(),
    nn.Linear(D*n_filters//4, classes),
)

# ====== 单元 24 (代码) ======
fc_bn_results = train_network(fc_bn_model, loss_func, train_loader, test_loader=test_loader, epochs=10, score_funcs={'Accuracy': accuracy_score}, device=device)
del fc_bn_model
cnn_bn_results = train_network(cnn_bn_model, loss_func, train_loader, test_loader=test_loader, epochs=10, score_funcs={'Accuracy': accuracy_score}, device=device)
del cnn_bn_model

# ====== 单元 25 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=fc_relu_results, label='FC-ReLU')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_bn_results, label='FC-ReLU-BN')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_relu_results, label='CNN-ReLU')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_bn_results, label='CNN-ReLU-BN')
plt.show()

# ====== 单元 26 (代码) ======
fc_ln_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(D,  n), nn.LayerNorm([n]), nn.LeakyReLU(leak_rate),
    *[nn.Sequential(nn.Linear(n,  n), nn.LayerNorm([n]), nn.LeakyReLU(leak_rate)) for _ in range(5)], 
    nn.Linear(n, classes),
)

# ====== 单元 27 (代码) ======
def cnnLayer(in_filters, out_filters=None, pool_factor=0,kernel_size=3):
    if out_filters is None:
        out_filters = in_filters # 这是一种常见模式，没有指定时自动设为默认
    padding=kernel_size//2 # padding 用于保持尺寸不变
    return nn.Sequential( # 把卷积层和激活函数组合成一个单元
        nn.Conv2d(in_filters, out_filters, kernel_size, padding=padding),
        nn.LayerNorm([out_filters, W//(2**pool_factor), H//(2**pool_factor)]), # 唯一变化：在卷积后加入 LayerNorm！
        nn.LeakyReLU(leak_rate)
    )

# ====== 单元 28 (代码) ======
cnn_ln_model = nn.Sequential(
    cnnLayer(C, n_filters), 
    cnnLayer(n_filters), 
    cnnLayer(n_filters),
    nn.MaxPool2d((2,2)), # 已经做了一轮池化，所以现在 pool_factor=1
    cnnLayer(n_filters, 2*n_filters, pool_factor=1),
    cnnLayer(2*n_filters, pool_factor=1),
    cnnLayer(2*n_filters, pool_factor=1),
    nn.MaxPool2d((2,2)), # 现在做了两轮池化，所以 pool_factor=2
    cnnLayer(2*n_filters, 4*n_filters, pool_factor=2), 
    cnnLayer(4*n_filters, pool_factor=2),
    nn.Flatten(),
    nn.Linear(D*n_filters//4, classes),
)

# ====== 单元 29 (代码) ======
fc_ln_results = train_network(fc_ln_model, loss_func, train_loader, test_loader=test_loader, epochs=10, score_funcs={'Accuracy': accuracy_score}, device=device)
del fc_ln_model
cnn_ln_results = train_network(cnn_ln_model, loss_func, train_loader, test_loader=test_loader, epochs=10, score_funcs={'Accuracy': accuracy_score}, device=device)
del cnn_ln_model

# ====== 单元 30 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=fc_relu_results, label='FC-ReLU')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_bn_results, label='FC-ReLU-BN')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_relu_results, label='CNN-ReLU')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_bn_results, label='CNN-ReLU-BN')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_ln_results, label='FC-ReLU-LN')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_ln_results, label='CNN-ReLU-LN')
plt.show()

# ====== 单元 31 (代码) ======
class SkipFC(nn.Module):
    def __init__(self, n_layers, in_size, out_size, leak_rate=0.1):
        """
        n_layers: 该 dense 跳跃连接块包含多少个隐藏层
        in_size: 进入该层的特征数
        out_size: 该块最后一层应使用的特征数
        leak_rate: LeakyReLU 激活函数的参数
        """
        super().__init__()

        # 最后一层会被特殊处理，所以先获取它的索引以便后两行使用
        l = n_layers-1
        # 线性层和 BN 层分别存放在 `layers` 和 `bns` 中。列表推导式一行创建所有层。`if i == l` 用于单独处理最后一层，它需要用 `out_size` 而不是 `in_size`
        self.layers = nn.ModuleList([nn.Linear(in_size*l, out_size) if i == l else nn.Linear(in_size, in_size) for i in range(n_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(out_size) if i == l else nn.BatchNorm1d(in_size) for i in range(n_layers)])
        # 因为我们自己写 `forward` 函数而不是使用 nn.Sequential，所以可以多次复用同一个激活对象
        self.activation = nn.LeakyReLU(leak_rate)

    def forward(self, x):
        # 首先需要一个位置来存放该块内每一层（除最后一层）的激活值。所有激活值会被合并作为最后一层的输入，这正是跳跃连接的实现方式！
        activations = []

        # 将线性层和归一化层 zip 成成对的元组，用 [:-1] 选取除最后一项以外的所有项
        for layer, bn in zip(self.layers[:-1], self.bns[:-1]):
            x = self.activation(bn(layer(x)))
            activations.append( x )
        # 将所有激活值拼接起来，作为最后一层的输入
        x = torch.cat(activations, dim=1)
        # 现在手动对这个拼接后的输入应用最后一个线性层和 BN 层，得到结果
        return self.activation(self.bns[-1](self.layers[-1](x)))
# 说明：定义一个 PyTorch 模块类，用于创建跳跃连接。它会创建一个由多层组成的较大块，以"dense"风格的跳跃连接组织，共含 `n_layers` 层。单独使用可以创建 dense 网络，串联使用可以创建错落的跳跃连接。

# ====== 单元 32 (代码) ======
fc_skip_model = nn.Sequential(
    nn.Flatten(),
    SkipFC(2, D, n),
    SkipFC(2, n, n),
    SkipFC(2, n, n),
    nn.Linear(n, classes),
)

fc_skip_results = train_network(fc_skip_model, loss_func, train_loader, test_loader=test_loader, epochs=10, score_funcs={'Accuracy': accuracy_score}, device=device)
del fc_skip_model

# ====== 单元 33 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=fc_relu_results, label='FC-ReLU')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_bn_results, label='FC-ReLU-BN')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_skip_results, label='FC-ReLU-BN-Skip')
plt.show()

# ====== 单元 34 (代码) ======
class SkipConv2d(nn.Module):
    def __init__(self, n_layers, in_channels, out_channels, kernel_size=3, leak_rate=0.1):
        super().__init__()
        
        # 最后一个卷积的输入和输出通道数不同，因此仍然需要这个索引
        l = n_layers-1
        # 一些简单的辅助变量
        f = (kernel_size, kernel_size)
        pad = (kernel_size-1)//2

        # 定义所使用的层，并通过同样的 `if i == l` 列表推导式改变最后一层的构造。我们将通过通道维度合并卷积，因此最后一层的输入输出通道数会变化。
        self.layers = nn.ModuleList([nn.Conv2d(in_channels*l, out_channels, kernel_size=f, padding=pad) if i == l else nn.Conv2d(in_channels, in_channels, kernel_size=f, padding=pad) for i in range(n_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) if i == l else nn.BatchNorm2d(in_channels) for i in range(n_layers)])

        self.activation = nn.LeakyReLU(leak_rate)

    def forward(self, x):
        # 这段代码与 SkipFC 类相同，但有必要强调最可能需要改动的那一行。
        activations = []

        for layer, bn in zip(self.layers[:-1], self.bns[:-1]):
            x = self.activation(bn(layer(x)))
            activations.append( x )
        # 也就是这里把所有激活值拼接起来。我们的张量按 (B, C, W, H) 组织，这是 PyTorch 的默认顺序。但你可以更改它，有时人们会使用 (B, W, H, C)。这种情况下通道 C 在索引 3 而不是 1，所以这种场景下需要将 `cat=3`。这也是把此代码适配到 RNN 上的方法
        x = torch.cat(activations, dim=1)

        return self.activation(self.bns[-1](self.layers[-1](x)))

# ====== 单元 35 (代码) ======
cnn_skip_model = nn.Sequential(
    nn.Conv2d(C, n_filters, (3,3), padding=1), 
    SkipConv2d(3, n_filters, 2*n_filters),
    nn.MaxPool2d((2,2)),
    nn.LeakyReLU(),
    SkipConv2d(3, 2*n_filters, 4*n_filters),
    nn.MaxPool2d((2,2)),
    SkipConv2d(2, 4*n_filters, 4*n_filters),
    nn.Flatten(),
    nn.Linear(D*n_filters//4, classes),
)

cnn_skip_results = train_network(cnn_skip_model, loss_func, train_loader, test_loader=test_loader, epochs=10, score_funcs={'Accuracy': accuracy_score}, device=device)
del cnn_skip_model

# ====== 单元 36 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_relu_results, label='CNN-ReLU')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_bn_results, label='CNN-ReLU-BN')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_skip_results, label='CNN-ReLU-BN-Skip')
plt.show()

# ====== 单元 37 (代码) ======
def infoShareBlock(n_filters):
    return nn.Sequential(
        nn.Conv2d(n_filters, n_filters, (1,1), padding=0), 
        nn.BatchNorm2d(n_filters),
        nn.LeakyReLU())

# ====== 单元 38 (代码) ======
def cnnLayer(in_filters, out_filters=None, kernel_size=3):
    if out_filters is None:
        out_filters = in_filters # 这是一种常见模式，没有指定时自动设为默认
    padding=kernel_size//2 # padding 用于保持尺寸不变
    return nn.Sequential( # 把卷积层和激活函数组合成一个单元
        nn.Conv2d(in_filters, out_filters, kernel_size, padding=padding),
        nn.BatchNorm2d(out_filters), # 唯一变化：在卷积后加入 BatchNorm2d！
        nn.LeakyReLU(leak_rate)
    )

# ====== 单元 39 (代码) ======
cnn_1x1_model = nn.Sequential(
    cnnLayer(C, n_filters), 
    cnnLayer(n_filters),
    infoShareBlock(n_filters), # 在 2 个 cnnLayer 后的第一个 info block
    cnnLayer(n_filters),
    nn.MaxPool2d((2,2)),
    cnnLayer(n_filters, 2*n_filters), 
    cnnLayer(2*n_filters),
    infoShareBlock(2*n_filters),
    cnnLayer(2*n_filters), 
    nn.MaxPool2d((2,2)),
    cnnLayer(2*n_filters, 4*n_filters), 
    cnnLayer(4*n_filters),
    infoShareBlock(4*n_filters),
    nn.Flatten(),
    nn.Linear(D*n_filters//4, classes),
)
# 现在训练这个模型
cnn_1x1_results = train_network(cnn_1x1_model, loss_func, train_loader, test_loader=test_loader, epochs=10, score_funcs={'Accuracy': accuracy_score}, device=device)
del cnn_1x1_model

# ====== 单元 40 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_relu_results, label='CNN-ReLU')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_bn_results, label='CNN-ReLU-BN')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_1x1_results, label='CNN-ReLU-BN-1x1')
plt.show()

# ====== 单元 41 (代码) ======
class ResidualBlockE(nn.Module):
    def __init__(self, channels, kernel_size=3, leak_rate=0.1):
        """
        channels: 该层输入/输出的通道数
        kernel_size: 卷积核大小
        leak_rate: LeakyReLU 激活函数的参数
        """
        super().__init__()
        # 为了保持输入形状，卷积层需要多少 padding
        pad = (kernel_size-1)//2

        # 在子网络中定义所用的 conv 和 BN 层，仅 2 个 conv/BN/激活 的隐藏层
        self.F = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(leak_rate),
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(leak_rate),
        )
     
    def forward(self, x):
        return x + self.F(x) # F() 包含长路径上的全部工作，我们只需将其加到输入上

# ====== 单元 42 (代码) ======
class ResidualBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, leak_rate=0.1):
        super().__init__()
        # 为了保持输入形状，卷积层需要多少 padding
        pad = (kernel_size-1)//2
        # bottleneck 应当更小，所以取 output/4 或 input。也可以试着把 max 改成 min，影响不大。
        bottleneck = max(out_channels//4, in_channels)
        # 定义所需的三组 BN 和卷积层。
        # 注意 1x1 卷积使用 padding=0，因为 1x1 不会改变形状！
        self.F = nn.Sequential(
            # 压缩
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(leak_rate),
            nn.Conv2d(in_channels, bottleneck, 1, padding=0),
            # 正常的一层完整卷积
            nn.BatchNorm2d(bottleneck),
            nn.LeakyReLU(leak_rate),
            nn.Conv2d(bottleneck, bottleneck, kernel_size, padding=pad),
            # 重新扩展回去
            nn.BatchNorm2d(bottleneck),
            nn.LeakyReLU(leak_rate),
            nn.Conv2d(bottleneck, out_channels, 1, padding=0)
        )

        # 默认情况下，shortcut 是恒等函数——直接把输入作为输出
        self.shortcut = nn.Identity()
        # 如果需要改变形状，就把 shortcut 变成包含 1x1 卷积和 BN 的小层
        if in_channels != out_channels:
            self.shortcut =  nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, padding=0),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        # shortcut(x) 充当"x"的角色，尽可能少做事以保持张量形状一致。
        return self.shortcut(x) + self.F(x)

# ====== 单元 43 (代码) ======
cnn_res_model = nn.Sequential(
    ResidualBottleNeck(C, n_filters), # 起始用 BottleNeck，因为需要更多通道。常见做法也可以先放一层普通隐藏层再开始残差块。
    nn.LeakyReLU(leak_rate), # 我们在每个残差块后插入一个激活函数。这是可选的。
    ResidualBlockE(n_filters),
    nn.LeakyReLU(leak_rate),
    nn.MaxPool2d((2,2)),
    ResidualBottleNeck(n_filters, 2*n_filters),
    nn.LeakyReLU(leak_rate),
    ResidualBlockE(2*n_filters),
    nn.LeakyReLU(leak_rate),
    nn.MaxPool2d((2,2)),
    ResidualBottleNeck(2*n_filters, 4*n_filters),
    nn.LeakyReLU(leak_rate),
    ResidualBlockE(4*n_filters),
    nn.LeakyReLU(leak_rate),
    nn.Flatten(),
    nn.Linear(D*n_filters//4, classes),
)

# ====== 单元 44 (代码) ======
cnn_res_results = train_network(cnn_res_model, loss_func, train_loader, test_loader=test_loader, epochs=10, score_funcs={'Accuracy': accuracy_score}, device=device)

# ====== 单元 45 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_results, label='CNN')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_relu_results, label='CNN-ReLU')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_bn_results, label='CNN-ReLU-BN')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_res_results, label='CNN-ReLU-BN-Res')
plt.show()

# ====== 单元 46 (代码) ======
zip_file_url = "https://download.pytorch.org/tutorial/data.zip"

import requests, zipfile, io
r = requests.get(zip_file_url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

# 压缩包结构为 data/names/[LANG].txt，其中 [LANG] 表示具体的语言

namge_language_data = {}

# 使用一些代码移除 UNICODE 字符以方便处理
# 例如把 "Ślusàrski" 转换为 Slusarski
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
alphabet = {}
for i in range(n_letters):
    alphabet[all_letters[i]] = i
    
# 将 Unicode 字符串转为纯 ASCII，参考 https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# 遍历每种语言，打开 zip 中对应的文件项，并读取文本文件里的所有行
for zip_path in z.namelist():
    if "data/names/" in zip_path and zip_path.endswith(".txt"):
        lang = zip_path[len("data/names/"):-len(".txt")]
        with z.open(zip_path) as myfile:
            lang_names = [unicodeToAscii(line).lower() for line in str(myfile.read(), encoding='utf-8').strip().split("\n")]
            namge_language_data[lang] = lang_names

# ====== 单元 47 (代码) ======
dataset = LanguageNameDataset(namge_language_data, alphabet)# 复用第 4 章的代码

train_lang_data, test_lang_data = torch.utils.data.random_split(dataset, (len(dataset)-300, 300))
train_lang_loader = DataLoader(train_lang_data, batch_size=32, shuffle=True, collate_fn=pad_and_pack)
test_lang_loader = DataLoader(test_lang_data, batch_size=32, shuffle=False, collate_fn=pad_and_pack)

# ====== 单元 48 (代码) ======
print(set_seed(42))

# ====== 单元 49 (代码) ======
rnn_3layer = nn.Sequential( # 简单的传统 RNN
  EmbeddingPackable(nn.Embedding(len(all_letters), 64)), #(B, T) -> (B, T, D)
  nn.RNN(64, n, num_layers=3, batch_first=True), #(B, T, D) -> ( (B,T,D) , (S, B, D)  )
  LastTimeStep(rnn_layers=3), # 将 RNN 输出归约为单个项，(B, D)
  nn.Linear(n, len(namge_language_data)), #(B, D) -> (B, classes)
)

# 应用梯度裁剪以提升性能
for p in rnn_3layer.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -5, 5))

rnn_results = train_network(rnn_3layer, loss_func, train_lang_loader, test_loader=test_lang_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=10)

# ====== 单元 50 (代码) ======
lstm_3layer = nn.Sequential(
  EmbeddingPackable(nn.Embedding(len(all_letters), 64)), #(B, T) -> (B, T, D)
  # nn.RNN 改为 nn.LSTM，现在升级为带 peephole 连接的 LSTM
  nn.LSTM(64, n, num_layers=3, batch_first=True), #(B, T, D) -> ( (B,T,D) , (S, B, D)  )
  LastTimeStep(rnn_layers=3), # 将 RNN 输出归约为单个项，(B, D)
  nn.Linear(n, len(namge_language_data)), #(B, D) -> (B, classes)
)
# 对各种 RNN（包括 LSTM）我们仍然希望使用梯度裁剪
for p in lstm_3layer.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -5, 5))

lstm_results = train_network(lstm_3layer, loss_func, train_lang_loader, test_loader=test_lang_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=10)

# ====== 单元 51 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=rnn_results, label='RNN: 3-Layer BiDir')
sns.lineplot(x='epoch', y='test Accuracy', data=lstm_results, label='LSTM: 3-Layer BiDir')
plt.show()

