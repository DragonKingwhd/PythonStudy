"""Chapter_7 — 来自 Inside-Deep-Learning 仓库，自动转成可运行 .py 脚本。

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

from idlmam import train_network, Flatten, View, weight_reset,  set_seed

# ====== 单元 1 (代码) ======
# [已剥离] %matplotlib inline
# [已剥离] from IPython.display import set_matplotlib_formats
# [已剥离] set_matplotlib_formats('png', 'pdf')

from IPython.display import display_pdf
from IPython.display import Latex

# ====== 单元 2 (代码) ======
torch.backends.cudnn.deterministic=True
print(set_seed(42))

# ====== 单元 3 (代码) ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch.cuda.is_available() else torch.device("cpu")

# ====== 单元 4 (代码) ======
# 输入中有多少个值？用它来帮助确定后续层的大小
D = 28*28 # 28 * 28 的图像
# 隐藏层大小
n = 2
# 输入有多少个通道？
C = 1
# 一共有多少类？
classes = 10

# ====== 单元 5 (代码) ======
class TransposeLinear(nn.Module): # 我们的类继承自 nn.Module，所有 PyTorch 层都必须继承它。
    def __init__(self, linearLayer, bias=True):
        """
        linearLayer: 我们希望用其转置来产生本层输出的那一层。即 Linear 层代表 W，而本层代表 W^T。通过复用 linearLayer 的权重，实现了权重共享。
        bias: 若为 True，将创建一个独立学习的新偏置项 b（与 linearLayer 中的不同）。若为 False，则不使用偏置向量。
        """
        super().__init__()

        # 创建一个新变量 weight 来存放对原始权重的 _引用_。
        self.weight = linearLayer.weight
        if bias:
            # 需要创建一个新的偏置向量。默认情况下，PyTorch 知道如何更新 Module 和 Parameter。由于普通 tensor 既不是 Module 也不是 Parameter，Parameter 类包装了 Tensor 类，让 PyTorch 知道该 tensor 中的值需要通过梯度下降更新
            self.bias = nn.Parameter(torch.Tensor(linearLayer.weight.shape[1]))
        else:
            # Parameter 类不能接受 None 作为输入。所以如果希望存在 bias 但可能不被使用，可以用 register_parameter 来创建它。重要的是无论 Module 的参数如何设置，PyTorch 始终能看到相同的参数。
            self.register_parameter('bias', None)

    # forward 函数是从输入产生输出的代码。
    def forward(self, x):
        # PyTorch 的 F 目录下包含许多供 Module 使用的 _函数_。例如，linear 函数接收输入（这里使用权重的转置）和偏置（若为 `None`，则不做任何处理），执行线性变换。
        return F.linear(x, self.weight.t(), self.bias)
# 说明：此类实现了转置操作 $W^\top$。要转置的矩阵 $W$ 必须作为构造函数的 `linearLayer` 参数传入。这样我们就能在原始 `nn.Linear` 层和该层的转置版本之间共享权重。

# ====== 单元 6 (代码) ======
# 因为我们要共享线性层的权重，所以单独定义它
linearLayer = nn.Linear(D,  n, bias=False)
# encoder 只做 flatten，然后使用该线性层
pca_encoder = nn.Sequential(
    nn.Flatten(),
    linearLayer,
)
# decoder 使用我们的 TransposeLinear 层 + 现在共享的 linearLayer 对象
pca_decoder = nn.Sequential(
    TransposeLinear(linearLayer, bias=False),
    View(-1, 1, 28, 28)# 将数据重塑回原始形状
)
# 定义最终的 PCA 模型，即 encoder 后接 decoder 的串联
pca_model = nn.Sequential(
    pca_encoder,
    pca_decoder
)

# ====== 单元 7 (代码) ======
print(nn.init.orthogonal_(linearLayer.weight))

# ====== 单元 8 (代码) ======
mse_loss = nn.MSELoss() # 原始损失函数

def mseWithOrthoLoss(x, y):# 我们的 PCA 损失函数
    # 从前面保存的 linearLayer 对象中取出 W。
    W = linearLayer.weight
    # 单位矩阵，作为正则项的目标
    I = torch.eye(W.shape[0]).to(device)
    # 计算原始损失 $\ell_{\mathit{MSE}}(f(\boldsymbol{x}), \boldsymbol{x})$
    normal_loss =  mse_loss(x, y)
    # 计算正则化惩罚 $\ell_{\mathit{MSE}}(W^\top W, \boldsymbol{I})$
    regularization_loss = 0.1*mse_loss(torch.mm(W, W.t()), I)
    # 返回两个损失之和
    return normal_loss + regularization_loss

# ====== 单元 9 (代码) ======
class AutoEncodeDataset(Dataset):
    """接收 (x, y) 标签对形式的数据集，将其转换为 (x, x) 对。
    这样便于复用其他代码"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset.__getitem__(idx)
        return x, x# 直接丢弃原始标签

# ====== 单元 10 (代码) ======
train_data = AutoEncodeDataset(torchvision.datasets.MNIST("./", train=True, transform=transforms.ToTensor(), download=True))
test_data_xy = torchvision.datasets.MNIST("./", train=False, transform=transforms.ToTensor(), download=True)
test_data_xx = AutoEncodeDataset(test_data_xy)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data_xx, batch_size=128)

# ====== 单元 11 (代码) ======
print(train_network(pca_model, mseWithOrthoLoss, train_loader, test_loader=test_loader, epochs=10, device=device))

# ====== 单元 12 (代码) ======
def encode_batch(encoder, dataset_to_encode):
    """
    encoder: 接收数据集并将其转换到新维度的 PyTorch 网络
    dataset_to_encode: 我们想要转换的 PyTorch `Dataset` 对象。

    返回一个元组 (projected, labels)，其中 `projected` 是数据集的编码后版本，`labels` 是 `dataset_to_encode` 提供的原始标签
    """
    # 创建保存结果的容器
    projected = []
    labels = []
    # 切换到评估模式
    encoder = encoder.eval()
    # 为简单起见切换到 CPU 模式，其实不必这样做
    encoder = encoder.cpu()
    with torch.no_grad():# 我们不想训练，所以用 torch.no_grad！
        for x, y in  DataLoader(dataset_to_encode, batch_size=128):
            z = encoder(x.cpu()) # 对原始数据编码
            projected.append( z.numpy() ) # 保存编码后的版本和标签
            labels.append( y.cpu().numpy().ravel() )
    # 将结果合并成单个大的 numpy 数组
    projected = np.vstack(projected)
    labels = np.hstack(labels)
    # 返回结果
    return projected, labels
# 现在让我们对数据做投影
projected, labels = encode_batch(pca_encoder, test_data_xy)

# ====== 单元 13 (代码) ======
sns.scatterplot(x=projected[:,0], y=projected[:,1], hue=[str(l) for l in labels], hue_order=[str(i) for i in range(10)], legend="full")
plt.show()

# ====== 单元 14 (代码) ======
def showEncodeDecode(encode_decode, x):
    """
    encode_decode: 同时执行编码与解码两步的 PyTorch Module
    x: 既作为原始输入绘制，又用于编码与解码后绘制
    """
    # 切换到评估模式
    encode_decode = encode_decode.eval()
    # 把所有东西移到 CPU，这样不必关心设备问题，
    # 而且此函数对性能不敏感
    encode_decode = encode_decode.cpu()
    with torch.no_grad():# 只要不训练就始终使用 no_grad
        x_recon = encode_decode(x.cpu())
    # 使用 matplotlib 创建并排图，原始图在左
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(x.numpy()[0,:])
    axarr[1].imshow(x_recon.numpy()[0,0,:])
plt.show()

# ====== 单元 15 (代码) ======
# 展示三个数据点的输入（左）和输出（右）
showEncodeDecode(pca_model, test_data_xy[0][0])
showEncodeDecode(pca_model, test_data_xy[2][0])
print(showEncodeDecode(pca_model, test_data_xy[10][0]))

# ====== 单元 16 (代码) ======
# 首先在 encoder 中加入 Tanh 非线性
pca_nonlinear_encode = nn.Sequential(
    nn.Flatten(),
    nn.Linear(D,  n),
    nn.Tanh(), # 唯一真正的变化：在末尾添加一个非线性操作
)
# decoder 现在有了自己的 Linear 层，看起来更像普通网络
pca_nonlinear_decode = nn.Sequential(
    nn.Linear(n, D),# 为简化起见，不再共享权重
    View(-1, 1, 28, 28)
)
# 将二者组合成编码-解码函数 $f(\cdot)$
pca_nonlinear = nn.Sequential(
    pca_nonlinear_encode,
    pca_nonlinear_decode
)

# ====== 单元 17 (代码) ======
print(train_network(pca_nonlinear, mse_loss, train_loader, test_loader=test_loader, epochs=10, device=device))

# ====== 单元 18 (代码) ======
projected, labels = encode_batch(pca_nonlinear_encode, test_data_xy)
sns.scatterplot(x=projected[:,0], y=projected[:,1], hue=[str(l) for l in labels], hue_order=[str(i) for i in range(10)], legend="full" )
plt.show()

# ====== 单元 19 (代码) ======
showEncodeDecode(pca_nonlinear, test_data_xy[0][0])
showEncodeDecode(pca_nonlinear, test_data_xy[2][0])
print(showEncodeDecode(pca_nonlinear, test_data_xy[10][0]))

# ====== 单元 20 (代码) ======
def getLayer(in_size, out_size):
    """
    in_size: 进入该层的神经元/特征数
    out_size: 该隐藏层应输出的神经元/输出数
    """
    return nn.Sequential( # 把概念上一个隐藏层"块"组织成 Sequential 对象
        nn.Linear(in_size,  out_size),
        nn.BatchNorm1d(out_size),
        nn.ReLU())

# ====== 单元 21 (代码) ======
# 按 2、3、4 整除是众多可用模式之一
auto_encoder = nn.Sequential(
    nn.Flatten(),
    getLayer(D, D//2), # 这些层每一层的输出尺寸都比上一层更小
    getLayer(D//2, D//3),
    getLayer(D//3, D//4),
    nn.Linear(D//4,  n), # 跳跃到目标维度
)

# Decoder 以相反的顺序使用相同的层/尺寸，以保持对称
auto_decoder = nn.Sequential(
    getLayer(n, D//4), # 现在每层的尺寸都在增加，因为我们处于 decoder 中。
    getLayer(D//4, D//3),
    getLayer(D//3, D//2),
    nn.Linear(D//2,  D),
    View(-1, 1, 28, 28) # 重塑形状以匹配原始尺寸
)
# 组合成深度自动编码器
auto_encode_decode = nn.Sequential(
    auto_encoder,
    auto_decoder
)

# ====== 单元 22 (代码) ======
print(train_network(auto_encode_decode, mse_loss, train_loader, test_loader=test_loader, epochs=10, device=device))

# ====== 单元 23 (代码) ======
projected, labels = encode_batch(auto_encoder, test_data_xy)
sns.scatterplot(x=projected[:,0], y=projected[:,1], hue=[str(l) for l in labels], hue_order=[str(i) for i in range(10)], legend="full")
plt.show()

# ====== 单元 24 (代码) ======
showEncodeDecode(auto_encode_decode, test_data_xy[0][0])
showEncodeDecode(auto_encode_decode, test_data_xy[2][0])
showEncodeDecode(auto_encode_decode, test_data_xy[6][0])
print(showEncodeDecode(auto_encode_decode, test_data_xy[23][0]))

# ====== 单元 25 (代码) ======
auto_encoder_big = nn.Sequential(
    nn.Flatten(),
    getLayer(D, D*2),
    getLayer(D*2, D*2),
    getLayer(D*2, D*2),
    nn.Linear(D*2,  D*2),
)

auto_decoder_big = nn.Sequential(
    getLayer(D*2, D*2),
    getLayer(D*2, D*2),
    getLayer(D*2, D*2),
    nn.Linear(D*2,  D),
    View(-1, 1, 28, 28)
)

auto_encode_decode_big = nn.Sequential(
    auto_encoder_big,
    auto_decoder_big
)

# ====== 单元 26 (代码) ======
print(train_network(auto_encode_decode_big, mse_loss, train_loader, test_loader=test_loader, epochs=10, device=device))

# ====== 单元 27 (代码) ======
showEncodeDecode(auto_encode_decode_big, test_data_xy[0][0])
showEncodeDecode(auto_encode_decode_big, test_data_xy[6][0])
print(showEncodeDecode(auto_encode_decode_big, test_data_xy[10][0]))

# ====== 单元 28 (代码) ======
normal = torch.distributions.Normal(0, 0.5)# 第一个参数是均值 $\mu$，第二个是标准差 $\sigma$

# ====== 单元 29 (代码) ======
def addNoise(x, device='cpu'): 
    """
    我们用这个辅助函数为一些数据添加噪声。
    x: 要添加噪声的数据
    device: 输入所在的 CPU 或 GPU。
    """
    return x + normal.sample(sample_shape=torch.Size(x.shape)).to(device) #$\boldsymbol{x} + s$

# ====== 单元 30 (代码) ======
showEncodeDecode(auto_encode_decode_big, addNoise(test_data_xy[6][0]))
print(showEncodeDecode(auto_encode_decode_big, addNoise(test_data_xy[23][0])))

# ====== 单元 31 (代码) ======
showEncodeDecode(auto_encode_decode, addNoise(test_data_xy[6][0]))
print(showEncodeDecode(auto_encode_decode, addNoise(test_data_xy[23][0])))

# ====== 单元 32 (代码) ======
class AdditiveGausNoise(nn.Module):
    def __init__(self):
        super().__init__()
        # 该对象的构造函数中不需要做任何事情。

    def forward(self, x):
        # 每个 PyTorch Module 都有一个 self.training 布尔属性，用于判断处于训练（True）还是评估（False）模式。
        if self.training:
             return addNoise(x, device=device)
        else: # 非训练时，原样返回数据
            return x

# ====== 单元 33 (代码) ======
dnauto_encoder_big = nn.Sequential(
    nn.Flatten(),
    AdditiveGausNoise(), # 唯一的新增！在这里注入噪声希望能有帮助。
    getLayer(D, D*2),
    getLayer(D*2, D*2),
    getLayer(D*2, D*2),
    nn.Linear(D*2,  D*2),
)

dnauto_decoder_big = nn.Sequential(
    getLayer(D*2, D*2),
    getLayer(D*2, D*2),
    getLayer(D*2, D*2),
    nn.Linear(D*2,  D),
    View(-1, 1, 28, 28)
)

dnauto_encode_decode_big = nn.Sequential(
    dnauto_encoder_big,
    dnauto_decoder_big
)
# 现在可以像往常一样训练。
print(train_network(dnauto_encode_decode_big, mse_loss, train_loader, test_loader=test_loader, epochs=10, device=device))

# ====== 单元 34 (代码) ======
showEncodeDecode(dnauto_encode_decode_big, test_data_xy[6][0])
print(showEncodeDecode(dnauto_encode_decode_big, addNoise(test_data_xy[6][0])))

# ====== 单元 35 (代码) ======
showEncodeDecode(dnauto_encode_decode_big, test_data_xy[23][0])
print(showEncodeDecode(dnauto_encode_decode_big, addNoise(test_data_xy[23][0])))

# ====== 单元 36 (代码) ======
dnauto_encoder_dropout = nn.Sequential(
    nn.Flatten(),
    nn.Dropout(p=0.2), # 对于输入，通常只丢弃 5-20% 的值。
    getLayer(D, D*2),
    nn.Dropout(), # 默认情况下 dropout 用 50% 的概率把值置零
    getLayer(D*2, D*2),
    nn.Dropout(),
    getLayer(D*2, D*2),
    nn.Dropout(),
    nn.Linear(D*2,  D*2)
)

dnauto_decoder_dropout = nn.Sequential(
    getLayer(D*2, D*2),
    nn.Dropout(),
    getLayer(D*2, D*2),
    nn.Dropout(),
    getLayer(D*2, D*2),
    nn.Dropout(),
    nn.Linear(D*2,  D),
    View(-1, 1, 28, 28)
)

dnauto_encode_decode_dropout = nn.Sequential(
    dnauto_encoder_big,
    dnauto_decoder_big
)
# 现在可以像往常一样训练。
print(train_network(dnauto_encode_decode_dropout, mse_loss, train_loader, test_loader=test_loader, epochs=10, device=device))

# ====== 单元 37 (代码) ======
showEncodeDecode(dnauto_encode_decode_dropout, test_data_xy[6][0]) # 干净数据
showEncodeDecode(dnauto_encode_decode_dropout, addNoise(test_data_xy[6][0])) # 高斯噪声
print(showEncodeDecode(dnauto_encode_decode_dropout, nn.Dropout()(test_data_xy[6][0])))

# ====== 单元 38 (代码) ======
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import re

all_data = []
resp = urlopen("https://cs.stanford.edu/people/karpathy/char-rnn/shakespear.txt")
shakespear_100k = resp.read()
shakespear_100k = shakespear_100k.decode('utf-8').lower()

# ====== 单元 39 (代码) ======
vocab2indx = {} # 词表 $\Sigma$
for char in shakespear_100k:
    if char not in vocab2indx: # 把每个新字符加入词表
        vocab2indx[char] = len(vocab2indx) # 根据当前词表大小设置索引

# 一些有用的代码，用于从索引反查原始字符。
indx2vocab = {}
# 直接遍历所有键值对，建立反向映射字典。
for k, v in vocab2indx.items():
    indx2vocab[v] = k
print("Vocab Size: ", len(vocab2indx))
print("Total Characters:", len(shakespear_100k))

# ====== 单元 40 (代码) ======
class AutoRegressiveDataset(Dataset):
    """
    通过将一个单独的、很长的源序列拆分成"块"，创建一个自回归数据集。
    """

    def __init__(self, large_string, MAX_CHUNK=500):
        """
        large_string: 原始的长源序列，将从中抽取块
        MAX_CHUNK: 单个块的最大允许长度
        """
        self.doc = large_string
        self.MAX_CHUNK = MAX_CHUNK

    def __len__(self):
        # 样本数等于字符数除以块大小
        return (len(self.doc)-1) // self.MAX_CHUNK

    def __getitem__(self, idx):
        # 计算第 idx 个块的起始位置
        start = idx*self.MAX_CHUNK
        # 获取输入子串
        sub_string = self.doc[start:start+self.MAX_CHUNK]
        # 根据词表将子串转换为整数
        x = [vocab2indx[c] for c in sub_string]

        # 取标签子串，整体向后偏移 1
        sub_string = self.doc[start+1:start+self.MAX_CHUNK+1]
        # 根据词表将标签子串转换为整数
        y = [vocab2indx[c] for c in sub_string]
        # 转换为张量
        return torch.tensor(x, dtype=torch.int64), torch.tensor(y, dtype=torch.int64)
# 说明：从大文本语料创建用于自回归问题的数据集。我们假设语料以一个长字符串形式存在；将多个文件拼接为一个长串也没问题，因为我们的块通常比文档要短。

# ====== 单元 41 (代码) ======
class AutoRegressive(nn.Module):

    def __init__(self, num_embeddings, embd_size, hidden_size, layers=1):
        super(AutoRegressive, self).__init__()
        self.hidden_size = hidden_size
        self.embd = nn.Embedding(num_embeddings, embd_size)
        self.layers = nn.ModuleList([nn.GRUCell(embd_size, hidden_size)] + 
                                     [nn.GRUCell(hidden_size, hidden_size) for i in range(layers-1)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for i in range(layers)])
        
        self.pred_class = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),# (B, *, D)
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size), # (B, *, D)
            nn.Linear(hidden_size, num_embeddings) #(B, *. D) -> B(B, *, VocabSize)
        )
        
    def initHiddenStates(self, B):
        """
        为 RNN 层创建初始的隐藏状态列表。

        B: 隐藏状态对应的 batch 大小。
        """
        return [torch.zeros(B, self.hidden_size, device=device) for _ in range(len(self.layers))]

    def step(self, x_in, h_prevs=None):
        """
        x_in: 当前时间步的输入；如果值需要被嵌入，形状为 (B)；如果已经嵌入过，形状为 (B, D)。

        h_prevs: 一个隐藏状态张量列表，针对网络的每一层，形状均为 (B, self.hidden_size)。这些张量包含 RNN 层当前的隐藏状态，会在本次调用中被更新。
        """
        # 把三个参数都准备成最终形式
        if len(x_in.shape) == 1: # (B)，需要嵌入它
            x_in = self.embd(x_in) # 现在为 (B, D)

        if h_prevs is None:
            h_prevs = self.initHiddenStates(x_in.shape[0])

        # 处理输入
        for l in range(len(self.layers)):
            h_prev = h_prevs[l]
            h = self.norms[l](self.layers[l](x_in, h_prev))

            h_prevs[l] = h
            x_in = h
        # 对 token 进行预测
        return self.pred_class(x_in)

    def forward(self, input):
        # 输入应为 (B, T)
        # batch 大小是多少？
        B = input.size(0)
        # 最大时间步数是多少？
        T = input.size(1)

        x = self.embd(input) #(B, T, D)

        # 初始隐藏状态
        h_prevs = self.initHiddenStates(B)

        last_activations = []
        for t in range(T):
            x_in = x[:,t,:] #(B, D)
            last_activations.append(self.step(x_in, h_prevs))

        last_activations = torch.stack(last_activations, dim=1) #(B, T, D)
        
        return last_activations

# ====== 单元 42 (代码) ======
autoRegData = AutoRegressiveDataset(shakespear_100k, MAX_CHUNK=250)
autoReg_loader = DataLoader(autoRegData, batch_size=128, shuffle=True)

autoReg_model = AutoRegressive(len(vocab2indx), 32, 128, layers=2)
autoReg_model = autoReg_model.to(device)

for p in autoReg_model.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -2, 2))

# ====== 单元 43 (代码) ======
def CrossEntLossTime(x, y):
    """
    x: 形状为 (B, T, V) 的输出
    y: 形状为 (B, T) 的标签

    """
    cel = nn.CrossEntropyLoss()

    T = x.size(1)

    loss = 0

    for t in range(T):# 遍历序列中的每一项
        loss += cel(x[:,t,:], y[:,t]) # 累加预测误差

    return loss

# ====== 单元 44 (代码) ======
print(train_network(autoReg_model, CrossEntLossTime, autoReg_loader, epochs=100, device=device))

# ====== 单元 45 (代码) ======
autoReg_model = autoReg_model.eval()
sampling = torch.zeros((1, 500), dtype=torch.int64, device=device)

# ====== 单元 46 (代码) ======
seed = "EMILIA:".lower()
cur_len = len(seed)
sampling[0,0:cur_len] = torch.tensor([vocab2indx[x] for x in seed])

# ====== 单元 47 (代码) ======
for i in tqdm(range(cur_len, sampling.size(1))):
    with torch.no_grad():
        h = autoReg_model(sampling[:,0:i]) # 处理之前所有的字符
        h = h[:,-1,:] # 取最后一个时间步
        h = F.softmax(h, dim=1) # 转换为概率
        next_tokens = torch.multinomial(h, 1) # 采样下一个预测
        sampling[:,i] = next_tokens # 设置下一个预测
        # 长度加一
        cur_len += 1

# ====== 单元 48 (代码) ======
s = [indx2vocab[x] for x in sampling.cpu().numpy().flatten()]
print("".join(s))

# ====== 单元 49 (代码) ======
cur_len = len(seed)
temperature = 0.75 # 主要变化，控制温度从而影响采样行为
for i in tqdm(range(cur_len, sampling.size(1))):
    with torch.no_grad():
        h = autoReg_model(sampling[:,0:i])
        h = h[:,-1,:] # 取最后一个时间步
        h = F.softmax(h/temperature, dim=1) # 转换为概率
        next_tokens = torch.multinomial(h, 1)
        sampling[:,i] = next_tokens

        cur_len += 1

# ====== 单元 50 (代码) ======
s = [indx2vocab[x] for x in sampling.cpu().numpy().flatten()]
print("".join(s))

# ====== 单元 51 (代码) ======
cur_len = len(seed)
temperature = 0.05 # 非常低的温度，几乎总是选择最可能的项
for i in tqdm(range(cur_len, sampling.size(1))):
    with torch.no_grad():
        h = autoReg_model(sampling[:,0:i])
        h = h[:,-1,:] # 取最后一个时间步
        h = F.softmax(h/temperature, dim=1) # 转换为概率
        next_tokens = torch.multinomial(h, 1)
        sampling[:,i] = next_tokens

        cur_len += 1
s = [indx2vocab[x] for x in sampling.cpu().numpy().flatten()]
print("".join(s))

# ====== 单元 52 (代码) ======
# 设置种子串以及存放生成内容的位置
seed = "EMILIA:".lower()
cur_len = len(seed)
sampling = torch.zeros((1, 500), dtype=torch.int64, device=device)
sampling[0,0:cur_len] = torch.tensor([vocab2indx[x] for x in seed])

# 选取一个温度
temperature = 0.75
with torch.no_grad():
    # 初始化隐藏状态以避免重复计算
    h_prevs = autoReg_model.initHiddenStates(1)
    # 将种子串依次喂入模型
    for i in range(0, cur_len):
        h = autoReg_model.step(sampling[:,i], h_prevs=h_prevs)

    # 一次生成一个字符的新文本
    for i in tqdm(range(cur_len, sampling.size(1))):
        h = F.softmax(h/temperature, dim=1) # 转换为概率
        next_tokens = torch.multinomial(h, 1)
        sampling[:,i] = next_tokens
        cur_len += 1
        # 现在只把新采样的字符喂入模型
        h = autoReg_model.step(sampling[:,i], h_prevs=h_prevs)

# ====== 单元 53 (代码) ======
s = [indx2vocab[x] for x in sampling.cpu().numpy().flatten()]
print("".join(s))

