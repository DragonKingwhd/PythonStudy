"""Chapter_10 — 来自 Inside-Deep-Learning 仓库，自动转成可运行 .py 脚本。

原始 notebook 位于 ../Inside-Deep-Learning/。
"""

# ====== 单元 0 (代码) ======
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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
mnist_train = torchvision.datasets.MNIST("./", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = torchvision.datasets.MNIST("./", train=False, transform=transforms.ToTensor(), download=True)

# ====== 单元 5 (代码) ======
class LargestDigit(Dataset):
    """
    创建数据集的一种修改版本：每次采样若干个样本，
    真实标签是采到样本中最大的那个标签。用于 MNIST 时，
    标签就对应数字本身（例如数字 "6" 的标签为 6）。
    """

    def __init__(self, dataset, toSample=3):
        """
        dataset: 用于采样的源数据集
        toSample: 每次从数据集中采样的样本数
        """
        self.dataset = dataset
        self.toSample = toSample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 从数据集中随机选取 n=self.toSample 个样本
        selected = np.random.randint(0,len(self.dataset), size=self.toSample)

        # 将 n 个形状为 (B, *) 的样本堆叠为 (B, n, *)
        x_new = torch.stack([self.dataset[i][0] for i in selected])
        # 标签为最大标签
        y_new = max([self.dataset[i][1] for i in selected])
        # 返回 (data, label) 对！
        return x_new, y_new

# ====== 单元 6 (代码) ======
B = 128
epochs = 10

largest_train = LargestDigit(mnist_train)
largest_test = LargestDigit(mnist_test)

train_loader = DataLoader(largest_train, batch_size=B, shuffle=True)
test_loader = DataLoader(largest_test, batch_size=B)

# ====== 单元 7 (代码) ======
# 希望数据集划分结果一致
print(set_seed(34))

# ====== 单元 8 (代码) ======
x, y = largest_train[0]

f, axarr = plt.subplots(1,3, figsize=(10,10))
for i in range(3):
    axarr[i].imshow(x[i,0,:].numpy(), cmap='gray', vmin=0, vmax=1)
print("True Label is = ", y)
plt.show()

# ====== 单元 9 (代码) ======
neurons = 256
classes = 10
simpleNet = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784*3,neurons), # 784*3 是因为每张图像有 784 个像素，而每个 bag 中有 3 张图
    nn.LeakyReLU(),
    nn.BatchNorm1d(neurons),
    nn.Linear(neurons,neurons),
    nn.LeakyReLU(),
    nn.BatchNorm1d(neurons),
    nn.Linear(neurons,neurons),
    nn.LeakyReLU(),
    nn.BatchNorm1d(neurons),
    nn.Linear(neurons, classes )
)    
simple_results = train_network(simpleNet, nn.CrossEntropyLoss(), train_loader, val_loader=test_loader, epochs=epochs, score_funcs={'Accuracy': accuracy_score}, device=device)

# ====== 单元 10 (代码) ======
sns.lineplot(x='epoch', y='val Accuracy', data=simple_results, label='Regular')
plt.show()

# ====== 单元 11 (代码) ======
class Flatten2(nn.Module):
    """
    接收形状为 (A, B, C, D, E, ...) 的张量，
    将除前两维以外的所有维度展平，
    得到形状为 (A, B, C*D*E*...) 的结果。
    """
    def forward(self, input):
        return input.view(input.size(0), input.size(1), -1)

# ====== 单元 12 (代码) ======
class Combiner(nn.Module):
    """
    该类用于将特征提取网络 F 与重要性预测网络 W 组合起来，
    通过加权求和的方式合并它们的输出。
    """

    def __init__(self, featureExtraction, weightSelection):
        """
        featureExtraction: 接收形状为 (B, T, D) 的输入，
            输出形状为 (B, T, D') 的新表示。
        weightSelection: 接收形状为 (B, T, D') 的输入，
            输出形状为 (B, T, 1) 或 (B, T) 的张量。需要归一化，
            使最后 T 个值的和为 1（torch.sum(_, dim=1) = 1.0）。
        """
        super(Combiner, self).__init__()
        self.featureExtraction = featureExtraction
        self.weightSelection = weightSelection

    def forward(self, input):
        """
        input: 形状为 (B, T, D) 的张量
        return: 形状为 (B, D') 的新张量
        """
        features = self.featureExtraction(input) #(B, T, D) $\boldsymbol{h}_i = F(\boldsymbol{x}_i)$
        weights = self.weightSelection(features) # 形状为 (B, T) 或 (B, T, 1)，对应 $\boldsymbol{\alpha}$
        if len(weights.shape) == 2: # (B, T) 形状
            weights.unsqueese(2) # 变为 (B, T, 1) 形状

        r = features*weights # (B, T, D)，计算 $\alpha_i \cdot \boldsymbol{h}_i$

        return torch.sum(r, dim=1) # 沿 T 维求和，得到最终形状 (B, D)，即 $\bar{\boldsymbol{x}}$

# ====== 单元 13 (代码) ======
T = 3
D = 784

# ====== 单元 14 (代码) ======
backboneNetwork = nn.Sequential(
    Flatten2(),# 现在形状为 (B, T, D)
    nn.Linear(D,neurons), # 形状变为 (B, T, neurons)
    nn.LeakyReLU(),
    nn.Linear(neurons,neurons),
    nn.LeakyReLU(),
    nn.Linear(neurons,neurons),
    nn.LeakyReLU(), # 出口处仍是 (B, T, neurons)
)

# ====== 单元 15 (代码) ======
attentionMechanism = nn.Sequential(
    # 形状是 (B, T, neurons)
    nn.Linear(neurons,neurons),
    nn.LeakyReLU(),
    nn.Linear(neurons, 1 ), # (B, T, 1)
    nn.Softmax(dim=1),
)

# ====== 单元 16 (代码) ======
simpleAttentionNet = nn.Sequential(
        # 输入为 (B, T, C, W, H)。combiner 会使用 backbone 与 attention 处理
        Combiner(backboneNetwork, attentionMechanism), # 结果为 (B, neurons)
        nn.BatchNorm1d(neurons),
        nn.Linear(neurons,neurons),
        nn.LeakyReLU(),
        nn.BatchNorm1d(neurons),
        nn.Linear(neurons, classes )
    )
simple_attn_results = train_network(simpleAttentionNet, nn.CrossEntropyLoss(), train_loader, val_loader=test_loader, epochs=epochs, score_funcs={'Accuracy': accuracy_score}, device=device)

# ====== 单元 17 (代码) ======
sns.lineplot(x='epoch', y='val Accuracy', data=simple_results, label='Regular')
sns.lineplot(x='epoch', y='val Accuracy', data=simple_attn_results, label='Simple Attention')
plt.show()

# ====== 单元 18 (代码) ======
print(set_seed(1))

# ====== 单元 19 (代码) ======
x, y = largest_train[0] # 选择一个数据点（也就是一个 bag）
x = x.to(device) # 移到计算设备上

with torch.no_grad():
    weights = attentionMechanism(backboneNetwork(x.unsqueeze(0))) # 计算 score(F(x))
    weights = weights.cpu().numpy().ravel() # 转换为 numpy 数组

f, axarr = plt.subplots(1,3, figsize=(10,10))# 为全部 3 个数字绘图
for i in range(3):
    axarr[i].imshow(x[i,0,:].cpu().numpy(), cmap='gray', vmin=0, vmax=1) # 绘制数字
    axarr[i].text(0.0, 0.5, str(round(weights[i],2)), dict(size=40, color='red')) # 在左上方画出注意力分数
    
print("True Label is = ", y)
plt.show()

# ====== 单元 20 (代码) ======
class DotScore(nn.Module):

    def __init__(self, H):
        """
        H: 进入点积打分的维度数。
        """
        super(DotScore, self).__init__()
        self.H = H

    def forward(self, states, context):
        """
        states: (B, T, H) 形状
        context: (B, H) 形状
        output: (B, T, 1)，根据 context 为 T 个 item 各打一个分数
        """
        T = states.size(1)
        # 计算 $\boldsymbol{h}_t^\top \bar{\boldsymbol{h}}$
        scores = torch.bmm(states,context.unsqueeze(2)) / np.sqrt(self.H) #(B, T, H) -> (B, T, 1)
        return scores

# ====== 单元 21 (代码) ======
class GeneralScore(nn.Module):

    def __init__(self, H):
        """
        H: 进入点积打分的维度数。
        """
        super(GeneralScore, self).__init__()
        self.w = nn.Bilinear(H, H, 1) # 存储 $W$

    def forward(self, states, context):
        """
        states: (B, T, H) 形状
        context: (B, H) 形状
        output: (B, T, 1)，根据 context 为 T 个 item 各打一个分数
        """
        T = states.size(1)
        # 将值重复 T 次
        context = torch.stack([context for _ in range(T)], dim=1) #(B, H) -> (B, T, H)
        # 计算 $\boldsymbol{h}_{t}^{\top} W \bar{\boldsymbol{h}}$
        scores = self.w(states, context) #(B, T, H) -> (B, T, 1)
        return scores

# ====== 单元 22 (代码) ======
class AdditiveAttentionScore(nn.Module):

    def __init__(self, H):
        super(AdditiveAttentionScore, self).__init__()
        self.v = nn.Linear(H, 1)
        self.w = nn.Linear(2*H, H)# 2*H 是因为我们要把两个输入拼接起来

    def forward(self, states, context):
        """
        states: (B, T, H) 形状
        context: (B, H) 形状
        output: (B, T, 1)，根据 context 为 T 个 item 各打一个分数
        """
        T = states.size(1)
        # 将值重复 T 次
        context = torch.stack([context for _ in range(T)], dim=1) #(B, H) -> (B, T, H)
        state_context_combined = torch.cat((states, context), dim=2) #(B, T, H) + (B, T, H)  -> (B, T, 2*H)
        scores = self.v(torch.tanh(self.w(state_context_combined))) # (B, T, 2*H) -> (B, T, 1)
        return scores

# ====== 单元 23 (代码) ======
class ApplyAttention(nn.Module):
    """
    该辅助模块用于将注意力机制的结果应用到一组输入上。
    """

    def __init__(self):
        super(ApplyAttention, self).__init__()

    def forward(self, states, attention_scores, mask=None):
        """
        states: (B, T, H) 形状，包含 T 个可能的输入
        attention_scores: (B, T, 1) 每个 item 在当前 context 下的分数
        mask: 如果所有 item 都有效则为 None；否则为形状为 (B, T) 的布尔张量，
            `True` 表示该 item 存在 / 有效。

        返回: 一个包含两个张量的元组。第一个是将注意力应用到 states 后
        得到的最终上下文，形状为 (B, H)。第二个是每个 state 的权重，
        形状为 (B, T, 1)。
        """

        if mask is not None:
            # 将不存在的位置置为一个大负值，以产生消失梯度
            attention_scores[~mask] = -1000.0
        # 计算每个分数对应的权重
        weights = F.softmax(attention_scores, dim=1) # 仍是 (B, T, 1)，但沿 T 求和为 1

        final_context = (states*weights).sum(dim=1) #(B, T, D) * (B, T, 1) -> (B, D)
        return final_context, weights

# ====== 单元 24 (代码) ======
def getMaskByFill(x, time_dimension=1, fill=0):
    """
    x: 形状为 (B, ..., T, ...) 的原始输入，至少 3 维，
        其中可能含有未使用的 item。B 为 batch size，
        T 为时间维度。
    time_dimension: 张量 `x` 中表示时间维度的轴
    fill: 用来表示某个 item 未被使用、应当被屏蔽（mask 中为 `False`）的常数。

    返回: 形状为 (B, T) 的布尔张量，`True` 表示该时刻的值有效可用，
        `False` 表示无效。
    """
    to_sum_over = list(range(1,len(x.shape))) # 跳过第 0 维，因为那是 batch 维度

    if time_dimension in to_sum_over:
        to_sum_over.remove(time_dimension)

    with torch.no_grad():
        # (x!=fill) 找出可能未被使用的位置，因为这些位置缺少
        # 用于标记未使用的 fill 值。
        # 我们沿该时刻对应的所有维度统计非 fill 值的数量
        # （归约后形状变成 (B, T)）。如果至少存在一个非 fill 值，
        # 说明该 item 在使用中，因此返回 True。
        mask = torch.sum((x != fill), dim=to_sum_over) > 0
    return mask

# ====== 单元 25 (代码) ======
with torch.no_grad():
    x = torch.rand((5,3,1,7,7))
    x[0,-1,:] = 0 # 第一条输入的最后一个 item 不用
    x[3,:] = 0 # 第 4 条输入完全不用！
    x[4,0,0,0] = 0 # 让第 5 条 _看起来_ 没用某部分，但其实在用！
    # 加上最后这行是为了证明这套逻辑在棘手输入上也能工作

    mask = getMaskByFill(x)
print(mask)

# ====== 单元 26 (代码) ======
class SmarterAttentionNet(nn.Module):

    def __init__(self, input_size, hidden_size, out_size, score_net=None):
        super(SmarterAttentionNet, self).__init__()
        self.backbone = nn.Sequential(
            Flatten2(),# 现在形状为 (B, T, D)
            nn.Linear(input_size,hidden_size), # 形状变为 (B, T, H)
            nn.LeakyReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.LeakyReLU(),
        )# 返回 (B, T, H)

        # 试着修改这里，看看结果如何变化！
        self.score_net = AdditiveAttentionScore(hidden_size) if (score_net is None) else score_net

        self.apply_attn = ApplyAttention()

        self.prediction_net = nn.Sequential( #(B, H),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size,hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, out_size ) #(B, H)
        )


    def forward(self, input):

        mask = getMaskByFill(input)

        h = self.backbone(input) #(B, T, D) -> (B, T, H)

        #h_context = torch.mean(h, dim=1)
        # 计算 torch.mean，但忽略被 mask 掉的部分
        # 首先把所有有效 item 累加
        h_context = (mask.unsqueeze(-1)*h).sum(dim=1)#(B, T, H) -> (B, H)
        # 然后除以有效 item 的数量，再加上一个小常数以防 bag 完全为空
        h_context = h_context/(mask.sum(dim=1).unsqueeze(-1)+1e-10)

        scores = self.score_net(h, h_context) # (B, T, H) , (B, H) -> (B, T, 1)

        final_context, _ = self.apply_attn(h, scores, mask=mask)

        return self.prediction_net(final_context)

# ====== 单元 27 (代码) ======
attn_dot = SmarterAttentionNet(D, neurons, classes, score_net=DotScore(neurons))
attn_gen = SmarterAttentionNet(D, neurons, classes, score_net=GeneralScore(neurons))
attn_add = SmarterAttentionNet(D, neurons, classes, score_net=AdditiveAttentionScore(neurons))

attn_results_dot = train_network(attn_dot, nn.CrossEntropyLoss(), train_loader, val_loader=test_loader,epochs=epochs, score_funcs={'Accuracy': accuracy_score}, device=device)
attn_results_gen = train_network(attn_gen, nn.CrossEntropyLoss(), train_loader, val_loader=test_loader,epochs=epochs, score_funcs={'Accuracy': accuracy_score}, device=device)
attn_results_add = train_network(attn_add, nn.CrossEntropyLoss(), train_loader, val_loader=test_loader,epochs=epochs, score_funcs={'Accuracy': accuracy_score}, device=device)

# ====== 单元 28 (代码) ======
sns.lineplot(x='epoch', y='val Accuracy', data=simple_results, label='Regular')
sns.lineplot(x='epoch', y='val Accuracy', data=simple_attn_results, label='Simple Attention')
sns.lineplot(x='epoch', y='val Accuracy', data=attn_results_dot, label='Dot')
sns.lineplot(x='epoch', y='val Accuracy', data=attn_results_gen, label='General')
sns.lineplot(x='epoch', y='val Accuracy', data=attn_results_add, label='Additive')
plt.show()

# ====== 单元 29 (代码) ======
class LargestDigitVariable(Dataset):
    """
    创建一种修改版本的数据集，每次采样可变数量的样本，
    真实标签为采样到的最大标签。用于 MNIST 时，标签对应数字本身
    （例如数字 "6" 的标签为 6）。如果没有采到最大数量的样本，
    会用 0 进行填充。
    """

    def __init__(self, dataset, maxToSample=6):
        """
        dataset: 用于采样的源数据集
        toSample: 每次从数据集中采样的样本数
        """
        self.dataset = dataset
        self.maxToSample = maxToSample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # 新增：本次应采样多少个 item？
        how_many = np.random.randint(1,self.maxToSample, size=1)[0]
        # 从数据集中随机选取 n=self.toSample 个样本
        selected = np.random.randint(0,len(self.dataset), size=how_many)

        # 将 n 个形状为 (B, *) 的样本堆叠为 (B, n, *)
        # 新增：用零值填充至最大尺寸
        x_new = torch.stack([self.dataset[i][0] for i in selected] +
                            [torch.zeros((1,28,28)) for i in range(self.maxToSample-how_many)])
        # 标签为最大标签
        y_new = max([self.dataset[i][1] for i in selected])
        # 返回 (data, label) 对
        return x_new, y_new

# ====== 单元 30 (代码) ======
largestV_train = LargestDigitVariable(mnist_train)
largestV_test = LargestDigitVariable(mnist_test)

trainV_loader = DataLoader(largest_train, batch_size=B, shuffle=True)
testV_loader = DataLoader(largest_test, batch_size=B)

# ====== 单元 31 (代码) ======
attn_dot = attn_dot.eval()

preds = []
truths = []
with torch.no_grad():
    for inputs, labels in testV_loader:
        pred = attn_dot(inputs.to(device))
        pred = torch.argmax(pred, dim=1).cpu().numpy()
        
        preds.extend(pred.ravel())
        truths.extend(labels.numpy().ravel())
print("Variable Length Accuracy: ", accuracy_score(preds, truths))

