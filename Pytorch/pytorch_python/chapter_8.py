"""Chapter_8 — 来自 Inside-Deep-Learning 仓库，自动转成可运行 .py 脚本。

原始 notebook 位于 ../Inside-Deep-Learning/。
"""

# ====== 单元 0 (代码) ======
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms


from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm



import os
from imageio import imread
from glob import glob
import json

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import pandas as pd

from sklearn.metrics import accuracy_score

import time

from idlmam import set_seed
from idlmam import train_network, Flatten, View, weight_reset, moveTo

# ====== 单元 1 (代码) ======
# [已剥离] %matplotlib inline
# [已剥离] from IPython.display import set_matplotlib_formats
# [已剥离] set_matplotlib_formats('png', 'pdf')

from IPython.display import display_pdf
from IPython.display import Latex

torch.backends.cudnn.deterministic=True
print(set_seed(42))

# ====== 单元 2 (代码) ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch.cuda.is_available() else torch.device("cpu")

# ====== 单元 3 (代码) ======
# 这个 URL 上有数据集的副本，但请务必去 Kaggle 注册账号并遵守相关许可规定。
data_url_zip = "https://github.com/kamalkraj/DATA-SCIENCE-BOWL-2018/blob/master/data/stage1_train.zip?raw=true"
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import re

# 如果还没下载过数据集就下载它
if not os.path.isdir('./data/stage1_train'):
    resp = urlopen(data_url_zip)
    os.mkdir("./data/stage1_train")
    zipfile = ZipFile(BytesIO(resp.read()))
    zipfile.extractall(path = './data/stage1_train')
# 获取刚解压出来的所有图像路径
paths = glob("./data/stage1_train/*")

# ====== 单元 4 (代码) ======
class DSB2018(Dataset):
    """2018 Data Science Bowl 的 Dataset 类。"""
    def __init__(self, paths):
        """paths: 数据集中每个图像文件夹的路径列表"""
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 每个 images 路径下只有一张图像。所以用末尾的 "[0]" 取出找到的"第一个"
        img_path = glob(self.paths[idx] + "/images/*")[0]
        # 但每个 mask 路径下有多张掩码图像
        mask_imgs = glob(self.paths[idx] + "/masks/*")
        # 图像形状为 (W, H, 4)，最后一维是未使用的 'alpha' 通道
        img = imread(img_path)[:,:,0:3]# 去掉 alpha，得到 (W, H, 3)
        # 现在我们希望变成 (3, W, H)，这是 PyTorch 的标准形状
        img = np.moveaxis(img, -1, 0)
        # 图像的最后一步：缩放到 [0, 1] 区间
        img = img/255.0

        # 每张掩码图像的形状都是 (W, H)，如果像素属于细胞核则值为 1，背景或 _其他_ 细胞核则为 0
        masks = [imread(f)/255.0 for f in mask_imgs]

        # 因为我们想做简单的分割，所以创建一张最终掩码，包含 _所有_ 掩码中 _全部_ 的细胞核像素
        final_mask = np.zeros(masks[0].shape)
        for m in masks:
            final_mask = np.logical_or(final_mask, m)
        final_mask = final_mask.astype(np.float32)

        # 数据集中的图像尺寸不一致。为了简化问题，我们将所有图像 resize 到 (256, 256)
        img, final_mask = torch.tensor(img), torch.tensor(final_mask).unsqueeze(0) # 先转换为 PyTorch 张量
        # interpolate 函数可对一批图像进行 resize，所以我们将每张图像视为大小为 1 的"批"
        img = F.interpolate(img.unsqueeze(0), (256, 256))
        final_mask = F.interpolate(final_mask.unsqueeze(0), (256, 256))
        # 现在形状是 (B=1, C, W, H)，需要转换回 FloatTensor 并取"批"中第一个项。将返回元组：(3, 256, 256), (1, 256, 256)
        return img.type(torch.FloatTensor)[0], final_mask.type(torch.FloatTensor)[0]
# 说明：2018 Data Science Bowl 数据集的类。每张图像都对应一个掩码文件夹，每个目标对应一张掩码。这里我们暂不做这种目标检测，而是让 DataSet 类遍历每张掩码并将它们"或"在一起，得到一张显示所有目标像素的总掩码。该操作在 __getitem__ 内完成，会返回一个元组：输入图像，以及我们要预测的掩码（即包含细胞核的所有像素）。

# ====== 单元 5 (代码) ======
# 创建 Dataset 类对象
dsb_data = DSB2018(paths)

plt.figure(figsize=(16,10))
# 绘制原始图像
plt.subplot(1, 2, 1)
plt.imshow(dsb_data[0][0].permute(1,2,0).numpy())
# 绘制掩码
plt.subplot(1, 2, 2)
plt.imshow(dsb_data[0][1].numpy()[0,:], cmap='gray')
plt.show()

# ====== 单元 6 (代码) ======
# 再绘制一张彩色图像
plt.figure(figsize=(16,10))
plt.subplot(1, 2, 1)
plt.imshow(dsb_data[1][0].permute(1,2,0).numpy())
plt.subplot(1, 2, 2)
plt.imshow(dsb_data[1][1].numpy()[0,:], cmap='gray')
plt.show()

# ====== 单元 7 (代码) ======
train_split, test_split = torch.utils.data.random_split(dsb_data, [500, len(dsb_data)-500])
train_seg_loader = DataLoader(train_split, batch_size=16, shuffle=True)
test_seg_loader = DataLoader(test_split,  batch_size=16)

# ====== 单元 8 (代码) ======
C = 3 # 输入有多少个通道？
n_filters = 32 # 通常应考虑的最小 filter 数。如果想优化架构，可以用 Optuna 选取更好的 filter 数量。
loss_func = nn.BCEWithLogitsLoss()# BCE 损失隐式假设是二分类问题

# ====== 单元 9 (代码) ======
# 定义辅助函数，为 CNN 创建一个隐藏层
def cnnLayer(in_filters, out_filters, kernel_size=3):
    """
    in_filters: 该层输入的通道数
    out_filters: 该层应输出的通道数
    kernel_size: 该层 filter 的大小
    """
    padding = kernel_size//2
    return nn.Sequential(
        nn.Conv2d(in_filters, out_filters, kernel_size, padding=padding),
        nn.BatchNorm2d(out_filters),
        nn.LeakyReLU(), # 为了让代码更短没有设置 leak 值。
    )
# 定义一个用于图像分割的模型
segmentation_model = nn.Sequential(
    cnnLayer(C, n_filters), # 第一层把通道数提升到较大的值
    *[cnnLayer(n_filters, n_filters) for _ in range(5)], # 再创建 5 个隐藏层
    # 为 _每个_ 位置做预测。注意输出只用 1 个通道，因为这是二分类问题，且使用 BCEWithLogitsLoss。
    nn.Conv2d(n_filters, 1, (3,3), padding=1), # 现在形状是 (1, W, H)
)
# 训练分割模型
seg_results = train_network(segmentation_model, loss_func, train_seg_loader, epochs=10, device=device, val_loader=test_seg_loader)

# ====== 单元 10 (代码) ======
index = 6 # 从数据集中挑选一个能展示特定效果的样本。改变这个值可以查看数据集中的其他样本。

with torch.no_grad():# 不训练时不需要梯度，所以禁用梯度！
    # 把一个测试数据点喂入模型。注意原始输出称为 logits
    logits = segmentation_model(test_split[index][0].unsqueeze(0).to(device))[0].cpu()
    # 对 logits 应用 $\sigma$ 得到概率，然后用阈值得到预测掩码
    pred = torch.sigmoid(logits) >= 0.5

# 绘制输入、真实掩码和预测
plt.figure(figsize=(16,10))
plt.subplot(1, 3, 1)
plt.imshow(test_split[index][0].permute(1,2,0).numpy(), cmap='gray') # 首先绘制网络的原始输入
plt.subplot(1, 3, 2)
plt.imshow(test_split[index][1].numpy()[0,:], cmap='gray') # 第二张是 ground truth
plt.subplot(1, 3, 3)
plt.imshow(pred.numpy()[0,:], cmap='gray') # 第三张是网络的预测

plt.annotate('Error: Hole', color="red", fontsize=20, xy=(130, 230),
            xycoords='data', xytext=(-60, 60),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            linewidth = 2.5,
                            color = 'red')
            )

plt.annotate('Error: Hole', color="red", fontsize=20, xy=(210, 75),
            xycoords='data', xytext=(-160, -60),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            linewidth = 2.5,
                            color = 'red')
            )
plt.annotate('Error: Phantom object', color="red", fontsize=20, xy=(247, 15),
            xycoords='data', xytext=(-240, -50),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            linewidth = 2.5,
                            color = 'red')
            )
plt.show()

# ====== 单元 11 (代码) ======
segmentation_model2 = nn.Sequential(
    cnnLayer(C, n_filters), # 第一层把通道数提升到较大的值
    cnnLayer(n_filters, n_filters),
    nn.MaxPool2d(2), # 高和宽各缩小 2 倍
    cnnLayer(n_filters, 2*n_filters),
    cnnLayer(2*n_filters, 2*n_filters),
    cnnLayer(2*n_filters, 2*n_filters),
    # 让高和宽都翻倍，抵消之前那次 MaxPool2d 的效果
    nn.ConvTranspose2d(2*n_filters, n_filters, (3,3), padding=1, output_padding=1, stride=2),
    nn.BatchNorm2d(n_filters),
    nn.LeakyReLU(),
    # 回到普通卷积
    cnnLayer(n_filters, n_filters),
    # 为 _每个_ 位置做预测
    nn.Conv2d(n_filters, 1, (3,3), padding=1), # 现在形状是 (B, 1, W, H)
)

seg_results2 = train_network(segmentation_model2, loss_func, train_seg_loader, epochs=10, device=device, val_loader=test_seg_loader)

# ====== 单元 12 (代码) ======
index = 6 # 与之前相同的样本

with torch.no_grad(): # 不训练时不需要梯度，所以禁用梯度！
    # 把一个测试数据点喂入模型。注意原始输出称为 logits
    pred = segmentation_model2(test_split[index][0].unsqueeze(0).to(device))[0].cpu()
    # 对 logits 应用 $\sigma$ 得到概率，然后用阈值得到预测掩码
    pred = torch.sigmoid(pred) >= 0.5


# 绘制输入、真实掩码和预测
plt.figure(figsize=(16,10))
plt.subplot(1, 3, 1)
plt.imshow(test_split[index][0].permute(1,2,0).numpy(), cmap='gray')  # 首先绘制网络的原始输入
plt.subplot(1, 3, 2)
plt.imshow(test_split[index][1].numpy()[0,:], cmap='gray') # 第二张是 ground truth
plt.subplot(1, 3, 3)
plt.imshow(pred.numpy()[0,:], cmap='gray') # 第三张是网络的预测
plt.show()

# ====== 单元 13 (代码) ======
del segmentation_model
del segmentation_model2

# ====== 单元 14 (代码) ======
sns.lineplot(x='epoch', y='val loss', data=seg_results, label='CNN')
sns.lineplot(x='epoch', y='val loss', data=seg_results2, label='CNN w/ transposed-conv')
plt.show()

# ====== 单元 15 (代码) ======
class UNetBlock2d(nn.Module): # 我们的类继承自 nn.Module，所有 PyTorch 层都必须继承它
    def __init__(self, in_channels, mid_channels, out_channels=None, layers=1, sub_network=None, filter_size=3):
        """
        in_channels: 该块输入的通道数
        mid_channels: 每个卷积 filter 输出的通道数
        out_channels: 若不为 `None`，则在网络末尾添加 1x1 卷积，将输出通道数转换为指定值。
        layers: 在 U-Net 块的输入侧和输出侧各创建多少个隐藏层
        sub_network: 在用 max pooling 将输入缩小 2 倍后应用的子网络。其输出通道数应等于 `mid_channels`
        filter_size: 卷积 filter 的大小
        """
        super().__init__()

        # 开始准备用于处理输入的层
        in_layers = [cnnLayer(in_channels, mid_channels, filter_size)]

        # 如果有子网络，输出端的输入数会翻倍。现在先算好
        if sub_network is None:
            inputs_to_outputs = 1
        else:
            inputs_to_outputs = 2

        # 准备用于得到最终输出的层，其输入会包含来自子网络的额外通道
        out_layers = [ cnnLayer(mid_channels*inputs_to_outputs, mid_channels, filter_size)]

        # 创建用于输入和输出端的其余隐藏层
        for _ in range(layers-1):
            in_layers.append(cnnLayer(mid_channels, mid_channels, filter_size))
            out_layers.append(cnnLayer(mid_channels, mid_channels, filter_size))
        # 使用 1x1 卷积确保特定的输出尺寸
        if out_channels is not None:
            out_layers.append(nn.Conv2d(mid_channels, out_channels, 1, padding=0))

        # 定义共三个子网络：
        # 1) in_model 执行初始几轮卷积
        self.in_model = nn.Sequential(*in_layers)
        # 2) 我们的子网络作用在 max-pool 后的结果上。我们把池化和上采样直接放进 sub-model 里
        if sub_network is not None:
            self.bottleneck = nn.Sequential(
                nn.MaxPool2d(2), # 缩小
                sub_network, # 在低分辨率下处理
                # 重新放大
                nn.ConvTranspose2d(mid_channels, mid_channels, filter_size, padding=filter_size//2, output_padding=1, stride=2)
            )
        else:
            self.bottleneck = None
        # 3) 输出模型处理拼接后的结果；若没有子网络，则只处理 in_model 的输出
        self.out_model = nn.Sequential(*out_layers)


    # forward 函数是从输入产生输出的代码。
    def forward(self, x):
        # 在当前尺度上做卷积
        full_scale_result = self.in_model(x) #(B, C, W, H)
        # 检查是否需要应用 bottleneck
        if self.bottleneck is not None:
            # 形状为 (B, C, W, H)，因为 bottleneck 同时做了池化和扩展
            bottle_result = self.bottleneck(full_scale_result)
            # 现在形状为 (B, 2*C, W, H)
            full_scale_result = torch.cat([full_scale_result, bottle_result], dim=1)
        # 在拼接（或未拼接！）的结果上计算输出
        return self.out_model(full_scale_result)
# 说明：实现 U-Net 方法中"块"的类。每个块需要知道进入和离开它的通道数。块包含三个部分：1) 输入网络，处理进入该块的原始输入；2) bottleneck，将当前结果缩小 2 倍后运行，然后再放大回原始尺寸；3) 输出网络，作用在前两个子网络拼接后的结果上。

# ====== 单元 16 (代码) ======
unet_model = nn.Sequential(
    UNetBlock2d(3, 32, layers=2, sub_network=
        UNetBlock2d(32, 64, out_channels=32, layers=2, sub_network=
            UNetBlock2d(64, 128, out_channels=64, layers=2)
        ),
    ),
    # 为 _每个_ 位置做预测
    nn.Conv2d(32, 1, (3,3), padding=1), # 现在形状是 (B, 1, W, H)
)

unet_results = train_network(unet_model, loss_func, train_seg_loader, epochs=10, device=device, val_loader=test_seg_loader)

# ====== 单元 17 (代码) ======
sns.lineplot(x='epoch', y='val loss', data=seg_results, label='CNN')
sns.lineplot(x='epoch', y='val loss', data=seg_results2, label='CNN w/ transposed-conv')
sns.lineplot(x='epoch', y='val loss', data=unet_results, label='UNet')
plt.show()

# ====== 单元 18 (代码) ======
print(set_seed(42))

# ====== 单元 19 (代码) ======
class Class2Detect(Dataset):
    """这个类用于将分类问题的数据集简单地转换为检测问题的数据集。"""

    def __init__(self, dataset, toSample=3, canvas_size=100):
        """
        dataset: 用作待检测"目标"来源的数据集
        toSample: 单张图像中放入的"目标"最大数量
        canvas_size: 放置目标的图像宽度和高度
        """
        self.dataset = dataset
        self.toSample = toSample
        self.canvas_size = canvas_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        boxes = []
        labels = []

        final_size = self.canvas_size
        # 首先创建一张较大的图像，用于存放所有待检测的"目标"
        img_p = torch.zeros((final_size,final_size), dtype=torch.float32)
        # 现在从原数据集中采样至多 self.toSample 个目标放入图像中
        for _ in range(np.random.randint(1,self.toSample+1)):

            # 从原始数据集中随机选取一个目标及其标签
            img, label = self.dataset[np.random.randint(0,len(self.dataset))]
            # 获取该图像的高度和宽度
            _, img_h, img_w = img.shape
            # 为 x 和 y 轴随机选择偏移量，实际上就是把图像放在随机位置
            offsets = np.random.randint(0,final_size-np.max(img.shape),size=(4))
            # 修改末端的 padding 以确保最终为 100,100 的特定形状
            offsets[1] = final_size - img.shape[1] - offsets[0]
            offsets[3] = final_size - img.shape[2] - offsets[2]

            with torch.no_grad():
                img_p = img_p + F.pad(img, tuple(offsets))
            # 现在创建"boxes"的值
            # 它们均为绝对像素坐标

            # x_min 由随机选择的偏移量决定
            xmin = offsets[0]
            # x_max 等于偏移量加上图像宽度
            xmax = offsets[0]+img_w
            # y 的最小/最大值遵循相同模式
            ymin = offsets[2]
            ymax = offsets[2]+img_h
            # 现在把 box 和对应的标签加入列表
            boxes.append( [xmin, ymin, xmax, ymax] )
            labels.append( label )


        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        return img_p, target
# 说明：此类定义了一个玩具版 MNIST 检测器。从 MNIST 数据集中取出的图像被放置在一张图像的随机位置。目标检测器将学习预测数字所在位置以及该位置上的数字是什么。

# ====== 单元 20 (代码) ======
train_data = Class2Detect(torchvision.datasets.MNIST("./", train=True, transform=transforms.ToTensor(), download=True))
test_data = Class2Detect(torchvision.datasets.MNIST("./", train=False, transform=transforms.ToTensor(), download=True))

def collate_fn(batch):
    """
    batch 会包含一个 python 对象列表。在我们的场景下，data loader 返回 (Tensor, Dict) 对。
    FasterRCNN 算法需要的是 List[Tensors] 和 List[Dict]。所以我们用这个函数将批数据转换为目标形式，
    再交给 Dataloader 使用
    """
    imgs = []
    labels = []
    for img, label in batch:
        imgs.append(img)
        labels.append(label)
    return imgs, labels

train_loader = DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=collate_fn)

# ====== 单元 21 (代码) ======
x, y = train_data[0] # 取一张图像及其标签
print(imshow(x.numpy()[0, :]))

# ====== 单元 22 (代码) ======
print(y) # 打印全部
print("Boxes: ", y['boxes']) # 打印一个张量，显示全部 3 个目标四个角的像素坐标
print("Labels: ", y['labels']) # 打印一个张量，显示全部 3 个目标的标签

# ====== 单元 23 (代码) ======
# 输入有多少个通道？
C = 1
# 一共有多少类？
classes = 10
# backbone 中有多少个 filter
n_filters = 32

# ====== 单元 24 (代码) ======
backbone = nn.Sequential(
    cnnLayer(C, n_filters),    
    cnnLayer(n_filters, n_filters),
    cnnLayer(n_filters, n_filters),
    nn.MaxPool2d((2,2)),
    cnnLayer(n_filters, 2*n_filters),
    cnnLayer(2*n_filters, 2*n_filters),
    cnnLayer(2*n_filters, 2*n_filters),
    nn.MaxPool2d((2,2)),
    cnnLayer(2*n_filters, 4*n_filters),
    cnnLayer(4*n_filters, 4*n_filters),
)
# 让 Faster RCNN 知道 backbone 的输出通道数
backbone.out_channels = n_filters*4

# ====== 单元 25 (代码) ======
# 要生成多少个候选框 $k$？这里每个 aspect ratio 都为 1，并对多种图像尺寸重复该过程
anchor_generator = AnchorGenerator(sizes=((32),), aspect_ratios=((1.0),)) # 为了运行更快，我们告诉 PyTorch 只寻找 32 x 32 大小的方形图像

# 告诉 PyTorch 使用 backbone 的最终输出作为 featuremap（['0']），并用自适应池化下采样到 7x7 网格（output_size=7）
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
# sampling_ratio 名字起得不好，它控制当预测到分数像素位置（如 5.8 而非 6）时，RoI 从特征图取切片的细节。我们不深入这些底层细节，2 在大多数任务里都是合理的默认值。

# 现在可以创建 FasterRCNN 对象。我们传入 backbone 网络、类别数、处理图像的最小和最大尺寸（我们知道所有图像都是 100 像素）、用于从图像中减去的均值和标准差，以及 anchor 生成器（RPN）和 RoI 对象
model = FasterRCNN(backbone, num_classes=10, image_mean = [0.5], image_std = [0.229], min_size=100, max_size=100, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

# ====== 单元 26 (代码) ======
model = model.train()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters())

for epoch in tqdm(range(1), desc="Epoch", disable=False):
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Train Batch", leave=False, disable=False):
        # 把这个 batch 移动到我们使用的设备上。
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        optimizer.zero_grad()
        # rcnn 需要 model(inputs, labels) —— 而不只是 model(inputs)
        losses = model(inputs, labels)
        # 计算损失，RCNN 会给我们一个待相加的损失列表。
        loss = 0
        for partial_loss in losses.values():
            loss += partial_loss
        # 现在像往常一样继续
        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()

# ====== 单元 27 (代码) ======
model = model.eval()
model = model.to(device)

# ====== 单元 28 (代码) ======
print(set_seed(161))

# ====== 单元 29 (代码) ======
x, y = test_data[0]
print(y) # 这是我们希望得到的理想结果

# ====== 单元 30 (代码) ======
with torch.no_grad():
    pred = model([x.to(device)])

# ====== 单元 31 (代码) ======
print(pred)

# ====== 单元 32 (代码) ======
import matplotlib.patches as patches

# ====== 单元 33 (代码) ======
def plotDetection(ax, abs_pos, label=None):
    """
    ax: 用于添加此图的 matplotlib 坐标轴
    abs_pos: 边界框的位置
    label: 要添加的预测标签
    """
    x1, y1, x2, y2 = abs_pos
    # 为边界框创建一个矩形
    rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    # 如果提供了标签则加入标签
    if label is not None:
        plt.text(x1+0.5, y1, label, color='black', bbox=dict(facecolor='white', edgecolor='white', pad=1.0))

    return

def showPreds(img, pred):
    """
    img: 要显示边界框预测的图像
    pred: 叠加在图像上显示的 Faster R-CNN 预测结果
    """
    fig,ax = plt.subplots(1)
    # 绘制图像
    ax.imshow(img.cpu().numpy()[0,:])
    # 获取预测结果
    boxes = pred['boxes'].cpu()
    labels = pred['labels'].cpu()
    scores = pred['scores'].cpu()

    num_preds = labels.shape[0]
    # 对于每个预测，如果分数足够高就绘制
    for i in range(num_preds):
        plotDetection(ax, boxes[i].cpu().numpy(), label=str(labels[i].item()))
    
    plt.show()

# ====== 单元 34 (代码) ======
print(showPreds(x, pred[0]))

# ====== 单元 35 (代码) ======
from torchvision.ops import nms

# ====== 单元 36 (代码) ======
print(pred[0]['boxes'])

# ====== 单元 37 (代码) ======
print(pred[0]['scores'])

# ====== 单元 38 (代码) ======
print(nms(pred[0]['boxes'], pred[0]['scores'], 0.5))

# ====== 单元 39 (代码) ======
def showPreds(img, pred, iou_max_overlap=0.5, min_score=0.05, label_names=None):
    """
    img: 进行目标检测的原始图像
    pred: FasterRCNN 在 img 上评估输出的字典
    iou_max_overlap: 用于非极大值抑制的 IoU 阈值
    min_score: 视为目标的最低 RPN 网络分数
    """
    fig,ax = plt.subplots(1)
    img = img.cpu().numpy()
    if img.shape[0] == 1:
        ax.imshow(img[0,:])
    else:
        ax.imshow(np.moveaxis(img, 0, 2))
    boxes = pred['boxes'].cpu()
    labels = pred['labels'].cpu()
    scores = pred['scores'].cpu()
    
    selected = nms(boxes, scores, iou_max_overlap).cpu().numpy()
    
    for i in selected:
        if scores[i].item() > min_score:
            if label_names is None:
                label = str(labels[i].item())
            else:
                label = label_names[labels[i].item()]
            plotDetection(ax, boxes[i].cpu().numpy(), label=label)
    
    plt.show()

# ====== 单元 40 (代码) ======
print(showPreds(x, pred[0]))

# ====== 单元 41 (代码) ======
rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# 这个 RCNN 检测器已为一组特定类别预设。你可以通过设置 num_classes=10 和 pretrained_backbone=True，并像我们之前用 MNIST 那样用自己的数据训练，来将其复用到你自己的问题上。

# ====== 单元 42 (代码) ======
rcnn = rcnn.eval()

# ====== 单元 43 (代码) ======
# COCO_INSTANCE_CATEGORY_NAMES，来自 PyTorch 文档。https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
NAME = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ====== 单元 44 (代码) ======
from PIL import Image
import requests
from io import BytesIO

urls = [
    "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/10best-cars-group-cropped-1542126037.jpg",
    "https://miro.medium.com/max/5686/1*ZqJFvYiS5GmLajfUfyzFQA.jpeg",
    "https://www.denverpost.com/wp-content/uploads/2018/03/virginia_umbc_001.jpg?w=910"
]

response = requests.get(urls[0])
img = Image.open(BytesIO(response.content))

# ====== 单元 45 (代码) ======
img = np.asarray(img)/256.0
img = torch.tensor(img, dtype=torch.float32).permute((2,0,1))

with torch.no_grad():
    pred = rcnn([img]) # 把图像传入模型

# ====== 单元 46 (代码) ======
print(showPreds(img, pred[0], iou_max_overlap=0.15, min_score=0.15, label_names=NAME))

