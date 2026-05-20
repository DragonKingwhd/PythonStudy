"""Chapter_3 — 来自 Inside-Deep-Learning 仓库，自动转成可运行 .py 脚本。

原始 notebook 位于 ../Inside-Deep-Learning/。
"""

# ====== 单元 0 (代码) ======
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import pandas as pd

from sklearn.metrics import accuracy_score

import time

from idlmam import train_simple_network, set_seed

# ====== 单元 1 (代码) ======
# [已剥离] %matplotlib inline
# [已剥离] from IPython.display import set_matplotlib_formats
# [已剥离] set_matplotlib_formats('png', 'pdf')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

torch.backends.cudnn.deterministic=True
print(set_seed(1))

# ====== 单元 2 (代码) ======
import torchvision 
from torchvision import transforms

# ====== 单元 3 (代码) ======
mnist_data_train = torchvision.datasets.MNIST("./data", train=True, download=True)
mnist_data_test = torchvision.datasets.MNIST("./data", train=False, download=True)
x_example, y_example = mnist_data_train[0]
print(type(x_example))

# ====== 单元 4 (代码) ======
mnist_data_train = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_data_test = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
x_example, y_example = mnist_data_train[0]
print(x_example.shape)

# ====== 单元 5 (代码) ======
print(imshow(x_example[0, :], cmap='gray'))

# ====== 单元 6 (代码) ======
x_as_color = torch.stack([x_example[0,:], x_example[0,:], x_example[0,:]], dim=0)
print(x_as_color.shape)

# ====== 单元 7 (代码) ======
print(imshow(x_as_color.permute(1, 2, 0)))

# ====== 单元 8 (代码) ======
x_as_color = torch.stack([x_example[0,:], x_example[0,:], x_example[0,:]])
x_as_color[0,:] = 0 #去掉红色通道
#绿色通道保持不变
x_as_color[2,:] = 0 #去掉蓝色通道
print(imshow(x_as_color.permute(1, 2, 0)))

# ====== 单元 9 (代码) ======
#取 3 张图片
x1, x2, x3 = mnist_data_train[0], mnist_data_train[1], mnist_data_train[2]
#丢弃标签
x1, x2, x3 = x1[0], x2[0], x3[0]
x_as_color = torch.stack([x1[0,:], x2[0,:], x3[0,:]], dim=0)
print(imshow(x_as_color.permute(1, 2, 0)))

# ====== 单元 10 (代码) ======
rand_order = torch.randperm(x_example.shape[1] * x_example.shape[2])
x_shuffled = x_example.view(-1)[rand_order].view(x_example.shape)
print(imshow(x_shuffled[0, :], cmap='gray'))

# ====== 单元 11 (代码) ======
from scipy.signal import convolve
img_indx = 58
img = mnist_data_train[img_indx][0][0,:]
plt.imshow(img, vmin=0, vmax=1, cmap='gray')
plt.show()

# ====== 单元 12 (代码) ======
blur_filter = np.asarray([[1,1,1],
                          [1,1,1],
                          [1,1,1]
                         ])/9.0

blurry_img = convolve(img, blur_filter)
plt.imshow(blurry_img, vmin=0, vmax=1, cmap='gray')
plt.show()

# ====== 单元 13 (代码) ======
#我们可以通过关注像素与其邻域之间的差异来检测边缘
edge_filter = np.asarray([[-1,-1,-1],
                          [-1, 8,-1],
                          [-1,-1,-1]
                         ])


edge_img = convolve(img, edge_filter)
plt.imshow(edge_img, vmin=0, vmax=1, cmap='gray')
plt.show()

# ====== 单元 14 (代码) ======
#我们也可以只检测水平方向的边缘
h_edge_filter = np.asarray([[-1,-1,-1],
                          [0, 0,0],
                          [1, 1, 1]
                         ])


h_edge_img = convolve(img, h_edge_filter)
plt.imshow(h_edge_img, vmin=0, vmax=1, cmap='gray')
plt.show()

# ====== 单元 15 (代码) ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch.cuda.is_available() else torch.device("cpu")

B = 32
mnist_train_loader = DataLoader(mnist_data_train, batch_size=B, shuffle=True)
mnist_test_loader = DataLoader(mnist_data_test, batch_size=B)

# ====== 单元 16 (代码) ======
#输入中有多少个数值？用它来确定后续层的尺寸
D = 28*28 #28 * 28 的图像
#输入有多少个通道？
C = 1
#一共有多少类？
classes = 10
#我们要用多少个滤波器？
filters = 16
#滤波器的大小是多少？
K = 3
#为了对比，我们再定义一个复杂度相近的线性模型
model_linear = nn.Sequential(
  nn.Flatten(), #(B, C, W, H) -> (B, C*W*H) = (B,D)
  nn.Linear(D, 256),
  nn.Tanh(),
  nn.Linear(256, classes),
)

#一个简单的卷积神经网络：
model_cnn = nn.Sequential(
  #Conv2d 的参数模式为：
  #Conv2d(输入通道数, 滤波器数/输出通道数, 滤波器大小)
  nn.Conv2d(C, filters, K, padding=K//2), #$x \circledast G$
  nn.Tanh(),#激活函数对任意大小的张量都适用
  nn.Flatten(), #将 (B, C, W, H) 转换成 (B, D)，这样后面就可以接 Linear 层
  nn.Linear(filters*D, classes),
)

# ====== 单元 17 (代码) ======
loss_func = nn.CrossEntropyLoss()
cnn_results = train_simple_network(model_cnn, loss_func, mnist_train_loader, test_loader=mnist_test_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=20)
fc_results = train_simple_network(model_linear, loss_func, mnist_train_loader, test_loader=mnist_test_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=20)

# ====== 单元 18 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_results, label='CNN')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results, label='Fully Conected')
plt.show()

# ====== 单元 19 (代码) ======
img_indx = 0
img, correct_class = mnist_data_train[img_indx]
img = img[0,:]
#分别向右下平移和向左上平移
img_lr = np.roll(np.roll(img, 1, axis=1), 1, axis=0)
img_ul = np.roll(np.roll(img, -1, axis=1), -1, axis=0)
#绘制图片
f, axarr = plt.subplots(1,3)
axarr[0].imshow(img, cmap='gray')
axarr[1].imshow(img_lr, cmap='gray')
axarr[2].imshow(img_ul, cmap='gray')
plt.show()

# ====== 单元 20 (代码) ======
#不是在训练，切到 eval 模式
model = model_cnn.cpu().eval()

def pred(model, img):
    with torch.no_grad():#评估时一定要关闭梯度
        w, h = img.shape#图像的宽和高
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        x = img.reshape(1,-1,w,h)#reshape 为 (B, C, W, H)
        logits = model(x) #得到 logits
        y_hat = F.softmax(logits, dim=1)#转换成概率
        return y_hat.numpy().flatten()#将预测结果转为 numpy 数组

# ====== 单元 21 (代码) ======
img_pred = pred(model, img)
img_lr_pred = pred(model, img_lr)
img_ul_pred = pred(model, img_ul)

print("Org Img Class {} Prob:         ".format(correct_class) , img_pred[correct_class])
print("Lower Right Img Class {} Prob: ".format(correct_class) , img_lr_pred[correct_class])
print("Uper Left Img Class {} Prob:   ".format(correct_class) , img_ul_pred[correct_class])

# ====== 单元 22 (代码) ======
model_cnn_pool = nn.Sequential(
  nn.Conv2d(C, filters, 3, padding=3//2), 
  nn.Tanh(),
  nn.Conv2d(filters, filters, 3, padding=3//2), 
  nn.Tanh(),
  nn.Conv2d(filters, filters, 3, padding=3//2), 
  nn.Tanh(),
  nn.MaxPool2d(2),
  nn.Conv2d(filters, 2*filters, 3, padding=3//2), 
  nn.Tanh(),
  nn.Conv2d(2*filters, 2*filters, 3, padding=3//2), 
  nn.Tanh(),
  nn.Conv2d(2*filters, 2*filters, 3, padding=3//2), 
  nn.Tanh(),
  nn.MaxPool2d(2),

  nn.Flatten(), 
  #为什么 Linear 层的输入维度要除以 $4^2$？因为把 2x2 的网格池化成 1 个值意味着从 4 个值降到 1 个，我们这么做了两次。
  nn.Linear(2*filters*D//(4**2), classes),
)

cnn_results_with_pool = train_simple_network(model_cnn_pool, loss_func, mnist_train_loader, test_loader=mnist_test_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=20)

# ====== 单元 23 (代码) ======
model = model_cnn_pool.cpu().eval()
img_pred = pred(model, img)
img_lr_pred = pred(model, img_lr)
img_ul_pred = pred(model, img_ul)

print("Org Img Class {} Prob:         ".format(correct_class) , img_pred[correct_class])
print("Lower Right Img Class {} Prob: ".format(correct_class) , img_lr_pred[correct_class])
print("Uper Left Img Class {} Prob:   ".format(correct_class) , img_ul_pred[correct_class])

# ====== 单元 24 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_results, label='Simple CNN')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_results_with_pool, label='CNN w/ Max Pooling')
plt.show()

# ====== 单元 25 (代码) ======
#一些内置的图像变换，参数取得比较夸张以便更明显地看到效果。
sample_transforms = {
    "Rotation" : transforms.RandomAffine(degrees=45),
    "Translation" : transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
    "Shear": transforms.RandomAffine(degrees=0, shear=45),
    "RandomCrop" : transforms.RandomCrop((20,20)),
    "Horizontal Flip" : transforms.RandomHorizontalFlip(p=1.0),
    "Vertical Flip": transforms.RandomVerticalFlip(p=1.0),
    "Perspective": transforms.RandomPerspective(p=1.0),   
    "ColorJitter" : transforms.ColorJitter(brightness=0.9, contrast=0.9)
}
#使用一个 transform 把 Tensor 图像转回 PIL 图像
pil_img = transforms.ToPILImage()(img)
#随机应用每一种 transform 并绘制结果
f, axarr = plt.subplots(2,4)
for count, (name, t) in enumerate(sample_transforms.items()):
    row = count % 4
    col = count // 4
    axarr[col,row].imshow(t(pil_img), cmap='gray')
    axarr[col,row].set_title(name)
plt.show()

# ====== 单元 26 (代码) ======
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.98, 1.02)),
    transforms.ToTensor(),
])

test_transform = transforms.ToTensor()

mnist_train_t = torchvision.datasets.MNIST("./data", train=True, transform=train_transform)
mnist_test_t = torchvision.datasets.MNIST("./data", train=False, transform=test_transform)
mnist_train_loader_t = DataLoader(mnist_train_t, shuffle=True,  batch_size=B, num_workers=5)
mnist_test_loader_t = DataLoader(mnist_test_t, batch_size=B, num_workers=5)

# ====== 单元 27 (代码) ======
model_cnn_pool = nn.Sequential(
  nn.Conv2d(C, filters, 3, padding=3//2), 
  nn.Tanh(),
  nn.Conv2d(filters, filters, 3, padding=3//2), 
  nn.Tanh(),
  nn.Conv2d(filters, filters, 3, padding=3//2), 
  nn.Tanh(),
  nn.MaxPool2d(2),
  nn.Conv2d(filters, 2*filters, 3, padding=3//2), 
  nn.Tanh(),
  nn.Conv2d(2*filters, 2*filters, 3, padding=3//2), 
  nn.Tanh(),
  nn.Conv2d(2*filters, 2*filters, 3, padding=3//2), 
  nn.Tanh(),
  nn.MaxPool2d(2),
  nn.Flatten(), 
  nn.Linear(2*filters*D//(4**2), classes),
)

cnn_results_with_pool_augmented = train_simple_network(model_cnn_pool, loss_func, mnist_train_loader_t, test_loader=mnist_test_loader_t, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=20)

# ====== 单元 28 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_results_with_pool, label='CNN w/ Max Pooling')
sns.lineplot(x='epoch', y='test Accuracy', data=cnn_results_with_pool_augmented, label='CNN w/ Max Pooling + Augmentation')
plt.show()

