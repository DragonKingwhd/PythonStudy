"""Chapter_2 — 来自 Inside-Deep-Learning 仓库，自动转成可运行 .py 脚本。

原始 notebook 位于 ../Inside-Deep-Learning/。
"""

# ====== 单元 0 (代码) ======
from tqdm import tqdm

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

import time

# ====== 单元 1 (代码) ======
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import * 
from idlmam import *

# ====== 单元 2 (代码) ======
# [已剥离] %matplotlib inline
# [已剥离] from IPython.display import set_matplotlib_formats
# [已剥离] set_matplotlib_formats('png', 'pdf')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

torch.backends.cudnn.deterministic=True
print(set_seed(42))

# ====== 单元 3 (代码) ======
def train_simple_network(model, loss_func, training_loader, epochs=20, device="cpu"):
    #这里完成黄色步骤：创建优化器并将模型放到计算设备上
    #SGD 即对参数 $\Theta$ 进行的随机梯度下降
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    #将模型放到正确的计算资源上（CPU 或 GPU）
    model.to(device)
    #接下来的两个 for 循环处理红色步骤：多轮（epochs）遍历所有数据（批 batch）
    for epoch in tqdm(range(epochs), desc="Epoch"):

        model = model.train()#将模型切换到训练模式
        running_loss = 0.0

        for inputs, labels in tqdm(training_loader, desc="Batch", leave=False):
            #将这一批数据搬到我们使用的设备上，这是最后一个红色步骤
            inputs = moveTo(inputs, device)
            labels = moveTo(labels, device)

            #首先是一个黄色步骤：准备好优化器。绝大多数 PyTorch 代码都会先做这一步，以保证状态是干净就绪的。

            #PyTorch 把梯度存储在一个可变数据结构中，使用前必须先清零，
            #否则会残留上一次迭代的旧梯度
            optimizer.zero_grad()

            #接下来两行代码执行两个蓝色步骤
            y_hat = model(inputs) #这里就是在计算 $f_\theta(\boldsymbol{x_i})$

            # 计算损失
            loss = loss_func(y_hat, labels)

            #接着剩下的两个黄色步骤：计算梯度并调用优化器的 .step()
            loss.backward()# 这一次调用就完成了 $\nabla_\Theta$ 的计算

            #然后只需更新所有参数即可
            optimizer.step()# $\Theta_{k+1} = \Theta_k − \eta \cdot \nabla_\Theta \ell(\hat{y}, y)$

            #这里只是顺便记录一些我们想要的信息
            running_loss += loss.item()
#说明：这段代码定义了一个简单的训练循环，可以用来学习本书中几乎所有神经网络 $f_\Theta(\cdot)$ 的参数 $\Theta$。

# ====== 单元 4 (代码) ======
#创建一个一维的输入
X = np.linspace(0, 20, num=200)
#创建对应的输出
y = X + np.sin(X)*2 + np.random.normal(size=X.shape)
sns.scatterplot(x=X, y=y)
plt.show()

# ====== 单元 5 (代码) ======
class Simple1DRegressionDataset(Dataset):
        
    def __init__(self, X, y):
        super(Simple1DRegressionDataset, self).__init__()
        self.X = X.reshape(-1,1)
        self.y = y.reshape(-1,1)
        
    
    def __getitem__(self, index):
        return torch.tensor(self.X[index,:], dtype=torch.float32), torch.tensor(self.y[index], dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]
    
training_loader = DataLoader(Simple1DRegressionDataset(X, y), shuffle=True)

# ====== 单元 6 (代码) ======
in_features = 1
out_features = 1
model = nn.Linear(in_features, out_features)
loss_func = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(train_simple_network(model, loss_func, training_loader, device=device))

# ====== 单元 7 (代码) ======
with torch.no_grad():
    Y_pred = model(torch.tensor(X.reshape(-1,1), device=device, dtype=torch.float32)).cpu().numpy()

# ====== 单元 8 (代码) ======
sns.scatterplot(x=X, y=y, color='blue', label='Data') #原始数据
sns.lineplot(x=X, y=Y_pred.ravel(), color='red', label='Linear Model') #模型学到的结果
plt.show()

# ====== 单元 9 (代码) ======
#输入“层”就是输入本身，是隐含的
model = nn.Sequential(
    nn.Linear(1,  10), #隐藏层
    nn.Linear(10, 1), #输出层
)

print(train_simple_network(model, loss_func, training_loader))

# ====== 单元 10 (代码) ======
with torch.no_grad():
    Y_pred = model(torch.tensor(X.reshape(-1,1), dtype=torch.float32)).cpu().numpy() #形状为 (N, 1)

sns.scatterplot(x=X, y=y, color='blue', label='Data') #原始数据
sns.lineplot(x=X, y=Y_pred.ravel(), color='red', label='Model') #模型学到的结果
plt.show()

# ====== 单元 11 (代码) ======
activation_input = np.linspace(-2, 2, num=200)
tanh_activation = np.tanh(activation_input)
sigmoid_activation = np.exp(activation_input)/(np.exp(activation_input)+1)
sns.lineplot(x=activation_input, y=activation_input, color='black', label="linear")
sns.lineplot(x=activation_input, y=tanh_activation, color='red', label="tanh(x)")
ax = sns.lineplot(x=activation_input, y=sigmoid_activation, color='blue', label="$\sigma(x)$")
ax.set_xlabel('Input value x')
ax.set_ylabel('Activation')
plt.show()

# ====== 单元 12 (代码) ======
model = nn.Sequential(
    nn.Linear(1,  10),#隐藏层
    nn.Tanh(),#激活函数
    nn.Linear(10, 1),#输出层
)

print(train_simple_network(model, loss_func, training_loader, epochs=200))

# ====== 单元 13 (代码) ======
with torch.no_grad():
    Y_pred = model(torch.tensor(X.reshape(-1,1), dtype=torch.float32)).cpu().numpy()

sns.scatterplot(x=X, y=y, color='blue', label='Data') #原始数据
sns.lineplot(x=X, y=Y_pred.ravel(), color='red', label='Model') #模型学到的结果
plt.show()

# ====== 单元 14 (代码) ======
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05)
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, style=y)
plt.show()

# ====== 单元 15 (代码) ======
classification_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
training_loader = DataLoader(classification_dataset)

# ====== 单元 16 (代码) ======
in_features = 2
out_features = 2
model = nn.Linear(in_features, out_features)

# ====== 单元 17 (代码) ======
loss_func = nn.CrossEntropyLoss()
print(train_simple_network(model, loss_func, training_loader, epochs=50))

# ====== 单元 18 (代码) ======
def visualize2DSoftmax(X, y, model, title=None):
    x_min = np.min(X[:,0])-0.5
    x_max = np.max(X[:,0])+0.5
    y_min = np.min(X[:,1])-0.5
    y_max = np.max(X[:,1])+0.5
    xv, yv = np.meshgrid(np.linspace(x_min, x_max, num=20), np.linspace(y_min, y_max, num=20), indexing='ij')
    xy_v = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
    with torch.no_grad():
        logits = model(torch.tensor(xy_v, dtype=torch.float32))
        y_hat = F.softmax(logits, dim=1).numpy()

    cs = plt.contourf(xv, yv, y_hat[:,0].reshape(20,20), levels=np.linspace(0,1,num=20), cmap=plt.cm.RdYlBu)
    ax = plt.gca()
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, style=y, ax=ax)
    if title is not None:
        ax.set_title(title)

print(visualize2DSoftmax(X, y, model))
plt.show()

# ====== 单元 19 (代码) ======
model = nn.Sequential(
    nn.Linear(2,  30),
    nn.Tanh(),
    nn.Linear(30,  30),
    nn.Tanh(),
    nn.Linear(30, 2),
)
print(train_simple_network(model, loss_func, training_loader, epochs=250))

# ====== 单元 20 (代码) ======
print(visualize2DSoftmax(X, y, model))

# ====== 单元 21 (代码) ======
def run_epoch(model, optimizer, data_loader, loss_func, device, results, score_funcs, prefix="", desc=None):
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    for inputs, labels in tqdm(data_loader, desc=desc, leave=False):
        #将当前 batch 移动到我们使用的设备上
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        y_hat = model(inputs) #这一步计算的就是 f_Θ(x(i))
        # 计算损失
        loss = loss_func(y_hat, labels)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #这里只是记录一些我们想要的信息
        running_loss.append(loss.item())

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            #将 labels 和预测值移回 CPU 以便后续计算/存储
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            #追加到目前为止的预测结果中
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())
    #本轮训练结束
    end = time.time()

    y_pred = np.asarray(y_pred)
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1: #是分类问题，转换成类别标签
        y_pred = np.argmax(y_pred, axis=1)
    #否则就当作回归问题处理

    results[prefix + " loss"].append( np.mean(running_loss) )
    for name, score_func in score_funcs.items():
        try:
            results[prefix + " " + name].append( score_func(y_true, y_pred) )
        except:
            results[prefix + " " + name].append(float("NaN"))
    return end-start #本轮耗费的时间

# ====== 单元 22 (代码) ======
def train_simple_network(model, loss_func, train_loader, test_loader=None, score_funcs=None,
                         epochs=50, device="cpu", checkpoint_file=None):
    to_track = ["epoch", "total time", "train loss"]
    if test_loader is not None:
        to_track.append("test loss")
    for eval_score in score_funcs:
        to_track.append("train " + eval_score )
        if test_loader is not None:
            to_track.append("test " + eval_score )

    total_train_time = 0 #我们在训练循环里总共花了多少时间？
    results = {}
    #把每一项都初始化为空列表
    for item in to_track:
        results[item] = []

    #SGD 即随机梯度下降
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    #将模型放到正确的计算资源上（CPU 或 GPU）
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Epoch"):
        model = model.train()#将模型切换到训练模式
        
        total_train_time += run_epoch(model, optimizer, train_loader, loss_func, device, results, score_funcs, prefix="train", desc="Training")

        results["total time"].append( total_train_time )
        results["epoch"].append( epoch )
        
        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(model, optimizer, test_loader, loss_func, device, results, score_funcs, prefix="test", desc="Testing")
                    
    if checkpoint_file is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results' : results
            }, checkpoint_file)

    return pd.DataFrame.from_dict(results)

# ====== 单元 23 (代码) ======
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# ====== 单元 24 (代码) ======
X_train, y_train = make_moons(n_samples=8000, noise=0.4)
X_test, y_test = make_moons(n_samples=200, noise=0.4)
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
training_loader = DataLoader(train_dataset, shuffle=True)
testing_loader = DataLoader(test_dataset)

# ====== 单元 25 (代码) ======
model = nn.Sequential(
    nn.Linear(2,  30),
    nn.Tanh(),
    nn.Linear(30,  30),
    nn.Tanh(),
    nn.Linear(30, 2),
)
results_pd = train_simple_network(model, loss_func, training_loader, epochs=5, test_loader=testing_loader, checkpoint_file='model.pt', score_funcs={'Acc':accuracy_score,'F1': f1_score})

# ====== 单元 26 (代码) ======
model_new = nn.Sequential(
    nn.Linear(2,  30),
    nn.Tanh(),
    nn.Linear(30,  30),
    nn.Tanh(),
    nn.Linear(30, 2),
)

visualize2DSoftmax(X_test, y_test, model_new, title="Initial Model")
plt.show()

checkpoint_dict = torch.load('model.pt', map_location=device)


model_new.load_state_dict(checkpoint_dict['model_state_dict'])

visualize2DSoftmax(X_test, y_test, model_new, title="Loaded Model")
plt.show()

# ====== 单元 27 (代码) ======
sns.lineplot(x='epoch', y='train Acc', data=results_pd, label='Train')
sns.lineplot(x='epoch', y='test Acc', data=results_pd, label='Validation')
plt.show()

# ====== 单元 28 (代码) ======
sns.lineplot(x='total time', y='train F1', data=results_pd, label='Train')
sns.lineplot(x='total time', y='test F1', data=results_pd, label='Validation')
plt.show()

# ====== 单元 29 (代码) ======
training_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
testing_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
model_gd = nn.Sequential(
    nn.Linear(2,  30),
    nn.Tanh(),
    nn.Linear(30,  30),
    nn.Tanh(),
    nn.Linear(30, 2),
)
results_true_gd = train_simple_network(model_gd, loss_func, training_loader, epochs=5, test_loader=testing_loader, checkpoint_file='model.pt', score_funcs={'Acc':accuracy_score,'F1': f1_score})

# ====== 单元 30 (代码) ======
sns.lineplot(x='total time', y='test Acc', data=results_pd, label='SGD, B=1')
sns.lineplot(x='total time', y='test Acc', data=results_true_gd, label='GD, B=N')
plt.show()

# ====== 单元 31 (代码) ======
training_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
model_sgd = nn.Sequential(
    nn.Linear(2,  30),
    nn.Tanh(),
    nn.Linear(30,  30),
    nn.Tanh(),
    nn.Linear(30, 2),
)
results_batched = train_simple_network(model_sgd, loss_func, training_loader, epochs=5, test_loader=testing_loader, checkpoint_file='model.pt', score_funcs={'Acc':accuracy_score,'F1': f1_score})

# ====== 单元 32 (代码) ======
sns.lineplot(x='total time', y='test Acc', data=results_pd, label='SGD, B=1')
sns.lineplot(x='total time', y='test Acc', data=results_true_gd, label='GD, B=N')
sns.lineplot(x='total time', y='test Acc', data=results_batched, label='SGD, B=32')
plt.show()

