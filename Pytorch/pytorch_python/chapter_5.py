"""Chapter_5 — 来自 Inside-Deep-Learning 仓库，自动转成可运行 .py 脚本。

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

from idlmam import train_simple_network, Flatten, weight_reset, set_seed, run_epoch

# ====== 单元 1 (代码) ======
# [已剥离] %matplotlib inline
# [已剥离] from IPython.display import set_matplotlib_formats
# [已剥离] set_matplotlib_formats('png', 'pdf')

from IPython.display import display_pdf
from IPython.display import Latex

# ====== 单元 2 (代码) ======
torch.backends.cudnn.deterministic=True
print(set_seed(45))

# ====== 单元 3 (代码) ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch.cuda.is_available() else torch.device("cpu")

# ====== 单元 4 (代码) ======
def train_network(model, loss_func, train_loader, val_loader=None, test_loader=None,score_funcs=None, 
                         epochs=50, device="cpu", checkpoint_file=None, 
                         lr_schedule=None, optimizer=None, disable_tqdm=False
                        ):
    """训练简单的神经网络

    关键字参数：
    model -- 要训练的 PyTorch 模型 / "Module"
    loss_func -- 损失函数，以两个参数（模型输出和标签）作为一个 batch 的输入，返回一个分数
    train_loader -- PyTorch DataLoader 对象，返回 (input, label) 形式的元组
    val_loader -- 可选的 PyTorch DataLoader，每个 epoch 之后用于评估
    test_loader -- 可选的 PyTorch DataLoader，每个 epoch 之后用于评估
    score_funcs -- 字典，包含用于评估模型表现的打分函数
    epochs -- 要执行的训练轮数
    device -- 训练所使用的计算设备
    lr_schedule -- 学习率调度器，用于在训练过程中改变 \eta。如果不为 None，那么用户必须同时提供 optimizer。
    optimizer -- 用于在学习中调整梯度的方法。

    """
    if score_funcs == None:
        score_funcs = {}# 空字典
    
    to_track = ["epoch", "total time", "train loss"]
    if val_loader is not None:
        to_track.append("val loss")
    if test_loader is not None:
        to_track.append("test loss")
    for eval_score in score_funcs:
        to_track.append("train " + eval_score )
        if val_loader is not None:
            to_track.append("val " + eval_score )
        if test_loader is not None:
            to_track.append("test "+ eval_score )
        
    total_train_time = 0 # 已经在训练循环中花了多长时间？
    results = {}
    # 用空列表初始化每一项
    for item in to_track:
        results[item] = []

    if optimizer == None:
        # AdamW 优化器是一个不错的默认优化器
        optimizer = torch.optim.AdamW(model.parameters())

    # 将模型放置到正确的计算资源（CPU 或 GPU）上
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Epoch", disable=disable_tqdm):
        model = model.train()# 将模型切换到训练模式

        total_train_time += run_epoch(model, optimizer, train_loader, loss_func, device, results, score_funcs, prefix="train", desc="Training")
        
        results["epoch"].append( epoch )
        results["total time"].append( total_train_time )
        
      
        if val_loader is not None:
            model = model.eval() # 把模型设为"评估"模式，因为我们不想再做任何参数更新！
            with torch.no_grad():
                run_epoch(model, optimizer, val_loader, loss_func, device, results, score_funcs, prefix="val", desc="Validating")

        # 在 PyTorch 中，惯例是在每个 epoch 之后更新学习率
        if lr_schedule is not None:
            if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val loss"][-1])
            else:
                lr_schedule.step()
                
        if test_loader is not None:
            model = model.eval() # 把模型设为"评估"模式，因为我们不想再做任何参数更新！
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

# ====== 单元 5 (代码) ======
epochs = 50 # 训练 50 个 epoch
B = 256 # 一个比较合理的平均 batch size
train_data = torchvision.datasets.FashionMNIST("./data", train=True, transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.FashionMNIST("./data", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(train_data, batch_size=B, shuffle=True)
test_loader = DataLoader(test_data, batch_size=B)

# ====== 单元 6 (代码) ======
# 输入中有多少个值？用它来帮助确定后续层的大小
D = 28*28 # 28 * 28 的图像
# 隐藏层大小
n = 128
# 输入有多少个通道？
C = 1
# 一共有多少类？
classes = 10

fc_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(D,  n),
    nn.Tanh(),
    nn.Linear(n,  n),
    nn.Tanh(),
    nn.Linear(n,  n),
    nn.Tanh(),
    nn.Linear(n, classes),
)

# ====== 单元 7 (代码) ======
eta_0 = 0.1

# ====== 单元 8 (代码) ======
loss_func = nn.CrossEntropyLoss()

# 用新写的 train_network 函数，以与之前等价的方式调用
fc_results = train_network(fc_model, loss_func, train_loader, test_loader=test_loader, epochs=epochs, optimizer=torch.optim.SGD(fc_model.parameters(), lr=eta_0), score_funcs={'Accuracy': accuracy_score}, device=device)

# ====== 单元 9 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results, label='Fully Connected')
plt.show()

# ====== 单元 10 (代码) ======
T=50 # 总 epoch 数
epochs_input = np.linspace(0, 50, num=50) # 生成所有不同的 t 值
eta_init = 0.001 # 假设的初始学习率 $\eta_0$
eta_min = 0.0001 # 假设的期望最小学习率 $\eta_{\mathit{min}}$
gamma = np.power(eta_min/eta_init,1./T) # 计算衰减率 $\gamma$

effective_learning_rate = eta_init*np.power(gamma, epochs_input) # 所有的 $\eta_t$ 值

sns.lineplot(x=epochs_input, y=eta_init, color='red', label="$\eta_0$")
ax = sns.lineplot(x=epochs_input, y=effective_learning_rate, color='blue', label="$\eta_0 \cdot \gamma^t$")
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
plt.show()

# ====== 单元 11 (代码) ======
fc_model.apply(weight_reset)# 重新随机化模型权重，这样就不需要再次定义它

eta_min = 0.0001 # 我们期望的最终学习率 $\eta_{\mathit{min}}$

gamma_expo = (eta_min/eta_0)**(1/epochs)# 计算能得到 $\eta_{\mathit{min}}$ 的 $\gamma$

optimizer = torch.optim.SGD(fc_model.parameters(), lr=eta_0) # 设置优化器
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma_expo)# 选择一个调度器并把优化器传入
# 像往常一样训练，并把所需的优化器和调度器传进去
fc_results_expolr = train_network(fc_model, loss_func, train_loader, test_loader=test_loader, epochs=epochs, optimizer=optimizer, lr_schedule=scheduler, score_funcs={'Accuracy': accuracy_score}, device=device)

# ====== 单元 12 (代码) ======
sns.lineplot(x=epochs_input, y=eta_init, color='red', label="$\eta_0$")
sns.lineplot(x=epochs_input, y=[eta_init]*18+[eta_init/3.16]*16+[eta_init/10]*16, color='green', label="StepLR")
ax = sns.lineplot(x=epochs_input, y=effective_learning_rate, color='blue', label="$\eta_0 \cdot \gamma^t$")
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
plt.show()

# ====== 单元 13 (代码) ======
fc_model.apply(weight_reset)

optimizer = torch.optim.SGD(fc_model.parameters(), lr=eta_0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, epochs//4, gamma=0.3)# 每经过 epochs/4 个 epoch 就按系数 $\gamma$ 下降一次，因此总共会发生 4 次。

fc_results_steplr = train_network(fc_model, loss_func, train_loader, test_loader=test_loader, epochs=epochs, optimizer=optimizer, lr_schedule=scheduler, score_funcs={'Accuracy': accuracy_score}, device=device)

# ====== 单元 14 (代码) ======
cos_lr = eta_min + 0.5*(eta_init-eta_min)*(1+np.cos(epochs_input/(T/3.0)*np.pi))# 对每个 $t$ 计算余弦调度的 $\eta_t$

sns.lineplot(x=epochs_input, y=eta_init, color='red', label="$\eta_0$")
sns.lineplot(x=epochs_input, y=cos_lr, color='purple', label="$\cos$")
sns.lineplot(x=epochs_input, y=[eta_init]*18+[eta_init/3.16]*16+[eta_init/10]*16, color='green', label="StepLR")
ax = sns.lineplot(x=epochs_input, y=effective_learning_rate, color='blue', label="$\eta_0 \cdot \gamma^t$")
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
plt.show()

# ====== 单元 15 (代码) ======
fc_model.apply(weight_reset)

optimizer = torch.optim.SGD(fc_model.parameters(), lr=eta_0)
# 让余弦下降/上升/下降（共3段），如果做超过 10 个 epoch，我会把这个值调得更大
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs//3, eta_min=0.0001)
fc_results_coslr = train_network(fc_model, loss_func, train_loader, test_loader=test_loader, epochs=epochs, optimizer=optimizer, lr_schedule=scheduler, score_funcs={'Accuracy': accuracy_score}, device=device)

# ====== 单元 16 (代码) ======
sns.lineplot(x='epoch', y='test loss', data=fc_results, label='Test Loss')
sns.lineplot(x='epoch', y='train loss', data=fc_results, label='Train Loss')
plt.show()

# ====== 单元 17 (代码) ======
fc_model.apply(weight_reset) # 再次重置权重，这样就不需要定义新模型。

# 创建训练和验证子集，因为我们没有显式的验证集和测试集
train_sub_set, val_sub_set = torch.utils.data.random_split(train_data, [int(len(train_data)*0.8), int(len(train_data)*0.2)])

# 为训练和验证子集创建 loader
train_sub_loader = DataLoader(train_sub_set, batch_size=B, shuffle=True)
val_sub_loader = DataLoader(val_sub_set, batch_size=B)
# test loader 保持不变，永远不要修改或偷看你的测试数据！

optimizer = torch.optim.SGD(fc_model.parameters(), lr=eta_0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)# 使用 gamma=0.2 的 plateau 调度器
# 开始训练模型！
fc_results_plateau = train_network(fc_model, loss_func, train_loader, val_loader=val_sub_loader, test_loader=test_loader, epochs=epochs, optimizer=optimizer, lr_schedule=scheduler, score_funcs={'Accuracy': accuracy_score}, device=device)

# ====== 单元 18 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results, label='SGD')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results_expolr, label='+Exponential Decay')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results_steplr, label='+StepLR')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results_coslr, label='+CosineLR')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results_plateau, label='+Plateau')
plt.show()

# ====== 单元 19 (代码) ======
fc_model.apply(weight_reset)

optimizer = torch.optim.SGD(fc_model.parameters(), lr=eta_0, momentum=0.9, nesterov=False)

fc_results_momentum = train_network(fc_model, loss_func, train_loader, test_loader=test_loader, epochs=epochs, optimizer=optimizer, score_funcs={'Accuracy': accuracy_score}, device=device)

# ====== 单元 20 (代码) ======
fc_model.apply(weight_reset)

optimizer = torch.optim.SGD(fc_model.parameters(), lr=eta_0, momentum=0.9, nesterov=True)

fc_results_nestrov = train_network(fc_model, loss_func, train_loader, test_loader=test_loader, epochs=epochs, optimizer=optimizer, score_funcs={'Accuracy': accuracy_score}, device=device)

# ====== 单元 21 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results, label='SGD')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results_momentum, label='SGD w/ Momentum')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results_nestrov, label='SGD w/ Nestrov Momentum')
plt.show()

# ====== 单元 22 (代码) ======
fc_model.apply(weight_reset)

# 我们不为 Adam 设置学习率，因为它的默认值通常就是你应该使用的，
# 而且它对学习率的大幅变化可能更敏感
optimizer = torch.optim.AdamW(fc_model.parameters())

fc_results_adam = train_network(fc_model, loss_func, train_loader, test_loader=test_loader, epochs=epochs, optimizer=optimizer, score_funcs={'Accuracy': accuracy_score}, device=device)

# ====== 单元 23 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results, label='SGD')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results_momentum, label='SGD w/ Momentum')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results_nestrov, label='SGD w/ Nestrov Momentum')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results_adam, label='AdamW')
plt.show()

# ====== 单元 24 (代码) ======
# Adam 配合余弦退火
fc_model.apply(weight_reset)
optimizer = torch.optim.AdamW(fc_model.parameters())

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs//3)
fc_results_adam_coslr = train_network(fc_model, loss_func, train_loader, test_loader=test_loader, epochs=epochs, optimizer=optimizer, lr_schedule=scheduler, score_funcs={'Accuracy': accuracy_score}, device=device)

# SGD+Nesterov 配合余弦退火
fc_model.apply(weight_reset)
optimizer = torch.optim.SGD(fc_model.parameters(), lr=eta_0, momentum=0.9, nesterov=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs//3)
fc_results_nestrov_coslr = train_network(fc_model, loss_func, train_loader, test_loader=test_loader, epochs=epochs, optimizer=optimizer, lr_schedule=scheduler, score_funcs={'Accuracy': accuracy_score}, device=device)

# ====== 单元 25 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results_nestrov, label='SGD w/ Nestrov')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results_nestrov_coslr, label='SGD w/ Nestrov+CosineLR')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results_adam, label='AdamW')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results_adam_coslr, label='AdamW+CosineLR')
plt.show()

# ====== 单元 26 (代码) ======
fc_model.apply(weight_reset)

for p in fc_model.parameters(): # 这一步实现 $\operatorname{clip}_5(\boldsymbol{g})$
    p.register_hook(lambda grad: torch.clamp(grad, -5, 5)) 

optimizer = torch.optim.AdamW(fc_model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs//3)
fc_results_nestrov_coslr_clamp = train_network(fc_model, loss_func, train_loader, test_loader=test_loader, epochs=epochs, optimizer=optimizer,  lr_schedule=scheduler, score_funcs={'Accuracy': accuracy_score}, device=device)

# ====== 单元 27 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results_nestrov_coslr, label='AdamW+CosineLR')
sns.lineplot(x='epoch', y='test Accuracy', data=fc_results_nestrov_coslr_clamp, label='AdamW+CosineLR+Clamp')
plt.show()

# ====== 单元 28 (代码) ======
# 如果你没有安装 optuna，取消下面一行的注释
#!pip install optuna

# ====== 单元 29 (代码) ======
import optuna # 现在可以导入 optuna

# ====== 单元 30 (代码) ======
def toyFunc(trial):
    # 下面两次调用向 optuna 请求两个参数，并为每个参数定义最小值和最大值。
    x = trial.suggest_uniform('x', -10.0, 10.0) #$x \sim \mathcal{U}(-10,10)$
    y = trial.suggest_uniform('y', -10.0, 10.0) #$y \sim \mathcal{U}(-10,10)$
    # 现在可以计算并返回结果。Optuna 将尝试最小化这个值
    return abs((x-3)*(y+2)) #|(x-3)\cdot(y+2)|

# ====== 单元 31 (代码) ======
study = optuna.create_study(direction='minimize') # 如果设为 direction='maximize'，Optuna 会尝试最大化 toyFunc 的返回值
print(study.optimize(toyFunc, n_trials=100))

# ====== 单元 32 (代码) ======
print(study.best_params) # 这个字典保存了 Optuna 找到的最优参数值

# ====== 单元 33 (代码) ======
fig = optuna.visualization.plot_contour(study)

# ====== 单元 34 (代码) ======
def objective(trial):
    
    train_subset = int(len(train_data)*0.8)
    test_subset = len(train_data)-train_subset
    
    split = torch.utils.data.random_split(train_data, [train_subset, test_subset])
    
    t_loader = DataLoader(split[0], batch_size=B, shuffle=True)
    v_loader = DataLoader(split[1], batch_size=B, shuffle=False)

    # 隐藏层大小
    n = trial.suggest_int('neurons_per_layer', 16, 256)
    layers = trial.suggest_int('hidden_layers', 1, 6)
    # 输入有多少个通道？
    C = 1
    # 一共有多少类？
    classes = 10

    # 至少有一个隐藏层，接收 D 个输入
    sequential_layers = [
        nn.Flatten(),
        nn.Linear(D,  n),
        nn.Tanh(),
    ]
    # 现在根据 Optuna 给出的 "layers" 参数添加可变数量的隐藏层
    for _ in range(layers-1):
        sequential_layers.append( nn.Linear(n,  n) )
        sequential_layers.append( nn.Tanh() )

    # 输出层
    sequential_layers.append( nn.Linear(n, classes) )

    # 把层的列表组合成一个 PyTorch Sequential 模块
    fc_model = nn.Sequential(*sequential_layers)
    # 全局学习率应当是多少？注意我们可以随时向 optuna 请求新的超参数。
    eta_global = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    
    optimizer = torch.optim.AdamW(fc_model.parameters(), lr=eta_global)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs//3)
    results = train_network(fc_model, loss_func, t_loader, test_loader=v_loader,
                                     epochs=10, optimizer=optimizer, lr_schedule=scheduler,
                                     score_funcs={'Accuracy': accuracy_score}, device=device, 
                                     disable_tqdm=True)

    return results['test Accuracy'].iloc[-1]  # 与 Trial 对象关联的目标值。

# ====== 单元 35 (代码) ======
study = optuna.create_study(direction='maximize')
print(study.optimize(objective, n_trials=10))
# 这里我们少做一些，以保证 notebook 在合理时间内跑完

# ====== 单元 36 (代码) ======
print(study.best_params)

# ====== 单元 37 (代码) ======
fig = optuna.visualization.plot_optimization_history(study)
fig.show()

# ====== 单元 38 (代码) ======
fig = optuna.visualization.plot_slice(study)
fig.show()

# ====== 单元 39 (代码) ======
fig = optuna.visualization.plot_contour(study, params=['neurons_per_layer', 'hidden_layers', "learning_rate"])
fig.show()

# ====== 单元 40 (代码) ======
def objectivePrunable(trial):
    
    train_subset = int(len(train_data)*0.8)
    test_subset = len(train_data)-train_subset
    
    split = torch.utils.data.random_split(train_data, [train_subset, test_subset])
    
    t_loader = DataLoader(split[0], batch_size=B, shuffle=True)
    v_loader = DataLoader(split[1], batch_size=B, shuffle=False)

    # 隐藏层大小
    n = trial.suggest_int('neurons_per_layer', 1, 256)
    layers = trial.suggest_int('hidden_layers', 1, 6)
    # 输入有多少个通道？
    C = 1
    # 一共有多少类？
    classes = 10

    # 至少有一个隐藏层，接收 D 个输入
    sequential_layers = [
        Flatten(),
        nn.Linear(D,  n),
        nn.Tanh(),
    ]

    for _ in range(layers-1):
        sequential_layers.append( nn.Linear(n,  n) )
        sequential_layers.append( nn.Tanh() )

    # 输出层
    sequential_layers.append( nn.Linear(n, classes) )
    

    fc_model = nn.Sequential(*sequential_layers)
    
    eta_global = trial.suggest_loguniform('learning_rate', 1e-6, 1e+2)

    # 需要在 train_network 调用之外创建优化器（以及任何学习率调度器），这样同一个优化器可以在多个 epoch 之间复用
    optimizer = torch.optim.AdamW(fc_model.parameters(), lr=eta_global)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs//3)
    
    for epoch in range(10):
    
        results = train_network(fc_model, loss_func, t_loader, val_loader=v_loader,
                                         epochs=1, optimizer=optimizer, lr_schedule=scheduler,
                                         score_funcs={'Accuracy': accuracy_score}, device=device, 
                                         disable_tqdm=True)
        cur_accuracy = results['val Accuracy'].iloc[-1]
        trial.report(cur_accuracy, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return cur_accuracy

# ====== 单元 41 (代码) ======
study2 = optuna.create_study(direction='maximize')
print(study2.optimize(objectivePrunable, n_trials=20))

# ====== 单元 42 (代码) ======
fig = optuna.visualization.plot_intermediate_values(study2)
fig.show()

