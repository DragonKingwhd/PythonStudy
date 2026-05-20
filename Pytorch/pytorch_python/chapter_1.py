"""Chapter_1 — 来自 Inside-Deep-Learning 仓库，自动转成可运行 .py 脚本。

原始 notebook 位于 ../Inside-Deep-Learning/。
"""

# ====== 单元 0 (代码) ======
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

# ====== 单元 1 (代码) ======
# [已剥离] %matplotlib inline
# [已剥离] from IPython.display import set_matplotlib_formats
# [已剥离] set_matplotlib_formats('png', 'pdf')

# ====== 单元 2 (代码) ======
import torch

# ====== 单元 3 (代码) ======
torch_scalar = torch.tensor(3.14)
torch_vector = torch.tensor([1, 2, 3, 4])
torch_matrix = torch.tensor([[1, 2,],
                             [3, 4,],
                             [5, 6,], 
                             [7, 8,]])
#你不必像我这样格式化，这样写只是为了清晰
torch_tensor3d = torch.tensor([
                            [
                            [ 1,  2,  3], 
                            [ 4,  5,  6],
                            ],
                            [
                            [ 7,  8,  9], 
                            [10, 11, 12],
                            ],
                            [
                            [13, 14, 15], 
                            [16, 17, 18],
                            ],
                            [
                            [19, 20, 21], 
                            [22, 23, 24],
                            ]
                              ])

# ====== 单元 4 (代码) ======
print(torch_scalar.shape)
print(torch_vector.shape)
print(torch_matrix.shape)
print(torch_tensor3d.shape)

# ====== 单元 5 (代码) ======
x_np = np.random.random((4,4))
print(x_np)

# ====== 单元 6 (代码) ======
x_pt = torch.tensor(x_np)
print(x_pt)

# ====== 单元 7 (代码) ======
print(x_np.dtype, x_pt.dtype)

# ====== 单元 8 (代码) ======
#我们强制把它们转换为 32 位浮点数
x_np = np.asarray(x_np, dtype=np.float32)
x_pt = torch.tensor(x_np, dtype=torch.float32)
print(x_np.dtype, x_pt.dtype)

# ====== 单元 9 (代码) ======
b_np = (x_np > 0.5)
print(b_np)
print(b_np.dtype)

# ====== 单元 10 (代码) ======
b_pt = (x_pt > 0.5)
print(b_pt)
print(b_pt.dtype)

# ====== 单元 11 (代码) ======
print(np.sum(x_np))

# ====== 单元 12 (代码) ======
print(torch.sum(x_pt))

# ====== 单元 13 (代码) ======
print(np.transpose(x_np))

# ====== 单元 14 (代码) ======
print(torch.transpose(x_pt, 0, 1))

# ====== 单元 15 (代码) ======
print(torch.transpose(torch_tensor3d, 0, 2).shape)

# ====== 单元 16 (代码) ======
import timeit
x = torch.rand(2**11, 2**11)
time_cpu = timeit.timeit("x@x", globals=globals(), number=100)

# ====== 单元 17 (代码) ======
print("Is CUDA available? :", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 单元 18 (代码) ======
x = x.to(device)
time_gpu = timeit.timeit("x@x", globals=globals(), number=100)

# ====== 单元 19 (代码) ======
def moveTo(obj, device):
    """
    obj: 要移动到设备上的 Python 对象，或要将其内部内容移动到设备上的对象
    device: 要将对象移动到的目标计算设备
    """
    if isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    elif hasattr(obj, "to"):
        return obj.to(device)
    else:
        return obj
    
some_tensors = [torch.tensor(1), torch.tensor(2)]
print(some_tensors)
print(moveTo(some_tensors, device))

# ====== 单元 20 (代码) ======
def f(x):
    return torch.pow((x-2.0), 2)

x_axis_vals = np.linspace(-7,9,100) 
y_axis_vals = f(torch.tensor(x_axis_vals)).numpy()

sns.lineplot(x=x_axis_vals, y=y_axis_vals, label='$f(x)=(x-2)^2$')
plt.show()

# ====== 单元 21 (代码) ======
def fP(x): #手动定义 f(x) 的导数
    return 2*x-4

y_axis_vals_p = fP(torch.tensor(x_axis_vals)).numpy()

#首先在 0 处画一条黑色基线，方便我们直观判断正负
sns.lineplot(x=x_axis_vals, y=[0.0]*len(x_axis_vals), label="0", color='black')
sns.lineplot(x=x_axis_vals, y=y_axis_vals, label='$f(x) = (x-2)^2$')
sns.lineplot(x=x_axis_vals, y=y_axis_vals_p, label="$f'(x)=2 x - 4$")
plt.show()

# ====== 单元 22 (代码) ======
x = torch.tensor([-3.5], requires_grad=True)
print(x.grad)

# ====== 单元 23 (代码) ======
value = f(x)
print(value)

# ====== 单元 24 (代码) ======
value.backward()
print(x.grad)

# ====== 单元 25 (代码) ======
x = torch.tensor([-3.5], requires_grad=True)

x_cur = x.clone()
x_prev = x_cur*100 #把初始的“前一个”解设得更大些
epsilon = 1e-5
eta = 0.1

while torch.linalg.norm(x_cur-x_prev) > epsilon:
    x_prev = x_cur.clone() #这里必须 clone，避免 x_prev 与 x_cur 指向同一个对象

    #计算函数值、梯度并更新
    value = f(x)
    value.backward()
    x.data -= eta * x.grad
    x.grad.zero_() #需要手动清零旧的梯度，PyTorch 不会自动清零

    #当前的值是多少？
    x_cur = x.data
    
print(x_cur)

# ====== 单元 26 (代码) ======
x_param = torch.nn.Parameter(torch.tensor([-3.5]), requires_grad=True)

# ====== 单元 27 (代码) ======
optimizer = torch.optim.SGD([x_param], lr=eta)

# ====== 单元 28 (代码) ======
for epoch in range(60):
    optimizer.zero_grad() #等价于 x.grad.zero_()
    loss_incurred  = f(x_param)
    loss_incurred.backward()
    optimizer.step() #等价于 x.data -= eta * x.grad
print(x_param.data)

# ====== 单元 29 (代码) ======
from torch.utils.data import Dataset
from sklearn.datasets import fetch_openml

# 从 https://www.openml.org/d/554 加载数据
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(X.shape)

# ====== 单元 30 (代码) ======
class SimpleDataset(Dataset):
        
    def __init__(self, X, y):
        super(SimpleDataset, self).__init__()
        self.X = X
        self.y = y
    
    def __getitem__(self, index):
        #这些“工作”本可以放进构造函数里，但你应该养成在 __getitem__ 里处理的习惯
        inputs = torch.tensor(self.X[index,:], dtype=torch.float32)
        targets = torch.tensor(int(self.y[index]), dtype=torch.int64)
        return inputs, targets 

    def __len__(self):
        return self.X.shape[0]
#现在我们可以创建一个 PyTorch 数据集
dataset = SimpleDataset(X, y)

# ====== 单元 31 (代码) ======
print("Length: ", len(dataset))
example, label = dataset[0]
print("Features: ", example.shape) #会返回 784
print("Label of index 0: ", label)

# ====== 单元 32 (代码) ======
plt.imshow(example.reshape((28,28)))
plt.show()

# ====== 单元 33 (代码) ======
train_size = int(len(dataset)*0.8)
test_size = len(dataset)-train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_size, test_size))
print("{} examples for training and {} for testing".format(len(train_dataset), len(test_dataset)))

