"""Chapter_4 — 来自 Inside-Deep-Learning 仓库，自动转成可运行 .py 脚本。

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

from idlmam import train_simple_network, Flatten, weight_reset

# ====== 单元 1 (代码) ======
# [已剥离] %matplotlib inline
# [已剥离] from IPython.display import set_matplotlib_formats
# [已剥离] set_matplotlib_formats('png', 'pdf')

# ====== 单元 2 (代码) ======
torch.backends.cudnn.deterministic=True
from idlmam import set_seed, moveTo
print(set_seed(42))

# ====== 单元 3 (代码) ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch.cuda.is_available() else torch.device("cpu")

# ====== 单元 4 (代码) ======
mnist_data_train = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_data_test = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

mnist_train_loader = DataLoader(mnist_data_train, batch_size=64, shuffle=True)
mnist_test_loader = DataLoader(mnist_data_test, batch_size=64)

#输入中有多少个数值？用它来确定后续层的尺寸
D = 28*28 #28 * 28 的图像
#隐藏层大小
n = 256
#输入有多少个通道？
C = 1
#一共有多少类？
classes = 10

#创建我们的常规模型
model_regular = nn.Sequential(
  Flatten(), 
  nn.Linear(D, n), 
  nn.Tanh(),
  nn.Linear(n, n), 
  nn.Tanh(),
  nn.Linear(n, n), 
  nn.Tanh(),
  nn.Linear(n, classes),
)

# ====== 单元 5 (代码) ======
loss_func = nn.CrossEntropyLoss()
regular_results = train_simple_network(model_regular, loss_func, mnist_train_loader, test_loader=mnist_test_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=10)

# ====== 单元 6 (代码) ======
#创建网络中我们准备共享的那一层权重
h_2 = nn.Linear(n, n)
model_shared = nn.Sequential(
  Flatten(),
  nn.Linear(D, n),
  nn.Tanh(),
  h_2, #第一次使用
  nn.Tanh(),
  h_2, #第二次使用，此时权重已经被共享
  nn.Tanh(),
  nn.Linear(n, classes),
)

# ====== 单元 7 (代码) ======
shared_results = train_simple_network(model_shared, loss_func, mnist_train_loader, test_loader=mnist_test_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=10)

# ====== 单元 8 (代码) ======
#现在我们可以绘制结果并加以对比
sns.lineplot(x='epoch', y='test Accuracy', data=regular_results, label='Normal')
sns.lineplot(x='epoch', y='test Accuracy', data=shared_results, label='Shared')
plt.show()

# ====== 单元 9 (代码) ======
zip_file_url = "https://download.pytorch.org/tutorial/data.zip"

import requests, zipfile, io
r = requests.get(zip_file_url)
z = zipfile.ZipFile(io.BytesIO(r.content))
print(z.extractall())

#压缩包按照 data/names/[LANG].txt 组织，其中 [LANG] 是某种具体的语言

# ====== 单元 10 (代码) ======
namge_language_data = {}

#我们会用一段代码去掉 UNICODE 符号，让后续处理更方便
#例如把 "Ślusàrski" 转换成 Slusarski
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
alphabet = {}
for i in range(n_letters):
    alphabet[all_letters[i]] = i

# 把一个 Unicode 字符串转成纯 ASCII，方法来自 https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

#遍历每一种语言，打开 zip 中对应的条目，从文本文件读出所有行
for zip_path in z.namelist():
    if "data/names/" in zip_path and zip_path.endswith(".txt"):
        lang = zip_path[len("data/names/"):-len(".txt")]
        with z.open(zip_path) as myfile:
            lang_names = [unicodeToAscii(line).lower() for line in str(myfile.read(), encoding='utf-8').strip().split("\n")]
            namge_language_data[lang] = lang_names
        print(lang, ": ", len(lang_names)) #顺便把每种语言的名字也打印出来

# ====== 单元 11 (代码) ======
class LanguageNameDataset(Dataset):
    
    def __init__(self, lang_name_dict, vocabulary):
        self.label_names = [x for x in lang_name_dict.keys()]
        self.data = []
        self.labels = []
        self.vocabulary = vocabulary
        for y, language in enumerate(self.label_names):
            for sample in lang_name_dict[language]:
                self.data.append(sample)
                self.labels.append(y)
        
    def __len__(self):
        return len(self.data)
    
    def string2InputVec(self, input_string):
        """
        本方法会根据当前对象使用的词表，把任意输入字符串转换为一个 long 类型的向量。
        input_string: 要转换为张量的字符串
        """
        T = len(input_string) #字符串有多长（多少个字符）？

        #新建一个张量用来存放结果
        name_vec = torch.zeros((T), dtype=torch.long)
        #遍历字符串，将对应的值填入张量
        for pos, character in enumerate(input_string):
            name_vec[pos] = self.vocabulary[character]

        return name_vec

    def __getitem__(self, idx):
        name = self.data[idx]
        label = self.labels[idx]

        #把正确的类别标签转换成 PyTorch 张量
        label_vec = torch.tensor([label], dtype=torch.long)

        return self.string2InputVec(name), label

# ====== 单元 12 (代码) ======
dataset = LanguageNameDataset(namge_language_data, alphabet)

train_data, test_data = torch.utils.data.random_split(dataset, (len(dataset)-300, 300))
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# ====== 单元 13 (代码) ======
with torch.no_grad():
    input_sequence = torch.tensor([0, 1, 1, 0, 2], dtype=torch.long)
    embd = nn.Embedding(3, 2)
    x_seq = embd(input_sequence)
    print(input_sequence.shape, x_seq.shape)
    print(x_seq)

# ====== 单元 14 (代码) ======
class LastTimeStep(nn.Module):
    """
    用于从 PyTorch RNN 模块的输出中提取最后一个时间步的隐藏激活值的类。
    """
    def __init__(self, rnn_layers=1, bidirectional=False):
        super(LastTimeStep, self).__init__()
        self.rnn_layers = rnn_layers
        if bidirectional:
            self.num_driections = 2
        else:
            self.num_driections = 1

    def forward(self, input):
        #结果要么是元组 (out, h_t)
        #要么是元组 (out, (h_t, c_t))
        rnn_output = input[0]
        last_step = input[1] #这里就是 h_t
        if(type(last_step) == tuple):#除非它是元组
            last_step = last_step[0]#这种情况下 h_t 是元组的第一个元素
        batch_size = last_step.shape[1] #根据文档，shape 为：'(num_layers * num_directions, batch, hidden_size)'
        #reshape 让各个维度分开
        last_step = last_step.view(self.rnn_layers, self.num_driections, batch_size, -1)
        #我们想要最后一层的结果
        last_step = last_step[self.rnn_layers-1]
        #重新排列，让 batch 维度排在最前面
        last_step = last_step.permute(1, 0, 2)
        #最后把最后两个维度展平成一个
        return last_step.reshape(batch_size, -1)

# ====== 单元 15 (代码) ======
D = 64
vocab_size = len(all_letters)
hidden_nodes = 256
classes = len(dataset.label_names)

first_rnn = nn.Sequential(
  nn.Embedding(vocab_size, D), #(B, T) -> (B, T, D)
  nn.RNN(D, hidden_nodes, batch_first=True), #(B, T, D) -> ( (B,T,D) , (S, B, D)  )
  #tanh 激活已经内置在 RNN 对象里了，所以这里不需要再加
  LastTimeStep(), #我们需要把 RNN 的输出归约为一个张量 (B, D)
  nn.Linear(hidden_nodes, classes), #(B, D) -> (B, classes)
)

# ====== 单元 16 (代码) ======
loss_func = nn.CrossEntropyLoss()
batch_one_train = train_simple_network(first_rnn, loss_func, train_loader, test_loader=test_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=5)

# ====== 单元 17 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=batch_one_train, label='RNN')
plt.show()

# ====== 单元 18 (代码) ======
pred_rnn = first_rnn.to("cpu").eval()
with torch.no_grad():
    preds = F.softmax(pred_rnn(dataset.string2InputVec("frank").reshape(1,-1)), dim=-1)
    for class_id in range(len(dataset.label_names)):
        print(dataset.label_names[class_id], ":", preds[0,class_id].item()*100 , "%")

# ====== 单元 19 (代码) ======
def pad_and_pack(batch):
    #1、2、3：把这一批数据的输入长度、输入和输出分别整理成单独的列表
    input_tensors = []
    labels = []
    lengths = []
    for x, y in batch:
        input_tensors.append(x)
        labels.append(y)
        lengths.append(x.shape[0]) #假设形状为 (T, *)
    #4：构造输入的 pad（填充）版本
    x_padded = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=False)
    #5：从填充版本和长度列表构造 packed（打包）版本
    x_packed = torch.nn.utils.rnn.pack_padded_sequence(x_padded, lengths, batch_first=False, enforce_sorted=False)
    #把长度列表转换成张量
    y_batched = torch.as_tensor(labels, dtype=torch.long)
    #6：返回 packed 输入及其标签组成的元组
    return x_packed, y_batched

# ====== 单元 20 (代码) ======
class EmbeddingPackable(nn.Module):
    """
    PyTorch 中的 Embedding 层不支持 PackedSequence 对象。
    这个包装类用来修复这一点：如果传入的是普通输入，就直接使用常规的
    Embedding 层；否则就在 packed sequence 上工作，并返回一个新的、
    包含对应结果的 PackedSequence。
    """
    def __init__(self, embd_layer):
        super(EmbeddingPackable, self).__init__()
        self.embd_layer = embd_layer

    def forward(self, input):
        if type(input) == torch.nn.utils.rnn.PackedSequence:
            # 我们需要先把输入解包
            sequences, lengths = torch.nn.utils.rnn.pad_packed_sequence(input.cpu(), batch_first=True)
            #做 embedding
            sequences = self.embd_layer(sequences.to(input.data.device))
            #然后再打包成新的 packed sequence
            return torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths.cpu(),
                                                           batch_first=True, enforce_sorted=False)
        else:#普通数据直接走 Embedding
            return self.embd_layer(input)

# ====== 单元 21 (代码) ======
B = 16
train_loader = DataLoader(train_data, batch_size=B, shuffle=True, collate_fn=pad_and_pack)
test_loader = DataLoader(test_data, batch_size=B, shuffle=False, collate_fn=pad_and_pack)

# ====== 单元 22 (代码) ======
rnn_packed = nn.Sequential(
  EmbeddingPackable(nn.Embedding(vocab_size, D)), #(B, T) -> (B, T, D)
  nn.RNN(D, hidden_nodes, batch_first=True), #(B, T, D) -> ( (B,T,D) , (S, B, D)  )
  LastTimeStep(), #我们需要把 RNN 的输出归约为一个张量 (B, D)
  nn.Linear(hidden_nodes, classes), #(B, D) -> (B, classes)
)

print(rnn_packed.to(device))

# ====== 单元 23 (代码) ======
packed_train = train_simple_network(rnn_packed, loss_func, train_loader, test_loader=test_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=20)

# ====== 单元 24 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=batch_one_train, label='RNN: Batch=1')
sns.lineplot(x='epoch', y='test Accuracy', data=packed_train, label='RNN:Pakced Input')
plt.show()

# ====== 单元 25 (代码) ======
sns.lineplot(x='total time', y='test Accuracy', data=batch_one_train, label='RNN: Batch=1')
sns.lineplot(x='total time', y='test Accuracy', data=packed_train, label='RNN:Pakced Input')
plt.show()

# ====== 单元 26 (代码) ======
pred_rnn = rnn_packed.to("cpu").eval()

with torch.no_grad():
    preds = F.softmax(pred_rnn(dataset.string2InputVec("frank").reshape(1,-1)), dim=-1)
    for class_id in range(len(dataset.label_names)):
        print(dataset.label_names[class_id], ":", preds[0,class_id].item()*100 , "%")

# ====== 单元 27 (代码) ======
rnn_3layer = nn.Sequential(
  EmbeddingPackable(nn.Embedding(vocab_size, D)), #(B, T) -> (B, T, D)
  nn.RNN(D, hidden_nodes, num_layers=3, batch_first=True), #(B, T, D) -> ( (B,T,D) , (S, B, D)  )
  LastTimeStep(rnn_layers=3), #我们需要把 RNN 的输出归约为一个张量 (B, D)
  nn.Linear(hidden_nodes, classes), #(B, D) -> (B, classes)
)

rnn_3layer.to(device)
rnn_3layer_results = train_simple_network(rnn_3layer, loss_func, train_loader, test_loader=test_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=20, lr=0.01)

# ====== 单元 28 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=packed_train, label='RNN: 1-Layer')
sns.lineplot(x='epoch', y='test Accuracy', data=rnn_3layer_results, label='RNN: 3-Layer')
plt.show()

# ====== 单元 29 (代码) ======
rnn_3layer_bidir = nn.Sequential(
  EmbeddingPackable(nn.Embedding(vocab_size, D)), #(B, T) -> (B, T, D)
  nn.RNN(D, hidden_nodes, num_layers=3, batch_first=True, bidirectional=True), #(B, T, D) -> ( (B,T,D) , (S, B, D)  )
  LastTimeStep(rnn_layers=3, bidirectional=True), #我们需要把 RNN 的输出归约为一个张量 (B, D)
  nn.Linear(hidden_nodes*2, classes), #(B, D) -> (B, classes)
)

rnn_3layer_bidir.to(device)
rnn_3layer_bidir_results = train_simple_network(rnn_3layer_bidir, loss_func, train_loader, test_loader=test_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=20, lr=0.01)

# ====== 单元 30 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=packed_train, label='RNN: 1-Layer')
sns.lineplot(x='epoch', y='test Accuracy', data=rnn_3layer_results, label='RNN: 3-Layer')
sns.lineplot(x='epoch', y='test Accuracy', data=rnn_3layer_bidir_results, label='RNN: 3-Layer BiDir')
plt.show()

