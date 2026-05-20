"""Chapter_13 — 来自 Inside-Deep-Learning 仓库，自动转成可运行 .py 脚本。

原始 notebook 位于 ../Inside-Deep-Learning/。
"""

# ====== 单元 0 (代码) ======
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from idlmam import set_seed

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

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
data_url_zip = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import re

#如果还没下载过该数据集，则下载它
if not os.path.isdir('./data/PetImages'):
    resp = urlopen(data_url_zip)
    zipfile = ZipFile(BytesIO(resp.read()))
    zipfile.extractall(path = './data')

#以下文件有问题，会让数据加载器出错
bad_files = [
    './data/PetImages/Dog/11702.jpg',
    "./data/PetImages/Cat/666.jpg"
]
for f in bad_files:
    if os.path.isfile(f):
        os.remove(f)

# ====== 单元 5 (代码) ======
import warnings
print(warnings.filterwarnings('ignore', '(Possibly )?corrupt EXIF data', UserWarning))

# ====== 单元 6 (代码) ======
all_images = torchvision.datasets.ImageFolder("./data/PetImages", transform=transforms.Compose(
    [
        transforms.Resize(130), #宽高中较小的一边变为 130 像素
        transforms.CenterCrop(128), #裁取中心 128 x 128 区域
        transforms.ToTensor(), #转换为 PyTorch 张量
    ]))

train_size = int(len(all_images)*0.8) #80% 用于训练
test_size = len(all_images)-train_size #剩余 20% 用于测试

train_data, test_data = torch.utils.data.random_split(all_images, (train_size, test_size)) #按指定大小创建随机划分

# ====== 单元 7 (代码) ======
B = 128
train_loader = DataLoader(train_data, batch_size=B, shuffle=True)
test_loader = DataLoader(test_data, batch_size=B)

# ====== 单元 8 (代码) ======
f, axarr = plt.subplots(2,4, figsize=(20,10)) #创建 8 张图像的网格 (2 x 4)
for i in range(2): #行
    for j in range(4): #列
        x, y = test_data[i*4+j] #从测试集中取一张图像
        axarr[i,j].imshow(x.numpy().transpose(1,2,0)) #绘制图像
        axarr[i,j].text(0.0, 0.5, str(round(y,2)), dict(size=20, color='red')) #在左上角写出标签
plt.show()

# ====== 单元 9 (代码) ======
model = torchvision.models.resnet18()
#我们准备对模型动一些"手术"
model.fc = nn.Linear(model.fc.in_features, 2)

# ====== 单元 10 (代码) ======
loss = nn.CrossEntropyLoss()
normal_results = train_network(model, loss, train_loader, epochs=10, device=device, test_loader=test_loader, score_funcs={'Accuracy': accuracy_score})

# ====== 单元 11 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=normal_results, label='Regular')
plt.show()

# ====== 单元 12 (代码) ======
model_pretrained = torchvision.models.resnet18(pretrained=True) #已经在某个数据集上预训练过的模型
#我们准备对模型动一些"手术"
model_pretrained.fc = nn.Linear(model_pretrained.fc.in_features, 2)

# ====== 单元 13 (代码) ======
filters_pretrained = model_pretrained.conv1.weight.data.cpu().numpy() #获取第一层卷积权重，转到 CPU 并转为 numpy 张量

# ====== 单元 14 (代码) ======
#归一化使最小值为 0、最大值为 1
filters_pretrained = filters_pretrained-np.min(filters_pretrained) #平移到 [0, 最大值] 区间
filters_pretrained = filters_pretrained/np.max(filters_pretrained) #缩放到 [0, 1]

# ====== 单元 15 (代码) ======
#权重形状为 (滤波器数, C, W, H)
#matplotlib 需要 (W, H, C)，因此调整通道维位置
filters_pretrained = np.moveaxis(filters_pretrained, 1, -1)

# ====== 单元 16 (代码) ======
i_max = int(round(np.sqrt(filters_pretrained.shape[0]))) #取 sqrt(数量) 形成方形图像网格
j_max = int(np.floor(filters_pretrained.shape[0]/float(i_max))) #除以行数
f, axarr = plt.subplots(i_max,j_max, figsize=(10,10)) #创建绘图网格
for i in range(i_max): #每一行
    for j in range(j_max): #每一列
        indx = i*j_max+j #滤波器索引
        axarr[i,j].imshow(filters_pretrained[indx,:]) #绘制对应滤波器
        axarr[i,j].set_axis_off() #关闭坐标轴以免杂乱
plt.show()

# ====== 单元 17 (代码) ======
def visualizeFilters(conv_filters):
    #归一化使最小值为 0、最大值为 1
    conv_filters = conv_filters-np.min(conv_filters)
    conv_filters = conv_filters/np.max(conv_filters)
    #权重形状为 (滤波器数, C, W, H)
    #matplotlib 需要 (W, H, C)，因此调整通道维位置
    conv_filters = np.moveaxis(conv_filters, 1, -1)
    
    i_max = int(round(np.sqrt(conv_filters.shape[0])))
    j_max = int(np.floor(conv_filters.shape[0]/float(i_max)))
    f, axarr = plt.subplots(i_max,j_max, figsize=(10,10))
    for i in range(i_max):
        for j in range(j_max):
            indx = i*j_max+j
            axarr[i,j].imshow(conv_filters[indx,:])
            axarr[i,j].set_axis_off()
plt.show()

# ====== 单元 18 (代码) ======
filters_catdog = model.conv1.weight.data.cpu().numpy() #本章开头训练得到的模型的卷积滤波器
print(visualizeFilters(filters_catdog))

# ====== 单元 19 (代码) ======
class NormalizeInput(nn.Module):
    def __init__(self, baseModel):
        """
        baseModel: 需要对输入做预处理的原始 ResNet 模型
        """
        super(NormalizeInput, self).__init__()
        self.baseModel = baseModel #我们想使用的模型，但需要先对它的输入做归一化
        #ImageNet 归一化使用的均值和标准差。这些"魔法"数字大家都默认使用
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), requires_grad=False) #注意 requires_grad=False，训练中不希望它们被更新
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1), requires_grad=False)

    def forward(self, input):
        #先对输入做归一化，再喂给目标模型
        input = (input-self.mean)/self.std
        return self.baseModel(input)

# ====== 单元 20 (代码) ======
model_pretrained = NormalizeInput(model_pretrained)

# ====== 单元 21 (代码) ======
warmstart_results = train_network(model_pretrained, loss, train_loader, epochs=10, device=device, test_loader=test_loader, score_funcs={'Accuracy': accuracy_score})

# ====== 单元 22 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=normal_results, label='Regular')
sns.lineplot(x='epoch', y='test Accuracy', data=warmstart_results, label='Warm')
plt.show()

# ====== 单元 23 (代码) ======
filters_catdog_finetuned = model_pretrained.baseModel.conv1.weight.data.cpu().numpy() #获取热启动后微调过的模型的滤波器
print(visualizeFilters(filters_catdog_finetuned))

# ====== 单元 24 (代码) ======
model_frozen = torchvision.models.resnet18(pretrained=True)
#首先关闭所有参数的梯度更新
for param in model_frozen.parameters():
    param.requires_grad = False
#新加的 FC 层默认 requires_grad = True
model_frozen.fc = nn.Linear(model_frozen.fc.in_features, 2)
model_frozen = NormalizeInput(model_frozen)
frozen_transfer_results = train_network(model_frozen, loss, train_loader, epochs=10, device=device, test_loader=test_loader, score_funcs={'Accuracy': accuracy_score})

# ====== 单元 25 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=normal_results, label='Regular')
sns.lineplot(x='epoch', y='test Accuracy', data=warmstart_results, label='Warm Start')
sns.lineplot(x='epoch', y='test Accuracy', data=frozen_transfer_results, label='Frozen')
plt.show()

# ====== 单元 26 (代码) ======
train_data_small, _ = torch.utils.data.random_split(train_data, (B*2,len(train_data)-B*2)) #小数据集大小 = 2 倍 batch size
train_loader_small = DataLoader(train_data_small, batch_size=B, shuffle=True) #为这个小数据集创建 loader

# ====== 单元 27 (代码) ======
#1) 从零开始训练
model = torchvision.models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)

normal_small_results = train_network(model, loss, train_loader_small, epochs=10, device=device, test_loader=test_loader, score_funcs={'Accuracy': accuracy_score})

#2) 训练热启动模型
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2) #我们准备对模型动一些"手术"
model = NormalizeInput(model)

warmstart_small_results = train_network(model, loss, train_loader_small, epochs=10, device=device, test_loader=test_loader, score_funcs={'Accuracy': accuracy_score})

#3) 训练时冻结权重
model = torchvision.models.resnet18(pretrained=True)
#首先关闭所有参数的梯度更新
for param in model.parameters():
    param.requires_grad = False
#新加的 FC 层默认 requires_grad = True
model.fc = nn.Linear(model.fc.in_features, 2)

model = NormalizeInput(model)

frozen_transfer_small_results = train_network(model, loss, train_loader, epochs=10, device=device, test_loader=test_loader, score_funcs={'Accuracy': accuracy_score})

# ====== 单元 28 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=normal_small_results, label='Regular')
sns.lineplot(x='epoch', y='test Accuracy', data=warmstart_small_results, label='Warm Start')
sns.lineplot(x='epoch', y='test Accuracy', data=frozen_transfer_small_results, label='Frozen')
plt.show()

# ====== 单元 29 (代码) ======
# 如果之前没装过，需要先安装 `torchtext` 和 `sentencepiece` 库。
# pip install torchtext
# pip install sentencepiece

# ====== 单元 30 (代码) ======
import torchtext
from torchtext.datasets import AG_NEWS

train_iter, test_iter = AG_NEWS(root='./data', split=('train', 'test'))
train_dataset_text = list(train_iter)
test_dataset_text = list(test_iter)

from torchtext.data.utils import get_tokenizer#分词器将字符串（如 "this is a string"）切分成 token 列表（如 ['this', 'is', 'a', 'string']）
tokenizer = get_tokenizer('basic_english') #使用默认的英文分词器即可

from collections import Counter #统计数据集中的行数
from torchtext.vocab import Vocab #我们需要根据训练集中所有词创建一个词表

counter = Counter()
for (label, line) in train_dataset_text: #遍历训练数据
    counter.update(tokenizer(line)) #统计出现的不同 token 数量及频率（例如 "the" 会出现很多次，而 "sasquatch" 可能只出现一次甚至没出现）
vocab = Vocab(counter, min_freq=10, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>')) #创建词表对象，过滤掉出现次数少于 10 的词，并添加未知、句首、句尾、填充等特殊 token

def text_transform(x): #字符串 -> 整数列表
    return [vocab['<BOS>']] + [vocab[token] for token in tokenizer(x)] + [vocab['<EOS>']] #vocab 类似字典，会自动处理未知 token，我们在前后分别拼接起始符和结束符

def label_transform(x):
    return x-1 #原始标签为 [1, 2, 3, 4]，需要转换为 [0, 1, 2, 3]

VOCAB_SIZE = len(vocab)
NUN_CLASS = len(np.unique([z[0] for z in train_dataset_text]))
padding_idx = VOCAB_SIZE
VOCAB_SIZE += 1

# ====== 单元 31 (代码) ======
# import torchtext
# from torchtext.datasets import text_classification

# train_dataset_text, test_dataset_text = text_classification.DATASETS['AG_NEWS'](root="./data/", ngrams=1, vocab=None)

# ====== 单元 32 (代码) ======
train_data_text_small, _ = torch.utils.data.random_split(train_dataset_text, (256,len(train_dataset_text)-256)) #切出一个非常小的数据集

# ====== 单元 33 (代码) ======
def pad_batch(batch):
    """
    将批中的每条数据填充到该批最长项的长度。
    同时重新排列，使返回值为 (输入, 标签)。
    """
    labels = [label_transform(z[0]) for z in batch]
    texts = [torch.tensor(text_transform(z[1]), dtype=torch.int64) for z in batch]
    
    max_len = max([text.size(0) for text in texts])
    
    PAD = padding_idx
    
    texts = [F.pad(text, (0,max_len-text.size(0)), value=PAD) for text in texts]
    
    x, y = torch.stack(texts), torch.tensor(labels, dtype=torch.int64)
    
    return x, y

# ====== 单元 34 (代码) ======
embed_dim = 128
gru = nn.Sequential(
  nn.Embedding(VOCAB_SIZE, embed_dim), #(B, T) -> (B, T, D)
  nn.GRU(embed_dim, embed_dim, num_layers=3, batch_first=True, bidirectional=True), #(B, T, D) -> ( (B,T,D) , (S, B, D)  )
  LastTimeStep(rnn_layers=3, bidirectional=True), #将 RNN 输出归约为单个张量 (B, 2*D)
  nn.Linear(embed_dim*2, NUN_CLASS), #(B, D) -> (B, 类别数)
)

#使用该 collate_fn 创建训练和测试 loader
train_text_loader = DataLoader(train_data_text_small, batch_size=32, shuffle=True, collate_fn=pad_batch)
test_text_loader = DataLoader(test_dataset_text, batch_size=32, collate_fn=pad_batch)
#训练基线 GRU 模型
gru_results = train_network(gru, nn.CrossEntropyLoss(), train_text_loader, test_loader=test_text_loader, device=device, epochs=10, score_funcs={'Accuracy': accuracy_score})

# ====== 单元 35 (代码) ======
# pip install transformers

# ====== 单元 36 (代码) ======
from transformers import DistilBertTokenizer, DistilBertModel #加载 DistilBert 相关类
#初始化分词器（字符串 -> 输入张量）和模型（输入张量 -> 输出张量）
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# ====== 单元 37 (代码) ======
def huggingface_batch(batch):
    """
    将批中的每条数据填充到该批最长项的长度。
    同时重新排列，使返回值为 (输入, 标签)。
    """
    labels = [label_transform(z[0]) for z in batch] #前三行与之前相同
    texts = [z[1] for z in batch] #改动：不再使用旧的 text_transform，直接取原始文本

    #新增：让 huggingface 对一批字符串进行编码
    texts = tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True)['input_ids']

    #回到旧逻辑，堆叠成张量并返回
    x, y = texts, torch.tensor(labels, dtype=torch.int64)
    return x, y
#这里也是常规操作，只是用新的 collate_fn 构造 data loader
train_text_bert_loader = DataLoader(train_data_text_small, batch_size=32, shuffle=True, collate_fn=huggingface_batch)
test_text_bert_loader = DataLoader(test_dataset_text, batch_size=32, collate_fn=huggingface_batch)

# ====== 单元 38 (代码) ======
class BertBasedClassifier(nn.Module): #用于冻结训练 BERT 模型的新类

    def __init__(self, bert_model, classes):
        """
        bert_model: 用作网络冻结首层的 BERT 分类模型
        classes: 该分类器的输出神经元数量 / 目标类别数。
        """
        super(BertBasedClassifier, self).__init__()
        self.bert_model = bert_model #BERT 会输出形状为 (B, T, D) 的张量
        #因此我们再自定义几层，将 (B, T, D) -> 形状为 (B, 类别数) 的预测
        self.attn = AttentionAvg(AdditiveAttentionScore(bert_model.config.dim)) #用注意力将形状降到 (B, D)
        self.fc1 = nn.Linear(bert_model.config.dim, bert_model.config.dim)  #做一点特征提取
        self.pred = nn.Linear(bert_model.config.dim, classes) #输出类别预测


    def forward(self, input):
        #输入形状为 (B, T)
        mask = getMaskByFill(input)
        #"with no_grad()" 起到冻结作用
        with torch.no_grad():
            #huggingface 返回的是元组，所以需要解包
            x = self.bert_model(input)[0] # (B, T, D)
        cntxt = x.sum(dim=1)/(mask.sum(dim=1).unsqueeze(1)+1e-5) #计算平均嵌入
        x = self.attn(x, cntxt, mask) #应用注意力
        x = F.relu(self.fc1(x)) #进行预测并返回
        return self.pred(x)

bertClassifier = BertBasedClassifier(bert_model, NUN_CLASS) #构建分类器！
bert_results = train_network(bertClassifier, nn.CrossEntropyLoss(), train_text_bert_loader, test_loader=test_text_bert_loader, device=device, epochs=10, score_funcs={'Accuracy': accuracy_score})

# ====== 单元 39 (代码) ======
sns.lineplot(x='epoch', y='test Accuracy', data=gru_results, label='Regular-GRU')
sns.lineplot(x='epoch', y='test Accuracy', data=bert_results, label='Frozen-BERT')
plt.show()

