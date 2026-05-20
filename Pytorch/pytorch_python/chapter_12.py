"""Chapter_12 — 来自 Inside-Deep-Learning 仓库，自动转成可运行 .py 脚本。

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
from idlmam import AttentionAvg, GeneralScore, DotScore, AdditiveAttentionScore #用于注意力机制

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
# !conda install -c pytorch torchtext 
# !conda install -c powerai sentencepiece 
# !pip install  torchtext 
# !pip install  sentencepiece

# ====== 单元 4 (代码) ======
print(set_seed(42))

# ====== 单元 5 (代码) ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch.cuda.is_available() else torch.device("cpu")

# ====== 单元 6 (代码) ======
import torchtext
from torchtext.datasets import AG_NEWS

train_iter, test_iter = AG_NEWS(root='./data', split=('train', 'test'))
train_dataset = list(train_iter)
test_dataset = list(test_iter)

# ====== 单元 7 (代码) ======
print(train_dataset[0])

# ====== 单元 8 (代码) ======
from torchtext.data.utils import get_tokenizer#分词器将字符串（如 "this is a string"）切分成 token 列表（如 ['this', 'is', 'a', 'string']）
tokenizer = get_tokenizer('basic_english') #使用默认的英文分词器即可

from collections import Counter #统计数据集中的行数
from torchtext.vocab import Vocab #我们需要根据训练集中所有词创建一个词表

counter = Counter()
for (label, line) in train_dataset: #遍历训练数据
    counter.update(tokenizer(line)) #统计出现的不同 token 数量及频率（例如 "the" 会出现很多次，而 "sasquatch" 可能只出现一次甚至没出现）
vocab = Vocab(counter, min_freq=10, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>')) #创建词表对象，过滤掉出现次数少于 10 的词，并添加未知、句首、句尾、填充等特殊 token

# ====== 单元 9 (代码) ======
def text_transform(x): #字符串 -> 整数列表
    return [vocab['<BOS>']] + [vocab[token] for token in tokenizer(x)] + [vocab['<EOS>']] #vocab 类似字典，会自动处理未知 token，我们在前后分别拼接起始符和结束符

def label_transform(x):
    return x-1 #原始标签为 [1, 2, 3, 4]，需要转换为 [0, 1, 2, 3]

#将第一条数据的文本转换为 token 列表
print(text_transform(train_dataset[0][1]))

# ====== 单元 10 (代码) ======
VOCAB_SIZE = len(vocab)
NUM_CLASS = len(np.unique([z[0] for z in train_dataset])) 
print("Vocab: ", VOCAB_SIZE)
print("Num Classes: ", NUM_CLASS)

padding_idx = vocab["<PAD>"]

embed_dim = 128
B = 64
epochs = 15

# ====== 单元 11 (代码) ======
def pad_batch(batch):
    """
    将批中的每条数据填充到该批最长项的长度。
    同时重新排列，使返回值为 (输入, 标签)。
    """
    labels = [label_transform(z[0]) for z in batch] #获取并转换批中每条标签
    texts = [torch.tensor(text_transform(z[1]), dtype=torch.int64) for z in batch] #获取每条文本、分词并放入张量
    #当前批中最长序列的长度
    max_len = max([text.size(0) for text in texts])
    #将每个文本张量填充到 max_len
    texts = [F.pad(text, (0,max_len-text.size(0)), value=padding_idx) for text in texts]
    #将 x 和 y 各自合并为单个张量
    x, y = torch.stack(texts), torch.tensor(labels, dtype=torch.int64)

    return x, y

# ====== 单元 12 (代码) ======
train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, collate_fn=pad_batch)
test_loader = DataLoader(test_dataset, batch_size=B, collate_fn=pad_batch)

# ====== 单元 13 (代码) ======
gru = nn.Sequential(
  nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=padding_idx), #(B, T) -> (B, T, D)
  nn.GRU(embed_dim, embed_dim, num_layers=3, batch_first=True, bidirectional=True), #(B, T, D) -> ( (B,T,D) , (S, B, D)  )
  LastTimeStep(rnn_layers=3, bidirectional=True), #将 RNN 输出归约为单个张量 (B, 2*D)
  nn.Linear(embed_dim*2, NUM_CLASS), #(B, D) -> (B, 类别数)
)

loss_func = nn.CrossEntropyLoss()
gru_results = train_network(gru, loss_func, train_loader, val_loader=test_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=epochs)

# ====== 单元 14 (代码) ======
sns.lineplot(x='epoch', y='val Accuracy', data=gru_results, label='GRU')
plt.show()

# ====== 单元 15 (代码) ======
simpleEmbdAvg = nn.Sequential(
    nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=padding_idx), #(B, T) -> (B, T, D)
    nn.Linear(embed_dim, embed_dim),
    nn.LeakyReLU(),
    nn.Linear(embed_dim, embed_dim),
    nn.LeakyReLU(),
    nn.Linear(embed_dim, embed_dim),
    nn.LeakyReLU(),
    nn.AdaptiveAvgPool2d((1,embed_dim)), #(B, T, D) -> (B, 1, D)
    nn.Flatten(), #(B, 1, D) -> (B, D)
    nn.Linear(embed_dim, embed_dim),
    nn.LeakyReLU(),
    nn.BatchNorm1d(embed_dim),
    nn.Linear(embed_dim, NUM_CLASS)
)
simpleEmbdAvg_results = train_network(simpleEmbdAvg, loss_func, train_loader, val_loader=test_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=epochs)

# ====== 单元 16 (代码) ======
sns.lineplot(x='epoch', y='val Accuracy', data=gru_results, label='GRU')
sns.lineplot(x='epoch', y='val Accuracy', data=simpleEmbdAvg_results, label='Average Embedding')
plt.show()

# ====== 单元 17 (代码) ======
sns.lineplot(x='total time', y='val Accuracy', data=gru_results, label='GRU')
sns.lineplot(x='total time', y='val Accuracy', data=simpleEmbdAvg_results, label='Average Embedding')
plt.show()

# ====== 单元 18 (代码) ======
class EmbeddingAttentionBag(nn.Module):

    def __init__(self, vocab_size, D, embd_layers=3, padding_idx=None):
        super(EmbeddingAttentionBag, self).__init__()
        self.padding_idx = padding_idx
        self.embd = nn.Embedding(vocab_size, D, padding_idx=padding_idx)
        if isinstance(embd_layers, int):
            self.embd_layers =  nn.Sequential( #(B, T, D) -> (B, T, D)
                *[nn.Sequential(nn.Linear(embed_dim, embed_dim),
                nn.LeakyReLU()) for _ in range(embd_layers)]
            )
        else:
            self.embd_layers = embd_layers
        self.attn = AttentionAvg(AdditiveAttentionScore(D))# 在第 10 章中已经定义的函数

    def forward(self, input):
        """
        input: 形状为 (B, T)，dtype=int64
        output: 形状为 (B, D)，dtype=float32
        """
        if self.padding_idx is not None:
            mask = input != self.padding_idx
        else:
            mask = input == input #所有项都为 `True`
        #mask 的形状为 (B, T)
        x = self.embd(input) #(B, T, D)
        x = self.embd_layers(x)#(B, T, D)
        #沿时间维度求平均
        context = x.sum(dim=1)/(mask.sum(dim=1).unsqueeze(1)+1e-5) #(B, T, D) -> (B, D)
        #如果只想做普通平均，此处直接返回 context 即可！
        return self.attn(x, context, mask=mask) # ((B, T, D), (B, D)) -> (B, D)

# ====== 单元 19 (代码) ======
#现在可以定义一个简单的模型！
attnEmbd = nn.Sequential(
    EmbeddingAttentionBag(VOCAB_SIZE, embed_dim, padding_idx=padding_idx), #(B, T) -> (B, D)
    nn.Linear(embed_dim, embed_dim),
    nn.LeakyReLU(),
    nn.BatchNorm1d(embed_dim),
    nn.Linear(embed_dim, NUM_CLASS)
)
attnEmbd_results = train_network(attnEmbd, loss_func, train_loader, val_loader=test_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=epochs)

# ====== 单元 20 (代码) ======
sns.lineplot(x='total time', y='val Accuracy', data=gru_results, label='GRU')
sns.lineplot(x='total time', y='val Accuracy', data=simpleEmbdAvg_results, label='Average Embedding')
sns.lineplot(x='total time', y='val Accuracy', data=attnEmbd_results, label='Attention Embedding')
plt.show()

# ====== 单元 21 (代码) ======
def cnnLayer(in_size, out_size): #偷懒了，本来应该把 k_size 也作为参数
    return nn.Sequential(
        nn.Conv1d(in_size, out_size, kernel_size=k_size, padding=k_size//2),
        nn.LeakyReLU(),
        nn.BatchNorm1d(out_size))

k_size = 3
cnnOverTime = nn.Sequential(
    nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=padding_idx), #(B, T) -> (B, T, D)
    LambdaLayer(lambda x : x.permute(0,2,1)), #(B, T, D) -> (B, D, T)
    #现在我们把 D 当作通道数来重新解读数据
    cnnLayer(embed_dim, embed_dim),
    cnnLayer(embed_dim, embed_dim),
    nn.AvgPool1d(2), #(B, D, T) -> (B, D, T/2)
    cnnLayer(embed_dim, embed_dim*2),
    cnnLayer(embed_dim*2, embed_dim*2),
    nn.AvgPool1d(2), #(B, 2*D, T/2) -> (B, 2*D, T/4)
    cnnLayer(embed_dim*2, embed_dim*4),
    cnnLayer(embed_dim*4, embed_dim*4),
    #经过若干轮池化与卷积后，将其压缩为固定长度
    nn.AdaptiveMaxPool1d(1), #(B, 4*D, T/4) -> (B, 4*D, 1)
    nn.Flatten(), #(B, 4*D, 1) -> (B, 4*D)
    nn.Linear(4*embed_dim, embed_dim),
    nn.LeakyReLU(),
    nn.BatchNorm1d(embed_dim),
    nn.Linear(embed_dim, NUM_CLASS)
)
cnn_results = train_network(cnnOverTime, loss_func, train_loader, val_loader=test_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=epochs)

# ====== 单元 22 (代码) ======
sns.lineplot(x='total time', y='val Accuracy', data=gru_results, label='GRU')
sns.lineplot(x='total time', y='val Accuracy', data=simpleEmbdAvg_results, label='Average Embedding')
sns.lineplot(x='total time', y='val Accuracy', data=attnEmbd_results, label='Attention Embedding')
sns.lineplot(x='total time', y='val Accuracy', data=cnn_results, label='CNN Adaptive Pooling')
plt.show()

# ====== 单元 23 (代码) ======
position = np.arange(0, 100)
sns.lineplot(position, np.sin(position), label="sin(position)")
plt.show()

# ====== 单元 24 (代码) ======
position = np.arange(0, 100)
sns.lineplot(x=position, y=np.sin(position), label="sin(position)")
sns.lineplot(x=position, y=np.sin(position/10), label="sin(position/10)")
plt.show()

# ====== 单元 25 (代码) ======
dimensions = 6
position = np.expand_dims(np.arange(0, 100), 1)
#以数值稳定的方式计算频率 f
div = np.exp(np.arange(0, dimensions*2, 2) * (-math.log(10000.0) / (dimensions*2)))
for i in range(dimensions):
    sns.lineplot(x=position[:,0], y=np.sin(position*div)[:,i], label="Dim-"+str(i))
plt.show()

# ====== 单元 26 (代码) ======
#改编自 https://github.com/pytorch/examples/blob/0c1654d6913f77f09c0505fb284d977d89c17c1a/word_language_model/model.py#L63
class PositionalEncoding(nn.Module):
    r"""向序列中的 token 注入相对或绝对位置信息。
        位置编码与词嵌入维度相同，因而二者可以相加。这里我们使用不同频率的正弦和余弦函数。
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{其中 pos 为词位置，i 为嵌入索引}
    参数:
        d_model: 嵌入维度（必填）。
        dropout: dropout 值（默认 0.1）。
        max_len: 输入序列的最大长度（默认 5000）。
    示例:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
        self.batch_first = batch_first

    def forward(self, x):
        r"""forward 函数的输入
        参数:
            x: 输入位置编码模型的序列（必填）。
        形状:
            x: [序列长度, 批大小, 嵌入维度]
            output: [序列长度, 批大小, 嵌入维度]
        示例:
            >>> output = pos_encoder(x)
        """
        if self.batch_first: #将输入形状从 (B, T, D) 转为 (T, B, D)
            x = x.permute(1, 0, 2)

        x = x *np.sqrt(self.d_model) + self.pe[:x.size(0), :]
        x = self.dropout(x)

        if self.batch_first: #再转回 (B, T, D)
            x = x.permute(1, 0, 2)

        return x

# ====== 单元 27 (代码) ======
simplePosEmbdAvg = nn.Sequential(
    nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=padding_idx), #(B, T) -> (B, T, D)
    PositionalEncoding(embed_dim, batch_first=True),
    nn.Linear(embed_dim, embed_dim),
    nn.LeakyReLU(),
    nn.Linear(embed_dim, embed_dim),
    nn.LeakyReLU(),
    nn.Linear(embed_dim, embed_dim),
    nn.LeakyReLU(),
    nn.AdaptiveAvgPool2d((1,None)), #(B, T, D) -> (B, 1, D)
    nn.Flatten(), #(B, 1, D) -> (B, D)
    nn.Linear(embed_dim, embed_dim),
    nn.LeakyReLU(),
    nn.BatchNorm1d(embed_dim),
    nn.Linear(embed_dim, NUM_CLASS)
)

# ====== 单元 28 (代码) ======
embd_layers =  nn.Sequential( #(B, T, D) -> (B, T, D)
    *([PositionalEncoding(embed_dim, batch_first=True)]+
      [nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LeakyReLU()) for _ in range(3)])
)

attnPosEmbd = nn.Sequential(
    EmbeddingAttentionBag(VOCAB_SIZE, embed_dim, padding_idx=padding_idx, embd_layers=embd_layers), #(B, T) -> (B, D) 
    nn.Linear(embed_dim, embed_dim),
    nn.LeakyReLU(),
    nn.BatchNorm1d(embed_dim),
    nn.Linear(embed_dim, NUM_CLASS)
)

posEmbdAvg_results = train_network(simplePosEmbdAvg, loss_func, train_loader, val_loader=test_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=epochs)
attnPosEmbd_results = train_network(attnPosEmbd, loss_func, train_loader, val_loader=test_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=epochs)

# ====== 单元 29 (代码) ======
sns.lineplot(x='total time', y='val Accuracy', data=simpleEmbdAvg_results, label='Average Embedding')
sns.lineplot(x='total time', y='val Accuracy', data=posEmbdAvg_results, label='Average Positional Embedding')
sns.lineplot(x='total time', y='val Accuracy', data=attnEmbd_results, label='Attention Embedding')
sns.lineplot(x='total time', y='val Accuracy', data=attnPosEmbd_results, label='Attention Positional Embedding')
plt.show()

# ====== 单元 30 (代码) ======
sns.lineplot(x='total time', y='val Accuracy', data=gru_results, label='GRU')
sns.lineplot(x='total time', y='val Accuracy', data=attnEmbd_results, label='Attention Embedding')
sns.lineplot(x='total time', y='val Accuracy', data=attnPosEmbd_results, label='Attention Positional Embedding')
plt.show()

# ====== 单元 31 (代码) ======
class SimpleTransformerClassifier(nn.Module):

    def __init__(self, vocab_size, D, padding_idx=None):
        super(SimpleTransformerClassifier, self).__init__()
        self.padding_idx = padding_idx
        self.embd = nn.Embedding(vocab_size, D, padding_idx=padding_idx)
        self.position = PositionalEncoding(D, batch_first=True)
        #下面这一行是 transformer 实现的主体！
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=D, nhead=8),num_layers=3)
        self.attn = AttentionAvg(AdditiveAttentionScore(D))
        self.pred = nn.Sequential(
            nn.Flatten(), #(B, 1, D) -> (B, D)
            nn.Linear(D, D),
            nn.LeakyReLU(),
            nn.BatchNorm1d(D),
            nn.Linear(D, NUM_CLASS)
        )

    def forward(self, input):
        if self.padding_idx is not None:
            mask = input != self.padding_idx
        else:
            mask = input == input #所有项都为 `True`
        x = self.embd(input) #(B, T, D)
        x = self.position(x) #(B, T, D)
        #由于我们代码的结果是 (B, T, D)，而 transformer
        #需要的输入是 (T, B, D)，因此前后都需要对维度做转置
        x = self.transformer(x.permute(1,0,2)) #(T, B, D)
        x = x.permute(1,0,2) #(B, T, D)
        #沿时间维度求平均
        context = x.sum(dim=1)/mask.sum(dim=1).unsqueeze(1)
        return self.pred(self.attn(x, context, mask=mask))
#构建并训练该模型！
simpleTransformer = SimpleTransformerClassifier(VOCAB_SIZE, embed_dim, padding_idx=padding_idx)
transformer_results = train_network(simpleTransformer, loss_func, train_loader, val_loader=test_loader, score_funcs={'Accuracy': accuracy_score}, device=device, epochs=epochs)

# ====== 单元 32 (代码) ======
sns.lineplot(x='total time', y='val Accuracy', data=gru_results, label='GRU')
sns.lineplot(x='total time', y='val Accuracy', data=attnEmbd_results, label='Attention Embedding')
sns.lineplot(x='total time', y='val Accuracy', data=attnPosEmbd_results, label='Attention Positional Embedding')
sns.lineplot(x='total time', y='val Accuracy', data=cnn_results, label='CNN Adaptive Pooling')
sns.lineplot(x='total time', y='val Accuracy', data=transformer_results, label='Transformer')
plt.show()

